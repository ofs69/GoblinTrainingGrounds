"""Attention -> conservative auto-crop rects, per shot, with an eyeball overlay.

Deploy-facing probe for the auto-crop feature (inference-only -- the training
corpus is never cropped): the released model's mask-net attention, UNMASKED
(deploy runs no banner-mask stage), is reduced to ONE static crop rect per
shot. Rects change only at detected cuts, so cropping introduces no camera
motion, pans, or transitions the source did not already have.

The rect reads the heatmap where it is sharp: heads are mixed weighted by
their per-row concentration (the ROI heads hold ~75% of their mass in ~10
cells; the diffuse heads carry a floor spread over the whole grid that would
otherwise veto every crop), each row's bbox is the smallest descending-value
cell set holding 60% of the mixed mass, and the shot's edges are a 10th/90th
percentile vote over the shot's SHARPER half of rows (a diffuse row has no
localized opinion; what it loses is measured, not ignored -- the escape
instrument reports the mass every rect leaves outside, over ALL rows).

A RECT IS CONTINUOUS -- fractions of the frame, never attention cells. The
map is sampled on the encoder's grid and that is where it is read, but the
rect it decides is not confined to it: the map is refined ``SUBCELL`` times
between cell centres before each row's box is taken, the votes are
percentiles of continuous edges, and the size and placement that come out are
frame fractions. One cell of the deploy grid is 4.2% of the frame, which used
to be the step of every edge, of every zoom (a ladder of six rungs between
the cap and the identity snap) and of the picture box that trims letterbox
bars. The attention field is smooth and its samples say where BETWEEN two
cells the mass sits; a shot's rect averages hundreds of those samples.
The safety margin is the zoom cap: the rect height never drops below ~58%
of the grid (~1.7x), however tight the vote -- eager to fire, limited in
how far it can go (decode-chain crops measured free to x2; the cap stays
inside that envelope). On top: a one-cell margin, a near-full rect snaps to
identity (no crop), and shots too short to measure take the clip-global
rect. Two rect aspects (``--aspect``): ``source`` keeps the source aspect
(square in grid cells -- the anamorphic squash the encoder was trained on
is preserved), ``square`` fits the encoder's NATIVE 1:1-pixel shape (the
clip's own probed aspect decides the cell shape -- never assumed), handing
V-JEPA undistorted pixels and cutting the squash fat on wide sources. A
crop never pads: the rect is real pixels only, stretched by the encoder's
own square resize downstream.

``--scan`` ranks cached clips by crop headroom (sampled spans). Without it,
each ``--ids`` clip gets an overlay render for the user to eyeball: original
frames, the off-crop region dimmed, the rect in green
(``<out>/<ID>_crop.mp4``), plus a per-shot rect table and the outside-mass
instrument (share of rows whose attention escapes the chosen rect) on stdout.

CPU-only (the forward runs from the latent cache; safe next to a GPU job).
The video pixels are rendered for the USER to watch.

Usage:
    python autocrop.py --ids <clip> --dur-s 90
    python autocrop.py --scan
"""
import argparse
import bisect
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import common
from . import perception as P
from .jepa_train import FEAT_DIR_DEFAULT, JepaClip, load_model

FLOOR_Q = 75.0       # per-row background floor subtracted from the mixed map
                     # before anything reads it (0 = off). The map is an ROI
                     # sitting on a near-uniform background that is measured
                     # CONTENT-INDEPENDENT: the border ring holds 18% of the
                     # mass and the four corner blocks 5.5%, and neither falls
                     # in rows whose ROI is centred and sharp (5.6% there).
                     # Since the row bbox is the bounding box of a cell SET,
                     # one such speck pins an edge to the frame -- it does so
                     # in 63% of rows. Subtracting the FLOOR_Q-th percentile
                     # cell removes specks by their mass rather than by their
                     # position, so a genuinely bright cell at the frame edge
                     # survives where ring-zeroing would have dropped it.
TOP_MASS_Q = 0.6     # per-row bbox: mixed-attention mass the cell set holds
CONC_CELLS = 10      # sharpness = mass in this many top cells (head weight
                     # in the mix, and the row's vote weight)
EDGE_Q = 10.0        # shot edge percentile over the sharper half of rows
MARGIN_CELLS = 1.0   # extra cells each side of the shot bbox -- an attention
                     # CELL, turned into a frame fraction by the grid it was
                     # read on, so the margin stays the piece of picture it
                     # has always been
SUBCELL = 4          # sub-cell refinement of the attention map before a row's
                     # box is taken: the map is interpolated onto this many
                     # samples per cell per axis, so an edge lands on
                     # 1/(4*grid) of the frame instead of 1/grid. The field is
                     # smooth and sampled at cell centres, so where a bright
                     # cell sits beside a dim one the crossing is between
                     # them, and this is what reads it
MIN_SIDE_FRAC = 0.6667  # rect height floor as a fraction of the grid: the
                     # zoom cap (x1.5 -- 16 cells of 24) is the safety
                     # margin, and the bbox may vote tighter without the
                     # rect ever following it below this. Decode-chain
                     # crops are measured free to x2 (crop must ride the
                     # extraction decode, never a re-encode: container
                     # start-time offsets shift the clock and charge kappa
                     # for it), so the cap is not a perception limit but a
                     # HEADROOM one -- set by eye on a clip where the
                     # uncapped vote took x1.71 and cut too close
IDENT_FRAC = 0.88    # rect side >= this fraction of the grid -> no crop
PIC_DEAD_LUMA = 26   # a cell whose brightest pixel never exceeds this across
                     # the sampled frames is DEAD -- a letterbox/pillarbox
                     # bar, not picture. Bars decode to a few counts of codec
                     # noise; content that never clears this in any sample is
                     # indistinguishable from a bar, and trimming it is the
                     # right call either way. Vote edges, rect placement and
                     # the deploy side's search candidates all clamp into the
                     # picture box the dead edges leave.
PIC_SAMPLES = 9      # frames sampled for the picture box (deploy reads its
                     # probe windows instead; the box is static either way)
MIN_SHOT_S = 1.0     # shots shorter than this take the clip-global rect
EXTEND_CAP_S = 120.0 # viz: max forward extension to reach shot boundaries
OUT_THR = 0.2        # instrument: a row "escapes" above this outside mass
                     # (the mixed map keeps a diffuse residual, so small
                     # outside shares are floor, not a wandering ROI)
DIM = 0.35           # off-crop brightness in the overlay
RECT_RGB = (80, 255, 120)
MAX_DUR_S = 300.0    # eyeball renders stay short
SCAN_SPANS = 8       # sampled spans per clip in --scan
SCAN_SPAN_ROWS = 256 # rows per sampled span


def px_aspect(dataset, vid_id):
    """DISPLAYED aspect (w/h) of a clip's video -- per clip, never assumed
    (the corpus is mostly but not only 16:9), honoring anamorphic SAR.

    Reads the FIRST row only: a video tied to a timecode track by a track
    reference is reported twice by an ffprobe new enough to read stream
    groups, and both rows describe the same picture."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,sample_aspect_ratio", "-of", "csv=p=0",
         str(P.video_path(dataset, vid_id))],
        capture_output=True, text=True, check=True)
    parts = r.stdout.strip().splitlines()[0].split(",")
    w, h = int(parts[0]), int(parts[1])
    sar = 1.0
    if len(parts) > 2 and ":" in parts[2]:
        num, den = parts[2].split(":")
        if int(num) > 0 and int(den) > 0:
            sar = int(num) / int(den)
    return w * sar / h


def picture_box(dataset, vid_id, dur_s, grid):
    """The PICTURE box (x0, x1, y0, y1) as FRACTIONS of the frame: the frame
    minus its dead edge lines (letterbox/pillarbox bars), read from frames
    sampled at the encoder's own squashed square. Per pixel LINE, not per
    cell, so a bar is trimmed where it actually ends rather than at the
    nearest cell. A picture too small for the zoom cap's smallest rect (or an
    all-dead sample set) hands back the whole frame."""
    res = grid * 16
    row_max = np.zeros(res, dtype=np.uint8)
    col_max = np.zeros(res, dtype=np.uint8)
    for t in np.linspace(0.02, 0.98, PIC_SAMPLES) * dur_s:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}",
             "-i", str(P.video_path(dataset, vid_id)), "-frames:v", "1",
             "-vf", f"scale={res}:{res}", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
        if len(r.stdout) < res * res * 3:
            continue
        f = np.frombuffer(r.stdout[:res * res * 3], np.uint8) \
            .reshape(res, res, 3).max(axis=2)
        row_max = np.maximum(row_max, f.max(axis=1))
        col_max = np.maximum(col_max, f.max(axis=0))
    return picture_span(row_max, col_max, res)


def picture_span(row_max, col_max, res):
    """The picture box from the two per-line maxima (the pixel evidence)."""
    def span(v):
        live = np.flatnonzero(v > PIC_DEAD_LUMA)
        return (0, 0) if len(live) == 0 else (int(live[0]), int(live[-1]) + 1)
    y0, y1 = span(row_max)
    x0, x1 = span(col_max)
    x0, x1, y0, y1 = x0 / res, x1 / res, y0 / res, y1 / res
    if x1 - x0 < MIN_SIDE_FRAC or y1 - y0 < MIN_SIDE_FRAC:
        return (0.0, 1.0, 0.0, 1.0)
    return (x0, x1, y0, y1)


def clamp_into(start, side, p0, p1):
    """Rect start so [start, start+side) sits inside the picture span
    [p0, p1) -- centered overhang when the side outgrows the span -- and
    always inside the frame."""
    hi = p1 - side
    v = (p0 + p1 - side) / 2.0 if hi < p0 else float(np.clip(start, p0, hi))
    return float(np.clip(v, 0.0, max(1.0 - side, 0.0)))


def gate_span(model, ds, f0, f1, max_frames=1024):
    """Forward rows [f0,f1) -> mixed attention maps (n, gh, gw), sum 1.

    Chunking restarts at every shot boundary (cut-aware, same as training).
    Heads are normalized per row, then mixed weighted by their concentration
    (mass in their CONC_CELLS top cells), so the sharp ROI heads decide the
    map and the diffuse heads' whole-grid floor fades instead of dragging
    every bbox to the frame edge.
    """
    n = f1 - f0
    gh, gw = ds.feats.shape[-2:]
    att = np.zeros((n, gh, gw), dtype=np.float32)
    with torch.no_grad():
        for lo, hi in zip(ds.shot_edges[:-1], ds.shot_edges[1:]):
            lo, hi = max(lo, f0), min(hi, f1)
            for s in range(lo, hi, max_frames):
                e = min(s + max_frames, hi)
                if e - s < 2:
                    continue
                dev = next(model.parameters()).device
                x = ds.feats[s:e].to(dev).float()[None]
                if ds.feat_scale != 1.0:
                    x = x * ds.feat_scale
                _v, _conf, gate = model(x, cut=ds.cut[s:e][None].to(dev))
                g = gate[0].cpu().numpy()                  # (rows,H,gh,gw)
                g = g / (g.sum(axis=(2, 3), keepdims=True) + 1e-9)
                conc = np.sort(g.reshape(len(g), g.shape[1], -1),
                               axis=2)[:, :, -CONC_CELLS:].sum(axis=2)
                m = (g * conc[:, :, None, None]).sum(axis=1)
                if FLOOR_Q > 0:
                    thr = np.percentile(m.reshape(len(m), -1), FLOOR_Q, axis=1)
                    m = np.clip(m - thr[:, None, None], 0.0, None)
                att[s - f0:e - f0] = \
                    m / (m.sum(axis=(1, 2), keepdims=True) + 1e-9)
    return att


def refine(att):
    """Bilinear refinement of mixed maps (n,g,g) -> (n,g*SUBCELL,g*SUBCELL),
    each row renormalized to sum 1.

    Samples sit between CELL CENTRES and are held flat outside the outermost
    ones. `autocrop.rs`'s twin -- one interpolation, or the two languages read
    different edges off the same attention."""
    n, g, _ = att.shape
    sub = g * SUBCELL
    f = np.clip((np.arange(sub) + 0.5) / SUBCELL - 0.5, 0.0, g - 1.0)
    i0 = np.floor(f).astype(np.int64)
    i1 = np.minimum(i0 + 1, g - 1)
    t = (f - i0).astype(np.float32)
    rows = (att[:, i0, :] * (1.0 - t)[None, :, None]
            + att[:, i1, :] * t[None, :, None])
    out = (rows[:, :, i0] * (1.0 - t)[None, None, :]
           + rows[:, :, i1] * t[None, None, :])
    tot = out.sum(axis=(1, 2), keepdims=True)
    return out / np.where(tot > 0, tot, 1.0)


def row_boxes(att):
    """Mixed maps (n,gh,gw) -> per-row tight bbox [x0,x1,y0,y1) as FRACTIONS
    of the frame, around the smallest descending-value cell set of the
    REFINED map holding TOP_MASS_Q of the mass, plus each row's concentration
    on the sampled cells (its vote weight in the shot).

    The refinement is what takes the box off the cell lattice: the set is the
    definition it always was -- values in descending order until the mass is
    held -- taken on a map interpolated between cell centres, so an edge lands
    where the field crosses the threshold instead of on the nearest centre."""
    n = len(att)
    conc = np.sort(att.reshape(n, -1), axis=1)[:, -CONC_CELLS:].sum(axis=1)
    fine = refine(att)
    sub = fine.shape[-1]
    fl = fine.reshape(n, -1)
    srt = np.sort(fl, axis=1)[:, ::-1]
    cum = np.cumsum(srt, axis=1)
    k = np.minimum((cum < TOP_MASS_Q).sum(axis=1), sub * sub - 1)
    thr = srt[np.arange(n), k]
    boxes = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        yy, xx = np.where(fine[i] >= thr[i])
        if len(xx) == 0:                       # no mass anywhere: no opinion
            boxes[i] = (0.0, 1.0, 0.0, 1.0)
            continue
        boxes[i] = (xx.min() / sub, (xx.max() + 1) / sub,
                    yy.min() / sub, (yy.max() + 1) / sub)
    return boxes, conc


def shot_rect(att, grid, aspect="source", px_ar=None, pic=None):
    """Attention rows of one shot -> (x, y, w, h) as fractions of the frame,
    identity (0, 0, 1, 1) when the bbox covers most of the frame anyway.
    Edges are voted by the sharper half of the rows, then clamped into the
    picture box (attention on dead bars is floor noise, and the clamp is
    also what keeps a diffuse vote from reaching the identity snap through
    the bars); diffuse rows are measured by the escape instrument instead
    of voting.

    ``aspect="square"`` fits the smallest NATIVE-1:1-pixel rect (1/px_ar
    wide-to-tall in grid cells -- the un-squashed shape the encoder
    actually processes) containing the vote, with the same height floor; a
    vote too tall for any in-frame 1:1 rect falls back to the source-aspect
    square. On a wide source a 1:1 rect always cuts squash fat, so it
    never snaps to identity."""
    px0, px1, py0, py1 = pic or (0.0, 1.0, 0.0, 1.0)
    boxes, conc = row_boxes(att)
    b = boxes[conc >= np.median(conc)]
    m = MARGIN_CELLS / grid
    x0 = max(float(np.percentile(b[:, 0], EDGE_Q)) - m, px0)
    x1 = min(float(np.percentile(b[:, 1], 100 - EDGE_Q)) + m, px1)
    y0 = max(float(np.percentile(b[:, 2], EDGE_Q)) - m, py0)
    y1 = min(float(np.percentile(b[:, 3], 100 - EDGE_Q)) + m, py1)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if aspect == "square":
        if not px_ar:
            raise ValueError("square rects need the clip's px aspect")
        ny = max(y1 - y0, MIN_SIDE_FRAC, (x1 - x0) * px_ar)
        nx = ny / px_ar
        if ny <= 1.0 and nx <= 1.0:
            rx = clamp_into(cx - nx / 2.0, nx, px0, px1)
            ry = clamp_into(cy - ny / 2.0, ny, py0, py1)
            return (rx, ry, nx, ny)
    # one side for both axes: a fraction of the width is the same fraction of
    # the height in grid cells, which is what keeps the crop the shape of the
    # source and the squash the encoder was trained on
    side = min(max(x1 - x0, y1 - y0, MIN_SIDE_FRAC), 1.0)
    if side >= IDENT_FRAC:
        return (0.0, 0.0, 1.0, 1.0)
    rx = clamp_into(cx - side / 2.0, side, px0, px1)
    ry = clamp_into(cy - side / 2.0, side, py0, py1)
    return (rx, ry, side, side)


def outside_mass(att, rect):
    """Per-row attention mass the rect leaves out (the instrument).

    Cells are boxes and the rect is continuous, so an edge cell counts by the
    area of it the rect covers -- the reading the crop itself makes."""
    x, y, w, h = rect
    gh, gw = att.shape[-2:]

    def cover(n, lo, hi):
        e = np.arange(n + 1) / n
        return np.clip(np.minimum(e[1:], hi) - np.maximum(e[:-1], lo), 0, None) * n

    wx, wy = cover(gw, x, x + w), cover(gh, y, y + h)
    return 1.0 - (att * wy[None, :, None] * wx[None, None, :]).sum(axis=(1, 2))


def rects_for_span(att, edges_rel, grid, min_rows, aspect="source",
                   px_ar=None, pic=None):
    """Per-shot rects over one forwarded span + the escape instrument."""
    glob = shot_rect(att, grid, aspect, px_ar, pic)
    rects = []
    esc = np.zeros(len(att), dtype=np.float32)
    for lo, hi in zip(edges_rel[:-1], edges_rel[1:]):
        if hi - lo < min_rows:
            rects.append(glob)
        else:
            rects.append(shot_rect(att[lo:hi], grid, aspect, px_ar, pic))
        if hi > lo:
            esc[lo:hi] = outside_mass(att[lo:hi], rects[-1])
    return rects, glob, esc


def load_clip(args, ck, vid_id):
    return JepaClip(args.dataset, vid_id, 384, 192,
                    feat_dir=ck.get("feat_dir", FEAT_DIR_DEFAULT),
                    masks_dir=None,     # deploy runs unmasked
                    row_hz=ck.get("row_hz"))


def scan(args, model, ck):
    """Rank cached clips by crop headroom from sampled spans."""
    feat_dir = Path(args.dataset) / ck.get("feat_dir", FEAT_DIR_DEFAULT)
    ids = args.ids or sorted(p.stem for p in feat_dir.glob("*.npz"))
    if args.sample and args.sample < len(ids):
        ids = [ids[i] for i in
               np.linspace(0, len(ids) - 1, args.sample).astype(int)]
    print(f"scanning {len(ids)} clips from {feat_dir.name}", flush=True)
    rows = []
    for k, vid_id in enumerate(ids):
        try:
            ds = load_clip(args, ck, vid_id)
            ar = px_aspect(args.dataset, vid_id) \
                if args.aspect == "square" else None
            T = len(ds.times_ms)
            min_rows = int(MIN_SHOT_S * ds.row_hz)
            pic = picture_box(args.dataset, vid_id,
                              float(ds.times_ms[-1]) / 1000.0,
                              ds.feats.shape[-1])
            span = min(SCAN_SPAN_ROWS, T)
            starts = sorted({int(s) for s in np.linspace(
                0, T - span, min(SCAN_SPANS, max(1, T // span)))})
            glob_att, zooms, escs = [], [], []
            for s in starts:
                att = gate_span(model, ds, s, s + span)
                glob_att.append(att)
                inner = [c - s for c in ds.shot_edges if s < c < s + span]
                for lo, hi in zip([0] + inner, inner + [span]):
                    # only shots fully inside the span rank fairly
                    if (lo == 0 and s not in ds.shot_edges) or hi == span:
                        continue
                    if hi - lo < min_rows:
                        continue
                    r = shot_rect(att[lo:hi], att.shape[-1], args.aspect,
                                  ar, pic)
                    zooms.append(1.0 / r[3])
                    escs.append(outside_mass(att[lo:hi], r))
            grid = glob_att[0].shape[-1]
            gz = 1.0 / shot_rect(np.concatenate(glob_att), grid,
                                 args.aspect, ar, pic)[3]
            med = float(np.median(zooms)) if zooms else 1.0
            mx = float(np.max(zooms)) if zooms else 1.0
            crop_share = float(np.mean([z > 1.0 for z in zooms])) \
                if zooms else 0.0
            esc_p95 = float(np.percentile(np.concatenate(escs), 95)) \
                if escs else 0.0
            rows.append((vid_id, gz, med, mx, crop_share, esc_p95,
                         len(zooms)))
            print(f"[{k + 1}/{len(ids)}] {vid_id}  glob x{gz:.2f}  "
                  f"shots {len(zooms)}: med x{med:.2f} max x{mx:.2f}  "
                  f"cropped {crop_share:.0%}  esc p95 {esc_p95:.1%}",
                  flush=True)
            del ds
        except Exception as e:
            print(f"[{k + 1}/{len(ids)}] {vid_id}  ERROR {e}", flush=True)
    rows.sort(key=lambda r: r[2], reverse=True)
    print("\ntop crop headroom (median shot zoom):", flush=True)
    for vid_id, gz, med, mx, crop_share, esc_p95, n in rows[:15]:
        print(f"  {vid_id}  med x{med:.2f}  max x{mx:.2f}  glob x{gz:.2f}  "
              f"cropped {crop_share:.0%}  esc p95 {esc_p95:.1%}  "
              f"({n} shots)", flush=True)


def viz(args, model, ck):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = P.DECODE_W, P.DECODE_H
    for vid_id in args.ids:
        ds = load_clip(args, ck, vid_id)
        T = len(ds.times_ms)
        min_rows = int(MIN_SHOT_S * ds.row_hz)
        start_s = 0.25 * float(ds.times_ms[-1]) / 1000.0 \
            if args.start_s is None else args.start_s
        raw0 = min(int(np.searchsorted(ds.times_ms, start_s * 1000.0)), T - 2)
        start_s = float(ds.times_ms[raw0]) / 1000.0
        dur = min(args.dur_s, MAX_DUR_S,
                  (float(ds.times_ms[-1]) - ds.times_ms[raw0]) / 1000.0)
        raw1 = max(int(np.searchsorted(ds.times_ms,
                                       (start_s + dur) * 1000.0)), raw0 + 2)
        # extend the forward to whole shots so partial shots do not get a
        # rect from a sliver of their rows (capped -- one giant shot keeps
        # the raw span and its rect is honestly partial)
        cap = int(EXTEND_CAP_S * ds.row_hz)
        i = bisect.bisect_right(ds.shot_edges, raw0) - 1
        f0 = ds.shot_edges[i] if raw0 - ds.shot_edges[i] <= cap else raw0
        j = bisect.bisect_left(ds.shot_edges, raw1)
        f1 = ds.shot_edges[j] if (j < len(ds.shot_edges) and
                                  ds.shot_edges[j] - raw1 <= cap) else raw1
        print(f"[{vid_id}] rendering {dur / 60:.1f} min from "
              f"{start_s / 60:.1f} min (forward rows {f0}..{f1})", flush=True)

        att = gate_span(model, ds, f0, f1, args.max_shot_frames)
        grid = att.shape[-1]
        edges_rel = [0] + [c - f0 for c in ds.shot_edges if f0 < c < f1] \
            + [f1 - f0]
        ar = px_aspect(args.dataset, vid_id) \
            if args.aspect == "square" else None
        pic = picture_box(args.dataset, vid_id,
                          float(ds.times_ms[-1]) / 1000.0, grid)
        if pic != (0.0, 1.0, 0.0, 1.0):
            print(f"[{vid_id}] picture box x {pic[0]:.3f}..{pic[1]:.3f} "
                  f"y {pic[2]:.3f}..{pic[3]:.3f} of the frame "
                  f"(dead bars trimmed)", flush=True)
        rects, glob, esc = rects_for_span(att, edges_rel, grid, min_rows,
                                          args.aspect, ar, pic)

        print(f"[{vid_id}] global rect {glob[2]:.3f}x{glob[3]:.3f} of the "
              f"frame; shots in render window:", flush=True)
        for (lo, hi), r in zip(zip(edges_rel[:-1], edges_rel[1:]), rects):
            if hi + f0 <= raw0 or lo + f0 >= raw1:
                continue
            t0 = float(ds.times_ms[lo + f0]) / 60000.0
            t1 = float(ds.times_ms[min(hi + f0, T - 1)]) / 60000.0
            tag = "identity" if min(r[2], r[3]) >= 1.0 - 1e-6 else \
                f"({r[0]:.3f},{r[1]:.3f}) {r[2]:.3f}x{r[3]:.3f} of frame " \
                f"(x{1 / r[2]:.2f},x{1 / r[3]:.2f})"
            src = " [clip rect]" if hi - lo < min_rows else ""
            print(f"  {t0:7.2f}-{t1:7.2f} min  {tag}{src}", flush=True)
        print(f"[{vid_id}] escape: p95 {np.percentile(esc, 95):.1%}, "
              f"rows >{OUT_THR:.0%} outside: {np.mean(esc > OUT_THR):.1%}",
              flush=True)

        out_path = out_dir / f"{vid_id}_crop.mp4"
        enc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(int(P.FPS)), "-i", "-", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             str(out_path)], stdin=subprocess.PIPE)
        t0 = time.time()
        n_out = 0
        for t_ms, frame in P.decode_frames(P.video_path(args.dataset, vid_id),
                                           start_s=start_s, max_s=dur):
            rel = int(np.clip(np.searchsorted(ds.times_ms, t_ms) - f0,
                              0, f1 - f0 - 1))
            x0, y0, rw, rh = rects[bisect.bisect_right(edges_rel, rel) - 1]
            img = frame.astype(np.float32)
            if min(rw, rh) < 1.0 - 1e-6:
                px0, px1 = int(x0 * w), int(round((x0 + rw) * w))
                py0, py1 = int(y0 * h), int(round((y0 + rh) * h))
                keep = img[py0:py1, px0:px1].copy()
                img *= DIM
                img[py0:py1, px0:px1] = keep
                img[py0:py0 + 2, px0:px1] = RECT_RGB
                img[max(py1 - 2, 0):py1, px0:px1] = RECT_RGB
                img[py0:py1, px0:px0 + 2] = RECT_RGB
                img[py0:py1, max(px1 - 2, 0):px1] = RECT_RGB
            enc.stdin.write(img.clip(0, 255).astype(np.uint8).tobytes())
            n_out += 1
            if n_out % 1800 == 0:
                print(f"[{vid_id}] {n_out / P.FPS / 60:.0f}/{dur / 60:.0f} "
                      f"min ({n_out / (time.time() - t0):.0f} f/s)",
                      flush=True)
        enc.stdin.close()
        enc.wait()
        print(f"[{vid_id}] wrote {out_path} ({n_out / P.FPS:.0f}s of video, "
              f"{time.time() - t0:.0f}s)", flush=True)
        del ds


def main():
    global TOP_MASS_Q, EDGE_Q, MIN_SIDE_FRAC, FLOOR_Q
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", nargs="+", default=[])
    ap.add_argument("--dataset", default="dataset_v2")
    ap.add_argument("--ckpt", default=common.DEFAULT_CKPT)
    ap.add_argument("--out", default="infer_out")
    ap.add_argument("--scan", action="store_true",
                    help="rank cached clips by crop headroom (no render)")
    ap.add_argument("--sample", type=int, default=0,
                    help="scan: evenly strided subset of this many clips")
    ap.add_argument("--aspect", choices=["source", "square"],
                    default="source",
                    help="rect aspect: source-preserving square-in-cells, "
                         "or the encoder's native 1:1-pixel shape")
    ap.add_argument("--floor-q", type=float, default=FLOOR_Q,
                    help="percentile of the background floor subtracted from "
                         "each row's mixed map before the box is built "
                         "(0 = off; 75 removes the content-independent "
                         "corner/border specks that pin boxes to the frame)")
    ap.add_argument("--top-mass-q", type=float, default=TOP_MASS_Q,
                    help="per-row bbox: mixed-attention mass the cell set "
                         "holds")
    ap.add_argument("--edge-q", type=float, default=EDGE_Q,
                    help="shot edge percentile over the sharper half of rows")
    ap.add_argument("--min-side-frac", type=float, default=MIN_SIDE_FRAC,
                    help="rect height floor as a fraction of the grid "
                         "(the zoom cap)")
    ap.add_argument("--start-s", type=float, default=None,
                    help="render start (default: 25%% into the clip)")
    ap.add_argument("--dur-s", type=float, default=90.0,
                    help=f"seconds to render (capped at {MAX_DUR_S:.0f})")
    ap.add_argument("--max-shot-frames", type=int, default=1024)
    args = ap.parse_args()
    if not args.scan and not args.ids:
        ap.error("--ids is required without --scan")
    TOP_MASS_Q = args.top_mass_q
    EDGE_Q = args.edge_q
    MIN_SIDE_FRAC = args.min_side_frac
    FLOOR_Q = args.floor_q

    model, ck = load_model(args.ckpt, "cpu")
    print(f"checkpoint {args.ckpt} (epoch {ck['epoch']}, "
          f"{model.heads} heads)", flush=True)
    if args.scan:
        scan(args, model, ck)
    else:
        viz(args, model, ck)


if __name__ == "__main__":
    main()
