"""Does the bundle reproduce the production pipeline? Measured, not assumed.

The export script's own checks run on random noise, which is not a distribution
any of these nets ever sees -- they catch structural breakage and nothing else.
This is the real gate: take a clip the corpus already holds, run the EXPORTED
graphs over its real frames, and compare against the artifacts the Python
pipeline actually produced for it.

Three stages, each against ground truth on disk (the third is opt-in):

  latents  -- decode the clip's frames exactly as extract.py does (the
              manifest's decode grid, 384x384, groups of 32k frames, all 2k
              tubelet alignments), run vjepa_pca.onnx, quantize at 8/127, and compare
              to the int8 cache row for row. The int8 step IS the tolerance:
              a difference the quantizer rounds away cannot reach the model.
              The cache's row_hz stamp is asserted against the manifest --
              the check that catches a bundle whose row clock is mis-stamped
              (every row of a drafted funscript would land at the wrong ms).

  tracks   -- run head.onnx + env_step.onnx over the CACHED latents and compare
              to what jepa_infer's own forward produces (velocity, level, band,
              envelope). This isolates the head from the encoder: if the
              latents drift, the tracks still tell you whether the head graph
              is faithful. The chunking around the graphs is this file's own.
              The head graphs run on the CPU provider -- deploy's own head
              backend (heads.rs owns no GPU session), and the one DML cannot
              run: its Einsum kernel segfaults on the pooling equation.

  rust     -- `--rust` hands the SHIPPED BINARY the same cached rows
              (`--from-latents`) and compares what it writes back. That is
              the one stage where heads.rs's chunk boundaries, ctx padding,
              short-tail lookback and stepped envelope buffer are the ones
              under test, rather than a Python re-implementation of them --
              and it gets there without the encoder's int8 divergence in the
              way, so a discrepancy is attributable instead of inferable.

``--ep`` picks the backend every graph runs on and defaults to DirectML: what
a user actually runs, and where fp16 range and fused-kernel numerics show up.
The encoder graph exports with packed-QKV attention, which ORT implements on
the vendor backends but NOT on its CPU provider, so the latent stage cannot
run on ``--ep CPUExecutionProvider`` at all -- that provider is the fp32
no-vendor-kernel reference, a deliberate numerics comparison rather than a
fallback, and a silent fall back to it is refused rather than reported as the
requested backend's own numbers.

    python parity.py --dataset projects/mine --id <clip> --windows 2 \
        --ep DmlExecutionProvider

What the shipped (packed-attention, fp16) bundle reads on DirectML: latents
corr 0.999630, 82.4% of int8 values bit-identical, 0.27% off by more than one
quantization step; tracks corr 1.00000000 on all seven (vmarg, level, both
rails, env, both rev heads), autoregressive envelope included.

Every stage runs on real data on purpose. Uniform noise reports this head as
broken (``max|d|`` ~ 4.5) when it is exact: uniform int8 is an 8-sigma input
that saturates the mask net, and no encoder ever emits it.

This covers the model. It does NOT cover the styling stage that turns
tracks into actions -- grid_check's behavior fixtures and the eval read
cover that.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # the repository
GS = ROOT / "goblinscript"                       # the Rust product's own tree

from . import common                                    # noqa: E402
from . import perception as P                           # noqa: E402
from .jepa_train import JepaClip, load_model      # noqa: E402
from .forward import predict_shots                # noqa: E402

CLIP_LEN = 32
INT8_SCALE = 8.0 / 127.0

# Bars for the --rust stage. A faithfully tiled decode leaves seam rows no
# worse than interior ones; measured at 0.07-1.08x across the nine tracks,
# so 2.0 is clear of the noise a dozen seam rows carry without being
# permissive. CORR_TOL is the floor a backend difference must stay above --
# fp16 against fp32 measures 0.99997, and anything that drops a track below
# 0.9999 is structural rather than numeric.
SEAM_TOL = 2.0
CORR_TOL = 0.9999
# ... and the ratio is only EVIDENCE where the seam error is large enough to
# mean something. Both means here are backend noise at 1e-6 pos on a 0..100
# scale, so their quotient moves when the INTERIOR gets quieter and nothing
# at the seam has changed at all: one bundle read `bhi` seam 8.583e-06 twice
# over, at 1.68x against a noisier interior and 2.48x against a quieter one.
# A real tiling fault does not look like that -- a mis-keyed base draw or a
# wrong ctx length puts the seam rows a whole signal apart, orders above
# this floor -- so flooring the test costs it no power against the faults it
# exists for. 1e-3 is a thousandth of the written list's own quantum (an
# action's position is an INTEGER) and a hundredth of the smallest gap any
# probability threshold reads.
SEAM_FLOOR = 1e-3


def corr(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def latent_parity(args, sess, man):
    """Exported encoder over real frames vs the production int8 cache."""
    res, grid = man["enc_res"], man["grid"]
    k = int(man.get("tubelet_stride", 1))
    n_align = int(man.get("alignments", 2))
    man_hz = float(man.get("row_hz") or man["grid_fps"] * n_align / (2 * k))
    cache = Path(args.dataset) / common.LATENTS_DIR / f"{args.id}.npz"
    if not cache.exists():
        raise SystemExit(f"no int8 cache for {args.id} ({cache}) -- this test "
                         "compares against the production artifact, so the "
                         "clip has to be one the corpus already extracted")
    feats = common.npz_memmap(cache, "feats")
    with np.load(cache) as z:
        scale = float(z["scale"])
        cache_hz = float(z["row_hz"]) if "row_hz" in z.files else None
    if cache_hz is not None and abs(cache_hz - man_hz) > 0.01 * man_hz:
        raise SystemExit(
            f"manifest row clock {man_hz:g} rows/s != cache stamp "
            f"{cache_hz:g} -- the bundle would write every action at the "
            f"wrong millisecond")
    if n_align != 2 * k:
        raise SystemExit(
            f"latent parity compares cache rows BY FRAME INDEX, which holds "
            f"only while the alignments interleave back to one row per frame "
            f"(n_align = 2k). This bundle is k={k}/alignments={n_align}, a "
            f"THINNED grid whose rows are not 1:1 with frames -- map rows to "
            f"frames before trusting it here")
    print(f"cache {feats.shape} int8, scale {scale:.6f}", flush=True)

    # Exactly extract.py's grid: every frame owns a row, its tubelet is the
    # pair (a, a+k), and the 2k alignments interleave back to one row/frame.
    group = (CLIP_LEN // 2) * 2 * k              # frames (== rows) per group
    n_groups = args.windows
    need = group * n_groups + k                  # +k: the last row's partner

    frames = []
    for _t_ms, f in P.decode_frames(P.video_path(Path(args.dataset), args.id),
                                    w=res, h=res, fps=man["grid_fps"]):
        frames.append(f)
        if len(frames) >= need:
            break
    print(f"decoded {len(frames)} frames at {res}x{res}", flush=True)

    worst_abs, worst_corr, n_rows, n_diff, n_off1 = 0.0, 1.0, 0, 0, 0
    t0 = time.time()
    for g in range(n_groups):
        base = g * group
        if base + group >= len(frames):
            break
        rows_f, rows_a = [], []
        for j in range(n_align):                 # the tubelet alignments
            aa = list(range(j, group, 2 * k))[:CLIP_LEN // 2]
            idx = [i for a in aa for i in (a, a + k)]
            x = np.stack([frames[base + i] for i in idx])[None]  # (1,32,r,r,3)
            y = sess.run(None, {"frames": x})[0]                 # (16,dim,g,g)
            rows_f.append(y)
            rows_a.append(np.asarray(aa))
        a_cat = np.concatenate(rows_a)
        order = np.argsort(a_cat, kind="stable")               # back to video order
        got = np.concatenate(rows_f)[order]                    # (32,dim,g,g)

        ref_i8 = np.asarray(feats[base:base + group], np.int32)
        got_i8 = np.clip(np.round(got / INT8_SCALE), -127, 127).astype(np.int32)
        ref = ref_i8 * scale

        # NaN (non-finite output, or a constant one) must stick: max() and
        # min() would drop it and pass the gate
        d_abs, c = float(np.abs(got - ref).max()), corr(got, ref)
        worst_abs = d_abs if np.isnan(d_abs) else max(worst_abs, d_abs)
        worst_corr = c if np.isnan(c) else min(worst_corr, c)
        d = np.abs(got_i8 - ref_i8)
        n_rows += group
        n_diff += int((d > 0).sum())
        n_off1 += int((d > 1).sum())
        print(f"  group {g}: rows {base}-{base+group-1}  "
              f"max|d| {d_abs:.4f}  corr {c:.6f}  "
              f"int8 exact {(d == 0).mean():.1%}", flush=True)

    if not n_rows:
        raise SystemExit(f"{args.id} is too short for latent parity: it needs "
                         f"{need} frames, {len(frames)} decoded")
    tot = n_rows * man["dim"] * grid * grid
    print(f"\nLATENTS ({n_rows} rows, {time.time()-t0:.0f}s)", flush=True)
    print(f"  max|d| {worst_abs:.4f}   (int8 step {INT8_SCALE:.4f} -- a "
          f"difference under this is rounded away before the model sees it)")
    print(f"  corr   {worst_corr:.6f}")
    print(f"  int8:  {100 * (1 - n_diff / tot):.2f}% identical, "
          f"{100 * n_off1 / tot:.3f}% off by more than one step")
    return worst_corr


def track_parity(args, man, bundle):
    """Exported head+env over the CACHED latents vs jepa_infer's own forward."""
    import onnxruntime as ort

    model, ck = load_model(str(args.ckpt), "cpu")
    model.pos_temp = man["pos_temp"]
    ds = JepaClip(str(Path(args.dataset)), args.id, 384, 192,
                  feat_dir=ck.get("feat_dir"))
    T = min(len(ds.vel), args.rows)
    shots = [(lo, min(hi, T)) for lo, hi in
             zip(ds.shot_edges[:-1], ds.shot_edges[1:]) if lo < T]
    _pred, ref = predict_shots(model, ds, shots, "cpu", man["chunk"],
                               collect_tracks=True, ctx=man["ctx"])

    # the head graphs run on the CPU provider REGARDLESS of --ep, for two
    # reasons that point the same way. It is the provider deploy actually
    # runs them on: heads.rs owns no GPU session -- the encoder has the
    # GPU -- so a DML head run would measure a configuration nothing
    # ships. And onnxruntime-directml (1.24.4) SEGFAULTS executing the
    # attention-pooling Einsum (bhij,bdij->bhd): a one-node graph with
    # that equation reproduces it. --ep keeps governing the encoder
    # stage, where it is the deploy backend.
    head = ort.InferenceSession(str(bundle / man["graphs"]["head"]),
                                providers=["CPUExecutionProvider"])
    envs = ort.InferenceSession(str(bundle / man["graphs"]["env_step"]),
                                providers=["CPUExecutionProvider"])
    chunk = man["chunk"]
    ctx = man.get("ctx", 0)     # context rows forwarded each side, discarded
    keys = ["vmarg", "level", "blo", "bhi", "env"]
    if man["heads"].get("rev"):
        keys += ["rev_top", "rev_bot"]
    got = {k: np.full(T, np.nan) for k in keys}
    ran = np.zeros(T, bool)                 # rows the graphs wrote
    t0 = time.time()
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        fs = s if e - s >= man["min_chunk"] else max(0, e - man["min_chunk"])
        fs = max(0, min(fs, s) - ctx)
        fe = min(T, e + ctx)
        if e - fs < man["min_chunk"]:
            continue
        off = s - fs
        ran[s:e] = True
        oe = off + (e - s)
        x = np.ascontiguousarray(ds.feats[fs:fe].numpy())[None]    # int8
        cut = ds.cut[fs:fe].numpy()[None].astype(np.float32)
        out = head.run(None, {"x_i8": x, "cut": cut})
        names = [o.name for o in head.get_outputs()]
        o = dict(zip(names, out))
        got["vmarg"][s:e] = o["vmarg"][0][off:oe]
        got["level"][s:e] = o["level"][0][off:oe]
        got["blo"][s:e] = o["ext_lo"][0][off:oe]
        got["bhi"][s:e] = o["ext_hi"][0][off:oe]
        if "rev_top" in got:
            got["rev_top"][s:e] = o["rev_top"][0][off:oe]
            got["rev_bot"][s:e] = o["rev_bot"][0][off:oe]
        # the envelope's AR decode: buffer reseeds at the FORWARD's start
        # (inside the discarded ctx prefix), exactly as the Python loop
        # does inside model.forward -- kept rows see a warm history
        h = o["h"]                                    # (1, tcn, W)
        state = np.zeros((1, man["env_state"]), np.float32)
        eg = np.empty(h.shape[2], np.float32)
        for t in range(h.shape[2]):
            feed = {"h_t": h[:, :, t:t + 1], "state": state}
            if man.get("env_flow"):
                # the flow head samples, and its base draw is keyed to the
                # ABSOLUTE row -- fs + t, the same quantity heads.rs feeds
                # and model.forward reads off env_row0
                feed["e0"] = np.float32(common.env_base_draw(
                    fs + t, man.get("env_seed", 0),
                    man.get("env_base_hi", common.ENV_BASE_HI)))[None, None]
            et, state = envs.run(None, feed)
            eg[t] = et[0]
        got["env"][s:e] = eg[off:oe]
        print(f"  chunk {s}-{e}", flush=True)

    print(f"\nTRACKS ({T} rows, {time.time()-t0:.0f}s)", flush=True)
    ok = True
    for k in keys:
        r = np.asarray(ref[k][:T], np.float64)
        g = got[k]
        # a row the graphs wrote as NaN or inf stays in, and fails below
        m = np.isfinite(r) & ran
        if not m.any():
            print(f"  {k:<6} no row to compare [FAIL]")
            ok = False
            continue
        ad = np.abs(g[m] - r[m])
        d = float(ad.max())
        row = int(np.flatnonzero(m)[int(ad.argmax())])
        c = corr(g[m], r[m])
        # velocity is normalized units (x v_std = pos/s); level/band are
        # 0..100; rev tracks are probabilities
        unit = "pos/s" if k in ("vmarg", "env") else \
            ("P" if k.startswith("rev_") else "pos")
        tol = 1e-3 if k in ("vmarg", "env") or k.startswith("rev_") \
            else 1e-2
        # The level and the rails are sharpened expectations (pos_temp
        # 0.25 scales the logits x4), so an fp32 accumulation-order
        # difference between torch and ORT on a bimodal row can move that
        # row's expectation by a tenth of a position unit while the other
        # rows agree to 1e-4; a clip with more near-tie rows has more such
        # rows. That is the same backend noise the Rust stage floors by
        # mean, so a track past the max bar still passes when its mean|d|
        # is under a tenth of the bar and its corr holds; a wrong graph
        # moves every row by the bar or more and fails both. The max sits
        # beside its row, its mean and the count over the bar, so the
        # ambiguous rows read as what they are
        mean_d = float(ad.mean())
        noise = d > tol and mean_d <= tol / 10 and c >= CORR_TOL
        good = d <= tol or noise
        ok &= good
        print(f"  {k:<6} max|d| {d:.3e} {unit:<5} @row {row:<5} "
              f"corr {c:.8f}  mean|d| {mean_d:.3e}  "
              f"rows>tol {int((ad > tol).sum())}"
              f"{'  below the floor' if noise else ''}  "
              f"[{'OK' if good else 'FAIL'}]")
    return ok


def find_exe(given):
    """The binary to drive: what was asked for, else the newest build."""
    if given:
        p = Path(given)
        if not p.exists():
            raise SystemExit(f"no goblinscript binary at {p}")
        return p
    found = [p for p in (GS / "target/release/goblinscript.exe",
                         GS / "target/debug/goblinscript.exe",
                         GS / "dist/goblinscript.exe") if p.exists()]
    if not found:
        raise SystemExit("no goblinscript binary built -- `cargo build "
                         "--release` in goblinscript/, or pass --exe")
    return max(found, key=lambda p: p.stat().st_mtime)


def rust_parity(args, man, bundle):
    """heads.rs's OWN tiling over the cached latents vs jepa_infer's forward.

    The tracks stage above drives the graphs from a Python loop, so it
    measures the exported graphs and re-implements the chunking around them.
    This one hands the shipped binary the same rows and reads back what it
    produced -- the chunk boundaries, the ctx padding, the short-tail
    lookback and the envelope's stepped buffer are the Rust ones, which
    nothing else here exercises without going through the encoder first.

    The binary runs on its own execution provider, so a small uniform offset
    against a torch reference is backend numerics and expected. A TILING bug
    does not look like that: it lands on the rows around a chunk boundary and
    leaves the interior alone, which is why the report splits the two.
    """
    import subprocess
    import tempfile

    exe = find_exe(args.exe)
    cuts = Path(args.dataset) / "boundaries" / f"{args.id}.json"
    if not cuts.exists():
        raise SystemExit(f"no boundaries for {args.id} ({cuts}) -- the Rust "
                         "side needs the same seams the reference saw")
    model, ck = load_model(str(args.ckpt), "cpu")
    model.pos_temp = man["pos_temp"]
    ds = JepaClip(str(Path(args.dataset)), args.id, 384, 192,
                  feat_dir=ck.get("feat_dir"))
    T = min(len(ds.vel), args.rows)
    shots = [(lo, min(hi, T)) for lo, hi in
             zip(ds.shot_edges[:-1], ds.shot_edges[1:]) if lo < T]
    _pred, ref = predict_shots(model, ds, shots, "cpu", man["chunk"],
                               collect_tracks=True, ctx=man["ctx"])

    with tempfile.TemporaryDirectory(prefix="goblin_parity_") as tmp:
        lat = Path(tmp) / "latents.i8"
        out = Path(tmp) / "tracks.json"
        # the binary reads a whole clip, so the dumped rows ARE the clip --
        # the same truncation the reference above scored
        ds.feats[:T].numpy().astype(np.int8).tofile(lat)
        cmd = [str(exe), "--bundle", str(bundle), "--model", man["pack"],
               "--from-latents", str(lat), "--cuts-json", str(cuts),
               "--dump-tracks", str(out), "--quiet"]
        print(f"{exe.name} over {T} rows ({lat.stat().st_size / 1e6:.0f} MB)",
              flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"{exe.name} failed ({r.returncode}):\n"
                             f"{r.stdout}\n{r.stderr}")
        got = __import__("json").loads(out.read_text("utf-8"))
    for line in r.stdout.splitlines():
        print(f"  {line}", flush=True)

    v_std = float(man["v_std"])
    chunk, ctx = man["chunk"], man.get("ctx", 0)
    # rows the tiling could plausibly damage: the chunk seams themselves
    edge = np.zeros(T, bool)
    for s in range(chunk, T, chunk):
        edge[max(0, s - 2):min(T, s + 2)] = True
    print(f"\nRUST TRACKS ({T} rows, {time.time() - t0:.0f}s, chunk {chunk} "
          f"+ ctx {ctx}, {int(edge.sum())} seam rows)", flush=True)
    ok = True
    for k in ["vmarg", "level", "blo", "bhi", "env", "rev_top", "rev_bot",
              "plat_top", "plat_bot"]:
        if k not in got or k not in ref:
            continue
        r_ = np.asarray(ref[k][:T], np.float64)
        # vmarg and env leave the binary in pos/s; the reference is normalized
        g = np.asarray([np.nan if v is None else v for v in got[k][:T]],
                       np.float64)
        if k in ("vmarg", "env"):
            g = g / v_std
        m = np.isfinite(r_) & np.isfinite(g)
        d = np.abs(g - r_)
        worst = int(np.nanargmax(np.where(m, d, -np.inf)))
        seam = m & edge
        interior = m & ~edge
        unit = "pos/s" if k in ("vmarg", "env") else \
            ("P" if k[:4] in ("rev_", "plat") else "pos")
        # This stage gates the TILING, not the graphs -- the tracks stage
        # above already holds the graphs to max|d|, running the same ORT on
        # the same provider as the reference. Here a different backend at a
        # different precision answers, so a max-based bar measures fp16 and
        # would fail a perfectly tiled decode. What a tiling fault looks
        # like instead: seam rows diverging where interior rows do not.
        # seam rows are a few dozen against thousands of interior ones, so
        # the two are compared by MEAN: a max over 340x more samples is
        # larger on any distribution and would read as a tiling fault on a
        # perfectly tiled decode
        sm = float(d[seam].mean()) if seam.any() else float("nan")
        im = float(d[interior].mean()) if interior.any() else float("nan")
        ratio = sm / im if im else float("nan")
        c = corr(g[m], r_[m])
        quiet = sm < SEAM_FLOOR
        good = (quiet or ratio <= SEAM_TOL) and (c >= CORR_TOL)
        ok &= good
        print(f"  {k:<9} max|d| {d[m].max():.3e} {unit:<5} @row {worst}  "
              f"corr {c:.8f}  "
              f"mean|d| seam {sm:.3e} / interior {im:.3e} ({ratio:.2f}x)"
              f"{' below the floor' if quiet else ''}"
              f"  [{'OK' if good else 'FAIL'}]")
    print(f"  seam rows {int(edge.sum())}, interior {int((~edge).sum())} -- a "
          f"seam mean under {SEAM_FLOOR:g} is backend noise whatever its "
          f"ratio; above it, a ratio near 1 is execution-provider numerics; "
          f"well above 1 is heads.rs's tiling")
    return ok


def resolve_manifest(bundle, pack=None):
    """The flat manifest of one pack: the perception's keys plus the pack's,
    with the pack's graphs under packs/<name>/, the way bundle.rs resolves
    it. The bundle carries one perception and several models, each a pack
    named for the release that shipped it; `pack` None is the default."""
    man = json.loads((bundle / "manifest.json").read_text("utf-8"))
    name = pack or man["default"]
    if name not in man["packs"]:
        raise SystemExit(f"no pack {name!r} in {bundle} "
                         f"(packs: {sorted(man['packs'])})")
    p = dict(man["packs"][name])
    flat = {k: v for k, v in man.items() if k not in ("packs", "default")}
    flat["graphs"] = dict(flat["graphs"], **{
        k: f"packs/{name}/{v}" for k, v in p.pop("graphs").items()})
    flat.update(p)
    flat["pack"] = name
    return flat


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--id", default=None,
                    help="a prepared clip of the project (its latents are "
                         "the reference)")
    ap.add_argument("--dataset", default=None, help="the project directory")
    ap.add_argument("--ckpt", default=None,
                    help="Python reference checkpoint. Default: the bundle "
                         "manifest's own source checkpoint -- comparing "
                         "against any OTHER model measures the models' "
                         "difference, not the export's")
    ap.add_argument("--bundle", default=str(GS / "bundle"))
    ap.add_argument("--pack", default=None,
                    help="the model of the bundle to check, by pack name; "
                         "default: the bundle's default pack. Each pack is "
                         "one export against its own checkpoint, so every "
                         "pack is checked on its own")
    ap.add_argument("--windows", type=int, default=4,
                    help="32-frame groups to check (each is 2 encoder forwards)")
    ap.add_argument("--rows", type=int, default=4096,
                    help="rows of the cache to run the head over. Must "
                         "EXCEED chunk+ctx (2048 on the shipped manifest) "
                         "for the --rust seam gate to mean anything: at "
                         "rows <= chunk+ctx every chunk forwards the same "
                         "full span, no two forwards differ, and the seam "
                         "ratio measures backend noise instead of tiling")
    ap.add_argument("--skip-latents", action="store_true")
    ap.add_argument("--skip-tracks", action="store_true")
    ap.add_argument("--rust", action="store_true",
                    help="also run the shipped binary's own head stage over "
                         "the same cached rows -- the only test that drives "
                         "heads.rs's chunking without going through the "
                         "encoder first")
    ap.add_argument("--exe", default=None,
                    help="the goblinscript binary --rust drives (default: the "
                         "newest of target/release, target/debug, dist)")
    # NOTE: --ep governs the ENCODER stage; the tracks stage's head graphs
    # are pinned to the CPU provider (see track_parity)
    ap.add_argument("--ep", default="DmlExecutionProvider",
                    help="execution provider for EVERY graph this runs. The "
                         "default is the backend a user actually runs, which "
                         "is where fp16 range and fused-kernel numerics show "
                         "up -- and the encoder graph's packed-QKV attention "
                         "has no CPU kernel at all, so CPU cannot run a "
                         "shipped bundle's encoder. Pass CPUExecutionProvider "
                         "deliberately, as the fp32 no-vendor-kernel numerics "
                         "reference, and expect it to be slow enough to plan "
                         "around")
    return ap


def run(args):
    """Check under a parsed ``build_parser()`` namespace. Returns True when
    every stage passed."""
    if args.dataset is None or args.id is None:
        raise SystemExit("--dataset and --id are required")
    import onnxruntime as ort
    bundle = Path(args.bundle)
    man = resolve_manifest(bundle, args.pack)
    if args.ckpt is None:
        # the manifest's pointer is repo-relative when the checkpoint
        # lives inside the tree
        p = Path(man["checkpoint"])
        args.ckpt = str(p if p.is_absolute() else ROOT / p)
    print(f"bundle pack {man['pack']}: {man['checkpoint']} (epoch "
          f"{man['epoch']}, basis {man['basis_id']})\n"
          f"python reference {args.ckpt}\n", flush=True)

    ok = True
    if not args.skip_latents:
        eps = [args.ep] if args.ep == "CPUExecutionProvider" \
            else [args.ep, "CPUExecutionProvider"]
        sess = ort.InferenceSession(str(bundle / man["graphs"]["encoder"]),
                                    providers=eps)
        got = sess.get_providers()[0]
        if got != args.ep:
            raise SystemExit(f"--ep {args.ep} was asked for but ORT placed the "
                             f"encoder on {got} -- a silent CPU fallback would "
                             f"report the wrong backend's numerics")
        print(f"encoder on {got}", flush=True)
        ok &= latent_parity(args, sess, man) >= 0.999
    if not args.skip_tracks:
        print()
        ok &= track_parity(args, man, bundle)
    if args.rust:
        print()
        ok &= rust_parity(args, man, bundle)
    return ok


def main():
    if not run(build_parser().parse_args()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
