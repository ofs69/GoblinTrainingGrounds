"""Decode-invariant check: durations stay durations, and the two decoders agree.

Runs on synthetic tracks (CPU, seconds, no dataset or GPU) and asserts four
layers:

  constants -- every tuned duration lands on the row count it is meant to on
               the project grid. A pin, not a derivation: these were tuned as
               kernel widths, so a "tidy" edit that moves one to a rounder
               number is a measured change wearing a spelling change, and it
               fails here.

  mirror    -- goblinscript's styling constants (style.rs) equal common.py's,
               value for value. They are the same decode written twice; the
               moment they disagree the Rust draft stops being the Python one,
               and nothing else in the tree would notice. The deploy bundle's
               ROW CLOCK is mirrored the same way: the app synthesizes row
               times where the harness reads them off a cache, so this is the
               only place those two definitions meet.

  grid      -- the same wall-clock signal sampled on a SECOND, arbitrary row
               grid gives wall-clock-equivalent output. There is one shipping
               grid, so this proves a property rather than supporting a second
               deployment: no row literal is baked into the grid-sensitive
               paths. A bare row count anywhere in them fails here.

  behavior  -- per-shot smoothers stay inside their shot (a shot shorter than
               its own kernel), and the viterbi graft alternates, spaces and
               survives a sloping level into the written actions.

New duration constants get a line in CONSTANTS; new grid-sensitive stages get
a behavior check. Run after touching any of them:

    python grid_check.py
"""
import json
import re
from pathlib import Path

import numpy as np

from . import common
from . import jepa_infer
from . import forward
from . import jepa_train
from . import autocrop

REPO = Path(__file__).resolve().parent.parent   # the repository
HZ = common.ROW_HZ        # the shipping grid
ALT = 60.0                # an arbitrary second grid: only to prove that
# nothing in the duration paths is a row literal. Not a deployment.

# (name, seconds, odd, rows on the shipping grid). The row count is the pin.
CONSTANTS = [
    ("DWELL_SMOOTH_S", common.DWELL_SMOOTH_S, True, 15),
    ("DWELL_MIN_S", common.DWELL_MIN_S, False, 16),
    ("EXTREMUM_SMOOTH_S", common.EXTREMUM_SMOOTH_S, True, 7),
    ("REV_SMOOTH_S", common.REV_SMOOTH_S, True, 3),
    ("LOCK_SMOOTH_S", common.LOCK_SMOOTH_S, True, 15),
    ("LOCK_EDGE_S", common.LOCK_EDGE_S, False, 6),
    ("REV_SNAP_S", common.REV_SNAP_S, False, 2),
    ("REV_TOL_S", common.REV_TOL_S, False, 2),
    ("EVENT_GAP_S", common.EVENT_GAP_S, False, 2),
    ("DECODE_SMOOTH_S", common.DECODE_SMOOTH_S, True, 31),
    ("DECODE_CHUNK_S", common.DECODE_CHUNK_S, False, 1024),
    ("DECODE_CTX_S", common.DECODE_CTX_S, False, 1024),
    ("LEVEL_SMOOTH_S", common.LEVEL_SMOOTH_S, True, 63),
    ("infer.DWELL_GAP_S", jepa_infer.DWELL_GAP_S, False, 8),
    ("infer.DWELL_MIN_CALL_S", jepa_infer.DWELL_MIN_CALL_S, False, 8),
    ("infer.VETO_FLANK_S", jepa_infer.VETO_FLANK_S, False, 60),
    ("infer.STILL_SMOOTH_S", jepa_infer.STILL_SMOOTH_S, True, 31),
    ("infer.SPEED_REF_S", jepa_infer.SPEED_REF_S, True, 19),
    ("infer.MIN_SPAN_S", jepa_infer.MIN_SPAN_S, False, 60),
    ("infer.BIAS_FIT_S", jepa_infer.BIAS_FIT_S, False, 240000),
    ("train.ENV_SMOOTH_S", jepa_train.ENV_SMOOTH_S, True, 19),
    ("common.SEG_MARGIN_S", common.SEG_MARGIN_S, False, 60),
    ("common.VAL_SEG_S", common.VAL_SEG_S, False, 4000),
    ("common.VAL_SEG_TARGET_S", common.VAL_SEG_TARGET_S, False, 1920),
    ("common.MIN_VAL_SEG_S", common.MIN_VAL_SEG_S, False, 480),
    ("train.EVAL_CTX_S", jepa_train.EVAL_CTX_S, False, 384),
    ("forward.MIN_PIECE_S", forward.MIN_PIECE_S, False, 60),
]

# common.py name -> style.rs name, for the constants the Rust styler carries
# its own copy of. Only the styling set: Rust runs no training or metrics.
MIRRORED = {
    "DECODE_SMOOTH_S": "DECODE_SMOOTH_S",
    "LOCK_SMOOTH_S": "LOCK_SMOOTH_S",
    "LOCK_EDGE_S": "LOCK_EDGE_S",
    # Not a duration: the device speed cap both decodes clamp their written
    # actions to. It is here because it drifted -- the Rust styler applied it
    # only when the user asked, so every shipped draft carried transitions the
    # Python decode would have pulled in.
    "MAX_POS_RATE": "MAX_POS_RATE",
}
MIRRORED_INFER = {
    "DWELL_GAP_S": "DWELL_GAP_S",
    "DWELL_MIN_CALL_S": "DWELL_MIN_CALL_S",
    # Also not a duration: the action list's Douglas-Peucker epsilon. Both
    # decodes simplify what they WRITE, and style.rs states the invariant in
    # a doc comment, which is the kind a check has to hold up.
    "RDP_EPS": "RDP_EPS",
    # Same footing as RDP_EPS: the seam-ease speed at shot cuts. Artifact
    # layer, written twice, so the default a sweep tunes is the default a
    # draft ships.
    "CUT_EASE": "CUT_EASE",
    # the phase reading's reach and run length, recorded by both decodes
    "REV_SUPPORT_S": "REV_SUPPORT_S",
    "PHASE_RUN_K": "PHASE_RUN_K",
}
STYLE_RS = REPO / "goblinscript" / "src" / "style.rs"
# autocrop.py name -> autocrop.rs name. The crop recipe is written twice and
# the two copies decide the same rects, so a constant moved on one side only
# is a silent divergence between what is tuned and what ships.
MIRRORED_CROP = {
    "FLOOR_Q": "FLOOR_Q",
    "PIC_DEAD_LUMA": "PIC_DEAD_LUMA",
    "TOP_MASS_Q": "TOP_MASS_Q",
    "EDGE_Q": "EDGE_Q",
    "MARGIN_CELLS": "MARGIN_CELLS",
    "SUBCELL": "SUBCELL",
    "MIN_SIDE_FRAC": "MIN_SIDE_FRAC",
    "IDENT_FRAC": "IDENT_FRAC",
}
CROP_RS = REPO / "goblinscript" / "src" / "autocrop.rs"
# The deploy bundle's grid, and the Rust side's copy of where row 0 sits.
BUNDLE_MANIFEST = (REPO / "goblinscript"
                   / "bundle" / "manifest.json")
BUNDLE_RS = REPO / "goblinscript" / "src" / "bundle.rs"
# Definitions this tree AUTHORS and goblinscript embeds verbatim. They are two
# repositories now, so each side carries its own copy and a copy is a thing that
# drifts: the prior would score artifacts against a different fit, and the
# projector would make the preview aim somewhere the render does not. Byte
# equality is the whole check -- neither file has a "close enough".
MIRRORED_FILES = {"artifact_prior.json": common.WEIGHTS_DIR / "artifact_prior.json"}


def track(t):
    """Synthetic position 0..100 over seconds ``t``: slow strokes, a fast
    burst, two >=1 s plateaus -- every regime the duration constants gate."""
    p = np.full_like(t, 50.0)
    p += np.where(t < 20.0, 35.0 * np.sin(2 * np.pi * 0.6 * t), 0.0)
    m = (t >= 20.0) & (t < 24.0)                     # top plateau w/ ripple
    p[m] = 88.0 + 3.0 * np.sin(2 * np.pi * 2.0 * t[m])
    m = (t >= 24.0) & (t < 34.0)                     # fast burst, ~2.5 Hz
    p[m] = 50.0 + 30.0 * np.sin(2 * np.pi * 2.5 * (t[m] - 24.0))
    m = (t >= 34.0) & (t < 36.0)                     # bottom plateau
    p[m] = 10.0
    m = t >= 36.0
    p[m] = 50.0 + 40.0 * np.sin(2 * np.pi * 1.0 * (t[m] - 36.0))
    return np.clip(p, 0.0, 100.0)


def sample(hz, dur_s=60.0):
    n = int(dur_s * hz)
    t = (np.arange(n) + 0.5) / hz
    return t, track(t)


def check_constants():
    for name, s, odd, want in CONSTANTS:
        got = common.rows_at(s, HZ, odd=odd)
        assert got == want, \
            f"{name} = {s:.6f}s -> {got} rows at {HZ:g} (pinned {want}). " \
            f"If this move is intended it is a MEASURED change: re-run the " \
            f"gate before repinning."
        # the same duration on any other grid, within half a row of rounding
        alt = common.rows_at(s, ALT, odd=odd)
        assert abs(alt / ALT - got / HZ) <= 0.5 / HZ + 1e-9, \
            f"{name} drifts across grids: {got / HZ:.4f}s vs {alt / ALT:.4f}s"
    print(f"constants: {len(CONSTANTS)} durations pinned at {HZ:g} rows/s "
          f"and grid-invariant")


def check_rust_mirror():
    """style.rs carries its own copy of the styling constants. Same numbers."""
    if not STYLE_RS.exists():                # a python-only checkout
        print("mirror: style.rs absent, skipped")
        return
    src = STYLE_RS.read_text(encoding="utf-8")
    found = dict(re.findall(
        r"const\s+(\w+):\s*(?:f64|usize)\s*=\s*([0-9.]+)\s*;", src))
    checked = 0
    for py_name, rs_name in MIRRORED.items():
        assert rs_name in found, f"style.rs has no const {rs_name}"
        py, rs = getattr(common, py_name), float(found[rs_name])
        assert abs(py - rs) < 5e-6, \
            f"style.rs {rs_name} = {rs} != common.{py_name} = {py} -- the " \
            f"Rust draft is no longer the Python one"
        checked += 1
    for py_name, rs_name in MIRRORED_INFER.items():
        assert rs_name in found, f"style.rs has no const {rs_name}"
        py, rs = getattr(jepa_infer, py_name), float(found[rs_name])
        assert abs(py - rs) < 5e-6, \
            f"style.rs {rs_name} = {rs} != jepa_infer.{py_name} = {py}"
        checked += 1
    print(f"mirror: {checked} styling constants identical in style.rs")


def check_crop_mirror():
    """autocrop.rs carries its own copy of the rect recipe. Same numbers."""
    if not CROP_RS.exists():
        print("mirror: autocrop.rs absent, skipped")
        return
    src = CROP_RS.read_text(encoding="utf-8")
    found = dict(re.findall(
        r"const\s+(\w+):\s*(?:f32|f64|usize|i64)\s*=\s*([0-9.]+)\s*;", src))
    for py_name, rs_name in MIRRORED_CROP.items():
        assert rs_name in found, f"autocrop.rs has no const {rs_name}"
        py, rs = getattr(autocrop, py_name), float(found[rs_name])
        assert abs(py - rs) < 5e-5, \
            f"autocrop.rs {rs_name} = {rs} != autocrop.{py_name} = {py} -- " \
            f"the shipped crop is no longer the tuned one"
    # the cap is a continuous fraction of the frame, and the one normalized
    # height deploy transcodes at is derived from it
    cap = 1.0 / autocrop.MIN_SIDE_FRAC
    assert abs(cap - 1.5) < 0.02, f"the zoom cap is {cap:.3f}x, not the x1.5 pinned"
    print(f"mirror: {len(MIRRORED_CROP)} crop constants identical in "
          f"autocrop.rs (zoom cap x{cap:.2f})")


# The sub-cell fixture: ONE Gaussian blob, its centre deliberately between
# cell centres, read through the whole refinement path. The expected numbers
# live in `autocrop.rs`'s own test (the CROP_FIXTURE lines) and are recomputed
# here from `autocrop.py`, so a refinement changed on one side only cannot
# survive both. Sub-cell edges are the whole reason the rect left the grid,
# and an interpolation is far easier to write two ways than to write twice.
CROP_FIXTURE = dict(cx=11.3, cy=9.7, sigma=2.5, grid=24, rows=40)


def check_crop_fixture():
    if not CROP_RS.exists():
        print("fixture: autocrop.rs absent, skipped")
        return
    src = CROP_RS.read_text(encoding="utf-8")
    want = {}
    for name, nums in re.findall(
            r"CROP_FIXTURE (box|rect) ([-0-9. ]+)", src):
        want[name] = [float(v) for v in nums.split()]
    assert set(want) == {"box", "rect"}, \
        "autocrop.rs carries no CROP_FIXTURE box and rect lines"

    g = CROP_FIXTURE["grid"]
    y, x = np.mgrid[0:g, 0:g]
    m = np.exp(-(((x - CROP_FIXTURE["cx"]) ** 2 + (y - CROP_FIXTURE["cy"]) ** 2)
                 / (2 * CROP_FIXTURE["sigma"] ** 2))).astype(np.float32)
    m /= m.sum()
    box, _conc = autocrop.row_boxes(m[None])
    rect = autocrop.shot_rect(np.repeat(m[None], CROP_FIXTURE["rows"], axis=0), g)
    for name, got in (("box", box[0]), ("rect", rect)):
        for i, (a, b) in enumerate(zip(got, want[name])):
            assert abs(a - b) < 1e-6, \
                f"crop {name}[{i}] = {a:.8f}, autocrop.rs pins {b:.8f} -- " \
                f"the two languages read one attention map differently"
    print(f"fixture: the sub-cell box and rect agree with autocrop.rs "
          f"(box x {want['box'][0]:.5f}..{want['box'][1]:.5f} of the frame, "
          f"{want['box'][0] * g:.2f} cells)")


def check_embedded_files():
    """goblinscript embeds this tree's prior and projector at build time, from
    its own copies. Byte-identical, or the two decodes are not the same one."""
    for name, ours in MIRRORED_FILES.items():
        theirs = REPO / "goblinscript" / "src" / name
        if not theirs.exists():
            print(f"mirror: goblinscript/src/{name} absent, skipped")
            continue
        a, b = ours.read_bytes(), theirs.read_bytes()
        assert a == b, \
            f"goblinscript/src/{name} is not this tree's {name} " \
            f"({len(a)} vs {len(b)} bytes) -- the embedded copy has drifted " \
            f"from the definition it is supposed to be"
    print(f"mirror: {len(MIRRORED_FILES)} embedded files byte-identical "
          f"({', '.join(MIRRORED_FILES)})")


def check_row_clock():
    """The deploy bundle's row clock is the extractor's row clock.

    goblinscript has no cache to read `times_ms` from -- it synthesizes row
    times from the manifest -- so this is the one place the two definitions
    meet. A row is the midpoint of the tubelet pair it was encoded from, and
    an offset here moves every action in every draft the app writes while
    every in-repo metric stays silent, because they all read the cache.
    """
    if not BUNDLE_MANIFEST.exists() or not BUNDLE_RS.exists():
        print("row clock: deploy bundle absent, skipped")
        return
    man = json.loads(BUNDLE_MANIFEST.read_text(encoding="utf-8"))
    k, fps = man["tubelet_stride"], man["grid_fps"]
    row_hz = man.get("row_hz") or fps * man["alignments"] / (2 * k)
    common.check_row_hz(row_hz, HZ, what="deploy bundle")
    # extract.py's own arithmetic for row 0: the midpoint of frames (0, k)
    want0 = ((0.0 / fps) + (k / fps)) / 2 * 1000.0
    found = re.search(r"SHIPPED_ROW0_MS:\s*f64\s*=\s*([0-9.]+)\s*;",
                      BUNDLE_RS.read_text(encoding="utf-8"))
    assert found, "bundle.rs has no SHIPPED_ROW0_MS to mirror"
    rs = float(found.group(1))
    assert abs(rs - want0) < 1e-4, \
        f"bundle.rs row 0 = {rs} ms != the extractor's {want0:.6f} ms -- " \
        f"every action the app writes is off by their difference"
    print(f"row clock: row 0 at {want0:.3f} ms, {1000.0 / row_hz:.3f} ms/row "
          f"(tubelet stride {k} at {fps:g} fps), mirrored in bundle.rs")


def spans_s(spans, hz):
    return sorted((round(a / hz, 1), round(b / hz, 1)) for a, b, *_ in spans)


def check_labels():
    out = {}
    for hz in (HZ, ALT):
        _t, p = sample(hz)
        out[hz] = spans_s(common.dwell_spans(p, row_hz=hz), hz)
    assert len(out[HZ]) >= 2, f"dwell detector found {out[HZ]}"
    # spans match by midpoint; one unmatched span per side is clip-edge
    # detector behavior, more is a duration bug (a halved min-span floor
    # roughly DOUBLES the span count)
    mids = np.array([(a + b) / 2 for a, b in out[ALT]])
    unmatched = sum(np.abs(mids - (a + b) / 2).min() > 0.3
                    for a, b in out[HZ])
    assert unmatched <= 1, \
        f"{unmatched} dwell spans unmatched: {out[HZ]} vs {out[ALT]}"
    assert abs(len(out[HZ]) - len(out[ALT])) <= 1, \
        f"dwell count differs: {len(out[HZ])} vs {len(out[ALT])}"
    print(f"labels: dwell_spans wall-clock stable "
          f"({len(out[HZ])} at {HZ:g} / {len(out[ALT])} at {ALT:g})")

    for hz in (HZ, ALT):
        _t, p = sample(hz)
        lab = common.reversal_labels(p, np.ones(len(p), dtype=bool),
                                     row_hz=hz)
        out[hz] = np.flatnonzero(lab) / hz
    a, b = out[HZ], out[ALT]
    assert abs(len(a) - len(b)) <= 2, \
        f"reversal count differs: {len(a)} vs {len(b)}"
    for r in a:
        # one row of the coarser grid: the odd-width kernels are the same
        # duration to within their rounding (7/30 s vs 13/60 s), so a call
        # can land one row apart and no further
        assert np.abs(b - r).min() <= 1.0 / HZ + 1e-9, \
            f"reversal at {r:.2f}s has no partner on the other grid"
    print(f"labels: reversal_labels wall-clock stable ({len(a)} reversals)")


def check_actions():
    times = {}
    for hz in (HZ, ALT):
        jepa_infer.FPS, jepa_infer.DT = hz, 1.0 / hz
        t, p = sample(hz)
        acts = jepa_infer.extrema_actions(p, t * 1000.0, [0, len(p)])
        times[hz] = np.array([a["at"] for a in acts], dtype=float) / 1000.0
    jepa_infer.FPS, jepa_infer.DT = HZ, 1.0 / HZ
    a, b = times[HZ], times[ALT]
    # a denser grid may resolve MORE reversals; every action on the shipping
    # grid must have a partner within one of its rows
    for r in a:
        assert np.abs(b - r).min() <= 1.0 / HZ + 1e-9, \
            f"action at {r:.2f}s has no partner on the other grid"
    print(f"actions: extrema_actions wall-clock stable "
          f"({len(a)} at {HZ:g} / {len(b)} at {ALT:g})")


def check_cut_ease_fixture():
    """The seam ease decides the same vertices in both languages.

    style.rs runs this fixture in
    ``the_cut_ease_runs_the_seam_move_between_real_reversals`` and pins
    the same two literal lists. The constant check above says the two sides
    agree on a NUMBER; only a fixture says they agree on the RULE.

    Two shots at 30 rows/s, cut at row 60: shot A rises 20 -> 40 then holds,
    shot B opens at 70. Off, shot A's forced closing vertex sits one row
    before the cut and the 30-unit level change is asked for in 33 ms
    (909 pos/s, which MAX_POS_RATE then pays for in DEPTH).

    On, BOTH forced vertices go. Dropping
    A's closer alone would run the move from A's last real reversal into B's
    opener -- 30 units over 67 ms, still 448 pos/s and still above the
    limit -- so B's opener goes too and the move runs to B's first real
    reversal: 30 units over 1034 ms, 29 pos/s, spanning the cut the way a
    scripter's stroke does. Every kept vertex is a real reversal.
    """
    hz = 30.0
    jepa_infer.FPS, jepa_infer.DT = hz, 1.0 / hz
    p = np.empty(120)
    p[:30] = np.linspace(20, 40, 30)
    p[30:60] = 40.0
    p[60:90] = 70.0
    p[90:] = np.linspace(70, 55, 30)
    t = np.arange(120) * (1000.0 / hz)
    want = {0.0: [(0, 20), (1967, 40), (2000, 70), (2967, 70), (3967, 55)],
            300.0: [(0, 20), (1933, 40), (2967, 70), (3967, 55)]}
    for ease, expect in want.items():
        acts = jepa_infer.extrema_actions(p, t, [0, 60, 120], cut_ease=ease,
                                          rdp_eps=jepa_infer.RDP_EPS)
        got = [(a["at"], a["pos"]) for a in acts]
        assert got == expect, \
            f"cut_ease={ease:g}: {got} != style.rs's {expect}"
    jepa_infer.FPS, jepa_infer.DT = HZ, 1.0 / HZ
    print("cut ease: the seam fixture decides identically in both languages")


def check_phase_runs_fixture():
    """The run-share phase reading decides the same runs in both languages.

    style.rs runs this table in
    ``contradicting_runs_match_the_decoder_table`` and pins the same
    literals: crossings and apexes at their event rows, the reach endpoint
    inside, agreement winning over a crossing of each direction, an exact
    zero reading +1, a run of three counted and a run of two not.
    """
    rise = np.array([-1.0] * 5 + [1.0] * 5)
    bump = np.array([-1.0] * 4 + [1.0] * 2 + [-1.0] * 4)
    zero = np.sign(np.array([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]))
    zero[zero == 0] = 1
    valleys = np.ones(60)
    for v in (5, 15, 25, 35, 45, 55):
        valleys[v - 2:v + 1] = -1.0
    rows = [5, 15, 25, 31, 35, 45, 55]
    table = [((rise, [2], [1], 2, 1), [(0, 1)]),
             ((rise, [6], [1], 2, 1), [(0, 1)]),
             ((rise, [7], [1], 2, 1), []),
             ((rise, [4], [-1], 2, 1), []),
             ((bump, [4], [1], 2, 1), []),
             ((zero, [3], [1], 2, 1), [(0, 1)]),
             ((zero, [4], [1], 2, 1), []),
             ((valleys, rows, [1] * 7, 2, 3), [(0, 3), (4, 7)]),
             ((valleys, rows[:6], [1] * 6, 2, 3), [(0, 3)])]
    for (sg, r, k, w, n), want in table:
        got, _d, _u = jepa_infer.contradicting_runs(
            np.array(r), np.array(k), sg, w, n)
        assert got == want,             f"contradicting_runs{(r, k, w, n)}: {got} != style.rs's {want}"
    assert jepa_infer.phase_runs_field({"apexes": 0, "in_runs": 0}) ==         {"apexes": 0, "in_runs": 0, "share": None}
    assert common.rows_at(jepa_infer.REV_SUPPORT_S, HZ) == 2
    print("phase runs: the fixture decides identically in both languages")


def check_amp_bound_fixture():
    """The amplitude bound decides the same endpoint in both languages.

    style.rs runs this table in
    ``the_amplitude_bound_holds_the_stroke_to_the_marginals_travel`` and
    pins the same literals. 30 rows/s; a 3-row segment is a 5 Hz stroke and
    a 60-row one 0.25 Hz.
    """
    dt = 1.0 / 30.0
    # (end, cur, tr_raw, n_rows, x) -> end
    table = [((90.0, 50.0, 10.0, 3, 0.0), 90.0),
             ((90.0, 50.0, 10.0, 3, 2.0), 70.0),
             ((90.0, 50.0, 30.0, 3, 2.0), 90.0),
             ((10.0, 50.0, 10.0, 3, 2.0), 30.0)]
    for (end, cur, tr, n, x), want in table:
        got = jepa_infer.amp_bound(end, cur, tr, n, dt, x)
        assert abs(got - want) < 1e-9, \
            f"amp_bound{(end, cur, tr, n, x)}: {got} != style.rs's {want}"
    print("amp bound: the fixture decides identically in both languages")


def check_cut_slew_fixture():
    """The seam slew clamps the same depths in both languages.

    style.rs runs this fixture in
    ``the_seam_slew_holds_the_cut_crossing_to_the_limit`` and pins the same
    four literal lists. The ease fixture above never needs the slew (its
    eased crossing lands at 29 pos/s); these are built so the ease runs out
    of levers and the slew is what remains -- both forced vertices drop and
    the move between real reversals is still over the limit (case 2), the
    sub-frame relocation puts the slam inside one shot (case 3), and the
    row-snap geometry puts it just below the whole-row window bound
    (case 4, which is what pins the HALF-ROW walk boundaries).
    """
    hz = 30.0
    jepa_infer.FPS, jepa_infer.DT = hz, 1.0 / hz
    t = np.arange(120) * (1000.0 / hz)
    p1 = np.empty(120)
    p1[:57] = np.linspace(10, 38, 57)    # shot A rises...
    p1[57:60] = np.linspace(38, 35, 3)   # ...turning just before the cut
    p1[60:64] = np.linspace(95, 91, 4)   # shot B opens high, dips...
    p1[64:67] = np.linspace(91, 97, 3)   # ...rises again...
    p1[67:] = np.linspace(97, 40, 53)    # ...then falls away
    # the row-snap leak: the video's own step lands ONE ROW BEFORE the
    # snapped edge (boundaries snap by searchsorted, so the edge lands on
    # the first row at or after the cut ms), so the slam is wholly inside
    # shot A and its end vertex rounds to just below c - 2 rows -- caught
    # only because the walk starts on a half-row boundary below that
    p2 = np.empty(120)
    p2[:56] = np.linspace(10, 38, 56)
    p2[56:59] = np.linspace(38, 35, 3)
    p2[59:62] = np.linspace(35, 95, 3)
    p2[62:] = np.linspace(96, 40, 58)
    # (track, edges, ease, sub) -> the written list. Case 3 is the
    # sub-frame leak the slew runs post-RDP for: the incoming dip's refined
    # time lands on the OUTGOING side of the cut, so the slam sits between
    # two vertices of ONE shot and only the seam-window walk can see it.
    want = [(p1, [0, 60, 120], 0.0, None,
             [(0, 10), (1867, 38), (1967, 35), (2000, 95),
              (2100, 91), (2233, 97), (3967, 40)]),
            (p1, [0, 60, 120], 250.0, None,
             [(0, 10), (1867, 38), (2033, 79), (2100, 91),
              (2233, 97), (3967, 40)]),
            (p1, [0, 60, 120], 250.0, {63: 1990.0},
             [(0, 10), (1867, 38), (2000, 71), (2033, 79),
              (2233, 97), (3967, 40)]),
            (p2, [0, 62, 120], 250.0, None,
             [(0, 10), (1833, 38), (1933, 35), (2000, 51),
              (2033, 59), (2067, 67), (3967, 40)])]
    for p, edges, ease, sub, expect in want:
        acts = jepa_infer.extrema_actions(p, t, edges, sub=sub,
                                          cut_ease=ease,
                                          rdp_eps=jepa_infer.RDP_EPS)
        got = [(a["at"], a["pos"]) for a in acts]
        assert got == expect, \
            f"cut_ease={ease:g} sub={sub}: {got} != style.rs's {expect}"
        for a, b in zip(got, got[1:]):
            v = abs(b[1] - a[1]) * 1000.0 / (b[0] - a[0])
            assert ease == 0.0 or v <= ease, \
                f"cut_ease={ease:g}: {a}->{b} runs {v:.0f} pos/s"
    jepa_infer.FPS, jepa_infer.DT = HZ, 1.0 / HZ
    print("cut slew: the seam ceiling clamps identically in both languages")


def check_short_shots():
    """A shot SHORTER than its smoothing kernel must not reach past its end.

    ``np.convolve(seg, box, mode="same")`` returns ``max(len(seg), len(box))``
    samples, so every per-shot smoother can derive indices that do not exist
    in the shot -- a 4-row shot crashed the panel outright
    (jepa_infer._extrema) and the stillness/speed smoothers went quietly
    wrong. Exercised across shot lengths that straddle every kernel width.
    """
    for hz in (HZ, ALT):
        jepa_infer.FPS, jepa_infer.DT = hz, 1.0 / hz
        t, p = sample(hz, dur_s=20.0)
        vel = np.gradient(p, t)
        n = len(p)
        for short in (1, 2, 3, 4, 6, 8, 14, 30):
            if n <= short + 2:
                continue
            # a final shot of exactly ``short`` rows, plus a normal one
            edges = [0, n - short, n]
            val = np.ones(n, dtype=bool)
            ti, _tw = jepa_infer._extrema(p, edges, val)
            assert not len(ti) or int(ti.max()) < n, \
                f"{hz:g} Hz: _extrema index {ti.max()} past {n} rows " \
                f"(shot of {short})"
            # the stillness/speed smoothers share the pattern
            sm = jepa_infer.still_metrics(p, vel, val) \
                if hasattr(jepa_infer, "still_metrics") else None
            assert sm is None or np.all(np.isfinite(np.asarray(sm, float)))
    jepa_infer.FPS, jepa_infer.DT = HZ, 1.0 / HZ
    print("short shots: per-shot smoothers stay inside their shot "
          "(kernel > shot)")


def check_metrics():
    got = {}
    for hz in (HZ, ALT):
        jepa_infer.FPS, jepa_infer.DT = hz, 1.0 / hz
        t, p = sample(hz)
        vel = np.gradient(p, 1.0 / hz)
        val = np.ones(len(p), dtype=bool)
        _prec, _rec, share = jepa_infer.still_metrics(p, vel, val)
        se = jepa_infer.speed_error(p, p, val)
        got[hz] = {"still_share": share, "fast_per_min": se[3]}
    jepa_infer.FPS, jepa_infer.DT = HZ, 1.0 / HZ
    for key in got[HZ]:
        a, b = got[HZ][key], got[ALT][key]
        assert abs(a - b) <= 0.15 * max(abs(a), abs(b), 1e-9) + 0.02, \
            f"{key} differs across grids: {a:.3f} vs {b:.3f}"
    print(f"metrics: stillness share {got[HZ]['still_share']:.3f}, fast "
          f"strokes/min {got[HZ]['fast_per_min']:.1f} -- grid-stable")


def check_speed_halves():
    """The slow-section spike splits into its styled and smooth halves
    exactly, and the styled window reaches STYLE_DILATE_S from a script's
    own fast stroke and no further. The script is a 4 s triangle whose
    speed ramps 24 to 36 u/s over 40 s, then a fast one, so the slow band
    is the first half of the ramp with a margin on both sides; one script
    spike at its 10 s trough makes a styled window around it. The draft is
    the same triangle 0.35 s out of phase, with the spike at two of its own
    troughs: 6.35 s, where the script is smooth, and 10.35 s, inside the
    window and past the script stroke's own smoothing reach. A spike at a
    trough is two one-row half-strokes; on a flank it would merge into the
    slow stroke it rides. A 10-unit script vertex at 20 s is scripter
    texture and opens no window."""
    import types
    from . import scoring
    n = int(60.0 * HZ)
    t = np.arange(n) / HZ
    ms = t * 1000.0

    def tri(t0, period):
        return 2.0 * np.abs(((t - t0) / period) % 1.0 - 0.5)

    slow_amp = 24.0 + 12.0 * t / 40.0
    fast = 20.0 + 80.0 * tri(0.0, 1.0 / 1.5)          # 240 u/s
    script = np.where(t < 40.0, 20.0 + 2.0 * slow_amp * tri(0.0, 4.0), fast)
    draft = np.where(t < 40.0, 20.0 + 2.0 * slow_amp * tri(0.35, 4.0), fast)
    script[int(10.0 * HZ)] += 40.0            # the scripter's own fast stroke
    script[int(20.0 * HZ)] += 10.0            # the scripter's texture
    draft[int(6.35 * HZ)] += 40.0             # on a smooth passage
    draft[int(10.35 * HZ)] += 40.0            # beside the script's stroke
    ref = scoring.SpeedReference(
        types.SimpleNamespace(t=ms, p=script, gaps=[]))
    at = np.searchsorted(ref.grid, [6350.0, 10350.0, 9300.0, 10700.0,
                                    20000.0])
    assert list(ref.styled[at]) == [False, True, False, False, False], \
        f"styled window at 6.35/10.35/9.3/10.7/20 s: {list(ref.styled[at])}"
    sp = scoring.score_speed(ms, draft, ref)
    for x in ("sl2x", "sl3x"):
        s, m = sp[f"{x}_styled"], sp[f"{x}_smooth"]
        assert abs(s + m - sp[x]) < 1e-9, f"{x}: {s} + {m} != {sp[x]}"
    ev = {k: round(sp[k] * ref.slow_min) for k in ("sl3x_styled",
                                                    "sl3x_smooth")}
    assert ev == {"sl3x_styled": 2, "sl3x_smooth": 2}, \
        f"spike halves counted {ev}, expected 2 half-strokes each"
    print(f"speed halves: styled {sp['sl3x_styled']:.2f} + smooth "
          f"{sp['sl3x_smooth']:.2f} = sl>3x {sp['sl3x']:.2f} per slow "
          f"minute, the window {scoring.STYLE_DILATE_S:g} s each side")


def check_graft():
    """The viterbi graft's structural guarantees: the refractory machine
    alternates and spaces (identity at min_gap=1), and a called reversal
    survives a sloping level all the way into the written actions."""
    n = 300
    jepa_infer.FPS, jepa_infer.DT = HZ, 1.0 / HZ
    rt = np.zeros(n)
    rb = np.zeros(n)
    rt[10:280:30] = 0.9
    rb[25:280:30] = 0.9
    r1, k1 = common.alternating_events(rt, rb, bias=0.5)
    r1b, k1b = common.alternating_events(rt, rb, bias=0.5, min_gap=1)
    assert np.array_equal(r1, r1b) and np.array_equal(k1, k1b), \
        "min_gap=1 is not the unconstrained machine"
    r4, k4 = common.alternating_events(rt, rb, bias=2.0, min_gap=4)
    assert (np.diff(r4) >= 4).all() and (k4[1:] * k4[:-1] == -1).all(), \
        "refractory spacing/alternation broken"
    # the refractory's exact reach, pinned: min_gap=g is g event-free rows
    # after an event, so a strong pair g rows apart loses one event and a
    # pair g + 1 apart keeps both. One row here is one decode.
    pk = np.full(8, 1e-3)
    pk[0] = 0.98
    for g in (2, 3):
        vl = np.full(8, 1e-3)
        vl[g] = 0.98
        n_at_g = len(common.alternating_events(pk, vl, min_gap=g)[0])
        vl[g], vl[g + 1] = 1e-3, 0.98
        n_past = len(common.alternating_events(pk, vl, min_gap=g)[0])
        assert (n_at_g, n_past) == (1, 2), (
            f"min_gap={g}: pair {g} apart -> {n_at_g} events, "
            f"{g + 1} apart -> {n_past} (expected 1, 2)")
    lvl = np.linspace(30.0, 70.0, n)          # the sum-monotonic trap
    env = np.full(n, 8.0)
    vm = np.where(np.arange(n) % 30 < 15, 40.0, -40.0)
    frc = []
    p = jepa_infer.style_positions_composed(
        vm, lvl, env, [0, n], rev=(rt, rb), rev_source="viterbi",
        force_out=frc)
    acts = jepa_infer.extrema_actions(p, np.arange(n) / HZ * 1000.0,
                                      [0, n], force=frc, rdp_eps=1.0)
    po = np.array([a["pos"] for a in acts], dtype=float)
    d = np.sign(np.diff(po))
    d = d[d != 0]
    revs = int(np.sum(d[1:] * d[:-1] < 0))
    assert len(frc) >= 15 and revs >= 15, \
        f"graft wrote {revs} of {len(frc)} called reversals"
    print(f"graft: refractory machine sound, {revs}/{len(frc)} called "
          "reversals written across a sloping level")


def main():
    check_constants()
    check_rust_mirror()
    check_crop_mirror()
    check_crop_fixture()
    check_embedded_files()
    check_row_clock()
    check_labels()
    check_actions()
    check_cut_ease_fixture()
    check_amp_bound_fixture()
    check_phase_runs_fixture()
    check_cut_slew_fixture()
    check_short_shots()
    check_metrics()
    check_speed_halves()
    check_graft()
    print(f"decode check PASSED at {HZ:g} rows/s")


if __name__ == "__main__":
    main()
