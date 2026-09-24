"""The deploy read: one set of drafts in, one ranked table out.

Screen and val rankings have been wrong about deploy three times, so a
model is judged on WRITTEN drafts at the deploy decode: jepa_infer drafts
a roster, ``scoring.py`` reads every draft against its script, and the
directory's record (``metrics.json``) carries the per-clip reads and the
pooled block every table here prints from. Reads are pure: a finished
directory is re-read from disk without touching the GPU, and a directory
of drafts from anywhere (a goblinscript bench, a stored release) is read
the same way.

Four things make this the read every model is ranked through:

* **Both tolerances, recall beside precision.** The 66.7 ms window and
  the 333 ms one answer different questions, and a model can win the
  tight mean by not finding the marginal reversals that would have
  dragged it up. A gain bought by inserted strokes is not a gain.
* **The worst clips beside the mean.** Every pooled number carries its
  worst-2 companion: a mean never cancels a tail.
* **The artifact floor at its own pooling.** The slow-section spike rate
  is per minute of slow script time, so it pools by slow minute, with
  delivery beside it: a model that writes a third fewer fast strokes
  collects a third fewer spikes without touching the mechanism that
  makes one.
* **The seed floor.** ``floor`` prints k replicates of one recipe with
  the RANGE and SD under them; a delta inside that band is the draw.

A clip the model never trained on is scored whole; a clip it trained on
is scored on the rows training held out, which the checkpoint stamps.
``goblintrain eval`` drafts and reads one model; this module's own CLI
prints tables over directories that already carry a record:

    python -m goblintrain.read table <dir> <dir>... [--ref <dir>] [--bootstrap]
    python -m goblintrain.read floor <dir> <dir>...
"""
import argparse
import math
import re
from pathlib import Path


from . import common
from . import scoring

NAN = float("nan")

# the seed-floor columns, in the order a candidate is read: position,
# then the clock, then what the draft delivered
FLOOR_COLS = (("corr", 7, 4), ("corrT2", 7, 4), ("kappa", 7, 4),
              ("kapT2", 7, 4),
              ("rec67", 6, 3), ("prec67", 7, 3), ("|dt|67", 7, 1),
              ("fast/min", 9, 2), ("sl>3x", 6, 2))


def clock_for(vid, rec, ref_rec, project, val_frac):
    """The clip's row clock: the directory's own stamp, else the reference
    directory's (same clips, same grid), else the cache, whole clip or at
    ``val_frac``; a record-less read at the checkpoint's own split has no
    clock to honour."""
    for r in (rec, ref_rec):
        c = ((r or {}).get("clips") or {}).get(vid, {}).get("clock")
        if c:
            return scoring.Clock.from_stamp(c)
    if val_frac is None:
        raise SystemExit(f"[{vid}] no stamped clock for a read at the "
                         f"checkpoint's own split; draft through "
                         f"goblintrain eval, or name a --ref that did")
    return scoring.Clock.from_cache(project, vid, common.LATENTS_DIR,
                                    float(val_frac))


def score_dir(out, drafts, ids, project, ref=None, val_frac=1.0,
              with_gaps=False, lag_field="applied_ms", rescore=False,
              log=print):
    """Read every draft of ``ids`` in ``drafts`` and write ``out``'s record
    with its pooled block. Existing artifact reads are kept unless
    ``rescore``."""
    out = Path(out)
    rec = scoring.read_record(out) or {}
    ref_rec = scoring.read_record(ref) if ref else None
    clips = rec.setdefault("clips", {})
    rec.setdefault("dataset", str(project))
    rec["ids"] = list(ids)
    rec["drafts"] = str(drafts)
    rec["lag"] = lag_field
    rec["with_gaps"] = with_gaps
    speed_clamp = bool(rec.get("speed_clamp",
                               (ref_rec or {}).get("speed_clamp", True)))
    rec["speed_clamp"] = speed_clamp
    n_new = 0
    for vid in ids:
        crec = clips.setdefault(vid, {})
        if crec.get("artifact") and not rescore:
            continue
        path = scoring.draft_path(drafts, vid)
        if not path.exists():
            log(f"  [{vid}] no draft at {path}")
            continue
        script = scoring.load_script(project, vid, lag_field,
                                     speed_clamp=speed_clamp)
        if script is None:
            log(f"  [{vid}] no script -- skipped")
            continue
        clock = clock_for(vid, rec, ref_rec, project, val_frac)
        art = scoring.score_artifact(script, scoring.load_actions(path),
                                     clock, with_gaps=with_gaps)
        if art is None:
            log(f"  [{vid}] too short / empty")
            continue
        crec["clock"] = clock.stamp()
        crec["artifact"] = art
        n_new += 1
    rec["pooled"] = scoring.pool_arm(clips, ids,
                                     style=rec.get("style", "composed"))
    scoring.write_record(out, rec)
    log(f"  {out.name}: {n_new} clips read, {rec['pooled']['n']} in the "
        f"pooled block -> {scoring.record_path(out)}")
    return rec


# ------------------------------------------------------------------ tables

def _num(v, w=6, p=3):
    return f"{v:{w}.{p}f}" if v is not None and math.isfinite(v) \
        else f"{'--':>{w}}"


def _pair(d, k, kt, w=6, p=3):
    return f"{_num(d.get(k), w, p)}/{_num(d.get(kt), w, p)}"


def table(records, ref=None, log=print):
    """The tables every arm is ranked through, one line per arm. Nothing
    is quoted without its tail or its guard."""
    recs = [(r.get("arm") or r.get("_name"), r["pooled"]) for r in records
            if r.get("pooled")]
    if not recs:
        log("nothing to print: no pooled block in any record")
        return
    ids = set(tuple(p.get("ids", [])) for _, p in recs)
    if len(ids) > 1:
        log("NOTE: the arms were pooled over different clip sets -- the "
            "rows are not comparable")

    log("\n=== precision-first: reversal precision at both tolerances and "
        "the matched |dt| p50/p95, each pooled/worst clip; recall behind ===")
    log(f"{'arm':>20} {'prec67':>13} {'prec333':>13} {'p50|dt|':>8} "
        f"{'p95|dt|':>13} {'rec67':>6} {'rec333':>7}")
    for name, p in recs:
        t, w = p["timing"].get("67", {}), p["timing"].get("333", {})
        log(f"{name:>20} {_pair(t, 'prec', 'prec_tail')} "
            f"{_pair(w, 'prec', 'prec_tail')} {_num(t.get('p50'), 8, 1)} "
            f"{_pair(t, 'p95', 'p95_tail', 6, 1)} {_num(t.get('recall'))} "
            f"{_num(w.get('recall'), 7)}")

    log("\n=== timing: both tolerances, recall beside each; slow-band "
        "recall (script <1 Hz) per reversal ===")
    log(f"{'arm':>20} {'|dt|67':>7} {'rec67':>6} {'prec67':>7} "
        f"{'|dt|333':>8} {'rec333':>7} {'prec333':>8} {'slow67':>7}")
    for name, p in recs:
        t, w = p["timing"].get("67", {}), p["timing"].get("333", {})
        log(f"{name:>20} {_num(t.get('dt_ms'), 7, 1)} "
            f"{_num(t.get('recall'))} {_num(t.get('prec'), 7)} "
            f"{_num(w.get('dt_ms'), 8, 1)} {_num(w.get('recall'), 7)} "
            f"{_num(w.get('prec'), 8)} "
            f"{_num(t.get('slow', {}).get('recall_pooled'), 7)}")

    n = recs[0][1]["n"]
    log(f"\n=== position: pooled/worst-2 over all {n} clips ===")
    log(f"{'arm':>20} {'corr':>13} {'kappa':>13} {'mae':>13}")
    for name, p in recs:
        pos = p.get("position", {})

        def pair(k, w=6, pr=4):
            v = pos.get(k)
            if not v:
                return f"{'--':>{2 * w + 1}}"
            return f"{_num(v['pooled'], w, pr)}/{_num(v['tail'], w, pr)}"
        log(f"{name:>20} {pair('corr')} {pair('kappa')} {pair('mae', pr=2)}")

    log("\n=== product panel (pooled/worst-2; travel is the draft's, the "
        "rest the composed track's) beside the spike guard (per = >2x "
        "events per fast stroke DELIVERED) ===")
    log(f"{'arm':>20} {'travel':>13} {'bandHi':>13} {'stillR':>13} "
        f"{'maeExt':>13} {'fast':>6} {'amp+':>6} {'per':>6} "
        f"{'stub%':>6} {'brok':>6} {'spdW':>11} {'gapW':>11}")
    for name, p in recs:
        pan, sp = p.get("panel", {}), p.get("speed", {})
        spt = p.get("speed_tail", {})
        tv = p.get("position", {}).get("travel", {})

        def pp(k, w=6, pr=3):
            v = pan.get(k)
            if not v:
                return f"{'--':>{2 * w + 1}}"
            return f"{_num(v['pooled'], w, pr)}/{_num(v['tail'], w, pr)}"
        log(f"{name:>20} {_num(tv.get('pooled'), 6, 3)}/"
            f"{_num(tv.get('tail'), 6, 3)} {pp('bandHi')} {pp('stillR')} "
            f"{pp('maeExt', pr=2)} "
            + " ".join(_num(sp.get(c), 6, 2) for c in scoring.GUARD_COLS)
            + " " + " ".join(f"{_num(sp.get(c), 5, 3)}/{_num(spt.get(c), 5, 3)}"
                             for c in scoring.SHAPE_COLS))

    anchor = next((p for nm, p in recs if nm == ref), recs[0][1])
    ra = anchor.get("artifact", {})
    log(f"\n=== artifact floor: slow-section rates per SLOW MINUTE, "
        f"delivery beside it. pred>3x = the anchor's "
        f"rate scaled by the delivery ratio (pure suppression). styled/smooth "
        f"= sl>3x split by whether the SCRIPT carries a fast stroke within "
        f"{scoring.STYLE_DILATE_S:g} s, smT2 the smooth half's worst-2 ===")
    log(f"{'arm':>20} {'sl>3x':>6} {'styled':>6} {'smooth':>6} {'smT2':>6} "
        f"{'sl>2x':>6} {'slpeak':>7} "
        f"{'fast/min':>9} {'x/SCRIPT':>8} {'>2x/str':>8} "
        f"{'>3x/str':>8} {'pred>3x':>8}")
    for name, p in recs:
        a = p.get("artifact", {})
        pred = (ra["sl3x"] * a["fast"] / ra["fast"]
                if ra.get("fast") and a.get("fast") else NAN)
        log(f"{name:>20} {_num(a.get('sl3x'), 6, 2)} "
            f"{_num(a.get('sl3x_styled'), 6, 2)} "
            f"{_num(a.get('sl3x_smooth'), 6, 2)} "
            f"{_num(a.get('sl3x_smooth_tail'), 6, 2)} "
            f"{_num(a.get('sl2x'), 6, 2)} "
            f"{_num(a.get('slpeak'), 7, 0)} "
            f"{_num(a.get('fast'), 9, 2)} "
            f"{_num(a.get('x_script'), 8, 3)} "
            f"{_num(a.get('x2_per_stroke'), 8, 4)} "
            f"{_num(a.get('x3_per_stroke'), 8, 4)} {_num(pred, 8, 2)}")


def bootstrap(records, ref, log=print, resamples=20000, seed=888):
    """Paired clip-bootstrap of every arm against ``ref`` on the ranking
    columns: the mean delta over clips, its 95% interval and P(>0). A
    delta whose interval spans zero is clip-sampling noise. Over CLIPS,
    never rows."""
    by = {r.get("arm") or r.get("_name"): r for r in records}
    if ref not in by:
        log(f"bootstrap: no record for the reference {ref}")
        return
    base = by[ref]["clips"]
    cols = (("position", "corr"), ("position", "kappa"), ("position", "mae"),
            ("position", "travel"), ("timing", "67", "recall"),
            ("timing", "67", "prec"), ("timing", "67", "dt_ms"),
            ("speed", "sl3x"), ("speed", "fast"), ("speed", "spdW"),
            ("speed", "gapW"))

    def pick(art, col):
        d = art
        for k in col:
            d = d.get(k) if isinstance(d, dict) else None
        return d if isinstance(d, (int, float)) else NAN
    log(f"\n=== paired clip-bootstrap vs {ref}: mean delta [95% CI] "
        f"P(>0), {resamples} resamples ===")
    for name, r in by.items():
        if name == ref:
            continue
        shared = [v for v in r["clips"] if v in base
                  and r["clips"][v].get("artifact")
                  and base[v].get("artifact")]
        bits = []
        for col in cols:
            d = [pick(r["clips"][v]["artifact"], col)
                 - pick(base[v]["artifact"], col) for v in shared]
            b = scoring.clip_bootstrap(d, resamples, seed)
            lab = "/".join(col[1:]) if col[0] != "position" else col[1]
            bits.append(f"{lab} {b['mean']:+.4f} [{b['lo']:+.4f},"
                        f"{b['hi']:+.4f}] {b['p_gt0']:.2f}")
        log(f"{name:>20} (n={len(shared)})\n    " + "\n    ".join(bits))


# ------------------------------------------------------------------- floor

def floor_cols(p):
    pos, t = p.get("position", {}), p.get("timing", {}).get("67", {})

    def g(k, sub):
        return pos.get(k, {}).get(sub, NAN)
    return {"corr": g("corr", "pooled"), "corrT2": g("corr", "tail"),
            "kappa": g("kappa", "pooled"), "kapT2": g("kappa", "tail"),
            "rec67": t.get("recall", NAN),
            "prec67": t.get("prec", NAN),
            "|dt|67": t.get("dt_ms", NAN),
            "fast/min": p.get("speed", {}).get("fast", NAN),
            "sl>3x": p.get("artifact", {}).get("sl3x", NAN)}


def read_loss(run_dir):
    """The training column from the run's log: the epoch line's own loss
    values. Two spikes are not a rate, so the COUNTS are quoted."""
    f = Path(run_dir) / "train.log"
    if not f.exists():
        return {}
    vals = [float(m.group(1)) for m in re.finditer(
        r"^ep\s+\d+/\d+ loss ([\d.]+)", f.read_text(
            encoding="utf-8", errors="replace"), re.M)]
    if not vals:
        return {}
    return {"epochs": len(vals), "max": max(vals), "vals": vals,
            "over4": sum(1 for v in vals if v > 4.0),
            "median": sorted(vals)[len(vals) // 2]}


def spread(vals):
    v = [x for x in vals if x is not None and math.isfinite(x)]
    if len(v) < 2:
        return NAN, NAN
    mean = sum(v) / len(v)
    return max(v) - min(v), math.sqrt(sum((x - mean) ** 2 for x in v)
                                      / (len(v) - 1))


def floor(records, runs=(), log=print):
    """k replicates of one recipe -> one spread per column. A delta inside
    RANGE is the draw."""
    rows = [(r.get("arm") or r.get("_name"), floor_cols(r["pooled"]))
            for r in records if r.get("pooled")]
    n = rows[0][1] and records[0]["pooled"]["n"]
    log(f"\n{len(rows)} replicates ({n} clips). corr/kappa pooled and "
        f"worst-2; fast/min at the\n"
        f"panel's equal-weight pooling; sl>3x per SLOW MINUTE, the pooling "
        f"the artifact ceiling is named in. Nothing here is a verdict --\n"
        f"it is the band every verdict is quoted against.\n")
    head = f"{'arm':>16}" + "".join(f" {c:>{w}}" for c, w, _ in FLOOR_COLS)
    log(head)
    for name, r in rows:
        log(f"{name:>16}" + "".join(
            " " + _num(r[c], w, p) for c, w, p in FLOOR_COLS))
    log("-" * len(head))
    for lab, idx in (("RANGE", 0), ("SD", 1)):
        log(f"{lab:>16}" + "".join(
            " " + _num(spread([r[c] for _, r in rows])[idx], w, p + 1)
            for c, w, p in FLOOR_COLS))
    log("\nA delta inside RANGE is the draw. Quote every number from here "
        "as `value, floor`.")
    if runs:
        loss = [(rd, read_loss(rd)) for rd in runs]
        ref = next((L["max"] for _, L in loss if L), None)
        log("\n=== the loss column, per replicate: an epoch mean over "
            "batch-size-1 steps, so a single window can carry it. >ref "
            f"counts epochs above {runs[0]}'s own max"
            + (f" ({ref:.3f})" if ref is not None else "") + " ===")
        log(f"{'run':>22} {'epochs':>7} {'median':>7} {'max':>8} "
            f"{'>4':>4} {'>ref':>5}")
        for rd, L in loss:
            if not L:
                log(f"{rd:>22} {'-- no train.log to read':>30}")
                continue
            over = (sum(1 for v in L["vals"] if v > ref)
                    if ref is not None else 0)
            log(f"{rd:>22} {L['epochs']:7d} {L['median']:7.3f} "
                f"{L['max']:8.3f} {L['over4']:4d} {over:5d}")


# -------------------------------------------------------------------- main

def records(dirs):
    """The records of draft directories, each labelled by its name."""
    out = []
    for d in dirs:
        r = scoring.read_record(d)
        if r is None:
            print(f"  {d}: no record (draft or read it first)")
            continue
        r["_name"] = Path(d).name
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("table", help="print the tables for draft directories")
    t.add_argument("dirs", nargs="+", help="directories carrying metrics.json")
    t.add_argument("--ref", default=None,
                   help="the directory anchoring pred>3x and the bootstrap "
                        "(default: the first)")
    t.add_argument("--bootstrap", action="store_true",
                   help="paired clip-bootstrap of every directory against "
                        "--ref")
    f = sub.add_parser("floor", help="the seed floor over replicates")
    f.add_argument("dirs", nargs="+",
                   help="one draft directory per replicate, seed the only "
                        "field the recipes differ in")
    f.add_argument("--runs", nargs="*", default=(),
                   help="the replicates' run directories, for the loss "
                        "column")
    args = ap.parse_args()
    recs = records(args.dirs)
    if args.mode == "table":
        ref = Path(args.ref or args.dirs[0]).name
        table(recs, ref=ref)
        if args.bootstrap:
            bootstrap(recs, ref)
    else:
        floor(recs, runs=args.runs)


if __name__ == "__main__":
    main()
