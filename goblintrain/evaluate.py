"""Eval: draft a roster with a model and read the written funscripts.

The model is a run of this project (``runs/<name>/model.pt``), a shipped
release (``v0.6.0``) or a checkpoint path. The drafts and their record
land in ``<project>/drafts/<model>-<roster>/``; a directory that already
holds every draft of the same checkpoint, scored on the same rows, is only
re-read, never re-drafted. A clip the model trained on is scored on the
rows training held out, a clip it never saw is scored whole, and a roster
that mixes the two is refused, because one read uses one scoring rule.
``--ref`` names another eval directory and prints it beside this one with
paired clip-bootstrap intervals.
"""
import shutil
import sys
import time
from pathlib import Path

import torch

from . import common, read, scoring
from .project import ProjectError, Tee


def resolve_model(project, spec):
    """``--run``: a run name of the project, a shipped release name, a run
    directory or a checkpoint path. Returns (checkpoint path, short name)."""
    p = Path(spec)
    if p.suffix == ".pt":
        if not p.is_file():
            raise ProjectError(f"no checkpoint at {p}")
        return p, p.stem
    if p.is_dir() and (p / "model.pt").is_file():     # a run directory
        return p / "model.pt", p.name
    run_model = project.root / "runs" / spec / "model.pt"
    if run_model.is_file():
        return run_model, spec
    release = common.WEIGHTS_DIR / "checkpoints" / f"{spec}.pt"
    if release.is_file():
        return release, spec
    if (project.root / "runs" / spec).is_dir():
        raise ProjectError(f"run {spec!r} has no model.pt yet; finish it with "
                           f"goblintrain train")
    raise ProjectError(f"{spec!r} is neither a run of this project nor a "
                       f"shipped release; releases come with: goblintrain "
                       f"fetch")


STAMP = "ckpt.sha256"


def bind_drafts(out, sha, log=print, why=None):
    """Keep ``out``'s drafts only if they came from the checkpoint ``sha``
    names (and ``why`` is None); else clear the directory so it is drafted
    again. A directory from before the stamp is kept as it is."""
    stamp = out / STAMP
    old = stamp.read_text("utf-8").strip() if stamp.is_file() else None
    if old is not None and old != sha:
        why = why or "its drafts come from another checkpoint"
    if why and out.is_dir():
        log(f"{out.name}: {why}; drafting again")
        shutil.rmtree(out)


def run(project, run_spec, roster="holdout", ref=None, name=None,
        rescore=False, log=print):
    """The ``eval`` command. Returns the exit code."""
    ids = common.load_roster(project.root, roster)
    if ids is None:
        raise ProjectError(f"no roster {roster!r} in the project; goblintrain "
                           f"prepare writes the holdout roster")
    if not ids:
        raise ProjectError(f"roster {roster!r} is empty")
    ckpt, label = resolve_model(project, run_spec)
    name = name or f"{label}-{roster}"
    out = project.root / "drafts" / name
    ref_dir = None
    if ref:
        ref_dir = project.root / "drafts" / ref
        if scoring.read_record(ref_dir) is None:
            raise ProjectError(f"--ref {ref}: no record at {ref_dir}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    trained = common.trained_ids(ck, project.root, ids)
    seen = [i for i in ids if i in trained]
    if seen and len(seen) < len(ids):
        raise ProjectError(
            f"roster {roster!r} mixes {len(seen)} clips the model trained on "
            f"with {len(ids) - len(seen)} it never saw; the two are scored "
            f"differently, so evaluate them as separate rosters")
    val_frac = None if seen else 1.0     # trained: the held-out rows only
    scored = 1.0 if val_frac else float(ck.get("val_frac", 0.15))
    del ck
    sha = common.ckpt_sha(ckpt)
    old = scoring.read_record(out)
    bind_drafts(out, sha, log, why=(
        f"its record uses val_frac {old.get('val_frac')}, this read uses "
        f"{scored:g}"
        if old and old.get("val_frac") not in (None, scored) else None))
    have = all(scoring.draft_path(out, v).exists() for v in ids)
    if have:
        log(f"{name}: every draft exists; reading them")
    else:
        from . import jepa_infer
        args = jepa_infer.build_parser().parse_args([])
        args.dataset = str(project.root)
        args.ids = [roster]
        args.ckpt = str(ckpt)
        args.out = str(out)
        args.val_frac = val_frac
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.device != "cuda":
            log("no CUDA device: this will be very slow")
        out.mkdir(parents=True, exist_ok=True)
        (out / STAMP).write_text(sha, encoding="utf-8")
        log(f"{name}: drafting {len(ids)} clips with {ckpt.name}"
            + (" on their held-out rows" if seen else " whole"))
        tee = Tee(out / "infer.log")
        sys.stdout = tee
        t0 = time.time()
        try:
            print(f"== {time.strftime('%Y-%m-%d %H:%M:%S')} eval {name}",
                  flush=True)
            jepa_infer.run(args)
        finally:
            sys.stdout = tee.out
            tee.close()
        log(f"drafted in {(time.time() - t0) / 60:.1f} min")
    rec = read.score_dir(out, out, ids, project.root, ref=ref_dir,
                         val_frac=val_frac, rescore=rescore, log=log)
    rec["_name"] = name
    records = [rec]
    if ref_dir is not None:
        r = scoring.read_record(ref_dir)
        r["_name"] = ref
        records.insert(0, r)
    read.table(records, ref=ref or name, log=log)
    if ref_dir is not None:
        read.bootstrap(records, ref, log=log)
    return 0
