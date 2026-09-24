"""Prepare: everything training needs from a clip, clip by clip, only what
is missing.

Three stages, each a cache the clip either has or does not: shot
boundaries (TransNetV2), latents (the frozen perception pass) and the
script-to-video lag fit against the released model. Every stage writes its
file whole or not at all, so the command can be stopped at any time and
run again. A clip without a script gets no lag fit.

After the caches, admission: a scripted clip joins the training roster when
its lag fit is confident (peak at least ``ADMIT_PEAK``), the fit does not
suspect inverted polarity, and no drift alarm spreads past one reversal
tolerance. ``rosters/holdout.json`` is drawn once from the admitted clips
and never rewritten; ``rosters/train.json`` is the admitted clips outside
it and is rewritten on every run.
"""
import json
import time

import torch

from . import boundaries, common, extract, lagfit
from .jepa_train import load_model
from .project import ProjectError, atomic_write_text

ADMIT_PEAK = 0.55
ADMIT_DRIFT_MS = 1000.0 * common.REV_TOL_S     # 66.7 ms: one reversal tolerance
HOLDOUT_EVERY = 8                              # one admitted clip in eight


def admitted(side):
    """Why a clip with this lag sidecar is not admitted, or None."""
    if side["peak"] < ADMIT_PEAK:
        return f"lag peak {side['peak']:.2f} below {ADMIT_PEAK}"
    if side.get("polarity_suspect"):
        return "polarity suspect (the script may be inverted)"
    if side.get("drift_suspect"):
        spread = side.get("drift_spread_ms")
        if spread is None or spread > ADMIT_DRIFT_MS:
            return "drift alarm (no one offset describes the clip)"
    return None


def write_rosters(project, admitted_ids, log):
    """The training roster from the admitted clips, minus a holdout drawn
    once."""
    hold = common.load_roster(project.root, "holdout")
    if hold is None:
        hold = sorted(admitted_ids)[HOLDOUT_EVERY - 1::HOLDOUT_EVERY]
        atomic_write_text(common.roster_path(project.root, "holdout"),
                          json.dumps({"note": "held out of training for "
                                      "evaluation; drawn once by prepare",
                                      "ids": hold}, indent=0) + "\n")
        log(f"holdout roster written: {len(hold)} clips")
    train = [i for i in sorted(admitted_ids) if i not in set(hold)]
    atomic_write_text(common.roster_path(project.root, "train"),
                      json.dumps({"note": "admitted clips outside the "
                                  "holdout; rewritten by prepare",
                                  "ids": train}, indent=0) + "\n")
    return train, hold


def caches(project, ids, log=print, ckpt=None, lag=True):
    """Boundaries, latents and (for scripted clips, when ``lag``) the lag
    fit of every clip in ``ids`` that lacks them."""
    recs = {r["id"]: r for r in project.manifest()}
    scripted = [i for i in ids if lag and project.is_scripted(recs[i])]
    need_b = [i for i in ids if not boundaries.path(project, i).is_file()]
    need_l = [i for i in ids if not extract.path(project, i).is_file()]
    need_g = [i for i in scripted if lagfit.load(project, i) is None]
    log(f"{len(ids)} clips: {len(need_b)} need boundaries, {len(need_l)} "
        f"need latents, {len(need_g)} need a lag fit")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (need_b or need_l or need_g) and device != "cuda":
        log("no CUDA device: this will be very slow")

    if need_b:
        t0 = time.time()
        model = boundaries.load_transnet(device)
        for n, i in enumerate(need_b, 1):
            boundaries.ensure(project, i, model, device, log)
        del model
        torch.cuda.empty_cache()
        log(f"boundaries: {len(need_b)} written in {(time.time() - t0) / 60:.1f} min")

    if need_l:
        t0 = time.time()
        pc = project.config["perception"]
        enc = extract.load_encoder(pc, device)
        basis = extract.Basis(pc, device)
        for i in need_l:
            extract.ensure(project, i, enc, basis, device, log)
        del enc, basis
        torch.cuda.empty_cache()
        log(f"latents: {len(need_l)} written in {(time.time() - t0) / 60:.1f} min")

    if need_g:
        t0 = time.time()
        ckpt = ckpt or common.DEFAULT_CKPT
        model, ck = load_model(ckpt, device)
        name = ckpt.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        for i in need_g:
            lagfit.ensure(project, i, model, ck, name, device, log)
        del model
        torch.cuda.empty_cache()
        log(f"lag fits: {len(need_g)} written in {(time.time() - t0) / 60:.1f} min")


def run(project, ids=None, log=print, ckpt=None):
    """The ``prepare`` command over ``ids`` (default: every clip). Returns
    the exit code."""
    recs = {r["id"]: r for r in project.manifest()}
    if ids:
        ids = common.resolve_ids(ids, project.root)
        unknown = [i for i in ids if i not in recs]
        if unknown:
            raise ProjectError(f"not in the project: {' '.join(unknown[:5])}")
    else:
        ids = sorted(recs)
    if not ids:
        log("nothing to prepare: the project has no clips")
        return 0
    caches(project, ids, log, ckpt)

    # admission over every scripted clip that has a lag fit, in the project
    admit, refused = [], []
    for i in sorted(r for r in recs if project.is_scripted(recs[r])):
        side = lagfit.load(project, i)
        if side is None or not extract.path(project, i).is_file():
            continue
        why = admitted(side)
        (refused if why else admit).append((i, why))
    for i, why in refused:
        if i in set(ids):
            log(f"  [{i}] not admitted: {why}")
    train, hold = write_rosters(project, [i for i, _ in admit], log)
    log(f"admitted {len(admit)} of {len(admit) + len(refused)} prepared clips; "
        f"train roster {len(train)}, holdout {len(hold)}")
    return 0
