"""Export: a model into a GoblinScript bundle as a pack, then parity.

A bundle holds one frozen perception (the encoder and TransNet graphs)
and any number of packs, each one model's graphs and decode constants
under ``packs/<name>/``. The first export into a directory writes the
whole bundle; a later one adds its pack to it after checking the
perception matches. The bundle lives at ``<project>/bundle`` unless
``--bundle`` says otherwise. Parity then runs the exported graphs over a
prepared clip of the project against the Python pipeline's own latents
and tracks, and with ``--rust`` hands the shipped binary the same rows.
"""
from pathlib import Path

from . import common
from .evaluate import resolve_model
from .project import ProjectError


def parity_clip(project, clip_id=None):
    """The clip the parity check runs on: the one asked for, else the first
    prepared clip of the gate roster, then of the holdout roster, then of
    the project."""
    from . import extract
    if clip_id:
        if not extract.path(project, clip_id).is_file():
            raise ProjectError(f"{clip_id} has no latents; goblintrain "
                               f"prepare {project.root.name} {clip_id}")
        return clip_id
    ids = ((common.load_roster(project.root, "gate") or [])
           + (common.load_roster(project.root, "holdout") or []))
    ids += [r["id"] for r in project.manifest() if project.is_scripted(r)]
    for i in ids:
        if extract.path(project, i).is_file():
            return i
    raise ProjectError("no prepared clip to run parity on; goblintrain "
                       "prepare first, or --no-parity")


def run(project, run_spec, pack=None, label="", bundle=None, clip_id=None,
        rust=False, exe=None, parity=True, log=print):
    """The ``export`` command. Returns the exit code."""
    from . import export_bundle
    from . import parity as parity_mod
    ckpt, name = resolve_model(project, run_spec)
    pack = pack or name
    bundle = Path(bundle) if bundle else project.root / "bundle"
    args = export_bundle.build_parser().parse_args([])
    args.ckpt = str(ckpt)
    args.pack = pack
    args.label = label
    if (bundle / "manifest.json").is_file():
        args.into = str(bundle)
        log(f"adding pack {pack} to the bundle at {bundle}")
    else:
        args.out = str(bundle)
        log(f"writing a new bundle at {bundle} with pack {pack}")
    export_bundle.run(args)
    if not parity:
        return 0
    cid = parity_clip(project, clip_id)
    pargs = parity_mod.build_parser().parse_args([])
    pargs.dataset = str(project.root)
    pargs.id = cid
    pargs.bundle = str(bundle)
    pargs.pack = pack
    pargs.ckpt = str(ckpt)
    pargs.rust = rust
    pargs.exe = exe
    log(f"parity of pack {pack} on clip {cid}")
    ok = parity_mod.run(pargs)
    log("parity PASSED" if ok else "parity FAILED")
    return 0 if ok else 1
