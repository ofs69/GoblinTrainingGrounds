"""Draft: one video in, one funscript out.

The video is imported without a script into a drafting project (by
default ``drafts``, created on first use), the perception caches are
filled for it, the model drafts it, and the written
funscript is copied beside the video as ``<stem>.funscript``. A second
draft of the same file finds the clip again by its content and only
re-runs what is missing; a different model drafts into its own directory,
and a directory whose checkpoint has changed is drafted again.
"""
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from . import common, importer, prepare
from .evaluate import STAMP, bind_drafts, resolve_model
from .project import Project, ProjectError, Tee, named

DEFAULT_MODEL = "v0.6.0"


def run(videos, model=None, project="drafts", out=None, force=False,
        log=print):
    """The ``draft`` command. Returns the exit code."""
    missing = importer.media.have_tools()
    if missing:
        raise ProjectError(f"{' and '.join(missing)} not found on PATH; see "
                           "README.md, What you need")
    root = named(project)
    if not (root / "config.json").is_file():
        Project.create(root)
        log(f"created the drafting project {root}")
    with Project.open(root) as project:
        ckpt, label = resolve_model(project, model or DEFAULT_MODEL)
        clips = []
        for v in videos:
            clip_id, new = importer.import_video(project, v, log)
            clips.append((Path(v), clip_id))
        ids = [c for _, c in clips]
        prepare.caches(project, ids, log)
        out_dir = project.root / "drafts" / label
        sha = common.ckpt_sha(ckpt)
        bind_drafts(out_dir, sha, log)
        need = [c for c in ids
                if not (out_dir / f"{c}_jepa.funscript").is_file()]
        if need:
            from . import jepa_infer
            args = jepa_infer.build_parser().parse_args([])
            args.dataset = str(project.root)
            args.ids = need
            args.ckpt = str(ckpt)
            args.out = str(out_dir)
            args.val_frac = 1.0
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / STAMP).write_text(sha, encoding="utf-8")
            log(f"drafting {len(need)} clip{'s' if len(need) != 1 else ''} "
                f"with {ckpt.name}")
            tee = Tee(out_dir / "infer.log")
            sys.stdout = tee
            t0 = time.time()
            try:
                print(f"== {time.strftime('%Y-%m-%d %H:%M:%S')} draft "
                      f"{' '.join(need)}", flush=True)
                jepa_infer.run(args)
            finally:
                sys.stdout = tee.out
                tee.close()
            log(f"drafted in {(time.time() - t0) / 60:.1f} min")
        rc = 0
        for video, clip_id in clips:
            src = out_dir / f"{clip_id}_jepa.funscript"
            dst = (Path(out) / f"{video.stem}.funscript" if out
                   else video.with_suffix(".funscript"))
            if dst.exists() and not force:
                log(f"  [{clip_id}] {dst.name} exists beside the video; "
                    f"--force overwrites it")
                rc = 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            log(f"  [{clip_id}] wrote {dst}")
        return rc
