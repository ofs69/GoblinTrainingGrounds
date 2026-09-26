"""Train: the deploy heads on a shipped trunk, or a fresh trunk and its
heads, from a recipe.

A recipe is a JSON file with a ``trunk`` section (the trunk trainer's
settings) and a ``heads`` section (the head refit's settings); every key
is one of the stage's flags and everything the recipe leaves out keeps
the flag's default. ``--from <release>`` skips the trunk stage and refits
the heads on the shipped trunk, which is the fine-tuning path: the
released model's trunk, the released recipe's heads, the project's own
clips. ``--init-from <release>`` runs the trunk stage from that release's
weights instead of a random init and then refits the heads: the trunk
fine-tuning path. ``--recipe`` alone trains the trunk first.

A run is ``<project>/runs/<name>``. It holds the effective recipe, the
log, the trunk stage under ``trunk/`` and the finished model as
``model.pt``. Running the same command again resumes an unfinished run at
its last epoch and does nothing to a finished one.
"""
import json
import sys
import time
from pathlib import Path

import torch

from . import common
from .project import ProjectError, Tee, atomic_write_text



def load_recipe(path):
    p = Path(path)
    if not p.is_file():
        raise ProjectError(f"no recipe at {p}")
    rec = json.loads(p.read_text("utf-8"))
    for key in ("name", "trunk", "heads"):
        if key not in rec:
            raise ProjectError(f"{p} is not a recipe: it lacks {key!r}")
    return rec


def apply_sets(recipe, sets):
    """``--set section.key=value`` overrides, values read as JSON."""
    for spec in sets or ():
        try:
            target, value = spec.split("=", 1)
            section, key = target.split(".", 1)
        except ValueError:
            raise ProjectError(f"--set takes SECTION.KEY=VALUE, not {spec!r}")
        if section not in ("trunk", "heads"):
            raise ProjectError(f"--set {spec!r}: the section is trunk or heads")
        try:
            value = json.loads(value)
        except ValueError:
            pass
        recipe[section][key] = value


def _namespace(parser, section, values):
    """The stage's defaults with the recipe section applied; a key that is
    not one of the stage's flags is an error, not a silent no-op."""
    ns = parser.parse_args([])
    unknown = [k for k in values if not hasattr(ns, k)]
    if unknown:
        raise ProjectError(f"the {section} section names no such setting: "
                           + ", ".join(unknown))
    for k, v in values.items():
        setattr(ns, k, v)
    return ns


def resolve_from(spec):
    """``--from``: a release name (its shipped bare trunk) or a checkpoint
    path. Returns (trunk path, short name)."""
    p = Path(spec)
    if p.suffix == ".pt" or p.exists():
        if not p.is_file():
            raise ProjectError(f"no checkpoint at {p}")
        return p, p.stem
    trunk = common.WEIGHTS_DIR / "checkpoints" / f"{spec}-trunk.pt"
    if not trunk.is_file():
        raise ProjectError(f"no shipped trunk for {spec!r} at {trunk}; run: "
                           f"goblintrain fetch")
    return trunk, spec


def check_lengths(project, ids, win_s, roster, log):
    """Refuse a roster whose clips are all shorter than the training window
    (nothing would train); name the ones that are, when only some are."""
    from .importer import hms
    dur = {r["id"]: r.get("duration_ms") or 0 for r in project.manifest()}
    if not ids:
        raise ProjectError(f"roster {roster!r} is empty; goblintrain "
                           f"prepare admits clips into it")
    short = [i for i in ids if dur.get(i, 0) < 1000.0 * win_s]
    need = hms(1000.0 * win_s)
    if len(short) == len(ids):
        raise ProjectError(
            f"every clip of roster {roster!r} is shorter than the recipe's "
            f"training window ({need}), so nothing would train. Import "
            f"clips of at least {need}")
    if short:
        log(f"{len(short)} of {len(ids)} clips are shorter than the "
            f"training window ({need}) and do not train: "
            f"{' '.join(short[:10])}{' ...' if len(short) > 10 else ''}")


def run(project, recipe=None, from_=None, init_from=None, roster="train",
        name=None, sets=(), log=print):
    """The ``train`` command. Returns the exit code."""
    if from_ is None and init_from is None and recipe is None:
        raise ProjectError("give --from <release> to fit the deploy heads on "
                           "a shipped trunk, --init-from <release> to "
                           "fine-tune the trunk too, or --recipe <file> to "
                           "train a fresh trunk")
    if from_ is not None and init_from is not None:
        raise ProjectError("--from skips the trunk stage and --init-from "
                           "trains it from a shipped trunk; give one or the "
                           "other")
    trunk_ckpt = from_name = None
    if from_ is not None:
        trunk_ckpt, from_name = resolve_from(from_)
    init_ckpt = init_name = None
    if init_from is not None:
        init_ckpt, init_name = resolve_from(init_from)
    if recipe is None:
        recipe = common.RECIPES_DIR / f"{from_name or init_name}.json"
        if not recipe.is_file():
            recipe = common.RECIPES_DIR / f"{common.DEFAULT_RECIPE}.json"
    rec = load_recipe(recipe)
    apply_sets(rec, sets)
    if init_ckpt is not None:
        rec["trunk"]["init_from"] = str(init_ckpt)
    ids = common.load_roster(project.root, roster)
    if ids is None:
        raise ProjectError(f"no roster {roster!r} in the project; "
                           f"goblintrain prepare writes the train roster")
    check_lengths(project, ids, common.recipe_window_s(
        rec, trunk=trunk_ckpt is None), roster, log)
    effective = {"name": rec["name"], "description": rec.get("description", ""),
                 "from": None if trunk_ckpt is None else trunk_ckpt.name,
                 "roster": roster,
                 "trunk": None if trunk_ckpt is not None else rec["trunk"],
                 "heads": rec["heads"]}
    if name is None:
        name = (f"{from_name}-heads" if trunk_ckpt is not None else
                f"{init_name}-ft" if init_ckpt is not None else rec["name"])
    run_dir = project.root / "runs" / name
    rec_path = run_dir / "recipe.json"
    if (run_dir / "model.pt").is_file():
        log(f"run {name} is complete: {run_dir / 'model.pt'}")
        return 0
    if rec_path.is_file():
        if json.loads(rec_path.read_text("utf-8")) != effective:
            raise ProjectError(f"run {name} exists under another recipe; "
                               f"pick another --name")
        log(f"resuming run {name}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(rec_path, json.dumps(effective, indent=1) + "\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        log("no CUDA device: this will be very slow")

    # the stages import torch-heavy modules; only now, so the checks above
    # answer fast
    from . import jepa_refit, jepa_train

    tee = Tee(run_dir / "train.log")
    sys.stdout = tee
    try:
        print(f"== {time.strftime('%Y-%m-%d %H:%M:%S')} train {name}: "
              f"recipe {rec['name']}, roster {roster}"
              + (f", trunk {trunk_ckpt.name}" if trunk_ckpt else "")
              + (f", trunk init {init_ckpt.name}" if init_ckpt else ""),
              flush=True)
        if trunk_ckpt is None:
            trunk_dir = run_dir / "trunk"
            trunk_ckpt = trunk_dir / "jepa_best.pt"
            if (trunk_dir / "complete.json").is_file():
                print(f"trunk stage complete: {trunk_ckpt}", flush=True)
            else:
                ns = _namespace(jepa_train.build_parser(), "trunk",
                                effective["trunk"])
                ns.dataset = str(project.root)
                ns.ids = [roster]
                ns.runs_dir = str(trunk_dir)
                ns.resume = (trunk_dir / "resume_state.pt").is_file()
                t0 = time.time()
                jepa_train.run(ns)
                atomic_write_text(trunk_dir / "complete.json", json.dumps(
                    {"minutes": round((time.time() - t0) / 60, 1)}) + "\n")
        ns = _namespace(jepa_refit.build_parser(), "heads", effective["heads"])
        ns.dataset = str(project.root)
        ns.ids = [roster]
        ns.ckpt = str(trunk_ckpt)
        ns.out = str(run_dir)
        ns.device = device
        t0 = time.time()
        jepa_refit.run(ns)
        print(f"heads stage: {(time.time() - t0) / 60:.1f} min", flush=True)
        print(f"run {name} is complete: {run_dir / 'model.pt'}", flush=True)
    finally:
        sys.stdout = tee.out
        tee.close()
    return 0
