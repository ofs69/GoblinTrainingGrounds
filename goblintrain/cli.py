"""The command line. One subcommand per stage; ``README.md`` is the contract."""
import argparse
import sys

from . import importer, tx
from .project import Project, ProjectError, named


def cmd_init(args):
    p = Project.create(named(args.project))
    print(f"created project {p.root.name} at {p.root}")
    print("it will hold your media and the private map; git ignores it")
    return 0


def cmd_import(args):
    with Project.open(named(args.project)) as p:
        return importer.run(p, args.paths, yes=args.yes,
                            keep_going=getattr(args, "continue"))


def cmd_status(args):
    from . import boundaries, common, extract, lagfit, prepare
    from .project import latent_stamp_problem
    with Project.open(named(args.project)) as p:
        recs = p.manifest()
        problems = p.verify()
        if not recs:
            print("no clips yet; add some with: goblintrain import "
                  f"{p.root.name} <video or folder>")
        else:
            hold = common.load_roster(p.root, "holdout")
            train = common.load_roster(p.root, "train") or []
            roster = {**{i: "train" for i in train},
                      **{i: "holdout" for i in hold or []}}
            print(f"{'id':>6}  {'length':>8}  {'actions':>7}  "
                  f"{'prepared':<8}  {'roster':<7}  note")
            for r in sorted(recs, key=lambda r: r["id"]):
                i = r["id"]
                acts = r.get("n_actions", 0) if p.is_scripted(r) else "none"
                lat = extract.path(p, i)
                ready = lat.is_file() and boundaries.path(p, i).is_file()
                note = ""
                side = lagfit.load(p, i) if p.is_scripted(r) else None
                if side is not None:
                    why = prepare.admitted(side)
                    note = f"not admitted: {why}" if why else ""
                if args.verify and lat.is_file():
                    bad = latent_stamp_problem(lat, p.config["perception"])
                    if bad:
                        problems.append(f"{i}: latents {bad}")
                print(f"{i:>6}  {importer.hms(r.get('duration_ms')):>8}  "
                      f"{acts:>7}  {'yes' if ready else 'no':<8}  "
                      f"{roster.get(i, '-'):<7}  {note}".rstrip())
            total = sum(r.get("duration_ms") or 0 for r in recs)
            n_un = sum(1 for r in recs if not p.is_scripted(r))
            n_sc = len(recs) - n_un
            print(f"{n_sc} scripted clip{'s' if n_sc != 1 else ''}"
                  + (f" and {n_un} without a script" if n_un else "")
                  + f", {importer.hms(total)} of video")
            print(f"train roster {len(train)}, "
                  + (f"holdout {len(hold)}" if hold is not None else
                     "holdout not drawn yet (a prepare of all clips "
                     "draws it)"))
        if tx.last_trash(p) is not None:
            print("the trash holds removed clips; goblintrain purge empties it")
        for msg in problems:
            print(f"problem: {msg}", file=sys.stderr)
        return 1 if problems else 0


def cmd_remove(args):
    with Project.open(named(args.project)) as p:
        if args.undo:
            trash = tx.last_trash(p)
            if trash is None:
                raise ProjectError("nothing to undo: the trash is empty")
            clip_id = tx.undo_remove(p, trash)
            print(f"restored {clip_id}")
            return 0
        if not args.id:
            raise ProjectError("give a clip id, or --undo")
        for clip_id in args.id:
            tx.remove_clip(p, clip_id)
            print(f"removed {clip_id} (goblintrain remove --undo puts the "
                  "last removal back)")
        return 0


def cmd_purge(args):
    with Project.open(named(args.project)) as p:
        n = tx.purge(p)
        print(f"purged {n} trash entr{'y' if n == 1 else 'ies'}")
        return 0


def cmd_prepare(args):
    from . import prepare          # torch loads only when a stage runs
    with Project.open(named(args.project)) as p:
        return prepare.run(p, args.ids, lag_fit=args.lag_fit)


def cmd_train(args):
    from . import train
    with Project.open(named(args.project)) as p:
        return train.run(p, recipe=args.recipe, from_=args.from_,
                         init_from=args.init_from, roster=args.roster,
                         name=args.name, sets=args.set)


def cmd_eval(args):
    from . import evaluate
    with Project.open(named(args.project)) as p:
        return evaluate.run(p, args.run, roster=args.roster, ref=args.ref,
                            name=args.name, rescore=args.rescore)


def cmd_export(args):
    from . import export
    with Project.open(named(args.project)) as p:
        return export.run(p, args.run, pack=args.pack, label=args.label,
                          bundle=args.bundle, clip_id=args.clip, rust=args.rust,
                          exe=args.exe, parity=not args.no_parity)


def cmd_draft(args):
    from . import draft
    return draft.run(args.videos, model=args.model, project=args.project,
                     out=args.out, force=args.force)


def cmd_fetch(args):
    from . import fetch
    return fetch.run(args.names)


def cmd_check(args):
    from . import grid_check
    grid_check.main()
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        prog="goblintrain",
        description="Train, fine-tune, evaluate and export GoblinScript models. "
                    "A project holds one dataset and lives in projects/ under "
                    "its name; see README.md.")
    sub = ap.add_subparsers(dest="command", required=True)

    def project_arg(sp):
        sp.add_argument("project",
                        help="the project's name (a directory in projects/)")

    sp = sub.add_parser("init", help="create a project in projects/")
    project_arg(sp)
    sp.set_defaults(fn=cmd_init)

    sp = sub.add_parser(
        "import",
        help="import hand-picked video/funscript pairs, one transaction each",
        description="A pair is clip.<video> beside clip.funscript. Name a "
                    "video (its sibling script is found), a video and a "
                    "script, or a folder (one level, every pair in it). "
                    "Everything is checked and listed before anything is "
                    "written.")
    project_arg(sp)
    sp.add_argument("paths", nargs="+", help="videos, scripts or folders")
    sp.add_argument("--yes", "-y", action="store_true",
                    help="import without asking after the listing")
    sp.add_argument("--continue", action="store_true",
                    help="import the other pairs when one is refused or fails")
    sp.set_defaults(fn=cmd_import)

    sp = sub.add_parser("status", help="every clip and stage, and any "
                                       "inconsistency between records and files")
    project_arg(sp)
    sp.add_argument("--verify", action="store_true",
                    help="also check every cache against the frozen perception")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("remove", help="move a clip to the trash; --undo puts "
                                       "the last removal back")
    project_arg(sp)
    sp.add_argument("id", nargs="*", help="clip id(s)")
    sp.add_argument("--undo", action="store_true")
    sp.set_defaults(fn=cmd_remove)

    sp = sub.add_parser("purge", help="empty the trash for good")
    project_arg(sp)
    sp.set_defaults(fn=cmd_purge)

    sp = sub.add_parser(
        "prepare",
        help="shot boundaries and latents for every clip that lacks them, "
             "then admission into the train and holdout rosters",
        description="Works clip by clip and only on what is missing; stop it "
                    "at any time and run it again. The scripts are taken as "
                    "synchronized with their videos unless --lag-fit. Needs "
                    "a CUDA device.")
    project_arg(sp)
    sp.add_argument("ids", nargs="*",
                    help="clip ids or roster names (default: every clip)")
    sp.add_argument("--lag-fit", action="store_true",
                    help="also fit each script's offset to its video against "
                         "the released model; training and eval then shift "
                         "the script by it, and a clip whose fit is not "
                         "confident, or that looks inverted or drifting, is "
                         "not admitted")
    sp.set_defaults(fn=cmd_prepare)

    sp = sub.add_parser(
        "train",
        help="fit the deploy heads on a shipped trunk (--from), or train a "
             "fresh trunk and its heads (--recipe)",
        description="A run is <project>/runs/<name> and ends in model.pt. "
                    "The same command again resumes an unfinished run. "
                    "Needs a CUDA device.")
    project_arg(sp)
    sp.add_argument("--from", dest="from_", metavar="RELEASE",
                    help="fine-tune: refit the deploy heads on this shipped "
                         "trunk (v0.6.0, v0.5.1) or bare trunk checkpoint, "
                         "with the recipe's heads section")
    sp.add_argument("--init-from", dest="init_from", metavar="RELEASE",
                    help="fine-tune the trunk too: run the recipe's trunk "
                         "stage starting from this shipped trunk (or bare "
                         "trunk checkpoint) instead of a random init, then "
                         "refit the heads as usual")
    sp.add_argument("--recipe", metavar="FILE",
                    help="the recipe (default: the one named by --from, "
                         "else recipes/v0.6.0.json)")
    sp.add_argument("--roster", default="train",
                    help="the roster to train on (default: train)")
    sp.add_argument("--name", help="the run name (default: the recipe's, or "
                                   "<release>-heads with --from)")
    sp.add_argument("--set", action="append", default=[],
                    metavar="SECTION.KEY=VALUE",
                    help="override one recipe key, e.g. heads.epochs=2")
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser(
        "eval",
        help="draft a roster with a model and read the written funscripts",
        description="Drafts land in <project>/drafts/<model>-<roster>/ with "
                    "their metrics.json. Drafts from the same checkpoint "
                    "are only read again; drafts from another checkpoint "
                    "are replaced. "
                    "Needs a CUDA device to draft.")
    project_arg(sp)
    sp.add_argument("--run", required=True, metavar="MODEL",
                    help="a run of this project, a shipped release (v0.6.0, "
                         "v0.5.1) or a checkpoint path")
    sp.add_argument("--roster", default="holdout",
                    help="the roster to draft (default: holdout)")
    sp.add_argument("--ref", metavar="NAME",
                    help="another eval directory under drafts/, printed "
                         "beside this one with paired bootstrap intervals")
    sp.add_argument("--name", help="the eval directory name (default: "
                                   "<model>-<roster>)")
    sp.add_argument("--rescore", action="store_true",
                    help="re-read drafts that already carry a read")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser(
        "export",
        help="export a model into a GoblinScript bundle as a pack, then "
             "run parity",
        description="The bundle is <project>/bundle unless --bundle says "
                    "otherwise; the first export writes it whole, a later "
                    "one adds its pack. Needs a CUDA device.")
    project_arg(sp)
    sp.add_argument("--run", required=True, metavar="MODEL",
                    help="a run of this project, a shipped release or a "
                         "checkpoint path")
    sp.add_argument("--pack", help="the pack's name (default: the model's)")
    sp.add_argument("--label", default="", help="one line about the pack")
    sp.add_argument("--bundle", metavar="DIR", help="the bundle directory")
    sp.add_argument("--clip", metavar="ID",
                    help="the prepared clip parity runs on (default: the "
                         "first of the holdout roster)")
    sp.add_argument("--rust", action="store_true",
                    help="also drive the built goblinscript binary")
    sp.add_argument("--exe", help="that binary (default: the newest build "
                                  "under goblinscript/)")
    sp.add_argument("--no-parity", action="store_true")
    sp.set_defaults(fn=cmd_export)

    sp = sub.add_parser(
        "draft", help="one video in, one funscript out",
        description="The funscript lands beside the video as "
                    "<stem>.funscript. The video is kept, transcoded, in a "
                    "drafting project (drafts by default) so a second draft "
                    "only re-runs what is missing.")
    sp.add_argument("videos", nargs="+", metavar="VIDEO")
    sp.add_argument("--model", metavar="MODEL",
                    help="a shipped release (default: v0.6.0), a checkpoint "
                         "path or a run directory")
    sp.add_argument("--project", metavar="NAME", default="drafts",
                    help="the drafting project's name (default: drafts; "
                         "created when absent)")
    sp.add_argument("--out", metavar="DIR",
                    help="write the funscripts here instead of beside the "
                         "videos")
    sp.add_argument("--force", action="store_true",
                    help="overwrite an existing funscript")
    sp.set_defaults(fn=cmd_draft)

    sp = sub.add_parser("fetch", help="download and hash-check the shipped "
                                      "weights, and fetch the V-JEPA encoder "
                                      "through torch.hub")
    sp.add_argument("names", nargs="*",
                    help="which checkpoints (default: all)")
    sp.set_defaults(fn=cmd_fetch)

    sp = sub.add_parser("check", help="the decode-invariant check: durations "
                                      "stay durations and the two decoders "
                                      "agree (CPU, seconds)")
    sp.set_defaults(fn=cmd_check)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except ProjectError as e:
        print(e, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted; the project is unchanged by anything unfinished",
              file=sys.stderr)
        return 130
