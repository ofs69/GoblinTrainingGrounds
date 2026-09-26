"""Import: hand-picked video/funscript pairs into a project, one transaction
each.

The user names a pair, a video whose sibling script is found, or a folder
they assembled (one level, no recursion). Every pair is looked at before
anything is written (:func:`preflight`), then imported on its own
(:func:`import_pair`). An original filename is never printed; a pair is
shown by its position in the batch until it has an ID.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

from . import common, funscript, media, tx
from .project import ProjectError, atomic_write_text

VIDEO_EXTS = [".mp4", ".mkv", ".avi", ".wmv", ".mov", ".m4v", ".webm",
              ".mpg", ".mpeg", ".ts", ".flv", ".m2ts"]
SCRIPT_EXT = ".funscript"

# Actions past the video's end are trimmed. A script whose last action lies
# further than this past the end gets a warning, because it may be for
# another cut of the video.
OVERRUN_MS = 1000.0


class Pair:
    def __init__(self, video, script):
        self.video = Path(video)
        self.script = Path(script)
        # filled by preflight
        self.refusal = None       # a reason, or None when importable
        self.warning = None       # imported, with a warning in the listing
        self.raw = None
        self.key = None
        self.sig = None
        self.duration_ms = None
        self.n_raw = 0
        self.clean = None


def find_pairs(paths):
    """Pairs and leftovers from what the user named. Returns
    ``(pairs, unpaired_videos, unpaired_scripts, not_found)`` where the
    unpaired lists hold counts only."""
    paths = [Path(p) for p in paths]
    pairs, n_video, n_script, missing = [], 0, 0, []
    if (len(paths) == 2 and all(p.is_file() for p in paths)
            and {_kind(paths[0]), _kind(paths[1])} == {"video", "script"}):
        v, s = (paths if _kind(paths[0]) == "video" else paths[::-1])
        return [Pair(v, s)], 0, 0, []
    for p in paths:
        if p.is_dir():
            got = _pairs_in_dir(p)
            pairs += got[0]
            n_video += got[1]
            n_script += got[2]
        elif p.is_file():
            kind = _kind(p)
            if kind == "video":
                s = _sibling(p, [SCRIPT_EXT])
                if s is None:
                    n_video += 1
                else:
                    pairs.append(Pair(p, s))
            elif kind == "script":
                v = _sibling(p, VIDEO_EXTS)
                if v is None:
                    n_script += 1
                else:
                    pairs.append(Pair(v, p))
            else:
                missing.append(p)
        else:
            missing.append(p)
    # a pair named twice (both of its files, or a file and its folder, as
    # a shell glob gives them) is one pair, not a duplicate to refuse
    seen, unique = set(), []
    for pr in pairs:
        key = (pr.video.resolve(), pr.script.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(pr)
    return unique, n_video, n_script, missing


def _kind(p):
    ext = p.suffix.lower()
    if ext == SCRIPT_EXT:
        return "script"
    if ext in VIDEO_EXTS:
        return "video"
    return None


def _sibling(p, exts):
    """The file beside ``p`` with the same stem and one of ``exts``, matched
    case-insensitively, first by ``exts`` order."""
    want = p.stem.lower()
    found = {}
    for q in p.parent.iterdir():
        if q.is_file() and q.stem.lower() == want and q != p:
            found.setdefault(q.suffix.lower(), q)
    for e in exts:
        if e in found:
            return found[e]
    return None


def _pairs_in_dir(d):
    videos, scripts = {}, {}
    for q in sorted(d.iterdir()):
        if not q.is_file():
            continue
        kind = _kind(q)
        if kind == "video":
            videos.setdefault(q.stem.lower(), {})[q.suffix.lower()] = q
        elif kind == "script":
            scripts.setdefault(q.stem.lower(), q)
    pairs = []
    for stem, s in sorted(scripts.items()):
        cands = videos.pop(stem, None)
        if cands is None:
            continue
        v = next(cands[e] for e in VIDEO_EXTS if e in cands)
        pairs.append(Pair(v, s))
    n_script = sum(1 for stem in scripts if stem not in {
        p.script.stem.lower() for p in pairs})
    return pairs, len(videos), n_script


def preflight(project, pairs):
    """Look at every pair without writing anything. Fills each pair's
    refusal or its import facts."""
    recs = project.manifest()
    by_key = {r.get("key"): r["id"] for r in recs if r.get("key")}
    by_sig = {r.get("sig"): r["id"] for r in recs if r.get("sig")}
    seen_key, seen_sig = {}, {}
    for n, pr in enumerate(pairs, 1):
        try:
            pr.raw = pr.script.read_bytes()
            vsize = os.path.getsize(pr.video)
        except OSError:
            pr.refusal = "unreadable file"
            continue
        acts = funscript.load_actions(pr.raw)
        if acts is None:
            pr.refusal = "the script is not funscript JSON"
            continue
        pr.n_raw = len(acts)
        pr.key = funscript.content_key(pr.raw, vsize)
        pr.sig = funscript.signature_from_bytes(pr.raw)
        dup = by_key.get(pr.key) or (pr.sig and by_sig.get(pr.sig))
        if dup:
            pr.refusal = f"duplicate of clip {dup}"
            continue
        dup = seen_key.get(pr.key) or (pr.sig and seen_sig.get(pr.sig))
        if dup:
            pr.refusal = f"duplicate of pair {dup} in this batch"
            continue
        pr.duration_ms = media.probe_duration_ms(pr.video)
        if not pr.duration_ms:
            pr.refusal = "ffprobe cannot read the video"
            continue
        unbounded = funscript.sanitize_aligned(acts)
        if unbounded and unbounded[-1][0] > pr.duration_ms + OVERRUN_MS:
            over = (unbounded[-1][0] - pr.duration_ms) / 1000
            pr.warning = (f"the script runs {over:.1f} s past the end of the "
                          "video and is cut there; it may be for another "
                          "cut of the video")
        pr.clean = funscript.sanitize_aligned(acts, pr.duration_ms)
        if len(pr.clean) < 2:
            pr.refusal = "fewer than two usable actions"
            continue
        seen_key[pr.key] = n
        if pr.sig:
            seen_sig[pr.sig] = n


def import_pair(project, pr, log=print):
    """One transaction: transcode into staging, write the script and meta,
    commit with the manifest record and the map entry. Returns the ID."""
    if pr.refusal:
        raise ProjectError(pr.refusal)
    cfg = project.config["transcode"]
    clip_id = project.next_id()
    with tx.Transaction(project, "add", [clip_id]) as t:
        vdst = t.path("videos", clip_id + ".mp4")
        last = [-1]

        def progress(frac):
            pct = int(frac * 100) // 25 * 25
            if pct > last[0]:
                last[0] = pct
                log(f"  [{clip_id}] transcoding {pct:3d}%")

        reason = media.transcode(pr.video, vdst, cfg, pr.duration_ms, progress)
        if reason:
            raise ProjectError(f"[{clip_id}] {reason}")
        vbytes = os.path.getsize(vdst)
        vsize = os.path.getsize(pr.video)
        sdst = t.path("scripts", clip_id + ".json")
        atomic_write_text(sdst, json.dumps(
            [{"at": int(at), "pos": pos} for at, pos in pr.clean]))
        meta = {
            "id": clip_id,
            "duration_ms": pr.duration_ms,
            "n_actions_raw": pr.n_raw,
            "n_actions": len(pr.clean),
            "source_bytes": vsize,
            "video_bytes": vbytes,
            "height": cfg["height"], "fps": cfg["fps"], "crf": cfg["crf"],
            "preset": cfg["preset"],
            "audio_rate": cfg["audio_rate"],
            "audio_channels": cfg["audio_channels"],
            "audio_bitrate": cfg["audio_bitrate"],
            **media.probe_source_rate(pr.video),
        }
        mdst = t.path("meta", clip_id + ".json")
        atomic_write_text(mdst, json.dumps(meta))
        record = {"id": clip_id, "key": pr.key, "sig": pr.sig, "status": "ok",
                  "duration_ms": pr.duration_ms, "n_actions": len(pr.clean),
                  "source_bytes": vsize, "video_bytes": vbytes}
        t.commit([(vdst, project.video_path(clip_id)),
                  (sdst, project.script_path(clip_id)),
                  (mdst, project.root / "meta" / f"{clip_id}.json")],
                 records=[record],
                 sources={clip_id: f"{pr.video}\t{pr.script}"})
    log(f"  [{clip_id}] done  {hms(pr.duration_ms)}  {vbytes / 1e6:.1f} MB  "
        f"{len(pr.clean)} actions")
    return clip_id


def video_key(path):
    """The key of a video imported without a script: its size and its
    first megabytes, so the same file is found again."""
    h = hashlib.md5(str(os.path.getsize(path)).encode())
    with open(path, "rb") as f:
        h.update(f.read(4 << 20))
    return "video:" + h.hexdigest()


def import_video(project, video, log=print):
    """A video without a script, for drafting: the same transaction as a
    pair minus the script, and found again by its key on a second call.
    Returns (ID, whether it was imported now)."""
    video = Path(video)
    if not video.is_file() or _kind(video) != "video":
        raise ProjectError(f"not a video file: {video.name}")
    key = video_key(video)
    for r in project.manifest():
        if r.get("key") == key:
            return r["id"], False
    duration_ms = media.probe_duration_ms(video)
    if not duration_ms:
        raise ProjectError(f"ffprobe cannot read {video.name}")
    cfg = project.config["transcode"]
    clip_id = project.next_id()
    with tx.Transaction(project, "add", [clip_id]) as t:
        vdst = t.path("videos", clip_id + ".mp4")
        last = [-1]

        def progress(frac):
            pct = int(frac * 100) // 25 * 25
            if pct > last[0]:
                last[0] = pct
                log(f"  [{clip_id}] transcoding {pct:3d}%")

        reason = media.transcode(video, vdst, cfg, duration_ms, progress)
        if reason:
            raise ProjectError(f"[{clip_id}] {reason}")
        vbytes = os.path.getsize(vdst)
        vsize = os.path.getsize(video)
        meta = {
            "id": clip_id, "duration_ms": duration_ms, "unscripted": True,
            "n_actions_raw": 0, "n_actions": 0,
            "source_bytes": vsize, "video_bytes": vbytes,
            "height": cfg["height"], "fps": cfg["fps"], "crf": cfg["crf"],
            "preset": cfg["preset"],
            "audio_rate": cfg["audio_rate"],
            "audio_channels": cfg["audio_channels"],
            "audio_bitrate": cfg["audio_bitrate"],
            **media.probe_source_rate(video),
        }
        mdst = t.path("meta", clip_id + ".json")
        atomic_write_text(mdst, json.dumps(meta))
        record = {"id": clip_id, "key": key, "status": "ok",
                  "duration_ms": duration_ms, "n_actions": 0,
                  "unscripted": True,
                  "source_bytes": vsize, "video_bytes": vbytes}
        t.commit([(vdst, project.video_path(clip_id)),
                  (mdst, project.root / "meta" / f"{clip_id}.json")],
                 records=[record], sources={clip_id: str(video)})
    log(f"  [{clip_id}] imported  {hms(duration_ms)}  {vbytes / 1e6:.1f} MB")
    return clip_id, True


def hms(ms):
    s = int(round((ms or 0) / 1000))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def run(project, paths, yes=False, keep_going=False, log=print,
        ask=None):
    """The ``import`` command. Returns the exit code."""
    missing_tools = media.have_tools()
    if missing_tools:
        raise ProjectError(f"{' and '.join(missing_tools)} not found on PATH; "
                           "see README.md, What you need")
    pairs, n_video, n_script, missing = find_pairs(paths)
    if missing:
        log(f"{len(missing)} argument{'s' if len(missing) != 1 else ''} "
            "not found or neither a video, a script nor a folder")
    if not pairs:
        raise ProjectError("no video/funscript pairs found; a pair is "
                           "clip.mp4 beside clip.funscript")
    preflight(project, pairs)
    ok = [p for p in pairs if not p.refusal]
    refused = len(pairs) - len(ok)
    log(f"found {len(pairs)} pair{'s' if len(pairs) != 1 else ''}: "
        f"{len(ok)} to import, {refused} refused"
        + (f", {n_video} video{'s' if n_video != 1 else ''} without a script"
           if n_video else "")
        + (f", {n_script} script{'s' if n_script != 1 else ''} without a video"
           if n_script else ""))
    win_ms = 1000.0 * common.default_recipe_window_s()
    for n, p in enumerate(pairs, 1):
        if p.refusal:
            log(f"  pair {n}: refused, {p.refusal}")
        else:
            log(f"  pair {n}: {hms(p.duration_ms)}, {len(p.clean)} actions"
                + (f" ({p.n_raw - len(p.clean)} dropped by sanitization)"
                   if p.n_raw != len(p.clean) else "")
                + (", too short to train" if p.duration_ms < win_ms else ""))
            if p.warning:
                log(f"  pair {n}: warning, {p.warning}")
    if any(p.duration_ms < win_ms for p in ok):
        log(f"a clip shorter than the training window of the "
            f"{common.DEFAULT_RECIPE} recipe ({hms(win_ms)}) can be evaluated "
            f"and drafted, but it does not train")
    if not ok:
        return 1
    if refused and not keep_going:
        log("nothing imported: a refusal stops the batch; fix it or pass "
            "--continue to import the other pairs")
        return 1
    log(f"to import: {hms(sum(p.duration_ms for p in ok))} of video")
    if not yes:
        if ask is None:
            if not sys.stdin.isatty():
                raise ProjectError("not a terminal: pass --yes to import "
                                   "without a prompt")
            ask = input
        try:
            answer = ask(f"import {len(ok)} pair{'s' if len(ok) != 1 else ''}? "
                         "[y/N] ")
        except EOFError:
            answer = ""
            log("")
        if answer.strip().lower() not in ("y", "yes"):
            log("nothing imported")
            return 1
    n_done = 0
    for n, p in enumerate(pairs, 1):
        if p.refusal:
            continue
        try:
            import_pair(project, p, log)
            n_done += 1
        except ProjectError as e:
            log(f"  pair {n}: {e}")
            if not keep_going:
                log(f"stopped at pair {n}; {n_done} imported, the rest "
                    "untouched (--continue goes on past a failure)")
                return 1
    log(f"imported {n_done} of {len(ok)}")
    return 0 if n_done == len(ok) else 1
