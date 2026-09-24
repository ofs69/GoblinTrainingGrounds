"""The smoke test: every command, end to end, on a folder of real pairs.

Run: python tests/smoke.py <folder-with-pairs> [--keep]

The folder holds a few short video/funscript pairs (eight one-minute clips
take about twenty minutes on a desktop GPU). Everything happens under
tests/.tmp/smoke, which is removed first unless --keep. The steps:

  import      the folder, unattended; then an import killed while it
              transcodes leaves a second project unchanged
  prepare     killed while it extracts leaves no cache behind; run again
              it completes, and the train and holdout rosters exist
  train       recipes/smoke.json (two epochs); a second call is a no-op
  eval        the holdout roster with the run's model
  export      the run into a bundle, with parity on a holdout clip
  check       the decode-invariant check
  draft       a copy of one video without its script, with the run's model
  remove      one clip, --undo, purge

Each step's output is kept in tests/.tmp/smoke/logs/. The test stops at the
first failing step and names it.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TMP = HERE / ".tmp" / "smoke"
LOGS = TMP / "logs"
PY = sys.executable


def run(step, *args, check=True):
    """One goblintrain command, its output in logs/<step>.log."""
    log = LOGS / f"{step}.log"
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call([PY, str(ROOT / "goblintrain.py"), *map(str, args)],
                             stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    dt = time.time() - t0
    print(f"  {step:<14} rc {rc}  {dt / 60:5.1f} min", flush=True)
    if check and rc != 0:
        raise SystemExit(f"FAILED at {step}: see {log}")
    return rc, log.read_text("utf-8", errors="replace")


def kill_tree(proc):
    if sys.platform == "win32":
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.kill()
    proc.wait()


def run_until(step, marker, *args, timeout=600):
    """Start a command and kill its whole process tree once ``marker``
    appears in its output. Returns the output so far."""
    log = LOGS / f"{step}.log"
    f = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen([PY, str(ROOT / "goblintrain.py"), *map(str, args)],
                            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                            env=dict(os.environ, PYTHONUNBUFFERED="1"))
    t0 = time.time()
    try:
        while time.time() - t0 < timeout:
            text = log.read_text("utf-8", errors="replace")
            if marker in text:
                kill_tree(proc)
                print(f"  {step:<14} killed at {marker!r}", flush=True)
                return text
            if proc.poll() is not None:
                raise SystemExit(f"FAILED at {step}: the command ended before "
                                 f"{marker!r} appeared; see {log}")
            time.sleep(0.5)
        kill_tree(proc)
        raise SystemExit(f"FAILED at {step}: {marker!r} did not appear in "
                         f"{timeout} s; see {log}")
    finally:
        f.close()


def manifest(root):
    p = root / "manifest.jsonl"
    return [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()]


def main(folder, keep=False):
    folder = Path(folder).resolve()
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".mp4")
    assert len(videos) >= 3, "the folder needs at least three pairs"
    if TMP.exists() and not keep:
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    proj = TMP / "proj"
    t_all = time.time()

    # ---- import
    if not (proj / "config.json").exists():
        run("init", "init", proj)
    run("import", "import", proj, folder, "--yes")
    ids = [r["id"] for r in manifest(proj)]
    assert len(ids) == len(videos), (ids, len(videos))
    _, out = run("status", "status", proj)
    assert f"{len(ids)} scripted clips" in out, out[-500:]

    # ---- an import killed mid-transcode changes nothing
    proj2 = TMP / "proj_killed_import"
    if proj2.exists():
        shutil.rmtree(proj2)
    run("init2", "init", proj2)
    run_until("import_kill", "transcoding", "import", proj2, folder, "--yes")
    _, out = run("status2", "status", proj2, check=False)
    assert manifest(proj2) == [], "a killed import left a manifest record"
    assert not any((proj2 / "videos").iterdir()), "a killed import left a video"
    assert "no clips yet" in out, out[-500:]
    assert not (proj2 / ".tx").exists() or not any((proj2 / ".tx").iterdir()), \
        "the staging directory survived recovery"

    # ---- prepare, killed while extracting, then whole
    run_until("prepare_kill", "latents  1", "prepare", proj)
    assert not any((proj / "latents").glob("*.npz")), \
        "a killed prepare left a latent cache"
    run("prepare", "prepare", proj)
    train = json.loads((proj / "rosters" / "train.json").read_text("utf-8"))["ids"]
    hold = json.loads((proj / "rosters" / "holdout.json").read_text("utf-8"))["ids"]
    assert len(train) >= 2 and len(hold) >= 1, (train, hold)
    assert all((proj / "latents" / f"{i}.npz").exists() for i in ids)
    _, out = run("prepare_again", "prepare", proj)
    assert "0 need boundaries, 0 need latents, 0 need a lag fit" in out, out[-300:]

    # ---- train, twice
    run("train", "train", proj, "--recipe", ROOT / "recipes" / "smoke.json",
        "--name", "smoke")
    assert (proj / "runs" / "smoke" / "model.pt").exists()
    _, out = run("train_again", "train", proj, "--recipe",
                 ROOT / "recipes" / "smoke.json", "--name", "smoke")
    assert "is complete" in out, out[-300:]

    # ---- eval
    _, out = run("eval", "eval", proj, "--run", "smoke")
    rec = json.loads((proj / "drafts" / "smoke-holdout" / "metrics.json")
                     .read_text("utf-8"))
    assert rec["pooled"]["n"] == len(hold), rec["pooled"]["n"]
    assert "=== artifact floor" in out

    # ---- export with parity
    run("export", "export", proj, "--run", "smoke")
    man = json.loads((proj / "bundle" / "manifest.json").read_text("utf-8"))
    assert "smoke" in man["packs"], man["packs"].keys()

    # ---- check
    _, out = run("check", "check")
    assert "decode check PASSED" in out, out[-300:]

    # ---- draft a script-less copy of one video with the run's model
    ddir = TMP / "draft"
    ddir.mkdir(exist_ok=True)
    video = ddir / "clip.mp4"
    shutil.copyfile(videos[0], video)
    run("draft", "draft", video, "--model", proj / "runs" / "smoke",
        "--project", TMP / "drafts_proj")
    fs = json.loads((ddir / "clip.funscript").read_text("utf-8"))
    assert len(fs["actions"]) > 10, len(fs["actions"])

    # ---- remove, undo, purge
    run("remove", "remove", proj, ids[-1])
    assert ids[-1] not in [r["id"] for r in manifest(proj)]
    run("undo", "remove", proj, "--undo")
    assert ids[-1] in [r["id"] for r in manifest(proj)]
    run("remove2", "remove", proj, ids[-1])
    run("purge", "purge", proj)
    assert not any((proj / ".trash").iterdir())

    print(f"smoke PASSED in {(time.time() - t_all) / 60:.1f} min "
          f"({len(ids)} clips)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], keep="--keep" in sys.argv[2:])
