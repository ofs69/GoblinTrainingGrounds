"""The smoke test: every command, end to end, on a folder of real pairs.

Run: python tests/smoke.py <folder-with-pairs> [--keep]

The folder holds a few short video/funscript pairs (eight one-minute clips
take about twenty minutes on a desktop GPU). Everything happens under
tests/.tmp/smoke and in the projects smoke-test, smoke-test-killed and
smoke-test-drafts, which are removed first unless --keep. The steps:

  import      the folder, unattended; then an import killed while it
              transcodes leaves a second project unchanged
  prepare     killed while it extracts leaves no cache behind; run again
              it completes, and the train and holdout rosters exist
  train       recipes/smoke.json (two epochs), killed once in the trunk
              stage and once in the heads stage, each resumed; a second
              call is a no-op; the checkpoints record the train roster's
              scripts
  eval        the holdout with v0.6.0, then with the run's model against
              it; a second call only re-reads
  refit       heads on the v0.6.0 trunk: the holdout reads whole, the
              train roster on its held-out rows; an eval directory whose
              checkpoint changed is drafted again
  export      the run into a bundle, with parity on a holdout clip; the
              graphs carry no local paths; the same pack again is refused
  check       the decode-invariant check
  draft       a copy of one video without its script, with the run's model
  remove      a holdout clip leaves the holdout and --undo puts it back;
              the highest ID, --undo, purge; a re-import of the folder,
              every file named twice, adds that pair once under a new ID

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
PROJ, PROJ2, DPROJ = "smoke-test", "smoke-test-killed", "smoke-test-drafts"
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
    # proc is the venv launcher; the interpreter holding the project lock is
    # its child and can take seconds to exit (a CUDA context tears down), so
    # the next command waits for it rather than finding the project in use
    sys.path.insert(0, str(ROOT))
    from goblintrain.project import _pid_alive
    t0 = time.time()
    for lock in (ROOT / "projects").glob("smoke-test*/.lock"):
        try:
            pid = int(lock.read_text().strip() or 0)
        except (OSError, ValueError):
            continue
        while pid and _pid_alive(pid) and time.time() - t0 < 120:
            time.sleep(0.5)


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


def run_until_file(step, path, *args, timeout=3600):
    """Start a command and kill its whole process tree once ``path``
    exists (a resume point, written after the log line that announces
    its epoch). Returns the output so far."""
    log = LOGS / f"{step}.log"
    f = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen([PY, str(ROOT / "goblintrain.py"), *map(str, args)],
                            stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                            env=dict(os.environ, PYTHONUNBUFFERED="1"))
    t0 = time.time()
    try:
        while time.time() - t0 < timeout:
            if Path(path).exists():
                kill_tree(proc)
                print(f"  {step:<14} killed at {Path(path).name}", flush=True)
                return log.read_text("utf-8", errors="replace")
            if proc.poll() is not None:
                raise SystemExit(f"FAILED at {step}: the command ended before "
                                 f"{Path(path).name} appeared; see {log}")
            time.sleep(0.5)
        kill_tree(proc)
        raise SystemExit(f"FAILED at {step}: {Path(path).name} did not "
                         f"appear in {timeout} s; see {log}")
    finally:
        f.close()


def roster(root, name):
    sys.path.insert(0, str(ROOT))
    from goblintrain import common
    return common.load_roster(root, name)


def manifest(root):
    p = root / "manifest.jsonl"
    return [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()]


def main(folder, keep=False):
    folder = Path(folder).resolve()
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".mp4")
    assert len(videos) >= 3, "the folder needs at least three pairs"
    projects = ROOT / "projects"
    if not keep:
        for d in [TMP] + [projects / n for n in (PROJ, PROJ2, DPROJ)]:
            if d.exists():
                shutil.rmtree(d)
    TMP.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    proj = projects / PROJ
    t_all = time.time()

    # ---- import
    if not (proj / "config.json").exists():
        run("init", "init", PROJ)
    run("import", "import", PROJ, folder, "--yes")
    ids = [r["id"] for r in manifest(proj)]
    assert len(ids) == len(videos), (ids, len(videos))
    _, out = run("status", "status", PROJ)
    assert f"{len(ids)} scripted clips" in out, out[-500:]

    # ---- an import killed mid-transcode changes nothing
    proj2 = projects / PROJ2
    if proj2.exists():
        shutil.rmtree(proj2)
    run("init2", "init", PROJ2)
    run_until("import_kill", "transcoding", "import", PROJ2, folder, "--yes")
    _, out = run("status2", "status", PROJ2, check=False)
    assert manifest(proj2) == [], "a killed import left a manifest record"
    assert not any((proj2 / "videos").iterdir()), "a killed import left a video"
    assert "no clips yet" in out, out[-500:]
    assert not (proj2 / ".tx").exists() or not any((proj2 / ".tx").iterdir()), \
        "the staging directory survived recovery"

    # ---- prepare, killed while extracting, then whole
    run_until("prepare_kill", "latents  1", "prepare", PROJ)
    assert not any((proj / "latents").glob("*.npz")), \
        "a killed prepare left a latent cache"
    run("prepare", "prepare", PROJ)
    train = json.loads((proj / "rosters" / "train.json").read_text("utf-8"))["ids"]
    hold = json.loads((proj / "rosters" / "holdout.json").read_text("utf-8"))["ids"]
    assert len(train) >= 2 and len(hold) >= 1, (train, hold)
    assert all((proj / "latents" / f"{i}.npz").exists() for i in ids)
    _, out = run("prepare_again", "prepare", PROJ)
    assert "0 need boundaries, 0 need latents" in out, out[-300:]
    assert "lag fit" not in out, out[-300:]

    # ---- train: killed in each stage, resumed, then a no-op
    smoke = ("--recipe", ROOT / "recipes" / "smoke.json")
    rdir = proj / "runs" / "smoke"
    run_until_file("train_kill1", rdir / "trunk" / "resume_state.pt",
                   "train", PROJ, *smoke, "--name", "smoke")
    out = run_until_file("train_kill2", rdir / "refit_resume.pt",
                         "train", PROJ, *smoke, "--name", "smoke")
    assert "RESUMED from epoch" in out, out[-500:]
    assert not (rdir / "model.pt").exists()
    _, out = run("train", "train", PROJ, *smoke, "--name", "smoke")
    assert "refit RESUMED" in out, out[-500:]
    assert (rdir / "model.pt").exists()
    _, out = run("train_again", "train", PROJ, *smoke, "--name", "smoke")
    assert "is complete" in out, out[-300:]
    import torch
    want = {r["sig"] for r in manifest(proj) if r["id"] in train}
    for ck in (rdir / "model.pt", rdir / "trunk" / "jepa_best.pt"):
        got = torch.load(ck, map_location="cpu", weights_only=False)
        assert set(got["trained_sigs"]) == want, (ck.name, len(want))

    # ---- eval: the release, the run against it, a re-read
    _, out = run("eval_release", "eval", PROJ, "--run", "v0.6.0")
    assert "whole" in out, out[-300:]
    _, out = run("eval", "eval", PROJ, "--run", "smoke",
                 "--ref", "v0.6.0-holdout")
    rec = json.loads((proj / "drafts" / "smoke-holdout" / "metrics.json")
                     .read_text("utf-8"))
    assert rec["pooled"]["n"] == len(hold), rec["pooled"]["n"]
    assert "=== artifact floor" in out
    assert "paired clip-bootstrap" in out, out[-500:]
    _, out = run("eval_again", "eval", PROJ, "--run", "smoke")
    assert "every draft exists" in out, out[-300:]

    # ---- heads on the release trunk
    run("refit", "train", PROJ, "--from", "v0.6.0", *smoke,
        "--set", "heads.epochs=1", "--name", "smoke-heads")
    _, out = run("eval_refit", "eval", PROJ, "--run", "smoke-heads")
    assert "whole" in out, out[-300:]
    _, out = run("eval_refit_train", "eval", PROJ, "--run", "smoke-heads",
                 "--roster", "train")
    assert "on their held-out rows" in out, out[-300:]
    run("eval_swap1", "eval", PROJ, "--run", "smoke", "--name", "swap")
    _, out = run("eval_swap2", "eval", PROJ, "--run", "smoke-heads",
                 "--name", "swap")
    assert "come from another checkpoint" in out, out[-300:]

    # ---- export with parity
    run("export", "export", PROJ, "--run", "smoke")
    man = json.loads((proj / "bundle" / "manifest.json").read_text("utf-8"))
    assert "smoke" in man["packs"], man["packs"].keys()
    for g in (proj / "bundle").rglob("*.onnx"):
        raw = g.read_bytes()
        for leak in (b"site-packages", b"stack_trace", str(ROOT).encode(),
                     str(Path.home()).encode()):
            assert leak not in raw, (g.name, leak[:12])
    rc, _ = run("export_again", "export", PROJ, "--run", "smoke",
                "--no-parity", check=False)
    assert rc != 0, "a second export of the same pack was not refused"

    # ---- check
    _, out = run("check", "check")
    assert "decode check PASSED" in out, out[-300:]

    # ---- draft a script-less copy of one video with the run's model
    ddir = TMP / "draft"
    ddir.mkdir(exist_ok=True)
    video = ddir / "clip.mp4"
    shutil.copyfile(videos[0], video)
    run("draft", "draft", video, "--model", proj / "runs" / "smoke",
        "--project", DPROJ)
    fs = json.loads((ddir / "clip.funscript").read_text("utf-8"))
    assert len(fs["actions"]) > 10, len(fs["actions"])

    # ---- a holdout clip leaves the holdout and comes back
    h = hold[0]
    run("remove_hold", "remove", PROJ, h)
    assert h not in roster(proj, "holdout")
    run("eval_removed", "eval", PROJ, "--run", "v0.6.0", "--name", "removed")
    rec = json.loads((proj / "drafts" / "removed" / "metrics.json")
                     .read_text("utf-8"))
    assert rec["pooled"]["n"] == len(hold) - 1, rec["pooled"]["n"]
    run("undo_hold", "remove", PROJ, "--undo")
    assert h in roster(proj, "holdout")
    _, out = run("status_rosters", "status", PROJ)
    assert f"holdout {len(hold)}" in out, out[-300:]

    # ---- the highest ID: remove, undo, purge; its ID is not reissued
    run("remove", "remove", PROJ, ids[-1])
    assert ids[-1] not in [r["id"] for r in manifest(proj)]
    run("undo", "remove", PROJ, "--undo")
    assert ids[-1] in [r["id"] for r in manifest(proj)]
    run("remove2", "remove", PROJ, ids[-1])
    run("purge", "purge", PROJ)
    assert not any((proj / ".trash").iterdir())
    scripts = sorted(p for p in folder.iterdir()
                     if p.suffix.lower() == ".funscript")
    run("reimport", "import", PROJ, *videos, *scripts, folder, "--yes",
        "--continue", check=False)
    new = [r["id"] for r in manifest(proj) if r["id"] not in ids]
    assert len(new) == 1 and new[0] > ids[-1], new

    print(f"smoke PASSED in {(time.time() - t_all) / 60:.1f} min "
          f"({len(ids)} clips)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], keep="--keep" in sys.argv[2:])
