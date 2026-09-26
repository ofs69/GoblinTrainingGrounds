"""Fetch: the shipped weights, downloaded and hash-checked.

Every checkpoint under ``weights/checkpoints/`` has a JSON sidecar beside
it in git (``<name>.json``: size, SHA-256, download URL, recipe). The
checkpoints themselves are release assets. ``fetch`` downloads whatever is
missing or fails its hash and refuses a file that does not match. The PCA
basis is committed and is only checked. The V-JEPA encoder comes from
torch.hub, which keeps it in its own cache and downloads it only when the
cache lacks it.
"""
import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request

from . import common
from .project import PERCEPTION, ProjectError, basis_id_of

CHECKPOINTS = common.WEIGHTS_DIR / "checkpoints"
RELEASE_URL = ("https://github.com/ofs69/GoblinTrainingGrounds/releases/"
               "download/{tag}/{name}")
TIMEOUT_S = 60           # per read; a stalled connection raises an error
NET_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecars():
    return sorted(CHECKPOINTS.glob("*.json"))


def status(side):
    """(target path, 'ok' | 'missing' | 'mismatch')."""
    meta = json.loads(side.read_text("utf-8"))
    target = CHECKPOINTS / meta["file"]
    if not target.is_file():
        return target, meta, "missing"
    if os.path.getsize(target) != meta["bytes"] or sha256(target) != meta["sha256"]:
        return target, meta, "mismatch"
    return target, meta, "ok"


def download(url, target, expect_sha, expect_bytes, log=print):
    tmp = target.with_name(target.name + ".part")
    log(f"  downloading {target.name} ({expect_bytes / 1e6:.1f} MB)")
    try:
        with (urllib.request.urlopen(url, timeout=TIMEOUT_S) as r,
              open(tmp, "wb") as f):
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except BaseException as e:
        tmp.unlink(missing_ok=True)
        if isinstance(e, NET_ERRORS):
            raise ProjectError(f"{target.name}: download failed ({e}) from "
                               f"{url}; check the connection and run fetch "
                               f"again")
        raise
    if os.path.getsize(tmp) != expect_bytes or sha256(tmp) != expect_sha:
        tmp.unlink()
        raise ProjectError(f"{target.name}: the download does not match its "
                           f"sidecar; nothing was kept")
    os.replace(tmp, target)


def run(names=None, log=print):
    """The ``fetch`` command. Returns the exit code."""
    if basis_id_of(common.BASIS_PATH) != PERCEPTION["basis_id"]:
        raise ProjectError(f"{common.BASIS_PATH} is not the frozen basis "
                           f"{PERCEPTION['basis_id']}; restore it from git")
    log(f"basis {PERCEPTION['basis_id']}: ok")
    sides = sidecars()
    if names:
        want = set(names)
        sides = [s for s in sides if s.stem in want or s.stem.split("-")[0] in want]
        if not sides:
            raise ProjectError(f"no shipped checkpoint named {' '.join(names)}")
    for side in sides:
        target, meta, state = status(side)
        if state == "ok":
            log(f"{target.name}: ok")
            continue
        log(f"{target.name}: {state}")
        download(meta["url"], target, meta["sha256"], meta["bytes"], log)
        log(f"{target.name}: fetched and verified")
    from . import extract
    log("V-JEPA encoder: loading through torch.hub (downloads it if missing)")
    try:
        extract.load_encoder(PERCEPTION, "cpu")
    except NET_ERRORS as e:
        raise ProjectError(f"V-JEPA encoder: download failed ({e}); check "
                           f"the connection and run fetch again")
    log("encoder: ok")
    return 0


def stamp(tag, log=print):
    """Maintainer: write the sidecar of every checkpoint for release
    ``tag``, from the files on disk."""
    for pt in sorted(CHECKPOINTS.glob("*.pt")):
        name = pt.stem
        meta = {"file": pt.name, "bytes": os.path.getsize(pt),
                "sha256": sha256(pt),
                "url": RELEASE_URL.format(tag=tag, name=pt.name),
                "recipe": name.split("-")[0],
                "kind": "trunk" if name.endswith("-trunk") else "model"}
        (CHECKPOINTS / f"{name}.json").write_text(
            json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        log(f"stamped {pt.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stamp", metavar="TAG",
                    help="maintainer: write the sidecars for release TAG "
                         "from the checkpoints on disk")
    args = ap.parse_args()
    if args.stamp:
        stamp(args.stamp)
    else:
        run()


if __name__ == "__main__":
    main()
