"""The transaction layer, exercised on a throwaway project under tests/.tmp:
a committed add, a crash before the records landed, a crash after they
landed, remove, undo, purge, rosters and IDs across a remove, the lock, and
duplicate-id refusal.

Run: python tests/test_tx.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from goblintrain import common, tx                           # noqa: E402
from goblintrain.project import Project, ProjectError        # noqa: E402

TMP = HERE / ".tmp" / "tx"


def fresh():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    return Project.create(TMP / "proj")


def record(i, minutes=1.0, actions=10):
    return {"id": i, "duration_ms": int(minutes * 60000), "n_actions": actions,
            "key": f"k{i}", "sig": f"s{i}", "status": "ok"}


def stage_clip(p, t, i):
    """A fake video and script in staging, and the moves that place them."""
    v = t.path("videos", f"{i}.mp4")
    s = t.path("scripts", f"{i}.funscript")
    v.write_bytes(b"video " + i.encode())
    s.write_text(json.dumps({"actions": []}))
    return [(v, p.video_path(i)), (s, p.script_path(i))]


def test_add_commit():
    with Project.open(TMP / "proj") as p:
        i = p.next_id()
        assert i == "000001"
        with tx.Transaction(p, "add", [i]) as t:
            moves = stage_clip(p, t, i)
            t.commit(moves, records=[record(i)], sources={i: "/somewhere/x.mp4"})
        assert p.ids() == [i]
        assert p.video_path(i).read_bytes() == b"video 000001"
        assert p.map_sources() == {i: "/somewhere/x.mp4"}
        assert not any(p.tx_dir.iterdir())
        assert p.verify() == []
        assert p.next_id() == "000002"


def test_crash_before_records():
    """Files moved, records never written: recovery puts everything back."""
    with Project.open(TMP / "proj") as p:
        i = p.next_id()
        t = tx.Transaction(p, "add", [i])
        moves = stage_clip(p, t, i)
        # the first half of commit by hand: journal + moves, then "die"
        t._journal({"state": "committing",
                    "moves": [[tx._rel(p, s), tx._rel(p, d)] for s, d in moves]})
        for s, d in moves:
            d.parent.mkdir(parents=True, exist_ok=True)
            os.rename(s, d)
        assert p.video_path(i).exists()
        before = p.ids()
    with Project.open(TMP / "proj") as p:      # open runs recovery
        assert p.ids() == before
        assert not p.video_path(i).exists()
        assert not p.script_path(i).exists()
        assert not any(p.tx_dir.iterdir())
        assert p.verify() == []


def test_failed_rename_rolls_back():
    """A rename fails inside commit: the files already moved go back at
    once and the staging directory is gone."""
    real_rename = os.rename
    calls = [0]

    def flaky(src, dst):
        calls[0] += 1
        if calls[0] == 2:
            raise OSError("simulated failure")
        real_rename(src, dst)

    with Project.open(TMP / "proj") as p:
        before = p.ids()
        i = p.next_id()
        os.rename = flaky
        try:
            with tx.Transaction(p, "add", [i]) as t:
                t.commit(stage_clip(p, t, i), records=[record(i)])
        except OSError:
            pass
        else:
            raise AssertionError("the failure did not happen")
        finally:
            os.rename = real_rename
        assert p.ids() == before
        assert not p.video_path(i).exists()
        assert not p.script_path(i).exists()
        assert not any(p.tx_dir.iterdir())
        assert p.verify() == []


def test_crash_after_records():
    """Records landed, cleanup did not: recovery only removes the journal."""
    with Project.open(TMP / "proj") as p:
        i = p.next_id()
        with tx.Transaction(p, "add", [i]) as t:
            moves = stage_clip(p, t, i)
            t.commit(moves, records=[record(i)], sources={i: "src"})
        # resurrect a committing journal for a finished add
        d = p.tx_dir / "stale-committing"
        d.mkdir()
        (d / tx.JOURNAL).write_text(json.dumps(
            {"txid": "stale-committing", "kind": "add", "ids": [i],
             "state": "committing", "moves": []}))
    with Project.open(TMP / "proj") as p:
        assert i in p.ids()
        assert p.video_path(i).exists()
        assert not any(p.tx_dir.iterdir())
        assert p.verify() == []


def test_map_prune():
    """A map line without a manifest record is dropped on recovery."""
    with Project.open(TMP / "proj") as p:
        p.write_map(p.map_lines() + ["000999\tghost"])
    with Project.open(TMP / "proj") as p:
        assert "000999" not in p.map_sources()
        assert p.verify() == []


def test_remove_undo_purge():
    with Project.open(TMP / "proj") as p:
        ids = p.ids()
        i = ids[0]
        # a derived file in a cache directory goes with the clip
        cache = p.root / "latents" / "scope"
        cache.mkdir(parents=True)
        (cache / f"{i}.npy").write_bytes(b"latents")
        trash = tx.remove_clip(p, i)
        assert i not in p.ids()
        assert i not in p.map_sources()
        assert not p.video_path(i).exists()
        assert (trash / "latents" / "scope" / f"{i}.npy").exists()
        assert p.verify() == []
        # a second removal makes "last" the newer one
        j = p.ids()[0]
        tx.remove_clip(p, j)
        last = tx.last_trash(p)
        assert tx.undo_remove(p, last) == j
        assert j in p.ids()
        assert p.video_path(j).exists()
        assert p.verify() == []
        # undo the first one too
        assert tx.undo_remove(p, tx.last_trash(p)) == i
        assert (cache / f"{i}.npy").read_bytes() == b"latents"
        assert p.map_sources()[i] == "/somewhere/x.mp4"
        assert p.verify() == []
        assert tx.last_trash(p) is None
        tx.remove_clip(p, i)
        assert tx.purge(p) == 1
        assert tx.last_trash(p) is None


def test_remove_rosters_and_ids():
    """A removed clip leaves every roster and its ID is never reissued, even
    after a purge; undo puts it back into its roster."""
    with Project.open(TMP / "proj") as p:
        for _ in range(2):
            i = p.next_id()
            with tx.Transaction(p, "add", [i]) as t:
                t.commit(stage_clip(p, t, i), records=[record(i)])
        top = max(p.ids())
        (p.root / "rosters" / "holdout.json").write_text(
            json.dumps({"ids": [top]}))
        tx.remove_clip(p, top)
        assert common.load_roster(p.root, "holdout") == []
        assert p.next_id() > top
        tx.undo_remove(p, tx.last_trash(p))
        assert common.load_roster(p.root, "holdout") == [top]
        tx.remove_clip(p, top)
        tx.purge(p)
    with Project.open(TMP / "proj") as p:
        assert p.next_id() > top
        assert common.load_roster(p.root, "holdout") == []
        assert p.verify() == []


def test_crash_during_remove():
    """Files moved to trash, manifest still names the clip: rolled back."""
    with Project.open(TMP / "proj") as p:
        i = p.ids()[0]
        t = tx.Transaction(p, "remove", [i])
        files = p.clip_files(i)
        moves = [(f, t.trash_path(tx._rel(p, f))) for f in files]
        t._journal({"state": "committing",
                    "moves": [[tx._rel(p, s), tx._rel(p, d)] for s, d in moves]})
        for s, d in moves:
            os.rename(s, d)
        assert not p.video_path(i).exists()
    with Project.open(TMP / "proj") as p:
        assert i in p.ids()
        assert p.video_path(i).exists()
        assert p.verify() == []
        assert not any(p.trash_dir.iterdir())


def test_duplicate_id_refused():
    with Project.open(TMP / "proj") as p:
        i = p.ids()[0]
        with tx.Transaction(p, "add", [i]) as t:
            moves = stage_clip(p, t, "000777")
            moves = [(s, d.with_name(d.name.replace("000777", "000778")))
                     for s, d in moves]
            try:
                t.commit(moves, records=[record(i)])
            except ProjectError:
                pass
            else:
                raise AssertionError("duplicate id accepted")
        assert not any(p.tx_dir.iterdir())
        assert p.verify() == []


def test_lock():
    with Project.open(TMP / "proj"):
        try:
            Project.open(TMP / "proj")
        except ProjectError:
            pass
        else:
            raise AssertionError("second open succeeded under the lock")
    # a stale lock from a dead pid is cleared
    (TMP / "proj" / ".lock").write_text("999999999")
    with Project.open(TMP / "proj") as p:
        assert p.verify() == []


def test_orphan_reported():
    with Project.open(TMP / "proj") as p:
        (p.root / "videos" / "000555.mp4").write_bytes(b"x")
        problems = p.verify()
        assert any("000555" in m for m in problems), problems
        (p.root / "videos" / "000555.mp4").unlink()


def main():
    fresh()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    order = [test_add_commit, test_crash_before_records,
             test_failed_rename_rolls_back, test_crash_after_records,
             test_map_prune, test_remove_undo_purge,
             test_remove_rosters_and_ids, test_crash_during_remove,
             test_duplicate_id_refused, test_lock, test_orphan_reported]
    assert set(order) == set(tests)
    for t in order:
        t()
        print(f"ok  {t.__name__}")
    shutil.rmtree(TMP)
    print(f"{len(order)} passed")


if __name__ == "__main__":
    main()
