"""The import command on a folder of real hand-picked pairs.

Run: python tests/test_import.py <folder-with-pairs> [<folder-with-a-duplicate-of-one>]

The first folder needs at least two pairs; a video without a script beside
it is fine and is expected to be reported, not imported. The optional second
folder holds a pair whose script duplicates one of the first folder's, to
check refusal. Everything happens in a throwaway project under tests/.tmp.
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from goblintrain import importer, media, tx                  # noqa: E402
from goblintrain.project import Project, ProjectError        # noqa: E402

TMP = HERE / ".tmp" / "import"


def main(folder, dup_folder=None):
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    root = TMP / "proj"
    Project.create(root)
    lines = []
    log = lines.append

    # --- a folder import, unattended
    with Project.open(root) as p:
        pairs, n_video, n_script, missing = importer.find_pairs([folder])
        assert len(pairs) >= 2, "the folder needs at least two pairs"
        code = importer.run(p, [folder], yes=True, log=log)
        assert code == 0, "\n".join(lines)
        ids = p.ids()
        assert len(ids) == len(pairs), (ids, len(pairs))
        assert ids == [f"{i:06d}" for i in range(1, len(pairs) + 1)]
        for i in ids:
            assert p.video_path(i).is_file()
            acts = json.loads(p.script_path(i).read_text("utf-8"))
            assert acts and all(set(a) == {"at", "pos"} for a in acts)
            meta = json.loads((root / "meta" / f"{i}.json").read_text("utf-8"))
            assert meta["id"] == i and meta["n_actions"] == len(acts)
            assert media.probe_duration_ms(p.video_path(i))
        assert p.verify() == []
        assert not any(p.tx_dir.iterdir())
        assert len(p.map_sources()) == len(ids)
        if n_video:
            assert any("without a script" in ln for ln in lines), lines
    print(f"ok  folder import ({len(ids)} pairs)")
    for ln in lines:
        print("    " + ln)

    # --- the same folder again: every pair is a duplicate, nothing happens
    lines.clear()
    with Project.open(root) as p:
        before = p.ids()
        code = importer.run(p, [folder], yes=True, log=log)
        assert code == 1
        assert p.ids() == before
        refusals = [ln for ln in lines if ": refused," in ln]
        assert len(refusals) == len(before), lines
        assert all("duplicate of clip" in ln for ln in refusals), refusals
    print("ok  re-import refused as duplicates")

    # --- a duplicate in another folder, by content
    if dup_folder:
        lines.clear()
        with Project.open(root) as p:
            before = p.ids()
            code = importer.run(p, [dup_folder], yes=True, log=log)
            assert code == 1 and p.ids() == before
            assert any("duplicate of clip" in ln for ln in lines), lines
        print("ok  cross-folder duplicate refused")

    # --- an interrupted import leaves the project unchanged
    with Project.open(root) as p:
        before = p.ids()
        first = p.ids()[0]
        tx.remove_clip(p, first)
        tx.purge(p)
        pairs, *_ = importer.find_pairs([folder])
        importer.preflight(p, pairs)
        pr = next(q for q in pairs if not q.refusal)

        def kill(frac):
            raise KeyboardInterrupt

        try:
            with tx.Transaction(p, "add", [p.next_id()]) as t:
                media.transcode(pr.video, t.path("videos", "x.mp4"),
                                p.config["transcode"], pr.duration_ms, kill)
                raise AssertionError("transcode was not interrupted")
        except KeyboardInterrupt:
            pass
        assert not any(p.tx_dir.iterdir())
        assert p.ids() == before[1:]
        assert p.verify() == []
    print("ok  interrupted import leaves no trace")

    # --- the pair comes back through a single-pair import
    lines.clear()
    with Project.open(root) as p:
        code = importer.run(p, [str(pr.video), str(pr.script)], yes=True, log=log)
        assert code == 0, lines
        assert len(p.ids()) == len(before)
        assert p.verify() == []
    print("ok  explicit pair import")

    # --- a refusal without --continue imports nothing
    lines.clear()
    with Project.open(root) as p:
        tx.remove_clip(p, p.ids()[0])
        before = p.ids()
        code = importer.run(p, [folder], yes=True, log=log)
        assert code == 1 and p.ids() == before
        assert any("a refusal stops the batch" in ln for ln in lines), lines
        code = importer.run(p, [folder], yes=True, keep_going=True, log=log)
        assert code == 0 and len(p.ids()) == len(before) + 1
        assert p.verify() == []
    print("ok  refusal stops the batch unless --continue")

    # --- a prompt answered no imports nothing
    with Project.open(root) as p:
        tx.remove_clip(p, p.ids()[0])
        before = p.ids()
        code = importer.run(p, [folder], keep_going=True, log=lambda s: None,
                            ask=lambda q: "n")
        assert code == 1 and p.ids() == before
    print("ok  declined prompt imports nothing")

    shutil.rmtree(TMP)
    print("all passed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(*sys.argv[1:3])
