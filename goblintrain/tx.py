"""Transactions: every change to a project completes or leaves no trace.

A transaction stages its work under ``<project>/.tx/<txid>/`` and commits
it with renames on the same volume. The manifest and the private map are
written last, each atomically. A journal (``tx.json``) in the staging
directory records the state and the planned moves, so that a process that
dies at any point can be cleaned up by the next command: :func:`recover`
rolls an unfinished commit back, or finishes bookkeeping for one whose
records already landed.

Two kinds:

``add``
    Files are staged, then moved into the project; ``records`` are
    appended to the manifest and ``sources`` to the map. Done once every
    new ID is in the manifest.

``remove``
    A clip's files are moved into ``<project>/.trash/<txid>/`` with a
    ``trash.json`` beside them that holds what ``--undo`` needs; the IDs
    leave the manifest and the map. Done once no removed ID is in the
    manifest.

The done-condition is what recovery reads, so a crash between the file
moves and the manifest write rolls back, and a crash after the manifest
write only cleans up.
"""
import json
import os
import secrets
import shutil
import time
from pathlib import Path

from .project import ProjectError

JOURNAL = "tx.json"
TRASH_RECORD = "trash.json"


def _new_txid():
    """Sorts in time order: date-time to the nanosecond, then pid and a
    random tail against two processes in the same instant."""
    ns = time.time_ns()
    return (time.strftime("%Y%m%dT%H%M%S", time.localtime(ns // 10**9))
            + f".{ns % 10**9:09d}-{os.getpid()}-{secrets.token_hex(3)}")


def _rel(project, path):
    return Path(path).resolve().relative_to(project.root.resolve()).as_posix()


class Transaction:
    def __init__(self, project, kind, ids):
        if kind not in ("add", "remove"):
            raise ValueError(kind)
        self.project = project
        self.kind = kind
        self.ids = list(ids)
        self.id = _new_txid()
        self.dir = project.tx_dir / self.id
        self.dir.mkdir(parents=True)
        self._journal({"state": "staging"})
        self.done = False
        self.committing = False

    def _journal(self, extra):
        rec = {"txid": self.id, "kind": self.kind, "ids": self.ids}
        rec.update(extra)
        from .project import atomic_write_text
        atomic_write_text(self.dir / JOURNAL, json.dumps(rec, indent=1))

    def path(self, *parts):
        """A staging path inside the transaction directory."""
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def trash_path(self, *parts):
        """Where a removed file goes: ``.trash/<txid>/<parts>``."""
        p = self.project.trash_dir / self.id
        p = p.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def abort(self):
        """Discard the staging directory. A no-op after commit, and after a
        commit that failed in the middle: that one has been rolled back
        already, or its journal is what the next recovery reads."""
        if not self.done and not self.committing and self.dir.exists():
            shutil.rmtree(self.dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.abort()
        return False

    def commit(self, moves, records=(), sources=None, trash_record=None):
        """Perform ``moves`` ((src, dst) pairs, all on the project's volume),
        then the record updates the kind implies, then clean up.

        ``records``: manifest records to append (add) or nothing (remove).
        ``sources``: ``{id: original source}`` for the map (add).
        ``trash_record``: what ``--undo`` needs, written beside the trashed
        files (remove).
        """
        project = self.project
        moves = [(Path(s), Path(d)) for s, d in moves]
        for s, d in moves:
            if not s.exists():
                raise ProjectError(f"transaction {self.id}: staged file "
                                   f"{_rel(project, s)} is missing")
            if d.exists():
                raise ProjectError(f"transaction {self.id}: "
                                   f"{_rel(project, d)} already exists")
        current = project.manifest()
        have = {r["id"] for r in current}
        if self.kind == "add":
            records = list(records)
            if [r["id"] for r in records] != self.ids:
                raise ValueError("records do not match the transaction's ids")
            clash = have & set(self.ids)
            if clash:
                raise ProjectError(f"transaction {self.id}: id(s) "
                                   f"{sorted(clash)} already in the manifest")
        else:
            missing = set(self.ids) - have
            if missing:
                raise ProjectError(f"transaction {self.id}: id(s) "
                                   f"{sorted(missing)} not in the manifest")
            if trash_record is not None:
                from .project import atomic_write_text
                atomic_write_text(self.trash_path(TRASH_RECORD),
                                  json.dumps(trash_record, indent=1))

        self.committing = True
        self._journal({"state": "committing",
                       "moves": [[_rel(project, s), _rel(project, d)]
                                 for s, d in moves]})
        try:
            for s, d in moves:
                d.parent.mkdir(parents=True, exist_ok=True)
                os.rename(s, d)
        except BaseException:
            # A rename failed or the user interrupted: the journal names
            # every move, so roll back now rather than at the next open.
            recover(project)
            raise

        if self.kind == "add":
            if sources:
                lines = project.map_lines()
                lines += [f"{i}\t{sources[i]}" for i in self.ids if i in sources]
                project.write_map(lines)
            project.write_manifest(current + records)
        else:
            drop = set(self.ids)
            project.write_manifest([r for r in current if r["id"] not in drop])
            project.write_map([ln for ln in project.map_lines()
                               if ln.partition("\t")[0] not in drop])
        # Records landed: the transaction is complete whatever happens now.
        self.done = True
        shutil.rmtree(self.dir)


def _is_done(project, journal):
    have = set(project.ids())
    ids = set(journal.get("ids", []))
    if journal.get("kind") == "add":
        return ids <= have
    return not (ids & have)


def recover(project):
    """Clean up after any transaction that did not finish. Returns the
    messages describing what was done, one per transaction touched."""
    notes = []
    if not project.tx_dir.is_dir():
        return notes
    for d in sorted(project.tx_dir.iterdir()):
        if not d.is_dir():
            d.unlink()
            continue
        try:
            journal = json.loads((d / JOURNAL).read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            journal = None
        if journal is None or journal.get("state") == "staging":
            shutil.rmtree(d)
            notes.append(f"{d.name}: discarded unfinished staging")
            continue
        if _is_done(project, journal):
            shutil.rmtree(d)
            notes.append(f"{d.name}: finished cleanup of a committed "
                         f"{journal.get('kind')}")
            continue
        # Committing, but the records never landed: put every file back, in
        # reverse order, so a file merged into a moved directory leaves it
        # before the directory itself goes back.
        for src, dst in reversed(journal.get("moves", [])):
            s, t = project.root / src, project.root / dst
            if t.exists() and not s.exists():
                s.parent.mkdir(parents=True, exist_ok=True)
                os.rename(t, s)
        if journal.get("kind") == "remove":
            trash = project.trash_dir / journal["txid"]
            if trash.exists():
                shutil.rmtree(trash)
        shutil.rmtree(d)
        notes.append(f"{d.name}: rolled back an interrupted "
                     f"{journal.get('kind')} of {journal.get('ids')}")
    # The map never names an ID the manifest does not.
    have = set(project.ids())
    lines = project.map_lines()
    kept = [ln for ln in lines if ln.partition("\t")[0] in have]
    if len(kept) != len(lines):
        project.write_map(kept)
        notes.append(f"map: dropped {len(lines) - len(kept)} entr"
                     f"{'y' if len(lines) - len(kept) == 1 else 'ies'} "
                     "without a manifest record")
    return notes


def remove_clip(project, clip_id):
    """Move a clip and everything derived from it to the trash and take it
    out of the records. Returns the trash directory."""
    if clip_id not in set(project.ids()):
        raise ProjectError(f"{clip_id} is not in the manifest")
    record = next(r for r in project.manifest() if r["id"] == clip_id)
    map_line = next((ln for ln in project.map_lines()
                     if ln.partition("\t")[0] == clip_id), None)
    if not record.get("negative"):
        project.retire_id(clip_id)
    with Transaction(project, "remove", [clip_id]) as tx:
        files = project.clip_files(clip_id)
        moves = [(f, tx.trash_path(_rel(project, f))) for f in files]
        tx.commit(moves, trash_record={
            "id": clip_id, "record": record, "map_line": map_line,
            "files": [_rel(project, f) for f in files]})
        return project.trash_dir / tx.id


def last_trash(project):
    """The most recent trash directory with a record, or None."""
    if not project.trash_dir.is_dir():
        return None
    dirs = sorted((d for d in project.trash_dir.iterdir()
                   if (d / TRASH_RECORD).is_file()), key=lambda d: d.name)
    return dirs[-1] if dirs else None


def undo_remove(project, trash):
    """Put a removed clip back from its trash directory."""
    rec = json.loads((trash / TRASH_RECORD).read_text("utf-8"))
    clip_id = rec["id"]
    if clip_id in set(project.ids()):
        raise ProjectError(f"{clip_id} is already in the manifest; the "
                           "trash entry was not restored")
    with Transaction(project, "add", [clip_id]) as tx:
        moves = [(trash / f, project.root / f) for f in rec["files"]]
        sources = {}
        if rec.get("map_line"):
            sources[clip_id] = rec["map_line"].partition("\t")[2]
        tx.commit(moves, records=[rec["record"]], sources=sources)
    shutil.rmtree(trash)
    return clip_id


def purge(project):
    """Empty the trash. Returns the number of entries removed."""
    n = 0
    if project.trash_dir.is_dir():
        for d in list(project.trash_dir.iterdir()):
            shutil.rmtree(d) if d.is_dir() else d.unlink()
            n += 1
    return n
