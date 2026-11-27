"""A durable store behind the write-ahead action protocol.

``state.protocol.ProtocolLedger`` states its durability requirement as an
interface: ``prepare`` must return only after the intent is stable.  Before
this revision nothing in the artifact implemented that interface -- E13's
crash study was a simulation over the state machine.  This module is the
implementation, and E20 kills real processes against it.

**Where the state lives.**  Each runtime that dispatches side effects (the
source before a handoff, the target after it, a delegate while executing a
delegated step) owns one SQLite database file on its own local storage.  The
ledger of record for an action is the store of the runtime that issued its
intent.  A handoff block carries a *snapshot* of that ledger; it is never the
authority for an action still in flight (see ``snapshot_for_handoff``).

**Minimum durability semantics** (what the protocol needs, and what this
store configures):

  D1  the PREPARED intent -- fingerprint, deterministic key, epoch -- is
      committed durably *before* the request leaves the process: one SQLite
      transaction in WAL mode with ``synchronous=FULL`` (the WAL is synced at
      every commit) and, on macOS, ``fullfsync`` so the sync reaches stable
      storage rather than the drive cache;
  D2  COMMITTED/FAILED is written after the response is observed; losing that
      write is safe, because recovery maps a surviving PREPARED to UNKNOWN;
  D3  the epoch bump that fences the source is committed durably *before* the
      snapshot is taken, so no dispatch can begin after the snapshot;
  D4  a snapshot ships every non-terminal record as UNKNOWN, never as absent
      and never as COMMITTED.

What is tested (E20): process crashes by SIGKILL at every protocol boundary.
What is not: power loss or kernel crash -- D1 relies on fsync/F_FULLFSYNC
semantics that we configure but did not exercise by cutting power.
"""
import json
import os
import sqlite3
import time
from typing import Dict, List, Optional

from state.protocol import ActionRecord, ProtocolLedger, Status

_SCHEMA = """
CREATE TABLE IF NOT EXISTS actions (
  fp TEXT PRIMARY KEY, family TEXT, schema_version TEXT, key TEXT,
  status TEXT, epoch INTEGER, attempts INTEGER, head TEXT, step INTEGER,
  agent TEXT, upstream_key INTEGER, evidence TEXT, owner TEXT,
  deleg_id TEXT, updated_ns INTEGER);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS log (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, op TEXT, fp TEXT, status TEXT,
  epoch INTEGER, t_ns INTEGER);
"""

_COLS = ("fp", "family", "schema_version", "key", "status", "epoch",
         "attempts", "head", "step", "agent", "upstream_key", "evidence",
         "owner", "deleg_id")


class SQLiteLedger(ProtocolLedger):
    """``ProtocolLedger`` whose every state transition is a durable commit."""

    def __init__(self, path: str, task_secret: str = "task", epoch: int = 0,
                 synchronous: str = "FULL", fullfsync: bool = True):
        super().__init__(task_secret=task_secret, epoch=epoch)
        self.path = path
        self.db = sqlite3.connect(path, isolation_level=None, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(f"PRAGMA synchronous={synchronous}")
        if fullfsync:
            self.db.execute("PRAGMA fullfsync=ON")
            self.db.execute("PRAGMA checkpoint_fullfsync=ON")
        self.db.executescript(_SCHEMA)
        self.config = {
            "journal_mode": self.db.execute("PRAGMA journal_mode").fetchone()[0],
            "synchronous": self.db.execute("PRAGMA synchronous").fetchone()[0],
            "fullfsync": self.db.execute("PRAGMA fullfsync").fetchone()[0]}
        row = self.db.execute("SELECT v FROM meta WHERE k='epoch'").fetchone()
        if row is None:
            self._meta("epoch", str(epoch))
        else:
            self.epoch = int(row[0])
        self._load()

    # -- persistence ------------------------------------------------------
    def _meta(self, k, v):
        self.db.execute("BEGIN IMMEDIATE")
        self.db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)",
                        (k, v))
        self.db.execute("COMMIT")

    def _load(self):
        self.records = {}
        for row in self.db.execute(
                "SELECT " + ",".join(_COLS) + " FROM actions"):
            d = dict(zip(_COLS, row))
            d["upstream_key"] = bool(d["upstream_key"])
            d["status"] = Status(d["status"])
            self.records[d["fp"]] = ActionRecord(**d)

    def _put(self, r: ActionRecord, op: str):
        """One durable transaction per transition (D1/D2)."""
        now = time.time_ns()
        vals = [getattr(r, c) for c in _COLS]
        vals[_COLS.index("status")] = r.status.value
        vals[_COLS.index("upstream_key")] = int(r.upstream_key)
        self.db.execute("BEGIN IMMEDIATE")
        self.db.execute(
            "INSERT OR REPLACE INTO actions(" + ",".join(_COLS)
            + ",updated_ns) VALUES (" + ",".join("?" * len(_COLS)) + ",?)",
            vals + [now])
        self.db.execute("INSERT INTO log(op, fp, status, epoch, t_ns) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (op, r.fp, r.status.value, self.epoch, now))
        self.db.execute("COMMIT")

    # -- write path (each returns only after the commit) ------------------
    def prepare(self, fp, signature, **kw) -> ActionRecord:
        r = super().prepare(fp, signature, **kw)
        self._put(r, "prepare")
        return r

    def commit(self, fp, evidence=""):
        super().commit(fp, evidence)
        if fp in self.records:
            self._put(self.records[fp], "commit")

    def fail(self, fp, evidence=""):
        super().fail(fp, evidence)
        if fp in self.records:
            self._put(self.records[fp], "fail")

    def mark_unknown(self, fp, evidence=""):
        before = self.records.get(fp)
        was = before.status if before else None
        super().mark_unknown(fp, evidence)
        if fp in self.records and self.records[fp].status is not was:
            self._put(self.records[fp], "unknown")

    def compensate(self, fp, evidence=""):
        super().compensate(fp, evidence)
        if fp in self.records:
            self._put(self.records[fp], "compensate")

    def put_record(self, r: ActionRecord, op: str = "merge"):
        """Install a record received from elsewhere (handoff / delegation)."""
        self.records[r.fp] = r
        self._put(r, op)

    def recover(self) -> List[ActionRecord]:
        out = super().recover()
        for r in out:
            self._put(r, "recover")
        return out

    def bump_epoch(self) -> int:
        """D3: the fencing epoch is durable before anything else happens."""
        e = super().bump_epoch()
        self._meta("epoch", str(e))
        return e

    def set_epoch(self, e: int):
        self.epoch = e
        self._meta("epoch", str(e))

    def fence(self, new_epoch: int) -> int:
        """D3: durably record that the task has moved to ``new_epoch``.

        Written before the handoff snapshot is taken.  From then on this
        runtime passes ``fenced_epoch()`` as the observed epoch on every
        dispatch, so the dispatcher blocks it as a stale writer.
        """
        self._meta("fenced_epoch", str(new_epoch))
        return new_epoch

    def fenced_epoch(self):
        row = self.db.execute(
            "SELECT v FROM meta WHERE k='fenced_epoch'").fetchone()
        return int(row[0]) if row else None

    # -- handoff ---------------------------------------------------------
    def snapshot_for_handoff(self) -> List[Dict]:
        """D4: non-terminal records ship as UNKNOWN, never absent/COMMITTED.

        A delegated record keeps its owner and ticket so the receiver can
        still accept the delegate's result.
        """
        out = []
        for r in self.entries():
            d = r.to_dict()
            if r.status in (Status.PREPARED, Status.INTENDED) and not r.owner:
                d["status"] = Status.UNKNOWN.value
            out.append(d)
        return out

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass


def load_snapshot(ledger: SQLiteLedger, records: List[Dict]) -> int:
    """Install a received snapshot into a (fresh) target ledger."""
    n = 0
    for d in records:
        ledger.put_record(ActionRecord.from_dict(d), "handoff")
        n += 1
    return n
