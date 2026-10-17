"""Tamper-evident, append-only audit log in SQLite.

Each row stores hash = sha256(prev_hash + canonical_json(row)). Editing, inserting, reordering or
deleting a row in the middle breaks the chain from that point on, and verify() names the first broken
row. SQLite triggers also refuse UPDATE and DELETE, so tampering needs a deliberate bypass.

A chain on its own cannot notice that its newest rows were cut off (a shorter chain is still a valid
chain), so verify() also checks an external anchor: the head (id, hash) this process last wrote, and
the last line of an append-only checkpoint file kept outside the database. Truncation, or rows added
by someone other than this log, then show up as an anchor mismatch.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    subject TEXT,
    details TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""


def canonical(row: dict) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def row_hash(prev_hash: str, row: dict) -> str:
    return hashlib.sha256((prev_hash + canonical(row)).encode()).hexdigest()


@dataclass
class VerifyResult:
    ok: bool
    checked: int
    first_bad_id: int | None = None
    reason: str = ""


class AuditLog:
    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock | None = None,
                 checkpoint_path: str | Path | None = None, checkpoint_every: int = 100):
        self.conn = conn
        self.lock = lock or threading.Lock()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_every = checkpoint_every
        self._head: tuple[int, str] | None = None  # (id, hash) this process last COMMITTED
        self._pending: tuple[int, str] | None = None  # written inside a transaction not yet committed
        self._last_checkpoint_id = 0
        with self.lock:
            conn.executescript(SCHEMA)
            conn.commit()

    def _last_hash(self) -> str:
        r = self.conn.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
        return r[0] if r else GENESIS

    def checkpoint(self) -> tuple[int, str] | None:
        """Append the current head to the checkpoint file (outside the database)."""
        if self.checkpoint_path is None:
            return None
        with self.lock:
            r = self.conn.execute("SELECT id, hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
        if r is None:
            return None
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "a") as f:
            f.write(f"{r[0]} {r[1]} {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
        return int(r[0]), r[1]

    def _last_checkpoint(self) -> tuple[int, str] | None:
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return None
        lines = [ln for ln in self.checkpoint_path.read_text().splitlines() if ln.strip()]
        if not lines:
            return None
        rid, h = lines[-1].split()[:2]
        return int(rid), h

    def append(self, actor: str, action: str, subject: str | None = None, details: dict | None = None,
               ts: str | None = None, commit: bool = True) -> str:
        """Add one entry. Never put raw PII in `details`; use pseudonymised tokens."""
        with self.lock:
            prev = self._last_hash()
            row = {
                "ts": ts or dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds"),
                "actor": actor,
                "action": action,
                "subject": subject,
                "details": canonical(details or {}),
            }
            h = row_hash(prev, row)
            cur = self.conn.execute(
                "INSERT INTO audit (ts, actor, action, subject, details, prev_hash, hash) VALUES (?,?,?,?,?,?,?)",
                (row["ts"], actor, action, subject, row["details"], prev, h),
            )
            self._pending = (int(cur.lastrowid), h)
            if commit:
                self.conn.commit()
                self.committed()
        return h

    def committed(self) -> None:
        """Call after committing a transaction that contained append(..., commit=False) entries."""
        if self._pending is not None:
            self._head, self._pending = self._pending, None
        self.maybe_checkpoint()

    def rolled_back(self) -> None:
        """Call after rolling back: entries from that transaction no longer exist, so forget them."""
        self._pending = None

    def maybe_checkpoint(self) -> None:
        """Write a checkpoint when checkpoint_every entries were added since the last one. Call after commit."""
        if self.checkpoint_path is None or self._head is None:
            return
        if self._head[0] - self._last_checkpoint_id >= self.checkpoint_every:
            cp = self.checkpoint()
            if cp:
                self._last_checkpoint_id = cp[0]

    def verify(self) -> VerifyResult:
        prev = GENESIS
        n = 0
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, ts, actor, action, subject, details, prev_hash, hash FROM audit ORDER BY id"
            ).fetchall()
        by_id: dict[int, str] = {}
        for rid, ts, actor, action, subject, details, prev_hash, h in rows:
            n += 1
            if prev_hash != prev:
                return VerifyResult(False, n, rid, "previous-hash link broken (row deleted, inserted or reordered)")
            expect = row_hash(prev, {"ts": ts, "actor": actor, "action": action, "subject": subject,
                                     "details": details})
            if expect != h:
                return VerifyResult(False, n, rid, "row content does not match its hash (row edited)")
            by_id[rid] = h
            prev = h
        for name, anchor in (("this process's last write", self._head), ("checkpoint file", self._last_checkpoint())):
            if anchor is None:
                continue
            aid, ah = anchor
            if by_id.get(aid) != ah:
                return VerifyResult(False, n, aid, f"anchored row missing or changed ({name}): log was truncated")
        return VerifyResult(True, n)

    def tail(self, limit: int = 50) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, ts, actor, action, subject, details, hash FROM audit ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {"id": r[0], "ts": r[1], "actor": r[2], "action": r[3], "subject": r[4],
             "details": json.loads(r[5]), "hash": r[6]}
            for r in rows
        ]
