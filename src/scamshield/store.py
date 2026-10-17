"""SQLite (WAL) storage: decisions, encrypted payment payloads, explanation notes, analyst cases.

Only pseudonymised tokens and model outputs are stored in the clear. The raw payment (account numbers,
bank codes) exists only as AES-256-GCM ciphertext, bound to its payment token as associated data.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .audit import AuditLog
from .security import Envelope, Sealed

SCHEMA = """
CREATE TABLE IF NOT EXISTS payments (
    payment_token TEXT PRIMARY KEY,
    sender_token TEXT NOT NULL,
    receiver_token TEXT NOT NULL,
    received_at TEXT NOT NULL,
    paid_at TEXT,
    amount_usd REAL NOT NULL,
    payment_format TEXT NOT NULL,
    action TEXT NOT NULL,
    anomaly REAL NOT NULL,
    risk REAL NOT NULL,
    rules TEXT NOT NULL,
    reasons TEXT NOT NULL,
    model_ms REAL,
    backend TEXT,
    key_id TEXT NOT NULL,
    nonce BLOB NOT NULL,
    ciphertext BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS payments_action ON payments(action);
CREATE INDEX IF NOT EXISTS payments_sender ON payments(sender_token);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    payment_token TEXT NOT NULL REFERENCES payments(payment_token),
    kind TEXT NOT NULL CHECK (kind IN ('warning', 'brief')),
    status TEXT NOT NULL CHECK (status IN ('pending', 'working', 'done', 'failed')),
    source TEXT NOT NULL,
    text TEXT NOT NULL,
    facts TEXT NOT NULL,
    model TEXT,
    gen_ms REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_status ON notes(status, id);
CREATE INDEX IF NOT EXISTS notes_updated ON notes(updated_at);

CREATE TABLE IF NOT EXISTS cases (
    payment_token TEXT PRIMARY KEY REFERENCES payments(payment_token),
    status TEXT NOT NULL CHECK (status IN ('open', 'released', 'stopped', 'contact_customer')),
    resolved_by TEXT,
    resolved_at TEXT,
    comment TEXT
);
"""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level="DEFERRED", timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@dataclass
class PaymentRecord:
    payment_token: str
    sender_token: str
    receiver_token: str
    paid_at: str | None
    amount_usd: float
    payment_format: str
    action: str
    anomaly: float
    risk: float
    rules: list[str]
    reasons: dict[str, Any]
    model_ms: float
    backend: str
    payload: dict[str, Any]             # raw payment incl. account numbers -> encrypted
    notes: list[tuple[str, str, dict]]  # (kind, template text, facts) queued for Granite


class Store:
    def __init__(self, path: str | Path, envelope: Envelope, checkpoint_path: str | Path | None = None):
        self.conn = connect(path)
        self.lock = threading.RLock()
        self.envelope = envelope
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        self.audit = AuditLog(self.conn, self.lock, checkpoint_path=checkpoint_path)

    # ------------------------------------------------------------------ writes
    def save_batch(self, records: Iterable[PaymentRecord], actor: str = "scorer") -> None:
        """Payments, their notes, their cases and their audit entries in ONE transaction."""
        ts = now()
        with self.lock:
            try:
                for r in records:
                    sealed = self.envelope.seal_json(r.payload, aad=r.payment_token)
                    self.conn.execute(
                        "INSERT INTO payments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (r.payment_token, r.sender_token, r.receiver_token, ts, r.paid_at, r.amount_usd,
                         r.payment_format, r.action, r.anomaly, r.risk, ",".join(r.rules), json.dumps(r.reasons),
                         r.model_ms, r.backend, sealed.key_id, sealed.nonce, sealed.ciphertext))
                    for kind, text, facts in r.notes:
                        self.conn.execute(
                            "INSERT INTO notes (payment_token, kind, status, source, text, facts, created_at, updated_at)"
                            " VALUES (?,?,?,?,?,?,?,?)",
                            (r.payment_token, kind, "pending", "template", text, json.dumps(facts), ts, ts))
                    if r.action == "HOLD":
                        self.conn.execute("INSERT INTO cases (payment_token, status) VALUES (?, 'open')",
                                          (r.payment_token,))
                    self.audit.append(actor, "payment_scored", r.payment_token, {
                        "action": r.action, "risk": round(r.risk, 4), "anomaly": round(r.anomaly, 4),
                        "rules": r.rules, "backend": r.backend}, ts=ts, commit=False)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                self.audit.rolled_back()
                raise
            self.audit.committed()

    def resolve_case(self, payment_token: str, status: str, user: str, comment: str = "") -> bool:
        with self.lock:
            cur = self.conn.execute(
                "UPDATE cases SET status=?, resolved_by=?, resolved_at=?, comment=? WHERE payment_token=? AND status='open'",
                (status, user, now(), comment[:500], payment_token))
            if cur.rowcount == 0:  # unknown payment, or already resolved: a decision is never silently overwritten
                self.conn.rollback()
                return False
            self.audit.append(user, "case_resolved", payment_token, {"status": status}, commit=False)
            self.conn.commit()
            self.audit.committed()
            return True

    # ------------------------------------------------------------------ notes queue (Granite worker)
    def claim_pending_notes(self, limit: int = 1) -> list[dict[str, Any]]:
        """Atomically move up to `limit` pending notes to 'working'.

        Briefs (an analyst is waiting on a held payment) go first, then the NEWEST customer messages: on a
        small VM Granite writes about one note a minute, so the most recent payments get upgraded first and
        older ones keep their (accurate) template text.
        """
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, payment_token, kind, facts FROM notes WHERE status='pending' "
                "ORDER BY CASE kind WHEN 'brief' THEN 0 ELSE 1 END, id DESC LIMIT ?", (limit,)).fetchall()
            for r in rows:
                self.conn.execute("UPDATE notes SET status='working', updated_at=? WHERE id=?", (now(), r[0]))
            self.conn.commit()
        return [{"id": r[0], "payment_token": r[1], "kind": r[2], "facts": json.loads(r[3])} for r in rows]

    def complete_note(self, note_id: int, text: str | None, model: str, gen_ms: float, ok: bool) -> None:
        ok = bool(ok and text)
        with self.lock:
            if ok and text:
                self.conn.execute(
                    "UPDATE notes SET status='done', source='granite', text=?, model=?, gen_ms=?, updated_at=? WHERE id=?",
                    (text, model, gen_ms, now(), note_id))
            else:  # keep the template text so the UI always has something true to show
                self.conn.execute("UPDATE notes SET status='failed', model=?, gen_ms=?, updated_at=? WHERE id=?",
                                  (model, gen_ms, now(), note_id))
            self.audit.append("granite-worker", "note_generated" if ok else "note_failed", str(note_id),
                              {"model": model, "gen_ms": round(gen_ms, 1)}, commit=False)
            self.conn.commit()
            self.audit.committed()

    def requeue_stuck_notes(self) -> int:
        with self.lock:
            n = self.conn.execute("UPDATE notes SET status='pending' WHERE status='working'").rowcount
            self.conn.commit()
            return n

    def notes_updated_since(self, ts: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT n.id, n.payment_token, n.kind, n.status, n.source, n.text, n.updated_at, p.sender_token "
                "FROM notes n JOIN payments p USING (payment_token) WHERE n.updated_at > ? AND n.status IN ('done','failed') "
                "ORDER BY n.updated_at", (ts,)).fetchall()
        return [{"id": r[0], "payment_token": r[1], "kind": r[2], "status": r[3], "source": r[4], "text": r[5],
                 "updated_at": r[6], "sender_token": r[7]} for r in rows]

    # ------------------------------------------------------------------ reads
    def _notes_for(self, token: str) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT kind, status, source, text, model, gen_ms FROM notes WHERE payment_token=? ORDER BY id", (token,)
        ).fetchall()
        return {r[0]: {"status": r[1], "source": r[2], "text": r[3], "model": r[4], "gen_ms": r[5]} for r in rows}

    _KEYS = ["payment_token", "sender_token", "receiver_token", "received_at", "paid_at", "amount_usd",
             "payment_format", "action", "anomaly", "risk", "rules", "reasons", "model_ms", "backend"]
    _COLS = ", ".join(_KEYS)

    def _row(self, r) -> dict[str, Any]:
        d = dict(zip(self._KEYS, r))
        d["rules"] = [x for x in d["rules"].split(",") if x]
        d["reasons"] = json.loads(d["reasons"])
        return d

    def list_cases(self, status: str = "open", limit: int = 100) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                f"SELECT {self._COLS}, k.status FROM payments p JOIN cases k USING (payment_token) "
                "WHERE k.status=? ORDER BY p.risk DESC, p.received_at DESC LIMIT ?", (status, limit)).fetchall()
            out = []
            for r in rows:
                d = self._row(r[:-1])
                d["case_status"] = r[-1]
                d["notes"] = self._notes_for(d["payment_token"])
                out.append(d)
        return out

    def get_payment(self, token: str, decrypt: bool = False) -> dict[str, Any] | None:
        with self.lock:
            r = self.conn.execute(f"SELECT {self._COLS}, key_id, nonce, ciphertext FROM payments WHERE payment_token=?",
                                  (token,)).fetchone()
            if r is None:
                return None
            d = self._row(r[:-3])
            d["notes"] = self._notes_for(token)
            k = self.conn.execute("SELECT status, resolved_by, resolved_at, comment FROM cases WHERE payment_token=?",
                                  (token,)).fetchone()
        d["case"] = dict(zip(["status", "resolved_by", "resolved_at", "comment"], k)) if k else None
        if decrypt:
            d["payload"] = self.envelope.open_json(Sealed(r[-3], r[-2], r[-1]), aad=token)
        return d

    def iter_payloads(self, chunk: int = 5000):
        """Decrypted raw payments in arrival order (for rebuilding in-memory state after a restart)."""
        last = 0
        while True:
            with self.lock:
                rows = self.conn.execute("SELECT rowid, payment_token, key_id, nonce, ciphertext FROM payments "
                                         "WHERE rowid > ? ORDER BY rowid LIMIT ?", (last, chunk)).fetchall()
            if not rows:
                return
            for rowid, token, kid, nonce, ct in rows:
                yield self.envelope.open_json(Sealed(kid, nonce, ct), aad=token)
            last = rows[-1][0]

    def customer_payments(self, sender_token: str, limit: int = 100) -> list[dict[str, Any]]:
        """What a customer may see: their own warned/held payments and the message for them (never the brief)."""
        with self.lock:
            rows = self.conn.execute(
                f"SELECT {self._COLS} FROM payments WHERE sender_token=? AND action!='ALLOW' "
                "ORDER BY received_at DESC LIMIT ?", (sender_token, limit)).fetchall()
            out = []
            for r in rows:
                d = self._row(r)
                notes = self._notes_for(d["payment_token"])
                # minimal disclosure: why we're checking, never risk scores or the analyst's brief
                out.append({
                    "payment_token": d["payment_token"], "paid_at": d["paid_at"], "amount_usd": d["amount_usd"],
                    "payment_format": d["payment_format"], "action": d["action"], "received_at": d["received_at"],
                    "message": notes.get("warning"),
                    "reasons": {"rules": d["reasons"].get("rules", []), "customer": d["reasons"].get("customer", [])},
                })
        return out

    def aggregates(self) -> dict[str, Any]:
        """Auditor view: counts and sums only, no identifiers."""
        with self.lock:
            mix = dict(self.conn.execute("SELECT action, COUNT(*) FROM payments GROUP BY action").fetchall())
            money = dict(self.conn.execute("SELECT action, SUM(amount_usd) FROM payments GROUP BY action").fetchall())
            cases = dict(self.conn.execute("SELECT status, COUNT(*) FROM cases GROUP BY status").fetchall())
            notes = dict(self.conn.execute("SELECT source, COUNT(*) FROM notes GROUP BY source").fetchall())
            backends = dict(self.conn.execute("SELECT backend, COUNT(*) FROM payments GROUP BY backend").fetchall())
        return {"decisions": mix, "amount_usd_by_decision": money, "cases": cases, "notes_by_source": notes,
                "backends": backends, "total_payments": sum(mix.values())}
