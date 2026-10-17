"""SQLite (WAL) storage: decisions, encrypted claim payloads, explanation notes, investigator cases.

Only pseudonymised tokens and model outputs are stored in the clear. The raw claim (with real
identifiers) exists only as AES-256-GCM ciphertext, bound to its claim token as associated data.
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
CREATE TABLE IF NOT EXISTS claims (
    claim_token TEXT PRIMARY KEY,
    provider_token TEXT NOT NULL,
    bene_token TEXT NOT NULL,
    received_at TEXT NOT NULL,
    service_date TEXT,
    amount REAL NOT NULL,
    is_inpatient INTEGER NOT NULL,
    action TEXT NOT NULL,
    anomaly REAL NOT NULL,
    provider_risk REAL,
    provider_claims INTEGER NOT NULL,
    rules TEXT NOT NULL,
    reasons TEXT NOT NULL,
    model_ms REAL,
    backend TEXT,
    key_id TEXT NOT NULL,
    nonce BLOB NOT NULL,
    ciphertext BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS claims_action ON claims(action);
CREATE INDEX IF NOT EXISTS claims_provider ON claims(provider_token);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_token TEXT NOT NULL REFERENCES claims(claim_token),
    kind TEXT NOT NULL CHECK (kind IN ('correction', 'brief')),
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
    claim_token TEXT PRIMARY KEY REFERENCES claims(claim_token),
    status TEXT NOT NULL CHECK (status IN ('open', 'released', 'confirmed_issue', 'needs_more_info')),
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
class ClaimRecord:
    claim_token: str
    provider_token: str
    bene_token: str
    service_date: str | None
    amount: float
    is_inpatient: int
    action: str
    anomaly: float
    provider_risk: float | None
    provider_claims: int
    rules: list[str]
    reasons: dict[str, Any]
    model_ms: float
    backend: str
    payload: dict[str, Any]          # raw claim incl. real identifiers -> encrypted
    notes: list[tuple[str, str, dict]]  # (kind, template text, facts) queued for Granite


class Store:
    def __init__(self, path: str | Path, envelope: Envelope, checkpoint_path: str | Path | None = None):
        self.conn = connect(path)
        self.lock = threading.RLock()
        self.envelope = envelope
        with self.lock:
            self.conn.executescript(SCHEMA)
            self._migrate()
            self.conn.commit()
        self.audit = AuditLog(self.conn, self.lock, checkpoint_path=checkpoint_path)

    def _migrate(self) -> None:
        """Bring databases created by earlier versions up to the current schema."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(claims)")}
        if "latency_ms" in cols and "model_ms" not in cols:  # v0.1: column held model time under a wrong name
            self.conn.execute("ALTER TABLE claims RENAME COLUMN latency_ms TO model_ms")

    # ------------------------------------------------------------------ writes
    def save_batch(self, records: Iterable[ClaimRecord], actor: str = "scorer") -> None:
        """Claims, their notes, their cases and their audit entries in ONE transaction."""
        ts = now()
        with self.lock:
            try:
                for r in records:
                    sealed = self.envelope.seal_json(r.payload, aad=r.claim_token)
                    self.conn.execute(
                        "INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (r.claim_token, r.provider_token, r.bene_token, ts, r.service_date, r.amount, r.is_inpatient,
                         r.action, r.anomaly, r.provider_risk, r.provider_claims, ",".join(r.rules),
                         json.dumps(r.reasons), r.model_ms, r.backend, sealed.key_id, sealed.nonce,
                         sealed.ciphertext))
                    for kind, text, facts in r.notes:
                        self.conn.execute(
                            "INSERT INTO notes (claim_token, kind, status, source, text, facts, created_at, updated_at)"
                            " VALUES (?,?,?,?,?,?,?,?)",
                            (r.claim_token, kind, "pending", "template", text, json.dumps(facts), ts, ts))
                    if r.action == "HOLD_FOR_REVIEW":
                        self.conn.execute("INSERT INTO cases (claim_token, status) VALUES (?, 'open')", (r.claim_token,))
                    self.audit.append(actor, "claim_scored", r.claim_token, {
                        "action": r.action, "provider": r.provider_token, "anomaly": round(r.anomaly, 4),
                        "provider_risk": None if r.provider_risk is None else round(r.provider_risk, 4),
                        "rules": r.rules, "backend": r.backend}, ts=ts, commit=False)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                self.audit.rolled_back()
                raise
            self.audit.committed()

    def resolve_case(self, claim_token: str, status: str, user: str, comment: str = "") -> bool:
        with self.lock:
            cur = self.conn.execute(
                "UPDATE cases SET status=?, resolved_by=?, resolved_at=?, comment=? WHERE claim_token=? AND status='open'",
                (status, user, now(), comment[:500], claim_token))
            if cur.rowcount == 0:  # unknown claim, or already resolved: a decision is never silently overwritten
                self.conn.rollback()
                return False
            self.audit.append(user, "case_resolved", claim_token, {"status": status}, commit=False)
            self.conn.commit()
            self.audit.committed()
            return True

    # ------------------------------------------------------------------ notes queue (Granite worker)
    def claim_pending_notes(self, limit: int = 1) -> list[dict[str, Any]]:
        """Atomically move up to `limit` pending notes to 'working'.

        Briefs (an investigator is waiting on a held payment) go first, then the NEWEST corrections: on a
        small VM Granite writes about one note a minute, so the most recent claims get upgraded first and
        older ones keep their (accurate) template text.
        """
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, claim_token, kind, facts FROM notes WHERE status='pending' "
                "ORDER BY CASE kind WHEN 'brief' THEN 0 ELSE 1 END, id DESC LIMIT ?", (limit,)).fetchall()
            for r in rows:
                self.conn.execute("UPDATE notes SET status='working', updated_at=? WHERE id=?", (now(), r[0]))
            self.conn.commit()
        return [{"id": r[0], "claim_token": r[1], "kind": r[2], "facts": json.loads(r[3])} for r in rows]

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
                "SELECT n.id, n.claim_token, n.kind, n.status, n.source, n.text, n.updated_at, c.provider_token "
                "FROM notes n JOIN claims c USING (claim_token) WHERE n.updated_at > ? AND n.status IN ('done','failed') "
                "ORDER BY n.updated_at", (ts,)).fetchall()
        return [{"id": r[0], "claim_token": r[1], "kind": r[2], "status": r[3], "source": r[4], "text": r[5],
                 "updated_at": r[6], "provider_token": r[7]} for r in rows]

    # ------------------------------------------------------------------ reads
    def _notes_for(self, token: str) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT kind, status, source, text, model, gen_ms FROM notes WHERE claim_token=? ORDER BY id", (token,)
        ).fetchall()
        return {r[0]: {"status": r[1], "source": r[2], "text": r[3], "model": r[4], "gen_ms": r[5]} for r in rows}

    def _row(self, r) -> dict[str, Any]:
        keys = ["claim_token", "provider_token", "bene_token", "received_at", "service_date", "amount",
                "is_inpatient", "action", "anomaly", "provider_risk", "provider_claims", "rules", "reasons",
                "model_ms", "backend"]
        d = dict(zip(keys, r))
        d["rules"] = [x for x in d["rules"].split(",") if x]
        d["reasons"] = json.loads(d["reasons"])
        return d

    _COLS = ("claim_token, provider_token, bene_token, received_at, service_date, amount, is_inpatient, action, "
             "anomaly, provider_risk, provider_claims, rules, reasons, model_ms, backend")

    def list_cases(self, status: str = "open", limit: int = 100) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                f"SELECT {self._COLS}, k.status FROM claims c JOIN cases k USING (claim_token) "
                "WHERE k.status=? ORDER BY c.provider_risk DESC, c.received_at DESC LIMIT ?", (status, limit)).fetchall()
            out = []
            for r in rows:
                d = self._row(r[:-1])
                d["case_status"] = r[-1]
                d["notes"] = self._notes_for(d["claim_token"])
                out.append(d)
        return out

    def get_claim(self, token: str, decrypt: bool = False) -> dict[str, Any] | None:
        with self.lock:
            r = self.conn.execute(f"SELECT {self._COLS}, key_id, nonce, ciphertext FROM claims WHERE claim_token=?",
                                  (token,)).fetchone()
            if r is None:
                return None
            d = self._row(r[:-3])
            d["notes"] = self._notes_for(token)
            k = self.conn.execute("SELECT status, resolved_by, resolved_at, comment FROM cases WHERE claim_token=?",
                                  (token,)).fetchone()
        d["case"] = dict(zip(["status", "resolved_by", "resolved_at", "comment"], k)) if k else None
        if decrypt:
            d["payload"] = self.envelope.open_json(Sealed(r[-3], r[-2], r[-1]), aad=token)
        return d

    def iter_payloads(self, chunk: int = 5000):
        """Decrypted raw claims in arrival order (for rebuilding in-memory state after a restart)."""
        last = 0
        while True:
            with self.lock:
                rows = self.conn.execute("SELECT rowid, claim_token, key_id, nonce, ciphertext FROM claims "
                                         "WHERE rowid > ? ORDER BY rowid LIMIT ?", (last, chunk)).fetchall()
            if not rows:
                return
            for rowid, token, kid, nonce, ct in rows:
                yield self.envelope.open_json(Sealed(kid, nonce, ct), aad=token)
            last = rows[-1][0]

    def provider_claims(self, provider_token: str, limit: int = 100) -> list[dict[str, Any]]:
        """What a provider may see: their own non-paid claims with the correction note (never the brief)."""
        with self.lock:
            rows = self.conn.execute(
                f"SELECT {self._COLS} FROM claims WHERE provider_token=? AND action!='PAY' "
                "ORDER BY received_at DESC LIMIT ?", (provider_token, limit)).fetchall()
            out = []
            for r in rows:
                d = self._row(r)
                notes = self._notes_for(d["claim_token"])
                # minimal disclosure: what to fix, never internal scores or other providers' patterns
                out.append({
                    "claim_token": d["claim_token"], "service_date": d["service_date"], "amount": d["amount"],
                    "action": d["action"], "received_at": d["received_at"], "note": notes.get("correction"),
                    "reasons": {"rules": d["reasons"].get("rules", []), "claim": d["reasons"].get("claim", [])},
                })
        return out

    def aggregates(self) -> dict[str, Any]:
        """Auditor view: counts and sums only, no identifiers."""
        with self.lock:
            mix = dict(self.conn.execute("SELECT action, COUNT(*) FROM claims GROUP BY action").fetchall())
            money = dict(self.conn.execute("SELECT action, SUM(amount) FROM claims GROUP BY action").fetchall())
            cases = dict(self.conn.execute("SELECT status, COUNT(*) FROM cases GROUP BY status").fetchall())
            notes = dict(self.conn.execute("SELECT source, COUNT(*) FROM notes GROUP BY source").fetchall())
            backends = dict(self.conn.execute("SELECT backend, COUNT(*) FROM claims GROUP BY backend").fetchall())
        return {"decisions": mix, "amount_by_decision": money, "cases": cases, "notes_by_source": notes,
                "backends": backends, "total_claims": sum(mix.values())}
