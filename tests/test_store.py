"""store.py: one transaction per batch, encryption at rest, notes queue, cases, customer-limited reads."""
import json
import sqlite3

import pytest
from cryptography.exceptions import InvalidTag

from scamshield.audit import AuditLog
from scamshield.security import Envelope
from scamshield.store import PaymentRecord, Store

KEY = bytes(range(32))
RAW = {"from_bank": "0119", "from_acct": "8C0FFEE01", "to_bank": "021174", "to_acct": "80BADBAD4"}
RAW_IDS = ["8C0FFEE01", "80BADBAD4", "0119:8C0FFEE01", "021174:80BADBAD4"]
CUSTOMER_KEYS = {"payment_token", "paid_at", "amount_usd", "payment_format", "action", "received_at", "message",
                 "reasons"}


def envelope():
    return Envelope({"k1": KEY}, "k1")


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "data" / "scamshield.db", tmp_path / "data" / "audit_checkpoints.log"


@pytest.fixture
def store(paths):
    s = Store(paths[0], envelope(), paths[1])
    yield s
    s.conn.close()


def reasons_for(action):
    if action == "ALLOW":
        return {}
    return {"rules": [{"rule": "big_new_payee", "title": "t", "advice": "a"}],
            "customer": [{"feature": "log_usd", "label": "payment amount", "value": "$5,000",
                          "direction": "higher than usual", "share": 0.7}],
            "drivers": [{"feature": "r_fan_in", "label": "fan in", "value": "90%", "contribution": 1.4}]}


def rec(token, action="ALLOW", sender="XSENDER1", notes=None, risk=0.2, amount=100.0) -> PaymentRecord:
    payload = {**RAW, "payment_id": f"T-{token}", "ts": "2022-09-07T10:00:00", "amount_paid": amount}
    return PaymentRecord(token, sender, "XRECEIVER", "2022-09-07T10:00:00", amount, "Wire", action, 0.42, risk,
                         ["big_new_payee"] if action != "ALLOW" else [], reasons_for(action), 1.5, "numpy", payload,
                         notes if notes is not None else [])


def count(store, table):
    return store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def warning(text="template warning"):
    return ("warning", text, {"rules": [], "amount_usd": 100, "held": False})


def brief(text="template brief"):
    return ("brief", text, {"payment_token": "P", "risk": 0.9})


# ------------------------------------------------------------------------------ transactions
def test_save_batch_writes_payments_notes_cases_audit(store):
    store.save_batch([rec("P1"), rec("P2", "WARN", notes=[warning()]), rec("P3", "HOLD", notes=[warning(), brief()])])
    assert count(store, "payments") == 3 and count(store, "notes") == 3 and count(store, "cases") == 1
    tail = store.audit.tail()
    assert [t["subject"] for t in tail] == ["P3", "P2", "P1"] and {t["action"] for t in tail} == {"payment_scored"}
    assert tail[0]["details"] == {"action": "HOLD", "risk": 0.2, "anomaly": 0.42, "rules": ["big_new_payee"],
                                  "backend": "numpy"}
    assert store.audit.verify().ok


@pytest.mark.parametrize("bad_index", [0, 2])
def test_failed_batch_persists_nothing_and_chain_still_verifies(store, paths, bad_index):
    store.save_batch([rec("P1")])
    batch = [rec("P2", "HOLD", notes=[warning(), brief()]), rec("P3", "WARN", notes=[warning()])]
    batch.insert(bad_index, rec("P1"))                    # duplicate payment_token -> whole batch rolls back
    with pytest.raises(sqlite3.IntegrityError):
        store.save_batch(batch)
    assert count(store, "payments") == 1 and count(store, "notes") == 0
    assert count(store, "cases") == 0 and count(store, "audit") == 1
    other = sqlite3.connect(paths[0])
    assert other.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 1
    other.close()
    assert store.audit.verify().ok
    store.save_batch([rec("P2")])
    assert count(store, "payments") == 2 and store.audit.verify().ok


def test_bad_note_kind_rolls_back(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.save_batch([rec("P1", "WARN", notes=[warning()]), rec("P2", "WARN", notes=[("poem", "x", {})])])
    assert count(store, "payments") == 0 and count(store, "audit") == 0 and store.audit.verify().ok


# ------------------------------------------------------------------------------ encryption at rest
def db_bytes(store, paths) -> bytes:
    store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return b"".join(p.read_bytes() for p in paths[0].parent.iterdir() if p.name.startswith(paths[0].name))


def test_payload_is_encrypted_at_rest(store, paths):
    store.save_batch([rec("P1", "HOLD", notes=[warning(), brief()]), rec("P2")])
    blob = db_bytes(store, paths)
    assert b"XSENDER1" in blob                            # sanity: tokens ARE stored in the clear
    for raw in RAW_IDS:
        assert raw.encode() not in blob, raw
    d = store.get_payment("P1", decrypt=True)
    assert d["payload"]["from_acct"] == "8C0FFEE01" and d["payload"]["to_bank"] == "021174"
    assert "payload" not in store.get_payment("P1")


def test_ciphertext_moved_to_another_row_fails(store):
    store.save_batch([rec("P1"), rec("P2")])
    kid, nonce, ct = store.conn.execute("SELECT key_id, nonce, ciphertext FROM payments WHERE payment_token='P1'").fetchone()
    store.conn.execute("UPDATE payments SET key_id=?, nonce=?, ciphertext=? WHERE payment_token='P2'", (kid, nonce, ct))
    store.conn.commit()
    assert store.get_payment("P1", decrypt=True)["payload"]["payment_id"] == "T-P1"
    with pytest.raises(InvalidTag):
        store.get_payment("P2", decrypt=True)


def test_key_rotation(paths):
    s1 = Store(paths[0], envelope(), paths[1])
    s1.save_batch([rec("P1")])
    s1.conn.close()
    s2 = Store(paths[0], Envelope({"k1": KEY, "k2": bytes(32)}, "k2"), paths[1])
    s2.save_batch([rec("P2")])
    assert s2.get_payment("P1", decrypt=True)["payload"]["payment_id"] == "T-P1"
    assert s2.conn.execute("SELECT key_id FROM payments WHERE payment_token='P2'").fetchone()[0] == "k2"
    s2.conn.close()


def test_get_payment_fields(store):
    assert store.get_payment("nope") is None and store.get_payment("nope", decrypt=True) is None
    store.save_batch([rec("P1", "HOLD", notes=[warning(), brief()]), rec("P2")])
    d = store.get_payment("P1")
    assert d["rules"] == ["big_new_payee"] and d["reasons"]["drivers"][0]["feature"] == "r_fan_in"
    assert d["case"] == {"status": "open", "resolved_by": None, "resolved_at": None, "comment": None}
    assert set(d["notes"]) == {"warning", "brief"} and d["notes"]["brief"]["source"] == "template"
    assert store.get_payment("P2")["rules"] == [] and store.get_payment("P2")["case"] is None


# ------------------------------------------------------------------------------ notes queue
@pytest.fixture
def queued(store):
    store.save_batch([rec("P1", "WARN", notes=[warning("w1")]), rec("P2", "HOLD", notes=[warning("w2"), brief("b1")])])
    store.save_batch([rec("P3", "WARN", notes=[warning("w3")]), rec("P4", "HOLD", notes=[warning("w4"), brief("b2")])])
    return store


def test_notes_queue_briefs_first_then_newest_warnings(queued):
    jobs = queued.claim_pending_notes(limit=10)
    assert [(j["kind"], j["payment_token"]) for j in jobs] == [
        ("brief", "P4"), ("brief", "P2"), ("warning", "P4"), ("warning", "P3"), ("warning", "P2"), ("warning", "P1")]
    assert jobs[0]["facts"] == {"payment_token": "P", "risk": 0.9}


def test_claim_pending_notes_moves_to_working_and_requeue(queued):
    assert [j["payment_token"] for j in queued.claim_pending_notes(limit=2)] == ["P4", "P2"]
    st = queued.conn.execute("SELECT kind, status, COUNT(*) FROM notes GROUP BY kind, status").fetchall()
    assert ("brief", "working", 2) in st and ("warning", "pending", 4) in st
    assert len(queued.claim_pending_notes(limit=10)) == 4
    assert queued.claim_pending_notes(limit=10) == []
    assert queued.requeue_stuck_notes() == 6 and queued.requeue_stuck_notes() == 0
    assert len(queued.claim_pending_notes(limit=10)) == 6


def test_complete_note_ok_failed_and_empty(queued):
    a, b, c = queued.claim_pending_notes(limit=3)
    queued.complete_note(a["id"], "Granite wrote this.", "granite.gguf", 812.34, ok=True)
    queued.complete_note(b["id"], None, "granite.gguf", 5.0, ok=False)
    queued.complete_note(c["id"], "", "granite.gguf", 1.0, ok=True)
    na = queued.get_payment(a["payment_token"])["notes"][a["kind"]]
    nb = queued.get_payment(b["payment_token"])["notes"][b["kind"]]
    nc = queued.get_payment(c["payment_token"])["notes"][c["kind"]]
    assert na == {"status": "done", "source": "granite", "text": "Granite wrote this.", "model": "granite.gguf",
                  "gen_ms": 812.34}
    assert nb["status"] == "failed" and nb["source"] == "template" and nb["text"] == "b1"
    assert nc["status"] == "failed" and nc["text"] == "w4"            # empty text keeps the template
    actions = [t["action"] for t in queued.audit.tail(3)]
    assert actions == ["note_failed", "note_failed", "note_generated"]
    assert queued.audit.verify().ok
    rows = queued.notes_updated_since("1970")
    assert len(rows) == 3 and {r["sender_token"] for r in rows} == {"XSENDER1"}
    assert queued.notes_updated_since(max(r["updated_at"] for r in rows)) == []


# ------------------------------------------------------------------------------ cases
def test_resolve_case_only_from_open(store):
    store.save_batch([rec("P1", "HOLD", notes=[brief()]), rec("P2")])
    assert [c["payment_token"] for c in store.list_cases()] == ["P1"]
    assert store.resolve_case("P1", "released", "analyst", "called back " * 100)
    case = store.get_payment("P1")["case"]
    assert case["status"] == "released" and case["resolved_by"] == "analyst" and len(case["comment"]) == 500
    n = count(store, "audit")
    assert store.resolve_case("P1", "stopped", "mallory") is False     # never silently overwritten
    assert store.get_payment("P1")["case"]["status"] == "released" and count(store, "audit") == n
    assert store.resolve_case("P2", "released", "analyst") is False    # ALLOW: no case
    assert store.resolve_case("nope", "released", "analyst") is False
    last = store.audit.tail(1)[0]
    assert (last["actor"], last["action"], last["subject"], last["details"]) == \
        ("analyst", "case_resolved", "P1", {"status": "released"})
    assert store.list_cases("open") == [] and [c["payment_token"] for c in store.list_cases("released")] == ["P1"]
    assert store.audit.verify().ok


def test_resolve_case_rejects_unknown_status(store):
    store.save_batch([rec("P1", "HOLD")])
    with pytest.raises(sqlite3.IntegrityError):
        store.resolve_case("P1", "deleted", "analyst")


def test_list_cases_sorted_by_risk(store):
    store.save_batch([rec("P1", "HOLD", risk=0.6), rec("P2", "HOLD", risk=0.95), rec("P3", "HOLD", risk=0.8)])
    assert [c["payment_token"] for c in store.list_cases()] == ["P2", "P3", "P1"]
    assert len(store.list_cases(limit=2)) == 2


# ------------------------------------------------------------------------------ customer + auditor views
def test_customer_payments_only_own_flagged_with_minimal_keys(store):
    store.save_batch([
        rec("A1", "WARN", sender="XA", notes=[warning("check A1")]),
        rec("A2", "ALLOW", sender="XA"),
        rec("A3", "HOLD", sender="XA", notes=[warning("paused A3"), brief("secret brief")]),
        rec("B1", "WARN", sender="XB", notes=[warning("check B1")]),
    ])
    mine = store.customer_payments("XA")
    assert {p["payment_token"] for p in mine} == {"A1", "A3"}
    for p in mine:
        assert set(p) == CUSTOMER_KEYS
        assert set(p["reasons"]) == {"rules", "customer"}               # never the model drivers
    by = {p["payment_token"]: p for p in mine}
    assert by["A1"]["message"]["text"] == "check A1" and by["A3"]["message"]["text"] == "paused A3"
    text = json.dumps(mine)
    for leak in ("secret brief", '"risk"', '"anomaly"', "r_fan_in", "XB", "B1", "0.42", "XRECEIVER", *RAW_IDS):
        assert leak not in text, leak
    assert [p["payment_token"] for p in store.customer_payments("XB")] == ["B1"]
    assert store.customer_payments("XC") == [] and store.customer_payments(None) == []
    assert len(store.customer_payments("XA", limit=1)) == 1


def test_aggregates_have_no_tokens_or_ids(store):
    store.save_batch([rec("PTOKEN1", "WARN", sender="XTOKENA", notes=[warning()]),
                      rec("PTOKEN2", "HOLD", sender="XTOKENB", notes=[warning(), brief()], amount=50.0),
                      rec("PTOKEN3", "ALLOW", sender="XTOKENA")])
    a = store.aggregates()
    assert a["total_payments"] == 3 and a["decisions"] == {"ALLOW": 1, "WARN": 1, "HOLD": 1}
    assert a["amount_usd_by_decision"]["HOLD"] == 50.0 and a["cases"] == {"open": 1}
    assert a["notes_by_source"] == {"template": 3} and a["backends"] == {"numpy": 3}
    text = json.dumps(a)
    for leak in ("PTOKEN", "XTOKEN", "XRECEIVER", *RAW_IDS):
        assert leak not in text


def test_iter_payloads_in_arrival_order_across_chunks(store):
    store.save_batch([rec("Z9"), rec("A1")])
    store.save_batch([rec("M5")])
    store.save_batch([rec("B2"), rec("Y8")])
    assert [p["payment_id"] for p in store.iter_payloads(chunk=2)] == ["T-Z9", "T-A1", "T-M5", "T-B2", "T-Y8"]
    assert [p["payment_id"] for p in store.iter_payloads()] == ["T-Z9", "T-A1", "T-M5", "T-B2", "T-Y8"]
    s2 = Store(":memory:", envelope())
    assert list(s2.iter_payloads(chunk=2)) == []


# ------------------------------------------------------------------------------ audit checkpoints
def test_checkpoint_written_after_save_batch(store, paths):
    store.audit.checkpoint_every = 2
    store.save_batch([rec("P1")])
    assert not paths[1].exists()
    store.save_batch([rec("P2")])
    lines = paths[1].read_text().splitlines()
    assert len(lines) == 1 and lines[0].split()[:2] == ["2", store.audit.tail(1)[0]["hash"]]
    store.save_batch([rec("P3"), rec("P4"), rec("P5")])
    assert len(paths[1].read_text().splitlines()) == 2


def _truncate_newest(db_path, n_rows):
    c = sqlite3.connect(db_path)
    c.execute("DROP TRIGGER audit_no_delete")
    c.execute("DELETE FROM audit WHERE id > (SELECT MAX(id) FROM audit) - ?", (n_rows,))
    c.commit()
    c.close()


def test_truncation_detected_by_same_and_new_instances(store, paths):
    store.audit.checkpoint_every = 2
    store.save_batch([rec(f"P{i}") for i in range(4)])
    assert store.audit.verify().ok
    _truncate_newest(paths[0], 2)
    r = store.audit.verify()
    assert not r.ok and "truncated" in r.reason
    fresh = Store(paths[0], envelope(), paths[1])
    r2 = fresh.audit.verify()
    assert not r2.ok and r2.first_bad_id == 4 and "checkpoint" in r2.reason
    fresh.conn.close()
    conn = sqlite3.connect(paths[0])
    assert AuditLog(conn, checkpoint_path=paths[1]).verify().first_bad_id == 4
    conn.close()


def test_new_instance_without_tampering_verifies(store, paths):
    store.audit.checkpoint_every = 1
    store.save_batch([rec("P1"), rec("P2")])
    store.conn.close()
    fresh = Store(paths[0], envelope(), paths[1])
    fresh.save_batch([rec("P3")])
    assert fresh.audit.verify().ok and fresh.audit.verify().checked == 3
    fresh.conn.close()
