"""store.py: one transaction per batch, encryption at rest, notes queue, cases, role-limited reads."""
import json
import sqlite3

import pytest
from cryptography.exceptions import InvalidTag

from claimsense.audit import AuditLog
from claimsense.security import Envelope
from claimsense.store import ClaimRecord, Store

KEY = bytes(range(32))
RAW_IDS = ["PRV51001", "BENE11001", "PHY390922", "CLM46614"]


def envelope():
    return Envelope({"k1": KEY}, "k1")


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "data" / "claimsense.db", tmp_path / "data" / "audit_checkpoints.log"


@pytest.fixture
def store(paths):
    s = Store(paths[0], envelope(), paths[1])
    yield s
    s.conn.close()


def rec(token, action="PAY", provider="PTOK1", notes=None, risk=0.2, amount=100.0, raw_suffix="",
        reasons=None) -> ClaimRecord:
    payload = {"ClaimID": f"CLM46614{raw_suffix}", "BeneID": "BENE11001", "Provider": "PRV51001",
               "AttendingPhysician": "PHY390922", "InscClaimAmtReimbursed": amount, "token": token}
    if reasons is None:
        reasons = {} if action == "PAY" else {
            "rules": [{"rule": "missing_attending", "title": "t", "fix": "f"}],
            "claim": [{"feature": "log_amount", "label": "reimbursed amount", "value": "$100",
                       "direction": "higher than usual", "share_of_anomaly": 0.5}],
            "provider": [{"feature": "claims_per_bene", "label": "claims per patient", "value": "9.0",
                          "peer_percentile": 99.0, "contribution": 0.5}],
        }
    return ClaimRecord(token, provider, "BTOK1", "2009-03-01", amount, 0, action, 0.42, risk, 12,
                       ["missing_attending"] if action != "PAY" else [], reasons, 1.5, "numpy", payload,
                       notes if notes is not None else [])


def count(store, table):
    return store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def correction(text="template correction"):
    return ("correction", text, {"rules": [], "amount": 100})


def brief(text="template brief"):
    return ("brief", text, {"provider_token": "PTOK1", "provider_risk": 0.9})


# ------------------------------------------------------------------------------ transactions
def test_save_batch_writes_claims_notes_cases_audit(store):
    store.save_batch([
        rec("C1"),
        rec("C2", "RETURN_WITH_FIX", notes=[correction()]),
        rec("C3", "HOLD_FOR_REVIEW", notes=[brief()]),
    ])
    assert count(store, "claims") == 3
    assert count(store, "notes") == 2
    assert count(store, "cases") == 1
    tail = store.audit.tail()
    assert [t["subject"] for t in tail] == ["C3", "C2", "C1"]
    assert all(t["action"] == "claim_scored" for t in tail)
    assert tail[0]["details"]["action"] == "HOLD_FOR_REVIEW" and tail[0]["details"]["provider"] == "PTOK1"
    assert store.audit.verify().ok


def test_failed_batch_on_first_record_leaves_nothing(store):
    store.save_batch([rec("C1")])
    with pytest.raises(sqlite3.IntegrityError):
        store.save_batch([rec("C1"), rec("C2", "HOLD_FOR_REVIEW", notes=[brief()])])
    assert count(store, "claims") == 1 and count(store, "notes") == 0 and count(store, "cases") == 0
    assert count(store, "audit") == 1
    assert store.audit.verify().ok


def _failing_mid_batch(store):
    store.save_batch([rec("C1")])
    with pytest.raises(sqlite3.IntegrityError):
        store.save_batch([rec("C2", "HOLD_FOR_REVIEW", notes=[brief()]),
                          rec("C3", "RETURN_WITH_FIX", notes=[correction()]),
                          rec("C1")])   # duplicate claim_token -> the whole batch must roll back


def test_failed_batch_mid_way_persists_nothing(store, paths):
    _failing_mid_batch(store)
    assert count(store, "claims") == 1
    assert count(store, "notes") == 0
    assert count(store, "cases") == 0
    assert count(store, "audit") == 1
    assert store.get_claim("C2") is None and store.get_claim("C3") is None
    # a second connection sees the same thing (nothing half-committed)
    other = sqlite3.connect(paths[0])
    assert other.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1
    other.close()
    # the store still works afterwards
    store.save_batch([rec("C2")])
    assert count(store, "claims") == 2 and store.audit.verify().ok


def test_audit_chain_verifies_right_after_a_failed_batch(store):
    _failing_mid_batch(store)
    r = store.audit.verify()
    assert r.ok, r


def test_check_constraint_failure_also_rolls_back(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.save_batch([rec("C1", "RETURN_WITH_FIX", notes=[correction()]),
                          rec("C2", "RETURN_WITH_FIX", notes=[("poem", "x", {})])])
    assert count(store, "claims") == 0 and count(store, "notes") == 0 and count(store, "audit") == 0


# ------------------------------------------------------------------------------ encryption at rest
def db_bytes(store, paths) -> bytes:
    store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    out = b""
    for p in paths[0].parent.iterdir():
        if p.name.startswith(paths[0].name):
            out += p.read_bytes()
    return out


def test_payload_is_encrypted_at_rest(store, paths):
    store.save_batch([rec("C1", "RETURN_WITH_FIX", notes=[correction()]), rec("C2", raw_suffix="9")])
    blob = db_bytes(store, paths)
    assert b"C1" in blob              # sanity: tokens ARE stored in the clear
    for raw in RAW_IDS:
        assert raw.encode() not in blob, raw
    d = store.get_claim("C1", decrypt=True)
    assert d["payload"]["Provider"] == "PRV51001" and d["payload"]["BeneID"] == "BENE11001"
    assert "payload" not in store.get_claim("C1")


def test_ciphertext_moved_to_another_row_fails(store):
    store.save_batch([rec("C1"), rec("C2")])
    kid, nonce, ct = store.conn.execute("SELECT key_id, nonce, ciphertext FROM claims WHERE claim_token='C1'").fetchone()
    store.conn.execute("UPDATE claims SET key_id=?, nonce=?, ciphertext=? WHERE claim_token='C2'", (kid, nonce, ct))
    store.conn.commit()
    assert store.get_claim("C1", decrypt=True)["payload"]["token"] == "C1"
    with pytest.raises(InvalidTag):
        store.get_claim("C2", decrypt=True)


def test_key_rotation_old_rows_still_open(paths):
    s1 = Store(paths[0], envelope(), paths[1])
    s1.save_batch([rec("C1")])
    s1.conn.close()
    s2 = Store(paths[0], Envelope({"k1": KEY, "k2": bytes(32)}, "k2"), paths[1])
    s2.save_batch([rec("C2")])
    assert s2.get_claim("C1", decrypt=True)["payload"]["token"] == "C1"
    assert s2.conn.execute("SELECT key_id FROM claims WHERE claim_token='C2'").fetchone()[0] == "k2"
    s2.conn.close()


def test_get_claim_unknown_is_none(store):
    assert store.get_claim("nope") is None
    assert store.get_claim("nope", decrypt=True) is None


def test_get_claim_fields(store):
    store.save_batch([rec("C1", "HOLD_FOR_REVIEW", notes=[brief()])])
    d = store.get_claim("C1")
    assert d["rules"] == ["missing_attending"]
    assert d["reasons"]["claim"][0]["feature"] == "log_amount"
    assert d["case"] == {"status": "open", "resolved_by": None, "resolved_at": None, "comment": None}
    assert d["notes"]["brief"]["status"] == "pending" and d["notes"]["brief"]["source"] == "template"
    store.save_batch([rec("C2")])
    assert store.get_claim("C2")["rules"] == [] and store.get_claim("C2")["case"] is None


# ------------------------------------------------------------------------------ notes queue
@pytest.fixture
def queued(store):
    store.save_batch([rec("C1", "RETURN_WITH_FIX", notes=[correction("c1")]),
                      rec("C2", "HOLD_FOR_REVIEW", notes=[brief("b1")])])
    store.save_batch([rec("C3", "RETURN_WITH_FIX", notes=[correction("c2")]),
                      rec("C4", "RETURN_WITH_FIX", notes=[correction("c3")]),
                      rec("C5", "HOLD_FOR_REVIEW", notes=[brief("b2")])])
    return store


def test_notes_queue_briefs_first_then_newest_corrections(queued):
    jobs = queued.claim_pending_notes(limit=10)
    assert [(j["kind"], j["claim_token"]) for j in jobs] == [
        ("brief", "C5"), ("brief", "C2"), ("correction", "C4"), ("correction", "C3"), ("correction", "C1")]
    assert jobs[0]["facts"] == {"provider_token": "PTOK1", "provider_risk": 0.9}


def test_claim_pending_notes_moves_to_working(queued):
    first = queued.claim_pending_notes(limit=2)
    assert [j["claim_token"] for j in first] == ["C5", "C2"]
    statuses = dict(queued.conn.execute("SELECT claim_token, status FROM notes").fetchall())
    assert statuses == {"C1": "pending", "C2": "working", "C3": "pending", "C4": "pending", "C5": "working"}
    assert [j["claim_token"] for j in queued.claim_pending_notes(limit=1)] == ["C4"]
    queued.claim_pending_notes(limit=10)
    assert queued.claim_pending_notes(limit=10) == []


def test_requeue_stuck_notes(queued):
    queued.claim_pending_notes(limit=3)
    assert queued.requeue_stuck_notes() == 3
    assert queued.requeue_stuck_notes() == 0
    assert len(queued.claim_pending_notes(limit=10)) == 5


def test_complete_note_ok_replaces_text(queued):
    job = queued.claim_pending_notes(limit=1)[0]
    queued.complete_note(job["id"], "Granite wrote this.", "granite-3.3-2b", 812.34, ok=True)
    n = queued.get_claim(job["claim_token"])["notes"]["brief"]
    assert n == {"status": "done", "source": "granite", "text": "Granite wrote this.", "model": "granite-3.3-2b",
                 "gen_ms": 812.34}
    last = queued.audit.tail(1)[0]
    assert last["action"] == "note_generated" and last["subject"] == str(job["id"])
    assert last["details"] == {"model": "granite-3.3-2b", "gen_ms": 812.3}
    assert queued.audit.verify().ok


def test_complete_note_failed_keeps_template(queued):
    job = queued.claim_pending_notes(limit=1)[0]
    queued.complete_note(job["id"], None, "granite-3.3-2b", 5.0, ok=False)
    n = queued.get_claim(job["claim_token"])["notes"]["brief"]
    assert n["status"] == "failed" and n["source"] == "template" and n["text"] == "b2"
    assert queued.audit.tail(1)[0]["action"] == "note_failed"


def test_complete_note_ok_with_empty_text_keeps_template(queued):
    job = queued.claim_pending_notes(limit=1)[0]
    queued.complete_note(job["id"], "", "m", 1.0, ok=True)
    n = queued.get_claim(job["claim_token"])["notes"]["brief"]
    assert n["status"] == "failed" and n["text"] == "b2"


def test_notes_updated_since(queued):
    assert queued.notes_updated_since("1970") == []
    j = queued.claim_pending_notes(limit=1)[0]
    queued.complete_note(j["id"], "done text", "m", 1.0, ok=True)
    rows = queued.notes_updated_since("1970")
    assert len(rows) == 1 and rows[0]["text"] == "done text" and rows[0]["provider_token"] == "PTOK1"
    assert queued.notes_updated_since(rows[0]["updated_at"]) == []


# ------------------------------------------------------------------------------ cases
def test_resolve_case_and_audit(store):
    store.save_batch([rec("C1", "HOLD_FOR_REVIEW", notes=[brief()]), rec("C2")])
    assert store.list_cases()[0]["claim_token"] == "C1"
    assert store.resolve_case("C1", "released", "inv", "looked fine " * 100)
    case = store.get_claim("C1")["case"]
    assert case["status"] == "released" and case["resolved_by"] == "inv" and case["resolved_at"]
    assert len(case["comment"]) == 500
    last = store.audit.tail(1)[0]
    assert (last["actor"], last["action"], last["subject"], last["details"]) == \
        ("inv", "case_resolved", "C1", {"status": "released"})
    assert store.list_cases("open") == []
    assert [c["claim_token"] for c in store.list_cases("released")] == ["C1"]
    assert store.audit.verify().ok


def test_resolve_unknown_case_is_false_and_not_audited(store):
    store.save_batch([rec("C2")])   # PAY: no case row
    n = count(store, "audit")
    assert store.resolve_case("C2", "released", "inv") is False
    assert store.resolve_case("nope", "released", "inv") is False
    assert count(store, "audit") == n


def test_resolve_case_rejects_unknown_status(store):
    store.save_batch([rec("C1", "HOLD_FOR_REVIEW")])
    with pytest.raises(sqlite3.IntegrityError):
        store.resolve_case("C1", "deleted", "inv")


def test_list_cases_sorted_by_provider_risk(store):
    store.save_batch([rec("C1", "HOLD_FOR_REVIEW", risk=0.6), rec("C2", "HOLD_FOR_REVIEW", risk=0.95),
                      rec("C3", "HOLD_FOR_REVIEW", risk=0.8)])
    cases = store.list_cases()
    assert [c["claim_token"] for c in cases] == ["C2", "C3", "C1"]
    assert all(c["case_status"] == "open" and "notes" in c for c in cases)
    assert len(store.list_cases(limit=2)) == 2


# ------------------------------------------------------------------------------ role-limited reads
PROVIDER_KEYS = {"claim_token", "service_date", "amount", "action", "received_at", "note", "reasons"}


def test_provider_claims_only_own_non_pay(store):
    store.save_batch([
        rec("A1", "RETURN_WITH_FIX", provider="PA", notes=[correction("fix A1")]),
        rec("A2", "PAY", provider="PA"),
        rec("A3", "HOLD_FOR_REVIEW", provider="PA", notes=[brief("secret brief")]),
        rec("B1", "RETURN_WITH_FIX", provider="PB", notes=[correction("fix B1")]),
    ])
    mine = store.provider_claims("PA")
    assert {c["claim_token"] for c in mine} == {"A1", "A3"}
    for c in mine:
        assert set(c) == PROVIDER_KEYS
        assert set(c["reasons"]) == {"rules", "claim"}          # never the provider-pattern reasons
    by = {c["claim_token"]: c for c in mine}
    assert by["A1"]["note"]["text"] == "fix A1"
    assert by["A3"]["note"] is None                             # briefs are never shown to providers
    text = json.dumps(mine)
    for leak in ("secret brief", '"anomaly"', "provider_risk", "provider_claims", "claims_per_bene", "PB", "B1",
                 "0.42"):
        assert leak not in text, leak
    assert [c["claim_token"] for c in store.provider_claims("PB")] == ["B1"]
    assert store.provider_claims("PC") == []
    assert len(store.provider_claims("PA", limit=1)) == 1


def test_aggregates_have_no_tokens_or_ids(store):
    store.save_batch([rec("CTOKEN1", "RETURN_WITH_FIX", provider="PTOKENX", notes=[correction()]),
                      rec("CTOKEN2", "HOLD_FOR_REVIEW", provider="PTOKENY", notes=[brief()], amount=50.0),
                      rec("CTOKEN3", "PAY", provider="PTOKENX")])
    a = store.aggregates()
    assert a["total_claims"] == 3
    assert a["decisions"] == {"PAY": 1, "RETURN_WITH_FIX": 1, "HOLD_FOR_REVIEW": 1}
    assert a["amount_by_decision"]["HOLD_FOR_REVIEW"] == 50.0
    assert a["cases"] == {"open": 1}
    assert a["notes_by_source"] == {"template": 2}
    assert a["backends"] == {"numpy": 3}
    text = json.dumps(a)
    for leak in ("CTOKEN", "PTOKEN", "BTOK", *RAW_IDS):
        assert leak not in text


def test_iter_payloads_in_arrival_order_across_chunks(store):
    store.save_batch([rec("Z9"), rec("A1")])
    store.save_batch([rec("M5")])
    store.save_batch([rec("B2"), rec("Y8")])
    assert [p["token"] for p in store.iter_payloads(chunk=2)] == ["Z9", "A1", "M5", "B2", "Y8"]
    assert [p["token"] for p in store.iter_payloads(chunk=1000)] == ["Z9", "A1", "M5", "B2", "Y8"]


def test_iter_payloads_empty(store):
    assert list(store.iter_payloads(chunk=2)) == []


# ------------------------------------------------------------------------------ audit checkpoints
def test_checkpoint_written_after_save_batch(store, paths):
    store.audit.checkpoint_every = 2
    store.save_batch([rec("C1")])
    assert not paths[1].exists()
    store.save_batch([rec("C2")])
    lines = paths[1].read_text().splitlines()
    assert len(lines) == 1
    rid, h = lines[0].split()[:2]
    assert int(rid) == 2 and h == store.audit.tail(1)[0]["hash"]
    store.save_batch([rec("C3"), rec("C4"), rec("C5")])
    assert len(paths[1].read_text().splitlines()) == 2


def _truncate_newest(db_path, n_rows):
    c = sqlite3.connect(db_path)
    c.execute("DROP TRIGGER audit_no_delete")
    c.execute("DELETE FROM audit WHERE id > (SELECT MAX(id) FROM audit) - ?", (n_rows,))
    c.commit()
    c.close()


def test_truncation_after_checkpoint_detected_by_same_store(store, paths):
    store.audit.checkpoint_every = 2
    store.save_batch([rec(f"C{i}") for i in range(4)])
    assert store.audit.verify().ok
    _truncate_newest(paths[0], 2)
    r = store.audit.verify()
    assert not r.ok and "truncated" in r.reason


def test_truncation_after_checkpoint_detected_by_new_instances(store, paths):
    store.audit.checkpoint_every = 2
    store.save_batch([rec(f"C{i}") for i in range(4)])
    store.conn.close()
    _truncate_newest(paths[0], 2)
    fresh = Store(paths[0], envelope(), paths[1])
    r = fresh.audit.verify()
    assert not r.ok and r.first_bad_id == 4 and "checkpoint" in r.reason
    fresh.conn.close()
    conn = sqlite3.connect(paths[0])
    r2 = AuditLog(conn, checkpoint_path=paths[1]).verify()
    assert not r2.ok and r2.first_bad_id == 4
    conn.close()


def test_new_instance_without_tampering_verifies(store, paths):
    store.audit.checkpoint_every = 1
    store.save_batch([rec("C1"), rec("C2")])
    store.conn.close()
    fresh = Store(paths[0], envelope(), paths[1])
    assert fresh.audit.verify().ok
    fresh.save_batch([rec("C3")])
    assert fresh.audit.verify().ok and fresh.audit.verify().checked == 3
    fresh.conn.close()
