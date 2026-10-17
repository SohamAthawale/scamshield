"""audit.py: hash-chained, append-only audit log."""
import json
import sqlite3
import threading

import pytest

from scamshield.audit import GENESIS, AuditLog, canonical, row_hash


@pytest.fixture
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "audit.db", check_same_thread=False)
    yield c
    c.close()


@pytest.fixture
def log(conn):
    return AuditLog(conn)


def fill(log, n=5):
    return [log.append("svc", "score", f"C{i}", {"action": "PAY", "i": i}, ts=f"2026-10-17T00:00:0{i}")
            for i in range(n)]


def drop_triggers(conn):
    conn.execute("DROP TRIGGER audit_no_update")
    conn.execute("DROP TRIGGER audit_no_delete")
    conn.commit()


def test_empty_log_verifies_and_starts_at_genesis(log, conn):
    r = log.verify()
    assert r.ok and r.checked == 0 and r.first_bad_id is None
    assert GENESIS == "0" * 64
    log.append("svc", "start")
    assert conn.execute("SELECT prev_hash FROM audit").fetchone()[0] == GENESIS


def test_append_and_verify_happy_path(log, conn):
    hashes = fill(log, 5)
    r = log.verify()
    assert r.ok and r.checked == 5
    rows = conn.execute("SELECT prev_hash, hash FROM audit ORDER BY id").fetchall()
    assert [h for _, h in rows] == hashes
    assert [p for p, _ in rows] == [GENESIS] + hashes[:-1]


def test_hash_matches_documented_formula(log, conn):
    h = log.append("alice", "login", None, {"b": 2, "a": 1}, ts="2026-10-17T10:00:00")
    row = {"ts": "2026-10-17T10:00:00", "actor": "alice", "action": "login", "subject": None,
           "details": canonical({"a": 1, "b": 2})}
    assert h == row_hash(GENESIS, row)
    assert conn.execute("SELECT details FROM audit").fetchone()[0] == '{"a":1,"b":2}'


def test_default_timestamp_is_utc_iso(log):
    log.append("svc", "x")
    ts = log.tail(1)[0]["ts"]
    assert ts.endswith("+00:00") and "T" in ts


def test_update_is_refused_by_trigger(log, conn):
    fill(log, 2)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE audit SET details='{}' WHERE id=1")


def test_delete_is_refused_by_trigger(log, conn):
    fill(log, 2)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM audit WHERE id=1")


def test_reopening_keeps_triggers_and_chain(tmp_path):
    path = tmp_path / "a.db"
    c1 = sqlite3.connect(path)
    fill(AuditLog(c1), 3)
    c1.close()
    c2 = sqlite3.connect(path)
    log2 = AuditLog(c2)
    log2.append("svc", "more")
    assert log2.verify().ok and log2.verify().checked == 4
    with pytest.raises(sqlite3.DatabaseError):
        c2.execute("DELETE FROM audit")
    c2.close()


@pytest.mark.parametrize("bad_id", [1, 3, 5])
def test_edited_details_are_detected_at_that_row(log, conn, bad_id):
    fill(log, 5)
    drop_triggers(conn)
    conn.execute("UPDATE audit SET details=? WHERE id=?", (json.dumps({"action": "PAY", "i": 99}), bad_id))
    conn.commit()
    r = log.verify()
    assert not r.ok
    assert r.first_bad_id == bad_id
    assert r.checked == bad_id
    assert "edited" in r.reason


@pytest.mark.parametrize("col,val", [("actor", "mallory"), ("action", "approve"), ("subject", "C999"),
                                     ("ts", "2020-01-01T00:00:00")])
def test_edited_other_columns_are_detected(log, conn, col, val):
    fill(log, 4)
    drop_triggers(conn)
    conn.execute(f"UPDATE audit SET {col}=? WHERE id=2", (val,))
    conn.commit()
    r = log.verify()
    assert not r.ok and r.first_bad_id == 2


def test_rehashing_an_edited_row_breaks_the_next_link(log, conn):
    """An attacker who also recomputes the edited row's hash is caught at the following row."""
    fill(log, 4)
    drop_triggers(conn)
    rid, ts, actor, action, subject, prev = conn.execute(
        "SELECT id, ts, actor, action, subject, prev_hash FROM audit WHERE id=2").fetchone()
    details = canonical({"action": "HOLD"})
    new_h = row_hash(prev, {"ts": ts, "actor": actor, "action": action, "subject": subject, "details": details})
    conn.execute("UPDATE audit SET details=?, hash=? WHERE id=2", (details, new_h))
    conn.commit()
    r = log.verify()
    assert not r.ok and r.first_bad_id == 3 and "link" in r.reason


def test_deleting_a_middle_row_is_detected(log, conn):
    fill(log, 5)
    drop_triggers(conn)
    conn.execute("DELETE FROM audit WHERE id=3")
    conn.commit()
    r = log.verify()
    assert not r.ok
    assert r.first_bad_id == 4
    assert "link" in r.reason


def test_deleting_the_first_row_is_detected(log, conn):
    fill(log, 3)
    drop_triggers(conn)
    conn.execute("DELETE FROM audit WHERE id=1")
    conn.commit()
    r = log.verify()
    assert not r.ok and r.first_bad_id == 2


def test_deleting_the_last_row_is_detected(log, conn):
    fill(log, 5)
    drop_triggers(conn)
    conn.execute("DELETE FROM audit WHERE id=5")
    conn.commit()
    assert not log.verify().ok


def test_tail_newest_first_with_parsed_details(log):
    hashes = fill(log, 5)
    t = log.tail(3)
    assert [r["id"] for r in t] == [5, 4, 3]
    assert t[0]["details"] == {"action": "PAY", "i": 4}
    assert t[0]["hash"] == hashes[-1]
    assert t[0]["subject"] == "C4" and t[0]["actor"] == "svc" and t[0]["action"] == "score"
    assert len(log.tail()) == 5
    assert AuditLog(sqlite3.connect(":memory:")).tail() == []


def test_append_without_commit_then_commit(log, conn):
    log.append("svc", "a", commit=False)
    log.append("svc", "b", commit=False)
    conn.commit()
    assert log.verify().ok and log.verify().checked == 2


def test_concurrent_appends_keep_a_valid_chain(log):
    def worker(k):
        for i in range(25):
            log.append(f"w{k}", "score", f"C{k}-{i}", {"i": i})

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    r = log.verify()
    assert r.ok and r.checked == 100
