"""ScamShield API end to end with TestClient (lifespan runs): history bootstrap, scoring parity, RBAC, privacy,
SSE, cases, audit, restart."""
import asyncio
import concurrent.futures
import datetime as dt
import json
import sqlite3
import threading

import jwt
import numpy as np
import pandas as pd
import pytest

from scamshield.fusion import ALLOW, HOLD, WARN

from conftest import (CUSTOMER_1, CUSTOMER_2, SPLITS, ApiEnv, bearer, login, policy_for, raw_account_numbers, scorers,
                      service_headers)

OUT_KEYS = {"payment_token", "action", "message", "reasons", "risk", "anomaly", "latency_ms", "model_ms", "backend",
            "batch_size"}
CUSTOMER_KEYS = {"payment_token", "paid_at", "amount_usd", "payment_format", "action", "received_at", "message",
                 "reasons"}


@pytest.fixture
def client(api_env):
    with api_env.client() as c:
        yield c


def post(client, env, p, headers=None):
    return client.post("/payments", json=p, headers=service_headers(env) if headers is None else headers)


def post_all(client, env, payments):
    out = []
    for p in payments:
        r = post(client, env, p)
        assert r.status_code == 200, r.text
        out.append(r.json())
    return out


def fresh_payment(i: int, ts: str = "2022-09-09T00:00:00", **over) -> dict:
    """A payment after the whole live stream, between brand-new accounts (no ordering constraints)."""
    p = {"payment_id": f"F{i}", "ts": ts, "from_bank": "0119", "from_acct": f"8F{i:07X}", "to_bank": "001",
         "to_acct": f"8E{i:07X}", "amount_paid": 100.0 + i, "currency_paid": "US Dollar", "amount_received": 100.0 + i,
         "currency_received": "US Dollar", "payment_format": "ACH"}
    p.update(over)
    return p


def offline_outcomes(aml):
    """What training would decide for the live days (vectorised, from training's own features)."""
    from scamshield.training import AE_IDX, decisions, rule_matrix

    Xt = aml.X[aml.split == "test"]
    a, r = scorers(aml.models_dir)
    err, _, _ = a.score(Xt[:, AE_IDX])
    risk = r.predict(Xt)
    return decisions(risk, err, rule_matrix(Xt, aml.ctx), policy_for(aml.models_dir)), risk, err


@pytest.fixture
def scored(client, api_env):
    """The whole live stream posted in order."""
    return post_all(client, api_env, api_env.stream())


# ------------------------------------------------------------------------------ start-up
def test_health_reports_history_bootstrap(client, api_env):
    h = client.get("/health").json()
    hist = api_env.aml.tx[api_env.aml.split != "test"]
    assert h == {"status": "ok", "anomaly_backend": "numpy", "risk_backend": "xgboost",
                 "history_payments": len(hist), "accounts_tracked": len(set(hist["sender"]) | set(hist["receiver"]))}


def test_history_end_is_respected(api_env):
    with api_env.client(history_end="2022-09-03") as c:
        n = c.get("/health").json()["history_payments"]
    assert n == int((api_env.aml.tx["ts"] < "2022-09-03").sum())


def test_missing_history_file_starts_empty(api_env, tmp_path):
    with api_env.client(history_file=tmp_path / "nope.csv") as c:
        h = c.get("/health").json()
    assert h["history_payments"] == 0 and h["accounts_tracked"] == 0


# ------------------------------------------------------------------------------ parity with training
def test_live_decisions_equal_training_decisions(client, api_env, scored):
    actions, risk, err = offline_outcomes(api_env.aml)
    assert [s["action"] for s in scored] == list(actions)
    np.testing.assert_allclose([s["risk"] for s in scored], np.round(risk, 4), atol=1e-12)
    np.testing.assert_allclose([s["anomaly"] for s in scored], np.round(err, 4), atol=1e-4)
    assert {ALLOW, WARN, HOLD} <= {s["action"] for s in scored}


def test_response_shape(client, api_env, scored):
    for s in scored:
        assert set(s) == OUT_KEYS
        assert s["backend"] == "numpy" and s["batch_size"] >= 1 and s["latency_ms"] >= 0
        assert (s["message"] is None) == (s["action"] == ALLOW)
    pseudo = api_env.pseudo()
    assert scored[0]["payment_token"] == pseudo.token("payment", "T0")
    held = next(s for s in scored if s["action"] == HOLD)
    assert held["message"].startswith("We've paused this payment")
    warned = next(s for s in scored if s["action"] == WARN)
    assert warned["message"].startswith("Before you send this payment")


def test_big_new_payee_customer_is_warned_with_the_rule(client, api_env, scored):
    i = next(j for j, p in enumerate(api_env.stream())
             if (p["from_bank"], p["from_acct"]) == CUSTOMER_1 and float(p["amount_paid"]) == 5000.0)
    s = scored[i]
    assert s["action"] in (WARN, HOLD)
    assert [r["rule"] for r in s["reasons"]["rules"]] == ["big_new_payee"]
    assert "first payment to this payee" in s["message"]


# ------------------------------------------------------------------------------ validation
def test_post_requires_the_service_token(client, api_env):
    p = api_env.stream()[0]
    assert client.post("/payments", json=p).status_code == 401
    for user in ("analyst", "auditor", "customer"):
        assert post(client, api_env, p, headers=login(client, user, api_env)).status_code == 403
    assert post(client, api_env, p).status_code == 200


def test_resubmitting_a_payment_id_is_409(client, api_env):
    p = api_env.stream()[0]
    assert post(client, api_env, p).status_code == 200
    assert post(client, api_env, {**p, "amount_paid": 1.0}).status_code == 409


@pytest.mark.parametrize("over", [{"payment_format": "Telepathy"}, {"amount_paid": -1}, {"ts": "yesterday"},
                                  {"from_acct": ""}, {"payment_id": ""}, {"amount_received": "lots"}])
def test_invalid_payments_are_422(client, api_env, over):
    assert post(client, api_env, {**api_env.stream()[0], **over}).status_code == 422


def test_missing_field_is_422(client, api_env):
    p = dict(api_env.stream()[0])
    del p["currency_received"]
    assert post(client, api_env, p).status_code == 422


def test_concurrent_posts_are_batched(api_env):
    with api_env.client(batch_window_ms=50.0) as c:
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
            rs = list(ex.map(lambda i: post(c, api_env, fresh_payment(i)), range(50)))
        assert all(r.status_code == 200 for r in rs), [r.text for r in rs if r.status_code != 200][:2]
        assert max(r.json()["batch_size"] for r in rs) > 1
        assert len({r.json()["payment_token"] for r in rs}) == 50
        m = c.get("/metrics", headers=login(c, "analyst", api_env)).json()
        assert m["payments_total"] == 50 and m["avg_batch_size"] > 1
        assert m["max_batch_size"] == max(r.json()["batch_size"] for r in rs)
        # payments_on_nnpa counts only payments the accelerator really scored (never on a CPU-only test machine)
        on_nnpa = sum(r.json()["backend"] == "torch_nnpa" for r in rs)
        assert m["payments_on_nnpa"] == on_nnpa and m["nnpa_share"] == round(on_nnpa / 50, 4)


def test_backdated_payment_does_not_fail_the_batch(client, api_env):
    assert post(client, api_env, fresh_payment(1, ts="2022-09-09T12:00:00")).status_code == 200
    late = fresh_payment(2, ts="2022-09-09T09:00:00", from_acct=fresh_payment(1)["from_acct"])
    assert post(client, api_env, late).status_code == 200


# ------------------------------------------------------------------------------ auth
def test_login_and_audit_entries(client, api_env):
    r = client.post("/auth/login", json={"username": "customer", "password": api_env.passwords["customer"]})
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "customer" and body["account_token"] == api_env.pseudo().token("account", ":".join(CUSTOMER_1))
    assert client.post("/auth/login", json={"username": "analyst", "password": "nope"}).status_code == 401
    assert client.post("/auth/login", json={"username": "ghost", "password": "x"}).status_code == 401
    assert client.post("/auth/login", json={"username": "analyst", "password": api_env.passwords["analyst"]}
                       ).json()["account_token"] is None
    events = [(e["actor"], e["action"]) for e in client.get("/audit/tail", headers=login(client, "auditor", api_env)).json()]
    assert ("customer", "login") in events and ("analyst", "login_failed") in events and ("ghost", "login_failed") in events
    assert ("system", "service_started") in events


def test_jwt_rejections(client, api_env):
    from scamshield.security import read_secret

    secret = read_secret("jwt_secret", api_env.secrets)
    now = dt.datetime.now(dt.timezone.utc)
    claims = {"sub": "analyst", "role": "analyst", "exp": now + dt.timedelta(hours=1)}
    bad = [jwt.encode({**claims, "exp": now - dt.timedelta(minutes=1)}, secret, algorithm="HS256"),
           jwt.encode(claims, "y" * 64, algorithm="HS256"),
           jwt.encode({**claims, "role": "admin"}, secret, algorithm="HS256"),
           jwt.encode({"sub": "analyst", "role": "analyst"}, secret, algorithm="HS256"),   # no exp
           "garbage"]
    for t in bad:
        assert client.get("/cases", headers={"Authorization": f"Bearer {t}"}).status_code == 401
    good = jwt.encode(claims, secret, algorithm="HS256")
    assert client.get("/cases", headers={"Authorization": f"Bearer {good}"}).status_code == 200


ENDPOINTS = [
    ("POST", "/payments", {"service"}),
    ("GET", "/metrics", {"analyst", "auditor"}),
    ("GET", "/stream", {"analyst", "auditor", "customer"}),
    ("GET", "/cases", {"analyst"}),
    ("GET", "/cases/{t}", {"analyst"}),
    ("POST", "/cases/{t}/resolve", {"analyst"}),
    ("GET", "/accounts/watchlist", {"analyst"}),
    ("GET", "/my/payments", {"customer"}),
    ("GET", "/audit/verify", {"auditor"}),
    ("GET", "/audit/tail", {"auditor"}),
    ("GET", "/audit/aggregates", {"auditor", "analyst"}),
]


def test_rbac_matrix(client, api_env, scored):
    t = next(s["payment_token"] for s in scored if s["action"] == HOLD)
    headers = {"service": service_headers(api_env), "analyst": login(client, "analyst", api_env),
               "auditor": login(client, "auditor", api_env), "customer": login(client, "customer", api_env)}
    extra = fresh_payment(1)
    for method, path, allowed in ENDPOINTS:
        url = path.replace("{t}", t)
        body = extra if path == "/payments" else {"status": "contact_customer"} if path.endswith("resolve") else None
        assert client.request(method, url, json=body).status_code == 401, (path, "anonymous")
        for role, h in headers.items():
            if role in allowed and path == "/stream":
                continue                    # an allowed /stream never ends; see the SSE tests
            code = client.request(method, url, json=body, headers=h).status_code
            assert code == (200 if role in allowed else 403), (path, role, code)


# ------------------------------------------------------------------------------ customer view
def test_customers_only_see_their_own_flagged_payments(client, api_env, scored):
    stream = api_env.stream()
    pseudo = api_env.pseudo()
    for user, acct in (("customer", CUSTOMER_1), ("customer2", CUSTOMER_2)):
        body = client.get("/my/payments", headers=login(client, user, api_env)).json()
        assert body["account_token"] == pseudo.token("account", ":".join(acct))
        expected = {s["payment_token"] for p, s in zip(stream, scored)
                    if (p["from_bank"], p["from_acct"]) == acct and s["action"] != ALLOW}
        assert expected and {p["payment_token"] for p in body["payments"]} == expected
        for p in body["payments"]:
            assert set(p) == CUSTOMER_KEYS and set(p["reasons"]) == {"rules", "customer"}
            assert p["message"]["source"] == "template"
        text = json.dumps(body)
        assert '"risk"' not in text and '"anomaly"' not in text and "drivers" not in text
        assert not any(raw in text for raw in raw_account_numbers(api_env.aml))


# ------------------------------------------------------------------------------ analyst
def test_cases_detail_and_resolution(client, api_env, scored):
    an = login(client, "analyst", api_env)
    holds = [s for s in scored if s["action"] == HOLD]
    cases = client.get("/cases", headers=an).json()
    assert {c["payment_token"] for c in cases} == {s["payment_token"] for s in holds}
    assert all("payload" not in c and c["notes"]["brief"]["status"] == "pending" for c in cases)
    t = holds[0]["payment_token"]
    d = client.get(f"/cases/{t}", headers=an).json()
    i = next(j for j, s in enumerate(scored) if s["payment_token"] == t)
    raw = api_env.stream()[i]
    assert d["payload"]["from_acct"] == raw["from_acct"] and d["payload"]["to_bank"] == raw["to_bank"]
    assert d["case"]["status"] == "open" and "not a finding of fraud" in d["notes"]["brief"]["text"]
    tail = client.get("/audit/tail", headers=login(client, "auditor", api_env)).json()
    viewed = [e for e in tail if e["action"] == "case_viewed"]
    assert viewed[0]["subject"] == t and viewed[0]["actor"] == "analyst" and viewed[0]["details"] == {"decrypted": True}
    assert client.get("/cases/XNOPE", headers=an).status_code == 404

    r = client.post(f"/cases/{t}/resolve", json={"status": "stopped", "comment": "customer confirmed scam"}, headers=an)
    assert r.status_code == 200 and r.json() == {"payment_token": t, "status": "stopped"}
    assert client.post(f"/cases/{t}/resolve", json={"status": "released"}, headers=an).status_code == 404
    assert client.get(f"/cases/{t}", headers=an).json()["case"]["status"] == "stopped"
    allowed = next(s["payment_token"] for s in scored if s["action"] == ALLOW)
    assert client.post(f"/cases/{allowed}/resolve", json={"status": "released"}, headers=an).status_code == 404
    assert client.post(f"/cases/{holds[1]['payment_token']}/resolve", json={"status": "deleted"},
                       headers=an).status_code == 422
    assert t in {c["payment_token"] for c in client.get("/cases?status=stopped", headers=an).json()}


def test_watchlist_uses_tokens(client, api_env, scored):
    from conftest import LIVE_MULE

    wl = client.get("/accounts/watchlist", headers=login(client, "analyst", api_env)).json()
    assert wl and all(w["payments"] >= 2 for w in wl)
    text = json.dumps(wl)
    assert not any(raw in text for raw in raw_account_numbers(api_env.aml))
    mule = api_env.pseudo().token("account", ":".join(LIVE_MULE))
    assert mule in {w["account"] for w in wl}


# ------------------------------------------------------------------------------ auditor + privacy
def test_aggregates(client, api_env, scored):
    a = client.get("/audit/aggregates", headers=login(client, "auditor", api_env)).json()
    assert a["total_payments"] == len(scored)
    assert a["decisions"] == {k: sum(s["action"] == k for s in scored) for k in {s["action"] for s in scored}}
    text = json.dumps(a)
    assert not any(s["payment_token"] in text for s in scored)


def test_audit_verify_then_tamper(client, api_env, scored):
    aud = login(client, "auditor", api_env)
    v = client.get("/audit/verify", headers=aud).json()
    assert v["ok"] is True and v["checked"] >= len(scored) and v["first_bad_id"] is None
    con = sqlite3.connect(api_env.db)
    con.execute("DROP TRIGGER audit_no_update")
    rid = con.execute("SELECT id FROM audit WHERE action='payment_scored' ORDER BY id LIMIT 1 OFFSET 3").fetchone()[0]
    con.execute("UPDATE audit SET details=? WHERE id=?", ('{"action":"ALLOW"}', rid))
    con.commit()
    con.close()
    v = client.get("/audit/verify", headers=aud).json()
    assert v["ok"] is False and v["first_bad_id"] == rid and "edited" in v["reason"]


def test_no_raw_account_numbers_at_rest_or_in_the_audit_log(client, api_env, scored):
    an = login(client, "analyst", api_env)
    held = next(s["payment_token"] for s in scored if s["action"] == HOLD)
    client.get(f"/cases/{held}", headers=an)
    client.post(f"/cases/{held}/resolve", json={"status": "released"}, headers=an)
    tail = client.get("/audit/tail?limit=500", headers=login(client, "auditor", api_env)).json()
    raws = raw_account_numbers(api_env.aml)
    tail_text = json.dumps(tail)
    assert not any(r in tail_text for r in raws)
    con = sqlite3.connect(api_env.db)
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    blob = b"".join(p.read_bytes() for p in api_env.db.parent.iterdir() if p.name.startswith(api_env.db.name))
    leaked = [r for r in raws if r.encode() in blob]
    assert leaked == []


# ------------------------------------------------------------------------------ SSE
async def _sse(app, query: str, n_blocks: int, opened: threading.Event | None = None, timeout: float = 5.0):
    """Drive the ASGI app directly (TestClient buffers streaming bodies); disconnect after n_blocks SSE blocks."""
    chunks, status = [], {}
    request_sent = False
    disconnect = asyncio.Event()

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(msg):
        if msg["type"] == "http.response.start":
            status["code"], status["headers"] = msg["status"], dict(msg.get("headers", []))
        elif msg["type"] == "http.response.body":
            chunks.append(msg.get("body", b""))
            if opened is not None and b": stream open" in b"".join(chunks):
                opened.set()
            if b"".join(chunks).count(b"\n\n") >= n_blocks or not msg.get("more_body", False):
                disconnect.set()

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
             "path": "/stream", "raw_path": b"/stream", "query_string": query.encode(), "root_path": "",
             "headers": [(b"host", b"testserver")], "client": ("127.0.0.1", 1), "server": ("testserver", 80),
             "state": {}}
    await asyncio.wait_for(app(scope, receive, send), timeout)
    return status, [b for b in b"".join(chunks).decode().split("\n\n") if b]


def parse(block: str) -> tuple[str, dict | None]:
    lines = block.splitlines()
    if lines[0].startswith(":"):
        return lines[0], None
    return lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))


def test_sse_customer_hello_without_metrics(client, api_env):
    tok = bearer(login(client, "customer", api_env))
    status, blocks = client.portal.call(_sse, client.app, f"token={tok}", 2)
    assert status["code"] == 200 and status["headers"][b"content-type"].startswith(b"text/event-stream")
    assert parse(blocks[0]) == ("hello", {"role": "customer",
                                          "account_token": api_env.pseudo().token("account", ":".join(CUSTOMER_1))})
    assert parse(blocks[1]) == (": stream open", None)            # no metrics event for customers
    assert client.app.state.svc.hub.subs == {}                     # unsubscribed after disconnect


@pytest.mark.parametrize("user", ["analyst", "auditor"])
def test_sse_staff_get_hello_then_metrics(client, api_env, user):
    _, blocks = client.portal.call(_sse, client.app, f"token={bearer(login(client, user, api_env))}", 3)
    assert parse(blocks[0])[1]["role"] == user and parse(blocks[0])[1]["account_token"] is None
    event, data = parse(blocks[1])
    assert event == "metrics" and "payments_total" in data
    assert parse(blocks[2]) == (": stream open", None)


def test_sse_rejects_bad_tokens(client, api_env):
    assert client.get("/stream?token=garbage").status_code == 401
    assert client.get("/stream").status_code == 401
    assert client.get(f"/stream?token={api_env.service_token}").status_code == 403


def test_sse_customer_receives_only_own_payment_events(client, api_env):
    stream = api_env.stream()
    own = next(p for p in stream if (p["from_bank"], p["from_acct"]) == CUSTOMER_1)
    others = [p for p in stream if (p["from_bank"], p["from_acct"]) != CUSTOMER_1][:5]
    opened = threading.Event()
    tok = bearer(login(client, "customer", api_env))
    fut = client.portal.start_task_soon(_sse, client.app, f"token={tok}", 3, opened)
    assert opened.wait(5)
    for p in others:
        assert post(client, api_env, p).status_code == 200
    mine = post(client, api_env, own).json()
    _, blocks = fut.result(timeout=5)
    event, data = parse(blocks[2])
    assert event == "payment" and data["payment_token"] == mine["payment_token"]
    assert data["sender_token"] == api_env.pseudo().token("account", ":".join(CUSTOMER_1))
    text = "\n\n".join(blocks)
    assert not any(r in text for r in raw_account_numbers(api_env.aml))


def test_sse_analyst_payment_events_carry_tokens_only(client, api_env):
    opened = threading.Event()
    fut = client.portal.start_task_soon(_sse, client.app, f"token={bearer(login(client, 'analyst', api_env))}", 6, opened)
    assert opened.wait(5)
    sent = post_all(client, api_env, api_env.stream()[:3])
    _, blocks = fut.result(timeout=5)
    payments = [parse(b)[1] for b in blocks if parse(b)[0] == "payment"]
    assert [p["payment_token"] for p in payments] == [s["payment_token"] for s in sent]
    assert set(payments[0]) == {"payment_token", "sender_token", "receiver_token", "amount_usd", "payment_format",
                                "action", "latency_ms", "rules"}
    assert not any(r in json.dumps(payments) for r in raw_account_numbers(api_env.aml))


def test_hub_role_filtering():
    from scamshield.api.auth import Principal
    from scamshield.api.main import Hub

    async def run():
        hub = Hub()
        qs = {n: hub.subscribe(p) for n, p in {
            "an": Principal("an", "analyst"), "au": Principal("au", "auditor"),
            "c1": Principal("c1", "customer", "XA"), "c2": Principal("c2", "customer", "XB")}.items()}
        hub.publish("payment", {"x": 1}, account_token="XA")
        hub.publish("metrics", {"m": 1}, roles=("analyst", "auditor"))
        hub.publish("metrics", {"m": 2})                                 # even if a caller forgets the roles
        hub.publish("note", {"n": 1}, account_token="XA", roles=("customer", "analyst"))
        hub.publish("note", {"brief": 1}, roles=("analyst",))
        return {k: [q.get_nowait()[0] for _ in range(q.qsize())] for k, q in qs.items()}

    assert asyncio.run(run()) == {"an": ["payment", "metrics", "metrics", "note", "note"],
                                  "au": ["metrics", "metrics"], "c1": ["payment", "note"], "c2": []}


# ------------------------------------------------------------------------------ restart
def test_restart_gives_the_same_decisions_as_an_uninterrupted_service(api_env, tmp_path):
    stream = api_env.stream()
    k = 60
    with api_env.client() as c:
        post_all(c, api_env, stream[:k])
        tracked = c.get("/health").json()["accounts_tracked"]
    with api_env.client() as c2:                                   # restart on the same DB
        h = c2.get("/health").json()
        assert h["accounts_tracked"] == tracked and c2.app.state.svc.restored_payments == k
        after_restart = post_all(c2, api_env, stream[k:])
        assert post(c2, api_env, stream[0]).status_code == 409    # stored payments are still known
        assert c2.get("/audit/verify", headers=login(c2, "auditor", api_env)).json()["ok"] is True

    other = ApiEnv(tmp_path / "uninterrupted", api_env.aml)      # different secrets, same data
    with other.client() as c3:
        straight = post_all(c3, other, stream)
    assert [s["action"] for s in after_restart] == [s["action"] for s in straight[k:]]
    assert [s["risk"] for s in after_restart] == [s["risk"] for s in straight[k:]]
    assert [s["anomaly"] for s in after_restart] == pytest.approx([s["anomaly"] for s in straight[k:]], abs=1e-4)


def test_restart_without_replay_would_differ(api_env):
    """Sanity check for the restart test: replay matters (state after k payments != state after none)."""
    stream = api_env.stream()
    with api_env.client() as c:
        post_all(c, api_env, stream[:60])
        before = c.get("/health").json()["accounts_tracked"]
    with api_env.client() as c2:
        assert c2.app.state.svc.restored_payments == 60
        assert c2.get("/health").json()["accounts_tracked"] == before
    hist = api_env.aml.tx[api_env.aml.split != "test"]
    assert before > len(set(hist["sender"]) | set(hist["receiver"]))


def test_metrics_count_payments_scored_on_nnpa():
    from scamshield.api.main import Metrics

    m = Metrics(nnpa_min_batch=1024)
    m.add(1.0, 0.1, "ALLOW", 10.0)
    for _ in range(3):
        m.add(1.0, 0.1, "WARN", 10.0, on_nnpa=True)
    snap = m.snapshot()
    assert snap["payments_on_nnpa"] == 3 and snap["nnpa_share"] == 0.75 and snap["nnpa_min_batch"] == 1024
    assert Metrics().snapshot()["nnpa_share"] == 0.0 and Metrics().snapshot()["nnpa_min_batch"] == 0


# ------------------------------------------------------------------------------ bulk payment files
def post_file(client, env, payments, headers=None):
    return client.post("/payments/batch", json={"payments": payments},
                       headers=service_headers(env) if headers is None else headers)


def test_payment_file_is_one_batch_in_order(client, api_env):
    payments = [fresh_payment(1000 + i) for i in range(40)]
    r = post_file(client, api_env, payments)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["batch_size"] == 40 and len(body["results"]) == 40
    assert all(x["batch_size"] == 40 for x in body["results"])
    # no NNPA on a CPU-only test machine, and a 40-payment file is far below NNPA_MIN_BATCH anyway
    assert body["backend"] != "torch_nnpa"
    tokens = [x["payment_token"] for x in body["results"]]
    assert len(set(tokens)) == 40
    m = client.get("/metrics", headers=login(client, "analyst", api_env)).json()
    assert m["max_batch_size"] >= 40 and m["payments_total"] >= 40


def test_payment_file_decisions_equal_single_posts(api_env, tmp_path):
    """Same payments, same starting state (two empty stores): a file and one-by-one posts decide alike."""
    payments = [fresh_payment(2000 + i, from_acct="8FSAME", amount_paid=50.0 * (i + 1) ** 2) for i in range(25)]
    with api_env.client(db_path=tmp_path / "a.db", checkpoint_path=tmp_path / "a.log") as a:
        one = post_all(a, api_env, payments)
    with api_env.client(db_path=tmp_path / "b.db", checkpoint_path=tmp_path / "b.log") as b:
        r = post_file(b, api_env, payments)
        assert r.status_code == 200, r.text
        res = r.json()["results"]
    assert [x["action"] for x in res] == [x["action"] for x in one]
    np.testing.assert_allclose([x["risk"] for x in res], [x["risk"] for x in one], atol=1e-4)
    np.testing.assert_allclose([x["anomaly"] for x in res], [x["anomaly"] for x in one], atol=1e-4)


def test_payment_file_rejects_repeats_and_resubmits_atomically(client, api_env):
    a, b = fresh_payment(3000), fresh_payment(3001)
    assert post_file(client, api_env, [a, a]).status_code == 409
    assert post(client, api_env, b).status_code == 200
    assert post_file(client, api_env, [a, b]).status_code == 409          # b already in: whole file refused
    assert post(client, api_env, a).status_code == 200                    # ... and a was not stored by it


def test_payment_file_needs_the_service_token_and_valid_size(client, api_env):
    p = [fresh_payment(4000)]
    assert post_file(client, api_env, p, headers={}).status_code == 401
    assert post_file(client, api_env, p, headers=login(client, "analyst", api_env)).status_code == 403
    assert post_file(client, api_env, []).status_code == 422
    big = [fresh_payment(5000 + i) for i in range(4097)]
    assert post_file(client, api_env, big).status_code == 422
