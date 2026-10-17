"""FastAPI app end to end with TestClient (lifespan runs), tiny models, tiny registry and real secrets."""
import asyncio
import concurrent.futures
import datetime as dt
import json
import sqlite3

import jwt
import pytest

from claimsense.fusion import HOLD, PAY, RETURN

from conftest import DECEASED, claim_json, login, service_headers

CLAIM_OUT_KEYS = {"claim_token", "provider_token", "action", "reasons", "anomaly", "provider_risk", "latency_ms",
                  "model_ms", "backend", "batch_size"}


@pytest.fixture
def client(api_env):
    with api_env.client() as c:
        yield c


def post(client, env, claim, headers=None):
    return client.post("/claims", json=claim, headers=service_headers(env) if headers is None else headers)


def seed_claims(client, env):
    """PRV51001: 1 clean PAY + 3 claims without attending (RETURN, RETURN, HOLD from the 3rd provider claim on
    under DEFAULT_POLICY, min_provider_claims=3). PRV51002: 1 RETURN."""
    out = {"pay": post(client, env, claim_json("C-PAY", amount=111.0)).json()}
    out["r1"] = post(client, env, claim_json("C-R1", amount=201.0, attending=None)).json()
    out["h1"] = post(client, env, claim_json("C-H1", amount=202.0, attending=None)).json()
    out["h2"] = post(client, env, claim_json("C-H2", amount=203.0, attending=None)).json()
    out["other"] = post(client, env, claim_json("C-O1", provider="PRV51002", amount=301.0, attending=None)).json()
    assert [out[k]["action"] for k in ("pay", "r1", "h1", "h2", "other")] == [PAY, RETURN, HOLD, HOLD, RETURN]
    return out


# ------------------------------------------------------------------------------ health + auth
def test_health(client, api_env):
    h = client.get("/health").json()
    assert h == {"status": "ok", "anomaly_backend": "numpy", "provider_backend": "xgboost", "registry": 12,
                 "providers_tracked": 0}


def test_login_ok_and_bad_password_are_audited(client, api_env):
    r = client.post("/auth/login", json={"username": "prv1", "password": api_env.passwords["prv1"]})
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "provider" and body["token_type"] == "bearer"
    assert body["provider_token"] == api_env.pseudo().token("provider", "PRV51001")
    assert client.post("/auth/login", json={"username": "inv", "password": "wrong"}).status_code == 401
    assert client.post("/auth/login", json={"username": "nobody", "password": "x"}).status_code == 401
    tail = client.get("/audit/tail", headers=login(client, "aud", api_env)).json()
    events = [(t["actor"], t["action"]) for t in tail]
    assert ("prv1", "login") in events
    assert ("inv", "login_failed") in events and ("nobody", "login_failed") in events
    assert ("system", "service_started") in events


def test_jwt_claims_and_rejections(client, api_env):
    from claimsense.security import read_secret

    tok = login(client, "inv", api_env)["Authorization"].split()[1]
    secret = read_secret("jwt_secret", api_env.secrets)
    c = jwt.decode(tok, secret, algorithms=["HS256"])
    assert c["sub"] == "inv" and c["role"] == "investigator" and c["exp"] > c["iat"]
    now = dt.datetime.now(dt.timezone.utc)
    expired = jwt.encode({"sub": "inv", "role": "investigator", "iat": now - dt.timedelta(hours=2),
                          "exp": now - dt.timedelta(hours=1)}, secret, algorithm="HS256")
    forged = jwt.encode({"sub": "inv", "role": "investigator", "exp": now + dt.timedelta(hours=1)}, "x" * 64,
                        algorithm="HS256")
    bad_role = jwt.encode({"sub": "inv", "role": "admin", "exp": now + dt.timedelta(hours=1)}, secret,
                          algorithm="HS256")
    for t in (expired, forged, bad_role, "garbage"):
        assert client.get("/cases", headers={"Authorization": f"Bearer {t}"}).status_code == 401


# ------------------------------------------------------------------------------ POST /claims
def test_post_claim_requires_service_token(client, api_env):
    c = claim_json("C1")
    assert client.post("/claims", json=c).status_code == 401
    assert post(client, api_env, c, headers=login(client, "inv", api_env)).status_code == 403
    assert post(client, api_env, c, headers=login(client, "prv1", api_env)).status_code == 403
    assert post(client, api_env, c).status_code == 200


def test_post_claim_response_shape(client, api_env):
    r = post(client, api_env, claim_json("CLM46614"))
    assert r.status_code == 200
    d = r.json()
    assert set(d) == CLAIM_OUT_KEYS
    p = api_env.pseudo()
    assert d["claim_token"] == p.token("claim", "CLM46614")
    assert d["provider_token"] == p.token("provider", "PRV51001")
    assert d["action"] in (PAY, RETURN, HOLD)
    assert d["backend"] == "numpy" and d["batch_size"] >= 1 and d["latency_ms"] >= 0
    assert "PRV51001" not in r.text and "BENE11001" not in r.text and "CLM46614" not in r.text


def test_unknown_beneficiary_is_422(client, api_env):
    r = post(client, api_env, claim_json("C1", bene="BENE99999"))
    assert r.status_code == 422 and "beneficiary" in r.text


@pytest.mark.parametrize("over", [{"InscClaimAmtReimbursed": -5}, {"ClaimStartDt": "NA"}, {"is_inpatient": 2},
                                  {"ClaimID": ""}, {"ClaimStartDt": "not-a-date"}])
def test_invalid_claim_body_is_422(client, api_env, over):
    assert post(client, api_env, claim_json("C1", **over)).status_code == 422


def test_na_strings_and_numeric_codes_are_accepted(client, api_env):
    c = claim_json("C1", ClaimEndDt="NA", OperatingPhysician="", ClmProcedureCode_1=7092.0, ClmDiagnosisCode_1=4019)
    r = post(client, api_env, c)
    assert r.status_code == 200, r.text
    inv = login(client, "inv", api_env)
    payload = client.get(f"/cases/{r.json()['claim_token']}", headers=inv).json()["payload"]
    assert payload["ClaimEndDt"] is None and payload["OperatingPhysician"] is None
    assert payload["ClmProcedureCode_1"] == "7092" and payload["ClmDiagnosisCode_1"] == "4019"


def test_resubmitting_same_claim_id_is_409(client, api_env):
    assert post(client, api_env, claim_json("C1")).status_code == 200
    r = post(client, api_env, claim_json("C1", amount=999.0))
    assert r.status_code == 409


def test_claim_after_death_is_returned_with_fix(client, api_env):
    bene = "BENE11002"
    r = post(client, api_env, claim_json("C-DEAD", bene=bene, provider="PRV59999", start="2009-07-01",
                                         end="2009-07-02"))
    d = r.json()
    assert dt.date.fromisoformat("2009-07-01") > dt.date.fromisoformat(DECEASED[bene])
    assert d["action"] == RETURN
    assert [x["rule"] for x in d["reasons"]["rules"]] == ["after_death"]
    from claimsense.store import Store

    st = Store(api_env.db, api_env.envelope())
    note = st.get_claim(d["claim_token"])["notes"]
    assert set(note) == {"correction"}
    assert note["correction"]["status"] == "pending" and note["correction"]["source"] == "template"
    assert "date of death" in note["correction"]["text"]
    st.conn.close()


@pytest.mark.parametrize("bene,inpatient,fires", [("BENE11012", 0, True), ("BENE11012", 1, False),
                                                  ("BENE11011", 1, True), ("BENE11011", 0, False)])
def test_no_coverage_rule_end_to_end(client, api_env, bene, inpatient, fires):
    c = claim_json("C-NC", bene=bene, provider="PRV59998", inpatient=inpatient,
                   **({"AdmissionDt": "2009-03-01", "DischargeDt": "2009-03-03"} if inpatient else {}))
    d = post(client, api_env, c).json()
    rules = [x["rule"] for x in d["reasons"].get("rules", [])]
    assert ("no_coverage" in rules) is fires
    assert d["action"] == (RETURN if fires else PAY)


def test_same_service_new_claim_id_is_flagged_duplicate(client, api_env):
    assert post(client, api_env, claim_json("C1")).json()["action"] == PAY
    d = post(client, api_env, claim_json("C2")).json()
    assert "duplicate" in [x["rule"] for x in d["reasons"]["rules"]]


def test_concurrent_posts_are_batched(api_env):
    with api_env.client(batch_window_ms=50.0) as c:
        headers = service_headers(api_env)

        def send(i):
            return c.post("/claims", json=claim_json(f"C{i}", provider=f"PRV5{i % 4:04d}", amount=100.0 + i,
                                                     bene=f"BENE{11001 + i % 10}"), headers=headers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
            rs = list(ex.map(send, range(50)))
        assert all(r.status_code == 200 for r in rs), [r.text for r in rs if r.status_code != 200][:2]
        sizes = [r.json()["batch_size"] for r in rs]
        assert max(sizes) > 1
        assert len({r.json()["claim_token"] for r in rs}) == 50
        m = c.get("/metrics", headers=login(c, "inv", api_env)).json()
        assert m["claims_total"] == 50 and m["avg_batch_size"] > 1
        assert c.get("/health").json()["providers_tracked"] == 4


# ------------------------------------------------------------------------------ RBAC
ENDPOINTS = [
    ("POST", "/claims", {"service"}),
    ("GET", "/cases", {"investigator"}),
    ("GET", "/cases/{t}", {"investigator"}),
    ("POST", "/cases/{t}/resolve", {"investigator"}),
    ("GET", "/providers/leaderboard", {"investigator"}),
    ("GET", "/my/claims", {"provider"}),
    ("GET", "/audit/verify", {"auditor"}),
    ("GET", "/audit/tail", {"auditor"}),
    ("GET", "/audit/aggregates", {"auditor", "investigator"}),
    ("GET", "/metrics", {"investigator", "auditor"}),
    ("GET", "/stream", {"investigator", "auditor", "provider"}),
]


def test_rbac_matrix(client, api_env):
    seeded = seed_claims(client, api_env)
    t = seeded["h1"]["claim_token"]
    headers = {"service": service_headers(api_env), "investigator": login(client, "inv", api_env),
               "auditor": login(client, "aud", api_env), "provider": login(client, "prv1", api_env)}
    for method, path, allowed in ENDPOINTS:
        url = path.replace("{t}", t)
        body = (claim_json("C-RBAC-new", amount=999.0) if path == "/claims" else
                {"status": "needs_more_info"} if path.endswith("resolve") else None)
        assert client.request(method, url, json=body).status_code == 401, (path, "anonymous")
        for role, h in headers.items():
            if role in allowed and path == "/stream":
                continue    # an allowed /stream never ends; covered by test_sse_*
            code = client.request(method, url, json=body, headers=h).status_code
            if role in allowed:
                assert code == 200, (path, role, code)
            else:
                assert code == 403, (path, role, code)


# ------------------------------------------------------------------------------ provider view
def test_provider_sees_only_own_claims(client, api_env):
    seeded = seed_claims(client, api_env)
    p1 = client.get("/my/claims", headers=login(client, "prv1", api_env)).json()
    p2 = client.get("/my/claims", headers=login(client, "prv2", api_env)).json()
    pseudo = api_env.pseudo()
    assert p1["provider_token"] == pseudo.token("provider", "PRV51001")
    assert {c["claim_token"] for c in p1["claims"]} == {seeded[k]["claim_token"] for k in ("r1", "h1", "h2")}
    assert {c["claim_token"] for c in p2["claims"]} == {seeded["other"]["claim_token"]}
    for c in p1["claims"] + p2["claims"]:
        assert set(c) == {"claim_token", "service_date", "amount", "action", "received_at", "note", "reasons"}
    text = json.dumps(p1)
    assert seeded["other"]["claim_token"] not in text and seeded["pay"]["claim_token"] not in text
    assert "risk score" not in text      # held claims never show the investigator brief
    returned = [c for c in p1["claims"] if c["action"] == RETURN][0]
    assert returned["note"]["source"] == "template" and "attending physician" in returned["note"]["text"]


# ------------------------------------------------------------------------------ investigator
def test_case_list_and_decrypted_detail_is_audited(client, api_env):
    seeded = seed_claims(client, api_env)
    inv = login(client, "inv", api_env)
    cases = client.get("/cases", headers=inv).json()
    assert {c["claim_token"] for c in cases} == {seeded["h1"]["claim_token"], seeded["h2"]["claim_token"]}
    assert all("payload" not in c for c in cases)
    t = seeded["h1"]["claim_token"]
    d = client.get(f"/cases/{t}", headers=inv).json()
    assert d["payload"]["ClaimID"] == "C-H1" and d["payload"]["Provider"] == "PRV51001"
    assert d["payload"]["BeneID"] == "BENE11001"
    assert d["case"]["status"] == "open" and d["notes"]["brief"]["status"] == "pending"
    tail = client.get("/audit/tail", headers=login(client, "aud", api_env)).json()
    viewed = [e for e in tail if e["action"] == "case_viewed"]
    assert viewed and viewed[0]["subject"] == t and viewed[0]["actor"] == "inv"
    assert viewed[0]["details"] == {"decrypted": True}
    assert client.get("/cases/CNOSUCHTOKEN", headers=inv).status_code == 404


def test_resolve_case(client, api_env):
    seeded = seed_claims(client, api_env)
    inv = login(client, "inv", api_env)
    t = seeded["h1"]["claim_token"]
    r = client.post(f"/cases/{t}/resolve", json={"status": "released", "comment": "ok"}, headers=inv)
    assert r.status_code == 200 and r.json() == {"claim_token": t, "status": "released"}
    assert t not in {c["claim_token"] for c in client.get("/cases", headers=inv).json()}
    assert t in {c["claim_token"] for c in client.get("/cases?status=released", headers=inv).json()}
    tail = client.get("/audit/tail", headers=login(client, "aud", api_env)).json()
    assert any(e["action"] == "case_resolved" and e["subject"] == t and e["details"] == {"status": "released"}
               for e in tail)
    # second resolve of the same case: see report (currently overwrites; 404/409 would also be defensible)
    again = client.post(f"/cases/{t}/resolve", json={"status": "confirmed_issue"}, headers=inv)
    assert again.status_code in (200, 404, 409)
    assert client.post("/cases/CNOPE/resolve", json={"status": "released"}, headers=inv).status_code == 404
    pay = seeded["pay"]["claim_token"]
    assert client.post(f"/cases/{pay}/resolve", json={"status": "released"}, headers=inv).status_code == 404
    assert client.post(f"/cases/{t}/resolve", json={"status": "deleted"}, headers=inv).status_code == 422


def test_leaderboard(client, api_env):
    seed_claims(client, api_env)
    lb = client.get("/providers/leaderboard", headers=login(client, "inv", api_env)).json()
    assert [r["provider"] for r in lb] == [api_env.pseudo().token("provider", "PRV51001")]   # only >= 3 claims
    assert lb[0]["claims"] == 4 and 0 <= lb[0]["risk"] <= 1 and isinstance(lb[0]["reasons"], list)


# ------------------------------------------------------------------------------ auditor
def test_aggregates_have_no_tokens(client, api_env):
    seeded = seed_claims(client, api_env)
    a = client.get("/audit/aggregates", headers=login(client, "aud", api_env)).json()
    assert a["total_claims"] == 5
    assert a["decisions"] == {PAY: 1, RETURN: 2, HOLD: 2}
    text = json.dumps(a)
    for v in seeded.values():
        assert v["claim_token"] not in text and v["provider_token"] not in text
    assert "PRV5" not in text and "BENE" not in text


def test_audit_verify_then_tamper(client, api_env):
    seed_claims(client, api_env)
    aud = login(client, "aud", api_env)
    v = client.get("/audit/verify", headers=aud).json()
    assert v["ok"] is True and v["checked"] > 5 and v["first_bad_id"] is None
    tampered = sqlite3.connect(api_env.db)
    tampered.execute("DROP TRIGGER audit_no_update")
    rid = tampered.execute("SELECT id FROM audit WHERE action='claim_scored' ORDER BY id LIMIT 1 OFFSET 2"
                           ).fetchone()[0]
    tampered.execute("UPDATE audit SET details=? WHERE id=?", ('{"action":"PAY"}', rid))
    tampered.commit()
    tampered.close()
    v = client.get("/audit/verify", headers=aud).json()
    assert v["ok"] is False and v["first_bad_id"] == rid and "edited" in v["reason"]


def test_db_file_contains_no_raw_identifiers(client, api_env):
    seed_claims(client, api_env)
    login(client, "inv", api_env)
    con = sqlite3.connect(api_env.db)
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    blob = b"".join(p.read_bytes() for p in api_env.db.parent.iterdir() if p.name.startswith(api_env.db.name))
    for raw in (b"PRV51001", b"PRV51002", b"BENE11001", b"PHY300001", b"C-H1", b"C-PAY"):
        assert raw not in blob, raw


# ------------------------------------------------------------------------------ SSE
async def _first_sse_events(app, path: str, query: str, n_events: int = 2, timeout: float = 5.0):
    """Drive the ASGI app directly (TestClient buffers streaming bodies), disconnect after n events."""
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
            status["code"] = msg["status"]
            status["headers"] = dict(msg.get("headers", []))
        elif msg["type"] == "http.response.body":
            chunks.append(msg.get("body", b""))
            if b"".join(chunks).count(b"\n\n") >= n_events or not msg.get("more_body", False):
                disconnect.set()

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": query.encode(), "root_path": "",
             "headers": [(b"host", b"testserver")], "client": ("127.0.0.1", 1234), "server": ("testserver", 80),
             "state": {}}
    await asyncio.wait_for(app(scope, receive, send), timeout)
    return status, b"".join(chunks).decode()


def test_sse_query_token_gets_hello_event(client, api_env):
    tok = login(client, "prv1", api_env)["Authorization"].split()[1]
    status, body = client.portal.call(_first_sse_events, client.app, "/stream", f"token={tok}")
    assert status["code"] == 200
    assert status["headers"][b"content-type"].startswith(b"text/event-stream")
    first = body.split("\n\n")[0].splitlines()
    assert first[0] == "event: hello"
    hello = json.loads(first[1].removeprefix("data: "))
    assert hello == {"role": "provider", "provider_token": api_env.pseudo().token("provider", "PRV51001")}
    assert client.app.state.svc.hub.subs == {}     # unsubscribed after disconnect


def test_sse_provider_does_not_get_global_metrics(client, api_env):
    seed_claims(client, api_env)
    tok = login(client, "prv2", api_env)["Authorization"].split()[1]
    assert client.get("/metrics", headers={"Authorization": f"Bearer {tok}"}).status_code == 403
    _, body = client.portal.call(_first_sse_events, client.app, "/stream", f"token={tok}", 2, 2.0)
    assert "event: metrics" not in body


def test_sse_investigator_gets_initial_metrics(client, api_env):
    tok = login(client, "inv", api_env)["Authorization"].split()[1]
    _, body = client.portal.call(_first_sse_events, client.app, "/stream", f"token={tok}", 2)
    assert "event: metrics" in body and '"claims_total"' in body


@pytest.mark.parametrize("user,role", [("inv", "investigator"), ("aud", "auditor")])
def test_sse_hello_for_other_roles(client, api_env, user, role):
    tok = login(client, user, api_env)["Authorization"].split()[1]
    _, body = client.portal.call(_first_sse_events, client.app, "/stream", f"token={tok}", 1)
    assert json.loads(body.splitlines()[1].removeprefix("data: "))["role"] == role


def test_sse_rejects_bad_query_tokens(client, api_env):
    assert client.get("/stream?token=garbage").status_code == 401
    assert client.get("/stream").status_code == 401
    assert client.get(f"/stream?token={api_env.service_token}").status_code == 403


def test_hub_filters_events_by_role():
    from claimsense.api.auth import Principal
    from claimsense.api.main import Hub

    async def run():
        hub = Hub()
        qs = {name: hub.subscribe(p) for name, p in {
            "inv": Principal("inv", "investigator"), "aud": Principal("aud", "auditor"),
            "p1": Principal("p1", "provider", "PT1"), "p2": Principal("p2", "provider", "PT2")}.items()}
        hub.publish("claim", {"x": 1}, provider_token="PT1")
        hub.publish("metrics", {"m": 1})
        return {k: [q.get_nowait()[0] for _ in range(q.qsize())] for k, q in qs.items()}

    got = asyncio.run(run())
    # providers only get their own claims; auditors only metrics; global metrics never reach providers
    assert got == {"inv": ["claim", "metrics"], "aud": ["metrics"], "p1": ["claim"], "p2": []}


# ------------------------------------------------------------------------------ restart
def test_restart_restores_provider_state(api_env):
    providers = ["PRV51001", "PRV51002", "PRV51003"]
    with api_env.client() as c:
        for i in range(7):
            assert post(c, api_env, claim_json(f"C{i}", provider=providers[i % 3], amount=100.0 + i)).status_code == 200
        before = c.get("/health").json()["providers_tracked"]
        states = {k: v.n for k, v in c.app.state.svc.engine.providers.items()}
    assert before == 3
    with api_env.client() as c2:
        assert c2.get("/health").json()["providers_tracked"] == len(set(providers))
        assert {k: v.n for k, v in c2.app.state.svc.engine.providers.items()} == states
        # the duplicate index is restored too: same service under a new claim number
        d = post(c2, api_env, claim_json("C-new", provider="PRV51001", amount=100.0)).json()
        assert "duplicate" in [x["rule"] for x in d["reasons"]["rules"]]
        # and an old ClaimID is still refused
        assert post(c2, api_env, claim_json("C0", provider="PRV51001", amount=100.0)).status_code == 409
        v = c2.get("/audit/verify", headers=login(c2, "aud", api_env)).json()
        assert v["ok"] is True
