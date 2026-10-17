"""simulator.py: stream loading, mule accounts from the pattern file, injected payments, and one replay."""
import asyncio
import json
import random
import statistics

import pytest

from scamshield import simulator
from scamshield.simulator import FIELDS, INJECTIONS, inject, load_stream, mule_accounts

from conftest import LIVE_MULE, SPLITS

BASE = {"payment_id": "T5", "ts": "2022-09-07 10:00", "from_bank": "0119", "from_acct": "8C0FFEE01", "to_bank": "001",
        "to_acct": "8ABCDEF01", "amount_paid": 120.0, "currency_paid": "Euro", "amount_received": 132.0,
        "currency_received": "US Dollar", "payment_format": "Cheque"}
HISTORY = {"0119:8C0FFEE01": [100.0, 120.0, 90.0, 110.0]}
MULES = [("021174", "80BADBAD4"), ("010", "8DEADBEEF")]


def test_load_stream(api_env):
    rows = load_stream(api_env.stream_file, None)
    n = int((api_env.aml.split == "test").sum())
    assert len(rows) == n and [r["payment_id"] for r in rows[:3]] == ["T0", "T1", "T2"]
    assert set(rows[0]) == set(FIELDS) | {"payment_id"}
    assert isinstance(rows[0]["amount_paid"], float) and isinstance(rows[0]["amount_received"], float)
    assert all(isinstance(rows[0][k], str) for k in ("from_bank", "from_acct", "ts", "payment_format"))
    assert all(len(r["from_bank"]) >= 3 for r in rows)                      # leading zeros kept
    assert rows[0]["ts"] >= SPLITS[1]
    # ids are positions in the FILE, so offset/limit keep them stable
    part = load_stream(api_env.stream_file, 5, offset=10)
    assert [r["payment_id"] for r in part] == [f"T{i}" for i in range(10, 15)]
    assert part == rows[10:15]


def test_loaded_stream_validates_as_api_payments(api_env):
    from scamshield.api.schemas import PaymentIn

    for r in load_stream(api_env.stream_file, None):
        p = PaymentIn(**r)
        assert p.sender == f"{r['from_bank']}:{r['from_acct']}"


def test_mule_accounts(aml):
    mules = mule_accounts(aml.raw_dir, start=SPLITS[1])
    assert LIVE_MULE in mules
    live_attempt = next(att for kind, att in aml.attempts if att[0].ts >= __import__("pandas").Timestamp(SPLITS[1]))
    assert set(mules) == {(r.to_bank, r.to_acct) for r in live_attempt}
    everything = mule_accounts(aml.raw_dir, start="2000-01-01")
    assert len(everything) > len(mules) and mules == sorted(mules)
    assert mule_accounts(aml.raw_dir / "nowhere") == []


@pytest.mark.parametrize("kind", INJECTIONS)
def test_inject_scales_the_senders_usual_amount(kind):
    rng = random.Random(1)
    c = inject(BASE, kind, HISTORY, MULES, rng, 7)
    usual = statistics.median(HISTORY["0119:8C0FFEE01"])
    assert c["payment_id"] == f"INJ7-{kind}"
    assert 8 * usual <= c["amount_paid"] <= 30 * usual and c["amount_received"] == c["amount_paid"]
    assert c["currency_received"] == c["currency_paid"] == "Euro"
    assert c["payment_format"] in ("ACH", "Wire")
    assert (c["from_bank"], c["from_acct"], c["ts"]) == (BASE["from_bank"], BASE["from_acct"], BASE["ts"])
    assert BASE["amount_paid"] == 120.0 and BASE["payment_id"] == "T5"     # base untouched


def test_inject_victim_to_mule_pays_a_known_mule():
    for seed in range(10):
        c = inject(BASE, "victim_to_mule", HISTORY, MULES, random.Random(seed), 0)
        assert (c["to_bank"], c["to_acct"]) in MULES


def test_inject_honest_new_payee_pays_a_brand_new_account():
    c = inject(BASE, "honest_new_payee", HISTORY, MULES, random.Random(0), 42)
    assert c["to_acct"] == "NEW0000042" and c["to_bank"] == BASE["to_bank"]


@pytest.mark.parametrize("history,mules,kind", [
    ({}, MULES, "honest_new_payee"),                                         # no history
    ({"0119:8C0FFEE01": [100.0, 100.0]}, MULES, "honest_new_payee"),        # < 3 prior amounts
    ({"0119:8C0FFEE01": [0.0, 0.0, 0.0, 5.0]}, MULES, "honest_new_payee"),  # usual <= 0
    (HISTORY, [], "victim_to_mule"),                                         # no mules
])
def test_inject_returns_none_when_impossible(history, mules, kind):
    assert inject(BASE, kind, history, mules, random.Random(0), 0) is None


def test_honest_new_payee_works_without_mules():
    assert inject(BASE, "honest_new_payee", HISTORY, [], random.Random(0), 0) is not None


# ------------------------------------------------------------------------------ replay against the app
class _ClientOverTestClient:
    """Stands in for httpx.AsyncClient and forwards to a running TestClient (whose loop runs the app)."""

    def __init__(self, tc, base_url, headers, timeout, limits):
        self.tc, self.headers = tc, headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, path, json):
        return await asyncio.to_thread(self.tc.post, path, json=json, headers=self.headers)


def _run(api_env, tmp_path, monkeypatch, payments, inject_rate, raw_dir):
    out = tmp_path / "sim"
    with api_env.client() as tc:
        monkeypatch.setattr(simulator.httpx, "AsyncClient", lambda **kw: _ClientOverTestClient(tc, **kw))
        summary = asyncio.run(simulator.run("http://test", api_env.service_token, payments, rate=0, concurrency=1,
                                            inject_rate=inject_rate, raw_dir=raw_dir, seed=3, out_dir=out))
    return summary, out


def test_run_replays_and_reports_injections(api_env, tmp_path, monkeypatch, aml):
    import pandas as pd

    # mules must be live accounts in the pattern file; use the whole file so this tiny stream has some
    monkeypatch.setattr(simulator, "mule_accounts", lambda raw, start="2022-09-09": mule_accounts(raw, "2000-01-01"))
    payments = load_stream(api_env.stream_file, None)
    summary, out = _run(api_env, tmp_path, monkeypatch, payments, 0.5, aml.raw_dir)
    inj = summary["injected"]
    n_inj = sum(v["injected"] for v in inj.values())
    assert summary["sent"] == len(payments) + n_inj and n_inj > 0
    assert sum(summary["outcomes"].values()) == summary["sent"]
    assert not any(k.startswith(("http_", "error:")) for k in summary["outcomes"]), summary["outcomes"]
    for k in INJECTIONS:
        assert inj[k]["warned"] + inj[k]["held"] + inj[k]["allowed"] == inj[k]["injected"]
    assert abs(inj["victim_to_mule"]["injected"] - inj["honest_new_payee"]["injected"]) <= 1   # round robin
    saved = json.loads(next(out.glob("replay-*.json")).read_text())
    assert saved["sent"] == summary["sent"]
    csv = pd.read_csv(next(out.glob("injected-*.csv")))
    assert len(csv) == n_inj and set(csv["kind"]) <= set(INJECTIONS)


def test_run_without_injection(api_env, tmp_path, monkeypatch, aml):
    payments = load_stream(api_env.stream_file, 20)
    summary, _ = _run(api_env, tmp_path, monkeypatch, payments, 0.0, aml.raw_dir)
    assert summary["sent"] == 20 and all(v["injected"] == 0 for v in summary["injected"].values())


def test_run_still_injects_honest_payees_when_there_are_no_mules(api_env, tmp_path, monkeypatch):
    payments = load_stream(api_env.stream_file, None)
    summary, _ = _run(api_env, tmp_path, monkeypatch, payments, 0.5, tmp_path / "no-raw-dir")
    assert summary["injected"]["honest_new_payee"]["injected"] > 0
