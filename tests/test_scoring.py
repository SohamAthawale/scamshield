"""scoring.Engine: same features as training, batch == single, reasons, replay, bootstrap, watchlist."""
import json

import numpy as np
import pandas as pd
import pytest

from scamshield.features import ANOMALY_FEATURES, FEATURES, FeatureContext, rule_hits
from scamshield.fusion import ALLOW, HOLD, WARN, Policy
from scamshield.models import Artifacts, Standardizer
from scamshield.scoring import AE_IDX, Engine, PaymentInput
from scamshield.training import decisions, rule_matrix

from conftest import policy_for, scorers


def items_for(df: pd.DataFrame, ctx: FeatureContext) -> list[PaymentInput]:
    out = []
    for r in df.to_dict("records"):
        ts = pd.Timestamp(r["ts"])
        out.append(PaymentInput(ts.value / 6e10, r["sender"], r["receiver"], ctx.to_usd(r["amount_paid"], r["currency_paid"]),
                                r["payment_format"], r["from_bank"] != r["to_bank"],
                                r["currency_paid"] != r["currency_received"], ts.hour))
    return out


@pytest.fixture(scope="module")
def models(aml):
    return {b: scorers(aml.models_dir, b) for b in ("numpy", "cpu")}


@pytest.fixture
def engine(aml, models):
    def make(backend="numpy", policy=None, bootstrap=True):
        a, r = models[backend]
        eng = Engine(aml.ctx, a, r, policy or policy_for(aml.models_dir))
        if bootstrap:
            eng.bootstrap(aml.tx[aml.split != "test"])
        return eng
    return make


@pytest.fixture
def live(aml):
    return items_for(aml.tx[aml.split == "test"], aml.ctx)


# ------------------------------------------------------------------------------ parity with training
def test_engine_features_equal_training_features(aml, engine, live):
    res = engine().score_batch(live)
    X = np.array([[r.features[k] for k in FEATURES] for r in res], dtype=np.float32)
    np.testing.assert_array_equal(X, aml.X[aml.split == "test"])
    assert all(list(r.features) == FEATURES for r in res)


def test_engine_decisions_equal_vectorised_training_decisions(aml, engine, live, models):
    res = engine().score_batch(live)
    Xt = aml.X[aml.split == "test"]
    a, r = models["numpy"]
    err, _, _ = a.score(Xt[:, AE_IDX])
    risk = r.predict(Xt)
    expected = decisions(risk, err, rule_matrix(Xt, aml.ctx), policy_for(aml.models_dir))
    assert [x.decision.action for x in res] == list(expected)
    np.testing.assert_allclose([x.decision.risk for x in res], risk, rtol=0, atol=1e-12)
    np.testing.assert_allclose([x.decision.anomaly for x in res], err, rtol=1e-6)
    assert {ALLOW, WARN, HOLD} <= set(expected)
    assert [x.decision.rules for x in res] == [rule_hits(x, aml.ctx) for x in Xt]


@pytest.mark.parametrize("backend", ["numpy", "cpu"])
def test_batch_equals_single_calls(engine, live, backend):
    batch = engine(backend).score_batch(live)
    e2 = engine(backend)
    single = [e2.score(it) for it in live]
    assert [b.decision.action for b in batch] == [s.decision.action for s in single]
    assert [b.features for b in batch] == [s.features for s in single]
    np.testing.assert_allclose([b.decision.anomaly for b in batch], [s.decision.anomaly for s in single], rtol=1e-5)
    np.testing.assert_allclose([b.decision.risk for b in batch], [s.decision.risk for s in single], atol=1e-7)


def test_chunked_batches_equal_one_batch(engine, live):
    whole = engine().score_batch(live)
    e2 = engine()
    chunked = [r for i in range(0, len(live), 7) for r in e2.score_batch(live[i:i + 7])]
    assert [a.decision.action for a in whole] == [b.decision.action for b in chunked]
    assert [a.features for a in whole] == [b.features for b in chunked]


def test_numpy_and_torch_backends_agree(engine, live):
    n = engine("numpy").score_batch(live)
    t = engine("cpu").score_batch(live)
    assert [a.decision.action for a in n] == [b.decision.action for b in t]
    np.testing.assert_allclose([a.decision.anomaly for a in n], [b.decision.anomaly for b in t], atol=1e-4)


def test_score_batch_empty(engine):
    assert engine().score_batch([]) == []


# ------------------------------------------------------------------------------ state
def test_bootstrap_counts_and_matches_running_the_history(aml, engine):
    eng = engine(bootstrap=False)
    hist = aml.tx[aml.split != "test"]
    assert eng.bootstrap(hist) == len(hist)
    accounts = set(hist["sender"]) | set(hist["receiver"])
    assert set(eng.fe.ids) == accounts


def test_replay_rebuilds_state_without_scoring(aml, engine, live):
    k = 40
    uninterrupted = engine()
    uninterrupted.score_batch(live[:k])
    nxt_a = uninterrupted.score_batch(live[k:])
    restarted = engine()
    restarted.replay(live[:k])
    assert restarted.watch == {}                                  # replay never touches the watchlist
    nxt_b = restarted.score_batch(live[k:])
    assert [a.features for a in nxt_a] == [b.features for b in nxt_b]
    assert [a.decision.action for a in nxt_a] == [b.decision.action for b in nxt_b]


def test_without_history_a_known_customer_looks_new(aml, engine, live):
    from conftest import CUSTOMER_1

    cold = engine(bootstrap=False).score_batch(live)
    warm = engine().score_batch(live)
    i = next(j for j, it in enumerate(live) if it.sender_key == ":".join(CUSTOMER_1))
    assert cold[i].features["s_is_new"] == 1 and warm[i].features["s_is_new"] == 0


# ------------------------------------------------------------------------------ reasons
def test_reasons_follow_the_decision(aml, engine, live):
    pol = policy_for(aml.models_dir)
    res = engine().score_batch(live)
    for r in res:
        d = r.decision
        if d.action == ALLOW:
            assert r.reasons == {}
            continue
        assert [x["rule"] for x in r.reasons["rules"]] == d.rules
        assert ("customer" in r.reasons) == d.anomaly_flag
        assert ("drivers" in r.reasons) == (d.risk >= pol.warn_threshold)
        if "customer" in r.reasons:
            assert 1 <= len(r.reasons["customer"]) <= 3
            assert all(c["feature"] in ANOMALY_FEATURES + ["hour"] for c in r.reasons["customer"])
        for x in r.reasons.get("drivers", []):
            assert x["contribution"] > 0 and (x["feature"] in FEATURES or x["feature"] == "hour")
        # every non-ALLOW decision carries at least one kind of reason (HOLD implies risk >= warn -> drivers)
        assert r.reasons["rules"] or "customer" in r.reasons or "drivers" in r.reasons


def test_explain_all(engine, live):
    res = engine().score_batch(live, explain_all=True)
    assert all("rules" in r.reasons for r in res)


def test_rule_payment_is_at_least_warned(aml, engine, live):
    from conftest import CUSTOMER_1

    res = engine().score_batch(live)
    i = next(j for j, it in enumerate(live) if it.sender_key == ":".join(CUSTOMER_1) and it.amount_usd == 5000.0)
    assert res[i].decision.rules == ["big_new_payee"] and res[i].decision.action in (WARN, HOLD)
    assert res[i].reasons["rules"][0]["rule"] == "big_new_payee"


# ------------------------------------------------------------------------------ watchlist
def test_watchlist(engine, live):
    eng = engine()
    res = eng.score_batch(live)
    by = {}
    for it, r in zip(live, res):
        w = by.setdefault(it.receiver_key, [])
        w.append(r.decision)
    wl = eng.watchlist(top=1000, min_payments=1)
    # only receivers with real risk (some inbound payment at or above the warn threshold) are listed
    risky = {k for k, ds in by.items() if max(d.risk for d in ds) >= eng.policy.warn_threshold}
    assert {w["account"] for w in wl} == risky
    for w in wl:
        ds = by[w["account"]]
        assert w["payments"] == len(ds)
        assert w["mean_risk"] == pytest.approx(np.mean([d.risk for d in ds]))
        assert w["max_risk"] == pytest.approx(max(d.risk for d in ds))
        assert w["flagged"] == sum(d.action != ALLOW for d in ds)
    keys = [(-w["mean_risk"], -w["payments"]) for w in wl]
    assert keys == sorted(keys)
    assert all(w["payments"] >= 2 for w in eng.watchlist(top=1000))
    assert len(eng.watchlist(top=3, min_payments=1)) == min(3, len(risky))


def test_live_mule_is_on_the_watchlist(engine, live):
    from conftest import LIVE_MULE

    eng = engine()
    eng.score_batch(live)
    wl = {w["account"]: w for w in eng.watchlist(top=1000)}
    assert wl[":".join(LIVE_MULE)]["payments"] == 5


# ------------------------------------------------------------------------------ construction
def test_engine_rejects_mismatched_models(aml, tmp_path, models):
    import shutil

    a, b = Artifacts(aml.models_dir), Artifacts(tmp_path)
    shutil.copy(a.ae_numpy, b.ae_numpy)
    s = Standardizer.load(a.scaler)
    Standardizer(list(reversed(s.features)), s.mean[::-1], s.std[::-1]).save(b.scaler)
    from scamshield.inference import load_anomaly_scorer

    with pytest.raises(ValueError, match="ANOMALY_FEATURES"):
        Engine(aml.ctx, load_anomaly_scorer(tmp_path, "numpy"), models["numpy"][1], Policy(1, 0.5, 0.9))

    class WrongRisk:
        features = list(reversed(FEATURES))

    with pytest.raises(ValueError, match="FEATURES"):
        Engine(aml.ctx, models["numpy"][0], WrongRisk(), Policy(1, 0.5, 0.9))


def test_from_artifacts(aml, fresh_models_dir, tmp_path, models):
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps({"anomaly_threshold": None, "warn_threshold": None, "hold_threshold": 0.99}))
    a, r = models["numpy"]
    eng = Engine.from_artifacts(fresh_models_dir, pol, a, r)
    learned = json.loads((fresh_models_dir / "thresholds.json").read_text())
    assert eng.policy == Policy(learned["anomaly_threshold"], learned["warn_threshold"], 0.99)
    assert eng.ctx == aml.ctx
    (fresh_models_dir / "thresholds.json").unlink()
    pol.write_text(json.dumps({"anomaly_threshold": 3, "warn_threshold": 0.4, "hold_threshold": 0.9}))
    assert Engine.from_artifacts(fresh_models_dir, pol, a, r).policy == Policy(3.0, 0.4, 0.9)
