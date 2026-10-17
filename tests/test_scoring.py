"""scoring.Engine on tiny artifacts: order, provider state, batch == single, duplicates, reasons, snapshot."""
import json

import numpy as np
import pytest

from claimsense.features import (CLAIM_FEATURES, PROVIDER_FEATURES, FeatureContext, ProviderState,
                                 claim_features, hard_errors)
from claimsense.fusion import HOLD, PAY, RETURN, Policy
from claimsense.inference import load_anomaly_scorer, load_provider_scorer
from claimsense.models import Artifacts, Standardizer
from claimsense.scoring import ClaimInput, Engine

from conftest import attending_key, make_claims


def to_items(df):
    return [ClaimInput(claim=r, bene=r, provider_key=r["Provider"], bene_key=r["BeneID"],
                       attending_key=attending_key(r)) for r in df.to_dict("records")]


@pytest.fixture(scope="module")
def scorers(tiny_models_dir):
    return {
        "numpy": load_anomaly_scorer(tiny_models_dir, backend="numpy"),
        "cpu": load_anomaly_scorer(tiny_models_dir, backend="cpu"),
        "provider": load_provider_scorer(tiny_models_dir),
    }


@pytest.fixture(scope="module")
def claims():
    return make_claims(seed=42, n=240, n_providers=5, unseen_share=0.1)


@pytest.fixture(scope="module")
def median_error(scorers, claims, ctx):
    X = np.array([[claim_features(r, r, ctx)[k] for k in CLAIM_FEATURES] for r in claims.to_dict("records")])
    err, _, _ = scorers["numpy"].score(X)
    return float(np.median(err))


@pytest.fixture
def make_engine(scorers, ctx, median_error):
    def _make(anomaly_threshold=None, provider_threshold=0.5, min_provider_claims=10, backend="numpy"):
        thr = median_error if anomaly_threshold is None else anomaly_threshold
        return Engine(ctx, scorers[backend], scorers["provider"], Policy(thr, provider_threshold, min_provider_claims))
    return _make


def test_score_batch_preserves_order_and_length(make_engine, claims, ctx):
    items = to_items(claims)
    out = make_engine().score_batch(items)
    assert len(out) == len(items)
    for it, res in zip(items, out):
        assert res.claim_features == claim_features(it.claim, it.bene, ctx)
        assert list(res.provider_features) == PROVIDER_FEATURES
        assert res.model_ms >= 0


def test_score_batch_empty(make_engine):
    assert make_engine().score_batch([]) == []


def test_provider_state_count_increments(make_engine, claims):
    eng = make_engine()
    items = to_items(claims)
    out = eng.score_batch(items)
    seen = {}
    for it, res in zip(items, out):
        seen[it.provider_key] = seen.get(it.provider_key, 0) + 1
        assert res.provider_claims == seen[it.provider_key]
    assert {k: s.n for k, s in eng.providers.items()} == seen
    res = eng.score(items[0])
    assert res.provider_claims == seen[items[0].provider_key] + 1


def test_provider_features_match_a_standalone_provider_state(make_engine, claims, ctx):
    eng = make_engine()
    items = to_items(claims)
    out = eng.score_batch(items)
    states = {}
    for it, res in zip(items, out):
        st = states.setdefault(it.provider_key, ProviderState())
        st.update(claim_features(it.claim, it.bene, ctx), it.bene_key, it.attending_key, ctx.rare_pair_threshold)
        assert res.provider_features == st.features()


@pytest.mark.parametrize("backend", ["numpy", "cpu"])
def test_batch_equals_sequential_single_calls(make_engine, claims, backend):
    items = to_items(claims)
    batch = make_engine(backend=backend, min_provider_claims=5).score_batch(items)
    eng = make_engine(backend=backend, min_provider_claims=5)
    single = [eng.score(it) for it in items]
    assert [r.decision.action for r in batch] == [r.decision.action for r in single]
    assert [r.decision.hard_errors for r in batch] == [r.decision.hard_errors for r in single]
    np.testing.assert_allclose([r.decision.anomaly for r in batch], [r.decision.anomaly for r in single],
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose([r.decision.provider_risk for r in batch],
                               [r.decision.provider_risk for r in single], atol=1e-6)
    # the policy actually exercises more than one branch
    assert len({r.decision.action for r in batch}) >= 2


def test_batch_split_into_chunks_equals_one_batch(make_engine, claims):
    items = to_items(claims)
    whole = make_engine().score_batch(items)
    eng = make_engine()
    chunked = [r for i in range(0, len(items), 17) for r in eng.score_batch(items[i:i + 17])]
    assert [r.decision.action for r in whole] == [r.decision.action for r in chunked]


def test_duplicate_flagged_on_second_identical_claim(make_engine, claims):
    eng = make_engine(anomaly_threshold=float("inf"))
    item = to_items(claims.iloc[[0]])[0]
    first, second = eng.score(item), eng.score(item)
    assert "duplicate" not in first.decision.hard_errors
    assert "duplicate" in second.decision.hard_errors
    assert second.decision.action != PAY
    assert any(r["rule"] == "duplicate" for r in second.reasons["rules"])


def test_duplicate_within_one_batch(make_engine, claims):
    item = to_items(claims.iloc[[3]])[0]
    out = make_engine(anomaly_threshold=float("inf")).score_batch([item, item, item])
    assert ["duplicate" in r.decision.hard_errors for r in out] == [False, True, True]


def test_duplicate_depends_on_pseudonymised_keys(make_engine, claims):
    eng = make_engine(anomaly_threshold=float("inf"))
    it = to_items(claims.iloc[[0]])[0]
    other = ClaimInput(it.claim, it.bene, "OTHER-PROVIDER", it.bene_key, it.attending_key)
    eng.score(it)
    assert "duplicate" not in eng.score(other).decision.hard_errors


def test_hard_errors_match_rules(make_engine, claims):
    out = make_engine(anomaly_threshold=float("inf"), provider_threshold=float("inf")).score_batch(to_items(claims))
    for r, res in zip(claims.to_dict("records"), out):
        assert [e for e in res.decision.hard_errors if e != "duplicate"] == hard_errors(r, r)
        assert res.decision.action == (PAY if not res.decision.hard_errors else RETURN)


def test_reasons_present_for_every_non_pay_decision(make_engine, claims):
    out = make_engine(provider_threshold=0.0, min_provider_claims=3).score_batch(to_items(claims))
    actions = {r.decision.action for r in out}
    assert {PAY, RETURN, HOLD} <= actions
    for res in out:
        d = res.decision
        if d.action == PAY:
            assert res.reasons == {}
            continue
        assert "rules" in res.reasons
        assert res.reasons["rules"] or res.reasons.get("claim"), "non-PAY decision with no claim-level reason"
        if d.anomaly_flag:
            assert len(res.reasons["claim"]) == 3
            assert all(c["feature"] in CLAIM_FEATURES for c in res.reasons["claim"])
        else:
            assert "claim" not in res.reasons
        assert [r["rule"] for r in res.reasons["rules"]] == d.hard_errors
        if d.action == HOLD:
            assert "provider" in res.reasons
            assert all(p["contribution"] > 0 for p in res.reasons["provider"])
        if not d.provider_ready:
            assert "provider" not in res.reasons


def test_explain_all_gives_reasons_for_pay(make_engine, claims):
    out = make_engine(anomaly_threshold=float("inf")).score_batch(to_items(claims), explain_all=True)
    pays = [r for r in out if r.decision.action == PAY]
    assert pays and all("rules" in r.reasons for r in pays)


def test_provider_not_ready_below_min_claims(make_engine, claims):
    eng = make_engine(anomaly_threshold=0.0, provider_threshold=0.0, min_provider_claims=4)
    prv = claims["Provider"].iloc[0]
    out = eng.score_batch(to_items(claims[claims["Provider"] == prv].iloc[:6]))
    assert [r.decision.provider_ready for r in out] == [False, False, False, True, True, True]
    assert [r.decision.action for r in out] == [RETURN] * 3 + [HOLD] * 3


def test_provider_snapshot_sorted_and_respects_min_claims(make_engine, claims):
    eng = make_engine(min_provider_claims=30)
    eng.score_batch(to_items(claims))
    counts = claims["Provider"].value_counts()
    snap = eng.provider_snapshot()
    assert {s["provider"] for s in snap} == set(counts[counts >= 30].index)
    assert all(s["claims"] == counts[s["provider"]] for s in snap)
    risks = [s["risk"] for s in snap]
    assert risks == sorted(risks, reverse=True)
    assert all(0 <= r <= 1 for r in risks)
    # same numbers as scoring the provider state directly
    for s in snap:
        f = eng.providers[s["provider"]].features()
        expected = eng.provider.predict(np.array([[f[c] for c in PROVIDER_FEATURES]], dtype=np.float32))[0]
        assert s["risk"] == pytest.approx(expected)


def test_provider_snapshot_empty(make_engine, claims):
    eng = make_engine(min_provider_claims=10_000)
    assert eng.provider_snapshot() == []
    eng.score_batch(to_items(claims.iloc[:5]))
    assert eng.provider_snapshot() == []


def test_engine_rejects_scaler_with_wrong_feature_order(tmp_path, tiny_models_dir, scorers, ctx):
    import shutil

    a, b = Artifacts(tiny_models_dir), Artifacts(tmp_path)
    shutil.copy(a.ae_numpy, b.ae_numpy)
    s = Standardizer.load(a.scaler)
    Standardizer(list(reversed(s.features)), s.mean[::-1], s.std[::-1]).save(b.scaler)
    anomaly = load_anomaly_scorer(tmp_path, backend="numpy")
    with pytest.raises(ValueError, match="feature"):
        Engine(ctx, anomaly, scorers["provider"], Policy(1, 1))


def test_from_artifacts(tiny_models_dir, tmp_path, scorers, claims):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"anomaly_threshold": None, "provider_threshold": 0.9,
                                  "min_provider_claims": None}))
    eng = Engine.from_artifacts(tiny_models_dir, policy, scorers["numpy"], scorers["provider"])
    assert eng.policy == Policy(0.5, 0.9, 10)    # learned anomaly + min claims, manual provider threshold
    assert eng.ctx == FeatureContext.from_json(Artifacts(tiny_models_dir).feature_context.read_text())
    assert len(eng.score_batch(to_items(claims.iloc[:10]))) == 10
