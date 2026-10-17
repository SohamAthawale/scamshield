"""training.py: threshold tuning, ranking metrics and the stream file, on small synthetic inputs only."""
import numpy as np
import pandas as pd
import pytest

from claimsense import training
from claimsense.training import (STREAM_COLUMNS, decision_summary, hold_curve, ranking_metrics, simulate_stream,
                                 tune_provider_threshold, write_stream)
from claimsense.simulator import load_stream

from conftest import make_claims

ANOM_T, MIN_CLAIMS = 1.0, 10


def group(n, risk, label, rules="missing_attending", anomaly=0.0, claims=50):
    return pd.DataFrame({"anomaly": [anomaly] * n, "rules": [rules] * n, "provider_claims": [claims] * n,
                         "provider_risk": [risk] * n, "label": [label] * n})


@pytest.fixture
def sim():
    """100 flagged+ready claims: 10 @0.9 fraud, 10 @0.6 fraud, 10 @0.5 honest, 70 @0.1 honest,
    plus noise that must never be held: unflagged high-risk honest claims and not-ready claims."""
    return pd.concat([
        group(10, 0.9, 1), group(10, 0.6, 1), group(10, 0.5, 0), group(70, 0.1, 0),
        group(20, 0.99, 0, rules="", anomaly=0.5),               # not flagged (no rule, anomaly below)
        group(20, 0.99, 0, rules=None, anomaly=0.2),             # NaN rules = no rule
        group(20, 0.99, 0, claims=MIN_CLAIMS - 1),               # flagged but provider not ready
        group(5, 0.95, 1, rules="", anomaly=ANOM_T),             # flagged by anomaly (inclusive threshold)
        group(5, np.nan, 0),                                     # no risk score
    ], ignore_index=True)


def test_hold_curve_shape_and_monotonic(sim):
    c = hold_curve(sim, ANOM_T, MIN_CLAIMS)
    assert list(c.columns) == ["threshold", "hold_share", "precision", "held"]
    assert c["threshold"].iloc[0] == 0.30 and c["threshold"].iloc[-1] == 0.99 and len(c) == 70
    assert (np.diff(c["held"]) <= 0).all()
    np.testing.assert_allclose(c["hold_share"], c["held"] / len(sim))


def test_hold_curve_counts(sim):
    c = hold_curve(sim, ANOM_T, MIN_CLAIMS).set_index("threshold")
    assert c.loc[0.50, "held"] == 35        # 10 + 10 + 10 + 5 anomaly-flagged
    assert c.loc[0.50, "precision"] == pytest.approx(25 / 35)
    assert c.loc[0.51, "held"] == 25 and c.loc[0.51, "precision"] == 1.0
    assert c.loc[0.61, "held"] == 15
    assert c.loc[0.96, "held"] == 0 and np.isnan(c.loc[0.96, "precision"])


@pytest.mark.parametrize("cap,min_prec,expected", [
    # 170 claims; held at t<=0.50: 35 (precision 25/35 = 0.714), t<=0.60: 25, t<=0.90: 15, t<=0.95: 5
    (0.50, 0.70, 0.30),     # 35/170 = 0.206 <= cap and 0.714 >= 0.70 from the lowest threshold up
    (0.50, 0.72, 0.51),     # precision binds
    (0.20, 0.70, 0.51),     # cap binds: 35/170 > 0.20, 25/170 = 0.147
    (0.10, 0.70, 0.61),     # 15/170 = 0.088
    (0.05, 0.70, 0.91),     # 5/170 = 0.029 (only the anomaly-flagged 0.95 group)
    (0.50, 0.60, 0.30),     # everything up from 0.30 meets 0.6 precision
    (0.50, 1.01, 0.95),     # impossible -> 0.95
    (0.0, 0.70, 0.95),      # cap 0 is never met by a non-empty hold set (and empty sets have NaN precision)
])
def test_tune_provider_threshold(sim, cap, min_prec, expected):
    assert tune_provider_threshold(sim, ANOM_T, MIN_CLAIMS, cap, min_prec) == pytest.approx(expected)


def test_tune_returns_lowest_threshold_meeting_both(sim):
    t = tune_provider_threshold(sim, ANOM_T, MIN_CLAIMS, 0.2, 0.7)
    c = hold_curve(sim, ANOM_T, MIN_CLAIMS)
    ok = c[(c["precision"] >= 0.7) & (c["hold_share"] <= 0.2)]
    assert t == ok["threshold"].min()
    lower = c[c["threshold"] < t]
    assert not ((lower["precision"] >= 0.7) & (lower["hold_share"] <= 0.2)).any()


def test_tune_on_sim_with_no_flagged_claims():
    s = group(50, 0.9, 1, rules="", anomaly=0.0)
    assert tune_provider_threshold(s, ANOM_T, MIN_CLAIMS) == 0.95


def test_ranking_metrics_known_example():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1])
    m = ranking_metrics(y, p, top_frac=0.5)
    assert m["n"] == 4 and m["positives"] == 2 and m["base_rate"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.75)                   # 3 of 4 positive/negative pairs ordered
    assert m["pr_auc"] == pytest.approx((1 + 2 / 3) / 2)         # precision at each hit: 1/1, 2/3
    assert m["precision_top50pct"] == 0.5 and m["recall_top50pct"] == 0.5


def test_ranking_metrics_perfect_and_default_top5():
    y = np.array([1] * 5 + [0] * 95)
    p = np.linspace(1, 0, 100)
    m = ranking_metrics(y, p)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert m["precision_top5pct"] == 1.0 and m["recall_top5pct"] == 1.0


def test_ranking_metrics_top_k_at_least_one():
    m = ranking_metrics(np.array([0, 1, 0]), np.array([0.1, 0.9, 0.2]), top_frac=0.01)
    assert m["precision_top1pct"] == 1.0 and m["recall_top1pct"] == 1.0


def test_decision_summary():
    s = pd.DataFrame({"action": ["PAY", "PAY", "HOLD_FOR_REVIEW", "RETURN_WITH_FIX"], "label": [0, 1, 1, 0],
                      "amount": [10.0, 20.0, 300.0, 40.0]})
    d = decision_summary(s)
    assert d["claims"] == 4 and d["mix"] == {"PAY": 0.5, "HOLD_FOR_REVIEW": 0.25, "RETURN_WITH_FIX": 0.25}
    assert d["hold_precision_flagged_provider"] == 1.0 and d["return_share_from_flagged_provider"] == 0.0
    assert d["dollars_held"] == 300.0 and d["base_rate_flagged_provider_claims"] == 0.5
    assert decision_summary(s[s["action"] == "PAY"])["hold_precision_flagged_provider"] is None


# ------------------------------------------------------------------------------ stream file + simulated stream
@pytest.fixture
def split_claims():
    c = make_claims(seed=9, n=120, n_providers=6, messy=False)
    prov = sorted(c["Provider"].unique())
    split = {p: ("test" if i % 2 else "train") for i, p in enumerate(prov)}
    return c.assign(split=c["Provider"].map(split), label=c["Provider"].map(lambda p: int(p.endswith(("1", "3")))))


def test_write_stream_only_test_providers_in_time_order(split_claims, tmp_path):
    path = write_stream(split_claims, tmp_path)
    s = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(s.columns) == STREAM_COLUMNS
    assert "label" not in s.columns and "split" not in s.columns
    test = split_claims[split_claims["split"] == "test"]
    assert s["ClaimID"].tolist() == test["ClaimID"].tolist()
    assert s["ClaimStartDt"].tolist() == sorted(s["ClaimStartDt"].tolist())
    assert s["ClaimStartDt"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()


def test_stream_file_round_trips_into_valid_api_claims(split_claims, tmp_path):
    from claimsense.api.schemas import ClaimIn

    rows = load_stream(write_stream(split_claims, tmp_path), limit=None)
    assert len(rows) == (split_claims["split"] == "test").sum()
    for r in rows:
        c = ClaimIn(**r)
        assert c.ClaimID == r["ClaimID"]
        assert c.ClmProcedureCode_1 is None or not c.ClmProcedureCode_1.endswith(".0")


def test_simulate_stream_matches_engine(split_claims, tiny_models_dir):
    sim = simulate_stream(split_claims, None, tiny_models_dir, anomaly_threshold=1e9, provider_threshold=0.0,
                          min_provider_claims=3, batch_size=16)
    assert len(sim) == len(split_claims)
    assert list(sim.columns) == ["ClaimID", "Provider", "label", "amount", "action", "anomaly", "provider_risk",
                                 "provider_claims", "rules"]
    assert sim["ClaimID"].tolist() == split_claims["ClaimID"].tolist()
    # with only rules flagging, an action is non-PAY exactly when a rule fired
    assert ((sim["action"] != "PAY") == (sim["rules"] != "")).all()
    assert ((sim["action"] == "HOLD_FOR_REVIEW") == ((sim["rules"] != "") & (sim["provider_claims"] >= 3))).all()
    counts = sim.groupby("Provider")["provider_claims"].max()
    assert (counts == split_claims["Provider"].value_counts().reindex(counts.index)).all()
    # tuning on it works end to end
    t = tune_provider_threshold(sim, 1e9, 3, hold_cap=1.0, min_precision=0.0)
    assert 0.30 <= t <= 0.95


def test_training_module_imports_xgboost_first():
    src = open(training.__file__).read()
    assert src.index("import xgboost") < src.index("import torch")
