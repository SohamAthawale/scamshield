"""training.py pieces on the synthetic AMLworld data (no full training run)."""
import numpy as np
import pandas as pd
import pytest

from scamshield.features import FEATURES, FeatureContext, FeatureEngine, rule_hits
from scamshield.fusion import ALLOW, HOLD, WARN, Policy, decide
from scamshield.training import (
    AE_IDX, HISTORY_COLUMNS, decisions, ranking_metrics, rule_matrix, scheme_hits, summarise, train_risk_model,
    tune_hold_threshold, tune_value_hold, write_history, write_stream,
)

from conftest import policy_for, scorers


# ------------------------------------------------------------------------------ metrics
def test_ranking_metrics_toy():
    m = ranking_metrics(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1]))
    assert m["n"] == 4 and m["positives"] == 2 and m["base_rate"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["pr_auc"] == pytest.approx((1 + 2 / 3) / 2)
    # best F1 is at the top 3: precision 2/3, recall 1 -> F1 0.8
    assert m["best_f1"] == pytest.approx(0.8)
    assert m["best_f1_precision"] == pytest.approx(2 / 3) and m["best_f1_recall"] == 1.0
    assert m["best_f1_threshold"] == 0.7


def test_ranking_metrics_perfect():
    y = np.array([0] * 90 + [1] * 10)
    m = ranking_metrics(y, np.arange(100) / 100)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == pytest.approx(1.0) and m["best_f1"] == 1.0
    assert m["best_f1_threshold"] == 0.9


# ------------------------------------------------------------------------------ fusion + rules parity
def test_rule_matrix_equals_rule_hits_on_engine_features(aml):
    vec = rule_matrix(aml.X, aml.ctx)
    one = np.array([bool(rule_hits(x, aml.ctx)) for x in aml.X])
    np.testing.assert_array_equal(vec, one)
    assert vec.sum() >= 2


def test_rule_matrix_equals_rule_hits_on_crafted_edges():
    ctx = FeatureContext(usd_per_unit={"US Dollar": 1.0})
    rows = []
    for new in (0.0, 1.0):
        for n_out in (4, 5, 6, 50):
            for ratio in (9.99, 10.0, 10.01, 1000.0):
                f = dict.fromkeys(FEATURES, 0.0)
                f.update(s_new_payee=new, s_log_n_out=np.log1p(n_out), s_amount_vs_usual=np.log(ratio))
                rows.append([f[k] for k in FEATURES])
    for dtype in (np.float64, np.float32):
        X = np.array(rows, dtype=dtype)
        np.testing.assert_array_equal(rule_matrix(X, ctx), [bool(rule_hits(x, ctx)) for x in X])


def test_decisions_equal_decide_on_model_outputs(aml):
    anomaly, risk = scorers(aml.models_dir)
    pol = policy_for(aml.models_dir)
    err, _, _ = anomaly.score(aml.X[:, AE_IDX])
    r = risk.predict(aml.X)
    rules = rule_matrix(aml.X, aml.ctx)
    vec = decisions(r, err, rules, pol)
    one = [decide(float(e), float(p), rule_hits(x, aml.ctx), pol).action for e, p, x in zip(err, r, aml.X)]
    assert list(vec) == one
    assert {ALLOW, WARN, HOLD} <= set(vec)


# ------------------------------------------------------------------------------ hold threshold
def test_tune_hold_threshold_lowest_meeting_precision_and_cap():
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0] * 10)
    risk = np.linspace(1, 0, 100)              # distinct scores, best first
    t = tune_hold_threshold(y, risk, min_precision=0.6, cap=0.05)
    held = risk >= t
    assert held.sum() == 5 and y[held].mean() >= 0.6 and held.mean() <= 0.05
    t2 = tune_hold_threshold(y, risk, min_precision=0.6, cap=0.5)
    assert t2 < t                              # a bigger cap -> lower threshold, more caught
    held2 = risk >= t2
    assert y[held2].mean() >= 0.6 and held2.mean() <= 0.5


def test_tune_hold_threshold_impossible_holds_nothing():
    y = np.array([0, 0, 1])
    risk = np.array([0.9, 0.8, 0.1])
    t = tune_hold_threshold(y, risk, min_precision=0.9, cap=1.0)
    assert t > risk.max() and not (risk >= t).any()


def test_tune_hold_threshold_respects_cap_and_precision_with_tied_scores():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    risk = np.array([0.9, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    t = tune_hold_threshold(y, risk, min_precision=0.6, cap=0.2)
    held = risk >= t
    assert held.mean() <= 0.2 and (not held.any() or y[held].mean() >= 0.6)


# ------------------------------------------------------------------------------ summaries
def test_summarise():
    y = np.array([1, 1, 0, 0, 1, 0])
    act = np.array([HOLD, WARN, WARN, ALLOW, ALLOW, HOLD])
    usd = np.array([100.0, 200.0, 5.0, 5.0, 300.0, 50.0])
    s = summarise(y, act, usd)
    assert s["payments"] == 6 and s["mix"] == {ALLOW: round(2 / 6, 5), WARN: round(2 / 6, 5), HOLD: round(2 / 6, 5)}
    assert s["hold_precision"] == 0.5 and s["warn_precision"] == 0.5
    assert s["laundering_recall_hold"] == pytest.approx(1 / 3)
    assert s["laundering_recall_warn_or_hold"] == pytest.approx(2 / 3)
    assert s["laundered_usd_total"] == 600.0 and s["laundered_usd_held"] == 100.0
    assert s["base_rate"] == 0.5 and "schemes_disrupted" not in s
    none = summarise(y, np.array([ALLOW] * 6), usd)
    assert none["hold_precision"] is None and none["warn_precision"] is None


def test_scheme_hits_and_summary(aml):
    train = aml.split == "train"
    n = int(train.sum())
    all_held = scheme_hits(aml.tx, train, np.array([HOLD] * n), aml.raw_dir)
    none_held = scheme_hits(aml.tx, train, np.array([ALLOW] * n), aml.raw_dir)
    in_train = [k for k, att in aml.attempts if att[0].ts < pd.Timestamp("2022-09-05")]
    assert sorted(all_held["pattern"]) == sorted(in_train)
    assert all_held["hit"].all() and not none_held["hit"].any()
    s = summarise(aml.y[train], np.array([HOLD] * n), np.ones(n), all_held)
    assert s["schemes_disrupted_total"] == f"{len(in_train)}/{len(in_train)}"
    # holding a single payment of an attempt breaks that chain
    act = np.array([ALLOW] * n)
    first_fan_in = aml.attempts[0][1][0]
    idx = np.flatnonzero((aml.tx["ts"][train] == first_fan_in.ts).values &
                         (aml.tx["amount_paid"][train] == first_fan_in.amount_paid).values)
    act[idx] = HOLD
    hits = scheme_hits(aml.tx, train, act, aml.raw_dir).set_index("pattern")["hit"]
    assert hits["FAN-IN"] and not hits["CYCLE"]


# ------------------------------------------------------------------------------ files for the API / simulator
def test_write_history_excludes_live_days_and_labels(aml, tmp_path):
    p = write_history(aml.tx, aml.split, tmp_path)
    h = pd.read_csv(p, dtype=str)
    assert list(h.columns) == HISTORY_COLUMNS and "label" not in h.columns
    assert len(h) == int((aml.split != "test").sum())
    assert h["ts"].str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}").all()
    assert h["ts"].tolist() == sorted(h["ts"].tolist()) and h["ts"].max() < "2022-09-07"
    assert (h["from_bank"].str.len() >= 3).all()                # leading zeros kept
    # same order as training: replaying it gives training's features for those rows
    hist = pd.read_csv(p, dtype={"from_bank": str, "to_bank": str})
    hist["ts"] = pd.to_datetime(hist["ts"])
    np.testing.assert_array_equal(FeatureEngine(aml.ctx).run(hist), aml.X[aml.split != "test"])


def test_write_stream_live_days_without_labels(aml, tmp_path):
    p = write_stream(aml.tx, aml.split, tmp_path)
    s = pd.read_csv(p, dtype=str, keep_default_na=False)
    assert "label" not in s.columns
    assert {"from_acct", "to_acct", "amount_received", "currency_received", "sender", "receiver"} <= set(s.columns)
    assert len(s) == int((aml.split == "test").sum()) and s["ts"].min() >= "2022-09-07"
    assert s["ts"].tolist() == sorted(s["ts"].tolist())


# ------------------------------------------------------------------------------ risk model
@pytest.fixture(scope="module")
def trained(aml):
    return train_risk_model(aml.X, aml.y, aml.split, neg_sample=0.5, seed=1)


def test_train_risk_model_trims_to_best_iteration(trained):
    booster, meta = trained
    assert meta["trees"] == meta["best_iteration"] + 1
    assert booster.num_boosted_rounds() == meta["trees"]
    assert booster.feature_names == FEATURES and meta["features"] == FEATURES
    assert set(meta["importance_gain"]) == set(FEATURES) and meta["negative_sample"] == 0.5


def test_trained_booster_round_trips_through_the_risk_scorer(trained, aml, tmp_path):
    import json

    from scamshield.inference.risk import load_risk_scorer

    booster, meta = trained
    booster.save_model(str(tmp_path / "risk_xgb.json"))
    (tmp_path / "risk_meta.json").write_text(json.dumps(meta))
    s = load_risk_scorer(tmp_path)
    assert s.booster.num_boosted_rounds() == meta["trees"]
    np.testing.assert_allclose(s.predict(aml.X), booster.inplace_predict(aml.X), rtol=0, atol=1e-7)


def test_train_risk_model_is_deterministic(trained, aml):
    b2, m2 = train_risk_model(aml.X, aml.y, aml.split, neg_sample=0.5, seed=1)
    assert m2["trees"] == trained[1]["trees"]
    np.testing.assert_array_equal(b2.inplace_predict(aml.X), trained[0].inplace_predict(aml.X))


# ------------------------------------------------------------------------------ value-aware hold
def _band():
    """Hold band (risk >= 0.5) of 10 payments, best first. Laundering is either very risky or very big."""
    risk = np.array([0.99, 0.95, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.4, 0.1])
    y = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    usd = np.array([1e3, 2e3, 500.0, 300.0, 3e6, 40.0, 60.0, 1e6, 10.0, 5e5, 9e9, 9e9])
    return y, risk, usd


def test_tune_value_hold_picks_the_best_feasible_choice():
    y, risk, usd = _band()
    t = tune_value_hold(y, risk, usd, hold_threshold=0.5, min_precision=0.8, keep_usd=0.95)
    assert t["val_base"] == {"holds": 10, "laundering_held": 5, "laundered_usd_held": 1e3 + 2e3 + 500 + 3e6 + 1e6}
    pol = Policy(9.0, 0.5, 0.5, t["hold_strict_threshold"], t["hold_min_usd"])
    held = decisions(risk, np.zeros(len(y)), np.zeros(len(y), bool), pol, usd) == HOLD
    # strict holds the top 3 (all laundering); the band holds only the two $1M+ payments
    assert t["hold_strict_threshold"] == 0.9 and t["hold_min_usd"] == 1e6
    assert list(np.flatnonzero(held)) == [0, 1, 2, 4, 7]
    assert y[held].mean() == 1.0 and t["val_chosen"]["hold_precision"] == 1.0
    assert t["val_chosen"]["laundering_held"] == 5 and t["val_chosen"]["holds"] == 5
    assert t["val_chosen"]["usd_kept_vs_base"] == pytest.approx(1.0)


def test_tune_value_hold_constraints_hold_on_the_tuning_data():
    rng = np.random.default_rng(0)
    n = 5000
    risk = rng.random(n)
    y = (rng.random(n) < risk ** 4).astype(int)
    usd = 10 ** rng.uniform(1, 7, n) * (1 + 3 * y)
    base_thr = 0.6
    t = tune_value_hold(y, risk, usd, base_thr, min_precision=0.7, keep_usd=0.9)
    assert t["hold_strict_threshold"] is not None and t["feasible_choices"] > 0
    pol = Policy(9.0, base_thr, base_thr, t["hold_strict_threshold"], t["hold_min_usd"])
    held = decisions(risk, np.zeros(n), np.zeros(n, bool), pol, usd) == HOLD
    base = risk >= base_thr
    assert not (held & ~base).any()                                    # never holds more than today
    assert y[held].mean() >= 0.7
    assert usd[held & (y == 1)].sum() >= 0.9 * usd[base & (y == 1)].sum()
    assert held.sum() == t["val_chosen"]["holds"] and y[held].sum() == t["val_chosen"]["laundering_held"]


def test_tune_value_hold_infeasible_keeps_the_plain_policy():
    y, risk, usd = _band()
    t = tune_value_hold(y, risk, usd, hold_threshold=0.5, min_precision=1.01)
    assert t["hold_strict_threshold"] is None and t["hold_min_usd"] is None and t["val_chosen"] is None
    t = tune_value_hold(y, risk, usd, hold_threshold=2.0)              # nothing in the band
    assert t["hold_strict_threshold"] is None and t["val_base"]["holds"] == 0


def test_tune_value_hold_holding_the_whole_band_is_todays_policy():
    y = np.array([1, 1, 1, 0])
    risk = np.array([0.9, 0.8, 0.7, 0.1])
    t = tune_value_hold(y, risk, np.ones(4), hold_threshold=0.6, min_precision=0.8, keep_usd=1.0)
    assert t["hold_strict_threshold"] == 0.6                           # strict == hold: plain policy


def test_tune_value_hold_respects_tied_scores():
    y = np.array([1, 1, 0, 0])
    risk = np.array([0.9, 0.8, 0.8, 0.8])
    usd = np.array([1.0, 1.0, 1.0, 1.0])
    t = tune_value_hold(y, risk, usd, hold_threshold=0.8, min_precision=0.9, keep_usd=0.0)
    # cannot hold only the first 0.8: strict = 0.9 holds payment 0, small amounts in the band are not held
    pol = Policy(9.0, 0.8, 0.8, t["hold_strict_threshold"], t["hold_min_usd"])
    held = decisions(risk, np.zeros(4), np.zeros(4, bool), pol, usd) == HOLD
    assert list(held) == [True, False, False, False]


def test_summarise_counts():
    y = np.array([1, 1, 0, 0, 1, 0])
    act = np.array([HOLD, WARN, WARN, ALLOW, ALLOW, HOLD])
    s = summarise(y, act, np.ones(6))
    assert s["holds"] == 2 and s["laundering_held"] == 1 and s["laundering_warned_or_held"] == 2
