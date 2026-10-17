"""features.py: one streaming engine for training and serving. Parity, causality, semantics, context, rules."""
import math

import numpy as np
import pandas as pd
import pytest

from scamshield.data import FORMATS
from scamshield.features import (
    ANOMALY_FEATURES, FEATURE_LABELS, FEATURES, IDLE_CAP_H, INTRINSIC, RECEIVER, RULES, SENDER, TAU_MIN,
    FeatureContext, FeatureEngine, rule_hits,
)

CTX = FeatureContext(usd_per_unit={"US Dollar": 1.0, "Euro": 1.1, "Yen": 0.0075})
CURRENCIES = list(CTX.usd_per_unit)


def random_payments(seed: int, n: int = 300, n_accounts: int = 12) -> pd.DataFrame:
    """Hypothesis-style random stream: few accounts (so histories interact), same-minute bursts, self-transfers,
    all formats, all currencies, zero amounts, cross-bank and same-bank payments."""
    rng = np.random.default_rng(seed)
    banks = ["010", "001", "0119"]
    accts = [(str(rng.choice(banks)), f"8{i:08X}") for i in range(n_accounts)]
    t = pd.Timestamp("2022-09-01") + pd.to_timedelta(np.sort(rng.integers(0, 3 * 24 * 60, n)), unit="m")
    rows = []
    for i in range(n):
        s = accts[int(rng.integers(n_accounts))]
        r = s if rng.random() < 0.05 else accts[int(rng.integers(n_accounts))]
        cp = str(rng.choice(CURRENCIES))
        cr = cp if rng.random() < 0.8 else str(rng.choice(CURRENCIES))
        amt = 0.0 if rng.random() < 0.03 else float(np.round(np.exp(rng.uniform(-2, 12)), 2))
        rows.append({"ts": t[i], "from_bank": s[0], "from_acct": s[1], "to_bank": r[0], "to_acct": r[1],
                     "amount_received": amt, "currency_received": cr, "amount_paid": amt, "currency_paid": cp,
                     "payment_format": str(rng.choice(FORMATS)), "label": int(rng.random() < 0.1)})
    df = pd.DataFrame(rows)
    df["sender"] = df["from_bank"] + ":" + df["from_acct"]
    df["receiver"] = df["to_bank"] + ":" + df["to_acct"]
    return df


def observe_rows(df: pd.DataFrame, ctx: FeatureContext = CTX) -> np.ndarray:
    fe = FeatureEngine(ctx)
    return np.array([fe.observe_txn(r) for r in df.to_dict("records")], dtype=np.float32)


def explicit_rows(df: pd.DataFrame, ctx: FeatureContext = CTX) -> np.ndarray:
    fe = FeatureEngine(ctx)
    out = []
    for r in df.to_dict("records"):
        ts = pd.Timestamp(r["ts"])
        out.append(fe.observe(ts.value / 6e10, r["sender"], r["receiver"], ctx.to_usd(r["amount_paid"], r["currency_paid"]),
                              r["payment_format"], r["from_bank"] != r["to_bank"],
                              r["currency_paid"] != r["currency_received"], ts.hour))
    return np.array(out, dtype=np.float32)


# ------------------------------------------------------------------------------ feature lists
def test_feature_lists():
    assert FEATURES == INTRINSIC + SENDER + RECEIVER
    assert ANOMALY_FEATURES == INTRINSIC + SENDER          # the customer model never sees receiver signals
    assert not set(ANOMALY_FEATURES) & set(RECEIVER)
    assert len(set(FEATURES)) == len(FEATURES)
    assert set(FEATURE_LABELS) == set(FEATURES)
    assert [f for f in FEATURES if f.startswith("fmt_")] == [f"fmt_{x.replace(' ', '')}" for x in FORMATS]


# ------------------------------------------------------------------------------ parity
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_run_equals_observe_row_by_row_random(seed):
    df = random_payments(seed)
    X = FeatureEngine(CTX).run(df)
    assert X.shape == (len(df), len(FEATURES)) and X.dtype == np.float32
    np.testing.assert_array_equal(X, observe_rows(df))
    np.testing.assert_array_equal(X, explicit_rows(df))
    assert np.isfinite(X).all()


def test_run_equals_observe_on_synthetic_aml(aml):
    np.testing.assert_array_equal(aml.X, observe_rows(aml.tx, aml.ctx))


def test_observe_txn_accepts_iso_strings(aml):
    fe1, fe2 = FeatureEngine(aml.ctx), FeatureEngine(aml.ctx)
    for r in aml.tx.head(80).to_dict("records"):
        a = fe1.observe_txn(r)
        b = fe2.observe_txn({**r, "ts": r["ts"].strftime("%Y-%m-%dT%H:%M:%S")})
        assert a == b


def test_observe_returns_python_floats_in_feature_order():
    f = FeatureEngine(CTX).observe(0.0, "A", "B", 100.0, "ACH", True, False, 13)
    assert len(f) == len(FEATURES) and all(isinstance(v, float) for v in f)


# ------------------------------------------------------------------------------ causality
@pytest.mark.parametrize("k", [1, 50, 151, 299])
def test_features_never_depend_on_later_payments(k):
    df = random_payments(7)
    full = FeatureEngine(CTX).run(df)
    np.testing.assert_array_equal(FeatureEngine(CTX).run(df.iloc[:k]), full[:k])
    # change everything about the future: earlier features must not move
    fut = df.copy()
    fut.loc[k:, "amount_paid"] *= 1000
    fut.loc[k:, "payment_format"] = "Bitcoin"
    fut.loc[k:, "receiver"] = fut.loc[k:, "sender"]
    np.testing.assert_array_equal(FeatureEngine(CTX).run(fut)[:k], full[:k])


def test_features_never_depend_on_labels(aml):
    flipped = aml.tx.assign(label=1 - aml.tx["label"])
    np.testing.assert_array_equal(FeatureEngine(aml.ctx).run(flipped), aml.X)
    np.testing.assert_array_equal(FeatureEngine(aml.ctx).run(aml.tx.drop(columns=["label"])), aml.X)


def test_payment_does_not_see_itself():
    """A payment's own amount/receiver are never part of its 'history' features."""
    fe = FeatureEngine(CTX)
    f = dict(zip(FEATURES, fe.observe(0.0, "A", "B", 1e6, "Wire", True, False, 0)))
    assert f["s_log_n_out"] == 0 and f["r_log_n_in"] == 0 and f["s_amount_vs_usual"] == 0
    assert f["s_is_new"] == 1 and f["r_is_new"] == 1 and f["r_log_senders"] == 0


def test_appending_future_rows_to_aml_keeps_earlier_features(aml):
    extra = aml.tx.tail(20).copy()
    extra["ts"] = extra["ts"] + pd.Timedelta(days=1)
    longer = pd.concat([aml.tx, extra], ignore_index=True)
    np.testing.assert_array_equal(FeatureEngine(aml.ctx).run(longer)[: len(aml.tx)], aml.X)


# ------------------------------------------------------------------------------ semantics
def feats(fe, t, s, r, usd, fmt="ACH", cross=True, mism=False, hour=12):
    return dict(zip(FEATURES, fe.observe(t, s, r, usd, fmt, cross, mism, hour)))


def test_first_payment_of_new_accounts():
    f = feats(FeatureEngine(CTX), 0.0, "A", "B", 100.0, "Wire", True, True, 6)
    assert f["log_usd"] == pytest.approx(math.log1p(100))
    assert f["fmt_Wire"] == 1 and sum(f[k] for k in FEATURES if k.startswith("fmt_")) == 1
    assert f["cross_bank"] == 1 and f["currency_mismatch"] == 1 and f["self_transfer"] == 0
    assert f["hour_sin"] == pytest.approx(1.0) and f["hour_cos"] == pytest.approx(0.0, abs=1e-12)
    assert f["s_new_payee"] == 1 and f["s_is_new"] == 1 and f["r_is_new"] == 1
    assert f["s_log_idle_h"] == pytest.approx(math.log1p(IDLE_CAP_H))
    assert f["r_log_idle_h"] == pytest.approx(math.log1p(IDLE_CAP_H))
    assert f["s_recent_out"] == 0 and f["r_recent_in"] == 0 and f["r_fan_in"] == 0


def test_repeat_payment_and_usual_amount():
    fe = FeatureEngine(CTX)
    feats(fe, 0.0, "A", "B", 100.0)
    f = feats(fe, 60.0, "A", "B", 1000.0)
    assert f["s_new_payee"] == 0 and f["s_is_new"] == 0
    assert f["s_log_n_out"] == pytest.approx(math.log1p(1)) and f["s_log_payees"] == pytest.approx(math.log1p(1))
    assert f["s_amount_vs_usual"] == pytest.approx(math.log1p(1000) - math.log1p(100))
    assert f["s_log_idle_h"] == pytest.approx(math.log1p(1.0))
    assert f["s_recent_out"] == pytest.approx(math.exp(-60 / TAU_MIN))
    assert f["r_log_n_in"] == pytest.approx(math.log1p(1)) and f["r_log_senders"] == pytest.approx(math.log1p(1))
    assert f["r_fan_in"] == 1.0 and f["r_share_ach_in"] == 1.0
    assert f["r_recent_in"] == pytest.approx(math.exp(-60 / TAU_MIN))


def test_usual_is_mean_of_log_amounts():
    fe = FeatureEngine(CTX)
    for i, a in enumerate([10.0, 100.0, 1000.0]):
        feats(fe, float(i), "A", f"P{i}", a)
    f = feats(fe, 10.0, "A", "P9", 50.0)
    usual = np.mean([math.log1p(10), math.log1p(100), math.log1p(1000)])
    assert f["s_amount_vs_usual"] == pytest.approx(math.log1p(50) - usual)
    assert f["s_log_payees"] == pytest.approx(math.log1p(3))


def test_receiver_fan_in_and_pass_through():
    fe = FeatureEngine(CTX)
    feats(fe, 0.0, "S1", "M", 100.0, "ACH")
    feats(fe, 1.0, "S2", "M", 100.0, "Wire")
    feats(fe, 2.0, "S1", "M", 100.0, "ACH")
    f = feats(fe, 3.0, "S3", "M", 100.0)
    assert f["r_log_n_in"] == pytest.approx(math.log1p(3)) and f["r_log_senders"] == pytest.approx(math.log1p(2))
    assert f["r_fan_in"] == pytest.approx(2 / 3) and f["r_share_ach_in"] == pytest.approx(2 / 3)
    out = feats(fe, 4.0, "M", "X", 380.0)                      # the mule sends on what it collected
    assert out["s_log_n_in"] == pytest.approx(math.log1p(4))
    assert out["s_pass_through"] == pytest.approx(math.log1p(0) - math.log1p(400))
    nxt = feats(fe, 5.0, "M", "Y", 10.0)
    assert nxt["s_pass_through"] == pytest.approx(math.log1p(380) - math.log1p(400))
    assert feats(fe, 6.0, "Z", "M", 1.0)["r_pass_through"] == pytest.approx(math.log1p(390) - math.log1p(400))


def test_self_transfer_and_idle_cap():
    fe = FeatureEngine(CTX)
    assert feats(fe, 0.0, "A", "A", 5.0, "Reinvestment", False)["self_transfer"] == 1
    f = feats(fe, 60.0 * 24 * 365, "A", "B", 5.0)              # a year later: capped at 30 days
    assert f["s_log_idle_h"] == pytest.approx(math.log1p(IDLE_CAP_H))
    assert f["s_recent_out"] == pytest.approx(0.0, abs=1e-30)


def test_decayed_counter_accumulates():
    fe = FeatureEngine(CTX)
    for t in (0.0, 0.0, 0.0):
        feats(fe, t, "A", "B", 1.0)
    f = feats(fe, TAU_MIN, "A", "B", 1.0)
    assert f["s_recent_out"] == pytest.approx(3 * math.exp(-1))


def test_unknown_format_and_negative_amount():
    f = feats(FeatureEngine(CTX), 0.0, "A", "B", -50.0, "Telepathy")
    assert all(f[k] == 0 for k in FEATURES if k.startswith("fmt_"))
    assert f["log_usd"] == 0.0


@pytest.mark.parametrize("hour", [0, 6, 12, 18, 23])
def test_hour_encoding_is_on_the_unit_circle(hour):
    f = feats(FeatureEngine(CTX), 0.0, "A", "B", 1.0, hour=hour)
    assert f["hour_sin"] ** 2 + f["hour_cos"] ** 2 == pytest.approx(1.0)
    assert f["hour_sin"] == pytest.approx(math.sin(2 * math.pi * hour / 24))


def test_run_uses_bank_columns_for_cross_bank_and_converts_currency():
    df = random_payments(5, n=40)
    X = FeatureEngine(CTX).run(df)
    i = FEATURES.index("cross_bank")
    np.testing.assert_array_equal(X[:, i], (df["from_bank"] != df["to_bank"]).astype(np.float32))
    usd = df["amount_paid"] * df["currency_paid"].map(CTX.usd_per_unit)
    np.testing.assert_allclose(X[:, FEATURES.index("log_usd")], np.log1p(usd), rtol=1e-6)


# ------------------------------------------------------------------------------ FeatureContext
def test_context_fit_walks_the_conversion_graph(aml):
    rates = aml.ctx.usd_per_unit
    assert rates["US Dollar"] == 1.0
    assert rates["Euro"] == pytest.approx(1.1, rel=0.01)
    assert rates["Yen"] == pytest.approx(0.0075, rel=0.02)      # only Yen<->Euro conversions exist


def conv_df(rows):
    return pd.DataFrame(rows, columns=["currency_paid", "currency_received", "amount_paid", "amount_received"])


def test_context_fit_uses_median_and_both_directions():
    df = conv_df([("Euro", "US Dollar", 100, 110), ("Euro", "US Dollar", 100, 120), ("Euro", "US Dollar", 100, 100),
                  ("US Dollar", "Pound", 100, 80), ("US Dollar", "US Dollar", 5, 5)])
    ctx = FeatureContext.fit(df)
    assert ctx.usd_per_unit["Euro"] == pytest.approx(1.1)
    assert ctx.usd_per_unit["Pound"] == pytest.approx(1 / 0.8)  # 1 USD = 0.8 Pound
    assert ctx.to_usd(10, "Pound") == pytest.approx(12.5)


def test_context_fit_ignores_zero_amount_conversions():
    df = conv_df([("Euro", "US Dollar", 0, 110), ("Euro", "US Dollar", 100, 0), ("Euro", "US Dollar", 100, 110)])
    assert FeatureContext.fit(df).usd_per_unit["Euro"] == pytest.approx(1.1)


def test_context_fit_raises_without_a_path_to_usd():
    df = conv_df([("Yen", "Rupee", 100, 60), ("US Dollar", "US Dollar", 1, 1)])
    with pytest.raises(ValueError, match="no conversion path"):
        FeatureContext.fit(df)


def test_context_to_usd_and_json_round_trip(aml):
    assert aml.ctx.to_usd(10, "US Dollar") == 10.0
    assert aml.ctx.to_usd(10, "Unknown Coin") == 10.0           # unknown currency treated as USD (see report)
    back = FeatureContext.from_json(aml.ctx.to_json())
    assert back == aml.ctx and back.rule_big_new_payee_ratio == 10.0
    np.testing.assert_array_equal(FeatureEngine(back).run(aml.tx), aml.X)


# ------------------------------------------------------------------------------ rules
def rule_feats(new_payee=1.0, n_out=5, ratio=10.0):
    f = dict.fromkeys(FEATURES, 0.0)
    f.update(s_new_payee=new_payee, s_log_n_out=math.log1p(n_out), s_amount_vs_usual=math.log(ratio))
    return f


@pytest.mark.parametrize("kw,fires", [
    ({}, True), ({"ratio": 50}, True), ({"n_out": 100}, True),
    ({"new_payee": 0.0}, False), ({"n_out": 4}, False), ({"ratio": 9.9}, False), ({"ratio": 0.1}, False),
])
def test_big_new_payee_rule(kw, fires):
    assert (rule_hits(rule_feats(**kw), CTX) == ["big_new_payee"]) is fires
    assert set(rule_hits(rule_feats(**kw), CTX)) <= set(RULES)


def test_rule_accepts_vectors_and_float32():
    f = rule_feats()
    v = np.array([f[k] for k in FEATURES])
    assert rule_hits(v, CTX) == rule_hits(v.astype(np.float32), CTX) == rule_hits(f, CTX) == ["big_new_payee"]


def test_rule_ratio_comes_from_context():
    strict = FeatureContext(usd_per_unit={"US Dollar": 1.0}, rule_big_new_payee_ratio=100.0)
    assert rule_hits(rule_feats(ratio=50), strict) == []
    assert rule_hits(rule_feats(ratio=100.0001), strict) == ["big_new_payee"]


def test_rule_fires_on_streamed_payments():
    fe = FeatureEngine(CTX)
    for i in range(5):
        fe.observe(float(i), "A", "B", 100.0, "ACH", True, False, 9)
    big_new = fe.observe(10.0, "A", "NEW", 5000.0, "Wire", True, False, 9)
    assert rule_hits(big_new, CTX) == ["big_new_payee"]
    big_old = fe.observe(11.0, "A", "B", 50000.0, "Wire", True, False, 9)
    assert rule_hits(big_old, CTX) == []                         # not a new payee


def test_rule_needs_five_prior_payments():
    fe = FeatureEngine(CTX)
    for i in range(4):
        fe.observe(float(i), "A", "B", 100.0, "ACH", True, False, 9)
    assert rule_hits(fe.observe(10.0, "A", "NEW", 5000.0, "Wire", True, False, 9), CTX) == []


def test_customers_in_synthetic_data_trip_the_rule(aml):
    from conftest import CUSTOMER_1, CUSTOMER_2

    for cust in (CUSTOMER_1, CUSTOMER_2):
        m = (aml.tx["sender"] == ":".join(cust)).values & (aml.tx["amount_paid"] == 5000.0).values
        assert m.sum() == 1 and aml.split[m][0] == "test"
        assert rule_hits(aml.X[m][0], aml.ctx) == ["big_new_payee"]


def test_slightly_out_of_order_payment_is_clamped():
    """A payment 30 min older than the last one: the gap is treated as 0, never negative or 'un-decayed'."""
    fe = FeatureEngine(CTX)
    fe.observe(600.0, "A", "B", 10.0, "ACH", True, False, 10)
    f = dict(zip(FEATURES, fe.observe(570.0, "A", "C", 10.0, "ACH", True, False, 9)))
    assert f["s_log_idle_h"] == 0.0 and f["s_recent_out"] == pytest.approx(1.0)


def test_out_of_order_payment_does_not_crash():
    fe = FeatureEngine(CTX)
    fe.observe(600.0, "A", "B", 10.0, "ACH", True, False, 10)
    f = dict(zip(FEATURES, fe.observe(400.0, "A", "D", 10.0, "ACH", True, False, 6)))
    assert f["s_log_idle_h"] >= 0
