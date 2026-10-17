"""Optional: the real IBM AMLworld HI-Small files. Skipped when shared/data/aml is missing. Run: pytest -m slow"""
import numpy as np
import pandas as pd
import pytest

from scamshield import data as D
from scamshield.features import FEATURES, FeatureContext, FeatureEngine, rule_hits
from scamshield.training import rule_matrix

from conftest import REAL_AML_DIR

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not any(REAL_AML_DIR.glob("*_Trans.csv")), reason="AMLworld data not present"),
]
N = 200_000


@pytest.fixture(scope="module")
def real():
    tx = D.load_transactions(str(REAL_AML_DIR), nrows=N)
    return tx


def test_real_loader(real):
    assert len(real) == N and real["ts"].is_monotonic_increasing
    assert real["from_bank"].str.startswith("0").any()                  # leading zeros kept
    # 9 hex digits; almost all start with 8, a handful of very busy accounts start with 1 (see report)
    assert real["from_acct"].str.fullmatch(r"[18][0-9A-F]{8}").all()
    assert set(real["payment_format"]) <= set(D.FORMATS)
    assert set(D.time_split(real["ts"])) <= {"train", "val", "test"}


def test_real_patterns_parse_and_match(real):
    pat = D.laundering_patterns(str(REAL_AML_DIR))
    assert len(pat) > 1000 and {"FAN-IN", "FAN-OUT", "CYCLE"} <= set(pat["pattern"])
    pat["ts"] = pd.to_datetime(pat["ts"], format="%Y/%m/%d %H:%M")
    m = pat.merge(real, on=["ts", "sender", "receiver", "amount_paid"])
    assert len(m) > 0 and (m["label"] == 1).mean() > 0.99


def test_real_context_fit_on_a_prefix_can_lack_currencies(real):
    with pytest.raises(ValueError, match="no conversion path"):
        FeatureContext.fit(real.head(50_000))


def test_real_features_parity_and_rules(real):
    saved = REAL_AML_DIR.parents[1] / "models_aml" / "feature_context.json"
    ctx = FeatureContext.from_json(saved.read_text()) if saved.exists() else FeatureContext(
        usd_per_unit={c: 1.0 for c in real["currency_paid"].unique()})
    sub = real.head(20_000)
    X = FeatureEngine(ctx).run(sub)
    assert np.isfinite(X).all()
    fe = FeatureEngine(ctx)
    obs = np.array([fe.observe_txn(r) for r in sub.head(5_000).to_dict("records")], dtype=np.float32)
    np.testing.assert_array_equal(obs, X[:5_000])
    np.testing.assert_array_equal(rule_matrix(X, ctx), [bool(rule_hits(x, ctx)) for x in X])
    assert X.shape[1] == len(FEATURES)


def test_shipped_risk_model_is_trimmed_to_best_iteration():
    import json

    import xgboost as xgb

    models = REAL_AML_DIR.parents[1] / "models_aml"
    if not (models / "risk_xgb.json").exists():
        pytest.skip("trained models not present")
    meta = json.loads((models / "risk_meta.json").read_text())
    b = xgb.Booster()
    b.load_model(str(models / "risk_xgb.json"))
    assert b.num_boosted_rounds() == meta["trees"] == meta["best_iteration"] + 1
    assert meta["features"] == FEATURES and b.feature_names == FEATURES
