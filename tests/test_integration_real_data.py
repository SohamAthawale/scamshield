"""Optional: parity on the real Kaggle CSVs. Skipped when shared/data/raw is missing. Run: pytest -m slow"""
import math

import numpy as np
import pytest

from claimsense import data
from claimsense.features import (
    CLAIM_FEATURES, PROVIDER_FEATURES, DuplicateIndex, FeatureContext, ProviderState, claim_features,
    claim_features_batch, duplicate_key, hard_errors, hard_errors_batch, provider_features_batch,
    provider_features_prefixes,
)

from conftest import RAW_DIR, attending_key

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not any(RAW_DIR.glob("Train_Inpatientdata*.csv")), reason="real dataset not present"),
]


def test_real_data_streaming_equals_batch():
    claims, bene, labels = data.load_all(str(RAW_DIR))
    assert len(bene) > 0 and set(labels["label"]) == {0, 1}
    split = data.split_providers(labels)
    train_prv = set(split.loc[split["split"] == "train", "Provider"])
    ctx = FeatureContext.fit(claims[claims["Provider"].isin(train_prv)])

    test_prv = sorted(split.loc[split["split"] == "test", "Provider"])[:40]
    df = claims[claims["Provider"].isin(test_prv)].reset_index(drop=True)
    assert len(df) > 500

    feats = claim_features_batch(df, ctx)
    assert np.isfinite(feats.to_numpy()).all()
    rows = df.to_dict("records")
    stream = np.array([[claim_features(r, r, ctx)[k] for k in CLAIM_FEATURES] for r in rows])
    np.testing.assert_allclose(stream, feats.to_numpy(), atol=1e-9)

    rules = hard_errors_batch(df)
    idx = DuplicateIndex()
    for i, r in enumerate(rows):
        expected = [k for k in rules.columns[:-1] if rules.iloc[i][k]]
        assert hard_errors(r, r) == expected
        if not math.isnan(r["InscClaimAmtReimbursed"]):
            assert idx.check_and_add(duplicate_key(r, r["BeneID"], r["Provider"])) == bool(rules.iloc[i]["duplicate"])

    batch = provider_features_batch(df, feats, ctx.rare_pair_threshold)
    states = {}
    for r, f in zip(rows, feats.to_dict("records")):
        states.setdefault(r["Provider"], ProviderState()).update(f, r["BeneID"], attending_key(r),
                                                                  ctx.rare_pair_threshold)
    for prv, st in states.items():
        np.testing.assert_allclose([st.features()[k] for k in PROVIDER_FEATURES], batch.loc[prv].to_numpy(),
                                   atol=1e-9)

    pref = provider_features_prefixes(df, feats, ctx.rare_pair_threshold)
    assert len(pref) == 4 * df["Provider"].nunique()
