"""features.py: streaming (one claim) and batch (DataFrame) implementations must agree."""
import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from claimsense.data import CHRONIC_COLS, DIAG_COLS, PROC_COLS
from claimsense.features import (
    CLAIM_FEATURES, FEATURE_LABELS, HARD_RULES, MAX_CLAIM_DAYS, PROVIDER_FEATURES, PROVIDER_LABELS,
    DuplicateIndex, FeatureContext, ProviderState, _code, _date, _present, claim_features,
    claim_features_batch, duplicate_key, hard_errors, hard_errors_batch, provider_features_batch,
    provider_features_prefixes,
)

from conftest import assert_close, attending_key, bene_fields, make_claims

RULES_NO_DUP = ["after_death", "end_before_start", "discharge_before_admission", "missing_attending", "no_coverage"]


def stream_features(df, ctx):
    return pd.DataFrame([claim_features(r, bene_fields(r), ctx) for r in df.to_dict("records")],
                        index=df.index)[CLAIM_FEATURES]


def api_style(claim: dict) -> dict:
    """What a JSON API would hand the streaming path: ISO date strings, None or 'NA' for missing."""
    out = {}
    for k, v in claim.items():
        if isinstance(v, pd.Timestamp):
            out[k] = v.date().isoformat()
        elif v is pd.NaT:
            out[k] = "NA"
        elif isinstance(v, float) and math.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


def base_claim(**over) -> dict:
    c = {
        "ClaimID": "CLM1", "BeneID": "BENE1", "Provider": "PRV1",
        "ClaimStartDt": "2009-04-12", "ClaimEndDt": "2009-04-18",
        "AdmissionDt": "2009-04-12", "DischargeDt": "2009-04-18",
        "InscClaimAmtReimbursed": 26000.0, "DeductibleAmtPaid": 1068.0,
        "AttendingPhysician": "PHY1", "OperatingPhysician": None, "OtherPhysician": None,
        "ClmAdmitDiagnosisCode": "4019", "DiagnosisGroupCode": "201",
        **{c: None for c in DIAG_COLS}, **{c: None for c in PROC_COLS},
        "is_inpatient": 1,
    }
    c["ClmDiagnosisCode_1"] = "4019"
    c.update(over)
    return c


def base_bene(**over) -> dict:
    b = {"DOB": "1943-01-01", "DOD": None, "RenalDiseaseIndicator": 0,
         "NoOfMonths_PartACov": 12, "NoOfMonths_PartBCov": 12, **{c: 0 for c in CHRONIC_COLS}}
    b.update(over)
    return b


def as_frame(claim: dict, bene: dict) -> pd.DataFrame:
    df = pd.DataFrame([{**claim, **bene}])
    for c in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt", "DOB", "DOD"]:
        df[c] = pd.to_datetime(df[c].replace({"NA": None}))
    return df


# ------------------------------------------------------------------------------ helpers
@pytest.mark.parametrize("v,expected", [
    (None, None), (float("nan"), None), (pd.NaT, None), ("", None), ("NA", None), ("nan", None),
    ("NaT", None), ("None", None), ("2009-04-12", dt.date(2009, 4, 12)),
    ("2009-04-12 00:00:00", dt.date(2009, 4, 12)), (pd.Timestamp("2009-04-12"), dt.date(2009, 4, 12)),
    (dt.datetime(2009, 4, 12, 13, 5), dt.date(2009, 4, 12)), (dt.date(2009, 4, 12), dt.date(2009, 4, 12)),
])
def test_date_parsing(v, expected):
    assert _date(v) == expected


@pytest.mark.parametrize("v,expected", [
    (7092.0, "7092"), ("7092", "7092"), ("7092.0", "7092"), (np.float64(7092.0), "7092"), (7092, "7092"),
    ("V5866", "V5866"), ("E8889", "E8889"), (" 4019 ", "4019"), ("0389", "0389"),
    (None, "NA"), (float("nan"), "NA"), ("NA", "NA"), ("", "NA"), ("nan", "NA"),
])
def test_code_normalisation(v, expected):
    assert _code(v) == expected


@pytest.mark.parametrize("v,expected", [
    (None, False), (float("nan"), False), (np.float32("nan"), False), ("", False), ("  ", False),
    ("NA", False), ("na", False), ("NaN", False), ("None", False), ("PHY1", True), (0, True), (0.0, True),
])
def test_present(v, expected):
    assert _present(v) is expected


def test_labels_cover_every_feature():
    assert set(FEATURE_LABELS) == set(CLAIM_FEATURES)
    assert set(PROVIDER_LABELS) == set(PROVIDER_FEATURES)
    assert len(set(CLAIM_FEATURES)) == len(CLAIM_FEATURES)


# ------------------------------------------------------------------------------ FeatureContext
def test_context_fit_counts(train_claims, ctx):
    assert ctx.pair_total == len(train_claims) == ctx.dx_total
    assert sum(ctx.pair_counts.values()) == ctx.pair_total
    assert sum(ctx.dx_counts.values()) == ctx.dx_total
    assert set(ctx.global_amount_median) == {"0", "1"}
    assert all("|" in k for k in ctx.pair_counts)


def test_context_fit_min_count_for_median():
    df = make_claims(seed=3, n=200, messy=False)
    ctx = FeatureContext.fit(df, min_count_for_median=10_000)
    assert ctx.dx_amount_median == {}
    assert ctx.min_count_for_median == 10_000
    ctx5 = FeatureContext.fit(df, min_count_for_median=5)
    key = df["is_inpatient"].astype(str) + "|" + df["ClmDiagnosisCode_1"].map(_code)
    counts = key.value_counts()
    assert set(ctx5.dx_amount_median) == set(counts[counts >= 5].index)
    k = next(iter(ctx5.dx_amount_median))
    sub = df[key == k]["InscClaimAmtReimbursed"].clip(lower=0)
    assert ctx5.dx_amount_median[k] == pytest.approx(float(np.median(np.log1p(sub))))


def test_context_rare_pair_threshold_is_95th_percentile(train_claims, ctx):
    pair = train_claims["ClmDiagnosisCode_1"].map(_code) + "|" + train_claims["ClmProcedureCode_1"].map(_code)
    rar = [ctx.pair_rarity(*p.split("|")) for p in pair]
    assert ctx.rare_pair_threshold == pytest.approx(float(np.quantile(rar, 0.95)))


def test_context_json_round_trip(ctx, live_claims):
    back = FeatureContext.from_json(ctx.to_json())
    assert back == ctx
    assert_close(claim_features_batch(live_claims, back), claim_features_batch(live_claims, ctx), atol=0, rtol=0)


def test_context_lookups(ctx):
    assert ctx.dx_rarity("never-seen") == pytest.approx(math.log1p(ctx.dx_total))
    assert ctx.pair_rarity("never", "seen") == pytest.approx(math.log1p(ctx.pair_total))
    common = max(ctx.dx_counts, key=ctx.dx_counts.get)
    assert 0 <= ctx.dx_rarity(common) < ctx.dx_rarity("never-seen")
    assert ctx.typical_log_amount(1, "never-seen") == ctx.global_amount_median["1"]
    assert ctx.typical_log_amount(0, "never-seen") == ctx.global_amount_median["0"]
    k = next(iter(ctx.dx_amount_median))
    inp, dx = k.split("|")
    assert ctx.typical_log_amount(int(inp), dx) == ctx.dx_amount_median[k]
    assert FeatureContext().typical_log_amount(1, "x") == 0.0


# ------------------------------------------------------------------------------ claim feature parity
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_claim_features_stream_equals_batch_random(seed, ctx):
    df = make_claims(seed=seed, n=250, unseen_share=0.15)
    batch = claim_features_batch(df, ctx)
    assert list(batch.columns) == CLAIM_FEATURES
    assert_close(stream_features(df, ctx), batch)


def test_claim_features_api_style_dicts_equal_batch(live_claims, ctx):
    """ISO strings / 'NA' / None (what JSON gives the API) must give the same features as Timestamps."""
    batch = claim_features_batch(live_claims, ctx)
    for i, r in enumerate(live_claims.to_dict("records")):
        c = api_style(r)
        f = claim_features(c, c, ctx)
        assert_close([f[k] for k in CLAIM_FEATURES], batch.iloc[i].to_numpy())


def test_claim_features_batch_accepts_pandas_str_dtype(live_claims, ctx):
    """pandas 3 default string dtype (missing = NaN) must give the same features as object columns."""
    s = live_claims.copy()
    for c in ["ClaimID", "BeneID", "Provider", "AttendingPhysician", "OperatingPhysician", "OtherPhysician",
              "ClmAdmitDiagnosisCode", *DIAG_COLS, *PROC_COLS]:
        s[c] = s[c].astype("str")
    assert_close(claim_features_batch(s, ctx), claim_features_batch(live_claims, ctx))
    assert_close(stream_features(s, ctx), claim_features_batch(live_claims, ctx))


def test_claim_features_batch_preserves_index(live_claims, ctx):
    df = live_claims.iloc[::-1].copy()
    df.index = [f"r{i}" for i in range(len(df))]
    out = claim_features_batch(df, ctx)
    assert list(out.index) == list(df.index)
    assert_close(out, stream_features(df, ctx))


def test_claim_features_keys_and_types(ctx):
    f = claim_features(base_claim(), base_bene(), ctx)
    assert list(f) == CLAIM_FEATURES
    assert all(isinstance(v, float) for v in f.values())


def test_claim_features_known_values(ctx):
    claim = base_claim(OperatingPhysician="PHY1", OtherPhysician="PHY9", ClmDiagnosisCode_2="25000",
                       ClmProcedureCode_1=7092.0)
    bene = base_bene(RenalDiseaseIndicator=1, ChronicCond_Cancer=1, ChronicCond_stroke=1, NoOfMonths_PartBCov=6)
    f = claim_features(claim, bene, ctx)
    assert f["log_amount"] == pytest.approx(math.log1p(26000))
    assert f["log_deductible"] == pytest.approx(math.log1p(1068))
    assert f["claim_days"] == 7.0
    assert f["admit_days"] == 6.0
    assert f["is_inpatient"] == 1.0
    assert f["n_diag"] == 2.0 and f["n_proc"] == 1.0
    assert f["has_operating"] == 1.0 and f["has_other"] == 1.0 and f["att_is_op"] == 1.0
    assert f["age"] == pytest.approx((dt.date(2009, 4, 12) - dt.date(1943, 1, 1)).days / 365.25)
    assert f["n_chronic"] == 2.0 and f["renal"] == 1.0
    assert f["part_a_months"] == 12.0 and f["part_b_months"] == 6.0
    assert f["admit_dx_matches"] == 1.0
    assert f["dx_px_rarity"] == pytest.approx(ctx.pair_rarity("4019", "7092"))
    assert f["amount_vs_dx"] == pytest.approx(math.log1p(26000) - ctx.typical_log_amount(1, "4019"))


def check_one_claim_parity(claim, bene, ctx):
    f = claim_features(claim, bene, ctx)
    b = claim_features_batch(as_frame(claim, bene), ctx).iloc[0]
    assert_close([f[k] for k in CLAIM_FEATURES], b.to_numpy())
    return f


@pytest.mark.parametrize("proc", [7092.0, "7092", "7092.0", np.float64(7092.0)])
def test_float_and_string_procedure_codes_are_the_same(proc, ctx):
    f = check_one_claim_parity(base_claim(ClmProcedureCode_1=proc), base_bene(), ctx)
    ref = claim_features(base_claim(ClmProcedureCode_1="7092"), base_bene(), ctx)
    assert f == ref
    assert ctx.pair_counts.get("4019|7092", 0) > 0 or f["dx_px_rarity"] == pytest.approx(math.log1p(ctx.pair_total))


def test_float_procedure_codes_in_mixed_object_column_batch(ctx):
    rows = [base_claim(ClmProcedureCode_1=v, ClaimID=f"C{i}") for i, v in enumerate([7092.0, "7092", None])]
    df = pd.concat([as_frame(r, base_bene()) for r in rows], ignore_index=True)
    out = claim_features_batch(df, ctx)
    assert out["dx_px_rarity"].iloc[0] == out["dx_px_rarity"].iloc[1]
    assert out["n_proc"].tolist() == [1.0, 1.0, 0.0]


@pytest.mark.parametrize("over,checks", [
    ({"ClaimStartDt": None}, {"claim_days": 1.0, "age": 0.0}),
    ({"ClaimStartDt": "NA"}, {"claim_days": 1.0, "age": 0.0}),
    ({"ClaimEndDt": None}, {"claim_days": 1.0}),
    ({"AdmissionDt": None}, {"admit_days": 0.0}),
    ({"DischargeDt": "NA"}, {"admit_days": 0.0}),
    ({"ClaimEndDt": "2009-04-01"}, {"claim_days": 0.0}),           # end before start clips at 0
    ({"ClaimEndDt": "2011-04-12"}, {"claim_days": MAX_CLAIM_DAYS}),  # absurdly long clips at max
    ({"DischargeDt": "2009-04-01"}, {"admit_days": 0.0}),
    ({"InscClaimAmtReimbursed": -500.0}, {"log_amount": 0.0}),
    ({"InscClaimAmtReimbursed": float("nan")}, {"log_amount": 0.0}),
    ({"InscClaimAmtReimbursed": None}, {"log_amount": 0.0}),
    ({"DeductibleAmtPaid": float("nan")}, {"log_deductible": 0.0}),
    ({"DeductibleAmtPaid": -10.0}, {"log_deductible": 0.0}),
    ({"AttendingPhysician": None, "OperatingPhysician": None}, {"att_is_op": 0.0, "has_operating": 0.0}),
    ({"AttendingPhysician": "NA", "OperatingPhysician": "NA"}, {"att_is_op": 0.0, "has_operating": 0.0}),
    ({"OperatingPhysician": "PHY2"}, {"att_is_op": 0.0, "has_operating": 1.0}),
    ({"ClmAdmitDiagnosisCode": None}, {"admit_dx_matches": 0.0}),
    ({"ClmAdmitDiagnosisCode": "NA", "ClmDiagnosisCode_1": "NA"}, {"admit_dx_matches": 0.0, "n_diag": 0.0}),
    ({"ClmAdmitDiagnosisCode": None, "ClmDiagnosisCode_1": None}, {"admit_dx_matches": 0.0, "n_diag": 0.0}),
    ({"ClmDiagnosisCode_2": "NA", "ClmDiagnosisCode_3": "", "ClmDiagnosisCode_4": "V5866"}, {"n_diag": 2.0}),
])
def test_edge_cases_stream_equals_batch(over, checks, ctx):
    f = check_one_claim_parity(base_claim(**over), base_bene(), ctx)
    for k, v in checks.items():
        assert f[k] == pytest.approx(v), k


def test_missing_dob_gives_age_zero(ctx):
    f = check_one_claim_parity(base_claim(), base_bene(DOB=None), ctx)
    assert f["age"] == 0.0


def test_missing_beneficiary_numbers_default_to_zero(ctx):
    bene = base_bene(RenalDiseaseIndicator=float("nan"), NoOfMonths_PartACov=float("nan"))
    f = check_one_claim_parity(base_claim(), bene, ctx)
    assert f["renal"] == 0.0 and f["part_a_months"] == 0.0


def test_unseen_diagnosis_code(ctx):
    f = check_one_claim_parity(base_claim(ClmDiagnosisCode_1="ZZZ99", ClmAdmitDiagnosisCode="ZZZ99"),
                               base_bene(), ctx)
    assert f["dx_rarity"] == pytest.approx(math.log1p(ctx.dx_total))
    assert f["dx_px_rarity"] == pytest.approx(math.log1p(ctx.pair_total))
    assert f["amount_vs_dx"] == pytest.approx(math.log1p(26000) - ctx.global_amount_median["1"])
    assert f["admit_dx_matches"] == 1.0


def test_unseen_codes_are_rarer_than_seen(ctx, live_claims):
    out = claim_features_batch(live_claims, ctx)
    dx = live_claims["ClmDiagnosisCode_1"].map(_code)
    unseen = dx.isin(["99999", "V9999", "E0000"])
    assert unseen.any()
    assert out.loc[unseen, "dx_rarity"].min() > out.loc[~unseen & (dx != "NA"), "dx_rarity"].max()


def test_claim_features_all_finite(live_claims, ctx):
    assert np.isfinite(claim_features_batch(live_claims, ctx).to_numpy()).all()


# ------------------------------------------------------------------------------ hard rules
@pytest.mark.parametrize("claim_over,bene_over,expected", [
    ({}, {}, []),
    ({}, {"DOD": "2009-04-01"}, ["after_death"]),
    ({}, {"DOD": "2009-04-12"}, []),                       # same day as death is allowed
    ({"ClaimStartDt": None}, {"DOD": "2009-04-01"}, []),
    ({"ClaimEndDt": "2009-04-11"}, {}, ["end_before_start"]),
    ({"ClaimEndDt": "2009-04-12"}, {}, []),
    ({"DischargeDt": "2009-04-11"}, {}, ["discharge_before_admission"]),
    ({"DischargeDt": "NA"}, {}, []),
    ({"AttendingPhysician": None}, {}, ["missing_attending"]),
    ({"AttendingPhysician": "NA"}, {}, ["missing_attending"]),
    ({"AttendingPhysician": "  "}, {}, ["missing_attending"]),
    ({"ClaimEndDt": "2009-04-01", "DischargeDt": "2009-04-01", "AttendingPhysician": None},
     {"DOD": "2009-01-01", "NoOfMonths_PartACov": 0},
     ["after_death", "end_before_start", "discharge_before_admission", "missing_attending", "no_coverage"]),
])
def test_hard_errors_rules(claim_over, bene_over, expected):
    claim, bene = base_claim(**claim_over), base_bene(**bene_over)
    assert hard_errors(claim, bene) == expected
    b = hard_errors_batch(as_frame(claim, bene)).iloc[0]
    assert [r for r in RULES_NO_DUP if b[r]] == expected
    assert not b["duplicate"]
    assert all(r in HARD_RULES for r in expected)


@pytest.mark.parametrize("inpatient,part_a,part_b,fires", [
    (1, 0, 12, True),             # inpatient needs Part A
    (1, 12, 0, False),
    (0, 12, 0, True),             # outpatient needs Part B
    (0, 0, 12, False),
    (1, -1, 12, True),
    (0, 12, -3, True),
    (1, 1, 1, False),
    (0, 0.5, 0.5, False),
    (1, None, 12, False),         # missing months -> no fire
    (0, 12, float("nan"), False),
    (1, float("nan"), 0, False),  # only the relevant part counts, even when it is missing
])
def test_no_coverage_rule(inpatient, part_a, part_b, fires):
    claim = base_claim(is_inpatient=inpatient)
    if not inpatient:
        claim.update(AdmissionDt=None, DischargeDt=None)
    bene = base_bene(NoOfMonths_PartACov=part_a, NoOfMonths_PartBCov=part_b)
    assert ("no_coverage" in hard_errors(claim, bene)) is fires
    df = as_frame(claim, bene)
    for c in ["NoOfMonths_PartACov", "NoOfMonths_PartBCov"]:   # loader-style numeric columns (missing = NaN)
        df[c] = df[c].astype(float)
    assert bool(hard_errors_batch(df).iloc[0]["no_coverage"]) is fires
    assert "no_coverage" in HARD_RULES


def test_no_coverage_batch_accepts_none_months_in_object_column():
    df = as_frame(base_claim(), base_bene(NoOfMonths_PartACov=None))
    assert df["NoOfMonths_PartACov"].dtype == object
    assert claim_features_batch(df, FeatureContext())["part_a_months"].iloc[0] == 0.0   # features cope
    assert not hard_errors_batch(df).iloc[0]["no_coverage"]


def test_no_coverage_rule_with_string_months_in_api_dicts():
    assert "no_coverage" in hard_errors(base_claim(is_inpatient=0), base_bene(NoOfMonths_PartBCov="0"))
    assert hard_errors(base_claim(is_inpatient=0), base_bene(NoOfMonths_PartBCov="NA")) == []


def test_hard_errors_batch_equals_stream(live_claims):
    batch = hard_errors_batch(live_claims)
    assert list(batch.columns) == RULES_NO_DUP + ["duplicate"]
    assert (batch.dtypes == bool).all()
    for i, r in enumerate(live_claims.to_dict("records")):
        expected = hard_errors(r, r)
        assert [k for k in RULES_NO_DUP if batch.iloc[i][k]] == expected
        assert hard_errors(api_style(r), api_style(r)) == expected
    # the synthetic generator injects every rule at least once (incl. 0 coverage months)
    assert batch[RULES_NO_DUP].any().all()


def test_no_coverage_parity_with_missing_months(live_claims):
    df = live_claims.copy()
    df["NoOfMonths_PartACov"] = df["NoOfMonths_PartACov"].astype(float)
    df.loc[df.index[::7], "NoOfMonths_PartACov"] = float("nan")
    df.loc[df.index[::5], "NoOfMonths_PartBCov"] = -1
    batch = hard_errors_batch(df)
    for i, r in enumerate(df.to_dict("records")):
        assert ("no_coverage" in hard_errors(r, r)) == bool(batch.iloc[i]["no_coverage"])
    assert batch["no_coverage"].any() and not batch["no_coverage"].all()


def test_duplicate_batch_marks_later_copies_only(live_claims):
    df = pd.concat([live_claims.iloc[:5], live_claims.iloc[[1, 3]].assign(ClaimID=["X1", "X2"])],
                   ignore_index=True)
    dup = hard_errors_batch(df)["duplicate"]
    assert dup.tolist() == [False] * 5 + [True, True]


def test_duplicate_stream_equals_batch(live_claims):
    df = live_claims[live_claims["InscClaimAmtReimbursed"].notna()]
    copies = df.sample(30, random_state=0).assign(ClaimID=lambda d: "DUP" + d["ClaimID"])
    df = pd.concat([df, copies], ignore_index=True)
    batch = hard_errors_batch(df)["duplicate"].tolist()
    idx = DuplicateIndex()
    stream = [idx.check_and_add(duplicate_key(r, r["BeneID"], r["Provider"])) for r in df.to_dict("records")]
    assert stream == batch
    assert sum(stream) >= 30


def test_duplicate_key_normalises_codes_and_amounts():
    a = duplicate_key(base_claim(ClmDiagnosisCode_1=4019.0, InscClaimAmtReimbursed=100), "B", "P")
    b = duplicate_key(base_claim(ClmDiagnosisCode_1="4019", InscClaimAmtReimbursed="100.00"), "B", "P")
    assert a == b == "B|P|2009-04-12|100.00|4019"
    assert duplicate_key(base_claim(ClaimStartDt=None), "B", "P").split("|")[2] == "NA"
    assert duplicate_key(base_claim(), "B", "P") != duplicate_key(base_claim(), "B", "P2")
    assert duplicate_key(base_claim(), "B", "P") != duplicate_key(base_claim(InscClaimAmtReimbursed=101), "B", "P")


def test_duplicate_index_basic():
    idx = DuplicateIndex()
    assert idx.check_and_add("a") is False
    assert idx.check_and_add("a") is True
    assert idx.check_and_add("b") is False


def test_duplicate_index_evicts_oldest_at_max_size():
    idx = DuplicateIndex(max_size=3)
    for k in "abc":
        assert not idx.check_and_add(k)
    assert not idx.check_and_add("d")      # evicts "a"
    assert len(idx._seen) == 3
    assert idx.check_and_add("b") and idx.check_and_add("c") and idx.check_and_add("d")
    assert not idx.check_and_add("a")      # forgotten, so new again; evicts "b"
    assert not idx.check_and_add("b")
    assert len(idx._seen) == 3


# ------------------------------------------------------------------------------ provider state parity
def stream_provider_states(df, feats, thr):
    states = {}
    for (_, r), (_, f) in zip(df.iterrows(), feats.iterrows()):
        st = states.setdefault(r["Provider"], ProviderState())
        st.update(f.to_dict(), r["BeneID"], attending_key(r), thr)
    return states


@pytest.mark.parametrize("seed", [21, 22])
def test_provider_state_equals_batch(seed, ctx):
    df = make_claims(seed=seed, n=300, n_providers=6, unseen_share=0.1)
    feats = claim_features_batch(df, ctx)
    batch = provider_features_batch(df, feats, ctx.rare_pair_threshold)
    assert list(batch.columns) == PROVIDER_FEATURES
    assert list(batch.index) == sorted(df["Provider"].unique())
    states = stream_provider_states(df, feats, ctx.rare_pair_threshold)
    for prv, st in states.items():
        f = st.features()
        assert list(f) == PROVIDER_FEATURES
        assert_close([f[k] for k in PROVIDER_FEATURES], batch.loc[prv].to_numpy())
        assert st.n == (df["Provider"] == prv).sum()


def test_provider_state_from_streaming_claim_features(live_claims, ctx):
    """End to end streaming path (claim_features -> ProviderState) equals the full batch path."""
    states = {}
    for r in live_claims.to_dict("records"):
        st = states.setdefault(r["Provider"], ProviderState())
        st.update(claim_features(r, r, ctx), r["BeneID"], attending_key(r), ctx.rare_pair_threshold)
    batch = provider_features_batch(live_claims, claim_features_batch(live_claims, ctx), ctx.rare_pair_threshold)
    for prv, st in states.items():
        assert_close([st.features()[k] for k in PROVIDER_FEATURES], batch.loc[prv].to_numpy())


def test_provider_state_counts_and_missing_attending():
    st = ProviderState()
    base = dict.fromkeys(CLAIM_FEATURES, 0.0)
    st.update({**base, "log_amount": 2.0, "is_inpatient": 1.0, "admit_days": 4.0, "dx_px_rarity": 9.0},
              "B1", "D1", 5.0)
    st.update({**base, "log_amount": 4.0, "att_is_op": 1.0}, "B1", None, 5.0)
    st.update({**base, "log_amount": 3.0}, "B2", "D1", 5.0)
    f = st.features()
    assert f["log_claims"] == pytest.approx(math.log1p(3))
    assert f["mean_log_amount"] == pytest.approx(3.0)
    assert f["max_log_amount"] == 4.0
    assert f["inpatient_share"] == pytest.approx(1 / 3)
    assert f["mean_admit_days"] == 4.0          # averaged over inpatient claims only
    assert f["log_unique_benes"] == pytest.approx(math.log1p(2))
    assert f["claims_per_bene"] == 1.5
    assert f["log_unique_attending"] == pytest.approx(math.log1p(1))
    assert f["claims_per_attending"] == 2.0     # claims WITH an attending / distinct attendings
    assert f["missing_attending_share"] == pytest.approx(1 / 3)
    assert f["rare_pair_share"] == pytest.approx(1 / 3)
    assert f["att_is_op_share"] == pytest.approx(1 / 3)


def test_empty_provider_state_has_finite_features():
    f = ProviderState().features()
    assert list(f) == PROVIDER_FEATURES
    assert all(math.isfinite(v) for v in f.values())
    assert f["log_claims"] == 0.0


def test_provider_batch_all_attending_missing(ctx):
    df = make_claims(seed=5, n=20, n_providers=1, messy=False)
    df["AttendingPhysician"] = None
    feats = claim_features_batch(df, ctx)
    batch = provider_features_batch(df, feats, ctx.rare_pair_threshold).iloc[0]
    st = stream_provider_states(df, feats, ctx.rare_pair_threshold)["PRV51000"].features()
    assert batch["missing_attending_share"] == 1.0 and batch["claims_per_attending"] == 0.0
    assert_close([st[k] for k in PROVIDER_FEATURES], batch.to_numpy())


@pytest.mark.parametrize("fractions", [(0.25, 0.5, 0.75, 1.0), (0.1, 0.33, 0.9)])
def test_provider_prefixes_equal_stream_on_first_ceil_f_n_claims(fractions, ctx):
    df = make_claims(seed=31, n=230, n_providers=7, unseen_share=0.1)
    feats = claim_features_batch(df, ctx)
    thr = ctx.rare_pair_threshold
    pref = provider_features_prefixes(df, feats, thr, fractions=fractions)
    assert len(pref) == df["Provider"].nunique() * len(fractions)
    assert sorted(pref["fraction"].unique()) == sorted(float(f) for f in fractions)
    for prv, grp in df.groupby("Provider"):
        rows = list(grp.index)
        for f in fractions:
            k = math.ceil(f * len(rows))
            st = ProviderState()
            for i in rows[:k]:
                r = df.loc[i]
                st.update(feats.loc[i].to_dict(), r["BeneID"], attending_key(r), thr)
            got = pref[(pref.index == prv) & (pref["fraction"] == float(f))]
            assert len(got) == 1
            assert st.n == k
            assert_close([st.features()[c] for c in PROVIDER_FEATURES], got[PROVIDER_FEATURES].iloc[0].to_numpy())


def test_provider_prefix_full_fraction_equals_batch(live_claims, ctx):
    feats = claim_features_batch(live_claims, ctx)
    full = provider_features_batch(live_claims, feats, ctx.rare_pair_threshold)
    pref = provider_features_prefixes(live_claims, feats, ctx.rare_pair_threshold, fractions=(1.0,))
    assert_close(pref[PROVIDER_FEATURES].sort_index(), full.sort_index())
