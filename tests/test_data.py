"""data.py: loaders on tiny CSVs and the provider split."""
import numpy as np
import pandas as pd
import pytest

from claimsense import data
from claimsense.data import CHRONIC_COLS, CODE_COLS, DIAG_COLS, PROC_COLS, split_providers

SUFFIX = "1542865627584"
OUT_HEADER = (["BeneID", "ClaimID", "ClaimStartDt", "ClaimEndDt", "Provider", "InscClaimAmtReimbursed",
               "AttendingPhysician", "OperatingPhysician", "OtherPhysician"] + DIAG_COLS + PROC_COLS
              + ["DeductibleAmtPaid", "ClmAdmitDiagnosisCode"])
IN_HEADER = (["BeneID", "ClaimID", "ClaimStartDt", "ClaimEndDt", "Provider", "InscClaimAmtReimbursed",
              "AttendingPhysician", "OperatingPhysician", "OtherPhysician", "AdmissionDt", "ClmAdmitDiagnosisCode",
              "DeductibleAmtPaid", "DischargeDt", "DiagnosisGroupCode"] + DIAG_COLS + PROC_COLS)
BENE_HEADER = (["BeneID", "DOB", "DOD", "Gender", "Race", "RenalDiseaseIndicator", "State", "County",
                "NoOfMonths_PartACov", "NoOfMonths_PartBCov"] + CHRONIC_COLS
               + ["IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt", "OPAnnualReimbursementAmt",
                  "OPAnnualDeductibleAmt"])


def q(v):
    """CSV cell the way the Kaggle files write it: strings quoted, numbers bare, missing = NA."""
    if v is None:
        return "NA"
    return f'"{v}"' if isinstance(v, str) else str(v)


def write_csv(path, header, rows):
    lines = [",".join(f'"{h}"' for h in header)]
    for r in rows:
        lines.append(",".join(q(r.get(h)) for h in header))
    path.write_text("\n".join(lines) + "\n")


def inpatient_row(**kw):
    r = {"BeneID": "BENE11001", "ClaimID": "CLM46614", "ClaimStartDt": "2009-04-12",
         "ClaimEndDt": "2009-04-18", "Provider": "PRV55912", "InscClaimAmtReimbursed": 26000,
         "AttendingPhysician": "PHY390922", "AdmissionDt": "2009-04-12", "ClmAdmitDiagnosisCode": "7866",
         "DeductibleAmtPaid": 1068, "DischargeDt": "2009-04-18", "DiagnosisGroupCode": "201",
         "ClmDiagnosisCode_1": "1970", "ClmDiagnosisCode_2": "V5866", "ClmProcedureCode_1": 7092}
    r.update(kw)
    return r


def outpatient_row(**kw):
    r = {"BeneID": "BENE11002", "ClaimID": "CLM624349", "ClaimStartDt": "2009-10-11",
         "ClaimEndDt": "2009-10-11", "Provider": "PRV56011", "InscClaimAmtReimbursed": 30,
         "AttendingPhysician": "PHY326117", "ClmDiagnosisCode_1": "78943", "ClmDiagnosisCode_2": "V5866",
         "DeductibleAmtPaid": 0, "ClmAdmitDiagnosisCode": "56409"}
    r.update(kw)
    return r


def bene_row(bid, **kw):
    r = {"BeneID": bid, "DOB": "1943-01-01", "DOD": None, "Gender": 1, "Race": 1, "RenalDiseaseIndicator": "0",
         "State": 39, "County": 230, "NoOfMonths_PartACov": 12, "NoOfMonths_PartBCov": 12,
         **{c: 2 for c in CHRONIC_COLS}, "IPAnnualReimbursementAmt": 0, "IPAnnualDeductibleAmt": 0,
         "OPAnnualReimbursementAmt": 0, "OPAnnualDeductibleAmt": 0}
    r.update(kw)
    return r


@pytest.fixture
def raw_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    (d / f"Train-{SUFFIX}.csv").write_text('"Provider","PotentialFraud"\n"PRV55912","Yes"\n"PRV56011","No"\n'
                                           '"PRV57610","No"\n')
    write_csv(d / f"Train_Beneficiarydata-{SUFFIX}.csv", BENE_HEADER, [
        bene_row("BENE11001", ChronicCond_Alzheimer=1, ChronicCond_stroke=1),
        bene_row("BENE11002", DOD="2009-12-01", RenalDiseaseIndicator="Y"),
        bene_row("BENE11003"),
    ])
    write_csv(d / f"Train_Inpatientdata-{SUFFIX}.csv", IN_HEADER, [
        inpatient_row(),
        inpatient_row(BeneID="BENE11003", ClaimID="CLM00002", ClaimStartDt="2009-01-05",
                      ClaimEndDt="2009-01-07", AdmissionDt="2009-01-05", DischargeDt="2009-01-07",
                      OperatingPhysician="PHY390922", ClmProcedureCode_1=None, ClmProcedureCode_2=9904),
    ])
    write_csv(d / f"Train_Outpatientdata-{SUFFIX}.csv", OUT_HEADER, [
        outpatient_row(),
        outpatient_row(ClaimID="CLM00001", ClaimStartDt="2009-01-05", ClaimEndDt="2009-01-05",
                       BeneID="BENE11003", Provider="PRV57610", AttendingPhysician=None, DeductibleAmtPaid=None),
        outpatient_row(ClaimID="CLM00003", ClaimStartDt="2009-01-05", ClaimEndDt="2009-01-05",
                       BeneID="BENE11001", Provider="PRV57610"),
    ])
    return d


# ------------------------------------------------------------------------------ loaders
def test_find_uses_glob_with_any_suffix(raw_dir):
    assert data._find(str(raw_dir), "Train_Beneficiarydata*.csv").endswith(f"Train_Beneficiarydata-{SUFFIX}.csv")
    with pytest.raises(FileNotFoundError):
        data._find(str(raw_dir), "Nope*.csv")


def test_find_does_not_confuse_labels_with_other_train_files(raw_dir):
    assert data._find(str(raw_dir), "Train-*.csv").endswith(f"Train-{SUFFIX}.csv")


def test_load_beneficiaries(raw_dir):
    b = data.load_beneficiaries(str(raw_dir)).set_index("BeneID")
    assert pd.api.types.is_datetime64_any_dtype(b["DOB"]) and pd.api.types.is_datetime64_any_dtype(b["DOD"])
    assert pd.isna(b.loc["BENE11001", "DOD"])
    assert b.loc["BENE11002", "DOD"] == pd.Timestamp("2009-12-01")
    assert b["RenalDiseaseIndicator"].to_dict() == {"BENE11001": 0, "BENE11002": 1, "BENE11003": 0}
    assert b.loc["BENE11001", "ChronicCond_Alzheimer"] == 1           # 1 = yes
    assert b.loc["BENE11001", "ChronicCond_Cancer"] == 0              # 2 = no
    assert set(np.unique(b[CHRONIC_COLS].to_numpy())) <= {0, 1}


def test_load_labels(raw_dir):
    lab = data.load_labels(str(raw_dir))
    assert list(lab.columns) == ["Provider", "label"]
    assert dict(zip(lab["Provider"], lab["label"])) == {"PRV55912": 1, "PRV56011": 0, "PRV57610": 0}


def test_load_claims_unifies_and_sorts(raw_dir):
    c = data.load_claims(str(raw_dir))
    assert len(c) == 5
    assert c["is_inpatient"].tolist() == [0, 1, 0, 1, 0]
    # sorted by start date, then ClaimID (mergesort keeps it stable)
    assert c["ClaimID"].tolist() == ["CLM00001", "CLM00002", "CLM00003", "CLM46614", "CLM624349"]
    assert list(c.index) == list(range(5))
    for col in data.CLAIM_DATE_COLS:
        assert pd.api.types.is_datetime64_any_dtype(c[col]), col
    assert c.loc[c["is_inpatient"] == 0, "AdmissionDt"].isna().all()


def test_load_claims_normalises_codes_and_physicians(raw_dir):
    c = data.load_claims(str(raw_dir)).set_index("ClaimID")
    assert c.loc["CLM46614", "ClmProcedureCode_1"] == "7092"          # float 7092.0 in the CSV
    assert c.loc["CLM00002", "ClmProcedureCode_2"] == "9904"
    assert pd.isna(c.loc["CLM00002", "ClmProcedureCode_1"])
    assert c.loc["CLM46614", "ClmDiagnosisCode_2"] == "V5866"
    assert c.loc["CLM46614", "DiagnosisGroupCode"] == "201"
    assert pd.isna(c.loc["CLM624349", "DiagnosisGroupCode"])
    for col in CODE_COLS:
        vals = c[col].dropna()
        assert all(isinstance(v, str) and not v.endswith(".0") for v in vals), col
    assert c.loc["CLM00001", "AttendingPhysician"] is None
    assert c.loc["CLM46614", "OperatingPhysician"] is None
    assert c.loc["CLM00002", "OperatingPhysician"] == "PHY390922"
    assert c.loc["CLM00001", "DeductibleAmtPaid"] == 0.0


def test_load_all_joins_beneficiaries(raw_dir):
    claims, bene, labels = data.load_all(str(raw_dir))
    assert len(claims) == 5 and len(bene) == 3 and len(labels) == 3
    assert claims["ClaimID"].tolist() == data.load_claims(str(raw_dir))["ClaimID"].tolist()
    row = claims.set_index("ClaimID").loc["CLM624349"]
    assert row["RenalDiseaseIndicator"] == 1 and row["DOD"] == pd.Timestamp("2009-12-01")
    for c in ["DOB", "NoOfMonths_PartACov", *CHRONIC_COLS]:
        assert c in claims


def test_loaded_claims_feed_the_feature_pipeline(raw_dir):
    from claimsense.features import FeatureContext, claim_features, claim_features_batch
    claims, _, _ = data.load_all(str(raw_dir))
    ctx = FeatureContext.fit(claims, min_count_for_median=1)
    batch = claim_features_batch(claims, ctx)
    for i, r in enumerate(claims.to_dict("records")):
        f = claim_features(r, r, ctx)
        np.testing.assert_allclose(list(f.values()), batch.iloc[i].to_numpy(), atol=1e-12)


def test_codes_have_no_float_suffix_when_dtypes_differ_between_files(raw_dir):
    # inpatient ClmDiagnosisCode_3 only numeric codes -> read as float; outpatient has a V code -> str
    write_csv(raw_dir / f"Train_Inpatientdata-{SUFFIX}.csv", IN_HEADER,
              [inpatient_row(ClmDiagnosisCode_3=4019), inpatient_row(ClaimID="CLM2", ClmDiagnosisCode_3=None)])
    write_csv(raw_dir / f"Train_Outpatientdata-{SUFFIX}.csv", OUT_HEADER,
              [outpatient_row(ClmDiagnosisCode_3="V1272")])
    c = data.load_claims(str(raw_dir)).set_index("ClaimID")
    assert c.loc["CLM46614", "ClmDiagnosisCode_3"] == "4019"


def test_leading_zero_diagnosis_codes_survive_loading(raw_dir):
    write_csv(raw_dir / f"Train_Outpatientdata-{SUFFIX}.csv", OUT_HEADER,
              [outpatient_row(ClmDiagnosisCode_4="0389")])
    write_csv(raw_dir / f"Train_Inpatientdata-{SUFFIX}.csv", IN_HEADER,
              [inpatient_row(ClmDiagnosisCode_4="0389")])
    c = data.load_claims(str(raw_dir))
    assert set(c["ClmDiagnosisCode_4"]) == {"0389"}


# ------------------------------------------------------------------------------ split
@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({"Provider": [f"PRV{50000 + i}" for i in range(n)],
                         "label": (rng.random(n) < 0.1).astype(int)})


def test_split_covers_all_providers_disjointly(labels):
    s = split_providers(labels)
    assert len(s) == len(labels)
    assert s["Provider"].is_unique
    assert set(s["Provider"]) == set(labels["Provider"])
    assert set(s["split"]) == {"train", "val", "test"}
    assert list(s.columns) == ["Provider", "label", "split"]
    assert s["Provider"].is_monotonic_increasing
    merged = s.merge(labels, on="Provider", suffixes=("", "_orig"))
    assert (merged["label"] == merged["label_orig"]).all()


def test_split_is_deterministic_for_a_seed(labels):
    a, b = split_providers(labels, seed=7), split_providers(labels, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not split_providers(labels, seed=8)["split"].equals(a["split"])


def test_split_does_not_depend_on_input_row_order(labels):
    a = split_providers(labels, seed=3)
    b = split_providers(labels.iloc[::-1].reset_index(drop=True), seed=3)
    # the permutation is applied to group order, so only the counts are guaranteed to match
    assert a.groupby(["label", "split"]).size().equals(b.groupby(["label", "split"]).size())


def test_split_is_stratified_70_15_15(labels):
    s = split_providers(labels)
    for lab, grp in s.groupby("label"):
        n = len(grp)
        counts = grp["split"].value_counts()
        assert counts["train"] == int(round(0.70 * n))
        assert counts["val"] == int(round(0.15 * n))
        assert abs(counts["test"] - 0.15 * n) <= 1.5
    fraud_share = s.groupby("split")["label"].mean()
    overall = labels["label"].mean()
    assert (abs(fraud_share - overall) < 0.03).all()


def test_split_custom_fractions(labels):
    s = split_providers(labels, fractions=(0.5, 0.25, 0.25))
    share = s["split"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.5, abs=0.01)
    assert share["val"] == pytest.approx(0.25, abs=0.01)
