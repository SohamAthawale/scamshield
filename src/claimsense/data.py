"""Loading the public Kaggle Medicare-style dataset and splitting it by provider.

File names carry a numeric suffix, so they are discovered by glob, never hardcoded.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

DIAG_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
CHRONIC_COLS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
]
CLAIM_DATE_COLS = ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]
BENE_DATE_COLS = ["DOB", "DOD"]
CODE_COLS = DIAG_COLS + PROC_COLS + ["ClmAdmitDiagnosisCode", "DiagnosisGroupCode"]


def _find(raw_dir: str, pattern: str) -> str:
    hits = sorted(glob.glob(os.path.join(raw_dir, pattern)))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern!r} in {raw_dir}")
    return hits[0]


def _code_str(s: pd.Series) -> pd.Series:
    """Codes come in as str or float (procedure codes); normalise to str without '.0', missing -> NA."""
    if pd.api.types.is_float_dtype(s):
        out = s.map(lambda v: None if pd.isna(v) else str(int(v)))
    else:
        out = s.map(lambda v: None if pd.isna(v) else str(v))
    return out.astype("object")


def load_beneficiaries(raw_dir: str) -> pd.DataFrame:
    b = pd.read_csv(_find(raw_dir, "Train_Beneficiarydata*.csv"))
    for c in BENE_DATE_COLS:
        b[c] = pd.to_datetime(b[c], errors="coerce")
    b["RenalDiseaseIndicator"] = (b["RenalDiseaseIndicator"].astype(str) == "Y").astype(int)
    for c in CHRONIC_COLS:
        b[c] = (b[c] == 1).astype(int)  # source encodes 1 = yes, 2 = no
    return b


def load_labels(raw_dir: str) -> pd.DataFrame:
    lab = pd.read_csv(_find(raw_dir, "Train-*.csv"))
    lab["label"] = (lab["PotentialFraud"] == "Yes").astype(int)
    return lab[["Provider", "label"]]


def load_claims(raw_dir: str) -> pd.DataFrame:
    """Inpatient and outpatient claims unified, with is_inpatient, sorted by start date then ClaimID."""
    # codes are identifiers, not numbers: read them as text so leading zeros ("0389") survive and
    # files whose code column happens to be all-numeric don't turn "4019" into "4019.0"
    text = {c: str for c in CODE_COLS}
    ip = pd.read_csv(_find(raw_dir, "Train_Inpatientdata*.csv"), dtype=text)
    op = pd.read_csv(_find(raw_dir, "Train_Outpatientdata*.csv"), dtype=text)
    ip["is_inpatient"] = 1
    op["is_inpatient"] = 0
    c = pd.concat([ip, op], ignore_index=True, sort=False)
    for col in CLAIM_DATE_COLS:
        c[col] = pd.to_datetime(c[col], errors="coerce")
    for col in CODE_COLS:
        if col in c:
            c[col] = _code_str(c[col])
    for col in ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]:
        c[col] = c[col].astype("object").where(c[col].notna(), None)
    c["DeductibleAmtPaid"] = c["DeductibleAmtPaid"].fillna(0.0)
    c = c.sort_values(["ClaimStartDt", "ClaimID"], kind="mergesort").reset_index(drop=True)
    return c


def load_all(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(claims joined with beneficiary fields, beneficiaries, provider labels)."""
    bene = load_beneficiaries(raw_dir)
    claims = load_claims(raw_dir).merge(bene, on="BeneID", how="left", validate="many_to_one")
    labels = load_labels(raw_dir)
    return claims, bene, labels


def split_providers(labels: pd.DataFrame, seed: int = 42,
                    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> pd.DataFrame:
    """Stratified split BY PROVIDER (never by claim). Returns Provider, label, split."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in labels.groupby("label"):
        idx = rng.permutation(len(grp))
        n_train = int(round(fractions[0] * len(grp)))
        n_val = int(round(fractions[1] * len(grp)))
        split = np.empty(len(grp), dtype=object)
        split[idx[:n_train]] = "train"
        split[idx[n_train:n_train + n_val]] = "val"
        split[idx[n_train + n_val:]] = "test"
        parts.append(grp.assign(split=split))
    return pd.concat(parts).sort_values("Provider").reset_index(drop=True)
