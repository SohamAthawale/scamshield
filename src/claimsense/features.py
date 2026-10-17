"""Single source of truth for features, imported by training code AND the scoring API.

Two levels:
  * claim features   -> the claim autoencoder (is this one claim unusual?)
  * provider features -> the provider risk model (is this provider's pattern unusual?), kept as
    rolling state that is updated one claim at a time in the API.

Every feature has a streaming (one claim at a time) and a batch (DataFrame) implementation.
tests/test_features.py proves they agree.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .data import CHRONIC_COLS, DIAG_COLS, PROC_COLS

# --------------------------------------------------------------------------------------
# Claim-level features
# --------------------------------------------------------------------------------------
CLAIM_FEATURES: list[str] = [
    "log_amount", "log_deductible", "claim_days", "admit_days", "is_inpatient",
    "n_diag", "n_proc", "has_operating", "has_other", "att_is_op",
    "age", "n_chronic", "renal", "part_a_months", "part_b_months",
    "dx_px_rarity", "dx_rarity", "amount_vs_dx", "admit_dx_matches",
]

FEATURE_LABELS: dict[str, str] = {
    "log_amount": "reimbursed amount",
    "log_deductible": "deductible paid",
    "claim_days": "claim duration (days)",
    "admit_days": "length of hospital stay (days)",
    "is_inpatient": "inpatient claim",
    "n_diag": "number of diagnosis codes",
    "n_proc": "number of procedure codes",
    "has_operating": "operating physician listed",
    "has_other": "other physician listed",
    "att_is_op": "attending and operating physician are the same",
    "age": "patient age",
    "n_chronic": "number of chronic conditions",
    "renal": "renal disease indicator",
    "part_a_months": "Part A coverage months",
    "part_b_months": "Part B coverage months",
    "dx_px_rarity": "rarity of the diagnosis + procedure combination",
    "dx_rarity": "rarity of the primary diagnosis",
    "amount_vs_dx": "amount compared with typical claims for this diagnosis",
    "admit_dx_matches": "admitting diagnosis matches primary diagnosis",
}

# Hard rules: deterministic billing errors. Each maps to (short title, how to fix it).
HARD_RULES: dict[str, tuple[str, str]] = {
    "after_death": (
        "Service date is after the patient's recorded date of death",
        "Check the service date and the patient identifier; the claim may be attached to the wrong patient.",
    ),
    "end_before_start": (
        "Claim end date is before the start date",
        "Swap or correct the start and end dates.",
    ),
    "discharge_before_admission": (
        "Discharge date is before the admission date",
        "Correct the admission or discharge date.",
    ),
    "missing_attending": (
        "No attending physician on the claim",
        "Add the attending physician's identifier.",
    ),
    "duplicate": (
        "Looks like a duplicate of a claim already submitted",
        "Check whether this service was already billed; withdraw it if so.",
    ),
}

MAX_CLAIM_DAYS = 400.0


def _date(v: Any) -> dt.date | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.date()
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    s = str(v).strip()
    if not s or s.upper() in {"NA", "NAN", "NAT", "NONE"}:
        return None
    return dt.date.fromisoformat(s[:10])


def _present(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    s = str(v).strip()
    return bool(s) and s.upper() not in {"NA", "NAN", "NONE"}


def _code(v: Any) -> str:
    """Normalise a diagnosis/procedure code to a string key; missing -> 'NA'."""
    if not _present(v):
        return "NA"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s


def _num(v: Any, default: float = 0.0) -> float:
    if not _present(v):
        return default
    return float(v)


@dataclass
class FeatureContext:
    """Lookup tables learned from TRAINING claims only (frequencies and typical amounts)."""

    pair_counts: dict[str, int] = field(default_factory=dict)
    pair_total: int = 0
    dx_counts: dict[str, int] = field(default_factory=dict)
    dx_total: int = 0
    dx_amount_median: dict[str, float] = field(default_factory=dict)  # "inp|dx1" -> median log1p(amount)
    global_amount_median: dict[str, float] = field(default_factory=dict)  # "0"/"1" -> median log1p(amount)
    rare_pair_threshold: float = 0.0
    min_count_for_median: int = 5

    @classmethod
    def fit(cls, claims: pd.DataFrame, min_count_for_median: int = 5) -> "FeatureContext":
        dx = claims["ClmDiagnosisCode_1"].map(_code)
        px = claims["ClmProcedureCode_1"].map(_code)
        pair = dx + "|" + px
        la = np.log1p(claims["InscClaimAmtReimbursed"].clip(lower=0).astype(float))
        inp = claims["is_inpatient"].astype(int).astype(str)
        key = inp + "|" + dx
        g = la.groupby(key)
        med = g.median()[g.size() >= min_count_for_median]
        ctx = cls(
            pair_counts={k: int(v) for k, v in pair.value_counts().items()},
            pair_total=int(len(pair)),
            dx_counts={k: int(v) for k, v in dx.value_counts().items()},
            dx_total=int(len(dx)),
            dx_amount_median={k: float(v) for k, v in med.items()},
            global_amount_median={k: float(v) for k, v in la.groupby(inp).median().items()},
            min_count_for_median=min_count_for_median,
        )
        # a pair is "rare" when it is in the rarest 5% of training claims
        rar = np.log1p(ctx.pair_total) - np.log1p(pair.map(ctx.pair_counts).astype(float))
        ctx.rare_pair_threshold = float(np.quantile(rar, 0.95))
        return ctx

    def pair_rarity(self, dx1: str, px1: str) -> float:
        return math.log1p(self.pair_total) - math.log1p(self.pair_counts.get(f"{dx1}|{px1}", 0))

    def dx_rarity(self, dx1: str) -> float:
        return math.log1p(self.dx_total) - math.log1p(self.dx_counts.get(dx1, 0))

    def typical_log_amount(self, is_inpatient: int, dx1: str) -> float:
        k = f"{int(is_inpatient)}|{dx1}"
        if k in self.dx_amount_median:
            return self.dx_amount_median[k]
        return self.global_amount_median.get(str(int(is_inpatient)), 0.0)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "FeatureContext":
        return cls(**json.loads(s))


def claim_features(claim: Mapping[str, Any], bene: Mapping[str, Any], ctx: FeatureContext) -> dict[str, float]:
    """Streaming path: features for one claim. `bene` holds the beneficiary's enrolment fields."""
    amount = max(_num(claim.get("InscClaimAmtReimbursed")), 0.0)
    start, end = _date(claim.get("ClaimStartDt")), _date(claim.get("ClaimEndDt"))
    adm, dis = _date(claim.get("AdmissionDt")), _date(claim.get("DischargeDt"))
    dob = _date(bene.get("DOB"))
    inp = int(_num(claim.get("is_inpatient")))
    dx1 = _code(claim.get("ClmDiagnosisCode_1"))
    px1 = _code(claim.get("ClmProcedureCode_1"))
    att, op = claim.get("AttendingPhysician"), claim.get("OperatingPhysician")

    claim_days = 1.0 if (start is None or end is None) else float((end - start).days + 1)
    admit_days = float((dis - adm).days) if (adm is not None and dis is not None) else 0.0
    age = (start - dob).days / 365.25 if (start is not None and dob is not None) else 0.0
    admit_dx = claim.get("ClmAdmitDiagnosisCode")

    return {
        "log_amount": math.log1p(amount),
        "log_deductible": math.log1p(max(_num(claim.get("DeductibleAmtPaid")), 0.0)),
        "claim_days": min(max(claim_days, 0.0), MAX_CLAIM_DAYS),
        "admit_days": min(max(admit_days, 0.0), MAX_CLAIM_DAYS),
        "is_inpatient": float(inp),
        "n_diag": float(sum(_present(claim.get(c)) for c in DIAG_COLS)),
        "n_proc": float(sum(_present(claim.get(c)) for c in PROC_COLS)),
        "has_operating": float(_present(op)),
        "has_other": float(_present(claim.get("OtherPhysician"))),
        "att_is_op": float(_present(att) and _present(op) and str(att) == str(op)),
        "age": float(age),
        "n_chronic": float(sum(int(_num(bene.get(c))) for c in CHRONIC_COLS)),
        "renal": float(_num(bene.get("RenalDiseaseIndicator"))),
        "part_a_months": _num(bene.get("NoOfMonths_PartACov")),
        "part_b_months": _num(bene.get("NoOfMonths_PartBCov")),
        "dx_px_rarity": ctx.pair_rarity(dx1, px1),
        "dx_rarity": ctx.dx_rarity(dx1),
        "amount_vs_dx": math.log1p(amount) - ctx.typical_log_amount(inp, dx1),
        "admit_dx_matches": float(_present(admit_dx) and dx1 != "NA" and _code(admit_dx) == dx1),
    }


def claim_features_batch(claims: pd.DataFrame, ctx: FeatureContext) -> pd.DataFrame:
    """Batch path: same features for a DataFrame of claims already joined with beneficiary fields."""
    c = claims
    amount = c["InscClaimAmtReimbursed"].astype(float).fillna(0.0).clip(lower=0.0)
    start, end = pd.to_datetime(c["ClaimStartDt"]), pd.to_datetime(c["ClaimEndDt"])
    adm, dis = pd.to_datetime(c["AdmissionDt"]), pd.to_datetime(c["DischargeDt"])
    dob = pd.to_datetime(c["DOB"])
    inp = c["is_inpatient"].astype(int)
    dx1 = c["ClmDiagnosisCode_1"].map(_code)
    px1 = c["ClmProcedureCode_1"].map(_code)
    att, op = c["AttendingPhysician"], c["OperatingPhysician"]
    att_p, op_p = att.map(_present), op.map(_present)

    claim_days = ((end - start).dt.days + 1).astype(float).where(start.notna() & end.notna(), 1.0)
    admit_days = (dis - adm).dt.days.astype(float).where(adm.notna() & dis.notna(), 0.0)
    age = ((start - dob).dt.days / 365.25).where(start.notna() & dob.notna(), 0.0)
    log_amount = np.log1p(amount)
    typical = pd.Series(
        [ctx.typical_log_amount(i, d) for i, d in zip(inp.tolist(), dx1.tolist())], index=c.index, dtype=float
    )
    admit_dx = c["ClmAdmitDiagnosisCode"].map(_code)

    out = pd.DataFrame(
        {
            "log_amount": log_amount,
            "log_deductible": np.log1p(c["DeductibleAmtPaid"].astype(float).fillna(0.0).clip(lower=0.0)),
            "claim_days": claim_days.clip(0.0, MAX_CLAIM_DAYS),
            "admit_days": admit_days.clip(0.0, MAX_CLAIM_DAYS),
            "is_inpatient": inp.astype(float),
            "n_diag": sum(c[col].map(_present).astype(float) for col in DIAG_COLS),
            "n_proc": sum(c[col].map(_present).astype(float) for col in PROC_COLS),
            "has_operating": op_p.astype(float),
            "has_other": c["OtherPhysician"].map(_present).astype(float),
            "att_is_op": (att_p & op_p & (att.astype(str) == op.astype(str))).astype(float),
            "age": age.astype(float),
            "n_chronic": c[CHRONIC_COLS].fillna(0).astype(int).sum(axis=1).astype(float),
            "renal": c["RenalDiseaseIndicator"].fillna(0).astype(float),
            "part_a_months": c["NoOfMonths_PartACov"].fillna(0).astype(float),
            "part_b_months": c["NoOfMonths_PartBCov"].fillna(0).astype(float),
            "dx_px_rarity": math.log1p(ctx.pair_total)
            - np.log1p((dx1 + "|" + px1).map(lambda k: ctx.pair_counts.get(k, 0)).astype(float)),
            "dx_rarity": math.log1p(ctx.dx_total) - np.log1p(dx1.map(lambda k: ctx.dx_counts.get(k, 0)).astype(float)),
            "amount_vs_dx": log_amount - typical,
            "admit_dx_matches": ((admit_dx != "NA") & (dx1 != "NA") & (admit_dx == dx1)).astype(float),
        },
        index=c.index,
    )
    return out[CLAIM_FEATURES]


# --------------------------------------------------------------------------------------
# Hard rules (deterministic errors)
# --------------------------------------------------------------------------------------
def duplicate_key(claim: Mapping[str, Any], bene_key: str, provider_key: str) -> str:
    """Same patient, same provider, same start date, same amount, same primary diagnosis."""
    start = _date(claim.get("ClaimStartDt"))
    return "|".join([
        bene_key, provider_key, start.isoformat() if start else "NA",
        f"{_num(claim.get('InscClaimAmtReimbursed')):.2f}" if _present(claim.get("InscClaimAmtReimbursed")) else "NA",
        _code(claim.get("ClmDiagnosisCode_1")),
    ])


def hard_errors(claim: Mapping[str, Any], bene: Mapping[str, Any]) -> list[str]:
    """Rule codes (keys of HARD_RULES) that fire for one claim. `duplicate` is handled by DuplicateIndex."""
    out = []
    start, end = _date(claim.get("ClaimStartDt")), _date(claim.get("ClaimEndDt"))
    adm, dis = _date(claim.get("AdmissionDt")), _date(claim.get("DischargeDt"))
    dod = _date(bene.get("DOD"))
    if start is not None and dod is not None and start > dod:
        out.append("after_death")
    if start is not None and end is not None and end < start:
        out.append("end_before_start")
    if adm is not None and dis is not None and dis < adm:
        out.append("discharge_before_admission")
    if not _present(claim.get("AttendingPhysician")):
        out.append("missing_attending")
    return out


def hard_errors_batch(claims: pd.DataFrame) -> pd.DataFrame:
    """Boolean column per rule (including duplicate: later copies of an earlier identical claim)."""
    c = claims
    start, end = pd.to_datetime(c["ClaimStartDt"]), pd.to_datetime(c["ClaimEndDt"])
    adm, dis = pd.to_datetime(c["AdmissionDt"]), pd.to_datetime(c["DischargeDt"])
    dod = pd.to_datetime(c["DOD"])
    dup_cols = ["BeneID", "Provider", "ClaimStartDt", "InscClaimAmtReimbursed", "ClmDiagnosisCode_1"]
    return pd.DataFrame(
        {
            "after_death": (start > dod).fillna(False),
            "end_before_start": (end < start).fillna(False),
            "discharge_before_admission": (dis < adm).fillna(False),
            "missing_attending": ~c["AttendingPhysician"].map(_present),
            "duplicate": c.duplicated(dup_cols, keep="first"),
        },
        index=c.index,
    ).astype(bool)


class DuplicateIndex:
    """Remembers recent claim fingerprints. Bounded so memory stays flat in a long stream."""

    def __init__(self, max_size: int = 2_000_000):
        self._seen: dict[str, None] = {}
        self.max_size = max_size

    def check_and_add(self, key: str) -> bool:
        if key in self._seen:
            return True
        self._seen[key] = None
        if len(self._seen) > self.max_size:
            self._seen.pop(next(iter(self._seen)))
        return False


# --------------------------------------------------------------------------------------
# Provider-level rolling state
# --------------------------------------------------------------------------------------
PROVIDER_FEATURES: list[str] = [
    "log_claims", "mean_log_amount", "max_log_amount", "inpatient_share", "mean_admit_days",
    "mean_claim_days", "log_unique_benes", "claims_per_bene", "log_unique_attending",
    "claims_per_attending", "mean_n_diag", "mean_dx_px_rarity", "rare_pair_share", "mean_age",
    "mean_n_chronic", "att_is_op_share", "missing_attending_share", "mean_amount_vs_dx",
    "mean_log_deductible",
]

PROVIDER_LABELS: dict[str, str] = {
    "log_claims": "claim volume",
    "mean_log_amount": "average reimbursed amount",
    "max_log_amount": "largest reimbursed amount",
    "inpatient_share": "share of inpatient claims",
    "mean_admit_days": "average hospital stay",
    "mean_claim_days": "average claim duration",
    "log_unique_benes": "number of distinct patients",
    "claims_per_bene": "claims per patient",
    "log_unique_attending": "number of distinct attending physicians",
    "claims_per_attending": "claims per attending physician",
    "mean_n_diag": "diagnosis codes per claim",
    "mean_dx_px_rarity": "average rarity of diagnosis + procedure combinations",
    "rare_pair_share": "share of rare diagnosis + procedure combinations",
    "mean_age": "average patient age",
    "mean_n_chronic": "average chronic conditions per patient",
    "att_is_op_share": "share of claims where attending = operating physician",
    "missing_attending_share": "share of claims with no attending physician",
    "mean_amount_vs_dx": "amounts compared with typical for the diagnosis",
    "mean_log_deductible": "average deductible paid",
}


class ProviderState:
    """Incremental aggregates for one provider. update() is O(1); features() is O(1)."""

    __slots__ = ("n", "s_amount", "max_amount", "n_inp", "s_admit", "s_days", "benes", "attending",
                 "s_diag", "s_rarity", "n_rare", "s_age", "s_chronic", "n_att_op", "n_missing_att",
                 "s_amount_vs_dx", "s_deductible")

    def __init__(self) -> None:
        self.n = 0
        self.s_amount = 0.0
        self.max_amount = 0.0
        self.n_inp = 0
        self.s_admit = 0.0
        self.s_days = 0.0
        self.benes: set[str] = set()
        self.attending: set[str] = set()
        self.s_diag = 0.0
        self.s_rarity = 0.0
        self.n_rare = 0
        self.s_age = 0.0
        self.s_chronic = 0.0
        self.n_att_op = 0
        self.n_missing_att = 0
        self.s_amount_vs_dx = 0.0
        self.s_deductible = 0.0

    def update(self, f: Mapping[str, float], bene_key: str, attending_key: str | None,
               rare_pair_threshold: float) -> None:
        self.n += 1
        self.s_amount += f["log_amount"]
        self.max_amount = max(self.max_amount, f["log_amount"])
        if f["is_inpatient"] >= 0.5:
            self.n_inp += 1
            self.s_admit += f["admit_days"]
        self.s_days += f["claim_days"]
        self.benes.add(bene_key)
        if attending_key:
            self.attending.add(attending_key)
        else:
            self.n_missing_att += 1
        self.s_diag += f["n_diag"]
        self.s_rarity += f["dx_px_rarity"]
        self.n_rare += int(f["dx_px_rarity"] > rare_pair_threshold)
        self.s_age += f["age"]
        self.s_chronic += f["n_chronic"]
        self.n_att_op += int(f["att_is_op"] >= 0.5)
        self.s_amount_vs_dx += f["amount_vs_dx"]
        self.s_deductible += f["log_deductible"]

    def features(self) -> dict[str, float]:
        n = max(self.n, 1)
        nb, na = max(len(self.benes), 1), max(len(self.attending), 1)
        return {
            "log_claims": math.log1p(self.n),
            "mean_log_amount": self.s_amount / n,
            "max_log_amount": self.max_amount,
            "inpatient_share": self.n_inp / n,
            "mean_admit_days": self.s_admit / max(self.n_inp, 1),
            "mean_claim_days": self.s_days / n,
            "log_unique_benes": math.log1p(len(self.benes)),
            "claims_per_bene": self.n / nb,
            "log_unique_attending": math.log1p(len(self.attending)),
            "claims_per_attending": (self.n - self.n_missing_att) / na,
            "mean_n_diag": self.s_diag / n,
            "mean_dx_px_rarity": self.s_rarity / n,
            "rare_pair_share": self.n_rare / n,
            "mean_age": self.s_age / n,
            "mean_n_chronic": self.s_chronic / n,
            "att_is_op_share": self.n_att_op / n,
            "missing_attending_share": self.n_missing_att / n,
            "mean_amount_vs_dx": self.s_amount_vs_dx / n,
            "mean_log_deductible": self.s_deductible / n,
        }


def provider_features_batch(claims: pd.DataFrame, feats: pd.DataFrame, rare_pair_threshold: float) -> pd.DataFrame:
    """Batch provider aggregates over all rows given. Index = Provider, columns = PROVIDER_FEATURES."""
    d = feats.copy()
    d["Provider"] = claims["Provider"].values
    d["BeneID"] = claims["BeneID"].values
    att_present = claims["AttendingPhysician"].map(_present).values
    d["att_key"] = np.where(att_present, claims["AttendingPhysician"].astype(str).values, None)
    d["missing_att"] = (~att_present).astype(int)
    d["rare"] = (d["dx_px_rarity"] > rare_pair_threshold).astype(int)
    d["admit_inp"] = np.where(d["is_inpatient"] >= 0.5, d["admit_days"], 0.0)
    d["att_op"] = (d["att_is_op"] >= 0.5).astype(int)
    g = d.groupby("Provider", sort=True)
    n = g.size().astype(float)
    n_inp = g["is_inpatient"].apply(lambda s: float((s >= 0.5).sum()))
    nb = g["BeneID"].nunique().astype(float)
    na = g["att_key"].nunique(dropna=True).astype(float)
    n_missing = g["missing_att"].sum().astype(float)
    out = pd.DataFrame({
        "log_claims": np.log1p(n),
        "mean_log_amount": g["log_amount"].sum() / n,
        "max_log_amount": g["log_amount"].max(),
        "inpatient_share": n_inp / n,
        "mean_admit_days": g["admit_inp"].sum() / n_inp.clip(lower=1.0),
        "mean_claim_days": g["claim_days"].sum() / n,
        "log_unique_benes": np.log1p(nb),
        "claims_per_bene": n / nb.clip(lower=1.0),
        "log_unique_attending": np.log1p(na),
        "claims_per_attending": (n - n_missing) / na.clip(lower=1.0),
        "mean_n_diag": g["n_diag"].sum() / n,
        "mean_dx_px_rarity": g["dx_px_rarity"].sum() / n,
        "rare_pair_share": g["rare"].sum() / n,
        "mean_age": g["age"].sum() / n,
        "mean_n_chronic": g["n_chronic"].sum() / n,
        "att_is_op_share": g["att_op"].sum() / n,
        "missing_attending_share": n_missing / n,
        "mean_amount_vs_dx": g["amount_vs_dx"].sum() / n,
        "mean_log_deductible": g["log_deductible"].sum() / n,
    })
    return out[PROVIDER_FEATURES].astype(float)


def provider_features_prefixes(claims: pd.DataFrame, feats: pd.DataFrame, rare_pair_threshold: float,
                               fractions: Iterable[float] = (0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Provider aggregates computed on the first f of each provider's claims (claims must be time-sorted).

    Training on partial histories lets the model score providers whose history is still filling up in the
    live stream, instead of only providers with a complete year of claims.
    """
    rank = claims.groupby("Provider", sort=False).cumcount()
    size = claims.groupby("Provider", sort=False)["Provider"].transform("size")
    parts = []
    for f in fractions:
        keep = (rank < np.ceil(size * f)).values
        pf = provider_features_batch(claims[keep], feats[keep], rare_pair_threshold)
        parts.append(pf.assign(fraction=float(f)))
    return pd.concat(parts)
