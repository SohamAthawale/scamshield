# xgboost MUST be imported before anything imports torch. On macOS, importing torch first and then
# training an XGBoost model in the same process segfaults (exit 139) because two OpenMP runtimes get
# loaded. Importing xgboost first avoids it. Local-only issue; the target (Linux s390x) is unaffected.
import xgboost  # noqa: F401  isort: skip

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from claimsense.data import CHRONIC_COLS, DIAG_COLS, PROC_COLS
from claimsense.features import CLAIM_FEATURES, PROVIDER_FEATURES, FeatureContext, _present

RAW_DIR = Path(__file__).resolve().parents[1] / "shared" / "data" / "raw"

# Code pools. TRAIN_DX are seen when fitting the FeatureContext; UNSEEN_DX only appear in "live" claims.
TRAIN_DX = ["4019", "25000", "V5866", "2724", "42731", "4280", "78650", "E8889", "0389", "5990"]
UNSEEN_DX = ["99999", "V9999", "E0000"]
TRAIN_PX = ["7092", "9904", "3893", "8154", "3995"]
OBJECT_COLS = (["ClaimID", "BeneID", "Provider", "AttendingPhysician", "OperatingPhysician", "OtherPhysician",
                "ClmAdmitDiagnosisCode", "DiagnosisGroupCode"] + DIAG_COLS + PROC_COLS)


def make_claims(seed: int, n: int = 300, n_providers: int = 8, n_benes: int = 60, unseen_share: float = 0.0,
                messy: bool = True) -> pd.DataFrame:
    """Synthetic claims joined with beneficiary fields, shaped like data.load_all() output.

    Dates are Timestamps/NaT, codes and physicians are str or None (object dtype), chronic flags 0/1,
    rows sorted by ClaimStartDt then ClaimID. `messy` adds negative/NaN amounts, missing dates,
    end-before-start, discharge-before-admission, deaths before service and missing attendings.
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2009-01-01")

    benes = {}
    for b in range(n_benes):
        dob = base - pd.Timedelta(days=int(rng.integers(60 * 365, 95 * 365)))
        dod = pd.NaT
        if rng.random() < 0.1:
            dod = base + pd.Timedelta(days=int(rng.integers(0, 365)))
        benes[f"BENE{11000 + b}"] = {
            "DOB": dob, "DOD": dod,
            "RenalDiseaseIndicator": int(rng.random() < 0.2),
            "NoOfMonths_PartACov": int(rng.choice([0, 6, 12, 12, 12])),
            "NoOfMonths_PartBCov": int(rng.choice([0, 6, 12, 12, 12])),
            **{c: int(rng.random() < 0.4) for c in CHRONIC_COLS},
        }
    bene_ids = list(benes)
    providers = [f"PRV{51000 + p}" for p in range(n_providers)]
    physicians = [f"PHY{300000 + i}" for i in range(25)]

    def dx_code():
        if unseen_share and rng.random() < unseen_share:
            return str(rng.choice(UNSEEN_DX))
        return str(rng.choice(TRAIN_DX))

    rows = []
    for i in range(n):
        inp = int(rng.random() < 0.3)
        start = base + pd.Timedelta(days=int(rng.integers(0, 365)))
        end = start + pd.Timedelta(days=int(rng.integers(0, 12 if inp else 2)))
        adm = dis = pd.NaT
        if inp:
            adm, dis = start, end
        if messy:
            u = rng.random()
            if u < 0.04:
                end = start - pd.Timedelta(days=int(rng.integers(1, 5)))
            elif u < 0.07:
                end = pd.NaT
            elif u < 0.09:
                start = pd.NaT
            if inp and rng.random() < 0.05:
                dis = adm - pd.Timedelta(days=2)
            if inp and rng.random() < 0.04:
                adm = pd.NaT

        amount = float(rng.choice([10, 50, 100, 300, 1000, 5000, 12000, 26000]))
        amount *= float(rng.uniform(0.5, 1.5))
        amount = round(amount, 0)
        if messy:
            u = rng.random()
            if u < 0.03:
                amount = -float(rng.integers(1, 500))
            elif u < 0.05:
                amount = float("nan")
            elif u < 0.08:
                amount = 0.0
        deductible = float(rng.choice([0, 0, 0, 10, 1068])) if not (messy and rng.random() < 0.05) else float("nan")

        att = str(rng.choice(physicians))
        if messy and rng.random() < 0.05:
            att = None
        u = rng.random()
        op = None if u < 0.6 else (att if u < 0.8 else str(rng.choice(physicians)))
        other = str(rng.choice(physicians)) if rng.random() < 0.3 else None

        n_dx = int(rng.integers(1 if not messy else 0, 11))
        dx = [dx_code() for _ in range(n_dx)] + [None] * (10 - n_dx)
        n_px = int(rng.integers(0, 4)) if inp else int(rng.random() < 0.05)
        px = [str(rng.choice(TRAIN_PX)) for _ in range(n_px)] + [None] * (6 - n_px)
        admit_dx = (dx[0] if rng.random() < 0.5 else dx_code()) if rng.random() < 0.8 else None

        b = str(rng.choice(bene_ids))
        row = {
            "ClaimID": f"CLM{100000 + i}", "BeneID": b, "Provider": str(rng.choice(providers)),
            "ClaimStartDt": start, "ClaimEndDt": end, "AdmissionDt": adm, "DischargeDt": dis,
            "InscClaimAmtReimbursed": amount, "DeductibleAmtPaid": deductible,
            "AttendingPhysician": att, "OperatingPhysician": op, "OtherPhysician": other,
            "ClmAdmitDiagnosisCode": admit_dx,
            "DiagnosisGroupCode": str(rng.integers(100, 999)) if inp else None,
            **{c: v for c, v in zip(DIAG_COLS, dx)},
            **{c: v for c, v in zip(PROC_COLS, px)},
            "is_inpatient": inp,
            **benes[b],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    for c in OBJECT_COLS:
        df[c] = df[c].astype("object").where(df[c].notna(), None)
    for c in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt", "DOB", "DOD"]:
        df[c] = pd.to_datetime(df[c])
    df = df.sort_values(["ClaimStartDt", "ClaimID"], kind="mergesort").reset_index(drop=True)
    return df


def bene_fields(row) -> dict:
    keys = ["DOB", "DOD", "RenalDiseaseIndicator", "NoOfMonths_PartACov", "NoOfMonths_PartBCov", *CHRONIC_COLS]
    return {k: row[k] for k in keys}


def attending_key(claim) -> str | None:
    v = claim.get("AttendingPhysician")
    return str(v) if _present(v) else None


@pytest.fixture(scope="session")
def train_claims() -> pd.DataFrame:
    return make_claims(seed=1, n=600, messy=False)


@pytest.fixture(scope="session")
def ctx(train_claims) -> FeatureContext:
    return FeatureContext.fit(train_claims)


@pytest.fixture
def live_claims() -> pd.DataFrame:
    """Messy claims with some diagnosis codes the context has never seen."""
    return make_claims(seed=7, n=400, unseen_share=0.1)


# ------------------------------------------------------------------------------- tiny model artifacts
def _quantiles(rng, n=101) -> list[float]:
    return sorted(float(v) for v in np.quantile(rng.normal(size=500), np.linspace(0, 1, n)))


def build_tiny_models(root: Path, seed: int = 0) -> Path:
    """Randomly initialised autoencoder (torch state_dict + npz + scaler) and a tiny XGBoost provider model."""
    import torch
    import xgboost as xgb

    from claimsense.features import claim_features_batch
    from claimsense.models import Artifacts, Standardizer, build_torch_autoencoder, torch_to_numpy_weights

    root.mkdir(parents=True, exist_ok=True)
    art = Artifacts(root)
    rng = np.random.default_rng(seed)

    # scaler fitted on synthetic claim features so standardised values are in a sane range
    tr = make_claims(seed=1, n=600, messy=False)
    fctx = FeatureContext.fit(tr)
    X = claim_features_batch(tr, fctx)[CLAIM_FEATURES].to_numpy(float)
    Standardizer.fit(CLAIM_FEATURES, X).save(art.scaler)

    torch.manual_seed(seed)
    model = build_torch_autoencoder(len(CLAIM_FEATURES))
    torch.save(model.state_dict(), art.ae_torch)
    np.savez(art.ae_numpy, **torch_to_numpy_weights(model))

    n = 300
    P = rng.normal(size=(n, len(PROVIDER_FEATURES))).astype(np.float32)
    y = (P[:, 1] + 0.5 * P[:, 3] + rng.normal(scale=0.5, size=n) > 0.5).astype(int)
    dtrain = xgb.DMatrix(P, label=y, feature_names=PROVIDER_FEATURES)
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 3, "eta": 0.5, "nthread": 1},
                        dtrain, num_boost_round=5)
    booster.save_model(str(art.provider_model))
    art.provider_meta.write_text(json.dumps({
        "features": PROVIDER_FEATURES,
        "peer_quantiles": {f: _quantiles(rng) for f in PROVIDER_FEATURES},
    }))

    art.feature_context.write_text(fctx.to_json())
    art.thresholds.write_text(json.dumps({"anomaly_threshold": 0.5, "provider_threshold": 0.6,
                                          "min_provider_claims": 10}))
    return root


@pytest.fixture(scope="session")
def tiny_models_dir(tmp_path_factory) -> Path:
    """Session-wide read-only artifacts. Copy before modifying (see `fresh_models_dir`)."""
    return build_tiny_models(tmp_path_factory.mktemp("models"))


@pytest.fixture
def fresh_models_dir(tmp_path, tiny_models_dir) -> Path:
    """A per-test copy of the tiny artifacts that a test may delete files from."""
    import shutil

    dst = tmp_path / "models"
    shutil.copytree(tiny_models_dir, dst)
    return dst


def assert_close(a, b, atol=1e-9, rtol=1e-9):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def isclose(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
