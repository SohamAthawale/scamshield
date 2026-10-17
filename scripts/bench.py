"""Inference benchmarks on the VM: claim autoencoder on Telum II NNPA vs CPU, and XGBoost provider scoring.

Runs inside the app image with the trained models, on REAL claim features (held-out test claims):

    docker run --rm -v ~/shared/models:/models:ro -v ~/shared/data:/data-in:ro -v ~/claimsense/docs/evidence/bench:/out \
        --entrypoint python claimsense-app:latest /app/scripts/bench.py

Writes bench_autoencoder.csv, bench_provider.csv, and bench_autoencoder.png to /out.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from claimsense.features import CLAIM_FEATURES, PROVIDER_FEATURES
from claimsense.inference.anomaly import TorchAnomalyScorer
from claimsense.inference.provider import XGBProviderScorer
from claimsense.models import Artifacts

MODELS = Path(os.environ.get("CLAIMSENSE_MODELS", "/models"))
OUT = Path(os.environ.get("BENCH_OUT", "/out"))
SEED = 0


def timeit(fn, n: int, warmup: int = 20) -> np.ndarray:
    for _ in range(warmup):
        fn()
    ts = np.empty(n)
    for i in range(n):
        t = time.perf_counter()
        fn()
        ts[i] = time.perf_counter() - t
    return ts * 1e6  # microseconds


def real_claim_features(n: int) -> np.ndarray:
    """Rebuild features for held-out claims with the training FeatureContext (no labels involved)."""
    from claimsense.data import load_beneficiaries
    from claimsense.features import FeatureContext, claim_features_batch

    ctx = FeatureContext.from_json((MODELS / "feature_context.json").read_text())
    s = pd.read_csv(os.environ.get("STREAM_FILE", "/data-in/stream_test.csv"), nrows=n, dtype={
        c: str for c in ["ClmAdmitDiagnosisCode", "DiagnosisGroupCode"] + [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
        + [f"ClmProcedureCode_{i}" for i in range(1, 7)]})
    b = load_beneficiaries(os.environ.get("CLAIMSENSE_RAW", "/data-in/raw"))
    s = s.merge(b, on="BeneID", how="left")
    return claim_features_batch(s, ctx).values


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    art = Artifacts(MODELS)
    X = real_claim_features(20000)
    ae = TorchAnomalyScorer(art, use_nnpa=True, nnpa_min_batch=1)
    Zall = ae.scaler.transform(X)

    rows = []
    for bs, n in [(1, 5000), (8, 5000), (32, 3000), (64, 3000), (256, 2000), (1024, 1000), (4096, 300), (16384, 60)]:
        idx = rng.integers(0, len(Zall), size=bs)
        Z = Zall[idx]
        r_cpu = ae.reconstruct(Z, force="cpu")
        r_nnpa = ae.reconstruct(Z, force="nnpa")
        maxdiff = float(np.abs(r_cpu - r_nnpa).max())
        e_cpu, e_nnpa = ((Z - r_cpu) ** 2).mean(1), ((Z - r_nnpa) ** 2).mean(1)
        for dev in ("cpu", "nnpa"):
            t = timeit(lambda: ae.reconstruct(Z, force=dev), n)
            rows.append({"model": "claim_autoencoder", "device": dev, "batch": bs, "iterations": n,
                         "p50_us": round(float(np.percentile(t, 50)), 1), "p99_us": round(float(np.percentile(t, 99)), 1),
                         "claims_per_sec": round(bs / (np.median(t) / 1e6)), "max_abs_diff_vs_cpu": maxdiff,
                         "max_score_diff_vs_cpu": float(np.abs(e_cpu - e_nnpa).max())})
            print(json.dumps(rows[-1]), flush=True)
    ae_df = pd.DataFrame(rows)
    ae_df.to_csv(OUT / "bench_autoencoder.csv", index=False)

    prov = XGBProviderScorer(art)
    P = rng.random((4096, len(PROVIDER_FEATURES)), dtype=np.float32)
    prows = []
    for bs, n in [(1, 5000), (32, 3000), (512, 500), (4096, 100)]:
        t = timeit(lambda: prov.predict(P[:bs]), n)
        prows.append({"model": "provider_xgboost", "device": "cpu", "batch": bs, "iterations": n,
                      "p50_us": round(float(np.percentile(t, 50)), 1), "p99_us": round(float(np.percentile(t, 99)), 1),
                      "providers_per_sec": round(bs / (np.median(t) / 1e6))})
        print(json.dumps(prows[-1]), flush=True)
    t = timeit(lambda: prov.contributions(P[0]), 500)
    prows.append({"model": "provider_xgboost_shap", "device": "cpu", "batch": 1, "iterations": 500,
                  "p50_us": round(float(np.percentile(t, 50)), 1), "p99_us": round(float(np.percentile(t, 99)), 1)})
    pd.DataFrame(prows).to_csv(OUT / "bench_provider.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for dev, color in (("cpu", "#6f6f6f"), ("nnpa", "#0f62fe")):
        d = ae_df[ae_df.device == dev]
        ax[0].plot(d.batch, d.claims_per_sec, "o-", color=color, label="Telum II NNPA" if dev == "nnpa" else "CPU")
        ax[1].plot(d.batch, d.p50_us, "o-", color=color, label="Telum II NNPA" if dev == "nnpa" else "CPU")
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xlabel("claims per batch")
        a.grid(alpha=.3)
        a.legend()
    ax[0].set_yscale("log")
    ax[0].set_ylabel("claims scored per second")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("p50 time per batch (µs)")
    fig.suptitle(f"Claim autoencoder on IBM z17 ({platform.machine()}): CPU vs on-chip AI accelerator")
    fig.tight_layout()
    fig.savefig(OUT / "bench_autoencoder.png", dpi=140)
    print("saved", OUT)


if __name__ == "__main__":
    main()
