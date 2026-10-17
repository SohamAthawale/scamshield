"""Training pipeline: split -> features -> provider XGBoost -> claim autoencoder -> thresholds -> stream file.

The notebooks call these functions step by step (and draw the charts); the CLI runs them end to end:

    python -m claimsense.training --raw shared/data/raw --models shared/models --data shared/data
"""
from __future__ import annotations

# xgboost must be imported before torch: on macOS both ship their own OpenMP runtime and importing
# torch first makes xgboost training segfault. Harmless on Linux/s390x.
import xgboost  # noqa: F401  isort: skip

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .features import (CLAIM_FEATURES, PROVIDER_FEATURES, FeatureContext, claim_features_batch,
                       hard_errors_batch, provider_features_batch, provider_features_prefixes)
from .models import Artifacts, Standardizer, build_torch_autoencoder, torch_to_numpy_weights

log = logging.getLogger("claimsense.training")

PREFIX_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
STREAM_COLUMNS = [
    "ClaimID", "BeneID", "Provider", "ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt",
    "InscClaimAmtReimbursed", "DeductibleAmtPaid", "AttendingPhysician", "OperatingPhysician",
    "OtherPhysician", "ClmAdmitDiagnosisCode", "DiagnosisGroupCode", *D.DIAG_COLS, *D.PROC_COLS, "is_inpatient",
]


# ------------------------------------------------------------------------------ metrics
def ranking_metrics(y: np.ndarray, p: np.ndarray, top_frac: float = 0.05) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(top_frac * len(y))))
    top = np.argsort(-p)[:k]
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        f"precision_top{int(top_frac * 100)}pct": float(y[top].mean()),
        f"recall_top{int(top_frac * 100)}pct": float(y[top].sum() / max(y.sum(), 1)),
        "base_rate": float(y.mean()),
    }


# ------------------------------------------------------------------------------ steps
def prepare(raw_dir: str | Path, seed: int = 42):
    claims, bene, labels = D.load_all(str(raw_dir))
    splits = D.split_providers(labels, seed=seed)
    claims = claims.merge(splits[["Provider", "split", "label"]], on="Provider", how="left", validate="many_to_one")
    train_claims = claims[claims["split"] == "train"]
    ctx = FeatureContext.fit(train_claims)
    feats = claim_features_batch(claims, ctx)
    return claims, bene, splits, ctx, feats


def train_provider_model(claims: pd.DataFrame, feats: pd.DataFrame, ctx: FeatureContext, seed: int = 42):
    import xgboost as xgb

    def agg(split: str, fractions=PREFIX_FRACTIONS) -> pd.DataFrame:
        m = (claims["split"] == split).values
        pf = provider_features_prefixes(claims[m], feats[m], ctx.rare_pair_threshold, fractions)
        lab = claims[m].groupby("Provider")["label"].first()
        return pf.join(lab)

    tr, va = agg("train"), agg("val", (1.0,))
    pos, neg = tr["label"].sum(), (1 - tr["label"]).sum()
    model = xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=4, min_child_weight=2, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=float(neg / max(pos, 1)),
        eval_metric="aucpr", early_stopping_rounds=60, random_state=seed, n_jobs=0, tree_method="hist",
    )
    model.fit(tr[PROVIDER_FEATURES], tr["label"], eval_set=[(va[PROVIDER_FEATURES], va["label"])], verbose=False)

    full_train = provider_features_batch(claims[claims["split"] == "train"], feats[claims["split"] == "train"],
                                         ctx.rare_pair_threshold)
    peer_q = {f: np.quantile(full_train[f].values, np.linspace(0, 1, 101)).tolist() for f in PROVIDER_FEATURES}
    imp = model.get_booster().get_score(importance_type="gain")
    meta = {
        "features": PROVIDER_FEATURES,
        "peer_quantiles": peer_q,
        "importance_gain": {f: float(imp.get(f, 0.0)) for f in PROVIDER_FEATURES},
        "best_iteration": int(model.best_iteration),
        "scale_pos_weight": float(neg / max(pos, 1)),
        "trained_on": "train-provider aggregates at history fractions " + str(PREFIX_FRACTIONS),
    }
    metrics = {}
    for split in ("val", "test"):
        for frac in (0.25, 1.0):
            d = agg(split, (frac,))
            p = model.predict_proba(d[PROVIDER_FEATURES])[:, 1]
            metrics[f"{split}_history{int(frac * 100)}"] = ranking_metrics(d["label"].values, p)
    return model, meta, metrics


def train_autoencoder(claims: pd.DataFrame, feats: pd.DataFrame, epochs: int = 8, batch_size: int = 1024,
                      lr: float = 1e-3, seed: int = 42, device: str = "cpu"):
    """Train on claims from NON-flagged training providers only, so 'normal' means normal billing."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    clean = ((claims["split"] == "train") & (claims["label"] == 0)).values
    X = feats.values[clean]
    scaler = Standardizer.fit(CLAIM_FEATURES, X)
    Z = torch.from_numpy(scaler.transform(X))
    model = build_torch_autoencoder(len(CLAIM_FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = []
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Z), generator=gen)
        total = 0.0
        for i in range(0, len(Z), batch_size):
            xb = Z[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(xb)
        history.append(total / len(Z))
        log.info("autoencoder epoch %d loss %.5f", epoch + 1, history[-1])
    model.eval()
    return model, scaler, history


def reconstruction_errors(model, scaler: Standardizer, X: np.ndarray, batch: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    import torch

    Z = scaler.transform(X)
    out = np.empty_like(Z)
    with torch.no_grad():
        for i in range(0, len(Z), batch):
            out[i:i + batch] = model(torch.from_numpy(Z[i:i + batch])).numpy()
    per_feature = (Z - out) ** 2
    return per_feature.mean(axis=1), per_feature


def simulate_stream(claims: pd.DataFrame, bene: pd.DataFrame, models_dir: Path, anomaly_threshold: float,
                    provider_threshold: float, min_provider_claims: int, batch_size: int = 512) -> pd.DataFrame:
    """Replay a split's claims through the real Engine (same code as the API). Returns one row per claim."""
    from .fusion import Policy
    from .inference.anomaly import load_anomaly_scorer
    from .inference.provider import load_provider_scorer
    from .scoring import ClaimInput, Engine

    art = Artifacts(models_dir)
    eng = Engine(FeatureContext.from_json(art.feature_context.read_text()),
                 load_anomaly_scorer(models_dir, "numpy"), load_provider_scorer(models_dir),
                 Policy(anomaly_threshold, provider_threshold, min_provider_claims))
    rows = claims.to_dict("records")
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        items = [ClaimInput(r, r, r["Provider"], r["BeneID"],
                            r["AttendingPhysician"] if isinstance(r["AttendingPhysician"], str) else None)
                 for r in chunk]
        for r, res in zip(chunk, eng.score_batch(items)):
            d = res.decision
            out.append((r["ClaimID"], r["Provider"], r["label"], r["InscClaimAmtReimbursed"], d.action,
                        d.anomaly, d.provider_risk, res.provider_claims, ",".join(d.hard_errors)))
    return pd.DataFrame(out, columns=["ClaimID", "Provider", "label", "amount", "action", "anomaly",
                                      "provider_risk", "provider_claims", "rules"])


def hold_curve(sim: pd.DataFrame, anomaly_threshold: float, min_provider_claims: int) -> pd.DataFrame:
    """For each provider threshold: share of claims held and precision (held claims from flagged providers)."""
    flagged = (sim["anomaly"] >= anomaly_threshold) | (sim["rules"].fillna("") != "")
    ready = sim["provider_claims"] >= min_provider_claims
    rows = []
    for t in np.round(np.arange(0.30, 1.0, 0.01), 2):
        held = flagged & ready & (sim["provider_risk"] >= t)
        rows.append((t, float(held.mean()), float(sim["label"][held].mean()) if held.any() else np.nan,
                     int(held.sum())))
    return pd.DataFrame(rows, columns=["threshold", "hold_share", "precision", "held"])


def tune_provider_threshold(sim: pd.DataFrame, anomaly_threshold: float, min_provider_claims: int,
                            hold_cap: float = 0.02, min_precision: float = 0.70) -> float:
    """Lowest threshold whose VALIDATION hold precision reaches min_precision while holding at most
    hold_cap of claims. Lower = more claims caught, so we take the lowest one that meets both.

    Holding a payment is costly for an honest provider, so precision is the binding constraint and the
    cap is a safety limit on investigator workload.
    """
    curve = hold_curve(sim, anomaly_threshold, min_provider_claims)
    ok = curve[(curve["precision"] >= min_precision) & (curve["hold_share"] <= hold_cap)]
    return float(ok["threshold"].min()) if len(ok) else 0.95


def decision_summary(sim: pd.DataFrame) -> dict:
    mix = sim["action"].value_counts(normalize=True).to_dict()
    held = sim[sim["action"] == "HOLD_FOR_REVIEW"]
    returned = sim[sim["action"] == "RETURN_WITH_FIX"]
    return {
        "claims": int(len(sim)),
        "mix": {k: round(float(v), 4) for k, v in mix.items()},
        "hold_precision_flagged_provider": float(held["label"].mean()) if len(held) else None,
        "return_share_from_flagged_provider": float(returned["label"].mean()) if len(returned) else None,
        "dollars_held": float(held["amount"].sum()),
        "base_rate_flagged_provider_claims": float(sim["label"].mean()),
    }


def write_stream(claims: pd.DataFrame, out_dir: Path) -> Path:
    """Test-provider claims in time order: the replay stream for the demo (no labels in it)."""
    s = claims[claims["split"] == "test"][STREAM_COLUMNS].copy()
    for c in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]:
        s[c] = s[c].dt.strftime("%Y-%m-%d")
    path = out_dir / "stream_test.csv"
    s.to_csv(path, index=False)
    return path


# ------------------------------------------------------------------------------ CLI
def run(raw: Path, models: Path, data_dir: Path, epochs: int, seed: int, min_provider_claims: int,
        hold_target: float, anomaly_quantile: float, min_precision: float = 0.70) -> dict:
    import torch
    import xgboost as xgb

    t0 = time.time()
    models.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    art = Artifacts(models)

    claims, bene, splits, ctx, feats = prepare(raw, seed)
    splits.to_csv(data_dir / "splits.csv", index=False)
    art.feature_context.write_text(ctx.to_json())
    log.info("prepared %d claims in %.1fs", len(claims), time.time() - t0)

    pmodel, meta, pmetrics = train_provider_model(claims, feats, ctx, seed)
    pmodel.get_booster().save_model(str(art.provider_model))
    art.provider_meta.write_text(json.dumps(meta))
    log.info("provider model: %s", json.dumps(pmetrics["val_history100"]))

    ae, scaler, history = train_autoencoder(claims, feats, epochs=epochs, seed=seed)
    torch.save(ae.state_dict(), art.ae_torch)
    np.savez(art.ae_numpy, **torch_to_numpy_weights(ae))
    scaler.save(art.scaler)
    try:
        torch.onnx.export(ae, torch.zeros(1, len(CLAIM_FEATURES)), str(art.ae_onnx), input_names=["z"],
                          output_names=["recon"], dynamic_axes={"z": {0: "batch"}, "recon": {0: "batch"}},
                          opset_version=17, dynamo=False)
    except Exception as e:  # ONNX is only for the optional zDLC path
        log.warning("onnx export skipped: %r", e)

    val_clean = ((claims["split"] == "val") & (claims["label"] == 0)).values
    err_val, _ = reconstruction_errors(ae, scaler, feats.values[val_clean])
    anomaly_threshold = float(np.quantile(err_val, anomaly_quantile))

    test_m = (claims["split"] == "test").values
    err_test, _ = reconstruction_errors(ae, scaler, feats.values[test_m])
    from sklearn.metrics import roc_auc_score

    rules_test = hard_errors_batch(claims[test_m])
    ae_metrics = {
        "train_loss_by_epoch": history,
        "threshold_quantile": anomaly_quantile,
        "anomaly_threshold": anomaly_threshold,
        "flag_rate_test": float((err_test >= anomaly_threshold).mean()),
        "claim_auc_vs_provider_label_test": float(roc_auc_score(claims.loc[test_m, "label"], err_test)),
        "rule_hits_test": {k: int(v) for k, v in rules_test.sum().items()},
    }

    # Tune the provider threshold on the VALIDATION stream, replayed through the real engine.
    provisional = {"anomaly_threshold": anomaly_threshold, "provider_threshold": 0.5,
                   "min_provider_claims": min_provider_claims}
    art.thresholds.write_text(json.dumps(provisional))
    val_claims = claims[claims["split"] == "val"]
    sim_val = simulate_stream(val_claims, bene, models, anomaly_threshold, 0.5, min_provider_claims)
    provider_threshold = tune_provider_threshold(sim_val, anomaly_threshold, min_provider_claims, hold_target,
                                                 min_precision)
    hold_curve(sim_val, anomaly_threshold, min_provider_claims).to_csv(data_dir / "hold_curve_val.csv", index=False)
    thresholds = {"anomaly_threshold": anomaly_threshold, "provider_threshold": provider_threshold,
                  "min_provider_claims": min_provider_claims, "hold_cap": hold_target,
                  "min_hold_precision_val": min_precision}
    art.thresholds.write_text(json.dumps(thresholds, indent=1))
    sim_val = simulate_stream(val_claims, bene, models, anomaly_threshold, provider_threshold, min_provider_claims)

    sim_test = simulate_stream(claims[test_m], bene, models, anomaly_threshold, provider_threshold,
                               min_provider_claims)
    stream_path = write_stream(claims, data_dir)

    metrics = {
        "data": {"claims": int(len(claims)), "providers": int(len(splits)),
                 "providers_by_split": splits.groupby("split")["label"].agg(["size", "sum"]).to_dict()},
        "provider_model": pmetrics,
        "autoencoder": ae_metrics,
        "thresholds": thresholds,
        "decisions_val": decision_summary(sim_val),
        "decisions_test": decision_summary(sim_test),
        "versions": {"xgboost": xgb.__version__, "torch": torch.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__},
        "seconds": round(time.time() - t0, 1),
        "stream_file": str(stream_path),
    }
    art.metrics.write_text(json.dumps(metrics, indent=1, default=str))
    sim_test.to_csv(data_dir / "sim_test_decisions.csv", index=False)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("shared/data/raw"))
    ap.add_argument("--models", type=Path, default=Path("shared/models"))
    ap.add_argument("--data", type=Path, default=Path("shared/data"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-provider-claims", type=int, default=10)
    ap.add_argument("--hold-target", type=float, default=0.02)
    ap.add_argument("--anomaly-quantile", type=float, default=0.99)
    ap.add_argument("--min-hold-precision", type=float, default=0.70)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    m = run(a.raw, a.models, a.data, a.epochs, a.seed, a.min_provider_claims, a.hold_target, a.anomaly_quantile,
            a.min_hold_precision)
    print(json.dumps({k: m[k] for k in ("provider_model", "thresholds", "decisions_test", "seconds")}, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
