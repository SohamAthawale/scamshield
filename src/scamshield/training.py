"""Training: time split -> streaming features -> XGBoost risk -> payment autoencoder -> thresholds -> stream file.

    python -m scamshield.training --raw shared/data/aml --models shared/models --data shared/data

The notebooks call the same functions step by step.
"""
from __future__ import annotations

# xgboost before torch: on macOS both ship OpenMP and torch-first makes xgboost training crash. Harmless on s390x.
import xgboost  # noqa: F401  isort: skip

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .features import ANOMALY_FEATURES, FEATURES, FeatureContext, FeatureEngine, rule_hits
from .fusion import ALLOW, HOLD, WARN, Policy, decide
from .models import Artifacts, Standardizer, build_torch_autoencoder, torch_to_numpy_weights

log = logging.getLogger("scamshield.training")
AE_IDX = [FEATURES.index(f) for f in ANOMALY_FEATURES]


# ------------------------------------------------------------------------------ metrics
def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    out = {"n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()),
           "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}
    # best F1 over thresholds (the usual minority-class metric for this dataset)
    tp = np.cumsum(y[order])
    prec, rec = tp / np.arange(1, len(y) + 1), tp / max(y.sum(), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    k = int(np.argmax(f1))
    out.update(best_f1=float(f1[k]), best_f1_precision=float(prec[k]), best_f1_recall=float(rec[k]),
               best_f1_threshold=float(p[order][k]))
    return out


# ------------------------------------------------------------------------------ steps
def prepare(raw_dir: str | Path):
    tx = D.load_transactions(str(raw_dir))
    split = D.time_split(tx["ts"])
    ctx = FeatureContext.fit(tx[split == "train"])
    t = time.time()
    X = FeatureEngine(ctx).run(tx)
    log.info("features for %d payments in %.0fs", len(tx), time.time() - t)
    return tx, split, ctx, X


def train_risk_model(X: np.ndarray, y: np.ndarray, split: np.ndarray, neg_sample: float = 0.25, seed: int = 42):
    """XGBoost on all features. Negatives are subsampled (and up-weighted) so training fits a 3-vCPU VM."""
    import xgboost as xgb

    rng = np.random.default_rng(seed)
    tr = (split == "train") & ((y == 1) | (rng.random(len(y)) < neg_sample))
    va = split == "val"
    w = np.where(y[tr] == 1, 1.0, 1.0 / neg_sample)
    pos, neg = y[tr].sum(), w[y[tr] == 0].sum()
    model = xgb.XGBClassifier(
        n_estimators=1500, learning_rate=0.05, max_depth=7, min_child_weight=1, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=float(neg / max(pos, 1)) ** 0.5,
        eval_metric="aucpr", early_stopping_rounds=100, random_state=seed, n_jobs=0, tree_method="hist",
    )
    model.fit(X[tr], y[tr], sample_weight=w, eval_set=[(X[va], y[va])], verbose=False)
    # Keep ONLY the trees up to the best iteration. The full booster also holds the early-stopping
    # patience rounds; scoring with it would not be the model the thresholds were tuned on.
    booster = model.get_booster()[: model.best_iteration + 1]
    booster.feature_names = FEATURES
    imp = booster.get_score(importance_type="gain")
    meta = {"features": FEATURES, "importance_gain": {f: float(imp.get(f, 0.0)) for f in FEATURES},
            "best_iteration": int(model.best_iteration), "trees": int(model.best_iteration + 1),
            "negative_sample": neg_sample}
    return booster, meta


def train_autoencoder(X: np.ndarray, y: np.ndarray, split: np.ndarray, epochs: int = 6, batch_size: int = 2048,
                      lr: float = 1e-3, max_rows: int = 600_000, seed: int = 42):
    """'Is this payment unusual for this customer?' Trained on normal training payments only."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero((split == "train") & (y == 0))
    if len(idx) > max_rows:
        idx = rng.choice(idx, max_rows, replace=False)
    Xa = X[idx][:, AE_IDX]
    scaler = Standardizer.fit(ANOMALY_FEATURES, Xa)
    Z = torch.from_numpy(scaler.transform(Xa))
    model = build_torch_autoencoder(len(ANOMALY_FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history, gen = [], torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        perm, total = torch.randperm(len(Z), generator=gen), 0.0
        for i in range(0, len(Z), batch_size):
            xb = Z[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(xb)
        history.append(total / len(Z))
        log.info("autoencoder epoch %d loss %.5f", epoch + 1, history[-1])
    return model.eval(), scaler, history


def reconstruction_errors(model, scaler: Standardizer, X: np.ndarray, batch: int = 65536) -> np.ndarray:
    import torch

    Z = scaler.transform(X[:, AE_IDX])
    out = np.empty(len(Z))
    with torch.no_grad():
        for i in range(0, len(Z), batch):
            z = torch.from_numpy(Z[i:i + batch])
            out[i:i + batch] = ((model(z) - z) ** 2).mean(1).numpy()
    return out


def rule_matrix(X: np.ndarray, ctx: FeatureContext) -> np.ndarray:
    """Vectorised rule_hits -> boolean 'any rule fired' per payment."""
    f = {c: X[:, i] for i, c in enumerate(FEATURES)}
    return (f["s_new_payee"] >= .5) & (f["s_log_n_out"] >= np.log1p(5)) & \
        (f["s_amount_vs_usual"] >= np.log(ctx.rule_big_new_payee_ratio))


def decisions(risk: np.ndarray, anomaly: np.ndarray, rules: np.ndarray, pol: Policy,
              usd: np.ndarray | None = None) -> np.ndarray:
    """Vectorised fusion.decide (tests check they agree)."""
    if pol.value_aware:
        if usd is None:
            raise ValueError("a value-aware policy needs the payments' USD amounts")
        hold = (risk >= pol.hold_strict_threshold) | ((risk >= pol.hold_threshold) & (usd >= pol.hold_min_usd))
    else:
        hold = risk >= pol.hold_threshold
    warn = ~hold & ((risk >= pol.warn_threshold) | (anomaly >= pol.anomaly_threshold) | rules)
    return np.where(hold, HOLD, np.where(warn, WARN, ALLOW))


def tune_hold_threshold(y: np.ndarray, risk: np.ndarray, min_precision: float = 0.6, cap: float = 0.002) -> float:
    """Lowest risk threshold whose VALIDATION hold precision >= min_precision and hold share <= cap."""
    order = np.argsort(-risk, kind="mergesort")
    r = risk[order]
    tp = np.cumsum(y[order])
    k = np.arange(1, len(y) + 1)
    # a threshold t holds EVERY payment with risk >= t, so only positions at the end of a run of tied
    # scores are achievable cut points
    end_of_tie = np.r_[r[1:] != r[:-1], True]
    ok = end_of_tie & (tp / k >= min_precision) & (k / len(y) <= cap)
    if not ok.any():
        return float(r[0]) + 1e-9
    kmax = int(np.flatnonzero(ok).max())
    return float(r[kmax])


# candidate minimum amounts for the value-aware hold: a coarse 1-2-5 grid, so the choice cannot be fitted
# to individual validation payments
HOLD_AMOUNT_GRID = (1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7)


def tune_value_hold(y: np.ndarray, risk: np.ndarray, usd: np.ndarray, hold_threshold: float,
                    min_precision: float = 0.8, keep_usd: float = 0.95,
                    amounts: tuple[float, ...] = HOLD_AMOUNT_GRID) -> dict:
    """Pick (hold_strict_threshold, hold_min_usd) on VALIDATION payments.

    Hold if risk >= strict, or if hold_threshold <= risk < strict and usd >= min_usd. The held set is a subset
    of today's (risk >= hold_threshold), so the volume cap still holds. Among choices with hold precision
    >= min_precision that keep >= keep_usd of the laundered USD today's policy pauses, take the one holding
    the most laundering payments; ties -> fewer holds, then the larger minimum amount, then the higher strict
    threshold. Returns both settings as None (plain policy) if nothing qualifies.
    """
    y, risk, usd = np.asarray(y), np.asarray(risk, dtype=np.float64), np.asarray(usd, dtype=np.float64)
    band = risk >= hold_threshold
    order = np.argsort(-risk[band], kind="mergesort")
    r, yy, uu = risk[band][order], y[band][order].astype(np.int64), usd[band][order]
    n = len(r)
    base = {"holds": int(n), "laundering_held": int(yy.sum()), "laundered_usd_held": float((uu * yy).sum())}
    out = {"hold_strict_threshold": None, "hold_min_usd": None, "min_precision": min_precision,
           "keep_usd": keep_usd, "val_base": base, "val_chosen": None, "feasible_choices": 0}
    if n == 0:
        return out
    # strict set = the first k payments (k = 0 .. n); only ends of tied runs are achievable cut points
    ks = np.r_[0, np.flatnonzero(np.r_[r[1:] != r[:-1], True]) + 1]
    pre_tp = np.r_[0, np.cumsum(yy)]
    pre_usd = np.r_[0.0, np.cumsum(uu * yy)]

    def suffix(v):  # suffix[k] = sum of v[k:]
        return np.r_[np.cumsum(v[::-1])[::-1], 0]

    best, best_key, feasible = None, None, 0
    for a in amounts:
        m = uu >= a
        cnt = ks + suffix(m.astype(np.int64))[ks]
        tp = pre_tp[ks] + suffix((m & (yy == 1)).astype(np.int64))[ks]
        lusd = pre_usd[ks] + suffix(np.where(m, uu * yy, 0.0))[ks]
        ok = (cnt > 0) & (tp >= min_precision * np.maximum(cnt, 1)) & (lusd >= keep_usd * base["laundered_usd_held"])
        feasible += int(ok.sum())
        for j in np.flatnonzero(ok):
            key = (int(tp[j]), -int(cnt[j]), a, -int(ks[j]))
            if best_key is None or key > best_key:
                best_key, best = key, (int(ks[j]), a, int(cnt[j]), int(tp[j]), float(lusd[j]))
    out["feasible_choices"] = feasible
    if best is None:
        return out
    k, a, cnt, tp, lusd = best
    # k == n: every payment in the band is held on risk alone, i.e. today's policy
    strict = float(r[k - 1]) if 0 < k < n else (float(hold_threshold) if k == n else float(r[0]) + 1e-9)
    out.update(hold_strict_threshold=strict, hold_min_usd=float(a),
               val_chosen={"holds": cnt, "laundering_held": tp, "laundered_usd_held": lusd,
                           "hold_precision": tp / cnt, "usd_kept_vs_base": lusd / max(base["laundered_usd_held"], 1e-12),
                           "strict_holds": k})
    return out


def summarise(y: np.ndarray, action: np.ndarray, usd: np.ndarray, patterns_hit: pd.DataFrame | None = None) -> dict:
    held, warned = action == HOLD, action == WARN
    flagged = held | warned
    out = {
        "payments": int(len(y)),
        "mix": {a: round(float((action == a).mean()), 5) for a in (ALLOW, WARN, HOLD)},
        "hold_precision": float(y[held].mean()) if held.any() else None,
        "warn_precision": float(y[warned].mean()) if warned.any() else None,
        "laundering_recall_hold": float(y[held].sum() / max(y.sum(), 1)),
        "laundering_recall_warn_or_hold": float(y[flagged].sum() / max(y.sum(), 1)),
        "laundered_usd_total": float(usd[y == 1].sum()),
        "laundered_usd_held": float(usd[(y == 1) & held].sum()),
        "holds": int(held.sum()),
        "laundering_held": int(y[held].sum()),
        "laundering_warned_or_held": int(y[flagged].sum()),
        "base_rate": float(y.mean()),
    }
    if patterns_hit is not None and len(patterns_hit):
        by = patterns_hit.groupby("pattern")["hit"].agg(["size", "sum"])
        out["schemes_disrupted"] = {k: f"{int(r['sum'])}/{int(r['size'])}" for k, r in by.iterrows()}
        out["schemes_disrupted_total"] = f"{int(by['sum'].sum())}/{int(by['size'].sum())}"
    return out


def scheme_hits(tx: pd.DataFrame, mask: np.ndarray, action: np.ndarray, raw_dir: str | Path) -> pd.DataFrame:
    """Per laundering attempt in the period: was ANY of its payments held (the chain is broken)?"""
    pat = D.laundering_patterns(str(raw_dir))
    pat["ts"] = pd.to_datetime(pat["ts"], format="%Y/%m/%d %H:%M")
    sub = tx[mask][["ts", "sender", "receiver", "amount_paid"]].copy()
    sub["held"] = action == HOLD
    m = pat.merge(sub, on=["ts", "sender", "receiver", "amount_paid"], how="inner")
    return m.groupby(["attempt", "pattern"])["held"].any().rename("hit").reset_index()


HISTORY_COLUMNS = ["ts", "from_bank", "sender", "to_bank", "receiver", "amount_paid", "currency_paid",
                   "currency_received", "payment_format"]


def write_history(tx: pd.DataFrame, split: np.ndarray, out_dir: Path) -> Path:
    """Payments before the live period, in the exact order training used: the API builds its rolling state
    from this file at start-up (no labels, no scores)."""
    h = tx.loc[split != "test", HISTORY_COLUMNS].copy()
    h["ts"] = h["ts"].dt.strftime("%Y-%m-%d %H:%M")
    path = out_dir / "history.csv"
    h.to_csv(path, index=False)
    return path


def write_stream(tx: pd.DataFrame, split: np.ndarray, out_dir: Path) -> Path:
    """Test-period payments in time order: the live replay for the demo (no labels)."""
    s = tx[split == "test"].drop(columns=["label"]).copy()
    s["ts"] = s["ts"].dt.strftime("%Y-%m-%d %H:%M")
    path = out_dir / "stream_test.csv"
    s.to_csv(path, index=False)
    return path


# ------------------------------------------------------------------------------ CLI
def run(raw: Path, models: Path, data_dir: Path, epochs: int, seed: int, hold_precision: float, hold_cap: float,
        warn_budget: float, anomaly_quantile: float, value_hold_precision: float | None = 0.8,
        value_hold_keep_usd: float = 0.95) -> dict:
    import torch
    import xgboost as xgb

    t0 = time.time()
    models.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    art = Artifacts(models)

    tx, split, ctx, X = prepare(raw)
    y = tx["label"].values.astype(int)
    usd = tx["amount_paid"].values * tx["currency_paid"].map(ctx.usd_per_unit).values
    art.feature_context.write_text(ctx.to_json())

    booster, meta = train_risk_model(X, y, split, seed=seed)
    booster.save_model(str(art.risk_model))
    art.risk_meta.write_text(json.dumps(meta))
    risk = booster.inplace_predict(X).astype(np.float64)  # exactly what the API computes
    risk_metrics = {s: ranking_metrics(y[split == s], risk[split == s]) for s in ("val", "test")}
    log.info("risk model val: %s", json.dumps(risk_metrics["val"]))

    ae, scaler, history = train_autoencoder(X, y, split, epochs=epochs, seed=seed)
    torch.save(ae.state_dict(), art.ae_torch)
    np.savez(art.ae_numpy, **torch_to_numpy_weights(ae))
    scaler.save(art.scaler)
    try:
        torch.onnx.export(ae, torch.zeros(1, len(ANOMALY_FEATURES)), str(art.ae_onnx), input_names=["z"],
                          output_names=["recon"], dynamic_axes={"z": {0: "batch"}, "recon": {0: "batch"}},
                          opset_version=17, dynamo=False)
    except Exception as e:
        log.warning("onnx export skipped: %r", e)
    anomaly = reconstruction_errors(ae, scaler, X)
    from sklearn.metrics import roc_auc_score

    va, te = split == "val", split == "test"
    anomaly_threshold = float(np.quantile(anomaly[va & (y == 0)], anomaly_quantile))
    rules = rule_matrix(X, ctx)

    # thresholds on VALIDATION days only
    hold_threshold = tune_hold_threshold(y[va], risk[va], hold_precision, hold_cap)
    other_warns = (anomaly[va] >= anomaly_threshold) | rules[va]
    budget_left = max(warn_budget - other_warns.mean(), 0.0)
    cand = np.sort(risk[va][~other_warns & (risk[va] < hold_threshold)])[::-1]
    k = int(budget_left * va.sum())
    warn_threshold = float(cand[k]) if 0 < k < len(cand) else hold_threshold
    value_hold = (tune_value_hold(y[va], risk[va], usd[va], hold_threshold, value_hold_precision, value_hold_keep_usd)
                  if value_hold_precision else {"hold_strict_threshold": None, "hold_min_usd": None})
    pol = Policy(anomaly_threshold=anomaly_threshold, warn_threshold=warn_threshold, hold_threshold=hold_threshold,
                 hold_strict_threshold=value_hold["hold_strict_threshold"], hold_min_usd=value_hold["hold_min_usd"])
    thresholds = {"anomaly_threshold": anomaly_threshold, "warn_threshold": warn_threshold,
                  "hold_threshold": hold_threshold, "hold_strict_threshold": pol.hold_strict_threshold,
                  "hold_min_usd": pol.hold_min_usd, "hold_min_precision_val": hold_precision,
                  "hold_cap": hold_cap, "warn_budget": warn_budget,
                  "value_hold_min_precision_val": value_hold_precision, "value_hold_keep_usd_val": value_hold_keep_usd}
    art.thresholds.write_text(json.dumps(thresholds, indent=1))

    # decisions per period + per-scheme disruption
    res = {}
    for name, m in (("val", va), ("test", te)):
        act = decisions(risk[m], anomaly[m], rules[m], pol, usd[m])
        res[name] = summarise(y[m], act, usd[m], scheme_hits(tx, m, act, raw))
    # sanity: vectorised fusion == fusion.decide on a sample (plus every test payment in the hold band)
    sample = np.union1d(np.flatnonzero(te)[:2000], np.flatnonzero(te & (risk >= hold_threshold)))
    for i in sample:
        d = decide(float(anomaly[i]), float(risk[i]), ["x"] if rules[i] else [], pol, float(usd[i])).action
        assert d == decisions(risk[i:i + 1], anomaly[i:i + 1], rules[i:i + 1], pol, usd[i:i + 1])[0]

    stream = write_stream(tx, split, data_dir)
    write_history(tx, split, data_dir)
    metrics = {
        "data": {"payments": int(len(y)), "laundering": int(y.sum()),
                 "by_split": {s: {"payments": int((split == s).sum()), "laundering": int(y[split == s].sum())}
                              for s in ("train", "val", "test")},
                 "accounts": int(pd.concat([tx["sender"], tx["receiver"]]).nunique()),
                 "usd_per_unit": ctx.usd_per_unit},
        "risk_model": risk_metrics,
        "autoencoder": {"loss_by_epoch": history, "threshold_quantile": anomaly_quantile,
                        "anomaly_threshold": anomaly_threshold,
                        "auc_vs_label_test": float(roc_auc_score(y[te], anomaly[te])),
                        "flag_rate_test": float((anomaly[te] >= anomaly_threshold).mean())},
        "rules": {"fire_rate_test": float(rules[te].mean()), "precision_test": float(y[te][rules[te]].mean())},
        "thresholds": thresholds,
        "value_hold_tuning_val": value_hold,
        "decisions": res,
        "versions": {"xgboost": xgb.__version__, "torch": torch.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__},
        "seconds": round(time.time() - t0, 1),
        "stream_file": str(stream),
    }
    art.metrics.write_text(json.dumps(metrics, indent=1, default=str))
    out = tx.loc[te, ["ts", "sender", "receiver", "amount_paid", "currency_paid", "payment_format", "label"]].copy()
    out["usd"], out["risk"], out["anomaly"], out["rules"] = usd[te], risk[te], anomaly[te], rules[te]
    out["action"] = decisions(risk[te], anomaly[te], rules[te], pol, usd[te])
    out.to_csv(data_dir / "test_decisions.csv", index=False)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("shared/data/aml"))
    ap.add_argument("--models", type=Path, default=Path("shared/models"))
    ap.add_argument("--data", type=Path, default=Path("shared/data"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hold-precision", type=float, default=0.6)
    ap.add_argument("--hold-cap", type=float, default=0.002)
    ap.add_argument("--warn-budget", type=float, default=0.0125)
    ap.add_argument("--anomaly-quantile", type=float, default=0.999)
    ap.add_argument("--value-hold-precision", type=float, default=0.8,
                    help="value-aware hold: min validation hold precision (0 = plain risk >= hold_threshold)")
    ap.add_argument("--value-hold-keep-usd", type=float, default=0.95,
                    help="value-aware hold: min share of today's paused laundered USD to keep on validation")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    m = run(a.raw, a.models, a.data, a.epochs, a.seed, a.hold_precision, a.hold_cap, a.warn_budget,
            a.anomaly_quantile, a.value_hold_precision, a.value_hold_keep_usd)
    print(json.dumps({k: m[k] for k in ("risk_model", "thresholds", "decisions", "seconds")}, indent=1, default=str))


if __name__ == "__main__":
    main()
