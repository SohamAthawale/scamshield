"""Value-aware HOLD: tune it on the validation days with the DEPLOYED models, then report test days old vs new.

    python scripts/value_hold_eval.py --raw aml --models models --out docs/evidence/value_hold

Nothing is retrained and nothing in --models is changed. The choice of (hold_strict_threshold, hold_min_usd)
sees validation payments only; the test days are scored once, with the chosen values.
"""
from __future__ import annotations

import xgboost as xgb  # noqa: F401  isort: skip  (before torch, see training.py)

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from scamshield import data as D
from scamshield.features import ANOMALY_FEATURES, FeatureContext, FeatureEngine
from scamshield.fusion import Policy
from scamshield.models import Artifacts, Standardizer, build_torch_autoencoder
from scamshield.training import (decisions, reconstruction_errors, rule_matrix, scheme_hits, summarise,
                                 tune_value_hold)

log = logging.getLogger("value_hold_eval")
ROWS = [("Hold rate", "mix.HOLD", "pct3"), ("Holds", "holds", "int"),
        ("Hold precision", "hold_precision", "pct1"), ("Hold recall (laundering payments)", "laundering_recall_hold", "pct1"),
        ("Laundering payments held", "laundering_held", "int"),
        ("Laundered USD paused", "laundered_usd_held", "usd"),
        ("Laundering warned or paused", "laundering_recall_warn_or_hold", "pct1"),
        ("Warn rate", "mix.WARN", "pct2"), ("Schemes broken (>=1 payment held)", "schemes_disrupted_total", "str")]


def get(d: dict, path: str):
    for k in path.split("."):
        d = d[k]
    return d


def fmt(v, kind: str) -> str:
    if v is None:
        return "n/a"
    return {"pct1": lambda: f"{100 * v:.1f}%", "pct2": lambda: f"{100 * v:.2f}%", "pct3": lambda: f"{100 * v:.3f}%",
            "int": lambda: f"{v:,}", "usd": lambda: f"${v / 1e6:.1f}M", "str": lambda: str(v)}[kind]()


def table(old: dict, new: dict, total_usd: float) -> str:
    lines = ["| | Old (risk >= hold) | New (value-aware) |", "|---|---|---|"]
    for label, path, kind in ROWS:
        lines.append(f"| {label} | {fmt(get(old, path), kind)} | {fmt(get(new, path), kind)} |")
    lines.append(f"| Laundered USD in period | ${total_usd / 1e6:.1f}M | ${total_usd / 1e6:.1f}M |")
    return "\n".join(lines)


def score_everything(raw: Path, art: Artifacts):
    """Every payment scored exactly as training did, with the DEPLOYED models (no retraining)."""
    import torch

    t0 = time.time()
    ctx = FeatureContext.from_json(art.feature_context.read_text())
    tx = D.load_transactions(str(raw))
    split = D.time_split(tx["ts"])
    X = FeatureEngine(ctx).run(tx)
    log.info("features for %d payments in %.0fs", len(tx), time.time() - t0)
    y = tx["label"].values.astype(int)
    usd = tx["amount_paid"].values * tx["currency_paid"].map(ctx.usd_per_unit).values
    booster = xgb.Booster()
    booster.load_model(str(art.risk_model))
    risk = booster.inplace_predict(X).astype(np.float64)
    ae = build_torch_autoencoder(len(ANOMALY_FEATURES))
    ae.load_state_dict(torch.load(art.ae_torch, map_location="cpu"))
    anomaly = reconstruction_errors(ae.eval(), Standardizer.load(art.scaler), X)
    return tx, split, y, usd, risk, anomaly, rule_matrix(X, ctx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("aml"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-precision", type=float, default=0.8)
    ap.add_argument("--keep-usd", type=float, default=0.95)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch

    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    art = Artifacts(a.models)
    tx, split, y, usd, risk, anomaly, rules = score_everything(a.raw, art)

    th = json.loads(art.thresholds.read_text())
    old = Policy(th["anomaly_threshold"], th["warn_threshold"], th["hold_threshold"])
    va, te = split == "val", split == "test"

    # 1) tune on validation only
    tuned = tune_value_hold(y[va], risk[va], usd[va], old.hold_threshold, a.min_precision, a.keep_usd)
    log.info("validation choice: %s", json.dumps(tuned))
    if tuned["hold_strict_threshold"] is None:
        raise SystemExit("no value-aware choice meets the validation targets; keep the plain policy")
    new = Policy(old.anomaly_threshold, old.warn_threshold, old.hold_threshold,
                 tuned["hold_strict_threshold"], tuned["hold_min_usd"])
    # validation-only sensitivity of the choice to the targets (no test numbers)
    sensitivity = []
    for p in (0.7, 0.75, 0.8, 0.85, 0.9):
        for k in (0.9, 0.95, 0.99):
            t = tune_value_hold(y[va], risk[va], usd[va], old.hold_threshold, p, k)
            sensitivity.append({"min_precision": p, "keep_usd": k, "hold_strict_threshold": t["hold_strict_threshold"],
                                "hold_min_usd": t["hold_min_usd"], "val": t["val_chosen"]})

    # 2) score both policies on both periods
    res = {}
    for name, m in (("val", va), ("test", te)):
        res[name] = {}
        for label, pol in (("old", old), ("new", new)):
            act = decisions(risk[m], anomaly[m], rules[m], pol, usd[m])
            res[name][label] = summarise(y[m], act, usd[m], scheme_hits(tx, m, act, a.raw))

    # 3) the old-policy numbers must reproduce the deployed metrics.json exactly (same models, same data)
    deployed = json.loads(art.metrics.read_text())["decisions"]
    repro = {s: {k: (res[s]["old"][k], deployed[s][k]) for k in deployed[s] if res[s]["old"].get(k) != deployed[s][k]}
             for s in ("val", "test")}

    thresholds_new = {**th, "hold_strict_threshold": new.hold_strict_threshold, "hold_min_usd": new.hold_min_usd,
                      "value_hold_min_precision_val": a.min_precision, "value_hold_keep_usd_val": a.keep_usd}
    out = {"targets": {"min_precision_val": a.min_precision, "keep_usd_val": a.keep_usd},
           "old_policy": th, "new_policy": thresholds_new, "validation_tuning": tuned,
           "validation_sensitivity": sensitivity, "decisions": res,
           "old_policy_reproduces_deployed_metrics": not any(repro.values()), "reproduction_diffs": repro,
           "versions": {"xgboost": xgb.__version__, "torch": torch.__version__, "numpy": np.__version__},
           "seconds": round(time.time() - t0, 1)}
    (a.out / "value_hold_eval.json").write_text(json.dumps(out, indent=1, default=str))
    (a.out / "thresholds_value_hold.json").write_text(json.dumps(thresholds_new, indent=1))
    md = [f"# Value-aware HOLD: old vs new (z17 VM, deployed models)\n",
          f"Chosen on validation days only (targets: hold precision >= {a.min_precision:.0%}, keep >= "
          f"{a.keep_usd:.0%} of the laundered USD the old policy pauses): hold if risk >= "
          f"{new.hold_strict_threshold:.4f}, or if risk >= {new.hold_threshold:.4f} and amount >= "
          f"${new.hold_min_usd:,.0f}.\n",
          "## Test days (9-10 Sep, scored once)\n", table(res["test"]["old"], res["test"]["new"],
                                                          res["test"]["old"]["laundered_usd_total"]),
          "\n## Validation days (7-8 Sep, used for the choice)\n", table(res["val"]["old"], res["val"]["new"],
                                                                         res["val"]["old"]["laundered_usd_total"]),
          f"\nOld-policy numbers reproduce the deployed metrics.json: {out['old_policy_reproduces_deployed_metrics']}\n"]
    (a.out / "value_hold_comparison.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
