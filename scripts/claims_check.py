"""Two checks on the headline claims, with the DEPLOYED models and policy, on the z17 VM.

    python scripts/claims_check.py --raw aml --models models --out docs/evidence/claims

1. How concentrated is the laundered money paused on the test days? (a few huge payments, or many?)
2. A measured fairness mitigation: honest ACH payments are warned far more often than other formats.
   Give ACH its own warning threshold, chosen on VALIDATION days to cap the honest-ACH false-alarm rate,
   and report the test days once per cap: the false-alarm gap and what it costs in laundering caught.
   Holds are unchanged (the threshold only moves risk-based warnings).
"""
from __future__ import annotations

import xgboost  # noqa: F401  isort: skip  (before torch, see training.py)

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scamshield.fusion import ALLOW, HOLD, WARN, Policy
from scamshield.models import Artifacts
from scamshield.training import decisions
from value_hold_eval import score_everything

log = logging.getLogger("claims_check")
CAPS = (0.08, 0.06, 0.04)


def concentration(usd: np.ndarray, fmt: np.ndarray) -> dict:
    """usd/fmt of the laundering payments that were paused."""
    order = np.argsort(-usd)
    u, total = usd[order], float(usd.sum())
    cum = np.cumsum(u) / total
    return {"payments": int(len(u)), "usd_total": total,
            "top_share": {str(k): float(u[:k].sum() / total) for k in (1, 5, 10, 20, 50) if k <= len(u)},
            "payments_for_share": {str(q): int(np.searchsorted(cum, q) + 1) for q in (0.5, 0.8, 0.9)},
            "largest_usd": [float(x) for x in u[:5]], "median_usd": float(np.median(u)),
            "usd_share_by_format": {k: float(v / total) for k, v in
                                    pd.Series(usd).groupby(fmt).sum().sort_values(ascending=False).items()}}


def ach_warn_threshold(risk, other, honest_ach, cap: float, lo: float, hi: float) -> float:
    """Lowest risk threshold t in [lo, hi] for ACH so that honest ACH payments flagged
    (other reasons, or risk >= t) stay <= cap. hi = no risk-only ACH warnings at all."""
    n = honest_ach.sum()
    base = (other & honest_ach).sum()
    r = np.sort(risk[honest_ach & ~other & (risk >= lo) & (risk < hi)])[::-1]
    k = int(np.floor(cap * n - base))  # how many risk-only warnings fit
    if k < 0:
        return hi
    if k >= len(r):
        return lo
    return float(np.nextafter(r[k], np.inf))  # holds r[:k] (and anything tied above r[k])


def fairness(y, act, fmt, bank) -> dict:
    flagged = act != ALLOW
    honest = y == 0
    size = pd.Series(bank).map(pd.Series(bank).value_counts()).values
    band = np.where(size < 100, "<100", np.where(size < 1000, "100-1k", np.where(size < 10000, "1k-10k", "10k+")))
    by_fmt = pd.Series(flagged[honest]).groupby(fmt[honest]).mean()
    counts = pd.Series(fmt).value_counts()
    by_fmt = by_fmt[counts.reindex(by_fmt.index) >= 1000]
    by_bank = pd.Series(flagged[honest]).groupby(band[honest]).mean()
    return {"false_alarm_overall": float(flagged[honest].mean()),
            "false_alarm_by_format": {k: float(v) for k, v in by_fmt.items()},
            "format_max_over_min": float(by_fmt.max() / by_fmt.min()),
            "false_alarm_by_bank_size": {k: float(v) for k, v in by_bank.items()},
            "laundering_warned_or_paused": float(flagged[y == 1].mean()),
            "ach_laundering_warned_or_paused": float(flagged[(y == 1) & (fmt == "ACH")].mean()),
            "hold_rate": float((act == HOLD).mean()), "warn_rate": float((act == WARN).mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("aml"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    art = Artifacts(a.models)
    tx, split, y, usd, risk, anomaly, rules = score_everything(a.raw, art)
    th = json.loads(art.thresholds.read_text())
    pol = Policy(th["anomaly_threshold"], th["warn_threshold"], th["hold_threshold"],
                 th.get("hold_strict_threshold"), th.get("hold_min_usd"))
    fmt_all = tx["payment_format"].values
    bank_all = tx["from_bank"].astype(str).values
    va, te = split == "val", split == "test"
    out: dict = {"policy": th}

    # 1) concentration of the laundered USD paused on the test days
    act_te = decisions(risk[te], anomaly[te], rules[te], pol, usd[te])
    paused = (act_te == HOLD) & (y[te] == 1)
    out["laundered_usd_paused_concentration_test"] = concentration(usd[te][paused], fmt_all[te][paused])

    # 2) ACH warning threshold chosen on validation, per cap
    def acts(m, ach_thr):
        hold = decisions(risk[m], anomaly[m], rules[m], pol, usd[m]) == HOLD
        warn_thr = np.where(fmt_all[m] == "ACH", ach_thr, pol.warn_threshold)
        warn = ~hold & ((risk[m] >= warn_thr) | (anomaly[m] >= pol.anomaly_threshold) | rules[m])
        return np.where(hold, HOLD, np.where(warn, WARN, ALLOW))

    hold_va = decisions(risk[va], anomaly[va], rules[va], pol, usd[va]) == HOLD
    other_va = hold_va | (anomaly[va] >= pol.anomaly_threshold) | rules[va]
    honest_ach_va = (y[va] == 0) & (fmt_all[va] == "ACH")
    rows = [{"cap": None, "ach_warn_threshold": pol.warn_threshold,
             "val": fairness(y[va], acts(va, pol.warn_threshold), fmt_all[va], bank_all[va]),
             "test": fairness(y[te], act_te, fmt_all[te], bank_all[te])}]
    for cap in CAPS:
        t = ach_warn_threshold(risk[va], other_va, honest_ach_va, cap, pol.warn_threshold, pol.hold_threshold)
        rows.append({"cap": cap, "ach_warn_threshold": t,
                     "val": fairness(y[va], acts(va, t), fmt_all[va], bank_all[va]),
                     "test": fairness(y[te], acts(te, t), fmt_all[te], bank_all[te])})
    out["ach_warn_cap"] = rows
    out["seconds"] = round(time.time() - t0, 1)
    (a.out / "claims_check.json").write_text(json.dumps(out, indent=1, default=str))

    c = out["laundered_usd_paused_concentration_test"]
    md = ["# Claims check (z17 VM, deployed models and value-aware policy)\n",
          "## 1. How concentrated is the laundered money paused? (test days)\n",
          f"{c['payments']} laundering payments paused, ${c['usd_total'] / 1e6:.1f}M in total.\n",
          "| Largest paused payments | Share of the money paused |", "|---|---|"]
    md += [f"| top {k} | {100 * v:.1f}% |" for k, v in c["top_share"].items()]
    md += ["", "| Share of the money | Payments needed |", "|---|---|"]
    md += [f"| {int(100 * float(q))}% | {n} |" for q, n in c["payments_for_share"].items()]
    md += ["", f"Largest single payment ${c['largest_usd'][0] / 1e6:.1f}M; median paused laundering payment "
           f"${c['median_usd']:,.0f}. By format: " +
           ", ".join(f"{k} {100 * v:.1f}%" for k, v in c["usd_share_by_format"].items()) + ".\n",
           "## 2. Measured mitigation: an ACH-only warning threshold (cap chosen on validation, test scored once)\n",
           "| Honest-ACH cap (val) | ACH warn threshold | ACH false alarms (test) | All honest false alarms | "
           "Worst/best format | Smallest banks (<100) | Laundering warned or paused | ACH laundering warned or paused |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["test"]
        md.append(f"| {'today' if r['cap'] is None else f'{100 * r['cap']:.0f}%'} | {r['ach_warn_threshold']:.3f} | "
                  f"{100 * t['false_alarm_by_format'].get('ACH', float('nan')):.1f}% | "
                  f"{100 * t['false_alarm_overall']:.2f}% | {t['format_max_over_min']:.0f}x | "
                  f"{100 * t['false_alarm_by_bank_size'].get('<100', float('nan')):.1f}% | "
                  f"{100 * t['laundering_warned_or_paused']:.1f}% | {100 * t['ach_laundering_warned_or_paused']:.1f}% |")
    (a.out / "claims_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
