"""Fairness checks on the test days: are some groups of honest customers bothered more often?

The AML data has no demographics, so we audit the proxies a bank could be unfair along:
  * payment currency  - people sending in Rupee, Yuan, Shekel... are often migrant communities
  * payment format    - who pays by cash, cheque or Bitcoin
  * sender bank size  - customers of small banks vs big banks

The key measure is the FALSE-ALARM rate: honest (non-laundering) payments that were still warned or held.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def attach_groups(dec: pd.DataFrame) -> pd.DataFrame:
    """dec: test decisions with sender, currency_paid, payment_format, label, action."""
    g = dec.copy()
    bank = g["sender"].str.split(":", n=1).str[0]
    size = bank.map(bank.value_counts())
    g["bank_size"] = pd.cut(size, [0, 100, 1000, 10000, np.inf], labels=["<100", "100-1k", "1k-10k", "10k+"])
    g["flagged"] = g["action"] != "ALLOW"
    g["held"] = g["action"] == "HOLD"
    return g


def group_rates(g: pd.DataFrame, by: str, min_payments: int = 1000) -> pd.DataFrame:
    honest = g[g["label"] == 0]
    out = pd.DataFrame({
        "payments": g.groupby(by, observed=True).size(),
        "false_alarm_rate": honest.groupby(by, observed=True)["flagged"].mean(),
        "false_hold_rate": honest.groupby(by, observed=True)["held"].mean(),
        "laundering_caught": g[g["label"] == 1].groupby(by, observed=True)["flagged"].mean(),
    })
    out = out[out["payments"] >= min_payments]
    overall = honest["flagged"].mean()
    out["vs_overall"] = out["false_alarm_rate"] / overall if overall else np.nan
    return out.sort_values("payments", ascending=False)


def disparity_summary(g: pd.DataFrame, attrs=("currency_paid", "payment_format", "bank_size"),
                      min_payments: int = 1000) -> pd.DataFrame:
    rows = []
    for a in attrs:
        r = group_rates(g, a, min_payments)
        if len(r) < 2:
            continue
        lo, hi = r["false_alarm_rate"].min(), r["false_alarm_rate"].max()
        rows.append({"attribute": a, "groups": len(r), "min_false_alarm": lo, "max_false_alarm": hi,
                     "max_over_min": hi / lo if lo > 0 else np.inf,
                     "worst_group": r["false_alarm_rate"].idxmax(), "best_group": r["false_alarm_rate"].idxmin()})
    return pd.DataFrame(rows)
