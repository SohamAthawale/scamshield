"""Fairness checks on the simulated test stream: are some patient groups returned or held more often?

ClaimSense never uses gender, race or state as model inputs. Age and chronic conditions ARE claim
features (they matter clinically), so disparities by age band are expected and are reported, not hidden.

"False positive" here means a claim from a provider that is NOT flagged in the dataset that was still
returned or held. Labels are provider-level, so this is a proxy: a returned claim from an honest provider
can be a genuine billing error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENDER = {1: "1 (code)", 2: "2 (code)"}  # the dataset gives codes only, no meaning; we don't guess


def attach_groups(sim: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Join simulated decisions (ClaimID, action, label) with patient attributes from the claims table."""
    cols = ["ClaimID", "Gender", "Race", "State", "DOB", "ClaimStartDt"]
    g = sim.merge(claims[cols], on="ClaimID", how="left", validate="one_to_one")
    age = (pd.to_datetime(g["ClaimStartDt"]) - pd.to_datetime(g["DOB"])).dt.days / 365.25
    g["age_band"] = pd.cut(age, [0, 65, 75, 85, 200], right=False, labels=["<65", "65-74", "75-84", "85+"])
    g["gender"] = g["Gender"].map(GENDER).fillna("unknown")
    g["race"] = "code " + g["Race"].astype(str)
    g["flagged"] = g["action"] != "PAY"
    g["held"] = g["action"] == "HOLD_FOR_REVIEW"
    return g


def group_rates(g: pd.DataFrame, by: str, min_claims: int = 200) -> pd.DataFrame:
    """Per group: share returned or held, share held, and the false-positive proxy among honest providers."""
    honest = g[g["label"] == 0]
    out = pd.DataFrame({
        "claims": g.groupby(by, observed=True).size(),
        "flag_rate": g.groupby(by, observed=True)["flagged"].mean(),
        "hold_rate": g.groupby(by, observed=True)["held"].mean(),
        "fpr_honest_providers": honest.groupby(by, observed=True)["flagged"].mean(),
    })
    out = out[out["claims"] >= min_claims]
    overall = g["flagged"].mean()
    out["flag_ratio_vs_overall"] = out["flag_rate"] / overall if overall else np.nan
    return out.sort_values("claims", ascending=False)


def disparity_summary(g: pd.DataFrame, attrs=("gender", "race", "age_band", "State"), min_claims: int = 200) -> pd.DataFrame:
    """Max/min ratio of flag rate across groups for each attribute (1.0 = perfectly even)."""
    rows = []
    for a in attrs:
        r = group_rates(g, a, min_claims)
        if len(r) < 2:
            continue
        lo, hi = r["flag_rate"].min(), r["flag_rate"].max()
        rows.append({"attribute": a, "groups": len(r), "min_flag_rate": lo, "max_flag_rate": hi,
                     "max_over_min": hi / lo if lo > 0 else np.inf,
                     "worst_group": r["flag_rate"].idxmax(), "best_group": r["flag_rate"].idxmin()})
    return pd.DataFrame(rows)
