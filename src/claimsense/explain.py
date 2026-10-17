"""Reasons for every decision, produced WITHOUT the LLM, plus the Granite prompts built from them.

Reasons only ever contain pseudonymised tokens, feature names, values and peer percentiles. Raw
identifiers and personal details never reach the prompt.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .features import FEATURE_LABELS, HARD_RULES, PROVIDER_LABELS

DOLLAR_LOG = {"log_amount", "log_deductible", "mean_log_amount", "max_log_amount", "mean_log_deductible"}
COUNT_LOG = {"log_claims", "log_unique_benes", "log_unique_attending"}
SHARES = {"inpatient_share", "rare_pair_share", "att_is_op_share", "missing_attending_share"}
FLAGS = {"is_inpatient", "has_operating", "has_other", "att_is_op", "renal", "admit_dx_matches"}


def display_value(feature: str, v: float) -> str:
    if feature in DOLLAR_LOG:
        return f"${math.expm1(v):,.0f}"
    if feature in COUNT_LOG:
        return f"{math.expm1(v):,.0f}"
    if feature in SHARES:
        return f"{100 * v:.1f}%"
    if feature in FLAGS:
        return "yes" if v >= 0.5 else "no"
    if feature in ("amount_vs_dx", "mean_amount_vs_dx"):
        return f"{math.exp(v):.1f}x typical"
    if feature in ("dx_px_rarity", "dx_rarity", "mean_dx_px_rarity"):
        return f"about 1 in {max(1, round(math.exp(v))):,} claims"
    return f"{v:.1f}"


def claim_reasons(features: Mapping[str, float], feature_names: Sequence[str], x_std: np.ndarray,
                  recon_std: np.ndarray, top_k: int = 3) -> list[dict[str, Any]]:
    """Top features by squared reconstruction error, with the direction of the surprise."""
    err = (np.asarray(x_std) - np.asarray(recon_std)) ** 2
    total = float(err.sum()) or 1.0
    order = np.argsort(-err)[:top_k]
    out = []
    for i in order:
        name = feature_names[i]
        out.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "value": display_value(name, float(features[name])),
            "direction": "higher than usual" if x_std[i] > recon_std[i] else "lower than usual",
            "share_of_anomaly": round(float(err[i]) / total, 3),
        })
    return out


def percentile(value: float, quantiles: Sequence[float]) -> float:
    """Peer percentile (0-100) of `value` given 101 quantiles of the peer distribution."""
    q = np.asarray(quantiles, dtype=float)
    lo = np.searchsorted(q, value, side="left")
    hi = np.searchsorted(q, value, side="right")
    return float(np.clip((lo + hi) / 2.0, 0, 100))


def provider_reasons(features: Mapping[str, float], feature_names: Sequence[str], contributions: np.ndarray,
                     peer_quantiles: Mapping[str, Sequence[float]], top_k: int = 3) -> list[dict[str, Any]]:
    """Features that pushed the provider's risk score up the most (XGBoost SHAP contributions)."""
    contrib = np.asarray(contributions, dtype=float)[: len(feature_names)]
    order = [i for i in np.argsort(-contrib) if contrib[i] > 0][:top_k]
    out = []
    for i in order:
        name = feature_names[i]
        v = float(features[name])
        out.append({
            "feature": name,
            "label": PROVIDER_LABELS.get(name, name),
            "value": display_value(name, v),
            "peer_percentile": round(percentile(v, peer_quantiles[name]), 1) if name in peer_quantiles else None,
            "contribution": round(float(contrib[i]), 4),
        })
    return out


def rule_reasons(codes: Sequence[str]) -> list[dict[str, str]]:
    return [{"rule": c, "title": HARD_RULES[c][0], "fix": HARD_RULES[c][1]} for c in codes if c in HARD_RULES]


def _pct(p: float | None) -> str:
    if p is None:
        return ""
    p = int(round(p))
    suffix = "th" if 10 <= p % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(p % 10, "th")
    return f"{p}{suffix} percentile of peers"


def correction_note_template(reasons: Mapping[str, Any]) -> str:
    """Plain-language note to the provider, used when Granite is off or fails."""
    lines = ["We could not process this claim automatically. Please check the following and resubmit:"]
    for r in reasons.get("rules", []):
        lines.append(f"- {r['title']}. {r['fix']}")
    for r in reasons.get("claim", []):
        lines.append(f"- The {r['label']} ({r['value']}) is {r['direction']} for claims like this. "
                     "Please confirm it is correct.")
    lines.append("No action is needed if everything is correct; a reviewer will confirm it.")
    return "\n".join(lines)


def investigator_brief_template(reasons: Mapping[str, Any], provider_token: str, risk: float | None) -> str:
    """Neutral evidence summary for the investigator, used when Granite is off or fails."""
    lines = [f"Provider {provider_token}: risk score {risk:.2f} (model estimate, not a finding of fraud)."
             if risk is not None else f"Provider {provider_token}: not enough history for a risk score."]
    if reasons.get("provider"):
        lines.append("Pattern drivers:")
        for r in reasons["provider"]:
            pct = _pct(r.get("peer_percentile"))
            lines.append(f"- {r['label']}: {r['value']}" + (f" ({pct})" if pct else ""))
    if reasons.get("claim") or reasons.get("rules"):
        lines.append("This claim:")
        for r in reasons.get("rules", []):
            lines.append(f"- {r['title']}")
        for r in reasons.get("claim", []):
            lines.append(f"- {r['label']}: {r['value']} ({r['direction']})")
    lines.append("Suggested next step: review recent claims from this provider before payment.")
    return "\n".join(lines)


SYSTEM_PROMPTS = {
    "correction": (
        "You write short, friendly notes to a medical practice's billing staff. The claim was returned, "
        "not rejected. Explain in plain English what looks wrong and exactly what to check. Do not accuse "
        "anyone. Use only the facts given. 2 to 4 short sentences. No greeting, no sign-off."
    ),
    "brief": (
        "You write neutral evidence briefs for a healthcare payment integrity investigator. Summarise why "
        "this claim was held, using only the facts given. State that the score is a model estimate and "
        "not a finding of fraud. Mention what evidence would confirm or rule out a problem. "
        "3 to 5 short sentences. No headings."
    ),
}


def build_messages(kind: str, facts: Mapping[str, Any]) -> list[dict[str, str]]:
    """Chat messages for Granite. `facts` must already be pseudonymised (tokens, features, values)."""
    if kind not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown note kind {kind!r}")
    lines = []
    for r in facts.get("rules", []):
        lines.append(f"Rule broken: {r['title']}. Suggested fix: {r['fix']}")
    for r in facts.get("claim", []):
        lines.append(f"Unusual on this claim: {r['label']} = {r['value']} ({r['direction']})")
    if kind == "brief":
        if facts.get("provider_risk") is not None:
            lines.append(f"Provider {facts.get('provider_token')} risk score: {facts['provider_risk']:.2f}")
        for r in facts.get("provider", []):
            pct = _pct(r.get("peer_percentile"))
            lines.append(f"Provider pattern: {r['label']} = {r['value']}" + (f" ({pct})" if pct else ""))
    if facts.get("amount") is not None:
        lines.append(f"Claim amount: ${facts['amount']:,.0f}")
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[kind]},
        {"role": "user", "content": "Facts:\n" + "\n".join(lines)},
    ]
