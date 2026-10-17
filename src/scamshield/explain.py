"""Reasons for every decision, produced WITHOUT the LLM, plus the Granite prompts built from them.

Reasons contain only pseudonymised tokens, feature labels and values. Account numbers, bank codes and
names never reach the prompt.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .features import FEATURE_LABELS, RULES

FLAGS = {"cross_bank", "self_transfer", "currency_mismatch", "s_new_payee", "s_is_new", "r_is_new"}
SHARES = {"r_fan_in", "r_share_ach_in"}
COUNTS = {"s_log_n_out", "s_log_n_in", "s_log_payees", "r_log_n_in", "r_log_n_out", "r_log_senders"}


def display_value(feature: str, v: float) -> str:
    if feature == "log_usd":
        return f"${math.expm1(v):,.0f}"
    if feature.startswith("fmt_") or feature in FLAGS:
        return "yes" if v >= 0.5 else "no"
    if feature in COUNTS:
        return f"{math.expm1(v):,.0f}"
    if feature in SHARES:
        return f"{100 * v:.0f}%"
    if feature == "s_amount_vs_usual":
        return f"{math.exp(v):.1f}x usual"
    if feature in ("s_pass_through", "r_pass_through"):
        return f"sends on {100 * math.exp(v):.0f}% of what it receives"
    if feature in ("s_log_idle_h", "r_log_idle_h"):
        h = math.expm1(v)
        return f"{h:.0f}" if h < 72 else f"{h:.0f} (about {h / 24:.0f} days)"
    if feature in ("s_recent_out", "r_recent_in"):
        return f"about {v:.0f} in the last day"
    if feature in ("hour_sin", "hour_cos"):
        return ""
    return f"{v:.2f}"


def _merge_hour(feature: str) -> str:
    return "hour" if feature in ("hour_sin", "hour_cos") else feature


def anomaly_reasons(features: Mapping[str, float], names: Sequence[str], z: np.ndarray, recon: np.ndarray,
                    top_k: int = 3, min_share: float = 0.10) -> list[dict[str, Any]]:
    """What made the payment unusual FOR THIS CUSTOMER (autoencoder per-feature error)."""
    err = (np.asarray(z) - np.asarray(recon)) ** 2
    total = float(err.sum()) or 1.0
    out, seen = [], set()
    for j, i in enumerate(np.argsort(-err)):
        name = _merge_hour(names[i])
        if name in seen:
            continue
        if len(out) >= top_k or (j > 0 and err[i] / total < min_share):
            break
        seen.add(name)
        out.append({"feature": name, "label": "time of day" if name == "hour" else FEATURE_LABELS.get(name, name),
                    "value": "" if name == "hour" else display_value(name, float(features[name])),
                    "direction": "higher than usual" if z[i] > recon[i] else "lower than usual",
                    "share": round(float(err[i]) / total, 3)})
    return out


def risk_reasons(features: Mapping[str, float], names: Sequence[str], contributions: np.ndarray,
                 top_k: int = 4) -> list[dict[str, Any]]:
    """What pushed the risk score up (XGBoost SHAP contributions), strongest first."""
    c = np.asarray(contributions, dtype=float)[: len(names)]
    out, seen = [], set()
    for i in np.argsort(-c):
        if c[i] <= 0 or len(out) >= top_k:
            break
        name = _merge_hour(names[i])
        if name in seen:
            continue
        seen.add(name)
        out.append({"feature": name, "label": "time of day" if name == "hour" else FEATURE_LABELS.get(name, name),
                    "value": "" if name == "hour" else display_value(name, float(features[name])),
                    "contribution": round(float(c[i]), 3)})
    return out


def rule_reasons(codes: Sequence[str]) -> list[dict[str, str]]:
    return [{"rule": c, "title": RULES[c][0], "advice": RULES[c][1]} for c in codes if c in RULES]


PAY_BY = {"fmt_ACH": "bank transfer (ACH)", "fmt_Bitcoin": "Bitcoin", "fmt_Cash": "cash", "fmt_Cheque": "cheque",
          "fmt_CreditCard": "credit card", "fmt_Reinvestment": "reinvestment", "fmt_Wire": "wire transfer"}


def _customer_phrase(r: Mapping[str, Any]) -> str | None:
    """Plain-English sentence for an anomaly reason a customer can act on; None for internal signals."""
    f, v, up = r["feature"], r.get("value", ""), r.get("direction") == "higher than usual"
    if f == "s_amount_vs_usual" and up:
        return f"This payment is about {v.replace(' usual', '')} your usual amount."
    if f == "log_usd" and up:
        return f"The amount ({v}) is larger than you usually send."
    if f == "s_new_payee" and v == "yes":
        return "It's your first payment to this payee."
    if f.startswith("fmt_") and v == "yes":
        return f"You don't usually pay by {PAY_BY.get(f, r['label'])}."
    if f == "cross_bank" and v == "yes":
        return "You don't usually pay accounts at other banks."
    if f == "currency_mismatch" and v == "yes":
        return "It involves a currency conversion, which is unusual for you."
    if f == "hour":
        return "It's at a time of day when you don't usually make payments."
    if f == "s_recent_out" and up:
        return "You've made more payments than usual in the last day."
    return None


def customer_lines(reasons: Mapping[str, Any]) -> list[str]:
    """What the customer is told: rule advice plus the anomaly reasons that make sense to a customer."""
    lines = [f"{r['title']}. {r['advice']}" for r in reasons.get("rules", [])]
    for r in reasons.get("customer", []):
        p = _customer_phrase(r)
        if p and p not in lines:
            lines.append(p)
    return lines


def customer_warning_template(reasons: Mapping[str, Any], held: bool) -> str:
    """Shown to the customer instantly (Granite may later write a friendlier version for their inbox)."""
    head = ("We've paused this payment so we can check it with you. Our team will call you on the number we "
            "already have; we will never ask you to move money or share a code."
            if held else
            "Before you send this payment, please check it. Scammers often pressure people to pay quickly.")
    lines = [head]
    specific = customer_lines(reasons)
    lines += [f"- {x}" for x in specific]
    if not specific:
        lines.append("- This payment doesn't look like your usual payments, or the account you're paying shows "
                     "patterns we often see in scam accounts.")
    lines.append("If someone contacted you out of the blue and asked for this payment, stop and call us.")
    return "\n".join(lines)


def analyst_brief_template(reasons: Mapping[str, Any], payment_token: str, risk: float) -> str:
    lines = [f"Payment {payment_token}: risk score {risk:.2f} (model estimate, not a finding of fraud)."]
    if reasons.get("drivers"):
        lines.append("Main drivers:")
        lines += [f"- {r['label']}" + (f": {r['value']}" if r["value"] else "") for r in reasons["drivers"]]
    for r in reasons.get("rules", []):
        lines.append(f"- Rule: {r['title']}")
    lines.append("Suggested next step: call the customer on their registered number before releasing the payment.")
    return "\n".join(lines)


SYSTEM_PROMPTS = {
    "warning": (
        "You write short, calm messages from a bank to a customer whose payment was paused or flagged. Explain in "
        "plain English why the bank wants to check it, using only the facts given. Never blame the customer, never "
        "say they are a criminal, and never ask them to share passwords or codes or to move money. Use only the facts "
        "and numbers listed; do not add any others. 2 to 4 short sentences. No greeting, no sign-off."
    ),
    "brief": (
        "You write neutral evidence briefs for a bank's scam and fraud analyst. Summarise why this payment was "
        "held. Use only the facts and numbers listed; do not add, infer or round any others. Say that the score is "
        "a model estimate and not a finding of fraud. Suggest what to check with the customer. 3 to 5 short "
        "sentences. No headings."
    ),
}


def build_messages(kind: str, facts: Mapping[str, Any]) -> list[dict[str, str]]:
    """Chat messages for Granite. `facts` must already be pseudonymised (tokens, labels, values)."""
    if kind not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown note kind {kind!r}")
    lines = []
    if facts.get("amount_usd") is not None:
        lines.append(f"Payment amount: ${facts['amount_usd']:,.0f} by {facts.get('payment_format', 'transfer')}")
    lines.append("Status: paused for a call-back" if facts.get("held") else "Status: customer asked to confirm")
    if kind == "warning":
        lines += [f"Why we're checking: {x}" for x in customer_lines(facts)]
    else:
        for r in facts.get("rules", []):
            lines.append(f"Scam rule: {r['title']}")
        for r in facts.get("customer", []):
            lines.append(f"Unusual for this customer: {r['label']}" + (f" = {r['value']}" if r["value"] else "")
                         + f" ({r['direction']})")
    if kind == "brief":
        if facts.get("risk") is not None:
            lines.append(f"Payment {facts.get('payment_token')} risk score: {facts['risk']:.2f}")
        for r in facts.get("drivers", []):
            lines.append(f"Risk driver: {r['label']}" + (f" = {r['value']}" if r["value"] else ""))
    return [{"role": "system", "content": SYSTEM_PROMPTS[kind]},
            {"role": "user", "content": "Facts:\n" + "\n".join(lines)}]
