"""Turn the risk model, the customer-anomaly model and the scam rules into one action.

  HOLD   risk >= hold_strict_threshold, OR (risk >= hold_threshold AND amount >= hold_min_usd):
         pause the payment; an analyst calls the customer back before it goes
  WARN   risk >= warn_threshold, OR the payment is unusual for this customer, OR a scam rule fired:
         show the customer a specific warning; they confirm or cancel (their choice, their money)
  ALLOW  everything else

Holds are rare and precise (tuned on validation for precision, with a cap on volume). Between hold_threshold
and hold_strict_threshold the model is less sure, so only payments big enough to be worth a call-back are held
there (value-aware hold); the rest of that band is warned. Without the two value-aware settings the policy is a
plain risk >= hold_threshold. Warnings are a friction budget (a few % of payments at most), because each one
interrupts an honest customer.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ALLOW = "ALLOW"
WARN = "WARN"
HOLD = "HOLD"
ACTIONS = (ALLOW, WARN, HOLD)


@dataclass
class Policy:
    anomaly_threshold: float
    warn_threshold: float
    hold_threshold: float
    # value-aware hold: both set, or both None (plain risk >= hold_threshold)
    hold_strict_threshold: float | None = None
    hold_min_usd: float | None = None

    def __post_init__(self) -> None:
        if self.warn_threshold > self.hold_threshold:
            raise ValueError("warn_threshold must not exceed hold_threshold")
        if (self.hold_strict_threshold is None) != (self.hold_min_usd is None):
            raise ValueError("set hold_strict_threshold and hold_min_usd together, or neither")
        if self.value_aware and self.hold_strict_threshold < self.hold_threshold:
            raise ValueError("hold_strict_threshold must not be below hold_threshold")

    @property
    def value_aware(self) -> bool:
        return self.hold_strict_threshold is not None

    def holds(self, risk: float, amount_usd: float | None) -> bool:
        if not self.value_aware:
            return risk >= self.hold_threshold
        if risk >= self.hold_strict_threshold:
            return True
        if risk < self.hold_threshold:
            return False
        if amount_usd is None:
            raise ValueError("a value-aware policy needs the payment's USD amount")
        return amount_usd >= self.hold_min_usd

    @classmethod
    def load(cls, policy_file: str | Path, thresholds_file: str | Path | None = None) -> "Policy":
        """config/policy.json (hand-set) over models/thresholds.json (learned); null means use the learned value."""
        learned = json.loads(Path(thresholds_file).read_text()) if thresholds_file else {}
        manual = json.loads(Path(policy_file).read_text())
        merged = {**learned, **{k: v for k, v in manual.items() if v is not None and not k.startswith("_")}}
        hold = float(merged["hold_threshold"])
        strict, min_usd = (None if merged.get(k) is None else float(merged[k])
                           for k in ("hold_strict_threshold", "hold_min_usd"))
        if strict is not None:
            strict = max(strict, hold)  # a hand-set hold_threshold above the learned strict one wins
        return cls(anomaly_threshold=float(merged["anomaly_threshold"]),
                   warn_threshold=float(merged["warn_threshold"]),
                   hold_threshold=hold, hold_strict_threshold=strict, hold_min_usd=min_usd)


@dataclass
class Decision:
    action: str
    risk: float
    anomaly: float
    anomaly_flag: bool
    rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def decide(anomaly: float, risk: float, rules: list[str], policy: Policy,
           amount_usd: float | None = None) -> Decision:
    anomaly_flag = anomaly >= policy.anomaly_threshold
    if policy.holds(risk, amount_usd):
        action = HOLD
    elif risk >= policy.warn_threshold or anomaly_flag or rules:
        action = WARN
    else:
        action = ALLOW
    return Decision(action, float(risk), float(anomaly), anomaly_flag, list(rules))
