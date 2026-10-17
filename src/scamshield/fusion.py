"""Turn the risk model, the customer-anomaly model and the scam rules into one action.

  HOLD   risk >= hold_threshold: pause the payment; an analyst calls the customer back before it goes
  WARN   risk >= warn_threshold, OR the payment is unusual for this customer, OR a scam rule fired:
         show the customer a specific warning; they confirm or cancel (their choice, their money)
  ALLOW  everything else

Holds are rare and precise (tuned on validation for precision, with a cap on volume). Warnings are a
friction budget (a few % of payments at most), because each one interrupts an honest customer.
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

    def __post_init__(self) -> None:
        if self.warn_threshold > self.hold_threshold:
            raise ValueError("warn_threshold must not exceed hold_threshold")

    @classmethod
    def load(cls, policy_file: str | Path, thresholds_file: str | Path | None = None) -> "Policy":
        """config/policy.json (hand-set) over models/thresholds.json (learned); null means use the learned value."""
        learned = json.loads(Path(thresholds_file).read_text()) if thresholds_file else {}
        manual = json.loads(Path(policy_file).read_text())
        merged = {**learned, **{k: v for k, v in manual.items() if v is not None and not k.startswith("_")}}
        return cls(anomaly_threshold=float(merged["anomaly_threshold"]),
                   warn_threshold=float(merged["warn_threshold"]),
                   hold_threshold=float(merged["hold_threshold"]))


@dataclass
class Decision:
    action: str
    risk: float
    anomaly: float
    anomaly_flag: bool
    rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def decide(anomaly: float, risk: float, rules: list[str], policy: Policy) -> Decision:
    anomaly_flag = anomaly >= policy.anomaly_threshold
    if risk >= policy.hold_threshold:
        action = HOLD
    elif risk >= policy.warn_threshold or anomaly_flag or rules:
        action = WARN
    else:
        action = ALLOW
    return Decision(action, float(risk), float(anomaly), anomaly_flag, list(rules))
