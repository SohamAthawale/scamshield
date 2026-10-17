"""Turn the two model scores and the hard rules into one action.

  HOLD_FOR_REVIEW   risky provider pattern AND this claim is unusual or breaks a rule
  RETURN_WITH_FIX   claim is unusual or breaks a rule, provider pattern is not high risk
  PAY               everything else

Provider risk only counts once the provider has enough history (min_provider_claims); until then a
single odd claim is treated as an honest mistake, never as a reason to hold payment.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PAY = "PAY"
RETURN = "RETURN_WITH_FIX"
HOLD = "HOLD_FOR_REVIEW"
ACTIONS = (PAY, RETURN, HOLD)


@dataclass
class Policy:
    anomaly_threshold: float
    provider_threshold: float
    min_provider_claims: int = 10

    @classmethod
    def load(cls, policy_file: str | Path, thresholds_file: str | Path | None = None) -> "Policy":
        """config/policy.json holds the hand-set values; models/thresholds.json the learned ones.

        Any value in policy.json that is not null overrides the learned one.
        """
        learned = json.loads(Path(thresholds_file).read_text()) if thresholds_file else {}
        manual = json.loads(Path(policy_file).read_text())
        merged = {**learned, **{k: v for k, v in manual.items() if v is not None}}
        return cls(
            anomaly_threshold=float(merged["anomaly_threshold"]),
            provider_threshold=float(merged["provider_threshold"]),
            min_provider_claims=int(merged.get("min_provider_claims", 10)),
        )


@dataclass
class Decision:
    action: str
    anomaly: float
    anomaly_flag: bool
    provider_risk: float | None
    provider_ready: bool
    provider_flag: bool
    hard_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def decide(anomaly: float, provider_risk: float | None, provider_claims: int,
           hard_errors: list[str], policy: Policy) -> Decision:
    anomaly_flag = anomaly >= policy.anomaly_threshold
    claim_flag = anomaly_flag or bool(hard_errors)
    ready = provider_risk is not None and provider_claims >= policy.min_provider_claims
    provider_flag = bool(ready and provider_risk >= policy.provider_threshold)
    if claim_flag and provider_flag:
        action = HOLD
    elif claim_flag:
        action = RETURN
    else:
        action = PAY
    return Decision(action, float(anomaly), anomaly_flag, provider_risk, ready, provider_flag, list(hard_errors))
