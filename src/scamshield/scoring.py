"""The scoring engine: rolling features -> risk model + customer-anomaly model + scam rules -> action + reasons.

The API calls score_batch() on micro-batches of payments. Rolling state is updated payment by payment in
arrival order; the two models run once per batch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import explain
from .features import ANOMALY_FEATURES, FEATURES, FeatureContext, FeatureEngine, rule_hits
from .fusion import ALLOW, Decision, Policy, decide
from .inference.anomaly import AnomalyScorer
from .inference.risk import RiskScorer
from .models import Artifacts

AE_IDX = [FEATURES.index(f) for f in ANOMALY_FEATURES]


@dataclass
class PaymentInput:
    """One payment ready for scoring. Account keys are already pseudonymised tokens."""

    t_minutes: float
    sender_key: str
    receiver_key: str
    amount_usd: float
    payment_format: str
    cross_bank: bool
    currency_mismatch: bool
    hour: int


@dataclass
class ScoreResult:
    decision: Decision
    features: dict[str, float]
    reasons: dict[str, Any] = field(default_factory=dict)
    model_ms: float = 0.0


class Engine:
    def __init__(self, ctx: FeatureContext, anomaly: AnomalyScorer, risk: RiskScorer, policy: Policy):
        if list(anomaly.scaler.features) != ANOMALY_FEATURES:
            raise ValueError("autoencoder features differ from features.ANOMALY_FEATURES; retrain")
        if list(risk.features) != FEATURES:
            raise ValueError("risk model features differ from features.FEATURES; retrain")
        self.ctx, self.anomaly, self.risk, self.policy = ctx, anomaly, risk, policy
        self.fe = FeatureEngine(ctx)
        # receiving accounts seen in live scoring: token -> [payments, sum risk, max risk, warned/held, last t]
        self.watch: dict[str, list[float]] = {}

    @classmethod
    def from_artifacts(cls, models_dir: str | Path, policy_file: str | Path, anomaly: AnomalyScorer,
                       risk: RiskScorer) -> "Engine":
        art = Artifacts(Path(models_dir))
        ctx = FeatureContext.from_json(art.feature_context.read_text())
        thresholds = art.thresholds if art.thresholds.exists() else None
        return cls(ctx, anomaly, risk, Policy.load(policy_file, thresholds))

    # ------------------------------------------------------------------ state
    def _observe(self, it: PaymentInput) -> list[float]:
        return self.fe.observe(it.t_minutes, it.sender_key, it.receiver_key, it.amount_usd, it.payment_format,
                               it.cross_bank, it.currency_mismatch, it.hour)

    def bootstrap(self, history: pd.DataFrame) -> int:
        """Build rolling state from core-banking history (tokenised sender/receiver columns), no scoring."""
        self.fe.run(history)
        return len(history)

    def replay(self, items: Sequence[PaymentInput]) -> None:
        """Re-apply already-decided payments (after a restart) without re-scoring them."""
        for it in items:
            self._observe(it)

    # ------------------------------------------------------------------ scoring
    def score_batch(self, items: Sequence[PaymentInput], explain_all: bool = False) -> list[ScoreResult]:
        if not items:
            return []
        t0 = time.perf_counter()
        X = np.array([self._observe(it) for it in items], dtype=np.float32)
        err, Z, R = self.anomaly.score(X[:, AE_IDX])
        risk = self.risk.predict(X)
        model_ms = (time.perf_counter() - t0) * 1000.0 / len(items)
        out = []
        for i, it in enumerate(items):
            f = dict(zip(FEATURES, X[i].tolist()))
            d = decide(float(err[i]), float(risk[i]), rule_hits(f, self.ctx), self.policy, it.amount_usd)
            res = ScoreResult(d, f, model_ms=model_ms)
            if explain_all or d.action != ALLOW:
                res.reasons = self.reasons(res, Z[i], R[i], X[i])
            self._watch(it, d)
            out.append(res)
        return out

    def score(self, item: PaymentInput) -> ScoreResult:
        return self.score_batch([item])[0]

    def reasons(self, res: ScoreResult, z: np.ndarray, r: np.ndarray, x: np.ndarray) -> dict[str, Any]:
        d = res.decision
        out: dict[str, Any] = {"rules": explain.rule_reasons(d.rules)}
        if d.anomaly_flag:
            out["customer"] = explain.anomaly_reasons(res.features, ANOMALY_FEATURES, z, r)
        if d.risk >= self.policy.warn_threshold:
            out["drivers"] = explain.risk_reasons(res.features, FEATURES, self.risk.contributions(x))
        return out

    # ------------------------------------------------------------------ mule watchlist
    def _watch(self, it: PaymentInput, d: Decision) -> None:
        w = self.watch.get(it.receiver_key)
        if w is None:
            w = self.watch[it.receiver_key] = [0.0, 0.0, 0.0, 0.0, 0.0]
        w[0] += 1
        w[1] += d.risk
        w[2] = max(w[2], d.risk)
        w[3] += d.action != ALLOW
        w[4] = it.t_minutes

    def watchlist(self, top: int = 15, min_payments: int = 2) -> list[dict[str, Any]]:
        """Receiving accounts whose incoming payments score highest (candidate mule accounts)."""
        rows = [{"account": k, "payments": int(w[0]), "mean_risk": w[1] / w[0], "max_risk": w[2],
                 "flagged": int(w[3])} for k, w in self.watch.items()
                if w[0] >= min_payments and w[2] >= self.policy.warn_threshold]
        return sorted(rows, key=lambda r: (-r["mean_risk"], -r["payments"]))[:top]
