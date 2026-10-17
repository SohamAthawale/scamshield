"""The scoring engine: features -> two models -> fusion -> reasons, for one claim or a micro-batch.

Used by the API (live stream) and by training (to simulate the stream on validation data when the
provider threshold is tuned), so the numbers in the model card come from the same code path as the demo.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import explain
from .features import (CLAIM_FEATURES, PROVIDER_FEATURES, DuplicateIndex, FeatureContext, ProviderState,
                       claim_features, duplicate_key, hard_errors)
from .fusion import PAY, Decision, Policy, decide
from .inference.anomaly import AnomalyScorer
from .inference.provider import ProviderScorer
from .models import Artifacts


@dataclass
class ClaimInput:
    """One claim ready for scoring. Keys are already pseudonymised tokens."""

    claim: Mapping[str, Any]          # raw claim fields (dates, amounts, codes)
    bene: Mapping[str, Any]           # beneficiary enrolment fields from the payer's registry
    provider_key: str
    bene_key: str
    attending_key: str | None


@dataclass
class ScoreResult:
    decision: Decision
    claim_features: dict[str, float]
    provider_features: dict[str, float]
    provider_claims: int
    reasons: dict[str, Any] = field(default_factory=dict)
    model_ms: float = 0.0


class Engine:
    def __init__(self, ctx: FeatureContext, anomaly: AnomalyScorer, provider: ProviderScorer, policy: Policy):
        self.ctx = ctx
        self.anomaly = anomaly
        self.provider = provider
        self.policy = policy
        self.providers: dict[str, ProviderState] = {}
        self.duplicates = DuplicateIndex()
        if list(anomaly.scaler.features) != CLAIM_FEATURES:
            raise ValueError("autoencoder scaler features differ from features.CLAIM_FEATURES; retrain")
        if list(provider.features) != PROVIDER_FEATURES:
            raise ValueError("provider model features differ from features.PROVIDER_FEATURES; retrain")

    @classmethod
    def from_artifacts(cls, models_dir: str | Path, policy_file: str | Path, anomaly: AnomalyScorer,
                       provider: ProviderScorer) -> "Engine":
        art = Artifacts(Path(models_dir))
        ctx = FeatureContext.from_json(art.feature_context.read_text())
        thresholds = art.thresholds if art.thresholds.exists() else None
        return cls(ctx, anomaly, provider, Policy.load(policy_file, thresholds))

    def score_batch(self, items: Sequence[ClaimInput], explain_all: bool = False) -> list[ScoreResult]:
        """Score claims in arrival order. Provider state is updated claim by claim (order matters);
        the two models run once on the whole batch."""
        if not items:
            return []
        t0 = time.perf_counter()
        cf_rows, pf_rows, n_claims, rules = [], [], [], []
        for it in items:
            f, errs, st = self._observe(it)
            cf_rows.append(f)
            pf_rows.append(st.features())
            n_claims.append(st.n)
            rules.append(errs)
        X = np.array([[r[k] for k in CLAIM_FEATURES] for r in cf_rows], dtype=np.float64)
        P = np.array([[r[k] for k in PROVIDER_FEATURES] for r in pf_rows], dtype=np.float32)
        err, Z, R = self.anomaly.score(X)
        risk = self.provider.predict(P)
        model_ms = (time.perf_counter() - t0) * 1000.0 / len(items)

        out = []
        for i in range(len(items)):
            d = decide(float(err[i]), float(risk[i]), n_claims[i], rules[i], self.policy)
            res = ScoreResult(d, cf_rows[i], pf_rows[i], n_claims[i], model_ms=model_ms)
            if explain_all or d.action != PAY:
                res.reasons = self.reasons(res, Z[i], R[i], P[i])
            out.append(res)
        return out

    def score(self, item: ClaimInput) -> ScoreResult:
        return self.score_batch([item])[0]

    def _observe(self, it: ClaimInput) -> tuple[dict[str, float], list[str], ProviderState]:
        """Update the rolling state with one claim: features, rule hits, duplicate index, provider state."""
        f = claim_features(it.claim, it.bene, self.ctx)
        errs = hard_errors(it.claim, it.bene)
        if self.duplicates.check_and_add(duplicate_key(it.claim, it.bene_key, it.provider_key)):
            errs.append("duplicate")
        st = self.providers.get(it.provider_key)
        if st is None:
            st = self.providers[it.provider_key] = ProviderState()
        st.update(f, it.bene_key, it.attending_key, self.ctx.rare_pair_threshold)
        return f, errs, st

    def replay(self, items: Sequence[ClaimInput]) -> None:
        """Rebuild rolling state from already-decided claims (after a restart) without re-scoring them."""
        for it in items:
            self._observe(it)

    def reasons(self, res: ScoreResult, z: np.ndarray, r: np.ndarray, p: np.ndarray) -> dict[str, Any]:
        d = res.decision
        out: dict[str, Any] = {"rules": explain.rule_reasons(d.hard_errors)}
        if d.anomaly_flag:
            out["claim"] = explain.claim_reasons(res.claim_features, CLAIM_FEATURES, z, r)
        if d.provider_ready and (d.provider_flag or d.action != PAY):
            contrib = self.provider.contributions(p)
            out["provider"] = explain.provider_reasons(res.provider_features, PROVIDER_FEATURES, contrib,
                                                       self.provider.peer_quantiles)
        return out

    def provider_snapshot(self) -> list[dict[str, Any]]:
        """Current risk for every provider seen so far (for the investigator leaderboard)."""
        keys = [k for k, s in self.providers.items() if s.n >= self.policy.min_provider_claims]
        if not keys:
            return []
        feats = [self.providers[k].features() for k in keys]
        P = np.array([[f[c] for c in PROVIDER_FEATURES] for f in feats], dtype=np.float32)
        risk = self.provider.predict(P)
        return sorted(
            ({"provider": k, "risk": float(r), "claims": self.providers[k].n} for k, r in zip(keys, risk)),
            key=lambda x: -x["risk"])
