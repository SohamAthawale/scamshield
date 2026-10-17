"""Inference adapters. Pick one with ANOMALY_BACKEND; each falls back safely."""
from .anomaly import AnomalyScorer, load_anomaly_scorer  # noqa: F401
from .risk import RiskScorer, load_risk_scorer  # noqa: F401
