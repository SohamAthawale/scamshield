"""Inference adapters. Pick one with ANOMALY_BACKEND / PROVIDER_BACKEND; each falls back safely."""
from .anomaly import AnomalyScorer, load_anomaly_scorer  # noqa: F401
from .provider import ProviderScorer, load_provider_scorer  # noqa: F401
