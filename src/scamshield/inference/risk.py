"""Payment risk scoring with XGBoost (trained in the notebook, loaded from portable JSON)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from ..models import Artifacts

log = logging.getLogger(__name__)


class RiskScorer(Protocol):
    name: str
    features: list[str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(n, F) payment features -> probability the payment is part of a laundering/mule scheme, (n,)."""

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """(F,) one payment -> per-feature contribution to the log-odds (SHAP values), (F,)."""


class XGBRiskScorer:
    name = "xgboost"

    def __init__(self, art: Artifacts):
        import xgboost as xgb

        self.xgb = xgb
        meta = json.loads(Path(art.risk_meta).read_text())
        self.features = meta["features"]
        self.booster = xgb.Booster()
        self.booster.load_model(str(art.risk_model))
        self.booster.set_param({"nthread": 1})

    def predict(self, X):
        return self.booster.inplace_predict(np.atleast_2d(np.asarray(X, dtype=np.float32))).astype(np.float64)

    def contributions(self, x):
        d = self.xgb.DMatrix(np.atleast_2d(np.asarray(x, dtype=np.float32)), feature_names=self.features)
        return self.booster.predict(d, pred_contribs=True)[0][: len(self.features)]


def load_risk_scorer(models_dir: str | Path, backend: str = "xgboost") -> RiskScorer:
    if backend != "xgboost":
        raise ValueError(f"unknown risk backend {backend!r}")
    s = XGBRiskScorer(Artifacts(Path(models_dir)))
    log.info("risk scorer: %s", s.name)
    return s
