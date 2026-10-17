"""Provider risk scoring with XGBoost (trained in the notebook, loaded from portable JSON)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from ..models import Artifacts

log = logging.getLogger(__name__)


class ProviderScorer(Protocol):
    name: str
    features: list[str]
    peer_quantiles: dict[str, list[float]]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(n, F) provider features -> probability of the 'flagged' pattern, shape (n,)."""

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """(F,) one provider -> per-feature contribution to the log-odds (SHAP values), shape (F,)."""


class XGBProviderScorer:
    name = "xgboost"

    def __init__(self, art: Artifacts):
        import xgboost as xgb

        self.xgb = xgb
        meta = json.loads(Path(art.provider_meta).read_text())
        self.features = meta["features"]
        self.peer_quantiles = meta["peer_quantiles"]
        self.booster = xgb.Booster()
        self.booster.load_model(str(art.provider_model))
        self.booster.set_param({"nthread": 1})

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        return self.booster.inplace_predict(X).astype(np.float64)

    def contributions(self, x):
        d = self.xgb.DMatrix(np.atleast_2d(np.asarray(x, dtype=np.float32)), feature_names=self.features)
        return self.booster.predict(d, pred_contribs=True)[0][: len(self.features)]


def load_provider_scorer(models_dir: str | Path, backend: str = "xgboost") -> ProviderScorer:
    if backend != "xgboost":
        raise ValueError(f"unknown provider backend {backend!r}")
    s = XGBProviderScorer(Artifacts(Path(models_dir)))
    log.info("provider scorer: %s", s.name)
    return s
