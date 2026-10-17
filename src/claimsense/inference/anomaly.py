"""Claim autoencoder scoring: torch on the Telum II NNPA, torch on CPU, or pure NumPy.

Measured on z17 with the trained model and real claims (docs/evidence/bench/bench_autoencoder.csv): the
NNPA is slower than the CPU up to ~64 claims per call (dispatch overhead), even at 256, 1.4x faster at
1024 and 1.6x at 4096, and slower again for one huge call of 16384. So batches of at least
`nnpa_min_batch` rows go to the NNPA in chunks of at most `nnpa_chunk` rows; smaller ones stay on the CPU.
"""
from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from ..models import Artifacts, Standardizer, build_torch_autoencoder, numpy_forward

log = logging.getLogger(__name__)


class AnomalyScorer(Protocol):
    name: str
    scaler: Standardizer

    def reconstruct(self, Z: np.ndarray) -> np.ndarray:
        """Standardised input (n, F) -> reconstruction (n, F), float32."""

    def score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw features (n, F) -> (error per row, standardised input, reconstruction)."""


class _Base:
    name = "base"
    scaler: Standardizer

    def score(self, X: np.ndarray):
        Z = self.scaler.transform(np.atleast_2d(X))
        R = self.reconstruct(Z)
        err = ((Z - R) ** 2).mean(axis=1)
        return err.astype(np.float64), Z, R


class NumpyAnomalyScorer(_Base):
    name = "numpy"

    def __init__(self, art: Artifacts):
        self.scaler = Standardizer.load(art.scaler)
        with np.load(art.ae_numpy) as w:
            self.weights = {k: w[k] for k in w.files}

    def reconstruct(self, Z):
        return numpy_forward(self.weights, Z)


class TorchAnomalyScorer(_Base):
    def __init__(self, art: Artifacts, use_nnpa: bool, nnpa_min_batch: int = 256, nnpa_chunk: int = 4096):
        import torch

        self.torch = torch
        self.scaler = Standardizer.load(art.scaler)
        state = torch.load(art.ae_torch, map_location="cpu", weights_only=True)
        n_features = len(self.scaler.features)
        self.cpu_model = build_torch_autoencoder(n_features).eval()
        self.cpu_model.load_state_dict(state)
        self.nnpa_model = None
        self.nnpa_min_batch = nnpa_min_batch
        self.nnpa_chunk = nnpa_chunk
        self._lock = threading.Lock()  # one NNPA call at a time from this process
        if use_nnpa:
            import torch_nnpa  # noqa: F401  (registers the 'nnpa' device)

            if not torch_nnpa.NNPAModule.is_available():
                raise RuntimeError("torch_nnpa installed but NNPA not available on this machine")
            self.device = torch_nnpa.device()
            self.nnpa_model = copy.deepcopy(self.cpu_model).to(self.device).eval()
        self.name = "torch_nnpa" if self.nnpa_model is not None else "torch_cpu"

    def reconstruct(self, Z, force: str | None = None):
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float32))
        use_nnpa = self.nnpa_model is not None and (
            force == "nnpa" or (force is None and len(Z) >= self.nnpa_min_batch))
        with torch.no_grad():
            if use_nnpa and force != "cpu":
                with self._lock:
                    parts = [self.nnpa_model(x[i:i + self.nnpa_chunk].to(self.device)).cpu()
                             for i in range(0, len(x), self.nnpa_chunk)]
                return torch.cat(parts).numpy()
            return self.cpu_model(x).numpy()


def load_anomaly_scorer(models_dir: str | Path, backend: str = "auto", nnpa_min_batch: int = 256) -> AnomalyScorer:
    """backend: auto | nnpa | cpu | numpy. auto = nnpa if available, then cpu torch, then numpy."""
    art = Artifacts(Path(models_dir))
    chains = {"auto": ["nnpa", "cpu", "numpy"], "nnpa": ["nnpa", "cpu", "numpy"],
              "cpu": ["cpu", "numpy"], "numpy": ["numpy"]}
    if backend not in chains:
        raise ValueError(f"unknown anomaly backend {backend!r}; use one of {sorted(chains)}")
    order = chains[backend]
    errors = []
    for b in order:
        try:
            if b == "numpy":
                s = NumpyAnomalyScorer(art)
            else:
                s = TorchAnomalyScorer(art, use_nnpa=(b == "nnpa"), nnpa_min_batch=nnpa_min_batch)
            log.info("anomaly scorer: %s", s.name)
            return s
        except Exception as e:  # fall through to the next backend, but say why
            errors.append(f"{b}: {e!r}")
            log.warning("anomaly backend %s unavailable: %r", b, e)
    raise RuntimeError("no anomaly backend could load: " + "; ".join(errors))
