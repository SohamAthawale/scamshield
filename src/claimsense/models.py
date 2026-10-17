"""Model definitions and artifact layout shared by training and serving.

The autoencoder is defined twice on purpose: as a torch module (training, torch/NNPA serving) and as a
NumPy forward pass over the same weights (fallback that needs no torch at all).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AE_HIDDEN = (14, 7, 14)
Z_CLIP = 8.0


@dataclass
class Artifacts:
    """File names inside the models directory (shared/models)."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def feature_context(self) -> Path: return self.root / "feature_context.json"
    @property
    def scaler(self) -> Path: return self.root / "claim_scaler.json"
    @property
    def ae_torch(self) -> Path: return self.root / "claim_ae.pt"
    @property
    def ae_numpy(self) -> Path: return self.root / "claim_ae_weights.npz"
    @property
    def ae_onnx(self) -> Path: return self.root / "claim_ae.onnx"
    @property
    def thresholds(self) -> Path: return self.root / "thresholds.json"
    @property
    def provider_model(self) -> Path: return self.root / "provider_xgb.json"
    @property
    def provider_meta(self) -> Path: return self.root / "provider_meta.json"
    @property
    def metrics(self) -> Path: return self.root / "metrics.json"


class Standardizer:
    """z = clip((x - mean) / std, -Z_CLIP, Z_CLIP). Constant features get std 1."""

    def __init__(self, features: list[str], mean: np.ndarray, std: np.ndarray):
        self.features = list(features)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.where(np.asarray(std, dtype=np.float64) > 1e-9, std, 1.0)

    @classmethod
    def fit(cls, features: list[str], X: np.ndarray) -> "Standardizer":
        return cls(features, X.mean(axis=0), X.std(axis=0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = (np.asarray(X, dtype=np.float64) - self.mean) / self.std
        return np.clip(z, -Z_CLIP, Z_CLIP).astype(np.float32)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(
            {"features": self.features, "mean": self.mean.tolist(), "std": self.std.tolist()}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Standardizer":
        d = json.loads(Path(path).read_text())
        return cls(d["features"], np.array(d["mean"]), np.array(d["std"]))


def build_torch_autoencoder(n_features: int, hidden: tuple[int, ...] = AE_HIDDEN):
    import torch.nn as nn

    sizes = [n_features, *hidden, n_features]
    layers: list = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def torch_to_numpy_weights(model) -> dict[str, np.ndarray]:
    import torch.nn as nn

    out, i = {}, 0
    for layer in model:
        if isinstance(layer, nn.Linear):
            out[f"W{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
            out[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
            i += 1
    return out


def numpy_forward(weights: dict[str, np.ndarray], Z: np.ndarray) -> np.ndarray:
    n_layers = len([k for k in weights if k.startswith("W")])
    h = np.asarray(Z, dtype=np.float32)
    for i in range(n_layers):
        h = h @ weights[f"W{i}"].T + weights[f"b{i}"]
        if i < n_layers - 1:
            h = np.maximum(h, 0.0)
    return h
