"""models.py, inference/anomaly.py and inference/provider.py on tiny artifacts built in conftest."""
import logging

import numpy as np
import pytest

from claimsense.features import CLAIM_FEATURES, PROVIDER_FEATURES
from claimsense.inference import load_anomaly_scorer, load_provider_scorer
from claimsense.models import (
    AE_HIDDEN, Z_CLIP, Artifacts, Standardizer, build_torch_autoencoder, numpy_forward, torch_to_numpy_weights,
)


# ------------------------------------------------------------------------------ Standardizer
def test_standardizer_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(5, 3, size=(200, 4))
    s = Standardizer.fit(["a", "b", "c", "d"], X)
    s.save(tmp_path / "s.json")
    t = Standardizer.load(tmp_path / "s.json")
    assert t.features == ["a", "b", "c", "d"]
    np.testing.assert_array_equal(t.mean, s.mean)
    np.testing.assert_array_equal(t.std, s.std)
    np.testing.assert_array_equal(t.transform(X), s.transform(X))
    Z = s.transform(X)
    assert Z.dtype == np.float32
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-4)


def test_standardizer_clips_to_z_clip():
    s = Standardizer(["a", "b"], np.zeros(2), np.ones(2))
    Z = s.transform([[100.0, -100.0], [3.0, -3.0]])
    np.testing.assert_array_equal(Z, np.array([[Z_CLIP, -Z_CLIP], [3.0, -3.0]], dtype=np.float32))


def test_standardizer_constant_feature_gets_unit_std(tmp_path):
    X = np.column_stack([np.full(10, 7.0), np.arange(10.0)])
    s = Standardizer.fit(["const", "x"], X)
    assert s.std[0] == 1.0
    np.testing.assert_array_equal(s.transform(X)[:, 0], 0.0)
    s.save(tmp_path / "s.json")
    assert Standardizer.load(tmp_path / "s.json").std[0] == 1.0


def test_artifacts_paths(tmp_path):
    a = Artifacts(str(tmp_path))
    assert a.root == tmp_path
    assert a.scaler.name == "claim_scaler.json" and a.ae_torch.name == "claim_ae.pt"
    assert a.ae_numpy.name == "claim_ae_weights.npz" and a.provider_model.name == "provider_xgb.json"
    assert a.provider_meta.name == "provider_meta.json" and a.thresholds.name == "thresholds.json"
    assert a.feature_context.name == "feature_context.json"


# ------------------------------------------------------------------------------ torch <-> numpy
def test_torch_autoencoder_shape():
    import torch.nn as nn

    m = build_torch_autoencoder(19)
    lin = [layer for layer in m if isinstance(layer, nn.Linear)]
    assert [(lin_.in_features, lin_.out_features) for lin_ in lin] == [(19, 14), (14, 7), (7, 14), (14, 19)]
    assert isinstance(m[-1], nn.Linear)                 # no activation on the output
    assert sum(isinstance(layer, nn.ReLU) for layer in m) == len(AE_HIDDEN)


def test_numpy_forward_equals_torch():
    import torch

    torch.manual_seed(3)
    m = build_torch_autoencoder(len(CLAIM_FEATURES)).eval()
    w = torch_to_numpy_weights(m)
    assert sorted(w) == ["W0", "W1", "W2", "W3", "b0", "b1", "b2", "b3"]
    assert all(v.dtype == np.float32 for v in w.values())
    Z = np.random.default_rng(1).normal(size=(64, len(CLAIM_FEATURES))).astype(np.float32) * 3
    with torch.no_grad():
        expected = m(torch.from_numpy(Z)).numpy()
    got = numpy_forward(w, Z)
    assert got.shape == Z.shape and got.dtype == np.float32
    np.testing.assert_allclose(got, expected, atol=1e-5)


# ------------------------------------------------------------------------------ anomaly backends
def test_auto_backend_falls_back_to_torch_cpu_without_nnpa(tiny_models_dir, caplog):
    with caplog.at_level(logging.WARNING, logger="claimsense.inference.anomaly"):
        s = load_anomaly_scorer(tiny_models_dir, backend="auto")
    assert s.name == "torch_cpu"
    assert any("nnpa" in r.getMessage() for r in caplog.records)


def test_nnpa_backend_falls_back_too(tiny_models_dir):
    assert load_anomaly_scorer(tiny_models_dir, backend="nnpa").name == "torch_cpu"


def test_cpu_backend(tiny_models_dir):
    s = load_anomaly_scorer(tiny_models_dir, backend="cpu")
    assert s.name == "torch_cpu" and s.nnpa_model is None


def test_auto_falls_back_to_numpy_without_torch_weights(fresh_models_dir):
    Artifacts(fresh_models_dir).ae_torch.unlink()
    assert load_anomaly_scorer(fresh_models_dir, backend="auto").name == "numpy"
    assert load_anomaly_scorer(fresh_models_dir, backend="cpu").name == "numpy"


def test_numpy_backend_needs_only_npz_and_scaler(tmp_path, tiny_models_dir):
    import shutil

    a, b = Artifacts(tiny_models_dir), Artifacts(tmp_path)
    shutil.copy(a.ae_numpy, b.ae_numpy)
    shutil.copy(a.scaler, b.scaler)
    s = load_anomaly_scorer(tmp_path, backend="numpy")
    assert s.name == "numpy"
    assert s.scaler.features == CLAIM_FEATURES


def test_no_backend_available_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no anomaly backend"):
        load_anomaly_scorer(tmp_path, backend="auto")


def test_unknown_backend_raises(tiny_models_dir):
    with pytest.raises((KeyError, ValueError)):
        load_anomaly_scorer(tiny_models_dir, backend="gpu")


def test_torch_and_numpy_scorers_agree(tiny_models_dir):
    t = load_anomaly_scorer(tiny_models_dir, backend="cpu")
    n = load_anomaly_scorer(tiny_models_dir, backend="numpy")
    X = np.random.default_rng(2).normal(size=(50, len(CLAIM_FEATURES))) * 4
    et, Zt, Rt = t.score(X)
    en, Zn, Rn = n.score(X)
    np.testing.assert_array_equal(Zt, Zn)
    np.testing.assert_allclose(Rt, Rn, atol=1e-5)
    np.testing.assert_allclose(et, en, atol=1e-5)


def test_score_shapes_and_error_definition(tiny_models_dir):
    s = load_anomaly_scorer(tiny_models_dir, backend="numpy")
    X = np.random.default_rng(3).normal(size=(7, len(CLAIM_FEATURES)))
    err, Z, R = s.score(X)
    assert err.shape == (7,) and err.dtype == np.float64
    assert Z.shape == R.shape == (7, len(CLAIM_FEATURES))
    np.testing.assert_allclose(err, ((Z - R) ** 2).mean(axis=1), rtol=1e-6)
    assert (err >= 0).all()
    err1, _, _ = s.score(X[0])                 # a single 1-D row is accepted
    assert err1.shape == (1,) and err1[0] == pytest.approx(err[0], rel=1e-6)


def test_torch_scorer_force_cpu(tiny_models_dir):
    s = load_anomaly_scorer(tiny_models_dir, backend="cpu")
    Z = np.zeros((3, len(CLAIM_FEATURES)), dtype=np.float32)
    np.testing.assert_array_equal(s.reconstruct(Z, force="cpu"), s.reconstruct(Z))


# ------------------------------------------------------------------------------ provider scorer
def test_provider_scorer_loads_and_predicts(tiny_models_dir):
    p = load_provider_scorer(tiny_models_dir)
    assert p.name == "xgboost"
    assert p.features == PROVIDER_FEATURES
    assert set(p.peer_quantiles) == set(PROVIDER_FEATURES)
    assert all(len(q) == 101 for q in p.peer_quantiles.values())
    X = np.random.default_rng(4).normal(size=(20, len(PROVIDER_FEATURES)))
    y = p.predict(X)
    assert y.shape == (20,) and y.dtype == np.float64
    assert ((y >= 0) & (y <= 1)).all()
    assert p.predict(X[0]).shape == (1,)
    assert p.predict(X[:1])[0] == pytest.approx(y[0])


def test_provider_contributions_sum_to_margin(tiny_models_dir):
    import xgboost as xgb

    p = load_provider_scorer(tiny_models_dir)
    x = np.random.default_rng(5).normal(size=len(PROVIDER_FEATURES)).astype(np.float32)
    c = p.contributions(x)
    assert c.shape == (len(PROVIDER_FEATURES),)
    full = p.booster.predict(xgb.DMatrix(x[None, :], feature_names=PROVIDER_FEATURES), pred_contribs=True)[0]
    prob = p.predict(x)[0]
    margin = np.log(prob / (1 - prob))
    assert float(c.sum() + full[-1]) == pytest.approx(margin, abs=1e-4)


def test_unknown_provider_backend(tiny_models_dir):
    with pytest.raises(ValueError):
        load_provider_scorer(tiny_models_dir, backend="onnx")
