"""models.py, inference/anomaly.py and inference/risk.py on tiny artifacts built in conftest."""
import logging

import numpy as np
import pytest

from scamshield.features import ANOMALY_FEATURES, FEATURES
from scamshield.inference import load_anomaly_scorer, load_risk_scorer
from scamshield.models import (
    AE_HIDDEN, Z_CLIP, Artifacts, Standardizer, build_torch_autoencoder, numpy_forward, torch_to_numpy_weights,
)


@pytest.fixture
def models_dir(aml):
    return aml.models_dir


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
    assert a.scaler.name == "ae_scaler.json" and a.ae_torch.name == "payment_ae.pt"
    assert a.ae_numpy.name == "payment_ae_weights.npz" and a.risk_model.name == "risk_xgb.json"
    assert a.risk_meta.name == "risk_meta.json" and a.thresholds.name == "thresholds.json"
    assert a.feature_context.name == "feature_context.json"


# ------------------------------------------------------------------------------ torch <-> numpy
def test_torch_autoencoder_shape():
    import torch.nn as nn

    n = len(ANOMALY_FEATURES)
    m = build_torch_autoencoder(n)
    lin = [layer for layer in m if isinstance(layer, nn.Linear)]
    assert [(lin_.in_features, lin_.out_features) for lin_ in lin] == [(n, 16), (16, 8), (8, 16), (16, n)]
    assert isinstance(m[-1], nn.Linear)                 # no activation on the output
    assert sum(isinstance(layer, nn.ReLU) for layer in m) == len(AE_HIDDEN)


def test_numpy_forward_equals_torch():
    import torch

    torch.manual_seed(3)
    m = build_torch_autoencoder(len(ANOMALY_FEATURES)).eval()
    w = torch_to_numpy_weights(m)
    assert sorted(w) == ["W0", "W1", "W2", "W3", "b0", "b1", "b2", "b3"]
    assert all(v.dtype == np.float32 for v in w.values())
    Z = np.random.default_rng(1).normal(size=(64, len(ANOMALY_FEATURES))).astype(np.float32) * 3
    with torch.no_grad():
        expected = m(torch.from_numpy(Z)).numpy()
    got = numpy_forward(w, Z)
    assert got.shape == Z.shape and got.dtype == np.float32
    np.testing.assert_allclose(got, expected, atol=1e-5)


# ------------------------------------------------------------------------------ anomaly backends
def test_auto_backend_falls_back_to_torch_cpu_without_nnpa(models_dir, caplog):
    with caplog.at_level(logging.WARNING, logger="scamshield.inference.anomaly"):
        s = load_anomaly_scorer(models_dir, backend="auto")
    assert s.name == "torch_cpu"
    assert any("nnpa" in r.getMessage() for r in caplog.records)


def test_nnpa_backend_falls_back_too(models_dir):
    assert load_anomaly_scorer(models_dir, backend="nnpa").name == "torch_cpu"


def test_cpu_backend(models_dir):
    s = load_anomaly_scorer(models_dir, backend="cpu")
    assert s.name == "torch_cpu" and s.nnpa_model is None


def test_auto_falls_back_to_numpy_without_torch_weights(fresh_models_dir):
    Artifacts(fresh_models_dir).ae_torch.unlink()
    assert load_anomaly_scorer(fresh_models_dir, backend="auto").name == "numpy"
    assert load_anomaly_scorer(fresh_models_dir, backend="cpu").name == "numpy"


def test_numpy_backend_needs_only_npz_and_scaler(tmp_path, models_dir):
    import shutil

    a, b = Artifacts(models_dir), Artifacts(tmp_path)
    shutil.copy(a.ae_numpy, b.ae_numpy)
    shutil.copy(a.scaler, b.scaler)
    s = load_anomaly_scorer(tmp_path, backend="numpy")
    assert s.name == "numpy"
    assert s.scaler.features == ANOMALY_FEATURES


def test_no_backend_available_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no anomaly backend"):
        load_anomaly_scorer(tmp_path, backend="auto")


def test_unknown_backend_raises(models_dir):
    with pytest.raises((KeyError, ValueError)):
        load_anomaly_scorer(models_dir, backend="gpu")


def test_torch_and_numpy_scorers_agree(models_dir):
    t = load_anomaly_scorer(models_dir, backend="cpu")
    n = load_anomaly_scorer(models_dir, backend="numpy")
    X = np.random.default_rng(2).normal(size=(50, len(ANOMALY_FEATURES))) * 4
    et, Zt, Rt = t.score(X)
    en, Zn, Rn = n.score(X)
    np.testing.assert_array_equal(Zt, Zn)
    np.testing.assert_allclose(Rt, Rn, atol=1e-5)
    np.testing.assert_allclose(et, en, atol=1e-5)


def test_score_shapes_and_error_definition(models_dir):
    s = load_anomaly_scorer(models_dir, backend="numpy")
    X = np.random.default_rng(3).normal(size=(7, len(ANOMALY_FEATURES)))
    err, Z, R = s.score(X)
    assert err.shape == (7,) and err.dtype == np.float64
    assert Z.shape == R.shape == (7, len(ANOMALY_FEATURES))
    np.testing.assert_allclose(err, ((Z - R) ** 2).mean(axis=1), rtol=1e-6)
    assert (err >= 0).all()
    err1, _, _ = s.score(X[0])                 # a single 1-D row is accepted
    assert err1.shape == (1,) and err1[0] == pytest.approx(err[0], rel=1e-6)


def test_torch_scorer_force_cpu(models_dir):
    s = load_anomaly_scorer(models_dir, backend="cpu")
    Z = np.zeros((3, len(ANOMALY_FEATURES)), dtype=np.float32)
    np.testing.assert_array_equal(s.reconstruct(Z, force="cpu"), s.reconstruct(Z))


# ------------------------------------------------------------------------------ risk scorer
def test_risk_scorer_loads_and_predicts(models_dir):
    p = load_risk_scorer(models_dir)
    assert p.name == "xgboost"
    assert p.features == FEATURES
    X = np.random.default_rng(4).normal(size=(20, len(FEATURES)))
    y = p.predict(X)
    assert y.shape == (20,) and y.dtype == np.float64
    assert ((y >= 0) & (y <= 1)).all()
    assert p.predict(X[0]).shape == (1,)
    assert p.predict(X[:1])[0] == pytest.approx(y[0])


def test_risk_contributions_sum_to_margin(models_dir):
    import xgboost as xgb

    p = load_risk_scorer(models_dir)
    x = np.random.default_rng(5).normal(size=len(FEATURES)).astype(np.float32)
    c = p.contributions(x)
    assert c.shape == (len(FEATURES),)
    full = p.booster.predict(xgb.DMatrix(x[None, :], feature_names=FEATURES), pred_contribs=True)[0]
    prob = p.predict(x)[0]
    margin = np.log(prob / (1 - prob))
    assert float(c.sum() + full[-1]) == pytest.approx(margin, abs=1e-4)


def test_unknown_risk_backend(models_dir):
    with pytest.raises(ValueError):
        load_risk_scorer(models_dir, backend="onnx")


def test_risk_scorer_matches_saved_booster(models_dir, aml):
    import xgboost as xgb

    b = xgb.Booster()
    b.load_model(str(models_dir / "risk_xgb.json"))
    p = load_risk_scorer(models_dir)
    np.testing.assert_array_equal(p.predict(aml.X), b.inplace_predict(aml.X).astype(np.float64))
    assert b.feature_names == FEATURES


def test_anomaly_scaler_is_over_anomaly_features(models_dir):
    s = load_anomaly_scorer(models_dir, backend="numpy")
    assert s.scaler.features == ANOMALY_FEATURES
    assert len(s.weights["W0"][0]) == len(ANOMALY_FEATURES)
