"""fusion.py: every branch of decide() and the Policy merge rules."""
import json

import pytest

from claimsense.fusion import ACTIONS, HOLD, PAY, RETURN, Decision, Policy, decide

POLICY = Policy(anomaly_threshold=1.0, provider_threshold=0.5, min_provider_claims=10)


@pytest.mark.parametrize("anomaly,risk,n,errs,action", [
    (0.1, 0.1, 50, [], PAY),                         # nothing unusual
    (0.1, 0.9, 50, [], PAY),                         # risky provider alone never blocks a normal claim
    (2.0, 0.1, 50, [], RETURN),                      # unusual claim, normal provider
    (0.1, 0.1, 50, ["missing_attending"], RETURN),   # rule broken, normal provider
    (2.0, 0.9, 50, [], HOLD),                        # unusual claim AND risky provider
    (0.1, 0.9, 50, ["duplicate"], HOLD),             # rule broken AND risky provider
    (2.0, 0.9, 9, [], RETURN),                       # risky but not enough history -> honest mistake
    (2.0, None, 50, [], RETURN),                     # no provider score at all
    (0.1, None, 50, [], PAY),
])
def test_decide_branches(anomaly, risk, n, errs, action):
    d = decide(anomaly, risk, n, errs, POLICY)
    assert d.action == action
    assert d.action in ACTIONS


def test_anomaly_threshold_is_inclusive():
    assert decide(1.0, 0.0, 50, [], POLICY).anomaly_flag is True
    assert decide(1.0, 0.0, 50, [], POLICY).action == RETURN
    assert decide(0.999999, 0.0, 50, [], POLICY).action == PAY


def test_provider_threshold_is_inclusive():
    assert decide(2.0, 0.5, 50, [], POLICY).action == HOLD
    assert decide(2.0, 0.4999, 50, [], POLICY).action == RETURN


def test_min_provider_claims_is_inclusive():
    d = decide(2.0, 0.9, 10, [], POLICY)
    assert d.provider_ready and d.provider_flag and d.action == HOLD
    d = decide(2.0, 0.9, 9, [], POLICY)
    assert not d.provider_ready and not d.provider_flag and d.action == RETURN


def test_provider_risk_none_is_not_ready():
    d = decide(2.0, None, 1000, ["after_death"], POLICY)
    assert d.provider_risk is None and not d.provider_ready and not d.provider_flag
    assert d.action == RETURN


def test_decision_fields_and_to_dict():
    errs = ["end_before_start"]
    d = decide(3, 0.7, 12, errs, POLICY)
    assert isinstance(d.anomaly, float) and d.anomaly == 3.0
    assert d.hard_errors == errs and d.hard_errors is not errs   # defensive copy
    assert d.to_dict() == {"action": HOLD, "anomaly": 3.0, "anomaly_flag": True, "provider_risk": 0.7,
                           "provider_ready": True, "provider_flag": True, "hard_errors": errs}
    json.dumps(d.to_dict())
    assert isinstance(d, Decision)


def test_hard_errors_alone_do_not_set_anomaly_flag():
    d = decide(0.0, 0.0, 50, ["missing_attending"], POLICY)
    assert d.anomaly_flag is False and d.action == RETURN


# ------------------------------------------------------------------------------ Policy.load
def write(path, obj):
    path.write_text(json.dumps(obj))
    return path


def test_policy_load_learned_values_when_policy_is_null(tmp_path):
    th = write(tmp_path / "thresholds.json", {"anomaly_threshold": 0.8, "provider_threshold": 0.3,
                                              "min_provider_claims": 7})
    pol = write(tmp_path / "policy.json", {"anomaly_threshold": None, "provider_threshold": None,
                                           "min_provider_claims": None})
    assert Policy.load(pol, th) == Policy(0.8, 0.3, 7)


def test_policy_load_manual_values_override_learned(tmp_path):
    th = write(tmp_path / "thresholds.json", {"anomaly_threshold": 0.8, "provider_threshold": 0.3,
                                              "min_provider_claims": 7})
    pol = write(tmp_path / "policy.json", {"anomaly_threshold": 1.5, "provider_threshold": None,
                                           "min_provider_claims": 20})
    assert Policy.load(pol, th) == Policy(1.5, 0.3, 20)


def test_policy_load_zero_is_an_override_not_null(tmp_path):
    th = write(tmp_path / "thresholds.json", {"anomaly_threshold": 0.8, "provider_threshold": 0.3})
    pol = write(tmp_path / "policy.json", {"anomaly_threshold": 0, "provider_threshold": 0.0})
    p = Policy.load(pol, th)
    assert p.anomaly_threshold == 0.0 and p.provider_threshold == 0.0


def test_policy_load_without_thresholds_file(tmp_path):
    pol = write(tmp_path / "policy.json", {"anomaly_threshold": 1, "provider_threshold": "0.5"})
    p = Policy.load(pol)
    assert p == Policy(1.0, 0.5, 10)
    assert isinstance(p.anomaly_threshold, float) and isinstance(p.min_provider_claims, int)


def test_policy_load_default_min_claims(tmp_path):
    th = write(tmp_path / "thresholds.json", {"anomaly_threshold": 0.8, "provider_threshold": 0.3})
    pol = write(tmp_path / "policy.json", {"min_provider_claims": None})
    assert Policy.load(pol, th).min_provider_claims == 10


def test_policy_load_missing_value_everywhere_raises(tmp_path):
    th = write(tmp_path / "thresholds.json", {"anomaly_threshold": 0.8})
    pol = write(tmp_path / "policy.json", {"provider_threshold": None})
    with pytest.raises(KeyError):
        Policy.load(pol, th)


def test_policy_load_ignores_extra_keys(tmp_path):
    th = write(tmp_path / "thresholds.json", {"anomaly_threshold": 0.8, "provider_threshold": 0.3,
                                              "val_precision": 0.9})
    pol = write(tmp_path / "policy.json", {"comment": "hand-set", "anomaly_threshold": None})
    assert Policy.load(pol, th) == Policy(0.8, 0.3, 10)
