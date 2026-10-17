"""fusion.py: ALLOW / WARN / HOLD, threshold edges, Policy validation and merge rules."""
import itertools
import json

import numpy as np
import pytest

from scamshield.fusion import ACTIONS, ALLOW, HOLD, WARN, Decision, Policy, decide
from scamshield.training import decisions

POL = Policy(anomaly_threshold=1.0, warn_threshold=0.3, hold_threshold=0.8)


@pytest.mark.parametrize("anomaly,risk,rules,action", [
    (0.0, 0.0, [], ALLOW),
    (0.0, 0.29, [], ALLOW),
    (0.0, 0.3, [], WARN),                      # warn threshold inclusive
    (0.0, 0.79, [], WARN),
    (0.0, 0.8, [], HOLD),                      # hold threshold inclusive
    (5.0, 0.99, ["big_new_payee"], HOLD),      # hold wins over everything
    (1.0, 0.0, [], WARN),                      # anomaly threshold inclusive
    (0.99, 0.0, [], ALLOW),
    (0.0, 0.0, ["big_new_payee"], WARN),       # any rule warns
    (0.0, 0.0, ["anything"], WARN),
    (5.0, 0.5, ["big_new_payee"], WARN),
])
def test_decide(anomaly, risk, rules, action):
    d = decide(anomaly, risk, rules, POL)
    assert d.action == action and d.action in ACTIONS


def test_decision_fields():
    rules = ["big_new_payee"]
    d = decide(2, 0.5, rules, POL)
    assert d == Decision(WARN, 0.5, 2.0, True, ["big_new_payee"])
    assert d.rules is not rules and isinstance(d.risk, float) and isinstance(d.anomaly, float)
    assert d.to_dict() == {"action": WARN, "risk": 0.5, "anomaly": 2.0, "anomaly_flag": True,
                           "rules": ["big_new_payee"]}
    json.dumps(d.to_dict())
    assert decide(0, 0.9, [], POL).anomaly_flag is False


def test_policy_rejects_warn_above_hold():
    with pytest.raises(ValueError):
        Policy(1.0, 0.9, 0.8)
    Policy(1.0, 0.8, 0.8)                         # equal is allowed: no risk-only warnings


def test_equal_warn_and_hold_means_risk_alone_never_warns():
    p = Policy(1.0, 0.5, 0.5)
    assert decide(0, 0.49, [], p).action == ALLOW and decide(0, 0.5, [], p).action == HOLD


def test_vectorised_decisions_equal_decide_on_a_grid():
    """training.decisions (used to tune and report) must agree with fusion.decide (used live)."""
    risks = [0.0, 0.29999, 0.3, 0.5, 0.79999, 0.8, 1.0]
    anomalies = [0.0, 0.99, 1.0, 3.0]
    combos = list(itertools.product(risks, anomalies, [False, True]))
    r = np.array([c[0] for c in combos])
    a = np.array([c[1] for c in combos])
    ru = np.array([c[2] for c in combos])
    vec = decisions(r, a, ru, POL)
    for (risk, anom, rule), v in zip(combos, vec):
        assert v == decide(anom, risk, ["big_new_payee"] if rule else [], POL).action


@pytest.mark.parametrize("seed", range(3))
def test_vectorised_decisions_equal_decide_random(seed):
    rng = np.random.default_rng(seed)
    w, h = sorted(rng.random(2))
    pol = Policy(float(rng.random() * 2), float(w), float(h))
    r, a, ru = rng.random(500), rng.random(500) * 2, rng.random(500) < 0.1
    vec = decisions(r, a, ru, pol)
    assert all(v == decide(float(ai), float(ri), ["x"] if ui else [], pol).action
               for v, ri, ai, ui in zip(vec, r, a, ru))


# ------------------------------------------------------------------------------ Policy.load
def write(p, obj):
    p.write_text(json.dumps(obj))
    return p


def test_policy_load_null_means_learned(tmp_path):
    th = write(tmp_path / "thresholds.json", {"anomaly_threshold": 2.0, "warn_threshold": 0.2, "hold_threshold": 0.9,
                                              "hold_cap": 0.002})
    pol = write(tmp_path / "policy.json", {"anomaly_threshold": None, "warn_threshold": None, "hold_threshold": None})
    assert Policy.load(pol, th) == Policy(2.0, 0.2, 0.9)


def test_policy_load_manual_overrides_and_comments_ignored(tmp_path):
    th = write(tmp_path / "thresholds.json", {"anomaly_threshold": 2.0, "warn_threshold": 0.2, "hold_threshold": 0.9})
    pol = write(tmp_path / "policy.json", {"_comment": "hand-set", "_hold_threshold": 0.0, "warn_threshold": 0.5,
                                           "hold_threshold": None, "anomaly_threshold": 0})
    assert Policy.load(pol, th) == Policy(0.0, 0.5, 0.9)


def test_policy_load_without_thresholds_file(tmp_path):
    pol = write(tmp_path / "policy.json", {"anomaly_threshold": 1, "warn_threshold": "0.25", "hold_threshold": 0.75})
    p = Policy.load(pol)
    assert p == Policy(1.0, 0.25, 0.75) and isinstance(p.anomaly_threshold, float)


def test_policy_load_missing_value_raises(tmp_path):
    th = write(tmp_path / "thresholds.json", {"anomaly_threshold": 2.0, "warn_threshold": 0.2})
    pol = write(tmp_path / "policy.json", {"hold_threshold": None})
    with pytest.raises(KeyError):
        Policy.load(pol, th)


def test_policy_load_validates_merged_values(tmp_path):
    th = write(tmp_path / "thresholds.json", {"anomaly_threshold": 2.0, "warn_threshold": 0.2, "hold_threshold": 0.9})
    pol = write(tmp_path / "policy.json", {"warn_threshold": 0.95})
    with pytest.raises(ValueError):
        Policy.load(pol, th)


def test_repo_policy_file_loads_with_learned_thresholds(tmp_path):
    from pathlib import Path

    repo_policy = Path(__file__).resolve().parents[1] / "config" / "policy.json"
    if not repo_policy.exists():
        pytest.skip("config/policy.json not present")
    th = write(tmp_path / "thresholds.json", {"anomaly_threshold": 2.0, "warn_threshold": 0.2, "hold_threshold": 0.9})
    p = Policy.load(repo_policy, th)
    assert p.warn_threshold <= p.hold_threshold


# ------------------------------------------------------------------------------ value-aware hold
VPOL = Policy(anomaly_threshold=1.0, warn_threshold=0.3, hold_threshold=0.8, hold_strict_threshold=0.9,
              hold_min_usd=20_000)


@pytest.mark.parametrize("risk,usd,action", [
    (0.95, 1.0, HOLD),                         # above strict: held whatever the amount
    (0.9, 0.0, HOLD),                          # strict threshold inclusive
    (0.85, 20_000, HOLD),                      # in the band and big enough (amount inclusive)
    (0.8, 1e9, HOLD),                          # hold threshold still inclusive in the band
    (0.85, 19_999.99, WARN),                   # in the band but small: warned, not held
    (0.8, 0.0, WARN),
    (0.79, 1e9, WARN),                         # below the band: size alone never holds
    (0.29, 1e9, ALLOW),
])
def test_value_aware_decide(risk, usd, action):
    assert decide(0.0, risk, [], VPOL, usd).action == action


def test_value_aware_small_band_payment_with_rule_or_anomaly_is_warned():
    assert decide(5.0, 0.85, ["big_new_payee"], VPOL, 10.0).action == WARN


def test_value_aware_needs_the_amount_only_inside_the_band():
    assert decide(0, 0.95, [], VPOL).action == HOLD
    assert decide(0, 0.5, [], VPOL).action == WARN
    with pytest.raises(ValueError):
        decide(0, 0.85, [], VPOL)
    with pytest.raises(ValueError):
        decisions(np.array([0.85]), np.array([0.0]), np.array([False]), VPOL)


def test_plain_policy_ignores_the_amount():
    assert not POL.value_aware and VPOL.value_aware
    for usd in (None, 0.0, 1e9):
        assert decide(0, 0.8, [], POL, usd).action == HOLD and decide(0, 0.79, [], POL, usd).action == WARN


def test_strict_equal_to_hold_is_the_plain_policy():
    p = Policy(1.0, 0.3, 0.8, hold_strict_threshold=0.8, hold_min_usd=1e12)
    for risk in (0.0, 0.3, 0.79, 0.8, 0.95):
        assert decide(0, risk, [], p, 1.0).action == decide(0, risk, [], POL).action


def test_value_aware_policy_validation():
    with pytest.raises(ValueError):
        Policy(1.0, 0.3, 0.8, hold_strict_threshold=0.9)             # amount missing
    with pytest.raises(ValueError):
        Policy(1.0, 0.3, 0.8, hold_min_usd=1000)                     # strict missing
    with pytest.raises(ValueError):
        Policy(1.0, 0.3, 0.8, hold_strict_threshold=0.7, hold_min_usd=1000)


@pytest.mark.parametrize("seed", range(3))
def test_vectorised_value_aware_decisions_equal_decide(seed):
    rng = np.random.default_rng(seed)
    w, h, s = sorted(rng.random(3))
    pol = Policy(float(rng.random() * 2), float(w), float(h), float(s), float(rng.choice([10.0, 1e3, 1e5])))
    n = 2000
    r, a, ru = rng.random(n), rng.random(n) * 2, rng.random(n) < 0.1
    usd = 10 ** rng.uniform(0, 7, n)
    r[:20], usd[:20] = h, pol.hold_min_usd                             # exact edges
    r[20:40] = s
    vec = decisions(r, a, ru, pol, usd)
    assert all(v == decide(float(ai), float(ri), ["x"] if ui else [], pol, float(ui_usd)).action
               for v, ri, ai, ui, ui_usd in zip(vec, r, a, ru, usd))
    assert {HOLD, WARN} <= set(vec[(r >= h) & (r < s)])


def test_policy_load_value_aware_from_learned_and_manual(tmp_path):
    th = write(tmp_path / "thresholds.json", {"anomaly_threshold": 2.0, "warn_threshold": 0.2, "hold_threshold": 0.7,
                                              "hold_strict_threshold": 0.85, "hold_min_usd": 20000})
    nulls = {"anomaly_threshold": None, "warn_threshold": None, "hold_threshold": None,
             "hold_strict_threshold": None, "hold_min_usd": None}
    assert Policy.load(write(tmp_path / "p1.json", nulls), th) == Policy(2.0, 0.2, 0.7, 0.85, 20000.0)
    p2 = write(tmp_path / "p2.json", {**nulls, "hold_min_usd": "50000"})
    assert Policy.load(p2, th).hold_min_usd == 50000.0
    # switching it off by hand: strict = hold
    p3 = write(tmp_path / "p3.json", {**nulls, "hold_strict_threshold": 0.7})
    assert Policy.load(p3, th).hold_strict_threshold == 0.7
    # a hand-set hold threshold above the learned strict one lifts strict with it
    p4 = write(tmp_path / "p4.json", {**nulls, "hold_threshold": 0.95})
    assert Policy.load(p4, th) == Policy(2.0, 0.2, 0.95, 0.95, 20000.0)


def test_policy_load_old_thresholds_file_is_plain(tmp_path):
    th = write(tmp_path / "thresholds.json", {"anomaly_threshold": 2.0, "warn_threshold": 0.2, "hold_threshold": 0.9})
    pol = write(tmp_path / "policy.json", {"hold_strict_threshold": None, "hold_min_usd": None})
    assert not Policy.load(pol, th).value_aware
