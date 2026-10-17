"""explain.py: rule-based reasons, fallback templates and the Granite prompt (no raw identifiers)."""
import math

import numpy as np
import pytest

from claimsense import explain
from claimsense.explain import (
    _pct, build_messages, claim_reasons, correction_note_template, display_value, investigator_brief_template,
    percentile, provider_reasons, rule_reasons,
)
from claimsense.features import CLAIM_FEATURES, FEATURE_LABELS, HARD_RULES, PROVIDER_FEATURES, PROVIDER_LABELS
from claimsense.security import Pseudonymizer


@pytest.mark.parametrize("feature,v,expected", [
    ("log_amount", math.log1p(26000), "$26,000"),
    ("mean_log_deductible", math.log1p(1068), "$1,068"),
    ("log_amount", 0.0, "$0"),
    ("log_claims", math.log1p(1234), "1,234"),
    ("log_unique_benes", math.log1p(7), "7"),
    ("inpatient_share", 0.256, "25.6%"),
    ("missing_attending_share", 1.0, "100.0%"),
    ("is_inpatient", 1.0, "yes"),
    ("att_is_op", 0.0, "no"),
    ("admit_dx_matches", 0.5, "yes"),
    ("amount_vs_dx", math.log(2.0), "2.0x typical"),
    ("mean_amount_vs_dx", math.log(0.5), "0.5x typical"),
    ("dx_rarity", math.log(1000), "about 1 in 1,000 claims"),
    ("dx_px_rarity", -3.0, "about 1 in 1 claims"),
    ("age", 70.26, "70.3"),
    ("claim_days", 3.0, "3.0"),
    ("unknown_feature", 1.234, "1.2"),
])
def test_display_value(feature, v, expected):
    assert display_value(feature, v) == expected


def test_every_feature_has_a_display_value():
    for f in CLAIM_FEATURES + PROVIDER_FEATURES:
        assert isinstance(display_value(f, 0.5), str)


QUANTS = sorted(np.quantile(np.random.default_rng(0).normal(size=1000), np.linspace(0, 1, 101)).tolist())


def test_percentile_bounds():
    assert percentile(-1e9, QUANTS) == 0.0
    assert percentile(1e9, QUANTS) == 100.0
    assert percentile(QUANTS[50] + 1e-12, QUANTS) == pytest.approx(51.0, abs=1)


def test_percentile_is_monotonic_and_bounded():
    vals = np.linspace(min(QUANTS) - 1, max(QUANTS) + 1, 500)
    ps = [percentile(v, QUANTS) for v in vals]
    assert all(0.0 <= p <= 100.0 for p in ps)
    assert all(b >= a for a, b in zip(ps, ps[1:]))


def test_percentile_with_flat_distribution():
    q = [0.0] * 90 + [1.0] * 11
    assert 0 <= percentile(0.0, q) <= 100
    assert percentile(0.0, q) == 45.0     # midpoint of the tied block
    assert percentile(-1, q) == 0.0 and percentile(2, q) == 100.0


def _errors(pairs):
    """{feature: (x_std, recon_std)} -> full-length x, r vectors over CLAIM_FEATURES."""
    names = CLAIM_FEATURES
    x, r = np.zeros(len(names)), np.zeros(len(names))
    for f, (xv, rv) in pairs.items():
        x[names.index(f)], r[names.index(f)] = xv, rv
    return names, x, r


def test_claim_reasons_picks_largest_errors_with_direction():
    # errors 25, 9, 4 (+0.01): shares 0.65, 0.23, 0.10x -> all three above min_share
    names, x, r = _errors({"log_amount": (5.0, 0.0), "age": (-1.0, 2.0), "claim_days": (0.0, 2.0),
                           "n_diag": (0.1, 0.0)})
    feats = {f: 0.0 for f in names}
    feats["log_amount"] = math.log1p(26000)
    out = claim_reasons(feats, names, x, r, top_k=3)
    assert [o["feature"] for o in out] == ["log_amount", "age", "claim_days"]
    assert [o["direction"] for o in out] == ["higher than usual", "lower than usual", "lower than usual"]
    assert out[0]["label"] == FEATURE_LABELS["log_amount"]
    assert out[0]["value"] == "$26,000"
    total = 25 + 9 + 4 + 0.01
    assert out[0]["share_of_anomaly"] == round(25 / total, 3)
    assert sum(o["share_of_anomaly"] for o in out) <= 1.0
    assert len(claim_reasons(feats, names, x, r, top_k=1)) == 1


def test_claim_reasons_drops_features_below_min_share():
    # errors 25, 9, 1: claim_days carries 1/35 < 10% of the error and is dropped
    names, x, r = _errors({"log_amount": (5.0, 0.0), "age": (-1.0, 2.0), "claim_days": (0.0, 1.0)})
    feats = {f: 0.0 for f in names}
    assert [o["feature"] for o in claim_reasons(feats, names, x, r)] == ["log_amount", "age"]
    assert [o["feature"] for o in claim_reasons(feats, names, x, r, min_share=0.0)] == \
        ["log_amount", "age", "claim_days"]
    assert [o["feature"] for o in claim_reasons(feats, names, x, r, min_share=0.5)] == ["log_amount"]


def test_claim_reasons_keeps_top_one_even_below_min_share():
    # 19 features with equal error: each has ~5% share, below 10%, but the top one is always kept
    names = CLAIM_FEATURES
    x, r = np.ones(len(names)), np.zeros(len(names))
    out = claim_reasons({f: 0.0 for f in names}, names, x, r)
    assert len(out) == 1
    assert out[0]["share_of_anomaly"] == round(1 / len(names), 3)


def test_claim_reasons_min_share_boundary_is_inclusive():
    # errors 16, 16, 4, 4 (total 40) -> the third reason carries exactly 10%
    names, x, r = _errors({"log_amount": (4.0, 0.0), "age": (-4.0, 0.0), "claim_days": (2.0, 0.0),
                           "n_diag": (2.0, 0.0)})
    out = claim_reasons({f: 0.0 for f in names}, names, x, r)
    assert len(out) == 3 and out[2]["share_of_anomaly"] == 0.1


def test_claim_reasons_zero_error_does_not_divide_by_zero():
    names = CLAIM_FEATURES
    z = np.zeros(len(names))
    out = claim_reasons({f: 0.0 for f in names}, names, z, z)
    assert len(out) == 1 and out[0]["share_of_anomaly"] == 0.0


def test_provider_reasons_only_positive_contributions():
    names = PROVIDER_FEATURES
    contrib = np.full(len(names) + 1, -0.1)       # last entry = XGBoost bias column
    contrib[-1] = 99.0
    contrib[names.index("claims_per_bene")] = 0.8
    contrib[names.index("rare_pair_share")] = 0.3
    feats = {f: 0.5 for f in names}
    pq = {"claims_per_bene": QUANTS}
    out = provider_reasons(feats, names, contrib, pq, top_k=3)
    assert [o["feature"] for o in out] == ["claims_per_bene", "rare_pair_share"]
    assert all(o["contribution"] > 0 for o in out)
    assert out[0]["label"] == PROVIDER_LABELS["claims_per_bene"]
    assert out[0]["peer_percentile"] == round(percentile(0.5, QUANTS), 1)
    assert out[1]["peer_percentile"] is None
    assert out[1]["value"] == "50.0%"


def test_provider_reasons_all_negative_is_empty():
    names = PROVIDER_FEATURES
    assert provider_reasons({f: 0.0 for f in names}, names, -np.ones(len(names)), {}) == []


def test_rule_reasons():
    out = rule_reasons(["missing_attending", "not_a_rule", "duplicate"])
    assert [r["rule"] for r in out] == ["missing_attending", "duplicate"]
    assert out[0]["title"] == HARD_RULES["missing_attending"][0]
    assert out[0]["fix"] == HARD_RULES["missing_attending"][1]
    assert rule_reasons([]) == []


@pytest.mark.parametrize("p,expected", [
    (None, ""), (1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"), (13, "13th"),
    (21, "21st"), (22, "22nd"), (23, "23rd"), (99.6, "100th"), (0, "0th"), (111, "111th"), (112, "112th"),
])
def test_pct_ordinals(p, expected):
    out = _pct(p)
    assert out == ("" if p is None else f"{expected} percentile of peers")


@pytest.fixture
def reasons():
    return {
        "rules": rule_reasons(["end_before_start", "duplicate"]),
        "claim": [{"feature": "log_amount", "label": "reimbursed amount", "value": "$26,000",
                   "direction": "higher than usual", "share_of_anomaly": 0.7}],
        "provider": [{"feature": "claims_per_bene", "label": "claims per patient", "value": "9.0",
                      "peer_percentile": 97.0, "contribution": 0.8}],
    }


def test_correction_note_template_contains_fixes(reasons):
    note = correction_note_template(reasons)
    for code in ["end_before_start", "duplicate"]:
        assert HARD_RULES[code][0] in note
        assert HARD_RULES[code][1] in note
    assert "reimbursed amount ($26,000) is higher than usual" in note
    assert "claims per patient" not in note          # provider pattern never goes to the provider


def test_correction_note_template_empty_reasons():
    note = correction_note_template({})
    assert note.startswith("We could not process") and "reviewer" in note


def test_investigator_brief_template(reasons):
    b = investigator_brief_template(reasons, "PABCDEF", 0.873)
    assert "Provider PABCDEF: risk score 0.87" in b
    assert "not a finding of fraud" in b
    assert "claims per patient: 9.0 (97th percentile of peers)" in b
    assert HARD_RULES["duplicate"][0] in b
    assert "reimbursed amount: $26,000 (higher than usual)" in b
    nb = investigator_brief_template({}, "PABCDEF", None)
    assert "not enough history" in nb and "This claim:" not in nb


def test_build_messages_shape_and_contents(reasons):
    facts = {**reasons, "provider_token": "PTOKEN", "provider_risk": 0.91, "amount": 26000.4}
    corr = build_messages("correction", facts)
    assert [m["role"] for m in corr] == ["system", "user"]
    assert corr[0]["content"] == explain.SYSTEM_PROMPTS["correction"]
    assert HARD_RULES["duplicate"][1] in corr[1]["content"]
    assert "Claim amount: $26,000" in corr[1]["content"]
    assert "PTOKEN" not in corr[1]["content"]          # correction notes never discuss the provider score
    brief = build_messages("brief", facts)
    assert "Provider PTOKEN risk score: 0.91" in brief[1]["content"]
    assert "(97th percentile of peers)" in brief[1]["content"]
    with pytest.raises(ValueError):
        build_messages("poem", facts)


def test_build_messages_never_leaks_raw_identifiers(reasons):
    p = Pseudonymizer(b"k" * 32)
    raw = {"bene": "BENE11001", "provider": "PRV51001", "physician": "PHY390922", "claim": "CLM46614"}
    facts = {
        **reasons,
        "provider_token": p.token("provider", raw["provider"]),
        "claim_token": p.token("claim", raw["claim"]),
        "provider_risk": 0.75,
        "amount": 1234.0,
        # raw identifiers sitting in unrelated fields must not reach the prompt
        "BeneID": raw["bene"], "Provider": raw["provider"], "AttendingPhysician": raw["physician"],
        "ClaimID": raw["claim"], "raw": dict(raw), "notes": f"seen {raw['bene']} at {raw['provider']}",
        "DOB": "1943-01-01",
    }
    for kind in ("correction", "brief"):
        text = "\n".join(m["content"] for m in build_messages(kind, facts))
        for v in raw.values():
            assert v not in text, (kind, v)
        for prefix in ("BENE", "PRV", "PHY", "CLM", "1943"):
            assert prefix not in text, (kind, prefix)
    assert facts["provider_token"] in "\n".join(m["content"] for m in build_messages("brief", facts))
