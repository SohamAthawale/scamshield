"""explain.py: reasons without the LLM, templates, and Granite prompts that never carry raw identifiers."""
import math

import numpy as np
import pytest

from scamshield.explain import (
    SYSTEM_PROMPTS, analyst_brief_template, anomaly_reasons, build_messages, customer_warning_template,
    display_value, risk_reasons, rule_reasons,
)
from scamshield.features import ANOMALY_FEATURES, FEATURE_LABELS, FEATURES, RULES
from scamshield.security import Pseudonymizer


@pytest.mark.parametrize("feature,v,expected", [
    ("log_usd", math.log1p(5000), "$5,000"),
    ("fmt_ACH", 1.0, "yes"), ("fmt_Wire", 0.0, "no"),
    ("cross_bank", 1.0, "yes"), ("s_new_payee", 0.4, "no"), ("r_is_new", 0.5, "yes"),
    ("s_log_n_out", math.log1p(1234), "1,234"), ("r_log_senders", math.log1p(7), "7"),
    ("r_fan_in", 0.875, "88%"), ("r_share_ach_in", 0.0, "0%"),
    ("s_amount_vs_usual", math.log(12.0), "12.0x usual"),
    ("s_pass_through", math.log(0.97), "sends on 97% of what it receives"),
    ("s_log_idle_h", math.log1p(5), "5"), ("r_log_idle_h", math.log1p(96), "96 (about 4 days)"),
    ("s_recent_out", 3.2, "about 3 in the last day"),
    ("hour_sin", 0.5, ""), ("hour_cos", -1.0, ""),
    ("something_else", 1.234, "1.23"),
])
def test_display_value(feature, v, expected):
    assert display_value(feature, v) == expected


def test_every_feature_has_a_display_value():
    for f in FEATURES:
        assert isinstance(display_value(f, 0.5), str)


def _vectors(errs: dict, names=ANOMALY_FEATURES):
    z, r = np.zeros(len(names)), np.zeros(len(names))
    for f, (zv, rv) in errs.items():
        z[names.index(f)], r[names.index(f)] = zv, rv
    return z, r


def test_anomaly_reasons_top_features_and_direction():
    z, r = _vectors({"log_usd": (4.0, 0.0), "s_new_payee": (0.0, 3.0), "s_log_idle_h": (2.0, 0.0)})
    f = dict.fromkeys(ANOMALY_FEATURES, 0.0)
    f["log_usd"] = math.log1p(9000)
    out = anomaly_reasons(f, ANOMALY_FEATURES, z, r)
    assert [o["feature"] for o in out] == ["log_usd", "s_new_payee", "s_log_idle_h"]
    assert [o["direction"] for o in out] == ["higher than usual", "lower than usual", "higher than usual"]
    assert out[0]["label"] == FEATURE_LABELS["log_usd"] and out[0]["value"] == "$9,000"
    assert out[0]["share"] == round(16 / 29, 3)


def test_anomaly_reasons_merge_hour_into_one_time_of_day_reason():
    z, r = _vectors({"hour_sin": (3.0, 0.0), "hour_cos": (-2.9, 0.0), "log_usd": (2.0, 0.0)})
    out = anomaly_reasons(dict.fromkeys(ANOMALY_FEATURES, 0.0), ANOMALY_FEATURES, z, r)
    assert [o["feature"] for o in out] == ["hour", "log_usd"]
    assert out[0]["label"] == "time of day" and out[0]["value"] == ""
    assert sum(o["feature"] == "hour" for o in out) == 1


def test_anomaly_reasons_min_share_and_top_k():
    z, r = _vectors({"log_usd": (5.0, 0.0), "s_new_payee": (1.0, 0.0), "s_log_payees": (4.0, 0.0),
                     "cross_bank": (3.0, 0.0)})
    f = dict.fromkeys(ANOMALY_FEATURES, 0.0)
    # errors 25, 16, 9, 1 of 51: the 1 is < 10% and dropped; top_k=3 caps it anyway
    out = anomaly_reasons(f, ANOMALY_FEATURES, z, r)
    assert [o["feature"] for o in out] == ["log_usd", "s_log_payees", "cross_bank"]
    assert len(anomaly_reasons(f, ANOMALY_FEATURES, z, r, top_k=1)) == 1
    assert [o["feature"] for o in anomaly_reasons(f, ANOMALY_FEATURES, z, r, min_share=0.2)] == \
        ["log_usd", "s_log_payees"]


def test_anomaly_reasons_keep_the_top_one_and_survive_zero_error():
    n = len(ANOMALY_FEATURES)
    out = anomaly_reasons(dict.fromkeys(ANOMALY_FEATURES, 0.0), ANOMALY_FEATURES, np.ones(n), np.zeros(n))
    assert len(out) == 1                                   # every share is 1/n < 10%
    zero = anomaly_reasons(dict.fromkeys(ANOMALY_FEATURES, 0.0), ANOMALY_FEATURES, np.zeros(n), np.zeros(n))
    assert len(zero) == 1 and zero[0]["share"] == 0.0


def test_risk_reasons_positive_only_strongest_first_and_hour_merged():
    c = np.full(len(FEATURES) + 1, -0.2)                   # + bias column, which must be ignored
    c[-1] = 50.0
    c[FEATURES.index("r_fan_in")] = 1.5
    c[FEATURES.index("hour_cos")] = 0.9
    c[FEATURES.index("hour_sin")] = 0.8
    c[FEATURES.index("log_usd")] = 0.3
    f = dict.fromkeys(FEATURES, 0.0)
    f["r_fan_in"] = 0.9
    out = risk_reasons(f, FEATURES, c)
    assert [o["feature"] for o in out] == ["r_fan_in", "hour", "log_usd"]
    assert out[0]["value"] == "90%" and out[1]["label"] == "time of day" and out[1]["value"] == ""
    assert all(o["contribution"] > 0 for o in out)
    assert risk_reasons(f, FEATURES, -np.ones(len(FEATURES))) == []


def test_risk_reasons_top_k():
    c = np.arange(len(FEATURES), dtype=float) + 1
    assert len(risk_reasons(dict.fromkeys(FEATURES, 0.0), FEATURES, c)) == 4
    assert len(risk_reasons(dict.fromkeys(FEATURES, 0.0), FEATURES, c, top_k=2)) == 2


def test_rule_reasons():
    out = rule_reasons(["big_new_payee", "unknown_rule"])
    assert out == [{"rule": "big_new_payee", "title": RULES["big_new_payee"][0], "advice": RULES["big_new_payee"][1]}]
    assert rule_reasons([]) == []


REASONS = {
    "rules": rule_reasons(["big_new_payee"]),
    "customer": [{"feature": "log_usd", "label": "payment amount", "value": "$5,000", "direction": "higher than usual",
                  "share": 0.6},
                 {"feature": "hour", "label": "time of day", "value": "", "direction": "lower than usual", "share": 0.2}],
    "drivers": [{"feature": "r_fan_in", "label": FEATURE_LABELS["r_fan_in"], "value": "90%", "contribution": 1.2},
                {"feature": "hour", "label": "time of day", "value": "", "contribution": 0.4}],
}


def test_customer_warning_template_warn_and_hold():
    warn = customer_warning_template(REASONS, held=False)
    hold = customer_warning_template(REASONS, held=True)
    assert warn.startswith("Before you send this payment") and hold.startswith("We've paused this payment")
    for t in (warn, hold):
        assert RULES["big_new_payee"][0] in t and RULES["big_new_payee"][1] in t
        assert "The amount ($5,000) is larger than you usually send." in t   # plain-English customer phrasing
        assert "It's at a time of day when you don't usually make payments." in t
        assert "stop and call us" in t
        assert "risk" not in t.lower() and "90%" not in t          # model drivers never shown to the customer
    assert "never ask you to move money" in hold


def test_customer_warning_template_without_customer_reasons():
    t = customer_warning_template({"rules": [], "drivers": REASONS["drivers"]}, held=True)
    assert "patterns we often see in scam accounts" in t


def test_analyst_brief_template():
    b = analyst_brief_template(REASONS, "XPAYTOKEN", 0.912)
    assert b.startswith("Payment XPAYTOKEN: risk score 0.91 (model estimate, not a finding of fraud).")
    assert f"- {FEATURE_LABELS['r_fan_in']}: 90%" in b and "- time of day\n" in b
    assert f"- Rule: {RULES['big_new_payee'][0]}" in b and "call the customer" in b


def test_build_messages_kinds():
    facts = {**REASONS, "amount_usd": 5000.4, "payment_format": "Wire", "held": True, "risk": 0.91,
             "payment_token": "XPAYTOKEN"}
    w = build_messages("warning", facts)
    b = build_messages("brief", facts)
    assert [m["role"] for m in w] == ["system", "user"] and w[0]["content"] == SYSTEM_PROMPTS["warning"]
    assert "Payment amount: $5,000 by Wire" in w[1]["content"] and "paused for a call-back" in w[1]["content"]
    assert "risk score" not in w[1]["content"] and "Risk driver" not in w[1]["content"]
    assert "Payment XPAYTOKEN risk score: 0.91" in b[1]["content"] and "Risk driver: time of day" in b[1]["content"]
    assert "customer asked to confirm" in build_messages("warning", {})[1]["content"]
    with pytest.raises(ValueError):
        build_messages("poem", facts)
    for kind in ("warning", "brief"):
        assert "Use only the facts" in SYSTEM_PROMPTS[kind]
    assert "not a finding of fraud" in SYSTEM_PROMPTS["brief"]


def test_build_messages_never_leak_raw_identifiers():
    p = Pseudonymizer(b"k" * 32)
    raw = {"from_acct": "8C0FFEE01", "to_acct": "80BADBAD4", "from_bank": "0119", "to_bank": "021174"}
    facts = {
        **REASONS, "amount_usd": 5000.0, "payment_format": "ACH", "held": True, "risk": 0.9,
        "payment_token": p.token("payment", "T17"),
        # raw identifiers in unrelated fields must not reach the prompt
        **raw, "sender": "0119:8C0FFEE01", "receiver": "021174:80BADBAD4", "payload": dict(raw),
        "note": "customer 0119:8C0FFEE01 paying 80BADBAD4",
    }
    for kind in ("warning", "brief"):
        text = "\n".join(m["content"] for m in build_messages(kind, facts))
        for v in ("8C0FFEE01", "80BADBAD4", "021174", "0119"):
            assert v not in text, (kind, v)
