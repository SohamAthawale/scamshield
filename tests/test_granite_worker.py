"""granite_worker.py with a fake llama.cpp model (no real model is loaded)."""
import sys
import types

import pytest

from scamshield.explain import build_messages, rule_reasons
from scamshield.granite_worker import DISCLAIMER, RAW_ID, GraniteWriter, trim_to_sentence

FACTS = {"rules": rule_reasons(["big_new_payee"]), "customer": [], "drivers": [], "amount_usd": 5000.0,
         "payment_format": "Wire", "held": True, "risk": 0.93, "payment_token": "XABCDEFGHIJKLMNOP"}


class FakeLlama:
    def __init__(self, reply="All good.", finish_reason="stop", **kwargs):
        self.kwargs, self.reply, self.finish_reason, self.calls = kwargs, reply, finish_reason, []

    def create_chat_completion(self, messages, temperature, max_tokens):
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        return {"choices": [{"message": {"role": "assistant", "content": self.reply},
                             "finish_reason": self.finish_reason}]}


def writer(reply, finish_reason="stop", max_tokens=120) -> GraniteWriter:
    w = GraniteWriter.__new__(GraniteWriter)
    w.llm, w.model, w.max_tokens = FakeLlama(reply, finish_reason), "fake.gguf", max_tokens
    return w


def test_init_uses_llama_cpp(monkeypatch, tmp_path):
    fake = types.ModuleType("llama_cpp")
    fake.Llama = lambda **kw: FakeLlama(**kw)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    w = GraniteWriter(str(tmp_path / "granite-3.3-2b.gguf"), threads=3, ctx=1024, max_tokens=99)
    assert w.model == "granite-3.3-2b.gguf" and w.max_tokens == 99
    assert w.llm.kwargs == {"model_path": str(tmp_path / "granite-3.3-2b.gguf"), "n_ctx": 1024, "n_threads": 3,
                            "verbose": False}


def test_write_sends_the_pseudonymised_prompt():
    w = writer("Please check the payee's details before you send this.")
    assert w.write("warning", FACTS) == "Please check the payee's details before you send this."
    call = w.llm.calls[0]
    assert call["messages"] == build_messages("warning", FACTS)
    assert call["temperature"] == 0.2 and call["max_tokens"] == 120


@pytest.mark.parametrize("raw", ["8000EBD30", "80BADBAD4", "8C0FFEE01", "NEW0000001", "paid 8ABCDEF12."])
def test_completion_with_a_raw_account_number_raises(raw):
    with pytest.raises(ValueError, match="raw identifier"):
        writer(f"The money went to {raw} today.").write("warning", FACTS)


@pytest.mark.parametrize("ok", ["Payment XABCDEFGHIJKLMNOP looks risky.", "bank 0119 and 021174", "$8,000,000.",
                                "800 dollars", "8000EBD3", "8000EBD301", "NEW123", "renewal 2022"])
def test_tokens_bank_codes_and_lookalikes_are_allowed(ok):
    assert RAW_ID.search(ok) is None
    writer(ok + " Done.").write("warning", FACTS)


def test_any_nine_hex_account_shape_is_rejected():
    # every 9-character hex string is account-shaped in this dataset, whatever its first character
    assert RAW_ID.search("paid to 9000EBD30") is not None


def test_raw_id_guard_catches_accounts_starting_with_1():
    with pytest.raises(ValueError, match="raw identifier"):
        writer("The money went to 100428660 today.").write("warning", FACTS)


@pytest.mark.parametrize("text", ["The score is a model estimate, not a finding of fraud.",
                                  "This is NOT A FINDING of fraud.", "It is not a definitive finding.",
                                  "This is not an official finding of wrongdoing."])
def test_brief_with_a_disclaimer_is_unchanged(text):
    assert DISCLAIMER.search(text)
    assert writer(text).write("brief", FACTS) == text


def test_brief_without_disclaimer_gets_it_appended():
    out = writer("The receiving account collects from many senders.").write("brief", FACTS)
    assert out.endswith("\n\nThis is a model estimate, not a finding of fraud.")


def test_warning_never_gets_the_disclaimer():
    assert "finding" not in writer("Please call us first.").write("warning", FACTS)


@pytest.mark.parametrize("reply", ["", "   ", "\n"])
def test_empty_completion_raises(reply):
    with pytest.raises(ValueError, match="empty"):
        writer(reply).write("brief", FACTS)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        writer("x.").write("correction", FACTS)


@pytest.mark.parametrize("text,expected", [
    ("One. Two. Thr", "One. Two."), ("Is it? Yes! Then", "Is it? Yes!"), ("Done.", "Done."),
    ("No end", ""), ("", ""), ("About $1.5k today. Then", "About $1.5k today."), ("Wait... and", "Wait..."),
])
def test_trim_to_sentence(text, expected):
    assert trim_to_sentence(text) == expected


def test_length_cut_completion_is_trimmed_before_checks():
    assert writer("Please check the payee. Then also the 8000EB", "length").write("warning", FACTS) == \
        "Please check the payee."
    out = writer("Held because of a rule. The receiv", "length").write("brief", FACTS)
    assert out == "Held because of a rule.\n\nThis is a model estimate, not a finding of fraud."
    with pytest.raises(ValueError, match="empty"):
        writer("Please check the", "length").write("warning", FACTS)
    assert writer("Ends without a stop", "stop").write("warning", FACTS) == "Ends without a stop"


def test_worker_prompt_from_a_stored_note_has_no_raw_ids(api_env):
    """End to end: the facts the API queues for Granite never contain raw account numbers."""
    from conftest import CUSTOMER_1, login, raw_account_numbers, service_headers
    from scamshield.store import Store

    with api_env.client() as c:
        for p in api_env.stream():
            assert c.post("/payments", json=p, headers=service_headers(api_env)).status_code == 200
    st = Store(api_env.db, api_env.envelope())
    jobs = st.claim_pending_notes(limit=1000)
    assert {j["kind"] for j in jobs} == {"warning", "brief"}
    raws = raw_account_numbers(api_env.aml) | {CUSTOMER_1[1]}
    for j in jobs:
        text = "\n".join(m["content"] for m in build_messages(j["kind"], j["facts"]))
        assert not any(r in text for r in raws)
        assert not RAW_ID.search(text)
    st.conn.close()
