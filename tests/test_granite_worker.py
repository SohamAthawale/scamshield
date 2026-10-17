"""granite_worker.py with a fake llama.cpp model (no real model is loaded)."""
import sys
import types

import pytest

from claimsense import granite_worker
from claimsense.explain import build_messages, rule_reasons
from claimsense.granite_worker import RAW_ID, GraniteWriter, trim_to_sentence

FACTS = {"rules": rule_reasons(["missing_attending"]), "claim": [], "provider": [], "amount": 120.0,
         "provider_token": "PABCDEFGHIJKLMNOP", "provider_risk": 0.8}


class FakeLlama:
    def __init__(self, reply="All good.", finish_reason="stop", **kwargs):
        self.kwargs = kwargs
        self.reply = reply
        self.finish_reason = finish_reason
        self.calls = []

    def create_chat_completion(self, messages, temperature, max_tokens):
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        return {"choices": [{"message": {"role": "assistant", "content": self.reply},
                             "finish_reason": self.finish_reason}]}


def writer(reply, finish_reason="stop", max_tokens=120) -> GraniteWriter:
    w = GraniteWriter.__new__(GraniteWriter)
    w.llm = FakeLlama(reply, finish_reason)
    w.model = "fake.gguf"
    w.max_tokens = max_tokens
    return w


def test_init_uses_llama_cpp(monkeypatch, tmp_path):
    fake = types.ModuleType("llama_cpp")
    fake.Llama = lambda **kw: FakeLlama(**kw)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    w = GraniteWriter(str(tmp_path / "granite-3.3-2b.gguf"), threads=3, ctx=1024, max_tokens=99)
    assert w.model == "granite-3.3-2b.gguf" and w.max_tokens == 99
    assert w.llm.kwargs == {"model_path": str(tmp_path / "granite-3.3-2b.gguf"), "n_ctx": 1024, "n_threads": 3,
                            "verbose": False}


def test_write_sends_pseudonymised_prompt():
    w = writer("Please add the attending physician and resubmit.")
    assert w.write("correction", FACTS) == "Please add the attending physician and resubmit."
    call = w.llm.calls[0]
    assert call["messages"] == build_messages("correction", FACTS)
    assert call["temperature"] == 0.2 and call["max_tokens"] == 120


def test_completion_is_stripped():
    assert writer("  Check the date.  \n").write("correction", FACTS) == "Check the date."


@pytest.mark.parametrize("raw", ["PRV51001", "BENE11001", "PHY390922", "CLM46614", "see PRV123."])
def test_completion_with_raw_identifier_raises(raw):
    with pytest.raises(ValueError, match="raw identifier"):
        writer(f"The claim from {raw} looks odd.").write("correction", FACTS)


@pytest.mark.parametrize("ok", ["Provider PABCDEFGHIJKLMNOP is unusual.", "PRV12 is short", "XPRV51001",
                                "PRV-51001", "CLAIMS 2009"])
def test_tokens_and_lookalikes_are_allowed(ok):
    assert RAW_ID.search(ok) is None
    writer(ok + " Done.").write("correction", FACTS)


def test_brief_without_disclaimer_gets_it_appended():
    out = writer("The provider bills many claims per patient.").write("brief", FACTS)
    assert out.startswith("The provider bills many claims per patient.")
    assert out.endswith("This is a model estimate, not a finding of fraud.")


def test_brief_with_disclaimer_is_unchanged():
    text = "Risk is elevated. The score is a model estimate and Not A Finding of fraud."
    assert writer(text).write("brief", FACTS) == text


def test_correction_never_gets_the_disclaimer():
    assert "not a finding" not in writer("Fix the date.").write("correction", FACTS)


@pytest.mark.parametrize("reply", ["", "   ", "\n\n"])
def test_empty_completion_raises(reply):
    with pytest.raises(ValueError, match="empty"):
        writer(reply).write("brief", FACTS)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        writer("x.").write("poem", FACTS)


# ------------------------------------------------------------------------------ token limit
@pytest.mark.parametrize("text,expected", [
    ("One. Two. Thr", "One. Two."),
    ("Is it? Yes! Then", "Is it? Yes!"),
    ("Complete sentence.", "Complete sentence."),
    ("Ends with a stop.  ", "Ends with a stop."),
    ("No sentence end here", ""),
    ("", ""),
    ("Amount $1.5k is high. Then", "Amount $1.5k is high."),   # '.' inside a number is not a sentence end
    ("Line one.\nLine two", "Line one."),
    ("Wait... and", "Wait..."),
])
def test_trim_to_sentence(text, expected):
    assert trim_to_sentence(text) == expected


def test_length_cut_completion_is_trimmed():
    w = writer("Please check the service date. Then also verify the pat", finish_reason="length")
    assert w.write("correction", FACTS) == "Please check the service date."


def test_length_cut_brief_trimmed_then_disclaimer_added():
    out = writer("Held because of a rule hit. The provider", finish_reason="length").write("brief", FACTS)
    assert out == "Held because of a rule hit.\n\nThis is a model estimate, not a finding of fraud."


def test_length_cut_without_any_full_sentence_raises():
    with pytest.raises(ValueError, match="empty"):
        writer("Please check the", finish_reason="length").write("correction", FACTS)


def test_no_trim_when_finished_normally():
    assert writer("Ends without a stop", finish_reason="stop").write("correction", FACTS) == "Ends without a stop"


def test_raw_id_after_the_cut_is_dropped_not_rejected():
    w = writer("Check the date. The claim CLM46614", finish_reason="length")
    assert w.write("correction", FACTS) == "Check the date."


def test_module_has_entrypoint():
    assert callable(granite_worker.main)
