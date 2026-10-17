"""Granite explanation worker: off the hot path, upgrades template notes to Granite-written ones.

Polls the notes queue in SQLite, builds the prompt from STRUCTURED, PSEUDONYMISED facts only (see
explain.build_messages), runs IBM Granite through llama.cpp (built with the zDNN backend for the Telum
II NNPA), and writes the text back. On any failure the template note stays, so every decision always has
a truthful explanation.

    python -m scamshield.granite_worker
"""
from __future__ import annotations

import logging
import os
import re
import signal
import time
from pathlib import Path

from .explain import build_messages
from .security import load_keyring
from .store import Store

log = logging.getLogger("scamshield.granite")

# the model must never produce raw account numbers (dataset format: 9 hex characters, e.g. 8000EBD30, 100428660)
RAW_ID = re.compile(r"\b([0-9A-F]{9}|NEW\d{7})\b")
# "not a finding", "not a definitive finding", "not a formal finding", ... of fraud
DISCLAIMER = re.compile(r"not\s+(a|an)\s+(\w+\s+)?finding", re.IGNORECASE)


def trim_to_sentence(text: str) -> str:
    """Cut a completion that hit the token limit back to its last full sentence."""
    m = list(re.finditer(r"[.!?](\s|$)", text))
    return text[: m[-1].end()].strip() if m else ""


class GraniteWriter:
    def __init__(self, model_path: str, threads: int, ctx: int = 2048, max_tokens: int = 180):
        from llama_cpp import Llama

        t = time.time()
        self.llm = Llama(model_path=model_path, n_ctx=ctx, n_threads=threads, verbose=False)
        self.model = Path(model_path).name
        self.max_tokens = max_tokens
        log.info("loaded %s in %.1fs (threads=%d)", self.model, time.time() - t, threads)

    def write(self, kind: str, facts: dict) -> str:
        out = self.llm.create_chat_completion(messages=build_messages(kind, facts), temperature=0.2,
                                              max_tokens=self.max_tokens)
        text = out["choices"][0]["message"]["content"].strip()
        if out["choices"][0].get("finish_reason") == "length":
            text = trim_to_sentence(text)
        if not text:
            raise ValueError("empty completion")
        if RAW_ID.search(text):
            raise ValueError("completion contains something that looks like a raw identifier")
        if kind == "brief" and not DISCLAIMER.search(text):
            text += "\n\nThis is a model estimate, not a finding of fraud."
        return text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    e = os.environ.get
    secrets = Path(e("SCAMSHIELD_SECRETS", "/run/secrets"))
    data = Path(e("SCAMSHIELD_DATA_DIR", "/data"))
    store = Store(e("SCAMSHIELD_DB", str(data / "scamshield.db")), load_keyring(secrets),
                  e("SCAMSHIELD_AUDIT_CHECKPOINTS", str(data / "audit_checkpoints.log")))
    model = e("GRANITE_MODEL", "/llm/granite-3.3-2b-instruct-be.Q8_0.gguf")
    writer = GraniteWriter(model, threads=int(e("GRANITE_THREADS", "3")), max_tokens=int(e("GRANITE_MAX_TOKENS", "120")))
    idle_sleep = float(e("GRANITE_IDLE_SLEEP", "0.5"))
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    store.requeue_stuck_notes()
    log.info("waiting for notes")
    while running:
        jobs = store.claim_pending_notes(limit=1)
        if not jobs:
            time.sleep(idle_sleep)
            continue
        job = jobs[0]
        t = time.perf_counter()
        try:
            text = writer.write(job["kind"], job["facts"])
            ok = True
        except Exception as ex:
            log.warning("note %s failed: %r", job["id"], ex)
            text, ok = None, False
        ms = (time.perf_counter() - t) * 1000.0
        store.complete_note(job["id"], text, writer.model, ms, ok)
        log.info("note %s (%s) %s in %.0f ms", job["id"], job["kind"], "ok" if ok else "kept template", ms)
    store.requeue_stuck_notes()


if __name__ == "__main__":
    main()
