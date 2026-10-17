"""Compare Granite GGUF builds on this machine: prompt and generation speed on a real ClaimSense prompt."""
import glob, json, os, resource, sys, time
sys.path.insert(0, "/app/src")
from llama_cpp import Llama
from scamshield.explain import build_messages

FACTS = {"rules": [{"title": "No attending physician on the claim", "fix": "Add the attending physician's identifier."}],
         "claim": [{"label": "reimbursed amount", "value": "$42,000", "direction": "higher than usual"},
                   {"label": "amount compared with typical claims for this diagnosis", "value": "9.8x typical",
                    "direction": "higher than usual"}], "amount": 42000}
threads = int(os.environ.get("THREADS", "3"))
results = []
for path in sorted(glob.glob(os.environ.get("MODELS", "/llm/*.gguf"))):
    t = time.time()
    llm = Llama(model_path=path, n_ctx=1024, n_threads=threads, n_batch=256, verbose=False)
    load = time.time() - t
    msgs = build_messages("correction", FACTS)
    llm.create_chat_completion(messages=msgs, max_tokens=1)          # warm-up (also caches the prompt)
    llm.reset()
    t = time.time(); p = llm.create_chat_completion(messages=msgs, max_tokens=1); t_prompt = time.time() - t
    llm.reset()
    t = time.time(); out = llm.create_chat_completion(messages=msgs, temperature=0.2, max_tokens=96); t_all = time.time() - t
    u = out["usage"]
    gen_tps = (u["completion_tokens"] - 1) / max(t_all - t_prompt, 1e-6)
    r = {"model": os.path.basename(path), "size_gb": round(os.path.getsize(path) / 1e9, 2), "load_s": round(load, 1),
         "prompt_tokens": u["prompt_tokens"], "prompt_s": round(t_prompt, 2),
         "prompt_tok_s": round(u["prompt_tokens"] / t_prompt, 1), "gen_tok_s": round(gen_tps, 2),
         "note_s_96tok": round(t_all, 1), "text": out["choices"][0]["message"]["content"].strip()}
    print(json.dumps(r), flush=True)
    results.append(r)
    del llm
print("max RSS MB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
