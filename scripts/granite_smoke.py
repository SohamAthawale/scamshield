"""Granite smoke test inside the L1CC image: load time, tokens/sec, memory, and one real note."""
import os, resource, sys, time
sys.path.insert(0, "/app/src")
from llama_cpp import Llama
from claimsense.explain import build_messages

path = os.environ.get("GRANITE_MODEL", "/home/ibm-user/shared/models/llm/granite-3.3-2b-instruct-be.Q4_K_M.gguf")
t = time.time()
llm = Llama(model_path=path, n_ctx=2048, n_threads=int(os.environ.get("THREADS", "2")), verbose=False)
print(f"model {os.path.basename(path)} loaded in {time.time()-t:.1f}s")
facts = {"rules": [{"title": "No attending physician on the claim", "fix": "Add the attending physician's identifier."}],
         "claim": [{"label": "reimbursed amount", "value": "$42,000", "direction": "higher than usual"}],
         "amount": 42000}
for kind in ("correction", "brief"):
    if kind == "brief":
        facts.update(provider_token="PQ7XK2M4HB5NRT3ZA", provider_risk=0.91,
                     provider=[{"label": "claims per patient", "value": "6.2", "peer_percentile": 99}])
    t = time.time()
    out = llm.create_chat_completion(messages=build_messages(kind, facts), temperature=0.2, max_tokens=160)
    dt = time.time() - t
    u = out["usage"]
    print(f"\n--- {kind}: {u['prompt_tokens']} prompt + {u['completion_tokens']} completion tokens in {dt:.1f}s "
          f"({u['completion_tokens']/dt:.1f} tok/s overall)")
    print(out["choices"][0]["message"]["content"].strip())
print(f"\nmax RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
