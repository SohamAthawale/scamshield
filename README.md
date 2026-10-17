# ClaimSense

**Real-time claim integrity on IBM LinuxONE.** IBM Z Datathon 2026, Theme 1: *Real-Time AI for Critical Decisions*.

ClaimSense checks every Medicare-style claim **before it is paid**, in milliseconds:

- **Honest mistakes come back with a plain-language fix** (`RETURN_WITH_FIX`).
- **A suspicious provider pattern pauses payment** and goes to a human investigator with an evidence brief (`HOLD_FOR_REVIEW`).
- **Everything else is paid at once** (`PAY`).

The honest majority gets help, not punishment.

> **The problem.** Australia's 2023 Philip Review estimated Medicare loses $1.5B to $3B a year, mostly through billing errors rather than deliberate fraud, and warned the figure could reach $8B. It pointed to real-time claim checking as a fix. We prove the approach on public US Medicare-style claims data. Sources: [health.gov.au](https://www.health.gov.au/ministers/the-hon-mark-butler-mp/media/medicare-non-compliance-and-fraud-costing-australians), [Philip Review](https://apo.org.au/node/322212).

## Why LinuxONE

- **Claims are high-volume transactions**, the same workload as card authorisation, which Telum was built for.
- **The AI runs on the same machine as the transactions:**
  - a PyTorch claim model on the **Telum II on-chip AI accelerator** (`torch_nnpa`);
  - an XGBoost provider model;
  - **IBM Granite** writing the explanations through llama.cpp with the zDNN backend.
- **Every claim is scored in-transaction**, instead of sampling after payment.
- **Payloads are encrypted with AES-256-GCM** on CPACF, and every decision lands in a tamper-evident audit chain.

## Results (measured on the IBM z17 VM, machine type 9175)

| | |
|---|---|
| Server latency per claim, live demo at 50 claims/s with Granite running | **p50 3.7 ms, p99 13.4 ms** |
| Peak throughput on 3 vCPUs | **283 claims/s**, server p50 4.2 ms |
| Claims scored | **100%**, before payment |
| Provider risk model on held-out providers | ROC-AUC **0.942**, PR-AUC 0.675 (base rate 0.094) |
| Held claims that come from flagged providers | **61%** (base rate 36%), with 0.5% of claims held |
| Injected honest billing errors caught | **65/65 rule-type errors**; amount typos 1/13 (see limitations) |
| Telum II NNPA vs CPU, claim autoencoder | **1.37× faster at 1,024 claims/batch, 1.61× at 4,096**; slower below 64 (dispatch overhead), so small batches stay on the CPU |

Details: [docs/MODEL_CARD.md](docs/MODEL_CARD.md), [docs/ENV_REPORT.md](docs/ENV_REPORT.md), [docs/evidence/](docs/evidence/).

## Architecture

```
simulator ──POST /claims──▶ api (FastAPI, uvicorn) ────────────────────────────────────────────────┐
                              ├ pseudonymise IDs (HMAC-SHA256)                                      │
                              ├ micro-batch claims arriving within 3 ms                             │
                              ├ claim features + rolling provider state (one claim at a time)        │
                              ├ claim autoencoder: PyTorch → Telum II NNPA (≥256/batch) or CPU      │
                              ├ provider risk: XGBoost (+ SHAP reasons, peer percentiles)            │
                              ├ hard rules: after death, swapped dates, duplicate, no coverage, …    │
                              ├ fusion → PAY | RETURN_WITH_FIX | HOLD_FOR_REVIEW                     │
                              ├ AES-256-GCM sealed payload + hash-chained audit ──▶ SQLite (WAL)    │
                              ├ template note now; Granite note queued ──▶ granite worker (llama.cpp + zDNN)
                              └ SSE /stream ──▶ dashboard /ui (investigator · billing office · auditor)
```

Everything runs as containers built **from the official L1CC Jupyter image** (`deploy/app.Dockerfile`, `docker-compose.yml`): non-root, read-only filesystem, only the API published, and only on localhost.

## Run it

On the LinuxONE VM (`Ubuntu24.04-Datathon`, flavour `AI_LAB`, Docker installed):

```bash
git clone <this repo> ~/claimsense && cd ~/claimsense
# data: the Kaggle CSVs in ~/shared/data/raw (scripts/push_data.sh copies them from a laptop)
docker pull registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest
docker compose build api
printf 'ACCEL_MODE=nnpa\nJUPYTER_TOKEN=%s\n' "$(openssl rand -hex 16)" > .env
docker run --rm --user 1000:1000 -v $PWD:/repo -v ~/shared:/home/ibm-user/shared -w /home/ibm-user/shared \
  --entrypoint python claimsense-app:latest -m claimsense.training --raw data/raw --models models --data data
docker run --rm --user 1000:1000 -v $PWD:/repo -w /repo --entrypoint python claimsense-app:latest \
  scripts/make_secrets.py --provider-id PRV54092
# Granite (big-endian GGUF) into ~/shared/models/llm/ (see docs/ENV_REPORT.md)
docker compose up -d                                   # api + Granite worker
docker compose --profile demo run --rm simulator       # replay held-out claims
```

From your laptop: `scripts/tunnel.sh`, then open <http://localhost:8000/ui>. The demo logins are in `secrets/demo_passwords.txt`.

Tests (444+): `PYTHONPATH=src python -m pytest -q tests` (add `-m slow` for the real-data parity test).

## Repo map

| Path | What |
|---|---|
| `src/claimsense/features.py` | Claim and provider features, streaming plus batch, parity-tested |
| `src/claimsense/training.py` | Split, both models, thresholds tuned on a replayed validation stream |
| `src/claimsense/scoring.py`, `fusion.py`, `explain.py` | Scoring engine, decision policy, reasons and Granite prompts |
| `src/claimsense/inference/` | NNPA/CPU/NumPy anomaly scorers, XGBoost scorer |
| `src/claimsense/security.py`, `audit.py`, `store.py` | Pseudonymisation, AES-GCM, passwords, audit chain, storage |
| `src/claimsense/api/`, `ui/` | FastAPI service and dashboard |
| `src/claimsense/granite_worker.py`, `simulator.py` | Granite explanation worker, claim stream with injected errors |
| `notebooks/` | EDA, training (including the NNPA demo) and fairness, executed on LinuxONE |
| `scripts/` | VM helpers, probe, benchmarks, secrets |
| `docs/` | Model card, security, environment report, demo script, evidence |

## Responsible AI

- **A human always decides.** Nothing is denied automatically.
- **Every decision carries its reasons:**
  - rule broken and how to fix it;
  - unusual claim features and their direction;
  - provider drivers as SHAP values with peer percentiles.
- **Granite only sees pseudonymised, structured facts.** Investigator briefs always say "model estimate, not a finding of fraud".
- **A fairness check is published.** Gender is even, while under-65 patients and race code 2 are flagged more often. Mitigations and next steps are in the [model card](docs/MODEL_CARD.md).

## Limitations

- **Synthetic data with provider-level labels.**
- **Amount typos are mostly missed.** This dataset has no fixed fee per item; MBS does, which would make an extra zero a deterministic check.
- **Granite is slow on 3 vCPUs** (about 1–2 minutes per note). It runs off the hot path; a template note is shown immediately.

## Team

| Role | Name |
|---|---|
| Lead and infra (VM, containers, benchmarks) | _fill in_ |
| ML (features, models, fairness) | _fill in_ |
| Backend and security (API, crypto, RBAC, audit) | _fill in_ |
| UI and pitch (dashboard, video, slides) | _fill in_ |

All solution code in this repository was written during the IBM Z Datathon 2026 (Oct 17–18). It uses public data and open-source libraries only, and none of the reference lab notebooks' code. _(If AI coding assistants were used, say so here, per the organisers' rules.)_
