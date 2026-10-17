# ScamShield

**Real-time scam and mule-account interception on IBM LinuxONE.** IBM Z Datathon 2026, Theme 1: *Real-Time AI for Critical Decisions*.

ScamShield checks every payment **before the money moves**, in milliseconds:

- **Most payments go straight through** (`ALLOW`).
- **Out-of-character or risky payments get a clear, specific warning.** The customer decides whether to send (`WARN`).
- **Payments matching a laundering or mule-account pattern are paused** while the bank calls the customer back on a number it already holds; an analyst gets an evidence brief (`HOLD`).

> **The problem.** Australians lost **$2.03B to scams in 2024** (494,732 reports; National Anti-Scam Centre / ACCC *Targeting Scams 2024*). The money moves through **mule accounts** within minutes. The **Scams Prevention Framework** (2025) now requires banks to detect, disrupt and respond. Once money has left, it's rarely recovered, so the check has to happen *in-transaction*.

## Why LinuxONE

- **Scoring every payment in-transaction is the workload Telum was built for.** Banks already run their payments on IBM Z.
- **The AI runs next to the payments:**
  - a PyTorch customer-behaviour model on the **Telum II on-chip AI accelerator** (`torch_nnpa`);
  - an XGBoost risk model over the payer's and the receiving account's rolling history;
  - **IBM Granite** (llama.cpp with the zDNN backend) writing customer messages and analyst briefs.
- **Account numbers are pseudonymised on arrival** and payloads are encrypted with AES-256-GCM (CPACF).
- **Every decision is hash-chained** in a tamper-evident audit log.

## Results (IBM z17 VM, machine type 9175, unseen test days 9–10 Sep)

| | |
|---|---|
| Payments checked | **100%**, before money moves |
| Server latency, live at 274 payments/s with Granite running | **p50 4.4 ms**, p99 53 ms (3 vCPUs) |
| Risk model on 862,792 unseen payments (0.11% laundering) | ROC-AUC **0.984**, PR-AUC 0.564, best F1 0.587 (85% precision, 45% recall) |
| Payments paused | **0.074%** (641), of which **73%** are laundering (≈660× the base rate) |
| Laundering payments warned or paused | **82%** |
| Laundered money paused before it moved | **$269M of $424M** |
| Value-aware hold vs a plain risk threshold | **31% fewer pauses (641 vs 936), 53% → 73% precision, same $269M paused** |
| Laundering schemes broken (at least one payment paused) | **147 of 174** (fan-in 16/18, fan-out 17/20, cycles 19/22, …) |
| Injected victim-to-mule payments (live replay) | **55/57 warned or paused** |
| Telum II NNPA vs CPU (payment model) | **1.2× at 1,024 payments/call, 1.4× at 4,096**; slower below ~256 (dispatch overhead), so small batches stay on the CPU and bursts shift to the accelerator |
| Start-up | rolling state for **514,022 accounts from 4.2M historical payments in 61 s** |

Details: [docs/MODEL_CARD.md](docs/MODEL_CARD.md), [docs/evidence/value_hold/](docs/evidence/value_hold/value_hold_comparison.md) (hold policy), [docs/ENV_REPORT.md](docs/ENV_REPORT.md), [docs/evidence/](docs/evidence/).

## Architecture

```
core banking ──POST /payments──▶ api (FastAPI) ─────────────────────────────────────────────────────┐
                                  ├ pseudonymise account numbers (HMAC-SHA256)                         │
                                  ├ micro-batch payments arriving within 3 ms (bursts → bigger batches) │
                                  ├ ONE streaming feature engine: payment + payer history +            │
                                  │   receiving-account (mule) history, O(1) per payment               │
                                  ├ customer-behaviour autoencoder: PyTorch → Telum II NNPA (≥1,024/call) or CPU
                                  ├ risk: XGBoost (+ SHAP drivers)          ├ rule: big first payment to a new payee
                                  ├ fusion → ALLOW | WARN | HOLD                                         │
                                  ├ AES-256-GCM sealed payload + hash-chained audit ──▶ SQLite (WAL)     │
                                  ├ instant customer message; Granite rewrite queued ──▶ granite worker (llama.cpp + zDNN)
                                  └ SSE /stream ──▶ dashboard /ui (analyst · customer · auditor)
```

Everything runs as containers built **from the official L1CC Jupyter image**: non-root, read-only filesystem, only the API published, and only on localhost.

## Run it

On the LinuxONE VM (`Ubuntu24.04-Datathon`, flavour `AI_LAB`, Docker installed):

```bash
git clone <this repo> ~/scamshield && cd ~/scamshield
# data: HI-Small_Trans.csv + HI-Small_Patterns.txt into ~/shared/scamshield/aml (SHA-256 in docs/ENV_REPORT.md)
docker pull registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest
docker compose build api
printf 'ACCEL_MODE=nnpa\nJUPYTER_TOKEN=%s\nSHARED=%s\n' "$(openssl rand -hex 16)" ~/shared/scamshield > .env
docker run --rm --user 1000:1000 -v ~/shared/scamshield:/home/ibm-user/shared -w /home/ibm-user/shared \
  --entrypoint python scamshield-app:latest -m scamshield.training --raw aml --models models --data data
docker run --rm --user 1000:1000 -v $PWD:/repo -w /repo --entrypoint python scamshield-app:latest \
  scripts/make_secrets.py --customer-account 0119:8000E0380
# Granite (big-endian GGUF) into ~/shared/models/llm/ (see docs/ENV_REPORT.md)
docker compose up -d                                   # api + Granite worker
docker compose --profile demo run --rm simulator       # replay the held-out days
```

From your laptop: `scripts/tunnel.sh`, then open <http://localhost:8000/ui>. The demo logins are in `secrets/demo_passwords.txt`.

Tests: `PYTHONPATH=src python -m pytest -q tests`.

## Repo map

| Path | What |
|---|---|
| `src/scamshield/features.py` | The streaming feature engine (payment, payer history, receiver/mule history) and the scam rule |
| `src/scamshield/training.py` | Time split, both models, thresholds tuned on validation days, scheme-level metrics |
| `src/scamshield/scoring.py`, `fusion.py`, `explain.py` | Scoring engine and mule watchlist, decision policy, reasons and Granite prompts |
| `src/scamshield/inference/` | NNPA/CPU/NumPy anomaly scorers, XGBoost risk scorer |
| `src/scamshield/security.py`, `audit.py`, `store.py` | Pseudonymisation, AES-GCM, passwords, audit chain, storage |
| `src/scamshield/api/`, `ui/` | FastAPI service and dashboard |
| `src/scamshield/granite_worker.py`, `simulator.py` | Granite worker, payment stream with injected victim and honest payments |
| `notebooks/` | Data, training (including the NNPA demo) and fairness, executed on LinuxONE |
| `docs/` | Model card, security, environment report, demo script, evidence |

## Responsible AI

- **A warning never blocks.** The customer decides.
- **A hold always goes to a human,** who calls the customer back on a number the bank already has.
- **Every decision carries its reasons:** the rule fired, what's unusual for this customer, and the risk drivers (SHAP).
- **Granite only sees pseudonymised, structured facts.** Its output is rejected if it contains anything shaped like an account number, and analyst briefs always say "model estimate, not a finding of fraud".
- **A fairness audit is published** (by currency, payment format and bank size). Honest ACH payments and small-bank customers carry more friction; mitigations are in the [model card](docs/MODEL_CARD.md).

## Limitations (honest)

- **The data is synthetic (IBM AMLSim)** and labels mark laundering networks, not individual scam victims.
- **The models can't tell a victim's payment to a mule from an honest large first payment to a new payee,** because the mule accounts here look like ordinary receivers. Both get the same protective friction: about 1 in 3 paused, the rest warned ([evidence](docs/evidence/replay/victim_vs_honest_offline.md)).
- **About 87% of laundering is ACH, and laundering by cash, cheque or card is mostly missed.**
- **Granite is slow on 3 vCPUs** (about 1–2 minutes per note). It runs off the hot path, and an instant template message is shown first.

## Team

| Role | Name |
|---|---|
| Lead and infra (VM, containers, benchmarks) | _fill in_ |
| ML (features, models, fairness) | _fill in_ |
| Backend and security (API, crypto, RBAC, audit) | _fill in_ |
| UI and pitch (dashboard, video, slides) | _fill in_ |

All solution code was written during the IBM Z Datathon 2026 (Oct 17–18). It uses public data and open-source libraries only, and none of the reference lab notebooks' code. _(If AI coding assistants were used, say so here, per the organisers' rules.)_
