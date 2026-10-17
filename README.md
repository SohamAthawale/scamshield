# ScamShield

**Real-time scam and mule-account interception on IBM LinuxONE.** IBM Z Datathon 2026, Theme 1: *Real-Time AI for Critical Decisions*.

ScamShield checks every payment **before the money moves**, in milliseconds:

- **Most payments go straight through** (`ALLOW`).
- **Out-of-character or risky payments get a clear, specific warning.** The customer decides whether to send (`WARN`).
- **Payments matching a laundering or mule-account pattern are paused** while the bank calls the customer back on a number it already holds; an analyst gets an evidence brief (`HOLD`).

> **What we can prove, and what we can't.** The public data (IBM AMLworld) labels *laundering*, not scams. So every result below measures the part we can test: **breaking the mule-account chains that scam money flows through**. The customer warnings are a safety net for the victim's side, but this data has no scam labels, so their effect on scam losses can't be measured here, and we don't claim it.

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
| Laundered money paused before it moved | **$269M of $424M**, but 95% of it is 10 very large ACH transfers (largest $79.7M); the median paused laundering payment is $11,266 |
| Value-aware hold vs a plain risk threshold | **31% fewer pauses (641 vs 936), 53% → 73% precision, same $269M paused** |
| Laundering schemes broken (at least one payment paused) | **147 of 174** (fan-in 16/18, fan-out 17/20, cycles 19/22, …) |
| Injected victim-to-mule payments (live replay) | **55/57 warned or paused** |
| Telum II NNPA vs CPU (payment model alone) | **1.2× at 1,024 payments/call, 1.4× at 4,096**; slower below ~256 (dispatch overhead), so single payments stay on the CPU |
| Telum II NNPA in the live service | **Bulk payment files** (`POST /payments/batch`, 2,048 per file): **40,960 of 40,960 payments scored on the NNPA**. End to end there's no measurable gain (median 1,562 vs 1,610 payments/s, ±20% noise): the model is a small slice of each payment's cost |
| Start-up | rolling state for **514,022 accounts from 4.2M historical payments in 61 s** |

Details: [docs/MODEL_CARD.md](docs/MODEL_CARD.md), [docs/evidence/value_hold/](docs/evidence/value_hold/value_hold_comparison.md) (hold policy), [docs/ENV_REPORT.md](docs/ENV_REPORT.md), [docs/evidence/](docs/evidence/).

## Architecture

```
core banking ──POST /payments (one) or /payments/batch (bulk file)──▶ api (FastAPI) ──────────────────┐
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
- **A fairness audit is published** (by currency, payment format and bank size). Honest ACH payments (10.4% false alarms) and small-bank customers (2.6% vs 0.74% at the biggest banks) carry more friction. One mitigation is measured, not just planned: an ACH-only warning threshold capped at 6% on validation brings test ACH false alarms to 6.9% and small-bank ones to 1.2%, at a cost of 82.0% → 79.2% of laundering warned or paused ([model card](docs/MODEL_CARD.md#fairness)).

## Limitations (honest)

- **The data is synthetic (IBM AMLSim)** and labels mark laundering networks, not individual scam victims.
- **The models can't tell a victim's payment to a mule from an honest large first payment to a new payee,** because the mule accounts here look like ordinary receivers. Both get the same protective friction: about 95% warned or paused ([evidence](docs/evidence/replay/victim_vs_honest_offline.md)).
- **About 87% of laundering is ACH, and laundering by cash, cheque or card is mostly missed.**
- **Granite is slow on 3 vCPUs** (about 1–2 minutes per note). Every customer message is the instant template, built from the same facts; Granite's value is the analyst's evidence brief, where a minute or two is fine.
- **The dollar figure rests on a few payments.** 10 transfers make up 95% of the $269M paused. Precision (73%) and schemes broken (147/174) are the steadier measures.
- **The evaluation is short.** Thresholds were tuned on 2 validation days and tested on 2 days; precision fell from validation to test under both hold policies (60% → 53% plain, 81.5% → 73% value-aware). Real use needs months of data and recalibration.
- **The Telum II only helps the model, not yet the service.** Single payments run on the CPU, and bulk files run the model on the NNPA without a measurable end-to-end gain. Moving the tree model and feature work onto the accelerator (e.g. Snap ML / zDLC) is the next step.

## Team

| Name | Role and contributions |
|---|---|
| Soham Athawale | Team lead · LinuxONE VM and containers, Telum II benchmarks, scoring API, security and audit chain |
| Vaishnavi Joshi | Features and both models, fairness audit, dashboard and Granite messages, pitch |

All solution code was written during the IBM Z Datathon 2026 (Oct 17–18). It uses public data and open-source libraries only, and none of the reference lab notebooks' code.

**AI assistance:** the team used Claude Code, an AI coding assistant, to help write code, tests and documentation during the event. Every number in this repo comes from a run on the LinuxONE VM, saved under `docs/evidence/`.
