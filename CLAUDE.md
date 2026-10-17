# ClaimSense: Master Plan

IBM Z Datathon 2026 (24 hours, October 17-18, 2026), **Theme 1: Real-Time AI for Critical Decisions** ("Detect. Decide. Respond.")

Real-time Medicare claim integrity on IBM LinuxONE. Score every claim the moment it is submitted, return honest billing errors with a plain-English fix, and route suspicious provider patterns to an investigator with an evidence brief.

---

## 0. How to use this file (Claude Code, read first)

- This file is the source of truth. Read it fully before doing anything.
- Work phase by phase. Never skip Phase 0. Tick checkboxes here as tasks finish and commit after each task.
- Record every environment finding (versions, what worked, what failed, the workaround) in `docs/ENV_REPORT.md`.
- Any claim that something works on IBM Z must come from a command actually run on the VM, with output saved under `docs/evidence/`.
- If a command or flag in this plan disagrees with the official docs for that tool, the docs win. Fix this file when that happens.
- When blocked for more than 30 minutes, take the fallback in Section 8 and move on.
- Section 0.1 holds facts confirmed at the kickoff (LinuxONE Setup & Tech Prep Session, 2026-10-16) and the official lab repo. They override older assumptions anywhere below.

### 0.1 Confirmed at kickoff (2026-10-16) and in the official repo

| Topic | Confirmed fact | Source |
|---|---|---|
| Format | 24-hour datathon, Oct 17-18 2026. Max score 40 (4 criteria x 10). | Kickoff title slide, community.ibm.com/zsystems/datathon |
| Hardware | IBM z17 / LinuxONE 5 at Marist, **Telum II** with on-chip AI accelerator. `lscpu \| grep -i nnpa` shows `nnpa` on the datathon VM. | Live demo (`lscpu` on `bgaitest`) |
| VM size | 1 server, **3 vCPU, 8 GB RAM, 50 GB disk**, 30-day trial, full root over SSH, user `linux1`. | "IBM LinuxONE Community Cloud (Z Datathon)" slide |
| Registration | Event code **`SSF2026`** at registration (also mentioned in the project description). Without it you do not get the `AI_LAB` flavour; fix = email linuxone@us.ibm.com to have the registration deleted, then re-register. Description field needs ≥75 characters. Click the activation email. | Kickoff + final "Important Links" slide |
| OS image | Pick **`Ubuntu 24.04-Datathon`** (or `RHEL9.8-Datathon` with podman). Flavour `AI_LAB`. | jupyter-lab-ml README, kickoff "just-in-time update" |
| Docker | **Not preinstalled on Ubuntu.** Install: `curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`, then `sudo usermod -aG docker $USER; newgrp docker`, `sudo systemctl start docker`, `exec bash`. | README Step 4 |
| Jupyter image | `registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest`. No `docker login` step in the official guide. Runs with `--network host` on **port 38888**, shared folder mounts at `/home/ibm-user/shared`, default token **`Your_Token1`**. | README Step 5-6 |
| Image contents | Python 3.12 venv at `/opt/venv`. **torch 2.11.0+cpu + torch_nnpa 1.5.0**, xgboost 3.4.1, tensorflow 2.18.1 + ibm-zdnn-plugin 2.3.0, **llama_cpp_python 0.3.36 built with `GGML_ZDNN=ON`**, OpenBLAS, `-march=z16`, onnx, tf2onnx, skl2onnx, scikit-learn 1.9, pandas 3.0, numpy 2.0.2, pypmml, rank-bm25, pypdf, transformers, httpx. | packages.txt + Lab 02 env cell |
| Not in image | snapml, onnxruntime, fastapi, uvicorn, pydantic, cryptography, PyJWT. `pip install` inside the container is allowed (FAQ 9). | packages.txt |
| Granite | Big-endian GGUFs are in the image under `/home/ibm-user/`. Seen: `granite-4.0-h-350m-base-be.f16.gguf` (653 MB, a *base* model). Lab 02 expects `granite-3.3-2b-instruct-be.Q4_K_M`. **Check which files exist before planning around one.** | Lab 02 env cell |
| Uploads | Jupyter upload max 200 MB per file; use `scp -i key.pem file linux1@148.100.X.X:~/shared/` for larger. Shared folder needs uid 1000 write access. | README FAQ 10, 15, 16 |
| Submission | Screenshots of the Jupyter notebook running on LinuxONE, a presentation video **within the time limit**, submitted via the official portal. | "Getting Started" Step 06 |
| Originality | "Implement all application and modeling code from scratch during the hackathon period." "Avoid submitting reference ML lab notebooks without changes." | Steps slide, Submission Playbook |

**Still unconfirmed (ask on Discord/mentors before H1):**
- [ ] Video time limit (minutes) and submission deadline time zone.
- [ ] Whether AI coding assistants (e.g. Claude Code) are allowed. Someone asked this in the kickoff chat; the answer was not captured. **Confirm in writing before relying on it**, and disclose usage in the README if allowed.
- [ ] Team size limits and whether the repo must be public.

## 1. Hard rules

1. All solution code is written in this repo during the datathon window (Oct 17-18). Environment setup (registration, VM, Docker, probing) can happen before H0; solution code cannot. Do not copy code from other projects, tutorials or **the four reference lab notebooks** (Fraud LSTM, Churn, MNIST, Loan Approval). Public datasets, libraries and container images are fine.
2. Target platform is s390x (IBM Z). It is **big-endian**. Verify every dependency on the VM before designing around it.
3. Never commit data, model weights, keys, tokens, `.env`, `*.pem` or `kaggle.json`. Create `.gitignore` before anything else.
4. No real patient data. Only the public Kaggle dataset.
5. The LLM never sits on the hot path. Claims are scored without waiting for Granite.
6. Boring, proven tools only. **Prefer what is already in the L1CC Jupyter image** (Section 0.1). Avoid packages that need pyarrow, numba/llvmlite or Rust builds unless Phase 0 proves they install.
7. Feature freeze at H18. After that: fixes, benchmarks, recording, slides.
8. **Memory budget is 8 GB for everything.** Never run Jupyter, the api and Granite at full tilt at once without checking `docker stats` / `free -h`.

## 2. Rubric map

Judges' "Submission Playbook" (kickoff slide 14), mapped to our phases:

| Criterion (max 10 each) | What the playbook asks for | How we hit it | Phases |
|---|---|---|---|
| Technical Design | Video screenshots of workloads running live on L1CC. At least 2 AI open-source frameworks (e.g. PyTorch + XGBoost). Containerised deployment with Docker/Podman. Platform security or Telum hardware AI acceleration. | PyTorch autoencoder on NNPA via torch_nnpa, XGBoost provider model, Granite via llama.cpp+zDNN, Docker Compose, CPACF crypto, NNPA vs CPU benchmark. | 0, 3, 4, 6 |
| Novelty & Originality | 100% of solution code written during the hackathon. Don't submit reference lab notebooks unchanged. Combine hybrid AI in creative, unexpected ways. Pair structured predictive models with LLM reasoning (Granite). | Two-level hybrid (claim-level unsupervised anomaly + provider-level supervised risk) fused into three actions, with Granite writing *different* notes for two audiences. Different domain and architecture from Lab 02. | 2, 5 |
| Pitch & Delivery | Strictly within the video time limit. Problem, target persona, value proposition. Live interactive demo recorded on LinuxONE. Team roles and contributions. | Section 3 story, `docs/DEMO_SCRIPT.md`, roles in Section 11. | 7 |
| Community Value | A specific, quantified beneficiary population. Explicit tie to one of the 3 themes. "AI for Good" framing. Ethical AI with transparent decision reasoning. | Philip Review numbers, Theme 1 named on slide 1, honest-majority framing, reasons always shown, fairness table, human-in-the-loop. | 5, 6, 7 |

Theme 1's own example list includes "Real-Time Payment Fraud Detector" and "Real-Time Insurance Risk Scoring"; say ClaimSense is in that family on the first slide.

**Differentiate from Lab 02 (Loan Approval: XGBoost + Granite + BM25 RAG).** Judges have seen it. Our hybrid must look different: real-time stream, unsupervised + supervised fusion, action routing, security core. Do not add a BM25 policy-RAG step unless it is clearly our own design and code.

## 3. The story (README + pitch)

- **Problem:** Medicare delivers 500M+ services a year (about $32B). The 2023 Philip Review estimated $1.5B to $3B is lost each year, mostly through billing errors rather than deliberate fraud, warned it could reach $8B without new controls, and pointed to real-time claim checking as a fix.
- **Primary persona:** a busy GP practice manager who wants to bill correctly. **Secondary:** an integrity investigator with a limited caseload.
- **Value:** every claim checked in milliseconds before payment. Honest mistakes come back with a fix. Suspicious patterns go to a human with evidence. The honest majority gets help, not punishment.
- **Why a mainframe:** claims are high-volume transactions, the same pattern as card authorisation that Telum was built for. Score 100% of claims in-transaction instead of sampling after payment. Telum II's on-chip accelerator does the inference without leaving the core.
- **Honest framing:** the dataset is US Medicare-style public data. "Australia's problem, proven on public claims data."

## 4. Environment

### 4.1 What we run on

- **L1CC VM:** `Ubuntu 24.04-Datathon`, flavour `AI_LAB`, 3 vCPU / 8 GB / 50 GB, user `linux1`, public IP `148.100.x.x`, reached over SSH with the key pair created in the portal (`chmod 600 key.pem`). If the VM stops answering ping/SSH: stop it, wait for SHUTOFF, start it again in the dashboard. A reused IP can trigger a "REMOTE HOST IDENTIFICATION HAS CHANGED" warning; clear it with `ssh-keygen -R <ip>`.
- **Hardware:** z17 / LinuxONE 5 with Telum II. Expect `ACCEL_MODE=nnpa`; Phase 0 still records the real machine type and flags from the VM as evidence. Spyre cards appeared on a slide as a scale-out option; they are **not** part of our VM, so never claim them.
- **Big-endian:** anything binary (GGUF models, raw tensor dumps) must be big-endian. Prefer portable formats: CSV, JSON, XGBoost JSON/UBJ, ONNX, `.npy`. Avoid Parquet/Feather.
- **CPACF** (on-core crypto for AES and SHA) is present, so the security story holds.

### 4.2 Containers, pods, Docker, Podman

- **Container:** an isolated process with its own filesystem image. Images must be built for s390x. Official Docker Hub images such as `python` are multi-arch and include s390x.
- **Pod:** a group of containers sharing one network namespace and deployed as one unit. Podman can create pods locally (`podman pod create`) and run a Kubernetes Pod YAML directly (`podman kube play`).
- **Docker has no pod object.** Docker Compose gives the same practical result.
- **Our choice:** Docker Compose on Ubuntu is primary (that is the official path). Ship `deploy/claimsense-pod.yaml` as a stretch for `podman kube play` and the "OpenShift-ready" line.

### 4.3 Jupyter notebooks (official setup, verified against the README)

- Notebooks are for EDA, training and charts. Production scoring runs in the api container, never in Jupyter.

```bash
docker pull registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest
mkdir -p ~/shared && chmod a+w ~/shared
docker run --network host --name notebook \
  -v /home/linux1/shared:/home/ibm-user/shared \
  -d registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest
# if it fails on seccomp, add: --security-opt seccomp=unconfined
# permission errors on the shared folder:
sudo chown -R 1000:1000 /home/linux1/shared
```

- **Security deviation from the official guide (deliberate):** the guide opens port 38888 to the internet (`iptables -I INPUT ... --dport 38888 -j ACCEPT`) with the well-known token `Your_Token1`, and `--network host` binds it on every interface. We do **not** open the firewall rule, and we override the token (append the `jupyter notebook --ip=0.0.0.0 --ServerApp.port=38888 --ServerApp.token='<from .env>' ...` arguments shown in the README). Access is only through the SSH tunnel. Mention this in SECURITY.md; it is a small, real security win.
- The `shared/` folder is the handoff. Notebooks write `shared/models/*` and `shared/data/*`. The api container mounts it read-only.

### 4.4 AI frameworks and acceleration (revised)

The L1CC Jupyter image already contains NNPA-enabled builds of everything we need, so **it is our api base image too**. This removes the old dependency on `icr.io` credentials, Snap ML and zDLC.

| Model | Train | Serve (hot path) | Accelerator path | Fallback |
|---|---|---|---|---|
| Claim autoencoder | PyTorch in Jupyter | PyTorch in the api, via **torch_nnpa** | NNPA through torch_nnpa | PyTorch CPU, then NumPy forward pass from `claim_ae_weights.npz` |
| Provider risk | XGBoost 3.4.1 in Jupyter | XGBoost in the api (`Booster.inplace_predict`) | CPU (tree inference; fine at our volume) | NumPy tree walker over the JSON dump (last resort) |
| Explanations | n/a | llama_cpp_python 0.3.36 in a separate worker container | zDNN backend built in (`GGML_ZDNN=ON`) | Templated reasons |

- **torch_nnpa API is unknown to us.** In Phase 0, inspect it before writing code: `python -c "import torch, torch_nnpa; print(torch_nnpa.__version__); print(dir(torch_nnpa))"`, and look for a README in its site-packages folder. Record how a model is moved to NNPA (device name, compile call or backend). Don't guess.
- **Benchmark (Phase 6):** the same autoencoder on CPU vs NNPA through torch_nnpa, batch 1 and 32. For TensorFlow, Lab 03 shows ibm-zdnn-plugin dispatching MatMul/Relu to NNPA transparently; we don't use TF, but this is the reference for the honest A/B approach.
- **Stretch only:** Snap ML (`icr.io/ibmz/ibmz-accelerated-for-snapml`) and zDLC. Use them only if Phase 0 finishes early and credentials work.

### 4.5 Granite on s390x (revised)

- llama.cpp is already built for s390x with zDNN + OpenBLAS inside the image (`llama_cpp_python`). **Do not build llama.cpp from source** unless the in-image build fails.
- Phase 0: `docker exec notebook bash -c 'ls -la /home/ibm-user/*.gguf'`, then load each candidate with `llama_cpp.Llama(model_path=..., n_ctx=2048, n_threads=2)` and run one chat completion.
- Model choice, in order:
  1. `granite-3.3-2b-instruct-be.Q4_K_M.gguf` if present (≈1.5 GB RAM, instruction-tuned). **Preferred.**
  2. Any other `*-instruct-be*.gguf` in the image.
  3. Convert on the laptop: `python convert_hf_to_gguf.py --outtype q8_0 --bigendian --outfile granite-3.3-2b-instruct-be.q8_0.gguf granite-3.3-2b-instruct/` (≈2.7 GB, fits). Then `scp` it into `~/shared/models/`. Do **not** use f16 for the 2B model (≈5 GB, too much for 8 GB total).
  4. `granite-4.0-h-350m-base-be.f16.gguf` is a *base* model; it will not follow instructions well. Only as a last resort, with a strict few-shot template.
- Serve with `python -m llama_cpp.server` only if its extras install (`pip install 'llama-cpp-python[server]'` will try to rebuild; check first). Otherwise run a tiny worker that imports `llama_cpp` directly and pulls jobs from SQLite. The second option is simpler and is the default.
- Threads: `n_threads=2`, leaving one vCPU for the api.

### 4.6 How Claude Code works with the VM

- Run Claude Code on the laptop in this repo. The VM is the runtime.
- `scripts/remote.sh "<cmd>"` wraps `ssh -i "$L1CC_KEY" linux1@"$L1CC_HOST"`.
- `scripts/sync.sh` rsyncs the repo to `~/claimsense` on the VM, excluding `data/`, `models/`, `.env`, `.git`, `*.pem`.
- `scripts/tunnel.sh` opens SSH tunnels so the laptop browser can reach the VM:
  `ssh -i "$L1CC_KEY" -N -L 8000:localhost:8000 -L 38888:localhost:38888 linux1@"$L1CC_HOST"`
- Publish container ports on `127.0.0.1` only. Access is always through the tunnel.

## 5. Architecture

```
[simulator] --POST /claims--> [api: FastAPI scorer] ---------------------------+
                                |                                              |
                                |- pseudonymise IDs (HMAC-SHA256)              |
                                |- features: claim-level + rolling provider    |
                                |- claim anomaly: PyTorch autoencoder          |
                                |    (torch_nnpa on Telum II, or CPU)          |
                                |- provider risk: XGBoost                      |
                                |- fusion -> PAY | RETURN_WITH_FIX | HOLD      |
                                |- encrypt payload (AES-256-GCM, CPACF)        |
                                |- store + hash-chained audit -> [SQLite WAL]  |
                                |- queue explanation ------> [granite worker: llama_cpp + zDNN]
                                '- SSE /stream -> [dashboard served by api]

[jupyter] (dev only): EDA, training, exports -> shared/models
```

| Service | Image | Role |
|---|---|---|
| api | `FROM registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest` + `pip install fastapi uvicorn pydantic PyJWT cryptography` | scoring, auth, dashboard, SSE |
| granite | same base image, different command | explanation notes |
| simulator | same base image (already pulled; avoids a second large pull) or `python:3.12-slim` | replays held-out claims in time order |
| jupyter | L1CC JupyterLab image | dev only (compose profile `dev`); stop it during the demo to free RAM |

- The base image is large. Check `df -h` after pulling; FAQ 12 shows how to prune images.
- Set compose `mem_limit`s: api ≈1.5 GB, granite ≈3 GB, simulator ≈256 MB. Jupyter gets what's left when it runs.
- SQLite in WAL mode is deliberate: zero extra dependencies, plenty for demo throughput.

## 6. Phases

H0 = the official datathon start. Setup tasks marked **(pre-H0 OK)** are environment work, not solution code, and can be done before the start.

### Phase 0 (H0 to H1): Environment probe and decisions

- [ ] **(pre-H0 OK)** Register on L1CC with event code `SSF2026`, activate via email, create the SSH key pair, create the `Ubuntu 24.04-Datathon` instance with flavour `AI_LAB`, confirm ACTIVE.
- [ ] **(pre-H0 OK)** SSH in, install Docker (Section 0.1), pull the Jupyter image, start it per 4.3 (custom token, no firewall rule), open it through the tunnel.
- [ ] Create repo skeleton (Section 9), `.gitignore`, `.env.example`, `scripts/remote.sh`, `sync.sh`, `tunnel.sh`.
- [ ] Write and run `scripts/probe.sh` on the VM. Save output to `docs/evidence/probe-<timestamp>.txt`. It collects:
  - `uname -m`, `cat /etc/os-release`, `nproc`, `free -h`, `df -h`
  - `grep -E '^(Type|Model)' /proc/sysinfo` and `lscpu` (machine type; expect z17)
  - `lscpu | grep -io nnpa` and `grep -m1 '^features' /proc/cpuinfo` (look for `nnpa` and `msa`)
  - `docker version`, `docker compose version`
  - `openssl version`, `openssl speed -evp aes-256-gcm -seconds 3`, `openssl speed -evp sha256 -seconds 3`
  - inside the Jupyter container: `python -V`, `pip list`, `ls -la /home/ibm-user/*.gguf`, the torch_nnpa inspection from 4.4, `python -c "import llama_cpp; print(llama_cpp.__version__)"`
- [ ] Set `ACCEL_MODE=nnpa` or `cpu` in `.env` based on the `nnpa` flag.
- [ ] Dependency smoke test in the image: `pip install fastapi uvicorn pydantic PyJWT cryptography`. For each, record wheel vs source build vs failed. `cryptography` on s390x may need a source build; check `pip download --only-binary=:all: cryptography` first.
- [ ] Granite smoke test (4.5): one completion, record tokens/sec and RSS memory.
- [ ] Write decisions into `docs/ENV_REPORT.md`.

**Exit:** probe saved, `ACCEL_MODE` set, Jupyter reachable through the tunnel, api deps installable, Granite model chosen, torch_nnpa usage recorded.

### Phase 1 (H1 to H3): Data and features

- [ ] Download Kaggle dataset `rohitrox/healthcare-provider-fraud-detection-analysis` (Kaggle API on the VM, or laptop then `scp` into `~/shared/data/raw/`; Jupyter upload caps at 200 MB). Discover file names by glob; don't hardcode them.
- [ ] `notebooks/01_eda.ipynb`: shapes, label balance (about 9% of providers flagged), join keys (`Provider`, `BeneID`, `ClaimID`), dates, missingness.
- [ ] **Split by provider**, never by claim, stratified on the label: 70/15/15. Save `shared/data/splits.csv`. The replay stream for the demo = claims from test providers, sorted by claim start date.
- [ ] `src/claimsense/features.py` is the single source of truth, imported by notebooks AND the api:
  - Claim-level (inpatient and outpatient unified, with `is_inpatient`): reimbursed amount, deductible, claim duration, admission days, number of diagnosis codes, number of procedure codes, which physician fields are present, patient age at claim, chronic condition count, **claim dated after patient's date of death**, diagnosis-procedure pair rarity (log frequency from training data), attending physician claim volume.
  - Provider rolling state (updated incrementally per claim): claim count, mean and max reimbursement, inpatient share, mean admission days, unique patients, unique attending physicians, claims per patient, share of rare code pairs, share of claims after death.
- [ ] `tests/test_features.py`: same output in batch mode and streaming mode on 1k rows.
- [ ] pandas 3.0 is in the image: watch for copy-on-write and dtype changes; pin behaviour with explicit dtypes.

**Exit:** features pass parity tests; splits saved.

### Phase 2 (H3 to H6): Models

- [ ] `notebooks/02_provider_model.ipynb`: XGBoost classifier on provider aggregates. Handle imbalance (`scale_pos_weight`). Report PR-AUC, ROC-AUC, recall in the top 5% of providers. Export `shared/models/provider_xgb.json` and `provider_meta.json` (feature order, global importances, peer percentiles per feature).
- [ ] Reload `provider_xgb.json` in a fresh process and confirm identical predictions (catches any endianness surprise). Record the check.
- [ ] `notebooks/03_claim_autoencoder.ipynb`: small PyTorch MLP autoencoder (for example 24 -> 16 -> 8 -> 16 -> 24) trained on standardised claim features from **non-flagged** providers only. Threshold = 99th percentile of reconstruction error on clean validation claims. Per-feature reconstruction error gives the "why".
- [ ] Export: `claim_ae.pt` (state_dict), `claim_ae.onnx` (for the stretch zDLC path), `claim_ae_weights.npz` (NumPy fallback), `scaler.json`, `thresholds.json`.
- [ ] Verify the autoencoder gives the same scores on CPU and on NNPA (within tolerance; NNPA uses reduced precision internally). Record the max absolute difference.
- [ ] `docs/MODEL_CARD.md`: data, splits, metrics, thresholds, known limits.

**Exit:** artifacts in `shared/models/`, metrics written down, CPU/NNPA parity recorded.

### Phase 3 (H6 to H10): Scoring service and containers

- [ ] Inference adapters behind one interface, selected by env var:
  - `AnomalyScorer`: `torch_nnpa` | `torch_cpu` | `numpy`
  - `ProviderScorer`: `xgboost` | `numpy_trees` (last resort)
- [ ] Fusion in `fusion.py`, thresholds in `config/policy.json`:
  - `provider_risk >= T_high` AND (`anomaly >= T_anom` OR a hard error rule) -> `HOLD_FOR_REVIEW`
  - `anomaly >= T_anom` OR a hard error rule (e.g. claim after date of death), with `provider_risk < T_high` -> `RETURN_WITH_FIX`
  - otherwise -> `PAY`
  - Tune `T_high` on validation so HOLD stays at or under about 2% of claims.
- [ ] FastAPI endpoints:
  - `POST /claims` (service token): validate -> pseudonymise -> features -> score -> fuse -> encrypt and store -> audit -> SSE event -> queue explanation if not PAY -> return decision, reasons, `latency_ms`
  - `GET /stream` (SSE), `GET /metrics` (p50, p99, throughput, decision mix, $ held)
  - `GET /cases`, `POST /cases/{id}/resolve` (investigator)
  - `POST /auth/login` (demo users per role), `GET /audit/verify` (auditor)
- [ ] Reasons are always produced without the LLM: top features by per-feature error (claims) and peer percentiles (providers), e.g. "claims per patient is at the 99th percentile of peers".
- [ ] `deploy/api.Dockerfile`, `deploy/granite.Dockerfile`, `deploy/simulator.Dockerfile`, `docker-compose.yml`:
  - all three build `FROM` the L1CC Jupyter image (one pull, shared layers)
  - internal bridge network (not `--network host`); only `api` published on `127.0.0.1:8000`
  - `shared/models` mounted read-only, a named volume for SQLite
  - healthchecks, non-root users, secrets as files (not baked into images, not in git), `mem_limit`s from Section 5
  - `jupyter` under profile `dev`, `simulator` under profile `demo`
- [ ] `simulator/replay.py`: replays test claims at a configurable rate, with a burst mode for the demo.

**Exit:** `docker compose --profile demo up` on the VM streams 1k claims; decisions visible via `curl`; p50 and p99 recorded.

### Phase 4 (H10 to H13): Security in the core

- [ ] **Pseudonymisation:** HMAC-SHA256 with a secret key over `BeneID`, `Provider`, physician IDs before features run. Raw IDs only exist inside the encrypted payload.
- [ ] **Encryption at rest:** AES-256-GCM over the full claim payload, random 96-bit nonce per record, key from a Docker secret file, key id stored per row for rotation.
- [ ] **CPACF evidence:** save `openssl speed -evp aes-256-gcm` output from the VM. OpenSSL on s390x uses CPACF instructions for AES-GCM and SHA. `icastats` only counts operations routed through libica, so don't present it as evidence unless ibmca is configured.
- [ ] **RBAC with JWT (HS256):**
  - `provider`: own decisions and fix notes only
  - `investigator`: held cases with decrypted details
  - `auditor`: aggregates and audit log, never PII
- [ ] **Tamper-evident audit:** append-only table, each row stores `sha256(prev_hash + canonical_row_json)`. `/audit/verify` recomputes the chain. Demo moment: edit one row in SQLite, verification fails and names the row.
- [ ] **Container and host hardening:** non-root, minimal published ports, read-only mounts, Jupyter token changed and port 38888 not opened to the internet (4.3).
- [ ] Stretch: sign audit checkpoints with a quantum-safe algorithm (ML-DSA) if the VM's OpenSSL lists it: `openssl list -signature-algorithms | grep -i ml-dsa`. Only claim it if it actually ran.
- [ ] `docs/SECURITY.md`: each control, where it lives in code, and its evidence file.

**Exit:** security tests pass (`tests/test_security.py`, `tests/test_audit.py`); SECURITY.md complete.

### Phase 5 (H13 to H16): Dashboard and Granite

- [ ] Dashboard at `/ui` (static HTML, vanilla JS, Chart.js from a CDN), served by the api:
  - role picker and login
  - live feed via SSE: claim token, amount, decision chip, latency
  - KPIs: claims/sec, p50 and p99 latency, decision mix, dollars held, "100% of claims scored", accelerator badge (`NNPA` or `CPU`, from the adapter actually in use)
  - investigator: provider risk leaderboard, case drawer with reasons and Granite brief
  - provider: returned claims with fix notes
  - auditor: aggregates and a "Verify audit chain" button
- [ ] Granite worker (separate container, 4.5): polls the explanation queue in SQLite, builds the prompt from **structured reasons only** (pseudonymised tokens, feature names, values, peer percentiles, never raw PII), calls `llama_cpp.Llama.create_chat_completion`, stores the note, the api pushes an SSE update.
  - **Correction note** (to provider): friendly, specific, says which field looks wrong and what to check.
  - **Investigator brief:** neutral, evidence-based, states uncertainty, explicitly "not a finding of fraud".
  - `temperature 0.2`, short max tokens, timeout, fall back to the templated reasons on any failure.

**Exit:** the full demo path works with Granite on AND off.

### Phase 6 (H16 to H18): Benchmarks and responsible AI

- [ ] `scripts/bench.py` on the VM, results to `docs/evidence/bench/` as CSV and PNG:
  1. Autoencoder: torch_nnpa vs torch CPU, batch 1 and batch 32, 10k iterations, p50, p99, throughput.
  2. Provider model: XGBoost predict latency, batch 1 and batch 32.
  3. Granite: tokens/sec and time-to-first-token (zDNN build), for the record.
  4. End to end: `POST /claims` at fixed concurrency, p50, p99, claims/sec.
- [ ] Report numbers honestly, including any case where NNPA is not faster at batch 1 (small models can be dominated by dispatch overhead). Explain what it means.
- [ ] Fairness on the test split (`notebooks/04_fairness.ipynb`): flag rate and false positive rate by patient gender, race code, age band and state. Write disparities and mitigations into MODEL_CARD.md.
- [ ] Pick 3 example claims per decision type for the demo.
- [ ] **Feature freeze.**

### Phase 7 (H18 to H21): Demo, video, slides

- [ ] Evidence screenshots from the VM (the judges explicitly want screenshots of workloads running live on L1CC, including **a Jupyter notebook running on LinuxONE**): probe output (`s390x`, z17 machine type, `nnpa`), training notebook with outputs, `docker compose ps`, benchmark chart, audit verify, `openssl speed`.
- [ ] `docs/DEMO_SCRIPT.md`, fitted to the official video limit (**confirm the number**):
  1. Problem and persona (Philip Review numbers), Theme 1 named.
  2. Architecture on LinuxONE in one slide.
  3. Live: the stream runs; a claim dated after the patient's death is returned with a Granite fix note in milliseconds; a high-risk provider pattern is held with an investigator brief; KPIs show every claim scored with p99 latency and the NNPA badge.
  4. Security: pseudonymised IDs, encrypted payloads, three role views, tamper test failing.
  5. Benchmarks and responsible AI: human in the loop, fairness table, honest hardware context.
  6. Impact (quantified beneficiaries) and team roles and contributions.
- [ ] Slides (6 to 8): problem, solution, architecture, demo stills, security, benchmarks, responsible AI, impact and team.
- [ ] README: what it is, one-command run, architecture, results, limitations, sources, team roles, a statement that all code was written during the event, and AI-assistant disclosure if applicable.

### Phase 8 (H21 to H24): Buffer and submission

- [ ] Fresh clone on the VM, follow the README exactly, fix anything that breaks.
- [ ] Submit repo, video, screenshots and slides via the official portal. Confirm the deadline time zone (global event).

## 7. Command cheat sheet (verify flags against the docs)

```bash
# VM setup (Ubuntu 24.04-Datathon)
chmod 600 key.pem && ssh -i key.pem linux1@148.100.X.X
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
sudo usermod -aG docker $USER; newgrp docker
sudo systemctl start docker; exec bash

# accelerator check
lscpu | grep -io nnpa
grep -E '^(Type|Model)' /proc/sysinfo

# Jupyter (see 4.3 for the token override; do NOT open 38888 in iptables)
docker pull registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest
docker logs notebook; docker rm -f notebook     # redeploy

# what's in the image
docker exec notebook bash -c 'ls -la /home/ibm-user/*.gguf; pip list 2>/dev/null | grep -Ei "torch|xgboost|llama|zdnn"'

# memory and disk
docker stats --no-stream; free -h; df -h
docker image prune -a

# Granite fallback, on the laptop: big-endian GGUF at q8_0 (fits in 8 GB)
python convert_hf_to_gguf.py --outtype q8_0 --bigendian \
  --outfile granite-3.3-2b-instruct-be.q8_0.gguf granite-3.3-2b-instruct/
scp -i key.pem granite-3.3-2b-instruct-be.q8_0.gguf linux1@148.100.X.X:~/shared/models/

# stretch: zDLC compile (needs icr.io credentials; check IBM/zDLC README for flags)
docker run --rm -v "$PWD/shared/models:/workdir:z" icr.io/ibmz/zdlc:<ver> \
  -O3 --EmitLib --mcpu=z17 --maccel=NNPA /workdir/claim_ae.onnx -o /workdir/claim_ae_nnpa

# stretch: run as a Kubernetes-style pod with Podman
podman kube play deploy/claimsense-pod.yaml
```

## 8. Fallback ladder (decide fast, log the decision)

| If this fails | Do this instead |
|---|---|
| Registered without `SSF2026` / no `AI_LAB` flavour | Email linuxone@us.ibm.com to delete the registration, re-register with the code. Meanwhile develop locally. |
| L1CC Jupyter image pull | Retry; check `df -h`. Else `python:3.12-slim` + `pip install torch xgboost` (check s390x wheels) and accept CPU-only. |
| torch_nnpa unclear or failing | PyTorch CPU in the api (same model). Keep trying NNPA in the benchmark notebook only. Say so honestly in the pitch. |
| PyTorch in the api at all | NumPy forward pass from `claim_ae_weights.npz`. |
| `cryptography` wheel/build | `apt install python3-cryptography` won't reach the container venv; try `pip install cryptography` with Rust + OpenSSL headers in the Dockerfile; last resort PyCryptodome, noting it does not use CPACF. |
| Granite instruct model missing | Laptop q8_0 big-endian conversion (Section 7). Else the in-image 350m base model with few-shot prompts. Else templated reasons. |
| Granite too slow or OOM | Lower `n_ctx` to 1024, cut max tokens, generate notes only for HOLD cases, stop Jupyter during the demo. |
| Docker Compose missing | `sudo apt install docker-compose-plugin`, else `podman kube play deploy/claimsense-pod.yaml`. |
| VM unresponsive | Dashboard: stop, wait for SHUTOFF, start. |

## 9. Repo layout

```
claimsense/
  CLAUDE.md  README.md  .gitignore  .env.example  docker-compose.yml
  config/policy.json
  deploy/  api.Dockerfile  granite.Dockerfile  simulator.Dockerfile  claimsense-pod.yaml
  notebooks/  01_eda.ipynb  02_provider_model.ipynb  03_claim_autoencoder.ipynb  04_fairness.ipynb
  src/claimsense/
    features.py  schemas.py  security.py  audit.py  fusion.py  explain.py
    inference/  base.py  anomaly_torch.py  anomaly_numpy.py  provider_xgb.py  provider_numpy.py
    api/  main.py  auth.py  routes_claims.py  routes_cases.py  routes_stream.py
    ui/  index.html  app.js  styles.css
  granite_worker/worker.py
  simulator/replay.py
  scripts/  probe.sh  remote.sh  sync.sh  tunnel.sh  bench.py
  tests/  test_features.py  test_fusion.py  test_security.py  test_audit.py
  docs/  ENV_REPORT.md  MODEL_CARD.md  SECURITY.md  DEMO_SCRIPT.md  evidence/
  shared/  (gitignored: data/, models/)
```

## 10. Definition of done

- [ ] One command brings the whole stack up on L1CC, and the README reproduces it from a fresh clone.
- [ ] Live demo recorded on L1CC; evidence screenshots (including a Jupyter notebook on LinuxONE) in `docs/evidence/`.
- [ ] PyTorch (claim autoencoder, NNPA where it works) and XGBoost (provider risk) both used for real; Granite (llama.cpp + zDNN) produces the explanations.
- [ ] Pseudonymisation, AES-GCM at rest, RBAC, tamper-evident audit, secrets outside images and git, Jupyter not exposed.
- [ ] Benchmarks with honest hardware context.
- [ ] MODEL_CARD with metrics, fairness results and limitations.
- [ ] Video within the limit; team roles stated.

## 11. Team roles (fill in)

- Lead and infra (VM, containers, benchmarks):
- ML (features, models, fairness):
- Backend and security (api, crypto, RBAC, audit):
- UI and pitch (dashboard, video, slides):

## 12. Sources for the pitch and README

- Philip Review numbers: https://www.health.gov.au/ministers/the-hon-mark-butler-mp/media/medicare-non-compliance-and-fraud-costing-australians
- Philip Review final report: https://apo.org.au/node/322212
- Real-time claim checking as a proposed fix: https://www.medicalrepublic.com.au/medicare-old-and-leaky-but-no-8bn-in-sight/88847
- Telum and in-transaction fraud detection: https://www.ibm.com/new/announcements/ibm-telum-processor-the-next-gen-microprocessor-for-ibm-z-and-ibm-linuxone
- Dataset: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
- llama.cpp on s390x: https://github.com/ggml-org/llama.cpp/blob/master/docs/build-s390x.md

### Official datathon links (from the kickoff)

- Datathon page and FAQ: https://community.ibm.com/zsystems/datathon/
- L1CC landing page: https://community.ibm.com/zsystems/l1cc/
- L1CC registration (event code `SSF2026`): https://linuxone.cloud.marist.edu/#/register?flag=VM
- L1CC instructions: https://ibm.biz/BdPcL8
- L1CC dashboard: https://linuxone.cloud.marist.edu/#/login
- ML lab guide, image and package list: https://github.com/linuxone-community-cloud/jupyter-lab-ml (`packages.txt`, `README_RHEL.md`)
- Virtual server deployment guide: https://github.com/linuxone-community-cloud/technical-resources/blob/master/faststart/deploy-virtual-server.md
- Windows PuTTY guide: https://github.com/linuxone-community-cloud/technical-resources/blob/master/faststart/PUTTY_Set_up.pdf
- L1CC support: linuxone@us.ibm.com
- Stretch only: IBM Z Accelerated for Snap ML https://github.com/IBM/ibmz-accelerated-for-snapml, IBM Z Deep Learning Compiler https://github.com/IBM/zDLC
