# ScamShield: Master Plan

IBM Z Datathon 2026 (24 hours, October 17-18, 2026), **Theme 1: Real-Time AI for Critical Decisions** ("Detect. Decide. Respond.")

Real-time scam and mule-account interception on IBM LinuxONE. Score every payment before the money moves: let normal payments through, warn customers about out-of-character or risky payments, and pause payments that match laundering/mule patterns for a call-back, with an evidence brief for the analyst.

> Branch `scamshield`. The earlier project, ClaimSense (Medicare claim integrity), is complete on `main` and stays there.

---

## 0. How to use this file (Claude Code, read first)

- This file is the source of truth. Tick checkboxes as tasks finish and commit after each task.
- Record every environment finding in `docs/ENV_REPORT.md`. Any claim that something works on IBM Z must come from a command actually run on the VM, with output under `docs/evidence/`.
- Numbers in README / MODEL_CARD / slides must come from `docs/evidence/` (VM runs). Never round up or invent.

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


## Status (2026-10-17)

**Built and running on the z17 VM** (address in your local `.env`, never committed): api + Granite worker in Docker Compose (`~/scamshield`), data in `~/shared/scamshield`.

Done:
- [x] Data: IBM AMLworld HI-Small (two mirrors, identical SHA-256), artifact days after 2022-09-10 cut, time split 1-6 / 7-8 / 9-10 Sep
- [x] One streaming feature engine for training and serving (exact parity verified on 50k payments)
- [x] XGBoost risk (test ROC-AUC 0.984, PR-AUC 0.564) + PyTorch customer-behaviour autoencoder on Telum II NNPA + big-new-payee rule
- [x] Policy tuned on validation: test hold 0.11% at 53% precision, 82% of laundering warned or paused, $269M of $424M paused, 150/174 schemes broken
- [x] API (micro-batching, SSE, RBAC customer/analyst/auditor/service), encrypted store, audit chain with anchor, history bootstrap (4.2M payments, 61 s)
- [x] Granite worker (customer messages + analyst briefs), dashboard, simulator with injected victim/honest payments
- [x] NNPA vs CPU benchmark (1.2x at 1,024, 1.4x at 4,096), notebooks 01-03 executed on LinuxONE, fairness audit
- [x] README, MODEL_CARD, SECURITY, ENV_REPORT, DEMO_SCRIPT

Still to do (people, not code):
- [ ] Confirm with organisers: video time limit, deadline time zone, **whether AI coding assistants are allowed** (disclose in README if so)
- [ ] Fill in team roles (README)
- [ ] Record the demo video following `docs/DEMO_SCRIPT.md`; screenshots (Jupyter on LinuxONE, terminal with `nnpa`, dashboard, tamper test)
- [ ] Decide which branch/repo to submit (ScamShield on `scamshield`, ClaimSense on `main`)

## 1. Hard rules

1. All solution code is written in this repo during the datathon window (Oct 17-18). Do not copy code from other projects, tutorials or the four reference lab notebooks.
2. Target platform is s390x (big-endian). Verify every dependency on the VM.
3. Never commit data, model weights, keys, tokens, `.env`, `secrets/`, `*.pem`.
4. Only public synthetic data (IBM AMLworld). No real customer data.
5. The LLM never sits on the hot path. Payments are decided without waiting for Granite.
6. Memory budget is 8 GB for everything; stop Granite when running notebooks or benchmarks.

## 2. Rubric map

| Criterion (max 10 each) | How ScamShield hits it |
|---|---|
| Technical Design | Live on L1CC (z17, `nnpa`); PyTorch on Telum II via `torch_nnpa` + XGBoost + Granite (llama.cpp/zDNN); Docker Compose from the L1CC image; CPACF crypto, RBAC, tamper-evident audit; measured NNPA crossover. |
| Novelty & Originality | Two-sided hybrid: mule/laundering risk (supervised, receiver-side history) + customer-behaviour anomaly (unsupervised, victim side) + scam rule, fused into allow/warn/hold; Granite writes different text for customer and analyst; scheme-level evaluation (150/174 schemes broken). Different domain and design from the labs. |
| Pitch & Delivery | `docs/DEMO_SCRIPT.md` (5 min), personas: customer + scam analyst. |
| Community Value | $2.03B lost to scams in Australia (2024); Scams Prevention Framework; warnings never block, holds go to a human; fairness by currency/format/bank size; honest limitations. |

## 3. The story

- **Problem:** Australians lost $2.03B to scams in 2024 (ACCC/National Anti-Scam Centre, Targeting Scams 2024). Money moves through mule accounts within minutes and is rarely recovered. The Scams Prevention Framework (2025) requires banks to detect, disrupt and respond.
- **Personas:** a customer being talked into a payment; the bank's scam analyst.
- **Value:** every payment checked in milliseconds before it leaves; honest customers barely notice; risky payments get a specific warning; mule-pattern payments are paused for a call-back.
- **Why a mainframe:** banks' payments already run on IBM Z; Telum puts the AI on the same chip, so 100% of payments are scored in-transaction.

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
core banking --POST /payments--> [api: FastAPI]
   pseudonymise accounts (HMAC) -> micro-batch -> streaming features (payment, payer, receiver/mule)
   -> autoencoder (Telum II NNPA for >=1,024/call, else CPU) + XGBoost risk + scam rule
   -> ALLOW / WARN / HOLD -> AES-256-GCM payload + hash-chained audit (SQLite WAL)
   -> instant customer message; Granite rewrite queued -> [granite worker: llama.cpp + zDNN]
   -> SSE /stream -> dashboard (analyst / customer / auditor)
```

## 6. Sources

- Scams: https://www.scamwatch.gov.au/research-and-resources/targeting-scams-report
- Data: IBM Transactions for Anti Money Laundering (AMLworld), https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (mirrors: huggingface.co/datasets/OsamaMIT/IBM-AML-HI-Small)
- Telum and in-transaction fraud detection: https://www.ibm.com/new/announcements/ibm-telum-processor-the-next-gen-microprocessor-for-ibm-z-and-ibm-linuxone
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
