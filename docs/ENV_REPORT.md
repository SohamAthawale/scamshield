# Environment report (IBM LinuxONE Community Cloud, 2026-10-17)

Everything here was run on the VM; raw outputs are in `docs/evidence/`.

## Machine

| Item | Value | Evidence |
|---|---|---|
| Architecture | s390x, big-endian | `probe-*.txt` |
| Machine type | 9175 (IBM z17 / LinuxONE 5), model 702 | `probe-*.txt` (`/proc/sysinfo`) |
| AI accelerator | `nnpa` present (Telum II); `torch_nnpa.NNPAModule.is_telum2_available()` = True | `probe-*.txt`, `bench/nnpa_smoke.txt`, notebook 02 |
| VM | Ubuntu 24.04.5 LTS (`Ubuntu24.04-Datathon`), flavour `AI_LAB`, 3 vCPU, 7.8 GiB RAM, 50 GB disk | dashboard, `probe-*.txt` |
| Crypto | OpenSSL 3.0.13 on the host, 3.5.8 in the container; AES-GCM and SHA-256 on CPACF | `probe-*.txt` (`openssl speed`) |
| Quantum-safe signatures | No ML-DSA in the host OpenSSL 3.0.13, so we do not claim it | `probe-*.txt` |

## Container stack

| Item | Value |
|---|---|
| Docker | 29.8.0 with Compose v2 (installed with get.docker.com; not preinstalled) |
| Base image | `registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest` (9.85 GB on disk): RHEL 10.2 UBI, uid 1000 `ibm-user`, Python 3.12.15 venv at `/opt/venv` |
| In the image | torch 2.11.0+cpu, torch_nnpa 1.5.0, xgboost 3.4.1, llama_cpp_python 0.3.36 (with libggml-zdnn), numpy 2.0.2, pandas 3.0.6, scikit-learn 1.9.0 |
| Not in the image | fastapi, uvicorn, PyJWT, cryptography |
| `cryptography` | Has no s390x wheel and its source build fails out of the box. Our Dockerfile installs `rust cargo openssl-devel` (UBI repos), builds cryptography 50.0.2 against OpenSSL 3.5.8, then removes the toolchain |
| App image | `deploy/app.Dockerfile`, FROM the L1CC image; one image for the api, the Granite worker and the simulator |

## Data

The data is IBM AMLworld **HI-Small** (`HI-Small_Trans.csv`, `HI-Small_Patterns.txt`), CDLA-Sharing-1.0. No Kaggle token was available, so we used two Hugging Face mirrors:
- `OsamaMIT/IBM-AML-HI-Small`: CSV, patterns and licence.
- `eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans`: a zipped CSV.

The extracted CSV from the second mirror has the **same SHA-256** as the first mirror's published LFS hash: `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` (5,078,345 rows). The VM downloaded it directly and verified the hash.

## Decisions (and why)

1. **Pivot from ClaimSense (Medicare claims) to ScamShield (payments).**
   - ClaimSense is complete and stays on the `main` branch.
   - Reasons for the pivot:
     - scams are an urgent Australian problem;
     - in-transaction payment scoring is Telum's home turf;
     - IBM's own AML data has per-payment labels and typed laundering schemes.
2. **One streaming feature engine** for training and serving (no batch re-implementation).
   - Parity holds by construction and was verified: 50,000 test payments, 100% identical decisions.
   - Speed: about 4.4 µs per payment; the 4.2M-payment history loads in 61 s on the z17.
3. **PyTorch on the NNPA through `torch_nnpa`** (`model.to(torch_nnpa.device())`). Payment autoencoder, real test payments (`bench/bench_autoencoder.csv`):

   | Payments per call | NNPA vs CPU |
   |---|---|
   | 1 | 0.70× |
   | 256 | 0.92× |
   | 1,024 | **1.21×** |
   | 4,096 | **1.42×** |
   | 16,384 (chunked) | **1.36×** |

   - Scores differ from the CPU by at most 0.007 (threshold 0.90).
   - The API therefore sends calls of ≥1,024 payments to the NNPA (chunks of ≤4,096) and allows batches up to 4,096, so bursts shift onto the accelerator.
4. **XGBoost on the CPU** for risk: about 85 µs per payment, 274k payments/s at batch 4,096 (`bench/bench_risk.csv`).
5. **Granite: `granite-3.3-2b-instruct-be.Q8_0.gguf`** (big-endian GGUF from `taronaeo/Granite-3.3-2B-Instruct-BE-GGUF`, Apache-2.0, SHA-256 checked). The image only ships a 350M *base* model. Measured (`bench/granite_options.txt`):
   - Speed: Q8_0 generates at about 1.5 tok/s on 3 threads, the best of the 2B builds.
   - The zDNN `op_offload` setting makes no difference.
   - Reusing the prefix KV cache cuts later prompts from about 23 s to about 7 s.
   - Result: about 1–2 minutes per note, off the hot path at low CPU priority (`cpu_shares: 128`), behind an instant template message.
6. **Security deviation from the lab guide (deliberate):** we never open port 38888 to the internet and never keep the default Jupyter token. Only the API is published, on `127.0.0.1:8000`, reached through the SSH tunnel.

## Gotchas hit (and fixes)

| Problem | Fix |
|---|---|
| The last 8 days of HI-Small are mostly laundering (generator artifact) | Cut at 2022-09-11 |
| The API scored with a different model from the one the thresholds were tuned on | The saved XGBoost booster still held the early-stopping patience rounds; it is now saved trimmed to the best iteration (found by the parity check) |
| `docker compose` refused to run: `JUPYTER_TOKEN` missing | A server-side `.env` (never synced or committed) |
| The API could not open SQLite: a new named volume is owned by root | The image creates `/data` owned by uid 1000 |
| Granite took all 3 vCPUs and raised scoring p99 | `cpu_shares: 128` on the worker |
| An SSH session dropped during long runs | Long jobs run with `nohup ... > log` on the VM, and we poll the log |
| A Granite brief invented a fact | The prompts now forbid any number not in the facts, ambiguous feature labels were reworded, and the disclaimer check catches "not a *definitive* finding" |
| macOS only: torch imported before xgboost crashes or deadlocks training | `import xgboost` first, plus `OMP_NUM_THREADS=1` locally. Linux s390x is unaffected |
