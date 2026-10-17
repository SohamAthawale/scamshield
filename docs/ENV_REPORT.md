# Environment report (IBM LinuxONE Community Cloud, 2026-10-17)

Everything here was run on the VM; raw outputs are in `docs/evidence/`.

## Machine

| Item | Value | Evidence |
|---|---|---|
| Architecture | s390x, big-endian | `probe-*.txt` |
| Machine type | 9175 (IBM z17 / LinuxONE 5), model 702 | `probe-*.txt` (`/proc/sysinfo`) |
| AI accelerator | `nnpa` present (Telum II); `torch_nnpa.NNPAModule.is_telum2_available()` = True | `probe-*.txt`, `bench/nnpa_smoke.txt` |
| VM | Ubuntu 24.04.5 LTS (`Ubuntu24.04-Datathon`), flavour `AI_LAB`, 3 vCPU, 7.8 GiB RAM, 50 GB disk | dashboard, `probe-*.txt` |
| Crypto | OpenSSL 3.0.13 on the host; 3.5.8 inside the container. AES-GCM and SHA-256 use CPACF | `probe-*.txt` (`openssl speed`) |
| Quantum-safe signatures | No ML-DSA in the host OpenSSL 3.0.13, so we do not claim it | `probe-*.txt` |

## Container stack

| Item | Value |
|---|---|
| Docker | 29.8.0 with Compose v2 (installed with get.docker.com; not preinstalled) |
| Base image | `registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest` (9.85 GB on disk). It is RHEL 10.2 UBI, runs as uid 1000 `ibm-user`, uses `tini` as entrypoint, and has a Python 3.12.15 venv at `/opt/venv` |
| In the image | torch 2.11.0+cpu, torch_nnpa 1.5.0, xgboost 3.4.1, llama_cpp_python 0.3.36 (with libggml-zdnn), numpy 2.0.2, pandas 3.0.6, scikit-learn 1.9.0, onnx 1.22.0 |
| Not in the image | fastapi, uvicorn, PyJWT, cryptography |
| `cryptography` | Has no s390x wheel and its source build fails out of the box. The fix: our Dockerfile installs `rust cargo openssl-devel` from the UBI repos with dnf, builds cryptography 50.0.2 against OpenSSL 3.5.8, then removes the toolchain |
| App image | `deploy/app.Dockerfile`, FROM the L1CC image; one image for the api, the Granite worker and the simulator |

## Decisions (and why)

1. **PyTorch on the NNPA through `torch_nnpa`** (`model.to(torch_nnpa.device())`), not zDLC/Snap ML. It is already in the image and needs no extra registry credentials.
   - Results match the CPU within 1e-2 per output.
   - The NNPA is slower than the CPU up to about 64 claims per call, at parity at 256, 1.37x faster at 1,024 and 1.61x at 4,096, and slower for one 16,384 call.
   - The API therefore sends batches of 256 or more to the NNPA in chunks of at most 4,096 (`bench/bench_autoencoder.csv`).
2. **XGBoost on the CPU** for provider risk. It scores one provider in about 85 µs.
3. **Granite: `granite-3.3-2b-instruct-be.Q8_0.gguf`** (big-endian GGUF from `taronaeo/Granite-3.3-2B-Instruct-BE-GGUF`, Apache-2.0, SHA-256 checked). The image only ships `granite-4.0-h-350m-base-be.f16.gguf`, a base model that doesn't follow instructions. Measured with 3 threads (`bench/granite_*.txt`):
   - 2B Q4_K_M: about 1.3 tok/s generation and 5 tok/s prompt. Its K-quants seem to lack a fast s390x path.
   - 2B Q4_0: 1.1 tok/s. 2B Q8_0: 1.46 tok/s, the best quality and the fastest of the 2B builds.
   - 1B-A400M (MoE): 6 to 8 tok/s, but it broke the "don't accuse" instruction, so we rejected it.
   - `op_offload` (zDNN) on or off makes no difference to prompt speed.
   - Reusing the prefix KV cache cuts the prompt cost of later notes from about 23 s to about 7 s.
   - Result: Granite writes a note in about 1 to 2 minutes on 3 vCPUs. It runs off the hot path at low CPU priority (`cpu_shares: 128`); a template note is shown at once and upgraded when Granite finishes.
4. **The dataset comes from a GitHub mirror** of the Kaggle files (no Kaggle token was available). Files are byte-identical in name and the row and label counts match the Kaggle card: 5,410 providers (506 flagged), 138,556 beneficiaries, 40,474 inpatient and 517,737 outpatient claims. SHA-256 sums are in `shared/data/SHA256SUMS`.
5. **Security deviation from the lab guide (deliberate).** We never open port 38888 to the internet and never keep the default Jupyter token. Only the API is published, on `127.0.0.1:8000`, and everything is reached through the SSH tunnel.

## Gotchas hit (and fixes)

| Problem | Fix |
|---|---|
| `docker compose` refused to run: `JUPYTER_TOKEN` missing | A server-side `.env` (never synced or committed) |
| API could not open SQLite: a new named volume is owned by root | The image creates `/data` owned by uid 1000 before the volume is first mounted |
| Granite took all 3 vCPUs and raised scoring p99 | `cpu_shares: 128` on the worker; scoring p99 went back to 15 ms at 50 claims/s with Granite busy |
| SSH session dropped during long runs | Long jobs run with `nohup ... > log` on the VM, and we poll the log |
| macOS only: torch imported before xgboost segfaults or deadlocks training | `import xgboost` first, plus `OMP_NUM_THREADS=1` for local training. Linux s390x is unaffected |
| Python output buffered when redirected to a file | `PYTHONUNBUFFERED=1` |
| Restarting the API lost provider histories | At startup the API decrypts stored claims in arrival order and replays them into the rolling state (20,201 claims in 1.3 s) |
