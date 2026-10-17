# Security controls

| Control | What it does | Where in code | Evidence / test |
|---|---|---|---|
| Pseudonymisation | HMAC-SHA256 (keyed) tokens for patient, provider, physician and claim IDs. They are computed on arrival, before any feature runs. Models, logs, the UI and Granite only ever see tokens | `security.Pseudonymizer`, `api/main.py` `Service._input` | `tests/test_security.py`, `tests/test_store.py` (raw IDs absent from the DB file) |
| Encryption at rest | The full raw claim is sealed with AES-256-GCM. Each record gets a random 96-bit nonce and a key id (for rotation), and the claim token is used as associated data, so a ciphertext moved to another row fails to decrypt | `security.Envelope`, `store.Store.save_batch` | `tests/test_security.py`, `tests/test_store.py` |
| CPACF | `cryptography` 50.0.2 is built in our image against OpenSSL 3.5.8. On IBM Z, OpenSSL's AES-GCM and SHA-256 use the CPACF on-core crypto instructions | `deploy/app.Dockerfile` | `docs/evidence/probe-*.txt` (`openssl speed` on the VM) |
| Role-based access (JWT HS256, 8 h) | `provider`: only its own returned/held claims and fix notes, with no scores or other providers' data. `investigator`: held cases, decrypted on open, plus leaderboard and briefs. `auditor`: aggregates and the audit chain only. `service`: can only submit claims | `api/auth.py`, `store.provider_claims` | `tests/test_api.py` (role matrix, SSE filtering) |
| Passwords | scrypt (n=2^14, r=8, p=1) with a random salt and constant-time comparison. Demo passwords are random and written only to `secrets/demo_passwords.txt` (mode 600) | `security.hash_password` | `tests/test_security.py` |
| Tamper-evident audit | Append-only SQLite table. SQLite triggers refuse UPDATE and DELETE, and each row stores `sha256(prev_hash + canonical_row)`. `verify()` finds edits, inserts, reorders and deletions, and an external anchor (the last committed head plus a checkpoint file outside the DB) catches truncation | `audit.AuditLog`, `GET /audit/verify` | `tests/test_audit.py`, `tests/test_store.py` |
| What gets audited | Every decision, login, failed login, case view (with decryption), case resolution, Granite note and service start | `store.py`, `api/main.py` | dashboard → Auditor |
| Decisions are never overwritten | Resolving an already-resolved case is refused | `store.resolve_case` | `tests/test_api.py` |
| Secrets | Secrets live in files in `./secrets` (mode 700/600) mounted read-only at `/run/secrets`. They are never baked into images or committed (`.gitignore`) | `scripts/make_secrets.py`, `docker-compose.yml` | — |
| Containers | Containers run as non-root (uid 1000) with a read-only root filesystem, `tmpfs /tmp`, `cap_drop: ALL`, `no-new-privileges` and memory limits | `docker-compose.yml` | `docker compose ps` |
| Network | Only the API is published, and only on `127.0.0.1:8000`; access is through the SSH tunnel. Jupyter uses its own token, is bound to 127.0.0.1, and the lab guide's public firewall rule for port 38888 was deliberately skipped | `docker-compose.yml`, `scripts/tunnel.sh` | `docs/ENV_REPORT.md` |
| LLM data minimisation | Granite gets only structured, pseudonymised facts. Output that looks like a raw ID is rejected, and briefs carry a "not a finding of fraud" disclaimer | `explain.build_messages`, `granite_worker.GraniteWriter` | `tests/test_granite_worker.py`, `tests/test_explain.py` |
| On-premises AI | All inference (PyTorch/NNPA, XGBoost, Granite) runs on the LinuxONE VM. No claim data leaves the machine | — | — |

## Not claimed

- **Quantum-safe signatures:** the host OpenSSL 3.0.13 lists no ML-DSA (`probe-*.txt`), so audit checkpoints are not signed.
- **`icastats` counters:** we don't present these as evidence, because plain OpenSSL doesn't go through libica.
- **SSE token in the URL:** the browser's EventSource can't set headers, so the stream takes `?token=`. That's acceptable behind the SSH tunnel; production would use a cookie.

## Demo: the tamper test

1. Auditor view → **Verify audit chain** → "Chain intact: N entries".
2. On the VM, edit one row, bypassing the triggers:
   ```bash
   docker compose exec api python -c "import sqlite3; c=sqlite3.connect('/data/claimsense.db'); c.execute('DROP TRIGGER audit_no_update'); c.execute(\"UPDATE audit SET details='{}' WHERE id=50\"); c.commit()"
   ```
3. **Verify audit chain** again → "Chain broken at entry #50: row content does not match its hash".
