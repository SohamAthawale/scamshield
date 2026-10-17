# Demo script

Written for **5 minutes**. Confirm the official limit and cut the parts marked *(optional)* if it's shorter.

## Before recording (10 min)

1. `scripts/tunnel.sh` on the laptop, so <http://localhost:8000/ui> works.
2. Fresh state with history already built:
   ```bash
   docker compose down -v && docker compose up -d api
   docker compose --profile demo run --rm simulator python -m claimsense.simulator --url http://api:8000 --limit 20000 --rate 0 --inject 0
   docker compose up -d granite
   ```
3. Start the live stream in a second VM terminal (it runs about 10 minutes):
   ```bash
   docker compose --profile demo run --rm simulator python -m claimsense.simulator --url http://api:8000 --offset 20000 --limit 12000 --rate 20 --inject 0.03 --concurrency 8
   ```
4. Open three browser windows, logged in as **investigator**, **provider** and **auditor** (`secrets/demo_passwords.txt`).
5. In a terminal on the VM, have ready: `lscpu | grep -io nnpa`, `grep Type /proc/sysinfo`, `docker compose ps`.

## Script

| Time | Show | Say |
|---|---|---|
| 0:00–0:35 | Title slide, then the problem slide | "Medicare loses $1.5–3B a year, mostly to billing errors, not fraud (Philip Review). Today, checks happen after payment, on samples. ClaimSense checks **every** claim **before** payment, in milliseconds. Theme 1: real-time AI for critical decisions." |
| 0:35–1:00 | Persona slide | "Two users: a GP practice manager who wants to bill correctly, and an integrity investigator with limited time. Honest mistakes get a fix; only suspicious patterns go to a human." |
| 1:00–1:30 | VM terminal: `uname -m`, machine type 9175, `nnpa`, `docker compose ps` | "This is running live on an IBM z17 in the LinuxONE Community Cloud. Three containers built from the official L1CC image: the scoring API, a Granite worker, and the claim stream." |
| 1:30–2:30 | Investigator dashboard | "Claims stream in: 20 a second, p50 under 4 ms, p99 around 15 ms, 100% scored. Badge: the claim model runs on the Telum II on-chip AI accelerator." Open a held case: "Held because this claim breaks a rule *and* the provider's pattern is high-risk: largest amounts at the 97th percentile of peers. Opening it decrypted the claim and wrote an audit entry. IBM Granite, running here on LinuxONE, wrote this brief from pseudonymised facts only, and it says plainly it is not a finding of fraud." Click **Request records**. |
| 2:30–3:05 | Provider (billing office) view | "The practice manager sees only their own returned claims, with a plain-English fix. The service date is after the patient's date of death, or the patient has no Part B coverage. Fix it and resubmit. No scores, no accusations." |
| 3:05–3:40 | Auditor view, then the tamper command (SECURITY.md), then Verify again | "Every decision, login and case view is hash-chained. Verify: intact. Now I edit one row directly in the database, bypassing the triggers. Verify: broken at that exact entry. Raw IDs are pseudonymised and payloads are AES-256-GCM encrypted on CPACF." |
| 3:40–4:20 | Benchmark chart, then the injected-errors table | "Telum II: 1.6× the CPU at 4,096-claim batches; for single claims the CPU path is already 40 µs, so we route by batch size, as measured. In a live replay we caught 65 of 65 injected rule errors. We miss amount typos, because this dataset has no fixed fees; Medicare's MBS does." |
| 4:20–4:45 | Responsible AI slide | "A human decides every hold. Reasons on every decision. Fairness checked: gender is even, under-65 patients are flagged 1.25×, and we say so, with mitigations." |
| 4:45–5:00 | Impact plus team slide | "Every claim checked before payment, and help for the honest majority. Team roles: …" |

## Recovery

- **The stream stopped:** rerun step 3 with a larger `--offset`.
- **A Granite note is still a template:** say "Granite upgrades notes asynchronously; this one is queued", and show one that is already done.
- **The dashboard is empty after a restart:** the API rebuilds provider history from stored claims on start, so wait about 2 s and refresh.
