# Demo script

Written for **5 minutes**. Confirm the official limit and cut the parts marked *(optional)* if it's shorter.

## Before recording (5 min)

1. `scripts/tunnel.sh` on the laptop, so <http://localhost:8000/ui> works.
2. On the VM: `cd ~/scamshield && docker compose up -d` (the API builds its state from 4.2M historical payments in about 1 minute).
3. Start the live stream in a second VM terminal (it runs about 10 minutes):
   ```bash
   docker compose --profile demo run --rm simulator python -m scamshield.simulator --url http://api:8000 --offset 40000 --limit 12000 --rate 20 --inject 0.03 --concurrency 8
   ```
4. Open three browser windows, logged in as **analyst**, **customer** and **auditor** (`secrets/demo_passwords.txt`).
5. In a terminal on the VM, have ready: `lscpu | grep -io nnpa`, `grep Type /proc/sysinfo`, `docker compose ps`.

## Script

| Time | Show | Say |
|---|---|---|
| 0:00–0:35 | Title slide, then the problem slide | "Australians lost $2.03 billion to scams last year. The money moves through mule accounts in minutes, and once it's gone it rarely comes back. The new Scams Prevention Framework says banks must detect and disrupt. ScamShield checks every payment *before* it leaves, in milliseconds. Theme 1." |
| 0:35–1:00 | Persona slide | "Two people: a customer about to be talked into sending money, and the bank's scam analyst with a limited caseload. Most payments go straight through, a risky one gets a clear warning, and a payment matching a mule pattern is paused for a call-back." |
| 1:00–1:30 | VM terminal: `uname -m`, machine type 9175, `nnpa`, `docker compose ps` | "This is live on an IBM z17 in the LinuxONE Community Cloud. The containers are built from the official L1CC image. At start-up it learned the history of 514,000 accounts from 4.2 million payments." |
| 1:30–2:30 | Analyst dashboard | "Payments stream in at about 4 ms each, 100% checked, with the Telum II badge." Open a paused payment: "Paused because it's a first ACH payment to this payee, the main risk drivers are listed, and opening it decrypted the details and wrote an audit entry. IBM Granite, running here on LinuxONE, wrote this brief from pseudonymised facts only, and it says plainly it is not a finding of fraud." Click **Customer confirmed: release** or **Scam confirmed: stop payment**. Point at the mule watchlist. |
| 2:30–3:05 | Customer view | "The customer sees only their own payments and a calm, specific message: 'This is your first payment to this payee and it's much larger than usual; check the payee through a contact you already trust.' No scores, no accusations, and they decide." |
| 3:05–3:40 | Auditor view, then the tamper command (SECURITY.md), then Verify again | "Every decision, login and case view is hash-chained. Verify: intact. Now I edit one row directly in the database. Verify: broken at that exact entry. Account numbers are pseudonymised on arrival and payloads are AES-256-GCM encrypted on CPACF." |
| 3:40–4:20 | Results slide, then the NNPA chart | "On two unseen days: we paused fewer than one payment in a thousand, and three in four of those were laundering. Because we only pause the borderline ones when they're big enough to matter, that's $269 million of $424 million in laundered money stopped before it moved, and 147 of 174 laundering schemes broken. The Telum II runs our customer model 1.4 times faster on large batches, so a burst of payments shifts onto the accelerator automatically." |
| 4:20–4:45 | Responsible AI slide | "Warnings never block, and holds always reach a human. Fairness is audited by currency, format and bank size; honest bank transfers carry more friction and we say so. One more limitation, said honestly: in this data a victim's payment looks like an honest large first payment, so both get the same protective check." |
| 4:45–5:00 | Impact plus team slide | "Every payment checked before it leaves, and the mule chain broken early. Team roles: …" |

## Recovery

- **The stream stopped:** rerun step 3 with a larger `--offset`.
- **A Granite note is still a template:** "Granite rewrites messages asynchronously; this one is queued", then show one already done.
- **The dashboard is empty after a restart:** the API rebuilds state (about 1 minute) and replays stored payments, so wait and refresh.
