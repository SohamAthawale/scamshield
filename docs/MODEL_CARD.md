# Model card: ScamShield

All numbers come from the training run on the IBM z17 VM (`docs/evidence/training/metrics_vm.json`), the hold-policy evaluation on the same VM with the same models (`docs/evidence/value_hold/`), the claims checks (`docs/evidence/claims/`), the live replays (`docs/evidence/replay/`) and the notebooks executed on the VM (`notebooks/`).

## What it is for

Every payment is scored before it is sent:

| Action | When | Who sees it |
|---|---|---|
| `ALLOW` | nothing unusual | nobody needs to act |
| `WARN` | risk ≥ warn threshold, **or** the payment is unusual for this customer, **or** it is a big first payment to a new payee | the customer, with a specific warning. **They decide.** |
| `HOLD` | risk ≥ strict hold threshold, **or** risk ≥ hold threshold and the payment is at least $5,000 | the payment waits; an analyst calls the customer back on a known number. **A human decides.** |

The scores are **model estimates, not findings of fraud**. Nothing is reported, frozen or refused automatically.

## Data

- **IBM "Transactions for Anti Money Laundering", HI-Small** (AMLworld; Altman et al., NeurIPS 2023 Datasets track; generated with IBM AMLSim; licence CDLA-Sharing-1.0).
  - Downloaded from Hugging Face mirrors; two independent mirrors are byte-identical (SHA-256 in `docs/ENV_REPORT.md`).
  - It is **synthetic**: no real person's data.
- **Size:** 5,078,345 payments, 515k accounts, 15 currencies, 7 payment formats. 0.10% are labelled laundering, belonging to 740 typed schemes (fan-in, fan-out, cycle, bipartite, stack, scatter-gather, gather-scatter, random).
- **Artifact removed:** after 10 September the generator stops producing normal traffic while laundering continues (up to 70% of the late rows are laundering). Those days are cut, leaving **5,077,237 payments, 4,522 laundering**.
- **Split by time,** as a bank deploys:

  | Split | Days | Payments | Laundering |
  |---|---|---|---|
  | Train | 1–6 Sep | 3,248,921 | 2,530 |
  | Validation | 7–8 Sep | 965,524 | 1,036 |
  | Test | 9–10 Sep | 862,792 | 956 |

## Features (`src/scamshield/features.py`)

**One streaming engine** computes every feature from the state *before* the payment, then updates it. Training runs the same engine over history, so train/serve parity holds by construction. We verified it: 50,000 test payments scored live gave 100% identical decisions, with risk scores equal to within 1e-16.

| Group | Features |
|---|---|
| Payment (15) | log USD amount (rates derived from the data's own conversions), format one-hots, cross-bank, self-transfer, currency conversion, hour of day |
| Payer history (9) | payments made and received, amount vs the payer's usual, **first payment to this payee**, number of payees, pass-through ratio, hours since previous payment, payments in the last day (decayed), new customer |
| Receiving account / mule history (9) | inflows, outflows, distinct senders, fan-in share, pass-through ratio, share of inflows by ACH, hours since previous payment, inflows in the last day (decayed), new account |

State is O(1) per account. The engine runs at about 4.4 µs per payment; 4.2M historical payments are replayed into it in 61 s on the z17.

## Models

| | Risk | Customer behaviour |
|---|---|---|
| Algorithm | XGBoost 3.4.1 (hist, depth 7, eta 0.05), early stopping on validation PR-AUC. **Saved trimmed to the best iteration.** | PyTorch 2.11 autoencoder (24-16-8-16-24) over the payment and payer-history features |
| Trained on | all training days; negatives subsampled to 25% and re-weighted | normal training payments only |
| Serving | CPU, about 85 µs per payment | `torch_nnpa` on the Telum II NNPA for calls of ≥1,024 payments (chunks of up to 4,096), CPU below that |
| Explanations | SHAP contributions (`pred_contribs`) | per-feature reconstruction error (features carrying ≥10% of the error) |

The "trimmed" note matters. The full booster also held the 100 early-stopping patience rounds, so the API was at first scoring with a slightly different model from the one the thresholds were tuned on (risk scores off by up to 0.27). We found this with the parity check and fixed it.

**Rule** (`big_new_payee`): a first payment to this payee, by a customer with at least 5 prior payments, at ≥10× their usual amount.
- On test it fires on 0.12% of payments with **9.2% precision (≈80× the base rate)**.
- Two other rules were tested and dropped:
  - "receiver suddenly collecting from many senders": only 2.7× lift;
  - "money passes straight through": no lift.

## Results on the unseen test days

### Risk model

| | Validation | Test |
|---|---|---|
| ROC-AUC | 0.986 | **0.984** |
| PR-AUC (base rate 0.11%) | 0.615 | **0.564** |
| Best F1 (precision / recall) | 0.643 (0.88 / 0.51) | **0.587 (0.85 / 0.45)** |

### Decisions

Thresholds were tuned on **validation days only**:
- hold threshold (0.7586) = the lowest risk with ≥60% precision while holding ≤0.2% of payments;
- value-aware hold: between the hold threshold and a strict threshold the model is less sure, so only payments big enough to be worth a call-back are paused there; the rest of that band is warned. Strict threshold (0.8520) and minimum amount ($5,000, from a coarse 1-2-5 grid) were chosen on validation for ≥80% hold precision while keeping ≥95% of the laundered USD the plain threshold pauses (`training.tune_value_hold`);
- warnings come from a budget of 1.25% of payments;
- the anomaly threshold is the 99.9th percentile of normal validation payments.

| | Validation | Test |
|---|---|---|
| Allow / warn / hold | 98.66% / 1.27% / 0.072% | 98.26% / 1.66% / 0.074% |
| Payments paused | 696 | **641** |
| Hold precision | 81.5% | **73.0%** |
| Laundering payments paused | 54.7% | **49.0%** |
| Laundering payments warned or paused | 81.4% | **82.0%** |
| Laundered USD paused | $337M of $460M | **$269M of $424M** |
| Schemes with ≥1 payment paused | 142/157 | **147/174** |

**Value-aware hold vs the plain threshold** (risk ≥ 0.7586 alone; same models, same days; `docs/evidence/value_hold/value_hold_comparison.md`):

| Test days | Plain threshold | Value-aware |
|---|---|---|
| Payments paused | 936 (0.108%) | **641 (0.074%)** |
| Hold precision | 53.3% | **73.0%** |
| Laundering payments paused | 52.2% (499) | 49.0% (468) |
| Laundered USD paused | $269.3M | $269.2M |
| Laundering warned or paused | 82.0% | 82.0% |
| Schemes broken | 150/174 | 147/174 |

The trade: 295 fewer call-backs. Honest customers paused fall from 437 to 173, so 6 in 10 of them get a warning instead. The cost is 31 fewer laundering payments paused and 3 schemes no longer broken (one fan-out, one random, one scatter-gather). Almost all laundered money sits in large payments, so the money paused barely moves. The validation target was 80% precision; on test it is 73%, so the target did not fully carry over. Every payment the value-aware policy stops pausing is warned instead, so "warned or paused" figures, the injected-payment totals and the fairness false-alarm rates below are the same under both policies.

**How concentrated is the money paused?** (`docs/evidence/claims/claims_check.md`) Very: of the 468 laundering payments paused on the test days, the largest one is $79.7M (30% of the $269M), the top 5 are 73% and the top 10 are 95%; the median is $11,266, and all of it is ACH. The dollar figure is true but rests on about ten transfers, so precision and schemes broken are the steadier measures.

Schemes broken on test, by pattern:

| Pattern | Broken |
|---|---|
| bipartite | 12/16 |
| cycle | 19/22 |
| fan-in | 16/18 |
| fan-out | 17/20 |
| gather-scatter | 32/37 |
| random | 13/17 |
| scatter-gather | 18/22 |
| stack | 20/22 |

### Customer-behaviour model

- Flags 0.12% of test payments.
- ROC-AUC 0.71 against the laundering label. These labels mark the *mule* side, while this model watches the *victim* side, so they can't fully measure it. That's why the simulator injects victim payments.

### Injected payments

**Live replay on the VM** (30k payments plus injections, 274 payments/s; run under the plain hold threshold, so some of the "paused" ones would now be warned, while the warned-or-paused totals are unchanged):

| Injected payment | Warned | Paused | Allowed |
|---|---|---|---|
| Victim → mule account | 37 | 18 | 2 |
| Honest, to a brand-new account | 40 | 16 | 1 |

**Offline, 290 of each kind** (`docs/evidence/replay/victim_vs_honest_offline.md`): victims, honest payees with history and brand-new accounts all get the **same** mix, about 32% paused, 64% warned and 4–6% allowed (plain hold threshold; the allowed share is the same under the value-aware hold).

## Fairness

There are no demographics in this data. We audit the false-alarm rate (honest payments warned or held) by proxies, with `src/scamshield/fairness.py` on the VM's test decisions (`docs/evidence/claims/fairness_vm.txt`; the same code as `notebooks/03_fairness.ipynb`). Warned-or-held is the same under the plain and the value-aware hold, so these rates hold for both. The overall rate is 1.65%.

- **Currency (a proxy for migrant communities):**
  - all fiat currencies sit at 1.6–2.3%;
  - Brazil Real is 1.39× the overall rate, Saudi Riyal 1.25×, Australian Dollar 1.18×, Yen 1.14×;
  - Bitcoin is 0.19×.
  - So no currency group is singled out, though the spread deserves monitoring.
- **Payment format:**
  - **honest ACH payments are warned or held 10.4% of the time** (6.3× overall), because most laundering moves by ACH;
  - cheque, card, cash and Bitcoin sit at 0.1–0.4%, wire 0.8%.
- **Bank size** (payments by the sender's bank in the test days):
  - customers of the smallest banks (<100) see 2.6% false alarms;
  - customers of the biggest banks (10k+) see 0.74% (3.4×).

**A measured mitigation** (`scripts/claims_check.py`, `docs/evidence/claims/claims_check.md`): give ACH its own warning threshold, chosen on **validation days** to cap the honest-ACH false-alarm rate, then score the test days once per cap. Holds don't change.

| Honest-ACH cap (validation) | ACH warn threshold | ACH false alarms (test) | All honest false alarms | Smallest banks | Laundering warned or paused | ACH laundering warned or paused |
|---|---|---|---|---|---|---|
| none (deployed) | 0.128 | 10.4% | 1.65% | 2.6% | 82.0% | 93.8% |
| 8% | 0.142 | 9.6% | 1.54% | 2.2% | 81.6% | 93.3% |
| 6% | 0.209 | 6.9% | 1.16% | 1.2% | 79.2% | 90.5% |
| 4% | 0.316 | 4.5% | 0.84% | 0.7% | 76.0% | 86.9% |

The 6% cap removes about a third of honest-ACH friction and halves it for small-bank customers, for 2.8 points of laundering warned or paused. It isn't deployed: how much friction to trade for coverage is the bank's call, and the table puts a number on it.

**Mitigations in place:**
- warnings never block;
- holds always go to a human;
- every decision carries its reasons.

**Next steps:**
- choose a per-format cap with the bank, then bank-size budgets the same way;
- add fairness to the release checklist.

## Limitations

- **The data is synthetic** (AMLSim), and labels mark laundering networks, not individual scam victims.
- **The victim side can't be separated from honest unusual payments in this data,** because receiver history doesn't distinguish the mule accounts. Real deployments would add evidence the dataset lacks: Confirmation of Payee mismatches, mule reports shared between banks (e.g. AFCX), and account age.
- **About 87% of laundering is ACH.** Laundering by cash, cheque or card is mostly missed.
- **The rolling state is built from 8 days of history.** Real use needs months, and recalibration and drift monitoring.
- **The evaluation is short:** thresholds tuned on 2 validation days, tested on 2. Hold precision fell from validation to test under both hold policies (60% → 53% plain, 81.5% → 73% value-aware).
- **The dollar figure rests on about ten transfers** (95% of the $269M paused).
- **The Telum II speeds up the model, not yet the service.** Bulk files run the customer model on the NNPA (40,960 of 40,960 payments in a VM replay), but end to end there is no measurable gain (median 1,562 vs 1,610 payments/s, `docs/evidence/bench/bulk_file_nnpa_vs_cpu.txt`). Single payments stay on the CPU, where small batches are faster.
- **Granite (2B, Q8_0) takes about 1–2 minutes per note on 3 vCPUs.** The template message, built from the same facts, is instant.

## Granite

- **Model:** `granite-3.3-2b-instruct` (big-endian GGUF, Q8_0) via llama.cpp built with zDNN, on the VM.
- **Input:** only structured, pseudonymised facts (rule text, feature labels and values, risk drivers, payment token, risk score).
- **Prompts** forbid adding numbers that aren't in the facts. That instruction was added after an early brief invented a "second payment to this payee".
- **Output guards:**
  - rejected if it contains anything shaped like an account number;
  - trimmed to the last full sentence at the token limit;
  - briefs always carry "not a finding of fraud";
  - on any failure the template message stays.
