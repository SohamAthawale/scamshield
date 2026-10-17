# Model card: ClaimSense

All numbers come from the training run on the IBM z17 VM (`docs/evidence/training/metrics_vm.json`) and the replays in `docs/evidence/`.

## What the models are for

ClaimSense scores every Medicare-style claim before payment and picks one of three actions:

| Action | When | Who sees it |
|---|---|---|
| `PAY` | nothing unusual | nobody needs to act |
| `RETURN_WITH_FIX` | the claim breaks a billing rule or is unusual, and the provider's pattern is not high risk | the billing office, with a plain-language fix |
| `HOLD_FOR_REVIEW` | the claim is unusual or breaks a rule, **and** the provider's rolling pattern scores as high risk | an investigator, with an evidence brief. **A human decides.** |

The models produce **risk estimates, not findings of fraud**. Nothing is ever denied automatically.

## Data

- Public Kaggle dataset *Healthcare Provider Fraud Detection Analysis* (rohitrox), downloaded from a GitHub mirror and checked against the Kaggle row and label counts (`docs/ENV_REPORT.md`).
- 558,211 claims (40,474 inpatient, 517,737 outpatient), 138,556 beneficiaries and 5,410 providers.
- 506 providers (9.4%) are labelled "potential fraud". **The labels are per provider, not per claim.**
- The data is synthetic, US Medicare-style, from late 2008 to the end of 2009. It contains no real patients.
- **Split by provider**, never by claim, stratified on the label: 70/15/15 with seed 42, giving 3,787 train, 812 validation and 811 test providers. The demo stream is the test providers' claims in time order.

## Features (`src/claimsense/features.py`)

There are 19 claim features: amount, deductible, durations, code counts, physician fields, age, chronic conditions, coverage months, how rare the diagnosis/procedure pair is, and the amount relative to the typical amount for that diagnosis.

There are 19 provider features, computed as rolling aggregates that are updated one claim at a time: volume, amounts, inpatient share, stays, patients, physicians, claims per patient, rare-pair share, missing-attending share, and so on.

The batch and streaming implementations are tested for parity, including on real data.

**Not used as inputs:** gender, race, state, county, and all identifiers. Identifiers are pseudonymised before features are computed.

## Models

| | Provider risk | Claim anomaly |
|---|---|---|
| Algorithm | XGBoost 3.4.1 (hist; depth 4; eta 0.03; early stopping on validation PR-AUC; `scale_pos_weight`) | PyTorch 2.11 MLP autoencoder 19-14-7-14-19, trained only on claims from non-flagged training providers |
| Training trick | trained on each provider's history at 25/50/75/100%, so it works while a provider's history is still filling up in the live stream | inputs standardised and clipped to ±8σ |
| Serving | XGBoost on CPU (about 85 µs per provider) | `torch_nnpa` on the Telum II NNPA for batches of 256 or more claims (chunks of up to 4,096), CPU below that |
| Explanations | SHAP contributions (`pred_contribs`) plus peer percentiles | per-feature reconstruction error, keeping features with at least 10% of the error |

## Results (held-out test providers)

### Provider model

| Metric | full history | first 25% of history |
|---|---|---|
| ROC-AUC | **0.942** | 0.928 |
| PR-AUC (base rate 0.094) | **0.675** | 0.655 |
| Precision in top 5% of providers | 0.732 | 0.780 |
| Recall in top 5% of providers | 0.395 | 0.421 |

### Claim autoencoder

- The threshold is the 99th percentile of reconstruction error on clean validation claims (2.958), which flags 0.95% of test claims.
- **On its own it does not detect fraud.** ROC-AUC against the provider label is 0.52. That is expected: it finds *unusual claims*, and fraud is labelled per provider. Its job is spotting errors, and it surfaced the zero-coverage eligibility problem by itself (see rules).

### Decisions when the test stream is replayed through the real engine

| | Validation | Test |
|---|---|---|
| Pay / return / hold | 98.70% / 0.72% / 0.58% | 98.78% / 0.73% / 0.50% |
| Held claims that come from flagged providers | 70.3% | **61.4%** (base rate 36.2%) |

The hold threshold was tuned on **validation**: the lowest provider risk at which at least 70% of held claims come from flagged providers, with at most 2% of claims held. That gave 0.56. The test figure, 61%, is lower than validation because only a few hundred claims are held.

### Injected honest billing errors (live replay on the VM)

The public data has almost none of these errors, so the simulator injects them; the API is never told which claims they are:

| Error | Caught |
|---|---|
| service after date of death | 13/13 |
| end date before start date | 13/13 |
| duplicate submission | 13/13 |
| missing attending physician | 13/13 |
| no Part A/B coverage for the service | 13/13 |
| amount with an extra zero (×10) | **1/13** |

## Hard rules (deterministic)

The rules are: after death, end before start, discharge before admission, missing attending physician, duplicate, and no coverage.

`no_coverage` was added after the autoencoder kept flagging claims for patients with 0 months of Part A or B coverage. Every such claim was already being returned. The rule gives them a specific reason and fix (0.37% of claims).

## Fairness (`notebooks/03_fairness.ipynb`, test stream)

| Attribute | Flag-rate spread (max/min) | Notes |
|---|---|---|
| Gender (codes 1, 2) | 1.04 | even |
| Race (codes 1, 2, 3, 5) | 1.44 | code 2 is flagged 1.22× the average; claims from honest providers in that group are flagged 1.71% vs 1.25% |
| Age band | 1.81 | under-65s are flagged 1.25×, over-85s 0.69× |
| State (47 groups with ≥200 claims) | 4.6 | mostly small-sample noise |

**Interpretation.**
- In Medicare, under-65 beneficiaries are mostly people with disabilities or ESRD. They are a minority, so an unsupervised model treats them as "unusual". This is a known risk of anomaly detection.
- The race gap comes through correlated features, because race is not an input.

**Mitigations in place:**
- A returned claim only asks the billing office to check, and the fix is spelled out.
- Holds always go to a human, with the reasons shown.
- The investigator brief always says "not a finding of fraud".

**Next steps:**
- per-group anomaly thresholds, or removing age from the claim model;
- re-measuring with the notebook;
- adding fairness to the release checklist.

## Limitations

- **Synthetic, provider-level labels.** We cannot measure claim-level fraud precision.
- **Amount typos are mostly missed.** In this dataset, amounts for the same diagnosis vary so much that about 10% of honest claims already exceed 5× the typical amount. Any rule that catches typos would also return 5 to 30% of honest claims; we tested both global and per-provider baselines and rejected them. Real Australian MBS items have scheduled fees, where an extra zero becomes a deterministic check.
- **Provider risk needs history.** A provider is not risk-scored until it has 10 claims, so a brand-new provider's odd claim is returned, never held.
- **Granite (2B, Q8_0) writes a note in about 1–2 minutes on the 3-vCPU VM.** A template note, built from the same facts, is shown instantly.
- **The models are trained on 2009 US-style data.** Production use would need local data, recalibration and monitoring.

## Explanations by Granite

- `granite-3.3-2b-instruct` (big-endian GGUF, Q8_0) runs on the VM with llama.cpp built with the zDNN backend.
- **Input:** only structured, pseudonymised facts (rule titles and fixes, feature labels and values, peer percentiles, provider token and risk). No identifiers or personal details reach the prompt.
- **Output guards:**
  - rejected if it contains anything shaped like a raw ID;
  - trimmed to the last full sentence if it hits the token limit;
  - briefs always end with "not a finding of fraud";
  - on any failure the template note stays.
