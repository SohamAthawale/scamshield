# Value-aware HOLD: old vs new (z17 VM, deployed models)

Chosen on validation days only (targets: hold precision >= 80%, keep >= 95% of the laundered USD the old policy pauses): hold if risk >= 0.8520, or if risk >= 0.7586 and amount >= $5,000.

## Test days (9-10 Sep, scored once)

| | Old (risk >= hold) | New (value-aware) |
|---|---|---|
| Hold rate | 0.108% | 0.074% |
| Holds | 936 | 641 |
| Hold precision | 53.3% | 73.0% |
| Hold recall (laundering payments) | 52.2% | 49.0% |
| Laundering payments held | 499 | 468 |
| Laundered USD paused | $269.3M | $269.2M |
| Laundering warned or paused | 82.0% | 82.0% |
| Warn rate | 1.63% | 1.66% |
| Schemes broken (>=1 payment held) | 150/174 | 147/174 |
| Laundered USD in period | $424.3M | $424.3M |

## Validation days (7-8 Sep, used for the choice)

| | Old (risk >= hold) | New (value-aware) |
|---|---|---|
| Hold rate | 0.101% | 0.072% |
| Holds | 980 | 696 |
| Hold precision | 60.0% | 81.5% |
| Hold recall (laundering payments) | 56.8% | 54.7% |
| Laundering payments held | 588 | 567 |
| Laundered USD paused | $337.0M | $337.0M |
| Laundering warned or paused | 81.4% | 81.4% |
| Warn rate | 1.24% | 1.27% |
| Schemes broken (>=1 payment held) | 145/157 | 142/157 |
| Laundered USD in period | $459.7M | $459.7M |

Old-policy numbers reproduce the deployed metrics.json: True
