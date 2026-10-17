# Injected victim vs honest payments (offline, laptop, same models and engine as the VM)

290 of each kind, injected into the first 40,000 test-day payments. For each injection the same customer makes a first payment of 8-30x their usual amount (70% ACH, 30% wire) to:

| Receiver | HOLD | WARN | ALLOW | median risk | receiver new | median distinct senders of receiver |
|---|---|---|---|---|---|---|
| a mule account (receives laundering in the data) | 0.32 | 0.64 | 0.04 | 0.34 | 0.01 | 3 |
| an established clean account (20-500 prior payments) | 0.32 | 0.64 | 0.04 | 0.42 | 0.00 | 3 |
| a brand-new account | 0.31 | 0.63 | 0.06 | 0.15 | 1.00 | 0 |

Conclusion: decisions are driven by the customer's own out-of-character behaviour; the mule accounts in this
synthetic data look like ordinary receivers, so receiver history cannot separate them here. Every kind gets
the same protective friction (about 1 in 3 paused for a call-back, the rest warned).

A policy that refuses to hold on customer-side signals was tested and rejected: on the labelled test days it
cut hold precision from 52.5% to 49.1% and hold recall from 52.3% to 43.5%.
