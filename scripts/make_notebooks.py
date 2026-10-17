"""Writes the three analysis notebooks (they are executed on LinuxONE, see README)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks"


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python",
                                                      "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = '''
import sys, os, platform, subprocess
sys.path.insert(0, os.path.expanduser("~/scamshield/src"))
import xgboost  # before torch (see training.py)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
SHARED = os.path.expanduser("~/shared")
RAW = f"{SHARED}/aml"
print(platform.machine(), platform.platform())
print(subprocess.run("grep -E '^Type:' /proc/sysinfo; lscpu | grep -io nnpa | head -1", shell=True,
                     capture_output=True, text=True).stdout)
'''

EDA = nb([
    ("markdown", "# 01 · The payment data (on IBM LinuxONE)\n"
                 "IBM's synthetic anti-money-laundering transactions (AMLworld HI-Small, generated with IBM AMLSim, "
                 "CDLA-Sharing-1.0). Labels are **per payment**: part of a laundering/mule scheme or not."),
    ("code", SETUP),
    ("code", '''
from scamshield import data as D
full = D.load_transactions(RAW, end=None)
day = full.groupby(full.ts.dt.date).label.agg(["size", "sum"]).rename(columns={"size": "payments", "sum": "laundering"})
day["laundering_share"] = day.laundering / day.payments
day
'''),
    ("markdown", "**Artifact:** after 10 September the generator stops producing normal traffic while laundering chains "
                 "continue (up to 70% of late rows are laundering). We cut those days so no model can learn "
                 "'late = laundering'."),
    ("code", '''
tx = D.load_transactions(RAW)          # cut at 2022-09-11
split = D.time_split(tx.ts)
print(f"{len(tx):,} payments, {tx.label.sum():,} laundering ({tx.label.mean():.3%}), "
      f"{pd.concat([tx.sender, tx.receiver]).nunique():,} accounts")
pd.crosstab(split, tx.label, margins=True)
'''),
    ("code", '''
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
fmt = tx.groupby("payment_format").label.mean().sort_values()
(fmt * 100).plot.barh(ax=ax[0], title="% laundering by payment format"); ax[0].set_xlabel("%")
pat = D.laundering_patterns(RAW).groupby("pattern").attempt.nunique().sort_values()
pat.plot.barh(ax=ax[1], title="laundering schemes by network pattern"); plt.tight_layout()
'''),
    ("markdown", "Most laundering moves by ACH. The 8 network shapes (fan-in, fan-out, cycles, scatter-gather...) are "
                 "what mule accounts do; ScamShield's receiver-side features look for them."),
])

TRAIN = nb([
    ("markdown", "# 02 · Training both models (on IBM LinuxONE)\n"
                 "Same functions as `python -m scamshield.training`, step by step. Nothing here touches the running service."),
    ("code", SETUP),
    ("code", '''
import json, time
from scamshield import training as T
t = time.time()
tx, split, ctx, X = T.prepare(RAW)     # ONE streaming feature engine, the same code the API runs
y = tx.label.values.astype(int)
print(f"{len(tx):,} payments -> {X.shape[1]} features in {time.time()-t:.0f}s on {os.cpu_count()} vCPUs")
{k: round(v, 4) for k, v in sorted(ctx.usd_per_unit.items())}  # exchange rates derived from the data itself
'''),
    ("markdown", "## Risk model: XGBoost on the payment + sender history + receiver (mule) history"),
    ("code", '''
booster, meta = T.train_risk_model(X, y, split)
risk = booster.inplace_predict(X)
pd.DataFrame({s: T.ranking_metrics(y[split == s], risk[split == s]) for s in ("val", "test")}).round(4)
'''),
    ("code", '''
imp = pd.Series(meta["importance_gain"]).sort_values()
imp.tail(12).plot.barh(figsize=(7, 4), title="Risk model: importance (gain)"); plt.tight_layout()
'''),
    ("markdown", "## Customer-side model: PyTorch autoencoder ('is this payment unusual for this customer?')"),
    ("code", '''
ae, scaler, history = T.train_autoencoder(X, y, split, epochs=4)
plt.figure(figsize=(5, 3)); plt.plot(range(1, len(history) + 1), history, "o-")
plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("autoencoder training"); plt.tight_layout()
'''),
    ("markdown", "## Run it on the Telum II on-chip AI accelerator\n"
                 "`torch_nnpa` registers an `nnpa` device: moving the model there is one line."),
    ("code", '''
import torch, torch_nnpa, copy
print("NNPA available:", torch_nnpa.NNPAModule.is_available(), "| Telum II:", torch_nnpa.NNPAModule.is_telum2_available())
dev = torch_nnpa.device()
ae_nnpa = copy.deepcopy(ae).to(dev).eval()
Z = torch.from_numpy(scaler.transform(X[split == "test"][:4096][:, T.AE_IDX]))
with torch.no_grad():
    print("max |cpu - nnpa| =", float((ae(Z) - ae_nnpa(Z.to(dev)).cpu()).abs().max()))
for name, fn in [("cpu", lambda: ae(Z)), ("nnpa", lambda: ae_nnpa(Z.to(dev)).cpu())]:
    with torch.no_grad():
        for _ in range(20):
            fn()
        t = time.perf_counter()
        for _ in range(200):
            fn()
        dt = (time.perf_counter() - t) / 200
    print(f"{name}: {dt*1e6:.0f} us per 4096-payment batch -> {4096/dt:,.0f} payments/s")
'''),
    ("markdown", "## Decisions on the unseen test days (from the production training run)"),
    ("code", '''
m = json.load(open(f"{SHARED}/models/metrics.json"))
print(json.dumps(m["thresholds"], indent=1))
d = m["decisions"]["test"]
pd.Series({"allow": d["mix"]["ALLOW"], "warn": d["mix"]["WARN"], "hold": d["mix"]["HOLD"],
           "hold precision": d["hold_precision"], "laundering held": d["laundering_recall_hold"],
           "laundering warned or held": d["laundering_recall_warn_or_hold"],
           "USD held (M)": d["laundered_usd_held"] / 1e6, "USD laundered (M)": d["laundered_usd_total"] / 1e6,
           "schemes disrupted": d["schemes_disrupted_total"]})
'''),
    ("code", '''
pd.Series(d["schemes_disrupted"]).rename("schemes with a payment held")
'''),
])

FAIR = nb([
    ("markdown", "# 03 · Fairness check (on IBM LinuxONE)\n"
                 "The AML data has no demographics, so we audit proxies a bank could be unfair along: **payment "
                 "currency** (migrant communities often send in Rupee, Yuan, Shekel...), **payment format**, and "
                 "**the sender's bank size**. Measure: the false-alarm rate, honest payments that were still warned or held."),
    ("code", SETUP),
    ("code", '''
from scamshield import fairness as F
dec = pd.read_csv(f"{SHARED}/data/test_decisions.csv")
g = F.attach_groups(dec)
F.disparity_summary(g).round(4)
'''),
    ("code", '''
for a in ["currency_paid", "payment_format", "bank_size"]:
    display(F.group_rates(g, a).round(4))
'''),
    ("code", '''
r = F.group_rates(g, "currency_paid")
(r.false_alarm_rate * 100).sort_values().plot.barh(figsize=(6, 4), title="% of honest payments warned or held, by currency")
plt.xlabel("%"); plt.tight_layout()
'''),
    ("markdown", "**Findings.** Across fiat currencies false-alarm rates are similar (about 1.6-2.3%), so no "
                 "migrant-community currency is singled out. Honest **ACH** payments carry most of the friction "
                 "(about 11% warned) because most laundering moves by ACH; laundering by cash, cheque or card is mostly "
                 "missed. Customers of the **smallest banks** see about 5x the false alarms of the biggest banks. "
                 "**Mitigations:** warnings never block (the customer decides), holds always go to a human, and "
                 "per-segment warning budgets are the next step, re-measured with this notebook."),
])

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, n in [("01_payment_data.ipynb", EDA), ("02_train_models.ipynb", TRAIN), ("03_fairness.ipynb", FAIR)]:
        (OUT / name).write_text(json.dumps(n, indent=1))
    print("written to", OUT)
