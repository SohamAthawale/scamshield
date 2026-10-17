"""Single source of truth for payment features: ONE streaming implementation, used for training and serving.

Each payment is described by
  * the payment itself (amount in USD, format, cross-bank, currency mismatch, hour),
  * the SENDER's history before it  (is this a new payee? how far from the customer's usual amount?),
  * the RECEIVER's history before it (mule signals: many unrelated senders, money passing straight through,
    a sudden burst of inflows).

Training runs this same engine over the historical stream, so train/serve parity holds by construction.
State is O(1) per account (numbers, not sets or windows): 24-hour activity is an exponentially decayed
counter, and "new payee" uses one global set of (sender, receiver) pairs. ~515k accounts fit in a few
hundred MB, which matters on an 8 GB VM.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .data import FORMATS

INTRINSIC = ["log_usd", *[f"fmt_{f.replace(' ', '')}" for f in FORMATS], "cross_bank", "self_transfer",
             "currency_mismatch", "hour_sin", "hour_cos"]
SENDER = ["s_log_n_out", "s_log_n_in", "s_amount_vs_usual", "s_new_payee", "s_log_payees", "s_pass_through",
          "s_log_idle_h", "s_recent_out", "s_is_new"]
RECEIVER = ["r_log_n_in", "r_log_n_out", "r_log_senders", "r_fan_in", "r_pass_through", "r_share_ach_in",
            "r_log_idle_h", "r_recent_in", "r_is_new"]
FEATURES: list[str] = INTRINSIC + SENDER + RECEIVER
# the customer-side anomaly model looks at the payment and the payer's own history only
ANOMALY_FEATURES: list[str] = INTRINSIC + SENDER

FEATURE_LABELS: dict[str, str] = {
    "log_usd": "payment amount", "fmt_ACH": "ACH transfer", "fmt_Bitcoin": "Bitcoin payment",
    "fmt_Cash": "cash payment", "fmt_Cheque": "cheque", "fmt_CreditCard": "credit card payment",
    "fmt_Reinvestment": "reinvestment", "fmt_Wire": "wire transfer", "cross_bank": "payment to another bank",
    "self_transfer": "transfer to own account", "currency_mismatch": "currency conversion",
    "hour_sin": "time of day", "hour_cos": "time of day",
    "s_log_n_out": "how many payments this customer has made", "s_log_n_in": "how many payments this customer has received",
    "s_amount_vs_usual": "amount compared with this customer's usual payments", "s_new_payee": "first payment to this payee",
    "s_log_payees": "number of different payees", "s_pass_through": "customer sends on what they receive",
    "s_log_idle_h": "hours since this customer's previous payment", "s_recent_out": "payments by this customer in the last day",
    "s_is_new": "customer has no history yet",
    "r_log_n_in": "payments the receiving account has received", "r_log_n_out": "payments the receiving account has made",
    "r_log_senders": "different accounts paying the receiving account", "r_fan_in": "share of the receiver's inflows from different senders",
    "r_pass_through": "receiving account sends on what it receives", "r_share_ach_in": "share of the receiver's inflows by ACH",
    "r_log_idle_h": "hours since the receiving account's previous payment", "r_recent_in": "payments into the receiving account in the last day",
    "r_is_new": "receiving account has no history yet",
}

# Deterministic scam heuristics. Each maps to (what we tell the customer, what they should do).
RULES: dict[str, tuple[str, str]] = {
    "big_new_payee": (
        "This is your first payment to this payee and it is much larger than your usual payments",
        "Check the payee's details through a contact you already trust before you send money.",
    ),
}
# (A "receiver suddenly collecting from many senders" rule was tested and dropped: 2.7x lift only, and the
# risk model already uses those receiver signals. A "money passes straight through" rule had no lift at all.)

IDLE_CAP_H = 24.0 * 30
TAU_MIN = 24.0 * 60.0  # decay constant for "last day" activity counters, in minutes


@dataclass
class FeatureContext:
    """Learned from TRAINING data only: exchange rates to USD (derived from the data's own conversions)."""

    usd_per_unit: dict[str, float] = field(default_factory=dict)
    rule_big_new_payee_ratio: float = 10.0

    @classmethod
    def fit(cls, tx: pd.DataFrame) -> "FeatureContext":
        conv = tx[tx["currency_paid"] != tx["currency_received"]]
        conv = conv[(conv["amount_paid"] > 0) & (conv["amount_received"] > 0)]
        ratio = (conv["amount_received"] / conv["amount_paid"]).groupby(
            [conv["currency_paid"], conv["currency_received"]]).median()
        rates = {"US Dollar": 1.0}
        changed = True
        while changed:  # walk the conversion graph outwards from USD
            changed = False
            for (paid, recv), r in ratio.items():
                if recv in rates and paid not in rates:      # 1 paid = r recv
                    rates[paid] = r * rates[recv]
                    changed = True
                elif paid in rates and recv not in rates:
                    rates[recv] = rates[paid] / r
                    changed = True
        missing = set(tx["currency_paid"].unique()) - set(rates)
        if missing:
            raise ValueError(f"no conversion path to USD for {missing}")
        return cls(usd_per_unit={k: float(v) for k, v in rates.items()})

    def to_usd(self, amount: float, currency: str) -> float:
        return float(amount) * self.usd_per_unit.get(currency, 1.0)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "FeatureContext":
        return cls(**json.loads(s))




class FeatureEngine:
    """Rolling per-account state. observe() returns the features of one payment, THEN updates the state."""

    def __init__(self, ctx: FeatureContext):
        self.ctx = ctx
        self.ids: dict[str, int] = {}
        self.pairs: set[int] = set()
        # one list per field, indexed by account id (lists of floats are the fastest structure in CPython)
        self.n_out: list[float] = []
        self.n_in: list[float] = []
        self.sum_log_out: list[float] = []
        self.amt_out: list[float] = []
        self.amt_in: list[float] = []
        self.n_in_ach: list[float] = []
        self.payees: list[float] = []
        self.senders: list[float] = []
        self.last_t: list[float] = []
        self.dec_out: list[float] = []
        self.dec_out_t: list[float] = []
        self.dec_in: list[float] = []
        self.dec_in_t: list[float] = []
        self._fields = [self.n_out, self.n_in, self.sum_log_out, self.amt_out, self.amt_in, self.n_in_ach,
                        self.payees, self.senders, self.last_t, self.dec_out, self.dec_out_t, self.dec_in,
                        self.dec_in_t]

    def _id(self, key: str) -> int:
        i = self.ids.get(key)
        if i is None:
            i = self.ids[key] = len(self.ids)
            for f in self._fields:
                f.append(0.0)
            self.last_t[i] = -1.0
        return i

    @staticmethod
    def _decayed(value: float, since: float, now: float) -> float:
        # a late (backdated) payment never "un-decays" the counter: the gap is clamped at 0
        return value * math.exp(-max(now - since, 0.0) / TAU_MIN) if value else 0.0

    def observe(self, t: float, sender: str, receiver: str, usd: float, fmt: str, cross_bank: bool,
                currency_mismatch: bool, hour: int) -> list[float]:
        s, r = self._id(sender), self._id(receiver)
        log_usd = math.log1p(max(usd, 0.0))
        pair = hash((s, r))
        new_payee = pair not in self.pairs

        # ---------------- features from the state BEFORE this payment
        sn_out, sn_in = self.n_out[s], self.n_in[s]
        usual = self.sum_log_out[s] / sn_out if sn_out else log_usd
        s_idle = IDLE_CAP_H if self.last_t[s] < 0 else min(max(t - self.last_t[s], 0.0) / 60.0, IDLE_CAP_H)
        rn_in, rn_out = self.n_in[r], self.n_out[r]
        r_idle = IDLE_CAP_H if self.last_t[r] < 0 else min(max(t - self.last_t[r], 0.0) / 60.0, IDLE_CAP_H)
        ang = 2.0 * math.pi * hour / 24.0
        f = [log_usd, *[1.0 if fmt == x else 0.0 for x in FORMATS], float(cross_bank), float(sender == receiver),
             float(currency_mismatch), math.sin(ang), math.cos(ang),
             # sender
             math.log1p(sn_out), math.log1p(sn_in), log_usd - usual, float(new_payee), math.log1p(self.payees[s]),
             math.log1p(self.amt_out[s]) - math.log1p(self.amt_in[s]), math.log1p(s_idle),
             self._decayed(self.dec_out[s], self.dec_out_t[s], t), float(sn_in + sn_out == 0),
             # receiver
             math.log1p(rn_in), math.log1p(rn_out), math.log1p(self.senders[r]),
             self.senders[r] / rn_in if rn_in else 0.0,
             math.log1p(self.amt_out[r]) - math.log1p(self.amt_in[r]),
             self.n_in_ach[r] / rn_in if rn_in else 0.0, math.log1p(r_idle),
             self._decayed(self.dec_in[r], self.dec_in_t[r], t), float(rn_in + rn_out == 0)]

        # ---------------- update the state with this payment
        self.n_out[s] += 1
        self.sum_log_out[s] += log_usd
        self.amt_out[s] += usd
        self.dec_out[s] = self._decayed(self.dec_out[s], self.dec_out_t[s], t) + 1.0
        self.dec_out_t[s] = max(self.dec_out_t[s], t)
        self.n_in[r] += 1
        self.amt_in[r] += usd
        self.n_in_ach[r] += fmt == "ACH"
        self.dec_in[r] = self._decayed(self.dec_in[r], self.dec_in_t[r], t) + 1.0
        self.dec_in_t[r] = max(self.dec_in_t[r], t)
        if new_payee:
            self.pairs.add(pair)
            self.payees[s] += 1
            self.senders[r] += 1
        self.last_t[s] = max(self.last_t[s], t)
        self.last_t[r] = max(self.last_t[r], t)
        return f

    def observe_txn(self, txn: Mapping[str, Any]) -> list[float]:
        """Convenience wrapper for one raw transaction (API payload or a DataFrame row as a dict)."""
        ts = pd.Timestamp(txn["ts"])
        return self.observe(
            ts.value / 6e10, txn["sender"], txn["receiver"],
            self.ctx.to_usd(txn["amount_paid"], txn["currency_paid"]), txn["payment_format"],
            txn["sender"].split(":", 1)[0] != txn["receiver"].split(":", 1)[0],
            txn["currency_paid"] != txn["currency_received"], ts.hour)

    def run(self, tx: pd.DataFrame, progress_every: int = 0) -> np.ndarray:
        """Feed a time-sorted DataFrame through the engine; returns features (n, len(FEATURES)) as float32."""
        n = len(tx)
        out = np.empty((n, len(FEATURES)), dtype=np.float32)
        t = (tx["ts"].values.astype("datetime64[m]").astype(np.int64)).astype(float)
        usd = tx["amount_paid"].values * tx["currency_paid"].map(self.ctx.usd_per_unit).fillna(1.0).values
        cross = (tx["from_bank"].values != tx["to_bank"].values)
        mism = (tx["currency_paid"].values != tx["currency_received"].values)
        hour = tx["ts"].dt.hour.values
        snd, rcv, fmt = tx["sender"].values, tx["receiver"].values, tx["payment_format"].values
        obs = self.observe
        for i in range(n):
            out[i] = obs(t[i], snd[i], rcv[i], float(usd[i]), fmt[i], bool(cross[i]), bool(mism[i]), int(hour[i]))
            if progress_every and i and i % progress_every == 0:
                print(f"  features {i:,}/{n:,}", flush=True)
        return out


def rule_hits(f: Mapping[str, float] | np.ndarray, ctx: FeatureContext) -> list[str]:
    """Scam heuristics fired by one payment's features (dict keyed by FEATURES, or the raw vector)."""
    if not isinstance(f, Mapping):
        f = dict(zip(FEATURES, f))
    out = []
    if f["s_new_payee"] >= 0.5 and f["s_log_n_out"] >= math.log1p(5) and \
            f["s_amount_vs_usual"] >= math.log(ctx.rule_big_new_payee_ratio):
        out.append("big_new_payee")
    return out
