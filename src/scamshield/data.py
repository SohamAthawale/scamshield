"""Loading IBM's synthetic anti-money-laundering transactions (AMLworld, "HI-Small") and splitting by time.

Source: IBM "Transactions for Anti Money Laundering" (Altman et al., NeurIPS 2023 Datasets track), generated
with IBM AMLSim, licence CDLA-Sharing-1.0. Files are found by glob, never hardcoded.

Known artifact: after 2022-09-10 the generator stops producing normal traffic while laundering chains
continue, so up to 70% of the late rows are laundering. Those days are cut (`end`) so no model can learn
"late = laundering".
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

COLUMNS = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amount_received", "currency_received",
           "amount_paid", "currency_paid", "payment_format", "label"]
FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
DEFAULT_END = "2022-09-11"                      # exclusive: keep 2022-09-01 .. 2022-09-10
DEFAULT_SPLITS = ("2022-09-07", "2022-09-09")   # train < 7th <= val < 9th <= test < end


def _find(raw_dir: str, pattern: str) -> str:
    hits = sorted(glob.glob(os.path.join(raw_dir, pattern)))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern!r} in {raw_dir}")
    return hits[0]


def load_transactions(raw_dir: str, end: str | None = DEFAULT_END, nrows: int | None = None) -> pd.DataFrame:
    """All transactions in time order (stable within a minute, as in the file), with account keys."""
    d = pd.read_csv(_find(raw_dir, "*_Trans.csv"), header=0, names=COLUMNS, nrows=nrows,
                    dtype={"from_bank": str, "from_acct": str, "to_bank": str, "to_acct": str,
                           "currency_received": str, "currency_paid": str, "payment_format": str})
    d["ts"] = pd.to_datetime(d["ts"], format="%Y/%m/%d %H:%M")
    if end is not None:
        d = d[d["ts"] < pd.Timestamp(end)]
    d = d.sort_values("ts", kind="mergesort").reset_index(drop=True)
    # an account is identified by (bank, account number)
    d["sender"] = d["from_bank"] + ":" + d["from_acct"]
    d["receiver"] = d["to_bank"] + ":" + d["to_acct"]
    d["label"] = d["label"].astype(np.int8)
    return d


def time_split(ts: pd.Series, splits: tuple[str, str] = DEFAULT_SPLITS) -> np.ndarray:
    """'train' / 'val' / 'test' by timestamp: a bank trains on the past and is judged on the future."""
    a, b = pd.Timestamp(splits[0]), pd.Timestamp(splits[1])
    return np.where(ts < a, "train", np.where(ts < b, "val", "test"))


def laundering_patterns(raw_dir: str) -> pd.DataFrame:
    """Which transactions belong to which typed laundering attempt (FAN-IN, CYCLE, ...)."""
    rows, kind, attempt = [], None, -1
    with open(_find(raw_dir, "*_Patterns.txt")) as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                kind = line.split("-", 1)[1].split(":")[0].strip()
                attempt += 1
            elif line.startswith("END") or not line:
                continue
            elif kind:
                p = line.split(",")
                rows.append((attempt, kind, p[0], f"{p[1]}:{p[2]}", f"{p[3]}:{p[4]}", float(p[7])))
    return pd.DataFrame(rows, columns=["attempt", "pattern", "ts", "sender", "receiver", "amount_paid"])
