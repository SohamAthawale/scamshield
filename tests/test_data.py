"""data.py: AMLworld loader, time split, laundering pattern file."""
import numpy as np
import pandas as pd
import pytest

from scamshield import data as D

from conftest import END, HEADER, SPLITS, write_aml


def test_load_transactions_columns_and_types(aml):
    tx = aml.tx
    assert list(tx.columns) == D.COLUMNS + ["sender", "receiver"]
    assert pd.api.types.is_datetime64_any_dtype(tx["ts"])
    assert tx["label"].dtype == np.int8
    for c in ("from_bank", "to_bank", "from_acct", "to_acct"):
        assert all(isinstance(v, str) for v in tx[c])
    assert tx["amount_paid"].dtype == float and tx["amount_received"].dtype == float
    assert set(tx["payment_format"]) <= set(D.FORMATS)


def test_bank_codes_keep_leading_zeros(aml):
    banks = set(aml.tx["from_bank"]) | set(aml.tx["to_bank"])
    assert {"010", "03208", "001", "0119", "021174"} <= banks
    assert "10" not in banks and "3208" not in banks
    r = aml.tx.iloc[0]
    assert r["sender"] == f"{r['from_bank']}:{r['from_acct']}" and r["receiver"] == f"{r['to_bank']}:{r['to_acct']}"


def test_rows_are_time_sorted_and_cut_at_end(aml):
    ts = aml.tx["ts"]
    assert ts.is_monotonic_increasing
    assert ts.max() < pd.Timestamp(END)
    all_rows = D.load_transactions(str(aml.raw_dir), end=None)
    assert len(all_rows) == len(aml.rows) > len(aml.tx)
    assert all_rows["ts"].max() >= pd.Timestamp(END)
    assert list(aml.tx.index) == list(range(len(aml.tx)))


def test_sort_is_stable_within_a_minute(aml):
    same = aml.tx[(aml.tx["ts"] == pd.Timestamp("2022-09-02 12:34")) & aml.tx["amount_paid"].isin([11.11, 22.22])]
    assert same["amount_paid"].tolist() == [11.11, 22.22]      # file order kept within the minute


def test_timestamp_format_and_nrows(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "HI-Small_Trans.csv").write_text(HEADER + "\n" + "\n".join([
        "2022/09/01 00:20,010,8000EBD30,010,8000EBD30,3697.34,US Dollar,3697.34,US Dollar,Reinvestment,0",
        "2022/09/01 00:15,03208,8000F4580,001,8000F5340,0.01,US Dollar,0.01,US Dollar,Cheque,0",
        "2022/09/01 00:20,03208,8000F4580,001,8000F5340,1.00,Euro,0.91,US Dollar,ACH,1",
    ]) + "\n")
    tx = D.load_transactions(str(raw), end=None)
    assert tx["ts"].tolist() == [pd.Timestamp("2022-09-01 00:15"), pd.Timestamp("2022-09-01 00:20"),
                                 pd.Timestamp("2022-09-01 00:20")]
    assert tx["payment_format"].tolist() == ["Cheque", "Reinvestment", "ACH"]   # stable within 00:20
    assert tx.loc[0, "sender"] == "03208:8000F4580" and tx["label"].tolist() == [0, 0, 1]
    assert len(D.load_transactions(str(raw), end=None, nrows=2)) == 2
    assert len(D.load_transactions(str(raw), end="2022-09-01 00:16")) == 1
    with pytest.raises(FileNotFoundError):
        D.load_transactions(str(tmp_path))


def test_time_split(aml):
    s = aml.split
    ts = aml.tx["ts"]
    assert set(s) == {"train", "val", "test"}
    assert (ts[s == "train"] < SPLITS[0]).all()
    assert ((ts[s == "val"] >= SPLITS[0]) & (ts[s == "val"] < SPLITS[1])).all()
    assert (ts[s == "test"] >= SPLITS[1]).all()
    # contiguous in time: a bank trains on the past and is judged on the future
    order = {"train": 0, "val": 1, "test": 2}
    assert (np.diff([order[x] for x in s]) >= 0).all()


def test_time_split_boundaries_and_defaults():
    ts = pd.Series(pd.to_datetime(["2022-09-06 23:59", "2022-09-07 00:00", "2022-09-08 23:59", "2022-09-09 00:00"]))
    assert D.time_split(ts).tolist() == ["train", "val", "val", "test"]
    assert D.DEFAULT_SPLITS == ("2022-09-07", "2022-09-09") and D.DEFAULT_END == "2022-09-11"


def test_laundering_patterns(aml):
    pat = D.laundering_patterns(str(aml.raw_dir))
    assert list(pat.columns) == ["attempt", "pattern", "ts", "sender", "receiver", "amount_paid"]
    assert pat.groupby("attempt")["pattern"].first().tolist() == [k for k, _ in aml.attempts]
    assert len(pat) == sum(len(a) for _, a in aml.attempts)
    first = aml.attempts[0][1][0]
    row = pat.iloc[0]
    assert row["ts"] == first.ts.strftime("%Y/%m/%d %H:%M")
    assert row["sender"] == f"{first.from_bank}:{first.from_acct}"
    assert row["receiver"] == f"{first.to_bank}:{first.to_acct}"
    assert row["amount_paid"] == first.amount_paid


def test_laundering_patterns_names_with_dashes(tmp_path):
    (tmp_path / "X_Patterns.txt").write_text(
        "BEGIN LAUNDERING ATTEMPT - GATHER-SCATTER:  Max 5-degree Fan-In\n"
        "2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1\n"
        "END LAUNDERING ATTEMPT - GATHER-SCATTER\n\n"
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 3 hops\n"
        "2022/09/01 01:06,012,80011F990,021174,800737690,10.00,Euro,10.00,Euro,ACH,1\n"
        "END LAUNDERING ATTEMPT - CYCLE\n")
    pat = D.laundering_patterns(str(tmp_path))
    assert pat["pattern"].tolist() == ["GATHER-SCATTER", "CYCLE"] and pat["attempt"].tolist() == [0, 1]
    assert pat.loc[0, "sender"] == "021174:800737690"


def test_patterns_rows_are_in_the_transactions(aml):
    pat = D.laundering_patterns(str(aml.raw_dir))
    pat["ts"] = pd.to_datetime(pat["ts"], format="%Y/%m/%d %H:%M")
    m = pat.merge(aml.tx, on=["ts", "sender", "receiver", "amount_paid"])
    assert len(m) == len(pat[pat["ts"] < END]) and (m["label"] == 1).all()


def test_write_aml_is_deterministic(tmp_path):
    write_aml(tmp_path / "a", seed=3)
    write_aml(tmp_path / "b", seed=3)
    assert (tmp_path / "a" / "XS-Small_Trans.csv").read_text() == (tmp_path / "b" / "XS-Small_Trans.csv").read_text()
