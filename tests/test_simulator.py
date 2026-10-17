"""simulator.py: injected billing errors, stream loading, and one replay against the in-process API."""
import asyncio
import datetime as dt
import json
import random

import pytest

from claimsense import simulator
from claimsense.simulator import INJECTIONS, deceased_benes, inject, load_stream, uncovered_benes

from conftest import DECEASED, NO_COVER, claim_json, write_bene_csv

D = dt.date.fromisoformat
DEAD = {"BENE11002": "2009-06-01", "BENE11007": "2009-02-15"}
UNCOVERED = {"A": ["BENE20001", "BENE20002"], "B": ["BENE30001"]}


@pytest.fixture
def claim():
    return claim_json("CLM1", start="2009-03-01", end="2009-03-04", amount=250.0, AdmissionDt="2009-03-01",
                      DischargeDt="2009-03-04", inpatient=1)


def test_inject_returns_a_copy_with_new_claim_id(claim):
    before = dict(claim)
    assert set(INJECTIONS) == {"amount_typo", "after_death", "dates_swapped", "duplicate", "no_attending",
                               "no_coverage"}
    for n, kind in enumerate(INJECTIONS):
        out = inject(claim, kind, DEAD, random.Random(0), n, UNCOVERED)
        assert out is not None and out is not claim
        assert out["ClaimID"] == f"CLM1-INJ{n}"
    assert claim == before


def test_inject_amount_typo(claim):
    out = inject(claim, "amount_typo", DEAD, random.Random(0), 0)
    assert out["InscClaimAmtReimbursed"] == 2500.0
    assert inject({**claim, "InscClaimAmtReimbursed": 0.0}, "amount_typo", DEAD, random.Random(0), 0) is None


@pytest.mark.parametrize("seed", range(5))
def test_inject_after_death(claim, seed):
    out = inject(claim, "after_death", DEAD, random.Random(seed), 0)
    assert out["BeneID"] in DEAD
    dod = D(DEAD[out["BeneID"]])
    assert D(out["ClaimStartDt"]) > dod
    assert D(out["ClaimEndDt"]) - D(out["ClaimStartDt"]) == D(claim["ClaimEndDt"]) - D(claim["ClaimStartDt"])
    assert out["AdmissionDt"] is None and out["DischargeDt"] is None and out["is_inpatient"] == 0


def test_inject_after_death_without_end_date(claim):
    out = inject({**claim, "ClaimEndDt": None}, "after_death", DEAD, random.Random(1), 0)
    assert out["ClaimEndDt"] == out["ClaimStartDt"]


def test_inject_after_death_impossible_without_deceased(claim):
    assert inject(claim, "after_death", {}, random.Random(0), 0) is None


@pytest.mark.parametrize("end", ["2009-03-04", "2009-03-01", None])
def test_inject_dates_swapped(claim, end):
    out = inject({**claim, "ClaimEndDt": end}, "dates_swapped", DEAD, random.Random(0), 0)
    assert D(out["ClaimEndDt"]) < D(out["ClaimStartDt"])


def test_inject_duplicate_keeps_content(claim):
    out = inject(claim, "duplicate", DEAD, random.Random(0), 3)
    assert {k: v for k, v in out.items() if k != "ClaimID"} == {k: v for k, v in claim.items() if k != "ClaimID"}
    assert out["ClaimID"] != claim["ClaimID"]


def test_inject_no_attending(claim):
    out = inject(claim, "no_attending", DEAD, random.Random(0), 0)
    assert out["AttendingPhysician"] is None
    assert {k: v for k, v in out.items() if k not in ("ClaimID", "AttendingPhysician")} == \
        {k: v for k, v in claim.items() if k not in ("ClaimID", "AttendingPhysician")}


@pytest.mark.parametrize("inpatient,pool", [(1, "A"), (0, "B")])
def test_inject_no_coverage_picks_a_bene_without_the_right_part(claim, inpatient, pool):
    c = {**claim, "is_inpatient": inpatient}
    for seed in range(5):
        out = inject(c, "no_coverage", DEAD, random.Random(seed), 0, UNCOVERED)
        assert out["BeneID"] in UNCOVERED[pool]
        assert {k: v for k, v in out.items() if k not in ("ClaimID", "BeneID")} == \
            {k: v for k, v in c.items() if k not in ("ClaimID", "BeneID")}


@pytest.mark.parametrize("uncovered", [None, {}, {"A": [], "B": ["BENE30001"]}])
def test_inject_no_coverage_impossible(claim, uncovered):
    assert inject({**claim, "is_inpatient": 1}, "no_coverage", DEAD, random.Random(0), 0, uncovered) is None


def test_injected_errors_trip_the_matching_rules(claim):
    from claimsense.features import duplicate_key, hard_errors

    bene = {"DOD": None, "NoOfMonths_PartACov": 12, "NoOfMonths_PartBCov": 12}
    assert hard_errors(claim, bene) == []
    rng = random.Random(0)
    dead = inject(claim, "after_death", DEAD, rng, 0)
    assert "after_death" in hard_errors(dead, {**bene, "DOD": DEAD[dead["BeneID"]]})
    assert "end_before_start" in hard_errors(inject(claim, "dates_swapped", DEAD, rng, 1), bene)
    assert "missing_attending" in hard_errors(inject(claim, "no_attending", DEAD, rng, 2), bene)
    nc = inject(claim, "no_coverage", DEAD, rng, 4, UNCOVERED)
    assert "no_coverage" in hard_errors(nc, {**bene, "NoOfMonths_PartACov": 0})
    dup = inject(claim, "duplicate", DEAD, rng, 3)
    assert duplicate_key(dup, "B", "P") == duplicate_key(claim, "B", "P")


# ------------------------------------------------------------------------------ files
STREAM_HEADER = ["ClaimID", "BeneID", "Provider", "ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt",
                 "InscClaimAmtReimbursed", "DeductibleAmtPaid", "AttendingPhysician", "OperatingPhysician",
                 "ClmDiagnosisCode_1", "ClmProcedureCode_1", "is_inpatient"]


def write_stream(path, rows):
    path.write_text("\n".join([",".join(STREAM_HEADER)] + [",".join(r) for r in rows]) + "\n")


def test_load_stream_types_and_missing(tmp_path):
    p = tmp_path / "stream.csv"
    write_stream(p, [
        ["CLM1", "BENE11001", "PRV51001", "2009-01-01", "2009-01-02", "", "", "500", "", "PHY1", "NA", "0389",
         "7092", "0"],
        ["CLM2", "BENE11002", "PRV51001", "2009-01-03", "", "2009-01-03", "2009-01-05", "", "1068", "", "",
         "V5866", "", "1"],
        ["CLM3", "BENE11003", "PRV51002", "2009-01-04", "2009-01-04", "", "", "10.5", "0", "PHY2", "", "4019",
         "", "0"],
    ])
    rows = load_stream(p, limit=None)
    assert len(rows) == 3
    a, b, _ = rows
    assert a["InscClaimAmtReimbursed"] == 500.0 and a["DeductibleAmtPaid"] == 0.0 and a["is_inpatient"] == 0
    assert a["AdmissionDt"] is None and a["OperatingPhysician"] is None
    assert a["ClmDiagnosisCode_1"] == "0389" and a["ClmProcedureCode_1"] == "7092"   # read as text
    assert b["InscClaimAmtReimbursed"] == 0.0 and b["DeductibleAmtPaid"] == 1068.0 and b["is_inpatient"] == 1
    assert b["ClaimEndDt"] is None and b["AttendingPhysician"] is None
    assert isinstance(a["InscClaimAmtReimbursed"], float) and isinstance(b["is_inpatient"], int)
    assert [r["ClaimID"] for r in load_stream(p, limit=2)] == ["CLM1", "CLM2"]
    assert [r["ClaimID"] for r in load_stream(p, limit=1, offset=1)] == ["CLM2"]


def test_loaded_stream_rows_validate_as_api_claims(tmp_path):
    from claimsense.api.schemas import ClaimIn

    p = tmp_path / "stream.csv"
    write_stream(p, [["CLM1", "BENE11001", "PRV51001", "2009-01-01", "", "", "", "5", "", "", "", "0389", "", "0"]])
    c = ClaimIn(**load_stream(p, None)[0])
    assert c.ClaimEndDt is None and c.ClmDiagnosisCode_1 == "0389" and c.AttendingPhysician is None


def test_deceased_and_uncovered_benes(tmp_path):
    assert deceased_benes(tmp_path) == {} and uncovered_benes(tmp_path) == {}
    write_bene_csv(tmp_path / "Train_Beneficiarydata-123.csv")
    assert deceased_benes(tmp_path) == DECEASED
    assert uncovered_benes(tmp_path) == {"A": [b for b, p in NO_COVER.items() if p == "A"],
                                         "B": [b for b, p in NO_COVER.items() if p == "B"]}


# ------------------------------------------------------------------------------ replay against the app
class _ClientOverTestClient:
    """Stands in for httpx.AsyncClient; forwards to a running TestClient (so the app's lifespan loop is used)."""

    def __init__(self, tc, base_url, headers, timeout, limits):
        self.tc, self.headers = tc, headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, path, json):
        return await asyncio.to_thread(self.tc.post, path, json=json, headers=self.headers)


def test_run_replays_and_reports_injections(api_env, tmp_path, monkeypatch):
    claims = [claim_json(f"CLM{i}", provider=f"PRV5100{i % 3}", bene=f"BENE{11003 + i % 7}",
                         start=f"2009-0{1 + i % 5}-1{i % 9}", end=None, amount=100.0 + 10 * i) for i in range(10)]
    out = tmp_path / "sim"
    with api_env.client(batch_window_ms=5.0) as tc:
        monkeypatch.setattr(simulator.httpx, "AsyncClient",
                            lambda **kw: _ClientOverTestClient(tc, **kw))
        summary = asyncio.run(simulator.run("http://test", api_env.service_token, claims, rate=0, concurrency=8,
                                            inject_rate=1.0, raw_dir=api_env.raw, seed=1, burst_every=0,
                                            burst_size=0, out_dir=out))
    assert summary["sent"] == 20
    assert sum(summary["outcomes"].values()) == 20
    assert not any(k.startswith(("http_", "error:")) for k in summary["outcomes"]), summary["outcomes"]
    inj = summary["injected_errors"]
    assert sum(v["injected"] for v in inj.values()) == 10
    # round robin over the 6 kinds: 10 injections -> 2,2,2,2,1,1
    assert [inj[k]["injected"] for k in INJECTIONS] == [2, 2, 2, 2, 1, 1]
    for kind in ("after_death", "dates_swapped", "duplicate", "no_attending", "no_coverage"):  # rules: always caught
        assert inj[kind]["caught"] == inj[kind]["injected"], (kind, inj)
    files = sorted(p.name for p in out.iterdir())
    assert any(f.startswith("replay-") for f in files) and any(f.startswith("injected-") for f in files)
    saved = json.loads(next(out.glob("replay-*.json")).read_text())
    assert saved["sent"] == 20


def test_run_keeps_injecting_other_kinds_when_one_kind_is_impossible(api_env, tmp_path, monkeypatch):
    write_bene_csv(api_env.raw / "Train_Beneficiarydata-1.csv", no_cover={})   # nobody lacks coverage
    claims = [claim_json(f"CLM{i}", provider=f"PRV5100{i % 3}", bene=f"BENE{11003 + i % 7}",
                         start=f"2009-0{1 + i % 5}-1{i % 9}", end=None, amount=100.0 + 10 * i) for i in range(12)]
    with api_env.client() as tc:
        monkeypatch.setattr(simulator.httpx, "AsyncClient", lambda **kw: _ClientOverTestClient(tc, **kw))
        summary = asyncio.run(simulator.run("http://test", api_env.service_token, claims, rate=0, concurrency=4,
                                            inject_rate=1.0, raw_dir=api_env.raw, seed=1, burst_every=0,
                                            burst_size=0, out_dir=tmp_path / "sim"))
    # every claim is followed by an injection of SOME possible kind
    assert summary["sent"] == 24
