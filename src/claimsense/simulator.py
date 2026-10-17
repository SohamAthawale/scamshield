"""Claim stream simulator: replays held-out claims (test providers, in time order) against POST /claims.

It also injects a small, labelled set of HONEST BILLING ERRORS (the public dataset contains almost none),
built from real claims:
  amount_typo     an extra zero on the amount (x10)
  after_death     service date moved past the patient's recorded date of death
  dates_swapped   end date before start date
  duplicate       the same service submitted twice under a new claim number
  no_attending    attending physician left blank

The API is never told which claims were injected; the simulator keeps that ground truth locally and
reports how many were caught (RETURN_WITH_FIX or HOLD_FOR_REVIEW).

    python -m claimsense.simulator --url http://api:8000 --rate 200 --limit 5000 --inject 0.01
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as dt
import json
import os
import random
import time
from pathlib import Path

import httpx
import pandas as pd

INJECTIONS = ("amount_typo", "after_death", "dates_swapped", "duplicate", "no_attending")


def _shift(date_str: str, days: int) -> str:
    return (dt.date.fromisoformat(date_str) + dt.timedelta(days=days)).isoformat()


def inject(claim: dict, kind: str, deceased: dict[str, str], rng: random.Random, n: int) -> dict | None:
    c = dict(claim)
    c["ClaimID"] = f"{claim['ClaimID']}-INJ{n}"
    if kind == "amount_typo":
        if not c["InscClaimAmtReimbursed"]:
            return None
        c["InscClaimAmtReimbursed"] = float(c["InscClaimAmtReimbursed"]) * 10
    elif kind == "after_death":
        if not deceased:
            return None
        bene, dod = rng.choice(list(deceased.items()))
        c["BeneID"] = bene
        gap = rng.randint(3, 40)
        length = 0 if not c.get("ClaimEndDt") else (dt.date.fromisoformat(c["ClaimEndDt"])
                                                     - dt.date.fromisoformat(c["ClaimStartDt"])).days
        c["ClaimStartDt"] = _shift(dod, gap)
        c["ClaimEndDt"] = _shift(dod, gap + max(length, 0))
        for k in ("AdmissionDt", "DischargeDt"):
            if c.get(k):
                c[k] = None
        c["is_inpatient"] = 0
    elif kind == "dates_swapped":
        if not c.get("ClaimEndDt") or c["ClaimEndDt"] == c["ClaimStartDt"]:
            c["ClaimEndDt"] = _shift(c["ClaimStartDt"], 5)
        c["ClaimStartDt"], c["ClaimEndDt"] = c["ClaimEndDt"], c["ClaimStartDt"]
    elif kind == "duplicate":
        pass  # identical service, new claim number
    elif kind == "no_attending":
        c["AttendingPhysician"] = None
    return c


def load_stream(path: Path, limit: int | None, offset: int = 0) -> list[dict]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if offset:
        df = df.iloc[offset:]
    if limit:
        df = df.head(limit)
    out = []
    for r in df.to_dict("records"):
        r = {k: (None if v in ("", "NA") else v) for k, v in r.items()}
        r["InscClaimAmtReimbursed"] = float(r["InscClaimAmtReimbursed"] or 0)
        r["DeductibleAmtPaid"] = float(r["DeductibleAmtPaid"] or 0)
        r["is_inpatient"] = int(r["is_inpatient"])
        out.append(r)
    return out


def deceased_benes(raw_dir: Path) -> dict[str, str]:
    files = sorted(raw_dir.glob("Train_Beneficiarydata*.csv"))
    if not files:
        return {}
    b = pd.read_csv(files[0], usecols=["BeneID", "DOD"], dtype=str)
    b = b[b["DOD"].notna() & (b["DOD"] != "NA")]
    return dict(zip(b["BeneID"], b["DOD"]))


async def run(url: str, token: str, claims: list[dict], rate: float, concurrency: int, inject_rate: float,
              raw_dir: Path, seed: int, burst_every: int, burst_size: int, out_dir: Path) -> dict:
    rng = random.Random(seed)
    deceased = deceased_benes(raw_dir) if inject_rate > 0 else {}
    queue: list[tuple[dict, str | None, int | None]] = []  # (claim, injected kind, index it must wait for)
    n_inj = 0
    for c in claims:
        queue.append((c, None, None))
        orig = len(queue) - 1
        if inject_rate and rng.random() < inject_rate:
            kind = INJECTIONS[n_inj % len(INJECTIONS)]
            ic = inject(c, kind, deceased, rng, n_inj)
            if ic is not None:
                # a duplicate is a RE-submission: it follows the original's answer, never races it
                queue.append((ic, kind, orig if kind == "duplicate" else None))
                n_inj += 1

    stats = collections.Counter()
    injected_results = []
    latencies = []
    sem = asyncio.Semaphore(concurrency)
    t_start = time.perf_counter()
    headers = {"Authorization": f"Bearer {token}"}
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=30.0, limits=limits) as client:
        async def send(claim: dict, kind: str | None) -> None:
            async with sem:
                t = time.perf_counter()
                try:
                    r = await client.post("/claims", json=claim)
                except httpx.HTTPError as e:
                    stats[f"error:{type(e).__name__}"] += 1
                    return
                rt = (time.perf_counter() - t) * 1000.0
                if r.status_code != 200:
                    stats[f"http_{r.status_code}"] += 1
                    return
                d = r.json()
                stats[d["action"]] += 1
                latencies.append((rt, d["latency_ms"], d["batch_size"]))
                if kind:
                    injected_results.append({"claim_id": claim["ClaimID"], "kind": kind, "action": d["action"],
                                             "rules": [x["rule"] for x in d["reasons"].get("rules", [])]})

        tasks: list[asyncio.Task] = []

        async def after(dep: asyncio.Task, claim: dict, kind: str) -> None:
            await dep
            await send(claim, kind)

        for i, (c, kind, wait_for) in enumerate(queue):
            coro = after(tasks[wait_for], c, kind) if wait_for is not None else send(c, kind)
            tasks.append(asyncio.create_task(coro))
            in_burst = burst_every and (i % burst_every) < burst_size
            if rate > 0 and not in_burst:
                target = t_start + (i + 1) / rate
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
        await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t_start
    lat = sorted(x[0] for x in latencies) or [0.0]
    srv = sorted(x[1] for x in latencies) or [0.0]
    caught = collections.Counter((r["kind"], r["action"] != "PAY") for r in injected_results)
    by_kind = {k: {"injected": caught[(k, True)] + caught[(k, False)], "caught": caught[(k, True)]} for k in INJECTIONS}
    summary = {
        "sent": len(queue), "elapsed_s": round(elapsed, 2), "claims_per_sec": round(len(latencies) / elapsed, 1),
        "outcomes": dict(stats),
        "client_latency_ms": {"p50": round(lat[len(lat) // 2], 2), "p99": round(lat[int(len(lat) * 0.99) - 1], 2)},
        "server_latency_ms": {"p50": round(srv[len(srv) // 2], 2), "p99": round(srv[int(len(srv) * 0.99) - 1], 2)},
        "avg_batch": round(sum(x[2] for x in latencies) / max(len(latencies), 1), 1),
        "injected_errors": by_kind,
        "injected_caught_total": f"{sum(v['caught'] for v in by_kind.values())}/{sum(v['injected'] for v in by_kind.values())}",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"replay-{stamp}.json").write_text(json.dumps(summary, indent=1))
    pd.DataFrame(injected_results).to_csv(out_dir / f"injected-{stamp}.csv", index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("CLAIMSENSE_URL", "http://localhost:8000"))
    ap.add_argument("--token-file", default=os.environ.get("SERVICE_TOKEN_FILE", "/run/secrets/service_token"))
    ap.add_argument("--stream", type=Path, default=Path(os.environ.get("STREAM_FILE", "/data-in/stream_test.csv")))
    ap.add_argument("--raw", type=Path, default=Path(os.environ.get("CLAIMSENSE_RAW", "/raw")))
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("SIM_OUT", "/data/sim")))
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--rate", type=float, default=100.0, help="claims/sec; 0 = as fast as possible")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--inject", type=float, default=0.01, help="share of claims followed by an injected error")
    ap.add_argument("--burst-every", type=int, default=0, help="every N claims ...")
    ap.add_argument("--burst-size", type=int, default=0, help="... send this many with no pacing")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    token = Path(a.token_file).read_text().strip()
    claims = load_stream(a.stream, a.limit, a.offset)
    s = asyncio.run(run(a.url, token, claims, a.rate, a.concurrency, a.inject, a.raw, a.seed, a.burst_every,
                        a.burst_size, a.out))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
