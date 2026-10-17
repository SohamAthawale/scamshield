"""Payment stream simulator: replays the held-out days (in time order) against POST /payments.

It also injects two labelled kinds of payment the public data does not contain:
  victim_to_mule   a real customer suddenly sends 8-30x their usual amount to an account that the
                   dataset shows receiving laundered money (the account a scammer gives a victim).
                   Should be WARNED or HELD.
  honest_new_payee the same kind of large first payment, but to a brand-new, clean account (paying a
                   builder, a car dealer). Measures how often we bother honest customers.

The API is never told which payments were injected; the simulator keeps that ground truth and reports it.

    python -m scamshield.simulator --url http://api:8000 --rate 100 --limit 5000 --inject 0.01
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as dt
import json
import os
import random
import statistics
import time
from pathlib import Path

import httpx
import pandas as pd

INJECTIONS = ("victim_to_mule", "honest_new_payee")
FIELDS = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amount_paid", "currency_paid", "amount_received",
          "currency_received", "payment_format"]


def load_stream(path: Path, limit: int | None, offset: int = 0) -> list[dict]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["payment_id"] = [f"T{i}" for i in range(len(df))]
    df = df.iloc[offset:offset + limit] if limit else df.iloc[offset:]
    out = []
    for r in df.to_dict("records"):
        p = {k: r[k] for k in FIELDS}
        p["payment_id"] = r["payment_id"]
        p["amount_paid"], p["amount_received"] = float(p["amount_paid"]), float(p["amount_received"])
        out.append(p)
    return out


def mule_accounts(raw_dir: Path, start: str = "2022-09-09") -> list[tuple[str, str]]:
    """(bank, account) pairs that receive laundered money in the live period, per the dataset's pattern file."""
    files = sorted(raw_dir.glob("*_Patterns.txt"))
    if not files:
        return []
    out = set()
    for line in open(files[0]):
        p = line.strip().split(",")
        if len(p) >= 11 and p[0] >= start.replace("-", "/"):
            out.add((p[3], p[4]))
    return sorted(out)


def inject(base: dict, kind: str, history: dict[str, list[float]], mules: list[tuple[str, str]],
           rng: random.Random, n: int) -> dict | None:
    """A new payment from base's sender, or None when that sender has too little history to be 'usual'."""
    prior = history.get(f"{base['from_bank']}:{base['from_acct']}", [])
    if len(prior) < 3:
        return None
    usual = statistics.median(prior)
    if usual <= 0:
        return None
    c = {**base, "payment_id": f"INJ{n}-{kind}"}
    c["amount_paid"] = c["amount_received"] = round(usual * rng.uniform(8, 30), 2)
    c["currency_received"] = c["currency_paid"]
    c["payment_format"] = "ACH" if rng.random() < 0.7 else "Wire"
    if kind == "victim_to_mule":
        if not mules:
            return None
        c["to_bank"], c["to_acct"] = rng.choice(mules)
    else:  # honest: a brand-new account at a real bank
        c["to_acct"] = f"NEW{n:07d}"
    return c


async def run(url: str, token: str, payments: list[dict], rate: float, concurrency: int, inject_rate: float,
              raw_dir: Path, seed: int, out_dir: Path, file_size: int = 0) -> dict:
    rng = random.Random(seed)
    mules = mule_accounts(raw_dir) if inject_rate > 0 else []
    history: dict[str, list[float]] = collections.defaultdict(list)
    queue: list[tuple[dict, str | None]] = []
    n_inj, attempt = 0, 0
    for p in payments:
        queue.append((p, None))
        if inject_rate and rng.random() < inject_rate:
            for _ in range(len(INJECTIONS)):  # rotate kinds; skip one that's impossible here (e.g. no mules)
                kind = INJECTIONS[attempt % len(INJECTIONS)]
                attempt += 1
                c = inject(p, kind, history, mules, rng, n_inj)
                if c is not None:
                    queue.append((c, kind))
                    n_inj += 1
                    break
        history[f"{p['from_bank']}:{p['from_acct']}"].append(p["amount_paid"])

    stats, injected, latencies, backends = collections.Counter(), [], [], collections.Counter()
    sem = asyncio.Semaphore(concurrency)
    t_start = time.perf_counter()
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=url, headers={"Authorization": f"Bearer {token}"}, timeout=30.0,
                                 limits=limits) as client:
        async def send(p: dict, kind: str | None) -> None:
            async with sem:
                t = time.perf_counter()
                try:
                    r = await client.post("/payments", json=p)
                except httpx.HTTPError as e:
                    stats[f"error:{type(e).__name__}"] += 1
                    return
                rt = (time.perf_counter() - t) * 1000.0
                if r.status_code != 200:
                    stats[f"http_{r.status_code}"] += 1
                    return
                record(p, kind, r.json(), rt)

        def record(p: dict, kind: str | None, d: dict, rt: float) -> None:
            stats[d["action"]] += 1
            latencies.append((rt, d["latency_ms"], d["batch_size"]))
            backends[d.get("backend", "?")] += 1
            if kind:
                injected.append({"payment_id": p["payment_id"], "kind": kind, "action": d["action"],
                                 "format": p["payment_format"], "risk": d["risk"],
                                 "rules": [x["rule"] for x in d["reasons"].get("rules", [])]})

        async def send_file(chunk: list[tuple[dict, str | None]]) -> None:
            """One bulk payment file (POST /payments/batch); every payment in it gets the file's latency."""
            async with sem:
                t = time.perf_counter()
                try:
                    r = await client.post("/payments/batch", json={"payments": [p for p, _ in chunk]})
                except httpx.HTTPError as e:
                    stats[f"error:{type(e).__name__}"] += len(chunk)
                    return
                rt = (time.perf_counter() - t) * 1000.0
                if r.status_code != 200:
                    stats[f"http_{r.status_code}"] += len(chunk)
                    return
                for (p, kind), d in zip(chunk, r.json()["results"]):
                    record(p, kind, d, rt)

        if file_size > 0:
            await asyncio.gather(*(send_file(queue[i:i + file_size]) for i in range(0, len(queue), file_size)))
            queue = []
        tasks = []
        for i, (p, kind) in enumerate(queue):
            tasks.append(asyncio.create_task(send(p, kind)))
            if rate > 0:
                delay = t_start + (i + 1) / rate - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
        await asyncio.gather(*tasks)

    sent = sum(stats.values())
    elapsed = time.perf_counter() - t_start
    lat = sorted(x[0] for x in latencies) or [0.0]
    srv = sorted(x[1] for x in latencies) or [0.0]
    inj = pd.DataFrame(injected, columns=["payment_id", "kind", "action", "format", "risk", "rules"])
    by_kind = {}
    for k in INJECTIONS:
        s = inj[inj["kind"] == k]
        by_kind[k] = {"injected": int(len(s)), "warned": int((s["action"] == "WARN").sum()),
                      "held": int((s["action"] == "HOLD").sum()), "allowed": int((s["action"] == "ALLOW").sum())}
    summary = {
        "sent": sent, "elapsed_s": round(elapsed, 2), "payments_per_sec": round(len(latencies) / elapsed, 1),
        "outcomes": dict(stats),
        "client_latency_ms": {"p50": round(lat[len(lat) // 2], 2), "p99": round(lat[max(int(len(lat) * 0.99) - 1, 0)], 2)},
        "server_latency_ms": {"p50": round(srv[len(srv) // 2], 2), "p99": round(srv[max(int(len(srv) * 0.99) - 1, 0)], 2)},
        "avg_batch": round(sum(x[2] for x in latencies) / max(len(latencies), 1), 1),
        "max_batch": max((x[2] for x in latencies), default=0),
        "anomaly_backend": dict(backends),
        "injected": by_kind,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"replay-{stamp}.json").write_text(json.dumps(summary, indent=1))
    inj.to_csv(out_dir / f"injected-{stamp}.csv", index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("SCAMSHIELD_URL", "http://localhost:8000"))
    ap.add_argument("--token-file", default=os.environ.get("SERVICE_TOKEN_FILE", "/run/secrets/service_token"))
    ap.add_argument("--stream", type=Path, default=Path(os.environ.get("STREAM_FILE", "/data-in/stream_test.csv")))
    ap.add_argument("--raw", type=Path, default=Path(os.environ.get("SCAMSHIELD_RAW", "/raw")))
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("SIM_OUT", "/sim")))
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--rate", type=float, default=100.0, help="payments/sec; 0 = as fast as possible")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--inject", type=float, default=0.01, help="share of payments followed by an injected one")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--file-size", type=int, default=0,
                    help="send bulk payment files of this many payments (POST /payments/batch); 0 = one by one")
    a = ap.parse_args()
    token = Path(a.token_file).read_text().strip()
    s = asyncio.run(run(a.url, token, load_stream(a.stream, a.limit, a.offset), a.rate, a.concurrency, a.inject,
                        a.raw, a.seed, a.out, a.file_size))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
