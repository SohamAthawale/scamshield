"""ScamShield API.

POST /payments -> pseudonymise accounts -> micro-batch -> rolling features + risk model + customer-anomaly
model + scam rules -> ALLOW / WARN / HOLD -> encrypt + store + audit -> live event -> instant message to
the customer (Granite rewrites it later) -> respond with the decision, reasons and latency.

At start-up the rolling state is built from core-banking history (the days before the live stream), so
"new payee" and "unusual for this customer" mean what they say from the first live payment.
"""
from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import explain
from ..fusion import ALLOW, HOLD, WARN
from ..inference import load_anomaly_scorer, load_risk_scorer
from ..scoring import Engine, PaymentInput
from ..security import Pseudonymizer, load_keyring, read_secret
from ..store import PaymentRecord, Store
from .auth import Authenticator, Principal, require
from .schemas import LoginIn, PaymentFileIn, PaymentFileOut, PaymentIn, PaymentOut, ResolveIn
from .settings import Settings

log = logging.getLogger("scamshield.api")
UI_DIR = Path(__file__).resolve().parent.parent / "ui"
HISTORY_COLUMNS = ["ts", "from_bank", "sender", "to_bank", "receiver", "amount_paid", "currency_paid",
                   "currency_received", "payment_format"]


# ---------------------------------------------------------------------------------------- live metrics
class Metrics:
    def __init__(self, window: int = 5000, nnpa_min_batch: int = 0):
        self.recent: collections.deque = collections.deque(maxlen=window)  # (t, latency_ms, model_ms)
        self.counts = collections.Counter()
        self.amounts = collections.Counter()
        self.batch_sizes: collections.deque = collections.deque(maxlen=500)
        self.on_nnpa = 0  # payments whose customer-anomaly score actually ran on the Telum II NNPA
        self.nnpa_min_batch = nnpa_min_batch
        self.started = time.time()

    def add(self, latency_ms: float, model_ms: float, action: str, amount_usd: float, on_nnpa: bool = False) -> None:
        self.recent.append((time.time(), latency_ms, model_ms))
        self.counts[action] += 1
        self.amounts[action] += amount_usd
        self.on_nnpa += bool(on_nnpa)

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        lat = np.array([r[1] for r in self.recent]) if self.recent else np.array([0.0])
        mdl = np.array([r[2] for r in self.recent]) if self.recent else np.array([0.0])
        last10 = sum(1 for r in self.recent if now - r[0] <= 10.0)
        total = sum(self.counts.values())
        return {
            "payments_total": total,
            "payments_per_sec": round(last10 / 10.0, 1),
            "latency_p50_ms": round(float(np.percentile(lat, 50)), 2),
            "latency_p99_ms": round(float(np.percentile(lat, 99)), 2),
            "model_p50_ms": round(float(np.percentile(mdl, 50)), 3),
            "decisions": {a: self.counts[a] for a in (ALLOW, WARN, HOLD)},
            "usd_held": round(self.amounts[HOLD], 2),
            "usd_warned": round(self.amounts[WARN], 2),
            "share_scored": 1.0 if total else 0.0,
            "avg_batch_size": round(float(np.mean(self.batch_sizes)), 1) if self.batch_sizes else 0.0,
            "max_batch_size": int(max(self.batch_sizes)) if self.batch_sizes else 0,
            "payments_on_nnpa": self.on_nnpa,
            "nnpa_share": round(self.on_nnpa / total, 4) if total else 0.0,
            "nnpa_min_batch": self.nnpa_min_batch,
            "uptime_s": round(now - self.started),
        }


# ---------------------------------------------------------------------------------------- SSE hub
class Hub:
    def __init__(self) -> None:
        self.subs: dict[asyncio.Queue, Principal] = {}

    def subscribe(self, p: Principal) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.subs[q] = p
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subs.pop(q, None)

    def publish(self, event: str, data: dict[str, Any], account_token: str | None = None,
                roles: tuple[str, ...] = ("analyst", "auditor", "customer")) -> None:
        for q, p in list(self.subs.items()):
            if p.role not in roles:
                continue
            if p.role == "customer" and (event == "metrics" or account_token != p.account_token):
                continue  # customers only ever see their own payments
            if p.role == "auditor" and event != "metrics":
                continue  # auditors get aggregates only
            if not q.full():  # a slow browser must never slow down scoring
                q.put_nowait((event, data))


# ---------------------------------------------------------------------------------------- service
class Service:
    def __init__(self, s: Settings):
        self.s = s
        sec = s.secrets_dir
        self.pseudo = Pseudonymizer(bytes.fromhex(read_secret("pseudo_key", sec)))
        self._tok: dict[str, str] = {}
        self.store = Store(s.db_path, load_keyring(sec), s.checkpoint_path)
        users = json.loads(read_secret("users", sec))
        self.auth = Authenticator(read_secret("jwt_secret", sec), users, read_secret("service_token", sec),
                                  s.jwt_ttl_minutes, self.account_token)
        self.anomaly = load_anomaly_scorer(s.models_dir, s.anomaly_backend, s.nnpa_min_batch)
        self.risk = load_risk_scorer(s.models_dir)
        self.engine = Engine.from_artifacts(s.models_dir, s.policy_file, self.anomaly, self.risk)
        self.history_payments = self._bootstrap()
        self.restored_payments = self._restore_state()
        self.metrics = Metrics(nnpa_min_batch=s.nnpa_min_batch if self.anomaly.name == "torch_nnpa" else 0)
        self.hub = Hub()
        self.queue: asyncio.Queue | None = None
        self.inflight: set[str] = set()
        # every touch of the engine (stateful) happens on this one thread
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")
        self.tasks: list[asyncio.Task] = []

    def account_token(self, raw: str) -> str:
        t = self._tok.get(raw)
        if t is None:
            t = self._tok[raw] = self.pseudo.token("account", raw)
        return t

    # ------------------------------------------------------------------ state
    def _bootstrap(self) -> int:
        """Rolling state from core-banking history (sorted history file written by training)."""
        path = self.s.history_file
        if not path.exists():
            log.warning("no history file at %s: starting with empty rolling state", path)
            return 0
        t, n = time.time(), 0
        for chunk in pd.read_csv(path, usecols=HISTORY_COLUMNS, chunksize=500_000,
                                 dtype={"from_bank": str, "to_bank": str, "sender": str, "receiver": str}):
            chunk = chunk[chunk["ts"] < self.s.history_end]
            if chunk.empty:
                break
            chunk["ts"] = pd.to_datetime(chunk["ts"])
            chunk["sender"] = chunk["sender"].map(self.account_token)
            chunk["receiver"] = chunk["receiver"].map(self.account_token)
            n += self.engine.bootstrap(chunk)
        log.info("rolling state from %d historical payments (%d accounts) in %.0fs", n, len(self.engine.fe.ids),
                 time.time() - t)
        return n

    def _input(self, raw: dict[str, Any]) -> PaymentInput:
        ts = pd.Timestamp(raw["ts"])
        ctx = self.engine.ctx
        return PaymentInput(
            t_minutes=ts.value / 6e10,
            sender_key=self.account_token(f"{raw['from_bank']}:{raw['from_acct']}"),
            receiver_key=self.account_token(f"{raw['to_bank']}:{raw['to_acct']}"),
            amount_usd=ctx.to_usd(raw["amount_paid"], raw["currency_paid"]),
            payment_format=raw["payment_format"], cross_bank=raw["from_bank"] != raw["to_bank"],
            currency_mismatch=raw["currency_paid"] != raw["currency_received"], hour=ts.hour)

    def _restore_state(self) -> int:
        """Payments scored before a restart are replayed into the rolling state (not re-scored)."""
        n, batch = 0, []
        for raw in self.store.iter_payloads():
            batch.append(self._input(raw))
            n += 1
            if len(batch) >= 5000:
                self.engine.replay(batch)
                batch = []
        self.engine.replay(batch)
        if n:
            log.info("replayed %d stored payments", n)
        return n

    # ------------------------------------------------------------------ batching
    async def submit(self, p: PaymentIn) -> dict[str, Any]:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((p, fut, time.perf_counter()))
        return await fut

    async def batch_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            deadline = loop.time() + self.s.batch_window_ms / 1000.0
            while len(batch) < self.s.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            try:
                outs = await loop.run_in_executor(self.pool, self._process, [b[0] for b in batch])
            except Exception as e:  # never leave a request hanging
                log.exception("batch failed")
                for _, fut, _ in batch:
                    if not fut.done():
                        fut.set_exception(e)
                continue
            self._finish([b[0] for b in batch], outs, [b[2] for b in batch])
            for (_, fut, _), out in zip(batch, outs):
                if not fut.done():
                    fut.set_result(out)

    async def submit_file(self, payments: list[PaymentIn]) -> list[dict[str, Any]]:
        """A bulk file is one batch. It runs on the same single engine thread as the live micro-batches,
        so rolling state is still updated one call at a time."""
        t0 = time.perf_counter()
        outs = await asyncio.get_running_loop().run_in_executor(self.pool, self._process, payments)
        self._finish(payments, outs, [t0] * len(payments))
        return outs

    def _finish(self, payments: list[PaymentIn], outs: list[dict[str, Any]], t0s: list[float]) -> None:
        self.metrics.batch_sizes.append(len(payments))
        for p, out, t0 in zip(payments, outs, t0s):
            out["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            out["batch_size"] = len(payments)
            self.metrics.add(out["latency_ms"], out["model_ms"], out["action"], out["_usd"],
                             out["backend"] == "torch_nnpa")
            self.hub.publish("payment", {
                "payment_token": out["payment_token"], "sender_token": out["_sender"],
                "receiver_token": out["_receiver"], "amount_usd": round(out["_usd"], 2),
                "payment_format": p.payment_format, "action": out["action"], "latency_ms": out["latency_ms"],
                "rules": [r["rule"] for r in out["reasons"].get("rules", [])],
            }, account_token=out["_sender"])

    def _process(self, payments: list[PaymentIn]) -> list[dict[str, Any]]:
        raws = [p.model_dump(mode="json") for p in payments]
        items = [self._input(r) for r in raws]
        results = self.engine.score_batch(items)
        backend = self.anomaly.name
        if backend == "torch_nnpa" and len(items) < self.s.nnpa_min_batch:
            backend = "torch_nnpa(cpu-path)"
        records, outs = [], []
        for raw, it, res in zip(raws, items, results):
            d = res.decision
            ptok = self.pseudo.token("payment", raw["payment_id"])
            notes, message = [], None
            if d.action != ALLOW:
                held = d.action == HOLD
                facts = {"rules": res.reasons.get("rules", []), "customer": res.reasons.get("customer", []),
                         "amount_usd": it.amount_usd, "payment_format": it.payment_format, "held": held}
                message = explain.customer_warning_template(res.reasons, held)
                notes.append(("warning", message, facts))
                if held:
                    brief = {**facts, "drivers": res.reasons.get("drivers", []), "risk": d.risk, "payment_token": ptok}
                    notes.append(("brief", explain.analyst_brief_template(res.reasons, ptok, d.risk), brief))
            records.append(PaymentRecord(
                ptok, it.sender_key, it.receiver_key, raw["ts"], it.amount_usd, it.payment_format, d.action,
                d.anomaly, d.risk, d.rules, res.reasons, res.model_ms, backend, raw, notes))
            outs.append({"payment_token": ptok, "action": d.action, "message": message, "reasons": res.reasons,
                         "risk": round(d.risk, 4), "anomaly": round(d.anomaly, 4),
                         "model_ms": round(res.model_ms, 4), "backend": backend,
                         "_usd": it.amount_usd, "_sender": it.sender_key, "_receiver": it.receiver_key})
        self.store.save_batch(records)
        return outs

    # ------------------------------------------------------------------ background tasks
    async def notes_loop(self) -> None:
        """Push Granite-written notes to browsers as the worker finishes them."""
        last = "1970"
        while True:
            await asyncio.sleep(1.0)
            try:
                rows = await asyncio.to_thread(self.store.notes_updated_since, last)
            except Exception:
                log.exception("notes poll failed")
                continue
            for r in rows:
                last = max(last, r["updated_at"])
                payload = {k: r[k] for k in ("payment_token", "kind", "status", "source", "text")}
                if r["kind"] == "warning":
                    self.hub.publish("note", payload, account_token=r["sender_token"], roles=("customer", "analyst"))
                else:  # analyst briefs never go to customers
                    self.hub.publish("note", payload, roles=("analyst",))

    async def metrics_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            self.hub.publish("metrics", self.metrics.snapshot(), roles=("analyst", "auditor"))


# ---------------------------------------------------------------------------------------- app
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = Service(settings)
        svc.queue = asyncio.Queue(maxsize=10_000)
        svc.store.requeue_stuck_notes()
        app.state.svc = svc
        app.state.auth = svc.auth
        svc.tasks = [asyncio.create_task(t) for t in (svc.batch_loop(), svc.notes_loop(), svc.metrics_loop())]
        svc.store.audit.append("system", "service_started", None, {
            "anomaly_backend": svc.anomaly.name, "risk_backend": svc.risk.name,
            "policy": vars(svc.engine.policy), "history_payments": svc.history_payments})
        log.info("ready: anomaly=%s risk=%s policy=%s", svc.anomaly.name, svc.risk.name, svc.engine.policy)
        yield
        for t in svc.tasks:
            t.cancel()
        svc.store.audit.checkpoint()
        svc.pool.shutdown(wait=False)

    app = FastAPI(title="ScamShield", version="0.2.0", lifespan=lifespan)

    def svc(request: Request) -> Service:
        return request.app.state.svc

    @app.get("/health")
    async def health(request: Request):
        s = svc(request)
        return {"status": "ok", "anomaly_backend": s.anomaly.name, "risk_backend": s.risk.name,
                "history_payments": s.history_payments, "accounts_tracked": len(s.engine.fe.ids)}

    @app.post("/auth/login")
    async def login(body: LoginIn, request: Request):
        s = svc(request)
        try:
            token, p = s.auth.login(body.username, body.password)
        except HTTPException:
            await asyncio.to_thread(s.store.audit.append, body.username[:64], "login_failed", None, {})
            raise
        await asyncio.to_thread(s.store.audit.append, p.name, "login", None, {"role": p.role})
        return {"access_token": token, "token_type": "bearer", "role": p.role, "account_token": p.account_token}

    @app.post("/payments", response_model=PaymentOut)
    async def post_payment(payment: PaymentIn, request: Request, p: Principal = Depends(require("service"))):
        s = svc(request)
        ptok = s.pseudo.token("payment", payment.payment_id)
        if ptok in s.inflight or await asyncio.to_thread(s.store.get_payment, ptok) is not None:
            raise HTTPException(409, "payment already submitted")
        s.inflight.add(ptok)
        try:
            out = await s.submit(payment)
        finally:
            s.inflight.discard(ptok)
        return {k: v for k, v in out.items() if not k.startswith("_")}

    @app.post("/payments/batch", response_model=PaymentFileOut)
    async def post_payment_file(file: PaymentFileIn, request: Request, p: Principal = Depends(require("service"))):
        """Score a bulk payment file in one call. All or nothing: any payment id already submitted (or repeated
        in the file) rejects the whole file with 409, so a resent file never double-counts."""
        s = svc(request)
        toks = [s.pseudo.token("payment", x.payment_id) for x in file.payments]
        if len(set(toks)) != len(toks):
            raise HTTPException(409, "payment id repeated in the file")
        if any(t in s.inflight for t in toks) or await asyncio.to_thread(s.store.any_payment, toks):
            raise HTTPException(409, "file contains a payment already submitted")
        s.inflight.update(toks)
        t0 = time.perf_counter()
        try:
            outs = await s.submit_file(file.payments)
        finally:
            s.inflight.difference_update(toks)
        return {"results": [{k: v for k, v in o.items() if not k.startswith("_")} for o in outs],
                "batch_size": len(outs), "backend": outs[0]["backend"],
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)}

    @app.get("/metrics")
    async def metrics(request: Request, p: Principal = Depends(require("analyst", "auditor"))):
        return svc(request).metrics.snapshot()

    @app.get("/stream")
    async def stream(request: Request, p: Principal = Depends(require("analyst", "auditor", "customer"))):
        s = svc(request)
        q = s.hub.subscribe(p)

        async def gen():
            try:
                yield f"event: hello\ndata: {json.dumps({'role': p.role, 'account_token': p.account_token})}\n\n"
                if p.role != "customer":  # customers only ever see their own payments
                    yield f"event: metrics\ndata: {json.dumps(s.metrics.snapshot())}\n\n"
                yield ": stream open\n\n"  # SSE comment: flushes proxies so the browser knows it's live
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event, data = await asyncio.wait_for(q.get(), 15.0)
                        yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                s.hub.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------------- analyst
    @app.get("/cases")
    async def cases(request: Request, status: str = "open", p: Principal = Depends(require("analyst"))):
        return await asyncio.to_thread(svc(request).store.list_cases, status)

    @app.get("/cases/{payment_token}")
    async def case_detail(payment_token: str, request: Request, p: Principal = Depends(require("analyst"))):
        s = svc(request)
        d = await asyncio.to_thread(s.store.get_payment, payment_token, True)
        if d is None:
            raise HTTPException(404, "no such payment")
        await asyncio.to_thread(s.store.audit.append, p.name, "case_viewed", payment_token, {"decrypted": True})
        return d

    @app.post("/cases/{payment_token}/resolve")
    async def resolve(payment_token: str, body: ResolveIn, request: Request, p: Principal = Depends(require("analyst"))):
        ok = await asyncio.to_thread(svc(request).store.resolve_case, payment_token, body.status, p.name, body.comment)
        if not ok:
            raise HTTPException(404, "no open case for that payment")
        return {"payment_token": payment_token, "status": body.status}

    @app.get("/accounts/watchlist")
    async def watchlist(request: Request, p: Principal = Depends(require("analyst"))):
        s = svc(request)
        return await asyncio.get_running_loop().run_in_executor(s.pool, s.engine.watchlist)

    # ---------------- customer
    @app.get("/my/payments")
    async def my_payments(request: Request, p: Principal = Depends(require("customer"))):
        return {"account_token": p.account_token,
                "payments": await asyncio.to_thread(svc(request).store.customer_payments, p.account_token)}

    # ---------------- auditor
    @app.get("/audit/verify")
    async def audit_verify(request: Request, p: Principal = Depends(require("auditor"))):
        s = svc(request)
        t = time.perf_counter()
        r = await asyncio.to_thread(s.store.audit.verify)
        await asyncio.to_thread(s.store.audit.append, p.name, "audit_verified", None, {"ok": r.ok, "checked": r.checked})
        return {**vars(r), "seconds": round(time.perf_counter() - t, 3)}

    @app.get("/audit/tail")
    async def audit_tail(request: Request, limit: int = 50, p: Principal = Depends(require("auditor"))):
        return await asyncio.to_thread(svc(request).store.audit.tail, min(limit, 500))

    @app.get("/audit/aggregates")
    async def aggregates(request: Request, p: Principal = Depends(require("auditor", "analyst"))):
        return await asyncio.to_thread(svc(request).store.aggregates)

    if UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

    @app.get("/")
    async def root():
        return RedirectResponse("/ui/")

    return app


def app_factory() -> FastAPI:  # uvicorn --factory scamshield.api.main:app_factory
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return create_app()
