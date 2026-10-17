"""ClaimSense API.

POST /claims -> pseudonymise -> micro-batch -> features + both models + fusion -> encrypt + store + audit
            -> SSE event -> queue a Granite note if not PAY -> respond with decision, reasons, latency.

Claims that arrive within BATCH_WINDOW_MS of each other are scored together: one autoencoder call and
one XGBoost call per batch. Large batches go to the Telum II NNPA, small ones to the CPU, following the
measured crossover (docs/evidence/bench/nnpa_smoke.txt).
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
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import explain
from ..data import CHRONIC_COLS, load_beneficiaries
from ..features import PROVIDER_FEATURES, _present
from ..fusion import HOLD, PAY, RETURN
from ..inference import load_anomaly_scorer, load_provider_scorer
from ..scoring import ClaimInput, Engine
from ..security import Pseudonymizer, load_keyring, read_secret
from ..store import ClaimRecord, Store
from .auth import Authenticator, Principal, require
from .schemas import ClaimIn, ClaimOut, LoginIn, ResolveIn
from .settings import Settings

log = logging.getLogger("claimsense.api")
UI_DIR = Path(__file__).resolve().parent.parent / "ui"
BENE_FIELDS = ["DOB", "DOD", "RenalDiseaseIndicator", "NoOfMonths_PartACov", "NoOfMonths_PartBCov", *CHRONIC_COLS]


# ---------------------------------------------------------------------------------------- live metrics
class Metrics:
    def __init__(self, window: int = 5000):
        self.recent: collections.deque = collections.deque(maxlen=window)  # (t, latency_ms, model_ms)
        self.counts = collections.Counter()
        self.amounts = collections.Counter()
        self.batch_sizes: collections.deque = collections.deque(maxlen=500)
        self.started = time.time()

    def add(self, latency_ms: float, model_ms: float, action: str, amount: float) -> None:
        self.recent.append((time.time(), latency_ms, model_ms))
        self.counts[action] += 1
        self.amounts[action] += amount

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        lat = np.array([r[1] for r in self.recent]) if self.recent else np.array([0.0])
        mdl = np.array([r[2] for r in self.recent]) if self.recent else np.array([0.0])
        last10 = sum(1 for r in self.recent if now - r[0] <= 10.0)
        total = sum(self.counts.values())
        return {
            "claims_total": total,
            "claims_per_sec": round(last10 / 10.0, 1),
            "latency_p50_ms": round(float(np.percentile(lat, 50)), 2),
            "latency_p99_ms": round(float(np.percentile(lat, 99)), 2),
            "model_p50_ms": round(float(np.percentile(mdl, 50)), 3),
            "decisions": {a: self.counts[a] for a in (PAY, RETURN, HOLD)},
            "amount_held": round(self.amounts[HOLD], 2),
            "amount_returned": round(self.amounts[RETURN], 2),
            "share_scored": 1.0 if total else 0.0,
            "avg_batch_size": round(float(np.mean(self.batch_sizes)), 1) if self.batch_sizes else 0.0,
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

    def publish(self, event: str, data: dict[str, Any], provider_token: str | None = None) -> None:
        for q, p in list(self.subs.items()):
            if p.role == "provider" and provider_token != p.provider_token:
                continue  # providers only ever see their own claims
            if p.role == "provider" and event == "metrics":
                continue
            if p.role == "auditor" and event != "metrics":
                continue  # auditors get aggregates only
            try:
                q.put_nowait((event, data))
            except asyncio.QueueFull:
                pass  # a slow browser must never slow down scoring


# ---------------------------------------------------------------------------------------- service
class Service:
    def __init__(self, s: Settings):
        self.s = s
        sec = s.secrets_dir
        self.pseudo = Pseudonymizer(bytes.fromhex(read_secret("pseudo_key", sec)))
        self.store = Store(s.db_path, load_keyring(sec), s.checkpoint_path)
        users = json.loads(read_secret("users", sec))
        self.auth = Authenticator(read_secret("jwt_secret", sec), users, read_secret("service_token", sec),
                                  s.jwt_ttl_minutes, lambda pid: self.pseudo.token("provider", pid))
        self.anomaly = load_anomaly_scorer(s.models_dir, s.anomaly_backend, s.nnpa_min_batch)
        self.provider = load_provider_scorer(s.models_dir)
        self.engine = Engine.from_artifacts(s.models_dir, s.policy_file, self.anomaly, self.provider)
        t = time.time()
        b = load_beneficiaries(str(s.raw_dir))
        for c in ("DOB", "DOD"):
            b[c] = b[c].dt.strftime("%Y-%m-%d")
        b = b.astype(object).where(b.notna(), None)
        self.registry: dict[str, dict] = {r["BeneID"]: {k: r[k] for k in BENE_FIELDS}
                                          for r in b[["BeneID", *BENE_FIELDS]].to_dict("records")}
        log.info("beneficiary registry: %d records in %.1fs", len(self.registry), time.time() - t)
        self._restore_state()
        self.metrics = Metrics()
        self.hub = Hub()
        self.queue: asyncio.Queue | None = None
        self.inflight: set[str] = set()
        # every touch of the engine (stateful) happens on this one thread
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")
        self.tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------ batching
    async def submit(self, claim: ClaimIn) -> dict[str, Any]:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((claim, fut, time.perf_counter()))
        return await fut

    async def batch_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            batch = [first]
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
            self.metrics.batch_sizes.append(len(batch))
            for (claim, fut, t0), out in zip(batch, outs):
                out["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
                out["batch_size"] = len(batch)
                self.metrics.add(out["latency_ms"], out["model_ms"], out["action"], claim.InscClaimAmtReimbursed)
                self.hub.publish("claim", {
                    "claim_token": out["claim_token"], "provider_token": out["provider_token"],
                    "amount": claim.InscClaimAmtReimbursed, "action": out["action"],
                    "latency_ms": out["latency_ms"], "is_inpatient": claim.is_inpatient,
                    "service_date": claim.ClaimStartDt.isoformat(), "rules": out["reasons"].get("rules", []),
                }, provider_token=out["provider_token"])
                if not fut.done():
                    fut.set_result(out)

    def _input(self, raw: dict[str, Any]) -> ClaimInput:
        att = raw.get("AttendingPhysician")
        return ClaimInput(raw, self.registry[raw["BeneID"]], self.pseudo.token("provider", raw["Provider"]),
                          self.pseudo.token("bene", raw["BeneID"]),
                          self.pseudo.token("physician", att) if _present(att) else None)

    def _restore_state(self) -> None:
        """Provider histories and the duplicate index live in memory; rebuild them from stored claims."""
        t, n, batch = time.time(), 0, []
        for raw in self.store.iter_payloads():
            if raw.get("BeneID") in self.registry:
                batch.append(self._input(raw))
                n += 1
            if len(batch) >= 5000:
                self.engine.replay(batch)
                batch = []
        self.engine.replay(batch)
        if n:
            log.info("restored rolling state from %d stored claims (%d providers) in %.1fs", n,
                     len(self.engine.providers), time.time() - t)

    def _process(self, claims: list[ClaimIn]) -> list[dict[str, Any]]:
        items, metas = [], []
        for c in claims:
            raw = c.model_dump(mode="json")
            it = self._input(raw)
            items.append(it)
            metas.append((raw, it.provider_key, it.bene_key, self.pseudo.token("claim", c.ClaimID)))
        results = self.engine.score_batch(items)
        backend = self.anomaly.name
        if backend == "torch_nnpa" and len(items) < self.s.nnpa_min_batch:
            backend = "torch_nnpa(cpu-path)"
        records, outs = [], []
        for (raw, ptok, btok, ctok), res in zip(metas, results):
            d = res.decision
            notes = []
            if d.action == RETURN:
                facts = {**res.reasons, "amount": raw["InscClaimAmtReimbursed"]}
                notes.append(("correction", explain.correction_note_template(res.reasons), facts))
            elif d.action == HOLD:
                facts = {**res.reasons, "amount": raw["InscClaimAmtReimbursed"], "provider_token": ptok,
                         "provider_risk": d.provider_risk}
                notes.append(("brief", explain.investigator_brief_template(res.reasons, ptok, d.provider_risk), facts))
            records.append(ClaimRecord(
                ctok, ptok, btok, raw["ClaimStartDt"], raw["InscClaimAmtReimbursed"], raw["is_inpatient"],
                d.action, d.anomaly, d.provider_risk, res.provider_claims, d.hard_errors, res.reasons,
                res.model_ms, backend, raw, notes))
            outs.append({"claim_token": ctok, "provider_token": ptok, "action": d.action, "reasons": res.reasons,
                         "anomaly": round(d.anomaly, 4),
                         "provider_risk": None if d.provider_risk is None else round(d.provider_risk, 4),
                         "model_ms": round(res.model_ms, 4), "backend": backend})
        self.store.save_batch(records)
        return outs

    # ------------------------------------------------------------------ background tasks
    async def notes_loop(self) -> None:
        """Push Granite notes to browsers as the worker finishes them."""
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
                payload = {k: r[k] for k in ("claim_token", "kind", "status", "source", "text")}
                if r["kind"] == "correction":
                    self.hub.publish("note", payload, provider_token=r["provider_token"])
                else:  # briefs are for investigators only
                    for q, p in list(self.hub.subs.items()):
                        if p.role == "investigator" and not q.full():
                            q.put_nowait(("note", payload))

    async def metrics_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            self.hub.publish("metrics", self.metrics.snapshot())

    def leaderboard(self, top: int = 15) -> list[dict[str, Any]]:
        rows = self.engine.provider_snapshot()[:top]
        for r in rows:
            f = self.engine.providers[r["provider"]].features()
            x = np.array([f[k] for k in PROVIDER_FEATURES], dtype=np.float32)
            r["reasons"] = explain.provider_reasons(f, PROVIDER_FEATURES, self.provider.contributions(x),
                                                    self.provider.peer_quantiles)
        return rows


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
            "anomaly_backend": svc.anomaly.name, "provider_backend": svc.provider.name,
            "policy": vars(svc.engine.policy)})
        log.info("ready: anomaly=%s provider=%s policy=%s", svc.anomaly.name, svc.provider.name, svc.engine.policy)
        yield
        for t in svc.tasks:
            t.cancel()
        svc.store.audit.checkpoint()
        svc.pool.shutdown(wait=False)

    app = FastAPI(title="ClaimSense", version="0.1.0", lifespan=lifespan)

    def svc(request: Request) -> Service:
        return request.app.state.svc

    @app.get("/health")
    async def health(request: Request):
        s = svc(request)
        return {"status": "ok", "anomaly_backend": s.anomaly.name, "provider_backend": s.provider.name,
                "registry": len(s.registry), "providers_tracked": len(s.engine.providers)}

    @app.post("/auth/login")
    async def login(body: LoginIn, request: Request):
        s = svc(request)
        try:
            token, p = s.auth.login(body.username, body.password)
        except HTTPException:
            await asyncio.to_thread(s.store.audit.append, body.username[:64], "login_failed", None, {})
            raise
        await asyncio.to_thread(s.store.audit.append, p.name, "login", None, {"role": p.role})
        return {"access_token": token, "token_type": "bearer", "role": p.role, "provider_token": p.provider_token}

    @app.post("/claims", response_model=ClaimOut)
    async def post_claim(claim: ClaimIn, request: Request, p: Principal = Depends(require("service"))):
        s = svc(request)
        if claim.BeneID not in s.registry:
            raise HTTPException(422, "unknown beneficiary: not enrolled")
        ctok = s.pseudo.token("claim", claim.ClaimID)
        if ctok in s.inflight or await asyncio.to_thread(s.store.get_claim, ctok) is not None:
            raise HTTPException(409, "claim already submitted")
        s.inflight.add(ctok)
        try:
            return await s.submit(claim)
        finally:
            s.inflight.discard(ctok)

    @app.get("/metrics")
    async def metrics(request: Request, p: Principal = Depends(require("investigator", "auditor"))):
        return svc(request).metrics.snapshot()

    @app.get("/stream")
    async def stream(request: Request, p: Principal = Depends(require("investigator", "auditor", "provider"))):
        s = svc(request)
        q = s.hub.subscribe(p)

        async def gen():
            try:
                yield f"event: hello\ndata: {json.dumps({'role': p.role, 'provider_token': p.provider_token})}\n\n"
                if p.role != "provider":  # providers only ever see their own claims
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

    # ---------------- investigator
    @app.get("/cases")
    async def cases(request: Request, status: str = "open", p: Principal = Depends(require("investigator"))):
        return await asyncio.to_thread(svc(request).store.list_cases, status)

    @app.get("/cases/{claim_token}")
    async def case_detail(claim_token: str, request: Request, p: Principal = Depends(require("investigator"))):
        s = svc(request)
        d = await asyncio.to_thread(s.store.get_claim, claim_token, True)
        if d is None:
            raise HTTPException(404, "no such claim")
        await asyncio.to_thread(s.store.audit.append, p.name, "case_viewed", claim_token, {"decrypted": True})
        return d

    @app.post("/cases/{claim_token}/resolve")
    async def resolve(claim_token: str, body: ResolveIn, request: Request,
                      p: Principal = Depends(require("investigator"))):
        ok = await asyncio.to_thread(svc(request).store.resolve_case, claim_token, body.status, p.name, body.comment)
        if not ok:
            raise HTTPException(404, "no open case for that claim")
        return {"claim_token": claim_token, "status": body.status}

    @app.get("/providers/leaderboard")
    async def leaderboard(request: Request, p: Principal = Depends(require("investigator"))):
        s = svc(request)
        return await asyncio.get_running_loop().run_in_executor(s.pool, s.leaderboard)

    # ---------------- provider
    @app.get("/my/claims")
    async def my_claims(request: Request, p: Principal = Depends(require("provider"))):
        return {"provider_token": p.provider_token,
                "claims": await asyncio.to_thread(svc(request).store.provider_claims, p.provider_token)}

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
    async def aggregates(request: Request, p: Principal = Depends(require("auditor", "investigator"))):
        return await asyncio.to_thread(svc(request).store.aggregates)

    if UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

    @app.get("/")
    async def root():
        return RedirectResponse("/ui/")

    return app


def app_factory() -> FastAPI:  # uvicorn --factory claimsense.api.main:app_factory
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return create_app()
