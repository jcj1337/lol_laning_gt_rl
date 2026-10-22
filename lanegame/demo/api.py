"""The demo's HTTP API (M6b): FastAPI over the session core, and the built frontend (docs/m6/m6_plan.md §4, §8.2;
docs/m6/m6a_technical_report.md §4.1 for the contract it implements).

    create_app(...)   the application; `python -m lanegame.demo` serves it with uvicorn on 127.0.0.1:8765

Routes (every response shape is a model of `schemas.py`, published at /openapi.json; `web/openapi.json` is a copy):

    GET  /api/status                       whether the engine loaded, the models, the constants, the worker's state
    GET  /api/models                       the public model cards
    POST /api/sessions                     {opponent, belief, seed?} -> {session, view, history}
    GET  /api/sessions/{id}                the view (the current decision point)
    POST /api/sessions/{id}/turns          {turn, code} -> {turn, position, belief?, view}; idempotent per turn
    GET  /api/sessions/{id}/history        ?from=&to=  resolved positions only, re-simulated from the codes
    POST /api/sessions/{id}/stop           {summary, recording, history, view}; no winner
    GET  /api/sessions/{id}/record         the recording (seeds and hidden beliefs withheld while active)
    POST /api/simulations                  {a, b, turns?} -> a game bundle (Watch), generated in the worker process
    GET  /api/replays, /api/replays/{name} the verified seed-0 research recordings as game bundles
    GET  /{anything else}                  the built frontend (web/dist), with the single-page fallback

Refusals are JSON {error, message} with the status the service assigns: 400 for a malformed request, 403 for an
unresolved position, 404 for an unknown session or replay, 409 for a conflict or a turn out of order, 410 after the
stop, 413 for a large body, 422 for an illegal action (the message is the menu's reason), 429 over a rate limit, 503
while the engine is unavailable.

Concurrency (plan §8.2, §3 rule 3). Human turns, histories and research replays run on the serving process's one
simulation lock. Bounded bot games run in `worker.py`'s separate process, which has its own registry and caches, so
nothing mutable is shared; at most `pending` simulations wait for it, and the rest are refused with 429. Sessions
expire after `idle_seconds` and are capped at `max_sessions`. Request bodies are limited to 16 KB, and session
creation, turn submission and simulations are rate-limited per client address.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import schemas as M
from . import worker as W
from .service import IDLE_SECONDS, MAX_SESSIONS, DemoService, ServiceError

log = logging.getLogger("lanegame.demo")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB_DIST = os.path.join(REPO, "web", "dist")
SESSION_ID = re.compile(r"[0-9a-f]{32}")
MAX_BODY = 16_384
DEFAULT_LIMITS = {"sessions": (30 / 60, 10), "turns": (20.0, 60), "simulations": (10 / 60, 5)}   # (per second, burst)
NO_FRONTEND = """<!doctype html><meta charset="utf-8"><title>Laning simulator</title>
<body style="font: 15px/1.5 system-ui, sans-serif; max-width: 640px; margin: 48px auto; padding: 0 16px">
<h1 style="font-size: 20px">The API is running, but the frontend is not built</h1>
<p>Build it once, then reload this page:</p>
<pre style="background: #f2f2f0; padding: 12px">cd web
npm install
npm run build</pre>
<p>Or run the frontend's dev server (<code>npm run dev</code> in <code>web/</code>), which proxies <code>/api</code> here.
The API schema is at <a href="/openapi.json">/openapi.json</a> and <a href="/docs">/docs</a>.</p>"""


class RateLimiter:
    """A token bucket per key: `rate` tokens a second up to `burst`."""

    def __init__(self, rate: float, burst: int):
        self.rate, self.burst = float(rate), float(burst)
        self._b: Dict[str, list] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._b.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self._b[key] = [tokens, now]
                return False
            self._b[key] = [tokens - 1.0, now]
            if len(self._b) > 10_000:                       # forget the oldest clients
                for k in list(self._b)[:5_000]:
                    del self._b[k]
            return True


class SimulationPool:
    """Bounded bot games in a separate process (worker.py). `state` is "starting" until the worker's registry has
    loaded, then "ready", or "unavailable" if it could not."""

    def __init__(self, workers: int = 1, pending: int = 4, timeout: float = 120.0, root: str = REPO):
        import multiprocessing
        self.state, self.error, self.timeout = "starting", None, float(timeout)
        self._sem = threading.BoundedSemaphore(int(pending))
        self._pool = ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context("spawn"),
                                         initializer=W.init_worker, initargs=(root,))

    def start(self) -> None:
        def done(fut):
            try:
                r = fut.result()
                self.state, self.error = ("ready", None) if r["ready"] else ("unavailable", r["error"])
            except Exception as e:                           # pragma: no cover (the process could not start)
                self.state, self.error = "unavailable", f"{type(e).__name__}: {e}"
            log.info("simulation worker: %s%s", self.state, f" ({self.error})" if self.error else "")
        self._pool.submit(W.ready).add_done_callback(done)

    def run(self, a: str, b: str, turns: int) -> dict:
        if self.state == "unavailable":
            raise ServiceError(503, "engine_unavailable", self.error or "the simulation worker is unavailable")
        if not self._sem.acquire(blocking=False):
            raise ServiceError(429, "busy", "too many games are being generated at once; try again in a moment")
        try:
            try:
                return self._pool.submit(W.simulate, a, b, turns).result(timeout=self.timeout)
            except ServiceError:
                raise
            except Exception as e:
                self.state, self.error = "unavailable", f"{type(e).__name__}: {e}"
                raise ServiceError(503, "engine_unavailable", f"the simulation worker failed: {e}") from e
        finally:
            self._sem.release()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def create_app(service: Optional[DemoService] = None, *, sim_workers: int = 1, static_dir: Optional[str] = WEB_DIST,
               idle_seconds: float = IDLE_SECONDS, max_sessions: int = MAX_SESSIONS, limits: Optional[dict] = None,
               replays: bool = True, sweep_seconds: float = 30.0) -> FastAPI:
    """The application. `service` is loaded at startup when not given (so the OpenAPI schema needs no engine);
    `sim_workers=0` generates bot games in this process under the simulation lock (tests, `--no-worker`); `limits`
    is {name: (rate per second, burst)} or {} for none."""
    limits = DEFAULT_LIMITS if limits is None else limits
    limiters = {k: RateLimiter(*v) for k, v in limits.items()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = app.state.service
        if svc is None:
            svc = app.state.service = await asyncio.to_thread(DemoService, None, max_sessions, idle_seconds)
        st = svc.status()
        log.info("engine: %s", "ready" if st["ready"] else f"unavailable ({st['error']})")
        if st["ready"] and replays:
            n = len(await asyncio.to_thread(svc.replays))
            log.info("research replays: %d verified", n)
        pool = None
        if st["ready"] and sim_workers > 0:
            pool = SimulationPool(sim_workers)
            pool.start()
        app.state.pool = pool
        app.state.in_process = sim_workers <= 0

        async def sweep():
            while True:
                await asyncio.sleep(sweep_seconds)
                n = await asyncio.to_thread(svc.expire)
                if n:
                    log.info("expired %d idle session(s)", n)
        task = asyncio.create_task(sweep())
        try:
            yield
        finally:
            task.cancel()
            if pool is not None:
                pool.close()

    app = FastAPI(title="Laning simulator demo API", version="1.0.0", lifespan=lifespan,
                  description="The M6 demo: Play against a trained M5 agent, Watch two agents, replay research games.")
    app.state.service, app.state.pool, app.state.in_process = service, None, sim_workers <= 0

    # ------------------------------------------------------------------ errors, limits
    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError):
        return JSONResponse(exc.public(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(x) for x in first.get("loc", ()) if x != "body")
        return JSONResponse({"error": "bad_request", "message": f"{where}: {first.get('msg', 'invalid')}"},
                            status_code=400)

    @app.middleware("http")
    async def _limit_body(request: Request, call_next):
        n = request.headers.get("content-length")
        if n and n.isdigit() and int(n) > MAX_BODY:
            return JSONResponse({"error": "too_large", "message": f"the request body is over {MAX_BODY} bytes"},
                                status_code=413)
        return await call_next(request)

    def svc(request: Request) -> DemoService:
        s = request.app.state.service
        if s is None:
            raise ServiceError(503, "engine_unavailable", "the server is starting")
        return s

    def limit(request: Request, name: str) -> None:
        lim = limiters.get(name)
        if lim is not None and not lim.allow(request.client.host if request.client else "?"):
            raise ServiceError(429, "rate_limited", "too many requests from this address; slow down a little")

    def sid_of(sid: str) -> str:
        if not SESSION_ID.fullmatch(sid):
            raise ServiceError(404, "unknown_session", "not a session id")
        return sid

    # ------------------------------------------------------------------ the API
    @app.get("/api/status", response_model=M.Status)
    def status(request: Request):
        s = request.app.state.service
        d = dict(s.status()) if s is not None else {"ready": False, "error": "the server is starting", "models": []}
        pool = request.app.state.pool
        d["simulations"] = ("in_process" if request.app.state.in_process else pool.state if pool else "unavailable")
        if d["ready"] and s._replays is not None:
            d["replays"] = len(s._replays)
        d["sessions"] = s.count() if s is not None else 0
        return d

    @app.get("/api/models", response_model=list[M.ModelCard])
    def models(request: Request):
        return svc(request).status()["models"]

    @app.post("/api/sessions", response_model=M.CreateResponse, responses={503: {"model": M.ErrorBody}})
    def create(req: M.CreateRequest, request: Request):
        limit(request, "sessions")
        return svc(request).create(req.opponent, req.belief, req.seed)

    @app.get("/api/sessions/{sid}", response_model=M.View, responses={404: {"model": M.ErrorBody}})
    def view(sid: str, request: Request):
        return svc(request).view(sid_of(sid))

    @app.post("/api/sessions/{sid}/turns", response_model=M.TurnResult,
              responses={409: {"model": M.ErrorBody}, 410: {"model": M.ErrorBody}, 422: {"model": M.ErrorBody}})
    def submit(sid: str, req: M.SubmitRequest, request: Request):
        limit(request, "turns")
        return svc(request).submit(sid_of(sid), req.turn, req.code)

    @app.get("/api/sessions/{sid}/history", response_model=M.History, responses={403: {"model": M.ErrorBody}})
    def history(sid: str, request: Request, from_: int = Query(0, alias="from", ge=0),
                to: Optional[int] = Query(None, ge=0)):
        return svc(request).history(sid_of(sid), from_, to)

    @app.post("/api/sessions/{sid}/stop", response_model=M.StopResponse)
    def stop(sid: str, request: Request):
        return svc(request).stop(sid_of(sid))

    @app.get("/api/sessions/{sid}/record", response_model=M.Recording)
    def record(sid: str, request: Request):
        return svc(request).record(sid_of(sid))

    @app.post("/api/simulations", response_model=M.GameBundle, responses={429: {"model": M.ErrorBody}})
    def simulate(req: M.SimulateRequest, request: Request):
        limit(request, "simulations")
        s = svc(request)
        pool = request.app.state.pool
        if pool is None:
            if not request.app.state.in_process:
                raise ServiceError(503, "engine_unavailable", s.error or "the simulation worker is not running")
            return s.simulate(req.a, req.b, req.turns)
        s._ready()
        return pool.run(req.a, req.b, req.turns)

    @app.get("/api/replays", response_model=list[M.ReplayEntry])
    def replays_index(request: Request):
        return svc(request).replays()

    @app.get("/api/replays/{name}", response_model=M.GameBundle, responses={404: {"model": M.ErrorBody}})
    def replay(name: str, request: Request):
        return svc(request).replay(name)

    # ------------------------------------------------------------------ the frontend
    index = os.path.join(static_dir, "index.html") if static_dir else None
    if index and os.path.isfile(index):
        assets = os.path.join(static_dir, "assets")
        if os.path.isdir(assets):
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str):
            if path == "api" or path.startswith("api/"):
                return JSONResponse({"error": "not_found", "message": f"no API route /{path}"}, status_code=404)
            full = os.path.normpath(os.path.join(static_dir, path)) if path else index
            if path and full.startswith(os.path.normpath(static_dir) + os.sep) and os.path.isfile(full):
                return FileResponse(full)
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
    else:
        @app.get("/", include_in_schema=False)
        def no_frontend():
            return HTMLResponse(NO_FRONTEND)

    return app


__all__ = ["create_app", "SimulationPool", "RateLimiter", "WEB_DIST", "MAX_BODY", "DEFAULT_LIMITS"]
