"""The JSON service over the session core (docs/m6/m6a_technical_report.md §4.1; the M6b API in `api.py` is a thin
layer over it, and `worker.py` runs a copy of it in the simulation process).

Every method takes and returns JSON-ready values, and every refusal is a `ServiceError` with an HTTP status. One lock
serializes all simulation in a process, which is the plan's single simulation thread (§8.2): the M5 modules' caches
are not locked, so no second thread may simulate in the same process.

    status()                         whether the registry loaded, and the models a session may seat
    create(opponent, belief, seed)   a human (seat A) against `sp:0` or `om:0`; returns the view and position 0
    submit(sid, turn, code)          resolves turn `turn` with the human's code (idempotent; session.py rules)
    view(sid), history(sid, a, b)    the decision point; resolved positions a..b and the turns between them
    stop(sid)                        the stop summary, the recording and the whole history (beliefs now visible)
    record(sid)                      the recording (seeds and hidden beliefs withheld while the session is active)
    simulate(a, b, turns)            a bounded bot game (Watch), generated in full, as a game bundle
    replays(), replay(name)          the verified seed-0 research recordings as game bundles (M6b; M6c builds the
                                     full bundle)

Sessions live in memory. A session idle for `idle_seconds` expires (`expire`, called by the API's sweeper), and the
least recently active one is dropped beyond `max_sessions`.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Dict, Optional

from . import session as S
from .registry import Registry

MODELS = ("sp:0", "om:0")
BELIEF_MODES = S.BELIEF_MODES
MAX_SIM_TURNS = S.TURN_CAP
IDLE_SECONDS = 30 * 60                  # plan §8.2 (proposed: 30 minutes)
MAX_SESSIONS = 256
TRACES = os.path.join("results", "m5", "traces")
RESEARCH_REPLAYS = OrderedDict([
    ("h2h_om_s0_vs_sp_s0", "Pure self-play (A) against opponent modelling (B): the seed-0 head to head"),
    ("om_s0_vs_g_shove", "The scripted g_shove (A) against opponent modelling (B)"),
    ("sp_s0_vs_g_shove", "Pure self-play (A) against the scripted g_shove (B)"),
])


class ServiceError(Exception):
    def __init__(self, status: int, kind: str, message: str):
        super().__init__(message)
        self.status, self.kind, self.message = status, kind, message

    def __reduce__(self):                                   # picklable across the worker process boundary
        return (ServiceError, (self.status, self.kind, self.message))

    def public(self) -> dict:
        return {"error": self.kind, "message": self.message}


_STATUS = {S.IllegalAction: (422, "illegal_action"), S.TurnConflict: (409, "turn_conflict"),
           S.TurnOutOfOrder: (409, "turn_out_of_order"), S.SessionStopped: (410, "session_stopped"),
           S.HiddenInformation: (403, "hidden_information")}


def _seat(model_id: str) -> S.Seat:
    if model_id not in MODELS:
        raise ServiceError(400, "bad_request", f"choose a model from {list(MODELS)}")
    return S.Seat.model(model_id)


def _int(x, name: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise ServiceError(400, "bad_request", f"{name} must be an integer")
    return x


class DemoService:
    def __init__(self, registry: Optional[Registry] = None, max_sessions: int = MAX_SESSIONS,
                 idle_seconds: float = IDLE_SECONDS, traces: str = TRACES, load: bool = True):
        """`registry`: a loaded registry to use; otherwise the service loads one (`load`), or stays without an
        engine (`load=False`: the frontend serves, Play and Watch report the engine unavailable)."""
        self._lock = threading.Lock()
        self._sessions: "OrderedDict[str, S.GameSession]" = OrderedDict()
        self.max_sessions, self.idle_seconds, self.traces = int(max_sessions), float(idle_seconds), traces
        self.registry, self.error = registry, None
        self._replays: Optional[Dict[str, dict]] = None
        if self.registry is None and not load:
            self.error = "the engine was not loaded"
        elif self.registry is None:
            try:
                r = Registry()
                if not r.available():
                    raise RuntimeError("the 7 checkpoint files of lanegame/demo/manifest.json are not on disk")
                self.registry = r.load()
            except Exception as e:                       # the frontend still loads; Play and Watch say so
                self.registry, self.error = None, f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ helpers
    def _ready(self) -> Registry:
        if self.registry is None:
            raise ServiceError(503, "engine_unavailable", self.error or "the engine is not loaded")
        return self.registry

    def _get(self, sid: str) -> S.GameSession:
        s = self._sessions.get(sid)
        if s is None:
            raise ServiceError(404, "unknown_session", "no such session: it expired, the server restarted, or the "
                                                        "id is wrong")
        s.last_active = time.time()
        self._sessions.move_to_end(sid)
        return s

    def _call(self, fn):
        with self._lock:
            try:
                return fn()
            except ServiceError:
                raise
            except S.SessionError as e:
                status, kind = _STATUS.get(type(e), (400, "refused"))
                raise ServiceError(status, kind, str(e)) from e

    def _expire_locked(self, now: float) -> int:
        old = [sid for sid, s in self._sessions.items() if now - s.last_active > self.idle_seconds]
        for sid in old:
            del self._sessions[sid]
        return len(old)

    # ------------------------------------------------------------------ lifecycle
    def count(self) -> int:
        return len(self._sessions)

    def expire(self, now: Optional[float] = None) -> int:
        """Drop every session idle for longer than `idle_seconds`; returns how many."""
        with self._lock:
            return self._expire_locked(time.time() if now is None else now)

    # ------------------------------------------------------------------ requests
    def status(self) -> dict:
        if self.registry is None:
            return {"ready": False, "error": self.error, "models": []}
        return {"ready": True, "models": self.registry.public(), "turn_cap": S.TURN_CAP,
                "lane_exit_position": S.LANE_EXIT, "read_position": S.READ_TURN}

    def create(self, opponent: str = "sp:0", belief: str = "hidden", seed=None) -> dict:
        reg = self._ready()
        if belief not in BELIEF_MODES:
            raise ServiceError(400, "bad_request", f"belief must be one of {list(BELIEF_MODES)}")
        seat = _seat(opponent)
        chance = None if seed is None else _int(seed, "seed")
        if chance is not None and not 0 <= chance < 2**53:
            raise ServiceError(400, "bad_request", "a chosen seed is an integer in 0..2^53")

        def go():
            self._expire_locked(time.time())
            s = S.GameSession(reg, (S.Seat.human(), seat), chance_seed=chance, belief_mode=belief)
            self._sessions[s.id] = s
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return {"session": s.id, "view": s.view(), "history": s.history()}
        return self._call(go)

    def view(self, sid: str) -> dict:
        return self._call(lambda: self._get(sid).view())

    def submit(self, sid: str, turn, code) -> dict:
        return self._call(lambda: self._get(sid).submit(turn, code))

    def history(self, sid: str, p_from=0, p_to=None) -> dict:
        a = _int(p_from, "from")
        b = None if p_to is None else _int(p_to, "to")
        try:
            return self._call(lambda: self._get(sid).history(a, b))
        except ValueError as e:
            raise ServiceError(400, "bad_request", str(e)) from e

    def stop(self, sid: str) -> dict:
        def go():
            s = self._get(sid)
            summary = s.stop("visitor")
            return {"summary": summary, "recording": s.recording(), "history": s.history(), "view": s.view()}
        return self._call(go)

    def record(self, sid: str) -> dict:
        return self._call(lambda: self._get(sid).recording())

    def simulate(self, a: str = "om:0", b: str = "sp:0", turns=MAX_SIM_TURNS) -> dict:
        """One bounded bot game as a game bundle (session.GameSession.bundle). In the API this runs in the worker
        process (worker.py); here it takes the process's simulation lock."""
        reg = self._ready()
        sa, sb, n = _seat(a), _seat(b), _int(turns, "turns")
        if not 1 <= n <= MAX_SIM_TURNS:
            raise ServiceError(400, "bad_request", f"turns must be in 1..{MAX_SIM_TURNS}")

        def go():
            s = S.simulate(reg, sa, sb, turns=n)
            label = f"{reg.card(a).label} (A) against {reg.card(b).label} (B): a new simulation"
            return s.bundle("new_simulation", label)
        return self._call(go)

    # ------------------------------------------------------------------ research replays
    def _build_replays(self) -> Dict[str, dict]:
        reg = self._ready()
        out: Dict[str, dict] = {}
        for name, label in RESEARCH_REPLAYS.items():
            path = os.path.join(self.traces, f"{name}.json")
            if not os.path.isfile(path):
                continue
            s, cmp = S.from_research(reg, path)                 # raises ReplayMismatch if it does not reproduce
            b = s.bundle(name, label)
            b["verification"] = cmp
            out[name] = b
        return out

    def replays(self) -> list:
        """The index of the verified research recordings this process serves (built once, under the lock, by
        re-simulating each through the session core; a recording that does not reproduce is a startup failure)."""
        with self._lock:
            if self._replays is None:
                self._replays = self._build_replays()
            return [{"name": n, "label": b["label"], "kind": b["meta"]["provenance"]["kind"],
                     "turns": len(b["turns"]), "seats": b["meta"]["seats"]} for n, b in self._replays.items()]

    def replay(self, name: str) -> dict:
        self.replays()
        b = (self._replays or {}).get(name)
        if b is None:
            raise ServiceError(404, "unknown_replay", f"no research replay named {name!r}")
        return b


__all__ = ["DemoService", "ServiceError", "MODELS", "RESEARCH_REPLAYS", "IDLE_SECONDS", "MAX_SESSIONS"]
