"""A small JSON service over the session core, for the local demo (docs/m6/m6a_technical_report.md §4.1).

Every method takes and returns JSON-ready values, and every refusal is a `ServiceError` with an HTTP status, so any
web layer can wrap it: the local prototype server (`prototypes/m6a/serve.py`, standard library only) today, and M6b's
FastAPI app later. One lock serializes all simulation, which is the plan's single simulation thread (§8.2): the M5
modules' caches are not locked.

    status()                         whether the registry loaded, and the models a session may seat
    create(opponent, belief, seed)   a human (seat A) against `sp:0` or `om:0`; returns the view and position 0
    submit(sid, turn, code)          resolves turn `turn` with the human's code (idempotent; session.py rules)
    view(sid), history(sid, a, b)    the decision point; resolved positions a..b and the turns between them
    stop(sid)                        the stop summary, the recording and the whole history (beliefs now visible)
    simulate(a, b, turns)            a bounded bot game (Watch), generated in full: the shape of a frame file

Sessions live in memory. The oldest is dropped beyond `max_sessions`; this is a local demo, not the hosted server.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

from . import session as S
from .registry import Registry

MODELS = ("sp:0", "om:0")
BELIEF_MODES = S.BELIEF_MODES
MAX_SIM_TURNS = S.TURN_CAP


class ServiceError(Exception):
    def __init__(self, status: int, kind: str, message: str):
        super().__init__(message)
        self.status, self.kind, self.message = status, kind, message

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
    def __init__(self, registry: Optional[Registry] = None, max_sessions: int = 64):
        self._lock = threading.Lock()
        self._sessions: "OrderedDict[str, S.GameSession]" = OrderedDict()
        self.max_sessions = int(max_sessions)
        self.registry, self.error = registry, None
        if self.registry is None:
            try:
                r = Registry()
                if not r.available():
                    raise RuntimeError("the 7 checkpoint files of lanegame/demo/manifest.json are not on disk")
                self.registry = r.load()
            except Exception as e:                       # the static pages still work without the engine
                self.registry, self.error = None, f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ helpers
    def _ready(self) -> Registry:
        if self.registry is None:
            raise ServiceError(503, "engine_unavailable", self.error or "the engine is not loaded")
        return self.registry

    def _get(self, sid: str) -> S.GameSession:
        s = self._sessions.get(sid)
        if s is None:
            raise ServiceError(404, "unknown_session", "no such session (it may have been dropped)")
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

        def go():
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

    def simulate(self, a: str = "om:0", b: str = "sp:0", turns=MAX_SIM_TURNS) -> dict:
        reg = self._ready()
        sa, sb, n = _seat(a), _seat(b), _int(turns, "turns")
        if not 1 <= n <= MAX_SIM_TURNS:
            raise ServiceError(400, "bad_request", f"turns must be in 1..{MAX_SIM_TURNS}")

        def go():
            s = S.simulate(reg, sa, sb, turns=n)
            rec, h = s.recording(), s.history()
            meta = {k: rec[k] for k in ("provenance", "config", "seeds", "policy_rule", "seats", "belief_mode",
                                        "turn_cap", "status", "lead_A", "marks")}
            return {"name": "new simulation", "meta": meta, "positions": h["positions"], "turns": h["turns"],
                    "beliefs": rec.get("beliefs"), "lane_exit_position": S.LANE_EXIT, "read_position": S.READ_TURN}
        return self._call(go)


__all__ = ["DemoService", "ServiceError", "MODELS"]
