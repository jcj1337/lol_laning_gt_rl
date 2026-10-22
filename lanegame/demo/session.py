"""The headless game session of the M6 demo (docs/m6/m6_plan.md §3 rules 1 to 5 and 10 to 12, §4.1, §11).

A `GameSession` owns one game: its environment and chance stream, one policy object per bot seat (its own sampling
stream and, for `om`, its own belief), the stored bot codes of the turn being decided, and the record of every resolved
turn. Only immutable objects are shared between sessions: the registry's networks in evaluation mode, the
configuration, the stateless belief and scripted types. The core has no web dependency; the API (M6b) is a thin layer
over it and must run all simulation on one thread, which also serializes the M5 modules' caches.

The turn loop is M5's (`experiments/rl/m5/games.play`, verified in the M6 plan §2.2), on `LaneEnv(m45_calibrated(),
record_ticks=True)` from s0 with k0 = 0:

    1  when a turn begins, every bot seat decides from the pre-turn state (and its history up to the previous turn) and
       the session stores the codes before any view of the turn exists. Nothing the session returns before the turn
       resolves carries those codes, their probabilities or a policy seed
    2  a human submission names its turn number and a code. The code is checked against the legal set of the pre-turn
       state; a rejected code changes nothing (state, turn number, random streams, stored bot codes)
    3  the environment resolves the turn with both codes; then every bot seat's `observe_turn` runs with the PRE-turn
       state, so the belief updates only after both codes are fixed
    4  the next turn begins (step 1), unless the turn cap is reached

Turn numbers. Position p is the state after p turns (p = 0 the initial position); turn t moves position t - 1 to
position t, so the engine's turn index is t - 1. A submission for the current turn resolves it; the same code for a
resolved turn returns the stored result (a duplicate); another code for a resolved turn, or a later turn number, is
refused. A per-session lock serializes submissions.

Seeds (plan §10 defaults). The chance seed is drawn by the server unless the caller chooses one (`chance_chosen` is
then recorded: a known seed lets anyone foresee the chance rolls). Bot seats sample their actions exactly as M5
evaluated them (players.crn_policy, the inverse CDF with the uniforms of default_rng([policy_seed, 7, seat]); scripted
seats the generator default_rng([policy_seed, 11, seat])). A research reproduction uses the chance seed as the policy
seed, M5's convention. Every other session draws a separate policy seed. A drawn chance seed and the policy seed stay
secret until the session stops.

Stopping is not a terminal state of the game and declares no winner: the summary describes the state. Position 94 is
marked as the lane exit (M5's `games.LANE_END`); the earned-gold lead at position 114 (`games.READ_TURN`) is reported as
this one game's lead, which M5 averaged over many games to realize E_20 at the lane exit.

Belief visibility (plan §3 rule 12): "hidden" (default) returns no belief data while the session is active; "shown"
adds the opponent-modelling seat's posterior after every resolved position. After the session stops, the whole belief
history is available in either mode.
"""

from __future__ import annotations

import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import actions as ACT
from . import frames as F
from . import record as R
from .registry import Registry, scripted_agent, scripted_description

HUMAN, MODEL, SCRIPTED = "human", "model", "scripted"
TURN_CAP = 150                 # the demo's limit (plan §10), not a rule of the game
MAX_TURNS = 500                # the longest M5 recording
LANE_EXIT = 94                 # experiments/rl/m5/games.LANE_END
READ_TURN = 114                # experiments/rl/m5/games.READ_TURN
AWAITING, STOPPED = "awaiting_action", "stopped"
STOP_REASONS = ("visitor", "turn_cap", "expired")
BELIEF_MODES = ("hidden", "shown")


class SessionError(Exception):
    """Base of the session's refusals; none changes the session."""


class IllegalAction(SessionError):
    pass


class TurnConflict(SessionError):
    pass


class TurnOutOfOrder(SessionError):
    pass


class SessionStopped(SessionError):
    pass


class HiddenInformation(SessionError):
    pass


@dataclass(frozen=True)
class Seat:
    kind: str                  # HUMAN | MODEL | SCRIPTED
    id: str = ""               # registry model id or scripted archetype name

    @staticmethod
    def human() -> "Seat":
        return Seat(HUMAN)

    @staticmethod
    def model(model_id: str) -> "Seat":
        return Seat(MODEL, model_id)

    @staticmethod
    def scripted(name: str) -> "Seat":
        return Seat(SCRIPTED, name)


class _ScriptedSeat:
    """A scripted archetype through the agent interface, as M5's `games.play` seats it: the obs_last_action
    observation and its own generator."""

    def __init__(self, name: str, policy_seed: int, seat: int):
        from ..experiments.rl.m5 import common as C
        self.agent, self.seat = scripted_agent(name), int(seat)
        self.rng = np.random.default_rng([int(policy_seed), 11, int(seat)])
        self.cfg_om = C.arm_cfg("om")

    def decide(self, lane, m4, k: int, last) -> int:
        from ..experiments.rl.m5 import players as PL
        return int(self.agent.act(PL.observation_of(self.cfg_om, lane, m4, k, last, self.seat), self.rng))


def running_total(xs: Sequence[float]) -> float:
    """Seat A's total reward added turn by turn, as M5's recordings computed it (Python's `sum` compensates rounding
    since 3.12, so it can differ in the last digit)."""
    t = 0.0
    for x in xs:
        t += x
    return t


def _draw_seed() -> int:
    """Below 2**53, so a browser's JSON numbers keep it exact (a larger seed would not survive a record download)."""
    return secrets.randbits(53)


def _seat_letter(s: int) -> str:
    return "AB"[s]


class GameSession:
    def __init__(self, registry: Registry, seats: Sequence[Seat], *, chance_seed: Optional[int] = None,
                 policy_seed: Optional[int] = None, research_convention: bool = False, belief_mode: str = "hidden",
                 turn_cap: int = TURN_CAP, session_id: Optional[str] = None, provenance: Optional[dict] = None,
                 _chance_chosen: Optional[bool] = None):
        from ..env import LaneEnv
        from ..experiments.rl.m5 import common as C
        seats = tuple(seats)
        if len(seats) != 2 or not all(isinstance(x, Seat) for x in seats):
            raise ValueError("a session has two Seat entries (A, B)")
        humans = [s for s, x in enumerate(seats) if x.kind == HUMAN]
        if len(humans) > 1:
            raise ValueError("at most one human seat (two humans would be multiplayer, out of scope)")
        if belief_mode not in BELIEF_MODES:
            raise ValueError(f"belief_mode must be one of {BELIEF_MODES}")
        if not 1 <= int(turn_cap) <= MAX_TURNS:
            raise ValueError(f"turn_cap must be in 1..{MAX_TURNS}")
        if research_convention and (chance_seed is None or policy_seed is not None or humans):
            raise ValueError("the research convention needs a given chance seed, no policy seed and no human seat")
        registry.load()
        self.registry, self.seats, self.turn_cap = registry, seats, int(turn_cap)
        self.human = humans[0] if humans else None
        self.belief_mode = belief_mode
        self.id = session_id or uuid.uuid4().hex
        self.chance_chosen = (chance_seed is not None) if _chance_chosen is None else bool(_chance_chosen)
        self._chance = int(chance_seed) if chance_seed is not None else _draw_seed()
        if research_convention:
            self._policy, self.policy_rule = self._chance, "chance"
        elif policy_seed is not None:
            self._policy, self.policy_rule = int(policy_seed), "given"
        else:
            self._policy, self.policy_rule = _draw_seed(), "drawn"
        kind = HUMAN if humans else "bots"
        self.provenance = provenance or {"kind": "human_game" if kind == HUMAN else "new_simulation",
                                         "note": R.EXAMPLE_NOTE}
        if self.provenance["kind"] not in R.KINDS:
            raise ValueError(f"unknown provenance kind {self.provenance['kind']!r}")
        self.cfg = C.arm_cfg("sp")
        self._bots: Dict[int, object] = {}
        for s, seat in enumerate(seats):
            if seat.kind == MODEL:
                self._bots[s] = registry.policy(seat.id, s, self._policy, self.turn_cap)
            elif seat.kind == SCRIPTED:
                self._bots[s] = _ScriptedSeat(seat.id, self._policy, s)
            elif seat.kind != HUMAN:
                raise ValueError(f"unknown seat kind {seat.kind!r}")
        self._om = {s: p for s, p in self._bots.items() if getattr(p, "has_belief", False)}
        self.env = LaneEnv(self.cfg, max_turns=self.turn_cap, record_ticks=True)
        self.env.reset(seed=self._chance)
        self.state, self.stop_reason = AWAITING, None
        self.turns: List[list] = []                         # [code A, code B, digest] per resolved turn
        self.rewards: List[float] = []
        # the belief's log posterior at every position (row p), one compact array per opponent-modelling seat
        self._lp: Dict[int, np.ndarray] = {}
        for s, p in self._om.items():
            self._lp[s] = np.zeros((self.turn_cap + 1, len(p.belief.lp)))
            self._lp[s][0] = p.belief.lp
        self._absent: List[List[int]] = [[0], [0]]
        self.marks: Dict[str, dict] = {}
        self._pending: Dict[int, int] = {}
        self._last: Optional[dict] = None
        self._lock = threading.RLock()
        self.created = self.last_active = time.time()
        self._begin_turn()

    # ------------------------------------------------------------------ the turn loop
    def _posterior(self, s: int, p: int) -> List[float]:
        return [round(float(x), 6) for x in np.exp(self._lp[s][p])]

    @property
    def lps(self) -> Dict[int, np.ndarray]:
        """Per opponent-modelling seat, the belief's log posterior at positions 0..n."""
        return {s: v[: self.position + 1] for s, v in self._lp.items()}

    @property
    def beliefs(self) -> Dict[int, List[List[float]]]:
        """Per opponent-modelling seat, the posterior at positions 0..n (rounded to 6 decimals)."""
        return {s: [self._posterior(s, p) for p in range(self.position + 1)] for s in self._om}

    def _begin_turn(self) -> None:
        """Step 1 of the module docstring: every bot seat decides now, from the pre-turn state."""
        from ..experiments.m45b import engine as E
        lane, m4, k, i = self.env.state, self.env.champ, self.env.k, self.env.turns
        pending = {}
        for s, p in self._bots.items():
            if isinstance(p, _ScriptedSeat):
                pending[s] = p.decide(lane, m4, k, self.env.last_actions)
            else:
                pending[s] = E.decide(p, self.cfg, lane, m4, s, i)
        self._pending = pending

    @property
    def position(self) -> int:
        return len(self.turns)

    def _resolve(self, human_code: Optional[int]) -> dict:
        from ..experiments.m45b import engine as E
        lane, m4, k, i = self.env.state, self.env.champ, self.env.k, self.env.turns
        codes = [human_code if s == self.human else self._pending[s] for s in (0, 1)]
        pre = F.position(self.cfg, lane, m4, k, i)
        r = self.env.step(*codes)
        for s, p in self._bots.items():
            if not isinstance(p, _ScriptedSeat):
                E.observe_turn(p, self.cfg, lane, m4, k, (codes[0], codes[1]))
        digest = R.state_digest(self.env.state, self.env.champ, self.env.k)
        self.turns.append([int(codes[0]), int(codes[1]), digest])
        self.rewards.append(float(r.rewards[0]))
        for s in self._om:
            self._lp[s][self.position] = self._om[s].belief.lp
        for s in (0, 1):
            self._absent[s].append(int(self.env.champ.side(s).absent))
        post = F.position(self.cfg, self.env.state, self.env.champ, self.env.k, i + 1)
        self._mark(i + 1)
        self._last = {"turn": F.turn(self.cfg, r.info, pre, post), "position": post}
        self._pending = {}
        if self.position >= self.turn_cap:
            self._stop("turn_cap")
        else:
            self._begin_turn()
        return self._last

    def _mark(self, p: int) -> None:
        from ..diagnostics import lane_exit as LX
        if p == LANE_EXIT:
            hist = {f"since_return_{x}": LX.since_return(self._absent[s], p) for s, x in ((0, "A"), (1, "B"))}
            d = LX.describe(self.cfg, (self.env.state, self.env.champ, self.env.k), hist)
            self.marks["lane_exit"] = {"position": p, "panel": {k: round(float(v), 4) for k, v in d.items()
                                                                 if v == v}}
        if p == READ_TURN:
            self.marks["lead_114"] = {"position": p, "lead_A": round(running_total(self.rewards[:READ_TURN]), 7),
                                      "note": "This game's earned-gold lead after 114 turns. M5 averaged this lead "
                                      "over many games to realize E_20 at the turn-94 lane exit; one game is an "
                                      "example, not an expectation."}

    def _stop(self, reason: str) -> None:
        if reason not in STOP_REASONS:
            raise ValueError(reason)
        self.state, self.stop_reason, self._pending = STOPPED, reason, {}

    # ------------------------------------------------------------------ submissions
    def submit(self, turn: int, code) -> dict:
        """Submit the human seat's code for turn `turn` (module docstring). Returns the turn's result (`turn`,
        `position`, and `belief` where allowed) with the current `view`."""
        with self._lock:
            self.last_active = time.time()
            if self.human is None:
                raise SessionError("this session has no human seat")
            if isinstance(turn, bool) or not isinstance(turn, int):
                raise TurnOutOfOrder(f"a turn number is an integer; got {turn!r}")
            n = self.position
            if 1 <= turn <= n:
                if not isinstance(code, bool) and isinstance(code, int) and self.turns[turn - 1][self.human] == code:
                    return self._result(turn)
                raise TurnConflict(f"turn {turn} is already resolved with another action")
            if self.state == STOPPED:
                raise SessionStopped(f"the session stopped ({self.stop_reason}) after turn {n}")
            if turn != n + 1:
                raise TurnOutOfOrder(f"the current turn is {n + 1}, not {turn}")
            why = ACT.explain(self.env.state, self.env.champ, self.human, self.cfg, code)
            if why is not None:
                raise IllegalAction(why)
            self._resolve(int(code))
            return self._result(turn)

    def advance(self, turns: int = 1) -> List[dict]:
        """Resolve up to `turns` turns of a session without a human seat."""
        with self._lock:
            if self.human is not None:
                raise SessionError("advance is for sessions without a human seat")
            out = []
            for _ in range(int(turns)):
                if self.state == STOPPED:
                    break
                self._resolve(None)
                out.append(self._result(self.position))
            return out

    def run(self) -> "GameSession":
        """Play a bot-only session to its turn cap."""
        self.advance(self.turn_cap)
        return self

    def stop(self, reason: str = "visitor") -> dict:
        with self._lock:
            if self.state != STOPPED:
                self._stop(reason)
            return self.summary()

    def forced(self) -> Optional[int]:
        """45 when the human seat's only legal code is Absent (the client offers one control to continue)."""
        if self.human is None or self.state == STOPPED:
            return None
        legal = self.env.legal(self.human)
        return legal[0] if len(legal) == 1 else None

    # ------------------------------------------------------------------ what the session may show
    def beliefs_visible(self) -> bool:
        return bool(self._om) and (self.belief_mode == "shown" or self.state == STOPPED)

    def _belief_at(self, p: int) -> Optional[dict]:
        if not self.beliefs_visible():
            return None
        from ..experiments.rl.m5 import common as C
        return {_seat_letter(s): {"types": list(C.BELIEF_TYPES), "posterior": self._posterior(s, p)} for s in self._om}

    def _result(self, t: int) -> dict:
        if self._last is not None and self._last["turn"]["t"] == t:
            res = dict(self._last)
        else:
            pos, trn = R.replay_frames(self.cfg, self._chance, self.turns, t - 1, t)
            res = {"turn": trn[0], "position": pos[1]}
        b = self._belief_at(t)
        if b is not None:
            res["belief"] = b
        res["view"] = self.view()
        return res

    def _seat_public(self, s: int) -> dict:
        seat = self.seats[s]
        d = {"seat": _seat_letter(s), "kind": seat.kind}
        if seat.kind == MODEL:
            c = self.registry.card(seat.id)
            d.update(id=c.id, label=c.label, arm=c.arm, seed=c.seed, file_sha256=c.file_sha256,
                     weights_digest=c.weights_digest, sampling="sampled, as M5 evaluated it")
        elif seat.kind == SCRIPTED:
            d.update(id=seat.id, label=f"Scripted: {seat.id}", rule=scripted_description(seat.id))
        return d

    def seeds(self) -> dict:
        stopped = self.state == STOPPED
        return {"chance": self._chance if (stopped or self.chance_chosen) else None,
                "chance_chosen": self.chance_chosen,
                "policy": self._policy if stopped else None,
                "policy_rule": {"chance": "equal to the chance seed (M5's research convention)",
                                "drawn": "drawn by the server, secret until the session stops",
                                "given": "given by the caller"}[self.policy_rule]}

    def view(self) -> dict:
        """The decision point: everything the client may know now. Never a stored bot code, a probability, a
        logit or a secret seed (tested)."""
        n = self.position
        d = {"session": self.id, "kind": self.provenance["kind"], "state": self.state, "stop_reason": self.stop_reason,
             "position": n, "next_turn": None if self.state == STOPPED else n + 1, "turn_cap": self.turn_cap,
             "lane_exit_position": LANE_EXIT, "read_position": READ_TURN,
             "seats": [self._seat_public(s) for s in (0, 1)], "belief_mode": self.belief_mode,
             "seeds": self.seeds(), "current": F.position(self.cfg, self.env.state, self.env.champ, self.env.k, n),
             "marks": dict(self.marks)}
        if self.state != STOPPED and self.human is not None:
            d["human_seat"] = _seat_letter(self.human)
            d["opponent_ready"] = True                         # the opponent has chosen; never what
            d["menu"] = ACT.menu(self.env.state, self.env.champ, self.human, self.cfg)
        b = self._belief_at(n)
        if b is not None:
            d["belief"] = b
        return d

    def history(self, p_from: int = 0, p_to: Optional[int] = None) -> dict:
        """Positions p_from..p_to and the turns between them, re-simulated from the record in a fresh environment
        (so a history request cannot change the session). Only resolved positions; beliefs only where allowed."""
        with self._lock:
            n = self.position
            p_to = n if p_to is None else p_to
            for v in (p_from, p_to):
                if isinstance(v, bool) or not isinstance(v, int):
                    raise ValueError("positions are integers")
            if p_to > n:
                raise HiddenInformation(f"position {p_to} is not resolved yet (the game is at position {n})")
            if not 0 <= p_from <= p_to:
                raise ValueError(f"positions {p_from}..{p_to} are not a range within 0..{n}")
            turns = [list(t) for t in self.turns]
            seed, vis = self._chance, self.beliefs_visible()
            beliefs = self.beliefs if vis else {}
        pos, trn = R.replay_frames(self.cfg, seed, turns, p_from, p_to)
        out = {"session": self.id, "from": p_from, "to": p_to, "positions": pos, "turns": trn}
        if vis:
            from ..experiments.rl.m5 import common as C
            out["beliefs"] = {_seat_letter(s): {"types": list(C.BELIEF_TYPES), "posteriors": v[p_from:p_to + 1]}
                              for s, v in beliefs.items()}
        return out

    def summary(self) -> dict:
        """The stop summary: the state at the stop, never a winner."""
        from ..diagnostics import lane_exit as LX
        d = LX.describe(self.cfg, (self.env.state, self.env.champ, self.env.k))
        return {"session": self.id, "state": self.state, "stop_reason": self.stop_reason, "turns": self.position,
                "position": F.position(self.cfg, self.env.state, self.env.champ, self.env.k, self.position),
                "panel": {k: round(float(v), 4) for k, v in d.items() if v == v}, "marks": dict(self.marks),
                "note": "Stopping is not an ending of the game and declares no winner; this describes the state "
                        "when the session stopped."}

    # ------------------------------------------------------------------ recordings
    def recording(self) -> dict:
        """The app recording (record.py). While the session is active, secrets and hidden beliefs are withheld."""
        from ..experiments.rl.m5 import common as C
        with self._lock:
            rec = {"schema": R.RECORD_SCHEMA, "version": R.RECORD_VERSION, "provenance": dict(self.provenance),
                   "config": {"name": "m45_calibrated", "digest": R.config_digest(self.cfg)}, "start": "s0",
                   "seeds": self.seeds(), "policy_rule": self.policy_rule,
                   "seats": [self._seat_public(s) for s in (0, 1)], "belief_mode": self.belief_mode,
                   "turn_cap": self.turn_cap, "status": {"state": self.state, "stop_reason": self.stop_reason,
                                                         "turns": self.position},
                   "turns": [list(t) for t in self.turns], "lead_A": round(running_total(self.rewards), 7),
                   "marks": dict(self.marks)}
            if self.beliefs_visible():
                rec["beliefs"] = {"types": list(C.BELIEF_TYPES),
                                  "posteriors": {_seat_letter(s): v for s, v in self.beliefs.items()}}
            return rec

    def _secret_record(self) -> dict:
        """The recording with its seeds, for the server's own replay (never sent while active)."""
        rec = self.recording()
        rec["seeds"] = dict(rec["seeds"], chance=self._chance, policy=self._policy)
        return rec

    @classmethod
    def rebuild(cls, registry: Registry, rec: dict, upto: Optional[int] = None) -> "GameSession":
        """A session re-simulated from a recording with its seeds: every bot seat's code is recomputed and must equal
        the recorded one, every state digest must match (ReplayMismatch), and the human seat's codes come from the
        recording. `upto` stops after that many turns (the session then awaits the next one)."""
        R.validate(rec)
        seeds = rec["seeds"]
        if seeds.get("chance") is None or seeds.get("policy") is None:
            raise HiddenInformation("the recording's seeds are withheld (the session is still active)")
        seats = [Seat(x["kind"], x.get("id", "")) for x in rec["seats"]]
        rule = rec.get("policy_rule", "given")
        kw = dict(chance_seed=seeds["chance"], belief_mode=rec["belief_mode"], turn_cap=rec["turn_cap"],
                  provenance=rec["provenance"], _chance_chosen=seeds["chance_chosen"])
        if rule == "chance":
            if seeds["policy"] != seeds["chance"]:
                raise R.ReplayMismatch("the research convention needs the policy seed to equal the chance seed")
            s = cls(registry, seats, research_convention=True, **kw)
        else:
            s = cls(registry, seats, policy_seed=seeds["policy"], **kw)
            s.policy_rule = rule
        turns = rec["turns"][: (len(rec["turns"]) if upto is None else int(upto))]
        for t, (a, b, digest) in enumerate(turns, start=1):
            for seat, code in ((0, a), (1, b)):
                if seat != s.human and s._pending[seat] != code:
                    raise R.ReplayMismatch(f"turn {t}: seat {_seat_letter(seat)} recomputes {s._pending[seat]}, "
                                         f"recorded {code}")
            s._resolve(a if s.human == 0 else (b if s.human == 1 else None))
            if s.turns[-1][2] != digest:
                raise R.ReplayMismatch(f"turn {t}: the state digest differs")
        if upto is None and rec["status"]["state"] == STOPPED and s.state != STOPPED:
            s._stop(rec["status"]["stop_reason"])
        return s


def replay(registry: Registry, rec: dict) -> dict:
    """Re-simulate a stopped recording with its policies (GameSession.rebuild) and report what was checked."""
    s = GameSession.rebuild(registry, rec)
    return {"turns": s.position, "lead_A": round(running_total(s.rewards), 7), "codes_recomputed":
            [_seat_letter(x) for x in s._bots], "digests_checked": s.position}


# ---------------------------------------------------------------------- research recordings and new simulations
_RESEARCH_SEATS = {"sp:0": Seat.model("sp:0"), "om:0": Seat.model("om:0"), "g_shove": Seat.scripted("g_shove")}


def from_research(registry: Registry, path: str) -> Tuple[GameSession, dict]:
    """Re-simulate a v1 M5 recording (results/m5/traces) whose seats are registry models or scripted archetypes,
    with M5's research convention (policy seed = chance seed). The actions must reproduce exactly, and so must
    `payoff_A`; returns the session and the comparison (the belief's recorded log posteriors at M5's check turns
    against the session's, which must agree within float32 precision)."""
    import json
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("schema") != "lanegame.viewer_recording" or d.get("config") != "m45_calibrated":
        raise ValueError(f"{path}: not an m45_calibrated viewer recording")
    seats = []
    for n in d["names"]:
        if n not in _RESEARCH_SEATS:
            raise ValueError(f"{path}: seat {n!r} is not in the demo registry (re-simulation needs its policy)")
        seats.append(_RESEARCH_SEATS[n])
    source = {"file": path.replace("\\", "/"), "sha256": R.file_sha256(path), "label": d.get("label", ""),
              "m5": {k: d["m5"][k] for k in ("cell", "focal_seat", "rule", "selection") if k in d.get("m5", {})}}
    s = GameSession(registry, seats, chance_seed=int(d["seed"]), research_convention=True, turn_cap=int(d["turns"]),
                    provenance={"kind": "research_record", "source": source,
                                "note": "A verified M5 recording, chosen by M5's fixed upper-median rule. One game "
                                        "illustrates; the findings rest on all six seeds."})
    s.run()
    codes = [[t[0], t[1]] for t in s.turns]
    if codes != [list(map(int, a)) for a in d["actions"]]:
        raise R.ReplayMismatch(f"{path}: the session's actions differ from the recording")
    payoff = running_total(s.rewards)
    if payoff != float(d["payoff_A"]):
        raise R.ReplayMismatch(f"{path}: payoff {payoff!r} differs from the recorded {d['payoff_A']!r}")
    cmp = {"actions": len(codes), "payoff_A": payoff}
    m5 = d.get("m5") or {}
    if m5.get("belief_log_posterior"):
        worst = 0.0
        for letter, rows in m5["belief_log_posterior"].items():
            s_idx = "AB".index(letter)
            for t, row in zip(m5["belief_turns"], rows):
                got = s.lps[s_idx][t]
                worst = max(worst, float(np.max(np.abs(np.asarray(row, dtype=np.float64) -
                                                       got.astype(np.float32).astype(np.float64)))))
        cmp["belief_max_abs_diff"] = worst
    return s, cmp


def simulate(registry: Registry, a: Seat, b: Seat, turns: int = TURN_CAP, chance_seed: Optional[int] = None,
             policy_seed: Optional[int] = None) -> GameSession:
    """A new bot-versus-bot game (provenance "new_simulation"), played to `turns` and stopped."""
    if HUMAN in (a.kind, b.kind):
        raise ValueError("simulate plays two bot seats")
    s = GameSession(registry, (a, b), chance_seed=chance_seed, policy_seed=policy_seed, turn_cap=turns)
    return s.run()


__all__ = ["GameSession", "Seat", "HUMAN", "MODEL", "SCRIPTED", "TURN_CAP", "LANE_EXIT", "READ_TURN", "SessionError",
           "IllegalAction", "TurnConflict", "TurnOutOfOrder", "SessionStopped", "HiddenInformation", "replay",
           "from_research", "simulate"]
