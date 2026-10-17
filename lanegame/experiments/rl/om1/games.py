"""OM1 games: a responder against an opponent from s0, paired seats, common random numbers, optional belief tracking.

Game m of a cell uses chance stream default_rng(sd) (N_CHANCE uniforms per turn whatever the actions) and, per seat q,
sampling streams (sd, 7, q) for networks and (sd, 13, q) for mixtures. It is played twice: the responder in seat A and
the opponent in seat B (orientation 0), then the reverse (orientation 1), on the same chance stream. Every responder of a
cell meets the opponent on the same games and streams, so responder contrasts are paired.

Responder specifications (tuples, picklable):
    ("type", name)                 an existing policy: a library type, "pi:<s>" (the focal policy) or an OOL name
    ("switch", s, rstar, theta)    the Bayesian switcher: plays pi_s until the full-signal belief in some library type
                                   reaches theta, then that type's responder rstar[type] (a type name or "pi"); it
                                   re-reads the belief every turn, so it can switch again or back
Belief tracking (`track`): the responder's view of the opponent, updated after each turn with the opponent's observed
action (library.Belief), recorded at CHECK_TURNS.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ....champions import N_CHANCE
from ....champions_m4 import context, resolve_turn_m4
from . import common as C
from . import library as LB


class Switcher:
    """The Bayesian switcher (module docstring)."""

    def __init__(self, s: int, rstar: Dict[str, str], theta: float, k0: int, seed: int, stream: int, T: int,
                 types=None):
        self.types = types if types is not None else LB.library()
        self.belief = LB.Belief(self.types)
        self.rstar, self.theta = dict(rstar), float(theta)
        self.args = (k0, seed, stream, T)
        self.base = LB.make_type(f"pi:{s}").player(*self.args)
        self.s = s
        self._players: Dict[str, object] = {}
        self.switched_turns = 0
        self.first_switch = -1
        self.choices: Dict[str, int] = {}

    def _player(self, name: str):
        if name == "pi":
            return self.base
        if name not in self._players:
            self._players[name] = LB.make_type(name).player(*self.args)
        return self._players[name]

    def current(self) -> str:
        post = self.belief.posterior("full")
        j = int(np.argmax(post))
        if post[j] >= self.theta:
            return self.rstar.get(self.types[j].name, "pi")
        return "pi"

    def decide(self, cfg, lane, m4, s: int, i: int):
        name = self.current()
        if name != "pi":
            self.switched_turns += 1
            if self.first_switch < 0:
                self.first_switch = i
        self.choices[name] = self.choices.get(name, 0) + 1
        p = self._player(name)
        if hasattr(p, "decide"):
            return p.decide(cfg, lane, m4, s, i)
        return int(p(cfg, lane, m4, s, i)), ""

    def observe(self, cfg, lane, m4, opp_seat: int, k: int, code: int) -> None:
        self.belief.update(cfg, lane, m4, opp_seat, k, code)


def _player(spec: tuple, k0: int, seed: int, stream: int, T: int, types=None):
    if spec[0] == "type":
        return LB.make_type(spec[1]).player(k0, seed, stream, T)
    if spec[0] == "switch":
        _, s, rstar, theta = spec
        return Switcher(int(s), dict(rstar), float(theta), k0, seed, stream, T, types)
    raise ValueError(spec)


def _decide(pol, cfg, lane, m4, s, i) -> int:
    if hasattr(pol, "decide"):
        return int(pol.decide(cfg, lane, m4, s, i)[0])
    return int(pol(cfg, lane, m4, s, i))


def play(cfg, responder: tuple, opponent: str, sd: int, T: int, orient: int, track: bool = False,
         eps_list=(), types=None, lane_step=None) -> dict:
    """One game from s0 (module docstring). Returns r [T] (the responder's reward per turn) and, when tracked (or for
    a switcher), the belief at CHECK_TURNS: log posteriors (full, coarse, eps variants) [len(CHECK_TURNS), n_types]."""
    from ...m45b import states as S
    lane, m4, k0 = S.s0(cfg)
    ctx = context(cfg)
    rs, os_ = (0, 1) if orient == 0 else (1, 0)
    rp = _player(responder, k0, sd, rs, T, types)
    op = LB.make_type(opponent).player(k0, sd, os_, T)
    belief = None
    if track and not isinstance(rp, Switcher):
        belief = LB.Belief(types if types is not None else LB.library(), eps_list)
    rng = np.random.default_rng(int(sd))
    r = np.zeros(T, dtype=np.float64)
    checks = {t: i for i, t in enumerate(C.CHECK_TURNS)}
    rec = None
    if belief is not None or isinstance(rp, Switcher):
        n = len((belief or rp.belief).types)
        rec = {"full": np.zeros((len(C.CHECK_TURNS), n)), "coarse": np.zeros((len(C.CHECK_TURNS), n))}
        for e in eps_list:
            rec[f"eps{e}"] = np.zeros((len(C.CHECK_TURNS), n))
    for i in range(T):
        u = rng.random(N_CHANCE)
        pols = (rp, op) if rs == 0 else (op, rp)
        a = _decide(pols[0], cfg, lane, m4, 0, i)
        b = _decide(pols[1], cfg, lane, m4, 1, i)
        opp_code = b if rs == 0 else a
        bl = rp.belief if isinstance(rp, Switcher) else belief
        if bl is belief and i >= C.CHECK_TURNS[-1]:
            bl = None                                   # tracking for identification stops after the last check
        if bl is not None:
            bl.update(cfg, lane, m4, os_, k0 + i, opp_code)
            if (i + 1) in checks:
                j = checks[i + 1]
                rec["full"][j] = bl.full
                rec["coarse"][j] = bl.coarse
                for e in eps_list:
                    rec[f"eps{e}"][j] = bl.full_eps[e] if e in bl.full_eps else bl.full
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=lane_step, u=u, ctx=ctx)
        r[i] = t.reward if rs == 0 else -t.reward
        lane, m4 = t.lane.state, t.champ
    out = {"r": r}
    if rec is not None:
        out["belief"] = rec
    if isinstance(rp, Switcher):
        out["switch"] = {"switched_turns": rp.switched_turns, "first_switch": rp.first_switch,
                         "choices": dict(rp.choices)}
    return out


__all__ = ["Switcher", "play"]
