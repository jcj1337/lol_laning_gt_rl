"""Opponent types and the Bayesian belief over them (OM1).

A type is an existing policy that can PLAY a seat (`player`) and report the probability of any action in a state
(`probs`): the likelihood a modeller assigns to the opponent's observed action.
    NetType      a learned network; its action distribution is the softmax over the legal codes (exact: the same
                 distribution its CRN sampling draws from)
    ScriptType   a deterministic M4.5c reference strategy; the modeller's likelihood trembles: (1 - eps) on the
                 scripted action, eps spread evenly over the legal codes (the strategy itself plays without tremble)
    MixType      (out of library) each turn plays strategy a with probability p, else b, by a CRN uniform

Belief. A uniform prior over the library; after each turn the opponent's chosen action is observed (an upper bound on
what a modeller learns: the game's observation is the state, from which the action is only partly inferable) and the
log posterior adds log P_tau(action | the opponent's start-of-turn state). Turns in which the opponent is absent carry no
information (every type plays Absent). A coarse variant observes only the behaviour class (Hold, Yield, Trade, Safe,
Recall, Ward, Absent) and the objective (Pressure or not), not the wave choice: its likelihood sums the type's
probabilities over the codes of the observed class.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....champions import champion_legal_actions
from ....config import CHAMP_ABSENT, CHAMP_WARD, B_RECALL, PRESSURE, champ_behavior, champ_objective
from . import common as C


# ---------------------------------------------------------------------------------------------- types
class NetType:
    kind = "net"

    def __init__(self, name: str, path: str):
        self.name, self.path = name, path

    def player(self, k0: int, seed: int, stream: int, T: int):
        from ..m45c import horizon as HZ
        return HZ.CRNNet(C.net_of(self.path), k0, seed, stream, T)

    def probs(self, cfg, lane, m4, s: int, k: int) -> Tuple[Tuple[int, ...], np.ndarray]:
        import torch
        from ....diagnostics.branch import Snapshot, observation
        obs = observation(cfg, Snapshot(lane, m4, k, (cfg.skill_a, cfg.skill_b)), s)
        with torch.no_grad():
            logits, _ = C.net_of(self.path)(torch.as_tensor(obs.vector, dtype=torch.float32).unsqueeze(0))
        lg = logits[0].double().numpy()[list(obs.legal)]
        p = np.exp(lg - lg.max())
        return tuple(int(c) for c in obs.legal), p / p.sum()


class ScriptType:
    kind = "script"

    def __init__(self, name: str, eps: float = C.EPS):
        from ...m45b.strategies import get
        self.name, self.eps, self.strategy = name, float(eps), get(name)

    def player(self, k0: int, seed: int, stream: int, T: int):
        return self.strategy

    def probs(self, cfg, lane, m4, s: int, k: int) -> Tuple[Tuple[int, ...], np.ndarray]:
        legal = tuple(int(c) for c in champion_legal_actions(lane, m4, s, cfg))
        a = int(self.strategy.decide(cfg, lane, m4, s, 0)[0])
        p = np.full(len(legal), self.eps / len(legal))
        p[legal.index(a)] += 1.0 - self.eps
        return legal, p


class MixPlayer:
    def __init__(self, a, b, p: float, seed: int, stream: int, T: int):
        self.a, self.b, self.p = a, b, float(p)
        self.u = np.random.default_rng([int(seed), 13, int(stream)]).random(int(T) + 1)

    def decide(self, cfg, lane, m4, s: int, i: int):
        return (self.a if self.u[i] < self.p else self.b).decide(cfg, lane, m4, s, i)


class MixType:
    kind = "mix"

    def __init__(self, name: str, a: str, b: str, p: float):
        from ...m45b.strategies import get
        self.name, self.a, self.b, self.p = name, get(a), get(b), float(p)

    def player(self, k0: int, seed: int, stream: int, T: int):
        return MixPlayer(self.a, self.b, self.p, seed, stream, T)


def make_type(name: str, eps: float = C.EPS):
    """A library type or an out-of-library opponent by name; "pi:<s>" is the focal policy of seed s."""
    if name.startswith("pi:"):
        return NetType(name, C.pi_path(int(name[3:])))
    if name in C.SCRIPTED:
        return ScriptType(name, eps)
    if name in C.LEARNED:
        return NetType(name, C.LEARNED[name])
    if name in C.OOL:
        kind, arg = C.OOL[name]
        if kind == "net":
            return NetType(name, arg)
        return MixType(name, *arg)
    raise ValueError(f"unknown type {name!r}")


def library(eps: float = C.EPS) -> List[object]:
    return [make_type(n, eps) for n in C.LIBRARY]


# ---------------------------------------------------------------------------------------------- the belief
COARSE_ABSENT, COARSE_WARD = 10, 11


def coarse(code: int) -> int:
    """The behaviour class (and the Pressure flag) of an action code, without its wave choice."""
    code = int(code)
    if code == CHAMP_ABSENT:
        return COARSE_ABSENT
    if code == CHAMP_WARD:
        return COARSE_WARD
    b = champ_behavior(code)
    if b == B_RECALL:
        return 2 * B_RECALL
    return 2 * b + (1 if champ_objective(code) == PRESSURE else 0)


class Belief:
    """The log posterior over `types` from observed actions (full and coarse signals), uniform prior."""

    def __init__(self, types: Sequence, eps_list: Sequence[float] = ()):
        self.types = list(types)
        n = len(self.types)
        self.full = np.full(n, -np.log(n))
        self.coarse = np.full(n, -np.log(n))
        self.updates = 0
        # optional sensitivity: the same full-signal update with other tremble allowances for the scripted types
        self.eps_list = tuple(eps_list)
        self.full_eps = {e: np.full(n, -np.log(n)) for e in self.eps_list}

    def update(self, cfg, lane, m4, s: int, k: int, code: int) -> None:
        legal = champion_legal_actions(lane, m4, s, cfg)
        if tuple(legal) == (CHAMP_ABSENT,):
            return
        code = int(code)
        cc = coarse(code)
        lf = np.zeros(len(self.types))
        lc = np.zeros(len(self.types))
        le = {e: np.zeros(len(self.types)) for e in self.eps_list}
        for j, t in enumerate(self.types):
            lg, p = t.probs(cfg, lane, m4, s, k)
            i = lg.index(code)
            lf[j] = np.log(max(p[i], 1e-300))
            lc[j] = np.log(max(float(sum(q for c, q in zip(lg, p) if coarse(c) == cc)), 1e-300))
            if self.eps_list:
                if t.kind == "script":
                    n = len(lg)
                    hit = p[i] > 1.0 / n + 1e-12 or n == 1
                    for e in self.eps_list:
                        le[e][j] = np.log((1 - e) + e / n if hit else e / n)
                else:
                    for e in self.eps_list:
                        le[e][j] = lf[j]
        self.full = _norm(self.full + lf)
        self.coarse = _norm(self.coarse + lc)
        for e in self.eps_list:
            self.full_eps[e] = _norm(self.full_eps[e] + le[e])
        self.updates += 1

    def posterior(self, which: str = "full", eps: Optional[float] = None) -> np.ndarray:
        lp = self.full_eps[eps] if eps is not None else (self.full if which == "full" else self.coarse)
        return np.exp(lp)


def _norm(lp: np.ndarray) -> np.ndarray:
    m = lp.max()
    return lp - (m + np.log(np.exp(lp - m).sum()))


__all__ = ["NetType", "ScriptType", "MixType", "MixPlayer", "make_type", "library", "Belief", "coarse"]
