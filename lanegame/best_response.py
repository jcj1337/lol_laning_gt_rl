"""Exact policy evaluation, best responses and exploitability.

Against a fixed time-dependent Markov policy of one player, the other player
faces a finite-horizon MDP, so an optimal (pure) best response is found by
backward induction:

  BR to A's policy pi:  V_k(s) = min_b  sum_a pi_k(a|s) Q_k(s, a, b)
  BR to B's policy sig: V_k(s) = max_a  sum_b sig_k(b|s) Q_k(s, a, b)

with the same rule at CHOICE nodes. Restricting the responder to Markov
policies is without loss of optimality in a finite-horizon MDP.

Exploitability (both measured relative to the equilibrium value v* = V*_0(s_0)):
  Expl_A(pi)  = v* - min_sigma V(pi, sigma | s_0)    >= 0
  Expl_B(sig) = max_pi V(pi, sig | s_0) - v*         >= 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .game import LaneGame, Transition
from .policy import Policy, TabularPolicy, choice_key
from .solver import build_q, choice_matrix
from .state import State


@dataclass
class InductionResult:
    V: List[Dict[State, float]]
    policy: Optional[TabularPolicy]      # responder's best response (None for evaluation)

    def value(self, game: LaneGame, root: Optional[State] = None, k0: int = 0) -> float:
        return self.V[k0][game.initial_state() if root is None else root]


def induct(game: LaneGame, layers: List[List[State]],
           pol_a: Optional[Policy], pol_b: Optional[Policy]) -> InductionResult:
    """Backward induction with fixed policies; a None side best-responds."""
    if pol_a is None and pol_b is None:
        raise ValueError("use solver.solve_equilibrium when neither policy is fixed")
    K = game.horizon
    responder = 0 if pol_a is None else (1 if pol_b is None else None)
    V: List[Dict[State, float]] = [dict() for _ in range(K)]
    br_moves, br_choices = {}, {}

    for k in reversed(range(K)):
        V_next = V[k + 1] if k + 1 < K else None
        cache: Dict[tuple, float] = {}

        def choice_value(kk, tr: Transition) -> float:
            if len(tr.options_a) == 1 and len(tr.options_b) == 1:
                return tr.options_a[0][0] - tr.options_b[0][0] + (
                    V_next[tr.pre_state] if V_next is not None else 0.0)
            key = choice_key(kk, tr.pre_state, tr.options_a, tr.options_b)
            if key in cache:
                return cache[key]
            C = choice_matrix(game, kk, tr, V_next)
            xa = (pol_a.choose(kk, tr.pre_state, 0, tr.options_a, tr.options_b)
                  if pol_a is not None and len(tr.options_a) > 1 else None)
            yb = (pol_b.choose(kk, tr.pre_state, 1, tr.options_b, tr.options_a)
                  if pol_b is not None and len(tr.options_b) > 1 else None)
            if xa is None and len(tr.options_a) == 1:
                xa = np.ones(1)
            if yb is None and len(tr.options_b) == 1:
                yb = np.ones(1)
            if responder == 0 and xa is None:
                vals = C @ yb
                i = int(np.argmax(vals))
                br_choices[key] = np.eye(len(vals))[i]
                v = float(vals[i])
            elif responder == 1 and yb is None:
                vals = xa @ C
                j = int(np.argmin(vals))
                br_choices[key] = np.eye(len(vals))[j]
                v = float(vals[j])
            else:
                v = float(xa @ C @ yb)
            cache[key] = v
            return v

        for s in layers[k]:
            la, lb, Q = build_q(game, k, s, choice_value)
            if responder == 0:
                vals = Q @ pol_b.move(k, s, 1, lb)
                i = int(np.argmax(vals))
                br_moves[(k, s)] = np.eye(len(la))[i]
                V[k][s] = float(vals[i])
            elif responder == 1:
                vals = pol_a.move(k, s, 0, la) @ Q
                j = int(np.argmin(vals))
                br_moves[(k, s)] = np.eye(len(lb))[j]
                V[k][s] = float(vals[j])
            else:
                V[k][s] = float(pol_a.move(k, s, 0, la) @ Q @ pol_b.move(k, s, 1, lb))

    br = None
    if responder is not None:
        br = TabularPolicy(responder, br_moves, br_choices, name="best-response")
    return InductionResult(V, br)


def evaluate(game: LaneGame, layers, pol_a: Policy, pol_b: Policy,
             root: Optional[State] = None, k0: int = 0) -> float:
    """Exact expected payoff to A when A plays pol_a and B plays pol_b."""
    return induct(game, layers, pol_a, pol_b).value(game, root, k0)


def best_response_value(game: LaneGame, layers, policy: Policy, side: int) -> InductionResult:
    """The opponent of `side` best-responds to `policy`."""
    if side == 0:
        return induct(game, layers, policy, None)
    return induct(game, layers, None, policy)


@dataclass
class Exploitability:
    side: int
    equilibrium_value: float
    worst_case_value: float          # value to A when the opponent best-responds
    exploitability: float

    def to_dict(self):
        return dict(self.__dict__)


def exploitability(game: LaneGame, layers, policy: Policy, side: int,
                   equilibrium_value: float, root: Optional[State] = None,
                   k0: int = 0) -> Exploitability:
    res = best_response_value(game, layers, policy, side)
    w = res.value(game, root, k0)
    e = (equilibrium_value - w) if side == 0 else (w - equilibrium_value)
    return Exploitability(side, equilibrium_value, w, e)
