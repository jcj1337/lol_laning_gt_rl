"""Backward induction for the finite-horizon zero-sum stochastic game.

Notation (report section E):
  V_K(s) = 0                                    (zero terminal bonus)
  W_k(n) = val[ C_k(n) ],  C_k(n)[i, j] = rho_A(o_i) - rho_B(o_j) + V_{k+1}(s'_{ij})
           value of the CHOICE node n reached after turn k
  Q_k(s, a, b) = g(s, a, b) + W_k(n(s, a, b))   (g = expected gank payoff to A)
  V_k(s) = max_{x in Delta(A_s)} min_{y in Delta(B_s)} x^T Q_k(s) y

The same recursion, with the max or min (or both) replaced by an expectation
under a fixed policy, gives best responses and policy evaluation
(see best_response.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .game import LaneGame, Transition
from .matrix_game import solve_matrix_game
from .policy import TabularPolicy, choice_key
from .state import State


def build_q(game: LaneGame, k: int, s: State, choice_value: Callable[[int, Transition], float]):
    la = game.legal(s, 0)
    lb = game.legal(s, 1)
    Q = np.empty((len(la), len(lb)))
    for i, a in enumerate(la):
        for j, b in enumerate(lb):
            tr = game.transition(s, k, a, b)
            Q[i, j] = tr.gank_payoff + choice_value(k, tr)
    return la, lb, Q


def choice_matrix(game: LaneGame, k: int, tr: Transition, V_next: Optional[Dict[State, float]]):
    C = np.empty((len(tr.options_a), len(tr.options_b)))
    for i, j, s2, r in tr.choice_outcomes(game.cfg):
        C[i, j] = r + (V_next[s2] if V_next is not None else 0.0)
    return C


@dataclass
class SolveStats:
    states: int = 0
    move_games: int = 0
    choice_games: int = 0
    methods: Dict[str, int] = field(default_factory=dict)
    mixed_states: int = 0
    max_gap: float = 0.0
    seconds: float = 0.0
    layer_sizes: List[int] = field(default_factory=list)

    def to_dict(self):
        return dict(self.__dict__)


@dataclass
class EquilibriumSolution:
    game: LaneGame
    layers: List[List[State]]
    V: List[Dict[State, float]]                       # V[k][s]
    Q: List[Dict[State, np.ndarray]]                  # Q[k][s]
    legal: List[Dict[State, Tuple[Tuple[int, ...], Tuple[int, ...]]]]
    x: List[Dict[State, np.ndarray]]                  # A's mixed strategy
    y: List[Dict[State, np.ndarray]]                  # B's mixed strategy
    gaps: List[Dict[State, float]]
    choice_x: Dict[tuple, np.ndarray]                 # A's choice strategies at CHOICE nodes
    choice_y: Dict[tuple, np.ndarray]
    choice_val: Dict[tuple, float]
    stats: SolveStats
    root: State = None
    k0: int = 0

    @property
    def value(self) -> float:
        """Equilibrium value V*_{k0}(root) (standard initial state unless a root was given)."""
        return self.V[self.k0][self.root]

    def policy(self, side: int) -> TabularPolicy:
        tables = self.x if side == 0 else self.y
        moves = {(k, s): p for k, layer in enumerate(tables) for s, p in layer.items()}
        choices = self.choice_x if side == 0 else self.choice_y
        return TabularPolicy(side, moves, choices, name="equilibrium")


def solve_equilibrium(game: LaneGame, layers: Optional[List[List[State]]] = None,
                      tol: float = 1e-9, root: Optional[State] = None, k0: int = 0
                      ) -> EquilibriumSolution:
    """Solve every state in `layers` (default: all states reachable from the root)."""
    t0 = time.perf_counter()
    K = game.horizon
    root = game.initial_state() if root is None else root
    if layers is None:
        layers = game.reachable(root=root, k0=k0)
    assert len(layers) == K
    stats = SolveStats(layer_sizes=[len(l) for l in layers])
    V = [dict() for _ in range(K)]
    Qs = [dict() for _ in range(K)]
    legal = [dict() for _ in range(K)]
    X = [dict() for _ in range(K)]
    Y = [dict() for _ in range(K)]
    gaps = [dict() for _ in range(K)]
    choice_x, choice_y, choice_val = {}, {}, {}

    def record(method, gap):
        stats.methods[method] = stats.methods.get(method, 0) + 1
        stats.max_gap = max(stats.max_gap, gap)

    for k in reversed(range(K)):
        V_next = V[k + 1] if k + 1 < K else None

        def choice_value(kk, tr: Transition) -> float:
            if len(tr.options_a) == 1 and len(tr.options_b) == 1:
                return tr.options_a[0][0] - tr.options_b[0][0] + (
                    V_next[tr.pre_state] if V_next is not None else 0.0)
            key = choice_key(kk, tr.pre_state, tr.options_a, tr.options_b)
            if key not in choice_val:
                C = choice_matrix(game, kk, tr, V_next)
                sol = solve_matrix_game(C, tol)
                choice_val[key] = sol.value
                choice_x[key] = sol.x
                choice_y[key] = sol.y
                stats.choice_games += 1
                record("choice-" + sol.method, sol.gap)
            return choice_val[key]

        for s in layers[k]:
            la, lb, Q = build_q(game, k, s, choice_value)
            sol = solve_matrix_game(Q, tol)
            V[k][s] = sol.value
            Qs[k][s] = Q
            legal[k][s] = (la, lb)
            X[k][s] = sol.x
            Y[k][s] = sol.y
            gaps[k][s] = sol.gap
            stats.move_games += 1
            record(sol.method, sol.gap)
            if np.count_nonzero(sol.x) > 1 or np.count_nonzero(sol.y) > 1:
                stats.mixed_states += 1
    stats.states = sum(len(l) for l in layers)
    stats.seconds = time.perf_counter() - t0
    return EquilibriumSolution(game, layers, V, Qs, legal, X, Y, gaps,
                               choice_x, choice_y, choice_val, stats, root, k0)
