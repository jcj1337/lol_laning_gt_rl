"""Time-dependent Markov policies.

A policy for side s has two parts:
  move(k, state, s, legal)            -> probability vector over `legal`
  choose(k, pre_state, s, opts_self, opts_opp) -> probability vector over opts_self
The choice part is consulted only when side s crashed during turn k
(`opts_self` has more than one option).

Scripted policies are written from their own perspective: `own_view(state, s)`
returns a state in which the acting side is stored in the A slot.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from .config import F, SH, SP
from .state import State


def own_view(state: State, s: int) -> State:
    return state if s == 0 else state.mirror()


def choice_key(k: int, pre_state: State, opts_a, opts_b) -> tuple:
    """Identifies a CHOICE-stage node: turn k, lane after the turn, both option menus."""
    return (k, pre_state, tuple(opts_a), tuple(opts_b))


class Policy:
    name = "policy"

    def move(self, k: int, state: State, s: int, legal: Sequence[int]) -> np.ndarray:
        raise NotImplementedError

    def choose(self, k: int, pre_state: State, s: int, opts_self, opts_opp) -> np.ndarray:
        raise NotImplementedError


class TabularPolicy(Policy):
    """Explicit tables, e.g. an equilibrium strategy produced by the solver.

    moves[(k, state)]        = probability vector aligned with legal actions of side s
    choices[choice_key(...)] = probability vector over side s's options
    """

    def __init__(self, side: int, moves: Dict, choices: Dict, name: str = "tabular"):
        self.side = side
        self.moves = moves
        self.choices = choices
        self.name = name

    def move(self, k, state, s, legal):
        assert s == self.side
        return self.moves[(k, state)]

    def choose(self, k, pre_state, s, opts_self, opts_opp):
        assert s == self.side
        oa, ob = (opts_self, opts_opp) if s == 0 else (opts_opp, opts_self)
        return self.choices[choice_key(k, pre_state, oa, ob)]


def greedy_resources(opts_self) -> np.ndarray:
    """Pick the option with the most resources (ties -> first)."""
    p = np.zeros(len(opts_self))
    p[int(np.argmax([o[0] for o in opts_self]))] = 1.0
    return p


class ScriptedPolicy(Policy):
    """Deterministic or randomized rule on the own-perspective state."""

    def __init__(self, name: str, rule: Callable[[int, State, Tuple[int, ...]], np.ndarray],
                 chooser: Callable = None):
        self.name = name
        self.rule = rule
        self.chooser = chooser or (lambda k, st, opts_self, opts_opp: greedy_resources(opts_self))

    def move(self, k, state, s, legal):
        p = np.asarray(self.rule(k, own_view(state, s), tuple(legal)), dtype=float)
        assert p.shape == (len(legal),) and abs(p.sum() - 1) < 1e-12
        return p

    def choose(self, k, pre_state, s, opts_self, opts_opp):
        return np.asarray(self.chooser(k, own_view(pre_state, s), opts_self, opts_opp), dtype=float)


def pure(legal: Sequence[int], action: int) -> np.ndarray:
    p = np.zeros(len(legal))
    p[list(legal).index(action)] = 1.0
    return p
