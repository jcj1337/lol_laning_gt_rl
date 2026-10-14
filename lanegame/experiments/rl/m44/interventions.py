"""Component interventions on a policy's action distribution in the 61-code four-action space.

An intervention replaces ONE decision component (wave management, behaviour, or siege objective) and leaves the
policy's other decisions to the policy, through an explicit rule on its probability vector p over obs.legal. It
generalizes the M4.3b follow-up's wave-only rule (`experiments/rl/m43b/intervene.py`, which this module reuses for
the wave component) to the four wave slots and to the behaviour and objective components.

Rules (both defined on p over obs.legal; the reshaped vector q is sampled with the same single uniform, so paired
rollouts stay paired):

  wave       "marginal"      every present, non-Recall code sends its mass to (w, behaviour, objective): the
                             behaviour / objective come from the policy's marginal over them. Under natural waves every
                             wave slot is legal with every legal behaviour / objective, so the replacement is always
                             legal (checked: a replacement outside obs.legal raises, it is never silently dropped).
             "conditional"   the mass of the codes whose wave is w, renormalised: the behaviour / objective come from
                             the policy's conditional distribution given that it plays w.
             Both keep Recall at the policy's own probability (Recall has no wave management) and leave ABSENT alone.
  behavior   "conditional" only: the codes whose behaviour is b, renormalised to the whole mass (Recall included:
             forcing a behaviour replaces the decision to Recall). B_RECALL forces the single Recall code.
  objective  "conditional" only: the codes whose objective is o, renormalised to the whole mass.

A marginal rule for behaviour / objective is not offered: (w, Trade, Pressure) -> (w, Safe, Pressure) is not a legal
code (Safe and Yield never carry an objective), and a silent fallback would fabricate a different intervention.

LEGALITY. `possible(legal, component, value)` says whether the intervention exists at a state (some legal code has
that component value). Candidates are built only where it is possible; a plan that meets an impossible value on a
LATER turn of a multi-turn window leaves that turn's distribution unchanged and counts it (`skipped_turns`), it never
plays an illegal or invented action.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from ....config import B_RECALL, CHAMP_ABSENT
from ....diagnostics.branch import DEFER, Plan
from ....env.agents import sample
from ..m43b.intervene import intervene as wave_intervene
from .actions import RECALL, codes_with

COMPONENTS = ("wave", "behavior", "objective")
RULES = {"wave": ("marginal", "conditional"), "behavior": ("conditional",), "objective": ("conditional",)}


def possible(legal: Sequence[int], component: str, value: int) -> bool:
    legal = tuple(int(a) for a in legal)
    if legal == (CHAMP_ABSENT,):
        return False
    if component == "behavior" and value == B_RECALL:
        return RECALL in legal
    kw = {component: value}
    return len(codes_with(legal, **kw)) > 0


def intervene(p, legal: Sequence[int], component: str, value: Optional[int], rule: str = "conditional") -> np.ndarray:
    """The reshaped probability vector over `legal` (module docstring); value None: p unchanged."""
    p = np.asarray(p, dtype=float)
    legal = tuple(int(a) for a in legal)
    if value is None or legal == (CHAMP_ABSENT,):
        return p
    if len(p) != len(legal):
        raise ValueError(f"p has {len(p)} entries for {len(legal)} legal actions")
    if component not in COMPONENTS:
        raise ValueError(f"unknown component {component!r}")
    if rule not in RULES[component]:
        raise ValueError(f"rule {rule!r} is not defined for the {component} component (allowed: {RULES[component]})")
    if component == "wave":
        return wave_intervene(p, legal, int(value), rule)
    q = np.zeros_like(p)
    if component == "behavior" and value == B_RECALL:
        if RECALL not in legal:
            raise ValueError(f"Recall is not legal here (legal {legal})")
        q[legal.index(RECALL)] = 1.0
        return q
    sl = [legal.index(a) for a in codes_with(legal, **{component: int(value)})]
    if not sl:
        raise ValueError(f"no legal action with {component} {value} (legal {legal})")
    mass = float(p[sl].sum())
    if mass > 0.0:
        q[sl] = p[sl] / mass
    else:                                                   # the policy never plays that slice: uniform over it
        q[sl] = 1.0 / len(sl)
    return q


class ComponentPlan(Plan):
    """For `turns` turns of a branch (t = 0 is the branch turn), force `component` to value_fn(obs, t) under `rule`
    (DEFER on every turn: the policy's distribution is reshaped and sampled with the rollout's uniform), then hand
    control back to the policy for good. An absent champion's turns count towards the window (unchanged ABSENT)."""

    def __init__(self, name: str, component: str, value_fn: Callable, turns: int = 1, rule: str = "conditional"):
        if component not in COMPONENTS or rule not in RULES[component]:
            raise ValueError(f"invalid intervention {component}/{rule}")
        self.name, self.component, self.value_fn, self.turns, self.rule = name, component, value_fn, int(turns), rule
        self.skipped_turns = 0

    def act(self, obs, t: int) -> Optional[int]:
        return DEFER if t < self.turns else None

    def transform(self, obs, t: int, p: np.ndarray) -> np.ndarray:
        if obs.legal == (CHAMP_ABSENT,):
            return p
        v = self.value_fn(obs, t)
        if v is None:
            return p
        if not possible(obs.legal, self.component, v):
            if t == 0:
                raise ValueError(f"{self.name}: the intervention {self.component} = {v} is not possible at the branch "
                                 f"state (legal {obs.legal}); build candidates with `possible` first")
            self.skipped_turns += 1
            return p
        return intervene(p, obs.legal, self.component, v, self.rule)


def const(value: int) -> Callable:
    def fn(obs, t):
        return value
    return fn


def component_plan(name: str, component: str, value: int, turns: int = 1, rule: str = "conditional") -> ComponentPlan:
    return ComponentPlan(name, component, const(value), turns, rule)


class ComponentAgent:
    """`agent` with one component forced on every present turn (whole games, e.g. "never Shove while holding a
    freeze" against a reference), as an environment agent: `distribution` is the reshaped vector, `act` samples it
    with the environment's inverse-CDF rule. value_fn(obs) -> value or None (no intervention this turn); impossible
    values leave the distribution unchanged (counted in `skipped_turns`)."""

    def __init__(self, agent, component: str, value_fn: Callable, rule: str = "conditional", name: Optional[str] = None):
        if component not in COMPONENTS or rule not in RULES[component]:
            raise ValueError(f"invalid intervention {component}/{rule}")
        self.base, self.component, self.value_fn, self.rule = agent, component, value_fn, rule
        self.name = name or f"{getattr(agent, 'name', 'agent')}[{component}:{rule}]"
        self.skipped_turns = 0

    def distribution(self, obs) -> np.ndarray:
        p = np.asarray(self.base.distribution(obs), dtype=float)
        if obs.legal == (CHAMP_ABSENT,):
            return p
        v = self.value_fn(obs)
        if v is None:
            return p
        if not possible(obs.legal, self.component, v):
            self.skipped_turns += 1
            return p
        return intervene(p, obs.legal, self.component, v, self.rule)

    def probabilities(self, obs) -> np.ndarray:
        full = np.zeros(len(obs.mask))
        full[list(obs.legal)] = self.distribution(obs)
        return full

    def act(self, obs, rng: Optional[np.random.Generator] = None) -> int:
        p = self.distribution(obs)
        if rng is None:
            return obs.legal[int(np.argmax(p))]
        return obs.legal[sample(rng, p)]

    def reset(self) -> None:
        f = getattr(self.base, "reset", None)
        if callable(f):
            f()

    def notify(self, info, side) -> None:
        f = getattr(self.base, "notify", None)
        if callable(f):
            f(info, side)


__all__ = ["COMPONENTS", "RULES", "possible", "intervene", "ComponentPlan", "component_plan", "const", "ComponentAgent"]
