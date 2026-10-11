"""Simple interpretable baseline policies (usable by either side).

All scripted baselines take the resources option after a crash in the "choice"
siege mode, except `uniform`, which also randomizes its crash choice uniformly.

Three families:

* `all_baselines()` - the revised baselines over combined actions
  (wave management, positioning). When positioning is disabled (original
  model) a Safe action falls back to the Contest action with the same wave
  management, so every baseline is defined in every configuration. In the
  "objective" siege mode a Contest action during an eligible siege needs a
  siege objective; each baseline carries a *siege tendency* (`siege=`):
  "pressure" (default: Apply Pressure), "vision" (Establish Vision), or
  "safe" (disengage: play the Safe action with the same wave management).
* `siege_baselines()` - the opponent types of the siege study (section G):
  the revised baselines plus variants with different siege tendencies.
* `v1_baselines()` - the five original Phase 1 baselines (wave management
  only; they always Contest).
"""

from __future__ import annotations

import numpy as np

from .config import (F, F_C, NONE, PRESSURE, SH, SH_C, SH_S, SP, SP_C, SP_S, VISION, base_of, objective_of,
                     posture_of, with_objective)
from .policy import ScriptedPolicy, pure

SIEGE_OBJECTIVE = {"pressure": PRESSURE, "vision": VISION}


def _uniform_rule(k, st, legal):
    return np.full(len(legal), 1.0 / len(legal))


def _uniform_choice(k, st, opts_self, opts_opp):
    return np.full(len(opts_self), 1.0 / len(opts_self))


def resolve(legal, a: int, siege: str = "pressure"):
    """The legal action closest to the base action `a` under the siege tendency, or None.

    1. `a` itself if legal.
    2. During an eligible siege (Contest actions carry objectives): the Contest action with the
       tendency's objective, or, for tendency "safe", the Safe action with the same wave management.
    3. Without positioning (original model): the Contest code with the same wave management.
    """
    if a in legal:
        return a
    b = base_of(a)
    if posture_of(b) == 0 and (b + 6 * PRESSURE) in legal:          # siege: Contest needs an objective
        if siege == "safe":
            alt = b + 3
            return alt if alt in legal else None
        return b + 6 * SIEGE_OBJECTIVE[siege]
    if b % 3 in legal and len(legal) <= 3:                            # positioning disabled
        return b % 3
    return None


def prefer(legal, *candidates, siege: str = "pressure") -> np.ndarray:
    """Pure strategy on the first candidate that resolves to a legal action."""
    for a in candidates:
        r = resolve(legal, a, siege)
        if r is not None:
            return pure(legal, r)
    raise ValueError(f"no candidate in {candidates} is legal in {legal}")


def uniform() -> ScriptedPolicy:
    """Uniform over legal actions and over crash-reward options."""
    return ScriptedPolicy("uniform", _uniform_rule, _uniform_choice)


def _named(name: str, siege: str) -> str:
    return name if siege == "pressure" else f"{name}[{siege}]"


# ------------------------------------------------------------- revised baselines
def aggressive(siege: str = "pressure") -> ScriptedPolicy:
    """Shove + Contest every turn: maximum pressure, maximum exposure."""
    return ScriptedPolicy(_named("aggressive", siege), lambda k, st, legal: prefer(legal, SH_C, siege=siege))


def freeze_contest(siege: str = "pressure") -> ScriptedPolicy:
    """Freeze + Contest whenever Freeze is legal; otherwise Slow Push + Safe (a defensive farmer)."""
    return ScriptedPolicy(_named("freeze_contest", siege),
                          lambda k, st, legal: prefer(legal, F_C if F_C in legal else SP_S, siege=siege))


def passive_safe() -> ScriptedPolicy:
    """Slow Push + Safe every turn: never contests, never shoves, never sieges."""
    return ScriptedPolicy("passive_safe", lambda k, st, legal: prefer(legal, SP_S))


def stack_builder(threshold_hp: int = 5, siege: str = "pressure") -> ScriptedPolicy:
    """Slow Push + Contest until the leading wave holds `threshold_hp` HP, then Shove + Contest
    (a 'build a big wave and crash it' heuristic)."""
    def rule(k, st, legal):
        return prefer(legal, SH_C if st.ah >= threshold_hp else SP_C, siege=siege)
    return ScriptedPolicy(_named("stack_builder", siege), rule)


def release_safe() -> ScriptedPolicy:
    """Shove + Safe every turn: lets every wave walk into the enemy tower without following it."""
    return ScriptedPolicy("release_safe", lambda k, st, legal: prefer(legal, SH_S))


def contest_slowpush(siege: str = "pressure") -> ScriptedPolicy:
    """Slow Push + Contest every turn (the original always_slowpush, which always contested)."""
    return ScriptedPolicy(_named("contest_slowpush", siege), lambda k, st, legal: prefer(legal, SP_C, siege=siege))


def all_baselines():
    return [uniform(), aggressive(), freeze_contest(), passive_safe(), stack_builder(),
            release_safe(), contest_slowpush()]


def siege_baselines():
    """Opponent types of the siege study: the revised baselines (Apply Pressure when sieging) plus
    vision-seeking and disengaging variants of the crash-prone ones."""
    return all_baselines() + [aggressive("vision"), stack_builder(siege="vision"), contest_slowpush("vision"),
                              stack_builder(siege="safe"), aggressive("safe")]


# ------------------------------------------------------------- original baselines
def always_shove() -> ScriptedPolicy:
    return ScriptedPolicy("always_shove", lambda k, st, legal: pure(legal, SH))


def always_slowpush() -> ScriptedPolicy:
    return ScriptedPolicy("always_slowpush", lambda k, st, legal: pure(legal, SP))


def defensive_freeze() -> ScriptedPolicy:
    """Freeze when legal, otherwise Slow Push (fallback: park the wave at the lane)."""
    return ScriptedPolicy("defensive_freeze",
                          lambda k, st, legal: pure(legal, F if F in legal else SP))


def stack_then_shove(threshold_hp: int = 5) -> ScriptedPolicy:
    """Slow Push until the leading wave holds `threshold_hp` HP, then Shove (a 'build then cash in' heuristic)."""
    def rule(k, st, legal):
        return pure(legal, SH if st.ah >= threshold_hp else SP)
    return ScriptedPolicy(f"stack_then_shove", rule)


def v1_baselines():
    """The five baselines of the original Phase 1 report (Contest-only wave actions)."""
    return [uniform(), always_shove(), always_slowpush(), defensive_freeze(), stack_then_shove()]
