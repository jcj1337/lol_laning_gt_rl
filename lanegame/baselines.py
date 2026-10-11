"""Simple interpretable baseline policies (usable by either side).

All scripted baselines take the resources option after a crash, except
`uniform`, which also randomizes its crash choice uniformly.

Two families:

* `all_baselines()` - the revised baselines over combined actions
  (wave management, positioning). When positioning is disabled (original
  model) a Safe action falls back to the Contest action with the same wave
  management, so every baseline is defined in every configuration.
* `v1_baselines()` - the five original Phase 1 baselines (wave management
  only; they always Contest).
"""

from __future__ import annotations

import numpy as np

from .config import F, F_C, SH, SH_C, SH_S, SP, SP_C, SP_S
from .policy import ScriptedPolicy, pure


def _uniform_rule(k, st, legal):
    return np.full(len(legal), 1.0 / len(legal))


def _uniform_choice(k, st, opts_self, opts_opp):
    return np.full(len(opts_self), 1.0 / len(opts_self))


def prefer(legal, *candidates) -> np.ndarray:
    """Pure strategy on the first legal candidate; a Safe code falls back to its Contest code."""
    for a in candidates:
        if a in legal:
            return pure(legal, a)
        if a % 3 in legal:
            return pure(legal, a % 3)
    raise ValueError(f"no candidate in {candidates} is legal in {legal}")


def uniform() -> ScriptedPolicy:
    """Uniform over legal actions and over crash-reward options."""
    return ScriptedPolicy("uniform", _uniform_rule, _uniform_choice)


# ------------------------------------------------------------- revised baselines
def aggressive() -> ScriptedPolicy:
    """Shove + Contest every turn: maximum pressure, maximum exposure."""
    return ScriptedPolicy("aggressive", lambda k, st, legal: prefer(legal, SH_C))


def freeze_contest() -> ScriptedPolicy:
    """Freeze + Contest whenever Freeze is legal; otherwise Slow Push + Safe (a defensive farmer)."""
    return ScriptedPolicy("freeze_contest",
                          lambda k, st, legal: prefer(legal, F_C if F_C in legal else SP_S))


def passive_safe() -> ScriptedPolicy:
    """Slow Push + Safe every turn: never contests, never shoves."""
    return ScriptedPolicy("passive_safe", lambda k, st, legal: prefer(legal, SP_S))


def stack_builder(threshold_hp: int = 5) -> ScriptedPolicy:
    """Slow Push + Contest until the leading wave holds `threshold_hp` HP, then Shove + Contest."""
    def rule(k, st, legal):
        return prefer(legal, SH_C if st.ah >= threshold_hp else SP_C)
    return ScriptedPolicy("stack_builder", rule)


def release_safe() -> ScriptedPolicy:
    """Shove + Safe every turn: lets every wave walk into the enemy tower without following it."""
    return ScriptedPolicy("release_safe", lambda k, st, legal: prefer(legal, SH_S))


def contest_slowpush() -> ScriptedPolicy:
    """Slow Push + Contest every turn (the original always_slowpush, which always contested)."""
    return ScriptedPolicy("contest_slowpush", lambda k, st, legal: prefer(legal, SP_C))


def all_baselines():
    return [uniform(), aggressive(), freeze_contest(), passive_safe(), stack_builder(),
            release_safe(), contest_slowpush()]


# ------------------------------------------------------------- original baselines
def always_shove() -> ScriptedPolicy:
    return ScriptedPolicy("always_shove", lambda k, st, legal: pure(legal, SH))


def always_slowpush() -> ScriptedPolicy:
    return ScriptedPolicy("always_slowpush", lambda k, st, legal: pure(legal, SP))


def defensive_freeze() -> ScriptedPolicy:
    """Freeze when legal, otherwise Slow Push (fallback: park the wave at the river)."""
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
