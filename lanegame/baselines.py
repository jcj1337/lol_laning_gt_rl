"""Simple interpretable baseline policies (usable by either side).

All scripted baselines take the resources option after a crash, except
`uniform`, which also randomizes its crash choice uniformly.
"""

from __future__ import annotations

import numpy as np

from .config import F, SH, SP
from .policy import ScriptedPolicy, pure


def _uniform_rule(k, st, legal):
    return np.full(len(legal), 1.0 / len(legal))


def _uniform_choice(k, st, opts_self, opts_opp):
    return np.full(len(opts_self), 1.0 / len(opts_self))


def uniform() -> ScriptedPolicy:
    """Uniform over legal actions and over crash-reward options."""
    return ScriptedPolicy("uniform", _uniform_rule, _uniform_choice)


def always_shove() -> ScriptedPolicy:
    return ScriptedPolicy("always_shove", lambda k, st, legal: pure(legal, SH))


def always_slowpush() -> ScriptedPolicy:
    return ScriptedPolicy("always_slowpush", lambda k, st, legal: pure(legal, SP))


def defensive_freeze() -> ScriptedPolicy:
    """Freeze whenever legal; otherwise Slow Push (fallback: park the wave at the river)."""
    return ScriptedPolicy("defensive_freeze",
                          lambda k, st, legal: pure(legal, F if F in legal else SP))


def stack_then_shove(threshold_hp: int = 5) -> ScriptedPolicy:
    """Slow Push until the leading wave holds `threshold_hp` HP, then Shove (a 'build then cash in' heuristic)."""
    def rule(k, st, legal):
        return pure(legal, SH if st.ah >= threshold_hp else SP)
    return ScriptedPolicy(f"stack_then_shove", rule)


def all_baselines():
    return [uniform(), always_shove(), always_slowpush(), defensive_freeze(), stack_then_shove()]
