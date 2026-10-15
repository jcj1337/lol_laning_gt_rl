"""The fixed evaluation opponents of M4.5c: existing M4.5b contingent strategies (lanegame/experiments/m45b/
strategies.py), wrapped as environment agents (`m43c.strategies.StrategyAgent`: deterministic, memoryless, legal by
construction). They are EVALUATION opponents only, never training opponents; nothing here defines new behaviour.

    g_farm             passive farming: Last Hit + Hold everywhere
    g_shove            plain Shove (+ Pressure)
    g_shove_sr         Shove (+ Pressure); Recall below 55% HP once the own wave has crashed (s0 equilibrium, M4.5b §7.4)
    wk_shove_reset_sh  Shove; while frozen on, reset below 60% HP (the r2 s0 equilibrium, M4.5b §7.1)
    g_shove_edge       Shove; Trade only with a clear duel edge (>= 0.05) at >= 60% HP (asymmetric-start equilibria)
    g_bully            aggressive Trading: Last Hit; Trade whenever the enemy is catchable without a dive
    g_safe             defensive: Last Hit + Safe everywhere (M4.5b: exploitable, costs 0.15-0.56 per turn)
    fz_guard_sh        freeze-oriented: keep a freeze with Thin, Trade a catchable walk-up at >= 50% HP; Shove outside
    g_shove_ward       Ward-oriented: Shove (+ Pressure) + Ward when exposed (vision down, own wave forward)

plus, where requested, an M4.4b aggro extension final through the W-dropping adapter (common.OODNetAgent), labelled
OUT OF DISTRIBUTION. CORE is the smaller set of the smoke stage.
"""

from __future__ import annotations

from typing import Dict

SUITE = ("g_farm", "g_shove", "g_shove_sr", "wk_shove_reset_sh", "g_shove_edge", "g_bully", "g_safe", "fz_guard_sh",
         "g_shove_ward")
CORE = ("g_farm", "g_shove", "g_bully", "fz_guard_sh")
ROLE = {"g_farm": "passive farming", "g_shove": "plain Shove", "g_shove_sr": "Shove with Recall",
        "wk_shove_reset_sh": "Shove with frozen-on reset", "g_shove_edge": "Shove, selective trades",
        "g_bully": "aggressive Trading", "g_safe": "defensive Safe", "fz_guard_sh": "freeze-oriented (guarded)",
        "g_shove_ward": "Ward-oriented Shove"}
OOD = ("ood:m44:aggro:ext:0",)             # M4.4b aggro 4.10M final, seed 0 (out of distribution)


def reference(name: str):
    """A fresh StrategyAgent named ref:<name> (one per call: the agent carries a turn counter)."""
    from ...m43c.strategies import StrategyAgent
    from ...m45b.strategies import get
    if name not in SUITE:
        raise ValueError(f"{name!r} is not in the M4.5c reference suite {SUITE}")
    return StrategyAgent(get(name), name=f"ref:{name}")


def reference_suite() -> Dict[str, object]:
    return {n: reference(n) for n in SUITE}


__all__ = ["SUITE", "CORE", "ROLE", "OOD", "reference", "reference_suite"]
