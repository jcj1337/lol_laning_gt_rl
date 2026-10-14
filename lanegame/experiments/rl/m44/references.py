"""The four-action scripted reference suite of M4.4: fixed, interpretable opponents for evaluation (never training
opponents). Every reference is an existing M4.3c / M4.3d strategy (lanegame/experiments/m43c/strategies.py,
lanegame/experiments/m43d/strategies.py) wrapped as an environment agent (`StrategyAgent`: reset / notify per game,
deterministic, legal by construction); nothing here defines new behaviour.

    farm4          Last Hit, Hold: the passive farmer (never pushes by power)
    thin4          Thin every turn, Hold: defends the wave, keeps a freeze
    slowpush4      Slow Push every turn; Pressure while sieging
    shove4         Shove every turn; Pressure while sieging (the baseline economy's equilibrium line)
    shove_hunt4    Shove + Pressure; chase and Trade while the duel edge is positive (no dives)
    freeze_hunt4   Thin (freeze keeper); chase and Trade while the duel edge is positive
    grow_crash_recall4   Slow Push -> Last Hit -> Shove at 20 HP, then walk back and Recall while the crash lands
    respect4       Last Hit; Safe while the enemy starts within reach, else Yield (the conceding side)
    sh_vis         Shove; Vision when none is active during a siege, else Pressure (M4.3d: Vision-then-Pressure)
    sh_dive        Shove + Pressure; Trade while the own wave is crashed (dives a tower farmer; M4.3d)

The first six are the M4.3c pilot's references; the last two isolate the interactions in which econ and aggro differ
(vision before pressure, dives under a busy tower).
"""

from __future__ import annotations

from typing import Dict

REFERENCES = ("farm4", "thin4", "slowpush4", "shove4", "shove_hunt4", "freeze_hunt4", "grow_crash_recall4",
              "respect4", "sh_vis", "sh_dive")
CORE = ("farm4", "shove4", "freeze_hunt4", "sh_vis", "sh_dive")      # the smaller set of the smoke / curve stages


def reference_suite() -> Dict[str, object]:
    """name -> a fresh StrategyAgent (one per call: the agent carries a turn counter)."""
    from ...m43c.strategies import StrategyAgent
    from ...m43d.strategies import lookup
    return {n: StrategyAgent(lookup(n), name=f"ref:{n}") for n in REFERENCES}


__all__ = ["REFERENCES", "CORE", "reference_suite"]
