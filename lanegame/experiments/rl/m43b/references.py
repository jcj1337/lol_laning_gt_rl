"""The fixed reference opponents of the M4.3b evaluation.

The M4.3a readout used the scripted natural-wave suite (`agents_m4.natural_suite`), which every trained policy beat by
10-19 resources. The M4.2e-B/C studies built a stronger strategy space: thirteen REACTIVE policies per seat (they read
HP, contact, the lane and the siege; `lanegame.experiments.m4.zoning_natural.POLICIES`) whose sampled policy games
gave the equilibria of the M4.2e-C report §6.4 (Clear + punish against Clear + punish; both sides slow-push at equal
power). `ReactiveAgent` plays such a policy as an environment agent (deterministic, one-hot distribution, a function
of the observation only), so it can be evaluated with the discounted evaluator, audited and used as a counterfactual
opponent exactly like a scripted or network agent.

`reference_suite(cfg)` is the suite of this study (the game is unchanged; only the opponents are stronger):

  farm            Hold the wave, Hold position: a pure last-hitter (the weak passive reference)
  push_punish     slow push (Thin while the own front is at most 2 HP ahead) + punish an opponent in reach + Pressure:
                  the equal-power equilibrium line of M4.2e-C §6.4
  clear_punish    Clear every turn + punish + Pressure: the stronger side's equilibrium line
  bully_shove     Trade every turn the enemy is on the lane; Clear while it is away; Pressure
  freeze_punish   the threshold freeze controller + punish
  respect         back off while the enemy starts within reach, else stand off (the zoned champion's best reply)
  crash_recall    build a stack, Clear it into the tower, walk back and Recall while the crash occupies the enemy
  trading_pusher  the M4.3a scripted reference kept for continuity (natural_suite: slow push + duel-edge trades)

Every policy of the suite Recalls at 40% HP when it can (and resets while the enemy is away below 70%), as in the
policy games. CURVE_REFS are the four used on the learning curves; the headline evaluation uses all of them.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ....config import GameConfig
from ....env import Observation

CURVE_REFS = ("farm", "push_punish", "clear_punish", "bully_shove")
HEAD_REFS = ("farm", "push_punish", "clear_punish", "bully_shove", "freeze_punish", "respect", "crash_recall",
             "trading_pusher")


class ReactiveAgent:
    """A `zoning_natural.Policy` (or any object with `act(cfg, lane, m4, side)`) as an environment agent."""

    def __init__(self, policy, name: Optional[str] = None):
        self.policy = policy
        self.name = name or getattr(policy, "name", "reactive")

    def choose_action(self, obs: Observation) -> int:
        if obs.cfg is None:
            raise ValueError("a ReactiveAgent needs Observation.cfg (observations built by LaneEnv carry it)")
        a = int(self.policy.act(obs.cfg, obs.state, obs.champ, obs.side))
        if a not in obs.legal:                                 # the policies fall back to legal codes; guard anyway
            raise ValueError(f"{self.name}: policy chose illegal action {a} (legal {obs.legal})")
        return a

    def distribution(self, obs: Observation) -> np.ndarray:
        p = np.zeros(len(obs.legal))
        p[obs.legal.index(self.choose_action(obs))] = 1.0
        return p

    def act(self, obs: Observation, rng=None) -> int:
        return self.choose_action(obs)


def reference_suite(cfg: Optional[GameConfig] = None) -> Dict[str, object]:
    from ....env.agents_m4 import natural_suite
    from ....viewer.policies import POLICIES as VIEWER
    from ...m4.zoning_natural import POLICIES as ZONING
    zoning = {p.name: p for p in ZONING}
    out = {}
    for n in ("farm", "push_punish", "clear_punish", "bully_shove", "freeze_punish", "respect"):
        out[n] = ReactiveAgent(zoning[n], n)
    out["crash_recall"] = ReactiveAgent(VIEWER["crash_recall"], "crash_recall")
    out["trading_pusher"] = natural_suite()["trading_pusher"]
    return out


__all__ = ["ReactiveAgent", "reference_suite", "CURVE_REFS", "HEAD_REFS"]
