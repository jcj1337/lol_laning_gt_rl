"""Opponent protocols and styles of M4.4c (Part 1.2, 3.2, 4.3, 6.2).

Protocols (the opponent's simultaneous branch-turn action; its continuation agent is unchanged):
    policy      it samples its own policy from its own observation with its own stream (identical for every candidate)
    modal       it commits to its own most probable action on the branch turn (`diagnostics.branch.Commit`), then
                follows its continuation agent

Styles (the opponent's CONTINUATION agent: the learned policy with ONE component forced in a stated situation,
`m44.interventions.ComponentAgent`; every other decision stays the network's own). They answer "what if the opponent
actively breaks / passively farms / respects the threat", without leaving the learned behaviour elsewhere:
    breaker     wave = Shove while frozen on (the enemy holds a freeze on it)
    passive     wave = Last Hit while frozen on
    stepback    behaviour = Safe while frozen on (it steps back from the held wave)
    ignore      behaviour = Hold while the enemy starts the turn within reach and own HP > 40% (it ignores the threat
                and keeps farming; it can still Recall when damaged)
    respect     behaviour = Safe while the enemy starts the turn within reach (contact), else the policy
    shover      wave = Shove whenever present
    wave:<slot> wave forced to a slot for the first `turns` turns of the branch (an opponent PLAN, not a style)
Scripted references (`m44.references`) are available as `ref:<name>`.

Note on forced behaviours: forcing the behaviour component replaces the policy's Recall mass by the forced behaviour on
those turns (Recall is a behaviour); every behaviour style therefore acts only in its situation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ....champions import in_contact
from ....config import B_HOLD, B_SAFE, W_HOLD, W_SHOVE
from ....diagnostics.branch import Commit, observation
from ..m44.interventions import ComponentAgent
from ..m44.records import holds_freeze
from . import plans as P


def frozen_on(obs) -> bool:
    """The enemy holds a freeze on this side (its wave is held next to the enemy tower)."""
    return holds_freeze(obs.state, 1 - obs.side)


def hp_frac(obs) -> float:
    from ....champions_m4 import context
    t = context(obs.cfg).t
    me = obs.champ.side(obs.side)
    return me.hp / t.hp_max[t.level_by_xp[min(me.xp, t.xp_cap)]]


def enemy_in_reach(obs) -> bool:
    me, en = obs.champ.side(obs.side), obs.champ.side(1 - obs.side)
    return me.absent == 0 and en.absent == 0 and in_contact(me.pos, en.pos, obs.cfg)


def style(agent, name: str):
    """The learned agent with one component forced in a situation (module docstring); "policy" = the agent itself."""
    if name == "policy" or name == "modal":
        return agent
    if name == "breaker":
        return ComponentAgent(agent, "wave", lambda obs: W_SHOVE if frozen_on(obs) else None, name=f"{agent.name}|breaker")
    if name == "passive":
        return ComponentAgent(agent, "wave", lambda obs: W_HOLD if frozen_on(obs) else None, name=f"{agent.name}|passive")
    if name == "stepback":
        return ComponentAgent(agent, "behavior", lambda obs: B_SAFE if frozen_on(obs) else None,
                              name=f"{agent.name}|stepback")
    if name == "ignore":
        return ComponentAgent(agent, "behavior", lambda obs: B_HOLD if (enemy_in_reach(obs) and hp_frac(obs) > 0.4)
                              else None, name=f"{agent.name}|ignore")
    if name == "respect":
        return ComponentAgent(agent, "behavior", lambda obs: B_SAFE if enemy_in_reach(obs) else None,
                              name=f"{agent.name}|respect")
    if name == "shover":
        return ComponentAgent(agent, "wave", lambda obs: W_SHOVE, name=f"{agent.name}|shover")
    if name.startswith("ref:"):
        from ..m44.references import reference_suite
        return reference_suite()[name[4:]]
    raise ValueError(f"unknown opponent style {name!r}")


def modal_plan(agent, cfg, snap, opp_seat: int) -> Commit:
    oo = observation(cfg, snap, opp_seat)
    po = np.asarray(agent.distribution(oo), dtype=float)
    return Commit(int(oo.legal[int(np.argmax(po))]), "opponent modal")


def wave_plan(slot: int, turns: int, name: str) -> P.StrategicPlan:
    """The opponent's wave forced to `slot` for the first `turns` turns of the branch (behaviour / objective: its own
    conditional distribution), then its continuation agent."""
    return P.StrategicPlan(name, P.const_rule(P.force({"wave": slot}, name)), max_turns=turns, on_absent="continue")


def behavior_plan(beh: int, turns: int, name: str) -> P.StrategicPlan:
    return P.StrategicPlan(name, P.const_rule(P.force({"behavior": beh}, name)), max_turns=turns, on_absent="continue")


def setup(protocol: str, agent, cfg, snap, focal: int, opp_plans: Optional[dict] = None) -> Tuple[tuple, object]:
    """(continuation agents of seats A, B; opponent plan) for an opponent protocol / style name."""
    o = 1 - focal
    plan = None
    opp = agent
    if protocol == "modal":
        plan = modal_plan(agent, cfg, snap, o)
    elif opp_plans and protocol in opp_plans:
        plan = opp_plans[protocol]()
    else:
        opp = style(agent, protocol)
    agents = (agent, opp) if focal == 0 else (opp, agent)
    return agents, plan


__all__ = ["frozen_on", "enemy_in_reach", "style", "modal_plan", "wave_plan", "behavior_plan", "setup"]
