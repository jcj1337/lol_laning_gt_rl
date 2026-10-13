"""Scripted strategies of the M4.3d study: what a champion does after its wave crashed into the enemy tower.

`Strategy4d` is the M4.3c four-action `Strategy` (the same pre-rules, in the same order: ABSENT while away; Recall at
40% HP when legal and not in the enemy's reach, else walk back with Safe; a free reset while the enemy is away; the
optional Recall rule; then behaviour, wave and objective) with two extensions that use existing actions only:

    behaviour "leave"   Safe while the own crashed minions are under the enemy tower (back off after crashing: the
                        champion walks to its own -1 at most, never ahead of its wave), Hold otherwise
    objective           "none" | "pressure" (Apply Pressure whenever legal, as M4.3c's pressure=True) | "vision"
                        (Establish Vision whenever legal and the own vision timer is 0) | "vision_pressure" (Vision when
                        the timer is 0, else Pressure: spend one siege turn on vision, then pressure under it)

Objectives are legal only with Hold or Trade, only while the own crashed mass is under the enemy tower and the
champion can reach that tower this turn (champions.objective_reachable); otherwise the strategy plays the same code
without an objective. Every decision is a function of (cfg, lane, champions, side, turn index), so a recording replays
from its actions.

The attacker set (A) of the crash grid. Every one Shoves every turn except the control `lh`:

    lh           Last Hit, Hold: the control (never crashes on purpose)
    sh_leave     Shove; while the own wave is crashed, back off (Safe)               "Shove and leave"
    sh_recall    Shove; while the own wave is crashed, walk back and Recall           "Shove and Recall"
    sh_stay      Shove, Hold, no objective: stays with its next wave                  "Shove and stay"
    sh_press     Shove, Hold + Pressure: stands in the enemy tower cell while it sieges, never trades   "remain forward"
    sh_harass    Shove + Pressure, Trade an enemy that starts within reach without a dive (M4.3c shove_punish4)
    sh_hunt      Shove + Pressure, chase and Trade while the duel edge is positive, no dives (M4.3c shove_hunt4)
    sh_dive      Shove + Pressure; while the own wave is crashed, Trade: dives an enemy in its tower cell
                                                                                      "Shove and dive"
    sh_dive_edge sh_dive, but Trade only while the duel edge (zoning_natural.duel_edge) is positive
    sh_vis       Shove, Hold + Vision then Pressure                                   "Vision, then pressure"
    sh_vis_dive  Shove; while crashed: Hold + Vision when no vision is active (a connected trade would suspend the
                 objective), else Trade + Pressure                                    "Vision, then dive"
    sh_bully     Shove + Pressure, Trade every turn the enemy is present, in the lane too (M4.3c freeze_bully with
                 Shove; the first M4.3d run's "sh_dive")

behaviour "siege_dive" / "siege_dive_edge" / "vision_dive" (the three rules above) are Hold outside the own siege and
while the enemy is away.

The defender set (B): M4.3c library strategies

    lh           Last Hit, Hold (farm4: farms the crash in its tower cell)
    thin         Thin, Hold (thin4: defends the wave)
    safe         Safe, Last Hit (step_back4: never ahead of the own -1)
    respect      Safe while the enemy starts within reach, else Yield (respect4)
    punish       Trade an enemy that starts within reach, no dives (punish4: hits a sieger standing in its tower cell)
    fight        Thin, Trade back while in reach with a non-negative duel edge (contest4)
    shove        Shove + Pressure (walk_up_shove4: contest by shoving back)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ...champions import CHAMP_ABSENT, champion_legal_actions
from ...champions_m4 import context
from ...config import B_HOLD, B_SAFE, B_TRADE, NONE, PRESSURE, VISION, W_HOLD, champ_act, wave_display_names
from ..m4 import zoning_natural as zn
from ..m43c import strategies as st

OBJECTIVES = ("none", "pressure", "vision", "vision_pressure")


class Strategy4d(st.Strategy):
    """st.Strategy with the behaviour "leave" and the objective modes of the module docstring."""

    OWN_BEHAVIORS = ("leave", "siege_dive", "siege_dive_edge", "vision_dive")

    def __init__(self, name: str, behavior: str = "hold", wave: str = "last_hit", objective: str = "none",
                 description: str = "", **kw):
        self.own = behavior if behavior in self.OWN_BEHAVIORS else None
        self.leave = behavior == "leave"
        super().__init__(name, behavior="hold" if self.own else behavior, wave=wave,
                         pressure=objective != "none", description=description, **kw)
        if objective not in OBJECTIVES:
            raise ValueError(f"unknown objective mode {objective!r}")
        self.objective = objective

    def _behavior(self, cfg, lane, m4, s: int) -> int:
        own = self.own
        if own is None:
            return st.behavior_rule(self.behavior, cfg, lane, m4, s)
        sieging = lane.siege_of(s) > 0
        if own == "leave":
            return B_SAFE if sieging else B_HOLD
        if not sieging or m4.side(1 - s).absent > 0:
            return B_HOLD
        if own == "siege_dive_edge" and zn.duel_edge(cfg, lane, m4, s) <= 0:
            return B_HOLD
        if own == "vision_dive" and lane.vision_of(s) == 0:
            return B_HOLD                    # place the ward first: a connected trade would suspend the objective
        return B_TRADE

    def _objectives(self, lane, s: int, beh: int) -> Tuple[int, ...]:
        if self.objective == "none" or beh not in (B_HOLD, B_TRADE):
            return (NONE,)
        if self.objective == "pressure":
            return (PRESSURE, NONE)
        no_vision = lane.vision_of(s) == 0
        if self.objective == "vision":
            return (VISION, NONE) if no_vision else (NONE,)
        return (VISION, PRESSURE, NONE) if no_vision else (PRESSURE, NONE)

    def decide(self, cfg, lane, m4, s: int, i: Optional[int] = None) -> Tuple[int, str]:
        legal = champion_legal_actions(lane, m4, s, cfg)
        if legal == (CHAMP_ABSENT,):
            return CHAMP_ABSENT, "absent"
        me, en, t = m4.side(s), m4.side(1 - s), context(cfg).t
        H = t.hp_max[t.level_by_xp[min(me.xp, t.xp_cap)]]
        reset = en.absent > 0 and me.hp <= zn.RESET_FRAC * H and lane.siege_of(1 - s) == 0
        if me.hp <= self.recall_frac * H or reset:
            why = "low HP" if me.hp <= self.recall_frac * H else "free reset (enemy away)"
            if self._can_recall(cfg, m4, s, legal):
                return st.RECALL, f"{why}: Recall"
            return st.WALK_BACK, f"{why}: walk back (Safe)"
        why_r = self._recall_wanted(cfg, lane, m4, s, i)
        if why_r is not None:
            if self._can_recall(cfg, m4, s, legal):
                return st.RECALL, f"{why_r}: Recall"
            return st.WALK_BACK, f"{why_r}: walk back to Recall (Safe)"
        beh = self._behavior(cfg, lane, m4, s)
        rule = self.wave_name(i)
        wave, why_w = st.wave_rule(rule, cfg, lane, m4, s, self.shove_hp)
        names = wave_display_names(cfg)
        intent = f"{names.get(wave, wave)} ({rule}{': ' + why_w if why_w else ''})"
        for o in self._objectives(lane, s, beh):
            c = champ_act(wave, beh, o)
            if c in legal:
                tag = {PRESSURE: " + Pressure", VISION: " + Vision"}.get(o, "")
                return c, intent + tag + (" (leave: Safe while crashed)" if self.leave and beh == B_SAFE else "")
        c = champ_act(W_HOLD, B_HOLD)
        return c, f"{intent} not legal -> fallback {cfg.label(c)}"


def S(name: str, **kw) -> Strategy4d:
    return Strategy4d(name, **kw)


ATTACKERS: Dict[str, st.Strategy] = {s.name: s for s in [
    S("lh", description="Last Hit, Hold: the control (never crashes on purpose)."),
    S("sh_leave", behavior="leave", wave="shove", description="Shove; back off (Safe) while the own wave is crashed."),
    S("sh_recall", wave="shove", recall="crash", description="Shove; walk back and Recall while the own wave is "
                                                             "crashed."),
    S("sh_stay", wave="shove", description="Shove, Hold, no objective: stay with the next wave."),
    S("sh_press", wave="shove", objective="pressure",
      description="Shove, Hold + Pressure: stand in the enemy tower cell while sieging, never trade."),
    S("sh_harass", behavior="punish", wave="shove", objective="pressure",
      description="Shove + Pressure; Trade an enemy that starts within reach (no dives)."),
    S("sh_hunt", behavior="hunt", wave="shove", objective="pressure",
      description="Shove + Pressure; chase and Trade while the duel edge is positive (no dives)."),
    S("sh_dive", behavior="siege_dive", wave="shove", objective="pressure",
      description="Shove + Pressure; while the own wave is crashed, Trade (dives an enemy in its tower cell)."),
    S("sh_dive_edge", behavior="siege_dive_edge", wave="shove", objective="pressure",
      description="sh_dive, but Trade only while the duel edge is positive."),
    S("sh_vis", wave="shove", objective="vision_pressure",
      description="Shove, Hold; Vision when none is active, else Pressure."),
    S("sh_vis_dive", behavior="vision_dive", wave="shove", objective="vision_pressure",
      description="Shove; while crashed: Hold + Vision when none is active, else Trade + Pressure (a dive under "
                  "vision)."),
    S("sh_bully", behavior="trade", wave="shove", objective="pressure",
      description="Shove + Pressure; Trade every turn the enemy is present, in the lane too (dives when needed)."),
]}

DEFENDERS: Dict[str, st.Strategy] = {
    "lh": st.LIBRARY["farm4"], "thin": st.LIBRARY["thin4"], "safe": st.LIBRARY["step_back4"],
    "respect": st.LIBRARY["respect4"], "punish": st.LIBRARY["punish4"], "fight": st.LIBRARY["contest4"],
    "shove": st.LIBRARY["walk_up_shove4"],
}


def lookup(name: str) -> st.Strategy:
    """A strategy by name: the attacker set, then the M4.3c library (and its phase strategies)."""
    if name in ATTACKERS:
        return ATTACKERS[name]
    if name.startswith("d:"):
        return DEFENDERS[name[2:]]
    return st.LIBRARY[name]


__all__ = ["Strategy4d", "ATTACKERS", "DEFENDERS", "lookup", "OBJECTIVES"]
