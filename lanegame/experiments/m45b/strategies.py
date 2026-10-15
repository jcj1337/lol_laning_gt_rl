"""Contingent scripted strategies of M4.5b (memoryless functions of the start-of-turn state; nothing here changes the
game).

`Contingent` extends the M4.3c four-action `Strategy` vocabulary (its wave rules and behaviour rules, the same low-HP
pre-rule) with phase-dependent rules, so a single strategy can switch between keeping a freeze, trading, shoving,
recalling and warding according to the lane, HP and power:

  phases       freezing = s holds a freeze (the enemy front at s's own -1 or deeper, both fronts fighting); frozen =
               the enemy holds one on s; otherwise the default phase. Each phase has its own wave and behaviour rule.
  pre-rules    in this order, each only when it applies:
               1 absent: ABSENT
               2 low HP (<= recall_frac of max HP), or a free reset (enemy away, HP <= reset_frac, no enemy siege):
                 Recall when legal and out of the enemy's start-of-turn contact (recall_in_contact: anyway), else walk
                 back with Safe (M4.3c rule)
               3 shove_recall h: at HP <= h, Shove until the own wave has crashed (own mass under the enemy tower) or has
                 no front, then Recall (walk back with Safe while Recall is not legal or the enemy is in contact)
               4 frozen_reset h: while frozen on and HP <= h, Recall (reset to contest the freeze at full HP)
               5 ward rule (WARD_RULES), when Ward is legal and the own vision is down
  behaviours   every M4.3c rule (hold, trade, punish, edge, standoff, respect, safe, tower, cautious, fight_back, hunt)
               plus
                 guard      Trade when the enemy is present, catchable this turn without a dive (its contact cell is
                            within champion_speed and not the enemy tower cell), standing at own cell <= trade_line
                            (default: anywhere), own HP >= trade_hp of max and the duel edge >= edge_min; else Hold
                 nodive     Trade when the enemy is catchable without a dive; else Hold (a bully that never dives)
                 wary       Safe when the enemy is present, can catch us this turn without diving, and its duel edge
                            on us is positive; else Hold
                 wary_yield as wary with Yield
  waves        every M4.3c rule (last_hit, thin, slow_push, shove, build_shove, grow_shove, shove_away, thin_last)

The duel edge is `zoning_natural.duel_edge` (expected trade damage per hit as a share of the victim's CURRENT HP,
attacker minus defender; minion fire is not in it). Every decision is a function of (cfg, lane, champions, side), so a
recording replays from its actions.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ...champions import CHAMP_ABSENT, champion_legal_actions
from ...champions_m4 import contact_cell_target, context
from ...config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, CHAMP_WARD, NONE, PRESSURE, W_HOLD, W_SHOVE,
                       GameConfig, champ_act, wave_display_names)
from ...positional import WARD, dist as graph_dist, mirror_cell
from ..m4 import zoning_natural as zn
from ..m43c import strategies as st

RECALL = st.RECALL
WALK_BACK = st.WALK_BACK
WARD_RULES = (None, "favorable", "always", "away", "forward", "exposed")
NEW_BEHAVIORS = ("guard", "nodive", "wary", "wary_yield")


def present(m4, s: int) -> bool:
    return m4.side(s).absent == 0


def hp_frac(cfg: GameConfig, m4, s: int) -> float:
    t = context(cfg).t
    me = m4.side(s)
    return me.hp / t.hp_max[t.level_by_xp[min(me.xp, t.xp_cap)]]


def catchable(cfg: GameConfig, m4, att: int) -> bool:
    """Can side `att`'s champion reach a contact cell of the other champion's START cell this turn without entering
    the enemy tower cell (a dive)? Both present."""
    me, en = m4.side(att), m4.side(1 - att)
    if me.absent or en.absent:
        return False
    cell = contact_cell_target(me.pos, en.pos, cfg)
    return cell != cfg.tower and graph_dist(cell, me.pos) <= cfg.champion_speed


def enemy_cell(m4, s: int) -> Optional[int]:
    """The enemy champion's cell in s's own coordinates (W stays W); None while away."""
    en = m4.side(1 - s)
    return None if en.absent else mirror_cell(en.pos)


def freezing(lane, s: int) -> bool:
    return st.held_at_own_tower(lane, s)


def frozen(lane, s: int) -> bool:
    return st.held_at_own_tower(lane, 1 - s)


class Contingent:
    """A phase-dependent scripted strategy (module docstring)."""

    def __init__(self, name: str, wave: str = "last_hit", beh: str = "hold", wave_freeze: Optional[str] = None,
                 beh_freeze: Optional[str] = None, wave_frozen: Optional[str] = None, beh_frozen: Optional[str] = None,
                 recall_frac: float = zn.RECALL_FRAC, reset_frac: float = zn.RESET_FRAC, trade_hp: float = 0.0,
                 edge_min: float = 0.0, trade_line: Optional[int] = None, shove_recall: Optional[float] = None,
                 frozen_reset: Optional[float] = None, ward: Optional[str] = None, pressure: bool = False,
                 recall_in_contact: bool = False, description: str = ""):
        for r in (wave, wave_freeze, wave_frozen):
            if r is not None and r not in st.WAVE_RULES:
                raise ValueError(f"unknown wave rule {r!r}")
        for r in (beh, beh_freeze, beh_frozen):
            if r is not None and r not in st.BEHAVIOR_RULES + NEW_BEHAVIORS:
                raise ValueError(f"unknown behaviour rule {r!r}")
        if ward not in WARD_RULES:
            raise ValueError(f"unknown ward rule {ward!r}")
        self.name, self.description = name, description
        self.wave, self.beh = wave, beh
        self.wave_freeze, self.beh_freeze = wave_freeze or wave, beh_freeze or beh
        self.wave_frozen, self.beh_frozen = wave_frozen or wave, beh_frozen or beh
        self.recall_frac, self.reset_frac = float(recall_frac), float(reset_frac)
        self.trade_hp, self.edge_min, self.trade_line = float(trade_hp), float(edge_min), trade_line
        self.shove_recall, self.frozen_reset = shove_recall, frozen_reset
        self.ward, self.pressure, self.recall_in_contact = ward, bool(pressure), bool(recall_in_contact)

    def __repr__(self) -> str:
        return self.name

    # ------------------------------------------------------------------ rules
    def _recall_or_back(self, cfg, m4, s, legal, why: str) -> Tuple[int, str]:
        if RECALL in legal and (self.recall_in_contact or not zn.contact_at_start(cfg, m4, s)):
            return RECALL, f"{why}: Recall"
        return WALK_BACK, f"{why}: walk back (Safe)"

    def behavior(self, rule: str, cfg, lane, m4, s: int) -> int:
        if rule == "guard":
            if not catchable(cfg, m4, s):
                return B_HOLD
            ec = enemy_cell(m4, s)
            if self.trade_line is not None and (ec == WARD or ec > self.trade_line):
                return B_HOLD
            if hp_frac(cfg, m4, s) < self.trade_hp or zn.duel_edge(cfg, lane, m4, s) < self.edge_min:
                return B_HOLD
            return B_TRADE
        if rule == "nodive":
            return B_TRADE if catchable(cfg, m4, s) else B_HOLD
        if rule in ("wary", "wary_yield"):
            if catchable(cfg, m4, 1 - s) and zn.duel_edge(cfg, lane, m4, 1 - s) > 0:
                return B_SAFE if rule == "wary" else B_YIELD
            return B_HOLD
        return st.behavior_rule(rule, cfg, lane, m4, s)

    def ward_wanted(self, cfg, lane, m4, s: int, legal) -> Optional[str]:
        r = self.ward
        if r is None or not cfg.ward_node or CHAMP_WARD not in legal or lane.vision_of(s) > 0:
            return None
        if r == "always":
            return "vision down: Ward"
        if r == "away":
            return "vision down, the enemy is away: Ward" if not present(m4, 1 - s) else None
        ex, eh = lane.side(1 - s)[:2]
        fx, fh = lane.side(s)[:2]
        if r == "favorable":
            return "vision down, enemy wave not in our half: Ward" if (eh <= 0 or -ex >= 0) else None
        if r == "forward":            # about to play forward: the own front is at the lane cell or beyond
            return "vision down, own wave forward: Ward" if (fh > 0 and fx >= 0) or lane.siege_of(s) > 0 else None
        if r == "exposed":            # forward AND the enemy wave not in our half (no route through it)
            fwd = (fh > 0 and fx >= 0) or lane.siege_of(s) > 0
            return "vision down, forward, enemy wave not in our half: Ward" if fwd and (eh <= 0 or -ex >= 0) else None
        return None

    def decide(self, cfg: GameConfig, lane, m4, s: int, i: Optional[int] = None) -> Tuple[int, str]:
        legal = champion_legal_actions(lane, m4, s, cfg)
        if legal == (CHAMP_ABSENT,):
            return CHAMP_ABSENT, "absent"
        me, en = m4.side(s), m4.side(1 - s)
        h = hp_frac(cfg, m4, s)
        reset = en.absent > 0 and h <= self.reset_frac and lane.siege_of(1 - s) == 0
        if h <= self.recall_frac or reset:
            return self._recall_or_back(cfg, m4, s, legal, "low HP" if h <= self.recall_frac else
                                        "free reset (enemy away)")
        if self.shove_recall is not None and h <= self.shove_recall:
            fx, fh = lane.side(s)[:2]
            if lane.siege_of(s) > 0 or fh <= 0:
                return self._recall_or_back(cfg, m4, s, legal, f"HP {h:.2f}: own wave crashed")
            c = champ_act(W_SHOVE, B_HOLD)
            if c in legal:
                return c, f"HP {h:.2f}: Shove before recalling"
        if self.frozen_reset is not None and frozen(lane, s) and h <= self.frozen_reset:
            return self._recall_or_back(cfg, m4, s, legal, f"frozen on at HP {h:.2f}: reset")
        why_w = self.ward_wanted(cfg, lane, m4, s, legal)
        if why_w:
            return CHAMP_WARD, why_w
        if freezing(lane, s):
            phase, wr, br = "freezing", self.wave_freeze, self.beh_freeze
        elif frozen(lane, s):
            phase, wr, br = "frozen", self.wave_frozen, self.beh_frozen
        else:
            phase, wr, br = "", self.wave, self.beh
        beh = self.behavior(br, cfg, lane, m4, s)
        wave, why = st.wave_rule(wr, cfg, lane, m4, s)
        names = wave_display_names(cfg)
        intent = (f"[{phase}] " if phase else "") + f"{names.get(wave, wave)} ({wr}{': ' + why if why else ''}), {br}"
        objs = (PRESSURE, NONE) if self.pressure and beh in (B_HOLD, B_TRADE) else (NONE,)
        for o in objs:
            c = champ_act(wave, beh, o)
            if c in legal:
                return c, intent + (" + Pressure" if o == PRESSURE else "")
        c = champ_act(W_HOLD, B_HOLD)
        return c, f"{intent} not legal -> fallback {cfg.label(c)}"

    def act(self, cfg: GameConfig, lane, m4, s: int, i: Optional[int] = None) -> int:
        return self.decide(cfg, lane, m4, s, i)[0]


def C(name: str, **kw) -> Contingent:
    return Contingent(name, **kw)


# ---------------------------------------------------------------------------------------------- the library
# Freezer (strong-side) strategies: the default phase rebuilds a freeze with Last Hit (or Shoves, "_sh")
FREEZERS: Dict[str, Contingent] = {p.name: p for p in [
    C("fz_passive", wave_freeze="thin", description="Keep the freeze (Thin + Hold), never trade; Last Hit + Hold "
                                                    "otherwise."),
    C("fz_lh", wave_freeze="last_hit", description="Keep the freeze with Last Hit + Hold only (no correction)."),
    C("fz_guard", wave_freeze="thin", beh_freeze="guard",
      description="Keep the freeze; Trade a catchable enemy (no dive) while own HP >= 50% and the duel edge >= 0."
                  " Last Hit + Hold otherwise.", trade_hp=0.5),
    C("fz_guard_line", wave_freeze="thin", beh_freeze="guard", trade_hp=0.5, trade_line=-1,
      description="fz_guard, but only against an enemy standing at our -1 or deeper (a walk-up)."),
    C("fz_guard_all", wave_freeze="thin", beh_freeze="guard", beh="guard", trade_hp=0.5,
      description="fz_guard, and the same guard rule outside the freeze."),
    C("fz_bully", wave_freeze="thin", beh_freeze="nodive", beh="nodive",
      description="Keep the freeze; Trade whenever the enemy is catchable without a dive, in every phase."),
    C("fz_guard_sr", wave_freeze="thin", beh_freeze="guard", trade_hp=0.5, shove_recall=0.55,
      description="fz_guard; at HP <= 55% Shove the wave in, then Recall while it crashes."),
    C("fz_passive_sr", wave_freeze="thin", shove_recall=0.55,
      description="fz_passive; at HP <= 55% Shove the wave in, then Recall."),
    C("fz_guard_sh", wave_freeze="thin", beh_freeze="guard", wave="shove", trade_hp=0.5, pressure=True,
      description="fz_guard while the freeze holds; Shove (+ Pressure) once it is broken."),
    C("shove_recall", wave="shove", shove_recall=0.55, pressure=True,
      description="Shove every turn (+ Pressure); at HP <= 55% Recall once the wave has crashed."),
    C("fz_passive_aw", wave="shove_away", wave_freeze="thin", pressure=True,
      description="fz_passive; while the enemy is away Shove (+ Pressure), else Thin."),
    C("fz_guard_aw", wave="shove_away", wave_freeze="thin", beh_freeze="guard", trade_hp=0.5, pressure=True,
      description="fz_guard; while the enemy is away Shove (+ Pressure), else Thin (convert a kill into a crash)."),
    C("fz_guard_aw_sr", wave="shove_away", wave_freeze="thin", beh_freeze="guard", trade_hp=0.5, pressure=True,
      shove_recall=0.55, description="fz_guard_aw; at HP <= 55% Shove the wave in, then Recall."),
    # the freeze-phase rule only, Shove (+ Pressure) outside the freeze (the M4.3c phase-strategy convention)
    C("fz_passive_sh", wave="shove", wave_freeze="thin", pressure=True,
      description="Keep the freeze passively (Thin + Hold); Shove (+ Pressure) outside it."),
    C("fz_guard_sh_sr", wave="shove", wave_freeze="thin", beh_freeze="guard", trade_hp=0.5, pressure=True,
      shove_recall=0.55, description="fz_guard_sh; at HP <= 55% Shove the wave in, then Recall."),
    C("fz_guard_line_sh", wave="shove", wave_freeze="thin", beh_freeze="guard", trade_hp=0.5, trade_line=-1,
      pressure=True, description="Keep the freeze, Trade only a walk-up (enemy at our -1 or deeper); Shove outside."),
    C("fz_bully_sh", wave="shove", wave_freeze="thin", beh_freeze="nodive", pressure=True,
      description="Keep the freeze, Trade whenever the enemy is catchable without a dive; Shove outside."),
]}
FREEZERS["fz_guard_sh"].description = "Keep the freeze, guard it (fz_guard); Shove (+ Pressure) once it is broken."


def _with_shove_default(p: Contingent) -> Contingent:
    """A weak-side response with Shove (+ Pressure) outside the frozen phase instead of Last Hit."""
    q = Contingent(p.name + "_sh", wave="shove", beh=p.beh, wave_frozen=p.wave_frozen, beh_frozen=p.beh_frozen,
                   recall_frac=p.recall_frac, reset_frac=p.reset_frac, frozen_reset=p.frozen_reset, pressure=True,
                   description=p.description + " Shove (+ Pressure) outside the frozen phase.")
    return q

# Weak-side (frozen-on) responses: the default phase is Last Hit + Hold
RESPONSES: Dict[str, Contingent] = {p.name: p for p in [
    C("wk_shove", wave_frozen="shove", description="Frozen on: walk up and Shove (Shove + Hold)."),
    C("wk_shove_safe", wave_frozen="shove", beh_frozen="safe", description="Frozen on: Shove + Safe."),
    C("wk_lh", description="Frozen on: walk up and Last Hit (Hold)."),
    C("wk_back", beh_frozen="safe", description="Frozen on: step back (Last Hit + Safe)."),
    C("wk_yield", beh_frozen="standoff", description="Frozen on: Last Hit + Yield (stand off)."),
    C("wk_contest", wave_frozen="thin", beh_frozen="fight_back",
      description="Frozen on: Thin, Trade back while in reach with a non-negative duel edge."),
    C("wk_wary_shove", wave_frozen="shove", beh_frozen="wary",
      description="Frozen on: Shove, but Safe while the freezer can catch us with a positive duel edge."),
    C("wk_shove_reset", wave_frozen="shove", frozen_reset=0.6,
      description="Frozen on: Shove; Recall below 60% HP to contest again at full HP."),
    C("wk_back_reset", beh_frozen="safe", frozen_reset=0.6, description="Frozen on: step back; reset below 60%."),
]}
RESPONSES.update({q.name: q for q in (_with_shove_default(p) for p in list(RESPONSES.values()))})

# General lane strategies (both seats; the integrated matrix)
GENERAL: Dict[str, Contingent] = {p.name: p for p in [
    C("g_farm", description="Last Hit + Hold everywhere."),
    C("g_thin", wave="thin", description="Thin + Hold everywhere (freeze keeper)."),
    C("g_slow", wave="slow_push", pressure=True, description="Slow Push + Hold (+ Pressure)."),
    C("g_shove", wave="shove", pressure=True, description="Shove + Hold (+ Pressure)."),
    C("g_shove_sr", wave="shove", pressure=True, shove_recall=0.55,
      description="Shove (+ Pressure); Recall below 55% once the wave crashed."),
    C("g_guard", wave="thin", beh="guard", trade_hp=0.5,
      description="Thin; Trade a catchable enemy (no dive) at HP >= 50% with a non-negative duel edge."),
    C("g_bully", beh="nodive", description="Last Hit; Trade whenever the enemy is catchable without a dive."),
    C("g_safe", beh="safe", description="Last Hit + Safe everywhere."),
    C("g_wary", beh="wary", description="Last Hit; Safe while a stronger enemy can catch us, else Hold."),
    C("g_wary_yield", beh="wary_yield", description="Last Hit; Yield while a stronger enemy can catch us, else Hold."),
    C("g_yield", beh="standoff", description="Last Hit + Yield everywhere."),
    C("g_ward", ward="favorable", description="g_farm + Ward when vision is down and the enemy wave is not in our "
                                              "half."),
    C("g_shove_ward", wave="shove", pressure=True, ward="exposed",
      description="g_shove + Ward when vision is down, the own wave is forward and the enemy wave not in our half."),
    C("g_guard_ward", wave="thin", beh="guard", trade_hp=0.5, ward="favorable",
      description="g_guard + Ward when vision is down and the enemy wave is not in our half."),
    # Shove + selective trading hybrids (the learned M4.4b aggro policy beats plain Shove mainly by selective trades)
    C("g_shove_guard", wave="shove", beh="guard", trade_hp=0.5, pressure=True,
      description="Shove (+ Pressure); Trade a catchable enemy (no dive) at HP >= 50% with a non-negative duel edge."),
    C("g_shove_guard_sr", wave="shove", beh="guard", trade_hp=0.5, pressure=True, shove_recall=0.55,
      description="g_shove_guard; below 55% HP Recall once the wave has crashed."),
    C("g_shove_edge", wave="shove", beh="guard", trade_hp=0.6, edge_min=0.05, pressure=True,
      description="Shove (+ Pressure); Trade only with a clearly positive duel edge (>= 0.05) at HP >= 60%."),
    C("g_shove_wary", wave="shove", beh="wary", pressure=True,
      description="Shove (+ Pressure); Safe while a stronger enemy can catch us."),
    C("g_slow_guard", wave="slow_push", beh="guard", trade_hp=0.5, pressure=True,
      description="Slow Push (+ Pressure); Trade a catchable enemy at HP >= 50% with a non-negative duel edge."),
    C("g_shove_guard_ward", wave="shove", beh="guard", trade_hp=0.5, pressure=True, ward="exposed",
      description="g_shove_guard + Ward when vision is down, the own wave forward, the enemy wave not in our half."),
]}

LIB: Dict[str, Contingent] = {**FREEZERS, **RESPONSES, **GENERAL}


def get(name: str):
    """A strategy by name: this library, then the M4.3c LIBRARY (four-action strategies), then the M4.5a Ward
    strategies."""
    if name in LIB:
        return LIB[name]
    if name in st.LIBRARY:
        return st.LIBRARY[name]
    from ..m45a.scenarios import STRATEGIES as WS
    return WS[name]


__all__ = ["Contingent", "C", "FREEZERS", "RESPONSES", "GENERAL", "LIB", "get", "catchable", "hp_frac", "freezing",
           "frozen", "present", "enemy_cell", "WARD_RULES"]
