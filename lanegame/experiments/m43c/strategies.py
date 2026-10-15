"""Scripted four-action strategies (M4.3c stabilization; docs/m4/m4_3/m4_3c_strategic_economics_calibration.md).

A `Strategy` is a `zoning_natural.Policy` (the same pre-rules, in the same order: ABSENT while away; at or below
`recall_frac` of max HP Recall when legal and not in the enemy's reach at the start of the turn, else walk back with
Safe; a free reset while the enemy is away and own HP is below RESET_FRAC) whose wave component comes from the
four-action WAVE_RULES below, optionally switched by turn (`schedule`), with an optional Recall rule (`recall`) and the
zoning_natural behaviour rules plus two conditional ones (BEHAVIOR_RULES). Nothing here changes the game; every
decision is a function of (cfg, lane, champions, side, turn index i), so a recording replays from its actions.

Wave rules (slot names of the four-action model: Last Hit / Thin / Slow Push / Shove):

    last_hit     Last Hit every turn
    thin         Thin every turn: 1 HP unless the enemy wave is the stronger, then the correction (the freeze keeper)
    slow_push    Slow Push every turn
    shove        Shove every turn
    build_shove  Slow Push until the own front holds `shove_hp` HP, then Shove (a deliberate crash)
    grow_shove   Slow Push -> Last Hit -> Shove: Slow Push while the own blob is not larger than the enemy's, Last Hit
                 while the own blob leads (the lead compounds by itself), Shove once the own front holds `shove_hp` HP
    shove_away   Shove while the enemy champion is away (dead or recalled), else Thin
    thin_last    Thin while the enemy blob is larger than the own blob, else Last Hit (Thin -> Last Hit, explicit)

Recall rules (`recall`, applied after the low-HP pre-rule and before the behaviour; all memoryless): None; "crash"
(walk back and Recall while the own crashed minions are under the enemy tower, the champion is not already in its
tower cell and no enemy front stands at the own -1 or deeper: the viewer's crash_recall rule); "freeze" (Recall while
the enemy front is held at the own -1 or deeper with at most `freeze_hp` HP, the own front alive and at least
`recall_gold` resources unspent: the freeze keeps the enemy wave next to the own tower; unspent gold is 0 after the
return, so it does not recall again at once); ("at", t) (Recall on turn t, or on turn t + 1 if it had to walk back
first: a scripted timing; on turn t + 1 a champion that recalled at t is absent, so it recalls once).

Behaviour rules: zoning_natural's hold, trade, punish, edge, standoff, respect, safe, tower, plus
    cautious     Safe while the enemy starts the turn within reach and the own duel edge is negative, else Hold
    fight_back   Trade while the enemy starts within reach and the own duel edge is not negative, else Hold
    hunt         Trade while the own duel edge is positive and the enemy can be caught this turn without a dive (its
                 start cell has a contact cell within champion_speed that is not the enemy tower cell), else Hold:
                 the power-aware chaser. zoning_natural's punish and edge need contact at the START of the turn, which a
                 champion standing in its own tower cell never has with an enemy one cell ahead of it (the tower-cell
                 contact rule), so a freezer can only threaten a walk-up by chasing

The inherited low-HP pre-rule never recalls while the enemy champion starts the turn in contact (an enemy Trade would
interrupt the channel). No behaviour moves a champion into its own tower cell, so a low-HP champion at its own -1
next to an enemy at the lane cell walks back to -1 and stays there, in contact, without recalling and without wave
damage, until something moves (M4.3c stabilization §A3). `recall_in_contact=True` recalls anyway when Recall is
legal, accepting the interruption risk; the default keeps the historical rule so earlier studies stay comparable.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from ...champions import CHAMP_ABSENT, champion_legal_actions
from ...champions_m4 import context
from ...config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, FOUR_WAVE_SHORT, NONE, PRESSURE, W_HOLD, W_SHOVE,
                       W_SLOWPUSH, W_THIN, GameConfig, champ_act, wave_display_names)
from ...mechanics import _blob_hp, start_packets
from ..m4 import zoning_natural as zn

LH, TN, SL, SH = W_HOLD, W_THIN, W_SLOWPUSH, W_SHOVE
SHORT4 = dict(FOUR_WAVE_SHORT)                     # LH, TN, SL, SH
RECALL = champ_act(W_HOLD, B_RECALL)
WALK_BACK = champ_act(W_HOLD, B_SAFE)
WAVE_RULES = ("last_hit", "thin", "slow_push", "shove", "build_shove", "grow_shove", "shove_away", "thin_last")
BEHAVIOR_RULES = ("hold", "trade", "punish", "edge", "standoff", "respect", "safe", "tower", "cautious", "fight_back",
                  "hunt")
SHOVE_HP = 20
FREEZE_HP = 14


def blobs(cfg: GameConfig, lane, s: int) -> Tuple[int, int]:
    """(own blob, enemy blob) HP at the start of the turn: the front plus the packets queued within one cell."""
    P = start_packets(lane, cfg)
    return _blob_hp(P[s], cfg), _blob_hp(P[1 - s], cfg)


def held_at_own_tower(lane, s: int) -> bool:
    """The enemy front stands at side s's own -1 or deeper while s's own front is alive (a freeze by s)."""
    ex, eh = lane.side(1 - s)[:2]
    fh = lane.side(s)[1]
    return eh > 0 and fh > 0 and -ex <= -1


def wave_rule(rule: str, cfg: GameConfig, lane, m4, s: int, shove_hp: int = SHOVE_HP) -> Tuple[int, str]:
    """(wave slot, why) of a four-action wave rule (module docstring)."""
    if rule == "last_hit":
        return LH, ""
    if rule == "thin":
        return TN, ""
    if rule == "slow_push":
        return SL, ""
    if rule == "shove":
        return SH, ""
    own_fh = lane.side(s)[1]
    if rule == "build_shove":
        return (SH, f"own front {own_fh} >= {shove_hp}: crash it") if own_fh >= shove_hp else (SL, "build")
    if rule == "grow_shove":
        if own_fh >= shove_hp:
            return SH, f"own front {own_fh} >= {shove_hp}: crash it"
        ob, eb = blobs(cfg, lane, s)
        return (LH, f"lead {ob}>{eb}: let it grow") if ob > eb else (SL, f"no lead ({ob}<={eb}): nudge")
    if rule == "shove_away":
        return (SH, "enemy away: shove") if m4.side(1 - s).absent > 0 else (TN, "")
    if rule == "thin_last":
        ob, eb = blobs(cfg, lane, s)
        return (TN, f"enemy blob {eb} > {ob}: thin") if eb > ob else (LH, "")
    raise ValueError(f"unknown wave rule {rule!r}")


def behavior_rule(rule: str, cfg: GameConfig, lane, m4, s: int) -> int:
    """zoning_natural.behavior_rule plus the conditional rules of the module docstring."""
    if rule == "hunt":
        from ...champions_m4 import contact_cell_target
        me, en = m4.side(s), m4.side(1 - s)
        if en.absent > 0 or zn.duel_edge(cfg, lane, m4, s) <= 0:
            return B_HOLD
        cell = contact_cell_target(me.pos, en.pos, cfg)
        return B_TRADE if abs(cell - me.pos) <= cfg.champion_speed and cell < cfg.tower else B_HOLD
    if rule in ("cautious", "fight_back"):
        en = m4.side(1 - s)
        if en.absent > 0 or not zn.contact_at_start(cfg, m4, s):
            return B_HOLD
        edge = zn.duel_edge(cfg, lane, m4, s)
        if rule == "cautious":
            return B_SAFE if edge < 0 else B_HOLD
        return B_TRADE if edge >= 0 and en.pos > -cfg.tower else B_HOLD
    return zn.behavior_rule(rule, cfg, lane, m4, s)


class Strategy(zn.Policy):
    """A four-action scripted strategy (module docstring). `schedule` = ((turn, wave rule), ...) in increasing turn
    order replaces `wave` from each listed turn index on; `pressure` adds Apply Pressure whenever it is legal with the
    chosen behaviour (Hold or Trade)."""

    def __init__(self, name: str, behavior: str = "hold", wave: str = "last_hit", pressure: bool = False,
                 recall_frac: float = zn.RECALL_FRAC, schedule: Sequence[Tuple[int, str]] = (),
                 recall=None, shove_hp: int = SHOVE_HP, freeze_hp: int = FREEZE_HP, recall_gold: float = 2.0,
                 recall_in_contact: bool = False, description: str = ""):
        super().__init__(name, behavior, wave, pressure, recall_frac)
        if behavior not in BEHAVIOR_RULES:
            raise ValueError(f"unknown behaviour rule {behavior!r}")
        for _, r in ((0, wave),) + tuple(schedule):
            if r not in WAVE_RULES:
                raise ValueError(f"unknown wave rule {r!r}")
        self.schedule = tuple((int(t), str(r)) for t, r in schedule)
        if list(self.schedule) != sorted(self.schedule):
            raise ValueError("schedule phases must be in increasing turn order")
        if not (recall in (None, "crash", "freeze") or (isinstance(recall, tuple) and recall[0] == "at")):
            raise ValueError(f"unknown recall rule {recall!r}")
        self.recall = recall
        self.shove_hp, self.freeze_hp, self.recall_gold = int(shove_hp), int(freeze_hp), float(recall_gold)
        self.recall_in_contact = bool(recall_in_contact)
        self.description = description

    def _can_recall(self, cfg, m4, s: int, legal) -> bool:
        return RECALL in legal and (self.recall_in_contact or not zn.contact_at_start(cfg, m4, s))

    def wave_name(self, i: Optional[int]) -> str:
        rule = self.wave
        if self.schedule:
            if i is None:
                raise ValueError(f"strategy {self.name!r} has a turn schedule: pass the turn index i")
            for t, r in self.schedule:
                if i >= t:
                    rule = r
        return rule

    def _recall_wanted(self, cfg, lane, m4, s: int, i: Optional[int]) -> Optional[str]:
        r = self.recall
        if r is None:
            return None
        ex, eh = lane.side(1 - s)[:2]
        if r == "crash":
            threat = eh > 0 and -ex <= -1
            if lane.siege_of(s) > 0 and m4.side(s).pos > -cfg.tower and not threat:
                return "own wave crashed"
            return None
        if r == "freeze":
            gold = m4.side(s).gold / cfg.gold_scale
            if held_at_own_tower(lane, s) and eh <= self.freeze_hp and gold >= self.recall_gold:
                return f"freeze holds (enemy front {eh} HP at own -1), {gold:.1f} gold to invest"
            return None
        if i is None:
            raise ValueError(f"strategy {self.name!r} recalls at a turn: pass the turn index i")
        return f"scheduled recall (turn {r[1]})" if i in (r[1], r[1] + 1) else None

    def decide(self, cfg: GameConfig, lane, m4, s: int, i: Optional[int] = None) -> Tuple[int, str]:
        """(action code, why): the rule that fired, and the fallback when the intended code was not legal."""
        legal = champion_legal_actions(lane, m4, s, cfg)
        if legal == (CHAMP_ABSENT,):
            return CHAMP_ABSENT, "absent"
        me, en, t = m4.side(s), m4.side(1 - s), context(cfg).t
        H = t.hp_max[t.level_by_xp[min(me.xp, t.xp_cap)]]
        reset = en.absent > 0 and me.hp <= zn.RESET_FRAC * H and lane.siege_of(1 - s) == 0
        if me.hp <= self.recall_frac * H or reset:
            why = "low HP" if me.hp <= self.recall_frac * H else "free reset (enemy away)"
            if self._can_recall(cfg, m4, s, legal):
                return RECALL, f"{why}: Recall"
            return WALK_BACK, f"{why}: walk back (Safe)"
        why_r = self._recall_wanted(cfg, lane, m4, s, i)
        if why_r is not None:
            if self._can_recall(cfg, m4, s, legal):
                return RECALL, f"{why_r}: Recall"
            return WALK_BACK, f"{why_r}: walk back to Recall (Safe)"
        beh = behavior_rule(self.behavior, cfg, lane, m4, s)
        rule = self.wave_name(i)
        wave, why_w = wave_rule(rule, cfg, lane, m4, s, self.shove_hp)
        names = wave_display_names(cfg)
        intent = f"{names.get(wave, wave)} ({rule}{': ' + why_w if why_w else ''})"
        objs = (PRESSURE, NONE) if self.pressure and beh in (B_HOLD, B_TRADE) else (NONE,)
        for o in objs:
            c = champ_act(wave, beh, o)
            if c in legal:
                return c, intent + (" + Pressure" if o == PRESSURE else "")
        c = champ_act(W_HOLD, B_HOLD)
        return c, f"{intent} not legal -> fallback {cfg.label(c)}"

    def act(self, cfg: GameConfig, lane, m4, s: int, i: Optional[int] = None) -> int:
        return self.decide(cfg, lane, m4, s, i)[0]


class StrategyAgent:
    """A `Strategy` as an environment agent (LaneEnv observations). The turn index i counts turns since the agent's
    `reset()` (evaluate_discounted's game loop and this study's loops call it per game) and advances on `notify`;
    the strategy itself is memoryless."""

    def __init__(self, strategy: Strategy, name: Optional[str] = None):
        self.strategy, self.name = strategy, name or strategy.name
        self.i = 0

    def reset(self) -> None:
        self.i = 0

    def notify(self, info: dict, side: int) -> None:
        self.i += 1

    def choose_action(self, obs) -> int:
        a = self.strategy.act(obs.cfg, obs.state, obs.champ, obs.side, self.i)
        if a not in obs.legal:
            raise ValueError(f"{self.name}: strategy chose illegal action {a} (legal {obs.legal})")
        return a

    def distribution(self, obs):
        import numpy as np
        p = np.zeros(len(obs.legal))
        p[obs.legal.index(self.choose_action(obs))] = 1.0
        return p

    def act(self, obs, rng=None) -> int:
        return self.choose_action(obs)


class Phased:
    """A phase strategy: `inner` while the phase condition `when` holds for this side, `default` otherwise (both
    Strategies; the same decide / act interface). Phases (memoryless, from the start-of-turn lane):

        frozen_on   the enemy front holds our wave at its tower: our front at our own +1 or deeper with the enemy's
                    front alive (the enemy keeps a freeze on us)
        freezing    we hold the enemy front at our own -1 or deeper with our front alive (we keep a freeze)

    It answers "what to do DURING a freeze" with everything else equal (the same default outside the phase)."""

    PHASES = ("frozen_on", "freezing")

    def __init__(self, name: str, when: str, inner: Strategy, default: Strategy, description: str = ""):
        if when not in self.PHASES:
            raise ValueError(f"unknown phase {when!r}")
        self.name, self.when, self.inner, self.default, self.description = name, when, inner, default, description

    def active(self, cfg, lane, m4, s: int) -> bool:
        return held_at_own_tower(lane, 1 - s) if self.when == "frozen_on" else held_at_own_tower(lane, s)

    def decide(self, cfg: GameConfig, lane, m4, s: int, i: Optional[int] = None) -> Tuple[int, str]:
        if self.active(cfg, lane, m4, s):
            c, why = self.inner.decide(cfg, lane, m4, s, i)
            return c, f"[{self.when}: {self.inner.name}] {why}"
        return self.default.decide(cfg, lane, m4, s, i)

    def act(self, cfg: GameConfig, lane, m4, s: int, i: Optional[int] = None) -> int:
        return self.decide(cfg, lane, m4, s, i)[0]


def S(name: str, **kw) -> Strategy:
    return Strategy(name, **kw)


# The named strategies of the stabilization study (both seats; the frozen-on / weaker side is usually B)
LIBRARY: Dict[str, Strategy] = {s.name: s for s in [
    S("farm4", description="Last Hit and Hold position: farm the lane, never push by power."),
    S("thin4", wave="thin", description="Thin every turn and Hold position: last-hit, correct an enemy surplus "
                                         "(keeps a freeze)."),
    S("slowpush4", wave="slow_push", pressure=True,
      description="Slow Push every turn: build a 1-HP window-strength lead and let it compound; Pressure while "
                  "sieging."),
    S("shove4", wave="shove", pressure=True, description="Shove every turn (full capacity); Pressure while sieging."),
    S("build_shove4", wave="build_shove", pressure=True,
      description="Slow Push until the own front holds 20 HP, then Shove it into the tower; Pressure while sieging."),
    S("grow_shove4", wave="grow_shove", pressure=True,
      description="Slow Push -> Last Hit -> Shove: nudge until ahead, let the lead grow, Shove at 20 HP; Pressure "
                  "while sieging."),
    S("grow_crash_recall4", wave="grow_shove", pressure=False, recall="crash",
      description="Slow Push -> Last Hit -> Shove, then walk back and Recall while the crash occupies the enemy."),
    S("freeze4", wave="thin", description="Freeze keeper: Thin every turn at the farm spot (Last Hit unless the "
                                           "enemy wave is the stronger)."),
    S("freeze_punish4", behavior="punish", wave="thin",
      description="Freeze keeper + punish: Trade an enemy that starts the turn within reach (no dives)."),
    S("freeze_edge4", behavior="edge", wave="thin",
      description="Freeze keeper + Trade when in reach and the duel edge is positive."),
    S("freeze_hunt4", behavior="hunt", wave="thin",
      description="Freeze keeper + hunt: chase and Trade whenever the duel edge is positive and no dive is needed."),
    S("freeze_bully4", behavior="trade", wave="thin",
      description="Freeze keeper + Trade every turn the enemy is present (chases)."),
    S("hunt4", behavior="hunt", description="Last Hit + hunt (chase and Trade while the duel edge is positive)."),
    S("shove_hunt4", behavior="hunt", wave="shove", pressure=True,
      description="Shove + hunt + Pressure: crash every wave, chase when stronger."),
    S("freeze_recall4", wave="thin", recall="freeze",
      description="Freeze keeper that Recalls while the freeze holds the enemy wave next to its tower."),
    S("walk_up4", description="Walk up and last-hit (Hold position, Last Hit)."),
    S("walk_up_shove4", wave="shove", pressure=True,
      description="Walk up and Shove: break the wave state with full capacity; Pressure while sieging."),
    S("walk_up_thin4", wave="thin", description="Walk up and Thin defensively."),
    S("step_back4", behavior="safe", description="Step back: Safe (never ahead of the own -1), Last Hit."),
    S("respect4", behavior="respect", description="Safe while the enemy starts within reach, else Yield; Last Hit."),
    S("cautious4", behavior="cautious",
      description="Hold, but Safe while the enemy is in reach and the duel edge is against us; Last Hit."),
    S("contest4", behavior="fight_back", wave="thin",
      description="Contest: Thin, Trade back while in reach with a non-negative duel edge."),
    S("bully4", behavior="trade", description="Trade every turn the enemy is present; Last Hit."),
    S("punish4", behavior="punish", description="Trade an enemy that starts within reach (no dives); Last Hit."),
    S("shove_punish4", behavior="punish", wave="shove", pressure=True,
      description="Shove + punish + Pressure (the M4.3b clear_punish line in four-action terms)."),
    S("sp_then_shove4", wave="slow_push", schedule=((10, "shove"),), pressure=True,
      description="Slow Push for 10 turns, then Shove (adapts once a gradual push is neutralized)."),
]}


# Phase strategies of the freeze games: the freeze-specific rule only while the freeze holds, Shove (the baseline
# equilibrium strategy of every restricted game, calibration report §4) otherwise
def _ph(name: str, when: str, inner: str, description: str) -> Phased:
    return Phased(name, when, LIBRARY[inner], LIBRARY["shove4"], description)


PHASE_LIBRARY: Dict[str, Phased] = {p.name: p for p in [
    _ph("pz_keep", "freezing", "freeze4", "Keep the freeze while it holds (Thin, Hold); Shove otherwise."),
    _ph("pz_hunt", "freezing", "freeze_hunt4", "Keep the freeze and hunt while it holds; Shove otherwise."),
    _ph("pz_bully", "freezing", "freeze_bully4", "Keep the freeze and Trade every turn while it holds; Shove "
                                                 "otherwise."),
    _ph("pz_walk", "frozen_on", "walk_up4", "Frozen on: walk up and last-hit; Shove otherwise."),
    _ph("pz_thin", "frozen_on", "walk_up_thin4", "Frozen on: walk up and Thin; Shove otherwise."),
    _ph("pz_back", "frozen_on", "step_back4", "Frozen on: step back (Safe); Shove otherwise."),
    _ph("pz_cautious", "frozen_on", "cautious4", "Frozen on: Hold, but Safe while a stronger enemy is in reach; Shove "
                                                 "otherwise."),
    _ph("pz_contest", "frozen_on", "contest4", "Frozen on: Thin and trade back when not weaker; Shove otherwise."),
]}
LIBRARY.update(PHASE_LIBRARY)


__all__ = ["Strategy", "Phased", "StrategyAgent", "LIBRARY", "PHASE_LIBRARY", "WAVE_RULES", "BEHAVIOR_RULES",
           "SHORT4", "wave_rule", "behavior_rule", "blobs", "held_at_own_tower"]
