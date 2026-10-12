"""Scripted agents for the M4 ruleset (plan WP8). NumPy only.

M2's scripted agents compare HP with integer thresholds in M2 units (6 HP); under M4 HP is in centi-HP and the
maximum grows with the level, so the M4 agents state every HP rule as a FRACTION of the current maximum H(L).
Every agent is deterministic and decides from its observation (one-hot `distribution`), except `AdaptiveBreaker`,
which also remembers whether it has seen the opponent maintain a Freeze (from the public turn results the harness
passes to `notify`).

    M4ScriptAgent        the configurable rule agent below
    m4_suite()           hold_farmer, aggressive_trader, freezer, freeze_breaker, xp_keeper, gold_recaller, presser
    gate_defenders()     the S3b defenders: freeze, slowpush, cshove (identical behavior rules, different waves)
    gate_attackers()     the S3b attackers: park_hold, park_safe, breaker, presser, stacker, adaptive_breaker

Rules of M4ScriptAgent, applied in order (each optional):
  1. absent -> ABSENT;
  2. low-HP recall: hp <= recall_frac * H(L): Recall when legal, else walk back (Slow Push + Safe);
  3. gold recall (`gold_recall`): when the next tier is affordable, hp <= gold_hp_frac * H(L) and the own wave is not
     under threat (no enemy front at own x <= -1): Recall when legal, else walk back;
  4. behavior: the base behavior; `edge_trade`: Trade when the duel edge is positive (below) and the enemy is
     present; `freeze_zone`: while freezing, Hold if the farm spot (own front cell) is within farm_reach of the
     freeze line, else Safe (so the Freeze stays maintained); `xp_safe`: Safe while the enemy can maintain a Freeze
     (Freeze legal for it and its champion within farm_reach of its freeze line);
  5. wave: Freeze whenever legal (`freeze`); conditional Shove (`cshove`: Shove while the own front holds at least
     the enemy front's HP or the enemy has none); stacking (`stack_hp`: Shove once the own front holds that much);
  6. objective: the preferred siege objective whenever it is legal.
`crash_recall` (between rules 3 and 4): while the own wave's crashed minions are under the enemy tower, the champion
is not already back at its own tower cell (so it does not Recall again on its return) and no enemy front threatens
the own half, walk back and Recall.

Natural waves (GameConfig.m4_natural, M4.2e-A): the wave slots are Hold (SP), Clear (SH) and Thin (F), all always
legal. `NaturalWaveAgent` replaces rule 5 by one of the mechanical wave rules of `NATURAL_WAVE_RULES`; `natural_suite()`
is its scripted population. The agents above still run under natural waves, with the slot meanings changed
(e.g. `freezer` Thins whenever it is in its freeze zone, because Thin is always legal).

Duel edge: E_i = h_i ((1 - q_i) dmg(1, L_i, N_i) + q_i dmg(2, L_i, N_i)) is i's expected damage per trade (hit and
double probabilities of M2 against a Hold defender); edge = E_me / hp_enemy - E_enemy / hp_me.

M4.2c continuous investment (item_model = "continuous"): the duel edge uses the invested-gold damage
(`M4Context.hit`), and "the next tier is affordable" (rule 3, ShoveRecallAgent) becomes "unspent gold >=
invest_gold resources" (default 4.0, the baseline's first tier price). No rule reads the opponent's investment
except through the duel edge.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..champions import champion_legal_actions, trade_double_probability, trade_hit_probability
from ..champions_m4 import context
from ..config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, F, NONE, PRESSURE, SH, SP, VISION, W_CLEAR,
                      W_HOLD, W_THIN, champ_act)
from ..progression import tables
from .core import Observation

ABSENT_ONLY = (CHAMP_ABSENT,)
RECALL_CODE = champ_act(SP, B_RECALL)


def duel_edge(obs: Observation) -> float:
    """Positive when the observer expects to win trades (relative to the remaining HP of both)."""
    cfg = obs_cfg(obs)
    ctx = context(cfg)
    t = ctx.t
    lane, m4, s = obs.state, obs.champ, obs.side
    me, en = m4.side(s), m4.side(1 - s)
    if en.absent > 0 or me.hp <= 0 or en.hp <= 0:
        return 0.0
    e = []
    for i, x in ((s, me), (1 - s, en)):
        h = trade_hit_probability(lane, i, B_HOLD, False, cfg)
        q = trade_double_probability(i, cfg)
        L = t.level_by_xp[min(x.xp, t.xp_cap)]
        e.append(h * ((1 - q) * ctx.hit(1, L, x.items) + q * ctx.hit(2, L, x.items)))
    return e[0] / en.hp - e[1] / me.hp


def cash_ready(me, cfg, t, invest_gold: float) -> bool:
    """The gold-driven Recall trigger: the next tier is affordable (tiers), or the unspent gold reaches
    `invest_gold` resources (continuous investment)."""
    if cfg.continuous_items:
        return me.gold > 0 and me.gold >= invest_gold * cfg.gold_scale
    return me.items < t.item_tiers and me.gold >= t.prices[me.items]


def obs_cfg(obs: Observation):
    """The configuration behind an observation (LaneEnv attaches it; skills enter the duel edge)."""
    if obs.cfg is None:
        raise ValueError("an M4 agent needs Observation.cfg (observations built by LaneEnv carry it)")
    return obs.cfg


class M4ScriptAgent:
    def __init__(self, name: str, wave: int = SP, behavior: int = B_HOLD, objective: int = PRESSURE,
                 freeze: bool = False, recall_frac: float = 2 / 6, gold_recall: bool = False,
                 gold_hp_frac: float = 1.0, edge_trade: bool = False, freeze_zone: bool = False,
                 xp_safe: bool = False, cshove: bool = False, stack_hp: int = 0, invest_gold: float = 4.0,
                 crash_recall: bool = False):
        self.name, self.wave, self.behavior, self.objective = name, wave, behavior, objective
        self.freeze, self.recall_frac, self.gold_recall, self.gold_hp_frac = freeze, recall_frac, gold_recall, gold_hp_frac
        self.edge_trade, self.freeze_zone, self.xp_safe = edge_trade, freeze_zone, xp_safe
        self.cshove, self.stack_hp = cshove, stack_hp
        self.invest_gold = invest_gold
        self.crash_recall = crash_recall

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _first(legal, candidates) -> int:
        for a in candidates:
            if a in legal:
                return a
        raise ValueError(f"no candidate is legal in {legal}")

    def _wave(self, lane, s: int, own_fx: int, own_fh: int, en_fx: int, en_fh: int) -> int:
        """Rule 5 without the Freeze override: the base wave, conditional Shove, stacking."""
        wave = self.wave
        if self.cshove:
            wave = SH if (en_fh == 0 or own_fh >= en_fh) else SP
        if self.stack_hp and own_fh >= self.stack_hp:
            wave = SH
        return wave

    def _walk_back(self, legal) -> int:
        return RECALL_CODE if RECALL_CODE in legal else champ_act(SP, B_SAFE)

    def choose_action(self, obs: Observation) -> int:
        legal = obs.legal
        if legal == ABSENT_ONLY:
            return CHAMP_ABSENT
        cfg = obs_cfg(obs)
        t = tables(cfg)
        lane, m4, s = obs.state, obs.champ, obs.side
        me, en = m4.side(s), m4.side(1 - s)
        L = t.level_by_xp[min(me.xp, t.xp_cap)]
        H = t.hp_max[L]
        own_fx, own_fh = lane.side(s)[:2]
        en_fx, en_fh = lane.side(1 - s)[:2]
        enemy_wave_threat = en_fh > 0 and -en_fx <= -1
        if me.hp <= self.recall_frac * H:
            return self._walk_back(legal)
        if self.gold_recall and cash_ready(me, cfg, t, self.invest_gold) \
                and me.hp <= self.gold_hp_frac * H and not enemy_wave_threat:
            return self._walk_back(legal)
        if self.crash_recall and lane.siege_of(s) > 0 and me.pos > -cfg.tower and not enemy_wave_threat:
            return self._walk_back(legal)
        freeze_legal = any(a % 3 == F for a in legal if a != CHAMP_ABSENT)
        # wave
        wave = self._wave(lane, s, own_fx, own_fh, en_fx, en_fh)
        freezing = self.freeze and freeze_legal
        if freezing:
            wave = F
        # behavior
        beh = self.behavior
        if self.xp_safe:
            en_freeze_legal = own_fh > 0 and own_fx >= 1
            en_in_zone = en.absent == 0 and abs(en.pos - cfg.freeze_line) <= cfg.farm_reach
            beh = B_SAFE if (en_freeze_legal and en_in_zone) else B_HOLD
        if self.freeze_zone and freezing:
            spot = max(-cfg.tower, min(cfg.tower, own_fx)) if own_fh > 0 else me.pos
            beh = B_HOLD if abs(spot - cfg.freeze_line) <= cfg.farm_reach else B_SAFE
        if self.edge_trade and en.absent == 0 and duel_edge(obs) > 0:
            beh = B_TRADE
        objs = (self.objective, NONE) if beh in (B_HOLD, B_TRADE) else (NONE,)
        cands = [champ_act(w, beh, o) for w in (wave, SP) for o in objs]
        cands += [champ_act(SP, B_HOLD)]
        return self._first(legal, cands)

    def distribution(self, obs: Observation) -> np.ndarray:
        p = np.zeros(len(obs.legal))
        p[obs.legal.index(self.choose_action(obs))] = 1.0
        return p

    def act(self, obs: Observation, rng=None) -> int:
        return self.choose_action(obs)


class AdaptiveBreaker(M4ScriptAgent):
    """S3b attacker: parks (Slow Push + Hold) until it has seen the opponent maintain a Freeze, then Shoves and
    Applies Pressure for the rest of the game. `notify(info, side)` receives each turn's public result (the
    environment's `info`; Freeze maintenance is public after the turn). `reset()` forgets (call per game)."""

    def __init__(self, name: str = "adaptive_breaker", **kw):
        super().__init__(name, SP, B_HOLD, PRESSURE, **kw)
        self.saw_freeze = False

    def reset(self) -> None:
        self.saw_freeze = False

    def notify(self, info: dict, side: int) -> None:
        if info["freeze_maintained"][1 - side]:
            self.saw_freeze = True

    def choose_action(self, obs: Observation) -> int:
        self.wave = SH if self.saw_freeze else SP
        return super().choose_action(obs)


class ShoveRecallAgent(M4ScriptAgent):
    """Conditional Shove -> Recall -> Purchase cycle (M4.2b calibration, Phase D). Deterministic and stateless: every
    decision is a function of the observation, so it is safe for the rollout probability cache. In order:

      1. absent -> ABSENT; HP <= recall_frac * H(L) -> Recall (walk back first when not legal);
      2. the next tier is affordable ("cash out"):
         a. an enemy front at own x <= -1 threatens the tower -> stay: Slow Push + Hold (clear it, do not leave);
         b. the own wave has crashed (siege) or its front stands at x >= 1, or the lane is empty -> Recall;
         c. otherwise, at HP >= shove_hp_frac * H with a favourable wave (own front HP >= the enemy front's, or no
            enemy front) -> Shove + Hold (push first, then recall at 2b);
         d. otherwise Recall when HP <= cashout_hp_frac * H, else farm (Slow Push + Hold) and reassess next turn;
      3. not affordable: farm (Slow Push + Hold); with `finish_push`, a front already standing at x >= 1 is shoved
         through (crashed) rather than left parked under the enemy tower, where the pusher farms at the enemy-tower
         efficiency while the defender farms at its own tower; build the wave and Shove once the own front holds
         >= stack_hp HP;
         with `pressure`, Apply Pressure on the turns it Shoves at HP >= shove_hp_frac * H.

    It never Shoves permanently: after the purchase it returns to rule 3. No rule looks at the opponent's identity."""

    def __init__(self, name: str = "shove_recall", shove_hp_frac: float = 2 / 3, cashout_hp_frac: float = 1 / 2,
                 stack_hp: int = 8, pressure: bool = False, finish_push: bool = True, recall_frac: float = 2 / 6,
                 invest_gold: float = 4.0):
        super().__init__(name, SP, B_HOLD, NONE, recall_frac=recall_frac, invest_gold=invest_gold)
        self.finish_push = finish_push
        self.shove_hp_frac, self.cashout_hp_frac, self.stack_hp_cycle, self.pressure = (shove_hp_frac, cashout_hp_frac,
                                                                                         stack_hp, pressure)

    def decide(self, obs: Observation) -> tuple:
        """(action, rule label) for the observation; the label names the rule that fired (diagnostics)."""
        legal = obs.legal
        if legal == ABSENT_ONLY:
            return CHAMP_ABSENT, "absent"
        cfg = obs_cfg(obs)
        t = tables(cfg)
        lane, m4, s = obs.state, obs.champ, obs.side
        me = m4.side(s)
        H = t.hp_max[t.level_by_xp[min(me.xp, t.xp_cap)]]
        own_fx, own_fh = lane.side(s)[:2]
        en_fx, en_fh = lane.side(1 - s)[:2]
        if me.hp <= self.recall_frac * H:
            return self._walk_back(legal), "low_hp_recall"
        farm = self._first(legal, [champ_act(SP, B_HOLD)])
        shove = self._first(legal, [champ_act(SH, B_HOLD), champ_act(SP, B_HOLD)])
        afford = cash_ready(me, cfg, t, self.invest_gold)
        if afford:
            if en_fh > 0 and -en_fx <= -1:
                return farm, "stay_threat"
            if lane.siege_of(s) > 0 or (own_fh > 0 and own_fx >= 1) or (own_fh == 0 and en_fh == 0):
                return self._walk_back(legal), "recall_pushed"
            if me.hp >= self.shove_hp_frac * H and (en_fh == 0 or own_fh >= en_fh):
                return shove, "shove_before_recall"
            if me.hp <= self.cashout_hp_frac * H:
                return self._walk_back(legal), "recall_unpushed"
            return farm, "wait"
        if own_fh > 0 and own_fx >= 1 and self.finish_push:
            return shove, "finish_push"
        if own_fh >= self.stack_hp_cycle:
            if self.pressure and me.hp >= self.shove_hp_frac * H:
                return self._first(legal, [champ_act(SH, B_HOLD, PRESSURE), champ_act(SH, B_HOLD),
                                           champ_act(SP, B_HOLD)]), "stack_shove_pressure"
            return shove, "stack_shove"
        return farm, "farm"

    def choose_action(self, obs: Observation) -> int:
        return self.decide(obs)[0]


def m4_suite() -> Dict[str, M4ScriptAgent]:
    """The M4 scripted baselines of plan WP8 (each with the low-HP recall rule at 1/3 of max HP unless noted)."""
    agents = [M4ScriptAgent("hold_farmer", SP, B_HOLD),
              M4ScriptAgent("aggressive_trader", SP, B_TRADE, recall_frac=1 / 6),
              M4ScriptAgent("freezer", SP, B_HOLD, freeze=True, freeze_zone=True, edge_trade=True),
              M4ScriptAgent("freeze_breaker", SH, B_HOLD, objective=NONE),
              M4ScriptAgent("xp_keeper", SP, B_HOLD, xp_safe=True),
              M4ScriptAgent("gold_recaller", SP, B_HOLD, gold_recall=True, gold_hp_frac=0.75),
              M4ScriptAgent("presser", SH, B_HOLD, PRESSURE)]
    return {a.name: a for a in agents}


def gate_defenders() -> Dict[str, M4ScriptAgent]:
    """S3b defenders (design §6.6): identical behavior rules (Hold, low-HP recall, Trade on a positive duel edge),
    different wave rules: freeze = Freeze when legal; slowpush = Slow Push; cshove = conditional Shove."""
    common = dict(behavior=B_HOLD, objective=PRESSURE, edge_trade=True)
    return {"freeze": M4ScriptAgent("freeze", SP, freeze=True, **common),
            "slowpush": M4ScriptAgent("slowpush", SP, **common),
            "cshove": M4ScriptAgent("cshove", SP, cshove=True, **common)}


def gate_attackers() -> Dict[str, M4ScriptAgent]:
    """S3b attackers (design §6.6, plan S3b), each with the low-HP recall rule; plus `trader` (Slow Push + Trade), an
    extra attacker reported separately so that trades and the tower-adjacent shot are exercised."""
    return {"park_hold": M4ScriptAgent("park_hold", SP, B_HOLD, NONE),
            "park_safe": M4ScriptAgent("park_safe", SP, B_SAFE, NONE),
            "breaker": M4ScriptAgent("breaker", SH, B_HOLD, NONE),
            "presser": M4ScriptAgent("presser", SH, B_HOLD, PRESSURE),
            "stacker": M4ScriptAgent("stacker", SP, B_HOLD, PRESSURE, stack_hp=8),
            "adaptive_breaker": AdaptiveBreaker(),
            "trader": M4ScriptAgent("trader", SP, B_TRADE, NONE)}


# ---------------------------------------------------------------------- M4.2e-A natural waves
NATURAL_WAVE_RULES = ("hold", "thin", "clear", "slow_push", "freeze", "build_clear")


class NaturalWaveAgent(M4ScriptAgent):
    """M4ScriptAgent whose wave rule (rule 5) is mechanical, for wave_dynamics "natural" (lanegame.config.W_HOLD etc.):

      hold         Hold
      thin         Thin
      clear        Clear
      slow_push    Thin while the own front is no more than `margin` HP ahead of the enemy front it fights, else Hold:
                   a small edge, then the reinforcements and the Lanchester law do the rest
      freeze       Thin while the enemy front holds at least `margin` HP more than the own front, else Hold: the enemy
                   surplus stays bounded, so the contact drifts toward the own tower and stays there
      build_clear  Thin to build; Clear once the own front holds at least `margin` HP (a deliberate crash)

    Every rule plays Hold while the fronts are not in contact."""

    def __init__(self, name: str, wave_rule: str = "hold", margin: int = 8, **kw):
        if wave_rule not in NATURAL_WAVE_RULES:
            raise ValueError(f"unknown wave_rule {wave_rule!r}")
        super().__init__(name, W_HOLD, **kw)
        self.wave_rule, self.margin = wave_rule, margin

    def _wave(self, lane, s: int, own_fx: int, own_fh: int, en_fx: int, en_fh: int) -> int:
        r = self.wave_rule
        if r == "hold":
            return W_HOLD
        if r == "thin":
            return W_THIN
        if r == "clear":
            return W_CLEAR
        if not (own_fh > 0 and en_fh > 0 and own_fx + en_fx >= -1):
            return W_HOLD
        if r == "slow_push":
            return W_THIN if own_fh <= en_fh + self.margin else W_HOLD
        if r == "freeze":
            return W_THIN if en_fh - own_fh >= self.margin else W_HOLD
        return W_CLEAR if own_fh >= self.margin else W_THIN          # build_clear


def natural_suite() -> Dict[str, M4ScriptAgent]:
    """Scripted population for natural waves: one agent per wave rule, all farming (Hold behavior) with Pressure when
    sieging, plus a crash recaller and a trading slow pusher."""
    agents = [NaturalWaveAgent("hold_farmer", "hold"),
              NaturalWaveAgent("thinner", "thin"),
              NaturalWaveAgent("clearer", "clear"),
              NaturalWaveAgent("slow_pusher", "slow_push", margin=2),
              NaturalWaveAgent("natural_freezer", "freeze", margin=8),
              NaturalWaveAgent("crasher", "build_clear", margin=16),
              NaturalWaveAgent("crash_recaller", "slow_push", margin=2, crash_recall=True),
              NaturalWaveAgent("trading_pusher", "slow_push", margin=2, behavior=B_TRADE, edge_trade=True)]
    return {a.name: a for a in agents}
