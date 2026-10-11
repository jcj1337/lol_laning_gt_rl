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

Duel edge: E_i = h_i ((1 - q_i) dmg(1, L_i, N_i) + q_i dmg(2, L_i, N_i)) is i's expected damage per trade (hit and
double probabilities of M2 against a Hold defender); edge = E_me / hp_enemy - E_enemy / hp_me.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..champions import champion_legal_actions, trade_double_probability, trade_hit_probability
from ..config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, F, NONE, PRESSURE, SH, SP, VISION, champ_act)
from ..progression import tables
from .core import Observation

ABSENT_ONLY = (CHAMP_ABSENT,)
RECALL_CODE = champ_act(SP, B_RECALL)


def duel_edge(obs: Observation) -> float:
    """Positive when the observer expects to win trades (relative to the remaining HP of both)."""
    cfg = obs_cfg(obs)
    t = tables(cfg)
    lane, m4, s = obs.state, obs.champ, obs.side
    me, en = m4.side(s), m4.side(1 - s)
    if en.absent > 0 or me.hp <= 0 or en.hp <= 0:
        return 0.0
    e = []
    for i, x in ((s, me), (1 - s, en)):
        h = trade_hit_probability(lane, i, B_HOLD, False, cfg)
        q = trade_double_probability(i, cfg)
        L = t.level_by_xp[min(x.xp, t.xp_cap)]
        e.append(h * ((1 - q) * t.dmg[1][L][x.items] + q * t.dmg[2][L][x.items]))
    return e[0] / en.hp - e[1] / me.hp


def obs_cfg(obs: Observation):
    """The configuration behind an observation (LaneEnv attaches it; skills enter the duel edge)."""
    if obs.cfg is None:
        raise ValueError("an M4 agent needs Observation.cfg (observations built by LaneEnv carry it)")
    return obs.cfg


class M4ScriptAgent:
    def __init__(self, name: str, wave: int = SP, behavior: int = B_HOLD, objective: int = PRESSURE,
                 freeze: bool = False, recall_frac: float = 2 / 6, gold_recall: bool = False,
                 gold_hp_frac: float = 1.0, edge_trade: bool = False, freeze_zone: bool = False,
                 xp_safe: bool = False, cshove: bool = False, stack_hp: int = 0):
        self.name, self.wave, self.behavior, self.objective = name, wave, behavior, objective
        self.freeze, self.recall_frac, self.gold_recall, self.gold_hp_frac = freeze, recall_frac, gold_recall, gold_hp_frac
        self.edge_trade, self.freeze_zone, self.xp_safe = edge_trade, freeze_zone, xp_safe
        self.cshove, self.stack_hp = cshove, stack_hp

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _first(legal, candidates) -> int:
        for a in candidates:
            if a in legal:
                return a
        raise ValueError(f"no candidate is legal in {legal}")

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
        if self.gold_recall and me.items < t.item_tiers and me.gold >= t.prices[me.items] \
                and me.hp <= self.gold_hp_frac * H and not enemy_wave_threat:
            return self._walk_back(legal)
        freeze_legal = any(a % 3 == F for a in legal if a != CHAMP_ABSENT)
        # wave
        wave = self.wave
        if self.cshove:
            wave = SH if (en_fh == 0 or own_fh >= en_fh) else SP
        if self.stack_hp and own_fh >= self.stack_hp:
            wave = SH
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
