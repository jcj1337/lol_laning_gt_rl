"""M4.2e-B: combat-driven zoning on natural waves (docs/natural_waves/m4_2e_zoning_report.md).

The question: can a credible combat threat make a weaker champion surrender position, farm or wave control when that
is rational, without a farming penalty that merely activates because a stronger champion is nearby?

Tools (nothing is trained):

  1. `decompose`: the one-turn concession inequality of the M4.2d-C study, re-derived on natural waves. From one
     state, every pair of lane behaviours is played for one turn and as a stationary pair for HORIZON turns with no
     chance events (u = None); every chance event enters as its exact expectation. The farm of the weaker champion B
     is attributed to reach, posture and trade factors (`gold_breakdown`), so the fixed penalties can be weighed
     against the explicit combat consequences. The flip multiplier m* is the attack multiplier at which B's best
     reply to A's Trade becomes a conceding behaviour.
  2. `policy_game`: sampled policy-matrix games. A set of REACTIVE scripted policies per seat (they read HP, contact,
     the lane and the siege) is played from a fixed state for WINDOW turns with the engine's own chance events
     (trades land, towers shoot, ganks happen, champions recall and die), on common random numbers. The A x B matrix
     of mean values is solved as a zero-sum game. This captures multi-turn consequences the one-turn inequality
     cannot: absences, pushes during an absence, under-tower farming, plates, XP.

Both tools run on the rule variants of `VARIANTS` (opt-in presets), the start states of `starts` (neutral lane,
pushed lane, siege, outside-tower freeze) and the power gaps of `GAPS` (investment and level leads of A).

    python -m lanegame.experiments.m4.zoning_natural                    all variants, states and gaps (gzipped JSON
                                                                        to results/m4_zoning_natural/, about 7 min)
    python -m lanegame.experiments.m4.zoning_natural t_ot6_dur          one named variant (the m4_lane() preset)
    python -m lanegame.experiments.m4.zoning_natural quick [all] NAMES  8 seeds, equal and large gaps, the neutral
                                                                        state (or all four with "all"); development

Coordinates: each side's OWN coordinates unless a trajectory says GLOBAL (own tower -2, the lane 0, enemy tower +2).
"""

from __future__ import annotations

import os
import sys
import time
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ...champions import (AWAY, CHAMP_ABSENT, CHANNEL, FARM, N_CHANCE, RETREAT, TRADING, champion_legal_actions,
                          in_contact, trade_double_probability, trade_hit_probability)
from ...champions_m4 import M4State, collection_units, context, initial_m4, resolve_turn_m4
from ...config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, NONE, PRESSURE, W_CLEAR, W_HOLD, W_THIN, GameConfig,
                       champ_act, exact_safe_base)
from ...io import dump_json
from ...matrix_game import solve_matrix_game
from ...mechanics import initial_state
from . import natural_waves as nw
from .common import RESULTS_DIR, meta

OUT = os.path.join(RESULTS_DIR, "m4_zoning_natural")
A, B = 0, 1
TURNS = [0]
LAMBDA = 0.25                 # resources per HP unit (1/6 of a bar) where a single number is needed
LAMBDAS = (0.0, 0.25, 0.4)
LAMBDA_XP = 0.05              # sensitivity only: resources per XP (a level early is about 20 XP and x1.05 power)
HORIZON = 6                   # stationary pairs of the decomposition
WINDOW = 32                   # turns of a sampled window (16 waves)
SEEDS = 16
RECALL_FRAC = 0.4
RESET_FRAC = 0.7              # recall while the enemy is away and own HP is below this fraction
HO, CL, TN = nw.HO, nw.CL, nw.TN


# ---------------------------------------------------------------------- rule variants
REACH = dict(collect_safe=0.9, collect_safe_slope=0.0, collect_contest_enemy_tower=0.9, retreat_farm_reach=1)
VARIANT_FIELDS = {
    "base": {},                                              # GameConfig.m4_natural(), M4.2e-A.1
    "tower": dict(collect_contest_enemy_tower=0.9),          # the 0.5 rate under the enemy tower removed
    "reach": REACH,                                          # ... plus the Safe / retreat posture base and the retreat reach 2
    "x2": dict(trade_exchanges=2),                           # one exchange per tick
    "reach_x2": {**REACH, "trade_exchanges": 2},
    "reach_x2_xp2": {**REACH, "trade_exchanges": 2, "xp_range": 2},
    "x2_xp2": dict(trade_exchanges=2, xp_range=2),
    "k2": dict(kill_bounty=2.0),                             # a kill worth two waves (League: about 2.4 early waves)
    "t_k2": dict(kill_bounty=2.0, collect_contest_enemy_tower=0.9),
    "x2_k2": dict(kill_bounty=2.0, trade_exchanges=2),
    "t_x2_k2": dict(kill_bounty=2.0, trade_exchanges=2, collect_contest_enemy_tower=0.9),
    "reach_x2_k2": {**REACH, "trade_exchanges": 2, "kill_bounty": 2.0},
    "k1": dict(kill_bounty=1.0),
    "t_k1": dict(kill_bounty=1.0, collect_contest_enemy_tower=0.9),
    "dur": dict(invest_durability=True),                    # investment also buys durability
    "loser": dict(trade_farm_rule="loser"),                  # the loser of a mutual trade pays the farm factor
    "ot6": dict(collect_contest_own_tower=0.6),              # last-hitting under one's own tower
    "t_ot6": dict(collect_contest_own_tower=0.6, collect_contest_enemy_tower=0.9),
    "x2_dur": dict(trade_exchanges=2, invest_durability=True),
    "x2_loser": dict(trade_exchanges=2, trade_farm_rule="loser"),
    "dur_loser": dict(invest_durability=True, trade_farm_rule="loser"),
    "x2_dur_loser": dict(trade_exchanges=2, invest_durability=True, trade_farm_rule="loser"),
    "t_ot6_dur_loser": dict(invest_durability=True, trade_farm_rule="loser", collect_contest_own_tower=0.6,
                            collect_contest_enemy_tower=0.9),
    "t_ot6_x2_dur_loser": dict(trade_exchanges=2, invest_durability=True, trade_farm_rule="loser",
                               collect_contest_own_tower=0.6, collect_contest_enemy_tower=0.9),
    "t_ot6_x2_dur_loser_k2": dict(trade_exchanges=2, invest_durability=True, trade_farm_rule="loser",
                                  collect_contest_own_tower=0.6, collect_contest_enemy_tower=0.9, kill_bounty=2.0),
    "t_ot6_x2_loser": dict(trade_exchanges=2, trade_farm_rule="loser", collect_contest_own_tower=0.6,
                           collect_contest_enemy_tower=0.9),
    "t_ot6_x2_dur": dict(trade_exchanges=2, invest_durability=True, collect_contest_own_tower=0.6,
                         collect_contest_enemy_tower=0.9),
    "tdl_ot5": dict(invest_durability=True, trade_farm_rule="loser", collect_contest_own_tower=0.5,
                    collect_contest_enemy_tower=0.9),
    "tdl_ot7": dict(invest_durability=True, trade_farm_rule="loser", collect_contest_own_tower=0.7,
                     collect_contest_enemy_tower=0.9),
    "tdl_ot9": dict(invest_durability=True, trade_farm_rule="loser", collect_contest_enemy_tower=0.9),
    "tdl_k1": dict(invest_durability=True, trade_farm_rule="loser", collect_contest_own_tower=0.6,
                   collect_contest_enemy_tower=0.9, kill_bounty=1.0),
    "tdl_k2": dict(invest_durability=True, trade_farm_rule="loser", collect_contest_own_tower=0.6,
                   collect_contest_enemy_tower=0.9, kill_bounty=2.0),
    "t_ot6_dur": dict(invest_durability=True, collect_contest_own_tower=0.6, collect_contest_enemy_tower=0.9),
    "t_ot6_loser": dict(trade_farm_rule="loser", collect_contest_own_tower=0.6, collect_contest_enemy_tower=0.9),
    "t_ot6_x2_dur_loser_xp2": dict(trade_exchanges=2, invest_durability=True, trade_farm_rule="loser",
                                   collect_contest_own_tower=0.6, collect_contest_enemy_tower=0.9, xp_range=2),
    # M4.2e-C (docs/natural_waves/m4_2e_wave_actions_report.md): the two candidates as variants of m4_natural()
    "lane": {k: getattr(GameConfig.m4_lane(), k) for k in GameConfig.M4_LANE_FIELDS},               # == m4_lane()
    "lasthit": {k: getattr(GameConfig.m4_lasthit(), k)
                for k in GameConfig.M4_LANE_FIELDS + GameConfig.M4_LASTHIT_FIELDS},                 # == m4_lasthit()
    "lasthit_ot5": {**{k: getattr(GameConfig.m4_lasthit(), k)
                       for k in GameConfig.M4_LANE_FIELDS + GameConfig.M4_LASTHIT_FIELDS}, "collect_contest_own_tower": 0.5},
    "lasthit_ot7": {**{k: getattr(GameConfig.m4_lasthit(), k)
                       for k in GameConfig.M4_LANE_FIELDS + GameConfig.M4_LASTHIT_FIELDS}, "collect_contest_own_tower": 0.7},
}
VARIANTS = ("base", "t_ot6", "x2_loser", "x2_dur_loser", "t_ot6_x2_loser", "t_ot6_x2_dur", "t_ot6_dur_loser",
            "t_ot6_x2_dur_loser", "t_ot6_x2_dur_loser_k2")      # the reported ablation; "t_ot6_dur" is m4_lane()


def variant(name: str, **kw) -> GameConfig:
    """The opt-in rule variants compared in the study, on top of the M4.2e-A.1 preset `GameConfig.m4_natural()`.
    `reach` removes the fixed collection-efficiency penalties that stood in for danger: the Safe / retreat posture base
    (0.8 - 0.2 (x + T)) and the 0.5 rate under the enemy tower both become the ordinary contest rate 0.9, and a
    threatened Yield collects within the ordinary farm reach (1) instead of 2; a champion's farm then depends only on
    where it stands and where the minions die. `x2` resolves a connected Trade once per tick (trade_exchanges = 2)."""
    base = dict(VARIANT_FIELDS[name])
    base.update(kw)
    cfg = GameConfig.m4_natural(**base)
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------- power gaps
def investment_ratio(cfg: GameConfig, sa: float, sb: float) -> float:
    from ...progression import investment
    inv = investment(cfg)
    return inv.factor(int(sa * cfg.gold_scale)) / inv.factor(int(sb * cfg.gold_scale))


GAPS = {"equal": dict(inv=(0, 0), levels=0),
        "moderate": dict(inv=(16, 4), levels=0),        # x1.24 attack power
        "large": dict(inv=(50, 0), levels=0),           # x1.76
        "levels": dict(inv=(0, 0), levels=3),           # +3 levels: x1.16 attack power and max HP
        "large_levels": dict(inv=(50, 0), levels=3)}    # x2.04 attack power


def with_gap(m4: M4State, cfg: GameConfig, gap: str) -> M4State:
    """A's investment and level lead over B, both at full HP for their level, no unspent gold."""
    g = GAPS[gap]
    t = context(cfg).t
    sa, sb = g["inv"]
    xp_b = m4.b.xp
    lb = t.level_by_xp[min(xp_b, t.xp_cap)]
    la = min(cfg.level_cap, lb + g["levels"])
    xp_a = max(m4.a.xp, t.xp_cum[la]) if g["levels"] else m4.a.xp
    a = m4.a._replace(items=int(sa * cfg.gold_scale), gold=0, xp=xp_a, hp=t.hp_max[t.level_by_xp[min(xp_a, t.xp_cap)]])
    b = m4.b._replace(items=int(sb * cfg.gold_scale), gold=0, hp=t.hp_max[lb])
    return M4State(a, b)


def power_ratio(cfg: GameConfig, m4: M4State) -> float:
    """A's expected damage per hit on B in B's HP units divided by B's on A in A's units (equal skills)."""
    ctx, t = context(cfg), context(cfg).t
    la, lb = t.level_by_xp[min(m4.a.xp, t.xp_cap)], t.level_by_xp[min(m4.b.xp, t.xp_cap)]
    da = ctx.hit(1, la, m4.a.items) / t.hp_unit[lb]
    db = ctx.hit(1, lb, m4.b.items) / t.hp_unit[la]
    return da / db


# ---------------------------------------------------------------------- start states
def starts(cfg: GameConfig) -> Dict[str, tuple]:
    """Deterministic states from s0 (both champions level up on the way; the gap is applied afterwards):

      neutral  k9 of Hold / Hold: both fronts and both champions at the lane
      pushed   A's slow push (Thin at k1, then Hold) at the first turn the contact stands at GLOBAL +1 (B's -1):
               A's stack next to B's tower, B's front and champion one cell in front of its tower
      siege    the same line one turn after A's crash, A standing in B's tower cell with Hold + Pressure:
               A's mass under B's tower, B in its tower cell
      freeze   the outside-tower freeze of the final review (B Thins at k1; A plays the bounded freezer, margin 8)
               at k36: B's wave held at A's -1, A's front in its tower cell, A at -2, B at its +1. Under the M4.2e-C
               capacity model (wave_clear_scaling "capacity") the freeze band is an absolute enemy size, so the
               setup is B Thins k1-4 and A plays the threshold freezer (Thin while the enemy front >= 14 HP once the
               contact is at A's -1); the state is the start of the longest hold"""
    out = {}
    rows = nw.play(cfg, nw.const(HO), 9)
    TURNS[0] += 9
    out["neutral"] = nw.end_state(rows)
    # pushed and siege
    st = {"crashed": False, "pressed": False}

    def pol(k, lane, m4):
        if lane.asg > 0 and not st["pressed"]:
            st["crashed"] = True
            return (nf_hop(), HO)
        return (TN if k == 1 else HO, HO)

    rows = nw.play(cfg, pol, 40)
    TURNS[0] += 40
    pushed = next(r for r in rows if nw.contact_cell(r["t"].lane.state) == 1)
    out["pushed"] = (pushed["t"].lane.state, pushed["t"].champ, pushed["k"] + 1)
    crash = next(r for r in rows if r["t"].lane.crash_hp[A] > 0)
    j = rows.index(crash)
    # one turn after the crash: A walks into B's tower cell with Hold + Pressure
    lane, m4, k = rows[j]["t"].lane.state, rows[j]["t"].champ, rows[j]["k"] + 1
    t = resolve_turn_m4(lane, m4, k, nw._legal(cfg, lane, m4, A, nf_hop()), HO, cfg, u=None)
    TURNS[0] += 1
    assert t.end[A] == cfg.tower and t.lane.state.asg > 0, (t.end, t.lane.state)
    out["siege"] = (t.lane.state, t.champ, k + 1)
    # freeze (final review stage_freeze setup; M4.2e-C: the threshold freezer, see the docstring)
    if cfg.wave_clear_scaling == "capacity":
        from . import wave_actions as wa
        rows, start = wa.freeze_setup(cfg, wa.threshold_freezer(wa.FREEZE_THRESHOLD),
                                      lambda k: TN if 1 <= k <= 4 else HO, 70)
        TURNS[0] += 70
        t = rows[start]["t"]                             # the state after the first held turn: B's front at A's -1
        lane, m4, k = t.lane.state, t.champ, rows[start]["k"] + 1
    else:
        fz = nw.bounded_freezer(8)
        rows = nw.play(cfg, lambda k, l, m: (fz(l), TN if k == 1 else HO), 36)
        TURNS[0] += 36
        lane, m4, k = nw.end_state(rows)
    assert lane.ax == -2 and lane.bx == 1 and lane.ah > 0 and lane.bh > 0, (lane,)
    out["freeze"] = (lane, m4, k)
    # common ground: both at the larger XP with full HP, no gold, no investment
    tb = context(cfg).t
    for n, (lane, m4, k) in out.items():
        xp = max(m4.a.xp, m4.b.xp)
        hp = tb.hp_max[tb.level_by_xp[min(xp, tb.xp_cap)]]
        out[n] = (lane, M4State(m4.a._replace(xp=xp, hp=hp, gold=0, items=0), m4.b._replace(xp=xp, hp=hp, gold=0, items=0)), k)
    return out


def nf_hop() -> int:
    return champ_act(W_HOLD, B_HOLD, PRESSURE)


# ---------------------------------------------------------------------- policies
class Policy:
    """A reactive scripted lane policy for either seat: a behaviour rule, a wave rule and an objective rule, each a
    function of (cfg, lane, m4, s). Common pre-rules: ABSENT while away; at or below RECALL_FRAC of max HP, Recall when
    legal, else walk back (Safe)."""

    def __init__(self, name: str, behavior: str = "hold", wave: str = "hold", pressure: bool = False,
                 recall_frac: float = RECALL_FRAC):
        self.name, self.behavior, self.wave, self.pressure, self.recall_frac = name, behavior, wave, pressure, recall_frac

    def __repr__(self):
        return self.name

    def act(self, cfg: GameConfig, lane, m4: M4State, s: int) -> int:
        legal = champion_legal_actions(lane, m4, s, cfg)
        if legal == (CHAMP_ABSENT,):
            return CHAMP_ABSENT
        me, en, t = m4.side(s), m4.side(1 - s), context(cfg).t
        L = t.level_by_xp[min(me.xp, t.xp_cap)]
        H = t.hp_max[L]
        # low HP: Recall out of the enemy's reach, else walk back; a free reset while the enemy is away (dead or
        # recalled) and own HP is below RESET_FRAC: recall now rather than at low HP in front of the returning enemy
        reset = en.absent > 0 and me.hp <= RESET_FRAC * H and lane.siege_of(1 - s) == 0
        if me.hp <= self.recall_frac * H or reset:
            c = champ_act(W_HOLD, B_RECALL)
            return c if (c in legal and not contact_at_start(cfg, m4, s)) else champ_act(W_HOLD, B_SAFE)
        beh = behavior_rule(self.behavior, cfg, lane, m4, s)
        wave = wave_rule(self.wave, cfg, lane, m4, s)
        objs = (PRESSURE, NONE) if self.pressure and beh in (B_HOLD, B_TRADE) else (NONE,)
        for o in objs:
            c = champ_act(wave, beh, o)
            if c in legal:
                return c
        return champ_act(W_HOLD, B_HOLD)


def own_front(lane, s: int) -> Tuple[int, int]:
    fx, fh = lane.side(s)[:2]
    return fx, fh


def contact_at_start(cfg, m4, s) -> bool:
    me, en = m4.side(s), m4.side(1 - s)
    return en.absent == 0 and me.absent == 0 and in_contact(me.pos, en.pos, cfg)


def duel_edge(cfg, lane, m4, s) -> float:
    ctx, t = context(cfg), context(cfg).t
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


def behavior_rule(rule: str, cfg, lane, m4, s) -> int:
    me, en = m4.side(s), m4.side(1 - s)
    c0 = contact_at_start(cfg, m4, s)
    present = en.absent == 0
    if rule == "hold":
        return B_HOLD
    if rule == "trade":                      # attack every turn the enemy is on the lane
        return B_TRADE if present else B_HOLD
    if rule == "punish":                     # attack only an opponent that starts the turn within reach (no dives)
        return B_TRADE if (present and c0 and en.pos > -cfg.tower) else B_HOLD
    if rule == "edge":                       # attack when the duel edge is positive and the opponent is in reach
        return B_TRADE if (present and c0 and duel_edge(cfg, lane, m4, s) > 0) else B_HOLD
    if rule == "standoff":
        return B_YIELD
    if rule == "respect":                    # back off while the enemy starts within reach, else stand off
        return B_SAFE if (present and c0) else B_YIELD
    if rule == "safe":
        return B_SAFE
    if rule == "tower":                      # concede the lane: farm from the tower side when the wave is there
        fx, fh = own_front(lane, s)
        spot = max(-cfg.tower, min(cfg.tower, fx)) if fh > 0 else me.pos
        if spot <= -1 and not c0:
            return B_HOLD
        return B_SAFE
    raise ValueError(rule)


def wave_rule(rule: str, cfg, lane, m4, s) -> int:
    fx, fh = own_front(lane, s)
    ex, eh = own_front(lane, 1 - s)
    contact = fh > 0 and eh > 0 and fx + ex >= -1
    if rule == "hold":
        return W_HOLD
    if rule == "thin":
        return W_THIN
    if rule == "clear":
        return W_CLEAR
    if rule == "shove_away":                 # shove while the enemy is away (dead or recalled), else Hold
        return W_CLEAR if m4.side(1 - s).absent > 0 else W_HOLD
    if not contact:
        return W_HOLD
    if rule == "slow_push":
        return W_THIN if fh <= eh + 2 else W_HOLD
    if rule == "freeze":
        if cfg.wave_clear_scaling == "capacity":          # M4.2e-C: the band is an absolute enemy size (threshold)
            return W_THIN if (fx <= -1 and eh >= 14) else W_HOLD
        return W_THIN if eh - fh >= 8 else W_HOLD
    if rule == "build_clear":
        return W_CLEAR if fh >= 16 else W_THIN
    raise ValueError(rule)


POLICIES = [Policy("farm"), Policy("bully", "trade"), Policy("punish", "punish"), Policy("edge", "edge"),
            Policy("standoff", "standoff"), Policy("respect", "respect"), Policy("tower", "tower"),
            Policy("push", wave="slow_push", pressure=True), Policy("push_punish", "punish", "slow_push", True),
            Policy("clear_punish", "punish", "clear", True), Policy("freeze_punish", "punish", "freeze"),
            Policy("bully_shove", "trade", "shove_away", True), Policy("punish_shove", "punish", "shove_away", True)]
CONCEDING = ("standoff", "respect", "tower")      # the policies that give way to a threat instead of contesting
AGGRESSIVE = ("bully", "punish", "edge", "push_punish", "clear_punish", "freeze_punish")


# ---------------------------------------------------------------------- sampled windows
def window(cfg: GameConfig, start, pa: Policy, pb: Policy, seed: int, turns: int = WINDOW) -> dict:
    lane, m4, k0 = start
    rng = np.random.default_rng(seed)
    acc = {k: [0, 0] for k in ("earned", "farm", "plates", "bounty", "xp", "deaths", "recalls", "trades", "connected",
                               "pinned", "contact", "damage", "ganks", "shots")}
    cells = [np.zeros(2 * cfg.tower + 1), np.zeros(2 * cfg.tower + 1)]
    crashes = [0, 0]
    for i in range(turns):
        u = tuple(rng.random(N_CHANCE))
        a, b = pa.act(cfg, lane, m4, A), pb.act(cfg, lane, m4, B)
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, u=u)
        TURNS[0] += 1
        for s in (A, B):
            acc["earned"][s] += t.earned_units[s]
            acc["farm"][s] += t.farm_units[s]
            acc["plates"][s] += t.plate_units[s]
            acc["bounty"][s] += t.bounty_units[s]
            acc["xp"][s] += t.xp_gain[s]
            acc["deaths"][s] += t.died[s]
            acc["recalls"][s] += t.recalled[s]
            acc["trades"][s] += t.behaviors[s] == B_TRADE
            acc["connected"][s] += t.connected[s]
            acc["pinned"][s] += t.pinned[s]
            acc["contact"][s] += (not t.away[s]) and (not t.away[1 - s]) and in_contact(t.start[s], t.start[1 - s], cfg)
            acc["damage"][s] += t.damage_taken[s]
            acc["ganks"][s] += t.ganked[s]
            acc["shots"][s] += t.tower_damage[s] > 0
            if not t.away[s]:
                cells[s][t.end[s] + cfg.tower] += 1
            crashes[s] += t.lane.crash_hp[s] > 0
        lane, m4 = t.lane.state, t.champ
    tb = context(cfg).t
    bar = [cfg.champion_hp if m4.side(s).absent else
           m4.side(s).hp / tb.hp_unit[tb.level_by_xp[min(m4.side(s).xp, tb.xp_cap)]] for s in (A, B)]
    g = cfg.gold_scale
    out = {k: [v[A] / g, v[B] / g] if k in ("earned", "farm", "plates", "bounty") else list(v) for k, v in acc.items()}
    out["bar"] = bar
    out["cells"] = [c.tolist() for c in cells]
    out["crashes"] = crashes
    out["gold_diff"] = out["earned"][A] - out["earned"][B]
    out["value"] = {str(lam): out["gold_diff"] + lam * (bar[A] - bar[B]) for lam in LAMBDAS}
    out["value"]["xp"] = out["gold_diff"] + LAMBDA * (bar[A] - bar[B]) + LAMBDA_XP * (out["xp"][A] - out["xp"][B])
    return out


def agg(rows: List[dict]) -> dict:
    out = {}
    for lam in list(LAMBDAS) + ["xp"]:
        v = np.array([r["value"][str(lam)] for r in rows])
        out[f"value_{lam}"] = round(float(v.mean()), 3)
        out[f"se_{lam}"] = round(float(v.std(ddof=1) / np.sqrt(len(v))), 3)
    for key in ("earned", "farm", "plates", "bounty", "xp", "deaths", "recalls", "trades", "connected", "pinned",
                "contact", "damage", "ganks", "shots", "bar", "crashes"):
        out[key] = [round(float(np.mean([r[key][s] for r in rows])), 3) for s in (A, B)]
    out["cells"] = [[round(float(np.mean([r["cells"][s][i] for r in rows])), 2) for i in range(len(rows[0]["cells"][s]))]
                    for s in (A, B)]
    return out


def policy_game(cfg: GameConfig, start, gap: str, a_pols=POLICIES, b_pols=POLICIES, seeds: int = SEEDS,
                turns: int = WINDOW, seed0: int = 5000) -> dict:
    lane, m4, k0 = start
    m4 = with_gap(m4, cfg, gap)
    st = (lane, m4, k0)
    cells = {}
    for pa in a_pols:
        for pb in b_pols:
            rows = [window(cfg, st, pa, pb, seed0 + sd, turns) for sd in range(seeds)]
            cells[(pa.name, pb.name)] = agg(rows)
    res = {"power_ratio": round(power_ratio(cfg, m4), 3), "cells": {f"{a}|{b}": v for (a, b), v in cells.items()},
           "games": {}}
    an, bn = [p.name for p in a_pols], [p.name for p in b_pols]
    for lam in list(LAMBDAS) + ["xp"]:
        M = np.array([[cells[(a, b)][f"value_{lam}"] for b in bn] for a in an])
        sol = solve_matrix_game(M)
        x, y = np.array(sol.x), np.array(sol.y)
        b_br = {a: bn[int(np.argmin(M[i]))] for i, a in enumerate(an)}
        a_br = {b: an[int(np.argmax(M[:, j]))] for j, b in enumerate(bn)}
        # how often A attacks and B stands in contact in the equilibrium mixture
        trades = sum(x[i] * y[j] * cells[(a, b)]["trades"][A] for i, a in enumerate(an) for j, b in enumerate(bn))
        contact = sum(x[i] * y[j] * cells[(a, b)]["contact"][B] for i, a in enumerate(an) for j, b in enumerate(bn))
        # B concedes to a: its best reply gives way (a conceding policy) AND actually sacrifices farm against a
        # (less than 80% of what plain contesting collects against the same a)
        def yielded(a, b):
            return b in CONCEDING and cells[(a, b)]["farm"][B] < 0.8 * cells[(a, "farm")]["farm"][B]
        concede = sum(float(x[i] * y[j]) for i, a in enumerate(an) for j, b in enumerate(bn) if yielded(a, b))
        res["games"][str(lam)] = {
            "value": round(float(sol.value), 3), "B_concede_mass": round(concede, 3),
            "B_concedes_vs": [a for a, b in b_br.items() if yielded(a, b)],
            "A": {a: round(float(v), 3) for a, v in zip(an, x) if v > 1e-6},
            "B": {b: round(float(v), 3) for b, v in zip(bn, y) if v > 1e-6},
            "B_best_reply": b_br, "A_best_reply": a_br,
            "A_trades_per_turn": round(float(trades) / turns, 3),
            "B_contact_frac": round(float(contact) / turns, 3),
            "matrix": [[round(float(v), 3) for v in row] for row in M]}
    return res


# ---------------------------------------------------------------------- one-turn decomposition (deterministic)
BEHAVIORS = (B_HOLD, B_YIELD, B_TRADE, B_SAFE)
BNAME = {B_HOLD: "Hold", B_YIELD: "Yield", B_TRADE: "Trade", B_SAFE: "Safe"}


def gold_breakdown(t, s: int, cfg: GameConfig) -> Dict[str, float]:
    """Exact attribution (resources) of side s's farm opportunities this turn: collected + base (last-hit
    inefficiency) + reach (out of reach, away, channeling) + posture (Safe / retreat base below the contest base) +
    zone + trade. The parts sum to the available farm exactly (zoning report §1)."""
    q = lambda v: Fraction(repr(float(v)))
    f, T, scale = q(cfg.farm_per_hp), cfg.tower, cfg.gold_scale
    o = 1 - s
    mode, pos = t.modes[s], t.end[s]
    out = {k: Fraction(0) for k in ("available", "collected", "base", "reach", "posture", "zone", "trade")}
    for x, hp in t.lane.kills[s].items():
        cmax = q(cfg.collect_contest) if x < T else q(cfg.collect_contest_enemy_tower)
        out["available"] += f * hp
        out["base"] += f * hp * (1 - cmax)
        reach = cfg.retreat_farm_reach if mode == RETREAT else cfg.farm_reach
        if mode in (AWAY, CHANNEL) or abs(x - pos) > reach:
            out["reach"] += f * hp * cmax
            continue
        b = cmax if mode in (FARM, TRADING) else min(cmax, exact_safe_base(x, cfg))
        out["posture"] += f * hp * (cmax - b)
        got = Fraction(hp * collection_units(x, mode, pos, t.modes[o], t.end[o], cfg), scale)
        out["collected"] += got
        tf = q(cfg.trade_farm_factor) if mode == TRADING else Fraction(1)
        kept = got / (f * hp * b * tf) if b * tf > 0 else Fraction(0)
        out["zone"] += f * hp * b * (1 - kept) if b > 0 else Fraction(0)
        out["trade"] += f * hp * b * kept * (1 - tf)
    assert sum(v for k, v in out.items() if k != "available") == out["available"], out
    assert out["collected"] == Fraction(t.farm_units[s], scale)
    return {k: float(v) for k, v in out.items()}


def expected_damage(t, m4: M4State, s: int, cfg: GameConfig) -> Dict[str, float]:
    """Expected centi-HP side s takes this turn (the opponent's trade, a tower shot, a gank), in units of u(L_s)."""
    ctx, tb = context(cfg), context(cfg).t
    o = 1 - s
    lv = t.levels_before
    unit = tb.hp_unit[lv[s]]
    trade = 0.0
    if t.connected[o]:
        h, qd = t.hit_p[o], t.double_p[o]
        per = h * (1 - qd) * ctx.hit(1, lv[o], m4.side(o).items) + h * qd * ctx.hit(2, lv[o], m4.side(o).items)
        trade = per * cfg.trade_exchanges
    tower = t.tower_p[s] * cfg.tower_shot_damage * unit
    gank = t.gank_p[s] * cfg.gank_damage_trials * t.gank_damage_p[s] * unit
    mit = 1.0
    if cfg.invest_durability and ctx.inv is not None:
        mit = ctx.inv.one / ctx.inv.multiplier(m4.side(s).items)
    return {"trade": mit * trade / unit, "tower": mit * tower / unit, "gank": mit * gank / unit}


def run_pair(cfg: GameConfig, start, ba: int, bb: int, horizon: int = HORIZON) -> List[dict]:
    lane, m4, k0 = start
    rows = []
    for i in range(horizon):
        a, b = champ_act(W_HOLD, ba, NONE), champ_act(W_HOLD, bb, NONE)
        assert a in champion_legal_actions(lane, m4, A, cfg) and b in champion_legal_actions(lane, m4, B, cfg)
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, u=None)
        TURNS[0] += 1
        rows.append({"end": t.end, "modes": t.modes, "connected": t.connected, "threatened": t.threatened,
                     "pinned": t.pinned, "earned": [t.earned_units[s] / cfg.gold_scale for s in (A, B)],
                     "gold": [gold_breakdown(t, s, cfg) for s in (A, B)],
                     "xp": [t.xp_gain[s] for s in (A, B)],
                     "dmg": [expected_damage(t, m4, s, cfg) for s in (A, B)]})
        lane, m4 = t.lane.state, t.champ
    return rows


def summarize(rows: List[dict], h: int, lam: float = LAMBDA) -> dict:
    r = rows[:h]
    earned = [sum(x["earned"][s] for x in r) for s in (A, B)]
    units = [sum(sum(x["dmg"][s].values()) for x in r) for s in (A, B)]
    trade_units = [sum(x["dmg"][s]["trade"] for x in r) for s in (A, B)]
    gold = {k: [round(sum(x["gold"][s][k] for x in r), 4) for s in (A, B)]
            for k in ("available", "collected", "base", "reach", "posture", "zone", "trade")}
    return {"earned": [round(e, 4) for e in earned], "dmg_units": [round(u, 4) for u in units],
            "trade_units": [round(u, 4) for u in trade_units], "gold": gold,
            "xp": [sum(x["xp"][s] for x in r) for s in (A, B)],
            "connected": [sum(x["connected"][s] for x in r) for s in (A, B)],
            "pinned": [sum(x["pinned"][s] for x in r) for s in (A, B)],
            "value": round(earned[A] - earned[B] + lam * (units[B] - units[A]), 4)}


def concedes(rows, bb: int, h: int) -> bool:
    """B's behaviour bb on this path is a concession: Safe, or a Yield that actually gave way (threatened and never
    pinned) within the horizon. A pinned stand-off farms and is hit exactly like Hold, so it is a contest."""
    r = rows[:h]
    if bb == B_SAFE:
        return True
    return bb == B_YIELD and sum(x["threatened"][B] for x in r) > 0 and sum(x["pinned"][B] for x in r) == 0


def flip_multiplier(paths: dict, h: int, lam: float, grid=np.arange(1.0, 6.001, 0.01)):
    """Linear-in-AP extrapolation from the equal-power path: the smallest A attack multiplier m at which B's best
    reply to A's Trade is a conceding behaviour (`concedes`). Only A's trade damage scales with m."""
    vals = {}
    for bb in BEHAVIORS:
        s = summarize(paths[(B_TRADE, bb)], h, lam)
        vals[bb] = (s["value"] - lam * s["trade_units"][B], lam * s["trade_units"][B])
    for m in grid:
        best = min(BEHAVIORS, key=lambda bb: vals[bb][0] + m * vals[bb][1])
        if concedes(paths[(B_TRADE, best)], best, h):
            return float(round(m, 2)), BNAME[best]
    return None, BNAME[min(BEHAVIORS, key=lambda bb: vals[bb][0] + grid[-1] * vals[bb][1])]


def decompose(cfg: GameConfig, start) -> dict:
    paths = {(ba, bb): run_pair(cfg, start, ba, bb) for ba in BEHAVIORS for bb in BEHAVIORS}
    out = {"pairs": {}, "flip": {}, "games": {}}
    for (a, b), rows in paths.items():
        out["pairs"][f"{BNAME[a]}-{BNAME[b]}"] = {f"h{h}": summarize(rows, h) for h in (1, HORIZON)}
    for h in (1, HORIZON):
        for lam in LAMBDAS:
            if lam == 0:
                continue
            out["flip"][f"h{h}/l{lam}"] = flip_multiplier(paths, h, lam)
            M = np.array([[summarize(paths[(a, b)], h, lam)["value"] for b in BEHAVIORS] for a in BEHAVIORS])
            sol = solve_matrix_game(M)
            out["games"][f"h{h}/l{lam}"] = {
                "value": round(float(sol.value), 4),
                "A": {BNAME[a]: round(float(v), 3) for a, v in zip(BEHAVIORS, sol.x) if v > 1e-6},
                "B": {BNAME[b]: round(float(v), 3) for b, v in zip(BEHAVIORS, sol.y) if v > 1e-6},
                "B_best_reply_to_Trade": BNAME[BEHAVIORS[int(np.argmin(M[2]))]]}
    return out


# ---------------------------------------------------------------------- representative trajectories
def trajectory(cfg: GameConfig, start, gap: str, pa: Policy, pb: Policy, seed: int, turns: int = WINDOW):
    """One sampled window turn by turn: [k, labels, GLOBAL lane strings, champion cells (GLOBAL; 'away'), HP bars,
    connected, damage taken (centi-HP), farm, earned cumulative A - B, events]."""
    lane, m4, k0 = start
    m4 = with_gap(m4, cfg, gap)
    rng = np.random.default_rng(seed)
    tb = context(cfg).t
    rows, cum = [], 0
    for i in range(turns):
        u = tuple(rng.random(N_CHANCE))
        a, b = pa.act(cfg, lane, m4, A), pb.act(cfg, lane, m4, B)
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, u=u)
        TURNS[0] += 1
        cum += t.reward_units
        st = t.lane.state
        cells = ["away" if t.away[s] else (t.end[s] if s == A else -t.end[s]) for s in (A, B)]
        bars = [None if t.champ.side(s).absent else round(t.champ.side(s).hp /
                tb.hp_unit[tb.level_by_xp[min(t.champ.side(s).xp, tb.xp_cap)]], 2) for s in (A, B)]
        ev = []
        for s, nm in ((A, "A"), (B, "B")):
            if t.died[s]:
                ev.append(f"{nm} dies")
            if t.recalled[s]:
                ev.append(f"{nm} recalls")
            if t.pinned[s]:
                ev.append(f"{nm} pinned")
            if t.threatened[s]:
                ev.append(f"{nm} evades")
            if t.ganked[s]:
                ev.append(f"{nm} ganked")
            if t.tower_damage[s]:
                ev.append(f"{nm} tower shot")
            if t.lane.crash_hp[s]:
                ev.append(f"{nm} crashes {t.lane.crash_hp[s]}")
            if t.plates_broken[s]:
                ev.append(f"{nm} plate")
        rows.append([k0 + i, [cfg.label(a), cfg.label(b)], nw.side_str(st, A), nw.side_str(st, B), cells, bars,
                     list(t.connected), list(t.damage_taken), [round(x, 3) for x in t.farm],
                     round(cum / cfg.gold_scale, 3), ev])
        lane, m4 = st, t.champ
    return rows


def trajectory_text(rows) -> str:
    out = []
    for r in rows:
        out.append(f"k{r[0]:3d} {r[1][0]:>8s} {r[1][1]:>8s} | {r[2]:14s} {r[3]:14s} | A@{str(r[4][0]):>4s} B@{str(r[4][1]):>4s} "
                   f"| hp {str(r[5][0]):>5s} {str(r[5][1]):>5s} | farm {r[8][0]:.2f} {r[8][1]:.2f} | cum {r[9]:+6.2f} | "
                   + ", ".join(r[10]))
    return "\n".join(out)


# ---------------------------------------------------------------------- driver
def job(args) -> tuple:
    name, n, g, seeds, turns = args
    cfg = variant(name)
    st = starts(cfg)
    t0 = time.time()
    pg = policy_game(cfg, st[n], g, seeds=seeds, turns=turns)
    pg["wall_s"] = round(time.time() - t0, 1)
    pg["turns"] = TURNS[0]
    return name, n, g, pg


def print_games(res: dict, lam: float = LAMBDA) -> None:
    for key, pg in res["policy_game"].items():
        g = pg["games"][str(lam)]
        print(f"  {key} (ratio {pg['power_ratio']}): V {g['value']:+.2f}  A {g['A']}  B {g['B']}  "
              f"A trades/turn {g['A_trades_per_turn']}  B contact {g['B_contact_frac']}  "
              f"B concedes vs {g['B_concedes_vs']}")


def matrix_text(pg: dict, lam: float = LAMBDA) -> str:
    g = pg["games"][str(lam)]
    an, bn = list(g["A_best_reply"].keys()), list(g["B_best_reply"].keys())
    lines = ["A/B".ljust(14) + "".join(b[:9].rjust(10) for b in bn)]
    for a, row in zip(an, g["matrix"]):
        lines.append(a.ljust(14) + "".join(f"{x:+10.2f}" for x in row))
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quick = "quick" in argv
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    TURNS[0] = 0
    names = [a for a in argv if a in VARIANT_FIELDS] or list(VARIANTS)
    states = ("neutral",) if (quick and "all" not in argv) else ("neutral", "pushed", "siege", "freeze")
    gaps = ("equal", "large") if quick else tuple(GAPS)
    seeds, turns = (8, WINDOW) if quick else (SEEDS, WINDOW)
    data = {"variants": {name: {"decompose": {}, "policy_game": {}} for name in names}}
    for name in names:
        cfg = variant(name)
        st = starts(cfg)
        data["variants"][name]["starts"] = {n: {"lane": list(v[0]), "m4": v[1].to_list(), "k": v[2]} for n, v in st.items()}
        for n in states:
            data["variants"][name]["decompose"][n] = decompose(cfg, st[n])
    from .common import pool_map
    jobs = [(name, n, g, seeds, turns) for name in names for n in states for g in gaps]
    for name, n, g, pg in pool_map(job, jobs, procs=1 if quick else 0):
        data["variants"][name]["policy_game"][f"{n}/{g}"] = pg
        TURNS[0] += pg.pop("turns")
    for name in names:
        print("=====", name)
        print_games(data["variants"][name])
        for n in states:
            d = data["variants"][name]["decompose"][n]
            print(f"  decompose {n}: flip {d['flip']}")
    data["budget"] = {"turns": TURNS[0], "wall_s": round(time.time() - t0, 1)}
    named = [a for a in argv if a in VARIANT_FIELDS]
    fn = ("zoning_natural_quick_" + "_".join(names) + ".json") if quick else (
        "zoning_natural.json" if not named else "zoning_natural_" + "_".join(names) + ".json")
    dump_json_gz({**meta(variant("base")), **data}, os.path.join(OUT, fn + ".gz"))
    print("budget", data["budget"])


def dump_json_gz(data: dict, path: str) -> None:
    """The per-cell results of a full run are tens of MB as plain JSON; they are written gzipped (`load_json` reads
    both forms)."""
    import gzip
    import json
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), default=str)


def load_json(path: str) -> dict:
    import gzip
    import json
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
