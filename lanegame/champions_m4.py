"""M4 champion state and turn resolution (docs/m4_integrated_design.md; plan WP4-WP6).

A separately versioned resolver for `GameConfig.m4()`. The M2 resolver `champions.resolve_turn` is not edited;
this module reuses its pure helpers (contact, movement, evasion, farm collection, gank and trade probabilities,
legality) and adds, in the order of design §5:

  step 1   opt-in (M4.2d, `farm_spot_rule="pre_crash_front"`): a champion whose front crashes in the probe keeps its
           pre-turn front cell as farm spot instead of walking back to the next packet
  step 2   the contact-cell chase rule (§7.4; `chase_rule="contact_cell"`)
  step 5   Freeze maintenance (§6.3): a Freeze is effective only while the freezer is present, not channeling, not
           in a connected trade and ends within farm_reach of the freeze line; otherwise it resolves as Slow Push
  step 6   the lane step with the effective waves, which also returns the minion tower hits (§11.2)
  step 7   gold (M2 collection rule) as exact integers in ledger units (micro-resources, §8.6); opt-in (M4.2d-C,
           `zoning_rule`): the zone factor needs a connected enemy Trade ("contact") or never applies ("none")
  step 7b  XP: xp_per_hp per enemy minion HP dying within xp_range cells of a present champion (§7.3)
  step 9   centi-HP damage: trade hits dmg(D, L, N), gank trials and tower shots of one HP unit u(L) of the victim,
           the tower-adjacent shot on a connected trader ending next to the enemy tower (§8.3, §8.4)
  step 10  death before level-up (§5)
  step 11  persistent tower damage and plate gold (§11)
  step 12  XP and level-ups: a surviving present champion gains the increase in max HP (§9.2)
  step 13  the gold ledger, clamped at G_rem(N) (§8.6)
  step 14  absence bookkeeping: a champion whose timer reaches 0 buys its tiers, then returns with full HP (§8.5)
  step 15  reward r_A = earned_A - earned_B (farm + plate gold + kill bounty), exact in ledger units

Every power-dependent quantity of a turn (trade damage, gank and tower damage, Pressure tower damage) uses the levels
and items held at the start of the turn. The chance vector is M2's (N_CHANCE = 10 uniforms per turn, drawn whatever
happens); the tower-adjacent shot reuses U_TOWER[i] (a trader ends either at T or at T - 1, never both).

State per side (own coordinates, own tower -T): `M4Side(pos, hp, absent, xp, items, gold, tower_dmg)` with hp in
centi-HP, gold in ledger units and tower_dmg the cumulative damage this side dealt to the ENEMY tower in centi-units.
Levels, max HP, attack power and plates broken are derived (lanegame.progression).

M4.2c experiment (`GameConfig.m4_continuous()`, item_model = "continuous"): the `items` slot holds the cumulative
INVESTED gold in ledger units (`M4Side.invested`), there is no G_rem clamp, and at the end of every absence (the
turn the timer reaches 0: after a Recall or a death, exactly where the baseline shops) all unspent gold moves into
it. Power-dependent damage uses `progression.Investment` (`M4Context.hit` / `.pressure`). The reward is unchanged:
it counts gold EARNED, so investing is never a reward event.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

from .champions import (AWAY, CHANNEL, FARM, N_CHANCE, NO_REACH, RETREAT, SAFE_MODE, TRADING, U_GANK, U_GANK_DAMAGE,
                        U_TOWER, U_TRADE_DOUBLE, U_TRADE_HIT, LaneStep, _decode, _evasion_cell, _gank_damage, _move,
                        _trade_damage, champion_gank_probability, champion_legal_actions, gank_damage_p, in_contact,
                        objective_reachable, recall_legal, trade_double_probability, trade_hit_probability)
from .config import B_RECALL, B_SAFE, B_TRADE, B_YIELD, B_ABSENT, F, NONE, PRESSURE, SP, GameConfig, exact_safe_base
from .mechanics import step_turn
from .progression import Tables, tables
from .state import State

A, B = 0, 1

__all__ = ["M4Side", "M4State", "M4Turn", "M4Context", "context", "initial_m4", "resolve_turn_m4",
           "champion_legal_actions", "objective_reachable", "recall_legal", "check_invariants", "level",
           "contact_cell_target", "collection_units"]


class M4Side(NamedTuple):
    pos: int              # own coordinates, -T .. T (-T while away)
    hp: int               # centi-HP, 0 .. H(L); 0 only while dead
    absent: int = 0       # turns still to sit out (M2 semantics)
    xp: int = 0           # 0 .. X(level_cap)
    items: int = 0        # purchased tiers, 0 .. item_tiers; item_model "continuous": invested gold, ledger units
    gold: int = 0         # spendable gold, ledger units, 0 .. G_rem(items) (no upper bound under "continuous")
    tower_dmg: int = 0    # cumulative damage dealt to the ENEMY tower, centi-units, 0 .. last threshold

    @property
    def invested(self) -> int:
        """Cumulative invested gold in ledger units (item_model "continuous"; the same slot as `items`)."""
        return self.items


class M4State(NamedTuple):
    a: M4Side
    b: M4Side

    def side(self, s: int) -> M4Side:
        return self.a if s == A else self.b

    def mirror(self) -> "M4State":
        return M4State(self.b, self.a)

    def to_list(self) -> list:
        return [list(self.a), list(self.b)]

    @classmethod
    def from_list(cls, v) -> "M4State":
        return cls(M4Side(*v[0]), M4Side(*v[1]))

    def replace_side(self, s: int, **kw) -> "M4State":
        sides = [self.a, self.b]
        sides[s] = sides[s]._replace(**kw)
        return M4State(*sides)


M4_SIDE_FIELDS = M4Side._fields


# ---------------------------------------------------------------------- per-configuration context
class M4Context:
    """Integer tables of one configuration: progression tables, the ledger scale and a memo of the exact per-HP
    farm value of every collection situation."""

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.t: Tables = tables(cfg)
        self.scale = cfg.gold_scale
        self.plate_units = _units(cfg.plate_gold, cfg)
        self.bounty_units = _units(cfg.kill_bounty, cfg)
        self.theta_max = cfg.plate_thresholds[-1]
        self._coll: Dict[tuple, int] = {}
        self.inv = None
        if cfg.continuous_items:
            from .progression import investment
            self.inv = investment(cfg)

    def hit(self, base: int, level: int, items: int) -> int:
        """Centi-HP of a trade hit of `base` (0, 1, 2) base HP at the attacker's level and items slot."""
        return self.t.dmg[base][level][items] if self.inv is None else self.inv.hit_damage(base, level, items)

    def pressure(self, level: int, items: int) -> int:
        """Centi-units an effective Apply Pressure turn adds to the attacker's tower damage."""
        return self.t.pressure[level][items] if self.inv is None else self.inv.pressure_damage(level, items)

    def collection(self, x: int, mode: str, pos: int, opp_mode: str, opp_pos: int) -> int:
        key = (x, mode, pos, opp_mode, opp_pos)
        v = self._coll.get(key)
        if v is None:
            v = collection_units(x, mode, pos, opp_mode, opp_pos, self.cfg)
            self._coll[key] = v
        return v


def _units(x: float, cfg: GameConfig) -> int:
    v = Fraction(repr(float(x))) * cfg.gold_scale
    if v.denominator != 1:
        raise ValueError(f"{x!r} is not an integer number of ledger units (validate() should have rejected it)")
    return int(v)


@lru_cache(maxsize=32)
def context(cfg: GameConfig) -> M4Context:
    return M4Context(cfg)


def collection_units(x: int, mode: str, pos: int, opp_mode: str, opp_pos: int, cfg: GameConfig) -> int:
    """Exact ledger units per HP of a farm opportunity at own cell x: the M2 rule `champions.champion_collection`
    (reach, posture base, zoning, trade factor) times farm_per_hp, evaluated on exact rationals (design §8.6)."""
    if mode in (AWAY, CHANNEL) or abs(x - pos) > (cfg.retreat_farm_reach if mode == RETREAT else cfg.farm_reach):
        return 0
    q = lambda v: Fraction(repr(float(v)))
    T = cfg.tower
    if mode in (FARM, TRADING):
        base = q(cfg.collect_contest) if x < T else q(cfg.collect_contest_enemy_tower)
    else:
        base = exact_safe_base(x, cfg)
    if cfg.zoning_rule == "presence":                     # M2: an opponent farming or trading near the kill
        zones = opp_mode in (FARM, TRADING) and abs(-x - opp_pos) <= cfg.farm_reach
    else:                                                 # M4.2d-C: "contact" = the opponent's Trade connected on me
        zones = cfg.zoning_rule == "contact" and opp_mode == TRADING
    zone = (q(cfg.zone_contest) if x > -T else q(cfg.zone_under_tower)) if zones else Fraction(0)
    c = base * (1 - zone)
    if mode == TRADING:
        c *= q(cfg.trade_farm_factor)
    v = q(cfg.farm_per_hp) * c * cfg.gold_scale
    if v.denominator != 1:
        raise ValueError(f"farm value {float(v)} at cell {x} is not an integer number of ledger units")
    return int(v)


# ---------------------------------------------------------------------- state helpers
def initial_m4(cfg: GameConfig) -> M4State:
    """Both champions at their own tower, level 1 with full HP H(1), no XP, items, gold or tower damage."""
    t = tables(cfg)
    s = M4Side(-cfg.tower, t.hp_max[1], 0, 0, 0, 0, 0)
    return M4State(s, s)


def level(side: M4Side, t: Tables) -> int:
    x = side.xp
    return t.level_by_xp[x if x < t.xp_cap else t.xp_cap]


def contact_cell_target(c: int, ref: int, cfg: GameConfig) -> int:
    """Design §7.4: the cell nearest to the chaser's start c (own coordinates) from which it is in contact with an
    opponent at its own cell `ref`; ties go to the smaller cell. Never empty: for ref = -T it is {T} (the dive)."""
    T = cfg.tower
    best = None
    for x in range(-T, T + 1):
        if in_contact(x, ref, cfg):
            d = abs(x - c)
            if best is None or d < best[0]:
                best = (d, x)
    return best[1]


def m2_chase_target(c: int, ref: int, cfg: GameConfig) -> int:
    """M2's chase target (champions.resolve_turn step 2), for the "m2" chase rule and the WP4 classification."""
    T = cfg.tower
    e = -ref
    if in_contact(c, ref, cfg):
        return c
    if e == T:
        return T
    return e - cfg.trade_range if c < e else e + cfg.trade_range


def check_invariants(m4: M4State, cfg: GameConfig) -> None:
    """Design §3.4; raises AssertionError on a violation (soak tests)."""
    t = tables(cfg)
    for s in (A, B):
        me = m4.side(s)
        L = level(me, t)
        assert 0 <= me.hp <= t.hp_max[L], f"side {s}: hp {me.hp} outside [0, {t.hp_max[L]}]"
        assert me.hp > 0 or me.absent > 0, f"side {s}: hp 0 while present"
        assert me.absent >= 0 and (me.absent == 0 or me.pos == -cfg.tower), f"side {s}: away but at {me.pos}"
        assert -cfg.tower <= me.pos <= cfg.tower
        assert 0 <= me.xp <= t.xp_cap, f"side {s}: xp {me.xp}"
        if cfg.continuous_items:
            assert me.items >= 0 and me.gold >= 0, f"side {s}: invested {me.items}, gold {me.gold}"
        else:
            assert 0 <= me.items <= t.item_tiers, f"side {s}: items {me.items}"
            assert 0 <= me.gold <= t.gold_cap[me.items], f"side {s}: gold {me.gold} > G_rem({me.items})"
        assert 0 <= me.tower_dmg <= cfg.plate_thresholds[-1], f"side {s}: tower_dmg {me.tower_dmg}"


# ---------------------------------------------------------------------- one turn
@dataclass(frozen=True)
class M4Turn:
    lane: object                            # mechanics.TurnResult of the authoritative lane step
    champ: M4State                          # state after the turn
    waves: Tuple[int, int]                  # chosen wave components
    effective_waves: Tuple[int, int]        # after Freeze maintenance (what the lane step played)
    freeze_maintained: Tuple[bool, bool]    # chose Freeze and maintained it this turn
    behaviors: Tuple[int, int]
    objectives: Tuple[int, int]
    effective_objectives: Tuple[int, int]
    start: Tuple[int, int]
    end: Tuple[int, int]
    farm_spot: Tuple[int, int]
    away: Tuple[bool, bool]
    recalled: Tuple[bool, bool]
    interrupted: Tuple[bool, bool]
    threatened: Tuple[bool, bool]
    connected: Tuple[bool, bool]
    modes: Tuple[str, str]
    reach: Tuple[int, int]
    lane_codes: Tuple[int, int]
    farm_available: Tuple[float, float]     # resources (as M2)
    farm: Tuple[float, float]               # collected, resources (= farm_units / scale)
    exposure: Tuple[int, int]
    gank_p: Tuple[float, float]
    hit_p: Tuple[float, float]
    double_p: Tuple[float, float]
    tower_p: Tuple[float, float]            # P(tower shot): a dive or the tower-adjacent shot
    adjacent_shot: Tuple[bool, bool]        # side i's tower shot (if any) was the tower-adjacent one
    trade_damage: Tuple[int, int]           # centi-HP dealt BY side i's trade
    tower_damage: Tuple[int, int]           # centi-HP taken by side i from the enemy tower
    damage_taken: Tuple[int, int]           # all centi-HP lost by side i
    ganked: Tuple[bool, bool]
    gank_damage_p: Tuple[float, float]
    gank_damage: Tuple[int, int]            # centi-HP lost to the gank
    died: Tuple[bool, bool]
    bounty: Tuple[float, float]             # kill bounty received, resources
    # ---- progression and economy (design §5 steps 7b, 11-15)
    levels_before: Tuple[int, int]
    levels_after: Tuple[int, int]
    xp_gain: Tuple[int, int]                # XP credited (after the cap)
    xp_discarded: Tuple[int, int]           # XP beyond the cap
    level_up: Tuple[int, int]               # levels gained this turn
    tower_hits: Tuple[int, int]             # minion-damage units on the enemy tower
    tower_damage_dealt: Tuple[int, int]     # centi-units added to tower_dmg (after the clamp)
    plates_broken: Tuple[int, int]          # thresholds crossed this turn
    plate_gold: Tuple[float, float]         # resources
    purchases: Tuple[int, int]              # tiers bought at the end of this turn ("continuous": 1 per investment)
    returned: Tuple[bool, bool]             # absence timer reached 0 this turn (plays the next turn)
    farm_units: Tuple[int, int]             # ledger units
    plate_units: Tuple[int, int]
    bounty_units: Tuple[int, int]
    earned_units: Tuple[int, int]           # farm + plate gold + bounty
    gold_clamped: Tuple[int, int]           # ledger units removed by the G_rem clamp (gold_unspendable)
    reward_units: int                       # earned_A - earned_B
    reward: float                           # reward to A in resources: reward_units / gold_scale, converted once
    invested: Tuple[int, int] = (0, 0)      # ledger units invested at the end of this turn ("continuous" only)


def resolve_turn_m4(lane: State, m4: M4State, k: int, a: int, b: int, cfg: GameConfig,
                    lane_step: Optional[LaneStep] = None, u: Optional[Sequence[float]] = None,
                    ctx: Optional[M4Context] = None) -> M4Turn:
    """Resolve decision turn k of the M4 ruleset from both committed actions (design §5, steps 0-15). `u` is the
    turn's chance vector (N_CHANCE uniforms; None = no chance event). Legality is the caller's responsibility."""
    if ctx is None:
        ctx = context(cfg)
    t = ctx.t
    if lane_step is None:
        def lane_step(s, kk, x, y, reach):
            return step_turn(s, kk, x, y, cfg, champion_reach=reach)
    T = cfg.tower
    (wa, ba, oa), (wb, bb, ob) = _decode(a), _decode(b)
    w, beh, obj = (wa, wb), (ba, bb), (oa, ob)
    me0 = (m4.a, m4.b)
    c0 = (m4.a.pos, m4.b.pos)
    lv0 = (level(m4.a, t), level(m4.b, t))
    it0 = (m4.a.items, m4.b.items)
    recalling = (ba == B_RECALL, bb == B_RECALL)
    absent = (ba == B_ABSENT, bb == B_ABSENT)

    # 1. probe: wave destinations under full support of every champion that intends to play the lane (raw codes);
    #    farm_spot_rule "pre_crash_front": a front that crashes in the probe leaves the farm spot at its pre-turn cell
    off = (recalling[A] or absent[A], recalling[B] or absent[B])
    probe = lane_step(lane, k, wa + 6 * oa, wb + 6 * ob, (NO_REACH if off[A] else T, NO_REACH if off[B] else T))
    spot = [0, 0]
    for s in (A, B):
        fx, fh = probe.state.side(s)[:2]
        spot[s] = max(-T, min(T, fx)) if fh > 0 else c0[s]
        if cfg.farm_spot_rule == "pre_crash_front" and probe.crash_hp[s] > 0 and lane.side(s)[1] > 0:
            spot[s] = max(-T, min(T, lane.side(s)[0]))       # M4.2d: keep the reach of the wave that crashes

    # 2. targets and bounded movement; traders chase the nearest contact cell of their opponent's reference cell
    end = [0, 0]
    chasers = []
    for s in (A, B):
        c = c0[s]
        if absent[s]:
            end[s] = -T
            continue
        if recalling[s]:
            end[s] = c
            continue
        if obj[s] != NONE:
            target = T
        elif beh[s] == B_TRADE and not absent[1 - s]:
            chasers.append(s)
            continue
        elif beh[s] == B_SAFE:
            target = min(spot[s], cfg.safe_line)
        else:
            target = spot[s]
        end[s] = _move(c, target, cfg)
    for s in chasers:
        o, c = 1 - s, c0[s]
        ref = c0[o] if beh[o] == B_TRADE and obj[o] == NONE else end[o]
        target = contact_cell_target(c, ref, cfg) if cfg.chase_rule == "contact_cell" else m2_chase_target(c, ref, cfg)
        end[s] = _move(c, target, cfg)

    # 2b. recall interruption
    interrupted = (recalling[A] and beh[B] == B_TRADE and in_contact(end[A], end[B], cfg),
                   recalling[B] and beh[A] == B_TRADE and in_contact(end[B], end[A], cfg))
    recalled = (recalling[A] and not interrupted[A], recalling[B] and not interrupted[B])
    away = (absent[A] or recalled[A], absent[B] or recalled[B])
    for s in (A, B):
        if away[s]:
            end[s] = -T

    # 3. evasion
    threatened = [False, False]
    for s in (A, B):
        o = 1 - s
        if beh[s] in (B_YIELD, B_SAFE) and beh[o] == B_TRADE and in_contact(end[s], end[o], cfg):
            threatened[s] = True
            own_target = spot[s] if beh[s] == B_YIELD else min(spot[s], cfg.safe_line)
            end[s] = _move(c0[s], _evasion_cell(own_target, end[o], cfg), cfg)

    # 4. trade contact
    in_range = not (away[A] or away[B]) and in_contact(end[A], end[B], cfg)
    connected = (beh[A] == B_TRADE and in_range, beh[B] == B_TRADE and in_range)
    eff_obj = (NONE if connected[A] else obj[A], NONE if connected[B] else obj[B])

    # 5. modes and wave contribution, then Freeze maintenance
    modes, reach = [], []
    for s in (A, B):
        if away[s] or interrupted[s]:
            modes.append(AWAY if away[s] else CHANNEL)
            reach.append(NO_REACH)
        elif connected[s]:
            modes.append(TRADING)
            reach.append(NO_REACH)
        elif beh[s] == B_SAFE or threatened[s]:
            modes.append(SAFE_MODE if beh[s] == B_SAFE else RETREAT)
            reach.append(min(end[s], cfg.safe_line))
        else:
            modes.append(FARM)
            reach.append(min(T, end[s] + cfg.farm_reach))
    maintained = tuple(w[s] == F and not away[s] and not interrupted[s] and not connected[s]
                       and abs(end[s] - cfg.freeze_line) <= cfg.farm_reach for s in (A, B))
    if cfg.freeze_maintenance:
        eff_w = tuple(w[s] if w[s] != F else (F if maintained[s] else SP) for s in (A, B))
    else:
        eff_w = w

    # 6. the authoritative lane step (effective waves and objectives)
    codes = (eff_w[A] + 6 * eff_obj[A], eff_w[B] + 6 * eff_obj[B])
    res = lane_step(lane, k, codes[A], codes[B], (reach[A], reach[B]))

    # 7. gold (exact ledger units) and 7b. XP
    avail, farm_u, xp_gain_raw = [0.0, 0.0], [0, 0], [0, 0]
    for s in (A, B):
        o = 1 - s
        for x, hp in res.kills[s].items():
            avail[s] += cfg.farm_per_hp * hp
            farm_u[s] += hp * ctx.collection(x, modes[s], end[s], modes[o], end[o])
            if not away[s] and abs(x - end[s]) <= cfg.xp_range:
                xp_gain_raw[s] += cfg.xp_per_hp * hp

    # 8. exposure and gank occurrence
    exposure = tuple(-T if away[s] else max(c0[s], end[s]) for s in (A, B))
    gank = tuple(0.0 if away[s] else
                 champion_gank_probability(lane, s, w[s], B_SAFE if interrupted[s] else beh[s], cfg, exposure[s])
                 for s in (A, B))

    # 9. chance events, centi-HP damage at the start-of-turn levels and items
    if u is None:
        u = (1.0,) * N_CHANCE
    hit, dbl, tower_p = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
    dealt, tower_dmg, adjacent = [0, 0], [0, 0], [False, False]
    for s in (A, B):
        if not connected[s]:
            continue
        o = 1 - s
        hit[s] = trade_hit_probability(lane, s, B_RECALL if interrupted[o] else beh[o], end[o] == -T, cfg)
        dbl[s] = trade_double_probability(s, cfg)
        d = _trade_damage(u[U_TRADE_HIT[s]], u[U_TRADE_DOUBLE[s]], hit[s], dbl[s])
        dealt[s] = ctx.hit(d, lv0[s], it0[s])
        busy = lane.siege_of(s) > 0 or res.crash_hp[s] > 0
        if end[s] == T:                                                   # a dive into the enemy tower cell
            tower_p[s] = cfg.tower_shot_p_busy if busy else cfg.tower_shot_p
        elif end[s] == T - 1 and (cfg.tower_adjacent_p > 0 or cfg.tower_adjacent_p_busy > 0):
            tower_p[s] = cfg.tower_adjacent_p_busy if busy else cfg.tower_adjacent_p
            adjacent[s] = True
        if u[U_TOWER[s]] < tower_p[s]:
            tower_dmg[s] = cfg.tower_shot_damage * t.hp_unit[lv0[s]]
    ganked = (u[U_GANK[A]] < gank[A], u[U_GANK[B]] < gank[B])
    pe = tuple(0.0 if away[s] else gank_damage_p(s, modes[s] in (RETREAT, SAFE_MODE), exposure[s], cfg)
               for s in (A, B))
    gank_dmg = [0, 0]
    for s in (A, B):
        if ganked[s]:
            if cfg.gank_damage_scale == "victim_level":
                unit = t.hp_unit[lv0[s]]
            elif cfg.gank_damage_scale == "attacker_level":
                unit = t.hp_unit[lv0[1 - s]]
            else:
                unit = t.hp_unit[1]
            gank_dmg[s] = _gank_damage(u[U_GANK_DAMAGE[s]], pe[s], cfg) * unit
    taken = (dealt[B] + tower_dmg[A] + gank_dmg[A], dealt[A] + tower_dmg[B] + gank_dmg[B])

    # 10. death (before level-up)
    died = (not away[A] and me0[A].hp - taken[A] <= 0, not away[B] and me0[B].hp - taken[B] <= 0)
    bounty_u = (ctx.bounty_units if died[B] else 0, ctx.bounty_units if died[A] else 0)

    # 11. tower damage and plates
    tau, hits_dealt, plates, plate_u = [0, 0], [0, 0], [0, 0], [0, 0]
    for s in (A, B):
        before = me0[s].tower_dmg
        add = 100 * res.tower_hits[s]
        if eff_obj[s] == PRESSURE:
            add += ctx.pressure(lv0[s], it0[s])
        tau[s] = min(ctx.theta_max, before + add)
        hits_dealt[s] = tau[s] - before
        plates[s] = sum(1 for th in cfg.plate_thresholds if before < th <= tau[s])
        if plates[s] and not (cfg.plate_gold_requires_presence and away[s]):
            plate_u[s] = plates[s] * ctx.plate_units

    # 12-14. XP and level-ups, the gold ledger, absence bookkeeping and purchases
    sides, xp_gain, xp_disc, lv1, purchases, returned, clamped, earned = [], [0, 0], [0, 0], [0, 0], [0, 0], \
        [False, False], [0, 0], [0, 0]
    continuous = ctx.inv is not None
    invested = [0, 0]
    for s in (A, B):
        me = me0[s]
        xp = me.xp
        if not away[s]:
            raw = xp + xp_gain_raw[s]
            xp = raw if raw < t.xp_cap else t.xp_cap
            xp_gain[s], xp_disc[s] = xp - me.xp, raw - xp
        L1 = t.level_by_xp[xp]
        lv1[s] = L1
        earned[s] = farm_u[s] + plate_u[s] + bounty_u[s]
        items = me.items
        gold = me.gold + earned[s]
        if not continuous:
            cap = t.gold_cap[items]
            if gold > cap:
                clamped[s] = gold - cap
                gold = cap
        if away[s] or died[s]:
            left = (cfg.recall_turns - 1 if recalled[s] else me.absent - 1) if away[s] else cfg.death_turns
            hp = me.hp if away[s] else 0
            if left == 0:                                                 # back next turn: shop, then full HP
                returned[s] = True
                if continuous:                                            # invest everything, however little
                    invested[s], items, gold = gold, items + gold, 0
                    purchases[s] = 1 if invested[s] else 0
                while items < t.item_tiers and gold >= t.prices[items]:
                    gold -= t.prices[items]
                    items += 1
                    purchases[s] += 1
                hp = t.hp_max[L1]
            sides.append(M4Side(-T, hp, left, xp, items, gold, tau[s]))
        else:
            hp = me.hp - taken[s] + t.hp_max[L1] - t.hp_max[lv0[s]]
            sides.append(M4Side(end[s], hp, 0, xp, items, gold, tau[s]))

    # 15. reward (exact ledger units)
    reward_units = earned[A] - earned[B]
    inv = 1.0 / ctx.scale
    turn = M4Turn(
        res, M4State(sides[A], sides[B]), w, eff_w, maintained, beh, obj, eff_obj, c0, (end[A], end[B]),
        (spot[A], spot[B]), away, recalled, interrupted, (threatened[A], threatened[B]), connected,
        (modes[A], modes[B]), (reach[A], reach[B]), codes, (avail[A], avail[B]),
        (farm_u[A] * inv, farm_u[B] * inv), exposure, gank, (hit[A], hit[B]), (dbl[A], dbl[B]),
        (tower_p[A], tower_p[B]), (adjacent[A], adjacent[B]), (dealt[A], dealt[B]), (tower_dmg[A], tower_dmg[B]),
        taken, ganked, pe, (gank_dmg[A], gank_dmg[B]), died, (bounty_u[A] * inv, bounty_u[B] * inv),
        lv0, (lv1[A], lv1[B]), (xp_gain[A], xp_gain[B]), (xp_disc[A], xp_disc[B]),
        (lv1[A] - lv0[A], lv1[B] - lv0[B]), res.tower_hits, (hits_dealt[A], hits_dealt[B]), (plates[A], plates[B]),
        (plate_u[A] * inv, plate_u[B] * inv), (purchases[A], purchases[B]), (returned[A], returned[B]),
        (farm_u[A], farm_u[B]), (plate_u[A], plate_u[B]), bounty_u, (earned[A], earned[B]), (clamped[A], clamped[B]),
        reward_units, reward_units / ctx.scale, (invested[A], invested[B]))
    return turn
