"""M4 champion state and turn resolution (docs/m4/m4_1/m4_integrated_design.md; plan WP4-WP6).

A separately versioned resolver for `GameConfig.m4()`. The M2 resolver `champions.resolve_turn` is not edited;
this module reuses its pure helpers (contact, movement, evasion, farm collection, gank and trade probabilities,
legality) and adds, in the order of design §5:

  step 1   opt-in (M4.2d, `farm_spot_rule="pre_crash_front"`): a champion whose front crashes in the probe keeps its
           pre-turn front cell as farm spot instead of walking back to the next packet
  step 2   the contact-cell chase rule (§7.4; `chase_rule="contact_cell"`); opt-in (M4.2d-B, `position_rule=
           "standoff"`): Yield farms from its stand-off cell, the rearmost cell within farm_reach of the enemy front
           (`yield_target`)
  step 3   opt-in (M4.2d-B, `position_rule` "pin" or "standoff"): a Yield champion that starts the turn in contact
           with an enemy Trade champion is pinned: it cannot evade and is traded exactly like a Hold
  step 5   Freeze maintenance (§6.3): a Freeze is effective only while the freezer is present, not channeling, not
           in a connected trade and ends within farm_reach of the freeze line; otherwise it resolves as Slow Push
  step 6   the lane step with the effective waves, which also returns the minion tower hits (§11.2)
  step 7   gold (M2 collection rule) as exact integers in ledger units (micro-resources, §8.6); opt-in (M4.2d-C,
           `zoning_rule`): the zone factor needs a connected enemy Trade ("contact") or never applies ("none")
  step 7b  XP: xp_per_hp per enemy minion HP dying within xp_range cells of a present champion (§7.3)
  step 9   centi-HP damage: trade hits dmg(D, L, N), gank trials and tower shots of one HP unit u(L) of the victim,
           the tower-adjacent shot on a connected trader ending next to the enemy tower (§8.3, §8.4). M4.2e-B opt-ins:
           `trade_exchanges` > 1 resolves a connected Trade once per tick (D = 0 .. 2n); `trade_farm_rule` "loser"
           charges the trade farm factor of a mutual trade to its loser only (the rolls are therefore made before
           step 7); `invest_durability` divides every centi-HP taken by the victim's investment multiplier;
           `collect_contest_own_tower` caps the collection rate of kills in the own tower cell
  step 10  death before level-up (§5)
  step 11  persistent tower damage and plate gold (§11)
  step 12  XP and level-ups: a surviving present champion gains the increase in max HP (§9.2)
  step 13  the gold ledger, clamped at G_rem(N) (§8.6)
  step 14  absence bookkeeping: a champion whose timer reaches 0 buys its tiers, then returns with full HP (§8.5);
           M4.3c opt-in `death_turns_by_level`: the absence after a death depends on the start-of-turn level
  step 15  reward r_A = earned_A - earned_B (farm + plate gold + kill bounty), exact in ledger units; M4.3c opt-ins
           (EXPERIMENTAL, docs/m4/m4_3/m4_3c_strategic_economics_calibration.md): `xp_reward_weight` lambda adds
           lambda (xp_gain_A - xp_gain_B) in ledger units (M4Turn.xp_reward_units); `terminal_rule = "final_plate"`
           marks the turn in which either tower's damage reaches its last plate threshold as terminal (M4Turn.terminal)

M4.5a opt-ins (EXPERIMENTAL, `GameConfig.m45_unified()`, docs/m4/m4_5/m4_5a_unified_mechanics_spec.md; every default
reproduces the earlier presets): `vision_rule = "ward"` adds the shared Vision node W (lanegame.positional.WARD) and the
Ward action (code 30): the warder walks to W (mode "ward": no wave damage, no gold), and its ward is placed at the end of
the turn unless the enemy's connected Trade damaged it or it died; movement, contact, chase targets, exposure, gold and
XP distances follow the movement graph. The trade rolls (9a) are made before the lane step (they read only the
start-of-turn state, so every earlier preset is unchanged). `trade_wave_rule = "capped"`: a connected trader keeps its
farm-mode reach and spends min(B_w, round(trade_wave_fraction C)). `wave_reach_rule = "ball"`: reach intervals
(positional.reach_interval). `minion_retaliation` / `minion_positional_aggro`: the lane step receives each side's
positional.MinionTarget (the enemy champion's end cell and, after positive trade damage, the retaliation tick) and
returns the minion damage units each champion took (TurnResult.champion_hits), converted here into centi-HP at the
victim's start-of-turn level (positional.minion_damage_centi) and summed with the turn's other damage (step 9b).
`champion_movement_rule = "tick"` (the M4.5a correction, step 5c): every champion reaches its unchanged end cell tick
by tick (positional.tick_path); Hold, an unthreatened Yield and Safe head on each tick for their behaviour's target
from the probe's front after that tick (`_tick_goals`), so they walk with their wave; the cells go to the lane step in
MinionTarget.path and are reported in M4Turn.path.

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

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

from .champions import (AWAY, CHANNEL, FARM, N_CHANCE, NO_REACH, RETREAT, SAFE_MODE, TRADING, U_GANK, U_GANK_DAMAGE,
                        U_TOWER, U_TRADE_DOUBLE, U_TRADE_HIT, LaneStep, _decode, _evasion_cell, _gank_damage, _move,
                        _trade_damage, champion_gank_probability, champion_legal_actions, gank_damage_p, in_contact,
                        objective_reachable, recall_legal, trade_double_probability, trade_hit_probability)
from .config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, B_ABSENT, F, NONE, PRESSURE, SH, SP, VISION,
                     W_SLOWPUSH, GameConfig, exact_safe_base)
from .mechanics import step_turn, strength_budget
from .positional import (WARD, MinionTarget, dist as graph_dist, exposure_cell, minion_damage_centi, mirror_cell,
                         reach_interval, tick_path)
from .progression import Tables, tables
from .state import State

A, B = 0, 1
WARD_MODE = "ward"                      # M4.5a: placing a ward at W (no wave damage, no gold; XP as present)

__all__ = ["M4Side", "M4State", "M4Turn", "M4Context", "context", "initial_m4", "resolve_turn_m4",
           "champion_legal_actions", "objective_reachable", "recall_legal", "check_invariants", "level",
           "contact_cell_target", "collection_units", "yield_target", "trade_damage_n", "trade_damage_distribution_n"]


class M4Side(NamedTuple):
    pos: int              # own coordinates, -T .. T (-T while away); positional.WARD (99) = the Vision node W (M4.5a)
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
        self.xp_units = _units(cfg.xp_reward_weight, cfg)        # M4.3c: reward ledger units per credited XP (0 = off)
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

    def wave_budget(self, slot: int, level: int, items: int) -> int:
        """Natural waves, wave_clear_scaling "attack_power": the champion's wave-damage budget for one turn in minion
        HP, round(ticks_per_turn * champion_wave_damage[slot] * AP / 10^4) (round half up, exact integers); slots
        outside wave_clear_scaled_slots keep ticks_per_turn * champion_wave_damage[slot]. "capacity" (M4.2e-C): the
        slot's share of the Clear capacity, `budget_from_capacity`."""
        cfg = self.cfg
        if cfg.wave_clear_scaling == "capacity":
            return self.budget_from_capacity(slot, self._scaled(SH, level, items))
        if slot not in cfg.wave_clear_scaled_slots:
            return cfg.ticks_per_turn * cfg.champion_wave_damage[slot]
        return self._scaled(slot, level, items)

    def _scaled(self, slot: int, level: int, items: int) -> int:
        """round(ticks_per_turn * champion_wave_damage[slot] * AP / 10^4), round half up in exact integers."""
        cfg = self.cfg
        base = cfg.ticks_per_turn * cfg.champion_wave_damage[slot]
        if self.inv is None:
            num, den = base * self.t.ap[level][items], 10_000
        else:
            num, den = base * self.inv.attack_power(level, items), 10_000 * self.inv.one
        return (2 * num + den) // (2 * den)

    def capacity(self, level: int, items: int) -> int:
        """wave_clear_scaling "capacity": the wave-clear capacity C (Clear's budget) of a champion at (level, items)."""
        return self._scaled(SH, level, items)

    def budget_from_capacity(self, slot: int, capacity: int) -> int:
        """wave_clear_scaling "capacity" (M4.2e-C): the per-turn budget of a wave slot as a share of the wave-clear
        capacity C: Clear = C, Hold = round(wave_hold_fraction C), Thin = round(wave_thin_fraction C); round half up on
        exact rationals, so Hold <= Thin <= Clear and every budget is non-decreasing in C. wave_hold_cap n > 0
        (EXPERIMENTAL) caps Hold at n; wave_thin_rule "adaptive" caps Thin further in the lane step (mechanics.thin_need).
        Slow Push (wave_slowpush, EXPERIMENTAL) has Thin's capacity round(wave_thin_fraction C); under wave_thin_rule
        "strength" this is only the cap of both (`state_budget`)."""
        if slot == SH:
            return capacity
        share = self.cfg.wave_hold_fraction if slot == SP else self.cfg.wave_thin_fraction
        b = _round_half_up(_q(share) * capacity)
        if slot == SP and self.cfg.wave_hold_cap > 0:
            b = min(b, self.cfg.wave_hold_cap)
        return b

    def state_budget(self, slot: int, capacity: int, lane: State, k: int, s: int) -> int:
        """wave_clear_scaling "capacity": the budget side s's wave slot spends on turn k from the start-of-turn lane.
        It is `budget_from_capacity` except under wave_thin_rule "strength" (M4.3c, EXPERIMENTAL), where Thin and Slow
        Push spend only what their target needs on top of the Last Hit (Hold) budget, up to that capacity
        (mechanics.strength_budget)."""
        b = self.budget_from_capacity(slot, capacity)
        if self.cfg.wave_thin_rule == "strength" and slot in (F, W_SLOWPUSH):
            b = strength_budget(lane, k, s, slot, self.budget_from_capacity(SP, capacity), b, self.cfg)
        return b

    def collection(self, x: int, mode: str, pos: int, opp_mode: str, opp_pos: int, pays_trade: bool = True) -> int:
        key = (x, mode, pos, opp_mode, opp_pos, pays_trade)
        v = self._coll.get(key)
        if v is None:
            v = collection_units(x, mode, pos, opp_mode, opp_pos, self.cfg, pays_trade)
            self._coll[key] = v
        return v

    def death_turns(self, level: int) -> int:
        """Turns a champion that dies at `level` sits out: death_turns, or death_turns_by_level[level - 1] (M4.3c)."""
        dtl = self.cfg.death_turns_by_level
        return dtl[level - 1] if dtl else self.cfg.death_turns

    def mitigate(self, damage: int, items: int) -> int:
        """invest_durability: centi-HP actually lost by a champion with `items` invested from `damage` centi-HP of
        incoming damage, floor(damage / M(S)) in exact integers; unchanged when the option is off."""
        if self.inv is None or not self.cfg.invest_durability or damage == 0:
            return damage
        return damage * self.inv.one // self.inv.multiplier(items)


def _q(x: float) -> Fraction:
    return Fraction(repr(float(x)))


def _round_half_up(v: Fraction) -> int:
    return (2 * v.numerator + v.denominator) // (2 * v.denominator)


def _units(x: float, cfg: GameConfig) -> int:
    v = Fraction(repr(float(x))) * cfg.gold_scale
    if v.denominator != 1:
        raise ValueError(f"{x!r} is not an integer number of ledger units (validate() should have rejected it)")
    return int(v)


@lru_cache(maxsize=32)
def context(cfg: GameConfig) -> M4Context:
    return M4Context(cfg)


def collection_units(x: int, mode: str, pos: int, opp_mode: str, opp_pos: int, cfg: GameConfig,
                     pays_trade: bool = True) -> int:
    """Exact ledger units per HP of a farm opportunity at own cell x: the M2 rule `champions.champion_collection`
    (reach, posture base, zoning, trade factor) times farm_per_hp, evaluated on exact rationals (design §8.6).
    M4.2e-B: a kill in the own tower cell (x = -T) is collected at most at collect_contest_own_tower; a connected
    trader pays trade_farm_factor only if `pays_trade` (trade_farm_rule "loser": the winner of a mutual trade keeps
    its farm)."""
    if mode in (AWAY, CHANNEL, WARD_MODE) or \
            graph_dist(x, pos) > (cfg.retreat_farm_reach if mode == RETREAT else cfg.farm_reach):
        return 0
    q = lambda v: Fraction(repr(float(v)))
    T = cfg.tower
    if mode in (FARM, TRADING):
        base = q(cfg.collect_contest) if x < T else q(cfg.collect_contest_enemy_tower)
    else:
        base = exact_safe_base(x, cfg)
    if x == -T:
        base = min(base, q(cfg.collect_contest_own_tower))
    if cfg.zoning_rule == "presence":                     # M2: an opponent farming or trading near the kill
        zones = opp_mode in (FARM, TRADING) and graph_dist(-x, opp_pos) <= cfg.farm_reach
    else:                                                 # M4.2d-C: "contact" = the opponent's Trade connected on me
        zones = cfg.zoning_rule == "contact" and opp_mode == TRADING
    zone = (q(cfg.zone_contest) if x > -T else q(cfg.zone_under_tower)) if zones else Fraction(0)
    c = base * (1 - zone)
    if mode == TRADING and pays_trade:
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
    cells = tuple(range(-T, T + 1)) + ((WARD,) if cfg.vision_rule == "ward" else ())   # M4.5a: W last (ties)
    for x in cells:
        if in_contact(x, ref, cfg):
            d = graph_dist(x, c)
            if best is None or d < best[0]:
                best = (d, x)
    return best[1]


def yield_target(spot: int, lane: State, s: int, cfg: GameConfig) -> int:
    """Where a Yield champion of side s farms (own coordinates). Its farm spot, or under position_rule "standoff" its
    stand-off cell: the rearmost cell still within farm_reach of the enemy's leading minions at the start of the turn
    (enemy front e, mirrored into own coordinates), never ahead of the farm spot, never behind the own tower:
    max(-T, min(spot, e - farm_reach)). With no enemy front on the lane it is the farm spot
    (docs/m4/m4_2/m4_positioning_report.md)."""
    if cfg.position_rule != "standoff":
        return spot
    ex, eh = lane.side(1 - s)[:2]
    if eh <= 0:
        return spot
    return max(-cfg.tower, min(spot, -ex - cfg.farm_reach))


def trade_damage_distribution_n(h: float, q: float, n: int) -> Tuple[float, ...]:
    """P(total base damage = d), d = 0 .. 2n, of n independent exchanges each dealing 0 / 1 / 2 base HP with
    probabilities (1 - h, h (1 - q), h q) (the n-fold convolution of champions.trade_damage_distribution)."""
    dist = [1.0]
    for _ in range(n):
        nxt = [0.0] * (len(dist) + 2)
        for i, p in enumerate(dist):
            nxt[i] += p * (1.0 - h)
            nxt[i + 1] += p * h * (1.0 - q)
            nxt[i + 2] += p * h * q
        dist = nxt
    return tuple(dist)


def trade_damage_n(u: float, h: float, q: float, n: int) -> int:
    """Inverse-CDF draw of the total base damage of n exchanges from one uniform (trade_exchanges > 1): the largest d
    with u < P(D >= d), so that, as for every other chance event, a uniform at or above the event probability means no
    event (u = 1.0, the no-chance convention, deals 0)."""
    c = 0.0
    dist = trade_damage_distribution_n(h, q, n)
    for d in range(len(dist) - 1, 0, -1):
        c += dist[d]
        if u < c:
            return d
    return 0


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
        assert -cfg.tower <= me.pos <= cfg.tower or (cfg.vision_rule == "ward" and me.pos == WARD), \
            f"side {s}: position {me.pos}"
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
    pinned: Tuple[bool, bool] = (False, False)   # position_rule "pin"/"standoff": Yield caught in contact (step 3)
    # natural waves with wave_clear_scaling: the per-turn wave budgets (minion HP) given to the lane step (None =
    # champion_wave_damage per tick); what each champion could spend, not what it reached
    wave_budget: Optional[Tuple[int, int]] = None
    # M4.3c (EXPERIMENTAL): the XP channel of the reward, xp_reward_weight (in ledger units per XP) times
    # xp_gain_A - xp_gain_B, already included in reward_units; and terminal_rule "final_plate": this turn ended the game
    xp_reward_units: int = 0
    terminal: bool = False
    # M4.5a (EXPERIMENTAL): Ward chosen / placed / interrupted (by positive damage from the enemy's connected Trade, or a
    # death); minion damage units each champion took and their centi-HP (before the durability division); per side,
    # whether its wave was eligible to retaliate (its champion took positive trade damage); per champion, the cause of
    # the minion fire it took on each tick; whether a connected trader's wave budget was capped
    warding: Tuple[bool, bool] = (False, False)
    ward_placed: Tuple[bool, bool] = (False, False)
    ward_interrupted: Tuple[bool, bool] = (False, False)
    minion_hits: Tuple[int, int] = (0, 0)
    minion_damage: Tuple[int, int] = (0, 0)
    retaliation: Tuple[bool, bool] = (False, False)
    minion_fire: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None
    trade_wave_capped: Tuple[bool, bool] = (False, False)
    minion_targets: Optional[tuple] = None  # the positional.MinionTarget per side given to the lane step (None: off)
    # champion_movement_rule "tick" (else None): per champion, its own cell after each tick (the last = end; a recaller
    # channels in its cell, an absent champion is at -T)
    path: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None


def resolve_turn_m4(lane: State, m4: M4State, k: int, a: int, b: int, cfg: GameConfig,
                    lane_step: Optional[LaneStep] = None, u: Optional[Sequence[float]] = None,
                    ctx: Optional[M4Context] = None) -> M4Turn:
    """Resolve decision turn k of the M4 ruleset from both committed actions (design §5, steps 0-15). `u` is the
    turn's chance vector (N_CHANCE uniforms; None = no chance event). Legality is the caller's responsibility."""
    if ctx is None:
        ctx = context(cfg)
    t = ctx.t
    if lane_step is None:
        def lane_step(s, kk, x, y, reach, budget=None, targets=None):
            return step_turn(s, kk, x, y, cfg, champion_reach=reach, champion_budget=budget, champion_targets=targets)
    T = cfg.tower
    # natural waves: a champion standing in the enemy tower cell reaches natural_tower_reach cells beyond it
    reach_cap = T + (cfg.natural_tower_reach if cfg.natural_waves else 0)
    (wa, ba, oa), (wb, bb, ob) = _decode(a), _decode(b)
    w, beh, obj = (wa, wb), (ba, bb), (oa, ob)
    me0 = (m4.a, m4.b)
    c0 = (m4.a.pos, m4.b.pos)
    lv0 = (level(m4.a, t), level(m4.b, t))
    it0 = (m4.a.items, m4.b.items)
    recalling = (ba == B_RECALL, bb == B_RECALL)
    absent = (ba == B_ABSENT, bb == B_ABSENT)
    ward_rule = cfg.vision_rule == "ward"
    warding = (ward_rule and oa == VISION, ward_rule and ob == VISION)     # M4.5a: the Ward code (legality: caller)
    lane_obj = (NONE if warding[A] else oa, NONE if warding[B] else ob)    # a ward is no lane-step objective

    # 1. probe: wave destinations under full support of every champion that intends to play the lane (raw codes);
    #    farm_spot_rule "pre_crash_front": a front that crashes in the probe leaves the farm spot at its pre-turn cell
    off = (recalling[A] or absent[A] or warding[A], recalling[B] or absent[B] or warding[B])
    # wave_clear_scaling "attack_power": per-turn wave-damage budgets from the start-of-turn attack power; "capacity":
    # from the start-of-turn capacity and (wave_thin_rule "strength") the start-of-turn lane
    budget = None
    if cfg.natural_waves and cfg.wave_clear_scaling == "capacity":
        budget = (ctx.state_budget(wa, ctx.capacity(lv0[A], it0[A]), lane, k, A),
                  ctx.state_budget(wb, ctx.capacity(lv0[B], it0[B]), lane, k, B))
    elif cfg.natural_waves and cfg.wave_clear_scaling != "none":
        budget = (ctx.wave_budget(wa, lv0[A], it0[A]), ctx.wave_budget(wb, lv0[B], it0[B]))

    def lane_call(x, y, rch, bud=None, targets=None):
        bud = budget if bud is None else bud
        if targets is not None:                                       # M4.5a minion targeting
            return lane_step(lane, k, x, y, rch, bud, targets)
        return lane_step(lane, k, x, y, rch) if bud is None else lane_step(lane, k, x, y, rch, bud)
    # lane codes are w + 6 o; with budgets the natural lane step reads only the objective o from them (a wave slot acts
    # through its budget alone), so the four-action Slow Push (w = W_SLOWPUSH = 3) passes 3 + 6 o
    probe = lane_call(wa + 6 * lane_obj[A], wb + 6 * lane_obj[B],
                      (NO_REACH if off[A] else reach_cap, NO_REACH if off[B] else reach_cap))
    spot = [0, 0]
    for s in (A, B):
        fx, fh = probe.state.side(s)[:2]
        spot[s] = max(-T, min(T, fx)) if fh > 0 else (0 if c0[s] == WARD else c0[s])   # M4.5a: from W, the lane at 0
        if cfg.farm_spot_rule == "pre_crash_front" and probe.crash_hp[s] > 0 and lane.side(s)[1] > 0:
            spot[s] = max(-T, min(T, lane.side(s)[0]))       # M4.2d: keep the reach of the wave that crashes
        elif cfg.farm_spot_rule == "last_front":             # M4.5b: a front lost in the probe keeps its last cell
            last = _last_front_cell(lane, probe, s)
            if last is not None:
                spot[s] = max(-T, min(T, last))

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
        if warding[s]:
            target = WARD                                               # M4.5a: the Vision node
        elif obj[s] != NONE:
            target = T
        elif beh[s] == B_TRADE and not absent[1 - s]:
            chasers.append(s)
            continue
        elif beh[s] == B_SAFE:
            target = min(spot[s], cfg.safe_line)
        elif beh[s] == B_YIELD:
            target = yield_target(spot[s], lane, s, cfg)
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

    # 3. evasion; position_rule "pin"/"standoff": a Yield that starts in the trader's contact is pinned (no evasion)
    threatened, pinned = [False, False], [False, False]
    pin = cfg.position_rule != "m2"
    for s in (A, B):
        o = 1 - s
        if beh[s] in (B_YIELD, B_SAFE) and beh[o] == B_TRADE and in_contact(end[s], end[o], cfg):
            if pin and beh[s] == B_YIELD and in_contact(c0[s], c0[o], cfg):
                pinned[s] = True
                continue
            threatened[s] = True
            own_target = yield_target(spot[s], lane, s, cfg) if beh[s] == B_YIELD else min(spot[s], cfg.safe_line)
            end[s] = _move(c0[s], _evasion_cell(own_target, end[o], cfg), cfg)

    # 4. trade contact
    in_range = not (away[A] or away[B]) and in_contact(end[A], end[B], cfg)
    connected = (beh[A] == B_TRADE and in_range, beh[B] == B_TRADE and in_range)
    eff_obj = (NONE if connected[A] else obj[A], NONE if connected[B] else obj[B])

    # 9a. the trade rolls (chance vector; the hit probability uses the start-of-turn lane, levels and items). They read
    #     only the start-of-turn state and the end cells, so they are made before the lane step: trade_farm_rule "loser"
    #     charges the farm factor by the trade's outcome, and M4.5a minion retaliation needs the damage of the exchange
    #     (which lands on tick 0). A warder (M4.5a) is traded like a Hold champion.
    if u is None:
        u = (1.0,) * N_CHANCE
    hit, dbl = [0.0, 0.0], [0.0, 0.0]
    dealt = [0, 0]
    for s in (A, B):
        if not connected[s]:
            continue
        o = 1 - s
        hit[s] = trade_hit_probability(lane, s, B_RECALL if interrupted[o] else (B_HOLD if pinned[o] else beh[o]),
                                       end[o] == -T, cfg)
        dbl[s] = trade_double_probability(s, cfg)
        if cfg.trade_exchanges == 1:
            d = _trade_damage(u[U_TRADE_HIT[s]], u[U_TRADE_DOUBLE[s]], hit[s], dbl[s])
        else:                                                              # M4.2e-B: one exchange per tick
            d = trade_damage_n(u[U_TRADE_HIT[s]], hit[s], dbl[s], cfg.trade_exchanges)
        dealt[s] = ctx.hit(d, lv0[s], it0[s])
    pays = [True, True]
    if cfg.trade_farm_rule == "loser" and connected[A] and connected[B]:
        # the side that lost the smaller fraction of its max HP keeps its farm (exact cross-multiplication)
        lost_a = ctx.mitigate(dealt[B], it0[A]) * t.hp_max[lv0[B]]
        lost_b = ctx.mitigate(dealt[A], it0[B]) * t.hp_max[lv0[A]]
        if lost_a < lost_b:
            pays[A] = False
        elif lost_b < lost_a:
            pays[B] = False

    # 5. modes and wave contribution, then Freeze maintenance (M4.5a: "ward" mode; a connected trader keeps a capped
    #    wave budget under trade_wave_rule "capped"; reach intervals under wave_reach_rule "ball")
    ball = cfg.wave_reach_rule == "ball"
    capped = cfg.trade_wave_rule == "capped"

    def farm_reach_of(s):
        return reach_interval(end[s], False, cfg.farm_reach, reach_cap, cfg.safe_line) if ball else \
            min(reach_cap, end[s] + cfg.farm_reach)
    modes, reach = [], []
    for s in (A, B):
        if away[s] or interrupted[s]:
            modes.append(AWAY if away[s] else CHANNEL)
            reach.append(NO_REACH)
        elif warding[s]:
            modes.append(WARD_MODE)
            reach.append(NO_REACH)
        elif connected[s]:
            modes.append(TRADING)
            reach.append(farm_reach_of(s) if capped else NO_REACH)
        elif beh[s] == B_SAFE or threatened[s]:
            modes.append(SAFE_MODE if beh[s] == B_SAFE else RETREAT)
            reach.append(reach_interval(end[s], True, cfg.farm_reach, reach_cap, cfg.safe_line) if ball else
                         min(end[s], cfg.safe_line))
        else:
            modes.append(FARM)
            reach.append(farm_reach_of(s))
    maintained = tuple(w[s] == F and not away[s] and not interrupted[s] and not connected[s]
                       and abs(end[s] - cfg.freeze_line) <= cfg.farm_reach for s in (A, B))
    if cfg.freeze_maintenance:
        eff_w = tuple(w[s] if w[s] != F else (F if maintained[s] else SP) for s in (A, B))
    else:
        eff_w = w

    # 5b. M4.5a: the capped wave budget of a connected trader, and each wave's minion target
    eff_budget, wave_capped = budget, (False, False)
    if capped and budget is not None and (connected[A] or connected[B]):
        cap = tuple(_round_half_up(_q(cfg.trade_wave_fraction) * ctx.capacity(lv0[s], it0[s])) for s in (A, B))
        eff_budget = tuple(min(budget[s], cap[s]) if connected[s] else budget[s] for s in (A, B))
        wave_capped = tuple(connected[s] and eff_budget[s] < budget[s] for s in (A, B))
    # 5c. M4.5a correction (champion_movement_rule "tick"): each champion's cell after each tick, on its way to the same
    #     end cell; Hold, an unthreatened Yield and Safe walk with their wave (`_tick_goals`)
    path = None
    if cfg.tick_movement:
        follows = tuple(not (away[s] or recalling[s] or warding[s] or obj[s] != NONE or s in chasers or threatened[s])
                        for s in (A, B))
        path = tuple((-T,) * cfg.ticks_per_turn if absent[s] else
                     (c0[s],) * cfg.ticks_per_turn if recalling[s] else            # channels in its cell
                     tick_path(c0[s], end[s], _tick_goals(probe, lane, s, beh[s], c0[s], cfg) if follows[s] else None,
                               cfg.champion_speed, cfg.ticks_per_turn, T, ward_rule)
                     for s in (A, B))
    targets, retaliate = None, (False, False)
    if cfg.minion_targeting:
        # side s's minions retaliate iff the enemy's connected Trade dealt positive damage (after mitigation) to s
        retaliate = tuple(cfg.minion_retaliation and connected[1 - s] and ctx.mitigate(dealt[1 - s], it0[s]) > 0
                          for s in (A, B))
        targets = tuple(None if away[1 - s] else
                        MinionTarget(mirror_cell(end[1 - s]), cfg.minion_retaliation_delay if retaliate[s] else None,
                                     end[1 - s] == c0[1 - s],
                                     None if path is None else tuple(mirror_cell(x) for x in path[1 - s]))
                        for s in (A, B))

    # 6. the authoritative lane step (effective waves and objectives)
    codes = (eff_w[A] + 6 * (NONE if warding[A] else eff_obj[A]), eff_w[B] + 6 * (NONE if warding[B] else eff_obj[B]))
    res = lane_call(codes[A], codes[B], (reach[A], reach[B]), eff_budget, targets)

    # 7. gold (exact ledger units) and 7b. XP
    avail, farm_u, xp_gain_raw = [0.0, 0.0], [0, 0], [0, 0]
    for s in (A, B):
        o = 1 - s
        for x, hp in res.kills[s].items():
            avail[s] += cfg.farm_per_hp * hp
            farm_u[s] += hp * ctx.collection(x, modes[s], end[s], modes[o], end[o], pays[s])
            if not away[s] and graph_dist(x, end[s]) <= cfg.xp_range:
                xp_gain_raw[s] += cfg.xp_per_hp * hp

    # 8. exposure and gank occurrence
    exposure = tuple(-T if away[s] else max(exposure_cell(c0[s], cfg.ward_exposure), exposure_cell(end[s], cfg.ward_exposure))
                     for s in (A, B))
    gank = tuple(0.0 if away[s] else
                 champion_gank_probability(lane, s, w[s], B_SAFE if interrupted[s] else beh[s], cfg, exposure[s])
                 for s in (A, B))

    # 9b. tower shots and ganks, centi-HP damage at the start-of-turn levels and items
    tower_p = [0.0, 0.0]
    tower_dmg, adjacent = [0, 0], [False, False]
    for s in (A, B):
        if not connected[s]:
            continue
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
    # M4.5a: the minion damage units each champion took, in centi-HP at its start-of-turn level
    minion_cp = tuple(minion_damage_centi(res.champion_hits[s], lv0[s], cfg.minion_damage_alpha,
                                          cfg.minion_damage_growth) for s in (A, B))
    taken = (ctx.mitigate(dealt[B] + tower_dmg[A] + gank_dmg[A] + minion_cp[A], it0[A]),   # invest_durability: / M(S)
             ctx.mitigate(dealt[A] + tower_dmg[B] + gank_dmg[B] + minion_cp[B], it0[B]))

    # 10. death (before level-up)
    died = (not away[A] and me0[A].hp - taken[A] <= 0, not away[B] and me0[B].hp - taken[B] <= 0)
    bounty_u = (ctx.bounty_units if died[B] else 0, ctx.bounty_units if died[A] else 0)

    # 10b. M4.5a: a ward is placed unless the enemy's connected Trade damaged the warder (after mitigation) or it died;
    #      it sets the vision timer for the next vision_turns turns (the lane step counted the old timer down)
    ward_hit = tuple(warding[s] and ctx.mitigate(dealt[1 - s], it0[s]) > 0 for s in (A, B))
    placed = tuple(warding[s] and not ward_hit[s] and not died[s] for s in (A, B))
    if placed[A] or placed[B]:
        st = res.state
        res = replace(res, state=st._replace(va=cfg.vision_turns if placed[A] else st.va,
                                             vb=cfg.vision_turns if placed[B] else st.vb),
                      vision_set=(res.vision_set[A] or placed[A], res.vision_set[B] or placed[B]))
    if ward_rule:
        eff_obj = tuple(VISION if placed[s] else (NONE if warding[s] else eff_obj[s]) for s in (A, B))

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
            left = (cfg.recall_turns - 1 if recalled[s] else me.absent - 1) if away[s] else ctx.death_turns(lv0[s])
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

    # 15. reward (exact ledger units); M4.3c opt-ins: the credited-XP channel, and the terminal turn of "final_plate"
    xp_reward_units = ctx.xp_units * (xp_gain[A] - xp_gain[B])
    reward_units = earned[A] - earned[B] + xp_reward_units
    terminal = cfg.terminal_rule == "final_plate" and max(tau[A], tau[B]) >= ctx.theta_max
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
        reward_units, reward_units / ctx.scale, (invested[A], invested[B]), (pinned[A], pinned[B]), eff_budget,
        xp_reward_units, terminal, warding, placed,
        (warding[A] and not placed[A], warding[B] and not placed[B]), res.champion_hits, minion_cp, retaliate,
        res.champion_fire, wave_capped, targets, path)
    return turn


def _last_front_cell(lane: State, probe, s: int, upto: Optional[int] = None) -> Optional[int]:
    """farm_spot_rule "last_front" (M4.5b): if side s's pre-turn front stops leading during the probe (by tick `upto`,
    default the end of the turn) because it crashed or died, the cell it held last while it still led: its pre-turn
    cell if it was lost on tick 0, else its cell after the previous tick. None while it still leads. (Fronts never walk
    back, so a leading packet behind the previous front cell is a rear packet: the front died.)"""
    px, ph = lane.side(s)[:2]
    if ph <= 0:
        return None
    last = px
    ticks = len(probe.fronts_by_tick) if upto is None else upto + 1
    for fr, cr in zip(probe.fronts_by_tick[:ticks], probe.crashed_by_tick[:ticks]):
        fx = fr[s]
        if cr[s] > 0 or fx is None or fx < last:
            return last
        last = fx
    return None


def _tick_goals(probe, lane: State, s: int, behavior: int, c0: int, cfg: GameConfig) -> Tuple[int, ...]:
    """champion_movement_rule "tick": where a farming champion of side s heads on each tick. Its farm spot is computed
    after every tick of the probe exactly as step 1 computes it at the end of the turn (the front's cell clipped to the
    lane, the own cell without a front (0 from W), the pre-turn front cell once the front has crashed under
    farm_spot_rule "pre_crash_front"; under "last_front" the front's last cell once it has crashed or died,
    `_last_front_cell`), then mapped through the behaviour's target: Safe min(spot, safe_line), Yield
    `yield_target`, Hold (and a Trade with nobody to chase) the spot itself."""
    T = cfg.tower
    out = []
    for t, (fronts, crashed) in enumerate(zip(probe.fronts_by_tick, probe.crashed_by_tick)):
        fx = fronts[s]
        sp = max(-T, min(T, fx)) if fx is not None else (0 if c0 == WARD else c0)
        if cfg.farm_spot_rule == "pre_crash_front" and crashed[s] > 0 and lane.side(s)[1] > 0:
            sp = max(-T, min(T, lane.side(s)[0]))
        elif cfg.farm_spot_rule == "last_front":
            last = _last_front_cell(lane, probe, s, t)
            if last is not None:
                sp = max(-T, min(T, last))
        if behavior == B_SAFE:
            sp = min(sp, cfg.safe_line)
        elif behavior == B_YIELD:
            sp = yield_target(sp, lane, s, cfg)
        out.append(sp)
    return tuple(out)
