"""M4 progression tables (docs/m4_integrated_design.md §8.2, §8.5, §8.6, §9, §11; plan WP2).

Pure integer functions. Every table is computed once per parameter set in double precision, rounded half up and
stored as integers; every later operation is integer arithmetic. With the defaults of `GameConfig.m4()`:

    H(L)         = floor(600 * 1.05^(L - 1) + 1/2)                    max HP, centi-HP          (600 at L = 1)
    u(L)         = floor(H(L) / 6 + 1/2)                              one "HP unit", centi-HP   (100 at L = 1)
    AP(L, N)     = floor(10^4 * 1.05^(L - 1) * 1.20^N + 1/2)           attack power, basis points
    dmg(D, L, N) = floor((D * 100 * AP(L, N) + 5000) / 10^4)          a hit of D base HP, centi-HP
    P(L, N)      = floor((400 * AP(L, N) + 5000) / 10^4)              Apply Pressure tower damage, centi-units
    n(L)         = 8 + 3 (L - 1)                                      XP from level L to L + 1
    X(L)         = 8 (L - 1) + 3 (L - 1)(L - 2) / 2                   cumulative XP of level L  (X(18) = 544)
    level_of(x)  = max{L : X(L) <= x}

Gold is an integer ledger in ledger units (micro-resources): the prices are `price_mu = (4 000 000, 6 000 000)`,
the spendable balance is clamped at G_rem(N) = sum of the prices of the tiers not yet bought, and a purchase is
an exact integer comparison.

`tables(cfg)` returns the (cached) `Tables` of a configuration; the module-level functions take a `Tables` and
default to the tables of `GameConfig.m4()`.

M4.2c continuous investment (`GameConfig.m4_continuous()`, an experiment): no tiers; a champion's cumulative invested
gold S (ledger units u, S = u / gold_scale resources) scales its attack power through a fixed-point multiplier

    M(u)         = 10^12 + round(10^12 * alpha * ((1 + u / gold_scale)^p - 1))         (10^12 = x1.0)
    AP*(L, u)    = AP(L, 0) * M(u)                                    attack power, basis points x 10^12 (exact int)
    dmg*(D, L, u) = floor((D * 100 * AP*(L, u) + 5000 * 10^12) / 10^16)   centi-HP (the same rounding as dmg)
    P*(L, u)     = floor((400 * AP*(L, u) + 5000 * 10^12) / 10^16)      centi-units

M(0) = 10^12 exactly, so a champion that never invests has the tier-0 numbers of the baseline. M is strictly
increasing per LEDGER unit over the supported range u <= INVEST_MAX_RESOURCES * gold_scale (the per-unit increment
10^6 alpha p (1 + S)^(p - 1) is >= 10^3 there, far above the float64 rounding of the power); beyond it `Investment`
raises instead of returning an unverified value. Hit damage stays quantized at 1 centi-HP, so a small investment
raises AP* even when a given hit's rounded damage does not change.

M4.2d opt-in curve (invest_curve = "saturating", docs/m4_investment_readiness_report.md): with A = invest_gain and
K = invest_scale resources (k = K * gold_scale ledger units),

    M(S)         = 1 + A (1 - sqrt(K / (K + S)))                      bounded: 1 <= M < 1 + A
    M(u)         = ONE + floor(A * (ONE - isqrt(floor(ONE^2 k / (k + u)))))      ONE = INVEST_ONE_SAT = 10^24

computed in exact integers (math.isqrt and a rational A), so it is platform independent and non-decreasing by
construction. The per-ledger-unit decrement of ONE sqrt(k / (k + u)) is ONE sqrt(k) / (2 (k + u)^1.5) >= 5 * 10^8
over the supported range (k >= 10^6 units, u <= 10^12 units), so M is STRICTLY increasing per ledger unit there.
The damage formulas above use the same ONE (dmg* = floor((D 100 AP* + 5000 ONE) / (10^4 ONE))).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Optional, Tuple

from .config import GameConfig


@dataclass(frozen=True)
class Tables:
    level_cap: int
    item_tiers: int
    hp_max: Tuple[int, ...]               # [L], L = 1 .. level_cap (index 0 unused, 0)
    hp_unit: Tuple[int, ...]              # [L]
    ap: Tuple[Tuple[int, ...], ...]       # [L][N]
    dmg: Tuple[Tuple[Tuple[int, ...], ...], ...]    # [D][L][N], D = 0, 1, 2
    pressure: Tuple[Tuple[int, ...], ...]  # [L][N]
    xp_cum: Tuple[int, ...]               # X(L) [L]
    xp_next: Tuple[int, ...]              # n(L) [L]; 0 at the cap
    xp_cap: int                           # X(level_cap)
    level_by_xp: Tuple[int, ...]          # level_of(x) for x = 0 .. xp_cap
    prices: Tuple[int, ...]               # ledger units per tier
    gold_cap: Tuple[int, ...]             # G_rem(N), N = 0 .. item_tiers
    thresholds: Tuple[int, ...]           # plate thresholds (centi-units)
    hp_max_cap: int                       # H(level_cap)
    ap_max: int                           # AP(level_cap, item_tiers)


def _round_half_up(x: float) -> int:
    import math
    return int(math.floor(x + 0.5))


@lru_cache(maxsize=64)
def _build(level_cap: int, item_tiers: int, hp_base: int, hp_growth: float, ap_level_growth: float,
           item_mult: float, champion_hp: int, xp_base: int, xp_step: int, pressure: int,
           prices: Tuple[float, ...], gold_unit: float, thresholds: Tuple[int, ...]) -> Tables:
    Ls = range(1, level_cap + 1)
    hp = [0] + [_round_half_up(hp_base * hp_growth ** (L - 1)) for L in Ls]
    unit = [0] + [(2 * hp[L] + champion_hp) // (2 * champion_hp) for L in Ls]       # floor(H / ch + 1/2)
    ap = [()] + [tuple(_round_half_up(10_000 * ap_level_growth ** (L - 1) * item_mult ** N)
                       for N in range(item_tiers + 1)) for L in Ls]
    dmg = tuple(tuple(() if L == 0 else tuple((D * 100 * a + 5000) // 10_000 for a in ap[L])
                      for L in range(level_cap + 1)) for D in range(3))
    prs = [()] + [tuple((pressure * a + 5000) // 10_000 for a in ap[L]) for L in Ls]
    xp_cum = [0] + [xp_base * (L - 1) + xp_step * (L - 1) * (L - 2) // 2 for L in Ls]
    xp_next = [0] + [(xp_base + xp_step * (L - 1)) if L < level_cap else 0 for L in Ls]
    xp_cap = xp_cum[level_cap]
    lvl = []
    L = 1
    for x in range(xp_cap + 1):
        while L < level_cap and xp_cum[L + 1] <= x:
            L += 1
        lvl.append(L)
    scale = Fraction(1) / Fraction(repr(float(gold_unit)))
    pr = tuple(int(Fraction(repr(float(p))) * scale) for p in prices[:item_tiers])
    cap = tuple(sum(pr[N:]) for N in range(item_tiers + 1))
    return Tables(level_cap, item_tiers, tuple(hp), tuple(unit), tuple(ap), dmg, tuple(prs), tuple(xp_cum),
                  tuple(xp_next), xp_cap, tuple(lvl), pr, cap, tuple(thresholds), hp[level_cap],
                  ap[level_cap][item_tiers])


def tables(cfg: GameConfig) -> Tables:
    """The progression tables of `cfg` (cached by parameter values)."""
    return _build(cfg.level_cap, cfg.item_tiers, cfg.hp_base, cfg.hp_growth, cfg.ap_level_growth, cfg.item_mult,
                  cfg.champion_hp, cfg.xp_base, cfg.xp_step, cfg.pressure_tower_damage, tuple(cfg.item_prices),
                  cfg.gold_unit, tuple(cfg.plate_thresholds))


DEFAULT = tables(GameConfig.m4())


# ---------------------------------------------------------------------- pure functions
def level_of(xp: int, t: Tables = DEFAULT) -> int:
    """Level of a champion with `xp` experience (xp beyond the cap counts as the cap)."""
    if xp < 0:
        raise ValueError("xp must be non-negative")
    return t.level_by_xp[xp if xp < t.xp_cap else t.xp_cap]


def hp_max(level: int, t: Tables = DEFAULT) -> int:
    return t.hp_max[level]


def hp_unit(level: int, t: Tables = DEFAULT) -> int:
    return t.hp_unit[level]


def attack_power(level: int, items: int, t: Tables = DEFAULT) -> int:
    return t.ap[level][items]


def hit_damage(base: int, level: int, items: int, t: Tables = DEFAULT) -> int:
    """Centi-HP dealt by a successful hit of `base` (0, 1 or 2) base HP at the attacker's (level, items)."""
    return t.dmg[base][level][items]


def pressure_damage(level: int, items: int, t: Tables = DEFAULT) -> int:
    """Centi-units an effective Apply Pressure turn adds to the attacker's tower-damage counter."""
    return t.pressure[level][items]


def xp_progress(xp: int, t: Tables = DEFAULT) -> Tuple[int, int, int]:
    """(level, XP into the level, XP needed for the next level; 0 at the cap)."""
    L = level_of(xp, t)
    return L, min(xp, t.xp_cap) - t.xp_cum[L], t.xp_next[L]


def level_up_hp(hp: int, xp_before: int, xp_after: int, t: Tables = DEFAULT) -> int:
    """HP after the step-12 level-up rule (design §9.2): add the increase in max HP, never a heal beyond it."""
    return hp + t.hp_max[level_of(xp_after, t)] - t.hp_max[level_of(xp_before, t)]


def next_price(items: int, t: Tables = DEFAULT) -> Optional[int]:
    """Ledger units of the next tier, or None when every tier is bought."""
    return t.prices[items] if items < t.item_tiers else None


def gold_cap(items: int, t: Tables = DEFAULT) -> int:
    """G_rem(N): the most gold that can still be spent (sum of the remaining tiers' prices)."""
    return t.gold_cap[items]


def purchase(items: int, gold: int, t: Tables = DEFAULT) -> Tuple[int, int, int]:
    """Automatic purchase at base (design §8.5): buy tiers in order while affordable.
    Returns (items', gold', tiers bought)."""
    bought = 0
    while items < t.item_tiers and gold >= t.prices[items]:
        gold -= t.prices[items]
        items += 1
        bought += 1
    return items, gold, bought


def clamp_gold(items: int, gold: int, t: Tables = DEFAULT) -> Tuple[int, int]:
    """(gold clamped at G_rem(items), amount removed). Only gold that can never be spent is removed."""
    cap = t.gold_cap[items]
    return (cap, gold - cap) if gold > cap else (gold, 0)


def plates_broken(tower_dmg: int, t: Tables = DEFAULT) -> int:
    """pi(tau): thresholds reached by the cumulative tower damage."""
    return sum(1 for th in t.thresholds if tower_dmg >= th)


# ---------------------------------------------------------------------- M4.2c continuous investment
INVEST_ONE = 10 ** 12                     # fixed-point unit of the investment multiplier M (x1.0), curve "power"
INVEST_ONE_SAT = 10 ** 24                 # ... curve "saturating" (exact integer evaluation, see the docstring)
INVEST_MAX_RESOURCES = 10 ** 6            # largest cumulative investment whose strict monotonicity is verified


class Investment:
    """Integer attack power of the continuous-investment experiment (see the module docstring). One per
    (curve parameters, gold_scale, AP(L, 0) table); memoizes M(u). `one` is the fixed-point unit of M."""

    def __init__(self, alpha: float, p: float, gold_scale: int, t: Tables, pressure: int, curve: str = "power",
                 gain: float = 1.5, scale_res: float = 16.0):
        self.alpha, self.p, self.scale, self.t, self.pressure_base = float(alpha), float(p), int(gold_scale), t, pressure
        self.curve = curve
        self.max_units = INVEST_MAX_RESOURCES * self.scale
        if curve == "power":
            self.one = INVEST_ONE
        elif curve == "saturating":
            self.one = INVEST_ONE_SAT
            g = Fraction(repr(float(gain)))
            k = Fraction(repr(float(scale_res))) * self.scale
            if g <= 0 or k.denominator != 1 or k < 1:
                raise ValueError("invest_gain must be positive and invest_scale a positive whole number of ledger "
                                 "units")
            self._gn, self._gd, self._k = g.numerator, g.denominator, int(k)
            self.gain, self.scale_res = float(gain), float(scale_res)
        else:
            raise ValueError(f"unknown invest_curve {curve!r}")
        self._m: dict = {0: self.one}

    def multiplier(self, units: int) -> int:
        """M(u): the investment multiplier, x`one` fixed point (10^12 for "power", 10^24 for "saturating")."""
        m = self._m.get(units)
        if m is None:
            if not 0 <= units <= self.max_units:
                raise ValueError(f"invested gold {units} ledger units outside the supported range [0, "
                                 f"{self.max_units}] (INVEST_MAX_RESOURCES = {INVEST_MAX_RESOURCES})")
            if self.curve == "power":
                m = INVEST_ONE + int(round(INVEST_ONE * self.alpha * ((1.0 + units / self.scale) ** self.p - 1.0)))
            else:
                one, k = self.one, self._k
                r = math.isqrt(one * one * k // (k + units))          # floor(ONE sqrt(k / (k + u))), exact
                m = one + self._gn * (one - r) // self._gd
            if len(self._m) > 100_000:
                self._m = {0: self.one}
            self._m[units] = m
        return m

    def factor(self, units: int) -> float:
        """M(u) as a float (reports)."""
        return self.multiplier(units) / self.one

    def attack_power(self, level: int, units: int) -> int:
        """AP*(L, u) in basis points x `one`."""
        return self.t.ap[level][0] * self.multiplier(units)

    def hit_damage(self, base: int, level: int, units: int) -> int:
        one = self.one
        return (base * 100 * self.attack_power(level, units) + 5000 * one) // (10_000 * one)

    def pressure_damage(self, level: int, units: int) -> int:
        one = self.one
        return (self.pressure_base * self.attack_power(level, units) + 5000 * one) // (10_000 * one)


@lru_cache(maxsize=16)
def _investment(alpha: float, p: float, gold_scale: int, t: Tables, pressure: int, curve: str = "power",
                gain: float = 1.5, scale_res: float = 16.0) -> Investment:
    return Investment(alpha, p, gold_scale, t, pressure, curve, gain, scale_res)


def investment(cfg: GameConfig) -> Investment:
    """The `Investment` of a continuous-investment configuration (cached)."""
    if not cfg.continuous_items:
        raise ValueError("investment() applies to item_model 'continuous'")
    if cfg.invest_curve == "power":
        return _investment(cfg.invest_alpha, cfg.invest_p, cfg.gold_scale, tables(cfg), cfg.pressure_tower_damage)
    return _investment(cfg.invest_alpha, cfg.invest_p, cfg.gold_scale, tables(cfg), cfg.pressure_tower_damage,
                       cfg.invest_curve, cfg.invest_gain, cfg.invest_scale)
