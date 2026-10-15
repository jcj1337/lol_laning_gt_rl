"""Start states of M4.5b (all INITIALIZED: constructed diagnostic states, labelled as such; natural states are
collected from scripted play with `engine.collect`). Every state is in A's own coordinates with A the focal side
(the stronger / trading / freezing / warding champion unless stated); B's records are in B's own coordinates.

Power. `invest_pair(cfg, level, gap)` gives the invested gold (A, B) of a gap at a level: both at the level's typical
investment S(L) (TYPICAL_INVEST) for "equal"; for a gap with multiplier m (GAP_MULT: "adv" x1.5, "large" x1.76) the
trailing side's investment is lowered so that M(S_A) / M(S_B) = m, or the leading side's raised when that is not
possible (S >= 0); then the trailing side has 0 and the leading side M = m. Investment multiplies both attack power and durability, so a multiplier m is a duel ratio m^2 in
trade damage per bar.

Builders
    trade_bench   Priority B: A trades B, B with its front of `b_wave` HP at B's own lane cell (0) and standing in it;
                  cover "cover" (A's 8-HP front at A's -1 in contact, A standing with it), "nocover" (no A front in the
                  lane: B's front walks into A at -1), "ahead" (A stands in B's front's cell, A's front behind it at -1)
    freeze_state  Priority A: A holds B's front of `held` HP at A's -1 with its own front of `own` HP in its tower
                  cell (A at -2, B walked up to its +1 unless `b_pos`); B's next wave at its -2 and optionally a
                  crashed B mass under A's tower; levels, power gap, HP fractions, B's remaining plates on A's tower
    vision_state  Priority C: the named lane situations of VISION_SITUATIONS
    zoning_state  Priority D: a lane situation with a power gap (neutral lane, A's push, A's freeze)

HOLDOUT_SPLIT fixes, before any calibration run, the seeds and state parameters reserved for `validate`; it is
written to results/m45b/holdout_split.json by `__main__ split` and never used by a calibration stage.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Tuple

from ...champions_m4 import M4State, context, initial_m4
from ...config import GameConfig
from ...state import State
from .common import DEV, HOLDOUT

TYPICAL_INVEST = {1: 0.0, 2: 0.0, 3: 0.0, 4: 2.0, 6: 8.0, 8: 16.0, 10: 24.0, 11: 32.0, 13: 48.0, 15: 64.0,
                  18: 128.0}
GAP_MULT = {"equal": 1.0, "adv": 1.5, "large": 1.76}
K_BENCH = 2                   # decision index of the benches: no wave spawns during the turn (spawns after k = 1, 3, ...)


def typical_invest(level: int) -> float:
    keys = sorted(TYPICAL_INVEST)
    best = max(k for k in keys if k <= level)
    return TYPICAL_INVEST[best]


def multiplier(cfg: GameConfig, invested: float) -> float:
    ctx = context(cfg)
    return ctx.inv.factor(int(round(invested * cfg.gold_scale))) if ctx.inv is not None else 1.0


def invest_for(cfg: GameConfig, mult: float) -> Optional[float]:
    """The investment S (resources, 0.01 resolution) with M(S) = mult, None when mult < 1 or unreachable."""
    if mult < 1.0 - 1e-12:
        return None
    lo, hi = 0.0, 100_000.0
    if multiplier(cfg, hi) < mult:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if multiplier(cfg, mid) < mult:
            lo = mid
        else:
            hi = mid
    return round(hi, 2)


def invest_pair(cfg: GameConfig, level: int, gap: str = "equal") -> Tuple[float, float]:
    """(S_A, S_B) of a power gap at a level (module docstring). "weak" = B leads by x1.5."""
    S = typical_invest(level)
    if gap == "equal":
        return S, S
    flip = gap == "weak"
    m = GAP_MULT["adv" if flip else gap]
    lead, trail = S, invest_for(cfg, multiplier(cfg, S) / m)
    if trail is None:                                   # cannot lower the trailing side: trailing 0, leading M = m
        trail, lead = 0.0, invest_for(cfg, m)
    return (trail, lead) if flip else (lead, trail)


def champs(cfg: GameConfig, level: Tuple[int, int], invested: Tuple[float, float], pos: Tuple[int, int],
           hp: Tuple[float, float] = (1.0, 1.0), gold: Tuple[float, float] = (0.0, 0.0),
           tower_dmg: Tuple[int, int] = (0, 0)) -> M4State:
    """Both champions present at the first XP of their level, `hp` fractions of the level's max HP."""
    t = context(cfg).t
    m = initial_m4(cfg)
    sides = []
    for s, side in enumerate((m.a, m.b)):
        L = level[s]
        sides.append(side._replace(pos=pos[s], xp=t.xp_cum[L], hp=max(1, int(round(hp[s] * t.hp_max[L]))),
                                   items=int(round(invested[s] * cfg.gold_scale)),
                                   gold=int(round(gold[s] * cfg.gold_scale)), tower_dmg=int(tower_dmg[s])))
    return M4State(*sides)


# ---------------------------------------------------------------------------------------------- Priority B
COVERS = ("cover", "nocover", "ahead")


def trade_bench(cfg: GameConfig, level: int, b_wave: int, cover: str = "cover", gap: str = "equal",
                hp: Tuple[float, float] = (1.0, 1.0), a_wave: int = 8):
    """(lane, champions, k) of the trade bench (module docstring)."""
    if cover == "cover":
        lane = State(-1, a_wave, (), 0, 0, b_wave, (), 0, 0, 0)
        pos = (-1, 0)
    elif cover == "nocover":
        lane = State(-4, 0, (), 0, 0, b_wave, (), 0, 0, 0)
        pos = (-1, 0)
    elif cover == "ahead":
        lane = State(-1, a_wave, (), 0, 0, b_wave, (), 0, 0, 0)
        pos = (0, 0)
    else:
        raise ValueError(cover)
    inv = invest_pair(cfg, level, gap)
    return lane, champs(cfg, (level, level), inv, pos, hp), K_BENCH


# ---------------------------------------------------------------------------------------------- Priority A
def freeze_state(cfg: GameConfig, level: int = 10, gap: str = "large", held: int = 12, own: int = 8,
                 hp: Tuple[float, float] = (1.0, 1.0), b_pos: int = 1, b_mass: int = 0, plates_left: bool = True,
                 gold: Tuple[float, float] = (0.0, 0.0), k: int = 51):
    """(lane, champions, k): A holds B's front of `held` HP at A's -1 (B's own +1) with its own `own`-HP front in its
    tower cell; B's next wave at its own -2; `b_mass` HP of B crashed under A's tower; A at its tower cell, B at its own
    `b_pos`. plates_left False: B has already taken every plate of A's tower (its tower damage at the last threshold),
    so a B crash pays no plate gold. k 51 = the phase of the M4.3c freeze start."""
    lane = State(-2, own, (), 0, 1, held, (-2,), b_mass, 0, 0)
    inv = invest_pair(cfg, level, gap)
    tdb = 0 if plates_left else cfg.plate_thresholds[-1]
    return lane, champs(cfg, (level, level), inv, (-2, b_pos), hp, gold, (0, tdb)), k


# ---------------------------------------------------------------------------------------------- Priority C
VISION_SITUATIONS = {
    # name: (lane, A pos, B pos, description)
    "neutral": (State(0, 8, (-2,), 0, 0, 8, (-2,), 0, 0, 0), 0, 0,
                "both 8-HP fronts and both champions at the lane cell (k 9 of Hold / Hold)"),
    "forward": (State(1, 16, (-2,), 0, -1, 4, (), 0, 0, 0), 1, -1,
                "A's 16-HP front at its +1 against B's 4-HP front at B's -1: A pushing toward B's tower"),
    "presiege": (State(1, 20, (-1,), 0, -2, 3, (), 0, 0, 0), 1, -2,
                 "A's 20-HP stack at its +1 about to crash into B's tower; B in its tower cell"),
    "siege": (State(0, 8, (-4,), 14, -4, 8, (), 0, 0, 0), 2, -2,
              "A's 14-HP mass crashed under B's tower, A in B's tower cell"),
    "defending": (State(-1, 4, (), 0, 1, 16, (-2,), 0, 0, 0), -1, 1,
                  "B's 16-HP front at A's -1 against A's 4-HP front: A defending near its tower"),
    "frozen_on": (State(1, 5, (-2,), 0, -2, 8, (), 0, 0, 0), 1, -2,
                  "B freezes on A: A's 5-HP front held at B's -1 (A's +1), A walked up to its +1"),
    # holdout only (amendment of 2026-10-14, before any calibration run)
    "push_lane": (State(0, 16, (-2,), 0, 0, 6, (-2,), 0, 0, 0), 0, 0,
                  "A's 16-HP front against B's 6-HP front at the lane cell, both champions there"),
    "deep_farm": (State(1, 8, (-2,), 8, -4, 8, (), 0, 0, 0), 1, -2,
                  "A's 8-HP mass crashed under B's tower, A's next front at its +1, A at +1, B in its tower cell"),
}


def vision_state(cfg: GameConfig, situation: str, level: int = 6, gap: str = "equal",
                 hp: Tuple[float, float] = (1.0, 1.0), k: int = 2):
    lane, pa, pb, _ = VISION_SITUATIONS[situation]
    inv = invest_pair(cfg, level, gap)
    return lane, champs(cfg, (level, level), inv, (pa, pb), hp), k


# ---------------------------------------------------------------------------------------------- Priority D
ZONING_SITUATIONS = {
    "neutral": (State(0, 8, (-2,), 0, 0, 8, (-2,), 0, 0, 0), 0, 0, "both fronts and champions at the lane cell"),
    "a_push": (State(1, 12, (-2,), 0, -1, 6, (-2,), 0, 0, 0), 1, -1,
               "A's 12-HP front at its +1 against B's 6-HP front: the fight near B's tower"),
    "b_push": (State(-1, 6, (-2,), 0, 1, 12, (-2,), 0, 0, 0), -1, 1,
               "B's 12-HP front at A's -1 against A's 6-HP front: the fight near A's tower"),
}


def zoning_state(cfg: GameConfig, situation: str, level: int = 6, gap: str = "large",
                 hp: Tuple[float, float] = (1.0, 1.0), k: int = 2):
    lane, pa, pb, _ = ZONING_SITUATIONS[situation]
    inv = invest_pair(cfg, level, gap)
    return lane, champs(cfg, (level, level), inv, (pa, pb), hp), k


# ---------------------------------------------------------------------------------------------- holdout split
# Fixed on 2026-10-14 before any calibration run (results/m45b/holdout_split.json). Calibration stages use DEV seeds
# and the DEV_GRID parameters only; `validate` uses HOLDOUT seeds and the HOLDOUT_GRID parameters.
DEV_GRID = {
    "levels": (2, 4, 10, 18), "gaps": ("equal", "large"), "held": (5, 12, 20), "hp": (1.0, 0.6),
    "trade_levels": (2, 6, 11, 18), "trade_waves": (8, 16, 24),
    "vision_situations": ("neutral", "forward", "presiege", "defending", "frozen_on"),
    "zoning": ("neutral", "a_push"),
}
HOLDOUT_GRID = {
    "levels": (3, 7, 13, 16), "gaps": ("adv",), "held": (8, 16), "hp": (0.8, 0.45),
    "trade_levels": (3, 8, 14), "trade_waves": (12, 20),
    "vision_situations": ("push_lane", "deep_farm"), "zoning": ("b_push",),
    "natural_starts": ("s0",), "natural_turns": (60, 150),
}
HOLDOUT_SPLIT = {"dev_seeds": [DEV[0], DEV[-1]], "holdout_seeds": [HOLDOUT[0], HOLDOUT[-1]], "dev_grid": DEV_GRID,
                 "holdout_grid": HOLDOUT_GRID, "fixed": "2026-10-14, before any M4.5b calibration run",
                 "amendments": ["2026-10-14, before any calibration run: the Vision situation 'frozen_on' was used by "
                                "the DEV Vision stage (baseline, DEV seeds) and is reclassified as DEV; the Vision "
                                "holdout situations are the new 'push_lane' and 'deep_farm' (defined at this "
                                "amendment); 'siege' is dropped (Ward is not legal from the enemy tower cell)"]}


@lru_cache(maxsize=4)
def s0(cfg: GameConfig):
    from ...mechanics import initial_state
    return initial_state(cfg), initial_m4(cfg), 0


__all__ = ["TYPICAL_INVEST", "GAP_MULT", "invest_for", "invest_pair", "multiplier", "champs", "trade_bench",
           "freeze_state", "vision_state", "zoning_state", "VISION_SITUATIONS", "ZONING_SITUATIONS", "COVERS",
           "DEV_GRID", "HOLDOUT_GRID", "HOLDOUT_SPLIT", "s0", "typical_invest"]
