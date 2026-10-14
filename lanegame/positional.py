"""M4.5a positional geometry (EXPERIMENTAL, `GameConfig.m45_unified()`; docs/m4/m4_5a_unified_mechanics_spec.md).

The champion positions form a small movement graph: the lane cells -T .. T (own coordinates) in a path, plus one shared
off-lane Vision node W attached to the lane cell 0 (`vision_rule = "ward"`). W is the same physical node for both
sides, so it is mirror-invariant: it is stored as the sentinel `WARD` in either side's own coordinates, and mirroring a
cell into the other side's coordinates maps a lane cell x to -x and W to W.

    d(x, y) = |x - y| (lane cells),   d(x, W) = 1 + |x|,   d(W, W) = 0

Every helper reduces to the historical lane arithmetic when no position is W, so the configurations without a Vision
node are untouched. Kill cells of minions may lie outside -T .. T (defenders behind a tower); the same distance
applies. This module imports nothing from the engine, so the engine modules can all import it.

M4.5a correction (champion_movement_rule "tick"): `tick_path` walks a champion to its end cell tick by tick, following
a goal per tick (a farming champion's wave front, from the probe); `MinionTarget.path` carries those cells to the lane
step, which evaluates positional aggro on each tick's positions after its movement.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import NamedTuple, Optional, Tuple

WARD = 99                       # the shared Vision node W, in every side's own coordinates (never a lane cell)


def is_ward(x: int) -> bool:
    return x == WARD


def mirror_cell(y: int) -> int:
    """A cell of the other side (its own coordinates) in my own coordinates: -y for a lane cell, W for W."""
    return y if y == WARD else -y


def dist(x: int, y: int) -> int:
    """Graph distance between two nodes given in the SAME side's coordinates."""
    if x == WARD:
        return 0 if y == WARD else 1 + abs(y)
    if y == WARD:
        return 1 + abs(x)
    return abs(x - y)


def move(c: int, target: int, speed: int, tower: int) -> int:
    """Where a champion at c ends after moving at most `speed` edges toward `target` along the shortest path. Lane
    moves are clipped to -tower .. tower (as champions._move); W is entered and left through the lane cell 0."""
    if c == target or speed <= 0:
        return c
    if c == WARD:                                       # the first edge leaves W for the lane cell 0
        c, speed = 0, speed - 1
        if speed == 0 or c == target:
            return c
    if target == WARD:                                  # walk to 0, then enter W if an edge is left
        if abs(c) + 1 <= speed:
            return WARD
        return c + max(-speed, min(speed, -c))
    step = max(-speed, min(speed, target - c))
    return max(-tower, min(tower, c + step))


def in_contact(x: int, y: int, tower: int, trade_range: int) -> bool:
    """Champions at own cells x and y (each in its own coordinates) can trade: graph distance <= trade_range, except
    that a champion in its own tower cell is reachable only from inside that cell (the historical rule)."""
    if x == -tower or y == -tower:
        return x != WARD and y != WARD and x + y == 0
    return dist(x, mirror_cell(y)) <= trade_range


def exposure_cell(x: int, ward_exposure: int) -> int:
    """The lane cell whose gank exposure a position carries: W counts as `ward_exposure` (the lane midpoint, 0)."""
    return ward_exposure if x == WARD else x


def reach_interval(end: int, mode_safe: bool, farm_reach: int, reach_cap: int, safe_line: int) -> Tuple[int, int]:
    """wave_reach_rule "ball": the cells (own coordinates) whose enemy minions a champion ending at `end` damages,
    {x : d(x, end) <= farm_reach, x <= reach_cap}, and for Safe / retreat also x <= min(end, safe_line); always an
    interval (lo, hi) of lane cells. From W: |x| <= farm_reach - 1 (only the lane cell 0 at farm_reach 1)."""
    if end == WARD:
        lo, hi = -(farm_reach - 1), min(reach_cap, farm_reach - 1)
    else:
        lo, hi = end - farm_reach, min(reach_cap, end + farm_reach)
    if mode_safe:
        hi = min(hi, safe_line, end if end != WARD else safe_line)
    return lo, hi


def reaches(spec, x: int) -> bool:
    """Does a champion's reach `spec` cover enemy minions at own cell x? An int is the historical reach LINE (every
    cell at or behind it); a pair (lo, hi) is the M4.5a reach ball as an interval."""
    if isinstance(spec, tuple):
        return spec[0] <= x <= spec[1]
    return x <= spec


def tick_speeds(speed: int, ticks: int) -> Tuple[int, ...]:
    """Edges a champion may walk on each tick of a turn: ceil((t+1) v / n) - ceil(t v / n), earlier ticks first (the
    split of the half-wave firepower and of the wave budgets); 1 + 1 at v = 2, n = 2."""
    return tuple(-((-(t + 1) * speed) // ticks) + ((-t * speed) // ticks) for t in range(ticks))


@lru_cache(maxsize=8192)                # a few thousand distinct arguments at most (cells, goals, speeds)
def tick_path(c0: int, end: int, goals: Optional[Tuple[int, ...]], speed: int, ticks: int, tower: int,
              ward: bool) -> Tuple[int, ...]:
    """champion_movement_rule "tick": the champion's cell after each tick of a turn, from its start cell c0 to its end
    cell (d(c0, end) <= speed). On tick t it moves to the node p that is

        within tick_speeds[t] edges of its previous cell and within the edges still left of the end cell,

    and among those nearest to goals[t], then nearest to the end cell, then nearest to its previous cell (no needless
    step), then the smaller lane cell (W last). goals None = the end cell on every tick (a direct walk, its edges spent
    on the earliest ticks). The last cell is always the end cell. Own coordinates throughout, so the same rule for both
    sides (mirror-equivariant)."""
    steps = tick_speeds(speed, ticks)
    nodes = tuple(range(-tower, tower + 1)) + ((WARD,) if ward else ())
    left = sum(steps)
    p, out = c0, []
    for t, v in enumerate(steps):
        left -= v
        g = end if goals is None else goals[t]
        prev = p
        p = min((x for x in nodes if dist(prev, x) <= v and dist(x, end) <= left),
                key=lambda x: (dist(x, g), dist(x, end), dist(x, prev), x == WARD, x))
        out.append(p)
    return tuple(out)


class MinionTarget(NamedTuple):
    """What one side's wave may target this turn (M4.5a, given to the lane step): the enemy champion's end cell in THIS
    side's own coordinates (W stays W), the first tick on which this side's minions retaliate against it (None: the
    enemy champion did not damage ours this turn), whether that champion stayed in its cell this turn (start cell =
    end cell), and (champion_movement_rule "tick") its cell after each tick, in this side's coordinates.

    champion_movement_rule "turn" (m45_unified_r1): the champion stands at `cell` for both ticks and positional aggro
    is designated at the start of a tick, only on a `stationary` champion. "tick" (m45_unified): the lane step reads
    `path[t]` after tick t's movement for both triggers; `stationary` is recorded but plays no role."""
    cell: int
    retaliate_from: Optional[int] = None
    stationary: bool = True
    path: Optional[Tuple[int, ...]] = None


def _q(x: float) -> Fraction:
    return Fraction(repr(float(x)))


@lru_cache(maxsize=4096)
def minion_damage_centi(units: int, level: int, alpha: int, growth: float) -> int:
    """Centi-HP a champion at `level` takes from `units` minion damage units (before the durability division):
    max(1, round_half_up(alpha * units / growth^(level - 1))) for units > 0, else 0. Exact rationals."""
    if units <= 0:
        return 0
    v = Fraction(alpha * units) / (_q(growth) ** (level - 1))
    return max(1, (2 * v.numerator + v.denominator) // (2 * v.denominator))
