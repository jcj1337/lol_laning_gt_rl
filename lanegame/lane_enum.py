"""Exhaustive enumeration of the reachable lane (M4 design §10.5.3, §12.2; plan WP3 test B1, WP7 test O1, S3a).

The fixed point over (lane state, k mod P) from the initial state, where each side may play every legal wave
action, objective NONE or PRESSURE (when sieging) and every champion reach value (no reach, cells -T .. T). This is a
superset of the lane steps any champion-mode resolver (M2 or M4) can request: M4's Freeze maintenance only replaces
a Freeze by Slow Push, and tower damage never feeds back into the lane. Vision timers and plate counters are
projected out (they never affect the lane and are observed exactly).

`lane_bounds(cfg)` returns the constants the evaluation guarantee and the lossless observation rely on: A_max (the
largest alive minion HP of one side, lane + crashed mass), the largest enemy HP dying in one turn for one side, the
largest crashed mass, the largest one-turn minion tower hits, and the largest rear-packet count in one cell. The
reference implementation is results/m4_design/scripts/lane_bounds.py (about 5 minutes).
"""

from __future__ import annotations

import time
from collections import Counter
from functools import lru_cache
from typing import Dict, Iterable, List, Set, Tuple

from .config import NONE, PRESSURE, GameConfig
from .mechanics import initial_state, step_turn, wave_actions
from .state import State

NO_REACH = -10 ** 6


def lane_signature(cfg: GameConfig) -> tuple:
    """The lane parameters the enumerated bounds depend on (a change re-runs the enumeration)."""
    return (cfg.base, cfg.tower, cfg.ticks_per_turn, cfg.spawn_interval, cfg.wave_hp, cfg.max_stacks, cfg.cap_rule,
            cfg.queue_fights, cfg.damage_law, cfg.shove_damage, cfg.tower_damage, cfg.slowpush_line,
            cfg.slowpush_mode, cfg.slowpush_creep_period, cfg.freeze_line, cfg.freeze_break_threshold,
            tuple(cfg.freeze_break_push), cfg.freeze_legal_max, cfg.tower_mode, cfg.tower_siege_damage,
            cfg.siege_engage, cfg.siege_cap, cfg.thin_damage, cfg.thin_reach)


def side_options(state: State, s: int, cfg: GameConfig) -> List[Tuple[int, int, int]]:
    reaches = (NO_REACH,) + tuple(range(-cfg.tower, cfg.tower + 1))
    objs = (NONE, PRESSURE) if state.siege_of(s) > 0 else (NONE,)
    return [(w, o, r) for w in wave_actions(state, s, cfg) for o in objs for r in reaches]


def project(st: State) -> State:
    return st._replace(va=0, vb=0, ap=0, bp=0)


def reachable_lane(cfg: GameConfig) -> Set[Tuple[State, int]]:
    """Every reachable (projected lane state, phase) pair."""
    return set(_reachable(cfg))


@lru_cache(maxsize=4)
def _reachable(cfg: GameConfig) -> frozenset:
    per = cfg.phase_period
    root = (project(initial_state(cfg)), 0)
    seen = {root}
    frontier = [root]
    while frontier:
        nxt = []
        for st, ph in frontier:
            for oa in side_options(st, 0, cfg):
                for ob in side_options(st, 1, cfg):
                    res = step_turn(st, ph, oa[0] + 6 * oa[1], ob[0] + 6 * ob[1], cfg, champion_reach=(oa[2], ob[2]))
                    s2 = (project(res.state), (ph + 1) % per)
                    if s2 not in seen:
                        seen.add(s2)
                        nxt.append(s2)
        frontier = nxt
    return frozenset(seen)


def lane_bounds(cfg: GameConfig) -> Dict[str, int]:
    t0 = time.time()
    seen = _reachable(cfg)
    a_max = k_max = siege_max = hits_max = rears_max = 0
    for st, ph in seen:
        for s in (0, 1):
            fx, fh, rears, sg = st.side(s)[:4]
            a_max = max(a_max, fh + cfg.wave_hp * len(rears) + sg)
            siege_max = max(siege_max, sg)
            if rears:
                rears_max = max(rears_max, max(Counter(rears).values()))
        for oa in side_options(st, 0, cfg):
            for ob in side_options(st, 1, cfg):
                res = step_turn(st, ph, oa[0] + 6 * oa[1], ob[0] + 6 * ob[1], cfg, champion_reach=(oa[2], ob[2]))
                for s in (0, 1):
                    k_max = max(k_max, sum(res.kills[s].values()))
                    hits_max = max(hits_max, res.tower_hits[s])
    return {"pairs": len(seen), "A_max": a_max, "K_max": k_max, "siege_max": siege_max, "tower_hits_max": hits_max,
            "rears_per_cell_max": rears_max, "seconds": round(time.time() - t0, 1)}


# The enumerated bounds of the default lane (results/m4_design/lane_bounds.txt, obs_alias.txt), re-checked by the
# slow test B1. A configuration with a different lane signature has no recorded bounds.
DEFAULT_LANE_BOUNDS = {"pairs": 20572, "A_max": 32, "K_max": 15, "siege_max": 23, "tower_hits_max": 11,
                       "rears_per_cell_max": 4}


def recorded_bounds(cfg: GameConfig) -> Dict[str, int]:
    if lane_signature(cfg) != lane_signature(GameConfig()):
        raise ValueError("no enumerated lane bounds are recorded for these lane parameters: run "
                         "lanegame.lane_enum.lane_bounds(cfg) (design §12.2 condition 1)")
    return dict(DEFAULT_LANE_BOUNDS)
