"""Exact one-turn expectations by enumeration of the turn's chance branches (M4.5b, Priorities A, B and D).

A turn's randomness is the chance vector u (ten uniforms; `champions.U_*`). Given both actions and the start state,
each event happens iff its uniform is below a probability the resolver computes from the start-of-turn state and the
champions' end cells: per side a trade hit (h), a double hit given a hit (q), a tower shot (p_t), a gank (p_g) and the
gank's damage (Binomial(gank_damage_trials, p_e) by inverse CDF). `turn_distribution` first resolves the turn without
chance (u = 1: no event) to read these probabilities, then resolves every combination of outcomes with uniforms that
force it and weights it by its exact probability. The lane step itself is deterministic.

    ganks = "off"     gank uniforms fixed at 1 (no gank): the trade-and-minion expectation of the turn
    ganks = "on"      gank occurrence and damage enumerated too (5 damage outcomes per gank)

`expect` returns the expectation of every numeric quantity of `outcome` (per side) and the branch list.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple

from ...champions import N_CHANCE, U_GANK, U_GANK_DAMAGE, U_TOWER, U_TRADE_DOUBLE, U_TRADE_HIT, gank_damage_distribution
from ...champions_m4 import context, resolve_turn_m4
from .engine import lane_cache

A, B = 0, 1
NO = 1.0                      # a uniform that never triggers (the resolver's "no chance" convention)
YES = 0.0                     # a uniform that always triggers when the probability is positive


def outcome(t, cfg) -> Dict[str, Tuple[float, float]]:
    """Per-side numbers of one resolved turn (centi-HP and bars are of the victim's start-of-turn max HP)."""
    tb = context(cfg).t
    lv = t.levels_before
    sc = cfg.gold_scale
    units = [[0, 0], [0, 0]]
    for tick in t.lane.trace or ():
        mf = tick.get("minion_fire")
        if mf:
            for s in (A, B):
                c, u = mf[s]
                if u > 0 and c:
                    units[s][0 if c == "retaliation" else 1] += u
    out = {}

    def put(k, f):
        out[k] = (float(f(A)), float(f(B)))
    put("dealt", lambda s: t.trade_damage[s])                                 # cHP dealt by s, before mitigation
    put("dealt_bar", lambda s: t.trade_damage[s] / tb.hp_max[lv[1 - s]])
    put("minion_cp", lambda s: t.minion_damage[s])                            # cHP of minion fire taken by s
    put("minion_bar", lambda s: t.minion_damage[s] / tb.hp_max[lv[s]])
    put("u_ret", lambda s: units[s][0])
    put("u_pos", lambda s: units[s][1])
    put("tower_cp", lambda s: t.tower_damage[s])
    put("gank_cp", lambda s: t.gank_damage[s])
    put("taken", lambda s: t.damage_taken[s])                                 # cHP lost (after durability)
    put("taken_bar", lambda s: t.damage_taken[s] / tb.hp_max[lv[s]])
    put("died", lambda s: t.died[s])
    put("connected", lambda s: t.connected[s])
    put("landed", lambda s: t.connected[s] and t.trade_damage[s] > 0)
    put("farm", lambda s: t.farm_units[s] / sc)
    put("earned", lambda s: t.earned_units[s] / sc)
    put("bounty", lambda s: t.bounty_units[s] / sc)
    put("xp", lambda s: t.xp_gain[s])
    put("ward_placed", lambda s: t.ward_placed[s])
    put("front_hp", lambda s: t.lane.state.side(s)[1])
    put("front_x", lambda s: t.lane.state.side(s)[0] if t.lane.state.side(s)[1] > 0 else 0)
    put("crash_hp", lambda s: t.lane.crash_hp[s])
    put("end", lambda s: t.end[s])
    return out


def _branches(p: float):
    """(uniform, probability) of an event with probability p: [(YES, p), (NO, 1 - p)] (degenerate ones dropped)."""
    out = []
    if p > 0:
        out.append((YES, p))
    if p < 1:
        out.append((NO, 1.0 - p))
    return out


def turn_distribution(cfg, lane, m4, k: int, a: int, b: int, ganks: str = "off") -> List[Tuple[float, object]]:
    """[(probability, M4Turn)] over every chance branch of the turn (module docstring)."""
    ls = lane_cache(cfg)
    ctx = context(cfg)
    base = resolve_turn_m4(lane, m4, k, a, b, cfg, lane_step=ls, u=(NO,) * N_CHANCE, ctx=ctx)
    axes = []                                        # per axis: list of (index -> uniform, probability)
    for s in (A, B):
        h, q = base.hit_p[s], base.double_p[s]
        if base.connected[s] and h > 0:
            hit = [((U_TRADE_HIT[s], YES), (U_TRADE_DOUBLE[s], YES), h * q),
                   ((U_TRADE_HIT[s], YES), (U_TRADE_DOUBLE[s], NO), h * (1 - q)),
                   ((U_TRADE_HIT[s], NO), (U_TRADE_DOUBLE[s], NO), 1 - h)]
            axes.append([(dict([x, y]), p) for x, y, p in hit if p > 0])
        if base.tower_p[s] > 0:
            axes.append([({U_TOWER[s]: u}, p) for u, p in _branches(base.tower_p[s])])
        if ganks == "on" and base.gank_p[s] > 0:
            pe = base.gank_damage_p[s]
            dist = gank_damage_distribution(pe, cfg)
            # inverse CDF of champions._gank_damage: damage d iff u in its interval; pick the interval's midpoint
            opts = [({U_GANK[s]: NO}, 1.0 - base.gank_p[s])]
            c = 0.0
            for d in range(len(dist)):                   # interval of d: [P(D < d), P(D <= d))
                lo, c = c, c + dist[d]
                if dist[d] > 0:
                    opts.append(({U_GANK[s]: YES, U_GANK_DAMAGE[s]: (lo + c) / 2}, base.gank_p[s] * dist[d]))
            axes.append(opts)
    out = []
    for combo in product(*axes) if axes else [()]:
        u = [NO] * N_CHANCE
        p = 1.0
        for setting, pr in combo:
            for i, x in setting.items():
                u[i] = x
            p *= pr
        if p <= 0:
            continue
        out.append((p, resolve_turn_m4(lane, m4, k, a, b, cfg, lane_step=ls, u=tuple(u), ctx=ctx)))
    tot = sum(p for p, _ in out)
    assert abs(tot - 1.0) < 1e-9, tot
    return out


def expect(cfg, lane, m4, k: int, a: int, b: int, ganks: str = "off") -> Dict[str, object]:
    """Expectations of `outcome` over the turn's chance branches, plus conditional expectations given that each
    side's trade landed / connected but missed."""
    dist = turn_distribution(cfg, lane, m4, k, a, b, ganks)
    keys = None
    mean: Dict[str, List[float]] = {}
    cond: Dict[str, Dict[str, List[float]]] = {}
    mass: Dict[str, float] = {}
    for p, t in dist:
        o = outcome(t, cfg)
        if keys is None:
            keys = list(o)
            mean = {k_: [0.0, 0.0] for k_ in keys}
        for k_ in keys:
            for s in (A, B):
                mean[k_][s] += p * o[k_][s]
        for s, tag in ((A, "A"), (B, "B")):
            if not t.connected[s]:
                continue
            g = f"{tag}_landed" if t.trade_damage[s] > 0 else f"{tag}_missed"
            mass[g] = mass.get(g, 0.0) + p
            c = cond.setdefault(g, {k_: [0.0, 0.0] for k_ in keys})
            for k_ in keys:
                for s2 in (A, B):
                    c[k_][s2] += p * o[k_][s2]
    for g, c in cond.items():
        for k_ in c:
            c[k_] = [x / mass[g] for x in c[k_]]
    return {"mean": mean, "cond": cond, "mass": mass, "branches": len(dist)}


__all__ = ["turn_distribution", "expect", "outcome"]
