"""Shared helpers of the siege-study experiments: configurations, measurements and output."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from ... import GameConfig, LaneGame
from ...analysis import path_statistics
from ...baselines import siege_baselines
from ...best_response import best_response_value, evaluate, exploitability
from ...config import PRESSURE, VISION, objective_of, posture_of
from ...solver import solve_equilibrium
from ...state import State
from ..common import md_table, save as _save  # noqa: F401

SIEGE = "siege"                      # results/siege/
K = 32                               # horizon of the headline results
K_SWEEP = 24                         # horizon of the parameter sweeps (see benchmark.md for the cost argument)
MAX_STATES = 1_500_000


def save(name: str, data: dict, markdown: str) -> None:
    _save(name, data, markdown, SIEGE)


# ----------------------------------------------------------------- configurations
def default(**kw) -> GameConfig:
    return GameConfig(**kw)


def previous(**kw) -> GameConfig:
    """The previous revision (post-crash resource/vision choice)."""
    return GameConfig.v2(**kw)


CONTROLLED = {
    # name: (config factory, description)
    "choice, tower 1 (previous revision)": (lambda K: previous(horizon=K), "original siege rewards, 1 HP/tick"),
    "choice, tower 2": (lambda K: previous(horizon=K, tower_siege_damage=2), "original siege rewards, 2 HP/tick"),
    "choice, tower 3": (lambda K: previous(horizon=K, tower_siege_damage=3), "original siege rewards, 3 HP/tick"),
    "objective, tower 1": (lambda K: default(horizon=K, tower_siege_damage=1), "Pressure/Vision, 1 HP/tick"),
    "objective, tower 2": (lambda K: default(horizon=K, tower_siege_damage=2), "Pressure/Vision, 2 HP/tick"),
    "objective, tower 3": (lambda K: default(horizon=K, tower_siege_damage=3), "Pressure/Vision, 3 HP/tick"),
}


# ----------------------------------------------------------------- solving
def solve(cfg: GameConfig, max_states: int = MAX_STATES):
    """Enumerate and solve; returns (game, layers, solution, timings dict)."""
    g = LaneGame(cfg)
    t0 = time.perf_counter()
    layers = g.reachable(max_states=max_states)
    t1 = time.perf_counter()
    sol = solve_equilibrium(g, layers)
    t2 = time.perf_counter()
    return g, layers, sol, {"enumerate_s": t1 - t0, "solve_s": t2 - t1, "transitions": len(g._cache)}


def state_counts(layers) -> dict:
    return {"nodes": sum(len(l) for l in layers), "distinct": len({s for l in layers for s in l}),
            "max_layer": max(len(l) for l in layers)}


# ----------------------------------------------------------------- measurements
def node_averages(sol) -> dict:
    """Average equilibrium behaviour of A over all reachable decision nodes, and over siege nodes."""
    n = 0
    contest = 0.0
    siege_nodes = 0
    pressure = vision = disengage = 0.0
    for k, layer in enumerate(sol.layers):
        for s in layer:
            la, _ = sol.legal[k][s]
            x = sol.x[k][s]
            n += 1
            contest += sum(p for a, p in zip(la, x) if posture_of(a) == 0)
            if s.asg > 0 and sol.game.cfg.objectives:
                siege_nodes += 1
                pressure += sum(p for a, p in zip(la, x) if objective_of(a) == PRESSURE)
                vision += sum(p for a, p in zip(la, x) if objective_of(a) == VISION)
                disengage += sum(p for a, p in zip(la, x) if posture_of(a) == 1)
    d = max(siege_nodes, 1)
    return {"nodes": n, "contest": contest / n, "siege_nodes": siege_nodes,
            "p_pressure": pressure / d, "p_vision": vision / d, "p_disengage": disengage / d}


def siege_node_table(g, sol, max_k: Optional[int] = None) -> dict:
    """Equilibrium siege decisions of A grouped by siege mass, defender's waiting wave, plates left, vision,
    turns remaining; plus the value difference between the best Pressure and the best Vision action."""
    from collections import defaultdict
    cfg = g.cfg
    groups = {g_: defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
              for g_ in ("siege mass", "defender wave", "plates left", "vision", "turns remaining", "thinnable")}
    K = g.horizon
    for k, layer in enumerate(sol.layers):
        if max_k is not None and k > max_k:
            continue
        for s in layer:
            if s.asg == 0:
                continue
            la, lb = sol.legal[k][s]
            x, y = sol.x[k][s], sol.y[k][s]
            Q = sol.Q[k][s]
            vals = Q @ y
            best_p = max(v for a, v in zip(la, vals) if objective_of(a) == PRESSURE)
            best_v = max(v for a, v in zip(la, vals) if objective_of(a) == VISION)
            best_s = max(v for a, v in zip(la, vals) if posture_of(a) == 1)
            pp = sum(p for a, p in zip(la, x) if objective_of(a) == PRESSURE)
            pv = sum(p for a, p in zip(la, x) if objective_of(a) == VISION)
            ps = sum(p for a, p in zip(la, x) if posture_of(a) == 1)
            thinnable = s.bh > 0 and -cfg.tower - cfg.thin_reach <= s.bx <= -cfg.tower
            keys = {"siege mass": f"{s.asg:2d} HP", "defender wave": f"{s.bh:2d} HP waiting" if thinnable else "no accessible wave",
                    "plates left": str(s.ap), "vision": "active" if s.va else "none",
                    "turns remaining": f"{(K - k) // 8 * 8:2d}-{(K - k) // 8 * 8 + 7:2d}",
                    "thinnable": "yes" if thinnable else "no"}
            for gname, key in keys.items():
                row = groups[gname][key]
                row[0] += 1
                row[1] += pp
                row[2] += pv
                row[3] += ps
                row[4] += best_p - best_v
                row[5] += best_p - best_s
                row[6] += best_v - best_s
    out = {}
    for gname, gd in groups.items():
        out[gname] = {key: {"nodes": r[0], "p_pressure": r[1] / r[0], "p_vision": r[2] / r[0], "p_disengage": r[3] / r[0],
                            "pressure_minus_vision": r[4] / r[0], "pressure_minus_safe": r[5] / r[0],
                            "vision_minus_safe": r[6] / r[0]} for key, r in sorted(gd.items())}
    return out


def eq_path(g, sol) -> dict:
    return path_statistics(g, sol.policy(0), sol.policy(1)).to_dict(g.horizon)


def exploit_rows(g, layers, sol, policies=None) -> dict:
    """Exploitability of each baseline (as A) with the decomposition of its value against the best response."""
    policies = siege_baselines() if policies is None else policies
    out = {}
    for pol in policies:
        br = best_response_value(g, layers, pol, 0)
        w = br.value(g)
        ps = path_statistics(g, pol, br.policy).to_dict(g.horizon)
        out[pol.name] = {
            "exploitability": sol.value - w,
            "value_vs_equilibrium": path_statistics(g, pol, sol.policy(1)).value,
            "farm": ps["farm"][0] - ps["farm"][1],
            "plates": ps["plates"][0] - ps["plates"][1],
            "crash": ps["crash_resources"][0] - ps["crash_resources"][1],
            "gank": -(ps["gank_loss"][0] - ps["gank_loss"][1]),
            "br_path": ps,
        }
    return out


def summary(cfg: GameConfig, with_baselines: bool = True, policies=None, max_states: int = MAX_STATES) -> dict:
    """Everything the controlled comparison reports for one configuration."""
    g, layers, sol, timing = solve(cfg, max_states)
    d = {"config": cfg.to_dict(), **state_counts(layers), **timing, "mixed": sol.stats.mixed_states,
         "max_gap": sol.stats.max_gap, "lps": sol.stats.methods.get("lp", 0), "value": sol.value,
         "nodes_avg": node_averages(sol), "path": eq_path(g, sol)}
    if cfg.objectives:
        d["siege_nodes"] = siege_node_table(g, sol, g.horizon - 4)
    if with_baselines:
        t = time.perf_counter()
        d["baselines"] = {n: {k: v for k, v in r.items() if k != "br_path"} for n, r in exploit_rows(g, layers, sol, policies).items()}
        d["baselines_s"] = time.perf_counter() - t
        d["br_paths"] = {n: r["br_path"] for n, r in exploit_rows(g, layers, sol, policies).items()} if False else None
    return d, (g, layers, sol)


def fmt_freq(d: dict, cut: float = 0.005) -> str:
    return ", ".join(f"{k} {v:.2f}" for k, v in sorted(d.items()) if v > cut)


def probe_value(cfg: GameConfig, state: State, k0: int, horizon: Optional[int] = None):
    """Equilibrium value and A's strategy for a probe state with (horizon - k0) turns remaining."""
    c = cfg if horizon is None else cfg.with_(horizon=horizon)
    g = LaneGame(c)
    sol = solve_equilibrium(g, root=state, k0=k0)
    la, lb = sol.legal[k0][state]
    return {"value": sol.value,
            "A": {c.label(a): round(float(p), 3) for a, p in zip(la, sol.x[k0][state]) if p > 1e-9},
            "B": {c.label(b): round(float(p), 3) for b, p in zip(lb, sol.y[k0][state]) if p > 1e-9},
            "Q_A": {c.label(a): float(v) for a, v in zip(la, sol.Q[k0][state] @ sol.y[k0][state])}}, g, sol
