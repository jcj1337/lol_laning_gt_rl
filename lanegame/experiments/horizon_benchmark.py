"""Horizon benchmark: state counts, runtime, memory, residuals, and how values change with K.

Safety: before each solve the reachable state count is enumerated with a hard
cap (max_states); the solver is launched only if the enumeration succeeds.
Peak memory is measured with tracemalloc (Python allocations, including
NumPy buffers) around enumeration + solve.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from .. import GameConfig, LaneGame
from ..analysis import path_statistics
from ..baselines import all_baselines
from ..best_response import exploitability
from ..config import ACTION_SHORT
from ..solver import solve_equilibrium
from ..state import State
from .common import md_table, save

HORIZONS = (16, 32, 64)
EXTENDED = (128, 256)
MAX_STATES = 2_000_000

# probe states evaluated with a fixed number of turns remaining (16), placed at k = K - 16
PROBES = {
    "A 2-stack hp6 at river": State(0, 6, (-4,), 0, -4, 2, (), 0, 0, 0),
    "A one wave hp2 at river": State(0, 2, (-4,), 0, -4, 2, (), 0, 0, 0),
    "B wave held at A freeze line": State(0, 0, (-4,), 0, 1, 2, (), 0, 0, 0),
}


def bench(K: int) -> dict:
    g = LaneGame(GameConfig(horizon=K))
    tracemalloc.start()
    t0 = time.perf_counter()
    layers = g.reachable(max_states=MAX_STATES)
    t1 = time.perf_counter()
    sol = solve_equilibrium(g, layers)
    t2 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    s0 = g.initial_state()
    la, lb = sol.legal[0][s0]
    ps = path_statistics(g, sol.policy(0), sol.policy(1))
    distinct = len({s for l in layers for s in l})
    d = {
        "horizon": K, "movement_ticks": K * g.cfg.ticks_per_turn,
        "spawn_intervals": K * g.cfg.ticks_per_turn / g.cfg.spawn_interval,
        "reachable_states": sol.stats.states, "max_layer": max(len(l) for l in layers),
        "distinct_lane_states": distinct,
        "move_games": sol.stats.move_games, "choice_games": sol.stats.choice_games,
        "methods": sol.stats.methods, "mixed_states": sol.stats.mixed_states,
        "enumerate_seconds": t1 - t0, "solve_seconds": t2 - t1,
        "peak_memory_mb": peak / 1e6, "max_certified_gap": sol.stats.max_gap,
        "value_s0": sol.value,
        "eq_s0": {"A": {ACTION_SHORT[a]: float(p) for a, p in zip(la, sol.x[0][s0])},
                  "B": {ACTION_SHORT[b]: float(p) for b, p in zip(lb, sol.y[0][s0])}},
        "eq_path_gank_loss_per_turn": [x / K for x in ps.gank_loss],
        "eq_path_crashes": ps.crashes,
        "baseline_exploitability": {b.name: exploitability(g, layers, b, 0, sol.value).exploitability
                                    for b in all_baselines()},
        "layer_sizes": [len(l) for l in layers],
    }
    if K >= 16:
        d["probe_values_16_remaining"] = {n: solve_equilibrium(g, root=s, k0=K - 16).value
                                          for n, s in PROBES.items()}
    return d


def run(horizons=HORIZONS, extended=EXTENDED) -> dict:
    res = {K: bench(K) for K in tuple(horizons) + tuple(extended)}
    rows = [(K, d["movement_ticks"], d["reachable_states"], d["max_layer"], d["distinct_lane_states"],
             d["move_games"], d["choice_games"], d["methods"].get("lp", 0), d["mixed_states"],
             round(d["enumerate_seconds"] + d["solve_seconds"], 3), round(d["peak_memory_mb"], 1),
             f"{d['max_certified_gap']:.1e}") for K, d in res.items()]
    md = ["# Horizon benchmark", "",
          md_table(["K", "ticks", "reachable (k,s)", "max layer", "distinct lane states", "move games",
                    "choice games", "LPs", "mixed states", "seconds", "peak MB", "max gap"], rows), "",
          "## Values and strategies by horizon", "",
          md_table(["K", "V*(s0)", "eq at s0 (A)", "eq-path gank loss/turn (A)", "eq-path crashes A (S/M/L)"],
                   [(K, round(d["value_s0"], 9), d["eq_s0"]["A"], round(d["eq_path_gank_loss_per_turn"][0], 4),
                     [round(c, 3) for c in d["eq_path_crashes"][0]]) for K, d in res.items()]), "",
          "## Exploitability of baselines (as A), total and per turn", "",
          md_table(["K"] + [b.name for b in all_baselines()],
                   [(K, *[f"{v:.2f} ({v / K:.3f}/turn)" for v in d["baseline_exploitability"].values()])
                    for K, d in res.items()]), "",
          "## Probe-state values with 16 turns remaining (independent of K if the end effect dominates)", "",
          md_table(["K"] + list(PROBES), [(K, *[round(v, 4) for v in d["probe_values_16_remaining"].values()])
                                          for K, d in res.items()])]
    save("horizon_benchmark", {"results": res, "max_states_limit": MAX_STATES}, "\n".join(md))
    return res


if __name__ == "__main__":
    run()
