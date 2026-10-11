"""Computational benchmark of the revised model: state counts, transition generation, solve time,
memory, residuals, and how values and exploitability change with the horizon K.

Safety: enumeration runs with a hard cap (max_states) before solving. Peak memory is the Python
allocation peak measured with tracemalloc around enumeration + solve (tracemalloc slows Python down,
so timings are reported from a separate untraced run).
"""

from __future__ import annotations

import time
import tracemalloc

from .. import GameConfig, LaneGame
from ..analysis import path_statistics
from ..baselines import all_baselines
from ..best_response import exploitability
from ..solver import solve_equilibrium
from .common import REVISION, md_table, save
from .probes import probes

HORIZONS = (16, 32, 64, 128)
MAX_STATES = 3_000_000
PROBES = ["B wave at A's freeze line, fighting A's wave", "A 2-wave stack at +1, B wave just spawned",
          "A's 3-wave crash under B's tower"]


def bench(K: int) -> dict:
    cfg = GameConfig(horizon=K)
    # timed run
    g = LaneGame(cfg)
    t0 = time.perf_counter()
    layers = g.reachable(max_states=MAX_STATES)
    t1 = time.perf_counter()
    sol = solve_equilibrium(g, layers)
    t2 = time.perf_counter()
    transitions = len(g._cache)
    # memory run (fresh game, traced)
    g2 = LaneGame(cfg)
    tracemalloc.start()
    l2 = g2.reachable(max_states=MAX_STATES)
    solve_equilibrium(g2, l2)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del g2, l2
    s0 = g.initial_state()
    la, lb = sol.legal[0][s0]
    ps = path_statistics(g, sol.policy(0), sol.policy(1)).to_dict(K)
    t3 = time.perf_counter()
    expl = {b.name: exploitability(g, layers, b, 0, sol.value).exploitability for b in all_baselines()}
    t4 = time.perf_counter()
    pr = probes(cfg.wave_hp)
    return {
        "horizon": K, "movement_ticks": K * cfg.ticks_per_turn,
        "spawn_intervals": K * cfg.ticks_per_turn / cfg.spawn_interval,
        "decision_nodes": sol.stats.states, "max_layer": max(len(l) for l in layers),
        "distinct_states": len({s for l in layers for s in l}),
        "cached_transitions": transitions,
        "move_games": sol.stats.move_games, "choice_games": sol.stats.choice_games,
        "methods": sol.stats.methods, "mixed_states": sol.stats.mixed_states,
        "enumerate_seconds": t1 - t0, "solve_seconds": t2 - t1,
        "transitions_per_second": transitions / (t1 - t0),
        "best_response_seconds": (t4 - t3) / len(expl),
        "peak_memory_mb": peak / 1e6, "max_certified_gap": sol.stats.max_gap,
        "value_s0": sol.value,
        "eq_s0": {"A": {cfg.label(a): float(p) for a, p in zip(la, sol.x[0][s0])}},
        "eq_path_per_turn": {"farm": ps["farm"][0] / K, "gank_loss": ps["gank_loss"][0] / K,
                             "contest": ps["contest_share"][0]},
        "baseline_exploitability": expl,
        "probe_values_16_remaining": {n: solve_equilibrium(g, root=pr[n], k0=K - 16).value for n in PROBES},
        "layer_sizes": [len(l) for l in layers],
    }


def run(horizons=HORIZONS) -> dict:
    res = {}
    for K in horizons:
        print("K", K, flush=True)
        res[K] = bench(K)
    names = [b.name for b in all_baselines()]
    md = ["# Horizon benchmark (revised model)", "",
          "Single-threaded Python; timings from an untraced run, peak memory from a separate tracemalloc run.", "",
          md_table(["K", "ticks", "decision nodes", "max layer", "distinct states", "cached transitions",
                    "transitions/s", "stage games", "choice games", "LPs", "mixed", "enumerate s", "solve s",
                    "BR s (each)", "peak MB", "max gap"],
                   [(K, d["movement_ticks"], d["decision_nodes"], d["max_layer"], d["distinct_states"],
                     d["cached_transitions"], int(d["transitions_per_second"]), d["move_games"], d["choice_games"],
                     d["methods"].get("lp", 0), d["mixed_states"], round(d["enumerate_seconds"], 2),
                     round(d["solve_seconds"], 2), round(d["best_response_seconds"], 2),
                     round(d["peak_memory_mb"], 1), f"{d['max_certified_gap']:.1e}") for K, d in res.items()]), "",
          "## Values and the equilibrium path by horizon", "",
          md_table(["K", "V*(s0)", "eq at s0 (A)", "eq-path farm/turn", "eq-path gank loss/turn", "eq-path P(Contest)"],
                   [(K, round(d["value_s0"], 9), d["eq_s0"]["A"], round(d["eq_path_per_turn"]["farm"], 4),
                     round(d["eq_path_per_turn"]["gank_loss"], 4), round(d["eq_path_per_turn"]["contest"], 3))
                    for K, d in res.items()]), "",
          "## Exploitability of baselines (as A): total, and per turn", "",
          md_table(["K"] + names, [(K, *[f"{d['baseline_exploitability'][n]:.2f} ({d['baseline_exploitability'][n] / K:.3f})"
                                         for n in names]) for K, d in res.items()]), "",
          "## Probe values with 16 turns remaining (same phase), by horizon", "",
          "The dynamics are time-homogeneous up to the phase k mod 4, so these must be identical for every K.", "",
          md_table(["K"] + PROBES, [(K, *[round(d["probe_values_16_remaining"][n], 6) for n in PROBES])
                                    for K, d in res.items()])]
    save("horizon_benchmark", {"results": res, "max_states_limit": MAX_STATES}, "\n".join(md), REVISION)
    return res


if __name__ == "__main__":
    run()
