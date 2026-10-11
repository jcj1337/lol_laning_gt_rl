"""Computational benchmark of the siege model: reachable states, transitions, solve time and memory by
horizon and tower rate, and the cost of the alternative plate representation (per-tower budget).

Run first: it establishes which horizons the sweeps can afford (results/siege/benchmark.md)."""

from __future__ import annotations

import time
import tracemalloc

from ... import GameConfig, LaneGame
from ...solver import solve_equilibrium
from .common import K, md_table, save

CASES = [
    ("previous revision (choice), tower 1", lambda K_: GameConfig.v2(horizon=K_), (16, 24, 32)),
    ("siege default (crash plates, vision 6), tower 1", lambda K_: GameConfig(horizon=K_), (16, 24, 32)),
    ("siege, tower 2", lambda K_: GameConfig(horizon=K_, tower_siege_damage=2), (16, 24, 32, 48)),
    ("siege, tower 3", lambda K_: GameConfig(horizon=K_, tower_siege_damage=3), (16, 24, 32, 48)),
    ("siege, vision 4", lambda K_: GameConfig(horizon=K_, vision_turns=4), (24,)),
    ("siege, vision 8", lambda K_: GameConfig(horizon=K_, vision_turns=8), (24,)),
    ("siege, vision 2 (old duration)", lambda K_: GameConfig(horizon=K_, vision_turns=2), (24,)),
    ("siege, no thinning", lambda K_: GameConfig(horizon=K_, thin_damage=0), (24,)),
    ("siege, per-tower plate budget (5)", lambda K_: GameConfig(horizon=K_, plate_mode="tower"), (16, 24)),
]
MAX_STATES = 2_000_000


def bench(cfg: GameConfig, memory: bool = True) -> dict:
    g = LaneGame(cfg)
    t0 = time.perf_counter()
    layers = g.reachable(max_states=MAX_STATES)
    t1 = time.perf_counter()
    sol = solve_equilibrium(g, layers)
    t2 = time.perf_counter()
    out = {"horizon": cfg.horizon, "nodes": sol.stats.states, "distinct": len({s for l in layers for s in l}),
           "max_layer": max(len(l) for l in layers), "transitions": len(g._cache),
           "enumerate_s": t1 - t0, "solve_s": t2 - t1, "lps": sol.stats.methods.get("lp", 0),
           "mixed": sol.stats.mixed_states, "max_gap": sol.stats.max_gap, "value": sol.value,
           "layer_sizes": [len(l) for l in layers]}
    if memory:
        del g, layers, sol
        g2 = LaneGame(cfg)
        tracemalloc.start()
        l2 = g2.reachable(max_states=MAX_STATES)
        solve_equilibrium(g2, l2)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out["peak_mb"] = peak / 1e6
        del g2, l2
    return out


def run() -> dict:
    res = {}
    for name, make, horizons in CASES:
        res[name] = {}
        for K_ in horizons:
            print(name, K_, flush=True)
            try:
                res[name][K_] = bench(make(K_), memory=(K_ <= 32))
            except RuntimeError as err:
                res[name][K_] = {"error": str(err)}
    rows = []
    for name, d in res.items():
        for K_, r in d.items():
            if "error" in r:
                rows.append((name, K_, "> cap", "", "", "", "", "", "", "", "", ""))
            else:
                rows.append((name, K_, r["nodes"], r["distinct"], r["max_layer"], r["transitions"],
                             round(r["enumerate_s"], 1), round(r["solve_s"], 1), r["lps"], r["mixed"],
                             round(r.get("peak_mb", float("nan")), 0), f"{r['max_gap']:.1e}"))
    md = ["# Computational benchmark (siege model)", "",
          "Single-threaded Python; enumeration and solve timed in an untraced run; peak memory (Python allocations, "
          "tracemalloc) from a separate traced run for K <= 32. 'nodes' = reachable (turn, state) pairs; "
          "'transitions' = cached joint-action outcomes.", "",
          md_table(["model", "K", "nodes", "distinct states", "max layer", "transitions", "enumerate s", "solve s",
                    "LPs", "mixed", "peak MB", "max gap"], rows), "",
          "## Reading the table", "",
          "- The siege model's state space is larger than the previous revision's because the vision timer "
          "(0..6 per side) and the pending plate opportunities (0..6 per side, only while a siege lasts) are "
          "Markov state, and because thinning creates intermediate HP values for the defender's waiting wave.",
          "- The per-tower plate budget (a persistent counter) multiplies the state space by a further factor "
          "and was rejected for that reason (section C of the study).",
          "- Faster tower fire shortens sieges and therefore shrinks the space in which vision timers and "
          "pending plates co-exist with lane states.",
          "- The sweeps (vision, ablations, crash values) are run at K = 24 where the 1 HP/tick model needs about "
          "half a minute per solve; headline results use K = 32."]
    save("benchmark", {"cases": {n: {str(k): v for k, v in d.items()} for n, d in res.items()}, "max_states": MAX_STATES},
         "\n".join(md))
    return res


if __name__ == "__main__":
    run()
