"""Shared helpers for experiment scripts: output paths, Markdown tables and common measurements."""

from __future__ import annotations

import os
import platform
import sys
import time
from typing import Iterable, List, Sequence

import numpy as np
import scipy

from ..io import dump_json

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
REVISION = "revision"          # sub-directory of results/ for the revised model's experiments


def out_path(name: str, subdir: str | None = None) -> str:
    d = RESULTS_DIR if subdir is None else os.path.join(RESULTS_DIR, subdir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def environment() -> dict:
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def md_table(header: Sequence[str], rows: Iterable[Sequence]) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"
        return str(v)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def save(name: str, data: dict, markdown: str, subdir: str | None = None) -> None:
    data = dict(data)
    data["environment"] = environment()
    dump_json(data, out_path(name + ".json", subdir))
    with open(out_path(name + ".md", subdir), "w", encoding="utf-8") as f:
        f.write(markdown.rstrip() + "\n")
    print(markdown)


# ----------------------------------------------------------------- measurements (revised experiments)
def solve(cfg):
    """Enumerate and solve; returns (game, layers, solution, seconds)."""
    from ..game import LaneGame
    from ..solver import solve_equilibrium
    g = LaneGame(cfg)
    t = time.perf_counter()
    layers = g.reachable()
    sol = solve_equilibrium(g, layers)
    return g, layers, sol, time.perf_counter() - t


def node_averages(sol) -> dict:
    """Average equilibrium behaviour of player A over all reachable decision nodes (unweighted)."""
    from ..config import F, posture_of, wave_of
    n = 0
    contest = 0.0
    wave = np.zeros(3)
    freeze_nodes = 0
    freeze_prob = 0.0
    for k, layer in enumerate(sol.layers):
        for s in layer:
            la, _ = sol.legal[k][s]
            x = sol.x[k][s]
            n += 1
            contest += sum(p for a, p in zip(la, x) if posture_of(a) == 0)
            for a, p in zip(la, x):
                wave[wave_of(a)] += p
            if F in la:
                freeze_nodes += 1
                freeze_prob += sum(p for a, p in zip(la, x) if wave_of(a) == F)
    return {"nodes": n, "contest": contest / n, "wave": (wave / n).tolist(),
            "freeze_legal_nodes": freeze_nodes,
            "freeze_prob_when_legal": freeze_prob / freeze_nodes if freeze_nodes else float("nan")}


def vision_share(sol, cfg) -> dict:
    """Mean probability that the equilibrium takes the vision option, per crash size (A's choice nodes)."""
    from ..config import SIZE_NAMES
    out = {}
    for key, x in sol.choice_x.items():
        k, pre, oa, ob = key
        if len(oa) > 1:
            size = SIZE_NAMES[list(cfg.crash_options).index(tuple(oa))]
            vis = sum(p for o, p in zip(oa, x) if o[1])
            out.setdefault(size, []).append(float(vis))
    return {k: float(np.mean(v)) for k, v in out.items()} | {f"{k}_nodes": len(v) for k, v in out.items()}


def eq_summary(g, sol) -> dict:
    """Equilibrium-path statistics (exact) plus node averages."""
    from ..analysis import path_statistics
    K = g.horizon
    ps = path_statistics(g, sol.policy(0), sol.policy(1)).to_dict(K)
    return {"value": sol.value, "path": ps, "nodes": node_averages(sol), "vision": vision_share(sol, g.cfg),
            "states": sol.stats.states, "distinct": len({s for l in sol.layers for s in l}),
            "mixed": sol.stats.mixed_states, "max_gap": sol.stats.max_gap}


def exploit_row(g, layers, sol, policy) -> dict:
    """Exploitability of `policy` as player A, and where the best response's gain comes from.

    farm/crash/gank are the baseline's expected (A minus B) totals along the best-response path;
    their sum is the baseline's value against the best response (= v* - exploitability).
    """
    from ..analysis import path_statistics
    from ..best_response import best_response_value, evaluate
    br = best_response_value(g, layers, policy, 0)
    w = br.value(g)
    ps = path_statistics(g, policy, br.policy).to_dict(g.horizon)
    return {
        "exploitability": sol.value - w,
        "value_vs_equilibrium": evaluate(g, layers, policy, sol.policy(1)),
        "farm": ps["farm"][0] - ps["farm"][1],
        "crash": ps["crash_resources"][0] - ps["crash_resources"][1],
        "gank": -(ps["gank_loss"][0] - ps["gank_loss"][1]),
        "br_path": ps,
    }


def fmt_freq(d: dict, cut: float = 0.005) -> str:
    return ", ".join(f"{k} {v:.2f}" for k, v in sorted(d.items()) if v > cut)
