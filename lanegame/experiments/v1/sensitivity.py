"""One-at-a-time parameter sensitivity around the default configuration.

For each perturbed parameter we re-solve the symmetric game and report the
quantities that the Phase-1 conclusions rest on: whether equilibrium play
crashes, how much the equilibrium mixes, whether the vision option is ever
taken, probe-state values and the per-turn exploitability of baselines.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ... import GameConfig, LaneGame
from ...analysis import path_statistics
from ...baselines import v1_baselines as all_baselines
from ...best_response import exploitability
from ...config import SIZE_NAMES
from ...solver import solve_equilibrium
from ...state import State
from ..common import md_table, save

GRID = {
    "gank_loss": [3.0, 5.0, 7.0],
    "gank_base": [0.10, 0.15, 0.20],
    "gank_position": [0.05, 0.10, 0.15],
    "vision_reduction": [0.25, 0.5, 0.75],
    "freeze_break_threshold": [4, 5, 6],
    "shove_damage": [0, 1, 2],
    "tower_damage": [0, 1, 2],
    "wave_hp": [2, 3],
}


def probes(W: int):
    return {
        "A 2-stack at lane": (State(0, 3 * W, (-4,), 0, -4, W, (), 0, 0, 0), 10),
        "A one wave at lane": (State(0, W, (-4,), 0, -4, W, (), 0, 0, 0), 10),
    }


def evaluate_config(cfg: GameConfig) -> dict:
    g = LaneGame(cfg)
    sol = solve_equilibrium(g)
    ps = path_statistics(g, sol.policy(0), sol.policy(1))
    vis = defaultdict(list)
    for key, x in sol.choice_x.items():
        k, pre, oa, ob = key
        if len(oa) > 1:
            vis[SIZE_NAMES[list(cfg.crash_options).index(tuple(oa))]].append(float(x[1]))
    K = cfg.horizon
    return {
        "states": sol.stats.states, "mixed_states": sol.stats.mixed_states,
        "eq_path_crashes_A": float(sum(ps.crashes[0])),
        "vision_share": {k: float(np.mean(v)) for k, v in vis.items()},
        "probes": {n: solve_equilibrium(g, root=s, k0=k).value for n, (s, k) in probes(cfg.wave_hp).items()},
        "exploitability_per_turn": {b.name: exploitability(g, sol.layers, b, 0, sol.value).exploitability / K
                                    for b in all_baselines()},
    }


def run(horizon: int = 32) -> dict:
    base = GameConfig.v1(horizon=horizon)
    res = {}
    for param, values in GRID.items():
        for v in values:
            kw = {param: v}
            if param == "wave_hp":
                kw["freeze_break_threshold"] = int(round(base.freeze_break_threshold * v / base.wave_hp))
            res[f"{param}={v}"] = {"param": param, "value": v, **evaluate_config(base.with_(**kw))}
    rows = []
    for name, d in res.items():
        vs = d["vision_share"]
        rows.append((name, d["states"], d["mixed_states"], round(d["eq_path_crashes_A"], 3),
                     "/".join(f"{vs.get(s, float('nan')):.2f}" for s in ("small", "medium", "large")),
                     *[round(x, 2) for x in d["probes"].values()],
                     *[round(x, 3) for x in d["exploitability_per_turn"].values()]))
    md = [f"# One-at-a-time sensitivity (horizon {horizon})", "",
          "Vision share = fraction of crash-choice nodes (small/medium/large) where the equilibrium takes vision.",
          "Exploitability columns are per decision turn. `wave_hp=3` rescales the freeze threshold proportionally.", "",
          md_table(["setting", "states", "mixed", "eq crashes", "vision S/M/L", "V(A 2-stack @lane)",
                    "V(A 1 wave @lane)"] + [f"expl {b.name}" for b in all_baselines()], rows)]
    save("sensitivity", {"horizon": horizon, "results": res}, "\n".join(md))
    return res


if __name__ == "__main__":
    run()
