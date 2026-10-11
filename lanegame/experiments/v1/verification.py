"""Independent numerical verification of the solver.

1. Every stage game of the K=32 solution re-solved by support enumeration
   (an LP-free method) -> max |value difference|.
2. Backward induction vs plain tree recursion (no state merging) on short
   horizons from several roots.
3. Exploitability of the computed equilibrium (both players) for K = 16/32/64.
4. Monte Carlo (sampled ganks) vs exact expected values, with z-scores.
"""

from __future__ import annotations

import time

import numpy as np

from ... import GameConfig, LaneGame
from ...baselines import uniform, v1_baselines as all_baselines
from ...best_response import evaluate, exploitability
from ...matrix_game import support_enumeration
from ...simulate import simulate
from ...solver import solve_equilibrium
from ...state import State
from ..common import md_table, save


def tree_minimax(game, s, k):
    from ...game import apply_choice
    if k == game.horizon:
        return 0.0
    la, lb = game.legal(s, 0), game.legal(s, 1)
    Q = np.zeros((len(la), len(lb)))
    for i, a in enumerate(la):
        for j, b in enumerate(lb):
            tr = game.transition(s, k, a, b)
            C = np.zeros((len(tr.options_a), len(tr.options_b)))
            for ii, oa in enumerate(tr.options_a):
                for jj, ob in enumerate(tr.options_b):
                    s2, r = apply_choice(tr.pre_state, oa, ob, game.cfg)
                    C[ii, jj] = r + tree_minimax(game, s2, k + 1)
            Q[i, j] = tr.payoff + support_enumeration(C)[0]
    return support_enumeration(Q)[0]


ROOTS = [("symmetric start", None, 0),
         ("A 2-stack at lane", State(0, 6, (-4,), 0, -4, 2, (), 0, 0, 0), 2),
         ("A medium wave near B tower", State(1, 4, (), 0, -3, 2, (), 0, 0, 0), 1),
         ("B wave at A freeze line", State(0, 0, (-2,), 0, 1, 2, (), 0, 0, 0), 3),
         ("two-packet siege, A vision", State(-4, 2, (), 2, -3, 2, (), 0, 1, 0), 2)]


def run() -> dict:
    out = {}
    # 1
    g = LaneGame(GameConfig.v1(horizon=32))
    sol = solve_equilibrium(g)
    diffs = []
    for k, layer in enumerate(sol.layers):
        for s in layer:
            diffs.append(abs(support_enumeration(sol.Q[k][s])[0] - sol.V[k][s]))
    out["stage_games_vs_support_enumeration"] = {"games": len(diffs), "max_abs_diff": float(max(diffs))}
    # 2
    tree = []
    for name, root, k0 in ROOTS:
        for depth in (2, 3, 4):
            gg = LaneGame(GameConfig.v1(horizon=k0 + depth))
            t = time.perf_counter()
            bi = solve_equilibrium(gg, root=root, k0=k0).value
            s = gg.initial_state() if root is None else root
            tm = tree_minimax(gg, s, k0)
            tree.append({"root": name, "depth": depth, "backward_induction": bi, "tree_minimax": tm,
                         "abs_diff": abs(bi - tm), "seconds": time.perf_counter() - t})
    out["tree_minimax"] = tree
    # 3
    expl = []
    for K in (16, 32, 64):
        gg = LaneGame(GameConfig.v1(horizon=K))
        so = solve_equilibrium(gg)
        ea = exploitability(gg, so.layers, so.policy(0), 0, so.value).exploitability
        eb = exploitability(gg, so.layers, so.policy(1), 1, so.value).exploitability
        expl.append({"K": K, "expl_A": ea, "expl_B": eb, "max_gap": so.stats.max_gap})
    out["equilibrium_exploitability"] = expl
    # 4
    mc = []
    pairs = [("equilibrium", sol.policy(0), "equilibrium", sol.policy(1)),
             ("equilibrium", sol.policy(0), "uniform", uniform())]
    for b in all_baselines()[1:]:
        pairs.append((b.name, b, "equilibrium", sol.policy(1)))
    for i, (na, pa, nb, pb) in enumerate(pairs):
        exact = evaluate(g, sol.layers, pa, pb)
        sim = simulate(g, pa, pb, 4000, seed=100 + i)
        mc.append({"A": na, "B": nb, "exact": exact, "mc_mean": sim.mean, "mc_stderr": sim.stderr,
                   "z": (sim.mean - exact) / sim.stderr})
    out["monte_carlo"] = mc

    md = ["# Verification", "",
          f"Stage games of the K=32 solution re-solved by support enumeration: {len(diffs)} games, "
          f"max |value difference| = {max(diffs):.2e}.", "",
          "## Backward induction vs tree recursion (support enumeration at every node)", "",
          md_table(["root", "turns", "backward induction", "tree minimax", "|diff|"],
                   [(r["root"], r["depth"], round(r["backward_induction"], 10), round(r["tree_minimax"], 10),
                     f"{r['abs_diff']:.1e}") for r in tree]), "",
          "## Exploitability of the computed equilibrium", "",
          md_table(["K", "Expl_A", "Expl_B", "max certified gap"],
                   [(e["K"], f"{e['expl_A']:.1e}", f"{e['expl_B']:.1e}", f"{e['max_gap']:.1e}") for e in expl]), "",
          "## Monte Carlo (4000 games, sampled ganks) vs exact expectation, K = 32", "",
          md_table(["A", "B", "exact", "MC mean", "stderr", "z"],
                   [(m["A"], m["B"], round(m["exact"], 4), round(m["mc_mean"], 4), round(m["mc_stderr"], 4),
                     round(m["z"], 2)) for m in mc])]
    save("verification", out, "\n".join(md))
    return out


if __name__ == "__main__":
    run()
