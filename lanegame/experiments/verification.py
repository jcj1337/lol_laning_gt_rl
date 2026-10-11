"""Independent numerical verification of the solver on the revised model.

1. Every stage game (and every crash-choice game) of the K = 32 solution re-solved by support
   enumeration (an LP-free method) -> max |value difference|.
2. Backward induction vs. plain tree recursion (no state merging, no caching of values, support
   enumeration at every node) from several roots.
3. Exploitability of the computed equilibrium (both players) for K = 16, 32, 64.
4. Monte Carlo (sampled ganks and mixed actions) vs. exact expected values, with z-scores.
5. The original model is reproduced exactly by GameConfig.v1().
"""

from __future__ import annotations

import time

import numpy as np

from .. import GameConfig, LaneGame
from ..baselines import all_baselines, uniform, v1_baselines
from ..best_response import evaluate, exploitability
from ..game import apply_choice
from ..matrix_game import support_enumeration
from ..simulate import simulate
from ..solver import solve_equilibrium
from ..state import State
from .common import REVISION, md_table, save, solve


def tree_minimax(game, s, k):
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


ROOTS = [("symmetric start", None, 0, 3),
         ("A 3-wave stack at the river", State(0, 12, (-4,), 0, -4, 4, (), 0, 0, 0), 2, 2),
         ("A 2-wave packet near B's tower", State(1, 8, (), 0, -3, 4, (), 0, 0, 0), 1, 2),
         ("B wave at A's freeze line", State(-4, 4, (), 0, 1, 4, (), 0, 0, 0), 3, 2),
         ("crashed mass under B's tower, A vision", State(-4, 4, (), 9, -3, 4, (), 0, 1, 0), 2, 2)]


def run() -> dict:
    out = {}
    cfg = GameConfig(horizon=32)
    g, layers, sol, _ = solve(cfg)
    # 1
    t = time.perf_counter()
    diffs = [abs(support_enumeration(sol.Q[k][s])[0] - sol.V[k][s]) for k, layer in enumerate(sol.layers) for s in layer]
    cdiffs = []
    for key, val in sol.choice_val.items():
        k, pre, oa, ob = key
        V_next = sol.V[k + 1] if k + 1 < g.horizon else None
        C = np.empty((len(oa), len(ob)))
        for i, o1 in enumerate(oa):
            for j, o2 in enumerate(ob):
                s2, r = apply_choice(pre, o1, o2, cfg)
                C[i, j] = r + (V_next[s2] if V_next is not None else 0.0)
        cdiffs.append(abs(support_enumeration(C)[0] - val))
    out["stage_games_vs_support_enumeration"] = {"games": len(diffs), "max_abs_diff": float(max(diffs)),
                                                 "choice_games": len(cdiffs),
                                                 "choice_max_abs_diff": float(max(cdiffs)) if cdiffs else 0.0,
                                                 "seconds": time.perf_counter() - t,
                                                 "methods": sol.stats.methods, "max_gap": sol.stats.max_gap}
    # 2
    tree = []
    for name, root, k0, depth in ROOTS:
        gg = LaneGame(GameConfig(horizon=k0 + depth))
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
        gg, ll, so, _ = solve(GameConfig(horizon=K))
        ea = exploitability(gg, ll, so.policy(0), 0, so.value).exploitability
        eb = exploitability(gg, ll, so.policy(1), 1, so.value).exploitability
        expl.append({"K": K, "expl_A": ea, "expl_B": eb, "max_gap": so.stats.max_gap})
    out["equilibrium_exploitability"] = expl
    # 4
    mc = []
    pairs = [("equilibrium", sol.policy(0), "equilibrium", sol.policy(1)),
             ("equilibrium", sol.policy(0), "uniform", uniform())]
    for b in all_baselines()[1:]:
        pairs.append((b.name, b, "equilibrium", sol.policy(1)))
    for i, (na, pa, nb, pb) in enumerate(pairs):
        exact = evaluate(g, layers, pa, pb)
        sim = simulate(g, pa, pb, 4000, seed=100 + i)
        mc.append({"A": na, "B": nb, "exact": exact, "mc_mean": sim.mean, "mc_stderr": sim.stderr,
                   "z": (sim.mean - exact) / sim.stderr})
    out["monte_carlo"] = mc
    # 5
    g1, l1, s1, _ = solve(GameConfig.v1(horizon=32))
    v1 = {"nodes": s1.stats.states, "mixed": s1.stats.mixed_states, "value": s1.value,
          "exploitability": {b.name: exploitability(g1, l1, b, 0, s1.value).exploitability for b in v1_baselines()}}
    out["v1_reproduction"] = v1

    sv = out["stage_games_vs_support_enumeration"]
    md = ["# Verification (revised model)", "",
          f"Stage games of the K = 32 solution re-solved by support enumeration: {sv['games']} games, max |value "
          f"difference| = {sv['max_abs_diff']:.2e}; crash-choice games: {sv['choice_games']}, max difference "
          f"{sv['choice_max_abs_diff']:.2e}. Solution methods: {sv['methods']}; max certified duality gap "
          f"{sv['max_gap']:.1e}.", "",
          "## Backward induction vs. tree recursion (support enumeration at every node)", "",
          md_table(["root", "turns", "backward induction", "tree minimax", "|diff|", "seconds"],
                   [(r["root"], r["depth"], round(r["backward_induction"], 10), round(r["tree_minimax"], 10),
                     f"{r['abs_diff']:.1e}", round(r["seconds"], 1)) for r in tree]), "",
          "## Exploitability of the computed equilibrium", "",
          md_table(["K", "Expl_A", "Expl_B", "max certified gap"],
                   [(e["K"], f"{e['expl_A']:.1e}", f"{e['expl_B']:.1e}", f"{e['max_gap']:.1e}") for e in expl]), "",
          "## Monte Carlo (4000 games, sampled ganks and mixed actions) vs. exact expectation, K = 32", "",
          md_table(["A", "B", "exact", "MC mean", "stderr", "z"],
                   [(m["A"], m["B"], round(m["exact"], 4), round(m["mc_mean"], 4), round(m["mc_stderr"], 4),
                     round(m["z"], 2)) for m in mc]), "",
          "## Original model reproduced by GameConfig.v1() (K = 32)", "",
          f"{v1['nodes']} decision nodes, {v1['mixed']} mixed, V*(s0) = {v1['value']:.3g}; exploitability: "
          + ", ".join(f"{k} {v:.4f}" for k, v in v1["exploitability"].items())
          + " (original report: 905 nodes, 52 mixed, 11.86 / 19.45 / 4.85 / 1.85 / 4.85)."]
    save("verification", out, "\n".join(md), REVISION)
    return out


if __name__ == "__main__":
    run()
