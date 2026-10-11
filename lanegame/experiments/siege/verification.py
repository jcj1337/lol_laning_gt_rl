"""Independent numerical verification of the solver on the siege model.

1. Every stage game of the K = 32 solution re-solved by support enumeration (LP-free) -> max |difference|.
2. Backward induction vs. plain tree recursion from siege roots (depth 2-3).
3. Exploitability of the computed equilibrium (both players) for K = 16, 24, 32.
4. Monte Carlo (sampled ganks and mixed actions) vs. exact expected values, with z-scores.
5. The previous revision and the original model reproduced by GameConfig.v2() / GameConfig.v1().
6. Accounting identities over every reachable transition at K = 24 (HP conservation with thinning; farm,
   plate and gank terms summing to the stage payoff; mirror antisymmetry of values).
"""

from __future__ import annotations

import time

import numpy as np

from ... import GameConfig, LaneGame
from ...baselines import all_baselines, siege_baselines, uniform, v1_baselines
from ...best_response import evaluate, exploitability
from ...game import apply_choice
from ...matrix_game import support_enumeration
from ...mechanics import step_turn, total_hp
from ...simulate import simulate
from ...solver import solve_equilibrium
from ...state import State
from .common import K, md_table, save, solve


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


ROOTS = [("symmetric start", None, 0, 2),
         ("A 2-wave packet near B's tower", State(1, 8, (), 0, -3, 4, (), 0, 0, 0), 1, 3),
         ("A's 9-HP mass under B's tower, 3 plates", State(-4, 4, (), 9, -3, 4, (), 0, 0, 0, 3, 0), 2, 3),
         ("... with vision, B's waiting wave stacked", State(-4, 4, (), 5, -3, 8, (), 0, 2, 0, 1, 0), 3, 2),
         ("both sides sieging", State(-4, 4, (), 4, -4, 4, (), 4, 0, 0, 1, 1), 2, 2)]


def run() -> dict:
    out = {}
    cfg = GameConfig(horizon=K)
    g, layers, sol, timing = solve(cfg)
    t = time.perf_counter()
    diffs = [abs(support_enumeration(sol.Q[k][s])[0] - sol.V[k][s]) for k, layer in enumerate(sol.layers) for s in layer]
    out["stage_games_vs_support_enumeration"] = {"games": len(diffs), "max_abs_diff": float(max(diffs)),
                                                 "seconds": time.perf_counter() - t, "methods": sol.stats.methods,
                                                 "max_gap": sol.stats.max_gap, "choice_games": sol.stats.choice_games}
    tree = []
    for name, root, k0, depth in ROOTS:
        gg = LaneGame(GameConfig(horizon=k0 + depth))
        t = time.perf_counter()
        bi = solve_equilibrium(gg, root=root, k0=k0).value
        s = gg.initial_state() if root is None else root
        tm = tree_minimax(gg, s, k0)
        tree.append({"root": name, "depth": depth, "backward_induction": bi, "tree_minimax": tm, "abs_diff": abs(bi - tm),
                     "seconds": time.perf_counter() - t})
    out["tree_minimax"] = tree
    expl = []
    for K_ in (16, 24, 32):
        gg, ll, so, _ = solve(GameConfig(horizon=K_))
        ea = exploitability(gg, ll, so.policy(0), 0, so.value).exploitability
        eb = exploitability(gg, ll, so.policy(1), 1, so.value).exploitability
        anti = max(abs(so.V[k][s] + so.V[k][s.mirror()]) for k, layer in enumerate(so.layers) for s in layer)
        expl.append({"K": K_, "expl_A": ea, "expl_B": eb, "max_gap": so.stats.max_gap, "max_antisymmetry": anti,
                     "nodes": so.stats.states})
    out["equilibrium_exploitability"] = expl
    mc = []
    pairs = [("equilibrium", sol.policy(0), "equilibrium", sol.policy(1)), ("equilibrium", sol.policy(0), "uniform", uniform())]
    for b in siege_baselines()[1:]:
        pairs.append((b.name, b, "equilibrium", sol.policy(1)))
    for i, (na, pa, nb, pb) in enumerate(pairs):
        exact = evaluate(g, layers, pa, pb)
        sim = simulate(g, pa, pb, 4000, seed=200 + i)
        mc.append({"A": na, "B": nb, "exact": exact, "mc_mean": sim.mean, "mc_stderr": sim.stderr, "z": (sim.mean - exact) / sim.stderr})
    out["monte_carlo"] = mc
    g2, l2, s2, _ = solve(GameConfig.v2(horizon=32))
    out["v2_reproduction"] = {"nodes": s2.stats.states, "mixed": s2.stats.mixed_states, "distinct": len({s for l in l2 for s in l}),
                              "exploitability": {b.name: exploitability(g2, l2, b, 0, s2.value).exploitability for b in all_baselines()}}
    g1, l1, s1, _ = solve(GameConfig.v1(horizon=32))
    out["v1_reproduction"] = {"nodes": s1.stats.states, "mixed": s1.stats.mixed_states,
                              "exploitability": {b.name: exploitability(g1, l1, b, 0, s1.value).exploitability for b in v1_baselines()}}
    # accounting identities at K = 24
    c24 = GameConfig(horizon=24)
    g24 = LaneGame(c24)
    l24 = g24.reachable(max_states=1_500_000)
    n = 0
    bad = {"hp": 0, "payoff": 0, "plates": 0, "kills": 0}
    for k, layer in enumerate(l24):
        for s in layer:
            for a in g24.legal(s, 0):
                for b in g24.legal(s, 1):
                    n += 1
                    r = step_turn(s, k, a, b, c24)
                    tr = g24.transition(s, k, a, b)
                    before, after = total_hp(s, c24), total_hp(r.state, c24)
                    for side in (0, 1):
                        if before[side] + r.events["spawned_hp"][side] != after[side] + r.events["killed_hp"][side] + r.events["tower_hp"][side]:
                            bad["hp"] += 1
                        if sum(r.kills[side].values()) != r.events["killed_hp"][1 - side] + r.events["tower_hp"][1 - side]:
                            bad["kills"] += 1
                    if abs(tr.payoff - (tr.farm_payoff + tr.plate_payoff + tr.gank_payoff)) > 1e-15:
                        bad["payoff"] += 1
                    if tr.plate_payoff != c24.plate_value * (r.plates[0] - r.plates[1]):
                        bad["plates"] += 1
    out["accounting"] = {"horizon": 24, "transitions": n, "violations": bad}

    sv = out["stage_games_vs_support_enumeration"]
    md = ["# Verification (siege model)", "",
          f"Stage games of the K = {K} solution re-solved by support enumeration: {sv['games']} games, max |value difference| = "
          f"{sv['max_abs_diff']:.2e} ({sv['seconds']:.0f} s). Solution methods: {sv['methods']}; crash-choice games: "
          f"{sv['choice_games']} (none exist in the siege model); max certified duality gap {sv['max_gap']:.1e}.", "",
          "## Backward induction vs. tree recursion (support enumeration at every node)", "",
          md_table(["root", "turns", "backward induction", "tree minimax", "|diff|", "seconds"],
                   [(r["root"], r["depth"], round(r["backward_induction"], 10), round(r["tree_minimax"], 10), f"{r['abs_diff']:.1e}",
                     round(r["seconds"], 1)) for r in tree]), "",
          "## Exploitability of the computed equilibrium and value antisymmetry", "",
          md_table(["K", "nodes", "Expl_A", "Expl_B", "max |V(s) + V(mirror s)|", "max certified gap"],
                   [(e["K"], e["nodes"], f"{e['expl_A']:.1e}", f"{e['expl_B']:.1e}", f"{e['max_antisymmetry']:.1e}", f"{e['max_gap']:.1e}")
                    for e in expl]), "",
          f"## Monte Carlo (4000 games, sampled ganks and mixed actions) vs. exact expectation, K = {K}", "",
          md_table(["A", "B", "exact", "MC mean", "stderr", "z"],
                   [(m["A"], m["B"], round(m["exact"], 4), round(m["mc_mean"], 4), round(m["mc_stderr"], 4), round(m["z"], 2)) for m in mc]), "",
          f"## Accounting identities over every reachable transition (K = 24, {n} transitions)", "",
          md_table(["identity", "violations"], [("HP conservation (spawned = alive + killed + tower + thinned)", bad["hp"]),
                                                ("kills recorded = enemy HP removed", bad["kills"]),
                                                ("stage payoff = farm + plates + gank", bad["payoff"]),
                                                ("plate payoff = plate_value x plates claimed", bad["plates"])]), "",
          "## Earlier models reproduced", "",
          f"- GameConfig.v2() (previous revision), K = 32: {out['v2_reproduction']['nodes']} nodes, {out['v2_reproduction']['mixed']} mixed, "
          f"{out['v2_reproduction']['distinct']} distinct states; exploitability "
          + ", ".join(f"{k} {v:.3f}" for k, v in out["v2_reproduction"]["exploitability"].items())
          + " (revision report: 14849 / 1030 / 2480; 10.08, 24.86, 9.45, 9.42, 5.78, 7.69, 6.59).",
          f"- GameConfig.v1() (original), K = 32: {out['v1_reproduction']['nodes']} nodes, {out['v1_reproduction']['mixed']} mixed; exploitability "
          + ", ".join(f"{k} {v:.4f}" for k, v in out["v1_reproduction"]["exploitability"].items())
          + " (original report: 905 / 52; 11.86 / 19.45 / 4.85 / 1.85 / 4.85)."]
    save("verification", out, "\n".join(md))
    return out


if __name__ == "__main__":
    run()
