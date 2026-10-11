"""Baseline policies: exact values, exploitability, best-response behaviour and payoff distributions.

This is the Phase-2 seed experiment: how much can a best-responding opponent
extract from each predictable policy, what does the exploiting policy look
like, and what is the realized (sampled) distribution of payoffs?
"""

from __future__ import annotations

import numpy as np

from ... import GameConfig, LaneGame
from ...analysis import path_statistics
from ...baselines import v1_baselines as all_baselines
from ...best_response import best_response_value, evaluate, exploitability
from ...simulate import simulate
from ...solver import solve_equilibrium
from ..common import md_table, save


def run(horizon: int = 32, episodes: int = 5000, seed: int = 7) -> dict:
    g = LaneGame(GameConfig.v1(horizon=horizon))
    layers = g.reachable()
    sol = solve_equilibrium(g, layers)
    eqA, eqB = sol.policy(0), sol.policy(1)
    pols = all_baselines()
    names = ["equilibrium"] + [p.name for p in pols]
    A_pols = [eqA] + pols
    B_pols = [eqB] + pols
    H = np.array([[evaluate(g, layers, pa, pb) for pb in B_pols] for pa in A_pols])

    per = {}
    for p in pols:
        ex = exploitability(g, layers, p, 0, sol.value)
        br = best_response_value(g, layers, p, 0)          # B best-responds to baseline A
        ps = path_statistics(g, p, br.policy)
        sim_br = simulate(g, p, br.policy, episodes, seed)
        sim_eq = simulate(g, p, eqB, episodes, seed + 1)
        per[p.name] = {
            "exploitability": ex.exploitability,
            "value_vs_equilibrium": evaluate(g, layers, p, eqB),
            "value_vs_best_response": ex.worst_case_value,
            "br_path": ps.to_dict(horizon),
            "sim_vs_best_response": sim_br.summary(),
            "sim_vs_equilibrium": sim_eq.summary(),
        }
    sim_eq_eq = simulate(g, eqA, eqB, episodes, seed + 2).summary()

    md = [f"# Baselines (horizon {horizon})", "",
          f"Equilibrium value V*(s0) = {sol.value:.3g}. Entries are exact expected payoffs to the row player (A).", "",
          "## Head-to-head (exact)", "",
          md_table(["A \\ B"] + names, [(n, *[round(v, 3) for v in row]) for n, row in zip(names, H)]), "",
          "## Exploitability and exploitation", "",
          md_table(["baseline (as A)", "exploitability", "vs equilibrium B", "B's best-response actions",
                    "B crashes vs it (S/M/L)", "A crashes (S/M/L)"],
                   [(n, round(d["exploitability"], 3), round(d["value_vs_equilibrium"], 3),
                     {k: round(v, 2) for k, v in d["br_path"]["action_freq"][1].items() if v > 1e-9},
                     [round(c, 2) for c in d["br_path"]["crashes_by_size"]["B"]],
                     [round(c, 2) for c in d["br_path"]["crashes_by_size"]["A"]]) for n, d in per.items()]), "",
          f"## Realized payoff distributions ({episodes} sampled games each, seed {seed})", "",
          md_table(["matchup", "mean", "stderr", "std", "q05", "q50", "q95", "ganks A/B"],
                   [("equilibrium vs equilibrium", round(sim_eq_eq["mean"], 3), round(sim_eq_eq["stderr"], 3),
                     round(sim_eq_eq["std"], 2), sim_eq_eq["quantiles"]["q05"], sim_eq_eq["quantiles"]["q50"],
                     sim_eq_eq["quantiles"]["q95"], [round(x, 2) for x in sim_eq_eq["mean_ganks"]])]
                   + [(f"{n} vs best response", round(d["sim_vs_best_response"]["mean"], 3),
                       round(d["sim_vs_best_response"]["stderr"], 3), round(d["sim_vs_best_response"]["std"], 2),
                       d["sim_vs_best_response"]["quantiles"]["q05"], d["sim_vs_best_response"]["quantiles"]["q50"],
                       d["sim_vs_best_response"]["quantiles"]["q95"],
                       [round(x, 2) for x in d["sim_vs_best_response"]["mean_ganks"]]) for n, d in per.items()])]
    save("baselines", {"horizon": horizon, "names": names, "head_to_head": H, "baselines": per,
                       "sim_equilibrium": sim_eq_eq}, "\n".join(md))
    return per


if __name__ == "__main__":
    run()
