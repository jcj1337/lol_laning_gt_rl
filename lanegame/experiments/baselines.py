"""Experiment F - baseline policies over combined actions: exact values, exploitability, best responses,
and realized payoff distributions.

These are the starting points for Phase 2 opponent types: how much can a best-responding opponent
extract from each predictable policy, where does that value come from (farm, crashes, ganks), what does
the exploiting policy look like, how much does the equilibrium itself earn against each baseline, and how
noisy are realized outcomes?
"""

from __future__ import annotations

import numpy as np

from .. import GameConfig
from ..analysis import path_statistics
from ..baselines import all_baselines
from ..best_response import best_response_value, evaluate
from ..simulate import simulate
from .common import REVISION, fmt_freq, md_table, save, solve

K = 32


def run(horizon: int = K, episodes: int = 5000, seed: int = 7) -> dict:
    cfg = GameConfig.v2(horizon=horizon)
    g, layers, sol, _ = solve(cfg)
    eqA, eqB = sol.policy(0), sol.policy(1)
    pols = all_baselines()
    names = ["equilibrium"] + [p.name for p in pols]
    A_pols = [eqA] + pols
    B_pols = [eqB] + pols
    H = np.array([[evaluate(g, layers, pa, pb) for pb in B_pols] for pa in A_pols])

    per = {}
    for p in pols:
        br = best_response_value(g, layers, p, 0)          # B best-responds to baseline A
        w = br.value(g)
        bp = path_statistics(g, p, br.policy).to_dict(horizon)
        ep = path_statistics(g, p, eqB).to_dict(horizon)
        sim_br = simulate(g, p, br.policy, episodes, seed)
        per[p.name] = {
            "exploitability": sol.value - w,
            "value_vs_equilibrium": evaluate(g, layers, p, eqB),
            "value_vs_best_response": w,
            "br_path": bp, "eq_path": ep,
            "sim_vs_best_response": sim_br.summary(),
        }
    sim_eq = simulate(g, eqA, eqB, episodes, seed + 1).summary()
    eq_path = path_statistics(g, eqA, eqB).to_dict(horizon)

    def parts(ps):
        return (ps["farm"][0] - ps["farm"][1], ps["crash_resources"][0] - ps["crash_resources"][1],
                -(ps["gank_loss"][0] - ps["gank_loss"][1]))

    md = [f"# Experiment F: baseline policies (K = {horizon})", "",
          f"Equilibrium value V*(s0) = {sol.value:.3g}. Baselines take the resource option after a crash "
          "(uniform randomizes it).", "",
          "- **uniform**: uniform over legal combined actions.",
          "- **aggressive**: Shove + Contest every turn.",
          "- **freeze_contest**: Freeze + Contest whenever legal, otherwise Slow Push + Safe.",
          "- **passive_safe**: Slow Push + Safe every turn.",
          "- **stack_builder**: Slow Push + Contest until the leading wave has 5+ HP, then Shove + Contest.",
          "- **release_safe**: Shove + Safe every turn (lets waves walk in, never follows them).",
          "- **contest_slowpush**: Slow Push + Contest every turn (the original always_slowpush).", "",
          "## Head-to-head (exact expected payoff to the row player A)", "",
          md_table(["A \\ B"] + names, [(n, *[round(v, 2) for v in row]) for n, row in zip(names, H)]), "",
          "## Exploitability and where the exploiter's gain comes from", "",
          "Decomposition = the baseline's (A minus B) farm, crash resources and gank payoff along the path "
          "against its best response; they sum to the value against the best response (= -exploitability).", "",
          md_table(["baseline (as A)", "exploitability", "per turn", "farm", "crash", "gank", "vs equilibrium B",
                    "B's best response plays"],
                   [(n, round(d["exploitability"], 2), round(d["exploitability"] / horizon, 3),
                     *[round(x, 2) for x in parts(d["br_path"])], round(d["value_vs_equilibrium"], 2),
                     fmt_freq(d["br_path"]["action_freq"][1], 0.02)) for n, d in per.items()]), "",
          "## Equilibrium vs. best response: how much of the available exploitation does the equilibrium capture?", "",
          "'captured' = (value the equilibrium earns against the baseline) / exploitability. The equilibrium "
          "guarantees V* against everything but does not seek out mistakes; the gap is what an opponent model "
          "could add (the Phase 2 question).", "",
          md_table(["baseline", "exploitability", "equilibrium earns", "captured"],
                   [(n, round(d["exploitability"], 2), round(-d["value_vs_equilibrium"], 2),
                     f"{-d['value_vs_equilibrium'] / d['exploitability']:.0%}" if d["exploitability"] > 1e-9 else "-")
                    for n, d in per.items()]), "",
          "## Crash, farm and positioning behaviour along the best-response path (per game)", "",
          md_table(["baseline", "baseline crashes S/M/L", "best response crashes S/M/L", "baseline farm (avail)",
                    "best response farm (avail)", "baseline P(Contest)", "best response P(Contest)"],
                   [(n, "/".join(f"{c:.1f}" for c in d["br_path"]["crashes_by_size"]["A"]),
                     "/".join(f"{c:.1f}" for c in d["br_path"]["crashes_by_size"]["B"]),
                     f"{d['br_path']['farm'][0]:.1f} ({d['br_path']['farm_available'][0]:.1f})",
                     f"{d['br_path']['farm'][1]:.1f} ({d['br_path']['farm_available'][1]:.1f})",
                     round(d["br_path"]["contest_share"][0], 2), round(d["br_path"]["contest_share"][1], 2))
                    for n, d in per.items()]), "",
          f"## Realized payoff distributions ({episodes} sampled games each, seed {seed}; farm is an expected "
          "reward, ganks are sampled)", "",
          md_table(["matchup", "mean", "stderr", "std", "q05", "q50", "q95", "ganks A/B"],
                   [("equilibrium vs equilibrium", round(sim_eq["mean"], 3), round(sim_eq["stderr"], 3),
                     round(sim_eq["std"], 2), sim_eq["quantiles"]["q05"], sim_eq["quantiles"]["q50"],
                     sim_eq["quantiles"]["q95"], [round(x, 2) for x in sim_eq["mean_ganks"]])]
                   + [(f"{n} vs best response", round(d["sim_vs_best_response"]["mean"], 3),
                       round(d["sim_vs_best_response"]["stderr"], 3), round(d["sim_vs_best_response"]["std"], 2),
                       round(d["sim_vs_best_response"]["quantiles"]["q05"], 2),
                       round(d["sim_vs_best_response"]["quantiles"]["q50"], 2),
                       round(d["sim_vs_best_response"]["quantiles"]["q95"], 2),
                       [round(x, 2) for x in d["sim_vs_best_response"]["mean_ganks"]]) for n, d in per.items()]), "",
          f"Equilibrium path per player: farm {eq_path['farm'][0]:.2f} of {eq_path['farm_available'][0]:.2f} "
          f"available, gank loss {eq_path['gank_loss'][0]:.2f}, no crashes; P(Contest) {eq_path['contest_share'][0]:.2f}."]
    save("baselines", {"horizon": horizon, "names": names, "head_to_head": H, "baselines": per,
                       "sim_equilibrium": sim_eq, "eq_path": eq_path}, "\n".join(md), REVISION)
    return per


if __name__ == "__main__":
    run()
