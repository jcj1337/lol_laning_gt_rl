"""Experiment D - equilibrium, best responses and exploitation in the siege model (K = 32, default config).

Nash equilibrium (value, strategies at s_0, path statistics with the farm / plates / gank decomposition),
exploitability of every siege baseline with the decomposition of the best responder's gain, head-to-head
values among the baselines and the equilibrium, how much of each baseline's exploitability the equilibrium
captures, the best responder's siege behaviour against each baseline, and the realized-payoff noise.
"""

from __future__ import annotations

import numpy as np

from ... import GameConfig
from ...analysis import path_statistics
from ...baselines import siege_baselines
from ...best_response import best_response_value
from ...simulate import simulate
from .common import K, fmt_freq, md_table, save, solve, node_averages


def run(horizon: int = K, episodes: int = 4000, seed: int = 11) -> dict:
    cfg = GameConfig(horizon=horizon)
    g, layers, sol, timing = solve(cfg)
    eqA, eqB = sol.policy(0), sol.policy(1)
    pols = siege_baselines()
    names = ["equilibrium"] + [p.name for p in pols]
    # exact values by forward propagation (equal to backward evaluation; far cheaper on 173k nodes)
    H = np.array([[path_statistics(g, pa, pb).value for pb in [eqB] + pols] for pa in [eqA] + pols])
    print("head-to-head done", flush=True)
    per = {}
    for p in pols:
        br = best_response_value(g, layers, p, 0)
        w = br.value(g)
        bp = path_statistics(g, p, br.policy).to_dict(horizon)
        per[p.name] = {"exploitability": sol.value - w, "value_vs_equilibrium": path_statistics(g, p, eqB).value,
                       "value_vs_best_response": w, "br_path": bp,
                       "sim_vs_best_response": simulate(g, p, br.policy, episodes, seed).summary()}
        print(p.name, flush=True)
    eq_path = path_statistics(g, eqA, eqB).to_dict(horizon)
    sim_eq = simulate(g, eqA, eqB, episodes, seed + 1).summary()
    s0 = g.initial_state()
    la, lb = sol.legal[0][s0]
    out = {"horizon": horizon, "value": sol.value, "timing": timing, "nodes": sol.stats.states, "mixed": sol.stats.mixed_states,
           "max_gap": sol.stats.max_gap, "eq_s0": {cfg.label(a): float(p) for a, p in zip(la, sol.x[0][s0])},
           "eq_path": eq_path, "node_avg": node_averages(sol), "names": names, "head_to_head": H.tolist(),
           "baselines": per, "sim_equilibrium": sim_eq}

    def parts(ps):
        return (ps["farm"][0] - ps["farm"][1], ps["plates"][0] - ps["plates"][1], -(ps["gank_loss"][0] - ps["gank_loss"][1]))

    md = [f"# Experiment D: equilibrium and exploitation (siege model, K = {horizon})", "",
          f"V*(s0) = {sol.value:.3g}; {sol.stats.states} decision nodes, {sol.stats.mixed_states} mixed, max certified gap "
          f"{sol.stats.max_gap:.1e}; enumerate {timing['enumerate_s']:.0f} s, solve {timing['solve_s']:.0f} s.", "",
          f"Equilibrium at s0 (A): {out['eq_s0']}", "",
          f"Equilibrium path per player: actions {fmt_freq(eq_path['action_freq'][0])}; farm {eq_path['farm'][0]:.2f} of "
          f"{eq_path['farm_available'][0]:.2f}; plates {eq_path['plates'][0]:.2f}; gank loss {eq_path['gank_loss'][0]:.2f}; "
          f"crashes {eq_path['crashes_by_size']['A']}; P(Contest) {eq_path['contest_share'][0]:.2f}; siege turns "
          f"{eq_path['siege_turns'][0]:.2f}.", "",
          f"Siege nodes (A, all reachable): P(Pressure) {out['node_avg']['p_pressure']:.3f}, P(Vision) "
          f"{out['node_avg']['p_vision']:.3f}, P(disengage) {out['node_avg']['p_disengage']:.3f} over "
          f"{out['node_avg']['siege_nodes']} nodes.", "",
          "## Baselines", "",
          "- siege tendency in brackets: [vision] = Establish Vision when sieging, [safe] = disengage; none = Apply Pressure.", "",
          "## Head-to-head (exact expected payoff to the row player A)", "",
          md_table(["A \\ B"] + names, [(n, *[round(v, 2) for v in row]) for n, row in zip(names, H)]), "",
          "## Exploitability and the best responder's gain [farm / plates / gank parts of the baseline's value]", "",
          md_table(["baseline (as A)", "exploitability", "per turn", "farm", "plates", "gank", "vs equilibrium B", "captured",
                    "B's best response plays"],
                   [(n, round(d["exploitability"], 2), round(d["exploitability"] / horizon, 3), *[round(x, 2) for x in parts(d["br_path"])],
                     round(d["value_vs_equilibrium"], 2),
                     f"{-d['value_vs_equilibrium'] / d['exploitability']:.0%}" if d["exploitability"] > 1e-9 else "-",
                     fmt_freq(d["br_path"]["action_freq"][1], 0.02)) for n, d in per.items()]), "",
          "## Siege behaviour along the best-response paths (per game)", "",
          md_table(["baseline (A)", "A crashes S/M/L", "BR crashes S/M/L", "BR siege turns", "BR pressure", "BR vision",
                    "BR disengage", "BR plates", "BR thinned HP", "A bounces (HP)", "BR protected turns", "BR P(C|prot)",
                    "BR P(C|unprot)", "A pressure", "A vision"],
                   [(n, "/".join(f"{c:.1f}" for c in d["br_path"]["crashes_by_size"]["A"]),
                     "/".join(f"{c:.1f}" for c in d["br_path"]["crashes_by_size"]["B"]), round(d["br_path"]["siege_turns"][1], 2),
                     round(d["br_path"]["pressure_turns"][1], 2), round(d["br_path"]["vision_turns_set"][1], 2),
                     round(d["br_path"]["siege_safe_turns"][1], 2), round(d["br_path"]["plates"][1], 2),
                     round(d["br_path"]["thinned_hp"][1], 2),
                     f"{d['br_path']['bounces'][0]:.2f} ({d['br_path']['bounce_hp'][0]:.1f})",
                     round(d["br_path"]["protected_turns"][1], 2),
                     round(d["br_path"]["p_contest_protected"][1], 2) if d["br_path"]["p_contest_protected"][1] is not None else "-",
                     round(d["br_path"]["p_contest_unprotected"][1], 2) if d["br_path"]["p_contest_unprotected"][1] is not None else "-",
                     round(d["br_path"]["pressure_turns"][0], 2), round(d["br_path"]["vision_turns_set"][0], 2)) for n, d in per.items()]), "",
          f"## Realized payoff distributions ({episodes} sampled games each)", "",
          md_table(["matchup", "mean", "stderr", "std", "q05", "q50", "q95", "ganks A/B", "plates A/B"],
                   [("equilibrium vs equilibrium", round(sim_eq["mean"], 3), round(sim_eq["stderr"], 3), round(sim_eq["std"], 2),
                     sim_eq["quantiles"]["q05"], sim_eq["quantiles"]["q50"], sim_eq["quantiles"]["q95"],
                     [round(x, 2) for x in sim_eq["mean_ganks"]], [round(x, 2) for x in sim_eq.get("mean_plates", [0, 0])])]
                   + [(f"{n} vs best response", round(d["sim_vs_best_response"]["mean"], 3), round(d["sim_vs_best_response"]["stderr"], 3),
                       round(d["sim_vs_best_response"]["std"], 2), round(d["sim_vs_best_response"]["quantiles"]["q05"], 2),
                       round(d["sim_vs_best_response"]["quantiles"]["q50"], 2), round(d["sim_vs_best_response"]["quantiles"]["q95"], 2),
                       [round(x, 2) for x in d["sim_vs_best_response"]["mean_ganks"]],
                       [round(x, 2) for x in d["sim_vs_best_response"].get("mean_plates", [0, 0])]) for n, d in per.items()])]
    save("equilibrium", out, "\n".join(md))
    return out


if __name__ == "__main__":
    run()
