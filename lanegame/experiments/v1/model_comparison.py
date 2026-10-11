"""Compare candidate pressure/collision formulations.

Every variant uses the same engine; only the contact-resolution law, the Slow
Push movement rule or the champion-posture rule changes. For each variant we
record: qualitative scenario checks, state-space size, equilibrium structure
on the symmetric game, dominance diagnostics, values of a few probe states
(which expose 'trap' artifacts) and baseline exploitability.
"""

from __future__ import annotations

import numpy as np

from ... import GameConfig, LaneGame
from ...analysis import path_statistics
from ...baselines import v1_baselines as all_baselines
from ...best_response import exploitability
from ...scenarios import all_scenarios, run_scenario
from ...solver import solve_equilibrium
from ...state import State
from ..common import md_table, save

ACTION_SHORT = {0: "SP", 1: "SH", 2: "F"}   # original action labels

VARIANTS = {
    "default (lanchester floor, SP hold)": {},
    "lanchester_ceil": {"damage_law": "lanchester_ceil"},
    "linear attrition": {"damage_law": "linear"},
    "pressure comparison": {"damage_law": "pressure"},
    "SP half speed": {"slowpush_mode": "half"},
    "SP creep": {"slowpush_mode": "creep"},
    "no tower damage": {"tower_damage": 0},
    "posture caps (SP lane, F -1)": {"posture_cap": (0, None, -1, 0, None, -1)},
}

PROBES = {
    "A surplus hp1 at lane": (State(0, 1, (-4,), 0, -4, 2, (), 0, 0, 0), 10),
    "A one wave hp2 at lane": (State(0, 2, (-4,), 0, -4, 2, (), 0, 0, 0), 10),
    "A 2-stack hp6 at lane": (State(0, 6, (-4,), 0, -4, 2, (), 0, 0, 0), 10),
    "B wave held at A freeze line": (State(0, 0, (-2,), 0, 1, 2, (), 0, 0, 0), 11),
}


def dominance(sol) -> dict:
    """Count states where one of A's actions weakly dominates all others in Q."""
    counts = {}
    for k, layer in enumerate(sol.layers):
        for s in layer:
            Q = sol.Q[k][s]
            la, _ = sol.legal[k][s]
            for i, a in enumerate(la):
                if all(np.all(Q[i] >= Q[j] - 1e-12) for j in range(len(la)) if j != i):
                    counts[ACTION_SHORT[a]] = counts.get(ACTION_SHORT[a], 0) + 1
                    break
    return counts


def eq_action_support(sol) -> dict:
    sup = {}
    for k, layer in enumerate(sol.layers):
        for s in layer:
            la, _ = sol.legal[k][s]
            for a, p in zip(la, sol.x[k][s]):
                if p > 1e-9:
                    sup[ACTION_SHORT[a]] = sup.get(ACTION_SHORT[a], 0) + 1
    return sup


def run(horizon: int = 32) -> dict:
    out = {}
    for name, kw in VARIANTS.items():
        cfg = GameConfig.v1(horizon=horizon, **kw)
        g = LaneGame(cfg)
        sol = solve_equilibrium(g)
        ps = path_statistics(g, sol.policy(0), sol.policy(1))
        scen = {}
        for sc in all_scenarios(cfg):
            ok, detail, _, _ = run_scenario(sc, cfg)
            scen[sc.name] = {"pass": bool(ok), "detail": detail}
        probes = {n: solve_equilibrium(g, root=s, k0=k).value for n, (s, k) in PROBES.items()}
        expl = {b.name: exploitability(g, sol.layers, b, 0, sol.value).exploitability
                for b in all_baselines()}
        out[name] = {
            "overrides": kw, "value": sol.value,
            "reachable_states": sol.stats.states, "mixed_states": sol.stats.mixed_states,
            "scenarios": scen,
            "scenarios_passed": sum(v["pass"] for v in scen.values()),
            "eq_path_crashes_A": ps.crashes[0], "eq_path_action_freq_A": {k: v / horizon for k, v in ps.action_freq[0].items()},
            "dominant_action_states_A": dominance(sol), "eq_support_counts_A": eq_action_support(sol),
            "probe_values": probes, "baseline_exploitability": expl,
        }
    n_sc = len(all_scenarios(GameConfig.v1()))
    rows = []
    for name, d in out.items():
        rows.append((name, f"{d['scenarios_passed']}/{n_sc}", d["reachable_states"], d["mixed_states"],
                     round(sum(d["eq_path_crashes_A"]), 3),
                     ", ".join(f"{k} {v:.2f}" for k, v in sorted(d["eq_path_action_freq_A"].items()) if v > 1e-9),
                     *[round(v, 2) for v in d["probe_values"].values()]))
    md = ["# Candidate collision models (horizon %d)" % horizon, "",
          md_table(["variant", "scenarios", "states", "mixed", "eq crashes", "eq path actions (A)",
                    *PROBES.keys()], rows), "", "## Failed scenario checks", ""]
    for name, d in out.items():
        fails = [f"`{k}` ({v['detail']})" for k, v in d["scenarios"].items() if not v["pass"]]
        md.append(f"- **{name}**: " + ("none" if not fails else "; ".join(fails)))
    md += ["", "## Baseline exploitability (as player A)", "",
           md_table(["variant"] + [b.name for b in all_baselines()],
                    [(n, *[round(v, 2) for v in d["baseline_exploitability"].values()]) for n, d in out.items()]),
           "", "## States where one of A's actions weakly dominates all others", "",
           md_table(["variant", "dominant-action states", "eq support counts"],
                    [(n, d["dominant_action_states_A"], d["eq_support_counts_A"]) for n, d in out.items()])]
    save("model_comparison", {"horizon": horizon, "variants": out}, "\n".join(md))
    return out


if __name__ == "__main__":
    run()
