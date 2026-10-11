"""Experiment B - vision effectiveness (50 / 75 / 90 %) x duration (4 / 6 / 8 turns), siege model, K = 24.

For each setting (exact re-solve): equilibrium frequency of Establish Vision at siege nodes, the value of
vision (best Vision minus best Safe / minus best Pressure at siege nodes), Contest probability while
protected vs. unprotected along best-response paths, pressure opportunities and pressure-under-vision
turns, proactive vision (vision established while the player's own wave is still being built), the
reachability of the Vision -> Shove -> Crash -> Pressure sequence, and whether vision is preferable to
immediate pressure in the representative post-crash state. Baseline exploitability is recorded too.
"""

from __future__ import annotations

import itertools

import numpy as np

from ... import GameConfig, LaneGame
from ...analysis import path_statistics
from ...baselines import siege_baselines
from ...best_response import best_response_value, exploitability
from ...config import PRESSURE, VISION, objective_of, posture_of
from ...reachability import find_vision_strategy
from ...state import State
from .common import K_SWEEP, md_table, probe_value, save, siege_node_table, solve, node_averages

REDUCTIONS = (0.5, 0.75, 0.9)
DURATIONS = (4, 6, 8)
TRACK = ["stack_builder", "contest_slowpush", "passive_safe", "stack_builder[vision]", "aggressive"]


def setting(rho: float, V: int, horizon: int) -> dict:
    cfg = GameConfig(horizon=horizon, vision_reduction=rho, vision_turns=V)
    g, layers, sol, timing = solve(cfg)
    na = node_averages(sol)
    tab = siege_node_table(g, sol, horizon - 4)
    vis_rows = tab["vision"]
    # value of vision at siege nodes without vision: mean (best Vision - best Safe) and (best Vision - best Pressure)
    none = vis_rows.get("none", {})
    pols = {p.name: p for p in siege_baselines()}
    paths = {}
    expl = {}
    for name in TRACK:
        pol = pols[name]
        br = best_response_value(g, layers, pol, 0)
        expl[name] = sol.value - br.value(g)
        ps = path_statistics(g, pol, br.policy).to_dict(horizon)
        paths[name] = {"baseline": {k: ps[k][0] for k in ("vision_turns_set", "pressure_turns", "siege_turns", "protected_turns",
                                                        "p_contest_protected", "p_contest_unprotected", "plates", "thinned_hp")},
                       "best_response": {k: ps[k][1] for k in ("vision_turns_set", "pressure_turns", "siege_turns", "protected_turns",
                                                             "p_contest_protected", "p_contest_unprotected", "plates", "thinned_hp")}}
    # proactive vision: among A's equilibrium Vision choices at siege nodes, share with A's own next wave not yet
    # at the lane (building) vs. already pushing
    proactive = [0.0, 0.0]
    pressure_under_vision = 0
    siege_nodes_with_vision = 0
    for k, layer in enumerate(sol.layers):
        for s in layer:
            if s.asg == 0:
                continue
            la, _ = sol.legal[k][s]
            x = sol.x[k][s]
            pv = sum(p for a, p in zip(la, x) if objective_of(a) == VISION)
            proactive[0 if s.ax <= 0 else 1] += pv
            if s.va > 0:
                siege_nodes_with_vision += 1
                pressure_under_vision += sum(p for a, p in zip(la, x) if objective_of(a) == PRESSURE)
    w = find_vision_strategy(g, horizon=horizon, min_crash_hp=4, max_states=1_500_000)
    w8 = find_vision_strategy(g, horizon=horizon, min_crash_hp=8, max_states=1_500_000)
    st = State(-4, 4, (), 11, -3, 4, (), 0, 0, 0, 3, 0)
    pv, _, _ = probe_value(cfg, st, horizon - 12)
    best = {}
    for kind, pred in (("pressure", lambda lab: "+P" in lab), ("vision", lambda lab: "+V" in lab), ("safe", lambda lab: "-S" in lab)):
        cand = {lab: q for lab, q in pv["Q_A"].items() if pred(lab)}
        best[kind] = max(cand.values()) if cand else None
    return {"rho": rho, "V": V, "nodes": sol.stats.states, "mixed": sol.stats.mixed_states, "solve_s": timing["solve_s"],
            "enumerate_s": timing["enumerate_s"], "value": sol.value, "node_avg": na,
            "vision_value_vs_safe": none.get("vision_minus_safe"), "vision_value_vs_pressure": -none.get("pressure_minus_vision", float("nan")),
            "siege_nodes_no_vision": none.get("nodes", 0), "p_vision_no_vision_nodes": none.get("p_vision"),
            "p_vision_with_vision_nodes": vis_rows.get("active", {}).get("p_vision"),
            "p_pressure_under_vision": pressure_under_vision / siege_nodes_with_vision if siege_nodes_with_vision else None,
            "proactive_vision_share": proactive[0] / (proactive[0] + proactive[1]) if sum(proactive) > 0 else None,
            "paths": paths, "exploitability": expl,
            "strategy_full_wave": {"found": w.found, "turns_after_vision": w.turns_after_vision, "pressure_turn": w.k},
            "strategy_medium": {"found": w8.found, "turns_after_vision": w8.turns_after_vision, "pressure_turn": w8.k},
            "post_crash_probe": {"value": pv["value"], "A": pv["A"], "best": best}}


def run(horizon: int = K_SWEEP) -> dict:
    res = {}
    for rho, V in itertools.product(REDUCTIONS, DURATIONS):
        print("vision", rho, V, flush=True)
        res[f"rho {rho}, {V} turns"] = setting(rho, V, horizon)
    md = [f"# Experiment B: vision effectiveness x duration (siege model, K = {horizon})", "",
          "## Equilibrium use and value of vision", "",
          "'P(Vision) no-vision nodes' = equilibrium probability of Establish Vision at siege nodes where A has no active "
          "vision; 'value vs Safe' / 'vs Pressure' = mean Q(best Vision) - Q(best Safe) / - Q(best Pressure) at those nodes "
          "against B's equilibrium strategy; 'P(Pressure | vision)' = probability of Apply Pressure at siege nodes with "
          "active vision; 'proactive' = share of A's vision choices made while A's own next wave is still at or behind "
          "the lane (building) rather than already pushing.", "",
          md_table(["rho", "turns", "nodes", "mixed", "solve s", "P(Vision) no-vision nodes", "P(Vision) nodes with vision",
                    "value vs Safe", "value vs Pressure", "P(Pressure | vision)", "proactive", "P(Vision) all siege nodes",
                    "P(Pressure) all siege nodes", "P(disengage)"],
                   [(r["rho"], r["V"], r["nodes"], r["mixed"], round(r["solve_s"], 1),
                     round(r["p_vision_no_vision_nodes"], 3) if r["p_vision_no_vision_nodes"] is not None else "-",
                     round(r["p_vision_with_vision_nodes"], 3) if r["p_vision_with_vision_nodes"] is not None else "-",
                     round(r["vision_value_vs_safe"], 3) if r["vision_value_vs_safe"] is not None else "-",
                     round(r["vision_value_vs_pressure"], 3) if r["vision_value_vs_pressure"] == r["vision_value_vs_pressure"] else "-",
                     round(r["p_pressure_under_vision"], 3) if r["p_pressure_under_vision"] is not None else "-",
                     round(r["proactive_vision_share"], 3) if r["proactive_vision_share"] is not None else "-",
                     round(r["node_avg"]["p_vision"], 3), round(r["node_avg"]["p_pressure"], 3),
                     round(r["node_avg"]["p_disengage"], 3)) for r in res.values()]), "",
          "## The strategy and the representative post-crash choice", "",
          "Strategy columns: shortest Vision -> Shove -> Crash -> Pressure witness from s_0 (crash of at least a full "
          "wave / at least a medium wave after the vision); 'after vision' must be <= duration. Post-crash probe: "
          "A's mass 11 under B's tower, B's wave waiting, 3 plates, 12 turns left.", "",
          md_table(["rho", "turns", "strategy (full wave): found", "after vision", "strategy (medium): found", "after vision",
                    "probe V*", "probe A plays", "Q Pressure", "Q Vision", "Q Safe"],
                   [(r["rho"], r["V"], r["strategy_full_wave"]["found"], r["strategy_full_wave"]["turns_after_vision"],
                     r["strategy_medium"]["found"], r["strategy_medium"]["turns_after_vision"],
                     round(r["post_crash_probe"]["value"], 3), ", ".join(f"{k} {v}" for k, v in r["post_crash_probe"]["A"].items()),
                     round(r["post_crash_probe"]["best"]["pressure"], 3), round(r["post_crash_probe"]["best"]["vision"], 3),
                     round(r["post_crash_probe"]["best"]["safe"], 3)) for r in res.values()]), "",
          "## Best-response paths against tracked baselines (per game; BR = the exploiting player)", "",
          "P(Contest | protected) / P(Contest | unprotected) are the best responder's Contest shares in turns it starts "
          "with / without active vision.", ""]
    for name in TRACK:
        md += [f"**{name} (as A) vs. its best response**", "",
               md_table(["rho", "turns", "exploitability", "BR vision turns", "BR pressure turns", "BR siege turns",
                         "BR protected turns", "BR P(C|prot)", "BR P(C|unprot)", "BR plates", "BR thinned", "baseline vision",
                         "baseline pressure"],
                        [(r["rho"], r["V"], round(r["exploitability"][name], 2),
                          round(r["paths"][name]["best_response"]["vision_turns_set"], 2),
                          round(r["paths"][name]["best_response"]["pressure_turns"], 2),
                          round(r["paths"][name]["best_response"]["siege_turns"], 2),
                          round(r["paths"][name]["best_response"]["protected_turns"], 2),
                          (round(r["paths"][name]["best_response"]["p_contest_protected"], 2)
                           if r["paths"][name]["best_response"]["p_contest_protected"] is not None else "-"),
                          (round(r["paths"][name]["best_response"]["p_contest_unprotected"], 2)
                           if r["paths"][name]["best_response"]["p_contest_unprotected"] is not None else "-"),
                          round(r["paths"][name]["best_response"]["plates"], 2),
                          round(r["paths"][name]["best_response"]["thinned_hp"], 2),
                          round(r["paths"][name]["baseline"]["vision_turns_set"], 2),
                          round(r["paths"][name]["baseline"]["pressure_turns"], 2)) for r in res.values()]), ""]
    save("vision_sweep", {"horizon": horizon, "results": res}, "\n".join(md))
    return res


if __name__ == "__main__":
    run()
