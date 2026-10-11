"""Controlled comparison (study sections E and G): siege decisions x tower clearing speed.

Six configurations at K = 32: {original siege rewards (previous revision), new siege objectives} x
{1, 2, 3 HP/tick tower fire}. For each: state space and solver cost, the symmetric equilibrium path,
siege decisions at siege nodes, exploitability of the baselines with the farm / plates / crash / gank
decomposition, and the crash, siege, bounce, plate and thinning statistics along the best-response paths
against the crash-prone baselines (where sieges actually happen; the symmetric equilibrium path is
crash-free in every configuration).
"""

from __future__ import annotations

import time

from ...analysis import path_statistics
from ...baselines import all_baselines, siege_baselines
from ...best_response import best_response_value
from ...state import State
from .common import CONTROLLED, K, fmt_freq, md_table, probe_value, save, solve, state_counts, node_averages, \
    eq_path, exploit_rows

CRASH_PRONE = ["passive_safe", "stack_builder", "contest_slowpush", "freeze_contest"]


def siege_stats(ps: dict, side: int) -> dict:
    """Siege-related path statistics of one side."""
    return {"crashes": ps["crashes_by_size"]["AB"[side]], "crash_hp": ps["crash_hp"][side],
            "sieges_started": ps["sieges_started"][side], "siege_turns": ps["siege_turns"][side],
            "pressure_turns": ps["pressure_turns"][side], "vision_turns_set": ps["vision_turns_set"][side],
            "siege_safe_turns": ps["siege_safe_turns"][side], "plates": ps["plates"][side],
            "thinned_hp": ps["thinned_hp"][side], "thin_farm": ps["thin_farm"][side],
            "vision_choices": ps["vision_choices"][side], "protected_turns": ps["protected_turns"][side],
            "p_contest_protected": ps["p_contest_protected"][side],
            "p_contest_unprotected": ps["p_contest_unprotected"][side],
            "crash_resources": ps["crash_resources"][side],
            # the OTHER side's bounces are the consequence of this side's sieges
            "opp_bounces": ps["bounces"][1 - side], "opp_bounce_hp": ps["bounce_hp"][1 - side],
            "opp_tower_farm": None}


def run(horizon: int = K) -> dict:
    res = {}
    for name, (make, desc) in CONTROLLED.items():
        cfg = make(horizon)
        print(name, flush=True)
        t = time.perf_counter()
        g, layers, sol, timing = solve(cfg)
        pols = siege_baselines() if cfg.objectives else all_baselines()
        rows = exploit_rows(g, layers, sol, pols)
        d = {"description": desc, "config": cfg.to_dict(), **state_counts(layers), **timing,
             "mixed": sol.stats.mixed_states, "lps": sol.stats.methods.get("lp", 0), "max_gap": sol.stats.max_gap,
             "value": sol.value, "nodes_avg": node_averages(sol), "path": eq_path(g, sol),
             "baselines": {n: {k: v for k, v in r.items() if k != "br_path"} for n, r in rows.items()},
             "br_siege": {n: {"baseline": siege_stats(r["br_path"], 0), "best_response": siege_stats(r["br_path"], 1),
                              "br_actions": r["br_path"]["action_freq"][1]}
                          for n, r in rows.items() if n in CRASH_PRONE},
             }
        # value of a 12-HP crash state (A's mass under B's tower, B's wave waiting, 3 plate opportunities), 20 turns left
        probe = State(-4, 4, (), 12, -3, 4, (), 0, 0, 0, 3 if cfg.objectives else 0, 0)
        pv, _, _ = probe_value(cfg, probe, 12)
        d["probe_crash12"] = pv
        probe_v = probe._replace(va=6 if cfg.objectives else 2)
        pv2, _, _ = probe_value(cfg, probe_v, 12)
        d["probe_crash12_vision"] = pv2
        d["seconds_total"] = time.perf_counter() - t
        res[name] = d

    names = list(res)
    base_names = [b.name for b in siege_baselines()]
    md = [f"# Controlled comparison: siege decisions x tower clearing speed (K = {horizon})", "",
          md_table(["configuration", "description"], [(n, res[n]["description"]) for n in names]), "",
          "## State space and solver", "",
          md_table(["configuration", "nodes", "distinct", "max layer", "transitions", "enumerate s", "solve s",
                    "LPs", "mixed", "max gap"],
                   [(n, d["nodes"], d["distinct"], d["max_layer"], d["transitions"], round(d["enumerate_s"], 1),
                     round(d["solve_s"], 1), d["lps"], d["mixed"], f"{d['max_gap']:.1e}") for n, d in res.items()]), "",
          "## Symmetric equilibrium path (per player, whole game)", "",
          md_table(["configuration", "V*", "actions (A)", "crashes S/M/L", "farm", "plates", "gank loss",
                    "P(Contest)", "siege turns", "pressure / vision / disengage turns"],
                   [(n, round(d["value"], 6), fmt_freq(d["path"]["action_freq"][0]),
                     "/".join(f"{c:.2f}" for c in d["path"]["crashes_by_size"]["A"]),
                     round(d["path"]["farm"][0], 2), round(d["path"]["plates"][0], 2), round(d["path"]["gank_loss"][0], 2),
                     round(d["path"]["contest_share"][0], 2), round(d["path"]["siege_turns"][0], 2),
                     f"{d['path']['pressure_turns'][0]:.2f} / {d['path']['vision_turns_set'][0]:.2f} / "
                     f"{d['path']['siege_safe_turns'][0]:.2f}") for n, d in res.items()]), "",
          "## Equilibrium siege decisions (A, averaged over reachable siege nodes)", "",
          md_table(["configuration", "siege nodes", "P(Apply Pressure)", "P(Establish Vision)", "P(disengage)",
                    "P(Contest) all nodes"],
                   [(n, d["nodes_avg"]["siege_nodes"], round(d["nodes_avg"]["p_pressure"], 3),
                     round(d["nodes_avg"]["p_vision"], 3), round(d["nodes_avg"]["p_disengage"], 3),
                     round(d["nodes_avg"]["contest"], 3)) for n, d in res.items()]), "",
          "## Value of a 12-HP crash state to the crasher (A's mass under B's tower, B's wave waiting behind it, "
          "20 turns left; siege model: 3 plate opportunities pending)", "",
          md_table(["configuration", "V (no vision)", "A plays", "V (with vision)", "A plays (vision)"],
                   [(n, round(d["probe_crash12"]["value"], 3), d["probe_crash12"]["A"],
                     round(d["probe_crash12_vision"]["value"], 3), d["probe_crash12_vision"]["A"]) for n, d in res.items()]), "",
          "## Exploitability of baselines (as A) [farm / plates / crash / gank parts of the value against the best response]", ""]
    for n, d in res.items():
        md += [f"**{n}**", "",
               md_table(["baseline", "exploitability", "vs equilibrium", "farm", "plates", "crash", "gank"],
                        [(b, round(r["exploitability"], 2), round(r["value_vs_equilibrium"], 2), round(r["farm"], 2),
                          round(r["plates"], 2), round(r["crash"], 2), round(r["gank"], 2)) for b, r in d["baselines"].items()]), ""]
    md += ["## Sieges along best-response paths against crash-prone baselines (per game; 'BR' = the exploiting player B)", "",
           "crashes S/M/L and siege turns of the best responder; bounces = stacked waves released by the baseline "
           "after the best responder's sieges clear; plates and thinning earned by the best responder.", ""]
    for n, d in res.items():
        rows = []
        for b, st in d["br_siege"].items():
            br = st["best_response"]
            rows.append((b, "/".join(f"{c:.2f}" for c in br["crashes"]), round(br["crash_hp"], 2),
                         round(br["siege_turns"], 2), round(br["pressure_turns"], 2), round(br["vision_turns_set"], 2),
                         round(br["siege_safe_turns"], 2), round(br["plates"], 2), round(br["thinned_hp"], 2),
                         round(br["crash_resources"], 2), round(br["opp_bounces"], 2), round(br["opp_bounce_hp"], 2),
                         round(br["protected_turns"], 2), fmt_freq(st["br_actions"], 0.02)))
        md += [f"**{n}**", "",
               md_table(["baseline (A)", "BR crashes S/M/L", "BR crash HP", "BR siege turns", "BR pressure", "BR vision",
                         "BR disengage", "BR plates", "BR thinned HP", "BR crash resources", "baseline bounces",
                         "bounce HP", "BR protected turns", "BR actions"], rows), ""]
    save("controlled", {"horizon": horizon, "results": res}, "\n".join(md))
    return res


if __name__ == "__main__":
    run()
