"""Experiment F - mechanic ablations (K = 24, 1 HP/tick unless stated).

Each row isolates one change relative to the previous revision's economy, within the siege-objective
architecture where a siege objective is involved:

  0 previous revision (post-crash choice: resources or 2 turns of 50% vision)
  1 plates alone          objectives on, plates 1/2/3 per crash, no thinning, vision 2 turns / 50%
  2 thinning alone        objectives on, no plates, thinning 1 HP/turn, vision 2 turns / 50%
  3 Apply Pressure        objectives on, plates + thinning, vision 2 turns / 50%
  4 stronger vision alone objectives on, no plates, no thinning, vision 2 turns / 75%
  5 longer vision alone   objectives on, no plates, no thinning, vision 6 turns / 50%
  6 faster tower alone    previous revision with 2 HP/tick
  7 proposed system       plates + thinning + vision 6 turns / 75%, 1 HP/tick (the default)
  8 proposed + tower 2    the same at 2 HP/tick
  9 proposed, thin 2      the default with 2 HP/turn thinning (bounce fully suppressible)
 10 proposed, no plates   the default without plates (thinning + vision only)

Note that rows 1-5 differ from row 0 also in the architecture (siege objective chosen each siege turn,
requiring Contest; crashes pay nothing by themselves), so 'alone' means 'the only economic effect enabled
within the new architecture'.
"""

from __future__ import annotations

from ... import GameConfig
from ...analysis import path_statistics
from ...baselines import siege_baselines, all_baselines
from ...best_response import best_response_value
from ...state import State
from .common import K_SWEEP, fmt_freq, md_table, probe_value, save, solve, node_averages, eq_path, exploit_rows

TRACK = ["passive_safe", "stack_builder", "contest_slowpush", "aggressive", "release_safe"]


def ladder(K: int):
    old_vis = dict(vision_turns=2, vision_reduction=0.5)
    return {
        "0 previous revision": GameConfig.v2(horizon=K),
        "1 plates alone": GameConfig(horizon=K, thin_damage=0, **old_vis),
        "2 thinning alone": GameConfig(horizon=K, plate_grant=(0, 0, 0), **old_vis),
        "3 Apply Pressure (plates + thinning)": GameConfig(horizon=K, **old_vis),
        "4 stronger vision alone (75%)": GameConfig(horizon=K, plate_grant=(0, 0, 0), thin_damage=0, vision_turns=2,
                                                    vision_reduction=0.75),
        "5 longer vision alone (6 turns)": GameConfig(horizon=K, plate_grant=(0, 0, 0), thin_damage=0, vision_turns=6,
                                                      vision_reduction=0.5),
        "6 faster tower alone (2 HP/tick)": GameConfig.v2(horizon=K, tower_siege_damage=2),
        "7 proposed system (default)": GameConfig(horizon=K),
        "8 proposed + tower 2 HP/tick": GameConfig(horizon=K, tower_siege_damage=2),
        "9 proposed, thinning 2 HP/turn": GameConfig(horizon=K, thin_damage=2),
        "10 proposed, no plates": GameConfig(horizon=K, plate_grant=(0, 0, 0)),
    }


def run(horizon: int = K_SWEEP) -> dict:
    res = {}
    for name, cfg in ladder(horizon).items():
        print(name, flush=True)
        g, layers, sol, timing = solve(cfg)
        pols = [p for p in (siege_baselines() if cfg.objectives else all_baselines()) if p.name in TRACK]
        rows = exploit_rows(g, layers, sol, pols)
        probe = State(-4, 4, (), 12, -3, 4, (), 0, 0, 0, 3 if (cfg.objectives and max(cfg.plate_grant) > 0) else 0, 0)
        pv, _, _ = probe_value(cfg, probe, horizon - 16)
        pre = State(1, 12, (), 0, -4, 4, (), 0, 0, 0)
        pre_v, _, _ = probe_value(cfg, pre, horizon - 16)
        sh = max(q for lab, q in pre_v["Q_A"].items() if lab.startswith("SH"))
        hold = max(q for lab, q in pre_v["Q_A"].items() if not lab.startswith("SH"))
        res[name] = {"config": cfg.to_dict(), "nodes": sol.stats.states, "mixed": sol.stats.mixed_states,
                     "solve_s": timing["solve_s"], "enumerate_s": timing["enumerate_s"], "value": sol.value,
                     "node_avg": node_averages(sol), "path": eq_path(g, sol),
                     "baselines": {n: {k: v for k, v in r.items() if k != "br_path"} for n, r in rows.items()},
                     "br_paths": {n: {"bounces": r["br_path"]["bounces"], "bounce_hp": r["br_path"]["bounce_hp"],
                                      "crashes_B": r["br_path"]["crashes_by_size"]["B"], "plates": r["br_path"]["plates"],
                                      "thinned": r["br_path"]["thinned_hp"], "vision": r["br_path"]["vision_turns_set"],
                                      "pressure": r["br_path"]["pressure_turns"], "siege_turns": r["br_path"]["siege_turns"],
                                      "vision_choices": r["br_path"]["vision_choices"]} for n, r in rows.items()},
                     "probe_crash12": pv, "pre_crash_12": {"value": pre_v["value"], "A": pre_v["A"], "crash_gain": sh - hold}}
    md = [f"# Experiment F: mechanic ablations (K = {horizon})", "",
          "## State space, equilibrium path and the two probes", "",
          "probe 'crash 12' = value to A of its 12-HP mass under B's tower with B's wave waiting (16 turns left); "
          "'crash gain' = Q(best Shove) - Q(best hold) for A's 12-HP packet one cell from B's empty tower (16 turns left), "
          "against B's equilibrium strategy.", "",
          md_table(["ablation", "nodes", "mixed", "solve s", "eq actions (A)", "eq P(Contest)", "eq farm", "eq gank loss",
                    "P(Pressure) siege nodes", "P(Vision) siege nodes", "V(crash 12 state)", "crash gain (12 HP)"],
                   [(n, d["nodes"], d["mixed"], round(d["solve_s"], 1), fmt_freq(d["path"]["action_freq"][0]),
                     round(d["path"]["contest_share"][0], 2), round(d["path"]["farm"][0], 2), round(d["path"]["gank_loss"][0], 2),
                     round(d["node_avg"]["p_pressure"], 3) if d["node_avg"]["siege_nodes"] else "-",
                     round(d["node_avg"]["p_vision"], 3) if d["node_avg"]["siege_nodes"] else "-",
                     round(d["probe_crash12"]["value"], 2), round(d["pre_crash_12"]["crash_gain"], 2)) for n, d in res.items()]), "",
          "## Exploitability of tracked baselines (as A) [farm / plates / crash / gank parts]", "",
          md_table(["ablation"] + TRACK,
                   [(n, *[f"{d['baselines'][b]['exploitability']:.2f} [{d['baselines'][b]['farm']:.1f}/{d['baselines'][b]['plates']:.1f}/"
                          f"{d['baselines'][b]['crash']:.1f}/{d['baselines'][b]['gank']:.1f}]" for b in TRACK]) for n, d in res.items()]), "",
          "## Best-response paths against passive_safe and stack_builder (per game): the exploiter's sieges", "",
          md_table(["ablation", "vs", "BR crashes S/M/L", "BR siege turns", "BR pressure", "BR vision", "BR plates",
                    "BR thinned HP", "baseline bounces", "bounce HP", "BR crash-choice visions"],
                   [(n, b, "/".join(f"{c:.2f}" for c in d["br_paths"][b]["crashes_B"]), round(d["br_paths"][b]["siege_turns"][1], 2),
                     round(d["br_paths"][b]["pressure"][1], 2), round(d["br_paths"][b]["vision"][1], 2),
                     round(d["br_paths"][b]["plates"][1], 2), round(d["br_paths"][b]["thinned"][1], 2),
                     round(d["br_paths"][b]["bounces"][0], 2), round(d["br_paths"][b]["bounce_hp"][0], 2),
                     round(d["br_paths"][b]["vision_choices"][1], 2))
                    for n, d in res.items() for b in ("passive_safe", "stack_builder")])]
    save("ablations", {"horizon": horizon, "results": res}, "\n".join(md))
    return res


if __name__ == "__main__":
    run()
