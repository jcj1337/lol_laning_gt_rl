"""Experiment H - sensitivity of the siege-study conclusions (K = 24, default model unless stated).

One-at-a-time changes of the parameters the new mechanics interact with most:

  gank_base      0.05, 0.10, 0.15 (default), 0.20   the position-independent threat; it cancels between the
                                                    players except through vision, so it sets how much of
                                                    vision's value is unconditional
  gank_loss      3.5, 5 (default), 7
  plate_value    0.5, 1.0 (default), 1.5
  plate_grant    (1,1,1), (1,2,3) (default), (2,3,4)
  thin_damage    0, 1 (default), 2
  farm_per_hp    0.125, 0.25 (default), 0.5
  vision         (0.5, 2) old, (0.75, 6) default

For each setting: equilibrium path summary, siege decisions at siege nodes, the value of the 12-HP crash
state and the crash gain of a 12-HP packet one cell from the tower, the post-crash Vision-minus-Pressure
difference, the exploitability of the tracked baselines, and the qualitative checks

  S1  the symmetric equilibrium path never crashes
  S2  vision is the modal equilibrium objective at siege nodes without active vision
  S3  the 12-HP crash state has positive value for the crasher
  S4  producing a 12-HP crash (Shove) beats holding, against B's equilibrium strategy
  S5  aggressive (Shove + Contest) is the most exploitable tracked baseline
  S6  thinning happens on the best-response path against passive_safe
"""

from __future__ import annotations

from ... import GameConfig
from ...baselines import siege_baselines
from ...state import State
from .common import K_SWEEP, fmt_freq, md_table, probe_value, save, solve, node_averages, eq_path, exploit_rows, \
    siege_node_table

TRACK = ["aggressive", "passive_safe", "stack_builder", "contest_slowpush", "release_safe"]
SETTINGS = {
    "default": {},
    "gank_base 0.05": {"gank_base": 0.05},
    "gank_base 0.10": {"gank_base": 0.10},
    "gank_base 0.20": {"gank_base": 0.20},
    "gank_loss 3.5": {"gank_loss": 3.5},
    "gank_loss 7": {"gank_loss": 7.0},
    "plate_value 0.5": {"plate_value": 0.5},
    "plate_value 1.5": {"plate_value": 1.5},
    "plate_grant 1/1/1": {"plate_grant": (1, 1, 1)},
    "plate_grant 2/3/4": {"plate_grant": (2, 3, 4)},
    "thin_damage 0": {"thin_damage": 0},
    "thin_damage 2": {"thin_damage": 2},
    "farm_per_hp 0.125": {"farm_per_hp": 0.125},
    "farm_per_hp 0.5": {"farm_per_hp": 0.5},
    "vision 50% / 2 turns (old)": {"vision_reduction": 0.5, "vision_turns": 2},
    "tower 2 HP/tick": {"tower_siege_damage": 2},
}


def evaluate(name: str, kw: dict, horizon: int) -> dict:
    cfg = GameConfig(horizon=horizon, **kw)
    g, layers, sol, timing = solve(cfg)
    na = node_averages(sol)
    tab = siege_node_table(g, sol, horizon - 4)
    none = tab["vision"].get("none", {})
    pols = [p for p in siege_baselines() if p.name in TRACK]
    rows = exploit_rows(g, layers, sol, pols)
    path = eq_path(g, sol)
    plates = cfg.plate_grant[2]
    pv, _, _ = probe_value(cfg, State(-4, 4, (), 12, -3, 4, (), 0, 0, 0, plates, 0), horizon - 16)
    post, _, _ = probe_value(cfg, State(-4, 4, (), 11, -3, 4, (), 0, 0, 0, plates, 0), horizon - 16)
    pre, _, _ = probe_value(cfg, State(1, 12, (), 0, -4, 4, (), 0, 0, 0), horizon - 16)
    sh = max(q for lab, q in pre["Q_A"].items() if lab.startswith("SH"))
    hold = max(q for lab, q in pre["Q_A"].items() if not lab.startswith("SH"))
    best = {}
    for kind, pred in (("pressure", lambda lab: "+P" in lab), ("vision", lambda lab: "+V" in lab), ("safe", lambda lab: "-S" in lab)):
        cand = {lab: q for lab, q in post["Q_A"].items() if pred(lab)}
        best[kind] = max(cand.values()) if cand else None
    expl = {n: r["exploitability"] for n, r in rows.items()}
    checks = {
        "S1": bool(sum(path["crashes_by_size"]["A"]) < 1e-9),
        "S2": bool(none.get("p_vision", 0) > max(none.get("p_pressure", 0), none.get("p_disengage", 0))),
        "S3": bool(pv["value"] > 0),
        "S4": bool(sh - hold > 0),
        "S5": bool(max(expl, key=expl.get) == "aggressive"),
        "S6": bool(rows["passive_safe"]["br_path"]["thinned_hp"][1] > 1e-9),
    }
    return {"params": kw, "nodes": sol.stats.states, "solve_s": timing["solve_s"], "value": sol.value,
            "eq_actions": fmt_freq(path["action_freq"][0]), "eq_contest": path["contest_share"][0],
            "eq_crashes": float(sum(path["crashes_by_size"]["A"])), "eq_farm": path["farm"][0], "eq_gank": path["gank_loss"][0],
            "siege": {"p_pressure": na["p_pressure"], "p_vision": na["p_vision"], "p_disengage": na["p_disengage"],
                      "p_vision_no_vision": none.get("p_vision"), "p_pressure_no_vision": none.get("p_pressure"),
                      "p_disengage_no_vision": none.get("p_disengage"), "nodes": na["siege_nodes"]},
            "crash12_value": pv["value"], "crash12_A": pv["A"], "crash_gain_12": sh - hold,
            "post_best": best, "exploitability": expl,
            "br_passive": {k: rows["passive_safe"]["br_path"][k][1] for k in ("plates", "thinned_hp", "vision_turns_set", "pressure_turns")},
            "passive_bounces": rows["passive_safe"]["br_path"]["bounces"][0], "checks": checks}


def run(horizon: int = K_SWEEP) -> dict:
    res = {}
    for name, kw in SETTINGS.items():
        print(name, flush=True)
        res[name] = evaluate(name, kw, horizon)
    tally = {c: sum(r["checks"][c] for r in res.values()) for c in res["default"]["checks"]}
    md = [f"# Experiment H: sensitivity of the siege-study conclusions (K = {horizon})", "",
          "Every row re-solves the game exactly. Checks (Y = holds):",
          "- S1 the symmetric equilibrium path never crashes",
          "- S2 Establish Vision is the modal equilibrium objective at siege nodes without active vision",
          "- S3 the 12-HP crash state (3 plates, B's wave waiting, 16 turns left) has positive value for the crasher",
          "- S4 producing a 12-HP crash now beats holding the packet, against B's equilibrium strategy",
          "- S5 aggressive is the most exploitable tracked baseline",
          "- S6 the best response to passive_safe thins minions", "",
          f"Across {len(res)} settings: " + ", ".join(f"{c} holds in {n}" for c, n in tally.items()), "",
          md_table(["setting", "nodes", "solve s", "eq actions (A)", "eq P(C)", "eq farm", "eq gank loss",
                    "P(Pressure) / P(Vision) / P(Safe) at siege nodes", "P(Vision) no-vision nodes", "V(crash 12)",
                    "crash gain 12", "post-crash V - P", *[f"expl {n}" for n in TRACK], "BR vs passive: plates / thinned / vision",
                    "S1-S6"],
                   [(n, r["nodes"], round(r["solve_s"], 1), r["eq_actions"], round(r["eq_contest"], 2), round(r["eq_farm"], 2),
                     round(r["eq_gank"], 2),
                     f"{r['siege']['p_pressure']:.2f} / {r['siege']['p_vision']:.2f} / {r['siege']['p_disengage']:.2f}",
                     round(r["siege"]["p_vision_no_vision"], 3) if r["siege"]["p_vision_no_vision"] is not None else "-",
                     round(r["crash12_value"], 2), round(r["crash_gain_12"], 2),
                     round(r["post_best"]["vision"] - r["post_best"]["pressure"], 2) if r["post_best"]["vision"] is not None and r["post_best"]["pressure"] is not None else "-",
                     *[round(r["exploitability"][b], 1) for b in TRACK],
                     f"{r['br_passive']['plates']:.1f} / {r['br_passive']['thinned_hp']:.1f} / {r['br_passive']['vision_turns_set']:.1f}",
                     "".join("Y" if r["checks"][c] else "n" for c in sorted(r["checks"]))) for n, r in res.items()])]
    save("sensitivity", {"horizon": horizon, "results": res, "tally": tally}, "\n".join(md))
    return res


if __name__ == "__main__":
    run()
