"""Experiment A - Apply Pressure vs. Establish Vision vs. disengage across siege states.

1. From the K = 32 solution of the default model: equilibrium siege decisions of A at every reachable siege
   node, grouped by siege mass, the defender's accessible wave, plates left, active vision, turns remaining
   and whether thinning is possible; with the value differences best-Pressure minus best-Vision and each
   minus best-Safe against B's equilibrium strategy.
2. Constructed siege probes solved exactly: siege mass {4, 8, 12} x defender's waiting wave {none, 4, 8} x
   existing vision {no, yes} x turns remaining {8, 16}, plus gank-loss variants; each reports A's strategy
   and the Q-value and decomposition (farm, plates, gank, continuation) of the best action of each kind.
"""

from __future__ import annotations

import itertools

from ... import GameConfig, LaneGame
from ...config import PRESSURE, VISION, objective_of, posture_of
from ...solver import solve_equilibrium
from ...state import State
from .common import K, md_table, save, siege_node_table, solve



def decompose_probe(cfg: GameConfig, state: State, k0: int) -> dict:
    g = LaneGame(cfg)
    sol = solve_equilibrium(g, root=state, k0=k0)
    la, lb = sol.legal[k0][state]
    y = sol.y[k0][state]
    Q = sol.Q[k0][state]
    rows = {}
    for i, a in enumerate(la):
        farm = gank = plate = 0.0
        for j, b in enumerate(lb):
            tr = g.transition(state, k0, a, b)
            farm += y[j] * tr.farm_payoff
            gank += y[j] * tr.gank_payoff
            plate += y[j] * tr.plate_payoff
        q = float(Q[i] @ y)
        rows[cfg.label(a)] = {"Q": q, "farm": farm, "plates": plate, "gank": gank,
                              "continuation": q - farm - gank - plate, "objective": objective_of(a),
                              "posture": posture_of(a)}
    best = {}
    for kind, pred in (("pressure", lambda r: r["objective"] == PRESSURE), ("vision", lambda r: r["objective"] == VISION),
                       ("safe", lambda r: r["posture"] == 1)):
        cand = [(lab, r) for lab, r in rows.items() if pred(r)]
        best[kind] = max(cand, key=lambda t: t[1]["Q"]) if cand else None
    return {"value": sol.value, "A": {cfg.label(a): round(float(p), 3) for a, p in zip(la, sol.x[k0][state]) if p > 1e-9},
            "B": {cfg.label(b): round(float(p), 3) for b, p in zip(lb, sol.y[k0][state]) if p > 1e-9},
            "actions": rows, "best": {k: (v[0], v[1]) if v else None for k, v in best.items()},
            "nodes": sol.stats.states}


def run(horizon: int = K) -> dict:
    cfg = GameConfig(horizon=horizon)
    g, layers, sol, timing = solve(cfg)
    grouped = siege_node_table(g, sol, horizon - 4)
    out = {"horizon": horizon, "grouped": grouped, "timing": timing}

    probes = {}
    for mass, dwave, vis, rem in itertools.product((4, 8, 12), (0, 4, 8), (0, 1), (8, 16)):
        plates = cfg.plate_grant[cfg.size_category(mass)]
        bx, bh = (-3, dwave) if dwave else (-4, 0)
        st = State(-4, 4, (), mass, bx, bh, (), 0, cfg.vision_turns // 2 if vis else 0, 0, plates, 0)
        k0 = horizon - rem
        probes[f"mass {mass}, defender {dwave}, vision {'yes' if vis else 'no'}, {rem} turns left"] = \
            {"mass": mass, "defender": dwave, "vision": vis, "remaining": rem, **decompose_probe(cfg, st, k0)}
    gl = {}
    for L in (3.5, 5.0, 7.0):
        c = cfg.with_(gank_loss=L)
        st = State(-4, 4, (), 12, -3, 4, (), 0, 0, 0, 3, 0)
        gl[L] = decompose_probe(c, st, horizon - 16)
    out["probes"] = probes
    out["gank_loss_probes"] = {str(L): v for L, v in gl.items()}

    md = [f"# Experiment A: Apply Pressure vs. Establish Vision across siege states (K = {horizon})", "",
          "## Equilibrium siege decisions of A at reachable siege nodes (k <= K - 4), grouped", "",
          "P(.) = equilibrium probability; 'P - V' = Q of the best Pressure action minus Q of the best Vision action "
          "against B's equilibrium strategy (resources); 'P - S' and 'V - S' likewise against the best Safe action.", ""]
    for gname, gd in grouped.items():
        md += [f"### By {gname}", "",
               md_table([gname, "nodes", "P(Pressure)", "P(Vision)", "P(disengage)", "P - V", "P - S", "V - S"],
                        [(key, r["nodes"], round(r["p_pressure"], 3), round(r["p_vision"], 3), round(r["p_disengage"], 3),
                          round(r["pressure_minus_vision"], 3), round(r["pressure_minus_safe"], 3),
                          round(r["vision_minus_safe"], 3)) for key, r in gd.items()]), ""]
    md += ["## Constructed siege probes (A's crashed mass under B's tower; B's waiting wave one cell behind its tower; "
           "pending plates = the crash's grant; existing vision = half the duration left)", "",
           "Each probe is solved exactly from its state. Q = farm + plates + gank + continuation against B's "
           "equilibrium strategy, for the best action of each kind.", ""]
    rows = []
    for name, p in probes.items():
        b = p["best"]
        rows.append((p["mass"], p["defender"] or "-", "yes" if p["vision"] else "no", p["remaining"], round(p["value"], 3),
                     ", ".join(f"{k} {v}" for k, v in p["A"].items()),
                     *[f"{b[kind][1]['Q']:.2f} ({b[kind][0]})" if b[kind] else "-" for kind in ("pressure", "vision", "safe")],
                     round(b["pressure"][1]["Q"] - b["vision"][1]["Q"], 3) if b["pressure"] and b["vision"] else "-"))
    md += [md_table(["mass", "defender wave", "vision", "turns left", "V*", "A plays", "best Pressure Q", "best Vision Q",
                     "best Safe Q", "P - V"], rows), "",
           "### Decomposition of the best action of each kind (mass 12, defender wave 4, 16 turns left)", ""]
    for key in ("mass 12, defender 4, vision no, 16 turns left", "mass 12, defender 4, vision yes, 16 turns left",
                "mass 4, defender 4, vision no, 16 turns left", "mass 8, defender 8, vision no, 16 turns left"):
        p = probes[key]
        md += [f"**{key}** (V* = {p['value']:.3f}; A plays {p['A']}; B plays {p['B']})", "",
               md_table(["action", "Q", "farm", "plates", "gank", "continuation"],
                        [(lab, round(r["Q"], 3), round(r["farm"], 3), round(r["plates"], 3), round(r["gank"], 3),
                          round(r["continuation"], 3)) for lab, r in p["actions"].items()]), ""]
    md += ["### Gank loss L (mass 12, defender wave 4, no vision, 16 turns left)", "",
           md_table(["L", "V*", "A plays", "best Pressure Q", "best Vision Q", "best Safe Q"],
                    [(L, round(p["value"], 3), ", ".join(f"{k} {v}" for k, v in p["A"].items()),
                      *[round(p["best"][kind][1]["Q"], 3) for kind in ("pressure", "vision", "safe")]) for L, p in gl.items()])]
    save("siege_options", out, "\n".join(md))
    return out


if __name__ == "__main__":
    run()
