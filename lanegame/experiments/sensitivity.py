"""Experiment E - sensitivity and robustness of the revised model's conclusions.

One-at-a-time perturbations of every hand-specified parameter group around the documented baseline
(GameConfig.v2()), plus selected joint sweeps. For every setting the game is re-solved exactly and the
quantities the conclusions rest on are recorded, together with explicit pass/fail checks of the
qualitative claims:

  C1  the symmetric equilibrium path never crashes
  C2  Contest is the majority posture on the equilibrium path
  C3  aggressive (Shove + Contest) is the most exploitable baseline
  C4  against passive_safe, most of the best response's gain is economic (farm + crash), not ganks
  C5  a 3-wave crash under the enemy tower has negative continuation value for the crasher
  C6  at A's freeze line (own -1, Freeze legal) Freeze + Contest is optimal more often than Freeze + Safe
"""

from __future__ import annotations

import itertools

import numpy as np

from .. import GameConfig
from ..analysis import path_statistics
from ..baselines import all_baselines, passive_safe
from ..best_response import best_response_value, exploitability
from ..config import F, posture_of, wave_of
from .common import REVISION, eq_summary, md_table, save, solve
from .probes import decompose, probes, solve_probe

K = 32
BASE = GameConfig.v2(horizon=K)
CRASH = BASE.crash_options


def scaled_crash(f):
    return tuple(tuple((r * f, v) for r, v in opts) for opts in CRASH)


def contest_flat(c):
    return (c, c + 0.03, c, 0.0, 0.0, 0.0)


def safe_line(x):
    return (None, None, None, x, x, x)


ONE_AT_A_TIME = {
    "farm_per_hp": [0.125, 0.5],
    "collect_contest": [0.8, 1.0],
    "collect_safe_slope": [0.1, 0.3],
    "zone_contest": [0.15, 0.45],
    "gank_loss": [3.5, 7.0],
    "gank_position": [0.05, 0.15],
    "gank_base": [0.10, 0.20],
    "vision_reduction": [0.25, 0.75],
    "contest_risk": [0.0, 0.04],           # flat Contest gank term (Shove + Contest gets +0.03 on top)
    "safe_line": [-2, 0],                  # how far forward a Safe champion goes (own coordinates)
    "shove_damage": [0, 2],
    "freeze_break_threshold": [8, 12],
    "tower_siege_damage": [2],
    "crash_reward_scale": [0.5, 2.0],
}
JOINT = {
    ("farm_per_hp", "gank_loss"): ([0.125, 0.25, 0.5], [3.5, 5.0, 7.0]),
    ("collect_safe_slope", "zone_contest"): ([0.1, 0.2, 0.3], [0.15, 0.3, 0.45]),
    ("crash_reward_scale", "farm_per_hp"): ([0.5, 1.0, 2.0], [0.125, 0.25, 0.5]),
    ("tower_siege_damage", "farm_per_hp"): ([1, 2], [0.125, 0.25, 0.5]),
}


def make(params: dict) -> GameConfig:
    kw = {}
    for p, v in params.items():
        if p == "crash_reward_scale":
            kw["crash_options"] = scaled_crash(v)
        elif p == "contest_risk":
            kw["gank_action"] = contest_flat(v)
        elif p == "safe_line":
            kw["posture_cap"] = safe_line(v)
        else:
            kw[p] = v
    return BASE.with_(**kw)


def freeze_line_postures(g, sol):
    """At nodes where A's leading wave is at own -1 and Freeze is legal: share where F + Contest / F + Safe
    is a best response to B's equilibrium strategy."""
    fc = fs = n = 0
    for k, layer in enumerate(sol.layers):
        if k > g.horizon - 8:
            continue
        for s in layer:
            la, _ = sol.legal[k][s]
            if F not in la or s.ah == 0 or s.ax != -1 or s.asg or s.bsg:
                continue
            rows = decompose(g, sol, k, s)
            v = sol.V[k][s]
            best = [a for a, r in zip(la, rows.values()) if r["Q"] >= v - 1e-9]
            n += 1
            fc += any(wave_of(a) == F and posture_of(a) == 0 for a in best)
            fs += any(wave_of(a) == F and posture_of(a) == 1 for a in best)
    return {"nodes": n, "fc": fc / n if n else float("nan"), "fs": fs / n if n else float("nan")}


def evaluate_setting(params: dict) -> dict:
    cfg = make(params)
    g, layers, sol, secs = solve(cfg)
    e = eq_summary(g, sol)
    expl = {b.name: exploitability(g, layers, b, 0, sol.value).exploitability for b in all_baselines()}
    br = best_response_value(g, layers, passive_safe(), 0)
    ps = path_statistics(g, passive_safe(), br.policy).to_dict(K)
    econ = (ps["farm"][0] - ps["farm"][1]) + (ps["crash_resources"][0] - ps["crash_resources"][1])
    gank = -(ps["gank_loss"][0] - ps["gank_loss"][1])
    pr = probes(cfg.wave_hp)
    crash_probe = solve_probe(cfg, pr["A's 3-wave crash under B's tower"])
    freeze_probe = solve_probe(cfg, pr["B wave at A's freeze line, fighting A's wave"])
    fl = freeze_line_postures(g, sol)
    order = sorted(expl, key=lambda n: -expl[n])
    checks = {
        "C1": sum(e["path"]["crashes_by_size"]["A"]) < 1e-9,
        "C2": e["path"]["contest_share"][0] > 0.5,
        "C3": order[0] == "aggressive",
        "C4": econ < 0 and abs(econ) > abs(min(gank, 0.0)),
        "C5": crash_probe["value"] < 0,
        "C6": fl["fc"] > fl["fs"],
    }
    return {"params": params, "seconds": secs, "states": e["states"], "mixed": e["mixed"],
            "eq_crashes": float(sum(e["path"]["crashes_by_size"]["A"])),
            "eq_contest": e["path"]["contest_share"][0], "node_contest": e["nodes"]["contest"],
            "freeze_when_legal": e["nodes"]["freeze_prob_when_legal"], "vision": e["vision"],
            "eq_collection": (e["path"]["farm"][0] / e["path"]["farm_available"][0]) if e["path"]["farm_available"][0] else None,
            "exploitability": expl, "order": order,
            "passive_safe_parts": {"economic": econ, "gank": gank},
            "probe_crash_value": crash_probe["value"], "probe_freeze_value": freeze_probe["value"],
            "probe_freeze_A": freeze_probe["A"], "freeze_line": fl, "checks": checks}


def run() -> dict:
    res = {"baseline": evaluate_setting({})}
    for p, vals in ONE_AT_A_TIME.items():
        for v in vals:
            print(p, v, flush=True)
            res[f"{p}={v}"] = evaluate_setting({p: v})
    joint = {}
    for (p1, p2), (v1s, v2s) in JOINT.items():
        for v1, v2 in itertools.product(v1s, v2s):
            print(p1, v1, p2, v2, flush=True)
            joint[f"{p1}={v1}, {p2}={v2}"] = evaluate_setting({p1: v1, p2: v2})

    names = [b.name for b in all_baselines()]

    def rows(d):
        out = []
        for name, r in d.items():
            vs = r["vision"]
            out.append((name, r["states"], r["mixed"], round(r["eq_crashes"], 2), round(r["eq_contest"], 2),
                        round(r["node_contest"], 2), round(r["freeze_when_legal"], 3),
                        "/".join(f"{vs.get(s, float('nan')):.2f}" for s in ("small", "medium", "large")),
                        *[round(r["exploitability"][n], 1) for n in names],
                        round(r["probe_crash_value"], 2), round(r["probe_freeze_value"], 2),
                        "".join("Y" if r["checks"][c] else "n" for c in sorted(r["checks"]))))
        return out

    header = ["setting", "nodes", "mixed", "eq crashes", "eq P(C)", "node P(C)", "P(F) legal", "vision S/M/L",
              *[f"expl {n}" for n in names], "V(crash under B)", "V(freeze probe)", "C1-C6"]
    all_rows = {**res, **joint}
    tally = {c: sum(r["checks"][c] for r in all_rows.values()) for c in res["baseline"]["checks"]}
    md = ["# Experiment E: sensitivity and robustness (K = 32)", "",
          "Every row re-solves the game exactly. Baseline = GameConfig.v2() (the revision report's default). "
          "`contest_risk` is the flat Contest gank term (Shove + Contest adds 0.03 on top); `safe_line` is the "
          "furthest own position a Safe champion goes; `crash_reward_scale` multiplies the crash reward table.", "",
          "Qualitative checks (Y = holds):",
          "- C1 the symmetric equilibrium path never crashes",
          "- C2 Contest is the majority posture on the equilibrium path",
          "- C3 aggressive is the most exploitable baseline",
          "- C4 against passive_safe the best response's gain is mostly economic (farm + crash), not ganks",
          "- C5 a 3-wave crash under the enemy tower has negative continuation value for the crasher",
          "- C6 at own -1 with Freeze legal, Freeze + Contest is optimal more often than Freeze + Safe", "",
          f"Across all {len(all_rows)} settings: " + ", ".join(f"{c} holds in {n}" for c, n in tally.items()), "",
          "## One parameter at a time", "", md_table(header, rows(res)), "",
          "## Joint sweeps", "", md_table(header, rows(joint)), "",
          "## Ranking of baselines by exploitability (most exploitable first)", "",
          md_table(["setting", "order"], [(n, " > ".join(r["order"])) for n, r in all_rows.items()])]
    save("sensitivity", {"one_at_a_time": res, "joint": joint, "tally": tally}, "\n".join(md), REVISION)
    return res


if __name__ == "__main__":
    run()
