"""Experiment C - when is Contest worth it, and when is Safe?

For every reachable decision node of A (k <= K - 8, to keep end effects out), against B's equilibrium
strategy, the Contest advantage is

    Delta(s) = max_w Q(s, (w, Contest)) . y*  -  max_w Q(s, (w, Safe)) . y*,

i.e. how much A loses by being forced to play Safe (Delta > 0) or Contest (Delta < 0) with its best wave
management. Delta and the equilibrium probability of Contest are grouped by the situation: where A's
minions are, A's vision, wave strengths, and how hard B contests. Representative states are decomposed
into immediate farm, immediate gank risk and continuation value. The gank loss L is swept, and the use
of Freeze (and Freeze + Contest vs. Freeze + Safe) is analysed separately.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .. import GameConfig
from ..config import F, posture_of, wave_of
from .common import REVISION, md_table, node_averages, save, solve
from .probes import K0, contest_advantage, contest_prob, decompose, probes, solve_probe

K = 32
LOSSES = [2.0, 3.5, 5.0, 7.0, 10.0]


def situation(s, cfg):
    if s.asg > 0:
        return "A has crashed minions under B's tower"
    if s.bsg > 0:
        return "B has crashed minions under A's tower"
    if s.ah == 0:
        return "A has no wave in the lane"
    return f"A's leading wave at {s.ax:+d}"


def grouped(g, sol, max_k):
    cfg = g.cfg
    groups = {k: defaultdict(list) for k in ("situation", "vision", "own stacks", "enemy stacks", "B contests")}
    freeze = defaultdict(lambda: np.zeros(7))     # [nodes, P(F), P(F-C), P(F-S), F optimal, F-C opt., F-S opt.]
    optimal = defaultdict(lambda: np.zeros(4))    # [nodes, Contest optimal, Safe optimal, both]
    for k, layer in enumerate(sol.layers):
        if k > max_k:
            continue
        for s in layer:
            la, lb = sol.legal[k][s]
            x, y = sol.x[k][s], sol.y[k][s]
            rows = decompose(g, sol, k, s)
            d = contest_advantage(rows)
            # best responses to B's equilibrium strategy (the actions any equilibrium of A may use here)
            v = sol.V[k][s]
            best = {a for a, r in zip(la, rows.values()) if r["Q"] >= v - 1e-9}
            c_opt = any(posture_of(a) == 0 for a in best)
            s_opt = any(posture_of(a) == 1 for a in best)
            optimal[situation(s, cfg)] += np.array([1.0, c_opt, s_opt, c_opt and s_opt])
            pc = contest_prob(la, x)
            pb = contest_prob(lb, y)
            rec = (d, pc)
            groups["situation"][situation(s, cfg)].append(rec)
            groups["vision"]["A has vision" if s.va else "no vision"].append(rec)
            groups["own stacks"][f"A front stacks {cfg.stacks(s.ah)}" if s.ah else "A no front"].append(rec)
            groups["enemy stacks"][f"B front stacks {cfg.stacks(s.bh)}" if s.bh else "B no front"].append(rec)
            groups["B contests"]["B contests with p > 0.5" if pb > 0.5 else "B contests with p <= 0.5"].append(rec)
            if F in la:
                key = situation(s, cfg)
                pf = sum(p for a, p in zip(la, x) if wave_of(a) == F)
                pfc = sum(p for a, p in zip(la, x) if wave_of(a) == F and posture_of(a) == 0)
                f_opt = any(wave_of(a) == F for a in best)
                fc_opt = any(wave_of(a) == F and posture_of(a) == 0 for a in best)
                fs_opt = any(wave_of(a) == F and posture_of(a) == 1 for a in best)
                freeze[key] += np.array([1.0, pf, pfc, pf - pfc, f_opt, fc_opt, fs_opt])
    out = {}
    for gname, gd in groups.items():
        out[gname] = {k: {"n": len(v), "p_contest": float(np.mean([r[1] for r in v])),
                          "share_contest_better": float(np.mean([r[0] > 1e-9 for r in v])),
                          "share_safe_better": float(np.mean([r[0] < -1e-9 for r in v])),
                          "mean_advantage": float(np.mean([r[0] for r in v]))}
                      for k, v in sorted(gd.items())}
    out["freeze"] = {k: {"nodes": int(v[0]), "p_freeze": v[1] / v[0], "p_freeze_contest": v[2] / v[0],
                         "p_freeze_safe": v[3] / v[0], "freeze_optimal": v[4] / v[0],
                         "freeze_contest_optimal": v[5] / v[0], "freeze_safe_optimal": v[6] / v[0]}
                     for k, v in sorted(freeze.items())}
    out["optimal"] = {k: {"nodes": int(v[0]), "contest_optimal": v[1] / v[0], "safe_optimal": v[2] / v[0],
                          "both_optimal": v[3] / v[0]} for k, v in sorted(optimal.items())}
    return out


def run(horizon: int = K) -> dict:
    cfg = GameConfig.v2(horizon=horizon)
    g, layers, sol, _ = solve(cfg)
    res = {"grouped": grouped(g, sol, horizon - 8)}

    # representative states, decomposed
    ps = probes(cfg.wave_hp)
    res["probes"] = {n: solve_probe(cfg, s) for n, s in ps.items()}

    # gank-loss severity
    sweep = {}
    for L in LOSSES:
        c = cfg.with_(gank_loss=L)
        gg, ll, so, _ = solve(c)
        from .common import eq_summary
        e = eq_summary(gg, so)
        sweep[L] = {"p_contest_nodes": e["nodes"]["contest"], "p_contest_path": e["path"]["contest_share"][0],
                    "freeze_when_legal": e["nodes"]["freeze_prob_when_legal"],
                    "probes": {n: solve_probe(c, s)["contest_advantage"] for n, s in ps.items()}}
    res["gank_loss_sweep"] = sweep

    # freeze-specific sweep: how much does zoning (denial) and Safe's collection fall-off matter?
    fz = {}
    for zone in (0.0, 0.3, 0.6):
        for slope in (0.1, 0.2, 0.3):
            c = cfg.with_(zone_contest=zone, collect_safe_slope=slope)
            gg, ll, so, _ = solve(c)
            na = node_averages(so)
            fp = solve_probe(c, ps["B wave at A's freeze line, fighting A's wave"])
            fz[f"zone {zone}, safe slope {slope}"] = {"freeze_when_legal": na["freeze_prob_when_legal"],
                                                      "contest_nodes": na["contest"],
                                                      "probe_A": fp["A"], "probe_value": fp["value"]}
    res["freeze_sweep"] = fz

    G = res["grouped"]
    md = [f"# Experiment C: Contest vs. Safe (K = {horizon}; nodes with k <= {horizon - 8})", "",
          "Contest advantage = best Contest action minus best Safe action for A, against B's equilibrium "
          "strategy (resources). 'Contest better' / 'Safe better' = share of nodes where the advantage is "
          "strictly positive / negative (the rest are ties).", ""]
    for gname in ("situation", "vision", "own stacks", "enemy stacks", "B contests"):
        md += [f"## By {gname}", "",
               md_table([gname, "nodes", "P(Contest) in eq", "Contest better", "Safe better", "mean advantage"],
                        [(k, v["n"], round(v["p_contest"], 3), round(v["share_contest_better"], 3),
                          round(v["share_safe_better"], 3), round(v["mean_advantage"], 3))
                         for k, v in G[gname].items()]), ""]
    md += ["## Representative states (k = 12 of 32): best Contest vs. best Safe action, decomposed", "",
           "Q = immediate farm difference + immediate expected gank payoff + continuation (crash rewards and "
           "the value of the next state), all against B's equilibrium strategy.", ""]
    for n in ("B wave at A's freeze line, fighting A's wave", "B 2-wave stack at A's freeze line"):
        d = res["probes"][n]
        md += [f"**All of A's actions in '{n}'** (V* = {d['value']:.3f})", "",
               md_table(["action", "Q", "farm", "gank", "continuation"],
                        [(lab, round(r["Q"], 3), round(r["farm"], 3), round(r["gank"], 3), round(r["continuation"], 3))
                         for lab, r in d["actions"].items()]), ""]
    for n, d in res["probes"].items():
        rows = d["actions"]
        bc = max((r for r in rows.items() if r[1]["posture"] == 0), key=lambda r: r[1]["Q"])
        bs = max((r for r in rows.items() if r[1]["posture"] == 1), key=lambda r: r[1]["Q"])
        md += [f"**{n}**: V* = {d['value']:.3f}; A plays {d['A']}; B plays {d['B']}", "",
               md_table(["action", "Q", "farm", "gank", "continuation"],
                        [(lab, round(r["Q"], 3), round(r["farm"], 3), round(r["gank"], 3), round(r["continuation"], 3))
                         for lab, r in (bc, bs)]), ""]
    md += ["## Gank-loss severity L", "",
           md_table(["L", "P(Contest) avg node", "eq path P(Contest)", "P(Freeze) where legal"]
                    + [f"adv: {n}" for n in ps],
                    [(L, round(d["p_contest_nodes"], 3), round(d["p_contest_path"], 3), round(d["freeze_when_legal"], 3),
                      *[round(d["probes"][n], 2) for n in ps]) for L, d in sweep.items()]), "",
           "## Which postures are optimal at all (best responses to B's equilibrium strategy), by situation", "",
           "The solver returns one equilibrium and breaks exact ties toward lower action codes "
           "(Slow Push < Shove < Freeze, Contest < Safe), so equilibrium frequencies of Freeze and Safe are lower "
           "bounds on how often these actions are optimal. This table counts nodes where at least one action with "
           "the given posture is a best response.", "",
           md_table(["situation", "nodes", "a Contest action optimal", "a Safe action optimal", "both optimal"],
                    [(k, v["nodes"], round(v["contest_optimal"], 3), round(v["safe_optimal"], 3),
                      round(v["both_optimal"], 3)) for k, v in G["optimal"].items()]), "",
           "## Freeze use (nodes where Freeze is legal), by A's situation", "",
           md_table(["situation", "nodes", "P(Freeze) in eq", "P(F + Contest)", "P(F + Safe)", "Freeze optimal",
                     "F + Contest optimal", "F + Safe optimal"],
                    [(k, v["nodes"], round(v["p_freeze"], 3), round(v["p_freeze_contest"], 3),
                      round(v["p_freeze_safe"], 3), round(v["freeze_optimal"], 3),
                      round(v["freeze_contest_optimal"], 3), round(v["freeze_safe_optimal"], 3))
                     for k, v in G["freeze"].items()]), "",
           "## Freeze use vs. zoning strength and Safe collection fall-off", "",
           md_table(["setting", "P(Freeze) where legal", "P(Contest) avg node",
                     "probe 'B wave at A's freeze line': A plays", "probe V*"],
                    [(k, round(v["freeze_when_legal"], 3), round(v["contest_nodes"], 3), v["probe_A"],
                      round(v["probe_value"], 3)) for k, v in fz.items()])]
    save("contest_safe", res, "\n".join(md), REVISION)
    return res


if __name__ == "__main__":
    run()
