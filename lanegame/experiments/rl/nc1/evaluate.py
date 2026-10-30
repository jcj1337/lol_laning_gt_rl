"""Stage 3: evaluation of the PPO pilot (pre-registration). Every readout is arm - paired control per seed (the control
is M5's sp snapshot of the seed at BUDGET), then the mean over six seeds with a 95% t interval.

    E1  head to head from s0 with `m5.games.cell_job`: W and F against the control on the same 256 paired-seat games of
        150 turns per seed (root H2H_ROOT, cell s); M5's lane-exit readout `analysis.own_lead` (lead at turn 114)
    E2  M5c's cross-arm families (`m5.cells`: conv, freeze, recall, gaps) with the arm in M5's "om" slot and the control
        in the "sp" slot, so `m5.strategic.cross_arm`'s "om - sp" is arm - control; NC1's rollout seeds (CELL_ROOT)
    E3  descriptive: training health, Q1 on the finals, a ridge probe of the log duel ratio from the last hidden layer,
        and the critic's predicted strength premium at the E4 gap cells
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from . import common as C

EOUT = os.path.join(C.CKPT, "eval")
H2H_N, H2H_T = 256, 150
FAMILIES = ("conv", "freeze", "recall", "gaps")
LONGEST = {"conv": 150, "freeze": 50, "recall": 100, "gaps": 150}


def h2h_jobs() -> list:
    return [{"key": f"h2h_s{s}", "path": os.path.join(EOUT, "h2h", f"s{s}.npz"),
             "focals": [("net", C.final_path(a, s)) for a in C.PILOT_ARMS], "opponent": ("net", C.control_path(s)),
             "N": H2H_N, "T": H2H_T, "root": C.H2H_ROOT, "cell": s} for s in C.SEEDS]


def cells_root(arm: str) -> str:
    return os.path.join(EOUT, "cells", arm)


def cell_jobs(arms=C.PILOT_ARMS) -> list:
    from ..m46a import engine as EN
    from ..m5 import cells as CE
    js = []
    for arm in arms:
        for s in C.SEEDS:
            specs = {"om": C.final_path(arm, s), "sp": C.control_path(s)}
            root = cells_root(arm)
            js += (CE.conv_jobs(s, specs, root) + CE.freeze_jobs(s, specs, root) + CE.recall_jobs(s, specs, root)
                   + CE.gaps_jobs(s, specs, root))
    for j in js:
        j["roll"] = EN.roll_seeds(C.CELL_ROOT * 10 + CE.FAMILY_OFFSET[j["family"]], len(j["roll"]))
    return js


def run(workers: int, arms=None) -> dict:
    """E2 for `arms` only (records are resumable), or everything and the analysis when `arms` is None."""
    from ..m5 import cells as CE
    from ..m5 import games as G
    if arms:
        CE.run(cell_jobs(arms), workers, f"E2 limitation families ({','.join(arms)})")
        return {}
    C.run_jobs(G.cell_job, h2h_jobs(), workers, "E1 head to head")
    CE.run(cell_jobs(), workers, "E2 limitation families")
    return analyse()


# ---------------------------------------------------------------------------------------------- readouts
def e1() -> dict:
    from ..m5 import analysis as AN
    out = {}
    for f, arm in enumerate(C.PILOT_ARMS):
        per = {s: AN.own_lead([os.path.join(EOUT, "h2h", f"s{s}.npz")], focal=f) for s in C.SEEDS}
        out[arm] = {"per_seed": {str(s): v["mean"] for s, v in per.items()},
                    "per_seed_se": {str(s): v["se"] for s, v in per.items()},
                    **AN.interval({s: v["mean"] for s, v in per.items()})}
    return out


def e2() -> dict:
    from ..m5 import strategic as SG
    out = {}
    for arm in C.PILOT_ARMS:
        fam = {}
        for f in FAMILIES:
            r = SG.cross_arm(f, C.SEEDS, root=cells_root(arm))
            fam[f] = {"status": r["status"],
                      "stats": {k.replace("om - sp", "arm - control").replace("specialist - om", "specialist - arm")
                                .replace("specialist - sp", "specialist - control"): v for k, v in r["stats"].items()}}
        out[arm] = fam
    return out


def health() -> dict:
    from .train import history
    out = {}
    for arm in C.PILOT_ARMS:
        for s in C.SEEDS:
            h = history(C.run_dir(arm, s))
            tail = h[-200:]
            out[f"{arm}_s{s}"] = {"updates": len(h), "agent_transitions": h[-1]["agent_transitions"] if h else 0,
                                  **{k: float(np.mean([e[k] for e in tail])) for k in
                                     ("entropy", "approx_kl", "explained_variance", "clipfrac", "grad_norm")}}
    for s in C.SEEDS:
        from .train import history as hist
        h = hist(os.path.join("results", "rl", "checkpoints", "m5", "train", "main", "sp", f"s{s}"))
        n = C.BUDGET // 4096
        tail = h[n - 200:n]
        out[f"control_s{s}"] = {"updates": n, **{k: float(np.mean([e[k] for e in tail])) for k in
                                                 ("entropy", "approx_kl", "explained_variance", "clipfrac",
                                                  "grad_norm")}}
    return out


def _load_net(path: str):
    from ....rl import load_checkpoint
    net, _, _ = load_checkpoint(path)
    net.eval()
    return net


def probe() -> dict:
    """Held-out R^2 of the log duel ratio from the last hidden layer (ridge, the strength explained by the layer the
    value head reads), on the Q2 natural states of target 0 (fit on N_train, scored on N_test)."""
    import torch
    from ....rl.network import strength_features
    from .probe import load
    tr, te = load(0, "N_train"), load(0, "N_test")
    idx = C.feature_index()
    y_tr = strength_features(torch.as_tensor(tr["obs"]), idx)[:, 0].double().numpy()
    y_te = strength_features(torch.as_tensor(te["obs"]), idx)[:, 0].double().numpy()

    def r2(net) -> float:
        with torch.no_grad():
            F_tr = net.value_features(torch.as_tensor(tr["obs"])).double().numpy()
            F_te = net.value_features(torch.as_tensor(te["obs"])).double().numpy()
        mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-8
        Z, Zt = np.c_[np.ones(len(F_tr)), (F_tr - mu) / sd], np.c_[np.ones(len(F_te)), (F_te - mu) / sd]
        I = np.eye(Z.shape[1])
        I[0, 0] = 0
        beta = np.linalg.solve(Z.T @ Z + 1.0 * I, Z.T @ y_tr)
        e = Zt @ beta - y_te
        return float(1 - (e ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum())

    out = {}
    for s in C.SEEDS:
        out[f"control_s{s}"] = r2(_load_net(C.control_path(s)))
        for arm in C.PILOT_ARMS:
            out[f"{arm}_s{s}"] = r2(_load_net(C.final_path(arm, s)))
    summ = {}
    for arm in C.PILOT_ARMS:
        summ[f"{arm} - control"] = C.mean_ci([out[f"{arm}_s{s}"] - out[f"control_s{s}"] for s in C.SEEDS])
        summ[arm] = C.mean_ci([out[f"{arm}_s{s}"] for s in C.SEEDS])
    summ["control"] = C.mean_ci([out[f"control_s{s}"] for s in C.SEEDS])
    return {"per_net": out, "summary": summ}


def critic_premium() -> dict:
    """The critic's predicted premium (V of the stronger seat A at t = 0, gap cell - the equal cell of the same
    situation and level, in resources) next to the realized G150 premium of the network playing itself (E2 gaps
    records), per network, averaged over the gap cells."""
    from ..m46a.common import env_cfg
    from ..m46a.e4 import EQUAL_CELLS, GAP_CELLS
    from ..m46a.states import zoning_gap_state
    from ..m45c.horizon import critic_value
    from ..m5 import strategic as SG
    cfg = env_cfg()
    gaps = {c: zoning_gap_state(cfg, *c) for c in GAP_CELLS + EQUAL_CELLS}
    items = GAP_CELLS + EQUAL_CELLS

    def pred(net) -> float:
        v = {c: critic_value(net, cfg, st[0], st[1], st[2], 0) for c, st in gaps.items()}
        return float(np.mean([v[c] - v[(c[0], c[1], "equal")] for c in GAP_CELLS]))

    def realized(arm: str, s: int, cand: str, opp: str) -> float:
        recs = SG._load_dir("gaps", f"s{s}_vs_{opp}", cells_root(arm))
        g = {items[i]: float(np.mean(r["G"][cand]["150"])) for i, r in recs.items()}
        if len(g) < len(items):
            return float("nan")
        return float(np.mean([g[c] - g[(c[0], c[1], "equal")] for c in GAP_CELLS]))

    out = {}
    for s in C.SEEDS:
        out[f"control_s{s}"] = {"predicted": pred(_load_net(C.control_path(s))),
                                "realized": realized(C.PILOT_ARMS[0], s, "sp", "sp")}
        for arm in C.PILOT_ARMS:
            out[f"{arm}_s{s}"] = {"predicted": pred(_load_net(C.final_path(arm, s))),
                                  "realized": realized(arm, s, "om", "om")}
    summ = {k: {f: C.mean_ci([out[f"{k}_s{s}"][f] for s in C.SEEDS]) for f in ("predicted", "realized")}
            for k in ("control",) + C.PILOT_ARMS}
    return {"per_net": out, "summary": summ}


def verdicts(E1: dict, E2: dict) -> Dict[str, dict]:
    out = {}
    for arm in C.PILOT_ARMS:
        regress = []
        for f in FAMILIES:
            st = E2[arm][f]["stats"].get(f"arm - control | all | G{LONGEST[f]}")
            if st and st["mean"] < -0.3 and st["U"] < 0:
                regress.append(f)
        g = E2[arm]["gaps"]["stats"].get("arm - control | all | G150", {})
        improves = E1[arm]["L"] > 0 and not regress
        gap_play = bool(g) and g["L"] > 0
        if improves:
            v = "improves play"
        elif E1[arm]["L"] <= 0 <= E1[arm]["U"] and -0.5 <= E1[arm]["L"] and E1[arm]["U"] <= 0.5:
            v = "no material effect"
        else:
            v = "inconclusive"
        out[arm] = {"E1": v, "improves_gap_play": gap_play, "regressing_families": regress}
    return out


def analyse() -> dict:
    E1, E2 = e1(), e2()
    out = {"meta": C.meta(stage="evaluate"), "E1": E1, "E2": E2, "verdicts": verdicts(E1, E2),
           "E3": {"health": health(), "probe": probe(), "critic_premium": critic_premium()}}
    C.save_json("evaluation", out)
    return out
