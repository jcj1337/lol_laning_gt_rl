"""E2: selective threats as plans on the trained policy (pre-registration §2 E2).

Families (focal A = the stronger side; horizon 60 = the 20-turn plans + 40; protocols "policy" and "g_wary"):
    E2c   constructed: M4.5b zoning situations neutral / a_push / b_push x levels 6 / 10 x gaps equal / x1.5 / x1.76 /
          x2.2 (24 cells), K = 24; validation: 6 cells (fixed seed) at 300 turns under "policy"; primary G60
    E2n   natural HP edge: 24 states per final from its own self-play (states.p_hp_edge), K = 12; primary Q60
    E2s   semi-natural gap: 24 states per final from E3's x1.76 training-split games, K = 12; primary G60
Candidates (policies.build): the policy, the five behaviour-only plans, and the two scripted plans (M4.5c's g_bully
and M4.4c's Farm wave restriction).
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from . import families as FA
from .engine import ledger_mean
from .states import collect_job
from .common import E2_STATE_ROOT, FINALS, ROLL_ROOT, env_cfg, holm, mean_ci, meta, run_jobs, save_json, t_pvalue

PLAN_N, H, REL = 20, 60, 0.4
CANDS = {"policy": ("policy",),
         "farm_beh": ("beh", "farm", PLAN_N, REL), "threaten_beh": ("beh", "threaten", PLAN_N, REL),
         "edge_beh": ("beh", "edge", PLAN_N, REL), "bully_beh": ("beh", "bully", PLAN_N, REL),
         "freeze_threat": ("beh", "freeze_threat", PLAN_N, REL),
         "g_bully_script": ("script", "g_bully", PLAN_N), "g_farm_script": ("script", "g_farm", PLAN_N)}
PLANS4 = ("threaten_beh", "edge_beh", "bully_beh", "freeze_threat")
PROTOS = ("policy", "g_wary")
CELLS = [(sit, lvl, gap) for sit in ("neutral", "a_push", "b_push") for lvl in (6, 10)
         for gap in ("equal", "x1.5", "x1.76", "x2.2")]
K_CON, K_NAT, N_NAT = 24, 12, 24
N_VAL_CELLS = 6


def constructed_states(cfg):
    from .states import zoning_gap_state
    return [zoning_gap_state(cfg, sit, lvl, gap) for sit, lvl, gap in CELLS]


def val_cells():
    rng = np.random.default_rng([96_700, 2])
    return sorted(int(i) for i in rng.choice(len(CELLS), size=N_VAL_CELLS, replace=False))


def semi_states(finals=FINALS, n: int = N_NAT) -> Dict[int, list]:
    from .e3 import semi_natural
    out = {}
    for s in finals:
        sn = semi_natural(s, "train", gaps=("x1.76",))
        rng = np.random.default_rng([96_100, 5, s])
        idx = sorted(rng.choice(len(sn), size=min(n, len(sn)), replace=False)) if sn else []
        out[s] = [sn[i] for i in idx]
    return out


def jobs(finals=FINALS, extended: Dict[str, bool] = None) -> tuple:
    cfg = env_cfg()
    extended = extended or {}
    con = constructed_states(cfg)
    vc = val_cells()
    js = FA.build_jobs("e2", "E2c", {s: con for s in finals}, CANDS, PROTOS, K_CON, H, ROLL_ROOT,
                       {s: vc for s in finals}, critic=True, extended=extended.get("E2c", False))
    col = dict(run_jobs(collect_job, [(s, "hp_edge", N_NAT, E2_STATE_ROOT, 0, 1) for s in finals],
                        min(6, len(finals)), "E2 natural states"))
    nat = {s: [x["start"] for x in col[(s, "hp_edge")]["states"]] for s in finals}
    js += FA.build_jobs("e2", "E2n", nat, CANDS, PROTOS, K_NAT, H, ROLL_ROOT,
                        {s: FA.validation_indices("E2n", len(v)) for s, v in nat.items()},
                        extended=extended.get("E2n", False))
    sn = semi_states(finals)
    semi = {s: [x["start"] for x in v] for s, v in sn.items()}
    js += FA.build_jobs("e2", "E2s", semi, CANDS, PROTOS, K_NAT, H, ROLL_ROOT,
                        {s: FA.validation_indices("E2s", len(v)) for s, v in semi.items()},
                        extended=extended.get("E2s", False))
    desc = {"E2c": {"cells": CELLS, "validation": vc},
            "E2n": {str(s): {"games": col[(s, "hp_edge")]["games"],
                             "states": [{"seed": x["seed"], "turn": x["turn"], "seat": x["seat"]}
                                        for x in col[(s, "hp_edge")]["states"]]} for s in finals},
            "E2s": {str(s): [{k: v for k, v in x.items() if k != "start"} for x in sn[s]] for s in finals}}
    return js, desc


def run(workers: int, finals=FINALS) -> dict:
    t0 = time.perf_counter()
    js, desc = jobs(finals)
    res = FA.run(js, workers, "E2 rollouts")
    out = analyse(res)
    save_json("e2_summary", {"meta": meta(stage="e2", cands=CANDS, protos=PROTOS, horizon=H, cells=CELLS),
                             "states": desc, "analysis": out, "elapsed_s": time.perf_counter() - t0})
    return out


def _sel_gap(gap: str):
    idx = {i for i, c in enumerate(CELLS) if c[2] == gap}
    return lambda r: r["idx"] in idx


def analyse(res: dict) -> dict:
    out = {}
    fams = {"E2c": "G", "E2n": "Q", "E2s": "G"}
    for fam, stat in fams.items():
        other = "Q" if stat == "G" else "G"
        F = {}
        for proto in PROTOS:
            P = {}
            selects = {"all": None}
            if fam == "E2c":
                selects = {gap: _sel_gap(gap) for gap in ("equal", "x1.5", "x1.76", "x2.2")}
            for sel_name, sel in selects.items():
                S = {}
                for c in CANDS:
                    if c == "policy":
                        S["abs:policy"] = FA.absolute(res, fam, proto, c, "G", H, sel)
                        continue
                    S[f"{c}-policy|{stat}"] = FA.contrast(res, fam, proto, c, stat, H, select=sel)
                    S[f"{c}-policy|{other}"] = FA.contrast(res, fam, proto, c, other, H, select=sel)
                    S[f"{c}-policy|G20"] = FA.contrast(res, fam, proto, c, "G", 20, select=sel)
                S[f"threaten_beh-farm_beh|{stat}"] = FA.contrast(res, fam, proto, "threaten_beh", stat, H,
                                                                 base="farm_beh", select=sel)
                S[f"bully_beh-farm_beh|{stat}"] = FA.contrast(res, fam, proto, "bully_beh", stat, H, base="farm_beh",
                                                              select=sel)
                S[f"g_bully_script-bully_beh|{stat}"] = FA.contrast(res, fam, proto, "g_bully_script", stat, H,
                                                                    base="bully_beh", select=sel)
                S[f"g_farm_script-farm_beh|{stat}"] = FA.contrast(res, fam, proto, "g_farm_script", stat, H,
                                                                  base="farm_beh", select=sel)
                # Holm over the four selective plans (p-values of the six finals' means)
                pv = {c: t_pvalue(list(v["mean"] for v in S[f"{c}-policy|{stat}"]["per_final"].values()))
                      for c in PLANS4}
                S["holm"] = {"p": pv, "rejected": holm(pv)}
                # channels at 20 and 50 turns
                ch = {}
                for c in CANDS:
                    if c == "policy":
                        continue
                    for key in ("unc_B", "unc_away_B", "away_B", "deaths_B", "xp_lost_B", "recalls_B", "earned_B",
                                "deaths_A", "d_total_A", "trade_conn_A", "earned_A", "farm_A", "d_total_B",
                                "n_safe_B", "n_trade_A"):
                        for h in (20, 50):
                            ch[f"{c}:{key}@{h}"] = FA.ledger_contrast(res, fam, proto, c, key, h, sel)
                S["channels"] = ch
                # forced turns of each plan (mean over rollouts and states)
                fo = {}
                for c in CANDS:
                    vals = [np.mean(r["forced"][c]) for recs in FA.records(res, fam, proto).values() for r in recs
                            if (sel is None or sel(r))]
                    fo[c] = float(np.mean(vals)) if vals else float("nan")
                S["forced_turns"] = fo
                # absolute channels of the policy (the stronger side's baseline)
                for key in ("earned_A", "earned_B", "away_B", "deaths_B", "unc_B"):
                    vals = {}
                    for s, recs in FA.records(res, fam, proto).items():
                        v = [ledger_mean(r, "policy", key, 50) for r in recs if (sel is None or sel(r))]
                        if v:
                            vals[s] = float(np.mean(v))
                    S[f"abs_ledger:policy:{key}@50"] = {"mean": mean_ci(vals.values())[0], "per_final": vals}
                P[sel_name] = S
            F[proto] = P
        F["validation"] = FA.validate(res, fam, stat, H, list(CANDS))
        out[fam] = F
    return out


__all__ = ["run", "analyse", "jobs", "CANDS", "CELLS", "PLANS4"]


def extend(workers: int, finals=FINALS) -> dict:
    """Exception 3 (pre-registration §1): the constructed family E2c failed the pooled validation, so every cell and
    protocol is re-run at 150 turns (same states and rollout seeds: the first 60 turns are the 60-turn rollouts)."""
    from . import engine as EN
    from .common import load_json
    t0 = time.perf_counter()
    cfg = env_cfg()
    con = constructed_states(cfg)
    js = FA.build_jobs("e2", "E2c", {s: con for s in finals}, CANDS, PROTOS, K_CON, H, ROLL_ROOT, None,
                       critic=True, extended=True)
    res = FA.run(js, workers, "E2c extended")
    out = {}
    for proto in PROTOS:
        P = {}
        for gap in ("equal", "x1.5", "x1.76", "x2.2"):
            sel = _sel_gap(gap)
            S = {}
            for h in (60, 150):
                for c in CANDS:
                    if c != "policy":
                        S[f"{c}-policy|G{h}"] = FA.contrast(res, "E2c", proto, c, "G", h, select=sel)
                S[f"threaten_beh-farm_beh|G{h}"] = FA.contrast(res, "E2c", proto, "threaten_beh", "G", h,
                                                               base="farm_beh", select=sel)
                S[f"g_bully_script-bully_beh|G{h}"] = FA.contrast(res, "E2c", proto, "g_bully_script", "G", h,
                                                                  base="bully_beh", select=sel)
                S[f"g_farm_script-farm_beh|G{h}"] = FA.contrast(res, "E2c", proto, "g_farm_script", "G", h,
                                                                base="farm_beh", select=sel)
            P[gap] = S
        out[proto] = P
    d60, d150 = [], []
    for rec in res.values():
        for c in CANDS:
            if c != "policy":
                d60.append(EN.per_state_diff(rec, c, "G", 60))
                d150.append(EN.per_state_diff(rec, c, "G", 150))
    d60, d150 = np.asarray(d60), np.asarray(d150)
    out["60_vs_150"] = {"pairs": int(len(d60)), "mean_change": float(np.mean(d150 - d60)),
                        "slope": float(np.polyfit(d60, d150, 1)[0])}
    d = load_json("e2_summary")
    d["extended"] = {"meta": meta(stage="e2_extend", horizon=FA.EXTENDED_H), "analysis": out,
                     "elapsed_s": time.perf_counter() - t0}
    save_json("e2_summary", d)
    return out
