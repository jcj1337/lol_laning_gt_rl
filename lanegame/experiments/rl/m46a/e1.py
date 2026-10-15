"""E1: the exact one-turn ledger of contesting (pre-registration §2 E1). No learning, no rollouts: every listed action
pair from every constructed cell, expectations over the turn's chance branches with ganks enumerated
(`m45b.exact.turn_distribution`).

Per (cell, A action, B action), per side: earned gold, farm, XP, HP lost (bars of the victim's start-of-turn max HP,
after durability), trade damage dealt (bars), minion fire taken (bars) split by cause in proportion to the units of each
cause, P(death), and for the freeze cells P(the freeze still holds after the turn), P(B's wave crashed into A's tower),
P(A's front is gone), E[A's front HP].

Decomposition rows (COUNTERFACTUAL configurations of existing fields, one trading cost removed at a time; labelled,
never candidates): no_farm_cost (trade_farm_factor 1.0), no_wave_cap (trade_wave_fraction 1.0), no_retaliation,
no_positional (minion_positional_aggro off), no_trade_gank (the Trade gank term equal to Hold's). Evaluated for the
guard / keep / Last Hit + Trade against the break and the walk-up in the freeze cells, and for A's Trade and Hold
against B's Hold, Safe and Yield in the zoning cells.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from ....config import B_HOLD, B_SAFE, B_TRADE, B_YIELD, W_HOLD, W_SHOVE, W_THIN, champ_act
from .common import env_cfg, meta, run_jobs, save_json, setup_worker

LH, TN, SH = W_HOLD, W_THIN, W_SHOVE
FREEZE_A = {"keep": champ_act(TN, B_HOLD), "guard": champ_act(TN, B_TRADE), "lh_hold": champ_act(LH, B_HOLD),
            "lh_trade": champ_act(LH, B_TRADE), "break_own": champ_act(SH, B_HOLD)}
FREEZE_B = {"break": champ_act(SH, B_HOLD), "walk_up": champ_act(LH, B_HOLD), "step_back": champ_act(LH, B_SAFE),
            "shove_safe": champ_act(SH, B_SAFE), "yield": champ_act(LH, B_YIELD), "contest": champ_act(SH, B_TRADE)}
ZONE_A = {"lh_hold": champ_act(LH, B_HOLD), "lh_trade": champ_act(LH, B_TRADE), "sh_hold": champ_act(SH, B_HOLD),
          "sh_trade": champ_act(SH, B_TRADE)}
ZONE_B = {"lh_hold": champ_act(LH, B_HOLD), "yield": champ_act(LH, B_YIELD), "safe": champ_act(LH, B_SAFE),
          "trade": champ_act(LH, B_TRADE), "sh_hold": champ_act(SH, B_HOLD)}
FREEZE_CELLS = [(lvl, gap, held) for lvl in (4, 10, 15) for gap in ("equal", "x1.76") for held in (5, 12, 20)]
ZONE_CELLS = [(sit, lvl, gap) for sit in ("neutral", "a_push", "b_push") for lvl in (6, 10) for gap in ("equal", "x1.76")]
VARIANTS = {"baseline": {}, "no_farm_cost": {"trade_farm_factor": 1.0}, "no_wave_cap": {"trade_wave_fraction": 1.0},
            "no_retaliation": {"minion_retaliation": False}, "no_positional": {"minion_positional_aggro": False},
            "no_trade_gank": {"champ_gank_behavior": (0.01, 0.01, 0.01, 0.0)}}
DECOMP_FREEZE_A = ("keep", "guard", "lh_trade", "lh_hold")
DECOMP_FREEZE_B = ("break", "walk_up")
DECOMP_ZONE_A = ("lh_hold", "lh_trade")
DECOMP_ZONE_B = ("lh_hold", "safe", "yield")
KEYS = ("earned", "farm", "xp", "taken_bar", "dealt_bar", "minion_bar", "min_ret_bar", "min_pos_bar", "died",
        "connected", "landed")


def config_variant(name: str):
    from ....config import GameConfig
    cfg = GameConfig.m45_calibrated(**VARIANTS[name])
    cfg.validate()
    return cfg


def _cell_state(cfg, kind: str, cell):
    from . import states as ST
    if kind == "freeze":
        lvl, gap, held = cell
        return ST.freeze_gap_state(cfg, lvl, gap, held)
    sit, lvl, gap = cell
    return ST.zoning_gap_state(cfg, sit, lvl, gap)


def pair_expectation(cfg, start, a: int, b: int) -> Dict[str, List[float]]:
    from ..m44.records import holds_freeze
    from ...m45b.exact import outcome, turn_distribution
    lane, m4, k = start
    dist = turn_distribution(cfg, lane, m4, k, a, b, ganks="on")
    acc = {kk: [0.0, 0.0] for kk in KEYS}
    extra = {"freeze_holds": 0.0, "b_crash": 0.0, "a_front_gone": 0.0, "a_front_hp": 0.0, "b_front_hp": 0.0}
    for p, t in dist:
        o = outcome(t, cfg)
        for kk in ("earned", "farm", "xp", "taken_bar", "dealt_bar", "minion_bar", "died", "connected", "landed"):
            for s in (0, 1):
                acc[kk][s] += p * o[kk][s]
        for s in (0, 1):
            u = o["u_ret"][s] + o["u_pos"][s]
            if u > 0:
                acc["min_ret_bar"][s] += p * o["minion_bar"][s] * o["u_ret"][s] / u
                acc["min_pos_bar"][s] += p * o["minion_bar"][s] * o["u_pos"][s] / u
        after = t.lane.state
        extra["freeze_holds"] += p * float(holds_freeze(after, 0))
        extra["b_crash"] += p * float(t.lane.crash_hp[1] > 0)
        fx, fh = after.side(0)[:2]
        gone = fh <= 0 or fx < lane.side(0)[0]         # no front, or the leading packet is a fresh one behind it
        extra["a_front_gone"] += p * float(gone)
        extra["a_front_hp"] += p * float(fh if not gone else 0)
        extra["b_front_hp"] += p * float(after.side(1)[1])
    return {**{k_: v for k_, v in acc.items()}, **{k_: v for k_, v in extra.items()}, "branches": len(dist)}


def cell_job(args) -> tuple:
    """(kind, cell, variant, a_names, b_names): every legal pair's expectation at one cell under one configuration."""
    kind, cell, variant, a_names, b_names = args
    setup_worker()
    from ....champions import champion_legal_actions
    cfg = config_variant(variant)
    start = _cell_state(cfg, kind, cell)
    A_ACT, B_ACT = (FREEZE_A, FREEZE_B) if kind == "freeze" else (ZONE_A, ZONE_B)
    la = set(champion_legal_actions(start[0], start[1], 0, cfg))
    lb = set(champion_legal_actions(start[0], start[1], 1, cfg))
    out = {}
    t0 = time.perf_counter()
    for an in a_names:
        for bn in b_names:
            a, b = A_ACT[an], B_ACT[bn]
            if a not in la or b not in lb:
                out[f"{an}|{bn}"] = None
                continue
            out[f"{an}|{bn}"] = pair_expectation(cfg, start, a, b)
    return (kind, tuple(cell), variant), {"pairs": out, "elapsed_s": time.perf_counter() - t0}


def run(workers: int) -> dict:
    t0 = time.perf_counter()
    js = []
    for cell in FREEZE_CELLS:
        js.append(("freeze", cell, "baseline", tuple(FREEZE_A), tuple(FREEZE_B)))
        for v in VARIANTS:
            if v != "baseline":
                js.append(("freeze", cell, v, DECOMP_FREEZE_A, DECOMP_FREEZE_B))
    for cell in ZONE_CELLS:
        js.append(("zone", cell, "baseline", tuple(ZONE_A), tuple(ZONE_B)))
        for v in VARIANTS:
            if v != "baseline":
                js.append(("zone", cell, v, DECOMP_ZONE_A, DECOMP_ZONE_B))
    res = run_jobs(cell_job, js, workers, "E1 exact")
    data = {"meta": meta(stage="e1", freeze_cells=FREEZE_CELLS, zone_cells=ZONE_CELLS, variants=VARIANTS,
                         actions={"freeze_A": FREEZE_A, "freeze_B": FREEZE_B, "zone_A": ZONE_A, "zone_B": ZONE_B}),
            "cells": [{"kind": k[0], "cell": list(k[1]), "variant": k[2], **v} for k, v in res],
            "elapsed_s": time.perf_counter() - t0}
    save_json("e1_exact", data)
    return data


# ---------------------------------------------------------------------------------------------- analysis
def priced(e: dict, price: float, side: int = 0) -> float:
    """HP-priced immediate value to `side`: (earned_s - earned_o) - price (taken_s - taken_o) in bars."""
    o = 1 - side
    return (e["earned"][side] - e["earned"][o]) - price * (e["taken_bar"][side] - e["taken_bar"][o])


def analyse(data: dict, prices=(1.0, 1.5, 2.0), e5_price: float = None) -> dict:
    """E1-P1 and E1-P2 tables (see the module docstring)."""
    base = {(c["kind"], tuple(c["cell"])): c["pairs"] for c in data["cells"] if c["variant"] == "baseline"}
    var = {(c["kind"], tuple(c["cell"]), c["variant"]): c["pairs"] for c in data["cells"]}
    ps = list(prices) + ([e5_price] if e5_price else [])
    out = {"freeze": [], "zone": [], "load_bearing": {}}
    for (kind, cell), pairs in base.items():
        if kind != "freeze":
            continue
        brk_k, brk_g = pairs.get("keep|break"), pairs.get("guard|break")
        wu_k, wu_g = pairs.get("keep|walk_up"), pairs.get("guard|walk_up")
        row = {"cell": list(cell)}
        if brk_k and brk_g:
            row.update({
                "P_hold_after_break_vs_keep": brk_k["freeze_holds"], "P_hold_after_break_vs_guard": brk_g["freeze_holds"],
                "P_hold_after_walkup_vs_keep": wu_k["freeze_holds"] if wu_k else None,
                "P_a_front_gone_break": brk_k["a_front_gone"],
                "B_taken_bar_break_vs_guard": brk_g["taken_bar"][1], "B_taken_bar_break_vs_keep": brk_k["taken_bar"][1],
                "A_taken_bar_guard_vs_break": brk_g["taken_bar"][0],
                "A_min_pos_bar_guard": brk_g["min_pos_bar"][0], "A_min_ret_bar_guard": brk_g["min_ret_bar"][0],
                "B_lead_break_vs_keep": brk_k["earned"][1] - brk_k["earned"][0],
                "B_lead_break_vs_guard": brk_g["earned"][1] - brk_g["earned"][0],
                "P_B_death_break_vs_guard": brk_g["died"][1]})
            for p in ps:
                # B's HP-priced value of breaking vs walking up / stepping back, against the guard and the keep
                for an in ("keep", "guard"):
                    for bn in ("walk_up", "step_back", "yield"):
                        e2 = pairs.get(f"{an}|{bn}")
                        e1 = pairs.get(f"{an}|break")
                        if e1 and e2:
                            row[f"B_break_minus_{bn}_vs_{an}@{p:g}"] = priced(e1, p, 1) - priced(e2, p, 1)
                # A's guard vs keep against the break
                row[f"A_guard_minus_keep_vs_break@{p:g}"] = priced(brk_g, p, 0) - priced(brk_k, p, 0)
                if wu_k and wu_g:
                    row[f"A_guard_minus_keep_vs_walkup@{p:g}"] = priced(wu_g, p, 0) - priced(wu_k, p, 0)
        out["freeze"].append(row)
    for (kind, cell), pairs in base.items():
        if kind != "zone":
            continue
        row = {"cell": list(cell)}
        for p in ps:
            for bn in ("lh_hold", "safe", "yield", "trade", "sh_hold"):
                t_, h_ = pairs.get(f"lh_trade|{bn}"), pairs.get(f"lh_hold|{bn}")
                if t_ and h_:
                    row[f"A_trade_minus_hold_vs_{bn}@{p:g}"] = priced(t_, p, 0) - priced(h_, p, 0)
            t_, h_, s_ = pairs.get("lh_trade|lh_hold"), pairs.get("lh_trade|lh_hold"), pairs.get("lh_trade|safe")
            if t_ and s_:
                row[f"B_safe_minus_hold_vs_trade@{p:g}"] = priced(s_, p, 1) - priced(t_, p, 1)
            h2, s2 = pairs.get("lh_hold|lh_hold"), pairs.get("lh_hold|safe")
            if h2 and s2:
                row[f"B_safe_minus_hold_vs_hold@{p:g}"] = priced(s2, p, 1) - priced(h2, p, 1)
        t_ = pairs.get("lh_trade|lh_hold")
        if t_:
            row.update({"A_trade_landed": t_["landed"][0], "A_dealt_bar": t_["dealt_bar"][0],
                        "A_taken_bar": t_["taken_bar"][0], "A_min_ret_bar": t_["min_ret_bar"][0],
                        "A_min_pos_bar": t_["min_pos_bar"][0], "B_taken_bar": t_["taken_bar"][1],
                        "A_farm_trade": t_["farm"][0], "A_farm_hold": (pairs.get("lh_hold|lh_hold") or {}).get(
                            "farm", [np.nan])[0]})
        out["zone"].append(row)
    # E1-P2: sign changes of the guard's and the threat's priced value when one cost is removed (price 1.5)
    for p in ps:
        lb = {}
        for v in VARIANTS:
            if v == "baseline":
                continue
            fz, zn = [], []
            for (kind, cell), pairs in base.items():
                vp = var.get((kind, cell, v))
                if vp is None:
                    continue
                if kind == "freeze":
                    for bn in DECOMP_FREEZE_B:
                        g0, k0_ = pairs.get(f"guard|{bn}"), pairs.get(f"keep|{bn}")
                        g1, k1 = vp.get(f"guard|{bn}"), vp.get(f"keep|{bn}")
                        if g0 and k0_ and g1 and k1:
                            a0 = priced(g0, p) - priced(k0_, p)
                            a1 = priced(g1, p) - priced(k1, p)
                            fz.append((a0, a1))
                else:
                    t0_, h0 = pairs.get("lh_trade|lh_hold"), pairs.get("lh_hold|lh_hold")
                    t1, h1 = vp.get("lh_trade|lh_hold"), vp.get("lh_hold|lh_hold")
                    if t0_ and h0 and t1 and h1:
                        zn.append((priced(t0_, p) - priced(h0, p), priced(t1, p) - priced(h1, p)))
            lb[v] = {"freeze_guard": {"n": len(fz), "sign_changes": int(sum(np.sign(a) != np.sign(b) for a, b in fz)),
                                      "mean_baseline": float(np.mean([a for a, _ in fz])) if fz else None,
                                      "mean_variant": float(np.mean([b for _, b in fz])) if fz else None},
                     "zone_threat": {"n": len(zn), "sign_changes": int(sum(np.sign(a) != np.sign(b) for a, b in zn)),
                                     "mean_baseline": float(np.mean([a for a, _ in zn])) if zn else None,
                                     "mean_variant": float(np.mean([b for _, b in zn])) if zn else None}}
            for key in ("freeze_guard", "zone_threat"):
                n = lb[v][key]["n"]
                lb[v][key]["load_bearing"] = bool(n and lb[v][key]["sign_changes"] >= n / 2)
        out["load_bearing"][f"{p:g}"] = lb
    return out


__all__ = ["run", "analyse", "pair_expectation", "FREEZE_CELLS", "ZONE_CELLS", "VARIANTS", "priced"]
