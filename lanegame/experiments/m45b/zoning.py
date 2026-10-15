"""Priority D: Safe, Yield and threat-based zoning (XP range 2 kept).

1. `one_turn` (exact): from the zoning situations (`states.ZONING_SITUATIONS`, INITIALIZED) with a power gap, every
   pair (A in {Hold, Trade}, B in {Hold, Yield, Safe}) of Last Hit actions for one turn, by exact enumeration of the
   trade dice: B's farm, XP, HP loss (bars), whether A connected. The PRICE OF STEPPING BACK is B's farm (resources)
   and XP lost by Yield / Safe against Hold for the same A action; the HP SAVED is B's HP loss avoided.
2. `matrix` (multi-turn): the stronger side's strategies ZA (g_farm = Hold, g_bully = Trade whenever catchable, g_guard
   = threat with HP and edge thresholds, g_shove, g_shove_guard) against the weaker side's ZB (g_farm, g_yield,
   g_safe, g_wary = Safe only under a catchable stronger enemy, g_wary_yield... ) from each situation, 100 turns,
   paired. Per cell: G20 / G50 / G100 and B's ledger at 20 / 50 turns (farm, XP, HP lost, deaths, Recalls, turns away,
   evasions), A's connected trades.

   Decomposition used in the report (all in B's resources, paired on seeds):
     passive cost of stepping back      B Safe - B Hold, against A = g_farm (nobody threatens)
     connected combat zoning            B Hold against A = g_bully / g_guard, minus B Hold against A = g_farm
     threat-only zoning                 B wary against A = g_guard (B retreats only under a catchable threat;
                                        A rarely connects), minus B Hold against A = g_farm
     exploitability of always-Safe      A's best G100 against B = g_safe minus its best against B = g_farm
"""

from __future__ import annotations

from typing import List

import numpy as np

from ...config import B_HOLD, B_SAFE, B_TRADE, B_YIELD, W_HOLD, champ_act
from . import common as Cm
from . import engine as E
from . import exact as X
from . import states as S
from . import strategies as ST

A_ACT = {"Hold": champ_act(W_HOLD, B_HOLD), "Trade": champ_act(W_HOLD, B_TRADE)}
B_ACT = {"Hold": champ_act(W_HOLD, B_HOLD), "Yield": champ_act(W_HOLD, B_YIELD), "Safe": champ_act(W_HOLD, B_SAFE)}
ZA = ("g_farm", "g_bully", "g_guard", "g_shove", "g_shove_guard")
ZB = ("g_farm", "g_yield", "g_safe", "g_wary", "g_wary_yield", "g_thin", "g_shove")
SITUATIONS = ("neutral", "a_push", "b_push")


def one_turn(over=(), situations=SITUATIONS, levels=(2, 6, 11), gaps=("equal", "large")) -> List[dict]:
    c = Cm.cfg("r2", **dict(over))
    out = []
    for sit in situations:
        for L in levels:
            for g in gaps:
                lane, m4, k = S.zoning_state(c, sit, L, g)
                cell = {"situation": sit, "level": L, "gap": g, "pairs": {}}
                for an, a in A_ACT.items():
                    for bn, b in B_ACT.items():
                        e = X.expect(c, lane, m4, k, a, b)["mean"]
                        cell["pairs"][f"{an}|{bn}"] = {"farm_B": e["farm"][1], "xp_B": e["xp"][1],
                                                       "hp_B": e["taken_bar"][1], "farm_A": e["farm"][0],
                                                       "xp_A": e["xp"][0], "hp_A": e["taken_bar"][0],
                                                       "conn_A": e["connected"][0], "died_B": e["died"][1]}
                p = cell["pairs"]
                for an in A_ACT:
                    for bn in ("Yield", "Safe"):
                        cell[f"price_{bn}_vs_{an}"] = {
                            "farm": p[f"{an}|Hold"]["farm_B"] - p[f"{an}|{bn}"]["farm_B"],
                            "xp": p[f"{an}|Hold"]["xp_B"] - p[f"{an}|{bn}"]["xp_B"],
                            "hp_saved": p[f"{an}|Hold"]["hp_B"] - p[f"{an}|{bn}"]["hp_B"]}
                out.append(cell)
    return out


B_LEDGER = ("farm_B@20", "xp_B@20", "d_total_B@20", "deaths_B@20", "away_B@20", "farm_B@50", "xp_B@50",
            "d_total_B@50", "deaths_B@50", "farm_A@50", "xp_A@50", "evaded_B", "trade_conn_A", "trade_dealt_A",
            "deaths_B", "recalls_B", "recalls_A", "d_min_A", "unc_B", "xp_lost_B", "n_safe_B", "n_yield_B")


def _cell(args):
    over, sit, level, gap, a, b, seeds, turns = args
    c = Cm.cfg("r2", **dict(over))
    start = S.zoning_state(c, sit, level, gap)
    rows = [E.play(c, start, ST.get(a), ST.get(b), sd, turns)["summary"] for sd in seeds]
    out = {"situation": sit, "level": level, "gap": gap, "a": a, "b": b,
           "per_seed": {m: [r[m] for r in rows] for m in ("G20", "G50", "G100", "farm_B@50", "xp_B@50", "earned_B@50")
                        if m in rows[0]}}
    for k in ("G20", "G50", "G100") + B_LEDGER:
        if k in rows[0]:
            out[k] = float(np.mean([r[k] for r in rows]))
    return out


def matrix(over=(), situations=SITUATIONS, levels=(6, 10), gaps=("large", "equal"), za=ZA, zb=ZB,
           seeds=Cm.DEV[:64], turns: int = 100, workers: int = Cm.WORKERS) -> dict:
    jobs = [(tuple(over), s, L, g, a, b, tuple(seeds), turns) for s in situations for L in levels for g in gaps
            for a in za for b in zb]
    cells = Cm.run_jobs(_cell, jobs, workers, f"zoning matrix {dict(over)}")
    idx = {(c["situation"], c["level"], c["gap"], c["a"], c["b"]): c for c in cells}
    dec = []
    for s in situations:
        for L in levels:
            for g in gaps:
                def ps(a, b, m):
                    return idx[(s, L, g, a, b)]["per_seed"][m]
                d = {"situation": s, "level": L, "gap": g}
                for m in ("earned_B@50", "xp_B@50"):
                    d[f"passive_safe_{m}"] = Cm.paired(ps("g_farm", "g_safe", m), ps("g_farm", "g_farm", m))
                    d[f"passive_yield_{m}"] = Cm.paired(ps("g_farm", "g_yield", m), ps("g_farm", "g_farm", m))
                    d[f"combat_bully_{m}"] = Cm.paired(ps("g_bully", "g_farm", m), ps("g_farm", "g_farm", m))
                    d[f"combat_guard_{m}"] = Cm.paired(ps("g_guard", "g_farm", m), ps("g_farm", "g_farm", m))
                    d[f"threat_wary_{m}"] = Cm.paired(ps("g_guard", "g_wary", m), ps("g_farm", "g_farm", m))
                    d[f"safe_vs_bully_{m}"] = Cm.paired(ps("g_bully", "g_safe", m), ps("g_bully", "g_farm", m))
                    d[f"wary_vs_guard_{m}"] = Cm.paired(ps("g_guard", "g_wary", m), ps("g_guard", "g_farm", m))
                best_vs = {b: max(idx[(s, L, g, a, b)]["G100"] for a in za) for b in zb}
                d["A_best_G100_vs"] = best_vs
                d["exploit_safe"] = best_vs["g_safe"] - best_vs["g_farm"]
                d["B_best_reply_G100"] = {a: min(zb, key=lambda b: idx[(s, L, g, a, b)]["G100"]) for a in za}
                d["threat_conn_A"] = idx[(s, L, g, "g_guard", "g_wary")]["trade_conn_A"]
                d["threat_evaded_B"] = idx[(s, L, g, "g_guard", "g_wary")]["evaded_B"]
                dec.append(d)
    return {"cells": [{k: v for k, v in c.items() if k != "per_seed"} for c in cells], "decomposition": dec}


__all__ = ["one_turn", "matrix", "ZA", "ZB", "SITUATIONS"]
