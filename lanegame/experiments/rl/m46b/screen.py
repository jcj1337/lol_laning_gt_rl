"""The M4.6b candidate screens S1-S6 (docs/m4/m4_6/m4_6b_preregistration.md §3). Every arm is an opt-in configuration
(curves.ARMS); the six M4.5c 16.384M finals play every arm (trained on the baseline: out of distribution under a
candidate). Statistic: G_h, the focal seat's discounted gold lead (the objective truncated at h); no critic.

Stages: early (level 8, S 9), mid (16, 42), late (18, 108); leads +8 / +16 resources (20 / 40 turns of the measured
income 0.40) added to one side's investment. The same states and seeds are used for every arm (paired across arms).
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from ....config import B_HOLD, B_SAFE, B_TRADE, W_HOLD, W_SHOVE, W_THIN, champ_act
from . import common as C
from .curves import ARM_NAMES, ARMS

STAGES = {"early": (8, 9.0), "mid": (16, 42.0), "late": (18, 108.0)}
LEADS = (0.0, 8.0, 16.0)
SITS = ("neutral", "a_push", "b_push")
K_ROLL, H_ROLL, PLAN_N, H_VAL = 16, 50, 10, 150
STRONG_CANDS = {"policy": ("policy",), "threaten": ("beh", "threaten", PLAN_N, 0.4), "hold": ("beh", "farm", PLAN_N, 0.4)}
WEAK_CANDS = {"policy": ("policy",), "hold": ("beh", "farm", PLAN_N, 0.4), "safe": ("beh", "safe", PLAN_N, 0.4),
              "trade": ("beh", "trade", PLAN_N, 0.4), "reset": ("reset", PLAN_N)}
FREEZER_CANDS = {"policy": ("policy",), "keep": ("hz", ("fixed", champ_act(W_THIN, B_HOLD), PLAN_N, "held")),
                 "guard": ("hz", ("fixed", champ_act(W_THIN, B_TRADE), PLAN_N, "held"))}
BREAKER_CANDS = {"policy": ("policy",), "break": ("fixed", champ_act(W_SHOVE, B_HOLD), 1),
                 "walk_up": ("beh", "farm", PLAN_N, 0.4), "step_back": ("beh", "safe", PLAN_N, 0.4),
                 "reset": ("reset", PLAN_N)}
S4_LEVELS, S4_HELD = (4, 10), (5, 12)
S4_INV = ("equal", "freezer+8", "breaker+8")
S2_FREEZE_LEVELS, S2_HELD = (4, 10, 15), (5, 12, 20)


def _arms(arms):
    return tuple(arms) if arms else ARM_NAMES


# ---------------------------------------------------------------------------------------------- states
def zone_state(cfg, sit: str, level: int, s_a: float, s_b: float, k: int = 2):
    from ...m45b import states as S
    lane, pa, pb, _ = S.ZONING_SITUATIONS[sit]
    return lane, S.champs(cfg, (level, level), (s_a, s_b), (pa, pb)), k


def freeze_state(cfg, level: int, held: int, s_a: float, s_b: float):
    from ...m45b import states as S
    lane, m4, k = S.freeze_state(cfg, level=level, gap="equal", held=held)
    a, b = m4.a, m4.b
    m4 = m4._replace(a=a._replace(items=int(round(s_a * cfg.gold_scale))),
                     b=b._replace(items=int(round(s_b * cfg.gold_scale))))
    return lane, m4, k


def freeze_inv(level: int, inv: str):
    from ...m45b.states import typical_invest
    base = typical_invest(level)
    return {"equal": (base, base), "freezer+8": (base + 8, base), "breaker+8": (base, base + 8)}[inv]


# ---------------------------------------------------------------------------------------------- S1
def s1_job(arm: str) -> tuple:
    cfg = C.setup(arm)
    from ....champions_m4 import context
    from ...m45b.states import TYPICAL_INVEST
    ctx = context(cfg)
    tb = ctx.t
    sc = cfg.gold_scale
    h, q = 0.55, 0.3

    def M(S):
        return ctx.inv.factor(int(round(S * sc)))

    out = {"stages": {}, "curve": {}, "capacity_by_level": {}}
    for S in (0, 2, 5, 9, 15, 22, 35, 42, 56, 76, 108, 116, 154, 234, 500):
        el = 0.0 if S == 0 else (math.log(M(S * 1.01)) - math.log(M(S * 0.99))) / (math.log(1.01) - math.log(0.99))
        out["curve"][str(S)] = {"M": M(S), "elasticity": el,
                                "dlnM_per_res": (math.log(M(S + 1)) - math.log(M(S)))}
    for stage, (L, S0) in STAGES.items():
        u0 = int(round(S0 * sc))
        rows = {}
        for d in (4.0, 8.0, 16.0):
            u = int(round((S0 + d) * sc))
            mA, mB = ctx.inv.factor(u), ctx.inv.factor(u0)
            dmg = lambda att, vic: h * ((1 - q) * ctx.hit(1, L, att) + q * ctx.hit(2, L, att)) / (tb.hp_max[L] * ctx.inv.factor(vic))
            rows[str(d)] = {"M_ratio": mA / mB, "duel_ratio": (mA / mB) ** 2,
                            "capacity_leader_trailer": [ctx.capacity(L, u), ctx.capacity(L, u0)],
                            "pressure_leader_trailer": [ctx.pressure(L, u), ctx.pressure(L, u0)],
                            "exchange_bars_leader_on_trailer": dmg(u, u0), "exchange_bars_trailer_on_leader": dmg(u0, u)}
        out["stages"][stage] = {"level": L, "S": S0, "M": ctx.inv.factor(u0), "capacity": ctx.capacity(L, u0),
                                "pressure": ctx.pressure(L, u0),
                                "pressure_turns_per_plate": 2000.0 / max(ctx.pressure(L, u0), 1),
                                "exchange_bars_equal": dmg(u0, u0), "leads": rows}
    for L, S in sorted(TYPICAL_INVEST.items()):
        out["capacity_by_level"][str(L)] = {"S": S, "C": ctx.capacity(L, int(round(S * sc))),
                                            "C_zero": ctx.capacity(L, 0)}
    from ....env.discounted import max_champion_wave_budget, per_turn_bound, tail_constant
    out["bounds"] = {"max_wave_budget": max_champion_wave_budget(cfg), "per_turn_bound": per_turn_bound(cfg),
                     "tail_constant": tail_constant(cfg)}
    return arm, out


def screen_s1(workers: int, arms=None) -> dict:
    res = dict(C.run_jobs(s1_job, list(_arms(arms)), min(workers, 4), "S1 mapping"))
    C.save_json("s1_mapping", {"meta": C.meta(stage="s1"), "arms": {a: ARMS[a] for a in res}, "results": res})
    return res


# ---------------------------------------------------------------------------------------------- S2
def s2_job(args) -> tuple:
    arm, kind, cell = args
    cfg = C.setup(arm)
    from ....champions import champion_legal_actions
    from ..m46a.e1 import FREEZE_A, FREEZE_B, ZONE_A, ZONE_B, pair_expectation
    if kind == "zone":
        sit, stage, d = cell
        L, S0 = STAGES[stage]
        st = zone_state(cfg, sit, L, S0 + d, S0)
        A_ACT, B_ACT = ZONE_A, ZONE_B
        an, bn = ("lh_hold", "lh_trade", "sh_hold"), ("lh_hold", "safe", "trade", "sh_hold")
    else:
        L, held, inv = cell
        st = freeze_state(cfg, L, held, *freeze_inv(L, inv))
        A_ACT, B_ACT = FREEZE_A, FREEZE_B
        an, bn = ("keep", "guard", "lh_hold"), ("break", "walk_up", "step_back")
    la = set(champion_legal_actions(st[0], st[1], 0, cfg))
    lb = set(champion_legal_actions(st[0], st[1], 1, cfg))
    out = {}
    for a in an:
        for b in bn:
            if A_ACT[a] in la and B_ACT[b] in lb:
                out[f"{a}|{b}"] = pair_expectation(cfg, st, A_ACT[a], B_ACT[b])
    return (arm, kind, tuple(cell)), out


def screen_s2(workers: int, arms=None) -> dict:
    js = []
    for arm in _arms(arms):
        js += [(arm, "zone", (sit, st, d)) for sit in SITS for st in STAGES for d in LEADS]
        js += [(arm, "freeze", (L, h, inv)) for L in S2_FREEZE_LEVELS for h in S2_HELD for inv in S4_INV]
    res = C.run_jobs(s2_job, js, workers, "S2 exact")
    data = {"meta": C.meta(stage="s2"), "cells": [{"arm": k[0], "kind": k[1], "cell": list(k[2]), "pairs": v}
                                                   for k, v in res]}
    C.save_json("s2_exact", data)
    return data


# ---------------------------------------------------------------------------------------------- paired jobs
def _job(arm, fam, final, idx, start, cands, horizon, label):
    from ..m46a import engine as EN
    path = os.path.join(C.CKPT, "screen", fam, arm, f"s{final}", f"{idx}__h{horizon}.json.gz")
    return {"spec": C.spec(final), "family": fam, "idx": idx, "start": start, "proto": "policy", "horizon": horizon,
            "roll": EN.roll_seeds(C.ROLL_ROOT, K_ROLL), "cands": cands, "critic": False, "tag": f"s{final}",
            "arm": arm, "path": path, "label": label}


def s3_cells():
    """[(idx, focal, sit, stage, lead)] in a fixed order."""
    out = []
    for sit in SITS:
        for st in STAGES:
            for d in LEADS:
                out.append(("strong", sit, st, d))
            for d in LEADS[1:]:
                out.append(("weak", sit, st, d))
    return list(enumerate(out))


def s3_val_indices():
    strong = [i for i, c in s3_cells() if c[0] == "strong"]
    rng = np.random.default_rng([C.ROLL_ROOT, 3])
    return sorted(int(x) for x in rng.choice(strong, size=6, replace=False))


def s3_jobs(arms=None, finals=C.FINALS) -> list:
    js = []
    val = set(s3_val_indices())
    for arm in _arms(arms):
        cfg = C.arm_cfg(arm)
        for i, (focal, sit, st, d) in s3_cells():
            L, S0 = STAGES[st]
            if focal == "strong":
                start, cands = zone_state(cfg, sit, L, S0 + d, S0), STRONG_CANDS
            else:
                start, cands = zone_state(cfg, sit, L, S0, S0 + d), WEAK_CANDS
            for f in finals:
                js.append(_job(arm, "S3", f, i, start, cands, H_ROLL, f"{focal} {sit} {st} +{d:g}"))
                if i in val:
                    js.append(_job(arm, "S3", f, i, start, cands, H_VAL, f"{focal} {sit} {st} +{d:g} (validation)"))
    return js


def s4_cells():
    out = []
    for L in S4_LEVELS:
        for h in S4_HELD:
            for inv in S4_INV:
                out.append(("freezer", L, h, inv))
                out.append(("breaker", L, h, inv))
    return list(enumerate(out))


def s4_jobs(arms=None, finals=C.FINALS) -> list:
    from .common import state_to_list  # noqa: F401  (kept for symmetry with the other builders)
    js = []
    for arm in _arms(arms):
        cfg = C.arm_cfg(arm)
        for i, (focal, L, h, inv) in s4_cells():
            st = freeze_state(cfg, L, h, *freeze_inv(L, inv))
            if focal == "breaker":
                lane, m4, k = st
                st, cands = (lane.mirror(), m4.mirror(), k), BREAKER_CANDS
            else:
                cands = FREEZER_CANDS
            for f in finals:
                js.append(_job(arm, "S4", f, i, st, cands, H_ROLL, f"{focal} L{L} held {h} {inv}"))
    return js


def s5_states(n: int = 24) -> Dict[str, Dict[int, list]]:
    from ..m46a import e5 as E5
    col = E5.collect(6)
    return {fam: {s: [x["start"] for x in col[(s, fam)]["states"][:n]] for s in C.FINALS}
            for fam in ("recall_full", "recall_mid")}


def s5_jobs(arms=None, states=None) -> list:
    from ..m46a.e5 import RECALL_CANDS
    from ..m46a import engine as EN
    states = states or s5_states()
    js = []
    for arm in _arms(arms):
        for fam, by in states.items():
            for f, sts in by.items():
                for i, st in enumerate(sts):
                    j = _job(arm, f"S5_{fam}", f, i, st, RECALL_CANDS, H_ROLL, f"{fam} {i}")
                    j["roll"] = EN.roll_seeds(C.ROLL_ROOT, 12)
                    js.append(j)
    return js


def _run(js: list, workers: int, label: str) -> Dict[str, dict]:
    from ..m46a import engine as EN
    t0 = time.perf_counter()
    res = C.run_jobs(EN.paired_job, js, workers, label)
    print(f"  [{label}] {len(js)} jobs in {time.perf_counter() - t0:.0f} s", flush=True)
    return {k + "|" + j["arm"]: v for (k, v), j in zip(res, js)}


def screen_s3(workers: int, arms=None) -> dict:
    res = _run(s3_jobs(arms), workers, "S3 rollouts")
    out = analyse_s3(res)
    C.save_json("s3_summary", {"meta": C.meta(stage="s3", K=K_ROLL, horizon=H_ROLL, plan_turns=PLAN_N),
                               "cells": [list(c) for _, c in s3_cells()], "validation_cells": s3_val_indices(),
                               "analysis": out})
    return out


def screen_s4(workers: int, arms=None) -> dict:
    res = _run(s4_jobs(arms), workers, "S4 rollouts")
    out = analyse_s4(res)
    C.save_json("s4_summary", {"meta": C.meta(stage="s4", K=K_ROLL, horizon=H_ROLL),
                               "cells": [list(c) for _, c in s4_cells()], "analysis": out})
    return out


def screen_s5(workers: int, arms=None) -> dict:
    res = _run(s5_jobs(arms), workers, "S5 rollouts")
    out = analyse_s5(res)
    C.save_json("s5_summary", {"meta": C.meta(stage="s5", K=12, horizon=H_ROLL), "analysis": out})
    return out


# ---------------------------------------------------------------------------------------------- S6
S6_GAMES, S6_TURNS = 16, 600


def screen_s6(workers: int, arms=None) -> dict:
    from . import audit as AU
    js = [(f, S6_GAMES, arm, C.NATURAL_ROOT, S6_TURNS) for arm in _arms(arms) for f in C.FINALS]
    res = C.run_jobs(AU.trajectories_job, js, workers, "S6 natural starts")
    by_arm: Dict[str, Dict[int, dict]] = {}
    for (f, d), j in zip(res, js):
        by_arm.setdefault(j[2], {})[f] = d
    out = {arm: analyse_s6(arm, data) for arm, data in by_arm.items()}
    C.save_json("s6_summary", {"meta": C.meta(stage="s6", games=S6_GAMES, turns=S6_TURNS, root=C.NATURAL_ROOT),
                               "analysis": out})
    return out


def analyse_s6(arm: str, data: Dict[int, dict]) -> dict:
    from . import audit as AU
    from ....champions_m4 import context
    cfg = C.arm_cfg(arm)
    ctx = context(cfg)
    Mf = lambda S: ctx.inv.factor(int(round(float(S) * cfg.gold_scale)))
    a = AU.analyse(data, Mf)
    # ratio distribution on present turns: attack ratio (levels once) and duel ratio
    att, duel, per_final = [], [], {}
    lane = {k: {"crash_hp": [], "plates150": [], "plates300": [], "deaths": [], "recalls": [], "collection": [],
                "shove": [], "trade_conn": [], "cap_mid": [], "cap_late": [], "lead300_on_100": []} for k in data}
    for f, d in data.items():
        for r in d["records"]:
            S, L, P = np.asarray(r["S"]), np.asarray(r["L"]), np.asarray(r["present"])
            both = (P[0] > 0) & (P[1] > 0)
            mA = np.array([Mf(x) for x in S[0][both]])
            mB = np.array([Mf(x) for x in S[1][both]])
            lvl = (L[0][both] - L[1][both]) * math.log(1.05)
            ar = np.exp(lvl) * mA / mB
            att += list(np.maximum(ar, 1 / ar))
            dr = np.exp(2 * lvl) * (mA / mB) ** 2
            duel += list(np.maximum(dr, 1 / dr))
            ln = r["lane"]
            x = lane[f]
            T = len(ln["crash_hp"][0])
            for s in (0, 1):
                x["crash_hp"].append(sum(ln["crash_hp"][s]) / T * 100)
                x["plates150"].append(sum(ln["plates"][s][:150]))
                x["plates300"].append(sum(ln["plates"][s][:300]))
                x["deaths"].append(sum(ln["deaths"][s]) / T * 100)
                x["recalls"].append(sum(ln["recalls"][s]) / T * 100)
                fa = sum(ln["farm_avail"][s])
                x["collection"].append(sum(ln["farm"][s]) / fa if fa > 0 else float("nan"))
                wt = sum(ln["wave_turns"][s])
                x["shove"].append(sum(ln["shove"][s]) / wt if wt else float("nan"))
                x["trade_conn"].append(sum(ln["trade_conn"][s]) / T * 100)
                x["cap_mid"].append(float(np.mean(ln["cap"][s][89:150])))
                x["cap_late"].append(float(np.mean(ln["cap"][s][239:320])))
        # D2's persistence slope: the signed earned lead at turn 300 regressed on the lead at turn 100, per final
        dE = np.array([np.asarray(r["E"][0]) - np.asarray(r["E"][1]) for r in d["records"]])
        if dE.shape[1] >= 300 and dE[:, 99].std() > 0:
            lane[f]["lead300_on_100"].append(float(np.polyfit(dE[:, 99], dE[:, 299], 1)[0]))
    att, duel = np.asarray(att), np.asarray(duel)
    out = {"audit": a,
           "ratios": {"attack_q": {str(q): float(np.quantile(att, q)) for q in (0.5, 0.9, 0.99)},
                      "duel_q": {str(q): float(np.quantile(duel, q)) for q in (0.5, 0.9, 0.99)},
                      "attack_share_1.1": float((att >= 1.1).mean()), "attack_share_1.25": float((att >= 1.25).mean()),
                      "duel_share_1.25": float((duel >= 1.25).mean()), "duel_share_1.5": float((duel >= 1.5).mean())},
           "lane": {}}
    for k in next(iter(lane.values())):
        per = [float(np.nanmean(v[k])) for v in lane.values() if v[k]]
        m, h = C.mean_ci(per)
        out["lane"][k] = {"mean": m, "ci": h, "per_final": per}
    # the same slope pooled over all games of the arm (the six finals' games together)
    dE = np.array([np.asarray(r["E"][0]) - np.asarray(r["E"][1]) for d in data.values() for r in d["records"]])
    if dE.shape[1] >= 300:
        out["lane"]["lead300_on_100"]["pooled"] = float(np.polyfit(dE[:, 99], dE[:, 299], 1)[0])
    return out


# ---------------------------------------------------------------------------------------------- analysis
def _records(res: Dict[str, dict], fam: str, arm: str, horizon: Optional[int] = None):
    out: Dict[int, Dict[int, dict]] = {}
    for rec in res.values():
        if rec["family"] != fam or rec.get("arm") != arm:
            continue
        if horizon is not None and rec["horizon"] != horizon:
            continue
        f = int(rec["tag"][1:])
        prev = out.setdefault(f, {}).get(rec["idx"])
        if prev is None or rec["horizon"] < prev["horizon"]:          # prefer the 50-turn job
            out[f][rec["idx"]] = rec
    return out


def _g(rec, cand, h):
    return np.asarray(rec["G"][cand][str(h)])


def analyse_s3(res: Dict[str, dict]) -> dict:
    from ..m46a import engine as EN
    cells = dict(s3_cells())
    arms = sorted({r.get("arm") for r in res.values()}, key=ARM_NAMES.index)
    out = {"premium": {}, "conversion": {}, "weak": {}, "channels": {}, "validation": {}}
    prem = {}
    for arm in arms:
        R = _records(res, "S3", arm)
        for st in STAGES:
            for d in LEADS[1:]:
                per_final = {}
                for f, by in R.items():
                    vals = []
                    for sit in SITS:
                        i_lead = next(i for i, c in cells.items() if c == ("strong", sit, st, d))
                        i_eq = next(i for i, c in cells.items() if c == ("strong", sit, st, 0.0))
                        if i_lead in by and i_eq in by:
                            vals.append(float(np.mean(_g(by[i_lead], "policy", H_ROLL) - _g(by[i_eq], "policy", H_ROLL))))
                    if vals:
                        per_final[f] = float(np.mean(vals))
                prem[(arm, st, d)] = per_final
                m, h = C.mean_ci(per_final.values())
                out["premium"][f"{arm}|{st}|+{d:g}"] = {"mean": m, "ci": h, "per_final": per_final}
        # conversion and weak-side responses
        for st in STAGES:
            for d in LEADS:
                vals = {}
                for f, by in R.items():
                    v = [EN.per_state_diff(by[i], "threaten", "G", H_ROLL, "hold") for i, c in cells.items()
                         if c[0] == "strong" and c[2] == st and c[3] == d and i in by]
                    if v:
                        vals[f] = float(np.mean(v))
                m, h = C.mean_ci(vals.values())
                out["conversion"][f"{arm}|{st}|+{d:g}|threaten-hold"] = {"mean": m, "ci": h, "per_final": vals}
                for cand in ("threaten", "hold"):
                    vals = {}
                    for f, by in R.items():
                        v = [EN.per_state_diff(by[i], cand, "G", H_ROLL) for i, c in cells.items()
                             if c[0] == "strong" and c[2] == st and c[3] == d and i in by]
                        if v:
                            vals[f] = float(np.mean(v))
                    m, h = C.mean_ci(vals.values())
                    out["conversion"][f"{arm}|{st}|+{d:g}|{cand}-policy"] = {"mean": m, "ci": h, "per_final": vals}
            for d in LEADS[1:]:
                abs_v = {}
                for f, by in R.items():
                    v = [float(np.mean(_g(by[i], "policy", H_ROLL))) for i, c in cells.items()
                         if c[0] == "weak" and c[2] == st and c[3] == d and i in by]
                    if v:
                        abs_v[f] = float(np.mean(v))
                m, h = C.mean_ci(abs_v.values())
                out["weak"][f"{arm}|{st}|+{d:g}|abs_policy"] = {"mean": m, "ci": h, "per_final": abs_v}
                best = {}
                for cand in WEAK_CANDS:
                    if cand == "policy":
                        continue
                    vals = {}
                    for f, by in R.items():
                        v = [EN.per_state_diff(by[i], cand, "G", H_ROLL) for i, c in cells.items()
                             if c[0] == "weak" and c[2] == st and c[3] == d and i in by]
                        if v:
                            vals[f] = float(np.mean(v))
                    m, h = C.mean_ci(vals.values())
                    out["weak"][f"{arm}|{st}|+{d:g}|{cand}-policy"] = {"mean": m, "ci": h, "per_final": vals}
                    best[cand] = m
                out["weak"][f"{arm}|{st}|+{d:g}|best"] = {"best": max(best, key=best.get) if best else None,
                                                         "value": max(best.values()) if best else float("nan")}
                # channels of each weak response at 50 turns (candidate - policy) and the policy's absolute levels
                for cand in WEAK_CANDS:
                    for key in ("earned_A", "earned_B", "farm_A", "unc_A", "unc_away_A", "away_A", "deaths_A",
                                "recalls_A", "d_total_A", "xp_A", "xp_lost_A", "deaths_B", "bounty_A", "bounty_B",
                                "plate_B", "crash_hp_B", "trade_conn_A"):
                        vals = {}
                        for f, by in R.items():
                            fn = EN.ledger_mean if cand == "policy" else EN.ledger_diff
                            v = [fn(by[i], cand, key, H_ROLL) for i, c in cells.items()
                                 if c[0] == "weak" and c[2] == st and c[3] == d and i in by]
                            if v:
                                vals[f] = float(np.mean(v))
                        m, h = C.mean_ci(vals.values())
                        out["channels"][f"{arm}|{st}|+{d:g}|weak|{cand}|{key}"] = {"mean": m, "ci": h}
            # strong-side channels of the policy in the lead cells relative to the equal cell (accounting)
            for d in LEADS[1:]:
                for key in ("earned_A", "earned_B", "farm_A", "farm_B", "unc_B", "unc_away_B", "away_B", "deaths_B",
                            "deaths_A", "bounty_A", "bounty_B", "plate_A", "plate_B", "xp_A", "xp_B", "d_total_A",
                            "trade_conn_A", "crash_hp_A"):
                    vals = {}
                    for f, by in R.items():
                        v = []
                        for sit in SITS:
                            i_lead = next(i for i, c in cells.items() if c == ("strong", sit, st, d))
                            i_eq = next(i for i, c in cells.items() if c == ("strong", sit, st, 0.0))
                            if i_lead in by and i_eq in by:
                                v.append(EN.ledger_mean(by[i_lead], "policy", key, H_ROLL)
                                         - EN.ledger_mean(by[i_eq], "policy", key, H_ROLL))
                        if v:
                            vals[f] = float(np.mean(v))
                    m, h = C.mean_ci(vals.values())
                    out["channels"][f"{arm}|{st}|+{d:g}|strong|lead-equal|{key}"] = {"mean": m, "ci": h}
        # validation: the 6 fixed strong cells at 150 turns, 50 vs 150 on the same trajectories
        V = _records(res, "S3", arm, H_VAL)
        d50, d150 = [], []
        for f, by in V.items():
            for i, rec in by.items():
                for cand in STRONG_CANDS:
                    if cand == "policy":
                        continue
                    d50.append(EN.per_state_diff(rec, cand, "G", H_ROLL))
                    d150.append(EN.per_state_diff(rec, cand, "G", H_VAL))
        if d50:
            d50, d150 = np.asarray(d50), np.asarray(d150)
            out["validation"][arm] = {"pairs": int(len(d50)), "mean_change": float(np.mean(d150 - d50)),
                                      "slope": float(np.polyfit(d50, d150, 1)[0]) if len(d50) > 2 else float("nan")}
    # cross-arm differences of the premium (paired by final)
    for arm in arms:
        if arm == "baseline":
            continue
        for st in STAGES:
            for d in LEADS[1:]:
                a, b = prem.get((arm, st, d), {}), prem.get(("baseline", st, d), {})
                diff = [a[f] - b[f] for f in a if f in b]
                ratio = [a[f] / b[f] for f in a if f in b and b[f] != 0]
                m, h = C.mean_ci(diff)
                out["premium"][f"{arm}-baseline|{st}|+{d:g}"] = {"mean": m, "ci": h,
                                                                   "ratio_of_means": (np.mean(list(a.values())) /
                                                                                      np.mean(list(b.values())))
                                                                   if b and np.mean(list(b.values())) != 0 else float("nan"),
                                                                   "ratio_per_final_mean": float(np.mean(ratio)) if ratio else float("nan")}
    return out


def analyse_s4(res: Dict[str, dict]) -> dict:
    from ..m46a import engine as EN
    cells = dict(s4_cells())
    arms = sorted({r.get("arm") for r in res.values()}, key=ARM_NAMES.index)
    out = {}
    for arm in arms:
        R = _records(res, "S4", arm)
        for i, (focal, L, h, inv) in cells.items():
            cands = FREEZER_CANDS if focal == "freezer" else BREAKER_CANDS
            row = {}
            absv = [float(np.mean(_g(by[i], "policy", H_ROLL))) for by in R.values() if i in by]
            row["abs_policy"] = dict(zip(("mean", "ci"), C.mean_ci(absv)))
            for cand in cands:
                if cand != "policy":
                    v = [EN.per_state_diff(by[i], cand, "G", H_ROLL) for by in R.values() if i in by]
                    row[f"{cand}-policy"] = dict(zip(("mean", "ci"), C.mean_ci(v)))
                for key in ("held_A", "frozen_A", "crash_hp_B", "crash_hp_A", "deaths_A", "deaths_B", "recalls_A",
                            "away_A", "d_total_A", "d_total_B", "earned_A", "earned_B", "unc_A", "xp_lost_A"):
                    fn = EN.ledger_mean if cand == "policy" else EN.ledger_diff
                    v = [fn(by[i], cand, key, 20) for by in R.values() if i in by]
                    row[f"{cand}|{key}@20"] = float(np.mean(v)) if v else float("nan")
            out[f"{arm}|{focal}|L{L}|held{h}|{inv}"] = row
    return out


def analyse_s5(res: Dict[str, dict]) -> dict:
    from ..m46a import engine as EN
    arms = sorted({r.get("arm") for r in res.values()}, key=ARM_NAMES.index)
    out = {}
    for arm in arms:
        for fam in ("recall_full", "recall_mid"):
            R = _records(res, f"S5_{fam}", arm)
            for name, (c, b) in {"recall-stay": ("recall", "stay"), "inv:recall-inv:stay": ("inv:recall", "inv:stay"),
                                 "recall-policy": ("recall", "policy"), "inv:policy-policy": ("inv:policy", "policy")}.items():
                per = {f: float(np.mean([EN.per_state_diff(r, c, "G", H_ROLL, b) for r in by.values()]))
                       for f, by in R.items()}
                m, h = C.mean_ci(per.values())
                out[f"{arm}|{fam}|{name}"] = {"mean": m, "ci": h, "per_final": per}
            per = {f: float(np.mean([EN.per_state_diff(r, "recall", "G", H_ROLL, "stay")
                                     - EN.per_state_diff(r, "inv:recall", "G", H_ROLL, "inv:stay") for r in by.values()]))
                   for f, by in R.items()}
            m, h = C.mean_ci(per.values())
            out[f"{arm}|{fam}|shopping"] = {"mean": m, "ci": h, "per_final": per}
    return out


__all__ = ["screen_s1", "screen_s2", "screen_s3", "screen_s4", "screen_s5", "screen_s6", "STAGES", "LEADS", "SITS",
           "s3_cells", "s4_cells", "zone_state", "freeze_state"]


# ---------------------------------------------------------------------------------------------- exploratory: P1 power
P1X_ROOT, P1X_K, P1X_H = 97_500, 128, 100
P1X_STAGES = ("mid", "late")
P1X_CELLS = ("equal", "A+8", "B+8", "A+16", "B+16")


def p1x_cells():
    return list(enumerate((st, sit, c) for st in P1X_STAGES for sit in SITS for c in P1X_CELLS))


def screen_p1x(workers: int, arms=None) -> dict:
    """EXPLORATORY (labelled; added after the pre-registered S3 premium proved underpowered): the lead premium with
    128 paired rollouts per cell, both orientations (leader in seat A: +premium; leader in seat B: the trailer's loss),
    policy against policy, horizon 100 (G50 and G100). The same states and seeds in every arm."""
    from ..m46a import engine as EN
    js = []
    for arm in _arms(arms):
        cfg = C.arm_cfg(arm)
        for i, (st, sit, c) in p1x_cells():
            L, S0 = STAGES[st]
            d = 0.0 if c == "equal" else float(c[2:])
            sa, sb = (S0 + d, S0) if c.startswith("A") else ((S0, S0 + d) if c.startswith("B") else (S0, S0))
            start = zone_state(cfg, sit, L, sa, sb)
            for f in C.FINALS:
                j = _job(arm, "P1X", f, i, start, {"policy": ("policy",)}, P1X_H, f"{st} {sit} {c}")
                j["roll"] = EN.roll_seeds(P1X_ROOT, P1X_K)
                js.append(j)
    res = _run(js, workers, "P1X rollouts")
    out = analyse_p1x(res)
    C.save_json("p1x_summary", {"meta": C.meta(stage="p1x", label="EXPLORATORY", K=P1X_K, horizon=P1X_H,
                                               root=P1X_ROOT), "analysis": out})
    return out


def analyse_p1x(res: Dict[str, dict]) -> dict:
    cells = dict(p1x_cells())
    arms = sorted({r.get("arm") for r in res.values()}, key=ARM_NAMES.index)
    prem: Dict[tuple, Dict[int, float]] = {}
    out = {"premium": {}, "diff": {}}
    for arm in arms:
        R = _records(res, "P1X", arm)
        for st in P1X_STAGES:
            for d in (8, 16):
                for h in (50, 100):
                    per = {}
                    for f, by in R.items():
                        v = []
                        for sit in SITS:
                            idx = {c: next(i for i, x in cells.items() if x == (st, sit, c)) for c in P1X_CELLS}
                            if not all(i in by for i in idx.values()):
                                continue
                            g = {c: float(np.mean(_g(by[idx[c]], "policy", h))) for c in P1X_CELLS}
                            # leader in A: G(A+d) - G(equal); leader in B: -(G(B+d) - G(equal)); averaged
                            v.append(0.5 * ((g[f"A+{d}"] - g["equal"]) - (g[f"B+{d}"] - g["equal"])))
                        if v:
                            per[f] = float(np.mean(v))
                    prem[(arm, st, d, h)] = per
                    m, hw = C.mean_ci(per.values())
                    out["premium"][f"{arm}|{st}|+{d}|G{h}"] = {"mean": m, "ci": hw, "per_final": per}
    for arm in arms:
        if arm == "baseline":
            continue
        for st in P1X_STAGES:
            for d in (8, 16):
                for h in (50, 100):
                    a, b = prem[(arm, st, d, h)], prem[("baseline", st, d, h)]
                    diff = [a[f] - b[f] for f in a if f in b]
                    m, hw = C.mean_ci(diff)
                    mb = float(np.mean(list(b.values()))) if b else float("nan")
                    out["diff"][f"{arm}-baseline|{st}|+{d}|G{h}"] = {"mean": m, "ci": hw,
                                                                     "ratio": float(np.mean(list(a.values()))) / mb
                                                                     if mb else float("nan")}
    return out


# ---------------------------------------------------------------------------------------------- S2 analysis
S2_PRICE = 1.5


def analyse_s2(data: dict = None, price: float = S2_PRICE) -> dict:
    """The exact one-turn ledger, HP priced at `price` resources per bar (M4.6a E1's convention):
    zone cells: A's Trade - Hold against B's Hold and against B's Safe, B's Safe - Hold and Trade - Hold against A's
    Trade; the 'both stronger' effect is the equal cell across arms, the lead effect the lead cell minus the equal cell
    within an arm. Freeze cells: P(the freeze survives) after a break or a walk-up against keep and guard, the freezer's
    guard - keep against the break, the breaker's break - walk up and break - step back against keep and guard."""
    from ..m46a.e1 import priced
    data = data or C.load("s2_exact")
    zone, freeze = {}, {}
    for c in data["cells"]:
        P, arm = c["pairs"], c["arm"]
        if c["kind"] == "zone":
            sit, st, d = c["cell"]
            r = {}
            for bn in ("lh_hold", "safe"):
                t_, h_ = P.get(f"lh_trade|{bn}"), P.get(f"lh_hold|{bn}")
                if t_ and h_:
                    r[f"A_trade_minus_hold_vs_{bn}"] = priced(t_, price, 0) - priced(h_, price, 0)
            t_h, t_s, t_t = P.get("lh_trade|lh_hold"), P.get("lh_trade|safe"), P.get("lh_trade|trade")
            if t_h and t_s:
                r["B_safe_minus_hold_vs_trade"] = priced(t_s, price, 1) - priced(t_h, price, 1)
            if t_h and t_t:
                r["B_trade_minus_hold_vs_trade"] = priced(t_t, price, 1) - priced(t_h, price, 1)
            if t_h:
                r["A_dealt_bar"], r["A_taken_bar"] = t_h["dealt_bar"][0], t_h["taken_bar"][0]
            zone[f"{arm}|{sit}|{st}|+{d:g}"] = r
        else:
            L, held, inv = c["cell"]
            r = {}
            for an in ("keep", "guard"):
                for bn in ("break", "walk_up"):
                    e = P.get(f"{an}|{bn}")
                    if e:
                        r[f"P_hold_{bn}_vs_{an}"] = e["freeze_holds"]
                for bn in ("walk_up", "step_back"):
                    e1, e2 = P.get(f"{an}|break"), P.get(f"{an}|{bn}")
                    if e1 and e2:
                        r[f"B_break_minus_{bn}_vs_{an}"] = priced(e1, price, 1) - priced(e2, price, 1)
            g, k = P.get("guard|break"), P.get("keep|break")
            if g and k:
                r["A_guard_minus_keep_vs_break"] = priced(g, price, 0) - priced(k, price, 0)
            freeze[f"{arm}|L{L}|held{held}|{inv}"] = r
    # the 'both stronger' and the lead effects (neutral situation, A's Trade - Hold against Hold)
    effects = {}
    for arm in sorted({k.split("|")[0] for k in zone}):
        for st in STAGES:
            eq = zone.get(f"{arm}|neutral|{st}|+0", {}).get("A_trade_minus_hold_vs_lh_hold")
            eqb = zone.get(f"baseline|neutral|{st}|+0", {}).get("A_trade_minus_hold_vs_lh_hold")
            for d in LEADS[1:]:
                ld = zone.get(f"{arm}|neutral|{st}|+{d:g}", {}).get("A_trade_minus_hold_vs_lh_hold")
                if eq is not None and ld is not None:
                    effects[f"{arm}|{st}|+{d:g}"] = {"both_stronger": eq - eqb, "lead_effect": ld - eq}
    out = {"price": price, "zone": zone, "freeze": freeze, "effects": effects}
    C.save_json("s2_analysis", {"meta": C.meta(stage="s2_analysis"), "analysis": out})
    return out
