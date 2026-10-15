"""Analysis and report of the M4.5c campaign's training and limited health evaluation: results/m45c/full/summary.json,
report.md and figures/ (generated from full_train.json, health_eval.json, health_training.json, health_behavior.json;
nothing here simulates).

Statistics. A matchup value is evaluate_discounted's paired estimate with its 95% interval over paired evaluation seeds.
The scripted mean of a checkpoint is the mean of its 9 scripted matchups; its interval combines their standard errors
as if independent (approximate: the opponents share evaluation seeds). Across training seeds (the unit of replication)
means carry a t interval with n - 1 = 5 degrees of freedom (t = 2.571); every seed is also shown on its own.

Pre-registered rules (pilot report §8.4), evaluated mechanically:
    plateau of a run    (a) |scripted mean(final) - scripted mean(6.144M)| < 0.3 on the curve, AND
                        (b) final vs its 6.144M snapshot (W = 1000, 64 paired seeds) within +-0.3 with the interval
                            covering 0
    extension           at least 3 of 6 runs not plateaued AND final - midpoint (W = 1000) > +0.5 with the interval
                        over seeds excluding 0 (evaluated on the seed mean; the per-seed count is reported)
    stop / investigate  KL > 0.05 sustained, a non-finite statistic, gradient norm > 100, entropy < 0.3 nats, or a
                        later snapshot losing significantly to an earlier one on at least 2 seeds
    seeds disagree      a cross-play value beyond +-2 resources
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .common import load_json, save_json, write_md
from .full import FULL_BUDGET, FULL_MIDPOINT, FULL_OUT, FULL_SEEDS
from . import health as _H
from .health import CHECKPOINTS, LATE_QUARTER

_CAMP = _H.MAIN


def spec(seed: int, n) -> str:
    return _H.spec(seed, n, _CAMP)


def _use(camp) -> None:
    """Point the module's campaign constants at a campaign phase (the 8.192M campaign by default, or the matched 2x
    extension). Every fraction of the schedule scales with its budget."""
    global _CAMP, FULL_BUDGET, FULL_MIDPOINT, LATE_QUARTER, CHECKPOINTS, POINTS, FULL_OUT, FIG
    _CAMP = camp
    FULL_BUDGET, FULL_MIDPOINT, LATE_QUARTER = camp.budget, camp.mid, camp.q3
    CHECKPOINTS = camp.checkpoints
    POINTS = ("initial", *CHECKPOINTS)
    FULL_OUT = camp.out
    FIG = os.path.join(camp.out, "figures")
from .references import OOD, ROLE, SUITE

FIG = os.path.join(FULL_OUT, "figures")
BLUE, RED, GREY, INK, MUTED, SEEDLINE = "#2a78d6", "#e34948", "#a3a29c", "#0b0b0b", "#52514e", "#b9b8b1"
T5 = 2.570582                       # t(0.975, 5)
PLATEAU_TOL, EXTEND_MIN, CROSS_GAP = 0.3, 0.5, 2.0
REFS = tuple(f"ref:{r}" for r in SUITE)
POINTS = ("initial", *CHECKPOINTS)


def _tci(v: Sequence[float]) -> List[float]:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    if len(v) == 1:
        return [float(v[0]), float("nan")]
    from scipy import stats
    return [float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))]


def _half(r: dict) -> float:
    return (r["ci95"][1] - r["ci95"][0]) / 2


def _mx(n) -> float:
    return 0.0 if n == "initial" else n / 1e6


# ---------------------------------------------------------------------------------------------- analysis
def curve(ev: dict, seeds: Sequence[int]) -> dict:
    cur = ev.get("curve", {})
    per: Dict[str, dict] = {}
    for s in seeds:
        rows = {}
        for n in POINTS:
            x = spec(s, n)
            rs = [cur.get(f"{x}|{o}") for o in REFS]
            if any(r is None for r in rs):
                continue
            est = [r["estimate"] for r in rs]
            se = math.sqrt(sum(r["stderr"] ** 2 for r in rs)) / len(rs)
            o = cur.get(f"{x}|{OOD[0]}")
            rows[str(n)] = {"scripted": float(np.mean(est)), "scripted_half": 1.96 * se, "by_opp": dict(zip(REFS, est)),
                            "ood": o["estimate"] if o else None, "ood_half": _half(o) if o else None}
        per[str(s)] = rows
    pooled = {}
    for n in POINTS:
        vals = [per[str(s)][str(n)] for s in seeds if str(n) in per[str(s)]]
        if len(vals) != len(seeds):
            continue
        pooled[str(n)] = {"scripted": _tci([v["scripted"] for v in vals]), "ood": _tci([v["ood"] for v in vals]),
                          "by_opp": {o: _tci([v["by_opp"][o] for v in vals]) for o in REFS},
                          "scripted_min_max": [min(v["scripted"] for v in vals), max(v["scripted"] for v in vals)]}
    out = {"per_seed": per, "pooled": pooled}
    # second half and last quarter, paired by seed
    for a, b, key in ((FULL_BUDGET, FULL_MIDPOINT, "final_minus_mid"), (FULL_BUDGET, LATE_QUARTER, "final_minus_q3"),
                      (FULL_MIDPOINT, "initial", "mid_minus_initial")):
        d = {}
        for s in seeds:
            pa, pb = per[str(s)].get(str(a)), per[str(s)].get(str(b))
            if pa and pb:
                d[str(s)] = {"scripted": pa["scripted"] - pb["scripted"],
                             "half": math.hypot(pa["scripted_half"], pb["scripted_half"]),
                             "ood": (pa["ood"] - pb["ood"]) if pa["ood"] is not None and pb["ood"] is not None else None,
                             "by_opp": {o: pa["by_opp"][o] - pb["by_opp"][o] for o in REFS}}
        out[key] = {"per_seed": d, "scripted": _tci([v["scripted"] for v in d.values()]),
                    "ood": _tci([v["ood"] for v in d.values()]),
                    "by_opp": {o: _tci([v["by_opp"][o] for v in d.values()]) for o in REFS}}
    return out


def h2h(ev: dict, seeds: Sequence[int]) -> dict:
    res = ev.get("h2h", {})
    out = {}
    for n, key in ((FULL_MIDPOINT, "final_vs_mid"), (LATE_QUARTER, "final_vs_q3")):
        per = {}
        for s in seeds:
            r = res.get(f"{spec(s, 'final')}|{spec(s, n)}")
            if r:
                per[str(s)] = {"estimate": r["estimate"], "half": _half(r), "seat_A": r["seat_A"], "seat_B": r["seat_B"]}
        out[key] = {"per_seed": per, "pooled": _tci([v["estimate"] for v in per.values()])}
    return out


def history(ev: dict, seeds: Sequence[int]) -> dict:
    res = ev.get("history", {})
    out = {}
    for s in seeds:
        rows = {}
        for k, r in res.items():
            x, y = k.split("|")
            if x != spec(s, "final"):
                continue
            n = 0 if y.startswith("initial") else int(y.split(":")[3])
            rows[str(n)] = {"estimate": r["estimate"], "half": _half(r), "loses_significantly": r["ci95"][1] < 0}
        out[str(s)] = dict(sorted(rows.items(), key=lambda kv: int(kv[0])))
    return out


def cycle(ev: dict, seeds: Sequence[int], hist: dict) -> dict:
    res = ev.get("cycle", {})
    per, flagged = {}, []
    for s in seeds:
        rows = []
        for k, r in res.items():
            x, y = k.split("|")
            if not x.startswith(f"snap:full:{s}:") and not x.startswith(f"final:full:{s}"):
                continue
            a = int(x.split(":")[3]) if x.startswith("snap") else FULL_BUDGET
            b = int(y.split(":")[3])
            rows.append({"later": a, "earlier": b, "estimate": r["estimate"], "half": _half(r),
                         "later_loses_significantly": r["ci95"][1] < 0})
        h = hist.get(str(s), {}).get(str(CHECKPOINTS[-2]))          # final vs 7.37M, from history
        if h:
            rows.append({"later": FULL_BUDGET, "earlier": CHECKPOINTS[-2], "estimate": h["estimate"], "half": h["half"],
                         "later_loses_significantly": h["loses_significantly"]})
        rows.sort(key=lambda r: r["later"])
        per[str(s)] = rows
        if any(r["later_loses_significantly"] for r in rows) or any(v["loses_significantly"]
                                                                     for v in hist.get(str(s), {}).values()):
            flagged.append(s)
    return {"per_seed": per, "seeds_with_a_significant_regression": flagged,
            "stop_criterion_met": len(flagged) >= 2}


def cross(ev: dict, seeds: Sequence[int]) -> dict:
    res = ev.get("cross", {})
    n = len(seeds)
    M = np.full((n, n), np.nan)
    H = np.full((n, n), np.nan)
    for i, a in enumerate(seeds):
        for j, b in enumerate(seeds):
            r = res.get(f"{spec(a, 'final')}|{spec(b, 'final')}")
            if r:
                M[i, j], H[i, j] = r["estimate"], _half(r)
                M[j, i], H[j, i] = -r["estimate"], _half(r)
    row = {str(s): float(np.nanmean(np.delete(M[i], i))) for i, s in enumerate(seeds)} if n > 1 else {}
    off = M[~np.eye(n, dtype=bool)]
    sig = int(sum(1 for i in range(n) for j in range(i + 1, n)
                  if np.isfinite(M[i, j]) and abs(M[i, j]) > H[i, j]))
    return {"seeds": list(seeds), "matrix": M.tolist(), "half": H.tolist(), "row_mean": row,
            "max_abs": float(np.nanmax(np.abs(off))) if np.isfinite(off).any() else float("nan"),
            "pairs_significant": sig, "pairs": n * (n - 1) // 2,
            "seeds_disagree": bool(np.nanmax(np.abs(off)) > CROSS_GAP) if np.isfinite(off).any() else None}


def cross_mid(ev: dict, seeds: Sequence[int], hh: dict, cr: dict) -> dict:
    """final_i vs midpoint_j (j != i) against the own-lineage gain final_i vs midpoint_i (h2h) and the cross-play
    level final_i vs final_j: transfer = mean_j [v(final_i, mid_j) - v(final_i, final_j)], the second-half gain of
    lineage j measured by an independent policy i."""
    res = ev.get("cross_mid", {})
    if not res:
        return {}
    M = np.asarray(cr["matrix"], float)
    per = {}
    for i, a in enumerate(seeds):
        vs_mid, transfer = [], []
        for j, b in enumerate(seeds):
            r = res.get(f"{spec(a, 'final')}|{spec(b, FULL_MIDPOINT)}")
            if r is None or i == j:
                continue
            vs_mid.append(r["estimate"])
            if np.isfinite(M[i, j]):
                transfer.append(r["estimate"] - M[i, j])
        own = hh["final_vs_mid"]["per_seed"].get(str(a), {}).get("estimate")
        per[str(a)] = {"vs_other_midpoints": float(np.mean(vs_mid)) if vs_mid else None,
                       "transfer_gain": float(np.mean(transfer)) if transfer else None, "own_lineage_gain": own}
    gain_of = {}
    for j, b in enumerate(seeds):                       # the gain of lineage j as measured by the other finals
        g = [res[f"{spec(a, 'final')}|{spec(b, FULL_MIDPOINT)}"]["estimate"] - M[i, j]
             for i, a in enumerate(seeds) if i != j and f"{spec(a, 'final')}|{spec(b, FULL_MIDPOINT)}" in res
             and np.isfinite(M[i, j])]
        gain_of[str(b)] = float(np.mean(g)) if g else None
    return {"per_final": per, "gain_of_lineage": gain_of,
            "vs_other_midpoints": _tci([v["vs_other_midpoints"] for v in per.values()]),
            "transfer_gain": _tci([v["transfer_gain"] for v in per.values()]),
            "own_lineage_gain": _tci([v["own_lineage_gain"] for v in per.values()])}


def rules(cv: dict, hh: dict, cy: dict, cr: dict, tr: Optional[dict], seeds: Sequence[int]) -> dict:
    plateau = {}
    for s in seeds:
        a = cv["final_minus_q3"]["per_seed"].get(str(s))
        b = hh["final_vs_q3"]["per_seed"].get(str(s))
        if not a or not b:
            continue
        ca = abs(a["scripted"]) < PLATEAU_TOL
        cb = abs(b["estimate"]) <= PLATEAU_TOL and b["estimate"] - b["half"] <= 0 <= b["estimate"] + b["half"]
        plateau[str(s)] = {"curve_last_quarter": a["scripted"], "curve_ok": ca, "final_vs_q3": b["estimate"],
                           "final_vs_q3_half": b["half"], "h2h_ok": cb, "plateaued": ca and cb}
    not_plat = [s for s, v in plateau.items() if not v["plateaued"]]
    fm = hh["final_vs_mid"]
    m, h = fm["pooled"]
    per_seed_big = [s for s, v in fm["per_seed"].items() if v["estimate"] - v["half"] > 0 and v["estimate"] > EXTEND_MIN]
    ext = len(not_plat) >= 3 and np.isfinite(m) and m > EXTEND_MIN and m - h > 0
    alarms = {k: v.get("alarms", []) for k, v in (tr or {}).get("stability", {}).items()}
    return {"plateau": plateau, "not_plateaued": not_plat,
            "extension": {"criterion_met": bool(ext), "not_plateaued_count": len(not_plat),
                          "final_minus_mid_seed_mean": [m, h], "seeds_individually_above_0.5_and_significant": per_seed_big},
            "stop": {"training_alarms": alarms, "any_training_alarm": any(alarms.values()),
                     "cycling_regressions_on_2_seeds": cy["stop_criterion_met"]},
            "seeds_disagree": cr["seeds_disagree"]}


CATS = ("LH", "TN", "SL", "SH", "Ward", "Recall")
BEHS = ("Hold", "Yield", "Trade", "Safe")
CONTRASTS = (("HP < 35%", "Recall"), ("HP >= 70%", "Recall"), ("HP < 35%", "beh_Safe"), ("HP >= 70%", "beh_Safe"),
             ("HP >= 70%", "beh_Trade"), ("holding a freeze (freezer)", "TN"), ("holding a freeze (freezer)", "beh_Trade"),
             ("frozen on", "SH"), ("frozen on", "Recall"), ("enemy champion away", "SH"),
             ("own crash under the enemy tower, Pressure legal", "Pressure"), ("enemy crash under my tower", "beh_Yield"),
             ("waves meet on the enemy half (>= 1)", "beh_Safe"), ("waves meet on my half (<= -1)", "beh_Trade"),
             ("Ward legal, own vision down", "Ward"), ("Ward legal, own vision up", "Ward"),
             ("stronger (power ratio >= 1.25)", "beh_Trade"), ("weaker (power ratio <= 0.8)", "beh_Safe"),
             ("level 1-5", "SH"), ("level 6-10", "SH"), ("level 11-18", "SH"), ("level 1-5", "beh_Trade"),
             ("level 11-18", "beh_Trade"))


def behavior(bh: dict, seeds: Sequence[int]) -> dict:
    rec, led = bh.get("records", {}), bh.get("ledger", {})
    out = {"overall": {}, "contrasts": {}, "summary": {}, "freezes": {}, "recalls": {}, "ledger": {}, "early": {}}
    for s in seeds:
        for tag, n in (("final", "final"), ("mid", FULL_MIDPOINT)):
            r = rec.get(f"{spec(s, n)}|self|1000")
            if not r:
                continue
            key = f"{s}|{tag}"
            cond = r["conditional"]
            out["overall"][key] = cond["all present decisions"]["mass"]
            out["contrasts"][key] = {f"{st}::{q}": [cond[st]["n"], cond[st].get("mass", {}).get(q)]
                                     for st, q in CONTRASTS if st in cond}
            out["summary"][key] = {k: v[0] for k, v in r["summary"].items()}
            f = r["freezes"]
            est = f["established"]
            out["freezes"][key] = {"per_1000": f["established_per_1000_turns"][0], "n": est["n"],
                                   "length": est.get("length_mean"), "denial_share": est.get("useful_denial_share"),
                                   "gold_diff": est["gold_diff"][0] if est["n"] else None, "held_share": f["held_share"]}
            rc = r["recalls"]
            out["recalls"][key] = {"low_hp_per_1000": rc["recall_low_hp"]["per_1000_turns"],
                                   "other_per_1000": rc["recall_other"]["per_1000_turns"],
                                   "other_hp": rc["recall_other"]["hp_frac"], "other_unspent": rc["recall_other"]["unspent_gold"],
                                   "other_bought_share": rc["recall_other"]["bought_any_share"]}
            lg = led.get(f"{spec(s, n)}|self")
            if lg:
                p = {k: v[0] for k, v in lg["per_100_turns"].items()}
                p["recall_interrupted_share"] = p["rec_intr"] / max(p["recalls"] + p["rec_intr"], 1e-9)
                p["minion_share_of_damage"] = p["d_min"] / max(p["d_trade"] + p["d_min"] + p["d_tower"] + p["d_gank"], 1e-9)
                p["wards_interrupted_share"] = p["ward_intr"] / max(p["wards"], 1e-9)
                out["ledger"][key] = p
        e = rec.get(f"{spec(s, 'final')}|self|120")
        if e:
            c = e["conditional"]
            out["early"][str(s)] = {st: {"n": c[st]["n"], "mass": c[st].get("mass")} for st in ("level 1-5", "level 6-10",
                                                                                               "level 11-18") if st in c}
        for opp in ("ref:g_bully", "ref:fz_guard_sh"):
            r = rec.get(f"{spec(s, 'final')}|{opp}|500")
            if r:
                out["summary"][f"{s}|vs {opp[4:]}"] = {k: v[0] for k, v in r["summary"].items()}
                f = r["freezes"]
                out["freezes"][f"{s}|vs {opp[4:]}"] = {"per_1000": f["established_per_1000_turns"][0],
                                                       "held_on_share": f["held_on_share"], "held_share": f["held_share"]}
    for s in seeds:
        g = bh.get("power_gap", {}).get(spec(s, "final"))
        if not g:
            continue
        z, c = g["zoning"], g["conditional"]
        thr = z.get("credible_threat", {})
        out.setdefault("power_gap", {})[str(s)] = {
            "by_power": {k: {kk: v.get(kk) for kk in ("n", "trade_attempt_rate", "trade_connect_rate", "mean_pos",
                                                      "at_own_tower", "death_rate", "collection_rate")}
                         for k, v in z.get("by_power", {}).items()},
            "threat": {k: {"n": v.get("n"), "behavior": v.get("behavior"), "mean_pos": v.get("mean_pos"),
                           "uncollected_per_turn": v.get("uncollected_per_turn"),
                           "xp_missed_per_turn": v.get("xp_missed_per_turn")} for k, v in thr.items()},
            "pressure_cost": z.get("pressure_cost"),
            "stronger_mass": c.get("stronger (power ratio >= 1.25)", {}).get("mass"),
            "weaker_mass": c.get("weaker (power ratio <= 0.8)", {}).get("mass"),
            "n": {k: c.get(k, {}).get("n") for k in ("stronger (power ratio >= 1.25)", "weaker (power ratio <= 0.8)")}}
    return out


def analyze(camp=None) -> dict:
    global POINTS
    if camp is not None:
        _use(camp)
    tr = load_json(f"{_CAMP.phase}_train", FULL_OUT)
    ev = load_json("health_eval", FULL_OUT) or {}
    trn = load_json("health_training", FULL_OUT) or {}
    bh = load_json("health_behavior", FULL_OUT) or {}
    seeds = list(FULL_SEEDS)
    if ev.get("curve_rest"):                         # the comprehensive curve: every grid point that was evaluated
        ev = dict(ev)
        ev["curve"] = {**ev["curve"], **ev["curve_rest"]}
        have = {int(k.split("|")[0].split(":")[3]) for k in ev["curve"] if k.startswith("snap:")}
        POINTS = ("initial", *sorted(have | {FULL_BUDGET}))
    cv, hh = curve(ev, seeds), h2h(ev, seeds)
    hi = history(ev, seeds)
    cy, cr = cycle(ev, seeds, hi), cross(ev, seeds)
    data = {"curve": cv, "h2h": hh, "history": hi, "cycle": cy, "cross": cr, "cross_mid": cross_mid(ev, seeds, hh, cr),
            "rules": rules(cv, hh, cy, cr, tr, seeds),
            "behavior": behavior(bh, seeds), "timing": ev.get("timing", {}),
            "specs": {k: ev.get(f"{k}_spec") for k in ("curve", "h2h", "history", "cycle", "cross", "cross_mid")},
            "comprehensive": comprehensive(ev, seeds)}
    save_json("summary", data, FULL_OUT)
    return data


def _mat(res: dict, xs: List[str], ys: List[str]) -> dict:
    """{x|y: (estimate, half)} as a matrix (rows xs, columns ys)."""
    M = np.full((len(xs), len(ys)), np.nan)
    H = np.full((len(xs), len(ys)), np.nan)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            r = res.get(f"{x}|{y}")
            if r:
                M[i, j], H[i, j] = r["estimate"], _half(r)
    return {"rows": xs, "cols": ys, "matrix": M.tolist(), "half": H.tolist()}


def comprehensive(ev: dict, seeds: Sequence[int]) -> dict:
    """The comprehensive parts when present: the W = 1000 headline (final and midpoint, and the 8.192M finals as the
    comparison group), the budget comparison (16.384M finals vs 8.192M finals; annealing test), the historical PPO
    references, and the per-run 9-checkpoint tournament."""
    out = {}
    refs = list(REFS) + list(OOD)
    hl, hm = ev.get("headline", {}), ev.get("headline_main", {})
    if hl or hm:
        rows = {}
        for lab, res, sp in (("final", hl, lambda s: spec(s, "final")), ("midpoint", hl, lambda s: spec(s, FULL_MIDPOINT)),
                             ("8.192M final", hm, lambda s: _H.spec(s, "final", _H.MAIN))):
            per = {}
            for s in seeds:
                v = [res.get(f"{sp(s)}|{o}") for o in refs]
                if all(v):
                    per[str(s)] = {"scripted": float(np.mean([x["estimate"] for x in v[:len(REFS)]])),
                                   "ood": v[-1]["estimate"], "by_opp": {o: x["estimate"] for o, x in zip(refs, v)},
                                   "seat_A": float(np.mean([x["seat_A"] for x in v[:len(REFS)]])),
                                   "seat_B": float(np.mean([x["seat_B"] for x in v[:len(REFS)]]))}
            if per:
                rows[lab] = {"per_seed": per, "scripted": _tci([v["scripted"] for v in per.values()]),
                             "ood": _tci([v["ood"] for v in per.values()]),
                             "by_opp": {o: _tci([v["by_opp"][o] for v in per.values()]) for o in refs},
                             "seat_A": _tci([v["seat_A"] for v in per.values()]),
                             "seat_B": _tci([v["seat_B"] for v in per.values()])}
        if "final" in rows and "8.192M final" in rows:
            d = {s: rows["final"]["per_seed"][s]["scripted"] - rows["8.192M final"]["per_seed"][s]["scripted"]
                 for s in rows["final"]["per_seed"] if s in rows["8.192M final"]["per_seed"]}
            rows["final minus 8.192M final"] = {"per_seed": d, "scripted": _tci(list(d.values())),
                                                "by_opp": {o: _tci([rows["final"]["per_seed"][s]["by_opp"][o]
                                                                    - rows["8.192M final"]["per_seed"][s]["by_opp"][o]
                                                                    for s in d]) for o in refs}}
        out["headline"] = rows
    for part, xs, ys in (("vs_main", [spec(s, "final") for s in seeds], [_H.spec(s, "final", _H.MAIN) for s in seeds]),
                         ("anneal", [_H.spec(s, "final", _H.MAIN) for s in seeds], [spec(s, FULL_MIDPOINT) for s in seeds])):
        res = ev.get(part, {})
        if not res:
            continue
        m = _mat(res, xs, ys)
        M = np.asarray(m["matrix"])
        H = np.asarray(m["half"])
        diag = [float(M[i, i]) for i in range(len(seeds))]
        off = [float(M[i, j]) for i in range(len(seeds)) for j in range(len(seeds)) if i != j]
        m.update({"same_seed": diag, "same_seed_mean": _tci(diag), "cross_seed_mean": _tci(off),
                  "row_means": _tci([float(np.nanmean(M[i])) for i in range(len(seeds))]),
                  "pairs_significant_positive": int(np.sum((M - H) > 0)), "pairs_significant_negative": int(np.sum((M + H) < 0)),
                  "pairs": int(np.isfinite(M).sum())})
        out[part] = m
    res = ev.get("historical", {})
    if res:
        from .health import PILOT_FINAL
        hist = {}
        for c, lab in ((_H.MAIN, "8.192M finals"), (_CAMP, "finals")):
            for y in (PILOT_FINAL, *OOD):
                v = [res.get(f"{_H.spec(s, 'final', c)}|{y}") for s in seeds]
                if all(v):
                    hist[f"{lab}|{y}"] = {"per_seed": [x["estimate"] for x in v], "mean": _tci([x["estimate"] for x in v])}
        out["historical"] = hist
    res = ev.get("tournament", {})
    if res:
        t = _CAMP.tournament
        per = {}
        for s in seeds:
            sp = [spec(s, n) for n in t]
            m = _mat(res, sp, sp)
            M = np.asarray(m["matrix"])
            for i in range(len(t)):
                for j in range(len(t)):
                    if np.isfinite(M[i, j]) and not np.isfinite(M[j, i]):
                        M[j, i] = -M[i, j]
            H = np.asarray(m["half"])
            Hs = np.where(np.isfinite(H), H, H.T)
            # a later checkpoint (row j > i) losing significantly to an earlier one (column i)
            later_loses = [(t[i], t[j]) for i in range(len(t)) for j in range(i + 1, len(t))
                           if np.isfinite(M[i, j]) and M[i, j] - Hs[i, j] > 0]
            score = [float(np.nanmean(np.delete(M[i], i))) for i in range(len(t))]
            per[str(s)] = {"score": score, "later_loses_significantly": later_loses}
        out["tournament"] = {"checkpoints": list(t), "per_seed": per,
                             "seeds_with_regression": [s for s, v in per.items() if v["later_loses_significantly"]]}
    return out


# ---------------------------------------------------------------------------------------------- estimate
ATTACKER_RATIO = 0.8          # M4.3b: an attacker's joint steps/s relative to self-play joint steps/s under load


def estimate(d: dict, tr: Optional[dict], workers: int = 11) -> dict:
    """The comprehensive evaluation's runtime from rates measured in THIS execution: evaluation throughput (health
    timing: network vs scripted and network vs network), the battery smoke's turns per job-second (else the horizon
    study's 550), and the campaign's per-process training throughput for the attackers."""
    from .attackers import ATTACK_BUDGET_F
    from .battery import volume
    from .health import JOBS, turns_of
    tm = d.get("timing", {})
    r_script = tm.get("curve", {}).get("turns_per_worker_s", 1150.0)
    nn = [tm[k]["turns_per_worker_s"] for k in ("h2h", "history", "cycle", "cross") if k in tm]
    r_net = float(np.mean(nn)) if nn else 600.0
    sm = load_json("smoke_summary", os.path.join(os.path.dirname(FULL_OUT), "battery_smoke"))
    r_smoke = sm["turns_per_job_second"] if sm else None
    r_bat = 550.0                 # the horizon study's at-scale rate (3.75 worker-hours); the tiny smoke is optimistic
    v = volume(rate=r_bat, workers=workers)
    if r_smoke:
        v["hours_range"] = [v["total_turns"] / (max(r_smoke, 650.0) * workers) / 3600, v["hours_range"][1]]
    std = {"curve_rest": turns_of(JOBS["curve_rest"]()) / (r_script * workers) / 3600,
           "headline": turns_of(JOBS["headline"]()) / (r_script * workers) / 3600,
           "tournament": turns_of(JOBS["tournament"]()) / (r_net * workers) / 3600}
    bh = load_json("health_behavior", FULL_OUT)
    std["behaviour (5 checkpoints, opponents, ledger)"] = 2.5 * (bh["elapsed_s"] if bh else 300.0) / 3600
    std["integrity, traces, viewer, Recall follow-up"] = 0.15
    tps = [r["transitions_per_s"] for r in (tr or {}).get("runs", {}).values() if r.get("status") == "complete"
           and not r["name"].endswith("_repeat")]
    tps = float(np.mean(tps)) if tps else 3200.0
    att_rate = ATTACKER_RATIO * tps / 2                       # learner transitions (= joint steps) per second
    att_train = ATTACK_BUDGET_F / att_rate / 3600
    att_eval = 6 * 64 * 2 * (460 + 1000 + 5 * 1000) / (r_net * 6) / 3600
    parts = {"standard evaluation (remaining)": sum(std.values()), "strategic battery": v["hours"],
             "best-response attackers": att_train + att_eval, "tests, regression, reports": 0.3}
    sub = sum(parts.values())
    return {"rates": {"net_vs_scripted": r_script, "net_vs_net": r_net, "battery": r_bat,
                      "battery_smoke": r_smoke or float("nan"),
                      "training_tps_per_process": tps, "attacker_transitions_per_s": att_rate},
            "standard_parts_h": std, "battery": {k: v[k] for k in ("per_final", "replication_per_midpoint",
                                                                    "total_turns", "hours", "hours_range")},
            "attackers_h": {"train": att_train, "evaluate": att_eval}, "parts_h": parts, "margin_h": 0.15 * sub,
            "total_h": 1.15 * sub, "range_h": [sub + 0.5 * (v["hours_range"][0] - v["hours"]),
                                               1.3 * sub + (v["hours_range"][1] - v["hours"])]}


# ---------------------------------------------------------------------------------------------- figures
def _style(ax):
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=8)


def _seed_panel(ax, xs, ys_by_seed: Dict[str, Sequence[float]], title: str, zero: bool = False, label_end: bool = True):
    """One panel: every seed a thin muted line with a direct end label, the seed mean in blue with its t band."""
    if zero:
        ax.axhline(0, color=GREY, linewidth=1)
    Y = []
    for s, y in ys_by_seed.items():
        y = np.asarray(y, float)
        ax.plot(xs[:len(y)], y, color=SEEDLINE, linewidth=1.0)
        Y.append(y)
    if label_end and Y and all(len(y) for y in Y):            # name only the highest and the lowest seed at the end
        ends = {s: np.asarray(y, float)[-1] for s, y in ys_by_seed.items()}
        for s, va in ((max(ends, key=ends.get), "bottom"), (min(ends, key=ends.get), "top")):
            y = np.asarray(ys_by_seed[s], float)
            ax.annotate(f"s{s}", (xs[len(y) - 1], y[-1]), xytext=(3, 0), textcoords="offset points", fontsize=7,
                        color=MUTED, va=va)
    L = min(len(y) for y in Y)
    A = np.vstack([y[:L] for y in Y])
    m = np.nanmean(A, 0)
    if A.shape[0] > 1:
        h = T5 * np.nanstd(A, 0, ddof=1) / math.sqrt(A.shape[0])
        ax.fill_between(xs[:L], m - h, m + h, color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(xs[:L], m, color=BLUE, linewidth=2)
    ax.set_title(title, fontsize=9, color=INK)
    _style(ax)


def _save(fig, name: str) -> str:
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p


def fig_curves(cv: dict) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per = cv["per_seed"]
    if not any(per.values()):
        return None
    panels = [("scripted", "scripted-suite mean")] + [(o, o[4:]) for o in REFS] + [("ood", "M4.4b aggro (OOD)")]
    cols = 4
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows), sharex=True)
    xs = np.array([_mx(n) for n in POINTS])
    for ax, (k, title) in zip(axes.ravel(), panels):
        ys = {}
        for s, rows_ in per.items():
            y = []
            for n in POINTS:
                r = rows_.get(str(n))
                if r is None:
                    break
                y.append(r["scripted"] if k == "scripted" else r["ood"] if k == "ood" else r["by_opp"][k])
            ys[s] = y
        _seed_panel(ax, xs, ys, title, zero=True)
        ax.axvline(FULL_MIDPOINT / 1e6, color=GREY, linewidth=0.8, linestyle=":")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.supxlabel("agent transitions (millions); 0 = untrained; dotted = midpoint 4.096M", fontsize=9, color=MUTED)
    fig.supylabel("discounted lead (resources, W = 460, 32 paired seeds)", fontsize=9, color=MUTED)
    fig.tight_layout()
    return _save(fig, "learning_curves.png")


def fig_training(trn: dict) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = trn.get("seeds", {})
    if not seeds:
        return None
    keys = (("entropy", "policy entropy (nats)"), ("approx_kl", "approximate KL"), ("clipfrac", "clip fraction"),
            ("explained_variance", "explained variance"), ("grad_norm", "gradient norm (pre-clip)"),
            ("value_loss", "value loss (scaled)"), ("act_cat_SH", "Shove share (sampled)"),
            ("act_cat_Ward", "Ward share (sampled)"), ("act_cat_Recall", "Recall share (sampled)"))
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.2), sharex=True)
    for ax, (k, title) in zip(axes.ravel(), keys):
        xs = np.asarray(next(iter(seeds.values()))["curve"]["agent_transitions"], float) / 1e6
        ys = {}
        for s, v in seeds.items():
            y = np.asarray(v["curve"][k], float)
            w = 5
            ys[s] = np.convolve(np.pad(y, (w // 2, w // 2), mode="edge"), np.ones(w) / w, mode="valid") if len(y) > w else y
        _seed_panel(ax, xs, ys, title, label_end=False)
    fig.supxlabel("agent transitions (millions); 10-update means, 5-point smoothing", fontsize=9, color=MUTED)
    fig.tight_layout()
    return _save(fig, "training_diagnostics.png")


def fig_cross(cr: dict) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    M = np.asarray(cr["matrix"], float)
    if not np.isfinite(M).any():
        return None
    lim = max(1.0, float(np.nanmax(np.abs(M))))
    cmap = LinearSegmentedColormap.from_list("div", [RED, "#f0efec", BLUE])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(np.where(np.isfinite(M), M, 0), cmap=cmap, vmin=-lim, vmax=lim)
    n = len(cr["seeds"])
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(n), [f"s{s}" for s in cr["seeds"]])
    ax.set_yticks(range(n), [f"s{s}" for s in cr["seeds"]])
    ax.set_title("cross-play of the finals: row's lead vs column\n(W = 1000, 64 paired seeds; blue = row ahead)",
                 fontsize=9, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    return _save(fig, "cross_play.png")


# ---------------------------------------------------------------------------------------------- report
def _f(x, d=2, sign=True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _pm(m, h, d=2) -> str:
    if m is None or not np.isfinite(m):
        return "n/a"
    return f"{m:+.{d}f}" + (f" ± {h:.{d}f}" if h is not None and np.isfinite(h) else "")


def table(header, rows) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _ptlabel(n) -> str:
    return "untrained" if n == "initial" else f"{n / 1e6:.3f}M"


def report(camp=None) -> str:
    if camp is not None:
        _use(camp)
    tr = load_json(f"{_CAMP.phase}_train", FULL_OUT)
    trn = load_json("health_training", FULL_OUT) or {}
    d = analyze()
    figs = [fig_curves(d["curve"]), fig_training(trn), fig_cross(d["cross"])]
    seeds = list(FULL_SEEDS)
    S = []
    S.append(f"# M4.5c {'matched 2x extension (16.384M)' if _CAMP.phase == 'ext' else 'full campaign (8.192M)'} — "
             "training and evaluation (generated tables)\n")
    S.append(f"Generated by `python -m lanegame.experiments.rl.m45c health_report --phase {_CAMP.phase}` from "
             f"{FULL_OUT.replace(os.sep, '/')}/*.json. "
             "Interpretation: docs/m4/m4_5/m4_5c_training_report.md.\n")
    # 1 training
    if tr:
        rows = []
        for name, r in tr["runs"].items():
            if r.get("status") != "complete":
                rows.append([name, r.get("status"), "", "", "", "", "", ""])
                continue
            rows.append([name, "complete", f"{r['agent_transitions']:,}", r["updates"], f"{r['elapsed_s'] / 60:.1f}",
                         f"{r['transitions_per_s']:,.0f}", f"{r['memory'].get('peak_mb', float('nan')):.0f}",
                         r["digest"][:12]])
        S.append("## 1. Runs\n")
        S.append(table(["run", "status", "agent transitions", "updates", "minutes", "transitions/s", "peak MB",
                        "final digest"], rows))
        rp = tr.get("reproducibility", {})
        if rp.get("available"):
            S.append(f"\nSeed-0 re-run: final weights identical **{rp['identical_final_weights']}**; "
                     f"{rp['snapshots_compared']} snapshots compared, differing {len(rp['snapshots_differing'])}; histories "
                     f"identical (every non-wall-clock field) **{rp['identical_histories']}**.")
        bad = [k for k, v in tr["checks"].items() if not v]
        S.append(f"\nChecks: {len(tr['checks'])} evaluated, failed: {bad or 'none'}. All passed: **{tr['all_passed']}**.\n")
    # 2 stability
    if trn.get("seeds"):
        S.append("## 2. Training diagnostics by tenth of the run (seed mean; min-max over seeds in brackets)\n")
        keys = ("entropy", "approx_kl", "clipfrac", "explained_variance", "grad_norm", "value_loss", "act_cat_SH",
                "act_cat_Ward", "act_cat_Recall", "act_cat_TN", "act_beh_Trade", "act_beh_Safe")
        rows = []
        for k in keys:
            A = np.array([v["tenths"][k] for v in trn["seeds"].values()])
            rows.append([k] + [f"{A[:, i].mean():.3f} [{A[:, i].min():.3f}-{A[:, i].max():.3f}]" if i in (0, 4, 9)
                               else f"{A[:, i].mean():.3f}" for i in range(10)])
        S.append(table(["quantity"] + [f"t{i + 1}" for i in range(10)], rows))
        rows = []
        for k in trn["seeds"][next(iter(trn["seeds"]))]["episode_tenths"]:
            A = np.array([v["episode_tenths"][k] for v in trn["seeds"].values()])
            rows.append([k] + [f"{A[:, i].mean():.3f}" for i in range(10)])
        S.append("\nSelf-play episode statistics of the training rollouts (seed mean by tenth):\n")
        S.append(table(["quantity"] + [f"t{i + 1}" for i in range(10)], rows))
        rows = [[f"s{s}", f"{v['stability']['approx_kl_max']:.4f}", f"{v['stability']['approx_kl_max_10_update_mean']:.4f}",
                 f"{v['stability']['grad_norm_max']:.2f}", f"{v['stability']['entropy_min']:.2f}",
                 v["stability"]["nonfinite_values"], v["stability"]["alarms"] or "none"] for s, v in trn["seeds"].items()]
        S.append("\nPre-registered alarms per run:\n")
        S.append(table(["run", "KL max", "KL max (10-update mean)", "grad norm max", "entropy min", "non-finite",
                        "alarms"], rows))
    # 3 curve
    cv = d["curve"]
    if cv["pooled"]:
        S.append("\n## 3. Learning curve against the fixed suite (W = 460, 32 paired seeds)\n")
        rows = []
        for n in POINTS:
            p = cv["pooled"].get(str(n))
            if not p:
                continue
            rows.append([_ptlabel(n), _pm(*p["scripted"]), f"[{p['scripted_min_max'][0]:+.2f}, {p['scripted_min_max'][1]:+.2f}]",
                         _pm(*p["ood"])] + [_f(cv["per_seed"][str(s)][str(n)]["scripted"]) for s in seeds])
        S.append("Scripted-suite mean (mean of 9 opponents) and the OOD M4.4b network; seed mean ± t interval (6 seeds), "
                 "range over seeds, then every seed:\n")
        S.append(table(["checkpoint", "scripted mean", "seed range", "M4.4b (OOD)"] + [f"s{s}" for s in seeds], rows))
        rows = []
        for o in REFS:
            pm = cv["pooled"].get(str(FULL_MIDPOINT), {}).get("by_opp", {}).get(o, [float("nan")] * 2)
            pf = cv["pooled"].get(str(FULL_BUDGET), {}).get("by_opp", {}).get(o, [float("nan")] * 2)
            df = cv["final_minus_mid"]["by_opp"].get(o, [float("nan")] * 2)
            pi = cv["pooled"].get("initial", {}).get("by_opp", {}).get(o, [float("nan")] * 2)
            rows.append([o[4:], ROLE[o[4:]], _pm(*pi), _pm(*pm), _pm(*pf), _pm(*df)])
        S.append("\nBy opponent (seed mean ± t interval):\n")
        S.append(table(["opponent", "role", "untrained", "midpoint 4.096M", "final 8.192M", "final − midpoint"], rows))
        rows = []
        for key, lab in (("mid_minus_initial", "midpoint − untrained"), ("final_minus_mid", "final − midpoint"),
                         ("final_minus_q3", "final − 6.144M")):
            x = cv[key]
            rows.append([lab, _pm(*x["scripted"]), _pm(*x["ood"])] + [_pm(x["per_seed"][str(s)]["scripted"],
                                                                          x["per_seed"][str(s)]["half"])
                                                                      for s in seeds if str(s) in x["per_seed"]])
        S.append("\nPaired curve differences (scripted mean; per seed with its approximate interval):\n")
        S.append(table(["difference", "seed mean", "M4.4b (OOD)"] + [f"s{s}" for s in seeds], rows))
    # 4 h2h, plateau, extension
    hh, ru = d["h2h"], d["rules"]
    if hh["final_vs_mid"]["per_seed"]:
        S.append("\n## 4. Second half of the schedule: head-to-head (W = 1000, 64 paired seeds)\n")
        rows = []
        for s in seeds:
            a = hh["final_vs_mid"]["per_seed"].get(str(s))
            b = hh["final_vs_q3"]["per_seed"].get(str(s))
            pl = ru["plateau"].get(str(s), {})
            rows.append([f"s{s}", _pm(a["estimate"], a["half"]) if a else "n/a", _pm(b["estimate"], b["half"]) if b else "n/a",
                         _f(pl.get("curve_last_quarter")), pl.get("curve_ok"), pl.get("h2h_ok"), pl.get("plateaued")])
        rows.append(["seed mean", _pm(*hh["final_vs_mid"]["pooled"]), _pm(*hh["final_vs_q3"]["pooled"]), "", "", "", ""])
        S.append(table(["run", "final vs midpoint", "final vs 6.144M", "curve: final − 6.144M", "(a) curve < 0.3",
                        "(b) h2h within ±0.3, covers 0", "plateaued"], rows))
        e = ru["extension"]
        S.append(f"\nExtension rule: not plateaued {e['not_plateaued_count']} of {len(seeds)}; final − midpoint seed mean "
                 f"{_pm(*e['final_minus_mid_seed_mean'])}; seeds individually > +0.5 and significant: "
                 f"{e['seeds_individually_above_0.5_and_significant'] or 'none'}. **Criterion met: {e['criterion_met']}.**\n")
    # 5 history and cycling
    hi, cy = d["history"], d["cycle"]
    if any(hi.values()):
        S.append("## 5. The finals against their own past (W = 460, 32 paired seeds) and adjacent checkpoints\n")
        pts = sorted({int(k) for v in hi.values() for k in v})
        rows = [[_ptlabel(n if n else "initial")] + [_pm(hi[str(s)][str(n)]["estimate"], hi[str(s)][str(n)]["half"])
                                                     if str(n) in hi[str(s)] else "n/a" for s in seeds] for n in pts]
        S.append("Final vs earlier checkpoint (positive = the final wins):\n")
        S.append(table(["earlier checkpoint"] + [f"s{s}" for s in seeds], rows))
        pairs = sorted({(r["later"], r["earlier"]) for v in cy["per_seed"].values() for r in v})
        idx = {s: {(r["later"], r["earlier"]): r for r in v} for s, v in cy["per_seed"].items()}
        rows = [[f"{a / 1e6:.3f}M vs {b / 1e6:.3f}M"] + [_pm(idx[str(s)][(a, b)]["estimate"], idx[str(s)][(a, b)]["half"])
                                                       if (a, b) in idx.get(str(s), {}) else "n/a" for s in seeds]
                for a, b in pairs]
        S.append("\nEach checkpoint vs its predecessor (positive = the later wins):\n")
        S.append(table(["pair"] + [f"s{s}" for s in seeds], rows))
        S.append(f"\nSeeds with a significant regression (a later checkpoint losing to an earlier one): "
                 f"{cy['seeds_with_a_significant_regression'] or 'none'}; stop criterion (>= 2 seeds) met: "
                 f"**{cy['stop_criterion_met']}**.\n")
    # 6 cross
    cr = d["cross"]
    if np.isfinite(np.asarray(cr["matrix"], float)).any():
        S.append("## 6. Cross-play of the six finals (W = 1000, 64 paired seeds)\n")
        M, H = np.asarray(cr["matrix"]), np.asarray(cr["half"])
        rows = [[f"s{a}"] + [("—" if i == j else _pm(M[i, j], H[i, j])) for j in range(len(seeds))]
                + [_f(cr["row_mean"][str(a)])] for i, a in enumerate(seeds)]
        S.append(table(["row vs column"] + [f"s{s}" for s in seeds] + ["row mean"], rows))
        S.append(f"\nLargest |value| {cr['max_abs']:.2f}; pairs whose interval excludes 0: {cr['pairs_significant']} of "
                 f"{cr['pairs']}; seeds disagree (> {CROSS_GAP}): **{cr['seeds_disagree']}**.\n")
    xm = d.get("cross_mid") or {}
    if xm:
        S.append("### 6b. Does the second-half gain transfer? Each final vs the OTHER runs' midpoints (W = 1000, 64 "
                 "paired seeds; added after the head-to-head result, not pre-registered)\n")
        rows = [[f"s{s_}", _f(v["own_lineage_gain"]), _f(v["vs_other_midpoints"]), _f(v["transfer_gain"]),
                 _f(xm["gain_of_lineage"].get(s_))] for s_, v in xm["per_final"].items()]
        rows.append(["seed mean", _pm(*xm["own_lineage_gain"]), _pm(*xm["vs_other_midpoints"]),
                     _pm(*xm["transfer_gain"]), ""])
        S.append(table(["final", "vs own midpoint", "vs other midpoints (mean)", "transfer gain: vs other midpoints − "
                        "vs other finals", "this lineage's gain, measured by the other finals"], rows))
        S.append("")
    # 7 behaviour
    bh = d["behavior"]
    if bh["overall"]:
        S.append("## 7. Behaviour (self-play records, 16 games x 1000 turns, both seats focal)\n")
        rows = [[k.replace("|", " ")] + [f"{m[c]:.3f}" for c in CATS] + [f"{m['beh_' + b]:.3f}" for b in BEHS]
                + [f"{m['Pressure']:.3f}", f"{m['entropy']:.2f}"] for k, m in bh["overall"].items()]
        S.append("Mean probability mass, all present decisions (behaviours as shares of the lane mass):\n")
        S.append(table(["seed checkpoint", *CATS, *BEHS, "Pressure", "entropy"], rows))
        keys = list(next(iter(bh["contrasts"].values())).keys())
        fins = [k for k in bh["contrasts"] if k.endswith("|final")]
        rows = []
        for q in keys:
            st, c = q.split("::")
            rows.append([st, c] + [f"{bh['contrasts'][k][q][1]:.2f} ({bh['contrasts'][k][q][0]})"
                                   if bh["contrasts"][k][q][1] is not None else f"n/a ({bh['contrasts'][k][q][0]})"
                                   for k in fins])
        S.append("\nState-conditioned masses of the finals (mass (decisions)):\n")
        S.append(table(["state", "quantity"] + [f"s{k.split('|')[0]}" for k in fins], rows))
        qs = ("wards_per_1000", "wards_placed_per_1000", "recalls_per_1000", "deaths_per_1000",
              "trade_attempts_per_1000", "trades_connected_per_1000", "pressure_turns_per_1000", "collection_rate",
              "freeze_turns_share", "frozen_on_share", "lead_per_100_turns")
        rows = [[k.replace("|", " ")] + [_f(v[q], 3 if q.endswith(("rate", "share")) else 1, q == "lead_per_100_turns")
                                         for q in qs] for k, v in bh["summary"].items()]
        S.append("\nGame summaries (per 1000 focal turns unless stated):\n")
        S.append(table(["seed checkpoint / opponent", *qs], rows))
        rows = [[k.replace("|", " "), _f(v.get("per_1000"), 1, False), v.get("n", ""), _f(v.get("length"), 1, False),
                 _f(v.get("denial_share"), 2, False), _f(v.get("gold_diff")), _f(v.get("held_share"), 3, False),
                 _f(v.get("held_on_share"), 3, False)] for k, v in bh["freezes"].items()]
        S.append("\nFreezes (M4.4 definition: held >= 4 turns; denial share = episodes where the freezer out-earned the "
                 "frozen-on side):\n")
        S.append(table(["seed checkpoint / opponent", "per 1000", "n", "mean length", "denial share", "gold diff",
                        "held share", "frozen-on share"], rows))
        rows = [[k.replace("|", " "), _f(v["low_hp_per_1000"], 1, False), _f(v["other_per_1000"], 1, False),
                 _f(v["other_hp"], 2, False), _f(v["other_unspent"], 1, False), _f(v["other_bought_share"], 2, False)]
                for k, v in bh["recalls"].items()]
        S.append("\nRecalls (completed; low HP = HP fraction <= the M4.4 threshold):\n")
        S.append(table(["seed checkpoint", "low-HP per 1000", "other per 1000", "other: HP", "other: unspent gold",
                        "other: bought"], rows))
        lk = ("earned", "unc", "d_trade", "d_min", "d_tower", "d_gank", "deaths", "recalls", "rec_intr",
              "recall_interrupted_share", "wards", "ward_placed", "ward_intr", "vis_exposed", "held", "frozen",
              "minion_share_of_damage")
        rows = [[k.replace("|", " ")] + [_f(v[q], 3 if "share" in q else 2, False) for q in lk] for k, v in bh["ledger"].items()]
        S.append("\nM4.5b ledger, self-play (per 100 turns; 48 games x 300 turns):\n")
        S.append(table(["seed checkpoint", *lk], rows))
        if bh["early"]:
            rows = []
            for s, v in bh["early"].items():
                for st, x in v.items():
                    m = x["mass"]
                    if m:
                        rows.append([f"s{s}", st, x["n"]] + [f"{m[c]:.3f}" for c in CATS]
                                    + [f"{m['beh_' + b]:.3f}" for b in BEHS])
            S.append("\nEarly game: the finals in 64 self-play games of 120 turns, by level:\n")
            S.append(table(["seed", "levels", "n", *CATS, *BEHS], rows))
        if bh.get("power_gap"):
            rows = []
            for s_, v in bh["power_gap"].items():
                for side, m in (("stronger", v["stronger_mass"]), ("weaker", v["weaker_mass"])):
                    bp = v["by_power"].get(side, {})
                    if m:
                        rows.append([f"s{s_}", side, bp.get("n")] + [f"{m[c]:.3f}" for c in CATS]
                                    + [f"{m['beh_' + b]:.3f}" for b in BEHS]
                                    + [_f(bp.get("trade_attempt_rate"), 3, False), _f(bp.get("mean_pos"), 2),
                                       _f(bp.get("collection_rate"), 3, False), _f(bp.get("death_rate"), 4, False)])
            S.append("\nInitialized power gap (self-play from the neutral start, A x1.76; 16 games x 300 turns), by "
                     "power class of the deciding side:\n")
            S.append(table(["seed", "side", "n", *CATS, *BEHS, "trade att. rate", "mean pos", "collection",
                            "death rate"], rows))
            rows = []
            for s_, v in bh["power_gap"].items():
                for k, t in v["threat"].items():
                    b = t.get("behavior") or {}
                    rows.append([f"s{s_}", k, t.get("n")] + [_f(b.get(x), 2, False) for x in ("Hold", "Yield", "Trade",
                                                                                                "Safe")]
                                + [_f(t.get("mean_pos"), 2), _f(t.get("uncollected_per_turn"), 3, False),
                                   _f(t.get("xp_missed_per_turn"), 2, False)])
            S.append("\nWeaker side under a credible threat (M4.4 definition) vs the same side not threatened:\n")
            S.append(table(["seed", "turns", "n", "Hold", "Yield", "Trade", "Safe", "mean pos", "uncollected / turn",
                            "XP missed / turn"], rows))
    cp = d.get("comprehensive") or {}
    if cp:
        S.append("\n## 10. Comprehensive standard evaluation and the budget comparison\n")
    hl = cp.get("headline", {})
    if hl:
        rows = []
        for lab, v in hl.items():
            if "ood" not in v:
                continue
            rows.append([lab, _pm(*v["scripted"]), _pm(*v["ood"]), _pm(*v["seat_A"]), _pm(*v["seat_B"])]
                        + [_f(v["per_seed"].get(str(s), {}).get("scripted")) for s in seeds])
        S.append("Headline (W = 1000, 64 paired seeds, both seats): scripted-suite mean and the OOD M4.4b network:\n")
        S.append(table(["policies", "scripted mean", "M4.4b (OOD)", "seat A", "seat B"] + [f"s{s}" for s in seeds], rows))
        if "final minus 8.192M final" in hl:
            x = hl["final minus 8.192M final"]
            rows = [[o[4:] if o.startswith("ref:") else "M4.4b (OOD)", _pm(*hl["final"]["by_opp"][o]),
                     _pm(*hl["8.192M final"]["by_opp"][o]), _pm(*x["by_opp"][o])] for o in x["by_opp"]]
            S.append("\nBy opponent (seed mean ± t interval): this campaign's finals, the 8.192M finals, difference:\n")
            S.append(table(["opponent", "finals", "8.192M finals", "difference"], rows))
    for part, title in (("vs_main", "Finals (rows) vs the 8.192M finals (columns), W = 1000, 64 paired seeds"),
                        ("anneal", "The 8.192M finals (rows, annealed to 0) vs this campaign's midpoints (columns, "
                                   "same 8,192,000 transitions, learning rate at 50%)")):
        m = cp.get(part)
        if not m:
            continue
        M, H = np.asarray(m["matrix"]), np.asarray(m["half"])
        rows = [[f"s{seeds[i]}"] + [_pm(M[i, j], H[i, j]) for j in range(len(seeds))] for i in range(len(seeds))]
        S.append(f"\n{title}:\n")
        S.append(table(["row \\ column"] + [f"s{s}" for s in seeds], rows))
        S.append(f"\nSame seed: {_pm(*m['same_seed_mean'])}; cross-seed: {_pm(*m['cross_seed_mean'])}; pairs with "
                 f"the interval above 0: {m['pairs_significant_positive']} of {m['pairs']}, below 0: "
                 f"{m['pairs_significant_negative']}.\n")
    hs = cp.get("historical", {})
    if hs:
        rows = [[k.split("|")[0], k.split("|")[1], _pm(*v["mean"]), " ".join(f"{x:+.2f}" for x in v["per_seed"])]
                for k, v in hs.items()]
        S.append("\nHistorical PPO references (W = 1000, 64 paired seeds):\n")
        S.append(table(["policies", "reference", "seed mean", "per seed"], rows))
    tt = cp.get("tournament")
    if tt:
        rows = [[f"s{s}"] + [f"{x:+.2f}" for x in v["score"]] + [len(v["later_loses_significantly"])]
                for s, v in tt["per_seed"].items()]
        S.append("\nPer-run tournament of 9 checkpoints (W = 460, 16 paired seeds): each checkpoint's mean value "
                 "against the other eight, and the number of (earlier, later) pairs where the later loses "
                 "significantly:\n")
        S.append(table(["run"] + [f"{n / 1e6:.2f}M" for n in tt["checkpoints"]] + ["regressions"], rows))
        S.append(f"\nRuns with a significant regression: {tt['seeds_with_regression'] or 'none'}.\n")
    # timing
    if d["timing"]:
        rows = [[k, f"{v['turns'] / 1e6:.2f}", f"{v['wall_s'] / 60:.1f}", v["workers"], f"{v['turns_per_worker_s']:,.0f}"]
                for k, v in d["timing"].items()]
        S.append("\n## 8. Evaluation cost\n")
        S.append(table(["part", "simulated turns (M)", "minutes", "workers", "turns per worker-second"], rows))
    try:
        e = estimate(d, tr)
        d["estimate"] = e
        save_json("summary", d, FULL_OUT)
        rows = [[k, f"{v:.2f}"] for k, v in e["parts_h"].items()] + [["margin (15%)", f"{e['margin_h']:.2f}"],
                                                                      ["**total**", f"**{e['total_h']:.1f}**"]]
        S.append("\n## 9. Updated estimate of the comprehensive evaluation (hours, 11 workers)\n")
        S.append(table(["component", "hours"], rows))
        S.append("\nStandard parts: " + "; ".join(f"{k} {v:.2f} h" for k, v in e["standard_parts_h"].items())
                 + f". Battery: {e['battery']['total_turns'] / 1e6:.1f} M turns. Rates measured here: "
                 + ", ".join(f"{k} {v:,.0f}" for k, v in e["rates"].items()) + ".\n")
    except Exception as ex:                                # pragma: no cover - estimate inputs missing
        S.append(f"\n(estimate unavailable: {ex!r})\n")
    S.append("\nFigures: " + ", ".join(f"`{os.path.basename(p)}`" for p in figs if p) + " (results/m45c/full/figures/).\n")
    text = "\n".join(S)
    write_md("report", text, FULL_OUT)
    return text


__all__ = ["analyze", "report", "comprehensive", "_use", "curve", "h2h", "history", "cycle", "cross", "rules", "behavior"]
