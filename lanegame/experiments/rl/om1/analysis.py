"""OM1 readouts (protocol §4): selection, identifiability, value of information, robustness, classification.

Units. Value contrasts are paired on common random numbers within a cell (the same games and streams for every
responder). Population claims average the 15 library types per seed and use the six seeds as units (t interval, 5 df);
per-type tables pool the seeds' pairs.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import common as C
from . import stages as ST

TYPES = C.LIBRARY
NT = len(TYPES)


def _cell_values(d: dict, sel: dict, opp: str, library_opp: bool) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-pair values (lead at READ_TURN and at T_EVAL) of pi, oracle, fixed and switch in one cell; the oracle and
    the fixed responder fall back to the responder they coincide with."""
    v = {}
    for lab in d["labels"]:
        R = d[f"{lab}__r"]
        v[lab] = {"read": ST.pair_lead(R, C.READ_TURN), "end": ST.pair_lead(R, R.shape[-1]),
                  "lane": ST.pair_lead(R, C.LANE_END)}
    if library_opp:
        if "oracle" not in v:
            v["oracle"] = v["pi"]
        rfix = sel["rfix"]
        if "fixed" not in v:
            v["fixed"] = v["pi"] if rfix == "pi" else v["oracle"]
    return v


def _diff(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], key: str = "read") -> np.ndarray:
    return a[key] - b[key]


# ---------------------------------------------------------------------------------------------- identifiability
def identifiability(sel: dict) -> dict:
    rstar = sel["rstar"]
    out = {"per_type": {}}
    variants = ("full", "coarse", "eps0.01", "eps0.2")
    agg = {v: {t: [] for t in C.CHECK_TURNS} for v in variants}
    agg_map = {t: [] for t in C.CHECK_TURNS}
    agg_dec = {t: [] for t in C.CHECK_TURNS}
    agg_dec_c = {t: [] for t in C.CHECK_TURNS}
    for ti, typ in enumerate(TYPES):
        P = {v: [] for v in variants}
        for s in C.SEEDS:
            d = ST.load(ST.raw("eval", f"eval|s{s}|{typ}"))
            for v in variants:
                P[v].append(np.exp(d[f"pi__bel_{v}"].astype(np.float64)).reshape(-1, len(C.CHECK_TURNS), NT))
        P = {v: np.concatenate(x) for v, x in P.items()}          # [games, checks, types]
        o = {"games": int(P["full"].shape[0])}
        for v in variants:
            o[f"p_true_ge_{C.ID_P}_{v}"] = {str(t): float((P[v][:, j, ti] >= C.ID_P).mean())
                                            for j, t in enumerate(C.CHECK_TURNS)}
        mp = P["full"].argmax(-1)
        mpc = P["coarse"].argmax(-1)
        o["map_correct"] = {str(t): float((mp[:, j] == ti).mean()) for j, t in enumerate(C.CHECK_TURNS)}
        dec = np.vectorize(lambda k: rstar[TYPES[k]] == rstar[typ])
        o["decision_relevant"] = {str(t): float(dec(mp[:, j]).mean()) for j, t in enumerate(C.CHECK_TURNS)}
        o["decision_relevant_coarse"] = {str(t): float(dec(mpc[:, j]).mean()) for j, t in enumerate(C.CHECK_TURNS)}
        j40 = C.CHECK_TURNS.index(C.ID_TURN)
        conf = np.bincount(mp[:, j40], minlength=NT) / mp.shape[0]
        o["map_at_40"] = {TYPES[k]: float(conf[k]) for k in np.argsort(-conf)[:4] if conf[k] > 0}
        # turns of observation until the true type's posterior first reaches ID_P (inf if not by turn 94)
        hit = P["full"][:, :, ti] >= C.ID_P
        first = np.where(hit.any(1), np.asarray(C.CHECK_TURNS)[hit.argmax(1)], np.inf)
        o["first_check_identified_median"] = float(np.median(first))
        out["per_type"][typ] = o
        for t in C.CHECK_TURNS:
            for v in variants:
                agg[v][t].append(o[f"p_true_ge_{C.ID_P}_{v}"][str(t)])
            agg_map[t].append(o["map_correct"][str(t)])
            agg_dec[t].append(o["decision_relevant"][str(t)])
            agg_dec_c[t].append(o["decision_relevant_coarse"][str(t)])
    out["type_mean"] = {f"p_true_ge_{C.ID_P}_{v}": {str(t): float(np.mean(agg[v][t])) for t in C.CHECK_TURNS}
                        for v in variants}
    out["type_mean"]["map_correct"] = {str(t): float(np.mean(agg_map[t])) for t in C.CHECK_TURNS}
    out["type_mean"]["decision_relevant"] = {str(t): float(np.mean(agg_dec[t])) for t in C.CHECK_TURNS}
    out["type_mean"]["decision_relevant_coarse"] = {str(t): float(np.mean(agg_dec_c[t])) for t in C.CHECK_TURNS}
    out["I_met"] = out["type_mean"]["decision_relevant"][str(C.ID_TURN)] >= 2 / 3
    return out


# ---------------------------------------------------------------------------------------------- value of information
def value(sel: dict) -> dict:
    per_seed = {k: {} for k in ("G_B", "G_O", "VOI", "G_B_end", "G_O_end", "G_B_lane", "G_O_lane", "V_pi")}
    per_type = {}
    sw_stats = {}
    cells = {}
    for typ in TYPES:
        pooled = {k: [] for k in ("B-pi", "O-pi", "O-fix", "B-pi_end", "O-pi_end", "pi", "O")}
        sw = []
        for s in C.SEEDS:
            d = ST.load(ST.raw("eval", f"eval|s{s}|{typ}"))
            v = _cell_values(d, sel, typ, True)
            cells[(s, typ)] = v
            pooled["B-pi"].append(_diff(v["switch"], v["pi"]))
            pooled["O-pi"].append(_diff(v["oracle"], v["pi"]))
            pooled["O-fix"].append(_diff(v["oracle"], v["fixed"]))
            pooled["B-pi_end"].append(_diff(v["switch"], v["pi"], "end"))
            pooled["O-pi_end"].append(_diff(v["oracle"], v["pi"], "end"))
            pooled["pi"].append(v["pi"]["read"])
            pooled["O"].append(v["oracle"]["read"])
            sw += d["switch"].get("switch", [])
        per_type[typ] = {k: C.mean_se(np.concatenate(x)) for k, x in pooled.items()}
        per_type[typ]["rstar"] = sel["rstar"][typ]
        chose = {}
        for g in sw:
            for k, n in g["choices"].items():
                chose[k] = chose.get(k, 0) + n
        tot = max(1, sum(chose.values()))
        sw_stats[typ] = {"switched_share": float(np.mean([g["switched_turns"] for g in sw]) / C.T_EVAL),
                         "first_switch_median": float(np.median([g["first_switch"] if g["first_switch"] >= 0
                                                                 else np.inf for g in sw])),
                         "never_switched": float(np.mean([g["first_switch"] < 0 for g in sw])),
                         "turn_share_by_responder": {k: n / tot for k, n in sorted(chose.items(), key=lambda z: -z[1])}}
    for s in C.SEEDS:
        per_seed["G_B"][s] = float(np.mean([_diff(cells[(s, t)]["switch"], cells[(s, t)]["pi"]).mean() for t in TYPES]))
        per_seed["G_O"][s] = float(np.mean([_diff(cells[(s, t)]["oracle"], cells[(s, t)]["pi"]).mean() for t in TYPES]))
        per_seed["VOI"][s] = float(np.mean([_diff(cells[(s, t)]["oracle"], cells[(s, t)]["fixed"]).mean()
                                            for t in TYPES]))
        per_seed["G_B_end"][s] = float(np.mean([_diff(cells[(s, t)]["switch"], cells[(s, t)]["pi"], "end").mean()
                                                for t in TYPES]))
        per_seed["G_O_end"][s] = float(np.mean([_diff(cells[(s, t)]["oracle"], cells[(s, t)]["pi"], "end").mean()
                                                for t in TYPES]))
        per_seed["G_B_lane"][s] = float(np.mean([_diff(cells[(s, t)]["switch"], cells[(s, t)]["pi"], "lane").mean()
                                                 for t in TYPES]))
        per_seed["G_O_lane"][s] = float(np.mean([_diff(cells[(s, t)]["oracle"], cells[(s, t)]["pi"], "lane").mean()
                                                 for t in TYPES]))
        per_seed["V_pi"][s] = float(np.mean([cells[(s, t)]["pi"]["read"].mean() for t in TYPES]))
    return {"seeds": {k: C.seed_stats(v) for k, v in per_seed.items()}, "per_type": per_type, "switcher": sw_stats}


# ---------------------------------------------------------------------------------------------- robustness
def robustness(sel: dict) -> dict:
    per_opp, per_seed = {}, {s: [] for s in C.SEEDS}
    for o in C.OOL:
        diffs, diffs_end, maps, sws = [], [], [], []
        for s in C.SEEDS:
            d = ST.load(ST.raw("ool", f"ool|s{s}|{o}"))
            v = _cell_values(d, sel, o, False)
            x = _diff(v["switch"], v["pi"])
            diffs.append(x)
            diffs_end.append(_diff(v["switch"], v["pi"], "end"))
            per_seed[s].append(float(x.mean()))
            P = np.exp(d["pi__bel_full"].astype(np.float64)).reshape(-1, len(C.CHECK_TURNS), NT)
            maps.append(P[:, C.CHECK_TURNS.index(C.ID_TURN)].argmax(-1))
            sws += d["switch"].get("switch", [])
        mp = np.concatenate(maps)
        conf = np.bincount(mp, minlength=NT) / len(mp)
        chose = {}
        for g in sws:
            for k, n in g["choices"].items():
                chose[k] = chose.get(k, 0) + n
        tot = max(1, sum(chose.values()))
        per_opp[o] = {"B-pi": C.mean_se(np.concatenate(diffs)), "B-pi_end": C.mean_se(np.concatenate(diffs_end)),
                      "mistaken_for_at_40": {TYPES[k]: float(conf[k]) for k in np.argsort(-conf)[:3] if conf[k] > 0},
                      "turn_share_by_responder": {k: n / tot for k, n in sorted(chose.items(), key=lambda z: -z[1])}}
    pooled = C.seed_stats({s: float(np.mean(v)) for s, v in per_seed.items()})
    violated = pooled["mean"] <= -C.M and pooled["mean"] + pooled["ci"] < 0
    return {"per_opponent": per_opp, "pooled_per_seed": pooled, "R_violated": bool(violated)}


# ---------------------------------------------------------------------------------------------- classification
def classify(ident: dict, val: dict, rob: dict) -> dict:
    I = ident["I_met"]
    gb, go = val["seeds"]["G_B"], val["seeds"]["G_O"]
    gb_ok = gb["mean"] >= C.M and gb["mean"] - gb["ci"] > 0
    go_small = go["mean"] < C.M or go["mean"] - go["ci"] <= 0
    narrow = []
    for t in TYPES:
        o = val["per_type"][t]["O-pi"]
        dec = ident["per_type"][t]["decision_relevant"][str(C.ID_TURN)]
        if o["mean"] >= 2 * C.M and o["mean"] - o["ci"] > 0 and dec >= 2 / 3:
            narrow.append(t)
    if I and gb_ok:
        outcome, label = 1, "feasible and worthwhile"
    elif narrow:
        outcome, label = 2, "narrow: worthwhile for specific types"
    elif I and go_small:
        outcome, label = 3, "identifiable, little value"
    elif not I:
        outcome, label = 4, "not identifiable"
    else:
        outcome, label = 0, "unclassified (I met, G_O material, G_B not): see the report"
    return {"outcome": outcome, "label": label, "I_met": I, "G_B_material": bool(gb_ok),
            "G_O_small": bool(go_small), "narrow_types": narrow, "R_violated": rob["R_violated"]}


def run() -> dict:
    sel = C.load("selection")
    ident = identifiability(sel)
    val = value(sel)
    rob = robustness(sel)
    out = {"meta": C.meta(stage="analyse"), "selection": sel, "identifiability": ident, "value": val,
           "robustness": rob, "classification": classify(ident, val, rob)}
    C.save_json("analysis", out)
    c = out["classification"]
    print(f"  outcome {c['outcome']}: {c['label']}; narrow types {c['narrow_types']}; R violated {c['R_violated']}",
          flush=True)
    return out


__all__ = ["identifiability", "value", "robustness", "classify", "run"]
