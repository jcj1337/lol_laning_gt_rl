"""Stage 2 readouts and the pre-registered classification (docs/m4/m4_7/m4_7_preregistration.md §5-§6).

Units. The two training seeds are the independent units; every readout is reported per seed with a within-seed 95%
interval over games (normal, game pairs as units), and the classification requires both seeds to agree. Leads are
the SHORT arm's paired-seat earned-gold lead: game m is played with each policy in each seat on the same chance seed,
d_m(t) = (r_m(short in A)(t) - r_m(control in A)(t)) / 2.

Common objectives (the same trajectories scored several ways; raw discounted returns under different gammas are never
compared with each other):
    lane      undiscounted lead over turns 0-93 (the primary P1; elapsed-turn window)
    windows   undiscounted lead in each common.WINDOWS window
    level14   undiscounted lead until the lower level of the (short in A) game reaches 14 (level-triggered)
    G<H>      discounted lead under each candidate horizon over the game (400 turns, no critic bootstrap)
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from . import behaviour as B
from . import common as C
from . import delayed as D
from . import evaluate as EV
from . import streams as SM

MATERIALITY = 0.25              # resources: the primary's and the regression criterion's materiality
B_KEYS = ("recall_choices_per_100", "recalls_per_100", "hp_at_recall", "unspent_at_recall", "shove", "hold", "yield",
          "trade", "safe", "deaths_per_100", "held_share", "frozen_on_shove", "ward_share", "pressure_share",
          "invested_end", "unspent_end", "earned_per_turn")


def _ci(x) -> dict:
    return C.mean_se(x)


def lead_readouts(rx: np.ndarray, ry: np.ndarray, Lx: np.ndarray) -> dict:
    """Readouts of x's paired-seat lead from per-game reward streams rx (x in A) and ry (y in A)."""
    d = 0.5 * (rx.astype(float) - ry.astype(float))
    out = {"games": int(d.shape[0]), "lane": _ci(d[:, :C.LANE_END].sum(1))}
    for name, (lo, hi) in C.WINDOWS.items():
        out[f"win_{name}"] = _ci(d[:, lo:hi].sum(1))
    out["after_lane"] = _ci(d[:, C.LANE_END:].sum(1))
    cut = SM.level_cut(Lx, 14)
    m = np.arange(d.shape[1])[None, :] < cut[:, None]
    out["level14"] = _ci((d * m).sum(1))
    out["level14_turn"] = _ci(cut)
    for h, g in C.HORIZONS.items():
        out[f"G{h}"] = _ci(SM.discounted(d, g))
    out["G_short"] = _ci(SM.discounted(d, C.SHORT_GAMMA))
    return out


def _diff_unpaired(a: dict, b: dict) -> dict:
    """The difference of two readouts with independent errors. NOT the protocol's interval (pre-registration §5 pairs
    game indices); kept only to report the analysis code's first, unpaired version (report §5.3)."""
    m = a["mean"] - b["mean"]
    se = float(np.hypot(a["se"], b["se"]))
    return {"mean": m, "se": se, "ci": 1.96 * se}


def head_to_head() -> dict:
    out = {}
    for s in C.PILOT_SEEDS:
        f = EV.finals(s)
        p = EV.load_pair("h2h", s, f["short"], f["control"])
        out[f"s{s}"] = {"final": lead_readouts(p["rx"], p["ry"], p["Lx"])}
        ms, mc = f"p/short:{s}@{C.MIDPOINT}", f"p/control:{s}@{C.MIDPOINT}"
        q = EV.load_pair("h2h", s, ms, mc)
        out[f"s{s}"]["midpoint"] = lead_readouts(q["rx"], q["ry"], q["Lx"])
    return out


def fixed_opponents() -> dict:
    from ..m45c.references import SUITE
    out = {}
    for s in C.PILOT_SEEDS:
        f = EV.finals(s)
        start = C.general("baseline", s)
        rows, raw = {}, {}
        for xn, x in (("short", f["short"]), ("control", f["control"]), ("start", start)):
            groups = {}
            if xn != "start":
                p = EV.load_pair("fixed", s, x, start)
                groups["start"] = (p["rx"], p["ry"], p["Lx"])
            parts = [EV.load_pair("fixed", s, x, C.general("baseline", o)) for o in EV.OTHER_SEEDS]
            groups["other_finals"] = tuple(np.concatenate([q[k] for q in parts]) for k in ("rx", "ry", "Lx"))
            parts = [EV.load_pair("fixed", s, x, f"ref:{r}") for r in SUITE]
            groups["scripted"] = tuple(np.concatenate([q[k] for q in parts]) for k in ("rx", "ry", "Lx"))
            groups["fixed_pool"] = tuple(np.concatenate([groups["other_finals"][i], groups["scripted"][i]])
                                         for i in range(3))
            raw[xn] = groups
            rows[xn] = {g: lead_readouts(*v) for g, v in groups.items()}
            rows[xn]["per_ref"] = {r: lead_readouts(*(EV.load_pair("fixed", s, x, f"ref:{r}")[k]
                                                      for k in ("rx", "ry", "Lx"))) for r in SUITE}
        # short - control on the same games (the same opponents, game seeds and seats): per-game paired differences
        diff, unpaired = {}, {}
        for g in ("start", "other_finals", "scripted", "fixed_pool"):
            (sx, sy, sl), (cx, cy, _) = raw["short"][g], raw["control"][g]
            diff[g] = lead_readouts(sx.astype(float) - cx.astype(float), sy.astype(float) - cy.astype(float), sl)
            sh, co = rows["short"][g], rows["control"][g]
            unpaired[g] = {k: _diff_unpaired(sh[k], co[k]) for k in sh if isinstance(sh[k], dict) and "se" in sh[k]}
        out[f"s{s}"] = {"by_policy": rows, "short_minus_control": diff, "short_minus_control_unpaired": unpaired}
    return out


def natural_behaviour() -> dict:
    from .timeline import policy_summary
    thr = B.thresholds()
    out = {"thresholds": thr}
    for s in C.PILOT_SEEDS:
        f = EV.finals(s)
        start = C.general("baseline", s)
        recs = {"short": EV.load_natural(f["short"]), "control": EV.load_natural(f["control"]),
                "start": EV.load_natural(start)}
        out[f"s{s}"] = {"summary": {k: B.summary(v) for k, v in recs.items()},
                        "short_minus_control": B.paired(recs["short"], recs["control"]),
                        "control_minus_start": B.paired(recs["control"], recs["start"]),
                        "short_minus_start": B.paired(recs["short"], recs["start"]),
                        "timeline": {k: {"t14": policy_summary(v)["timeline"]["14"]["pooled"],
                                         "recall": policy_summary(v)["recall"],
                                         "investment": policy_summary(v)["investment"]} for k, v in recs.items()}}
    return out


def cells() -> dict:
    tl = C.load("timeline")
    out = {}
    for s in C.PILOT_SEEDS:
        row = {}
        for arm in ("start",) + C.ARMS:
            def pf(fam):
                return lambda idx: EV.cell_path(fam, arm, s, idx)
            row[arm] = {"cells": D.analyse_cells("baseline", s, tl, pf("cells")),
                        "natural": D.analyse_natural("baseline", s, pf("natural")),
                        "freeze": D.analyse_freeze("baseline", s, tl, pf("freeze"))}
        out[f"s{s}"] = row
    return out


# ---------------------------------------------------------------------------------------------- classification
def _sig(x: dict, sign: int, m: float) -> bool:
    """x clears materiality m in direction sign and its 95% interval excludes 0."""
    return sign * x["mean"] >= m and sign * x["mean"] - x["ci"] > 0


def classify(h2h: dict, fixed: dict, nat: dict) -> dict:
    seeds = [f"s{s}" for s in C.PILOT_SEEDS]
    p1 = {k: h2h[k]["final"]["lane"] for k in seeds}
    benefit = all(_sig(p1[k], +1, MATERIALITY) for k in seeds)
    lane_regress = all(_sig(p1[k], -1, MATERIALITY) for k in seeds)
    r_full = {k: fixed[k]["short_minus_control"]["fixed_pool"]["G100"] for k in seeds}
    r_after = {k: fixed[k]["short_minus_control"]["fixed_pool"]["after_lane"] for k in seeds}
    regress = all(_sig(r_full[k], -1, MATERIALITY) for k in seeds) or \
        all(_sig(r_after[k], -1, MATERIALITY) for k in seeds)
    thr = nat["thresholds"]["lane"]
    changed = {}
    for key in B_KEYS:
        xs = [nat[k]["short_minus_control"]["lane"][key] for k in seeds]
        signs = {int(np.sign(x["mean"])) for x in xs}
        ok = len(signs) == 1 and 0 not in signs and all(
            abs(x["mean"]) >= thr[key] and abs(x["mean"]) - x["ci"] > 0 for x in xs)
        changed[key] = {"meaningful": bool(ok), "per_seed": [x["mean"] for x in xs], "threshold": thr[key]}
    behaviour = any(v["meaningful"] for v in changed.values())
    small = all(abs(p1[k]["mean"]) < MATERIALITY or abs(p1[k]["mean"]) - p1[k]["ci"] <= 0 for k in seeds)
    if benefit and not regress:
        outcome, rec = 1, "benefit: pursue the shorter horizon (confirm before any adoption)"
    elif benefit and regress:
        outcome, rec = 2, "trade-off: the lane phase gains, ordinary play loses; a modelling decision"
    elif lane_regress or (regress and not benefit):
        outcome, rec = 5, "regression: keep the current horizon"
    elif small and behaviour:
        outcome, rec = 3, "behaviour changes without a lane-phase benefit: unresolved, keep the current horizon " \
                          "unless the behaviour change itself is wanted"
    elif small and not behaviour:
        outcome, rec = 4, "no meaningful change: keep the current horizon"
    else:
        outcome, rec = 6, "inconclusive: the question stays unresolved"
    return {"outcome": outcome, "recommendation": rec, "P1": p1, "benefit": benefit, "lane_regression": lane_regress,
            "regression": regress, "regression_full": r_full, "regression_after": r_after, "behaviour": changed,
            "behaviour_changed": behaviour, "materiality": MATERIALITY}


def run() -> dict:
    C.require_prereg()
    from .train import check
    data = {"meta": C.meta(stage="analysis"), "short_gamma": C.SHORT_GAMMA, "short_h": C.SHORT_H,
            "training": check(save=False)}
    parts = {"h2h": head_to_head, "fixed": fixed_opponents, "natural": natural_behaviour, "cells": cells}
    for k, fn in parts.items():
        try:
            data[k] = fn()
        except FileNotFoundError as ex:                     # a family not yet evaluated
            data[k] = {"missing": repr(ex)}
    if all("missing" not in data[k] for k in ("h2h", "fixed", "natural")):
        data["classification"] = classify(data["h2h"], data["fixed"], data["natural"])
        # sensitivity (amendment of 2026-10-16): the first code's unpaired fixed-opponent interval
        unp = {k: {"short_minus_control": v["short_minus_control_unpaired"]} for k, v in data["fixed"].items()}
        data["classification_unpaired_fixed_interval"] = classify(data["h2h"], unp, data["natural"])
    C.save_json("analysis", data)
    return data


__all__ = ["lead_readouts", "head_to_head", "fixed_opponents", "natural_behaviour", "cells", "classify", "run",
           "MATERIALITY", "B_KEYS"]
