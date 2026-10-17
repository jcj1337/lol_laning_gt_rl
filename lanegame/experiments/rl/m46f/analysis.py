"""Readouts of M4.6f (docs/m4/m4_6/m4_6f_preregistration.md): value, behaviour, training, the extension rule and the
pre-registered outcome mapping. Every between-arm statistic is computed per seed pair (separate - shared, the same s)
and summarized by the mean and the 95% t interval over the six seeds (5 df): seeds are the independent units; the
rollouts within a seed only make each seed's estimate precise.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from . import behaviour as B
from . import common as C
from . import value as V

# pre-registered materiality (m) and equivalence (e) margins
MARGINS = {"V1": (0.25, 0.15), "B1": (0.20, 0.30), "B2": (0.15, 0.25)}
V1_FLOOR = 0.5                  # the separate arm's own ratio required for a V1 benefit
V2_REGRESSION = 0.10
GUARD_REGRESSION = 0.20


def classify(st: dict, m: float, e: float) -> str:
    """benefit / regression / small / no material difference / inconclusive (pre-registration). Precedence where the
    definitions overlap: an interval that excludes 0 is "small" (not "no material difference") unless |mean| >= m."""
    mu, h = st.get("mean", np.nan), st.get("ci", np.nan)
    if not (np.isfinite(mu) and np.isfinite(h)):
        return "n/a"
    lo, hi = mu - h, mu + h
    if lo > 0 and mu >= m:
        return "benefit"
    if hi < 0 and mu <= -m:
        return "regression"
    if lo > 0 or hi < 0:
        return "small"
    if lo > -e and hi < e:
        return "no material difference"
    return "inconclusive"


def stats(per_seed: Dict[int, float]) -> dict:
    return C.seed_stats({s: v for s, v in per_seed.items() if v is not None and np.isfinite(v)})


# ---------------------------------------------------------------------------------------------- value
def _calib_tags(phase: str) -> Dict[str, Dict[int, str]]:
    return {a: {s: V._tag(a, s, phase) for s in C.SEEDS} for a in C.ARMS}


def value_calib(phase: str = "main", K: Optional[int] = None) -> dict:
    """Per arm and seed: M4.6e P1 / P2 readouts on the M4.6d set; the starting critics from M4.6d's records (the
    same states and rollout seeds; K 16 subset in phase "mid"). `K` restricts every network to its first K rollouts."""
    K = V.K_CAL_MID if phase == "mid" else K
    per = {"start": {}, "shared": {}, "separate": {}}
    for s in C.SEEDS:
        d = V.start_calib_dir(s)
        ext = V.load_calib(d, K)
        if phase == "mid":
            keep = ~np.char.startswith(ext["fam"].astype(str), "natural")
            ext = {k: v[keep] for k, v in ext.items()}
        per["start"][s] = V.calib_readouts(ext)
        for a in C.ARMS:
            d = V.calib_dir(V._tag(a, s, phase))
            if os.path.isdir(d) and len(os.listdir(d)) >= 42:
                per[a][s] = V.calib_readouts(V.load_calib(d, K))
    out = {"per_seed": per}

    def g(r, *path):
        x = r
        for p in path:
            if not isinstance(x, dict) or p not in x:
                return np.nan
            x = x[p]
        return x if isinstance(x, (int, float)) else np.nan

    readouts = {"edge+plaus ratio": ("classes", "edge+plaus", "ratio"), "edge ratio": ("classes", "edge", "ratio"),
                "plaus ratio": ("classes", "plaus", "ratio"), "large ratio": ("classes", "large", "ratio"),
                "gap ratio": ("classes", "gap", "ratio"), "large+gap ratio": ("classes", "large+gap", "ratio"),
                "edge+plaus predicted": ("premiums", "edge+plaus", "pred"),
                "edge+plaus realized": ("premiums", "edge+plaus", "realized"),
                "x1.76 predicted": ("premiums", "x1.76", "pred"), "x1.76 realized": ("premiums", "x1.76", "realized"),
                "bias freeze (A)": ("bias", "freeze"), "bias lane (A)": ("bias", "lane"), "bias gap (A)": ("bias", "gap"),
                "natural MAE": ("natural", "mae"), "natural bias": ("natural", "bias"),
                "natural slope": ("natural", "slope"), "natural R2": ("natural", "r2")}
    tab = {}
    for name, path in readouts.items():
        row = {}
        for arm in ("start",) + C.ARMS:
            row[arm] = stats({s: g(per[arm].get(s, {}), *path) for s in C.SEEDS})
        vals = {arm: {s: g(per[arm].get(s, {}), *path) for s in C.SEEDS} for arm in ("start",) + C.ARMS}
        row["separate-shared"] = stats(C.per_seed_diff(vals["separate"], vals["shared"]))
        row["shared-start"] = stats(C.per_seed_diff(vals["shared"], vals["start"]))
        row["separate-start"] = stats(C.per_seed_diff(vals["separate"], vals["start"]))
        tab[name] = row
    out["table"] = tab
    return out


def value_natural(phase: str = "main") -> dict:
    tags = {a: {s: V._tag(a, s, phase) for s in C.SEEDS} for a in C.ARMS}
    tags["start"] = {s: f"start_s{s}" for s in C.SEEDS}            # the starts are labelled once (phase main)
    per = {}
    for arm, ts in tags.items():
        per[arm] = {}
        for s, t in ts.items():
            d = os.path.join(C.CKPT, "value", "natural", t)
            if os.path.isdir(d) and len(os.listdir(d)) == V.NAT_GAMES // 8:
                per[arm][s] = V.natural_readouts(V.load_natural(t))
    tab = {}
    for k in ("mae", "bias", "slope", "r2", "corr"):
        vals = {arm: {s: per[arm][s].get(k, np.nan) for s in per[arm]} for arm in per}
        row = {arm: stats(v) for arm, v in vals.items()}
        if all(a in vals for a in C.ARMS):
            row["separate-shared"] = stats(C.per_seed_diff(vals["separate"], vals["shared"]))
        tab[k] = row
    drift = {}
    if phase in ("main", "ext"):
        for s in C.SEEDS:
            try:
                drift[s] = V.drift(s, phase)
            except Exception as ex:                      # pragma: no cover - missing records
                drift[s] = {"error": repr(ex)}
        dt = {}
        for k in ("kl_start_shared", "kl_start_separate", "tv_shared_separate"):
            dt[k] = stats({s: d.get(k, np.nan) for s, d in drift.items()})
        dt["kl separate - shared"] = stats({s: d["kl_start_separate"] - d["kl_start_shared"] for s, d in drift.items()
                                            if "kl_start_shared" in d})
        drift = {"per_seed": drift, "table": dt}
    return {"per_seed": per, "table": tab, "drift": drift}


def value_probes(phase: str = "main") -> dict:
    tags = {a: {s: V._tag(a, s, phase) for s in C.SEEDS} for a in C.ARMS}
    tags["start"] = {s: f"start_s{s}" for s in C.SEEDS}            # the starts are probed once (phase main)
    per = {}
    for arm, ts in tags.items():
        per[arm] = {}
        for s, t in ts.items():
            p = os.path.join(C.CKPT, "value", "probes", f"{t}.json.gz")
            if os.path.exists(p):
                per[arm][s] = C.load_gz(p)
    tab = {}
    keys = ("log_duel<-value_features", "log_duel<-actor_trunk", "hp_own<-value_features", "hp_own<-actor_trunk",
            "value_features_move_x1.76_over_natural_spread")
    for k in keys:
        vals = {arm: {s: r.get(k, np.nan) for s, r in v.items()} for arm, v in per.items()}
        row = {arm: stats(v) for arm, v in vals.items()}
        if all(a in vals for a in C.ARMS):
            row["separate-shared"] = stats(C.per_seed_diff(vals["separate"], vals["shared"]))
        tab[k] = row
    resp = {}
    some = next((r for v in per.values() for r in v.values()), None)
    if some:
        for k in some["responses"]:
            if not any(t in k for t in ("invested +8", "invested +32", "level +1", "unspent gold +4", "HP 60%",
                                        "enemy HP 60%")):
                continue
            resp[k] = {arm: stats({s: r["responses"].get(k, np.nan) for s, r in v.items()}) for arm, v in per.items()}
    return {"table": tab, "responses": resp}


# ---------------------------------------------------------------------------------------------- training
def training(phase: str = "main") -> dict:
    from .train import history, summary
    per = {a: {} for a in C.ARMS}
    for a in C.ARMS:
        for s in C.SEEDS:
            h = history(a, s, phase)
            if not h:
                continue
            n = len(h)
            last = h[-100:]
            gp = [e for e in h if "gp_total_norm" in e]
            row = {"updates": n, "ev_last100": float(np.mean([e["explained_variance"] for e in last])),
                   "ev_first10": float(np.mean([e["explained_variance"] for e in h[:10]])),
                   "value_loss_last100_res2": float(np.mean([e["value_loss"] for e in last])) / C.REWARD_SCALE ** 2,
                   "value_loss_first10_res2": float(np.mean([e["value_loss"] for e in h[:10]])) / C.REWARD_SCALE ** 2,
                   "entropy_last100": float(np.mean([e["entropy"] for e in last])),
                   "kl_mean": float(np.mean([e["approx_kl"] for e in h])),
                   "clipfrac_mean": float(np.mean([e["clipfrac"] for e in h])),
                   "grad_norm_mean": float(np.mean([e["grad_norm"] for e in h])),
                   "return_std_last100": float(np.mean([e["return_std"] for e in last])) / C.REWARD_SCALE,
                   "value_mean_last100": float(np.mean([e["value_mean"] for e in last])) / C.REWARD_SCALE}
            if gp:
                for k in ("total_norm", "actor_norm", "critic_norm", "trunk_pol", "trunk_val", "trunk_cos", "ctrunk_val",
                          "clip_shrink"):
                    row[f"gp_{k}_mean"] = float(np.mean([e[f"gp_{k}"] for e in gp]))
                    row[f"gp_{k}_last"] = float(np.mean([e[f"gp_{k}"] for e in gp[-8:]]))
                row["gp_critic_share_of_total_sq"] = float(np.mean([(e["gp_critic_norm"] / e["gp_total_norm"]) ** 2
                                                                    for e in gp]))
            # the explained-variance curve in 10 blocks
            row["ev_blocks"] = [float(np.mean([e["explained_variance"] for e in h[i * n // 10:(i + 1) * n // 10]]))
                                for i in range(10)]
            sm = summary(a, s, phase)
            if sm:
                row["transitions_per_s"] = sm.get("transitions_per_s")
                row["elapsed_s"] = sm.get("elapsed_s")
            per[a][s] = row
    tab = {}
    keys = sorted({k for a in per.values() for r in a.values() for k, v in r.items() if isinstance(v, (int, float))})
    for k in keys:
        vals = {a: {s: r.get(k, np.nan) for s, r in per[a].items()} for a in C.ARMS}
        tab[k] = {a: stats(v) for a, v in vals.items()}
        tab[k]["separate-shared"] = stats(C.per_seed_diff(vals["separate"], vals["shared"]))
    return {"per_seed": per, "table": tab}


# ---------------------------------------------------------------------------------------------- head to head
def _h2h(path: str) -> Optional[dict]:
    return C.load_gz(path) if os.path.exists(path) else None


def h2h(phase: str = "main") -> dict:
    d = os.path.join(B.edir(phase), "h2h")
    ivc, st = {}, {a: {} for a in C.ARMS}
    oth, sui = {a: {} for a in C.ARMS}, {a: {} for a in C.ARMS}
    sui_ref = {a: defaultdict(dict) for a in C.ARMS}
    for s in C.SEEDS:
        r = _h2h(os.path.join(d, f"ivc_s{s}.json.gz"))
        if r:
            ivc[s] = r["estimate"]
        for a in C.ARMS:
            r = _h2h(os.path.join(d, f"start_{a}_s{s}.json.gz"))
            if r:
                st[a][s] = r["estimate"]
            vs = [_h2h(os.path.join(d, f"others_{a}_s{s}_o{o}.json.gz")) for o in C.SEEDS if o != s]
            if all(vs):
                oth[a][s] = float(np.mean([v["estimate"] for v in vs]))
            vs = {ref: _h2h(os.path.join(d, f"suite_{a}_s{s}_{ref}.json.gz")) for ref in B.CORE_SUITE}
            if all(vs.values()):
                sui[a][s] = float(np.mean([v["estimate"] for v in vs.values()]))
                for ref, v in vs.items():
                    sui_ref[a][ref][s] = v["estimate"]
    out = {"B1 separate vs shared (W 1000)": stats(ivc),
           "vs start: shared": stats(st["shared"]), "vs start: separate": stats(st["separate"]),
           "vs start: separate - shared": stats(C.per_seed_diff(st["separate"], st["shared"])),
           "vs other starts: shared": stats(oth["shared"]), "vs other starts: separate": stats(oth["separate"]),
           "vs other starts: separate - shared": stats(C.per_seed_diff(oth["separate"], oth["shared"])),
           "vs suite: shared": stats(sui["shared"]), "vs suite: separate": stats(sui["separate"]),
           "vs suite: separate - shared": stats(C.per_seed_diff(sui["separate"], sui["shared"]))}
    # amendment 1: the powered B1 (256 further episodes; pooled with the registered 64 by episode count)
    pw, pooled = {}, {}
    for s in C.SEEDS:
        r = _h2h(os.path.join(d, f"ivc_power_s{s}.json.gz"))
        if r and s in ivc:
            pw[s] = r["estimate"]
            pooled[s] = (64 * ivc[s] + B.POWER_EPISODES * r["estimate"]) / (64 + B.POWER_EPISODES)
    if pw:
        out["B1 powered: 256 new episodes (amendment 1)"] = stats(pw)
        out["B1 powered: pooled 320 episodes (amendment 1)"] = stats(pooled)
        out["per_seed_power"] = {"new": pw, "pooled": pooled}
    out["suite by reference"] = {ref: {a: stats(sui_ref[a][ref]) for a in C.ARMS} | {
        "separate-shared": stats(C.per_seed_diff(sui_ref["separate"][ref], sui_ref["shared"][ref]))}
        for ref in B.CORE_SUITE}
    out["per_seed"] = {"ivc": ivc, "start": st, "others": oth, "suite": sui}
    if phase == "main":
        cur = {}
        for n in C.CURVE_POINTS:
            vals = {}
            for s in C.SEEDS:
                r = _h2h(os.path.join(B.edir("main"), "curve", f"curve_s{s}_{n}.json.gz"))
                if r:
                    vals[s] = r["estimate"]
            cur[n] = stats(vals)
        out["curve"] = cur
    return out


# ---------------------------------------------------------------------------------------------- cells
def load_cells(family: str, phase: str) -> List[dict]:
    """One row per (seed, cell, opponent): meta, per-candidate mean G at the checkpoints, ledger diffs."""
    lst = C.load_json(f"cells_{family}_{phase}")
    if not lst:
        return []
    rows = []
    for j in lst["jobs"]:
        if not os.path.exists(j["path"]):
            continue
        rec = C.load_gz(j["path"])
        row = {"seed": int(j["tag"].split("_")[0][1:]), "opp": j["tag"].split("_vs_")[1], "idx": j["idx"],
               "label": j["label"], "meta": j["meta"], "H": j["horizon"], "K": j["K"]}
        for cand in ("shared", "separate"):
            row[cand] = {int(c): float(np.mean(v)) for c, v in rec["G"][cand].items()}
        row["dG"] = {c: row["separate"][c] - row["shared"][c] for c in row["shared"]}
        keys = rec["keys"]
        row["ledger_diff"] = {int(c): dict(zip(keys, e["diff"])) for c, e in rec["ledger"]["separate"].items()}
        row["ledger_shared"] = {int(c): dict(zip(keys, e["mean"])) for c, e in rec["ledger"]["shared"].items()}
        rows.append(row)
    return rows


def _by_seed_cell(rows: List[dict], h: int, key=lambda r: r["dG"]) -> Dict[int, Dict[int, float]]:
    """{seed: {cell idx: mean over the two opponents}}."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = key(r).get(h, np.nan) if isinstance(key(r), dict) else np.nan
        acc[r["seed"]][r["idx"]].append(v)
    return {s: {i: float(np.mean(v)) for i, v in d.items()} for s, d in acc.items()}


def _meta(rows: List[dict]) -> Dict[int, dict]:
    return {r["idx"]: {**r["meta"], "label": r["label"]} for r in rows}


def conv(phase: str = "main") -> dict:
    rows = load_cells("conv", phase)
    if not rows:
        return {}
    meta = _meta(rows)
    eq_of = {i: next((j for j, mm in meta.items() if mm["group"] == m["group"] and mm["cond"] == "equal"), None)
             for i, m in meta.items()}
    conv_i = [i for i, m in meta.items() if m["cond"] in B.CONV_CONDS]
    def_i = [i for i, m in meta.items() if m["cond"] in B.DEF_CONDS]
    eq_i = [i for i, m in meta.items() if m["cond"] == "equal"]
    big_i = [i for i, m in meta.items() if m["cond"] == "x1.76"]
    out = {}
    for h in (50, 150):
        d = _by_seed_cell(rows, h)
        per = {"B2": {}, "conversion": {}, "defence": {}, "equal cells": {}, "conversion - equal": {},
               "defence - equal": {}, "large x1.76 (diagnostic)": {}}
        for s, dd in d.items():
            per["B2"][s] = float(np.mean([dd[i] for i in conv_i + def_i]))
            per["conversion"][s] = float(np.mean([dd[i] for i in conv_i]))
            per["defence"][s] = float(np.mean([dd[i] for i in def_i]))
            per["equal cells"][s] = float(np.mean([dd[i] for i in eq_i]))
            per["conversion - equal"][s] = float(np.mean([dd[i] - dd[eq_of[i]] for i in conv_i]))
            per["defence - equal"][s] = float(np.mean([dd[i] - dd[eq_of[i]] for i in def_i]))
            per["large x1.76 (diagnostic)"][s] = float(np.mean([dd[i] for i in big_i]))
        out[f"G{h}"] = {k: stats(v) for k, v in per.items()}
        out[f"G{h}"]["per_seed"] = per
    # by cell kind at G150
    kinds = defaultdict(list)
    for i, m in meta.items():
        kinds[("lead " + m["group"].split(" ")[1] if m["group"].startswith("lead") else "lane") + " " + m["cond"]].append(i)
    d = _by_seed_cell(rows, 150)
    out["by kind G150"] = {k: stats({s: float(np.mean([dd[i] for i in ii])) for s, dd in d.items()})
                           for k, ii in sorted(kinds.items())}
    # denial: the opponent's lost farm and XP in conversion cells; one's own in defence cells (ledger, 150 turns)
    den = {}
    for name, ii, side in (("conversion: opponent farm lost (unc_B)", conv_i, "unc_B"),
                           ("conversion: opponent XP lost (xp_lost_B)", conv_i, "xp_lost_B"),
                           ("conversion: own earned (earned_A)", conv_i, "earned_A"),
                           ("conversion: opponent deaths", conv_i, "deaths_B"),
                           ("defence: own farm lost (unc_A)", def_i, "unc_A"),
                           ("defence: own XP lost (xp_lost_A)", def_i, "xp_lost_A"),
                           ("defence: own deaths", def_i, "deaths_A")):
        dd = _by_seed_cell(rows, 150, key=lambda r, k=side: {c: v.get(k, np.nan) for c, v in r["ledger_diff"].items()})
        den[name] = stats({s: float(np.mean([v[i] for i in ii])) for s, v in dd.items()})
        base = _by_seed_cell(rows, 150, key=lambda r, k=side: {c: v.get(k, np.nan) for c, v in r["ledger_shared"].items()})
        den[name]["shared_level"] = float(np.mean([np.mean([v[i] for i in ii]) for v in base.values()]))
    out["denial (ledger, 150 turns; separate - shared)"] = den
    return out


def _ledger_stat(rows, idxs, key, h):
    dd = _by_seed_cell(rows, h, key=lambda r, k=key: {c: v.get(k, np.nan) for c, v in r["ledger_diff"].items()})
    base = _by_seed_cell(rows, h, key=lambda r, k=key: {c: v.get(k, np.nan) for c, v in r["ledger_shared"].items()})
    st = stats({s: float(np.mean([v[i] for i in idxs])) for s, v in dd.items()})
    st["shared_level"] = float(np.mean([np.mean([v[i] for i in idxs]) for v in base.values()])) if base else np.nan
    return st


def freeze(phase: str = "main") -> dict:
    rows = load_cells("freeze", phase)
    if not rows:
        return {}
    meta = _meta(rows)
    out = {}
    d = _by_seed_cell(rows, 50)
    for role in ("freezer", "breaker"):
        r_out = {}
        for cond in B.FREEZE_NAMES:
            ii = [i for i, m in meta.items() if m["role"] == role and m["cond"] == cond]
            r_out[f"{cond} G50"] = stats({s: float(np.mean([dd[i] for i in ii])) for s, dd in d.items()})
        fav = [i for i, m in meta.items() if m["role"] == role and m["cond"] in ("edge", "plaus", "hp")]
        r_out["favourable (edge, plaus, hp) G50"] = stats({s: float(np.mean([dd[i] for i in fav])) for s, dd in d.items()})
        if role == "freezer":
            for k in ("held_A", "unc_B", "xp_lost_B", "deaths_B", "deaths_A"):
                r_out[f"favourable: {k}"] = _ledger_stat(rows, fav, k, 50)
        else:
            for k in ("unc_A", "xp_lost_A", "deaths_A", "frozen_A"):
                r_out[f"favourable (freezer advantaged): {k}"] = _ledger_stat(rows, fav, k, 50)
        out[role] = r_out
    return out


def recall(phase: str = "main") -> dict:
    rows = load_cells("recall", phase)
    if not rows:
        return {}
    meta = _meta(rows)
    out = {}
    for role in ("weak", "strong"):
        r_out = {}
        for h in (50, 100):
            d = _by_seed_cell(rows, h)
            for cond in B.RW_CONDS:
                ii = [i for i, m in meta.items() if m["role"] == role and m["cond"] == cond]
                r_out[f"{cond} G{h}"] = stats({s: float(np.mean([dd[i] for i in ii])) for s, dd in d.items()})
            ii = [i for i, m in meta.items() if m["role"] == role and m["cond"] != "equal"]
            r_out[f"stronger opponent (edge, x1.76) G{h}" if role == "weak" else f"advantaged (edge, x1.76) G{h}"] = \
                stats({s: float(np.mean([dd[i] for i in ii])) for s, dd in d.items()})
        ii = [i for i, m in meta.items() if m["role"] == role and m["cond"] != "equal"]
        for k in ("deaths_A", "recalls_A", "n_safe_A", "n_yield_A", "away_A", "deaths_B", "earned_A"):
            r_out[f"{k} (100 turns)"] = _ledger_stat(rows, ii, k, 100)
        out[role] = r_out
    ft = os.path.join(B.edir(phase), "first_turn.json.gz")
    if os.path.exists(ft):
        f = C.load_gz(ft)
        acc = defaultdict(lambda: defaultdict(list))
        for s, cells in f.items():
            for lab, row in cells.items():
                if not lab.startswith("weak") or " equal " in lab:
                    continue
                for actor, p in row.items():
                    for k, v in p.items():
                        acc[actor][k].append(v)
        out["first-turn probabilities, weak side vs a stronger opponent"] = {
            a: {k: float(np.mean(v)) for k, v in d.items()} for a, d in acc.items()}
        acc = defaultdict(lambda: defaultdict(list))
        for s, cells in f.items():
            for lab, row in cells.items():
                if lab.startswith("freezer") and any(c in lab for c in (" edge", " plaus", " hp")):
                    for actor, p in row.items():
                        for k, v in p.items():
                            acc[actor][k].append(v)
        out["first-turn probabilities, freezer in favourable freeze cells"] = {
            a: {k: float(np.mean(v)) for k, v in d.items()} for a, d in acc.items()}
    return out


def natural(phase: str = "main") -> dict:
    rows = load_cells("natural", phase)
    if not rows:
        return {}
    meta = _meta(rows)
    out = {}
    for role, hs in (("leader", (50, 150)), ("trailer", (50, 150)), ("freezer", (50,)), ("breaker", (50,))):
        ii = [i for i, m in meta.items() if m["role"] == role]
        for h in hs:
            d = _by_seed_cell(rows, h)
            out[f"{role} G{h}"] = stats({s: float(np.mean([dd[i] for i in ii if i in dd])) for s, dd in d.items()})
    return out


def economy(phase: str = "main") -> dict:
    """EXPLORATORY (amendment 2): natural self-play economy profiles; per network, then paired differences."""
    d = os.path.join(B.edir(phase), "economy")
    per = {}
    for a in ("start",) + C.ARMS:
        per[a] = {}
        for s in C.SEEDS:
            base = os.path.join(B.edir("main"), "economy") if a == "start" else d    # starts: profiled once
            p = os.path.join(base, f"{a}_s{s}.json.gz")
            if os.path.exists(p):
                r = C.load_gz(p)
                per[a][s] = {**r["summary"], **r["economy"]}
    if not per["shared"]:
        return {}
    keys = sorted(next(iter(per["shared"].values())))
    tab = {}
    for k in keys:
        vals = {a: {s: r.get(k, np.nan) for s, r in per[a].items()} for a in per}
        row = {a: stats(v) for a, v in vals.items() if v}
        row["separate-shared"] = stats(C.per_seed_diff(vals["separate"], vals["shared"]))
        if vals["start"]:
            row["shared-start"] = stats(C.per_seed_diff(vals["shared"], vals["start"]))
            row["separate-start"] = stats(C.per_seed_diff(vals["separate"], vals["start"]))
        tab[k] = row
    return {"table": tab}


# ---------------------------------------------------------------------------------------------- rules
def scorecard(phase: str = "main") -> dict:
    vc = value_calib(phase)
    t = vc["table"]
    v1 = t["edge+plaus ratio"]["separate-shared"]
    sep_ratio = t["edge+plaus ratio"]["separate"]["mean"]
    v1_cls = classify(v1, *MARGINS["V1"])
    if v1_cls == "benefit" and not sep_ratio >= V1_FLOOR:
        v1_cls = "small"
    v2 = t["natural MAE"]["separate-shared"]
    v2_reg = bool(v2.get("mean", np.nan) >= V2_REGRESSION and v2["mean"] - v2["ci"] > 0) if v2.get("n") else None
    hh = h2h(phase)
    b1 = hh["B1 separate vs shared (W 1000)"]
    cv = conv(phase)
    b2 = cv.get("G150", {}).get("B2", {})
    guards = {"vs start: separate - shared": hh["vs start: separate - shared"],
              "vs suite: separate - shared": hh["vs suite: separate - shared"]}
    guard_reg = {k: bool(v.get("n") and v["mean"] <= -GUARD_REGRESSION and v["mean"] + v["ci"] < 0)
                 for k, v in guards.items()}
    sh_gain = t["edge+plaus ratio"]["shared-start"]
    out = {"V1": {"stat": v1, "class": v1_cls, "separate_ratio": sep_ratio,
                  "shared_ratio": t["edge+plaus ratio"]["shared"]["mean"],
                  "start_ratio": t["edge+plaus ratio"]["start"]["mean"]},
           "V2": {"stat": v2, "regression": v2_reg},
           "V3": {"stat": t["large+gap ratio"]["separate-shared"]},
           "B1": {"stat": b1, "class": classify(b1, *MARGINS["B1"])},
           "B1 powered (amendment 1, not used for the outcome)": {
               "stat": hh.get("B1 powered: pooled 320 episodes (amendment 1)", {}),
               "class": classify(hh.get("B1 powered: pooled 320 episodes (amendment 1)", {}), *MARGINS["B1"])},
           "B2": {"stat": b2, "class": classify(b2, *MARGINS["B2"]) if b2 else "n/a"},
           "guards": {k: {"stat": v, "regression": guard_reg[k]} for k, v in guards.items()},
           "shared_gain_over_start": {"stat": sh_gain,
                                      "material": bool(sh_gain.get("n") and sh_gain["mean"] >= 0.25
                                                       and sh_gain["mean"] - sh_gain["ci"] > 0)}}
    harm = [k for k, x in (("B1", out["B1"]["class"] == "regression"), ("B2", out["B2"]["class"] == "regression"),
                           ("V2", bool(v2_reg))) if x] + [k for k, x in guard_reg.items() if x]
    v1b = v1_cls == "benefit"
    beh = (out["B1"]["class"], out["B2"]["class"])
    if harm:
        outcome = "harm: the separate arm regresses on " + ", ".join(harm)
    elif v1b and "benefit" in beh:
        outcome = "2: separating the critic helps"
    elif v1b and all(c in ("no material difference", "small") for c in beh):
        outcome = "3: better values without a meaningful behavioural benefit"
    elif not v1b and out["shared_gain_over_start"]["material"] and "benefit" not in beh:
        outcome = "1: more training alone is sufficient"
    elif not v1b and not sep_ratio >= V1_FLOOR:
        outcome = ("4: not resolved in this budget: V1 is not a benefit and the separate critic recovers the premium "
                   "only partially (ratio < 0.5)")
    else:
        outcome = "4: inconclusive"
    out["harm"] = harm
    out["outcome"] = outcome
    return out


def extension_rule() -> dict:
    """The pre-registered extension rule on the 4.096M results (and the 2.048M snapshots)."""
    from .train import check
    sc = scorecard("main")

    def sep_ratio(vc):
        return {s: vc["per_seed"]["separate"].get(s, {}).get("classes", {}).get("edge+plaus", {}).get("ratio", np.nan)
                for s in C.SEEDS}

    mid = sep_ratio(value_calib("mid"))
    fin = sep_ratio(value_calib("main", K=V.K_CAL_MID))          # the same 16 rollout seeds as the snapshots
    rise = stats(C.per_seed_diff(fin, mid))
    v1 = sc["V1"]["stat"]
    a = bool((v1.get("n") and v1["mean"] - v1["ci"] > 0) or (rise.get("n") and rise["mean"] - rise["ci"] > 0))
    resolved = ("benefit", "regression", "no material difference")
    b = bool(sc["B1"]["class"] not in resolved and sc["B2"]["class"] not in resolved and
             (sc["B1"]["stat"].get("mean", -1) > 0 or sc["B2"]["stat"].get("mean", -1) > 0))
    chk = check("main", save=False)
    c = all(v for k, v in chk["checks"].items() if "stop condition" in k or "explained variance" in k)
    d = {"meta": C.meta(stage="ext_rule"), "a_separate_critic_learning": a, "a_detail": {"V1": v1,
                                                                                        "rise_2048_to_4096": rise},
         "b_behaviour_unresolved_positive": b, "b_detail": {"B1": sc["B1"], "B2": sc["B2"]},
         "c_health": c, "fire": bool(a and b and c)}
    d["reasons"] = {"a": a, "b": b, "c": c}
    C.save_json("extension_decision", d)
    return d


def run() -> dict:
    res = {"meta": C.meta(stage="analysis")}
    phases = ["main"]
    if os.path.exists(os.path.join(C.OUT, "extension_decision.json")) and \
            (C.load_json("extension_decision") or {}).get("fire"):
        phases.append("ext")
    for ph in phases:
        r = {"training": training(ph), "value_calib": value_calib(ph), "value_natural": value_natural(ph),
             "value_probes": value_probes(ph), "h2h": h2h(ph), "conv": conv(ph), "freeze": freeze(ph),
             "recall": recall(ph), "natural": natural(ph), "economy": economy(ph)}
        r["scorecard"] = scorecard(ph)
        res[ph] = r
    res["mid"] = {"value_calib": value_calib("mid")}
    C.save_json("analysis", res)
    return res


__all__ = ["classify", "value_calib", "value_natural", "value_probes", "training", "h2h", "conv", "freeze", "recall",
           "natural", "scorecard", "extension_rule", "run", "MARGINS"]
