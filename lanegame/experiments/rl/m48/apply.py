"""Stage 2 readouts of M4.8 (protocol §5 S1-S4, §6, §7): the natural checkpoints of the matched baseline and C1
policies around the pre-level-14 boundary.

Units. Population claims use the six training seeds of each curve as units (t intervals, 5 df); paired C1 - baseline
differences pair the seeds. Rollouts estimate within-state uncertainty only. State distributions (what the policies'
own games look like at the checkpoint) are kept apart from outcomes from matched states (the same states continued by
both packages).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import analysis as AN
from . import common as C
from . import families as F
from . import rollouts as R
from . import states as ST
from .validate import material, reversal

H_ALL = C.WINDOWS


# ---------------------------------------------------------------------------------------------- rows
def rows() -> List[dict]:
    states = ST.natural_states()
    js = F.natural_jobs(states)
    out = []
    for st, j in zip(states, js):
        x = R.load(j["path"])
        d = st["desc"]
        row = {"env": st["env"], "s": st["s"], "g": st["g"], "t0": st["t0"], "desc": d, "rec": st["rec"]}
        row.update(AN.combo_row(d, x["own"]))
        for h in H_ALL:
            row[f"rec_d{h}"] = st["rec"]["delta"][str(h)]
        for name, pre in (("pkg_other", "oth"), ("alt", "alt")):
            if name in x:
                row.update(AN.combo_row(d, x[name], pre + ":"))
                row.update(AN.paired_row(x[name], x["own"], pre + "-own:"))
        if "A_fresh" in x:
            row.update(AN.paired_row(x["A_fresh"], x["own"], "A_fresh:"))
        row["cat"], row["catX"] = AN.category(d)
        row["cases"] = [tuple(c) for c in ST.cases_of(d)]
        row["feat"] = AN.features(d)
        out.append(row)
    return out


def at(rs: List[dict], t0: int = C.T0_PRIMARY, env: str = None, s: int = None) -> List[dict]:
    return [r for r in rs if r["t0"] == t0 and (env is None or r["env"] == env) and (s is None or r["s"] == s)]


def true_sd(rs: List[dict], h: int, pre: str = "") -> float:
    d = np.asarray([r[f"{pre}d{h}"] for r in rs])
    se2 = np.mean([r[f"{pre}se{h}"] ** 2 for r in rs])
    return float(np.sqrt(max(0.0, d.var(ddof=1) - se2)))


def per_seed(rs: List[dict], fn, t0: int = C.T0_PRIMARY) -> dict:
    """fn(states of one package) -> float; seed statistics per curve and the paired C1 - baseline difference."""
    vals = {env: {s: fn(at(rs, t0, env, s)) for s in C.SEEDS} for env in C.ENVS}
    return {"baseline": C.seed_stats(vals["baseline"]), "C1": C.seed_stats(vals["C1"]),
            "C1-baseline": C.seed_stats({s: vals["C1"][s] - vals["baseline"][s] for s in C.SEEDS}),
            "pooled": C.seed_stats({f"{e}{s}": vals[e][s] for e in C.ENVS for s in C.SEEDS})}


# ---------------------------------------------------------------------------------------------- S1
def s1(rs: List[dict]) -> dict:
    out = {}
    for h in H_ALL:
        o = {"material": per_seed(rs, lambda x: float(np.mean([material(r[f"d{h}"], r[f"se{h}"]) for r in x]))),
             "material_point": per_seed(rs, lambda x: float(np.mean([abs(r[f"d{h}"]) >= C.M_STATE for r in x]))),
             "material_0.25": per_seed(rs, lambda x: float(np.mean([material(r[f"d{h}"], r[f"se{h}"], 0.25)
                                                                   for r in x]))),
             "material_1.0": per_seed(rs, lambda x: float(np.mean([material(r[f"d{h}"], r[f"se{h}"], 1.0)
                                                                  for r in x]))),
             "reversal": per_seed(rs, lambda x: float(np.mean([reversal(r["desc"]["D0"], r[f"E{h}"], r[f"se{h}"])
                                                              for r in x]))),
             "mean_abs_d": per_seed(rs, lambda x: float(np.mean([abs(r[f"d{h}"]) for r in x]))),
             "true_sd": per_seed(rs, lambda x: true_sd(x, h)),
             "halfwidth_median": float(np.median([1.96 * r[f"se{h}"] for r in at(rs)]))}
        pool = at(rs)
        d = np.asarray([r[f"d{h}"] for r in pool])
        o["pooled"] = {"n": len(pool), "material": float(np.mean([material(r[f"d{h}"], r[f"se{h}"]) for r in pool])),
                       "reversal": float(np.mean([reversal(r["desc"]["D0"], r[f"E{h}"], r[f"se{h}"]) for r in pool])),
                       "true_sd": true_sd(pool, h), "sd_d": float(d.std(ddof=1)),
                       "quantiles_d": {str(q): float(np.percentile(d, q)) for q in (5, 10, 25, 50, 75, 90, 95)},
                       "abs_quantiles_d": {str(q): float(np.percentile(np.abs(d), q)) for q in (50, 75, 90, 95)},
                       "near_even_D0": float(np.mean([abs(r["desc"]["D0"]) < C.M_STATE for r in pool])),
                       "near_even_decided": float(np.mean([abs(r["desc"]["D0"]) < C.M_STATE
                                                           and abs(r[f"E{h}"]) > 1.96 * r[f"se{h}"] for r in pool]))}
        out[str(h)] = o
    out["abs_D0"] = per_seed(rs, lambda x: float(np.mean([abs(r["desc"]["D0"]) for r in x])))
    out["levels_at_94"] = {"min_level_mean": float(np.mean([min(r["desc"]["level_A"], r["desc"]["level_B"])
                                                           for r in at(rs)])),
                           "share_both_ge14": float(np.mean([min(r["desc"]["level_A"], r["desc"]["level_B"]) >= 14
                                                             for r in at(rs)])),
                           "share_any_ge14": float(np.mean([max(r["desc"]["level_A"], r["desc"]["level_B"]) >= 14
                                                            for r in at(rs)])),
                           "level_quantiles": {str(q): float(np.percentile([r["desc"][f"level_{x}"] for r in at(rs)
                                                                            for x in ("A", "B")], q))
                                               for q in (10, 50, 90)}}
    return out


# ---------------------------------------------------------------------------------------------- S2
def s2(rs: List[dict], t0: int = C.T0_PRIMARY) -> dict:
    pool = at(rs, t0)
    out = {"categories": {}}
    for cat in AN.CATEGORIES:
        sub = [r for r in pool if r["cat"] == cat]
        o = {"n": len(sub), "share": len(sub) / len(pool),
             "share_by_env": {e: float(np.mean([r["cat"] == cat for r in pool if r["env"] == e])) for e in C.ENVS}}
        if sub:
            sg = [AN.sign_of(r["catX"]) for r in sub]
            o["lead_x"] = C.mean_se([g * r["desc"]["D0"] for g, r in zip(sg, sub)])
            for h in H_ALL:
                o[f"d{h}"] = C.mean_se([g * r[f"d{h}"] for g, r in zip(sg, sub)])
                o[f"material{h}"] = float(np.mean([material(r[f"d{h}"], r[f"se{h}"]) for r in sub]))
                o[f"drift40-{h}"] = C.mean_se([g * (r["d40"] - r[f"d{h}"]) for g, r in zip(sg, sub)])
                o[f"res<={h}"] = float(np.mean([r[f"res<={h}"] for r in sub]))
            for h in (10, 20):
                comp = {}
                for key in ("farm", "avail", "lost_away", "bounty", "plate", "returned", "away"):
                    comp[key] = float(np.mean([g * (r[f"c{h}_{key}_A"] - r[f"c{h}_{key}_B"])
                                               for g, r in zip(sg, sub)]))
                o[f"components{h}"] = comp
        out["categories"][cat] = o
    # explained variance: pending vs strength vs the lead itself
    feats = np.asarray([[r["feat"][k] for k in AN.PENDING_FEATURES + AN.STRENGTH_FEATURES + AN.LEAD_FEATURES]
                        for r in pool])
    npf, nsf = len(AN.PENDING_FEATURES), len(AN.STRENGTH_FEATURES)
    folds = np.asarray([r["g"] % 5 for r in pool])
    names = AN.PENDING_FEATURES + AN.STRENGTH_FEATURES + AN.LEAD_FEATURES

    def cols(keys):
        return feats[:, [names.index(k) for k in keys]]

    groups = {"all": feats, "pending": feats[:, :npf], "strength": feats[:, npf:npf + nsf],
              "lead": feats[:, npf + nsf:], "pending+lead": np.column_stack([feats[:, :npf], feats[:, npf + nsf:]]),
              # finer descriptive split (not pre-registered): HP acts mostly through who has to reset next
              "reset": cols(("recall_away", "dead", "returns_in", "fresh", "due", "unspent")),
              "wave": cols(("stock", "near", "siege_under_own", "freeze")), "hp": cols(("hp_present",)),
              "power": cols(("invested", "level")),
              "reset+wave+hp": cols(("recall_away", "dead", "returns_in", "fresh", "due", "unspent", "stock", "near",
                                     "siege_under_own", "freeze", "hp_present"))}
    ev = {}
    for h in H_ALL:
        y = np.asarray([r[f"d{h}"] for r in pool])
        noise = float(np.mean([r[f"se{h}"] ** 2 for r in pool]) / y.var())
        ev[str(h)] = {k: AN.r2(y, X) for k, X in groups.items()}
        ev[str(h)].update({f"cv_{k}": AN.cv_r2(y, X, folds) for k, X in groups.items()})
        ev[str(h)]["noise_share"] = noise
        Xc = np.column_stack([np.ones(len(y)), feats])
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        ev[str(h)]["coef"] = dict(zip(("intercept",) + AN.PENDING_FEATURES + AN.STRENGTH_FEATURES + AN.LEAD_FEATURES,
                                      [float(b) for b in beta]))
    out["explained"] = ev
    out["cases_at_t0"] = {}
    for case in ST.CASES + ("due",):
        sub = [r for r in pool if any(c[0] == case for c in r["cases"])]
        o = {"n": len(sub)}
        if sub:
            sg = [AN.sign_of(next(c[1] for c in r["cases"] if c[0] == case)) for r in sub]
            for h in (5, 10, 20, 40):
                o[f"d{h}"] = C.mean_se([g * r[f"d{h}"] for g, r in zip(sg, sub)])
                o[f"material{h}"] = float(np.mean([material(r[f"d{h}"], r[f"se{h}"]) for r in sub]))
        out["cases_at_t0"][case] = o
    return out


# ---------------------------------------------------------------------------------------------- S3
def s3(rs: List[dict]) -> dict:
    pool = at(rs)
    out = {"windows": {}, "policies": {}, "anchors": {}, "new_after": {}}
    for a, b in ((5, 10), (10, 20), (20, 40)):
        out["windows"][f"{a}->{b}"] = {
            "mean_abs_change": float(np.mean([abs(r[f"d{b}"] - r[f"d{a}"]) for r in pool])),
            "class_changes": float(np.mean([material(r[f"d{a}"], r[f"se{a}"]) != material(r[f"d{b}"], r[f"se{b}"])
                                            for r in pool])),
            "class_changes_among_material": float(np.mean(
                [material(r[f"d{a}"], r[f"se{a}"]) != material(r[f"d{b}"], r[f"se{b}"]) for r in pool
                 if material(r[f"d{a}"], r[f"se{a}"]) or material(r[f"d{b}"], r[f"se{b}"])] or [np.nan])),
            "leader_changes": float(np.mean([np.sign(r[f"E{a}"]) != np.sign(r[f"E{b}"]) for r in pool])),
            "corr": float(np.corrcoef([r[f"d{a}"] for r in pool], [r[f"d{b}"] for r in pool])[0, 1])}
    for h in H_ALL:
        o = {}
        for k in ("returned", "died", "recalled"):
            o[k] = float(np.mean([(r[f"c40_{k}_A"] + r[f"c40_{k}_B"]) - (r[f"c{h}_{k}_A"] + r[f"c{h}_{k}_B"])
                                  for r in pool]))
        out["new_after"][str(h)] = o
    for pre, label in (("alt", "alternative policy (other seed, same curve)"), ("oth", "other curve's package")):
        sub = [r for r in pool if f"{pre}:d10" in r]
        o = {"label": label, "n": len(sub)}
        for h in H_ALL:
            a = np.asarray([r[f"d{h}"] for r in sub])
            b = np.asarray([r[f"{pre}:d{h}"] for r in sub])
            o[str(h)] = {"mean_diff": C.mean_se(b - a), "mean_abs_diff": float(np.mean(np.abs(b - a))),
                         "corr": float(np.corrcoef(a, b)[0, 1]),
                         "paired_se_median": float(np.median([r[f"{pre}-own:pse{h}"] for r in sub])),
                         "class_changes": float(np.mean([material(r[f"d{h}"], r[f"se{h}"])
                                                         != material(r[f"{pre}:d{h}"], r[f"{pre}:se{h}"])
                                                         for r in sub])),
                         "class_changes_among_material": float(np.mean(
                             [material(r[f"d{h}"], r[f"se{h}"]) != material(r[f"{pre}:d{h}"], r[f"{pre}:se{h}"])
                              for r in sub if material(r[f"d{h}"], r[f"se{h}"])
                              or material(r[f"{pre}:d{h}"], r[f"{pre}:se{h}"])] or [np.nan])),
                         "leader_changes": float(np.mean([np.sign(r[f"E{h}"]) != np.sign(r[f"{pre}:E{h}"])
                                                          for r in sub])),
                         "material_signif_diff": float(np.mean([abs(r[f"{pre}-own:pd{h}"]) >= C.M_STATE
                                                                and abs(r[f"{pre}-own:pd{h}"])
                                                                > 1.96 * r[f"{pre}-own:pse{h}"] for r in sub]))}
        out["policies"][pre] = o
    for t0 in (84, C.T0_PRIMARY, 104, C.T0_REPL):
        sub = at(rs, t0)
        o = {"n": len(sub), "K": float(np.median([r["K"] for r in sub])),
             "level_min_mean": float(np.mean([min(r["desc"]["level_A"], r["desc"]["level_B"]) for r in sub]))}
        for h in (5, 10, 20, 40):
            o[str(h)] = {"material": per_seed(rs, lambda x: float(np.mean([material(r[f"d{h}"], r[f"se{h}"])
                                                                          for r in x])), t0),
                         "true_sd": per_seed(rs, lambda x: true_sd(x, h), t0),
                         "halfwidth_median": float(np.median([1.96 * r[f"se{h}"] for r in sub]))}
        out["anchors"][str(t0)] = o
    return out


# ---------------------------------------------------------------------------------------------- the window rule (§6)
def window_rule(rs: List[dict], s2o: dict) -> dict:
    pool = at(rs)
    out = {"candidates": {}}
    ordinary = s2o["categories"]["ordinary"]
    pend = [r for r in pool if any(AN.pending_items(r["desc"]).values())]
    out["states_with_pending"] = len(pend)
    for h in C.CANDIDATES:
        cov = float(np.mean([r[f"res<={h}"] for r in pend]))
        cov_kinds = {k: float(np.mean([r[f"{k}<={h}"] for r in pend if f"{k}<={h}" in r]))
                     for k in ("resA", "resD", "resS")}
        drift = {}
        fails = []
        for cat, o in s2o["categories"].items():
            if cat == "ordinary" or o["n"] < 30:
                continue
            dd = o[f"drift40-{h}"]["mean"] - ordinary[f"drift40-{h}"]["mean"]
            drift[cat] = {"n": o["n"], "excess_drift": dd, "drift": o[f"drift40-{h}"],
                          "ordinary": ordinary[f"drift40-{h}"]}
            if abs(dd) > C.M_POP:
                fails.append(cat)
        hw = float(np.median([1.96 * r[f"se{h}"] for r in pool]))
        out["candidates"][str(h)] = {"coverage": cov, "coverage_by_kind": cov_kinds, "a": cov >= 0.8,
                                     "drift": drift, "b": not fails, "b_fails": fails,
                                     "halfwidth_median": hw, "c": hw <= C.M_STATE}
    chosen = None
    for h in C.CANDIDATES:
        c = out["candidates"][str(h)]
        if c["a"] and c["b"] and c["c"]:
            chosen = h
            break
    out["default"] = chosen
    out["recommendation"] = (f"E_{chosen}" if chosen is not None
                             else "multi-window readout (E_10 and E_20 with resolution flags)")
    return out


# ---------------------------------------------------------------------------------------------- S4: baseline vs C1
def slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx = x.var(ddof=1)
    return float(np.cov(x, y, ddof=1)[0, 1] / vx) if vx > 0 else float("nan")


def s4(rs: List[dict]) -> dict:
    out = {}
    # (a) distributions at t0 = 94 (each curve's own games)
    dist = {"abs_D0": per_seed(rs, lambda x: float(np.mean([abs(r["desc"]["D0"]) for r in x])))}
    for h in (5, 10, 20, 40):
        dist[f"abs_E{h}"] = per_seed(rs, lambda x: float(np.mean([abs(r[f"E{h}"]) for r in x])))
    for cat in AN.CATEGORIES:
        dist[f"share_{cat}"] = per_seed(rs, lambda x: float(np.mean([r["cat"] == cat for r in x])))
    dist["abs_xp_diff"] = per_seed(rs, lambda x: float(np.mean([abs(r["desc"]["xp_diff"]) for r in x])))
    dist["abs_hp_diff"] = per_seed(rs, lambda x: float(np.nanmean([abs(r["desc"]["hp_diff"]) for r in x])))
    dist["level_min"] = per_seed(rs, lambda x: float(np.mean([min(r["desc"]["level_A"], r["desc"]["level_B"])
                                                              for r in x])))
    dist["away_share"] = per_seed(rs, lambda x: float(np.mean([(r["desc"]["status_A"] > 0) + (r["desc"]["status_B"] > 0)
                                                               for r in x]) / 2))
    out["distributions"] = dist
    # (b) lead conversion under each package's own continuation: slope of Delta_h on D0
    conv = {}
    for h in H_ALL:
        conv[str(h)] = per_seed(rs, lambda x: slope([r["desc"]["D0"] for r in x], [r[f"d{h}"] for r in x]))
    out["conversion_own"] = conv
    # (c) matched states: every t0 = 94 state continued by both packages (C1 package minus baseline package)
    pool = at(rs)
    matched = {}
    for h in H_ALL:
        def diff(r):
            own_minus = r[f"oth-own:pd{h}"]                       # other package - own package (CRN)
            return own_minus if r["env"] == "baseline" else -own_minus

        by_seed = {}
        for s in C.SEEDS:
            sub = [r for r in pool if r["s"] == s]
            lead_gain = [np.sign(r["desc"]["D0"]) * diff(r) for r in sub if r["desc"]["D0"] != 0]
            by_seed[s] = {"leader_gain": float(np.mean(lead_gain)), "mean_diff": float(np.mean([diff(r) for r in sub])),
                          "slope_diff": slope([r["desc"]["D0"] for r in sub], [diff(r) for r in sub])}
        matched[str(h)] = {k: C.seed_stats({s: by_seed[s][k] for s in C.SEEDS})
                           for k in ("leader_gain", "mean_diff", "slope_diff")}
        for origin in C.ENVS:
            sub = [r for r in pool if r["env"] == origin]
            matched[str(h)][f"leader_gain_states_from_{origin}"] = C.mean_se(
                [np.sign(r["desc"]["D0"]) * diff(r) for r in sub if r["desc"]["D0"] != 0])
    out["matched_states"] = matched
    # (d) earlier conclusions (M4.7 section 2): |lead| at 94 and the 50 -> 94 conversion slope, raw and with E_h
    by = {(r["env"], r["s"], r["g"], r["t0"]): r for r in rs}
    m47 = {"abs_lead_94": {"D0": per_seed(rs, lambda x: float(np.mean([abs(r["desc"]["D0"]) for r in x])))}}
    for h in (5, 10, 20):
        m47["abs_lead_94"][f"E{h}"] = per_seed(rs, lambda x: float(np.mean([abs(r[f"E{h}"]) for r in x])))

    def conv_50_94(key):
        def fn(x94):
            pairs = [(by[(r["env"], r["s"], r["g"], C.T0_REPL)], r) for r in x94]
            x = [key(a) for a, _ in pairs]
            y = [key(b) - key(a) for a, b in pairs]
            return slope(x, y)
        return fn

    m47["slope_50_94"] = {"D0": per_seed(rs, conv_50_94(lambda r: r["desc"]["D0"]))}
    for h in (5, 10, 20):
        m47["slope_50_94"][f"E{h}"] = per_seed(rs, conv_50_94(lambda r, h=h: r[f"E{h}"]))
    out["m47_recheck"] = m47
    # controlled: the value of an already-completed reset at the lane exit (A_fresh - orig), per curve
    fr = {}
    for h in (5, 10, 20, 40):
        def fresh(x, h=h):
            v = [r[f"A_fresh:pd{h}"] for r in x if f"A_fresh:pd{h}" in r]
            return float(np.mean(v)) if v else float("nan")
        fr[str(h)] = per_seed(rs, fresh)
    fr["n"] = {e: sum(1 for r in at(rs, env=e) if "A_fresh:pd10" in r) for e in C.ENVS}
    out["fresh_variant"] = fr
    return out


# ---------------------------------------------------------------------------------------------- classification (§7)
def classify(val: dict, rule: dict, s1o: dict, rs: List[dict]) -> dict:
    nat_acct = float(max(r["acct_max"] for r in rs))
    exact = val["V1_accounting"]["max_abs_units"] == 0 and nat_acct == 0 and val["V2_mirror"]["exact"]
    hs = [rule["default"]] if rule["default"] else [10, 20]
    cal = {}
    ok_cal = True
    for h in hs:
        c = val["V6_calibration"][str(h)]
        lo, hi = c["slope"]["slope"] - c["slope"]["ci"], c["slope"]["slope"] + c["slope"]["ci"]
        overlaps = hi >= 0.8 and lo <= 1.2
        resid_ok = abs(c["mean_resid"]["mean"]) <= C.M_POP
        cal[str(h)] = {"slope_interval": [lo, hi], "overlaps_0.8_1.2": overlaps, "mean_resid": c["mean_resid"],
                       "resid_ok": resid_ok}
        ok_cal = ok_cal and overlaps and resid_ok
    added = {}
    adds = False
    for h in hs:
        p = s1o[str(h)]["pooled"]
        added[str(h)] = {"material_share": p["material"], "true_sd": p["true_sd"]}
        adds = adds or p["material"] >= 0.10 or p["true_sd"] >= C.M_POP
    halfwidths = {str(h): s1o[str(h)]["halfwidth_median"] for h in C.CANDIDATES}
    noisy = all(v > C.M_STATE for v in halfwidths.values())
    unstable = (rule.get("_policy_class_changes", 0) > 0.5)
    if not exact or not ok_cal or noisy or unstable:
        label = "inadequate"
    elif adds:
        label = "useful"
    else:
        label = "valid, limited added value"
    return {"exact_checks": exact, "natural_acct_max": nat_acct, "calibration": cal, "calibration_ok": ok_cal,
            "added_information": added, "adds": adds, "halfwidths": halfwidths, "label": label,
            "note": "the Stage 1 mechanism check (adjustments accounted for by recorded events matching the cases) "
                    "is judged in the report from validation.json; it does not change this label automatically"}


# ---------------------------------------------------------------------------------------------- run
def run(rs: List[dict] = None, val: dict = None) -> dict:
    rs = rs if rs is not None else rows()
    if val is None:
        val = C.load("validation")
    out = {"meta": C.meta(stage="apply"), "n_states": len(rs),
           "n_by_t0": {str(t): len(at(rs, t)) for t in C.T0S}}
    out["S1"] = s1(rs)
    out["S2"] = s2(rs)
    out["S2_t84"] = s2(rs, 84)
    out["S2_t104"] = s2(rs, 104)
    out["S3"] = s3(rs)
    out["window_rule"] = window_rule(rs, out["S2"])
    pol = out["S3"]["policies"]
    hs = [out["window_rule"]["default"]] if out["window_rule"]["default"] else [10, 20]
    changes = [pol[p][str(h)]["class_changes_among_material"] for p in pol for h in hs]
    changes.append(out["S3"]["windows"]["10->20"]["class_changes_among_material"])
    out["window_rule"]["_policy_class_changes"] = max(v for v in changes if np.isfinite(v))
    out["window_rule"]["_class_changes_detail"] = {"policies": {p: {str(h): pol[p][str(h)][
        "class_changes_among_material"] for h in hs} for p in pol},
        "windows_10_20": out["S3"]["windows"]["10->20"]["class_changes_among_material"]}
    out["S4"] = s4(rs)
    from .validate import reversals
    out["reversals"] = {str(h): reversals(at(rs), h, lambda r: r["cat"]) for h in (10, 20)}
    out["classification"] = classify(val, out["window_rule"], out["S1"], rs)
    C.save_json("application", out)
    print(f"  application: {len(rs)} states; default window {out['window_rule']['recommendation']}; "
          f"method {out['classification']['label']}", flush=True)
    return out


__all__ = ["rows", "s1", "s2", "s3", "s4", "window_rule", "classify", "run"]
