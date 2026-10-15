"""Analysis of the horizon-sensitivity study (horizon.py): results/m45c/horizon/summary.json, report.md, figures/.

Notation (per family, opponent protocol and candidate c; the baseline is "policy"):
    d_{u,k}(h)   paired difference of rollout k from state u at horizon h: X_c - X_policy, X = G_h (raw truncated
                 return) or Q_h (G_h + 0.99^h V(s_h), critic-bootstrapped)
    m_u(h)       state mean over the K paired rollouts; w_u(h) their variance
    M(h)         mean of m_u over states; SE over states (states are the units; rollouts nested)
    sigma_w^2    pooled within-state variance of one paired rollout difference (mean of w_u)
    sigma_b^2    between-state variance of the true state effects: var(m_u) - sigma_w^2 / K (>= 0)
    reference    Q_300 on the long subset (12 natural states per family, protocol "policy"; every constructed cell)

A (state, candidate) pair is excluded when the forced first code was not legal there (the plan fell back to the
policy, so the comparison is empty); the counts are reported.
"""

from __future__ import annotations

import glob
import gzip
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import load_json, save_json, write_md
from .horizon import (CONSTRUCTED, FAMILY_CANDS, GAMMA, HOUT, K, LEDGER_CP, LONG_CP, NATURAL, PARTIAL, SHORT_CP)

FIG = os.path.join(HOUT, "figures")
H_MAIN = (1, 5, 10, 20, 30, 50)
H_ALL = (1, 5, 10, 20, 30, 50, 100, 150, 300)
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GREY, INK, MUTED = "#a3a29c", "#0b0b0b", "#52514e"


def tcrit(n: int) -> float:
    from scipy import stats
    return float(stats.t.ppf(0.975, max(n - 1, 1)))


def load() -> List[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(PARTIAL, "*.json.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def intended(spec) -> Optional[int]:
    return int(spec[1]) if spec and spec[0] == "fixed" else None


def units(jobs: List[dict], fam: str, proto: str, cand: str, X: str = "G", long_only: bool = False):
    """[(state index, {cp: paired diffs (K,)}, legal share)] for one comparison."""
    out = []
    for j in jobs:
        if j["family"] != fam or j["protocol"] != proto or cand not in j["GQ"]:
            continue
        if long_only and j["horizon"] < 300:
            continue
        code = intended(j["candidates"][cand])
        legal = 1.0 if code is None else float(np.mean([c == code for c in j["first_codes"][cand]]))
        if legal == 0.0:
            continue
        gc, gp = j["GQ"][cand][X], j["GQ"]["policy"][X]
        out.append((j["index"], {int(c): np.asarray(gc[c]) - np.asarray(gp[c]) for c in gc}, legal))
    return out


def aggregate(us, h: int) -> Optional[dict]:
    rows = [(u[1][h]) for u in us if h in u[1]]
    if len(rows) < 2:
        return None
    m = np.array([r.mean() for r in rows])
    w = np.array([r.var(ddof=1) for r in rows])
    n = len(m)
    se = float(m.std(ddof=1) / math.sqrt(n))
    sw2 = float(w.mean())
    sb2 = max(float(m.var(ddof=1) - sw2 / K), 0.0)
    return {"n": n, "mean": float(m.mean()), "se": se, "ci": tcrit(n) * se, "sigma_w2": sw2, "sigma_b2": sb2,
            "var_rollout": float(np.concatenate(rows).var(ddof=1)),
            "share_sig": float(np.mean(np.abs(m) > 2 * np.sqrt(w / K) + 1e-12))}


# ---------------------------------------------------------------------------------------------- analyses
def horizon_tables(jobs) -> dict:
    """M(h) +- CI for G and Q, by family, protocol and candidate, at every horizon with data."""
    out = {}
    for j in jobs:
        out.setdefault(j["family"], {}).setdefault(j["protocol"], set()).update(c for c in j["GQ"] if c != "policy")
    res = {}
    for fam, protos in out.items():
        for proto, cands in protos.items():
            for cand in sorted(cands):
                for X in ("G", "Q"):
                    us = units(jobs, fam, proto, cand, X)
                    usl = units(jobs, fam, proto, cand, X, long_only=True)
                    row = {"n_states": len(us), "n_long": len(usl),
                           "all": {h: aggregate(us, h) for h in H_ALL if aggregate(us, h)},
                           "long": {h: aggregate(usl, h) for h in H_ALL if aggregate(usl, h)}}
                    res.setdefault(fam, {}).setdefault(proto, {}).setdefault(cand, {})[X] = row
    return res


def rankings(tab: dict) -> dict:
    """Candidate ranking (policy at 0 included) by M(h) on the long units, per family and protocol, and how it moves
    with h: the top candidate at each h, and agreement of every horizon's ranking with the 300 ranking (Kendall tau)."""
    from scipy.stats import kendalltau
    out = {}
    for fam, protos in tab.items():
        for proto, cands in protos.items():
            for X in ("G", "Q"):
                per_h = {}
                for h in H_ALL:
                    vals = {"policy": 0.0}
                    for c, d in cands.items():
                        r = d[X]["long"].get(h)
                        if r:
                            vals[c] = r["mean"]
                    if len(vals) > 1:
                        per_h[h] = vals
                if 300 not in per_h:
                    continue
                ref = per_h[300]
                names = sorted(ref)
                row = {}
                for h, vals in per_h.items():
                    common = [n for n in names if n in vals]
                    tau = kendalltau([vals[n] for n in common], [ref[n] for n in common])[0] if len(common) > 2 else float("nan")
                    row[h] = {"top": max(vals, key=vals.get), "order": sorted(vals, key=vals.get, reverse=True),
                              "tau_vs_300": float(tau) if tau == tau else None}
                out.setdefault(fam, {}).setdefault(proto, {})[X] = row
    return out


def agreement(jobs) -> dict:
    """Per (state, candidate) on the long units: does the state-level conclusion at h agree with the one at 150 / 300?
    A conclusion is 'better' / 'worse' when |m_u| > 2 SE_u (SE from the K paired rollouts), else 'unclear'.
    Contradiction = significant at both horizons with opposite signs. Also the correlation across (state, candidate)
    of m_u(h) with m_u(300)."""
    out = {}
    for X in ("G", "Q"):
        per = {h: {"agree": 0, "contradict": 0, "short_unclear": 0, "long_unclear": 0, "n": 0, "pairs": []}
               for h in (5, 10, 20, 30, 50, 100, 150)}
        for j in jobs:
            if j["horizon"] < 300:
                continue
            for cand in j["GQ"]:
                if cand == "policy":
                    continue
                us = units([j], j["family"], j["protocol"], cand, X)
                if not us:
                    continue
                d = us[0][1]
                for ref_h in (300,):
                    r = d[ref_h]
                    rm, rs = r.mean(), r.std(ddof=1) / math.sqrt(len(r))
                    rc = 0 if abs(rm) <= 2 * rs else int(np.sign(rm))
                    for h in per:
                        x = d[h]
                        xm, xs = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
                        xc = 0 if abs(xm) <= 2 * xs + 1e-12 else int(np.sign(xm))
                        p = per[h]
                        p["n"] += 1
                        p["pairs"].append((float(xm), float(rm)))
                        if xc == 0:
                            p["short_unclear"] += 1
                        if rc == 0:
                            p["long_unclear"] += 1
                        if xc and rc:
                            p["agree" if xc == rc else "contradict"] += 1
        for h, p in per.items():
            a = np.array(p.pop("pairs"))
            p["corr_with_300"] = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if len(a) > 2 else None
            p["slope_300_on_h"] = float(np.polyfit(a[:, 0], a[:, 1], 1)[0]) if len(a) > 2 else None
        out[X] = per
    return out


def contradictions(jobs) -> List[dict]:
    """Every (state, candidate) whose 50-turn conclusion is significant and contradicts a significant 300-turn
    conclusion, with what changed between 50 and 300 (ledger differences) and the critic's prediction at 50."""
    out = []
    for j in jobs:
        if j["horizon"] < 300:
            continue
        for cand in j["GQ"]:
            if cand == "policy":
                continue
            us = units([j], j["family"], j["protocol"], cand, "G")
            if not us:
                continue
            d = us[0][1]
            s50, s300 = d[50], d[300]
            sig = lambda x: 0 if abs(x.mean()) <= 2 * x.std(ddof=1) / math.sqrt(len(x)) else int(np.sign(x.mean()))
            if sig(s50) and sig(s300) and sig(s50) != sig(s300):
                led = j["ledger"][cand]
                keys = j["keys"]
                d50 = dict(zip(keys, led["50"]["diff"]))
                d300 = dict(zip(keys, led["300"]["diff"]))
                q = units([j], j["family"], j["protocol"], cand, "Q")[0][1]
                out.append({"family": j["family"], "state": j["index"], "protocol": j["protocol"], "candidate": cand,
                            "G50": float(s50.mean()), "G300": float(s300.mean()), "Q50": float(q[50].mean()),
                            "Q300": float(q[300].mean()),
                            "at50": {k: d50[k] for k in ("hpA", "hpB", "invA", "invB", "lvlA", "lvlB", "absA", "absB",
                                                         "siegeA", "siegeB", "heldA", "V")},
                            "change_50_300": {k: d300[k] - d50[k] for k in ("earned_A", "earned_B", "xp_A", "xp_B",
                                                                           "deaths_A", "deaths_B", "plates_A",
                                                                           "plates_B", "unc_A", "unc_B")}})
    return out


def efficiency(jobs, budget: float = 1_000_000.0) -> dict:
    """For each natural family and candidate (protocol "policy"): bias of G_h and Q_h against Q_300 on the long
    subset, the variance components at h, and the RMSE of the best (states S, rollouts K) allocation of a fixed budget
    of `budget` rollout-turns per comparison (candidate + baseline: 2 h turns per paired rollout)."""
    out = {}
    for fam in NATURAL:
        for cand in FAMILY_CANDS[fam]:
            ref = {u[0]: u[1][300].mean() for u in units(jobs, fam, "policy", cand, "Q", long_only=True)}
            if len(ref) < 3:
                continue
            row = {}
            for X in ("G", "Q"):
                us_all = units(jobs, fam, "policy", cand, X)
                us_long = units(jobs, fam, "policy", cand, X, long_only=True)
                for h in (5, 10, 20, 30, 50, 100, 150, 300):
                    src = us_all if h <= 50 else us_long
                    a = aggregate(src, h)
                    if not a:
                        continue
                    bias_s = [u[1][h].mean() - ref[u[0]] for u in us_long if u[0] in ref and h in u[1]]
                    b = float(np.mean(bias_s)) if bias_s else float("nan")
                    bse = float(np.std(bias_s, ddof=1) / math.sqrt(len(bias_s))) if len(bias_s) > 1 else float("nan")
                    best = None
                    for k in (1, 2, 4, 8, 16, 32):
                        S = budget / (2 * h * k)
                        if S < 2:
                            continue
                        var = a["sigma_b2"] / S + a["sigma_w2"] / (S * k)
                        mse = (b if np.isfinite(b) else 0.0) ** 2 + var
                        if best is None or mse < best["mse"]:
                            best = {"K": k, "S": S, "se": math.sqrt(var), "mse": mse, "rmse": math.sqrt(mse)}
                    row[f"{X}{h}"] = {"bias_vs_Q300": b, "bias_se": bse, "sigma_w2": a["sigma_w2"],
                                      "sigma_b2": a["sigma_b2"], "best": best,
                                      "share_sigma_w": a["sigma_w2"] / K / (a["sigma_w2"] / K + a["sigma_b2"] + 1e-12)}
            out.setdefault(fam, {})[cand] = row
    return out


def unresolved(jobs) -> dict:
    """On the long natural units: what predicts the part of the long-run effect that a 50-turn rollout misses,
    late = m_u(Q300) - m_u(G50)? Correlation with the critic's prediction 0.99^50 (V_c(s_50) - V_p(s_50)) and with
    the state differences at 50."""
    rows = []
    for j in jobs:
        if j["horizon"] < 300 or j["family"] not in NATURAL:
            continue
        keys = j["keys"]
        for cand in j["GQ"]:
            if cand == "policy":
                continue
            g = units([j], j["family"], j["protocol"], cand, "G")
            q = units([j], j["family"], j["protocol"], cand, "Q")
            if not g:
                continue
            d50 = dict(zip(keys, j["ledger"][cand]["50"]["diff"]))
            rows.append({"late": float(q[0][1][300].mean() - g[0][1][50].mean()),
                         "late_G": float(g[0][1][300].mean() - g[0][1][50].mean()),
                         "pred_V": GAMMA ** 50 * d50["V"], **{k: d50[k] for k in keys}})
    if len(rows) < 5:
        return {}
    late = np.array([r["late"] for r in rows])
    feats = ("pred_V", "hpA", "hpB", "invA", "invB", "lvlA", "lvlB", "absA", "absB", "visA", "heldA", "frozenA",
             "siegeA", "siegeB", "earned_A", "earned_B", "deaths_A", "deaths_B")
    corr = {}
    for f in feats:
        x = np.array([r[f] for r in rows], dtype=float)
        ok = np.isfinite(x)
        if ok.sum() > 4 and x[ok].std() > 0:
            corr[f] = float(np.corrcoef(x[ok], late[ok])[0, 1])
    pv = np.array([r["pred_V"] for r in rows])
    return {"n": len(rows), "late_mean": float(late.mean()), "late_sd": float(late.std(ddof=1)),
            "corr": corr, "pred_V_slope": float(np.polyfit(pv, late, 1)[0]),
            "share_late_explained_by_V": float(np.corrcoef(pv, late)[0, 1] ** 2)}


def mechanics(jobs) -> dict:
    """Mean paired ledger / state differences (candidate - policy) at the early checkpoints, by family and candidate
    (protocol "policy"; natural units and constructed cells pooled per family): the immediate mechanical and local
    economic consequences."""
    out = {}
    for j in jobs:
        if j["protocol"] != "policy":
            continue
        keys = j["keys"]
        for cand, by_cp in j["ledger"].items():
            if cand == "policy":
                continue
            code = intended(j["candidates"][cand])
            if code is not None and not any(c == code for c in j["first_codes"][cand]):
                continue
            for cp, s in by_cp.items():
                acc = out.setdefault(j["family"], {}).setdefault(cand, {}).setdefault(cp, {})
                for k, v in zip(keys, s["diff"]):
                    if v is not None and np.isfinite(v):
                        acc.setdefault(k, []).append(v)
    return {f: {c: {cp: {k: [float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None, len(v)]
                         for k, v in d.items()} for cp, d in cs.items()} for c, cs in fc.items()} for f, fc in out.items()}


def freeze_case(jobs) -> dict:
    """The constructed freeze grid: per cell, opponent protocol and candidate: persistence P(A holds at t), denial
    (B's farm lost and XP lost), freezer HP, deaths and the value differences over horizons."""
    out = {}
    for j in jobs:
        if j["family"] not in ("F_freeze_c", "B_freeze") or not j.get("series"):
            continue
        key = f"{j['family']}|{j['index']}|{j['protocol']}"
        row = {"label": j["label"], "horizon": j["horizon"]}
        for cand, ser in j["series"].items():
            g = j["GQ"][cand]["G"]
            gp = j["GQ"]["policy"]["G"]
            q = j["GQ"][cand]["Q"]
            qp = j["GQ"]["policy"]["Q"]
            dv = {}
            for h in H_ALL:
                if str(h) in g:
                    d = np.asarray(g[str(h)]) - np.asarray(gp[str(h)])
                    e = np.asarray(q[str(h)]) - np.asarray(qp[str(h)])
                    dv[h] = {"dG": float(d.mean()), "dG_se": float(d.std(ddof=1) / math.sqrt(len(d))),
                             "dQ": float(e.mean()), "dQ_se": float(e.std(ddof=1) / math.sqrt(len(e))),
                             "G": float(np.mean(g[str(h)]))}
            row[cand] = {"held": [ser["heldA"][t - 1] for t in (1, 2, 5, 10, 20, 30, 50)],
                         "unc_B": [ser["unc_B"][t - 1] for t in (5, 10, 20, 50)],
                         "xp_lost_B": [ser["xp_lost_B"][t - 1] for t in (5, 10, 20, 50)],
                         "hpA": [ser["hpA"][t - 1] for t in (1, 5, 10, 20)],
                         "deaths": [[ser["deaths_A"][t - 1], ser["deaths_B"][t - 1]] for t in (10, 20, 50)],
                         "lead": [ser["lead"][t - 1] for t in (5, 10, 20, 50)], "value": dv}
        out[key] = row
    return out


def constructed_pooled(jobs, fam: str) -> dict:
    """Constructed families: per opponent protocol and candidate, the paired difference pooled over the family's
    cells (fixed effect: mean of the cell means; SE from the within-cell rollout variances, sqrt(sum w_u / K) / n), at
    every horizon, for G and Q; plus, per cell, the 300-turn sign and the first horizon whose sign is significant."""
    out = {}
    for proto in sorted({j["protocol"] for j in jobs if j["family"] == fam}):
        for cand in FAMILY_CANDS[fam]:
            for X in ("G", "Q"):
                us = units(jobs, fam, proto, cand, X)
                if not us:
                    continue
                row = {}
                for h in H_ALL:
                    rs = [u[1][h] for u in us if h in u[1]]
                    if not rs:
                        continue
                    m = float(np.mean([r.mean() for r in rs]))
                    se = float(math.sqrt(sum(r.var(ddof=1) / len(r) for r in rs)) / len(rs))
                    row[h] = {"mean": m, "se": se, "n_cells": len(rs)}
                cells = []
                for u in us:
                    d = u[1]
                    ref = d[300]
                    rsig = 0 if abs(ref.mean()) <= 2 * ref.std(ddof=1) / math.sqrt(len(ref)) else int(np.sign(ref.mean()))
                    first = None
                    for h in H_ALL:
                        x = d[h]
                        if abs(x.mean()) > 2 * x.std(ddof=1) / math.sqrt(len(x)) + 1e-12:
                            first = (h, int(np.sign(x.mean())))
                            break
                    cells.append({"cell": u[0], "sign_300": rsig, "first_sig": first,
                                  "agrees": (first[1] == rsig) if (first and rsig) else None})
                out.setdefault(proto, {}).setdefault(cand, {})[X] = {"pooled": row, "cells": cells}
    return out


def freeze_early(fc: dict) -> dict:
    """The hypothesis that unsuccessful freezes are identifiable early: for the sustained keep plan in every
    constructed (cell, protocol), the 300-turn verdict (dQ300 > 0 = the freeze paid) against the early indicators
    P(still held at 10) and the sign of dG / dQ at 10, 20 and 50 (|d| > 2 SE)."""
    rows = []
    for key, r in fc.items():
        if not key.startswith("F_freeze_c") or "plan:keep" not in r:
            continue
        v = r["plan:keep"]["value"]
        ref = v["300"] if "300" in v else v.get(300)
        if not ref:
            continue
        def s(h, k):
            x = v[str(h)] if str(h) in v else v[h]
            return 0 if abs(x[k]) <= 2 * x[k + "_se"] else int(np.sign(x[k]))
        rows.append({"cell": key, "held10": r["plan:keep"]["held"][3], "held5": r["plan:keep"]["held"][2],
                     "dQ300": ref["dQ"], "sig300": s(300, "dQ"),
                     **{f"dG{h}": s(h, "dG") for h in (10, 20, 50)}, **{f"dQ{h}": s(h, "dQ") for h in (5, 10, 20, 50)}})
    out = {"n": len(rows), "rows": rows}
    for k in ("dG10", "dG20", "dG50", "dQ5", "dQ10", "dQ20", "dQ50"):
        sig = [x for x in rows if x[k] and x["sig300"]]
        out[k] = {"significant_both": len(sig), "agree": sum(x[k] == x["sig300"] for x in sig),
                  "short_significant": sum(bool(x[k]) for x in rows)}
    paid = [x for x in rows if x["sig300"] > 0]
    failed = [x for x in rows if x["sig300"] < 0]
    out["held10_when_paid"] = [x["held10"] for x in paid]
    out["held10_when_failed"] = [x["held10"] for x in failed]
    return out


def earliest_informative(tab: dict) -> dict:
    """Per comparison (protocol "policy" for natural families; every protocol for constructed): the first horizon at
    which M(h) is significant (|M| > CI) and whether its sign then agrees with the 300 value (long units)."""
    out = {}
    for fam, protos in tab.items():
        for proto, cands in protos.items():
            for cand, d in cands.items():
                for X in ("G", "Q"):
                    lg = d[X]["long"]
                    if 300 not in lg:
                        continue
                    ref = lg[300]
                    rsig = 0 if abs(ref["mean"]) <= ref["ci"] else int(np.sign(ref["mean"]))
                    first = None
                    for h in H_ALL:
                        r = lg.get(h)
                        if r and abs(r["mean"]) > r["ci"]:
                            first = (h, int(np.sign(r["mean"])))
                            break
                    out.setdefault(fam, {}).setdefault(proto, {}).setdefault(cand, {})[X] = {
                        "first_significant_h": first[0] if first else None,
                        "sign_then": first[1] if first else 0, "sign_300": rsig,
                        "agrees": (first[1] == rsig) if (first and rsig) else None}
    return out


# ---------------------------------------------------------------------------------------------- figures
def _style(ax):
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig_curves(tab: dict, X: str, name: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = list(NATURAL)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for ax, fam in zip(axes.ravel(), fams):
        cands = tab.get(fam, {}).get("policy", {})
        ax.axhline(0, color=GREY, linewidth=1)
        for i, (c, d) in enumerate(sorted(cands.items())):
            lg = d[X]["long"]
            hs = [h for h in H_ALL if h in lg]
            if not hs:
                continue
            m = np.array([lg[h]["mean"] for h in hs])
            ci = np.array([lg[h]["ci"] for h in hs])
            col = PALETTE[i % len(PALETTE)]
            ax.fill_between(hs, m - ci, m + ci, color=col, alpha=0.12, linewidth=0)
            ax.plot(hs, m, color=col, linewidth=2, marker="o", markersize=3, label=c)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20, 50, 100, 300])
        ax.set_xticklabels(["1", "5", "10", "20", "50", "100", "300"])
        ax.set_title(fam, fontsize=10, color=INK)
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
    fig.supxlabel("horizon h (turns, log scale)", fontsize=9, color=MUTED)
    fig.supylabel(f"paired difference candidate − policy, {X}_h (resources; 95% CI over the 12 long states)",
                  fontsize=9, color=MUTED)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def fig_variance(eff: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    hs = (5, 10, 20, 30, 50, 100, 150, 300)
    for ax, X in zip(axes, ("G", "Q")):
        for i, fam in enumerate(NATURAL):
            vals = []
            for h in hs:
                v = [eff[fam][c][f"{X}{h}"]["sigma_w2"] for c in eff.get(fam, {}) if f"{X}{h}" in eff[fam][c]]
                vals.append(math.sqrt(np.mean(v)) if v else np.nan)
            ax.plot(hs, vals, color=PALETTE[i], linewidth=2, marker="o", markersize=3, label=fam)
        ax.set_xscale("log")
        ax.set_xticks(hs)
        ax.set_xticklabels([str(h) for h in hs])
        ax.set_title(f"SD of one paired rollout difference, {X}_h", fontsize=10, color=INK)
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
    axes[0].set_ylabel("resources", fontsize=9, color=MUTED)
    fig.supxlabel("horizon h (turns, log scale)", fontsize=9, color=MUTED)
    fig.tight_layout()
    p = os.path.join(FIG, "rollout_sd.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def fig_freeze(fc: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(k for k in fc if k.startswith("F_freeze_c"))
    protos = ("policy", "wk_shove", "wk_back", "wk_shove_reset")
    cells = sorted({int(k.split("|")[1]) for k in keys})
    fig, axes = plt.subplots(len(protos), len(cells), figsize=(2.6 * len(cells), 2.2 * len(protos)), sharex=True,
                             sharey=True)
    ts = (1, 2, 5, 10, 20, 30, 50)
    cands = ("policy", "plan:keep", "plan:guard", "1t:break")
    for r, proto in enumerate(protos):
        for c, cell in enumerate(cells):
            ax = axes[r][c]
            row = fc.get(f"F_freeze_c|{cell}|{proto}")
            if row:
                for i, cand in enumerate(cands):
                    if cand in row:
                        ax.plot(ts, row[cand]["held"], color=PALETTE[i], linewidth=1.8, marker="o", markersize=2.5,
                                label=cand)
                if r == 0:
                    ax.set_title(row["label"].replace("constructed ", ""), fontsize=7, color=INK)
            if c == 0:
                ax.set_ylabel(f"vs {proto}", fontsize=8, color=MUTED)
            ax.set_xscale("log")
            ax.set_xticks([1, 5, 10, 50])
            ax.set_xticklabels(["1", "5", "10", "50"])
            _style(ax)
    axes[0][0].legend(fontsize=6, frameon=False)
    fig.supxlabel("turn t (log scale)", fontsize=9, color=MUTED)
    fig.suptitle("P(the freezer still holds the freeze at turn t), constructed freeze cells (gap, held HP, level); "
                 "rows: opponent", fontsize=10, color=INK)
    fig.tight_layout()
    p = os.path.join(FIG, "freeze_persistence.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------------------------- summary
def summarize() -> dict:
    jobs = load()
    man = load_json("manifest", HOUT) or {}
    tab = horizon_tables(jobs)
    data = {"meta": man.get("meta"), "timing": man.get("timing"), "n_jobs": len(jobs),
            "turns": int(sum(j["turns"] for j in jobs)), "job_seconds": float(sum(j["elapsed_s"] for j in jobs)),
            "tables": tab, "rankings": rankings(tab), "agreement": agreement(jobs),
            "contradictions": contradictions(jobs), "efficiency": efficiency(jobs), "unresolved": unresolved(jobs),
            "mechanics": mechanics(jobs), "freeze": freeze_case(jobs), "earliest": earliest_informative(tab),
            "constructed": {f: constructed_pooled(jobs, f) for f in CONSTRUCTED},
            "legal": legality(jobs), "states": (man.get("states") or {})}
    data["freeze_early"] = freeze_early(data["freeze"])
    save_json("summary", data, HOUT)
    fig_curves(tab, "G", "advantage_G.png")
    fig_curves(tab, "Q", "advantage_Q.png")
    fig_variance(data["efficiency"])
    fig_freeze(data["freeze"])
    report_md()                                  # from the saved JSON (string keys), as a reader would
    return data


def _t(header, rows) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def report_md(d: Optional[dict] = None) -> str:
    """results/m45c/horizon/report.md: every table of the study, generated from summary.json."""
    d = d or load_json("summary", HOUT)
    f = lambda r, h: f"{r[h]['mean']:+.2f} ± {r[h]['ci']:.2f}" if r and h in r else ""
    parts = ["# M4.5c horizon study — generated tables\n",
             f"*{d['n_jobs']} jobs, {d['turns']:,} simulated turns, {d['job_seconds'] / 3600:.2f} worker-hours. Paired "
             "differences candidate − policy (resources); ± = 95% t interval over states. Prose: "
             "docs/m4/m4_5/m4_5c_horizon_study.md.*\n"]
    rows = []
    for fam in NATURAL:
        for cand, dd in sorted(d["tables"][fam]["policy"].items()):
            for X in ("G", "Q"):
                a, lg = dd[X]["all"], dd[X]["long"]
                rows.append([fam, cand, X, f(a, "5"), f(a, "10"), f(a, "20"), f(a, "50"), f(lg, "50"), f(lg, "150"),
                             f(lg, "300")])
    parts.append("## Natural families (protocol: policy)\n\nColumns 5–50: all 40 states; last three: the 12 long states.\n")
    parts.append(_t(["family", "candidate", "X", "5", "10", "20", "50", "50 (long)", "150 (long)", "300 (long)"], rows))
    for fam in CONSTRUCTED:
        rows = []
        for proto, cs in d["constructed"][fam].items():
            for cand, x in cs.items():
                for X in ("G", "Q"):
                    p = x[X]["pooled"]
                    rows.append([proto, cand, X] + [f"{p[h]['mean']:+.2f} ± {1.96 * p[h]['se']:.2f}" if h in p else ""
                                                    for h in ("1", "5", "10", "20", "50", "300")])
        parts.append(f"\n## {fam} (constructed; pooled over cells, SE from the within-cell rollouts)\n")
        parts.append(_t(["opponent", "candidate", "X", "1", "5", "10", "20", "50", "300"], rows))
    rows = []
    for X in ("G", "Q"):
        for h, v in d["agreement"][X].items():
            rows.append([X, h, v["n"], v["agree"], v["contradict"], v["short_unclear"], v["long_unclear"],
                         f"{v['corr_with_300']:.2f}", f"{v['slope_300_on_h']:.2f}"])
    parts.append("\n## State-level agreement with 300 turns (every long unit and candidate)\n")
    parts.append(_t(["X", "h", "pairs", "agree", "contradict", "unclear at h", "unclear at 300", "corr with 300",
                     "slope of 300 on h"], rows))
    rk = d["rankings"]
    rows = []
    for X in ("G", "Q"):
        for h in ("5", "10", "20", "50", "150"):
            cells = [(fm, p) for fm in rk for p in rk[fm] if X in rk[fm][p] and h in rk[fm][p][X]]
            agree = sum(rk[fm][p][X][h]["top"] == rk[fm][p][X]["300"]["top"] for fm, p in cells)
            taus = [rk[fm][p][X][h]["tau_vs_300"] for fm, p in cells if rk[fm][p][X][h]["tau_vs_300"] is not None]
            rows.append([X, h, f"{agree}/{len(cells)}", f"{np.median(taus):.2f}"])
    parts.append("\n## Ranking stability (family × protocol; long units)\n")
    parts.append(_t(["X", "h", "top candidate = top at 300", "median Kendall tau vs 300"], rows))
    eff = d["efficiency"]
    rows = []
    for X in ("G", "Q"):
        for h in (5, 10, 20, 30, 50, 100, 150, 300):
            k = f"{X}{h}"
            v = [r[k] for cs in eff.values() for r in cs.values() if k in r]
            sw = math.sqrt(np.mean([x["sigma_w2"] for x in v]))
            sb = math.sqrt(np.mean([x["sigma_b2"] for x in v]))
            cost = np.mean([h * (x["sigma_b2"] + x["sigma_w2"] / 8) for x in v])
            rows.append([X, h, f"{sw:.2f}", f"{sb:.2f}", f"{cost:.1f}"])
    base = float(rows[-1][4]) if rows else 1.0
    base = [float(r[4]) for r in rows if r[0] == "G" and r[1] == 300][0]
    rows = [r + [f"{base / float(r[4]):.1f}×"] for r in rows]
    parts.append("\n## Variance and precision per unit of compute (natural families, protocol policy)\n\n"
                 "sigma_w: SD of one paired rollout difference; sigma_b: SD of the true state effects; cost index "
                 "h (sigma_b² + sigma_w²/8) = variance of a family-level estimate times its rollout-turn budget "
                 "(K = 8 rollouts per state); precision per compute relative to G300.\n")
    parts.append(_t(["X", "h", "sigma_w", "sigma_b", "cost index", "precision per compute"], rows))
    u = d["unresolved"]
    parts.append(f"\n## What a 50-turn rollout leaves unresolved (natural long units)\n\nlate = m(Q300) − m(G50) over "
                 f"{u['n']} (state, candidate) pairs: mean {u['late_mean']:+.3f}, SD {u['late_sd']:.3f}; critic "
                 f"prediction 0.99^50 ΔV(s_50): slope {u['pred_V_slope']:.2f}, R² {u['share_late_explained_by_V']:.3f}; "
                 "correlations: " + ", ".join(f"{k} {v:+.2f}" for k, v in u["corr"].items()) + "\n")
    fe = d["freeze_early"]
    parts.append("\n## Freeze plan (sustained keep) — early verdicts against the 300-turn verdict\n")
    parts.append(_t(["estimator", "significant at both", "agree", "significant at h"],
                    [[k, v["significant_both"], v["agree"], v["short_significant"]] for k, v in fe.items()
                     if isinstance(v, dict)]))
    return write_md("report", "\n".join(parts), HOUT)


def legality(jobs) -> dict:
    out = {}
    for j in jobs:
        for cand, spec in j["candidates"].items():
            code = intended(spec)
            if code is None:
                continue
            ok = any(c == code for c in j["first_codes"][cand])
            d = out.setdefault(j["family"], {}).setdefault(cand, [0, 0])
            d[0] += ok
            d[1] += 1
    return out


__all__ = ["summarize", "load", "units", "aggregate", "horizon_tables", "rankings", "agreement", "contradictions",
           "efficiency", "unresolved", "mechanics", "freeze_case", "earliest_informative"]
