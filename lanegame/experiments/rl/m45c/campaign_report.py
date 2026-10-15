"""Generated tables and figures of the M4.5c final research report: results/m45c/final/report.md, summary.json and
figures/. Nothing here simulates; it reads the records of the 8.192M campaign (results/m45c/full/), the matched 2x
extension (results/m45c/ext/), the strategic battery (results/m45c/battery/ and its raw jobs), the attackers
(results/m45c/attackers/) and the traces (results/m45c/traces/).

Statistics. Training seeds are the unit of replication: seed means carry t intervals (5 degrees of freedom), and
budget differences are PAIRED by seed (the extension's runs are fresh runs of the same seeds: same initial weights,
same evaluation seeds). Battery values are paired differences candidate - the policy's own choice; per evaluated policy
a family mean over states with a t interval over states; pooled over the six policies with a t interval over policies;
a policy "supports" a sign when its own interval excludes 0.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .common import CKPT, OUT, load_json, save_json, write_md
from .full import FULL_SEEDS

FOUT = os.path.join(OUT, "final")
FIG = os.path.join(FOUT, "figures")
BLUE, ORANGE, GREY, INK, MUTED, SEEDLINE = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e", "#b9b8b1"
SEEDS = list(FULL_SEEDS)


def _t(v: Sequence[float]) -> List[float]:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    if len(v) == 1:
        return [float(v[0]), float("nan")]
    from scipy import stats
    return [float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))]


def _pm(m, h, d=2) -> str:
    if m is None or not np.isfinite(m):
        return "n/a"
    return f"{m:+.{d}f}" + (f" ± {h:.{d}f}" if h is not None and np.isfinite(h) else "")


def _f(x, d=2, sign=True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def table(header, rows) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- budget comparison
def budget_training() -> dict:
    out = {}
    for phase in ("full", "ext"):
        d = load_json("health_training", os.path.join(OUT, phase))
        tr = load_json(f"{phase}_train", os.path.join(OUT, phase))
        if not d:
            continue
        S = d["seeds"]
        keys = ("entropy", "approx_kl", "explained_variance", "grad_norm", "act_cat_SH", "act_cat_Ward", "act_cat_Recall",
                "act_beh_Trade", "act_beh_Safe")
        row = {k: {"last_tenth": _t([S[s]["tenths"][k][-1] for s in S]), "tenths": np.mean(
            [S[s]["tenths"][k] for s in S], 0).tolist()} for k in keys}
        for k in S[next(iter(S))]["episode_tenths"]:
            row[k] = {"last_tenth": _t([S[s]["episode_tenths"][k][-1] for s in S]),
                      "tenths": np.mean([S[s]["episode_tenths"][k] for s in S], 0).tolist()}
        row["alarms"] = {s: v["stability"]["alarms"] for s, v in S.items()}
        if tr:
            runs = {n: r for n, r in tr["runs"].items() if r.get("status") == "complete" and not n.endswith("_repeat")}
            row["runs"] = {n: {"transitions": r["agent_transitions"], "updates": r["updates"],
                               "minutes": r["elapsed_s"] / 60, "tps": r["transitions_per_s"],
                               "peak_mb": r["memory"].get("peak_mb"), "digest": r["digest"][:16],
                               "initial_digest": r["initial_digest"][:16]} for n, r in runs.items()}
            row["checks"] = {"n": len(tr["checks"]), "failed": [k for k, v in tr["checks"].items() if not v],
                             "all_passed": tr["all_passed"]}
        out[phase] = row
    return out


BEH_ROWS = (("overall", "SH", "Shove mass"), ("overall", "Ward", "Ward mass"), ("overall", "Recall", "Recall mass"),
            ("overall", "TN", "Thin mass"), ("overall", "beh_Trade", "Trade (of lane)"),
            ("overall", "beh_Safe", "Safe (of lane)"), ("overall", "entropy", "entropy"),
            ("freezes", "per_1000", "freezes per 1000 turns"), ("freezes", "denial_share", "freeze denial share"),
            ("freezes", "length", "freeze length"), ("summary", "wards_per_1000", "Wards per 1000 turns"),
            ("summary", "recalls_per_1000", "Recalls per 1000 turns"), ("summary", "deaths_per_1000", "deaths per 1000"),
            ("summary", "trade_attempts_per_1000", "trade attempts per 1000"),
            ("summary", "collection_rate", "farm collection"),
            ("ledger", "recall_interrupted_share", "Recall attempts interrupted"),
            ("ledger", "minion_share_of_damage", "minion share of damage"))
CONTRAST_ROWS = ("HP < 35%::Recall", "HP < 35%::beh_Safe", "HP >= 70%::beh_Trade", "holding a freeze (freezer)::TN",
                 "holding a freeze (freezer)::beh_Trade", "frozen on::SH", "enemy champion away::SH",
                 "Ward legal, own vision down::Ward", "own crash under the enemy tower, Pressure legal::Pressure",
                 "level 1-5::SH", "level 1-5::beh_Trade")


def budget_behaviour() -> dict:
    """Per seed and paired (ext - full) differences of the finals' behaviour readouts (self-play)."""
    sm = {ph: load_json("summary", os.path.join(OUT, ph)) for ph in ("full", "ext")}
    if not sm["full"] or not sm["ext"]:
        return {}
    out = {}
    for sec, k, lab in BEH_ROWS:
        v = {}
        for ph in ("full", "ext"):
            b = sm[ph]["behavior"].get(sec, {})
            v[ph] = [b.get(f"{s}|final", {}).get(k) for s in SEEDS]
        d = [e - f for e, f in zip(v["ext"], v["full"]) if e is not None and f is not None]
        out[lab] = {"full": _t(v["full"]), "ext": _t(v["ext"]), "diff": _t(d), "per_seed_full": v["full"],
                    "per_seed_ext": v["ext"]}
    for q in CONTRAST_ROWS:
        v = {}
        for ph in ("full", "ext"):
            c = sm[ph]["behavior"].get("contrasts", {})
            v[ph] = [(c.get(f"{s}|final", {}).get(q) or [None, None])[1] for s in SEEDS]
        d = [e - f for e, f in zip(v["ext"], v["full"]) if e is not None and f is not None]
        out[q] = {"full": _t(v["full"]), "ext": _t(v["ext"]), "diff": _t(d)}
    gap = {}
    for ph in ("full", "ext"):
        pg = sm[ph]["behavior"].get("power_gap", {})
        gap[ph] = {
            "weaker_yield": _t([pg[s]["weaker_mass"]["beh_Yield"] for s in pg if pg[s].get("weaker_mass")]),
            "weaker_at_tower": _t([pg[s]["by_power"].get("weaker", {}).get("at_own_tower") for s in pg]),
            "stronger_trade_rate": _t([pg[s]["by_power"].get("stronger", {}).get("trade_attempt_rate") for s in pg]),
            "stronger_pos": _t([pg[s]["by_power"].get("stronger", {}).get("mean_pos") for s in pg]),
            "denial_when_near": _t([(pg[s]["pressure_cost"]["opp_uncollected_when_stronger_near"]["mean"]
                                     - pg[s]["pressure_cost"]["opp_uncollected_otherwise"]["mean"]) for s in pg])}
    out["_power_gap"] = gap
    return out


def budget_strength() -> dict:
    sm = {ph: load_json("summary", os.path.join(OUT, ph)) for ph in ("full", "ext")}
    out = {}
    for ph in ("full", "ext"):
        if sm[ph]:
            cv = sm[ph]["curve"]
            out[ph] = {"curve": {k: v["scripted"] for k, v in cv["pooled"].items()},
                       "curve_ood": {k: v["ood"] for k, v in cv["pooled"].items()},
                       "h2h": {k: v["pooled"] for k, v in sm[ph]["h2h"].items()},
                       "cross_max": sm[ph]["cross"]["max_abs"], "cross_sig": sm[ph]["cross"]["pairs_significant"],
                       "cycle": sm[ph]["cycle"]["seeds_with_a_significant_regression"],
                       "rules": sm[ph]["rules"]}
    if sm["ext"]:
        out["comprehensive"] = sm["ext"].get("comprehensive", {})
    return out


# ---------------------------------------------------------------------------------------------- battery
def _battery(name: str) -> Optional[dict]:
    return load_json(f"summary_{name}", os.path.join(OUT, "battery"))


def battery_tables(name: str = "report_ext") -> dict:
    """Per family, protocol and candidate: the primary statistic at the family's horizon (50, 60, or 150 after a failed
    validation) pooled over the evaluated policies, how many policies support each sign, and the other statistic."""
    from .battery import DEFAULT_H, FAMILIES, candidates
    sm = _battery(name)
    if not sm:
        return {}
    an = sm["analyses"]
    out = {"tags": sm["tags"], "families": {}, "validation": {}}
    fams = [f for f in FAMILIES if any(f in a["families"] for a in an.values())]
    for fam in fams:
        info = [a["families"][fam] for a in an.values() if fam in a["families"]]
        X = info[0]["primary"]
        Y = "G" if X == "Q" else "Q"
        hmax = max(int(i["horizon"]) for i in info)
        rows = []
        protos = sorted({p for i in info for p in i["table"]})
        for p in protos:
            for c in candidates(fam):
                def vals(stat, h):
                    return [i["table"].get(p, {}).get(c, {}).get(stat, {}).get(str(h)) for i in info]
                main = vals(X, DEFAULT_H)
                means = [r["mean"] for r in main if r]
                pos = sum(1 for r in main if r and r["mean"] - r["ci"] > 0)
                neg = sum(1 for r in main if r and r["mean"] + r["ci"] < 0)
                row = {"protocol": p, "candidate": c, "primary": X, "n_policies": len(means),
                       f"{X}50": _t(means), f"{Y}50": _t([r["mean"] for r in vals(Y, DEFAULT_H) if r]),
                       f"{X}5": _t([r["mean"] for r in vals(X, 5) if r]),
                       f"{X}20": _t([r["mean"] for r in vals(X, 20) if r]),
                       "policies_positive": pos, "policies_negative": neg, "per_policy": means,
                       "states": [r["n"] for r in main if r]}
                if hmax > DEFAULT_H and hmax != 60:
                    row[f"{X}{hmax}"] = _t([r["mean"] for r in vals(X, hmax) if r])
                rows.append(row)
        out["families"][fam] = {"kind": FAMILIES[fam][0], "primary": X, "horizon": hmax, "rows": rows,
                                "extended_on": [t for t, a in an.items() if a["families"].get(fam, {}).get("extended")]}
        out["validation"][fam] = {t: a["validation"].get(fam, {}) for t, a in an.items()}
    return out


def constructed_cells(tags: Sequence[str], hs=(5, 20, 50)) -> dict:
    """Per constructed cell (power gap, level, situation), protocol and candidate: the observed return difference G_h
    pooled over the evaluated policies (t interval over policies), from the raw battery jobs. A cell re-run at 150
    (exception 3) is read from its 150-turn job."""
    from .battery import EXTENDED_H, GRIDS, candidates, load
    acc: Dict[tuple, list] = {}
    for t in tags:
        js = [j for j in load(t) if j["family"] in GRIDS]
        ext = {(j["family"], j["index"], j["protocol"]) for j in js if j["horizon"] == EXTENDED_H}
        for j in js:
            key = (j["family"], j["index"], j["protocol"])
            if key in ext and j["horizon"] != EXTENDED_H:
                continue
            gp = j["GQ"]["policy"]["G"]
            for c in candidates(j["family"]):
                if c not in j["GQ"]:
                    continue
                gc = j["GQ"][c]["G"]
                row = {h: float(np.mean(np.asarray(gc[str(h)]) - np.asarray(gp[str(h)]))) for h in hs if str(h) in gc}
                if EXTENDED_H and str(EXTENDED_H) in gc:
                    row[EXTENDED_H] = float(np.mean(np.asarray(gc[str(EXTENDED_H)]) - np.asarray(gp[str(EXTENDED_H)])))
                acc.setdefault((j["family"], j["index"], j["protocol"], c), []).append(row)
    out = {}
    for (fam, idx, proto, c), rows in acc.items():
        cell = GRIDS[fam][idx]
        hk = sorted({h for r in rows for h in r})
        out.setdefault(fam, []).append({"cell": list(cell) if isinstance(cell, tuple) else cell, "index": idx,
                                        "protocol": proto, "candidate": c, "n_policies": len(rows),
                                        **{f"G{h}": _t([r.get(h) for r in rows]) for h in hk}})
    return out


def extension_comparison(tags: Sequence[str]) -> dict:
    """Exception 3 check: for every (policy, family) re-run at 150 turns, the family-level paired difference of every
    (protocol, candidate) at 50 and at 150 turns ON THE SAME TRAJECTORIES (the 150-turn rollouts share their first 50
    turns with the 50-turn ones: same states, same common random numbers), with the primary statistic; and the
    contradictions (both significant over states, opposite signs)."""
    from .battery import EXTENDED_H, FAMILIES, aggregate, load, primary_statistic, units
    out = {"comparisons": [], "summary": {}}
    for t in tags:
        js = [j for j in load(t) if j["horizon"] == EXTENDED_H]
        for fam in sorted({j["family"] for j in js}):
            X = primary_statistic(fam)
            fj = [j for j in js if j["family"] == fam]
            for p in sorted({j["protocol"] for j in fj}):
                cands = sorted({c for j in fj if j["protocol"] == p for c in j["GQ"] if c != "policy"})
                for c in cands:
                    us = units(fj, fam, p, c, X)
                    a50, a150 = aggregate(us, 50), aggregate(us, EXTENDED_H)
                    if not a50 or not a150:
                        continue
                    sig = lambda a: 0 if abs(a["mean"]) <= a["ci"] else int(np.sign(a["mean"]))
                    out["comparisons"].append({"tag": t, "family": fam, "protocol": p, "candidate": c, "statistic": X,
                                               "at50": a50["mean"], "ci50": a50["ci"], "at150": a150["mean"],
                                               "ci150": a150["ci"], "sig50": sig(a50), "sig150": sig(a150)})
    rows = out["comparisons"]
    if rows:
        d = [r["at150"] - r["at50"] for r in rows]
        out["summary"] = {"comparisons": len(rows), "families_rerun": sorted({(r["tag"], r["family"]) for r in rows}),
                          "mean_change_150_minus_50": _t(d),
                          "slope_150_on_50": float(np.polyfit([r["at50"] for r in rows], [r["at150"] for r in rows], 1)[0])
                          if len(rows) > 2 else float("nan"),
                          "significant_at_both": sum(1 for r in rows if r["sig50"] and r["sig150"]),
                          "contradictions": [r for r in rows if r["sig50"] and r["sig150"] and r["sig50"] != r["sig150"]],
                          "significant_only_at_150": sum(1 for r in rows if r["sig150"] and not r["sig50"]),
                          "significant_only_at_50": sum(1 for r in rows if r["sig50"] and not r["sig150"])}
    return out


def mechanics_summary(tags: Sequence[str]) -> dict:
    """The exact one-turn expectations (battery.mechanics) pooled over policies, per family and candidate, computed
    from the per-state rows and RESTRICTED to states where the candidate's intended first code is legal (elsewhere a
    candidate falls back to the policy's own action and is not a real alternative): the share of states where it is
    legal, the share of those where the policy's MODAL action already is that code, and the mean immediate lead
    difference (candidate - modal action, both against the opponent's modal reply) over those states."""
    from .battery import candidates
    acc: Dict[str, Dict[str, Dict[str, list]]] = {}
    for t in tags:
        p = os.path.join(CKPT, "battery", t, "mechanics.json")
        d = None
        for q in (p, p + ".gz"):
            if os.path.exists(q):
                d = json.load(gzip.open(q, "rt") if q.endswith(".gz") else open(q, encoding="utf-8"))
        if not d:
            continue
        for fam, v in d["families"].items():
            cs = candidates(fam)
            for cand, spec in cs.items():
                intended = int(spec[1]) if spec[0] == "fixed" else None
                legal, modal_is, lead = [], [], []
                for r in v["states"]:
                    code = r["codes"].get(cand)
                    ok = code is not None and (intended is None or code == intended)
                    legal.append(ok)
                    if not ok or cand not in r["mean"] or "policy (modal)" not in r["mean"]:
                        continue
                    modal_is.append(r["codes"]["policy (modal)"] == code)
                    m, b = r["mean"][cand], r["mean"]["policy (modal)"]
                    lead.append((m["earned"][0] - m["earned"][1]) - (b["earned"][0] - b["earned"][1]))
                x = acc.setdefault(fam, {}).setdefault(cand, {"legal": [], "modal": [], "lead": []})
                x["legal"].append(float(np.mean(legal)) if legal else float("nan"))
                x["modal"].append(float(np.mean(modal_is)) if modal_is else float("nan"))
                x["lead"].append(float(np.mean(lead)) if lead else float("nan"))
    return {fam: {c: {"legal_share": _t(v["legal"]), "modal_is_candidate": _t(v["modal"]),
                      "immediate_lead_diff": _t(v["lead"])} for c, v in cands.items()} for fam, cands in acc.items()}


def validation_summary(bt: dict) -> dict:
    rows = {}
    for fam, per in bt.get("validation", {}).items():
        verdicts = [v.get("verdict", "missing") for v in per.values()]
        rows[fam] = {"pass": sum(v == "pass" for v in verdicts), "fail": sum(str(v).startswith("fail") for v in verdicts),
                     "insufficient": sum(v == "insufficient" for v in verdicts),
                     "bias": _t([v.get("bias") for v in per.values() if v.get("bias") is not None]),
                     "slope": _t([v.get("slope") for v in per.values() if v.get("slope") is not None]),
                     "contradictions": int(sum(len(v.get("contradictions", [])) for v in per.values()))}
    return rows


# ---------------------------------------------------------------------------------------------- attackers, traces
def attackers_summary() -> dict:
    d = load_json("attackers_ext", os.path.join(OUT, "attackers"))
    if not d:
        return {}
    rows = {}
    for s, r in d["results"].items():
        tr = {k: v["estimate"] for k, v in r["transfer"].items()}
        rows[s] = {"W460": r["eval"]["W460"]["estimate"], "W460_ci": r["eval"]["W460"]["ci95"],
                   "W1000": r["eval"]["W1000"]["estimate"], "W1000_ci": r["eval"]["W1000"]["ci95"],
                   "seat_A": r["eval"]["W1000"]["seat_A"], "seat_B": r["eval"]["W1000"]["seat_B"],
                   "transfer_other_ext": _t([v for k, v in tr.items() if ":ext:" in k]),
                   "vs_own_8192M_final": next((v for k, v in tr.items() if ":full:" in k), None),
                   "train_minutes": r["train_elapsed_s"] / 60, "tps": r["transitions_per_s"],
                   "entropy_first_last": r["entropy_first_last"], "curve": r["curve"]}
    return {"per_seed": rows, "W1000": _t([v["W1000"] for v in rows.values()]),
            "W460": _t([v["W460"] for v in rows.values()]), "budget_note": d.get("budget_note"),
            "significant": sum(1 for v in rows.values() if v["W1000_ci"][0] > 0)}


def attackers_behaviour_summary() -> dict:
    """The attackers against their targets (attackers.behaviour): mean masses and game rates of the attacker and of the
    target in those games, over the six pairs, next to the target's own self-play profile."""
    d = load_json("attackers_behaviour_ext", os.path.join(OUT, "attackers"))
    sm = load_json("summary", os.path.join(OUT, "ext"))
    if not d:
        return {}
    keys = ("LH", "TN", "SL", "SH", "Ward", "Recall", "beh_Hold", "beh_Yield", "beh_Trade", "beh_Safe", "Pressure")
    rates = ("trade_attempts_per_1000", "trades_connected_per_1000", "deaths_per_1000", "recalls_per_1000",
             "wards_per_1000", "lead_per_100_turns")
    out = {}
    for who in ("attacker", "target"):
        R = [d["records"].get(f"{who}|final:ext:{s}") for s in SEEDS]
        R = [r for r in R if r]
        out[who] = {"mass": {k: _t([r["conditional"]["all present decisions"]["mass"][k] for r in R]) for k in keys},
                    "rates": {k: _t([r["summary"][k][0] for r in R]) for k in rates},
                    "freezes_per_1000": _t([r["freezes"]["established_per_1000_turns"][0] for r in R]),
                    "freeze_denial": _t([r["freezes"]["established"].get("useful_denial_share") for r in R])}
    if sm:
        out["target_selfplay"] = {"mass": {k: _t([sm["behavior"]["overall"][f"{s}|final"][k] for s in SEEDS]) for k in keys}}
    return out


def traces_summary() -> dict:
    d = load_json("traces", os.path.join(OUT, "traces"))
    if not d:
        return {}
    return {"all_verified": d["all_verified"],
            "selected": {k: {kk: v.get(kk) for kk in ("found", "path", "label", "seed", "side", "t0", "t1")}
                         for k, v in d["selected"].items()}}


# ---------------------------------------------------------------------------------------------- figures
def _style(ax):
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=8)


def _save(fig, name: str) -> str:
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_budget_curves(bs: dict) -> Optional[str]:
    """Scripted-suite mean against the fraction of each run's own schedule, both budgets (seed mean, t band)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "full" not in bs or "ext" not in bs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key, title in ((axes[0], "curve", "scripted-suite mean (W = 460; untrained −7.0 not shown)"),
                           (axes[1], "curve_ood", "M4.4b aggro network, OOD (W = 460; untrained −10.1 not shown)")):
        for ph, col, lab, budget in (("full", ORANGE, "8.192M runs", 8_192_000), ("ext", BLUE, "16.384M runs", 16_384_000)):
            pts = sorted((int(k), v) for k, v in bs[ph][key].items() if k != "initial")    # untrained: off-scale
            x = np.array([p for p, _ in pts]) / 1e6
            m = np.array([v[0] for _, v in pts])
            h = np.array([v[1] for _, v in pts])
            ax.fill_between(x, m - h, m + h, color=col, alpha=0.15, linewidth=0)
            ax.plot(x, m, color=col, linewidth=2, label=lab)
        ax.axhline(0, color=GREY, linewidth=1)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("agent transitions (millions)", fontsize=8, color=MUTED)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, "budget_curves.png")


def fig_battery(bt: dict, name: str) -> Optional[str]:
    """Forest plot: per family, the primary statistic at 50 turns of every (protocol, candidate), pooled over
    policies with its t interval; zero = the policy's own choice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = bt.get("families", {})
    if not fams:
        return None
    n = len(fams)
    cols = 3
    rows_ = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_, cols, figsize=(5.0 * cols, 3.6 * rows_))
    for ax, (fam, v) in zip(np.ravel(axes), fams.items()):
        X = v["primary"]
        items = [(f"{r['protocol']}  {r['candidate']}", r[f"{X}50"]) for r in v["rows"] if np.isfinite(r[f"{X}50"][0])]
        y = np.arange(len(items))[::-1]
        for yi, (lab, (m, h)) in zip(y, items):
            col = INK if (np.isfinite(h) and (m - h > 0 or m + h < 0)) else GREY
            ax.plot([m - h, m + h] if np.isfinite(h) else [m, m], [yi, yi], color=col, linewidth=1.5)
            ax.plot([m], [yi], "o", color=BLUE if m >= 0 else ORANGE, markersize=4)
        ax.axvline(0, color=GREY, linewidth=1)
        ax.set_yticks(y, [lab for lab, _ in items], fontsize=6)
        ax.set_title(f"{fam} ({X}50)", fontsize=9, color=INK)
        _style(ax)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.supxlabel("candidate − policy's own choice (resources); dark interval = excludes 0", fontsize=9, color=MUTED)
    fig.tight_layout()
    return _save(fig, f"battery_{name}.png")


# ---------------------------------------------------------------------------------------------- report
def report() -> str:
    bt_train, bb, bs = budget_training(), budget_behaviour(), budget_strength()
    bt = battery_tables("report_ext")
    rep = battery_tables("report_full_replication")
    mech = mechanics_summary([f"s{s}_ext" for s in SEEDS])
    val = validation_summary(bt)
    att, trc = attackers_summary(), traces_summary()
    attb = attackers_behaviour_summary()
    try:
        cells = constructed_cells([f"s{s}_ext" for s in SEEDS])
    except Exception as ex:                             # pragma: no cover - raw jobs missing
        cells = {"error": repr(ex)}
    try:
        extc = extension_comparison([f"s{s}_ext" for s in SEEDS] + [f"s{s}_full" for s in SEEDS])
    except Exception as ex:                             # pragma: no cover
        extc = {"error": repr(ex)}
    data = {"extension_comparison": extc, "budget_training": bt_train, "budget_behaviour": bb, "budget_strength": bs, "battery": bt,
            "battery_replication_8192M": rep, "mechanics": mech, "validation": val, "attackers": att, "traces": trc,
            "constructed_cells": cells, "attackers_behaviour": attb}
    save_json("summary", data, FOUT)
    figs = [fig_budget_curves(bs), fig_battery(bt, "ext"), fig_battery(rep, "replication_8192M")]
    S = ["# M4.5c — final research report (generated tables)\n",
         "Generated by `python -m lanegame.experiments.rl.m45c campaign_report` from results/m45c/{full,ext,battery,"
         "attackers,traces}/. Interpretation: docs/m4/m4_5/m4_5c_research_report.md.\n"]
    # 1 training of both budgets
    S.append("## 1. Training (both budgets)\n")
    for ph, lab in (("full", "8.192M"), ("ext", "16.384M")):
        r = bt_train.get(ph)
        if not r or "runs" not in r:
            continue
        rows = [[n, f"{x['transitions']:,}", x["updates"], f"{x['minutes']:.1f}", f"{x['tps']:,.0f}",
                 f"{(x['peak_mb'] or 0):.0f}", x["digest"], x["initial_digest"]] for n, x in r["runs"].items()]
        S.append(f"{lab} runs (checks: {r['checks']['n']}, failed {r['checks']['failed'] or 'none'}):\n")
        S.append(table(["run", "transitions", "updates", "minutes", "transitions/s", "peak MB", "final digest",
                        "initial digest"], rows))
        S.append("")
    keys = [k for k in bt_train.get("ext", {}) if isinstance(bt_train["ext"].get(k), dict) and "last_tenth" in bt_train["ext"][k]]
    if keys and "full" in bt_train:
        rows = [[k, _pm(*bt_train["full"][k]["last_tenth"], d=3), _pm(*bt_train["ext"][k]["last_tenth"], d=3)] for k in keys]
        S.append("Last tenth of training, seed mean ± t interval:\n")
        S.append(table(["quantity", "8.192M runs", "16.384M runs"], rows))
    # 2 budget comparison
    cp = bs.get("comprehensive", {})
    S.append("\n## 2. Budget comparison: 16.384M vs 8.192M finals\n")
    if cp.get("vs_main"):
        m = cp["vs_main"]
        S.append(f"16.384M finals vs 8.192M finals (W = 1000, 64 paired seeds): same seed {_pm(*m['same_seed_mean'])}, "
                 f"cross-seed {_pm(*m['cross_seed_mean'])}; pairs above 0: {m['pairs_significant_positive']} of "
                 f"{m['pairs']}, below 0: {m['pairs_significant_negative']}.\n")
    if cp.get("anneal"):
        m = cp["anneal"]
        S.append(f"Annealing test, 8.192M finals (annealed) vs the extension's 8.192M midpoints (learning rate 50%): "
                 f"same seed {_pm(*m['same_seed_mean'])}, cross-seed {_pm(*m['cross_seed_mean'])}; above 0: "
                 f"{m['pairs_significant_positive']} of {m['pairs']}, below 0: {m['pairs_significant_negative']}.\n")
    hl = cp.get("headline", {})
    if hl:
        rows = [[lab, _pm(*v["scripted"]), _pm(*v.get("ood", [float("nan")] * 2))] for lab, v in hl.items()]
        S.append("Headline against the fixed suite (W = 1000, 64 paired seeds, both seats):\n")
        S.append(table(["policies", "scripted mean", "M4.4b (OOD)"], rows))
    for ph, lab in (("full", "8.192M"), ("ext", "16.384M")):
        if ph in bs:
            h = bs[ph]["h2h"]
            S.append(f"\n{lab}: final vs own midpoint {_pm(*h['final_vs_mid'])}, vs own 3/4 point {_pm(*h['final_vs_q3'])}; "
                     f"cross-play max |value| {bs[ph]['cross_max']:.2f} ({bs[ph]['cross_sig']} of 15 significant); "
                     f"seeds with a significant adjacent regression: {bs[ph]['cycle'] or 'none'}; extension criterion "
                     f"met: {bs[ph]['rules']['extension']['criterion_met']} (not plateaued: "
                     f"{bs[ph]['rules']['extension']['not_plateaued_count']} of 6).")
    if bb:
        rows = [[lab, _pm(*v["full"], d=3), _pm(*v["ext"], d=3), _pm(*v["diff"], d=3)] for lab, v in bb.items()
                if not lab.startswith("_")]
        S.append("\n\nBehaviour of the finals in self-play (seed mean ± t; difference paired by seed, 16.384M − 8.192M):\n")
        S.append(table(["readout", "8.192M finals", "16.384M finals", "difference"], rows))
        g = bb.get("_power_gap", {})
        if g:
            rows = [[k, _pm(*g["full"][k], d=3), _pm(*g["ext"][k], d=3)] for k in g["full"]]
            S.append("\nInitialized ×1.76 power gap (self-play from the neutral start):\n")
            S.append(table(["readout", "8.192M finals", "16.384M finals"], rows))
    # 3 battery
    for lab, b in (("primary policy set: the six 16.384M finals", bt), ("replication on the 8.192M finals", rep)):
        if not b:
            continue
        S.append(f"\n## 3. Strategic battery — {lab}\n")
        for fam, v in b["families"].items():
            X = v["primary"]
            Y = "G" if X == "Q" else "Q"
            hcol = f"{X}{v['horizon']}" if v["horizon"] > 60 else None
            rows = []
            for r in v["rows"]:
                rows.append([r["protocol"], r["candidate"], _pm(*r[f"{X}5"]), _pm(*r[f"{X}20"]), _pm(*r[f"{X}50"])]
                            + ([_pm(*r[hcol])] if hcol and hcol in r else [])
                            + [_pm(*r[f"{Y}50"]), f"{r['policies_positive']}+ / {r['policies_negative']}−",
                               int(np.mean(r["states"])) if r["states"] else 0])
            S.append(f"### {fam} ({v['kind']}; primary {X}; horizon {v['horizon']}"
                     f"{'; re-run at 150 after failed validation on ' + ', '.join(v['extended_on']) if v['extended_on'] else ''})\n")
            S.append(table(["protocol", "candidate", f"{X}5", f"{X}20", f"{X}50"] + ([hcol] if hcol else [])
                           + [f"{Y}50", "policies + / −", "states"], rows))
            S.append("")
    if cells and "error" not in cells:
        S.append("\n## 3b. Constructed cells (G, observed return; pooled over the six 16.384M finals)\n")
        for fam, rows_ in cells.items():
            hk = sorted({k for r in rows_ for k in r if k.startswith("G")}, key=lambda k: int(k[1:]))
            rows = [[str(r["cell"]), r["protocol"], r["candidate"]] + [_pm(*r.get(k, [float("nan")] * 2)) for k in hk]
                    for r in sorted(rows_, key=lambda r: (r["index"], r["protocol"], r["candidate"]))]
            S.append(f"### {fam}\n")
            S.append(table(["cell", "protocol", "candidate", *hk], rows))
            S.append("")
    if val:
        rows = [[f, v["pass"], v["fail"], v["insufficient"], _pm(*v["bias"]), _pm(*v["slope"]), v["contradictions"]]
                for f, v in val.items()]
        S.append("\n## 4. Horizon validation (§6.3 rule per policy and family; 300-turn units)\n")
        S.append(table(["family", "pass", "fail", "insufficient", "bias (mean over policies)", "slope", "contradictions"],
                       rows))
    es = (extc or {}).get("summary") or {}
    if es:
        S.append("\n## 4b. Exception 3: the families re-run at 150 turns, 50 vs 150 on the same trajectories\n")
        S.append(f"{es['comparisons']} (policy, family, protocol, candidate) comparisons in "
                 f"{len(es['families_rerun'])} re-run (policy, family) pairs. Mean change 150 − 50: "
                 f"{_pm(*es['mean_change_150_minus_50'])}; slope of the 150-turn value on the 50-turn value "
                 f"{es['slope_150_on_50']:.2f}; significant at both horizons {es['significant_at_both']}, of which "
                 f"contradicting {len(es['contradictions'])}; significant only at 150: {es['significant_only_at_150']}; "
                 f"only at 50: {es['significant_only_at_50']}.\n")
        if es["contradictions"]:
            rows = [[r["tag"], r["family"], r["protocol"], r["candidate"], _pm(r["at50"], r["ci50"]),
                     _pm(r["at150"], r["ci150"])] for r in es["contradictions"]]
            S.append(table(["policy", "family", "protocol", "candidate", "50", "150"], rows))
    if mech:
        rows = []
        for fam, cands in mech.items():
            for c, v in cands.items():
                rows.append([fam, c, _f(v["legal_share"][0], 2, False), _f(v["modal_is_candidate"][0], 2, False),
                             _pm(*v["immediate_lead_diff"])])
        S.append("\n## 5. Exact one-turn mechanics, where the candidate's action is legal (the share of states); how "
                 "often the policy's MODAL action already is it; the exact immediate lead of the candidate minus the "
                 "modal action (opponent: its modal reply; chance branches and ganks enumerated)\n")
        S.append(table(["family", "candidate", "legal share", "modal = candidate", "immediate lead difference"], rows))
    if att:
        rows = [[f"s{s}", _f(v["W460"]), _f(v["W1000"]), f"[{v['W1000_ci'][0]:+.2f}, {v['W1000_ci'][1]:+.2f}]",
                 _f(v["seat_A"]), _f(v["seat_B"]), _pm(*v["transfer_other_ext"]), _f(v["vs_own_8192M_final"]),
                 f"{v['train_minutes']:.0f}"] for s, v in att["per_seed"].items()]
        S.append("\n## 6. Best-response attackers (exploitability lower bounds)\n")
        S.append(f"{att.get('budget_note')}\n")
        S.append(table(["target", "vs target W460", "vs target W1000", "95% interval", "seat A", "seat B",
                        "vs other 16.384M finals", "vs same-seed 8.192M final", "train min"], rows))
        S.append(f"\nMean vs target (W = 1000): {_pm(*att['W1000'])}; significant on {att['significant']} of 6.\n")
        if attb:
            ks = list(attb["attacker"]["mass"])
            rows = [[k, _f(attb["attacker"]["mass"][k][0], 3, False), _f(attb["target"]["mass"][k][0], 3, False),
                     _f(attb.get("target_selfplay", {}).get("mass", {}).get(k, [float("nan")])[0], 3, False)] for k in ks]
            rows += [[k, _f(attb["attacker"]["rates"][k][0], 1), _f(attb["target"]["rates"][k][0], 1), ""]
                     for k in attb["attacker"]["rates"]]
            rows += [["freezes per 1000", _f(attb["attacker"]["freezes_per_1000"][0], 1, False),
                      _f(attb["target"]["freezes_per_1000"][0], 1, False), ""],
                     ["freeze denial share", _f(attb["attacker"]["freeze_denial"][0], 2, False),
                      _f(attb["target"]["freeze_denial"][0], 2, False), ""]]
            S.append("What the attackers do against their targets (8 games x 500 turns, both seats; means over the six "
                     "pairs) and the targets' own self-play profile:\n")
            S.append(table(["readout", "attacker", "target (vs attacker)", "target in self-play"], rows))
    if trc:
        rows = [[k, v.get("found"), (v.get("label") or "")[:90], (v.get("path") or "").replace(os.sep, "/")]
                for k, v in trc["selected"].items() if v.get("found")]
        S.append(f"\n## 7. Representative trajectories (viewer recordings; all verified by replay: {trc['all_verified']})\n")
        S.append(table(["policy | situation", "candidates", "label", "recording"], rows))
    S.append("\nFigures: " + ", ".join(f"`{os.path.basename(p)}`" for p in figs if p) + " (results/m45c/final/figures/).\n")
    text = "\n".join(S)
    write_md("report", text, FOUT)
    return text


__all__ = ["report", "attackers_behaviour_summary", "constructed_cells", "extension_comparison", "budget_training", "budget_behaviour", "budget_strength", "battery_tables", "mechanics_summary",
           "validation_summary", "attackers_summary", "traces_summary"]
