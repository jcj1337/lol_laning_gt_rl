"""Generated tables of M4.6c (results/m46c/report.md) and the pre-registered scorecard (pre-registration §5), computed
from results/m46c/analysis.json. The narrative report is docs/m4/m4_6/m4_6c_report.md."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import common as C
from .archive import STAGES

ARM_LABEL = {"baseline": "baseline", "C1": "C1"}


def f(st, d: int = 2) -> str:
    return C.fci(st, d) if isinstance(st, dict) else "n/a"


def sig(st) -> str:
    return {1: " (+)", -1: " (−)", 0: ""}[C.excludes0(st)] if isinstance(st, dict) else ""


def table(head: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- scorecard
def scorecard(A: dict, sfx: str = "") -> dict:
    """The pre-registered scorecard; sfx "_ext" = the 8.192M extension's families (amendment of 2026-10-15)."""
    pre = "ext/" if sfx else ""
    pr = A.get("premium" + sfx, {}).get("diff", {})
    out = {}
    # ---- A
    p = pr.get("adapted|mid|+8|G50")
    if p:
        met = (p["mean"] - p["ci"] > 0) and p["ratio_of_means"] >= 1.25 and p["mean"] >= 0.08 and p["pos"] >= 5
        incon = (not met) and p["ratio_of_means"] >= 1.25 and p["mean"] - p["ci"] <= 0 < p["mean"]
        sup = {k: pr.get(f"adapted|{k}|G50") for k in ("mid|+16", "late|+8", "late|+16")}
        pv = {k: v["p"] for k, v in sup.items() if v and np.isfinite(v.get("p", np.nan))}
        from ..m46a.common import holm
        hs = holm(pv) if pv else {}
        contra = [k for k, v in sup.items() if v and hs.get(k) and v["mean"] < 0]
        out["A"] = {"outcome": "met" if met and not contra else ("inconclusive" if incon else "not met"),
                    "primary": p, "supporting": sup, "supporting_holm_significant": hs, "contradictions": contra}
    # ---- B
    flags = []
    e = pr.get("adapted|early|+8|G20")
    if e and e["ratio_of_means"] > 1.5 and e["mean"] - e["ci"] > 0:
        flags.append("early +8 premium at G20 more than 1.5x the baseline's")
    nat = A.get("natural" + sfx, {})
    arms = nat.get("diff", {}).get("arms (C1:s in C1 - baseline:s in baseline)", {})
    pc = nat.get("per_combo", {})
    sl = arms.get("slope_150_on_50")
    c1s = pc.get(f"{pre}C1|C1", {}).get("slope_150_on_50", {}).get("mean", np.nan)
    if sl and sl["mean"] > 0.3 and C.excludes0(sl) > 0 and c1s > 1.3:
        flags.append("early natural leads grow (slope 150 on 50)")
    sh = pc.get(f"{pre}C1|C1", {}).get("attack_share_1.25", {}).get("mean", 0.0)
    if sh > 0.05:
        flags.append("attack ratio >= 1.25 on more than 5% of present turns")
    dc, db = (pc.get(f"{pre}C1|C1", {}).get("decisive150_share", {}).get("mean", 0.0),
              pc.get(f"{pre}baseline|baseline", {}).get("decisive150_share", {}).get("mean", 0.0))
    if dc > 0.10 and dc > 2 * db:
        flags.append("decisive-game share at turn 150 > 10% and > 2x the baseline's")
    out["B"] = {"flags": flags, "outcome": "flag" if flags else "no flag",
                "early_G20": e, "slope_150_on_50_diff": sl, "slope_C1": c1s, "attack_share_1.25_C1": sh,
                "decisive150": [dc, db]}
    # ---- C
    from .analysis import NAT_KEYS_ABS, NAT_KEYS_REL
    issues, notes = [], []
    for k in NAT_KEYS_REL:
        st = arms.get(k)
        if not st or not np.isfinite(st.get("rel", np.nan)):
            continue
        if abs(st["rel"]) > 0.25 and C.excludes0(st):
            issues.append(f"{k} {st['rel']:+.0%}")
        elif abs(st["rel"]) > 0.15:
            notes.append(f"{k} {st['rel']:+.0%}")
    for k in NAT_KEYS_ABS:
        st = arms.get(k)
        if st and abs(st["mean"]) > 0.10 and C.excludes0(st):
            issues.append(f"{k} {st['mean']:+.3f}")
    out["C"] = {"issues": issues, "notes": notes, "outcome": "issue" if issues else ("note" if notes else "pass")}
    # ---- D
    bat = A.get("battery" + sfx, {})
    dflags = []
    for grp in list(STAGES) + ["all"]:
        for d in (8, 16):
            for c in ("hold", "safe", "trade", "reset"):
                a = bat.get(f"C1|weak|{grp}|+{d}|{c}-policy")
                b = bat.get(f"baseline|weak|{grp}|+{d}|{c}-policy")
                if a and a["mean"] > 0.15 and C.excludes0(a) > 0 and not (b and C.excludes0(b) > 0):
                    dflags.append(f"battery {grp} +{d}: {c} beats the C1 trailer policy ({f(a)})")
            de = bat.get(f"C1-baseline|weak|{grp}|+{d}|policy|deaths_A")
            if de and de.get("mean_baseline", 0) > 0 and de["mean_C1"] >= 1.5 * de["mean_baseline"]:
                for c in ("safe", "reset"):
                    a = bat.get(f"C1|weak|{grp}|+{d}|{c}-policy")
                    if a and C.excludes0(a) > 0:
                        dflags.append(f"contest-and-die {grp} +{d}: deaths x{de['mean_C1'] / de['mean_baseline']:.2f}, "
                                      f"{c} beats the policy")
    cf = A.get("natural_cf" + sfx, {})
    for c in ("hold1", "safe1", "trade1", "recall1"):
        a, b = cf.get(f"C1|trailer|{c}-policy"), cf.get(f"baseline|trailer|{c}-policy")
        if a and a["mean"] > 0.15 and C.excludes0(a) > 0 and not (b and C.excludes0(b) > 0):
            dflags.append(f"natural trailer states: {c} beats the C1 policy ({f(a)})")
    out["D"] = {"flags": dflags, "outcome": "flag" if dflags else "no flag"}
    # ---- E
    tr = A.get("train_check" + sfx, {})
    h = (A.get("h2h", {}).get("ext", {}).get("extstart|baseline") if sfx else
         A.get("h2h", {}).get("start", {}).get("baseline"))
    eiss = []
    if tr and not tr.get("all_passed", False):
        eiss.append("training checks failed: " + ", ".join(tr.get("failed", [])[:5]))
    if h and h["mean"] < -0.2:
        eiss.append(f"baseline continuation worse than its start ({f(h)})")
    out["E"] = {"issues": eiss, "outcome": "issue" if eiss else "pass", "baseline_vs_start": h}
    # ---- recommendation
    a = out.get("A", {}).get("outcome")
    n_bcd = sum(out[k]["outcome"] in ("flag", "issue") for k in ("B", "C", "D"))
    if out["E"]["outcome"] == "issue":
        rec = "3: resolve the health / continuation issue first"
    elif a == "met" and n_bcd == 0:
        rec = "1: C1 ready to become the candidate environment"
    elif a == "not met" or (a == "met" and n_bcd >= 2):
        rec = "2: retain the baseline"
    else:
        rec = "3: resolve one specifically identified issue first"
    out["recommendation_rule"] = rec
    return out


# ---------------------------------------------------------------------------------------------- tables
def premium_tables(A: dict, sfx: str = "") -> str:
    P = A.get("premium" + sfx, {})
    pooled, diff = P.get("pooled", {}), P.get("diff", {})
    rows = []
    for st in STAGES:
        for d in (8, 16):
            for h in (20, 50):
                b, c = pooled.get(f"self|baseline|{st}|{d}|{h}"), pooled.get(f"self|C1|{st}|{d}|{h}")
                df = diff.get(f"adapted|{st}|+{d}|G{h}")
                if not df:
                    continue
                rows.append([st, f"+{d}", f"G{h}", f(b), f(c), f(df) + sig(df), f"×{df['ratio_of_means']:.2f}",
                             f"{df['pos']}+ / {df['neg']}−"])
    s = "### Lead premium after adaptation (each arm's own policies in its own environment)\n\n"
    s += table(["stage", "lead", "h", "baseline", "C1", "C1 − baseline", "ratio", "seeds"], rows)
    rows = []
    for st in STAGES:
        for d in (8, 16):
            pre = diff.get(f"pre_adaptation|{st}|+{d}|G50")
            ad = diff.get(f"adaptation_within_C1|{st}|+{d}|G50")
            co = diff.get(f"continuation_within_baseline|{st}|+{d}|G50")
            sb = pooled.get(f"start|baseline|{st}|{d}|50")
            sc = pooled.get(f"start|C1|{st}|{d}|50")
            j3 = pooled.get(f"j3|C1|{st}|{d}|50")
            rows.append([st, f"+{d}", f(sb), f(sc), f(pre) + sig(pre), f(j3), f(ad) + sig(ad), f(co) + sig(co)])
    s += "\n\n### Decomposition (G50)\n\n" + table(
        ["stage", "lead", "start, baseline env", "start, C1 env (cross-env)", "pre-adaptation C1 − baseline",
         "baseline-continued in C1 env (cross-env)", "C1-trained − baseline-continued, C1 env",
         "baseline continuation − start"], rows)
    rows = []
    for st in STAGES:
        for d in (8, 16):
            for env in C.ARMS:
                cv, dv = pooled.get(f"conv|{env}|{st}|{d}|50"), pooled.get(f"def|{env}|{st}|{d}|50")
                g = pooled.get(f"general|{env}|{st}|0|50")
                rows.append([st, f"+{d}", env, f(cv) + sig(cv), f(dv) + sig(dv), f(g) + sig(g)])
    s += "\n\n### Conversion and defence learned (J2: arm policy − start, against start:s; G50; lead-specific)\n\n"
    s += table(["stage", "lead", "arm", "conversion gain", "defence gain", "general (equal cell)"], rows)
    return s


def training_tables(A: dict, sfx: str = "") -> str:
    T = A.get("training" + sfx, {})
    rows = []
    for k, v in sorted(T.items()):
        if not isinstance(v, dict) or "status" not in v:
            continue
        st = v.get("starts") or {}
        rows.append([k, v["status"], f"{v['elapsed_min']:.1f}", f"{v['tps']:.0f}",
                     f"{v['entropy_first10']:.2f} → {v['entropy_last10']:.2f}", f"{v['kl_mean']:.4f} / {v['kl_max']:.4f}",
                     f"{v['clipfrac_mean']:.3f}", f"{v['grad_max']:.2f}", f"{v['ev_first10']:.2f} → {v['ev_last10']:.2f}",
                     f"{v['shove_first10']:.3f} → {v['shove_last10']:.3f}",
                     f"{st.get('archive_transition_share_finished', float('nan')):.3f}"])
    s = "## Training health\n\n" + table(["run", "status", "min", "transitions/s", "entropy", "KL mean / max", "clipfrac",
                                          "grad max", "explained var.", "Shove share", "archive share"], rows)
    if sfx:
        return s
    H = A.get("h2h", {})
    from .evaluate import CURVE_POINTS
    rows = []
    for n in CURVE_POINTS + (C.BUDGET,):
        rows.append([f"{n / 1e6:.3f}M"] + [f(H.get("curve", {}).get(f"{a}|{n}")) + sig(H.get("curve", {}).get(f"{a}|{n}"))
                                           for a in C.ARMS])
    s += "\n\n### Learning curves: snapshot against its start:s, own environment, from s0 (W 460, 32 seeds)\n\n"
    s += table(["transitions", "baseline arm (baseline env)", "C1 arm (C1 env; start:s cross-env)"], rows)
    return s


def h2h_tables(A: dict) -> str:
    H = A.get("h2h", {})
    rows = []
    for arm in C.ARMS:
        rows.append([arm, f(H.get("start", {}).get(arm)) + sig(H.get("start", {}).get(arm)),
                     f(H.get("others", {}).get(arm)) + sig(H.get("others", {}).get(arm)),
                     f(H.get("suite", {}).get(f"{arm}|final")), f(H.get("suite", {}).get(f"{arm}|start")),
                     f(H.get("suite", {}).get(f"{arm}|final-start")) + sig(H.get("suite", {}).get(f"{arm}|final-start")),
                     f"{H.get('rr', {}).get(arm, {}).get('mean_abs', float('nan')):.2f} "
                     f"({H.get('rr', {}).get(arm, {}).get('sig_pairs', 0)} of {H.get('rr', {}).get(arm, {}).get('n_pairs', 0)} sig.)"])
    s = "## Robustness (from s0, own environment)\n\n" + table(
        ["arm", "final − start:s (W 1000)", "final vs other starts (W 460)", "vs 9 fixed references: final",
         "start:s (same env)", "final − start", "round robin \\|value\\| mean"], rows)
    rows = [[env, f(H.get("cross", {}).get(env)) + sig(H.get("cross", {}).get(env))] for env in C.ARMS]
    s += "\n\n### C1:s against baseline:s (W 460, 32 seeds; value to C1:s)\n\n" + table(["environment", "C1:s − baseline:s"], rows)
    return s


def battery_tables(A: dict, sfx: str = "") -> str:
    B = A.get("battery" + sfx, {})
    rows = []
    for grp in list(STAGES) + ["all"]:
        for d in (8, 16):
            for arm in C.ARMS:
                base = f"{arm}|weak|{grp}|+{d}"
                rows.append([grp, f"+{d}", arm, f(B.get(f"{base}|abs_policy")), *[f(B.get(f"{base}|{c}-policy")) +
                             sig(B.get(f"{base}|{c}-policy")) for c in ("hold", "safe", "trade", "reset")],
                             f(B.get(f"{base}|policy|deaths_A")), f(B.get(f"{base}|policy|away_A"))])
    s = "## Trailer (weak focal, seat A trails by +d): policy and forced 10-turn responses (G50; candidate − policy)\n\n"
    s += table(["stage", "lead", "arm", "policy G50", "Hold only", "Safe only", "Trade only", "reset",
                "policy deaths /50", "policy turns away /50"], rows)
    rows = []
    for grp in list(STAGES) + ["all"]:
        for d in (8, 16):
            for arm in C.ARMS:
                base = f"{arm}|strong|{grp}|+{d}"
                rows.append([grp, f"+{d}", arm, f(B.get(f"{base}|abs_policy")),
                             f(B.get(f"{base}|threaten-policy")) + sig(B.get(f"{base}|threaten-policy")),
                             f(B.get(f"{base}|hold-policy")) + sig(B.get(f"{base}|hold-policy")),
                             f(B.get(f"{base}|policy|deaths_A")), f(B.get(f"{base}|policy|deaths_B"))])
    s += "\n\n## Leader (strong focal): policy and forced 10-turn plans (G50)\n\n" + table(
        ["stage", "lead", "arm", "policy G50", "threaten − policy", "Hold only − policy", "leader deaths /50",
         "trailer deaths /50"], rows)
    return s


def cf_tables(A: dict, sfx: str = "") -> str:
    X = A.get("natural_cf" + sfx, {})
    rows = []
    for role, cands in (("trailer", ("hold1", "safe1", "trade1", "recall1")), ("leader", ("trade1", "hold1", "safe1"))):
        for arm in C.ARMS:
            k = f"{arm}|{role}"
            sel = X.get(f"selectivity|{k}", {})
            rows.append([role, arm, f(X.get(f"{k}|abs_policy")),
                         " / ".join(f"{c}: {f(X.get(f'{k}|{c}-policy'))}{sig(X.get(f'{k}|{c}-policy'))}" for c in cands),
                         f(X.get(f"{k}|regret_bound")),
                         f"T {sel.get('mean_Ptrade', float('nan')):.2f} S {sel.get('mean_Psafe', float('nan')):.2f} "
                         f"H {sel.get('mean_PHold', float('nan')):.2f} R {sel.get('mean_PRecall', float('nan')):.2f}",
                         f"{sel.get('corr_Ptrade_vs_trade-hold', float('nan')):+.2f} / "
                         f"{sel.get('corr_Psafe_vs_safe-hold', float('nan')):+.2f}"])
    return "## Natural lead states: one-turn behaviour-only alternatives (G50, alternative − policy)\n\n" + table(
        ["focal", "arm", "policy G50", "alternatives", "ex-post regret bound", "policy masses (Trade, Safe, Hold, Recall)",
         "corr(P, value): Trade / Safe"], rows)


def freeze_tables(A: dict) -> str:
    Z = A.get("freeze", {})
    rows = []
    for L in ("all", 4, 10, 15):
        for arm in C.ARMS:
            b = f"{arm}|breaker|L{L}|all"
            fz = f"{arm}|freezer|L{L}|all"
            rows.append([L, arm, f(Z.get(f"{b}|abs_policy")),
                         *[f(Z.get(f"{b}|{c}-policy")) + sig(Z.get(f"{b}|{c}-policy")) for c in ("break", "walk_up",
                                                                                                 "step_back", "reset")],
                         f(Z.get(f"{fz}|abs_policy")),
                         *[f(Z.get(f"{fz}|{c}-policy")) + sig(Z.get(f"{fz}|{c}-policy")) for c in ("keep", "guard")],
                         f(Z.get(f"{fz}|policy|held_A@20"), 1), f(Z.get(f"{b}|policy|deaths_A@20"))])
    s = "## Freeze cells (G50, candidate − policy)\n\n" + table(
        ["level", "arm", "breaker policy", "break", "walk up", "step back", "reset", "freezer policy", "keep", "guard",
         "freeze held turns /20 (freezer policy)", "breaker deaths /20"], rows)
    rows = []
    for arm in C.ARMS:
        rows.append([arm, f(Z.get(f"{arm}|natbreaker|abs_policy")),
                     *[f(Z.get(f"{arm}|natbreaker|{c}-policy")) + sig(Z.get(f"{arm}|natbreaker|{c}-policy"))
                       for c in ("break1", "walk1", "back1")],
                     f(Z.get(f"{arm}|natfreezer|abs_policy")),
                     *[f(Z.get(f"{arm}|natfreezer|{c}-policy")) + sig(Z.get(f"{arm}|natfreezer|{c}-policy"))
                       for c in ("keep1", "guard1")], f(Z.get(f"{arm}|natfreezer|policy|held_A@20"), 1)])
    s += "\n\n### Natural freeze states (one-turn alternatives, G50)\n\n" + table(
        ["arm", "breaker policy", "break", "walk up", "step back", "freezer policy", "keep", "guard",
         "held turns /20"], rows)
    return s


def specialist_tables(A: dict) -> str:
    S = A.get("specialists", {})
    rows = []
    for env, pol in (("baseline", "start"), ("baseline", "baseline"), ("C1", "start"), ("C1", "C1")):
        k = f"{env}|{pol}"
        rows.append([env, f"{pol}:s" + (" (cross-env)" if env == "C1" and pol == "start" else ""),
                     f(S.get(f"{k}|own_abs")), f(S.get(f"{k}|advantage_mean")) + sig(S.get(f"{k}|advantage_mean")),
                     *[f(S.get(f"{k}|advantage_g{t}")) for t in (0, 2, 4)]])
    s = "## M4.6a gap specialists in seat A against π in seat B (G50; specialists cross-env under C1)\n\n" + table(
        ["environment", "π", "π against itself", "specialist advantage (mean)", "g0", "g2", "g4"], rows)
    rows = [[env, f(S.get(f"{env}|defence_gain_mean")) + sig(S.get(f"{env}|defence_gain_mean"))] for env in C.ARMS]
    s += "\n\n" + table(["environment", "defence gain: G(spec vs start) − G(spec vs arm policy)"], rows)
    return s


NAT_ROWS = (("all:crash_hp_per100", "crashed HP /100"), ("plates_by150", "plates by 150"), ("plates_by300", "plates by 300"),
            ("all:deaths_per100", "deaths /100"), ("all:recalls_per100", "Recalls /100"), ("all:rec_intr_per100", "Recalls interrupted /100"),
            ("all:collection", "farm collection"), ("all:trade_att_per100", "trade attempts /100"),
            ("all:trade_conn_per100", "connected trades /100"), ("all:held_share", "freeze-held share"),
            ("freeze_episodes_per100", "freeze episodes /100"), ("freeze_episode_mean_len", "freeze episode length"),
            ("all:away_share", "away share"), ("all:ward_per100", "Ward turns /100"), ("all:shove_share", "Shove share"),
            ("early:shove_share", "Shove share 1-100"), ("mid:shove_share", "Shove share 101-240"),
            ("late:shove_share", "Shove share 241-600"), ("late:crash_hp_per100", "crashed HP /100, late"),
            ("late:held_share", "freeze-held share, late"), ("early:cap_mean", "capacity C 1-100"),
            ("mid:cap_mean", "capacity C 101-240"), ("late:cap_mean", "capacity C 241-600"),
            ("all:damage_taken_per100", "damage taken (centi-HP) /100"))
LEAD_ROWS = (("abs_lead@50_median", "|earned lead| t50 median"), ("abs_lead@150_median", "|earned lead| t150 median"),
             ("abs_lead@300_median", "|earned lead| t300 median"), ("abs_lead@150_q90", "|earned lead| t150 q90"),
             ("rel_lead@150_median", "relative lead t150"), ("abs_inv_lead@150_median", "|invested lead| t150"),
             ("slope_150_on_50", "slope lead150 on lead50"), ("slope_300_on_100", "slope lead300 on lead100"),
             ("slope_600_on_300", "slope lead600 on lead300"), ("decisive150_share", "games |lead150| >= 15% of earned"),
             ("attack_q90", "attack ratio q90"), ("attack_share_1.25", "attack ratio >= 1.25 share"),
             ("duel_q90", "duel ratio q90"), ("leader_duel_early", "leader duel ratio 20-50"),
             ("leader_duel_mid", "leader duel ratio 90-150"), ("leader_duel_late", "leader duel ratio 240-320"),
             ("leader_duel_very_late", "leader duel ratio 450-600"), ("S_end", "invested at 600"), ("E_end", "earned at 600"))


def natural_tables(A: dict, sfx: str = "") -> str:
    N = A.get("natural" + sfx, {})
    pre = "ext/" if sfx else ""
    pc, df = N.get("per_combo", {}), N.get("diff", {})
    arms = df.get("arms (C1:s in C1 - baseline:s in baseline)", {})
    pread = df.get("pre-adaptation (start:s in C1 - start:s in baseline)", {})
    s = ""
    for title, rows_def in (("Natural self-play: lane and wave metrics (48 games × 600 turns per policy)", NAT_ROWS),
                            ("Natural self-play: leads, ratios and persistence", LEAD_ROWS)):
        rows = []
        for k, lab in rows_def:
            def g(c):
                return pc.get(c, {}).get(k, {}).get("mean", float("nan"))
            a = arms.get(k, {})
            rows.append([lab, f"{g('start|baseline'):.3f}", f"{g(pre + 'baseline|baseline'):.3f}",
                         f"{g('start|C1'):.3f}", f"{g(pre + 'C1|C1'):.3f}", f(a, 3) + sig(a),
                         f"{a.get('rel', float('nan')):+.0%}" if np.isfinite(a.get("rel", np.nan)) else "",
                         f(pread.get(k, {}), 3) + sig(pread.get(k, {}))])
        s += f"## {title}\n\n" + table(["metric", "start (baseline env)", "baseline arm", "start (C1 env, cross)",
                                        "C1 arm", "C1 arm − baseline arm", "relative", "pre-adaptation C1 − baseline"],
                                       rows) + "\n\n"
    return s


def ext_h2h_tables(A: dict) -> str:
    E = A.get("h2h", {}).get("ext", {})
    rows = [[arm, f(E.get(f"extstart|{arm}")) + sig(E.get(f"extstart|{arm}")),
             f(E.get(f"extvsmain|{arm}")) + sig(E.get(f"extvsmain|{arm}")),
             f(E.get(f"extothers|{arm}")) + sig(E.get(f"extothers|{arm}"))] for arm in C.ARMS]
    s = "## Extension head to head (from s0, own environment)\n\n" + table(
        ["arm", "8.192M − start:s (W 1000)", "8.192M − 4.096M final (W 460)", "8.192M vs other starts (W 460)"], rows)
    rows = [[env, f(E.get(f"extcross|{env}")) + sig(E.get(f"extcross|{env}"))] for env in C.ARMS]
    return s + "\n\n" + table(["environment", "ext C1:s − ext baseline:s (W 460)"], rows)


def records_tables(A: dict) -> str:
    R = A.get("records", {})
    rows = []
    for cls in ("all present", "invested lead >= 8", "invested lead 3-8", "even (|lead| < 3)", "trailing 3-8",
                "trailing >= 8", "frozen on", "frozen on, trailing >= 4", "holding a freeze", "HP < 50%, trailing >= 4",
                "enemy crash under my tower", "level 11-18"):
        for key in ("baseline|baseline", "C1|C1"):
            m = R.get(key, {}).get(cls)
            if not m or not m.get("mass"):
                continue
            ms = m["mass"]
            rows.append([cls, key.split("|")[0], m["n"], *[f"{ms.get(c, float('nan')):.2f}" for c in
                                                            ("LH", "TN", "SL", "SH", "Recall", "Ward", "beh_Hold",
                                                             "beh_Yield", "beh_Trade", "beh_Safe")]])
    return "## State-conditioned action masses (self-play records, 8 games × 600 turns per policy)\n\n" + table(
        ["state class", "arm", "decisions", "LH", "TN", "SL", "SH", "Recall", "Ward", "Hold", "Yield", "Trade", "Safe"],
        rows)


def critic_tables(A: dict) -> str:
    P = A.get("premium", {}).get("critic", {})
    rows = []
    for env in C.ARMS:
        for key in ("equal|mid|0", "leader|early|16", "trailer|early|16", "leader|mid|16", "trailer|mid|16",
                    "leader|late|16", "trailer|late|16", "leader|mid|8", "trailer|mid|8"):
            v = {s: P.get(f"{env}|s{s}", {}).get(key, np.nan) for s in C.SEEDS}
            rows.append([env, key, f(C.seed_stats(v))])
    return "## Critic calibration: V(cell) − realized Q50, seat A (arm policies, own environment)\n\n" + table(
        ["arm", "cell (role|stage|lead)", "bias"], rows)


def extension_tables(A: dict) -> str:
    M = A.get("premium_mid", {})
    pooled = A.get("premium", {}).get("pooled", {})
    rows = []
    for arm in C.ARMS:
        fin = pooled.get(f"self|{arm}|mid|8|50")
        mid = M.get(arm)
        move = (fin["mean"] - mid["mean"]) if fin and mid else float("nan")
        rows.append([arm, f(mid), f(fin), f"{move:+.3f}"])
    s = "## Extension rule: mid +8 G50 premium at the 2.048M snapshots and the finals\n\n" + table(
        ["arm", "2.048M", "4.096M", "change"], rows)
    P = A.get("premium150", {})
    rows = []
    for arm in C.ARMS:
        for st, d in (("early", 16), ("mid", 8), ("mid", 16), ("late", 16)):
            rows.append([arm, st, f"+{d}", *[f(P.get(f"{arm}|{st}|+{d}|G{h}")) for h in (50, 100, 150)]])
    s += "\n\n### Continuation check: neutral cells, horizon 150 (K 64)\n\n" + table(
        ["arm", "stage", "lead", "G50", "G100", "G150"], rows)
    return s


def channel_tables(A: dict) -> str:
    ch = A.get("premium", {}).get("channels", {})
    keys = ("earned", "farm", "plate", "bounty", "deaths", "away", "unc", "unc_away", "trade_conn", "xp", "recalls")
    rows = []
    for st in STAGES:
        for env in C.ARMS:
            for side in ("leader", "trailer"):
                vals = []
                for k in keys:
                    v = [ch.get(f"{env}|{st}|+16|s{s}", {}).get(f"{side}_{k}", np.nan) for s in C.SEEDS]
                    vals.append(f"{np.nanmean(v):+.2f}")
                rows.append([st, env, side] + vals)
    return "## Channels of a +16 lead (lead cells − equal cell, both orientations, 50 turns, undiscounted)\n\n" + table(
        ["stage", "arm", "side", *keys], rows)


def run() -> str:
    A = C.load("analysis") or {}
    for sfx, ph in (("", "main"), ("_ext", "ext")):
        tr = C.load(f"train_{ph}") or {}
        A["train_check" + sfx] = {"all_passed": tr.get("all_passed"),
                                  "failed": [k for k, v in tr.get("checks", {}).items() if not v]}
    sc = scorecard(A)
    sc_ext = scorecard(A, "_ext") if A.get("premium_ext") else {}
    C.save_json("scorecard", {"meta": C.meta(stage="scorecard"), "scorecard": sc, "scorecard_ext": sc_ext})
    parts = ["# M4.6c generated tables", "",
             "*Generated from results/m46c/analysis.json. Values: mean over the six seeds ± t(5) 95% half-width; "
             "(+)/(−) the interval excludes 0. Narrative: docs/m4/m4_6/m4_6c_report.md.*", "", "## Scorecard", ""]
    for k in ("A", "B", "C", "D", "E"):
        v = sc.get(k, {})
        parts += _score_line(k, v)
    parts += ["", f"**Rule outcome (4.096M):** {sc.get('recommendation_rule')}", ""]
    if sc_ext:
        parts += ["### Scorecard of the 8.192M extension (amendment of 2026-10-15)", ""]
        for k in ("A", "B", "C", "D", "E"):
            parts += _score_line(k, sc_ext.get(k, {}))
        parts += ["", f"**Rule outcome (8.192M):** {sc_ext.get('recommendation_rule')}", ""]
    for fn in (training_tables, premium_tables, extension_tables, channel_tables, critic_tables, h2h_tables,
               battery_tables, cf_tables, freeze_tables, specialist_tables, natural_tables, records_tables):
        try:
            parts += [fn(A), ""]
        except Exception as ex:                            # noqa: BLE001 - a missing family leaves a note, not a crash
            parts += [f"*{fn.__name__}: not available ({ex!r})*", ""]
    if A.get("premium_ext"):
        parts += ["# The 8.192M extension (amendment of 2026-10-15)", ""]
        for fn in (lambda a: training_tables(a, "_ext"), lambda a: premium_tables(a, "_ext"),
                   lambda a: battery_tables(a, "_ext"), lambda a: cf_tables(a, "_ext"),
                   lambda a: natural_tables(a, "_ext"), ext_h2h_tables):
            try:
                parts += [fn(A), ""]
            except Exception as ex:                        # noqa: BLE001
                parts += [f"*extension table not available ({ex!r})*", ""]
    text = "\n".join(parts) + "\n"
    C.write_md("report", text)
    return text


def _score_line(k: str, v: dict) -> list:
    detail = ""
    if k == "A" and v.get("primary"):
        p = v["primary"]
        detail = f"difference {f(p)}, ratio ×{p['ratio_of_means']:.2f}, seeds {p['pos']}+/{p['neg']}−"
    elif k in ("B", "D"):
        detail = "; ".join(v.get("flags", [])) or "-"
    elif k == "C":
        detail = "issues: " + ("; ".join(v.get("issues", [])) or "-") + ". notes: " + ("; ".join(v.get("notes", [])) or "-")
    elif k == "E":
        detail = "; ".join(v.get("issues", [])) or "-"
    return [f"- **{k}**: {v.get('outcome')} — {detail}"]


__all__ = ["run", "scorecard"]
