"""Generated tables of M4.6b (results/m46b/report.md). The interpretation is in
docs/m4/m4_6/m4_6b_investment_report.md; this module only formats the stage summaries."""

from __future__ import annotations

import numpy as np

from .common import fmt, load, write_md
from .curves import ARMS

ARMS_ORDER = ("baseline", "C1", "C2", "C3")


def _t(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _c(v, d=2):
    if not v or v.get("mean") is None or not np.isfinite(v.get("mean", np.nan)):
        return "n/a"
    return fmt(v["mean"], v.get("ci", float("nan")), d)


def verify_section():
    s = ["## Measurement fixes: verification\n"]
    v1 = load("verify_v1_positions")
    if v1:
        rows = []
        for k, v in v1["results"].items():
            p = v["positions"]
            rows.append([k, f"{v['identical_fields']} / {v['identical_fields'] + v['different_fields']}",
                         f"{p['stronger.mean_pos']['stored']:+.2f} → {p['stronger.mean_pos']['fixed']:+.2f}",
                         f"{p['stronger.ward_share']['fixed']:.3f}",
                         f"{p['even.mean_pos']['stored']:+.2f} → {p['even.mean_pos']['fixed']:+.2f}",
                         f"{p['weaker.mean_pos']['stored']:+.2f} → {p['weaker.mean_pos']['fixed']:+.2f}",
                         f"{v['credible_threat_n']['threat']['stored']} → {v['credible_threat_n']['threat']['fixed']}"])
        s.append("### V1: the M4.5c power-gap stage on its original seeds\n")
        s.append(_t(rows, ["final", "other readouts identical", "stronger cell (stored → lane only)", "stronger at W",
                           "even cell", "weaker cell", "credible-threat turns"]))
    v2 = load("verify_v2_battery")
    if v2:
        s.append(f"\n### V2: battery families {', '.join(v2['meta']['families'])} for finals {v2['meta']['finals']}\n")
        s.append(f"{v2['jobs']} jobs; {v2['compared']} compared with the stored M4.5c raw jobs; **{v2['identical']} "
                 f"identical** (G and Q of every rollout of every original candidate); not identical: "
                 f"{len(v2['not_identical'])}.\n")
    a = load("verify_v2_analysis")
    if a:
        rows = []
        for k, F in a["results"].items():
            for c, v in F.items():
                if c in ("channels", "states") or "-policy" in c and "beh" not in c and not c.startswith("plan:"):
                    continue
                if isinstance(v, dict) and "mean" in v:
                    rows.append([k, c, f"{v['mean']:+.3f}", " / ".join(f"{x:+.3f}" for x in v["per_final"])])
        s.append("\nCorrected readings (two finals; primary statistic of the family at 50 turns):\n")
        s.append(_t(rows, ["family | opponent", "contrast", "mean", "per final"]))
    return "\n".join(s)


def audit_section():
    a = (load("audit") or {}).get("analysis")
    if not a:
        return "## Audit\n\nnot run\n"
    s = ["## Audit of the current curve (s0 self-play, 6 finals × 24 games × 600 turns)\n"]
    rows = [[t, f"{r['S_median']:.1f}", f"{r['g_median']:.1f}", f"{r['L_median']:.0f}", f"{r['abs_dE_median']:.1f}",
             f"{r['abs_dE_q90']:.1f}", f"{r['abs_dS_median']:.1f}", f"{r['rel_lead_median']:.3f}",
             f"{r['abs_dL_mean']:.2f}"] for t, r in a["trajectory"].items()]
    s.append(_t(rows, ["turn", "invested S (median)", "unspent g", "level", "|earned lead| median", "q90",
                       "|invested lead| median", "relative lead", "mean |level gap|"]))
    s.append(f"\nDiscount-weighted invested gold (weight 0.99^t): {a['discount_weighted_S']}; income per turn: "
             f"{ {k: round(v, 3) for k, v in a['income_per_turn'].items()} }\n")
    rows = []
    for k, r in a["decomposition"].items():
        rows.append([k, f"{r['S_mean']:.1f}", f"{r['abs_lead_mean']:.1f}", f"×{r['relative_gold_lead']:.4f}",
                     f"{r['elasticity_at_mean_S']:.3f}", f"×{r['M_ratio_if_all_earned_invested']:.4f}",
                     f"×{r['M_ratio_actual']:.4f}", f"×{r['level_ratio_AP']:.4f}", f"×{r['attack_ratio']:.4f}",
                     f"×{r['m44_power_ratio']:.4f}", f"×{r['duel_ratio']:.4f}", f"{r['unspent_lead_mean']:+.2f}",
                     f"{r['share_leader_more_invested']:.2f}"])
    s.append("\n### From a gold lead to strength (oriented to the gold leader; geometric means over present turns)\n")
    s.append(_t(rows, ["stage", "mean S", "mean |lead|", "relative gold lead", "elasticity", "M ratio if all earned "
                       "were invested", "M ratio (actual)", "level factor", "attack ratio", "M4.4 'power'",
                       "duel ratio", "unspent lead", "leader more invested"]))
    rows = [[k, v["n"], v["turn_median"], f"{v['S_median']:.0f}", f"{v['m44_power_ratio_mean']:.4f}",
             f"{v['share_turns_after_150']:.2f}"] for k, v in a["lead_bins"].items()]
    s.append("\n### M4.6a's binning by |earned lead|, re-derived\n")
    s.append(_t(rows, ["|lead| bin", "turns", "median turn", "median S", "M4.4 power ratio (max(r, 1/r))",
                       "share after turn 150"]))
    rows = []
    for st, v in a["what_leads_buy"].items():
        for k, r in v["leads"].items():
            e = r["exchange_bars"]
            rows.append([st, f"S {v['S']:.0f}, L{v['level']}", k, f"×{r['M_ratio']:.4f}", f"×{r['duel_ratio']:.4f}",
                         f"{r['capacity'][0]} → {r['capacity'][1]}", f"{r['pressure_per_turn'][0]} → "
                         f"{r['pressure_per_turn'][1]}", f"{e['leader_on_trailer']:.4f} / {e['trailer_on_leader']:.4f} "
                         f"(equal {e['equal']:.4f})"])
    s.append("\n### What a lead buys (current curve)\n")
    s.append(_t(rows, ["stage", "typical", "lead", "M ratio", "duel ratio", "C (trailer → leader)",
                       "Pressure per turn", "bars per exchange: leader on trailer / trailer on leader"]))
    rows = [[k, f"{v['M']:.3f}", f"{v['dM_per_res']:.4f}", f"{v['elasticity']:.3f}"] for k, v in a["curve_table"].items()]
    s.append("\n### The current curve\n")
    s.append(_t(rows, ["S", "M(S)", "dM/dS per resource", "elasticity"]))
    p = a["persistence"]
    s.append("\nLead persistence: " + "; ".join(f"{k}: slope {v['slope_lead_on_lead']:.2f}, relative lead "
                                               f"{v['rel_lead_t0']:.3f} → {v['rel_lead_t1']:.3f}" for k, v in p.items()) + "\n")
    s.append("\nReturns: " + "; ".join(f"{k}: {v['returns_per_100_turns']:.1f} per 100 turns, {v['bought_per_return']:.1f} "
                                      f"invested per return" for k, v in a["returns"].items()) + "\n")
    return "\n".join(s)


def s1_section():
    d = load("s1_mapping")
    if not d:
        return ""
    s = ["## S1: mapping tables per arm\n"]
    rows = []
    for arm in ARMS_ORDER:
        r = d["results"].get(arm)
        if not r:
            continue
        for st, v in r["stages"].items():
            L = v["leads"]
            rows.append([arm, st, f"{v['M']:.3f}", v["capacity"], v["pressure"], f"{v['pressure_turns_per_plate']:.2f}"]
                        + [f"×{L[k]['duel_ratio']:.3f} (C {L[k]['capacity_leader_trailer'][0]})" for k in ("4.0", "8.0", "16.0")])
    s.append(_t(rows, ["arm", "stage", "M", "C", "Pressure", "Pressure turns per 2,000", "duel ratio +4 (leader C)",
                       "+8", "+16"]))
    rows = [[arm] + [f"{d['results'][arm]['curve'][S]['M']:.2f} / {d['results'][arm]['curve'][S]['elasticity']:.2f}"
                     for S in ("5", "15", "35", "56", "108", "234")] for arm in ARMS_ORDER if arm in d["results"]]
    s.append("\nM / elasticity at S = 5, 15, 35, 56, 108, 234:\n")
    s.append(_t(rows, ["arm", "5", "15", "35", "56", "108", "234"]))
    rows = [[arm] + [d["results"][arm]["capacity_by_level"][L]["C"] for L in ("4", "6", "8", "10", "13", "15", "18")]
            for arm in ARMS_ORDER if arm in d["results"]]
    s.append("\nWave capacity C at the level's typical investment (M4.5b TYPICAL_INVEST):\n")
    s.append(_t(rows, ["arm", "L4", "L6", "L8", "L10", "L13", "L15", "L18"]))
    return "\n".join(s)


def s2_section():
    if not load("s2_exact"):
        return ""
    from .screen import analyse_s2
    a = analyse_s2()
    z, f = a["zone"], a["freeze"]
    s = [f"## S2: the exact one-turn ledger (HP priced at {a['price']:g} per bar)\n"]
    rows = []
    for st in ("early", "mid", "late"):
        for key, label in (("A_trade_minus_hold_vs_lh_hold", "A: Trade − Hold, against Hold"),
                           ("A_trade_minus_hold_vs_safe", "A: Trade − Hold, against Safe"),
                           ("B_safe_minus_hold_vs_trade", "B: Safe − Hold, against Trade")):
            rows.append([st, label] + ["/".join(f"{z[f'{arm}|neutral|{st}|+{d}'].get(key, float('nan')):+.3f}"
                                                for d in (0, 8, 16)) for arm in ARMS_ORDER])
    s.append("Neutral situation, A ahead by 0 / +8 / +16 invested:\n")
    s.append(_t(rows, ["stage", "contrast"] + list(ARMS_ORDER)))
    rows = []
    for L in (4, 10, 15):
        for h in (5, 12, 20):
            for inv in ("equal", "freezer+8", "breaker+8"):
                r = [f[f"{arm}|L{L}|held{h}|{inv}"] for arm in ARMS_ORDER]
                rows.append([f"L{L}", h, inv]
                            + [f"{x.get('P_hold_break_vs_guard', float('nan')):.0f} / {x.get('P_hold_break_vs_keep', float('nan')):.0f}"
                               f" ; {x.get('B_break_minus_walk_up_vs_guard', float('nan')):+.2f}"
                               f" ; {x.get('A_guard_minus_keep_vs_break', float('nan')):+.2f}" for x in r])
    s.append("\nFreezes: P(the freeze survives a break) against guard / keep ; the breaker's break − walk up against the "
             "guard ; the freezer's guard − keep against the break. A walk-up never breaks a freeze (P = 1 everywhere):\n")
    s.append(_t(rows, ["level", "held", "investment"] + list(ARMS_ORDER)))
    return "\n".join(s)


def s3_section():
    d = load("s3_summary")
    if not d:
        return ""
    a = d["analysis"]
    s = ["## S3: purchased leads (pre-registered; K = 16, horizon 50)\n"]
    rows = []
    for st in ("early", "mid", "late"):
        for dl in ("+8", "+16"):
            r = [st, dl] + [_c(a["premium"].get(f"{arm}|{st}|{dl}")) for arm in ARMS_ORDER]
            r += [_c(a["premium"].get(f"{arm}-baseline|{st}|{dl}")) for arm in ARMS_ORDER[1:]]
            rows.append(r)
    s.append(_t(rows, ["stage", "lead", "baseline premium", "C1", "C2", "C3", "C1 − baseline", "C2 − baseline",
                       "C3 − baseline"]))
    rows = []
    for st in ("early", "mid", "late"):
        for dl in ("+8", "+16"):
            rows.append([st, dl] + [f"{a['weak'][f'{arm}|{st}|{dl}|abs_policy']['mean']:+.2f} / best "
                                    f"{a['weak'][f'{arm}|{st}|{dl}|best']['best']} "
                                    f"{a['weak'][f'{arm}|{st}|{dl}|best']['value']:+.2f}" for arm in ARMS_ORDER])
    s.append("\nThe trailer (focal): its G50 under the policy / its best forced response minus the policy:\n")
    s.append(_t(rows, ["stage", "lead"] + list(ARMS_ORDER)))
    s.append(f"\nValidation (6 cells at 150 turns): {a['validation']}\n")
    return "\n".join(s)


def p1x_section():
    d = load("p1x_summary")
    if not d:
        return ""
    a = d["analysis"]
    s = ["## EXPLORATORY: the lead premium at higher power (K = 128, both orientations, horizon 100)\n"]
    rows = []
    for st in ("mid", "late"):
        for dl in ("+8", "+16"):
            for h in ("G50", "G100"):
                rows.append([st, dl, h] + [_c(a["premium"].get(f"{arm}|{st}|{dl}|{h}")) for arm in ARMS_ORDER]
                            + [_c(a["diff"].get(f"{arm}-baseline|{st}|{dl}|{h}")) +
                               f" (×{a['diff'].get(f'{arm}-baseline|{st}|{dl}|{h}', {}).get('ratio', float('nan')):.2f})"
                               for arm in ARMS_ORDER[1:]])
    s.append(_t(rows, ["stage", "lead", "horizon", "baseline", "C1", "C2", "C3", "C1 − baseline (ratio)",
                       "C2 − baseline", "C3 − baseline"]))
    return "\n".join(s)


def s45_section():
    s = []
    d4 = load("s4_summary")
    if d4:
        a = d4["analysis"]
        rows = []
        for k in sorted(a):
            arm, focal, L, h, inv = k.split("|")
            r = a[k]
            cands = ("keep", "guard") if focal == "freezer" else ("break", "walk_up", "step_back", "reset")
            rows.append([arm, focal, L, h, inv, _c(r["abs_policy"])] + [_c(r.get(f"{c}-policy")) for c in cands]
                        + [f"{r['policy|held_A@20']:.1f}" if focal == "freezer" else ""])
        s.append("## S4: freeze contests (K = 16, horizon 50; candidate − policy)\n")
        s.append(_t(rows, ["arm", "focal", "level", "held", "investment", "policy's G50", "keep / break",
                           "guard / walk up", "— / step back", "— / reset", "held turns in 20 (freezer)"]))
    d5 = load("s5_summary")
    if d5:
        a = d5["analysis"]
        rows = [[arm, fam] + [_c(a.get(f"{arm}|{fam}|{k}")) for k in ("recall-stay", "inv:recall-inv:stay", "shopping",
                                                                       "inv:policy-policy")]
                for arm in ARMS_ORDER for fam in ("recall_full", "recall_mid")]
        s.append("\n## S5: Recall and shopping (K = 12, horizon 50, G50)\n")
        s.append(_t(rows, ["arm", "family", "Recall − stay", "same, gold pre-invested", "shopping component",
                           "pre-invested state − original"]))
    return "\n".join(s)


def s6_section():
    d = load("s6_summary")
    if not d:
        return ""
    a = d["analysis"]
    s = ["## S6: natural starts (16 games × 600 turns per final and arm; baseline-trained policies)\n"]
    rows = []
    for arm in ARMS_ORDER:
        r = a[arm]["ratios"]
        dec = a[arm]["audit"]["decomposition"]
        rows.append([arm, f"{r['attack_q']['0.9']:.3f}", f"{r['attack_share_1.1']:.3f}", f"{r['attack_share_1.25']:.4f}",
                     f"{r['duel_q']['0.9']:.3f}", f"{r['duel_share_1.25']:.3f}"]
                    + [f"×{dec[k]['duel_ratio']:.3f}" for k in ("early", "mid", "late", "very late")])
    s.append(_t(rows, ["arm", "attack ratio q90", "share ≥ 1.1", "share ≥ 1.25", "duel ratio q90", "duel share ≥ 1.25",
                       "leader's duel ratio: early", "mid", "late", "very late"]))
    keys = ("crash_hp", "plates150", "plates300", "deaths", "recalls", "collection", "shove", "trade_conn", "cap_mid",
            "cap_late")
    rows = [[arm] + [_c(a[arm]["lane"][k], 3) for k in keys] for arm in ARMS_ORDER]
    s.append("\nLane sanity (per side; per 100 turns where a rate):\n")
    s.append(_t(rows, ["arm", "crashed HP / 100", "plates by 150", "plates by 300", "deaths / 100", "Recalls / 100",
                       "farm collection", "Shove share", "connected trades / 100", "C turns 90-150", "C turns 240-320"]))
    rows = [[arm, _c(a[arm]["lane"]["lead300_on_100"]),
             f"{a[arm]['lane']['lead300_on_100'].get('pooled', float('nan')):.2f}"] for arm in ARMS_ORDER]
    s.append("\nLead persistence (D2): the signed earned lead at turn 300 regressed on the lead at turn 100:\n")
    s.append(_t(rows, ["arm", "slope (mean of the six finals)", "slope (all games pooled)"]))
    return "\n".join(s)


def scorecard_section():
    """The pre-registered criteria D1–D4, evaluated mechanically from the stage summaries (point estimates where the
    criterion is a point criterion; the interval where it names one)."""
    s3, s1, s6 = load("s3_summary"), load("s1_mapping"), load("s6_summary")
    if not (s3 and s1 and s6):
        return ""
    P = s3["analysis"]["premium"]
    R1 = s1["results"]
    L6 = s6["analysis"]
    rows = []
    for arm in ARMS_ORDER[1:]:
        d1 = []
        for st in ("mid", "late"):
            b, c, dd = P[f"baseline|{st}|+8"], P[f"{arm}|{st}|+8"], P[f"{arm}-baseline|{st}|+8"]
            ok = c["mean"] >= 1.5 * b["mean"] and dd["mean"] - dd["ci"] > 0
            d1.append(f"{st} {c['mean']:+.2f} vs 1.5 × {b['mean']:+.2f}, diff {dd['mean']:+.2f} ± {dd['ci']:.2f}: "
                      + ("pass" if ok else "fail"))
        e, eb = P[f"{arm}|early|+8"]["mean"], P["baseline|early|+8"]["mean"]
        share = L6[arm]["ratios"]["attack_share_1.25"]
        slope = L6[arm]["lane"]["lead300_on_100"]
        d2ok = (e <= 1.5 * eb, share <= 0.05, slope["mean"] <= 1.2 and slope.get("pooled", 0.0) <= 1.2)
        d2 = (f"early {e:+.2f} vs ≤ {1.5 * eb:+.2f}: {'pass' if d2ok[0] else 'fail'}; share ≥ 1.25 {share:.4f}: "
              f"{'pass' if d2ok[1] else 'fail'}; slope {slope['mean']:.2f} / {slope.get('pooled', float('nan')):.2f}: "
              f"{'pass' if d2ok[2] else 'fail'}")
        lane, base = L6[arm]["lane"], L6["baseline"]["lane"]
        d3 = []
        for k in ("crash_hp", "plates150", "deaths", "recalls", "collection"):
            r = lane[k]["mean"] / base[k]["mean"] - 1
            d3.append(f"{k} {r:+.0%}" + ("" if abs(r) <= 0.25 else " FAIL"))
        sh = lane["shove"]["mean"] - base["shove"]["mean"]
        d3.append(f"Shove {sh:+.3f}" + ("" if abs(sh) <= 0.10 else " FAIL"))
        d3ok = all("FAIL" not in x for x in d3)
        d4 = []
        for st in ("early", "mid", "late"):
            dc = R1[arm]["stages"][st]["capacity"] - R1["baseline"]["stages"][st]["capacity"]
            d4.append(f"C {st} {dc:+d}" + ("" if abs(dc) <= 2 else " FAIL"))
        pr = R1[arm]["stages"]["mid"]["pressure"] / R1["baseline"]["stages"]["mid"]["pressure"] - 1
        d4.append(f"Pressure mid {pr:+.1%}" + ("" if abs(pr) <= 0.25 else " FAIL"))
        d4ok = all("FAIL" not in x for x in d4)
        d1ok = all(x.endswith("pass") for x in d1)
        verdict = "advances" if (d1ok and all(d2ok) and d3ok and d4ok) else "does not advance"
        rows.append([arm, "; ".join(d1), d2, ", ".join(d3) + (": pass" if d3ok else ": fail"),
                     ", ".join(d4) + (": pass" if d4ok else ": fail"), verdict])
    return ("## Pre-registered scorecard (D1–D4)\n\n"
            + _t(rows, ["arm", "D1 consequential leads (+8)", "D2 not decisive", "D3 equal lane (vs baseline)",
                        "D4 waves and towers (S1 stage states)", "rule"]))


def run() -> str:
    parts = ["# M4.6b generated tables\n", "Interpretation: docs/m4/m4_6/m4_6b_investment_report.md.\n",
             "Arms: " + "; ".join(f"{k}: {v['label']}" for k, v in ARMS.items()) + "\n",
             scorecard_section(), verify_section(), audit_section(), s1_section(), s2_section(), s3_section(), p1x_section(),
             s45_section(), s6_section()]
    text = "\n".join(p for p in parts if p)
    write_md("report", text)
    return text


__all__ = ["run"]
