"""Generated tables of M4.6a (results/m46a/report.md) from the stage summaries. The interpretation is in
docs/m4/m4_6a_report.md; this file only formats numbers."""

from __future__ import annotations

import json
from typing import List

import numpy as np

from .common import OUT, fmt, load_json, write_md


def _t(rows: List[list], head: List[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def _f(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}"


def _c(c: dict, d=2) -> str:
    if not c:
        return "n/a"
    s = fmt(c.get("mean", float("nan")), c.get("ci", float("nan")), d)
    if "k_pos" in c:
        n = c.get("finals", len(c.get("per_final", {})))
        s += f" ({c['k_pos']}+/{c['k_neg']}− of {n})"
    return s


def e1_section(e5_price: float = None) -> str:
    from .e1 import analyse
    d = load_json("e1_exact")
    if not d:
        return "## E1\n\nnot run\n"
    a = analyse(d, e5_price=e5_price)
    s = ["## E1 — exact one-turn ledger\n"]
    s.append("### Freeze cells: does a walk-up Shove break the freeze this turn?\n")
    rows = []
    for r in a["freeze"]:
        if "P_hold_after_break_vs_keep" not in r:
            continue
        L, gap, held = r["cell"]
        rows.append([L, gap, held, f"{r['P_hold_after_break_vs_keep']:.2f}", f"{r['P_hold_after_break_vs_guard']:.2f}",
                     f"{r['B_taken_bar_break_vs_keep']:.3f}", f"{r['B_taken_bar_break_vs_guard']:.3f}",
                     f"{r['A_taken_bar_guard_vs_break']:.3f}", f"{r['A_min_pos_bar_guard']:.3f}",
                     f"{r['P_B_death_break_vs_guard']:.3f}",
                     _f(r.get("B_break_minus_walk_up_vs_guard@1.5")), _f(r.get("B_break_minus_step_back_vs_guard@1.5")),
                     _f(r.get("A_guard_minus_keep_vs_break@1.5"))])
    s.append(_t(rows, ["level", "gap", "held HP", "P(freeze holds) vs keep", "P(holds) vs guard",
                       "B HP lost vs keep (bar)", "B HP lost vs guard", "A HP lost guarding", "of which positional",
                       "P(B dies) vs guard", "B: break − walk up vs guard (priced 1.5)",
                       "B: break − step back vs guard", "A: guard − keep vs break"]))
    s.append("\n### Zoning cells: the one-turn threat game (HP priced at 1.5 resources per bar)\n")
    rows = []
    for r in a["zone"]:
        sit, L, gap = r["cell"]
        rows.append([sit, L, gap, _f(r.get("A_trade_minus_hold_vs_lh_hold@1.5")),
                     _f(r.get("A_trade_minus_hold_vs_safe@1.5")), _f(r.get("A_trade_minus_hold_vs_yield@1.5")),
                     _f(r.get("B_safe_minus_hold_vs_trade@1.5")), _f(r.get("B_safe_minus_hold_vs_hold@1.5")),
                     f"{r['A_dealt_bar']:.3f}", f"{r['A_taken_bar']:.3f}", f"{r['A_farm_trade'] - r['A_farm_hold']:+.3f}"])
    s.append(_t(rows, ["situation", "level", "gap", "A: Trade − Hold vs Hold", "vs Safe", "vs Yield",
                       "B: Safe − Hold vs Trade", "B: Safe − Hold vs Hold", "A dealt (bar)", "A taken (bar)",
                       "A farm: Trade − Hold"]))
    if e5_price:
        s.append(f"\nAt E5's measured price ({e5_price:.2f} per bar):\n")
        rows = []
        for r in a["zone"]:
            sit, L, gap = r["cell"]
            k = f"{e5_price:g}"
            rows.append([sit, L, gap, _f(r.get(f"A_trade_minus_hold_vs_lh_hold@{k}")),
                         _f(r.get(f"A_trade_minus_hold_vs_safe@{k}")), _f(r.get(f"B_safe_minus_hold_vs_trade@{k}"))])
        s.append(_t(rows, ["situation", "level", "gap", "A: Trade − Hold vs Hold", "vs Safe", "B: Safe − Hold vs Trade"]))
    s.append("\n### Decomposition (one cost removed at a time; counterfactual configurations, not candidates)\n")
    rows = []
    for p, lb in a["load_bearing"].items():
        for v, x in lb.items():
            rows.append([p, v, f"{x['freeze_guard']['sign_changes']}/{x['freeze_guard']['n']}",
                         _f(x["freeze_guard"]["mean_baseline"]), _f(x["freeze_guard"]["mean_variant"]),
                         f"{x['zone_threat']['sign_changes']}/{x['zone_threat']['n']}",
                         _f(x["zone_threat"]["mean_baseline"]), _f(x["zone_threat"]["mean_variant"]),
                         "yes" if (x["freeze_guard"]["load_bearing"] or x["zone_threat"]["load_bearing"]) else "no"])
    s.append(_t(rows, ["price", "removed cost", "guard − keep: sign changes", "mean (baseline)", "mean (removed)",
                       "Trade − Hold: sign changes", "mean (baseline)", "mean (removed)", "load-bearing"]))
    return "\n".join(s)


def run() -> str:
    parts = ["# M4.6a generated tables\n", "Interpretation: docs/m4/m4_6a_report.md.\n"]
    e5 = load_json("e5_summary")
    price = None
    if e5:
        try:
            price = float(e5["analysis"]["hp_late"]["price_per_bar|Q"]["mean"])
        except Exception:
            price = None
    parts.append(e1_section(price))
    for name, fn in (("e3", e3_section), ("e5", e5_section), ("e2", e2_section), ("e4", e4_section),
                     ("e6", e6_section), ("exploratory", exploratory_section)):
        try:
            parts.append(fn())
        except Exception as ex:                          # pragma: no cover - partial runs
            parts.append(f"## {name.upper()}\n\nnot available ({type(ex).__name__}: {ex})\n")
    text = "\n".join(parts)
    write_md("report", text)
    return text


E3_GAPS = ("x1.24", "x1.5", "x1.76", "x2.2")


def e3_premium(tab: dict, gap: str, h: str, key: str, level: str = None, sign: float = 1.0):
    """Cross-final mean and t interval of the paired premium (gap minus equal) of one quantity, levels pooled or one
    level."""
    from .common import mean_ci
    vals = []
    for s, byL in tab.items():
        ls = [level] if level else list(byL)
        v = [byL[L][gap][h]["premium"].get(key) for L in ls if gap in byL[L]]
        v = [x for x in v if x is not None]
        if v:
            vals.append(sign * float(np.mean(v)))
    return mean_ci(vals)


def e3_abs(tab: dict, gap: str, h: str, key: str):
    from .common import mean_ci
    vals = []
    for s, byL in tab.items():
        v = [byL[L][gap][h]["mean"].get(key) for L in byL if gap in byL[L]]
        if v:
            vals.append(float(np.mean(v)))
    return mean_ci(vals)


def e3_section() -> str:
    d = load_json("e3_summary")
    if not d:
        return "## E3\n\nnot run\n"
    tab = d["summary"]["table"]
    s = ["## E3 — power-gap self-play (stronger minus weaker; premium = gap minus the equal-power game of the same "
         "seeds; t intervals over the six finals; levels 6 / 10 / 14 pooled unless stated)\n"]
    rows = []
    for gap in E3_GAPS:
        r = [gap]
        for h in ("20", "50", "100", "300"):
            r.append(fmt(*e3_premium(tab, gap, h, "G")))
        r.append(fmt(*e3_premium(tab, gap, "100", "lead")))
        r.append(fmt(*e3_premium(tab, gap, "300", "lead")))
        rows.append(r)
    s.append("### E3-P1: the power premium\n")
    s.append(_t(rows, ["gap", "G20", "G50", "G100", "G300", "lead @100 (undiscounted)", "lead @300"]))
    rows = []
    for gap in E3_GAPS:
        for L in ("6", "10", "14"):
            rows.append([gap, L, fmt(*e3_premium(tab, gap, "100", "G", L)), fmt(*e3_premium(tab, gap, "100", "lead", L))])
    s.append("\nBy level (G100 and lead at 100):\n")
    s.append(_t(rows, ["gap", "level", "G100 premium", "lead @100 premium"]))
    for h in ("100", "300"):
        rows = []
        for gap in E3_GAPS:
            P = lambda k, sg=1.0: fmt(*e3_premium(tab, gap, h, k, sign=sg))
            rows.append([gap, P("lead"), P("farm_s"), P("bounty_s"), P("plate_s"), P("farm_avail_w", -1.0),
                         _posture(tab, gap, h), P("unc_away_w"), P("bounty_w", -1.0), P("plate_w", -1.0),
                         P("away_w"), P("deaths_w"), P("recalls_w"), P("xp_lost_w"), P("deaths_s")])
        s.append(f"\n### E3-P2: decomposition of the premium at {h} turns (resources; positive = in the stronger "
                 "side's favour)\n")
        s.append(_t(rows, ["gap", "lead", "stronger: farm", "stronger: bounty", "stronger: plates",
                           "weaker: minion flow lost", "weaker: lost while present", "weaker: lost while away",
                           "weaker: bounty (−)", "weaker: plates (−)", "Δ weaker turns away", "Δ weaker deaths",
                           "Δ weaker Recalls", "Δ weaker XP lost", "Δ stronger deaths"]))
    rows = []
    for gap in E3_GAPS:
        g = {k: [] for k in ("s_trade_rate", "s_conn_rate", "s_pos", "s_W", "w_pos", "w_W", "w_yield", "w_safe",
                             "w_tower", "w_unc_near", "w_unc_far", "near_share", "turns")}
        for sd, byL in tab.items():
            for L, byg in byL.items():
                for k in g:
                    g[k].append(byg[gap]["gap_turns"].get(k, float("nan")))
        rows.append([gap] + [f"{np.mean(g[k]):.3f}" for k in ("s_trade_rate", "s_conn_rate", "s_pos", "s_W", "w_pos",
                                                               "w_W", "w_yield", "w_safe", "w_tower", "near_share")]
                    + [f"{np.mean(g['w_unc_near']) - np.mean(g['w_unc_far']):+.3f}", int(np.mean(g["turns"]))])
    s.append("\n### E3-P3: behaviour while the gap lasts (power ratio ≥ 1.25, both present; means over finals and "
             "levels)\n")
    s.append(_t(rows, ["gap", "stronger Trade rate", "connected rate", "stronger cell (lane cells only)",
                       "stronger at W", "weaker cell (lane only)", "weaker at W", "weaker Yield", "weaker Safe",
                       "weaker at own tower", "weaker catchable share", "weaker farm lost: catchable − not",
                       "turns per cell"]))
    cont = d["summary"].get("continuity", {})
    if cont:
        rows = []
        for s_, c in sorted(cont.items()):
            bp = c.get("by_power", {})
            pc = c.get("pressure_cost") or {}
            near = (pc.get("opp_uncollected_when_stronger_near") or {}).get("mean")
            other = (pc.get("opp_uncollected_otherwise") or {}).get("mean")
            rows.append([s_, _f((bp.get("stronger") or {}).get("trade_attempt_rate"), 3),
                         _f((bp.get("stronger") or {}).get("mean_pos")), _f((bp.get("weaker") or {}).get("at_own_tower"), 3),
                         _f((c.get("weaker_mass") or {}).get("beh_Yield"), 3),
                         _f(near - other if near is not None and other is not None else None, 3)])
        s.append("\nM4.5c stage re-run on new seeds (viewer neutral, level 3, A ×1.76; M4.5c: Trade rate 0.26, "
                 "cell +2.03, at tower 0.38, Yield 0.30, denial 0.014):\n")
        s.append(_t(rows, ["final", "stronger Trade rate", "stronger cell", "weaker at tower", "weaker Yield",
                           "denial when near"]))
    return "\n".join(s)


def _posture(tab, gap, h) -> str:
    from .common import mean_ci
    vals = []
    for sd, byL in tab.items():
        v = [byL[L][gap][h]["premium"]["unc_w"] - byL[L][gap][h]["premium"]["unc_away_w"] for L in byL]
        vals.append(float(np.mean(v)))
    return fmt(*mean_ci(vals))


def e5_section() -> str:
    d = load_json("e5_summary")
    if not d:
        return "## E5\n\nnot run\n"
    a = d["analysis"]
    s = ["## E5 — the price of a reset and of HP (paired, K = 12, horizon 50; Q50 primary, G50 beside it)\n"]
    rows = []
    for fam in ("recall_full", "recall_mid"):
        x = a[fam]
        for key, lab in (("recall-policy", "1t Recall − policy"), ("stay-policy", "1t stay − policy"),
                         ("shove-policy", "1t Shove − policy"), ("recall-stay", "Recall − stay"),
                         ("inv:recall-inv:stay", "Recall − stay, gold pre-invested"),
                         ("shopping", "shopping component"),
                         ("inv:policy-policy", "pre-invested state − original (policy)")):
            rows.append([fam, lab, _c(x.get(f"{key}|Q")), _c(x.get(f"{key}|G"))])
    s.append(_t(rows, ["family", "contrast", "ΔQ50", "ΔG50"]))
    rows = []
    for fam in ("recall_full", "recall_mid"):
        x = a[fam]
        for key in ("away", "deaths", "farm", "earned", "unc", "xp_lost", "d_total"):
            rows.append([fam, key, _c(x.get(f"ledger:recall-policy:{key}@20")),
                         _c(x.get(f"ledger:recall-policy:{key}_B@20"))])
    s.append("\nRecall − policy, ledger at 20 turns (focal A, opponent B):\n")
    s.append(_t(rows, ["family", "quantity", "A", "B"]))
    rows = []
    for fam in ("hp_early", "hp_late"):
        x = a[fam]
        rows.append([fam, _c(x.get("hp-0.3|Q")), _c(x.get("hp-0.3|G")), _c(x.get("price_per_bar|Q")),
                     _c(x.get("price_per_bar|G")), _c(x.get("ledger:hp-0.3:recalls@20")),
                     _c(x.get("ledger:hp-0.3:deaths@50")), _c(x.get("ledger:hp-0.3:away@50"))])
    s.append("\nPrice of HP (focal HP lowered by 0.3 bar, both continued by the policy):\n")
    s.append(_t(rows, ["family", "ΔQ50", "ΔG50", "price per bar (Q)", "price per bar (G)", "Δ Recalls @20",
                       "Δ deaths @50", "Δ turns away @50"]))
    rows = [[fam, a[fam]["validation_Q"].get("pairs"), _f(a[fam]["validation_Q"].get("bias")),
             _f(a[fam]["validation_Q"].get("slope")), a[fam]["validation_Q"].get("contradictions"),
             a[fam]["validation_Q"].get("verdict")] for fam in a]
    s.append("\nValidation (pooled, 300 turns):\n")
    s.append(_t(rows, ["family", "pairs", "bias", "slope", "contradictions", "verdict"]))
    n = {f: {k: v["n"] for k, v in d["states"][f].items()} for f in d["states"]}
    s.append(f"\nStates per final: {json.dumps(n)}\n")
    ext = (d.get("extended") or {}).get("analysis", {})
    for fam, x in ext.items():
        rows = [[k, _c(v)] for k, v in x.items() if k != "50_vs_150"]
        s.append(f"\nException 3: {fam} re-run at 150 turns (50 vs 150 on the same trajectories: "
                 f"{x.get('50_vs_150')}):\n")
        s.append(_t(rows, ["contrast | statistic and horizon", "value"]))
    return "\n".join(s)


E2_PLANS = ("farm_beh", "threaten_beh", "edge_beh", "bully_beh", "freeze_threat", "g_bully_script", "g_farm_script")


def e2_section() -> str:
    d = load_json("e2_summary")
    if not d:
        return "## E2\n\nnot run\n"
    a = d["analysis"]
    s = ["## E2 — selective threats as plans on the trained policy (20-turn plans, horizon 60; candidate − policy, "
         "t intervals over the six finals, (k+/k− of n) finals individually significant)\n"]
    stat = {"E2c": "G", "E2n": "Q", "E2s": "G"}
    for fam, title in (("E2c", "constructed zoning cells"), ("E2s", "semi-natural ×1.76 gap states (E3 games)"),
                       ("E2n", "natural HP-edge states")):
        if fam not in a:
            continue
        st = stat[fam]
        for proto in ("policy", "g_wary"):
            P = a[fam].get(proto, {})
            rows = []
            for sel, S in P.items():
                absg = S.get("abs:policy", {})
                r = [sel, fmt(absg.get("mean", float("nan")), absg.get("ci", float("nan")))]
                for c in E2_PLANS:
                    r.append(_c(S.get(f"{c}-policy|{st}")))
                rows.append(r)
            s.append(f"\n### {fam}: {title}, opponent {proto} (Δ{st}60)\n")
            s.append(_t(rows, ["gap / set", "policy's absolute G60"] + [f"{c} − policy" for c in E2_PLANS]))
            rows = []
            for sel, S in P.items():
                rows.append([sel, _c(S.get(f"threaten_beh-farm_beh|{st}")), _c(S.get(f"bully_beh-farm_beh|{st}")),
                             _c(S.get(f"g_bully_script-bully_beh|{st}")), _c(S.get(f"g_farm_script-farm_beh|{st}")),
                             "; ".join(f"{k} p={v:.3f}{'*' if S['holm']['rejected'][k] else ''}"
                                       for k, v in S["holm"]["p"].items())])
            s.append(_t(rows, ["gap / set", "threaten − farm_beh", "bully_beh − farm_beh",
                               "g_bully_script − bully_beh (wave confound)", "g_farm_script − farm_beh",
                               "Holm (4 selective plans vs policy; * = rejected)"]))
            focus = "x1.76" if "x1.76" in P else "all"
            S = P.get(focus)
            if S:
                rows = []
                for c in ("farm_beh", "threaten_beh", "edge_beh", "bully_beh", "g_bully_script"):
                    ch = S["channels"]
                    rows.append([c, f"{S['forced_turns'].get(c, float('nan')):.1f}"] +
                                [fmt(ch[f"{c}:{k}@50"]["mean"], ch[f"{c}:{k}@50"]["ci"]) for k in
                                 ("earned_A", "earned_B", "unc_B", "unc_away_B", "away_B", "deaths_B", "recalls_B",
                                  "deaths_A", "d_total_A", "trade_conn_A", "n_safe_B")])
                s.append(f"\nChannels at 50 turns ({focus}; candidate − policy):\n")
                s.append(_t(rows, ["plan", "forced turns", "A earned", "B earned", "B farm lost", "of which away",
                                   "B turns away", "B deaths", "B Recalls", "A deaths", "A HP lost (bars)",
                                   "A connected trades", "B Safe turns"]))
        v = a[fam].get("validation", {})
        s.append(f"\nValidation ({fam}, pooled, 300 turns): pairs {v.get('pairs')}, bias {_f(v.get('bias'))}, "
                 f"slope {_f(v.get('slope'))}, contradictions {v.get('contradictions')}: **{v.get('verdict')}**\n")
    ext = (d.get("extended") or {}).get("analysis", {})
    if ext:
        s.append(f"\n### Exception 3: E2c re-run at 150 turns (60 vs 150 on the same trajectories: "
                 f"{ext.get('60_vs_150')})\n")
        for proto in ("policy", "g_wary"):
            rows = []
            for gap, S in ext.get(proto, {}).items():
                rows.append([gap] + [_c(S.get(f"{c}|G150")) for c in
                                     ("threaten_beh-policy", "edge_beh-policy", "bully_beh-policy",
                                      "freeze_threat-policy", "farm_beh-policy", "threaten_beh-farm_beh",
                                      "g_bully_script-bully_beh", "g_farm_script-farm_beh")])
            s.append(f"\nOpponent {proto}, ΔG150:\n")
            s.append(_t(rows, ["gap", "threaten − policy", "edge − policy", "bully − policy",
                               "freeze + threat − policy", "Hold only − policy", "threaten − Hold only",
                               "g_bully − bully", "g_farm − Hold only"]))
    return "\n".join(s)


def e4_section() -> str:
    d = load_json("e4_summary")
    if not d:
        return "## E4\n\nnot run\n"
    a = d["analysis"]
    s = ["## E4 — best responses trained from gap and freeze starts (seat A: target / attacker / M4.5c s0-attacker; "
         "seat B: the target; per target: mean ± 95% CI over evaluation states; pooled: t interval over targets)\n"]
    tr = load_json("e4_training") or {}
    if tr.get("runs"):
        rows = [[k, v["updates"], f"{v['train_s'] / 60:.0f} min", f"{v['steps'] / max(v['train_s'], 1):.0f}",
                 f"{v['entropy_first_last'][0]:.2f} → {v['entropy_first_last'][1]:.2f}", f"{v['kl_max']:.4f}",
                 v["finite"]] for k, v in sorted(tr["runs"].items())]
        s.append(_t(rows, ["run", "updates", "time", "learner transitions/s", "entropy", "max KL", "finite"]))
    for fam, F in sorted(a.items()):
        if not fam.startswith("E4"):
            continue
        kind = "gap" if fam.startswith("E4gap") else "freeze"
        rows = []
        for key in sorted(k for k in F if "|G" in k and not k.startswith("abs")):
            c = F[key]
            per = "; ".join(f"s{t}: {fmt(x['mean'], x['ci'])}{'*' if x['sig'] else ''}"
                            for t, x in sorted(c["per_target"].items()))
            rows.append([key, fmt(c["mean"], c["ci"]), f"{c['k_pos']}+/{c['k_neg']}−", per])
        s.append(f"\n### {fam}\n")
        s.append(_t(rows, ["contrast", "pooled", "targets significant", "per target (* = significant)"]))
        absrows = []
        for key in sorted(k for k in F if k.startswith("abs|")):
            for t, v in sorted(F[key].items()):
                absrows.append([key[4:], t] + [_f(v[c]) for c in ("target", "attacker", "s0att")])
        s.append(_t(absrows, ["horizon", "target", "absolute: target in seat A", "attacker in seat A",
                              "s0-attacker in seat A"]))
        ch = F.get("channels", {})
        rows = [[k, fmt(v["mean"], v.get("ci", float("nan")))] for k, v in sorted(ch.items())]
        s.append(_t(rows, ["channel (candidate − target, or the target's absolute level)", "value"]))
    s0 = a.get("s0_eval", {})
    if s0:
        rows = [[k, fmt(v["estimate"], 1.96 * v["stderr"]), _f(v["seat_A"]), _f(v["seat_B"])] for k, v in sorted(s0.items())]
        s.append("\nFrom s0 against the target (evaluate_discounted, 64 paired seeds, W = 460):\n")
        s.append(_t(rows, ["attacker", "value", "seat A", "seat B"]))
    return "\n".join(s)


def exploratory_section() -> str:
    s = ["## Exploratory measurements (added after pre-registered results; labelled)\n"]
    p = load_json("e3_positions")
    if p:
        rows = []
        for k, v in sorted(p["results"].items()):
            st, w = v["stronger"], v["weaker"]
            rows.append([k, st["turns"], _f(st["mean_all"]), _f(st["mean_lane"]), f"{st['share_W']:.3f}",
                         _f(w["mean_all"]), _f(w["mean_lane"]), f"{w['share_W']:.3f}"])
        s.append("### Position readout of the M4.5c power-gap stage (Ward node W = 99 included vs lane cells only)\n")
        s.append(_t(rows, ["policy", "stronger turns", "stronger: M4.5c metric", "stronger: lane cells", "stronger at W",
                           "weaker: M4.5c metric", "weaker: lane cells", "weaker at W"]))
    n = load_json("e3_natural_gaps")
    if n:
        rows = []
        for k, v in sorted(n["results"].items()):
            rows.append([k, v["turns"], f"{v['quantiles']['0.5']:.3f}", f"{v['quantiles']['0.99']:.3f}",
                         f"{v['quantiles']['0.999']:.3f}", f"{v['share']['1.25']:.4f}", f"{v['share']['1.5']:.4f}",
                         " / ".join(f"{x['mean_ratio']:.3f}" for x in v["by_lead"].values()),
                         " / ".join(f"{x['share_1.25']:.4f}" for x in v["by_phase"].values())])
        s.append("\n### Natural power gaps in s0 self-play (16 games × 600 turns per final)\n")
        s.append(_t(rows, ["final", "turns (both present)", "median ratio", "99%", "99.9%", "share ≥ 1.25",
                           "share ≥ 1.5", "mean ratio by |gold lead| 0-3 / 3-6 / 6-10 / 10-20 / 20+",
                           "share ≥ 1.25 by turns 1-50 / 51-150 / 151-300 / 301-600"]))
    e4 = load_json("e4_summary") or {}
    for kind, T in (e4.get("exploratory") or {}).items():
        a = T["analysis"]
        rows = [[f"attacker of s{t}", fmt(v["mean"], v["ci"]), f"{v['k_pos']}+/{v['k_neg']}−",
                 "; ".join(f"s{o}: {fmt(x['mean'], x['ci'])}" for o, x in sorted(v["per_opponent"].items()))]
                for t, v in sorted(a.items()) if isinstance(v, dict)]
        s.append(f"\n### {kind}: attackers against the finals they never trained against (G100, attacker − that final's "
                 "own play in seat A)\n")
        s.append(_t(rows, ["attacker", "mean over the 5 opponents", "opponents significant", "per opponent"]))
        s.append("Channels (100 turns, pooled): " + ", ".join(f"{k.split(':')[1]} {v:+.2f}" for k, v in a.items()
                                                              if not isinstance(v, dict)) + "\n")
    return "\n".join(s)


def e6_section() -> str:
    d = load_json("e6_summary")
    if not d:
        return "## E6\n\nnot run\n"
    s = ["## E6 — credit assignment\n"]
    rows = []
    for k, v in sorted(d["snr_e2"].items()):
        rows.append([k, v["states"], _f(v["effect"]), f"{v['rollout_sd']:.2f}", f"{v['snr']:.3f}",
                     f"{v['forced_turns']:.1f}", _f(v["effect_per_forced_turn"], 3), f"{v['rollouts_for_2se']:.0f}"])
    s.append(_t(rows, ["family | opponent | plan", "states", "mean effect (G60)", "SD of one paired rollout",
                       "|effect| / SD", "forced turns", "effect per forced turn", "paired rollouts for 2 SE"]))
    crit = d.get("critic_e3") or {}
    rows = []
    for f, byc in sorted(crit.items()):
        for c, v in byc.items():
            rows.append([f, c, v["n"], _f(v["bias"]), f"{v['mae']:.2f}", f"{v['rmse']:.2f}", f"{v['target_sd']:.2f}",
                         f"{v['explained_variance']:.2f}"])
    s.append("\nCritic error on E3 games (V(s) − 100-turn bootstrapped return), by the seat's turn class:\n")
    s.append(_t(rows, ["final", "turn class", "n", "bias", "MAE", "RMSE", "SD of target", "explained variance"]))
    cg = d.get("critic_by_gap") or {}
    rows = [[k, v["n"], _f(v["bias"]), f"{v['rmse']:.2f}", f"{v['explained_variance']:.3f}"] for k, v in cg.items()]
    s.append("\nCritic error by power gap and role (EXPLORATORY; finals and levels pooled):\n")
    s.append(_t(rows, ["gap | role | turn class", "n", "bias", "RMSE", "explained variance"]))
    return "\n".join(s)


__all__ = ["run"]
