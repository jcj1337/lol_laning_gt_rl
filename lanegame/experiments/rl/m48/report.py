"""Generated tables of M4.8 (results/m48/report.md) from pilot.json, validation.json and application.json."""

from __future__ import annotations

import numpy as np

from . import common as C

H = (5, 10, 20, 40)


def f(x, d: int = 2, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def mci(st: dict, d: int = 2) -> str:
    if not st or not np.isfinite(st.get("mean", np.nan)):
        return "n/a"
    h = st.get("ci", np.nan)
    return f"{st['mean']:+.{d}f} ± {h:.{d}f}" if np.isfinite(h) else f"{st['mean']:+.{d}f}"


def pct(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def cell(c) -> str:
    """A table cell with its pipes escaped (|lead| would split the markdown row)."""
    return str(c).replace("|", "\\|")


def table(head, rows) -> str:
    out = ["| " + " | ".join(cell(h) for h in head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def seedline(ps: dict, key: str, d: int = 2, as_pct: bool = False) -> list:
    o = []
    for k in ("baseline", "C1", "C1-baseline"):
        st = ps[k]
        if as_pct:
            m, h = 100 * st["mean"], 100 * st["ci"]
            o.append(f"{m:.0f}% ± {h:.0f}" if k != "C1-baseline" else f"{m:+.0f} ± {h:.0f} pts")
        else:
            o.append(mci(st, d))
    return o


def run() -> str:
    P, V, A = C.load("pilot"), C.load("validation"), C.load("application")
    pr = C.load("protocol")
    L = ["# M4.8 lane-exit evaluation: generated tables", "",
         f"Protocol `{pr['record']['document']}` SHA-256 `{pr['record']['sha256'][:16]}…` (frozen "
         f"{pr['meta']['timestamp']}). Generated from `results/m48/pilot.json`, `validation.json`, "
         "`application.json`. Resources: 1 = one fresh wave. Seat orientation: case tables are folded to the focal "
         "seat X; natural-state tables are in seat A's orientation unless folded to the category's X.", ""]
    # ---- pilot
    L += ["## Pilot (throughput and within-state spread)", "",
          f"{P['states']} states at turn 94, K {P['K']}: {P['turns_per_s_worker']:.0f} turns/s per worker, "
          f"{P['turns_per_s_wall']:.0f} on the wall clock.", ""]
    L += [table(["h", "SD of Δ_h p25 / p50 / p75 / p90", "K for ±0.25 (p50)", "K for ±0.5 (p50)"],
                [[h, " / ".join(f"{P['sd'][str(h)]['quantiles'][str(q)]:.2f}" for q in (25, 50, 75, 90)),
                  P["sd"][str(h)]["K_for_0.25"]["50"], P["sd"][str(h)]["K_for_0.5"]["50"]] for h in C.WINDOWS]), ""]
    # ---- checks
    cl = A["classification"]
    L += ["## Checks (V1-V3)", "",
          table(["check", "result"],
                [["V1 accounting: lead(final) − D0 = Σ reward units, every continuation",
                  f"validation {V['V1_accounting']['continuations']:,} continuations, max residual "
                  f"{V['V1_accounting']['max_abs_units']:.0f} units; natural max {cl['natural_acct_max']:.0f}"],
                 ["V2 orientation: mirrored continuation = exact negation",
                  f"{V['V2_mirror']['states']} states, {V['V2_mirror']['continuations']:,} continuations, max "
                  f"|r + r_mirror| = {V['V2_mirror']['max_abs_r_sum']}"],
                 ["V3 restoration: natural checkpoints replay their records",
                  f"{A['n_states']} checkpoints, every turn verified against the M4.6d records"]]), ""]
    # ---- Stage 1 cases
    L += ["## Stage 1: validation cases (folded to the focal seat X)", ""]
    rows = []
    for key, cs in V["cases"].items():
        case, x = key.split("|")
        if x != "both" and case not in ("stable", "due"):
            continue
        rows.append([case, cs["n"], f"{cs['n_by_env']['baseline']}/{cs['n_by_env']['C1']}", mci(cs["lead_x"])]
                    + [mci(cs[f"d{h}"]) for h in H]
                    + [f"{pct(cs['material10'])} / {pct(cs['material20'])}",
                       f"{pct(cs['res<=10'])} / {pct(cs['res<=20'])}", mci(cs["rec_d10"])])
    L += [table(["case", "n", "baseline/C1", "X's raw lead D0", "Δ5", "Δ10", "Δ20", "Δ40", "material @10 / @20",
                 "pending resolved by 10 / 20", "recorded Δ10 (one sample per state)"], rows), ""]
    L += ["Orientation check (case X = A against X = B):", ""]
    rows = []
    for case in ("returner", "lost", "denied", "opposing"):
        a, b = V["cases"].get(f"{case}|A"), V["cases"].get(f"{case}|B")
        if a and b:
            rows.append([case, f"{a['n']} / {b['n']}", mci(a["d10"]), mci(b["d10"]), mci(a["d20"]), mci(b["d20"])])
    L += [table(["case", "n (X=A / X=B)", "Δ10 X=A", "Δ10 X=B", "Δ20 X=A", "Δ20 X=B"], rows), ""]
    for h in (10, 20):
        rows = []
        for key, cs in V["cases"].items():
            case, x = key.split("|")
            if x != "both" and case not in ("stable", "due"):
                continue
            c = cs[f"components{h}"]
            rows.append([case, f"{c['farm']['X']:.2f} / {c['farm']['Y']:.2f}",
                         f"{c['avail']['X']:.2f} / {c['avail']['Y']:.2f}",
                         f"{c['lost_away']['X']:.2f} / {c['lost_away']['Y']:.2f}", f(c["bounty"]["X-Y"]),
                         f(c["plate"]["X-Y"]), f"{c['returned']['X']:.2f} / {c['returned']['Y']:.2f}",
                         f"{c['away']['X']:.2f} / {c['away']['Y']:.2f}", f"{c['died']['X']:.2f} / {c['died']['Y']:.2f}"])
        L += [f"Event accounting over the first {h} turns (means per continuation; X / Y):", "",
              table(["case", "farm collected", "farm available", "farm lost while away", "bounty X−Y", "plates X−Y",
                     "returns", "turns away", "deaths"], rows), ""]
    rows = []
    for key, cs in V["cases"].items():
        case, x = key.split("|")
        if x != "both" and case not in ("stable", "due"):
            continue
        rows.append([case, mci(cs["xp_lead_x"], 1), mci(cs["xpd10"], 1), mci(cs["xpd20"], 1), mci(cs["hp_lead_x"]),
                     mci(cs["hpd10"]), f"{pct(cs['retX<=10'])} / {pct(cs['retY<=10'])}",
                     f"{pct(cs['retX<=20'])} / {pct(cs['retY<=20'])}", f"{cs['stkX_med']:.0f} / {cs['stkY_med']:.0f}",
                     f"{cs['got_x']['mean']:.2f} / {cs['got_y']['mean']:.2f}"])
    L += ["Non-gold dimensions and resolution (X's view):", "",
          table(["case", "XP lead at t0", "XP lead +10", "XP lead +20", "HP lead at t0 (both present)", "HP lead +10",
                 "back by 10 (X / Y)", "back by 20 (X / Y)", "stock gone, median turn (X / Y)",
                 "stock collected share (X / Y)"], rows), ""]
    # ---- variants
    vs = V["variants"]
    L += ["## Stage 1: controlled variants on due states (seat A; D0 unchanged by construction)", "",
          f"{vs['n']} states; A's unspent gold {mci(vs['unspent_A'])}, HP {mci(vs['hp_A'])}, turns since return "
          f"{mci(vs['since_return_A'], 1)}.", ""]
    rows = []
    for v in ("A_fresh", "A_away1"):
        rows.append([v] + [mci(vs[v][f"pd{h}"]) for h in H] + [pct(vs[v]["material10"]),
                                                                 f"{vs[v]['pd10_by_env']['baseline']['mean']:+.2f} / "
                                                                 f"{vs[v]['pd10_by_env']['C1']['mean']:+.2f}"])
    L += [table(["variant − orig", "E5", "E10", "E20", "E40", "material @10", "@10 baseline / C1"], rows), ""]
    # ---- worked example
    ex = V.get("worked_example") or {}
    if ex:
        d = ex["desc"]
        x = ex["X"]
        y = "B" if x == "A" else "A"
        L += ["## Worked example (the first returner state whose returner trails by < 2)", "",
              f"{ex['env']} seed {ex['s']}, validation game {ex['g']}, turn {ex['t0']}; X = seat {x}.", "",
              table(["", "X", "Y"],
                    [["status (0 present, 1 Recall, 2 dead)", d[f"status_{x}"], d[f"status_{y}"]],
                     ["turns still away", d[f"returns_in_{x}"], d[f"returns_in_{y}"]],
                     ["turns since last return", d[f"since_return_{x}"], d[f"since_return_{y}"]],
                     ["HP", f(d[f"hp_{x}"], 2, False), f(d[f"hp_{y}"], 2, False)],
                     ["level / XP", f"{d[f'level_{x}']:.0f} / {d[f'xp_{x}']:.0f}",
                      f"{d[f'level_{y}']:.0f} / {d[f'xp_{y}']:.0f}"],
                     ["earned (invested + unspent)", f"{d[f'earned_{x}']:.2f} ({d[f'invested_{x}']:.2f} + "
                      f"{d[f'unspent_{x}']:.2f})", f"{d[f'earned_{y}']:.2f} ({d[f'invested_{y}']:.2f} + "
                      f"{d[f'unspent_{y}']:.2f})"],
                     ["enemy minion HP alive on its ledger (at its own −1 or deeper)",
                      f"{d[f'stock_{x}']:.0f} ({d[f'near_{x}']:.0f})", f"{d[f'stock_{y}']:.0f} ({d[f'near_{y}']:.0f})"]]),
              ""]
        L += [table(["h", "E_h (X's lead)", "95% ±", "Δ_h", "pending resolved", "recorded change (one sample)",
                     "XP lead at h"],
                    [[h, f(v["E"]), f(v["ci"], 2, False), f(v["d"]), pct(v["res"]), f(v["rec_d"]), f(v["xpd"], 1)]
                     for h, v in ex["readout"].items()]), ""]
        rows = []
        for h, c in ex["components"].items():
            rows.append([h] + [f"{c[k]['X']:.2f} / {c[k]['Y']:.2f}" for k in ("farm", "avail", "lost_away", "bounty",
                                                                              "plate", "returned", "away")])
        L += [table(["h", "farm", "available", "lost away", "bounty", "plate", "returns", "turns away"], rows), ""]
        L += ["The game's recorded continuation (one realized sample):", "",
              table(["turn", "X's lead change", "farm X", "farm Y", "away X", "away Y", "events"],
                    [[t["turn"], f(t["lead_change_x"]), f(t["farm_X"], 2, False), f(t["farm_Y"], 2, False),
                      "yes" if t["away_X"] else "", "yes" if t["away_Y"] else "", ", ".join(t["events"])]
                     for t in ex["recorded"][:15]]), ""]
    # ---- calibration
    cal = V.get("V6_calibration")
    if cal:
        L += ["## V6: predictions against the recorded continuations (natural states, turn 94)", ""]
        L += [table(["h", "slope of realized on predicted", "mean residual", "R² of prediction", "var realized",
                     "var predicted", "MSE of D0 / E_h for the lead at t0+h"],
                    [[h, mci({"mean": c["slope"]["slope"], "ci": c["slope"]["ci"]}), mci(c["mean_resid"]),
                      f(c["r2_pred"], 3, False), f(c["var_real"], 2, False), f(c["var_pred"], 2, False),
                      f"{c['mse_D0']:.2f} / {c['mse_E']:.2f}"] for h, c in ((h, cal[str(h)]) for h in C.WINDOWS)]),
              ""]
        rows = []
        for t, o in cal["later"].items():
            rows.append([t, o["n"], f(o["corr_D0"], 3, False)] + [f(o[f"corr_E{h}"], 3, False) for h in H]
                        + [mci(o[f"dmse_E{h}"]) for h in (10, 20)])
        L += ["Does E_h predict the later recorded lead better than D0?", "",
              table(["turn", "n", "corr D0", "corr E5", "corr E10", "corr E20", "corr E40",
                     "ΔMSE E10 − D0", "ΔMSE E20 − D0"], rows), ""]
        for t in ("84", "104"):
            c2 = V.get(f"V6_calibration_t{t}")
            if c2:
                L += [f"Calibration at turn {t}: " + "; ".join(
                    f"h {h} slope {c2[str(h)]['slope']['slope']:+.2f} ± {c2[str(h)]['slope']['ci']:.2f}, residual "
                    f"{c2[str(h)]['mean_resid']['mean']:+.2f}" for h in (5, 10, 20)), ""]
    # ---- reversals
    L += ["## V8: material leader reversals and their accounting", ""]
    for h, rv in V["reversals"].items():
        L += [f"Validation states, h {h}: {rv['n']} of {rv['n_states']} ({pct(rv['share'])}); dominant component "
              + ", ".join(f"{k} {v}" for k, v in sorted(rv["dominant"].items(), key=lambda z: -z[1]))
              + "; by case " + ", ".join(f"{k} {v}" for k, v in sorted(rv["by_label"].items(), key=lambda z: -z[1])),
              ""]
    for h, rv in A.get("reversals", {}).items():
        L += [f"Natural turn-94 states, h {h}: {rv['n']} of {rv['n_states']} ({pct(rv['share'])}); dominant "
              + ", ".join(f"{k} {v}" for k, v in sorted(rv["dominant"].items(), key=lambda z: -z[1]))
              + "; by category " + ", ".join(f"{k} {v}" for k, v in sorted(rv["by_label"].items(),
                                                                          key=lambda z: -z[1])), ""]
    # ---- Stage 2 S1
    S1 = A["S1"]
    L += ["## Stage 2 S1: how often raw checkpoint gold gives a materially different assessment (turn 94)", "",
          f"Levels at turn 94: lower level mean {S1['levels_at_94']['min_level_mean']:.1f}; both ≥ 14 in "
          f"{pct(S1['levels_at_94']['share_both_ge14'])}, either in {pct(S1['levels_at_94']['share_any_ge14'])}; "
          f"seat levels p10/p50/p90 " + " / ".join(f"{v:.0f}" for v in S1["levels_at_94"]["level_quantiles"].values())
          + ".", ""]
    rows = []
    for h in C.WINDOWS:
        o = S1[str(h)]
        rows.append([h] + seedline(o["material"], "", as_pct=True) + [pct(o["pooled"]["reversal"]),
                                                                      mci(o["true_sd"]["pooled"]),
                                                                      mci(o["mean_abs_d"]["pooled"]),
                                                                      f(o["halfwidth_median"], 2, False)])
    L += [table(["h", "material share baseline", "C1", "C1 − baseline", "leader reversals (pooled)",
                 "SD of the true adjustment", "mean |Δ_h|", "median 95% ±"], rows), ""]
    rows = []
    for h in C.WINDOWS:
        o = S1[str(h)]
        rows.append([h, pct(o["pooled"]["material"]), pct(o["material_point"]["pooled"]["mean"]),
                     pct(o["material_0.25"]["pooled"]["mean"]), pct(o["material_1.0"]["pooled"]["mean"]),
                     " / ".join(f"{o['pooled']['quantiles_d'][q]:+.2f}" for q in ("5", "25", "50", "75", "95")),
                     pct(o["pooled"]["near_even_decided"])])
    L += [table(["h", "material (≥0.5, certain)", "point ≥0.5", "certain ≥0.25", "certain ≥1.0",
                 "Δ_h p5 / p25 / p50 / p75 / p95", "states with |D0| < 0.5 that E_h decides (share of all states)"], rows), ""]
    # ---- S2
    S2 = A["S2"]
    L += ["## Stage 2 S2: which situations explain the adjustments (turn 94, folded to the category's X)", ""]
    rows = []
    for cat, o in S2["categories"].items():
        if not o["n"]:
            rows.append([cat, 0] + [""] * 9)
            continue
        rows.append([cat, o["n"], pct(o["share"]), mci(o["lead_x"])] + [mci(o[f"d{h}"]) for h in H]
                    + [pct(o["material10"]), mci(o["drift40-10"]), f"{pct(o['res<=10'])} / {pct(o['res<=20'])}"])
    L += [table(["category", "n", "share", "X's D0", "Δ5", "Δ10", "Δ20", "Δ40", "material @10", "drift Δ40 − Δ10",
                 "resolved by 10 / 20"], rows), ""]
    rows = []
    for cat, o in S2["categories"].items():
        if o["n"] and "components10" in o:
            c = o["components10"]
            rows.append([cat] + [f(c[k]) for k in ("farm", "avail", "lost_away", "bounty", "plate", "returned",
                                                    "away")])
    L += ["Folded event accounting over 10 turns (X − Y):", "",
          table(["category", "farm", "available", "lost away", "bounty", "plates", "returns", "turns away"], rows), ""]
    rows = [[h, f(e["noise_share"], 2, False)] + [f"{e['cv_' + k]:.2f} ({e[k]:.2f})" for k in
                                                   ("pending", "strength", "lead", "pending+lead", "all")]
            for h, e in S2["explained"].items()]
    L += ["Share of the variance of Δ̂_h across states explained: 5-fold cross-validated R² (in-sample in brackets); "
          "the rollout-noise share bounds what any model can explain:", "",
          table(["h", "rollout-noise share", "pending features", "strength features", "D0", "pending + D0",
                 "all"], rows), ""]
    rows = [[h] + [f"{e['cv_' + k]:.2f}" for k in ("reset", "wave", "hp", "power", "reset+wave+hp")]
            for h, e in S2["explained"].items()]
    L += ["Finer split (descriptive, not pre-registered; cross-validated R²): HP acts mostly through who has to reset "
          "next, so it is shown on its own:", "",
          table(["h", "reset status", "wave", "HP", "investment and level", "reset + wave + HP"], rows), ""]
    rows = [[case, o["n"]] + ([mci(o["d10"]), mci(o["d20"]), pct(o["material10"])] if o["n"] else ["", "", ""])
            for case, o in S2["cases_at_t0"].items()]
    L += ["Stage 1's case criteria among the natural turn-94 states (selected subsets, folded to X):", "",
          table(["case", "n", "Δ10", "Δ20", "material @10"], rows), ""]
    # ---- S3
    S3 = A["S3"]
    L += ["## Stage 2 S3: stability", "",
          table(["windows", "mean |change|", "corr", "class changes (all)", "class changes among material",
                 "leader changes"],
                [[k, f(o["mean_abs_change"], 2, False), f(o["corr"], 2, False), pct(o["class_changes"]),
                  pct(o["class_changes_among_material"]), pct(o["leader_changes"])] for k, o in S3["windows"].items()]),
          ""]
    rows = []
    for pre, o in S3["policies"].items():
        for h in (5, 10, 20, 40):
            x = o[str(h)]
            rows.append([o["label"], o["n"], h, mci(x["mean_diff"]), f(x["mean_abs_diff"], 2, False),
                         f(x["corr"], 2, False), pct(x["class_changes_among_material"]), pct(x["leader_changes"]),
                         pct(x["material_signif_diff"])])
    L += ["Continuation policies (the same states, common random numbers):", "",
          table(["alternative", "n", "h", "alt − own", "mean |alt − own|", "corr", "class changes among material",
                 "leader changes", "states with a material, certain difference"], rows), ""]
    rows = []
    for t0, o in S3["anchors"].items():
        rows.append([t0, o["n"], f"{o['K']:.0f}", f(o["level_min_mean"], 1, False)]
                    + [f"{100 * o[str(h)]['material']['pooled']['mean']:.0f}% / {o[str(h)]['true_sd']['pooled']['mean']:.2f}"
                       for h in (5, 10, 20, 40)])
    L += ["Anchors (material share at that anchor's K / SD of the true adjustment, which does not depend on K):", "",
          table(["t0", "n", "K", "lower level", "h 5", "h 10", "h 20", "h 40"], rows), ""]
    L += ["New consequences after h (resets, deaths and Recalls in turns h+1..40, both seats, per continuation): "
          + "; ".join(f"h {h}: returns {o['returned']:.2f}, deaths {o['died']:.2f}" for h, o in
                      S3["new_after"].items()), ""]
    # ---- window rule
    W = A["window_rule"]
    rows = []
    for h, c in W["candidates"].items():
        rows.append([h, f"{pct(c['coverage'])} ({'met' if c['a'] else 'not met'})",
                     ", ".join(f"{k} {pct(v)}" for k, v in c["coverage_by_kind"].items()),
                     ("met" if c["b"] else "fails: " + ", ".join(c["b_fails"])) + "; " + ", ".join(
                         f"{k} {v['excess_drift']:+.2f}" for k, v in c["drift"].items()),
                     f"{c['halfwidth_median']:.2f} ({'met' if c['c'] else 'not met'})"])
    L += ["## Default window rule (protocol §6)", "",
          table(["h", "(a) coverage ≥ 80%", "by kind (absent / due / stock)", "(b) excess drift ≤ 0.25",
                 "(c) median ± ≤ 0.5"], rows), "",
          f"**Result: {W['recommendation']}.** Largest share of material classifications changed between policies "
          f"or windows 10 → 20: {pct(W['_policy_class_changes'])}.", ""]
    # ---- S4
    S4 = A["S4"]
    L += ["## Stage 2 S4: baseline and C1 (six training seeds per curve; C1 − baseline paired by seed)", ""]
    rows = []
    for k, o in S4["distributions"].items():
        pctk = k.startswith("share_") or k == "away_share"
        rows.append([k] + seedline(o, k, as_pct=pctk))
    L += ["(a) State distributions at turn 94 (each curve's own games):", "",
          table(["quantity", "baseline", "C1", "C1 − baseline"], rows), ""]
    L += ["(b) Lead conversion under each package's own continuation (slope of Δ_h on D0):", "",
          table(["h", "baseline", "C1", "C1 − baseline"], [[h] + seedline(S4["conversion_own"][str(h)], "")
                                                            for h in C.WINDOWS]), ""]
    rows = []
    for h in C.WINDOWS:
        o = S4["matched_states"][str(h)]
        rows.append([h, mci(o["leader_gain"]), mci(o["slope_diff"]), mci(o["mean_diff"]),
                     mci(o["leader_gain_states_from_baseline"]), mci(o["leader_gain_states_from_C1"])])
    L += ["(c) Matched states: the same turn-94 states continued by the C1 package and by the baseline package "
          "(environment and its trained policy; not the isolated effect of the investment curve):", "",
          table(["h", "leader's extra gain under C1 (seeds)", "slope difference (seeds)", "mean difference",
                 "leader gain, baseline-origin states", "leader gain, C1-origin states"], rows), ""]
    m = S4["m47_recheck"]
    rows = [["|lead| at turn 94, raw D0 (M4.7: 3.29 / 3.64 / +0.35 ± 0.40)"] + seedline(m["abs_lead_94"]["D0"], "")]
    rows += [[f"|E_{h}| at turn 94"] + seedline(m["abs_lead_94"][f"E{h}"], "") for h in (5, 10, 20)]
    rows += [["conversion slope 50 → 94, raw D0 (M4.7: −0.20 / +0.04 / +0.24 ± 0.15)"]
             + seedline(m["slope_50_94"]["D0"], "")]
    rows += [[f"conversion slope 50 → 94 with E_{h} at both ends"] + seedline(m["slope_50_94"][f"E{h}"], "")
             for h in (5, 10, 20)]
    L += ["(d) Earlier conclusions re-checked (M4.7 §2):", "",
          table(["quantity", "baseline", "C1", "C1 − baseline"], rows), ""]
    rc = C.load("recheck")
    if rc:
        rows = []
        for s, o in rc["per_seed"].items():
            rows.append([s, o["n_games"], mci(o["P1_raw"])] + [mci(o[f"P1_E{h}"]) for h in (5, 10, 20)]
                        + [mci(o["P1_E10-raw"]), f"{o['max_replay_dev']:.1g}"])
        L += [f"(d) M4.7's pilot primary P1 (the short arm's lane lead against the control, head to head; M4.7: "
              f"+0.26 ± 0.17 and −0.09 ± 0.17) with the lane-exit readout at turn 94 (supplementary, protocol "
              f"amendment; K {rc['K']}, {rc['T']}-turn continuations by the same pair):", "",
              table(["seed", "games", "raw P1 (lead at 94)", "P1 with E5", "with E10", "with E20", "E10 − raw",
                     "replay deviation"], rows), ""]
    fr = S4["fresh_variant"]
    L += [f"Controlled: E_h(A_fresh) − E_h(orig) on the turn-94 states where A is due "
          f"(n baseline {fr['n']['baseline']}, C1 {fr['n']['C1']}):", "",
          table(["h", "baseline", "C1", "C1 − baseline"], [[h] + seedline(fr[str(h)], "") for h in H]), ""]
    # ---- classification
    L += ["## Method classification (protocol §7)", "",
          table(["criterion", "result"],
                [["V1–V3 exact", cl["exact_checks"]],
                 ["calibration", "; ".join(f"h {h}: slope interval [{c['slope_interval'][0]:.2f}, "
                                           f"{c['slope_interval'][1]:.2f}], residual {c['mean_resid']['mean']:+.2f}"
                                           for h, c in cl["calibration"].items()) + f" → {cl['calibration_ok']}"],
                 ["added information", "; ".join(f"h {h}: material {pct(v['material_share'])}, true SD "
                                                 f"{v['true_sd']:.2f}" for h, v in cl["added_information"].items())
                  + f" → {cl['adds']}"],
                 ["median half-widths", ", ".join(f"h {h}: {v:.2f}" for h, v in cl["halfwidths"].items())],
                 ["**label**", f"**{cl['label']}**"]]), "", f"_{cl['note']}_", ""]
    txt = "\n".join(L)
    C.write_md("report", txt)
    return txt


__all__ = ["run"]
