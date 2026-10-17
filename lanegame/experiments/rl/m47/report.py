"""Generated tables of M4.7 (results/m47/report.md) from the stage outputs in results/m47/*.json.

Stage 1 intervals are 95% t intervals over the six training seeds (5 degrees of freedom); C1 - baseline is paired by
seed. Stage 2 intervals are within-seed 95% normal intervals over games (two seeds, reported separately).
"""

from __future__ import annotations

from typing import List

import numpy as np

from . import common as C


def _f(x, d=2, sign=False) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _ci(st: dict, d=2, sign=True) -> str:
    if not st or not np.isfinite(st.get("mean", np.nan)):
        return "n/a"
    h = st.get("ci", np.nan)
    m = f"{st['mean']:+.{d}f}" if sign else f"{st['mean']:.{d}f}"
    return f"{m} ± {h:.{d}f}" if np.isfinite(h) else m


def _table(head: List[str], rows: List[List[str]]) -> List[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


# ---------------------------------------------------------------------------------------------- Stage 1
def stage1() -> List[str]:
    L: List[str] = ["## Stage 1: what happens before level 14", ""]
    tl = C.load("timeline")
    if tl:
        A = tl["aggregate"]
        b, c, dd = A["by_env"]["baseline"], A["by_env"]["C1"], A["diff_C1_minus_baseline"]
        L += ["### Turns to a level (natural self-play from s0, matched M4.6c 8.192M policies; 48 games x 6 seeds)", ""]
        rows = []
        for lv in C.LEVEL_MARKS:
            k = f"timeline/{lv}/pooled"
            rows.append([f"L{lv}", _ci(b[f"{k}/mean"], 1, False), f"{_f(b[f'{k}/p10']['mean'], 0)}-"
                         f"{_f(b[f'{k}/p90']['mean'], 0)}", _ci(b[f"timeline/{lv}/seat_A/mean"], 1, False),
                         _ci(b[f"timeline/{lv}/seat_B/mean"], 1, False),
                         _ci(b[f"timeline/{lv}/both_seats/mean"], 1, False),
                         _ci(c[f"{k}/mean"], 1, False), _ci(dd[f"{k}/mean"], 2)])
        L += _table(["level", "baseline mean", "p10-p90", "seat A", "seat B", "both seats", "C1 mean",
                     "C1 - baseline"], rows)
        lin = tl.get("lineage", {})
        if lin:
            rows = [[k, *[_ci(v["stats"].get(f"timeline/{lv}/pooled/mean"), 1, False) for lv in (10, 12, 14, 18)],
                     _ci(v["stats"].get("at_level/14/invested"), 1, False)] for k, v in lin.items()]
            L += ["Lineage (M4.6c natural records, other seeds, 600 turns): start = the M4.5c 16.384M finals; "
                  "baseline / C1 = the 4.096M continuations; ext_* = the 8.192M continuations.", ""]
            L += _table(["policies", "L10", "L12", "L14", "L18", "invested at L14"], rows)
        L += ["### The state when a seat reaches a level", ""]
        rows = []
        for lv in (10, 12, 14, 18):
            k = f"at_level/{lv}"
            rows.append([f"L{lv}"] + [_ci(b[f"{k}/{m}"], 2, False) for m in
                                      ("turn", "earned", "invested", "unspent", "hp", "deaths", "recalls",
                                       "interrupted", "earned_lead_abs")] + [_ci(dd[f"{k}/invested"], 2)])
        L += _table(["level", "turn", "earned", "invested", "unspent", "HP", "deaths", "Recalls done",
                     "interrupted", "abs earned lead", "C1 - baseline invested"], rows)
        L += ["### Common elapsed-turn windows (baseline; C1 - baseline below each metric)", ""]
        wins = list(C.WINDOWS)
        rows = []
        for m in ("earned_per_turn", "farm_per_turn", "bounty_per_turn", "plate_per_turn", "deaths_per_100",
                  "recall_choices_per_100", "recalls_per_100", "interrupted_per_100", "hp_present", "away_share",
                  "wave_shove", "wave_last_hit", "wave_thin", "wave_slow_push", "beh_hold", "beh_yield", "beh_trade",
                  "beh_safe", "beh_recall", "ward_share", "pressure_share", "held_share", "frozen_on_shove",
                  "frozen_on_safe", "trade_connect_per_100", "unspent", "invested"):
            rows.append([m] + [_f(b[f"windows/{w}/{m}"]["mean"], 3) for w in wins])
            rows.append([f"  C1 - baseline"] + [_ci(dd[f"windows/{w}/{m}"], 3) for w in wins])
        L += _table(["metric"] + wins, rows)
        L += ["### Gold leads (seat A's cumulative earned-gold lead)", ""]
        rows = []
        for t in ("25", "50", "75", "94", "150", "250", "399"):
            k = f"leads/{t}"
            rows.append([t, _ci(b[f"{k}/abs_mean"], 2, False), _ci(c[f"{k}/abs_mean"], 2, False),
                         _ci(b[f"{k}/rel_abs_mean"], 3, False), _ci(b[f"{k}/invested_lead_sd"], 2, False),
                         _ci(dd[f"{k}/abs_mean"], 2)])
        L += _table(["turn", "abs lead baseline", "abs lead C1", "abs lead / earned (baseline)",
                     "SD invested lead (baseline)", "C1 - baseline"], rows)
        rows = [[p, _ci(b[f"leads/conversion/{p}/slope"], 3), _ci(c[f"leads/conversion/{p}/slope"], 3),
                 _ci(dd[f"leads/conversion/{p}/slope"], 3)] for p in ("25-75", "50-94", "94-150", "150-250")]
        L += ["Conversion: slope of the later change of the lead on the lead at the first turn (negative = leads "
              "shrink, 0 = persist unchanged, positive = grow).", ""]
        L += _table(["turns", "baseline", "C1", "C1 - baseline"], rows)
        rows = []
        for ph in ("lane", "after"):
            k = f"recall/{ph}"
            rows.append([ph, _ci(b[f"{k}/per_game_seat"], 2, False), _ci(b[f"{k}/hp_at_choice/mean"], 3, False),
                         _ci(b[f"{k}/unspent_at_choice/mean"], 2, False), _ci(b[f"{k}/interrupted_share"], 3, False),
                         _ci(dd[f"{k}/per_game_seat"], 2), _ci(dd[f"{k}/hp_at_choice/mean"], 3)])
        L += ["### Recall choices (lane = turns 0-93, after = 94-399)", ""]
        L += _table(["phase", "choices per seat", "HP at choice", "unspent at choice", "interrupted share",
                     "C1 - baseline choices", "C1 - baseline HP"], rows)
        rows = []
        for ph in ("lane", "after"):
            k = f"freezes/{ph}"
            rows.append([ph, _ci(b[f"{k}/episodes_per_game_seat"], 2, False), _ci(b[f"{k}/mean_length"], 2, False),
                         _ci(b[f"{k}/long_share_ge3"], 3, False), _ci(dd[f"{k}/episodes_per_game_seat"], 2),
                         _ci(dd[f"{k}/mean_length"], 2)])
        L += ["### Freeze episodes (maximal runs of held turns, by the phase of their first turn)", ""]
        L += _table(["phase", "episodes per seat", "mean length", "share >= 3 turns", "C1 - baseline episodes",
                     "C1 - baseline length"], rows)
    oc = C.load("occupancy")
    if oc:
        L += ["### Training occupancy (reconstructed, M4.6e exposure games) and objective weight", ""]
        rows = []
        for tag, o in oc["occupancy"].items():
            rows.append([tag] + [_f(o["below"][str(lv)]["training"], 3) for lv in (8, 10, 12, 14, 16)] +
                        [_f(o["below"]["14"]["s0"], 3), _f(o["below"]["14"]["archive"], 3)])
        L += _table(["policy", "below L8", "below L10", "below L12", "below L14", "below L16", "s0 games below L14",
                     "archive games below L14"], rows)
        rows = []
        for h, w in oc["objective_weights"].items():
            rows.append([h, _f(w["gamma"], 5), _f(w["median_duration"], 0)] +
                        [_f(w["before"][k], 3) for k in ("L10", "L12", "L14", "L18")] +
                        [_f(w["in_phase_decisions_before_end"], 3), _f(w["beyond_300"], 3)])
        L += [f"Natural level times: {', '.join(f'{k} {v:.1f}' for k, v in oc['level_marks'].items())}. "
              "Weight before a level = share of the objective's weight from s0 before that turn; in-phase decisions = "
              "average share before L14 of the weight of a decision taken at a uniformly drawn turn before L14.", ""]
        L += _table(["H", "gamma", "median duration", "before L10", "before L12", "before L14", "before L18",
                     "in-phase decisions before L14", "beyond turn 300"], rows)
    rt = C.load("returns")
    if rt:
        b = rt["aggregate"]["by_env"]["baseline"]
        c = rt["aggregate"]["by_env"]["C1"]
        L += ["### Existing trajectories rescored: realized return from turn t0 (no bootstrap), split at turn 94", ""]
        rows = []
        for h in C.HORIZONS:
            for t0 in ("0", "25", "50", "75"):
                k = f"variance/{h}/{t0}"
                rows.append([h, t0, _ci(b[f"{k}/sd_total"], 2, False), _ci(b[f"{k}/var_share_post"], 3, False),
                             _ci(b[f"{k}/corr_pre_post"], 3), _ci(c[f"{k}/var_share_post"], 3, False)])
        L += _table(["H", "t0", "SD of return (baseline)", "variance share after L14 (baseline)",
                     "corr(before, after)", "variance share after L14 (C1)"], rows)
        rows = []
        for t0 in ("25", "50", "75"):
            for h in ("100", "60"):
                k = f"advantage/{t0}/{h}"
                rows.append([t0, h] + [_ci(b[f"{k}/{p}/invested_lead"], 3) for p in ("total", "pre", "post")] +
                            [_ci(c[f"{k}/{p}/invested_lead"], 3) for p in ("total", "pre", "post")])
        L += ["State value of a natural invested lead (OLS coefficient per resource, with unspent lead, level and HP "
              "differences as covariates; observational):", ""]
        L += _table(["t0", "H", "baseline total", "baseline before L14", "baseline after L14", "C1 total",
                     "C1 before L14", "C1 after L14"], rows)
    dl = C.load("delayed")
    if dl:
        L += delayed_tables(dl)
    return L


def delayed_tables(dl: dict) -> List[str]:
    L = ["### Delayed rewards in paired benefits (300-turn rollouts, gamma 0.99 split at L14)", "",
         f"Tail weight beyond 300 turns: {', '.join(f'H{h} {v:.3f}' for h, v in dl['tail_weight_beyond_T'].items())}. "
         "post share = (after L14) / total under H100, level-triggered cut (shown when the total is at least 0.1). "
         "Stage investments: "
         f"{dl['stage_invest']}.", ""]
    A = dl["aggregate"]
    for fam, title in (("cells", "Constructed lead premiums (situations pooled)"),
                       ("natural", "Natural early states: candidate - policy"),
                       ("freeze", "Early freeze cells: plan - policy (the intervening seat's view)")):
        L += [f"**{title}**", ""]
        rows = []
        for env in C.ENVS:
            g = A[fam]["by_env"][env]
            keys = sorted({k.rsplit("/", 1)[0] for k in g})
            for k in keys:
                if fam == "natural" and not k.endswith(("|all", "|t<=35", "|t>=50")):
                    continue
                tot = g[f"{k}/H100"]["mean"]
                post = g[f"{k}/post_level"]["mean"]
                rows.append([env, k, _ci(g[f"{k}/H100"]), _ci(g[f"{k}/pre_level"]), _ci(g[f"{k}/post_level"]),
                             _f(post / tot, 2) if abs(tot) >= 0.1 else "-", _ci(g[f"{k}/post_elapsed"]),
                             _ci(g[f"{k}/H80"]), _ci(g[f"{k}/H60"]), _ci(g[f"{k}/H40"]), _ci(g[f"{k}/undisc_0_100"])])
        L += _table(["env", "benefit", "H100", "before L14", "after L14", "post share", "after (elapsed cut)",
                     "H80", "H60", "H40", "undiscounted 0-99"], rows)
    rk = A["ranking"]["by_env"]
    rows = [[env, h, _ci(rk[env][f"{h}/best_change_share"], 3, False), _ci(rk[env][f"{h}/tau_mean"], 3, False)]
            for env in C.ENVS for h in C.HORIZONS if h != "100"]
    L += ["**Natural states: does the best option (the policy or one of its five alternatives) change with the "
          "horizon?**", ""]
    L += _table(["env", "H", "share of states whose best option changes vs H100", "mean Kendall tau vs H100"], rows)
    return L


# ---------------------------------------------------------------------------------------------- Stage 2
def stage2() -> List[str]:
    L: List[str] = [f"## Stage 2: the matched horizon pilot (control gamma 0.99, short gamma 1 - 1/{C.SHORT_H} = "
                    f"{C.SHORT_GAMMA:.5f})", ""]
    sm = C.load("smoke")
    if sm:
        L += ["### Smoke checks", ""]
        L += _table(["check", "result"], [[k, "PASS" if v else "FAIL"] for k, v in sm["checks"].items()])
        L += ["GAE spy (first calls of each smoke run): " + "; ".join(
            f"{k}: gamma {v[0]['gamma']:.5f}, truncated {v[0]['truncated']}, "
            f"bootstrapped {v[0]['truncated_bootstrap_nonzero']}, "
            f"adv rms {v[0]['adv_rms']:.3f}, rms change if gamma 0.99 {v[0]['adv_rms_diff_vs_gamma_0.99']:.3f}"
            for k, v in sm["gae_spy"].items() if v), ""]
    an = C.load("analysis")
    if not an:
        return L
    tr = an.get("training", {})
    if tr.get("runs"):
        rows = []
        for n, r in tr["runs"].items():
            if r.get("status") != "complete":
                rows.append([n, r.get("status", "?")] + [""] * 7)
                continue
            rows.append([n, "complete", f"{r['agent_transitions']:,}", _f(r["transitions_per_s"], 0),
                         _f(r["wall_s"] / 60, 1), _f(r["entropy_first_last"][1], 3) if r.get("entropy_first_last")
                         else "n/a", _f(r["approx_kl_mean"], 4), _f(r["explained_variance"], 3)
                         if "explained_variance" in r else "n/a", _f((r.get("starts") or {}).get(
                             "archive_transition_share_finished", np.nan), 3)])
        L += ["### Training", ""]
        L += _table(["run", "status", "transitions", "per s", "wall min", "final entropy", "mean KL",
                     "explained variance", "archive share"], rows)
        L += [f"Checks: {sum(tr['checks'].values())}/{len(tr['checks'])} pass; all passed: {tr['all_passed']}.", ""]
    h2h = an.get("h2h", {})
    if h2h and "missing" not in h2h:
        L += ["### Head to head from s0: the short arm's paired-seat lead against the control", ""]
        keys = ["lane"] + [f"win_{w}" for w in C.WINDOWS] + ["after_lane", "level14"] + \
            [f"G{h}" for h in C.HORIZONS]
        rows = []
        for s, d in h2h.items():
            for ph in ("final", "midpoint"):
                rows.append([f"{s} {ph} ({d[ph]['games']} pairs)"] + [_ci(d[ph][k]) for k in keys])
        L += _table(["seed"] + keys, rows)
    fx = an.get("fixed", {})
    if fx and "missing" not in fx:
        L += ["### Against fixed opponents (paired seats; each policy's lead)", ""]
        rows = []
        for s, d in fx.items():
            for xn, g in d["by_policy"].items():
                for grp in ("start", "other_finals", "scripted", "fixed_pool"):
                    if grp in g:
                        rows.append([s, xn, grp, _ci(g[grp]["lane"]), _ci(g[grp]["after_lane"]), _ci(g[grp]["G100"]),
                                     _ci(g[grp]["G_short"])])
            for grp, v in d["short_minus_control"].items():
                rows.append([s, "short - control", grp, _ci(v["lane"]), _ci(v["after_lane"]), _ci(v["G100"]),
                             _ci(v["G_short"])])
        L += _table(["seed", "policy", "opponents", "lane 0-93", "after 94-399", "G100", f"G{C.SHORT_H}"], rows)
    nat = an.get("natural", {})
    if nat and "missing" not in nat:
        L += ["### Natural self-play behaviour (short - control, per game; threshold = Stage 1 across-seed SD)", ""]
        thr = nat["thresholds"]
        rows = []
        for w in ("lane", "after"):
            for k in thr[w]:
                row = [w, k, _f(thr[w][k], 4)]
                for s in C.PILOT_SEEDS:
                    d = nat[f"s{s}"]
                    row += [_f(d["summary"]["control"][w][k]["mean"], 3), _ci(d["short_minus_control"][w][k], 3),
                            _ci(d["control_minus_start"][w][k], 3)]
                rows.append(row)
        head = ["window", "metric", "threshold"]
        for s in C.PILOT_SEEDS:
            head += [f"s{s} control", f"s{s} short - control", f"s{s} control - start"]
        L += _table(head, rows)
    cl = an.get("cells", {})
    if cl and "missing" not in cl:
        L += ["### Paired families under each policy (Stage 1 states and seeds)", ""]
        rows = []
        for s, d in cl.items():
            for fam in ("cells", "natural", "freeze"):
                keys = sorted(d["start"][fam])
                if fam == "natural":
                    keys = [k for k in keys if k.endswith(("|all", "|t<=35", "|t>=50"))]
                for k in keys:
                    rows.append([s, fam, k] + [f"{_f(d[a][fam][k]['H100'], 2, True)} / "
                                               f"{_f(d[a][fam][k][f'H{C.SHORT_H}'], 2, True)}"
                                               for a in ("start", "control", "short")])
        L += ["Each cell: benefit under H100 / under the short horizon (mean over states and rollouts).", ""]
        L += _table(["seed", "family", "benefit", "start", "control", "short"], rows)
    cs = an.get("classification")
    if cs:
        L += ["### Pre-registered classification", "",
              f"Outcome {cs['outcome']}: {cs['recommendation']}.", "",
              f"P1 (lane lead, materiality {cs['materiality']}): " +
              "; ".join(f"{k} {_ci(v)}" for k, v in cs["P1"].items()) + ".",
              f"Regression flag: {cs['regression']} (fixed pool G100 " +
              "; ".join(f"{k} {_ci(v)}" for k, v in cs["regression_full"].items()) + ").",
              "Meaningful behaviour changes: " +
              (", ".join(k for k, v in cs["behaviour"].items() if v["meaningful"]) or "none") + ".", ""]
        cu = an.get("classification_unpaired_fixed_interval")
        if cu:
            L += [f"Sensitivity: with the first code's unpaired fixed-opponent interval the outcome would be "
                  f"{cu['outcome']} ({cu['recommendation']}); regression flag {cu['regression']}.", ""]
    pr = C.load("preregistration") or {}
    if pr.get("amendments"):
        L += ["### Amendments to the pre-registration record", ""]
        L += [f"- {a['time']}: {a['note']}" for a in pr["amendments"]] + [""]
    return L


def run() -> str:
    L = ["# M4.7 early-lane horizon study: generated tables", "",
         "Generated by `python -m lanegame.experiments.rl.m47 report`; the narrative is "
         "docs/m4/m4_7/m4_7_horizon_report.md.", ""]
    L += stage1()
    L += stage2()
    path = C.write_md("report", "\n".join(L))
    print(f"wrote {path}", flush=True)
    return path


__all__ = ["run", "stage1", "stage2", "delayed_tables"]
