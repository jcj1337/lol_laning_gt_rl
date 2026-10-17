"""Generated tables of M4.6d (results/m46d/report.md) from results/m46d/analysis.json and freezes.json."""

from __future__ import annotations

from typing import List

import numpy as np

from . import common as C

FZN = {"policy": "policy", "guard_sel": "guard_sel", "guard_walk": "guard_walk*", "passive": "passive",
       "push": "push"}


def f(x, d=2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (list, tuple)):
        m, h = x[0], x[1]
        if m is None or not np.isfinite(m):
            return "n/a"
        mark = "**" if (np.isfinite(h) and abs(m) > h) else ""
        return f"{mark}{m:+.{d}f} ± {h:.{d}f}{mark}" if np.isfinite(h) else f"{m:+.{d}f}"
    return f"{x:+.{d}f}" if np.isfinite(x) else "n/a"


def table(head: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def descriptive(fz: dict) -> str:
    out = ["## 1. Descriptive: freeze episodes in natural self-play (accounting, not causal)\n"]
    for arm in ("C1", "baseline"):
        tb = fz["tables"][arm]
        out.append(f"### {arm} (6 policies x 48 games x 400 turns)\n")
        rows = []

        def add(name, x):
            if not x.get("episodes"):
                return
            rows.append([name, str(x["episodes"]), f"{x['n']:.2f}", f"{x['maintained']:.2f}",
                         f"{x['origin'].get('pushed-in', 0):.2f}", f"{x['contest'].get('walk-up', 0):.2f}",
                         f"{x['contest'].get('uncontested', 0):.2f}", f"{x['held_hp']:.1f} / {x['own_hp']:.1f}",
                         f"{x['duel']:.2f}", f"{x['hpf']:.2f} / {x['hpb']:.2f}", f"{x['f_wave']['SH']:.2f}",
                         f"{x['b_wave']['SH']:.2f}", f"{x['end'].get('crashed_in', 0):.2f}", f"{x['d_earned']:+.2f}",
                         f"{x['d_xp']:+.1f}", f"{x['b_deaths']:.2f}"])
        add("all", tb["all"])
        for k, v in tb["by_length"].items():
            add(k, v)
        for k, v in tb["by_level"].items():
            add(f"freezer {k}", v)
        for k, v in tb["by_duel"].items():
            add(k, v)
        for k, v in tb["by_component"].items():
            add(k, v)
        add("favourable", tb["favourable"])
        add("favourable, maintained", tb["favourable_maintained"])
        add("favourable, freezer L <= 9", tb["favourable_early"])
        out.append(table(["class", "episodes", "mean length", "share >= 4", "pushed-in", "walk-up", "uncontested",
                          "held / own wave HP", "duel", "HP f / b", "freezer Shove share", "breaker Shove share",
                          "ends crashed in", "earned f - b (window)", "XP f - b", "breaker deaths"], rows))
        out.append("")
        out.append("Frozen-on side's policy masses (both present):\n")
        rows = [[k.replace(" | ", ", "), str(v["n"]), f"{v['p_shove']:.2f}", f"{v['p_hold']:.2f}", f"{v['p_safe']:.2f}",
                 f"{v['p_trade']:.2f}", f"{v['p_recall']:.2f}"] for k, v in tb["frozen_on_masses"].items()]
        out.append(table(["breaker HP, strength", "turns", "Shove", "Hold", "Safe", "Trade", "Recall"], rows))
        out.append("")
        adv = fz[f"advantages_{arm}"]
        rows = []
        for lb in sorted(adv, key=lambda x: int(x[1:].split("-")[0])):
            r = adv[lb]
            rows.append([lb, f"{np.exp(r['log_duel']['0.5']):.2f} / {np.exp(r['log_duel']['0.9']):.2f}",
                         f"{r['share_duel_ge_1.10']:.2f}", f"{r['share_duel_ge_1.25']:.2f}",
                         f"{r['share_duel_ge_1.5']:.3f}", f"{r['dhp']['0.5']:.2f} / {r['dhp']['0.9']:.2f}",
                         f"{r['share_fight_ge_2']:.2f}", f"{r['dS']['0.5']:.1f} / {r['dS']['0.9']:.1f}"])
        out.append("Natural strength differences, turns 5-100, both present (by the lower level):\n")
        out.append(table(["level", "duel median / q90", "duel >= 1.10", ">= 1.25", ">= 1.5", "HP gap median / q90",
                          "HP-adjusted >= 2", "invested gap median / q90"], rows))
        out.append("")
    return "\n".join(out)


def a_section(a: dict, title: str) -> str:
    if not a or not a.get("cells"):
        return f"## {title}\n\n(not run)\n"
    out = [f"## {title}\n"]
    rows = []
    for idx, r in a["cells"].items():
        vp, ct = r["vs_policy"], r["contrasts"]
        held = [f"{r['channels'][x]['held_abs_A@20'][0]:.1f}" for x in r["freezers"]]
        mx = r["matrix"].get("50", {})
        x = mx.get("x", [])
        eq = " ".join(f"{FZN[n]} {w:.2f}" for n, w in zip(mx.get("freezers", []), x) if w > 0.01)
        mm = mx.get("maximin", [])
        best_mm = mx["freezers"][int(np.argmax(mm))] if mm else ""
        rows.append([r["label"], f(r["abs"]["50"]), *(f(vp[k]["50"]) if k in vp else "" for k in
                                                       ("guard_sel", "guard_walk", "passive", "push")),
                     f(ct["guard_sel-passive"]["50"]), f(ct.get("guard_walk-passive", {}).get("50")),
                     " / ".join(held), eq, FZN.get(best_mm, best_mm),
                     ", ".join(f"{FZN[n]}: {b}" for n, b in zip(mx.get("freezers", []), mx.get("best_reply", [])))])
    out.append("Per cell, G50 (A's units), against the breaker's policy unless stated; interval over 6 policies; "
               "**bold** = interval excludes 0. guard_walk* is exploratory (amendment 3).\n")
    out.append(table(["cell", "policy (abs)", "guard_sel - policy", "guard_walk* - policy", "passive - policy",
                      "push - policy", "guard_sel - passive", "guard_walk* - passive",
                      "held turns /20 (" + "/".join(FZN[x] for x in next(iter(a["cells"].values()))["freezers"]) + ")",
                      "restricted equilibrium (freezer)", "best maximin", "breaker's best reply"], rows))
    out.append("")
    rows = []
    for name, r in a.get("classes", {}).items():
        rows.append([name, str(r["cells"]), *(f(r[k]["50"]) if k in r else "" for k in
                                              ("guard_sel", "guard_walk", "passive", "push", "guard_sel-passive",
                                               "guard_sel-push", "guard_walk-passive", "guard_walk-push")),
                     " / ".join(f"{v[0]:.1f}" for v in r["held_A@20"].values())])
    out.append("Classes (contrasts averaged over the class's cells within each policy; G50):\n")
    out.append(table(["class", "cells", "guard_sel - policy", "guard_walk* - policy", "passive - policy",
                      "push - policy", "guard_sel - passive", "guard_sel - push", "guard_walk* - passive",
                      "guard_walk* - push", "held /20"], rows))
    out.append("")
    # channels for selected cells
    rows = []
    for idx, r in a["cells"].items():
        if not any(k in r["label"] for k in ("plaus", "x1.76", "equal")):
            continue
        for fz in r["freezers"][1:]:
            ch = r["channels"][fz]
            rows.append([r["label"], FZN[fz], f"{ch['held_abs_A@20'][0]:.1f}", f(ch["farm_B@50"]),
                         f(ch["farm_avail_B@50"]), f"{ch['unc_B@50'][0] - ch['unc_away_B@50'][0]:+.2f}",
                         f(ch["unc_away_B@50"]), f(ch["xp_B@50"], 1), f(ch["deaths_B@50"]), f(ch["away_B@50"], 1),
                         f(ch["d_total_A@50"]), f(ch["deaths_A@50"]), f(ch["farm_A@50"]), f(ch["earned_A@50"])])
    out.append("Channels over 50 turns (plan minus policy, both against the breaker's policy; undiscounted): keeping "
               "the wave (held), denial (B's farm, available farm, uncollected while present / away, XP), the "
               "freezer's own costs:\n")
    out.append(table(["cell", "freezer", "held /20", "B farm", "B available", "B lost present", "B lost away",
                      "B XP", "B deaths", "B turns away", "A HP lost (bars)", "A deaths", "A farm", "A earned"],
                     rows))
    out.append("")
    return "\n".join(out)


def al_section(al: dict) -> str:
    if not al or not al.get("cells"):
        return "## 2b. Long horizon (AL)\n\n(not run)\n"
    rows = []
    for idx, r in al["cells"].items():
        for k in ("guard_sel", "passive", "push"):
            v = r["vs_policy"][k]
            rows.append([r["label"], k, f(v["20"]), f(v["50"]), f(v["100"]), f(v["150"])])
        for k in ("guard_sel-passive", "guard_sel-push"):
            v = r["contrasts"][k]
            rows.append([r["label"], k, f(v["20"]), f(v["50"]), f(v["100"]), f(v["150"])])
    return "## 2b. Long horizon (AL, H 150): plan - policy against the breaker's policy\n\n" + table(
        ["cell", "contrast", "G20", "G50", "G100", "G150"], rows) + "\n"


def b_section(b: dict) -> str:
    if not b:
        return "## 3. Workstream B\n\n(not run)\n"
    out = ["## 3. Workstream B: the breaker's options (B's units, option - policy, G50)\n"]
    for fz in ("policy", "guard_sel", "guard_walk"):
        rows = []
        for idx, r in b.items():
            if fz not in r["freezer"]:
                continue
            x = r["freezer"][fz]
            opt = x["opt"]
            best = x["best"] + (" (clear)" if not x["uncertain_with"] else f" (~ {', '.join(x['uncertain_with'])})")
            rows.append([r["label"], f(x["policy_abs_B"]), *(f(opt[k]) for k in ("commit", "contest_sel", "concede",
                                                                                 "concede3", "reset")),
                         best, "yes" if x["material"] else ""])
        out.append(f"### freezer = {FZN[fz]}\n")
        out.append(table(["cell", "B policy (abs, B units)", "commit", "contest_sel", "concede (10)", "concede3*",
                          "reset", "best option", "policy materially inferior (discovery)"], rows))
        out.append("")
    return "\n".join(out)


def n_section(n: dict) -> str:
    if not n:
        return "## 4. Natural early freeze states (N)\n\n(not run)\n"
    rows = []
    for kind, r in n.items():
        rows.append([kind, f(r["abs_policy"]), *(f(r["freezer"].get(k)) for k in ("guard_sel", "guard_walk",
                                                                                   "passive", "push")),
                     f(r["guard-passive"]), f(r.get("walk-passive")),
                     " / ".join(f"{v[0]:.1f}" for v in r["held20"].values()),
                     " ".join(f"{b} {f(v)}" for b, v in r["breaker"]["policy"].items())])
    return "## 4. Natural early freeze states (N; G50; interval over 6 policies)\n\n" + table(
        ["kind", "policy (abs)", "guard_sel", "guard_walk*", "passive", "push", "guard_sel - passive",
         "guard_walk* - passive", "held /20", "breaker options vs the freezer's policy (B units)"], rows) + "\n"


def conf_section(cf: dict) -> str:
    if not cf:
        return "## 5. Confirmation\n\n(not run)\n"
    rows = []
    for grp, r in cf.items():
        if "opt" in r:
            rows.append([grp, r["label"], " ".join(f"{k} {f(v)}" for k, v in r["opt"].items())])
        else:
            rows.append([grp, r["label"], " ".join(f"{k} {f(v)}" for k, v in r["freezer"].items()) + " | " +
                         " ".join(f"{k} {f(v)}" for k, v in r.get("breaker", {}).items()) +
                         (f" | guard_sel - passive {f(r['guard-passive'])} guard_sel - push {f(r['guard-push'])}"
                          if "guard-passive" in r else "")])
    return "## 5. Confirmation sample (independent rollout seeds, K 48)\n\n" + table(["group", "cell", "estimates"],
                                                                                      rows) + "\n"


def conf_classes_section(cc: dict) -> str:
    if not cc:
        return ""
    rows = []
    for name, r in cc.items():
        if name.startswith("_"):
            continue
        d = r.get("discovery", {})
        rows.append([name, str(r["cells"]), *(f(r[k]) for k in ("guard_sel", "guard_walk", "passive", "push",
                                                               "guard_sel-passive", "guard_sel-push",
                                                               "guard_walk-passive", "guard_walk-push")),
                     " / ".join(f(d.get(k)) for k in ("guard_sel", "guard_walk", "guard_sel-push"))])
    return ("## 5b. Confirmation of the A classes (independent seeds, K 48; G50; discovery in the last column)\n\n" +
            table(["class", "cells", "guard_sel - policy", "guard_walk* - policy", "passive - policy",
                   "push - policy", "guard_sel - passive", "guard_sel - push", "guard_walk* - passive",
                   "guard_walk* - push", "discovery: guard_sel / guard_walk* - policy, guard_sel - push"], rows) + "\n")


def c_section(c: dict) -> str:
    out = ["## 6. Workstream C: specialists\n"]
    v = c.get("verify", {})
    for arm in ("baseline", "C1"):
        if arm not in v:
            continue
        rows = [[gr, str(r["cells"]), f(r["own_abs"]), f(r["adv"]),
                 " ".join(f"g{t} {f(x)}" for t, x in r["by_specialist"].items()),
                 f(r["ledger50"]["earned_A"]), f(r["ledger50"]["earned_B"]), f(r["ledger50"]["deaths_B"]),
                 f(r["ledger50"]["away_B"], 1), f(r["ledger50"]["rec_intr_B"]), f(r["ledger50"]["trade_conn_A"], 1),
                 f(r["ledger50"]["deaths_A"])]
                for gr, r in sorted(v[arm].items())]
        out.append(f"### Gap verification, {arm} environment{' (CROSS-ENVIRONMENT for the specialists)' if arm == 'C1' else ''}\n")
        out.append(table(["group", "cells", "general strong side (abs G50)", "specialist - general (G50)",
                          "by specialist", "A earned", "B earned", "B deaths", "B turns away", "B Recalls interrupted",
                          "A connected trades", "A deaths"], rows))
        out.append("")
    d = c.get("decomp", {})
    if d and "gap" in d:
        out.append("### Decomposition (baseline environment; units = pair x cell)\n")
        out.append(f"Gap (spec - gen, G50): {f(d['gap'])}; same-state one-turn substitutions summed over the sampled "
                   f"turns: {f(d['U_sum_sampled'])}; by kind: " +
                   "; ".join(f"{k} gap {f(x['gap'])} U-sum {f(x['U_sum'])} (n {x['units']})" for k, x in
                             d.get("by_kind", {}).items()) + "\n")
        out.append(table(["block (turns)", "increment"], [[k, f(x)] for k, x in d["blocks"].items()]))
        out.append("")
        out.append(table(["state class at block start (spec path)", "summed increment"],
                         [[k, f(x)] for k, x in sorted(d["S_by_class"].items())]))
        out.append("")
        out.append(table(["state class (gen path)", "summed one-turn substitution value"],
                         [[k, f(x)] for k, x in sorted(d["U_by_class"].items())]))
        out.append("")
        out.append(table(["turn k", "U_k - gen"], [[k, f(x)] for k, x in d["U_by_k"].items()]))
        out.append("")
        out.append(table(["class, spec action, gen action (turns 0-3, spec path)", "increment"],
                         [[k.replace("|", ", "), f(x)] for k, x in d["S_by_action_k<=3"].items()]))
        out.append("")
        out.append(table(["class, spec action, gen action (gen path)", "value"],
                         [[k.replace("|", ", "), f(x)] for k, x in d["U_by_action"].items()]))
        out.append("")
        out.append(table(["class, spec / gen action (gen path)", "one-turn value (interval over units)", "units",
                          "rollout-turns"], [[k.replace("|", ", "), f(v["value"]), str(v["units"]),
                                             str(v["rollout_turns"])] for k, v in d.get("window_by_unit", {}).items()]))
        out.append("")
    return "\n".join(out)


def critic_section(c: dict) -> str:
    out = ["## 7. Critic\n"]
    cal = c.get("calib", {})
    for arm in ("C1", "baseline"):
        if arm not in cal:
            continue
        rows = [[k, str(r["n_states"]), str(r["H"]), f(r["VA"]), f(r["G_H"]), f(r["bias_A_noboot"]),
                 f(r["bias_A_boot"]), f(r["VB"]), f(r["bias_B_noboot"]), f(r["zero_sum_VA+VB"]), f(r["G300_to_H"]),
                 f(r["G100_to_300"]), f(r["corr_V_G"])]
                for k, r in sorted(cal[arm].items())]
        out.append(f"### Calibration, {arm} (self-play returns without bootstrap; interval over policies)\n")
        out.append(table(["states", "n", "H", "V_A", "G_H (A)", "bias A (V - G_H)", "bias A with critic tail", "V_B",
                          "bias B (V_B + G_H)", "V_A + V_B", "G_300 -> G_H", "G_100 -> G_300", "corr(V, G) across states"],
                         rows))
        out.append("")
    rk = c.get("rank", {})
    if rk and "summary" in rk:
        rows = [[k, str(v["states"]), f"{v['spearman']:.2f}", f"{v['sign_agree']:.2f}", f"{v['best_match']:.2f}",
                 f"{v['actor_mass_on_best']:.2f}", "; ".join(f"{a}: {', '.join(b)}" for a, b in v["missed"])]
                for k, v in rk["summary"].items()]
        out.append("### Ranking of one-turn decisions (critic look-ahead vs 150-turn rollouts)\n")
        out.append(table(["family", "states", "Spearman (mean)", "sign agreement", "same best action",
                          "actor mass on the rollout-best action", "profitable (>= 0.25) but rarely chosen (< 0.10)"],
                         rows))
        out.append("")
        rows = []
        for st in rk["states"]:
            for cn, v in st["cands"].items():
                rows.append([st["label"], "AB"[st["decider"]], cn, f(v["roll150"]), f(v["roll100"]), f(v["crit"]),
                             f"{v['prob'][0]:.3f}"])
        out.append(table(["state", "decider", "action", "rollout adv G150", "G100", "critic adv", "actor prob"], rows))
        out.append("")
    ry = c.get("rank_yield", {})
    if ry and "error" not in ry:
        rows = [[lab, cn, f(v["roll150"]), f(v["crit"]), f"{v['prob']:.3f}"] for lab, d in sorted(ry.items())
                for cn, v in d.items()]
        out.append("### EXPLORATORY: the weak side's Yield and Safe codes at the Recall windows (vs its policy)\n")
        out.append(table(["state", "action", "rollout adv G150", "critic adv", "actor prob"], rows))
        out.append("")
    return "\n".join(out)


def run() -> str:
    an = C.load_json("analysis") or {}
    fz = C.load_json("freezes") or {}
    parts = ["# M4.6d generated tables\n",
             "*Generated by `python -m lanegame.experiments.rl.m46d report` from results/m46d/analysis.json and "
             "freezes.json. Intervals are 95% t intervals over the six general policies unless stated. Exploratory "
             "items are marked * (amendments 2-3).*\n"]
    if fz:
        parts.append(descriptive(fz))
    parts.append(a_section(an.get("A", {}), "2. Workstream A: freezer plans from constructed freeze cells (C1)"))
    parts.append(al_section(an.get("AL", {})))
    parts.append(a_section(an.get("Ab", {}), "2c. Baseline-environment reference (Ab)"))
    parts.append(b_section(an.get("B", {})))
    parts.append(n_section(an.get("N", {})))
    parts.append(conf_section(an.get("conf", {})))
    parts.append(conf_classes_section(an.get("conf_classes", {})))
    if an.get("C"):
        parts.append(c_section(an["C"]))
        parts.append(critic_section(an["C"]))
    text = "\n".join(parts)
    C.write_md("report", text)
    return text


__all__ = ["run"]
