"""results/m44a/report.md: every table of the M4.4a readiness report, generated from the stage JSON files (audit,
train_smoke, eval_smoke, demo, m44b_plan). Missing inputs are skipped with a note."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..common import md_table
from .actions import BEH_NAMES, WAVE_NAMES
from .common import load_json, write_md

STATES = ("all present turns", "fronts in contact at the lane cell", "my wave stronger (window strength +1 or more)",
          "enemy wave stronger (-1 or less)", "enemy wave much stronger (-4 or less)",
          "holding a freeze (freeze state, I am the freezer)", "frozen on (freeze state at the enemy tower)",
          "my crash under the enemy tower", "enemy crash under my tower", "enemy champion absent")
SHORT = {"all present turns": "all", "fronts in contact at the lane cell": "contact at lane",
         "my wave stronger (window strength +1 or more)": "my wave stronger", "enemy wave stronger (-1 or less)":
         "enemy stronger", "enemy wave much stronger (-4 or less)": "enemy much stronger",
         "holding a freeze (freeze state, I am the freezer)": "holding freeze",
         "frozen on (freeze state at the enemy tower)": "frozen on", "my crash under the enemy tower": "my crash",
         "enemy crash under my tower": "enemy crash", "enemy champion absent": "enemy absent"}


def f2(x, d=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{d}f}"


def pct(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{100 * x:.0f}%"


def ci(v, d=2) -> str:
    if not v or v[0] is None or not np.isfinite(v[0]):
        return "–"
    return f"{v[0]:+.{d}f} ± {v[1]:.{d}f}" if np.isfinite(v[1]) else f"{v[0]:+.{d}f}"


# ---------------------------------------------------------------------------------------------- sections
def audit_section(a: dict) -> List[str]:
    rows = [[k, v["preset"], v["obs_size"], v["n_actions"], v["absence_scale"], v["turns_played"],
             len(v["observed_legal_codes"]), len(v["observed_outside_ever_legal"]), v["observations_out_of_range"],
             v["masks_not_equal_legal"], v["mirror_failures"]] for k, v in a["environments"].items()]
    out = ["## 1. Audit (random play, action space, checkpoint compatibility)", "",
           md_table(["env", "preset", "obs", "actions", "absence scale", "turns", "legal codes seen",
                     "outside ever-legal", "obs out of [0,1]", "mask != legal", "mirror failures"], rows), "",
           f"Ever-legal codes: {a['action_space']['ever_legal']} of 61; never legal: {a['action_space']['never_legal']}.",
           "", "Checkpoint compatibility (`common.load_policy`, no cross-environment opt-in):", "",
           md_table(["checkpoint", "four", "econ", "aggro"],
                    [[f"`{p}`", r["four"], r["econ"], r["aggro"]] for p, r in a["checkpoint_compatibility"].items()])]
    return out


def train_section(t: dict) -> List[str]:
    rows = []
    for n, r in t["runs"].items():
        rows.append([n, f"{r['agent_transitions']:,}", f"{r['elapsed_s']:.0f}", f"{r['transitions_per_s']:,.0f}",
                     f"{r['entropy_first_last'][0]:.2f} → {r['entropy_first_last'][1]:.2f}",
                     f"{r['explained_variance_first_last10'][1]:.2f}", f"{r['approx_kl_mean']:.4f}",
                     f"{r['grad_norm_max']:.2f}", f"{r['value_absmax_resources']:.1f}",
                     f"{r['value_s0_resources'][0]:+.2f}", r["snapshots"]])
    tp = t["throughput"]
    out = ["## 2. Smoke training (the M4.4b matrix at 1/10 budget)", "",
           md_table(["run", "transitions", "s", "transitions/s", "entropy", "EV (last 10)", "KL mean", "grad max",
                     "max |V| (res.)", "V(s0)", "snapshots"], rows), "",
           f"Wall clock {tp['wall_s']:.0f} s for {len(t['runs'])} runs on {tp['processes']} processes; per-process "
           f"throughput min / mean / max {tp['transitions_per_s_per_process_min_mean_max'][0]:,.0f} / "
           f"{tp['transitions_per_s_per_process_min_mean_max'][1]:,.0f} / "
           f"{tp['transitions_per_s_per_process_min_mean_max'][2]:,.0f} agent transitions/s.", "",
           f"Pipeline checks: **{sum(t['checks']['checks'].values())} / {len(t['checks']['checks'])} passed**; "
           f"learning checks (informational): {sum(t['checks']['learning'].values())} / {len(t['checks']['learning'])}."]
    failed = [k for k, v in t["checks"]["checks"].items() if not v] + [k for k, v in t["checks"]["learning"].items() if not v]
    if failed:
        out += ["", "Not passed: " + "; ".join(failed)]
    if t.get("reproducibility"):
        out += ["", md_table(["run", "weights digest", "re-run digest", "identical weights", "identical histories"],
                            [[n, r["digest"][:16], r["repeat_digest"][:16], r["identical_weights"],
                              r["identical_histories"]] for n, r in t["reproducibility"].items()])]
    out += ["", md_table(["seed", "initial digest (four / econ / aggro)", "identical"],
                         [[k, " / ".join(d[:12] for d in v["digests"].values()), v["identical_across_environments"]]
                          for k, v in t["initialization"].items()])]
    return out


def strength_tables(e: dict, title: str) -> List[str]:
    refs, xp, hist, curve = {}, [], [], {}
    for k, v in e["strength"].items():
        env, x, y = k.split("|")
        if x.startswith("snap:") and y.startswith("ref:"):
            curve.setdefault((env, x.rsplit(":", 1)[0]), {}).setdefault(int(x.rsplit(":", 1)[1]), {})[y[4:]] = v["estimate"]
        elif y.startswith("ref:"):
            refs.setdefault(x, {})[y[4:]] = v
        elif y.startswith("final:"):
            xp.append([env, x, y, ci([v["estimate"], 1.96 * v["stderr"]])])
        else:
            hist.append([env, x, y, ci([v["estimate"], 1.96 * v["stderr"]])])
    names = sorted({r for d in refs.values() for r in d})
    out = [f"### {title}: value against the references (payoff to the policy, both seats, ± 95%)", "",
           md_table(["policy"] + names, [[x] + [ci([d[r]["estimate"], 1.96 * d[r]["stderr"]]) if r in d else "–"
                                               for r in names] for x, d in refs.items()])]
    if xp:
        out += ["", f"### {title}: cross-play between seeds", "", md_table(["env", "x", "y", "value to x"], xp)]
    if hist:
        out += ["", f"### {title}: against its own past (snapshots, initial network)", "",
                md_table(["env", "final", "opponent", "value to the final"], hist)]
    if curve:
        rows = []
        for (env, base), pts in curve.items():
            for p in sorted(pts):
                rows.append([base, f"{p:,}"] + [f2(pts[p].get(r)) for r in sorted(pts[p])])
        rr = sorted({r for pts in curve.values() for d in pts.values() for r in d})
        out += ["", f"### {title}: learning curve (snapshots against references)", "",
                md_table(["run", "transitions"] + rr, rows)]
    return out


def wave_table(beh: Dict[str, dict], group: str = "natural:s0", opp: str = "self") -> List[str]:
    rows = []
    for k, v in beh.items():
        env, spec, o, start = k.split("|")
        if o != opp:
            continue
        g = v["metrics"].get(group)
        if g is None:
            continue
        bs = g["wave_management"]["by_state"]
        for st in STATES:
            r = bs[st]
            if r["n"] == 0:
                continue
            rows.append([spec, SHORT[st], r["n"]] + [pct(r["chose"][w]) for w in WAVE_NAMES])
    return [md_table(["policy", "state", "turns"] + list(WAVE_NAMES), rows)] if rows else ["(no data)"]


def freeze_table(beh: Dict[str, dict], group: str, opp: str = "self") -> List[str]:
    rows = []
    for k, v in beh.items():
        env, spec, o, start = k.split("|")
        if o != opp or group not in v["metrics"]:
            continue
        f = v["metrics"][group]["freezes"]
        e = f["established"]
        if e["n"] == 0:
            rows.append([spec, pct(f["held_share"]), ci(f["established_per_1000_turns"], 1), 0] + ["–"] * 9)
            continue
        ends = ", ".join(f"{a} {b}" for a, b in e["end"].items() if b)
        rows.append([spec, pct(f["held_share"]), ci(f["established_per_1000_turns"], 1), e["n"],
                     f2(e["length_mean"], 1), e["length_max"], ends, ci(e["gold_diff"]), pct(e["useful_denial_share"]),
                     ci(e["frozen_on_xp_missed"], 1), ci(e["freezer_preserving_share"]),
                     pct(e["freezer_shoved_at_end"]), ci(e["after_gold_diff"])])
    return [md_table(["policy", "held share", "established /1000 turns", "episodes", "mean length", "max", "end",
                      "gold diff (freezer)", "useful denial", "frozen-on XP lost", "freezer LH+TN share",
                      "freezer Shoved at end", "gold diff next 20"], rows)] if rows else ["(no data)"]


def zoning_table(beh: Dict[str, dict], group: str) -> List[str]:
    rows, threat = [], []
    for k, v in beh.items():
        env, spec, o, start = k.split("|")
        if o != "self" or group not in v["metrics"]:
            continue
        z = v["metrics"][group]["zoning"]
        for cls, r in z["by_power"].items():
            if r["n"]:
                rows.append([spec, cls, r["n"], f2(r["mean_pos"]), pct(r["at_own_tower"]), pct(r["trade_attempt_rate"]),
                             pct(r["trade_connect_rate"]), f2(r["connected_dealt_bars"]), f2(r["connected_taken_bars"]),
                             pct(r["collection_rate"]), f2(1000 * r["death_rate"], 1)])
        for cls, r in z["credible_threat"].items():
            if r["n"]:
                b = r["behavior"]
                threat.append([spec, cls, r["n"]] + [pct(b[x]) for x in BEH_NAMES] +
                              [f2(r["mean_pos"]), f2(r["uncollected_per_turn"], 3), f2(r["xp_missed_per_turn"], 2),
                               pct(r["died_within_4"]), pct(r["recalled_within_4"])])
    out = [md_table(["policy", "power class", "turns", "mean cell", "at own tower", "trade attempt", "connected",
                     "dealt (bars)", "taken (bars)", "collection", "deaths /1000"], rows) if rows else "(no data)", ""]
    out += [md_table(["policy", "situation", "turns"] + list(BEH_NAMES) + ["mean cell", "gold left /turn",
                                                                           "XP lost /turn", "died ≤4", "recalled ≤4"],
                     threat) if threat else "(no credible-threat turns)"]
    return out


def vision_table(beh: Dict[str, dict], group: str = "natural:s0") -> List[str]:
    rows, dives = [], []
    for k, v in beh.items():
        env, spec, o, start = k.split("|")
        if o != "self" or group not in v["metrics"]:
            continue
        vt = v["metrics"][group]["vision_tower"]
        per = vt["per_1000_turns"]
        ob = vt["objective_choice_when_legal"]
        n_leg = sum(x["n"] for x in ob.values())
        pv = sum(x["n"] * (x["Vision"] if np.isfinite(x["Vision"]) else 0) for x in ob.values()) / max(n_leg, 1)
        pp = sum(x["n"] * (x["Pressure"] if np.isfinite(x["Pressure"]) else 0) for x in ob.values()) / max(n_leg, 1)
        fe = vt["forward_exposure"]
        rows.append([spec, n_leg, pct(pp), pct(pv), pct(ob["stronger"]["Vision"]) if ob["stronger"]["n"] else "–",
                     vt["extended_pressure_runs"]["n"], pct(vt["extended_pressure_runs"]["with_vision_share"]),
                     f"{fe['vision']['n']} / {fe['no vision']['n']}", f2(fe["vision"]["gank_p"], 3),
                     f2(fe["no vision"]["gank_p"], 3), f2(per["ganks"], 1), f2(per["plates"], 1),
                     vt["plate_turn"]["plate 1"]["median"]])
        d = vt["dives"]
        dr = vt["defender_reply_on_dive_turn"]
        nd = max(sum(dr.values()), 1)
        dives.append([spec, f2(per["dives"], 1), f2(per["busy_dives"], 1), d["busy"]["n"], f2(d["busy"]["mean_tower_p"]),
                      d["not busy"]["n"], f2(d["not busy"]["mean_tower_p"]), pct(d["busy"]["diver_died_within_2"]),
                      pct(d["not busy"]["diver_died_within_2"]), f2(per["expected_tower_shots"], 1),
                      ", ".join(f"{a} {pct(b / nd)}" for a, b in dr.items() if b)])
    return [md_table(["policy", "objective-legal turns", "Pressure", "Vision", "Vision when stronger",
                      "pressure runs ≥3", "with vision", "forward turns vision / none", "gank p (vision)",
                      "gank p (none)", "ganks /1000", "plates /1000", "1st plate (median turn)"], rows), "",
            md_table(["policy", "dives /1000", "busy dives /1000", "busy n", "busy shot p", "not-busy n",
                      "not-busy shot p", "busy diver died ≤2", "not-busy diver died ≤2", "tower shots exp. /1000",
                      "defender on the dive turn"], dives)]


def plate_table(beh: Dict[str, dict], group: str = "natural:s0") -> List[str]:
    rows = []
    for k, v in beh.items():
        env, spec, o, start = k.split("|")
        if o != "self" or group not in v["metrics"]:
            continue
        pp = v["metrics"][group]["plate_phases"]
        for ph in ("attack plates left", "attack plates exhausted"):
            r = pp[ph]
            rows.append([spec, ph, r["turns"]] + [pct(r["waves"][w]) for w in WAVE_NAMES] +
                        [pct(r["pressure_when_legal"]), pct(r["vision_when_legal"]), pct(r["crash_rate"]),
                         pct(r["freeze_state_share"]), f2(r["farm_per_turn"], 3), f2(r["plate_per_turn"], 3),
                         f2(r["earned_per_turn"], 3)])
        ex = pp["attack_plates_exhausted_at_turn"]
        rows.append([spec, f"exhausted at turn (median of {ex['n']}; never {ex['never']})", f2(ex["median"], 0)] +
                    [""] * 11)
    return [md_table(["policy", "phase", "turns"] + list(WAVE_NAMES) + ["Pressure (when legal)", "Vision (when legal)", "crash rate",
                                                                         "freeze state", "farm /turn", "plates /turn",
                                                                         "earned /turn"], rows)]


def effect_table(beh: Dict[str, dict], group: str = "natural:s0") -> List[str]:
    rows = []
    for k, v in beh.items():
        env, spec, o, start = k.split("|")
        if o != "self" or group not in v["metrics"]:
            continue
        ef = v["metrics"][group]["wave_management"]["neutral_action_effect_after_4_turns"]
        rows.append([spec] + [f"{pct(ef[w]['moved_forward_rate'])} / {pct(ef[w]['gained_strength_rate'])} "
                              f"(n {ef[w]['n']})" for w in WAVE_NAMES])
    return [md_table(["policy"] + [f"{w}: moved fwd / stronger" for w in WAVE_NAMES], rows)]


def reaction_table(beh: Dict[str, dict], group: str = "natural:s0") -> List[str]:
    """Shove / Thin shares in reply to a repeated opponent pattern, and around the focal side's own crashes."""
    rows = []
    for k, v in beh.items():
        env, spec, o, start = k.split("|")
        if o != "self" or group not in v["metrics"]:
            continue
        wm = v["metrics"][group]["wave_management"]
        rp, ar = wm["reply_to_opponent_pattern"], wm["around_my_crash"]
        cells = []
        for key in ("opponent Thinned 3 turns", "opponent Shoved 3 turns", "opponent Last Hit 3 turns", "otherwise"):
            r = rp[key]
            cells.append(f"{pct(r['shares']['Shove'])} / {pct(r['shares']['Thin'])} (n {r['n']})" if r["n"] else "–")
        for side in ("crasher", "defender"):
            b, a = ar[f"{side}_before"], ar[f"{side}_after"]
            cells.append(f"{pct(b['shares']['Shove'])} → {pct(a['shares']['Shove'])}" if b["n"] and a["n"] else "–")
        rows.append([spec] + cells)
    return [md_table(["policy", "after opp. Thin x3: Shove / Thin", "after opp. Shove x3", "after opp. Last Hit x3",
                      "otherwise", "crasher Shove 5 before → after", "defender Shove 5 before → after"], rows)]


def pressure_cost_table(beh: Dict[str, dict], group: str) -> List[str]:
    rows = []
    for k, v in beh.items():
        env, spec, o, start = k.split("|")
        if o != "self" or group not in v["metrics"]:
            continue
        pc = v["metrics"][group]["zoning"]["pressure_cost"]
        rows.append([spec, f"{f2(pc['opp_uncollected_when_stronger_near']['mean'], 3)} (n {pc['opp_uncollected_when_stronger_near']['n']})",
                     f"{f2(pc['opp_uncollected_otherwise']['mean'], 3)} (n {pc['opp_uncollected_otherwise']['n']})",
                     f"{pct(pc['own_collection_trading']['mean'])} (n {pc['own_collection_trading']['n']})",
                     f"{pct(pc['own_collection_holding']['mean'])} (n {pc['own_collection_holding']['n']})"])
    return [md_table(["policy", "opponent's gold left /turn: I am stronger and near", "otherwise",
                      "my collection when I Trade", "when I Hold"], rows)]


def cf_table(cf: Dict[str, dict]) -> List[str]:
    rows = []
    for k, v in cf.items():
        env, spec, key, protocol = k.split("|")
        s = v["summary"]
        if not s.get("n_states"):
            rows.append([spec, key, protocol, 0, v["states_found"], v["states_skipped"]] + ["–"] * 6)
            continue
        a, b = s["contrast"].split(" - ")
        rows.append([spec, key, protocol, s["n_states"], v["states_found"], v["states_skipped"], s["contrast"],
                     ci(s["mean_contrast"]), f"{pct(s['share_significant_positive'])} / {pct(s['share_significant_negative'])}",
                     f"{pct(s['frequency'][a][0])} / {pct(s['frequency'][b][0])}",
                     ci(s.get("critic_mean_contrast_Q1", [float("nan"), float("nan")])),
                     pct(s.get("critic_sign_agreement", float("nan")))])
    return [md_table(["policy", "situation", "opponent protocol", "states", "found", "skipped (not legal)", "contrast",
                      "measured (± 95%, states)", "sig. + / −", "policy freq. a / b", "critic one-step contrast",
                      "critic sign agrees"], rows)]


# ---------------------------------------------------------------------------------------------- report
def report() -> str:
    out = ["# M4.4a — results tables", "",
           "Generated by `python -m lanegame.experiments.rl.m44 report` from results/m44a/*.json. Interpretation and "
           "definitions: docs/m4/m4_4/m4_4a_ppo_readiness.md. Rates are pooled over turns unless a ± interval is shown "
           "(intervals: 95% over games, seeds or states as stated).", ""]
    a = load_json("audit")
    out += audit_section(a) if a else ["(audit missing)"]
    t = load_json("train_smoke")
    out += [""] + (train_section(t) if t else ["(train_smoke missing)"])
    e = load_json("eval_smoke")
    if e:
        out += ["", "## 3. Smoke evaluation pipeline (smoke finals, tiny sizes: plumbing, not results)", "",
                md_table(["final", "digest verified = training record", "refused other envs", "trained on"],
                         [[k, v["matches_training_record"], all(v["refused_other_envs_without_opt_in"].values()),
                           v["trained_on"]] for k, v in e["load"].items()]), ""]
        out += strength_tables(e, "Smoke")
        out += ["", "### Smoke: viewer traces (exported, then replayed through `viewer.session.play`)", "",
                md_table(["recording", "config", "turns", "actions equal", "payoff equal"],
                         [[k, v["verify"]["config"], v["verify"]["turns"], v["verify"]["actions_equal"],
                           v["verify"]["payoff_equal"]] for k, v in e["traces"].items()])]
        out += ["", "### Smoke: counterfactual pipeline", ""] + cf_table(e["counterfactual"])
        out += ["", f"Smoke evaluation wall clock: {e['elapsed_s']:.0f} s."]
    d = load_json("demo")
    if d:
        beh = d["behavior"]
        out += ["", "## 4. Demonstration on the M4.3c pilot finals (EXPLORATORY checkpoints; not an M4.4 result)", "",
                "`pilot:base:*` = `m4_four_actions()`, `pilot:candidate:*` = `m4_four_actions_econ()`; 2,048,000 "
                "transitions each, entropy 0.03 (M4.3c stabilization §10).", "",
                "### 4.1 Wave action by decision state (natural self-play from s0)", ""] + wave_table(beh)
        out += ["", "### 4.2 Does the chosen action produce its intended wave effect? (neutral lane, 4 turns later: "
                "the meeting point moved toward the enemy / my window strength led by >= 1)", ""] + effect_table(beh)
        out += ["", "### 4.2b Reactions: reply to a repeated opponent pattern; Shove share in the 5 turns before / after "
                "the focal side's own crash (crasher and defender)", ""] + reaction_table(beh)
        out += ["", "### 4.3 Freeze episodes (natural self-play from s0)", ""] + freeze_table(beh, "natural:s0")
        out += ["", "### 4.4 Freeze episodes from the INITIALIZED freeze start (labelled; never pooled with s0)", ""] + \
            freeze_table(beh, "initialized:freeze/equal")
        out += ["", "### 4.5 Zoning in natural self-play", ""] + zoning_table(beh, "natural:s0")
        out += ["", "### 4.6 Zoning from the INITIALIZED power gap (A x1.76 attack power; both seats are the policy)",
                ""] + zoning_table(beh, "initialized:neutral/large")
        out += ["", "Price of applying pressure (initialized power gap): the opponent's available gold left uncollected, and "
                "the focal side's own collection rate by its behaviour", ""] + pressure_cost_table(beh, "initialized:neutral/large")
        out += ["", "### 4.7 Vision, pressure, ganks, plates and dives (natural self-play)", ""] + vision_table(beh)
        out += ["", "### 4.8 Plate phases (from the tower-damage state)", ""] + plate_table(beh)
        out += ["", "### 4.9 Paired counterfactuals (one-step component interventions, window 150, policy "
                "continuation in both seats)", ""] + cf_table(d["counterfactual"])
        out += ["", f"Demonstration wall clock: {d['elapsed_s']:.0f} s."]
    p = load_json("m44b_plan")
    if p:
        est = p["estimate"]
        out += ["", "## 5. M4.4b estimate", "",
                f"Training: {est['training']['wall_minutes_9_processes']:.0f} min wall clock (9 runs in parallel), "
                f"{est['training']['cpu_hours']:.1f} CPU-hours. Evaluation: about "
                f"{est['evaluation']['wall_minutes_on_9_workers']:.0f} min on 9 workers "
                f"({est['evaluation']['cpu_hours_total']:.1f} CPU-hours). Full specification: results/m44a/m44b_plan.md."]
    text = "\n".join(out)
    write_md("report", text)
    return text


__all__ = ["report"]
