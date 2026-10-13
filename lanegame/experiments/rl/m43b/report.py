"""The results report of the M4.3b pilot: results/rl/m43b/report.md, assembled from the stage JSON files (tables
only; the interpretation is in docs/rl/m4_3b_ppo_pilot_report.md). Stages that have not run are skipped."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .common import CONDITIONS, PRIMARY, SEEDS, fmt_ci, load, md_table, mean_ci, write_md
from .references import CURVE_REFS, HEAD_REFS

WAVES, BEHS = ("Hold", "Thin", "Clear"), ("Hold", "Yield", "Trade", "Safe", "Recall")


def _f(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def sec_train(train: dict) -> List[str]:
    runs = train["runs"]
    L = ["## 1. Training runs", "",
         f"Budget {train['meta']['budget']:,} agent transitions per run; {len(runs)} runs on {train['meta']['workers']} "
         f"worker processes; wall-clock {train['elapsed_s'] / 60:.1f} min. Commit `{train['meta']['commit'][:10]}`.", "",
         md_table(["run", "condition", "seed", "agent transitions", "env turns", "updates", "wall-clock s", "transitions/s",
                   "entropy first → last", "explained variance last 10", "grad norm mean / max", "clip frac", "KL mean / max",
                   "V̂(s0)", "max |V̂|", "single-choice", "games", "hist. games"],
                  [[n, s["condition"], s["seed"], f"{s['agent_transitions']:,}", f"{s['env_turns']:,}", s["updates"],
                    f"{s['elapsed_s']:.0f}", f"{s['transitions_per_s']:,.0f}",
                    f"{s['entropy_first_last'][0]:.2f} → {s['entropy_first_last'][1]:.2f}",
                    f"{s['explained_variance_first_last10'][1]:.2f}", f"{s['grad_norm_mean']:.2f} / {s['grad_norm_max']:.2f}",
                    f"{s['clipfrac_mean']:.3f}", f"{s['approx_kl_mean']:.4f} / {s['approx_kl_max']:.4f}",
                    f"{s['value_s0_resources']:+.2f}", f"{s['value_absmax_resources']:.1f}",
                    f"{s['single_choice_transitions']:,}", s["games_finished"], s["games_historical"]]
                   for n, s in runs.items()]), "",
         f"Health checks: {sum(train['checks']['checks'].values())} of {len(train['checks']['checks'])} passed "
         f"(all passed: {train['checks']['all_passed']})."]
    failed = [k for k, v in train["checks"]["checks"].items() if not v]
    if failed:
        L += ["", "Failed: " + "; ".join(failed)]
    return L + [""]


def sec_curves(curves: dict) -> List[str]:
    pts = curves["points"]
    refs = curves["meta"]["refs"]
    L = ["## 2. Learning curves (development window)", "", curves["label"] + ".", ""]
    by = {}
    for p in pts:
        by.setdefault((p["condition"], p["seed"]), []).append(p)
    for v in by.values():
        v.sort(key=lambda p: p["transitions"])
    # table: per condition, mean over seeds at selected points
    step = 102_400
    rnd = lambda t: int(round(t / step)) * step                 # historical runs' snapshots fall a few transitions late
    marks = sorted({rnd(p["transitions"]) for p in pts})
    sel = [m for m in marks if m in (0, 102_400, 307_200, 512_000, 1_024_000, 1_536_000, 2_048_000) or m == marks[-1]]
    rows = []
    for c in CONDITIONS:
        for m in sel:
            vals = {r: [p["vs"][r]["estimate"] for (cc, s), v in by.items() if cc == c for p in v if rnd(p["transitions"]) == m]
                    for r in refs}
            if not any(vals[r] for r in refs):
                continue
            prof = [p["profile"] for (cc, s), v in by.items() if cc == c for p in v if rnd(p["transitions"]) == m]
            row = [c, f"{m:,}"] + [fmt_ci(*mean_ci(vals[r])) for r in refs]
            row += [f"{np.mean([q['waves'][w] for q in prof]):.2f}" for w in WAVES]
            row += [f"{np.mean([q['behaviors'][b] for q in prof]):.2f}" for b in ("Trade", "Safe", "Recall")]
            row += [f"{np.mean([q['norm_entropy'] for q in prof]):.2f}", f"{np.mean([q['per_game_deaths'] for q in prof]):.1f}",
                    f"{np.mean([q['per_game_recalls'] for q in prof]):.1f}"]
            rows.append(row)
    L += [md_table(["condition", "transitions"] + [f"vs {r}" for r in refs] + ["Hold", "Thin", "Clear", "Trade", "Safe",
                                                                                "Recall", "H/log|legal|", "deaths/game",
                                                                                "recalls/game"], rows), "",
          "Values: mean over seeds ± 1.96 SE over seeds (n = 3, indicative). Action columns: shares of present turns "
          "in self-play (6 games of 460 turns per point); deaths and recalls per game and side.", ""]
    return L


def sec_headline(head: dict) -> List[str]:
    S = head["summary"]
    refs = [r for r in HEAD_REFS if any(r in S["vs_refs"].get(c, {}) for c in CONDITIONS)]
    L = ["## 3. Headline evaluation", "", head["label"] + ".", "", "### 3.1 Finals against the reference suite", "",
         md_table(["condition", "seed"] + [f"vs {r}" for r in refs] + ["vs initial net"],
                  [[c, s] + [f"{S['vs_refs'][c][r]['per_seed'][i][0]:+.2f} [{S['vs_refs'][c][r]['per_seed'][i][1]:+.2f}, "
                             f"{S['vs_refs'][c][r]['per_seed'][i][2]:+.2f}]" for r in refs]
                   + [f"{S['vs_initial'][c]['per_seed'][i]:+.2f}" if c in S["vs_initial"] else "n/a"]
                   for c in CONDITIONS if c in S["vs_refs"] for i, s in enumerate(SEEDS)
                   if i < len(S["vs_refs"][c][refs[0]]["per_seed"])]), "",
         "Per seed: estimate [95% paired CI over evaluation episodes].", "",
         md_table(["condition"] + [f"vs {r}" for r in refs] + ["vs initial", "seed SD (mean over refs)", "episode half-width (mean)"],
                  [[c] + [fmt_ci(S["vs_refs"][c][r]["mean"], S["vs_refs"][c][r]["ci_seeds"]) for r in refs]
                   + [fmt_ci(S["vs_initial"][c]["mean"], S["vs_initial"][c]["ci_seeds"]) if c in S["vs_initial"] else "n/a",
                      f"{S['variation'][c]['seed_sd_mean']:.2f}", f"{S['variation'][c]['episode_halfwidth_mean']:.2f}"]
                   for c in CONDITIONS if c in S["vs_refs"]]), "",
         "Mean over seeds ± 1.96 SE over seeds (three independently trained policies; indicative). The last two "
         "columns separate the two kinds of variation: the standard deviation across seeds of the value against a "
         "reference, and the half-width of one matchup's paired interval across evaluation episodes.", "",
         "### 3.2 Cross-play between finals (payoff to the row)", ""]
    finals, M = S["cross"]["finals"], S["cross"]["matrix"]
    lab = [f.replace("final:", "") for f in finals]
    L += [md_table(["row \\ column"] + lab, [[lab[i]] + [f"{M[i][j]:+.2f}" if i != j else "·" for j in range(len(lab))]
                                             for i in range(len(lab))]), "",
          md_table(["row condition \\ column condition"] + list(CONDITIONS),
                   [[c1] + [fmt_ci(S["cross_conditions"][f"{c1}|{c2}"]["mean"], S["cross_conditions"][f"{c1}|{c2}"]["ci"])
                            if f"{c1}|{c2}" in S["cross_conditions"] else "n/a" for c2 in CONDITIONS] for c1 in CONDITIONS]), "",
          "Condition against condition: mean over the seed pairs (9, or 6 within a condition) ± 1.96 SE over pairs.", ""]
    if S.get("progress"):
        pts = sorted({pt for c in S["progress"] for pt in S["progress"][c]}, key=int)
        L += ["### 3.3 Self-play progress: the final against its own earlier snapshots (payoff to the final)", "",
              md_table(["condition"] + [f"vs snapshot at {int(pt):,}" for pt in pts],
                       [[c] + [fmt_ci(S["progress"][c][pt]["mean"], S["progress"][c][pt]["ci_seeds"]) + " (" +
                               ", ".join(f"{v:+.1f}" for v in S["progress"][c][pt]["per_seed"]) + ")" if pt in S["progress"][c] else "n/a"
                               for pt in pts] for c in S["progress"]]), "",
              "Mean over seeds ± 1.96 SE (per-seed values in brackets). A positive value means the later network beats "
              "the earlier one in direct play, which the fixed references cannot show once they are dominated.", ""]
    return L


def sec_exploit(ex: dict) -> List[str]:
    L = ["## 4. Exploitability (approximate best responses)", "", ex["label"] + ".", "",
         md_table(["condition", "seed", "attacker vs target, W = 460", "W = 1000 [95% CI]", "vs the other seeds' finals (W = 1000)",
                   "attacker entropy first → last", "train s"],
                  [[v["condition"], v["seed"], f"{v['eval']['dev']['estimate']:+.2f}",
                    f"{v['eval']['head']['estimate']:+.2f} [{v['eval']['head']['ci95'][0]:+.2f}, {v['eval']['head']['ci95'][1]:+.2f}]",
                    ", ".join(f"{t['estimate']:+.2f}" for t in v["transfer"].values()),
                    f"{v['entropy_first_last'][0]:.2f} → {v['entropy_first_last'][1]:.2f}", f"{v['train_elapsed_s']:.0f}"]
                   for k, v in sorted(ex["results"].items())]), "",
         md_table(["condition", "mean gain over seeds (W = 1000)", "transfer mean"],
                  [[c, fmt_ci(s["mean"], s["ci_seeds"]), f"{s['transfer_mean']:+.2f}"] for c, s in ex["summary"].items()]), "",
         "The attacker's payoff is a lower bound on the target's exploitability (the attacker's budget is half the "
         "target's, warm-started at parity). A negative transfer value means the exploit is specific to the seed it "
         "was trained on.", ""]
    return L


def _self_tables(behavior: dict, spec_prefix: str) -> Optional[dict]:
    tabs = [v["tables"] for k, v in behavior["results"].items() if k.startswith(spec_prefix) and k.endswith("|self")]
    if not tabs:
        return None
    return {sit: {key: float(np.mean([t[sit][key] for t in tabs])) for key in tabs[0][sit]} for sit in tabs[0]}


def sec_behavior(behavior: dict) -> List[str]:
    L = ["## 5. Conditional behaviour (self-play, present turns)", "",
         f"{behavior['meta']['games']} self-play games of {behavior['meta']['window']} turns per final (both seats "
         "focal); the situations overlap by design. Mean over the seeds of a condition.", ""]
    for c in CONDITIONS:
        tab = _self_tables(behavior, f"final:{c}:")
        if not tab:
            continue
        L += [f"### 5.{list(CONDITIONS).index(c) + 1} {c}: {CONDITIONS[c]['label']}", "",
              md_table(["situation", "n", "share", "P(Hold)", "P(Thin)", "P(Clear)", "P(Trade)", "P(Safe)", "P(Yield)",
                        "P(Recall)", "P(Pressure)", "chose Clear", "chose Trade", "H/log|legal|", "reward/turn"],
                       [[s, int(r["n"]), f"{r['share']:.2f}", f"{r['p_Hold']:.2f}", f"{r['p_Thin']:.2f}", f"{r['p_Clear']:.2f}",
                         f"{r['p_bT']:.2f}", f"{r['p_bS']:.2f}", f"{r['p_bY']:.2f}", f"{r['p_bR']:.2f}", f"{r['p_P']:.2f}",
                         f"{r['chose_Clear']:.2f}", f"{r['chose_bT']:.2f}", f"{r['norm_entropy']:.2f}", f"{r['reward']:+.3f}"]
                        for s, r in tab.items() if r["n"] >= 30]), ""]
    L += ["### 5.4 State dependence of the realized choices", "",
          md_table(["policy (vs)", "I(wave; state) / H(wave)", "H(wave)", "I(behaviour; state) / H(behaviour)", "H(behaviour)",
                    "P(Clear) min → max across situations (n ≥ 200)", "P(Trade) min → max"],
                   [[k, f"{v['state_dependence']['wave']['mi_share']:.2f}", f"{v['state_dependence']['wave']['H_action']:.2f}",
                     f"{v['state_dependence']['behavior']['mi_share']:.2f}", f"{v['state_dependence']['behavior']['H_action']:.2f}",
                     (f"{v['state_dependence']['spread_p_Clear']['min']:.2f} ({v['state_dependence']['spread_p_Clear']['min_in'][:28]}) → "
                      f"{v['state_dependence']['spread_p_Clear']['max']:.2f} ({v['state_dependence']['spread_p_Clear']['max_in'][:28]})")
                     if "spread_p_Clear" in v["state_dependence"] else "n/a",
                     (f"{v['state_dependence']['spread_p_bT']['min']:.2f} → {v['state_dependence']['spread_p_bT']['max']:.2f}")
                     if "spread_p_bT" in v["state_dependence"] else "n/a"]
                    for k, v in sorted(behavior["results"].items())]), "",
          "I / H: the mutual information between the realized action category and a coarse state partition (enemy "
          "present, contact cell, power class, HP class, own and enemy siege) as a share of the action's entropy; 0 = "
          "the choice does not depend on the partition, 1 = it is a function of it.", ""]
    return L


def sec_counterfactual(cf: dict) -> List[str]:
    L = ["## 6. Counterfactual first moves", "",
         f"From {cf['meta']['states']} sampled reachable self-play states per situation, {cf['meta']['rollouts']} paired "
         f"rollouts per candidate over W = {cf['meta']['window']} turns (discounted), the policy as continuation and "
         "opponent. Advantage = Q(commit candidate) − Q(policy's own move), mean over states ± 95% (states are the "
         "unit); 'modal best' = share of states in which the policy's modal action is the best candidate; 'regret' = "
         "Q(best) − Q(modal).", ""]
    specs = sorted({k.split("|")[0] for k in cf["results"]})
    for spec in specs:
        rows = []
        for key in cf["situations"]:
            v = cf["results"].get(f"{spec}|{key}")
            if not v or not v["summary"]:
                continue
            s = v["summary"]
            c = s["candidates"]
            rows.append([key, s["n_states"]] + [fmt_ci(c[w]["mean_adv"], c[w]["ci"]) if w in c else "n/a" for w in WAVES]
                        + [fmt_ci(c[n]["mean_adv"], c[n]["ci"]) if n in c else "n/a" for n in ("Hold+Trade", "Clear+Trade", "Hold+Safe", "Recall", "Clear+P")]
                        + [", ".join(f"{k}:{n}" for k, n in sorted(s["modal_actions"].items(), key=lambda x: -x[1])),
                           ", ".join(f"{k}:{n}" for k, n in sorted(s["best_actions"].items(), key=lambda x: -x[1])),
                           f"{s['modal_is_best_share']:.0%}", fmt_ci(s["modal_regret_mean"], s["modal_regret_ci"]),
                           f"{s['mean_pi_modal']:.2f}"])
        L += [f"### {spec}", "",
              md_table(["situation", "states", "Hold", "Thin", "Clear", "Hold+Trade", "Clear+Trade", "Hold+Safe", "Recall",
                        "Clear+P", "modal actions", "best candidates", "modal best", "regret", "π(modal)"], rows), ""]
    return L


def sec_plans(pl: dict) -> List[str]:
    L = ["## 6b. Plan-level counterfactuals (several coordinated turns)", "",
         f"From {pl['meta']['states']} sampled self-play states per situation, the focal champion commits to a multi-turn "
         f"plan (the opponent is the policy throughout; the policy resumes when the plan ends), {pl['meta']['rollouts']} "
         f"paired rollouts over W = {pl['meta']['window']} turns (discounted). Advantage = Q(plan) − Q(policy's own play), "
         "mean over states ± 95% (states are the unit); 'sig. + / −' = share of states with a significant gain / loss. "
         "Plans: slow_push = Thin 2 turns then Hold 12; clear = Clear 14 turns; hold = Hold 14 turns; freeze = the "
         f"threshold freeze controller (Thin while the enemy front at own −1 holds ≥ {pl['meta']['freeze_threshold']} HP, else "
         f"Hold) for {pl['meta']['freeze_turns']} turns; shove_away = Clear while the enemy is absent (≤ 12 turns).", ""]
    rows = []
    for key, v in sorted(pl["results"].items()):
        s = v["summary"]
        for p in v["plans"]:
            c = s[p]
            rows.append([v["spec"], v["situation"], p, s["n_states"], fmt_ci(c["mean_adv"], c["ci"]),
                         f"{c['share_pos']:.0%} / {c['share_neg']:.0%}", f"{c['farm_diff']:+.2f}",
                         f"{c['deaths_focal']:.2f} / {c['deaths_policy']:.2f}"])
    L += [md_table(["policy", "situation", "plan", "states", "plan − policy", "sig. + / −", "farm diff (plan − policy)",
                    "focal deaths: plan / policy"], rows), ""]
    return L


def sec_recall(rec: dict) -> List[str]:
    m = rec["mechanics"]
    L = ["## 7. Recall and death economics", "", "### 7.1 The mechanics",
         "", md_table(["quantity", "value"], [[k, v] for k, v in m.items()]), "",
         "### 7.2 Frequencies in self-play (12 games of 460 turns per final)", "",
         md_table(["policy", "Recall legal (share of present turns)", "P(Recall | legal)", "chose Recall | legal",
                   "recalls/game", "deaths/game", "returns after death", "returns after Recall", "invested after death (per game)",
                   "invested after Recall (per game)", "gold at death", "gold at Recall"],
                  [[k, f"{p['recall_legal_share']:.2f}", f"{p['p_recall_when_legal']:.3f}", f"{p['chose_recall_when_legal']:.3f}",
                    f"{p['per_game_recalls']:.1f}", f"{p['per_game_deaths']:.1f}", f"{p['per_game_returns_after_death']:.1f}",
                    f"{p['per_game_returns_after_recall']:.1f}", f"{p['per_game_invested_after_death']:.1f}",
                    f"{p['per_game_invested_after_recall']:.1f}", _f(p["gold_at_death_mean"], 1, False), _f(p["gold_at_recall_mean"], 1, False)]
                   for k, p in rec["profiles"].items()]), "",
         f"### 7.3 Counterfactual Recalls (W = {rec['meta']['window']}, {rec['meta']['rollouts']} paired rollouts per candidate)", ""]
    specs = list(rec["profiles"])
    for spec in specs:
        rows = []
        for cls, desc in rec["classes"].items():
            v = rec["counterfactual"].get(f"{spec}|{cls}")
            if not v or not v["summary"]:
                continue
            s = v["summary"]
            rows.append([cls, s["n_states"], f"{s['mean_hp_frac']:.2f}", f"{s['mean_gold']:.1f}", f"{s['mean_pi_recall']:.3f}",
                         fmt_ci(s["recall_vs_policy"]["mean"], s["recall_vs_policy"]["ci"]), f"{s['recall_vs_policy']['share_pos']:.0%} / {s['recall_vs_policy']['share_neg']:.0%}",
                         fmt_ci(s["recall_vs_stay"]["mean"], s["recall_vs_stay"]["ci"]),
                         fmt_ci(s["recall_vs_safe"]["mean"], s["recall_vs_safe"]["ci"]),
                         f"{s['deaths_policy']:.2f} / {s['deaths_recall']:.2f}", f"{s['invested_policy']:.1f} / {s['invested_after_recall']:.1f}"])
        reg = rec["regression"].get(spec, {})
        L += [f"#### {spec}", "",
              md_table(["class", "states", "mean HP", "mean unspent gold", "π(Recall)", "Recall − policy", "sig. + / −",
                        "Recall − stay", "Recall − Safe", "deaths in window: policy / Recall", "invested: policy / Recall"], rows), ""]
        if reg:
            L += ["Regression of Q(Recall) − Q(stay) on the state (all sampled Recall-legal states): " +
                  ", ".join(f"{k} {v:+.3f} (SE {reg['se'][k]:.3f})" for k, v in reg["coef"].items()) +
                  f"; n = {reg['n']}, R² = {reg['r2']:.2f}.", ""]
    if rec.get("vs_opponents"):
        L += ["### 7.3b Counterfactual Recalls in games AGAINST A REFERENCE (the reference is the opponent of the rollouts)", "",
              md_table(["policy", "opponent", "class", "states", "mean HP", "unspent gold", "π(Recall)", "Recall − policy",
                        "sig. + / −", "Recall − stay", "Recall − Safe", "deaths in window: policy / Recall"],
                       [[v["spec"], v["opponent"], v["class"], v["summary"].get("n_states", 0),
                         _f(v["summary"].get("mean_hp_frac"), 2, False), _f(v["summary"].get("mean_gold"), 1, False),
                         _f(v["summary"].get("mean_pi_recall"), 3, False),
                         fmt_ci(v["summary"]["recall_vs_policy"]["mean"], v["summary"]["recall_vs_policy"]["ci"]) if v["summary"] else "n/a",
                         f"{v['summary']['recall_vs_policy']['share_pos']:.0%} / {v['summary']['recall_vs_policy']['share_neg']:.0%}" if v["summary"] else "n/a",
                         fmt_ci(v["summary"]["recall_vs_stay"]["mean"], v["summary"]["recall_vs_stay"]["ci"]) if v["summary"] else "n/a",
                         fmt_ci(v["summary"]["recall_vs_safe"]["mean"], v["summary"]["recall_vs_safe"]["ci"]) if v["summary"] else "n/a",
                         f"{v['summary']['deaths_policy']:.2f} / {v['summary']['deaths_recall']:.2f}" if v["summary"] else "n/a"]
                        for k, v in sorted(rec["vs_opponents"].items())]), ""]
    L += [f"### 7.4 The value of the absence: death against Recall from the same state (W = {rec['meta']['window']})", ""]
    for spec, v in rec["death_vs_recall"].items():
        dv = v["summary"]
        L += [f"#### {spec} (farm per present turn used for the Recall-turn adjustment: {v['farm_per_present_turn']:.2f})", "",
              md_table(["states", "n", "V(dead) − V(alive)", "V(recalled) − V(alive)", "full cost of death (+0.5 bounty)",
                        "full cost of Recall (+ the Recall turn)", "death premium over Recall"],
                       [[b, d["n"], fmt_ci(*d["dead_minus_alive"]), fmt_ci(*d["recalled_minus_alive"]), _f(d["death_full_cost"]),
                         _f(d["recall_full_cost"]), _f(d["death_premium_over_recall"])] for b, d in dv.items()]), ""]
    return L


def sec_trajectories(tj: dict) -> List[str]:
    L = ["## 8. Exported trajectories (viewer recordings)", "", tj["how_to_view"] + ".", "",
         md_table(["recording", "A", "B", "seed", "payoff to A", "deaths A/B", "recalls A/B", "crashes A/B", "plates A/B",
                   "A: Hold/Thin/Clear", "A: Trade/Safe/Recall", "B: Hold/Thin/Clear", "B: Trade/Safe/Recall"],
                  [[n, r["names"][0], r["names"][1], r["seed"], f"{r['payoff_A']:+.1f}",
                    f"{r['counters']['deaths'][0]}/{r['counters']['deaths'][1]}", f"{r['counters']['recalls'][0]}/{r['counters']['recalls'][1]}",
                    f"{r['counters']['crashes'][0]}/{r['counters']['crashes'][1]}", f"{r['counters']['plates'][0]}/{r['counters']['plates'][1]}",
                    "/".join(f"{r['action_shares']['0'][w]:.2f}" for w in WAVES), "/".join(f"{r['action_shares']['0'][b]:.2f}" for b in ("Trade", "Safe", "Recall")),
                    "/".join(f"{r['action_shares']['1'][w]:.2f}" for w in WAVES), "/".join(f"{r['action_shares']['1'][b]:.2f}" for b in ("Trade", "Safe", "Recall"))]
                   for n, r in tj["recordings"].items()]), ""]
    for n, r in tj["recordings"].items():
        ev = [f"k{t}: {e}" for t, e in r["events"] if not e.startswith(("A reaches", "B reaches", "A returns", "B returns"))][:40]
        L += [f"**{n}** ({r['label']}): " + "; ".join(ev), ""]
    return L


def report() -> str:
    train, curves, head = load("train"), load("curves"), load("headline")
    ex, behavior, cf, rec, tj = load("exploit"), load("behavior"), load("counterfactual"), load("recall"), load("trajectories")
    pl = load("plans")
    L = ["# M4.3b PPO self-play pilot on `GameConfig.m4_lasthit()` — results", "",
         "Generated by `python -m lanegame.experiments.rl.m43b report` from the stage files in this folder. The "
         "interpretation is in `docs/rl/m4_3b_ppo_pilot_report.md`.", "",
         "Conditions: " + "; ".join(f"**{c}** = {v['label']}" for c, v in CONDITIONS.items()) + ".", ""]
    if train:
        L += sec_train(train)
    if curves:
        L += sec_curves(curves)
    if head:
        L += sec_headline(head)
    if ex:
        L += sec_exploit(ex)
    if behavior:
        L += sec_behavior(behavior)
    if cf:
        L += sec_counterfactual(cf)
    if pl:
        L += sec_plans(pl)
    if rec:
        L += sec_recall(rec)
    if tj:
        L += sec_trajectories(tj)
    text = "\n".join(L)
    write_md("report", text)
    return text


__all__ = ["report"]
