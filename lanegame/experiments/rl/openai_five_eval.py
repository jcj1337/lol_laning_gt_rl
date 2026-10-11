"""Evaluation of the OpenAI Five-inspired self-play study (stages `evaluate`, `exploit`, `report`).

Every number is a payoff to the first-named policy per game over BOTH seats on held-out episode seeds
(SeedSequence roots 40_000-42_000, disjoint from every training stream), common random numbers across pairs,
evaluated with the frozen networks (no update during evaluation). Equal skill 0.5 / 0.5 unless stated.

  evaluate  A. each final policy against the scripted training suite of m3b (13 agents, in-distribution for
               m3b only) and against 10 HELD-OUT scripted agents never trained against by any condition;
            B. each snapshot against a fixed reference panel (learning curves at matched agent transitions) and
               every pair of snapshots of a run (forgetting and cycles inside a run);
            C. cross-play between all final policies (and six published M3b policies as external references);
            F. a 3 x 3 skill matrix (own x enemy skill in {0.2, 0.5, 0.8}) against a scripted farmer and against a
               fixed learned reference, with the behavior profile in every cell; mirror games for behavior.
  exploit   D. approximate best responses (PPO attackers, warm-started from the target's own weights and from
               scratch, equal budgets for every target), against the best scripted counter-strategy.
  report    tables (results/rl/openai_five/report.md) and figures.

Approximate exploitability is a lower bound from a finite-budget attacker, never a Nash certificate: the champion
game has no exact solver.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

from ...config import GameConfig
from .common import CKPT, SCRIPTED, md_table
from .openai_five import CONDITIONS, LABEL, SEEDS, SKILLS, WORKERS, _load, _write, fmt_seed_ci, run_dir, seed_ci, \
    snapshots_of

TAG = "main"
HELD_OUT = ["shove_hold", "shove_trade", "hold_vision", "freeze_hold", "recall_spammer", "uniform",
            "legacy:aggressive", "legacy:passive_safe", "legacy:freeze_contest", "legacy:stack_builder[vision]"]
CURVE_REFS = ["hold_farmer", "aggressive_trader", "punisher", "opportunistic_trader", "pub_sprand_s1"]
SKILL_REFS = ["hold_farmer", "pub_sprand_s1"]
EVAL_SEED, CURVE_SEED, BR_SEED = 40_000, 41_000, 42_000
EPISODES, CURVE_EPISODES, MATRIX_EPISODES, SKILL_EPISODES = 300, 200, 150, 200
BR_STEPS = 102_400


def published() -> Dict[str, str]:
    """The M3b policies of the previous report (205k steps vs scripted, then 307k frozen-snapshot self-play)."""
    out = {}
    for s in SEEDS:
        out[f"pub_sp_s{s}"] = os.path.join(CKPT, f"m3b_equal_s{s}", "final.pt")
        out[f"pub_sprand_s{s}"] = os.path.join(CKPT, f"m3b_skillrand_s{s}", "final.pt")
    return out


def main_policies(tag: str = TAG) -> Dict[str, str]:
    return {f"{c}_s{s}": os.path.join(run_dir(tag, c, s), "final.pt") for c in CONDITIONS for s in SEEDS}


def _spec(name: str, pols: Dict[str, str]) -> str:
    return pols.get(name, name)


def _h2h(args):
    from .evaluation import _h2h as h2h
    key, x, y, episodes, seed, skill, keep = args
    k, r = h2h((key, x, y, episodes, seed, skill))
    if keep != "all":
        r.pop("y_stats", None)
        if keep == "none":
            r.pop("x_stats", None)
    return k, r


# ---------------------------------------------------------------------------------------------- evaluate
def evaluate(tag: str = TAG, workers: int = WORKERS):
    t0 = time.perf_counter()
    pols = main_policies(tag)
    pub = published()
    allp = {**pols, **pub}
    missing = [p for p in allp.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    jobs = []
    # A. fixed scripted opponents (training suite and held out), main and published policies
    for n in allp:
        for o in SCRIPTED + HELD_OUT:
            jobs.append((("A", n, o), allp[n], o, EPISODES, EVAL_SEED, None, "x"))
    # B. learning curves and within-run snapshot matrices
    snaps = {}
    for c in CONDITIONS:
        for s in SEEDS:
            run = f"{c}_s{s}"
            snaps[run] = snapshots_of(run_dir(tag, c, s))
            for t, p in snaps[run]:
                for o in CURVE_REFS:
                    jobs.append((("B", run, t, o), p, _spec(o, allp), CURVE_EPISODES, CURVE_SEED, None, "none"))
            for i, (ti, pi) in enumerate(snaps[run]):
                for tj, pj in snaps[run][i + 1:]:
                    jobs.append((("M", run, tj, ti), pj, pi, MATRIX_EPISODES, CURVE_SEED, None, "none"))
    # C. cross-play (upper triangle) and mirror games
    names = list(allp)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            jobs.append((("C", a, b), allp[a], allp[b], EPISODES, EVAL_SEED, None, "all"))
        jobs.append((("mirror", a), allp[a], allp[a], EPISODES, EVAL_SEED, None, "x"))
    # F. skill matrix
    for n in pols:
        for ref in SKILL_REFS:
            for mx in SKILLS:
                for my in SKILLS:
                    jobs.append((("F", n, ref, mx, my), pols[n], _spec(ref, allp), SKILL_EPISODES, EVAL_SEED,
                                 (mx, my), "x"))
    print(f"evaluate: {len(jobs)} head-to-head jobs")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(_h2h, jobs, chunksize=4))
    elapsed = time.perf_counter() - t0
    data = {"tag": tag, "policies": pols, "published": pub, "snapshots": snaps, "elapsed_s": elapsed,
            "episodes": {"A": EPISODES, "B": CURVE_EPISODES, "M": MATRIX_EPISODES, "C": EPISODES, "F": SKILL_EPISODES},
            "seeds": {"eval": EVAL_SEED, "curve": CURVE_SEED},
            "results": {"|".join(str(x) for x in k): v for k, v in res.items()}}
    _write(f"evaluate_{tag}", data, f"# evaluate ({tag})\n\nRaw results; see report.md. {len(jobs)} jobs, "
           f"{elapsed / 60:.1f} min.\n")
    return data


# ---------------------------------------------------------------------------------------------- exploit
def _br_job(args):
    name, path, seed, steps, eval_eps, warm = args
    import torch
    from ...rl import PPOConfig, approximate_best_response, load_agent, load_checkpoint
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    target = load_agent(path, name=name)
    init = load_checkpoint(path)[0] if warm else None
    r = approximate_best_response(cfg, target, PPOConfig(total_steps=steps, seed=900 + seed), eval_episodes=eval_eps,
                                  eval_seed=BR_SEED, name=f"br_{name}_{'warm' if warm else 'scratch'}_s{seed}",
                                  init_net=init)
    r.pop("agent")
    r["eval"] = {k: r["eval"][k] for k in ("mean", "stderr", "ci95", "win_rate", "x_stats")}
    return (name, "warm" if warm else "scratch", seed), r


def exploit(tag: str = TAG, steps: int = BR_STEPS, eval_eps: int = 500, workers: int = WORKERS):
    t0 = time.perf_counter()
    pols = main_policies(tag)
    targets = dict(pols)
    targets["pub_sprand_s1"] = published()["pub_sprand_s1"]          # calibration against the M3c number
    jobs = []
    for n, p in targets.items():
        jobs += [(n, p, 0, steps, eval_eps, True), (n, p, 1, steps, eval_eps, True), (n, p, 0, steps, eval_eps, False)]
    print(f"exploit: {len(jobs)} best-response trainings of {steps:,} steps")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(_br_job, jobs))
    elapsed = time.perf_counter() - t0
    data = {"tag": tag, "steps": steps, "eval_episodes": eval_eps, "eval_seed": BR_SEED, "elapsed_s": elapsed,
            "targets": targets, "results": {"|".join(str(x) for x in k): v for k, v in res.items()}}
    _write(f"exploit_{tag}", data, f"# exploit ({tag})\n\nRaw results; see report.md. {elapsed / 60:.1f} min.\n")
    return data


# ---------------------------------------------------------------------------------------------- report
def _r(ev, *key):
    return ev["results"]["|".join(str(x) for x in key)]


def _runs():
    return [(c, s, f"{c}_s{s}") for c in CONDITIONS for s in SEEDS]


def _cross(ev, a, b):
    k = f"C|{a}|{b}"
    if k in ev["results"]:
        return ev["results"][k]["mean"], ev["results"][k]["stderr"]
    r = ev["results"][f"C|{b}|{a}"]
    return -r["mean"], r["stderr"]


def summarize(tag: str = TAG) -> dict:
    """Per-policy and per-condition summary numbers (used by the report and the figures)."""
    ev = _load(f"evaluate_{tag}")
    ex = _load(f"exploit_{tag}") if os.path.exists(os.path.join("results", "rl", "openai_five", f"exploit_{tag}.json")) else None
    tr = _load(f"train_{tag}")
    pols, pub = ev["policies"], ev["published"]
    allnames = list(pols) + list(pub)
    per = {}
    for n in allnames:
        a = {o: _r(ev, "A", n, o) for o in SCRIPTED + HELD_OUT}
        suite = [a[o]["mean"] for o in SCRIPTED]
        held = [a[o]["mean"] for o in HELD_OUT]
        worst_o = min(SCRIPTED + HELD_OUT, key=lambda o: a[o]["mean"])
        per[n] = {"suite_avg": float(np.mean(suite)), "held_avg": float(np.mean(held)),
                  "worst": a[worst_o]["mean"], "worst_opponent": worst_o,
                  "best_scripted_response": -a[worst_o]["mean"], "best_scripted_response_se": a[worst_o]["stderr"],
                  "vs": {o: a[o]["mean"] for o in SCRIPTED + HELD_OUT}}
    # cross-play
    for n in allnames:
        row = {m: _cross(ev, n, m)[0] for m in allnames if m != n}
        per[n]["cross"] = row
        per[n]["cross_main_avg"] = float(np.mean([v for m, v in row.items() if m in pols]))
        per[n]["vs_pub_sprand_s1"] = row.get("pub_sprand_s1", 0.0)
    # exploitability
    if ex is not None:
        for n in list(pols) + ["pub_sprand_s1"]:
            w = [ex["results"][f"{n}|warm|{s}"] for s in (0, 1)]
            sc = ex["results"][f"{n}|scratch|0"]
            ppo_best = max([x["value"] for x in w] + [sc["value"]])
            per.setdefault(n, {})
            per[n].update({"br_warm": [x["value"] for x in w], "br_warm_se": [x["stderr"] for x in w],
                           "br_scratch": sc["value"], "br_scratch_se": sc["stderr"], "br_ppo_best": ppo_best})
            if "best_scripted_response" in per[n]:
                per[n]["exploit_lb"] = max(ppo_best, per[n]["best_scripted_response"])
    train = {f"{r['condition']}_s{r['seed']}": r for r in tr["runs"]}
    return {"ev": ev, "ex": ex, "train": train, "per": per, "pols": pols, "pub": pub}


def _cond_vals(per, cond, key):
    return [per[f"{cond}_s{s}"][key] for s in SEEDS]


def report(tag: str = TAG):
    S = summarize(tag)
    ev, ex, per, train = S["ev"], S["ex"], S["per"], S["train"]
    lines = [f"# OpenAI Five-inspired self-play: results ({tag})", "",
             "Generated by `python -m lanegame.experiments.rl.openai_five report` from evaluate_*.json, exploit_*.json "
             "and train_*.json. Payoffs are resources per game to the row policy over both seats on held-out seeds; "
             "`± x` after a single policy's number is a 95% interval over evaluation episodes; after a condition's "
             "number it is a 95% t interval over the 3 independent training seeds (t = 4.30, wide by design).", ""]
    # ---- training cost
    rows = []
    for c in CONDITIONS:
        rs = [train[f"{c}_s{s}"] for s in SEEDS]
        rows.append([LABEL[c], f"{np.mean([r['agent_transitions'] for r in rs]):,.0f}",
                     f"{np.mean([r['env_turns'] for r in rs]):,.0f}", f"{np.mean([r['updates'] for r in rs]):.0f}",
                     f"{np.mean([r['elapsed_s'] for r in rs]):.0f}",
                     "-" if rs[0]["historical_transition_share"] is None else
                     f"{np.mean([r['historical_transition_share'] for r in rs]):.3f}"])
    lines += ["## Training cost (mean per seed)", "",
              md_table(["condition", "agent transitions", "environment turns", "PPO updates", "train time (s, 1 thread)",
                        "share of transitions from historical games"], rows), ""]
    # ---- main comparison
    rows = []
    for c in CONDITIONS:
        row = [LABEL[c], fmt_seed_ci(_cond_vals(per, c, "suite_avg")), fmt_seed_ci(_cond_vals(per, c, "held_avg")),
               fmt_seed_ci(_cond_vals(per, c, "worst")), fmt_seed_ci(_cond_vals(per, c, "vs_pub_sprand_s1")),
               fmt_seed_ci(_cond_vals(per, c, "cross_main_avg"))]
        if ex is not None:
            row += [fmt_seed_ci(_cond_vals(per, c, "br_ppo_best")), fmt_seed_ci(_cond_vals(per, c, "exploit_lb"))]
        rows.append(row)
    hdr = ["condition", "vs 13 m3b-suite scripted (avg)", "vs 10 held-out scripted (avg)", "worst scripted matchup",
           "vs published M3b sprand_s1", "cross-play vs the 9 main policies (avg)"]
    if ex is not None:
        hdr += ["best PPO attacker (lower bound)", "exploitability lower bound (max with scripted)"]
    lines += ["## Main comparison (equal skill 0.5 / 0.5)", "", md_table(hdr, rows), ""]
    # paired differences
    prow = []
    for a, b in (("sp80", "m3b"), ("sp100", "m3b"), ("sp80", "sp100")):
        for key, lab in (("suite_avg", "suite"), ("held_avg", "held-out"), ("cross_main_avg", "cross-play"),
                         ("vs_pub_sprand_s1", "vs pub_sprand_s1")) + ((("exploit_lb", "exploitability LB"),) if ex else ()):
            d = [per[f"{a}_s{s}"][key] - per[f"{b}_s{s}"][key] for s in SEEDS]
            prow.append([f"{a} − {b}", lab, ", ".join(f"{x:+.2f}" for x in d), fmt_seed_ci(d)])
    lines += ["Paired differences by seed (same seed = same initial network and training seed):", "",
              md_table(["difference", "metric", "per seed (0, 1, 2)", "mean ± 95% t"], prow), ""]
    # ---- per policy
    rows = []
    for c, s, n in _runs():
        p = per[n]
        rows.append([n, f"{p['suite_avg']:+.2f}", f"{p['held_avg']:+.2f}", f"{p['worst']:+.2f} ({p['worst_opponent']})",
                     f"{p['vs_pub_sprand_s1']:+.2f}", f"{p['cross_main_avg']:+.2f}"] +
                    ([f"{p['br_warm'][0]:+.2f}, {p['br_warm'][1]:+.2f}", f"{p['br_scratch']:+.2f}",
                      f"{p['best_scripted_response']:+.2f}", f"{p['exploit_lb']:+.2f}"] if ex else []))
    for n in S["pub"]:
        p = per[n]
        extra = []
        if ex:
            extra = ([f"{p['br_warm'][0]:+.2f}, {p['br_warm'][1]:+.2f}", f"{p['br_scratch']:+.2f}",
                      f"{p['best_scripted_response']:+.2f}", f"{p['exploit_lb']:+.2f}"] if "br_warm" in p else
                     ["", "", f"{p['best_scripted_response']:+.2f}", ""])
        rows.append([n + " (published)", f"{p['suite_avg']:+.2f}", f"{p['held_avg']:+.2f}",
                     f"{p['worst']:+.2f} ({p['worst_opponent']})", f"{p['vs_pub_sprand_s1']:+.2f}",
                     f"{p['cross_main_avg']:+.2f}"] + extra)
    hdr = ["policy", "suite avg", "held-out avg", "worst scripted", "vs pub_sprand_s1", "cross-play avg (main 9)"]
    if ex:
        hdr += ["warm attacker (2 seeds)", "scratch attacker", "best scripted counter", "exploitability LB"]
    lines += ["## Per policy", "", md_table(hdr, rows), ""]
    # ---- scripted detail
    opp = SCRIPTED + HELD_OUT
    rows = [[n] + [f"{per[n]['vs'][o]:+.2f}" for o in opp] for _, _, n in _runs()]
    lines += ["### Every scripted matchup (payoff to the policy; the first 13 are the m3b training suite)", "",
              md_table(["policy"] + opp, rows), ""]
    # ---- cross-play matrix
    names = list(S["pols"]) + list(S["pub"])
    rows = []
    for a in names:
        rows.append([a] + ["" if a == b else f"{per[a]['cross'][b]:+.2f}" for b in names])
    lines += ["## Cross-play (payoff to the row policy, 300 episodes per seat)", "", md_table(["row \\ column"] + names, rows), ""]
    # condition-level block matrix
    rows = []
    for c1 in CONDITIONS + ("pub_sp", "pub_sprand"):
        row = [c1]
        for c2 in CONDITIONS + ("pub_sp", "pub_sprand"):
            vals = [per[a]["cross"][b] for a in names if a.rsplit("_s", 1)[0] == c1
                    for b in names if b.rsplit("_s", 1)[0] == c2 and b != a]
            row.append(f"{np.mean(vals):+.2f}")
        rows.append(row)
    lines += ["Block averages (row family vs column family; diagonal = across seeds of one family):", "",
              md_table(["row \\ column"] + list(CONDITIONS) + ["pub_sp", "pub_sprand"], rows), ""]
    # ---- learning curves
    curves = learning_curves(ev)
    for o in CURVE_REFS:
        steps = sorted({t for c in CONDITIONS for t in curves[c][o]})
        rows = []
        for t in steps:
            rows.append([f"{t:,}"] + [fmt_seed_ci(curves[c][o][t]) if len(curves[c][o].get(t, [])) == len(SEEDS)
                                      else "" for c in CONDITIONS])
        lines += [f"### Snapshots vs `{o}` (learning curve, {CURVE_EPISODES} episodes per seat)", "",
                  md_table(["agent transitions"] + [LABEL[c] for c in CONDITIONS], rows), ""]
    # ---- within-run matrices: forgetting / cycles
    rows = []
    for c, s, n in _runs():
        st = within_run(ev, n)
        rows.append([n, f"{st['final_vs_first']:+.2f}", f"{st['final_vs_mid']:+.2f}", f"{st['min_vs_older']:+.2f}",
                     st["n_pairs"], st["older_wins"], f"{st['monotone_share']:.2f}"])
    lines += ["## Historical checkpoints inside each run", "",
              "`older wins` counts pairs (newer, older) where the older snapshot beats the newer one by more than "
              "2 standard errors; `monotone share` is the share of snapshot triples i < j < k with payoff(k vs i) >= "
              "payoff(j vs i) − 2 SE (no sign of forgetting older opponents).", "",
              md_table(["run", "final vs first snapshot", "final vs middle snapshot", "worst newer-vs-older payoff",
                        "pairs", "older wins", "monotone share"], rows), ""]
    # ---- behavior
    lines += behavior_tables(S)
    # ---- skill
    lines += skill_tables(S)
    # ---- exploitability calibration
    if ex:
        p = per["pub_sprand_s1"]
        lines += ["## Exploitability calibration", "",
                  f"The same attacker harness against the published `pub_sprand_s1` (M3c warm-start estimate +0.11): "
                  f"warm {p['br_warm'][0]:+.2f}, {p['br_warm'][1]:+.2f}; scratch {p['br_scratch']:+.2f} "
                  f"({ex['steps']:,} attacker steps, {ex['eval_episodes']} held-out episodes per seat).", ""]
    _write(f"report_{tag}", {"summary": {n: {k: v for k, v in d.items() if k not in ("cross", "vs")}
                                         for n, d in per.items()}}, "\n".join(lines))
    try:
        figures(S)
    except Exception as e:                                        # figures are optional
        print(f"figures failed: {e!r}")


def learning_curves(ev) -> dict:
    """{condition: {reference: {transitions: [payoff per seed]}}}"""
    out = {c: {o: {} for o in CURVE_REFS} for c in CONDITIONS}
    for c, s, n in _runs():
        for t, _ in ev["snapshots"][n]:
            for o in CURVE_REFS:
                out[c][o].setdefault(t, []).append(_r(ev, "B", n, t, o)["mean"])
    return out


def within_run(ev, n) -> dict:
    snaps = [t for t, _ in ev["snapshots"][n]]
    M, SE = {}, {}
    for i, ti in enumerate(snaps):
        for tj in snaps[i + 1:]:
            r = _r(ev, "M", n, tj, ti)
            M[(tj, ti)], SE[(tj, ti)] = r["mean"], r["stderr"]
    last, first, mid = snaps[-1], snaps[0], snaps[len(snaps) // 2]
    older_wins = sum(1 for k, v in M.items() if v < -2 * SE[k])
    trip = ok = 0
    for i in range(len(snaps)):
        for j in range(i + 1, len(snaps)):
            for k in range(j + 1, len(snaps)):
                trip += 1
                a, b = (snaps[k], snaps[i]), (snaps[j], snaps[i])
                ok += M[a] >= M[b] - 2 * np.hypot(SE[a], SE[b])
    return {"final_vs_first": M.get((last, first), 0.0), "final_vs_mid": M.get((last, mid), 0.0) if mid != last else 0.0,
            "min_vs_older": min(M.values()) if M else 0.0, "n_pairs": len(M), "older_wins": older_wins,
            "monotone_share": ok / trip if trip else 1.0, "matrix": {f"{a}|{b}": v for (a, b), v in M.items()}}


BEH = [("H", "Hold"), ("Y", "Yield"), ("T", "Trade"), ("S", "Safe"), ("R", "Recall")]


def _behavior_row(x: dict) -> List[str]:
    b, w, o = x["behaviors"], x["waves"], x["objectives"]
    return ([f"{b.get(k, 0.0):.2f}" for k, _ in BEH] +
            [f"{w.get('SH', 0.0):.2f}", f"{w.get('F', 0.0):.2f}", f"{o.get('Vision', 0.0):.3f}", f"{o.get('Pressure', 0.0):.3f}",
             f"{x['farm']:.2f}", f"{x['farm'] / x['farm_available']:.2f}" if x.get("farm_available") else "",
             f"{x.get('trade_damage_dealt', 0):.2f}", f"{x.get('damage_taken', 0):.2f}", f"{x.get('deaths', 0):.2f}",
             f"{x.get('gank_damage_taken', 0):.2f}", f"{x['recalls']:.2f}", f"{x['exposure']:+.2f}",
             f"{x['plates']:.2f}"])


BEH_HDR = ["Hold", "Yield", "Trade", "Safe", "Recall", "Shove", "Freeze", "Vision", "Pressure", "farm",
           "farm / available", "trade dmg dealt", "dmg taken", "deaths", "gank dmg taken", "recalls", "exposure", "plates"]


def _avg_stats(xs: List[dict]) -> dict:
    out = {}
    for k in xs[0]:
        if isinstance(xs[0][k], dict):
            keys = set().union(*(x[k] for x in xs))
            out[k] = {kk: float(np.mean([x[k].get(kk, 0.0) for x in xs])) for kk in keys}
        else:
            out[k] = float(np.mean([x.get(k, 0.0) for x in xs]))
    return out


def behavior_tables(S) -> List[str]:
    ev = S["ev"]
    lines = ["## Learned behavior (equal skill)", "",
             "Share of the policy's turns per behavior and wave action, siege objectives, economy and combat per game. "
             "`exposure` is the mean exposure cell (forward positioning; turns away count as −2).", ""]
    for title, key_fn in (("Mirror games (the policy against itself)", lambda n: ("mirror", n)),
                          ("Against `hold_farmer`", lambda n: ("A", n, "hold_farmer")),
                          ("Against `aggressive_trader`", lambda n: ("A", n, "aggressive_trader"))):
        rows = []
        for c in CONDITIONS:
            rows.append([LABEL[c]] + _behavior_row(_avg_stats([_r(ev, *key_fn(f"{c}_s{s}"))["x_stats"] for s in SEEDS])))
        for fam in ("pub_sp", "pub_sprand"):
            rows.append([fam + " (published)"] + _behavior_row(_avg_stats([_r(ev, *key_fn(f"{fam}_s{s}"))["x_stats"]
                                                                           for s in SEEDS])))
        lines += [f"### {title} (mean over 3 seeds)", "", md_table(["condition"] + BEH_HDR, rows), ""]
    rows = []
    for c, s, n in _runs():
        rows.append([n] + _behavior_row(_r(ev, "mirror", n)["x_stats"]))
    lines += ["### Mirror games per policy", "", md_table(["policy"] + BEH_HDR, rows), ""]
    return lines


def skill_tables(S) -> List[str]:
    ev = S["ev"]
    lines = ["## Mechanical skill (own skill x enemy skill; all conditions trained with skills drawn from "
             "{0.2, 0.5, 0.8})", ""]
    for ref in SKILL_REFS:
        for c in CONDITIONS:
            rows, trows, erows = [], [], []
            for mx in SKILLS:
                row, trow, erow = [f"{mx:.1f}"], [f"{mx:.1f}"], [f"{mx:.1f}"]
                for my in SKILLS:
                    rs = [_r(ev, "F", f"{c}_s{s}", ref, mx, my) for s in SEEDS]
                    row.append(fmt_seed_ci([r["mean"] for r in rs]))
                    trow.append(f"{np.mean([r['x_stats']['behaviors'].get('T', 0.0) for r in rs]):.2f}")
                    erow.append(f"{np.mean([r['x_stats']['exposure'] for r in rs]):+.2f} / "
                                f"{np.mean([r['x_stats'].get('deaths', 0.0) for r in rs]):.2f}")
                rows.append(row)
                trows.append(trow)
                erows.append(erow)
            hdr = ["own \\ enemy"] + [f"{m:.1f}" for m in SKILLS]
            lines += [f"### {LABEL[c]} vs `{ref}`", "", "Payoff:", "", md_table(hdr, rows), "",
                      "Trade share of turns:", "", md_table(hdr, trows), "",
                      "Mean exposure / deaths per game:", "", md_table(hdr, erows), ""]
    return lines


# ---------------------------------------------------------------------------------------------- figures
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (as in experiments/rl/figures.py)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def figures(S, tag: str = TAG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join("results", "rl", "openai_five")
    ev, per = S["ev"], S["per"]
    # 1. learning curves against the reference panel
    curves = learning_curves(ev)
    fig, axes = plt.subplots(1, len(CURVE_REFS), figsize=(4.2 * len(CURVE_REFS), 3.6), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, o in zip(axes, CURVE_REFS):
        for c, col in zip(CONDITIONS, SERIES):
            ts = sorted(t for t, v in curves[c][o].items() if len(v) == len(SEEDS))
            ys = np.array([curves[c][o][t] for t in ts])
            x = np.array(ts) / 1000
            ax.plot(x, ys.mean(1), color=col, linewidth=2, label=LABEL[c])
            ax.fill_between(x, ys.min(1), ys.max(1), color=col, alpha=0.15, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        _style(ax, f"vs {o}", "agent transitions (thousands)", "payoff per game")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_learning_curves.png"), dpi=130)
    plt.close(fig)
    # 2. robustness summary: held-out scripted, cross-play, exploitability per policy
    keys = [("held_avg", "vs held-out scripted (avg)"), ("cross_main_avg", "cross-play vs main 9 (avg)")]
    if S["ex"] is not None:
        keys.append(("exploit_lb", "exploitability lower bound"))
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, (k, title) in zip(np.atleast_1d(axes), keys):
        for i, (c, col) in enumerate(zip(CONDITIONS, SERIES)):
            v = _cond_vals(per, c, k)
            m, h = seed_ci(v)
            ax.bar(i, m, color=col, alpha=0.35, width=0.6)
            ax.scatter([i] * len(v), v, color=col, s=22, zorder=3)
            ax.errorbar(i, m, yerr=h, color=INK, capsize=4, linewidth=1)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(["m3b", "100/0", "80/20"])
        ax.axhline(0, color=MUTED, linewidth=0.8)
        _style(ax, title, "", "payoff per game")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_robustness.png"), dpi=130)
    plt.close(fig)
    # 3. cross-play heat map
    names = list(S["pols"]) + list(S["pub"])
    M = np.array([[0.0 if a == b else per[a]["cross"][b] for b in names] for a in names])
    lim = np.abs(M).max()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(M[i, j]) > 0.6 * lim else INK)
    ax.set_title("Cross-play: payoff to row policy", loc="left", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_crossplay.png"), dpi=130)
    plt.close(fig)
    # 4. behavior shares in mirror games
    fig, ax = plt.subplots(figsize=(7, 3.4))
    fig.patch.set_facecolor(SURFACE)
    labels = [lab for _, lab in BEH] + ["Shove", "Freeze"]
    width = 0.27
    for i, (c, col) in enumerate(zip(CONDITIONS, SERIES)):
        xs = [_r(ev, "mirror", f"{c}_s{s}")["x_stats"] for s in SEEDS]
        vals = np.array([[x["behaviors"].get(k, 0.0) for k, _ in BEH] + [x["waves"].get("SH", 0.0), x["waves"].get("F", 0.0)]
                         for x in xs])
        pos = np.arange(len(labels)) + (i - 1) * width
        ax.bar(pos, vals.mean(0), width=width, color=col, label=LABEL[c])
        ax.scatter(np.repeat(pos, len(SEEDS)).reshape(len(labels), -1).T.ravel(), vals.ravel(), color=INK, s=5, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.legend(fontsize=7, frameon=False)
    _style(ax, "Action shares in mirror games (dots: seeds)", "", "share of turns")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_behavior.png"), dpi=130)
    plt.close(fig)
    # 5. training diagnostics
    tr = S["train"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, (k, title) in zip(axes, (("entropy", "policy entropy"), ("approx_kl", "approximate KL per update"),
                                     ("explained_variance", "value explained variance"))):
        for c, col in zip(CONDITIONS, SERIES):
            for s in SEEDS:
                d = np.array(tr[f"{c}_s{s}"]["diagnostics"][k])
                ax.plot(d[:, 0] / 1000, d[:, 1], color=col, linewidth=1.2, alpha=0.8, label=LABEL[c] if s == 0 else None)
        _style(ax, title, "agent transitions (thousands)", "")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_training_diagnostics.png"), dpi=130)
    plt.close(fig)
    print(f"wrote figures to {out}")


# ---------------------------------------------------------------------------------------------- additional runs
EXTRA = {"abl_sp50": ("ablation", "sp50"), "et_sp100": ("envturn", "sp100"), "et_sp80": ("envturn", "sp80")}
EXTRA_LABEL = {"abl_sp50": "shared 50/50 (ablation, 307k transitions)",
               "et_sp100": "shared 100/0 at equal environment turns (614k transitions)",
               "et_sp80": "shared 80/20 at equal environment turns (~553k transitions)"}


def extra_policies() -> Dict[str, str]:
    return {f"{fam}_s{s}": os.path.join(run_dir(tag, c, s), "final.pt") for fam, (tag, c) in EXTRA.items()
            for s in SEEDS}


def evaluate_extra(workers: int = WORKERS):
    """Reduced evaluation of the additional runs on the main protocol: scripted opponents (A), learning curves (B),
    cross-play against the 9 main policies and the published references (C), mirror games."""
    t0 = time.perf_counter()
    pols, main, pub = extra_policies(), main_policies(), published()
    ref = {**main, **pub}
    missing = [p for p in pols.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    jobs, snaps = [], {}
    for n, p in pols.items():
        for o in SCRIPTED + HELD_OUT:
            jobs.append((("A", n, o), p, o, EPISODES, EVAL_SEED, None, "x"))
        for m, q in ref.items():
            jobs.append((("C", n, m), p, q, EPISODES, EVAL_SEED, None, "none"))
        jobs.append((("mirror", n), p, p, EPISODES, EVAL_SEED, None, "x"))
        fam, s = n.rsplit("_s", 1)
        tag, c = EXTRA[fam]
        snaps[n] = snapshots_of(run_dir(tag, c, int(s)))
        for t, sp in snaps[n]:
            for o in CURVE_REFS:
                jobs.append((("B", n, t, o), sp, ref.get(o, o), CURVE_EPISODES, CURVE_SEED, None, "none"))
    print(f"evaluate_extra: {len(jobs)} head-to-head jobs", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(_h2h, jobs, chunksize=4))
    elapsed = time.perf_counter() - t0
    _write("evaluate_extra", {"policies": pols, "snapshots": snaps, "elapsed_s": elapsed,
                              "results": {"|".join(str(x) for x in k): v for k, v in res.items()}},
           f"# evaluate_extra\n\nRaw results; see report_extra.md. {len(jobs)} jobs, {elapsed / 60:.1f} min.\n")


def exploit_extra(steps: int = BR_STEPS, eval_eps: int = 500, workers: int = WORKERS):
    t0 = time.perf_counter()
    jobs = []
    for n, p in extra_policies().items():
        jobs += [(n, p, 0, steps, eval_eps, True), (n, p, 1, steps, eval_eps, True), (n, p, 0, steps, eval_eps, False)]
    print(f"exploit_extra: {len(jobs)} best-response trainings", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(_br_job, jobs))
    elapsed = time.perf_counter() - t0
    _write("exploit_extra", {"steps": steps, "eval_episodes": eval_eps, "eval_seed": BR_SEED, "elapsed_s": elapsed,
                             "results": {"|".join(str(x) for x in k): v for k, v in res.items()}},
           f"# exploit_extra\n\nRaw results; see report_extra.md. {elapsed / 60:.1f} min.\n")


def report_extra():
    S = summarize(TAG)
    per_main = S["per"]
    ev, ex = _load("evaluate_extra"), _load("exploit_extra")
    tr = {}
    for t in ("ablation", "envturn"):
        p = os.path.join("results", "rl", "openai_five", f"train_{t}.json")
        if os.path.exists(p):
            for r in _load(f"train_{t}")["runs"]:
                tr[f"{'abl' if t == 'ablation' else 'et'}_{r['condition']}_s{r['seed']}"] = r
    main = list(S["pols"])
    per = {}
    for n in ev["policies"]:
        a = {o: _r(ev, "A", n, o) for o in SCRIPTED + HELD_OUT}
        worst_o = min(a, key=lambda o: a[o]["mean"])
        cross = {m: _r(ev, "C", n, m)["mean"] for m in main + list(S["pub"])}
        w = [ex["results"][f"{n}|warm|{s}"]["value"] for s in (0, 1)]
        sc = ex["results"][f"{n}|scratch|0"]["value"]
        per[n] = {"suite_avg": float(np.mean([a[o]["mean"] for o in SCRIPTED])),
                  "held_avg": float(np.mean([a[o]["mean"] for o in HELD_OUT])), "worst": a[worst_o]["mean"],
                  "worst_opponent": worst_o, "vs_pub_sprand_s1": cross["pub_sprand_s1"],
                  "cross_main_avg": float(np.mean([cross[m] for m in main])), "br_warm": w, "br_scratch": sc,
                  "exploit_lb": max(w + [sc, -a[worst_o]["mean"]]),
                  "cross_by_cond": {c: float(np.mean([cross[f"{c}_s{s}"] for s in SEEDS])) for c in CONDITIONS}}
    keys = [("suite_avg", "vs m3b-suite scripted"), ("held_avg", "vs held-out scripted"), ("worst", "worst scripted"),
            ("vs_pub_sprand_s1", "vs pub_sprand_s1"), ("cross_main_avg", "cross-play vs main 9"),
            ("exploit_lb", "exploitability LB")]
    rows = []
    for c in CONDITIONS:
        rows.append([LABEL[c] + " (main)"] + [fmt_seed_ci([per_main[f"{c}_s{s}"][k] for s in SEEDS]) for k, _ in keys])
    for fam in EXTRA:
        rows.append([EXTRA_LABEL[fam]] + [fmt_seed_ci([per[f"{fam}_s{s}"][k] for s in SEEDS]) for k, _ in keys])
    lines = ["# OpenAI Five-inspired self-play: additional runs", "",
             "Same protocol, evaluation seeds and attacker budget as report_main.md. abl_sp50: 50% current-policy "
             "games at the main budget (307,200 agent transitions). et_*: the shared conditions trained to about "
             "the m3b runs' number of ENVIRONMENT turns (307,200), i.e. about twice the agent transitions.", "",
             md_table(["condition"] + [lab for _, lab in keys], rows), ""]
    rows = []
    for fam in EXTRA:
        for s in SEEDS:
            n, p = f"{fam}_s{s}", per[f"{fam}_s{s}"]
            t = tr.get(n, {})
            rows.append([n, f"{t.get('agent_transitions', 0):,}", f"{t.get('env_turns', 0):,}",
                         f"{t.get('elapsed_s', 0):.0f}", f"{p['suite_avg']:+.2f}", f"{p['held_avg']:+.2f}",
                         f"{p['worst']:+.2f} ({p['worst_opponent']})", f"{p['vs_pub_sprand_s1']:+.2f}",
                         f"{p['cross_main_avg']:+.2f}", " / ".join(f"{p['cross_by_cond'][c]:+.2f}" for c in CONDITIONS),
                         f"{p['br_warm'][0]:+.2f}, {p['br_warm'][1]:+.2f}", f"{p['br_scratch']:+.2f}",
                         f"{p['exploit_lb']:+.2f}"])
    lines += ["## Per policy", "", md_table(["policy", "agent transitions", "env turns", "train s", "suite avg",
                                             "held-out avg", "worst scripted", "vs pub_sprand_s1", "cross vs main 9",
                                             "cross vs m3b / sp100 / sp80", "warm attackers", "scratch attacker",
                                             "exploitability LB"], rows), ""]
    main_curves = learning_curves(S["ev"])
    for o in CURVE_REFS:
        fams = {"sp100": main_curves["sp100"][o], "sp80": main_curves["sp80"][o]}
        for fam in EXTRA:
            d = {}
            for s in SEEDS:
                for t, _ in ev["snapshots"][f"{fam}_s{s}"]:
                    d.setdefault(t, []).append(_r(ev, "B", f"{fam}_s{s}", t, o)["mean"])
            fams[fam] = d
        steps = sorted({t for d in fams.values() for t in d})
        rows = [[f"{t:,}"] + [fmt_seed_ci(d[t]) if len(d.get(t, [])) == len(SEEDS) else "" for d in fams.values()]
                for t in steps]
        lines += [f"### Snapshots vs {o}", "", md_table(["agent transitions"] + list(fams), rows), ""]
    rows = []
    for fam in EXTRA:
        rows.append([EXTRA_LABEL[fam]] + _behavior_row(_avg_stats([_r(ev, "mirror", f"{fam}_s{s}")["x_stats"]
                                                                   for s in SEEDS])))
    lines += ["## Behavior in mirror games", "", md_table(["condition"] + BEH_HDR, rows), ""]
    _write("report_extra", {"summary": per}, "\n".join(lines))


# ---------------------------------------------------------------------------------------------- per-turn behavior
DETAIL_OPPONENTS = ["self", "hold_farmer", "aggressive_trader", "pub_sprand_s1"]
DETAIL_EPISODES, DETAIL_SEED = 150, 43_000


def _detail_job(args):
    """Per-turn behavior of policy x (both seats, recorded episodes): positioning, Trade conditional on the HP
    difference and on the phase of the game, recall timing and HP, wave actions."""
    key, x_spec, y_spec, episodes, seed = args
    import torch
    from ...env import LaneEnv, play_episode
    from .evaluation import _resolve
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    x = _resolve(x_spec, cfg)
    y = x if y_spec == "self" else _resolve(y_spec, cfg)
    env = LaneEnv(cfg, cache_size=20_000)
    children = np.random.SeedSequence(seed).spawn(episodes)
    c: Dict[str, float] = {}

    def add(k, v=1.0):
        c[k] = c.get(k, 0.0) + v

    first_recall, recall_hp, games = [], [], 0
    for seat in (0, 1):
        for ch in children:
            log = play_episode(env, x, y, ch, record=True) if seat == 0 else play_episode(env, y, x, ch, record=True)
            games += 1
            fr = None
            for info in log.turns:
                me, en = info["champ_before"][seat], info["champ_before"][1 - seat]
                beh = info["behavior_labels"][seat]
                if me[2] > 0 or beh == "Absent":
                    add("absent_turns")
                    continue
                add("present")
                pos = me[0]
                add("pos_back" if pos < 0 else "pos_lane" if pos == 0 else "pos_forward")
                wave = info["action_labels"][seat].split("-")[0]
                add(f"wave_{wave}")
                phase = "early" if info["k"] < 11 else "mid" if info["k"] < 22 else "late"
                add(f"n_{phase}")
                if en[2] == 0:
                    d = me[1] - en[1]
                    b = "behind" if d <= -2 else "ahead" if d >= 2 else "even"
                    add(f"n_hp_{b}")
                    if beh == "Trade":
                        add(f"trade_hp_{b}")
                if beh == "Trade":
                    add("trade")
                    add(f"trade_{phase}")
                if beh == "Recall":
                    add("recall")
                    recall_hp.append(me[1])
                    if fr is None:
                        fr = info["k"]
                add(f"beh_{beh}")
            if fr is not None:
                first_recall.append(fr)
    out = {"games": games, "counts": c, "recall_games": len(first_recall),
           "first_recall_turn": float(np.mean(first_recall)) if first_recall else None,
           "recall_hp": float(np.mean(recall_hp)) if recall_hp else None}
    return key, out


def behavior_detail(workers: int = WORKERS):
    t0 = time.perf_counter()
    pols, pub = main_policies(), published()
    allp = {**pols, **{k: pub[k] for k in ("pub_sprand_s1", "pub_sp_s0")}}
    jobs = [((n, o), p, o if o in ("self",) or o in SCRIPTED else allp.get(o, o), DETAIL_EPISODES, DETAIL_SEED)
            for n, p in allp.items() for o in DETAIL_OPPONENTS]
    print(f"behavior_detail: {len(jobs)} jobs", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(_detail_job, jobs))
    elapsed = time.perf_counter() - t0

    def ratio(r, num, den):
        d = r["counts"].get(den, 0.0)
        return r["counts"].get(num, 0.0) / d if d else float("nan")

    def fam_rows(opp):
        rows = []
        groups = [(LABEL[c], [f"{c}_s{s}" for s in SEEDS]) for c in CONDITIONS] + \
                 [("pub_sprand_s1 (published)", ["pub_sprand_s1"]), ("pub_sp_s0 (published)", ["pub_sp_s0"])]
        for lab, names in groups:
            rs = [res[(n, opp)] for n in names]
            f = lambda num, den: np.nanmean([ratio(r, num, den) for r in rs])
            rows.append([lab, f"{f('pos_back', 'present'):.2f}", f"{f('pos_lane', 'present'):.2f}",
                         f"{f('pos_forward', 'present'):.2f}",
                         f"{f('trade_hp_behind', 'n_hp_behind'):.2f}", f"{f('trade_hp_even', 'n_hp_even'):.2f}",
                         f"{f('trade_hp_ahead', 'n_hp_ahead'):.2f}",
                         f"{f('trade_early', 'n_early'):.2f}", f"{f('trade_mid', 'n_mid'):.2f}",
                         f"{f('trade_late', 'n_late'):.2f}",
                         f"{np.mean([r['recall_games'] / r['games'] for r in rs]):.2f}",
                         " ".join(f"{r['first_recall_turn']:.1f}" if r["first_recall_turn"] is not None else "-" for r in rs),
                         " ".join(f"{r['recall_hp']:.1f}" if r["recall_hp"] is not None else "-" for r in rs),
                         f"{f('wave_SH', 'present'):.2f}", f"{f('wave_F', 'present'):.2f}"])
        return rows

    hdr = ["policy family", "pos behind lane", "pos at lane", "pos forward", "Trade | HP behind ≥2",
           "Trade | HP even", "Trade | HP ahead ≥2", "Trade early (k<11)", "Trade mid", "Trade late (k≥22)",
           "games with a recall", "first recall turn (per seed)", "HP at recall (per seed)", "Shove", "Freeze"]
    lines = ["# Per-turn behavior of the learned policies", "",
             f"Recorded episodes ({DETAIL_EPISODES} per seat, SeedSequence({DETAIL_SEED}), equal skill 0.5). Shares are over "
             "the policy's present turns; `Trade | HP ...` is the Trade share on turns where both champions are present, "
             "by the policy's HP minus the enemy's; positions are the champion's cell at the start of the turn in own "
             "coordinates (lane cell = 0, own tower = −2).", ""]
    for opp in DETAIL_OPPONENTS:
        lines += [f"## Against {'itself (mirror)' if opp == 'self' else opp}", "", md_table(hdr, fam_rows(opp)), ""]
    _write("behavior_detail", {"results": {f"{k[0]}|{k[1]}": v for k, v in res.items()}, "elapsed_s": elapsed},
           "\n".join(lines))
