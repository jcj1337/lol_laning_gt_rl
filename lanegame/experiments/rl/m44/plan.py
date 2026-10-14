"""The M4.4b experiment specification: matrix, exact commands, seeds, checkpoint conventions, evaluation schedule,
expected outputs and the runtime estimate measured in M4.4a (results/m44a/m44b_plan.json and .md). Nothing here runs
M4.4b; `python -m lanegame.experiments.rl.m44 train_main` refuses to start without --confirm-m44b."""

from __future__ import annotations

import os

import numpy as np

from .common import (BEHAVIOR_SEED, BUDGET, CF_ROLL_SEED, CF_STATE_SEED, CKPT, ENV_ORDER, ENVS, OUT_B, REF_SEED,
                     SCENARIO_SEED, SEEDS, SNAP_EVERY, TRACE_SEED, XPLAY_SEED, load_json, meta, run_dir, save_json,
                     training_spec, write_md)
from .references import CORE, REFERENCES

COMMANDS = [
    ("train", "python -m lanegame.experiments.rl.m44 train_main --confirm-m44b --workers 9",
     "9 runs (3 environments x seeds 0, 1, 2) of 2,048,000 agent transitions, one process each"),
    ("evaluate", "python -m lanegame.experiments.rl.m44 eval_main --workers 9",
     "load / digest checks, references (W = 1000), cross-play, history, learning curves, behaviour diagnostics, "
     "counterfactuals, viewer traces -> results/m44b/eval_main.json(.gz)"),
    ("one environment only (example)", "python -m lanegame.experiments.rl.m44 train_main --confirm-m44b --envs aggro "
     "--workers 3", "the same runs restricted to one environment (identical settings)"),
]


def _ms_per_turn(eval_rec: dict) -> dict:
    """Measured per-turn costs of the smoke evaluation (single worker process each, 9 in parallel)."""
    from .stages import SIZES
    z = SIZES["smoke"]
    ref, net = [], []
    for k, v in eval_rec["strength"].items():
        env, x, y = k.split("|")
        if x.startswith("snap:") and y.startswith("ref:"):
            turns = 2 * z["curve_eps"] * z["curve_window"]
        elif y.startswith("ref:"):
            turns = 2 * z["ref_eps"] * z["ref_window"]
        elif y.startswith("final:"):
            turns = 2 * z["xplay_eps"] * z["xplay_window"]
        else:
            turns = 2 * z["hist_eps"] * z["hist_window"]
        (ref if y.startswith("ref:") else net).append(1000 * v["elapsed_s"] / turns)
    beh = [1000 * v["elapsed_s"] / v["turns"] for v in eval_rec["behavior"].values()]
    cf = []
    for v in eval_rec["counterfactual"].values():
        n = len(v["states"])
        if n:
            cands = np.mean([len(s["candidates"]) + 1 for s in v["states"]])
            # rollouts of the window + the one-step critic rollouts + the state-sampling games (bounded above)
            cf.append(1000 * v["elapsed_s"] / (n * cands * v["n_roll"] * v["window"]))
    return {"vs_reference_ms_per_turn": float(np.median(ref)), "network_vs_network_ms_per_turn": float(np.median(net)),
            "recorded_game_ms_per_turn": float(np.median(beh)),
            "counterfactual_ms_per_rollout_turn_upper": float(np.median(cf)) if cf else float("nan")}


def estimate(train_rec: dict, eval_rec: dict, workers: int = 9) -> dict:
    from .stages import SIZES
    tp = train_rec["throughput"]["transitions_per_s_per_process_min_mean_max"]
    run_s = [BUDGET / tp[0], BUDGET / tp[1]]
    ms = _ms_per_turn(eval_rec)
    z = SIZES["main"]
    n_runs = len(ENV_ORDER) * len(SEEDS)
    snaps = BUDGET // SNAP_EVERY
    curve_points = snaps // z["curve_every"]
    turns = {
        "references": n_runs * len(z["refs"]) * 2 * z["ref_eps"] * z["ref_window"],
        "learning_curves": n_runs * curve_points * len(z["curve_refs"]) * 2 * z["curve_eps"] * z["curve_window"],
        "cross_play": len(ENV_ORDER) * 3 * 2 * z["xplay_eps"] * z["xplay_window"],
        "history": n_runs * (z["hist_points"] + 1) * 2 * z["hist_eps"] * z["hist_window"],
        "behaviour": n_runs * (z["beh_games"] * z["beh_turns"] + len(z["ref_opps"]) * 2 * z["ref_games"] * z["beh_turns"]
                               + 2 * z["scen_games"] * z["scen_turns"]),
        "counterfactual_rollouts": n_runs * len(z["cf_keys"]) * z["cf_states"] * 4 * z["cf_roll"] * z["cf_window"],
    }
    cpu = {"references": turns["references"] * ms["vs_reference_ms_per_turn"],
           "learning_curves": turns["learning_curves"] * ms["vs_reference_ms_per_turn"],
           "cross_play": turns["cross_play"] * ms["network_vs_network_ms_per_turn"],
           "history": turns["history"] * ms["network_vs_network_ms_per_turn"],
           "behaviour": turns["behaviour"] * ms["recorded_game_ms_per_turn"],
           "counterfactual_rollouts": turns["counterfactual_rollouts"] * ms["counterfactual_ms_per_rollout_turn_upper"]}
    cpu_s = {k: v / 1000.0 for k, v in cpu.items()}
    total_cpu = sum(cpu_s.values())
    ckpt_mb = 0.153 * (snaps + 1) * n_runs
    return {"training": {"per_process_transitions_per_s_min_mean": tp[:2], "per_run_minutes_slowest_mean":
                         [run_s[0] / 60, run_s[1] / 60], "wall_minutes_9_processes": run_s[0] / 60 * 1.05,
                         "total_transitions": n_runs * BUDGET, "cpu_hours": n_runs * run_s[1] / 3600},
            "evaluation": {"measured_costs": ms, "turns": turns, "cpu_minutes": {k: v / 60 for k, v in cpu_s.items()},
                           "cpu_hours_total": total_cpu / 3600, "wall_minutes_on_9_workers": total_cpu / 60 / workers},
            "disk": {"checkpoint_mb_each": 0.153, "snapshots_per_run": snaps, "checkpoints_mb_total": ckpt_mb,
                     "location": os.path.join(CKPT, "main"), "tracked_by_git": False},
            "hardware": {"logical_cpus": os.cpu_count(), "processes": workers},
            "note": "training throughput measured under the same 9-process load in smoke_train; evaluation costs "
                    "scaled from the smoke evaluation's per-turn timings (counterfactual cost is an upper bound: it "
                    "includes state sampling)"}


def write_plan(workers: int = 9) -> dict:
    from .common import OUT_A
    from .stages import SCENARIOS, SIZES
    train_rec = load_json("train_smoke")
    eval_rec = load_json("eval_smoke")
    if train_rec is None or eval_rec is None:
        raise RuntimeError("run smoke_train and smoke_eval first (the estimate uses their measured throughput)")
    est = estimate(train_rec, eval_rec, workers)
    matrix = [{"env": k, "preset": f"GameConfig.{ENVS[k].preset}()", "role": ENVS[k].role, "seed": s,
               "run_dir": run_dir("main", k, s).replace("\\", "/"), "spec": f"final:main:{k}:{s}"}
              for k in ENV_ORDER for s in SEEDS]
    plan = {"meta": meta(stage="plan"), "status": "PLANNED, NOT RUN (M4.4b; awaiting review of M4.4a)",
            "matrix": matrix, "training": training_spec(), "commands": [list(c) for c in COMMANDS],
            "checkpoint_convention": {
                "run_dir": "results/rl/checkpoints/m44/main/<env>_s<seed>/",
                "files": ["run.json (game configuration, PPO and shared settings, model spec)",
                          "history.jsonl (one line per update: losses, entropy, KL, explained variance, gradient "
                          "norms, critic scales, per-seat counts, throughput)",
                          f"snapshots/snap_<agent transitions, 9 digits>.pt every {SNAP_EVERY:,} transitions",
                          "final.pt (format 2: SHA-256 weights digest verified on load, provenance with git commit)"],
                "overwrite": "refused (train_shared raises on a non-empty run directory in phase main)",
                "specs": "final:main:<env>:<seed>, snap:main:<env>:<seed>:<transitions>, initial:<env>:<seed>"},
            "evaluation": {"sizes": {k: (list(v) if isinstance(v, tuple) else v) for k, v in SIZES["main"].items()},
                           "references": list(REFERENCES), "curve_references": list(CORE),
                           "initialized_scenarios": [f"{a}/{b}" for a, b in SCENARIOS],
                           "seeds": {"behaviour": BEHAVIOR_SEED, "scenarios": SCENARIO_SEED, "references": REF_SEED,
                                     "cross_play_and_history": XPLAY_SEED, "cf_states": CF_STATE_SEED,
                                     "cf_rollouts": CF_ROLL_SEED, "traces": TRACE_SEED},
                           "frequency": f"snapshots every {SNAP_EVERY:,} transitions (80 per run); learning curve at "
                                        "every 4th snapshot (every 102,400) against the core references; finals "
                                        "against the full suite, each other, their own snapshots and initial network"},
            "comparisons": [
                "econ vs aggro (same economy, tower retaliation on busy dives only): dive frequency, busy-dive rate, "
                "tower shots taken, defender reply to dives, Vision before pressure, deaths per 1000 turns",
                "four vs econ / aggro (the broader control): Shove share by state and by plate phase, freeze "
                "episodes, concession under a credible threat, references and cross-play values",
                "within each environment: seed-to-seed spread (3 seeds; t interval with 2 degrees of freedom, "
                "t = 4.30) before any between-environment claim"],
            "expected_outputs": ["results/m44b/train_main.json", "results/m44b/eval_main.json(.gz)",
                                 "results/m44b/trajectories/*.json (viewer recordings)",
                                 "results/rl/checkpoints/m44/main/ (git-ignored)"],
            "estimate": est,
            "caveats": ["three seeds per environment is a starting comparison, not a convergence guarantee",
                        "the budget (2,048,000) equals the M4.3b / M4.3c pilot budget; whether policies have "
                        "converged is read from the learning curves (value against fixed references and against "
                        "the own snapshots), not assumed"]}
    save_json("m44b_plan", plan, OUT_A)
    write_md("m44b_plan", plan_markdown(plan), OUT_A)
    return plan


def plan_markdown(p: dict) -> str:
    from ..common import md_table
    e = p["estimate"]
    t, ev = e["training"], e["evaluation"]
    lines = ["# M4.4b experiment specification", "", f"**Status: {p['status']}.**", "",
             f"Generated by `python -m lanegame.experiments.rl.m44 plan` at commit `{p['meta']['commit'][:10]}` "
             f"(working tree with uncommitted M4.4a changes: {p['meta']['working_tree_dirty']}).", "",
             "## Matrix", "",
             md_table(["env", "preset", "seed", "run directory"],
                      [[m["env"], m["preset"], m["seed"], f"`{m['run_dir']}`"] for m in p["matrix"]]), "",
             "Seeds 0, 1, 2 in every environment: a seed fixes the initial network and the collector streams, so the "
             "three environments start each seed from bit-identical weights (paired initialization).", "",
             "## Commands", ""]
    for name, cmd, what in p["commands"]:
        lines += [f"- **{name}**: `{cmd}` — {what}"]
    s = p["training"]["ppo"]
    lines += ["", "## Training settings (identical in every environment)", "",
              md_table(["setting", "value"], [
                  ["algorithm", "PPO, clipped surrogate 0.2, value clip 0.5, 4 epochs, minibatch 256, Adam 3e-4 annealed"],
                  ["self-play", "shared policy, 100% current-policy games (both seats trained)"],
                  ["entropy coefficient", s["ent_coef"]], ["gamma / GAE lambda", f"{s['gamma']} / {s['gae_lambda']}"],
                  ["rollout", f"{s['n_envs']} environments x {s['segment_length']} turns x 2 seats = "
                              f"{p['training']['transitions_per_update']} transitions per update"],
                  ["games", f"{s['game_length']} turns, bootstrapped truncation (never terminated)"],
                  ["reward scale", s["reward_scale"]], ["network", "88 -> 128 -> 128 (tanh) -> 61 masked logits + value"],
                  ["budget", f"{p['training']['budget_main']:,} agent transitions = {p['training']['updates_main']} updates"],
                  ["snapshots", f"every {p['training']['shared']['snapshot_every']:,} transitions"]]), "",
              "## Runtime estimate (measured in M4.4a)", "",
              md_table(["quantity", "value"], [
                  ["training throughput per process (9 concurrent runs), min / mean",
                   f"{t['per_process_transitions_per_s_min_mean'][0]:,.0f} / {t['per_process_transitions_per_s_min_mean'][1]:,.0f} transitions/s"],
                  ["per run (slowest / mean)", f"{t['per_run_minutes_slowest_mean'][0]:.1f} / {t['per_run_minutes_slowest_mean'][1]:.1f} min"],
                  ["training wall clock (9 runs in parallel, +5%)", f"{t['wall_minutes_9_processes']:.0f} min"],
                  ["training CPU", f"{t['cpu_hours']:.1f} CPU-hours for {t['total_transitions']:,} transitions"],
                  ["evaluation CPU (scaled from the smoke timings)", f"{ev['cpu_hours_total']:.1f} CPU-hours"],
                  ["evaluation wall clock on 9 workers", f"{ev['wall_minutes_on_9_workers']:.0f} min"],
                  ["checkpoints (git-ignored)", f"{e['disk']['checkpoints_mb_total']:.0f} MB"]]), "",
              "Evaluation CPU by component (minutes): " + ", ".join(f"{k} {v:.0f}" for k, v in ev["cpu_minutes"].items()),
              "", "## Comparisons", ""] + [f"- {c}" for c in p["comparisons"]] + ["", "## Caveats", ""] + \
        [f"- {c}" for c in p["caveats"]]
    return "\n".join(lines)


__all__ = ["COMMANDS", "estimate", "write_plan", "plan_markdown"]
