"""OpenAI Five-inspired self-play study (docs/openai_five_selfplay_report.md, docs/openai_five_adaptation.md).

Three training conditions, identical except for who the learner plays and whose transitions are trained on:

  m3b    the existing M3b method, unchanged (`train_selfplay`): ONE learner seat per game against 30% scripted
         agents (the 13-agent suite, uniform), 35% the latest frozen snapshot, 35% a uniformly drawn snapshot
         (pool of 8 with the first snapshot kept as an anchor); scripted-only until the first snapshot.
  sp100  shared-policy self-play (`train_shared`), 100% current-policy games: one network plays both seats and
         both seats' transitions are trained on.
  sp80   shared-policy self-play, 80% current-policy games / 20% games against a frozen past version drawn
         uniformly from the last 8 snapshots (only the current network's seat is trained on); 100% current until
         the first snapshot exists.

Common protocol (paired by seed): `GameConfig.champions()` (K = 32), PPO defaults of `PPOConfig`, the same seed
gives the same initial network in every condition, mechanical skills drawn per episode and per seat from
{0.2, 0.5, 0.8} (observed by the policy), a snapshot every 25,600 agent transitions, budget in AGENT
TRANSITIONS (trainable samples). Rollouts hold ~1,024 agent transitions per update in every condition (m3b:
16 environments x 64 turns x 1 seat; sp100: 8 x 64 x 2 seats; sp80: 8 x 64 x ~1.8), so batch size, number of
updates and gradient steps per sample match; environment turns and wall-clock time are reported separately.

Stages (each writes results/rl/openai_five/<stage>.json and .md; runs that already have a final.pt are never
retrained, so an interrupted stage can be resumed by running it again):

  python -m lanegame.experiments.rl.openai_five smoke | pilot | train | evaluate | exploit | report
  python -m lanegame.experiments.rl.openai_five legacy_exact | train_ablation | train_envturn | evaluate_extra |
                                                exploit_extra | report_extra | exploit_strong |
                                                exploit_strong_extra | behavior_detail
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np

from ...config import GameConfig
from ...io import dump_json
from .common import CKPT, SCRIPTED, md_table, scripted_agents

OUT = os.path.join("results", "rl", "openai_five")
CK = os.path.join(CKPT, "of5")
SKILLS = (0.2, 0.5, 0.8)
SEEDS = (0, 1, 2)
SNAP_EVERY = 25_600
CONDITIONS = ("m3b", "sp100", "sp80")
LABEL = {"m3b": "M3b frozen-snapshot (existing)", "sp100": "shared self-play 100/0",
         "sp80": "shared self-play 80/20 (OpenAI Five-inspired)", "sp50": "shared self-play 50/50"}
FRACTION = {"sp100": 1.0, "sp80": 0.8, "sp50": 0.5}
WORKERS = 10


# ---------------------------------------------------------------------------------------------- training
def run_dir(tag: str, cond: str, seed: int) -> str:
    return os.path.join(CK, f"{tag}_{cond}_s{seed}")


def snapshots_of(out: str) -> List[tuple]:
    """[(agent transitions, path)] of a run's snapshots, sorted (both trainers' file naming)."""
    snaps = []
    for p in glob.glob(os.path.join(out, "snapshots", "snap_*.pt")):
        snaps.append((int(os.path.basename(p)[5:-3]), p))
    return sorted(snaps)


def _history(out: str) -> List[dict]:
    with open(os.path.join(out, "history.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _summary(tag: str, cond: str, seed: int, out: str, skipped: bool) -> dict:
    from ...rl import load_checkpoint
    h = _history(out)
    _, _, meta = load_checkpoint(os.path.join(out, "final.pt"))
    last = h[-1]
    if cond == "m3b":
        transitions = turns = last["steps"]
        hist_frac = None
    else:
        transitions, turns = last["agent_transitions"], last["env_turns"]
        hist_frac = sum(e["rollout_transitions_historical"] for e in h) / transitions
    every = max(1, len(h) // 50)
    diag = {k: [[e["steps"], e[k]] for e in h[::every]] for k in
            ("entropy", "approx_kl", "clipfrac", "explained_variance", "value_loss", "policy_loss")}
    return {"tag": tag, "condition": cond, "seed": seed, "run_dir": out, "final": os.path.join(out, "final.pt"),
            "snapshots": snapshots_of(out), "agent_transitions": transitions, "env_turns": turns,
            "updates": len(h), "elapsed_s": last["elapsed"], "historical_transition_share": hist_frac,
            "opponents": meta.get("opponents"), "diagnostics": diag, "skipped": skipped,
            "games_historical": sum(e.get("games_historical", 0) for e in h),
            "games_current": sum(e.get("games_current", 0) for e in h)}


def _train_job(args) -> dict:
    tag, cond, seed, budget = args
    import torch
    from ...rl import PPOConfig, SharedConfig, train_selfplay, train_shared
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    out = run_dir(tag, cond, seed)
    name = os.path.basename(out)
    if os.path.exists(os.path.join(out, "final.pt")):          # completed earlier: never retrain over it
        return _summary(tag, cond, seed, out, skipped=True)
    if os.path.isdir(out) and os.listdir(out):
        raise FileExistsError(f"{out} holds an unfinished run; inspect and remove it before retraining")
    if cond == "m3b":
        ppo = PPOConfig(total_steps=budget, seed=seed, skill_levels=SKILLS)
        train_selfplay(cfg, ppo, list(scripted_agents(cfg, SCRIPTED).values()), pool_size=8,
                       snapshot_every=SNAP_EVERY // ppo.batch_size, p_scripted=0.3, p_latest=0.35, out_dir=out,
                       name=name, verbose=False)
    else:
        ppo = PPOConfig(total_steps=budget, n_envs=8, seed=seed, skill_levels=SKILLS)
        train_shared(cfg, ppo, SharedConfig(current_fraction=FRACTION[cond], snapshot_every=SNAP_EVERY, pool_size=8),
                     out_dir=out, name=name, verbose=False)
    return _summary(tag, cond, seed, out, skipped=False)


def train_runs(tag: str, budget: int, conditions=CONDITIONS, seeds=SEEDS, budgets: Optional[dict] = None,
               workers: int = WORKERS) -> List[dict]:
    jobs = [(tag, c, s, (budgets or {}).get(c, budget)) for c in conditions for s in seeds]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        runs = list(ex.map(_train_job, jobs))
    wall = time.perf_counter() - t0
    for r in runs:
        print(f"  {r['tag']}_{r['condition']}_s{r['seed']}: {r['agent_transitions']:,} transitions, "
              f"{r['env_turns']:,} turns, {r['elapsed_s']:.0f}s" + (" (existing)" if r["skipped"] else ""))
    _write(f"train_{tag}", {"tag": tag, "budget": budget, "budgets": budgets, "wall_s": wall, "runs": runs},
           _train_md(tag, runs, wall))
    return runs


def _train_md(tag: str, runs: List[dict], wall: float) -> str:
    rows = []
    for r in runs:
        d = r["diagnostics"]
        rows.append([f"{r['condition']}_s{r['seed']}", f"{r['agent_transitions']:,}", f"{r['env_turns']:,}",
                     r["updates"], f"{r['elapsed_s']:.0f}",
                     "-" if r["historical_transition_share"] is None else f"{r['historical_transition_share']:.3f}",
                     f"{d['entropy'][-1][1]:.2f}", f"{d['explained_variance'][-1][1]:.2f}",
                     f"{max(x for _, x in d['approx_kl']):.4f}", len(r["snapshots"])])
    return "\n".join([f"# Training runs `{tag}`", "",
                      f"Wall clock {wall / 60:.1f} min ({WORKERS} processes, one torch thread each).", "",
                      md_table(["run", "agent transitions", "environment turns", "updates", "train time (s)",
                                "historical share of transitions", "final entropy", "final explained variance",
                                "max approx KL", "snapshots"], rows), ""])


# ---------------------------------------------------------------------------------------------- helpers
def _write(name: str, data: dict, md: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    dump_json(data, os.path.join(OUT, f"{name}.json"))
    with open(os.path.join(OUT, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {OUT}/{name}.json and .md")


def _load(name: str) -> dict:
    with open(os.path.join(OUT, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def seed_ci(values) -> tuple:
    """(mean, half-width of the 95% t interval over independent training seeds)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(T975.get(len(v) - 1, 1.96) * v.std(ddof=1) / np.sqrt(len(v)))


def fmt_seed_ci(values) -> str:
    m, h = seed_ci(values)
    return f"{m:+.2f}" if np.isnan(h) else f"{m:+.2f} ± {h:.2f}"


# ---------------------------------------------------------------------------------------------- stages
def smoke():
    """Tiny end-to-end check of the three trainers (10,240 agent transitions each, one seed)."""
    return train_runs("smoke", 10_240, seeds=(0,))


def pilot():
    """One moderate run per condition (51,200 agent transitions, seed 0) and a quick look at its strength."""
    runs = train_runs("pilot", 51_200, seeds=(0,))
    refs = ["hold_farmer", "aggressive_trader", "punisher", "opportunistic_trader"]
    jobs = []
    for r in runs:
        for t, p in r["snapshots"] + [(r["agent_transitions"], r["final"])]:
            for o in refs:
                jobs.append(((r["condition"], t, o), p, o, 150, 40_500, None))
    from .evaluation import _h2h
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = dict(ex.map(_h2h, jobs, chunksize=2))
    rows = []
    for (c, t, o), v in sorted(res.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        rows.append([c, f"{t:,}", o, f"{v['mean']:+.2f} ± {1.96 * v['stderr']:.2f}"])
    _write("pilot_eval", {"results": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}},
           "# Pilot evaluation (150 episodes per seat, equal skill 0.5)\n\n" +
           md_table(["condition", "agent transitions", "opponent", "payoff"], rows) + "\n")


# ---------------------------------------------------------------------------------------------- legacy calibration
LEGACY_HORIZON, LEGACY_STEPS, LEGACY_SNAP = 16, 204_800, 20_480


def _legacy_exact_job(args) -> dict:
    """Exact calibration on the legacy siege model at K = 16 (4,951 decision nodes): train from scratch with the
    m3b method (an exact replica of the M3b legacy job, written to a NEW directory) or with shared self-play, then
    compute the EXACT exploitability of every snapshot as a Markov policy (stochastic and greedy)."""
    cond, seed = args
    import torch
    from ...baselines import siege_baselines, uniform
    from ...best_response import exploitability
    from ...env import ScriptedAgent, as_policy
    from ...game import LaneGame
    from ...rl import NetAgent, PPOConfig, SharedConfig, load_checkpoint, train_selfplay, train_shared
    from ...solver import solve_equilibrium
    torch.set_num_threads(1)
    cfg = GameConfig(horizon=LEGACY_HORIZON)
    g = LaneGame(cfg)
    layers = g.reachable()
    sol = solve_equilibrium(g, layers)
    out = run_dir("legacy", cond, seed)
    t0 = time.perf_counter()
    if not os.path.exists(os.path.join(out, "final.pt")):
        if os.path.isdir(out) and os.listdir(out):
            raise FileExistsError(f"{out} holds an unfinished run")
        if cond == "m3b":
            train_selfplay(cfg, PPOConfig(total_steps=LEGACY_STEPS, seed=200 + seed),
                           [ScriptedAgent(b) for b in siege_baselines()], pool_size=8,
                           snapshot_every=LEGACY_SNAP // 1024, p_scripted=0.3, p_latest=0.35, out_dir=out,
                           name=os.path.basename(out), verbose=False)
        else:
            train_shared(cfg, PPOConfig(total_steps=LEGACY_STEPS, n_envs=8, seed=200 + seed),
                         SharedConfig(current_fraction=FRACTION[cond], snapshot_every=LEGACY_SNAP, pool_size=8),
                         out_dir=out, name=os.path.basename(out), verbose=False)
    train_s = time.perf_counter() - t0
    series = []
    for t, path in snapshots_of(out):
        net = load_checkpoint(path)[0]
        e_sto = exploitability(g, layers, as_policy(NetAgent(net, "s"), cfg), 0, sol.value).exploitability
        e_gre = exploitability(g, layers, as_policy(NetAgent(net, "g", greedy=True), cfg), 0, sol.value).exploitability
        series.append({"step": t, "exploitability": e_sto, "exploitability_greedy": e_gre})
    h = _history(out)
    turns = h[-1]["steps"] if cond == "m3b" else h[-1]["env_turns"]
    return {"condition": cond, "seed": seed, "v_star": sol.value, "series": series, "train_s": train_s,
            "env_turns": turns, "uniform_exploitability": exploitability(g, layers, uniform(), 0, sol.value).exploitability,
            "final_entropy": h[-1]["entropy"]}


def legacy_exact(workers: int = 9):
    t0 = time.perf_counter()
    jobs = [(c, s) for c in ("m3b", "sp100", "sp80") for s in SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        runs = list(ex.map(_legacy_exact_job, jobs))
    wall = time.perf_counter() - t0
    published = {}
    p = os.path.join("results", "rl", "m3b_selfplay.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            published = {r["seed"]: r["series"] for r in json.load(f)["legacy"]}
    repro = []
    for r in runs:
        if r["condition"] == "m3b" and r["seed"] in published:
            a = [x["exploitability"] for x in r["series"]]
            b = [x["exploitability"] for x in published[r["seed"]]]
            repro.append({"seed": r["seed"], "identical": a == b,
                          "max_abs_diff": float(np.max(np.abs(np.subtract(a, b)))) if len(a) == len(b) else None})
    steps = [x["step"] for x in runs[0]["series"]]
    rows = []
    for i, st in enumerate(steps):
        row = [f"{st:,}"]
        for c in ("m3b", "sp100", "sp80"):
            rs = [r for r in runs if r["condition"] == c]
            vals = [r["series"][i]["exploitability"] for r in rs if i < len(r["series"])]
            row.append(f"{fmt_seed_ci(vals)} ({', '.join(f'{v:.2f}' for v in vals)})")
        rows.append(row)
    grows = []
    for c in ("m3b", "sp100", "sp80"):
        rs = [r for r in runs if r["condition"] == c]
        grows.append([LABEL[c], fmt_seed_ci([r["series"][-1]["exploitability_greedy"] for r in rs]),
                      fmt_seed_ci([min(x["exploitability"] for x in r["series"]) for r in rs]),
                      f"{np.mean([r['final_entropy'] for r in rs]):.2f}",
                      f"{np.mean([r['env_turns'] for r in rs]):,.0f}"])
    md = "\n".join([
        "# Exact calibration on the legacy siege model (K = 16)", "",
        f"Generated by `python -m lanegame.experiments.rl.openai_five legacy_exact` in {wall / 60:.1f} min. "
        f"`GameConfig(horizon=16)`: an exactly solvable game (equilibrium value V* = {runs[0]['v_star']:+.4f}; exact "
        f"exploitability of uniform play {runs[0]['uniform_exploitability']:.2f}). Each condition trains from scratch "
        f"for {LEGACY_STEPS:,} agent transitions (seeds 200-202, the seeds of the published M3b legacy runs; same seed = "
        f"same initial network), a snapshot every {LEGACY_SNAP:,} transitions; the network is a Markov policy of the "
        "exact game, so each snapshot's exploitability is EXACT (best response by backward induction). m3b is the "
        "unchanged M3b method with the legacy scripted suite.", "",
        "Exploitability of the stochastic snapshot policy (mean ± 95% t over 3 seeds; per seed in parentheses):", "",
        md_table(["agent transitions", LABEL["m3b"], LABEL["sp100"], LABEL["sp80"]], rows), "",
        md_table(["condition", "final greedy-policy exploitability", "lowest snapshot exploitability",
                  "final entropy", "environment turns"], grows), "",
        "Reproducibility of the re-run m3b legacy runs against the published M3b series (results/rl/m3b_selfplay.json): "
        + ", ".join(f"seed {x['seed']}: {'bit-identical' if x['identical'] else 'max diff ' + str(x['max_abs_diff'])}"
                    for x in repro) + ".", ""])
    _write("legacy_exact", {"runs": runs, "reproducibility": repro, "wall_s": wall}, md)
    return runs


def run():
    """All stages of the study in order (the CLI entry `experiments-rl openai_five`)."""
    from . import openai_five_eval as E
    pilot()
    train_runs("main", MAIN_BUDGET)
    E.evaluate()
    E.exploit()
    E.exploit_strong()
    E.behavior_detail()
    E.report()
    legacy_exact()
    train_runs("ablation", MAIN_BUDGET, conditions=("sp50",))
    train_runs("envturn", MAIN_BUDGET, conditions=("sp100", "sp80"),
               budgets={"sp100": 2 * MAIN_BUDGET, "sp80": int(1.8 * MAIN_BUDGET)})
    E.evaluate_extra()
    E.exploit_extra()
    E.exploit_strong_extra()
    E.report_extra()


MAIN_BUDGET = 307_200


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if stage == "smoke":
        smoke()
    elif stage == "pilot":
        pilot()
    elif stage == "train":
        train_runs("main", MAIN_BUDGET)
    elif stage == "legacy_exact":
        legacy_exact()
    elif stage == "train_ablation":
        train_runs("ablation", MAIN_BUDGET, conditions=("sp50",))
    elif stage == "train_envturn":
        # secondary comparison at (about) equal ENVIRONMENT turns: the shared conditions get twice the samples
        train_runs("envturn", MAIN_BUDGET, conditions=("sp100", "sp80"),
                   budgets={"sp100": 2 * MAIN_BUDGET, "sp80": int(1.8 * MAIN_BUDGET)})
    else:
        from . import openai_five_eval
        getattr(openai_five_eval, stage)()
