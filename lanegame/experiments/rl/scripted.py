"""M3a: PPO against fixed scripted opponents on the champion game.

1. Single opponents (learning validation): one PPO learner per (opponent, seed), trained only against that
   opponent, evaluated on held-out seeds in both seats, compared with uniform play and with the best scripted
   response to the same opponent (same evaluation seeds).
2. Mixture: one learner per seed against the 13 scripted agents (uniform mixture), evaluated against each of them
   and compared with every scripted agent's results against the same 13 (the "best single scripted agent").

Runs execute in parallel processes (one thread each); every run is reproducible from its seed.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ...config import GameConfig
from .common import CKPT, SCRIPTED, ci, curve, md_table, mean_ci, scripted_agents, smooth_last, write

SINGLE = ["yield_farmer", "aggressive_trader", "hold_farmer"]
SEEDS = [0, 1, 2]
EVAL_SEED = 10_000


def _job(args):
    """Worker: train one PPO learner and evaluate it. Returns JSON-able results."""
    kind, opponents, seed, steps, eval_eps = args
    import torch
    from ...rl import NetAgent, OpponentSampler, PPOConfig, head_to_head, train_ppo
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    pool = scripted_agents(cfg)
    run = f"m3a_{kind}_s{seed}"
    ppo = PPOConfig(total_steps=steps, seed=seed)
    res = train_ppo(cfg, ppo, OpponentSampler([pool[o] for o in opponents]), out_dir=os.path.join(CKPT, run),
                    verbose=False, name=run)
    agent = NetAgent(res.net, run)
    evals = {o: head_to_head(cfg, agent, pool[o], eval_eps, EVAL_SEED) for o in opponents}
    for e in evals.values():
        e.pop("y_stats", None)
    return {"kind": kind, "seed": seed, "opponents": opponents, "steps": ppo.n_updates * ppo.batch_size,
            "elapsed_s": res.elapsed, "curve": curve(res.history), "final_train_return": smooth_last(res.history),
            "checkpoint": os.path.join(CKPT, run, "final.pt"), "eval": evals,
            "entropy_last": res.history[-1]["entropy"], "explained_variance_last": res.history[-1]["explained_variance"]}


def _scripted_row(args):
    """Worker: payoffs of one scripted agent (or uniform) against each opponent (both seats, eval seeds)."""
    name, opponents, eval_eps = args
    from ...env import champion_agents
    from ...rl.evaluate import head_to_head
    cfg = GameConfig.champions()
    agents = champion_agents(cfg)
    return name, {o: (lambda r: (r["mean"], r["stderr"]))(head_to_head(cfg, agents[name], agents[o], eval_eps, EVAL_SEED))
                  for o in opponents}


def run(single_steps: int = 102_400, mix_steps: int = 204_800, eval_eps: int = 500, workers: int = 10):
    t0 = time.perf_counter()
    jobs = [("single_" + o, [o], s, single_steps, eval_eps) for o in SINGLE for s in SEEDS]
    jobs += [("mixture", list(SCRIPTED), s, mix_steps, 300) for s in SEEDS]
    rows = [(n, list(SCRIPTED), 300) for n in SCRIPTED + ["uniform"]]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        runs = list(ex.map(_job, jobs))
        baseline = dict(ex.map(_scripted_row, rows))
    elapsed = time.perf_counter() - t0

    # ---- single opponents
    single_rows, single = [], {}
    for o in SINGLE:
        rs = [r for r in runs if r["kind"] == "single_" + o]
        vals = [r["eval"][o]["mean"] for r in rs]
        best = max((n for n in SCRIPTED if n != o), key=lambda n: baseline[n][o][0])
        single[o] = {"runs": rs, "eval_by_seed": vals, "best_scripted": best, "best_scripted_value": baseline[best][o],
                     "uniform_value": baseline["uniform"][o]}
        per_seed = ", ".join(ci(r["eval"][o]["mean"], r["eval"][o]["stderr"]) for r in rs)
        first = np.mean([r["curve"][0][1] for r in rs])
        single_rows.append([o, mean_ci(vals), per_seed, f"{first:+.2f}", ci(*baseline["uniform"][o]),
                            f"{best} {ci(*baseline[best][o])}", f"{np.mean([r['elapsed_s'] for r in rs]):.0f}"])

    # ---- mixture
    mix = [r for r in runs if r["kind"] == "mixture"]
    opp_rows = []
    avg_scripted = {n: float(np.mean([baseline[n][o][0] for o in SCRIPTED])) for n in SCRIPTED + ["uniform"]}
    best_single = max(SCRIPTED, key=lambda n: avg_scripted[n])
    for o in SCRIPTED:
        vals = [r["eval"][o]["mean"] for r in mix]
        best = max(SCRIPTED, key=lambda n: baseline[n][o][0] if n != o else -1e9)
        opp_rows.append([o, mean_ci(vals), ci(*baseline[best_single][o]), f"{best} {ci(*baseline[best][o])}"])
    mix_avg = [float(np.mean([r["eval"][o]["mean"] for o in SCRIPTED])) for r in mix]
    beh = {}
    for r in mix:
        for o in SCRIPTED:
            for k, v in r["eval"][o]["x_stats"]["behaviors"].items():
                beh[k] = beh.get(k, 0.0) + v / (len(mix) * len(SCRIPTED))

    curves = {f"single {o}": [r["curve"] for r in single[o]["runs"]] for o in SINGLE}
    curves["mixture"] = [r["curve"] for r in mix]
    md = "\n".join([
        "# M3a: masked PPO against fixed scripted opponents (champion game)", "",
        f"Generated by `python -m lanegame experiments-rl scripted` in {elapsed / 60:.1f} min on {workers} processes. "
        f"`GameConfig.champions()` (K = 32, equal skill). PPO defaults (`lanegame.rl.PPOConfig`): 2 x 128 tanh MLP, "
        "16 environments x 64 steps per rollout, 4 epochs, minibatch 256, lr 3e-4 annealed, gamma 1, lambda 0.95, "
        "clip 0.2, entropy 0.01, unscaled rewards; the learner plays a random seat each episode. Evaluation: "
        f"held-out episode seeds (SeedSequence({EVAL_SEED})), both seats, {eval_eps} episodes per seat for single "
        "opponents and 300 per seat for the mixture, the same seeds for every agent compared (common random "
        "numbers). Mean ± 95% CI (per seed: from episodes; across seeds: from the 3 seed means, indicative only).", "",
        f"## 1. Single opponent ({single_steps:,} steps per run, seeds {SEEDS})", "",
        md_table(["opponent", "PPO payoff (3 seeds)", "per seed", "first-rollout training return", "uniform play",
                  "best scripted response", "train time per run (s)"], single_rows), "",
        f"## 2. Mixture of the {len(SCRIPTED)} scripted agents ({mix_steps:,} steps per run, seeds {SEEDS})", "",
        f"Average payoff over the {len(SCRIPTED)} opponents: PPO {mean_ci(mix_avg)} (per seed "
        + ", ".join(f"{v:+.2f}" for v in mix_avg) + f"); best single scripted agent `{best_single}` "
        f"{avg_scripted[best_single]:+.2f}; uniform {avg_scripted['uniform']:+.2f}.", "",
        md_table(["opponent", "PPO (3 seeds)", f"`{best_single}`", "best scripted response to it"], opp_rows), "",
        "Behavior mix of the PPO learners (share of turns, averaged over opponents and seeds): "
        + ", ".join(f"{k} {v:.2f}" for k, v in sorted(beh.items(), key=lambda kv: -kv[1]) if v >= 0.005) + ".", ""])
    write("m3a_scripted", {"single": single, "mixture": mix, "baseline": baseline, "avg_scripted": avg_scripted,
                           "best_single_scripted": best_single, "curves": curves, "elapsed_s": elapsed,
                           "single_steps": single_steps, "mix_steps": mix_steps}, md)
    return single, mix, baseline


if __name__ == "__main__":
    run()
