"""M3b: self-play against a population of frozen snapshots.

1. Champion game, equal skill: one run per seed, initialized from the M3a mixture policy of the same seed
   (which becomes the first snapshot), trained against 30% scripted agents, 35% the latest snapshot and 35% a
   uniformly drawn snapshot (pool of 8, a new snapshot every 25 updates). Progress is measured on fixed
   held-out seeds against fixed references (two scripted agents and the frozen initial policy).
2. Champion game, randomized skill: the same, with each training episode's skills drawn from {0.2, 0.5, 0.8}
   per seat (the policy observes both skills), as the basis of the skill experiments in M3c.
3. Legacy siege model at K = 16 (exact): self-play from scratch with the legacy scripted suite, and the EXACT
   exploitability of every snapshot (best response by backward induction to the network as a Markov policy).
   This is the certified check that frozen-snapshot self-play moves toward lower exploitability at all.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ...config import GameConfig
from .common import CKPT, SCRIPTED, curve, md_table, mean_ci, scripted_agents, write

SEEDS = [0, 1, 2]
REFS = ["hold_farmer", "aggressive_trader", "punisher"]
EVAL_SEED = 20_000


def _champion_job(args):
    variant, seed, steps, snapshot_every, eval_every, eval_eps = args
    import torch
    from ...rl import NetAgent, PPOConfig, head_to_head, load_checkpoint, train_selfplay
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    scripted = scripted_agents(cfg)
    init_path = os.path.join(CKPT, f"m3a_mixture_s{seed}", "final.pt")
    init_net = load_checkpoint(init_path)[0]
    init_agent = NetAgent(init_net, "m3a_init")
    refs = {n: scripted[n] for n in REFS}
    refs["m3a_init"] = init_agent

    def eval_fn(net):
        me = NetAgent(net, "learner")
        return {n: head_to_head(cfg, me, a, eval_eps, EVAL_SEED)["mean"] for n, a in refs.items()}

    run = f"m3b_{variant}_s{seed}"
    ppo = PPOConfig(total_steps=steps, seed=100 + seed,
                    skill_levels=(0.2, 0.5, 0.8) if variant == "skillrand" else None)
    res = train_selfplay(cfg, ppo, list(scripted.values()), init_net=init_net, pool_size=8,
                         snapshot_every=snapshot_every, p_scripted=0.3, p_latest=0.35,
                         out_dir=os.path.join(CKPT, run), name=run, eval_fn=eval_fn, eval_every=eval_every,
                         verbose=False)
    h = res.train.history
    return {"variant": variant, "seed": seed, "run": run, "steps": h[-1]["steps"], "elapsed_s": res.train.elapsed,
            "curve": curve(h), "evals": [(e["steps"], e["eval"]) for e in h if "eval" in e],
            "pool": [a.name for a in res.pool.snapshots], "snapshots": res.snapshot_paths,
            "checkpoint": os.path.join(CKPT, run, "final.pt"), "init": init_path,
            "entropy": [(e["steps"], e["entropy"]) for e in h[::10]]}


def _legacy_job(args):
    seed, steps, snapshot_every = args
    import torch
    from ...baselines import siege_baselines
    from ...best_response import exploitability
    from ...env import ScriptedAgent, as_policy
    from ...game import LaneGame
    from ...rl import NetAgent, PPOConfig, load_checkpoint, train_selfplay
    from ...solver import solve_equilibrium
    torch.set_num_threads(1)
    cfg = GameConfig(horizon=16)
    g = LaneGame(cfg)
    layers = g.reachable()
    sol = solve_equilibrium(g, layers)
    scripted = [ScriptedAgent(b) for b in siege_baselines()]
    run = f"m3b_legacy_s{seed}"
    res = train_selfplay(cfg, PPOConfig(total_steps=steps, seed=200 + seed), scripted, pool_size=8,
                         snapshot_every=snapshot_every, p_scripted=0.3, p_latest=0.35,
                         out_dir=os.path.join(CKPT, run), name=run, verbose=False)
    series = []
    for path in res.snapshot_paths:
        net, _, meta = load_checkpoint(path)
        e_sto = exploitability(g, layers, as_policy(NetAgent(net, "s"), cfg), 0, sol.value).exploitability
        e_gre = exploitability(g, layers, as_policy(NetAgent(net, "g", greedy=True), cfg), 0, sol.value).exploitability
        series.append({"step": meta["step"], "exploitability": e_sto, "exploitability_greedy": e_gre})
    from ...baselines import uniform
    e_uni = exploitability(g, layers, uniform(), 0, sol.value).exploitability
    return {"seed": seed, "run": run, "v_star": sol.value, "series": series, "uniform_exploitability": e_uni,
            "curve": curve(res.train.history), "elapsed_s": res.train.elapsed}


def run(steps: int = 307_200, legacy_steps: int = 204_800, workers: int = 9):
    t0 = time.perf_counter()
    jobs = [("equal", s, steps, 25, 50, 100) for s in SEEDS] + [("skillrand", s, steps, 25, 100, 100) for s in SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_c = [ex.submit(_champion_job, j) for j in jobs]
        fut_l = [ex.submit(_legacy_job, (s, legacy_steps, 20)) for s in SEEDS]
        champ = [f.result() for f in fut_c]
        legacy = [f.result() for f in fut_l]
    elapsed = time.perf_counter() - t0

    # progress table (equal skill): evaluation payoff vs each reference over training, mean over seeds
    eq = [r for r in champ if r["variant"] == "equal"]
    steps_list = [s for s, _ in eq[0]["evals"]]
    rows = []
    for i, st in enumerate(steps_list):
        row = [f"{st:,}"]
        for n in REFS + ["m3a_init"]:
            row.append(mean_ci([r["evals"][i][1][n] for r in eq]))
        rows.append(row)
    sk = [r for r in champ if r["variant"] == "skillrand"]
    sk_rows = []
    for i, (st, _) in enumerate(sk[0]["evals"]):
        sk_rows.append([f"{st:,}"] + [mean_ci([r["evals"][i][1][n] for r in sk]) for n in REFS + ["m3a_init"]])
    # legacy exact exploitability over training
    lrows = []
    n_snap = min(len(r["series"]) for r in legacy)
    for i in range(n_snap):
        lrows.append([f"{legacy[0]['series'][i]['step']:,}",
                      mean_ci([r["series"][i]["exploitability"] for r in legacy]),
                      ", ".join(f"{r['series'][i]['exploitability']:.2f}" for r in legacy),
                      mean_ci([r["series"][i]["exploitability_greedy"] for r in legacy])])
    md = "\n".join([
        "# M3b: self-play against frozen snapshots", "",
        f"Generated by `python -m lanegame experiments-rl selfplay` in {elapsed / 60:.1f} min ({workers} processes). "
        "Opponent mixture per training episode: 30% a scripted agent (13 agents, uniform), 35% the latest frozen "
        "snapshot, 35% a snapshot drawn uniformly from the pool (at most 8; the initial policy is kept as an anchor); "
        "a new snapshot every 25 updates (25,600 steps). Frozen snapshots never receive gradients.", "",
        f"## 1. Champion game, equal skill ({steps:,} steps per seed, seeds {SEEDS}, initialized from the M3a mixture policy)", "",
        f"Payoff of the learner against fixed references during training ({100} held-out episodes per seat, "
        "mean over the 3 seeds ± 95% CI across seeds):", "",
        md_table(["training steps"] + REFS + ["frozen initial (M3a) policy"], rows), "",
        f"## 2. Champion game, randomized skill in {{0.2, 0.5, 0.8}} per seat ({steps:,} steps per seed)", "",
        "Same references, evaluated at equal skill 0.5 / 0.5:", "",
        md_table(["training steps"] + REFS + ["frozen initial (M3a) policy"], sk_rows), "",
        f"## 3. Legacy siege model, K = 16: EXACT exploitability of the self-play snapshots ({legacy_steps:,} steps, "
        f"from scratch, snapshot every 20 updates)", "",
        f"Equilibrium value V* = {legacy[0]['v_star']:+.4f}; exact exploitability of uniform play "
        f"{legacy[0]['uniform_exploitability']:.2f}. Exploitability = V* minus the value of an exact best response "
        "against the snapshot as a Markov policy (stochastic network policy; greedy = argmax).", "",
        md_table(["snapshot step", "exploitability (mean ± CI over seeds)", "per seed", "greedy policy"], lrows), ""])
    write("m3b_selfplay", {"champion": champ, "legacy": legacy, "elapsed_s": elapsed, "steps": steps,
                           "legacy_steps": legacy_steps}, md)
    return champ, legacy


if __name__ == "__main__":
    run()
