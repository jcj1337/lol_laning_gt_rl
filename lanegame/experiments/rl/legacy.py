"""M3a/M3c exact sanity benchmark: the same PPO trainer on the legacy siege model at K = 16, where everything
can be computed exactly (backward induction over the 4,951 reachable decision nodes).

For each scripted opponent: the exact best-response value (`best_response_value`), the exact value of each
trained network policy (the network is a Markov policy of the exact game: `as_policy` + `evaluate`, both
seats), the fraction of the exact best-response gain over uniform play that PPO achieved, and the exact
exploitability of the learned policy (`exploitability` against the solved equilibrium value; a best response to
one opponent is expected to be exploitable by others). This validates the RL methodology on a game with
certified answers; it says nothing exact about the champion game.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ...config import GameConfig
from .common import CKPT, ci, curve, md_table, mean_ci, smooth_last, write

HORIZON = 16
OPPONENTS = ["passive_safe", "aggressive", "freeze_contest"]
SEEDS = [0, 1, 2]


def _baseline(name):
    from ...baselines import siege_baselines
    return next(b for b in siege_baselines() if b.name == name)


def _exact_context():
    from ...game import LaneGame
    from ...solver import solve_equilibrium
    cfg = GameConfig(horizon=HORIZON)
    g = LaneGame(cfg)
    layers = g.reachable()
    sol = solve_equilibrium(g, layers)
    return cfg, g, layers, sol


def _exact_values(g, layers, policy, opp, v_star):
    """(value to the policy as A, as B, mean over seats, exploitability as A) exactly."""
    from ...best_response import evaluate, exploitability
    va = evaluate(g, layers, policy, opp)
    vb = -evaluate(g, layers, opp, policy)
    e = exploitability(g, layers, policy, 0, v_star).exploitability
    return va, vb, 0.5 * (va + vb), e


def _job(args):
    opp_name, seed, steps, gank_reward = args
    import torch
    from ...env import ScriptedAgent, as_policy
    from ...rl import NetAgent, OpponentSampler, PPOConfig, train_ppo
    torch.set_num_threads(1)
    cfg, g, layers, sol = _exact_context()
    opp = _baseline(opp_name)
    run = f"m3a_legacy_{gank_reward}_{opp_name}_s{seed}"
    res = train_ppo(cfg, PPOConfig(total_steps=steps, seed=seed, expected_gank_reward=(gank_reward == "expected")),
                    OpponentSampler([ScriptedAgent(opp)]), out_dir=os.path.join(CKPT, run), verbose=False, name=run)
    out = {"opponent": opp_name, "seed": seed, "gank_reward": gank_reward, "elapsed_s": res.elapsed,
           "curve": curve(res.history), "final_train_return": smooth_last(res.history)}
    for mode, greedy in (("stochastic", False), ("greedy", True)):
        pol = as_policy(NetAgent(res.net, run, greedy=greedy), cfg)
        va, vb, vm, e = _exact_values(g, layers, pol, opp, sol.value)
        out[mode] = {"value_A": va, "value_B": vb, "value": vm, "exploitability": e}
    return out


def _selfplay_job(args):
    """Legacy self-play from scratch; EXACT exploitability of every snapshot."""
    seed, steps, gank_reward, ent_coef = args
    import torch
    from ...baselines import siege_baselines
    from ...best_response import exploitability
    from ...env import ScriptedAgent, as_policy
    from ...rl import NetAgent, PPOConfig, load_checkpoint, train_selfplay
    torch.set_num_threads(1)
    cfg, g, layers, sol = _exact_context()
    run = f"m3c_legacy_selfplay_{gank_reward}_ent{ent_coef}_s{seed}"
    res = train_selfplay(cfg, PPOConfig(total_steps=steps, seed=200 + seed, ent_coef=ent_coef,
                                        expected_gank_reward=(gank_reward == "expected")),
                         [ScriptedAgent(b) for b in siege_baselines()], pool_size=8, snapshot_every=20,
                         out_dir=os.path.join(CKPT, run), name=run, verbose=False)
    series = []
    for path in res.snapshot_paths:
        net, _, meta = load_checkpoint(path)
        e = exploitability(g, layers, as_policy(NetAgent(net, "s"), cfg), 0, sol.value).exploitability
        series.append({"step": meta["step"], "exploitability": e})
    return {"seed": seed, "gank_reward": gank_reward, "ent_coef": ent_coef, "series": series,
            "entropy_last": res.train.history[-1]["entropy"], "elapsed_s": res.train.elapsed}


def run(steps: int = 102_400, workers: int = 10, selfplay_steps: int = 204_800):
    from ...baselines import uniform
    from ...best_response import best_response_value, evaluate
    t0 = time.perf_counter()
    cfg, g, layers, sol = _exact_context()
    exact = {}
    for o in OPPONENTS:
        opp = _baseline(o)
        br = best_response_value(g, layers, opp, 1).value(g)         # A best-responds to opp in seat B (= either seat)
        uni = 0.5 * (evaluate(g, layers, uniform(), opp) - evaluate(g, layers, opp, uniform()))
        eq = 0.5 * (evaluate(g, layers, sol.policy(0), opp) - evaluate(g, layers, opp, sol.policy(1)))
        exact[o] = {"best_response": br, "uniform": uni, "equilibrium": eq}
    jobs = [(o, s, steps, gr) for gr in ("realized", "expected") for o in OPPONENTS for s in SEEDS]
    sp_jobs = [(s, selfplay_steps, "expected", ent) for ent in (0.01, 0.05) for s in SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_sp = [ex.submit(_selfplay_job, j) for j in sp_jobs]
        runs = list(ex.map(_job, jobs))
        sp = [f.result() for f in fut_sp]
    elapsed = time.perf_counter() - t0
    rows = []
    for o, gr in [(o, gr) for gr in ("realized", "expected") for o in OPPONENTS]:
        rs = [r for r in runs if r["opponent"] == o and r["gank_reward"] == gr]
        e = exact[o]
        gain = lambda v: (v - e["uniform"]) / (e["best_response"] - e["uniform"])
        rows.append([o, gr, f"{e['best_response']:+.3f}", f"{e['equilibrium']:+.3f}", f"{e['uniform']:+.3f}",
                     mean_ci([r["stochastic"]["value"] for r in rs]), mean_ci([r["greedy"]["value"] for r in rs]),
                     ", ".join(f"{100 * gain(r['stochastic']['value']):.0f}%" for r in rs),
                     mean_ci([r["final_train_return"] for r in rs]),
                     ", ".join(f"{r['stochastic']['exploitability']:.2f}" for r in rs)])
    md = "\n".join([
        f"# M3a/M3c exact benchmark: PPO on the legacy siege model (K = {HORIZON})", "",
        f"Generated by `python -m lanegame experiments-rl legacy` in {elapsed / 60:.1f} min. `GameConfig(horizon={HORIZON})` "
        f"(the exactly solved siege model; {sum(len(l) for l in layers):,} decision nodes, equilibrium value "
        f"V* = {sol.value:+.4f}). PPO as in M3a ({steps:,} steps per run, seeds {SEEDS}), trained against one legacy "
        "scripted opponent with realized ganks (the environment's reward). Every value below is EXACT (expected "
        "payoff over all chance events, averaged over both seats), computed by backward induction with the network "
        "wrapped as a Markov policy (`lanegame.env.as_policy`). 'Share of BR gain' = (V_PPO - V_uniform) / "
        "(V_BR - V_uniform) for the stochastic policy, per seed. Exploitability = V* minus the value a best "
        "response obtains against the learned policy (the learned policy is a best response to ONE opponent, so it "
        "is expected to be exploitable).", "",
        md_table(["opponent", "training reward", "exact best response", "equilibrium policy", "uniform", "PPO stochastic (exact)",
                  "PPO greedy (exact)", "share of BR gain", "PPO training return (last 10 rollouts)",
                  "exact exploitability of PPO policy"], rows), "",
        "'Training reward' realized = the environment's default (realized gank losses, L = 5); expected = "
        "`gank_reward=\"expected\"`, the gank term replaced by its conditional expectation L (p_B - p_A), an exact "
        "variance reduction in the legacy game (same expected returns; see `lanegame.env.core`). Reported "
        "separately as required for any change to the training signal.", "",
        f"## Self-play on the legacy game with the variance-reduced reward ({selfplay_steps:,} steps, from scratch, "
        "pool 8, snapshot every 20 updates)", "",
        "EXACT exploitability of the snapshots (compare `m3b_selfplay.md` section 3: realized rewards, entropy "
        f"0.01: 2.52 -> 3.43). Uniform play: {_uniform_expl(g, layers, sol):.2f}.", "",
        _sp_table(sp), ""])
    write("m3a_legacy_exact", {"exact": exact, "runs": runs, "selfplay": sp, "v_star": sol.value, "steps": steps,
                               "elapsed_s": elapsed}, md)
    return exact, runs


def _uniform_expl(g, layers, sol):
    from ...baselines import uniform
    from ...best_response import exploitability
    return exploitability(g, layers, uniform(), 0, sol.value).exploitability


def _sp_table(sp) -> str:
    rows = []
    for ent in sorted({r["ent_coef"] for r in sp}):
        rs = [r for r in sp if r["ent_coef"] == ent]
        n = min(len(r["series"]) for r in rs)
        for i in range(n):
            rows.append([f"{ent}", f"{rs[0]['series'][i]['step']:,}",
                         mean_ci([r["series"][i]["exploitability"] for r in rs]),
                         ", ".join(f"{r['series'][i]['exploitability']:.2f}" for r in rs)])
        rows.append([f"{ent}", "final policy entropy", f"{np.mean([r['entropy_last'] for r in rs]):.3f}", ""])
    return md_table(["entropy coefficient", "snapshot step", "exact exploitability (mean ± CI over 3 seeds)",
                     "per seed"], rows)


if __name__ == "__main__":
    run()
