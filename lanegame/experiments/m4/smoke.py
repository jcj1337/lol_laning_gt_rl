"""Bounded training smoke checks (NOT research results; plan WP1 stepping-stone check, plan §8 condition 3).

    python -m lanegame.experiments.m4.smoke [--m2c-steps 300000] [--m4-steps 50000]

1. The "m2c" stepping stone (plan WP1): shared self-play PPO with continuing-time segments on the M2 champion game
   with M4 time semantics (a test fixture). Checks: the training runs with truncations bootstrapped and no
   termination; a policy without time features shows no time-conditioned Shove (its Shove rate is stationary
   across 64-turn windows of long games: chi-square test of homogeneity); the critic's value scale is sane
   (V_hat(s0) / reward_scale against the discounted self-play return, which is 0 by symmetry, and explained variance).
2. An M4 integration smoke: the same trainer on GameConfig.m4() for a few rollouts: finite losses, transitions per
   second, explained variance. It verifies the integration only.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ...config import SH, GameConfig
from .common import Timer, md_table, meta, save


def shove_by_window(cfg: GameConfig, agent, games: int = 24, turns: int = 512, window: int = 64, seed: int = 61_000):
    from ...env import LaneEnv, play_episode
    env = LaneEnv(cfg, max_turns=turns, cache_size=20_000)
    nw = turns // window
    shove = np.zeros(nw)
    present = np.zeros(nw)
    for g in range(games):
        log = play_episode(env, agent, agent, seed + g, record=True)
        for t in log.turns:
            w = t["k"] // window if t["k"] < nw * window else None
            if w is None:
                continue
            for s in (0, 1):
                if not t["away"][s] and t["actions"][s] != 45:
                    present[w] += 1
                    shove[w] += int(t["actions"][s] % 3 == SH)
    rate = shove / np.maximum(present, 1)
    from scipy.stats import chi2_contingency
    chi2, p, _, _ = chi2_contingency(np.vstack([shove, present - shove]))
    return {"rate_by_window": rate.tolist(), "present_by_window": present.tolist(), "chi2": float(chi2),
            "p_value": float(p), "stationary_at_1pct": bool(p > 0.01)}


def train(cfg: GameConfig, steps: int, seed: int, out: str):
    from ...rl import PPOConfig, SharedConfig, train_shared
    from ...rl.agent import NetAgent
    import torch
    p = PPOConfig.continuing(cfg, total_steps=steps, n_envs=16, seed=seed, torch_threads=1)
    res = train_shared(cfg, p, SharedConfig(), out_dir=out, name=os.path.basename(out), verbose=False, overwrite=True)
    hist = res.history
    from ...env import LaneEnv
    env = LaneEnv(cfg, max_turns=1)
    o = env.reset(seed=0)[0]
    with torch.no_grad():
        v0 = float(res.net.value(torch.as_tensor(o.vector).unsqueeze(0))[0]) / p.reward_scale
    finite = all(np.isfinite(h["policy_loss"]) and np.isfinite(h["value_loss"]) for h in hist)
    return res, {"steps": res.agent_transitions, "env_turns": res.env_turns, "elapsed_s": res.elapsed,
                 "transitions_per_s": res.agent_transitions / res.elapsed, "finite_losses": bool(finite),
                 "explained_variance_last10": float(np.nanmean([h["explained_variance"] for h in hist[-10:]])),
                 "entropy_first_last": [hist[0]["entropy"], hist[-1]["entropy"]],
                 "value_s0_resources": v0, "ppo": p.to_dict()}, NetAgent(res.net, os.path.basename(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2c-steps", type=int, default=300_000)
    ap.add_argument("--m4-steps", type=int, default=50_000)
    ap.add_argument("--out", default=os.path.join("results", "rl", "checkpoints", "m4_smoke"))
    a = ap.parse_args(argv)
    timer = Timer()
    m2c, m4 = GameConfig.champions_continuing(), GameConfig.m4()
    d = {"meta": meta(None, m2c_steps=a.m2c_steps, m4_steps=a.m4_steps),
         "label": "integration smoke checks, NOT research results"}
    _, d["m2c"], agent = train(m2c, a.m2c_steps, 0, os.path.join(a.out, "m2c"))
    d["m2c"]["shove_stationarity"] = shove_by_window(m2c, agent)
    _, d["m4"], agent4 = train(m4, a.m4_steps, 0, os.path.join(a.out, "m4"))
    d["m4"]["shove_stationarity"] = shove_by_window(m4, agent4, games=8)
    d["elapsed_s"] = timer()
    s = d["m2c"]["shove_stationarity"]
    L = ["# Training smoke checks (M4.2) — integration only, NOT research results", "",
         f"Commit `{d['meta']['commit'][:10]}`. Shared self-play PPO (current-policy fraction 0.8), "
         f"`PPOConfig.continuing` (gamma 0.99, lambda 0.95, segments of 128, games of 512 with seeded staggered first "
         f"games, reward scale 0.1). Command: `{d['meta']['command']}`.", "",
         md_table(["run", "transitions", "env turns", "transitions/s", "finite losses", "explained variance (last 10)",
                   "entropy first → last", "V_hat(s0) (resources)"],
                  [[k, d[k]["steps"], d[k]["env_turns"], f"{d[k]['transitions_per_s']:,.0f}", d[k]["finite_losses"],
                    f"{d[k]['explained_variance_last10']:.2f}",
                    f"{d[k]['entropy_first_last'][0]:.2f} → {d[k]['entropy_first_last'][1]:.2f}",
                    f"{d[k]['value_s0_resources']:+.2f}"] for k in ("m2c", "m4")]), "",
         "Shove rate of the trained policy (self-play, present turns) per 64-turn window of 512-turn games:", "",
         md_table(["run", "rates by window", "chi-square p", "stationary (p > 0.01)"],
                  [[k, ", ".join(f"{x:.3f}" for x in d[k]["shove_stationarity"]["rate_by_window"]),
                    f"{d[k]['shove_stationarity']['p_value']:.3f}", d[k]["shove_stationarity"]["stationary_at_1pct"]]
                   for k in ("m2c", "m4")]), "",
         "By symmetry the self-play value of s0 is 0; V_hat(s0) far from 0 relative to the return scale "
         "(order 1 / (1 - gamma) x 0.1-0.3 = 10-30 resources) would indicate a mis-scaled critic."]
    save("smoke", d, "\n".join(L))
    print("smoke", {k: (d[k]["finite_losses"], round(d[k]["transitions_per_s"]), d[k]["shove_stationarity"]["p_value"])
                    for k in ("m2c", "m4")}, f"({timer():.0f} s)")
    return d


if __name__ == "__main__":
    main()
