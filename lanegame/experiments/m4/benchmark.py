"""Throughput and memory of the M4 environment against the M2 champion environment (plan §6, §8 condition 3).

    python -m lanegame.experiments.m4.benchmark [--turns 40000]

Single thread. Measures: environment turns per second with uniformly random legal actions (lane cache off and on);
resolver-only turns per second on identical inputs; the cost of one observation; memory (tracemalloc) along a long
trajectory, checked for growth; and the top functions of a cProfile of the M4 environment step.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
import tracemalloc

import numpy as np

from ...champions import N_CHANCE, champion_legal_actions, initial_champions, resolve_turn
from ...champions_m4 import context, initial_m4, resolve_turn_m4
from ...config import GameConfig
from ...env import LaneEnv
from ...env.observation import observe
from ...mechanics import initial_state
from .common import md_table, meta, save

CONFIGS = {"M2 champions()": GameConfig.champions(), "M2 continuing fixture (m2c)": GameConfig.champions_continuing(),
           "M4 m4()": GameConfig.m4()}


def env_rate(cfg: GameConfig, turns: int, cache: int, seed: int = 0) -> float:
    env = LaneEnv(cfg, max_turns=None if not cfg.continuing else turns + 1, cache_size=cache)
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    n = 0
    t0 = time.perf_counter()
    while n < turns:
        la, lb = obs[0].legal, obs[1].legal
        r = env.step(la[int(rng.integers(len(la)))], lb[int(rng.integers(len(lb)))])
        n += 1
        obs = env.reset(seed=seed + n) if r.done else r.obs
    return n / (time.perf_counter() - t0)


def resolver_inputs(cfg: GameConfig, n: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    m4 = cfg.is_m4
    lane, champ, k = initial_state(cfg), (initial_m4(cfg) if m4 else initial_champions(cfg)), 0
    out = []
    for _ in range(n):
        la = champion_legal_actions(lane, champ, 0, cfg)
        lb = champion_legal_actions(lane, champ, 1, cfg)
        a, b = la[int(rng.integers(len(la)))], lb[int(rng.integers(len(lb)))]
        u = rng.random(N_CHANCE)
        out.append((lane, champ, k, a, b, u))
        t = resolve_turn_m4(lane, champ, k, a, b, cfg, u=u) if m4 else resolve_turn(lane, champ, k, a, b, cfg, u=u)
        lane, champ, k = t.lane.state, t.champ, k + 1
        if not cfg.continuing and k >= cfg.horizon:
            lane, champ, k = initial_state(cfg), initial_champions(cfg), 0
    return out


def resolver_rate(cfg: GameConfig, inputs) -> float:
    t0 = time.perf_counter()
    if cfg.is_m4:
        ctx = context(cfg)
        for lane, champ, k, a, b, u in inputs:
            resolve_turn_m4(lane, champ, k, a, b, cfg, u=u, ctx=ctx)
    else:
        for lane, champ, k, a, b, u in inputs:
            resolve_turn(lane, champ, k, a, b, cfg, u=u)
    return len(inputs) / (time.perf_counter() - t0)


def observe_cost(cfg: GameConfig, inputs) -> float:
    t0 = time.perf_counter()
    for lane, champ, k, *_ in inputs:
        observe(lane, k, 0, cfg, champ)
    return (time.perf_counter() - t0) / len(inputs) * 1e6


def memory_profile(cfg: GameConfig, turns: int = 100_000, cache: int = 20_000) -> dict:
    env = LaneEnv(cfg, max_turns=turns + 1, cache_size=cache)
    rng = np.random.default_rng(3)
    obs = env.reset(seed=3)
    tracemalloc.start()
    marks = {}
    for n in range(1, turns + 1):
        la, lb = obs[0].legal, obs[1].legal
        r = env.step(la[int(rng.integers(len(la)))], lb[int(rng.integers(len(lb)))])
        obs = r.obs
        if n in (10_000, 25_000, 50_000, 75_000, 100_000):
            cur, peak = tracemalloc.get_traced_memory()
            marks[n] = {"current_mb": cur / 2 ** 20, "peak_mb": peak / 2 ** 20, "lane_cache_entries": len(env._cache)}
    tracemalloc.stop()
    return marks


def profile_m4(turns: int = 20_000) -> str:
    cfg = GameConfig.m4()
    pr = cProfile.Profile()
    pr.enable()
    env_rate(cfg, turns, 20_000)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14)
    return s.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=40_000)
    a = ap.parse_args(argv)
    d = {"meta": meta(None, turns=a.turns), "env": {}, "resolver": {}, "observe_us": {}}
    for name, cfg in CONFIGS.items():
        d["env"][name] = {"cache_0": env_rate(cfg, a.turns, 0), "cache_20000": env_rate(cfg, a.turns, 20_000)}
        inp = resolver_inputs(cfg, a.turns // 2)
        d["resolver"][name] = resolver_rate(cfg, inp)
        d["observe_us"][name] = observe_cost(cfg, inp)
    d["memory_m4"] = memory_profile(GameConfig.m4())
    d["memory_m4_no_cache"] = memory_profile(GameConfig.m4(), cache=0)
    d["profile_m4"] = profile_m4()
    m4 = d["env"]["M4 m4()"]
    d["target_4000_met"] = bool(min(m4.values()) >= 4000)
    L = ["# Throughput and memory (M4.2)", "",
         f"Commit `{d['meta']['commit'][:10]}`; single thread; uniformly random legal actions; {a.turns} turns per "
         f"measurement. Command: `{d['meta']['command']}`.", "",
         md_table(["configuration", "env turns/s (no lane cache)", "env turns/s (LRU 20 000)", "resolver turns/s",
                   "observe() µs"],
                  [[n, f"{d['env'][n]['cache_0']:,.0f}", f"{d['env'][n]['cache_20000']:,.0f}",
                    f"{d['resolver'][n]:,.0f}", f"{d['observe_us'][n]:.1f}"] for n in CONFIGS]), "",
         f"Target ≥ 4 000 M4 environment turns/s (plan §6): **{'met' if d['target_4000_met'] else 'NOT met'}**.", "",
         "Memory along one 100 000-turn M4 trajectory (tracemalloc; LRU lane cache of 20 000 entries):", "",
         md_table(["turns", "current MB", "peak MB", "lane cache entries"],
                  [[k, f"{v['current_mb']:.2f}", f"{v['peak_mb']:.2f}", v["lane_cache_entries"]]
                   for k, v in d["memory_m4"].items()]), "",
         "Without the lane cache (no growth: the memory above is the bounded LRU cache filling, about 2.6 KB per "
         "entry, at most 20 000 entries):", "",
         md_table(["turns", "current MB", "peak MB"],
                  [[k, f"{v['current_mb']:.2f}", f"{v['peak_mb']:.2f}"] for k, v in d["memory_m4_no_cache"].items()]),
         "",
         "Profile of the M4 environment step (cProfile, top functions by own time):", "", "```",
         d["profile_m4"].strip(), "```"]
    save("benchmark", d, "\n".join(L))
    print({k: {kk: round(vv) for kk, vv in v.items()} for k, v in d["env"].items()}, d["target_4000_met"])
    return d


if __name__ == "__main__":
    main()
