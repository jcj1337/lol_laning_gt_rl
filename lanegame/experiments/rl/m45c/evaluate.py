"""Strength evaluation of M4.5c checkpoints against FIXED opponents (self-play reward is uninformative: both seats are
the same network in a symmetric zero-sum game, so the expected training return of a seat is 0 whatever is learned).

Every number is `env.discounted.evaluate_discounted` (protocol E1 from s0; each evaluation seed is played twice on
the same chance stream, the evaluated policy in seat A and in seat B; the value is the mean of the pair, in resources
of discounted gold lead; 95% interval over seeds; seat_A / seat_B reported separately; the guaranteed truncation bound
gamma^W Omega with it). Stages:

    curve     the pilot's snapshots at CURVE_POINTS (and the untrained network) vs the fixed suite (+ the OOD M4.4
              network), W = 460, CURVE_EPISODES paired seeds
    headline  the final and the half-way snapshot vs the suite, W = 1000, HEAD_EPISODES paired seeds
    history   the final vs its own snapshots and its untrained network, W = 460

The same seeds are used for every checkpoint of a stage (common random numbers): differences between checkpoints
against one opponent are paired.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence

from .common import (CURVE_SEED, FINAL_SEED, HIST_SEED, PILOT_SEED, PILOT_STOP, agent_from_spec, env_cfg, run_jobs,
                     snapshot_paths)

DEV_WINDOW = 460                # development window (as M4.4b's curves)
HEAD_WINDOW = 1000              # headline window (guaranteed tail bound ~0.005 resources)
CURVE_EPISODES = 32
HEAD_EPISODES = 64
HIST_EPISODES = 32
# learning-curve points (agent transitions of existing snapshots, common.snapshot_schedule): the first two snapshots,
# then every 102,400 (every 4th snapshot) up to the final network
CURVE_POINTS = (28_672, 53_248, 102_400, 204_800, 307_200, 409_600, 512_000, 614_400, 716_800, 819_200, 921_600,
                1_024_000)
HIST_POINTS = (28_672, 102_400, 258_048, 512_000, 770_048)


def _value(cfg, x, y, episodes: int, window: int, seed: int) -> dict:
    from ....env.discounted import evaluate_discounted
    r = evaluate_discounted(cfg, x, y, episodes=episodes, seed=seed, window=window)
    return {"estimate": r["estimate"], "ci95": r["ci95"], "stderr": r["stderr"], "seat_A": r["seat_A"],
            "seat_B": r["seat_B"], "pairs": r["pairs"], "tail_bound": r["tail_bound"]["guaranteed"],
            "elapsed_s": r["elapsed_s"]}


def _job(args) -> tuple:
    xs, ys, episodes, window, seed = args
    import torch
    torch.set_num_threads(1)
    cfg = env_cfg()
    return (xs, ys), _value(cfg, agent_from_spec(xs, cfg), agent_from_spec(ys, cfg), episodes, window, seed)


def snap_spec(n: int, phase: str = "pilot", seed: int = PILOT_SEED) -> str:
    """The specification of the checkpoint at n transitions: the snapshot (the snapshot taken at the last update IS
    the final network: identical weights, checked by the report)."""
    return f"snap:{phase}:{seed}:{n}"


def curve_jobs(opponents: Sequence[str], points: Sequence[int] = CURVE_POINTS, episodes: int = CURVE_EPISODES,
               window: int = DEV_WINDOW, seed: int = CURVE_SEED, phase: str = "pilot", run_seed: int = PILOT_SEED) -> list:
    have = dict(snapshot_paths(phase, run_seed))
    specs = [f"initial:{run_seed}"] + [snap_spec(n, phase, run_seed) for n in points if n in have]
    return [(x, y, episodes, window, seed) for x in specs for y in opponents]


def headline_jobs(opponents: Sequence[str], episodes: int = HEAD_EPISODES, window: int = HEAD_WINDOW,
                  seed: int = FINAL_SEED) -> list:
    specs = [f"final:pilot:{PILOT_SEED}", snap_spec(PILOT_STOP // 2)]
    return [(x, y, episodes, window, seed) for x in specs for y in opponents]


def history_jobs(points: Sequence[int] = HIST_POINTS, episodes: int = HIST_EPISODES, window: int = DEV_WINDOW,
                 seed: int = HIST_SEED) -> list:
    have = dict(snapshot_paths("pilot", PILOT_SEED))
    final = f"final:pilot:{PILOT_SEED}"
    ys = [snap_spec(n) for n in points if n in have] + [f"initial:{PILOT_SEED}"]
    return [(final, y, episodes, window, seed) for y in ys]


def run(jobs: list, workers: int, label: str = "evaluate") -> Dict[str, dict]:
    t0 = time.perf_counter()
    res = run_jobs(_job, jobs, workers, label)
    out = {f"{k[0]}|{k[1]}": v for k, v in res}
    print(f"[{label}] {len(jobs)} matchups in {time.perf_counter() - t0:.0f} s", flush=True)
    return out


__all__ = ["DEV_WINDOW", "HEAD_WINDOW", "CURVE_EPISODES", "HEAD_EPISODES", "HIST_EPISODES", "CURVE_POINTS",
           "HIST_POINTS", "curve_jobs", "headline_jobs", "history_jobs", "run", "snap_spec"]
