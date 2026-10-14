"""Evaluation of M4.4 policies: discounted value against the scripted references, cross-play between independent
seeds, against the policy's own historical snapshots and against its untrained initial network.

Every number is `env.discounted.evaluate_discounted` (protocol E1: from s0; each seed played twice, the evaluated
policy in seat A and in seat B on the same chance stream; the value is the mean of the two; 95% interval over seeds;
the guaranteed truncation bound gamma^W Omega is reported with it). These numbers measure STRENGTH against fixed
opponents, not strategy: they are read together with the behaviour diagnostics and the counterfactuals.

Why self-play reward is uninformative: in a current-policy game both seats are the same network, the game is
zero-sum and s0 is symmetric, so the expected training return of seat A is exactly 0 whatever the network has
learned (both seats improve together). Progress is therefore measured against FIXED opponents (references, frozen
snapshots, the initial network).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

from .common import agent_from_spec, env_cfg, snapshot_paths, spec_env

HEAD_WINDOW = 1000              # headline window (guaranteed tail bound 0.004-0.006 resources)
DEV_WINDOW = 460


def _value(cfg, x, y, episodes: int, window: int, seed: int) -> dict:
    from ....env.discounted import evaluate_discounted
    r = evaluate_discounted(cfg, x, y, episodes=episodes, seed=seed, window=window)
    return {"estimate": r["estimate"], "ci95": r["ci95"], "stderr": r["stderr"], "seat_A": r["seat_A"],
            "seat_B": r["seat_B"], "pairs": r["pairs"], "tail_bound": r["tail_bound"]["guaranteed"],
            "elapsed_s": r["elapsed_s"]}


def _job(args) -> tuple:
    env, xs, ys, episodes, window, seed, cross = args
    import torch
    torch.set_num_threads(1)
    cfg = env_cfg(env)
    x = agent_from_spec(xs, cfg, cross_env=cross)
    y = agent_from_spec(ys, cfg, cross_env=cross)
    return (env, xs, ys), _value(cfg, x, y, episodes, window, seed)


def jobs_references(env: str, specs: Sequence[str], refs: Sequence[str], episodes: int, window: int, seed: int) -> list:
    return [(env, s, f"ref:{r}", episodes, window, seed, False) for s in specs for r in refs]


def jobs_crossplay(env: str, specs: Sequence[str], episodes: int, window: int, seed: int) -> list:
    return [(env, a, b, episodes, window, seed, False) for i, a in enumerate(specs) for b in specs[i + 1:]]


def jobs_history(env: str, phase: str, seed_: int, points: Sequence[int], episodes: int, window: int, seed: int) -> list:
    """The final network of (phase, env, seed_) against its snapshots at the given transition counts (historical
    opponents: improvement over the policy's own past) and against its initial network."""
    final = f"final:{phase}:{env}:{seed_}"
    snaps = dict(snapshot_paths(phase, env, seed_))
    out = [(env, final, f"snap:{phase}:{env}:{seed_}:{p}", episodes, window, seed, False) for p in points if p in snaps]
    out.append((env, final, f"initial:{env}:{seed_}", episodes, window, seed, False))
    return out


def jobs_curve(env: str, phase: str, seed_: int, points: Sequence[int], refs: Sequence[str], episodes: int,
               window: int, seed: int) -> list:
    """Learning curve: the run's snapshots at `points` (agent transitions) against fixed references. The snapshot at
    the last update equals the final network, so the curve ends at the final."""
    snaps = dict(snapshot_paths(phase, env, seed_))
    return [(env, f"snap:{phase}:{env}:{seed_}:{p}", f"ref:{r}", episodes, window, seed, False)
            for p in points if p in snaps for r in refs]


def run(jobs: list, workers: int = 9) -> Dict[str, dict]:
    from .common import run_jobs
    t0 = time.perf_counter()
    res = run_jobs(_job, jobs, workers, label="evaluate")
    out = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res}
    print(f"[evaluate] {len(jobs)} matchups in {time.perf_counter() - t0:.0f} s", flush=True)
    return out


__all__ = ["HEAD_WINDOW", "DEV_WINDOW", "jobs_references", "jobs_crossplay", "jobs_history", "jobs_curve", "run"]
