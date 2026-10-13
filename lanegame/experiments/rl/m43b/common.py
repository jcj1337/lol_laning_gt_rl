"""Shared definitions of the M4.3b pilot: the environment, the training conditions and budgets, output paths, seeds
of every evaluation stream, small statistics and I/O helpers.

Seeds. Training seeds are 0, 1, 2 (PPOConfig.seed; the collector uses seed + 1). Attackers train with seeds 700 +.
Every evaluation stream of this study derives from SeedSequence roots 61,000-69,999: disjoint from the training
streams and from the roots of the earlier studies (10,000-58,999, 77,777).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....config import GameConfig
from ....io import dump_json
from ..common import md_table

NAME = "m43b"
OUT = os.path.join("results", "rl", NAME)
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
Z95 = 1.96

# ---- training grid
BUDGET = 2_048_000               # agent transitions per run (1,000 updates of 16 x 128 turns x 2 seats)
N_ENVS = 16
SNAP_EVERY = 25_600              # as in M4.3a (the historical pool of hist80 spans the last 8 x 25,600 transitions)
POOL = 8
CURVE_EVERY = 4                  # every 4th snapshot is a learning-curve point (every 102,400 transitions)
SEEDS = (0, 1, 2)
CONDITIONS: Dict[str, dict] = {
    "base": dict(current_fraction=1.0, ent_coef=0.01, label="100% current, entropy coefficient 0.01 (baseline)"),
    "ent03": dict(current_fraction=1.0, ent_coef=0.03, label="100% current, entropy coefficient 0.03 (control)"),
    "hist80": dict(current_fraction=0.8, ent_coef=0.01, label="80% current / 20% historical (pool 8), entropy 0.01"),
    # the follow-up (archive.py): half of the training games start from a witnessed archived state
    "mixed": dict(current_fraction=1.0, ent_coef=0.01, start_mix=0.5,
                  start_archive=os.path.join("results", "rl", NAME, "archive_train.json.gz"),
                  label="100% current, entropy 0.01, 50% of games from the archive of reachable states (follow-up)"),
}
PILOT = ("base", "ent03", "hist80")      # the pilot grid; "mixed" is the follow-up
PRIMARY = "base"

# ---- evaluation
WINDOW_DEV = 460                 # development window (guaranteed tail bound 1.05 resources)
WINDOW_HEAD = 1000               # headline window (bound 0.005)
CURVE_EPISODES = 16              # paired episodes per matchup on the curves (dev window)
HEAD_EPISODES = 32               # paired episodes per headline matchup
CURVE_SEED, HEAD_SEED, PROFILE_SEED, BEHAVIOR_SEED = 61_000, 62_000, 63_000, 64_000
CF_STATE_SEED, CF_ROLL_SEED, RECALL_STATE_SEED, RECALL_ROLL_SEED = 65_000, 65_500, 66_000, 66_500
TRAJ_SEED, ATTACK_TRAIN_SEED, ATTACK_EVAL_SEED, ARCHIVE_SEED = 67_000, 700, 68_000, 69_000
ATTACK_BUDGET = 1_024_000        # agent transitions per attacker (warm-started from its target)


def cfg_game() -> GameConfig:
    return GameConfig.m4_lasthit()


# ---- runs and files
def run_name(cond: str, seed: int) -> str:
    return f"{cond}_s{seed}"


def run_dir(cond: str, seed: int) -> str:
    return os.path.join(CKPT, run_name(cond, seed))


def final_path(cond: str, seed: int) -> str:
    return os.path.join(run_dir(cond, seed), "final.pt")


def attacker_dir(cond: str, seed: int) -> str:
    return os.path.join(CKPT, "attackers", run_name(cond, seed))


def snapshot_paths(cond: str, seed: int) -> List[Tuple[int, str]]:
    """[(agent transitions, path)] of the run's snapshots, in order."""
    d = os.path.join(run_dir(cond, seed), "snapshots")
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if f.startswith("snap_") and f.endswith(".pt"):
            out.append((int(f[5:-3]), os.path.join(d, f)))
    return out


def curve_points(cond: str, seed: int, every: int = CURVE_EVERY) -> List[Tuple[int, str]]:
    """Every `every`-th snapshot plus the final network: the learning-curve points of a run."""
    snaps = snapshot_paths(cond, seed)
    pts = [s for i, s in enumerate(snaps, start=1) if i % every == 0]
    fp = final_path(cond, seed)
    if os.path.exists(fp):
        info = read_json(os.path.join(run_dir(cond, seed), "run.json"))
        total = int(info.get("ppo", {}).get("total_steps", 0))
        if not pts or pts[-1][0] < total:
            pts.append((total, fp))
    return pts


def history(cond: str, seed: int) -> List[dict]:
    p = os.path.join(run_dir(cond, seed), "history.jsonl")
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ppo_config(cond: str, seed: int, steps: int = BUDGET, **kw):
    from ....rl import PPOConfig
    c = CONDITIONS[cond]
    base = dict(total_steps=steps, n_envs=N_ENVS, seed=seed, torch_threads=1, ent_coef=c["ent_coef"])
    for k in ("start_mix", "start_archive"):
        if k in c:
            base[k] = c[k]
    base.update(kw)
    return PPOConfig.continuing(cfg_game(), **base)


def shared_config(cond: str):
    from ....rl import SharedConfig
    return SharedConfig(current_fraction=CONDITIONS[cond]["current_fraction"], snapshot_every=SNAP_EVERY, pool_size=POOL)


# ---- agents by specification (picklable strings for worker processes)
def agent_from_spec(spec: str, cfg: Optional[GameConfig] = None):
    """'final:<cond>:<seed>', 'snap:<cond>:<seed>:<transitions>', 'initial:<seed>', 'attacker:<cond>:<seed>',
    'ref:<name>' or a checkpoint path."""
    from ....rl import load_agent
    cfg = cfg if cfg is not None else cfg_game()
    parts = spec.split(":")
    kind = parts[0]
    if kind == "final":
        return load_agent(final_path(parts[1], int(parts[2])), name=spec)
    if kind == "snap":
        path_ = dict(snapshot_paths(parts[1], int(parts[2])))[int(parts[3])]
        return load_agent(path_, name=spec)
    if kind == "attacker":
        return load_agent(os.path.join(attacker_dir(parts[1], int(parts[2])), "final.pt"), name=spec)
    if kind == "initial":
        import torch
        from ....env import LaneEnv
        from ....rl import NetAgent, PolicyValueNet
        torch.manual_seed(int(parts[1]))                      # exactly the untrained network of that training seed
        env = LaneEnv(cfg)
        return NetAgent(PolicyValueNet(env.obs_size, env.n_actions, (128, 128)), spec)
    if kind == "ref":
        from .references import reference_suite
        return reference_suite(cfg)[parts[1]]
    if spec.endswith(".pt"):
        return load_agent(spec)
    raise ValueError(f"unknown agent specification {spec!r}")


# ---- statistics
def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and the 1.96 x standard-error half-width over independent values (seeds: n = 3, indicative only)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan")), float("nan")
    return float(v.mean()), float(Z95 * v.std(ddof=1) / np.sqrt(len(v)))


def fmt_ci(m: float, h: float, d: int = 2) -> str:
    if not np.isfinite(h):
        return f"{m:+.{d}f}"
    return f"{m:+.{d}f} ± {h:.{d}f}"


# ---- I/O
def path(name: str, ext: str = "json") -> str:
    return os.path.join(OUT, f"{name}.{ext}")


def save_json(name: str, data: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    dump_json(data, path(name))
    print(f"wrote {path(name)}", flush=True)


def read_json(p: str) -> dict:
    """A JSON file, plain or gzipped (by extension; `manifest.compress_results` gzips the large results)."""
    if p.endswith(".gz"):
        import gzip
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load(name: str) -> Optional[dict]:
    """results/rl/m43b/<name>.json, or its gzipped form when only that exists."""
    p = path(name)
    if os.path.exists(p):
        return read_json(p)
    if os.path.exists(p + ".gz"):
        return read_json(p + ".gz")
    return None


def write_md(name: str, text: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(path(name, "md"), "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    print(f"wrote {path(name, 'md')}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:                                      # pragma: no cover - outside a checkout
        return "unknown"


def meta(**kw) -> dict:
    import sys
    import torch
    return {"commit": git_commit(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
            "torch": torch.__version__, "numpy": np.__version__, "config": cfg_game().to_dict(), **kw}


def run_jobs(fn: Callable, jobs: Sequence, workers: int, label: str = "", progress: Optional[Callable] = None) -> list:
    """Run `fn(job)` over `jobs` in worker processes (as_completed order); returns the results in job order."""
    if not jobs:
        return []
    out = [None] * len(jobs)
    t0 = time.perf_counter()
    if workers <= 1 or len(jobs) == 1:
        for i, j in enumerate(jobs):
            out[i] = fn(j)
            if progress:
                progress(i, out[i])
        return out
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        futs = {ex.submit(fn, j): i for i, j in enumerate(jobs)}
        done = 0
        for fu in as_completed(futs):
            i = futs[fu]
            out[i] = fu.result()
            done += 1
            if progress:
                progress(i, out[i])
            elif label:
                print(f"  [{label}] {done}/{len(jobs)} done, {time.perf_counter() - t0:.0f} s", flush=True)
    return out


__all__ = ["NAME", "OUT", "CKPT", "BUDGET", "N_ENVS", "SNAP_EVERY", "POOL", "CURVE_EVERY", "SEEDS", "CONDITIONS",
           "PILOT", "PRIMARY", "WINDOW_DEV", "WINDOW_HEAD", "cfg_game", "run_name", "run_dir", "final_path", "attacker_dir",
           "snapshot_paths", "curve_points", "history", "ppo_config", "shared_config", "agent_from_spec", "mean_ci",
           "fmt_ci", "save_json", "load", "read_json", "write_md", "git_commit", "meta", "run_jobs", "md_table", "Z95"]
