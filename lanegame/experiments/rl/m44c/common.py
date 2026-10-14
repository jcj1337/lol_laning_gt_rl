"""Shared definitions of M4.4c: the evaluated checkpoints, the evaluation seed roots, statistics and I/O.

Checkpoints (unchanged M4.4b files, loaded through the M4.4a compatibility guard with their weights digest verified):

    primary       final:ext:<env>:<seed>    4,096,000 agent transitions (matched 2x extension, its own learning-rate
                                            schedule from initialization: NOT a continuation of the 2.05M run)
    replication   final:main:<env>:<seed>   2,048,000 agent transitions (the pre-registered M4.4b finals)

The two budgets are reported side by side and never pooled; the training seeds (0, 1, 2) are the replicates of the
training procedure, starting states the replicates within one network.

Seeds. Every M4.4c stream derives from SeedSequence roots 81,000-81,999, disjoint from training (torch / numpy seeds
0-3), from M4.4a / M4.4b evaluation (80,000-80,999) and from every earlier study (10,000-79,999).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..m44.common import (ENV_ORDER, SEEDS, agent_from_spec, env_cfg, final_path, git_commit, git_dirty, load_json as
                          _load_json, run_jobs, save_json as _save_json, write_md as _write_md)

OUT = os.path.join("results", "m44c")
TRAJ_DIR = os.path.join(OUT, "trajectories")
PRIMARY, REPLICATION = "ext", "main"
PHASE_LABEL = {"ext": "4.10M", "main": "2.05M"}

# evaluation seed roots (81,000-81,999); one block of 10 per investigation
STATE_SEED = 81_000             # natural state collection (+ investigation offset)
ROLL_SEED = 81_100              # paired rollouts (+ offset)
INIT_SEED = 81_200              # rollouts from initialized states (+ offset)
MECH_SEED = 81_300              # mechanistic enumerations (+ offset)
TRACE_SEED = 81_500             # viewer-seeded games for the demonstration traces (integer seeds 81,500-81,599)
OFFSET = {"recall": 0, "freeze": 10, "zoning": 20, "vision": 30, "waves": 40, "test": 90}
ALL_ROOTS = tuple(r + o for r in (STATE_SEED, ROLL_SEED, INIT_SEED, MECH_SEED) for o in OFFSET.values())


def spec(env: str, seed: int, phase: str = PRIMARY) -> str:
    return f"final:{phase}:{env}:{seed}"


def available(env: str, seed: int, phase: str = PRIMARY) -> bool:
    return os.path.exists(final_path(phase, env, seed))


def policies(envs: Sequence[str] = ENV_ORDER, seeds: Sequence[int] = SEEDS, phases: Sequence[str] = (PRIMARY,)) -> List[Tuple[str, int, str]]:
    """(env, seed, phase) of every existing final, in a fixed order."""
    return [(e, s, p) for p in phases for e in envs for s in seeds if available(e, s, p)]


def load(env: str, seed: int, phase: str = PRIMARY):
    """A final as a frozen NetAgent for its own environment (compatibility guard, digest verified)."""
    import torch
    torch.set_num_threads(1)
    cfg = env_cfg(env)
    return cfg, agent_from_spec(spec(env, seed, phase), cfg)


def weights_digest(agent) -> str:
    """sha256 of the network's current weights (to show that evaluation never changes them)."""
    from ....rl.agent import weights_digest as wd
    return wd(agent.net.state_dict())


# ---------------------------------------------------------------------------------------------- statistics
def t_ci(values: Sequence[float], conf: float = 0.95) -> Tuple[float, float, int]:
    """(mean, t half-width, n) over the finite values (half-width nan for n < 2)."""
    from scipy import stats
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n < 2:
        return float(v.mean()), float("nan"), 1
    h = float(stats.t.ppf(0.5 + conf / 2, n - 1) * v.std(ddof=1) / np.sqrt(n))
    return float(v.mean()), h, n


def fmt(m: float, h: Optional[float] = None, d: int = 2, sign: bool = True) -> str:
    if m is None or not np.isfinite(m):
        return "n/a"
    s = f"{m:+.{d}f}" if sign else f"{m:.{d}f}"
    if h is None or not np.isfinite(h):
        return s
    return f"{s} [{m - h:+.{d}f}, {m + h:+.{d}f}]" if sign else f"{s} [{m - h:.{d}f}, {m + h:.{d}f}]"


def pct(x: float, d: int = 0) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------------------------------------- resumable jobs
PARTIAL = os.path.join(OUT, "partial")


def _job_key(job) -> str:
    import hashlib
    return hashlib.sha256(repr(job).encode("utf-8")).hexdigest()[:16]


def run_resumable(fn, jobs: Sequence, workers: int, name: str) -> list:
    """`run_jobs`, with every finished job's result written at once to results/m44c/partial/<name>/<key>.pkl (key = a
    hash of the job tuple) and reused on a re-run: a crash after the jobs (or a stopped run) never loses finished work.
    The caller removes the folder (`clear_partial`) after the final results file is written."""
    import pickle
    from ..m43b.common import run_jobs as _rj
    d = os.path.join(PARTIAL, name)
    os.makedirs(d, exist_ok=True)
    keys = [_job_key(j) for j in jobs]
    out: List[object] = [None] * len(jobs)
    todo = []
    for i, (j, k) in enumerate(zip(jobs, keys)):
        p = os.path.join(d, k + ".pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                saved_job, res = pickle.load(f)
            if saved_job == j:
                out[i] = res
                continue
        todo.append(i)
    if len(todo) < len(jobs):
        print(f"  [{name}] {len(jobs) - len(todo)} of {len(jobs)} jobs reused from {d}", flush=True)
    t0 = time.perf_counter()
    done = [0]

    def progress(ii, res):
        i = todo[ii]
        tmp = os.path.join(d, keys[i] + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump((jobs[i], res), f)
        os.replace(tmp, os.path.join(d, keys[i] + ".pkl"))
        out[i] = res
        done[0] += 1
        print(f"  [{name}] {done[0]}/{len(todo)} done, {time.perf_counter() - t0:.0f} s", flush=True)
    _rj(fn, [jobs[i] for i in todo], workers, name, progress=progress)
    return out


def clear_partial(name: str) -> None:
    import shutil
    shutil.rmtree(os.path.join(PARTIAL, name), ignore_errors=True)


# ---------------------------------------------------------------------------------------------- I/O
def meta(**kw) -> dict:
    import sys
    import torch
    return {"commit": git_commit(), "working_tree_dirty": git_dirty(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "study": "M4.4c", **kw}


def save_json(name: str, data: dict) -> str:
    return _save_json(name, data, OUT)


def load_json(name: str) -> Optional[dict]:
    return _load_json(name, OUT)


def write_md(name: str, text: str) -> str:
    return _write_md(name, text, OUT)


__all__ = ["OUT", "TRAJ_DIR", "PRIMARY", "REPLICATION", "PHASE_LABEL", "STATE_SEED", "ROLL_SEED", "INIT_SEED",
           "MECH_SEED", "TRACE_SEED", "OFFSET", "ALL_ROOTS", "spec", "available", "policies", "load", "weights_digest",
           "t_ci", "fmt", "pct", "meta", "save_json", "load_json", "write_md", "run_jobs", "ENV_ORDER", "SEEDS",
           "env_cfg", "run_resumable", "clear_partial", "PARTIAL"]
