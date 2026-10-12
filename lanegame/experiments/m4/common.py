"""Shared helpers of the M4 calibration and validation drivers (plan WP10): results/m4/, JSON before Markdown,
run metadata (commit, command, configuration, seeds), a process pool, and paired statistics."""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
import time
from typing import Callable, Iterable, List, Sequence

import numpy as np

from ...io import dump_json
from ..common import RESULTS_DIR, environment, md_table

OUT = os.path.join(RESULTS_DIR, "m4")
Z95 = 1.96


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(RESULTS_DIR),
                                       text=True).strip()
    except Exception:                                   # pragma: no cover - outside a checkout
        return "unknown"


def meta(cfg=None, **kw) -> dict:
    d = {"commit": git_commit(), "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
         "environment": environment()}
    if cfg is not None:
        d["config"] = cfg.to_dict()
    d.update(kw)
    return d


def save(name: str, data: dict, markdown: str) -> str:
    """JSON first (raw measurements), then Markdown; returns the JSON path."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".json")
    dump_json(data, path)
    with open(os.path.join(OUT, name + ".md"), "w", encoding="utf-8") as f:
        f.write(markdown.rstrip() + "\n")
    return path


def pool_map(fn: Callable, items: Sequence, procs: int = 0) -> List:
    """Map in worker processes (spawn context; deterministic: every item carries its own seeds)."""
    procs = procs or max(1, (os.cpu_count() or 2) - 1)
    if procs == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with mp.get_context("spawn").Pool(procs) as pool:
        return pool.map(fn, items, chunksize=1)


def mean_se(x) -> tuple:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def fmt_ci(m: float, se: float, digits: int = 3) -> str:
    return f"{m:+.{digits}f} ± {Z95 * se:.{digits}f}"


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.t0


__all__ = ["OUT", "Z95", "git_commit", "meta", "save", "pool_map", "mean_se", "fmt_ci", "Timer", "md_table"]
