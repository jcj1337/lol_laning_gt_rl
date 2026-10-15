"""Shared definitions of M4.5b: seeds, configurations, statistics, I/O and the job runner.

Seeds (integer chance seeds, so that every game can be replayed by the viewer from its Setup):

    DEV       87,000-87,499   every calibration and diagnostic experiment; one list shared by all cells of a
                              comparison (common random numbers)
    HOLDOUT   88,000-88,499   the final evaluation of the selected candidate only (`validate`); fixed before any
                              calibration run (`states.HOLDOUT_SPLIT`, results/m45b/holdout_split.json)
    TRACE     87,900-87,999   viewer demonstrations

Both ranges are disjoint from every earlier study (10,000-81,999, 85,000-86,999, 90,000-91,999). A game played with
seed n draws its chance vector exactly as `LaneEnv.reset(seed=n)` does (numpy default_rng(n), N_CHANCE uniforms per
turn whatever the actions), so the viewer reproduces it.

Configurations: `cfg(name, **overrides)` with name one of BASES; the M4.5a revision-2 baseline is "r2"
(= `GameConfig.m45_unified()`), the historical comparator "aggro" (= `m4_four_actions_econ_aggro()`, 88-entry
observation, siege Vision, no minion targeting), "r1" (= `m45_unified_r1()`) and the selected candidate "cal"
(= `m45_calibrated()`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...config import GameConfig

OUT = os.path.join("results", "m45b")
TRAJ_DIR = os.path.join(OUT, "trajectories")
DEV = tuple(range(87_000, 87_500))
HOLDOUT = tuple(range(88_000, 88_500))
TRACE = tuple(range(87_900, 88_000))
GAMMA = 0.99
WORKERS = max(1, (os.cpu_count() or 2) - 1)

BASES: Dict[str, Callable[..., GameConfig]] = {
    "cal": GameConfig.m45_calibrated,
    "r2": GameConfig.m45_unified,
    "r1": GameConfig.m45_unified_r1,
    "aggro": GameConfig.m4_four_actions_econ_aggro,
}


def cfg(name: str = "r2", **overrides) -> GameConfig:
    """A validated configuration: a base preset (BASES, or a GameConfig preset method name) with field overrides."""
    if name in BASES:
        c = BASES[name](**overrides)
    else:
        c = getattr(GameConfig, name)(**overrides)
    c.validate()
    return c


def describe_cfg(c: GameConfig, ref: Optional[GameConfig] = None) -> Dict[str, object]:
    """The fields of `c` that differ from `ref` (default: the r2 baseline)."""
    ref = ref if ref is not None else cfg("r2")
    a, b = c.to_dict(), ref.to_dict()
    return {k: a[k] for k in a if a[k] != b.get(k)}


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


def ci(values) -> List[float]:
    """[mean, 95% t half-width, n] (JSON-friendly)."""
    m, h, n = t_ci(values)
    return [m, h, n]


def paired(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """[mean, half-width, n] of the paired differences a - b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return ci(a - b)


def fmt(m: float, h: Optional[float] = None, d: int = 2, sign: bool = True) -> str:
    if m is None or not np.isfinite(m):
        return "n/a"
    s = f"{m:+.{d}f}" if sign else f"{m:.{d}f}"
    if h is None or not np.isfinite(h):
        return s
    return f"{s} ± {h:.{d}f}"


def fci(x, d: int = 2, sign: bool = True) -> str:
    """Format a [mean, half-width, n] triple."""
    return fmt(x[0], x[1], d, sign)


def signif(x) -> str:
    """'+' / '-' when the 95% interval excludes 0, else '0'."""
    m, h = x[0], x[1]
    if not np.isfinite(m) or not np.isfinite(h):
        return "?"
    return "+" if m - h > 0 else "-" if m + h < 0 else "0"


# ---------------------------------------------------------------------------------------------- jobs
def run_jobs(fn: Callable, jobs: Sequence, workers: int = WORKERS, label: str = "") -> list:
    """`fn(job)` over `jobs` in worker processes; results in job order (lanegame.experiments.rl.m43b.common)."""
    from ..rl.m43b.common import run_jobs as _rj
    t0 = time.perf_counter()
    out = _rj(fn, list(jobs), workers, "")
    if label:
        print(f"  [{label}] {len(jobs)} jobs, {time.perf_counter() - t0:.0f} s", flush=True)
    return out


# ---------------------------------------------------------------------------------------------- I/O
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:                                      # pragma: no cover
        return "unknown"


def git_dirty() -> Optional[bool]:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:                                      # pragma: no cover
        return None


def meta(**kw) -> dict:
    return {"study": "M4.5b", "commit": git_commit(), "working_tree_dirty": git_dirty(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
            "numpy": np.__version__, **kw}


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_json(name: str, data: dict) -> str:
    """results/m45b/<name>.json, gzipped when it is >= 1 MB."""
    import gzip
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".json")
    text = json.dumps(data, default=_default, indent=1)
    if len(text) >= 1_000_000:
        with gzip.open(p + ".gz", "wt", encoding="utf-8") as f:
            f.write(text)
        if os.path.exists(p):
            os.remove(p)
        p += ".gz"
    else:
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        if os.path.exists(p + ".gz"):
            os.remove(p + ".gz")
    print(f"wrote {p}", flush=True)
    return p


def load_json(name: str) -> Optional[dict]:
    import gzip
    p = os.path.join(OUT, name + ".json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(p + ".gz"):
        with gzip.open(p + ".gz", "rt", encoding="utf-8") as f:
            return json.load(f)
    return None


def write_md(name: str, text: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    print(f"wrote {p}", flush=True)
    return p


def md_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


__all__ = ["OUT", "TRAJ_DIR", "DEV", "HOLDOUT", "TRACE", "GAMMA", "WORKERS", "BASES", "cfg", "describe_cfg", "t_ci",
           "ci", "paired", "fmt", "fci", "signif", "run_jobs", "meta", "save_json", "load_json", "write_md",
           "md_table"]
