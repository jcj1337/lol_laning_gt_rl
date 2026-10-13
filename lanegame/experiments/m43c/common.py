"""Shared definitions of the M4.3c design diagnostics (docs/m4/m4_3c_economic_objective_design.md).

Nothing here changes the game. The diagnostics evaluate the M4.3b finals and the reactive references on the unchanged
research preset `GameConfig.m4_lasthit()` and score the SAME paired trajectories under alternative economic objectives.

Objective channels. Every alternative objective of the design document is a linear combination of three per-turn,
zero-sum channels, all read from `StepResult.info` of one turn and stated from seat s's perspective:

    g_t   (earned_s - earned_o) / gold_scale          gold: farm + plate gold + kill bounty (the CURRENT reward, exactly)
    x_t   xp_gain_s - xp_gain_o                        XP credited this turn (after the level-cap clamp)
    l_t   Lambda(xp_s') - Lambda(xp_s) - (the same for o)    level progress, Lambda(x) = L(x) + (x - X(L)) / n(L)

so that, for a discounted window, J(objective) = G + lambda X (constant XP value) or G + mu Lam (level-progress value).
Storing the three discounted sums per rollout scores every lambda and mu on identical randomness.

Seeds: SeedSequence roots 71,000-74,999, disjoint from every earlier study (10,000-69,999 and 77,777). The re-scoring
of the M4.3b wave-only counterfactuals reuses that study's own roots (67,500-67,699) so that its gold numbers are
reproduced exactly.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ...config import GameConfig
from ...io import dump_json
from ...progression import Tables, tables

NAME = "m43c"
OUT = os.path.join("results", NAME)
Z95 = 1.96
PROGRESSION_SEED, XP_CF_SEED, FREEZE_SEED, DEATH_SEED = 71_000, 72_000, 73_000, 74_000

# lambda: resources per credited XP; mu: resources per level of level progress (see the design document §B)
LAMBDAS = (0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4)
MUS = (0.0, 0.5, 1.0, 2.0)


def cfg_game() -> GameConfig:
    """The research preset of M4.3b, unchanged."""
    return GameConfig.m4_lasthit()


# ---------------------------------------------------------------------------------------------- objective channels
def level_progress(xp: int, t: Tables) -> float:
    """Lambda(x) = L(x) + (x - X(L)) / n(L): the level as a continuous quantity (= level_cap at the cap)."""
    x = min(int(xp), t.xp_cap)
    L = t.level_by_xp[x]
    if L >= t.level_cap:
        return float(t.level_cap)
    return L + (x - t.xp_cum[L]) / t.xp_next[L]


def channels(info: dict, s: int, t: Tables, scale: int) -> Tuple[float, float, float]:
    """(g, x, l) of one turn from seat s's perspective (module docstring). `info` is a ruleset-m4 StepResult.info."""
    o = 1 - s
    eu = info["earned_units"]
    g = (eu[s] - eu[o]) / scale
    xg = info["xp_gain"]
    x = float(xg[s] - xg[o])
    before = info["champ_before"]
    after = info["champ_after"]
    l = (level_progress(after[s][3], t) - level_progress(before[s][3], t)
         - level_progress(after[o][3], t) + level_progress(before[o][3], t))
    return g, x, l


def objective_value(G: float, X: float, Lam: float, lam: float = 0.0, mu: float = 0.0) -> float:
    """J = G + lambda X + mu Lam for discounted channel sums (G, X, Lam)."""
    return G + lam * X + mu * Lam


# ---------------------------------------------------------------------------------------------- statistics, I/O
def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan")), float("nan")
    return float(v.mean()), float(Z95 * v.std(ddof=1) / np.sqrt(len(v)))


def quantiles(values: Sequence[float], qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> Dict[str, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return {f"q{int(100 * q)}": float("nan") for q in qs}
    return {f"q{int(100 * q)}": float(np.quantile(v, q)) for q in qs}


def path(name: str, ext: str = "json") -> str:
    return os.path.join(OUT, f"{name}.{ext}")


GZIP_BYTES = 1_000_000            # the M4.3b artifact rule: result files of 1 MB or more are stored gzipped


def save_json(name: str, data: dict) -> None:
    """results/m43c/<name>.json, gzipped to <name>.json.gz (and the plain file removed) when it is >= 1 MB."""
    import gzip
    import shutil
    os.makedirs(OUT, exist_ok=True)
    p = path(name)
    dump_json(data, p)
    if os.path.getsize(p) >= GZIP_BYTES:
        with open(p, "rb") as src, gzip.open(p + ".gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(p)
        p += ".gz"
    elif os.path.exists(p + ".gz"):
        os.remove(p + ".gz")
    print(f"wrote {p}", flush=True)


def load(name: str) -> Optional[dict]:
    """results/m43c/<name>.json or its gzipped form."""
    import gzip
    import json
    p = path(name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(p + ".gz"):
        with gzip.open(p + ".gz", "rt", encoding="utf-8") as f:
            return json.load(f)
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
    return {"commit": git_commit(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
            "numpy": np.__version__, "preset": "GameConfig.m4_lasthit()", **kw}


def ctx_tables(cfg: GameConfig) -> Tables:
    return tables(cfg)


__all__ = ["NAME", "OUT", "LAMBDAS", "MUS", "cfg_game", "level_progress", "channels", "objective_value", "mean_ci",
           "quantiles", "save_json", "load", "write_md", "meta", "PROGRESSION_SEED", "XP_CF_SEED", "FREEZE_SEED",
           "DEATH_SEED"]
