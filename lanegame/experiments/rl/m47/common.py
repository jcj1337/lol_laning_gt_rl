"""Shared definitions of M4.7, the early-lane horizon study (docs/m4/m4_7/m4_7_preregistration.md for Stage 2).

Environments. Every game is played under the mechanics of `GameConfig.m45_calibrated()` ("baseline") or of the M4.6c
C1 curve ("C1" = `m45_calibrated(invest_gain=3.24, invest_scale=64.0)`, provisional, not adopted). The discount gamma is
part of the game's OBJECTIVE, not of its mechanics: Stage 2's treatment is `m45_calibrated(gamma=SHORT_GAMMA)`, a
keyword override of the existing field, whose play is identical to the baseline's and whose score is different. No
preset, default or rule changes.

Policies (specification strings, `policy_path`):
    ext/<env>:<s>      the M4.6c 8.192M continuation of seed s in environment env (baseline | C1): the matched pair of
                       Stage 1 and the common starting point of Stage 2 (ext/baseline:<s>)
    start:<s>          the M4.5c 16.384M final of seed s (the M4.6c starting point; lineage only)
    p/<arm>:<s>        the Stage 2 continuation of ext/baseline:<s> in arm control | short (final.pt)
    p/<arm>:<s>@<n>    its snapshot at n agent transitions
    spec:<path>        any checkpoint

Scores. Rollouts store per-turn reward streams (seat A's earned-gold lead per turn, resources), so any discount and any
window is computed after the fact from the SAME trajectories: HORIZONS gives the candidate discounts by their expected
duration H = 1 / (1 - gamma). No score in this study is bootstrapped with a critic.

Seeds: the range 58,000-58,999 (disjoint from every earlier study). Raw files under results/rl/checkpoints/m47/
(git-ignored); summaries and reports under results/m47/.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Dict, Optional

import numpy as np

from ..m46a.common import load_gz, save_gz, state_from_list, state_to_list  # noqa: F401
from ..m46a.common import load_json as _load_json
from ..m46a.common import save_json as _save_json
from ..m46a.common import write_md as _write_md
from ..m46c import common as C46

NAME = "m47"
OUT = os.path.join("results", NAME)
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
PREREG_DOC = os.path.join("docs", "m4", "m4_7", "m4_7_preregistration.md")
ENVS = ("baseline", "C1")
SEEDS = (0, 1, 2, 3, 4, 5)
GAMMA = 0.99
REWARD_SCALE = 0.1

# ---------------------------------------------------------------------------------------------- horizons
# expected duration H of the independent-hazard reading (gamma = 1 - 1/H); "100" is the current objective
HORIZONS: Dict[str, float] = {"100": 0.99, "80": 1 - 1 / 80, "60": 1 - 1 / 60, "50": 1 - 1 / 50, "40": 1 - 1 / 40}
LEVEL_MARKS = (6, 8, 10, 12, 14, 16, 18)
FOCUS_LEVELS = (10, 12, 14)
# common elapsed-turn windows (turn index, half open); "lane" = turns 0-93, the natural median before level 14
WINDOWS = {"0-25": (0, 25), "25-50": (25, 50), "50-75": (50, 75), "75-94": (75, 94), "94-150": (94, 150),
           "150-250": (150, 250), "250-400": (250, 400)}
LANE_END = 94                   # median first turn at level 14 in natural self-play (Stage 1, both environments)

# ---------------------------------------------------------------------------------------------- seeds
DELAY_CELL_ROOT = 58_000        # Stage 1: constructed lead cells, paired long rollouts
DELAY_NAT_ROOT = 58_100         # Stage 1: natural early states (self-play from s0) and their paired rollouts
DELAY_FRZ_ROOT = 58_200         # Stage 1: early freeze cells (M4.6d builder), paired long rollouts
TRAIN_SEED = 58_300             # Stage 2: PPO seed of the continuation of ext/baseline:s = TRAIN_SEED + s (both arms)
H2H_ROOT = 58_400               # Stage 2: head to head from s0
NAT_ROOT = 58_500               # Stage 2: natural self-play records
CELL_ROOT = 58_600              # Stage 2: paired cells (leads, freezes, natural early states)
FIXED_ROOT = 58_700             # Stage 2: games against fixed opponents
TRACE_ROOT = 58_800             # representative trajectories

# ---------------------------------------------------------------------------------------------- Stage 2 settings
ARMS = ("control", "short")
PILOT_SEEDS = (0, 1)
SHORT_H = 60                    # the treatment's expected duration (chosen from Stage 1; report §3)
SHORT_GAMMA = 1 - 1 / SHORT_H
BUDGET = 4_096_000              # agent transitions per run (= the learning-rate schedule horizon)
MIDPOINT = 2_048_000
SNAP_EVERY = C46.SNAP_EVERY


def arm_gamma(arm: str) -> float:
    if arm == "control":
        return GAMMA
    if arm == "short":
        return SHORT_GAMMA
    raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")


_CFGS: Dict[tuple, object] = {}


def env_cfg(env: str = "baseline", gamma: Optional[float] = None):
    """The validated configuration of a mechanics environment with an optional gamma override (cached per process)."""
    key = (env, gamma)
    if key not in _CFGS:
        from ....config import GameConfig
        kw = dict(C46.arm_overrides(env))
        if gamma is not None and gamma != GAMMA:
            kw["gamma"] = float(gamma)
        cfg = GameConfig.m45_calibrated(**kw)
        cfg.validate()
        _CFGS[key] = cfg
    return _CFGS[key]


def arm_cfg(arm: str):
    """Stage 2: the training configuration of an arm (baseline mechanics; the arm's gamma)."""
    return env_cfg("baseline", arm_gamma(arm))


def setup(env: str = "baseline", cache: int = C46.CACHE):
    """One torch thread and a bounded exact lane memo for an environment; returns its gamma-0.99 configuration (play
    does not depend on gamma, so every evaluation plays under it and scores the streams afterwards)."""
    import torch
    torch.set_num_threads(1)
    from ...m45b import engine as E
    cfg = env_cfg(env)
    c = E._CACHES.get(cfg)
    if c is None or c.size != cache:
        if len(E._CACHES) > 4:
            E._CACHES.clear()
        E._CACHES[cfg] = E.LaneCache(cfg, cache)
    return cfg


# ---------------------------------------------------------------------------------------------- policies
def run_dir(arm: str, seed: int, phase: str = "main") -> str:
    return os.path.join(CKPT, "train", phase, arm, f"s{seed}")


def policy_path(spec: str) -> str:
    """Checkpoint path of a policy specification (module docstring)."""
    if spec.startswith("p/"):
        arm, rest = spec[2:].split(":", 1)
        if arm not in ARMS:
            raise ValueError(f"unknown pilot arm in {spec!r}")
        if "@" in rest:
            s, n = rest.split("@")
            return os.path.join(run_dir(arm, int(s)), "snapshots", f"snap_{int(n):09d}.pt")
        return os.path.join(run_dir(arm, int(rest)), "final.pt")
    return C46.policy_path(spec)


_NETS: Dict[str, object] = {}


def net_of(spec_or_path: str):
    """The network of a specification or path (digest verified, cached per process; the 90 / 61 M4.5 encoding)."""
    if spec_or_path not in _NETS:
        path = spec_or_path if spec_or_path.endswith(".pt") else policy_path(spec_or_path)
        _NETS[spec_or_path] = C46.net_of(path)
    return _NETS[spec_or_path]


def general(env: str, s: int) -> str:
    """The matched M4.6c 8.192M policy of seed s in an environment."""
    return f"ext/{env}:{s}"


# ---------------------------------------------------------------------------------------------- discounting
def disc(gamma: float, T: int) -> np.ndarray:
    return gamma ** np.arange(T, dtype=float)


def weight_before(t: float, gamma: float) -> float:
    """Share of the discounted objective's weight on turns before t (from turn 0): 1 - gamma^t."""
    return float(1.0 - gamma ** t)


def mean_weight_before_end(T: int, gamma: float) -> float:
    """Average, over decision turns 0..T-1 drawn uniformly, of the share of each decision's objective weight that
    falls before turn T (the end of the phase)."""
    k = np.arange(T, dtype=float)
    return float(np.mean(1.0 - gamma ** (T - k)))


# ---------------------------------------------------------------------------------------------- statistics
t95 = C46.t95
seed_stats = C46.seed_stats
fci = C46.fci


def mean_se(x) -> dict:
    """Mean, standard error and 95% normal half-width of i.i.d. values (games within one seed)."""
    v = np.asarray([y for y in np.ravel(x) if np.isfinite(y)], dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "ci": float("nan"), "n": 0}
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": float(v.mean()), "se": se, "ci": 1.96 * se if n > 1 else float("nan"), "n": n}


# ---------------------------------------------------------------------------------------------- I/O and provenance
def meta(**kw) -> dict:
    m = C46.meta(**kw)
    m["study"] = "M4.7"
    return m


def save_json(name: str, data: dict, out: str = OUT) -> str:
    return _save_json(name, data, out)


def load(name: str, out: str = OUT):
    return _load_json(name, out)


def write_md(name: str, text: str, out: str = OUT) -> str:
    return _write_md(name, text, out)


def run_jobs(fn, jobs, workers: int, label: str = ""):
    return C46.run_jobs(fn, jobs, workers, label)


def prereg_record() -> dict:
    with open(PREREG_DOC, "rb") as f:
        b = f.read()
    return {"document": PREREG_DOC, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def write_prereg() -> dict:
    """Freeze the Stage 2 pre-registration (SHA-256 and time) before any pilot training run."""
    path = os.path.join(OUT, "preregistration.json")
    if os.path.exists(path):
        old = load("preregistration")
        if old["record"]["sha256"] != prereg_record()["sha256"]:
            raise SystemExit("the M4.7 pre-registration changed after it was frozen; record an amendment instead")
        return old
    d = {"meta": meta(stage="prereg"), "record": prereg_record(),
         "note": "frozen after Stage 1 and the pilot smoke test, before any pilot training run"}
    save_json("preregistration", d)
    return d


def require_prereg() -> None:
    path = os.path.join(OUT, "preregistration.json")
    if not os.path.exists(path):
        raise SystemExit("freeze the M4.7 pre-registration first (stage prereg)")
    if load("preregistration")["record"]["sha256"] != prereg_record()["sha256"]:
        raise SystemExit("the M4.7 pre-registration document differs from its frozen SHA-256")


def amend(note: str) -> dict:
    d = load("preregistration") or {}
    d.setdefault("amendments", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note})
    save_json("preregistration", d)
    return d


__all__ = ["NAME", "OUT", "CKPT", "ENVS", "SEEDS", "GAMMA", "HORIZONS", "WINDOWS", "LANE_END", "ARMS", "PILOT_SEEDS",
           "SHORT_H", "SHORT_GAMMA", "BUDGET", "env_cfg", "arm_cfg", "arm_gamma", "setup", "run_dir", "policy_path",
           "net_of", "general", "disc", "weight_before", "mean_weight_before_end", "seed_stats", "mean_se", "fci",
           "meta", "save_json", "load", "write_md", "run_jobs", "write_prereg", "require_prereg", "amend"]
