"""Shared definitions of M4.6c: the C1-versus-baseline shared-policy PPO pilot (docs/m4/m4_6/m4_6c_preregistration.md).

Arms. "baseline" is `GameConfig.m45_calibrated()` itself; "C1" is `m45_calibrated(invest_gain=3.24, invest_scale=64.0)`,
the M4.6b candidate (`m46b.curves.ARMS["C1"]`). Nothing else differs between the arms: the reward (the zero-sum
discounted earned-gold lead), every rule, the observation and the action space are the same.

Policies (specification strings, `policy_path`):
    start:<s>              the M4.5c 16.384M final of seed s (`final:ext:<s>`), the starting point of both arms
    <arm>:<s>              the final of the arm's continuation of start:<s> (arm = baseline | C1)
    <arm>:<s>@<n>          a snapshot of that run at n agent transitions
    spec:<path>            any checkpoint (M4.6a specialists etc.)

Networks are loaded by `net_of` from the checkpoint file with its digest verified. A network trained under one arm can
play the other arm's environment; every such use is labelled CROSS-ENVIRONMENT in tables (the M4.5c finals under C1
are out of distribution in exactly the M4.6b sense).

Seeds: the range 98,000-98,999 (disjoint from every earlier study; M4.6b used 97,000-97,999). Raw jobs, archives and
checkpoints under results/rl/checkpoints/m46c/ (git-ignored); summaries and reports under results/m46c/.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, Optional, Sequence

import numpy as np

from ..m46a.common import (fmt, holm, load_gz, mean_ci, run_jobs, save_gz, significant, state_from_list,  # noqa: F401
                           state_to_list, t_pvalue)
from ..m46a.common import load_json as _load_json
from ..m46a.common import save_json as _save_json
from ..m46a.common import write_md as _write_md

NAME = "m46c"
OUT = os.path.join("results", NAME)
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
PREREG_DOC = os.path.join("docs", "m4", "m4_6", "m4_6c_preregistration.md")
M45C_EXT = os.path.join("results", "rl", "checkpoints", "m45c", "ext")
M46A_E4 = os.path.join("results", "rl", "checkpoints", "m46a", "e4")

ARMS = ("baseline", "C1")
SEEDS = (0, 1, 2, 3, 4, 5)
CACHE = 3000                    # exact lane memo of evaluation workers (results do not depend on it)

# ---------------------------------------------------------------------------------------------- seeds
TRAIN_SEED = 98_000             # PPO seed of the continuation of final s: TRAIN_SEED + s (the same in both arms)
ARCH_ROOT = 98_100              # natural games of the training archive
EVAL_STATE_ROOT = 98_200        # held-out natural evaluation states
ROLL_ROOT = 98_300              # paired rollouts
NAT_ROOT = 98_400               # natural self-play games
H2H_ROOT = 98_500               # head-to-head evaluations from s0
CURVE_ROOT = 98_600             # learning curves
TRACE_ROOT = 98_700             # representative trajectories
ARCH_CON_ROOT = 98_800          # constructed archive draws

# ---------------------------------------------------------------------------------------------- training settings
BUDGET = 4_096_000              # agent transitions per run (= the learning-rate schedule horizon)
MIDPOINT = 2_048_000
EXT_BUDGET = 8_192_000          # the pre-registered matched extension (phase "ext"), fired 2026-10-15
N_ENVS = 16
SEGMENT = 128
TRANSITIONS_PER_UPDATE = N_ENVS * SEGMENT * 2
ENT_COEF = 0.03
HIDDEN = (128, 128)
LANE_CACHE = 5000               # the M4.5c campaign's exact memo size
GAME_LENGTH = 512               # s0 games (M4.5c)
ARCHIVE_LENGTH = 150            # archive games (M4.6a E4)
ARCHIVE_TARGET_SHARE = 0.25     # share of agent transitions in archive-started games (design)
# per-game probability of an archive start giving that transition share with games of 512 and 150 turns
ARCHIVE_BETA = round(ARCHIVE_TARGET_SHARE * GAME_LENGTH /
                     (ARCHIVE_TARGET_SHARE * GAME_LENGTH + (1 - ARCHIVE_TARGET_SHARE) * ARCHIVE_LENGTH), 3)
MIRROR_P = 0.5
SNAP_EVERY = 25_600
POOL = 8
CURRENT_FRACTION = 1.0

# pre-registered stop-and-investigate thresholds (M4.5c pilot report §8.4, unchanged)
KL_SUSTAINED = 0.05
GRAD_ALARM = 100.0
ENTROPY_FLOOR = 0.3


# ---------------------------------------------------------------------------------------------- arms
_CFGS: Dict[str, object] = {}


def arm_overrides(arm: str) -> dict:
    from ..m46b.curves import ARMS as M46B_ARMS
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    return dict(M46B_ARMS[arm]["overrides"])


def arm_cfg(arm: str = "baseline"):
    """The validated GameConfig of an arm (cached per process)."""
    if arm not in _CFGS:
        from ....config import GameConfig
        cfg = GameConfig.m45_calibrated(**arm_overrides(arm))
        cfg.validate()
        _CFGS[arm] = cfg
    return _CFGS[arm]


def setup(arm: str = "baseline", cache: int = CACHE):
    """One torch thread and a bounded exact lane memo for the arm's configuration; returns the configuration."""
    import torch
    torch.set_num_threads(1)
    from ...m45b import engine as E
    cfg = arm_cfg(arm)
    c = E._CACHES.get(cfg)
    if c is None or c.size != cache:
        if len(E._CACHES) > 4:
            E._CACHES.clear()
        E._CACHES[cfg] = E.LaneCache(cfg, cache)
    return cfg


# ---------------------------------------------------------------------------------------------- policies
def run_dir(arm: str, seed: int, phase: str = "main") -> str:
    return os.path.join(CKPT, "train", phase, arm, f"s{seed}")


def start_path(seed: int) -> str:
    return os.path.join(M45C_EXT, f"s{seed}", "final.pt")


def specialist_path(kind: str, target: int) -> str:
    return os.path.join(M46A_E4, f"{kind}_s{target}", "final.pt")


def snapshot_path(arm: str, seed: int, n: int, phase: str = "main") -> str:
    return os.path.join(run_dir(arm, seed, phase), "snapshots", f"snap_{n:09d}.pt")


def policy_path(spec: str) -> str:
    """Checkpoint path of a policy specification (module docstring)."""
    if spec.startswith("spec:"):
        return spec[5:]
    kind, rest = spec.split(":", 1)
    if kind == "start":
        return start_path(int(rest))
    phase = "main"
    if "/" in kind:
        phase, kind = kind.split("/", 1)
    if kind not in ARMS:
        raise ValueError(f"unknown policy specification {spec!r}")
    if "@" in rest:
        s, n = rest.split("@")
        return snapshot_path(kind, int(s), int(n), phase)
    return os.path.join(run_dir(kind, int(rest), phase), "final.pt")


def trained_on(spec: str) -> str:
    """The arm whose environment a policy was trained on (start and specialists: baseline)."""
    if spec.startswith(("start:", "spec:")):
        return "baseline"
    kind = spec.split(":", 1)[0].split("/")[-1]
    return kind


_NETS: Dict[str, object] = {}


def net_of(spec_or_path: str):
    """The PolicyValueNet of a policy specification or checkpoint path (digest verified, cached per process).
    Environment compatibility is the caller's responsibility (`trained_on` labels cross-environment use); shapes are
    always checked against the 90-input / 61-output M4.5 encoding."""
    if spec_or_path not in _NETS:
        from ....rl import load_checkpoint
        path = spec_or_path if spec_or_path.endswith(".pt") else policy_path(spec_or_path)
        net, _, _ = load_checkpoint(path)
        if net.obs_size != 90 or net.n_actions != 61:
            raise ValueError(f"{path}: {net.obs_size} inputs / {net.n_actions} outputs, not the M4.5 encoding")
        net.eval()
        _NETS[spec_or_path] = net
    return _NETS[spec_or_path]


def install_nets(specs: Sequence[str]) -> None:
    """Make the M4.6a paired engine resolve these specifications/paths through `net_of` (its own cache)."""
    from ..m46a import common as A
    for s in specs:
        if s not in A._NETS:
            A._NETS[s] = net_of(s)


# ---------------------------------------------------------------------------------------------- statistics
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 14: 2.145, 15: 2.131, 17: 2.110, 20: 2.086, 23: 2.069, 29: 2.045, 30: 2.042,
       35: 2.030, 47: 2.012, 60: 2.000}


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T95:
        return T95[df]
    keys = sorted(T95)
    if df > keys[-1]:
        return 1.96
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    return T95[lo] + (T95[hi] - T95[lo]) * (df - lo) / (hi - lo)


def seed_stats(per_seed: Dict[int, float]) -> dict:
    """Mean over seeds with a t interval (df = n - 1), the per-seed values and the sign count."""
    v = np.asarray([x for x in per_seed.values() if np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "ci": float("nan"), "n": 0, "per_seed": per_seed, "pos": 0, "neg": 0}
    m = float(v.mean())
    h = float(t95(n - 1) * v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": m, "ci": h, "n": n, "per_seed": {str(k): float(x) for k, x in per_seed.items()},
            "pos": int((v > 0).sum()), "neg": int((v < 0).sum()), "sd": float(v.std(ddof=1)) if n > 1 else float("nan")}


def excludes0(st: dict) -> int:
    """+1 / -1 if the interval excludes 0 on that side, else 0."""
    m, h = st.get("mean", np.nan), st.get("ci", np.nan)
    if not (np.isfinite(m) and np.isfinite(h)):
        return 0
    return 1 if m - h > 0 else (-1 if m + h < 0 else 0)


def fci(st: dict, d: int = 2) -> str:
    m, h = st.get("mean", np.nan), st.get("ci", np.nan)
    if not np.isfinite(m):
        return "n/a"
    return f"{m:+.{d}f} ± {h:.{d}f}" if np.isfinite(h) else f"{m:+.{d}f}"


# ---------------------------------------------------------------------------------------------- I/O and provenance
def git_commit() -> str:
    from ..m45c.common import git_commit as g
    return g()


def meta(**kw) -> dict:
    from ..m45c.common import git_dirty
    try:
        import torch
        tv = torch.__version__
    except Exception:                                      # pragma: no cover
        tv = None
    return {"study": "M4.6c", "commit": git_commit(), "working_tree_dirty": git_dirty(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": tv, "numpy": np.__version__, **kw}


def save_json(name: str, data: dict, out: str = OUT) -> str:
    return _save_json(name, data, out)


def load(name: str, out: str = OUT):
    return _load_json(name, out)


def write_md(name: str, text: str, out: str = OUT) -> str:
    return _write_md(name, text, out)


def prereg_record() -> dict:
    with open(PREREG_DOC, "rb") as f:
        b = f.read()
    return {"document": PREREG_DOC, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def write_prereg() -> dict:
    """Freeze the pre-registration (its SHA-256 and the time) before any training or comparative stage."""
    path = os.path.join(OUT, "preregistration.json")
    if os.path.exists(path):
        old = load("preregistration")
        if old["record"]["sha256"] != prereg_record()["sha256"]:
            raise SystemExit("the M4.6c pre-registration changed after it was frozen; record an amendment instead")
        return old
    d = {"meta": meta(stage="prereg"), "record": prereg_record(),
         "note": "frozen after the smoke test (pipeline only) and before any training run of the pilot"}
    save_json("preregistration", d)
    return d


def require_prereg() -> None:
    path = os.path.join(OUT, "preregistration.json")
    if not os.path.exists(path):
        raise SystemExit("freeze the M4.6c pre-registration first (stage prereg)")
    old = load("preregistration")
    if old["record"]["sha256"] != prereg_record()["sha256"]:
        raise SystemExit("the M4.6c pre-registration document differs from its frozen SHA-256")


def amend(note: str) -> dict:
    d = load("preregistration") or {}
    d.setdefault("amendments", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note})
    save_json("preregistration", d)
    return d


def jdump(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


__all__ = ["NAME", "OUT", "CKPT", "ARMS", "SEEDS", "arm_cfg", "arm_overrides", "setup", "run_dir", "start_path",
           "specialist_path", "snapshot_path", "policy_path", "trained_on", "net_of", "install_nets", "seed_stats",
           "excludes0", "fci", "t95", "meta", "save_json", "load", "write_md", "write_prereg", "require_prereg",
           "amend", "run_jobs", "mean_ci", "load_gz", "save_gz", "state_to_list", "state_from_list", "BUDGET",
           "MIDPOINT", "ARCHIVE_BETA", "ARCHIVE_LENGTH", "GAME_LENGTH"]
