"""Shared definitions of M4.6f (docs/m4/m4_6/m4_6f_preregistration.md).

Environment: the M4.6c arm "C1" = `m45_calibrated(invest_gain=3.24, invest_scale=64.0)` (provisional, NOT adopted), the
environment of the M4.6d / M4.6e diagnostics. It is the study's environment because the diagnosed critics live there;
nothing here changes it or adopts it.

Arms (the only difference is actor-versus-critic parameter sharing; one shared POLICY plays both seats in both arms):
    shared     the M4.6c network itself (`PolicyValueNet`, one trunk for actor and critic): the control
    separate   `SeparateCriticNet.from_shared(start)`: the same actor, the critic trunk a copy of the shared trunk

Policies (specification strings, `policy_path`):
    start:<s>            the M4.6c 8.192M C1 final of seed s (`ext/C1:<s>`), the common starting point
    <arm>:<s>            the 4.096M continuation of start:<s> in an arm (phase "main")
    <arm>:<s>@<n>        its snapshot at n agent transitions
    ext/<arm>:<s>[@<n>]  the conditional 8.192M extension (phase "ext")
Evaluation jobs pass checkpoint PATHS to the shared engines (`m46c.common.net_of` loads either class, digest verified).

Seeds: the range 89,000-89,999 (disjoint from every earlier study), except the critic calibration, which deliberately
reuses the M4.6d calibration seeds (CRIT_ROOT 99,500) so that the new critics' labels are paired, rollout for rollout,
with the M4.6d labels of the starting critics on the same states. Raw records under results/rl/checkpoints/m46f/
(git-ignored); summaries and reports under results/m46f/.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, Sequence

import numpy as np

from ..m46a.common import load_gz, save_gz, state_from_list, state_to_list  # noqa: F401
from ..m46a.common import load_json as _load_json
from ..m46a.common import save_json as _save_json
from ..m46a.common import write_md as _write_md
from ..m46c import common as C46

NAME = "m46f"
OUT = os.path.join("results", NAME)
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
PREREG_DOC = os.path.join("docs", "m4", "m4_6", "m4_6f_preregistration.md")
ENV = "C1"
ARMS = ("shared", "separate")
SEEDS = (0, 1, 2, 3, 4, 5)
GAMMA = 0.99
REWARD_SCALE = 0.1

# ---------------------------------------------------------------------------------------------- budgets
BUDGET = 4_096_000              # agent transitions per run (= the learning-rate schedule horizon), phase "main"
MIDPOINT = 2_048_000
CURVE_POINTS = (1_024_000, 2_048_000, 3_072_000)
EXT_BUDGET = 8_192_000          # the conditional extension (phase "ext"): new runs from start:s with this schedule
PHASE_BUDGET = {"main": BUDGET, "ext": EXT_BUDGET}
GRAD_EVERY = 25                 # updates between gradient diagnostics (training hook, read only)

# ---------------------------------------------------------------------------------------------- seeds
TRAIN_SEED = 89_000             # PPO seed of the continuation of start:s = TRAIN_SEED + s, the same in both arms
NAT_ROOT = 89_200               # natural self-play labels
H2H_ROOT = 89_300               # head to head from s0
CELL_ROOT = 89_400              # paired cell rollouts (behaviour families)
NATCF_ROOT = 89_500             # paired rollouts from natural states
GRAD_ROOT = 89_600              # gradient-diagnostic minibatch draws
EQUIV_ROOT = 89_700             # initial-equivalence observations
CALIB_ROOT = 99_500             # = m46d.common.CRIT_ROOT (deliberate reuse: paired with the M4.6d labels)


# ---------------------------------------------------------------------------------------------- paths and policies
def env_cfg():
    return C46.arm_cfg(ENV)


def setup(cache: int = C46.CACHE):
    return C46.setup(ENV, cache)


def start_path(seed: int) -> str:
    return C46.policy_path(f"ext/C1:{seed}")


def run_dir(arm: str, seed: int, phase: str = "main") -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return os.path.join(CKPT, "train", phase, arm, f"s{seed}")


def snapshot_path(arm: str, seed: int, n: int, phase: str = "main") -> str:
    return os.path.join(run_dir(arm, seed, phase), "snapshots", f"snap_{n:09d}.pt")


def policy_path(spec: str) -> str:
    """Checkpoint path of a policy specification (module docstring)."""
    kind, rest = spec.split(":", 1)
    if kind == "start":
        return start_path(int(rest))
    phase = "main"
    if "/" in kind:
        phase, kind = kind.split("/", 1)
    if "@" in rest:
        s, n = rest.split("@")
        return snapshot_path(kind, int(s), int(n), phase)
    return os.path.join(run_dir(kind, int(rest), phase), "final.pt")


def net_of(spec_or_path: str):
    """The network (either class) of a specification or checkpoint path, digest verified, cached per process."""
    return C46.net_of(spec_or_path if spec_or_path.endswith(".pt") else policy_path(spec_or_path))


def seed_of(spec: str) -> int:
    return int(spec.split(":", 1)[1].split("@")[0])


# ---------------------------------------------------------------------------------------------- statistics
t95 = C46.t95
seed_stats = C46.seed_stats
excludes0 = C46.excludes0
fci = C46.fci
mean_ci = C46.mean_ci


# ---------------------------------------------------------------------------------------------- I/O and provenance
def meta(**kw) -> dict:
    m = C46.meta(**kw)
    m["study"] = "M4.6f"
    return m


def save_json(name: str, data: dict) -> str:
    return _save_json(name, data, OUT)


def load_json(name: str):
    return _load_json(name, OUT)


def write_md(name: str, text: str) -> str:
    return _write_md(name, text, OUT)


def jdump(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=float)


def prereg_record() -> dict:
    with open(PREREG_DOC, "rb") as f:
        b = f.read()
    return {"document": PREREG_DOC, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def write_prereg() -> dict:
    """Freeze the pre-registration (SHA-256, time) after the smoke test and before any training run of the pilot."""
    path = os.path.join(OUT, "preregistration.json")
    if os.path.exists(path):
        old = load_json("preregistration")
        if old and old["record"]["sha256"] != prereg_record()["sha256"]:
            raise SystemExit("the M4.6f pre-registration changed after it was frozen; record an amendment instead")
        return old
    data = {"meta": meta(stage="prereg"), "record": prereg_record(),
            "note": "frozen after the smoke test (pipeline, equivalence and throughput only) and before any pilot run"}
    save_json("preregistration", data)
    return data


def require_prereg() -> None:
    if not os.path.exists(os.path.join(OUT, "preregistration.json")):
        raise SystemExit("freeze the M4.6f pre-registration first (stage prereg)")
    old = load_json("preregistration")
    if old["record"]["sha256"] != prereg_record()["sha256"]:
        raise SystemExit("the M4.6f pre-registration document differs from its frozen SHA-256")


def amend(note: str) -> dict:
    d = load_json("preregistration") or {}
    d.setdefault("amendments", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note})
    save_json("preregistration", d)
    return d


def run_jobs(fn, jobs: Sequence, workers: int, label: str = "") -> list:
    return C46.run_jobs(fn, jobs, workers, label)


def checkpoint_sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def per_seed_diff(a: Dict[int, float], b: Dict[int, float]) -> Dict[int, float]:
    return {s: a[s] - b[s] for s in a if s in b and np.isfinite(a[s]) and np.isfinite(b[s])}


__all__ = ["NAME", "OUT", "CKPT", "ENV", "ARMS", "SEEDS", "BUDGET", "MIDPOINT", "EXT_BUDGET", "PHASE_BUDGET",
           "env_cfg", "setup", "start_path", "run_dir", "snapshot_path", "policy_path", "net_of", "seed_of",
           "seed_stats", "excludes0", "fci", "mean_ci", "t95", "meta", "save_json", "load_json", "write_md", "jdump",
           "write_prereg", "require_prereg", "amend", "run_jobs", "save_gz", "load_gz", "state_to_list",
           "state_from_list", "checkpoint_sha", "per_seed_diff"]
