"""Shared definitions of M4.6d (docs/m4/m4_6/m4_6d_preregistration.md).

Environments: the M4.6c arms. "C1" = `m45_calibrated(invest_gain=3.24, invest_scale=64.0)` is the principal
(provisional, NOT adopted) environment; "baseline" = `m45_calibrated()` is the preserved reference. Nothing here
changes either configuration.

Policies (M4.6c specification strings, `m46c.common.policy_path`):
    ext/C1:<s>        the 8.192M C1-arm continuation of the M4.5c final s (principal general policies)
    ext/baseline:<s>  the 8.192M baseline-arm continuation (reference general policies)
    start:<s>         the M4.5c 16.384M final s (the M4.6a specialists' training opponents for s = 0, 2, 4)
    <path>.pt         any checkpoint; the M4.6a specialists are `specialist(kind, target)` (a checkpoint path)

Seeds: the range 99,000-99,999 (disjoint from every earlier study; M4.6c used 98,000-98,999). Raw records under
results/rl/checkpoints/m46d/ (git-ignored); summaries and reports under results/m46d/.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from ..m46a.common import (fmt, holm, load_gz, run_jobs, save_gz, significant, state_from_list,  # noqa: F401
                           state_to_list, t_pvalue)
from ..m46a.common import load_json as _load_json
from ..m46a.common import save_json as _save_json
from ..m46a.common import write_md as _write_md
from ..m46c import common as C46

NAME = "m46d"
OUT = os.path.join("results", NAME)
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
PREREG_DOC = os.path.join("docs", "m4", "m4_6", "m4_6d_preregistration.md")
SEEDS = (0, 1, 2, 3, 4, 5)
SPEC_TARGETS = (0, 2, 4)
CACHE = 2000                    # exact lane memo per worker (results do not depend on it)
GAMMA = 0.99
REWARD_SCALE = 0.1              # PPOConfig.reward_scale of every evaluated run (checked in run.json)

# ---------------------------------------------------------------------------------------------- seeds
REC_ROOT = 99_000               # natural self-play records
CELL_ROOT = 99_100              # constructed-cell draws
NATSTATE_ROOT = 99_150          # natural freeze / lead state selection
MAT_ROOT = 99_200               # matrix rollouts, discovery sample
CONF_ROOT = 99_300              # matrix rollouts, independent confirmation sample
SPEC_ROOT = 99_400              # specialist evaluation and substitutions
CRIT_ROOT = 99_500              # critic long-return rollouts
TRACE_ROOT = 99_600             # explanatory trajectories


def general(arm: str, s: int) -> str:
    """The 8.192M general policy of an arm (M4.6c)."""
    return f"ext/{arm}:{s}"


def specialist(kind: str, target: int) -> str:
    """The checkpoint path of an M4.6a specialist (`net_of` accepts it directly)."""
    return C46.specialist_path(kind, target)


def setup(arm: str = "C1", cache: int = CACHE):
    return C46.setup(arm, cache)


arm_cfg = C46.arm_cfg
net_of = C46.net_of
install_nets = C46.install_nets


def checkpoint_digest(spec: str) -> str:
    """SHA-256 (first 16 hex) of a policy's checkpoint file (provenance record)."""
    p = spec if spec.endswith(".pt") else C46.policy_path(spec)
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------- statistics
def mean_ci(values: Iterable[float]) -> Tuple[float, float]:
    return C46.mean_ci(values)


def nested_ci(per_unit: Dict[object, Sequence[float]]) -> Tuple[float, float, int]:
    """Mean and 95% half-width over top-level units (e.g. policies), each unit the mean of its own values (states):
    the sampling hierarchy's top level carries the interval. Returns (mean, half-width, units)."""
    means = [float(np.mean(v)) for v in per_unit.values() if len(v)]
    m, h = mean_ci(means)
    return m, h, len(means)


def pos_neg(values: Sequence[float], hw: Sequence[float] = None) -> str:
    """'k+/k-' counts of units by sign (or by interval when half-widths are given)."""
    v = np.asarray(values, float)
    if hw is None:
        return f"{int((v > 0).sum())}+/{int((v < 0).sum())}-"
    h = np.asarray(hw, float)
    return f"{int(((v - h) > 0).sum())}+/{int(((v + h) < 0).sum())}-"


# ---------------------------------------------------------------------------------------------- I/O
def meta(**kw) -> dict:
    m = C46.meta(**kw)
    m["study"] = "M4.6d"
    return m


def save_json(name: str, data: dict) -> str:
    return _save_json(name, data, OUT)


def load_json(name: str):
    return _load_json(name, OUT)


def write_md(name: str, text: str) -> str:
    return _write_md(name, text, OUT)


def prereg_record() -> dict:
    with open(PREREG_DOC, "rb") as f:
        b = f.read()
    return {"document": PREREG_DOC, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def write_prereg() -> dict:
    """Freeze the pre-registration (SHA-256, time) before any principal counterfactual stage."""
    path = os.path.join(OUT, "preregistration.json")
    if os.path.exists(path):
        old = load_json("preregistration")
        if old and old["record"]["sha256"] != prereg_record()["sha256"]:
            raise SystemExit("the pre-registration changed after it was frozen; record an amendment instead")
        return old
    data = {"meta": meta(stage="prereg"), "record": prereg_record(),
            "note": "frozen after the descriptive stage (records, freezes) and before any counterfactual stage"}
    save_json("preregistration", data)
    return data


def require_prereg() -> None:
    if not os.path.exists(os.path.join(OUT, "preregistration.json")):
        raise SystemExit("run the `prereg` stage first (docs/m4/m4_6/m4_6d_preregistration.md)")


def amend(note: str) -> dict:
    d = load_json("preregistration") or {}
    d.setdefault("amendments", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note})
    save_json("preregistration", d)
    return d


__all__ = ["NAME", "OUT", "CKPT", "SEEDS", "SPEC_TARGETS", "GAMMA", "general", "specialist", "setup", "arm_cfg",
           "net_of", "install_nets", "checkpoint_digest", "mean_ci", "nested_ci", "pos_neg", "meta", "save_json",
           "load_json", "write_md", "write_prereg", "require_prereg", "amend", "run_jobs", "save_gz", "load_gz",
           "state_to_list", "state_from_list", "fmt", "holm", "significant", "t_pvalue"]
