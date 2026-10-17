"""Shared definitions of M4.6e (docs/m4/m4_6/m4_6e_preregistration.md).

Environment: the M4.6c arm "C1" = `m45_calibrated(invest_gain=3.24, invest_scale=64.0)` (provisional, NOT adopted), the
environment of the principal M4.6d policies; "baseline" = `m45_calibrated()` is used only as an exposure reference.
Nothing here changes either configuration.

Policies (M4.6c specification strings): `ext/C1:<s>` (the six 8.192M general policies of M4.6d) and their training
snapshots `ext/C1:<s>@<n>`. Labels are POLICY-SPECIFIC: every return label is the policy playing itself (the pairing its
critic predicts) in the C1 environment, reported from the labelled observation's own seat, in resources (the critic's
output divided by the training reward scale 0.1), discounted with gamma 0.99 and truncated at a stated horizon.

Seeds: the range 95,000-95,999 (disjoint from every earlier study). Raw records under results/rl/checkpoints/m46e/
(git-ignored); summaries and reports under results/m46e/.
"""

from __future__ import annotations

import os

import numpy as np

from ..m46d import common as D
from ..m46d.common import (CKPT as D_CKPT, GAMMA, REWARD_SCALE, SEEDS, arm_cfg, fmt, general, load_gz,  # noqa: F401
                           mean_ci, net_of, run_jobs, save_gz, setup, state_from_list, state_to_list)
from ..m46a.common import load_json as _load_json
from ..m46a.common import save_json as _save_json
from ..m46a.common import write_md as _write_md

NAME = "m46e"
OUT = os.path.join("results", NAME)
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
PREREG_DOC = os.path.join("docs", "m4", "m4_6", "m4_6e_preregistration.md")
ARM = "C1"

# ---------------------------------------------------------------------------------------------- seeds
OBS_ROOT = 95_000               # observation checks (sampled states)
EXPO_ROOT = 95_100              # reconstructed training occupancy
NAT_ROOT = 95_200               # natural labelled games
CON_ROOT = 95_300               # constructed labelled rollouts
TAIL_ROOT = 95_400              # horizon / residual-tail rollouts
FIT_ROOT = 95_500               # supervised fits and splits
SAMPLE_ROOT = 95_600            # archive sample of the constructed label set
GRAD_ROOT = 95_700              # gradient-share batches

# ---------------------------------------------------------------------------------------------- label design
H_LABEL = 300                   # label horizon (turns); the tail beyond it is measured separately (`tail`)
LABEL_EVERY = 15                # constructed rollouts: a label every 15 turns ...
LABEL_LAST = 150                # ... from turn 0 to 150 (rollouts of LABEL_LAST + H_LABEL turns)
K_CON = 2                       # rollouts per constructed start (same seeds for every variant and policy)
NAT_GAMES, NAT_TURNS, NAT_EVERY, NAT_FIRST = 32, 600, 10, 5
ARCHIVE_SAMPLE = {"natural": 40, "natural+lead": 40, "zone": 60, "freeze": 60}

POLICIES = tuple(general(ARM, s) for s in SEEDS)


def meta(**kw) -> dict:
    m = D.meta(**kw)
    m["study"] = "M4.6e"
    return m


def save_json(name: str, data: dict) -> str:
    return _save_json(name, data, OUT)


def load_json(name: str):
    return _load_json(name, OUT)


def write_md(name: str, text: str) -> str:
    return _write_md(name, text, OUT)


def prereg_record() -> dict:
    import hashlib
    with open(PREREG_DOC, "rb") as f:
        b = f.read()
    return {"document": PREREG_DOC, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def write_prereg() -> dict:
    """Freeze the protocol (SHA-256, time) before the supervised fits."""
    path = os.path.join(OUT, "preregistration.json")
    if os.path.exists(path):
        old = load_json("preregistration")
        if old and old["record"]["sha256"] != prereg_record()["sha256"]:
            raise SystemExit("the pre-registration changed after it was frozen; record an amendment instead")
        return old
    data = {"meta": meta(stage="prereg"), "record": prereg_record(),
            "note": "frozen after the observation, exposure and label stages and before any supervised fit"}
    save_json("preregistration", data)
    return data


def require_prereg() -> None:
    if not os.path.exists(os.path.join(OUT, "preregistration.json")):
        raise SystemExit("run the `prereg` stage first (docs/m4/m4_6/m4_6e_preregistration.md)")


def amend(note: str) -> dict:
    import time
    d = load_json("preregistration") or {}
    d.setdefault("amendments", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note})
    save_json("preregistration", d)
    return d


def discounted(rewards, start: int, horizon: int) -> float:
    """sum_{i < horizon} gamma^i r[start + i] (rewards to the seat of interest)."""
    r = np.asarray(rewards[start:start + horizon], dtype=np.float64)
    return float(np.dot(GAMMA ** np.arange(len(r)), r))


__all__ = ["NAME", "OUT", "CKPT", "ARM", "SEEDS", "POLICIES", "GAMMA", "REWARD_SCALE", "general", "setup", "arm_cfg",
           "net_of", "run_jobs", "save_gz", "load_gz", "state_to_list", "state_from_list", "mean_ci", "fmt", "meta",
           "save_json", "load_json", "write_md", "write_prereg", "require_prereg", "amend", "discounted", "D_CKPT"]
