"""Shared definitions of NC1 (docs/capacity/nc1_preregistration.md).

Environment `m45_calibrated()` (M4.5c, M5's sp arm), except Q3, which reuses M4.6e's C1 labels. Nothing here changes a
rule, observation, reward, default or stored checkpoint.

Arms of the PPO pilot: one network transform of `ext/baseline:s` each (ARMS), continued under M5's sp protocol with
M5's PPO seed 70,000 + s and the learning rate annealed over M5's 32,768,000 budget, stopped at BUDGET. Each run is
paired with M5's sp snapshot of the same seed at BUDGET.
    C   a copy of ext/baseline:s (2 x 128): the control, run only to reproduce M5's history (smoke)
    W   `widen_hidden` to 2 x 256
    F   `with_strength_features`: 2 x 128 plus M4.6e's six observation-derived strength features

Seeds: 48,000-48,999 for everything NC1 draws (PPO seeds are M5's, by design). Raw records under
results/rl/checkpoints/nc1/ (git-ignored); summaries and reports under results/nc1/.
"""

from __future__ import annotations

import hashlib
import os

from ..m46a.common import load_gz, mean_ci, save_gz, t95  # noqa: F401
from ..m46a.common import load_json as _load_json
from ..m46a.common import meta as _meta
from ..m46a.common import save_json as _save_json
from ..m46a.common import write_md as _write_md
from ..m45c.common import run_jobs  # noqa: F401

NAME = "nc1"
OUT = os.path.join("results", NAME)
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
PREREG_DOC = os.path.join("docs", "capacity", "nc1_preregistration.md")

SEEDS = (0, 1, 2, 3, 4, 5)
SCHEDULE = 32_768_000          # M5's budget: the learning-rate schedule horizon
BUDGET = 16_384_000            # NC1's stop point, an exact prefix of that schedule (M5's registered midpoint)
WIDE = (256, 256)
WIDEN_SEED = 48_000            # the incoming weights of W's new units: torch seed WIDEN_SEED + s

Q1_ROOT = 48_100
N_TRAIN_ROOT, P_TRAIN_ROOT, N_TEST_ROOT, P_TEST_ROOT = 48_200, 48_300, 48_400, 48_500
FIT_ROOT = 48_600
NAT_EVAL_ROOT = 48_700
H2H_ROOT = 48_800
CELL_ROOT = 48_900

ARMS = {
    "C": "a copy of ext/baseline:s (2 x 128); reproduction control only",
    "W": "widen_hidden(ext/baseline:s, (256, 256))",
    "F": "with_strength_features(ext/baseline:s): 2 x 128 + 6 strength features",
}
PILOT_ARMS = ("W", "F")
FEATURE_KEYS = ("own_champ_attack_power", "enemy_champ_attack_power", "own_champ_hp_frac", "enemy_champ_hp_frac",
                "own_champ_absent", "enemy_champ_absent")


def feature_index(cfg=None):
    from ....config import GameConfig
    from ..m46e.features import obs_index
    idx = obs_index(cfg or GameConfig.m45_calibrated())
    return [idx[k] for k in FEATURE_KEYS]


def initial_net(arm: str, seed: int):
    """The arm's network at the start of its run (module docstring)."""
    from ....rl.network import widen_hidden, with_strength_features
    from ..m5.train import initial_net as m5_initial
    net = m5_initial("sp", seed)
    if arm == "C":
        return net
    if arm == "W":
        return widen_hidden(net, WIDE, seed=WIDEN_SEED + seed)
    if arm == "F":
        return with_strength_features(net, feature_index())
    raise ValueError(arm)


def run_dir(arm: str, seed: int, phase: str = "main") -> str:
    if arm not in ARMS:
        raise ValueError(arm)
    return os.path.join(CKPT, "train", phase, arm, f"s{seed}")


def final_path(arm: str, seed: int) -> str:
    return os.path.join(run_dir(arm, seed), "final.pt")


def control_path(seed: int, at: int = BUDGET) -> str:
    """M5's sp snapshot of seed s at `at` transitions: the paired control."""
    return os.path.join("results", "rl", "checkpoints", "m5", "train", "main", "sp", f"s{seed}", "snapshots",
                        f"snap_{at:09d}.pt")


def meta(**kw) -> dict:
    m = _meta(**kw)
    m["study"] = "NC1"
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
    return {"document": PREREG_DOC.replace("\\", "/"), "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def write_prereg(note: str) -> dict:
    """Freeze the protocol (SHA-256, time). Refuses to re-freeze a changed document."""
    old = load_json("preregistration")
    if old:
        if old["record"]["sha256"] != prereg_record()["sha256"]:
            raise SystemExit("the pre-registration changed after it was frozen; record an amendment instead")
        return old
    data = {"meta": meta(stage="prereg"), "record": prereg_record(), "note": note}
    save_json("preregistration", data)
    return data


def require_prereg() -> None:
    old = load_json("preregistration")
    if not old:
        raise SystemExit(f"run the `prereg` stage first ({PREREG_DOC})")
    if old["record"]["sha256"] != prereg_record()["sha256"]:
        raise SystemExit("the pre-registration document differs from the frozen one")


def amend(note: str) -> dict:
    import time
    d = load_json("preregistration") or {}
    d.setdefault("amendments", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note})
    save_json("preregistration", d)
    return d
