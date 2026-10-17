"""Shared definitions of OM1, the opponent-modelling feasibility study (docs/opponent_modelling/om1_protocol.md).

EVALUATION ONLY: no training, no network, observation, rule, reward or default change. OM1 asks whether Bayesian
opponent modelling could pay in this game at all, before anything is designed: (1) can the lane phase reveal which kind
of opponent one faces (identifiability), and (2) would knowing it change what to play (value of information)?

The focal policy pi_s is the adopted M5 baseline's matched policy `ext/baseline:s` (the M4.6c 8.192M continuation in
`m45_calibrated()`), s = 0..5; the training seeds are the units of population claims.

Opponent types (LIBRARY): existing policies of distinct styles, nothing new is trained or scripted.
    scripted   the nine M4.5c reference strategies (`m45c.references.SUITE`): deterministic functions of the state
    learned    existing checkpoints of distinct kinds (paths in LEARNED): an earlier-lineage final (peer), an M4.5c
               best-response attacker, an M4.6a freeze and gap specialist, an M4.7 short-horizon (H 60) policy, and the
               immature M4.5c pilot policy
Out-of-library opponents (OOL): other instances of the learned families and stochastic mixtures of scripted styles that
no library type represents.

Seeds: the range 60,000-60,999 (disjoint from every earlier study). Raw files under results/rl/checkpoints/om1/
(git-ignored); summaries and reports under results/om1/.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Dict

import numpy as np

from ..m46a.common import load_gz, save_gz  # noqa: F401
from ..m46a.common import load_json as _load_json
from ..m46a.common import save_json as _save_json
from ..m46a.common import write_md as _write_md
from ..m46c import common as C46

NAME = "om1"
OUT = os.path.join("results", NAME)
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
PROTOCOL_DOC = os.path.join("docs", "opponent_modelling", "om1_protocol.md")
SEEDS = (0, 1, 2, 3, 4, 5)
ENV = "baseline"
CACHE = 2500

# ---------------------------------------------------------------------------------------------- the type library
SCRIPTED = ("g_farm", "g_shove", "g_shove_sr", "wk_shove_reset_sh", "g_shove_edge", "g_bully", "g_safe",
            "fz_guard_sh", "g_shove_ward")
_CK = os.path.join("results", "rl", "checkpoints")
LEARNED: Dict[str, str] = {
    "peer": os.path.join(_CK, "m45c", "full", "s3", "final.pt"),           # M4.5c 8.192M final, seed 3
    "attacker": os.path.join(_CK, "m45c", "attack", "ext_s3", "final.pt"),  # M4.5c best-response attacker (vs s3)
    "freezer": os.path.join(_CK, "m46a", "e4", "freeze_s2", "final.pt"),    # M4.6a freeze specialist
    "gapper": os.path.join(_CK, "m46a", "e4", "gap_s2", "final.pt"),        # M4.6a large-gap specialist
    "short": os.path.join(_CK, "m47", "train", "main", "short", "s0", "final.pt"),  # M4.7 H 60 arm
    "early": os.path.join(_CK, "m45c", "pilot", "s100", "final.pt"),       # M4.5c pilot, 1.024M (immature)
}
LIBRARY = SCRIPTED + tuple(LEARNED)
# out-of-library opponents: (kind, argument)
OOL: Dict[str, tuple] = {
    "attacker_b": ("net", os.path.join(_CK, "m45c", "attack", "ext_s0", "final.pt")),
    "freezer_b": ("net", os.path.join(_CK, "m46a", "e4", "freeze_s4", "final.pt")),
    "gapper_b": ("net", os.path.join(_CK, "m46a", "e4", "gap_s4", "final.pt")),
    "short_b": ("net", os.path.join(_CK, "m47", "train", "main", "short", "s1", "final.pt")),
    "c1_policy": ("net", C46.policy_path("ext/C1:2")),
    "mix_bully_safe": ("mix", ("g_bully", "g_safe", 0.5)),
    "mix_shove_farm": ("mix", ("g_shove", "g_farm", 0.5)),
}

# ---------------------------------------------------------------------------------------------- modelling settings
EPS = 0.05                      # trembling-hand allowance of a scripted type's likelihood
EPS_SENS = (0.01, 0.2)
THETA = 0.9                     # belief needed before the switcher acts on a type
THETA_SENS = (0.7, 0.97)
ID_P = 0.9                      # "identified": the true type's posterior >= ID_P
CHECK_TURNS = (1, 2, 3, 5, 10, 15, 20, 30, 40, 60, 94)
ID_TURN = 40                    # the identifiability criterion's turn (well inside the lane phase)
M = 0.25                        # material value difference (resources; M4.7 / M4.8)

# ---------------------------------------------------------------------------------------------- game settings
LANE_END = 94
H_READ = 20                     # the adopted lane-exit window (E_20 at turn 94 = the lead at turn 114 in expectation)
T_SEL, T_EVAL = 120, 150
READ_TURN = LANE_END + H_READ   # 114

# ---------------------------------------------------------------------------------------------- seeds
SMOKE_ROOT = 60_000
SEL_ROOT = 60_100               # stage A (selection matrix)
EVAL_ROOT = 60_200              # stage B (pi, oracle, switcher against the library)
OOL_ROOT = 60_300               # stage C (pi and switcher against out-of-library opponents)


def game_seed(root: int, cell: int, m: int) -> int:
    """Chance and sampling seed of game m of a cell (identical for every responder of the cell: common random
    numbers)."""
    return int(root) * 10 ** 7 + int(cell) * 10 ** 4 + int(m)


def pi_path(s: int) -> str:
    return C46.policy_path(f"ext/baseline:{s}")


def setup(cache: int = CACHE):
    return C46.setup(ENV, cache)


def lane_step(cfg):
    from ...m45b import engine as E
    return E.lane_cache(cfg)


_NETS: Dict[str, object] = {}


def net_of(path: str):
    if path not in _NETS:
        _NETS[path] = C46.net_of(path)
    return _NETS[path]


# ---------------------------------------------------------------------------------------------- statistics and I/O
seed_stats = C46.seed_stats
fci = C46.fci


def mean_se(x) -> dict:
    v = np.asarray([y for y in np.ravel(x) if np.isfinite(y)], dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "ci": float("nan"), "n": 0}
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": float(v.mean()), "se": se, "ci": 1.96 * se if n > 1 else float("nan"), "n": n}


def meta(**kw) -> dict:
    m = C46.meta(**kw)
    m["study"] = "OM1"
    return m


def save_json(name: str, data: dict, out: str = OUT) -> str:
    return _save_json(name, data, out)


def load(name: str, out: str = OUT):
    return _load_json(name, out)


def write_md(name: str, text: str, out: str = OUT) -> str:
    return _write_md(name, text, out)


def run_jobs(fn, jobs, workers: int, label: str = ""):
    return C46.run_jobs(fn, jobs, workers, label)


def protocol_record() -> dict:
    with open(PROTOCOL_DOC, "rb") as f:
        b = f.read()
    return {"document": PROTOCOL_DOC, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def freeze_protocol() -> dict:
    path = os.path.join(OUT, "protocol.json")
    if os.path.exists(path):
        old = load("protocol")
        if old["record"]["sha256"] != protocol_record()["sha256"]:
            raise SystemExit("the OM1 protocol changed after it was frozen; record an amendment instead")
        return old
    d = {"meta": meta(stage="protocol"), "record": protocol_record(),
         "note": "frozen after the smoke and throughput check, before any selection or evaluation game"}
    save_json("protocol", d)
    return d


def require_protocol() -> None:
    path = os.path.join(OUT, "protocol.json")
    if not os.path.exists(path):
        raise SystemExit("freeze the OM1 protocol first (stage protocol)")
    if load("protocol")["record"]["sha256"] != protocol_record()["sha256"]:
        raise SystemExit("the OM1 protocol document differs from its frozen SHA-256")


def amend(note: str) -> dict:
    d = load("protocol") or {}
    d.setdefault("amendments", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note})
    save_json("protocol", d)
    return d
