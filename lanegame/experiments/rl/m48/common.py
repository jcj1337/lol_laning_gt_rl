"""Shared definitions of M4.8, the lane-exit evaluation study (docs/m4/m4_8/m4_8_protocol.md).

EVALUATION ONLY: no training, no rule, reward, objective (gamma 0.99, continuing time) or default changes. A lane exit
is an evaluation checkpoint (`lanegame.diagnostics.lane_exit`); the game continues after it.

Environments and policies. The matched M4.6c 8.192M continuations of the same six M4.5c finals, one per investment
curve: `ext/baseline:s` in `m45_calibrated()` and `ext/C1:s` in C1 = `m45_calibrated(invest_gain=3.24,
invest_scale=64.0)` (provisional, not adopted), s = 0..5: equal lineage and budgets. A "package" is a curve's
environment played by that curve's policy of the same seed; every continuation runs in the environment of the package
that plays it.

Seat policies (`seat`): ("own",) the job's package policy with CRN sampling (`m45c.horizon.CRNNet`: the uniform of turn
i of seat q is fixed by (seed, 7, q)); ("net", spec) another network, same sampling streams.

Seeds: the range 59,000-59,999 (disjoint from every earlier study; M4.7 used 58,000-58,999). The natural checkpoint
states are replays of the M4.6d natural records (`m46d.records`, game seeds 99,000,000 + 100 s + g), verified turn by
turn against the stored records. Raw files under results/rl/checkpoints/m48/ (git-ignored); summaries and reports under
results/m48/.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Dict

import numpy as np

from ..m46a.common import load_gz, save_gz, state_from_list, state_to_list  # noqa: F401
from ..m46a.common import load_json as _load_json
from ..m46a.common import save_json as _save_json
from ..m46a.common import write_md as _write_md
from ..m46c import common as C46

NAME = "m48"
OUT = os.path.join("results", NAME)
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
PROTOCOL_DOC = os.path.join("docs", "m4", "m4_8", "m4_8_protocol.md")
ENVS = ("baseline", "C1")
SEEDS = (0, 1, 2, 3, 4, 5)
CACHE = 2500                    # exact lane memo per worker (results do not depend on it)

# ---------------------------------------------------------------------------------------------- windows and checkpoints
T_MAX = 40                      # continuation length; every window is a prefix of the same continuations
CANDIDATES = (5, 10, 20)        # candidate default windows
DIAGNOSTIC = (30, 40)           # longer diagnostic windows (unresolved / new consequences)
WINDOWS = CANDIDATES + DIAGNOSTIC
T0_PRIMARY = 94                 # fixed elapsed-turn anchor: the median first turn at level 14 (M4.7)
T0_SENS = (84, 104)             # earlier / later sensitivity anchors
T0_REPL = 50                    # M4.7's turn-50 anchor (replication of its 50 -> 94 conversion slope)
T0S = (T0_REPL, 84, T0_PRIMARY, 104)

# ---------------------------------------------------------------------------------------------- materiality
M_STATE = 0.5                   # a material adjustment of one state's readout (resources: half a wave, one bounty)
M_POP = 0.25                    # a material difference between population means (M4.7's M)
DUE_TURNS = 12                  # a present seat this many turns past its last return is "due" a reset
FRESH_TURNS = 2                 # a present seat back from an absence at most this many turns ago is "fresh"

# ---------------------------------------------------------------------------------------------- seeds
PILOT_ROOT = 59_000             # pilot self-play games (throughput and variance only)
PILOT_ROLL = 59_050             # pilot continuations
VAL_GAME_ROOT = 59_100          # Stage 1 validation self-play games
VAL_ROLL = 59_200               # Stage 1 continuations (cases, mirrors, controlled variants)
VAL_PICK = 59_250               # Stage 1 case selection (which qualifying states are taken)
NAT_ROLL = 59_300               # Stage 2 continuations of the natural checkpoint states (every package and alternative)
TRACE_ROOT = 59_800             # worked examples (none drawn by default)


def env_cfg(env: str):
    return C46.arm_cfg(env)


def setup(env: str = "baseline", cache: int = CACHE):
    """One torch thread and a bounded exact lane memo for an environment; returns its configuration."""
    return C46.setup(env, cache)


def lane_step(cfg):
    from ...m45b import engine as E
    return E.lane_cache(cfg)


def package(env: str, s: int) -> str:
    """The policy of a package: the matched M4.6c 8.192M continuation of seed s in environment env."""
    return f"ext/{env}:{s}"


def other_seed(s: int) -> int:
    """The alternative continuation policy's seed (sensitivity): the next training seed of the same curve."""
    return (s + 1) % len(SEEDS)


_NETS: Dict[str, object] = {}


def net_of(spec: str):
    if spec not in _NETS:
        _NETS[spec] = C46.net_of(spec)
    return _NETS[spec]


def crn(spec: str, k0: int, seed: int, stream: int, T: int):
    from ..m45c import horizon as HZ
    return HZ.CRNNet(net_of(spec), k0, seed, stream, T)


def roll_seed(root: int, index: int, r: int) -> int:
    """Chance and sampling seed of continuation r of state `index` (identical for every package, policy pairing and
    variant of that state: common random numbers)."""
    return int(root) * 10 ** 9 + int(index) * 1000 + int(r)


# ---------------------------------------------------------------------------------------------- statistics
t95 = C46.t95
seed_stats = C46.seed_stats
fci = C46.fci
excludes0 = C46.excludes0


def mean_se(x) -> dict:
    v = np.asarray([y for y in np.ravel(x) if np.isfinite(y)], dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "ci": float("nan"), "n": 0}
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": float(v.mean()), "se": se, "ci": 1.96 * se if n > 1 else float("nan"), "n": n}


# ---------------------------------------------------------------------------------------------- I/O and provenance
def meta(**kw) -> dict:
    m = C46.meta(**kw)
    m["study"] = "M4.8"
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
    """Freeze the protocol (SHA-256 and time) after the pilot, before any main evaluation run."""
    path = os.path.join(OUT, "protocol.json")
    if os.path.exists(path):
        old = load("protocol")
        if old["record"]["sha256"] != protocol_record()["sha256"]:
            raise SystemExit("the M4.8 protocol changed after it was frozen; record an amendment instead")
        return old
    d = {"meta": meta(stage="protocol"), "record": protocol_record(),
         "note": "frozen after the throughput and variance pilot, before any Stage 1 or Stage 2 evaluation run"}
    save_json("protocol", d)
    return d


def require_protocol() -> None:
    path = os.path.join(OUT, "protocol.json")
    if not os.path.exists(path):
        raise SystemExit("freeze the M4.8 protocol first (stage protocol)")
    if load("protocol")["record"]["sha256"] != protocol_record()["sha256"]:
        raise SystemExit("the M4.8 protocol document differs from its frozen SHA-256")


def amend(note: str) -> dict:
    d = load("protocol") or {}
    d.setdefault("amendments", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note})
    save_json("protocol", d)
    return d


__all__ = ["NAME", "OUT", "CKPT", "ENVS", "SEEDS", "T_MAX", "CANDIDATES", "DIAGNOSTIC", "WINDOWS", "T0_PRIMARY",
           "T0_SENS", "T0_REPL", "T0S", "M_STATE", "M_POP", "DUE_TURNS", "FRESH_TURNS", "env_cfg", "setup",
           "lane_step", "package", "other_seed", "net_of", "crn", "roll_seed", "seed_stats", "mean_se", "fci",
           "excludes0", "meta", "save_json", "load", "write_md", "run_jobs", "freeze_protocol", "require_protocol",
           "amend"]
