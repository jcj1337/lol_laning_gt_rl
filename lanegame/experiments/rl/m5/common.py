"""Shared definitions of M5, the final training and evaluation campaign (design docs/m5/m5_design.md;
pre-registration docs/m5/m5_preregistration.md). M5a, this package's first stage set, prepares the campaign and runs
only bounded readiness checks; the campaign itself is gated (`--confirm-m5-train` and the per-stage flags).

Arms (both continue the adopted baseline policy `ext/baseline:s`, the M4.6c 8.192M continuation of the M4.5c 16.384M
final of seed s, s = 0..5; one matched pair per seed):
    om  (A) opponent modelling  `GameConfig.m45_calibrated(obs_last_action=True)` (103 entries) + the OM2 belief's
                                13 features = 116 network inputs, widened from ext/baseline:s with zero weights (the
                                same actor and value outputs at the start); the OM2 treatment collector: the M4.6c
                                start mixture, 50% current-policy self-play and 50% games against one of OM2's 15 fixed
                                training opponents (only the learner trains there), one belief per learner seat
                                (lambda 0.97, eps 0.05, 11 library types + "mixed" + "unknown"), reset at every game
                                start; the library is fixed for the whole campaign
    sp  (B) pure self-play      `GameConfig.m45_calibrated()` (90 entries, no previous action, no belief); the M4.6c
                                continuation exactly: the M4.6c start mixture and SharedConfig(current_fraction=1.0):
                                every training game is a current-policy game (the current network plays both seats,
                                both train); snapshots are saved every 25,600 transitions into a pool of 8 that is
                                never sampled (current_fraction 1 draws no historical game)
Shared by both arms: the M4.6c PPO settings, PPO seed TRAIN_SEED + s, a fresh Adam at 3e-4 annealed linearly to 0 over
BUDGET agent transitions, BUDGET additional agent transitions per run, snapshots every 25,600.

Policy specifications (`policy_path`):
    start:<s>                 ext/baseline:s (the common start of the pair)
    <arm>:<s>                 the final of run (arm, s) of phase "main"
    <arm>:<s>@<n>             the first snapshot of that run at or after n agent transitions (MIDPOINT, P75, curves)
    <phase>/<arm>:<s>[@<n>]   the same in another phase (smoke, bench)
    m46c:<spec>               an M4.6c specification (m46c.common.policy_path), e.g. m46c:ext/baseline:3
    om2:<arm>:<s>             an OM2 pilot final (control | treatment)
    spec:<path>               any checkpoint

Seeds: the block 70,000-70,999 (`SEED_BLOCK`; disjoint from every earlier study's roots, tested). Raw files under
results/rl/checkpoints/m5/ (git-ignored); summaries under results/m5/.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..m46a.common import load_gz, save_gz  # noqa: F401  (re-exported)
from ..m46a.common import load_json as _load_json
from ..m46a.common import save_json as _save_json
from ..m46a.common import write_md as _write_md
from ..m46c import common as C46
from ..om2 import common as O2

NAME = "m5"
OUT = os.path.join("results", NAME)
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
DOCS = os.path.join("docs", "m5")
DESIGN_DOC = os.path.join(DOCS, "m5_design.md")
PREREG_DOC = os.path.join(DOCS, "m5_preregistration.md")
READINESS_PROTOCOL_DOC = os.path.join(DOCS, "m5a_readiness_protocol.md")

ARMS = ("om", "sp")
ARM_LABEL = {"om": "A: opponent modelling", "sp": "B: pure self-play"}
SEEDS = (0, 1, 2, 3, 4, 5)
PHASES = ("smoke", "bench", "main")

# ---------------------------------------------------------------------------------------------- budgets (proposal)
BUDGET = 32_768_000             # additional agent transitions per run (= the learning-rate schedule horizon)
MIDPOINT = 16_384_000           # the matched half-budget checkpoint
P75 = 24_576_000                # the convergence probe's 75% snapshot
SNAP_EVERY = C46.SNAP_EVERY     # 25,600
CURVE_POINTS = tuple(range(2_048_000, BUDGET + 1, 2_048_000))   # 16 monitoring points per run
PRIOR_TRANSITIONS = 16_384_000 + 8_192_000                      # lineage of ext/baseline:s (verified in readiness)

# ---------------------------------------------------------------------------------------------- seeds
SEED_BLOCK = (70_000, 70_999)
TRAIN_SEED = 70_000             # PPO seed of pair s: TRAIN_SEED + s, the same in both arms
BENCH_ROOT = 70_050             # concurrency benchmark runs (PPO seed BENCH_ROOT + k)
CURVE_ROOT = 70_100             # M5b learning curves (snapshots vs the fixed reference panel)
HIST_ROOT = 70_150              # M5b history comparisons (final vs own snapshots, vs the start)
H2H_ROOT = 70_200               # head to head from s0 (arms, seeds, history)
FAMILY_ROOT = 70_300            # M5d opponent families (styles, competent, misspecified, imitate-then-switch)
BATTERY_STATE_ROOT = 70_400     # M5c strategic battery: natural states
BATTERY_ROLL_ROOT = 70_450      # M5c strategic battery: paired rollouts
CELL_ROOT = 70_500              # M5c M4.6 limitation cells (lead premium, freezes, Recall windows, large gaps)
LANE_EXIT_ROOT = 70_550         # M5c lane-exit readout continuations
ATTACK_TRAIN_ROOT = 70_600      # M5d attackers: PPO seed ATTACK_TRAIN_ROOT + 10 * target_kind + s
ATTACK_EVAL_ROOT = 70_700       # M5d attackers: independent evaluation games
READY_ROOT = 70_800             # M5a readiness probes
TRACE_ROOT = 70_900             # representative viewer traces
ROOTS = {"TRAIN_SEED": TRAIN_SEED, "BENCH_ROOT": BENCH_ROOT, "CURVE_ROOT": CURVE_ROOT, "HIST_ROOT": HIST_ROOT,
         "H2H_ROOT": H2H_ROOT, "FAMILY_ROOT": FAMILY_ROOT, "BATTERY_STATE_ROOT": BATTERY_STATE_ROOT,
         "BATTERY_ROLL_ROOT": BATTERY_ROLL_ROOT, "CELL_ROOT": CELL_ROOT, "LANE_EXIT_ROOT": LANE_EXIT_ROOT,
         "ATTACK_TRAIN_ROOT": ATTACK_TRAIN_ROOT, "ATTACK_EVAL_ROOT": ATTACK_EVAL_ROOT, "READY_ROOT": READY_ROOT,
         "TRACE_ROOT": TRACE_ROOT}

# ---------------------------------------------------------------------------------------------- the belief (OM2, unchanged)
LAMBDA, EPS, K = O2.LAMBDA, O2.EPS, O2.K
BELIEF_TYPES = O2.BELIEF_TYPES
TRAIN_OPPONENTS = O2.TRAIN_OPPONENTS
P_LIB = O2.P_LIB
OBS_OM, NET_IN_OM = O2.OBS, O2.NET_IN          # 103, 116
OBS_SP = 90


# ---------------------------------------------------------------------------------------------- configurations
_CFGS: Dict[str, object] = {}


def arm_cfg(arm: str):
    """The validated GameConfig of an arm (cached per process). The dynamics are identical: obs_last_action changes
    the observation only (mathematical reference §15.4)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    if arm not in _CFGS:
        from ....config import GameConfig
        cfg = GameConfig.m45_calibrated(obs_last_action=True) if arm == "om" else GameConfig.m45_calibrated()
        cfg.validate()
        _CFGS[arm] = cfg
    return _CFGS[arm]


def net_inputs(arm: str) -> int:
    return NET_IN_OM if arm == "om" else OBS_SP


def setup(arm: str = "sp", cache: int = C46.CACHE):
    """One torch thread and a bounded exact lane memo for the arm's configuration (the two arms share the lane step:
    `obs_last_action` does not change it); returns the configuration."""
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
    if arm not in ARMS:
        raise ValueError(arm)
    return os.path.join(CKPT, "train", phase, arm, f"s{seed}")


def start_spec(seed: int) -> str:
    return f"ext/baseline:{int(seed)}"


def start_path(seed: int) -> str:
    return C46.policy_path(start_spec(seed))


def snapshot_at(arm: str, seed: int, n: int, phase: str = "main") -> str:
    """The first snapshot of a run at or after n agent transitions (the om arm's updates are not a fixed size, so its
    snapshots fall a few thousand transitions after the multiples of SNAP_EVERY)."""
    d = os.path.join(run_dir(arm, seed, phase), "snapshots")
    if not os.path.isdir(d):
        raise FileNotFoundError(d)
    best = None
    for f in sorted(os.listdir(d)):
        if f.startswith("snap_") and f.endswith(".pt"):
            m = int(f[5:-3])
            if m >= n and (best is None or m < best):
                best = m
    if best is None:
        raise FileNotFoundError(f"{d}: no snapshot at or after {n:,}")
    return os.path.join(d, f"snap_{best:09d}.pt")


def policy_path(spec: str) -> str:
    """Checkpoint path of a policy specification (module docstring)."""
    if spec.startswith("spec:"):
        return spec[5:]
    if spec.startswith("m46c:"):
        return C46.policy_path(spec[5:])
    if spec.startswith("om2:"):
        arm, s = spec[4:].split(":")
        return os.path.join(O2.run_dir(arm, int(s)), "final.pt")
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
        return snapshot_at(kind, int(s), int(n), phase)
    return os.path.join(run_dir(kind, int(rest), phase), "final.pt")


def arm_of(spec: str) -> str:
    """The input format of a policy: "om" (116 inputs, needs the belief) or "sp" (90 inputs)."""
    if spec.startswith("om2:"):
        return "om"
    kind = spec.split(":", 1)[0].split("/")[-1]
    return kind if kind in ARMS else "sp"


# ---------------------------------------------------------------------------------------------- provenance
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint_identity(path: str) -> dict:
    """The identity of a checkpoint: file SHA-256, weights digest (verified on load), shape and lineage metadata."""
    from ....rl import checkpoint_info, load_checkpoint
    load_checkpoint(path)                                  # verifies the weights against the stored digest
    info = checkpoint_info(path)
    md = info.get("metadata") or {}
    spec = info.get("spec") or {}
    keep = ("name", "kind", "run", "seed", "agent_transitions", "env_turns", "update", "snapshot_id", "stop_at")
    return {"path": path.replace("\\", "/"), "file_sha256": file_sha256(path), "weights_digest": info["digest"],
            "obs_size": spec.get("obs_size"), "n_actions": spec.get("n_actions"),
            "metadata": {k: md[k] for k in keep if k in md},
            "schedule_total_steps": (md.get("ppo") or {}).get("total_steps"),
            "provenance_commit": (info.get("provenance") or {}).get("git_commit")}


def code_digest(root: str = "lanegame") -> dict:
    """SHA-256 over every Python source file of the package (sorted paths), with the git commit and dirty flag: the code
    revision of a run, valid also for an uncommitted working tree."""
    from ..m45c.common import git_commit, git_dirty
    h = hashlib.sha256()
    n = 0
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        for f in sorted(fn):
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                h.update(p.replace("\\", "/").encode())
                h.update(file_sha256(p).encode())
                n += 1
    try:
        diff = subprocess.check_output(["git", "diff", "HEAD", "--stat"], text=True, stderr=subprocess.DEVNULL)
    except Exception:                                      # pragma: no cover
        diff = ""
    return {"git_commit": git_commit(), "working_tree_dirty": git_dirty(), "python_files": n,
            "sources_sha256": h.hexdigest(), "diff_stat": diff.strip().splitlines()[-1:] if diff.strip() else []}


def meta(**kw) -> dict:
    m = C46.meta(**kw)
    m["study"] = "M5"
    return m


def save_json(name: str, data: dict, out: str = OUT) -> str:
    return _save_json(name, data, out)


def load(name: str, out: str = OUT):
    return _load_json(name, out)


def write_md(name: str, text: str, out: str = OUT) -> str:
    return _write_md(name, text, out)


def run_jobs(fn, jobs, workers: int, label: str = ""):
    return C46.run_jobs(fn, jobs, workers, label)


def jdump(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, default=float)
    os.replace(tmp, path)


def jload(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------- pre-registration
def doc_record(path: str) -> dict:
    with open(path, "rb") as f:
        b = f.read()
    return {"document": path.replace("\\", "/"), "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def write_prereg(status: str = "awaiting campaign authorization") -> dict:
    """Freeze the pre-registration (SHA-256 and time). A frozen record is never rewritten: a changed document must be
    recorded as a dated amendment (`amend`), never re-frozen."""
    path = os.path.join(OUT, "preregistration.json")
    if os.path.exists(path):
        old = load("preregistration")
        if old["record"]["sha256"] != doc_record(PREREG_DOC)["sha256"]:
            raise SystemExit("the M5 pre-registration changed after it was frozen; record a dated amendment instead")
        return old
    d = {"meta": meta(stage="prereg"), "record": doc_record(PREREG_DOC), "design": doc_record(DESIGN_DOC),
         "status": status, "note": "frozen after the M5a readiness checks, before any M5 campaign run; changes need a "
         "dated amendment"}
    save_json("preregistration", d)
    return d


def require_prereg() -> dict:
    path = os.path.join(OUT, "preregistration.json")
    if not os.path.exists(path):
        raise SystemExit("freeze the M5 pre-registration first (stage prereg)")
    d = load("preregistration")
    if d["record"]["sha256"] != doc_record(PREREG_DOC)["sha256"]:
        raise SystemExit("the M5 pre-registration document differs from its frozen SHA-256")
    return d


def amend(note: str, document_sha256: Optional[str] = None) -> dict:
    d = load("preregistration") or {}
    d.setdefault("amendments", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note,
                                           "document_sha256": document_sha256})
    save_json("preregistration", d)
    return d


# ---------------------------------------------------------------------------------------------- statistics
seed_stats = C46.seed_stats
t95 = C46.t95
fci = C46.fci


def mean_se(x: Iterable[float]) -> dict:
    v = np.asarray([y for y in np.ravel(list(x)) if np.isfinite(y)], dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "ci": float("nan"), "n": 0}
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": float(v.mean()), "se": se, "ci": 1.96 * se if n > 1 else float("nan"), "n": n}


def all_roots() -> List[int]:
    return sorted(ROOTS.values())


__all__ = ["NAME", "OUT", "CKPT", "ARMS", "SEEDS", "PHASES", "BUDGET", "MIDPOINT", "P75", "SNAP_EVERY", "CURVE_POINTS",
           "SEED_BLOCK", "ROOTS", "arm_cfg", "net_inputs", "setup", "run_dir", "start_spec", "start_path",
           "snapshot_at", "policy_path", "arm_of", "file_sha256", "checkpoint_identity", "code_digest", "meta",
           "save_json", "load", "write_md", "run_jobs", "jdump", "jload", "doc_record", "write_prereg",
           "require_prereg", "amend", "seed_stats", "t95", "fci", "mean_se"]
