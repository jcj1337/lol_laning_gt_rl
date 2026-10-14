"""Shared definitions of M4.4: the three environments, the training settings, run paths, seeds, agent specifications
and the checkpoint compatibility guard.

Environments (the presets are not changed by M4.4; their mechanics are the M4.3c / M4.3d reports'):

    four    GameConfig.m4_four_actions()              control: the original four-action economy
    econ    GameConfig.m4_four_actions_econ()         the calibrated economic candidate (M4.3c stabilization)
    aggro   GameConfig.m4_four_actions_econ_aggro()   econ + corrected tower retaliation on dives (M4.3d); the
                                                      candidate for substantial training

All three have the same 88-entry observation layout and the same 61-code action space, so one network architecture
serves all of them. They are NOT interchangeable for a trained network: econ / aggro scale the absence entry by 5
instead of 3 (champion_absent_max) and change the economy, so a checkpoint is only loaded for the environment it was
trained on unless a cross-environment evaluation is requested explicitly (`load_policy(..., cross_env=True)`).

Training settings (identical for every environment; the M4.3b condition `ent03`, which the M4.3c pilot used):
`lanegame.rl.train_shared` with `PPOConfig.continuing` (gamma = the game's 0.99, GAE lambda 0.95, segments of 128
turns, games of 512 turns with seeded staggered first games, reward scale 0.1, value clip 0.5, Adam 3e-4 annealed to
0, 4 epochs, minibatch 256, clip 0.2, entropy 0.03, 16 environments, 2 x 128 MLP), 100% current-policy games (both
seats trained), snapshots every 25,600 agent transitions (kept for historical evaluation, not as training opponents).
One update = 16 environments x 128 turns x 2 seats = 4,096 agent transitions; 2,048,000 transitions = 500 updates.

Seeds. Training seeds 0, 1, 2 in EVERY environment: a seed fixes the initial network (torch.manual_seed) and the
collector's streams (seed + 1), so run (env, s) and run (env', s) start from bit-identical weights and differ only by
the environment (a paired design; their trajectories diverge as soon as the dynamics do). Evaluation streams of M4.4
derive from SeedSequence roots 80,000-80,999, disjoint from every earlier study (10,000-79,999).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....config import GameConfig
from ....io import dump_json

NAME = "m44"
OUT_A = os.path.join("results", "m44a")
OUT_B = os.path.join("results", "m44b")
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
PHASES = ("smoke", "main", "ext")       # run-directory phase: M4.4a smoke runs, M4.4b main runs, M4.4b matched
                                        # 2x-budget runs (the extension of the training-extension policy)
Z95 = 1.96


# ---------------------------------------------------------------------------------------------- environments
@dataclass(frozen=True)
class EnvSpec:
    key: str
    preset: str                         # GameConfig classmethod
    role: str


ENVS: Dict[str, EnvSpec] = {
    "four": EnvSpec("four", "m4_four_actions", "control: the original four-action economy (M4.3c redesign)"),
    "econ": EnvSpec("econ", "m4_four_actions_econ", "the calibrated economic candidate (M4.3c stabilization)"),
    "aggro": EnvSpec("aggro", "m4_four_actions_econ_aggro",
                     "the calibrated economy with corrected tower retaliation on dives (M4.3d); the candidate for "
                     "substantial training"),
}
ENV_ORDER = ("four", "econ", "aggro")


def env_cfg(key: str) -> GameConfig:
    if key not in ENVS:
        raise ValueError(f"unknown environment {key!r}; choose from {list(ENVS)}")
    cfg = getattr(GameConfig, ENVS[key].preset)()
    cfg.validate()
    return cfg


# presets a checkpoint or configuration can be identified with (by exact equality of every field)
KNOWN_PRESETS = ("m45_unified", "m45_unified_r1", "m4_four_actions_econ_aggro", "m4_four_actions_econ",
                 "m4_four_actions", "m4_lasthit", "m4_lane", "m4_natural", "m4_natural_prototype", "m4_emergent", "m4",
                 "champions")


def preset_name(cfg: GameConfig) -> Optional[str]:
    """The named preset `cfg` equals field by field, or None (a variant)."""
    for name in KNOWN_PRESETS:
        try:
            if getattr(GameConfig, name)() == cfg:
                return name
        except Exception:                                   # pragma: no cover - a preset that needs arguments
            continue
    return None


def env_key(cfg: GameConfig) -> Optional[str]:
    """The M4.4 environment key of `cfg` (None for any other configuration)."""
    for key in ENV_ORDER:
        if env_cfg(key) == cfg:
            return key
    return None


# ---------------------------------------------------------------------------------------------- training settings
SEEDS = (0, 1, 2)
BUDGET = 2_048_000                      # agent transitions per M4.4b run (500 updates of 4,096)
SMOKE_BUDGET = 204_800                  # agent transitions per M4.4a smoke run (50 updates)
EXT_BUDGET = 4_096_000                  # agent transitions per M4.4b matched extension run (1,000 updates; every
                                        # setting else identical, so the learning rate anneals over the longer run)
N_ENVS = 16
ENT_COEF = 0.03
CURRENT_FRACTION = 1.0
SNAP_EVERY = 25_600
POOL = 8
HIDDEN = (128, 128)


def ppo_config(key: str, seed: int, steps: int = BUDGET, **kw):
    """The PPO configuration of one run; identical for every environment except the game itself."""
    from ....rl import PPOConfig
    base = dict(total_steps=steps, n_envs=N_ENVS, seed=seed, torch_threads=1, ent_coef=ENT_COEF, hidden=HIDDEN)
    base.update(kw)
    return PPOConfig.continuing(env_cfg(key), **base)


def shared_config():
    from ....rl import SharedConfig
    return SharedConfig(current_fraction=CURRENT_FRACTION, snapshot_every=SNAP_EVERY, pool_size=POOL)


def training_spec() -> dict:
    """The complete training specification (JSON), as recorded in every report and run directory."""
    p = ppo_config("aggro", 0).to_dict()
    p.pop("seed")
    p.pop("total_steps")
    return {"environments": {k: {"preset": f"GameConfig.{ENVS[k].preset}()", "role": ENVS[k].role} for k in ENV_ORDER},
            "seeds": list(SEEDS), "budget_main": BUDGET, "budget_smoke": SMOKE_BUDGET, "ppo": p,
            "shared": shared_config().to_dict(), "transitions_per_update": N_ENVS * p["segment_length"] * 2,
            "updates_main": BUDGET // (N_ENVS * p["segment_length"] * 2),
            "network": {"class": "PolicyValueNet", "obs_size": 88, "n_actions": 61, "hidden": list(HIDDEN),
                        "activation": "tanh"}}


# ---------------------------------------------------------------------------------------------- runs and files
def run_name(key: str, seed: int) -> str:
    return f"{key}_s{seed}"


def run_dir(phase: str, key: str, seed: int) -> str:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; choose from {PHASES}")
    return os.path.join(CKPT, phase, run_name(key, seed))


def final_path(phase: str, key: str, seed: int) -> str:
    return os.path.join(run_dir(phase, key, seed), "final.pt")


def snapshot_paths(phase: str, key: str, seed: int) -> List[Tuple[int, str]]:
    """[(agent transitions, path)] of a run's snapshots, in order."""
    d = os.path.join(run_dir(phase, key, seed), "snapshots")
    if not os.path.isdir(d):
        return []
    return [(int(f[5:-3]), os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith("snap_") and f.endswith(".pt")]


def history(phase: str, key: str, seed: int) -> List[dict]:
    with open(os.path.join(run_dir(phase, key, seed), "history.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# the M4.3c stabilization pilot (2 seeds x 2,048,000 transitions, entropy 0.03; EXPLORATORY): the only four-action
# checkpoints trained before M4.4, used by M4.4a to demonstrate the diagnostics on non-trivial policies
PILOT_DIR = os.path.join("results", "rl", "checkpoints", "m43c_pilot")
PILOT_ENV = {"base": "four", "candidate": "econ"}


def pilot_path(variant: str, seed: int) -> str:
    return os.path.join(PILOT_DIR, f"{variant}_s{seed}", "final.pt")


# ---------------------------------------------------------------------------------------------- evaluation seeds
BEHAVIOR_SEED = 80_000          # natural (s0) games of the behaviour diagnostics
SCENARIO_SEED = 80_100          # initialized scenarios (freeze, power gap)
CF_STATE_SEED = 80_200          # counterfactual state sampling
CF_ROLL_SEED = 80_300           # counterfactual rollouts (paired streams)
REF_SEED = 80_400               # games against the reference suite
TRACE_SEED = 80_500             # viewer recordings
XPLAY_SEED = 80_600             # cross-play and historical-snapshot games
SMOKE_EVAL_SEED = 80_700        # smoke-test evaluation pipeline


# ---------------------------------------------------------------------------------------------- compatibility
@dataclass
class Compatibility:
    """How a checkpoint relates to the environment it is about to play."""
    path: str
    obs_size: Tuple[int, int]                  # (checkpoint, environment)
    n_actions: Tuple[int, int]
    trained_on: Optional[str]                  # named preset of the checkpoint's configuration
    env: Optional[str]                         # named preset of the environment
    differing: Dict[str, Tuple[object, object]] = field(default_factory=dict)

    @property
    def same_shape(self) -> bool:
        return self.obs_size[0] == self.obs_size[1] and self.n_actions[0] == self.n_actions[1]

    @property
    def same_config(self) -> bool:
        return not self.differing

    def to_dict(self) -> dict:
        return {"path": self.path, "obs_size": list(self.obs_size), "n_actions": list(self.n_actions),
                "trained_on": self.trained_on, "env": self.env, "same_shape": self.same_shape,
                "same_config": self.same_config, "differing": {k: [_js(a), _js(b)] for k, (a, b) in self.differing.items()}}


def _js(v):
    return list(v) if isinstance(v, tuple) else v


def compatibility(path: str, ckpt_cfg: GameConfig, spec: dict, env_cfg_: GameConfig) -> Compatibility:
    from ....env.observation import n_actions, obs_size
    a, b = ckpt_cfg.to_dict(), env_cfg_.to_dict()
    diff = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    return Compatibility(path, (int(spec["obs_size"]), obs_size(env_cfg_)), (int(spec["n_actions"]), n_actions(env_cfg_)),
                         preset_name(ckpt_cfg), preset_name(env_cfg_), diff)


def load_policy(path: str, cfg: GameConfig, cross_env: bool = False, name: Optional[str] = None,
                greedy: bool = False):
    """A checkpoint as a frozen `NetAgent` for environment `cfg`, after the compatibility guard:

      * the weights digest is verified (`load_checkpoint`);
      * a different observation length or action count is ALWAYS refused (e.g. a 46-output M4.3b checkpoint on the
        61-code four-action space): its outputs would be silently re-indexed;
      * a checkpoint trained on a different configuration with the same shapes (e.g. `m4_four_actions()` against
        `m4_four_actions_econ()`, whose absence entry has another scale) is refused unless `cross_env=True`, a
        deliberate cross-environment evaluation; the differing fields are attached to the agent (`agent.compat`).
    """
    from ....rl import NetAgent, load_checkpoint
    net, ckpt_cfg, meta = load_checkpoint(path)
    comp = compatibility(path, ckpt_cfg, net.spec(), cfg)
    if not comp.same_shape:
        raise ValueError(f"{path}: the network has {comp.obs_size[0]} inputs and {comp.n_actions[0]} action outputs, the "
                         f"environment ({comp.env or 'unnamed configuration'}) has {comp.obs_size[1]} and "
                         f"{comp.n_actions[1]}; the checkpoint belongs to another action / observation encoding "
                         f"(trained on {comp.trained_on or 'an unnamed configuration'}) and cannot play here")
    if not comp.same_config and not cross_env:
        raise ValueError(f"{path}: trained on {comp.trained_on or 'an unnamed configuration'}, evaluated on "
                         f"{comp.env or 'an unnamed configuration'}; differing fields {sorted(comp.differing)}. Pass "
                         "cross_env=True for a deliberate cross-environment evaluation")
    agent = NetAgent(net, name or meta.get("name") or os.path.splitext(os.path.basename(path))[0], greedy=greedy,
                     copy_net=False)
    agent.compat = comp
    agent.meta = meta
    return agent


# ---------------------------------------------------------------------------------------------- agent specifications
def agent_from_spec(spec: str, cfg: GameConfig, cross_env: bool = False):
    """An agent by a picklable specification string, checked against `cfg`:

        final:<phase>:<env>:<seed>              final network of an M4.4 run (phase smoke | main)
        snap:<phase>:<env>:<seed>:<transitions> a snapshot of an M4.4 run (historical opponent)
        initial:<env>:<seed>                    the untrained network of that training seed (bit-identical init)
        pilot:<base|candidate>:<seed>           an M4.3c pilot final (base = four, candidate = econ)
        ref:<name>                              a scripted reference (references.reference_suite)
        <path>.pt                               any checkpoint
    """
    parts = spec.split(":")
    kind = parts[0]
    if kind == "final":
        return load_policy(final_path(parts[1], parts[2], int(parts[3])), cfg, cross_env, name=spec)
    if kind == "snap":
        p = dict(snapshot_paths(parts[1], parts[2], int(parts[3])))[int(parts[4])]
        return load_policy(p, cfg, cross_env, name=spec)
    if kind == "initial":
        import torch
        from ....env import LaneEnv
        from ....rl import NetAgent, PolicyValueNet
        if env_cfg(parts[1]) != cfg and not cross_env:
            raise ValueError(f"{spec}: an initial network of environment {parts[1]!r} on another configuration")
        torch.manual_seed(int(parts[2]))                   # exactly train_shared's initialization for that seed
        env = LaneEnv(cfg)
        return NetAgent(PolicyValueNet(env.obs_size, env.n_actions, HIDDEN), spec)
    if kind == "pilot":
        return load_policy(pilot_path(parts[1], int(parts[2])), cfg, cross_env, name=spec)
    if kind == "ref":
        from .references import reference_suite
        return reference_suite()[parts[1]]
    if spec.endswith(".pt"):
        return load_policy(spec, cfg, cross_env)
    raise ValueError(f"unknown agent specification {spec!r}")


def spec_env(spec: str) -> Optional[str]:
    """The environment a network specification was trained on (None for references and plain paths)."""
    parts = spec.split(":")
    if parts[0] in ("final", "snap"):
        return parts[2]
    if parts[0] == "initial":
        return parts[1]
    if parts[0] == "pilot":
        return PILOT_ENV[parts[1]]
    return None


# ---------------------------------------------------------------------------------------------- statistics and I/O
def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and the 1.96 x standard-error half-width (nan for fewer than two values)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    if len(v) < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(Z95 * v.std(ddof=1) / np.sqrt(len(v)))


def fmt_ci(m: float, h: float, d: int = 2) -> str:
    if not np.isfinite(m):
        return "n/a"
    if not np.isfinite(h):
        return f"{m:+.{d}f}"
    return f"{m:+.{d}f} ± {h:.{d}f}"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:                                      # pragma: no cover - outside a checkout
        return "unknown"


def git_dirty() -> Optional[bool]:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:                                      # pragma: no cover
        return None


def meta(**kw) -> dict:
    import sys
    try:
        import torch
        tv = torch.__version__
    except Exception:                                      # pragma: no cover
        tv = None
    return {"commit": git_commit(), "working_tree_dirty": git_dirty(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version.split()[0], "torch": tv, "numpy": np.__version__, "study": "M4.4a", **kw}


def out_path(name: str, ext: str = "json", out: str = OUT_A) -> str:
    return os.path.join(out, f"{name}.{ext}")


GZIP_BYTES = 1_000_000


def save_json(name: str, data: dict, out: str = OUT_A) -> str:
    """<out>/<name>.json, gzipped (plain file removed) when it is >= 1 MB (the M4.3b artifact rule)."""
    import gzip
    import shutil
    os.makedirs(out, exist_ok=True)
    p = out_path(name, "json", out)
    dump_json(data, p)
    if os.path.getsize(p) >= GZIP_BYTES:
        with open(p, "rb") as src, gzip.open(p + ".gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(p)
        p += ".gz"
    elif os.path.exists(p + ".gz"):
        os.remove(p + ".gz")
    print(f"wrote {p}", flush=True)
    return p


def load_json(name: str, out: str = OUT_A) -> Optional[dict]:
    import gzip
    p = out_path(name, "json", out)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(p + ".gz"):
        with gzip.open(p + ".gz", "rt", encoding="utf-8") as f:
            return json.load(f)
    return None


def write_md(name: str, text: str, out: str = OUT_A) -> str:
    os.makedirs(out, exist_ok=True)
    p = out_path(name, "md", out)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    print(f"wrote {p}", flush=True)
    return p


def run_jobs(fn, jobs: Sequence, workers: int, label: str = "") -> list:
    from ..m43b.common import run_jobs as _rj
    return _rj(fn, jobs, workers, label)


__all__ = ["NAME", "OUT_A", "OUT_B", "CKPT", "PHASES", "EnvSpec", "ENVS", "ENV_ORDER", "env_cfg", "preset_name",
           "env_key", "SEEDS", "BUDGET", "SMOKE_BUDGET", "EXT_BUDGET", "N_ENVS", "ENT_COEF", "CURRENT_FRACTION", "SNAP_EVERY", "POOL",
           "HIDDEN", "ppo_config", "shared_config", "training_spec", "run_name", "run_dir", "final_path",
           "snapshot_paths", "history", "PILOT_DIR", "PILOT_ENV", "pilot_path", "Compatibility", "compatibility",
           "load_policy", "agent_from_spec", "spec_env", "mean_ci", "fmt_ci", "meta", "save_json", "load_json",
           "write_md", "run_jobs", "BEHAVIOR_SEED", "SCENARIO_SEED", "CF_STATE_SEED", "CF_ROLL_SEED", "REF_SEED",
           "TRACE_SEED", "XPLAY_SEED", "SMOKE_EVAL_SEED"]
