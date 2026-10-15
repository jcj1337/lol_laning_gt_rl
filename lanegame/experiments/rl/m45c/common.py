"""Shared definitions of M4.5c: the environment, the training settings, the transition accounting, run paths, seeds,
agent specifications (with the checkpoint compatibility guard and the labelled out-of-distribution adapter for M4.4
networks) and I/O.

Environment. `GameConfig.m45_calibrated()` (M4.5b), FROZEN for this phase: 90-entry observation, 61 action codes (27
ever legal, Ward = code 30), continuing time, gamma 0.99, the zero-sum discounted gold lead as reward. Nothing in this
package changes it; `env_cfg()` returns the preset and the tests check it equals the preset field by field.

Training settings: the M4.4b settings (the M4.3b `ent03` condition) unchanged, on the new 90-input network trained from
scratch: `lanegame.rl.train_shared` with `PPOConfig.continuing` (gamma 0.99, GAE lambda 0.95, segments of 128 turns,
games of 512 turns with seeded staggered first games, reward scale 0.1, value clip 0.5, Adam 3e-4 annealed linearly to
0, 4 epochs, minibatch 256, clip 0.2, entropy 0.03, max gradient norm 0.5, 16 environments, 2 x 128 tanh MLP), 100%
current-policy games (both seats trained), snapshots every 25,600 agent transitions (historical evaluation opponents,
not training opponents).

Transition accounting (one convention everywhere in M4.5c, the trainer's own):

    joint environment step   one simultaneous move of both seats in one environment (history: env_turns)
    agent transition         one (observation, action, reward) of one seat played by the learning network; in 100%
                             current-policy self-play every joint step yields exactly 2 (history: agent_transitions)
    PPO sample               an agent transition in the PPO batch: all of them (the value loss uses all; the policy and
                             entropy terms use the "choice" samples with more than one legal action)
    budget / total_steps     PPOConfig.total_steps and every "transitions" figure of M4.4 and M4.5c = AGENT transitions

    one update = 16 environments x 128 joint steps x 2 seats = 4,096 agent transitions = 2,048 joint steps
               = 4 epochs x 16 minibatches of 256 = 64 gradient steps (each sample used 4 times)

The pilot (`PILOT_*`): seed 100 (disjoint from the campaign seeds 0, 1, 2, ... so the campaign's initializations are
untouched by the pilot), the learning-rate schedule of a 4,096,000-transition run (M4.4b's extension budget: the same
learning rate as those runs at every update), stopped at 1,024,000 agent transitions (= 512,000 joint steps = 250
updates; `train_shared(stop_at=...)`), so the end of the pilot is not flattened by an annealed-to-zero learning rate.

Seeds. Evaluation streams of the pilot derive from roots 92,000-92,999; 93,000-94,999 are RESERVED for the full
campaign's evaluation (never used by the pilot). Disjoint from every earlier study (10,000-81,999, 85,000-88,499,
90,000-91,999).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....config import GameConfig

NAME = "m45c"
OUT = os.path.join("results", "m45c")
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
PHASES = ("smoke", "pilot", "bench", "full")
PRESET = "m45_calibrated"


def env_cfg() -> GameConfig:
    cfg = GameConfig.m45_calibrated()
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------------------------- training settings
N_ENVS = 16
SEGMENT = 128
TRANSITIONS_PER_UPDATE = N_ENVS * SEGMENT * 2            # 4,096 agent transitions (2,048 joint steps)
ENT_COEF = 0.03
CURRENT_FRACTION = 1.0
SNAP_EVERY = 25_600
POOL = 8
HIDDEN = (128, 128)

PILOT_SEED = 100
PILOT_SCHEDULE = 4_096_000                               # learning-rate schedule horizon (M4.4b extension budget)
PILOT_STOP = 1_024_000                                   # the pilot's budget (agent transitions): 250 updates
SMOKE_SEED = 100
SMOKE_STEPS = 40_960                                     # 10 updates
CAMPAIGN_SEEDS = tuple(range(10))                        # candidates for the full campaign (none trained here)


def snapshot_schedule(stop: int, every: int = SNAP_EVERY, per_update: int = TRANSITIONS_PER_UPDATE) -> List[int]:
    """Transition counts of the snapshots train_shared takes up to `stop`: the first update that reaches each multiple
    of `every` (28,672, 53,248, 77,824, 102,400, ... for 25,600 and 4,096)."""
    out, nxt, n = [], every, 0
    while n < stop:
        n += per_update
        if n >= nxt:
            out.append(n)
            while nxt <= n:
                nxt += every
    return out


def ppo_config(seed: int, total_steps: int, **kw):
    """The PPO configuration of an M4.5c run: M4.4b's settings with the given seed and schedule horizon."""
    from ....rl import PPOConfig
    base = dict(total_steps=total_steps, n_envs=N_ENVS, seed=seed, torch_threads=1, ent_coef=ENT_COEF, hidden=HIDDEN)
    base.update(kw)
    return PPOConfig.continuing(env_cfg(), **base)


def shared_config():
    from ....rl import SharedConfig
    return SharedConfig(current_fraction=CURRENT_FRACTION, snapshot_every=SNAP_EVERY, pool_size=POOL)


def training_spec() -> dict:
    """The training specification (JSON) recorded with every run and in the report."""
    p = ppo_config(0, PILOT_SCHEDULE).to_dict()
    for k in ("seed", "total_steps"):
        p.pop(k)
    from ....env import LaneEnv
    env = LaneEnv(env_cfg())
    return {"environment": f"GameConfig.{PRESET}()", "game_config": env_cfg().to_dict(), "ppo": p,
            "shared": shared_config().to_dict(),
            "network": {"class": "PolicyValueNet", "obs_size": env.obs_size, "n_actions": env.n_actions,
                        "hidden": list(HIDDEN), "activation": "tanh", "initialization": "from scratch (torch.manual_seed)"},
            "accounting": {"transitions_per_update": TRANSITIONS_PER_UPDATE,
                           "joint_steps_per_update": TRANSITIONS_PER_UPDATE // 2,
                           "gradient_steps_per_update": p["epochs"] * (TRANSITIONS_PER_UPDATE // p["minibatch_size"]),
                           "convention": "budgets and 'transitions' are AGENT transitions (2 per joint step)"},
            "pilot": {"seed": PILOT_SEED, "schedule_total_steps": PILOT_SCHEDULE, "stop_at": PILOT_STOP,
                      "updates": PILOT_STOP // TRANSITIONS_PER_UPDATE, "joint_steps": PILOT_STOP // 2}}


# ---------------------------------------------------------------------------------------------- runs and files
def run_name(seed: int, tag: str = "") -> str:
    return f"s{seed}{tag}"


def run_dir(phase: str, seed: int, tag: str = "") -> str:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; choose from {PHASES}")
    return os.path.join(CKPT, phase, run_name(seed, tag))


def final_path(phase: str, seed: int, tag: str = "") -> str:
    return os.path.join(run_dir(phase, seed, tag), "final.pt")


def snapshot_paths(phase: str, seed: int, tag: str = "") -> List[Tuple[int, str]]:
    """[(agent transitions, path)] of a run's snapshots, in order."""
    d = os.path.join(run_dir(phase, seed, tag), "snapshots")
    if not os.path.isdir(d):
        return []
    return [(int(f[5:-3]), os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith("snap_") and f.endswith(".pt")]


def history(phase: str, seed: int, tag: str = "") -> List[dict]:
    with open(os.path.join(run_dir(phase, seed, tag), "history.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------------------------- evaluation seeds
CURVE_SEED = 92_000             # learning curves: snapshots vs the fixed suite
FINAL_SEED = 92_100             # headline: final (and mid-run) network vs the fixed suite
HIST_SEED = 92_200              # final vs its own snapshots and its untrained network
BEHAVIOR_SEED = 92_300          # behaviour records (self-play and vs references)
LEDGER_SEEDS = tuple(range(92_400, 92_500))       # integer seeds of the M4.5b ledger engine games
SMOKE_EVAL_SEED = 92_900        # smoke-test evaluation pipeline
FULL_RESERVED = (93_000, 94_999)                  # the full campaign's evaluation (unused by the pilot)


# ---------------------------------------------------------------------------------------------- agents
class OODNetAgent:
    """An M4.4 network (88 inputs) playing an M4.5 environment (90 inputs) through an adapter that drops the two W
    entries of the observation (`m4_feature_names`: own_champ_W, enemy_champ_W), as in M4.5b's learned reference.
    OUT OF DISTRIBUTION: the network never saw minion targeting, the reach ball, XP range 2 or tick movement, and code
    30 (its siege-Vision code) is the Ward action here. Labelled in every table that uses it."""

    ood = True

    def __init__(self, agent, cfg: GameConfig, name: str):
        from ....env.observation_m4 import m4_feature_names
        self.net, self.name = agent.net, name
        self.drop = [i for i, n in enumerate(m4_feature_names(cfg)) if n.endswith("champ_W")]
        if len(m4_feature_names(cfg)) - len(self.drop) != agent.net.obs_size:
            raise ValueError(f"{name}: the adapter does not map {len(m4_feature_names(cfg))} entries onto "
                             f"{agent.net.obs_size} inputs")
        self.source = getattr(agent, "meta", {})

    def probabilities(self, obs) -> np.ndarray:
        import torch
        v = np.delete(obs.vector, self.drop)
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(v, dtype=torch.float32).unsqueeze(0))
            logits = logits[0].double()
            logits = torch.where(torch.as_tensor(obs.mask), logits, torch.full_like(logits, -np.inf))
            return torch.softmax(logits, dim=-1).numpy()

    def distribution(self, obs) -> np.ndarray:
        p = self.probabilities(obs)[list(obs.legal)]
        return p / p.sum()

    def act(self, obs, rng: Optional[np.random.Generator] = None) -> int:
        from ....env.agents import sample
        p = self.distribution(obs)
        return obs.legal[int(np.argmax(p))] if rng is None else obs.legal[sample(rng, p)]


def initial_agent(seed: int, cfg: Optional[GameConfig] = None):
    """The untrained network of a training seed (exactly train_shared's initialization)."""
    import torch
    from ....env import LaneEnv
    from ....rl import NetAgent, PolicyValueNet
    torch.manual_seed(int(seed))
    env = LaneEnv(cfg or env_cfg())
    return NetAgent(PolicyValueNet(env.obs_size, env.n_actions, HIDDEN), f"initial:{seed}")


def agent_from_spec(spec: str, cfg: Optional[GameConfig] = None):
    """An agent by a picklable specification string, checked against the M4.5c environment:

        final:<phase>:<seed>[:<tag>]            final network of an M4.5c run
        snap:<phase>:<seed>:<transitions>       a snapshot of an M4.5c run
        initial:<seed>                          the untrained network of that training seed
        ref:<name>                              a fixed scripted opponent (references.SUITE)
        ood:m44:<env>:<phase>:<seed>            an M4.4 final through the W-dropping adapter (OUT OF DISTRIBUTION)
        <path>.pt                               any checkpoint (the guard applies)
    """
    from ..m44.common import load_policy
    cfg = cfg or env_cfg()
    parts = spec.split(":")
    kind = parts[0]
    if kind == "final":
        tag = parts[3] if len(parts) > 3 else ""
        return load_policy(final_path(parts[1], int(parts[2]), tag), cfg, name=spec)
    if kind == "snap":
        p = dict(snapshot_paths(parts[1], int(parts[2])))[int(parts[3])]
        return load_policy(p, cfg, name=spec)
    if kind == "initial":
        a = initial_agent(int(parts[1]), cfg)
        a.name = spec
        return a
    if kind == "ref":
        from .references import reference
        return reference(parts[1])
    if kind == "ood":
        from ..m44 import common as M44
        if parts[1] != "m44":
            raise ValueError(f"unknown out-of-distribution source {parts[1]!r}")
        src = M44.agent_from_spec(f"final:{parts[3]}:{parts[2]}:{parts[4]}", M44.env_cfg(parts[2]))
        return OODNetAgent(src, cfg, spec)
    if spec.endswith(".pt"):
        return load_policy(spec, cfg)
    raise ValueError(f"unknown agent specification {spec!r}")


# ---------------------------------------------------------------------------------------------- statistics and I/O
def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    from ..m44.common import mean_ci as _m
    return _m(values)


def fmt_ci(m: float, h: float, d: int = 2) -> str:
    from ..m44.common import fmt_ci as _f
    return _f(m, h, d)


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
    try:
        import torch
        tv = torch.__version__
    except Exception:                                      # pragma: no cover
        tv = None
    return {"study": "M4.5c", "commit": git_commit(), "working_tree_dirty": git_dirty(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0], "torch": tv,
            "numpy": np.__version__, "environment": f"GameConfig.{PRESET}()", **kw}


def save_json(name: str, data: dict, out: str = OUT) -> str:
    from ..m44.common import save_json as _s
    return _s(name, data, out)


def load_json(name: str, out: str = OUT) -> Optional[dict]:
    from ..m44.common import load_json as _l
    return _l(name, out)


def write_md(name: str, text: str, out: str = OUT) -> str:
    from ..m44.common import write_md as _w
    return _w(name, text, out)


def run_jobs(fn, jobs: Sequence, workers: int, label: str = "") -> list:
    from ..m43b.common import run_jobs as _rj
    return _rj(fn, jobs, workers, label)


WORKERS = max(1, (os.cpu_count() or 2) - 1)

__all__ = ["NAME", "OUT", "CKPT", "PHASES", "PRESET", "env_cfg", "N_ENVS", "SEGMENT", "TRANSITIONS_PER_UPDATE",
           "ENT_COEF", "CURRENT_FRACTION", "SNAP_EVERY", "POOL", "HIDDEN", "PILOT_SEED", "PILOT_SCHEDULE", "PILOT_STOP",
           "SMOKE_SEED", "SMOKE_STEPS", "CAMPAIGN_SEEDS", "snapshot_schedule", "ppo_config", "shared_config", "training_spec", "run_name",
           "run_dir", "final_path", "snapshot_paths", "history", "CURVE_SEED", "FINAL_SEED", "HIST_SEED",
           "BEHAVIOR_SEED", "LEDGER_SEEDS", "SMOKE_EVAL_SEED", "FULL_RESERVED", "OODNetAgent", "initial_agent",
           "agent_from_spec", "mean_ci", "fmt_ci", "meta", "save_json", "load_json", "write_md", "run_jobs", "WORKERS"]
