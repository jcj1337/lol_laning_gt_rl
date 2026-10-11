"""Self-play against a population of frozen snapshots (roadmap M3b).

The learner never plays against itself while it changes. Instead it plays against opponents drawn, per episode,
from a population:

  * a scripted agent (uniformly from the scripted suite) with probability p_scripted,
  * otherwise a frozen snapshot of the learner: the most recent one with probability p_latest, else one drawn
    uniformly from the pool.

Every `snapshot_every` updates a frozen copy of the learner (a NetAgent: no gradients) joins the pool; the pool
keeps at most `pool_size` snapshots (the oldest is dropped, except the initial snapshot, which is kept as an
anchor if `keep_first`). Snapshots are also written to out_dir/snapshots/ with metadata, so the population can be
re-evaluated later. Opponent draws use the training run's seeded generator, so a run is reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import GameConfig
from .agent import NetAgent, save_checkpoint
from .network import PolicyValueNet
from .ppo import PPOConfig
from .train import OpponentSampler, TrainResult, train_ppo


class SnapshotPool:
    def __init__(self, pool_size: int = 8, keep_first: bool = True):
        self.pool_size, self.keep_first = int(pool_size), keep_first
        self.snapshots: List[NetAgent] = []
        self.steps: List[int] = []

    def add(self, net: PolicyValueNet, step: int, name: Optional[str] = None) -> NetAgent:
        agent = NetAgent(net, name or f"snap@{step}")          # a frozen deep copy
        self.snapshots.append(agent)
        self.steps.append(step)
        while len(self.snapshots) > self.pool_size:
            drop = 1 if (self.keep_first and len(self.snapshots) > 1) else 0
            del self.snapshots[drop]
            del self.steps[drop]
        return agent

    @property
    def latest(self) -> Optional[NetAgent]:
        return self.snapshots[-1] if self.snapshots else None

    def __len__(self) -> int:
        return len(self.snapshots)


class SelfPlaySampler(OpponentSampler):
    def __init__(self, scripted: Sequence, pool: SnapshotPool, p_scripted: float = 0.3, p_latest: float = 0.35):
        self.scripted, self.pool = list(scripted), pool
        self.p_scripted, self.p_latest = p_scripted, p_latest
        self.counts: Dict[str, int] = {}

    def sample(self, rng: np.random.Generator):
        u = rng.random()
        if not self.pool or (self.scripted and u < self.p_scripted):
            a = self.scripted[int(rng.integers(len(self.scripted)))]
        elif u < self.p_scripted + self.p_latest:
            a = self.pool.latest
        else:
            a = self.pool.snapshots[int(rng.integers(len(self.pool)))]
        self.counts[a.name] = self.counts.get(a.name, 0) + 1
        return a

    def describe(self) -> dict:
        return {"p_scripted": self.p_scripted, "p_latest": self.p_latest, "pool_size": self.pool.pool_size,
                "scripted": [a.name for a in self.scripted], "draws": dict(self.counts)}


@dataclass
class SelfPlayResult:
    train: TrainResult
    pool: SnapshotPool
    snapshot_paths: List[str] = field(default_factory=list)


def train_selfplay(game_cfg: GameConfig, cfg: PPOConfig, scripted: Sequence, init_net: Optional[PolicyValueNet] = None,
                   pool_size: int = 8, snapshot_every: int = 10, p_scripted: float = 0.3, p_latest: float = 0.35,
                   out_dir: Optional[str] = None, name: str = "selfplay", eval_fn=None, eval_every: int = 0,
                   verbose: bool = True) -> SelfPlayResult:
    """PPO against a growing pool of frozen snapshots plus the scripted suite. If `init_net` is given (e.g. a
    policy trained against scripted opponents), training continues from it and it is the first snapshot."""
    pool = SnapshotPool(pool_size)
    paths: List[str] = []

    def snapshot(net, step):
        agent = pool.add(net, step)
        if out_dir:
            path = os.path.join(out_dir, "snapshots", f"snap_{step:08d}.pt")
            save_checkpoint(path, net, game_cfg, {"name": agent.name, "step": step, "run": name, "ppo": cfg.to_dict()})
            paths.append(path)

    if init_net is not None:
        snapshot(init_net, 0)
    sampler = SelfPlaySampler(scripted, pool, p_scripted, p_latest)

    def on_update(u, net, entry):
        if u % snapshot_every == 0:
            snapshot(net, entry["steps"])
        entry["pool"] = [a.name for a in pool.snapshots]

    res = train_ppo(game_cfg, cfg, sampler, out_dir=out_dir, net=init_net, on_update=on_update, verbose=verbose,
                    name=name, eval_fn=eval_fn, eval_every=eval_every)
    return SelfPlayResult(res, pool, paths)
