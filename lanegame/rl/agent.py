"""A trained (or frozen) network as an environment `Agent`, and checkpoint files.

`NetAgent` exposes `distribution(obs)` (probabilities over `obs.legal`, exactly the network's masked softmax)
and `act(obs, rng)` (a draw from it with the caller's NumPy generator, or the most probable legal action when
`greedy=True`). Because its decision depends on the observation only, a NetAgent on a legacy configuration is a
Markov policy of the exact game: `lanegame.env.as_policy(agent, cfg)` gives its exact value and exploitability.
A NetAgent never trains: its network is a frozen copy (no gradients).

Checkpoints (`save_checkpoint` / `load_checkpoint`) are torch files holding the network weights, its shape, the
game configuration and free-form metadata (training configuration, step, seed, opponent mix, ...).
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import numpy as np
import torch

from ..config import GameConfig
from ..env.agents import sample
from .network import PolicyValueNet


class NetAgent:
    def __init__(self, net: PolicyValueNet, name: str = "ppo", greedy: bool = False, copy_net: bool = True):
        self.net = copy.deepcopy(net) if copy_net else net
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.name, self.greedy = name, greedy

    def probabilities(self, obs) -> np.ndarray:
        """Full-length probability vector over the action space (0 on illegal actions)."""
        with torch.no_grad():
            x = torch.as_tensor(obs.vector, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.net(x)
            logits = logits[0].double()
            mask = torch.as_tensor(obs.mask)
            logits = torch.where(mask, logits, torch.full_like(logits, -np.inf))
            return torch.softmax(logits, dim=-1).numpy()

    def distribution(self, obs) -> np.ndarray:
        p = self.probabilities(obs)[list(obs.legal)]
        if self.greedy:
            q = np.zeros(len(p))
            q[int(np.argmax(p))] = 1.0
            return q
        return p / p.sum()

    def act(self, obs, rng: Optional[np.random.Generator] = None) -> int:
        p = self.distribution(obs)
        if self.greedy or rng is None:
            return obs.legal[int(np.argmax(p))]
        return obs.legal[sample(rng, p)]


def save_checkpoint(path: str, net: PolicyValueNet, game_cfg: GameConfig, metadata: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"format": "lanegame.rl.checkpoint", "version": 1, "spec": net.spec(),
                "state_dict": net.state_dict(), "game_config": game_cfg.to_dict(), "metadata": metadata or {}}, path)


def load_checkpoint(path: str):
    """(network, game configuration, metadata)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    assert d.get("format") == "lanegame.rl.checkpoint", f"{path} is not a lanegame RL checkpoint"
    spec = d["spec"]
    net = PolicyValueNet(spec["obs_size"], spec["n_actions"], spec["hidden"])
    net.load_state_dict(d["state_dict"])
    return net, GameConfig.from_dict(d["game_config"]), d["metadata"]


def load_agent(path: str, name: Optional[str] = None, greedy: bool = False) -> NetAgent:
    net, _, meta = load_checkpoint(path)
    return NetAgent(net, name or meta.get("name") or os.path.splitext(os.path.basename(path))[0], greedy=greedy,
                    copy_net=False)
