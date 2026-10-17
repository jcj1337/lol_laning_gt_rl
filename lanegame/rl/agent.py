"""A trained (or frozen) network as an environment `Agent`, and checkpoint files.

`NetAgent` exposes `distribution(obs)` (probabilities over `obs.legal`, exactly the network's masked softmax)
and `act(obs, rng)` (a draw from it with the caller's NumPy generator, or the most probable legal action when
`greedy=True`). Because its decision depends on the observation only, a NetAgent on a legacy configuration is a
Markov policy of the exact game: `lanegame.env.as_policy(agent, cfg)` gives its exact value and exploitability.
A NetAgent never trains: its network is a frozen copy (no gradients).

Checkpoints (`save_checkpoint` / `load_checkpoint`) are torch files holding the network weights, its shape, the
game configuration and free-form metadata (training configuration, step, seed, opponent mix, ...). Since format
version 2 they also carry a SHA-256 digest of the weights (`digest`, also returned by `weights_digest`), which
`load_checkpoint` verifies (a mismatch raises ValueError: the file was altered or truncated), and the library
versions and git commit that wrote them (`provenance`). Version-1 files load unchanged. The network class is rebuilt
from the stored spec (`network.build_network`): a `PolicyValueNet` unless the spec names a separate critic.
"""

from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import sys
from typing import Optional

import numpy as np
import torch

from ..config import GameConfig
from ..env.agents import sample
from .network import PolicyValueNet, build_network

CHECKPOINT_FORMAT = "lanegame.rl.checkpoint"
CHECKPOINT_VERSION = 2


class NetAgent:
    def __init__(self, net: PolicyValueNet, name: str = "ppo", greedy: bool = False, copy_net: bool = True):
        self.net = copy.deepcopy(net) if copy_net else net
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.name, self.greedy = name, greedy

    def probabilities(self, obs) -> np.ndarray:
        """Full-length probability vector over the action space (0 on illegal actions)."""
        if len(obs.mask) != self.net.n_actions:
            raise ValueError(f"{self.name}: the network has {self.net.n_actions} action outputs but this environment's "
                             f"action space has {len(obs.mask)} codes (a checkpoint of another action encoding, e.g. a "
                             f"46-code checkpoint on the 61-code four-action preset)")
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


def weights_digest(state_dict) -> str:
    """SHA-256 over the parameter names and their float32/int bytes, in key order: a stable identity of a network's
    weights (equal digests <=> bit-identical weights)."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k]
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:                                      # pragma: no cover - outside a checkout
        return None


def provenance() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "git_commit": _git_commit()}


def save_checkpoint(path: str, net: PolicyValueNet, game_cfg: GameConfig, metadata: Optional[dict] = None,
                    exist_ok: bool = True) -> str:
    """Write the checkpoint; returns the weights digest. `exist_ok=False` refuses to replace an existing file
    (FileExistsError) instead of overwriting it."""
    if not exist_ok and os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite the checkpoint {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    digest = weights_digest(sd)
    torch.save({"format": CHECKPOINT_FORMAT, "version": CHECKPOINT_VERSION, "spec": net.spec(), "state_dict": sd,
                "digest": digest, "provenance": provenance(), "game_config": game_cfg.to_dict(),
                "metadata": metadata or {}}, path)
    return digest


def load_checkpoint(path: str, verify: bool = True):
    """(network, game configuration, metadata). With `verify` (default) a version-2 file whose weights do not match
    its digest raises ValueError."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a lanegame RL checkpoint")
    if verify and "digest" in d and weights_digest(d["state_dict"]) != d["digest"]:
        raise ValueError(f"{path}: the weights do not match the checkpoint's digest (altered or truncated file)")
    net = build_network(d["spec"])
    net.load_state_dict(d["state_dict"])
    return net, GameConfig.from_dict(d["game_config"]), d["metadata"]


def checkpoint_info(path: str) -> dict:
    """Format version, digest, provenance and metadata of a checkpoint without building the network."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {"version": d.get("version"), "digest": d.get("digest"), "provenance": d.get("provenance"),
            "spec": d.get("spec"), "metadata": d.get("metadata")}


def load_agent(path: str, name: Optional[str] = None, greedy: bool = False) -> NetAgent:
    net, _, meta = load_checkpoint(path)
    return NetAgent(net, name or meta.get("name") or os.path.splitext(os.path.basename(path))[0], greedy=greedy,
                    copy_net=False)
