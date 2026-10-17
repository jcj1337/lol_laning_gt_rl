"""Policy/value network and the masked categorical distribution.

The network is a small MLP over the fixed-length own-perspective observation (lanegame.env.observation):
a shared trunk of two hidden layers (128 tanh units by default) and two linear heads, policy logits over the
fixed action space and a scalar state value. Illegal actions are masked by replacing their logits with
MASK_VALUE (-1e9): their probability is exactly 0 in float32 (exp underflows), so they are never sampled and
contribute exactly 0 to the entropy; the probabilities of the legal actions are a softmax over the legal
logits only. Orthogonal initialization as in common PPO implementations (policy head gain 0.01: an almost
uniform initial policy over the legal actions; value head gain 1).

`SeparateCriticNet` (M4.6f, opt-in) has the same interface and the same actor (`trunk` and `policy_head`, under the
same parameter names), but the value head reads its own critic trunk (`critic_trunk`) instead of the actor's: actor and
critic share no parameters, so the value loss never reaches the actor and the policy loss never reaches the critic.
`SeparateCriticNet.from_shared(net)` copies a trained `PolicyValueNet` into both trunks (separate storage): its action
probabilities and values equal the source network's exactly. `build_network(spec)` rebuilds either class from its
`spec()` (a spec without a "critic" entry is a `PolicyValueNet`, so earlier checkpoints load unchanged).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

MASK_VALUE = -1e9


def _layer(n_in: int, n_out: int, gain: float) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class PolicyValueNet(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        self.obs_size, self.n_actions, self.hidden = int(obs_size), int(n_actions), tuple(int(h) for h in hidden)
        layers, n = [], self.obs_size
        for h in self.hidden:
            layers += [_layer(n, h, np.sqrt(2.0)), nn.Tanh()]
            n = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = _layer(n, self.n_actions, 0.01)
        self.value_head = _layer(n, 1, 1.0)

    def forward(self, obs: torch.Tensor):
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(obs)).squeeze(-1)

    def spec(self) -> dict:
        return {"obs_size": self.obs_size, "n_actions": self.n_actions, "hidden": list(self.hidden)}

    def value_features(self, obs: torch.Tensor) -> torch.Tensor:
        """The representation the value head reads (here the shared trunk's last layer)."""
        return self.trunk(obs)


def _trunk(n_in: int, hidden: Sequence[int]) -> Tuple[nn.Sequential, int]:
    layers, n = [], n_in
    for h in hidden:
        layers += [_layer(n, h, np.sqrt(2.0)), nn.Tanh()]
        n = h
    return nn.Sequential(*layers), n


class SeparateCriticNet(nn.Module):
    """Actor and critic with separate trunks (module docstring). The actor is `trunk` -> `policy_head` (the parameter
    names of `PolicyValueNet`); the critic is `critic_trunk` -> `value_head`."""

    def __init__(self, obs_size: int, n_actions: int, hidden: Sequence[int] = (128, 128),
                 critic_hidden: Optional[Sequence[int]] = None):
        super().__init__()
        self.obs_size, self.n_actions, self.hidden = int(obs_size), int(n_actions), tuple(int(h) for h in hidden)
        self.critic_hidden = tuple(int(h) for h in (critic_hidden if critic_hidden is not None else hidden))
        self.trunk, n = _trunk(self.obs_size, self.hidden)
        self.policy_head = _layer(n, self.n_actions, 0.01)
        self.critic_trunk, nc = _trunk(self.obs_size, self.critic_hidden)
        self.value_head = _layer(nc, 1, 1.0)

    def forward(self, obs: torch.Tensor):
        return self.policy_head(self.trunk(obs)), self.value_head(self.critic_trunk(obs)).squeeze(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.critic_trunk(obs)).squeeze(-1)

    def value_features(self, obs: torch.Tensor) -> torch.Tensor:
        """The representation the value head reads (the critic trunk's last layer)."""
        return self.critic_trunk(obs)

    def actor_parameters(self):
        return list(self.trunk.parameters()) + list(self.policy_head.parameters())

    def critic_parameters(self):
        return list(self.critic_trunk.parameters()) + list(self.value_head.parameters())

    def spec(self) -> dict:
        return {"obs_size": self.obs_size, "n_actions": self.n_actions, "hidden": list(self.hidden),
                "critic": "separate", "critic_hidden": list(self.critic_hidden)}

    @classmethod
    def from_shared(cls, net: "PolicyValueNet") -> "SeparateCriticNet":
        """The actor of `net` and a critic whose trunk is a copy of `net`'s trunk (no shared storage) with `net`'s value
        head: exactly the same action probabilities and values. The global torch RNG is left untouched."""
        if not isinstance(net, PolicyValueNet):
            raise TypeError("from_shared needs a PolicyValueNet (shared actor-critic trunk)")
        with torch.random.fork_rng(devices=[]):
            out = cls(net.obs_size, net.n_actions, net.hidden)
        with torch.no_grad():
            out.trunk.load_state_dict(net.trunk.state_dict())
            out.critic_trunk.load_state_dict(net.trunk.state_dict())
            out.policy_head.load_state_dict(net.policy_head.state_dict())
            out.value_head.load_state_dict(net.value_head.state_dict())
        return out


def build_network(spec: dict) -> nn.Module:
    """A freshly initialized network of the class and shape described by `spec` (a network's `spec()`)."""
    kind = spec.get("critic", "shared")
    if kind == "shared":
        return PolicyValueNet(spec["obs_size"], spec["n_actions"], spec["hidden"])
    if kind == "separate":
        return SeparateCriticNet(spec["obs_size"], spec["n_actions"], spec["hidden"], spec.get("critic_hidden"))
    raise ValueError(f"unknown critic kind {kind!r} in a network spec")


class MaskedCategorical:
    """Categorical distribution over the actions allowed by a boolean mask (batch-first)."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every row of the mask needs at least one legal action")
        self.mask = mask
        self.logits = torch.where(mask, logits, torch.full_like(logits, MASK_VALUE))
        self.log_probs = torch.log_softmax(self.logits, dim=-1)
        self.probs = self.log_probs.exp()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.log_probs.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> torch.Tensor:
        return -(torch.where(self.mask, self.probs * self.log_probs, torch.zeros_like(self.probs))).sum(-1)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return torch.multinomial(self.probs, 1, generator=generator).squeeze(-1)

    def mode(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1)
