"""Policy/value network and the masked categorical distribution.

The network is a small MLP over the fixed-length own-perspective observation (lanegame.env.observation):
a shared trunk of two hidden layers (128 tanh units by default) and two linear heads, policy logits over the
fixed action space and a scalar state value. Illegal actions are masked by replacing their logits with
MASK_VALUE (-1e9): their probability is exactly 0 in float32 (exp underflows), so they are never sampled and
contribute exactly 0 to the entropy; the probabilities of the legal actions are a softmax over the legal
logits only. Orthogonal initialization as in common PPO implementations (policy head gain 0.01: an almost
uniform initial policy over the legal actions; value head gain 1).
"""

from __future__ import annotations

from typing import Optional, Sequence

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
