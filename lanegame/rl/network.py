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

NC1 (opt-in, function-preserving starts for the capacity pilot): `widen_hidden` adds hidden units with zero outgoing
weights; `StrengthFeatureNet` appends M4.6e's six observation-derived strength features to the trunk's input, and
`with_strength_features` starts one from a trained `PolicyValueNet` with zero weights on them. A spec with
"features": "strength6" rebuilds a `StrengthFeatureNet`.
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


def widen_inputs(net: "PolicyValueNet", obs_size: int) -> "PolicyValueNet":
    """A PolicyValueNet with `obs_size` >= net.obs_size inputs whose first net.obs_size inputs carry net's weights and
    whose extra inputs have zero weights (OM2): its outputs on [x, anything] equal net's on x exactly. The source is
    not modified."""
    if type(net) is not PolicyValueNet:
        raise TypeError("widen_inputs supports PolicyValueNet")
    if obs_size < net.obs_size:
        raise ValueError(f"cannot narrow {net.obs_size} inputs to {obs_size}")
    out = PolicyValueNet(obs_size, net.n_actions, net.hidden)
    src = net.state_dict()
    dst = out.state_dict()
    with torch.no_grad():
        for k, v in src.items():
            if k == "trunk.0.weight":
                w = torch.zeros_like(dst[k])
                w[:, :net.obs_size] = v
                dst[k] = w
            else:
                dst[k] = v.clone()
    out.load_state_dict(dst)
    return out


def widen_hidden(net: "PolicyValueNet", hidden: Sequence[int], seed: int = 0) -> "PolicyValueNet":
    """A PolicyValueNet with hidden widths `hidden` >= net.hidden (same depth) that computes net's outputs (NC1). The
    old units keep their weights; each new unit takes its incoming weights from a fresh orthogonal initialization
    (seeded; the global torch RNG is left untouched) and has zero outgoing weights, so it changes no output until
    training moves them. Outputs agree to float rounding (a wider inner dimension can change the BLAS kernel). The
    source is not modified."""
    if type(net) is not PolicyValueNet:
        raise TypeError("widen_hidden supports PolicyValueNet")
    hidden = tuple(int(h) for h in hidden)
    if len(hidden) != len(net.hidden) or any(h < o for h, o in zip(hidden, net.hidden)):
        raise ValueError(f"cannot widen hidden {net.hidden} to {hidden}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        out = PolicyValueNet(net.obs_size, net.n_actions, hidden)
    src = net.state_dict()
    dst = out.state_dict()
    with torch.no_grad():
        for k, v in src.items():
            w = dst[k].clone()
            if v.dim() == 2:
                w[:, v.shape[1]:] = 0.0
                w[:v.shape[0], :v.shape[1]] = v
            else:
                w[:v.shape[0]] = v
            dst[k] = w
    out.load_state_dict(dst)
    return out


STRENGTH_FEATURES = ("log_duel", "hp_own", "hp_enemy", "log_hp_ratio", "log_duel_hp", "present_log_duel")


def strength_features(obs: torch.Tensor, index: Sequence[int]) -> torch.Tensor:
    """M4.6e's six observation-derived strength features (`m46e.features.explicit`), computed in float64 and returned
    as float32. `index` = the observation positions of own / enemy attack power, own / enemy HP fraction and own /
    enemy absence. They add no information: log_duel = 2 (logit(ap_own) - logit(ap_enemy)) is the log duel ratio."""
    x = obs.double()
    ap_o, ap_e, hp_o, hp_e, ab_o, ab_e = (x[..., i] for i in index)

    def logit(p):
        p = p.clamp(1e-9, 1 - 1e-9)
        return torch.log(p / (1.0 - p))

    ld = 2.0 * (logit(ap_o) - logit(ap_e))
    lhr = torch.log(hp_o.clamp(min=0.05)) - torch.log(hp_e.clamp(min=0.05))
    both = ((ab_o == 0) & (ab_e == 0)).double()
    return torch.stack([ld, hp_o, hp_e, lhr, ld + lhr, both * ld], dim=-1).float()


class StrengthFeatureNet(PolicyValueNet):
    """A PolicyValueNet over the observation plus `strength_features` (NC1, opt-in): the policy still takes the
    `obs_size`-entry observation, and the trunk reads obs_size + 6 inputs. `with_strength_features` builds one from a
    trained PolicyValueNet with zero weights on the six features, so it starts as exactly that policy."""

    def __init__(self, obs_size: int, n_actions: int, hidden: Sequence[int] = (128, 128),
                 index: Sequence[int] = ()):
        if len(index) != 6:
            raise ValueError("StrengthFeatureNet needs the 6 observation positions of the strength features")
        super().__init__(int(obs_size) + len(STRENGTH_FEATURES), n_actions, hidden)
        self.obs_size, self.index = int(obs_size), tuple(int(i) for i in index)

    def _inputs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, strength_features(obs, self.index)], dim=-1)

    def forward(self, obs: torch.Tensor):
        return super().forward(self._inputs(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return super().value(self._inputs(obs))

    def value_features(self, obs: torch.Tensor) -> torch.Tensor:
        return super().value_features(self._inputs(obs))

    def spec(self) -> dict:
        return {**super().spec(), "obs_size": self.obs_size, "features": "strength6", "feature_index": list(self.index)}


def with_strength_features(net: "PolicyValueNet", index: Sequence[int]) -> StrengthFeatureNet:
    """`net` plus zero-weight strength-feature inputs: the same outputs (to float rounding). The global torch RNG is
    left untouched; the source is not modified."""
    if type(net) is not PolicyValueNet:
        raise TypeError("with_strength_features supports PolicyValueNet")
    with torch.random.fork_rng(devices=[]):
        out = StrengthFeatureNet(net.obs_size, net.n_actions, net.hidden, index)
        wide = widen_inputs(net, net.obs_size + len(STRENGTH_FEATURES))
    out.load_state_dict(wide.state_dict())
    return out


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
    if spec.get("features") == "strength6" and kind == "shared":
        return StrengthFeatureNet(spec["obs_size"], spec["n_actions"], spec["hidden"], spec["feature_index"])
    if "features" in spec:
        raise ValueError(f"unknown features {spec['features']!r} for critic {kind!r} in a network spec")
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
