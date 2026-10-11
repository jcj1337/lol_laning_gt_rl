"""Proximal Policy Optimization with action masking: configuration, generalized advantage estimation and the
clipped update (in the style of CleanRL's single-file PPO).

Episode ends. A step is `terminated` when the game itself ended (the horizon K was reached: nothing follows,
the value of what follows is 0) and `truncated` when the episode was cut short (env.max_turns) although the
game would go on: then the return is bootstrapped with the value of the observation the step produced. In
both cases advantages do not propagate across the episode boundary. For a step that did not end the episode,
the bootstrap value is the value of the next observation (also at the end of a rollout):

    delta_t = r_t + gamma * (1 - terminated_t) * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1},     done_t = terminated_t or truncated_t
    R_t     = A_t + V(s_t)                                            (value target)

The game is undiscounted and finite (K = 32), so gamma defaults to 1; the observation contains the turns
remaining, so the value function can represent the finite horizon.

Steps whose legal set has a single action (an absent champion) carry no policy gradient (log-probability 0
whatever the logits); they are kept for the value function and excluded from the policy and entropy averages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from .network import MaskedCategorical, PolicyValueNet


@dataclass
class PPOConfig:
    total_steps: int = 100_000       # learner environment steps (one per turn per environment)
    n_envs: int = 16
    n_steps: int = 64                # steps per environment per rollout (rollout = n_envs * n_steps)
    epochs: int = 4
    minibatch_size: int = 256
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_clip: float = 2.0             # value-function clipping range (returns are in resources, |R| up to ~15)
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    norm_adv: bool = True
    reward_scale: float = 1.0        # rewards are multiplied by this before learning (reporting is unscaled)
    hidden: tuple = (128, 128)
    seed: int = 0
    torch_threads: int = 1           # one thread: reproducible and fastest for a 2 x 128 MLP
    max_turns: Optional[int] = None  # truncate training episodes (exercises the bootstrap; None = full games)
    lane_cache: int = 20000          # bounded LRU cache of lane steps per training environment
    skill_levels: Optional[tuple] = None   # if set, each training episode draws (skill_a, skill_b) from these
                                           # levels independently (skill is observed, so one policy conditions on it)

    @property
    def batch_size(self) -> int:
        return self.n_envs * self.n_steps

    @property
    def n_updates(self) -> int:
        return max(1, self.total_steps // self.batch_size)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        if self.skill_levels is not None:
            d["skill_levels"] = list(self.skill_levels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PPOConfig":
        d = dict(d)
        if "hidden" in d:
            d["hidden"] = tuple(d["hidden"])
        if d.get("skill_levels") is not None:
            d["skill_levels"] = tuple(d["skill_levels"])
        return cls(**d)


def compute_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, terminated: np.ndarray,
                truncated: np.ndarray, gamma: float, lam: float):
    """Advantages and value targets for time-major arrays of shape (T, N); see the module docstring."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float64)
    last = np.zeros(rewards.shape[1:], dtype=np.float64)
    term = terminated.astype(np.float64)
    done = np.logical_or(terminated, truncated).astype(np.float64)
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * (1.0 - term[t]) * next_values[t] - values[t]
        last = delta + gamma * lam * (1.0 - done[t]) * last
        adv[t] = last
    return adv, adv + values


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    v = np.var(target)
    return float("nan") if v == 0 else float(1.0 - np.var(target - pred) / v)


def ppo_update(net: PolicyValueNet, optimizer: torch.optim.Optimizer, batch: Dict[str, torch.Tensor], cfg: PPOConfig,
               generator: torch.Generator) -> Dict[str, float]:
    """Clipped-surrogate PPO epochs over one rollout. `batch` holds flat tensors: obs, mask, actions, logprobs,
    values, advantages, returns, choice (1.0 where more than one action was legal)."""
    n = batch["obs"].shape[0]
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, cfg.minibatch_size):
            idx = perm[start:start + cfg.minibatch_size]
            logits, v = net(batch["obs"][idx])
            dist = MaskedCategorical(logits, batch["mask"][idx])
            newlogp = dist.log_prob(batch["actions"][idx])
            logratio = newlogp - batch["logprobs"][idx]
            ratio = logratio.exp()
            w = batch["choice"][idx]
            nw = w.sum().clamp(min=1.0)
            adv = batch["advantages"][idx]
            if cfg.norm_adv:
                m = (adv * w).sum() / nw
                sd = (((adv - m) ** 2 * w).sum() / nw).sqrt()
                adv = (adv - m) / (sd + 1e-8)
            pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef))
            pg_loss = (pg * w).sum() / nw
            ret, old_v = batch["returns"][idx], batch["values"][idx]
            if cfg.clip_vloss:
                v_clip = old_v + (v - old_v).clamp(-cfg.vf_clip, cfg.vf_clip)
                v_loss = 0.5 * torch.max((v - ret) ** 2, (v_clip - ret) ** 2).mean()
            else:
                v_loss = 0.5 * ((v - ret) ** 2).mean()
            entropy = (dist.entropy() * w).sum() / nw
            loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                kl = (((ratio - 1) - logratio) * w).sum() / nw
                cf = ((((ratio - 1).abs() > cfg.clip_coef).float()) * w).sum() / nw
            for key, val in (("policy_loss", pg_loss), ("value_loss", v_loss), ("entropy", entropy),
                             ("approx_kl", kl), ("clipfrac", cf)):
                stats[key].append(float(val.detach()))
        if cfg.target_kl is not None and stats["approx_kl"][-1] > cfg.target_kl:
            break
    return {k: float(np.mean(v)) for k, v in stats.items()}
