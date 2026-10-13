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

Continuing time (rulesets "m2c" and "m4", M4 design §10.4). The game has no terminal state, so `terminated` is never
set; a training game ends only at its time limit `game_length`, a TRUNCATION bootstrapped with the critic's value of
the true pre-reset observation, and rollouts are fixed-length segments of `segment_length` turns that bootstrap the
value of the current observation when they cut a game. gamma = cfg.gamma (0.99) is part of the game. With an exact
critic the truncated and segmented lambda-return targets have the infinite-horizon expectation; with an imperfect
critic V_hat the one-step target at a boundary state s' errs by exactly gamma (V_hat(s') - V(s')), which propagates
backwards with weight (gamma lambda)^j. Bootstrapping therefore APPROXIMATES the continuation value; it is not
unbiased. `compute_gae` is unchanged: termination drops the successor value, truncation keeps it, and both stop the
lambda-recursion. `PPOConfig.continuing(game_cfg)` is the preset.
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
    expected_gank_reward: bool = False     # legacy configurations only: LaneEnv(gank_reward="expected"), an exact
                                           # variance reduction (same expected returns); see lanegame.env.core
    skill_levels: Optional[tuple] = None   # if set, each training episode draws (skill_a, skill_b) from these
                                           # levels independently (skill is observed, so one policy conditions on it)
    # ---- continuing time (M4 design §10.4); None = the legacy whole-episode rollouts
    segment_length: Optional[int] = None   # rollout length per environment (overrides n_steps when set)
    game_length: Optional[int] = None      # time limit of one training game (a bootstrapped truncation); the first
                                           # game of each environment gets a seeded random length in [1, game_length]
    start_archive: Optional[str] = None    # JSON archive of witnessed start states (lanegame.rl.starts)
    start_mix: float = 0.0                 # beta: P(a training game starts from a uniformly drawn archive state)
    allow_gamma_mismatch: bool = False     # continuing time: explicit opt-in to train with a gamma different from the
                                           # game's (a different objective, design §10.2); without it that is an error

    @classmethod
    def continuing(cls, game_cfg, **kw) -> "PPOConfig":
        """The continuing-time preset (design §10.4): gamma = game_cfg.gamma, lambda = 0.95, segments of 128 turns,
        games of 512 turns. reward_scale 0.1 and vf_clip 0.5 are provisional (returns of order 1 / (1 - gamma) times
        +-0.1-0.3 resources; retuned in the S5 smoke stage)."""
        if not game_cfg.continuing:
            raise ValueError("PPOConfig.continuing needs a continuing-time game configuration")
        base = dict(gamma=game_cfg.gamma, gae_lambda=0.95, segment_length=128, game_length=512, reward_scale=0.1,
                    vf_clip=0.5)
        base.update(kw)
        return cls(**base)

    @property
    def rollout_length(self) -> int:
        """Steps per environment per rollout: segment_length when set, else n_steps."""
        return self.segment_length if self.segment_length is not None else self.n_steps

    @property
    def batch_size(self) -> int:
        return self.n_envs * self.rollout_length

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


def check_time_semantics(game_cfg, cfg: "PPOConfig") -> None:
    """Continuing time needs a game length (a game never ends by itself); a training gamma different from the game's
    optimizes a different objective, which is allowed only as a stated choice: `allow_gamma_mismatch=True` (recorded
    with the run's configuration). Without the opt-in a mismatch raises ValueError. Finite (legacy) games are
    unaffected."""
    if not game_cfg.continuing:
        return
    if cfg.game_length is None or cfg.game_length < 1:
        raise ValueError("continuing time: PPOConfig.game_length (the bootstrapped time limit) must be set")
    if cfg.gamma != game_cfg.gamma and not cfg.allow_gamma_mismatch:
        raise ValueError(f"training gamma {cfg.gamma} differs from the game's gamma {game_cfg.gamma}: the policy would "
                         "optimize a different objective (design §10.2). Use PPOConfig.continuing(game_cfg), or set "
                         "allow_gamma_mismatch=True to make it a stated choice")


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
    values, advantages, returns, choice (1.0 where more than one action was legal). The returned statistics are
    minibatch means; `grad_norm` is the total gradient norm BEFORE clipping (`grad_norm_max` its largest value), so
    a run can be checked for exploding or vanishing gradients independently of `max_grad_norm`."""
    n = batch["obs"].shape[0]
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clipfrac": [], "grad_norm": []}
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
            gnorm = nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)     # the PRE-clip total norm
            optimizer.step()
            with torch.no_grad():
                kl = (((ratio - 1) - logratio) * w).sum() / nw
                cf = ((((ratio - 1).abs() > cfg.clip_coef).float()) * w).sum() / nw
            for key, val in (("policy_loss", pg_loss), ("value_loss", v_loss), ("entropy", entropy),
                             ("approx_kl", kl), ("clipfrac", cf), ("grad_norm", gnorm)):
                stats[key].append(float(val.detach()))
        if cfg.target_kl is not None and stats["approx_kl"][-1] > cfg.target_kl:
            break
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    out["grad_norm_max"] = float(np.max(stats["grad_norm"]))
    return out
