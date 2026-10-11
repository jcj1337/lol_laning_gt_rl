"""Multi-agent reinforcement learning on the laning environment (roadmap M3): masked PPO with one shared
own-perspective network, frozen-snapshot self-play, shared-policy self-play with a current / historical opponent
mixture (OpenAI Five-inspired, `shared`) and evaluation. Requires PyTorch (optional extra `rl`);
`lanegame.env` never imports this package.

    from lanegame import GameConfig
    from lanegame.env import champion_agents
    from lanegame.rl import PPOConfig, OpponentSampler, train_ppo

    cfg = GameConfig.champions()
    res = train_ppo(cfg, PPOConfig(total_steps=50_000, seed=0), OpponentSampler([champion_agents(cfg)["hold_farmer"]]))
"""

from .agent import NetAgent, load_agent, load_checkpoint, save_checkpoint
from .network import MASK_VALUE, MaskedCategorical, PolicyValueNet
from .ppo import PPOConfig, compute_gae, explained_variance, ppo_update
from .train import Collector, OpponentSampler, TrainResult, train_ppo
from .evaluate import approximate_best_response, best_scripted_response, head_to_head, round_robin
from .selfplay import SelfPlayResult, SelfPlaySampler, SnapshotPool, train_selfplay
from .shared import (CURRENT, HISTORICAL, LEARNER, Matchup, MatchupSampler, SharedCollector, SharedConfig,
                     SharedResult, train_shared)

__all__ = ["PolicyValueNet", "MaskedCategorical", "MASK_VALUE", "PPOConfig", "compute_gae", "explained_variance",
           "ppo_update", "Collector", "OpponentSampler", "TrainResult", "train_ppo", "NetAgent", "save_checkpoint",
           "load_checkpoint", "load_agent", "head_to_head", "round_robin", "best_scripted_response",
           "approximate_best_response", "SnapshotPool", "SelfPlaySampler", "SelfPlayResult", "train_selfplay",
           "SharedConfig", "SharedCollector", "SharedResult", "MatchupSampler", "Matchup", "train_shared", "LEARNER",
           "CURRENT", "HISTORICAL"]
