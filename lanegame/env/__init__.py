"""Simultaneous-move RL environment around the existing laning game (roadmap Milestone 1).

    from lanegame.env import LaneEnv, ScriptedAgent, run_episodes
    from lanegame.baselines import aggressive, passive_safe

    env = LaneEnv(GameConfig())
    stats = run_episodes(env, ScriptedAgent(aggressive()), ScriptedAgent(passive_safe()), episodes=200, seed=42)

NumPy only; the game mechanics come from `lanegame.mechanics` / `lanegame.game` unchanged. The optional
PettingZoo adapter is in `lanegame.env.pettingzoo_adapter`.
"""

from .agents import (Agent, AgentPolicy, ChampionRuleAgent, ChampionScriptAgent, champion_suite, GreedyObservationAgent, LegacyChampionAgent,
                     ScriptedAgent, UniformAgent, agent_chooser, as_policy, champion_agents, episode_streams,
                     play_episode, run_episodes)
from .core import Chooser, LaneEnv, Observation, StepResult, greedy_chooser
from .observation import feature_names, legal_mask, legal_set, n_actions, obs_size, observe
from .trace import TRACE_SCHEMA, TRACE_VERSION, EpisodeLog, EpisodeStats, read_trace, trace_dict, write_trace

__all__ = ["LaneEnv", "Observation", "StepResult", "Chooser", "greedy_chooser",
           "observe", "legal_mask", "legal_set", "obs_size", "n_actions", "feature_names",
           "Agent", "ScriptedAgent", "ChampionRuleAgent", "ChampionScriptAgent", "champion_suite", "LegacyChampionAgent", "UniformAgent", "champion_agents", "GreedyObservationAgent", "AgentPolicy", "as_policy", "agent_chooser",
           "episode_streams", "play_episode", "run_episodes",
           "EpisodeLog", "EpisodeStats", "trace_dict", "write_trace", "read_trace", "TRACE_SCHEMA", "TRACE_VERSION"]
