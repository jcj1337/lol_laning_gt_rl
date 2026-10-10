from typing import Dict

from .env import LaneEnv, state_ss, SP, SH, F
from .agents import QLearningAgent
from .policies import Policy

ACTION_NAMES = {SP: "SP", SH: "SH", F: "F"}


def evaluate(agent: QLearningAgent, env: LaneEnv, opponents: Dict[str, Policy], games: int = 200):
    old_eps = agent.eps
    agent.eps = 0.0  

    out = {}
    for name, opp_pi in opponents.items():
        total_return = 0.0
        wins = 0
        action_counts = {SP: 0, SH: 0, F: 0}

        for _ in range(games):
            obs_y, obs_o = env.reset()
            done = False
            ep_return = 0.0

            while not done:
                a_y = agent.act(obs_y)
                a_o = opp_pi(obs_o)
                action_counts[a_y] += 1

                (obs_y2, obs_o2), (r_y, r_o), done = env.step(a_y, a_o)
                ep_return += r_y
                obs_y, obs_o = obs_y2, obs_o2

            total_return += ep_return
            
            if env.g > 0:
                wins += 1

        # set metrics in dict 
        out[name] = {
            "avg_return": total_return / games,
            "win_rate": wins / games,
            "action_freq": {ACTION_NAMES[a]: action_counts[a] for a in (SP, SH, F)},
        }

    agent.eps = old_eps
    return out