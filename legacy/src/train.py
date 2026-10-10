from typing import List, Dict

from .env import LaneEnv
from .agents import QLearningAgent
from .policies import Policy, always_shove, freeze_if_possible, stack_then_crash, random_mixed
from .eval import evaluate
from .utils import set_seed


def train_with_opponent_pool(
    env: LaneEnv,
    agent: QLearningAgent,
    pool: List[Policy],
    episodes: int = 5000,
    snapshot_every: int = 500,
    eval_every: int = 1000,
    eval_games: int = 200,
    fixed_eval_opponents: Dict[str, Policy] | None = None,
):
    if fixed_eval_opponents is None:
        fixed_eval_opponents = {
            "always_shove": always_shove,
            "freeze_if_possible": freeze_if_possible,
            "stack_then_crash": stack_then_crash,
        }

    log = []

    for ep in range(1, episodes + 1):
        opp_pi = pool[ep % len(pool)]  

        obs_y, obs_o = env.reset()
        done = False

        while not done:
            a_y = agent.act(obs_y)
            a_o = opp_pi(obs_o)

            (obs_y2, obs_o2), (r_y, r_o), done = env.step(a_y, a_o)
            agent.update(obs_y, a_y, r_y, obs_y2)
            obs_y, obs_o = obs_y2, obs_o2

        # snapshot current greedy policy into the pool
        if snapshot_every and ep % snapshot_every == 0:
            pool.append(agent.snapshot_greedy_policy())

        # evaluate periodically
        if eval_every and ep % eval_every == 0:
            results = evaluate(agent, env, fixed_eval_opponents, games=eval_games)
            log.append((ep, results))
            short = {k: round(v["avg_return"], 3) for k, v in results.items()}
            print(f"[ep {ep}] avg_return:", short)

    return log


def main():
    from .env import LaneParams

    set_seed(0)

    params = LaneParams(T=40, p_v=0.6, L=5.0)
    env = LaneEnv(params)
    agent = QLearningAgent(alpha=0.12, gamma=0.95, eps=0.15)

    # Start pool with baselines + a random-mixed style
    pool: List[Policy] = [
        always_shove,
        freeze_if_possible,
        stack_then_crash,
        random_mixed(0.4, 0.4, 0.2),
    ]

    train_with_opponent_pool(
        env=env,
        agent=agent,
        pool=pool,
        episodes=5000,
        snapshot_every=500,
        eval_every=1000,
        eval_games=200,
    )

    final = evaluate(agent, env, {
        "always_shove": always_shove,
        "freeze_if_possible": freeze_if_possible,
        "stack_then_crash": stack_then_crash,
    }, games=500)

    print("\nFinal evaluation:")
    for name, stats in final.items():
        print(name, stats)


if __name__ == "__main__":
    main()