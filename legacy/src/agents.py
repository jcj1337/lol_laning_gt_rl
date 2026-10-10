import random
from collections import defaultdict
from typing import Dict, List, Callable

from .env import ACTIONS, state_ss

Policy = Callable[[state_ss], int]


class QLearningAgent:
    def __init__(self, alpha=0.15, gamma=0.95, eps=0.15):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.actions = list(ACTIONS)

        # Q maps: state -> [Q(s,SP), Q(s,SH), Q(s,F)]
        self.Q: Dict[state_ss, List[float]] = defaultdict(
            lambda: [0.0 for _ in self.actions]
        )

    def act(self, s: state_ss) -> int:
        # e-greedy
        if random.random() < self.eps:
            return random.choice(self.actions)

        q = self.Q[s]
        best_i = max(range(len(self.actions)), key=lambda i: q[i])
        return self.actions[best_i]

    def update(self, s: state_ss, a: int, r: float, s_next: state_ss) -> None:
        a_i = self.actions.index(a)
        q_sa = self.Q[s][a_i]

        target = r + self.gamma * max(self.Q[s_next])
        self.Q[s][a_i] = q_sa + self.alpha * (target - q_sa)

    def snapshot_greedy_policy(self) -> Policy:
        """
        Freeze the current greedy policy (based on Q)
        so it can be used as a new policy to be used by opponent agent
        """
        Q_copy = {k: v[:] for k, v in self.Q.items()}
        actions = self.actions[:]

        def pi(s: state_ss) -> int:
            q = Q_copy.get(s, [0.0 for _ in actions])
            best_i = max(range(len(actions)), key=lambda i: q[i])
            return actions[best_i]

        return pi