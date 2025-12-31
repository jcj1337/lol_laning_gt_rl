import random
from collections import defaultdict
from typing import Dict, List, Callable

from .env import ACTIONS, state_ss

Policy = Callable[[state_ss], int]
class QLearningAgent: 
    def __init__(self, alpha=0.15, gamma=0.95, eps=0.15) :
        self.alpha = alpha
        self.gamma = gamma
        self.actions = list(ACTIONS)
        self.eps = eps
        # initalize q values as 0's for all state-action pairs
        self.Q: Dict[Obs, List[float]] = defaultdict(lambda: [0.0 for _ in ACTIONS])

        def act(self, s: state_ss) -> int: 
            # explore 
            if random.random() < self.eps: 
                return random.choice(self.actions)  
            q = self.Q[s] 
            # q learn, best action
            high_i = max(range(len(self.actions)), key=lambda i: q[i])
            return self.actions[high_i]
        def update(self, s: state_ss, a: int, r: float, s_next: state_ss) -> None :
            a_i = self.actions.index(a) 
            q_sa = self.Q[s][a_i]
            target = r + self.gamma * max(self.Q[s_next]) 
            # update
            self.Q[s][ai] = q_sa + self.alpha * (target - q_sa)

        # for the other agent to train against, i.e. the frozen policy to train against
        def snapshot_greedy_policy(self) -> Policy :
            Q_copy = {k: v[:] for k, v in self.Q.items()}
            actions = self.actions[:]

            def pi(s: Obs) -> int:
                q = Q_copy.get(s, [0.0 for _ in actions])
                best_i = max(range(len(actions)), key=lambda i: q[i])
                return actions[best_i]

            return pi

            
