"""Thin PettingZoo Parallel API adapter over `LaneEnv` (optional dependency: pip install "lanegame[rl]").

Agents are "A" and "B"; both act every turn (the Parallel API is simultaneous by construction). Each
observation is {"observation": float32 vector, "action_mask": int8 mask}, the convention PettingZoo's
tests and most masked-policy implementations understand. Rewards, terminations and truncations are the
underlying StepResult's; `infos` carry the same turn record for both agents.

Crash-reward choices of the older configurations ("choice" siege mode) are resolved by the environment's
greedy default unless `choosers` are given, because the Parallel API has no second decision stage; the
siege model has no such stage. Nothing in the game mechanics is touched.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import GameConfig
from ..render import state_row
from ..state import State
from .core import LaneEnv

try:                                                  # optional dependencies
    from pettingzoo.utils.env import ParallelEnv as _ParallelEnv
    from gymnasium import spaces as _spaces
    HAVE_PETTINGZOO = True
except ImportError:                                   # pragma: no cover - depends on the installation
    _ParallelEnv = object
    _spaces = None
    HAVE_PETTINGZOO = False

AGENTS = ("A", "B")


class LaneParallelEnv(_ParallelEnv):
    metadata = {"render_modes": ["ansi"], "name": "lanegame_lane_v1", "is_parallelizable": False}

    def __init__(self, cfg: Optional[GameConfig] = None, max_turns: Optional[int] = None, choosers=None,
                 render_mode: Optional[str] = None, cache_size: int = 0):
        if not HAVE_PETTINGZOO:
            raise ImportError("LaneParallelEnv needs pettingzoo and gymnasium: pip install \"lanegame[rl]\"")
        self.env = LaneEnv(GameConfig() if cfg is None else cfg, max_turns=max_turns, choosers=choosers,
                           cache_size=cache_size)
        self.render_mode = render_mode
        self.possible_agents = list(AGENTS)
        self.agents = []
        n, m = self.env.n_actions, self.env.obs_size
        self._observation_space = _spaces.Dict({
            "observation": _spaces.Box(low=0.0, high=1.0, shape=(m,), dtype=np.float32),
            "action_mask": _spaces.Box(low=0, high=1, shape=(n,), dtype=np.int8),
        })
        self._action_space = _spaces.Discrete(n)

    # ---- spaces
    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    # ---- episode
    @staticmethod
    def _pack(obs):
        return {"observation": obs.vector, "action_mask": obs.mask.astype(np.int8)}

    def reset(self, seed=None, options=None):
        options = options or {}
        root = options.get("root")
        if isinstance(root, list):
            root = State.from_list(root)
        champ = options.get("champ")                  # champion mode: ChampState or its list form
        if isinstance(champ, list):
            from ..champions import ChampState
            champ = ChampState.from_list(champ)
        oa, ob = self.env.reset(seed=seed, root=root, k0=int(options.get("k0", 0)), champ=champ)
        self.agents = list(self.possible_agents)
        return {"A": self._pack(oa), "B": self._pack(ob)}, {"A": {}, "B": {}}

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("the episode is over; call reset()")
        r = self.env.step(int(actions["A"]), int(actions["B"]))
        oa, ob = r.obs
        observations = {"A": self._pack(oa), "B": self._pack(ob)}
        rewards = {"A": float(r.rewards[0]), "B": float(r.rewards[1])}
        terminations = {"A": r.terminated, "B": r.terminated}
        truncations = {"A": r.truncated, "B": r.truncated}
        infos = {"A": r.info, "B": r.info}
        if r.done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """Global state (perfect information): A's observation vector."""
        return self.env.observe()[0].vector

    def render(self):
        if self.env.state is None:
            return ""
        return state_row(self.env.state, self.env.cfg)

    def close(self):
        pass
