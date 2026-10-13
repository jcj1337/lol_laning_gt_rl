"""PettingZoo Parallel API adapter over `LaneEnv` (optional dependency: pip install "lanegame[rl]").

The adapter is the project's standard multi-agent interface for external tooling. It is a thin, lossless view of
`LaneEnv`: every observation, mask, reward, truncation flag and info record is the underlying environment's own
(tests/test_pettingzoo_m4.py checks trajectory equality turn by turn), and nothing in the game mechanics is touched.
The training code (lanegame.rl) steps `LaneEnv` directly, for its per-seat bookkeeping and batched forward passes;
see docs/rl/m4_3a_rl_infrastructure_report.md for that decision.

Agents are "A" and "B"; both act every turn (the Parallel API is simultaneous by construction). Each observation is
{"observation": float32 vector in [0, 1], "action_mask": int8 mask}, the convention PettingZoo's tests and most
masked-policy implementations understand (`action_space(agent).sample(mask=obs["action_mask"])` draws a legal
action). Rewards are zero-sum (B's is exactly -A's); `terminations` is the game's own end (never under continuing
time), `truncations` the `max_turns` time limit; `infos` carry the same turn record for both agents (perfect
information): the per-turn reward decomposition, events, states before and after, and the M4 fields under ruleset m4.

Seeding. `reset(seed)` seeds the environment's chance stream exactly as `LaneEnv.reset(seed)` does (an int, a NumPy
SeedSequence or None). Agents' own action sampling is theirs; to reproduce a `play_episode(env, a, b, seed)` game,
seed the adapter with `episode_streams(seed)[1]` and draw the agents' actions from `episode_streams(seed)[2:]`.

Start states. `reset(options=...)` accepts
    {"root": State or its list form, "k0": int, "champ": champion state or its list form}   (the LaneEnv arguments)
    {"start": StartState}                                                                   (lanegame.env.starts)
Under ruleset m4 a list-form champion state is read as an M4State, otherwise as a ChampState.

Continuing time (rulesets "m2c" and "m4"). The game has no terminal state, so an episode ends only by the
`max_turns` truncation; without `max_turns` the adapter steps forever and `agents` never empties (PettingZoo's
`parallel_api_test` stops after its own cycle count). Give `max_turns` to any consumer that expects episodes.

Crash-reward choices of the older configurations ("choice" siege mode) are resolved by the environment's greedy
default unless `choosers` are given, because the Parallel API has no second decision stage; the siege model and the
champion games have no such stage.
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

    # ---- introspection
    @property
    def cfg(self) -> GameConfig:
        return self.env.cfg

    @property
    def max_turns(self) -> Optional[int]:
        return self.env.max_turns

    @max_turns.setter
    def max_turns(self, value: Optional[int]) -> None:
        self.env.max_turns = value

    # ---- spaces
    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    # ---- episode
    @staticmethod
    def _pack(obs):
        return {"observation": obs.vector, "action_mask": obs.mask.astype(np.int8)}

    def _champ_from(self, champ):
        if champ is None or not isinstance(champ, list):
            return champ
        from .starts import champ_from_list
        return champ_from_list(self.env.cfg, champ)

    def reset(self, seed=None, options=None):
        options = options or {}
        start = options.get("start")
        if start is not None:
            root, k0, champ = start.lane, int(start.phase), start.champ
        else:
            root = options.get("root")
            if isinstance(root, list):
                root = State.from_list(root)
            k0 = int(options.get("k0", 0))
            champ = self._champ_from(options.get("champ"))
        oa, ob = self.env.reset(seed=seed, root=root, k0=k0, champ=champ)
        self.agents = list(self.possible_agents)
        return {"A": self._pack(oa), "B": self._pack(ob)}, {"A": {"legal": oa.legal}, "B": {"legal": ob.legal}}

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("the episode is over; call reset()")
        r = self.env.step(int(actions["A"]), int(actions["B"]))
        oa, ob = r.obs
        observations = {"A": self._pack(oa), "B": self._pack(ob)}
        rewards = {"A": float(r.rewards[0]), "B": float(r.rewards[1])}
        terminations = {"A": bool(r.terminated), "B": bool(r.terminated)}
        truncations = {"A": bool(r.truncated), "B": bool(r.truncated)}
        infos = {"A": r.info, "B": r.info}
        if r.done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """Global state (perfect information): A's observation vector. Under ruleset m4 that vector is lossless
        (lanegame.env.observation_m4.decode_m4 reads the full state back)."""
        return self.env.observe()[0].vector

    def render(self):
        if self.env.state is None:
            return ""
        return state_row(self.env.state, self.env.cfg)

    def close(self):
        pass

    def aec(self):
        """The same environment through PettingZoo's AEC (turn-based) API, for AEC-only consumers."""
        from pettingzoo.utils.conversions import parallel_to_aec
        return parallel_to_aec(self)
