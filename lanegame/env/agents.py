"""Agents on top of `Observation`, complete episodes, and the bridge to the exact game's `Policy`.

Three kinds of agent are distinguished, because only the first two can be evaluated exactly:

* An agent with an EXPLICIT action distribution: `distribution(obs)` returns probabilities aligned with
  `obs.legal`, and `act(obs, rng)` samples from it with the agent's own generator. `ScriptedAgent` (a
  `lanegame.policy.Policy` behind an Observation) and `GreedyObservationAgent` (a deterministic argmax of
  a linear score, i.e. a one-hot distribution) are of this kind. Because the observation is a function of
  (k, state, side), such an agent IS a time-dependent Markov policy of the exact game, and `as_policy`
  turns it into a `Policy` for `evaluate`, `best_response_value`, `exploitability` and `path_statistics`.
* A DETERMINISTIC observation-based agent is the special case whose distribution is one-hot.
* An agent that only exposes `act` (random draws, internal history, a replay buffer, ...) can play episodes
  but has no exactly evaluable distribution; `as_policy` refuses it instead of guessing.

Random streams. `play_episode(env, agent_a, agent_b, seed)` derives three independent generators from one
seed (`SeedSequence.spawn`): the environment's chance events, A's action sampling and B's. Replacing one
agent by another never changes the draws of the environment or of the other agent.

Champion mode (roadmap M2a) has its own action space, so legacy `Policy` objects do not apply directly:
`ChampionRuleAgent` plays a fixed (wave, behavior, objective) preference, `LegacyChampionAgent` maps a
legacy policy into champion actions (Contest -> Hold, Safe -> Safe, objectives kept where reachable),
`UniformAgent` is uniform over the legal set, and `champion_agents(cfg)` collects named instances. There
is no exact solver for champion mode, so `as_policy` refuses champion-mode agents.

Crash-reward choices ("choice" siege mode of the v1 / v2 configurations) are a second decision stage inside
a turn. An agent may implement `choose(k, pre_state, side, opts_self, opts_opp, rng) -> int` and, for exact
evaluation, `choice_distribution(k, pre_state, side, opts_self, opts_opp)`; `play_episode` wires `choose`
into the environment. An agent without `choose` gets the environment's greedy default, which
`StepResult.info["choice_default_used"]` reports.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from ..config import (B_HOLD, B_SAFE, B_TRADE, B_YIELD, CONTEST, F, NONE, PRESSURE, SH, SP, VISION, GameConfig,
                      champ_act, objective_of, posture_of, wave_of)
from ..mechanics import legal_actions
from ..policy import Policy, greedy_resources
from .core import Chooser, LaneEnv, Observation
from .observation import n_actions, obs_size, observe
from .trace import EpisodeLog, EpisodeStats


@runtime_checkable
class Agent(Protocol):
    name: str

    def act(self, obs: Observation, rng: np.random.Generator) -> int: ...


def sample(rng: np.random.Generator, p) -> int:
    """Index drawn from the probability vector `p` with one uniform draw (inverse CDF; much cheaper than
    Generator.choice for the tiny vectors used here)."""
    c = np.cumsum(p)
    i = int(np.searchsorted(c, rng.random() * c[-1], side="right"))
    return min(i, len(c) - 1)


class ScriptedAgent:
    """A `Policy` of the exact game (scripted baseline, equilibrium table, ...) playing through Observations.

    `distribution` is the policy's own probability vector over `obs.legal`; `act` samples from it."""

    def __init__(self, policy: Policy, name: Optional[str] = None):
        self.policy = policy
        self.name = name or policy.name

    def distribution(self, obs: Observation) -> np.ndarray:
        if obs.champ is not None:
            raise TypeError(f"{self.name}: a legacy Policy cannot play champion mode directly; "
                            "wrap it in LegacyChampionAgent")
        return np.asarray(self.policy.move(obs.k, obs.state, obs.side, obs.legal), dtype=float)

    def act(self, obs: Observation, rng: np.random.Generator) -> int:
        return obs.legal[sample(rng, self.distribution(obs))]

    def choice_distribution(self, k, pre_state, side, opts_self, opts_opp) -> np.ndarray:
        return np.asarray(self.policy.choose(k, pre_state, side, opts_self, opts_opp), dtype=float)

    def choose(self, k, pre_state, side, opts_self, opts_opp, rng: np.random.Generator) -> int:
        return sample(rng, self.choice_distribution(k, pre_state, side, opts_self, opts_opp))


class GreedyObservationAgent:
    """Deterministic, stateless agent: the legal action with the highest linear score of the observation.

    scores = W @ obs.vector + b (W: n_actions x obs_size); ties go to the lowest action code. Its action
    distribution is one-hot, so it is exactly evaluable through `as_policy`. Crash-reward choices (older
    configurations) take the most resources. Used to exercise the exact bridge and as a stand-in for a
    trained deterministic network."""

    def __init__(self, weights, bias=None, name: str = "greedy_obs"):
        self.W = np.asarray(weights, dtype=float)
        self.b = np.zeros(self.W.shape[0]) if bias is None else np.asarray(bias, dtype=float)
        assert self.W.ndim == 2 and self.b.shape == (self.W.shape[0],)
        self.name = name

    @classmethod
    def random(cls, cfg: GameConfig, seed: int, scale: float = 1.0, name: Optional[str] = None):
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(n_actions(cfg), obs_size(cfg))) * scale
        b = rng.normal(size=n_actions(cfg)) * scale
        return cls(W, b, name or f"greedy_obs{seed}")

    def scores(self, obs: Observation) -> np.ndarray:
        return self.W @ obs.vector.astype(float) + self.b

    def distribution(self, obs: Observation) -> np.ndarray:
        sc = self.scores(obs)
        legal = obs.legal
        best = int(np.argmax([sc[a] for a in legal]))        # first maximum = lowest code on ties
        p = np.zeros(len(legal))
        p[best] = 1.0
        return p

    def act(self, obs: Observation, rng: Optional[np.random.Generator] = None) -> int:
        return obs.legal[int(np.argmax(self.distribution(obs)))]

    def choice_distribution(self, k, pre_state, side, opts_self, opts_opp) -> np.ndarray:
        return greedy_resources(opts_self)

    def choose(self, k, pre_state, side, opts_self, opts_opp, rng=None) -> int:
        return int(np.argmax(self.choice_distribution(k, pre_state, side, opts_self, opts_opp)))


# ---------------------------------------------------------------------- champion-mode agents (M2a)
class UniformAgent:
    """Uniform over the legal set (any configuration)."""

    def __init__(self, name: str = "uniform"):
        self.name = name

    def distribution(self, obs: Observation) -> np.ndarray:
        return np.full(len(obs.legal), 1.0 / len(obs.legal))

    def act(self, obs: Observation, rng: np.random.Generator) -> int:
        return obs.legal[sample(rng, self.distribution(obs))]


class ChampionRuleAgent:
    """Deterministic champion-mode agent: the first legal action among
    (wave, behavior, objective), (wave, behavior), (SP, behavior, objective), (SP, behavior).
    `freeze=True` uses Freeze whenever it is legal (the preferred wave otherwise)."""

    def __init__(self, name: str, wave: int = SP, behavior: int = B_HOLD, objective: int = PRESSURE,
                 freeze: bool = False):
        self.name, self.wave, self.behavior, self.objective, self.freeze = name, wave, behavior, objective, freeze

    def choice(self, legal) -> int:
        waves = ([F] if self.freeze else []) + [self.wave, SP]
        for w in waves:
            for o in (self.objective, NONE):
                a = champ_act(w, self.behavior, o)
                if a in legal:
                    return a
        raise ValueError(f"{self.name}: no candidate is legal in {legal}")

    def distribution(self, obs: Observation) -> np.ndarray:
        p = np.zeros(len(obs.legal))
        p[obs.legal.index(self.choice(obs.legal))] = 1.0
        return p

    def act(self, obs: Observation, rng=None) -> int:
        return self.choice(obs.legal)


class LegacyChampionAgent:
    """A legacy `Policy` (wave x Contest/Safe x objective) playing champion mode: its distribution over the
    lane's legacy legal set is mapped action by action, Contest -> Hold and Safe -> Safe, keeping the siege
    objective when it is legal for the champion and dropping it otherwise (champion too far from the tower)."""

    def __init__(self, policy: Policy, cfg: GameConfig, name: Optional[str] = None):
        self.policy, self.cfg = policy, cfg
        self.name = name or f"legacy:{policy.name}"

    def distribution(self, obs: Observation) -> np.ndarray:
        legacy = legal_actions(obs.state, obs.side, self.cfg)
        q = np.asarray(self.policy.move(obs.k, obs.state, obs.side, legacy), dtype=float)
        p = np.zeros(len(obs.legal))
        for a, w in zip(legacy, q):
            if w == 0:
                continue
            beh = B_HOLD if posture_of(a) == CONTEST else B_SAFE
            c = champ_act(wave_of(a), beh, objective_of(a))
            if c not in obs.legal:
                c = champ_act(wave_of(a), beh)
            p[obs.legal.index(c)] += w
        return p / p.sum()

    def act(self, obs: Observation, rng: np.random.Generator) -> int:
        return obs.legal[sample(rng, self.distribution(obs))]


def champion_agents(cfg: GameConfig) -> dict:
    """Named champion-mode agents: fixed behaviors, Shove variants, a freezer, uniform play, and the
    legacy siege baselines mapped into champion actions (`legacy:<name>`)."""
    from ..baselines import siege_baselines
    agents = [ChampionRuleAgent("hold", SP, B_HOLD), ChampionRuleAgent("yield", SP, B_YIELD),
              ChampionRuleAgent("trade", SP, B_TRADE), ChampionRuleAgent("safe", SP, B_SAFE),
              ChampionRuleAgent("shove_hold", SH, B_HOLD), ChampionRuleAgent("shove_trade", SH, B_TRADE),
              ChampionRuleAgent("hold_vision", SP, B_HOLD, VISION), ChampionRuleAgent("freeze_hold", SP, B_HOLD, freeze=True),
              UniformAgent()]
    agents += [LegacyChampionAgent(p, cfg) for p in siege_baselines()]
    return {a.name: a for a in agents}


# ---------------------------------------------------------------------- the exact bridge
class AgentPolicy(Policy):
    """An observation-based agent with an explicit distribution, seen as a `Policy` of the exact game.

    `move(k, state, side, legal)` rebuilds the Observation the agent would see and returns its distribution;
    `choose` returns the agent's `choice_distribution` (or the trivial distribution when there is one option).
    Valid because `observe` is a function of (k, state, side): the agent is then a Markov policy and
    `evaluate` / `exploitability` compute its exact value and exploitability."""

    def __init__(self, agent, cfg: GameConfig, name: Optional[str] = None):
        if cfg.champion_mode:
            raise TypeError("champion mode has no exact solver; AgentPolicy applies to the legacy configurations")
        if not callable(getattr(agent, "distribution", None)):
            raise TypeError(f"{getattr(agent, 'name', agent)!r} exposes no explicit action distribution; an agent "
                            "that only samples actions (or uses internal history) cannot be evaluated exactly")
        self.agent = agent
        self.cfg = cfg
        self.name = name or getattr(agent, "name", "agent")
        self._n = n_actions(cfg)

    def move(self, k, state, s, legal) -> np.ndarray:
        legal = tuple(legal)
        mask = np.zeros(self._n, dtype=bool)
        mask[list(legal)] = True
        obs = Observation(k, s, state, legal, observe(state, k, s, self.cfg), mask)
        p = np.asarray(self.agent.distribution(obs), dtype=float)
        if p.shape != (len(legal),) or np.any(p < 0) or abs(p.sum() - 1.0) > 1e-6:
            raise ValueError(f"{self.name}: invalid distribution {p} over {legal}")
        return p / p.sum()

    def choose(self, k, pre_state, s, opts_self, opts_opp) -> np.ndarray:
        if len(opts_self) == 1:
            return np.ones(1)
        f = getattr(self.agent, "choice_distribution", None)
        if not callable(f):
            raise TypeError(f"{self.name} has no explicit crash-reward choice distribution; the older "
                            "configurations' choice stage cannot be evaluated exactly for it")
        p = np.asarray(f(k, pre_state, s, opts_self, opts_opp), dtype=float)
        if p.shape != (len(opts_self),) or np.any(p < 0) or abs(p.sum() - 1.0) > 1e-6:
            raise ValueError(f"{self.name}: invalid choice distribution {p}")
        return p / p.sum()


def as_policy(agent, cfg: GameConfig) -> Policy:
    """The exact-game `Policy` equivalent to `agent` (a ScriptedAgent returns its own policy).
    Champion mode has no exact counterpart: TypeError."""
    if cfg.champion_mode:
        raise TypeError("champion mode has no exact solver; as_policy applies to the legacy configurations")
    if isinstance(agent, ScriptedAgent):
        return agent.policy
    return AgentPolicy(agent, cfg)


# ---------------------------------------------------------------------- episodes
def episode_streams(seed):
    """(environment seed, A's generator, B's generator), three independent children of one SeedSequence."""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    env_ss, a_ss, b_ss = ss.spawn(3)
    return ss, env_ss, np.random.default_rng(a_ss), np.random.default_rng(b_ss)


def agent_chooser(agent, rng: np.random.Generator) -> Optional[Chooser]:
    """The environment chooser callback for `agent` (None if the agent has no `choose`)."""
    f = getattr(agent, "choose", None)
    if not callable(f):
        return None
    return lambda k, pre_state, side, opts_self, opts_opp: f(k, pre_state, side, opts_self, opts_opp, rng)


def play_episode(env: LaneEnv, agent_a, agent_b, seed, record: bool = False, root=None, k0: int = 0) -> EpisodeLog:
    """One complete episode. `seed` (int, SeedSequence or None) is split into the environment's chance
    stream and the two agents' sampling streams. `record=True` keeps every turn's `info` in the log."""
    ss, env_seed, rng_a, rng_b = episode_streams(seed)
    env.choosers = (agent_chooser(agent_a, rng_a), agent_chooser(agent_b, rng_b))
    oa, ob = env.reset(seed=env_seed, root=root, k0=k0)
    log = EpisodeLog(seed=ss.entropy, k0=env.k0, horizon=env.horizon,
                     names=(getattr(agent_a, "name", "A"), getattr(agent_b, "name", "B")))
    while True:
        a = agent_a.act(oa, rng_a)
        b = agent_b.act(ob, rng_b)
        r = env.step(a, b)
        log.add(r, record)
        if r.done:
            return log
        oa, ob = r.obs


def run_episodes(env: LaneEnv, agent_a, agent_b, episodes: int, seed, record_first: bool = False,
                 root=None, k0: int = 0) -> EpisodeStats:
    """`episodes` independent episodes (one SeedSequence child each) with aggregate statistics and timing."""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    t0 = time.perf_counter()
    logs: List[EpisodeLog] = []
    for e, child in enumerate(ss.spawn(episodes)):
        logs.append(play_episode(env, agent_a, agent_b, child, record=(record_first and e == 0), root=root, k0=k0))
    elapsed = time.perf_counter() - t0
    return EpisodeStats.from_logs(logs, elapsed, env)
