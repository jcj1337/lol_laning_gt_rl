"""Counterfactual branching from reachable states (diagnostics milestone, Phase B).

A `Snapshot` is everything the champion environment's Markov state holds: the lane `State`, the `ChampState`,
the decision index k and the two seat skills. `restore(env, snap, seed)` puts a `LaneEnv` exactly into that state
(through the environment's own `reset(root=..., k0=..., champ=...)`) with a fresh chance stream. One champion-mode
turn draws exactly N_CHANCE uniforms whatever the actions, so every action sequence played from a snapshot with the
same seed sees the same chance vector on every turn (common random numbers turn by turn).

`branch(...)` estimates, for one focal seat, the downstream return of committing to each of several candidate
first moves (a legal action code, or a short scripted `Plan`) and then following a fixed continuation policy,
against a fixed opponent policy:

    G_m(c) = sum_{t >= k} r_t^focal          (rollout m of candidate c, to the end of the game)

Pairing. Rollout m of every candidate uses the same three streams (environment chance, focal sampling, opponent
sampling; `rollout_seed(root, m)`), and each seat draws exactly ONE uniform per turn whether or not a plan
overrides its choice, so candidates are compared on identical randomness until (and, where the trajectories
re-merge, after) they diverge. The opponent's action on the branch turn is drawn from its own observation of the
snapshot, which does not contain the focal player's uncommitted action, so it is identical for every candidate:
nobody sees the other's current action. Sampling uses the same inverse-CDF rule as `env.agents.sample`, so a
rollout of an agent pair without a plan reproduces `play_episode` exactly.

Policies are frozen: agents are only queried through `distribution(obs)` (a NetAgent's network has no gradients),
and identical observations are answered from a cache, so equal states get bit-identical probabilities in every
rollout. Estimates are Monte Carlo values of the given continuation policies on this opponent, not game-theoretic
values.

Continuing time (rulesets "m2c" / "m4"; plan WP9). A game never ends, so a rollout needs a window (`max_turns`), and
`gamma` (default: the game's) discounts it: G_m(c) = sum_{t < W} gamma^t r_t^focal from the branch turn. A Snapshot
then carries the M4State (positions, centi-HP, absence, XP, items, gold, tower damage) and the phase k mod P, which
is all the dynamics depend on. An optional critic bootstrap adds gamma^W V_hat(s_W); it is an APPROXIMATION whose
bias gamma^W E[V_hat - V] is unknown (design §10.5.4), and results carrying it are labelled. These are approximate
champion-game measurements; exact values exist only for the legacy games (lanegame.solver).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..champions import ChampSide, ChampState
from ..config import GameConfig
from ..env import LaneEnv
from ..env.agents import episode_streams
from ..state import State

A, B = 0, 1


# ---------------------------------------------------------------------------------------------- snapshots
@dataclass(frozen=True)
class Snapshot:
    """A restorable champion-mode environment state."""
    lane: State
    champ: ChampState
    k: int
    skills: Tuple[float, float] = (0.5, 0.5)
    # OM2 (GameConfig.obs_last_action): the (A, B) actions of the turn before this state; None = not recorded (an
    # observation of an obs_last_action configuration then refuses to be built; (-1, -1) = no previous turn)
    last_actions: Optional[Tuple[int, int]] = None

    @classmethod
    def from_env(cls, env: LaneEnv) -> "Snapshot":
        return cls(env.state, env.champ, env.k, (env.cfg.skill_a, env.cfg.skill_b), tuple(env.last_actions))

    @classmethod
    def from_info(cls, info: dict, skills: Tuple[float, float]) -> "Snapshot":
        """The state at the START of a recorded turn (`StepResult.info` of that turn)."""
        return cls(State.from_list(info["state_before"]), champ_from_list(info["champ_before"]), int(info["k"]),
                   (float(skills[0]), float(skills[1])))

    def mirror(self) -> "Snapshot":
        """The same situation with the seats swapped (seat B's view becomes seat A's)."""
        la = None if self.last_actions is None else (self.last_actions[1], self.last_actions[0])
        return Snapshot(self.lane.mirror(), self.champ.mirror(), self.k, (self.skills[1], self.skills[0]), la)

    def with_side(self, seat: int, **kw) -> "Snapshot":
        """The snapshot with one champion's record changed (pos, hp, absent); used for HP-value comparisons."""
        sides = [self.champ.a, self.champ.b]
        sides[seat] = sides[seat]._replace(**kw)
        return Snapshot(self.lane, type(self.champ)(*sides), self.k, self.skills, self.last_actions)

    def config(self, base: GameConfig) -> GameConfig:
        return base.with_(skill_a=self.skills[0], skill_b=self.skills[1])

    def to_dict(self) -> dict:
        d = {"lane": self.lane.to_list(), "champ": self.champ.to_list(), "k": self.k, "skills": list(self.skills)}
        if self.last_actions is not None:
            d["last_actions"] = list(self.last_actions)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        la = d.get("last_actions")
        return cls(State.from_list(d["lane"]), champ_from_list(d["champ"]), int(d["k"]),
                   (float(d["skills"][0]), float(d["skills"][1])), None if la is None else (int(la[0]), int(la[1])))


def champ_from_list(v):
    """ChampState (M2: [pos, hp, absent] per side) or M4State (M4: seven fields per side) from its list form."""
    if len(v[0]) == 7:
        from ..champions_m4 import M4State
        return M4State.from_list(v)
    return ChampState.from_list(v)


def observation(cfg: GameConfig, snap: Snapshot, seat: int):
    """The Observation seat `seat` receives in `snap` (exactly what LaneEnv would hand it)."""
    from ..env.core import Observation
    from ..env.observation import legal_set, n_actions, observe_full
    c = snap.config(cfg)
    legal = legal_set(snap.lane, seat, c, snap.champ)
    mask = np.zeros(n_actions(c), dtype=bool)
    mask[list(legal)] = True
    vec = observe_full(snap.lane, snap.k, seat, c, snap.champ, snap.last_actions)
    return Observation(snap.k, seat, snap.lane, legal, vec, mask, snap.champ, c)


def restore(env: LaneEnv, snap: Snapshot, seed):
    """Reset `env` into `snap` with chance stream `seed`; returns both observations."""
    if (env.cfg.skill_a, env.cfg.skill_b) != tuple(snap.skills):
        raise ValueError(f"environment skills {(env.cfg.skill_a, env.cfg.skill_b)} differ from the snapshot's "
                         f"{snap.skills}; build it with snap.config(base)")
    if not env.cfg.champion_mode:
        raise ValueError("snapshots are champion-mode states")
    return env.reset(seed=seed, root=snap.lane, k0=snap.k, champ=snap.champ, last_actions=snap.last_actions)


def rollout_seed(root: int, m: int, *key: int) -> np.random.SeedSequence:
    """A fresh SeedSequence for rollout m (optionally under a state key). A new object every call: spawning from
    a SeedSequence advances its counter, so reusing one object across candidates would break the pairing."""
    return np.random.SeedSequence(entropy=int(root), spawn_key=tuple(int(x) for x in key) + (int(m),))


# ---------------------------------------------------------------------------------------------- plans
DEFER = -1          # a plan's answer "the continuation policy's own action this turn; the plan stays active"


class Plan:
    """A short scripted deviation of the focal player. `act(obs, t)` returns a legal action code for turn t of the
    branch (t = 0 is the branch turn), DEFER to let the continuation policy act this turn while the plan stays
    active, or None to hand control to the continuation policy for good.

    `transform(obs, t, p)` is consulted on every DEFER turn (while the plan is active): it maps the continuation
    policy's probability vector over `obs.legal` to the vector the focal seat samples from this turn, with the same
    uniform draw (the pairing of the rollout streams is unchanged). The default is the identity, so a plan that
    only implements `act` behaves exactly as before. A distribution-level plan (M4.3b follow-up: an intervention
    on one action dimension that keeps the policy's other decisions) returns DEFER from `act` and reshapes p here."""
    name = "plan"

    def act(self, obs, t: int) -> Optional[int]:
        raise NotImplementedError

    def transform(self, obs, t: int, p: np.ndarray) -> np.ndarray:
        return p


class Commit(Plan):
    """Commit to one action on the branch turn, then follow the continuation policy."""

    def __init__(self, action: int, name: Optional[str] = None):
        self.action, self.name = int(action), name or f"a{int(action)}"

    def act(self, obs, t: int) -> Optional[int]:
        return self.action if t == 0 else None


class FollowPolicy(Plan):
    """No deviation: the continuation policy's own (sampled) action from the branch turn on."""
    name = "policy"

    def act(self, obs, t: int) -> Optional[int]:
        return None


class FunctionPlan(Plan):
    """A plan given by a function (obs, t) -> action or None."""

    def __init__(self, name: str, fn: Callable):
        self.name, self.fn = name, fn

    def act(self, obs, t: int) -> Optional[int]:
        return self.fn(obs, t)


# ---------------------------------------------------------------------------------------------- policies
class ProbCache:
    """Action distributions of frozen agents, memoized per (agent, observation vector, legal mask). The observation
    vector is an injective function of the full state and the mask is the legal set, so equal keys mean equal
    decision situations."""

    def __init__(self, max_entries: int = 400_000):
        self.max_entries = max_entries
        self._d: Dict[tuple, np.ndarray] = {}
        self.hits = self.misses = 0

    def distribution(self, agent, obs) -> np.ndarray:
        key = (id(agent), obs.vector.tobytes(), obs.mask.tobytes())
        p = self._d.get(key)
        if p is None:
            self.misses += 1
            p = np.asarray(agent.distribution(obs), dtype=float)
            if len(self._d) >= self.max_entries:
                self._d.clear()
            self._d[key] = p
        else:
            self.hits += 1
        return p


def sample_index(p: np.ndarray, u: float) -> int:
    """The index `env.agents.sample` draws from p with the uniform u (same inverse-CDF rule)."""
    c = np.cumsum(p)
    i = int(np.searchsorted(c, u * c[-1], side="right"))
    return min(i, len(c) - 1)


# ---------------------------------------------------------------------------------------------- rollouts
@dataclass
class Rollout:
    ret: float                         # focal player's return from the branch turn: to the end of the game, or (with
                                       # gamma) the discounted sum over the window, plus any bootstrap
    first_actions: Tuple[int, int]     # actions of (A, B) on the branch turn
    farm: Tuple[float, float]          # farm collected from the branch turn on, (focal, opponent)
    plates: Tuple[float, float]
    bounty: Tuple[float, float]
    deaths: Tuple[int, int]
    recalls: Tuple[int, int]
    turns: int
    first_reward: float                # focal reward on the branch turn itself
    infos: Optional[List[dict]] = None
    undiscounted: float = 0.0          # sum of the focal rewards over the rollout (descriptive)
    bootstrap: Optional[float] = None  # gamma^W V_hat(s_W) when a critic bootstrap was added (an APPROXIMATION)
    m4: Optional[dict] = None          # M4: focal / opponent XP gained, plates, purchases, tower damage dealt


def rollout(env: LaneEnv, snap: Snapshot, focal: int, plan: Plan, agents: Sequence, seed,
            cache: Optional[ProbCache] = None, record: bool = False, max_turns: Optional[int] = None,
            gamma: Optional[float] = None, bootstrap: Optional[Callable] = None,
            opponent_plan: Optional[Plan] = None) -> Rollout:
    """One rollout from `snap`: the focal seat follows `plan` and then `agents[focal]`; the other seat follows
    `agents[1 - focal]` throughout, or, with `opponent_plan` (M4.4a: an explicit protocol for the opponent's
    simultaneous action, e.g. "the opponent commits to its own modal action on the branch turn"), that plan first and
    then its agent, under exactly the plan rules of the focal seat (one uniform per seat per turn either way).
    `seed` (int or SeedSequence) is split exactly as in `play_episode`.
    `max_turns` stops after that many turns (the return is then truncated, not bootstrapped). Continuing time needs
    `max_turns` (the window W); `gamma` (default under continuing time: the game's) discounts the return, and
    `bootstrap(obs_focal) -> V_hat` adds gamma^W V_hat(s_W) (an approximation, design §10.5.4)."""
    if env.continuing:
        if max_turns is None:
            raise ValueError("continuing time: a rollout needs a window (max_turns)")
        if gamma is None:
            gamma = env.cfg.gamma
    g = 1.0 if gamma is None else float(gamma)
    cache = cache if cache is not None else ProbCache()
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    _, env_ss, rng_a, rng_b = episode_streams(ss)
    rngs = (rng_a, rng_b)
    obs = restore(env, snap, env_ss)
    o = 1 - focal
    plans = {focal: plan, o: opponent_plan}
    live = {focal: True, o: opponent_plan is not None}       # without an opponent plan: exactly the original loop
    t = 0
    ret = first_reward = 0.0
    farm, plates, bounty = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
    deaths, recalls = [0, 0], [0, 0]
    first, infos = None, ([] if record else None)
    undisc, disc_w = 0.0, 1.0
    m4 = {"xp": [0, 0], "plates": [0, 0], "purchases": [0, 0], "tower_damage": [0, 0], "freeze_maintained": [0, 0],
          "crashes": [0, 0], "invested": [0, 0]} if env.cfg.is_m4 else None
    while True:
        acts = [0, 0]
        for s in (A, B):
            ob = obs[s]
            p = cache.distribution(agents[s], ob)
            pa = None
            if live[s]:
                pl = plans[s]
                pa = pl.act(ob, t)
                if pa is None:
                    live[s] = False
                elif pa == DEFER:
                    q = np.asarray(pl.transform(ob, t, p), dtype=float)
                    if q.shape != p.shape or not np.all(np.isfinite(q)) or q.sum() <= 0 or (q < 0).any():
                        raise ValueError(f"plan {pl.name!r} returned an invalid distribution at turn {ob.k}")
                    p = q
                elif pa not in ob.legal:
                    raise ValueError(f"plan {pl.name!r} chose illegal action {pa} at turn {ob.k} "
                                     f"(legal {ob.legal})")
            u = rngs[s].random()                              # exactly one draw per seat per turn
            a = ob.legal[sample_index(p, u)]
            if pa is not None and pa != DEFER:
                a = int(pa)
            acts[s] = a
        r = env.step(acts[0], acts[1])
        info = r.info
        if first is None:
            first = (acts[0], acts[1])
            first_reward = r.rewards[focal]
        ret += disc_w * r.rewards[focal]
        undisc += r.rewards[focal]
        disc_w *= g
        for i, s in enumerate((focal, o)):
            farm[i] += info["farm"][s]
            plates[i] += info["plates"][s]
            bounty[i] += info["bounty"][s]
            deaths[i] += int(info["died"][s])
            recalls[i] += int(info["recalled"][s])
            if m4 is not None:
                m4["xp"][i] += info["xp_gain"][s]
                m4["plates"][i] += info["plates_broken"][s]
                m4["purchases"][i] += info["purchases"][s]
                m4["tower_damage"][i] += info["tower_damage_dealt"][s]
                m4["freeze_maintained"][i] += int(info["freeze_maintained"][s])
                m4["crashes"][i] += int(info["crash_hp"][s] > 0)
                m4["invested"][i] += info["invested"][s]
        if record:
            infos.append(info)
        t += 1
        if r.done or (max_turns is not None and t >= max_turns):
            break
        obs = r.obs
    boot = None
    if bootstrap is not None:
        boot = disc_w * float(bootstrap(r.obs[focal]))
        ret += boot
    return Rollout(ret, first, (farm[0], farm[1]), (plates[0], plates[1]), (bounty[0], bounty[1]),
                   (deaths[0], deaths[1]), (recalls[0], recalls[1]), t, first_reward, infos, undisc, boot, m4)


@dataclass
class BranchResult:
    """Paired rollout returns: returns[c][m] for candidate c and rollout m (same streams across candidates)."""
    snapshot: Snapshot
    focal: int
    candidates: List[str]
    returns: Dict[str, np.ndarray]
    first_reward: Dict[str, np.ndarray]
    deaths: Dict[str, np.ndarray]                     # (n, 2): focal, opponent
    farm: Dict[str, np.ndarray]
    plates: Dict[str, np.ndarray]
    bounty: Dict[str, np.ndarray]
    recalls: Dict[str, np.ndarray]
    opp_first: Dict[str, np.ndarray]                  # opponent's branch-turn action per rollout
    meta: dict = field(default_factory=dict)
    m4: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)   # M4 per-candidate counters (n, 2)

    @property
    def n(self) -> int:
        return len(next(iter(self.returns.values())))

    def mean(self, c: str) -> Tuple[float, float]:
        x = self.returns[c]
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))

    def diff(self, c1: str, c2: str) -> Tuple[float, float]:
        """Paired mean of G(c1) - G(c2) and its standard error."""
        d = self.returns[c1] - self.returns[c2]
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))

    def advantage(self, c: str, weights: Dict[str, float]) -> Tuple[float, float]:
        """Paired estimate of Q(c) - sum_b w_b Q(b) (w = the policy's probabilities over the candidates that are
        its legal actions), with its standard error; the per-rollout control variate keeps it low-variance."""
        base = sum(w * self.returns[b] for b, w in weights.items())
        d = self.returns[c] - base
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def branch(cfg: GameConfig, snap: Snapshot, focal: int, candidates: Dict[str, Plan], agents: Sequence, n: int,
           root: int, key: Sequence[int] = (), cache: Optional[ProbCache] = None,
           max_turns: Optional[int] = None, gamma: Optional[float] = None,
           bootstrap: Optional[Callable] = None, opponent_plan: Optional[Plan] = None) -> BranchResult:
    """Paired rollouts of every candidate from `snap` (see the module docstring). `cfg` is the base champion
    configuration; the snapshot's skills are applied to it. Continuing time: `max_turns` is the window and the
    returns are discounted with `gamma` (default the game's); a `bootstrap` critic makes the result an approximation
    (meta["bootstrapped"]). `opponent_plan` (the same for every candidate) fixes the opponent's protocol (`rollout`)."""
    env = LaneEnv(snap.config(cfg), cache_size=20_000)
    cache = cache if cache is not None else ProbCache()
    out = {k: {} for k in ("returns", "first_reward", "deaths", "farm", "plates", "bounty", "recalls", "opp_first")}
    m4_out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, plan in candidates.items():
        rs = [rollout(env, snap, focal, plan, agents, rollout_seed(root, m, *key), cache, max_turns=max_turns,
                      gamma=gamma, bootstrap=bootstrap, opponent_plan=opponent_plan) for m in range(n)]
        if rs and rs[0].m4 is not None:
            m4_out[name] = {k: np.array([r.m4[k] for r in rs], dtype=float) for k in rs[0].m4}
        out["returns"][name] = np.array([r.ret for r in rs])
        out["first_reward"][name] = np.array([r.first_reward for r in rs])
        out["deaths"][name] = np.array([r.deaths for r in rs], dtype=float)
        out["farm"][name] = np.array([r.farm for r in rs])
        out["plates"][name] = np.array([r.plates for r in rs])
        out["bounty"][name] = np.array([r.bounty for r in rs])
        out["recalls"][name] = np.array([r.recalls for r in rs], dtype=float)
        out["opp_first"][name] = np.array([r.first_actions[1 - focal] for r in rs])
    meta = {"n": n, "root": int(root), "key": [int(x) for x in key], "max_turns": max_turns,
            "gamma": gamma if gamma is not None else (cfg.gamma if cfg.continuing else 1.0),
            "bootstrapped": bootstrap is not None,
            "opponent_plan": getattr(opponent_plan, "name", None)}
    if bootstrap is not None:
        meta["label"] = "APPROXIMATION: critic bootstrap gamma^W V_hat(s_W) added (design §10.5.4)"
    res = BranchResult(snap, focal, list(candidates), **out, meta=meta)
    res.m4 = m4_out
    return res
