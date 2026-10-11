"""Shared-policy self-play with a current / historical opponent mixture (OpenAI Five-inspired; see
docs/openai_five_adaptation.md for what is taken from Berner et al. 2019 and what is our own choice).

One trainable network pi_theta(a | o) plays the game. At every episode reset, independently:

  * with probability `current_fraction` (OpenAI Five: 0.8) it is a CURRENT-POLICY game: the current network
    controls both seats, each seat acting on its own-perspective observation and own legal mask, and BOTH
    seats' transitions are PPO samples (a 32-turn game yields 64 agent-transitions);
  * otherwise it is a HISTORICAL game: the current network controls a uniformly drawn seat against a frozen past
    version drawn uniformly from the bounded snapshot pool. Only the current network's transitions are samples
    (32 per game); the snapshot acts through its saved weights (a `NetAgent`: a deep copy without gradients)
    and never trains.

Bootstrap: the pool is empty until the first snapshot (or holds the untrained initial network with
`snapshot_initial`), so every game is a current-policy game until a past version exists. The matchup of an
episode is drawn when it starts and stays fixed until it ends; snapshots join the pool only between updates,
so a rollout never sees the pool change.

Rewards. Each seat's reward is its own zero-sum game reward, exactly `StepResult.rewards[seat]` (r to A, -r to
B), times `reward_scale`; nothing is negated a second time and nothing is added. In a current-policy game the
two seats' returns cancel by construction, so training returns of those games say nothing about progress:
progress is measured against fixed references (lanegame.experiments.rl.openai_five).

PPO. A rollout is collected with ONE fixed parameter vector: the network is not updated during collection, and
each recorded log-probability is that of the action actually sampled under it. Every environment has two
trajectory slots (seat A and seat B); a slot is trainable while the current network controls that seat. GAE
(`compute_gae`, the same semantics as train.py) runs over all slots; a slot's episode is entirely trainable or
entirely not and advantages never cross episode boundaries, so only trainable entries are passed to the
unchanged `ppo_update`.

Accounting. `env_turns` counts environment steps; `agent_transitions` counts trainable samples. The budget
(`PPOConfig.total_steps`) and the learning-rate schedule are in agent transitions, so that methods collecting a
different number of samples per game can be compared at equal sample counts.

Continuing time (rulesets "m2c" / "m4"): segments of `PPOConfig.rollout_length` turns, per-game time limits with a
seeded random first-game length per environment, starts from rho_train, truncations bootstrapped from the pre-reset
observation (as in lanegame.rl.train; `GameTimes`). A matchup may change only at a game boundary: `collect` raises if
an environment's matchup differs from the one drawn when its game started.

Checkpoints. Snapshots (out_dir/snapshots/snap_<transitions>.pt) and the final network carry the step counts,
seed, configurations, model spec and snapshot identity; an existing non-empty run directory is refused unless
`overwrite=True`, and no checkpoint file is ever replaced silently.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import GameConfig
from ..env import LaneEnv
from ..env.agents import episode_streams
from ..env.trace import EpisodeLog
from .agent import NetAgent, save_checkpoint
from .network import MaskedCategorical, PolicyValueNet
from .ppo import PPOConfig, check_time_semantics, compute_gae, explained_variance, ppo_update
from .selfplay import SnapshotPool
from .train import GameTimes

LEARNER = None                   # controller of a seat played by the current (trainable) network
CURRENT, HISTORICAL = "current", "historical"
METHOD = "shared_selfplay"
FORMAT_VERSION = 1


@dataclass
class SharedConfig:
    current_fraction: float = 0.8      # share of training games in which the current network plays both seats
    snapshot_every: int = 25_600       # agent transitions between frozen snapshots
    pool_size: int = 8                 # most recent snapshots kept as historical opponents (oldest dropped)
    snapshot_initial: bool = False     # make the untrained initial network the first snapshot

    def validate(self) -> None:
        if not 0.0 <= self.current_fraction <= 1.0:
            raise ValueError("current_fraction must be in [0, 1]")
        if self.snapshot_every <= 0 or self.pool_size <= 0:
            raise ValueError("snapshot_every and pool_size must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Matchup:
    kind: str                          # CURRENT or HISTORICAL
    controllers: Tuple[object, object]     # per seat: LEARNER or a frozen agent
    opponent: str                      # "self" or the snapshot's name
    learner_seat: int                  # -1 in a current-policy game (both seats)


class MatchupSampler:
    """Draws the matchup of an episode: with probability `current_fraction`, or whenever the pool is empty, a
    current-policy game; otherwise a historical game against a snapshot drawn uniformly from the pool, the
    current network in a uniformly drawn seat. Draws use the caller's generator (reproducible)."""

    def __init__(self, pool: SnapshotPool, current_fraction: float):
        self.pool, self.current_fraction = pool, float(current_fraction)
        self.counts: Dict[str, int] = {}

    def sample(self, rng: np.random.Generator) -> Matchup:
        u = rng.random()
        if not self.pool or u < self.current_fraction:
            m = Matchup(CURRENT, (LEARNER, LEARNER), "self", -1)
        else:
            opp = self.pool.snapshots[int(rng.integers(len(self.pool)))]
            seat = int(rng.integers(2))
            m = Matchup(HISTORICAL, (LEARNER, opp) if seat == 0 else (opp, LEARNER), opp.name, seat)
        self.counts[m.opponent] = self.counts.get(m.opponent, 0) + 1
        return m

    def describe(self) -> dict:
        return {"current_fraction": self.current_fraction, "pool_size": self.pool.pool_size,
                "historical_sampling": "uniform over the pool", "draws": dict(self.counts)}


@dataclass
class SharedEpisode:
    kind: str
    opponent: str
    learner_seat: int
    payoff_a: float                    # realized total payoff to seat A
    skills: Tuple[float, float]
    log: EpisodeLog

    @property
    def learner_return(self) -> float:
        """Return of the current network's seat (historical games; seat A in current-policy games)."""
        return self.payoff_a if self.learner_seat <= 0 else -self.payoff_a


class SharedCollector:
    """`n_envs` environments in lock-step, two trajectory slots each (slot 2i + seat). All trainable seats of a
    step are evaluated in one batched forward pass of the current network."""

    def __init__(self, game_cfg: GameConfig, cfg: PPOConfig, sampler: MatchupSampler, seed: int):
        if cfg.expected_gank_reward and game_cfg.champion_mode:
            raise ValueError("expected_gank_reward applies to legacy configurations only")
        check_time_semantics(game_cfg, cfg)
        self.cfg, self.game_cfg, self.sampler = cfg, game_cfg, sampler
        self._gank_reward = "expected" if cfg.expected_gank_reward else "realized"
        self._env_cache: List[Dict[Tuple[float, float], LaneEnv]] = [{} for _ in range(cfg.n_envs)]
        root = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(root.spawn(1)[0])          # matchups, seats, skills
        self._episode_seeds = root.spawn(1)[0]
        n = cfg.n_envs
        self.envs: List[Optional[LaneEnv]] = [None] * n
        self.match: List[Optional[Matchup]] = [None] * n
        self.agent_rng: List[Tuple[np.random.Generator, np.random.Generator]] = [None] * n
        self.obs = [None] * n
        self.logs: List[Optional[EpisodeLog]] = [None] * n
        self.skills: List[Tuple[float, float]] = [(game_cfg.skill_a, game_cfg.skill_b)] * n
        self.payoff = np.zeros(n)
        self.times = GameTimes(game_cfg, cfg, root, n)                # continuing time only (no draws otherwise)
        self._game_match: List[Optional[Matchup]] = [None] * n
        for i in range(n):
            self._reset(i)

    def _env(self, i: int, skills: Tuple[float, float]) -> LaneEnv:
        env = self._env_cache[i].get(skills)
        if env is None:
            game = self.game_cfg if skills == (self.game_cfg.skill_a, self.game_cfg.skill_b) else \
                self.game_cfg.with_(skill_a=skills[0], skill_b=skills[1])
            env = LaneEnv(game, max_turns=self.cfg.max_turns, cache_size=self.cfg.lane_cache,
                          gank_reward=self._gank_reward)
            self._env_cache[i][skills] = env
        return env

    def _reset(self, i: int) -> None:
        ss = self._episode_seeds.spawn(1)[0]
        _, env_ss, rng_a, rng_b = episode_streams(ss)
        self.match[i] = self.sampler.sample(self.rng)
        skills = (self.game_cfg.skill_a, self.game_cfg.skill_b)
        if self.cfg.skill_levels is not None:                      # per-episode skills, observed by both seats
            lv = self.cfg.skill_levels
            skills = (float(lv[int(self.rng.integers(len(lv)))]), float(lv[int(self.rng.integers(len(lv)))]))
        self.skills[i] = skills
        self.envs[i] = self._env(i, skills)
        self.agent_rng[i] = (rng_a, rng_b)
        self._game_match[i] = self.match[i]
        self.obs[i] = self.times.reset(i, self.envs[i], env_ss)
        names = tuple("learner" if c is LEARNER else c.name for c in self.match[i].controllers)
        self.logs[i] = EpisodeLog(seed=None, k0=0, horizon=self.envs[i].horizon, names=names)
        self.payoff[i] = 0.0

    def collect(self, net: PolicyValueNet, generator: torch.Generator):
        """One rollout of `n_steps` environment steps per environment with the network's current (fixed)
        parameters. Returns (time-major arrays over T x 2N slots, finished episodes, counts)."""
        cfg, T, N = self.cfg, self.cfg.rollout_length, self.cfg.n_envs
        S = 2 * N
        for i in range(N):
            if self.match[i] is not self._game_match[i]:
                raise RuntimeError(f"environment {i}: the matchup changed during a game; matchups may change only at "
                                   "game boundaries")
        m, n_act = self.envs[0].obs_size, self.envs[0].n_actions
        obs_buf = np.zeros((T, S, m), dtype=np.float32)
        mask_buf = np.zeros((T, S, n_act), dtype=bool)
        act_buf = np.zeros((T, S), dtype=np.int64)
        logp_buf = np.zeros((T, S), dtype=np.float32)
        val_buf = np.zeros((T, S), dtype=np.float64)
        rew_buf = np.zeros((T, S), dtype=np.float64)
        term_buf = np.zeros((T, S), dtype=bool)
        trunc_buf = np.zeros((T, S), dtype=bool)
        next_val = np.zeros((T, S), dtype=np.float64)
        active = np.zeros((T, S), dtype=bool)
        hist_slot = np.zeros((T, S), dtype=bool)                   # trainable slot of a historical game
        finished: List[SharedEpisode] = []
        for t in range(T):
            for i in range(N):
                ctrl = self.match[i].controllers
                for s in (0, 1):
                    if ctrl[s] is LEARNER:
                        j = 2 * i + s
                        o = self.obs[i][s]
                        obs_buf[t, j] = o.vector
                        mask_buf[t, j] = o.mask
                        active[t, j] = True
                        hist_slot[t, j] = self.match[i].kind == HISTORICAL
            idx = np.flatnonzero(active[t])
            with torch.no_grad():
                logits, values = net(torch.as_tensor(obs_buf[t, idx]))
                dist = MaskedCategorical(logits, torch.as_tensor(mask_buf[t, idx]))
                actions = dist.sample(generator)
                logp = dist.log_prob(actions)
            act_buf[t, idx], logp_buf[t, idx], val_buf[t, idx] = actions.numpy(), logp.numpy(), values.numpy()
            if t > 0:                                              # bootstrap of the previous step: same episode
                live = active[t - 1] & ~(term_buf[t - 1] | trunc_buf[t - 1])
                next_val[t - 1, live] = val_buf[t, live]
            trunc_final = []
            for i in range(N):
                env, ctrl = self.envs[i], self.match[i].controllers
                acts = [0, 0]
                for s in (0, 1):
                    acts[s] = int(act_buf[t, 2 * i + s]) if ctrl[s] is LEARNER else \
                        ctrl[s].act(self.obs[i][s], self.agent_rng[i][s])
                r = env.step(acts[0], acts[1])
                for s in (0, 1):
                    j = 2 * i + s
                    term_buf[t, j], trunc_buf[t, j] = r.terminated, r.truncated
                    if ctrl[s] is LEARNER:
                        rew_buf[t, j] = r.rewards[s] * cfg.reward_scale
                        if r.truncated:
                            trunc_final.append((j, r.obs[s].vector))
                self.logs[i].add(r)
                self.payoff[i] += r.rewards[0]
                self.obs[i] = r.obs
                if r.done:
                    mt = self.match[i]
                    finished.append(SharedEpisode(mt.kind, mt.opponent, mt.learner_seat, float(self.payoff[i]),
                                                  self.skills[i], self.logs[i]))
                    self._reset(i)
            if trunc_final:
                with torch.no_grad():
                    v = net.value(torch.as_tensor(np.stack([x for _, x in trunc_final])))
                for (j, _), vj in zip(trunc_final, v.numpy()):
                    next_val[t, j] = vj
        # bootstrap the last step of unfinished episodes with the value of the current observation
        live = active[T - 1] & ~(term_buf[T - 1] | trunc_buf[T - 1])
        if live.any():
            js = np.flatnonzero(live)
            last = np.stack([self.obs[j // 2][j % 2].vector for j in js])
            with torch.no_grad():
                next_val[T - 1, js] = net.value(torch.as_tensor(last)).numpy()
        counts = {"env_turns": T * N, "agent_transitions": int(active.sum()),
                  "transitions_historical": int(hist_slot.sum())}
        counts["transitions_current"] = counts["agent_transitions"] - counts["transitions_historical"]
        return {"obs": obs_buf, "mask": mask_buf, "actions": act_buf, "logprobs": logp_buf, "values": val_buf,
                "rewards": rew_buf, "terminated": term_buf, "truncated": trunc_buf, "next_values": next_val,
                "active": active}, finished, counts


def rollout_batch(roll: dict, cfg: PPOConfig) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    """GAE over every slot, then the flat PPO batch of the trainable entries only (with their advantages and
    returns, for diagnostics)."""
    adv, ret = compute_gae(roll["rewards"], roll["values"], roll["next_values"], roll["terminated"],
                           roll["truncated"], cfg.gamma, cfg.gae_lambda)
    a = roll["active"]
    batch = {"obs": torch.as_tensor(roll["obs"][a]), "mask": torch.as_tensor(roll["mask"][a]),
             "actions": torch.as_tensor(roll["actions"][a]), "logprobs": torch.as_tensor(roll["logprobs"][a]),
             "values": torch.as_tensor(roll["values"][a]).float(), "advantages": torch.as_tensor(adv[a]).float(),
             "returns": torch.as_tensor(ret[a]).float(),
             "choice": torch.as_tensor((roll["mask"][a].sum(-1) > 1).astype(np.float32))}
    return batch, adv[a], ret[a]


@dataclass
class SharedResult:
    net: PolicyValueNet
    pool: SnapshotPool
    history: List[dict] = field(default_factory=list)
    snapshot_paths: List[str] = field(default_factory=list)
    agent_transitions: int = 0
    env_turns: int = 0
    elapsed: float = 0.0


def _prepare_out_dir(out_dir: str, overwrite: bool) -> None:
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not overwrite:
        raise FileExistsError(f"{out_dir} exists and is not empty; choose a new run directory or pass overwrite=True")
    os.makedirs(os.path.join(out_dir, "snapshots"), exist_ok=True)


def train_shared(game_cfg: GameConfig, cfg: PPOConfig, scfg: SharedConfig, out_dir: Optional[str] = None,
                 net: Optional[PolicyValueNet] = None, name: str = "shared", verbose: bool = True,
                 overwrite: bool = False) -> SharedResult:
    """PPO with shared-policy self-play and a current / historical mixture (see the module docstring).
    `cfg.total_steps` is the budget in AGENT TRANSITIONS; `cfg.n_envs x cfg.n_steps` environment steps per
    rollout. With `out_dir`: run.json, history.jsonl, snapshots/ and final.pt."""
    scfg.validate()
    torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    env0 = LaneEnv(game_cfg)
    if net is None:
        net = PolicyValueNet(env0.obs_size, env0.n_actions, cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    pool = SnapshotPool(scfg.pool_size, keep_first=False)
    res = SharedResult(net, pool)
    hist_f = None
    meta_base = {"method": METHOD, "format_version": FORMAT_VERSION, "run": name, "seed": cfg.seed,
                 "ppo": cfg.to_dict(), "shared": scfg.to_dict(),
                 "model": {"class": "PolicyValueNet", "activation": "tanh", **net.spec()}}
    if out_dir:
        _prepare_out_dir(out_dir, overwrite)
        with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
            json.dump({**meta_base, "game_config": game_cfg.to_dict()}, f, indent=1)
        hist_f = open(os.path.join(out_dir, "history.jsonl"), "w", encoding="utf-8")

    def snapshot(update: int) -> None:
        sid = f"snap@{res.agent_transitions}"
        pool.add(net, res.agent_transitions, sid)
        if out_dir:
            path = os.path.join(out_dir, "snapshots", f"snap_{res.agent_transitions:09d}.pt")
            save_checkpoint(path, net, game_cfg, {**meta_base, "name": sid, "snapshot_id": sid, "kind": "snapshot",
                                                  "agent_transitions": res.agent_transitions,
                                                  "env_turns": res.env_turns, "update": update,
                                                  "step": res.agent_transitions}, exist_ok=overwrite)
            res.snapshot_paths.append(path)

    if scfg.snapshot_initial:
        snapshot(0)
    sampler = MatchupSampler(pool, scfg.current_fraction)
    collector = SharedCollector(game_cfg, cfg, sampler, cfg.seed + 1)
    next_snap = scfg.snapshot_every
    t0 = time.perf_counter()
    u = 0
    try:
        while res.agent_transitions < cfg.total_steps:
            frac_done = res.agent_transitions / cfg.total_steps
            if cfg.anneal_lr:
                for g in opt.param_groups:
                    g["lr"] = cfg.lr * (1.0 - frac_done)
            tc = time.perf_counter()
            roll, finished, counts = collector.collect(net, gen)
            t_collect = time.perf_counter() - tc
            batch, adv, ret = rollout_batch(roll, cfg)
            tu = time.perf_counter()
            net.train()
            stats = ppo_update(net, opt, batch, cfg, gen)
            net.eval()
            t_update = time.perf_counter() - tu
            u += 1
            res.agent_transitions += counts["agent_transitions"]
            res.env_turns += counts["env_turns"]
            cur = [e for e in finished if e.kind == CURRENT]
            hist = [e for e in finished if e.kind == HISTORICAL]
            by_opp: Dict[str, List[float]] = {}
            for e in hist:
                by_opp.setdefault(e.opponent, []).append(e.learner_return)
            entry = {"update": u, "agent_transitions": res.agent_transitions, "env_turns": res.env_turns,
                     "steps": res.agent_transitions, "elapsed": time.perf_counter() - t0,
                     **{f"rollout_{k}": v for k, v in counts.items()},
                     "games_current": len(cur), "games_historical": len(hist),
                     "current_seat_a_return": float(np.mean([e.payoff_a for e in cur])) if cur else float("nan"),
                     "historical_return": float(np.mean([e.learner_return for e in hist])) if hist else float("nan"),
                     "historical_by_opponent": {k: float(np.mean(v)) for k, v in by_opp.items()},
                     "explained_variance": explained_variance(batch["values"].numpy().astype(np.float64), ret),
                     "sps": counts["agent_transitions"] / (t_collect + t_update), "collect_s": t_collect,
                     "update_s": t_update, "lr": opt.param_groups[0]["lr"], "pool": [a.name for a in pool.snapshots],
                     **stats}
            if res.agent_transitions >= next_snap:
                snapshot(u)
                while next_snap <= res.agent_transitions:
                    next_snap += scfg.snapshot_every
            res.history.append(entry)
            if hist_f:
                hist_f.write(json.dumps(entry) + "\n")
                hist_f.flush()
            if verbose and (u == 1 or u % 20 == 0 or res.agent_transitions >= cfg.total_steps):
                print(f"[{name}] update {u} transitions {res.agent_transitions} turns {res.env_turns} "
                      f"hist {len(hist)}/{len(finished)} ent {stats['entropy']:.2f} kl {stats['approx_kl']:.4f} "
                      f"ev {entry['explained_variance']:.2f} {entry['sps']:.0f} sps", flush=True)
    finally:
        if hist_f:
            hist_f.close()
    res.elapsed = time.perf_counter() - t0
    if out_dir:
        save_checkpoint(os.path.join(out_dir, "final.pt"), net, game_cfg,
                        {**meta_base, "name": name, "kind": "final", "snapshot_id": f"final@{res.agent_transitions}",
                         "agent_transitions": res.agent_transitions, "env_turns": res.env_turns, "update": u,
                         "steps": res.agent_transitions, "elapsed_s": res.elapsed,
                         "opponents": sampler.describe(), "time_semantics": collector.times.describe()},
                        exist_ok=overwrite)
    return res
