"""Experience collection and the PPO training loop.

One learner network plays against opponents drawn from an `OpponentSampler` (fixed scripted agents in M3a; a
mixture of scripted agents and frozen snapshots in M3b). Each training environment runs one episode at a time;
at every reset it draws a fresh opponent and a seat (A or B, uniformly), so the shared own-perspective network
learns to play both seats. The learner's reward is its own zero-sum game reward (r for seat A, -r for seat B),
times `reward_scale` for learning; every reported return is unscaled.

Reproducibility: everything random derives from `PPOConfig.seed`: torch's initialization and sampling generator,
the opponent/seat (and, with `skill_levels`, skill) generator, and one SeedSequence child per episode (split as in `lanegame.env.play_episode`
into the environment's chance stream and the two agents' streams). With `torch_threads = 1` a run is
bit-for-bit repeatable on one machine.

Continuing time (rulesets "m2c" / "m4", M4 design §10.4): rollouts are segments of `PPOConfig.rollout_length` turns,
a game is reset only at its time limit (a truncation bootstrapped from the PRE-reset final observation), the first
game of each environment has a seeded random length in [1, game_length] so that the batch is not lock-stepped in
game age, and games start from rho_train (s0, or with probability start_mix a uniformly drawn archived state). The
opponent of a game is drawn when the game starts and may not change before it ends: `collect` raises otherwise.
The extra generators of continuing time are spawned after the legacy ones, so legacy runs draw exactly as before.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import GameConfig
from ..env import LaneEnv
from ..env.agents import episode_streams
from ..env.trace import EpisodeLog
from .agent import NetAgent, save_checkpoint
from .network import MaskedCategorical, PolicyValueNet
from .ppo import PPOConfig, check_time_semantics, compute_gae, explained_variance, ppo_update


class OpponentSampler:
    """Draws an opponent agent per episode: `agents[i]` with probability `weights[i]` (uniform by default)."""

    def __init__(self, agents: Sequence, weights: Optional[Sequence[float]] = None):
        self.agents = list(agents)
        w = np.ones(len(self.agents)) if weights is None else np.asarray(weights, dtype=float)
        self.weights = w / w.sum()

    def sample(self, rng: np.random.Generator):
        return self.agents[int(rng.choice(len(self.agents), p=self.weights))]

    def describe(self) -> dict:
        return {getattr(a, "name", str(a)): float(w) for a, w in zip(self.agents, self.weights)}


@dataclass
class EpisodeRecord:
    opponent: str
    seat: int
    learner_return: float
    log: EpisodeLog


class Collector:
    """`n_envs` environments stepped in lock-step; the learner's actions come from one batched forward pass."""

    def __init__(self, game_cfg: GameConfig, cfg: PPOConfig, sampler: OpponentSampler, seed: int):
        check_time_semantics(game_cfg, cfg)
        self.cfg, self.game_cfg, self.sampler = cfg, game_cfg, sampler
        self._gank_reward = "expected" if cfg.expected_gank_reward else "realized"
        self.envs = [LaneEnv(game_cfg, max_turns=cfg.max_turns, cache_size=cfg.lane_cache, gank_reward=self._gank_reward)
                     for _ in range(cfg.n_envs)]
        root = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(root.spawn(1)[0])          # opponents and seats
        self._episode_seeds = root.spawn(1)[0]
        self._episodes_started = 0
        n = cfg.n_envs
        self.times = GameTimes(game_cfg, cfg, root, n)                # continuing time only (no draws otherwise)
        self._game_opp = [None] * n
        self.opp, self.seat, self.opp_rng = [None] * n, [0] * n, [None] * n
        self.obs = [None] * n                                         # (obs_A, obs_B) per environment
        self.logs: List[Optional[EpisodeLog]] = [None] * n
        self.ret = np.zeros(n)
        for i in range(n):
            self._reset(i)

    def _reset(self, i: int) -> None:
        ss = self._episode_seeds.spawn(1)[0]
        self._episodes_started += 1
        _, env_ss, rng_a, rng_b = episode_streams(ss)
        self.opp[i] = self.sampler.sample(self.rng)
        self.seat[i] = int(self.rng.integers(2))
        if self.cfg.skill_levels is not None:                        # per-episode skills (both observed)
            lv = self.cfg.skill_levels
            game = self.game_cfg.with_(skill_a=float(lv[int(self.rng.integers(len(lv)))]),
                                       skill_b=float(lv[int(self.rng.integers(len(lv)))]))
            self.envs[i] = LaneEnv(game, max_turns=self.cfg.max_turns, cache_size=self.cfg.lane_cache,
                                   gank_reward=self._gank_reward)
        self.opp_rng[i] = rng_b if self.seat[i] == 0 else rng_a
        self._game_opp[i] = self.opp[i]
        self.obs[i] = self.times.reset(i, self.envs[i], env_ss)
        names = ("learner", getattr(self.opp[i], "name", "opponent"))
        self.logs[i] = EpisodeLog(seed=None, k0=0, horizon=self.envs[i].horizon,
                                  names=names if self.seat[i] == 0 else names[::-1])
        self.ret[i] = 0.0

    def collect(self, net: PolicyValueNet, generator: torch.Generator):
        cfg, T, N = self.cfg, self.cfg.rollout_length, self.cfg.n_envs
        for i in range(N):
            if self.opp[i] is not self._game_opp[i]:
                raise RuntimeError(f"environment {i}: the opponent changed during a game; matchups may change only "
                                   "at game boundaries")
        m, n_act = self.envs[0].obs_size, self.envs[0].n_actions
        obs_buf = np.zeros((T, N, m), dtype=np.float32)
        mask_buf = np.zeros((T, N, n_act), dtype=bool)
        act_buf = np.zeros((T, N), dtype=np.int64)
        logp_buf = np.zeros((T, N), dtype=np.float32)
        val_buf = np.zeros((T, N), dtype=np.float64)
        rew_buf = np.zeros((T, N), dtype=np.float64)
        term_buf = np.zeros((T, N), dtype=bool)
        trunc_buf = np.zeros((T, N), dtype=bool)
        next_val = np.zeros((T, N), dtype=np.float64)
        finished: List[EpisodeRecord] = []
        for t in range(T):
            for i in range(N):
                o = self.obs[i][self.seat[i]]
                obs_buf[t, i] = o.vector
                mask_buf[t, i] = o.mask
            with torch.no_grad():
                logits, values = net(torch.as_tensor(obs_buf[t]))
                dist = MaskedCategorical(logits, torch.as_tensor(mask_buf[t]))
                actions = dist.sample(generator)
                logp = dist.log_prob(actions)
            act_buf[t], logp_buf[t], val_buf[t] = actions.numpy(), logp.numpy(), values.numpy()
            if t > 0:
                live = ~(term_buf[t - 1] | trunc_buf[t - 1])
                next_val[t - 1, live] = val_buf[t, live]
            trunc_final = []
            for i in range(N):
                env, seat = self.envs[i], self.seat[i]
                mine = int(act_buf[t, i])
                theirs = self.opp[i].act(self.obs[i][1 - seat], self.opp_rng[i])
                r = env.step(*((mine, theirs) if seat == 0 else (theirs, mine)))
                rew = r.rewards[seat]
                rew_buf[t, i] = rew * cfg.reward_scale
                term_buf[t, i], trunc_buf[t, i] = r.terminated, r.truncated
                self.logs[i].add(r)
                self.ret[i] += rew
                self.obs[i] = r.obs
                if r.done:
                    if r.truncated:
                        trunc_final.append((i, r.obs[seat].vector))
                    finished.append(EpisodeRecord(getattr(self.opp[i], "name", "opponent"), seat, float(self.ret[i]),
                                                  self.logs[i]))
                    self._reset(i)
            if trunc_final:
                with torch.no_grad():
                    v = net.value(torch.as_tensor(np.stack([x for _, x in trunc_final])))
                for (i, _), vi in zip(trunc_final, v.numpy()):
                    next_val[t, i] = vi
        # bootstrap the last step of unfinished episodes with the value of the current observation
        live = ~(term_buf[T - 1] | trunc_buf[T - 1])
        if live.any():
            last = np.stack([self.obs[i][self.seat[i]].vector for i in range(N)])
            with torch.no_grad():
                v = net.value(torch.as_tensor(last)).numpy()
            next_val[T - 1, live] = v[live]
        return {"obs": obs_buf, "mask": mask_buf, "actions": act_buf, "logprobs": logp_buf, "values": val_buf,
                "rewards": rew_buf, "terminated": term_buf, "truncated": trunc_buf, "next_values": next_val}, finished


class GameTimes:
    """Per-game time limits and start states under continuing time (M4 design §10.4); a pass-through reset for
    finite time. The first game of environment i is `first_lengths[i]` turns long, a seeded draw from
    [1, game_length]; every later game lasts game_length turns. With start_mix = beta > 0 a game starts from a
    uniformly drawn archived (witnessed) state with probability beta, else from s0."""

    def __init__(self, game_cfg: GameConfig, cfg: PPOConfig, root: np.random.SeedSequence, n_envs: int):
        self.continuing = game_cfg.continuing
        self.games_started = [0] * n_envs
        self.first_lengths: List[int] = []
        self.archive = []
        self.start_counts = {"initial": 0, "archive": 0}
        if not self.continuing:
            return
        self.game_length = int(cfg.game_length)
        len_ss, start_ss = root.spawn(2)
        rng = np.random.default_rng(len_ss)
        self.first_lengths = [int(x) for x in rng.integers(1, self.game_length + 1, size=n_envs)]
        self.start_rng = np.random.default_rng(start_ss)
        self.start_mix = float(cfg.start_mix)
        if cfg.start_archive:
            from ..env.starts import load_archive
            self.archive = load_archive(cfg.start_archive, game_cfg)
        if self.start_mix > 0 and not self.archive:
            raise ValueError("start_mix > 0 needs a non-empty start_archive")

    def reset(self, i: int, env: LaneEnv, env_seed):
        if not self.continuing:
            return env.reset(seed=env_seed)
        env.max_turns = self.first_lengths[i] if self.games_started[i] == 0 else self.game_length
        self.games_started[i] += 1
        if self.start_mix > 0 and self.start_rng.random() < self.start_mix:
            st = self.archive[int(self.start_rng.integers(len(self.archive)))]
            self.start_counts["archive"] += 1
            return env.reset(seed=env_seed, root=st.lane, k0=st.phase, champ=st.champ)
        self.start_counts["initial"] += 1
        return env.reset(seed=env_seed)

    def describe(self) -> dict:
        if not self.continuing:
            return {"time": "finite"}
        return {"time": "continuing", "game_length": self.game_length, "first_lengths": self.first_lengths,
                "start_mix": self.start_mix, "archive_states": len(self.archive), "starts": dict(self.start_counts)}


@dataclass
class TrainResult:
    net: PolicyValueNet
    history: List[dict] = field(default_factory=list)
    episodes: List[dict] = field(default_factory=list)       # per finished training episode: opponent, seat, return
    elapsed: float = 0.0


def train_ppo(game_cfg: GameConfig, cfg: PPOConfig, sampler: OpponentSampler, out_dir: Optional[str] = None,
              net: Optional[PolicyValueNet] = None, on_update: Optional[Callable] = None, verbose: bool = True,
              name: str = "ppo", eval_fn: Optional[Callable] = None, eval_every: int = 0) -> TrainResult:
    """Train `net` (a fresh network if None) with PPO against `sampler`'s opponents.

    `on_update(update, net, entry)` is called after every update (M3b adds snapshots there); `eval_fn(net)`
    every `eval_every` updates (and after the last) returns a dict stored in the history. With `out_dir`, the
    history is written to history.jsonl and the final network to final.pt."""
    torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    env0 = LaneEnv(game_cfg)
    if net is None:
        net = PolicyValueNet(env0.obs_size, env0.n_actions, cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    collector = Collector(game_cfg, cfg, sampler, cfg.seed + 1)
    res = TrainResult(net)
    t0 = time.perf_counter()
    hist_f = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        hist_f = open(os.path.join(out_dir, "history.jsonl"), "w", encoding="utf-8")
    steps = 0
    try:
        for u in range(cfg.n_updates):
            if cfg.anneal_lr:
                for g in opt.param_groups:
                    g["lr"] = cfg.lr * (1.0 - u / cfg.n_updates)
            tc = time.perf_counter()
            roll, finished = collector.collect(net, gen)
            t_collect = time.perf_counter() - tc
            adv, ret = compute_gae(roll["rewards"], roll["values"], roll["next_values"], roll["terminated"],
                                   roll["truncated"], cfg.gamma, cfg.gae_lambda)
            flat = lambda x: torch.as_tensor(x.reshape(-1, *x.shape[2:]))
            batch = {"obs": flat(roll["obs"]), "mask": flat(roll["mask"]), "actions": flat(roll["actions"]),
                     "logprobs": flat(roll["logprobs"]), "values": flat(roll["values"]).float(),
                     "advantages": flat(adv).float(), "returns": flat(ret).float(),
                     "choice": flat((roll["mask"].sum(-1) > 1).astype(np.float32))}
            tu = time.perf_counter()
            net.train()
            stats = ppo_update(net, opt, batch, cfg, gen)
            t_update = time.perf_counter() - tu
            steps += cfg.batch_size
            rets = [e.learner_return for e in finished]
            entry = {"update": u + 1, "steps": steps, "elapsed": time.perf_counter() - t0,
                     "episodes": len(finished), "mean_return": float(np.mean(rets)) if rets else float("nan"),
                     "explained_variance": explained_variance(roll["values"].ravel(), ret.ravel()),
                     "sps": cfg.batch_size / (t_collect + t_update), "collect_s": t_collect, "update_s": t_update,
                     "lr": opt.param_groups[0]["lr"], **stats}
            by_opp: Dict[str, List[float]] = {}
            for e in finished:
                by_opp.setdefault(e.opponent, []).append(e.learner_return)
                res.episodes.append({"update": u + 1, "opponent": e.opponent, "seat": e.seat, "return": e.learner_return})
            entry["return_by_opponent"] = {k: float(np.mean(v)) for k, v in by_opp.items()}
            if eval_fn is not None and eval_every and ((u + 1) % eval_every == 0 or u + 1 == cfg.n_updates):
                net.eval()
                entry["eval"] = eval_fn(net)
            if on_update is not None:
                on_update(u + 1, net, entry)
            res.history.append(entry)
            if hist_f:
                hist_f.write(json.dumps(entry) + "\n")
                hist_f.flush()
            if verbose and ((u + 1) % max(1, cfg.n_updates // 10) == 0 or u == 0):
                ev = entry.get("eval")
                print(f"[{name}] update {u + 1}/{cfg.n_updates} steps {steps} return {entry['mean_return']:+.2f} "
                      f"ent {stats['entropy']:.2f} kl {stats['approx_kl']:.4f} ev {entry['explained_variance']:.2f} "
                      f"{entry['sps']:.0f} sps" + (f" eval {ev}" if ev else ""), flush=True)
    finally:
        if hist_f:
            hist_f.close()
    res.elapsed = time.perf_counter() - t0
    if out_dir:
        save_checkpoint(os.path.join(out_dir, "final.pt"), net, game_cfg,
                        {"name": name, "ppo": cfg.to_dict(), "steps": steps, "opponents": sampler.describe(),
                         "elapsed_s": res.elapsed, "time_semantics": collector.times.describe()})
    return res
