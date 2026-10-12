"""Discounted evaluation of continuing-time games (M4 design §10.5; plan WP1). NumPy only.

    evaluate_discounted(cfg, x, y, episodes, seed, window=1000, protocol="E1")

estimates J_W = E[sum_{k < W} gamma^k r_A,k] for x against y, both seats on the SAME episode seeds (common random
numbers, as `rl.evaluate.head_to_head`), and reports the payoff to x.

Protocols (design §10.5.1), never mixed:
  E1 (primary, every headline number): from the standard initial state s0 only; any other start is rejected.
  E2 (secondary, diagnostic): the mean over a held-out archive A_eval of witnessed states, which must be
     mirror-closed and disjoint from the training archive A_train; reported separately and labelled.

Error terms, reported separately:
  * the Monte Carlo error: a paired 95% confidence interval over seeds (each seed's value is the mean of x's
    discounted payoff in seat A and in seat B);
  * the truncation error |J - J_W| <= gamma^W Omega, a GUARANTEE derived from the simulator (design §10.5.3):
        Omega = a + b / (1 - gamma) + bounty / (1 - gamma^(death_turns + 1)) + n_plates * plate_gold,
        a = f_max (A_max + wave_hp (s - 1) / s),  b = f_max wave_hp / s,  s = spawn_every_turns,
    with f_max the largest per-HP farm value and A_max = 32 the enumerated largest alive minion HP of one side
    (lanegame.lane_enum). At gamma = 0.99, Omega = 68.3: the bound is 0.0030 at W = 1000 and 0.671 at W = 460.
  * optionally (tail_episodes > 0) an EMPIRICAL tail estimate on a subsample extended to tail_window turns. It is
    labelled as an estimate: it is not a guarantee.

Stateful agents (e.g. agents_m4.AdaptiveBreaker) get per-game lifecycle hooks when they define them: `reset()` before
every game and `notify(info, side)` after every turn with the public turn result (`game_agents`).

W = 1000 is the headline window; W = 460 is for development runs only and must be reported with its guaranteed
bound and an empirical tail estimate. Critic bootstrapping is not part of this evaluator (design §10.5.4).
"""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import GameConfig, exact_safe_base
from .agents import episode_streams
from .core import LaneEnv

Z95 = 1.96
HEADLINE_WINDOW = 1000
DEV_WINDOW = 460


def max_farm_per_hp(cfg: GameConfig) -> float:
    """The largest resources per enemy HP a champion can collect (f * max base efficiency over reachable cells)."""
    T = cfg.tower
    q = lambda v: Fraction(repr(float(v)))
    bases = [q(cfg.collect_contest), q(cfg.collect_contest_enemy_tower)]
    bases += [exact_safe_base(x, cfg) for x in range(-T, T + cfg.thin_reach + 1)]
    return float(q(cfg.farm_per_hp) * max(bases))


def tail_constant(cfg: GameConfig, gamma: Optional[float] = None, a_max: Optional[int] = None) -> float:
    """Omega of design §10.5.3 (the truncation bound is gamma^W Omega)."""
    from ..lane_enum import recorded_bounds
    g = cfg.gamma if gamma is None else gamma
    if a_max is None:
        a_max = recorded_bounds(cfg)["A_max"]
    f = max_farm_per_hp(cfg)
    s = cfg.spawn_every_turns
    a = f * (a_max + cfg.wave_hp * (s - 1) / s)
    b = f * cfg.wave_hp / s
    plates = len(cfg.plate_thresholds) * cfg.plate_gold if cfg.is_m4 else 0.0
    return a + b / (1.0 - g) + cfg.kill_bounty / (1.0 - g ** (cfg.death_turns + 1)) + plates


def tail_bound(cfg: GameConfig, window: int, gamma: Optional[float] = None) -> float:
    g = cfg.gamma if gamma is None else gamma
    return g ** window * tail_constant(cfg, g)


def per_turn_bound(cfg: GameConfig) -> float:
    """A guaranteed bound on |r_k| for one turn (design §10.5.3): farm on at most 2 hp_cap fight HP, tower fire and
    thinning, plus the plates one turn can cross, plus one bounty."""
    from ..progression import tables
    from ..mechanics import minion_damage
    f = max_farm_per_hp(cfg)
    kills = cfg.ticks_per_turn * cfg.hp_cap + cfg.ticks_per_turn * cfg.tower_siege_damage + cfg.thin_damage
    if cfg.wave_dynamics == "natural":
        # per tick: the lane front and the crashed mass each lose at most the Lanchester damage of a blob of hp_cap
        # plus rear_count_max queued waves (observe_m4 raises beyond it), plus the tower; the champion strikes once
        blob = minion_damage(cfg.hp_cap + cfg.rear_count_max * cfg.wave_hp, cfg)
        per_tick = 2 * blob + cfg.tower_damage + cfg.tower_siege_damage + max(cfg.champion_wave_damage)
        kills = cfg.ticks_per_turn * per_tick + cfg.thin_damage
    plates = 0
    if cfg.is_m4:
        t = tables(cfg)
        th = cfg.plate_thresholds
        if cfg.continuous_items:
            # attack power grows without bound with the investment: one Pressure turn may cross every plate
            plates = len(th)
        else:
            add = cfg.ticks_per_turn * 100 * minion_damage(cfg.siege_cap, cfg) + t.pressure[t.level_cap][t.item_tiers]
            plates = max(sum(1 for j in range(i, len(th)) if th[j] - th[i] <= add - 1) for i in range(len(th)))
    return f * kills + plates * cfg.plate_gold + cfg.kill_bounty


# ---------------------------------------------------------------------- games
def _stateful(agent) -> bool:
    return callable(getattr(agent, "reset", None)) or callable(getattr(agent, "notify", None))


def game_agents(agents) -> tuple:
    """The two seats' agents for one game, with per-game lifecycle hooks: an agent with `reset()` is reset before the
    game, and `notify(info, side)` (called after every turn by `_play`) gives it the public turn result. A stateful
    agent that plays BOTH seats is copied for seat B, so the seats never share memory."""
    import copy
    a, b = agents
    if a is b and _stateful(a):
        b = copy.deepcopy(a)
    for ag in (a, b):
        if callable(getattr(ag, "reset", None)):
            ag.reset()
    return a, b


def _play(env: LaneEnv, agents, seed, turns: int, start=None, record_every: int = 0, on_step=None):
    """One game of `turns` turns from s0 or `start` (a StartState); returns (rewards to A, snapshots). Agents with
    `reset()` / `notify(info, side)` get them per game and per turn (`game_agents`); `on_step(info)` observes every
    turn's result (counters)."""
    ss, env_ss, rng_a, rng_b = episode_streams(seed)
    rngs = (rng_a, rng_b)
    agents = game_agents(agents)
    notify = [getattr(ag, "notify", None) for ag in agents]
    env.max_turns = turns
    if start is None:
        obs = env.reset(seed=env_ss)
    else:
        obs = env.reset(seed=env_ss, root=start.lane, k0=start.phase, champ=start.champ)
    rewards = np.zeros(turns)
    snaps = []
    t = 0
    while True:
        a = agents[0].act(obs[0], rngs[0])
        b = agents[1].act(obs[1], rngs[1])
        r = env.step(a, b)
        for side in (0, 1):
            if callable(notify[side]):
                notify[side](r.info, side)
        if on_step is not None:
            on_step(r.info)
        rewards[t] = r.rewards[0]
        t += 1
        if record_every and t % record_every == 0 and env.champ is not None and hasattr(env.champ.a, "xp"):
            snaps.append((t, env.champ))
        if r.done:
            break
        obs = r.obs
    return rewards[:t], snaps


def discounted_sum(rewards: np.ndarray, gamma: float, start: int = 0) -> float:
    k = np.arange(start, start + len(rewards))
    return float(np.sum(rewards * gamma ** k))


def evaluate_discounted(cfg: GameConfig, x, y, episodes: int = 200, seed: int = 31_337, gamma: Optional[float] = None,
                        window: int = HEADLINE_WINDOW, protocol: str = "E1", archive: Optional[Sequence] = None,
                        train_archive: Optional[Sequence] = None, skill: Optional[Tuple[float, float]] = None,
                        tail_episodes: int = 0, tail_window: int = HEADLINE_WINDOW, block: int = 64,
                        cache_size: int = 20_000) -> dict:
    """Discounted payoff to x against y (design §10.5); see the module docstring. E2 runs `episodes` seeds from every
    archived state (both seats)."""
    if not cfg.continuing:
        raise ValueError("evaluate_discounted applies to continuing-time rulesets; use head_to_head for finite games")
    g = cfg.gamma if gamma is None else gamma
    if g != cfg.gamma:
        raise ValueError(f"the evaluator must use the game's gamma {cfg.gamma} (design §10.2); got {g}")
    if protocol == "E1":
        if archive is not None:
            raise ValueError("protocol E1 evaluates from the standard initial state s0 only; an archive is E2")
        starts = [None]
    elif protocol == "E2":
        from .starts import disjoint, mirror_closed
        if not archive:
            raise ValueError("protocol E2 needs the held-out archive A_eval")
        if not mirror_closed(archive):
            raise ValueError("protocol E2: A_eval must be mirror-closed")
        if train_archive is None:
            raise ValueError("protocol E2: pass the training archive A_train (possibly empty) to check disjointness")
        if not disjoint(archive, train_archive):
            raise ValueError("protocol E2: A_eval overlaps the training archive A_train")
        starts = list(archive)
    else:
        raise ValueError("protocol must be 'E1' or 'E2'")
    cfg_a, cfg_b = cfg, cfg
    if skill is not None:
        cfg_a = cfg.with_(skill_a=skill[0], skill_b=skill[1])
        cfg_b = cfg.with_(skill_a=skill[1], skill_b=skill[0])
    env_a, env_b = LaneEnv(cfg_a, cache_size=cache_size), LaneEnv(cfg_b, cache_size=cache_size)
    t0 = time.perf_counter()
    children = np.random.SeedSequence(seed).spawn(episodes)
    pair_vals, seat_a, seat_b, blocks_a, blocks_b = [], [], [], [], []
    for si, st in enumerate(starts):
        for m, ss in enumerate(children):
            # x in seat A, then x in seat B, from the same start state on the same seed (paired); with a
            # mirror-closed archive the two seats see every situation from both sides
            ra, _ = _play(env_a, (x, y), np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (si, 0)), window, st)
            rb, _ = _play(env_b, (y, x), np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (si, 0)), window,
                          st)
            va, vb = discounted_sum(ra, g), -discounted_sum(rb, g)
            seat_a.append(va)
            seat_b.append(vb)
            pair_vals.append(0.5 * (va + vb))
            nb = window // block
            blocks_a.append([float(ra[i * block:(i + 1) * block].sum()) for i in range(nb)])
            blocks_b.append([float(-rb[i * block:(i + 1) * block].sum()) for i in range(nb)])
    pv = np.asarray(pair_vals)
    mean = float(pv.mean())
    se = float(pv.std(ddof=1) / np.sqrt(len(pv))) if len(pv) > 1 else float("nan")
    omega = tail_constant(cfg, g)
    out = {"protocol": protocol, "x": getattr(x, "name", "x"), "y": getattr(y, "name", "y"), "gamma": g,
           "window": window, "episodes_per_start": episodes, "starts": len(starts), "pairs": len(pv), "seed": seed,
           "estimate": mean, "stderr": se, "ci95": [mean - Z95 * se, mean + Z95 * se],
           "seat_A": float(np.mean(seat_a)), "seat_B": float(np.mean(seat_b)),
           "tail_bound": {"guaranteed": g ** window * omega, "omega": omega,
                          "note": "|J - J_W| <= gamma^W Omega (design §10.5.3), a deterministic guarantee"},
           "undiscounted_block_means": {"block": block,
                                        "x_seat_A": np.mean(blocks_a, axis=0).tolist() if blocks_a else [],
                                        "x_seat_B": np.mean(blocks_b, axis=0).tolist() if blocks_b else []},
           "skill": list(skill) if skill else [cfg.skill_a, cfg.skill_b]}
    if tail_episodes > 0:
        if tail_window <= window:
            raise ValueError("tail_window must exceed window")
        tails = []
        for m, ss in enumerate(children[:tail_episodes]):
            st = starts[0]
            ra, _ = _play(env_a, (x, y), np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (0, 1)),
                          tail_window, st)
            rb, _ = _play(env_b, (y, x), np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (0, 1)),
                          tail_window, st)
            tails.append(0.5 * (discounted_sum(ra[window:], g, window) - discounted_sum(rb[window:], g, window)))
        tv = np.asarray(tails)
        out["empirical_tail"] = {"label": "EMPIRICAL ESTIMATE, not a guarantee", "episodes": tail_episodes,
                                 "tail_window": tail_window, "mean": float(tv.mean()),
                                 "mean_abs": float(np.abs(tv).mean()), "max_abs": float(np.abs(tv).max())}
    if protocol == "E2":
        out["label"] = "E2 (held-out archive, diagnostic): never averaged into E1 or used to rank methods"
    out["elapsed_s"] = time.perf_counter() - t0
    return out
