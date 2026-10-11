"""Digest functions behind tests/test_legacy_pins.py (M4 plan WP0).

Every function returns a SHA-256 hex digest of a deterministic, platform-stable serialization of legacy outputs.
The expected digests in test_legacy_pins.py were computed on the unchanged engine (commit 085908d, before any M4
code); a later change that alters any legacy output changes a digest.

Serialization: Python `repr` of tuples of ints / floats / strings (floats with repr's shortest round-trip form),
and raw float32 bytes of observation vectors.
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterator, List, Tuple

import numpy as np

from lanegame import GameConfig
from lanegame.mechanics import initial_state, legal_actions, step_turn

N_TUPLES = 2000
PRESETS = {"v1": GameConfig.v1, "v2": GameConfig.v2, "siege": GameConfig, "champions": GameConfig.champions}


def _h() -> "hashlib._Hash":
    return hashlib.sha256()


def _turn_repr(res) -> str:
    """Every legacy field of a TurnResult except the optional tick trace."""
    kills = tuple(tuple(sorted(k.items())) for k in res.kills)
    events = tuple(sorted((k, tuple(v)) for k, v in res.events.items()))
    return repr((res.state.to_list(), res.crash_hp, events, res.exposure, kills, res.plates, res.thinned,
                 res.vision_set))


def lane_tuples(cfg: GameConfig, n: int = N_TUPLES, seed: int = 20261011) -> Iterator[Tuple]:
    """n (state, k, a, b) tuples from seeded random legal walks of the lane (restart every horizon turns)."""
    rng = random.Random(seed)
    s, k = initial_state(cfg), 0
    for _ in range(n):
        a = rng.choice(legal_actions(s, 0, cfg))
        b = rng.choice(legal_actions(s, 1, cfg))
        yield s, k, a, b
        s = step_turn(s, k, a, b, cfg).state
        k += 1
        if k >= cfg.horizon:
            s, k = initial_state(cfg), 0


def lane_digest(name: str) -> str:
    """step_turn outputs and both seats' observe() vectors (legacy lane configurations)."""
    from lanegame.env.observation import observe
    cfg = PRESETS[name]()
    h = _h()
    reach_rng = random.Random(7)
    for s, k, a, b in lane_tuples(cfg):
        res = step_turn(s, k, a, b, cfg)
        h.update(_turn_repr(res).encode())
        if cfg.champion_mode:          # the champion lane step under explicit reach (observe needs a champion state)
            reach = (reach_rng.choice((-10 ** 6, -2, -1, 0, 1, 2)), reach_rng.choice((-10 ** 6, -2, -1, 0, 1, 2)))
            h.update(_turn_repr(step_turn(s, k, a, b, cfg, champion_reach=reach)).encode())
            continue
        h.update(observe(s, k, 0, cfg).tobytes())
        h.update(observe(s, k, 1, cfg).tobytes())
    return h.hexdigest()


def _champ_turn_repr(ct) -> str:
    fields = []
    for f in ct.__dataclass_fields__:
        if f == "lane":
            fields.append(_turn_repr(ct.lane))
        elif f == "champ":
            fields.append(repr(ct.champ.to_list()))
        else:
            fields.append(repr(getattr(ct, f)))
    return "|".join(fields)


def champion_digest(n: int = N_TUPLES, seed: int = 20261011) -> str:
    """champions(): lane step under every champion reach of the walk, resolve_turn with random chance vectors,
    champion legal sets and both observations."""
    from lanegame.champions import N_CHANCE, champion_legal_actions, initial_champions, resolve_turn
    from lanegame.env.observation import observe
    cfg = GameConfig.champions()
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    lane, champ, k = initial_state(cfg), initial_champions(cfg), 0
    h = _h()
    for _ in range(n):
        la = champion_legal_actions(lane, champ, 0, cfg)
        lb = champion_legal_actions(lane, champ, 1, cfg)
        a, b = rng.choice(la), rng.choice(lb)
        u = tuple(float(x) for x in nrng.random(N_CHANCE))
        ct = resolve_turn(lane, champ, k, a, b, cfg, u=u)
        h.update(repr((la, lb)).encode())
        h.update(_champ_turn_repr(ct).encode())
        h.update(observe(lane, k, 0, cfg, champ).tobytes())
        h.update(observe(lane, k, 1, cfg, champ).tobytes())
        lane, champ, k = ct.lane.state, ct.champ, k + 1
        if k >= cfg.horizon:
            lane, champ, k = initial_state(cfg), initial_champions(cfg), 0
    return h.hexdigest()


def rollout_digest(episodes: int = 200, seed: int = 4242) -> str:
    """A 200-episode seeded champions() rollout digest (rewards and info fields) over scripted-suite pairings."""
    from lanegame.env import LaneEnv, champion_suite, play_episode
    cfg = GameConfig.champions()
    env = LaneEnv(cfg)
    suite = champion_suite()
    h = _h()
    children = np.random.SeedSequence(seed).spawn(episodes)
    keys = ("reward", "farm", "farm_available", "plates", "bounty", "died", "ganked", "gank_draws", "chance",
            "behaviors", "champ_after", "state_after", "trade_damage", "damage_taken", "exposure", "kills")
    for e, ss in enumerate(children):
        x, y = suite[e % len(suite)], suite[(e * 7 + 3) % len(suite)]
        log = play_episode(env, x, y, ss, record=True)
        for info in log.turns:
            h.update(repr(tuple(_jsonable(info[k]) for k in keys)).encode())
        h.update(repr(log.payoff).encode())
    return h.hexdigest()


def _jsonable(x):
    if isinstance(x, dict):
        return tuple(sorted((k, _jsonable(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_jsonable(v) for v in x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


M3B_CHECKPOINT = "results/rl/checkpoints/m3b_equal_s0/final.pt"


def checkpoint_digest(path: str = M3B_CHECKPOINT, episodes: int = 40, seed: int = 10_000) -> str:
    """One M3b checkpoint's head_to_head against hold_farmer at fixed seeds (both seats), bit-identical."""
    from lanegame.env import champion_suite
    from lanegame.rl import head_to_head, load_agent
    cfg = GameConfig.champions()
    agent = load_agent(path)
    hold = [a for a in champion_suite() if a.name == "hold_farmer"][0]
    r = head_to_head(cfg, agent, hold, episodes=episodes, seed=seed)
    return _h_of(repr((r["mean"], r["stderr"], r["seat_A"], r["seat_B"], r["win_rate"],
                       _jsonable(r["x_stats"]), _jsonable(r["y_stats"]))))


def _h_of(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def solution_digest(sol) -> str:
    """Every reachable (k, state) of a solved game with its value rounded to 9 decimals (LP noise is ~1e-15)."""
    h = _h()
    for k, layer in enumerate(sol.layers):
        for s in sorted(layer):
            h.update(repr((k, s.to_list(), round(sol.V[k][s] + 0.0, 9) + 0.0)).encode())
    return h.hexdigest()


def solver_values() -> List[Tuple[float, int, str]]:
    """(root value, reachable states, solution digest) of the three K = 16 conftest models (v1, revised, siege)."""
    from lanegame import LaneGame
    from lanegame.solver import solve_equilibrium
    out = []
    for cfg in (GameConfig.v1(horizon=16), GameConfig.v2(horizon=16), GameConfig(horizon=16)):
        g = LaneGame(cfg)
        layers = g.reachable()
        sol = solve_equilibrium(g, layers)
        out.append((sol.value, sum(len(x) for x in layers), solution_digest(sol)))
    return out


if __name__ == "__main__":
    import sys
    import time
    for name in ("v1", "v2", "siege", "champions"):
        t0 = time.time()
        print(f"lane:{name}", lane_digest(name), f"{time.time() - t0:.1f}s")
    t0 = time.time()
    print("champion", champion_digest(), f"{time.time() - t0:.1f}s")
    t0 = time.time()
    print("rollout", rollout_digest(), f"{time.time() - t0:.1f}s")
    import os
    if os.path.exists(M3B_CHECKPOINT):
        t0 = time.time()
        print("checkpoint", checkpoint_digest(), f"{time.time() - t0:.1f}s")
    if "--solver" in sys.argv:
        print("solver", repr(solver_values()))
