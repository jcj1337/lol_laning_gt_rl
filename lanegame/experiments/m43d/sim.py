"""Paired-seed games of the M4.3d study: `m43c.harness.play` plus an optional chance mask and the M4.3d ledger.

Every game runs through the real environment (LaneEnv, the M4 resolver, its own chance stream). The start states
are the M4.3c harness starts (`harness.start`): "neutral" (turn 9: both 8-HP fronts and both champions at the lane
cell) and "siege" (turn 24: one turn after A's crash, A's 14-HP mass under B's tower, A in B's tower cell with its
next 8-HP front at the lane cell, B in its tower cell with its next wave at its base), at a chosen level and power
gap. Game e of every cell uses SeedSequence(root, spawn_key=(e,)) (common random numbers).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ...champions_m4 import context
from ...config import GameConfig
from ...env import LaneEnv
from ..m43c import harness as h
from .common import MASKS, MaskedChance
from .ledger import turn_extra, window_sums

A, B = 0, 1


def play(cfg: GameConfig, st, agents: Sequence, seed, turns: int, mask: Optional[str] = None,
         record: bool = False) -> dict:
    """One game; returns harness rows, the M4.3d ledger rows and (record) the raw infos."""
    lane, m4, k = st
    ctx = context(cfg)
    env = LaneEnv(cfg, max_turns=turns, cache_size=5_000)
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    obs = env.reset(seed=ss, root=lane, k0=k, champ=m4)
    if mask:
        env._rng = MaskedChance(env._rng, MASKS[mask])
    for ag in agents:
        if callable(getattr(ag, "reset", None)):
            ag.reset()
    rows, extra, infos = [], [], []
    alive0 = [h._alive_enemy(env.state, s, cfg) for s in (A, B)]
    while True:
        a, b = agents[0].act(obs[0]), agents[1].act(obs[1])
        r = env.step(a, b)
        for side, ag in enumerate(agents):
            if callable(getattr(ag, "notify", None)):
                ag.notify(r.info, side)
        rows.append(h.turn_row(r.info, env.state, cfg, ctx))
        extra.append(turn_extra(r.info, cfg, ctx.t))
        if record:
            infos.append(r.info)
        if r.done:
            break
        obs = r.obs
    alive1 = [h._alive_enemy(env.state, s, cfg) for s in (A, B)]
    return {"rows": rows, "extra": extra, "infos": infos, "alive0": alive0, "alive1": alive1,
            "end_champ": env.champ, "end_lane": env.state, "k0": k}


def summarize(game: dict, cfg: GameConfig, windows: Sequence[int]) -> dict:
    """Per window (and "all"): the harness window statistics and the M4.3d ledger sums."""
    out = {"all": {"h": h.window(game, cfg), "x": window_sums(game["extra"])}}
    for n in windows:
        out[str(n)] = {"h": h.window(game, cfg, n), "x": window_sums(game["extra"], n)}
    return out


__all__ = ["play", "summarize"]
