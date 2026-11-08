"""M4.3c design diagnostics (experiments/m43c): objective channels, channel rollouts, the extended-absence
device, the constructed reward example, the progression event bookkeeping. Scripted agents only (no checkpoints)."""

import numpy as np
import pytest

from lanegame.engine.champions_m4 import check_invariants
from lanegame.engine.config import GameConfig
from lanegame.diagnostics import FollowPolicy, Snapshot
from lanegame.diagnostics.branch import rollout, rollout_seed
from lanegame.env import LaneEnv
from experiments.m43c.common import channels, level_progress
from lanegame.engine.progression import tables

CFG = GameConfig.m4_lasthit()


def _agents():
    from experiments.m4.zoning_natural import POLICIES
    from experiments.m43b.references import ReactiveAgent
    pol = {p.name: p for p in POLICIES}
    return ReactiveAgent(pol["clear_punish"], "clear_punish"), ReactiveAgent(pol["push_punish"], "push_punish")


def _snapshot(turns: int = 40, seed: int = 5):
    env = LaneEnv(CFG)
    a, b = _agents()
    obs = env.reset(seed=seed)
    rng = np.random.default_rng(0)
    for _ in range(turns):
        obs = env.step(a.act(obs[0], rng), b.act(obs[1], rng)).obs
    return Snapshot.from_env(env)


def test_level_progress_is_the_level_at_thresholds_and_monotone():
    t = tables(CFG)
    for L in range(1, 19):
        assert level_progress(t.xp_cum[L], t) == L
    vals = [level_progress(x, t) for x in range(0, t.xp_cap + 50)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert vals[-1] == 18.0


def test_gold_channel_is_the_reward_exactly_and_xp_channel_the_credited_difference():
    env = LaneEnv(CFG, max_turns=150)
    a, b = _agents()
    obs = env.reset(seed=3)
    rng = np.random.default_rng(1)
    t = tables(CFG)
    seen_xp = False
    while True:
        r = env.step(a.act(obs[0], rng), b.act(obs[1], rng))
        ga, xa, la = channels(r.info, 0, t, CFG.gold_scale)
        gb, xb, lb = channels(r.info, 1, t, CFG.gold_scale)
        assert ga == r.rewards[0] and gb == r.rewards[1]
        assert xa == r.info["xp_gain"][0] - r.info["xp_gain"][1] == -xb
        assert la == pytest.approx(-lb)
        seen_xp = seen_xp or xa != 0
        if r.done:
            break
        obs = r.obs
    assert seen_xp

