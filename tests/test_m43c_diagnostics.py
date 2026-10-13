"""M4.3c design diagnostics (lanegame/experiments/m43c): objective channels, channel rollouts, the extended-absence
device, the constructed reward example, the progression event bookkeeping. Scripted agents only (no checkpoints)."""

import numpy as np
import pytest

from lanegame.champions_m4 import check_invariants
from lanegame.config import GameConfig
from lanegame.diagnostics import FollowPolicy, Snapshot
from lanegame.diagnostics.branch import rollout, rollout_seed
from lanegame.env import LaneEnv
from lanegame.experiments.m43c.common import channels, level_progress
from lanegame.experiments.m43c.rollouts import channel_rollout, paired
from lanegame.progression import tables

CFG = GameConfig.m4_lasthit()


def _agents():
    from lanegame.experiments.m4.zoning_natural import POLICIES
    from lanegame.experiments.rl.m43b.references import ReactiveAgent
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


def test_channel_rollout_reproduces_branch_rollout_for_follow_and_wave_plans():
    from lanegame.experiments.rl.m43b.intervene import wave_plan
    snap = _snapshot()
    agents = _agents()
    env = LaneEnv(snap.config(CFG), cache_size=1000)
    for plan in (FollowPolicy(), wave_plan("hold", "marginal", 14)):
        for m in range(2):
            ref = rollout(env, snap, 0, plan, agents, rollout_seed(9, m), max_turns=60)
            got = channel_rollout(env, snap, 0, plan, agents, rollout_seed(9, m), window=60)
            assert got.G == ref.ret


def test_extended_absence_keeps_the_champion_away_and_the_state_valid():
    snap = _snapshot()
    seat = 0
    dead = snap.with_side(seat, pos=-CFG.tower, hp=0, absent=3)
    env = LaneEnv(dead.config(CFG), cache_size=1000)
    away = []

    def on_turn(t, info):
        away.append(bool(info["away"][seat]))
        check_invariants(env.champ, CFG)
        for o in env.observe():
            assert float(np.max(o.vector)) <= 1.0

    channel_rollout(env, dead, seat, None, _agents(), rollout_seed(4, 0), window=20, extra_absence={seat: 4},
                    on_turn=on_turn)
    assert away[:7] == [True] * 7 and away[7] is False


def test_one_turn_death_is_a_recall_on_the_same_streams():
    from lanegame.experiments.m43c.death import variants
    snap = _snapshot()
    var = variants(snap, 0, CFG)
    res = paired(CFG, snap, 0, {"recall": None, "death_1": None}, _agents(), 3, 11, window=40,
                 variants={k: var[k] for k in ("recall", "death_1")})
    assert np.array_equal(res["recall"]["G"], res["death_1"]["G"])


def test_terminal_sums_stop_at_the_last_plate():
    snap = _snapshot()
    done = snap.with_side(1, tower_dmg=CFG.plate_thresholds[-1])
    env = LaneEnv(done.config(CFG), cache_size=1000)
    r = channel_rollout(env, done, 0, None, _agents(), rollout_seed(2, 0), window=30)
    assert r.term_turn == 0 and r.G_T == 0.0 and r.X_T == 0.0 and r.G != 0.0


def test_constructed_reward_example():
    from lanegame.experiments.m43c.examples import one_turn
    d = one_turn()
    assert d["reward_units"] == 16_875_000 and d["reward"] == 1.6875
    assert d["farm_units"] == [7_875_000, 6_000_000]
    assert d["plate_units"] == [10_000_000, 0] and d["bounty_units"] == [5_000_000, 0]
    assert d["died"] == [False, True] and d["plates"] == [1, 0]


def test_progression_events_are_consistent():
    from lanegame.experiments.m43c.progression import play_game
    g = play_game(CFG, _agents(), np.random.SeedSequence(12), window=200)
    for s in (0, 1):
        turns = [x for x in g["plate"][s] if x is not None]
        assert turns == sorted(turns)
    finals = [x for x in (g["plate"][0][-1], g["plate"][1][-1]) if x is not None]
    assert g["end"] == (min(finals) if finals else None)
    if g["end"]:
        assert g["weight_after_end"] == pytest.approx(CFG.gamma ** g["end"])


def test_freeze_gap_mirror_gives_b_the_investment():
    from lanegame.experiments.m4.zoning_natural import starts
    from lanegame.experiments.m43c.freeze import gapped
    st = starts(CFG)["freeze"]
    _, ahead, _ = gapped(st, CFG, "a_ahead")
    _, behind, _ = gapped(st, CFG, "a_behind")
    assert ahead.a.items > 0 and ahead.b.items == 0
    assert behind.b.items == ahead.a.items and behind.a.items == 0


@pytest.mark.parametrize("r", [3, 2, 1])
def test_xp_range_ledger_identity_and_histogram_pricing(r):
    """XP-range screen bookkeeping: ledger = credited + discarded + missed (range, away), and the measured range loss
    is exactly the present-turn ledger HP farther than r from the champion."""
    from lanegame.experiments.m43c.xp_range import OFF, cfg_range, run_turns, window_stats
    from lanegame.env.agents import episode_streams
    cfg = cfg_range(r)
    a, b = _agents()
    _, env_ss, rng_a, rng_b = episode_streams(np.random.SeedSequence(31))
    env = LaneEnv(cfg, max_turns=80)
    rows, _ = run_turns(cfg, env, env.reset(seed=env_ss), lambda t: (a, b), (rng_a, rng_b), 80)
    w = window_stats(rows, 80)
    for s, tag in ((0, "a"), (1, "b")):
        raw = sum(x["raw"][s] for x in rows)
        assert w[f"avail_{tag}"] - w[f"missed_range_{tag}"] - w[f"missed_away_{tag}"] == raw
        h = w[f"hist_{tag}"]
        far = sum(int(h[e, x]) for e in range(h.shape[0]) for x in range(h.shape[1]) if abs(x - e) > r)
        assert far == w[f"missed_range_{tag}"]
    assert cfg.xp_range == r and GameConfig.m4_lasthit().xp_range == 3


def test_xp_range_three_reproduces_the_freeze_harness():
    from lanegame.experiments.m43c import freeze as fz
    from lanegame.experiments.m43c.xp_range import _freeze_job
    from lanegame.experiments.m4.zoning_natural import starts
    from lanegame.experiments.rl.m43b.references import ReactiveAgent
    _, v = _freeze_job(("freeze", "freezer", "approach_farm", "equal", 3, 2))
    pols = fz._policies()
    st = fz.gapped(starts(CFG)["freeze"], CFG, "equal")
    for ep, g in enumerate(v["games"]):
        rows = fz.play(CFG, st, (ReactiveAgent(pols["freezer"]), ReactiveAgent(pols["approach_farm"])),
                       np.random.SeedSequence(fz.FREEZE_SEED, spawn_key=(0, ep)), 40)
        old = fz._window_stats(rows, 40)
        for k in ("gold_a", "gold_b", "xp_a", "xp_b", "dmg_b", "held"):
            assert g["40"][k] == pytest.approx(old[k])
