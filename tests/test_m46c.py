"""M4.6c: the C1-versus-baseline pilot's infrastructure (arms, the opt-in collector hook, the mixed start distribution,
the archive's constructed states and held-out cells, the premium statistic)."""

import numpy as np
import pytest

from lanegame.config import GameConfig


def test_arms_differ_only_in_the_investment_curve():
    from lanegame.experiments.rl.m46b.curves import ARMS as B_ARMS
    from lanegame.experiments.rl.m46c import common as C
    base, c1 = C.arm_cfg("baseline").to_dict(), C.arm_cfg("C1").to_dict()
    assert base == GameConfig.m45_calibrated().to_dict()
    assert C.arm_overrides("C1") == B_ARMS["C1"]["overrides"] == {"invest_gain": 3.24, "invest_scale": 64.0}
    assert {k for k in base if base[k] != c1[k]} == {"invest_gain", "invest_scale"}
    assert base["xp_reward_weight"] == 0.0 and base["gamma"] == 0.99 and base["terminal_rule"] == "none"


def test_policy_specifications():
    from lanegame.experiments.rl.m46c import common as C
    assert C.policy_path("start:3").replace("\\", "/").endswith("m45c/ext/s3/final.pt")
    assert C.policy_path("C1:2").replace("\\", "/").endswith("m46c/train/main/C1/s2/final.pt")
    assert C.policy_path("baseline:1@2048000").replace("\\", "/").endswith(
        "train/main/baseline/s1/snapshots/snap_002048000.pt")
    assert C.trained_on("C1:4") == "C1" and C.trained_on("start:4") == "baseline"
    with pytest.raises(ValueError):
        C.policy_path("C2:0")


def _tiny_ppo(cfg, **kw):
    from lanegame.rl import PPOConfig
    base = dict(total_steps=2 * 2 * 16 * 2, n_envs=2, segment_length=16, game_length=40, seed=5, torch_threads=1,
                minibatch_size=32, lane_cache=500)
    base.update(kw)
    return PPOConfig.continuing(cfg, **base)


def _digest(res):
    from lanegame.rl import weights_digest
    return weights_digest(res.net.state_dict())


def test_collector_hook_default_is_bit_identical():
    from lanegame.rl import SharedConfig, train_shared
    from lanegame.rl.shared import SharedCollector
    cfg = GameConfig.m45_calibrated()
    ppo, sc = _tiny_ppo(cfg), SharedConfig(current_fraction=1.0, snapshot_every=10_000)
    a = train_shared(cfg, ppo, sc, verbose=False)
    b = train_shared(cfg, ppo, sc, verbose=False, collector_cls=SharedCollector)
    assert _digest(a) == _digest(b)


def _archive(tmp_path, cfg):
    from lanegame.env.starts import StartState, save_archive
    from lanegame.experiments.rl.m46c.archive import freeze_draw, zone_draw
    rng = np.random.default_rng(0)
    sts = []
    for i in range(6):
        (lane, m4, k), _ = zone_draw(cfg, rng)
        sts.append(StartState(lane, m4, k % cfg.phase_period, None, "zone test"))
        (lane, m4, k), _ = freeze_draw(cfg, rng)
        sts.append(StartState(lane, m4, k % cfg.phase_period, None, "freeze test"))
    p = str(tmp_path / "a.json.gz")
    save_archive(p, sts, cfg)
    return p


def test_mixed_collector_beta_zero_equals_shared_collector(tmp_path):
    from lanegame.experiments.rl.m46c.collector import mixed_collector_cls
    from lanegame.rl import SharedConfig, train_shared
    cfg = GameConfig.m45_calibrated()
    ppo, sc = _tiny_ppo(cfg), SharedConfig(current_fraction=1.0, snapshot_every=10_000)
    a = train_shared(cfg, ppo, sc, verbose=False)
    b = train_shared(cfg, ppo, sc, verbose=False, collector_cls=mixed_collector_cls(_archive(tmp_path, cfg), 0.0, 10))
    assert _digest(a) == _digest(b)


def test_mixed_collector_starts_lengths_and_mirrors(tmp_path):
    from lanegame.experiments.rl.m46c.collector import mixed_collector_cls
    from lanegame.rl import MatchupSampler, SnapshotPool
    import torch
    from lanegame.rl import PolicyValueNet
    from lanegame.env import LaneEnv
    cfg = GameConfig.m45_calibrated()
    ppo = _tiny_ppo(cfg, n_envs=4, segment_length=64, game_length=50)
    cls = mixed_collector_cls(_archive(tmp_path, cfg), 1.0, 7)
    col = cls(cfg, ppo, MatchupSampler(SnapshotPool(2), 1.0), seed=3)
    env0 = LaneEnv(cfg)
    net = PolicyValueNet(env0.obs_size, env0.n_actions, (16, 16))
    gen = torch.Generator().manual_seed(0)
    for _ in range(3):
        col.collect(net, gen)
    d = col.describe_starts()
    assert d["games_started"]["initial"] == 0 and d["games_started"]["archive"] > 4
    assert 0 < d["games_started"]["mirrored"] < d["games_started"]["archive"]
    assert all(e.max_turns <= 7 for e in col.envs)                    # archive games: archive_length (first: <=)
    assert d["archive_transition_share_finished"] == 1.0
    with pytest.raises(ValueError):
        cls(cfg, _tiny_ppo(cfg, start_mix=0.5, start_archive=d["archive"]), MatchupSampler(SnapshotPool(2), 1.0), 3)


def test_constructed_states_are_valid_in_both_arms_and_disjoint_from_evaluation_cells():
    from lanegame.env import LaneEnv
    from lanegame.experiments.rl.m46c import common as C
    from lanegame.experiments.rl.m46c.archive import eval_keys, freeze_draw, lead_cell, s_of_level, zone_draw
    base = C.arm_cfg("baseline")
    keys = eval_keys(base)
    rng = np.random.default_rng(1)
    for _ in range(40):
        for draw in (zone_draw, freeze_draw):
            (lane, m4, k), _ = draw(base, rng)
            assert (lane, m4, k % base.phase_period) not in keys
            for arm in C.ARMS:
                env = LaneEnv(C.arm_cfg(arm), max_turns=2)
                o = env.reset(seed=0, root=lane, k0=k, champ=m4)
                assert o[0].mask.any() and o[1].mask.any()
    lane, m4, k = lead_cell(base, "mid", "neutral", "B+8")
    assert m4.a.items == 42 * base.gold_scale and m4.b.items == 50 * base.gold_scale and k == 2
    assert s_of_level(9) == 15.0 and s_of_level(18) == 56.0


def test_premium_cancels_the_equal_cell_and_seat_effects():
    from lanegame.experiments.rl.m46c.analysis import CELL_IDX, premium_of
    rng = np.random.default_rng(0)
    recs = {}
    for (st, sit, c), i in CELL_IDX.items():
        seat = 0.3                                                       # a seat-A advantage in every cell
        lead = 0.0 if c == "equal" else (1.0 if c.startswith("A") else -1.0) * (0.5 if c.endswith("+8") else 0.9)
        recs[i] = {"G": {"self": {"50": list(seat + lead + 0.0 * rng.normal(size=4))}}}
    assert premium_of(recs, "self", "mid", 8, 50) == pytest.approx(0.5)
    assert premium_of(recs, "self", "late", 16, 50) == pytest.approx(0.9)


def test_seed_stats_t_interval():
    from lanegame.experiments.rl.m46c.common import seed_stats
    st = seed_stats({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0})
    assert st["mean"] == pytest.approx(3.5)
    assert st["ci"] == pytest.approx(2.571 * np.std([1, 2, 3, 4, 5, 6], ddof=1) / np.sqrt(6))
    assert st["pos"] == 6 and st["neg"] == 0
