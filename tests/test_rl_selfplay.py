"""M3b/M3c: the snapshot pool, the self-play opponent sampler, a short self-play run, and the evaluation
harness (both seats, common random numbers, skill assigned to players, approximate best responses).
Skipped without PyTorch."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lanegame import GameConfig                                                       # noqa: E402
from lanegame.env import champion_agents                                              # noqa: E402
from lanegame.rl import (NetAgent, PolicyValueNet, PPOConfig, SelfPlaySampler, SnapshotPool,   # noqa: E402
                         approximate_best_response, head_to_head, load_checkpoint, train_selfplay)

CFG = GameConfig.champions(horizon=10)


def test_snapshot_pool_is_bounded_frozen_and_keeps_its_anchor():
    net = PolicyValueNet(50, 46)
    pool = SnapshotPool(pool_size=3)
    first = pool.add(net, 0)
    x = torch.rand(4, 50)
    before = first.net(x)[0].clone()
    for step in (10, 20, 30, 40):
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.1)                                   # the learner keeps changing ...
        pool.add(net, step)
    assert len(pool) == 3 and pool.steps == [0, 30, 40] and pool.latest.name == "snap@40"
    assert torch.equal(first.net(x)[0], before)               # ... the frozen snapshot does not
    assert all(not p.requires_grad for a in pool.snapshots for p in a.net.parameters())


def test_selfplay_sampler_mixture_and_reproducibility():
    agents = champion_agents(CFG)
    scripted = [agents["hold_farmer"], agents["trade"]]
    pool = SnapshotPool(4)
    s = SelfPlaySampler(scripted, pool, p_scripted=0.3, p_latest=0.35)
    rng = np.random.default_rng(0)
    assert all(s.sample(rng) in scripted for _ in range(20))   # empty pool: scripted only
    pool.add(PolicyValueNet(50, 46), 0)
    pool.add(PolicyValueNet(50, 46), 10)
    draws = [s.sample(np.random.default_rng(1)).name for _ in range(1)]
    rng = np.random.default_rng(2)
    names = [s.sample(rng).name for _ in range(3000)]
    frac_scripted = np.mean([n in ("hold_farmer", "trade") for n in names])
    frac_latest = np.mean([n == "snap@10" for n in names])
    assert abs(frac_scripted - 0.3) < 0.04 and abs(frac_latest - (0.35 + 0.35 / 2)) < 0.04
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    assert [s.sample(rng_a).name for _ in range(50)] == [s.sample(rng_b).name for _ in range(50)]
    assert draws


def test_short_selfplay_run_adds_snapshots_and_is_reproducible(tmp_path):
    agents = champion_agents(CFG)
    cfg = PPOConfig(total_steps=4 * 4 * 20, n_envs=4, n_steps=20, minibatch_size=40, epochs=1, seed=4, lane_cache=0)
    init = PolicyValueNet(50, 46)
    a = train_selfplay(CFG, cfg, [agents["hold_farmer"]], init_net=init, pool_size=3, snapshot_every=2,
                       out_dir=str(tmp_path), verbose=False)
    assert [x.name for x in a.pool.snapshots] == ["snap@0", "snap@160", "snap@320"]
    assert len(a.snapshot_paths) == 3 and a.train.history[-1]["pool"] == ["snap@0", "snap@160", "snap@320"]
    net, cfg_loaded, meta = load_checkpoint(a.snapshot_paths[1])
    assert meta["step"] == 160 and cfg_loaded == CFG
    # reproducible: the same initial network and seed give the same run
    start = PolicyValueNet(50, 46)
    runs = [train_selfplay(CFG, cfg, [agents["hold_farmer"]], init_net=copy.deepcopy(start), pool_size=3,
                           snapshot_every=2, verbose=False) for _ in range(2)]
    assert [h["mean_return"] for h in runs[0].train.history] == [h["mean_return"] for h in runs[1].train.history]
    for p, q in zip(runs[0].train.net.parameters(), runs[1].train.net.parameters()):
        assert torch.equal(p, q)


def test_head_to_head_uses_both_seats_common_seeds_and_player_skills():
    agents = champion_agents(CFG)
    x = agents["hold_farmer"]
    r = head_to_head(CFG, x, x, 40, seed=3)
    assert r["mean"] == 0.0 and r["episodes"] == 80             # identical deterministic players on the same seeds
    r = head_to_head(CFG, agents["aggressive_trader"], agents["aggressive_trader"], 60, seed=3, skill=(0.9, 0.1))
    assert r["skill"] == [0.9, 0.1] and r["seat_A"]["mean"] > 0 and r["seat_B"]["mean"] > 0   # skill follows the player
    r2 = head_to_head(CFG, agents["aggressive_trader"], agents["yield_farmer"], 30, seed=3)
    assert r2["mean"] > 0 and abs(r2["win_rate"] + r2["loss_rate"] + r2["tie_rate"] - 1) < 1e-12
    assert set(r2["x_stats"]["behaviors"]) <= {"H", "Y", "T", "S", "R", "ABSENT"}


def test_approximate_best_response_runs_and_reports_a_held_out_value():
    agents = champion_agents(CFG)
    r = approximate_best_response(CFG, agents["yield_farmer"],
                                  PPOConfig(total_steps=640, n_envs=4, n_steps=20, minibatch_size=40, epochs=1, seed=1,
                                            lane_cache=0), eval_episodes=10, eval_seed=99)
    assert r["train_steps"] == 640 and np.isfinite(r["value"]) and r["eval"]["episodes"] == 20
    assert isinstance(r["agent"], NetAgent) and len(r["curve"]) == 8


def test_warm_started_best_response_starts_from_the_target():
    """A warm start copies the target's weights (the target itself is left untouched) and is flagged."""
    agents = champion_agents(CFG)
    target_net = PolicyValueNet(50, 46)
    before = [p.clone() for p in target_net.parameters()]
    target = NetAgent(target_net, "target")
    r = approximate_best_response(CFG, target,
                                  PPOConfig(total_steps=160, n_envs=4, n_steps=20, minibatch_size=40, epochs=1, seed=2,
                                            lane_cache=0), eval_episodes=5, eval_seed=7, init_net=target_net)
    assert r["warm_start"] and all(torch.equal(p, q) for p, q in zip(target_net.parameters(), before))
    assert all(torch.equal(p, q) for p, q in zip(target.net.parameters(), before))
    assert agents
