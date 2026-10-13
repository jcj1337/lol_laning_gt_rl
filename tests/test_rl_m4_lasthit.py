"""M4.3a: the shared-policy self-play trainer on the M4.3 target environment, GameConfig.m4_lasthit().

Both seats' experience from one network in current-policy games (mirror observations, antisymmetric rewards,
per-seat accounting); continuing-time segments and game time limits (truncations where the game times say, never
a termination, bootstraps from the pre-reset observation and across the segment boundary, no lambda leakage); every
action legal; historical games against frozen snapshots on this environment; training reproducibility and checkpoint
round trips (final.pt reproduces the action distribution, run.json carries the exact configuration); the discounted
evaluator on a trained network. Skipped without PyTorch."""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lanegame import GameConfig                                                          # noqa: E402
from lanegame.champions import swap_seats                                                # noqa: E402
from lanegame.env import LaneEnv, observe                                                # noqa: E402
from lanegame.env.discounted import evaluate_discounted                                  # noqa: E402
from lanegame.rl import (CURRENT, HISTORICAL, MatchupSampler, NetAgent, PolicyValueNet, PPOConfig,   # noqa: E402
                         SharedCollector, SharedConfig, SnapshotPool, compute_gae, load_agent, load_checkpoint,
                         train_shared, weights_digest)
from lanegame.rl.shared import rollout_batch                                             # noqa: E402

CFG = GameConfig.m4_lasthit()
OBS, NACT = LaneEnv(CFG).obs_size, LaneEnv(CFG).n_actions


def ppo(**kw):
    base = dict(total_steps=800, n_envs=4, segment_length=24, game_length=40, minibatch_size=64, epochs=1, seed=3,
                hidden=(32, 32), lane_cache=2000)
    base.update(kw)
    return PPOConfig.continuing(CFG, **base)


def net(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet(OBS, NACT, (32, 32))


def test_current_policy_games_train_both_seats_with_mirror_observations_and_antisymmetric_rewards():
    col = SharedCollector(CFG, ppo(), MatchupSampler(SnapshotPool(4), 1.0), seed=1)
    roll, finished, counts = col.collect(net(), torch.Generator().manual_seed(0))
    T, N = 24, 4
    assert roll["active"].all() and counts["transitions_seat_a"] == counts["transitions_seat_b"] == T * N
    assert counts["agent_transitions"] == 2 * counts["env_turns"] and counts["terminations"] == 0
    r = roll["rewards"]
    assert np.array_equal(r[:, 0::2], -r[:, 1::2])                                   # exactly antisymmetric
    # slot B's observation at every step is seat A's view of the mirrored game: decode both and compare
    from lanegame.env.observation_m4 import decode_m4
    for t in range(T):
        for i in range(N):
            pa, la, ca, _ = decode_m4(roll["obs"][t, 2 * i], CFG)
            pb, lb, cb, _ = decode_m4(roll["obs"][t, 2 * i + 1], CFG)
            assert pa == pb and lb == la.mirror() and cb == ca.mirror()
            assert np.array_equal(roll["obs"][t, 2 * i + 1], observe(la.mirror(), pa, 0, swap_seats(CFG), ca.mirror()))
    # every action taken was legal (the environment itself raises otherwise; this checks the recorded masks)
    a = roll["active"]
    assert roll["mask"][a][np.arange(a.sum()), roll["actions"][a]].all()
    assert all(e.kind == CURRENT for e in finished)


def test_game_time_limits_truncations_and_bootstraps():
    cfg = ppo(segment_length=30, game_length=12)
    col = SharedCollector(CFG, cfg, MatchupSampler(SnapshotPool(4), 1.0), seed=2)
    first = list(col.times.first_lengths)
    assert all(1 <= x <= 12 for x in first)
    n = net()
    roll, finished, counts = col.collect(n, torch.Generator().manual_seed(0))
    T, N = 30, 4
    trunc = roll["truncated"][:, 0::2]
    assert not roll["terminated"].any() and counts["terminations"] == 0
    for i in range(N):                                   # truncations exactly at the first length, then every 12
        want = []
        t = first[i] - 1
        while t < T:
            want.append(t)
            t += 12
        assert list(np.flatnonzero(trunc[:, i])) == want
    assert counts["truncations"] == int(trunc.sum()) == len(finished)
    nv, v = roll["next_values"], roll["values"]
    for j in range(2 * N):
        for t in range(T - 1):
            if not roll["truncated"][t, j]:
                assert nv[t, j] == v[t + 1, j]           # within a game: the next step's value
            else:
                assert nv[t, j] != 0.0                   # a truncation bootstraps (never treated as terminal)
    # the rollout is a deterministic function of the seeds
    col2 = SharedCollector(CFG, cfg, MatchupSampler(SnapshotPool(4), 1.0), seed=2)
    roll2, _, _ = col2.collect(n, torch.Generator().manual_seed(0))
    for k in roll:
        assert np.array_equal(roll[k], roll2[k])
    # the pre-reset bootstrap, checked independently: replay environment 0's first game on a fresh LaneEnv seeded by
    # the collector's deterministic episode stream (the first child of the episode SeedSequence) with the recorded
    # actions; the recorded bootstrap is the network's value of the final observation of that game, for both seats
    from lanegame.env.agents import episode_streams
    root = np.random.SeedSequence(2)
    root.spawn(1)                                        # the matchup generator
    first_child = root.spawn(1)[0].spawn(1)[0]           # episode seeds -> environment 0's first game
    _, env_ss, _, _ = episode_streams(first_child)
    env = LaneEnv(CFG, max_turns=first[0])
    obs = env.reset(seed=env_ss)
    for t in range(first[0]):
        r = env.step(int(roll["actions"][t, 0]), int(roll["actions"][t, 1]))
        assert r.rewards[0] * cfg.reward_scale == roll["rewards"][t, 0]
        obs = r.obs
    assert r.truncated
    with torch.no_grad():
        want = float(n.value(torch.as_tensor(obs[0].vector).unsqueeze(0))[0])
        want_b = float(n.value(torch.as_tensor(obs[1].vector).unsqueeze(0))[0])
    assert nv[first[0] - 1, 0] == pytest.approx(want, abs=1e-6) and nv[first[0] - 1, 1] == pytest.approx(want_b, abs=1e-6)
    # GAE: nothing leaks across a game boundary, and the truncated step's advantage is its own TD error
    adv, _ = compute_gae(roll["rewards"], roll["values"], nv, roll["terminated"], roll["truncated"], cfg.gamma,
                         cfg.gae_lambda)
    t0 = first[0] - 1
    assert adv[t0, 0] == pytest.approx(roll["rewards"][t0, 0] + cfg.gamma * nv[t0, 0] - v[t0, 0])


def test_segment_boundary_bootstrap_equals_the_next_segments_first_value():
    """Without an update in between, the value that bootstraps the last step of a segment is exactly the value
    recorded for the first step of the next segment (same observation, same network)."""
    col = SharedCollector(CFG, ppo(segment_length=10, game_length=100), MatchupSampler(SnapshotPool(4), 1.0), seed=4)
    n = net()
    g = torch.Generator().manual_seed(0)
    r1, _, _ = col.collect(n, g)
    r2, _, _ = col.collect(n, g)
    live = r1["active"][-1] & ~(r1["terminated"][-1] | r1["truncated"][-1])
    assert live.all()
    assert np.allclose(r1["next_values"][-1, live], r2["values"][0, live], atol=1e-6)
    assert not np.array_equal(r1["obs"][-1], r2["obs"][0])     # the next segment starts one turn later


def test_historical_games_on_m4_lasthit_train_the_learner_seat_only_and_keep_the_snapshot_frozen():
    snap = net(7)
    pool = SnapshotPool(4, keep_first=False)
    pool.add(snap, 100, "snap@100")
    frozen = {k: v.clone() for k, v in pool.snapshots[0].net.state_dict().items()}
    col = SharedCollector(CFG, ppo(), MatchupSampler(pool, 0.0), seed=5)
    n = net()
    roll, finished, counts = col.collect(n, torch.Generator().manual_seed(1))
    assert (roll["active"].sum(axis=1) == 4).all() and counts["transitions_historical"] == counts["agent_transitions"]
    assert all(e.kind == HISTORICAL and e.opponent == "snap@100" for e in finished)
    batch, _, _ = rollout_batch(roll, col.cfg)
    opt = torch.optim.Adam(n.parameters(), lr=1e-2)
    from lanegame.rl import ppo_update
    ppo_update(n, opt, batch, col.cfg, torch.Generator().manual_seed(0))
    assert all(torch.equal(frozen[k], v) for k, v in pool.snapshots[0].net.state_dict().items())


def test_training_is_reproducible_and_checkpoints_round_trip(tmp_path):
    out = str(tmp_path / "run")
    scfg = SharedConfig(current_fraction=0.8, snapshot_every=300)
    a = train_shared(CFG, ppo(total_steps=1200), scfg, out_dir=out, verbose=False, name="a")
    b = train_shared(CFG, ppo(total_steps=1200), scfg, verbose=False, name="b")
    assert weights_digest(a.net.state_dict()) == weights_digest(b.net.state_dict())
    strip = lambda h: [{k: v for k, v in e.items() if k not in ("elapsed", "sps", "collect_s", "update_s")} for e in h]
    assert json.dumps(strip(a.history)) == json.dumps(strip(b.history))
    c = train_shared(CFG, ppo(total_steps=1200, seed=4), scfg, verbose=False, name="c")
    assert weights_digest(c.net.state_dict()) != weights_digest(a.net.state_dict())
    # history fields of the audit
    h = a.history[-1]
    for key in ("grad_norm", "grad_norm_max", "value_mean", "value_absmax", "return_mean", "adv_std",
                "rollout_truncations", "rollout_terminations", "rollout_transitions_seat_a", "rollout_single_choice"):
        assert key in h and np.isfinite(h[key])
    assert all(e["rollout_terminations"] == 0 for e in a.history)
    assert sum(e["rollout_truncations"] for e in a.history) == sum(e["games_current"] + e["games_historical"] for e in a.history)
    # checkpoints
    net_, cfg_, meta = load_checkpoint(os.path.join(out, "final.pt"))
    assert cfg_ == CFG and meta["ppo"]["game_length"] == 40 and meta["time_semantics"]["time"] == "continuing"
    agent, loaded = NetAgent(a.net, "a"), load_agent(os.path.join(out, "final.pt"))
    env = LaneEnv(CFG, max_turns=5)
    for s in range(4):
        o = env.reset(seed=s)[s % 2]
        assert np.array_equal(agent.probabilities(o), loaded.probabilities(o))
    run = json.load(open(os.path.join(out, "run.json"), encoding="utf-8"))
    assert GameConfig.from_dict(run["game_config"]) == CFG and run["ppo"]["gamma"] == CFG.gamma
    assert len(a.snapshot_paths) >= 1 and all(load_checkpoint(p)[1] == CFG for p in a.snapshot_paths)


def test_discounted_evaluation_of_a_trained_network_is_antisymmetric_and_bounded():
    res = train_shared(CFG, ppo(total_steps=400), SharedConfig(current_fraction=1.0), verbose=False)
    agent = NetAgent(res.net, "p")
    from lanegame.env.agents_m4 import natural_suite
    y = natural_suite()["hold_farmer"]
    a = evaluate_discounted(CFG, agent, y, episodes=2, window=30)
    b = evaluate_discounted(CFG, y, agent, episodes=2, window=30)
    assert a["estimate"] == pytest.approx(-b["estimate"]) and a["tail_bound"]["guaranteed"] > 0
    self_play = evaluate_discounted(CFG, agent, agent, episodes=2, window=30)
    assert abs(self_play["estimate"]) < 1e-9
