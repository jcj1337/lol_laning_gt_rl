"""Shared-policy self-play with a current / historical mixture (lanegame.rl.shared): one network for both seats,
own-perspective observations, both seats' samples in current-policy games and only the learner's in historical
games, frozen snapshots, rollout log-probabilities, masks, zero-sum rewards, episode ends, snapshot round trips,
the mixture's statistics, seat symmetry, reproducibility and checkpoint safety. Skipped without PyTorch."""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lanegame import GameConfig                                                       # noqa: E402
from lanegame.champions import swap_seats                                             # noqa: E402
from lanegame.env import LaneEnv, observe, run_episodes                               # noqa: E402
from lanegame.rl import (CURRENT, HISTORICAL, LEARNER, MaskedCategorical, MatchupSampler, NetAgent,   # noqa: E402
                         PolicyValueNet, PPOConfig, SharedCollector, SharedConfig, SnapshotPool, load_agent,
                         load_checkpoint, ppo_update, save_checkpoint, train_shared)
from lanegame.rl.shared import rollout_batch                                          # noqa: E402

CFG = GameConfig.champions(horizon=10)
OBS, NACT = LaneEnv(CFG).obs_size, LaneEnv(CFG).n_actions


def _ppo(**kw):
    base = dict(total_steps=400, n_envs=4, n_steps=10, minibatch_size=64, seed=3)
    base.update(kw)
    return PPOConfig(**base)


def _net(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet(OBS, NACT)


def _pool_with(net, n=1):
    pool = SnapshotPool(8, keep_first=False)
    for i in range(n):
        pool.add(net, 100 * (i + 1), f"snap@{100 * (i + 1)}")
    return pool


def _collector(fraction, pool=None, **kw):
    pool = pool if pool is not None else SnapshotPool(8, keep_first=False)
    return SharedCollector(CFG, _ppo(**kw), MatchupSampler(pool, fraction), seed=11)


def test_current_policy_games_use_one_network_for_both_seats_and_record_both():
    net = _net()
    col = _collector(1.0)
    roll, finished, counts = col.collect(net, torch.Generator().manual_seed(0))
    assert roll["active"].all()                                     # both slots of every environment trainable
    assert counts["agent_transitions"] == 2 * counts["env_turns"] == 2 * 4 * 10
    assert all(e.kind == CURRENT and e.opponent == "self" for e in finished)
    # every recorded log-probability and value is the SAME network's, for seat A and seat B alike
    a = roll["active"]
    with torch.no_grad():
        logits, v = net(torch.as_tensor(roll["obs"][a]))
        lp = MaskedCategorical(logits, torch.as_tensor(roll["mask"][a])).log_prob(torch.as_tensor(roll["actions"][a]))
    assert np.allclose(lp.numpy(), roll["logprobs"][a], atol=1e-6)
    assert np.allclose(v.numpy(), roll["values"][a], atol=1e-6)


def test_each_seat_receives_its_own_perspective_observation_and_mask():
    col = _collector(1.0, skill_levels=(0.2, 0.8))
    before = [(o[0].vector.copy(), o[1].vector.copy(), o[0].mask.copy(), o[1].mask.copy(), o[0], o[1])
              for o in col.obs]
    skills = list(col.skills)
    roll, _, _ = col.collect(_net(), torch.Generator().manual_seed(0))
    for i, (va, vb, ma, mb, oa, ob) in enumerate(before):
        assert np.array_equal(roll["obs"][0, 2 * i], va) and np.array_equal(roll["obs"][0, 2 * i + 1], vb)
        assert np.array_equal(roll["mask"][0, 2 * i], ma) and np.array_equal(roll["mask"][0, 2 * i + 1], mb)
        # seat B's vector is seat A's view of the mirrored game (own coordinates, own skill first)
        cfg_i = CFG.with_(skill_a=skills[i][0], skill_b=skills[i][1])
        mirror = observe(ob.state.mirror(), ob.k, 0, swap_seats(cfg_i), ob.champ.mirror())
        assert np.array_equal(vb, mirror)


def test_actions_are_legal_and_masks_are_the_engine_legal_sets():
    col = _collector(0.5, _pool_with(_net(1)), n_steps=25)
    roll, _, _ = col.collect(_net(), torch.Generator().manual_seed(1))
    a = roll["active"]
    acts, masks = roll["actions"][a], roll["mask"][a]
    assert masks[np.arange(len(acts)), acts].all()
    assert masks.any(-1).all()


def test_zero_sum_rewards_are_not_negated_twice():
    col = _collector(1.0, n_steps=20)                              # two complete 10-turn games per environment
    roll, finished, _ = col.collect(_net(), torch.Generator().manual_seed(2))
    r = roll["rewards"]
    assert np.allclose(r[:, 0::2], -r[:, 1::2])                    # seat B's reward is -r, exactly once
    # each finished game's seat-A return equals its log's payoff; slot sums per game reproduce +payoff / -payoff
    by_env = {}
    for t in range(20):
        for i in range(4):
            by_env.setdefault((i, t // 10), [0.0, 0.0])
            by_env[(i, t // 10)][0] += r[t, 2 * i]
            by_env[(i, t // 10)][1] += r[t, 2 * i + 1]
    payoffs = sorted(round(e.payoff_a, 9) for e in finished)
    assert payoffs == sorted(round(v[0], 9) for v in by_env.values())
    assert all(abs(v[0] + v[1]) < 1e-9 for v in by_env.values())
    assert all(abs(e.log.payoff - e.payoff_a) < 1e-9 for e in finished)


def test_historical_games_train_only_the_learner_seat_and_snapshots_stay_frozen():
    snap_src = _net(5)
    pool = _pool_with(snap_src, 2)
    frozen = [{k: v.clone() for k, v in a.net.state_dict().items()} for a in pool.snapshots]
    col = _collector(0.0, pool, n_steps=20)
    net = _net()
    roll, finished, counts = col.collect(net, torch.Generator().manual_seed(3))
    act = roll["active"]
    assert (act.sum(axis=1) == 4).all()                           # exactly one trainable seat per environment
    assert counts["transitions_historical"] == counts["agent_transitions"] == counts["env_turns"]
    assert all(e.kind == HISTORICAL and e.opponent.startswith("snap@") for e in finished)
    for e in finished:                                            # the learner's seat is the trainable slot
        assert e.learner_seat in (0, 1)
    # train on the batch: the learner changes, the snapshots do not and carry no gradients
    batch, _, _ = rollout_batch(roll, col.cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    before = [p.clone() for p in net.parameters()]
    ppo_update(net, opt, batch, col.cfg, torch.Generator().manual_seed(0))
    assert any(not torch.equal(b, p) for b, p in zip(before, net.parameters()))
    for a, sd in zip(pool.snapshots, frozen):
        assert all(torch.equal(sd[k], v) for k, v in a.net.state_dict().items())
        assert all(not p.requires_grad for p in a.net.parameters())


def test_learner_seat_in_historical_games_matches_the_trainable_slot():
    pool = _pool_with(_net(5))
    col = _collector(0.0, pool, n_steps=1)
    matches = list(col.match)
    roll, _, _ = col.collect(_net(), torch.Generator().manual_seed(4))
    for i, m in enumerate(matches):
        s = m.learner_seat
        assert roll["active"][0, 2 * i + s] and not roll["active"][0, 2 * i + 1 - s]
        assert m.controllers[s] is LEARNER and isinstance(m.controllers[1 - s], NetAgent)


def test_rollout_logprobs_are_the_rollout_policy_even_after_the_update():
    net = _net()
    col = _collector(0.8, _pool_with(_net(7)), n_steps=20)
    roll, _, _ = col.collect(net, torch.Generator().manual_seed(5))
    batch, _, _ = rollout_batch(roll, col.cfg)
    with torch.no_grad():
        logits, _ = net(batch["obs"])
        lp0 = MaskedCategorical(logits, batch["mask"]).log_prob(batch["actions"])
    assert torch.allclose(lp0, batch["logprobs"], atol=1e-6)      # old log-probs = the collecting network's
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    ppo_update(net, opt, batch, col.cfg, torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits, _ = net(batch["obs"])
        lp1 = MaskedCategorical(logits, batch["mask"]).log_prob(batch["actions"])
    assert not torch.allclose(lp1, batch["logprobs"], atol=1e-6)  # ... and the batch was not rewritten


def test_termination_truncation_and_bootstrap_values():
    net = _net()
    col = _collector(1.0, n_steps=12)                             # horizon 10: terminated at t = 9
    roll, _, _ = col.collect(net, torch.Generator().manual_seed(6))
    assert roll["terminated"][9].all() and not roll["terminated"][[0, 8, 10, 11]].any()
    assert not roll["truncated"].any()
    assert np.allclose(roll["next_values"][9], 0.0)               # nothing follows the horizon
    assert np.allclose(roll["next_values"][:9], roll["values"][1:10])
    # max_turns = 4: truncated at t = 3 and 7, bootstrapped with the value of the final observation
    col = _collector(1.0, n_steps=8, max_turns=4)
    roll, finished, _ = col.collect(net, torch.Generator().manual_seed(6))
    assert roll["truncated"][[3, 7]].all() and not roll["terminated"].any()
    assert len(finished) == 8
    assert not np.allclose(roll["next_values"][3], 0.0)
    assert np.allclose(roll["next_values"][:3], roll["values"][1:4])
    # GAE: the advantage of the truncated step uses only its own bootstrap (no leakage from the next game)
    from lanegame.rl import compute_gae
    adv, _ = compute_gae(roll["rewards"], roll["values"], roll["next_values"], roll["terminated"],
                         roll["truncated"], 1.0, 0.95)
    d = roll["rewards"][3] + roll["next_values"][3] - roll["values"][3]
    assert np.allclose(adv[3], d)


def test_snapshot_files_reproduce_action_distributions_and_metadata(tmp_path):
    out = str(tmp_path / "run")
    res = train_shared(CFG, _ppo(total_steps=240), SharedConfig(current_fraction=0.8, snapshot_every=80),
                       out_dir=out, verbose=False, name="t")
    assert len(res.snapshot_paths) == 3 and os.path.exists(os.path.join(out, "final.pt"))
    env = LaneEnv(CFG)
    obs = [env.reset(seed=s)[s % 2] for s in range(6)]
    for path, agent in zip(res.snapshot_paths, res.pool.snapshots):
        loaded = load_agent(path)
        for o in obs:
            assert np.array_equal(loaded.probabilities(o), agent.probabilities(o))
        _, cfg, meta = load_checkpoint(path)
        assert cfg == CFG and meta["kind"] == "snapshot" and meta["snapshot_id"] == agent.name
        assert meta["agent_transitions"] == int(agent.name.split("@")[1]) and meta["seed"] == 3
        assert meta["shared"]["current_fraction"] == 0.8 and meta["model"]["hidden"] == [128, 128]
    run = json.load(open(os.path.join(out, "run.json"), encoding="utf-8"))
    assert run["method"] == "shared_selfplay" and run["ppo"]["seed"] == 3
    # an existing run directory is refused; a checkpoint is never replaced silently
    with pytest.raises(FileExistsError):
        train_shared(CFG, _ppo(total_steps=40), SharedConfig(), out_dir=out, verbose=False)
    with pytest.raises(FileExistsError):
        save_checkpoint(res.snapshot_paths[0], res.net, CFG, exist_ok=False)


def test_mixture_statistics_bootstrap_and_seats():
    rng = np.random.default_rng(0)
    empty = MatchupSampler(SnapshotPool(4), 0.8)
    assert all(empty.sample(rng).kind == CURRENT for _ in range(200))      # bootstrap: no past version yet
    pool = _pool_with(_net(1), 2)
    s = MatchupSampler(pool, 0.8)
    n = 20_000
    draws = [s.sample(rng) for _ in range(n)]
    cur = np.mean([d.kind == CURRENT for d in draws])
    assert abs(cur - 0.8) < 4.5 * np.sqrt(0.8 * 0.2 / n)
    hist = [d for d in draws if d.kind == HISTORICAL]
    seat = np.mean([d.learner_seat for d in hist])
    assert abs(seat - 0.5) < 4.5 * np.sqrt(0.25 / len(hist))
    first = np.mean([d.opponent == "snap@100" for d in hist])
    assert abs(first - 0.5) < 4.5 * np.sqrt(0.25 / len(hist))
    for d in hist:
        assert d.controllers[d.learner_seat] is LEARNER and d.controllers[1 - d.learner_seat] is not LEARNER
    # the same seed draws the same matchups
    a = [MatchupSampler(pool, 0.8).sample(np.random.default_rng(9)).opponent for _ in range(1)]
    b = [MatchupSampler(pool, 0.8).sample(np.random.default_rng(9)).opponent for _ in range(1)]
    assert a == b


def test_training_mixture_and_accounting():
    res = train_shared(CFG, _ppo(total_steps=1600), SharedConfig(current_fraction=0.8, snapshot_every=200),
                       verbose=False)
    h = res.history
    assert res.env_turns == len(h) * 4 * 10 == h[-1]["env_turns"]
    assert res.agent_transitions == sum(e["rollout_agent_transitions"] for e in h) == h[-1]["agent_transitions"] >= 1600
    assert [e["agent_transitions"] for e in h] == list(np.cumsum([e["rollout_agent_transitions"] for e in h]))
    for e in h:
        assert e["rollout_transitions_current"] + e["rollout_transitions_historical"] == e["rollout_agent_transitions"]
        assert e["rollout_transitions_current"] % 2 == 0          # current-policy games contribute in pairs
        assert e["rollout_env_turns"] == 40
    assert h[0]["rollout_transitions_historical"] == 0            # bootstrap: the pool is empty at first
    assert sum(e["rollout_transitions_historical"] for e in h) > 0
    assert all(np.isfinite([e["policy_loss"], e["value_loss"], e["entropy"], e["approx_kl"]]).all() for e in h)
    pure = train_shared(CFG, _ppo(total_steps=400), SharedConfig(current_fraction=1.0, snapshot_every=80),
                        verbose=False)
    assert pure.agent_transitions == 2 * pure.env_turns            # 100% current: two samples per turn
    assert all(e["games_historical"] == 0 for e in pure.history)


def test_training_is_reproducible():
    runs = [train_shared(CFG, _ppo(total_steps=600, skill_levels=(0.2, 0.8)), SharedConfig(snapshot_every=150),
                         verbose=False) for _ in range(2)]
    for p, q in zip(runs[0].net.parameters(), runs[1].net.parameters()):
        assert torch.equal(p, q)
    strip = lambda h: [{k: v for k, v in e.items() if k not in ("elapsed", "sps", "collect_s", "update_s")} for e in h]
    assert json.dumps(strip(runs[0].history)) == json.dumps(strip(runs[1].history))
    other = train_shared(CFG, _ppo(total_steps=600, seed=4, skill_levels=(0.2, 0.8)), SharedConfig(snapshot_every=150),
                         verbose=False)
    assert any(not torch.equal(p, q) for p, q in zip(runs[0].net.parameters(), other.net.parameters()))


def test_seat_swap_symmetry_of_a_shared_policy():
    """One network in both seats of the symmetric game: no systematic seat advantage (both seats' payoffs over
    many games agree within sampling error), and seat B's decision equals seat A's on the mirrored game."""
    agent = NetAgent(_net(2), "p")
    st = run_episodes(LaneEnv(CFG, cache_size=5000), agent, agent, 1500, 123)
    assert abs(st.payoffs.mean()) < 4.5 * st.stderr + 1e-9
    env = LaneEnv(CFG)
    oa, ob = env.reset(seed=1)
    for _ in range(4):
        r = env.step(agent.act(oa), agent.act(ob))
        oa, ob = r.obs
        mirror = LaneEnv(CFG)
        mirror.reset(seed=0, root=ob.state.mirror(), k0=ob.k, champ=ob.champ.mirror())
        assert np.allclose(agent.probabilities(ob), agent.probabilities(mirror.observe()[0]))


def test_legacy_configuration_is_compatible():
    cfg = GameConfig(horizon=8)
    res = train_shared(cfg, PPOConfig(total_steps=256, n_envs=2, n_steps=16, minibatch_size=64, seed=0),
                       SharedConfig(current_fraction=0.8, snapshot_every=64), verbose=False)
    assert res.agent_transitions >= 256 and len(res.pool) >= 1
