"""OM2: the opt-in observation of the opponent's previous action, Snapshot restoration with last actions, network
widening, the trainer's collector hooks (default bit-identical), the belief module and the OM2 collector.
Checkpoint-dependent tests skip when the git-ignored files are absent."""

import os

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.config import (B_HOLD, B_RECALL, B_TRADE, CHAMP_ABSENT, CHAMP_WARD, PRESSURE, SH, SP, W_SLOWPUSH,
                             W_THIN, champ_act)
from lanegame.env import LaneEnv
from lanegame.env.observation import feature_names, obs_size
from lanegame.env.observation_m4 import LAST_ACTION_NAMES, LAST_ACTION_SIZE, encode_last_action
from lanegame.opponent import Belief, UniformType, loglik, prior_features

BASE = GameConfig.m45_calibrated()
OM = GameConfig.m45_calibrated(obs_last_action=True)
START = os.path.join("results", "rl", "checkpoints", "m46c", "train", "ext", "baseline", "s0", "final.pt")


def _names(v):
    return [n for n, x in zip(LAST_ACTION_NAMES, v) if x]


# ---------------------------------------------------------------------------------------------- the observation field
def test_field_is_off_by_default_and_refused_outside_m4():
    assert not BASE.obs_last_action and obs_size(BASE) == 90
    assert obs_size(OM) == 90 + LAST_ACTION_SIZE == 103
    assert len(feature_names(OM)) == 103 and feature_names(OM)[90] == "opp_last_none"
    with pytest.raises(ValueError):
        GameConfig.v2(obs_last_action=True).validate()
    with pytest.raises(ValueError):
        GameConfig.m45_calibrated(obs_last_action=1).validate()


def test_encoding_of_every_kind_of_action():
    assert _names(encode_last_action(-1, OM)) == ["none"]
    assert _names(encode_last_action(CHAMP_ABSENT, OM)) == ["absent"]
    assert _names(encode_last_action(CHAMP_WARD, OM)) == ["ward"]
    assert _names(encode_last_action(champ_act(SP, B_RECALL), OM)) == ["recall"]
    assert _names(encode_last_action(champ_act(SH, B_TRADE, PRESSURE), OM)) == ["trade", "shove", "pressure"]
    assert _names(encode_last_action(champ_act(W_THIN, B_HOLD), OM)) == ["hold", "thin"]
    assert _names(encode_last_action(champ_act(W_SLOWPUSH, B_HOLD), OM)) == ["hold", "slow_push"]
    assert _names(encode_last_action(champ_act(SP, B_HOLD), OM)) == ["hold", "last_hit"]


def test_observation_is_the_unchanged_state_encoding_then_the_opponents_previous_action():
    e0, e1 = LaneEnv(BASE, max_turns=60), LaneEnv(OM, max_turns=60)
    o0, o1 = e0.reset(seed=4), e1.reset(seed=4)
    assert e1.last_actions == (-1, -1)
    for s in (0, 1):
        assert np.array_equal(o0[s].vector, o1[s].vector[:90]) and _names(o1[s].vector[90:]) == ["none"]
    rng = np.random.default_rng(2)
    for _ in range(40):
        a, b = int(rng.choice(e0.legal(0))), int(rng.choice(e0.legal(1)))
        r0, r1 = e0.step(a, b), e1.step(a, b)
        assert r0.rewards == r1.rewards and r0.info["reward_units"] == r1.info["reward_units"]
        assert e1.last_actions == (a, b)
        assert np.array_equal(r1.obs[0].vector[90:], np.asarray(encode_last_action(b, OM), np.float32))
        assert np.array_equal(r1.obs[1].vector[90:], np.asarray(encode_last_action(a, OM), np.float32))
        for s in (0, 1):
            assert np.array_equal(r0.obs[s].vector, r1.obs[s].vector[:90])


def test_snapshot_restores_mirrors_and_refuses_unknown_last_actions():
    from lanegame.diagnostics.branch import Snapshot, observation, restore
    env = LaneEnv(OM, max_turns=60)
    env.reset(seed=8)
    rng = np.random.default_rng(3)
    for _ in range(15):
        env.step(int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1))))
    snap = Snapshot.from_env(env)
    assert snap.last_actions == env.last_actions
    assert Snapshot.from_dict(snap.to_dict()) == snap
    obs = env.observe()
    for s in (0, 1):
        assert np.array_equal(observation(OM, snap, s).vector, obs[s].vector)
    m = snap.mirror()
    assert np.array_equal(observation(OM, m, 0).vector, observation(OM, snap, 1).vector)
    env2 = LaneEnv(OM, max_turns=60)
    o2 = restore(env2, snap, seed=1)
    assert np.array_equal(o2[0].vector, obs[0].vector) and env2.last_actions == env.last_actions
    with pytest.raises(ValueError):
        observation(OM, Snapshot(snap.lane, snap.champ, snap.k), 0)
    assert np.array_equal(observation(BASE, Snapshot(snap.lane, snap.champ, snap.k), 0).vector, obs[0].vector[:90])


# ---------------------------------------------------------------------------------------------- network and trainer
def test_widened_network_computes_the_source_exactly():
    torch = pytest.importorskip("torch")
    from lanegame.rl import PolicyValueNet
    from lanegame.rl.network import widen_inputs
    torch.manual_seed(0)
    src = PolicyValueNet(90, 61)
    w = widen_inputs(src, 115)
    x = torch.rand(7, 90)
    lg0, v0 = src(x)
    lg1, v1 = w(torch.cat([x, torch.rand(7, 25)], 1))
    assert torch.equal(lg0, lg1) and torch.equal(v0, v1) and w.obs_size == 115
    with pytest.raises(ValueError):
        widen_inputs(src, 80)


def test_collector_hooks_leave_default_training_bit_identical():
    """The digest was produced by the committed trainer before the hooks existed (HEAD e051b427) and reproduced with
    them: a run with current-policy and historical games."""
    torch = pytest.importorskip("torch")
    from lanegame.rl import PolicyValueNet, SharedConfig, train_shared, weights_digest
    from lanegame.rl.ppo import PPOConfig
    torch.set_num_threads(1)
    torch.manual_seed(3)
    ppo = PPOConfig.continuing(BASE, total_steps=2 * 4 * 32 * 2, n_envs=4, segment_length=32, game_length=40,
                               minibatch_size=64, seed=11, lane_cache=500)
    res = train_shared(BASE, ppo, SharedConfig(current_fraction=0.7, snapshot_every=128, pool_size=2),
                       net=PolicyValueNet(90, 61), verbose=False)
    assert weights_digest(res.net.state_dict()) == "4e0c2d1acc42d30661e583f267eac8d5bf79843ce6c1dcf3f47bd603650c045a"


# ---------------------------------------------------------------------------------------------- the belief
class _Fixed:
    kind = "fixed"

    def __init__(self, name, weights):
        self.name, self.w = name, weights

    def probs(self, cfg, lane, m4, s, k):
        from lanegame.champions import champion_legal_actions
        legal = tuple(int(c) for c in champion_legal_actions(lane, m4, s, cfg))
        p = np.array([self.w.get(c, 1.0) for c in legal], float)
        return legal, p / p.sum()


def test_belief_is_exact_bayes_forgets_and_skips_uninformative_turns():
    b = Belief(3, lam=1.0)
    l1, l2 = np.log([0.5, 0.25, 0.25]), np.log([0.2, 0.4, 0.4])
    b.update(l1)
    b.update(l2)
    post = np.array([0.5 * 0.2, 0.25 * 0.4, 0.25 * 0.4])
    assert np.allclose(b.posterior(), post / post.sum())
    f = Belief(3, lam=0.5)
    f.update(l1)
    f.update(l2)
    lp = 0.5 * np.log([0.5, 0.25, 0.25] / np.sum([0.5, 0.25, 0.25])) + l2
    assert np.allclose(f.posterior(), np.exp(lp) / np.exp(lp).sum())
    before = f.lp.copy()
    f.update(np.zeros(3))
    assert np.array_equal(before, f.lp) and f.updates == 2
    assert np.allclose(prior_features(4), 0.25)


def test_catch_alls_take_mixtures_and_random_play_but_not_a_single_style():
    """"mixed" (a different library style each turn) wins against a per-turn mixture of two styles, "unknown"
    (uniform) against random play, and the style itself against a consistent opponent."""
    from lanegame.experiments.m45b import states as S
    from lanegame.champions import champion_legal_actions
    from lanegame.opponent import LibraryMixtureType
    lane, m4, k = S.s0(BASE)
    legal = champion_legal_actions(lane, m4, 1, BASE)
    a, c = legal[0], legal[1]
    eps, n = 0.05, len(legal)
    ta = _Fixed("a", {x: (1 - eps) * n / eps if x == a else 1.0 for x in legal})
    tc = _Fixed("c", {x: (1 - eps) * n / eps if x == c else 1.0 for x in legal})
    types = [ta, tc, LibraryMixtureType((0, 1)), UniformType()]
    rng = np.random.default_rng(0)
    beliefs = {"mixture": Belief(4, 0.97), "single": Belief(4, 0.97), "random": Belief(4, 0.97)}
    for t in range(40):
        acts = {"mixture": a if rng.random() < 0.5 else c, "single": a, "random": legal[int(rng.integers(n))]}
        for key, b in beliefs.items():
            b.update(loglik(types, BASE, [(lane, m4, 1, k, acts[key])])[0])
    assert int(np.argmax(beliefs["mixture"].posterior())) == 2
    assert int(np.argmax(beliefs["single"].posterior())) == 0 and beliefs["single"].posterior()[0] > 0.9
    assert int(np.argmax(beliefs["random"].posterior())) == 3 and beliefs["random"].posterior()[3] > 0.9


@pytest.mark.skipif(not os.path.exists(START), reason="checkpoints absent")
def test_precomputed_observation_likelihood_equals_rebuilt():
    pytest.importorskip("torch")
    from lanegame.experiments.rl.om2 import common as C
    types = C.belief_types()
    env = LaneEnv(OM, max_turns=60)
    obs = env.reset(seed=5)
    rng = np.random.default_rng(1)
    for _ in range(12):
        r = env.step(int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1))))
        obs = r.obs
    o = obs[1]
    code = int(o.legal[0])
    a = loglik(types, OM, [(o.state, o.champ, 1, o.k, code)])
    b = loglik(types, OM, [(o.state, o.champ, 1, o.k, code, o.vector, o.legal)])
    assert np.allclose(a, b, atol=1e-12)


@pytest.mark.skipif(not os.path.exists(START), reason="checkpoints absent")
def test_om2_collector_feeds_the_belief_to_the_treatment_and_the_prior_to_the_control():
    torch = pytest.importorskip("torch")
    from lanegame.experiments.rl.om2 import common as C
    from lanegame.experiments.rl.om2.collector import om_collector_cls
    from lanegame.rl import PPOConfig, SharedConfig
    from lanegame.rl.shared import MatchupSampler
    from lanegame.rl.selfplay import SnapshotPool
    torch.set_num_threads(1)
    cfg = C.setup()
    ppo = PPOConfig.continuing(cfg, total_steps=1000, n_envs=3, segment_length=24, game_length=60, seed=4,
                               lane_cache=500, start_mix=0.0)
    net = __import__("lanegame.experiments.rl.om2.train", fromlist=["x"]).initial_net(0)
    out = {}
    for arm in C.ARMS:
        col = om_collector_cls(arm)(cfg, ppo, MatchupSampler(SnapshotPool(2, keep_first=False), 1.0), 9)
        roll, finished, counts = col.collect(net, torch.Generator().manual_seed(1))
        x = roll["obs"][roll["active"]]
        assert x.shape[1] == C.NET_IN
        out[arm] = x[:, C.OBS:]
    assert np.all(out["control"] == prior_features(C.K))
    assert out["treatment"].std(0).max() > 0 and np.allclose(out["treatment"].sum(1), 1, atol=1e-5)


@pytest.mark.skipif(not os.path.exists(START), reason="checkpoints absent")
def test_reference_player_plays_like_the_widened_start_in_the_control():
    """The exploratory reference plays pi_s (90 inputs) on the evaluation's games; at initialization the control is
    pi_s widened with zero weights, so both must play the same game."""
    pytest.importorskip("torch")
    from lanegame.experiments.m45b import engine as E
    from lanegame.experiments.rl.om2 import common as C
    from lanegame.experiments.rl.om2 import evaluate as EV
    from lanegame.experiments.rl.om2.train import initial_net
    cfg = C.setup()
    types = C.belief_types()
    ls = E.lane_cache(cfg)
    pi = C.net90(C.start_path(0))
    wide = initial_net(0)
    a = EV.play(cfg, ("pi", pi), ("opp", "g_bully"), 77, 40, types, ls)["r"]
    b = EV.play(cfg, ("arm", wide, "control"), ("opp", "g_bully"), 77, 40, types, ls)["r"]
    assert np.array_equal(a, b)
