"""WP1: continuing time, segments, bootstrapping and discounted evaluation (plan WP1, tests T1.1-T1.13).

T1.14 (the legacy pins) is tests/test_legacy_pins.py.
"""

import random

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import N_CHANCE, champion_legal_actions, initial_champions, resolve_turn
from lanegame.env import LaneEnv, UniformAgent, champion_suite, play_episode
from lanegame.env.observation import feature_names, observe
from lanegame.mechanics import initial_state

M2C = GameConfig.champions_continuing()
M4 = GameConfig.m4()


def random_play(env, turns, seed=0):
    rng = np.random.default_rng(seed)
    ag = UniformAgent()
    obs = env.reset(seed=seed)
    results = []
    for _ in range(turns):
        r = env.step(ag.act(obs[0], rng), ag.act(obs[1], rng))
        results.append(r)
        if r.done:
            break
        obs = r.obs
    return results


@pytest.mark.parametrize("cfg", [M2C, M4], ids=["m2c", "m4"])
def test_t1_1_never_terminated(cfg):
    env = LaneEnv(cfg, max_turns=2000, cache_size=5000)
    res = random_play(env, 2000)
    assert len(res) == 2000 and not any(r.terminated for r in res)
    assert env.horizon is None and env.continuing


@pytest.mark.parametrize("cfg", [M2C, M4], ids=["m2c", "m4"])
def test_t1_2_truncated_exactly_at_max_turns_and_step_raises_after(cfg):
    env = LaneEnv(cfg, max_turns=37)
    res = random_play(env, 100)
    assert len(res) == 37 and res[-1].truncated and not any(r.truncated for r in res[:-1])
    with pytest.raises(RuntimeError):
        env.step(0, 0)
    env.reset(seed=1)
    env.step(0, 0)


@pytest.mark.parametrize("cfg", [M2C, M4], ids=["m2c", "m4"])
def test_t1_3_observation_has_no_time_and_is_periodic_in_k(cfg):
    names = feature_names(cfg)
    assert not any(w in n for n in names for w in ("turn", "remaining", "horizon", "countdown"))
    env = LaneEnv(cfg, max_turns=300)
    res = random_play(env, 300, seed=4)
    P = cfg.phase_period
    for r in res[::7]:
        o = r.obs[0]
        for shift in (P, 5 * P, 101 * P):
            assert np.array_equal(observe(o.state, o.k + shift, 0, cfg, o.champ), o.vector)
    # reset(k0) keeps only k0 mod P
    env.reset(seed=0, k0=4 * P + 3)
    assert env.k == 3 and env.k0 == 3


def test_reset_rejects_negative_k0_and_accepts_any_other():
    env = LaneEnv(M4, max_turns=5)
    with pytest.raises(ValueError):
        env.reset(seed=0, k0=-1)
    env.reset(seed=0, k0=10 ** 9)


# ---------------------------------------------------------------------- GAE semantics on toy chains
def gae(*a, **k):
    from lanegame.rl.ppo import compute_gae
    return compute_gae(*a, **k)


def test_t1_4_exact_critic_gives_zero_advantages_and_terminal_treatment_is_biased():
    pytest.importorskip("torch")
    g, lam, c = 0.99, 0.95, 0.3
    V = c / (1 - g)
    T, N = 50, 3
    rewards = np.full((T, N), c)
    values = np.full((T, N), V)
    nxt = np.full((T, N), V)
    trunc = np.zeros((T, N), dtype=bool)
    trunc[17, 0] = trunc[33, 1] = trunc[49, 2] = True        # time limits and a segment end
    term = np.zeros((T, N), dtype=bool)
    adv, ret = gae(rewards, values, nxt, term, trunc, g, lam)
    assert np.allclose(adv, 0.0, atol=1e-9) and np.allclose(ret, V)
    # the same truncation treated as termination: the target at that step loses gamma * V
    adv2, _ = gae(rewards, values, nxt, trunc.copy(), trunc, g, lam)
    assert adv2[17, 0] == pytest.approx(c - V) == pytest.approx(-g * V)
    assert np.allclose(adv2[18:33, 0], 0.0, atol=1e-9)


def test_t1_6_segment_bootstrap_and_no_lambda_recursion_across_games():
    pytest.importorskip("torch")
    g, lam = 0.99, 0.95
    rng = np.random.default_rng(0)
    T = 40
    rewards = rng.normal(size=(T, 1))
    values = rng.normal(size=(T, 1))
    nxt = np.vstack([values[1:], rng.normal(size=(1, 1))])   # the last step: V(s_T) of the current observation
    trunc = np.zeros((T, 1), dtype=bool)
    trunc[19, 0] = True                                         # game 1 = steps 0..19, game 2 = 20..39
    nxt[19, 0] = 1.234                                          # bootstrap of game 1's pre-reset observation
    term = np.zeros_like(trunc)
    adv, _ = gae(rewards, values, nxt, term, trunc, g, lam)
    spiked = rewards.copy()
    spiked[25, 0] += 100.0
    adv_s, _ = gae(spiked, values, nxt, term, trunc, g, lam)
    assert np.array_equal(adv[:20], adv_s[:20])                 # game 1 unaffected by game 2
    assert not np.allclose(adv[20:26], adv_s[20:26])
    # the segment end bootstraps V(s_T): last advantage is the one-step TD error
    assert adv[-1, 0] == pytest.approx(rewards[-1, 0] + g * nxt[-1, 0] - values[-1, 0])


def test_t1_7_biased_critic_truncation_step_formula():
    pytest.importorskip("torch")
    g, lam, delta = 0.99, 0.95, 0.7
    T = 10
    r = np.arange(T, dtype=float).reshape(T, 1) * 0.1
    Vtrue = np.linspace(1, 2, T + 1).reshape(T + 1, 1)
    values = Vtrue[:-1] + delta
    nxt = Vtrue[1:] + delta
    trunc = np.zeros((T, 1), dtype=bool)
    trunc[-1, 0] = True
    adv, _ = gae(r, values, nxt, np.zeros_like(trunc), trunc, g, lam)
    s, sp = T - 1, T
    assert adv[-1, 0] == pytest.approx(r[-1, 0] + g * (Vtrue[sp, 0] + delta) - (Vtrue[s, 0] + delta))


# ---------------------------------------------------------------------- collectors
def small_ppo(**kw):
    from lanegame.rl import PPOConfig
    base = dict(total_steps=200, n_envs=3, epochs=1, minibatch_size=32, hidden=(16,), seed=3, lane_cache=0)
    base.update(kw)
    return PPOConfig.continuing(M2C, **base) if kw.pop("continuing", True) else PPOConfig(**base)


def test_t1_5_truncation_bootstraps_from_the_pre_reset_observation():
    torch = pytest.importorskip("torch")
    from lanegame.rl import OpponentSampler, PolicyValueNet
    from lanegame.rl.train import Collector
    cfg = small_ppo(segment_length=30, game_length=6)
    col = Collector(M2C, cfg, OpponentSampler([champion_suite()[0]]), seed=11)
    finals = []
    for env in col.envs:
        orig = env.step

        def spy(a, b, _orig=orig):
            r = _orig(a, b)
            if r.truncated:
                finals.append(r.obs)
            return r
        env.step = spy

    class SpyNet(PolicyValueNet):
        seen = []

        def value(self, x):
            SpyNet.seen.append(x.numpy().copy())
            return super().value(x)

    net = SpyNet(col.envs[0].obs_size, col.envs[0].n_actions, (16,))
    roll, finished = col.collect(net, torch.Generator().manual_seed(0))
    assert roll["truncated"].sum() > 0 and not roll["terminated"].any()
    boot = np.concatenate(SpyNet.seen[:-1]) if len(SpyNet.seen) > 1 else np.zeros((0,))
    # every truncation's bootstrap input equals the final pre-reset observation of the learner's seat, never a
    # first observation of the next game
    pre = [o[col.seat[0]].vector for o in finals]          # seats may differ per env; check membership instead
    pre_all = {o[s].vector.tobytes() for o in finals for s in (0, 1)}
    for row in boot:
        assert row.astype(np.float32).tobytes() in pre_all
    first_obs = LaneEnv(M2C).reset(seed=0)[0].vector
    assert all(row.astype(np.float32).tobytes() != first_obs.tobytes() for row in boot)
    assert len(pre) == int(roll["truncated"].sum())


def test_t1_8_matchup_change_mid_game_raises():
    torch = pytest.importorskip("torch")
    from lanegame.rl import OpponentSampler, PolicyValueNet, SharedConfig
    from lanegame.rl.shared import Matchup, MatchupSampler, SharedCollector
    from lanegame.rl.selfplay import SnapshotPool
    from lanegame.rl.train import Collector
    suite = champion_suite()
    cfg = small_ppo(segment_length=5, game_length=50)
    col = Collector(M2C, cfg, OpponentSampler(suite[:2]), seed=1)
    net = PolicyValueNet(col.envs[0].obs_size, col.envs[0].n_actions, (16,))
    gen = torch.Generator().manual_seed(0)
    col.collect(net, gen)
    col.opp[0] = suite[5]                                    # forced change while the game is running
    with pytest.raises(RuntimeError, match="game"):
        col.collect(net, gen)
    sc = SharedCollector(M2C, cfg, MatchupSampler(SnapshotPool(4), 1.0), seed=1)
    sc.collect(net, gen)
    sc.match[1] = Matchup("historical", (None, suite[0]), "x", 0)
    with pytest.raises(RuntimeError, match="game"):
        sc.collect(net, gen)


def test_t1_9_first_game_lengths_are_seeded_and_staggered():
    pytest.importorskip("torch")
    from lanegame.rl import OpponentSampler
    from lanegame.rl.train import Collector
    cfg = small_ppo(n_envs=8, game_length=512)
    l1 = Collector(M2C, cfg, OpponentSampler(champion_suite()[:1]), seed=5).times.first_lengths
    l2 = Collector(M2C, cfg, OpponentSampler(champion_suite()[:1]), seed=5).times.first_lengths
    l3 = Collector(M2C, cfg, OpponentSampler(champion_suite()[:1]), seed=6).times.first_lengths
    assert l1 == l2 and l1 != l3 and len(set(l1)) > 1 and all(1 <= x <= 512 for x in l1)
    assert l1 == [int(x) for x in np.random.default_rng(np.random.SeedSequence(5).spawn(4)[2]).integers(1, 513, 8)]
    col = Collector(M2C, cfg, OpponentSampler(champion_suite()[:1]), seed=5)
    assert [e.max_turns for e in col.envs] == l1                # the first games use them


def test_t1_10_m2c_dynamics_equal_champions():
    legacy, cont = GameConfig.champions(), M2C
    rng = random.Random(10)
    nrng = np.random.default_rng(10)
    lane, champ, k = initial_state(legacy), initial_champions(legacy), 0
    for i in range(5000):
        la = champion_legal_actions(lane, champ, 0, legacy)
        lb = champion_legal_actions(lane, champ, 1, legacy)
        assert la == champion_legal_actions(lane, champ, 0, cont) and lb == champion_legal_actions(lane, champ, 1, cont)
        a, b = rng.choice(la), rng.choice(lb)
        u = nrng.random(N_CHANCE)
        t1 = resolve_turn(lane, champ, k, a, b, legacy, u=u)
        t2 = resolve_turn(lane, champ, k, a, b, cont, u=u)
        for f in t1.__dataclass_fields__:
            if f == "lane":
                assert t1.lane.state == t2.lane.state and t1.lane.kills == t2.lane.kills
            else:
                assert getattr(t1, f) == getattr(t2, f), f
        for side in (0, 1):
            ol, oc = observe(lane, k, side, legacy, champ), observe(lane, k, side, cont, champ)
            assert np.array_equal(ol[2:], oc)                   # only the two countdown entries differ
        lane, champ, k = t1.lane.state, t1.champ, k + 1
        if k >= legacy.horizon:
            lane, champ, k = initial_state(legacy), initial_champions(legacy), 0
    # environment rewards: identical action sequences and seeds give identical rewards; only termination differs
    suite = champion_suite()
    e1, e2 = LaneEnv(legacy), LaneEnv(cont, max_turns=legacy.horizon)
    for s in range(20):
        l1 = play_episode(e1, suite[s % 9], suite[(s * 4) % 9], s)
        l2 = play_episode(e2, suite[s % 9], suite[(s * 4) % 9], s)
        assert l1.rewards == l2.rewards and l1.actions == l2.actions


def test_t1_11_discounted_evaluator(monkeypatch):
    from lanegame.env import discounted as D
    from lanegame.env.starts import StartState
    x, y = UniformAgent("x"), UniformAgent("y")
    c, W, g = 0.37, 300, M4.gamma

    def stub(env, agents, seed, turns, start=None, record_every=0):
        sign = 1.0 if agents[0] is x else -1.0
        return np.full(turns, sign * c), []
    monkeypatch.setattr(D, "_play", stub)
    r = D.evaluate_discounted(M4, x, y, episodes=4, window=W)
    assert r["estimate"] == pytest.approx(c * (1 - g ** W) / (1 - g), rel=1e-12)
    assert r["tail_bound"]["guaranteed"] == pytest.approx(g ** W * D.tail_constant(M4))
    assert D.tail_constant(M4) == pytest.approx(68.339, abs=1e-3)
    st = StartState(initial_state(M4), __import__("lanegame.champions_m4", fromlist=["x"]).initial_m4(M4), 1)
    with pytest.raises(ValueError, match="E1"):
        D.evaluate_discounted(M4, x, y, episodes=2, window=10, protocol="E1", archive=[st])
    lane2 = initial_state(M4)._replace(ah=3)
    asym = StartState(lane2, st.champ, 1)
    with pytest.raises(ValueError, match="mirror"):
        D.evaluate_discounted(M4, x, y, episodes=2, window=10, protocol="E2", archive=[asym], train_archive=[])
    closed = [asym, asym.mirror()]
    with pytest.raises(ValueError, match="overlap"):
        D.evaluate_discounted(M4, x, y, episodes=2, window=10, protocol="E2", archive=closed, train_archive=[asym])
    ok = D.evaluate_discounted(M4, x, y, episodes=2, window=10, protocol="E2", archive=closed, train_archive=[])
    assert ok["starts"] == 2 and "E2" in ok["label"]
    with pytest.raises(ValueError, match="gamma"):
        D.evaluate_discounted(M4, x, y, episodes=2, window=10, gamma=0.98)


def test_evaluator_runs_on_the_real_environment_and_is_antisymmetric():
    from lanegame.env.discounted import evaluate_discounted
    suite = champion_suite()
    r = evaluate_discounted(M2C, suite[0], suite[0], episodes=3, window=40)
    assert abs(r["estimate"]) < 1e-9                            # a policy against itself on paired seeds scores 0
    a = evaluate_discounted(M2C, suite[0], suite[2], episodes=3, window=40)
    b = evaluate_discounted(M2C, suite[2], suite[0], episodes=3, window=40)
    assert a["estimate"] == pytest.approx(-b["estimate"])


def test_t1_12_play_episode_without_max_turns_raises_under_continuing_time():
    with pytest.raises(ValueError, match="max_turns"):
        play_episode(LaneEnv(M4), UniformAgent(), UniformAgent(), 0)
    log = play_episode(LaneEnv(M4, max_turns=12), UniformAgent(), UniformAgent(), 0)
    assert log.length == 12 and log.horizon is None


def test_t1_13_skill_levels_keep_gank_reward():
    pytest.importorskip("torch")
    from lanegame.rl import OpponentSampler, PPOConfig
    from lanegame.rl.train import Collector
    from lanegame.env.agents import ScriptedAgent
    from lanegame.baselines import siege_baselines
    cfg = PPOConfig(total_steps=64, n_envs=2, n_steps=8, expected_gank_reward=True, skill_levels=(0.3, 0.7), seed=0)
    col = Collector(GameConfig(), cfg, OpponentSampler([ScriptedAgent(siege_baselines()[0])]), seed=0)
    assert all(e.gank_reward == "expected" for e in col.envs)


def test_continuing_preset_and_time_semantics_checks():
    pytest.importorskip("torch")
    from lanegame.rl import PPOConfig
    from lanegame.rl.ppo import check_time_semantics
    p = PPOConfig.continuing(M4)
    assert (p.gamma, p.gae_lambda, p.segment_length, p.game_length, p.rollout_length) == (0.99, 0.95, 128, 512, 128)
    with pytest.raises(ValueError):
        PPOConfig.continuing(GameConfig.champions())
    with pytest.raises(ValueError, match="game_length"):
        check_time_semantics(M4, PPOConfig(gamma=0.99))
    with pytest.warns(UserWarning, match="gamma"):
        check_time_semantics(M4, PPOConfig(gamma=0.98, game_length=100))
