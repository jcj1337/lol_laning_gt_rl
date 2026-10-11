"""M3a: masked PPO. Action masking, finite probabilities and gradients, GAE (termination versus truncation,
by hand), the clipped update, checkpoints, reproducible rollouts and training, seat handling, a learning-signal
check, and the exact bridge for a network policy on a legacy configuration. Skipped without PyTorch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lanegame import GameConfig, LaneGame                                   # noqa: E402
from lanegame.best_response import evaluate                                 # noqa: E402
from lanegame.baselines import passive_safe                                 # noqa: E402
from lanegame.env import LaneEnv, ScriptedAgent, as_policy, champion_agents, run_episodes   # noqa: E402
from lanegame.rl import (Collector, MaskedCategorical, NetAgent, OpponentSampler, PolicyValueNet, PPOConfig,  # noqa: E402
                         compute_gae, load_checkpoint, ppo_update, save_checkpoint, train_ppo)

CFG = GameConfig.champions(horizon=12)


def small(**kw):
    base = dict(total_steps=1024, n_envs=4, n_steps=32, minibatch_size=64, epochs=2, seed=3, lane_cache=0)
    base.update(kw)
    return PPOConfig(**base)


# ------------------------------------------------------------------ masking
def test_illegal_actions_have_zero_probability_and_are_never_sampled():
    torch.manual_seed(0)
    logits = torch.randn(64, 46) * 5
    mask = torch.rand(64, 46) < 0.2
    mask[:, 7] = True
    mask[0] = False
    mask[0, 45] = True                                                       # a single legal action (ABSENT)
    d = MaskedCategorical(logits, mask)
    assert torch.all(d.probs[~mask] == 0) and torch.allclose(d.probs.sum(-1), torch.ones(64))
    gen = torch.Generator().manual_seed(1)
    for _ in range(50):
        a = d.sample(gen)
        assert bool(mask[torch.arange(64), a].all())
    assert torch.isfinite(d.entropy()).all() and d.entropy()[0] == 0 and d.log_prob(torch.full((64,), 7))[0] < -1e8
    assert d.log_prob(torch.tensor([45] * 64))[0] == 0
    with pytest.raises(ValueError):
        MaskedCategorical(logits[:1], torch.zeros(1, 46, dtype=torch.bool))


def test_probabilities_and_gradients_are_finite():
    net = PolicyValueNet(50, 46)
    obs = torch.rand(32, 50)
    mask = torch.rand(32, 46) < 0.3
    mask[:, 0] = True
    logits, v = net(obs)
    d = MaskedCategorical(logits, mask)
    loss = -(d.log_prob(d.sample()) * torch.randn(32)).mean() - 0.01 * d.entropy().mean() + (v ** 2).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())
    single = torch.zeros(1, 46, dtype=torch.bool)
    single[0, 45] = True
    net.zero_grad()
    logits, _ = net(obs[:1])
    MaskedCategorical(logits, single).log_prob(torch.tensor([45])).sum().backward()
    assert all(float(p.grad.abs().max()) == 0 for p in net.policy_head.parameters())   # a forced action: no gradient


# ------------------------------------------------------------------ advantages
def test_gae_by_hand_termination_truncation_and_bootstrap():
    g, lam = 0.9, 0.8
    r = np.array([[1.0], [2.0], [3.0]])
    v = np.array([[0.5], [0.4], [0.3]])
    nv = np.array([[0.4], [0.3], [9.0]])
    # terminated at the last step: its successor value is ignored
    adv, ret = compute_gae(r, v, nv, np.array([[0], [0], [1]], bool), np.zeros((3, 1), bool), g, lam)
    d2 = 3.0 - 0.3
    d1 = 2.0 + g * 0.3 - 0.4
    d0 = 1.0 + g * 0.4 - 0.5
    a2, a1 = d2, d1 + g * lam * d2
    assert adv[:, 0] == pytest.approx([d0 + g * lam * a1, a1, a2]) and ret[:, 0] == pytest.approx(adv[:, 0] + v[:, 0])
    # truncated at the last step: bootstrapped with V(final observation) = 9
    adv, _ = compute_gae(r, v, nv, np.zeros((3, 1), bool), np.array([[0], [0], [1]], bool), g, lam)
    assert adv[2, 0] == pytest.approx(3.0 + g * 9.0 - 0.3)
    # an episode boundary in the middle stops propagation
    adv, _ = compute_gae(r, v, nv, np.array([[0], [1], [0]], bool), np.zeros((3, 1), bool), g, lam)
    assert adv[1, 0] == pytest.approx(2.0 - 0.4) and adv[0, 0] == pytest.approx(d0 + g * lam * (2.0 - 0.4))
    # gamma = lambda = 1 on one terminated episode: advantage = Monte Carlo return - V
    adv, ret = compute_gae(r, v, nv, np.array([[0], [0], [1]], bool), np.zeros((3, 1), bool), 1.0, 1.0)
    assert ret[:, 0] == pytest.approx([6.0, 5.0, 3.0])


# ------------------------------------------------------------------ the update
def test_ppo_update_moves_probability_toward_positive_advantage():
    torch.manual_seed(0)
    net = PolicyValueNet(8, 4, (16, 16))
    obs = torch.rand(256, 8)
    mask = torch.ones(256, 4, dtype=torch.bool)
    with torch.no_grad():
        logits, v = net(obs)
        d = MaskedCategorical(logits, mask)
    actions = torch.randint(0, 4, (256,))
    adv = torch.where(actions == 2, torch.ones(256), -torch.ones(256))
    batch = {"obs": obs, "mask": mask, "actions": actions, "logprobs": d.log_prob(actions), "values": v,
             "advantages": adv, "returns": v + adv, "choice": torch.ones(256)}
    before = float(d.probs[:, 2].mean())
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    stats = ppo_update(net, opt, batch, PPOConfig(epochs=4, minibatch_size=64), torch.Generator().manual_seed(0))
    with torch.no_grad():
        after = float(MaskedCategorical(net(obs)[0], mask).probs[:, 2].mean())
    assert after > before + 0.05 and all(np.isfinite(list(stats.values())))
    assert 0 <= stats["clipfrac"] <= 1 and stats["approx_kl"] >= -1e-6


# ------------------------------------------------------------------ rollouts and training
def test_rollouts_are_reproducible_and_mark_termination_and_truncation():
    agents = champion_agents(CFG)
    sampler = OpponentSampler([agents["hold_farmer"], agents["aggressive_trader"]])
    torch.manual_seed(0)
    net = PolicyValueNet(50, 46)
    cfg = small(n_steps=CFG.horizon * 2)
    r1, f1 = Collector(CFG, cfg, sampler, 5).collect(net, torch.Generator().manual_seed(0))
    r2, f2 = Collector(CFG, cfg, sampler, 5).collect(net, torch.Generator().manual_seed(0))
    for k in r1:
        assert np.array_equal(r1[k], r2[k]), k
    assert [e.learner_return for e in f1] == [e.learner_return for e in f2]
    # full games: terminated exactly at the horizon, no truncation, successor value 0 there
    assert r1["terminated"].sum() == cfg.n_envs * 2 and not r1["truncated"].any()
    assert np.all(r1["terminated"][CFG.horizon - 1]) and np.all(r1["next_values"][r1["terminated"]] == 0)
    for e in f1:                                     # learner returns are its own zero-sum game rewards
        assert e.learner_return == pytest.approx(e.log.payoff if e.seat == 0 else -e.log.payoff)
    # truncated episodes are bootstrapped with the value of the final observation
    rt, _ = Collector(CFG, small(n_steps=10, max_turns=4), sampler, 5).collect(net, torch.Generator().manual_seed(0))
    assert rt["truncated"].sum() == 4 * 2 and not rt["terminated"].any()
    assert np.all(rt["next_values"][rt["truncated"]] != 0)
    # every action taken was legal
    assert np.all(r1["mask"][np.arange(r1["mask"].shape[0])[:, None], np.arange(4)[None, :], r1["actions"]])


def test_training_is_reproducible_and_checkpoints_round_trip(tmp_path):
    agents = champion_agents(CFG)
    sampler = OpponentSampler([agents["yield_farmer"]])
    a = train_ppo(CFG, small(), sampler, out_dir=str(tmp_path / "a"), verbose=False)
    b = train_ppo(CFG, small(), sampler, verbose=False)
    assert [h["mean_return"] for h in a.history] == [h["mean_return"] for h in b.history]
    for p, q in zip(a.net.parameters(), b.net.parameters()):
        assert torch.equal(p, q)
    net, cfg, meta = load_checkpoint(str(tmp_path / "a" / "final.pt"))
    assert cfg == CFG and meta["ppo"]["seed"] == 3 and meta["steps"] == small().n_updates * small().batch_size
    x = torch.rand(5, 50)
    assert torch.equal(net(x)[0], a.net(x)[0]) and torch.equal(net(x)[1], a.net(x)[1])
    save_checkpoint(str(tmp_path / "c.pt"), net, cfg, {"name": "copy"})
    assert load_checkpoint(str(tmp_path / "c.pt"))[2] == {"name": "copy"}
    assert (tmp_path / "a" / "history.jsonl").read_text().count("\n") == len(a.history)


def test_short_training_shows_a_learning_signal():
    """Against an exploitable opponent (a Yield farmer, beaten by persistent Trade) a few thousand steps already
    raise the learner's return well above that of the initial (near-uniform) policy."""
    agents = champion_agents(GameConfig.champions())
    cfg = PPOConfig(total_steps=12_288, n_envs=8, n_steps=64, minibatch_size=256, seed=0)
    res = train_ppo(GameConfig.champions(), cfg, OpponentSampler([agents["yield_farmer"]]), verbose=False)
    first, last = res.history[0]["mean_return"], np.mean([h["mean_return"] for h in res.history[-3:]])
    assert last > first + 2.0


# ------------------------------------------------------------------ agents and the exact bridge
def test_net_agent_plays_legally_in_both_seats():
    torch.manual_seed(1)
    agent = NetAgent(PolicyValueNet(50, 46), "net")
    env = LaneEnv(CFG)
    for st in (run_episodes(env, agent, champion_agents(CFG)["hold_farmer"], 5, seed=2),
               run_episodes(env, champion_agents(CFG)["hold_farmer"], agent, 5, seed=2)):
        assert st.turns == 5 * CFG.horizon
    oa, _ = env.reset(seed=0)
    p = agent.distribution(oa)
    assert p.shape == (len(oa.legal),) and abs(p.sum() - 1) < 1e-9 and np.all(p >= 0)
    greedy = NetAgent(agent.net, "g", greedy=True)
    assert greedy.act(oa) == greedy.act(oa) == oa.legal[int(np.argmax(p))]
    assert all(not q.requires_grad for q in agent.net.parameters())


def test_network_policy_is_exactly_evaluable_on_a_legacy_configuration():
    """A network agent is a Markov policy of the exact game: its exact value against a scripted opponent agrees
    with 2,000 environment episodes within 4.5 standard errors (siege model, K = 6)."""
    cfg = GameConfig(horizon=6)
    torch.manual_seed(2)
    env = LaneEnv(cfg)
    agent = NetAgent(PolicyValueNet(env.obs_size, env.n_actions), "net")
    g = LaneGame(cfg)
    layers = g.reachable()
    exact = evaluate(g, layers, as_policy(agent, cfg), passive_safe())
    st = run_episodes(env, agent, ScriptedAgent(passive_safe()), 2000, seed=4)
    assert abs(st.mean - exact) < 4.5 * st.stderr
