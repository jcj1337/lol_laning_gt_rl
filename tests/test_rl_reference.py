"""M4.3a PPO correctness audit: independent numerical references for the pieces a passing training run cannot
certify on its own.

  * GAE against the forward-view lambda-return (an independent formula) on random data with random termination and
    truncation flags, including gamma = lambda = 1 (Monte Carlo) and lambda = 0 (one-step TD).
  * The clipped PPO objective, the clipped value loss and the entropy of `ppo_update` against a re-implementation
    on torch.distributions.Categorical with -inf masking: equal loss AND equal parameter gradients on one minibatch.
  * Masking: illegal actions have exactly zero probability and receive exactly zero gradient; the legal distribution
    is invariant to the illegal logits; log-probabilities of taken actions are the same whether computed through
    MaskedCategorical (training) or NetAgent.probabilities (acting).
  * Advantage normalization and the choice weights: steps with a single legal action do not move the policy and do
    not enter the policy statistics.
  * Checkpoint integrity: the digest identifies the weights and an altered file is refused.

Skipped without PyTorch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lanegame import GameConfig                                                               # noqa: E402
from lanegame.env import LaneEnv                                                              # noqa: E402
from lanegame.rl import (MaskedCategorical, NetAgent, PolicyValueNet, PPOConfig, compute_gae,   # noqa: E402
                         load_checkpoint, ppo_update, save_checkpoint, weights_digest)


# ------------------------------------------------------------------ GAE against the forward view
def lambda_return_forward(r, v, nv, term, trunc, g, lam):
    """Reference: for each step the forward-view lambda-return (1 - lam) sum_n lam^(n-1) G^(n) within the episode,
    where the n-step return bootstraps nv of its last step (0 after a termination) and the last available n-step
    return gets the remaining weight lam^(n-1) (Sutton & Barto 12.2 at a horizon). Episode boundaries: done."""
    T = r.shape[0]
    done = term | trunc
    ret = np.zeros(T)
    for t in range(T):
        # the episode containing t ends at the first done step e >= t (or at T - 1 by the rollout cut)
        e = t
        while e < T - 1 and not done[e]:
            e += 1
        total, disc = 0.0, 1.0
        # G^(n) for n = 1 .. e - t + 1, all bootstrapping nv[t + n - 1] after their last reward
        partial = 0.0
        for n in range(1, e - t + 2):
            k = t + n - 1
            partial += disc * r[k]
            boot = 0.0 if term[k] else nv[k]
            G_n = partial + disc * g * boot
            weight = (1 - lam) * lam ** (n - 1) if k < e else lam ** (n - 1)
            total += weight * G_n
            disc *= g
        ret[t] = total
    return ret, ret - v


@pytest.mark.parametrize("g,lam", [(0.99, 0.95), (1.0, 1.0), (0.9, 0.0), (0.5, 0.7)])
def test_gae_equals_the_forward_view_lambda_return(g, lam):
    rng = np.random.default_rng(12)
    T, N = 60, 5
    r = rng.normal(size=(T, N))
    v = rng.normal(size=(T, N))
    term = rng.random((T, N)) < 0.05
    trunc = (rng.random((T, N)) < 0.05) & ~term
    nv = np.where(term, 0.0, rng.normal(size=(T, N)))
    # within an episode nv[t] is v[t + 1]; at boundaries it is a free bootstrap value
    for n in range(N):
        for t in range(T - 1):
            if not (term[t, n] or trunc[t, n]):
                nv[t, n] = v[t + 1, n]
    adv, ret = compute_gae(r, v, nv, term, trunc, g, lam)
    for n in range(N):
        ret_ref, adv_ref = lambda_return_forward(r[:, n], v[:, n], nv[:, n], term[:, n], trunc[:, n], g, lam)
        assert np.allclose(ret[:, n], ret_ref, atol=1e-10) and np.allclose(adv[:, n], adv_ref, atol=1e-10)


def test_gae_special_cases_monte_carlo_and_one_step_td():
    rng = np.random.default_rng(1)
    T = 20
    r, v = rng.normal(size=(T, 1)), rng.normal(size=(T, 1))
    nv = np.vstack([v[1:], [[0.0]]])
    term = np.zeros((T, 1), bool)
    term[-1] = True
    # gamma = lambda = 1 on one terminated episode: returns are reversed cumulative sums of the rewards
    _, ret = compute_gae(r, v, nv, term, np.zeros_like(term), 1.0, 1.0)
    assert np.allclose(ret[:, 0], np.cumsum(r[::-1, 0])[::-1])
    # lambda = 0: the advantage is the one-step TD error
    adv, _ = compute_gae(r, v, nv, term, np.zeros_like(term), 0.9, 0.0)
    td = r[:, 0] + 0.9 * np.where(term[:, 0], 0.0, nv[:, 0]) - v[:, 0]
    assert np.allclose(adv[:, 0], td)


# ------------------------------------------------------------------ the PPO loss and its gradient
def reference_loss(net, batch, cfg):
    """The clipped PPO objective written independently of lanegame.rl.ppo, on torch.distributions.Categorical with
    -inf on illegal logits (CleanRL ppo.py conventions: per-minibatch advantage normalization, max of the clipped and
    unclipped value errors, entropy bonus)."""
    logits, v = net(batch["obs"])
    masked = logits.masked_fill(~batch["mask"], float("-inf"))
    dist = torch.distributions.Categorical(logits=masked)
    w = batch["choice"]
    nw = w.sum()
    newlogp = dist.log_prob(batch["actions"])
    ratio = (newlogp - batch["logprobs"]).exp()
    adv = batch["advantages"]
    m = (adv * w).sum() / nw
    sd = (((adv - m) ** 2 * w).sum() / nw).sqrt()
    adv = (adv - m) / (sd + 1e-8)
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
    pg = (torch.max(unclipped, clipped) * w).sum() / nw
    v_clipped = batch["values"] + torch.clamp(v - batch["values"], -cfg.vf_clip, cfg.vf_clip)
    v_loss = 0.5 * torch.max((v - batch["returns"]) ** 2, (v_clipped - batch["returns"]) ** 2).mean()
    ent = (dist.entropy() * w).sum() / nw
    return pg - cfg.ent_coef * ent + cfg.vf_coef * v_loss, pg, v_loss, ent


def _rollout_batch(seed=0, n=192, obs=24, n_act=10):
    torch.manual_seed(seed)
    net = PolicyValueNet(obs, n_act, (32, 32))
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.rand(n, obs, generator=g)
    mask = torch.rand(n, n_act, generator=g) < 0.5
    mask[:, 0] = True
    mask[:7] = False
    mask[:7, 3] = True                                           # seven steps with a single legal action
    with torch.no_grad():
        logits, v = net(x)
        d = MaskedCategorical(logits, mask)
        a = d.sample(g)
        lp = d.log_prob(a)
    adv = torch.randn(n, generator=g)
    ret = v + adv + 0.3 * torch.randn(n, generator=g)
    batch = {"obs": x, "mask": mask, "actions": a, "logprobs": lp, "values": v, "advantages": adv, "returns": ret,
             "choice": (mask.sum(-1) > 1).float()}
    # perturb the network so that ratios differ from 1 (some outside the clip range)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return net, batch


def test_ppo_update_loss_and_gradient_equal_the_reference_on_one_minibatch():
    cfg = PPOConfig(epochs=1, minibatch_size=192, clip_coef=0.2, vf_clip=0.3, ent_coef=0.02, vf_coef=0.6,
                    max_grad_norm=1e12)                          # no clipping: the gradient is left intact
    net, batch = _rollout_batch()
    import copy
    ref_net = copy.deepcopy(net)
    loss, pg, vl, ent = reference_loss(ref_net, batch, cfg)
    ref_net.zero_grad()
    loss.backward()
    ref_grads = [p.grad.clone() for p in ref_net.parameters()]
    pg, vl, ent = pg.detach(), vl.detach(), ent.detach()
    opt = torch.optim.SGD(net.parameters(), lr=0.0)             # a step that changes nothing
    stats = ppo_update(net, opt, batch, cfg, torch.Generator().manual_seed(0))
    assert stats["policy_loss"] == pytest.approx(float(pg), abs=1e-6)
    assert stats["value_loss"] == pytest.approx(float(vl), abs=1e-6)
    assert stats["entropy"] == pytest.approx(float(ent), abs=1e-6)
    for p, g_ref in zip(net.parameters(), ref_grads):
        assert torch.allclose(p.grad, g_ref, atol=1e-6, rtol=1e-5)
    assert stats["grad_norm"] == pytest.approx(float(torch.sqrt(sum((g ** 2).sum() for g in ref_grads))), rel=1e-5)
    # the clipping and the KL estimator are what they claim: a fraction in [0, 1], k3 >= 0 in expectation sense
    assert 0 < stats["clipfrac"] < 1 and np.isfinite(stats["approx_kl"])


# ------------------------------------------------------------------ masking
def test_illegal_actions_have_zero_probability_and_zero_gradient_and_do_not_shape_the_legal_policy():
    torch.manual_seed(3)
    net = PolicyValueNet(16, 8, (32,))
    x = torch.rand(40, 16)
    mask = torch.rand(40, 8) < 0.5
    mask[:, 1] = True
    mask[:, 6] = False                                           # action 6 is illegal everywhere
    logits, _ = net(x)
    logits.retain_grad()
    d = MaskedCategorical(logits, mask)
    assert torch.all(d.probs[~mask] == 0.0)                      # exactly zero, not merely small
    a = d.sample(torch.Generator().manual_seed(0))
    (d.log_prob(a).sum() + d.entropy().sum()).backward()
    assert torch.all(logits.grad[~mask] == 0.0)                  # exactly zero gradient on illegal logits
    # the legal distribution does not depend on what the network outputs for illegal actions
    with torch.no_grad():
        p_before = MaskedCategorical(net(x)[0], mask).probs[mask]
        net.policy_head.bias[6] += 50.0
        p_after = MaskedCategorical(net(x)[0], mask).probs[mask]
    assert torch.allclose(p_before, p_after)


def test_training_and_acting_log_probabilities_agree():
    """The log-probability recorded at collection (MaskedCategorical, float32, -1e9 masking) equals the acting
    agent's distribution (NetAgent.probabilities, float64, -inf masking) up to float32 rounding."""
    cfg = GameConfig.m4_lasthit()
    env = LaneEnv(cfg, max_turns=20)
    torch.manual_seed(4)
    net = PolicyValueNet(env.obs_size, env.n_actions)
    agent = NetAgent(net, "n")
    oa, ob = env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(20):
        for o in (oa, ob):
            with torch.no_grad():
                d = MaskedCategorical(net(torch.as_tensor(o.vector).unsqueeze(0))[0], torch.as_tensor(o.mask).unsqueeze(0))
            p_train = d.probs[0].numpy()
            p_act = agent.probabilities(o)
            assert np.allclose(p_train, p_act, atol=1e-6) and p_act[~o.mask].sum() == 0.0
            assert np.allclose(agent.distribution(o), p_act[list(o.legal)] / p_act[list(o.legal)].sum())
        r = env.step(agent.act(oa, rng), agent.act(ob, rng))
        oa, ob = r.obs


def test_single_choice_steps_carry_no_policy_gradient_and_are_excluded_from_policy_statistics():
    cfg = PPOConfig(epochs=1, minibatch_size=192, max_grad_norm=1e12, vf_coef=0.0, ent_coef=0.0)
    net, batch = _rollout_batch(seed=5)
    import copy
    # the same batch with the single-choice steps' advantages replaced: nothing about the update may change
    alt = dict(batch)
    alt["advantages"] = batch["advantages"].clone()
    alt["advantages"][:7] = 1e3
    nets = [copy.deepcopy(net), copy.deepcopy(net)]
    stats = []
    for n_, b_ in zip(nets, (batch, alt)):
        opt = torch.optim.SGD(n_.parameters(), lr=0.0)
        stats.append(ppo_update(n_, opt, b_, cfg, torch.Generator().manual_seed(0)))
    for p, q in zip(nets[0].parameters(), nets[1].parameters()):
        assert torch.equal(p.grad, q.grad)
    for key in ("policy_loss", "entropy", "approx_kl", "clipfrac", "grad_norm"):
        assert stats[0][key] == stats[1][key]


# ------------------------------------------------------------------ checkpoint integrity
def test_checkpoint_digest_identifies_the_weights_and_an_altered_file_is_refused(tmp_path):
    cfg = GameConfig.m4_lasthit()
    torch.manual_seed(6)
    net = PolicyValueNet(88, 46)
    path = str(tmp_path / "c.pt")
    digest = save_checkpoint(path, net, cfg, {"name": "x"})
    assert digest == weights_digest(net.state_dict()) and len(digest) == 64
    loaded, cfg2, meta = load_checkpoint(path)
    assert cfg2 == cfg and meta == {"name": "x"} and weights_digest(loaded.state_dict()) == digest
    from lanegame.rl import checkpoint_info
    info = checkpoint_info(path)
    assert info["version"] == 2 and info["digest"] == digest and info["provenance"]["torch"] == torch.__version__
    # a different network has a different digest; a tampered file is refused
    with torch.no_grad():
        net.value_head.bias.add_(1e-3)
    assert weights_digest(net.state_dict()) != digest
    d = torch.load(path, weights_only=False)
    d["state_dict"]["value_head.bias"].add_(1e-3)
    torch.save(d, path)
    with pytest.raises(ValueError, match="digest"):
        load_checkpoint(path)
    assert load_checkpoint(path, verify=False)[0] is not None
    # version-1 files (no digest) still load
    del d["digest"]
    d["version"] = 1
    torch.save(d, path)
    assert load_checkpoint(path)[1] == cfg
