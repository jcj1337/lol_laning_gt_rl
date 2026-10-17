"""M4.6f: the separate-critic network (initialization, gradient separation, checkpoints, training) and the pilot's
instrumentation (gradient probe, cells, classification)."""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lanegame import GameConfig                                                       # noqa: E402
from lanegame.env import LaneEnv                                                      # noqa: E402
from lanegame.rl import (MaskedCategorical, PolicyValueNet, PPOConfig, SeparateCriticNet, SharedConfig,  # noqa: E402
                         build_network, load_checkpoint, save_checkpoint, train_shared, weights_digest)

CFG = GameConfig.champions(horizon=10)
OBS, NACT = LaneEnv(CFG).obs_size, LaneEnv(CFG).n_actions
START = os.path.join("results", "rl", "checkpoints", "m46c", "train", "ext", "C1", "s0", "final.pt")


def _ppo(**kw):
    base = dict(total_steps=320, n_envs=4, n_steps=10, minibatch_size=64, seed=3)
    base.update(kw)
    return PPOConfig(**base)


def _shared(seed=0):
    torch.set_num_threads(1)              # as train_shared: orthogonal initialization depends on the thread count
    torch.manual_seed(seed)
    return PolicyValueNet(OBS, NACT)


def _obs_batch(n=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, OBS, generator=g)


def test_from_shared_reproduces_probabilities_and_values_exactly():
    net = _shared()
    sep = SeparateCriticNet.from_shared(net)
    x = _obs_batch()
    mask = torch.ones(len(x), NACT, dtype=torch.bool)
    mask[:, ::3] = False
    l0, v0 = net(x)
    l1, v1 = sep(x)
    assert torch.equal(MaskedCategorical(l0, mask).probs, MaskedCategorical(l1, mask).probs)
    assert torch.equal(v0, v1) and torch.equal(net.value(x), sep.value(x))
    assert torch.equal(sep.value_features(x), net.value_features(x))
    # the actor's parameters keep PolicyValueNet's names and values
    sd0, sd1 = net.state_dict(), sep.state_dict()
    for k, v in sd0.items():
        assert torch.equal(sd1[k], v)
    assert set(sd1) - set(sd0) == {f"critic_trunk.{k}" for k in net.trunk.state_dict()}


def test_from_shared_leaves_the_global_rng_untouched():
    net = _shared()
    torch.manual_seed(123)
    a = torch.rand(3)
    torch.manual_seed(123)
    SeparateCriticNet.from_shared(net)
    assert torch.equal(torch.rand(3), a)


def test_actor_and_critic_share_no_storage_and_are_independent():
    net = _shared()
    sep = SeparateCriticNet.from_shared(net)
    assert not ({p.data_ptr() for p in sep.actor_parameters()} & {p.data_ptr() for p in sep.critic_parameters()})
    x = _obs_batch()
    l0, v0 = sep(x)
    with torch.no_grad():
        for p in sep.critic_trunk.parameters():
            p.add_(0.1)
    l1, v1 = sep(x)
    assert torch.equal(l0, l1) and not torch.equal(v0, v1)
    with torch.no_grad():                                 # and the source network is untouched
        assert torch.equal(net(x)[1], v0)


def _grad_norms(net, loss_kind):
    x = _obs_batch()
    mask = torch.ones(len(x), NACT, dtype=torch.bool)
    lg, v = net(x)
    dist = MaskedCategorical(lg, mask)
    if loss_kind == "policy":
        loss = -(torch.linspace(-1, 1, len(x)) * dist.log_prob(dist.mode())).mean() - 0.03 * dist.entropy().mean()
    else:
        loss = 0.5 * (v - torch.linspace(-1, 1, len(x))).pow(2).mean()
    net.zero_grad()
    loss.backward()
    return {name: float(sum(0.0 if p.grad is None else p.grad.norm() ** 2 for p in mod.parameters()) ** 0.5)
            for name, mod in net.named_children()}


def test_gradient_routing_separates_actor_and_critic():
    sep = SeparateCriticNet.from_shared(_shared())
    g = _grad_norms(sep, "policy")
    assert g["trunk"] > 0 and g["policy_head"] > 0 and g["critic_trunk"] == 0 and g["value_head"] == 0
    g = _grad_norms(sep, "value")
    assert g["trunk"] == 0 and g["policy_head"] == 0 and g["critic_trunk"] > 0 and g["value_head"] > 0
    g = _grad_norms(_shared(), "value")                   # the control: the value loss reaches the shared trunk
    assert g["trunk"] > 0 and g["policy_head"] == 0


def test_checkpoints_round_trip_both_classes(tmp_path):
    sep = SeparateCriticNet.from_shared(_shared())
    p = str(tmp_path / "sep.pt")
    d = save_checkpoint(p, sep, CFG, {"kind": "test"})
    back, _, _ = load_checkpoint(p)
    assert isinstance(back, SeparateCriticNet) and weights_digest(back.state_dict()) == d
    assert back.spec() == sep.spec() and back.spec()["critic"] == "separate"
    net = _shared()
    p2 = str(tmp_path / "shared.pt")
    save_checkpoint(p2, net, CFG)
    back2, _, _ = load_checkpoint(p2)
    assert type(back2) is PolicyValueNet and "critic" not in back2.spec()
    assert type(build_network({"obs_size": OBS, "n_actions": NACT, "hidden": [128, 128]})) is PolicyValueNet
    with pytest.raises(ValueError):
        build_network({"obs_size": OBS, "n_actions": NACT, "hidden": [8], "critic": "other"})


@pytest.mark.skipif(not os.path.exists(START), reason="needs the git-ignored M4.6c checkpoint")
def test_existing_checkpoint_still_loads_as_a_shared_network():
    net, _, meta = load_checkpoint(START)
    assert type(net) is PolicyValueNet and net.obs_size == 90 and net.n_actions == 61


def test_train_shared_with_a_separate_critic_is_reproducible_and_recorded(tmp_path):
    runs = []
    for i in range(2):
        net = SeparateCriticNet.from_shared(_shared())
        out = str(tmp_path / f"r{i}")
        res = train_shared(CFG, _ppo(), SharedConfig(current_fraction=1.0, snapshot_every=160), out_dir=out, net=net,
                           verbose=False)
        runs.append(weights_digest(res.net.state_dict()))
        with open(os.path.join(out, "run.json"), encoding="utf-8") as f:
            run = json.load(f)
        assert run["model"]["class"] == "SeparateCriticNet" and run["model"]["critic"] == "separate"
        back, _, _ = load_checkpoint(os.path.join(out, "final.pt"))
        assert isinstance(back, SeparateCriticNet)
        assert weights_digest(back.state_dict()) == runs[-1]
    assert runs[0] == runs[1]


def test_without_a_value_loss_the_separate_critic_never_moves():
    """vf_coef 0: the policy and entropy terms alone move the actor and leave the critic exactly unchanged in the
    separate network; in the shared network the same training changes the values through the shared trunk."""
    sep = SeparateCriticNet.from_shared(_shared())
    c0 = {k: v.clone() for k, v in sep.state_dict().items()}
    res = train_shared(CFG, _ppo(vf_coef=0.0), SharedConfig(current_fraction=1.0), net=sep, verbose=False)
    sd = res.net.state_dict()
    assert all(torch.equal(sd[k], c0[k]) for k in c0 if k.startswith(("critic_trunk.", "value_head.")))
    assert any(not torch.equal(sd[k], c0[k]) for k in c0 if k.startswith(("trunk.", "policy_head.")))
    sh = _shared()
    x = _obs_batch()
    v0 = sh.value(x).detach().clone()
    res = train_shared(CFG, _ppo(vf_coef=0.0), SharedConfig(current_fraction=1.0), net=sh, verbose=False)
    assert not torch.equal(res.net.value(x).detach(), v0)


def test_gradient_probe_is_read_only():
    from lanegame.experiments.rl.m46f.train import GradProbe

    class Probe(GradProbe):
        def __init__(self, ppo, net):
            self.base, self.ppo, self.net, self.every, self.n_mb, self.calls = (lambda r, f: {}), ppo, net, 1, 2, 0

    digests, recs = [], []
    for hook in (False, True):
        for kind in ("shared", "separate"):
            net = _shared() if kind == "shared" else SeparateCriticNet.from_shared(_shared())
            ppo = _ppo()
            res = train_shared(CFG, ppo, SharedConfig(current_fraction=1.0), net=net, verbose=False,
                               rollout_stats=Probe(ppo, net) if hook else None)
            digests.append((kind, weights_digest(res.net.state_dict())))
            if hook:
                recs.append((kind, res.history[0]))
    assert digests[:2] == digests[2:]
    for kind, e in recs:
        assert e["gp_total_norm"] > 0 and e["gp_actor_norm"] > 0 and 0 < e["gp_clip_shrink"] <= 1.0
        assert (e["gp_trunk_val"] == 0.0) == (kind == "separate")
        assert (e["gp_ctrunk_val"] > 0.0) == (kind == "separate")


def test_classification_rules():
    from lanegame.experiments.rl.m46f.analysis import classify
    assert classify({"mean": 0.30, "ci": 0.10}, 0.2, 0.3) == "benefit"
    assert classify({"mean": -0.30, "ci": 0.10}, 0.2, 0.3) == "regression"
    assert classify({"mean": 0.10, "ci": 0.05}, 0.2, 0.3) == "small"
    assert classify({"mean": 0.05, "ci": 0.15}, 0.2, 0.3) == "no material difference"
    assert classify({"mean": 0.10, "ci": 0.30}, 0.2, 0.3) == "inconclusive"
    assert classify({"mean": float("nan"), "ci": 0.1}, 0.2, 0.3) == "n/a"


def test_policy_specifications_and_cells():
    from lanegame.experiments.rl.m46f import behaviour as B
    from lanegame.experiments.rl.m46f import common as C
    assert C.policy_path("separate:2").endswith(os.path.join("train", "main", "separate", "s2", "final.pt"))
    assert C.policy_path("ext/shared:1@2048000").endswith(os.path.join("ext", "shared", "s1", "snapshots",
                                                                        "snap_002048000.pt"))
    assert C.policy_path("start:3") == C.start_path(3)
    cfg = C.env_cfg()
    conv = B.conv_cells(cfg)
    conds = [m["cond"] for _, _, m in conv]
    assert len(conv) == 54
    assert sum(c in B.CONV_CONDS for c in conds) == 24 and sum(c in B.DEF_CONDS for c in conds) == 16
    groups = {m["group"] for _, _, m in conv}
    assert all(any(m["group"] == g and m["cond"] == "equal" for _, _, m in conv) for g in groups)
    fz = B.freeze_cells(cfg)
    assert len(fz) == 36 and {m["role"] for _, _, m in fz} == {"freezer", "breaker"}
    lane, m4, k = fz[0][1]
    lane_b, m4_b, _ = fz[1][1]
    assert lane_b == lane.mirror() and m4_b == m4.mirror()
    rw = B.recall_cells(cfg)
    assert len(rw) == 24
    for lab, (lane, m4, k), m in rw:
        weak = m4.a if m["role"] == "weak" else m4.b
        strong = m4.b if m["role"] == "weak" else m4.a
        assert weak.hp < strong.hp


@pytest.mark.skipif(not os.path.exists(START), reason="needs the git-ignored M4.6c checkpoint")
def test_initial_equivalence_on_the_real_start():
    from lanegame.experiments.rl.m46f import verify
    rec = verify.check_seed(0)
    assert all(rec["checks"].values()), [k for k, v in rec["checks"].items() if not v]
