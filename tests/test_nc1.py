"""NC1 (network capacity): the function-preserving network changes used by the PPO arms."""

import pytest

torch = pytest.importorskip("torch")

from lanegame.rl.network import PolicyValueNet, build_network, widen_hidden  # noqa: E402


def _outputs(net, x):
    with torch.no_grad():
        return net(x)


def test_widen_hidden_keeps_outputs_and_global_rng():
    torch.manual_seed(1)
    net = PolicyValueNet(90, 61)
    x = torch.randn(512, 90) * 2
    state = torch.get_rng_state().clone()
    wide = widen_hidden(net, (256, 256), seed=7)
    assert torch.equal(state, torch.get_rng_state())
    (l0, v0), (l1, v1) = _outputs(net, x), _outputs(wide, x)
    assert torch.allclose(l0, l1, atol=1e-6) and torch.allclose(v0, v1, atol=1e-6)
    assert wide.spec()["hidden"] == [256, 256]
    assert sum(p.numel() for p in wide.parameters()) == 105_022


def test_widen_hidden_is_seeded_and_new_units_learn():
    torch.manual_seed(2)
    net = PolicyValueNet(90, 61)
    a, b = widen_hidden(net, (256, 256), seed=3), widen_hidden(net, (256, 256), seed=3)
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    logits, v = a(torch.randn(64, 90))
    (logits.logsumexp(-1).mean() + v.mean()).backward()
    assert float(a.policy_head.weight.grad[:, 128:].abs().sum()) > 0
    assert float(a.trunk[2].weight.grad[:, 128:].abs().sum()) > 0


def test_widen_hidden_rejects_narrowing_and_depth_change():
    net = PolicyValueNet(90, 61)
    with pytest.raises(ValueError):
        widen_hidden(net, (64, 128))
    with pytest.raises(ValueError):
        widen_hidden(net, (128, 128, 128))


def test_widened_spec_rebuilds():
    wide = widen_hidden(PolicyValueNet(90, 61), (256, 256))
    rebuilt = build_network(wide.spec())
    rebuilt.load_state_dict(wide.state_dict())
    x = torch.randn(8, 90)
    assert torch.equal(_outputs(wide, x)[0], _outputs(rebuilt, x)[0])


def _feature_index():
    from lanegame.config import GameConfig
    from lanegame.experiments.rl.m46e.features import obs_index
    idx = obs_index(GameConfig.m45_calibrated())
    return [idx[k] for k in ("own_champ_attack_power", "enemy_champ_attack_power", "own_champ_hp_frac",
                             "enemy_champ_hp_frac", "own_champ_absent", "enemy_champ_absent")], idx


def test_strength_features_match_m46e_explicit():
    import numpy as np
    from lanegame.experiments.rl.m46e.features import explicit
    from lanegame.rl.network import strength_features
    index, idx = _feature_index()
    rng = np.random.default_rng(0)
    X = rng.uniform(0.01, 0.99, size=(2000, 90)).astype(np.float32)
    X[:, index[4]] = rng.integers(0, 2, 2000)
    X[:, index[5]] = rng.integers(0, 2, 2000)
    X[:7, index[2]] = 0.0
    assert np.array_equal(strength_features(torch.as_tensor(X), index).numpy(), explicit(X, idx))


def test_with_strength_features_keeps_outputs_and_rebuilds():
    from lanegame.rl.network import StrengthFeatureNet, with_strength_features
    index, _ = _feature_index()
    torch.manual_seed(4)
    net = PolicyValueNet(90, 61)
    x = torch.rand(256, 90)
    state = torch.get_rng_state().clone()
    feat = with_strength_features(net, index)
    assert torch.equal(state, torch.get_rng_state())
    (l0, v0), (l1, v1) = _outputs(net, x), _outputs(feat, x)
    assert torch.allclose(l0, l1, atol=1e-6) and torch.allclose(v0, v1, atol=1e-6)
    spec = feat.spec()
    assert spec["obs_size"] == 90 and spec["features"] == "strength6" and feat.trunk[0].in_features == 96
    rebuilt = build_network(spec)
    assert isinstance(rebuilt, StrengthFeatureNet)
    rebuilt.load_state_dict(feat.state_dict())
    assert torch.equal(_outputs(rebuilt, x)[0], _outputs(feat, x)[0])


def test_m46e_make_mlp_default_is_the_critic_shape():
    import numpy as np
    from torch import nn
    from lanegame.experiments.rl.m46e.fit import make_mlp
    from lanegame.rl.network import _layer
    torch.manual_seed(11)
    old = nn.Sequential(_layer(90, 128, np.sqrt(2.0)), nn.Tanh(), _layer(128, 128, np.sqrt(2.0)), nn.Tanh(),
                        _layer(128, 1, 1.0))
    torch.manual_seed(11)
    new = make_mlp(90)
    assert list(old.state_dict()) == list(new.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(old.state_dict().values(), new.state_dict().values()))
    assert make_mlp(90, (64, 64, 64))[4].out_features == 64
