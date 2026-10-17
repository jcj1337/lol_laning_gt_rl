"""M4.7: the horizon study's objective override (configuration, trainer, GAE), the reward-stream engine and its
scoring arithmetic. Checkpoint- and record-dependent tests skip when the git-ignored files are absent."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lanegame import GameConfig                                                       # noqa: E402
from lanegame.experiments.rl.m47 import common as C                                   # noqa: E402
from lanegame.experiments.rl.m47 import streams as SM                                 # noqa: E402
from lanegame.rl import PolicyValueNet, SharedConfig, train_shared                    # noqa: E402
from lanegame.rl.ppo import PPOConfig, check_time_semantics, compute_gae              # noqa: E402

RECORDS = os.path.join("results", "rl", "checkpoints", "m46d", "records", "baseline", "ext_baseline_0.npz")


def _net(seed=0):
    torch.set_num_threads(1)              # orthogonal initialization depends on the thread count
    torch.manual_seed(seed)
    return PolicyValueNet(90, 61)


# ---------------------------------------------------------------------------------------------- the objective
def test_short_arm_differs_from_the_baseline_only_in_gamma():
    dc, ds = C.arm_cfg("control").to_dict(), C.arm_cfg("short").to_dict()
    assert dc == GameConfig.m45_calibrated().to_dict()
    assert sorted(k for k in dc if dc[k] != ds[k]) == ["gamma"]
    assert ds["gamma"] == pytest.approx(1 - 1 / C.SHORT_H, abs=1e-15)
    assert GameConfig.m45_calibrated().gamma == 0.99            # the default objective is unchanged


def test_trainer_takes_the_games_gamma_and_refuses_a_mismatch():
    cfg = C.arm_cfg("short")
    p = PPOConfig.continuing(cfg, total_steps=1000)
    assert p.gamma == cfg.gamma and not p.allow_gamma_mismatch
    check_time_semantics(cfg, p)
    with pytest.raises(ValueError):
        check_time_semantics(C.arm_cfg("control"), p)


def test_gae_receives_the_short_gamma_and_bootstraps_truncations():
    from lanegame.experiments.rl.m47.train import GaeSpy
    cfg = C.arm_cfg("short")
    ppo = PPOConfig.continuing(cfg, total_steps=2 * 2 * 16 * 2, n_envs=2, segment_length=16, game_length=8,
                               minibatch_size=32, seed=5, lane_cache=500)
    spy = GaeSpy(n=5).install()
    try:
        train_shared(cfg, ppo, SharedConfig(current_fraction=1.0, snapshot_every=10_000, pool_size=2),
                     net=_net(), verbose=False)
    finally:
        spy.uninstall()
    assert spy.calls and all(c["gamma"] == cfg.gamma and c["lambda"] == 0.95 for c in spy.calls)
    assert all(c["terminated"] == 0 for c in spy.calls)
    assert sum(c["truncated"] for c in spy.calls) > 0
    assert all(c["truncated_bootstrap_nonzero"] == c["truncated"] for c in spy.calls)
    assert all(c["adv_rms_diff_vs_gamma_0.99"] > 0 for c in spy.calls)
    from lanegame.rl import shared as SH
    assert SH.compute_gae is compute_gae                        # the spy is removed


def test_gae_truncation_uses_the_next_value_and_termination_would_not():
    r = np.array([[1.0], [0.0]])
    v = np.array([[0.0], [0.0]])
    nv = np.array([[0.0], [10.0]])
    adv, _ = compute_gae(r, v, nv, np.zeros_like(r, bool), np.array([[False], [True]]), 0.9, 0.95)
    assert adv[1, 0] == pytest.approx(9.0)                      # bootstrapped: gamma * V(next)
    adv_t, _ = compute_gae(r, v, nv, np.array([[False], [True]]), np.array([[False], [True]]), 0.9, 0.95)
    assert adv_t[1, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------------- discount arithmetic
def test_weight_shares_match_brute_force():
    g, T = 0.98, 94
    w = g ** np.arange(100_000)
    assert C.weight_before(T, g) == pytest.approx(w[:T].sum() / w.sum(), rel=1e-9)
    brute = np.mean([1 - g ** (T - k) for k in range(T)])
    assert C.mean_weight_before_end(T, g) == pytest.approx(brute)
    assert C.HORIZONS["100"] == 0.99 and C.HORIZONS["60"] == pytest.approx(1 - 1 / 60)


def test_split_and_level_cut():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(5, 40))
    cut = np.array([0, 10, 39, 40, 25])
    pre, post = SM.split_at(r, cut, 0.97)
    assert np.allclose(pre + post, SM.discounted(r, 0.97))
    assert np.allclose(pre[0], 0.0) and np.allclose(post[3], 0.0)
    L = np.zeros((2, 6, 2), np.int8)
    L[0, :, 0] = [12, 13, 14, 14, 15, 15]
    L[0, :, 1] = [12, 12, 13, 14, 14, 15]
    L[1] = 13
    assert SM.level_cut(L, 14).tolist() == [3, 6]


def test_return_split_is_exact():
    from lanegame.experiments.rl.m47.returns import split_returns
    rng = np.random.default_rng(1)
    r = rng.normal(size=(3, 200))
    tot, pre, post = split_returns(r, 30, 0.99, end=94)
    assert np.allclose(pre + post, tot)
    assert np.allclose(tot, (r[:, 30:] * 0.99 ** np.arange(170)).sum(1))


# ---------------------------------------------------------------------------------------------- the stream engine
def test_stream_game_reproduces_the_m46a_engine():
    from lanegame.experiments.m45b import states as S
    from lanegame.experiments.rl.m45c import horizon as HZ
    from lanegame.experiments.rl.m46a import engine as EN
    cfg = C.setup("baseline", cache=500)
    net = _net(2)
    start = S.s0(cfg)
    T, sd = 40, 1234
    r, L, first = SM.stream_game(cfg, start, HZ.CRNNet(net, 0, sd, 0, T), HZ.CRNNet(net, 0, sd, 1, T), sd, T)
    out = EN.rollout(cfg, start, HZ.CRNNet(net, 0, sd, 0, T), HZ.CRNNet(net, 0, sd, 1, T), sd, T, (T,), ())
    assert SM.discounted(r.astype(float), 0.99) == pytest.approx(out[T]["G"], abs=1e-4)
    assert L[0].tolist() == [1, 1] and first[0] >= 0 and first[1] >= 0
    r2, L2, _ = SM.stream_game(cfg, start, HZ.CRNNet(net, 0, sd, 0, T), HZ.CRNNet(net, 0, sd, 1, T), sd, T)
    assert np.array_equal(r, r2) and np.array_equal(L, L2)          # deterministic


def test_variants_change_only_seat_a():
    from lanegame.experiments.rl.m46c.archive import lead_cell
    cfg = C.env_cfg("baseline")
    st = lead_cell(cfg, "early", "neutral", "equal")
    v = SM.VARIANTS["inv+4"](cfg, st)
    assert v[1].a.items == st[1].a.items + 4 * cfg.gold_scale and v[1].b == st[1].b and v[0] == st[0]
    h = SM.VARIANTS["hp-0.3"](cfg, st)
    assert h[1].a.hp < st[1].a.hp and h[1].b == st[1].b


def test_policy_paths():
    assert C.policy_path("p/short:1").replace("\\", "/").endswith("m47/train/main/short/s1/final.pt")
    assert C.policy_path("p/control:0@2048000").replace("\\", "/").endswith(
        "m47/train/main/control/s0/snapshots/snap_002048000.pt")
    assert C.policy_path("ext/baseline:0").replace("\\", "/").endswith("m46c/train/ext/baseline/s0/final.pt")
    with pytest.raises(ValueError):
        C.policy_path("p/other:0")


# ---------------------------------------------------------------------------------------------- record-based readouts
@pytest.mark.skipif(not os.path.exists(RECORDS), reason="M4.6d records are git-ignored")
def test_timeline_and_behaviour_on_a_record():
    from lanegame.experiments.rl.m47 import behaviour as B
    from lanegame.experiments.rl.m47 import timeline as TL
    d = TL.load_records("baseline", 0)
    s = TL.policy_summary(d)
    t10, t14 = s["timeline"]["10"]["pooled"]["mean"], s["timeline"]["14"]["pooled"]["mean"]
    assert 30 < t10 < t14 < 150
    pg = B.per_game(d, "lane")
    assert set(pg) == set(B.METRICS)
    assert all(len(v) == d["L"].shape[0] for v in pg.values())
    sh = pg["shove"] + pg["last_hit"] + pg["thin"] + pg["slow_push"]
    assert np.allclose(sh[np.isfinite(sh)], 1.0)
    p = B.paired(d, d)
    assert all(abs(x["mean"]) < 1e-12 for w in p.values() for x in w.values() if np.isfinite(x["mean"]))
