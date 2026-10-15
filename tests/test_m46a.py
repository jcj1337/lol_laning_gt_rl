"""M4.6a harness invariants: the default trainer is unchanged, the seat-aware start archive mirrors for seat B, plans
force only their components and share the baseline's randomness, state variants change only what they name, and a
duplicated baseline gives exactly zero paired differences."""

import numpy as np
import pytest
import torch

from lanegame.config import B_HOLD, B_TRADE, GameConfig, champ_behavior
from lanegame.experiments.rl.m46a import engine as EN
from lanegame.experiments.rl.m46a import policies as P
from lanegame.experiments.rl.m46a.common import env_cfg, net_of, setup_worker, spec
from lanegame.experiments.rl.m46a.states import freeze_gap_state, invest_pair_mult, zoning_gap_state


def test_environment_is_the_calibrated_preset():
    assert env_cfg() == GameConfig.m45_calibrated()


def test_gap_multiplier():
    from lanegame.experiments.m45b.states import multiplier
    cfg = env_cfg()
    for L in (6, 10, 14):
        for m in (1.24, 1.5, 1.76, 2.2):
            a, b = invest_pair_mult(cfg, L, m)
            assert abs(multiplier(cfg, a) / multiplier(cfg, b) - m) < 0.01


def _tiny_train(collector_cls=None, archive=None, mix=0.0):
    from lanegame.experiments.rl.m45c.common import agent_from_spec, ppo_config
    from lanegame.rl import OpponentSampler, train_ppo
    cfg = env_cfg()
    kw = dict(lane_cache=200, n_envs=2, segment_length=16)
    if archive:
        kw.update(start_archive=archive, start_mix=mix, game_length=20)
    ppo = ppo_config(7, 2 * 16 * 2, **kw)
    return train_ppo(cfg, ppo, OpponentSampler([agent_from_spec(spec(0), cfg)]), verbose=False,
                     collector_cls=collector_cls)


def test_default_trainer_unchanged_by_the_collector_option():
    from lanegame.rl.train import Collector
    a = _tiny_train(None).net.state_dict()
    b = _tiny_train(Collector).net.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_focal_collector_mirrors_for_seat_b(tmp_path):
    from lanegame.env.starts import StartState, save_archive
    from lanegame.experiments.rl.m45c.common import agent_from_spec, ppo_config
    from lanegame.experiments.rl.m46a.e4 import focal_collector_cls
    from lanegame.rl import OpponentSampler
    cfg = env_cfg()
    st = zoning_gap_state(cfg, "a_push", 10, "x1.76")
    path = str(tmp_path / "a.json")
    save_archive(path, [StartState(st[0], st[1], st[2] % cfg.phase_period)], cfg)
    ppo = ppo_config(3, 4096, lane_cache=100, n_envs=8, segment_length=8, start_archive=path, start_mix=1.0,
                     game_length=20)
    seats = set()
    for seed in range(11, 15):
        col = focal_collector_cls()(cfg, ppo, OpponentSampler([agent_from_spec(spec(0), cfg)]), seed)
        for i, env in enumerate(col.envs):
            seats.add(col.seat[i])
            if col.seat[i] == 0:
                assert env.state == st[0] and env.champ == st[1]
            else:
                assert env.state == st[0].mirror() and env.champ == st[1].mirror()
    assert seats == {0, 1}


def test_behavior_plan_forces_its_component_and_shares_randomness():
    from lanegame.experiments.rl.m45c import horizon as HZ
    cfg = setup_worker()
    st = zoning_gap_state(cfg, "neutral", 10, "x1.76")
    net = net_of(spec(0))
    for seed in range(6):
        own = HZ.CRNNet(net, st[2], seed, 0, 5)
        thr = P.BehaviorPlan("threaten", HZ.CRNNet(net, st[2], seed, 0, 5), P.rule_threaten, 3)
        farm = P.BehaviorPlan("farm", HZ.CRNNet(net, st[2], seed, 0, 5), P.rule_farm, 3)
        assert champ_behavior(thr.decide(cfg, st[0], st[1], 0, 0)[0]) == B_TRADE
        assert champ_behavior(farm.decide(cfg, st[0], st[1], 0, 0)[0]) == B_HOLD
        # after the plan's turns the plan is exactly the policy (same uniforms)
        assert thr.decide(cfg, st[0], st[1], 0, 4)[0] == own.decide(cfg, st[0], st[1], 0, 4)[0]


def test_release_hands_low_hp_turns_to_the_policy():
    from lanegame.experiments.rl.m45c import horizon as HZ
    cfg = setup_worker()
    st = P.variant_hp(cfg, zoning_gap_state(cfg, "neutral", 10, "x1.76"), 0.7)
    net = net_of(spec(0))
    own = HZ.CRNNet(net, st[2], 1, 0, 5)
    thr = P.BehaviorPlan("threaten", HZ.CRNNet(net, st[2], 1, 0, 5), P.rule_threaten, 3)
    assert thr.decide(cfg, st[0], st[1], 0, 0)[0] == own.decide(cfg, st[0], st[1], 0, 0)[0]
    assert thr.forced == 0


def test_variants_change_only_what_they_name():
    cfg = env_cfg()
    st = freeze_gap_state(cfg, 10, "x1.76", 12)
    st = (st[0], st[1]._replace(a=st[1].a._replace(gold=5 * cfg.gold_scale)), st[2])
    iv = P.variant_invest(cfg, st)
    assert iv[0] == st[0] and iv[1].b == st[1].b and iv[1].a.gold == 0
    assert iv[1].a.items == st[1].a.items + 5 * cfg.gold_scale
    hv = P.variant_hp(cfg, st, 0.3)
    assert hv[1].a._replace(hp=st[1].a.hp) == st[1].a and hv[1].a.hp < st[1].a.hp


def test_duplicate_baseline_has_zero_paired_difference():
    cfg = setup_worker()
    st = zoning_gap_state(cfg, "b_push", 6, "x1.5")
    job = {"spec": spec(1), "family": "t", "idx": 0, "start": st, "proto": "policy", "horizon": 10,
           "roll": EN.roll_seeds(3, 2), "cands": {"policy": ("policy",), "again": ("policy",),
                                                 "net": ("net", spec(1))}, "critic": True, "path": None}
    _, rec = EN.paired_job(job)
    for c in ("again", "net"):
        assert EN.per_state_diff(rec, c, "G", 10) == 0.0
        assert EN.per_state_diff(rec, c, "Q", 10) == 0.0
