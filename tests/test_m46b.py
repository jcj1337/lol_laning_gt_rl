"""M4.6b: the measurement fixes (lane-only positions, graph distance, the away channel of the ledger, behaviour-only
battery candidates) and the investment arms (opt-in, validated, strength-matched; the baseline untouched)."""

import numpy as np
import pytest

from lanegame.config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, GameConfig, champ_behavior
from lanegame.positional import WARD


def test_lane_only_positions_and_ward_share():
    from lanegame.experiments.rl.m44.metrics import lane_mean, ward_share
    assert lane_mean([-1, 0, WARD, 1]) == pytest.approx(0.0)
    assert ward_share([-1, 0, WARD, 1]) == pytest.approx(0.25)
    assert np.isnan(lane_mean([WARD, WARD]))


def test_record_distance_uses_the_movement_graph():
    from lanegame.experiments.rl.m45c.common import env_cfg
    from lanegame.experiments.m45b import states as S
    from lanegame.champions_m4 import context
    from lanegame.experiments.rl.m44.records import decision_state
    from lanegame.champions import champion_legal_actions
    cfg = env_cfg()
    lane, m4, k = S.zoning_state(cfg, "neutral", level=6, gap="equal")
    ctx = context(cfg)
    d = decision_state(cfg, ctx, lane, m4, k, 0, champion_legal_actions(lane, m4, 0, cfg))
    assert d["dist"] == 0 and d["epos"] == 0                         # both at the lane cell: unchanged
    m4w = m4._replace(b=m4.b._replace(pos=WARD))
    d = decision_state(cfg, ctx, lane, m4w, k, 0, champion_legal_actions(lane, m4w, 0, cfg))
    assert d["epos"] == WARD and d["dist"] == 1                      # W is one edge from the lane cell 0


def test_ledger_channels_are_an_accounting_identity():
    from lanegame.experiments.m45b import engine as E
    from lanegame.experiments.m45b import states as S
    from lanegame.experiments.m45b.strategies import get
    from lanegame.experiments.rl.m45c.common import env_cfg
    cfg = env_cfg()
    g = E.play(cfg, S.s0(cfg), get("g_bully"), get("g_shove_sr"), 11, 150)["summary"]
    for s in ("A", "B"):
        assert g[f"earned_{s}"] == pytest.approx(g[f"farm_{s}"] + g[f"bounty_{s}"] + g[f"plate_{s}"], abs=1e-9)
        assert g[f"farm_{s}"] == pytest.approx(g[f"farm_avail_{s}"] - g[f"unc_{s}"], abs=1e-9)
        assert -1e-9 <= g[f"unc_away_{s}"] <= g[f"unc_{s}"] + 1e-9
    assert g["away_A"] + g["away_B"] > 0 and g["unc_away_A"] + g["unc_away_B"] > 0


def test_behaviour_only_battery_candidates_force_only_the_behaviour():
    from lanegame.experiments.rl.m45c import horizon as HZ
    from lanegame.experiments.rl.m46a.common import net_of, setup_worker, spec
    from lanegame.experiments.m45b import states as S
    cfg = setup_worker()
    lane, m4, k = S.zoning_state(cfg, "neutral", level=10, gap="large")
    net = net_of(spec(0))
    for seed in range(4):
        own = HZ.CRNNet(net, k, seed, 0, 12)
        for name, beh in (("1t:trade_beh", B_TRADE), ("1t:hold_beh", B_HOLD), ("1t:safe_beh", B_SAFE)):
            pol = HZ.build_candidate(HZ.FAMILY_CANDS["C_trade"][name], own)
            assert champ_behavior(pol.decide(cfg, lane, m4, 0, 0)[0]) == beh
        # after its one turn the candidate is the policy itself (same uniforms)
        pol = HZ.build_candidate(HZ.FAMILY_CANDS["C_trade"]["1t:trade_beh"], own)
        assert pol.decide(cfg, lane, m4, 0, 1)[0] == own.decide(cfg, lane, m4, 0, 1)[0]
    # the scripted references are kept, labelled, alongside the twins
    assert "plan:bullyx10" in HZ.FAMILY_CANDS["H_zone_c"] and "plan:bully_behx10" in HZ.FAMILY_CANDS["H_zone_c"]


def test_arms_are_opt_in_validated_and_strength_matched():
    from lanegame.experiments.rl.m46b.common import arm_cfg
    from lanegame.experiments.rl.m46b.curves import ARMS, multiplier
    assert arm_cfg("baseline") == GameConfig.m45_calibrated()
    assert GameConfig.m45_calibrated().invest_gain == 1.5 and GameConfig.m45_calibrated().invest_scale == 16.0
    for arm in ARMS:
        cfg = arm_cfg(arm)
        assert multiplier(arm, 0.0) == pytest.approx(1.0)
        Ms = [multiplier(arm, S) for S in (0, 5, 20, 50, 100, 200)]
        assert all(b > a for a, b in zip(Ms, Ms[1:]))              # strictly increasing
    # every candidate is weaker than the baseline at small investment and stronger at large investment
    for arm in ("C1", "C2", "C3"):
        assert multiplier(arm, 5) < multiplier("baseline", 5)
        assert multiplier(arm, 150) > multiplier("baseline", 150)
    # concavity of the saturating candidates, linearity of the reference
    for arm in ("C1", "C2"):
        d1 = multiplier(arm, 60) - multiplier(arm, 40)
        d2 = multiplier(arm, 120) - multiplier(arm, 100)
        assert d2 < d1
    assert multiplier("C3", 100) - multiplier("C3", 50) == pytest.approx(multiplier("C3", 50) - multiplier("C3", 0),
                                                                          rel=1e-6)


def test_reset_plan_recalls_once_and_resets_between_rollouts():
    from lanegame.experiments.rl.m45c import horizon as HZ
    from lanegame.experiments.rl.m46a.common import net_of, setup_worker, spec
    from lanegame.experiments.rl.m46a.policies import ResetPlan
    from lanegame.experiments.m45b import states as S
    from lanegame.champions import champion_legal_actions
    from lanegame.config import W_HOLD
    cfg = setup_worker()
    lane, m4, k = S.zoning_state(cfg, "b_push", level=10, gap="equal")
    m4 = m4._replace(a=m4.a._replace(pos=-1))                          # Recall legal at the own -1
    assert champ_act_recall(W_HOLD) in champion_legal_actions(lane, m4, 0, cfg)
    own = HZ.CRNNet(net_of(spec(0)), k, 1, 0, 12)
    plan = ResetPlan(own, 10)
    assert plan.decide(cfg, lane, m4, 0, 0)[0] == champ_act_recall(W_HOLD) and plan.done
    assert plan.decide(cfg, lane, m4, 0, 1)[0] == own.decide(cfg, lane, m4, 0, 1)[0]   # after the Recall: the policy
    assert plan.decide(cfg, lane, m4, 0, 0)[0] == champ_act_recall(W_HOLD)              # a new rollout: resets


def champ_act_recall(w):
    from lanegame.config import champ_act
    return champ_act(w, B_RECALL)
