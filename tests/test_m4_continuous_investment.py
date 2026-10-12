"""M4.2c continuous gold investment (GameConfig.m4_continuous(), an experiment; docs/m4/m4_continuous_investment_report.md):
every return to base invests all unspent gold, AP(L, S) = AP(L, 0) [1 + alpha ((1 + S)^p - 1)], no tiers, no clamp.
The validated two-tier m4() is untouched (tests/test_m4_economy.py)."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame import progression as pg
from lanegame.champions import N_CHANCE, U_TRADE_HIT
from lanegame.champions_m4 import M4Side, M4State, check_invariants, context, resolve_turn_m4
from lanegame.config import B_HOLD, B_RECALL, B_TRADE, CHAMP_ABSENT, SP, champ_act
from lanegame.env import LaneEnv, UniformAgent
from lanegame.env.agents_m4 import m4_suite
from lanegame.env.discounted import per_turn_bound
from lanegame.env.observation_m4 import decode_m4, observe_m4
from lanegame.state import State

C = GameConfig.m4_continuous()
INV = pg.investment(C)
T = C.tower
U0 = (1.0,) * N_CHANCE
HO, TR, RECALL = champ_act(SP, B_HOLD), champ_act(SP, B_TRADE), champ_act(SP, B_RECALL)
MU = C.gold_scale
LANE = State(0, 4, (), 0, 0, 4, (), 0, 0, 0, 0, 0)


def side(pos=0, hp=None, xp=0, invested=0, absent=0, gold=0, tau=0):
    t = pg.tables(C)
    return M4Side(pos, t.hp_max[pg.level_of(xp, t)] if hp is None else hp, absent, xp, invested, gold, tau)


def come_back(st: M4State, k: int = 2):
    """A sits out the last turn of an absence (its timer reaches 0): the return turn."""
    return resolve_turn_m4(LANE, st, k, CHAMP_ABSENT, HO, C, u=U0)


def test_preset_validates_and_keeps_the_two_tier_baseline():
    C.validate()
    assert C.continuous_items and C.item_tiers == 0 and C.is_m4
    assert not GameConfig.m4().continuous_items and GameConfig.m4().item_tiers == 2
    with pytest.raises(ValueError, match="item_tiers"):
        GameConfig.m4(item_model="continuous").validate()
    with pytest.raises(ValueError, match="invest_alpha"):
        GameConfig.m4(invest_alpha=0.06).validate()


@pytest.mark.parametrize("gold", [1, MU, 4 * MU, 7 * MU, 7 * MU + 3])
def test_a_return_invests_exactly_the_unspent_gold(gold):
    st = M4State(side(-T, absent=1, gold=gold, invested=5 * MU), side(0))
    t = come_back(st)
    a = t.champ.a
    assert t.returned[0] and t.invested[0] == gold and t.purchases[0] == 1
    assert a.invested == 5 * MU + gold and a.gold == 0
    assert t.gold_clamped == (0, 0)
    # spending is not a reward event: the same turn with no gold to invest pays the same reward
    t0 = come_back(M4State(side(-T, absent=1, gold=0, invested=5 * MU), side(0)))
    assert t.reward_units == t0.reward_units == t.earned_units[0] - t.earned_units[1]


def test_recall_with_one_gold_invests_one_through_the_environment():
    env = LaneEnv(C, max_turns=6)
    obs = env.reset(seed=3, root=LANE, champ=M4State(side(-T, gold=MU), side(0)))
    invested = []
    for k in range(C.recall_turns + 1):
        a = RECALL if k == 0 else (CHAMP_ABSENT if env.champ.a.absent > 0 else HO)
        r = env.step(a, HO)
        invested.append(r.info["invested"][0])
    assert sum(invested) == MU and env.champ.a.invested == MU and env.champ.a.gold == 0
    assert invested[C.recall_turns - 1] == MU                          # at the end of the last away turn


def test_investments_accumulate_and_equal_totals_give_equal_power():
    first = come_back(M4State(side(-T, absent=1, gold=3 * MU), side(0))).champ
    assert first.a.invested == 3 * MU
    second = come_back(first.replace_side(0, absent=1, gold=4 * MU)).champ
    once = come_back(M4State(side(-T, absent=1, gold=7 * MU), side(0))).champ
    assert second.a.invested == once.a.invested == 7 * MU and second.a.gold == once.a.gold == 0
    for L in (1, 6, 11, 18):
        assert INV.attack_power(L, 3 * MU + 4 * MU) == INV.attack_power(L, 7 * MU)


def test_investment_persists_through_death_and_is_invested_on_the_death_return():
    st = M4State(side(0, hp=50, invested=9 * MU, gold=2 * MU), side(0))
    u = list(U0)
    u[U_TRADE_HIT[1]] = 0.0
    t = resolve_turn_m4(LANE, st, 2, HO, TR, C, u=tuple(u))
    assert t.died[0] and t.champ.a.invested == 9 * MU and t.champ.a.absent == C.death_turns
    lane, m4 = t.lane.state, t.champ
    gold_at_death = m4.a.gold
    for k in range(3, 3 + C.death_turns):
        t = resolve_turn_m4(lane, m4, k, CHAMP_ABSENT, HO, C, u=U0)
        lane, m4 = t.lane.state, t.champ
    assert t.returned[0] and m4.a.gold == 0 and m4.a.invested == 9 * MU + gold_at_death


def test_power_is_strictly_increasing_per_ledger_unit_and_neutral_at_zero():
    t = pg.tables(C)
    for L in (1, 9, 18):
        assert INV.attack_power(L, 0) == t.ap[L][0] * pg.INVEST_ONE
        for d in (0, 1, 2):
            assert INV.hit_damage(d, L, 0) == t.dmg[d][L][0]
        assert INV.pressure_damage(L, 0) == t.pressure[L][0]
    grid = sorted({0, 1, 2, 3, MU - 1, MU, MU + 1, 7 * MU, 10 ** 9, 10 ** 11, INV.max_units - 1, INV.max_units}
                  | set(np.random.default_rng(0).integers(0, INV.max_units, 200).tolist()))
    ms = [INV.multiplier(u) for u in grid]
    assert all(b > a for a, b in zip(ms, ms[1:]))
    for u in (0, 1, MU, 10 ** 9, INV.max_units - 1):
        assert INV.multiplier(u + 1) > INV.multiplier(u)
    for L in (1, 18):
        dmg = [INV.hit_damage(2, L, u) for u in grid]
        assert all(b >= a for a, b in zip(dmg, dmg[1:]))
    with pytest.raises(ValueError, match="supported range"):
        INV.multiplier(INV.max_units + 1)


def test_rewards_are_zero_sum_ledger_exact_and_mirror_symmetric():
    agents = m4_suite()
    for seed, (x, y) in enumerate([(agents["gold_recaller"], agents["aggressive_trader"]),
                                   (UniformAgent(), UniformAgent())]):
        env = LaneEnv(C, max_turns=300)
        rng = np.random.default_rng(seed)
        obs = env.reset(seed=seed)
        earned, total = [0, 0], 0
        while True:
            before = env.champ
            r = env.step(x.act(obs[0], rng), y.act(obs[1], rng))
            i = r.info
            assert r.rewards[0] == -r.rewards[1]
            assert i["reward_units"] == i["earned_units"][0] - i["earned_units"][1]
            total += i["reward_units"]
            for s in (0, 1):
                earned[s] += i["earned_units"][s]
                me = env.champ.side(s)
                assert me.gold + me.invested == earned[s]                    # nothing lost, nothing clamped
                assert i["gold_unspendable"][s] == 0
                if i["invested"][s]:
                    assert i["returned"][s]
                if i["died"][s]:
                    assert me.invested == before.side(s).invested
            check_invariants(env.champ, C)
            if r.done:
                break
            obs = r.obs
        assert total == earned[0] - earned[1]


def test_mirror_symmetry_of_one_turn():
    st = M4State(side(1, invested=6 * MU, gold=2 * MU, xp=40), side(-1, invested=MU, xp=10))
    u = list(U0)
    u[U_TRADE_HIT[0]] = 0.0
    u[U_TRADE_HIT[1]] = 0.0
    t = resolve_turn_m4(LANE, st, 2, TR, TR, C, u=tuple(u))
    um = list(u)
    um[U_TRADE_HIT[0]], um[U_TRADE_HIT[1]] = u[U_TRADE_HIT[1]], u[U_TRADE_HIT[0]]
    tm = resolve_turn_m4(LANE.mirror(), st.mirror(), 2, TR, TR, C, u=tuple(um))
    assert tm.champ == t.champ.mirror() and tm.reward_units == -t.reward_units
    assert tm.trade_damage == t.trade_damage[::-1] and t.trade_damage[0] > t.trade_damage[1]


def test_observation_decodes_exactly_below_the_references_and_refuses_when_saturated():
    for a, b in [(0, 0), (1, 7 * MU + 1), (123_456_789, 3), (C.invest_obs_ref * MU - 1, C.invest_obs_gold_ref * MU - 1)]:
        st = M4State(side(-1, invested=a, gold=b, xp=60), side(2, invested=b, gold=a % (C.invest_obs_gold_ref * MU)))
        v = observe_m4(LANE, 5, 0, C, st)
        assert v.dtype == np.float32 and v.min() >= 0 and v.max() <= 1
        assert decode_m4(v, C)[2] == st
    st = M4State(side(-1, invested=C.invest_obs_ref * MU), side(2))
    v = observe_m4(LANE, 5, 0, C, st)
    assert v.max() <= 1
    with pytest.raises(ValueError, match="saturated"):
        decode_m4(v, C)


def test_scripted_recall_trigger_and_duel_edge_read_the_investment():
    from lanegame.env.agents_m4 import cash_ready
    t = pg.tables(C)
    assert not cash_ready(side(gold=4 * MU - 1), C, t, 4.0) and cash_ready(side(gold=4 * MU), C, t, 4.0)
    ctx = context(C)
    assert ctx.hit(1, 1, 7 * MU) > ctx.hit(1, 1, 0)


def test_per_turn_bound_allows_every_plate_under_unbounded_power():
    assert per_turn_bound(C) >= len(C.plate_thresholds) * C.plate_gold + C.kill_bounty
