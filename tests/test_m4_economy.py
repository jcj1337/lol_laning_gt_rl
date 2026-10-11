"""WP6: tower durability, plates, the gold ledger, items and Recall / death purchases (design §5 steps 11-15, §8.5,
§8.6, §11; plan WP6)."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame import progression as pg
from lanegame.champions import N_CHANCE, U_TRADE_DOUBLE, U_TRADE_HIT
from lanegame.champions_m4 import M4Side, M4State, check_invariants, initial_m4, resolve_turn_m4
from lanegame.config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, PRESSURE, SH, SP, champ_act
from lanegame.env import LaneEnv, UniformAgent, champion_suite
from lanegame.mechanics import initial_state
from lanegame.state import State

M4 = GameConfig.m4()
T = M4.tower
U0 = (1.0,) * N_CHANCE
HO, SA, TR = champ_act(SP, B_HOLD), champ_act(SP, B_SAFE), champ_act(SP, B_TRADE)
RECALL = champ_act(SP, B_RECALL)
MU = M4.gold_scale


def side(pos=0, hp=None, xp=0, items=0, absent=0, gold=0, tau=0):
    return M4Side(pos, pg.hp_max(pg.level_of(xp)) if hp is None else hp, absent, xp, items, gold, tau)


def siege_lane(mass):
    """A's crashed mass under B's tower, B's next wave waiting behind it."""
    return State(-4, 4, (), mass, -3, 4, (), 0, 0, 0, 0, 0)


def test_plate_is_paid_exactly_once_when_its_threshold_is_crossed():
    st = M4State(side(-2, tau=1599), side(-2))
    t = resolve_turn_m4(siege_lane(4), st, 2, HO, HO, M4, u=U0)
    assert t.tower_hits[0] >= 1 and t.champ.a.tower_dmg >= 1600
    assert t.plates_broken == (1, 0) and t.plate_units == (MU, 0) and t.plate_gold == (1.0, 0.0)
    assert t.reward_units == t.earned_units[0] - t.earned_units[1]
    t2 = resolve_turn_m4(t.lane.state, t.champ, 3, HO, HO, M4, u=U0)
    assert t2.plates_broken[0] == 0 and t2.plate_units[0] == 0


def test_several_plates_in_one_turn_are_all_paid_and_tau_clamps():
    st = M4State(side(T, tau=1599), side(-2))
    press = champ_act(SP, B_HOLD, PRESSURE)
    t = resolve_turn_m4(siege_lane(48), st, 2, press, HO, M4, u=U0)
    assert t.effective_objectives[0] == PRESSURE
    assert t.tower_damage_dealt[0] == 100 * t.tower_hits[0] + 400 >= 2001
    assert t.plates_broken[0] == 2 and t.plate_units[0] == 2 * MU
    st = M4State(side(T, tau=5990), side(-2))
    t = resolve_turn_m4(siege_lane(48), st, 2, press, HO, M4, u=U0)
    assert t.champ.a.tower_dmg == 6000 and t.tower_damage_dealt[0] == 10 and t.plates_broken[0] == 1
    t = resolve_turn_m4(t.lane.state, t.champ, 3, press if t.lane.state.asg else HO, HO, M4, u=U0)
    assert t.champ.a.tower_dmg == 6000 and t.plates_broken[0] == 0 and t.tower_damage_dealt[0] == 0


def test_plate_gold_while_away_and_the_presence_alternative():
    st = M4State(side(-T, absent=2, hp=300, tau=1599), side(-2))
    t = resolve_turn_m4(siege_lane(4), st, 2, CHAMP_ABSENT, HO, M4, u=U0)
    assert t.plate_units[0] == MU                                         # default: paid while away (design §11.3)
    t2 = resolve_turn_m4(siege_lane(4), st, 2, CHAMP_ABSENT, HO, M4.with_(plate_gold_requires_presence=True), u=U0)
    assert t2.plates_broken[0] == 1 and t2.plate_units[0] == 0            # alternative A9: broken but unpaid


def play_m4(env, actions_fn, turns, champ, lane=None, seed=0):
    env.max_turns = turns
    env.reset(seed=seed, root=lane, champ=champ)
    out = []
    for t in range(turns):
        a, b = actions_fn(t, env)
        r = env.step(a, b)
        out.append(r)
        if r.done:
            break
    return out


def test_recall_purchase_timing_with_plate_gold_earned_while_away():
    """Design §8.5: Recall on t -> away t, t + 1, purchase at the end of t + 1 (after that turn's gold), play t + 2
    with the new tier and full HP at the current level. Plate gold earned while away is spendable."""
    env = LaneEnv(M4, max_turns=5)
    start = M4State(side(-1, hp=200, xp=80, gold=3_500_000, tau=1550), side(0))

    def acts(t, e):
        me = e.champ.a
        a = RECALL if t == 0 else (CHAMP_ABSENT if me.absent > 0 else HO)
        return a, HO
    res = play_m4(env, acts, 4, start, lane=siege_lane(6))
    i0, i1, i2 = res[0].info, res[1].info, res[2].info
    assert i0["recalled"][0] and i0["purchases"][0] == 0 and i0["items"][0] == 0
    assert i1["away"][0] and i1["returned"][0]
    # the plate (1550 -> >= 1600) paid 1.0 during the absence; 3.5 + 1.0 >= 4.0 -> tier 1 bought at the end of t + 1
    assert sum(r.info["plate_units"][0] for r in res[:2]) == MU
    assert i1["purchases"][0] == 1 and i1["items"][0] == 1 and i1["gold"][0] == 500_000
    after = res[1].obs[0].champ.a
    assert after.items == 1 and after.hp == pg.hp_max(pg.level_of(80)) and after.absent == 0 and after.pos == -T
    assert i2["purchases"][0] == 0 and not i2["away"][0]
    # spending never enters the reward
    for r in res:
        assert r.info["reward_units"] == r.info["earned_units"][0] - r.info["earned_units"][1]


def test_death_purchase_timing():
    st = M4State(side(0, hp=50, gold=6_000_000), side(0))
    lane = State(0, 4, (), 0, 0, 4, (), 0, 0, 0, 0, 0)
    u = list(U0)
    u[U_TRADE_HIT[1]] = 0.0
    t = resolve_turn_m4(lane, st, 2, HO, TR, M4, u=tuple(u))
    assert t.died[0] and t.purchases[0] == 0 and t.champ.a.absent == 3
    lane, m4 = t.lane.state, t.champ
    bought = []
    for k in range(3, 7):
        a = CHAMP_ABSENT if m4.a.absent > 0 else HO
        t = resolve_turn_m4(lane, m4, k, a, HO, M4, u=U0)
        bought.append(t.purchases[0])
        lane, m4 = t.lane.state, t.champ
        if k == 5:
            assert t.returned[0] and m4.a.items == 1 and m4.a.hp == pg.hp_max(pg.level_of(m4.a.xp))
    assert bought == [0, 0, 1, 0]                                       # end of t + 3, then plays t + 4


def test_purchases_happen_only_on_return_turns_and_ledger_identities_hold_exactly():
    """500-turn rollouts: sum of rewards = earned_A - earned_B in integer ledger units; spendable gold =
    earned - purchases - clamped; purchases only on a return turn; the clamp never removes spendable gold."""
    suite = champion_suite()
    from lanegame.env.agents_m4 import m4_suite
    agents = m4_suite()
    for seed, (x, y) in enumerate([(agents["gold_recaller"], agents["aggressive_trader"]),
                                   (agents["presser"], agents["hold_farmer"]), (UniformAgent(), UniformAgent())]):
        env = LaneEnv(M4, max_turns=500)
        rng = np.random.default_rng(seed)
        obs = env.reset(seed=seed)
        earned, spent, clamped = [0, 0], [0, 0], [0, 0]
        total = 0
        prices = pg.DEFAULT.prices
        while True:
            items_before = (env.champ.a.items, env.champ.b.items)
            r = env.step(x.act(obs[0], rng), y.act(obs[1], rng))
            i = r.info
            total += i["reward_units"]
            assert i["reward_units"] == i["earned_units"][0] - i["earned_units"][1]
            assert i["earned_units"][0] == i["farm_units"][0] + i["plate_units"][0] + i["bounty_units"][0]
            for s in (0, 1):
                earned[s] += i["earned_units"][s]
                clamped[s] += i["gold_unspendable"][s]
                n = i["purchases"][s]
                if n:
                    assert i["returned"][s]
                spent[s] += sum(prices[items_before[s] + j] for j in range(n))
                me = r.obs[0].champ.side(s)
                assert me.gold == earned[s] - spent[s] - clamped[s]
                if i["gold_unspendable"][s]:
                    assert me.gold == pg.gold_cap(me.items) or i["purchases"][s]
            check_invariants(env.champ, M4)
            if r.done:
                break
            obs = r.obs
        assert total == earned[0] - earned[1]


def test_no_tier_is_bought_while_present():
    st = M4State(side(0, gold=9_000_000), side(0))
    lane = State(0, 4, (), 0, 0, 4, (), 0, 0, 0, 0, 0)
    t = resolve_turn_m4(lane, st, 2, HO, HO, M4, u=U0)
    assert t.purchases == (0, 0) and t.champ.a.items == 0 and t.champ.a.gold >= 9_000_000


def test_gold_is_clamped_at_the_remaining_prices():
    st = M4State(side(0, items=2, gold=0), side(0, items=1, gold=5_999_990))
    lane = State(0, 12, (), 0, 0, 12, (), 0, 0, 0, 0, 0)
    t = resolve_turn_m4(lane, st, 2, HO, HO, M4, u=U0)
    assert t.champ.a.gold == 0 and t.gold_clamped[0] == t.earned_units[0] > 0
    assert t.champ.b.gold == 6_000_000 and t.gold_clamped[1] == t.earned_units[1] - 10
