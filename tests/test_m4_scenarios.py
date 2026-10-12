"""WP8: replayable witnesses of the design scenarios (§15) and their mechanical assertions."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame import progression as pg
from lanegame.champions import N_CHANCE, U_TRADE_HIT, trade_hit_probability
from lanegame.champions_m4 import level, resolve_turn_m4
from lanegame.config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, F, SH, SP, champ_act
from lanegame.env.agents_m4 import duel_edge
from lanegame.env.observation import feature_names, observe
from lanegame.mechanics import wave_actions
from lanegame.scenarios_m4 import load_witnesses, replay, scenario_config, scenario_predicates

W = load_witnesses()
T = 2
U1 = (1.0,) * N_CHANCE


def env_of(name):
    return replay(name, W[name])


@pytest.mark.parametrize("name", sorted(W))
def test_witness_reproduces_its_state_and_satisfies_the_scenario(name):
    env = env_of(name)
    cfg = scenario_config(name)
    assert scenario_predicates(cfg)[name](env.state, env.champ, env.k % cfg.phase_period)
    assert len(W[name]["witness"]["actions"]) == W[name]["k"]


def test_all_ten_design_scenarios_have_witnesses():
    assert {n.split("_")[0] for n in W if n[0].isdigit()} == ({str(i) for i in range(1, 11)} - {"6"}) | {"6a", "6b"}


def test_1_equal_power_freeze_separates_gold_from_xp():
    env = env_of("1_equal_power_freeze")
    cfg, lane, m4, k = env.cfg, env.state, env.champ, env.k
    gold = xp = 0
    for i in range(8):
        wa = F if F in wave_actions(lane, 0, cfg) else SP
        t = resolve_turn_m4(lane, m4, k + i, champ_act(wa, B_HOLD), champ_act(SP, B_SAFE), cfg, u=None)
        gold += t.farm_units[1]
        xp += t.xp_gain[1]
        assert t.end[1] <= -1
        lane, m4 = t.lane.state, t.champ
    assert gold == 0 and xp > 0                    # Safe behind the freeze: XP, no gold


def test_2_strong_freezer_has_a_positive_duel_edge():
    env = env_of("2_strong_freezer")
    oa, ob = env.observe()
    assert duel_edge(oa) > 0 > duel_edge(ob)


def test_3_weak_breaker_rule():
    env = env_of("3_weak_breaker")
    cfg, lane, m4, k = env.cfg, env.state, env.champ, env.k
    for i in range(12):
        wa = F if F in wave_actions(lane, 0, cfg) else SP
        t = resolve_turn_m4(lane, m4, k + i, champ_act(wa, B_HOLD), champ_act(SH, B_HOLD), cfg, u=None)
        if t.freeze_maintained[0] and lane.bh >= 6 and t.reach[1] >= lane.bx:
            assert t.lane.events["freeze_held"][1] == 0          # a reaching shove of >= 6 HP breaks it
        lane, m4 = t.lane.state, t.champ


def test_4_skill_moves_hit_probability_economy_moves_damage():
    env = env_of("4_skill_vs_economy")
    cfg, lane, m4 = env.cfg, env.state, env.champ
    t = pg.tables(cfg)
    assert trade_hit_probability(lane, 0, B_HOLD, False, cfg) > trade_hit_probability(lane, 1, B_HOLD, False, cfg)
    La, Lb = level(m4.a, t), level(m4.b, t)
    assert t.dmg[1][Lb][m4.b.items] > t.dmg[1][La][m4.a.items]


def test_5_recall_to_buy():
    env = env_of("5_low_hp_recall_to_buy")
    cfg, lane, m4, k = env.cfg, env.state, env.champ, env.k
    recall = champ_act(SP, B_RECALL)
    t1 = resolve_turn_m4(lane, m4, k, recall, champ_act(SP, B_HOLD), cfg, u=U1)
    assert t1.recalled[0]
    t2 = resolve_turn_m4(t1.lane.state, t1.champ, k + 1, CHAMP_ABSENT, champ_act(SP, B_HOLD), cfg, u=U1)
    assert t2.purchases[0] >= 1 and t2.champ.a.items == m4.a.items + t2.purchases[0]
    assert t2.champ.a.hp == pg.tables(cfg).hp_max[level(t2.champ.a, pg.tables(cfg))]


@pytest.mark.parametrize("name", ["6a_recall_cheap", "6b_recall_expensive"])
def test_6_recall_absence_lasts_exactly_two_turns(name):
    env = env_of(name)
    cfg, lane, m4, k = env.cfg, env.state, env.champ, env.k
    t1 = resolve_turn_m4(lane, m4, k, champ_act(SP, B_RECALL), champ_act(SP, B_HOLD), cfg, u=U1)
    t2 = resolve_turn_m4(t1.lane.state, t1.champ, k + 1, CHAMP_ABSENT, champ_act(SP, B_HOLD), cfg, u=U1)
    assert t1.away[0] and t2.away[0] and t2.returned[0] and t2.champ.a.absent == 0


def test_7_tower_damage_persists_and_plates_pay_once():
    rec = W["7_crash_bounce_plates"]
    env = env_of("7_crash_bounce_plates")
    from lanegame.env.agents_m4 import gate_attackers, m4_suite
    pool = {**m4_suite(), **gate_attackers()}
    x, y = pool[rec["agents"][0]], pool[rec["agents"][1]]
    env.max_turns = env.turns + 80
    obs = env.observe()
    tau0, paid = env.champ.a.tower_dmg, 0
    for _ in range(80):
        r = env.step(x.act(obs[0]), y.act(obs[1]))
        assert env.champ.a.tower_dmg >= tau0
        tau0 = env.champ.a.tower_dmg
        paid += r.info["plates_broken"][0]
        if r.done:
            break
        obs = r.obs
    assert pg.plates_broken(tau0) == pg.plates_broken(W["7_crash_bounce_plates"]["champ"][0][6]) + paid


def test_8_level_up_in_combat_death_first():
    env = env_of("8_levelup_in_combat")
    cfg, lane, m4, k = env.cfg, env.state, env.champ, env.k
    t = pg.tables(cfg)
    L = level(m4.a, t)
    # survive: no damage, enough XP from a constructed big fight is not needed: check the HP rule on this turn
    tr = resolve_turn_m4(lane, m4, k, champ_act(SP, B_HOLD), champ_act(SP, B_HOLD), cfg, u=U1)
    if tr.level_up[0] and not tr.died[0]:
        assert tr.champ.a.hp == m4.a.hp - tr.damage_taken[0] + t.hp_max[tr.levels_after[0]] - t.hp_max[L]
    # a lethal trade kills even if the XP crosses a threshold
    u = list(U1)
    u[U_TRADE_HIT[1]] = 0.0
    m4_low = m4.replace_side(0, hp=1)
    td = resolve_turn_m4(lane, m4_low, k, champ_act(SP, B_HOLD), champ_act(SP, B_TRADE), cfg, u=tuple(u))
    if td.connected[1]:
        assert td.died[0] and td.champ.a.hp == 0


def test_9_beyond_turn_32_the_observation_has_no_time():
    env = env_of("9_beyond_turn_32")
    cfg = env.cfg
    assert env.k >= 64
    for s in (0, 1):
        v = observe(env.state, env.k, s, cfg, env.champ)
        assert np.array_equal(v, observe(env.state, env.k % 4, s, cfg, env.champ))
    assert not any("turn" in n for n in feature_names(cfg))


def test_10_caps_without_overflow():
    env = env_of("10_caps")
    cfg = env.cfg
    names = feature_names(cfg)
    oa, _ = env.observe()
    assert oa.vector[names.index("own_champ_xp_progress")] == 0 and oa.vector[names.index("own_champ_can_buy")] == 0
    assert env.champ.a.gold == 0 and env.champ.a.xp == 544
    env.max_turns = env.turns + 50
    from lanegame.env.agents_m4 import m4_suite
    h = m4_suite()["hold_farmer"]
    for _ in range(50):
        oa, ob = env.observe()
        r = env.step(h.act(oa), h.act(ob))
        assert env.champ.a.xp <= 544 and env.champ.a.gold == 0 and env.champ.a.items == 2
        assert r.obs[0].vector.max() <= 1.0
        if r.done:
            break


def test_xp_denial_needs_the_own_tower_cell():
    env = env_of("xp_denial")
    assert env.champ.b.pos == -T and env.champ.a.pos >= 0


def test_item_purchase_on_the_last_absent_turn():
    env = env_of("item_purchase")
    cfg, lane, m4, k = env.cfg, env.state, env.champ, env.k
    t = resolve_turn_m4(lane, m4, k, CHAMP_ABSENT, champ_act(SP, B_HOLD), cfg, u=U1)
    assert t.returned[0] and t.purchases[0] >= 1


def test_wave_crash_hits_the_tower():
    env = env_of("wave_crash")
    t = resolve_turn_m4(env.state, env.champ, env.k, champ_act(SP, B_HOLD), champ_act(SP, B_HOLD), env.cfg, u=U1)
    assert t.tower_hits[0] >= 3


def test_tower_plate_is_paid_once_when_crossed():
    env = env_of("tower_plate")
    lane, m4, k, cfg = env.state, env.champ, env.k, env.cfg
    paid = 0
    for i in range(6):
        t = resolve_turn_m4(lane, m4, k + i, champ_act(SP, B_HOLD), champ_act(SP, B_HOLD), cfg, u=U1)
        paid += t.plates_broken[0]
        assert t.plate_units[0] == t.plates_broken[0] * cfg.gold_scale
        lane, m4 = t.lane.state, t.champ
    assert paid == pg.plates_broken(m4.a.tower_dmg) - pg.plates_broken(env.champ.a.tower_dmg)
