"""WP5: combat and progression in the resolver (design §5 steps 9-12, §7.3, §8; plan WP5)."""

import dataclasses
import random

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame import progression as pg
from lanegame.champions import (N_CHANCE, U_GANK, U_GANK_DAMAGE, U_TOWER, U_TRADE_DOUBLE, U_TRADE_HIT, _gank_damage,
                                champion_legal_actions)
from lanegame.champions_m4 import M4Side, M4State, check_invariants, initial_m4, resolve_turn_m4
from lanegame.config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, NONE, SP, champ_act
from lanegame.mechanics import initial_state, step_turn
from lanegame.state import State

M4 = GameConfig.m4()
T = M4.tower
CONTACT = State(0, 4, (), 0, 0, 4, (), 0, 0, 0, 0, 0)
NO_EVENTS = (1.0,) * N_CHANCE
TR, HO, SA = champ_act(SP, B_TRADE), champ_act(SP, B_HOLD), champ_act(SP, B_SAFE)
RECALL = champ_act(SP, B_RECALL)


def side(pos=0, hp=None, xp=0, items=0, absent=0, gold=0, tau=0):
    L = pg.level_of(xp)
    return M4Side(pos, pg.hp_max(L) if hp is None else hp, absent, xp, items, gold, tau)


def u_with(**idx):
    u = list(NO_EVENTS)
    for k, v in idx.items():
        name, s = k.rsplit("_", 1)
        u[{"hit": U_TRADE_HIT, "double": U_TRADE_DOUBLE, "tower": U_TOWER, "gank": U_GANK,
           "gdmg": U_GANK_DAMAGE}[name][int(s)]] = v
    return tuple(u)


def test_power_changes_damage_only_never_hit_probabilities():
    u = u_with(hit_0=0.0, double_0=0.0, hit_1=0.0, double_1=0.99)     # A: double hit, B: single hit
    base = resolve_turn_m4(CONTACT, M4State(side(0), side(0)), 2, TR, TR, M4, u=u)
    assert base.connected == (True, True)
    for xa, na, xb, nb in [(0, 0, 0, 0), (70, 1, 0, 0), (148, 2, 19, 1), (544, 2, 544, 0), (8, 0, 300, 2)]:
        t = resolve_turn_m4(CONTACT, M4State(side(0, xp=xa, items=na), side(0, xp=xb, items=nb)), 2, TR, TR, M4, u=u)
        assert t.hit_p == base.hit_p and t.double_p == base.double_p
        La, Lb = pg.level_of(xa), pg.level_of(xb)
        assert t.trade_damage == (pg.hit_damage(2, La, na), pg.hit_damage(1, Lb, nb))
    # skill moves the probabilities, never the damage of a hit
    sk = resolve_turn_m4(CONTACT, M4State(side(0), side(0)), 2, TR, TR, M4.with_(skill_a=0.9, skill_b=0.2), u=u)
    assert sk.hit_p != base.hit_p and sk.trade_damage == base.trade_damage


def test_gank_damage_is_trials_times_the_victims_hp_unit():
    for xp in (0, 70, 300, 544):
        L = pg.level_of(xp)
        for ud in (0.05, 0.5, 0.97):
            u = u_with(gank_0=0.0, gdmg_0=ud)
            t = resolve_turn_m4(CONTACT, M4State(side(0, xp=xp), side(0, xp=544, items=2)), 2, HO, HO, M4, u=u)
            assert t.ganked[0]
            X = _gank_damage(ud, t.gank_damage_p[0], M4)
            assert t.gank_damage[0] == X * pg.hp_unit(L)
    # the alternatives of the parameter table
    u = u_with(gank_0=0.0, gdmg_0=0.97)
    st = M4State(side(0, xp=70), side(0, xp=300))
    t_att = resolve_turn_m4(CONTACT, st, 2, HO, HO, M4.with_(gank_damage_scale="attacker_level"), u=u)
    t_none = resolve_turn_m4(CONTACT, st, 2, HO, HO, M4.with_(gank_damage_scale="none"), u=u)
    X = _gank_damage(0.97, t_att.gank_damage_p[0], M4)
    assert t_att.gank_damage[0] == X * pg.hp_unit(pg.level_of(300)) and t_none.gank_damage[0] == X * 100


def big_contact():
    return State(0, 12, (), 0, 0, 12, (), 0, 0, 0, 0, 0)          # 6 HP of each side die at the lane this turn


def test_level_up_adds_the_max_hp_increase_including_several_levels():
    shove_hold = champ_act(1, B_HOLD)                                 # 7 of B's HP die at the lane: 14 XP
    t = resolve_turn_m4(big_contact(), M4State(side(0, xp=7, hp=300), side(0)), 2, shove_hold, HO, M4, u=NO_EVENTS)
    assert t.xp_gain[0] == 14 and t.levels_before[0] == 1 and t.levels_after[0] == 3 and t.level_up[0] == 2
    assert t.champ.a.hp == 300 + pg.hp_max(3) - pg.hp_max(1)
    t = resolve_turn_m4(big_contact(), M4State(side(0, xp=7, hp=300), side(0)), 2, HO, HO, M4, u=NO_EVENTS)
    assert t.xp_gain[0] == 10 and t.level_up[0] == 1 and t.champ.a.hp == 300 + 30
    t = resolve_turn_m4(big_contact(), M4State(side(0, xp=69, hp=50), side(0)), 2, HO, HO, M4, u=NO_EVENTS)
    assert t.levels_after[0] == 6 and t.champ.a.hp == 50 + 766 - 729        # design §15 scenario 8
    assert t.champ.a.hp <= pg.hp_max(6)


def test_death_precedes_the_level_up():
    # A at L5 with 50 centi-HP, B at L6 lands a single hit of 128 centi: A dies although its XP crosses X(6) = 70
    u = u_with(hit_1=0.0, double_1=0.99)
    st = M4State(side(0, xp=69, hp=50), side(0, xp=75))
    t = resolve_turn_m4(big_contact(), st, 2, HO, TR, M4, u=u)
    assert t.trade_damage[1] == 128 and t.died[0]
    assert t.champ.a.hp == 0 and t.champ.a.absent == M4.death_turns and t.champ.a.pos == -T
    assert t.champ.a.xp > 70 and t.levels_after[0] == 6                    # XP of the death turn is kept
    assert t.bounty[1] == M4.kill_bounty
    # the same turn with 150 centi-HP: survives at 22 and gains 766 - 729
    t2 = resolve_turn_m4(big_contact(), st.replace_side(0, hp=150), 2, HO, TR, M4, u=u)
    assert not t2.died[0] and t2.champ.a.hp == 150 - 128 + 766 - 729


def test_respawn_uses_the_new_maximum():
    st = M4State(side(-T, hp=0, xp=75, absent=1), side(0))
    t = resolve_turn_m4(CONTACT, st, 2, CHAMP_ABSENT, HO, M4, u=NO_EVENTS)
    assert t.returned[0] and t.champ.a.hp == pg.hp_max(6) and t.champ.a.absent == 0


def kills_lane_step(kills):
    """A lane step whose authoritative call reports the given kills (for constructed XP / gold geometry)."""
    calls = []

    def step(s, k, a, b, reach):
        res = step_turn(s, k, a, b, M4, champion_reach=reach)
        calls.append(1)
        return dataclasses.replace(res, kills=kills) if len(calls) == 2 else res
    return step


def test_xp_range_three_versus_gold_reach_one():
    # A's minions die at B's +2 (under A's tower, the freeze geometry). B's XP and gold by B's cell:
    kills = ({}, {2: 4})
    # (B's cell, B's behavior, B's front cell so that B ends exactly there, gold expected, XP expected)
    cases = ((1, HO, 1, True, True), (0, HO, 0, False, True), (-1, SA, 0, False, True), (-2, HO, -4, False, False))
    for b_pos, beh, front, gold, xp in cases:
        st = M4State(side(-2), side(b_pos))
        lane = State(-4, 4, (), 0, front, 4, (), 0, 0, 0, 0, 0)
        t = resolve_turn_m4(lane, st, 1, SA, beh, M4, lane_step=kills_lane_step(kills), u=NO_EVENTS)
        assert t.end[1] == b_pos
        assert (t.farm_units[1] > 0) == gold, (b_pos, t.farm_units)
        assert (t.xp_gain[1] > 0) == xp, (b_pos, t.xp_gain)
        if xp:
            assert t.xp_gain[1] == 2 * 4


def test_xp_needs_presence_channeling_counts_recalled_does_not():
    lane = big_contact()
    # A recalls from -1 and leaves: no XP; B gets XP
    t = resolve_turn_m4(lane, M4State(side(-1), side(0)), 2, RECALL, HO, M4, u=NO_EVENTS)
    assert t.recalled[0] and t.xp_gain[0] == 0 and t.xp_gain[1] > 0
    # A's recall is interrupted by B's trade: A channels, present, earns XP but no gold
    t = resolve_turn_m4(lane, M4State(side(-1), side(1)), 2, RECALL, TR, M4, u=NO_EVENTS)
    assert t.interrupted[0] and t.modes[0] == "channel"
    assert t.xp_gain[0] > 0 and t.farm_units[0] == 0
    # an absent champion earns nothing
    t = resolve_turn_m4(lane, M4State(side(-T, absent=2), side(0)), 2, CHAMP_ABSENT, HO, M4, u=NO_EVENTS)
    assert t.xp_gain[0] == 0


def test_tower_adjacent_shot():
    # B holds at its own -1 (A's +1, next to B's tower); A trades from +1 and stays there in contact
    lane = State(1, 4, (), 0, -1, 4, (), 0, 0, 0, 0, 0)
    st = M4State(side(1, xp=148), side(-1))
    t = resolve_turn_m4(lane, st, 1, TR, HO, M4, u=u_with(tower_0=0.49))
    assert t.end[0] == T - 1 and t.connected[0] and t.adjacent_shot[0]
    assert t.tower_p[0] == 0.5 and t.tower_damage[0] == pg.hp_unit(pg.level_of(148))
    assert resolve_turn_m4(lane, st, 1, TR, HO, M4, u=u_with(tower_0=0.51)).tower_damage[0] == 0
    # busy tower (A's crashed minions under it): 0.25
    busy = lane._replace(asg=4)
    tb = resolve_turn_m4(busy, st, 1, TR, HO, M4, u=u_with(tower_0=0.3))
    assert tb.tower_p[0] == 0.25 and tb.tower_damage[0] == 0
    # holding (not trading) next to the enemy tower never draws the shot
    th = resolve_turn_m4(lane, st, 1, HO, HO, M4, u=u_with(tower_0=0.0))
    assert th.end[0] == T - 1 and th.tower_p[0] == 0.0 and th.tower_damage[0] == 0
    # the ablation switches it off; the dive rule is unchanged
    off = M4.with_(tower_adjacent_p=0.0, tower_adjacent_p_busy=0.0)
    assert resolve_turn_m4(lane, st, 1, TR, HO, off, u=u_with(tower_0=0.0)).tower_damage[0] == 0


def test_dive_shot_scales_with_the_victims_unit():
    lane = State(-4, 4, (), 8, -3, 4, (), 0, 0, 0, 0, 0)           # A sieging, B hugging its tower
    st = M4State(side(1, xp=300), side(-2))
    t = resolve_turn_m4(lane, st, 2, TR, SA, M4, u=u_with(tower_0=0.0))
    assert t.end[0] == T and t.tower_p[0] == M4.tower_shot_p_busy
    assert t.tower_damage[0] == pg.hp_unit(pg.level_of(300))


def test_soak_10000_random_turns_respects_caps_and_invariants():
    rng, nrng = random.Random(0), np.random.default_rng(0)
    lane, m4, k = initial_state(M4), initial_m4(M4), 0
    discarded = 0
    for _ in range(10_000):
        a = rng.choice(champion_legal_actions(lane, m4, 0, M4))
        b = rng.choice(champion_legal_actions(lane, m4, 1, M4))
        t = resolve_turn_m4(lane, m4, k, a, b, M4, u=nrng.random(N_CHANCE))
        check_invariants(t.champ, M4)
        discarded += sum(t.xp_discarded)
        for s in (0, 1):
            assert t.champ.side(s).xp <= 544 and t.champ.side(s).items <= 2
            assert max(t.damage_taken) <= 3 * 660 + 4 * 229
        lane, m4, k = t.lane.state, t.champ, k + 1
    assert discarded > 0 and m4.a.xp == 544 and m4.b.xp == 544
