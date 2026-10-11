"""WP3: tower hits and durability mode (design §11; plan WP3)."""

import random

import pytest

from lanegame import GameConfig
from lanegame.config import PRESSURE, SH, SP, act
from lanegame.lane_enum import DEFAULT_LANE_BOUNDS, lane_bounds, lane_signature
from lanegame.mechanics import initial_state, legal_actions, minion_damage, step_turn
from lanegame.state import State

CFG = GameConfig.m4()


def crash_hits(hp: int) -> int:
    """Total minion tower hits of one undisturbed crash of `hp` HP (no defenders, 1 HP/tick tower fire)."""
    # A front of `hp` HP one cell outside B's tower, alone in the lane, shoving: it crashes on the first tick.
    s = State(1, hp, (), 0, -4, 0, (), 0, 0, 0, 0, 0)._replace(bh=0, bx=0)
    total, k = 0, 1                                    # k = 1: no spawn at the end of turn 1 (spawns follow even k)
    res = step_turn(s, k, SH, SP, CFG)
    assert res.crash_hp[0] == hp
    total += res.tower_hits[0]
    s = res.state
    for k in range(2, 40):
        s = s._replace(bx=0, bh=0, br=())             # keep B's lane empty: the crash is undisturbed
        s = s._replace(ax=0, ah=0, ar=())             # and A sends nothing more
        res = step_turn(s, k, SP, SP, CFG)
        total += res.tower_hits[0]
        s = res.state
        if s.asg == 0:
            break
    return total


@pytest.mark.parametrize("hp,hits", [(4, 4), (8, 9), (12, 18)])
def test_crash_tower_hits_match_the_design_table(hp, hits):
    assert crash_hits(hp) == hits


def test_tower_hits_are_start_of_tick_mass_damage_and_zero_without_a_siege():
    rng = random.Random(3)
    s, k = initial_state(CFG), 0
    for _ in range(3000):
        a = rng.choice(legal_actions(s, 0, CFG))
        b = rng.choice(legal_actions(s, 1, CFG))
        res = step_turn(s, k, a, b, CFG, record=True)
        for side in (0, 1):
            m, hits = s.siege_of(side), 0
            for tr in res.trace:
                if m > 0:
                    hits += minion_damage(m, CFG)
                m = tr["siege"][side]
            assert res.tower_hits[side] == hits                 # the prototype's trace computation
            if s.siege_of(side) == 0 and res.crash_hp[side] == 0:
                assert res.tower_hits[side] == 0
            assert res.tower_hits[side] <= 24                    # analytical per-turn bound (design §10.5.3)
        s, k = res.state, k + 1


def test_durability_mode_grants_and_claims_no_plate_opportunities():
    rng = random.Random(5)
    s, k = initial_state(CFG), 0
    assert (s.ap, s.bp) == (0, 0)
    pressured = 0
    for _ in range(4000):
        la, lb = legal_actions(s, 0, CFG), legal_actions(s, 1, CFG)
        a = next((x for x in la if x // 6 == PRESSURE), rng.choice(la))
        b = rng.choice(lb)
        pressured += a // 6 == PRESSURE
        res = step_turn(s, k, a, b, CFG)
        assert res.plates == (0, 0) and (res.state.ap, res.state.bp) == (0, 0)
        s, k = res.state, k + 1
    assert pressured > 0


def test_m4_lane_has_the_default_lane_signature():
    assert lane_signature(CFG) == lane_signature(GameConfig())


@pytest.mark.slow
def test_b1_enumerated_lane_bounds():
    """Exhaustive reachable-lane enumeration (about 5 min): the constants behind the evaluation tail bound and the
    lossless observation."""
    b = lane_bounds(CFG)
    assert b["pairs"] == DEFAULT_LANE_BOUNDS["pairs"]
    assert b["A_max"] <= 32
    assert b["K_max"] <= 27
    assert b["tower_hits_max"] <= 24
    assert b["rears_per_cell_max"] <= CFG.rear_count_max == 4
    assert b["siege_max"] <= CFG.siege_cap == 48
    for k in ("A_max", "K_max", "siege_max", "tower_hits_max", "rears_per_cell_max"):
        assert b[k] == DEFAULT_LANE_BOUNDS[k], k
