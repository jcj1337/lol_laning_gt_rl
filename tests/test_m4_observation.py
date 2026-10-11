"""WP7: the lossless M4 observation and its decoder (design §12; plan WP7, tests O1-O7)."""

import collections
import random

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame import progression as pg
from lanegame.champions import N_CHANCE, champion_legal_actions, swap_seats
from lanegame.champions_m4 import M4_SIDE_FIELDS, M4Side, M4State, initial_m4, resolve_turn_m4
from lanegame.env.observation import feature_names, obs_size, observe
from lanegame.env.observation_m4 import decode_m4
from lanegame.mechanics import initial_state
from lanegame.state import State

M4 = GameConfig.m4()
ASYM = GameConfig.m4(skill_a=0.7, skill_b=0.35)


def roundtrip(lane, m4, k, cfg):
    for side in (0, 1):
        v = observe(lane, k, side, cfg, m4)
        assert v.dtype == np.float32 and v.min() >= 0.0 and v.max() <= 1.0 and len(v) == 88
        ph, l2, m2, skills = decode_m4(v, cfg)
        exp_l = lane if side == 0 else lane.mirror()
        exp_m = m4 if side == 0 else m4.mirror()
        assert ph == k % cfg.phase_period and l2 == exp_l._replace(ap=0, bp=0) and m2 == exp_m
        own = cfg.skill_a if side == 0 else cfg.skill_b
        assert skills == (np.float32(own), np.float32(cfg.skill_b if side == 0 else cfg.skill_a))


def soak_states(n, seed, cfg=M4):
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    lane, m4, k = initial_state(cfg), initial_m4(cfg), 0
    for _ in range(n):
        yield lane, m4, k
        a = rng.choice(champion_legal_actions(lane, m4, 0, cfg))
        b = rng.choice(champion_legal_actions(lane, m4, 1, cfg))
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, u=nrng.random(N_CHANCE))
        lane, m4, k = t.lane.state, t.champ, k + 1


def test_layout_size_and_names():
    assert obs_size(M4) == 88 and len(feature_names(M4)) == 88 and len(set(feature_names(M4))) == 88


def test_o1_full_state_roundtrip_on_a_random_legal_soak():
    """50 000 states of random-legal play (both seats), asymmetric skills."""
    for lane, m4, k in soak_states(50_000, 1, ASYM):
        roundtrip(lane, m4, k, ASYM)


@pytest.mark.slow
def test_o1_lane_block_roundtrip_on_the_exhaustive_reachable_lane():
    from lanegame.lane_enum import reachable_lane
    seen = reachable_lane(M4)
    assert len(seen) == 20572
    m4 = initial_m4(M4)
    for lane, ph in seen:
        roundtrip(lane, m4, ph, M4)


@pytest.mark.slow
def test_o2_m2_aliased_lane_states_have_distinct_m4_encodings():
    from lanegame.lane_enum import reachable_lane
    legacy = GameConfig()
    groups = collections.defaultdict(list)
    for lane, ph in reachable_lane(M4):
        groups[observe(lane, ph, 0, legacy)[2:].tobytes()].append((lane, ph))
    colliding = [g for g in groups.values() if len(g) > 1]
    assert len(colliding) == 96                                    # design §12.3
    m4 = initial_m4(M4)
    for g in colliding:
        enc = {observe(lane, ph, 0, M4, m4).tobytes() for lane, ph in g}
        assert len(enc) == len(g)


def obs_with_gold(gold, items=0):
    m4 = initial_m4(M4).replace_side(0, gold=gold, items=items)
    return observe(initial_state(M4), 0, 0, M4, m4), m4


def test_o3_gold_is_exact_and_the_affordability_bits_are_exact():
    names = feature_names(M4)
    i_g, i_buy, i_two = names.index("own_champ_gold"), names.index("own_champ_can_buy"), names.index("own_champ_can_buy_two")
    v4, _ = obs_with_gold(4_000_000)
    v9, _ = obs_with_gold(9_000_000)
    assert not np.array_equal(v4, v9)                               # the draft feature min(1, g / price) aliased them
    for g, buy, two in ((3_999_999, 0, 0), (4_000_000, 1, 0), (9_999_999, 1, 0), (10_000_000, 1, 1)):
        v, _ = obs_with_gold(g)
        assert (v[i_buy], v[i_two]) == (buy, two)
    v, _ = obs_with_gold(5_999_999, items=1)
    assert (v[i_buy], v[i_two]) == (0, 0)
    v, _ = obs_with_gold(6_000_000, items=1)
    assert (v[i_buy], v[i_two]) == (1, 0)
    # exhaustively: every integer balance 0 .. 10^7 decodes exactly from its float32 feature
    g = np.arange(0, 10 ** 7 + 1, dtype=np.int64)
    f32 = (g / 1e7).astype(np.float32)
    assert np.array_equal(np.rint(f32.astype(np.float64) * 1e7).astype(np.int64), g)


def test_o4_hp_xp_and_tower_damage_decode_exactly():
    t = pg.DEFAULT
    for L in range(1, 19):
        H = t.hp_max[L]
        h = np.arange(0, H + 1)
        assert np.array_equal(np.rint((h / H).astype(np.float32).astype(np.float64) * H).astype(int), h)
        if L < 18:
            n = t.xp_next[L]
            x = np.arange(0, n)
            assert np.array_equal(np.rint((x / n).astype(np.float32).astype(np.float64) * n).astype(int), x)
        lv = (L - 1) / 17
        assert 1 + int(round(float(np.float32(lv)) * 17)) == L
    tau = np.arange(0, 6001)
    assert np.array_equal(np.rint((tau / 6000).astype(np.float32).astype(np.float64) * 6000).astype(int), tau)


def test_o5_runtime_guards_raise_instead_of_clipping():
    m4 = initial_m4(M4)
    five = State(0, 4, (-3, -3, -3, -3, -3), 0, -4, 4, (), 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="rear_count_max"):
        observe(five, 0, 0, M4, m4)
    four = State(0, 4, (-3, -3, -3, -3), 0, -4, 4, (), 0, 0, 0, 0, 0)
    observe(four, 0, 0, M4, m4)
    with pytest.raises(ValueError, match="siege"):
        observe(State(0, 4, (), 49, -4, 4, (), 0, 0, 0, 0, 0), 0, 0, M4, m4)


def test_o6_mirror_property_and_phase_periodicity():
    for i, (lane, m4, k) in enumerate(soak_states(3000, 2, ASYM)):
        vb = observe(lane, k, 1, ASYM, m4)
        assert np.array_equal(vb, observe(lane.mirror(), k, 0, swap_seats(ASYM), m4.mirror()))
        if i % 10 == 0:
            for side in (0, 1):
                assert np.array_equal(observe(lane, k, side, ASYM, m4), observe(lane, k + 4 * 37, side, ASYM, m4))


def test_o7_every_state_field_is_observed():
    assert M4_SIDE_FIELDS == ("pos", "hp", "absent", "xp", "items", "gold", "tower_dmg")
    assert State._fields == ("ax", "ah", "ar", "asg", "bx", "bh", "br", "bsg", "va", "vb", "ap", "bp")
    # a state in which every field differs from its default and between the sides decodes to itself
    lane = State(1, 7, (0, -2, -4), 0, -3, 5, (-4,), 3, 4, 2, 0, 0)
    m4 = M4State(M4Side(1, 512, 0, 100, 1, 1_234_567, 1234), M4Side(-2, 0, 3, 300, 2, 0, 4321))
    roundtrip(lane, m4, 3, ASYM)
    # an extra field without a feature would make the decoder's reconstruction incomplete
    ph, l2, m2, _ = decode_m4(observe(lane, 3, 0, ASYM, m4), ASYM)
    assert set(m2.a._asdict()) == set(M4_SIDE_FIELDS) and set(l2._asdict()) == set(State._fields)
