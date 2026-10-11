"""Champion mode M2c: stochastic trading, mechanical skill and tower shots. The hit-chance formula, skill
monotonicity (offense and defense), behavior and tower protection, the 0/1/2 damage distribution, the
opportunity cost of a connected trade, symmetry, and reproducibility of the chance stream."""

import itertools

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import (CHANNEL, N_CHANCE, RECALL, TRADING, U_TOWER, U_TRADE_DOUBLE, U_TRADE_HIT, ChampSide,
                                ChampState, mirror_chance, resolve_turn, swap_seats, trade_damage_distribution,
                                trade_double_probability, trade_hit_probability)
from lanegame.config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, CHAMP_ABSENT, SH, SP, champ_act
from lanegame.env import LaneEnv, play_episode
from lanegame.state import State

CFG = GameConfig.champions(horizon=16)
T = CFG.tower
CONTACT = State(0, 4, (), 0, 0, 4, (), 0, 0, 0)                 # fronts fighting at the lane (x = 0)
STACKED = State(0, 12, (), 0, 0, 4, (), 0, 0, 0)                # A's front carries two stacks
SIEGE_A = State(-4, 4, (), 8, -3, 4, (), 0, 0, 0, 2, 0)         # A's 8 HP under B's tower, B's wave waiting at -3
DIVE = State(1, 4, (), 0, -2, 4, (), 0, 0, 0)                   # B's front fights at its own tower cell
DIVE_BUSY = State(1, 4, (), 8, -2, 4, (), 0, 0, 0, 2, 0)        # ... while A's crashed minions are under that tower
ZERO = (0.0,) * N_CHANCE                                        # every chance event that can happen does
ONE = (1.0,) * N_CHANCE                                         # none does


def ch(pa, pb, hpa=6, hpb=6):
    return ChampState(ChampSide(pa, hpa, 0), ChampSide(pb, hpb, 0))


def H(w=SP):
    return champ_act(w, B_HOLD)


def Y(w=SP):
    return champ_act(w, B_YIELD)


def TR(w=SP):
    return champ_act(w, B_TRADE)


def S(w=SP):
    return champ_act(w, B_SAFE)


# ------------------------------------------------------------------ the formula
def test_hit_chance_defaults():
    p = lambda beh, tower=False, lane=CONTACT, cfg=CFG: trade_hit_probability(lane, 0, beh, tower, cfg)
    assert p(B_HOLD) == pytest.approx(0.55) and p(B_TRADE) == pytest.approx(0.55)
    assert p(B_YIELD) == pytest.approx(0.30) and p(B_SAFE) == pytest.approx(0.15) and p(B_RECALL) == pytest.approx(0.55)
    assert p(B_HOLD, tower=True) == pytest.approx(0.40)                     # tower protection
    assert p(B_HOLD, lane=STACKED) == pytest.approx(0.65)                   # +2 stacks of minion support
    assert trade_hit_probability(STACKED, 1, B_HOLD, False, CFG) == pytest.approx(0.45)
    assert trade_double_probability(0, CFG) == pytest.approx(0.3)
    assert trade_damage_distribution(0.55, 0.3) == pytest.approx((0.45, 0.385, 0.165))
    lo = CFG.with_(skill_a=0.0, skill_b=1.0)
    assert p(B_SAFE, tower=True, cfg=lo) == CFG.trade_hit_min                # clipped below
    assert trade_hit_probability(STACKED, 0, B_HOLD, False, CFG.with_(skill_a=1.0, skill_b=0.0)) == CFG.trade_hit_max


def test_skill_monotonicity_offense_and_defense():
    """Higher attacker skill never lowers the hit chance or the 2-HP chance; higher defender skill never raises
    them; at equal skill (any level) the numbers are those of the 0.5 / 0.5 game."""
    grid = np.linspace(0, 1, 6)
    for beh, tower, lane in itertools.product((B_HOLD, B_YIELD, B_TRADE, B_SAFE), (False, True), (CONTACT, STACKED)):
        for m_def in grid:
            hs = [trade_hit_probability(lane, 0, beh, tower, CFG.with_(skill_a=m, skill_b=m_def)) for m in grid]
            qs = [trade_double_probability(0, CFG.with_(skill_a=m, skill_b=m_def)) for m in grid]
            assert all(np.diff(hs) >= -1e-12) and all(np.diff(qs) >= -1e-12)
        for m_att in grid:
            hs = [trade_hit_probability(lane, 0, beh, tower, CFG.with_(skill_a=m_att, skill_b=m)) for m in grid]
            assert all(np.diff(hs) <= 1e-12)
        for m in grid:
            assert trade_hit_probability(lane, 0, beh, tower, CFG.with_(skill_a=m, skill_b=m)) == \
                pytest.approx(trade_hit_probability(lane, 0, beh, tower, CFG))
    strong, weak = CFG.with_(skill_a=0.8, skill_b=0.3), CFG
    e = lambda cfg, s: sum(d * p for d, p in enumerate(trade_damage_distribution(
        trade_hit_probability(CONTACT, s, B_TRADE, False, cfg), trade_double_probability(s, cfg))))
    assert e(strong, 0) > e(weak, 0) > e(strong, 1)                     # expected damage both ways


def test_defensive_behaviors_lower_the_hit_chance():
    for lane in (CONTACT, STACKED):
        h = {b: trade_hit_probability(lane, 0, b, False, CFG) for b in (B_HOLD, B_YIELD, B_TRADE, B_SAFE)}
        assert h[B_SAFE] <= h[B_YIELD] <= h[B_HOLD] == h[B_TRADE]


def test_skill_changes_execution_only():
    """Positions, modes, the lane, farm and gank probabilities do not depend on skill; only the trade
    probabilities do."""
    for a, b, champ in ((TR(), H(), ch(0, 0)), (TR(SH), TR(), ch(-T, 0)), (Y(), TR(), ch(0, 0)), (H(), S(), ch(1, -1))):
        t0 = resolve_turn(CONTACT, champ, 2, a, b, CFG)
        t1 = resolve_turn(CONTACT, champ, 2, a, b, CFG.with_(skill_a=0.9, skill_b=0.1))
        assert (t0.end, t0.modes, t0.connected, t0.reach, t0.farm, t0.gank_p, t0.lane.state) == \
               (t1.end, t1.modes, t1.connected, t1.reach, t1.farm, t1.gank_p, t1.lane.state)
        if t0.connected[0]:
            assert t1.hit_p[0] > t0.hit_p[0]


# ------------------------------------------------------------------ resolution
def test_connected_trade_rolls_and_pays_its_opportunity_cost():
    t = resolve_turn(CONTACT, ch(0, 0), 2, TR(SH), H(), CFG, u=ZERO)
    assert t.connected == (True, False) and t.modes[0] == TRADING
    assert t.hit_p == (pytest.approx(0.55), 0.0) and t.trade_damage == (2, 0) and t.damage_taken == (0, 2)
    assert t.champ.b.hp == 4 and t.champ.a.hp == 6                       # the Hold defender does not hit back
    hold = resolve_turn(CONTACT, ch(0, 0), 2, H(SH), H(), CFG, u=ZERO)
    assert t.lane.events["killed_hp"][1] < hold.lane.events["killed_hp"][1]   # no shove contribution
    t_sp = resolve_turn(CONTACT, ch(0, 0), 2, TR(), H(), CFG, u=ZERO)        # same kills: farm is halved
    h_sp = resolve_turn(CONTACT, ch(0, 0), 2, H(), H(), CFG, u=ZERO)
    assert t_sp.lane.kills == h_sp.lane.kills and t_sp.farm[0] == pytest.approx(CFG.trade_farm_factor * h_sp.farm[0])
    miss = resolve_turn(CONTACT, ch(0, 0), 2, TR(), H(), CFG, u=ONE)
    assert miss.trade_damage == (0, 0) and miss.champ == ch(0, 0)


def test_evaded_or_unreachable_trade_deals_nothing_and_farms():
    for lane, b, champ in ((CONTACT, Y(), ch(0, 0)), (CONTACT, S(), ch(0, -1)), (SIEGE_A, H(), ch(-T, -T))):
        t = resolve_turn(lane, champ, 2, TR(), b, CFG, u=ZERO)               # last: a dive 4 cells away is out of reach
        assert not t.connected[0] and t.hit_p == (0.0, 0.0) and t.trade_damage == (0, 0) and t.damage_taken[1] == 0
        assert t.modes[0] == "farm"


def test_trade_against_trade_is_simultaneous():
    t = resolve_turn(CONTACT, ch(0, 0, hpa=2, hpb=2), 2, TR(), TR(), CFG, u=ZERO)
    assert t.connected == (True, True) and t.trade_damage == (2, 2)
    assert t.champ.a.hp == t.champ.b.hp == 1                              # M2c: nonlethal floor


def test_partial_escape_and_interrupted_recall_use_their_behavior_terms():
    # a Yield champion pinned at its tower cell by a dive cannot escape: the reduced Yield chance applies
    t = resolve_turn(SIEGE_A, ch(1, -T), 2, TR(), Y(), CFG, u=ZERO)
    assert t.connected[0] and t.threatened[1]
    assert t.hit_p[0] == pytest.approx(0.55 - 0.25 - CFG.trade_hit_tower + CFG.trade_hit_minions * 0)
    t = resolve_turn(CONTACT, ch(0, -1), 2, TR(), RECALL, CFG, u=ZERO)
    assert t.interrupted[1] and t.modes[1] == CHANNEL and t.hit_p[0] == pytest.approx(0.55)


def test_dive_takes_a_tower_shot_unless_minions_hold_the_tower():
    """Only a dive reaches a champion in its own tower cell; the diver takes the tower shot with probability
    tower_shot_p, reduced to tower_shot_p_busy while its own crashed minions are under the tower."""
    t = resolve_turn(DIVE, ch(1, -T), 2, TR(), H(), CFG, u=ZERO)
    assert t.end == (T, -T) and t.connected[0] and t.tower_p[0] == CFG.tower_shot_p
    assert t.tower_damage == (CFG.tower_shot_damage, 0) and t.damage_taken[0] == CFG.tower_shot_damage
    assert t.hit_p[0] == pytest.approx(0.55 - CFG.trade_hit_tower)       # the tower also protects the defender
    t = resolve_turn(DIVE_BUSY, ch(1, -T), 2, TR(), H(), CFG, u=ZERO)
    assert t.end == (T, -T) and t.tower_p[0] == CFG.tower_shot_p_busy
    u = list(ZERO)
    u[U_TOWER[0]] = (CFG.tower_shot_p_busy + 1) / 2                      # between busy and free probabilities
    assert resolve_turn(DIVE_BUSY, ch(1, -T), 2, TR(), H(), CFG, u=u).tower_damage[0] == 0
    assert resolve_turn(DIVE, ch(1, -T), 2, TR(), H(), CFG, u=u).tower_damage[0] == CFG.tower_shot_damage
    # no tower shot without a dive, and none for a non-trader at the enemy tower (siege objective)
    t = resolve_turn(CONTACT, ch(0, 0), 2, TR(), H(), CFG, u=ZERO)
    assert t.tower_p == (0.0, 0.0) and t.tower_damage == (0, 0)
    t = resolve_turn(SIEGE_A, ch(0, -T), 2, champ_act(SP, B_HOLD, 1), S(), CFG, u=ZERO)
    assert t.end[0] == T and t.tower_p == (0.0, 0.0)


def test_damage_distribution_matches_the_formula():
    """10,000 seeded environment steps of Trade vs Hold from one state: the realized 0/1/2 frequencies agree
    with (1 - h, h (1 - q), h q) within 4.5 standard errors (strong attacker against a weaker defender)."""
    cfg = CFG.with_(skill_a=0.7, skill_b=0.4)
    env = LaneEnv(cfg)
    n = 10_000
    counts = np.zeros(3)
    for i in range(n):
        env.reset(seed=i, root=CONTACT, k0=2, champ=ch(0, 0))
        r = env.step(TR(), H())
        counts[r.info["trade_damage"][0]] += 1
    h, q = r.info["hit_p"][0], r.info["double_p"][0]
    assert h == pytest.approx(0.55 + 0.4 * 0.3) and q == pytest.approx(0.3 + 0.2 * 0.3)
    p = np.array(trade_damage_distribution(h, q))
    se = np.sqrt(p * (1 - p) / n)
    assert np.all(np.abs(counts / n - p) < 4.5 * se), (counts / n, p)


# ------------------------------------------------------------------ invariants
def _random_play(cfg, episodes, seed):
    env = LaneEnv(cfg)
    rng = np.random.default_rng(seed)
    for e in range(episodes):
        oa, ob = env.reset(seed=1000 + e)
        while True:
            lane, champ, k = env.state, env.champ, env.k
            a, b = oa.legal[rng.integers(len(oa.legal))], ob.legal[rng.integers(len(ob.legal))]
            r = env.step(a, b)
            yield lane, champ, k, a, b, r
            if r.done:
                break
            oa, ob = r.obs


def test_resolution_is_mirror_equivariant_with_mirrored_chance():
    asym = CFG.with_(skill_a=0.8, skill_b=0.3)
    rng = np.random.default_rng(0)
    n = 0
    for lane, champ, k, a, b, r in _random_play(asym, 6, 1):
        u = tuple(rng.random(N_CHANCE))
        t = resolve_turn(lane, champ, k, a, b, asym, u=u)
        m = resolve_turn(lane.mirror(), champ.mirror(), k, b, a, swap_seats(asym), u=mirror_chance(u))
        assert m.champ == t.champ.mirror() and m.trade_damage == t.trade_damage[::-1]
        assert m.tower_damage == t.tower_damage[::-1] and m.ganked == t.ganked[::-1]
        assert np.allclose(m.hit_p, t.hit_p[::-1]) and np.allclose(m.double_p, t.double_p[::-1])
        n += t.connected[0] or t.connected[1]
    assert n > 10


def test_no_trade_damage_is_guaranteed_and_damage_needs_a_connected_trade():
    for lane, champ, k, a, b, r in _random_play(CFG, 8, 2):
        i = r.info
        for s in (0, 1):
            assert (i["hit_p"][s] > 0) == i["trade_connected"][s]
            assert i["hit_p"][s] <= CFG.trade_hit_max < 1                 # a miss is always possible
            if i["trade_connected"][s]:
                assert i["modes"][s] == TRADING and i["champion_reach"][s] < -10   # no wave contribution
            assert i["trade_damage"][s] in ((0, 1, 2) if i["trade_connected"][s] else (0,))
            assert i["damage_taken"][s] == i["trade_damage"][1 - s] + i["tower_damage"][s]
            assert 1 <= i["champ_after"][s][1] <= CFG.champion_hp


def test_chance_stream_is_reproducible_and_action_independent():
    """Same seed, same trajectory; the chance vectors of a turn are the same whatever the agents play (common
    random numbers)."""
    from lanegame.env import champion_agents
    agents = champion_agents(CFG)
    env = LaneEnv(CFG)
    l1 = play_episode(env, agents["trade"], agents["hold"], seed=11, record=True)
    l2 = play_episode(env, agents["trade"], agents["hold"], seed=11, record=True)
    l3 = play_episode(env, agents["safe"], agents["shove_trade"], seed=11, record=True)
    assert l1.rewards == l2.rewards and l1.trade_damage_dealt == l2.trade_damage_dealt
    assert [t["chance"] for t in l1.turns] == [t["chance"] for t in l3.turns]
    assert l1.trade_damage_dealt[0] > 0
