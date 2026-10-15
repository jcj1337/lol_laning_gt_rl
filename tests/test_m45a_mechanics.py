"""M4.5a unified positional mechanics (GameConfig.m45_unified(); docs/m4/m4_5/m4_5a_unified_mechanics_spec.md).

Deterministic scenario tests of every new rule (the movement graph and the Vision node W, Ward placement and its
interruption, per-tick minion retaliation and positional aggro with exclusive targeting, minion damage scaling, the
capped wave budget of a connected trader, the reach ball, XP range 2) and integration tests (mirror equivariance,
zero-sum rewards, masks, the observation and its decoder, PettingZoo, seeded replay, the checkpoint guard, the
historical presets).
"""

import hashlib
import json

import numpy as np
import pytest

from lanegame.champions import (N_CHANCE, U_TRADE_HIT, champion_legal_actions, champion_gank_probability,
                                in_contact, mirror_chance)
from lanegame.champions_m4 import (FARM, M4State, check_invariants, collection_units, context, initial_m4,
                                   resolve_turn_m4)
from lanegame.config import (B_HOLD, B_SAFE, B_TRADE, B_YIELD, CHAMP_ABSENT, CHAMP_WARD, F, PRESSURE, SH, SP,
                             VISION, GameConfig, champ_act)
from lanegame.env import LaneEnv
from lanegame.env.observation import feature_names
from lanegame.env.observation_m4 import decode_m4
from lanegame.mechanics import step_turn, total_hp
from lanegame.positional import WARD, dist, in_contact as graph_contact, minion_damage_centi, move
from lanegame.state import State

CFG = GameConfig.m45_unified()
T = CFG.tower
HOLD, TRADE, SAFE, YIELD = (champ_act(SP, b) for b in (B_HOLD, B_TRADE, B_SAFE, B_YIELD))
SHOVE_TRADE, THIN_TRADE, SHOVE_SAFE = champ_act(SH, B_TRADE), champ_act(F, B_TRADE), champ_act(SH, B_SAFE)
PRESS = champ_act(SP, B_HOLD, PRESSURE)


def lane(ax, ah, bx, bh, ar=(), br=(), asg=0, bsg=0, va=0, vb=0):
    return State(ax, ah, tuple(ar), asg, bx, bh, tuple(br), bsg, va, vb, 0, 0)


def champs(pa, pb, level=1, hp=None, cfg=CFG, absent_b=0):
    t = context(cfg).t
    m = initial_m4(cfg)
    xp, H = t.xp_cum[level], t.hp_max[level]
    return M4State(m.a._replace(pos=pa, xp=xp, hp=H if hp is None else hp),
                   m.b._replace(pos=pb if not absent_b else -T, xp=xp, hp=H, absent=absent_b))


def chance(hit=()):
    """A chance vector in which exactly the trades of the sides in `hit` land (1 base HP) and nothing else happens."""
    u = [0.999] * N_CHANCE
    for s in hit:
        u[U_TRADE_HIT[s]] = 0.0
    return u


def resolve(lane_, m4, a, b, u=None, cfg=CFG, k=9, record=False):
    def lane_step(s, kk, x, y, reach, budget=None, targets=None):
        return step_turn(s, kk, x, y, cfg, record=record, champion_reach=reach, champion_budget=budget,
                         champion_targets=targets)
    return resolve_turn_m4(lane_, m4, k, a, b, cfg, lane_step=lane_step, u=u)


# ---------------------------------------------------------------------------------------------- configuration
def test_preset_changes_exactly_its_fields_and_validates():
    CFG.validate()
    base = GameConfig.m4_four_actions_econ_aggro()
    a, b = base.to_dict(), CFG.to_dict()
    assert sorted(k for k in a if a[k] != b[k]) == sorted(GameConfig.M45_UNIFIED_FIELDS)
    assert CFG.m45_mechanics and CFG.ward_node and CFG.minion_targeting and CFG.xp_range == 2
    assert (CFG.minion_range, CFG.minion_retaliation_delay, CFG.minion_damage_alpha, CFG.minion_damage_growth,
            CFG.trade_wave_fraction, CFG.ward_exposure, CFG.vision_turns, CFG.vision_reduction) == \
        (1, 1, 50, 1.08, 0.5, 0, 6, 0.75)


@pytest.mark.parametrize("name", ["m4_four_actions_econ_aggro", "m4_four_actions_econ", "m4_four_actions", "m4_lasthit",
                                  "m4_lane", "m4_natural", "m4_emergent", "m4", "champions"])
def test_historical_presets_keep_every_m45_field_off(name):
    cfg = getattr(GameConfig, name)()
    assert not cfg.m45_mechanics and cfg.vision_rule == "siege" and cfg.wave_reach_rule == "line"
    assert cfg.trade_wave_rule == "none" and not cfg.minion_targeting


@pytest.mark.parametrize("kw", [dict(vision_rule="ward"), dict(minion_retaliation=True), dict(wave_reach_rule="ball"),
                                dict(minion_damage_alpha=10)])
def test_legacy_rulesets_reject_the_m45_fields(kw):
    with pytest.raises(ValueError):
        GameConfig.champions(**kw).validate()


@pytest.mark.parametrize("kw", [dict(vision_rule="bush"), dict(ward_exposure=3), dict(minion_retaliation_delay=3),
                                dict(minion_damage_growth=0.9), dict(trade_wave_fraction=1.5),
                                dict(minion_retaliation=True, trade_exchanges=2)])
def test_m45_validation(kw):
    with pytest.raises(ValueError):
        GameConfig.m45_unified(**kw).validate()
    with pytest.raises(ValueError):                    # scripted waves refuse every M4.5a rule
        GameConfig.m4(vision_rule="ward").validate()
    with pytest.raises(ValueError):                    # parameters of an inactive rule are refused
        GameConfig.m4_four_actions_econ_aggro(minion_range=2).validate()


def test_spaces_labels_and_feature_names():
    env = LaneEnv(CFG)
    assert env.n_actions == 61 and env.obs_size == 90
    assert LaneEnv(GameConfig.m4_four_actions_econ_aggro()).obs_size == 88
    names = feature_names(CFG)
    assert "own_champ_W" in names and "enemy_champ_W" in names and len(names) == 90
    assert CFG.label(CHAMP_WARD) == "WARD" and GameConfig.m4_four_actions_econ_aggro().label(CHAMP_WARD) == "LH-H+V"


# ---------------------------------------------------------------------------------------------- the movement graph
def test_graph_distances_and_mirror_invariance():
    assert [dist(x, WARD) for x in range(-2, 3)] == [3, 2, 1, 2, 3] and dist(WARD, WARD) == 0
    assert dist(-3, WARD) == 4 and dist(1, -1) == 2
    for x in range(-2, 3):                             # W is the same node in both sides' coordinates
        assert dist(x, WARD) == dist(-x, WARD)


@pytest.mark.parametrize("c,target,end", [(-1, WARD, WARD), (0, WARD, WARD), (1, WARD, WARD), (-2, WARD, 0),
                                          (2, WARD, 0), (WARD, -2, -1), (WARD, 2, 1), (WARD, 0, 0), (WARD, WARD, WARD),
                                          (-2, 2, 0), (1, -2, -1)])
def test_movement_passes_through_the_lane_cell_zero(c, target, end):
    assert move(c, target, 2, T) == end
    assert move(c, target, 1, T) == (target if dist(c, target) <= 1 else
                                     (0 if c == WARD else (0 if target == WARD and abs(c) == 1 else
                                                           c + (1 if (0 if target == WARD else target) > c else -1))))


def test_contact_through_w_and_the_own_tower_rule():
    assert graph_contact(WARD, WARD, T, 1) and graph_contact(WARD, 0, T, 1) and graph_contact(0, WARD, T, 1)
    assert not graph_contact(WARD, 1, T, 1) and not graph_contact(WARD, -1, T, 1)
    assert not graph_contact(WARD, -2, T, 1) and not graph_contact(-2, WARD, T, 1)
    for x in range(-2, 3):                             # lane cells: identical to the historical rule
        for y in range(-2, 3):
            hist = (x + y == 0) if (x == -T or y == -T) else abs(x + y) <= 1
            assert in_contact(x, y, CFG) == hist == graph_contact(x, y, T, 1)


# ---------------------------------------------------------------------------------------------- legality
def test_ward_legality_by_position_and_the_other_codes():
    L = lane(0, 8, 0, 8)
    for pos, legal in ((-2, False), (-1, True), (0, True), (1, True), (2, False), (WARD, True)):
        acts = champion_legal_actions(L, champs(pos, 0), 0, CFG)
        assert (CHAMP_WARD in acts) == legal, pos
        assert not any(a in acts for a in (31, 32, 36, 37, 38) + tuple(range(56, 61)))
        assert (champ_act(SP, B_HOLD, PRESSURE) in acts) is False          # no siege: no Pressure
        if pos == WARD:
            assert champ_act(SP, 4) not in acts                            # no Recall from W
    absent = champs(0, 0)._replace(a=champs(0, 0).a._replace(absent=1, pos=-T))
    assert champion_legal_actions(L, absent, 0, CFG) == (CHAMP_ABSENT,)
    # Pressure needs the enemy tower within reach: never from W; the old siege Vision codes never come back
    siege = lane(-2, 8, 0, 8, asg=8)
    assert PRESS in champion_legal_actions(siege, champs(0, 0), 0, CFG)
    assert PRESS not in champion_legal_actions(siege, champs(WARD, 0), 0, CFG)
    assert not any(a in champion_legal_actions(siege, champs(0, 0), 0, CFG) for a in (31, 32, 36, 37, 38))
    # the historical preset keeps its siege Vision objective
    hist = GameConfig.m4_four_actions_econ_aggro()
    assert champ_act(SP, B_HOLD, VISION) in champion_legal_actions(siege, champs(0, 0, cfg=hist), 0, hist)


def test_both_players_access_w_symmetrically():
    rng = np.random.default_rng(3)
    for _ in range(40):
        pa, pb = (int(x) for x in rng.choice([-2, -1, 0, 1, 2, WARD], 2))
        L = lane(int(rng.integers(-2, 1)), 8, int(rng.integers(-2, 1)), 8)
        m = champs(pa, pb)
        assert champion_legal_actions(L, m, 0, CFG) == champion_legal_actions(L.mirror(), m.mirror(), 1, CFG)


# ---------------------------------------------------------------------------------------------- Ward
NEUTRAL = lane(0, 8, 0, 8, ar=(-3,), br=(-3,))


def test_ward_against_a_passive_opponent_places_vision_and_costs_the_turn():
    t = resolve(NEUTRAL, champs(0, 0), CHAMP_WARD, HOLD)
    assert t.end == (WARD, 0) and t.modes[0] == "ward" and t.warding == (True, False)
    assert t.ward_placed == (True, False) and t.ward_interrupted == (False, False)
    assert t.lane.state.va == CFG.vision_turns and t.lane.state.vb == 0 and t.lane.vision_set == (True, False)
    assert t.effective_objectives[0] == VISION
    assert t.farm[0] == 0.0 and t.lane.events["champion_hp"][1] == 0   # no gold, no wave damage while warding
    assert t.xp_gain[0] > 0                                              # XP by graph distance from W
    assert t.champ.a.pos == WARD
    check_invariants(t.champ, CFG)
    # vision protects the following turns (occurrence x 0.25), not the ward turn
    nxt = t.lane.state
    p_vis = champion_gank_probability(nxt, 0, SP, B_HOLD, CFG, 0)
    p_none = champion_gank_probability(nxt._replace(va=0), 0, SP, B_HOLD, CFG, 0)
    assert p_vis == pytest.approx(0.25 * p_none) and p_none > 0


def test_ward_is_interrupted_only_by_positive_enemy_trade_damage():
    hit = resolve(NEUTRAL, champs(0, 0), CHAMP_WARD, TRADE, u=chance(hit=(1,)))
    assert hit.connected == (False, True) and hit.trade_damage[1] > 0
    assert hit.ward_placed == (False, False) and hit.ward_interrupted == (True, False)
    assert hit.lane.state.va == 0 and hit.effective_objectives[0] != VISION
    miss = resolve(NEUTRAL, champs(0, 0), CHAMP_WARD, TRADE, u=chance())
    assert miss.connected == (False, True) and miss.trade_damage == (0, 0)
    assert miss.ward_placed == (True, False) and miss.lane.state.va == CFG.vision_turns


def test_ward_timers_are_independent_and_both_may_ward():
    both = resolve(NEUTRAL, champs(0, 0), CHAMP_WARD, CHAMP_WARD)
    assert both.end == (WARD, WARD) and both.ward_placed == (True, True)
    assert (both.lane.state.va, both.lane.state.vb) == (6, 6)
    # A wards while B's earlier vision keeps counting down on its own
    t = resolve(lane(0, 8, 0, 8, vb=4), champs(0, 0), CHAMP_WARD, HOLD)
    assert (t.lane.state.va, t.lane.state.vb) == (6, 3)


def test_a_warder_dying_places_no_ward():
    t = resolve(NEUTRAL, champs(0, 0, hp=50), CHAMP_WARD, TRADE, u=chance(hit=(1,)))
    assert t.died[0] and not t.ward_placed[0] and t.bounty[1] > 0


def test_returning_from_w_needs_movement():
    m = champs(WARD, 0)
    t = resolve(lane(-2, 8, 0, 8), m, SAFE, HOLD)
    assert t.end[0] == -1                                # W -> 0 -> -1
    t = resolve(lane(-2, 8, 2, 8), champs(WARD, -2), HOLD, HOLD)
    assert t.end[0] in (-1, 0, 1) and t.end[0] != WARD
    t = resolve(lane(2, 8, -2, 8), champs(WARD, 0), HOLD, HOLD)     # own front far ahead: only one lane cell
    assert t.end[0] == 1


def test_farming_and_xp_reach_from_w():
    for x, coll in ((0, True), (1, False), (-1, False), (2, False)):
        assert (collection_units(x, FARM, WARD, FARM, 0, CFG) > 0) == coll
    # XP: kills at -1 .. 1 are within 2 of W, kills at +-2 are not
    assert [dist(x, WARD) <= CFG.xp_range for x in (-2, -1, 0, 1, 2)] == [False, True, True, True, False]


# ---------------------------------------------------------------------------------------------- minion targeting
BIG = lane(-1, 8, 0, 24)                                  # A's front at -1, B's 24-HP front at the lane cell


def test_successful_trade_triggers_retaliation_from_the_next_tick():
    t = resolve(BIG, champs(-1, 0, level=2), TRADE, HOLD, u=chance(hit=(0,)), record=True)
    assert t.connected[0] and t.trade_damage[0] > 0 and t.retaliation == (False, True)
    assert t.minion_fire[0] == ("", "retaliation")       # tick 0: minions on minions; tick 1: on the trader
    assert t.minion_hits[0] > 0 and t.minion_hits[1] == 0
    assert t.minion_damage[0] == minion_damage_centi(t.minion_hits[0], 2, 50, 1.08)
    assert t.damage_taken[0] == context(CFG).mitigate(t.minion_damage[0], 0)
    # the retaliating blob fires its tick-1 firepower at A and none at A's minions
    tick1 = t.lane.trace[1]
    assert tick1["minion_fire"][0][0] == "retaliation" and tick1["minion_fire"][0][1] == t.minion_hits[0]


def test_retaliation_delay_zero_fires_from_the_exchange_tick():
    cfg = GameConfig.m45_unified(minion_retaliation_delay=0)
    t = resolve(BIG, champs(-1, 0, level=2, cfg=cfg), TRADE, HOLD, u=chance(hit=(0,)), cfg=cfg)
    assert t.minion_fire[0] == ("retaliation", "retaliation")


@pytest.mark.parametrize("u", [chance(), None])
def test_missed_or_zero_damage_trades_trigger_no_retaliation(u):
    t = resolve(BIG, champs(-1, 0, level=2), TRADE, HOLD, u=u)
    assert t.connected[0] and t.trade_damage[0] == 0
    assert t.retaliation == (False, False) and t.minion_hits == (0, 0) and t.minion_fire[0] == ("", "")


def test_mutual_successful_trades_retaliate_on_both_sides():
    L = lane(-1, 16, -1, 16)
    t = resolve(L, champs(-1, -1, level=2), TRADE, TRADE, u=chance(hit=(0, 1)))
    assert t.connected == (True, True) and t.retaliation == (True, True)
    assert t.minion_fire == (("", "retaliation"), ("", "retaliation")) and t.minion_hits[0] == t.minion_hits[1] > 0


RECALL = champ_act(SP, 4)


def test_retaliation_needs_the_trader_within_minion_range():
    # B dives into A's tower cell and interrupts A's Recall with a landing hit. The fronts fight at the lane cell, two
    # cells from the diver: A's minions cannot reach it ...
    t = resolve(lane(0, 8, 0, 8), champs(-2, 0), RECALL, TRADE, u=chance(hit=(1,)))
    assert t.interrupted[0] and t.connected[1] and t.end[1] == 2 and t.retaliation == (True, False)
    assert dist(0, -t.end[1]) > CFG.minion_range and t.minion_hits[1] == 0 and t.minion_fire[1] == ("", "")
    # ... while a front one cell from it (A's 16-HP front at its -1) turns on the diver from tick 1. On tick 0 the diver
    # stands in that front's cell on its way (its own front is behind, at its 0): positional aggro, although it moves
    t = resolve(lane(-1, 16, 0, 8), champs(-2, 0), RECALL, TRADE, u=chance(hit=(1,)))
    assert t.path[1] == (1, 2)
    assert t.connected[1] and t.minion_fire[1] == ("positional", "retaliation") and t.minion_hits[1] > 0


def test_a_blob_without_tick_firepower_does_not_fire():
    # a 5-HP front has firepower 1 per turn, all of it on tick 0: retaliation (tick 1) has nothing to fire
    t = resolve(lane(-1, 8, 0, 5), champs(-1, 0, level=2), TRADE, HOLD, u=chance(hit=(0,)))
    assert t.retaliation == (False, True) and t.minion_hits == (0, 0) and t.minion_fire[0] == ("", "")


POS = lane(-1, 8, 0, 8)                    # the fronts fight at A's -1 / the lane cell; B's front stays at the lane cell


def test_positional_aggro_on_a_champion_ahead_of_its_wave():
    # A stands in B's front's cell (its own front behind, at -1) and Trades a retreating B, who evades out of contact
    t = resolve(POS, champs(0, -2), TRADE, SAFE)
    assert t.end[0] == 0 and not t.connected[0]
    assert t.minion_fire[0][0] == "positional" and t.minion_hits[0] > 0 and t.retaliation == (False, False)
    assert t.damage_taken[0] == t.minion_damage[0] > 0


def test_positional_aggro_ends_when_the_champion_falls_back_or_its_wave_covers_it():
    t = resolve(POS, champs(0, -2), HOLD, HOLD)           # Hold walks back to its wave on tick 0
    assert t.path[0] == (-1, -1) and t.minion_hits == (0, 0)
    t = resolve(lane(0, 8, 0, 8), champs(0, -2), TRADE, SAFE)   # own front in the same cell takes the attacks
    assert t.minion_hits[0] == 0


def test_a_champion_following_its_wave_is_not_positionally_targeted():
    """A Hold champion walks with its wave tick by tick (champion_movement_rule "tick"): it stays at -1 while its
    front fights there on tick 0 and steps to 0 with it on tick 1, after B's front has died, so it never stands in the
    enemy front's cell ahead of its own."""
    L = lane(-1, 16, 0, 2)                                 # A's 16-HP front kills B's 2-HP front and walks to 0
    t = resolve(L, champs(-1, -2), HOLD, HOLD, k=10)
    assert t.start[0] == -1 and t.end[0] == 0 and t.lane.state.ax == 0 and t.path[0] == (-1, 0)
    assert t.minion_hits[0] == 0 and t.minion_fire[0] == ("", "")


def test_positional_aggro_needs_occupancy_not_adjacency():
    # A stands one cell from B's front, which waits (A's crashed mass is under B's tower): never targeted
    t = resolve(lane(-2, 8, -1, 8, asg=8), champs(0, -1), TRADE, TRADE, u=chance(), record=True)
    assert t.path[0] == (0, 0) and t.minion_fire[0] == ("", "") and t.minion_hits[0] == 0
    # A has no wave and stands at 0; B's front one cell away walks into A's cell during tick 0 and on past it during
    # tick 1 (k = 10: no spawn, so A's farm spot stays its own cell): targeted on tick 0 only, when A occupies its cell
    t = resolve(lane(0, 0, -1, 16), champs(0, -2), HOLD, HOLD, record=True, k=10)
    assert t.end[0] == 0 and t.minion_fire[0] == ("positional", "")
    assert t.lane.trace[0]["minion_fire"][0][0] == "positional" and t.lane.trace[1]["minion_fire"][0] == ["", 0]


def test_own_crashed_mass_covers_a_pressuring_champion():
    # A dives B in B's tower cell, where B's front stands (in contact with A's front at +1); the dive misses
    dive = champs(2, -2)
    covered = resolve(lane(1, 8, -2, 8, asg=8), dive, TRADE, HOLD, u=chance(), k=10)
    assert covered.path[0] == (2, 2) and covered.connected[0] and covered.minion_hits[0] == 0
    exposed = resolve(lane(1, 8, -2, 8, asg=0), dive, TRADE, HOLD, u=chance(), k=10)
    assert exposed.path[0] == (2, 2) and exposed.minion_fire[0][0] == "positional" and exposed.minion_hits[0] > 0
    assert exposed.retaliation == (False, False)


def test_targeting_redirects_firepower_without_creating_or_duplicating_damage():
    """On every tick, B's wave spends its firepower f_t either on A's champion or on A's minions: with targeting the
    units on the champion equal what the same blob dealt to A's front without targeting, and A's front takes none."""
    L = lane(-1, 8, 0, 8)                                 # fronts in contact at -1 / 0
    # A's Trade (it misses) keeps A in B's front's cell at 0, ahead of its own front: positional aggro
    on = resolve(L, champs(0, 0), TRADE, HOLD, u=chance(), record=True)
    off_cfg = GameConfig.m45_unified(minion_retaliation=False, minion_positional_aggro=False)
    off = resolve(L, champs(0, 0, cfg=off_cfg), TRADE, HOLD, u=chance(), cfg=off_cfg, record=True)
    assert on.end == off.end == (0, 0) and on.retaliation == (False, False)
    fire = [tick["minion_fire"][0] for tick in on.lane.trace]
    assert fire[0][0] == "positional"
    f0_on, f0_off = on.lane.trace[0]["fight"], off.lane.trace[0]["fight"]
    assert f0_on is not None and f0_off is not None
    # damage on A's front: B's champion's last hit only, versus that plus B's minions; the difference is exactly the
    # units B's minions spent on A's champion instead
    assert fire[0][1] > 0 and f0_off[4] - f0_on[4] == fire[0][1]
    assert off.minion_hits == (0, 0) and off.minion_fire is None
    # minion HP is conserved either way (champion hits are not minion deaths)
    for t in (on, off):
        ev = t.lane.events
        before, after = total_hp(L, CFG), total_hp(t.lane.state, CFG)
        for s in (0, 1):
            assert before[s] + ev["spawned_hp"][s] == after[s] + ev["killed_hp"][s] + ev["tower_hp"][s]
    assert on.lane.state.ah > off.lane.state.ah                  # aggro preserved A's minions


def test_minion_damage_scaling_with_level_and_investment():
    vals = [minion_damage_centi(2, L, 50, 1.08) for L in range(1, 19)]
    assert vals[0] == 100 and all(a >= b for a, b in zip(vals, vals[1:])) and min(vals) >= 1
    assert minion_damage_centi(1, 18, 50, 1.08) >= 1 and minion_damage_centi(0, 1, 50, 1.08) == 0
    t = context(CFG).t
    rel = [minion_damage_centi(2, L, 50, 1.08) / t.hp_max[L] for L in (1, 6, 11, 18)]
    assert rel[0] > 0.15 and rel[-1] < 0.025 and rel == sorted(rel, reverse=True)
    # invested gold divides the minion damage with the turn's other damage (invest_durability)
    ctx = context(CFG)
    inv = 128 * CFG.gold_scale
    assert ctx.mitigate(100, inv) < ctx.mitigate(100, 0) == 100
    # stronger waves hit harder: units grow with blob HP (the half-wave law)
    hits = [resolve(lane(-1, 8, 0, h), champs(-1, 0, level=2), TRADE, HOLD, u=chance(hit=(0,))).minion_hits[0]
            for h in (8, 16, 24)]
    assert hits == sorted(hits) and hits[0] < hits[-1]


def test_minion_damage_can_kill_and_pays_the_bounty():
    t = resolve(POS, champs(0, -2, hp=30), TRADE, SAFE)
    assert t.minion_hits[0] > 0 and t.died[0] and t.bounty[1] > 0 and t.champ.a.absent > 0


# ---------------------------------------------------------------------------------------------- wave management
def test_connected_trader_keeps_a_capped_wave_budget():
    L = lane(-1, 8, 0, 8)
    sh = resolve(L, champs(-1, 0), SHOVE_TRADE, HOLD, u=chance())
    C = context(CFG).capacity(1, 0)
    assert sh.connected[0] and sh.trade_wave_capped[0] and sh.wave_budget[0] == (C + 1) // 2 < C
    assert sh.lane.events["champion_hp"][1] > 0                      # it still removes enemy minions
    tn = resolve(L, champs(-1, 0), THIN_TRADE, HOLD, u=chance())
    assert tn.connected[0] and not tn.trade_wave_capped[0]
    hist = GameConfig.m4_four_actions_econ_aggro()
    old = resolve(L, champs(-1, 0, cfg=hist), SHOVE_TRADE, HOLD, u=chance(), cfg=hist)
    assert old.connected[0] and old.lane.events["champion_hp"][1] == 0     # historically: no wave damage


def test_reach_ball_limits_wave_damage_to_one_cell():
    L = lane(0, 0, 1, 8)                                   # A has no wave; B's front at A's -1; A stands at +1
    ball = resolve(L, champs(1, -2), champ_act(SH, B_HOLD), HOLD, k=10)
    assert ball.end[0] == 1 and ball.reach[0] == (0, 2) and ball.lane.events["champion_hp"][1] == 0
    line_cfg = GameConfig.m45_unified(wave_reach_rule="line")
    line = resolve(L, champs(1, -2, cfg=line_cfg), champ_act(SH, B_HOLD), HOLD, cfg=line_cfg, k=10)
    assert line.reach[0] == 2 and line.lane.events["champion_hp"][1] > 0


def test_shove_safe_cannot_clear_minions_out_of_reach():
    # the fronts fight at the lane cell (they do not move): a Safe champion at -1 cannot reach B's front ...
    far = resolve(lane(0, 8, 0, 8), champs(-1, 0), SHOVE_SAFE, HOLD)
    assert far.end[0] == -1 and far.reach[0] == (-2, -1) and far.lane.events["champion_hp"][1] == 0
    # ... but reaches it once the fronts fight at its own -1
    near = resolve(lane(-1, 8, 1, 8), champs(-1, 0), SHOVE_SAFE, HOLD)
    assert near.lane.events["champion_hp"][1] > 0


def test_safe_evades_with_room_and_pays_in_farm_not_xp():
    t = resolve(lane(0, 8, 0, 8), champs(0, 0), SAFE, TRADE, u=chance(hit=(1,)))
    assert t.threatened[0] and not t.connected[1] and t.damage_taken[0] == 0
    hold = resolve(lane(0, 8, 0, 8), champs(0, 0), HOLD, HOLD)
    assert t.farm[0] < hold.farm[0] and t.xp_gain[0] > 0


def test_thin_while_trading_maintains_a_freeze_through_the_wave():
    """Scenario A in miniature: the freezer (A, at its tower) Thins while it trades the walking-up B; B's held front is
    at A's -1. The connected freezer still damages minions, and a landing hit turns B's wave onto it on tick 1."""
    L = lane(-2, 8, 1, 13)
    t = resolve(L, champs(-2, 1, level=10), THIN_TRADE, champ_act(SH, B_HOLD), u=chance(hit=(0,)))
    assert t.end[0] == -1 and t.connected[0] and t.trade_damage[0] > 0
    assert t.lane.events["champion_hp"][1] > 0 and t.retaliation == (False, True)
    assert t.minion_fire[0][1] == "retaliation" and t.minion_hits[0] > 0


# ---------------------------------------------------------------------------------------------- integration
def _random_states(n, seed, cfg=CFG):
    rng = np.random.default_rng(seed)
    out = []
    env = LaneEnv(cfg, max_turns=10 ** 6)
    env.reset(seed=seed)
    while len(out) < n:
        a, b = int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1)))
        out.append((env.state, env.champ, env.k, env.legal(0), env.legal(1)))
        env.step(a, b)
    return out, rng


def test_resolver_is_mirror_equivariant_on_random_states():
    states, rng = _random_states(400, 11)
    for lane_, m4, k, la, lb in states:
        a, b = int(rng.choice(la)), int(rng.choice(lb))
        u = tuple(float(x) for x in rng.random(N_CHANCE))
        t = resolve_turn_m4(lane_, m4, k, a, b, CFG, u=u)
        tm = resolve_turn_m4(lane_.mirror(), m4.mirror(), k, b, a, CFG, u=mirror_chance(u))
        assert tm.champ == t.champ.mirror() and tm.lane.state == t.lane.state.mirror()
        assert tm.reward_units == -t.reward_units and tm.end == t.end[::-1]
        assert tm.minion_hits == t.minion_hits[::-1] and tm.minion_damage == t.minion_damage[::-1]
        assert tm.ward_placed == t.ward_placed[::-1] and tm.retaliation == t.retaliation[::-1]
        assert tm.minion_fire == (t.minion_fire[::-1] if t.minion_fire else None)


def test_env_soak_invariants_zero_sum_masks_and_observation_decoding():
    env = LaneEnv(CFG, max_turns=400)
    rng = np.random.default_rng(5)
    seen = {"ward": 0, "placed": 0, "interrupted": 0, "retaliation": 0, "positional": 0, "at_w": 0}
    for g in range(6):
        env.reset(seed=g)
        while not env.done:
            ma, mb = env.masks()
            assert set(np.flatnonzero(ma)) == set(env.legal(0)) and set(np.flatnonzero(mb)) == set(env.legal(1))
            a, b = int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1)))
            r = env.step(a, b)
            assert r.rewards[1] == -r.rewards[0]
            check_invariants(env.champ, CFG)
            i = r.info
            seen["ward"] += sum(i["ward"])
            seen["placed"] += sum(i["ward_placed"])
            seen["interrupted"] += sum(i["ward_interrupted"])
            seen["retaliation"] += sum(c == "retaliation" for side in i["minion_fire"] for c in side)
            seen["positional"] += sum(c == "positional" for side in i["minion_fire"] for c in side)
            for s in (0, 1):
                seen["at_w"] += env.champ.side(s).pos == WARD
                ph, ln, m4, _ = decode_m4(r.obs[s].vector, CFG)
                me, en = env.champ.side(s), env.champ.side(1 - s)
                assert (m4.a.pos, m4.a.hp, m4.b.pos, m4.b.hp) == (me.pos, me.hp, en.pos, en.hp)
                assert ln == (env.state if s == 0 else env.state.mirror())._replace(ap=0, bp=0)
    assert all(v > 0 for v in seen.values()), seen


def test_seeded_replay_is_deterministic():
    def run():
        env = LaneEnv(CFG, max_turns=150, record_ticks=True)
        env.reset(seed=77)
        rng = np.random.default_rng(77)
        h = hashlib.sha256()
        while not env.done:
            r = env.step(int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1))))
            h.update(json.dumps(r.info, sort_keys=True, default=repr).encode())
        return h.hexdigest()
    assert run() == run()


def test_lane_step_cache_does_not_change_results():
    rng = np.random.default_rng(9)
    plain, cached = LaneEnv(CFG, max_turns=200), LaneEnv(CFG, max_turns=200, cache_size=4096)
    plain.reset(seed=9)
    cached.reset(seed=9)
    while not plain.done:
        a, b = int(rng.choice(plain.legal(0))), int(rng.choice(plain.legal(1)))
        r1, r2 = plain.step(a, b), cached.step(a, b)
        assert r1.info["state_after"] == r2.info["state_after"] and r1.info["minion_hits"] == r2.info["minion_hits"]
        assert r1.info["champ_after"] == r2.info["champ_after"]


def test_pettingzoo_parallel_api_on_the_m45_preset():
    pytest.importorskip("pettingzoo")
    pytest.importorskip("gymnasium")
    from pettingzoo.test import parallel_api_test
    from lanegame.env.pettingzoo_adapter import LaneParallelEnv
    env = LaneParallelEnv(CFG, max_turns=50)
    parallel_api_test(env, num_cycles=40)
    obs, _ = env.reset(seed=1)
    assert obs["A"]["observation"].shape == (90,) and obs["A"]["action_mask"].shape == (61,)


def test_checkpoint_guard_refuses_88_input_networks_and_names_the_preset(tmp_path):
    torch = pytest.importorskip("torch")
    from lanegame.experiments.rl.m44.common import load_policy, preset_name
    from lanegame.rl import PolicyValueNet, save_checkpoint
    assert preset_name(CFG) == "m45_unified"
    old = GameConfig.m4_four_actions_econ_aggro()
    torch.manual_seed(0)
    path = str(tmp_path / "old.pt")
    save_checkpoint(path, PolicyValueNet(88, 61, (8,)), old, {"name": "old"})
    with pytest.raises(ValueError, match="90"):
        load_policy(path, CFG)
    with pytest.raises(ValueError):
        load_policy(path, CFG, cross_env=True)          # a shape mismatch is refused even across environments
    load_policy(path, old)                              # and the checkpoint still loads on its own preset


@pytest.mark.parametrize("name,digest", [
    ("m4_four_actions_econ_aggro", "11006e5b84f5d6ed"),
    ("m4_lasthit", "e5cb610ca11083ab"),
])
def test_historical_presets_are_pinned(name, digest):
    """Random games on historical presets reproduce the pre-M4.5a engine exactly (digests recorded at d9f92df, where
    the M4.5a engine was also checked byte-for-byte against the committed one on 12 champion presets)."""
    cfg = getattr(GameConfig, name)()
    env = LaneEnv(cfg, max_turns=200, record_ticks=True)
    h = hashlib.sha256()
    for g in range(3):
        env.reset(seed=500 + g)
        rng = np.random.default_rng(g)
        while not env.done:
            r = env.step(int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1))))
            assert "minion_hits" not in r.info and all("minion_fire" not in tk for tk in r.info["lane_trace"])
            h.update(json.dumps(r.info, sort_keys=True, default=repr).encode())
    assert h.hexdigest()[:16] == digest


# ---------------------------------------------------------------------------------------------- viewer and demonstrations
def test_viewer_helpers_and_w_rendering():
    from lanegame.viewer import render
    from lanegame.viewer import session as ss
    assert ss.cell_text(WARD) == "W" and ss.cell_text(None) == "W" and ss.cell_text(-1) == "-1"
    assert ss.reach_text((-1, 1)) == "-1..+1" and ss.reach_text(-10 ** 6) == "none" and ss.reach_text(2) == "<= +2"
    rec = ss.play(ss.Setup("m45_unified", "neutral", "equal", 0), prefix=[(CHAMP_WARD, TRADE)], turns=0)
    view = ss.frame_view(rec, 0, rec.cfg.ticks_per_turn + 1)
    assert view["ward_node"] and view["champions"]["A"]["at_ward"] and view["champions"]["A"]["pos_global"] is None
    assert view["ward"]["A"] in ("placed", "interrupted") and view["minion_fire"] is not None
    render.lane_figure(view, rec.cfg)
    assert any("ward" in e for e in ss.turn_events(rec.turns[0].info, rec.cfg))
    assert ss.action_name(rec.cfg, CHAMP_WARD) == "Ward (Vision at W)"


@pytest.mark.parametrize("name", ["m45a_E2_ward_interrupted", "m45a_C1_bigwave_level2"])
def test_demonstrations_show_their_event_and_replay_exactly(name, tmp_path):
    from lanegame.experiments.m45a import demos
    from lanegame.experiments.rl.m44.trace import verify
    demo = next(d for d in demos.DEMOS if d.name == name)
    r = demos.export(demo, str(tmp_path))
    v = verify(r["path"])
    assert v["actions_equal"] and v["payoff_equal"]
    s = r["summary"]
    if name == "m45a_E2_ward_interrupted":
        assert s["A"]["wards_interrupted"] >= 1
    else:
        assert s["A"]["retaliation_ticks_taken"] >= 1 and s["A"]["minion_centi_taken"] > 0
