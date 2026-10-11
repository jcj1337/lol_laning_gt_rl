"""Farming economy and resource accounting (revised model).

The accounting identity behind every test: each HP of minions is created by a spawn and
leaves the lane exactly once - killed in a fight, killed by a tower while sieging, or (only
under the original "overflow" cap rule) deleted by the HP cap. Each killed HP is one farm
opportunity worth farm_per_hp for the killer's side, collected with an efficiency in [0, 1].
"""

import numpy as np
import pytest

from lanegame import CONTEST, SAFE, SH_C, SH_S, SP_C, SP_S, GameConfig, LaneGame
from lanegame.baselines import uniform
from lanegame.game import collection_efficiency, turn_farm
from lanegame.mechanics import step_turn, total_hp
from lanegame.simulate import play_episode
from lanegame.state import State

from conftest import all_transitions


def test_hp_conservation_on_every_transition(rev_game, rev_layers):
    """HP alive before + HP spawned = HP alive after + killed in fights + killed by towers + overflow."""
    cfg = rev_game.cfg
    for k, s, a, b in all_transitions(rev_game, rev_layers):
        r = step_turn(s, k, a, b, cfg)
        before, after = total_hp(s, cfg), total_hp(r.state, cfg)
        ev = r.events
        for side in (0, 1):
            assert before[side] + ev["spawned_hp"][side] == (after[side] + ev["killed_hp"][side]
                                                             + ev["tower_hp"][side] + ev["overflow_hp"][side])
            assert ev["overflow_hp"][side] == 0           # the "block" cap rule never deletes minions


def test_every_removed_hp_is_one_farm_opportunity_for_the_opponent(rev_game, rev_layers):
    cfg = rev_game.cfg
    for k, s, a, b in all_transitions(rev_game, rev_layers):
        r = step_turn(s, k, a, b, cfg)
        avail, coll = turn_farm(r, a, b, cfg)
        for side in (0, 1):
            enemy = 1 - side
            removed = r.events["killed_hp"][enemy] + r.events["tower_hp"][enemy]
            assert sum(r.kills[side].values()) == removed
            assert abs(avail[side] - cfg.farm_per_hp * removed) < 1e-12
            assert -1e-12 <= coll[side] <= avail[side] + 1e-12


def test_tower_kills_are_farmed_under_the_defenders_tower(rev_game, rev_layers):
    cfg = rev_game.cfg
    seen = 0
    for k, s, a, b in all_transitions(rev_game, rev_layers):
        r = step_turn(s, k, a, b, cfg)
        for side in (0, 1):
            d = r.events["tower_hp"][side]               # side's crashed minions killed by the enemy tower
            if d:
                seen += 1
                assert r.kills[1 - side].get(-cfg.tower, 0) >= d
    assert seen > 0


def test_siege_mass_accounting(rev_game, rev_layers):
    """Crashed mass under a tower changes only by new crashes (+HP) and tower fire (-HP)."""
    cfg = rev_game.cfg
    for k, s, a, b in all_transitions(rev_game, rev_layers):
        r = step_turn(s, k, a, b, cfg)
        for side, (sg0, sg1) in enumerate(((s.asg, r.state.asg), (s.bsg, r.state.bsg))):
            assert sg1 == sg0 + r.crash_hp[side] - r.events["tower_hp"][side] - r.events["siege_killed_hp"][side]


def test_merging_creates_no_farm(rev_cfg):
    """A reinforcement joining its own wave changes nothing economically."""
    cfg = rev_cfg
    W = cfg.wave_hp
    s = State(0, W, (-2,), 0, -4, 0, (), 0, 0, 0)        # A wave parked at the river, reinforcement 2 behind
    r = step_turn(s, 3, SP_C, SP_S, cfg)
    assert r.events["merged"][0] == 1 and r.state.ah == 2 * W
    assert r.kills == ({}, {}) and turn_farm(r, SP_C, SP_S, cfg) == ((0.0, 0.0), (0.0, 0.0))


def test_transitions_without_kills_pay_no_farm(rev_game, rev_layers):
    """No action, position or merge pays farm by itself: without a kill there is no farm."""
    quiet = 0
    for k, s, a, b in all_transitions(rev_game, rev_layers):
        tr = rev_game.transition(s, k, a, b)
        r = step_turn(s, k, a, b, rev_game.cfg)
        if sum(r.events["killed_hp"]) + sum(r.events["tower_hp"]) == 0:
            assert tr.farm == (0.0, 0.0) and tr.farm_payoff == 0.0
            quiet += 1
    assert quiet > 0


def test_capped_reinforcement_queues_instead_of_vanishing(rev_cfg):
    cfg = rev_cfg
    H = cfg.hp_cap
    s = State(0, H, (-2,), 0, -4, 0, (), 0, 0, 0)        # full stack at the river; reinforcement arrives
    r = step_turn(s, 3, SP_C, SP_S, cfg)
    assert r.events["overflow_hp"] == [0, 0] and r.events["merged"][0] == 0
    assert r.state.ah == H and r.state.ar[0] == -1        # waits directly behind the full stack
    assert total_hp(r.state, cfg)[0] == total_hp(s, cfg)[0] + r.events["spawned_hp"][0]


def test_no_minion_is_farmed_twice_over_whole_games(rev_cfg):
    """Over complete games, the HP each side farms from never exceeds the HP the opponent ever spawned."""
    cfg = rev_cfg.with_(horizon=32)
    g = LaneGame(cfg)
    rng = np.random.default_rng(5)
    pol = uniform()
    for _ in range(20):
        s = g.initial_state()
        killed = [0, 0]
        spawned = [cfg.wave_hp, cfg.wave_hp]
        for k in range(cfg.horizon):
            la, lb = g.legal(s, 0), g.legal(s, 1)
            a, b = la[rng.integers(len(la))], lb[rng.integers(len(lb))]
            r = step_turn(s, k, a, b, cfg)
            for side in (0, 1):
                killed[side] += sum(r.kills[side].values())
                spawned[side] += r.events["spawned_hp"][side]
            s = g.transition(s, k, a, b).pre_state
        for side in (0, 1):
            alive = total_hp(s, cfg)[1 - side]
            assert killed[side] + alive == spawned[1 - side]


def test_collection_efficiency_properties(rev_cfg):
    cfg = rev_cfg
    T = cfg.tower
    xs = range(-T, T + 1)
    for opp in (CONTEST, SAFE):
        for x in xs:
            cc = collection_efficiency(x, CONTEST, opp, cfg)
            cs = collection_efficiency(x, SAFE, opp, cfg)
            assert 0.0 <= cs <= cc <= 1.0                          # Contest never collects less than Safe
        safe = [collection_efficiency(x, SAFE, opp, cfg) for x in xs]
        assert all(safe[i] >= safe[i + 1] for i in range(len(safe) - 1))   # Safe collects less further forward
    for x in xs:
        for mine in (CONTEST, SAFE):                               # an opponent who contests zones you
            assert collection_efficiency(x, mine, CONTEST, cfg) <= collection_efficiency(x, mine, SAFE, cfg)
    # the Contest/Safe gap is state dependent: small at your own tower, large far forward
    gap = [collection_efficiency(x, CONTEST, SAFE, cfg) - collection_efficiency(x, SAFE, SAFE, cfg) for x in xs]
    assert gap[0] < 0.15 and gap[-2] > 0.5
    # farming under your own tower: most of the farm, whatever the postures (worst case: Safe while the
    # attacker stays to zone, 0.8 * 0.85 = 0.68)
    for mine in (CONTEST, SAFE):
        for opp in (CONTEST, SAFE):
            assert collection_efficiency(-T, mine, opp, cfg) >= 0.6


def test_crash_rewards_are_separate_from_under_tower_farm(rev_cfg):
    """A crash gives the crasher a reward menu and the defender farm opportunities; neither replaces the other."""
    cfg = rev_cfg
    H = cfg.hp_cap
    g = LaneGame(cfg)
    s = State(1, H, (), 0, 0, 0, (), 0, 0, 0)            # A's 3-wave packet one cell from B's empty tower
    tr = g.transition(s, 2, SH_C, SP_C)
    assert tr.crash_hp == (H, 0) and len(tr.options_a) == 2
    assert tr.farm_available[1] > 0                       # B already farms the first tower kill this turn
    # the crash menu is the same whatever B does: B's farming does not reduce A's crash reward
    tr2 = g.transition(s, 2, SH_C, SP_S)
    assert tr2.options_a == tr.options_a and tr2.farm_available == tr.farm_available


def test_farm_denial_is_not_double_counted(rev_game, rev_layers):
    """Denied farm (available - collected) appears in nobody's payoff: the only farm term is
    collected_A - collected_B."""
    for k, s, a, b in all_transitions(rev_game, rev_layers):
        tr = rev_game.transition(s, k, a, b)
        assert tr.payoff == (tr.farm[0] - tr.farm[1]) + tr.gank_payoff


def test_farming_disabled_gives_zero_farm(rev_cfg):
    g = LaneGame(rev_cfg.with_(farm_per_hp=0.0, horizon=8))
    for k, s, a, b in all_transitions(g, g.reachable()):
        assert g.transition(s, k, a, b).farm == (0.0, 0.0)


def test_accounting_with_defenders_engaging_crashed_minions(rev_cfg):
    """The siege_engage variant (defender minions fight the crashed mass) keeps every identity."""
    cfg = rev_cfg.with_(siege_engage=True, horizon=14)
    g = LaneGame(cfg)
    fights = 0
    for k, s, a, b in all_transitions(g, g.reachable()):
        r = step_turn(s, k, a, b, cfg)
        before, after = total_hp(s, cfg), total_hp(r.state, cfg)
        for side in (0, 1):
            ev = r.events
            assert before[side] + ev["spawned_hp"][side] == after[side] + ev["killed_hp"][side] + ev["tower_hp"][side]
            assert sum(r.kills[1 - side].values()) == ev["killed_hp"][side] + ev["tower_hp"][side]
            fights += ev["siege_killed_hp"][side]
    assert fights > 0
