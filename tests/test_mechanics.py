"""State and transition correctness of the wave mechanics."""

import itertools

import numpy as np
import pytest

from lanegame import F, SH, SP, GameConfig, LaneGame
from lanegame.game import apply_choice, crash_options, gank_probability
from lanegame.mechanics import initial_state, legal_actions, step_turn, total_packets
from lanegame.scenarios import all_scenarios, run_scenario
from lanegame.state import State

from conftest import all_transitions


def test_initial_state(cfg):
    s = initial_state(cfg)
    assert s.ax == -cfg.base and s.bx == -cfg.base and s.ah == s.bh == cfg.wave_hp
    assert s.asg == s.bsg == s.va == s.vb == 0


def test_spawn_and_travel_timing(cfg):
    """Fresh waves spawned at tick 0 meet at the midpoint at tick 4, when the next pair spawns."""
    s = initial_state(cfg)
    r0 = step_turn(s, 0, SH, SH, cfg, record=True)
    assert [p[0] for p in r0.trace[0]["A"]] == [-3] and [p[0] for p in r0.trace[1]["A"]] == [-2]
    assert r0.state.ar == () and r0.state.br == ()                 # no spawn at tick 2
    r1 = step_turn(r0.state, 1, SH, SH, cfg, record=True)
    t4 = r1.trace[1]
    assert t4["fight"] is not None and t4["fight"][0] == 0 and -t4["fight"][2] == 0   # contact at x = 0, tick 4
    assert r1.events["spawned"] == [1, 1]                          # spawn at tick 4
    assert -cfg.base in (r1.state.ar + ((r1.state.ax,) if r1.state.ah else ()))


def test_spawns_only_every_spawn_interval(game, layers):
    cfg = game.cfg
    for k, s, a, b in itertools.islice(all_transitions(game, layers), 4000):
        r = step_turn(s, k, a, b, cfg)
        expect = 1 if ((k + 1) * cfg.ticks_per_turn) % cfg.spawn_interval == 0 else 0
        assert r.events["spawned"] == [expect, expect]


def test_transitions_are_deterministic(game, layers):
    cfg = game.cfg
    for k, s, a, b in itertools.islice(all_transitions(game, layers), 3000):
        r1 = step_turn(s, k, a, b, cfg, record=True)
        r2 = step_turn(s, k, a, b, cfg, record=True)
        assert r1.state == r2.state and r1.crash_hp == r2.crash_hp and r1.trace == r2.trace


def test_packet_conservation(game, layers):
    """Packets appear only by spawning and disappear only by merging, dying or tower consumption."""
    cfg = game.cfg
    for k, s, a, b in all_transitions(game, layers):
        r = step_turn(s, k, a, b, cfg)
        before = total_packets(s)
        after = total_packets(r.state)
        ev = r.events
        for side in (0, 1):
            assert after[side] == (before[side] + ev["spawned"][side] - ev["merged"][side]
                                   - ev["died"][side] - ev["consumed"][side])


def test_no_passing_through_and_bounds(game, layers):
    """Opposing packets never cross; positions, HP and queues stay within their domains."""
    cfg = game.cfg
    for k, s, a, b in all_transitions(game, layers):
        r = step_turn(s, k, a, b, cfg, record=True)
        for tr in r.trace:
            A = [p[0] for p in tr["A"]]                  # global = own for A
            B = [-p[0] for p in tr["B"]]                 # global
            if A and B:
                assert max(A) <= min(B), (s, k, a, b, tr)
            for p in tr["A"] + tr["B"]:
                assert 1 <= p[1] <= cfg.hp_cap
                assert -cfg.base <= p[0] <= cfg.tower   # never beyond the enemy tower cell
            assert min(tr["siege"]) >= 0
        st = r.state
        for side in (0, 1):
            fx, fh, rears, sg, v = st.side(side)
            assert all(x < fx for x in rears)            # rears are strictly behind the front
            assert 0 <= v <= cfg.vision_turns


def test_states_use_integer_keys_only(layers):
    for layer in layers:
        for s in layer:
            for f in s:
                vals = f if isinstance(f, tuple) else (f,)
                assert all(type(v) is int for v in vals)


def test_tower_consumes_one_packet_per_turn(cfg):
    s = State(-4, cfg.wave_hp, (), 2, -4, cfg.wave_hp, (), 0, 0, 0)
    r1 = step_turn(s, 2, SP, SP, cfg)
    r2 = step_turn(r1.state, 3, SP, SP, cfg)
    assert (r1.state.asg, r2.state.asg) == (1, 0)
    assert r1.events["consumed"] == [1, 0] and r2.events["consumed"] == [1, 0]


def test_packets_arriving_during_a_turn_are_not_consumed_that_turn(cfg):
    s = State(1, 2 * cfg.wave_hp, (), 0, -4, cfg.wave_hp, (), 0, 0, 0)   # A medium wave one cell from B tower
    r = step_turn(s, 2, SH, SP, cfg)
    assert r.crash_hp[0] > 0 and r.state.asg == 1 and r.events["consumed"] == [0, 0]


def test_crash_rewards_cannot_be_duplicated(game, layers):
    """A crash reward requires a packet to newly enter the siege queue; queued packets never pay again."""
    cfg = game.cfg
    for k, s, a, b in all_transitions(game, layers):
        r = step_turn(s, k, a, b, cfg)
        for side in (0, 1):
            assert (r.crash_hp[side] > 0) == (r.events["crashed"][side] > 0)
            assert r.events["crashed"][side] <= 1 or r.crash_hp[side] > 0
    # Two packets waiting under the tower and nothing else moving in: no crash reward.
    s = State(-4, cfg.wave_hp, (), 2, -3, cfg.wave_hp, (), 0, 0, 0)
    r = step_turn(s, 2, SP, SP, cfg)
    assert r.crash_hp == (0, 0)
    assert crash_options(0, cfg) == ((0.0, False),)


def test_vision_expires_after_two_turns(cfg):
    s = initial_state(cfg)
    s, r = apply_choice(s, (0.0, True), (0.0, False), cfg)
    assert (s.va, s.vb) == (cfg.vision_turns, 0) and r == 0.0
    seen = []
    for k in range(4):
        seen.append(s.va)
        s = step_turn(s, k, SP, SP, cfg).state
    assert seen == [2, 1, 0, 0]


def test_vision_reduces_gank_probability(cfg):
    s = State(0, 2, (), 0, -4, 2, (), 0, 0, 0)
    for a in (SP, SH):
        p0 = gank_probability(s, 0, a, cfg, exposure=2)
        p1 = gank_probability(s._replace(va=1), 0, a, cfg, exposure=2)
        assert abs(p1 - p0 * (1 - cfg.vision_reduction)) < 1e-15


def test_gank_probability_monotonicity(cfg):
    s = State(0, 2, (), 0, -4, 2, (), 0, 0, 0)
    p = [gank_probability(s, 0, SP, cfg, e) for e in range(-cfg.tower, cfg.tower + 1)]
    assert all(p[i] < p[i + 1] for i in range(len(p) - 1))                 # deeper = riskier
    assert gank_probability(s, 0, SH, cfg, 0) > gank_probability(s, 0, SP, cfg, 0) > \
        gank_probability(s, 0, F, cfg, 0)                                   # aggression = riskier
    big = s._replace(ah=3 * cfg.wave_hp)
    assert gank_probability(big, 0, SP, cfg, 0) < gank_probability(s, 0, SP, cfg, 0)  # bigger wave = safer
    for e in range(-cfg.tower, cfg.tower + 1):
        for a in (SP, SH, F):
            assert 0.0 <= gank_probability(s, 0, a, cfg, e) <= cfg.gank_max


def test_freeze_legality(cfg):
    # B's wave on A's half: A may freeze, B may not
    s = State(-4, 2, (), 0, 1, 2, (), 0, 0, 0)          # B front at global -1
    assert F in legal_actions(s, 0, cfg) and F not in legal_actions(s, 1, cfg)
    # A's wave deep in B's half: A may not freeze
    s = State(1, 2, (), 0, -3, 2, (), 0, 0, 0)
    assert F not in legal_actions(s, 0, cfg)
    # no enemy wave on the lane: freeze is meaningless and illegal
    s = State(-2, 2, (), 0, 0, 0, (), 0, 0, 0)
    assert legal_actions(s, 0, cfg) == (SP, SH)


def test_illegal_actions_are_rejected(game):
    s = State(1, 2, (), 0, -3, 2, (), 0, 0, 0)
    with pytest.raises(ValueError):
        game.transition(s, 1, F, SP)


def test_reinforcement_arrival_emerges_from_movement(cfg):
    """A wave parked closer to B's base is contacted by B's next wave sooner."""
    def contact_tick(ax):
        s = State(ax, 2 * cfg.wave_hp, (), 0, -4, cfg.wave_hp, (), 0, 0, 0)   # B wave just spawned at +4
        tick = 0
        for k in range(2, 6):
            r = step_turn(s, k, SP, SP, cfg, record=True)
            for tr in r.trace:
                tick += 1
                if tr["fight"] is not None:
                    return tick
            s = r.state
        return None
    times = {ax: contact_tick(ax) for ax in (0, 1)}      # Slow Push parks at the river or beyond
    assert times[1] < times[0]


@pytest.mark.parametrize("sc", all_scenarios(GameConfig()), ids=lambda sc: sc.name)
def test_scenarios(sc):
    ok, detail, lines, results = run_scenario(sc, GameConfig())
    assert ok, detail + "\n" + "\n".join(lines)
