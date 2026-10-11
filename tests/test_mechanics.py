"""State and transition correctness of the wave mechanics.

Generic tests run on both models; tests marked v1 pin down rules of the original model
that the revised model deliberately changed (packet-count tower consumption, the original
Freeze legality rule, the original gank action terms).
"""

import itertools

import numpy as np
import pytest

from lanegame import F, SH, SP, SH_C, SP_C, GameConfig, LaneGame
from lanegame.game import apply_choice, crash_options, gank_probability
from lanegame.mechanics import initial_state, lane_packets, legal_actions, step_turn, total_packets
from lanegame.scenarios import all_scenarios, run_scenario
from lanegame.state import State

from conftest import all_transitions


# ------------------------------------------------------------------ both models
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
        assert r1.kills == r2.kills


def test_packet_conservation(game, layers):
    """Packets appear only by spawning and disappear only by merging, dying, crashing or tower consumption."""
    cfg = game.cfg
    for k, s, a, b in all_transitions(game, layers):
        r = step_turn(s, k, a, b, cfg)
        ev = r.events
        if cfg.tower_mode == "packet":           # crashed packets stay countable in the queue
            before, after = total_packets(s), total_packets(r.state)
            for side in (0, 1):
                assert after[side] == (before[side] + ev["spawned"][side] - ev["merged"][side]
                                       - ev["died"][side] - ev["consumed"][side])
        else:                                    # crashed packets become HP mass under the tower
            before, after = lane_packets(s), lane_packets(r.state)
            for side in (0, 1):
                assert after[side] == (before[side] + ev["spawned"][side] - ev["merged"][side]
                                       - ev["died"][side] - ev["crashed"][side])


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
            assert sg <= (cfg.siege_cap if cfg.tower_mode == "mass" else 10 ** 6)


def test_states_use_integer_keys_only(layers):
    for layer in layers:
        for s in layer:
            for f in s:
                vals = f if isinstance(f, tuple) else (f,)
                assert all(type(v) is int for v in vals)


def test_crash_events_match_crash_rewards(game, layers):
    """A crash reward requires a packet to newly enter the siege; crashed minions never pay again."""
    cfg = game.cfg
    for k, s, a, b in all_transitions(game, layers):
        r = step_turn(s, k, a, b, cfg)
        for side in (0, 1):
            assert (r.crash_hp[side] > 0) == (r.events["crashed"][side] > 0)
            assert r.crash_hp[side] == r.events["crashed_hp"][side]
            tr = game.transition(s, k, a, b)
            opts = tr.options_a if side == 0 else tr.options_b
            assert (len(opts) > 1) == (r.crash_hp[side] > 0)
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
    W = cfg.wave_hp
    s = State(0, W, (), 0, -4, W, (), 0, 0, 0)
    for a in cfg.action_codes():
        p0 = gank_probability(s, 0, a, cfg, exposure=2)
        p1 = gank_probability(s._replace(va=1), 0, a, cfg, exposure=2)
        assert abs(p1 - p0 * (1 - cfg.vision_reduction)) < 1e-15


def test_gank_probability_bounds_and_position(cfg):
    W = cfg.wave_hp
    s = State(0, W, (), 0, -4, W, (), 0, 0, 0)
    p = [gank_probability(s, 0, SP, cfg, e) for e in range(-cfg.tower, cfg.tower + 1)]
    assert all(p[i] < p[i + 1] for i in range(len(p) - 1))                 # deeper = riskier
    big = s._replace(ah=3 * W)
    assert gank_probability(big, 0, SP, cfg, 0) < gank_probability(s, 0, SP, cfg, 0)  # bigger wave = safer
    for e in range(-cfg.tower, cfg.tower + 1):
        for a in cfg.action_codes():
            assert 0.0 <= gank_probability(s, 0, a, cfg, e) <= cfg.gank_max


def test_illegal_actions_are_rejected(game):
    W = game.cfg.wave_hp
    s = State(1, W, (), 0, -3, W, (), 0, 0, 0)          # B's wave deep in B's half: A may not freeze
    with pytest.raises(ValueError):
        game.transition(s, 1, F, SP)


def test_reinforcement_arrival_emerges_from_movement(cfg):
    """A wave further forward is contacted by the enemy's next wave sooner."""
    W = cfg.wave_hp

    def contact_tick(ax):
        s = State(ax, 2 * W, (), 0, -4, W, (), 0, 0, 0)   # B wave just spawned at +4
        tick = 0
        for k in range(4, 8):
            r = step_turn(s, k, SP, SP, cfg, record=True)
            for tr in r.trace:
                tick += 1
                if tr["fight"] is not None:
                    return tick
            s = r.state
        return None
    times = {ax: contact_tick(ax) for ax in (0, 1)}
    assert times[1] < times[0]


_SCENARIO_CASES = [("v1", GameConfig.v1(), sc) for sc in all_scenarios(GameConfig.v1())] + \
                  [("revised", GameConfig(), sc) for sc in all_scenarios(GameConfig())]


@pytest.mark.parametrize("name,cfg,sc", _SCENARIO_CASES, ids=[f"{m}-{sc.name}" for m, _, sc in _SCENARIO_CASES])
def test_scenarios(name, cfg, sc):
    ok, detail, lines, results = run_scenario(sc, cfg)
    assert ok, detail + "\n" + "\n".join(lines)


def test_scenario_start_reachability_flags():
    """Scenarios flagged as reachable start from a state reachable from s_0 in the same dynamics phase;
    the others start from constructed states that are not."""
    cfg = GameConfig(horizon=16)
    layers = LaneGame(cfg).reachable()
    P = cfg.phase_period
    for sc in all_scenarios(cfg):
        reach = any(sc.start in set(layers[k]) for k in range(len(layers)) if k % P == sc.k0 % P)
        assert reach == sc.start_reachable, sc.name


# ------------------------------------------------------------------ original model only
def test_v1_tower_consumes_one_packet_per_turn(v1_cfg):
    cfg = v1_cfg
    s = State(-4, cfg.wave_hp, (), 2, -4, cfg.wave_hp, (), 0, 0, 0)
    r1 = step_turn(s, 2, SP, SP, cfg)
    r2 = step_turn(r1.state, 3, SP, SP, cfg)
    assert (r1.state.asg, r2.state.asg) == (1, 0)
    assert r1.events["consumed"] == [1, 0] and r2.events["consumed"] == [1, 0]


def test_v1_packets_arriving_during_a_turn_are_not_consumed_that_turn(v1_cfg):
    cfg = v1_cfg
    s = State(1, 2 * cfg.wave_hp, (), 0, -4, cfg.wave_hp, (), 0, 0, 0)   # A medium wave one cell from B tower
    r = step_turn(s, 2, SH, SP, cfg)
    assert r.crash_hp[0] > 0 and r.state.asg == 1 and r.events["consumed"] == [0, 0]


def test_v1_queued_packets_do_not_pay_again(v1_cfg):
    cfg = v1_cfg
    s = State(-4, cfg.wave_hp, (), 2, -3, cfg.wave_hp, (), 0, 0, 0)
    r = step_turn(s, 2, SP, SP, cfg)
    assert r.crash_hp == (0, 0)


def test_v1_gank_action_terms(v1_cfg):
    cfg = v1_cfg
    s = State(0, 2, (), 0, -4, 2, (), 0, 0, 0)
    assert gank_probability(s, 0, SH, cfg, 0) > gank_probability(s, 0, SP, cfg, 0) > \
        gank_probability(s, 0, F, cfg, 0)                                   # aggression = riskier


def test_v1_freeze_legality(v1_cfg):
    cfg = v1_cfg
    s = State(-4, 2, (), 0, 1, 2, (), 0, 0, 0)          # B front at global -1
    assert F in legal_actions(s, 0, cfg) and F not in legal_actions(s, 1, cfg)
    s = State(1, 2, (), 0, -3, 2, (), 0, 0, 0)          # A's wave deep in B's half: A may not freeze
    assert F not in legal_actions(s, 0, cfg)
    s = State(-2, 2, (), 0, 0, 0, (), 0, 0, 0)          # no enemy wave on the lane
    assert legal_actions(s, 0, cfg) == (SP, SH)
    s = State(-4, 2, (), 0, 0, 2, (), 0, 0, 0)          # enemy wave at the river: legal in v1
    assert F in legal_actions(s, 0, cfg)


def test_v1_reproduces_original_solution():
    """GameConfig.v1() is the original Phase 1 model: same state space, same equilibrium, same exploitability."""
    from lanegame.baselines import v1_baselines
    from lanegame.best_response import exploitability
    from lanegame.solver import solve_equilibrium
    g = LaneGame(GameConfig.v1(horizon=32))
    sol = solve_equilibrium(g)
    assert sol.stats.states == 905 and sol.stats.mixed_states == 52
    assert len({s for l in sol.layers for s in l}) == 71
    assert abs(sol.value) < 1e-12
    expected = {"uniform": 11.860403464012775, "always_shove": 19.45, "always_slowpush": 4.85,
                "defensive_freeze": 1.85, "stack_then_shove": 4.85}
    for b in v1_baselines():
        assert abs(exploitability(g, sol.layers, b, 0, sol.value).exploitability - expected[b.name]) < 1e-9
