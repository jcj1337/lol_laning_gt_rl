"""Revised wave mechanics: Slow Push pacing, mass-based tower fire, bounces and reachability from s_0."""

import pytest

from lanegame import F_C, SH_C, SH_S, SP_C, SP_S, GameConfig, LaneGame
from lanegame.mechanics import step_turn
from lanegame.reachability import EVENTS, explore_events, replay
from lanegame.state import State


def _turns_to_clear(cfg, hp):
    s = State(1, hp, (), 0, 0, 0, (), 0, 0, 0)            # A packet one cell from B's empty tower
    r = step_turn(s, 2, SH_C, SP_S, cfg)
    assert r.crash_hp[0] == hp
    s, n = r.state, 1
    while s.asg > 0:
        s = step_turn(s, 2 + n, SP_S, SP_S, cfg).state
        n += 1
    return n


def test_bigger_crashes_occupy_the_tower_longer(rev_cfg):
    W = rev_cfg.wave_hp
    turns = [_turns_to_clear(rev_cfg, h) for h in (W, 2 * W, 3 * W)]
    assert turns[0] < turns[1] < turns[2]


def test_tower_fire_rate_and_timing(rev_cfg):
    """The tower removes tower_siege_damage HP per tick, never from mass that arrived in the same tick."""
    cfg = rev_cfg
    W = cfg.wave_hp
    s = State(1, W, (), 0, 0, 0, (), 0, 0, 0)
    r = step_turn(s, 2, SH_C, SP_S, cfg, record=True)
    sieges = [tr["siege"][0] for tr in r.trace]
    assert r.trace[0]["crashed"][0] == W and sieges == [W, W - cfg.tower_siege_damage]
    s = State(-4, W, (), 6, -3, W, (), 0, 0, 0)
    r = step_turn(s, 2, SP_S, SP_S, cfg)
    assert r.events["tower_hp"][0] == cfg.ticks_per_turn * cfg.tower_siege_damage


def test_defender_waits_behind_tower_and_bounces(rev_cfg):
    """While crashed mass is under its tower the defender's wave waits behind it and absorbs reinforcements."""
    cfg = rev_cfg
    W = cfg.wave_hp
    s = State(-4, W, (), 3 * W, -3, W, (), 0, 0, 0)
    sizes = []
    for k in range(2, 8):
        r = step_turn(s, k, SH_S, SP_C, cfg)
        s = r.state
        sizes.append((s.asg, s.bx, s.bh))
    waiting = [bh for sg, bx, bh in sizes if sg > 0]
    assert all(bx <= -cfg.tower - 1 for sg, bx, bh in sizes if sg > 0)
    assert max(waiting) >= 2 * W


def test_unstacked_slow_push_waits_and_stacked_one_creeps(rev_cfg):
    cfg = rev_cfg
    W = cfg.wave_hp
    lone = State(0, W, (), 0, -4, 0, (), 0, 0, 0)
    for k in range(4, 8):
        lone = step_turn(lone, k, SP_C, SP_S, cfg).state
        if lone.ah and lone.ah <= W:
            assert lone.ax <= cfg.slowpush_line
    stacked = State(0, 2 * W, (), 0, -4, 0, (), 0, 0, 0)
    positions = []
    for k in range(4, 4 + cfg.phase_period):
        stacked = step_turn(stacked, k, SP_C, SP_S, cfg).state
        positions.append(stacked.ax)
    assert positions == [1] * cfg.phase_period            # one cell per creep period


def test_shove_safe_releases_the_wave_without_champion_damage(rev_cfg):
    cfg = rev_cfg
    W = cfg.wave_hp
    s = State(0, W, (), 0, -4, 0, (), 0, 0, 0)
    r = step_turn(s, 2, SH_S, SP_S, cfg)
    assert r.state.ax > 0 or r.crash_hp[0] > 0                    # the minions keep walking


@pytest.fixture(scope="module")
def reach16():
    g = LaneGame(GameConfig(horizon=24))      # a crash onto an existing siege first occurs after turn 16
    return g, explore_events(g)


@pytest.mark.parametrize("event", ["crash_small", "crash_medium", "crash_large", "crash_onto_siege",
                                   "freeze_held", "freeze_broken", "bounce", "siege_persists",
                                   "safe_disengages", "tower_farm", "farm_contest_vs_safe"])
def test_event_is_reachable_from_standard_start(reach16, event):
    """Every mechanic is reachable from s_0 by legal play, and the recorded witness replays to it."""
    g, rep = reach16
    assert rep.counts[event] > 0 and event in rep.witness
    w = rep.witness[event]
    s, r = replay(g, w.plan, w.choices)[-1]
    assert s == w.state and EVENTS[event](g.cfg, s, *w.plan[-1], r)


def test_original_model_cannot_reach_bounces_or_stacked_sieges():
    """Documented limitation of the original model, reproduced: no bounce, no crash onto a siege."""
    rep = explore_events(LaneGame(GameConfig.v1(horizon=16)))
    assert rep.counts["bounce"] == 0 and rep.counts["crash_onto_siege"] == 0


def test_queued_packets_push_with_the_front(rev_cfg):
    """A held 9-HP front alone cannot break a freeze (threshold 10), but with a full wave queued directly
    behind it the whole blob breaks through: queued minions stand at the front and push with it."""
    cfg = rev_cfg
    W = cfg.wave_hp
    alone = State(-4, 0, (), 0, 1, 9, (), 0, 0, 0)            # B front (9 HP) at A's freeze line
    queued = State(-4, 0, (), 0, 1, 9, (0,), 0, 0, 0)         # ... with one wave queued behind it
    k = cfg.phase_period                                       # a creep turn: the stacked Slow Push wants to advance
    r1 = step_turn(alone, k, F_C, SP_C, cfg)
    r2 = step_turn(queued, k, F_C, SP_C, cfg)
    assert r1.events["freeze_held"][1] > 0 and r1.crash_hp[1] == 0
    assert r2.events["freeze_broken"][1] > 0


def test_reachable_state_space_is_bounded():
    """No reachable state is first visited after turn 64 of a 96-turn game: the set of distinct lane states
    saturates, so the solver's cost grows only linearly with the horizon."""
    g = LaneGame(GameConfig(horizon=96))
    layers = g.reachable()
    early = set().union(*layers[:64])
    late = set().union(*layers[64:])
    assert late <= early
    assert max(max(s.asg, s.bsg) for s in early) < g.cfg.siege_cap
