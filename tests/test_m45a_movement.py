"""M4.5a movement correction (champion_movement_rule "tick"; docs/m4/m4_5/m4_5a_unified_mechanics_report.md §10).

Champions walk tick by tick to their unchanged end cell; Hold, an unthreatened Yield and Safe follow their wave's front
of each tick; positional aggro is evaluated on each tick's positions after its movement, whether the champion moved
or not. Tests: the path rule, the scenarios of the correction brief (A-F) against the superseded revision
`m45_unified_r1()` and the "end cell" rule it patched, mirror symmetry, the quantities the correction must not touch,
the revision bookkeeping (preset, recordings, viewer) and the pinned reproduction of the committed M4.5a engine.
"""

import hashlib
import itertools
import json

import numpy as np
import pytest

from lanegame.champions import N_CHANCE, mirror_chance
from lanegame.champions_m4 import resolve_turn_m4
from lanegame.config import B_HOLD, B_YIELD, SP, GameConfig, champ_act
from lanegame.env import LaneEnv
from lanegame.experiments.m45a import movement as mv
from lanegame.positional import WARD, dist, tick_path, tick_speeds

R2, R1 = GameConfig.m45_unified(), GameConfig.m45_unified_r1()
T = R2.tower
NODES = tuple(range(-T, T + 1)) + (WARD,)


# ---------------------------------------------------------------------------------------------- configuration
def test_revision_presets_differ_in_the_movement_rule_only():
    R1.validate()
    R2.validate()
    a, b = R1.to_dict(), R2.to_dict()
    assert [k for k in a if a[k] != b[k]] == ["champion_movement_rule"]
    assert (R1.champion_movement_rule, R2.champion_movement_rule) == ("turn", "tick")
    assert R2.tick_movement and not R1.tick_movement and R1.m45_mechanics
    # a stored configuration without the field (recorded before the correction) loads as revision 1
    d = R2.to_dict()
    del d["champion_movement_rule"]
    assert GameConfig.from_dict(d) == R1


@pytest.mark.parametrize("name", ["m4_four_actions_econ_aggro", "m4_lasthit", "m4", "champions"])
def test_historical_presets_keep_turn_movement(name):
    assert getattr(GameConfig, name)().champion_movement_rule == "turn"


def test_movement_rule_validation():
    with pytest.raises(ValueError):
        GameConfig.m45_unified(champion_movement_rule="glide").validate()
    with pytest.raises(ValueError):                          # an M4.5a rule: natural waves only
        GameConfig.m4(champion_movement_rule="tick").validate()
    with pytest.raises(ValueError):
        GameConfig.champions(champion_movement_rule="tick").validate()


# ---------------------------------------------------------------------------------------------- the path rule
def test_tick_speeds_split_the_turn_earlier_ticks_first():
    assert tick_speeds(2, 2) == (1, 1) and tick_speeds(1, 2) == (1, 0)
    assert tick_speeds(3, 2) == (2, 1) and tick_speeds(4, 2) == (2, 2) and sum(tick_speeds(5, 3)) == 5


def test_tick_path_reaches_the_end_cell_within_the_speed_of_every_tick():
    for c0, end in itertools.product(NODES, NODES):
        if dist(c0, end) > R2.champion_speed:
            continue
        for goals in [None] + list(itertools.product(NODES, NODES)):
            p = tick_path(c0, end, goals, R2.champion_speed, R2.ticks_per_turn, T, True)
            assert len(p) == 2 and p[-1] == end
            assert dist(c0, p[0]) <= 1 and dist(p[0], p[1]) <= 1
        direct = tick_path(c0, end, None, R2.champion_speed, R2.ticks_per_turn, T, True)
        assert dist(direct[0], end) == max(0, dist(c0, end) - 1)          # its edges spent on the earliest tick


def test_tick_path_follows_a_goal_when_the_end_allows_it():
    assert tick_path(-1, 0, (-1, 0), 2, 2, T, True) == (-1, 0)           # waits with its wave, then steps
    assert tick_path(-1, 0, None, 2, 2, T, True) == (0, 0)               # a direct walk steps at once
    assert tick_path(0, 0, (1, 0), 2, 2, T, True) == (1, 0)              # out with its wave and back
    assert tick_path(-2, 0, (2, 2), 2, 2, T, True) == (-1, 0)            # the end cell binds
    assert tick_path(1, WARD, None, 2, 2, T, True) == (0, WARD)          # W is entered through 0 ...
    assert tick_path(WARD, 1, None, 2, 2, T, True) == (0, 1)             # ... and left through it


# ---------------------------------------------------------------------------------------------- the scenarios
# per scenario: the minion fire on A and on B per tick under "end cell" / r1 / r2, and the r2 paths
P, R, _ = "positional", "retaliation", ""
EXPECTED = {
    "A1": (((P, _), (_, _)), ((_, _), (_, _)), ((_, _), (_, _)), ((0, 1), (-2, -2))),
    "A2": (((P, _), (_, _)), ((_, _), (_, _)), ((_, _), (_, _)), ((-1, 0), (-2, -2))),
    "A3": (((P, _), (_, _)), ((_, _), (_, _)), ((_, _), (_, _)), ((-2, -1), (-2, -2))),
    "A4": (((_, _), (_, _)), ((_, _), (_, _)), ((_, _), (_, _)), ((-2, -2), (-2, -2))),
    "B1": (((_, _), (_, _)), ((_, _), (_, _)), ((P, _), (_, _)), ((1, 2), (-1, -1))),
    "B2": (((_, _), (_, _)), ((_, _), (_, _)), ((_, _), (_, _)), ((1, 2), (-1, -1))),
    "C1": (((P, P), (_, _)), ((_, _), (_, _)), ((P, P), (_, _)), ((-1, -1), (-1, 0))),
    "C2": (((P, R), (_, _)), ((_, R), (_, _)), ((P, R), (_, _)), ((-1, -1), (-1, 0))),
    "C3": (((P, _), (_, _)), ((_, _), (_, _)), ((P, _), (_, _)), ((0, 0), (-2, -2))),
    "D1": (((_, _), (_, _)), ((_, _), (_, _)), ((P, _), (_, _)), ((1, 0), (-1, 0))),
    "D2": (((_, _), (_, _)), ((_, _), (_, _)), ((_, _), (_, _)), ((0, 0), (-2, -2))),
    "D3": (((_, _), (_, _)), ((_, _), (_, _)), ((P, _), (_, _)), ((0, 0), (-2, -2))),
    "W1": (((_, _), (_, _)), ((_, _), (_, _)), ((P, _), (_, _)), ((0, WARD), (-1, 0))),
    "S1": (((P, _), (_, _)), ((P, _), (_, _)), ((P, _), (_, _)), ((0, 0), (-2, -2))),
    "N1": (((_, _), (_, _)), ((_, _), (_, _)), ((_, _), (_, _)), ((0, 0), (-1, -1))),
    "R1": (((_, _), (_, R)), ((_, _), (_, R)), ((_, _), (P, R)), ((-2, -2), (1, 2))),
}
SCEN = {s.key: s for s in mv.SCENARIOS}


def _run(rule, sc, k=10, record=False):
    return mv.resolve(rule, sc.lane, mv._champs(mv.CFGS[rule], *sc.pos), k, *sc.actions, mv._chance(sc.hit),
                      record=record)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_scenario_fire_under_each_rule(key):
    sc = SCEN[key]
    *fires, path = EXPECTED[key]
    for rule, want in zip(mv.RULES, fires):
        t = _run(rule, sc)
        assert t.minion_fire == want, (rule, t.minion_fire)
        assert (t.minion_hits[0] > 0) == any(want[0]) and (t.minion_hits[1] > 0) == any(want[1])
    t2, t1 = _run("r2", sc), _run("r1", sc)
    assert t2.path == path and t1.path is None
    # the correction moves nothing but the champions' cells during the ticks
    assert t2.end == t1.end and t2.connected == t1.connected and t2.reach == t1.reach
    assert t2.trade_damage == t1.trade_damage and t2.retaliation == t1.retaliation


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_scenario_mirror_symmetry(key):
    """Scenario E: A and B exchanged give the exchanged result."""
    sc = SCEN[key]
    m4 = mv._champs(R2, *sc.pos)
    u = mv._chance(sc.hit)
    t = mv.resolve("r2", sc.lane, m4, 10, *sc.actions, u)
    tm = mv.resolve("r2", sc.lane.mirror(), m4.mirror(), 10, sc.actions[1], sc.actions[0], mirror_chance(u))
    assert tm.minion_fire == t.minion_fire[::-1] and tm.path == t.path[::-1]
    assert tm.minion_hits == t.minion_hits[::-1] and tm.champ == t.champ.mirror()
    assert tm.lane.state == t.lane.state.mirror() and tm.reward_units == -t.reward_units


def test_scenario_a_hold_follows_its_wave_tick_by_tick():
    """Scenario A: the brief's example. A's wave stays at 0 on tick 0 (it beats B's front there) and walks to +1 on
    tick 1; A walks with it. The end-cell rule placed A at +1 for both ticks, in B's front's cell ahead of its own."""
    t = _run("r2", SCEN["A1"], record=True)
    fronts_a = [tr["A"][0][0] for tr in t.lane.trace]
    assert fronts_a == [0, 1] and t.path[0] == (0, 1) and t.minion_hits == (0, 0)


def test_scenario_a_hold_has_no_immunity():
    """Hold is not exempt: walking back through the enemy wave (D1) is hit, and r1's exemption hid it."""
    t2, t1 = _run("r2", SCEN["D1"]), _run("r1", SCEN["D1"])
    assert t2.behaviors[0] == B_HOLD and t2.minion_hits[0] > 0 and t1.minion_hits[0] == 0


def test_scenario_b_and_c_moving_into_the_wave_is_hit():
    for key in ("B1", "C1", "C2", "C3"):
        t = _run("r2", SCEN[key])
        assert t.start[0] != t.end[0] and t.minion_fire[0][0] == P, key
        assert t.minion_damage[0] > 0 and t.damage_taken[0] >= t.minion_damage[0] // 2
    # a missed or unconnected Trade triggers no retaliation; a landing one does, from tick 1, on top
    assert _run("r2", SCEN["C1"]).retaliation == (False, False) and _run("r2", SCEN["C1"]).connected[0]
    assert not _run("r2", SCEN["C3"]).connected[0]
    c2 = _run("r2", SCEN["C2"])
    assert c2.retaliation == (False, True) and c2.minion_fire[0] == (P, R)


def test_scenario_b_crossing_within_a_tick_is_not_occupancy():
    """B2: A and B's front swap cells within tick 0 (A from 0 to +1, the front from +1 to 0); positions are compared
    at tick boundaries, so A is never in the front's cell (a limitation of the discrete ticks, documented)."""
    t = _run("r2", SCEN["B2"], record=True)
    assert t.path[0] == (1, 2) and [-tr["B"][0][0] for tr in t.lane.trace] == [0, 0] and t.minion_hits[0] == 0


def test_scenario_d_cover_ends_positional_aggro_on_the_tick_it_arrives():
    t = _run("r2", SCEN["D1"], record=True)
    a_front, b_front = t.lane.trace[1]["A"][0][0], -t.lane.trace[1]["B"][0][0]
    assert t.path[0][1] == 0 == a_front == b_front and t.minion_fire[0] == (P, _)
    # the same lane with both fronts arriving in A's cell on the same tick: covered at once (D2); without A's wave
    # there (D3) the arriving front hits A on that tick
    assert _run("r2", SCEN["D2"]).minion_hits[0] == 0 and _run("r2", SCEN["D3"]).minion_fire[0] == (P, _)


def test_ward_and_recall_paths():
    w = _run("r2", SCEN["W1"])
    assert w.path[0] == (0, WARD) and w.minion_hits[0] > 0 and w.ward_placed[0]     # minions do not interrupt
    r = _run("r2", SCEN["R1"])
    assert r.interrupted[0] and r.path[0] == (-2, -2)                                # a recaller channels in place


def test_yield_follows_its_wave_under_the_m2_position_rule():
    """Yield shares Hold's timing problem when it farms from its spot (position_rule "m2"); the stand-off rule of
    m45_unified keeps it one cell behind the enemy's starting front (A4)."""
    yld = champ_act(SP, B_YIELD)
    sc = SCEN["A2"]
    for name, cfg in (("r2", R2.with_(position_rule="m2")), ("r1", R1.with_(position_rule="m2"))):
        t = resolve_turn_m4(sc.lane, mv._champs(cfg, *sc.pos), 10, yld, sc.actions[1], cfg, u=mv._chance())
        assert t.end[0] == 0 and t.minion_hits[0] == 0
        if name == "r2":
            assert t.path[0] == (-1, 0)


# ---------------------------------------------------------------------------------------------- random states
def _turns(n, seed, cfg=R2):
    env = LaneEnv(cfg, max_turns=10 ** 6)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        a, b = int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1)))
        out.append((env.state, env.champ, env.k, a, b, tuple(float(x) for x in rng.random(N_CHANCE))))
        env.step(a, b)
    return out


def test_paths_respect_the_speed_and_the_correction_touches_only_minion_targeting():
    for lane_, m4, k, a, b, u in _turns(500, 21):
        t2 = resolve_turn_m4(lane_, m4, k, a, b, R2, u=u)
        t1 = resolve_turn_m4(lane_, m4, k, a, b, R1, u=u)
        for s in (0, 1):
            p = t2.path[s]
            if t2.away[s]:
                continue
            if t2.behaviors[s] == 4:                                    # Recall: channels in its cell
                assert p == (t2.start[s],) * 2
                continue
            assert p[-1] == t2.end[s] and dist(t2.start[s], p[0]) <= 1 and dist(p[0], p[1]) <= 1
        for f in ("end", "connected", "reach", "modes", "wave_budget", "trade_damage", "threatened", "retaliation",
                  "farm_spot", "exposure"):
            assert getattr(t2, f) == getattr(t1, f), f
        if t2.minion_hits == t1.minion_hits == (0, 0):
            assert t2.champ == t1.champ and t2.lane.state == t1.lane.state and t2.reward_units == t1.reward_units


def test_resolver_mirror_equivariance_on_random_states():
    for lane_, m4, k, a, b, u in _turns(400, 23):
        t = resolve_turn_m4(lane_, m4, k, a, b, R2, u=u)
        tm = resolve_turn_m4(lane_.mirror(), m4.mirror(), k, b, a, R2, u=mirror_chance(u))
        assert tm.path == t.path[::-1] and tm.minion_fire == t.minion_fire[::-1]
        assert tm.champ == t.champ.mirror() and tm.reward_units == -t.reward_units


def test_env_reports_the_path_and_replays_deterministically():
    def run(cfg):
        env = LaneEnv(cfg, max_turns=120, record_ticks=True)
        env.reset(seed=31)
        rng = np.random.default_rng(31)
        h, paths = hashlib.sha256(), 0
        while not env.done:
            r = env.step(int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1))))
            h.update(json.dumps(r.info, sort_keys=True, default=repr).encode())
            if cfg.tick_movement:
                assert r.info["champion_path"] is not None
                assert all(r.info["champion_path"][s][-1] == r.info["champ_end"][s]
                           for s in (0, 1) if not r.info["away"][s])
                paths += 1
            else:
                assert "champion_path" not in r.info
        return h.hexdigest(), paths
    assert run(R2) == run(R2) and run(R2)[1] == 120
    run(R1)


def test_r1_preset_reproduces_the_committed_m45a_engine():
    """Random games on m45_unified_r1() equal those of m45_unified() at ea3d9cb (MinionTarget entries reduced to the
    three fields that existed there; digest recorded from a worktree of that commit)."""
    h = hashlib.sha256()
    for g in range(3):
        env = LaneEnv(R1, max_turns=150, record_ticks=True)
        env.reset(seed=900 + g)
        rng = np.random.default_rng(g)
        while not env.done:
            info = dict(env.step(int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1)))).info)
            info["minion_targets"] = [None if t is None else list(t)[:3] for t in info["minion_targets"]]
            h.update(json.dumps(info, sort_keys=True, default=repr).encode())
    assert h.hexdigest()[:16] == "6eeead1ea6a22614"


# ---------------------------------------------------------------------------------------------- recordings, viewer
def test_recordings_carry_and_route_the_rules_revision(tmp_path):
    from lanegame.viewer import session as ss
    assert ss.recording_stamp("m45_unified") == {"config": "m45_unified", "rules_revision": 2}
    assert ss.recording_stamp("m4_lasthit") == {"config": "m4_lasthit"}
    base = {"schema": ss.RECORDING_SCHEMA, "version": 1, "config": "m45_unified", "scenario": "s0", "gap": "equal",
            "seed": 0, "actions": [[0, 0]]}
    for extra, want in (({}, "m45_unified_r1"), ({"rules_revision": 1}, "m45_unified_r1"),
                        ({"rules_revision": 2}, "m45_unified")):
        path = tmp_path / f"r{len(extra)}{extra.get('rules_revision', 0)}.json"
        path.write_text(json.dumps({**base, **extra}))
        setup, actions, meta = ss.load_recording(str(path))
        assert setup.config == want and actions == [(0, 0)]
        assert (meta.get("replayed_as") == want) == (want == "m45_unified_r1")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**base, "rules_revision": 7}))
    with pytest.raises(ValueError):
        ss.load_recording(str(bad))


def test_viewer_draws_champions_at_their_tick_cells():
    from lanegame.viewer import render
    from lanegame.viewer import session as ss
    hold = champ_act(SP, B_HOLD)
    rec = ss.play(ss.Setup("m45_unified", "walk_back", "equal", 0), prefix=[(hold, hold)])
    path = rec.turns[0].info["champion_path"]
    assert path[0] == (1, 0)
    for f in (1, 2):
        view = ss.frame_view(rec, 0, f)
        assert view["champions"]["A"]["pos_global"] == path[0][f - 1]
        assert view["champions"]["B"]["pos_global"] == -path[1][f - 1]
        render.lane_figure(view, rec.cfg)
    assert ss.frame_view(rec, 0, 3)["champions"]["A"]["pos_global"] == rec.champs[1].a.pos


@pytest.mark.parametrize("name", ["m45a_M1_hold_follows", "m45a_M2_chase_missed", "m45a_M3_chase_landed",
                                  "m45a_M4_walk_back"])
def test_movement_demonstrations_show_their_event_and_replay_exactly(name, tmp_path):
    from lanegame.experiments.m45a import demos
    from lanegame.experiments.rl.m44.trace import verify
    demo = next(d for d in demos.DEMOS if d.name == name)
    r = demos.export(demo, str(tmp_path))
    v = verify(r["path"])
    assert v["actions_equal"] and v["payoff_equal"] and v["config"] == "m45_unified"
    with open(r["path"], encoding="utf-8") as f:
        assert json.load(f)["rules_revision"] == 2
    a = r["summary"]["A"]
    if name == "m45a_M1_hold_follows":
        assert a["minion_units_taken"] == 0
    elif name == "m45a_M2_chase_missed":
        assert a["positional_ticks_taken"] >= 1 and a["retaliation_ticks_taken"] == 0
    elif name == "m45a_M3_chase_landed":
        assert a["positional_ticks_taken"] >= 1 and a["retaliation_ticks_taken"] >= 1
    else:
        assert a["positional_ticks_taken"] == 1
