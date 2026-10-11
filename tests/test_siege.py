"""Siege objectives (Apply Pressure / Establish Vision), plates, thinning and persistent vision (siege model)."""

import itertools

import numpy as np
import pytest

from lanegame import (CONTEST, NONE, PRESSURE, SAFE, SH, SH_C, SH_S, SP, SP_C, SP_S, VISION, GameConfig, LaneGame,
                      act, base_of, objective_of, posture_of, wave_of)
from lanegame.baselines import all_baselines, siege_baselines, stack_builder
from lanegame.best_response import exploitability
from lanegame.game import collection_efficiency, gank_probability, turn_farm
from lanegame.mechanics import initial_state, legal_actions, step_turn, thinning_target, total_hp
from lanegame.reachability import (ALL_EVENTS, SIEGE_EVENTS, explore_events, find_vision_strategy, replay,
                                   strategy_complete, strategy_trace)
from lanegame.solver import solve_equilibrium
from lanegame.state import State

from conftest import all_transitions

SP_CP, SH_CP = act(SP, CONTEST, PRESSURE), act(SH, CONTEST, PRESSURE)
SP_CV, SH_CV = act(SP, CONTEST, VISION), act(SH, CONTEST, VISION)


# ------------------------------------------------------------------ legality and encoding
def test_action_encoding_with_objectives():
    for w in (SP, SH, 2):
        for p in (CONTEST, SAFE):
            for o in (NONE, PRESSURE, VISION):
                a = act(w, p, o)
                assert (wave_of(a), posture_of(a), objective_of(a)) == (w, p, o) and base_of(a) == act(w, p)
    assert (SP_CP, SH_CP, SP_CV, SH_CV) == (6, 7, 12, 13)


def test_legal_sets_siege_model(sg_game, sg_layers):
    """Without an eligible siege: the 4 or 6 base actions. With one: Contest actions carry Pressure or Vision,
    Safe actions carry nothing; Contest without an objective is not offered."""
    cfg = sg_game.cfg
    seen_siege = 0
    for layer in sg_layers:
        for s in layer:
            for side in (0, 1):
                la = sg_game.legal(s, side)
                assert la == tuple(sorted(la))
                if s.siege_of(side) > 0:
                    seen_siege += 1
                    assert all(objective_of(a) != NONE for a in la if posture_of(a) == CONTEST)
                    assert all(objective_of(a) == NONE for a in la if posture_of(a) == SAFE)
                    assert {SP_CP, SH_CP, SP_CV, SH_CV, SP_S, SH_S} <= set(la) and len(la) in (6, 9)
                else:
                    assert all(objective_of(a) == NONE for a in la) and len(la) in (4, 6)
    assert seen_siege > 0


def test_objectives_never_legal_in_older_models(rev_game, rev_layers, v1_game, v1_layers):
    for g, layers in ((rev_game, rev_layers), (v1_game, v1_layers)):
        for layer in layers:
            for s in layer:
                for side in (0, 1):
                    assert all(objective_of(a) == NONE for a in g.legal(s, side))


def test_pressure_requires_an_eligible_siege(sg_cfg):
    s = initial_state(sg_cfg)
    assert SP_CP not in legal_actions(s, 0, sg_cfg)
    with pytest.raises(AssertionError):
        step_turn(s, 0, SP_CP, SP_C, sg_cfg)
    with pytest.raises(ValueError):
        LaneGame(sg_cfg).transition(s, 0, SP_CV, SP_C)


# ------------------------------------------------------------------ vision timing
def test_vision_is_not_retroactive_and_protects_the_next_turns(sg_game, sg_layers):
    """Establishing vision never changes the gank probability of the turn it is established in; the timer is
    vision_turns afterwards; while it is positive every action's risk is reduced by vision_reduction."""
    cfg = sg_game.cfg
    seen = 0
    for k, s, a, b in all_transitions(sg_game, sg_layers):
        if objective_of(a) != VISION:
            continue
        seen += 1
        tr_v = sg_game.transition(s, k, a, b)
        tr_p = sg_game.transition(s, k, act(wave_of(a), CONTEST, PRESSURE), b)
        assert tr_v.gank_p[0] == tr_p.gank_p[0]
        assert tr_v.pre_state.va == cfg.vision_turns
        assert tr_v.vision_set == (True, False) or tr_v.vision_set == (True, True)
    assert seen > 0
    # protection: p(with timer) = (1 - rho) * p(without), for any action and exposure
    s = State(-4, 4, (), 8, -3, 4, (), 0, 3, 0, 2, 0)
    for a in legal_actions(s, 0, cfg):
        for e in range(-cfg.tower, cfg.tower + 1):
            assert abs(gank_probability(s, 0, a, cfg, e) - (1 - cfg.vision_reduction) * gank_probability(s._replace(va=0), 0, a, cfg, e)) < 1e-15


def test_vision_countdown_refresh_and_expiry(sg_cfg):
    cfg = sg_cfg
    V = cfg.vision_turns
    s = State(-4, 4, (), 12, -3, 4, (), 0, 0, 0, 3, 0)
    r = step_turn(s, 2, SP_CV, SP_S, cfg)
    assert r.state.va == V and r.vision_set == (True, False)
    timers = []
    st = r.state
    for k in range(3, 3 + V + 1):
        st = step_turn(st, k, SP_S, SP_S, cfg).state
        timers.append(st.va)
    assert timers == list(range(V - 1, -1, -1)) + [0]
    # refresh: re-establishing with 4 turns left resets to V (no stacking)
    st = step_turn(r.state, 3, SP_S, SP_S, cfg).state
    st = step_turn(st, 4, SP_S, SP_S, cfg).state
    assert st.va == V - 2 and st.asg > 0
    st2 = step_turn(st, 5, SP_CV, SP_S, cfg).state
    assert st2.va == V
    # expiry is independent of later wave management, positioning and siege choices
    for plan in ([SP_C, SH_C, SP_S, SH_S, SP_C, SH_S], [SP_S] * 6):
        st = r.state
        for k, a in enumerate(plan):
            legal = legal_actions(st, 0, cfg)
            a = a if a in legal else (a + 6 * PRESSURE if a + 6 * PRESSURE in legal else SP_S)
            st = step_turn(st, 3 + k, a, SP_S, cfg).state
        assert st.va == 0


def test_existing_vision_protects_apply_pressure(sg_cfg):
    cfg = sg_cfg
    s = State(-4, 4, (), 12, -3, 4, (), 0, 0, 0, 3, 0)
    g = LaneGame(cfg)
    r = g.transition(s, 2, SP_CV, SP_C)
    s2 = r.pre_state
    assert s2.va == cfg.vision_turns and s2.asg > 0
    tr = g.transition(s2, 3, SP_CP, SP_C)
    tr0 = g.transition(s2._replace(va=0), 3, SP_CP, SP_C)
    assert tr.plates[0] > 0 and tr.exposure[0] == cfg.tower
    assert abs(tr.gank_p[0] - (1 - cfg.vision_reduction) * tr0.gank_p[0]) < 1e-15
    # pressure does not cancel the vision
    assert tr.pre_state.va == cfg.vision_turns - 1


# ------------------------------------------------------------------ plates
def test_plates_only_through_pressure_and_never_duplicated(sg_game, sg_layers):
    """A plate is claimed only by Apply Pressure, at most plates_per_turn per turn, never beyond the pending
    opportunities; opportunities open only with a crash (by its size) and lapse when the siege ends."""
    cfg = sg_game.cfg
    claimed = 0
    for k, s, a, b in all_transitions(sg_game, sg_layers):
        r = step_turn(s, k, a, b, cfg)
        tr = sg_game.transition(s, k, a, b)
        for side, act_s in ((0, a), (1, b)):
            pend0 = s.plates_of(side)
            c = r.plates[side]
            if objective_of(act_s) != PRESSURE:
                assert c == 0
            else:
                assert c == min(cfg.plates_per_turn, pend0)
            claimed += c
            assert tr.plates[side] == cfg.plate_value * c
            grant = sum(cfg.plate_grant[cfg.size_category(h)] for h in [r.crash_hp[side]] if h > 0)
            if r.state.siege_of(side) == 0:
                assert r.state.plates_of(side) == 0
            else:
                assert r.state.plates_of(side) == min(cfg.plates_cap, pend0 - c + grant) or \
                    (r.events["crashed"][side] > 1 and r.state.plates_of(side) >= min(cfg.plates_cap, pend0 - c + grant))
            if s.siege_of(side) == 0:
                assert pend0 == 0                               # no opportunities without a siege ("crash" mode)
    assert claimed > 0


def test_plate_total_bounded_by_crash_grants_over_whole_games(sg_cfg):
    """Over complete games, each side's plate count never exceeds the opportunities its crashes opened."""
    cfg = sg_cfg.with_(horizon=32)
    g = LaneGame(cfg)
    rng = np.random.default_rng(3)
    for _ in range(30):
        s = g.initial_state()
        plates, grants = [0, 0], [0, 0]
        for k in range(cfg.horizon):
            la, lb = g.legal(s, 0), g.legal(s, 1)
            # prefer pressure when available so plates are actually claimed
            a = la[rng.integers(len(la))]
            b = lb[rng.integers(len(lb))]
            r = step_turn(s, k, a, b, cfg)
            for side in (0, 1):
                plates[side] += r.plates[side]
                if r.crash_hp[side] > 0:
                    grants[side] += cfg.plate_grant[cfg.size_category(r.crash_hp[side])]
            s = g.transition(s, k, a, b).pre_state
        for side in (0, 1):
            assert plates[side] <= grants[side]


def test_plates_are_tower_income_not_farm(sg_game, sg_layers):
    for k, s, a, b in itertools.islice(all_transitions(sg_game, sg_layers), 30000):
        tr = sg_game.transition(s, k, a, b)
        r = step_turn(s, k, a, b, sg_game.cfg)
        avail, coll = turn_farm(r, a, b, sg_game.cfg)
        assert tr.farm == coll and tr.plate_payoff == tr.plates[0] - tr.plates[1]
        assert abs(tr.payoff - (tr.farm_payoff + tr.plate_payoff + tr.gank_payoff)) < 1e-15


# ------------------------------------------------------------------ thinning
def test_thinning_requires_pressure_access_and_an_occupied_tower(sg_game, sg_layers):
    cfg = sg_game.cfg
    T = cfg.tower
    seen = 0
    for k, s, a, b in all_transitions(sg_game, sg_layers):
        r = step_turn(s, k, a, b, cfg)
        for side, act_s in ((0, a), (1, b)):
            t = r.thinned[side]
            assert 0 <= t <= cfg.thin_damage
            if objective_of(act_s) != PRESSURE:
                assert t == 0
            if t > 0:
                seen += 1
                assert s.siege_of(side) > 0 and posture_of(act_s) == CONTEST
                # the thinned HP died beyond the enemy tower cell and is a farm opportunity of the thinner
                assert any(x >= T for x in r.kills[side])
                assert r.events["thinned_hp"][1 - side] == t
    assert seen > 0


def test_thinning_hits_only_accessible_existing_minions(sg_cfg):
    cfg = sg_cfg
    W = cfg.wave_hp
    H = 3 * W
    # B's wave waiting one cell behind its tower: accessible
    s = State(-4, W, (), H, -3, W, (), 0, 0, 0, 3, 0)
    r = step_turn(s, 2, SP_CP, SP_S, cfg)
    assert r.thinned == (1, 0) and r.state.bh == W - 1 and r.kills[0].get(3, 0) == 1
    # B's wave at its base (two cells behind the tower) walks up to -3 on tick 1 and is thinned there: movement
    # precedes thinning within a tick, and the thinned HP is recorded where the packet stands (+3)
    s = State(-4, W, (), H, -4, W, (), 0, 0, 0, 3, 0)
    r = step_turn(s, 2, SP_CP, SP_S, cfg, record=True)
    assert r.trace[0]["thinned"] == [1, 0] and r.trace[1]["thinned"] == [0, 0]
    assert r.thinned == (1, 0) and r.state.bh == W - 1 and r.kills[0] == {3: 1}
    # B's packet standing IN its tower cell (-2) is accessible too
    s = State(-4, W, (), H, -2, W, (), 0, 0, 0, 3, 0)
    assert step_turn(s, 2, SP_CP, SP_S, cfg).thinned == (1, 0)
    # no B minions at all: nothing to thin, the plate is still claimed
    s = State(-4, W, (), H, -4, 0, (), 0, 0, 0, 3, 0)
    r = step_turn(s, 2, SP_CP, SP_S, cfg)
    assert r.thinned == (0, 0) and r.plates == (1, 0)
    # a thinned minion is removed from the state (HP conservation)
    s = State(-4, W, (), H, -3, 1, (), 0, 0, 0, 3, 0)
    r = step_turn(s, 2, SP_CP, SP_S, cfg)
    assert r.state.bh == 0 or r.state.bh == W           # the 1-HP packet dies; a spawn may follow
    assert r.events["died"][1] == 1 and r.thinned == (1, 0)
    # the per-turn limit: thin_damage per turn, however many ticks the target is accessible
    s = State(-4, W, (), H, -3, 2 * W, (), 0, 0, 0, 3, 0)
    r = step_turn(s, 2, SP_CP, SP_S, cfg)
    assert r.thinned == (cfg.thin_damage, 0)


def test_thinning_needs_the_tower_to_be_busy_and_no_lane_fight(sg_cfg):
    """A's arriving 1-HP wave fights B's waiting wave on tick 1 (no thinning during a lane fight) and dies.
    On tick 2 thinning happens only if the tower is still occupied by A's crashed mass: with 1 HP of mass
    (removed by tower fire on tick 1) the tower is free and nothing is thinned; with 3 HP it is."""
    cfg = sg_cfg
    for mass, expect in ((1, 0), (3, 1)):
        s = State(2, 1, (), mass, -3, 8, (), 0, 0, 0, 1, 0)      # A front in B's tower cell, B's wave right behind it
        r = step_turn(s, 2, SP_CP, SP_C, cfg, record=True)
        assert r.trace[0]["fight"] is not None and r.trace[0]["thinned"] == [0, 0]
        assert r.events["died"][0] == 1
        assert r.trace[1]["thinned"] == [expect, 0] and r.thinned == (expect, 0)
    # the whole per-turn budget may be spent on the first accessible tick
    cfg2 = sg_cfg.with_(thin_damage=2)
    r = step_turn(State(-4, 4, (), 6, -3, 8, (), 0, 0, 0, 1, 0), 2, SP_CP, SP_S, cfg2, record=True)
    assert r.trace[0]["thinned"] == [2, 0] and r.thinned == (2, 0)


def test_thinning_reduces_the_bounce_but_does_not_erase_it(sg_cfg):
    """From the 'bounce' start (12 HP under B's tower, B wave waiting; A's waves wait at the lane under Slow
    Push so nothing else touches B's wave), continuous pressure thins 1 HP per turn over the 6-turn siege:
    B releases a 10-HP wave instead of a capped 12-HP wave with a 4-HP reinforcement queued behind it (16 HP
    of blob). With Shove, A's arriving waves also fight the waiting wave, so even less is left; with Vision
    nothing touches it."""
    cfg = sg_cfg
    W = cfg.wave_hp

    def released(plan_a):
        """(front HP, total B lane HP) on the turn the mass is gone."""
        s = State(-4, W, (), 3 * W, -3, W, (), 0, 0, 0, 3, 0)
        for k in range(2, 12):
            a = plan_a if s.asg > 0 else SP_S
            s = step_turn(s, k, a, SP_C, cfg).state
            if s.asg == 0:
                return s.bh, s.bh + W * len(s.br)
        return None
    # no pressure: a capped 12-HP wave plus a queued 4-HP reinforcement (16 HP of blob)
    assert released(SP_S) == (3 * W, 4 * W)
    # 6 pressure turns thin 6 HP: a 10-HP wave, nothing queued
    assert released(SP_CP) == (4 * W - 6, 4 * W - 6)
    assert released(SP_CV) == (3 * W, 4 * W)       # vision does not touch the minions
    assert released(SH_CP)[1] < 4 * W - 6          # shoving waves fight the waiting wave as well


def test_hp_conservation_with_thinning(sg_game, sg_layers):
    cfg = sg_game.cfg
    for k, s, a, b in all_transitions(sg_game, sg_layers):
        r = step_turn(s, k, a, b, cfg)
        before, after = total_hp(s, cfg), total_hp(r.state, cfg)
        ev = r.events
        for side in (0, 1):
            assert before[side] + ev["spawned_hp"][side] == after[side] + ev["killed_hp"][side] + ev["tower_hp"][side]
            assert ev["thinned_hp"][side] <= ev["killed_hp"][side]


def test_pressure_exposure_and_safe_disengagement(sg_game, sg_layers):
    """Apply Pressure / Establish Vision put the champion in the enemy tower cell (+T); Safe during a siege
    keeps it at its safe line and claims nothing."""
    cfg = sg_game.cfg
    for k, s, a, b in itertools.islice(all_transitions(sg_game, sg_layers), 40000):
        if s.asg == 0:
            continue
        tr = sg_game.transition(s, k, a, b)
        if objective_of(a) != NONE:
            assert tr.exposure[0] == cfg.tower
        else:
            assert posture_of(a) == SAFE and tr.exposure[0] <= -1 and tr.plates[0] == 0 and tr.thinned[0] == 0


# ------------------------------------------------------------------ state and solver
def test_state_fields_are_bounded_and_markov(sg_layers, sg_cfg):
    cfg = sg_cfg
    for layer in sg_layers:
        for s in layer:
            assert 0 <= s.va <= cfg.vision_turns and 0 <= s.vb <= cfg.vision_turns
            assert 0 <= s.ap <= cfg.plates_cap and 0 <= s.bp <= cfg.plates_cap
            assert (s.ap == 0 or s.asg > 0) and (s.bp == 0 or s.bsg > 0)
            assert all(type(v) is int for f in s for v in (f if isinstance(f, tuple) else (f,)))


def test_v2_preset_reproduces_the_revision_report():
    """GameConfig.v2() is the previous revision: same state space, same mixed count, same exploitability."""
    g = LaneGame(GameConfig.v2(horizon=32))
    sol = solve_equilibrium(g)
    assert sol.stats.states == 14849 and sol.stats.mixed_states == 1030
    assert len({s for l in sol.layers for s in l}) == 2480
    expected = {"uniform": 10.078, "aggressive": 24.863, "freeze_contest": 9.445, "passive_safe": 9.415,
                "stack_builder": 5.782, "release_safe": 7.690, "contest_slowpush": 6.590}
    for b in all_baselines():
        assert abs(exploitability(g, sol.layers, b, 0, sol.value).exploitability - expected[b.name]) < 5e-4


def test_siege_baselines_resolve_objectives(sg_game, sg_layers):
    """Every siege baseline returns a valid distribution at every reachable node, and the siege tendency is
    honoured whenever an objective is available."""
    for pol in siege_baselines():
        for k, layer in enumerate(sg_layers):
            for s in layer[:200]:
                la = sg_game.legal(s, 0)
                p = pol.move(k, s, 0, la)
                assert p.shape == (len(la),) and abs(p.sum() - 1) < 1e-12 and np.all(p >= 0)
    s = State(-4, 4, (), 12, -3, 4, (), 0, 0, 0, 3, 0)
    la = legal_actions(s, 0, sg_game.cfg)
    assert la[int(np.argmax(stack_builder().move(2, s, 0, la)))] == SP_CP
    assert la[int(np.argmax(stack_builder(siege="vision").move(2, s, 0, la)))] == SP_CV
    assert la[int(np.argmax(stack_builder(siege="safe").move(2, s, 0, la)))] == SP_S


# ------------------------------------------------------------------ reachability from the standard start
@pytest.fixture(scope="module")
def siege_reach():
    g = LaneGame(GameConfig(horizon=20))
    return g, explore_events(g)


@pytest.mark.parametrize("event", list(SIEGE_EVENTS) + ["bounce", "crash_large", "crash_onto_siege"])
def test_siege_events_reachable_from_standard_start(siege_reach, event):
    g, rep = siege_reach
    assert rep.counts[event] > 0 and event in rep.witness, event
    w = rep.witness[event]
    s, r = replay(g, w.plan, w.choices)[-1]
    assert s == w.state and ALL_EVENTS[event](g.cfg, s, *w.plan[-1], r)


@pytest.fixture(scope="module")
def vision_strategy():
    g = LaneGame(GameConfig(horizon=24))
    return g, find_vision_strategy(g, horizon=24)


def test_vision_shove_crash_pressure_strategy_is_reachable(vision_strategy):
    """The complete Vision -> (prepare a wave) -> Crash -> Pressure-under-that-vision sequence is reachable from
    s_0 by some legal joint-action history, and the witness replays with every claimed property."""
    g, w = vision_strategy
    cfg = g.cfg
    assert w.found, "strategy not found within the searched horizon"
    assert w.turns_after_vision <= cfg.vision_turns
    tr = strategy_trace(g, w.plan)
    assert tr[w.vision_turn]["objective"] == "Vision"
    assert tr[w.crash_turn]["crash_hp"] > 0 and w.vision_turn < w.crash_turn < w.k
    last = tr[w.k]
    assert last["objective"] == "Pressure" and last["vision_before"] > 0 and last["plates_claimed"] > 0
    assert abs(last["gank_p"] - (1 - cfg.vision_reduction) * last["gank_p_without_vision"]) < 1e-12
    # no vision was re-established between the protecting vision and the pressure turn
    assert all(row["objective"] != "Vision" for row in tr[w.vision_turn + 1:w.k])
    # the strategy predicate holds on a direct replay of the plan's last turn
    s = g.initial_state()
    flag = 0
    from lanegame.reachability import _next_flag
    for k, (a, b) in enumerate(w.plan[:-1]):
        r = step_turn(s, k, a, b, cfg)
        flag = _next_flag(flag, s, a, r, 0)
        s = r.state
    a, b = w.plan[-1]
    assert strategy_complete(s, a, step_turn(s, w.k, a, b, cfg), flag, 0)
