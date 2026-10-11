"""Combined actions (wave management x positioning) in the revised model."""

import pytest

from lanegame import (CONTEST, F, F_C, F_S, SAFE, SH, SH_C, SH_S, SP, SP_C, SP_S, GameConfig, LaneGame, act,
                      posture_of, wave_of)
from lanegame.game import champion_exposure, gank_probability, turn_gank_probabilities
from lanegame.mechanics import legal_actions, step_turn
from lanegame.state import State

from conftest import all_transitions


def test_legal_action_sets(rev_game, rev_layers):
    """Both postures are always available; Freeze (with either posture) only with enemy minions at own x <= -1."""
    cfg = rev_game.cfg
    for layer in rev_layers:
        for s in layer:
            for side in (0, 1):
                la = rev_game.legal(s, side)
                assert la == tuple(sorted(la)) and len(la) in (4, 6)
                assert {SP_C, SH_C, SP_S, SH_S} <= set(la)
                efx, efh = (s.bx, s.bh) if side == 0 else (s.ax, s.ah)
                freeze = efh > 0 and -efx <= cfg.freeze_legal_max
                assert (F_C in la) == (F_S in la) == freeze


def test_revised_freeze_legality(rev_cfg):
    cfg = rev_cfg
    W = cfg.wave_hp
    assert F_C not in legal_actions(State(-4, W, (), 0, 0, W, (), 0, 0, 0), 0, cfg)   # enemy at the river
    assert F_C in legal_actions(State(-4, W, (), 0, 1, W, (), 0, 0, 0), 0, cfg)       # enemy at own -1
    assert F_C in legal_actions(State(-3, W, (), 0, 2, W, (), 0, 0, 0), 0, cfg)       # enemy contesting own tower
    assert F_C not in legal_actions(State(-4, W, (), 0, 0, 0, (), 0, 0, 0), 0, cfg)   # no enemy wave


def test_action_encoding():
    for w in (SP, SH, F):
        for p in (CONTEST, SAFE):
            a = act(w, p)
            assert wave_of(a) == w and posture_of(a) == p
    assert (SP_C, SH_C, F_C, SP_S, SH_S, F_S) == (0, 1, 2, 3, 4, 5)


def test_joint_actions_are_simultaneous(rev_game, rev_solution):
    """Every stage game is a full |A_s| x |B_s| matrix: each player's payoff depends on both current choices."""
    for k, layer in enumerate(rev_solution.layers):
        for s in layer:
            la, lb = rev_solution.legal[k][s]
            assert rev_solution.Q[k][s].shape == (len(la), len(lb))


def test_safe_never_raises_gank_risk(rev_game, rev_layers):
    """For the same wave action and the same opponent action, Safe is never riskier than Contest."""
    cfg = rev_game.cfg
    for k, s, a, b in all_transitions(rev_game, rev_layers):
        if posture_of(a) != CONTEST:
            continue
        safe = act(wave_of(a), SAFE)
        pc = rev_game.transition(s, k, a, b).gank_p[0]
        ps = rev_game.transition(s, k, safe, b).gank_p[0]
        assert ps <= pc + 1e-15


def test_safe_exposure_is_capped_and_contest_follows_the_wave(rev_cfg):
    cfg = rev_cfg
    W = cfg.wave_hp
    s = State(-4, W, (), 2 * W, -3, W, (), 0, 0, 0)       # A's crashed mass is under B's tower
    r = step_turn(s, 2, SP_C, SP_C, cfg)
    assert champion_exposure(r.exposure[0], SP_C, cfg) == cfg.tower    # Contest stays to pressure
    assert champion_exposure(r.exposure[0], SP_S, cfg) <= -1           # Safe walks away


def test_minions_move_and_fight_the_same_under_safe(rev_game, rev_layers):
    """Positioning changes the champion, not the minions: with Slow Push or Freeze (no champion damage)
    the lane evolves identically under Contest and Safe."""
    cfg = rev_game.cfg
    for k, s, a, b in all_transitions(rev_game, rev_layers):
        if wave_of(a) == SH or posture_of(a) != CONTEST:
            continue
        r1 = step_turn(s, k, a, b, cfg)
        r2 = step_turn(s, k, act(wave_of(a), SAFE), b, cfg)
        assert r1.state == r2.state and r1.kills == r2.kills


def test_contest_adds_pressure_beyond_the_safe_line(rev_cfg):
    """Shove + Contest kills enemy minions at the river faster than Shove + Safe (champion damage needs presence)."""
    cfg = rev_cfg
    g = LaneGame(cfg)
    s = g.initial_state()
    killed = {}
    for a in (SH_C, SH_S):
        st, tot = s, 0
        for k in range(3):
            r = step_turn(st, k, a, SP_C, cfg)
            tot += r.events["killed_hp"][1]
            st = r.state
        killed[a] = tot
    assert killed[SH_C] > killed[SH_S]


def test_safe_shove_still_helps_near_own_tower(rev_cfg):
    """A Safe champion standing at its safe line still adds shove damage to a fight there."""
    cfg = rev_cfg
    W = cfg.wave_hp
    s = State(-1, W, (), 0, 0, W, (), 0, 0, 0)            # A wave at own -1, B wave at the river: in contact
    r_safe = step_turn(s, 3, SH_S, SP_C, cfg)
    r_sp = step_turn(s, 3, SP_S, SP_C, cfg)
    assert r_safe.events["killed_hp"][1] > r_sp.events["killed_hp"][1]


def test_positioning_is_symmetric(rev_game, rev_layers):
    for k, s, a, b in list(all_transitions(rev_game, rev_layers))[:20000]:
        tr = rev_game.transition(s, k, a, b)
        trm = rev_game.transition(s.mirror(), k, b, a)
        assert trm.gank_p == tr.gank_p[::-1] and trm.exposure == tr.exposure[::-1]


def test_contest_raises_risk_near_own_tower_only_by_the_flat_term(rev_cfg):
    cfg = rev_cfg
    W = cfg.wave_hp
    s = State(-1, W, (), 0, 1, W, (), 0, 0, 0)
    for w in (SP, F):
        pc = gank_probability(s, 0, act(w, CONTEST), cfg, -1)
        ps = gank_probability(s, 0, act(w, SAFE), cfg, -1)
        assert abs((pc - ps) - (cfg.gank_action[act(w, CONTEST)] - cfg.gank_action[act(w, SAFE)])) < 1e-12
