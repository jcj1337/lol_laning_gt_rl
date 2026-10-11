"""Mathematical invariants (both models): zero-sum, mirror symmetry, terminal handling, valid distributions."""

import numpy as np

from lanegame import GameConfig, LaneGame
from lanegame.game import apply_choice
from lanegame.matrix_game import solve_matrix_game
from lanegame.mechanics import step_turn
from lanegame.solver import solve_equilibrium

from conftest import all_transitions


def test_zero_sum_payoffs(game, layers):
    """Every payoff component is A's gain = B's loss: farm_A - farm_B, L(p_B - p_A) and rho_A - rho_B."""
    cfg = game.cfg
    for k, s, a, b in all_transitions(game, layers):
        tr = game.transition(s, k, a, b)
        assert abs(tr.gank_payoff - cfg.gank_loss * (tr.gank_p[1] - tr.gank_p[0])) < 1e-15
        assert tr.farm_payoff == tr.farm[0] - tr.farm[1]
        assert tr.payoff == tr.farm_payoff + tr.gank_payoff      # nothing else enters the stage payoff
        for i, j, s2, r in tr.choice_outcomes(cfg):
            assert r == tr.options_a[i][0] - tr.options_b[j][0]


def test_cached_transitions_equal_direct_computation(game, layers):
    """The transition cache is keyed on (state, k mod phase_period, a, b); it must reproduce step_turn at every k."""
    cfg = game.cfg
    for k, s, a, b in all_transitions(game, layers):
        tr = game.transition(s, k, a, b)
        r = step_turn(s, k, a, b, cfg)
        assert tr.pre_state == r.state and tr.crash_hp == r.crash_hp


def test_mirror_equivariance_of_transitions(game, layers):
    """Swapping the players and reflecting the lane commutes with the dynamics, farming and gank risk."""
    cfg = game.cfg
    for k, s, a, b in all_transitions(game, layers):
        m = s.mirror()
        assert game.legal(m, 0) == game.legal(s, 1) and game.legal(m, 1) == game.legal(s, 0)
        r = step_turn(s, k, a, b, cfg)
        rm = step_turn(m, k, b, a, cfg)
        assert rm.state == r.state.mirror()
        assert rm.crash_hp == r.crash_hp[::-1]
        assert rm.kills == r.kills[::-1]
        tr, trm = game.transition(s, k, a, b), game.transition(m, k, b, a)
        assert abs(tr.gank_payoff + trm.gank_payoff) < 1e-15
        assert abs(tr.farm_payoff + trm.farm_payoff) < 1e-15
        assert trm.farm == tr.farm[::-1] and trm.farm_available == tr.farm_available[::-1]


def test_value_antisymmetry(solution):
    """V_k(mirror s) = -V_k(s) for every reachable state; V_0(s_0) = 0 since s_0 is self-mirror."""
    for k, layer in enumerate(solution.layers):
        lay = set(layer)
        for s in layer:
            assert s.mirror() in lay
            assert abs(solution.V[k][s] + solution.V[k][s.mirror()]) < 1e-9
    assert abs(solution.value) < 1e-9


def test_terminal_handling_one_turn_game(cfg):
    """With K = 1 the value is the matrix game of immediate payoffs only (zero terminal bonus)."""
    g = LaneGame(cfg.with_(horizon=1))
    s = g.initial_state()
    sol = solve_equilibrium(g)
    la, lb = g.legal(s, 0), g.legal(s, 1)
    M = np.array([[g.transition(s, 0, a, b).payoff for b in lb] for a in la])
    assert abs(sol.value - solve_matrix_game(M).value) < 1e-12


def test_final_turn_crash_choices_take_resources(game, solution):
    """In the final turn vision has no future value, so every crash choice takes the most resources."""
    K = game.horizon
    seen = 0
    for key, x in solution.choice_x.items():
        k, pre, oa, ob = key
        if k == K - 1 and len(oa) > 1:
            seen += 1
            assert oa[int(np.argmax(x))][0] == max(o[0] for o in oa)
    assert seen > 0


def test_strategies_are_valid_distributions(solution):
    for k, layer in enumerate(solution.layers):
        for s in layer:
            for p in (solution.x[k][s], solution.y[k][s]):
                assert np.all(p >= 0) and abs(p.sum() - 1) < 1e-12
            la, lb = solution.legal[k][s]
            assert len(solution.x[k][s]) == len(la) and len(solution.y[k][s]) == len(lb)
    for key in solution.choice_x:
        for p in (solution.choice_x[key], solution.choice_y[key]):
            assert np.all(p >= 0) and abs(p.sum() - 1) < 1e-12


def test_transition_distribution_is_degenerate_and_valid(game, layers):
    """Phase-1 lane dynamics are deterministic: each joint action has exactly one successor
    lane (probability 1); gank probabilities are valid Bernoulli parameters."""
    for k, s, a, b in all_transitions(game, layers):
        tr = game.transition(s, k, a, b)
        assert all(0.0 <= p <= game.cfg.gank_max for p in tr.gank_p)
        assert len({s2 for *_, s2, _ in tr.choice_outcomes(game.cfg)}) <= len(tr.options_a) * len(tr.options_b)


def test_dynamics_are_periodic_in_k(cfg):
    """Transitions from the same state repeat with period phase_period in k (the basis of the cache)."""
    g = LaneGame(cfg)
    layers = g.reachable(horizon=10)
    P = cfg.phase_period
    for k in range(6):
        for s in layers[k][:40]:
            for a in g.legal(s, 0):
                for b in g.legal(s, 1):
                    assert step_turn(s, k, a, b, cfg).state == step_turn(s, k + P, a, b, cfg).state
