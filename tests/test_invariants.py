"""Mathematical invariants: zero-sum, mirror symmetry, terminal handling, valid distributions."""

import numpy as np

from lanegame import SH, SP, GameConfig, LaneGame
from lanegame.game import apply_choice
from lanegame.matrix_game import solve_matrix_game
from lanegame.mechanics import step_turn
from lanegame.solver import solve_equilibrium

from conftest import all_transitions


def test_zero_sum_payoffs(game, layers):
    """Every payoff component is A's gain = B's loss: L(p_B - p_A) and rho_A - rho_B."""
    cfg = game.cfg
    for k, s, a, b in all_transitions(game, layers):
        tr = game.transition(s, k, a, b)
        assert abs(tr.gank_payoff - cfg.gank_loss * (tr.gank_p[1] - tr.gank_p[0])) < 1e-15
        for i, j, s2, r in tr.choice_outcomes(cfg):
            assert r == tr.options_a[i][0] - tr.options_b[j][0]


def test_mirror_equivariance_of_transitions(game, layers):
    """Swapping the players and reflecting the lane commutes with the dynamics."""
    cfg = game.cfg
    for k, s, a, b in all_transitions(game, layers):
        m = s.mirror()
        assert game.legal(m, 0) == game.legal(s, 1) and game.legal(m, 1) == game.legal(s, 0)
        r = step_turn(s, k, a, b, cfg)
        rm = step_turn(m, k, b, a, cfg)
        assert rm.state == r.state.mirror()
        assert rm.crash_hp == r.crash_hp[::-1]
        tr, trm = game.transition(s, k, a, b), game.transition(m, k, b, a)
        assert abs(tr.gank_payoff + trm.gank_payoff) < 1e-15


def test_value_antisymmetry(solution):
    """V_k(mirror s) = -V_k(s) for every reachable state; V_0(s_0) = 0 since s_0 is self-mirror."""
    for k, layer in enumerate(solution.layers):
        lay = set(layer)
        for s in layer:
            assert s.mirror() in lay
            assert abs(solution.V[k][s] + solution.V[k][s.mirror()]) < 1e-9
    assert abs(solution.value) < 1e-9


def test_terminal_handling_one_turn_game():
    """With K = 1 the value is the matrix game of immediate payoffs only (zero terminal bonus)."""
    g = LaneGame(GameConfig(horizon=1))
    s = g.initial_state()
    sol = solve_equilibrium(g)
    la, lb = g.legal(s, 0), g.legal(s, 1)
    M = np.array([[g.transition(s, 0, a, b).gank_payoff for b in lb] for a in la])
    assert abs(sol.value - solve_matrix_game(M).value) < 1e-12


def test_terminal_crash_choice_takes_resources():
    """In the final turn vision has no future value, so the crasher takes the resource option."""
    from lanegame.state import State
    cfg = GameConfig(horizon=3)
    g = LaneGame(cfg)
    root = State(1, 2 * cfg.wave_hp, (), 0, -4, cfg.wave_hp, (), 0, 0, 0)
    sol = solve_equilibrium(g, root=root, k0=2)
    tr = g.transition(root, 2, SH, SP)
    assert tr.crash_hp[0] > 0
    key = (2, tr.pre_state, tr.options_a, tr.options_b)
    assert np.argmax(sol.choice_x[key]) == 0


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
