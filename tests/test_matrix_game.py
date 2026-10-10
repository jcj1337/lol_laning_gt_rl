"""Matrix-game solver: known games, certificates and an independent reference solver."""

import numpy as np
import pytest

from lanegame.matrix_game import certify, solve_matrix_game, support_enumeration

TOL = 1e-9


def check_solution(M, sol):
    M = np.asarray(M, float)
    assert np.all(sol.x >= 0) and abs(sol.x.sum() - 1) < 1e-12        # primal feasibility
    assert np.all(sol.y >= 0) and abs(sol.y.sum() - 1) < 1e-12        # dual feasibility
    lo, up, gap = certify(M, sol.x, sol.y)
    assert -1e-12 <= gap <= TOL                                       # optimality residual
    assert lo - 1e-12 <= sol.value <= up + 1e-12


def test_matching_pennies():
    M = [[1, -1], [-1, 1]]
    sol = solve_matrix_game(M)
    check_solution(M, sol)
    assert abs(sol.value) < TOL
    assert np.allclose(sol.x, [0.5, 0.5]) and np.allclose(sol.y, [0.5, 0.5])


def test_rock_paper_scissors():
    M = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    sol = solve_matrix_game(M)
    check_solution(M, sol)
    assert abs(sol.value) < TOL
    assert np.allclose(sol.x, 1 / 3) and np.allclose(sol.y, 1 / 3)


def test_weighted_rps_known_solution():
    # Winner gets 2 for rock-scissors, 1 otherwise; equilibrium (1/4, 1/2, 1/4)... computed below.
    M = np.array([[0, -1, 2], [1, 0, -1], [-2, 1, 0]], float)
    sol = solve_matrix_game(M)
    check_solution(M, sol)
    # symmetric zero-sum game (M = -M^T) -> value 0
    assert abs(sol.value) < TOL
    # indifference: x^T M = 0 with x on simplex -> x = (1/4, 1/2, 1/4)
    assert np.allclose(sol.x, [0.25, 0.5, 0.25], atol=1e-9)


def test_asymmetric_2x2_closed_form():
    a, b, c, d = 3.0, -1.0, -2.0, 1.0           # no saddle point
    M = [[a, b], [c, d]]
    sol = solve_matrix_game(M)
    check_solution(M, sol)
    denom = a - b - c + d
    v = (a * d - b * c) / denom
    p = (d - c) / denom
    q = (d - b) / denom
    assert abs(sol.value - v) < TOL
    assert np.allclose(sol.x, [p, 1 - p]) and np.allclose(sol.y, [q, 1 - q])


def test_pure_saddle_point():
    M = [[4, 2, 5], [3, 1, 0], [6, 2.5, 7]]     # saddle at (row 2, col 1) value 2.5
    sol = solve_matrix_game(M)
    check_solution(M, sol)
    assert sol.method == "saddle"
    assert abs(sol.value - 2.5) < TOL


def test_dominated_row_gets_zero_weight():
    M = [[1, -1], [-1, 1], [-2, -2]]
    sol = solve_matrix_game(M)
    check_solution(M, sol)
    assert sol.x[2] < 1e-12


@pytest.mark.parametrize("shape", [(1, 3), (3, 1), (2, 3), (3, 2), (3, 3), (2, 2)])
def test_random_games_match_support_enumeration(shape):
    rng = np.random.default_rng(1234 + shape[0] * 10 + shape[1])
    for _ in range(200):
        M = rng.normal(size=shape).round(3)
        sol = solve_matrix_game(M)
        check_solution(M, sol)
        v_ref, x_ref, y_ref = support_enumeration(M)
        assert abs(sol.value - v_ref) < 1e-8


def test_invariance_to_affine_transformation():
    rng = np.random.default_rng(7)
    for _ in range(50):
        M = rng.normal(size=(3, 3))
        s1 = solve_matrix_game(M)
        s2 = solve_matrix_game(2.5 * M + 7.0)
        assert abs(s2.value - (2.5 * s1.value + 7.0)) < 1e-8


def test_degenerate_game_with_ties():
    M = np.zeros((3, 3))
    sol = solve_matrix_game(M)
    check_solution(M, sol)
    assert sol.value == 0.0
