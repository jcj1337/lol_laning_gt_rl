"""Monte Carlo simulation: agreement with exact expectations and reproducibility."""

import numpy as np
import pytest

from lanegame import GameConfig, LaneGame
from lanegame.analysis import path_statistics
from lanegame.baselines import always_shove, always_slowpush, defensive_freeze, uniform
from lanegame.best_response import evaluate
from lanegame.simulate import simulate


@pytest.mark.parametrize("pair", [
    ("eq", "eq"), ("eq", "uniform"), ("uniform", "always_shove"),
    ("defensive_freeze", "always_slowpush"),
])
def test_monte_carlo_matches_exact_value(game, layers, solution, pair):
    pols = {"eq": None, "uniform": uniform(), "always_shove": always_shove(),
            "defensive_freeze": defensive_freeze(), "always_slowpush": always_slowpush()}
    pa = solution.policy(0) if pair[0] == "eq" else pols[pair[0]]
    pb = solution.policy(1) if pair[1] == "eq" else pols[pair[1]]
    exact = evaluate(game, layers, pa, pb)
    sim = simulate(game, pa, pb, episodes=3000, seed=2024)
    # 4.5 standard errors: per-test false-alarm probability < 1e-5
    assert abs(sim.mean - exact) <= 4.5 * sim.stderr + 1e-12, (exact, sim.mean, sim.stderr)


def test_forward_statistics_match_backward_value(game, layers, solution):
    for pa, pb in [(uniform(), always_shove()), (solution.policy(0), defensive_freeze())]:
        assert abs(path_statistics(game, pa, pb).value - evaluate(game, layers, pa, pb)) < 1e-9


def test_simulation_is_reproducible(game):
    a = simulate(game, uniform(), uniform(), episodes=200, seed=11)
    b = simulate(game, uniform(), uniform(), episodes=200, seed=11)
    c = simulate(game, uniform(), uniform(), episodes=200, seed=12)
    assert np.array_equal(a.payoffs, b.payoffs) and np.array_equal(a.ganks, b.ganks)
    assert not np.array_equal(a.payoffs, c.payoffs)


def test_simulation_without_randomness_equals_exact_value():
    """With gank risk switched off and deterministic policies, every episode equals the exact value."""
    cfg = GameConfig(horizon=16, gank_base=0.0, gank_position=0.0, gank_action=(0.0, 0.0, 0.0),
                     gank_strength=0.0)
    g = LaneGame(cfg)
    layers = g.reachable()
    for pa, pb in [(always_shove(), always_slowpush()), (always_slowpush(), defensive_freeze())]:
        sim = simulate(g, pa, pb, episodes=5, seed=0)
        assert np.all(sim.payoffs == sim.payoffs[0])
        assert abs(sim.payoffs[0] - evaluate(g, layers, pa, pb)) < 1e-12
