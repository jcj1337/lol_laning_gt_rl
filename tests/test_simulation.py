"""Monte Carlo simulation: agreement with exact expectations, payoff aggregation and reproducibility."""

import numpy as np
import pytest

from lanegame import GameConfig, LaneGame
from lanegame.analysis import path_statistics
from lanegame.baselines import (aggressive, always_shove, always_slowpush, defensive_freeze, freeze_contest,
                                passive_safe, stack_builder, uniform)
from lanegame.best_response import evaluate
from lanegame.simulate import simulate

PAIRS = {
    "v1": [("eq", "eq"), ("eq", uniform), ("uniform", always_shove), (defensive_freeze, always_slowpush),
           (always_shove, "eq")],
    "revised": [("eq", "eq"), ("eq", uniform), (aggressive, passive_safe), (freeze_contest, stack_builder),
                ("uniform", aggressive)],
    "siege": [("eq", "eq"), ("eq", uniform), (aggressive, passive_safe), (freeze_contest, stack_builder),
              (lambda: stack_builder(siege="vision"), passive_safe)],
}


def _resolve(p, solution, side):
    if p == "eq":
        return solution.policy(side)
    if p == "uniform":
        return uniform()
    return p()


@pytest.mark.parametrize("i", range(5))
def test_monte_carlo_matches_exact_value(model, game, layers, solution, i):
    pairs = PAIRS[model]
    pa, pb = _resolve(pairs[i][0], solution, 0), _resolve(pairs[i][1], solution, 1)
    exact = evaluate(game, layers, pa, pb)
    sim = simulate(game, pa, pb, episodes=3000, seed=2024 + i)
    # 4.5 standard errors: per-test false-alarm probability < 1e-5
    assert abs(sim.mean - exact) <= 4.5 * sim.stderr + 1e-12, (exact, sim.mean, sim.stderr)


def test_forward_statistics_match_backward_value(model, game, layers, solution, baselines):
    for pa, pb in [(baselines[0], baselines[1]), (solution.policy(0), baselines[3])]:
        ps = path_statistics(game, pa, pb)
        assert abs(ps.value - evaluate(game, layers, pa, pb)) < 1e-9
        # the expected total decomposes exactly into its components
        parts = (ps.farm[0] - ps.farm[1]) + (ps.crash_resources[0] - ps.crash_resources[1]) \
            + (ps.plates[0] - ps.plates[1]) - (ps.gank_loss[0] - ps.gank_loss[1])
        assert abs(ps.value - parts) < 1e-9
        assert all(0 <= c <= a + 1e-12 for c, a in zip(ps.farm, ps.farm_available))


def test_simulation_is_reproducible(game):
    a = simulate(game, uniform(), uniform(), episodes=200, seed=11)
    b = simulate(game, uniform(), uniform(), episodes=200, seed=11)
    c = simulate(game, uniform(), uniform(), episodes=200, seed=12)
    assert np.array_equal(a.payoffs, b.payoffs) and np.array_equal(a.ganks, b.ganks)
    assert np.array_equal(a.farm, b.farm)
    assert not np.array_equal(a.payoffs, c.payoffs)


@pytest.mark.parametrize("base", [GameConfig.v1(horizon=16), GameConfig.v2(horizon=16), GameConfig(horizon=16)],
                         ids=["v1", "revised", "siege"])
def test_simulation_without_randomness_equals_exact_value(base):
    """With gank risk switched off and deterministic policies, every episode equals the exact value, and the
    farm and crash resources each episode records equal the exact path totals."""
    cfg = base.with_(gank_base=0.0, gank_position=0.0, gank_action=(0.0,) * 6, gank_strength=0.0)
    g = LaneGame(cfg)
    layers = g.reachable()
    pairs = ([(always_shove(), always_slowpush()), (always_slowpush(), defensive_freeze())] if not cfg.positioning
             else [(aggressive(), passive_safe()), (stack_builder(), freeze_contest()),
                   (stack_builder(siege="vision"), passive_safe())])
    for pa, pb in pairs:
        sim = simulate(g, pa, pb, episodes=5, seed=0)
        assert np.all(sim.payoffs == sim.payoffs[0])
        assert abs(sim.payoffs[0] - evaluate(g, layers, pa, pb)) < 1e-12
        ps = path_statistics(g, pa, pb)
        assert np.allclose(sim.farm[0], ps.farm, atol=1e-12)
        assert np.allclose(sim.crash_resources[0], ps.crash_resources, atol=1e-12)
        assert np.allclose(sim.plates[0], ps.plates, atol=1e-12)
        assert abs(sim.payoffs[0] - ((ps.farm[0] - ps.farm[1]) + (ps.crash_resources[0] - ps.crash_resources[1])
                                     + (ps.plates[0] - ps.plates[1]))) < 1e-9
