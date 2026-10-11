"""The optional variance reduction of the legacy configurations: `LaneEnv(gank_reward="expected")` pays the gank
term as its conditional expectation. With deterministic policies the legacy lane is deterministic, so every single
episode then returns exactly the exact game value; with stochastic policies the mean is unchanged and the variance
is lower. Champion mode refuses the option."""

import numpy as np
import pytest

from lanegame import GameConfig, LaneGame
from lanegame.baselines import aggressive, freeze_contest, passive_safe, uniform
from lanegame.best_response import evaluate
from lanegame.env import LaneEnv, ScriptedAgent, run_episodes

CFG = GameConfig(horizon=8)


@pytest.fixture(scope="module")
def exact():
    g = LaneGame(CFG)
    return g, g.reachable()


def test_every_episode_equals_the_exact_value_for_deterministic_pairs(exact):
    g, layers = exact
    for pa, pb in ((aggressive(), passive_safe()), (freeze_contest(), aggressive())):
        v = evaluate(g, layers, pa, pb)
        st = run_episodes(LaneEnv(CFG, gank_reward="expected"), ScriptedAgent(pa), ScriptedAgent(pb), 20, seed=1)
        assert np.allclose(st.payoffs, v, atol=1e-9)
        realized = run_episodes(LaneEnv(CFG), ScriptedAgent(pa), ScriptedAgent(pb), 20, seed=1)
        assert realized.payoffs.std() > 0                               # the default keeps the realized ganks


def test_mean_unchanged_and_variance_lower_for_stochastic_pairs(exact):
    g, layers = exact
    v = evaluate(g, layers, uniform(), passive_safe())
    e = run_episodes(LaneEnv(CFG, gank_reward="expected"), ScriptedAgent(uniform()), ScriptedAgent(passive_safe()), 1500, seed=2)
    r = run_episodes(LaneEnv(CFG), ScriptedAgent(uniform()), ScriptedAgent(passive_safe()), 1500, seed=2)
    assert abs(e.mean - v) < 4.5 * e.stderr and abs(r.mean - v) < 4.5 * r.stderr
    assert e.payoffs.std() < 0.5 * r.payoffs.std()


def test_option_is_refused_in_champion_mode():
    with pytest.raises(ValueError):
        LaneEnv(GameConfig.champions(), gank_reward="expected")
    with pytest.raises(ValueError):
        LaneEnv(CFG, gank_reward="sometimes")
