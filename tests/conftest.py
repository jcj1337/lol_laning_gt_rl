"""Shared fixtures.

Two model generations are tested:

  v1       GameConfig.v1(): the original Phase 1 model (kept bit-for-bit reproducible)
  revised  GameConfig():    the revised model (farming, Contest/Safe, mass-based tower fire)

Generic invariants use the parametrized fixtures `cfg`, `game`, `layers`,
`solution`, which run every test once per model. Model-specific tests use the
`v1_*` or `rev_*` fixtures.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lanegame import GameConfig, LaneGame  # noqa: E402
from lanegame.baselines import all_baselines, v1_baselines  # noqa: E402
from lanegame.solver import solve_equilibrium  # noqa: E402

HORIZON = 16
MODELS = {"v1": GameConfig.v1(horizon=HORIZON), "revised": GameConfig(horizon=HORIZON)}
_CACHE = {}


def _built(name):
    if name not in _CACHE:
        g = LaneGame(MODELS[name])
        layers = g.reachable()
        _CACHE[name] = (g, layers, solve_equilibrium(g, layers))
    return _CACHE[name]


@pytest.fixture(scope="session", params=list(MODELS))
def model(request):
    return request.param


@pytest.fixture(scope="session")
def cfg(model):
    return MODELS[model]


@pytest.fixture(scope="session")
def game(model):
    return _built(model)[0]


@pytest.fixture(scope="session")
def layers(model):
    return _built(model)[1]


@pytest.fixture(scope="session")
def solution(model):
    return _built(model)[2]


@pytest.fixture(scope="session")
def baselines(model):
    return v1_baselines() if model == "v1" else all_baselines()


@pytest.fixture(scope="session")
def v1_cfg():
    return MODELS["v1"]


@pytest.fixture(scope="session")
def v1_game():
    return _built("v1")[0]


@pytest.fixture(scope="session")
def v1_layers():
    return _built("v1")[1]


@pytest.fixture(scope="session")
def rev_cfg():
    return MODELS["revised"]


@pytest.fixture(scope="session")
def rev_game():
    return _built("revised")[0]


@pytest.fixture(scope="session")
def rev_layers():
    return _built("revised")[1]


@pytest.fixture(scope="session")
def rev_solution():
    return _built("revised")[2]


def all_transitions(game, layers):
    """Yield (k, state, a, b) for every reachable state and legal joint action."""
    for k, layer in enumerate(layers):
        for s in layer:
            for a in game.legal(s, 0):
                for b in game.legal(s, 1):
                    yield k, s, a, b
