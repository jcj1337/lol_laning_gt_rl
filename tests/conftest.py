"""Shared fixtures.

Three model generations are tested:

  v1       GameConfig.v1(): the original Phase 1 model (kept bit-for-bit reproducible)
  revised  GameConfig.v2(): the previous revision (farming, Contest/Safe, post-crash choice)
  siege    GameConfig():    the siege revision (Apply Pressure / Establish Vision, plates, thinning)

Generic invariants use the parametrized fixtures `cfg`, `game`, `layers`,
`solution`, which run every test once per model. Model-specific tests use the
`v1_*`, `rev_*` or `sg_*` fixtures; `econ_*` fixtures run a test on both the
revised and the siege model (the two models with a farming economy).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lanegame import GameConfig, LaneGame  # noqa: E402
from lanegame.baselines import all_baselines, siege_baselines, v1_baselines  # noqa: E402
from lanegame.solver import solve_equilibrium  # noqa: E402

HORIZON = 16
MODELS = {"v1": GameConfig.v1(horizon=HORIZON), "revised": GameConfig.v2(horizon=HORIZON),
          "siege": GameConfig(horizon=HORIZON)}
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
    if model == "v1":
        return v1_baselines()
    return all_baselines() if model == "revised" else siege_baselines()


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


@pytest.fixture(scope="session")
def sg_cfg():
    return MODELS["siege"]


@pytest.fixture(scope="session")
def sg_game():
    return _built("siege")[0]


@pytest.fixture(scope="session")
def sg_layers():
    return _built("siege")[1]


@pytest.fixture(scope="session")
def sg_solution():
    return _built("siege")[2]


@pytest.fixture(scope="session", params=["revised", "siege"])
def econ_model(request):
    return request.param


@pytest.fixture(scope="session")
def econ_cfg(econ_model):
    return MODELS[econ_model]


@pytest.fixture(scope="session")
def econ_game(econ_model):
    return _built(econ_model)[0]


@pytest.fixture(scope="session")
def econ_layers(econ_model):
    return _built(econ_model)[1]


def all_transitions(game, layers):
    """Yield (k, state, a, b) for every reachable state and legal joint action."""
    for k, layer in enumerate(layers):
        for s in layer:
            for a in game.legal(s, 0):
                for b in game.legal(s, 1):
                    yield k, s, a, b
