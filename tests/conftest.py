import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lanegame import GameConfig, LaneGame  # noqa: E402
from lanegame.solver import solve_equilibrium  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return GameConfig(horizon=16)


@pytest.fixture(scope="session")
def game(cfg):
    return LaneGame(cfg)


@pytest.fixture(scope="session")
def layers(game):
    return game.reachable()


@pytest.fixture(scope="session")
def solution(game, layers):
    return solve_equilibrium(game, layers)


def all_transitions(game, layers):
    """Yield (k, state, a, b) for every reachable state and legal joint action."""
    for k, layer in enumerate(layers):
        for s in layer:
            for a in game.legal(s, 0):
                for b in game.legal(s, 1):
                    yield k, s, a, b
