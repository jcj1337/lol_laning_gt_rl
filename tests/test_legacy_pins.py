"""Legacy pins (M4 plan WP0, validation stage S0).

The digests below were computed on the unchanged engine before any M4 code existed (branch design/m4-integrated,
085908d). They cover every legacy ruleset: v1, v2, the siege default and champions() (lane steps, observations,
champion resolution with chance vectors, a 200-episode scripted rollout, an M3b checkpoint's head-to-head) and the
exact solver (K = 16 values of the three conftest models; the K = 32 siege solution's metadata). A failing pin means
a legacy output changed: that is a regression, never a reason to update the digest.
"""

import json
import os

import pytest

import legacy_pins as pins
from lanegame import GameConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANE = {
    "v1": "1862f1edd2fd4859c867f10952e0bacbf495f5453665fd5c0f07f61b34ddef51",
    "v2": "fade3f765eff13fc2954f3d0f9a6b45337b4899ff054b1b62aeb56103cc33842",
    "siege": "66a31e73004f1a43c8ffefa529a9ad1e151b8a68aa0f653f90000de5ed5a29aa",
    "champions": "575ea950debe0bdab5a8b2943ec086fc7a27b49b3f368c59e86c9120256e08e6",
}
CHAMPION = "d659fc4b0e0bea89f1c2a3350069b68084a011c7a721d96341b47b3f1dc65fe3"
ROLLOUT = "d8b25c95bd94c2aae07a3618949c9f240a9187392f49d681234f86c8f8edcdc9"
CHECKPOINT = "4c98400ad55a75b4f4911657a7b66519896d84ad4afdd06c2d958eb0028ede78"
SOLVER_K16 = [(0.0, 337, "4e275ffa168bbf36c4608cc370016c740c96a1ca9988ff5d45b7a21632ff8875"),
              (0.0, 1799, "218c4087f1a35e235b80abbb68d98e9a5712f2497b793cae0cadb789170feff1"),
              (0.0, 4951, "5ecd2821f6775c7449457c5c04b0903bda9230fece8618ebdd7940439f484f43")]


@pytest.mark.parametrize("name", list(LANE))
def test_lane_steps_and_observations_are_pinned(name):
    assert pins.lane_digest(name) == LANE[name]


def test_champion_resolution_and_observations_are_pinned():
    assert pins.champion_digest() == CHAMPION


def test_champion_rollout_is_pinned():
    assert pins.rollout_digest() == ROLLOUT


def test_m3b_checkpoint_head_to_head_is_pinned():
    path = os.path.join(ROOT, pins.M3B_CHECKPOINT)
    if not os.path.exists(path):
        pytest.skip("M3b checkpoint not present (git-ignored)")
    pytest.importorskip("torch")
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        assert pins.checkpoint_digest() == CHECKPOINT
    finally:
        os.chdir(cwd)


def test_solver_k16_values_are_pinned(solution, model):
    i = {"v1": 0, "revised": 1, "siege": 2}[model]
    value, n_states, digest = SOLVER_K16[i]
    assert solution.value == pytest.approx(value, abs=1e-12)
    assert sum(len(x) for x in solution.layers) == n_states
    assert pins.solution_digest(solution) == digest


def test_siege_k32_solution_metadata_is_pinned():
    with open(os.path.join(ROOT, "results", "siege", "solution_K32.json"), encoding="utf-8") as f:
        d = json.load(f)
    cur = GameConfig(horizon=32).to_dict()
    assert all(cur[k] == v for k, v in d["config"].items()), "the K = 32 siege configuration changed"
    assert d["value"] == 0.0 and d["stats"]["states"] == 172907


@pytest.mark.slow
def test_siege_k32_solution_resolves_identically():
    from lanegame import LaneGame
    from lanegame.solver import solve_equilibrium
    with open(os.path.join(ROOT, "results", "siege", "solution_K32.json"), encoding="utf-8") as f:
        d = json.load(f)
    g = LaneGame(GameConfig(horizon=32))
    layers = g.reachable()
    sol = solve_equilibrium(g, layers)
    assert sum(len(x) for x in layers) == d["stats"]["states"]
    assert sol.value == pytest.approx(d["value"], abs=1e-9)
