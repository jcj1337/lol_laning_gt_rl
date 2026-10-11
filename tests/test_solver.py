"""Equilibrium solver, best responses and exploitability, with independent checks (both models)."""

import numpy as np
import pytest

from lanegame import GameConfig, LaneGame
from lanegame.best_response import best_response_value, evaluate, exploitability
from lanegame.game import apply_choice
from lanegame.matrix_game import support_enumeration
from lanegame.policy import ScriptedPolicy
from lanegame.solver import solve_equilibrium
from lanegame.state import State


def tree_minimax(game, s, k):
    """Independent reference: plain recursion over the game TREE (no state merging, no caching,
    no layer bookkeeping), solving every stage and choice game by support enumeration."""
    if k == game.horizon:
        return 0.0
    cfg = game.cfg
    la, lb = game.legal(s, 0), game.legal(s, 1)
    Q = np.zeros((len(la), len(lb)))
    for i, a in enumerate(la):
        for j, b in enumerate(lb):
            tr = game.transition(s, k, a, b)
            C = np.zeros((len(tr.options_a), len(tr.options_b)))
            for ii, oa in enumerate(tr.options_a):
                for jj, ob in enumerate(tr.options_b):
                    s2, r = apply_choice(tr.pre_state, oa, ob, cfg)
                    C[ii, jj] = r + tree_minimax(game, s2, k + 1)
            Q[i, j] = tr.payoff + support_enumeration(C)[0]
    return support_enumeration(Q)[0]


V1_ROOTS = [
    (None, 0, 3),
    (State(0, 6, (-4,), 0, -4, 2, (), 0, 0, 0), 2, 3),     # A 2-stack at the river
    (State(1, 4, (), 0, -3, 2, (), 0, 0, 0), 1, 3),        # A medium wave near B's tower
    (State(0, 0, (-2,), 0, 1, 2, (), 0, 0, 0), 3, 3),      # B wave held at A's freeze line
    (State(-4, 2, (), 2, -3, 2, (), 0, 1, 0), 2, 3),       # two-packet siege, A has vision
]
REV_ROOTS = [
    (None, 0, 2),
    (State(0, 12, (-4,), 0, -4, 4, (), 0, 0, 0), 2, 2),    # A 3-wave stack at the river
    (State(1, 8, (), 0, -3, 4, (), 0, 0, 0), 1, 2),        # A 2-wave packet near B's tower (crash + choices)
    (State(-4, 4, (), 0, 1, 4, (), 0, 0, 0), 3, 2),        # B wave at A's freeze line (6x4 / 6x6 stage games)
    (State(-4, 4, (), 9, -3, 4, (), 0, 1, 0), 2, 2),       # crashed mass under B's tower, A has vision
]


@pytest.mark.parametrize("root,k0,depth", V1_ROOTS)
def test_backward_induction_matches_tree_minimax_v1(root, k0, depth):
    g = LaneGame(GameConfig.v1(horizon=k0 + depth))
    sol = solve_equilibrium(g, root=root, k0=k0)
    s = g.initial_state() if root is None else root
    assert abs(sol.value - tree_minimax(g, s, k0)) < 1e-8


@pytest.mark.parametrize("root,k0,depth", REV_ROOTS)
def test_backward_induction_matches_tree_minimax_revised(root, k0, depth):
    g = LaneGame(GameConfig(horizon=k0 + depth))
    sol = solve_equilibrium(g, root=root, k0=k0)
    s = g.initial_state() if root is None else root
    assert abs(sol.value - tree_minimax(g, s, k0)) < 1e-8


def test_certified_optimality_residuals(solution):
    assert solution.stats.max_gap <= 1e-9
    assert all(g <= 1e-9 for layer in solution.gaps for g in layer.values())


def test_equilibrium_policies_have_zero_exploitability(game, layers, solution):
    v = solution.value
    ea = exploitability(game, layers, solution.policy(0), 0, v)
    eb = exploitability(game, layers, solution.policy(1), 1, v)
    assert abs(ea.exploitability) < 1e-8 and abs(eb.exploitability) < 1e-8
    assert abs(evaluate(game, layers, solution.policy(0), solution.policy(1)) - v) < 1e-8


def test_equilibrium_is_subgame_perfect(game, layers, solution):
    """Exploitability is ~0 from every reachable state, not just the root."""
    br_b = best_response_value(game, layers, solution.policy(0), 0)
    br_a = best_response_value(game, layers, solution.policy(1), 1)
    for k, layer in enumerate(layers):
        for s in layer:
            assert abs(br_b.V[k][s] - solution.V[k][s]) < 1e-8
            assert abs(br_a.V[k][s] - solution.V[k][s]) < 1e-8


def _random_policy(seed):
    rng = np.random.default_rng(seed)
    cache = {}

    def rule(k, st, legal):
        key = (k, st)
        if key not in cache:
            cache[key] = rng.dirichlet(np.ones(len(legal)))
        return cache[key]

    def chooser(k, st, opts_self, opts_opp):
        key = ("c", k, st, opts_self, opts_opp)
        if key not in cache:
            cache[key] = rng.dirichlet(np.ones(len(opts_self)))
        return cache[key]
    return ScriptedPolicy(f"random{seed}", rule, chooser)


@pytest.mark.parametrize("side", [0, 1])
def test_exploitability_is_nonnegative(game, layers, solution, baselines, side):
    pols = baselines + [_random_policy(s) for s in range(5)]
    for p in pols:
        e = exploitability(game, layers, p, side, solution.value)
        assert e.exploitability >= -1e-9, (p.name, e)


def test_best_response_dominates_every_scripted_alternative(game, layers, solution, baselines):
    """The best response to a fixed policy is at least as good as any scripted reply, and evaluating it
    as a policy reproduces its value."""
    for pa in baselines:
        br = best_response_value(game, layers, pa, 0)
        w = br.value(game)
        for pb in baselines:
            assert w <= evaluate(game, layers, pa, pb) + 1e-9
        assert abs(evaluate(game, layers, pa, br.policy) - w) < 1e-9


def test_best_response_against_mixed_equilibrium_is_indifferent(game, layers, solution):
    """Against the equilibrium policy every action in the opponent's support earns the value."""
    for k, layer in enumerate(layers):
        for s in layer:
            Q = solution.Q[k][s]
            x, y = solution.x[k][s], solution.y[k][s]
            v = solution.V[k][s]
            row = Q @ y
            col = x @ Q
            assert np.all(row <= v + 1e-9) and np.all(col >= v - 1e-9)
            assert np.all(np.abs(row[x > 1e-9] - v) < 1e-8)
            assert np.all(np.abs(col[y > 1e-9] - v) < 1e-8)


def test_solver_handles_different_action_set_sizes(model, solution):
    sizes = {Q.shape for layer in solution.Q for Q in layer.values()}
    if model == "v1":
        assert (2, 2) in sizes and ((3, 2) in sizes or (2, 3) in sizes) and (3, 3) in sizes
    else:
        # Under the revised Freeze rule both players can never freeze at once (each would need the other's
        # wave on its own half, i.e. crossed fronts), so the largest stage game is 6 x 4.
        assert sizes == {(4, 4), (4, 6), (6, 4)}


def test_solver_on_full_6x6_stage_games():
    """With the original Freeze legality (enemy at the river or closer) both players can freeze at the river,
    so 6 x 6 stage games occur; the solver handles them with certified gaps and zero exploitability."""
    g = LaneGame(GameConfig(horizon=10, freeze_legal_max=0))
    sol = solve_equilibrium(g)
    sizes = {Q.shape for layer in sol.Q for Q in layer.values()}
    assert (6, 6) in sizes and sol.stats.max_gap <= 1e-9
    assert sol.stats.mixed_states > 0
    assert abs(exploitability(g, sol.layers, sol.policy(0), 0, sol.value).exploitability) < 1e-8


def test_stage_games_match_support_enumeration_sample(solution):
    """A sample of the solver's stage games re-solved by the LP-free reference method."""
    games = [(k, s) for k, layer in enumerate(solution.layers) for s in layer]
    rng = np.random.default_rng(0)
    for idx in rng.choice(len(games), size=min(150, len(games)), replace=False):
        k, s = games[idx]
        assert abs(support_enumeration(solution.Q[k][s])[0] - solution.V[k][s]) < 1e-8
