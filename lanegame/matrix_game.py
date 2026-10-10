"""Two-player zero-sum matrix games: exact-in-exact-arithmetic shortcuts plus LP.

For a payoff matrix M (row player maximizes x^T M y, column player minimizes):

    v* = max_{x in simplex} min_j (x^T M)_j = min_{y in simplex} max_i (M y)_i

Solution path:
  1. degenerate shapes (one row or one column): pure optimization;
  2. pure saddle point check: maximin == minimax;
  3. otherwise the row-player LP  max v  s.t.  M^T x >= v 1,  1^T x = 1,  x >= 0
     solved with scipy.optimize.linprog (HiGHS); the column strategy is read
     from the LP duals and, if the duals fail the optimality check, from the
     column-player LP.

Every solution is certified a posteriori:

    lower = min_j (x^T M)_j   (what x guarantees)
    upper = max_i (M y)_i     (what y concedes)
    gap   = upper - lower >= 0, and gap == 0 iff (x, y) is an equilibrium.

The returned value is (lower + upper) / 2 and |value - v*| <= gap / 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

PROB_TOL = 1e-12


@dataclass
class MatrixSolution:
    value: float
    x: np.ndarray          # row (maximizer) mixed strategy
    y: np.ndarray          # column (minimizer) mixed strategy
    gap: float             # certified duality gap max(My) - min(x^T M)
    method: str            # "trivial" | "saddle" | "lp"


def _clean(p: np.ndarray) -> np.ndarray:
    p = np.where(p < PROB_TOL, 0.0, p)
    s = p.sum()
    return p / s


def certify(M: np.ndarray, x: np.ndarray, y: np.ndarray):
    lower = float(np.min(x @ M))
    upper = float(np.max(M @ y))
    return lower, upper, upper - lower


def _lp_row(M: np.ndarray):
    """Row player's LP; returns (x, dual-derived y)."""
    m, n = M.shape
    # variables z = (x_1..x_m, v); minimize -v
    c = np.zeros(m + 1)
    c[-1] = -1.0
    A_ub = np.hstack([-M.T, np.ones((n, 1))])        # v - (M^T x)_j <= 0
    b_ub = np.zeros(n)
    A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
    b_eq = np.ones(1)
    bounds = [(0, None)] * m + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs")
    if res.status != 0:
        raise RuntimeError(f"row LP failed: {res.message}")
    x = _clean(np.maximum(res.x[:m], 0.0))
    y = np.maximum(-np.asarray(res.ineqlin.marginals), 0.0)
    y = _clean(y) if y.sum() > 0 else np.full(n, 1.0 / n)
    return x, y


def _lp_col(M: np.ndarray) -> np.ndarray:
    m, n = M.shape
    c = np.zeros(n + 1)
    c[-1] = 1.0                                      # minimize u
    A_ub = np.hstack([M, -np.ones((m, 1))])          # (M y)_i - u <= 0
    b_ub = np.zeros(m)
    A_eq = np.hstack([np.ones((1, n)), np.zeros((1, 1))])
    bounds = [(0, None)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.ones(1), bounds=bounds,
                  method="highs")
    if res.status != 0:
        raise RuntimeError(f"column LP failed: {res.message}")
    return _clean(np.maximum(res.x[:n], 0.0))


def solve_matrix_game(M, tol: float = 1e-9) -> MatrixSolution:
    M = np.asarray(M, dtype=float)
    m, n = M.shape

    if m == 1 or n == 1:
        x = np.zeros(m)
        y = np.zeros(n)
        if m == 1:
            x[0] = 1.0
            y[int(np.argmin(M[0]))] = 1.0
        else:
            y[0] = 1.0
            x[int(np.argmax(M[:, 0]))] = 1.0
        lo, up, gap = certify(M, x, y)
        return MatrixSolution((lo + up) / 2, x, y, gap, "trivial")

    row_mins = M.min(axis=1)
    col_maxs = M.max(axis=0)
    i = int(np.argmax(row_mins))
    j = int(np.argmin(col_maxs))
    if col_maxs[j] - row_mins[i] <= 0.0:              # exact pure saddle point
        x = np.zeros(m)
        y = np.zeros(n)
        x[i] = 1.0
        y[j] = 1.0
        lo, up, gap = certify(M, x, y)
        return MatrixSolution((lo + up) / 2, x, y, gap, "saddle")

    # Shift for numerical conditioning (the game is invariant to adding constants).
    shift = float(M.mean())
    Ms = M - shift
    x, y = _lp_row(Ms)
    lo, up, gap = certify(M, x, y)
    if gap > tol:
        y = _lp_col(Ms)
        lo, up, gap = certify(M, x, y)
    return MatrixSolution((lo + up) / 2, x, y, gap, "lp")


def support_enumeration(M, tol: float = 1e-10):
    """Independent reference solver: enumerate equal-size supports (small games only).

    Returns (value, x, y) for the first equilibrium found. Every zero-sum game
    has an equilibrium with |supp x| == |supp y| whose payoffs equalize on the
    supports, so exhaustive enumeration finds one. Exponential: use for tests.
    """
    from itertools import combinations
    M = np.asarray(M, dtype=float)
    m, n = M.shape
    for k in range(1, min(m, n) + 1):
        for I in combinations(range(m), k):
            for J in combinations(range(n), k):
                # unknowns: x_I (k), v ; equations: (x^T M)_j = v for j in J, sum x = 1
                Ax = np.zeros((k + 1, k + 1))
                Ax[:k, :k] = M[np.ix_(I, J)].T
                Ax[:k, k] = -1.0
                Ax[k, :k] = 1.0
                Ay = np.zeros((k + 1, k + 1))
                Ay[:k, :k] = M[np.ix_(I, J)]
                Ay[:k, k] = -1.0
                Ay[k, :k] = 1.0
                rhs = np.zeros(k + 1)
                rhs[k] = 1.0
                try:
                    sx = np.linalg.solve(Ax, rhs)
                    sy = np.linalg.solve(Ay, rhs)
                except np.linalg.LinAlgError:
                    continue
                if np.any(sx[:k] < -tol) or np.any(sy[:k] < -tol):
                    continue
                x = np.zeros(m)
                y = np.zeros(n)
                x[list(I)] = sx[:k]
                y[list(J)] = sy[:k]
                lo, up, gap = certify(M, x, y)
                if gap <= 1e-8:
                    return (lo + up) / 2, x, y
    raise RuntimeError("no equilibrium found by support enumeration")
