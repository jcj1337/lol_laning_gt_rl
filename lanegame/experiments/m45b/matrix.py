"""Priority E: integrated strategy payoff matrices and restricted-game equilibria.

A matrix game: seat A plays strategy i, seat B strategy j, from one start, on the same DEV chance seeds for every
cell (common random numbers); the payoff is A's discounted gold lead G_h (h = 200 by default; the RL objective,
truncated) averaged over the seeds. From a symmetric start with equal power the game is symmetric, so the matrix is
antisymmetric in expectation; `solve` reports the raw matrix and its antisymmetrized version (M - M^T) / 2, which has
the same expectation and less noise. Reported: the zero-sum value and equilibrium supports (`matrix_game`), every
strategy's best response, the row strategies weakly dominated within the set, the rows with the largest worst-case
payoff (maximin), and per-cell ledgers (deaths, Recalls, wards, minion fire, freezes, plates, positions) for the
interpretation.

These are restricted games: values and equilibria hold only within the strategy set; a strategy outside it may beat
every one of them.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ...matrix_game import solve_matrix_game
from . import common as Cm
from . import engine as E
from . import states as S
from . import strategies as ST

SET = ("g_farm", "g_thin", "g_slow", "g_shove", "g_shove_sr", "grow_shove4", "grow_crash_recall4", "g_guard",
       "g_bully", "shove_hunt4", "g_safe", "g_wary", "g_yield", "g_ward", "g_shove_ward", "g_guard_ward")
# the integrated set: SET plus the Shove + selective-trade hybrids and the freeze-phase strategies (M4.5b)
SET_FULL = SET + ("g_shove_guard", "g_shove_edge", "g_shove_wary", "g_slow_guard", "fz_guard_sh", "fz_passive_sh",
                  "wk_shove_reset_sh", "wk_wary_shove_sh")
LEDGER = ("lead", "deaths_A", "deaths_B", "recalls_A", "recalls_B", "wards_A", "ward_placed_A", "ward_intr_A",
          "d_min_A", "d_min_B", "d_gank_A", "d_gank_B", "held_A", "held_B", "plates_A", "plates_B", "crashes_A",
          "crashes_B", "trade_conn_A", "trade_conn_B", "xp_lead", "farm_A", "farm_B", "unc_A", "unc_B", "away_A",
          "away_B", "n_safe_A", "n_SH_A", "t_pos_A", "t_ret_A")


def start_of(c, name: str, gap: str = "equal"):
    """A named matrix start: s0, or a zoning / Vision situation at level 6 with a power gap."""
    if name == "s0":
        return S.s0(c)
    if name in S.ZONING_SITUATIONS:
        return S.zoning_state(c, name, 6, gap)
    if name in S.VISION_SITUATIONS:
        return S.vision_state(c, name, 6, gap)
    raise ValueError(name)


def _cell(args):
    base, over, start_name, gap, a, b, seeds, turns, metric = args
    c = Cm.cfg(base, **dict(over))
    start = start_of(c, start_name, gap)
    rows = [E.play(c, start, ST.get(a), ST.get(b), sd, turns)["summary"] for sd in seeds]
    out = {"a": a, "b": b, "payoff": float(np.mean([r[metric] for r in rows])),
           "se": float(np.std([r[metric] for r in rows], ddof=1) / np.sqrt(len(rows))),
           "per_seed": [r[metric] for r in rows]}
    for k in LEDGER + ("G20", "G50", "G100"):
        if k in rows[0]:
            out[k] = float(np.mean([r[k] for r in rows]))
    return out


def run(base: str = "r2", over=(), start: str = "s0", gap: str = "equal", strategies: Sequence[str] = SET,
        seeds: Sequence[int] = Cm.DEV[:48], turns: int = 200, metric: str = "G200",
        workers: int = Cm.WORKERS) -> dict:
    jobs = [(base, tuple(over), start, gap, a, b, tuple(seeds), turns, metric) for a in strategies for b in strategies]
    cells = Cm.run_jobs(_cell, jobs, workers, f"matrix {base} {start} {gap}")
    n = len(strategies)
    M = np.array([[cells[i * n + j]["payoff"] for j in range(n)] for i in range(n)])
    return {"base": base, "overrides": dict(over), "start": start, "gap": gap, "strategies": list(strategies),
            "seeds": [seeds[0], seeds[-1], len(seeds)], "turns": turns, "metric": metric, "M": M.tolist(),
            "cells": [{k: v for k, v in c.items() if k != "per_seed"} for c in cells],
            "per_seed": [c["per_seed"] for c in cells], "analysis": solve(M, strategies, gap == "equal" and
                                                                           start == "s0")}


def solve(M, names: Sequence[str], symmetric: bool) -> dict:
    M = np.asarray(M, dtype=float)
    out = {}
    mats = {"raw": M}
    if symmetric:
        mats["antisym"] = (M - M.T) / 2
    for tag, X in mats.items():
        sol = solve_matrix_game(X)
        n = len(names)
        br_b = {names[i]: names[int(np.argmin(X[i]))] for i in range(n)}       # B's best reply to each A row
        br_a = {names[j]: names[int(np.argmax(X[:, j]))] for j in range(n)}     # A's best reply to each B column
        dominated = [names[i] for i in range(n) if any(np.all(X[k] >= X[i]) and np.any(X[k] > X[i])
                                                       for k in range(n) if k != i)]
        maximin = sorted(((float(X[i].min()), names[i]) for i in range(n)), reverse=True)[:5]
        out[tag] = {"value": float(sol.value), "A": {names[i]: float(x) for i, x in enumerate(sol.x) if x > 1e-4},
                    "B": {names[j]: float(y) for j, y in enumerate(sol.y) if y > 1e-4}, "B_best_reply": br_b,
                    "A_best_reply": br_a, "dominated_rows": dominated, "maximin": maximin,
                    "row_mean": {names[i]: float(X[i].mean()) for i in range(n)}}
    return out


def equilibrium_ci(res: dict, boots: int = 200, rng_seed: int = 0) -> dict:
    """Bootstrap over seeds (resampling the same seed indices in every cell, which keeps the pairing): how often
    each strategy is in the equilibrium support of the (antisymmetrized when symmetric) matrix."""
    rng = np.random.default_rng(rng_seed)
    names = res["strategies"]
    n = len(names)
    P = np.array(res["per_seed"])                       # (n*n, seeds)
    m = P.shape[1]
    sym = res["gap"] == "equal" and res["start"] == "s0"
    counts = {x: 0 for x in names}
    mass = {x: 0.0 for x in names}
    vals = []
    for _ in range(boots):
        idx = rng.integers(0, m, m)
        M = P[:, idx].mean(axis=1).reshape(n, n)
        if sym:
            M = (M - M.T) / 2
        sol = solve_matrix_game(M)
        vals.append(sol.value)
        for i, x in enumerate(sol.x):
            if x > 1e-4:
                counts[names[i]] += 1
                mass[names[i]] += x
    return {"support_freq": {k: v / boots for k, v in counts.items()}, "mean_mass": {k: v / boots for k, v in
                                                                                     mass.items()},
            "value": Cm.ci(vals)}


__all__ = ["SET", "run", "solve", "equilibrium_ci", "start_of", "LEDGER"]
