"""Part 5 of the M4.3d study: restricted zero-sum matrix games (the M4.3c instrument) on the M4.3d configurations.

Two families of games, both solved with `lanegame.matrix_game.solve_matrix_game` on the matrix of mean values to A:

  * the M4.3c scenarios (freeze F*, neutral N*, freeze-phase P*: `m43c.economics.MATRIX_SCENARIOS`) with the M4.3c
    strategy sets extended by the M4.3d strategies that use Vision, dives and Recall timing (EXTRA_A / EXTRA_B), played
    here (`run`): 64 paired games of 100 turns per cell, SeedSequence(STRATEGIC_SEED, spawn_key=(e,));
  * the crash scenarios (siege / neutral starts at levels 4 and 10, three power gaps) read from a crash grid
    (`grid_games`): A = the attacker set, B = the defender set.

Objectives: J = discounted gold difference over 100 turns (gamma 0.99, the CURRENT reward) and G40 = undiscounted gold
difference over the first 40 turns. An equilibrium of such a game is an equilibrium of the restricted game only.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import numpy as np

from ..m43c import economics as ec
from ..m43c import harness as h
from ..m43c import strategies as st
from . import common as c
from .strategies import lookup

EXTRA_A = {"F": ("sh_vis",), "N": ("sh_vis", "sh_dive", "sh_recall"), "P": ()}
EXTRA_B = {"F": ("sh_vis",), "N": ("sh_vis", "sh_dive", "sh_recall"), "P": ()}
EPISODES, TURNS = 64, 100


def scenario_sets(sc: str):
    base, gap, level, hb, As, Bs = ec.MATRIX_SCENARIOS[sc]
    fam = sc[0]
    return base, gap, level, hb, tuple(As) + EXTRA_A[fam], tuple(Bs) + EXTRA_B[fam]


def _job(args) -> tuple:
    vname, sc, a, b, n, turns, root = args
    cfg = c.variant_cfg(vname)
    base, gap, level, hb, _, _ = scenario_sets(sc)
    stt = h.start(base, cfg, gap=gap, level=level, hb=hb)
    sa, sb = lookup(a), lookup(b)
    games = h.cell(cfg, stt, lambda: (st.StrategyAgent(sa), st.StrategyAgent(sb)), root, n, turns, windows=(40,))
    keys = ("earned", "farm", "plate", "bounty", "farm_avail", "died", "recalled", "away", "hp_lost", "trade_conn",
            "ganked", "crashes", "plates_n", "held_by")
    out = {"G": [round(g["all"]["G"], 5) for g in games],
           "G40": [round(g["40"]["earned"][0] - g["40"]["earned"][1], 5) for g in games]}
    for w in ("all", "40"):
        out[w] = {f"{k}_{t}": round(float(np.mean([g[w][k][s] for g in games])), 5)
                  for k in keys for s, t in ((0, "a"), (1, "b"))}
    return f"{sc}|{a}|{b}", out


def solve(M: np.ndarray, As: Sequence[str], Bs: Sequence[str]) -> dict:
    from ...matrix_game import solve_matrix_game
    sol = solve_matrix_game(M)
    x, y = np.asarray(sol.x), np.asarray(sol.y)
    return {"value": round(float(sol.value), 4), "gap": float(sol.gap), "A": list(As), "B": list(Bs),
            "matrix": np.round(M, 3).tolist(),
            "x": {a: round(float(p), 4) for a, p in zip(As, x) if p > 1e-6},
            "y": {b: round(float(p), 4) for b, p in zip(Bs, y) if p > 1e-6},
            "B_best_reply": {a: Bs[int(np.argmin(M[i]))] for i, a in enumerate(As)},
            "A_best_reply": {b: As[int(np.argmax(M[:, j]))] for j, b in enumerate(Bs)}}


def run(variants: Sequence[str], scenarios: Optional[Sequence[str]] = None, workers: int = 11,
        root: int = c.STRATEGIC_SEED, episodes: int = EPISODES, tag: str = "") -> Dict[str, dict]:
    out = {}
    for v in variants:
        t0 = time.perf_counter()
        old = c.load(f"strategic_{v}{tag}")
        cells = dict(old["cells"]) if old else {}
        scs = list(scenarios or ec.MATRIX_SCENARIOS)
        jobs = []
        for sc in scs:
            As, Bs = scenario_sets(sc)[4:]
            jobs += [(v, sc, a, b, episodes, TURNS, root) for a in As for b in Bs if f"{sc}|{a}|{b}" not in cells]
        print(f"strategic {v}{tag}: {len(jobs)} cells", flush=True)
        cells.update(dict(c.run_jobs(_job, jobs, workers, f"strategic {v}{tag}")))
        games = dict(old["games"]) if old else {}
        for sc in scs:
            As, Bs = scenario_sets(sc)[4:]
            games[sc] = {obj: solve(np.array([[float(np.mean(cells[f"{sc}|{a}|{b}"][obj])) for b in Bs] for a in As]),
                                    As, Bs) for obj in ("G", "G40")}
        res = {"meta": c.meta(stage="strategic", variant=v, description=c.VARIANTS[v][0], preset=c.VARIANTS[v][1],
                              overrides=c.overrides_json(v), episodes=episodes, turns=TURNS, root=root,
                              elapsed_s=time.perf_counter() - t0),
               "cells": cells, "games": games}
        c.save_json(f"strategic_{v}{tag}", res)
        out[v] = res
    return out


def grid_games(grid: dict, variant: str, start: str, gap: str, a_set: Sequence[str], b_set: Sequence[str],
               obj: str = "G", mask: Optional[str] = None) -> dict:
    """The restricted game of one crash-grid situation (A = a_set, B = b_set)."""
    cells = grid.cells if hasattr(grid, "cells") else grid["cells"]

    def val(a, b):
        cl = cells[f"{variant}|{start}|{gap}|{a}|{b}|{mask or '-'}"]
        return float(np.mean(cl["per_game"][obj]))
    M = np.array([[val(a, b) for b in b_set] for a in a_set])
    return solve(M, a_set, b_set)


__all__ = ["run", "solve", "grid_games", "scenario_sets", "EXTRA_A", "EXTRA_B"]
