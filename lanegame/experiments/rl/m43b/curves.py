"""Learning curves against fixed references and the behaviour profile over training.

Every curve point of a run (every 4th snapshot, i.e. every 102,400 agent transitions, plus the final network, plus
the untrained initial network of the seed as the point at 0) is evaluated with the discounted evaluator (protocol
E1, the development window W = 460 with its guaranteed tail bound of 1.05 resources, 16 paired episodes per matchup,
one fixed seed for every point so that the curves are paired across points) against the four curve references, and
profiled in self-play (action shares, mean category probabilities, entropies, counters: `behavior.profile`).

Development numbers: they rank snapshots and show trends; the headline evaluation (`evaluate.headline`) is separate.
"""

from __future__ import annotations

import time
from typing import Sequence

from .common import (CONDITIONS, CURVE_EPISODES, CURVE_SEED, PROFILE_SEED, SEEDS, WINDOW_DEV, agent_from_spec, cfg_game,
                     curve_points, load, meta, run_jobs, save_json)
from .references import CURVE_REFS

PROFILE_GAMES = 6


def _curve_job(args) -> dict:
    cond, seed, transitions, spec, refs, episodes, window, prof_games = args
    import torch
    from ....env.discounted import evaluate_discounted
    from .behavior import profile
    torch.set_num_threads(1)
    cfg = cfg_game()
    x = agent_from_spec(spec, cfg)
    t0 = time.perf_counter()
    vs = {}
    for r in refs:
        y = agent_from_spec(f"ref:{r}", cfg)
        e = evaluate_discounted(cfg, x, y, episodes=episodes, seed=CURVE_SEED, window=window)
        vs[r] = {k: e[k] for k in ("estimate", "stderr", "ci95", "seat_A", "seat_B")}
    prof = profile(cfg, x, prof_games, window, PROFILE_SEED)
    return {"condition": cond, "seed": seed, "transitions": transitions, "spec": spec, "vs": vs, "profile": prof,
            "elapsed_s": time.perf_counter() - t0}


def curves(conds: Sequence[str] = tuple(CONDITIONS), seeds: Sequence[int] = SEEDS, workers: int = 9,
           refs: Sequence[str] = CURVE_REFS, episodes: int = CURVE_EPISODES, window: int = WINDOW_DEV,
           force: bool = False) -> dict:
    """Points already in curves.json (same condition, seed and transitions) are reused unless `force`."""
    t0 = time.perf_counter()
    jobs = []
    for c in conds:
        for s in seeds:
            jobs.append((c, s, 0, f"initial:{s}", tuple(refs), episodes, window, PROFILE_GAMES))
            for tr, p in curve_points(c, s):
                jobs.append((c, s, tr, p, tuple(refs), episodes, window, PROFILE_GAMES))
    prev = (load("curves") or {}) if not force else {}
    have = {(p["condition"], p["seed"], p["transitions"]): p for p in prev.get("points", [])}
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in have]
    print(f"curves: {len(jobs)} points, {len(todo)} to compute", flush=True)
    res = run_jobs(_curve_job, todo, workers, label="curves")
    for p in res:
        have[(p["condition"], p["seed"], p["transitions"])] = p
    points = [have[(j[0], j[1], j[2])] for j in jobs]
    data = {"meta": meta(stage="curves", window=window, episodes=episodes, refs=list(refs), eval_seed=CURVE_SEED,
                         profile_seed=PROFILE_SEED, profile_games=PROFILE_GAMES),
            "label": f"development window W = {window} (guaranteed tail bound 1.05 resources), {episodes} paired "
                     "episodes per matchup, one evaluation seed for every point",
            "elapsed_s": time.perf_counter() - t0, "points": points}
    save_json("curves", data)
    return data


__all__ = ["curves"]
