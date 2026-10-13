"""The headline evaluation of the M4.3b finals (protocol E1, W = 1000, 32 paired episodes per matchup, 95% paired
confidence intervals; the guaranteed truncation bound is 0.005 resources).

Matchups: every final against every reference of the suite; every pair of finals (cross-play across seeds and
conditions); every final against the untrained initial network of its seed. The attackers' matchups are in
`exploit`. Two kinds of uncertainty are kept apart: the paired Monte Carlo interval of one matchup (evaluation
episodes) and the spread across the three independently trained seeds of a condition (reported as mean +- 1.96 SE
over seeds, indicative at n = 3).
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, List, Sequence

import numpy as np

from .common import (CONDITIONS, HEAD_EPISODES, HEAD_SEED, SEEDS, WINDOW_HEAD, agent_from_spec, cfg_game, load, mean_ci,
                     meta, run_jobs, save_json, snapshot_paths)
from .references import HEAD_REFS

PROGRESS_POINTS = (102_400, 409_600, 1_024_000)      # final against its own earlier snapshots (self-play progress)


def nearest_snapshot(cond: str, seed: int, transitions: int):
    """(transitions, spec) of the run's snapshot nearest to `transitions` (historical runs' counts are a few off)."""
    snaps = snapshot_paths(cond, seed)
    if not snaps:
        return None
    tr = min((t for t, _ in snaps), key=lambda t: abs(t - transitions))
    return tr, f"snap:{cond}:{seed}:{tr}"


def h2h_job(args) -> tuple:
    xs, ys, episodes, window, seed = args
    import torch
    from ....env.discounted import evaluate_discounted
    torch.set_num_threads(1)
    cfg = cfg_game()
    x, y = agent_from_spec(xs, cfg), agent_from_spec(ys, cfg)
    e = evaluate_discounted(cfg, x, y, episodes=episodes, seed=seed, window=window)
    return (xs, ys), {k: e[k] for k in ("estimate", "stderr", "ci95", "seat_A", "seat_B", "pairs", "window",
                                        "elapsed_s")} | {"tail_bound": e["tail_bound"]["guaranteed"]}


def summarize(pairs: Dict[str, dict], conds: Sequence[str], seeds: Sequence[int], refs: Sequence[str]) -> dict:
    """Per condition and reference: the per-seed estimates (each with its paired CI) and the across-seed mean."""
    out = {"vs_refs": {}, "vs_initial": {}, "cross": {}}
    for c in conds:
        out["vs_refs"][c] = {}
        for r in refs:
            per = [pairs[f"final:{c}:{s}|ref:{r}"] for s in seeds if f"final:{c}:{s}|ref:{r}" in pairs]
            if per:
                m, h = mean_ci([p["estimate"] for p in per])
                out["vs_refs"][c][r] = {"per_seed": [[p["estimate"], p["ci95"][0], p["ci95"][1]] for p in per],
                                        "mean": m, "ci_seeds": h,
                                        "mean_episode_halfwidth": float(np.mean([1.96 * p["stderr"] for p in per]))}
        per = [pairs[f"final:{c}:{s}|initial:{s}"] for s in seeds if f"final:{c}:{s}|initial:{s}" in pairs]
        if per:
            m, h = mean_ci([p["estimate"] for p in per])
            out["vs_initial"][c] = {"per_seed": [p["estimate"] for p in per], "mean": m, "ci_seeds": h}
    finals = [f"final:{c}:{s}" for c in conds for s in seeds]
    M = {}
    for a in finals:
        for b in finals:
            if a == b:
                M[f"{a}|{b}"] = 0.0
            elif f"{a}|{b}" in pairs:
                M[f"{a}|{b}"] = pairs[f"{a}|{b}"]["estimate"]
            elif f"{b}|{a}" in pairs:
                M[f"{a}|{b}"] = -pairs[f"{b}|{a}"]["estimate"]
    out["cross"] = {"finals": finals, "matrix": [[M.get(f"{a}|{b}", float("nan")) for b in finals] for a in finals]}
    # condition against condition: mean over the 9 seed pairs
    cc = {}
    for c1 in conds:
        for c2 in conds:
            vals = [M[f"final:{c1}:{s1}|final:{c2}:{s2}"] for s1 in seeds for s2 in seeds
                    if f"final:{c1}:{s1}|final:{c2}:{s2}" in M and (c1 != c2 or s1 != s2)]
            if vals:
                cc[f"{c1}|{c2}"] = {"mean": float(np.mean(vals)), "n": len(vals),
                                    "ci": float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")}
    out["cross_conditions"] = cc
    # self-play progress: the final against its own earlier snapshots
    prog = {}
    for c in conds:
        for pt in PROGRESS_POINTS:
            vals = []
            for s in seeds:
                ns = nearest_snapshot(c, s, pt)
                if ns and f"final:{c}:{s}|{ns[1]}" in pairs:
                    vals.append(pairs[f"final:{c}:{s}|{ns[1]}"]["estimate"])
            if vals:
                m, h = mean_ci(vals)
                prog.setdefault(c, {})[str(pt)] = {"per_seed": vals, "mean": m, "ci_seeds": h}
    out["progress"] = prog
    # within-condition seed spread against the mean reference value: the average over refs of the seed SD
    for c in conds:
        sds = [np.std([p[0] for p in out["vs_refs"][c][r]["per_seed"]], ddof=1) for r in refs if r in out["vs_refs"][c]]
        eps = [out["vs_refs"][c][r]["mean_episode_halfwidth"] for r in refs if r in out["vs_refs"][c]]
        out.setdefault("variation", {})[c] = {"seed_sd_mean": float(np.mean(sds)) if sds else float("nan"),
                                               "episode_halfwidth_mean": float(np.mean(eps)) if eps else float("nan")}
    return out


def headline(conds: Sequence[str] = tuple(CONDITIONS), seeds: Sequence[int] = SEEDS, workers: int = 9,
             episodes: int = HEAD_EPISODES, window: int = WINDOW_HEAD, refs: Sequence[str] = HEAD_REFS,
             force: bool = False) -> dict:
    """Matchups already in headline.json are reused unless `force` (same protocol, seed and episodes)."""
    t0 = time.perf_counter()
    finals = [f"final:{c}:{s}" for c in conds for s in seeds]
    jobs: List[tuple] = []
    for f in finals:
        for r in refs:
            jobs.append((f, f"ref:{r}", episodes, window, HEAD_SEED))
        jobs.append((f, f"initial:{f.split(':')[2]}", episodes, window, HEAD_SEED))
    for a, b in itertools.combinations(finals, 2):
        jobs.append((a, b, episodes, window, HEAD_SEED))
    for c in conds:
        for s in seeds:
            for pt in PROGRESS_POINTS:
                ns = nearest_snapshot(c, s, pt)
                if ns:
                    jobs.append((f"final:{c}:{s}", ns[1], episodes, window, HEAD_SEED))
    prev = (load("headline") or {}) if not force else {}
    pairs = dict(prev.get("pairs", {}))
    todo = [j for j in jobs if f"{j[0]}|{j[1]}" not in pairs]
    print(f"headline: {len(jobs)} matchups of {2 * episodes} games x {window} turns, {len(todo)} to compute", flush=True)
    res = run_jobs(h2h_job, todo, workers, label="headline")
    pairs.update({f"{k[0]}|{k[1]}": v for k, v in res})
    data = {"meta": meta(stage="headline", window=window, episodes=episodes, seed=HEAD_SEED, refs=list(refs)),
            "label": f"protocol E1, headline window W = {window} (guaranteed tail bound {pairs[next(iter(pairs))]['tail_bound']:.4f} "
                     f"resources), {episodes} paired episodes per matchup, payoff to x in discounted resources",
            "elapsed_s": time.perf_counter() - t0, "pairs": pairs, "summary": summarize(pairs, conds, seeds, refs)}
    save_json("headline", data)
    return data


__all__ = ["headline", "h2h_job", "summarize", "nearest_snapshot", "PROGRESS_POINTS"]
