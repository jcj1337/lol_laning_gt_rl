"""Generic plumbing of the M4.4c experiments: one job = (policy, situation, opponent protocol, chunk of states).

An investigation module defines `experiment(name) -> Experiment` (its states, candidates, opponent plans, policy
frequencies and per-rollout extras); jobs carry only (module, experiment name, policy, situation, protocol, chunk), so
they pickle across worker processes, and each job re-collects its states deterministically from the seed root (state
collection costs seconds; nothing is cached on disk). Every job verifies that the network's weights digest is unchanged
by the evaluation.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from ....diagnostics.branch import ProbCache, observation
from . import engine as E
from . import opponents as O
from .common import (ENV_ORDER, INIT_SEED, OFFSET, PRIMARY, ROLL_SEED, SEEDS, STATE_SEED, clear_partial, load, meta,
                     policies, run_resumable, save_json, weights_digest)
from .states import to_json


@dataclass
class Experiment:
    name: str
    offset: str                                        # key of common.OFFSET (seed block)
    sizes: Dict[str, int]                              # situation -> number of states
    protocols: Dict[str, Sequence[str]]                # situation -> opponent protocols / styles
    states: Callable                                   # (cfg, agent, env, seed, phase) -> {situation: [state]}
    candidates: Callable                               # (situation, cfg, snap, seat) -> {name: Plan}
    frequency: Callable                                # (agent, cfg, snap, seat) -> dict
    n_roll: Dict[str, int]                             # situation -> rollouts per candidate
    window: int = E.WINDOW
    ep_h: Sequence[int] = E.EP_H
    track_freeze: bool = False
    extra: Optional[Callable] = None                   # cfg -> (infos, focal) -> dict
    opp_plans: Optional[Callable] = None               # situation -> {protocol: plan factory}
    initialized: Sequence[str] = ()                    # situations built from constructed states (other seed root)
    chunk: int = 8
    notes: dict = field(default_factory=dict)


def get(module: str, name: str) -> Experiment:
    return importlib.import_module(f"lanegame.experiments.rl.m44c.{module}").experiment(name)


def job(args) -> dict:
    module, name, env, seed, phase, situation, protocol, lo, hi, n_roll = args
    X = get(module, name)
    cfg, x = load(env, seed, phase)
    d0 = weights_digest(x)
    found = X.states(cfg, x, env, seed, phase)[situation][lo:hi]
    cache = ProbCache()
    ex = X.extra(cfg) if X.extra is not None else None
    opp_plans = X.opp_plans(situation) if X.opp_plans is not None else None
    root = (INIT_SEED if situation in X.initialized else ROLL_SEED) + OFFSET[X.offset]
    sit_i = sorted(X.sizes).index(situation)
    pro_i = sorted(set(p for ps in X.protocols.values() for p in ps)).index(protocol)
    out = []
    for i, st in enumerate(found):
        snap, s = st["snapshot"], st["seat"]
        ob = observation(cfg, snap, s)
        cands = {k: v for k, v in X.candidates(situation, cfg, snap, s).items()
                 if not hasattr(v, "applicable") or v.applicable(ob)}
        agents, plan = O.setup(protocol, x, cfg, snap, s, opp_plans)
        res = E.evaluate(cfg, snap, s, cands, agents, n_roll or X.n_roll[situation], root,
                         key=(seed, 0 if phase == PRIMARY else 1, sit_i, pro_i, lo + i), window=X.window,
                         opponent_plan=plan, ep_h=X.ep_h, extra=ex, track_freeze=X.track_freeze, cache=cache)
        out.append({**to_json(st), "index": lo + i, "frequency": X.frequency(x, cfg, snap, s), "result": res})
    return {"env": env, "seed": seed, "phase": phase, "situation": situation, "protocol": protocol, "lo": lo,
            "states": out, "digest_before": d0, "digest_after": weights_digest(x)}


CONTROLS = ("four",)          # the Shove-dominance control runs only the first protocol of each situation (its own
                              # policy as the opponent): it answers "what does the original economy do", not the
                              # opponent-style questions, which concern the calibrated economies


def run(module: str, name: str, envs=None, seeds=None, phases: Sequence[str] = (PRIMARY,), workers: int = 11,
        protocols: Optional[Sequence[str]] = None, situations: Optional[Sequence[str]] = None,
        out_name: Optional[str] = None, controls: Sequence[str] = CONTROLS, limit: Optional[int] = None,
        n_roll: Optional[int] = None) -> dict:
    """Every (policy, situation, protocol, chunk) job of an experiment; results/m44c/<out_name or name>.json(.gz).
    Environments in `controls` run only the first protocol of each situation. `limit` (states per situation) and
    `n_roll` override the experiment's sizes (tests of the pipeline only)."""
    t0 = time.perf_counter()
    X = get(module, name)
    jobs = []
    for env, seed, phase in policies(envs or ENV_ORDER, seeds or SEEDS, phases):
        for situation, n in X.sizes.items():
            if situations and situation not in situations:
                continue
            n = min(n, limit) if limit else n
            for j, protocol in enumerate(X.protocols[situation]):
                if (protocols and protocol not in protocols) or (env in controls and j > 0):
                    continue
                for lo in range(0, n, X.chunk):
                    jobs.append((module, name, env, seed, phase, situation, protocol, lo, min(n, lo + X.chunk), n_roll))
    res = run_resumable(job, jobs, workers, out_name or name)
    cells: Dict[str, dict] = {}
    for r in res:
        k = f"{r['phase']}|{r['env']}|{r['seed']}|{r['situation']}|{r['protocol']}"
        c = cells.setdefault(k, {"env": r["env"], "seed": r["seed"], "phase": r["phase"], "situation": r["situation"],
                                 "protocol": r["protocol"], "states": [], "digests_unchanged": True})
        c["states"] += r["states"]
        c["digests_unchanged"] &= r["digest_before"] == r["digest_after"]
    for c in cells.values():
        c["states"].sort(key=lambda st: st["index"])
    data = {"meta": meta(stage=out_name or name, module=module, experiment=name, sizes=X.sizes, limit=limit,
                         protocols={k: list(v) for k, v in X.protocols.items()}, n_roll=n_roll or X.n_roll, window=X.window,
                         ep_h=list(X.ep_h), initialized=list(X.initialized), notes=X.notes,
                         state_seed=STATE_SEED + OFFSET[X.offset], roll_seed=ROLL_SEED + OFFSET[X.offset],
                         init_seed=INIT_SEED + OFFSET[X.offset], jobs=len(jobs), controls=list(controls)),
            "cells": cells, "elapsed_s": time.perf_counter() - t0}
    save_json(out_name or name, data)
    clear_partial(out_name or name)
    return data


__all__ = ["Experiment", "get", "job", "run"]
