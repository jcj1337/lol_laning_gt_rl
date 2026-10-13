"""Plan-level counterfactuals: strategies that need several coordinated turns (M4.3b §7 of the brief).

A single-move counterfactual (`behavior.counterfactual`) commits one first move and hands control back to the
policy, so it cannot credit a strategy whose value appears only after several turns of consistent play: a slow push
(Thin a little, then Hold while the edge compounds), a held freeze (the threshold controller of M4.2e-C), a sustained
Clear, pure last-hitting, or Clearing while the enemy is away. This module commits the focal champion to such a plan
for a fixed number of turns (the opponent is the policy throughout, the focal champion returns to the policy when
the plan ends) and compares the discounted return over CF_WINDOW turns with the policy's own play on common random
numbers (`lanegame.diagnostics.branch` with `FunctionPlan`s), from sampled reachable self-play states of the
situations where each plan applies.

Plans (wave + position; every code is legal for a present champion under natural waves):
  slow_push    Thin + Hold for 2 turns, then Hold + Hold for 12 turns (the M4.2e-C slow push from a neutral lane)
  clear        Clear + Hold for 14 turns (sustained clearing)
  hold         Hold + Hold for 14 turns (pure last-hitting)
  freeze       the threshold freeze controller for 20 turns: Thin + Hold while the enemy front stands at own -1
               with >= 14 HP, else Hold + Hold (the M4.2e-C controller, Hold position)
  shove_away   Clear + Hold while the enemy champion is absent (at most 12 turns), then the policy
A plan ends early (hands control to the policy) when the focal champion is absent.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from ....config import B_HOLD, CHAMP_ABSENT, W_CLEAR, W_HOLD, W_THIN, champ_act
from .behavior import CF_WINDOW, sample_states, situation_masks
from .common import (CF_ROLL_SEED, CF_STATE_SEED, CONDITIONS, PRIMARY, SEEDS, agent_from_spec, cfg_game, load, meta,
                     run_jobs, save_json)

HO, TN, CL = champ_act(W_HOLD, B_HOLD), champ_act(W_THIN, B_HOLD), champ_act(W_CLEAR, B_HOLD)
PLAN_SITUATIONS = {"neutral": ("slow_push", "clear", "hold"), "own_pushing": ("slow_push", "clear", "hold"),
                   "freeze_like": ("freeze", "clear", "hold"), "enemy_absent": ("shove_away", "hold")}
P_STATES, P_ROLLOUTS = 8, 16
FREEZE_THRESHOLD, FREEZE_TURNS = 14, 20


def _present(obs) -> bool:
    return obs.legal != (CHAMP_ABSENT,)


def make_plan(name: str):
    from ....diagnostics import FunctionPlan

    def slow_push(obs, t):
        if not _present(obs) or t >= 14:
            return None
        return TN if t < 2 else HO

    def clear(obs, t):
        return CL if (_present(obs) and t < 14) else None

    def hold(obs, t):
        return HO if (_present(obs) and t < 14) else None

    def freeze(obs, t):
        if not _present(obs) or t >= FREEZE_TURNS:
            return None
        s = obs.side
        en_fx, en_fh = obs.state.side(1 - s)[:2]
        return TN if (en_fh >= FREEZE_THRESHOLD and -en_fx <= -1) else HO

    def shove_away(obs, t):
        if not _present(obs) or t >= 12:
            return None
        return CL if obs.champ.side(1 - obs.side).absent > 0 else None

    return FunctionPlan(name, {"slow_push": slow_push, "clear": clear, "hold": hold, "freeze": freeze,
                               "shove_away": shove_away}[name])


def _plan_job(args) -> tuple:
    spec, key, plans, n_states, n_roll, state_seed, roll_seed = args
    import torch
    from ....diagnostics import FollowPolicy, ProbCache, branch
    torch.set_num_threads(1)
    cfg = cfg_game()
    agent = agent_from_spec(spec, cfg)
    t0 = time.perf_counter()
    states = sample_states(cfg, agent, situation_masks([key]), n_states, state_seed)[key]
    cache = ProbCache()
    out = []
    for i, (snap, seat, feats) in enumerate(states):
        cands = {p: make_plan(p) for p in plans}
        cands["policy"] = FollowPolicy()
        res = branch(cfg, snap, seat, cands, (agent, agent), n_roll, roll_seed, key=(i,), cache=cache,
                     max_turns=CF_WINDOW)
        out.append({"seat": seat, "k": snap.k, "features": feats, "q": {c: res.mean(c) for c in res.candidates},
                    "adv": {p: res.diff(p, "policy") for p in plans},
                    "decomposition": {c: {"farm_diff": float((res.farm[c][:, 0] - res.farm[c][:, 1]).mean()),
                                          "deaths": res.deaths[c].mean(0).tolist(),
                                          "crashes": res.m4[c]["crashes"].mean(0).tolist() if c in res.m4 else None}
                                      for c in res.candidates}})
    return (spec, key), {"spec": spec, "situation": key, "plans": list(plans), "states": out,
                         "elapsed_s": time.perf_counter() - t0}


def aggregate(states: List[dict], plans: Sequence[str]) -> dict:
    out = {"n_states": len(states)}
    for p in plans:
        d = np.array([s["adv"][p][0] for s in states])
        se = np.array([s["adv"][p][1] for s in states])
        out[p] = {"mean_adv": float(d.mean()), "ci": float(1.96 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
                  "share_pos": float(np.mean(d > 1.96 * se)), "share_neg": float(np.mean(d < -1.96 * se)),
                  "farm_diff": float(np.mean([s["decomposition"][p]["farm_diff"] - s["decomposition"]["policy"]["farm_diff"] for s in states])),
                  "deaths_focal": float(np.mean([s["decomposition"][p]["deaths"][0] for s in states])),
                  "deaths_policy": float(np.mean([s["decomposition"]["policy"]["deaths"][0] for s in states]))}
    return out


def plan_counterfactual(conds: Sequence[str] = tuple(CONDITIONS), seeds: Sequence[int] = SEEDS, workers: int = 4,
                        specs: Optional[Sequence[str]] = None, n_states: int = P_STATES, n_roll: int = P_ROLLOUTS,
                        force: bool = False) -> dict:
    t0 = time.perf_counter()
    if specs is None:
        specs = [f"final:{c}:{seeds[0]}" for c in conds]
    prev = (load("plans") or {}) if not force else {}
    results = dict(prev.get("results", {}))
    jobs = [(spec, key, plans, n_states, n_roll, CF_STATE_SEED + 300 + 10 * i + j, CF_ROLL_SEED + 300 + 10 * i + j)
            for i, spec in enumerate(specs) for j, (key, plans) in enumerate(PLAN_SITUATIONS.items())]
    todo = [j for j in jobs if f"{j[0]}|{j[1]}" not in results]
    print(f"plans: {len(jobs)} jobs, {len(todo)} to compute", flush=True)
    res = run_jobs(_plan_job, todo, workers, label="plans")
    results.update({f"{k[0]}|{k[1]}": v for k, v in res})
    for v in results.values():
        v["summary"] = aggregate(v["states"], v["plans"])
    data = {"meta": meta(stage="plans", window=CF_WINDOW, states=n_states, rollouts=n_roll, situations=PLAN_SITUATIONS,
                         freeze_threshold=FREEZE_THRESHOLD, freeze_turns=FREEZE_TURNS),
            "elapsed_s": time.perf_counter() - t0, "results": results}
    save_json("plans", data)
    return data


__all__ = ["plan_counterfactual", "make_plan", "PLAN_SITUATIONS"]
