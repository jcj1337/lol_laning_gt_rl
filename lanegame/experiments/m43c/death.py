"""Part D death-timer audit: what does an absence of d turns cost, at which game phase, under which objective?

States: present-champion turns of self-play games of the M4.3b `base` finals (seeds 0-2), sampled in three phases:
early (turns 20-60: levels about 5-10, no plate fallen yet in most games), mid (turns 80-130: levels about 12-16,
plates falling) and late (turns 200-300: both champions at level 18, towers mostly exhausted). From each state the
focal champion's record is replaced, on the SAME rollout streams, by

  alive        unchanged (the policy continues)
  recall       "just recalled": at its tower, absent 1 more turn (recall_turns - 1), HP kept (full on return)
  death_d      "just died": HP 0, at its tower, absent for d turns (d = 1, 3, 5, 8, 12; the current rule is d = 3).
               d > 3 uses the extended-absence device of `rollouts` (outside the engine; the champion shops when the
               engine first returns it and is then held away for the remaining turns)

and every variant is rolled out for W = 300 turns against the same policy. Reported: V(variant) - V(alive) under the
gold objective (the current one), gold + lambda XP and gold + mu level progress, with and without the Part C
termination; the full cost of a death adds back the kill bounty the opponent received on the death turn (0.5), which
precedes the state. XP lost and plates conceded are reported as mechanisms. This measures the consequences of absence
lengths in the current game; it implements no death-timer rule.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np

from .common import DEATH_SEED, LAMBDAS, cfg_game, mean_ci, meta, save_json

SPECS = ("final:base:0", "final:base:1", "final:base:2")
PHASES = {"early": (20, 60), "mid": (80, 130), "late": (200, 300)}
DEATH_TURNS = (1, 3, 5, 8, 12)
N_STATES, N_ROLL, WINDOW = 8, 16, 300


def variants(snap, seat: int, cfg) -> Dict[str, tuple]:
    T = cfg.tower
    out = {"alive": (snap, None),
           "recall": (snap.with_side(seat, pos=-T, absent=cfg.recall_turns - 1), None)}
    for d in DEATH_TURNS:
        n = min(d, cfg.champion_absent_max)
        out[f"death_{d}"] = (snap.with_side(seat, pos=-T, hp=0, absent=n), ({seat: d - n} if d > n else None))
    return out


def _job(args) -> tuple:
    spec, phase, state_seed, roll_seed = args
    import torch
    from ...diagnostics import ProbCache
    from ..rl.m43b.behavior import sample_states
    from ..rl.m43b.common import agent_from_spec
    from .rollouts import paired
    torch.set_num_threads(1)
    cfg = cfg_game()
    agent = agent_from_spec(spec, cfg)
    lo, hi = PHASES[phase]
    mask = {phase: (lambda R, S: (R["k"] >= lo) & (R["k"] <= hi) & ~R["enemy_absent"])}
    t0 = time.perf_counter()
    found = sample_states(cfg, agent, mask, N_STATES, state_seed)[phase]
    cache = ProbCache()
    out = []
    for i, (snap, seat, feats) in enumerate(found):
        var = variants(snap, seat, cfg)
        cands = {name: None for name in var}
        res = paired(cfg, snap, seat, cands, (agent, agent), N_ROLL, roll_seed, key=(i,), cache=cache, window=WINDOW,
                     variants=var)
        base = res["alive"]
        rec = {}
        for name in var:
            if name == "alive":
                continue
            r = res[name]
            rec[name] = {ch: float((r[ch] - base[ch]).mean()) for ch in ("G", "X", "Lam", "G_T", "X_T", "Lam_T")}
            rec[name]["xp_focal"] = float(r["xp"][:, 0].mean() - base["xp"][:, 0].mean())
            rec[name]["xp_opp"] = float(r["xp"][:, 1].mean() - base["xp"][:, 1].mean())
            rec[name]["plates_opp"] = float(r["plates"][:, 1].mean() - base["plates"][:, 1].mean())
            rec[name]["plates_focal"] = float(r["plates"][:, 0].mean() - base["plates"][:, 0].mean())
            rec[name]["deaths_focal"] = float(r["deaths"][:, 0].mean() - base["deaths"][:, 0].mean())
            rec[name]["absent_focal"] = float(r["absent"][:, 0].mean() - base["absent"][:, 0].mean())
            rec[name]["se_G"] = float((r["G"] - base["G"]).std(ddof=1) / np.sqrt(len(r["G"])))
        ch = snap.champ
        out.append({"seat": seat, "k": snap.k, "levels": [feats["level"], feats["elevel"]], "hp_frac": feats["hp_frac"],
                    "gold": feats["gold"], "invested": feats["invested"],
                    "tau": [ch.side(seat).tower_dmg, ch.side(1 - seat).tower_dmg], "variants": rec})
    return f"{spec}|{phase}", {"states": out, "elapsed_s": time.perf_counter() - t0}


def summarize(cells: Dict[str, dict], bounty: float = 0.5) -> dict:
    out = {}
    for phase in PHASES:
        states = [st for k, v in cells.items() if k.endswith(f"|{phase}") for st in v["states"]]
        if not states:
            continue
        row = {"n_states": len(states), "turn": float(np.mean([s["k"] for s in states])),
               "level": float(np.mean([s["levels"][0] for s in states])),
               "hp_frac": float(np.mean([s["hp_frac"] for s in states]))}
        for name in states[0]["variants"]:
            add = bounty if name.startswith("death") else 0.0
            v = [s["variants"][name] for s in states]
            e = {"cost_gold": mean_ci([-(x["G"]) + add for x in v]),
                 "cost_gold_T": mean_ci([-(x["G_T"]) + add for x in v])}
            for lam in LAMBDAS[1:]:
                e[f"cost_lam={lam:.4g}"] = mean_ci([-(x["G"] + lam * x["X"]) + add for x in v])
            e["cost_mu=1"] = mean_ci([-(x["G"] + x["Lam"]) + add for x in v])
            e["cost_lam=0.125_T"] = mean_ci([-(x["G_T"] + 0.125 * x["X_T"]) + add for x in v])
            for key in ("xp_focal", "xp_opp", "plates_opp", "plates_focal", "deaths_focal", "absent_focal"):
                e[key] = mean_ci([x[key] for x in v])
            row[name] = e
        out[phase] = row
    return out


def run(workers: int = 9) -> dict:
    from ..rl.m43b.common import run_jobs
    jobs = []
    for i, spec in enumerate(SPECS):
        for j, phase in enumerate(PHASES):
            jobs.append((spec, phase, DEATH_SEED + 100 * i + 10 * j, DEATH_SEED + 500 + 100 * i + 10 * j))
    t0 = time.perf_counter()
    res = dict(run_jobs(_job, jobs, workers, "death"))
    out = {"meta": meta(stage="death", window=WINDOW, n_roll=N_ROLL, death_turns=list(DEATH_TURNS),
                        phases={k: list(v) for k, v in PHASES.items()}, seed_root=DEATH_SEED),
           "cells": res, "summary": summarize(res)}
    out["meta"]["elapsed_s"] = time.perf_counter() - t0
    save_json("death", out)
    return out


__all__ = ["run", "variants", "summarize"]
