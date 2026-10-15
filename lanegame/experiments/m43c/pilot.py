"""M4.3c stabilization, Stage B: a bounded PPO pilot (docs/m4/m4_3/m4_3c_strategic_economics_calibration.md §8).

EXPLORATORY: a short shared-policy self-play run per (variant, seed) shows whether learned play reacts to a candidate
economy at all; it is not a convergence or equilibrium result. Training is `lanegame.rl.train_shared` with the M4.3b
settings of its best condition `ent03` (PPOConfig.continuing: gamma 0.99, GAE 0.95, segments 128, games of 512 turns,
16 environments, reward x 0.1; entropy 0.03; 100% current-policy games) on `economics.variant_cfg(variant)`.
Checkpoints go to results/rl/checkpoints/m43c_pilot/ (git-ignored); summaries to results/m43c/pilot.json.

Evaluation of each final network (stochastic policy, its own sampling stream per seat):
    behavior   self-play games from s0: the wave-action mix (Last Hit / Thin / Slow Push / Shove) overall and by
               situation (freezing: we hold the enemy front at our own -1 or deeper; frozen on: the reverse; neutral:
               the fronts in contact at the lane cell; pushing: our front on the enemy half, not in contact), Trade by
               the sign of the duel edge, Safe / Recall rates, deaths, crashes, plate turns, levels
    refs       paired games against scripted strategies (both seats), discounted J over REF_TURNS turns

    python -m lanegame.experiments.m43c pilot [--variants base ot80_tf100_dS1] [--seeds 0 1] [--steps 2048000]
"""

from __future__ import annotations

import os
import time
from typing import Dict, Sequence

import numpy as np

from ...config import B_TRADE, CHAMP_ABSENT, champ_behavior, champ_wave
from . import harness as h
from . import strategies as st
from .common import mean_ci, meta, save_json
from .economics import VARIANTS, variant_cfg

CKPT = os.path.join("results", "rl", "checkpoints", "m43c_pilot")
STEPS = 2_048_000
SEEDS = (0, 1)
PILOT_SEED = 78_000
BEHAVIOR_GAMES, BEHAVIOR_TURNS = 32, 300
REFS = ("farm4", "thin4", "shove4", "hunt4", "shove_hunt4", "freeze_hunt4")
REF_GAMES, REF_TURNS = 32, 300


def run_dir(v: str, seed: int) -> str:
    return os.path.join(CKPT, f"{v}_s{seed}")


def _train_job(args) -> dict:
    v, seed, steps = args
    import torch
    from ...rl import PPOConfig, SharedConfig, train_shared
    torch.set_num_threads(1)
    cfg = variant_cfg(v)
    ppo = PPOConfig.continuing(cfg, total_steps=steps, n_envs=16, seed=seed, torch_threads=1, ent_coef=0.03)
    t0 = time.perf_counter()
    res = train_shared(cfg, ppo, SharedConfig(current_fraction=1.0), out_dir=run_dir(v, seed), name=f"{v}_s{seed}",
                       verbose=False, overwrite=True)
    hist = res.history
    last = hist[-10:]
    return {"variant": v, "seed": seed, "transitions": res.agent_transitions, "elapsed_s": time.perf_counter() - t0,
            "entropy_first_last": [hist[0]["entropy"], float(np.mean([e["entropy"] for e in last]))],
            "explained_variance_last": float(np.mean([e["explained_variance"] for e in last])),
            "finite": bool(all(np.isfinite(e[k]) for e in hist for k in ("policy_loss", "value_loss", "entropy")))}


class NetSeat:
    """A NetAgent as a harness agent with its own sampling stream."""

    def __init__(self, agent, rng):
        self.agent, self.rng, self.name = agent, rng, agent.name

    def act(self, obs):
        return self.agent.act(obs, self.rng)


def _net(v: str, seed: int):
    from ...rl import load_agent
    return load_agent(os.path.join(run_dir(v, seed), "final.pt"), name=f"{v}_s{seed}")


def _situation(lane, s: int) -> str:
    own = lane.side(s)
    en = lane.side(1 - s)
    fx, fh, ex, eh = own[0], own[1], en[0], en[1]
    if fh > 0 and eh > 0 and -ex <= -1:
        return "freezing"
    if fh > 0 and eh > 0 and fx >= 1:
        return "frozen_on"
    if fh > 0 and eh > 0 and fx + ex >= -1:
        return "neutral"
    if fh > 0 and fx >= 1:
        return "pushing"
    return "other"


def _behavior_job(args) -> tuple:
    v, seed, n, turns = args
    import torch
    from ..m4 import zoning_natural as zn
    torch.set_num_threads(1)
    cfg = variant_cfg(v)
    agent = _net(v, seed)
    from ...champions_m4 import initial_m4
    from ...mechanics import initial_state
    stt = (initial_state(cfg), initial_m4(cfg), 0)
    counts: Dict[str, np.ndarray] = {}
    trade = {"edge+": [0, 0], "edge-": [0, 0]}
    recs = []
    for e, ss in enumerate(h.seeds(PILOT_SEED, n)):
        rng = np.random.default_rng(np.random.SeedSequence(PILOT_SEED, spawn_key=(e, 1)))
        seats = (NetSeat(agent, rng), NetSeat(agent, rng))

        def on_turn(i, r, env, _c=counts, _t=trade):
            from ...champions_m4 import M4State
            from ...state import State
            info = r.info
            lane = State.from_list(info["state_before"])
            m4 = M4State.from_list(info["champ_before"])
            for s in (0, 1):
                a = info["actions"][s]
                if a == CHAMP_ABSENT:
                    continue
                sit = _situation(lane, s)
                vec = _c.setdefault(sit, np.zeros(4))
                vec[champ_wave(a)] += 1
                tot = _c.setdefault("all", np.zeros(4))
                tot[champ_wave(a)] += 1
                beh = champ_behavior(a)
                b = _c.setdefault("behavior", np.zeros(5))
                b[min(beh, 4)] += 1
                if m4.side(1 - s).absent == 0:
                    key = "edge+" if zn.duel_edge(cfg, lane, m4, s) > 0 else "edge-"
                    _t[key][0] += int(beh == B_TRADE)
                    _t[key][1] += 1

        g = h.play(cfg, stt, seats, ss, turns, on_turn=on_turn)
        w = h.window(g, cfg)
        plates = []
        for i, r in enumerate(g["rows"]):
            for s in (0, 1):
                plates += [i + 1] * r["plates_n"][s]
        recs.append({"died": w["died"], "recalled": w["recalled"], "crashes": w["crashes"], "plates": sorted(plates),
                     "level_end": w["level_end"], "trade_conn": w["trade_conn"], "earned": w["earned"]})
    return f"{v}|{seed}", {"counts": {k: v_.tolist() for k, v_ in counts.items()}, "trade": trade, "games": recs}


def _ref_job(args) -> tuple:
    v, seed, ref, n, turns = args
    import torch
    torch.set_num_threads(1)
    cfg = variant_cfg(v)
    agent = _net(v, seed)
    from ...champions_m4 import initial_m4
    from ...mechanics import initial_state
    stt = (initial_state(cfg), initial_m4(cfg), 0)
    vals = []
    for e, ss in enumerate(h.seeds(PILOT_SEED + 1, n)):
        rng = np.random.default_rng(np.random.SeedSequence(PILOT_SEED + 1, spawn_key=(e, 1)))
        net = NetSeat(agent, rng)
        ga = h.play(cfg, stt, (net, st.StrategyAgent(st.LIBRARY[ref])), ss, turns)
        gb = h.play(cfg, stt, (st.StrategyAgent(st.LIBRARY[ref]), net), ss, turns)
        vals.append(0.5 * (h.window(ga, cfg)["G"] - h.window(gb, cfg)["G"]))
    return f"{v}|{seed}|{ref}", vals


WAVE_ORDER = ((0, "Last Hit"), (2, "Thin"), (3, "Slow Push"), (1, "Shove"))      # slot index -> name
BEHAVIOR_ORDER = ((0, "Hold"), (1, "Yield"), (2, "Trade"), (3, "Safe"), (4, "Recall"))


def summarize_behavior(rec: dict) -> dict:
    out = {}
    for sit, vec in rec["counts"].items():
        vec = np.asarray(vec, dtype=float)
        if vec.sum() > 0:
            order = BEHAVIOR_ORDER if sit == "behavior" else WAVE_ORDER
            out[sit] = {"n": int(vec.sum()), "share": {name: round(float(vec[i] / vec.sum()), 3) for i, name in order}}
    tr = rec["trade"]
    out["trade_rate"] = {k: (round(v[0] / v[1], 3) if v[1] else None) for k, v in tr.items()}
    g = rec["games"]
    out["deaths"] = float(np.mean([sum(x["died"]) / 2 for x in g]))
    out["recalls"] = float(np.mean([sum(x["recalled"]) / 2 for x in g]))
    out["crashes"] = float(np.mean([sum(x["crashes"]) / 2 for x in g]))
    out["connected_trades"] = float(np.mean([sum(x["trade_conn"]) / 2 for x in g]))
    pl = [x["plates"] for x in g]
    for j in range(3):
        vals = [p[j] for p in pl if len(p) > j]
        out[f"plate{j + 1}"] = (float(np.median(vals)) if vals else None, len(vals) / len(g))
    return out


def run(variants: Sequence[str] = ("base", "ot80_tf100_dS1"), seeds: Sequence[int] = SEEDS, steps: int = STEPS,
        workers: int = 6, train: bool = True) -> dict:
    from ..rl.m43b.common import run_jobs
    t0 = time.perf_counter()
    out = {"meta": meta(stage="pilot", variants=list(variants), seeds=list(seeds), steps=steps, ent_coef=0.03,
                        label="EXPLORATORY bounded PPO pilot; not a convergence result",
                        descriptions={v: VARIANTS[v][0] for v in variants})}
    if train:
        runs = run_jobs(_train_job, [(v, s, steps) for v in variants for s in seeds], workers, "pilot train")
        out["train"] = {f"{r['variant']}|{r['seed']}": r for r in runs}
    beh = dict(run_jobs(_behavior_job, [(v, s, BEHAVIOR_GAMES, BEHAVIOR_TURNS) for v in variants for s in seeds],
                        workers, "pilot behavior"))
    out["behavior"] = {k: summarize_behavior(r) for k, r in beh.items()}
    refs = dict(run_jobs(_ref_job, [(v, s, ref, REF_GAMES, REF_TURNS) for v in variants for s in seeds for ref in REFS],
                         workers, "pilot refs"))
    out["refs"] = {k: mean_ci(x) for k, x in refs.items()}
    out["meta"]["elapsed_s"] = time.perf_counter() - t0
    save_json("pilot", out)
    return out


__all__ = ["run", "summarize_behavior"]
