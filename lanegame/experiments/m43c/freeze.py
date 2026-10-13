"""Part A §5 / Part D freeze audit: does an early freeze create the intended dilemma for the frozen-on champion?

Controlled scripted scenarios on the unchanged `m4_lasthit()`, from the deterministic start states of the M4.2e-B
harness (`zoning_natural.starts`; both champions at the same XP, full HP, no gold or investment):

  Scenario A (neutral)  the `neutral` start (k9: both fronts and both champions at the lane cell): both champions
                        last-hit (Hold / Hold), both Clear, or one Clears against a last-hitter.
  Scenario B (freeze)   the `freeze` start (k51: B's wave held at A's -1, A's front in its tower cell, A at its tower,
                        B at its +1 next to A's tower). A maintains the freeze with the M4.2e-C threshold controller
                        (Thin while B's front stands at A's -1 or deeper with >= 14 HP, else Hold) and either holds its
                        position ("freezer") or also punishes an opponent that starts in reach ("freezer_punish").
                        B answers with one fixed response for the whole window:
                          approach_farm   Hold position, Hold wave: walk up and last-hit next to A's tower
                          approach_trade  Trade every turn A is present, Hold wave
                          stand_off       Yield (the stand-off cell: the rearmost cell still in reach of the minions)
                          step_back       Safe (never ahead of its safe line, own -1)
                          break_clear     Clear + Hold position: kill A's minions so that B's wave advances
                          break_punish    Clear + punish + Pressure (the M4.2e-C clear_punish line)
                          learned         the M4.3b baseline final (seed 0): what the trained policy actually does
                        and, for comparison, A not freezing at all (A plays Clear, "no_freeze").

Every scripted policy shares the M4.2e-B pre-rules (Recall at 40% HP when legal, else walk back). Paired chance:
the same 64 seeds for every response (SeedSequence(FREEZE_SEED, spawn_key=(gap, episode))). Windows of 20 and 40
turns are reported (10 and 20 waves), from B's perspective unless stated. Gaps: equal, A ahead (x1.76: 50 vs 0
invested) and A behind (the mirror).
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np

from ...config import B_TRADE, W_CLEAR
from .common import FREEZE_SEED, cfg_game, mean_ci, meta, save_json

WINDOWS = (20, 40)
EPISODES = 64
GAPS = ("equal", "a_ahead", "a_behind")


def _policies():
    from ..m4.zoning_natural import Policy
    return {
        "freezer": Policy("freezer", "hold", "freeze"),
        "freezer_punish": Policy("freezer_punish", "punish", "freeze"),
        "no_freeze": Policy("clear_hold", "hold", "clear"),
        "approach_farm": Policy("farm"),
        "approach_trade": Policy("bully", "trade"),
        "stand_off": Policy("standoff", "standoff"),
        "step_back": Policy("safe", "safe"),
        "break_clear": Policy("clear_hold", "hold", "clear"),
        "break_punish": Policy("clear_punish", "punish", "clear", True),
        "hold_hold": Policy("farm"),
        "clear_clear": Policy("clear_hold", "hold", "clear"),
    }


def gapped(start, cfg, gap: str):
    """The start with A ahead (x1.76 attack power, 50 vs 0 invested), behind (the mirror) or equal."""
    from ...champions_m4 import M4State
    from ..m4.zoning_natural import with_gap
    lane, m4, k = start
    if gap == "equal":
        return start
    if gap == "a_ahead":
        return lane, with_gap(m4, cfg, "large"), k
    if gap == "a_behind":
        m = with_gap(M4State(m4.b, m4.a), cfg, "large")
        return lane, M4State(m4.a._replace(items=m.b.items, gold=0), m4.b._replace(items=m.a.items, gold=0)), k
    raise ValueError(gap)


def play(cfg, start, agents, seed, turns: int) -> List[dict]:
    """One scripted window from `start`; per-turn records of both sides."""
    from ...champions_m4 import context
    from ...env import LaneEnv
    from ...env.agents import episode_streams
    lane, m4, k = start
    ctx = context(cfg)
    t_tab = ctx.t
    _, env_ss, rng_a, rng_b = episode_streams(seed)
    rngs = (rng_a, rng_b)
    env = LaneEnv(cfg, max_turns=turns, cache_size=5_000)
    obs = env.reset(seed=env_ss, root=lane, k0=k, champ=m4)
    rows = []
    for _ in range(turns):
        a = agents[0].act(obs[0], rngs[0])
        b = agents[1].act(obs[1], rngs[1])
        r = env.step(a, b)
        i = r.info
        st = r.obs[0].state
        rec = {"earned": [x / cfg.gold_scale for x in i["earned_units"]],
               "farm": [x / cfg.gold_scale for x in i["farm_units"]],
               "available": list(i["farm_available"]),
               "plates": [x / cfg.gold_scale for x in i["plate_units"]],
               "bounty": [x / cfg.gold_scale for x in i["bounty_units"]],
               "xp": list(i["xp_gain"]),
               "damage": [i["damage_taken"][s] / t_tab.hp_max[i["levels_before"][s]] for s in (0, 1)],
               "died": list(i["died"]), "ganked": list(i["ganked"]), "connected": list(i["trade_connected"]),
               "recalled": list(i["recalled"]), "away": list(i["away"]), "end": list(i["champ_end"]),
               "crash": [int(x > 0) for x in i["crash_hp"]],
               "held": int(st.bh > 0 and st.bx >= 1 and st.ah > 0 and st.ax <= -1),  # B's front at A's -1 or deeper
               "wave_a": i["waves"][0], "wave_b": i["waves"][1], "beh_b": i["behaviors"][1]}
        rows.append(rec)
        if r.done:
            break
        obs = r.obs
    return rows


def _window_stats(rows: List[dict], n: int) -> dict:
    rows = rows[:n]
    s = lambda key, side: float(sum(r[key][side] for r in rows))
    pres_b = [r for r in rows if not r["away"][1]]
    return {"gold_a": s("earned", 0), "gold_b": s("earned", 1), "farm_a": s("farm", 0), "farm_b": s("farm", 1),
            "avail_a": s("available", 0), "avail_b": s("available", 1),
            "bounty_a": s("bounty", 0), "bounty_b": s("bounty", 1), "plates_a": s("plates", 0), "plates_b": s("plates", 1),
            "xp_a": s("xp", 0), "xp_b": s("xp", 1), "dmg_a": s("damage", 0), "dmg_b": s("damage", 1),
            "deaths_a": s("died", 0), "deaths_b": s("died", 1), "ganks_a": s("ganked", 0), "ganks_b": s("ganked", 1),
            "trades_a": s("connected", 0), "trades_b": s("connected", 1), "recalls_b": s("recalled", 1),
            "away_b": s("away", 1), "crash_a": s("crash", 0), "crash_b": s("crash", 1),
            "held": float(sum(r["held"] for r in rows)),
            "pos_b": float(np.mean([r["end"][1] for r in pres_b])) if pres_b else float("nan"),
            "b_clear_share": float(np.mean([r["wave_b"] == W_CLEAR for r in pres_b])) if pres_b else float("nan"),
            "b_trade_share": float(np.mean([r["beh_b"] == B_TRADE for r in pres_b])) if pres_b else float("nan")}


def _job(args) -> tuple:
    scenario, a_name, b_name, gap, gi, episodes = args
    import torch
    from ..m4.zoning_natural import starts
    from ..rl.m43b.common import agent_from_spec
    from ..rl.m43b.references import ReactiveAgent
    torch.set_num_threads(1)
    cfg = cfg_game()
    pols = _policies()
    start = gapped(starts(cfg)[scenario], cfg, gap)
    xa = ReactiveAgent(pols[a_name], a_name)
    yb = agent_from_spec("final:base:0", cfg) if b_name == "learned" else ReactiveAgent(pols[b_name], b_name)
    t0 = time.perf_counter()
    games = []
    for ep in range(episodes):
        rows = play(cfg, start, (xa, yb), np.random.SeedSequence(FREEZE_SEED, spawn_key=(gi, ep)), max(WINDOWS))
        games.append({str(n): _window_stats(rows, n) for n in WINDOWS})
    return f"{scenario}|{a_name}|{b_name}|{gap}", {"games": games, "elapsed_s": time.perf_counter() - t0}


def summarize(games: List[dict], n: int, lam: float = 1 / 8) -> dict:
    keys = games[0][str(n)].keys()
    out = {k: mean_ci([g[str(n)][k] for g in games]) for k in keys}
    out["gold_diff_b"] = mean_ci([g[str(n)]["gold_b"] - g[str(n)]["gold_a"] for g in games])
    out["xp_diff_b"] = mean_ci([g[str(n)]["xp_b"] - g[str(n)]["xp_a"] for g in games])
    out["value_b_gold_xp"] = mean_ci([g[str(n)]["gold_b"] - g[str(n)]["gold_a"] + lam * (g[str(n)]["xp_b"] - g[str(n)]["xp_a"])
                                      for g in games])
    return out


SCENARIOS = (
    [("neutral", "hold_hold", "hold_hold"), ("neutral", "clear_clear", "clear_clear"),
     ("neutral", "clear_clear", "hold_hold")]
    + [("freeze", a, b) for a in ("freezer", "freezer_punish") for b in
       ("approach_farm", "approach_trade", "stand_off", "step_back", "break_clear", "break_punish", "learned")]
    + [("freeze", "no_freeze", b) for b in ("approach_farm", "break_clear", "learned")]
)


def run(workers: int = 9, episodes: int = EPISODES, gaps: Sequence[str] = GAPS) -> dict:
    from ..rl.m43b.common import run_jobs
    jobs = []
    for gi, gap in enumerate(gaps):
        for sc, a, b in SCENARIOS:
            if sc == "neutral" and gap != "equal":
                continue
            jobs.append((sc, a, b, gap, gi, episodes))
    t0 = time.perf_counter()
    res = dict(run_jobs(_job, jobs, workers, "freeze"))
    out = {"meta": meta(stage="freeze", windows=list(WINDOWS), episodes=episodes, seed_root=FREEZE_SEED),
           "cells": {k: {str(n): summarize(v["games"], n) for n in WINDOWS} for k, v in res.items()}}
    out["meta"]["elapsed_s"] = time.perf_counter() - t0
    save_json("freeze", out)
    return out


__all__ = ["run", "play", "gapped", "SCENARIOS"]
