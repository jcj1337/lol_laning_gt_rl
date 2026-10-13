"""The crash grid of the M4.3d study: what happens after A's wave crashes into B's tower.

A cell = (configuration, start, power gap, A's strategy, B's strategy, chance mask). Each cell plays `episodes`
paired games of `turns` turns through the environment (`sim.play`); game e of every cell uses
SeedSequence(root, spawn_key=(e,)), so two cells that differ only by one strategy, the configuration or the mask see
the same chance stream turn by turn until their states diverge.

Starts (`STARTS`): "S4" / "S10" = the M4.3c harness `siege` start at level 4 / 10 (one turn after A's crash: A's 14-HP
mass under B's tower, A in B's tower cell, A's next front at the lane cell, B in its tower cell); "N4" / "N10" = the
`neutral` start (turn 9: both 8-HP fronts and both champions at the lane cell). Gaps: "equal", "large" (A x1.76 attack
power: 50 resources invested against none, duel ratio x3.1) and "b_large" (B x1.76: A is the weak side).

Stored per cell and window ("12": the first crash and its bounce; "40"; "all" = every turn): the mean and 95% interval
of every harness window statistic and every M4.3d ledger sum, per side and as A - B, and per game the discounted gold
difference G (J of the restricted game) and the undiscounted gold differences of 12 and 40 turns (for paired
comparisons between cells).
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import numpy as np

from ..m43c import harness as h
from ..m43c import strategies as st
from . import common as c
from . import sim
from .strategies import ATTACKERS, DEFENDERS

STARTS: Dict[str, tuple] = {"S4": ("siege", 4), "S10": ("siege", 10), "N4": ("neutral", 4), "N10": ("neutral", 10)}
GAPS = ("equal", "large", "b_large")
A_SET = tuple(ATTACKERS)
B_SET = tuple(DEFENDERS)
WINDOWS = (12, 40)
H_KEYS = ("earned", "farm", "plate", "bounty", "farm_avail", "xp", "xp_avail", "xp_missed", "hp_lost", "dmg_trade",
          "dmg_gank", "dmg_tower", "trade_att", "trade_conn", "ganked", "shot", "died", "recalled", "away", "invested",
          "crashes", "crash_hp", "tower", "plates_n", "held_by")
X_KEYS = ("absence", "position", "tower", "posture", "trade", "collected", "ideal", "ideal_tower", "collected_tower",
          "tower_hp", "lane_hp", "present", "at-2", "at-1", "at+0", "at+1", "at+2", "gank_p", "exp_gank", "gank_dmg",
          "exp_tower", "tower_dmg", "trade_dmg", "tower_risk_turn", "exposure")


def _ms(vals) -> list:
    m, ci = c.mean_ci(vals)
    return [round(m, 5), round(ci, 5) if np.isfinite(ci) else None]


def _agg(summ: list, w: str) -> dict:
    out = {}
    for k in H_KEYS:
        for s, tag in ((0, "a"), (1, "b")):
            out[f"{k}_{tag}"] = _ms([g[w]["h"][k][s] for g in summ])
        out[f"{k}_diff"] = _ms([g[w]["h"][k][0] - g[w]["h"][k][1] for g in summ])
    for k in X_KEYS:
        for s, tag in ((0, "a"), (1, "b")):
            vals = [g[w]["x"][s][k] for g in summ if np.isfinite(g[w]["x"][s][k])]
            if vals:
                out[f"x_{k}_{tag}"] = _ms(vals)
    if w == "all":
        out["xp_pending_change_b"] = _ms([g[w]["h"]["xp_pending_change"][1] for g in summ])
        out["xp_pending_change_a"] = _ms([g[w]["h"]["xp_pending_change"][0] for g in summ])
        out["G"] = _ms([g[w]["h"]["G"] for g in summ])
    return out


def cell_key(cfg_name: str, start: str, gap: str, a: str, b: str, mask: Optional[str]) -> str:
    return f"{cfg_name}|{start}|{gap}|{a}|{b}|{mask or '-'}"


def _job(args) -> tuple:
    cfg_name, start, gap, a, b, mask, root, n, turns = args
    cfg = c.variant_cfg(cfg_name)
    base, level = STARTS[start]
    stt = h.start(base, cfg, gap=gap, level=level)
    sa, sb = ATTACKERS[a], DEFENDERS[b]
    summ = []
    for ss in h.seeds(root, n):
        g = sim.play(cfg, stt, (st.StrategyAgent(sa), st.StrategyAgent(sb)), ss, turns, mask=mask)
        summ.append(sim.summarize(g, cfg, WINDOWS))
    out = {"n": n, "per_game": {"G": [round(g["all"]["h"]["G"], 5) for g in summ],
                                "G40": [round(g["40"]["h"]["earned"][0] - g["40"]["h"]["earned"][1], 5) for g in summ],
                                "G12": [round(g["12"]["h"]["earned"][0] - g["12"]["h"]["earned"][1], 5) for g in summ],
                                "farmB40": [round(g["40"]["h"]["farm"][1], 5) for g in summ],
                                "farmA40": [round(g["40"]["h"]["farm"][0], 5) for g in summ]}}
    for w in ("12", "40", "all"):
        out[w] = _agg(summ, w)
    return cell_key(cfg_name, start, gap, a, b, mask), out


def run(tag: str, configs: Sequence[str], starts: Sequence[str] = tuple(STARTS), gaps: Sequence[str] = GAPS,
        a_set: Sequence[str] = A_SET, b_set: Sequence[str] = B_SET, masks: Sequence[Optional[str]] = (None,),
        root: int = c.TOWER_FARM_SEED, episodes: int = 64, turns: int = 100, workers: int = 11,
        merge: bool = True) -> dict:
    """Play every cell not yet stored in results/m43d/grid_<tag>; store and return the grid."""
    t0 = time.perf_counter()
    old = c.load(f"grid_{tag}") if merge else None
    cells = dict(old["cells"]) if old else {}
    jobs = [(cf, sn, gp, a, b, m, root, episodes, turns) for cf in configs for sn in starts for gp in gaps
            for a in a_set for b in b_set for m in masks
            if cell_key(cf, sn, gp, a, b, m) not in cells]
    print(f"grid {tag}: {len(jobs)} cells to play ({len(cells)} stored)", flush=True)
    cells.update(dict(c.run_jobs(_job, jobs, workers, f"grid {tag}")))
    variants = sorted({k.split("|")[0] for k in cells})
    res = {"meta": c.meta(stage="grid", tag=tag, episodes=episodes, turns=turns, root=root, windows=list(WINDOWS),
                          starts={k: list(v) for k, v in STARTS.items()},
                          variants={v: {"description": c.VARIANTS[v][0], "preset": c.VARIANTS[v][1],
                                        "overrides": c.overrides_json(v)} for v in variants},
                          elapsed_s=time.perf_counter() - t0 + (old["meta"].get("elapsed_s", 0.0) if old else 0.0)),
           "cells": cells}
    c.save_json(f"grid_{tag}", res)
    return res


__all__ = ["run", "cell_key", "STARTS", "GAPS", "A_SET", "B_SET", "WINDOWS"]
