"""Representative trajectories of M4.6c (exploratory, labelled): turn-by-turn traces of a policy against itself from
a chosen start, under its own environment, on TRACE_ROOT seeds. Each row: the turn, both seats' action labels, HP
fractions after the turn, cells, connected trades, deaths, Recalls, freeze flags, invested gold, each side's capacity C
and the cumulative earned lead of seat A. The same seed is used for both arms, so the two traces of a scenario share
their chance stream and sampling uniforms until the trajectories diverge."""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from ....champions import N_CHANCE
from . import common as C


def trace(env: str, spec: str, start, seed: int, turns: int) -> List[dict]:
    cfg = C.setup(env)
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    from ..m44.records import holds_freeze
    from ..m45c import horizon as HZ
    ctx = context(cfg)
    tb = ctx.t
    ls = E.lane_cache(cfg)
    net = C.net_of(spec)
    lane, m4, k0 = start
    pa, pb = HZ.CRNNet(net, k0, seed, 0, turns), HZ.CRNNet(net, k0, seed, 1, turns)
    rng = np.random.default_rng(seed)
    rows, lead = [], 0.0
    sc = cfg.gold_scale
    for i in range(turns):
        u = rng.random(N_CHANCE)
        a = E.decide(pa, cfg, lane, m4, 0, i)
        b = E.decide(pb, cfg, lane, m4, 1, i)
        caps = []
        for x in (0, 1):
            me = m4.side(x)
            L = tb.level_by_xp[min(me.xp, tb.xp_cap)]
            caps.append(int(ctx.capacity(L, me.items)))
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=ls, u=u, ctx=ctx)
        lane, m4 = t.lane.state, t.champ
        lead += (t.earned_units[0] - t.earned_units[1]) / sc
        hp = []
        for x in (0, 1):
            me = m4.side(x)
            L = tb.level_by_xp[min(me.xp, tb.xp_cap)]
            hp.append(round(me.hp / tb.hp_max[L], 2) if me.absent == 0 else None)
        rows.append({"turn": i + 1, "A": cfg.label(a), "B": cfg.label(b), "hp": hp, "cells": list(t.end),
                     "conn": list(t.connected), "died": list(t.died), "recall": list(t.recalled),
                     "held": [holds_freeze(lane, 0), holds_freeze(lane, 1)], "S": [m4.a.items / sc, m4.b.items / sc],
                     "C": caps, "crash": list(t.lane.crash_hp), "lead_A": round(lead, 3)})
    return rows


def scenarios() -> Dict[str, tuple]:
    from .archive import freeze_cell, lead_cell
    return {"mid neutral, A +16 invested": (lambda cfg: lead_cell(cfg, "mid", "neutral", "A+16"), 30),
            "early neutral, A +16 invested": (lambda cfg: lead_cell(cfg, "early", "neutral", "A+16"), 30),
            "freeze level 10, held 12, breaker (B) +8": (lambda cfg: freeze_cell(cfg, 10, 12, "breaker+8"), 20),
            "freeze level 4, held 12, freezer (A) +8": (lambda cfg: freeze_cell(cfg, 4, 12, "freezer+8"), 20)}


def run(seeds=(0,), trace_seeds=(0, 1), prefix: str = "ext/") -> dict:
    out = {}
    for name, (build, turns) in scenarios().items():
        for s in seeds:
            for ts in trace_seeds:
                sd = C.TRACE_ROOT * 100 + ts
                for env in C.ARMS:
                    cfg = C.arm_cfg(env)
                    out[f"{name}|{env}|s{s}|t{ts}"] = trace(env, f"{prefix}{env}:{s}", build(cfg), sd, turns)
    C.save_json("traces", {"meta": C.meta(stage="traces", label="EXPLORATORY", policies=prefix or "4.096M"),
                           "traces": out})
    return out


__all__ = ["trace", "run", "scenarios"]
