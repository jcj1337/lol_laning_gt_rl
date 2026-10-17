"""Supplementary re-check (protocol amendment of 2026-10-17; descriptive, not part of the classification): M4.7's
primary readout, the short arm's lane-phase lead P1 (turns 0-93 of the head-to-head games against the control), with
the lane-exit readout at turn 94.

M4.7's head-to-head games (`m47.evaluate.h2h_job`): game m of seed s uses chance seed 58,400 * 10,000 + m from s0, x =
p/short:s and y = p/control:s; orientation x played x in seat A and y in seat B, orientation y the reverse, each seat
with CRN sampling streams (seed, 7, seat). Each game is replayed to turn 94 here and checked against M4.7's stored
reward stream; from the state at turn 94 the same pair continues K_RECHECK times for T_RECHECK turns (root
RECHECK_ROOT). P1 per game = the mean over the two orientations of the short arm's raw lead at turn 94; P1_h replaces
the raw lead by E_h.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from ....diagnostics import lane_exit as LX
from . import common as C

K_RECHECK, T_RECHECK = 8, 20
RECHECK_ROOT = 59_400
GAMES_PER_JOB = 64
LANE_END = 94


def job(j: dict) -> tuple:
    path = j["path"]
    if os.path.exists(path):
        return j["key"], {"path": path, "cached": True}
    from ...m45b import states as S
    from ..m47 import common as C47
    from ..m47 import evaluate as EV
    cfg = C.setup("baseline")
    s = j["s"]
    x, y = f"p/short:{s}", f"p/control:{s}"
    nets = {x: C47.net_of(x), y: C47.net_of(y)}
    stored = EV.load_pair("h2h", s, x, y)
    pos = {int(m): i for i, m in enumerate(stored["games"])}
    start = S.s0(cfg)
    ls = C.lane_step(cfg)
    from ..m45c import horizon as HZ
    out = {"games": [], "D0": [], "E": {h: [] for h in (5, 10, 20)}, "max_dev": 0.0}
    for m in j["games"]:
        sd = C47.H2H_ROOT * 10_000 + m
        row_D0, row_E = [], {h: [] for h in (5, 10, 20)}
        for o, (a, b) in enumerate(((x, y), (y, x))):
            g = LX.continuation(cfg, start, HZ.CRNNet(nets[a], 0, sd, 0, 400), HZ.CRNNet(nets[b], 0, sd, 1, 400), sd,
                                LANE_END, lane_step=ls, keep_states=True)
            ref = stored["rx" if o == 0 else "ry"][pos[m], :LANE_END].astype(float)
            out["max_dev"] = max(out["max_dev"], float(np.abs(g["r"] - ref).max()))
            st = g["states"][-1]
            D0 = LX.lead(st[1], cfg)
            k0 = st[2]
            seeds = [C.roll_seed(RECHECK_ROOT, (s * 2048 + m) * 2 + o, r) for r in range(K_RECHECK)]
            led = LX.continuations(cfg, st, lambda q: (HZ.CRNNet(nets[a], k0, q, 0, T_RECHECK),
                                                       HZ.CRNNet(nets[b], k0, q, 1, T_RECHECK)),
                                   seeds, T_RECHECK, lane_step=ls)
            sign = 1.0 if o == 0 else -1.0                      # the short arm's orientation
            row_D0.append(sign * D0)
            for h in (5, 10, 20):
                row_E[h].append(sign * (D0 + float(LX.deltas(led["r"], h).mean())))
        out["games"].append(int(m))
        out["D0"].append(float(np.mean(row_D0)))
        for h in (5, 10, 20):
            out["E"][h].append(float(np.mean(row_E[h])))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    C.save_gz(path, out)
    return j["key"], {"path": path, "max_dev": out["max_dev"]}


def jobs() -> List[dict]:
    from ..m47 import common as C47
    from ..m47 import evaluate as EV
    js = []
    for s in C47.PILOT_SEEDS:
        for b in range(0, EV.N_H2H, GAMES_PER_JOB):
            key = f"recheck|s{s}|{b}"
            js.append({"key": key, "s": s, "games": list(range(b, min(EV.N_H2H, b + GAMES_PER_JOB))),
                       "path": os.path.join(C.CKPT, "recheck", f"s{s}_{b}.json.gz")})
    return js


def run(workers: int) -> dict:
    js = jobs()
    C.run_jobs(job, js, workers, "recheck M4.7 P1")
    out = {"meta": C.meta(stage="recheck"), "K": K_RECHECK, "T": T_RECHECK, "per_seed": {}}
    for s in sorted({j["s"] for j in js}):
        parts = [C.load_gz(j["path"]) for j in js if j["s"] == s]
        D0 = np.concatenate([p["D0"] for p in parts])
        o = {"n_games": int(len(D0)), "max_replay_dev": float(max(p["max_dev"] for p in parts)),
             "P1_raw": C.mean_se(D0)}
        for h in (5, 10, 20):
            E = np.concatenate([p["E"][str(h)] if str(h) in p["E"] else p["E"][h] for p in parts])
            o[f"P1_E{h}"] = C.mean_se(E)
            o[f"P1_E{h}-raw"] = C.mean_se(E - D0)
        out["per_seed"][str(s)] = o
    C.save_json("recheck", out)
    for s, o in out["per_seed"].items():
        print(f"  s{s}: raw {o['P1_raw']['mean']:+.3f} ± {o['P1_raw']['ci']:.3f}; " + "; ".join(
            f"E{h} {o[f'P1_E{h}']['mean']:+.3f} ± {o[f'P1_E{h}']['ci']:.3f}" for h in (5, 10, 20))
              + f"; replay dev {o['max_replay_dev']:.2g}", flush=True)
    return out


__all__ = ["job", "jobs", "run", "K_RECHECK", "T_RECHECK", "RECHECK_ROOT"]
