"""The holdout evaluation (M4.5b §E): the selected candidate against the M4.5a revision-2 baseline on the HOLDOUT seeds
(88,000-88,499) and the HOLDOUT_GRID states (`states.HOLDOUT_SPLIT`), which no calibration stage used. Every block runs
the same code as its development stage, with the holdout seeds and states, once per configuration:

    trades   the exact bench at levels 3 / 8 / 14 and enemy waves 12 / 20 HP, with the holdout analogues of the Priority B
             criteria (HB1-HB5), and the multi-turn Trade-versus-Hold values on holdout bench states
    freeze   the freezer x response grid on level {3, 7, 13, 16} x gap adv (x1.5) x held {8, 16} x A's HP {0.8, 0.45}
    vision   the one-ward values in the holdout situations (push_lane, deep_farm) at levels 7 / 13, gap adv; the Ward-rule
             contrasts from s0 and from push_lane
    zoning   the behaviour matrix and decomposition in the b_push situation at levels 7 / 13, gap adv
    matrix   the integrated restricted game from s0 (holdout seeds), and 8-strategy games from NATURAL states: the states
             reached at turns 60 and 150 of g_shove_guard v g_guard games from s0 (holdout seeds)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import common as Cm
from . import engine as E
from . import freeze as F
from . import matrix as MX
from . import sensitivity as SE
from . import states as S
from . import strategies as ST
from . import trades as T
from . import vision as V
from . import zoning as Z

HG = S.HOLDOUT_GRID
H_TRADE_LEVELS, H_TRADE_WAVES = HG["trade_levels"], HG["trade_waves"]
H_FREEZE = [(L, g, h, hp) for L in HG["levels"] for g in HG["gaps"] for h in HG["held"] for hp in HG["hp"]]

HCRITERIA = {
    "HB1": "L3, equal power, own cover, B Hold: trading into a 20-HP wave is not HP-positive (delta <= 0)",
    "HB2": "L3, equal power, own cover: delta into a 12-HP wave exceeds delta into a 20-HP wave",
    "HB3": "L3, A x1.5, own cover, 20-HP wave: delta >= +0.05 bar",
    "HB4": "L14, equal power, own cover, 20-HP wave: minion fire <= 15% of the trade damage dealt",
    "HB5": "equal power, own cover, 20-HP wave: the minion share falls strictly over levels 3, 8, 14",
}


def h_criteria(rows: List[dict]) -> Dict[str, dict]:
    def get(**kw):
        return next(r for r in rows if all(r[k] == v for k, v in kw.items()) and r["hp"] == [1.0, 1.0])
    out = {}
    v = T.delta(get(level=3, wave=20, cover="cover", gap="equal", b="hold"))
    out["HB1"] = {"value": v, "pass": v <= 0}
    a, b = (T.delta(get(level=3, wave=w, cover="cover", gap="equal", b="hold")) for w in (12, 20))
    out["HB2"] = {"value": [a, b], "pass": a > b}
    v = T.delta(get(level=3, wave=20, cover="cover", gap="adv", b="hold"))
    out["HB3"] = {"value": v, "pass": v >= 0.05}
    v = get(level=14, wave=20, cover="cover", gap="equal", b="hold")["minion_share"]
    out["HB4"] = {"value": v, "pass": v <= 0.15}
    sh = [get(level=L, wave=20, cover="cover", gap="equal", b="hold")["minion_share"] for L in H_TRADE_LEVELS]
    out["HB5"] = {"value": sh, "pass": all(x > y for x, y in zip(sh, sh[1:]))}
    return out


def trades_block(over) -> dict:
    c = Cm.cfg("r2", **dict(over))
    rows = T.bench_table(c, levels=H_TRADE_LEVELS, waves=H_TRADE_WAVES, covers=("cover", "nocover"),
                         gaps=("equal", "adv"), b_acts=("hold", "safe"), hps=((1.0, 1.0),))
    mt = T.multi_turn(over, levels=H_TRADE_LEVELS, waves=H_TRADE_WAVES, covers=("cover",), gaps=("equal", "adv"),
                      seeds=Cm.HOLDOUT[:200])
    return {"criteria": h_criteria(rows), "rows": rows, "multi_turn": mt}


def natural_starts(cfg, turns=(60, 150), pair=("g_shove_guard", "g_guard"), seeds=Cm.HOLDOUT[400:404]) -> List[tuple]:
    out = []
    for sd in seeds:
        g = E.play(cfg, S.s0(cfg), ST.get(pair[0]), ST.get(pair[1]), sd, max(turns) + 1, keep_states=True)
        for t in turns:
            lane, m4, k = g["states"][t]
            if m4.a.absent == 0 and m4.b.absent == 0:
                out.append((sd, t, (lane, m4, k)))
    return out


def _nat_job(args):
    over, idx, a, b, seeds, turns = args
    c = Cm.cfg("r2", **dict(over))
    _, _, start = natural_starts(c)[idx]
    rows = [E.play(c, start, ST.get(a), ST.get(b), sd, turns)["summary"]["G200"] for sd in seeds]
    return float(np.mean(rows))


def matrix_block(over, workers: int = Cm.WORKERS) -> dict:
    res = MX.run("r2", tuple(over), "s0", "equal", MX.SET_FULL, seeds=Cm.HOLDOUT[:48], turns=200, workers=workers)
    res["bootstrap"] = MX.equilibrium_ci(res)
    c = Cm.cfg("r2", **dict(over))
    nat = natural_starts(c)
    names = SE.MATRIX8
    games = []
    for i, (sd, t, _) in enumerate(nat):
        jobs = [(tuple(over), i, a, b, tuple(Cm.HOLDOUT[100:132]), 200) for a in names for b in names]
        vals = Cm.run_jobs(_nat_job, jobs, workers, f"natural matrix {i}")
        M = np.array(vals).reshape(len(names), len(names))
        games.append({"seed": sd, "turn": t, "M": M.tolist(), "analysis": MX.solve(M, names, False)})
    return {"s0": {k: v for k, v in res.items() if k != "per_seed"}, "natural": games}


def run(over=(), blocks=("trades", "freeze", "vision", "zoning", "matrix"), workers: int = Cm.WORKERS) -> dict:
    over = tuple(over)
    out = {"overrides": dict(over)}
    if "trades" in blocks:
        out["trades"] = trades_block(over)
    if "freeze" in blocks:
        out["freeze"] = F.grid(over, H_FREEZE, seeds=Cm.HOLDOUT[:40], workers=workers)
    if "vision" in blocks:
        out["vision_local"] = V.local(over, situations=HG["vision_situations"], levels=(7, 13), gaps=HG["gaps"],
                                      bases=("g_farm", "g_shove"), opponents=V.OPPONENTS, seeds=Cm.HOLDOUT[:200],
                                      workers=workers)
        out["vision_strategies"] = V.strategies(over, starts=(("s0", "equal"), ("push_lane", "adv")),
                                                seeds=Cm.HOLDOUT[:48], workers=workers)["contrasts"]
    if "zoning" in blocks:
        out["zoning"] = Z.matrix(over, situations=HG["zoning"], levels=(7, 13), gaps=HG["gaps"],
                                 seeds=Cm.HOLDOUT[:64], workers=workers)
    if "matrix" in blocks:
        out["matrix"] = matrix_block(over, workers)
    return out


__all__ = ["run", "h_criteria", "HCRITERIA", "H_FREEZE", "natural_starts"]
