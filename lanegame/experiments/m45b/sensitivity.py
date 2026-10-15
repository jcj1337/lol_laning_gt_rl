"""Joint sensitivity analysis and candidate selection (M4.5b §9).

A BATTERY of fast indicator experiments, each a reduced version of a Priority stage, is run for every configuration
of a small factorial around the candidate; each indicator is one number with a paired 95% interval where it is
stochastic. Indicators (DEV seeds and DEV states only):

  trade        the six exact Priority B criteria (`trades.criteria`), and the multi-turn paired G20 of Trade versus
               Hold on the first turn at (L2, 24 HP, cover, equal) and (L2, 8 HP, cover, equal)
  freeze       on (L10, large, held 12) and (L4, equal, held 12): the split-sample freeze premium (G100) of the
               restricted freezer x response game, and the paired G100 of guarding versus passive freezing against the
               walk-up Shove
  vision       the paired G50 of Ward versus farming on the first turn, base g_farm against g_bully, in the neutral and
               the frozen_on situations (level 4)
  zoning       the passive cost of stepping back (B's earned resources at 50 turns, g_safe - g_farm, against g_farm)
               and the threat-only zoning effect (g_guard v g_wary minus g_farm v g_farm), neutral lane, L6, A x1.76
  matrix       a symmetric 8-strategy restricted game from s0 (G200, antisymmetrized): its equilibrium support and
               Shove's worst-case margin (the minimum over columns of the g_shove row)

`factorial` runs the battery on a grid of overrides; `main_effects` summarizes each indicator's range over each
parameter's levels (the others averaged) and flags the indicators whose sign changes inside the grid.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import numpy as np

from ...matrix_game import solve_matrix_game
from . import common as Cm
from . import engine as E
from . import freeze as F
from . import matrix as MX
from . import states as S
from . import strategies as ST
from . import trades as T
from . import vision as V

MATRIX8 = ("g_farm", "g_thin", "g_shove", "g_shove_sr", "g_guard", "g_safe", "g_ward", "fz_guard_sh")
FREEZE_ROWS = ("fz_passive_sh", "fz_guard_sh", "fz_guard_sh_sr", "g_shove", "shove_recall")
FREEZE_COLS = ("wk_shove_sh", "wk_lh_sh", "wk_back_sh", "wk_shove_reset_sh", "wk_wary_shove_sh", "wk_shove_safe_sh")


def _paired_games(c, start, pa1, pa2, pb, seeds, turns, metric):
    r1 = [E.play(c, start, pa1, pb, sd, turns)["summary"][metric] for sd in seeds]
    r2 = [E.play(c, start, pa2, pb, sd, turns)["summary"][metric] for sd in seeds]
    return Cm.paired(r1, r2)


def _ind_trade(over):
    c = Cm.cfg("r2", **dict(over))
    rows = T.bench_table(c, covers=("cover",), gaps=("equal", "large"), b_acts=("hold",), hps=((1.0, 1.0),))
    crit = T.criteria(rows)
    out = {f"trade_{k}": v["value"] for k, v in crit.items()}
    out["trade_pass"] = int(sum(v["pass"] for v in crit.values()))
    for w in (8, 24):
        r = T._mt_job((tuple(over), 2, w, "cover", "equal", "g_farm", "g_farm", tuple(Cm.DEV[:200]), 20))
        out[f"trade_mt_L2_{w}_G20"] = r["G20"]
    return out


def _ind_freeze(over, state):
    c = Cm.cfg("r2", **dict(over))
    cells = [F._cell((tuple(over), state, a, b, tuple(Cm.DEV[:40]), 100)) for a in FREEZE_ROWS for b in FREEZE_COLS]
    an = F.analyse(cells, FREEZE_ROWS, FREEZE_COLS)
    tag = f"L{state[0]}_{state[1]}_h{state[2]}"
    out = {f"freeze_{tag}_premium_split": an["G100"]["premium"]["premium_split"][0],
           f"freeze_{tag}_value": an["G100"]["value"],
           f"freeze_{tag}_guard_minus_passive_vs_shove": an["contrasts_G100"]["fz_guard_sh - fz_passive_sh"]["wk_shove_sh"],
           f"freeze_{tag}_guard_minus_shove_vs_shove": an["contrasts_G100"]["fz_guard_sh - g_shove"]["wk_shove_sh"],
           f"freeze_{tag}_support": an["G100"]["A"]}
    return out


def _ind_vision(over):
    out = {}
    for sit in ("neutral", "frozen_on"):
        r = V._local((tuple(over), sit, 4, "equal", "g_farm", "g_bully", tuple(Cm.DEV[:200]), 60))
        out[f"vision_{sit}_G50"] = r["G50"]
        out[f"vision_{sit}_placed"] = r["placed_first"]
    return out


def _ind_zoning(over):
    from .zoning import _cell
    def ps(a, b):
        return _cell((tuple(over), "neutral", 6, "large", a, b, tuple(Cm.DEV[:64]), 60))["per_seed"]["earned_B@50"]
    ff = ps("g_farm", "g_farm")
    return {"zoning_passive_safe_cost": Cm.paired(ff, ps("g_farm", "g_safe")),
            "zoning_threat_only": Cm.paired(ff, ps("g_guard", "g_wary")),
            "zoning_combat_guard": Cm.paired(ff, ps("g_guard", "g_farm"))}


def _ind_matrix(over):
    res = MX.run("r2", tuple(over), "s0", "equal", MATRIX8, seeds=Cm.DEV[:32], turns=200, workers=1)
    an = res["analysis"]["antisym"]
    M = np.array(res["M"])
    Ma = (M - M.T) / 2
    i = MATRIX8.index("g_shove")
    return {"matrix_support": an["A"], "matrix_shove_margin": float(Ma[i].min()),
            "matrix_n_support": len(an["A"])}


def _battery_job(args):
    kind, over = args
    if kind == "trade":
        return _ind_trade(over)
    if kind == "freeze_a":
        return _ind_freeze(over, (10, "large", 12, 1.0))
    if kind == "freeze_b":
        return _ind_freeze(over, (4, "equal", 12, 1.0))
    if kind == "vision":
        return _ind_vision(over)
    if kind == "zoning":
        return _ind_zoning(over)
    if kind == "matrix":
        return _ind_matrix(over)
    raise ValueError(kind)


KINDS = ("trade", "freeze_a", "freeze_b", "vision", "zoning", "matrix")


def factorial(grid: Dict[str, Sequence], kinds: Sequence[str] = KINDS, workers: int = Cm.WORKERS) -> List[dict]:
    confs = [tuple(zip(grid, v)) for v in itertools.product(*grid.values())]
    jobs = [(k, cf) for cf in confs for k in kinds]
    res = Cm.run_jobs(_battery_job, jobs, workers, f"sensitivity {len(confs)} configurations")
    out = []
    for i, cf in enumerate(confs):
        d = {"overrides": dict(cf)}
        for j in range(len(kinds)):
            d.update(res[i * len(kinds) + j])
        out.append(d)
    return out


def scalar(v):
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
        return v[0]
    return v if isinstance(v, (int, float)) else None


def main_effects(rows: List[dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per indicator and parameter: the indicator's mean at each level of the parameter (the others averaged)."""
    params = list(rows[0]["overrides"])
    inds = [k for k in rows[0] if k != "overrides" and scalar(rows[0][k]) is not None]
    out = {}
    for ind in inds:
        out[ind] = {}
        for p in params:
            levels = sorted({r["overrides"][p] for r in rows})
            out[ind][p] = {str(l): float(np.mean([scalar(r[ind]) for r in rows if r["overrides"][p] == l]))
                           for l in levels}
        vals = [scalar(r[ind]) for r in rows]
        out[ind]["_range"] = {"min": float(np.min(vals)), "max": float(np.max(vals)),
                              "sign_changes": bool(np.min(vals) < 0 < np.max(vals))}
    return out


__all__ = ["factorial", "main_effects", "KINDS", "MATRIX8"]
