"""Priority C: Vision positioning and economics.

1. `local` (the value of one ward): from the Vision situations (`states.VISION_SITUATIONS`: neutral, forward,
   presiege, defending, frozen_on; INITIALIZED) at a level and power gap, A's first action is Ward (where legal) or the
   continuation's own action; both then continue with stated strategies (A: `base`, B: an opponent style). Paired G1
   (the turn's farm), G20, G50, G100, and the 10-turn ledger: wards placed / interrupted, ganks, gank damage, deaths.
2. `strategies`: Ward rules (none, favorable, forward, exposed, away, always; strategies.WARD_RULES) on top of base
   strategies against opponent styles, from s0 and from situations, 200 turns: G200, wards attempted / placed /
   interrupted, the share of protected turns played at an exposure cell >= 0, gank damage and gank deaths per game.
3. `sweep`: (1) and (2)'s key cells under vision_turns {4, 6, 8} x vision_reduction {0.5, 0.75} and ward_exposure.
"""

from __future__ import annotations

import itertools
from typing import List, Sequence

import numpy as np

from ...config import CHAMP_WARD
from . import common as Cm
from . import engine as E
from . import states as S
from . import strategies as ST
from .strategies import C

SITUATIONS = ("neutral", "forward", "presiege", "defending", "frozen_on")
OPPONENTS = ("g_farm", "g_shove", "g_guard", "g_bully")
BASES = ("g_farm", "g_shove", "g_guard")
LEDGER10 = ("ward_placed_A@10", "deaths_A@10", "d_total_A@10", "earned_A@10", "xp_A@10", "deaths_B@10")


def warded(base: str, rule) -> object:
    """`base` with a Ward rule (None = the base itself)."""
    if rule is None:
        return ST.get(base)
    p = ST.get(base)
    q = C(f"{base}+w_{rule}", wave=p.wave, beh=p.beh, wave_freeze=p.wave_freeze, beh_freeze=p.beh_freeze,
          wave_frozen=p.wave_frozen, beh_frozen=p.beh_frozen, recall_frac=p.recall_frac, reset_frac=p.reset_frac,
          trade_hp=p.trade_hp, edge_min=p.edge_min, trade_line=p.trade_line, shove_recall=p.shove_recall,
          frozen_reset=p.frozen_reset, ward=rule, pressure=p.pressure,
          description=f"{base} + Ward rule {rule!r}")
    return q


def _local(args):
    over, sit, level, gap, base, opp, seeds, turns = args
    c = Cm.cfg("r2", **dict(over))
    start = S.vision_state(c, sit, level, gap)
    a_cont, b_pol = ST.get(base), ST.get(opp)
    from ...champions import champion_legal_actions
    lane, m4, k = start
    if CHAMP_WARD not in champion_legal_actions(lane, m4, 0, c):
        return {"situation": sit, "level": level, "gap": gap, "base": base, "opponent": opp, "legal": False}
    plans = {"ward": E.Then("ward", E.Fixed("W", CHAMP_WARD, a_cont), a_cont, 1), "base": a_cont}
    res = {k_: [E.play(c, start, p, b_pol, sd, turns)["summary"] for sd in seeds] for k_, p in plans.items()}
    out = {"situation": sit, "level": level, "gap": gap, "base": base, "opponent": opp, "legal": True,
           "overrides": dict(over)}
    for met in ("G1", "G5", "G20", "G50", "G100"):
        if met in res["ward"][0]:
            out[met] = Cm.paired([r[met] for r in res["ward"]], [r[met] for r in res["base"]])
    for key in LEDGER10 + ("ganks_A", "d_gank_A", "d_by_gank_A"):
        out[key] = {k_: float(np.mean([r[key] for r in res[k_]])) for k_ in plans}
    out["placed_first"] = float(np.mean([r["ward_placed_A@10"] > 0 for r in res["ward"]]))
    return out


def local(over=(), situations=SITUATIONS, levels=(4, 10), gaps=("equal", "large", "weak"), bases=("g_farm", "g_shove"),
          opponents=OPPONENTS, seeds=Cm.DEV[:200], turns: int = 100, workers: int = Cm.WORKERS) -> List[dict]:
    jobs = [(tuple(over), s, L, g, b, o, tuple(seeds), turns) for s in situations for L in levels for g in gaps
            for b in bases for o in opponents]
    return Cm.run_jobs(_local, jobs, workers, f"vision local {dict(over)}")


def _strat(args):
    over, start_name, gap, base, rule, opp, seeds, turns = args
    c = Cm.cfg("r2", **dict(over))
    from .matrix import start_of
    start = start_of(c, start_name, gap)
    pa, pb = warded(base, rule), ST.get(opp)
    rows = [E.play(c, start, pa, pb, sd, turns)["summary"] for sd in seeds]
    out = {"start": start_name, "gap": gap, "base": base, "rule": rule, "opponent": opp, "overrides": dict(over),
           "per_seed": [r["G200"] for r in rows]}
    for k in ("G200", "G50", "wards_A", "ward_placed_A", "ward_intr_A", "vis_turns_A", "vis_exposed_A", "ganks_A",
              "d_gank_A", "d_by_gank_A", "deaths_A", "at_W_A", "farm_A", "d_min_pos_A", "t_pos_A", "xp_A@50",
              "xp_B@50", "lead"):
        out[k] = float(np.mean([r[k] for r in rows]))
    return out


RULES = (None, "favorable", "forward", "exposed", "away", "always")


def strategies(over=(), starts=(("s0", "equal"), ("neutral", "equal"), ("neutral", "large"), ("neutral", "weak")),
               bases=BASES, rules=RULES, opponents=OPPONENTS, seeds=Cm.DEV[:48], turns: int = 200,
               workers: int = Cm.WORKERS) -> dict:
    jobs = [(tuple(over), s, g, b, r, o, tuple(seeds), turns) for s, g in starts for b in bases for r in rules
            for o in opponents]
    cells = Cm.run_jobs(_strat, jobs, workers, f"vision strategies {dict(over)}")
    # paired contrast of each Ward rule against its base (same seeds, same opponent)
    idx = {(c["start"], c["gap"], c["base"], c["rule"], c["opponent"]): c for c in cells}
    con = []
    for (s, g, b, r, o), cl in idx.items():
        if r is None:
            continue
        ref = idx[(s, g, b, None, o)]
        con.append({"start": s, "gap": g, "base": b, "rule": r, "opponent": o,
                    "dG200": Cm.paired(cl["per_seed"], ref["per_seed"]), "wards": cl["wards_A"],
                    "placed": cl["ward_placed_A"], "interrupted": cl["ward_intr_A"],
                    "exposed_share": cl["vis_exposed_A"] / cl["vis_turns_A"] if cl["vis_turns_A"] else float("nan"),
                    "d_gank": [cl["d_gank_A"], ref["d_gank_A"]], "gank_deaths": [cl["d_by_gank_A"], ref["d_by_gank_A"]]})
    return {"cells": [{k: v for k, v in c.items() if k != "per_seed"} for c in cells], "contrasts": con}


SWEEP = [dict(vision_turns=t, vision_reduction=r) for t in (4, 6, 8) for r in (0.5, 0.75)] + \
        [dict(ward_exposure=e) for e in (-1, 1, 2)]


def sweep(seeds_local=Cm.DEV[:120], seeds_strat=Cm.DEV[:32], workers: int = Cm.WORKERS) -> List[dict]:
    out = []
    for over in SWEEP:
        ov = tuple(over.items())
        loc = local(ov, situations=("neutral", "forward", "presiege", "frozen_on"), levels=(4, 10), gaps=("equal",),
                    bases=("g_farm", "g_shove"), opponents=("g_farm", "g_bully"), seeds=seeds_local, workers=workers)
        st = strategies(ov, starts=(("s0", "equal"), ("neutral", "equal")), bases=("g_farm", "g_shove"),
                        rules=(None, "favorable", "exposed", "always"), opponents=("g_farm", "g_shove", "g_bully"),
                        seeds=seeds_strat, workers=workers)
        out.append({"overrides": over, "local": loc, "strategies": st["contrasts"]})
    return out


__all__ = ["local", "strategies", "sweep", "warded", "SITUATIONS", "OPPONENTS", "RULES", "SWEEP"]
