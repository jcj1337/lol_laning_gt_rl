"""M4.5b stages (docs/m4/m4_5/m4_5b_calibration_report.md). Results go to results/m45b/<stage>.json (gzipped when large);
`report` writes results/m45b/report.md from them.

    python -m lanegame.experiments.m45b split            fix the holdout split (results/m45b/holdout_split.json)
    python -m lanegame.experiments.m45b <stage> [...]    one or more stages, in order (STAGES)
"""

from __future__ import annotations

import os
import sys
import time

from . import common as Cm

STAGES = {}


def stage(name):
    def deco(fn):
        STAGES[name] = fn
        return fn
    return deco


@stage("split")
def _split():
    from .states import HOLDOUT_SPLIT
    p = os.path.join(Cm.OUT, "holdout_split.json")
    if os.path.exists(p):
        print(f"{p} exists: the split is fixed; not rewritten")
        return
    Cm.save_json("holdout_split", {"meta": Cm.meta(), **HOLDOUT_SPLIT})


@stage("trades")
def _trades():
    from . import trades as T
    c = Cm.cfg("r2")
    t0 = time.time()
    table = T.bench_table(c)
    crit = T.criteria(table)
    sweep = T.sweep()
    price = T.hp_price()
    mt = T.multi_turn()
    Cm.save_json("trades", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), "criteria_text": T.CRITERIA,
                            "table": table, "criteria": crit, "sweep": [{k: v for k, v in s.items() if k != "rows"}
                                                                         for s in sweep],
                            "sweep_rows": sweep, "hp_price": price, "multi_turn": mt})


@stage("freeze")
def _freeze():
    from . import freeze as F
    t0 = time.time()
    g = F.grid()
    Cm.save_json("freeze_grid", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), "grid": g})


@stage("freeze_local")
def _freeze_local():
    from . import freeze as F
    t0 = time.time()
    loc = F.local()
    res = F.reset()
    Cm.save_json("freeze_local", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), "local": loc, "reset": res})


@stage("freeze_sweep")
def _freeze_sweep():
    from . import freeze as F
    t0 = time.time()
    sw = F.sweep()
    Cm.save_json("freeze_sweep", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), "sweep": sw})


@stage("vision")
def _vision():
    from . import vision as V
    t0 = time.time()
    loc = V.local()
    st = V.strategies()
    Cm.save_json("vision", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), "local": loc, "strategies": st})


@stage("vision_sweep")
def _vision_sweep():
    from . import vision as V
    t0 = time.time()
    sw = V.sweep()
    Cm.save_json("vision_sweep", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), "sweep": sw})


@stage("zoning")
def _zoning():
    from . import zoning as Z
    t0 = time.time()
    one = Z.one_turn()
    mat = Z.matrix()
    Cm.save_json("zoning", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), "one_turn": one, "matrix": mat})


@stage("timing")
def _timing():
    from . import timing as T
    t0 = time.time()
    r = T.run()
    Cm.save_json("timing", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), **r})


@stage("matrix")
def _matrix():
    from . import matrix as MX
    t0 = time.time()
    out = []
    for base, start, gap in (("r2", "s0", "equal"), ("aggro", "s0", "equal"), ("r2", "neutral", "large"),
                             ("r2", "a_push", "equal")):
        res = MX.run(base, (), start, gap, strategies=MX.SET_FULL)
        res["bootstrap"] = MX.equilibrium_ci(res) if gap == "equal" and start == "s0" else None
        out.append(res)
    Cm.save_json("matrix", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), "games": out})


SENS_GRID = {"minion_damage_alpha": (30, 50, 80), "minion_retaliation_delay": (0, 1),
             "trade_wave_fraction": (0.25, 0.5, 1.0)}


@stage("sensitivity")
def _sensitivity():
    from . import sensitivity as SE
    t0 = time.time()
    rows = SE.factorial(SENS_GRID)
    Cm.save_json("sensitivity", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), "grid": SENS_GRID,
                                 "rows": rows, "main_effects": SE.main_effects(rows)})


CAL = (("farm_spot_rule", "last_front"), ("minion_retaliation_delay", 0))       # m45_calibrated() over m45_unified()


@stage("sensitivity_cal")
def _sensitivity_cal():
    """The factorial on the corrected mechanics (farm_spot_rule "last_front"), and one-at-a-time variations of the
    retained parameters around the candidate."""
    from . import sensitivity as SE
    t0 = time.time()
    grid = {"farm_spot_rule": ("last_front",), "minion_damage_alpha": (30, 50, 80), "minion_retaliation_delay": (0, 1)}
    rows = SE.factorial(grid)
    oat = []
    for field, v in (("minion_damage_growth", 1.05), ("minion_damage_growth", 1.12), ("vision_turns", 8),
                     ("ward_exposure", -1), ("trade_wave_fraction", 0.25), ("trade_wave_fraction", 1.0)):
        oat += SE.factorial({**{k: (v_,) for k, v_ in CAL}, field: (v,)})
    Cm.save_json("sensitivity_cal", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), "grid": grid,
                                     "rows": rows, "main_effects": SE.main_effects(rows), "one_at_a_time": oat})


@stage("candidate")
def _candidate():
    """The DEV battery of every Priority stage for the candidate (the baseline's DEV results are the other stages)."""
    from . import freeze as F, matrix as MX, timing as TI, trades as T, vision as V, zoning as Z
    t0 = time.time()
    c = Cm.cfg("cal")
    out = {"overrides": dict(CAL)}
    out["trades"] = {"table": T.bench_table(c), "criteria": T.criteria(T.bench_table(c)),
                     "multi_turn": T.multi_turn(CAL), "natural": {k: v for k, v in T.natural(CAL).items()
                                                                  if k != "rows"}}
    out["freeze"] = F.grid(CAL)
    out["vision"] = {"local": V.local(CAL), "strategies": V.strategies(CAL)}
    out["zoning"] = {"one_turn": Z.one_turn(CAL), "matrix": Z.matrix(CAL)}
    games = []
    for base_start, gap in (("s0", "equal"), ("neutral", "large"), ("a_push", "equal")):
        res = MX.run("r2", CAL, base_start, gap, strategies=MX.SET_FULL)
        res["bootstrap"] = MX.equilibrium_ci(res) if gap == "equal" and base_start == "s0" else None
        games.append(res)
    out["matrix"] = games
    out["timing"] = TI.run(CAL)
    out["meta"] = Cm.meta(elapsed_s=round(time.time() - t0, 1))
    Cm.save_json("candidate_dev", out)


@stage("trades_natural")
def _trades_natural():
    from . import trades as T
    t0 = time.time()
    out = {}
    for over in ((), (("minion_retaliation_delay", 0),), (("farm_spot_rule", "last_front"),), CAL):
        r = T.natural(over)
        out[str(dict(over))] = {k: v for k, v in r.items() if k != "rows"}
    Cm.save_json("trades_natural", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), "settings": out})


@stage("validate")
def _validate():
    """The holdout evaluation, once per configuration (baseline revision 2, then the candidate)."""
    from . import validate as VA
    t0 = time.time()
    out = {"meta": None, "configs": {}}
    for name, over in (("m45_unified (revision 2)", ()), ("m45_calibrated", CAL)):
        out["configs"][name] = VA.run(over)
    out["meta"] = Cm.meta(elapsed_s=round(time.time() - t0, 1), holdout_seeds=[Cm.HOLDOUT[0], Cm.HOLDOUT[-1]])
    Cm.save_json("validation", out)


@stage("demos")
def _demos():
    from . import demos as D
    t0 = time.time()
    r = D.run()
    Cm.save_json("demos", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), **r})


@stage("reference")
def _reference():
    from . import reference as RF
    if not RF.available():
        print("the M4.4b aggro finals are not on disk (results/rl/checkpoints/m44/ext): stage skipped")
        return
    t0 = time.time()
    rows = RF.run()
    Cm.save_json("reference", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), "rows": rows})


@stage("report")
def _report():
    from . import report as R
    R.write()


def main(argv) -> None:
    if not argv:
        print(__doc__)
        print("stages:", ", ".join(STAGES))
        return
    for name in argv:
        if name not in STAGES:
            raise SystemExit(f"unknown stage {name!r}; stages: {', '.join(STAGES)}")
    for name in argv:
        t0 = time.time()
        print(f"== stage {name}", flush=True)
        STAGES[name]()
        print(f"== stage {name} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
