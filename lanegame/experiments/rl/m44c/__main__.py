"""Command line of M4.4c: `python -m lanegame.experiments.rl.m44c [stages] [options]` (results in results/m44c/).

Stages (each writes results/m44c/<stage>.json(.gz); none trains a network or changes the game):
    recall        Part 2: interrupted Recalls, 4.10M finals (Recall / Safe / Hold / Safe>Recall / Retreat)
    recall_rep    the same on the 2.05M finals of econ and aggro (contested and repeated states, protocol "policy")
    freeze        Part 3: multi-turn freeze plans against five opponent protocols / styles, natural states
    freeze_init   Part 3: the same plans from INITIALIZED freeze states (power gap x level x opponent HP grid)
    freeze_rep    Part 3 on the 2.05M finals of econ and aggro (protocol "policy")
    zoning        Part 4: stronger-side strategies x weaker-side responses from INITIALIZED power-gap states
    zoning_nat    Part 4: the same from NATURAL states with a large HP-aware duel edge
    zoning_mech   Part 4: behaviour matrix and closed-form trade economics (mechanical diagnosis)
    vision        Part 5: Vision-then-Pressure vs immediate Pressure vs conditional, natural siege states
    waves_mech    Part 6: wave-action budgets and one-turn transitions at natural states (identical / different)
    waves         Part 6: multi-turn wave sequences against wave-forced opponents, natural states
    traces        counterfactual pairs from one state, exported for the viewer and verified by replay
    report        results/m44c/report.md and results/m44c/figures/
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

STAGES = ["recall", "recall_rep", "freeze", "freeze_init", "freeze_rep", "zoning", "zoning_nat", "zoning_mech",
          "vision", "waves_mech", "waves", "traces", "report"]


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m44c")
    ap.add_argument("stages", nargs="+", help=str(STAGES))
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--envs", default=None, help="comma-separated subset of four,econ,aggro")
    ap.add_argument("--seeds", default=None, help="comma-separated subset of the training seeds")
    a = ap.parse_args(argv)
    envs = a.envs.split(",") if a.envs else None
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    t0 = time.perf_counter()
    for st in a.stages:
        print(f"===== {st} =====", flush=True)
        if st == "recall":
            from .recall import run
            run(envs, seeds, workers=a.workers)
        elif st == "recall_rep":
            from .recall import run
            run(envs or ["econ", "aggro"], seeds, phases=("main",), workers=a.workers,
                sizes={"contested": 32, "repeated": 24}, name="recall_rep",
                protocols={"contested": ("policy",), "repeated": ("policy",)})
        elif st == "freeze":
            from .freeze import run
            run(envs, seeds, workers=a.workers)
        elif st == "freeze_init":
            from .freeze import run_init
            run_init(envs, seeds, workers=a.workers)
        elif st == "freeze_rep":
            from .freeze import run
            run(envs or ["econ", "aggro"], seeds, phases=("main",), workers=a.workers, name="freeze_rep",
                protocols=("policy",))
        elif st == "zoning":
            from .zoning import run_init
            run_init(envs, seeds, workers=a.workers)
        elif st == "zoning_nat":
            from .zoning import run_natural
            run_natural(envs, seeds, workers=a.workers)
        elif st == "zoning_mech":
            from .zoning import run_mechanics
            run_mechanics(envs, seeds, workers=a.workers)
        elif st == "vision":
            from .vision import run
            run(envs, seeds, workers=a.workers)
        elif st == "waves_mech":
            from .waves import run_mechanics
            run_mechanics(envs, seeds, workers=a.workers)
        elif st == "waves":
            from .waves import run
            run(envs, seeds, workers=a.workers)
        elif st == "traces":
            from .traces import run
            run(workers=a.workers)
        elif st == "report":
            from .report import report
            report()
        else:
            raise SystemExit(f"unknown stage {st!r}; choose from {STAGES}")
        print(f"[{st}: {time.perf_counter() - t0:.0f} s elapsed]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
