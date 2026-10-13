"""Command line of the M4.3b pilot: `python -m lanegame.experiments.rl.m43b [stages] [options]`."""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

STAGES = ["benchmark", "train", "curves", "headline", "exploit", "behavior", "counterfactual", "recall",
          "trajectories", "figures", "report", "archive", "plans"]


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m43b")
    ap.add_argument("stages", nargs="*", default=["all"], help=f"{STAGES} or all")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--steps", type=int, default=None, help="training budget in agent transitions (default 2,048,000)")
    ap.add_argument("--conditions", default=None, help="comma-separated subset of the conditions")
    ap.add_argument("--seeds", default=None, help="comma-separated subset of the seeds")
    ap.add_argument("--attack-steps", type=int, default=None)
    a = ap.parse_args(argv)
    stages = [s for s in STAGES[1:] if s not in ("archive", "plans")] if "all" in a.stages else a.stages
    from . import common
    conds = a.conditions.split(",") if a.conditions else list(common.PILOT)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(common.SEEDS)
    t0 = time.perf_counter()
    for st in stages:
        print(f"===== {st} =====", flush=True)
        if st == "benchmark":
            from .train import benchmark
            benchmark()
        elif st == "train":
            from .train import train
            train(conds, seeds, a.steps or common.BUDGET, a.workers)
        elif st == "curves":
            from .curves import curves
            curves(conds, seeds, a.workers)
        elif st == "headline":
            from .evaluate import headline
            headline(conds, seeds, a.workers)
        elif st == "exploit":
            from .exploit import exploit
            exploit(conds, seeds, a.attack_steps or common.ATTACK_BUDGET, a.workers)
        elif st == "behavior":
            from .behavior import behavior
            behavior(conds, seeds, a.workers)
        elif st == "counterfactual":
            from .behavior import counterfactual
            counterfactual(conds, seeds, a.workers)
        elif st == "recall":
            from .recall import recall
            recall(conds, seeds, a.workers)
        elif st == "trajectories":
            from .trajectories import trajectories
            trajectories(conds, seeds)
        elif st == "figures":
            from .figures import figures
            figures()
        elif st == "report":
            from .report import report
            report()
        elif st == "archive":
            from .archive import archive_stage
            archive_stage()
        elif st == "plans":
            from .plans import plan_counterfactual
            plan_counterfactual(conds, seeds, min(a.workers, 4))
        else:
            raise SystemExit(f"unknown stage {st!r}; choose from {STAGES} or all")
        print(f"[{st}: {time.perf_counter() - t0:.0f} s elapsed]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
