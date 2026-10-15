"""Command line of M4.5c: `python -m lanegame.experiments.rl.m45c [stages] [options]`.

Pilot stages (authorized; results/m45c/, checkpoints in results/rl/checkpoints/m45c/, git-ignored):
    audit          the frozen preset, action space, masks, Ward legality, observation / mirror checks, reference-suite
                   legality, the checkpoint guard and the OOD adapter (seconds)
    smoke          a 10-update correctness run and its bit-identical re-run (pipeline checks)
    smoke_eval     every evaluation instrument on the smoke final at tiny sizes
    pilot          THE pilot: seed 100, the 4,096,000-transition learning-rate schedule stopped at 1,024,000 agent
                   transitions (one process); refuses to overwrite an existing pilot
    prefix_check   re-train the pilot's first 7 updates (its first snapshot) in a fresh directory; weights and history must equal the pilot's
    eval_pilot     learning curve, headline (W = 1000) and history of the pilot checkpoints
    behavior_pilot state-conditioned action tables, freezes, recalls and the M4.5b ledger of the pilot checkpoints
    bench          throughput: the per-update breakdown (initial and final network), multi-process scaling and the
                   lane-cache memory / speed probe
    report         results/m45c/report.md, figures and the full-campaign plan (plan.py)

Full campaign (NOT authorized by the pilot brief): `train_full` refuses to start without --confirm-m45c-full, which is
reserved for after the review of the measured pilot and the recommended budget.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

STAGES = ["audit", "smoke", "smoke_eval", "pilot", "prefix_check", "eval_pilot", "behavior_pilot", "bench", "report",
          "train_full"]


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m45c")
    ap.add_argument("stages", nargs="+", help=str(STAGES))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--parts", default="curve,headline,history", help="eval_pilot parts")
    ap.add_argument("--confirm-m45c-full", action="store_true", help="required by train_full (after budget approval)")
    a = ap.parse_args(argv)
    from .common import WORKERS
    workers = a.workers or WORKERS
    t0 = time.perf_counter()
    for st in a.stages:
        print(f"===== {st} =====", flush=True)
        if st == "audit":
            from .stages import audit
            audit()
        elif st == "smoke":
            from .train import smoke
            smoke()
        elif st == "smoke_eval":
            from .stages import smoke_eval
            smoke_eval(workers)
        elif st == "pilot":
            from .train import pilot
            pilot()
        elif st == "prefix_check":
            from .train import prefix_check
            prefix_check()
        elif st == "eval_pilot":
            from .stages import eval_pilot
            eval_pilot(workers, a.parts.split(","))
        elif st == "behavior_pilot":
            from .stages import behavior_pilot
            behavior_pilot(workers)
        elif st == "bench":
            from .common import PILOT_SEED
            from .train import cache_probe, profile, scaling
            profile(f"initial:{PILOT_SEED}")
            profile(f"final:pilot:{PILOT_SEED}")
            scaling()
            cache_probe()
        elif st == "report":
            from .report import report
            report()
        elif st == "train_full":
            if not a.confirm_m45c_full:
                raise SystemExit("train_full is the full M4.5c multi-seed campaign; it is NOT part of the pilot. It "
                                 "needs the explicit approval of the budget in results/m45c/full_plan.md and then "
                                 "--confirm-m45c-full.")
            raise SystemExit("train_full is specified in results/m45c/full_plan.md and will be implemented with the "
                             "approved budget; nothing was started.")
        else:
            raise SystemExit(f"unknown stage {st!r}; choose from {STAGES}")
        print(f"[{st}: {time.perf_counter() - t0:.0f} s elapsed]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
