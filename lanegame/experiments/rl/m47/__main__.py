"""CLI of M4.7: `python -m lanegame.experiments.rl.m47 <stage> [<stage> ...] [--workers N] [--part X]`.

Stage 1 (what happens before level 14; existing artifacts plus new paired rollouts):
    timeline        descriptives of the matched M4.6c policies from existing records (results/m47/timeline.json)
    occupancy       training occupancy (M4.6e exposure games) and objective weight by level (occupancy.json)
    returns         existing trajectories' discounted returns at the candidate horizons (returns.json)
    delayed         new paired 300-turn rollouts with reward streams: --part cells | natural | freeze (default all)
    delayed_analyse the delayed-reward readings (delayed.json)
Stage 2 (the matched horizon pilot; only after Stage 1 justified it):
    smoke           pipeline and objective checks: both arms of seed 0 for 10 updates, a bit-identical re-run
    prereg          freeze docs/m4/m4_7/m4_7_preregistration.md (after the smoke test, before any pilot run)
    train           the matched continuations (PILOT_SEEDS x 2 arms, BUDGET each); needs --confirm-m47-train
    train_check     rebuild the training record from the run directories
    status          progress of running runs (read only)
    eval            the pilot evaluation: --part h2h | fixed | natural | cells (default all)
    analyse         every Stage 2 readout and the pre-registered classification (analysis.json)
    report          generated tables (results/m47/report.md)

`--priority below_normal` lowers the OS priority of this process and its workers (inherited), so that the study
yields the CPU to interactive applications; results do not depend on it.
"""

from __future__ import annotations

import argparse
import time


def _lower_priority() -> None:
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS") else 10)
    except Exception as ex:                                # pragma: no cover
        print(f"could not lower the priority: {ex!r}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m47")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--part", type=str, default=None)
    ap.add_argument("--priority", choices=("normal", "below_normal"), default="normal")
    ap.add_argument("--confirm-m47-train", action="store_true",
                    help="required by train: the user approved the pilot's compute budget")
    args = ap.parse_args(argv)
    if args.priority == "below_normal":
        _lower_priority()
    from . import common as C
    for st in args.stages:
        t0 = time.perf_counter()
        print(f"== M4.7 stage {st} ({args.workers} workers)", flush=True)
        if st == "timeline":
            from .timeline import run
            run()
        elif st == "occupancy":
            from .occupancy import run
            run()
        elif st == "returns":
            from .returns import run
            run()
        elif st == "delayed":
            from .delayed import run
            run(args.workers, args.part)
        elif st == "delayed_analyse":
            from .delayed import analyse
            analyse()
        elif st == "smoke":
            from .train import smoke
            d = smoke(args.workers)
            for k, v in d["checks"].items():
                print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
            print("throughput:", d["throughput"], flush=True)
        elif st == "prereg":
            C.write_prereg()
        elif st == "train":
            if not args.confirm_m47_train:
                raise SystemExit("train runs the M4.7 pilot budget; it needs --confirm-m47-train (the user's approval)")
            from .train import train
            d = train(workers=args.workers)
            print("all passed:", d["all_passed"], flush=True)
        elif st == "train_check":
            from .train import check
            d = check()
            print("all passed:", d["all_passed"], "failed:", [k for k, v in d["checks"].items() if not v], flush=True)
        elif st == "status":
            from .train import status
            status()
        elif st == "eval":
            C.require_prereg()
            from . import evaluate
            evaluate.run(args.workers, args.part)
        elif st == "analyse":
            from . import analysis
            analysis.run()
        elif st == "report":
            from .report import run
            run()
        else:
            raise SystemExit(f"unknown stage {st!r}")
        print(f"== stage {st} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
