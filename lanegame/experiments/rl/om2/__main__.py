"""CLI of OM2: `python -m lanegame.experiments.rl.om2 <stage> [<stage> ...] [--workers N]`.

    smoke       readiness: both arms of seed 0 for SMOKE_UPDATES updates and a bit-identical re-run (results/om2/smoke.json)
    prereg      freeze docs/opponent_modelling/om2_preregistration.md (after the smoke, before any pilot run)
    train       the pilot runs (PILOT_SEEDS x 2 arms, BUDGET each); needs --confirm-om2-train (the user's approval)
    eval        the evaluation families against every final (pre-registration §3)
    analyse     contrasts and the pre-registered classification (results/om2/analysis_main.json)
    explore     exploratory reference, NOT pre-registered: the common start pi_s on the same games, and each
                arm against pi_s (results/om2/explore.json)
    report      generated tables (results/om2/report.md)

`--priority below_normal` lowers the OS priority of this process and its workers.
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
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.om2")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--priority", choices=("normal", "below_normal"), default="normal")
    ap.add_argument("--confirm-om2-train", action="store_true",
                    help="required by train: the user approved the pilot's compute budget")
    args = ap.parse_args(argv)
    if args.priority == "below_normal":
        _lower_priority()
    from . import common as C
    for st in args.stages:
        t0 = time.perf_counter()
        print(f"== OM2 stage {st} ({args.workers} workers)", flush=True)
        if st == "smoke":
            from .train import smoke
            d = smoke(min(3, args.workers))
            for k, v in d["checks"].items():
                print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
            print("throughput:", d.get("throughput_transitions_per_s"), flush=True)
        elif st == "prereg":
            C.write_prereg()
        elif st == "train":
            if not args.confirm_om2_train:
                raise SystemExit("train runs the OM2 pilot budget; it needs --confirm-om2-train (the user's approval)")
            from .train import train
            d = train(args.workers)
            print("all complete:", d["all_complete"], flush=True)
        elif st == "eval":
            C.require_prereg()
            from .evaluate import run
            print(run(args.workers), flush=True)
        elif st == "analyse":
            from .analysis import run
            run()
        elif st == "explore":
            from .explore import run
            run(args.workers)
        elif st == "report":
            from .report import run
            run()
        else:
            raise SystemExit(f"unknown stage {st!r}")
        print(f"== stage {st} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
