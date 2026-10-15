"""CLI of M4.6a: `python -m lanegame.experiments.rl.m46a <stage> [--workers N] [--finals 0,1,...]`.

Stages (in the pre-registered order): prereg, smoke, e1, e3, e5, e2, e4_archives, e4_smoke, e4_train, e4_eval, e6,
report. Every experiment stage needs `prereg` first and is resumable from its per-job files.
"""

from __future__ import annotations

import argparse
import time


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m46a")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--finals", type=str, default=None)
    ap.add_argument("--targets", type=str, default=None)
    ap.add_argument("--kinds", type=str, default=None)
    args = ap.parse_args(argv)
    from . import common as C
    finals = tuple(int(x) for x in args.finals.split(",")) if args.finals else C.FINALS
    targets = tuple(int(x) for x in args.targets.split(",")) if args.targets else C.E4_TARGETS
    kinds = tuple(args.kinds.split(",")) if args.kinds else ("gap", "freeze")
    for st in args.stages:
        t0 = time.perf_counter()
        print(f"== M4.6a stage {st} ({args.workers} workers)", flush=True)
        if st == "prereg":
            C.write_prereg()
        elif st == "smoke":
            from .smoke import run
            run()
        else:
            C.require_prereg()
            if st == "e1":
                from .e1 import run
                run(args.workers)
            elif st == "e3":
                from .e3 import run
                run(args.workers, finals)
            elif st == "e3_positions":
                from .e3 import positions
                positions(args.workers, finals)
            elif st == "e3_natural_gaps":
                from .e3 import natural_gaps
                natural_gaps(args.workers, finals)
            elif st == "e5":
                from .e5 import run
                run(args.workers, finals)
            elif st == "e5_extend":
                from .e5 import extend
                extend(args.workers, finals=finals)
            elif st == "e2":
                from .e2 import run
                run(args.workers, finals)
            elif st == "e2_extend":
                from .e2 import extend
                extend(args.workers, finals)
            elif st == "e4_archives":
                from .e4 import build_archives
                build_archives(args.workers, targets)
            elif st == "e4_smoke":
                from .e4 import smoke
                print(smoke(targets[0]), flush=True)
            elif st == "e4_train":
                from .e4 import train
                train(args.workers, targets, kinds)
            elif st == "e4_eval":
                from .e4 import evaluate
                evaluate(args.workers, targets, kinds)
            elif st == "e4_transfer":
                from .e4 import transfer
                for k in kinds:
                    transfer(args.workers, targets, k)
            elif st == "e4_reanalyse":
                from .e4 import reanalyse
                reanalyse(targets)
            elif st == "e6":
                from .e6 import run
                run()
            elif st == "report":
                from .report import run
                run()
            else:
                raise SystemExit(f"unknown stage {st!r}")
        print(f"== stage {st} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
