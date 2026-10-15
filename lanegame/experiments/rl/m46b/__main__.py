"""CLI of M4.6b: `python -m lanegame.experiments.rl.m46b <stage> [--workers N]`.

Stages: verify_v1, verify_v2, verify_v2_analysis (measurement fixes); audit (current curve); prereg (freeze the protocol); screen_*
(candidate screens, after prereg); report.
"""

from __future__ import annotations

import argparse
import time


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m46b")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arms", type=str, default=None)
    args = ap.parse_args(argv)
    from . import common as C
    for st in args.stages:
        t0 = time.perf_counter()
        print(f"== M4.6b stage {st} ({args.workers} workers)", flush=True)
        if st == "verify_v1":
            from .verify import v1
            v1(args.workers)
        elif st == "verify_v2":
            from .verify import v2
            v2(args.workers)
        elif st == "verify_v2_analysis":
            from .verify import v2_analysis
            v2_analysis()
        elif st == "audit":
            from .audit import run
            run(args.workers)
        elif st == "prereg":
            C.write_prereg()
        elif st.startswith("screen"):
            C.require_prereg()
            from . import screen
            arms = tuple(args.arms.split(",")) if args.arms else None
            getattr(screen, st)(args.workers, arms)
        elif st == "report":
            from .report import run
            run()
        else:
            raise SystemExit(f"unknown stage {st!r}")
        print(f"== stage {st} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
