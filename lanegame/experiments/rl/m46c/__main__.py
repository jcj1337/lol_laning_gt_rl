"""CLI of M4.6c: `python -m lanegame.experiments.rl.m46c <stage> [--workers N]`.

Stages:
    archive          build the shared training start archive (and check it under both arms)
    smoke            pipeline smoke test (both arms, 10 updates, plus a bit-identical re-run)
    prereg           freeze the pre-registration (before any pilot training run)
    train            the matched continuation runs (6 seeds x 2 arms, 4,096,000 transitions)
    train_check      rebuild the training record from the run directories
    status           progress of running runs (read only)
    eval_<family>    an evaluation family (evaluate.FAMILIES), after prereg
    analyse          the analysis of every finished family
    report           generated tables (results/m46c/report.md)
"""

from __future__ import annotations

import argparse
import time


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m46c")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--phase", type=str, default="main")
    args = ap.parse_args(argv)
    from . import common as C
    for st in args.stages:
        t0 = time.perf_counter()
        print(f"== M4.6c stage {st} ({args.workers} workers)", flush=True)
        if st == "archive":
            from .archive import build, check_states
            d = build(args.workers)
            d["check"] = check_states()
            C.save_json("archive", d)
            print(d["check"], d["by_label"], flush=True)
        elif st == "smoke":
            from .train import smoke
            d = smoke(args.workers)
            for k, v in d["checks"].items():
                print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
        elif st == "prereg":
            C.write_prereg()
        elif st == "train":
            from .train import train
            d = train(workers=args.workers, phase=args.phase)
            print("all passed:", d["all_passed"], flush=True)
        elif st == "train_check":
            from .train import check
            d = check(phase=args.phase)
            bad = [k for k, v in d["checks"].items() if not v]
            print("all passed:", d["all_passed"], "failed:", bad, flush=True)
        elif st == "status":
            from .train import status
            status(args.phase)
        elif st.startswith("eval_"):
            C.require_prereg()
            from . import evaluate
            evaluate.run_family(st[5:], args.workers)
        elif st == "traces":
            from .traces import run as traces_run
            traces_run()
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
