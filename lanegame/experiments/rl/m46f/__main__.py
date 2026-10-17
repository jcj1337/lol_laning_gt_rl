"""CLI of M4.6f: `python -m lanegame.experiments.rl.m46f <stage> [<stage> ...] [--workers N] [--phase P] [--part X]`.

Stages:
    verify         initial-equivalence verification of both arms for every seed (results/m46f/verify.json)
    smoke          pipeline smoke test: verification for seed 0, both arms for 10 updates, a bit-identical re-run
    prereg         freeze docs/m4/m4_6/m4_6f_preregistration.md (after the smoke test, before any pilot run)
    train          the matched continuation runs of a phase (--phase main | ext; ext only after ext_rule fired)
    train_check    rebuild the training record of a phase from the run directories
    status         progress of running runs (read only)
    value          value evaluation: --part calib, natural, probes, responses (default all), --phase main | mid | ext
    behaviour      behavioural evaluation: --part h2h, curve, conv, freeze, recall, natural (default all),
                   --phase main | ext
    ext_rule       apply the pre-registered extension rule to the 4.096M results (results/m46f/extension_decision.json)
    analyse        every readout (results/m46f/analysis.json)
    report         generated tables (results/m46f/report.md)
"""

from __future__ import annotations

import argparse
import time


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m46f")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--phase", type=str, default="main")
    ap.add_argument("--part", type=str, default=None)
    args = ap.parse_args(argv)
    from . import common as C
    for st in args.stages:
        t0 = time.perf_counter()
        print(f"== M4.6f stage {st} ({args.workers} workers, phase {args.phase})", flush=True)
        if st == "verify":
            from .verify import run
            d = run()
            for k, v in d["checks"].items():
                print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
        elif st == "smoke":
            from .train import smoke
            d = smoke(args.workers)
            for k, v in d["checks"].items():
                print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
            print("throughput:", d["throughput"], flush=True)
        elif st == "prereg":
            C.write_prereg()
        elif st == "train":
            from .train import train
            d = train(workers=args.workers, phase=args.phase)
            print("all passed:", d["all_passed"], flush=True)
        elif st == "train_check":
            from .train import check
            d = check(phase=args.phase)
            print("all passed:", d["all_passed"], "failed:", [k for k, v in d["checks"].items() if not v], flush=True)
        elif st == "status":
            from .train import status
            status(args.phase)
        elif st == "value":
            C.require_prereg()
            from . import value
            value.run(args.workers, phase=args.phase, part=args.part)
        elif st == "behaviour":
            C.require_prereg()
            from . import behaviour
            behaviour.run(args.workers, phase=args.phase, part=args.part)
        elif st == "ext_rule":
            from . import analysis
            d = analysis.extension_rule()
            print("extension fires:", d["fire"], d["reasons"], flush=True)
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
