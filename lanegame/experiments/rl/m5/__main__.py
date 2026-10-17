"""CLI of M5: `python -m lanegame.experiments.rl.m5 <stage> [<stage> ...] [--workers N]`.

M5a readiness (bounded compute; docs/m5/m5a_readiness_protocol.md):
    ready_protocol  record the readiness protocol's SHA-256 (before the first check)
    R0 ... R10      one readiness check each (R4 = the smoke and its re-run, R9 = the concurrency benchmark, R10 = the
                    instruments at reduced samples); `ready` runs R0-R10 in order
    budget          the campaign budget from the measured rates (results/m5/budget.json)
    manifest        the run manifest the training stage would write, without training (results/m5/manifest_preview.json)
    prereg          freeze docs/m5/m5_preregistration.md (SHA-256 in results/m5/preregistration.json)

The campaign (each stage needs the user's approval of its compute budget, given by its flag):
    train           M5b training, 12 runs x BUDGET               --confirm-m5-train  [--restart-incomplete]
    train_check     the pre-registered health checks of the training runs (no compute)
    status          progress of the training runs (no compute)
    monitor         M5b learning curves and history comparisons  --confirm-m5-eval
    battery         M5c strategic battery on every final          --confirm-m5-eval
    cells           M5c M4 limitation families                     --confirm-m5-eval
    families        M5d opponent families and head to head         --confirm-m5-eval
    attack_tables   the om finals' self-play belief tables (adaptation test)   --confirm-m5-attackers
    attackers       M5d attacker training                           --confirm-m5-attackers
    attack_eval     the attackers' independent evaluation          --confirm-m5-attackers
    analyse         the pre-registered analysis (no compute beyond reading records)
    report          generated tables (results/m5/report.md)

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


GATES = {"train": "confirm_m5_train", "monitor": "confirm_m5_eval", "battery": "confirm_m5_eval",
         "cells": "confirm_m5_eval", "families": "confirm_m5_eval", "attack_tables": "confirm_m5_attackers",
         "attackers": "confirm_m5_attackers", "attack_eval": "confirm_m5_attackers"}


def _print_checks(d: dict) -> None:
    for k, v in (d.get("checks") or {}).items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m5")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--priority", choices=("normal", "below_normal"), default="normal")
    ap.add_argument("--restart-incomplete", action="store_true",
                    help="train: preserve an incomplete run directory under partial/ and restart that run")
    ap.add_argument("--confirm-m5-train", action="store_true", help="the user approved the M5b training budget")
    ap.add_argument("--confirm-m5-eval", action="store_true", help="the user approved the M5b-M5d evaluation budget")
    ap.add_argument("--confirm-m5-attackers", action="store_true", help="the user approved the attacker budget")
    args = ap.parse_args(argv)
    if args.priority == "below_normal":
        _lower_priority()
    from . import common as C
    from . import readiness as R
    stages = []
    for st in args.stages:
        stages += list(R.PARTS) if st == "ready" else [st]
    for st in stages:
        gate = GATES.get(st)
        if gate and not getattr(args, gate):
            raise SystemExit(f"stage {st} runs campaign compute; it needs --{gate.replace('_', '-')} (the user's "
                             "approval of its budget). Nothing was started.")
        t0 = time.perf_counter()
        print(f"== M5 stage {st} ({args.workers} workers)", flush=True)
        if st == "ready_protocol":
            print(R.freeze_protocol()["record"], flush=True)
        elif st in R.PARTS:
            R.freeze_protocol()
            from . import bench, instruments
            fn = {"R0": R.r0_artifacts, "R1": R.r1_seeds, "R2": R.r2_init, "R3": R.r3_defaults,
                  "R4": lambda: R.r4_smoke(min(4, args.workers)), "R5": R.r5_crn, "R6": R.r6_inputs,
                  "R7": R.r7_shares, "R8": R.r8_equivalence, "R9": lambda: bench.r9_bench(),
                  "R10": lambda: instruments.r10_instruments(args.workers)}[st]
            d = fn()
            _print_checks(d)
            print(f"  {st}: {'passed' if d.get('passed') else 'NOT passed'}", flush=True)
        elif st == "budget":
            from .bench import budget
            d = budget()
            print({k: v for k, v in d.get("totals", {}).items()}, flush=True)
        elif st == "manifest":
            from .train import run_manifest
            C.save_json("manifest_preview", run_manifest("main", C.ARMS, C.SEEDS, C.BUDGET))
        elif st == "prereg":
            print(C.write_prereg()["record"], flush=True)
        elif st == "train":
            from .train import train
            d = train(args.workers, restart_incomplete=args.restart_incomplete)
            print("all passed:", d["all_passed"], flush=True)
        elif st == "train_check":
            from .train import check
            d = check("main")
            _print_checks(d)
        elif st == "status":
            from .train import status
            status("main")
        elif st == "report":
            from .report import report
            report()
        elif st in ("monitor", "battery", "cells", "families", "analyse"):
            C.require_prereg()
            from . import evaluate as EV
            getattr(EV, f"stage_{st}")(args.workers)
        elif st in ("attack_tables", "attackers", "attack_eval"):
            C.require_prereg()
            from . import evaluate as EV
            getattr(EV, f"stage_{st}")(args.workers)
        else:
            raise SystemExit(f"unknown stage {st!r}; see the module docstring")
        print(f"== {st} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
