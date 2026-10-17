"""CLI of M4.8: `python -m lanegame.experiments.rl.m48 <stage> [<stage> ...] [--workers N] [--part X]`.

    pilot       throughput and variance pilot (24 natural states at turn 94, K 64); results/m48/pilot.json
    protocol    freeze docs/m4/m4_8/m4_8_protocol.md (after the pilot, before any main evaluation run)
    states      replay the M4.6d natural records to the checkpoints (verified) and play the validation games
    select      pick the Stage 1 validation states (checkpoint criteria only)
    val         Stage 1 continuations (selected cases, mirrors, controlled variants)
    nat         Stage 2 continuations of the natural checkpoints: --part primary | sens (default both)
    recheck     supplementary (amendment of 2026-10-17): M4.7's head-to-head lane lead P1 with E_h at turn 94
    analyse     every readout (results/m48/validation.json, application.json)
    report      generated tables (results/m48/report.md)

Evaluation only: there is no training stage and no --confirm flag. `--priority below_normal` lowers the OS priority of
this process and its workers; results do not depend on it.
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
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m48")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--part", type=str, default=None)
    ap.add_argument("--priority", choices=("normal", "below_normal"), default="normal")
    args = ap.parse_args(argv)
    if args.priority == "below_normal":
        _lower_priority()
    from . import common as C
    from . import families as F
    from . import rollouts as R
    from . import states as ST
    for st in args.stages:
        t0 = time.perf_counter()
        print(f"== M4.8 stage {st} ({args.workers} workers)", flush=True)
        if st == "pilot":
            from .analysis import pilot
            pilot(args.workers)
        elif st == "protocol":
            C.freeze_protocol()
        elif st == "states":
            C.run_jobs(ST.natural_job, [(env, s, tuple(range(48)), C.T0S) for env in C.ENVS for s in C.SEEDS],
                       args.workers, "natural states")
            C.run_jobs(ST.validation_job, [(env, s) for env in C.ENVS for s in C.SEEDS], args.workers,
                       "validation games")
        elif st == "select":
            sel = ST.select_validation()
            C.save_gz(f"{C.CKPT}/states/selected.json.gz", sel)
            from collections import Counter
            print(Counter((x["case"], x["X"]) for x in sel), flush=True)
        elif st == "val":
            C.require_protocol()
            sel = C.load_gz(f"{C.CKPT}/states/selected.json.gz")
            R.run(F.validation_jobs(sel), args.workers, "validation")
        elif st == "nat":
            C.require_protocol()
            states = ST.natural_states()
            if args.part in (None, "primary"):
                R.run(F.natural_jobs([x for x in states if x["t0"] == C.T0_PRIMARY]), args.workers, "natural t94")
            if args.part in (None, "sens"):
                R.run(F.natural_jobs([x for x in states if x["t0"] != C.T0_PRIMARY]), args.workers, "natural sens")
        elif st == "recheck":
            C.require_protocol()
            from .recheck import run
            run(args.workers)
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
