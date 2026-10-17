"""CLI of OM1: `python -m lanegame.experiments.rl.om1 <stage> [<stage> ...] [--workers N]`.

    smoke       pipeline checks and throughput (results/om1/smoke.json)
    protocol    freeze docs/opponent_modelling/om1_protocol.md (after the smoke, before any selection game)
    A           stage A: the selection matrix (every menu responder against every library type)
    select      r*(type) and r_fix from stage A (results/om1/selection.json)
    B           stage B: pi, the oracle, r_fix and the switcher against every library type, per seed
    C           stage C: pi and the switcher against the out-of-library opponents, per seed
    analyse     every readout and the pre-registered classification (results/om1/analysis.json)
    report      generated tables (results/om1/report.md)

Evaluation only: there is no training stage and no --confirm flag. `--priority below_normal` lowers the OS priority.
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
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.om1")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--priority", choices=("normal", "below_normal"), default="normal")
    args = ap.parse_args(argv)
    if args.priority == "below_normal":
        _lower_priority()
    from . import common as C
    from . import stages as ST
    for st in args.stages:
        t0 = time.perf_counter()
        print(f"== OM1 stage {st} ({args.workers} workers)", flush=True)
        if st == "smoke":
            d = ST.smoke(args.workers)
            for k, v in d["checks"].items():
                print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
            print("throughput:", d["turns_per_s_one_worker"], flush=True)
        elif st == "protocol":
            C.freeze_protocol()
        elif st == "A":
            C.require_protocol()
            js = sorted(ST.sel_jobs(), key=lambda j: (j["r"] in C.SCRIPTED and j["opp"] in C.SCRIPTED))
            C.run_jobs(ST.sel_job, js, args.workers, "selection")
        elif st == "select":
            d = ST.select()
            print("r*:", d["rstar"], "\nr_fix:", d["rfix"], f"(candidate {d['rfix_candidate']})", flush=True)
        elif st in ("B", "C"):
            C.require_protocol()
            sel = C.load("selection")
            if sel is None:
                raise SystemExit("run stage select first")
            js = ST.eval_jobs(sel) if st == "B" else ST.ool_jobs(sel)
            js = sorted(js, key=lambda j: -len(j["responders"]))
            C.run_jobs(ST.eval_job, js, args.workers, f"stage {st}")
        elif st == "analyse":
            from .analysis import run
            run()
        elif st == "report":
            from .report import run
            run()
        else:
            raise SystemExit(f"unknown stage {st!r}")
        print(f"== stage {st} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
