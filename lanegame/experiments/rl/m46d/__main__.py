"""CLI of M4.6d: python -m lanegame.experiments.rl.m46d <stage> [<stage> ...] [--workers N].

Stages (in order; each resumable, per-job records cached under results/rl/checkpoints/m46d/):
    records       full-state natural self-play of the 8.192M general policies (C1 and baseline), 48 x 400 turns each
    freezes       descriptive: freeze episodes classified and disaggregated; natural early advantages (no rollouts)
    prereg        freeze docs/m4/m4_6/m4_6d_preregistration.md (SHA-256)
    matrix        workstreams A + B: freezer plans x breaker plans from constructed and natural freeze states
    confirm       the independent confirmation sample of the matrix's selected contrasts
    specialists   workstream C: the gap, its decomposition and matched-state substitutions
    critic        the long-return critic diagnostic
    traces        explanatory trajectories
    analyse       all summaries; report  the generated tables (results/m46d/report.md)
"""

from __future__ import annotations

import argparse
import time


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m46d")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--part", default=None)
    a = ap.parse_args(argv)
    from . import common as C
    for st in a.stages:
        t0 = time.perf_counter()
        print(f"== {st}", flush=True)
        if st == "records":
            from . import records as R
            C.run_jobs(R.record_job, R.record_jobs(), a.workers, "M4.6d records")
        elif st == "freezes":
            from . import freezes as F
            F.run()
        elif st == "prereg":
            C.write_prereg()
        elif st == "matrix":
            C.require_prereg()
            from . import matrix as M
            M.run(a.workers, part=a.part)
        elif st == "confirm":
            C.require_prereg()
            from . import matrix as M
            M.run_confirm(a.workers)
        elif st == "specialists":
            C.require_prereg()
            from . import specialists as SP
            SP.run(a.workers, part=a.part)
        elif st == "critic":
            C.require_prereg()
            from . import critic as CR
            CR.run(a.workers, part=a.part)
        elif st == "traces":
            from . import traces as TR
            TR.run()
        elif st == "analyse":
            from . import analysis as AN
            AN.run()
        elif st == "report":
            from . import report as RP
            RP.run()
        else:
            raise SystemExit(f"unknown stage {st!r}")
        print(f"== {st} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
