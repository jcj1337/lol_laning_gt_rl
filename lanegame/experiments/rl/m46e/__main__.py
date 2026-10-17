"""CLI of M4.6e: python -m lanegame.experiments.rl.m46e <stage> [<stage> ...] [--workers N] [--part P].

Stages (each resumable; per-job records cached under results/rl/checkpoints/m46e/):
    observation   Q1: controlled single-feature pairs, mirror and decode checks, entry ranges, scale and
                  checkpoint compatibility, the shared trunk's value / policy gradient share
    exposure      Q2: archive start composition (direct) and reconstructed training occupancy from snapshots
    labels        Q3: natural and constructed policy-specific return labels; --part natural,constructed,tail
    prereg        freeze docs/m4/m4_6/m4_6e_preregistration.md (SHA-256) before any fit
    fit           Q4: the supervised value study (arms A-D, three splits)
    ranking       look-ahead rankings of the fitted predictors at the M4.6d decision states
    analyse       the fit readouts and the horizon check (results/m46e/analysis.json)
    report        the generated tables (results/m46e/report.md)
"""

from __future__ import annotations

import argparse
import time


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m46e")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--part", default=None)
    a = ap.parse_args(argv)
    from . import common as C
    for st in a.stages:
        t0 = time.perf_counter()
        print(f"== {st}", flush=True)
        if st == "observation":
            from . import observation as O
            O.run(a.workers)
        elif st == "exposure":
            from . import exposure as X
            X.run(a.workers, part=a.part)
        elif st == "labels":
            from . import labels as LB
            LB.run(a.workers, part=a.part)
        elif st == "prereg":
            C.write_prereg()
        elif st == "fit":
            C.require_prereg()
            from . import fit as F
            F.run(a.workers, part=a.part)
        elif st == "ranking":
            C.require_prereg()
            from . import ranking as RK
            RK.run(a.workers)
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
