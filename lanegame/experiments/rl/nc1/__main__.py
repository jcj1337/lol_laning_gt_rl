"""CLI of NC1: python -m lanegame.experiments.rl.nc1 <stage> [<stage> ...] [--workers N].

Stages (resumable; raw records under results/rl/checkpoints/nc1/):
    smoke       C, W and F on seeds 0 and 1 for 25 updates: C reproduces M5's sp history, W and F sample M5's first
                rollout; the measured throughput
    prereg      freeze docs/capacity/nc1_preregistration.md (SHA-256) before any fit or training
    train       the PPO pilot, W and F x 6 seeds to 16,384,000 transitions (needs --confirm-nc1-train)
    q1          stage 1 Q1: unit usage of the 18 stored networks
    q3          stage 1 Q3: M4.6e's value network refitted at other widths (with the exact-reproduction check)
    q2data      stage 1 Q2: the teachers' recorded decisions (N / P x train / test per target)
    q2fit       stage 1 Q2: the students (7 x 3 targets x 3 fit seeds)
    q2behaviour stage 1 Q2: fit seed 0 of each union student on E4's gap streams and from s0
    q2analyse   stage 1 Q2: lambda, kappa, retention, the reproduction check and the decision (results/nc1/q2.json)
    evaluate    stage 3: E1 head to head, E2 limitation families (--arms W runs E2 for W only), E3 (evaluation.json)
    q1pilot     Q1's unit usage on the 12 pilot finals and their controls
"""

from __future__ import annotations

import argparse
import time

GATES = {"train": "confirm_nc1_train"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.nc1")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arms", default=None, help="evaluate: run E2 for these arms only (e.g. W), no analysis")
    ap.add_argument("--confirm-nc1-train", action="store_true",
                    help="the user approved the pilot's compute (12 PPO runs to 16,384,000 transitions)")
    a = ap.parse_args(argv)
    from . import common as C
    for st in a.stages:
        gate = GATES.get(st)
        if gate and not getattr(a, gate):
            raise SystemExit(f"stage {st!r} needs --{gate.replace('_', '-')} (the user's approval of its compute)")
        t0 = time.perf_counter()
        print(f"== {st}", flush=True)
        if st == "smoke":
            from . import train as T
            T.smoke(a.workers)
        elif st == "prereg":
            C.write_prereg("frozen after the smoke stage and before any fit, behaviour rollout or PPO training")
        elif st == "train":
            from . import train as T
            T.train(a.workers)
        elif st == "q3":
            C.require_prereg()
            from . import q3
            q3.run(a.workers)
        elif st == "evaluate":
            C.require_prereg()
            from . import evaluate as EV
            EV.run(a.workers, a.arms.split(",") if a.arms else None)
        elif st == "q1pilot":
            from . import q1
            q1.run(a.workers, pilot=True)
        elif st == "q2analyse":
            from . import probe
            probe.analyse()
        elif st == "report":
            from . import report as RP
            print(RP.run())
        elif st == "q1":
            C.require_prereg()
            from . import q1
            q1.run(a.workers)
        elif st == "q2data":
            from . import probe
            probe.run_data(a.workers)
        elif st == "q2fit":
            C.require_prereg()
            from . import probe
            probe.run_fits(a.workers)
        elif st == "q2behaviour":
            C.require_prereg()
            from . import probe
            probe.run_behaviour(a.workers)
        else:
            raise SystemExit(f"unknown stage {st!r}")
        print(f"== {st} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
