"""python -m lanegame.experiments.m43c <stage> [<stage> ...] [--workers N]

Stages (docs/m4/m4_3c_economic_objective_design.md §9): examples, progression, xp_cf, freeze, death, report;
xp_range (docs/m4/m4_3c_xp_range_screen.md; also writes its own tables); wave_control
(docs/m4/m4_3c_wave_control_screen.md; also writes its own tables); four_actions (docs/m4/m4_3c_four_action_wave_redesign.md;
also writes its own tables; --no-ppo skips its PPO smoke run). M4.3c stabilization
(docs/m4/m4_3c_strategic_economics_calibration.md): sequences (Stage A3). `all` runs every stage in that order. Nothing
is trained; the M4.3b checkpoints must be in place (results/rl/checkpoints/m43b/, or the bundle of
results/rl/m43b/ARTIFACTS.md) for progression, xp_cf, freeze, death.
"""

from __future__ import annotations

import argparse
import time

STAGES = ("examples", "progression", "xp_cf", "freeze", "death", "report", "xp_range", "wave_control",
          "four_actions", "sequences", "economics", "econ_probes", "calibration_report", "pilot")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m lanegame.experiments.m43c")
    p.add_argument("stages", nargs="+", choices=STAGES + ("all",))
    p.add_argument("--workers", type=int, default=9)
    p.add_argument("--no-ppo", action="store_true", help="four_actions: skip the PPO smoke run")
    p.add_argument("--variants", nargs="+", default=None,
                   help="economics / econ_probes / pilot: the economics.VARIANTS to run (default: every variant)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1], help="pilot: training seeds")
    p.add_argument("--steps", type=int, default=2_048_000, help="pilot: agent transitions per training run")
    a = p.parse_args(argv)
    stages = STAGES if "all" in a.stages else tuple(s for s in STAGES if s in a.stages)
    for s in stages:
        t0 = time.perf_counter()
        print(f"== {s}", flush=True)
        if s == "examples":
            from .examples import run
            run()
        elif s == "progression":
            from .progression import run
            run(workers=a.workers)
        elif s == "xp_cf":
            from .xp_cf import run
            run(workers=a.workers)
        elif s == "freeze":
            from .freeze import run
            run(workers=a.workers)
        elif s == "death":
            from .death import run
            run(workers=a.workers)
        elif s == "report":
            from .report import run
            run()
        elif s == "xp_range":
            from .xp_range import report, run
            report(run(workers=a.workers))
        elif s == "wave_control":
            from .wave_control import report, run
            report(run(workers=a.workers))
        elif s == "four_actions":
            from .four_actions import report, run
            report(run(ppo=not a.no_ppo))
        elif s == "sequences":
            from .sequences import report, run
            report(run(workers=a.workers))
        elif s == "economics":
            from .economics import VARIANTS, run
            run(a.variants or list(VARIANTS), workers=a.workers)
        elif s == "econ_probes":
            from .economics_probes import run
            run(a.variants or ["base"], workers=a.workers)
        elif s == "calibration_report":
            from .calibration_report import run
            run()
        elif s == "pilot":
            from .pilot import run
            run(a.variants or ["base", "candidate"], seeds=a.seeds, steps=a.steps, workers=a.workers)
        print(f"== {s} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
