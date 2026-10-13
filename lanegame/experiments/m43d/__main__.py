"""python -m lanegame.experiments.m43d <stage> [<stage> ...] [--workers N]

Stages of the M4.3d study (docs/m4/m4_3d_tower_pressure_calibration.md), in order:

    audit       Part 1: the risk and collection rules from the engine's own functions, hazard Monte Carlo
    tower_farm  Part 2: the crash grid on the own-tower rates 0.6 / 0.8 / 0.85 / 0.9 (econ otherwise)
    pressure    Part 3: the crash grid on econ with the attacker made immune to ganks / tower shots (paired)
    sensitivity Part 4: the crash grid on the risk-calibration variants
    strategic   Part 5: restricted matrix games of the M4.3c scenarios plus the crash scenarios
    validation  Part 6: the selected comparison on an independent seed set
    traces      representative lane traces
    report      results/m43d/*.md tables from the stored results

Nothing is trained. Results: results/m43d/.
"""

from __future__ import annotations

import argparse
import time

STAGES = ("audit", "tower_farm", "pressure", "sensitivity", "strategic", "validation", "traces", "report")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m lanegame.experiments.m43d")
    p.add_argument("stages", nargs="+", choices=STAGES + ("all",))
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--variants", nargs="+", default=None)
    a = p.parse_args(argv)
    stages = STAGES if "all" in a.stages else tuple(s for s in STAGES if s in a.stages)
    for s in stages:
        t0 = time.perf_counter()
        print(f"== {s}", flush=True)
        if s == "audit":
            from . import audit
            from .common import write_md
            res = audit.run(tuple(a.variants or ("econ",)))
            write_md("audit", audit.report(res, (a.variants or ("econ",))[0]))
        else:
            from . import stages as stg
            getattr(stg, s)(workers=a.workers, variants=a.variants)
        print(f"== {s} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
