"""The Phase 1 siege study's experiments (docs/archive/phase1/phase1_siege_vision_study.md), written to artifacts/siege/.

    python -m experiments.siege all | benchmark timing controlled siege_options crash_value vision_sweep ablations
                                      sensitivity equilibrium verification figures

`figures` reads the JSON the other experiments write, so it runs last. Each experiment solves the exact game on the
default model `GameConfig()`; `benchmark` and `sensitivity` take the longest (minutes).
"""

from __future__ import annotations

import importlib
import sys
import time

STAGES = ["benchmark", "timing", "controlled", "siege_options", "crash_value", "vision_sweep", "ablations",
          "sensitivity", "equilibrium", "verification", "figures"]


def main(argv=None) -> None:
    names = list(sys.argv[1:] if argv is None else argv)
    if not names:
        raise SystemExit(__doc__)
    names = STAGES if "all" in names else names
    for n in names:
        if n not in STAGES:
            raise SystemExit(f"unknown experiment {n}; choose from {STAGES}")
        print(f"\n===== {n} =====")
        t = time.perf_counter()
        importlib.import_module(f"experiments.siege.{n}").run()
        print(f"[{n}: {time.perf_counter() - t:.1f}s]")


if __name__ == "__main__":
    main()
