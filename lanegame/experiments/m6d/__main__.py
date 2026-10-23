"""M6d stages (docs/m6/m6_plan.md, M6d). Results go to results/m6/.

    python -m lanegame.experiments.m6d <stage> [<stage> ...]

    overview  every result number in docs/m6/research_overview.md checked against the M5 records
              (results/m5/analysis_main.json, strategic_main.json, monitoring_main.json and
              battery/summary_battery_{om,sp}.json.gz). Forward: each claim's text, produced from the records with the
              overview's rounding, is printed in its section. Reverse: every signed decimal in the overview, and every
              unsigned decimal in §5 and §6, lies inside a checked claim or constant. Verdicts: the classes the
              wording relies on (P1 demonstrated, P2a/P2b and S2-S4 material loss, S1 preserved, S6 inconclusive, the
              five behaviours per arm) match the records. Constants from the mathematical reference and the
              pre-registration are checked against a quote in their cited section, and against a record where one
              holds them. Writes results/m6/m6d_overview_check.json and exits non-zero on any failure (about a second)

The stage reads tracked files only. It trains nothing and changes no record.
"""

from __future__ import annotations

import sys
import time

STAGES = {}


def stage(name):
    def deco(fn):
        STAGES[name] = fn
        return fn
    return deco


@stage("overview")
def _overview():
    from . import overview_check as O
    if not O.run():
        raise SystemExit("the overview check failed (see results/m6/m6d_overview_check.json)")


def main(argv) -> None:
    if not argv:
        print(__doc__)
        return
    for name in argv:
        if name not in STAGES:
            raise SystemExit(f"unknown stage {name!r}; stages: {', '.join(STAGES)}")
    for name in argv:
        t0 = time.time()
        print(f"== stage {name}", flush=True)
        STAGES[name]()
        print(f"== stage {name} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
