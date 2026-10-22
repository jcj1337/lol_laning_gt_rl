"""M6a stages (docs/m6/m6a_technical_report.md). Results go to results/m6/<stage>.json.

    python -m lanegame.experiments.m6a <stage> [<stage> ...]

    verify    re-simulate the three seed-0 M5 recordings through the session core, and replay the 12 attacker
              recordings from their actions (about 10 s)
    measure   turn cost with frames, history replay, recording size and bot-versus-bot generation (about 1 minute)

The `export` stage of M6a wrote the throwaway prototypes' data files; it went with the prototypes in M6b
(docs/m6/m6b_report.md), and the app serves the same recordings through its API (lanegame/demo/service.py).

Every stage needs the 7 checkpoint files of lanegame/demo/manifest.json (git-ignored) and stops with a message when
they are missing. None trains, and none changes a recording or a result of an earlier milestone.
"""

from __future__ import annotations

import sys
import time

from . import common as Cm

STAGES = {}


def stage(name):
    def deco(fn):
        STAGES[name] = fn
        return fn
    return deco


@stage("verify")
def _verify():
    from . import measure as M
    t0 = time.time()
    r = M.verify()
    Cm.save_json("m6a_verify", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), **r})


@stage("measure")
def _measure():
    from . import measure as M
    t0 = time.time()
    r = M.measure()
    Cm.save_json("m6a_measurements", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), **r})


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
