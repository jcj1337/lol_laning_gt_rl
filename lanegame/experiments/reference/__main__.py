"""Reference generator stages (docs/m4/laning_game_mathematical_reference.md). Results go to
results/reference/<stage>.json; `report` writes results/reference/report.md from them.

    python -m lanegame.experiments.reference <stage> [<stage> ...] [--workers N]

    examples   deterministic tables and worked examples (seconds)
    actions    legality, the observation and the reach of the wave choice in random legal play (about 1 minute)
    selfplay   32 games of `sp:0` against itself, 1,000 turns each (about 2 minutes on one worker)
    death      paired rollouts for the cost of a death and of a Recall (about 20 worker-minutes)
    report     results/reference/report.md
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


def _need_policy() -> bool:
    if Cm.policy_available():
        return True
    print(f"the M5 checkpoint of {Cm.POLICY} is not on disk (results/rl/checkpoints/m5, git-ignored): stage skipped")
    return False


@stage("examples")
def _examples(workers: int):
    from . import examples as E
    t0 = time.time()
    r = E.run()
    Cm.save_json("examples", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), **r})


@stage("actions")
def _actions(workers: int):
    from . import actions as A
    t0 = time.time()
    r = A.run(workers)
    Cm.save_json("actions", {"meta": Cm.meta(elapsed_s=round(time.time() - t0, 1)), **r})


@stage("selfplay")
def _selfplay(workers: int):
    from . import play as P
    if not _need_policy():
        return
    t0 = time.time()
    r = P.selfplay(workers)
    Cm.save_json("selfplay", {"meta": Cm.meta(policy=Cm.POLICY, elapsed_s=round(time.time() - t0, 1)), **r})


@stage("death")
def _death(workers: int):
    from . import play as P
    if not _need_policy():
        return
    t0 = time.time()
    r = P.death(workers)
    Cm.save_json("death", {"meta": Cm.meta(policy=Cm.POLICY, elapsed_s=round(time.time() - t0, 1)), **r})


@stage("report")
def _report(workers: int):
    from . import report as R
    R.write()


def main(argv) -> None:
    from ..m45b.common import WORKERS
    workers = WORKERS
    if "--workers" in argv:
        i = argv.index("--workers")
        workers = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if not argv:
        print(__doc__)
        return
    for name in argv:
        if name not in STAGES:
            raise SystemExit(f"unknown stage {name!r}; stages: {', '.join(STAGES)}")
    for name in argv:
        t0 = time.time()
        print(f"== stage {name}", flush=True)
        STAGES[name](workers)
        print(f"== stage {name} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
