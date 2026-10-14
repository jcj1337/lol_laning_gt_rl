"""Command line of M4.4: `python -m lanegame.experiments.rl.m44 [stages] [options]`.

M4.4a stages (infrastructure; results/m44a/):
    audit          action space, compatibility of every known checkpoint with the three environments, observation and
                   mirror checks on random play (seconds)
    smoke_train    the M4.4b matrix (3 environments x 3 seeds) at SMOKE_BUDGET agent transitions, one process per run
    smoke_repro    re-train one run per environment and compare weights digests and histories
    smoke_eval     the evaluation pipeline on the smoke finals at small sizes (load / digest, references, cross-play,
                   history, behaviour diagnostics, counterfactuals, viewer traces with replay verification)
    demo           the strategic diagnostics and counterfactuals on the M4.3c pilot finals (2M transitions, EXPLORATORY)
    plan           the M4.4b experiment specification with the runtime estimate from smoke_train's throughput
    report         results/m44a/report.md

M4.4b stages (train_main refuses to start without --confirm-m44b):
    train_main     3 environments x 3 seeds x 2,048,000 agent transitions (results/rl/checkpoints/m44/main/)
    eval_main      the pre-registered evaluation on the main finals (results/m44b/eval_main.json.gz)
    repro_main     re-train seed 0 of every environment at the full budget; digests and histories compared
    extra_main     supplementary analyses: snapshot tournament, extra references, snapshot behaviour, integrity,
                   labelled cross-environment transfer test (supplement.py)
    traces_main    representative games found by state-defined search, exported for the viewer (supplement.py)
    recall_main    follow-up of the traces: chosen Recalls interrupted by an enemy Trade (supplement.py)
    report_main    results/m44b/report.md and results/m44b/figures/ (report_b.py)
    train_ext      the matched 2x-budget extension (training-extension policy of M4.4b): the same 9 runs at 4,096,000
                   transitions, every other setting identical (results/rl/checkpoints/m44/ext/); needs --confirm-m44b
    eval_ext       the pre-defined reduced evaluation of the extension: references, extension final vs main final,
                   self-play behaviour, snapshot behaviour (supplement.py)
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

STAGES = ["audit", "smoke_train", "smoke_repro", "smoke_eval", "demo", "plan", "report", "train_main", "eval_main",
          "repro_main", "extra_main", "traces_main", "recall_main", "report_main", "train_ext", "eval_ext"]
M44A = ["audit", "smoke_train", "smoke_repro", "smoke_eval", "demo", "plan", "report"]


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m44")
    ap.add_argument("stages", nargs="*", default=["m44a"], help=f"{STAGES}, or m44a (every M4.4a stage)")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--envs", default=None, help="comma-separated subset of four,econ,aggro")
    ap.add_argument("--seeds", default=None, help="comma-separated subset of the training seeds")
    ap.add_argument("--steps", type=int, default=None, help="agent transitions per run (default: the phase's budget)")
    ap.add_argument("--confirm-m44b", action="store_true", help="required by train_main (the full M4.4b budget)")
    a = ap.parse_args(argv)
    from . import common as C
    stages = M44A if "m44a" in a.stages else a.stages
    envs = a.envs.split(",") if a.envs else list(C.ENV_ORDER)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(C.SEEDS)
    t0 = time.perf_counter()
    for st in stages:
        print(f"===== {st} =====", flush=True)
        if st == "audit":
            from .stages import audit
            audit()
        elif st == "smoke_train":
            from .train import train
            train("smoke", envs, seeds, a.steps, a.workers)
        elif st == "smoke_repro":
            from .train import reproduce
            reproduce("smoke", [(k, seeds[0]) for k in envs], workers=min(a.workers, len(envs)))
        elif st == "smoke_eval":
            from .stages import evaluation
            evaluation("smoke", envs, seeds, a.workers)
        elif st == "demo":
            from .stages import demo
            demo(a.workers)
        elif st == "plan":
            from .plan import write_plan
            write_plan()
        elif st == "report":
            from .report import report
            report()
        elif st == "train_main":
            if not a.confirm_m44b:
                raise SystemExit("train_main is the M4.4b training budget (9 runs x 2,048,000 transitions); it is not "
                                 "part of M4.4a. Re-run with --confirm-m44b once the M4.4a review has approved it.")
            from .train import train
            train("main", envs, seeds, a.steps, a.workers)
        elif st == "eval_main":
            from .stages import evaluation
            evaluation("main", envs, seeds, a.workers)
        elif st == "repro_main":
            from .train import reproduce
            reproduce("main", [(k, seeds[0]) for k in envs], workers=min(a.workers, len(envs)))
        elif st == "extra_main":
            from .supplement import extra
            extra(envs, seeds, a.workers)
        elif st == "traces_main":
            from .supplement import traces
            traces(envs, a.workers)
        elif st == "recall_main":
            from .supplement import recall_followup
            recall_followup(envs, seeds, a.workers)
        elif st == "train_ext":
            if not a.confirm_m44b:
                raise SystemExit("train_ext is the M4.4b matched extension (9 runs x 4,096,000 transitions); re-run with "
                                 "--confirm-m44b")
            from .train import train
            train("ext", envs, seeds, a.steps, a.workers)
        elif st == "eval_ext":
            from .supplement import eval_ext
            eval_ext(envs, seeds, a.workers)
        elif st == "report_main":
            from .report_b import report_main
            report_main()
        else:
            raise SystemExit(f"unknown stage {st!r}; choose from {STAGES} or m44a")
        print(f"[{st}: {time.perf_counter() - t0:.0f} s elapsed]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
