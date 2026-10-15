"""Command line of M4.5c: `python -m lanegame.experiments.rl.m45c [stages] [options]`.

Pilot stages (authorized; results/m45c/, checkpoints in results/rl/checkpoints/m45c/, git-ignored):
    audit          the frozen preset, action space, masks, Ward legality, observation / mirror checks, reference-suite
                   legality, the checkpoint guard and the OOD adapter (seconds)
    smoke          a 10-update correctness run and its bit-identical re-run (pipeline checks)
    smoke_eval     every evaluation instrument on the smoke final at tiny sizes
    pilot          THE pilot: seed 100, the 4,096,000-transition learning-rate schedule stopped at 1,024,000 agent
                   transitions (one process); refuses to overwrite an existing pilot
    prefix_check   re-train the pilot's first 7 updates (its first snapshot) in a fresh directory; weights and history must equal the pilot's
    eval_pilot     learning curve, headline (W = 1000) and history of the pilot checkpoints
    behavior_pilot state-conditioned action tables, freezes, recalls and the M4.5b ledger of the pilot checkpoints
    bench          throughput: the per-update breakdown (initial and final network), multi-process scaling and the
                   lane-cache memory / speed probe
    report         results/m45c/report.md, figures and the full-campaign plan (plan.py)
    horizon        the horizon-sensitivity study of local counterfactuals (horizon.py; resumable; ~25 min on 11 workers)
    horizon_report results/m45c/horizon/summary.json, report.md and figures (horizon_report.py)

Full campaign (approved 2026-10-14: 6 seeds x 8,192,000, lane cache 5,000, + a full-budget re-run of seed 0; full.py;
checkpoints results/rl/checkpoints/m45c/full/, records results/m45c/full/):
    train_preflight  the campaign code path at 7 updates (first snapshot) (seed 0 and its re-run, smoke phase; seconds)
    train_full       THE campaign; refuses to start without --confirm-m45c-full and never overwrites a run.
                     --seeds 0,1,2,3,4,5 (default) and --repeat {with,separate,only}: the seed-0 re-run as a 7th
                     concurrent process (with), not at all in this call (separate), or alone (only)
    train_status     progress, throughput, ETA and the pre-registered alarms of the runs (read only, safe while training);
                     --phase full (default) or ext
    train_check      rebuild results/m45c/<phase>/<phase>_train.json from the run directories (checks, reproducibility)
    train_ext        the matched 2x extension (approved 2026-10-15): six FRESH runs of 16,384,000 (seeds 0-5, phase
                     ext, every other setting unchanged); refuses to start without --confirm-m45c-ext

Limited post-training health evaluation (health.py; resumable; results/m45c/full/):
    health_training  the training histories by tenth and the pre-registered alarms (no simulation)
    health_eval      --health-parts curve,h2h,history,cycle,cross (default all): learning curve, final vs midpoint and
                     6.144M (W = 1000), final vs own past, adjacent checkpoints, cross-play of the finals. The parts
                     curve_rest, headline, tournament complete the comprehensive standard evaluation and need
                     --confirm-m45c-comprehensive
    traces_smoke     the representative-game search, export and viewer replay check at tiny sizes on --spec (traces.py)
    traces           representative games of the six finals for the viewer; needs --confirm-m45c-comprehensive
    attackers_smoke  the best-response attacker pipeline at 2 updates on final:full:0 (attackers.py)
    attackers_behaviour  records of each attacker against its target (what is exploited; after the attackers)
    attackers        THE six attackers (half the target budget); NOT authorized: needs --confirm-m45c-attackers
    health_behavior  self-play / early-game / two-opponent records and the M4.5b ledger of finals and midpoints
    health_report    results/m45c/full/summary.json, report.md and figures (health_report.py)

Revised strategic battery (battery.py: 50-turn paired counterfactuals with the 300-turn validation subset):
    battery_plan     the battery's simulated volume and runtime from its job list (no simulation)
    battery_smoke    every family end to end at tiny sizes on --spec (default final:full:0); results/m45c/battery_smoke/
    battery          THE battery on the six finals of --phase (ext: the 16.384M extension) with the replication
                     families on the 8.192M finals; refuses to start without --confirm-m45c-battery
    battery_extend   exception 3: re-run every (policy, family) that failed the 300-turn validation at 150 turns
                     (--confirm-m45c-battery)
    campaign_report  results/m45c/final/report.md, summary.json, figures: both budgets, battery, attackers, traces
    battery_report   results/m45c/battery/report_<phase>.md (+ report_full_replication.md) and summary.json
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

STAGES = ["audit", "smoke", "smoke_eval", "pilot", "prefix_check", "eval_pilot", "behavior_pilot", "bench", "report",
          "horizon", "horizon_report", "train_preflight", "train_full", "train_status", "train_check", "train_ext", "health_training", "health_eval",
          "health_behavior", "health_report", "battery_plan", "battery_smoke", "battery", "battery_extend", "battery_report", "campaign_report",
          "attackers_smoke", "attackers", "attackers_behaviour", "traces_smoke", "traces"]


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m45c")
    ap.add_argument("stages", nargs="+", help=str(STAGES))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--parts", default="curve,headline,history", help="eval_pilot parts")
    ap.add_argument("--confirm-m45c-full", action="store_true", help="required by train_full (after budget approval)")
    ap.add_argument("--seeds", default="0,1,2,3,4,5", help="train_full seeds")
    ap.add_argument("--phase", default="full", choices=("full", "ext"), help="campaign phase of train_status / "
                    "train_check / health stages")
    ap.add_argument("--confirm-m45c-ext", action="store_true", help="required by train_ext (approved extension)")
    ap.add_argument("--spec", default="final:full:0", help="battery_smoke policy")
    ap.add_argument("--confirm-m45c-battery", action="store_true", help="required by battery (comprehensive approval)")
    ap.add_argument("--confirm-m45c-comprehensive", action="store_true", help="comprehensive health_eval parts")
    ap.add_argument("--confirm-m45c-attackers", action="store_true", help="required by attackers")
    ap.add_argument("--health-parts", default="curve,h2h,history,cycle,cross", help="health_eval parts")
    ap.add_argument("--repeat", default="with", choices=("with", "separate", "only"),
                    help="train_full: the seed-0 re-run concurrently (with), not in this call (separate), or alone (only)")
    a = ap.parse_args(argv)
    from .common import WORKERS
    workers = a.workers or WORKERS
    t0 = time.perf_counter()
    for st in a.stages:
        print(f"===== {st} =====", flush=True)
        if st == "audit":
            from .stages import audit
            audit()
        elif st == "smoke":
            from .train import smoke
            smoke()
        elif st == "smoke_eval":
            from .stages import smoke_eval
            smoke_eval(workers)
        elif st == "pilot":
            from .train import pilot
            pilot()
        elif st == "prefix_check":
            from .train import prefix_check
            prefix_check()
        elif st == "eval_pilot":
            from .stages import eval_pilot
            eval_pilot(workers, a.parts.split(","))
        elif st == "behavior_pilot":
            from .stages import behavior_pilot
            behavior_pilot(workers)
        elif st == "bench":
            from .common import PILOT_SEED
            from .train import cache_probe, profile, scaling
            profile(f"initial:{PILOT_SEED}")
            profile(f"final:pilot:{PILOT_SEED}")
            scaling()
            cache_probe()
        elif st == "report":
            from .report import report
            report()
        elif st == "horizon":
            from .horizon import run as run_horizon
            run_horizon(workers)
        elif st == "horizon_report":
            from .horizon_report import summarize
            summarize()
        elif st == "train_preflight":
            from .full import preflight
            print(preflight()["checks"], flush=True)
        elif st == "train_full":
            if not a.confirm_m45c_full:
                raise SystemExit("train_full is the full M4.5c multi-seed campaign (NOT part of the pilot); it needs "
                                 "--confirm-m45c-full.")
            from .full import train
            seeds = [int(x) for x in a.seeds.split(",") if x != ""]
            if a.repeat == "only":
                d = train((), True, "repeat")
            else:
                d = train(seeds, a.repeat == "with", "main")
            print(f"all checks passed: {d['all_passed']}", flush=True)
        elif st == "train_ext":
            if not a.confirm_m45c_ext:
                raise SystemExit("train_ext is the matched 2x extension of the M4.5c campaign; it needs --confirm-m45c-ext.")
            from .full import train
            seeds = [int(x) for x in a.seeds.split(",") if x != ""]
            d = train(seeds, False, "ext", phase="ext")
            print(f"all checks passed: {d['all_passed']}", flush=True)
        elif st == "train_status":
            from .full import status
            status(a.phase)
        elif st == "train_check":
            from .full import CAMPAIGNS, check
            budget, out, rep = CAMPAIGNS[a.phase]
            d = check(phase=a.phase, budget=budget, repeat=rep, out=out)
            bad = [k for k, v in d["checks"].items() if not v]
            print(f"complete runs: {d['complete']}; failed checks: {bad or 'none'}", flush=True)
        elif st == "health_training":
            from .health import CAMPS, training
            training(camp=CAMPS[a.phase])
        elif st == "health_eval":
            from .health import COMPREHENSIVE, evaluate as health_evaluate
            parts = a.health_parts.split(",")
            if set(parts) & set(COMPREHENSIVE) and not a.confirm_m45c_comprehensive:
                raise SystemExit(f"{sorted(set(parts) & set(COMPREHENSIVE))} belong to the comprehensive evaluation; "
                                 "they need its approval and then --confirm-m45c-comprehensive.")
            from .health import CAMPS
            health_evaluate(workers, parts, camp=CAMPS[a.phase])
        elif st == "health_behavior":
            from .health import CAMPS, behavior as health_behavior
            health_behavior(workers, camp=CAMPS[a.phase])
        elif st == "health_report":
            from .health import CAMPS
            from .health_report import report as health_report
            health_report(CAMPS[a.phase])
        elif st == "battery_plan":
            import json
            from .battery import volume
            v = volume()
            print(json.dumps({k: v[k] for k in ("per_final", "replication_per_midpoint", "total_turns", "hours",
                                                "hours_range")}, indent=1), flush=True)
        elif st == "battery_smoke":
            from .battery import smoke as battery_smoke
            r = battery_smoke(a.spec, workers)
            print({k: r[k] for k in ("jobs", "turns", "validation")}, flush=True)
        elif st == "battery":
            from .battery import run as run_battery
            run_battery(a.confirm_m45c_battery, workers, phase=a.phase)
        elif st == "campaign_report":
            from .campaign_report import report as campaign_report
            campaign_report()
        elif st == "battery_extend":
            from .battery import extend as battery_extend
            battery_extend(a.confirm_m45c_battery, workers, phase=a.phase)
        elif st == "battery_report":
            from .battery import report as battery_report
            from .full import FULL_SEEDS
            primary = [f"s{s}_{a.phase}" for s in FULL_SEEDS]
            battery_report(primary, name=f"report_{a.phase}")
            if a.phase != "full":
                battery_report([f"s{s}_full" for s in FULL_SEEDS], name="report_full_replication")
        elif st == "traces_smoke":
            from .traces import smoke as traces_smoke
            r = traces_smoke(a.spec, min(workers, 2))
            print({"all_verified": r["all_verified"], "found": {k: v.get("found") for k, v in r["selected"].items()}},
                  flush=True)
        elif st == "traces":
            if not a.confirm_m45c_comprehensive:
                raise SystemExit("traces belong to the comprehensive evaluation; they need its approval and then "
                                 "--confirm-m45c-comprehensive.")
            from .full import FULL_SEEDS
            from .traces import traces as run_traces
            run_traces([f"final:{a.phase}:{s}" for s in FULL_SEEDS], workers)
        elif st == "attackers_smoke":
            from .attackers import smoke as attackers_smoke
            r = attackers_smoke()
            print({k: r["result"][k] for k in ("updates", "transitions_per_s", "eval")}, flush=True)
        elif st == "attackers_behaviour":
            from .attackers import behaviour as attackers_behaviour
            attackers_behaviour(workers, phase=a.phase)
        elif st == "attackers":
            from .attackers import run as run_attackers
            run_attackers(a.confirm_m45c_attackers, workers, phase=a.phase)
        else:
            raise SystemExit(f"unknown stage {st!r}; choose from {STAGES}")
        print(f"[{st}: {time.perf_counter() - t0:.0f} s elapsed]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
