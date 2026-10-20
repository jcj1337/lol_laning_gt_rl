"""CLI of M5: `python -m lanegame.experiments.rl.m5 <stage> [<stage> ...] [--workers N]`.

M5a readiness (bounded compute; docs/m5/m5a_readiness_protocol.md):
    ready_protocol  record the readiness protocol's SHA-256 (before the first check)
    R0 ... R10      one readiness check each (R4 = the smoke and its re-run, R9 = the concurrency benchmark, R10 = the
                    instruments at reduced samples); `ready` runs R0-R10 in order
    B1              the battery and cells instruments on the real finals at tiny sizes, before M5c (scratch root)
    A1 ... A4       the correction pass's checks (results/m5/corrections_readiness.json): A1 the monitored smoke
                    against the stored R4 runs, A2 the main training's inputs, A3 the targeted tests and digest pins,
                    A4 the full regression suite; `corrections` runs A1-A4, B1 and C1
    C1              the attacker path on the real finals at tiny sizes, before the core attackers (scratch phase)
    budget          the campaign budget from the measured rates (results/m5/budget.json, the M5a record)
    plan_future     the evaluation work by horizon and purpose and the core / optional attacker estimate
                    (results/m5/plan_future.json; no compute)
    manifest        the run manifest the training stage would write, without training (results/m5/manifest_preview.json)

The pre-registration and its amendment chain (common.py):
    prereg          freeze docs/m5/m5_preregistration.md (SHA-256 in results/m5/preregistration.json)
    prereg_archive  archive the active version's bytes (results/m5/prereg_versions/) BEFORE editing the document
    amend           record the edited document as the next version   --reason TEXT --amendment-doc PATH [--authorization TEXT]
    prereg_check    verify the chain and that the document is its active version

The campaign (each stage needs the user's approval of its compute budget, given by its flag):
    train           M5b training, 12 runs x BUDGET               --confirm-m5-train  [--restart-incomplete]
    train_check     the pre-registered health checks of the training runs (no compute)
    train_resolve   record the investigation of a failed check   --check NAME --classification TEXT --reason TEXT
    status          progress of the training runs (no compute)
    progress        the coordinator's latest progress record (no compute)
    monitor         M5b learning curves and history comparisons  --confirm-m5-eval
    battery         M5c strategic battery on every final          --confirm-m5-eval
    battery_extend  M4.5c's exception 3: re-run every (final, family) that failed horizon validation at 150 turns
                                                                   --confirm-m5-eval
    cells           M5c M4 limitation families                     --confirm-m5-eval
    families        M5d opponent families and head to head         --confirm-m5-eval
    attackers       M5d core attacker training (12)                --confirm-m5-attackers  [--restart-incomplete]
    attack_check    the attackers' health and identity checks (no compute)
    attack_eval     the attackers' independent evaluation          --confirm-m5-attackers
    attack_tables   the om finals' self-play belief tables (optional ablation only)
                                                                   --confirm-m5-attackers --include-om-table
    traces          representative games as viewer recordings (results/m5/traces/; exploratory, a few minutes)
    attack_behaviour the attackers' action components against their starting point (exploratory, a few minutes)
    reproduce       the artifact inventory with SHA-256 and the stage commands (results/m5/reproducibility.md)
    analyse         the pre-registered analyses from the records that exist (no compute beyond reading records)
    report          generated tables (results/m5/report.md)
`--include-om-table` adds the OPTIONAL om_table adaptation ablation (6 attackers and its cross-evaluation) to the attacker
stages; it is a separate decision on top of --confirm-m5-attackers. Every evaluation stage refuses to run unless the
training check passed or every failed check has a recorded resolution (`health.require_training_ok`).

`--priority below_normal` lowers the OS priority of this process and its workers.
"""

from __future__ import annotations

import argparse
import json
import time


def _lower_priority() -> None:
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS") else 10)
    except Exception as ex:                                # pragma: no cover
        print(f"could not lower the priority: {ex!r}", flush=True)


GATES = {"train": "confirm_m5_train", "monitor": "confirm_m5_eval", "battery": "confirm_m5_eval",
         "battery_extend": "confirm_m5_eval",
         "cells": "confirm_m5_eval", "families": "confirm_m5_eval", "attack_tables": "confirm_m5_attackers",
         "attackers": "confirm_m5_attackers", "attack_eval": "confirm_m5_attackers"}


TARGETED_TESTS = ["tests/test_m5.py", "tests/test_m5_corrections.py", "tests/test_om2.py", "tests/test_legacy_pins.py",
                  "tests/test_m45a_mechanics.py::test_historical_presets_are_pinned", "tests/test_m45b_candidate.py",
                  "-q"]


def _print_checks(d: dict) -> None:
    for k, v in (d.get("checks") or {}).items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)


def plan_future() -> dict:
    """The future-stage estimate (amendment 1): evaluation work by horizon and purpose; the core attacker stage (12)
    and the optional om_table addition (6 attackers, tables, cross-evaluation) kept apart."""
    from . import common as C
    from . import evaluate as EV
    b = C.load("budget") or {}
    q = (b.get("attackers") or {}).get("quarter") or {}
    tps = q.get("tps") or {}
    hb = EV.horizon_breakdown()
    rates = hb["rates"]
    w = 6

    def att_h(kinds_n: dict) -> float:
        from .bench import schedule_wall
        d = []
        for kind, n in kinds_n.items():
            r = tps.get("sp_target" if kind == "sp" else "om_target", float("nan"))
            d += [q.get("budget_per_attacker", 8_192_000) / r / 3600] * n
        return schedule_wall(d, w)

    def eval_h(include_optional: bool) -> float:
        t = EV.plan(include_optional)["M5d_attack_eval"]
        return sum(v / rates.get(k, float("nan")) for k, v in t.items()) / 3600 / w

    tab_turns = 6 * 32 * 512 * 2                           # self_play_table: 32 games x 512 turns, two om seats
    out = {"horizon_breakdown": hb,
           "attackers": {"budget_per_attacker": q.get("budget_per_attacker"), "tps": tps,
                         "core_12": {"training_wall_h": att_h({"om": 6, "sp": 6}), "eval_wall_h": eval_h(False)},
                         "optional_om_table_6": {
                             "training_wall_h_added": att_h({"om": 6, "sp": 6, "om_table": 6}) - att_h({"om": 6, "sp": 6}),
                             "tables_wall_h": tab_turns / rates.get("game_om2", float("nan")) / 3600 / w,
                             "eval_wall_h_added": eval_h(True) - eval_h(False)}},
           "meta": C.meta(stage="plan_future")}
    C.save_json("plan_future", out)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.experiments.rl.m5")
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--priority", choices=("normal", "below_normal"), default="normal")
    ap.add_argument("--restart-incomplete", action="store_true",
                    help="train, attackers: preserve an incomplete run directory under partial/ and restart that run")
    ap.add_argument("--confirm-m5-train", action="store_true", help="the user approved the M5b training budget")
    ap.add_argument("--confirm-m5-eval", action="store_true", help="the user approved the M5b-M5d evaluation budget")
    ap.add_argument("--confirm-m5-attackers", action="store_true", help="the user approved the attacker budget")
    ap.add_argument("--include-om-table", action="store_true",
                    help="attacker stages: add the OPTIONAL om_table ablation (a separate decision)")
    ap.add_argument("--reason", default="")
    ap.add_argument("--amendment-doc", default="")
    ap.add_argument("--authorization", default="")
    ap.add_argument("--check", default="")
    ap.add_argument("--classification", default="")
    args = ap.parse_args(argv)
    if args.priority == "below_normal":
        _lower_priority()
    from . import common as C
    from . import readiness as R
    stages = []
    for st in args.stages:
        stages += list(R.PARTS) if st == "ready" else (list(R.CORR_PARTS) if st == "corrections" else [st])
    for st in stages:
        gate = GATES.get(st)
        if gate and not getattr(args, gate):
            raise SystemExit(f"stage {st} runs campaign compute; it needs --{gate.replace('_', '-')} (the user's "
                             "approval of its budget). Nothing was started.")
        if st == "attack_tables" and not args.include_om_table:
            raise SystemExit("stage attack_tables serves only the optional om_table ablation; it needs "
                             "--include-om-table (a separate decision). Nothing was started.")
        t0 = time.perf_counter()
        print(f"== M5 stage {st} ({args.workers} workers)", flush=True)
        if st == "ready_protocol":
            print(R.freeze_protocol()["record"], flush=True)
        elif st in R.PARTS:
            R.freeze_protocol()
            from . import bench, instruments
            fn = {"R0": R.r0_artifacts, "R1": R.r1_seeds, "R2": R.r2_init, "R3": R.r3_defaults,
                  "R4": lambda: R.r4_smoke(min(4, args.workers)), "R5": R.r5_crn, "R6": R.r6_inputs,
                  "R7": R.r7_shares, "R8": R.r8_equivalence, "R9": lambda: bench.r9_bench(),
                  "R10": lambda: instruments.r10_instruments(args.workers)}[st]
            d = fn()
            _print_checks(d)
            print(f"  {st}: {'passed' if d.get('passed') else 'NOT passed'}", flush=True)
        elif st in R.CORR_PARTS:
            fn = {"A1": lambda: R.a1_monitored_smoke(), "A2": R.a2_campaign_inputs,
                  "B1": lambda: R.b1_battery_smoke(args.workers),
                  "C1": lambda: R.c1_attack_readiness(min(4, args.workers)),
                  "A3": lambda: R.corr_pytest("A3", TARGETED_TESTS, "targeted_tests_amendment1.log"),
                  "A4": lambda: R.corr_pytest("A4", [], "regression_suite_amendment1.log")}[st]
            d = fn()
            _print_checks(d)
            print(f"  {st}: {'passed' if d.get('passed') else 'NOT passed'} {d.get('data', {}).get('summary', '')}",
                  flush=True)
        elif st == "budget":
            from .bench import budget
            d = budget()
            print({k: v for k, v in d.get("totals", {}).items()}, flush=True)
        elif st == "plan_future":
            d = plan_future()
            print(json.dumps({"wall_h_by_part": d["horizon_breakdown"]["wall_h_by_part"],
                              "attackers": d["attackers"]}, indent=1, default=float), flush=True)
        elif st == "manifest":
            from .train import run_manifest
            C.save_json("manifest_preview", run_manifest("main", C.ARMS, C.SEEDS, C.BUDGET))
        elif st == "prereg":
            print(C.write_prereg()["record"], flush=True)
        elif st == "prereg_archive":
            print("archived:", C.archive_active(), flush=True)
        elif st == "amend":
            if not args.reason or not args.amendment_doc:
                raise SystemExit("amend needs --reason and --amendment-doc")
            d = C.amend(args.reason, args.amendment_doc, args.authorization)
            print("active:", d["active"], flush=True)
        elif st == "prereg_check":
            d = C.require_prereg()
            print("pre-registration verified; active:", C.active_version(d), flush=True)
        elif st == "train":
            from .train import train
            d = train(args.workers, restart_incomplete=args.restart_incomplete)
            print("all passed:", d["all_passed"], flush=True)
            if not d["all_passed"]:
                print("  failed checks:", d.get("failed_checks"), flush=True)
        elif st == "train_check":
            from .train import check
            d = check("main")
            _print_checks(d)
            print("all passed:", d["all_passed"], flush=True)
        elif st == "train_resolve":
            from .health import resolve
            if not (args.check and args.classification and args.reason):
                raise SystemExit("train_resolve needs --check, --classification and --reason")
            print(resolve("main", args.check, args.classification, args.reason), flush=True)
        elif st == "status":
            from .train import status
            status("main")
        elif st == "progress":
            d = C.load("progress_main") or {}
            print(json.dumps({k: d.get(k) for k in ("time", "elapsed_h", "projected_total_h", "measured_estimate_h",
                                                    "states", "alarms")}, indent=1, default=float), flush=True)
        elif st == "report":
            from .report import report
            report()
        elif st == "traces":
            from .traces import export_all
            d = export_all(args.workers)
            print(f"{len(d['recordings'])} recordings in results/m5/traces", flush=True)
        elif st == "attack_behaviour":
            from .traces import behaviour
            behaviour(args.workers)
        elif st == "reproduce":
            from .reproduce import inventory
            d = inventory()
            print({k: (v["files"], v["digest"][:12]) for k, v in d["records"].items()}, flush=True)
        elif st in ("monitor", "battery", "battery_extend", "cells", "families", "analyse"):
            C.require_prereg()
            from . import evaluate as EV
            getattr(EV, f"stage_{st}")(args.workers)
        elif st == "attackers":
            C.require_prereg()
            from . import evaluate as EV
            d = EV.stage_attackers(args.workers, include_optional=args.include_om_table,
                                   restart_incomplete=args.restart_incomplete)
            print("all passed:", d["all_passed"], flush=True)
            if not d["all_passed"]:
                print("  failed checks:", d.get("failed_checks"), flush=True)
        elif st == "attack_check":
            from .attackers import CORE_KINDS, OPTIONAL_KINDS, check
            d = check("main", CORE_KINDS + (OPTIONAL_KINDS if args.include_om_table else ()))
            _print_checks(d)
            print("all passed:", d["all_passed"], flush=True)
        elif st in ("attack_tables", "attack_eval"):
            C.require_prereg()
            from . import evaluate as EV
            getattr(EV, f"stage_{st}")(args.workers, include_optional=args.include_om_table)
        else:
            raise SystemExit(f"unknown stage {st!r}; see the module docstring")
        print(f"== {st} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
