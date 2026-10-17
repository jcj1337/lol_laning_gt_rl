"""R9, the concurrency benchmark, and the campaign budget (readiness protocol R9; design §10).

Benchmark: for each configuration, a set of short training runs of the campaign's arms (and attackers) is started at
once (one process each, the campaign's settings; PPO seeds BENCH_ROOT + k, never a campaign seed). A run's throughput is
measured only inside the window in which every run of the configuration is active, after a warm-up of WARMUP_S: the
transitions it completed in that window over the window's length. The aggregate is the sum over runs.

Calibration: short runs see a colder lane memo and other start-up effects than a full run. The sp arm's code path is
exactly the M4.6c / M4.6f continuation's, whose full-length runs were measured at 6 concurrent (results/m46f/
train_ext.json: 2,450-2,620 transitions/s); the om arm's is exactly OM2's treatment (results/om2/train_main.json:
1,416-1,645 at 5 workers beside the control runs). The budget uses the full-run figures where they exist and the
benchmark's ratios elsewhere, and reports both.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Sequence

import numpy as np

from . import common as C

WARMUP_S = 15.0
BENCH_UPDATES = {"om": 36, "sp": 30, "att": 30}
CONFIGS = (("om4", {"om": 4}), ("om5", {"om": 5}), ("om6", {"om": 6}), ("om8", {"om": 8}),
           ("sp6", {"sp": 6}), ("sp8", {"sp": 8}), ("mix6", {"om": 3, "sp": 3}),
           ("att6", {"att_om": 3, "att_sp": 3}))
HIST = {"sp_full_6": (2_450, 2_620), "om_full_5": (1_416, 1_645)}


def _jobs(name: str, mix: Dict[str, int]) -> List[tuple]:
    from . import readiness as R
    out = []
    k = 0
    for kind, n in mix.items():
        for _ in range(n):
            d = os.path.join(C.CKPT, "bench", name, f"{kind}_{k}")
            if kind in C.ARMS:
                stop = BENCH_UPDATES[kind] * (3072 if kind == "om" else 4096)
                out.append(("train", ("bench", kind, k % 6, C.BUDGET, stop, True, d, C.BENCH_ROOT + k)))
            else:
                tk = kind.split("_")[1]
                spec = R.om_proxy(k % 3) if tk == "om" else R.sp_proxy(k % 2)
                out.append(("attack", (tk, k % 6, 16_384_000, BENCH_UPDATES["att"] * 2048, "bench", True, spec, d)))
            k += 1
    return out


def _run(job) -> dict:
    kind, args = job
    if kind == "train":
        from .train import train_job
        return train_job(args)
    from .attackers import train_job
    return train_job(args)


def _window(runs: List[dict]) -> dict:
    """Per-run and aggregate throughput inside the common active window."""
    from .train import history
    hs = []
    for r in runs:
        if r.get("status") != "complete":
            return {"error": [r.get("error") for r in runs if r.get("status") != "complete"]}
        h = history(r["run_dir"])
        t0 = r["train_start_epoch"]
        hs.append([(t0 + e["elapsed"], e["agent_transitions"]) for e in h])
    lo = max(h[0][0] for h in hs) + WARMUP_S
    hi = min(h[-1][0] for h in hs)
    if hi <= lo:
        lo = max(h[0][0] for h in hs)
    per = []
    for h in hs:
        t = np.asarray([x[0] for x in h])
        n = np.asarray([x[1] for x in h], dtype=float)
        a, b = np.interp(lo, t, n), np.interp(hi, t, n)
        per.append((b - a) / max(hi - lo, 1e-9))
    return {"window_s": hi - lo, "per_run": per, "aggregate": float(sum(per)),
            "peak_mb": [((r.get("memory") or {}).get("peak_mb")) for r in runs]}


def r9_bench(configs=CONFIGS) -> dict:
    from . import readiness as R
    out, shares = {}, None
    t_all = time.perf_counter()
    for name, mix in configs:
        js = _jobs(name, mix)
        t0 = time.perf_counter()
        res = C.run_jobs(_run, js, len(js), f"bench {name}")
        w = _window(res)
        kinds = [j[1][1] if j[0] == "train" else f"att_{j[1][0]}" for j in js]
        by = {}
        for kd, v in zip(kinds, w.get("per_run", [])):
            by.setdefault(kd, []).append(v)
        out[name] = {"mix": mix, "wall_s": time.perf_counter() - t0, **w,
                     "per_run_by_kind": {k: float(np.mean(v)) for k, v in by.items()}}
        if name == "mix6":
            shares = [r.get("shares") for r, k in zip(res, kinds) if k == "om"]
        print(f"  [bench {name}] {out[name].get('per_run_by_kind')} aggregate {out[name].get('aggregate')}", flush=True)
    rec = {"configs": out, "shares_om": shares, "historical_full_runs": HIST, "wall_s": time.perf_counter() - t_all,
           "machine": machine()}
    ok = all("per_run" in v for v in out.values())
    checks = {"every benchmark configuration ran": ok}
    if ok:
        rec["recommendation"] = recommend(out)
    return R._record("R9", {"checks": checks, "data": rec})


def machine() -> dict:
    try:
        import psutil
        return {"logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                "ram_gib": psutil.virtual_memory().total / 2 ** 30}
    except Exception:                                      # pragma: no cover
        return {"logical_cpus": os.cpu_count()}


def recommend(out: Dict[str, dict]) -> dict:
    """The worker count with the best aggregate om throughput (the long arm), and the measured scaling."""
    agg = {n: out[n]["aggregate"] for n in ("om4", "om5", "om6", "om8") if n in out}
    best = max(agg, key=agg.get)
    return {"om_aggregate_by_workers": agg, "best": best, "workers": int(best[2:]),
            "sp_aggregate": {n: out[n]["aggregate"] for n in ("sp6", "sp8") if n in out}}


# ---------------------------------------------------------------------------------------------- budget
def rate_model(cfgs: Dict[str, dict]):
    """Per-run transitions/s of an om or sp run when n_om om runs and n_sp sp runs are active (short-run benchmark
    scale): the measured configurations, interpolated in the total number of runs; mixed loads scale the measured
    3 + 3 mix by the om-only contention curve."""
    om_pts = sorted((int(n[2:]), v["per_run_by_kind"]["om"]) for n, v in cfgs.items()
                    if n.startswith("om") and "per_run_by_kind" in v)
    sp_pts = sorted((int(n[2:]), v["per_run_by_kind"]["sp"]) for n, v in cfgs.items()
                    if n.startswith("sp") and "per_run_by_kind" in v)
    mix = cfgs.get("mix6", {}).get("per_run_by_kind", {})

    def interp(pts, n):
        xs, ys = zip(*pts)
        return float(np.interp(n, xs, ys)) if n >= xs[0] else float(ys[0] * (1 + 0.0 * (xs[0] - n)))

    def f(n):                                              # contention relative to 6 runs (om-only curve)
        return interp(om_pts, n) / interp(om_pts, 6)

    def rate(arm: str, n_om: int, n_sp: int) -> float:
        n = n_om + n_sp
        if n_sp == 0:
            return interp(om_pts, n)
        if n_om == 0:
            return interp(sp_pts, n) if n >= sp_pts[0][0] else sp_pts[0][1] * f(n) / f(sp_pts[0][0])
        return mix[arm] * f(n)
    return rate


def simulate(work: Dict[str, int], workers: int, rate, budget: float, order: Sequence[str]) -> float:
    """Wall seconds of the campaign: runs started in `order` (a list of arm names) on `workers` processes, each run
    progressing at rate(arm, active om, active sp) (event-driven, piecewise constant rates)."""
    queue = list(order)
    active: List[list] = []                                # [arm, remaining transitions]
    t = 0.0
    while queue or active:
        while queue and len(active) < workers:
            active.append([queue.pop(0), float(budget)])
        n_om = sum(1 for a in active if a[0] == "om")
        n_sp = len(active) - n_om
        rates = [rate(a[0], n_om, n_sp) for a in active]
        dt = min(r_[1] / r for r_, r in zip(active, rates))
        t += dt
        for a, r in zip(active, rates):
            a[1] -= r * dt
        active = [a for a in active if a[1] > 1e-6]
    return t


def schedule_wall(durations: Sequence[float], workers: int) -> float:
    """Wall time of jobs on `workers` processes, longest first (list scheduling, as run_jobs submits in order)."""
    free = [0.0] * workers
    for d in sorted(durations, reverse=True):
        i = int(np.argmin(free))
        free[i] += d
    return max(free)


def budget(workers: int = None) -> dict:
    """The campaign budget from the readiness measurements (R9 throughputs, R10 per-turn costs) and the design's sizes
    (`evaluate.plan`). Written to results/m5/budget.json."""
    from . import evaluate as EV
    rec = (C.load("readiness") or {}).get("parts", {})
    r9 = (rec.get("R9") or {}).get("data", {})
    r10 = (rec.get("R10") or {}).get("data", {})
    if not r9 or not r10:
        raise SystemExit("run R9 and R10 first")
    cfgs = r9["configs"]
    W = workers or 6
    om_bench = cfgs["om6"]["per_run_by_kind"]["om"]
    sp_bench = cfgs.get("sp6")["per_run_by_kind"]["sp"]
    sp_full = float(np.mean(HIST["sp_full_6"]))
    # two calibrations of the short-run rates to full-length runs: "conservative" applies the sp factor (identical sp
    # code path at 6 concurrent: M4.6f) to both arms; "full_run" applies each arm's own full-run evidence (sp: M4.6f;
    # om: OM2's treatment runs at 5 workers against the benchmark's om5)
    cal = sp_full / sp_bench
    om5 = cfgs.get("om5", cfgs.get("om6"))["per_run_by_kind"]["om"]
    cal_om_full = float(np.mean(HIST["om_full_5"])) / om5
    tps = {"om": om_bench * cal, "sp": sp_full, "om_bench": om_bench, "sp_bench": sp_bench, "calibration": cal,
           "om_full_run_calibration": cal_om_full, "om_full_run_estimate": om_bench * cal_om_full}
    base = rate_model(cfgs)
    CAL = {"om": cal, "sp": cal}

    def rate(arm, n_om, n_sp):
        return base(arm, n_om, n_sp) * CAL[arm]

    sched = {}
    for w in (4, 5, 6, 8):
        for name, order in (("om_first", ["om"] * 6 + ["sp"] * 6), ("interleaved", ["om", "sp"] * 6)):
            sched[f"W{w}_{name}"] = simulate({}, w, rate, C.BUDGET, order) / 3600
    CAL["om"] = cal_om_full
    sched_full = {f"W{w}_{name}": simulate({}, w, rate, C.BUDGET, order) / 3600 for w in (4, 5, 6, 8)
                  for name, order in (("om_first", ["om"] * 6 + ["sp"] * 6), ("interleaved", ["om", "sp"] * 6))}
    CAL["om"] = cal
    best = min(sched, key=sched.get)
    W = int(best.split("_")[0][1:]) if workers is None else W
    best_full = min(sched_full, key=sched_full.get)
    out = {"workers": W, "throughput": tps, "calibration_note": "short-run rates x (sp full-run / sp short-run); OM2's "
           "full treatment runs (1,416-1,645 at 5 workers beside control runs) are a cross-check for om",
           "training": {"schedules_wall_hours": sched, "schedules_wall_hours_full_run_calibration": sched_full,
                        "best": best, "best_full_run_calibration": best_full,
                        "wall_hours_range": [sched_full[best_full], sched[best]],
                        "om_run_hours_range_at_6": [C.BUDGET / (base("om", 6, 0) * cal_om_full) / 3600,
                                                    C.BUDGET / (base("om", 6, 0) * cal) / 3600],
                        "run_hours": {"om_at_6_om_only": C.BUDGET / rate("om", 6, 0) / 3600,
                                      "om_in_3+3_mix": C.BUDGET / rate("om", 3, 3) / 3600,
                                      "sp_at_6_sp_only": C.BUDGET / rate("sp", 0, 6) / 3600,
                                      "sp_in_3+3_mix": C.BUDGET / rate("sp", 3, 3) / 3600},
                        "wall_hours": sched[best] if workers is None else
                        min(v for k, v in sched.items() if k.startswith(f"W{W}_")),
                        "midpoint_wall_hours_if_stopped": (sched[best] if workers is None else
                                                           min(v for k, v in sched.items()
                                                               if k.startswith(f"W{W}_"))) / 2}}
    rates = r10.get("rates", {})
    plan = EV.plan()
    ev = {}
    for part, items in plan.items():
        secs = 0.0
        for kind, turns in items.items():
            r = rates.get(kind)
            if r is None:
                continue
            secs += turns / r
        ev[part] = {"turns": items, "worker_hours": secs / 3600, "wall_hours": secs / 3600 / W}
    out["evaluation"] = ev
    att = cfgs.get("att6", {}).get("per_run_by_kind", {})
    from .attackers import HALF_BUDGET, QUARTER_BUDGET
    a_om, a_sp = att.get("att_om"), att.get("att_sp")
    if a_om and a_sp:
        a_om, a_sp = a_om * cal, a_sp * cal
        for name, bud in (("half", HALF_BUDGET), ("quarter", QUARTER_BUDGET)):
            runs = [bud / a_om] * 12 + [bud / a_sp] * 6        # 6 om + 6 om_table (om-format targets), 6 sp
            out.setdefault("attackers", {})[name] = {
                "budget_per_attacker": bud, "tps": {"om_target": a_om, "sp_target": a_sp},
                "wall_hours_18": schedule_wall(runs, W) / 3600,
                "wall_hours_12_without_adaptation_test": schedule_wall(runs[:6] + runs[12:], W) / 3600}
    tot = out["training"]["wall_hours"] + sum(v["wall_hours"] for v in ev.values())
    out["totals"] = {"training_wall_h": out["training"]["wall_hours"],
                     "evaluation_wall_h": sum(v["wall_hours"] for v in ev.values()),
                     "attackers_half_wall_h": out.get("attackers", {}).get("half", {}).get("wall_hours_18"),
                     "attackers_quarter_wall_h": out.get("attackers", {}).get("quarter", {}).get("wall_hours_18"),
                     "without_attackers_wall_h": tot}
    out["meta"] = C.meta(stage="budget")
    C.save_json("budget", out)
    return out


__all__ = ["r9_bench", "budget", "schedule_wall", "CONFIGS", "recommend", "machine"]
