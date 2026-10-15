"""The full M4.5c campaign plan: runtime estimates from MEASURED pilot throughput, the evaluation design, and the
recommended budget (results/m45c/full_plan.json; tables in results/m45c/report.md §10-§11).

Measured inputs (results/m45c/):
    pilot_train.json        single-process end-to-end training throughput over the whole pilot (agent transitions / s:
                            collection + PPO update + snapshots + diagnostics) and its peak memory
    bench_scaling.json      per-process throughput with 1, 3, 6, 11 concurrent training processes (short runs): the
                            slowdown factor under load
    bench_cache.json        memory and collection speed against the lane-cache size (an exact memo: results unchanged)
    pilot_eval.json         evaluation throughput (simulated turns per worker-second) of the curve / headline stages
    pilot_behavior.json, strategic_smoke.json   behaviour and counterfactual-framework costs

Extrapolation (labelled as such in every table): a run of B agent transitions on one of n concurrent processes takes
B / (tps_pilot x slowdown(n) x cache_factor), slowdown interpolated linearly between the measured process counts;
cache_factor = 1 at lane_cache 20,000 and the measured collection-speed ratio at 5,000 (used when the concurrent runs
would not fit in memory at 20,000). S seeds run concurrently on min(S, MAX_PROCS) processes in ceil(S / MAX_PROCS)
waves, or one after another. Per-run throughput is assumed constant over a run (the pilot's per-update throughput
over its 250 updates is reported to check this).
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .common import TRANSITIONS_PER_UPDATE, load_json, save_json

MAX_PROCS = 11                                     # 12 logical CPUs, one left for the system
RAM_GB = 17.0                                      # this machine
RAM_RESERVE_GB = 3.0                               # left for the system and the evaluation parent process
BUDGETS = (1_000_000, 2_048_000, 4_000_000, 4_096_000, 5_000_000, 8_000_000, 8_192_000, 10_000_000, 16_384_000)
SEED_COUNTS = (1, 3, 5, 6, 8, 10)

# ---- the recommendation (from the pilot's measured results; docs/m4/m4_5/m4_5c_pilot_report.md §8)
RECOMMENDED = {
    "label": "default", "seeds": 6, "seed_list": [0, 1, 2, 3, 4, 5], "transitions_per_seed": 8_192_000,
    "updates_per_seed": 2_000, "joint_steps_per_seed": 4_096_000,
    "schedule": "learning rate 3e-4 annealed linearly to 0 over each run's own 8,192,000 transitions (M4.4b protocol)",
    "self_play": "100% current-policy shared self-play (as M4.4b); no historical training opponents",
    "snapshot_every": 25_600, "curve_every": 409_600, "midpoint": 4_096_000,
    "lane_cache": 20_000, "concurrency": "6 processes at once (fits in memory at lane_cache 20,000)",
}
ALTERNATIVES = {
    "lower_cost": {"label": "lower cost", "seeds": 3, "transitions_per_seed": 4_096_000, "midpoint": 2_048_000,
                   "lane_cache": 20_000},
    "higher_confidence": {"label": "higher confidence", "seeds": 10, "transitions_per_seed": 16_384_000,
                          "midpoint": 8_192_000, "lane_cache": 5_000, "curve_every": 819_200},
}

# ---- the full evaluation design (pilot report §9), per seed unless stated; W = window, P = paired seeds
N_OPP = 10                                         # 9 scripted (references.SUITE) + the OOD M4.4b network
EVAL = {
    "curve": "initial + every curve_every snapshot vs the 10 opponents, P 32, W 460",
    "headline": "final and midpoint vs the 10 opponents, P 64, W 1000",
    "history": "final vs 8 evenly spaced own snapshots and the initial network, P 32, W 460",
    "tournament": "9 evenly spaced checkpoints of the run, round robin (36 pairs), P 16, W 460 (cycling check)",
    "cross_play": "every pair of finals (all seeds), P 64, W 1000",
    "behaviour": "records: 5 checkpoints in self-play 16 x 1000 turns, the final vs 6 opponents 8 x 500 x 2 seats; ledger",
    "strategic": "6 counterfactual families x 48 natural states x 16 paired rollouts x 100 turns (strategic.py), plus "
                 "initialized power-gap / freeze scenarios for families B and C (natural power gaps are rare)",
}


def eval_turns(seeds: int, budget: int, midpoint: int, curve_every: int = 409_600) -> Dict[str, float]:
    curve_pts = budget // curve_every + 1
    return {"curve": seeds * curve_pts * N_OPP * 2 * 32 * 460,
            "headline": seeds * 2 * N_OPP * 2 * 64 * 1000,
            "history": seeds * 9 * 2 * 32 * 460,
            "tournament": seeds * 36 * 2 * 16 * 460,
            "cross_play": seeds * (seeds - 1) // 2 * 2 * 64 * 1000,
            "strategic": seeds * 6 * 48 * 4.5 * 16 * 100 * 1.25}          # +25%: state collection and scenarios


def _measured() -> dict:
    pt, sc, ca = load_json("pilot_train"), load_json("bench_scaling"), load_json("bench_cache")
    if pt is None or sc is None:
        raise RuntimeError("run the pilot and bench stages first")
    run = pt["run"]
    s = sc["scaling"]
    slow = {int(k): v["slowdown_vs_1"] for k, v in s.items()}
    cache = {}
    if ca:
        for p in ca["probe"]:
            late = [r for r in p["rows"] if r["rollouts"] >= 110]          # caches full (>= 14,080 steps per env)
            cache[p["lane_cache"]] = {"collect_s_per_rollout": float(np.mean([r["collect_s_per_rollout"] for r in late])),
                                      "rss_mb_full": float(max(r["rss_mb"] for r in late))}
    return {"tps_pilot": run["transitions_per_s"], "env_turns_per_s_pilot": run["env_turns_per_s"],
            "collect_s": run["collect_s"], "update_s": run["update_s"], "elapsed_s": run["elapsed_s"],
            "slowdown": slow, "peak_mb_pilot": run.get("memory", {}).get("peak_mb"), "cache": cache}


def cache_factor(lane_cache: int, m: dict) -> float:
    """End-to-end speed relative to lane_cache 20,000 (collection is ~80% of an update; the PPO update is unchanged)."""
    c = m["cache"]
    if lane_cache == 20_000 or lane_cache not in c or 20_000 not in c:
        return 1.0
    ratio = c[20_000]["collect_s_per_rollout"] / c[lane_cache]["collect_s_per_rollout"]
    share = m["collect_s"] / (m["collect_s"] + m["update_s"])
    return 1.0 / (share / ratio + (1.0 - share))


def memory_gb(procs: int, lane_cache: int, m: dict) -> float:
    c = m["cache"].get(lane_cache)
    mb = c["rss_mb_full"] if c else (m["peak_mb_pilot"] or 1600.0)
    return procs * mb / 1024.0


def slowdown(n: int, slow: Dict[int, float]) -> float:
    ks = sorted(slow)
    return float(np.interp(min(max(n, ks[0]), ks[-1]), ks, [slow[k] for k in ks]))


def run_hours(budget: int, procs: int, m: dict, lane_cache: int = 20_000) -> float:
    return budget / (m["tps_pilot"] * slowdown(procs, m["slowdown"]) * cache_factor(lane_cache, m)) / 3600.0


def campaign_hours(budget: int, seeds: int, m: dict, concurrent: bool = True, lane_cache: int = 20_000) -> float:
    if not concurrent:
        return seeds * run_hours(budget, 1, m, lane_cache)
    p = min(seeds, MAX_PROCS)
    return math.ceil(seeds / MAX_PROCS) * run_hours(budget, p, m, lane_cache)


def eval_rate() -> Dict[str, float]:
    """Measured simulated turns per worker-second of the pilot's evaluation stages."""
    ev = load_json("pilot_eval") or {}
    out = {}
    for part in ("curve", "headline", "history"):
        if part in ev and f"{part}_spec" in ev:
            turns = sum(2 * r["pairs"] * ev[f"{part}_spec"]["window"] for r in ev[part].values())
            t = ev["timing"][part]
            out[part] = turns / (t["wall_s"] * t["workers"])
    sm = load_json("strategic_smoke")
    if sm:
        out["strategic_smoke_wall_s"] = sm["elapsed_s"]
    beh = load_json("pilot_behavior")
    if beh:
        out["behaviour_wall_s"] = beh["elapsed_s"]
    return out


def eval_hours(seeds: int, budget: int, midpoint: int, rates: Dict[str, float], workers: int = MAX_PROCS,
               curve_every: int = 409_600) -> dict:
    r = float(np.mean([rates[k] for k in ("curve", "headline") if k in rates]))
    t = eval_turns(seeds, budget, midpoint, curve_every)
    h = {k: v / (r * workers) / 3600.0 for k, v in t.items()}
    h["behaviour"] = seeds * 1.5 * rates.get("behaviour_wall_s", 60.0) / 3600.0
    return {"hours": h, "turns": t, "turns_per_worker_s": r}


def plan() -> dict:
    m = _measured()
    rates = eval_rate()
    rows = []
    for b in BUDGETS:
        r = {"budget": b, "updates": math.ceil(b / TRANSITIONS_PER_UPDATE), "joint_steps": b // 2,
             "one_run_alone_h": run_hours(b, 1, m), "one_run_under_11_h": run_hours(b, MAX_PROCS, m)}
        for s in SEED_COUNTS:
            r[f"seeds{s}_concurrent_h"] = campaign_hours(b, s, m, True)
            r[f"seeds{s}_sequential_h"] = campaign_hours(b, s, m, False)
        rows.append(r)
    mem = {f"{n} processes": {str(c): memory_gb(n, c, m) for c in sorted(m["cache"]) or [20_000]}
           for n in (3, 6, 8, 10, 11)}
    opt_rows = {}
    for k, o in {"default": RECOMMENDED, **ALTERNATIVES}.items():
        lc = o.get("lane_cache", 20_000)
        train_h = campaign_hours(o["transitions_per_seed"], o["seeds"], m, True, lc)
        ev = eval_hours(o["seeds"], o["transitions_per_seed"], o["midpoint"], rates,
                        curve_every=o.get("curve_every", 409_600))
        std = sum(v for kk, v in ev["hours"].items() if kk != "strategic")
        opt_rows[k] = {"label": o["label"], "seeds": o["seeds"], "transitions_per_seed": o["transitions_per_seed"],
                       "total_agent_transitions": o["seeds"] * o["transitions_per_seed"], "lane_cache": lc,
                       "memory_gb": memory_gb(min(o["seeds"], MAX_PROCS), lc, m),
                       "training_h": train_h, "standard_evaluation_h": std,
                       "strategic_battery_h": ev["hours"]["strategic"], "evaluation_parts_h": ev["hours"],
                       "evaluation_turns": ev["turns"]}
        opt_rows[k]["total_h"] = train_h + std + ev["hours"]["strategic"]
    data = {"measured": m, "evaluation_rates": rates, "estimates": rows, "memory_gb_at_full_cache": mem,
            "options": opt_rows, "recommended": RECOMMENDED, "alternatives": ALTERNATIVES, "evaluation_design": EVAL,
            "note": "estimates are EXTRAPOLATED from the measured pilot and benchmarks (module docstring)"}
    save_json("full_plan", data)
    return data


__all__ = ["plan", "run_hours", "campaign_hours", "eval_rate", "eval_hours", "eval_turns", "slowdown", "cache_factor",
           "memory_gb", "RECOMMENDED", "ALTERNATIVES", "EVAL", "BUDGETS", "SEED_COUNTS", "MAX_PROCS"]
