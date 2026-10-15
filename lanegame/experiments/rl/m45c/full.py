"""The full M4.5c training campaign (approved 2026-10-14): six independent policies on `GameConfig.m45_calibrated()`,
seeds 0-5, 8,192,000 agent transitions each (4,096,000 joint steps, 2,000 updates), the learning rate 3e-4 annealed
linearly to 0 over each run's own budget, 100% current-policy shared self-play, snapshots every 25,600 transitions
(evaluation opponents only), the midpoint at 4,096,000 (a snapshot of the same run: a checkpoint inside the 8.192M
schedule, NOT a separately trained 4.096M policy, whose learning rate would have annealed to 0), and a full-budget
re-run of seed 0 for the bit-identity check (M4.4b protocol).

Everything about training is the pilot's (`train.train_one`, `common.ppo_config`, `common.shared_config`, the
`RolloutStats` diagnostics); the only setting that differs from the pilot is `lane_cache = 5,000` (an exact memo of
lane steps: speed and memory change, results never do; tested), chosen by the runtime audit for memory.

Robustness (the M4.4c lesson): each worker writes its own summary (`<run>/summary.json`) as soon as its run ends, and
returns an error record instead of raising, so a failure of one run or of the parent process loses nothing that was
computed; `check()` rebuilds the campaign record from the run directories alone. The parent samples system and
per-process memory every 30 s (`results/m45c/full/resources_<label>.jsonl`). `status()` reads the histories while the
runs train (read only).

Outputs: run directories `results/rl/checkpoints/m45c/full/s<seed>[_repeat]/` (git-ignored: run.json, history.jsonl,
snapshots/, final.pt, summary.json); campaign record `results/m45c/full/full_train.json`.

The matched 2x EXTENSION (approved 2026-10-15 after the pre-registered extension criterion was met; pilot report §8.4
item 2): phase "ext", the same six seeds as FRESH runs (paired initial weights, no weights transferred) with every
setting unchanged except the budget and therefore the schedule horizon, 16,384,000 agent transitions (the learning rate
annealed over 16,384,000; its midpoint 8,192,000 is a snapshot of that schedule). No re-run (reproducibility was shown
on the 8.192M campaign). Records in results/m45c/ext/, checkpoints in results/rl/checkpoints/m45c/ext/.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .common import OUT, TRANSITIONS_PER_UPDATE, meta, run_dir, run_jobs, save_json, training_spec

FULL_OUT = os.path.join(OUT, "full")
FULL_SEEDS = (0, 1, 2, 3, 4, 5)
FULL_BUDGET = 8_192_000                      # agent transitions per seed (= the schedule horizon)
FULL_MIDPOINT = 4_096_000                    # a snapshot of each run (learning rate at 50%)
FULL_LANE_CACHE = 5_000
EXT_OUT = os.path.join(OUT, "ext")
EXT_BUDGET = 16_384_000
EXT_MIDPOINT = 8_192_000
# phase: (budget, records directory, with the seed-0 re-run)
CAMPAIGNS = {"full": (FULL_BUDGET, FULL_OUT, True), "ext": (EXT_BUDGET, EXT_OUT, False)}
REPEAT_SEED = 0
REPEAT_TAG = "_repeat"
PPO_KW = {"lane_cache": FULL_LANE_CACHE}
PREFLIGHT_UPDATES = 7                         # the first snapshot (28,672)

# pre-registered stop-and-investigate thresholds (pilot report §8.4, item 4)
KL_SUSTAINED = 0.05                          # mean approximate KL over any 10 consecutive updates
GRAD_ALARM = 100.0
ENTROPY_FLOOR = 0.3


# ---------------------------------------------------------------------------------------------- jobs
def _job(args) -> dict:
    """(phase, seed, total, stop, tag): one run; writes <run>/summary.json; returns the summary or an error record."""
    phase, seed, total, stop, tag = args
    out = run_dir(phase, seed, tag)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    try:
        from .train import train_one
        s = train_one((phase, seed, total, stop, out, phase == "smoke", tag, PPO_KW))
        s.update({"status": "complete", "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "wall_s": time.perf_counter() - t0, "pid": os.getpid()})
    except BaseException as ex:                            # noqa: BLE001 - recorded, never swallowed silently
        import traceback
        s = {"status": "failed", "name": f"s{seed}{tag}", "phase": phase, "seed": seed, "run_dir": out,
             "error": repr(ex), "traceback": traceback.format_exc(), "started": started,
             "wall_s": time.perf_counter() - t0}
    if os.path.isdir(out):
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1)
    return s


def jobs(seeds: Sequence[int] = FULL_SEEDS, repeat: bool = True, phase: str = "full", total: int = FULL_BUDGET,
         stop: Optional[int] = None, tag: str = "") -> list:
    out = [(phase, s, total, stop, tag) for s in seeds]
    if repeat:
        out.append((phase, REPEAT_SEED, total, stop, tag + REPEAT_TAG))
    return out


class _ResourceLog(threading.Thread):
    """Samples system memory and the RSS of this process's children every `every` seconds into a JSONL file."""

    def __init__(self, path: str, every: float = 30.0):
        super().__init__(daemon=True)
        self.path, self.every, self.stop_flag = path, every, threading.Event()

    def run(self) -> None:
        try:
            import psutil
        except ImportError:                                # pragma: no cover
            return
        me = psutil.Process()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            while not self.stop_flag.is_set():
                vm, sw = psutil.virtual_memory(), psutil.swap_memory()
                kids = []
                for c in me.children(recursive=True):
                    try:
                        kids.append(c.memory_info().rss / 2 ** 20)
                    except Exception:                      # pragma: no cover - a child exited
                        pass
                f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "avail_gib": vm.available / 2 ** 30,
                                    "used_gib": vm.used / 2 ** 30, "swap_used_gib": sw.used / 2 ** 30,
                                    "cpu_percent": psutil.cpu_percent(), "children_rss_mb": sorted(kids)}) + "\n")
                f.flush()
                self.stop_flag.wait(self.every)


def train(seeds: Sequence[int] = FULL_SEEDS, repeat: bool = True, label: str = "main", phase: str = "full") -> dict:
    """Train the given runs of a campaign phase concurrently (one process each), then rebuild and check its record."""
    budget, out, with_repeat = CAMPAIGNS[phase]
    js = jobs(seeds, repeat, phase=phase, total=budget)
    for a in js:
        d = run_dir(a[0], a[1], a[4])
        if os.path.isdir(d) and os.listdir(d):
            raise SystemExit(f"{d} exists: the full campaign never overwrites a run")
    log = _ResourceLog(os.path.join(out, f"resources_{label}.jsonl"))
    log.start()
    t0 = time.perf_counter()
    print(f"[full:{label}] {len(js)} runs concurrently: {[f's{a[1]}{a[4]}' for a in js]}", flush=True)
    try:
        res = run_jobs(_job, js, len(js), f"full {label}")
    finally:
        log.stop_flag.set()
    wall = time.perf_counter() - t0
    for r in res:
        print(f"[full:{label}] {r['name']}: {r['status']}", flush=True)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"wall_{label}.json"), "w", encoding="utf-8") as f:
        json.dump({"label": label, "runs": [r["name"] for r in res], "wall_s": wall,
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
    return check(phase=phase, budget=budget, repeat=with_repeat, out=out)


def preflight() -> dict:
    """The campaign's code path end to end at PREFLIGHT_UPDATES updates of the 8,192,000 schedule (seed 0 and its
    re-run, lane cache 5,000) in the smoke phase: summaries, checks, reproducibility comparison."""
    stop = PREFLIGHT_UPDATES * TRANSITIONS_PER_UPDATE
    js = jobs((REPEAT_SEED,), True, phase="smoke", stop=stop, tag="_full_preflight")
    res = run_jobs(_job, js, len(js), "full preflight")
    data = check(phase="smoke", tag="_full_preflight", seeds=(REPEAT_SEED,), budget=stop, save=False)
    data["runs_status"] = [r["status"] for r in res]
    from .common import save_json as _s
    _s("preflight", data, FULL_OUT)
    return data


# ---------------------------------------------------------------------------------------------- checks
def _load_summary(phase: str, seed: int, tag: str) -> Optional[dict]:
    p = os.path.join(run_dir(phase, seed, tag), "summary.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _history(phase: str, seed: int, tag: str) -> List[dict]:
    p = os.path.join(run_dir(phase, seed, tag), "history.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stability_scan(h: List[dict]) -> dict:
    """The pre-registered stop-and-investigate conditions over a whole history (pilot report §8.4 item 4)."""
    if not h:
        return {"updates": 0}
    kl = np.array([e["approx_kl"] for e in h], dtype=float)
    roll = np.convolve(kl, np.ones(10) / 10, mode="valid") if len(kl) >= 10 else kl
    keys = ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "grad_norm", "explained_variance")
    nonfinite = sum(1 for e in h for k in keys if k in e and not np.isfinite(e[k]))
    ent = np.array([e["entropy"] for e in h])
    gmax = float(max(e.get("grad_norm_max", e["grad_norm"]) for e in h))
    out = {"updates": len(h), "nonfinite_values": int(nonfinite), "approx_kl_max": float(kl.max()),
           "approx_kl_max_10_update_mean": float(roll.max()), "grad_norm_max": gmax,
           "entropy_min": float(ent.min()), "entropy_last": float(ent[-1]),
           "illegal_sampled_total": int(sum(e.get("illegal_sampled", 0) for e in h)),
           "never_legal_offered_total": int(sum(e.get("never_legal_offered", 0) for e in h))}
    out["alarms"] = [m for m, bad in (
        ("non-finite statistic", nonfinite > 0),
        (f"approximate KL above {KL_SUSTAINED} sustained over 10 updates", roll.max() > KL_SUSTAINED),
        (f"gradient norm above {GRAD_ALARM}", gmax > GRAD_ALARM),
        (f"entropy below {ENTROPY_FLOOR} nats", ent.min() < ENTROPY_FLOOR),
        ("illegal action sampled", out["illegal_sampled_total"] > 0),
        ("never-legal code offered", out["never_legal_offered_total"] > 0)) if bad]
    return out


def _snapshot_digests(phase: str, seed: int, tag: str) -> Dict[int, str]:
    from ....rl import checkpoint_info
    d = os.path.join(run_dir(phase, seed, tag), "snapshots")
    if not os.path.isdir(d):
        return {}
    return {int(f[5:-3]): checkpoint_info(os.path.join(d, f))["digest"] for f in sorted(os.listdir(d))
            if f.startswith("snap_") and f.endswith(".pt")}


def reproducibility(phase: str = "full", seed: int = REPEAT_SEED, tag: str = "") -> dict:
    """The re-run against the primary run: final weights, every snapshot, and every non-wall-clock history field."""
    from ..m44.train import _histories_equal
    a, b = _load_summary(phase, seed, tag), _load_summary(phase, seed, tag + REPEAT_TAG)
    if not a or not b or a.get("status") != "complete" or b.get("status") != "complete":
        return {"available": False}
    da, db = _snapshot_digests(phase, seed, tag), _snapshot_digests(phase, seed, tag + REPEAT_TAG)
    differing = sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))
    ha, hb = _history(phase, seed, tag), _history(phase, seed, tag + REPEAT_TAG)
    first_diff = None
    skip = ("elapsed", "sps", "collect_s", "update_s")
    for i, (x, y) in enumerate(zip(ha, hb)):
        kx = {k: v for k, v in x.items() if k not in skip}
        ky = {k: v for k, v in y.items() if k not in skip}
        if json.dumps(kx, sort_keys=True) != json.dumps(ky, sort_keys=True):
            first_diff = {"update": i + 1, "fields": sorted(k for k in set(kx) | set(ky) if kx.get(k) != ky.get(k))}
            break
    return {"available": True, "run": a["name"], "repeat": b["name"],
            "final_digest": a["digest"], "repeat_final_digest": b["digest"],
            "identical_final_weights": a["digest"] == b["digest"],
            "identical_checkpoint_digest": a.get("checkpoint_digest") == b.get("checkpoint_digest"),
            "snapshots_compared": len(set(da) | set(db)), "snapshots_differing": differing,
            "identical_snapshots": not differing and len(da) == len(db) > 0,
            "identical_histories": _histories_equal(a["run_dir"], b["run_dir"]),
            "history_lengths": [len(ha), len(hb)], "first_differing_history_entry": first_diff,
            "identical_initial_digest": a["initial_digest"] == b["initial_digest"]}


def check(phase: str = "full", tag: str = "", seeds: Sequence[int] = FULL_SEEDS, budget: int = FULL_BUDGET,
          save: bool = True, repeat: bool = True, out: str = FULL_OUT) -> dict:
    """The campaign record rebuilt from the run directories: per-run status, pipeline and learning checks, the
    stability scan, the midpoint snapshot, throughput, memory, and the seed-0 reproducibility comparison."""
    from .common import snapshot_paths
    from .train import _omega, learning_checks, pipeline_checks
    omega = _omega()
    runs, checks, learning, scans = {}, {}, {}, {}
    names = [(s, tag) for s in seeds] + ([(REPEAT_SEED, tag + REPEAT_TAG)] if repeat else [])
    for seed, t in names:
        s = _load_summary(phase, seed, t)
        name = f"s{seed}{t}"
        h = _history(phase, seed, t)
        scans[name] = stability_scan(h)
        if s is None:
            runs[name] = {"status": "missing" if not h else f"incomplete ({len(h)} updates in history)"}
            continue
        runs[name] = s
        if s.get("status") != "complete":
            continue
        checks.update(pipeline_checks(s, omega))
        learning.update(learning_checks(s))
        snaps = dict(snapshot_paths(phase, seed, t))
        if phase in CAMPAIGNS:
            checks[f"{name}: midpoint snapshot present ({budget // 2:,})"] = budget // 2 in snaps
        checks[f"{name}: final snapshot present and equal to final.pt"] = _final_equals_last_snapshot(phase, seed, t)
        checks[f"{name}: {budget // TRANSITIONS_PER_UPDATE} updates"] = s["updates"] == budget // TRANSITIONS_PER_UPDATE
        checks[f"{name}: lane cache {FULL_LANE_CACHE}"] = s["ppo"]["lane_cache"] == FULL_LANE_CACHE
        checks[f"{name}: no pre-registered stop condition"] = not scans[name]["alarms"]
    repro = reproducibility(phase, REPEAT_SEED, tag) if repeat else {"available": False, "note": "no re-run"}
    if repro.get("available"):
        checks["seed 0 re-run: bit-identical final weights"] = repro["identical_final_weights"]
        checks["seed 0 re-run: bit-identical snapshots"] = repro["identical_snapshots"]
        checks["seed 0 re-run: identical histories (every non-wall-clock field)"] = repro["identical_histories"]
    complete = [n for n, r in runs.items() if r.get("status") == "complete"]
    data = {"meta": meta(stage="train_full_check", phase=phase), "spec": training_spec(),
            "campaign": {"seeds": list(seeds), "budget": budget, "midpoint": budget // 2, "lane_cache": FULL_LANE_CACHE,
                         "repeat": f"s{REPEAT_SEED}{REPEAT_TAG}" if repeat else None,
                         "schedule": "lr 3e-4 annealed linearly to 0 over the "
                         "run's own budget", "self_play": "100% current policy"},
            "runs": runs, "complete": complete, "stability": scans, "checks": checks, "learning": learning,
            "reproducibility": repro, "all_passed": bool(checks) and all(checks.values()) and len(complete) == len(names)}
    walls = {}
    if os.path.isdir(out):
        for f in sorted(os.listdir(out)):
            if f.startswith("wall_") and f.endswith(".json"):
                with open(os.path.join(out, f), encoding="utf-8") as fh:
                    walls[f[5:-5]] = json.load(fh)
    data["wall"] = walls
    if save:
        save_json(f"{phase}_train", data, out)
    return data


def _final_equals_last_snapshot(phase: str, seed: int, tag: str) -> bool:
    from ....rl import checkpoint_info
    from .common import final_path, snapshot_paths
    snaps = snapshot_paths(phase, seed, tag)
    if not snaps or not os.path.exists(final_path(phase, seed, tag)):
        return False
    return checkpoint_info(snaps[-1][1])["digest"] == checkpoint_info(final_path(phase, seed, tag))["digest"]


# ---------------------------------------------------------------------------------------------- monitoring
def status(phase: str = "full", seeds: Sequence[int] = FULL_SEEDS, budget: Optional[int] = None) -> dict:
    """Progress of the runs from their histories (read only; safe while training)."""
    rows = {}
    budget = budget or CAMPAIGNS.get(phase, (FULL_BUDGET, None, True))[0]
    with_repeat = CAMPAIGNS.get(phase, (None, None, True))[2]
    for seed, t in [(s, "") for s in seeds] + ([(REPEAT_SEED, REPEAT_TAG)] if with_repeat else []):
        h = _history(phase, seed, t)
        name = f"s{seed}{t}"
        if not h:
            rows[name] = {"updates": 0}
            print(f"  {name:10s} not started", flush=True)
            continue
        e = h[-1]
        recent = h[-20:]
        tps = float(np.mean([x["sps"] for x in recent]))
        overall = e["agent_transitions"] / e["elapsed"]
        eta = (budget - e["agent_transitions"]) / max(overall, 1.0)
        scan = stability_scan(h)
        rows[name] = {"updates": e["update"], "transitions": e["agent_transitions"], "recent_tps": tps,
                      "overall_tps": overall, "eta_s": eta, "entropy": e["entropy"], "approx_kl": e["approx_kl"],
                      "explained_variance": e["explained_variance"], "grad_norm": e["grad_norm"], "lr": e["lr"],
                      "shove": e.get("act_cat_SH"), "alarms": scan["alarms"]}
        print(f"  {name:10s} upd {e['update']:5d} {e['agent_transitions'] / 1e6:6.3f}M  {tps:6.0f} tps (run {overall:5.0f}) "
              f"ETA {eta / 60:5.1f} min  ent {e['entropy']:.2f} kl {e['approx_kl']:.4f} ev {e['explained_variance']:.2f} "
              f"g {e['grad_norm']:.2f} lr {e['lr']:.2e} alarms {scan['alarms'] or '-'}", flush=True)
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"  system: {vm.available / 2 ** 30:.2f} GiB available of {vm.total / 2 ** 30:.1f}", flush=True)
    except ImportError:                                    # pragma: no cover
        pass
    return rows


__all__ = ["FULL_OUT", "FULL_SEEDS", "FULL_BUDGET", "FULL_MIDPOINT", "FULL_LANE_CACHE", "REPEAT_SEED", "REPEAT_TAG",
           "EXT_OUT", "EXT_BUDGET", "EXT_MIDPOINT", "CAMPAIGNS",
           "jobs", "train", "preflight", "check", "status", "stability_scan", "reproducibility"]
