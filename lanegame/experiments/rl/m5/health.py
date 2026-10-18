"""M5b runtime health monitoring and the training coordinator's failure policy (pre-registration §3, amendment 1).

The registered stop conditions are M4.5c's (`m45c.full.stability_scan`): a non-finite statistic, an approximate KL above
0.05 sustained over 10 updates, a gradient norm above 100, entropy below 0.3 nats, an illegal action sampled, a
never-legal code offered. M5a checked them only after every run had finished. Here they are watched while training runs:

    RunMonitor   inside each training process, `train_shared(on_update=...)`: after every update it extends the
                 stop-condition statistics with that update's history entry (the same definitions as stability_scan,
                 computed incrementally) and records the FIRST occurrence of each alarm, with its evidence, in
                 <run>/health.jsonl. It only reads the entry, so a monitored run is the unmonitored run bit for bit
                 (tested against the stored M5a smoke runs).
        DEFECT alarms        non-finite statistic, illegal action sampled, never-legal code offered. Each one means
                             that every later update of the run is uninformative (corrupted weights; a broken mask), so
                             the run is ended at that update (`StopCondition`): its history up to the update stays on
                             disk, no final.pt is written, and the run is a technical failure until investigated.
        INVESTIGATE alarms   sustained KL, gradient norm, entropy floor. These can be properties of learning, and the
                             registered policy keeps such a run as it is ("never re-run with another seed"), so the run
                             continues to its budget. Continuing loses nothing: if the investigation finds a defect, the
                             affected runs are restarted after the fix anyway. The alarm blocks every evaluation stage
                             until a recorded resolution (`resolve`) classifies it.
    Progress     in the coordinator process: every PROGRESS_EVERY seconds the tail of each run's history and its
                 health record -> results/m5/progress_<phase>.json (current) and .jsonl (log): run states, transitions,
                 recent and overall throughput, ETA, alarms, and the projected campaign wall time against the measured
                 estimate.
    run_campaign the jobs in a process pool, submitted in run order. A run that ends without status "complete"
                 (an exception, a DEFECT stop, a dead worker) is a technical failure of its pair; when more than
                 MAX_FAILED_PAIRS pairs have failed the campaign stops: pending runs are cancelled, running ones
                 terminated (their directories are kept for `--restart-incomplete`), and the stop is recorded.

Resuming. The trainer stores no optimizer or collector state, so a run cannot resume mid-schedule. Recovery from an
interruption is a deterministic restart of that run from its original start checkpoint (`--restart-incomplete`); it
repeats the lost work (up to a whole run's wall time) and must reproduce the preserved history prefix
(`train.prefix_check`, enforced by `train.check`).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import common as C

HEALTH_FILE = "health.jsonl"
PROGRESS_EVERY = 60.0
MAX_FAILED_PAIRS = 2            # "more than two pairs failing technically: stop and report"
STAT_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "grad_norm", "explained_variance")
DEFECT = ("non-finite statistic", "illegal action sampled", "never-legal code offered")


def _alarm_names():
    from ..m45c.full import ENTROPY_FLOOR, GRAD_ALARM, KL_SUSTAINED
    return {"kl": f"approximate KL above {KL_SUSTAINED} sustained over 10 updates",
            "grad": f"gradient norm above {GRAD_ALARM}", "entropy": f"entropy below {ENTROPY_FLOOR} nats",
            "nonfinite": "non-finite statistic", "illegal": "illegal action sampled",
            "never_legal": "never-legal code offered"}


class StopCondition(RuntimeError):
    """A DEFECT alarm ended the run (module docstring)."""


# ---------------------------------------------------------------------------------------------- in the training process
class RunMonitor:
    """`train_shared(on_update=RunMonitor(run_dir))`: the registered stop conditions, update by update (module
    docstring). `stop_on_defect=False` records DEFECT alarms without ending the run (tests)."""

    def __init__(self, run_dir: str, stop_on_defect: bool = True):
        from ..m45c.full import ENTROPY_FLOOR, GRAD_ALARM, KL_SUSTAINED
        self.path = os.path.join(run_dir, HEALTH_FILE)
        self.stop_on_defect = stop_on_defect
        self.kl_lim, self.grad_lim, self.ent_lim = KL_SUSTAINED, GRAD_ALARM, ENTROPY_FLOOR
        self.names = _alarm_names()
        self.window: deque = deque(maxlen=10)
        self.n = 0
        self.raised: Dict[str, dict] = {}

    def _record(self, key: str, entry: dict, evidence: dict) -> None:
        name = self.names[key]
        if name in self.raised:
            return
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "alarm": name,
               "class": "defect" if name in DEFECT else "investigate", "update": int(entry["update"]),
               "agent_transitions": int(entry["agent_transitions"]), "evidence": evidence}
        self.raised[name] = rec
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    def __call__(self, entry: dict) -> None:
        self.n += 1
        e = entry
        bad = [k for k in STAT_KEYS if k in e and not np.isfinite(e[k])]
        if bad:
            self._record("nonfinite", e, {k: repr(e[k]) for k in bad})
        self.window.append(float(e["approx_kl"]))
        if len(self.window) == 10:
            m = float(np.convolve(np.asarray(self.window), np.ones(10) / 10, mode="valid")[0])
            if m > self.kl_lim:
                self._record("kl", e, {"kl_10_update_mean": m, "window": list(self.window)})
        g = float(e.get("grad_norm_max", e["grad_norm"]))
        if g > self.grad_lim:
            self._record("grad", e, {"grad_norm_max": g})
        if float(e["entropy"]) < self.ent_lim:
            self._record("entropy", e, {"entropy": float(e["entropy"])})
        if int(e.get("illegal_sampled", 0)) > 0:
            self._record("illegal", e, {"illegal_sampled": int(e["illegal_sampled"])})
        if int(e.get("never_legal_offered", 0)) > 0:
            self._record("never_legal", e, {"never_legal_offered": int(e["never_legal_offered"])})
        if self.stop_on_defect:
            hit = [n for n in DEFECT if n in self.raised]
            if hit:
                raise StopCondition(f"stop condition at update {e['update']}: {', '.join(hit)}")


def alarms(run_dir: str) -> List[dict]:
    """The alarms a run's monitor recorded (first occurrence of each)."""
    p = os.path.join(run_dir, HEALTH_FILE)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


# ---------------------------------------------------------------------------------------------- reading a running run
def tail_entries(path: str, n: int = 20, chunk: int = 1 << 16) -> List[dict]:
    """The last n complete history entries of a (possibly growing) history.jsonl, reading only its tail."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        buf = b""
        pos = size
        while pos > 0 and buf.count(b"\n") <= n + 1:
            step = min(chunk, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
    lines = buf.split(b"\n")
    if pos > 0:
        lines = lines[1:]                                  # the first piece may be a partial line
    out = []
    for x in lines:
        x = x.strip()
        if not x:
            continue
        try:
            out.append(json.loads(x))
        except ValueError:                                 # a line being written
            continue
    return out[-n:]


def run_progress(run_dir: str, budget: int) -> dict:
    """The state of one run from its directory: queued / running / complete / failed, with progress and alarms."""
    summ = C.jload(os.path.join(run_dir, "summary.json"))
    tail = tail_entries(os.path.join(run_dir, "history.jsonl"))
    al = alarms(run_dir)
    if summ is not None:
        state = summ.get("status", "unknown")
    elif tail:
        state = "running"
    else:
        state = "queued"
    out = {"state": state, "alarms": [a["alarm"] for a in al],
           "defect_alarms": [a["alarm"] for a in al if a["class"] == "defect"]}
    if tail:
        e = tail[-1]
        overall = e["agent_transitions"] / max(e["elapsed"], 1e-9)
        recent = float(np.mean([x["sps"] for x in tail]))
        rem = max(budget - e["agent_transitions"], 0)
        out.update({"update": int(e["update"]), "agent_transitions": int(e["agent_transitions"]),
                    "fraction": e["agent_transitions"] / budget, "elapsed_h": e["elapsed"] / 3600,
                    "tps_overall": overall, "tps_recent": recent,
                    "eta_h": rem / max(overall, 1.0) / 3600 if state == "running" else 0.0,
                    "entropy": float(e["entropy"]), "approx_kl": float(e["approx_kl"]),
                    "explained_variance": float(e["explained_variance"])})
    if summ is not None and summ.get("status") != "complete":
        out["error"] = summ.get("error")
    return out


def project_wall(runs: Dict[str, dict], workers: int, budget: int, rates: Dict[str, float]) -> float:
    """Remaining campaign wall hours: running runs finish at their own rate; queued runs start, in order, when a slot
    frees up, at the observed rate of their arm (else the budget's measured rate)."""
    import heapq
    slots: List[float] = []
    queued: List[float] = []
    for name, r in runs.items():
        arm = name.split("_")[0]
        if r["state"] == "running":
            slots.append(r.get("eta_h", 0.0))
        elif r["state"] == "queued":
            queued.append(budget / max(rates.get(arm, 1.0), 1.0) / 3600)
    slots = sorted(slots)[:workers]
    while len(slots) < workers:
        slots.append(0.0)
    heapq.heapify(slots)
    for d in queued:
        t = heapq.heappop(slots)
        heapq.heappush(slots, t + d)
    return max(slots) if slots else 0.0


class Progress(threading.Thread):
    """The coordinator's progress log (module docstring)."""

    def __init__(self, phase: str, names: Sequence[str], workers: int, budget: int, every: float = PROGRESS_EVERY,
                 out: str = C.OUT):
        super().__init__(daemon=True)
        self.phase, self.names, self.workers, self.budget, self.every = phase, list(names), workers, budget, every
        self.path_now = os.path.join(out, f"progress_{phase}.json")
        self.path_log = os.path.join(out, f"progress_{phase}.jsonl")
        self.stop_flag = threading.Event()
        self.t0 = time.time()
        b = C.load("budget") or {}
        th = b.get("throughput") or {}
        self.rates = {"om": th.get("om", 1100.0), "sp": th.get("sp", 2535.0)}
        self.estimate = (b.get("training") or {}).get("wall_hours_range")

    def snapshot(self) -> dict:
        runs = {}
        for n in self.names:
            arm, s = n.split("_s")
            runs[n] = run_progress(C.run_dir(arm, int(s), self.phase), self.budget)
        rates = dict(self.rates)
        for arm in C.ARMS:
            obs = [r["tps_overall"] for n, r in runs.items() if n.startswith(arm) and r["state"] == "running"
                   and r.get("fraction", 0) > 0.02]
            if obs:
                rates[arm] = float(np.mean(obs))
        el = (time.time() - self.t0) / 3600
        rem = project_wall(runs, self.workers, self.budget, rates)
        states = {}
        for r in runs.values():
            states[r["state"]] = states.get(r["state"], 0) + 1
        return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "phase": self.phase, "elapsed_h": el,
                "projected_remaining_h": rem, "projected_total_h": el + rem, "measured_estimate_h": self.estimate,
                "rates_used": rates, "states": states,
                "alarms": {n: r["alarms"] for n, r in runs.items() if r["alarms"]}, "runs": runs}

    def write(self) -> dict:
        d = self.snapshot()
        C.jdump(self.path_now, d)
        os.makedirs(os.path.dirname(self.path_log) or ".", exist_ok=True)
        with open(self.path_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": d["time"], "elapsed_h": d["elapsed_h"],
                                "projected_total_h": d["projected_total_h"], "states": d["states"],
                                "alarms": d["alarms"],
                                "runs": {n: {k: r.get(k) for k in ("state", "agent_transitions", "tps_recent")}
                                         for n, r in d["runs"].items()}}, default=float) + "\n")
        act = [f"{n} {r.get('fraction', 0) * 100:4.1f}% {r.get('tps_recent', 0):5.0f}tps" for n, r in d["runs"].items()
               if r["state"] == "running"]
        print(f"[M5 progress {d['time']}] {d['states']} elapsed {el_fmt(d['elapsed_h'])} projected total "
              f"{el_fmt(d['projected_total_h'])} (estimate {self.estimate}); "
              + "; ".join(act) + (f"; ALARMS {d['alarms']}" if d["alarms"] else ""), flush=True)
        return d

    def run(self) -> None:
        while not self.stop_flag.wait(self.every):
            try:
                self.write()
            except Exception as ex:                        # pragma: no cover - logging must not stop the campaign
                print(f"[M5 progress] could not write the progress record: {ex!r}", flush=True)


def el_fmt(h: float) -> str:
    return f"{h:.2f} h"


# ---------------------------------------------------------------------------------------------- the coordinator
def failed_pairs(results: Dict[str, dict]) -> List[int]:
    """Pairs with at least one run that ended without status "complete"."""
    return sorted({int(n.split("_s")[1]) for n, r in results.items() if r.get("status") != "complete"})


def _terminate_children() -> int:
    try:
        import psutil
    except ImportError:                                    # pragma: no cover
        return 0
    kids = psutil.Process().children(recursive=True)
    for k in kids:
        try:
            k.terminate()
        except Exception:                                  # pragma: no cover
            pass
    psutil.wait_procs(kids, timeout=30)
    return len(kids)


def run_campaign(fn, jobs: List[tuple], names: List[str], workers: int, max_failed_pairs: int = MAX_FAILED_PAIRS,
                 on_result=None) -> dict:
    """Run the training jobs (submitted in order) under the failure policy (module docstring). Returns {"results":
    {name: summary}, "stopped": None or the stop record}."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    results: Dict[str, dict] = {}
    stopped = None
    if not jobs:
        return {"results": results, "stopped": None}
    ex = ProcessPoolExecutor(max_workers=min(workers, len(jobs)))
    try:
        futs = {ex.submit(fn, j): n for j, n in zip(jobs, names)}
        pending = set(futs)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fu in done:
                n = futs[fu]
                try:
                    r = fu.result()
                except BaseException as e:                 # noqa: BLE001 - a dead worker is a technical failure
                    r = {"status": "failed", "name": n, "error": f"worker process failed: {e!r}"}
                results[n] = r
                if on_result:
                    on_result(n, r)
                print(f"  [M5 training] {n}: {r.get('status')}"
                      + (f" ({r.get('error')})" if r.get("status") != "complete" else "")
                      + f"; {len(results)}/{len(jobs)} finished", flush=True)
            fp = failed_pairs(results)
            if len(fp) > max_failed_pairs and pending:
                stopped = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "failed_pairs": fp,
                           "rule": f"more than {max_failed_pairs} pairs failed technically: stop and report",
                           "cancelled_or_terminated": sorted(futs[f] for f in pending)}
                for fu in pending:
                    fu.cancel()
                stopped["terminated_processes"] = _terminate_children()
                for fu in pending:
                    results.setdefault(futs[fu], {"status": "stopped by the coordinator", "name": futs[fu]})
                print(f"  [M5 training] STOPPED: {stopped['rule']} (pairs {fp})", flush=True)
                break
    finally:
        ex.shutdown(wait=stopped is None, cancel_futures=True)
    return {"results": results, "stopped": stopped}


# ---------------------------------------------------------------------------------------------- resolutions and the gate
RESOLUTION_CLASSES = ("training property (not a defect)", "implementation defect (runs restarted)")


def resolutions(phase: str = "main", out: str = C.OUT) -> Dict[str, dict]:
    return (C.load(f"train_resolutions_{phase}", out) or {}).get("resolved", {})


def resolve(phase: str, check: str, classification: str, reason: str, out: str = C.OUT) -> dict:
    """Record the investigation of a failed check (never a technical failure: a missing run cannot be resolved)."""
    if classification not in RESOLUTION_CLASSES:
        raise SystemExit(f"classification must be one of {RESOLUTION_CLASSES}")
    d = C.load(f"train_resolutions_{phase}", out) or {"resolved": {}}
    d["resolved"][check] = {"classification": classification, "reason": reason,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.save_json(f"train_resolutions_{phase}", d, out)
    return d


def require_training_ok(phase: str = "main", verify_files: bool = True, out: str = C.OUT) -> dict:
    """The gate of every evaluation stage: the training check exists, every run is complete, every failed check is
    resolved as a training property, and the evaluated checkpoints are the ones the check recorded (SHA-256)."""
    d = C.load(f"train_{phase}", out)
    if not d:
        raise SystemExit(f"no training check (results/m5/train_{phase}.json): run stage train_check first")
    missing = [n for n, r in d["runs"].items() if r.get("status") != "complete"]
    if missing or len(d.get("complete", [])) != len(C.ARMS) * len(C.SEEDS):
        raise SystemExit(f"training is incomplete or failed ({missing or 'runs missing'}): evaluation refused; "
                         "investigate and report (pre-registration §3)")
    res = resolutions(phase, out)
    open_ = [k for k, v in d["checks"].items() if not v
             and (res.get(k) or {}).get("classification") != "training property (not a defect)"]
    if open_:
        raise SystemExit("failed training checks without a recorded resolution (stage train_resolve): "
                         + "; ".join(open_))
    if verify_files:
        bad = []
        for n, ident in (d.get("identities") or {}).items():
            for k, rec in ident.items():
                p = rec["path"]
                if not os.path.exists(p) or C.file_sha256(p) != rec["file_sha256"]:
                    bad.append(f"{n} {k}")
        if not d.get("identities"):
            bad.append("no checkpoint identities recorded")
        if bad:
            raise SystemExit("checkpoints differ from the training check's identities: " + "; ".join(bad))
    return d


def _probe_job(args) -> dict:
    """(name, status[, seconds]): a stand-in training job for testing `run_campaign` (no training)."""
    if len(args) > 2:
        time.sleep(float(args[2]))
    return {"name": args[0], "status": args[1]}


__all__ = ["RunMonitor", "StopCondition", "alarms", "tail_entries", "run_progress", "project_wall", "Progress",
           "failed_pairs", "run_campaign", "resolve", "resolutions", "require_training_ok", "DEFECT",
           "MAX_FAILED_PAIRS", "HEALTH_FILE"]
