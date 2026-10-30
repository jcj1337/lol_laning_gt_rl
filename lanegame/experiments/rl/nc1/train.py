"""NC1 PPO pilot (pre-registration, stage 2): the arms W and F continue `ext/baseline:s` under M5's sp protocol.

Everything but the network is M5's sp run, imported: `m5.train.ppo_config("sp", s, SCHEDULE)` (PPO seed 70,000 + s,
the M4.6c settings, lr annealed over 32,768,000), `shared_config()`, `collector_cls("sp")` (the M4.6c start mixture,
100% current-policy self-play), `Stats` (with the per-update actions digest) and `health.RunMonitor`. `stop_at`
ends the run at BUDGET, an exact prefix of that schedule, so each run pairs with M5's sp snapshot at BUDGET.

The smoke stage proves the pairing: the control C must reproduce M5's sp history exactly on every deterministic field,
and W and F, which start as exactly ext/baseline:s, must sample the same first rollout (update 1's actions digest).
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import List

import numpy as np

from . import common as C

SMOKE_UPDATES = 25
TRANSITIONS_PER_UPDATE = 4096


def ppo_config(arm: str, seed: int):
    from ..m5.train import ppo_config as m5_ppo
    ppo = m5_ppo("sp", seed, C.SCHEDULE)
    return dataclasses.replace(ppo, hidden=C.WIDE) if arm == "W" else ppo


def train_job(args) -> dict:
    """(arm, seed, stop, out): one run under M5's RunMonitor; writes and returns summary.json (an error record instead
    of raising)."""
    arm, seed, stop, out = args
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(1)
        from ....rl import train_shared, weights_digest
        from ..m5 import common as M5
        from ..m5.health import RunMonitor
        from ..m5.train import Stats, collector_cls, shared_config
        cfg = M5.setup("sp")
        ppo = ppo_config(arm, seed)
        net = C.initial_net(arm, seed)
        init_digest = weights_digest(net.state_dict())
        res = train_shared(cfg, ppo, shared_config(), out_dir=out, net=net, name=f"nc1_{arm}_s{seed}", verbose=False,
                           overwrite=False, stop_at=stop, rollout_stats=Stats("sp", ppo),
                           collector_cls=collector_cls("sp"), on_update=RunMonitor(out))
        h = res.history
        s = {"status": "complete", "arm": arm, "seed": seed, "run_dir": out, "schedule_total_steps": C.SCHEDULE,
             "stop_at": stop, "agent_transitions": res.agent_transitions, "env_turns": res.env_turns,
             "updates": len(h), "elapsed_s": res.elapsed, "transitions_per_s": res.agent_transitions / res.elapsed,
             "steady_transitions_per_s": float(np.median([e["sps"] for e in h[len(h) // 3:]])) if h else None,
             "snapshots": len(res.snapshot_paths), "initial_digest": init_digest,
             "digest": weights_digest(res.net.state_dict()), "spec": res.net.spec(), "ppo": ppo.to_dict(),
             "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
             "wall_s": time.perf_counter() - t0, "pid": os.getpid()}
    except BaseException as ex:                            # noqa: BLE001 - recorded in the summary, never dropped
        import traceback
        from ..m5.health import StopCondition
        s = {"status": "stopped" if isinstance(ex, StopCondition) else "failed", "arm": arm, "seed": seed,
             "run_dir": out, "error": repr(ex), "traceback": traceback.format_exc(), "started": started,
             "wall_s": time.perf_counter() - t0}
    if os.path.isdir(out):
        from ..m5.common import jdump
        jdump(os.path.join(out, "summary.json"), s)
    return s


def history(out: str) -> List[dict]:
    from ..m5.train import history as h
    return h(out)


def smoke(workers: int) -> dict:
    """C, W and F on seeds 0 and 1 for SMOKE_UPDATES updates, concurrently (the pilot's load): the reproduction checks
    and the measured throughput."""
    import shutil
    from ..m5.train import DETERMINISTIC_KEYS
    stop = SMOKE_UPDATES * TRANSITIONS_PER_UPDATE
    jobs = []
    for arm in ("C", "W", "F"):
        for s in (0, 1):
            d = C.run_dir(arm, s, "smoke")
            if os.path.isdir(d):
                shutil.rmtree(d)
            jobs.append((arm, s, stop, d))
    res = C.run_jobs(train_job, jobs, workers, "NC1 smoke")
    out = {"meta": C.meta(stage="smoke", updates=SMOKE_UPDATES, concurrent=min(workers, len(jobs))), "runs": {}}
    for (arm, s, _, d), r in zip(jobs, res):
        ours, m5 = history(d), history(os.path.join("results", "rl", "checkpoints", "m5", "train", "main", "sp",
                                                    f"s{s}"))
        n = len(ours)
        same = [all(ours[i].get(k) == m5[i].get(k) for k in DETERMINISTIC_KEYS if k in ours[i]) for i in range(n)]
        out["runs"][f"{arm}_s{s}"] = {
            "status": r["status"], "error": r.get("error"), "updates": n,
            "identical_to_m5_updates": int(sum(same)),
            "first_rollout_identical": bool(n and ours[0].get("actions_digest") == m5[0].get("actions_digest")),
            "steady_transitions_per_s": r.get("steady_transitions_per_s"),
            "transitions_per_s": r.get("transitions_per_s"), "spec": r.get("spec")}
    runs = out["runs"]
    out["checks"] = {
        "control_reproduces_m5": all(runs[f"C_s{s}"]["identical_to_m5_updates"] == SMOKE_UPDATES for s in (0, 1)),
        "arms_first_rollout_identical": all(runs[f"{a}_s{s}"]["first_rollout_identical"]
                                            for a in C.PILOT_ARMS for s in (0, 1)),
        "all_complete": all(v["status"] == "complete" for v in runs.values())}
    rate = {a: float(np.mean([runs[f"{a}_s{s}"]["steady_transitions_per_s"] or np.nan for s in (0, 1)]))
            for a in ("C", "W", "F")}
    out["throughput"] = {"steady_transitions_per_s": rate,
                         "projected_run_h": {a: C.BUDGET / r / 3600 for a, r in rate.items()}}
    C.save_json("smoke", out)
    return out


def train(workers: int) -> dict:
    """The pilot: PILOT_ARMS x SEEDS to BUDGET (resumable at run granularity: a complete run is kept, an incomplete
    directory is moved aside and restarted from its start)."""
    import shutil
    C.require_prereg()
    jobs = []
    for arm in C.PILOT_ARMS:
        for s in C.SEEDS:
            d = C.run_dir(arm, s)
            summ = os.path.join(d, "summary.json")
            if os.path.exists(summ):
                from ..m5.common import jload
                if (jload(summ) or {}).get("status") == "complete":
                    continue
            if os.path.isdir(d):
                shutil.move(d, os.path.join(C.CKPT, "train", "partial", f"{arm}_s{s}_{time.strftime('%Y%m%d-%H%M%S')}"))
            jobs.append((arm, s, C.BUDGET, d))
    t0 = time.perf_counter()
    res = C.run_jobs(train_job, jobs, workers, "NC1 training")
    out = C.load_json("training") or {"runs": {}}
    for r in res:
        out["runs"][f"{r['arm']}_s{r['seed']}"] = {k: v for k, v in r.items() if k != "traceback"}
    out["meta"] = C.meta(stage="train", workers=workers, elapsed_s=time.perf_counter() - t0)
    C.save_json("training", out)
    return out
