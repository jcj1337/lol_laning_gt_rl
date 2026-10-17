"""Continuation jobs of M4.8: one job = one checkpoint state x a set of combinations x K continuations of T_MAX turns.

Combination (`combos[name]`): {"env": the environment it plays in, "a": seat A's policy spec, "b": seat B's,
"variant": a state variant (VARIANTS), "mirror": bool}. Continuation r of every combination of a state uses the same
seed `common.roll_seed(root, index, r)` for the chance stream and both sampling streams (common random numbers). A
mirrored combination plays the mirrored state with the mirrored chance vector and the seats' sampling streams swapped;
for the same policies it is the exact mirror image of the unmirrored one (checked).

State variants of seat A (CONTROLLED, constructed; D0 is unchanged by construction because earned = invested +
unspent):
    orig      the checkpoint as it is
    A_fresh   A has just come back from a reset: unspent gold invested, full HP, at its own tower cell, present
    A_away1   A has just recalled: away for one more turn at its tower cell (HP and unspent gold as they are; it
              invests and refills when it comes back, as after any Recall)

Stored per combination (npz, "<combo>__<key>"): r [K, T] (seat A's reward per turn, float32), acct [K] (exact
accounting residual in ledger units, all 0), and for every window h in WINDOWS the per-continuation sums over the first
h turns of the ledger (c<h>_<key> [K, 2]: farm, plate, bounty, avail, lost_away, died, recalled, returned, away, xpg)
and the state after h turns (s<h>_<key> [K, 2]: xp, level, absent, hp); ret [K, 2] (turns until each seat next comes
back), stock_t [K, 2] (turns until the checkpoint's enemy stock has died on each ledger), stock_got [K, 2] (the share of
that stock collected). With "full": the whole per-turn ledger (f__<key>).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np

from ....diagnostics import lane_exit as LX
from . import common as C

SUM_KEYS = ("farm", "plate", "bounty", "avail", "lost_away", "died", "recalled", "returned", "away", "xpg")
STATE_KEYS = ("xp", "level", "absent", "hp")


# ---------------------------------------------------------------------------------------------- variants
def variant_fresh(cfg, start):
    from ....champions_m4 import context
    lane, m4, k = start
    tb = context(cfg).t
    a = m4.a
    L = tb.level_by_xp[min(a.xp, tb.xp_cap)]
    a2 = a._replace(pos=-cfg.tower, hp=tb.hp_max[L], absent=0, items=a.items + a.gold, gold=0)
    return lane, m4._replace(a=a2), k


def variant_away1(cfg, start):
    lane, m4, k = start
    a = m4.a
    return lane, m4._replace(a=a._replace(pos=-cfg.tower, absent=1)), k


VARIANTS = {"orig": lambda cfg, st: st, "A_fresh": variant_fresh, "A_away1": variant_away1}


# ---------------------------------------------------------------------------------------------- one job
def summarize(led: Dict[str, np.ndarray], stock_hp, cfg, full: bool) -> Dict[str, np.ndarray]:
    out = {"r": led["r"].astype(np.float32), "acct": led["acct_units"].astype(np.int64)}
    lost = (led["avail"] - led["farm"]).astype(np.float64)
    for h in C.WINDOWS:
        w = slice(0, h)
        for key in SUM_KEYS:
            if key == "lost_away":
                v = (lost[:, w] * led["away"][:, w]).sum(1)
            else:
                v = led[key][:, w].astype(np.float64).sum(1)
            out[f"c{h}_{key}"] = v.astype(np.float32)
        for key in STATE_KEYS:
            out[f"s{h}_{key}"] = led[key][:, h - 1].astype(np.float32)
    out["ret"] = LX.return_times(led).astype(np.float32)
    out["stock_t"] = LX.stock_times(led, stock_hp, cfg.farm_per_hp).astype(np.float32)
    out["stock_got"] = LX.stock_collected(led, stock_hp, cfg.farm_per_hp).astype(np.float32)
    if full:
        for key in LX.SIDE_KEYS:
            out[f"f__{key}"] = led[key]
    return out


def job(j: dict) -> tuple:
    """`j`: key, path, start (list form), index, root, K, T, combos, full (bool). Writes an npz (resumable)."""
    path = j["path"]
    if os.path.exists(path):
        return j["key"], {"path": path, "cached": True}
    t0 = time.perf_counter()
    start0 = C.state_from_list(j["start"])
    K, T = int(j["K"]), int(j["T"])
    seeds = [C.roll_seed(j["root"], j["index"], r) for r in range(K)]
    arrays: Dict[str, np.ndarray] = {}
    turns = 0
    for name, cb in j["combos"].items():
        cfg = C.setup(cb["env"])
        st = VARIANTS[cb.get("variant", "orig")](cfg, start0)
        mirror = bool(cb.get("mirror", False))
        if mirror:
            st = LX.mirror_state(st)
        k0 = st[2]
        sa, sb = (1, 0) if mirror else (0, 1)
        led = LX.continuations(cfg, st, lambda sd: (C.crn(cb["a"], k0, sd, sa, T), C.crn(cb["b"], k0, sd, sb, T)),
                               seeds, T, lane_step=C.lane_step(cfg), mirror=mirror)
        d = LX.describe(cfg, st)
        for key, v in summarize(led, (d["stock_A"], d["stock_B"]), cfg, bool(j.get("full"))).items():
            arrays[f"{name}__{key}"] = v
        turns += K * T
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays, combos=np.asarray(list(j["combos"])),
                        elapsed=np.asarray(time.perf_counter() - t0), turns=np.asarray(turns))
    os.replace(tmp, path)
    return j["key"], {"path": path, "elapsed_s": time.perf_counter() - t0, "turns": turns}


def load(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """{combo: {key: array}} plus "_meta"."""
    with np.load(path, allow_pickle=False) as z:
        flat = {k: z[k] for k in z.files}
    out: Dict[str, Dict[str, np.ndarray]] = {"_meta": {"elapsed": float(flat.pop("elapsed")),
                                                       "turns": int(flat.pop("turns"))}}
    names = [str(x) for x in flat.pop("combos")]
    for name in names:
        out[name] = {k.split("__", 1)[1]: v for k, v in flat.items() if k.startswith(name + "__")}
    return out


def run(jobs: List[dict], workers: int, label: str) -> Dict[str, dict]:
    jobs = sorted(jobs, key=lambda x: -int(x["K"]) * len(x["combos"]))
    t0 = time.perf_counter()
    res = C.run_jobs(job, jobs, workers, label)
    print(f"  [{label}] {len(jobs)} jobs in {time.perf_counter() - t0:.0f} s", flush=True)
    return dict(res)


__all__ = ["VARIANTS", "variant_fresh", "variant_away1", "summarize", "job", "load", "run", "SUM_KEYS", "STATE_KEYS"]
