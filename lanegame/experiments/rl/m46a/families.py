"""Families of paired jobs (E2, E5): job lists, the validation subset, the pooled validation rule and the per-policy
aggregation (pre-registration §1).

A family = (name, kind "natural" | "semi" | "constructed", per-final states, candidates, protocols, K, horizon). Every
(final, state, protocol) is one `engine.paired_job`; the validation subset's "policy" jobs run to VALIDATION_H instead
of the family horizon (their first h turns are the h-turn estimate itself, the same rollouts).

Aggregation: per final, the mean over its states of the per-state paired mean (candidate - baseline) of the primary
statistic at the family horizon; across the six finals a t interval; k/6 = finals whose own interval (states as units)
excludes 0 in that direction.

Validation (pooled over the finals, per family): over the validation (state, candidate) pairs, the bias
mean(Delta_300 - Delta_h) within max(0.15, 2 SE), the slope of Delta_300 on Delta_h in [0.8, 1.25], and no
contradiction (both significant with opposite signs, states' K rollouts as units).
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import engine as EN
from .common import CKPT, FINALS, VAL_ROOT, mean_ci, run_jobs, significant, spec

VALIDATION_H, EXTENDED_H = 300, 150
VAL_SHARE, VAL_MIN = 0.15, 4


def validation_indices(fam: str, n: int, minimum: int = VAL_MIN) -> List[int]:
    if n <= 0:
        return []
    m = min(n, max(minimum, math.ceil(VAL_SHARE * n)))
    rng = np.random.default_rng([VAL_ROOT, sum(map(ord, fam))])
    return sorted(int(i) for i in rng.choice(n, size=m, replace=False))


def job_path(stage: str, fam: str, final: int, idx: int, proto: str, horizon: int) -> str:
    return os.path.join(CKPT, stage, fam, f"s{final}", f"{idx}__{proto}__h{horizon}.json.gz")


def build_jobs(stage: str, fam: str, states: Dict[int, list], cands: dict, protos: Sequence[str], K: int, horizon: int,
               roll_root: int, val: Optional[Dict[int, List[int]]] = None, critic: bool = True,
               extended: bool = False) -> list:
    """states: {final: [start, ...]}; val: {final: [indices]} of the validation subset."""
    out = []
    roll = EN.roll_seeds(roll_root, K)
    for s, sts in states.items():
        vs = set((val or {}).get(s, []))
        for i, st in enumerate(sts):
            for p in protos:
                h = EXTENDED_H if extended else (VALIDATION_H if (i in vs and p == "policy") else horizon)
                out.append({"spec": spec(s), "family": fam, "idx": i, "start": st, "proto": p, "horizon": h,
                            "roll": roll, "cands": cands, "critic": critic, "tag": f"s{s}",
                            "path": job_path(stage, fam, s, i, p, h), "label": f"{fam} state {i}"})
    return out


def run(jobs: list, workers: int, label: str) -> Dict[str, dict]:
    t0 = time.perf_counter()
    res = run_jobs(EN.paired_job, jobs, workers, label)
    print(f"  [{label}] {len(jobs)} jobs in {time.perf_counter() - t0:.0f} s", flush=True)
    return {k: v for k, v in res}


def records(res: Dict[str, dict], fam: str, proto: str) -> Dict[int, List[dict]]:
    """{final: [record by state index]} for one family and protocol (any horizon)."""
    out: Dict[int, Dict[int, dict]] = {}
    for rec in res.values():
        if rec["family"] != fam or rec["proto"] != proto:
            continue
        s = int(rec["tag"][1:])
        out.setdefault(s, {})[rec["idx"]] = rec
    return {s: [d[i] for i in sorted(d)] for s, d in out.items()}


def contrast(res: Dict[str, dict], fam: str, proto: str, cand: str, stat: str, h: int, base: Optional[str] = None,
             select=None) -> dict:
    """Per-final means and the cross-final t interval of a paired contrast; `select(rec) -> bool` filters states."""
    per, k_pos, k_neg = {}, 0, 0
    for s, recs in records(res, fam, proto).items():
        vals = [EN.per_state_diff(r, cand, stat, h, base) for r in recs if (select is None or select(r))]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        m, hw = mean_ci(vals)
        per[s] = {"mean": m, "ci": hw, "n": len(vals)}
        sg = significant(m, hw)
        k_pos += sg > 0
        k_neg += sg < 0
    m, hw = mean_ci([v["mean"] for v in per.values()])
    return {"mean": m, "ci": hw, "per_final": per, "k_pos": k_pos, "k_neg": k_neg, "finals": len(per)}


def ledger_contrast(res: Dict[str, dict], fam: str, proto: str, cand: str, key: str, h: int, select=None) -> dict:
    per = {}
    for s, recs in records(res, fam, proto).items():
        vals = [EN.ledger_diff(r, cand, key, h) for r in recs if (select is None or select(r))]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            per[s] = float(np.mean(vals))
    m, hw = mean_ci(list(per.values()))
    return {"mean": m, "ci": hw, "per_final": per}


def absolute(res: Dict[str, dict], fam: str, proto: str, cand: str, stat: str, h: int, select=None) -> dict:
    per = {}
    for s, recs in records(res, fam, proto).items():
        vals = [EN.per_state_mean(r, cand, stat, h) for r in recs if (select is None or select(r))]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            per[s] = float(np.mean(vals))
    m, hw = mean_ci(list(per.values()))
    return {"mean": m, "ci": hw, "per_final": per}


def validate(res: Dict[str, dict], fam: str, stat: str, h: int, cands: Sequence[str]) -> dict:
    """The pooled validation rule (module docstring) on the family's 300-turn 'policy' jobs."""
    pairs = []
    for rec in res.values():
        if rec["family"] != fam or rec["proto"] != "policy" or rec["horizon"] != VALIDATION_H:
            continue
        src = rec[stat] if rec[stat] is not None else rec["G"]
        base = rec["baseline"]
        for c in cands:
            if c == base:
                continue
            a = np.asarray(src[c][str(h)]) - np.asarray(src[base][str(h)])
            b = np.asarray(src[c][str(VALIDATION_H)]) - np.asarray(src[base][str(VALIDATION_H)])
            pairs.append((a, b))
    if not pairs:
        return {"pairs": 0, "verdict": "no validation units"}
    dh = np.array([a.mean() for a, _ in pairs])
    d3 = np.array([b.mean() for _, b in pairs])
    diff = d3 - dh
    bias, se = float(diff.mean()), float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    slope = float(np.polyfit(dh, d3, 1)[0]) if len(dh) > 2 and dh.std() > 0 else float("nan")
    contra = 0
    for a, b in pairs:
        ma, ha = mean_ci(a)
        mb, hb = mean_ci(b)
        if significant(ma, ha) * significant(mb, hb) < 0:
            contra += 1
    ok = abs(bias) <= max(0.15, 2 * se) and (0.8 <= slope <= 1.25) and contra == 0
    return {"pairs": len(pairs), "bias": bias, "bias_se": se, "slope": slope, "contradictions": contra,
            "verdict": "pass" if ok else "fail"}


__all__ = ["validation_indices", "build_jobs", "run", "records", "contrast", "ledger_contrast", "absolute",
           "validate", "VALIDATION_H", "EXTENDED_H"]
