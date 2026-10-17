"""Stage 1: how much of ordinary training lies before levels 10, 12 and 14, and how much objective weight does.

Training occupancy is RECONSTRUCTED from M4.6e's existing exposure games (results/rl/checkpoints/m46e/exposure/; no new
games): frozen M4.6c policies playing themselves from the M4.6c training start distribution, 48 s0 games of 512 turns
and 192 archive games of 150 turns (half mirrored), weighted to the measured archive transition share 0.25
(m46e.exposure). It is an estimate: final-policy rollouts do not reproduce a run's history exactly, and M4.6e's check
across snapshots (1.0M, 4.1M, 8.192M) bounds the drift. Each seat-turn is one training transition (both seats are the
shared network's samples). The M4.5c from-scratch campaign trained from s0 only; its occupancy is the s0 column.

Objective weight is exact arithmetic: under discount gamma the share of the objective's weight from s0 on turns before
t is 1 - gamma^t, and for a decision at turn k the share before the phase end T is 1 - gamma^(T - k).
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from . import common as C

EXPO = os.path.join("results", "rl", "checkpoints", "m46e", "exposure")
ARCHIVE_SHARE = 0.25
SPECS = ("ext_baseline_0", "ext_baseline_1", "ext_baseline_2", "ext_C1_0", "ext_C1_1", "ext_C1_2",
         "ext_C1_0_1024000", "ext_C1_0_4096000")


def load_tag(tag: str) -> Tuple[List[Tuple[str, np.ndarray]], List[np.ndarray]]:
    """(archive games with labels, s0 games) of one reconstructed policy; arrays [T, 2, 9], column 0 = level."""
    arch, s0 = [], []
    d = os.path.join(EXPO, tag)
    for fn in sorted(os.listdir(d)):
        with np.load(os.path.join(d, fn)) as z:
            for j, lb in enumerate(z["labels"]):
                x = z[f"g{j}"]
                if fn.startswith("s0"):
                    s0.append(x)
                else:
                    arch.append((str(lb), x))
    return arch, s0


def occupancy(tag: str) -> dict:
    arch, s0 = load_tag(tag)
    La = np.concatenate([x[:, :, 0] for _, x in arch])            # [turns, 2]
    Ls = np.concatenate([x[:, :, 0] for x in s0])
    out = {"archive_turns": int(len(La)), "s0_turns": int(len(Ls)), "below": {}, "turn_below": {}}
    for lv in (8, 10, 12, 14, 16):
        pa, ps = float((La < lv).mean()), float((Ls < lv).mean())
        out["below"][str(lv)] = {"archive": pa, "s0": ps, "training": ARCHIVE_SHARE * pa + (1 - ARCHIVE_SHARE) * ps}
    T = s0[0].shape[0]
    for t in (50, 75, C.LANE_END, 150):
        out["turn_below"][str(t)] = {"s0": t / T, "training_s0_part": (1 - ARCHIVE_SHARE) * t / T}
    starts = np.asarray([x[0, :, 0].min() for _, x in arch])
    out["archive_start_level"] = {"p10": float(np.percentile(starts, 10)), "median": float(np.median(starts)),
                                  "p90": float(np.percentile(starts, 90)),
                                  "below14_share": float((starts < 14).mean())}
    by = {}
    for lb in sorted(set(l for l, _ in arch)):
        xx = np.concatenate([x[:, :, 0] for l, x in arch if l == lb])
        by[lb] = {"below14": float((xx < 14).mean()), "games": sum(1 for l, _ in arch if l == lb)}
    out["archive_by_label"] = by
    return out


def objective_weights(t_marks: Dict[str, float]) -> dict:
    """Share of the objective's weight from s0 before each natural level time, and the average share before the phase
    end for decisions taken inside the phase, for every candidate horizon."""
    out = {}
    for h, g in C.HORIZONS.items():
        row = {"gamma": g, "expected_duration": 1 / (1 - g), "median_duration": float(np.log(0.5) / np.log(g)),
               "before": {k: C.weight_before(t, g) for k, t in t_marks.items()},
               "beyond_300": float(g ** 300)}
        row["in_phase_decisions_before_end"] = C.mean_weight_before_end(int(round(t_marks["L14"])), g)
        out[h] = row
    return out


def run() -> dict:
    occ = {}
    for tag in SPECS:
        if os.path.isdir(os.path.join(EXPO, tag)):
            occ[tag] = occupancy(tag)
            print(f"  [occupancy] {tag}: below L14 {occ[tag]['below']['14']['training']:.3f}", flush=True)
    tl = C.load("timeline")
    marks = {"L10": 51.0, "L12": 71.0, "L14": float(C.LANE_END), "L18": 150.0}
    if tl:
        b = tl["aggregate"]["by_env"]["baseline"]
        marks = {f"L{lv}": b[f"timeline/{lv}/pooled/mean"]["mean"] for lv in (10, 12, 14, 18)}
    data = {"meta": C.meta(stage="occupancy"), "source": EXPO, "archive_share": ARCHIVE_SHARE, "occupancy": occ,
            "level_marks": marks, "objective_weights": objective_weights(marks)}
    C.save_json("occupancy", data)
    return data


__all__ = ["occupancy", "objective_weights", "run"]
