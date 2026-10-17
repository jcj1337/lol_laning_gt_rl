"""Per-game behaviour metrics of a self-play record (m46d.records format), in common elapsed-turn windows.

Every metric is computed per game (both seats averaged: self-play is symmetric), so a policy's value is a mean over
games with a game-level standard error, and two policies recorded on the same seeds pair game by game. Windows:
"lane" = turns 0-93 (the natural median before level 14), "after" = turns 94 to the record's end.

The materiality thresholds of Stage 2 (`thresholds`) are the across-seed standard deviations of these per-seed means
for the six independently trained M4.6c baseline policies (Stage 1 records), rounded up: a change smaller than the
difference between two separately trained baseline seeds is not called a meaningful change of behaviour.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ....config import B_RECALL, PRESSURE, VISION
from . import common as C

WINDOWS = {"lane": (0, C.LANE_END), "after": (C.LANE_END, None)}
METRICS = ("earned_per_turn", "recall_choices_per_100", "recalls_per_100", "hp_at_recall", "unspent_at_recall",
           "shove", "last_hit", "thin", "slow_push", "hold", "yield", "trade", "safe", "deaths_per_100", "held_share",
           "frozen_on_shove", "frozen_on_safe", "ward_share", "pressure_share", "plates", "invested_end",
           "unspent_end", "hp_present")


def _nanmean(x: np.ndarray, m: np.ndarray, axis=(1, 2)) -> np.ndarray:
    s = np.where(m, x, 0.0).sum(axis)
    n = m.sum(axis)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def per_game(d: Dict[str, np.ndarray], window: str) -> Dict[str, np.ndarray]:
    """Metric name -> [G] values of one record dict (arrays [G, T, 2])."""
    lo, hi = WINDOWS[window]
    T = d["L"].shape[1]
    hi = T if hi is None else min(hi, T)
    sl = slice(lo, hi)
    x = {k: v[:, sl] for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim >= 3 and v.shape[1] == T}
    n = (hi - lo) * 2
    present = x["absent"] <= 0
    wave_t = x["wave"] >= 0
    beh_t = x["beh"] >= 0
    rc = x["beh"] == B_RECALL
    fo = (x["held0"][:, :, ::-1] > 0) & present & wave_t
    out = {"earned_per_turn": x["earned"].sum((1, 2)) / n,
           "recall_choices_per_100": 100 * rc.sum((1, 2)) / n,
           "recalls_per_100": 100 * x["recalled"].sum((1, 2)) / n,
           "hp_at_recall": _nanmean(x["hp"], rc), "unspent_at_recall": _nanmean(x["g"], rc),
           "deaths_per_100": 100 * x["died"].sum((1, 2)) / n,
           "held_share": x["held"].mean((1, 2)),
           "frozen_on_shove": _nanmean((x["wave"] == 1).astype(float), fo),
           "frozen_on_safe": _nanmean((x["beh"] == 3).astype(float), fo),
           "ward_share": _nanmean((x["obj"] == VISION).astype(float), present),
           "pressure_share": _nanmean((x["obj"] == PRESSURE).astype(float), present),
           "plates": (x["plate"] > 0).sum((1, 2)) / 2,
           "invested_end": x["S"][:, -1].mean(1), "unspent_end": x["g"][:, -1].mean(1),
           "hp_present": _nanmean(x["hp"], present)}
    for j, nm in enumerate(("last_hit", "shove", "thin", "slow_push")):
        out[nm] = _nanmean((x["wave"] == j).astype(float), wave_t)
    for j, nm in enumerate(("hold", "yield", "trade", "safe")):
        out[nm] = _nanmean((x["beh"] == j).astype(float), beh_t)
    return out


def summary(d: Dict[str, np.ndarray]) -> dict:
    """{window: {metric: mean_se dict}} of one record."""
    return {w: {k: C.mean_se(v) for k, v in per_game(d, w).items()} for w in WINDOWS}


def paired(da: Dict[str, np.ndarray], db: Dict[str, np.ndarray]) -> dict:
    """{window: {metric: mean_se of the per-game difference a - b}} for records on the same game seeds."""
    out = {}
    for w in WINDOWS:
        pa, pb = per_game(da, w), per_game(db, w)
        out[w] = {k: C.mean_se(pa[k] - pb[k]) for k in pa}
    return out


def thresholds() -> dict:
    """Across-seed SD of the six M4.6c baseline policies' per-seed means (Stage 1 records), per window and metric."""
    from .timeline import load_records
    per = {w: {k: [] for k in METRICS} for w in WINDOWS}
    for s in C.SEEDS:
        d = load_records("baseline", s)
        for w in WINDOWS:
            for k, v in per_game(d, w).items():
                per[w][k].append(float(np.nanmean(v)))
    return {w: {k: float(np.std(v, ddof=1)) for k, v in m.items()} for w, m in per.items()}


__all__ = ["WINDOWS", "METRICS", "per_game", "summary", "paired", "thresholds"]
