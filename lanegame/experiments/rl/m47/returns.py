"""Stage 1: discounted returns of EXISTING natural trajectories, recomputed at candidate horizons (no new games, no
critic).

The M4.6d records store seat A's reward (earned-gold lead) for every turn of 48 self-play games of 400 turns per
policy. From a state at turn t0 the realized return under discount gamma is G = sum_{i >= t0} gamma^(i - t0) r_i,
truncated at the record's end (the omitted tail is at most gamma^(400 - t0) Omega; reported). It splits exactly into the
part before the phase end (turn LANE_END, about level 14) and the part after it.

This is a REWEIGHTING of unchanged behaviour: the policies were trained for gamma 0.99, and the numbers say how the same
trajectories score under shorter objectives, not what agents trained for them would do.

Two readings:
    variance   how much of the variation of the realized return from an early state comes from rewards after the
               phase, and how the two parts covary (a positive covariance means early-phase gains continue later)
    advantage  STATE VALUE of natural advantages (descriptive regression, seat A's perspective): the realized return's
               dependence on the invested lead, the unspent-gold lead, the level difference and the HP difference at
               t0, split before / after the phase end. Observational, not a counterfactual (delayed.py has those)
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from . import common as C
from .timeline import load_records

T0S = (0, 10, 25, 50, 75)
REG_T0S = (25, 50, 75)


def split_returns(r: np.ndarray, t0: int, gamma: float, end: int = C.LANE_END):
    """r [G, T] -> (total, pre, post) realized returns from t0 (pre: turns t0..end-1, post: end..T-1)."""
    T = r.shape[1]
    w = gamma ** np.arange(T - t0, dtype=float)
    x = r[:, t0:] * w
    cut = max(0, end - t0)
    return x.sum(1), x[:, :cut].sum(1), x[:, cut:].sum(1)


def _features(d: Dict[str, np.ndarray], t0: int) -> np.ndarray:
    hp = np.where(d["absent"][:, t0] > 0, 1.0, d["hp"][:, t0])
    return np.column_stack([d["S"][:, t0, 0] - d["S"][:, t0, 1], d["g"][:, t0, 0] - d["g"][:, t0, 1],
                            d["L"][:, t0, 0] - d["L"][:, t0, 1], hp[:, 0] - hp[:, 1]])


FEATURES = ("invested_lead", "unspent_lead", "level_diff", "hp_diff")


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta[1:]


def policy_returns(d: Dict[str, np.ndarray]) -> dict:
    r = d["r"]
    out = {"variance": {}, "advantage": {}}
    for h, g in C.HORIZONS.items():
        rows = {}
        for t0 in T0S:
            tot, pre, post = split_returns(r, t0, g)
            vt = float(np.var(tot, ddof=1))
            rows[str(t0)] = {"sd_total": float(np.sqrt(vt)), "sd_pre": float(pre.std(ddof=1)),
                             "sd_post": float(post.std(ddof=1)),
                             "var_share_post": float(np.var(post, ddof=1) / vt) if vt > 0 else float("nan"),
                             "cov_share": float(2 * np.cov(pre, post, ddof=1)[0, 1] / vt) if vt > 0 else float("nan"),
                             "corr_pre_post": float(np.corrcoef(pre, post)[0, 1]),
                             "tail_weight_beyond_record": float(g ** (r.shape[1] - t0))}
        out["variance"][h] = rows
    for t0 in REG_T0S:
        X = _features(d, t0)
        row = {}
        for h, g in C.HORIZONS.items():
            tot, pre, post = split_returns(r, t0, g)
            row[h] = {"total": dict(zip(FEATURES, _ols(X, tot).tolist())),
                      "pre": dict(zip(FEATURES, _ols(X, pre).tolist())),
                      "post": dict(zip(FEATURES, _ols(X, post).tolist()))}
        row["feature_sd"] = dict(zip(FEATURES, X.std(0, ddof=1).tolist()))
        out["advantage"][str(t0)] = row
    return out


def run() -> dict:
    from .timeline import aggregate
    per_seed = {env: {} for env in C.ENVS}
    for env in C.ENVS:
        for s in C.SEEDS:
            per_seed[env][s] = policy_returns(load_records(env, s))
    data = {"meta": C.meta(stage="returns"), "per_seed": per_seed, "aggregate": aggregate(per_seed),
            "horizons": C.HORIZONS, "lane_end": C.LANE_END, "features": FEATURES}
    C.save_json("returns", data)
    return data


__all__ = ["split_returns", "policy_returns", "run", "T0S", "REG_T0S", "FEATURES"]
