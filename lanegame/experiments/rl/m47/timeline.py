"""Stage 1, descriptive: what happens before level 14 in natural self-play, from EXISTING records (no new games).

Sources (git-ignored, read only):
    M4.6d records   results/rl/checkpoints/m46d/records/<env>/ext_<env>_<s>.npz: the matched M4.6c 8.192M policies
                    ext/baseline:s and ext/C1:s, 48 self-play games of 400 turns each from s0, full per-turn state,
                    action and outcome (m46d.records). Game g of seed s uses the same chance stream and the same
                    sampling uniforms in both environments, so the two arms pair game by game.
    M4.6c natural   results/rl/checkpoints/m46c/eval/natural/<env>/<policy>.json.gz: 48 games of 600 turns of the
                    M4.5c finals (start:s), the 4.096M and the 8.192M continuations; cumulative earned gold, investment,
                    unspent gold, level, presence and HP per turn (m46c.natural). Used for the lineage comparison and
                    as an independent replication of the timeline on other seeds.

Units: turn index i is the start of turn i (turn 0 = s0). A seat "reaches level L at turn i" if its level at the start
of turn i is at least L for the first time. Gold in resources (one fresh wave). Every per-seed number is a mean over
that seed's 48 games and both seats; seeds are the independent units (6), arms are paired by seed (and game).
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Dict

import numpy as np

from ....config import B_RECALL, PRESSURE, VISION
from . import common as C

M46D_REC = os.path.join("results", "rl", "checkpoints", "m46d", "records")
M46C_NAT = os.path.join("results", "rl", "checkpoints", "m46c", "eval", "natural")
LEAD_T = (25, 50, 75, 94, 150, 250, 399)
CONV_PAIRS = ((25, 75), (50, 94), (94, 150), (150, 250))
WAVES = ("last_hit", "shove", "thin", "slow_push")
BEHS = ("hold", "yield", "trade", "safe", "recall")


# ---------------------------------------------------------------------------------------------- loading
def load_records(env: str, s: int) -> Dict[str, np.ndarray]:
    with np.load(os.path.join(M46D_REC, env, f"ext_{env}_{s}.npz")) as z:
        return {k: z[k] for k in z.files}


def load_natural(env: str, policy: str) -> Dict[str, np.ndarray]:
    """M4.6c natural games of a policy file tag (start_s, baseline_s, ext_baseline_s, ...) as arrays [G, T, 2]."""
    with gzip.open(os.path.join(M46C_NAT, env, policy + ".json.gz"), "rt", encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for k in ("E", "S", "g", "L", "P", "hp"):
        out[k] = np.stack([np.asarray(r["arr"][k], dtype=float).T for r in d["records"]])   # [G, T, 2]
    return out


# ---------------------------------------------------------------------------------------------- primitives
def first_turn(L: np.ndarray, lv: int) -> np.ndarray:
    """[G, 2] first turn index at which a seat's level is >= lv (T if never)."""
    m = L >= lv
    T = L.shape[1]
    return np.where(m.any(1), m.argmax(1), T)


def take(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """x [G, T, 2] at per-(game, seat) turn indices t [G, 2] (clipped to the record)."""
    T = x.shape[1]
    tt = np.clip(t, 0, T - 1)
    g = np.arange(x.shape[0])[:, None]
    s = np.arange(2)[None, :]
    return x[g, tt, s]


def dist(v) -> dict:
    v = np.asarray(v, dtype=float).ravel()
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)), "min": float(v.min()),
            "max": float(v.max())}


def _share(mask_num: np.ndarray, mask_den: np.ndarray) -> float:
    d = float(mask_den.sum())
    return float(mask_num.sum()) / d if d > 0 else float("nan")


# ---------------------------------------------------------------------------------------------- per-policy summary
def policy_summary(d: Dict[str, np.ndarray]) -> dict:
    """Every Stage 1 descriptive of one policy's records (one seed)."""
    L, T = d["L"], d["L"].shape[1]
    E = np.cumsum(d["earned"], 1) - d["earned"]                 # earned before turn i (start-of-turn convention)
    out: dict = {"games": int(L.shape[0]), "turns": int(T)}
    # --- timeline: turns to levels, by seat, first seat, both seats
    tl = {}
    for lv in C.LEVEL_MARKS:
        t = first_turn(L, lv)
        tl[str(lv)] = {"seat_A": dist(t[:, 0]), "seat_B": dist(t[:, 1]), "pooled": dist(t),
                       "first_seat": dist(t.min(1)), "both_seats": dist(t.max(1)),
                       "seat_gap_abs": dist(np.abs(t[:, 0] - t[:, 1])), "reached_share": float((t < T).mean())}
    out["timeline"] = tl
    # --- the state when each seat reaches a level
    at = {}
    cd, cr, ci = (np.cumsum(d[k], 1) - d[k] for k in ("died", "recalled", "intr"))
    for lv in C.LEVEL_MARKS:
        t = first_turn(L, lv)
        at[str(lv)] = {"turn": float(t.mean()), "invested": float(take(d["S"], t).mean()),
                       "unspent": float(take(d["g"], t).mean()), "earned": float(take(E, t).mean()),
                       "hp": float(take(d["hp"], t).mean()), "deaths": float(take(cd, t).mean()),
                       "recalls": float(take(cr, t).mean()), "interrupted": float(take(ci, t).mean()),
                       "earned_lead_abs": float(np.abs(take(E, t)[:, 0] - take(E, t)[:, 1]).mean())}
    out["at_level"] = at
    # --- common elapsed-turn windows
    present = d["absent"] <= 0
    wave_t = d["wave"] >= 0
    beh_t = d["beh"] >= 0
    held_on = d["held"][:, :, ::-1]                              # the opponent holds a freeze on the seat (after)
    held0_on = d["held0"][:, :, ::-1]                            # ... at the start of the turn
    win = {}
    for name, (lo, hi) in C.WINDOWS.items():
        hi = min(hi, T)
        sl = slice(lo, hi)
        n = (hi - lo) * L.shape[0] * 2
        w = {"turns": hi - lo,
             "earned_per_turn": float(d["earned"][:, sl].sum() / n),
             "farm_per_turn": float(d["farm"][:, sl].sum() / n),
             "farm_avail_per_turn": float(d["avail"][:, sl].sum() / n),
             "bounty_per_turn": float(d["bounty"][:, sl].sum() / n),
             "plate_per_turn": float(d["plate"][:, sl].sum() / n),
             "deaths_per_100": float(100 * d["died"][:, sl].sum() / n),
             "recalls_per_100": float(100 * d["recalled"][:, sl].sum() / n),
             "interrupted_per_100": float(100 * d["intr"][:, sl].sum() / n),
             "recall_choices_per_100": float(100 * (d["beh"][:, sl] == B_RECALL).sum() / n),
             "trade_connect_per_100": float(100 * d["conn"][:, sl].sum() / n),
             "hp_present": float(d["hp"][:, sl][present[:, sl]].mean()),
             "away_share": float(1 - present[:, sl].mean()),
             "level": float(L[:, sl].mean()), "invested": float(d["S"][:, sl].mean()),
             "unspent": float(d["g"][:, sl].mean()),
             "ward_share": _share(d["obj"][:, sl] == VISION, present[:, sl]),
             "pressure_share": _share(d["obj"][:, sl] == PRESSURE, present[:, sl]),
             "held_share": float(d["held"][:, sl].mean()), "frozen_on_share": float(held_on[:, sl].mean()),
             "crash_hp_per_turn": float(d["crash"][:, sl].sum() / n),
             "damage_taken_per_turn": float(d["taken"][:, sl].sum() / n)}
        for j, nm in enumerate(WAVES):
            w[f"wave_{nm}"] = _share((d["wave"][:, sl] == j) & wave_t[:, sl], wave_t[:, sl])
        for j, nm in enumerate(BEHS):
            w[f"beh_{nm}"] = _share((d["beh"][:, sl] == j) & beh_t[:, sl], beh_t[:, sl])
        # responses when frozen on at the start of the turn (present): break (Shove), concede (Safe), walk-up (Hold)
        fo = (held0_on[:, sl] > 0) & present[:, sl] & wave_t[:, sl]
        w["frozen_on_turns_share"] = float(fo.mean())
        w["frozen_on_shove"] = _share(fo & (d["wave"][:, sl] == 1), fo)
        w["frozen_on_safe"] = _share(fo & (d["beh"][:, sl] == 3), fo)
        w["frozen_on_hold"] = _share(fo & (d["beh"][:, sl] == 0), fo)
        win[name] = w
    out["windows"] = win
    # --- level bands (per seat): descriptive only, the bands have different lengths
    bands = {"L<10": (0, 10), "L10-11": (10, 12), "L12-13": (12, 14), "L>=14": (14, 99)}
    bd = {}
    for name, (a, b) in bands.items():
        m = (L >= a) & (L < b)
        bd[name] = {"seat_turn_share": float(m.mean()),
                    "earned_per_turn": float(d["earned"][m].mean()) if m.any() else float("nan"),
                    "shove": _share(m & (d["wave"] == 1), m & wave_t),
                    "trade": _share(m & (d["beh"] == 2), m & beh_t),
                    "recall_choice": _share(m & (d["beh"] == B_RECALL), m & beh_t),
                    "deaths_per_100": float(100 * d["died"][m].mean()) if m.any() else float("nan")}
    out["level_bands"] = bd
    # --- Recall decisions: HP and gold when Recall is chosen, outcome, phase
    rc = d["beh"] == B_RECALL
    tt = np.broadcast_to(np.arange(T)[None, :, None], rc.shape)
    rec = {}
    for ph, m in (("lane", tt < C.LANE_END), ("after", tt >= C.LANE_END)):
        sel = rc & m
        k = int(sel.sum())
        rec[ph] = {"choices": k, "per_game_seat": k / (L.shape[0] * 2),
                   "hp_at_choice": dist(d["hp"][sel]) if k > 1 else None,
                   "unspent_at_choice": dist(d["g"][sel]) if k > 1 else None,
                   "completed_share": float(d["recalled"][sel].mean()) if k else float("nan"),
                   "interrupted_share": float(d["intr"][sel].mean()) if k else float("nan")}
    first_rec = np.where((d["recalled"] > 0).any(1), (d["recalled"] > 0).argmax(1), T)
    rec["first_completed_turn"] = dist(first_rec)
    out["recall"] = rec
    # --- investment timing and unspent gold
    inv = {}
    for t in (50, 94, 150, 250, T - 1):
        inv[str(t)] = {"invested": float(d["S"][:, t].mean()), "unspent": float(d["g"][:, t].mean()),
                       "earned": float(E[:, t].mean())}
    t14 = first_turn(L, 14)
    inv["at_L14"] = {"invested": float(take(d["S"], t14).mean()), "unspent": float(take(d["g"], t14).mean()),
                     "invested_share_of_final": float((take(d["S"], t14) / np.maximum(d["S"][:, -1], 1e-9)).mean())}
    out["investment"] = inv
    # --- gold leads: formation and conversion (seat A's earned-gold lead; zero-sum)
    lead = E[:, :, 0] - E[:, :, 1]
    ilead = d["S"][:, :, 0] - d["S"][:, :, 1]
    ld = {}
    for t in LEAD_T:
        tot = E[:, t].sum(1) / 2
        ld[str(t)] = {"abs_mean": float(np.abs(lead[:, t]).mean()), "sd": float(lead[:, t].std(ddof=1)),
                      "abs_p90": float(np.percentile(np.abs(lead[:, t]), 90)),
                      "rel_abs_mean": float((np.abs(lead[:, t]) / np.maximum(tot, 1e-9)).mean()),
                      "invested_lead_sd": float(ilead[:, t].std(ddof=1))}
    conv = {}
    for a, b in CONV_PAIRS:
        x, y = lead[:, a], lead[:, b] - lead[:, a]
        vx = float(np.var(x, ddof=1))
        conv[f"{a}-{b}"] = {"slope": float(np.cov(x, y, ddof=1)[0, 1] / vx) if vx > 0 else float("nan"),
                            "corr": float(np.corrcoef(x, y)[0, 1]) if vx > 0 else float("nan")}
    ld["conversion"] = conv
    ld["var_share_formed_by_94"] = float(np.var(lead[:, 94], ddof=1) / np.var(lead[:, T - 1], ddof=1))
    out["leads"] = ld
    # --- freeze episodes (held runs) by phase of their first turn
    fz = {"lane": [], "after": []}
    for g in range(L.shape[0]):
        for s in (0, 1):
            h = d["held"][g, :, s] > 0
            i = 0
            while i < T:
                if h[i]:
                    j = i
                    while j < T and h[j]:
                        j += 1
                    fz["lane" if i < C.LANE_END else "after"].append(j - i)
                    i = j
                else:
                    i += 1
    out["freezes"] = {ph: {"episodes_per_game_seat": len(v) / (L.shape[0] * 2),
                           "mean_length": float(np.mean(v)) if v else float("nan"),
                           "long_share_ge3": float(np.mean(np.asarray(v) >= 3)) if v else float("nan")}
                      for ph, v in fz.items()}
    return out


def natural_summary(d: Dict[str, np.ndarray]) -> dict:
    """The timeline part of `policy_summary` for an M4.6c natural record (cumulative E, S, g, L, hp only)."""
    L = d["L"]
    out = {"timeline": {}, "at_level": {}}
    for lv in C.LEVEL_MARKS:
        t = first_turn(L, lv)
        out["timeline"][str(lv)] = {"pooled": dist(t), "first_seat": dist(t.min(1)), "both_seats": dist(t.max(1))}
        out["at_level"][str(lv)] = {"invested": float(take(d["S"], t).mean()), "unspent": float(take(d["g"], t).mean()),
                                    "earned": float(take(d["E"], t).mean())}
    return out


# ---------------------------------------------------------------------------------------------- aggregation
def _flatten(d: dict, prefix: str = "") -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "/"))
        elif isinstance(v, (int, float, np.floating, np.integer)) and v is not None:
            out[key] = float(v)
    return out


def aggregate(per_seed: Dict[str, Dict[int, dict]]) -> dict:
    """Seed statistics of every flattened scalar, per environment, and the paired difference C1 - baseline."""
    flat = {env: {s: _flatten(v) for s, v in d.items()} for env, d in per_seed.items()}
    keys = sorted(set.intersection(*[set(f[min(f)]) for f in flat.values()]))
    out = {"by_env": {}, "diff_C1_minus_baseline": {}}
    for env, f in flat.items():
        out["by_env"][env] = {k: C.seed_stats({s: f[s][k] for s in f}) for k in keys}
    if set(flat) >= {"baseline", "C1"}:
        b, c = flat["baseline"], flat["C1"]
        common = sorted(set(b) & set(c))
        out["diff_C1_minus_baseline"] = {k: C.seed_stats({s: c[s][k] - b[s][k] for s in common}) for k in keys}
    return out


def run() -> dict:
    per_seed = {env: {} for env in C.ENVS}
    for env in C.ENVS:
        for s in C.SEEDS:
            per_seed[env][s] = policy_summary(load_records(env, s))
            print(f"  [timeline] {env} s{s}: L14 at {per_seed[env][s]['timeline']['14']['pooled']['mean']:.1f}",
                  flush=True)
    lineage = {}
    for env, tags in (("baseline", ("start", "baseline", "ext_baseline")), ("C1", ("C1", "ext_C1"))):
        for tag in tags:
            rows = {}
            for s in C.SEEDS:
                try:
                    rows[s] = natural_summary(load_natural(env, f"{tag}_{s}"))
                except FileNotFoundError:
                    continue
            if rows:
                lineage[f"{env}/{tag}"] = {"per_seed": rows,
                                           "stats": {k: C.seed_stats({s: _flatten(r)[k] for s, r in rows.items()})
                                                     for k in _flatten(rows[min(rows)])}}
    data = {"meta": C.meta(stage="timeline"), "sources": {"records": M46D_REC, "natural": M46C_NAT},
            "per_seed": per_seed, "aggregate": aggregate(per_seed), "lineage": lineage,
            "windows": C.WINDOWS, "lane_end": C.LANE_END}
    C.save_json("timeline", data)
    return data


__all__ = ["load_records", "load_natural", "first_turn", "take", "policy_summary", "natural_summary", "aggregate",
           "run", "LEAD_T"]
