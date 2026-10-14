"""Paired multi-turn counterfactual evaluation from a saved state (M4.4c Part 1.2-1.4).

Protocol for one state (a `diagnostics.Snapshot` reached in a recorded game, or an initialized state), focal seat f:

  1. candidates   named plans (`plans.StrategicPlan`, or `diagnostics.branch.FollowPolicy` for "the learned policy"),
                  each only where its branch-turn decision is legal (`StrategicPlan.applicable`)
  2. opponent     simultaneous. It samples its own continuation agent from its own observation with its own stream
                  (identical for every candidate: protocol "policy"), or follows an opponent plan on top of it (protocol
                  "modal": its most probable action on the branch turn; or a style plan); the opponent never sees the
                  focal side's current action
  3. continuation after a plan hands back control, both seats follow the stated continuation agents (default: the policy
                  itself in both seats)
  4. pairing      rollout m of every candidate uses the same three streams (`diagnostics.branch.rollout_seed(root, m,
                  *key)`: environment chance, focal sampling, opponent sampling), one uniform per seat per turn: common
                  random numbers until the trajectories diverge. The environment's chance vector of every turn is
                  action-independent, so it stays identical across candidates for the whole window (checked); the
                  agents' later choices are NOT forced to match (their observations differ once the states do)
  5. outcomes     per rollout, from the recorded turn infos: the focal discounted return over several windows
                  (G_h = sum_{t<h} 0.99^t r_t, h = 5 ... 300; 300 is the M4.4b window) and episode ledgers over the first
                  turns (gold, XP, XP lost for good, uncollected gold, HP lost, deaths, Recalls, crashes, plates, tower
                  damage, ganks, trades, Pressure / Vision turns, investment, freeze state) for both sides

Uncertainty is reported at two levels and never mixed: within one network, paired differences over rollouts give a
per-state standard error and the states are the units of the across-state 95% t interval; across the training seeds
(the replicates of the training procedure) the three per-seed means give a t interval with 2 degrees of freedom.

What it is not: the value of a plan UNDER THE STATED CONTINUATION AND OPPONENT POLICIES; not a best response, an
equilibrium action, or optimal in any game-theoretic sense.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....champions_m4 import context
from ....config import PRESSURE
from ....diagnostics.branch import FollowPolicy, Plan, ProbCache, Snapshot, rollout, rollout_seed
from ....env import LaneEnv
from ....state import State
from ..m44.records import holds_freeze
from .common import t_ci

RET_H = (5, 10, 20, 50, 100, 300)
EP_H = (3, 10, 40)
WINDOW = 300
SIDE_KEYS = ("gold", "xp", "xplost", "unc", "dmg", "died", "rec", "intr", "crash", "plates", "tower", "gankp", "ganked",
             "gankdmg", "tatt", "tconn", "tdealt", "press", "visset", "visact", "invest", "held", "away")
ANY_KEYS = ("died", "rec")            # reported as "happened within h turns" (0 / 1) instead of a count
EVENT_KEYS = ("t_rec_f", "t_left_f", "t_death_f", "t_death_o", "freeze_len", "first_hp_f")


# ---------------------------------------------------------------------------------------------- one rollout
def summarize(infos: List[dict], f: int, cfg, ep_h: Sequence[int] = EP_H, ret_h: Sequence[int] = RET_H) -> Dict[str, float]:
    """Flat outcome dict of one rollout (module docstring, step 5); `f` = focal seat, suffix _f focal, _o opponent."""
    ctx = context(cfg)
    hpmax = ctx.t.hp_max
    sc = cfg.gold_scale
    o = 1 - f
    T = len(infos)
    sign = 1.0 if f == 0 else -1.0
    r = np.fromiter((inf["reward"] for inf in infos), float, T) * sign
    cr = np.cumsum(r * cfg.gamma ** np.arange(T))
    out = {f"G{h}": float(cr[min(h, T) - 1]) for h in ret_h}
    out["U300"] = float(r.sum())
    E = min(max(ep_h), T)
    cols = {k: np.zeros((E, 2)) for k in SIDE_KEYS}
    for t in range(E):
        inf = infos[t]
        lb = inf["levels_before"]
        before = State.from_list(inf["state_before"])
        after = State.from_list(inf["state_after"])
        for s in (0, 1):
            e = 1 - s
            row = cols
            row["gold"][t, s] = inf["earned_units"][s] / sc
            row["xp"][t, s] = inf["xp_gain"][s]
            avail = cfg.xp_per_hp * sum(inf["kills"][s].values())
            row["xplost"][t, s] = avail - inf["xp_gain"][s] - inf["xp_discarded"][s]
            row["unc"][t, s] = inf["farm_available"][s] - inf["farm"][s]
            row["dmg"][t, s] = inf["damage_taken"][s] / hpmax[lb[s]]
            row["died"][t, s] = float(inf["died"][s])
            row["rec"][t, s] = float(inf["recalled"][s])
            row["intr"][t, s] = float(inf["recall_interrupted"][s])
            row["crash"][t, s] = inf["crash_hp"][s]
            row["plates"][t, s] = inf["plates_broken"][s]
            row["tower"][t, s] = inf["tower_damage_dealt"][s] / 100.0
            row["gankp"][t, s] = inf["gank_p"][s]
            row["ganked"][t, s] = float(inf["ganked"][s])
            row["gankdmg"][t, s] = inf["gank_damage"][s] / hpmax[lb[s]]
            row["tatt"][t, s] = float(inf["trade_attempted"][s])
            row["tconn"][t, s] = float(inf["trade_connected"][s])
            row["tdealt"][t, s] = inf["trade_damage"][s] / hpmax[lb[e]]
            row["press"][t, s] = float(inf["effective_objectives"][s] == PRESSURE)
            row["visset"][t, s] = float(inf["vision_set"][s])
            row["visact"][t, s] = float(before.vision_of(s) > 0)
            row["invest"][t, s] = inf["invested"][s] / sc
            row["held"][t, s] = float(holds_freeze(after, s))
            row["away"][t, s] = float(inf["away"][s])
    for h in ep_h:
        hh = min(h, E)
        for k in SIDE_KEYS:
            for s, tag in ((f, "f"), (o, "o")):
                v = cols[k][:hh, s]
                out[f"{k}_{tag}{h}"] = float(v.max() if k in ANY_KEYS else v.sum())
    # event times (turn index from the branch turn; nan = not within the episode window E)
    def first(x):
        i = np.flatnonzero(x > 0)
        return float(i[0]) if len(i) else float("nan")
    out["t_rec_f"] = first(cols["rec"][:, f])
    out["t_death_f"] = first(cols["died"][:, f])
    out["t_death_o"] = first(cols["died"][:, o])
    out["t_left_f"] = first(cols["rec"][:, f] + cols["died"][:, f])
    held = cols["held"][:, f]
    out["freeze_len"] = float(np.argmin(held) if (held == 0).any() else E)    # consecutive held turns after t = 0..
    ca = infos[0]["champ_after"][f]
    out["first_hp_f"] = ca[1] / hpmax[infos[0]["levels"][f]] if ca[2] == 0 else float("nan")
    return out


def freeze_end(infos: List[dict], f: int, cfg, horizon: int = 40) -> str:
    """How a freeze held by the focal side at the branch state ended in one rollout (DESCRIPTIVE: an observed outcome,
    never a treatment). The first turn t* whose post-turn lane is not held by f:
        crashed_in          the held (opponent) wave crashed into f's tower on t*
        absence             either champion died or recalled on or before t*
        held_wave_cleared   the opponent front is gone (killed by f's champion, wave or tower)
        own_wave_died       f's own front is gone
        pushed_out          the fronts still fight, no longer at f's -1 (moved toward the lane)
        contact_lost        anything else
        held                still held after `horizon` turns (censored)"""
    o = 1 - f
    away_seen = False
    for t, inf in enumerate(infos[:horizon]):
        away_seen = away_seen or inf["died"][0] or inf["died"][1] or inf["recalled"][0] or inf["recalled"][1]
        lane = State.from_list(inf["state_after"])
        if holds_freeze(lane, f):
            continue
        if inf["crash_hp"][o] > 0:
            return "crashed_in"
        if away_seen:
            return "absence"
        ofx, ofh = lane.side(f)[:2]
        efx, efh = lane.side(o)[:2]
        if efh <= 0:
            return "held_wave_cleared"
        if ofh <= 0:
            return "own_wave_died"
        if ofx + efx >= -1:
            return "pushed_out"
        return "contact_lost"
    return "held"


FREEZE_ENDS = ("crashed_in", "absence", "held_wave_cleared", "own_wave_died", "pushed_out", "contact_lost", "held")


# ---------------------------------------------------------------------------------------------- one state
def evaluate(cfg, snap: Snapshot, focal: int, candidates: Dict[str, Plan], agents: Sequence, n: int, root: int,
             key: Sequence[int] = (), window: int = WINDOW, opponent_plan: Optional[Plan] = None,
             ep_h: Sequence[int] = EP_H, ret_h: Sequence[int] = RET_H, ref: str = "policy",
             diff_keys: Optional[Sequence[str]] = None, track_freeze: bool = False,
             extra: Optional[Callable] = None, cache: Optional[ProbCache] = None, keep_rollouts: bool = False) -> dict:
    """Paired rollouts of every candidate from `snap` (module docstring). `agents` = the continuation agents of seats
    (A, B); `opponent_plan` = the opponent's protocol (stateful plans are reset on every rollout's branch turn).
    Returns per-candidate means of every outcome, paired differences against `ref` (mean, standard error) for
    `diff_keys` (default: every return window and the main ledgers), plan logs, freeze end types and pairing checks.
    `extra(infos, focal) -> dict` adds experiment-specific per-rollout numbers (averaged like the others)."""
    env = LaneEnv(snap.config(cfg), cache_size=20_000)
    cache = cache if cache is not None else ProbCache()
    per: Dict[str, Dict[str, np.ndarray]] = {}
    logs: Dict[str, List[dict]] = {}
    ends: Dict[str, Dict[str, int]] = {}
    firsts, chance0, chance_last, opp_first = {}, {}, {}, {}
    for name, plan in candidates.items():
        rows = []
        logs[name] = []
        ends[name] = {e: 0 for e in FREEZE_ENDS}
        firsts[name], chance0[name], chance_last[name], opp_first[name] = [], [], [], []
        for m in range(n):
            r = rollout(env, snap, focal, plan, agents, rollout_seed(root, m, *key), cache, record=True,
                        max_turns=window, opponent_plan=opponent_plan)
            row = summarize(r.infos, focal, cfg, ep_h, ret_h)
            if extra is not None:
                row.update(extra(r.infos, focal))
            rows.append(row)
            if hasattr(plan, "last"):
                logs[name].append(plan.last)
            if track_freeze:
                ends[name][freeze_end(r.infos, focal, cfg)] += 1
            firsts[name].append(r.first_actions[focal])
            opp_first[name].append(r.first_actions[1 - focal])
            chance0[name].append(tuple(r.infos[0]["chance"]))
            chance_last[name].append(tuple(r.infos[-1]["chance"]))
        per[name] = {k: np.array([row[k] for row in rows], dtype=float) for k in rows[0]}
    names = list(candidates)
    keys = list(per[names[0]])
    if diff_keys is None:
        diff_keys = [k for k in keys if k.startswith("G")] + [k for k in keys if k.split("_")[0] in
                                                               ("gold", "xp", "dmg", "died", "rec")]
    out = {"candidates": names, "n": n, "window": window, "ref": ref,
           "mean": {c: {k: _nanmean(per[c][k]) for k in keys} for c in names},
           "diff": {}, "plan_log": {}, "first_action": {c: _counts(firsts[c]) for c in names},
           "pairing": {"opponent_first_action_identical": all(opp_first[c] == opp_first[names[0]] for c in names),
                       "chance_identical_first_turn": all(chance0[c] == chance0[names[0]] for c in names),
                       "chance_identical_last_turn": all(chance_last[c] == chance_last[names[0]] for c in names)}}
    if ref in per:
        for c in names:
            if c == ref:
                continue
            out["diff"][c] = {}
            for k in diff_keys:
                d = per[c][k] - per[ref][k]
                d = d[np.isfinite(d)]
                out["diff"][c][k] = [float(d.mean()) if len(d) else float("nan"),
                                     float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")]
    for c in names:
        if logs[c]:
            L = logs[c]
            out["plan_log"][c] = {"applied": _nanmean([x["applied"] for x in L]),
                                  "fallback": _nanmean([x["fallback"] for x in L]),
                                  "policy": _nanmean([x["policy"] for x in L]),
                                  "active_turns": _nanmean([x["active_turns"] for x in L]),
                                  "stop_reasons": _counts([x["stop_reason"] for x in L])}
    if track_freeze:
        out["freeze_end"] = ends
    if keep_rollouts:
        out["rollouts"] = {c: {k: per[c][k].tolist() for k in ("G20", "G300") if k in per[c]} for c in names}
    return out


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def _counts(xs) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


# ---------------------------------------------------------------------------------------------- aggregation
def state_values(states: List[dict], cand: str, metric: str, ref: Optional[str] = None) -> List[float]:
    """Per-state values: the candidate's mean (ref None) or its paired difference against the state's reference."""
    out = []
    for st in states:
        r = st["result"]
        if cand not in r["candidates"]:
            continue
        if ref is None:
            out.append(r["mean"][cand].get(metric, float("nan")))
        elif cand == ref:
            out.append(0.0)
        elif ref == r["ref"]:
            out.append(r["diff"].get(cand, {}).get(metric, [float("nan")])[0])
        else:                                                   # a contrast between two non-reference candidates
            if ref not in r["candidates"]:
                continue
            out.append(r["mean"][cand][metric] - r["mean"][ref][metric])
    return out


def across_states(states: List[dict], cand: str, metric: str, ref: Optional[str] = None) -> dict:
    v = state_values(states, cand, metric, ref)
    m, h, n = t_ci(v)
    return {"mean": m, "hw": h, "n": n}


def share_significant(states: List[dict], cand: str, metric: str = "G300") -> Dict[str, float]:
    """Share of states whose paired difference against the reference is significantly positive / negative
    (|d| > 1.96 SE, within that state's rollouts)."""
    pos = neg = n = 0
    for st in states:
        r = st["result"]
        if cand not in r["diff"]:
            continue
        d, se = r["diff"][cand][metric]
        if not np.isfinite(se):
            continue
        n += 1
        pos += d > 1.96 * se
        neg += d < -1.96 * se
    return {"n": n, "pos": pos / n if n else float("nan"), "neg": neg / n if n else float("nan")}


def across_seeds(values: Sequence[float]) -> dict:
    """The per-seed means and their t interval (2 df for three seeds): the training procedure's replication."""
    m, h, n = t_ci(values)
    return {"values": [float(x) for x in values], "mean": m, "hw": h, "n": n}


__all__ = ["RET_H", "EP_H", "WINDOW", "SIDE_KEYS", "summarize", "freeze_end", "FREEZE_ENDS", "evaluate",
           "state_values", "across_states", "share_significant", "across_seeds", "FollowPolicy"]
