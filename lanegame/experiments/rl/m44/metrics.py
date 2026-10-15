"""Strategic metrics of M4.4, computed from game records only (records.py). Precise definitions are in
docs/m4/m4_4/m4_4a_ppo_readiness.md §4; the short form is repeated in each function.

Every metric takes a list of `GameRecord`s and a focal agent name (`focal`; None = every side of every record) and
reports counts with the rates, so a reader can see how much evidence a number rests on. Rates that are averages of
per-game values carry a 95% interval over (game, focal side) units (`ci`); pooled rates over turns do not (turns of
one game are not independent).

Principles:
  * an ACTION LABEL is never taken as evidence of a STATE: "chose Slow Push" and "a slow push formed" are separate
    numbers; a freeze is a lane state that persisted, whatever actions were chosen; zoning is measured by its
    consequences (position, uncollected gold and XP, deaths), never by "chose Trade";
  * natural and initialized games are never pooled: `GameRecord.initialized` separates games from s0 from games
    started in a constructed state (a scripted freeze, a constructed power gap); callers report them separately
    (`split_by_start`);
  * phases come from the simulation state (plates left on a tower), never from a fixed turn number.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ....config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD
from .actions import BEH_NAMES, WAVE_NAMES
from .common import mean_ci
from .records import GameRecord

A, B = 0, 1
LH, TN, SL, SH = 0, 1, 2, 3                    # WAVE_ORDER indices
MIN_FREEZE = 4                                 # turns: two waves (one every 2 turns) arrive while the lane stays held
SUSTAINED_FREEZE = 10
AFTER_WINDOW = 20                              # turns after a freeze / crash for the economic consequence
WARD_CELL = 99                                 # positional.WARD: the off-lane Vision node W (M4.5a), not a lane cell


def lane_mean(values) -> float:
    """M4.6b measurement fix: the mean cell over LANE cells only (the Ward node W = 99 excluded; nan if none)."""
    v = [float(x) for x in values if np.isfinite(x) and x != WARD_CELL]
    return float(np.mean(v)) if v else float("nan")


def ward_share(values) -> float:
    """The share of the given positions at the Ward node W (finite entries only)."""
    v = [x for x in values if np.isfinite(x)]
    return float(np.mean([x == WARD_CELL for x in v])) if v else float("nan")


STRONGER, WEAKER = 1.25, 0.8                   # attack-power ratio classes (as M4.3b)
NEAR = 3                                       # |pos + enemy pos| <= champion_speed + trade_range: catchable


def _sides(rec: GameRecord, focal: Optional[str]) -> List[int]:
    return [A, B] if focal is None else [s for s in (A, B) if rec.names[s] == focal]


def _units(records: Sequence[GameRecord], focal: Optional[str]):
    """(record, focal side) pairs."""
    for rec in records:
        for s in _sides(rec, focal):
            yield rec, s


def split_by_start(records: Sequence[GameRecord]) -> Dict[str, List[GameRecord]]:
    out: Dict[str, List[GameRecord]] = {}
    for r in records:
        out.setdefault(("initialized:" if r.initialized else "natural:") + r.start, []).append(r)
    return out


def _share(x: np.ndarray, m: np.ndarray) -> float:
    n = int(m.sum())
    return float(x[m].mean()) if n else float("nan")


def _wave_shares(w: np.ndarray) -> Dict[str, float]:
    n = len(w)
    return {WAVE_NAMES[i]: (float((w == i).sum()) / n if n else float("nan")) for i in range(4)}


def _beh_shares(b: np.ndarray) -> Dict[str, float]:
    n = len(b)
    return {BEH_NAMES[i]: (float((b == i).sum()) / n if n else float("nan")) for i in range(5)}


def _ci(vals) -> List[float]:
    m, h = mean_ci(vals)
    return [m, h]


def _runs(flag: np.ndarray) -> List[Tuple[int, int]]:
    """Maximal runs [t0, t1] (inclusive) of True."""
    out, t0 = [], None
    for t, f in enumerate(flag):
        if f and t0 is None:
            t0 = t
        elif not f and t0 is not None:
            out.append((t0, t - 1))
            t0 = None
    if t0 is not None:
        out.append((t0, len(flag) - 1))
    return out


def power_class(p: float) -> str:
    return "stronger" if p >= STRONGER else ("weaker" if p <= WEAKER else "even")


# ---------------------------------------------------------------------------------------------- 1. wave management
WAVE_STATES = {
    # decision-state classes of the focal side (start of the turn); overlapping by design
    "all present turns": lambda d, s: np.ones(len(d["present"]), bool),
    "fronts in contact at the lane cell": lambda d, s: (d["contact"][:, s] > 0) & (d["wave_pos"][:, s] == 0),
    "my wave stronger (window strength +1 or more)": lambda d, s: d["wave_adv"][:, s] >= 1,
    "waves even (|difference| < 1)": lambda d, s: np.abs(d["wave_adv"][:, s]) < 1,
    "enemy wave stronger (-1 or less)": lambda d, s: d["wave_adv"][:, s] <= -1,
    "enemy wave much stronger (-4 or less)": lambda d, s: d["wave_adv"][:, s] <= -4,
    "waves meet on my half (position <= -1)": lambda d, s: d["wave_pos"][:, s] <= -1,
    "waves meet on the enemy half (position >= 1)": lambda d, s: d["wave_pos"][:, s] >= 1,
    "holding a freeze (freeze state, I am the freezer)": lambda d, s: d["held"][:, s] > 0,
    "frozen on (freeze state at the enemy tower)": lambda d, s: d["held_on"][:, s] > 0,
    "my crash under the enemy tower": lambda d, s: d["own_siege"][:, s] > 0,
    "enemy crash under my tower": lambda d, s: d["en_siege"][:, s] > 0,
    "enemy champion absent": lambda d, s: d["en_present"][:, s] == 0,
}


def wave_management(records: Sequence[GameRecord], focal: Optional[str] = None) -> dict:
    """Wave-action frequencies, overall and conditional on the decision state (WAVE_STATES), with the policy's mean
    probability masses where recorded; crashes (count, size, timing); where the waves meet over time; wave advantage;
    actions in the 5 turns before / after a crash (crasher and defender); the reply to an opponent that kept Thinning
    or Shoving for 3 turns; and the realized EFFECT of Slow Push against Last Hit at a neutral lane (did the wave
    actually move or gain strength 4 turns later), so that an action label is never read as an outcome."""
    by_state: Dict[str, dict] = {}
    pooled_w: Dict[str, List[np.ndarray]] = {k: [] for k in WAVE_STATES}
    pooled_p: Dict[str, List[np.ndarray]] = {k: [] for k in WAVE_STATES}
    per_game_shares = {n: [] for n in WAVE_NAMES}
    crashes, crash_sizes, first_crash, crash_rate = [], [], [], []
    pos_bins = {b: [] for b in ("0-49", "50-99", "100-199", "200-299", "300+")}
    adv = {"mine": [], "even": [], "enemy": []}
    around = {"crasher_before": [], "crasher_after": [], "defender_before": [], "defender_after": []}
    reply = {"opponent Thinned 3 turns": [], "opponent Shoved 3 turns": [], "opponent Last Hit 3 turns": [],
             "otherwise": []}
    effect = {n: {"n": 0, "moved_forward": 0, "gained_strength": 0} for n in WAVE_NAMES}
    for rec, s in _units(records, focal):
        d = rec.data
        T = rec.turns
        o = 1 - s
        pres = (d["present"][:, s] > 0) & (d["wave"][:, s] >= 0)
        w = d["wave"][:, s].astype(int)
        if pres.any():
            for i, n in enumerate(WAVE_NAMES):
                per_game_shares[n].append(float((w[pres] == i).mean()))
        for name, f in WAVE_STATES.items():
            m = pres & f(d, s)
            pooled_w[name].append(w[m])
            if np.isfinite(d["pw0"][:, s]).any():
                pooled_p[name].append(np.stack([d[f"pw{i}"][m, s] for i in range(4)], axis=1))
        c = d["crash"][:, s]
        ci = np.flatnonzero(c > 0)
        crashes.append(len(ci))
        crash_sizes += c[ci].tolist()
        first_crash.append(int(ci[0]) + 1 if len(ci) else None)
        crash_rate.append(100.0 * len(ci) / T)
        turns = np.arange(T)
        for b, (lo, hi) in {"0-49": (0, 50), "50-99": (50, 100), "100-199": (100, 200), "200-299": (200, 300),
                            "300+": (300, 10 ** 9)}.items():
            m = (turns >= lo) & (turns < hi) & np.isfinite(d["wave_pos"][:, s])
            if m.any():
                pos_bins[b].append(float(d["wave_pos"][m, s].mean()))
        a = d["wave_adv"][:, s]
        adv["mine"].append(float((a >= 1).mean()))
        adv["even"].append(float((np.abs(a) < 1).mean()))
        adv["enemy"].append(float((a <= -1).mean()))
        for t in ci:                                       # my crash at t: my actions / the defender's around it
            for side, key in ((s, "crasher"), (o, "defender")):
                ww, pp = d["wave"][:, side].astype(int), (d["present"][:, side] > 0) & (d["wave"][:, side] >= 0)
                bef = [ww[u] for u in range(max(0, t - 5), t) if pp[u]]
                aft = [ww[u] for u in range(t + 1, min(T, t + 6)) if pp[u]]
                around[f"{key}_before"] += bef
                around[f"{key}_after"] += aft
        ow = d["wave"][:, o].astype(int)
        for t in range(3, T):
            if not pres[t]:
                continue
            last = ow[t - 3:t]
            key = ("opponent Thinned 3 turns" if (last == TN).all() else "opponent Shoved 3 turns" if (last == SH).all()
                   else "opponent Last Hit 3 turns" if (last == LH).all() else "otherwise")
            reply[key].append(w[t])
        neutral = pres & (d["contact"][:, s] > 0) & (d["wave_pos"][:, s] == 0)
        for t in np.flatnonzero(neutral):
            if t + 4 >= T or not np.isfinite(d["wave_pos"][t + 4, s]):
                continue
            e = effect[WAVE_NAMES[w[t]]]
            e["n"] += 1
            e["moved_forward"] += int(d["wave_pos"][t + 4, s] > d["wave_pos"][t, s])
            e["gained_strength"] += int(d["wave_adv"][t + 4, s] >= 1)
    for name in WAVE_STATES:
        ws = np.concatenate(pooled_w[name]) if pooled_w[name] else np.zeros(0, int)
        row = {"n": int(len(ws)), "chose": _wave_shares(ws)}
        if pooled_p[name]:
            ps = np.concatenate(pooled_p[name])
            if len(ps):
                row["mean_prob"] = {WAVE_NAMES[i]: float(np.nanmean(ps[:, i])) for i in range(4)}
        by_state[name] = row
    for e in effect.values():
        n = max(e["n"], 1)
        e["moved_forward_rate"] = e["moved_forward"] / n if e["n"] else float("nan")
        e["gained_strength_rate"] = e["gained_strength"] / n if e["n"] else float("nan")
    cs = np.asarray(crash_sizes, dtype=float)
    return {
        "frequency_per_game": {n: _ci(v) for n, v in per_game_shares.items()},
        "by_state": by_state,
        "crashes": {"per_game": _ci(crashes), "per_100_turns": _ci(crash_rate), "n": int(len(cs)),
                    "size_mean": float(cs.mean()) if len(cs) else float("nan"),
                    "size_quantiles_25_50_75": np.percentile(cs, [25, 50, 75]).tolist() if len(cs) else [],
                    "size_classes": {"small (< 12 HP)": int((cs < 12).sum()), "medium (12-23)": int(((cs >= 12) & (cs < 24)).sum()),
                                     "large (>= 24)": int((cs >= 24).sum())},
                    "first_crash_turn": _ci([x for x in first_crash if x is not None]),
                    "games_without_crash": int(sum(x is None for x in first_crash))},
        "wave_position_by_turn": {b: _ci(v) for b, v in pos_bins.items()},
        "wave_advantage_share": {k: _ci(v) for k, v in adv.items()},
        "around_my_crash": {k: {"n": len(v), "shares": _wave_shares(np.asarray(v, int))} for k, v in around.items()},
        "reply_to_opponent_pattern": {k: {"n": len(v), "shares": _wave_shares(np.asarray(v, int))} for k, v in reply.items()},
        "neutral_action_effect_after_4_turns": effect,
    }


# ---------------------------------------------------------------------------------------------- 2. freezes
END_TYPES = ("crashed_in", "held_wave_cleared", "pushed_out", "own_wave_died", "contact_lost", "censored")


def freeze_episodes(rec: GameRecord, s: int) -> List[dict]:
    """The freeze episodes of side s (the FREEZER) in one record.

    Freeze state F_s(t) (start of turn t): both fronts alive and fighting, and the enemy front at s's own -1 or deeper
    (records.holds_freeze): s holds the enemy wave next to its own tower. An EPISODE is a maximal run of consecutive
    turns in the freeze state; ESTABLISHED when it lasts >= MIN_FREEZE turns (two waves arrive while it holds),
    SUSTAINED when >= SUSTAINED_FREEZE. The END is classified from the state after the last held turn t1:
        crashed_in        the held wave crashed into the freezer's tower on t1 (the frozen-on side pushed through)
        held_wave_cleared the enemy front is gone without a crash (the freezer or its tower killed the held wave)
        pushed_out        the fronts still fight but no longer at the freezer's -1 (the wave moved toward the lane)
        own_wave_died     the freezer's own front is gone
        contact_lost      anything else (the fronts separated)
        censored          the record ends on the last held turn (whether and how the freeze ended is not observed)
    Per episode: the freezer's and the frozen-on side's actions, positions and power at the start, the gold / XP each
    earned during the episode and in the AFTER_WINDOW turns after it, the frozen-on side's XP lost for good and
    uncollected available gold, and who chose Shove on the last two held turns."""
    d = rec.data
    o = 1 - s
    T = rec.turns
    out = []
    for t0, t1 in _runs(d["held"][:, s] > 0):
        L = t1 - t0 + 1
        nxt = t1 + 1
        if d["en_crash"][t1, s] > 0:
            end = "crashed_in"
        elif nxt >= T:
            end = "censored"                # the record ends on the last held turn: how (or whether) it ended is unseen
        else:
            if d["en_fh"][nxt, s] == 0:
                end = "held_wave_cleared"
            elif d["own_fh"][nxt, s] == 0:
                end = "own_wave_died"
            elif d["contact"][nxt, s] > 0:
                end = "pushed_out"
            else:
                end = "contact_lost"
        sl = slice(t0, t1 + 1)
        fw = d["wave"][sl, s].astype(int)
        fw = fw[fw >= 0]
        ow_ = d["wave"][sl, o].astype(int)
        ow_ = ow_[ow_ >= 0]
        ob = d["beh"][sl, o].astype(int)
        ob = ob[(ob >= 0) & (d["present"][sl, o] > 0)]
        aft = slice(nxt, min(T, nxt + AFTER_WINDOW))
        last2 = range(max(t0, t1 - 1), t1 + 1)
        ep = {
            "start": int(t0), "length": int(L), "end": end, "established": L >= MIN_FREEZE,
            "sustained": L >= SUSTAINED_FREEZE, "power_class": power_class(float(d["power"][t0, s])),
            "power": float(d["power"][t0, s]), "level_diff": float(d["level"][t0, s] - d["elevel"][t0, s]),
            "plates_left": int(d["plates_left"][t0, s]),
            "freezer_waves": _wave_shares(fw), "frozen_on_waves": _wave_shares(ow_), "frozen_on_beh": _beh_shares(ob),
            "freezer_preserving_share": float(np.isin(fw, (LH, TN)).mean()) if len(fw) else float("nan"),
            "frozen_on_mean_pos": lane_mean(d["pos"][sl, o][d["present"][sl, o] > 0]),
            "frozen_on_ward_share": ward_share(d["pos"][sl, o][d["present"][sl, o] > 0]),
            "gold_diff": float((d["earned"][sl, s] - d["earned"][sl, o]).sum()),
            "farm_diff": float((d["farm"][sl, s] - d["farm"][sl, o]).sum()),
            "xp_diff": float((d["xp"][sl, s] - d["xp"][sl, o]).sum()),
            "frozen_on_xp_missed": float(d["xp_missed"][sl, o].sum()),
            "frozen_on_uncollected": float((d["farm_avail"][sl, o] - d["farm"][sl, o]).sum()),
            "freezer_uncollected": float((d["farm_avail"][sl, s] - d["farm"][sl, s]).sum()),
            "after_gold_diff": float((d["earned"][aft, s] - d["earned"][aft, o]).sum()) if nxt < T else float("nan"),
            "after_turns": int(max(0, min(T, nxt + AFTER_WINDOW) - nxt)),
            "freezer_shoved_at_end": bool(any(d["wave"][u, s] == SH for u in last2)),
            "frozen_on_shoved_at_end": bool(any(d["wave"][u, o] == SH for u in last2)),
            "frozen_on_deaths": int(d["died"][sl, o].sum()), "freezer_deaths": int(d["died"][sl, s].sum()),
        }
        out.append(ep)
    return out


def freezes(records: Sequence[GameRecord], focal: Optional[str] = None) -> dict:
    """Freeze episodes in which the focal side is the FREEZER (`freeze_episodes`), aggregated; plus the focal side's
    turns spent frozen on. `useful denial`: an established episode in which the freezer out-earned the frozen-on side
    (gold_diff > 0); reported with the XP the frozen-on side lost for good, never inferred from the actions chosen.
    `preserving share`: the share of the freezer's held turns on Last Hit or Thin; an action pattern consistent with
    keeping the freeze, NOT evidence that the freeze was kept (the episode length is)."""
    eps, turns, held_turns, held_on_turns, per_game_est = [], 0, 0, 0, []
    for rec, s in _units(records, focal):
        e = freeze_episodes(rec, s)
        eps += e
        turns += rec.turns
        held_turns += int((rec.data["held"][:, s] > 0).sum())
        held_on_turns += int((rec.data["held_on"][:, s] > 0).sum())
        per_game_est.append(1000.0 * sum(x["established"] for x in e) / rec.turns)
    est = [e for e in eps if e["established"]]

    def summary(group: List[dict]) -> dict:
        if not group:
            return {"n": 0}
        L = np.array([e["length"] for e in group])
        g = [e["gold_diff"] for e in group]
        return {
            "n": len(group), "length_mean": float(L.mean()), "length_median": float(np.median(L)),
            "length_p90": float(np.percentile(L, 90)), "length_max": int(L.max()),
            "end": {k: sum(e["end"] == k for e in group) for k in END_TYPES},
            "freezer_power": {k: sum(e["power_class"] == k for e in group) for k in ("stronger", "even", "weaker")},
            "gold_diff": _ci(g), "farm_diff": _ci([e["farm_diff"] for e in group]),
            "xp_diff": _ci([e["xp_diff"] for e in group]),
            "useful_denial_share": float(np.mean([x > 0 for x in g])),
            "frozen_on_xp_missed": _ci([e["frozen_on_xp_missed"] for e in group]),
            "frozen_on_uncollected": _ci([e["frozen_on_uncollected"] for e in group]),
            "after_gold_diff": _ci([e["after_gold_diff"] for e in group if np.isfinite(e["after_gold_diff"])]),
            "freezer_preserving_share": _ci([e["freezer_preserving_share"] for e in group]),
            "freezer_waves": {n: _ci([e["freezer_waves"][n] for e in group]) for n in WAVE_NAMES},
            "frozen_on_waves": {n: _ci([e["frozen_on_waves"][n] for e in group]) for n in WAVE_NAMES},
            "frozen_on_beh": {n: _ci([e["frozen_on_beh"][n] for e in group]) for n in BEH_NAMES},
            "freezer_shoved_at_end": float(np.mean([e["freezer_shoved_at_end"] for e in group])),
            "frozen_on_shoved_at_end": float(np.mean([e["frozen_on_shoved_at_end"] for e in group])),
            "frozen_on_deaths_per_episode": float(np.mean([e["frozen_on_deaths"] for e in group])),
            "plates_phase": {"plates left": sum(e["plates_left"] > 0 for e in group),
                             "plates exhausted": sum(e["plates_left"] == 0 for e in group)},
        }

    return {"focal_turns": turns, "held_share": held_turns / max(turns, 1), "held_on_share": held_on_turns / max(turns, 1),
            "established_per_1000_turns": _ci(per_game_est), "all_episodes": summary(eps), "established": summary(est),
            "sustained": summary([e for e in eps if e["sustained"]]),
            "established_by_power": {k: summary([e for e in est if e["power_class"] == k])
                                     for k in ("stronger", "even", "weaker")},
            "definition": {"state": "both fronts alive and in contact, enemy front at the freezer's own -1 or deeper",
                           "established_turns": MIN_FREEZE, "sustained_turns": SUSTAINED_FREEZE,
                           "after_window": AFTER_WINDOW}}


# ---------------------------------------------------------------------------------------------- 3. zoning
def credible_threat(d: Dict[str, np.ndarray], s: int) -> np.ndarray:
    """CREDIBLE THREAT on side s at the start of a turn: both champions present, the enemy is the stronger by
    attack power (my power ratio <= WEAKER) AND by the HP-aware duel edge (edge < 0), s is outside its own tower cell
    (where only a dive reaches it) and the enemy is within catch distance (|pos + enemy pos| <= NEAR)."""
    both = (d["present"][:, s] > 0) & (d["en_present"][:, s] > 0)
    return both & (d["power"][:, s] <= WEAKER) & (d["edge"][:, s] < 0) & (d["pos"][:, s] > -2) & (d["dist"][:, s] <= NEAR)


def zoning(records: Sequence[GameRecord], focal: Optional[str] = None) -> dict:
    """Combat threat and resource denial. Per power class of the focal side (both present): position, share at the
    own tower cell, trade attempt / connection rates, connected-trade outcomes, deaths. Under a CREDIBLE THREAT
    (`credible_threat`) against the not-threatened turns of the same weaker side: behaviour, position, gold left
    uncollected, XP lost, and deaths / Recalls in the next 4 turns. From the stronger side: the opponent's uncollected
    gold when the focal side is stronger and near, and the focal side's own collection when it Trades vs when it
    Holds (the price of applying pressure). Natural power gaps (from s0) and initialized ones are separated by the
    caller (split_by_start)."""
    cls = {"stronger": [], "even": [], "weaker": []}
    rows = {k: {"pos": [], "at_own_tower": [], "trade_att": [], "trade_conn": [], "dealt": [], "taken": [], "died": [],
                "farm_rate": [], "n": 0} for k in cls}
    threat = {"threat": {"beh": [], "pos": [], "uncollected": [], "xp_missed": [], "died4": [], "recalled4": [], "n": 0},
              "weaker, no threat": {"beh": [], "pos": [], "uncollected": [], "xp_missed": [], "died4": [], "recalled4": [], "n": 0}}
    pressure = {"opp_uncollected_when_stronger_near": [], "opp_uncollected_otherwise": [],
                "own_collection_trading": [], "own_collection_holding": []}
    deaths = {"deaths": 0, "low_hp_recalls": 0, "other_recalls": 0, "turns": 0, "deaths_after_connected_trade_3": 0,
              "connected_trades": 0}
    for rec, s in _units(records, focal):
        d = rec.data
        o = 1 - s
        T = rec.turns
        both = (d["present"][:, s] > 0) & (d["en_present"][:, s] > 0)
        deaths["turns"] += T
        deaths["deaths"] += int(d["died"][:, s].sum())
        rc = d["recalled"][:, s] > 0
        deaths["low_hp_recalls"] += int((rc & (d["hp_frac"][:, s] <= 0.4)).sum())
        deaths["other_recalls"] += int((rc & (d["hp_frac"][:, s] > 0.4)).sum())
        for t in np.flatnonzero(d["trade_conn"][:, s] > 0):
            deaths["connected_trades"] += 1
            deaths["deaths_after_connected_trade_3"] += int(d["died"][t:t + 4, s].sum() > 0)
        for t in np.flatnonzero(both):
            k = power_class(float(d["power"][t, s]))
            r = rows[k]
            r["n"] += 1
            r["pos"].append(d["pos"][t, s])
            r["at_own_tower"].append(d["pos"][t, s] == -2)
            r["trade_att"].append(d["trade_att"][t, s])
            r["trade_conn"].append(d["trade_conn"][t, s])
            if d["trade_conn"][t, s]:
                r["dealt"].append(d["trade_dealt"][t, s])
                r["taken"].append(d["trade_taken"][t, s])
            r["died"].append(d["died"][t, s])
            if d["farm_avail"][t, s] > 0:
                r["farm_rate"].append(d["farm"][t, s] / d["farm_avail"][t, s])
        th = credible_threat(d, s)
        weaker = both & (d["power"][:, s] <= WEAKER)
        for key, m in (("threat", th), ("weaker, no threat", weaker & ~th)):
            for t in np.flatnonzero(m):
                g = threat[key]
                g["n"] += 1
                g["beh"].append(int(d["beh"][t, s]))
                g["pos"].append(d["pos"][t, s])
                g["uncollected"].append(d["farm_avail"][t, s] - d["farm"][t, s])
                g["xp_missed"].append(d["xp_missed"][t, s])
                g["died4"].append(float(d["died"][t:t + 4, s].sum() > 0))
                g["recalled4"].append(float(d["recalled"][t:t + 4, s].sum() > 0))
        stronger_near = both & (d["power"][:, s] >= STRONGER) & (d["dist"][:, s] <= NEAR)
        for t in np.flatnonzero(both):
            v = d["farm_avail"][t, o] - d["farm"][t, o]
            (pressure["opp_uncollected_when_stronger_near"] if stronger_near[t] else pressure["opp_uncollected_otherwise"]).append(v)
            if d["farm_avail"][t, s] > 0:
                rate = d["farm"][t, s] / d["farm_avail"][t, s]
                if d["beh"][t, s] == B_TRADE:
                    pressure["own_collection_trading"].append(rate)
                elif d["beh"][t, s] == B_HOLD:
                    pressure["own_collection_holding"].append(rate)

    def mean(x):
        return float(np.mean(x)) if len(x) else float("nan")

    by_power = {k: {"n": r["n"], "mean_pos": lane_mean(r["pos"]), "ward_share": ward_share(r["pos"]),
                    "at_own_tower": mean(r["at_own_tower"]),
                    "trade_attempt_rate": mean(r["trade_att"]), "trade_connect_rate": mean(r["trade_conn"]),
                    "connected_dealt_bars": mean(r["dealt"]), "connected_taken_bars": mean(r["taken"]),
                    "death_rate": mean(r["died"]), "collection_rate": mean(r["farm_rate"])} for k, r in rows.items()}
    thr = {k: {"n": g["n"], "behavior": _beh_shares(np.asarray(g["beh"], int)), "mean_pos": lane_mean(g["pos"]),
               "ward_share": ward_share(g["pos"]),
               "uncollected_per_turn": mean(g["uncollected"]), "xp_missed_per_turn": mean(g["xp_missed"]),
               "died_within_4": mean(g["died4"]), "recalled_within_4": mean(g["recalled4"])} for k, g in threat.items()}
    tu = max(deaths["turns"], 1)
    return {"by_power": by_power, "credible_threat": thr,
            "pressure_cost": {k: {"n": len(v), "mean": mean(v)} for k, v in pressure.items()},
            "deaths_per_1000_turns": 1000.0 * deaths["deaths"] / tu,
            "low_hp_recalls_per_1000_turns": 1000.0 * deaths["low_hp_recalls"] / tu,
            "other_recalls_per_1000_turns": 1000.0 * deaths["other_recalls"] / tu,
            "connected_trades_per_1000_turns": 1000.0 * deaths["connected_trades"] / tu,
            "death_within_3_after_connected_trade": deaths["deaths_after_connected_trade_3"] / max(deaths["connected_trades"], 1),
            "definition": {"stronger": f"power ratio >= {STRONGER}", "weaker": f"power ratio <= {WEAKER}",
                           "credible_threat": "both present, power <= 0.8 and duel edge < 0, outside the own tower "
                                              f"cell, |pos + enemy pos| <= {NEAR}"}}


# ---------------------------------------------------------------------------------------------- 4. vision and tower
def vision_tower(records: Sequence[GameRecord], focal: Optional[str] = None) -> dict:
    """Siege objectives, vision, ganks, plates, dives and tower shots.

    objective turns   turns on which Pressure / Vision is legal (own crash under the enemy tower, tower reachable)
    extended pressure a run of >= 3 consecutive turns with Pressure chosen; "with vision" when the vision timer was
                      active at its first turn or vision was established in the 2 turns before it
    forward           the champion ended the turn at its own +1 or +2
    dive              a connected trade ending in the enemy tower cell; "busy" when the diver's own crash is under
                      that tower or crashes that turn (the condition under which econ halves the shot and aggro
                      does not); the DEFENDER's reply is the dived side's behaviour on the dive turn and the next"""
    obj = {"stronger": [], "even": [], "weaker": []}
    obj_novis = {"stronger": [], "even": [], "weaker": []}
    runs_total = runs_vis = 0
    fwd = {"vision": {"n": 0, "gank_p": [], "exp_gank": [], "gank_dmg": [], "died": []},
           "no vision": {"n": 0, "gank_p": [], "exp_gank": [], "gank_dmg": [], "died": []}}
    g = {"turns": 0, "ganks": 0, "exp_gank": 0.0, "gank_dmg": 0.0, "gank_deaths": 0, "vision_set": 0,
         "pressure_eff": 0, "plates": 0}
    plate_turns: List[List[int]] = [[], [], []]
    dives = {"busy": {"n": 0, "tower_p": [], "shot": [], "died2": [], "dealt": [], "taken": []},
             "not busy": {"n": 0, "tower_p": [], "shot": [], "died2": [], "dealt": [], "taken": []}}
    adj = {"busy": 0, "not busy": 0}
    defender = {"Hold": 0, "Yield": 0, "Trade": 0, "Safe": 0, "Recall": 0, "absent": 0}
    defender_next = dict(defender)
    shots = {"expected": 0.0, "realized": 0, "expected_busy": 0.0, "expected_not_busy": 0.0}
    for rec, s in _units(records, focal):
        d = rec.data
        o = 1 - s
        T = rec.turns
        g["turns"] += T
        g["ganks"] += int(d["ganked"][:, s].sum())
        g["exp_gank"] += float(d["exp_gank"][:, s].sum())
        g["gank_dmg"] += float(d["gank_dmg"][:, s].sum())
        g["gank_deaths"] += int(((d["died"][:, s] > 0) & (d["gank_dmg"][:, s] > 0)).sum())
        g["vision_set"] += int(d["vision_set"][:, s].sum())
        g["pressure_eff"] += int(d["pressure_eff"][:, s].sum())
        g["plates"] += int(d["plates"][:, s].sum())
        cum = np.cumsum(d["plates"][:, s])
        for j in range(3):
            hit = np.flatnonzero(cum >= j + 1)
            if len(hit):
                plate_turns[j].append(int(hit[0]) + 1)
        legal = (d["obj_legal"][:, s] > 0) & (d["present"][:, s] > 0)
        for t in np.flatnonzero(legal):
            k = power_class(float(d["power"][t, s]))
            obj[k].append(int(d["obj"][t, s]))
            if d["vision"][t, s] == 0:
                obj_novis[k].append(int(d["obj"][t, s]))
        for t0, t1 in _runs(d["obj"][:, s] == 1):
            if t1 - t0 + 1 >= 3:
                runs_total += 1
                runs_vis += int(d["vision"][t0, s] > 0 or d["vision_set"][max(0, t0 - 2):t0, s].sum() > 0)
        f = (d["present"][:, s] > 0) & (d["end_pos"][:, s] >= 1)
        for t in np.flatnonzero(f):
            key = "vision" if d["vision"][t, s] > 0 else "no vision"
            x = fwd[key]
            x["n"] += 1
            x["gank_p"].append(d["gank_p"][t, s])
            x["exp_gank"].append(d["exp_gank"][t, s])
            x["gank_dmg"].append(d["gank_dmg"][t, s])
            x["died"].append(d["died"][t, s])
        shots["expected"] += float(d["tower_p"][:, s].sum())
        shots["realized"] += int((d["tower_dmg"][:, s] > 0).sum())
        busy = d["busy"][:, s] > 0
        shots["expected_busy"] += float(d["tower_p"][busy, s].sum())
        shots["expected_not_busy"] += float(d["tower_p"][~busy, s].sum())
        for t in np.flatnonzero(d["adj_trade"][:, s] > 0):
            adj["busy" if busy[t] else "not busy"] += 1
        for t in np.flatnonzero(d["dive"][:, s] > 0):
            x = dives["busy" if busy[t] else "not busy"]
            x["n"] += 1
            x["tower_p"].append(d["tower_p"][t, s])
            x["shot"].append(d["tower_dmg"][t, s] > 0)
            x["died2"].append(float(d["died"][t:t + 3, s].sum() > 0))
            x["dealt"].append(d["trade_dealt"][t, s])
            x["taken"].append(d["damage_taken"][t, s])
            for u, tab in ((t, defender), (t + 1, defender_next)):
                if u >= T:
                    continue
                if d["present"][u, o] == 0:
                    tab["absent"] += 1
                else:
                    tab[BEH_NAMES[int(d["beh"][u, o])]] += 1

    def obj_shares(v):
        v = np.asarray(v, int)
        n = len(v)
        return {"n": n, "Pressure": float((v == 1).mean()) if n else float("nan"),
                "Vision": float((v == 2).mean()) if n else float("nan"),
                "none": float((v == 0).mean()) if n else float("nan")}

    def mean(x):
        return float(np.mean(x)) if len(x) else float("nan")

    tu = max(g["turns"], 1)
    return {
        "objective_choice_when_legal": {k: obj_shares(v) for k, v in obj.items()},
        "objective_choice_when_legal_no_active_vision": {k: obj_shares(v) for k, v in obj_novis.items()},
        "extended_pressure_runs": {"n": runs_total, "with_vision_share": runs_vis / runs_total if runs_total else float("nan")},
        "forward_exposure": {k: {"n": x["n"], "gank_p": mean(x["gank_p"]), "expected_gank_bars": mean(x["exp_gank"]),
                                 "realized_gank_bars": mean(x["gank_dmg"]), "death_rate": mean(x["died"])}
                             for k, x in fwd.items()},
        "per_1000_turns": {"ganks": 1000 * g["ganks"] / tu, "expected_gank_bars": 1000 * g["exp_gank"] / tu,
                           "realized_gank_bars": 1000 * g["gank_dmg"] / tu, "gank_deaths": 1000 * g["gank_deaths"] / tu,
                           "vision_established": 1000 * g["vision_set"] / tu,
                           "pressure_turns": 1000 * g["pressure_eff"] / tu, "plates": 1000 * g["plates"] / tu,
                           "expected_tower_shots": 1000 * shots["expected"] / tu,
                           "realized_tower_shots": 1000 * shots["realized"] / tu,
                           "expected_tower_shots_busy": 1000 * shots["expected_busy"] / tu,
                           "expected_tower_shots_not_busy": 1000 * shots["expected_not_busy"] / tu,
                           "dives": 1000 * sum(x["n"] for x in dives.values()) / tu,
                           "busy_dives": 1000 * dives["busy"]["n"] / tu,
                           "tower_adjacent_trades": 1000 * sum(adj.values()) / tu},
        "plate_turn": {f"plate {j + 1}": {"n": len(v), "median": float(np.median(v)) if v else float("nan")}
                       for j, v in enumerate(plate_turns)},
        "dives": {k: {"n": x["n"], "mean_tower_p": mean(x["tower_p"]), "shot_rate": mean(x["shot"]),
                      "diver_died_within_2": mean(x["died2"]), "dealt_bars": mean(x["dealt"]),
                      "taken_bars": mean(x["taken"])} for k, x in dives.items()},
        "tower_adjacent_trades": adj,
        "defender_reply_on_dive_turn": defender, "defender_reply_next_turn": defender_next,
    }


# ---------------------------------------------------------------------------------------------- 5. plate phases
def plate_phases(records: Sequence[GameRecord], focal: Optional[str] = None) -> dict:
    """Behaviour and income by the ACTUAL plate state of the simulation (never a turn cutoff): the focal side's
    attack phase "plates left on the enemy tower" / "enemy tower's plates exhausted", the defensive phase (plates left
    on the own tower), and the turn at which each side's attack plates ran out. Per phase: wave shares, Pressure /
    Vision choice when legal, crash rate, freeze-state share (held), income per turn by source."""
    keys = ("attack plates left", "attack plates exhausted", "own plates left", "own plates exhausted")
    acc = {k: {"w": [], "obj": [], "crash": [], "held": [], "farm": [], "plate": [], "bounty": [], "earned": [],
               "turns": 0} for k in keys}
    exhausted_at = []
    for rec, s in _units(records, focal):
        d = rec.data
        pres = d["present"][:, s] > 0
        left = d["plates_left"][:, s]
        own_left = d["own_plates_left"][:, s]
        z = np.flatnonzero(left == 0)
        exhausted_at.append(int(z[0]) + 1 if len(z) else None)
        for k, m in (("attack plates left", left > 0), ("attack plates exhausted", left == 0),
                     ("own plates left", own_left > 0), ("own plates exhausted", own_left == 0)):
            a = acc[k]
            a["turns"] += int(m.sum())
            mw = m & pres & (d["wave"][:, s] >= 0)
            a["w"].append(d["wave"][mw, s].astype(int))
            mo = m & pres & (d["obj_legal"][:, s] > 0)
            a["obj"].append(d["obj"][mo, s].astype(int))
            a["crash"].append((d["crash"][m, s] > 0).astype(float))
            a["held"].append(d["held"][m, s].astype(float))
            for f, src in (("farm", "farm"), ("plate", "plate_gold"), ("bounty", "bounty"), ("earned", "earned")):
                a[f].append(d[src][m, s])
    out = {}
    for k, a in acc.items():
        w = np.concatenate(a["w"]) if a["w"] else np.zeros(0, int)
        ob = np.concatenate(a["obj"]) if a["obj"] else np.zeros(0, int)
        cat = lambda key: np.concatenate(a[key]) if a[key] else np.zeros(0)
        n_ob = len(ob)
        out[k] = {"turns": a["turns"], "wave_turns": int(len(w)), "waves": _wave_shares(w),
                  "objective_legal_turns": n_ob,
                  "pressure_when_legal": float((ob == 1).mean()) if n_ob else float("nan"),
                  "vision_when_legal": float((ob == 2).mean()) if n_ob else float("nan"),
                  "crash_rate": float(cat("crash").mean()) if a["turns"] else float("nan"),
                  "freeze_state_share": float(cat("held").mean()) if a["turns"] else float("nan"),
                  **{f"{f}_per_turn": (float(cat(f).mean()) if a["turns"] else float("nan"))
                     for f in ("farm", "plate", "bounty", "earned")}}
    ex = [x for x in exhausted_at if x is not None]
    out["attack_plates_exhausted_at_turn"] = {"n": len(ex), "never": int(sum(x is None for x in exhausted_at)),
                                              "median": float(np.median(ex)) if ex else float("nan"),
                                              "quartiles": np.percentile(ex, [25, 75]).tolist() if ex else []}
    return out


# ---------------------------------------------------------------------------------------------- 6. recalls
LOW_HP = 0.4                                   # HP fraction of a "damaged" Recall (as zoning's low-HP recalls)
INVEST_LEAD = 5.0                              # resources invested ahead / behind (50 invested = x1.76 attack power)


def recalls(records: Sequence[GameRecord], focal: Optional[str] = None) -> dict:
    """Recalls, deaths and investment (M4.4b; added for the brief's Part 4.5, bounded to the existing mechanics).

    absence           a Recall on turn t or a death on turn t; the champion is back at the first later turn u with
                      `present` (start of the turn). Away turns = u - t - 1 (the Recall turn itself is not counted)
    per absence       HP fraction and unspent gold at t, investment bought (invested[u] - invested[t]), attack-power
                      ratio at t and at u (did it come back stronger relative to the enemy), the opponent's gold lead
                      over the absence (t+1 .. u-1) and the XP the absent side lost for good there; censored when the
                      record ends before u
    by investment lead  the focal side's behaviour by invested - enemy invested (ahead >= INVEST_LEAD, behind <=
                      -INVEST_LEAD, else level), both champions present: trade attempts, Shove and Safe / Yield shares,
                      mean cell, collection rate; a description of what a policy does with an item lead, not of how
                      the lead was obtained"""
    ev = {"recall_low_hp": [], "recall_other": [], "death": []}
    lead = {"ahead": {"n": 0, "trade": [], "shove": [], "concede": [], "pos": [], "collect": []},
            "level": {"n": 0, "trade": [], "shove": [], "concede": [], "pos": [], "collect": []},
            "behind": {"n": 0, "trade": [], "shove": [], "concede": [], "pos": [], "collect": []}}
    turns = 0
    for rec, s in _units(records, focal):
        d = rec.data
        o = 1 - s
        T = rec.turns
        turns += T
        pres = d["present"][:, s] > 0
        for t in np.flatnonzero((d["recalled"][:, s] > 0) | (d["died"][:, s] > 0)):
            kind = "death" if d["died"][t, s] > 0 else ("recall_low_hp" if d["hp_frac"][t, s] <= LOW_HP else "recall_other")
            back = np.flatnonzero(pres[t + 1:])
            u = int(t + 1 + back[0]) if len(back) else None
            row = {"hp_frac": float(d["hp_frac"][t, s]), "gold": float(d["gold"][t, s]),
                   "power_before": float(d["power"][t, s]), "censored": u is None}
            if u is not None:
                away = slice(t + 1, u)
                row.update({"away": u - t - 1, "bought": float(d["invested"][u, s] - d["invested"][t, s]),
                            "power_after": float(d["power"][u, s]),
                            "opp_lead_while_away": float((d["earned"][away, o] - d["earned"][away, s]).sum()),
                            "xp_missed_while_away": float(d["xp_missed"][away, s].sum())})
            ev[kind].append(row)
        both = pres & (d["en_present"][:, s] > 0) & (d["beh"][:, s] >= 0)
        diff = d["invested"][:, s] - d["einvested"][:, s]
        for t in np.flatnonzero(both):
            k = "ahead" if diff[t] >= INVEST_LEAD else ("behind" if diff[t] <= -INVEST_LEAD else "level")
            x = lead[k]
            x["n"] += 1
            x["trade"].append(float(d["beh"][t, s] == B_TRADE))
            x["shove"].append(float(d["wave"][t, s] == SH))
            x["concede"].append(float(d["beh"][t, s] in (B_SAFE, B_YIELD)))
            x["pos"].append(float(d["pos"][t, s]))
            if d["farm_avail"][t, s] > 0:
                x["collect"].append(float(d["farm"][t, s] / d["farm_avail"][t, s]))

    def mean(v):
        return float(np.mean(v)) if len(v) else float("nan")

    def summary(rows: List[dict]) -> dict:
        done = [r for r in rows if not r["censored"]]
        return {"n": len(rows), "per_1000_turns": 1000.0 * len(rows) / max(turns, 1), "censored": len(rows) - len(done),
                "hp_frac": mean([r["hp_frac"] for r in rows]), "unspent_gold": mean([r["gold"] for r in rows]),
                "away_turns": mean([r["away"] for r in done]), "bought": mean([r["bought"] for r in done]),
                "bought_any_share": mean([r["bought"] > 0 for r in done]),
                "power_before": mean([r["power_before"] for r in done]), "power_after": mean([r["power_after"] for r in done]),
                "power_gain_share": mean([r["power_after"] > r["power_before"] for r in done]),
                "opp_lead_while_away": mean([r["opp_lead_while_away"] for r in done]),
                "xp_missed_while_away": mean([r["xp_missed_while_away"] for r in done])}

    return {"focal_turns": turns, **{k: summary(v) for k, v in ev.items()},
            "by_investment_lead": {k: {"n": x["n"], "trade_rate": mean(x["trade"]), "shove_share": mean(x["shove"]),
                                       "safe_or_yield_share": mean(x["concede"]), "mean_pos": lane_mean(x["pos"]),
                                       "ward_share": ward_share(x["pos"]),
                                       "collection_rate": mean(x["collect"])} for k, x in lead.items()},
            "definition": {"low_hp_recall": f"hp fraction <= {LOW_HP}", "investment_lead": f"+-{INVEST_LEAD} resources"}}


# ---------------------------------------------------------------------------------------------- everything
def all_metrics(records: Sequence[GameRecord], focal: Optional[str] = None) -> dict:
    """Every metric family, separately for each start group (natural:s0, initialized:<scenario>)."""
    out = {}
    for group, recs in split_by_start(records).items():
        out[group] = {"games": len(recs), "turns": int(sum(r.turns for r in recs)),
                      "wave_management": wave_management(recs, focal), "freezes": freezes(recs, focal),
                      "zoning": zoning(recs, focal), "vision_tower": vision_tower(recs, focal),
                      "plate_phases": plate_phases(recs, focal), "recalls": recalls(recs, focal)}
    return out


__all__ = ["WAVE_STATES", "wave_management", "freeze_episodes", "freezes", "credible_threat", "zoning", "vision_tower",
           "plate_phases", "recalls", "all_metrics", "split_by_start", "power_class", "MIN_FREEZE", "SUSTAINED_FREEZE",
           "AFTER_WINDOW", "STRONGER", "WEAKER", "NEAR", "END_TYPES"]
