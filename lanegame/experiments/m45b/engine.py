"""The M4.5b paired game loop, the per-side ledger, state collection and paired branching.

One game: from a start (lane, champions, k0), both seats' policies decide from the same start-of-turn state
(simultaneous moves), the turn is resolved by `champions_m4.resolve_turn_m4` (the environment's resolver, with a
lane-step cache and tick traces), and the chance vector of turn i is the i-th draw of N_CHANCE uniforms from
numpy default_rng(seed), exactly as `LaneEnv.reset(seed=seed)` draws it whatever the actions. Two games with the same
seed and start therefore see the same chance stream turn by turn (common random numbers), and `viewer.session.play`
reproduces a game from its Setup.

A policy is any object with `decide(cfg, lane, m4, s, i) -> (code, why)` (the M4.3c `Strategy` interface; i = the turn
index from the game's start); its code must be legal (checked).

Ledger (`Ledger.summary`), per side s in {A, B} (suffix _A / _B), over the whole game unless a horizon is given:

  gold      earned (farm + plate gold + bounty), farm, plate, bounty, farm_avail (0.125 per enemy minion HP that died
            on s's ledger, whatever s's position: the farm at full collection), unc = farm_avail - farm, unc_away = the
            part of unc on turns s was away (M4.6b; unc - unc_away = lost while present: reach, posture, trading)
  XP        xp (credited), xp_avail (every enemy minion HP that died), xp_lost (avail - credited - discarded: died
            out of range or while away)
  damage    incoming, in bars of the victim's start-of-turn max HP, BEFORE the victim's durability division (the
            division applies to the turn's sum, so the parts are exact up to that common factor): d_trade (the enemy's
            trade), d_min (minions; d_min_ret retaliation, d_min_pos positional aggro, split by the units of each tick),
            d_tower, d_gank; d_total = the HP actually lost (after the division)
  minions   u_ret / u_pos minion damage units by cause; t_ret / t_pos ticks with fire by cause
  combat    trade_att, trade_conn, trade_land (positive damage dealt), trade_dealt (bars of the victim's max HP,
            before its durability), evaded (a threatened Yield / Safe that moved out of contact), pinned
  absence   deaths (d_gank / d_trade / d_minion / d_tower: the largest incoming source that turn), recalls,
            rec_intr (interrupted Recalls), away (turns away)
  Vision    wards (Ward chosen), ward_placed, ward_intr, vis_turns (turns played with the own vision active),
            vis_exposed (... at an exposure cell >= 0), ganks, gank_p (sum of the gank probabilities), gank_dmg (bars)
  waves     held (turns after which s holds a freeze: the enemy front at s's own -1 or deeper, both fronts fighting),
            frozen (turns after which the enemy holds one on s), crashes, crash_hp, tower (centi-units of damage
            dealt to the enemy tower), plates, capped (connected Trades with a capped wave budget)
  actions   n_<behaviour> (Hold / Yield / Trade / Safe / Recall turns), n_<wave> (LH / TN / SL / SH turns), at_W
  end       hp_end (bar at the end; 1.0 while away), level_end, invested_end, xp_end

snapshots of SNAP_KEYS at the horizons SNAP_H as <key>_<side>@<h> (and xp_lead@<h>), and for seat A: G<h> = sum_{t < h} 0.99^t r_t (the reward to A, resources; the RL objective's discounted lead) at
h in HORIZONS, lead = the undiscounted gold lead, xp_lead, and the freeze bookkeeping of A's freeze held at the start
(held_run0: consecutive held turns from the start; 0 if A does not hold one at the start).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...champions import N_CHANCE, champion_legal_actions
from ...champions_m4 import M4State, context, resolve_turn_m4
from ...config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, F, SH, SP, W_SLOWPUSH, GameConfig)
from ...mechanics import step_turn
from ...positional import WARD
from ...state import State
from ..rl.m44.records import holds_freeze
from .common import GAMMA

A, B = 0, 1
HORIZONS = (1, 5, 10, 20, 50, 100, 200, 300)
SNAP_H = (10, 20, 50)              # horizons of the cumulative snapshots (XP is capped at level 18 in long windows)
SNAP_KEYS = ("earned", "farm", "xp", "deaths", "recalls", "away", "d_total", "trade_dealt", "d_min", "held",
             "frozen", "ward_placed", "plates")
SIDE_KEYS = ("earned", "farm", "plate", "bounty", "farm_avail", "unc", "unc_away", "xp", "xp_avail", "xp_lost",
             "d_trade", "d_min", "d_min_ret", "d_min_pos", "d_tower", "d_gank", "d_total",
             "u_ret", "u_pos", "t_ret", "t_pos",
             "trade_att", "trade_conn", "trade_land", "trade_dealt", "evaded", "pinned",
             "deaths", "d_by_gank", "d_by_trade", "d_by_minion", "d_by_tower", "recalls", "rec_intr", "away",
             "wards", "ward_placed", "ward_intr", "vis_turns", "vis_exposed", "ganks", "gank_p", "gank_dmg",
             "held", "frozen", "crashes", "crash_hp", "tower", "plates", "capped",
             "n_hold", "n_yield", "n_trade", "n_safe", "n_recall", "n_LH", "n_TN", "n_SL", "n_SH", "at_W",
             "hp_end", "level_end", "invested_end", "xp_end")
BEH_KEY = {B_HOLD: "n_hold", B_YIELD: "n_yield", B_TRADE: "n_trade", B_SAFE: "n_safe", B_RECALL: "n_recall"}
WAVE_KEY = {SP: "n_LH", F: "n_TN", W_SLOWPUSH: "n_SL", SH: "n_SH"}
Start = Tuple[State, M4State, int]


class LaneCache:
    """The environment's lane-step memo (`LaneEnv._lane_turn`) with tick traces on (minion fire per tick)."""

    def __init__(self, cfg: GameConfig, size: int = 40_000):
        self.cfg, self.size, self.d = cfg, size, OrderedDict()

    def __call__(self, s, kk, x, y, reach, budget=None, targets=None):
        cfg = self.cfg
        key = (s, kk % cfg.phase_period, x, y, reach, budget, targets)
        r = self.d.get(key)
        if r is None:
            r = step_turn(s, kk, x, y, cfg, record=True, champion_reach=reach, champion_budget=budget,
                          champion_targets=targets)
            self.d[key] = r
            if len(self.d) > self.size:
                self.d.popitem(last=False)
        return r


_CACHES: Dict[GameConfig, LaneCache] = {}


def lane_cache(cfg: GameConfig) -> LaneCache:
    c = _CACHES.get(cfg)
    if c is None:
        if len(_CACHES) > 6:
            _CACHES.clear()
        c = _CACHES[cfg] = LaneCache(cfg)
    return c


def decide(pol, cfg, lane, m4, s: int, i: int) -> int:
    if hasattr(pol, "decide"):
        return int(pol.decide(cfg, lane, m4, s, i)[0])
    return int(pol(cfg, lane, m4, s, i))


from ...diagnostics.lane_exit import observe_turn  # noqa: E402,F401  (M5 opt-in hook; canonical in diagnostics)


class Ledger:
    """Accumulates the per-side ledger of the module docstring turn by turn."""

    def __init__(self, cfg: GameConfig, start: Start):
        self.cfg, self.ctx = cfg, context(cfg)
        self.v = {k: [0.0, 0.0] for k in SIDE_KEYS}
        self.rewards: List[float] = []
        self.held_run0 = 0
        self.held0 = holds_freeze(start[0], A)
        self._run_open = self.held0
        self.turns = 0
        self.snaps: Dict[int, Dict[str, Tuple[float, float]]] = {}

    def add(self, t, lane_before: State) -> None:
        cfg, tb, v = self.cfg, self.ctx.t, self.v
        sc = cfg.gold_scale
        res = t.lane
        after = res.state
        lv = t.levels_before
        fire_units = [[0, 0], [0, 0]]                    # per champion: retaliation, positional units
        fire_ticks = [[0, 0], [0, 0]]
        if res.trace is not None:
            for tick in res.trace:
                mf = tick.get("minion_fire")
                if not mf:
                    continue
                for s in (A, B):
                    cause, units = mf[s]
                    if units > 0 and cause:
                        j = 0 if cause == "retaliation" else 1
                        fire_units[s][j] += units
                        fire_ticks[s][j] += 1
        for s in (A, B):
            o = 1 - s
            H = tb.hp_max[lv[s]]
            v["earned"][s] += t.earned_units[s] / sc
            v["farm"][s] += t.farm_units[s] / sc
            v["plate"][s] += t.plate_units[s] / sc
            v["bounty"][s] += t.bounty_units[s] / sc
            v["farm_avail"][s] += t.farm_available[s]
            v["unc"][s] += t.farm_available[s] - t.farm_units[s] / sc
            if t.away[s]:
                v["unc_away"][s] += t.farm_available[s] - t.farm_units[s] / sc
            xp_av = cfg.xp_per_hp * sum(res.kills[s].values())
            v["xp"][s] += t.xp_gain[s]
            v["xp_avail"][s] += xp_av
            v["xp_lost"][s] += xp_av - t.xp_gain[s] - t.xp_discarded[s]
            d_tr, d_mi, d_to, d_ga = (t.trade_damage[o] / H, t.minion_damage[s] / H, t.tower_damage[s] / H,
                                      t.gank_damage[s] / H)
            v["d_trade"][s] += d_tr
            v["d_min"][s] += d_mi
            u = sum(fire_units[s])
            if u > 0:
                v["d_min_ret"][s] += d_mi * fire_units[s][0] / u
                v["d_min_pos"][s] += d_mi * fire_units[s][1] / u
            v["d_tower"][s] += d_to
            v["d_gank"][s] += d_ga
            v["d_total"][s] += t.damage_taken[s] / H
            v["u_ret"][s] += fire_units[s][0]
            v["u_pos"][s] += fire_units[s][1]
            v["t_ret"][s] += fire_ticks[s][0]
            v["t_pos"][s] += fire_ticks[s][1]
            v["trade_att"][s] += t.behaviors[s] == B_TRADE
            v["trade_conn"][s] += t.connected[s]
            v["trade_land"][s] += t.connected[s] and t.trade_damage[s] > 0
            v["trade_dealt"][s] += t.trade_damage[s] / tb.hp_max[lv[o]]
            v["evaded"][s] += t.threatened[s]
            v["pinned"][s] += t.pinned[s]
            if t.died[s]:
                v["deaths"][s] += 1
                src = max((d_ga, "d_by_gank"), (d_tr, "d_by_trade"), (d_mi, "d_by_minion"), (d_to, "d_by_tower"))
                v[src[1]][s] += 1
            v["recalls"][s] += t.recalled[s]
            v["rec_intr"][s] += t.interrupted[s]
            v["away"][s] += t.away[s]
            v["wards"][s] += t.warding[s]
            v["ward_placed"][s] += t.ward_placed[s]
            v["ward_intr"][s] += t.ward_interrupted[s]
            if not t.away[s] and lane_before.vision_of(s) > 0:
                v["vis_turns"][s] += 1
                v["vis_exposed"][s] += t.exposure[s] >= 0
            v["ganks"][s] += t.ganked[s]
            v["gank_p"][s] += t.gank_p[s]
            v["gank_dmg"][s] += d_ga
            v["held"][s] += holds_freeze(after, s)
            v["frozen"][s] += holds_freeze(after, o)
            v["crashes"][s] += res.crash_hp[s] > 0
            v["crash_hp"][s] += res.crash_hp[s]
            v["tower"][s] += t.tower_damage_dealt[s]
            v["plates"][s] += t.plates_broken[s]
            v["capped"][s] += t.trade_wave_capped[s]
            if not t.away[s]:
                k = BEH_KEY.get(t.behaviors[s])
                if k:
                    v[k][s] += 1
                if t.behaviors[s] != B_RECALL and not t.warding[s]:
                    v[WAVE_KEY[t.waves[s]]][s] += 1
                v["at_W"][s] += t.end[s] == WARD
        self.rewards.append(t.reward)
        if self._run_open:
            if holds_freeze(after, A):
                self.held_run0 += 1
            else:
                self._run_open = False
        self.turns += 1
        if self.turns in SNAP_H:
            self.snaps[self.turns] = {k: (self.v[k][0], self.v[k][1]) for k in SNAP_KEYS}

    def finish(self, m4: M4State) -> None:
        tb = self.ctx.t
        sc = self.cfg.gold_scale
        for s in (A, B):
            me = m4.side(s)
            L = tb.level_by_xp[min(me.xp, tb.xp_cap)]
            self.v["hp_end"][s] = 1.0 if me.absent else me.hp / tb.hp_max[L]
            self.v["level_end"][s] = L
            self.v["invested_end"][s] = me.items / sc if self.cfg.continuous_items else me.items
            self.v["xp_end"][s] = me.xp

    def summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, (a, b) in self.v.items():
            out[k + "_A"] = float(a)
            out[k + "_B"] = float(b)
        r = np.asarray(self.rewards, dtype=float)
        disc = GAMMA ** np.arange(len(r))
        cr = np.cumsum(r * disc)
        for h in HORIZONS:
            if h <= len(r):
                out[f"G{h}"] = float(cr[h - 1])
        out["lead"] = float(r.sum())
        out["xp_lead"] = out["xp_A"] - out["xp_B"]
        out["held0"] = float(self.held0)
        out["held_run0"] = float(self.held_run0)
        out["turns"] = float(self.turns)
        for h, d in self.snaps.items():
            for k, (a, b) in d.items():
                out[f"{k}_A@{h}"] = float(a)
                out[f"{k}_B@{h}"] = float(b)
            out[f"xp_lead@{h}"] = float(d["xp"][0] - d["xp"][1])
        return out


def play(cfg: GameConfig, start: Start, pa, pb, seed: int, turns: int, keep_states: bool = False,
         keep_rows: bool = False, check_legal: bool = True, stop: Optional[Callable] = None) -> dict:
    """One game (module docstring). Returns {"summary": ledger summary} plus, on request, "states" (the
    (lane, champions, k) at the start of every turn and the final one) and "rows" (per-turn compact records: actions,
    rewards, HP fractions, cells, freeze flags). `stop(t, lane, m4, i) -> bool` ends the game after turn i."""
    lane, m4, k0 = start
    ctx = context(cfg)
    tb = ctx.t
    ls = lane_cache(cfg)
    rng = np.random.default_rng(seed)
    led = Ledger(cfg, start)
    states = [(lane, m4, k0)] if keep_states else None
    rows = [] if keep_rows else None
    for i in range(turns):
        u = rng.random(N_CHANCE)
        a = decide(pa, cfg, lane, m4, A, i)
        b = decide(pb, cfg, lane, m4, B, i)
        if check_legal:
            for s, c in ((A, a), (B, b)):
                if c not in champion_legal_actions(lane, m4, s, cfg):
                    raise ValueError(f"seat {'AB'[s]} chose illegal action {c} at turn {i} "
                                     f"(legal {champion_legal_actions(lane, m4, s, cfg)})")
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=ls, u=u, ctx=ctx)
        observe_turn(pa, cfg, lane, m4, k0 + i, (a, b))
        observe_turn(pb, cfg, lane, m4, k0 + i, (a, b))
        led.add(t, lane)
        if rows is not None:
            rows.append(row_of(t, lane, m4, tb))
        lane, m4 = t.lane.state, t.champ
        if states is not None:
            states.append((lane, m4, k0 + i + 1))
        if stop is not None and stop(t, lane, m4, i):
            break
    led.finish(m4)
    out = {"summary": led.summary()}
    if states is not None:
        out["states"] = states
    if rows is not None:
        out["rows"] = rows
    return out


def hp_frac(m4: M4State, s: int, tb) -> float:
    me = m4.side(s)
    if me.absent:
        return 1.0
    return me.hp / tb.hp_max[tb.level_by_xp[min(me.xp, tb.xp_cap)]]


def row_of(t, lane: State, m4: M4State, tb) -> dict:
    """A compact per-turn record (both seats)."""
    after = t.lane.state
    return {"a": t.waves, "beh": t.behaviors, "r": t.reward, "hp0": (hp_frac(m4, A, tb), hp_frac(m4, B, tb)),
            "start": t.start, "end": t.end, "conn": t.connected, "land": (t.trade_damage[0] > 0, t.trade_damage[1] > 0),
            "held": (holds_freeze(after, A), holds_freeze(after, B)), "held0": (holds_freeze(lane, A),
                                                                                  holds_freeze(lane, B)),
            "died": t.died, "rec": t.recalled, "away": t.away, "returned": t.returned, "mfire": t.minion_fire,
            "mcp": t.minion_damage, "earned": t.earned_units, "ward": t.warding, "placed": t.ward_placed}


# ---------------------------------------------------------------------------------------------- composite policies
class Then:
    """`first` for the first `n` turns of the game (turn index i < n), then `after`; optionally `first` only while
    `cond(cfg, lane, m4, s)` holds (a plan with a stop condition: once it fails, `after` for good is approximated by
    the memoryless rule "first while i < n and cond holds")."""

    def __init__(self, name: str, first, after, n: int, cond: Optional[Callable] = None):
        self.name, self.first, self.after, self.n, self.cond = name, first, after, int(n), cond

    def decide(self, cfg, lane, m4, s: int, i: int):
        if i < self.n and (self.cond is None or self.cond(cfg, lane, m4, s)):
            c, why = self.first.decide(cfg, lane, m4, s, i) if hasattr(self.first, "decide") else \
                (self.first(cfg, lane, m4, s, i), "")
            legal = champion_legal_actions(lane, m4, s, cfg)
            if c in legal:
                return c, f"[plan {self.name}] {why}"
        return self.after.decide(cfg, lane, m4, s, i)

    def act(self, cfg, lane, m4, s: int, i: int) -> int:
        return self.decide(cfg, lane, m4, s, i)[0]

    def inner_policies(self):
        return (self.first, self.after)


class Fixed:
    """A fixed action code when legal, else `fallback` (a policy)."""

    def __init__(self, name: str, code: int, fallback):
        self.name, self.code, self.fallback = name, int(code), fallback

    def decide(self, cfg, lane, m4, s: int, i: int):
        if self.code in champion_legal_actions(lane, m4, s, cfg):
            return self.code, f"fixed {cfg.label(self.code)}"
        return self.fallback.decide(cfg, lane, m4, s, i)

    def act(self, cfg, lane, m4, s: int, i: int) -> int:
        return self.decide(cfg, lane, m4, s, i)[0]

    def inner_policies(self):
        return (self.fallback,)


# ---------------------------------------------------------------------------------------------- state collection
def collect(cfg: GameConfig, start: Start, pa, pb, seeds: Sequence[int], turns: int,
            predicate: Callable, per_game: int = 2, min_turn: int = 4, rng_seed: int = 0) -> List[dict]:
    """Decision states of games of (pa, pb): every (turn, seat) whose START-OF-TURN state satisfies
    `predicate(cfg, lane, m4, seat) -> bool` (decision state only, never outcomes), at most `per_game` per game chosen
    by a seeded shuffle. Returns [{"start": (lane, m4, k), "seat", "turn", "seed"}]; seat B states are mirrored to
    seat A (the focal side is always A)."""
    rng = np.random.default_rng(rng_seed)
    out = []
    for sd in seeds:
        g = play(cfg, start, pa, pb, sd, turns, keep_states=True)
        cand = [(i, s) for i, (lane, m4, k) in enumerate(g["states"][:-1]) if i >= min_turn for s in (A, B)
                if predicate(cfg, lane, m4, s)]
        rng.shuffle(cand)
        for i, s in cand[:per_game]:
            lane, m4, k = g["states"][i]
            if s == B:
                lane, m4 = lane.mirror(), m4.mirror()
            out.append({"start": (lane, m4, k), "seat": s, "turn": i, "seed": sd})
    return out


def branch(cfg: GameConfig, start: Start, candidates: Dict[str, object], opponent, seeds: Sequence[int],
           turns: int) -> Dict[str, List[Dict[str, float]]]:
    """Paired rollouts from one start: candidate policy c for seat A against `opponent` for seat B, rollout m with
    seed seeds[m] for every candidate (common random numbers). Returns {candidate: [summary per rollout]}."""
    return {name: [play(cfg, start, pol, opponent, sd, turns)["summary"] for sd in seeds]
            for name, pol in candidates.items()}


__all__ = ["HORIZONS", "SIDE_KEYS", "Ledger", "LaneCache", "lane_cache", "play", "collect", "branch", "Then", "Fixed",
           "hp_frac", "row_of", "decide"]
