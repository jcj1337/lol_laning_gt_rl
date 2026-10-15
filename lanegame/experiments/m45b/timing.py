"""Priority F: the three movement and timing approximations left by the M4.5a correction, measured in play.

Every turn of a set of games (scripted pairings and uniform random legal play, DEV seeds) is analysed from the
resolver's own records:

A. Wave-probe accuracy. The resolver's probe (`resolve_turn_m4` step 1: full champion support, no minion targeting)
   is recomputed exactly and compared with the authoritative lane step: per side, does the front after each tick
   differ (cell or presence)? does the end-of-turn farm spot differ from the authoritative end front (clipped to the
   lane)? For followers (Hold, an unthreatened Yield, Safe) on such turns: positional fire taken, and the farm they
   could not collect (enemy minion HP that died out of their reach) against matched turns.
B. Crossing between ticks. A champion o and the enemy front SWAP cells within one tick (o at c and the enemy front
   at c + 1 before the tick, o at c + 1 and the front at c after it, in o's coordinates) while o's own front is behind
   or absent: the configuration positional aggro would have caught had the crossing been occupancy. Counted per
   champion-turn, with the firepower the front had on that tick (the units a swept-path rule could have fired) and the
   behaviour of the crossing champion.
C. Trade-contact timing. Every connected Trade is classified by when its contact first exists on the champions'
   tick paths: at the start of the turn, after tick 0, or only at the end (a full-speed chase). The exchange is
   attributed to tick 0 in every case; under a contact-timed exchange a trade whose contact exists only at the end
   would land on tick 1 and draw no retaliation with delay 1. Reported: shares of each class, landed rate, and the
   retaliation units and centi-HP that only the end-contact trades drew.
D. Wave consequences of minion fire on champions (Priority B): fire ticks, the share in a lane fight, the share that
   kept an otherwise-dying enemy front alive on that tick, and fire while the attacker's mass stood under the tower.
E. Probe error sources and a two-pass correction (`probe_decomposition`).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ...champions import N_CHANCE, NO_REACH, _decode, champion_legal_actions, in_contact
from ...champions_m4 import context, resolve_turn_m4
from ...config import B_ABSENT, B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, NONE, VISION
from ...mechanics import minion_damage, step_turn, _blob_hp
from ...positional import WARD, dist as graph_dist, mirror_cell, minion_damage_centi
from . import common as Cm
from . import engine as E
from . import states as S
from . import strategies as ST

PAIRINGS = (("g_farm", "g_farm"), ("g_shove", "g_farm"), ("g_shove", "g_shove"), ("g_bully", "g_farm"),
            ("g_guard", "g_shove"), ("g_shove_guard", "g_shove"), ("g_ward", "g_bully"), ("g_shove_ward", "g_shove"),
            ("fz_guard_sh", "wk_shove_sh"), ("g_wary", "g_bully"), ("g_safe", "g_shove"), ("g_slow_guard", "g_shove"))


def probe(cfg, lane, m4, k, a, b):
    """The resolver's step-1 probe, recomputed exactly (same codes, reach and budgets; no minion targets)."""
    ctx = context(cfg)
    t = ctx.t
    T = cfg.tower
    reach_cap = T + (cfg.natural_tower_reach if cfg.natural_waves else 0)
    (wa, ba, oa), (wb, bb, ob) = _decode(a), _decode(b)
    w, beh, obj = (wa, wb), (ba, bb), (oa, ob)
    lv = [t.level_by_xp[min(m4.side(s).xp, t.xp_cap)] for s in (0, 1)]
    it = [m4.side(s).items for s in (0, 1)]
    warding = tuple(cfg.vision_rule == "ward" and obj[s] == VISION for s in (0, 1))
    lane_obj = tuple(NONE if warding[s] else obj[s] for s in (0, 1))
    off = tuple(beh[s] in (B_RECALL, B_ABSENT) or warding[s] for s in (0, 1))
    budget = tuple(ctx.state_budget(w[s], ctx.capacity(lv[s], it[s]), lane, k, s) for s in (0, 1))
    return step_turn(lane, k, w[0] + 6 * lane_obj[0], w[1] + 6 * lane_obj[1], cfg,
                     champion_reach=tuple(NO_REACH if off[s] else reach_cap for s in (0, 1)), champion_budget=budget)


class ProbeOverride:
    """A lane step whose FIRST call of a turn (the resolver's probe) returns `override` (a TurnResult whose state,
    crash_hp, fronts_by_tick and crashed_by_tick stand for the probe); every later call is the normal lane step."""

    def __init__(self, cfg, override):
        self.inner, self.override, self.used = E.lane_cache(cfg), override, False

    def __call__(self, s, kk, x, y, reach, budget=None, targets=None):
        if not self.used:
            self.used = True
            return self.override
        return self.inner(s, kk, x, y, reach, budget, targets)


def probe_decomposition(cfg, lane, m4, k, a, b, t, u, acc) -> None:
    """E. Sources of the probe error and the materiality of a two-pass correction.
    (i) targeting only: the authoritative lane step re-run with the same codes, reach and budgets but no minion
        targets; its per-tick fronts against the authoritative ones isolate the M4.5a targeting approximation;
    (ii) two-pass: the turn re-resolved with the authoritative lane step's result standing in for the probe (the
        champions then follow the fronts that actually happen), same actions and chance. The changes of the end cells,
        rewards, farm, XP and positional fire measure what a probe-correcting resolution would change; a third pass
        tells whether that correction is a fixed point."""
    res = t.lane
    notarget = step_turn(lane, k, t.lane_codes[0], t.lane_codes[1], cfg, champion_reach=t.reach,
                         champion_budget=t.wave_budget)
    for s in (0, 1):
        if t.away[s]:
            continue
        acc["target_tick_diff"] += any(notarget.fronts_by_tick[i][s] != res.fronts_by_tick[i][s]
                                       for i in range(cfg.ticks_per_turn))
    ctx = context(cfg)
    t2 = resolve_turn_m4(lane, m4, k, a, b, cfg, lane_step=ProbeOverride(cfg, res), u=u, ctx=ctx)
    t3 = resolve_turn_m4(lane, m4, k, a, b, cfg, lane_step=ProbeOverride(cfg, t2.lane), u=u, ctx=ctx)
    acc["twopass_turns"] += 1
    acc["twopass_end_diff"] += sum(t2.end[s] != t.end[s] for s in (0, 1))
    acc["twopass_reward_absdiff"] += abs(t2.reward - t.reward)
    acc["twopass_farm_absdiff"] += sum(abs(t2.farm[s] - t.farm[s]) for s in (0, 1))
    acc["twopass_farm_gain"] += sum(t2.farm[s] - t.farm[s] for s in (0, 1))
    acc["twopass_xp_absdiff"] += sum(abs(t2.xp_gain[s] - t.xp_gain[s]) for s in (0, 1))
    pos = lambda tt, s: sum(1 for x in (tt.minion_fire[s] if tt.minion_fire else ()) if x == "positional")
    acc["twopass_pos_ticks_1"] += sum(pos(t, s) for s in (0, 1))
    acc["twopass_pos_ticks_2"] += sum(pos(t2, s) for s in (0, 1))
    acc["twopass_not_fixed"] += any(t3.end[s] != t2.end[s] for s in (0, 1))
    acc["twopass_state_diff"] += (t2.lane.state != t.lane.state) or (t2.champ != t.champ)


def _front_cells(lane, res):
    """Per tick (0 = start of the turn, then after each tick): each side's front cell (None: no front)."""
    start = tuple(lane.side(s)[0] if lane.side(s)[1] > 0 else None for s in (0, 1))
    return (start,) + tuple(res.fronts_by_tick)


def analyse_turn(cfg, lane, m4, k, a, b, t, acc: Dict[str, float]) -> None:
    T = cfg.tower
    res = t.lane
    pr = probe(cfg, lane, m4, k, a, b)
    # ---- A. probe accuracy
    for s in (0, 1):
        if t.away[s]:
            continue
        acc["turns"] += 1
        tick_diff = any(pr.fronts_by_tick[i][s] != res.fronts_by_tick[i][s] for i in range(cfg.ticks_per_turn))
        fx, fh = res.state.side(s)[:2]
        auth_spot = max(-T, min(T, fx)) if fh > 0 else None
        pfx, pfh = pr.state.side(s)[:2]
        probe_spot = max(-T, min(T, pfx)) if pfh > 0 else None
        spot_diff = auth_spot != probe_spot
        follower = t.behaviors[s] in (B_HOLD, B_SAFE) or (t.behaviors[s] == B_YIELD and not t.threatened[s])
        follower = follower and t.objectives[s] == NONE and not t.warding[s]
        acc["probe_tick_diff"] += tick_diff
        acc["probe_spot_diff"] += spot_diff
        if follower:
            acc["follower_turns"] += 1
            pos = sum(1 for x in (t.minion_fire[s] if t.minion_fire else ()) if x == "positional")
            unc = t.farm_available[s] - t.farm[s]
            if spot_diff:
                acc["follower_spot_diff"] += 1
                acc["follower_pos_ticks_diff"] += pos
                acc["follower_unc_diff"] += unc
            else:
                acc["follower_pos_ticks_same"] += pos
                acc["follower_unc_same"] += unc
    # ---- A'. farm out of reach: enemy minion HP that died on s's ledger farther than s's reach from its end cell
    #      (present, not channelling or warding), the farm a better-placed champion could have collected
    from ...champions import AWAY, CHANNEL, RETREAT
    from ...champions_m4 import WARD_MODE
    for s in (0, 1):
        if t.modes[s] in (AWAY, CHANNEL, WARD_MODE):
            continue
        rch = cfg.retreat_farm_reach if t.modes[s] == RETREAT else cfg.farm_reach
        acc["oor_farm"] += cfg.farm_per_hp * sum(hp for x, hp in res.kills[s].items()
                                                 if graph_dist(x, t.end[s]) > rch)
        acc["avail_farm"] += t.farm_available[s]
        acc["collected_farm"] += t.farm[s]
    # ---- B. crossings
    fronts = _front_cells(lane, res)
    path = t.path
    for o in (0, 1):
        if t.away[o] or path is None:
            continue
        s = 1 - o
        cells = (t.start[o],) + tuple(path[o])
        for i in range(cfg.ticks_per_turn):
            c0, c1 = cells[i], cells[i + 1]
            ef0, ef1 = fronts[i][s], fronts[i + 1][s]
            if c0 == WARD or c1 == WARD or ef0 is None or ef1 is None:
                continue
            e0, e1 = -ef0, -ef1                              # the enemy front in o's coordinates
            own1 = fronts[i + 1][o]
            if c1 == c0 + 1 and e0 == c0 + 1 and e1 == c0:   # o stepped forward through the oncoming front
                uncovered = own1 is None or own1 < c1
                acc["cross"] += 1
                acc["cross_uncovered"] += uncovered
                if uncovered:
                    acc["cross_beh_" + str(t.behaviors[o])] += 1
                    # the front's firepower on this tick from its HP after the tick's fight (a slight underestimate
                    # of the units a swept-path rule could have fired)
                    tr = res.trace[i] if res.trace else None
                    if tr:
                        P = tr["B" if s == 1 else "A"]
                        u = minion_damage(_blob_hp(P, cfg), cfg, i) if P else 0
                        lvl = t.levels_before[o]
                        acc["cross_units"] += u
                        acc["cross_centi"] += minion_damage_centi(u, lvl, cfg.minion_damage_alpha,
                                                                  cfg.minion_damage_growth) if u else 0
            elif c1 == c0 - 1 and e0 == c0 - 1 and e1 == c0:  # o stepped back through a front walking toward it
                acc["cross_back"] += 1
    # ---- D. wave consequences of minion fire on champions (Priority B): per tick on which side s's wave fired at
    #      champion o: in a lane fight, did o's front survive only because s's minions fired at o (HP before the tick
    #      above the damage it took but not above that damage plus the diverted units)? with o's crashed mass under
    #      s's tower and no siege fight that tick, the defender fired at a champion while the attacker's mass stood under its tower (no siege fight that tick)
    for tick in res.trace or ():
        mf = tick.get("minion_fire")
        if not mf:
            continue
        for o in (0, 1):
            cause, units = mf[o]
            if not cause or units <= 0:
                continue
            s = 1 - o
            acc["fire_ticks"] += 1
            acc["fire_units"] += units
            fight = tick.get("fight")
            if fight is not None:
                hp_o, dmg_o = (fight[1], fight[4]) if o == 0 else (fight[3], fight[5])
                acc["fire_in_fight"] += 1
                if dmg_o < hp_o <= dmg_o + units:
                    acc["wave_preserved"] += 1
            elif tick["siege"][o] > 0 and tick.get("siege_fight") is None:
                acc["fire_while_besieged"] += 1
    # ---- C. trade-contact timing
    for s in (0, 1):
        if not t.connected[s]:
            continue
        o = 1 - s
        acc["conn"] += 1
        landed = t.trade_damage[s] > 0
        acc["conn_landed"] += landed
        if in_contact(t.start[s], t.start[o], cfg):
            cls = "start"
        elif path is not None and in_contact(path[s][0], path[o][0], cfg):
            cls = "tick0"
        else:
            cls = "end"
        acc["conn_" + cls] += 1
        acc["landed_" + cls] += landed
        ret = sum(1 for x in (t.minion_fire[s] if t.minion_fire else ()) if x == "retaliation")
        if ret and res.trace:
            units = sum(tr["minion_fire"][s][1] for tr in res.trace if tr.get("minion_fire")
                        and tr["minion_fire"][s][0] == "retaliation")
            acc["ret_units_" + cls] += units
            acc["ret_centi_" + cls] += minion_damage_centi(units, t.levels_before[s], cfg.minion_damage_alpha,
                                                           cfg.minion_damage_growth)


def _job(args):
    over, kind, pa, pb, seeds, turns = args
    c = Cm.cfg("r2", **dict(over))
    from collections import defaultdict
    acc = defaultdict(float)
    ls = E.lane_cache(c)
    ctx = context(c)
    for sd in seeds:
        lane, m4, k0 = S.s0(c)
        rng = np.random.default_rng(sd)
        prng = np.random.default_rng([sd, 99])
        for i in range(turns):
            u = rng.random(N_CHANCE)
            if kind == "random":
                a = int(prng.choice(champion_legal_actions(lane, m4, 0, c)))
                b = int(prng.choice(champion_legal_actions(lane, m4, 1, c)))
            else:
                a = E.decide(ST.get(pa), c, lane, m4, 0, i)
                b = E.decide(ST.get(pb), c, lane, m4, 1, i)
            t = resolve_turn_m4(lane, m4, k0 + i, a, b, c, lane_step=ls, u=u, ctx=ctx)
            analyse_turn(c, lane, m4, k0 + i, a, b, t, acc)
            probe_decomposition(c, lane, m4, k0 + i, a, b, t, tuple(u), acc)
            lane, m4 = t.lane.state, t.champ
    return {"kind": kind, "pair": [pa, pb], "acc": dict(acc)}


def run(over=(), seeds=Cm.DEV[:16], turns: int = 300, workers: int = Cm.WORKERS) -> dict:
    jobs = [(tuple(over), "scripted", a, b, tuple(seeds), turns) for a, b in PAIRINGS]
    jobs += [(tuple(over), "random", "", "", tuple(seeds[i::4]), turns) for i in range(4)]
    res = Cm.run_jobs(_job, jobs, workers, "timing")
    tot: Dict[str, Dict[str, float]] = {"scripted": {}, "random": {}}
    for r in res:
        d = tot[r["kind"]]
        for k, v in r["acc"].items():
            d[k] = d.get(k, 0.0) + v
    return {"per_pairing": [r for r in res if r["kind"] == "scripted"], "totals": tot}


__all__ = ["run", "probe", "analyse_turn", "PAIRINGS"]
