"""Deterministic worked examples and tables of the reference, computed with the engine on `m45_calibrated()`.

  tables        progression (XP to the next level, cumulative XP, max HP, HP unit, attack power), the investment
                multiplier M(S), and the wave capacity C(L, S) with the Thin / Slow Push cap round(C / 2)
  wave_actions  the `neutral` start of M4.5b (`m45b.states.vision_state`: both 8-HP fronts and both champions at the
                lane cell, a rear packet at each tower, level 3, nothing invested, turn 9) played for 8 turns with A on
                Last Hit, Thin, Slow Push or Shove (Hold position) against B on Last Hit + Hold; no chance events
                (u = None: no gank, trade roll or tower shot). Minion fire on champions is deterministic and is played
  wave_actions_vs_shove  the same four A actions against B on Shove + Hold (totals only)
  windows       window strengths and the Thin / Slow Push needs of three start-of-turn lanes
  freeze_hold   turns (of 40) for which a freeze holds from the M4.5b freeze start (`m45b.states.freeze_state`), by
                level, power gap, held enemy front and the freezer's rule (the threshold controller of §4.8 or Thin +
                Hold), the frozen-on champion on Last Hit + Hold at its +1, no chance events
  retaliation   a level-2 Trade that lands next to a 16-HP enemy front, the trader walking one or two edges: the
                minion fire it takes (retaliation from tick 0 reads the trader's cell after tick 0)
  one_turn      a CONSTRUCTED turn in which A earns farm, a plate and a kill at once (the reward identity)
  kills         one-turn kills from full HP: a combination of events, a gank with positional fire, a double hit
  low_hp        the exact one-turn death probability of a champion at 30% HP that stays or steps back
  freeze_break  whether a freeze survives one walk-up Shove, kept passively or guarded, by level and held wave
  odds          gank occurrence by exposure for common actions, gank damage per trial, trade hit and double
                probabilities by the defender's behaviour, minion damage on champions
  tower         the minion hits of an undisturbed crash, Apply Pressure damage and capacity by level and investment
  bounds        the evaluation tail constant Omega with its terms, the per-turn reward bound, gamma^W Omega
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ...config import B_HOLD, B_SAFE, B_TRADE, B_YIELD, F, SH, SP, W_SLOWPUSH, champ_act
from ...positional import WARD
from .common import cfg

WAVES = {SP: "Last Hit", F: "Thin", W_SLOWPUSH: "Slow Push", SH: "Shove"}


def lane_row(lane, m4) -> str:
    """Global cells -4 .. +4 and W: A's packets as A<hp>, B's as B<hp>, crashed masses in brackets under the tower they
    siege, champions a / b (with their HP in centi-HP when damaged is not shown; see the rows)."""
    cells: Dict[object, List[str]] = {x: [] for x in range(-4, 5)}
    cells["W"] = []
    if lane.ah > 0:
        cells[lane.ax].append(f"A{lane.ah}")
    for x in lane.ar:
        cells[x].append("A8")
    if lane.bh > 0:
        cells[-lane.bx].append(f"B{lane.bh}")
    for x in lane.br:
        cells[-x].append("B8")
    if lane.asg:
        cells[2].append(f"[A{lane.asg}]")
    if lane.bsg:
        cells[-2].append(f"[B{lane.bsg}]")
    for side, name in ((m4.a, "a"), (m4.b, "b")):
        if side.absent:
            continue
        cells["W" if side.pos == WARD else (side.pos if name == "a" else -side.pos)].append(name)
    keys = list(range(-4, 5)) + ["W"]
    return " | ".join(f"{x if x == 'W' else format(x, '+d')}:{','.join(cells[x]) if cells[x] else '.'}" for x in keys)


def tables() -> dict:
    from ...champions_m4 import context
    from ...progression import investment
    c = cfg()
    ctx = context(c)
    t, inv, g = ctx.t, investment(c), c.gold_scale
    prog = [{"L": L, "xp_next": t.xp_next[L] or None, "xp_cum": t.xp_cum[L], "hp_max": t.hp_max[L],
             "hp_unit": t.hp_unit[L], "ap": t.ap[L][0] / 1e4} for L in range(1, 19)]
    mult = [{"S": S, "M": inv.factor(S * g)} for S in (0, 1, 2, 4, 8, 16, 32, 48, 64, 100, 128, 200, 300, 500)]
    levels, invest = (1, 3, 6, 9, 11, 13, 15, 18), (0, 8, 16, 32, 64, 128, 300)
    cap = {L: {S: {"C": ctx.capacity(L, S * g), "half": ctx.budget_from_capacity(F, ctx.capacity(L, S * g)),
                   "last_hit": ctx.budget_from_capacity(SP, ctx.capacity(L, S * g))} for S in invest} for L in levels}
    return {"progression": prog, "multiplier": mult, "capacity": cap, "levels": levels, "invest": invest}


def wave_actions(turns: int = 8) -> dict:
    from ...champions_m4 import context, resolve_turn_m4
    from ..m45b.states import vision_state
    c = cfg()
    ctx = context(c)
    tb = ctx.t
    start = vision_state(c, "neutral", level=3, gap="equal", k=9)
    out = {}
    for wa, name in WAVES.items():
        lane, m4, k = start
        rows = [{"turn": 0, "k": k, "lane": lane_row(lane, m4), "state": lane.to_list(), "champ": m4.to_list()}]
        tot = {"gold": [0.0, 0.0], "xp": [0, 0], "crash": [0, 0], "tower": [0, 0], "minion_cHP": [0, 0]}
        for i in range(turns):
            a, b = champ_act(wa, B_HOLD), champ_act(SP, B_HOLD)
            t = resolve_turn_m4(lane, m4, k, a, b, c, u=None)
            gold = [t.earned_units[s] / c.gold_scale for s in (0, 1)]
            for s in (0, 1):
                tot["gold"][s] += gold[s]
                tot["xp"][s] += t.xp_gain[s]
                tot["crash"][s] += int(t.lane.crash_hp[s] > 0)
                tot["tower"][s] += t.tower_damage_dealt[s]
                tot["minion_cHP"][s] += t.minion_damage[s]
            lane, m4, k = t.lane.state, t.champ, k + 1
            hp = [m4.side(s).hp / tb.hp_max[t.levels_after[s]] for s in (0, 1)]
            rows.append({"turn": i + 1, "k": k - 1, "budgets": list(t.wave_budget), "lane": lane_row(lane, m4),
                         "gold": gold, "xp": list(t.xp_gain), "crash_hp": list(t.lane.crash_hp),
                         "tower_damage": list(t.tower_damage_dealt), "minion_cHP": list(t.minion_damage),
                         "minion_fire": [list(x) for x in t.minion_fire], "path": [list(p) for p in t.path],
                         "hp": hp, "levels": list(t.levels_after), "end": list(t.end)})
        out[name] = {"rows": rows, "totals": tot}
    return out


def wave_actions_vs_shove(turns: int = 8) -> dict:
    from ...champions_m4 import resolve_turn_m4
    from ..m45b.states import vision_state
    c = cfg()
    out = {}
    for wa, name in WAVES.items():
        lane, m4, k = vision_state(c, "neutral", level=3, gap="equal", k=9)
        tot = {"gold": [0.0, 0.0], "xp": [0, 0], "crash": [0, 0], "minion_cHP": [0, 0]}
        for i in range(turns):
            t = resolve_turn_m4(lane, m4, k, champ_act(wa, B_HOLD), champ_act(SH, B_HOLD), c, u=None)
            for s in (0, 1):
                tot["gold"][s] += t.earned_units[s] / c.gold_scale
                tot["xp"][s] += t.xp_gain[s]
                tot["crash"][s] += int(t.lane.crash_hp[s] > 0)
                tot["minion_cHP"][s] += t.minion_damage[s]
            lane, m4, k = t.lane.state, t.champ, k + 1
        out[name] = tot
    return out


def freeze_hold(turns: int = 40) -> List[dict]:
    from ...champions_m4 import resolve_turn_m4
    from ..m45b.states import freeze_state
    c = cfg()

    def held(lane) -> bool:
        return lane.ah > 0 and lane.bh > 0 and lane.ax + lane.bx >= -1 and -lane.bx <= -1

    out = []
    for level in (4, 10, 16, 18):
        for gap in ("equal", "large"):
            for rule in ("controller", "thin"):
                for size in (5, 8, 12, 14, 20):
                    lane, m4, k = freeze_state(c, level=level, gap=gap, held=size)
                    n = 0
                    for _ in range(turns):
                        contact = lane.ah > 0 and lane.bh > 0 and lane.ax + lane.bx >= -1
                        thin = rule == "thin" or (contact and lane.ax <= -1 and lane.bh >= 14)
                        t = resolve_turn_m4(lane, m4, k, champ_act(F if thin else SP, B_HOLD), champ_act(SP, B_HOLD),
                                            c, u=None)
                        lane, m4, k = t.lane.state, t.champ, k + 1
                        if not held(lane):
                            break
                        n += 1
                    out.append({"level": level, "gap": gap, "rule": rule, "held_hp": size, "turns_held": n})
    return out


def retaliation() -> List[dict]:
    """A (level 2) Trades B, who stands in its own 16-HP front at B's -1 (A's +1), A's 8-HP front fighting it at A's
    lane cell; A's hit is forced (trade-hit uniform 0), nothing else happens. A starts at its -2 (two edges to its
    contact cell 0, at -1 after tick 0) or at its -1 (one edge, at 0 after tick 0)."""
    from ...champions import N_CHANCE, U_TRADE_HIT
    from ...champions_m4 import M4Side, M4State, context, resolve_turn_m4
    from ...config import B_TRADE
    from ...state import State
    c = cfg()
    t = context(c).t
    xp = t.xp_cum[2]
    u = [1.0] * N_CHANCE
    u[U_TRADE_HIT[0]] = 0.0
    out = []
    for start in (-2, -1):
        lane = State(0, 8, (), 0, -1, 16, (), 0, 0, 0, 0, 0)
        m4 = M4State(M4Side(start, t.hp_max[2], 0, xp), M4Side(-1, t.hp_max[2], 0, xp))
        r = resolve_turn_m4(lane, m4, 9, champ_act(SP, B_TRADE), champ_act(SP, B_HOLD), c, u=u)
        out.append({"start": start, "path": [list(p) for p in r.path], "connected": list(r.connected),
                    "trade_damage": list(r.trade_damage), "minion_fire": [list(x) for x in r.minion_fire],
                    "minion_hits": list(r.minion_hits), "minion_cHP": list(r.minion_damage),
                    "hp_max": t.hp_max[2]})
    return out


def windows() -> List[dict]:
    from ...mechanics import start_packets, strength_need, window_strength
    from ..m45b.states import freeze_state, vision_state
    c = cfg()
    lane, _, k = vision_state(c, "neutral", level=3, gap="equal", k=9)
    big = lane._replace(bh=24)
    fz, _, kf = freeze_state(c, level=10, gap="equal", held=5)
    out = []
    for name, ln, kk in (("neutral (turn 9: A8 v B8 at the lane cell)", lane, k),
                         ("the same with B's front at 24 HP", big, k),
                         ("freeze (turn 51: A8 in A's tower cell, B5 held at A's -1)", fz, kf)):
        P = start_packets(ln, c)
        out.append({"state": name, "S": [float(window_strength(P[s], kk, c)) for s in (0, 1)],
                    "thin_need": [strength_need(P, s, kk, 0, c) for s in (0, 1)],
                    "slowpush_need": [strength_need(P, s, kk, 1, c) for s in (0, 1)]})
    return out


def one_turn() -> dict:
    """A CONSTRUCTED turn (the combination is rare in play) in which A earns farm, a plate and a kill at once,
    resolved by the engine:

      lane   A's crashed mass of 10 HP under B's tower; B's 8-HP front waiting one cell behind its tower (B's -3 =
             A's +3) and engaging that mass; A's next 8-HP front at the lane cell
      A      level 5, nothing invested, at its +1, tower damage 1,500 (100 short of the first plate):
             Shove + Hold + Apply Pressure (legal: own siege, d(+1, +2) <= 2)
      B      level 5, at its lane cell (0) with 200 of 729 centi-HP, Last Hit + Hold
      chance B's gank uniform 0.0 (the gank happens), its damage uniform 0.999 (4 of 4 trials); nothing else happens
    """
    from ...champions import N_CHANCE, U_GANK, U_GANK_DAMAGE
    from ...champions_m4 import M4Side, M4State, context, resolve_turn_m4
    from ...config import PRESSURE
    from ...state import State
    c = cfg()
    t = context(c).t
    lane = State(0, 8, (), 10, -3, 8, (), 0, 0, 0, 0, 0)
    xp = t.xp_cum[5]
    m4 = M4State(M4Side(1, t.hp_max[5], 0, xp, 0, 0, 1500), M4Side(0, 200, 0, xp, 0, 0, 0))
    u = [1.0] * N_CHANCE
    u[U_GANK[1]] = 0.0
    u[U_GANK_DAMAGE[1]] = 0.999
    a, b = champ_act(SH, B_HOLD, PRESSURE), champ_act(SP, B_HOLD)
    r = resolve_turn_m4(lane, m4, 12, a, b, c, u=u)
    sc = c.gold_scale
    return {"label": "CONSTRUCTED", "k": 12, "state_before": lane.to_list(), "champ_before": m4.to_list(),
            "lane_before": lane_row(lane, m4), "lane_after": lane_row(r.lane.state, r.champ),
            "actions": [c.label(a), c.label(b)], "budgets": list(r.wave_budget),
            "kills": [{str(x): h for x, h in r.lane.kills[s].items()} for s in (0, 1)],
            "modes": list(r.modes), "start": list(r.start), "end": list(r.end), "reach": list(r.reach),
            "path": [list(p) for p in r.path], "exposure": list(r.exposure),
            "minion_fire": [list(x) for x in r.minion_fire], "minion_hits": list(r.minion_hits),
            "gank_p": list(r.gank_p), "gank_damage": list(r.gank_damage), "minion_damage": list(r.minion_damage),
            "damage_taken": list(r.damage_taken), "died": list(r.died),
            "tower_hits": list(r.tower_hits), "tower_damage_dealt": list(r.tower_damage_dealt),
            "plates": list(r.plates_broken), "farm": [x / sc for x in r.farm_units],
            "plate_gold": [x / sc for x in r.plate_units], "kill_gold": [x / sc for x in r.bounty_units],
            "earned": [x / sc for x in r.earned_units], "reward": r.reward, "reward_units": r.reward_units,
            "xp_gain": list(r.xp_gain), "champ_after": r.champ.to_list(), "state_after": r.lane.state.to_list()}


def odds() -> dict:
    from ...champions import (champion_gank_probability, gank_damage_p, trade_double_probability,
                              trade_hit_probability)
    from ...champions_m4 import context
    from ...mechanics import initial_state
    from ...positional import minion_damage_centi
    c = cfg()
    lane = initial_state(c)
    rows = {}
    for label, w, beh in (("Hold + Last Hit", SP, B_HOLD), ("Hold + Shove", SH, B_HOLD), ("Trade + Shove", SH, B_TRADE),
                          ("Yield + Last Hit", SP, B_YIELD), ("Safe + Last Hit", SP, B_SAFE)):
        rows[label] = {e: champion_gank_probability(lane, 0, w, beh, c, e) for e in range(-2, 3)}
    dmg = {e: {"normal": gank_damage_p(0, False, e, c), "defensive": gank_damage_p(0, True, e, c)} for e in range(-2, 3)}
    hit = {name: {"open": trade_hit_probability(lane, 0, beh, False, c),
                  "in_tower": trade_hit_probability(lane, 0, beh, True, c)}
           for name, beh in (("Hold", B_HOLD), ("Yield", B_YIELD), ("Trade", B_TRADE), ("Safe", B_SAFE))}
    t = context(c).t
    minion = [{"units": m, "L": L, "S": S, "cHP_before_M": minion_damage_centi(m, L, c.minion_damage_alpha,
                                                                               c.minion_damage_growth),
               "hp_max": t.hp_max[L]} for m, L, S in ((2, 1, 0), (2, 4, 0), (2, 11, 32), (2, 18, 128), (4, 1, 0))]
    return {"gank_p": rows, "ward_exposure": c.ward_exposure, "gank_damage_p": dmg, "trade_hit": hit,
            "trade_double": trade_double_probability(0, c), "minion": minion}


def kills() -> dict:
    """One-turn kills from full HP on constructed states, chance uniforms forced (§6.5):
      combination  level 1, nothing invested: A Trades at its +1, B Trades at the lane cell, both 8-HP fronts at the
                   lane cell; B's double hit, the tower-adjacent shot on A and a four-trial gank on A; with and without
                   A's own hit landing (which draws B's wave's retaliation)
      positional   level 1: A Trades at its +1 into B's front at B's -1 (A's front behind it), A's hit misses; a
                   four-trial gank on A, and the positional fire of B's front of 8, 16 or 24 HP
      double_hit   A's single double hit on a Hold B at full HP, no other event, for several (level, invested) pairs
    """
    from ...champions import N_CHANCE, U_GANK, U_GANK_DAMAGE, U_TOWER, U_TRADE_DOUBLE, U_TRADE_HIT
    from ...champions_m4 import M4Side, M4State, context, resolve_turn_m4
    from ...config import B_TRADE
    from ...state import State
    c = cfg()
    ctx = context(c)
    t, g = ctx.t, c.gold_scale

    def side(pos, L, S=0):
        return M4Side(pos, t.hp_max[L], 0, t.xp_cum[L], int(S * g), 0, 0)

    def rec(r, s):
        return {"trade_cHP": r.trade_damage[1 - s], "tower_cHP": r.tower_damage[s], "gank_cHP": r.gank_damage[s],
                "minion_units": r.minion_hits[s], "minion_cHP": r.minion_damage[s], "taken": r.damage_taken[s],
                "died": bool(r.died[s])}

    neutral = State(0, 8, (-2,), 0, 0, 8, (-2,), 0, 0, 0, 0, 0)
    combination = []
    for own_hit in (False, True):
        u = [1.0] * N_CHANCE
        u[U_TRADE_HIT[1]] = u[U_TRADE_DOUBLE[1]] = u[U_TOWER[0]] = u[U_GANK[0]] = 0.0
        u[U_GANK_DAMAGE[0]] = 0.999
        if own_hit:
            u[U_TRADE_HIT[0]] = 0.0
        r = resolve_turn_m4(neutral, M4State(side(1, 1), side(0, 1)), 9, champ_act(SP, B_TRADE),
                            champ_act(SP, B_TRADE), c, u=u, ctx=ctx)
        combination.append({"own_hit": own_hit, "hp": t.hp_max[1], **rec(r, 0)})
    positional = []
    for bh in (8, 16, 24):
        u = [1.0] * N_CHANCE
        u[U_GANK[0]] = 0.0
        u[U_GANK_DAMAGE[0]] = 0.999
        lane = State(0, 8, (-2,), 0, -1, bh, (), 0, 0, 0, 0, 0)
        r = resolve_turn_m4(lane, M4State(side(1, 1), side(-1, 1)), 9, champ_act(SP, B_TRADE), champ_act(SP, B_HOLD),
                            c, u=u, ctx=ctx)
        positional.append({"enemy_front_hp": bh, "hp": t.hp_max[1], **rec(r, 0)})
    double_hit = []
    for La, Sa, Lb in ((18, 300, 10), (18, 128, 10), (18, 300, 11), (12, 64, 1), (10, 16, 1), (18, 0, 1)):
        u = [1.0] * N_CHANCE
        u[U_TRADE_HIT[0]] = u[U_TRADE_DOUBLE[0]] = 0.0
        r = resolve_turn_m4(neutral, M4State(side(0, La, Sa), side(0, Lb)), 9, champ_act(SP, B_TRADE),
                            champ_act(SP, B_HOLD), c, u=u, ctx=ctx)
        double_hit.append({"La": La, "Sa": Sa, "Lb": Lb, "hp": t.hp_max[Lb],
                           "ratio": 1.05 ** (La - Lb) * ctx.inv.factor(int(Sa * g)), **rec(r, 1)})
    return {"combination": combination, "positional": positional, "double_hit": double_hit}


def tower() -> dict:
    """An undisturbed crashed mass of h HP (no defender engages it): the minion hits it lands on the tower and the ticks
    it survives, by the parity of its first tick (tick 0 or tick 1 of a turn); Apply Pressure damage P(L, S) and the
    Pressure turns that exhaust a tower from 0."""
    import math
    from ...champions_m4 import context
    from ...mechanics import minion_damage
    from ...progression import INVEST_MAX_RESOURCES, investment
    c = cfg()
    ctx = context(c)
    t, inv, g = ctx.t, investment(c), c.gold_scale
    crash = []
    for h in (8, 16, 24):
        by_parity = []
        for first in (0, 1):
            m, tick, hits, n = h, first, 0, 0
            while m > 0:
                hits += minion_damage(m, c, tick)
                m -= min(c.tower_siege_damage, m)
                tick, n = (tick + 1) % 2, n + 1
            by_parity.append({"first_tick": first, "hits": hits, "ticks": n})
        crash.append({"hp": h, "by_parity": by_parity})
    pressure = []
    for L, S in ((1, 0), (3, 0), (5, 0), (9, 15), (9, 16), (14, 32), (14, 33), (18, 0), (18, 55), (18, 128)):
        p = ctx.pressure(L, S * g)
        pressure.append({"L": L, "S": S, "ap": t.ap[L][0] / 1e4 * inv.factor(S * g), "P": p, "C": ctx.capacity(L, S * g),
                         "turns": math.ceil(c.plate_thresholds[-1] / p)})
    return {"crash": crash, "pressure": pressure,
            "pressure_at_max_S": ctx.pressure(18, INVEST_MAX_RESOURCES * g), "max_S": INVEST_MAX_RESOURCES}


def low_hp() -> List[dict]:
    """A champion at 30% HP at the lane cell (CONSTRUCTED: A's 16-HP front at its +1 against B's 4-HP front, the M4.5b
    "forward" lane; B at its own -1 with full HP; equal level and investment at level 3 with nothing invested and at
    the rounded `sp:0` self-play means after 50, 94 and 150 turns). The exact probability that A dies this turn, by
    enumeration of the chance outcomes that can hurt it (B's trade hit and double, A's tower shot, A's gank occurrence
    and its Binomial(4, p_e) damage), for A staying (Last Hit + Hold, Shove + Hold) or stepping back (Last Hit + Safe)
    against B Trading or only farming."""
    from math import comb
    from ...champions import N_CHANCE, U_GANK, U_GANK_DAMAGE, U_TOWER, U_TRADE_DOUBLE, U_TRADE_HIT
    from ...champions_m4 import context, resolve_turn_m4
    from ...config import B_SAFE, B_TRADE
    from ...state import State
    from ..m45b.states import champs
    c = cfg()
    ctx = context(c)
    lane = State(1, 16, (-2,), 0, -1, 4, (), 0, 0, 0, 0, 0)
    hi = 0.999999

    def outcomes(pa, q, pb, pe, tp):
        gank, cum = [(1 - pb, None)], 0.0
        for d in range(5):
            p = comb(4, d) * pe ** d * (1 - pe) ** (4 - d)
            gank.append((pb * p, cum + p / 2))
            cum += p
        for wt, (uh, ud) in ((1 - pa, (hi, hi)), (pa * (1 - q), (0.0, hi)), (pa * q, (0.0, 0.0))):
            for wtow, utow in ((1 - tp, hi), (tp, 0.0)):
                for wg, ug in gank:
                    if wt * wtow * wg == 0:
                        continue
                    u = [hi] * N_CHANCE
                    u[U_TRADE_HIT[1]], u[U_TRADE_DOUBLE[1]], u[U_TOWER[0]] = uh, ud, utow
                    if ug is not None:
                        u[U_GANK[0]], u[U_GANK_DAMAGE[0]] = 0.0, ug
                    yield wt * wtow * wg, u

    out = []
    for L, S in ((3, 0), (9, 15), (14, 33), (18, 55)):
        m4 = champs(c, (L, L), (S, S), (0, -1), (0.3, 1.0))
        for a_name, a in (("Last Hit + Hold", champ_act(SP, B_HOLD)), ("Shove + Hold", champ_act(SH, B_HOLD)),
                          ("Last Hit + Safe", champ_act(SP, B_SAFE))):
            for b_name, b in (("Trade", champ_act(SP, B_TRADE)), ("Hold", champ_act(SP, B_HOLD))):
                r0 = resolve_turn_m4(lane, m4, 2, a, b, c, u=[hi] * N_CHANCE)
                pa = r0.hit_p[1] if r0.connected[1] else 0.0
                args = (pa, r0.double_p[1], r0.gank_p[0], r0.gank_damage_p[0], r0.tower_p[0])
                die = sum(w for w, u in outcomes(*args) if resolve_turn_m4(lane, m4, 2, a, b, c, u=u).died[0])
                die_trade = sum(w for w, u in outcomes(pa, args[1], 0.0, args[3], 0.0)
                                if resolve_turn_m4(lane, m4, 2, a, b, c, u=u).died[0])
                out.append({"L": L, "S": S, "a": a_name, "b": b_name, "p_die": die, "p_die_trade_only": die_trade,
                            "hit": pa, "double": args[1], "gank": args[2], "gank_damage_p": args[3],
                            "M": ctx.inv.factor(m4.a.items)})
    return out


def freeze_break(draws: int = 200) -> List[dict]:
    """One turn from the M4.5b freeze start (`m45b.states.freeze_state`, equal power at the typical investment of the
    level): B plays the walk-up Shove (Shove + Hold); A keeps (Thin + Hold) or guards (Thin + Trade). Survival is the
    share of `draws` random chance vectors (default_rng(1)) after which A still holds the freeze
    (`m44.records.holds_freeze`), with the mean HP each side loses, in bars."""
    from ...champions import N_CHANCE
    from ...champions_m4 import context, resolve_turn_m4
    from ...config import B_TRADE
    from ..m45b.states import freeze_state
    from ..rl.m44.records import holds_freeze
    c = cfg()
    ctx = context(c)
    walk = champ_act(SH, B_HOLD)
    out = []
    for held in (5, 12, 20):
        for L in (3, 4, 5, 6, 7, 8, 10, 12, 15):
            lane, m4, k = freeze_state(c, level=L, gap="equal", held=held)
            row = {"held_hp": held, "L": L, "C_B": ctx.capacity(L, m4.b.items)}
            for name, a in (("keep", champ_act(F, B_HOLD)), ("guard", champ_act(F, B_TRADE))):
                rng = np.random.default_rng(1)
                surv = lost_a = lost_b = 0.0
                for _ in range(draws):
                    r = resolve_turn_m4(lane, m4, k, a, walk, c, u=rng.random(N_CHANCE))
                    surv += holds_freeze(r.lane.state, 0)
                    lost_a += r.damage_taken[0] / ctx.t.hp_max[L]
                    lost_b += r.damage_taken[1] / ctx.t.hp_max[L]
                row[name] = {"survives": surv / draws, "A_bars": lost_a / draws, "B_bars": lost_b / draws}
            out.append(row)
    return out


def bounds() -> dict:
    """Omega (`env/discounted.tail_constant`) with its four terms, the per-turn reward bound, and gamma^W Omega."""
    from ...config import GameConfig
    from ...env.discounted import max_champion_wave_budget, per_turn_bound, reward_per_hp, tail_constant
    from ...lane_enum import recorded_bounds
    out = {}
    for name, c in (("m45_calibrated", cfg()), ("m4_lasthit", GameConfig.m4_lasthit())):
        om = tail_constant(c)
        f, s, g = reward_per_hp(c), c.spawn_every_turns, c.gamma
        a_max = recorded_bounds(c)["A_max"]
        d_min = min(c.death_turns_by_level) if c.death_turns_by_level else c.death_turns
        parts = {"a": f * (a_max + c.wave_hp * (s - 1) / s), "b_over_1_minus_gamma": f * c.wave_hp / s / (1 - g),
                 "kills": c.kill_bounty / (1 - g ** (d_min + 1)),
                 "plates": len(c.plate_thresholds) * c.plate_gold}
        out[name] = {"omega": om, "omega_terms": parts, "A_max": a_max, "farm_per_hp_max": f,
                     "per_turn_bound": per_turn_bound(c), "max_wave_budget": max_champion_wave_budget(c),
                     "bound": {W: g ** W * om for W in (50, 150, 300, 460, 500, 1000)}}
    return out


def run() -> dict:
    return {"tables": tables(), "wave_actions": wave_actions(), "wave_actions_vs_shove": wave_actions_vs_shove(),
            "windows": windows(), "freeze_hold": freeze_hold(), "retaliation": retaliation(), "one_turn": one_turn(),
            "kills": kills(), "low_hp": low_hp(), "freeze_break": freeze_break(), "odds": odds(), "tower": tower(),
            "bounds": bounds()}


__all__ = ["run", "tables", "wave_actions", "wave_actions_vs_shove", "windows", "freeze_hold", "retaliation",
           "one_turn", "kills", "low_hp", "freeze_break", "odds", "tower", "bounds", "lane_row", "WAVES"]
