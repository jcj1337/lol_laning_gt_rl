"""Deterministic worked examples for docs/m4/laning_game_mathematical_reference.md, computed with the engine itself.

  wave_actions    the same start (the M4.2e-B `neutral` start, k9: both fronts of 8 HP at the lane cell, both
                  champions there at level 3) played for 8 turns with A on Hold, Thin or Clear (Hold position) against
                  B on Hold (Hold position); no chance events (u = None: no gank, trade or tower shot can happen). Per
                  turn: both fronts, rears, crashed masses, the champions' budgets, gold and XP.
  one_turn        a recorded turn of a learned policy against a reference in which one side earns farm, plate gold and
                  a kill bounty in the same turn (the reward identity of reference §8), found by replaying games.
  tables          the progression and capacity tables of reference §7.

`python -m lanegame.experiments.m43c examples` writes results/m43c/examples.md and examples.json.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ...config import B_HOLD, W_CLEAR, W_HOLD, W_THIN, champ_act
from .common import cfg_game, save_json, write_md

NAMES = {W_HOLD: "Hold", W_THIN: "Thin", W_CLEAR: "Clear"}


def _lane_row(lane, m4) -> str:
    """One line of global cells -4 .. +4: A's packets as A<hp>, B's as B<hp> (B's own x mirrored), champions a / b,
    crashed masses in brackets under the tower they siege."""
    cells = {x: [] for x in range(-4, 5)}
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
    if not m4.a.absent:
        cells[m4.a.pos].append("a")
    if not m4.b.absent:
        cells[-m4.b.pos].append("b")
    return " | ".join(f"{x:+d}:{','.join(v) if v else '.'}" for x, v in cells.items())


def wave_actions(turns: int = 8) -> dict:
    from ...champions_m4 import context, resolve_turn_m4
    from ..m4.zoning_natural import starts
    cfg = cfg_game()
    ctx = context(cfg)
    lane0, m40, k0 = starts(cfg)["neutral"]
    out = {}
    for wa in (W_HOLD, W_THIN, W_CLEAR):
        lane, m4, k = lane0, m40, k0
        rows = [{"turn": 0, "lane": _lane_row(lane, m4), "state": lane.to_list()}]
        tot = {"gold": [0.0, 0.0], "xp": [0, 0], "crash": [0, 0], "tower": [0, 0]}
        for i in range(turns):
            a, b = champ_act(wa, B_HOLD), champ_act(W_HOLD, B_HOLD)
            la = context(cfg).t.level_by_xp[min(m4.a.xp, ctx.t.xp_cap)]
            lb = context(cfg).t.level_by_xp[min(m4.b.xp, ctx.t.xp_cap)]
            bud = (ctx.wave_budget(wa, la, m4.a.items), ctx.wave_budget(W_HOLD, lb, m4.b.items))
            t = resolve_turn_m4(lane, m4, k, a, b, cfg, u=None)
            lane, m4, k = t.lane.state, t.champ, k + 1
            for s in (0, 1):
                tot["gold"][s] += t.earned_units[s] / cfg.gold_scale
                tot["xp"][s] += t.xp_gain[s]
                tot["crash"][s] += int(t.lane.crash_hp[s] > 0)
                tot["tower"][s] += t.tower_damage_dealt[s]
            rows.append({"turn": i + 1, "k": k - 1, "budgets": list(bud), "lane": _lane_row(lane, m4),
                         "gold": [t.earned_units[s] / cfg.gold_scale for s in (0, 1)], "xp": list(t.xp_gain),
                         "kills": [{str(x): h for x, h in t.lane.kills[s].items()} for s in (0, 1)],
                         "crash_hp": list(t.lane.crash_hp), "tower_damage": list(t.tower_damage_dealt),
                         "levels": list(t.levels_after)})
        out[NAMES[wa]] = {"rows": rows, "totals": tot}
    return out


def one_turn() -> dict:
    """A CONSTRUCTED turn (the combination is rare in recorded play: none in 120 replayed policy-vs-reference games of
    300 turns) in which A earns farm, a plate and a kill bounty at once, resolved by the engine:

      lane   A's crashed mass of 10 HP under B's tower; B's 8-HP front waiting one cell behind its tower (B's -3 =
             A's +3) and engaging that mass; A's next 8-HP front at the lane cell
      A      level 5, nothing invested, at its +1, tower damage 1,500 (100 short of the first plate):
             Clear + Hold + Apply Pressure (legal: own siege, +1 + 2 >= T)
      B      level 5, at its lane cell (0) with 200 of 729 centi-HP, Hold + Hold
      chance B's gank uniform 0.0 (the gank happens), its damage uniform 0.999 (4 of 4 trials); nothing else happens
    """
    from ...champions import N_CHANCE, U_GANK, U_GANK_DAMAGE
    from ...champions_m4 import M4Side, M4State, context, resolve_turn_m4
    from ...config import PRESSURE
    from ...state import State
    cfg = cfg_game()
    t = context(cfg).t
    lane = State(0, 8, (), 10, -3, 8, (), 0, 0, 0, 0, 0)
    xp = t.xp_cum[5]
    m4 = M4State(M4Side(1, t.hp_max[5], 0, xp, 0, 0, 1500), M4Side(0, 200, 0, xp, 0, 0, 0))
    u = [1.0] * N_CHANCE
    u[U_GANK[1]] = 0.0
    u[U_GANK_DAMAGE[1]] = 0.999
    a, b = champ_act(W_CLEAR, B_HOLD, PRESSURE), champ_act(W_HOLD, B_HOLD)
    r = resolve_turn_m4(lane, m4, 12, a, b, cfg, u=u)
    sc = cfg.gold_scale
    return {"label": "CONSTRUCTED", "state_before": lane.to_list(), "champ_before": m4.to_list(),
            "actions": [cfg.label(a), cfg.label(b)], "budgets": [context(cfg).wave_budget(W_CLEAR, 5, 0), 1],
            "kills": [{str(x): h for x, h in r.lane.kills[s].items()} for s in (0, 1)],
            "modes": list(r.modes), "end": list(r.end), "gank_p": list(r.gank_p), "gank_damage": list(r.gank_damage),
            "died": list(r.died), "tower_damage_dealt": list(r.tower_damage_dealt), "plates": list(r.plates_broken),
            "farm_units": list(r.farm_units), "plate_units": list(r.plate_units), "bounty_units": list(r.bounty_units),
            "earned_units": list(r.earned_units), "reward_units": r.reward_units, "reward": r.reward,
            "farm": [x / sc for x in r.farm_units], "xp_gain": list(r.xp_gain), "champ_after": r.champ.to_list(),
            "state_after": r.lane.state.to_list()}


def tables_md() -> str:
    from ...champions_m4 import context
    from ...progression import investment
    from ...config import SP, F
    cfg = cfg_game()
    ctx = context(cfg)
    t = ctx.t
    inv = investment(cfg)
    g = cfg.gold_scale
    lines = ["| L | n(L) XP to next | X(L) cumulative | H(L) centi-HP | u(L) | AP(L, 0) |", "|---|---|---|---|---|---|"]
    for L in range(1, 19):
        lines.append(f"| {L} | {t.xp_next[L] or '-'} | {t.xp_cum[L]} | {t.hp_max[L]} | {t.hp_unit[L]} | "
                     f"{t.ap[L][0] / 1e4:.3f} |")
    lines += ["", "| S invested | M(S) |", "|---|---|"]
    for S in (0, 1, 2, 4, 8, 16, 32, 48, 64, 100, 200, 500):
        lines.append(f"| {S} | {inv.factor(S * g):.3f} |")
    lines += ["", "| L \\ S | " + " | ".join(str(S) for S in (0, 8, 16, 32, 64, 128, 300)) + " |",
              "|---" * 8 + "|"]
    for L in (1, 3, 6, 9, 11, 13, 15, 18):
        cells = []
        for S in (0, 8, 16, 32, 64, 128, 300):
            C = ctx.capacity(L, S * g)
            cells.append(f"{ctx.budget_from_capacity(SP, C)}/{ctx.budget_from_capacity(F, C)}/{C}")
        lines.append(f"| {L} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run() -> dict:
    wa = wave_actions()
    ot = one_turn()
    out = {"wave_actions": wa, "one_turn": ot}
    save_json("examples", out)
    md: List[str] = ["# M4.3c worked examples (generated by `python -m lanegame.experiments.m43c examples`)", ""]
    for name, d in wa.items():
        md += [f"## A plays {name}, B plays Hold (both Hold position), from the neutral start", "",
               "| turn | budgets A/B | lane (global cells; A<hp> / B<hp> packets, [mass] crashed, a / b champions) | "
               "gold A/B | XP A/B | crash HP A/B |", "|---|---|---|---|---|---|"]
        for r in d["rows"]:
            if r["turn"] == 0:
                md.append(f"| 0 | | {r['lane']} | | | |")
                continue
            md.append(f"| {r['turn']} | {r['budgets'][0]}/{r['budgets'][1]} | {r['lane']} | "
                      f"{r['gold'][0]:.3f}/{r['gold'][1]:.3f} | {r['xp'][0]}/{r['xp'][1]} | "
                      f"{r['crash_hp'][0]}/{r['crash_hp'][1]} |")
        tt = d["totals"]
        md += ["", f"Totals: gold {tt['gold'][0]:.3f} / {tt['gold'][1]:.3f}, XP {tt['xp'][0]} / {tt['xp'][1]}, "
               f"crashes {tt['crash'][0]} / {tt['crash'][1]}, tower damage dealt {tt['tower'][0]} / {tt['tower'][1]}", ""]
    if ot:
        md += ["## One turn with farm, plate gold and a bounty", "", "```", str(ot), "```", ""]
    md += ["## Tables", "", tables_md()]
    write_md("examples", "\n".join(md))
    return out


__all__ = ["run", "wave_actions", "one_turn", "tables_md"]
