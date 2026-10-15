"""M4.2d-B: independent champion positioning (docs/m4/m4_2/m4_positioning_report.md).

Short deterministic trajectories of the M4 resolver with no chance events (u = None: no ganks, no trade hits, so HP
stays full and positions are not disturbed by deaths). The threat of a trade is reported as its EXPECTED damage
(hit probability x damage of the connected trade, plus the expected tower shot), the gank exposure as the summed gank
probability. Policies are fixed scripted rules; nothing is trained or sampled.

    python -m lanegame.experiments.m4.positioning        all stages, JSON to results/m4_positioning/

Rules compared (GameConfig.position_rule): "m2" (default), "pin" (a Yield that starts the turn in contact with an enemy
Trade cannot evade it) and "standoff" ("pin" plus Yield farms one cell behind its farm spot).
"Strong" A = two item tiers at the start of the window (attack power x1.44); "equal" = no items on either side.
Start state "neutral" = s0 after 9 turns of mutual Slow Push (symmetric contact at the lane).
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List

from ...champions import in_contact
from ...champions_m4 import context, resolve_turn_m4
from ...config import B_HOLD, B_SAFE, B_TRADE, B_YIELD, BEHAVIOR_NAMES, SP, GameConfig, champ_act
from ...io import dump_json
from . import wave_readiness as wr
from .common import RESULTS_DIR, meta

OUT = os.path.join(RESULTS_DIR, "m4_positioning")
A, B = 0, 1
RULES = ("m2", "pin", "standoff")
HP_VALUE = 0.25      # resources per 100 centi-HP: the diagnostics-study valuation, for the illustrative net only
LANE_BEHAVIORS = (B_HOLD, B_YIELD, B_TRADE, B_SAFE)
W = 24               # window (turns) of every multi-turn line


def cfg_of(rule: str) -> GameConfig:
    return GameConfig.m4(position_rule=rule)


def start(cfg: GameConfig, strong: bool):
    lane, m4, k = wr.end_state(wr.play(cfg, wr.hold, wr.hold, 9))
    if strong:
        m4 = m4.replace_side(A, items=2)
    return lane, m4, k


def expected_damage(t, pre, cfg) -> List[float]:
    """Expected centi-HP side s TAKES this turn from the enemy's connected trade and the tower (no ganks)."""
    ctx = context(cfg)
    out = [0.0, 0.0]
    for s in (A, B):
        o = 1 - s
        if t.connected[o]:
            h, q = t.hit_p[o], t.double_p[o]
            L, N = t.levels_before[o], pre.side(o).items
            out[s] += h * (1 - q) * ctx.hit(1, L, N) + h * q * ctx.hit(2, L, N)
        out[s] += t.tower_p[s] * cfg.tower_shot_damage * ctx.t.hp_unit[t.levels_before[s]]
    return out


# ---------------------------------------------------------------------- policies (cfg, lane, m4, side, k) -> (wave, behavior)
def fixed(beh: int) -> Callable:
    return lambda cfg, lane, m4, s, k: (SP, beh)


def periodic_trade(period: int, base: int = B_HOLD, phase: int = 0) -> Callable:
    """`base` every turn except every `period`-th turn (k mod period == phase), which is a Trade."""
    return lambda cfg, lane, m4, s, k: (SP, B_TRADE if k % period == phase else base)


def respect(cfg, lane, m4, s, k):
    """Observable reactive rule: Safe while the turn starts in contact with the enemy champion (a trade would pin a
    Yield), otherwise Yield."""
    me, op = m4.side(s), m4.side(1 - s)
    exposed = op.absent == 0 and in_contact(me.pos, op.pos, cfg)
    return (SP, B_SAFE if exposed else B_YIELD)


def schedule(blocks) -> Callable:
    """[(turns, behavior), ...] relative to the first turn the policy sees."""
    first = []

    def pol(cfg, lane, m4, s, k):
        if not first:
            first.append(k)
        i, n = k - first[0], 0
        for turns, beh in blocks:
            n += turns
            if i < n:
                return (SP, beh)
        return (SP, blocks[-1][1])
    return pol


# ---------------------------------------------------------------------- measurement
def summarize(rows, cfg) -> dict:
    ts = [r["t"] for r in rows]
    n = len(ts)
    dmg = [expected_damage(r["t"], r["pre_m4"], cfg) for r in rows]
    farm = [sum(t.farm[s] for t in ts) for s in (A, B)]
    edmg = [sum(d[s] for d in dmg) for s in (A, B)]
    return {
        "turns": n,
        "cells": [[t.end[A], t.end[B]] for t in ts],
        "contact_end_frac": round(sum(in_contact(t.end[A], t.end[B], cfg) for t in ts) / n, 3),
        "farm": [round(f, 3) for f in farm],
        "farm_per_turn": [round(f / n, 4) for f in farm],
        "xp": [sum(t.xp_gain[s] for t in ts) for s in (A, B)],
        "trades_chosen": [sum(t.behaviors[s] == B_TRADE for t in ts) for s in (A, B)],
        "trades_connected": [sum(t.connected[s] for t in ts) for s in (A, B)],
        "evaded": [sum(t.threatened[s] for t in ts) for s in (A, B)],
        "pinned": [sum(t.pinned[s] for t in ts) for s in (A, B)],
        "expected_damage_taken": [round(e, 1) for e in edmg],
        "expected_ganks": [round(sum(t.gank_p[s] for t in ts), 3) for s in (A, B)],
        "net_A_illustrative": round(farm[A] - farm[B] - HP_VALUE * (edmg[A] - edmg[B]) / 100, 3),
    }


def play(cfg, pa, pb, turns, lane, m4, k0) -> List[dict]:
    rows = []
    for row in wr.play(cfg, pa, pb, turns, lane, m4, k0):
        row["pre_m4"] = m4
        rows.append(row)
        m4 = row["t"].champ
    return rows


def one_turn(cfg, lane, m4, k, ba, bb) -> dict:
    t = resolve_turn_m4(lane, m4, k, champ_act(SP, ba), champ_act(SP, bb), cfg, u=None)
    wr.TURNS[0] += 1
    d = expected_damage(t, m4, cfg)
    return {"A": BEHAVIOR_NAMES[ba], "B": BEHAVIOR_NAMES[bb], "start": [m4.a.pos, m4.b.pos], "end": list(t.end),
            "farm": [round(t.farm[A], 4), round(t.farm[B], 4)], "edmg_taken": [round(d[A], 1), round(d[B], 1)],
            "connected": list(t.connected), "evaded": list(t.threatened), "pinned": list(t.pinned),
            "modes": list(t.modes), "gank_p": [round(g, 4) for g in t.gank_p],
            "net_A_illustrative": round(t.farm[A] - t.farm[B] - HP_VALUE * (d[A] - d[B]) / 100, 4)}


# ---------------------------------------------------------------------- stages
def stage_equal() -> dict:
    """E1: equal strength contesting the same wave, W turns from the neutral lane."""
    out = {}
    for rule in RULES:
        cfg = cfg_of(rule)
        st = start(cfg, strong=False)
        out[rule] = {f"{BEHAVIOR_NAMES[x]}/{BEHAVIOR_NAMES[y]}": summarize(play(cfg, fixed(x), fixed(y), W, *st), cfg)
                     for x, y in ((B_HOLD, B_HOLD), (B_YIELD, B_YIELD), (B_HOLD, B_YIELD))}
    return out


def stage_farm_cost() -> dict:
    """E1b: the farming and exposure cost of the stand-off cell, equal strength, W turns from the "neutral" lane and
    from the "pushed" lane of the wave-readiness study (A holds a stacked wave at the lane: fights leave the spot)."""
    out = {}
    for rule in ("m2", "standoff"):
        cfg = cfg_of(rule)
        st = wr.starts(cfg)
        out[rule] = {f"{sn}: {BEHAVIOR_NAMES[x]}/{BEHAVIOR_NAMES[y]}": summarize(play(cfg, fixed(x), fixed(y), W,
                                                                                       *st[sn]), cfg)
                     for sn in ("neutral", "pushed")
                     for x, y in ((B_HOLD, B_HOLD), (B_YIELD, B_HOLD), (B_HOLD, B_YIELD), (B_YIELD, B_YIELD))}
    return out


def stage_matrix() -> dict:
    """E2/E3: the one-turn joint-action matrix (A in Hold, Yield, Trade, Safe x B in the same) for a strong A, at the
    configuration reached after 6 turns of (A, B) = (Hold, Yield) ["forward"] and of (Yield, Yield) ["standoff"]."""
    out = {}
    for rule in RULES:
        cfg = cfg_of(rule)
        out[rule] = {}
        for name, (pa, pb) in {"forward": (B_HOLD, B_YIELD), "standoff": (B_YIELD, B_YIELD)}.items():
            rows = play(cfg, fixed(pa), fixed(pb), 6, *start(cfg, strong=True))
            lane, m4, k = rows[-1]["t"].lane.state, rows[-1]["t"].champ, rows[-1]["k"] + 1
            out[rule][name] = [one_turn(cfg, lane, m4, k, x, y) for x in LANE_BEHAVIORS for y in LANE_BEHAVIORS]
    return out


def stage_deterrence() -> dict:
    """E3/E4: a strong A that Holds at the wave and Trades only every 4th turn, against fixed and reactive B rules
    (W turns); and the same A Holding without ever trading."""
    out = {}
    lines = {"A hold+trade/4 vs B Yield": (periodic_trade(4), fixed(B_YIELD)),
             "A hold+trade/4 vs B Hold": (periodic_trade(4), fixed(B_HOLD)),
             "A hold+trade/4 vs B Safe": (periodic_trade(4), fixed(B_SAFE)),
             "A hold+trade/4 vs B respect": (periodic_trade(4), respect),
             "A hold (never trades) vs B Yield": (fixed(B_HOLD), fixed(B_YIELD)),
             "A hold (never trades) vs B respect": (fixed(B_HOLD), respect),
             "A trade every turn vs B Yield": (fixed(B_TRADE), fixed(B_YIELD))}
    for rule in ("m2", "standoff"):
        cfg = cfg_of(rule)
        out[rule] = {name: summarize(play(cfg, pa, pb, W, *start(cfg, strong=True)), cfg)
                     for name, (pa, pb) in lines.items()}
    return out


def stage_intent_switch() -> dict:
    """E4/E5: A changes intent Yield (stand-off) -> Hold (at the wave) -> Yield, 8 turns each; or one Trade from the
    stand-off followed by Hold; B plays the reactive `respect` rule or a fixed Yield. Per-turn cells are reported."""
    out = {}
    a_lines = {"standoff>hold>standoff": [(8, B_YIELD), (8, B_HOLD), (8, B_YIELD)],
               "standoff>trade once>hold": [(8, B_YIELD), (1, B_TRADE), (15, B_HOLD)]}
    for rule in ("m2", "standoff"):
        cfg = cfg_of(rule)
        out[rule] = {}
        for an, blocks in a_lines.items():
            for bn, pb in (("respect", respect), ("Yield", fixed(B_YIELD))):
                rows = play(cfg, schedule(blocks), pb, W, *start(cfg, strong=True))
                s = summarize(rows, cfg)
                s["B_behaviors"] = "".join("HYTSRA"[r["t"].behaviors[B]] for r in rows)
                s["blocks"] = [summarize(rows[i:i + 8], cfg) for i in (0, 8, 16)]
                for blk in s["blocks"]:
                    blk.pop("cells")
                out[rule][f"{an} vs B {bn}"] = s
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    data = {"equal": stage_equal(), "farm_cost": stage_farm_cost(), "matrix": stage_matrix(), "deterrence": stage_deterrence(),
            "intent_switch": stage_intent_switch()}
    data["budget"] = {"turns": wr.TURNS[0], "wall_s": round(time.time() - t0, 2)}
    dump_json({**meta(GameConfig.m4()), **data}, os.path.join(OUT, "positioning.json"))
    keys = ("contact_end_frac", "farm_per_turn", "trades_connected", "evaded", "pinned", "expected_damage_taken",
            "expected_ganks", "net_A_illustrative")
    for stage in ("equal", "farm_cost", "deterrence", "intent_switch"):
        print(f"== {stage}")
        for rule, lines in data[stage].items():
            for name, s in lines.items():
                print(f"  {rule:8s} {name:36s}", {k: s[k] for k in keys})
    print("budget", data["budget"])


if __name__ == "__main__":
    main()
