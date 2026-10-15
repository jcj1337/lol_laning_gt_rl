"""M4.2d-A: wave bounce and Freeze-breaking readiness (docs/m4/m4_2/m4_wave_readiness_report.md).

Deterministic trajectories of the M4 resolver with no chance events (u = None: no ganks, no trade hits; gank exposure
is reported as the summed gank probability). Every policy is a short scripted rule; nothing is trained or sampled.

    python -m lanegame.experiments.m4.wave_readiness            all stages, JSON to results/m4_wave_readiness/

Stages:
  scenarios   the four assignment scenarios and the strategy comparison, under farm_spot_rule "m2" (default) and
              "pre_crash_front" (the opt-in Freeze-breaking correction)
  sweep       48 one-off Shove windows from s0 against a passive defender: does any crash bounce?
  candidates  EXPERIMENT ONLY: candidate restoring-force rules, applied by patching mechanics._tick for the duration
              of the stage (never a ruleset; the engine file is not changed)

Start states: "pushed" = s0 after one 3-turn Shove by A on turns 5-7 (A then holds a stacked wave at the lane);
"neutral" = s0 after 9 turns of mutual Slow Push (symmetric contact at the lane).
"""

from __future__ import annotations

import contextlib
import inspect
import os
import time
from typing import Callable, Dict, List

from ... import mechanics
from ...champions import champion_legal_actions
from ...champions_m4 import initial_m4, resolve_turn_m4
from ...config import B_ABSENT, B_HOLD, B_RECALL, B_SAFE, CHAMP_ABSENT, F, NONE, SH, SP, GameConfig, champ_act
from ...io import dump_json
from ...mechanics import initial_state, wave_actions
from .common import RESULTS_DIR, meta

OUT = os.path.join(RESULTS_DIR, "m4_wave_readiness")
A, B = 0, 1
TURNS = [0]


# ---------------------------------------------------------------------- driver
def _code(cfg, lane, m4, s, want) -> int:
    legal = champion_legal_actions(lane, m4, s, cfg)
    if legal == (CHAMP_ABSENT,):
        return CHAMP_ABSENT
    w, b = want
    if b == B_RECALL:
        c = champ_act(SP, B_RECALL)
        return c if c in legal else champ_act(SP, B_SAFE)
    if w == F and F not in wave_actions(lane, s, cfg):
        w = SP
    c = champ_act(w, b, NONE)
    assert c in legal, (want, legal)
    return c


def play(cfg, pa, pb, turns, lane=None, m4=None, k0=0) -> List[dict]:
    lane = lane if lane is not None else initial_state(cfg)
    m4 = m4 if m4 is not None else initial_m4(cfg)
    rows = []
    for i in range(turns):
        k = k0 + i
        a, b = _code(cfg, lane, m4, A, pa(cfg, lane, m4, A, k)), _code(cfg, lane, m4, B, pb(cfg, lane, m4, B, k))
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, u=None)
        TURNS[0] += 1
        rows.append({"k": k, "pre": lane, "t": t})
        lane, m4 = t.lane.state, t.champ
    return rows


def end_state(rows):
    t = rows[-1]["t"]
    return t.lane.state, t.champ, rows[-1]["k"] + 1


# ---------------------------------------------------------------------- policies (cfg, lane, m4, side, k) -> (wave, behavior)
def hold(cfg, lane, m4, s, k):
    return (SP, B_HOLD)


def shove_window(lo, hi):
    return lambda cfg, lane, m4, s, k: (SH if lo <= k < hi else SP, B_HOLD)


class Pusher:
    """Slow Push until the own front holds `stack` HP, Shove until it crashes, then Slow Push (farm) - or, with
    `recall`, walk back (Safe) and Recall as soon as the own crashed mass reaches `recall_mass` HP."""

    def __init__(self, stack=8, recall=False, recall_mass=1, shove=True):
        self.stack, self.recall, self.recall_mass, self.shove = stack, recall, recall_mass, shove
        self.phase = "build" if shove else "after"

    def __call__(self, cfg, lane, m4, s, k):
        fh, sg = lane.side(s)[1], lane.side(s)[3]
        me = m4.side(s)
        if me.absent:
            return (SP, B_ABSENT)
        if self.phase in ("build", "shove") and sg > 0:
            self.phase = "after"
        if self.phase == "build" and fh >= self.stack:
            self.phase = "shove"
        if self.phase == "shove":
            return (SH, B_HOLD)
        if self.phase == "after" and self.recall and sg >= self.recall_mass:
            self.phase = "recall"
        if self.phase == "recall":
            if me.pos <= cfg.recall_line:
                self.phase = "done"
                return (SP, B_RECALL)
            return (SP, B_SAFE)
        return (SP, B_HOLD)


class RecallAt:
    """Hold; from turn `at` walk back and Recall once (a Recall from a neutral lane)."""

    def __init__(self, at):
        self.at, self.done = at, False

    def __call__(self, cfg, lane, m4, s, k):
        me = m4.side(s)
        if me.absent:
            return (SP, B_ABSENT)
        if k >= self.at and not self.done:
            if me.pos <= cfg.recall_line:
                self.done = True
                return (SP, B_RECALL)
            return (SP, B_SAFE)
        return (SP, B_HOLD)


def control(cfg, lane, m4, s, k):
    """Lane-control defender: Freeze when legal, else Shove while the own front holds at least the enemy front's HP."""
    fh, eh = lane.side(s)[1], lane.side(1 - s)[1]
    if F in wave_actions(lane, s, cfg):
        return (F, B_HOLD)
    return (SH if fh > 0 and fh >= eh else SP, B_HOLD)


def punish_absence(cfg, lane, m4, s, k):
    """Shove while the opponent is away, else Slow Push."""
    return (SH if m4.side(1 - s).absent > 0 else SP, B_HOLD)


# ---------------------------------------------------------------------- measurements
def crash_events(rows, s=A) -> List[dict]:
    """Crashes of side s with their mass and the turns the enemy tower needs to clear the crashed mass."""
    out = []
    for i, r in enumerate(rows):
        m = r["t"].lane.crash_hp[s]
        if m:
            clear = next((j - i for j in range(i, len(rows)) if rows[j]["t"].lane.state.side(s)[3] == 0), None)
            out.append({"k": r["k"], "hp": m, "clear_turns": clear})
    return out


def summarize(rows, cfg) -> dict:
    t = [r["t"] for r in rows]
    g = cfg.gold_scale
    first = next((r["k"] for r in rows if r["t"].lane.crash_hp[A]), None)
    after = [r for r in rows if first is not None and r["k"] > first]
    away = [r for r in rows if r["t"].away[A]]
    ret = next((r for r in rows if r["t"].returned[A]), None)
    return {
        "turns": len(rows),
        "farm": [round(sum(x.farm[s] for x in t), 3) for s in (A, B)],
        "farm_available": [round(sum(x.farm_available[s] for x in t), 3) for s in (A, B)],
        "earned_A_minus_B": round(sum(x.reward_units for x in t) / g, 3),
        "xp": [sum(x.xp_gain[s] for x in t) for s in (A, B)],
        "plates": [sum(x.plates_broken[s] for x in t) for s in (A, B)],
        "tower_dmg_A": t[-1].champ.a.tower_dmg,
        "crashes": [crash_events(rows, A), crash_events(rows, B)],
        # the bounce: after A's first crash, B's wave reaches A's half (B front at its own x >= 1)
        "bounce": any(r["t"].lane.state.bh > 0 and r["t"].lane.state.bx >= 1 for r in after),
        "turns_A_away": [r["k"] for r in away],
        "missed_gold_A": round(sum(r["t"].farm_available[A] for r in away), 3),
        "missed_xp_A": sum(cfg.xp_per_hp * sum(r["t"].lane.kills[A].values()) for r in away),
        "lane_on_return": ret["t"].lane.state.describe() if ret else None,
        "expected_ganks_A": round(sum(x.gank_p[A] for x in t), 3),
        "max_pos_A": max(x.end[A] for x in t),
        "end_lane": t[-1].lane.state.describe(),
    }


def trajectory(rows) -> List[list]:
    """Compact per-turn rows: k, A front (x, hp), A rears, A siege, B front, B rears, B siege, champion cells
    (None while away), A crash HP, farm A/B."""
    out = []
    for r in rows:
        t = r["t"]
        st = t.lane.state
        out.append([r["k"], [st.ax, st.ah], list(st.ar), st.asg, [st.bx, st.bh], list(st.br), st.bsg,
                    [None if t.away[s] else t.end[s] for s in (A, B)], t.lane.crash_hp[A],
                    [round(t.farm[A], 3), round(t.farm[B], 3)]])
    return out


def starts(cfg):
    pushed = end_state(play(cfg, shove_window(5, 8), hold, 9))
    neutral = end_state(play(cfg, hold, hold, 9))
    return {"pushed": pushed, "neutral": neutral}


LINES: Dict[str, Callable] = {
    # name: (start, A policy factory, B policy, window)
    "hold": ("pushed", lambda: Pusher(shove=False), hold, 48),
    "shove_farm": ("pushed", lambda: Pusher(8), hold, 48),
    "shove_recall_small_crash": ("pushed", lambda: Pusher(8, recall=True), hold, 48),
    "hold_recall_large_crash": ("pushed", lambda: Pusher(shove=False, recall=True, recall_mass=7), hold, 48),
    "shove_farm_vs_control": ("pushed", lambda: Pusher(8), control, 48),
    "neutral_hold": ("neutral", lambda: hold, hold, 40),
    "neutral_recall": ("neutral", lambda: RecallAt(10), hold, 40),
    "neutral_recall_punished": ("neutral", lambda: RecallAt(10), punish_absence, 40),
}


def stage_scenarios(rules=("m2", "pre_crash_front"), traces=("shove_farm", "hold_recall_large_crash",
                                                               "shove_farm_vs_control")) -> dict:
    out = {}
    for rule in rules:
        cfg = GameConfig.m4(farm_spot_rule=rule)
        st = starts(cfg)
        res = {}
        for name, (start, fa, pb, w) in LINES.items():
            lane, m4, k0 = st[start]
            rows = play(cfg, fa(), pb, w, lane, m4, k0)
            res[name] = summarize(rows, cfg)
            if name in traces:
                res[name]["trajectory"] = trajectory(rows)
        out[rule] = {"starts": {k: v[0].describe() for k, v in st.items()}, "lines": res}
    return out


def stage_sweep(rule="m2") -> dict:
    cfg = GameConfig.m4(farm_spot_rule=rule)
    cells = []
    for lo in range(3, 9):
        for n in range(1, 9):
            rows = play(cfg, shove_window(lo, lo + n), hold, lo + n + 56)
            s = summarize(rows, cfg)
            tail = rows[-8:]
            cells.append({"shove_turns": [lo, lo + n - 1], "crashes": len(s["crashes"][A]), "bounce": s["bounce"],
                          "recrashing": any(r["t"].lane.crash_hp[A] for r in rows[-16:]),
                          "end_fronts": [rows[-1]["t"].lane.state.ax, rows[-1]["t"].lane.state.bx],
                          "tail_farm_gap_per_turn": round(sum(r["t"].farm[A] - r["t"].farm[B] for r in tail) / 8, 3)})
    return {"rule": rule, "cells": cells, "with_crash": sum(c["crashes"] > 0 for c in cells),
            "bounces": sum(c["bounce"] for c in cells)}


# ---------------------------------------------------------------------- candidate rules (experiment only)
_SP_LINE = "            elif w == SP:\n                want = _sp_moves(x, h, t, k, cfg)\n"
CANDIDATES = {
    # P2: while a side's crashed mass is under the enemy tower, its Slow Push fronts do not park
    "P2_no_parking_during_own_siege": "siege[s] > 0 or _sp_moves(x, h, t, k, cfg)",
    # C1: a Slow Push front past the lane (own x >= 1) keeps walking
    "C1_no_parking_past_lane": "x >= 1 or _sp_moves(x, h, t, k, cfg)",
    # C3 + P2: only an UNSTACKED front past the lane keeps walking (stacks still creep), plus P2
    "C3P2_unstacked_walks_plus_P2":
        "(x >= 1 and _blob_hp(packets, cfg) <= cfg.wave_hp) or siege[s] > 0 or _sp_moves(x, h, t, k, cfg)",
}


@contextlib.contextmanager
def patched_sp_rule(expr: str):
    """Temporarily replace the Slow Push movement condition of mechanics._tick (experiment only)."""
    src = inspect.getsource(mechanics._tick)
    assert src.count(_SP_LINE) == 1, "mechanics._tick changed; update the candidate patch"
    ns = dict(mechanics.__dict__)
    exec(src.replace(_SP_LINE, f"            elif w == SP:\n                want = {expr}\n"), ns)
    orig = mechanics._tick
    mechanics._tick = ns["_tick"]
    try:
        yield
    finally:
        mechanics._tick = orig


def stage_candidates(names=("hold", "shove_farm", "shove_recall_small_crash", "shove_farm_vs_control",
                            "neutral_hold")) -> dict:
    cfg = GameConfig.m4()
    out = {}
    for cand, expr in CANDIDATES.items():
        with patched_sp_rule(expr):
            st = starts(cfg)
            out[cand] = {"rule": expr, "lines": {}}
            for name in names:
                start, fa, pb, w = LINES[name]
                lane, m4, k0 = st[start]
                s = summarize(play(cfg, fa(), pb, w, lane, m4, k0), cfg)
                out[cand]["lines"][name] = {k: s[k] for k in ("earned_A_minus_B", "farm", "xp", "crashes", "bounce",
                                                              "missed_gold_A", "end_lane")}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    data = {"scenarios": stage_scenarios(), "sweep": stage_sweep(), "candidates": stage_candidates()}
    data["budget"] = {"turns": TURNS[0], "wall_s": round(time.time() - t0, 2)}
    dump_json({**meta(GameConfig.m4()), **data}, os.path.join(OUT, "wave_readiness.json"))
    for rule, d in data["scenarios"].items():
        print(f"farm_spot_rule = {rule}")
        for name, s in d["lines"].items():
            print(f"  {name:26s} A-B {s['earned_A_minus_B']:+6.2f} farm {s['farm']} xp {s['xp']} "
                  f"crashes {[(c['k'], c['hp'], c['clear_turns']) for c in s['crashes'][A]]} bounce {s['bounce']} "
                  f"missed {s['missed_gold_A']}/{s['missed_xp_A']}xp E[ganks] {s['expected_ganks_A']}")
    sw = data["sweep"]
    print(f"sweep: {len(sw['cells'])} windows, {sw['with_crash']} with a crash, {sw['bounces']} bounces")
    for cand, d in data["candidates"].items():
        print(cand, {k: v["earned_A_minus_B"] for k, v in d["lines"].items()},
              "bounce", [v["bounce"] for v in d["lines"].values()])
    print("budget", data["budget"])


if __name__ == "__main__":
    main()
