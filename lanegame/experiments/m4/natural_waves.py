"""M4.2e-A: natural wave mechanics (docs/m4_2e_natural_waves_report.md).

Deterministic trajectories of `GameConfig.m4_natural()`: the M4 resolver with no chance events (u = None: no ganks,
no trade hits) and short scripted policies, plus lane-only traces (`mechanics.step_turn` with the champion's farm-mode
reach, own front + farm_reach). Nothing is trained or sampled.

    python -m lanegame.experiments.m4.natural_waves            all stages, JSON to results/m4_natural_waves/

Stages (report section in brackets):
  timescale   [3] lane-only tick traces of W = 4 and W = 8 under the same natural rules: spawns, contact, merges, deaths
  actions     [5] Hold / Thin / Clear against a Holding opponent, and the ticks each needs to kill a fresh wave
  stacking    [4] one to six turns of Thin, then Hold: does the small edge accumulate into a crash?
  crash       [5] deliberate crashes: build with Thin, then Hold or Clear
  bounce      [7] crash size, clearing time, post-crash fronts, the defender's push back; tower-damage variants
  freeze      [6] a bounded-surplus Thin freeze reached from s0, then broken (or not) by Hold / Thin / Clear
  recall      [8] Recall cost against the wave state: neutral, after own crashes, failed push, enemy crash timing
  reach       [9] champion damage needs presence, reach and availability (Trade, Safe, Recall)
  scripted    natural_suite pairs (correctness and variety of outcomes, not balance)

Coordinates in every trajectory are GLOBAL (A walks +, own tower of A at -2, B's at +2; the lane cell is 0).
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

from ...champions import NO_REACH, champion_legal_actions
from ...champions_m4 import initial_m4, resolve_turn_m4
from ...config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, W_CLEAR, W_HOLD, W_THIN, GameConfig,
                       champ_act)
from ...io import dump_json
from ...mechanics import initial_state, step_turn
from .common import RESULTS_DIR, meta

OUT = os.path.join(RESULTS_DIR, "m4_natural_waves")
A, B = 0, 1
TURNS = [0]

HO, CL, TN = champ_act(W_HOLD, B_HOLD), champ_act(W_CLEAR, B_HOLD), champ_act(W_THIN, B_HOLD)
SAFE = champ_act(W_HOLD, B_SAFE)
RECALL = champ_act(W_HOLD, B_RECALL)


def natural(**kw) -> GameConfig:
    return GameConfig.m4_natural(**kw)


def natural_w4(**kw) -> GameConfig:
    """The same natural rules on the historical 4-HP wave (per-HP economy back at its M4 values)."""
    base = dict(wave_hp=4, farm_per_hp=0.25, xp_per_hp=2, gold_unit=1e-6, siege_cap=48)
    base.update(kw)
    return GameConfig.m4_natural(**base)


# ---------------------------------------------------------------------- resolver driver
def _legal(cfg, lane, m4, s, code) -> int:
    legal = champion_legal_actions(lane, m4, s, cfg)
    if legal == (CHAMP_ABSENT,):
        return CHAMP_ABSENT
    if code in legal:
        return code
    for c in (code % 15, champ_act(code % 3, B_HOLD), HO):           # drop the objective, then the behavior
        if c in legal:
            return c
    raise AssertionError((code, legal))


def play(cfg, policy: Callable, turns: int, lane=None, m4=None, k0: int = 0) -> List[dict]:
    """policy(k, lane, m4) -> (code_A, code_B); returns one row per turn {k, pre, t}."""
    lane = lane if lane is not None else initial_state(cfg)
    m4 = m4 if m4 is not None else initial_m4(cfg)
    rows = []
    for k in range(k0, k0 + turns):
        a, b = policy(k, lane, m4)
        a, b = _legal(cfg, lane, m4, A, a), _legal(cfg, lane, m4, B, b)
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, u=None)
        TURNS[0] += 1
        rows.append({"k": k, "pre": lane, "t": t})
        lane, m4 = t.lane.state, t.champ
    return rows


def end_state(rows):
    t = rows[-1]["t"]
    return t.lane.state, t.champ, rows[-1]["k"] + 1


def contact_cell(st) -> Optional[int]:
    """Global cell of A's front when the two fronts are in contact, else None."""
    if st.ah > 0 and st.bh > 0 and st.ax + st.bx >= -1:
        return st.ax
    return None


def side_str(st, s: int) -> str:
    fx, fh, rears, sg = st.side(s)[:4]
    g = 1 if s == A else -1
    out = f"{g * fx:+d}:{fh}" if fh > 0 else "--"
    if rears:
        out += " r[" + ",".join(f"{g * x:+d}" for x in rears) + "]"
    if sg:
        out += f" sg{sg}"
    return out


def trajectory(rows, cfg) -> List[list]:
    """[k, actions, A lane, B lane, champion cells (global; 'away'), crash HP, farm A/B, xp A/B, tower hits]"""
    out = []
    for r in rows:
        t = r["t"]
        st = t.lane.state
        cells = ["away" if t.away[s] else (t.end[s] if s == A else -t.end[s]) for s in (A, B)]
        out.append([r["k"], [_label(cfg, t, s) for s in (A, B)], side_str(st, A), side_str(st, B), cells,
                    list(t.lane.crash_hp), [round(x, 4) for x in t.farm], list(t.xp_gain), list(t.tower_hits)])
    return out


def _label(cfg, t, s) -> str:
    code = champ_act(t.waves[s], t.behaviors[s], t.objectives[s]) if t.behaviors[s] <= B_RECALL else CHAMP_ABSENT
    return cfg.label(code)


def summarize(rows, cfg) -> dict:
    farm, xp, earned, plates, hits = [0.0, 0.0], [0, 0], [0, 0], [0, 0], [0, 0]
    crashes = [[], []]
    for i, r in enumerate(rows):
        t = r["t"]
        for s in (A, B):
            farm[s] += t.farm[s]
            xp[s] += t.xp_gain[s]
            earned[s] += t.earned_units[s]
            plates[s] += t.plates_broken[s]
            hits[s] += t.tower_hits[s]
            if t.lane.crash_hp[s]:
                crashes[s].append({"k": r["k"], "hp": t.lane.crash_hp[s]})
    cells = [contact_cell(r["t"].lane.state) for r in rows]
    return {"turns": len(rows), "earned_A_minus_B": round((earned[A] - earned[B]) / cfg.gold_scale, 4),
            "farm": [round(x, 4) for x in farm], "xp": xp, "plates": plates, "tower_hits": hits,
            "crashes": crashes, "contact": cells,
            "fronts": [[r["t"].lane.state.ah, r["t"].lane.state.bh] for r in rows]}


# ---------------------------------------------------------------------- stage: timescale (lane only)
def farm_reach(cfg):
    """Lane-only stand-in for the resolver's farm-mode reach: the champion stands at its own front (own tower when it
    has none) and acts up to farm_reach cells ahead."""
    T = cfg.tower

    def f(st):
        out = []
        for s in (A, B):
            fx, fh = st.side(s)[:2]
            pos = max(-T, min(T, fx)) if fh > 0 else -T
            out.append(min(T, pos + cfg.farm_reach))
        return tuple(out)
    return f


def lane_ticks(cfg, wa: int, wb: int, turns: int) -> List[dict]:
    """Tick-level lane trace from s0 with constant wave slots (wa, wb) and farm-mode reach."""
    st, rows, reach = initial_state(cfg), [], farm_reach(cfg)
    for k in range(turns):
        res = step_turn(st, k, wa, wb, cfg, record=True, champion_reach=reach(st))
        TURNS[0] += 1
        for tr in res.trace:
            rows.append({"tick": cfg.ticks_per_turn * k + tr["tick"] + 1, "A": [list(p) for p in tr["A"]],
                         "B": [[-p[0], p[1]] for p in tr["B"]], "fight": tr["fight"], "champion": tr["champion"],
                         "crashed": tr["crashed"], "siege": tr["siege"]})
        if (k + 1) * cfg.ticks_per_turn % cfg.spawn_interval == 0:
            rows[-1]["spawn_after"] = True
        st = res.state
    return rows


def timescale_metrics(rows, cfg) -> dict:
    """Spawn ticks, first contact, each side-A merge and front death (from consecutive tick snapshots)."""
    first_contact = next((r["tick"] for r in rows if r["fight"]), None)
    merges, deaths, max_front = [], [], 0
    prev = None
    for r in rows:
        a = r["A"]
        max_front = max(max_front, a[0][1] if a else 0)
        if prev is not None and prev:
            died = bool(r["fight"]) and r["fight"][1] <= r["fight"][4]          # A's front HP <= damage taken
            if died:
                deaths.append(r["tick"])
            elif a and len(a) < len(prev) and a[0][1] > prev[0][1] - 4:
                merges.append(r["tick"])
        prev = a
    return {"spawn_ticks": [cfg.spawn_interval * i for i in range(1 + rows[-1]["tick"] // cfg.spawn_interval)],
            "first_contact_tick": first_contact, "A_merge_ticks": merges, "A_front_death_ticks": deaths,
            "fresh_fight_ticks_no_champion": cfg.wave_hp,                       # W / (1 HP per tick)
            "reinforcement_spacing_cells": cfg.spawn_interval, "max_A_front": max_front}


def stage_timescale() -> dict:
    out = {}
    for name, cfg in (("W4", natural_w4()), ("W8", natural())):
        for lbl, wa in (("hold_hold", W_HOLD), ("thin_hold", W_THIN)):
            rows = lane_ticks(cfg, wa, W_HOLD, 16)
            out[f"{name}_{lbl}"] = {"metrics": timescale_metrics(rows, cfg), "ticks": rows}
    return out


# ---------------------------------------------------------------------- policies
def const(a, b=HO):
    return lambda k, lane, m4: (a, b)


def window(code, lo, hi, after=HO, b=HO):
    return lambda k, lane, m4: (code if lo <= k <= hi else after, b)


def until_crash(pre: Callable, b=HO):
    """A plays pre(k) until its wave crashes, then Holds (the defender's reply is then unconstrained)."""
    st = {"done": False}

    def p(k, lane, m4):
        if lane.asg > 0:
            st["done"] = True
        return (HO if st["done"] else pre(k), b)
    return p


def bounded_freezer(margin: int = 8):
    """A Thins while the enemy front holds >= margin HP more than its own front in contact, else Holds."""
    def wave(lane):
        if contact_cell(lane) is None:
            return HO
        return TN if lane.bh - lane.ah >= margin else HO
    return wave


# ---------------------------------------------------------------------- stage: actions
def kill_ticks(cfg, slot: int) -> int:
    """Ticks for A (a fresh wave, champion at its front) to kill a fresh B wave already in contact, with no
    reinforcement on either side (spawns pushed out of the window; lane only)."""
    from ...state import State
    cfg = cfg.with_(spawn_interval=100 * cfg.ticks_per_turn)
    st = State(0, cfg.wave_hp, (), 0, -1, cfg.wave_hp, (), 0, 0, 0)
    for k in range(1, 20):
        res = step_turn(st, k, slot, W_HOLD, cfg, record=True, champion_reach=(cfg.tower, NO_REACH))
        TURNS[0] += 1
        for i, tr in enumerate(res.trace):
            if tr["fight"] and tr["fight"][3] <= tr["fight"][5]:
                return 2 * (k - 1) + i + 1
        st = res.state
    return -1


def stage_actions(cfg) -> dict:
    lines = {}
    for name, code in (("hold", HO), ("thin", TN), ("clear", CL)):
        lines[name] = summarize(play(cfg, const(code), 40), cfg)
    return {"kill_fresh_wave_ticks": {n: kill_ticks(cfg, w) for n, w in (("hold", W_HOLD), ("thin", W_THIN),
                                                                           ("clear", W_CLEAR))},
            "vs_hold_40": lines,
            "clear_trace": trajectory(play(cfg, const(CL), 14), cfg)}


# ---------------------------------------------------------------------- stage: stacking and deliberate crashes
def stage_stacking(cfg) -> dict:
    out = {}
    for n in (1, 2, 4, 6):
        rows = play(cfg, window(TN, 1, n), 40)
        out[f"thin_k1_{n}"] = summarize(rows, cfg)
        if n == 1:
            out["thin_k1_1"]["trajectory"] = trajectory(rows, cfg)
    return out


def stage_crash(cfg) -> dict:
    out = {}
    pols = {"thin_k1_3_hold": lambda k: TN if 1 <= k <= 3 else HO,
            "thin_k1_3_clear_k8": lambda k: TN if 1 <= k <= 3 else (CL if k >= 8 else HO),
            "thin_k1_3_clear_k10": lambda k: TN if 1 <= k <= 3 else (CL if k >= 10 else HO),
            "thin_always": lambda k: TN,
            "clear_always": lambda k: CL}
    for name, f in pols.items():
        out[name] = summarize(play(cfg, lambda k, l, m, f=f: (f(k), HO), 24), cfg)
    return out


# ---------------------------------------------------------------------- stage: bounce
def crash_report(rows) -> Optional[dict]:
    """A's first crash and what follows: clearing time, fronts when the mass is gone, the defender's push back."""
    ci = next((i for i, r in enumerate(rows) if r["t"].lane.crash_hp[A]), None)
    if ci is None:
        return None
    end = next((i for i in range(ci, len(rows)) if rows[i]["t"].lane.state.asg == 0), None)
    rep = {"crash_k": rows[ci]["k"], "crash_hp": rows[ci]["t"].lane.crash_hp[A],
           "followup_crash_hp": sum(r["t"].lane.crash_hp[A] for r in rows[ci + 1:(end or ci) + 1]),
           "clear_turns": None if end is None else end - ci,
           "tower_hits_during_siege": sum(r["t"].tower_hits[A] for r in rows[ci:(end if end is not None else len(rows)) + 1])}
    if end is not None:
        st = rows[end]["t"].lane.state
        rep["at_clear"] = {"A_front": st.ah, "B_front": st.bh, "B_front_cell": -st.bx if st.bh else None}
        deep = next((rows[j]["k"] - rows[ci]["k"] for j in range(end, len(rows))
                     if (c := contact_cell(rows[j]["t"].lane.state)) is not None and c <= -1), None)
        back = next(((rows[j]["k"] - rows[ci]["k"], rows[j]["t"].lane.crash_hp[B]) for j in range(end, len(rows))
                     if rows[j]["t"].lane.crash_hp[B]), None)
        rep["defender_in_attacker_half_after_turns"] = deep
        rep["bounce_crash"] = None if back is None else {"after_turns": back[0], "hp": back[1]}
    return rep


BOUNCE_LINES = {"thin_k1_hold": lambda k: TN if k == 1 else HO,
                "thin_k1_4_hold": lambda k: TN if 1 <= k <= 4 else HO,
                "thin_until_crash": lambda k: TN,
                "thin_k1_3_clear_k8": lambda k: TN if 1 <= k <= 3 else (CL if k >= 8 else HO)}


def stage_bounce(cfg) -> dict:
    out = {"lines": {}, "tower_variants": {}}
    for name, f in BOUNCE_LINES.items():
        rows = play(cfg, until_crash(f), 44)
        out["lines"][name] = {"report": crash_report(rows), "contact": [contact_cell(r["t"].lane.state) for r in rows]}
        if name == "thin_until_crash":
            out["lines"][name]["trajectory"] = trajectory(rows, cfg)
    for tsd, td in ((2, 1), (2, 2)):
        c2 = natural(tower_siege_damage=tsd, tower_damage=td)
        out["tower_variants"][f"siege{tsd}_contested{td}"] = {
            name: crash_report(play(c2, until_crash(BOUNCE_LINES[name]), 40))
            for name in ("thin_until_crash", "thin_k1_4_hold")}
    return out


# ---------------------------------------------------------------------- stage: freeze
def stage_freeze(cfg) -> dict:
    fz = bounded_freezer(8)
    setup = play(cfg, lambda k, l, m: (fz(l), TN if k == 1 else HO), 30)
    lane0, m40, k0 = end_state(setup)
    out = {"setup": summarize(setup, cfg), "setup_trajectory": trajectory(setup[22:], cfg),
           "state_k30": [side_str(lane0, A), side_str(lane0, B)], "breaks": {}}
    for name, bw in (("hold", HO), ("thin", TN), ("clear", CL)):
        rows = play(cfg, lambda k, l, m, bw=bw: (fz(l), bw), 16, lane0, m40, k0)
        s = summarize(rows, cfg)
        s["held_turns"] = next((i for i, c in enumerate(s["contact"]) if c is None or c > -1), len(rows))
        out["breaks"][name] = s
    # the same freeze state without the freezer's control (A Holds): how much of the hold is the tower's doing?
    s = summarize(play(cfg, const(HO), 30, lane0, m40, k0), cfg)
    s["held_turns"] = next((i for i, c in enumerate(s["contact"]) if c is None or c > -1), 30)
    out["uncontrolled_from_k30"] = s
    # band sweep of the freezer from s0 (B Thins on k1 only): turns with the contact at A's tower side
    out["band_sweep"] = {}
    for margin in (2, 4, 6, 8, 10):
        f = bounded_freezer(margin)
        s = summarize(play(cfg, lambda k, l, m, f=f: (f(l), TN if k == 1 else HO), 50), cfg)
        out["band_sweep"][margin] = {"turns_contact_at_or_behind_-1": sum(1 for c in s["contact"]
                                                                          if c is not None and c <= -1),
                                     "crashes": s["crashes"], "farm": s["farm"]}
    return out


# ---------------------------------------------------------------------- stage: recall
def recall_line(pre: Callable, trigger: Callable, b: Callable = lambda k: HO):
    """A plays pre(k, lane) until trigger(k, lane); then walks back (Safe) until Recall is legal, Recalls, Holds."""
    st = {"phase": 0, "recall_k": None}

    def p(k, lane, m4):
        if st["phase"] == 0 and trigger(k, lane):
            st["phase"] = 1
        if st["phase"] == 1:
            if m4.a.absent == 0 and m4.a.pos <= GameConfig.m4_natural().recall_line:
                st["phase"], st["recall_k"] = 2, k
                return (RECALL, b(k))
            return (SAFE, b(k))
        if st["phase"] == 2:
            return (HO, b(k))
        return (pre(k, lane), b(k))
    return p, st


def _recall_pair(cfg, pre, trigger, turns=40, b=lambda k: HO) -> dict:
    p_rec, st = recall_line(pre, trigger, b)
    rec = play(cfg, p_rec, turns)
    stay = play(cfg, lambda k, l, m: (pre(k, l), b(k)), turns)
    sr, ss = summarize(rec, cfg), summarize(stay, cfg)
    rk = st["recall_k"]
    back = rk + cfg.recall_turns if rk is not None else None
    lane_back = lambda rows: None if back is None or back - 1 >= len(rows) else \
        [side_str(rows[back - 1]["t"].lane.state, A), side_str(rows[back - 1]["t"].lane.state, B)]
    return {"recall_k": rk, "back_k": back,
            "missed_gold": round(ss["farm"][A] - sr["farm"][A], 4), "missed_xp": ss["xp"][A] - sr["xp"][A],
            "cost_A_minus_B": round(ss["earned_A_minus_B"] - sr["earned_A_minus_B"], 4),
            "lane_at_return": lane_back(rec), "lane_at_return_if_stayed": lane_back(stay),
            "crashes_recall": sr["crashes"], "crashes_stay": ss["crashes"]}


def _crashed(pre: Callable):
    s = {"c": False}

    def f(k, lane):
        if lane.asg > 0:
            s["c"] = True
        return HO if s["c"] else pre(k)
    return f


def stage_recall(cfg) -> dict:
    out = {"own_state": {
        "neutral_k12": _recall_pair(cfg, lambda k, l: HO, lambda k, l: k >= 12),
        "after_small_crash": _recall_pair(cfg, _crashed(lambda k: TN), lambda k, l: l.asg > 0),
        "after_large_crash": _recall_pair(cfg, _crashed(lambda k: TN if 1 <= k <= 4 else HO), lambda k, l: l.asg > 0),
        "after_failed_push": _recall_pair(cfg, lambda k, l: CL if k <= 6 else HO, lambda k, l: k >= 8)}}
    # the enemy builds (Thin k1-4) and crashes 25 HP into A's tower at k15; A recalls at different times
    bpol = lambda k: TN if 1 <= k <= 4 else HO
    out["enemy_crash_k15"] = {f"recall_from_k{rk}": _recall_pair(cfg, lambda k, l: HO, lambda k, l, rk=rk: k >= rk,
                                                                 36, bpol)
                              for rk in (6, 11, 13, 15, 17, 20)}
    return out


# ---------------------------------------------------------------------- stage: reach
def stage_reach(cfg) -> dict:
    """From the neutral contact at the lane (k9), one turn of A's Thin under different behaviors: what reaches B?"""
    lane, m4, k0 = end_state(play(cfg, const(HO), 9))
    out = {}
    for name, code in (("thin_hold", TN), ("thin_trade_connected", champ_act(W_THIN, B_TRADE)),
                       ("thin_safe", champ_act(W_THIN, B_SAFE)), ("clear_hold", CL),
                       ("clear_trade_connected", champ_act(W_CLEAR, B_TRADE)), ("hold_hold", HO),
                       ("recall", RECALL)):
        b = champ_act(W_HOLD, B_TRADE) if "trade" in name else HO
        if code == RECALL and RECALL not in champion_legal_actions(lane, m4, A, cfg):
            continue
        t = resolve_turn_m4(lane, m4, k0, code, b, cfg, u=None)
        TURNS[0] += 1
        out[name] = {"connected": list(t.connected), "reach_A": t.reach[A], "mode_A": t.modes[A],
                     "champion_hp_on_B": t.lane.events["champion_hp"][B], "B_front": t.lane.state.bh,
                     "farm_A": round(t.farm[A], 4)}
    return out


# ---------------------------------------------------------------------- stage: scripted pairs
def stage_scripted(cfg, turns: int = 100) -> dict:
    from ...env.agents_m4 import natural_suite
    from ...env.core import LaneEnv
    import numpy as np
    su = natural_suite()
    pairs = (("slow_pusher", "hold_farmer"), ("crasher", "hold_farmer"), ("natural_freezer", "slow_pusher"),
             ("clearer", "thinner"), ("crash_recaller", "natural_freezer"), ("trading_pusher", "hold_farmer"))
    out = {}
    for x, y in pairs:
        env = LaneEnv(cfg, max_turns=turns)
        obs = env.reset(seed=0)
        rng = np.random.default_rng(0)
        tot, crashes = 0.0, [0, 0]
        while True:
            r = env.step(su[x].act(obs[0], rng), su[y].act(obs[1], rng))
            TURNS[0] += 1
            tot += r.rewards[0]
            for s in (A, B):
                crashes[s] += r.info["crash_hp"][s] > 0 if "crash_hp" in r.info else 0
            if r.done:
                break
            obs = r.obs
        out[f"{x}_vs_{y}"] = {"reward_A_sum": round(tot, 4), "crashes": crashes}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cfg = natural()
    data = {"timescale": stage_timescale(), "actions": stage_actions(cfg), "stacking": stage_stacking(cfg),
            "crash": stage_crash(cfg), "bounce": stage_bounce(cfg), "freeze": stage_freeze(cfg),
            "recall": stage_recall(cfg), "reach": stage_reach(cfg), "scripted": stage_scripted(cfg)}
    data["budget"] = {"turns": TURNS[0], "wall_s": round(time.time() - t0, 2)}
    dump_json({**meta(cfg), **data}, os.path.join(OUT, "natural_waves.json"))
    ts = data["timescale"]
    for k, v in ts.items():
        print("timescale", k, {m: v["metrics"][m] for m in ("first_contact_tick", "A_merge_ticks", "A_front_death_ticks",
                                                            "max_A_front")})
    print("kill fresh wave ticks", data["actions"]["kill_fresh_wave_ticks"])
    for n, s in data["actions"]["vs_hold_40"].items():
        print(f"  {n:6s} vs hold: A-B {s['earned_A_minus_B']:+.3f} farm {s['farm']} crashes {s['crashes'][A]}")
    for n, s in data["stacking"].items():
        print(f"stacking {n}: crashes A {s['crashes'][A]} B {s['crashes'][B]} A-B {s['earned_A_minus_B']:+.3f}")
    for n, s in data["crash"].items():
        print(f"crash {n}: {s['crashes'][A]}")
    for n, s in data["bounce"]["lines"].items():
        print(f"bounce {n}: {s['report']}")
    for n, s in data["bounce"]["tower_variants"].items():
        print(f"tower {n}: {s}")
    fr = data["freeze"]
    print("freeze state k30", fr["state_k30"], "setup contact", fr["setup"]["contact"])
    for n, s in fr["breaks"].items():
        print(f"  break {n}: held {s['held_turns']} crashes {s['crashes']} farm {s['farm']} contact {s['contact']}")
    print("  uncontrolled (A Holds) from k30: held", fr["uncontrolled_from_k30"]["held_turns"],
          "farm", fr["uncontrolled_from_k30"]["farm"], "crashes", fr["uncontrolled_from_k30"]["crashes"])
    print("  band sweep", fr["band_sweep"])
    for grp, d in data["recall"].items():
        for n, s in d.items():
            print(f"recall {grp} {n}: rk {s['recall_k']} missed {s['missed_gold']} / {s['missed_xp']} xp "
                  f"cost {s['cost_A_minus_B']:+.3f}")
    print("reach", data["reach"])
    print("scripted", data["scripted"])
    print("budget", data["budget"])


if __name__ == "__main__":
    main()
