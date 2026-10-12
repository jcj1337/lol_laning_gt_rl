"""M4.2e-A.1: finalizing natural wave mechanics (docs/natural_waves/m4_2e_natural_waves_final_review.md).

Deterministic trajectories of the final preset `GameConfig.m4_natural()` against the M4.2e-A prototype
`GameConfig.m4_natural_prototype()`: the M4 resolver with no chance events (u = None) and short scripted policies, plus
lane-only traces where noted. Nothing is trained or sampled.

    python -m lanegame.experiments.m4.natural_waves_final       all stages, JSON to results/m4_natural_waves/

Stages (review section in brackets):
  siege        [1, 5] prototype: who removes the crashed mass, and does the siege's length drive the bounce?
  sustain      [2, 5] entry crash rule: tower damage against the reinforcement stream (self-sustaining sieges)
  actions      [7] Hold / Thin / Clear, slow pushes and push + Pressure against a Holding opponent, 80 turns; the
               collection-rate sensitivity of the same lines
  power        [4] Clear by attack power: from a neutral stacked lane, from s0, the Thin-scaling counterfactual, and
               Pressure during a siege with and without the extra tower reach; the budget table by level and investment
  freeze       [6] outside-tower freeze: reach (setups x margins), maintenance, uncontrolled hold, breaks
  bounce       [5] crash size, siege length, defender front, bounce timing; the long push-bounce cycle
  recall       [7] Recall cost against the wave state (the M4.2e-A stage on the final preset)
  reach        [3] champion damage needs presence and reach (the M4.2e-A stage on the final preset)
  granularity  [3] lane only: the integer floor(h / 8) law against a half-wave resolution law
  scripted     [9] natural_suite pairs (correctness and variety, not balance)
  performance  [9] lane step and env step, prototype against final

Coordinates are GLOBAL (A walks +, A's tower at -2, B's at +2, the lane cell is 0). AP multipliers in the power stage
override the attack-power budget of side A only (`play_pow`); the budget table gives the (level, investment) pairs that
reach them.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

from ...champions_m4 import context, initial_m4, resolve_turn_m4
from ...config import (B_HOLD, PRESSURE, W_CLEAR, W_HOLD, W_THIN, GameConfig, champ_act)
from ...io import dump_json
from ...mechanics import initial_state, step_turn
from . import natural_waves as nw
from .common import meta

OUT = nw.OUT
A, B = 0, 1
TURNS = [0]
HO, CL, TN = nw.HO, nw.CL, nw.TN
HOP, CLP = champ_act(W_HOLD, B_HOLD, PRESSURE), champ_act(W_CLEAR, B_HOLD, PRESSURE)


def final(**kw) -> GameConfig:
    return GameConfig.m4_natural(**kw)


def proto(**kw) -> GameConfig:
    return GameConfig.m4_natural_prototype(**kw)


# ---------------------------------------------------------------------- drivers
def play(cfg, policy: Callable, turns: int, lane=None, m4=None, k0: int = 0, mult=None) -> List[dict]:
    """nw.play, optionally with fixed wave-clear multipliers per side: the budget of a scaled slot is
    round(ticks_per_turn * champion_wave_damage[slot] * mult[side]) instead of the attack-power budget."""
    if mult is None:
        rows = nw.play(cfg, policy, turns, lane, m4, k0)
        TURNS[0] += len(rows)
        return rows
    lane = lane if lane is not None else initial_state(cfg)
    m4 = m4 if m4 is not None else initial_m4(cfg)

    def lane_step(s, kk, x, y, reach, budget=None):
        b = tuple(int(round(cfg.ticks_per_turn * cfg.champion_wave_damage[w]
                            * (mult[i] if w in cfg.wave_clear_scaled_slots else 1)))
                  for i, w in enumerate((x % 3, y % 3)))
        return step_turn(s, kk, x, y, cfg, champion_reach=reach, champion_budget=b)
    rows = []
    for k in range(k0, k0 + turns):
        a, b = policy(k, lane, m4)
        a, b = nw._legal(cfg, lane, m4, A, a), nw._legal(cfg, lane, m4, B, b)
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, lane_step=lane_step, u=None)
        TURNS[0] += 1
        rows.append({"k": k, "pre": lane, "t": t})
        lane, m4 = t.lane.state, t.champ
    return rows


def sieges(rows, s: int = A) -> List[dict]:
    """Each siege of side s: first crash turn, first crash HP, all HP that entered it (follow-ups included), its length
    in turns and the defender's front HP when it ends."""
    out, cur = [], None
    for r in rows:
        t = r["t"]
        st = t.lane.state
        if t.lane.crash_hp[s]:
            if cur is None:
                cur = {"k": r["k"], "first_hp": t.lane.crash_hp[s], "total_hp": 0, "turns": 0, "defender_front": None}
            cur["total_hp"] += t.lane.crash_hp[s]
        if cur is not None:
            cur["turns"] += 1
            if st.siege_of(s) == 0:
                cur["defender_front"] = st.bh if s == A else st.ah
                out.append(cur)
                cur = None
    if cur is not None:
        out.append(cur)
    return out


def held_turns(cells, start: int = 0) -> int:
    n = 0
    for c in cells[start:]:
        if c is None or c > -1:
            break
        n += 1
    return n


def longest_hold(cells) -> int:
    return max((held_turns(cells, i) for i in range(len(cells))), default=0)


def line(rows, cfg) -> dict:
    s = nw.summarize(rows, cfg)
    return {"A_minus_B": s["earned_A_minus_B"], "farm": s["farm"], "plates": s["plates"], "tower_hits": s["tower_hits"],
            "sieges_A": sieges(rows, A)[:6], "sieges_B": sieges(rows, B)[:6]}


# ---------------------------------------------------------------------- stage: siege (prototype diagnosis)
def stage_siege() -> dict:
    """Prototype: per turn of A's first siege (A Thins until it crashes, then Holds), what removed the crashed mass and
    what B's front was fighting; then the tower-damage sweep of crash length against bounce timing."""
    cfg = proto()
    rows = play(cfg, nw.until_crash(lambda k: TN), 44)
    turns, tot = [], {"tower": 0, "defenders": 0, "tower_on_followups": 0, "ticks_defender_vs_followup": 0,
                      "ticks_defender_vs_mass": 0}
    for r in rows:
        k = r["k"]
        if r["pre"].asg == 0 and not r["t"].lane.crash_hp[A]:
            continue
        tr = step_turn(r["pre"], k, r["t"].lane_codes[A], r["t"].lane_codes[B], cfg, record=True,
                       champion_reach=r["t"].reach).trace
        TURNS[0] += 1
        lane_fights = sum(1 for x in tr if x["fight"])
        siege_fights = sum(1 for x in tr if x["siege_fight"])
        tower_on_front = sum(1 for x in tr if x["fight"] and x["fight"][0] == cfg.tower)
        ev = r["t"].lane.events
        tot["tower"] += ev["tower_hp"][A]
        tot["defenders"] += ev["siege_killed_hp"][A]
        tot["tower_on_followups"] += tower_on_front * cfg.tower_damage
        tot["ticks_defender_vs_followup"] += lane_fights
        tot["ticks_defender_vs_mass"] += siege_fights
        st = r["t"].lane.state
        turns.append({"k": k, "mass": st.asg, "A": nw.side_str(st, A), "B": nw.side_str(st, B),
                      "tower_hp": ev["tower_hp"][A], "defender_hp": ev["siege_killed_hp"][A],
                      "lane_fight_ticks": lane_fights, "siege_fight_ticks": siege_fights, "tower_hits": r["t"].tower_hits[A]})
        if st.asg == 0 and k > rows[0]["k"] + 2:
            break
    sweep = {}
    for tsd in (1, 2, 3, 4):
        for td in (1, 2):
            c = proto(tower_siege_damage=tsd, tower_damage=td)
            sweep[f"siege{tsd}_contested{td}"] = {
                n: nw.crash_report(play(c, nw.until_crash(nw.BOUNCE_LINES[n]), 44))
                for n in ("thin_until_crash", "thin_k1_4_hold", "thin_k1_hold")}
    return {"thin_until_crash_siege": turns, "totals": tot, "tower_sweep": sweep}


# ---------------------------------------------------------------------- stage: sustain
def stage_sustain() -> dict:
    """Entry crash rule: siege length and the HP that entered it, by tower damage (1 HP/tick: the prototype's tower)."""
    out = {}
    for tsd in (1, 2, 3, 4):
        c = final(tower_siege_damage=tsd)
        out[f"tower{tsd}"] = {n: sieges(play(c, lambda k, l, m, f=f: (f(k), HO), 60), A)[:3]
                              for n, f in (("thin_k1", lambda k: TN if k == 1 else HO), ("clear", lambda k: CL),
                                           ("thin", lambda k: TN))}
    return out


# ---------------------------------------------------------------------- stage: actions
ACTION_LINES = {
    "hold": lambda k, l: HO, "thin": lambda k, l: TN, "clear": lambda k, l: CL,
    "thin_k1": lambda k, l: TN if k == 1 else HO, "thin_k1_4": lambda k, l: TN if 1 <= k <= 4 else HO,
    "thin_k1_4_then_pressure": lambda k, l: HOP if l.asg > 0 else (TN if 1 <= k <= 4 else HO),
    "clear_then_clear_pressure": lambda k, l: CLP if l.asg > 0 else CL}


def stage_actions() -> dict:
    out = {}
    for name, cfg in (("prototype", proto()), ("final", final()),
                      ("final_enemy_tower_collect_1", final(collect_contest_enemy_tower=1.0))):
        out[name] = {n: line(play(cfg, lambda k, l, m, f=f: (f(k, l), HO), 80), cfg) for n, f in ACTION_LINES.items()}
    out["kill_fresh_wave_ticks_final"] = {n: nw.kill_ticks(final(), w) for n, w in (("hold", W_HOLD), ("thin", W_THIN),
                                                                                   ("clear", W_CLEAR))}
    return out


# ---------------------------------------------------------------------- stage: power
MULTS = (1.0, 1.5, 2.0, 3.0, 4.0)


def budget_table(cfg) -> List[dict]:
    """Clear's per-turn budget (HP) for representative (level, invested resources) states of the preset."""
    ctx = context(cfg)
    rows = []
    for L, S in ((1, 0), (3, 0), (6, 4), (9, 0), (9, 16), (13, 32), (18, 64), (18, 10 ** 4)):
        units = S * cfg.gold_scale
        ap = ctx.inv.attack_power(L, units) / ctx.inv.one / 10_000
        rows.append({"level": L, "invested": S, "attack_power": round(ap, 3),
                     "clear_budget": ctx.wave_budget(W_CLEAR, L, units), "thin_budget": ctx.wave_budget(W_THIN, L, units)})
    return rows


def first_push(cfg, mult: float, k0: int) -> dict:
    """From the Hold / Hold lane at k0, A Clears until its wave crashes, then Holds (B Holds throughout)."""
    lane, m4, k = nw.end_state(play(cfg, nw.const(HO), k0))
    st = {"c": False}

    def pol(kk, l, m):
        if l.asg > 0:
            st["c"] = True
        return (HO if st["c"] else CL, HO)
    rows = play(cfg, pol, 24, lane, m4, k, mult=(mult, 1.0))
    sg = sieges(rows, A)
    s = nw.summarize(rows, cfg)
    return {"fronts_before": [lane.ah, lane.bh], "turns_to_crash": None if not sg else sg[0]["k"] - k,
            "siege": sg[0] if sg else None, "A_minus_B_24": s["earned_A_minus_B"]}


def stage_power() -> dict:
    cfg = final()
    out = {"budget_table": budget_table(cfg), "from_neutral": {}, "from_s0": {}, "thin_scaled_counterfactual": {},
           "pressure_siege": {}}
    for m in MULTS:
        out["from_neutral"][str(m)] = {f"k{k0}": first_push(cfg, m, k0) for k0 in (9, 10)}
        out["from_s0"][str(m)] = {n: line(play(cfg, lambda k, l, mm, f=f: (f(k), HO), 40, mult=(m, 1.0)), cfg)
                                  for n, f in (("clear", lambda k: CL), ("thin3_clear", lambda k: TN if k <= 3 else CL))}
    both = final(wave_clear_scaled_slots=(W_CLEAR, W_THIN))
    for m in (1.0, 2.0, 4.0):
        out["thin_scaled_counterfactual"][str(m)] = {
            n: line(play(c, lambda k, l, mm: (TN, HO), 40, mult=(m, 1.0)), c)
            for n, c in (("thin_fixed", cfg), ("thin_scaled", both))}
    for reach in (0, 1):
        c = final(natural_tower_reach=reach)
        for m in (1.0, 2.0, 4.0):
            res = {}
            for n, after in (("hold", HO), ("hold_pressure", HOP), ("clear_pressure", CLP)):
                st = {"c": False}

                def pol(k, l, mm, after=after, st=st):
                    if l.asg > 0:
                        st["c"] = True
                    if not st["c"]:
                        return (TN if k == 1 else HO, HO)
                    return (after if l.asg > 0 else HO, HO)
                res[n] = line(play(c, pol, 44, mult=(m, 1.0)), c)
            out["pressure_siege"][f"reach{reach}_ap{m}"] = res
    return out


# ---------------------------------------------------------------------- stage: freeze
SETUPS = {"B_thin_k1": lambda k: TN if k == 1 else HO, "B_thin_k1_2": lambda k: TN if 1 <= k <= 2 else HO,
          "B_thin_k2": lambda k: TN if k == 2 else HO, "B_thin_k1_4": lambda k: TN if 1 <= k <= 4 else HO}


def stage_freeze() -> dict:
    out = {"reach": {}}
    for name, cfg in (("prototype", proto()), ("final", final())):
        out["reach"][name] = {}
        for sn, bp in SETUPS.items():
            for margin in (6, 8, 10):
                fz = nw.bounded_freezer(margin)
                s = nw.summarize(play(cfg, lambda k, l, m, fz=fz, bp=bp: (fz(l), bp(k)), 60), cfg)
                out["reach"][name][f"{sn}_m{margin}"] = {
                    "longest_hold": longest_hold(s["contact"]), "A_minus_B": s["earned_A_minus_B"],
                    "crashes": s["crashes"]}
    cfg = final()
    fz = nw.bounded_freezer(8)
    setup = play(cfg, lambda k, l, m: (fz(l), TN if k == 1 else HO), 36)
    lane0, m40, k0 = nw.end_state(setup)
    out["setup_trajectory"] = nw.trajectory(setup[24:], cfg)
    out["state_k36"] = [nw.side_str(lane0, A), nw.side_str(lane0, B)]
    out["from_k36"] = {}
    for name, pa, pb in (("freezer_vs_hold", lambda l: fz(l), lambda k: HO),
                         ("hold_vs_hold_uncontrolled", lambda l: HO, lambda k: HO),
                         ("freezer_vs_thin", lambda l: fz(l), lambda k: TN),
                         ("freezer_vs_clear", lambda l: fz(l), lambda k: CL),
                         ("freezer_vs_one_clear", lambda l: fz(l), lambda k: CL if k == k0 else HO)):
        rows = play(cfg, lambda k, l, m, pa=pa, pb=pb: (pa(l), pb(k)), 20, lane0, m40, k0)
        s = nw.summarize(rows, cfg)
        out["from_k36"][name] = {"held_turns": held_turns(s["contact"]), "A_minus_B": s["earned_A_minus_B"],
                                 "farm": s["farm"], "crashes": s["crashes"], "contact": s["contact"]}
        if name == "freezer_vs_clear":
            out["break_trajectory"] = nw.trajectory(rows[:6], cfg)
    # the breaker's power: B Clears from the freeze state with a stronger champion
    out["break_by_power"] = {}
    for m in (1.0, 2.0, 4.0):
        rows = play(cfg, lambda k, l, mm: (fz(l), CL if k == k0 else HO), 20, lane0, m40, k0, mult=(1.0, m))
        s = nw.summarize(rows, cfg)
        out["break_by_power"][str(m)] = {"held_turns": held_turns(s["contact"]), "crashes_B": s["crashes"][B][:3]}
    return out


# ---------------------------------------------------------------------- stage: bounce
def stage_bounce() -> dict:
    out = {}
    for name, cfg in (("prototype", proto()), ("final", final())):
        out[name] = {n: nw.crash_report(play(cfg, nw.until_crash(f), 50)) for n, f in nw.BOUNCE_LINES.items()}
    cfg = final()
    rows = play(cfg, nw.until_crash(lambda k: TN), 70)
    s = nw.summarize(rows, cfg)
    out["final_cycle_thin_until_crash"] = {"crashes": s["crashes"], "contact": s["contact"],
                                           "trajectory": nw.trajectory(rows[6:40], cfg)}
    return out


# ---------------------------------------------------------------------- stage: granularity (lane only)
def stage_granularity() -> dict:
    """Lane only (farm-mode reach stand-in), fixed champion damage: the integer law floor(h / 8) at W = 8 against a
    half-wave resolution (W = 16 sub-HP, every damage doubled, firepower floor(h / 8) sub-HP). Crash HP is reported
    in W = 8 units."""
    import lanegame.mechanics as M
    w8 = final(wave_clear_scaling="none")
    w16 = final(wave_clear_scaling="none", wave_hp=16, champion_wave_damage=(0, 4, 2), tower_siege_damage=6,
                siege_cap=192, farm_per_hp=0.0625)
    lines = {"hold": lambda k: W_HOLD, "thin": lambda k: W_THIN, "clear": lambda k: W_CLEAR,
             "thin_k1_4": lambda k: W_THIN if 1 <= k <= 4 else W_HOLD}
    orig = M.minion_damage
    out = {}
    try:
        for name, cfg, law in (("floor_h_over_8_W8", w8, None),
                               ("half_wave_W16", w16, lambda h, c: max(1, h // (c.wave_hp // 2)))):
            M.minion_damage = orig if law is None else law
            sc = 8 / cfg.wave_hp
            res = {}
            for ln, f in lines.items():
                st, reach, cr = initial_state(cfg), nw.farm_reach(cfg), [[], []]
                for k in range(60):
                    r = step_turn(st, k, f(k), W_HOLD, cfg, champion_reach=reach(st))
                    TURNS[0] += 1
                    for s in (A, B):
                        if r.crash_hp[s]:
                            cr[s].append([k, r.crash_hp[s] * sc])
                    st = r.state
                res[ln] = {"A": cr[A][:6], "B": cr[B][:4]}
            out[name] = res
    finally:
        M.minion_damage = orig
    return out


# ---------------------------------------------------------------------- stage: performance
def stage_performance(n: int = 1000) -> dict:
    import numpy as np
    from ...env import LaneEnv, UniformAgent
    out = {}
    for name, cfg in (("prototype", proto()), ("final", final())):
        rng = np.random.default_rng(0)
        st, t0 = initial_state(cfg), time.perf_counter()
        for k in range(n):
            st = step_turn(st, k, int(rng.integers(3)), int(rng.integers(3)), cfg,
                           champion_reach=(int(rng.integers(-2, 4)), int(rng.integers(-2, 4))),
                           champion_budget=None if cfg.wave_clear_scaling == "none" else (int(rng.integers(0, 12)),
                                                                                          int(rng.integers(0, 12)))).state
        lane_us = (time.perf_counter() - t0) / n * 1e6
        env, ag = LaneEnv(cfg, max_turns=n), UniformAgent()
        obs, t0, steps = env.reset(seed=1), time.perf_counter(), 0
        while True:
            r = env.step(ag.act(obs[0], rng), ag.act(obs[1], rng))
            steps += 1
            if r.done:
                break
            obs = r.obs
        env_us = (time.perf_counter() - t0) / steps * 1e6
        TURNS[0] += n + steps
        out[name] = {"lane_step_us": round(lane_us, 1), "env_step_us": round(env_us, 1)}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cfg = final()
    data = {"siege": stage_siege(), "sustain": stage_sustain(), "actions": stage_actions(), "power": stage_power(),
            "freeze": stage_freeze(), "bounce": stage_bounce(), "recall": nw.stage_recall(cfg),
            "reach": nw.stage_reach(cfg), "granularity": stage_granularity(), "scripted": nw.stage_scripted(cfg),
            "performance": stage_performance()}
    TURNS[0] += nw.TURNS[0]
    data["budget"] = {"turns": TURNS[0], "wall_s": round(time.time() - t0, 2)}
    dump_json({**meta(cfg), **data}, os.path.join(OUT, "natural_waves_final.json"))
    print("siege totals", data["siege"]["totals"])
    for n, v in data["siege"]["tower_sweep"].items():
        print(" ", n, {k: (r["clear_turns"], r["defender_in_attacker_half_after_turns"],
                           (r["bounce_crash"] or {}).get("after_turns")) for k, r in v.items()})
    for n, v in data["sustain"].items():
        print("sustain", n, {k: [(s["turns"], s["total_hp"]) for s in x] for k, x in v.items()})
    for n, v in data["actions"].items():
        if isinstance(v, dict) and isinstance(v.get("hold"), dict):
            print("actions", n, {k: x["A_minus_B"] for k, x in v.items()})
    print("power budget", data["power"]["budget_table"])
    for m, v in data["power"]["from_neutral"].items():
        print("from neutral AP", m, {k: (x["turns_to_crash"], x["siege"] and x["siege"]["first_hp"]) for k, x in v.items()})
    for n, v in data["freeze"]["from_k36"].items():
        print("freeze", n, v["held_turns"], v["A_minus_B"], v["crashes"])
    print("freeze break by power", data["freeze"]["break_by_power"])
    for n, v in data["bounce"].items():
        if n in ("prototype", "final"):
            print("bounce", n, {k: r and (r["crash_hp"], r["clear_turns"], r["defender_in_attacker_half_after_turns"])
                                for k, r in v.items()})
    print("granularity", data["granularity"])
    print("performance", data["performance"])
    print("budget", data["budget"])


if __name__ == "__main__":
    main()
