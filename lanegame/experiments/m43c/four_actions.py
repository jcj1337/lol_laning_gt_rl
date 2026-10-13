"""M4.3c four-action wave management (docs/m4/m4_3c_four_action_wave_redesign.md): deterministic validation of the
EXPERIMENTAL preset `GameConfig.m4_four_actions()` and the formulation comparison that selected its Thin and Slow Push
rules. Nothing here changes the game; `m4_lasthit()` is the unchanged baseline.

The four wave actions are per-turn damage budgets on enemy minions out of one wave-clear capacity C = round(4 AP):

    Last Hit   1                                    (wave_hold_cap 1)
    Thin       min(round(C / 2), 1 + need_0)         (wave_thin_rule "strength")
    Slow Push  min(round(C / 2), 1 + need_1)         (wave_slowpush)
    Clear      C

need_m = max(0, ceil(S_enemy - S_own + m)): S is the window strength of mechanics.window_strength (the average minion
HP fighting at a side's front over the next reinforcement cycle, reinforcements weighted by when they arrive).

Deterministic scripted play through the M4 resolver with no chance event (u = None: no trades, no HP lost, no
recalls). Champion power is FIXED over a window: each side's capacity is that of a (level, invested resources) tier,
computed by the engine once, and every budget is taken from it by the engine's own rule (`M4Context.state_budget`) or,
in the formulation comparison only, by a candidate rule of this module. Levels and XP still evolve for the accounting.
Starts are the M4.2e-B harness states, generated under `m4_lasthit()` exactly as in the earlier M4.3c screens:

    neutral    turn 9, both 8-HP fronts and both champions at the lane cell
    freeze     turn 51, B's 5-HP front held at A's -1 (a fresh B wave behind it, 6 HP of B's crashed under A's tower),
               A's 8-HP front in A's tower cell, A at -2, B at its +1
    imbalance  the neutral lane with the fronts set to other sizes (A 8 or 4 HP, B 8-24 HP)

Stages:
    formulations  Part B: candidate Thin / Slow Push rules on a fixed scenario set
    baseline      the neutral, freeze and imbalance scenarios under the CURRENT m4_lasthit() budgets (comparison)
    neutral       Test 1: each action of A against B's Last Hit from the neutral lane; the 4 x 4 action matrix
    imbalance     Test 2: A against an oversized B wave; damage, crashes, overshoot, power monotonicity
    freeze        Test 3: A's action every turn from the calibrated freeze against five responses of B; enlarged B waves
    slowpush      Test 4: Slow Push against Last Hit and Clear from balanced and unfavourable starts
    power         Test 5: budget invariants on sampled states for every capacity; outcome monotonicity over C = 4..23
    integration   Test 6: env soak (conservation, mirror symmetry, masks, recorded budgets) and a PPO smoke run
    traces        lane strips of representative runs

Coordinates in records and traces are GLOBAL (A's tower at -2, B's at +2).

    python -m lanegame.experiments.m43c four_actions        # writes results/m43c/four_actions.{json(.gz),md}
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ...config import (B_HOLD, B_SAFE, CHAMP_ABSENT, F, W_CLEAR, W_HOLD, W_SLOWPUSH, W_THIN, GameConfig, champ_act,
                       champ_behavior, champ_objective, champ_wave)
from .common import meta, save_json, write_md
from .wave_control import lane_numbers, longest, strip

LH, CL, TN, SL = W_HOLD, W_CLEAR, W_THIN, W_SLOWPUSH
ACTIONS = (LH, TN, SL, CL)
SHORT = {LH: "LH", TN: "TN", SL: "SL", CL: "CL"}
NAMES = {LH: "Last Hit", TN: "Thin", SL: "Slow Push", CL: "Clear"}
LAMBDA = 1 / 8
# power tiers: name -> ((level, invested resources) of A, of B); capacities are computed by the engine
TIERS = {"L1": ((1, 0), (1, 0)), "L6": ((6, 0), (6, 0)), "L10": ((10, 0), (10, 0)), "L14": ((14, 0), (14, 0)),
         "L18": ((18, 0), (18, 0)), "L18 +64": ((18, 64), (18, 64)),
         "L10 A+": ((10, 50), (10, 0)), "L13 A+": ((13, 50), (13, 0)), "L18 A++": ((18, 10_000), (18, 0))}
EQUAL_TIERS = ("L1", "L6", "L10", "L14", "L18", "L18 +64")
GAP_TIERS = ("L10 A+", "L13 A+", "L18 A++")
NEUTRAL_TURNS, IMBALANCE_TURNS, FREEZE_TURNS, SLOWPUSH_TURNS = 60, 12, 40, 40
IMBALANCES = ((8, 12), (8, 16), (8, 20), (8, 24), (4, 20))
IMBALANCE_CAPS = (4, 6, 11, 22)                 # A's capacity (controlled cap 2, 3, 6, 11); B Last Hits with C = 6
IMBALANCE_B_CAP = 6
RESPONSES = ("walk_up", "step_back", "retreat", "thin", "clear")
FREEZE_ENLARGED = (13, 21)                      # B's held front at the freeze start set to 13 / 21 HP (from 5)
SLOWPUSH_STARTS = ((8, 8), (8, 10), (8, 12), (8, 14))
SLOWPUSH_TIERS = ("L1", "L10", "L18", "L10 A+", "L18 A++")
CAP_RANGE = tuple(range(4, 24))                 # every capacity a champion can have (C <= 23)
SAMPLE_SEEDS, SAMPLE_TURNS = (0, 1, 2, 3), 300
SOAK_SEEDS, SOAK_TURNS = (11, 12, 13), 400
PPO_STEPS, PPO_ENVS, PPO_SEED = 32_768, 8, 0
TRACE_TURNS = 16


def cfg_four(**kw) -> GameConfig:
    return GameConfig.m4_four_actions(**kw)


def capacities(tier: str) -> tuple:
    from ...champions_m4 import context
    cfg = cfg_four()
    ctx = context(cfg)
    return tuple(ctx.capacity(L, S * cfg.gold_scale) for L, S in TIERS[tier])


def code(slot: int, behavior: int = B_HOLD) -> int:
    return champ_act(slot, behavior)


# ---------------------------------------------------------------------------------------------- candidate rules
def _ctrl(C: int) -> int:
    return (C + 1) // 2                          # round(C / 2), half up


def _window3(packets, cfg) -> int:
    """HP of the leading packet and every packet that joins it this turn (within ticks_per_turn + 1 cells)."""
    if not packets:
        return 0
    fx = packets[0][0]
    return packets[0][1] + sum(p[1] for p in packets[1:] if p[0] >= fx - cfg.ticks_per_turn - 1)


def _class_top(phi: int, cfg) -> int:
    """Largest HP whose half-wave firepower max(1, floor(2h / W)) is <= phi (0 if phi < 1)."""
    return (cfg.wave_hp * (phi + 1)) // 2 - 1 if phi >= 1 else 0


def _rule_strength(thin_margin: int = 0, sp_margin: int = 1, floor: bool = True, cap_share: bool = True,
                   flat_sp: bool = False) -> Callable:
    def make(cfg):
        from ...mechanics import start_packets, strength_need

        def fn(slot, C, lane, k, s):
            if slot == LH:
                return 1
            if slot == CL:
                return C
            cap = _ctrl(C) if cap_share else C
            if slot == SL and flat_sp:
                return min(cap, 2)
            need = strength_need(start_packets(lane, cfg), s, k, thin_margin if slot == TN else sp_margin, cfg)
            return min(cap, (1 if floor else 0) + need)
        return fn
    return make


def _rule_share(cfg):
    return lambda slot, C, lane, k, s: {LH: 1, CL: C, TN: _ctrl(C), SL: _ctrl(C)}[slot]


def _rule_lasthit(cfg):
    """The CURRENT three-action model, m4_lasthit(): Hold round(C/8), Thin round(C/2), Clear C."""
    return lambda slot, C, lane, k, s: {LH: (2 * C + 8) // 16, CL: C, TN: _ctrl(C), SL: _ctrl(C)}[slot]


def _rule_adaptive(cfg):
    from ...mechanics import start_packets, thin_need

    def fn(slot, C, lane, k, s):
        if slot in (TN, SL):
            return min(_ctrl(C), thin_need(start_packets(lane, cfg), s, cfg))
        return 1 if slot == LH else C
    return fn


def _rule_class(cfg):
    from ...mechanics import start_packets

    def fn(slot, C, lane, k, s):
        if slot in (LH, CL):
            return 1 if slot == LH else C
        P = start_packets(lane, cfg)
        he, ho = _window3(P[1 - s], cfg), _window3(P[s], cfg)
        f_own = max(1, (2 * ho) // cfg.wave_hp) if ho > 0 else 0
        top = _class_top(f_own - (0 if slot == TN else 1), cfg)
        return min(_ctrl(C), 1 + max(0, he - top))
    return fn


# name -> (description, factory(cfg) -> fn(slot, C, lane, k, s), has a distinct Slow Push)
RULES = {
    "lasthit": ("CURRENT m4_lasthit(): Hold = round(C/8), Thin = round(C/2) (three actions)", _rule_lasthit, False),
    "share": ("M4.2e-C shares with Hold capped at 1: Thin = Slow Push = round(C/2)", _rule_share, False),
    "adaptive": ("M4.3c screen: Thin = min(round(C/2), thin_need), start-of-turn blobs, no floor", _rule_adaptive,
                 False),
    "class1": ("one-turn window, firepower-class parity (Thin) / one-class lead (Slow Push), + Last Hit",
               _rule_class, True),
    "strength": ("SELECTED: window strength, margins 0 / 1, + Last Hit, cap round(C/2)", _rule_strength(), True),
    "strength_tol3": ("window strength, Thin tolerates 3 HP, Slow Push margin 1", _rule_strength(thin_margin=-3),
                      True),
    "strength_sp4": ("window strength, Slow Push margin 4 (one firepower class)", _rule_strength(sp_margin=4), True),
    "strength_nofloor": ("window strength without the Last Hit floor", _rule_strength(floor=False), True),
    "strength_capC": ("window strength with cap C instead of round(C/2)", _rule_strength(cap_share=False), True),
    "flat_sp": ("window-strength Thin; Slow Push = min(round(C/2), 2) every turn (fixed +1 edge)",
                _rule_strength(flat_sp=True), True),
}


# ---------------------------------------------------------------------------------------------- driver
def _legal(cfg, lane, m4, s, a) -> int:
    from ...champions import champion_legal_actions
    legal = champion_legal_actions(lane, m4, s, cfg)
    if legal == (CHAMP_ABSENT,):
        return CHAMP_ABSENT
    for c in (a, champ_act(champ_wave(a), champ_behavior(a)), champ_act(champ_wave(a), B_HOLD), code(LH)):
        if c in legal:
            return c
    raise AssertionError((a, legal))


def play(cfg: GameConfig, start, caps, policy: Callable, turns: int, rule: Optional[str] = None,
         trace: int = 0) -> dict:
    """Scripted play from start = (lane, m4, k) with fixed capacities caps = (C_A, C_B); policy(i, lane, m4) ->
    (code_A, code_B). Budgets: the engine's `M4Context.state_budget` (rule None) or the candidate RULES[rule]."""
    from ...champions_m4 import context, resolve_turn_m4
    from ...mechanics import start_packets, step_turn, strength_need, window_strength
    ctx = context(cfg)
    fn = RULES[rule][1](cfg) if rule else (lambda slot, C, ln, kk, s: ctx.state_budget(slot, C, ln, kk, s))
    lane, m4, k = start
    used = {}

    def lane_step(st, kk, x, y, reach, budget=None):
        b = (fn(x % 6, caps[0], st, kk, 0), fn(y % 6, caps[1], st, kk, 1))     # lane code = slot + 6 objective
        used["b"] = b
        return step_turn(st, kk, x, y, cfg, champion_reach=reach, champion_budget=b)

    rows, strips = [], [] if trace else None
    if trace:
        strips.append(f"{'turn':<5}" + "".join(f"{x:+d}".ljust(11) for x in range(-4, 5)) + "  A/B damage")
        strips.append(f"{0:<5}{strip(lane, m4)}")
    for i in range(turns):
        a, b = policy(i, lane, m4)
        a, b = _legal(cfg, lane, m4, 0, a), _legal(cfg, lane, m4, 1, b)
        P = start_packets(lane, cfg)
        S = [window_strength(P[s], k, cfg) for s in (0, 1)]
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, lane_step=lane_step, u=None)
        ev = t.lane.events
        rec = {"w": list(t.waves), "bud": list(used["b"]), "S": [float(x) for x in S],
               "need": [[strength_need(P, s, k, m, cfg) for m in (0, 1)] for s in (0, 1)],
               "dmg": [ev["champion_hp"][1], ev["champion_hp"][0]], "crash": list(t.lane.crash_hp),
               "gold": [x / cfg.gold_scale for x in t.earned_units], "xp": list(t.xp_gain),
               "end": [t.end[0], -t.end[1]], **lane_numbers(t.lane.state, cfg)}
        rows.append(rec)
        lane, m4, k = t.lane.state, t.champ, k + 1
        if trace and i < trace:
            tag = "/".join(f"{SHORT[rec['w'][s]]}{rec['dmg'][s]}" for s in (0, 1))
            strips.append(f"{i + 1:<5}{strip(lane, m4):<99}  {tag}")
    return {"rows": rows, "strips": strips}


def stats(rows: List[dict], start_held: int = 0) -> dict:
    """Window statistics of one run (A = side 0, B = side 1; differences are B - A)."""
    s = lambda key, side: float(sum(r[key][side] for r in rows))
    held = [r["held"] for r in rows]
    contacts = [r["contact"] for r in rows if r["contact"] is not None]
    first = [next((i + 1 for i, r in enumerate(rows) if r["crash"][x] > 0), None) for x in (0, 1)]
    out = {"turns": len(rows), "crashes": [sum(1 for r in rows if r["crash"][x] > 0) for x in (0, 1)],
           "crash_hp": [s("crash", 0), s("crash", 1)], "first_crash": first,
           "dmg": [s("dmg", 0), s("dmg", 1)], "gold": [s("gold", 0), s("gold", 1)], "xp": [s("xp", 0), s("xp", 1)],
           "held": int(sum(held)), "longest_hold": longest(held),
           "first_break": next((i + 1 for i, h in enumerate(held) if not h), None) if start_held else None,
           "contact_turns": len(contacts), "contact_mean": float(np.mean(contacts)) if contacts else None,
           "final_contact": rows[-1]["contact"],
           "mean_blob": [float(np.mean([r["blob"][x] for r in rows])) for x in (0, 1)]}
    for x, side in ((0, "a"), (1, "b")):
        for slot in ACTIONS:
            tr = [r for r in rows if r["w"][x] == slot]
            out[f"{side}_{SHORT[slot]}_turns"] = len(tr)
            out[f"{side}_{SHORT[slot]}_dmg"] = float(sum(r["dmg"][x] for r in tr))
            out[f"{side}_{SHORT[slot]}_budget"] = float(sum(r["bud"][x] for r in tr))
    out["gold_diff_b"] = out["gold"][1] - out["gold"][0]
    out["xp_diff_b"] = out["xp"][1] - out["xp"][0]
    out["value_b"] = out["gold_diff_b"] + LAMBDA * out["xp_diff_b"]
    return out


def series(rows: List[dict]) -> dict:
    return {key: [r[key] for r in rows] for key in ("blob", "front", "S", "need", "bud", "dmg", "crash", "contact",
                                                     "held", "w")}


def const(a: int, b: int) -> Callable:
    return lambda i, lane, m4: (a, b)


# codes no present champion can ever play: a Recall other than code 12 (Last Hit + Recall), or an objective with a
# behavior other than Hold and Trade
NEVER_LEGAL = frozenset(a for a in range(61) if a != CHAMP_ABSENT and (
    (champ_behavior(a) == 4 and a != champ_act(LH, 4)) or (champ_objective(a) and champ_behavior(a) not in (0, 2))))

RESPONSE_CODE = {"walk_up": code(LH), "step_back": code(LH, B_SAFE), "retreat": code(LH, B_SAFE),
                 "thin": code(TN), "clear": code(CL), "slowpush": code(SL)}


def response_cfg(response: str) -> GameConfig:
    """`retreat` is Safe with safe_line -2 (stay in the own tower cell), a configuration override of its runs only."""
    return cfg_four(**({"safe_line": -2} if response == "retreat" else {}))


def with_fronts(start, ha: Optional[int] = None, hb: Optional[int] = None):
    lane, m4, k = start
    kw = {}
    if ha is not None:
        kw["ah"] = ha
    if hb is not None:
        kw["bh"] = hb
    return lane._replace(**kw), m4, k


# ---------------------------------------------------------------------------------------------- Part B
def stage_formulations(st) -> dict:
    """Each candidate rule on one fixed scenario set (A's capacity varies, B Last Hits unless stated)."""
    cfg = cfg_four()
    out = {}
    caps_list = IMBALANCE_CAPS
    freeze_tiers = (("L1", capacities("L1")), ("L10", capacities("L10")), ("L10 A+", capacities("L10 A+")),
                    ("L18 A++", capacities("L18 A++")))
    for name, (desc, _, has_sp) in RULES.items():
        r = {"description": desc, "has_slowpush": has_sp}
        neu = {}
        for C in caps_list:
            for slot in ((TN, SL) if has_sp else (TN,)):
                rows = play(cfg, st["neutral"], (C, IMBALANCE_B_CAP), const(code(slot), code(LH)), NEUTRAL_TURNS,
                            rule=name)["rows"]
                neu[f"{SHORT[slot]}|{C}"] = stats(rows)
        r["neutral"] = neu
        imb = {}
        for ha, hb in IMBALANCES:
            for C in caps_list:
                rows = play(cfg, with_fronts(st["neutral"], ha, hb), (C, IMBALANCE_B_CAP), const(code(TN), code(LH)),
                            IMBALANCE_TURNS, rule=name)["rows"]
                imb[f"{ha}v{hb}|{C}"] = stats(rows)
        r["imbalance"] = imb
        frz = {}
        for tier, caps in freeze_tiers:
            for resp in ("walk_up", "step_back", "clear"):
                c2 = response_cfg(resp)
                rows = play(c2, st["freeze"], caps, const(code(TN), RESPONSE_CODE[resp]), FREEZE_TURNS,
                            rule=name)["rows"]
                frz[f"{tier}|{resp}"] = stats(rows, start_held=1)
        r["freeze"] = frz
        if has_sp:
            rows = play(cfg, st["neutral"], (6, 6), const(code(SL), code(TN)), NEUTRAL_TURNS, rule=name)["rows"]
            r["sp_vs_thin"] = stats(rows)
        out[name] = r
    # Last Hit reference (identical under every rule)
    out["_lasthit"] = {f"{tier}|{resp}": stats(play(response_cfg(resp), st["freeze"], caps,
                                                    const(code(LH), RESPONSE_CODE[resp]), FREEZE_TURNS)["rows"], 1)
                       for tier, caps in freeze_tiers for resp in ("walk_up", "step_back", "clear")}
    return out


def formulation_summary(r: dict) -> dict:
    """The comparison criteria of one candidate (report table)."""
    neu = r["neutral"]
    thin_push = sum(neu[f"TN|{C}"]["crashes"][0] + neu[f"TN|{C}"]["crashes"][1] for C in IMBALANCE_CAPS)
    thin_drift = max(abs(neu[f"TN|{C}"]["contact_mean"] or 0.0) for C in IMBALANCE_CAPS)
    imb = r["imbalance"]
    b_crash = {C: sum(imb[f"{ha}v{hb}|{C}"]["crashes"][1] for ha, hb in IMBALANCES) for C in IMBALANCE_CAPS}
    a_crash = {C: sum(imb[f"{ha}v{hb}|{C}"]["crashes"][0] for ha, hb in IMBALANCES) for C in IMBALANCE_CAPS}
    nonmono = 0
    for ha, hb in IMBALANCES:
        seq = [imb[f"{ha}v{hb}|{C}"]["crashes"][1] for C in IMBALANCE_CAPS]
        nonmono += sum(1 for x, y in zip(seq, seq[1:]) if y > x)
    frz = r["freeze"]
    out = {"thin_neutral_crashes": thin_push, "thin_neutral_drift": thin_drift, "imb_b_crashes": b_crash,
           "imb_a_crashes": a_crash, "imb_nonmonotone": nonmono,
           "freeze_walk": {t: frz[f"{t}|walk_up"]["longest_hold"] for t in ("L1", "L10", "L10 A+", "L18 A++")},
           "freeze_step": {t: frz[f"{t}|step_back"]["longest_hold"] for t in ("L1", "L10", "L10 A+", "L18 A++")}}
    if r["has_slowpush"]:
        out["sp_first"] = {C: neu[f"SL|{C}"]["first_crash"][0] for C in IMBALANCE_CAPS}
        out["sp_crashes"] = {C: neu[f"SL|{C}"]["crashes"][0] for C in IMBALANCE_CAPS}
        out["sp_dmg_per_turn"] = {C: neu[f"SL|{C}"]["dmg"][0] / NEUTRAL_TURNS for C in IMBALANCE_CAPS}
        out["sp_vs_thin_crashes"] = r["sp_vs_thin"]["crashes"]
    return out


def stage_baseline(st) -> dict:
    """The neutral, freeze and imbalance scenarios under the CURRENT three-action budgets of m4_lasthit() (rule
    "lasthit": Hold round(C/8), Thin round(C/2)), for the comparison with the four-action model."""
    out = {}
    cfg = cfg_four()
    for tier in TIERS:
        caps = capacities(tier)
        for slot in (LH, TN, CL):
            rows = play(cfg, st["neutral"], caps, const(code(slot), code(LH)), NEUTRAL_TURNS, rule="lasthit")["rows"]
            out[f"neutral|{tier}|{SHORT[slot]}"] = stats(rows)
        for slot in (LH, TN):
            for resp in ("walk_up", "step_back", "clear"):
                rows = play(response_cfg(resp), st["freeze"], caps, const(code(slot), RESPONSE_CODE[resp]),
                            FREEZE_TURNS, rule="lasthit")["rows"]
                out[f"freeze|{tier}|{SHORT[slot]}|{resp}"] = stats(rows, start_held=1)
    for ha, hb in IMBALANCES:
        for C in IMBALANCE_CAPS:
            rows = play(cfg, with_fronts(st["neutral"], ha, hb), (C, IMBALANCE_B_CAP), const(code(TN), code(LH)),
                        IMBALANCE_TURNS, rule="lasthit")["rows"]
            out[f"imbalance|{ha}v{hb}|{C}"] = {"stats": stats(rows), "dmg_first4": [r["dmg"][0] for r in rows[:4]],
                                               "over_need": sum(max(0, r["dmg"][0] - 1 - r["need"][0][0])
                                                                for r in rows)}
    return out


# ---------------------------------------------------------------------------------------------- Test 1
def stage_neutral(st) -> dict:
    out = {"vs_lasthit": {}, "matrix": {}}
    cfg = cfg_four()
    for tier in TIERS:
        caps = capacities(tier)
        for slot in ACTIONS:
            r = play(cfg, st["neutral"], caps, const(code(slot), code(LH)), NEUTRAL_TURNS)
            out["vs_lasthit"][f"{tier}|{SHORT[slot]}"] = {"caps": caps, "stats": stats(r["rows"]),
                                                         "series": series(r["rows"])}
    for tier in ("L1", "L10", "L18", "L10 A+", "L18 A++"):
        caps = capacities(tier)
        for wa in ACTIONS:
            for wb in ACTIONS:
                rows = play(cfg, st["neutral"], caps, const(code(wa), code(wb)), NEUTRAL_TURNS)["rows"]
                out["matrix"][f"{tier}|{SHORT[wa]}|{SHORT[wb]}"] = stats(rows)
    return out


# ---------------------------------------------------------------------------------------------- Test 2
def stage_imbalance(st) -> dict:
    out = {}
    cfg = cfg_four()
    for ha, hb in IMBALANCES:
        start = with_fronts(st["neutral"], ha, hb)
        for C in IMBALANCE_CAPS:
            for slot in ACTIONS:
                rows = play(cfg, start, (C, IMBALANCE_B_CAP), const(code(slot), code(LH)), IMBALANCE_TURNS)["rows"]
                out[f"{ha}v{hb}|{C}|{SHORT[slot]}"] = {
                    "stats": stats(rows), "dmg_first4": [r["dmg"][0] for r in rows[:4]],
                    "bud_first4": [r["bud"][0] for r in rows[:4]],
                    "need_first4": [r["need"][0][0] for r in rows[:4]],
                    "b_blob_first4": [r["blob"][1] for r in rows[:4]],
                    "over_need": sum(max(0, r["dmg"][0] - 1 - r["need"][0][0]) for r in rows),
                    "contact_at": {str(n): rows[n - 1]["contact"] for n in (4, 8, 12)},
                    "series": series(rows)}
    return out


# ---------------------------------------------------------------------------------------------- Test 3
def stage_freeze(st) -> dict:
    out = {"standard": {}, "enlarged": {}}
    for tier in TIERS:
        caps = capacities(tier)
        for slot in ACTIONS:
            for resp in RESPONSES:
                cfg = response_cfg(resp)
                rows = play(cfg, st["freeze"], caps, const(code(slot), RESPONSE_CODE[resp]), FREEZE_TURNS)["rows"]
                out["standard"][f"{tier}|{SHORT[slot]}|{resp}"] = {"stats": stats(rows, start_held=1),
                                                                    "series": series(rows)}
    for hb in FREEZE_ENLARGED:
        start = with_fronts(st["freeze"], hb=hb)
        for tier in TIERS:
            caps = capacities(tier)
            for slot in (LH, TN):
                rows = play(cfg_four(), start, caps, const(code(slot), RESPONSE_CODE["walk_up"]), FREEZE_TURNS)["rows"]
                out["enlarged"][f"{hb}|{tier}|{SHORT[slot]}"] = {
                    "stats": stats(rows, start_held=1), "dmg_first4": [r["dmg"][0] for r in rows[:4]],
                    "need_first4": [r["need"][0][0] for r in rows[:4]]}
    return out


# ---------------------------------------------------------------------------------------------- Test 4
def stage_slowpush(st) -> dict:
    out = {}
    cfg = cfg_four()
    for ha, hb in SLOWPUSH_STARTS:
        start = with_fronts(st["neutral"], ha, hb)
        for tier in SLOWPUSH_TIERS:
            caps = capacities(tier)
            for wa in (LH, SL, CL):
                for wb in (LH, TN):
                    rows = play(cfg, start, caps, const(code(wa), code(wb)), SLOWPUSH_TURNS)["rows"]
                    out[f"{ha}v{hb}|{tier}|{SHORT[wa]}|{SHORT[wb]}"] = {
                        "stats": stats(rows), "blob_diff": [r["blob"][0] - r["blob"][1] for r in rows],
                        "crash_sizes": [r["crash"][0] for r in rows if r["crash"][0] > 0],
                        "contact": [r["contact"] for r in rows]}
    return out


# ---------------------------------------------------------------------------------------------- Test 5
def sample_states(cfg: GameConfig) -> list:
    """(lane, k) of random legal play with chance events (uniform agents), SAMPLE_SEEDS x SAMPLE_TURNS."""
    from ...env import LaneEnv, UniformAgent
    out = []
    for seed in SAMPLE_SEEDS:
        env, ag, rng = LaneEnv(cfg, max_turns=SAMPLE_TURNS), UniformAgent(), np.random.default_rng(seed)
        obs = env.reset(seed=seed)
        while True:
            out.append((env.state, env.k))
            r = env.step(ag.act(obs[0], rng), ag.act(obs[1], rng))
            if r.done:
                break
            obs = r.obs
    return out


def stage_power(st) -> dict:
    """(a) budget invariants for every capacity on sampled states; (b) outcome monotonicity over C = 4..23."""
    from ...champions_m4 import context
    from ...mechanics import start_packets, strength_need
    cfg = cfg_four()
    ctx = context(cfg)
    states = sample_states(cfg)
    viol = {"lasthit_not_1": 0, "order": 0, "not_monotone_in_C": 0, "above_need": 0, "below_floor": 0}
    zero_need = {"TN": 0, "SL": 0}
    binds = {C: {"TN": 0, "SL": 0} for C in CAP_RANGE}
    # identical budgets (a necessary condition for two actions to be interchangeable in a state)
    same = {C: {"LH=TN": 0, "TN=SL": 0, "LH=SL": 0, "SL=CL": 0, "all_distinct": 0} for C in CAP_RANGE}
    mean_bud = {C: {SHORT[a]: 0.0 for a in ACTIONS} for C in CAP_RANGE}
    n = 0
    for lane, k in states:
        P = start_packets(lane, cfg)
        for s in (0, 1):
            n += 1
            needs = {TN: strength_need(P, s, k, 0, cfg), SL: strength_need(P, s, k, 1, cfg)}
            zero_need["TN"] += needs[TN] == 0
            zero_need["SL"] += needs[SL] == 0
            prev = None
            for C in CAP_RANGE:
                b = {a: ctx.state_budget(a, C, lane, k, s) for a in ACTIONS}
                viol["lasthit_not_1"] += b[LH] != 1
                viol["order"] += not (b[LH] <= b[TN] <= b[SL] <= b[CL] == C)
                viol["above_need"] += b[TN] > 1 + needs[TN] or b[SL] > 1 + needs[SL]
                viol["below_floor"] += min(b[TN], b[SL]) < min(1, C)
                if prev is not None:
                    viol["not_monotone_in_C"] += any(b[a] < prev[a] for a in ACTIONS)
                prev = b
                for a in ACTIONS:
                    mean_bud[C][SHORT[a]] += b[a]
                binds[C]["TN"] += b[TN] < 1 + needs[TN]
                binds[C]["SL"] += b[SL] < 1 + needs[SL]
                same[C]["LH=TN"] += b[LH] == b[TN]
                same[C]["TN=SL"] += b[TN] == b[SL]
                same[C]["LH=SL"] += b[LH] == b[SL]
                same[C]["SL=CL"] += b[SL] == b[CL]
                same[C]["all_distinct"] += len({b[a] for a in ACTIONS}) == 4
    for C in CAP_RANGE:
        for key in mean_bud[C]:
            mean_bud[C][key] /= n
        for key in binds[C]:
            binds[C][key] /= n
        for key in same[C]:
            same[C][key] /= n
    out = {"states": n, "violations": viol, "zero_need": {kk: v / n for kk, v in zero_need.items()},
           "cap_binds": binds, "mean_budget": mean_bud, "same_budget": same}
    # (b) outcomes: A's capacity sweeps 4..23, B Last Hits at 6
    sweep = {}
    for ha, hb in IMBALANCES:
        start = with_fronts(st["neutral"], ha, hb)
        for slot in (LH, TN, SL):
            res = []
            for C in CAP_RANGE:
                rows = play(cfg, start, (C, IMBALANCE_B_CAP), const(code(slot), code(LH)), IMBALANCE_TURNS)["rows"]
                s_ = stats(rows)
                res.append({"C": C, "crashes": s_["crashes"], "contact": s_["final_contact"], "dmg": s_["dmg"][0]})
            sweep[f"{ha}v{hb}|{SHORT[slot]}"] = res
    for hb in FREEZE_ENLARGED:
        start = with_fronts(st["freeze"], hb=hb)
        for slot in (LH, TN):
            res = []
            for C in CAP_RANGE:
                rows = play(cfg, start, (C, IMBALANCE_B_CAP), const(code(slot), code(LH)), FREEZE_TURNS)["rows"]
                s_ = stats(rows, start_held=1)
                res.append({"C": C, "crashes": s_["crashes"], "longest_hold": s_["longest_hold"],
                            "dmg": s_["dmg"][0]})
            sweep[f"freeze{hb}|{SHORT[slot]}"] = res
    for slot in (TN, SL):
        res = []
        for C in CAP_RANGE:
            rows = play(cfg, st["neutral"], (C, IMBALANCE_B_CAP), const(code(slot), code(LH)), NEUTRAL_TURNS)["rows"]
            s_ = stats(rows)
            res.append({"C": C, "crashes": s_["crashes"], "first": s_["first_crash"][0], "dmg": s_["dmg"][0]})
        sweep[f"neutral|{SHORT[slot]}"] = res
    out["sweep"] = sweep
    # monotonicity summary: Thin must never crash A's own wave, and B's crashes must not rise with A's capacity
    thin_keys = [kk for kk in sweep if kk.endswith("|TN") and not kk.startswith("neutral")]
    out["thin_own_crashes"] = {kk: sum(x["crashes"][0] for x in sweep[kk]) for kk in thin_keys}
    out["thin_b_crash_increases"] = {kk: sum(1 for x, y in zip(sweep[kk], sweep[kk][1:])
                                             if y["crashes"][1] > x["crashes"][1]) for kk in thin_keys}
    lh_keys = [kk for kk in sweep if kk.endswith("|LH")]
    out["lasthit_identical_over_C"] = {kk: len({(tuple(x["crashes"]), x["dmg"]) for x in sweep[kk]}) == 1
                                       for kk in lh_keys}
    return out


# ---------------------------------------------------------------------------------------------- Test 6
def stage_integration(ppo: bool = True) -> dict:
    """Env soak on the 61-code preset (conservation, mirror symmetry, masks, recorded budgets), and a PPO smoke run."""
    from ...champions import N_CHANCE, mirror_chance, swap_seats
    from ...champions_m4 import context, resolve_turn_m4
    from ...env import LaneEnv, UniformAgent
    from ...mechanics import total_hp
    cfg = cfg_four()
    ctx = context(cfg)
    out = {"n_actions": LaneEnv(cfg).n_actions, "obs_size": LaneEnv(cfg).obs_size}
    chk = {"turns": 0, "hp_conservation": 0, "mirror": 0, "mask": 0, "budget_record": 0, "slowpush_chosen": 0,
           "slowpush_legal_mismatch": 0}
    for seed in SOAK_SEEDS:
        env, ag = LaneEnv(cfg, max_turns=SOAK_TURNS), UniformAgent()
        rng, crng = np.random.default_rng(seed), np.random.default_rng(seed + 100)
        obs = env.reset(seed=seed)
        while True:
            lane, m4, k = env.state, env.champ, env.k
            for o in obs:
                legal = set(o.legal)
                chk["mask"] += (not legal <= set(range(61)) or bool(np.flatnonzero(o.mask).tolist() != sorted(legal))
                                or bool(legal & NEVER_LEGAL))
                if legal != {CHAMP_ABSENT}:
                    for b in range(5):
                        for ob in range(3):
                            h, sp = champ_act(LH, b, ob), champ_act(SL, b, ob)
                            if h != champ_act(LH, 4):              # the Recall code has no Slow Push twin
                                chk["slowpush_legal_mismatch"] += (h in legal) != (sp in legal)
            a, b = ag.act(obs[0], rng), ag.act(obs[1], rng)
            chk["slowpush_chosen"] += (a >= 46 and a != CHAMP_ABSENT) + (b >= 46 and b != CHAMP_ABSENT)
            u = tuple(float(x) for x in crng.random(N_CHANCE))
            t = resolve_turn_m4(lane, m4, k, a, b, cfg, u=u)
            m = resolve_turn_m4(lane.mirror(), m4.mirror(), k, b, a, swap_seats(cfg), u=mirror_chance(u))
            chk["mirror"] += not (m.champ == t.champ.mirror() and m.lane.state == t.lane.state.mirror()
                                  and m.reward_units == -t.reward_units and m.wave_budget == t.wave_budget[::-1])
            before, after, ev = total_hp(lane, cfg), total_hp(t.lane.state, cfg), t.lane.events
            chk["hp_conservation"] += any(before[s] + ev["spawned_hp"][s] != after[s] + ev["killed_hp"][s]
                                          + ev["tower_hp"][s] for s in (0, 1))
            want = tuple(ctx.state_budget(t.waves[s], ctx.capacity(t.levels_before[s], m4.side(s).items), lane, k, s)
                         for s in (0, 1))
            chk["budget_record"] += t.wave_budget != want
            r = env.step(a, b)
            chk["budget_record"] += r.info["wave_budget"] != t.wave_budget
            chk["turns"] += 1
            if r.done:
                break
            obs = r.obs
    out["soak"] = chk
    if ppo:
        out["ppo"] = ppo_smoke()
    return out


def ppo_smoke() -> dict:
    """A tiny shared-policy PPO run on m4_four_actions() (PPO_STEPS transitions): shapes, finiteness, masks, and the
    wave mix of the resulting policy. NOT a training result. The same budget on m4_lasthit() gives the throughput
    reference."""
    from ...env import LaneEnv
    from ...rl import PPOConfig, SharedConfig, train_shared
    from ...rl.agent import NetAgent
    out = {}
    for name, cfg in (("four_actions", cfg_four()), ("lasthit", GameConfig.m4_lasthit())):
        ppo = PPOConfig.continuing(cfg, total_steps=PPO_STEPS, n_envs=PPO_ENVS, seed=PPO_SEED, torch_threads=1)
        res = train_shared(cfg, ppo, SharedConfig(current_fraction=1.0), out_dir=None, name=name, verbose=False)
        h = res.history
        keys = ("policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm")
        rec = {"n_actions": res.net.n_actions, "updates": len(h), "agent_transitions": res.agent_transitions,
               "env_turns_per_s": res.env_turns / res.elapsed,
               "finite": all(np.isfinite(e[k]) for e in h for k in keys),
               "entropy_first_last": [h[0]["entropy"], h[-1]["entropy"]]}
        if name == "four_actions":
            agent = NetAgent(res.net, "smoke")
            env = LaneEnv(cfg, max_turns=200)
            rng = np.random.default_rng(1)
            obs = env.reset(seed=3)
            waves, illegal = np.zeros(4), 0
            while True:
                a, b = agent.act(obs[0], rng), agent.act(obs[1], rng)
                illegal += (a not in obs[0].legal) + (b not in obs[1].legal)
                for x in (a, b):
                    if x != CHAMP_ABSENT:
                        waves[champ_wave(x)] += 1
                r = env.step(a, b)
                if r.done:
                    break
                obs = r.obs
            rec["wave_mix"] = {NAMES[w]: float(waves[w] / waves.sum()) for w in ACTIONS}
            rec["illegal_actions"] = int(illegal)
            # a 46-output network refuses the 61-code space with a clear error
            from ...rl.network import PolicyValueNet
            old = NetAgent(PolicyValueNet(88, 46), "old46")
            try:
                old.act(env.reset(seed=0)[0], rng)
                rec["old_checkpoint_refused"] = False
            except ValueError:
                rec["old_checkpoint_refused"] = True
        out[name] = rec
    return out


# ---------------------------------------------------------------------------------------------- traces
TRACES = (("neutral SL v LH, L10", "neutral", None, "L10", SL, "walk_up"),
          ("neutral CL v LH, L10", "neutral", None, "L10", CL, "walk_up"),
          ("neutral SL v TN, L10", "neutral", None, "L10", SL, "thin"),
          ("8v24 TN v LH, A C=4", "imbalance", (8, 24), (4, IMBALANCE_B_CAP), TN, "walk_up"),
          ("8v24 TN v LH, A C=22", "imbalance", (8, 24), (22, IMBALANCE_B_CAP), TN, "walk_up"),
          ("8v24 TN v LH, A C=11", "imbalance", (8, 24), (11, IMBALANCE_B_CAP), TN, "walk_up"),
          ("freeze TN v walk-up, L10", "freeze", None, "L10", TN, "walk_up"),
          ("freeze SL v walk-up, L10", "freeze", None, "L10", SL, "walk_up"),
          ("freeze LH v clear, L10", "freeze", None, "L10", LH, "clear"),
          ("freeze B21 TN v walk-up, L18 A++", "freeze21", None, "L18 A++", TN, "walk_up"),
          ("freeze B21 LH v walk-up, L18 A++", "freeze21", None, "L18 A++", LH, "walk_up"))


def traces(st) -> dict:
    out = {}
    for label, sc, fronts, tier, slot, resp in TRACES:
        if sc == "imbalance":
            start, caps = with_fronts(st["neutral"], *fronts), tier
        elif sc == "freeze21":
            start, caps = with_fronts(st["freeze"], hb=21), capacities(tier)
        else:
            start, caps = st[sc], capacities(tier)
        r = play(response_cfg(resp), start, caps, const(code(slot), RESPONSE_CODE[resp]), TRACE_TURNS,
                 trace=TRACE_TURNS)
        out[label] = {"caps": caps, "strips": r["strips"],
                      "rows": [{k: x[k] for k in ("w", "bud", "need", "S", "dmg", "blob", "crash", "held")}
                               for x in r["rows"]]}
    return out


# ---------------------------------------------------------------------------------------------- run
def run(ppo: bool = True) -> dict:
    from ..m4.zoning_natural import starts
    t0 = time.perf_counter()
    st = starts(GameConfig.m4_lasthit())
    out = {"meta": meta(stage="four_actions", preset="GameConfig.m4_four_actions()",
                        fields={f: getattr(cfg_four(), f) for f in GameConfig.M4_FOUR_ACTIONS_FIELDS},
                        tiers={k: {"levels_invested": v, "capacities": capacities(k)} for k, v in TIERS.items()},
                        starts={k: [st[k][0].to_list(), st[k][1].to_list(), st[k][2]] for k in ("neutral", "freeze")},
                        rules={k: v[0] for k, v in RULES.items()})}
    for name, fn in (("formulations", stage_formulations), ("baseline", stage_baseline), ("neutral", stage_neutral),
                     ("imbalance", stage_imbalance),
                     ("freeze", stage_freeze), ("slowpush", stage_slowpush), ("power", stage_power),
                     ("traces", traces)):
        t1 = time.perf_counter()
        out[name] = fn(st)
        print(f"  {name}: {time.perf_counter() - t1:.1f} s", flush=True)
    out["formulation_summary"] = {k: formulation_summary(v) for k, v in out["formulations"].items()
                                  if not k.startswith("_")}
    t1 = time.perf_counter()
    out["integration"] = stage_integration(ppo=ppo)
    print(f"  integration: {time.perf_counter() - t1:.1f} s", flush=True)
    out["meta"]["elapsed_s"] = time.perf_counter() - t0
    save_json("four_actions", out)
    return out


# ---------------------------------------------------------------------------------------------- tables
def _f(x, nd: int = 2, sign: bool = True) -> str:
    if x is None:
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _c(x) -> str:
    return "-" if x is None else str(x)


def _report_formulations(out: dict, L: list) -> None:
    lh = out["formulations"]["_lasthit"]
    L.append("## Part B: formulation comparison\n")
    L.append("A's capacity C in {4, 6, 11, 22}, B Last Hits with C = 6. Columns: Thin from the neutral lane (60 turns, "
             "crashes summed over the four C); Thin against 8v12 / 8v16 / 8v20 / 8v24 / 4v20 (12 turns): B's crashes "
             "by C, A's own crashes (over-thinning) and the steps where B's crashes rise with C; Thin every turn from "
             "the freeze, longest hold of 40 turns against a walk-up / step-back at L1, L10, L10 A+, L18 A++ (Last Hit "
             f"gives {lh['L10|walk_up']['longest_hold']} / {lh['L10|step_back']['longest_hold']} at every tier); "
             "Slow Push from the neutral lane: A's first crash by C, crashes in 60 turns, HP per turn; Slow Push "
             "against Thin at C = 6 (A/B crashes).\n")
    L.append("| rule | Thin neutral crashes | Thin imbalance B crashes (C 4/6/11/22) | A crashes | rises | freeze "
             "walk-up | step-back | SP first crash | SP crashes | SP HP/turn | SP v Thin |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in out["formulation_summary"].items():
        if "sp_first" in s:
            sp = ("/".join(_c(s["sp_first"][C]) for C in IMBALANCE_CAPS),
                  "/".join(str(s["sp_crashes"][C]) for C in IMBALANCE_CAPS),
                  "/".join(f"{s['sp_dmg_per_turn'][C]:.2f}" for C in IMBALANCE_CAPS),
                  "/".join(map(str, s["sp_vs_thin_crashes"])))
        else:
            sp = ("-",) * 4
        L.append(f"| `{name}` | {s['thin_neutral_crashes']} | "
                 f"{'/'.join(str(s['imb_b_crashes'][C]) for C in IMBALANCE_CAPS)} | "
                 f"{sum(s['imb_a_crashes'].values())} | {s['imb_nonmonotone']} | "
                 f"{'/'.join(str(v) for v in s['freeze_walk'].values())} | "
                 f"{'/'.join(str(v) for v in s['freeze_step'].values())} | {sp[0]} | {sp[1]} | {sp[2]} | {sp[3]} |")
    L.append("")
    for name, (desc, _, _) in RULES.items():
        L.append(f"- `{name}`: {desc}")
    L.append("")


def _report_baseline(out: dict, L: list) -> None:
    b = out["baseline"]
    L.append("## Comparison with the CURRENT three-action model\n")
    L.append("The same scenarios with the budgets of `m4_lasthit()` (Hold round(C/8), Thin round(C/2)) against the "
             "four-action model. B Last Hits / Holds (its own capacity). Cells: A/B crashes in 60 turns (neutral); "
             "longest hold of 40 turns (freeze, B walks up).\n")
    L.append("| tier | neutral Hold v Hold → LH v LH | neutral Thin v Hold → Thin v LH | freeze Hold → LH | "
             "freeze Thin → Thin |")
    L.append("|---|---|---|---|---|")
    for tier in TIERS:
        n3 = [b[f"neutral|{tier}|{s}"]["crashes"] for s in ("LH", "TN")]
        n4 = [out["neutral"]["vs_lasthit"][f"{tier}|{s}"]["stats"]["crashes"] for s in ("LH", "TN")]
        f3 = [b[f"freeze|{tier}|{s}|walk_up"]["longest_hold"] for s in ("LH", "TN")]
        f4 = [out["freeze"]["standard"][f"{tier}|{s}|walk_up"]["stats"]["longest_hold"] for s in ("LH", "TN")]
        L.append(f"| {tier} | {n3[0][0]}/{n3[0][1]} → {n4[0][0]}/{n4[0][1]} | {n3[1][0]}/{n3[1][1]} → "
                 f"{n4[1][0]}/{n4[1][1]} | {f3[0]} → {f4[0]} | {f3[1]} → {f4[1]} |")
    L.append("\nThin against an oversized wave (12 turns, B Last Hits with C = 6). Cell: A's damage in turns 1-4 · HP "
             "removed above 1 + need · B/A crashes, three-action → four-action.\n")
    L.append("| start | A's C | three-action Thin | four-action Thin |")
    L.append("|---|---|---|---|")
    for ha, hb in IMBALANCES:
        for C in IMBALANCE_CAPS:
            r3 = b[f"imbalance|{ha}v{hb}|{C}"]
            r4 = out["imbalance"][f"{ha}v{hb}|{C}|TN"]
            L.append(f"| {ha}v{hb} | {C} | {','.join(map(str, r3['dmg_first4']))} · {r3['over_need']} · "
                     f"{r3['stats']['crashes'][1]}/{r3['stats']['crashes'][0]} | "
                     f"{','.join(map(str, r4['dmg_first4']))} · {r4['over_need']} · "
                     f"{r4['stats']['crashes'][1]}/{r4['stats']['crashes'][0]} |")
    L.append("")


def _report_neutral(out: dict, L: list) -> None:
    m = out["meta"]
    L.append("## Test 1: neutral lane\n")
    L.append("A's action against B's Last Hit, 60 turns from the neutral start. Cell: A/B crashes · A's first crash · "
             "A's damage per turn · mean contact cell · B - A gold.\n")
    L.append("| tier | " + " | ".join(NAMES[a] for a in ACTIONS) + " |")
    L.append("|---|" + "---|" * len(ACTIONS))
    for tier in TIERS:
        cells = []
        for a in ACTIONS:
            s = out["neutral"]["vs_lasthit"][f"{tier}|{SHORT[a]}"]["stats"]
            cells.append(f"{s['crashes'][0]}/{s['crashes'][1]} · {_c(s['first_crash'][0])} · "
                         f"{s['dmg'][0] / s['turns']:.2f} · {_f(s['contact_mean'])} · {_f(s['gold_diff_b'])}")
        L.append(f"| {tier} | " + " | ".join(cells) + " |")
    L.append("\nAction matrix (A's action in rows, B's in columns), 60 turns. Cell: A/B crashes · A - B gold.\n")
    for tier in ("L1", "L10", "L18", "L10 A+", "L18 A++"):
        caps = m["tiers"][tier]["capacities"]
        L.append(f"**{tier}** (C {caps[0]}/{caps[1]})\n")
        L.append("| A \\ B | " + " | ".join(NAMES[b] for b in ACTIONS) + " |")
        L.append("|---|" + "---|" * len(ACTIONS))
        for a in ACTIONS:
            cells = []
            for b in ACTIONS:
                s = out["neutral"]["matrix"][f"{tier}|{SHORT[a]}|{SHORT[b]}"]
                cells.append(f"{s['crashes'][0]}/{s['crashes'][1]} · {_f(-s['gold_diff_b'])}")
            L.append(f"| {NAMES[a]} | " + " | ".join(cells) + " |")
        L.append("")


def _report_imbalance(out: dict, L: list) -> None:
    L.append("## Test 2: oversized enemy wave\n")
    L.append("A's front against a larger B front at the lane cell, B Last Hits with C = 6, 12 turns. Cell: A's damage "
             "in turns 1-4 · B/A crashes · contact after turn 12 (lane = 0, A's tower = -2, `-` = no contact).\n")
    L.append("| start | A's C | " + " | ".join(NAMES[a] for a in ACTIONS) + " |")
    L.append("|---|---|" + "---|" * len(ACTIONS))
    for ha, hb in IMBALANCES:
        for C in IMBALANCE_CAPS:
            cells = []
            for a in ACTIONS:
                r = out["imbalance"][f"{ha}v{hb}|{C}|{SHORT[a]}"]
                s = r["stats"]
                cells.append(f"{','.join(map(str, r['dmg_first4']))} · {s['crashes'][1]}/{s['crashes'][0]} · "
                             f"{_c(r['contact_at']['12'])}")
            L.append(f"| {ha}v{hb} | {C} | " + " | ".join(cells) + " |")
    over = sum(r["over_need"] for k, r in out["imbalance"].items() if k.endswith("|TN"))
    L.append(f"\nThin damage above 1 + need, summed over every Thin run: {over} HP.\n")


def _report_freeze(out: dict, L: list) -> None:
    L.append("## Test 3: established freeze\n")
    L.append("A plays one action every turn from the freeze start (40 turns) against five responses of B. Cell: longest "
             "hold / first turn not held (held = B's front at A's -1 or deeper with both fronts alive).\n")
    L.append("| tier | A | " + " | ".join(RESPONSES) + " |")
    L.append("|---|---|" + "---|" * len(RESPONSES))
    for tier in TIERS:
        for a in ACTIONS:
            cells = []
            for resp in RESPONSES:
                s = out["freeze"]["standard"][f"{tier}|{SHORT[a]}|{resp}"]["stats"]
                cells.append(f"{s['longest_hold']}/{_c(s['first_break'])}")
            L.append(f"| {tier} | {NAMES[a]} | " + " | ".join(cells) + " |")
    L.append("\nEconomics at L10 (40 turns). Cell: B - A gold / gold + XP/8; A/B crashes.\n")
    L.append("| A | " + " | ".join(RESPONSES) + " |")
    L.append("|---|" + "---|" * len(RESPONSES))
    for a in ACTIONS:
        cells = []
        for resp in RESPONSES:
            s = out["freeze"]["standard"][f"L10|{SHORT[a]}|{resp}"]["stats"]
            cells.append(f"{_f(s['gold_diff_b'])} / {_f(s['value_b'])}; {s['crashes'][0]}/{s['crashes'][1]}")
        L.append(f"| {NAMES[a]} | " + " | ".join(cells) + " |")
    L.append("\nEnlarged held wave (B's front set to 13 or 21 HP at the freeze start), B walks up. Cell: longest hold · "
             "A/B crashes · A's damage in turns 1-4.\n")
    L.append("| tier | " + " | ".join(f"B{hb} {NAMES[a]}" for hb in FREEZE_ENLARGED for a in (LH, TN)) + " |")
    L.append("|---|" + "---|" * (2 * len(FREEZE_ENLARGED)))
    for tier in TIERS:
        cells = []
        for hb in FREEZE_ENLARGED:
            for a in (LH, TN):
                r = out["freeze"]["enlarged"][f"{hb}|{tier}|{SHORT[a]}"]
                s = r["stats"]
                cells.append(f"{s['longest_hold']} · {s['crashes'][0]}/{s['crashes'][1]} · "
                             f"{','.join(map(str, r['dmg_first4']))}")
        L.append(f"| {tier} | " + " | ".join(cells) + " |")
    L.append("")


def _report_slowpush(out: dict, L: list) -> None:
    L.append("## Test 4: Slow Push\n")
    L.append("From A's 8 HP against B's 8-14 HP at the lane cell, 40 turns. Cell: A's first crash · A/B crashes · mean "
             "HP of A's crashes · A - B gold.\n")
    cols = [(wa, wb) for wb in (LH, TN) for wa in (LH, SL, CL)]
    L.append("| start | tier | " + " | ".join(f"{SHORT[wa]} v {SHORT[wb]}" for wa, wb in cols) + " |")
    L.append("|---|---|" + "---|" * len(cols))
    for ha, hb in SLOWPUSH_STARTS:
        for tier in SLOWPUSH_TIERS:
            cells = []
            for wa, wb in cols:
                r = out["slowpush"][f"{ha}v{hb}|{tier}|{SHORT[wa]}|{SHORT[wb]}"]
                s = r["stats"]
                size = float(np.mean(r["crash_sizes"])) if r["crash_sizes"] else None
                cells.append(f"{_c(s['first_crash'][0])} · {s['crashes'][0]}/{s['crashes'][1]} · "
                             f"{_f(size, 1, False)} · {_f(-s['gold_diff_b'])}")
            L.append(f"| {ha}v{hb} | {tier} | " + " | ".join(cells) + " |")
    r = out["slowpush"]["8v8|L10|SL|LH"]
    L.append(f"\nSlow Push v Last Hit at L10 from 8v8, A's blob minus B's blob after each of the first 20 turns: "
             f"{r['blob_diff'][:20]}\n")


def _report_power(out: dict, L: list) -> None:
    p = out["power"]
    L.append("## Test 5: power levels\n")
    L.append(f"Budget invariants on {p['states']} sampled (state, side) pairs x every capacity 4-23: violations "
             f"{p['violations']}. Share of pairs with a zero need: Thin {p['zero_need']['TN']:.3f}, Slow Push "
             f"{p['zero_need']['SL']:.3f}.\n")
    caps_show = (4, 6, 8, 11, 14, 17, 22)
    L.append("| C | " + " | ".join(map(str, caps_show)) + " |")
    L.append("|---|" + "---|" * len(caps_show))
    L.append("| Thin cap binds (share) | " + " | ".join(f"{p['cap_binds'][C]['TN']:.3f}" for C in caps_show) + " |")
    L.append("| Slow Push cap binds | " + " | ".join(f"{p['cap_binds'][C]['SL']:.3f}" for C in caps_show) + " |")
    for a in ACTIONS:
        L.append(f"| mean {NAMES[a]} budget | "
                 + " | ".join(f"{p['mean_budget'][C][SHORT[a]]:.2f}" for C in caps_show) + " |")
    for key in ("LH=TN", "TN=SL", "LH=SL", "SL=CL", "all_distinct"):
        L.append(f"| share with {key} budgets | "
                 + " | ".join(f"{p['same_budget'][C][key]:.3f}" for C in caps_show) + " |")
    L.append("\nOutcome sweeps over A's capacity C = 4..23 (B Last Hits, C = 6):\n")
    for key, res in p["sweep"].items():
        if key.startswith("freeze"):
            vals = " ".join(f"{x['C']}:{x['longest_hold']}({x['crashes'][0]}/{x['crashes'][1]})" for x in res)
            L.append(f"- `{key}` longest hold (A/B crashes): {vals}")
        elif key.startswith("neutral"):
            vals = " ".join(f"{x['C']}:{_c(x['first'])}({x['crashes'][0]}/{x['crashes'][1]})" for x in res)
            L.append(f"- `{key}` first crash of A (A/B crashes): {vals}")
        else:
            vals = " ".join(f"{x['C']}:{x['crashes'][1]}/{x['crashes'][0]}" for x in res)
            L.append(f"- `{key}` B/A crashes: {vals}")
    own = {k: v for k, v in p["thin_own_crashes"].items() if not k.startswith("freeze")}
    L.append(f"\nThin's own crashes on the imbalance sweeps: {own}; steps where B's crashes rise with C: "
             f"{p['thin_b_crash_increases']}; Last Hit identical for every C: "
             f"{all(p['lasthit_identical_over_C'].values())}.\n")


def report(out: dict) -> str:
    m = out["meta"]
    L = ["# M4.3c four-action wave management: deterministic validation\n",
         f"Preset `{m['preset']}` (fields {m['fields']}), commit `{m['commit']}` + working tree, {m['timestamp']}. "
         "Deterministic scripted play, no chance events, fixed capacities per window; coordinates GLOBAL. Design and "
         "reading: docs/m4/m4_3c_four_action_wave_redesign.md.\n",
         "Tiers (capacity C of A / B): " + ", ".join(f"{k} {v['capacities'][0]}/{v['capacities'][1]}"
                                                     for k, v in m["tiers"].items()) + "\n"]
    _report_formulations(out, L)
    _report_baseline(out, L)
    _report_neutral(out, L)
    _report_imbalance(out, L)
    _report_freeze(out, L)
    _report_slowpush(out, L)
    _report_power(out, L)
    it = out["integration"]
    L.append("## Test 6: integration\n")
    L.append(f"Action space {it['n_actions']} codes, observation {it['obs_size']} entries. Soak (uniform legal play "
             f"with chance, {len(SOAK_SEEDS)} x {SOAK_TURNS} turns): {it['soak']}.\n")
    for name, r in it.get("ppo", {}).items():
        L.append(f"- PPO smoke `{name}`: {r}")
    L.append("\n## Traces\n")
    for label, r in out["traces"].items():
        L.append(f"**{label}** (C {r['caps'][0]}/{r['caps'][1]})\n")
        L.append("```")
        L.extend(r["strips"])
        L.append("```\n")
    md = "\n".join(L)
    write_md("four_actions", md)
    return md
