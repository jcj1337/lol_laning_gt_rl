"""M4.2e-C: wave action redesign and mechanical calibration (docs/natural_waves/m4_2e_wave_actions_report.md).

Deterministic trajectories of the candidate `GameConfig.m4_lasthit()` (Hold as last-hit damage, Thin and Hold as shares
of the wave-clear capacity, the half-wave Lanchester law, no separate Pressure thinning) against `GameConfig.m4_lane()`
(Hold 0, Thin fixed, whole-wave law): the M4 resolver with no chance events (u = None) and short scripted policies,
plus lane-only traces where noted. Champion power is varied through an attack-power multiplier per side
(`play(mult=...)`), which sets the capacity C = round(4 x mult) and every slot's share of it; alternative action
models are tested through explicit budget functions (`play(budget_fn=...)`). Nothing is trained; the sampled zoning
games are run by lanegame.experiments.m4.zoning_natural on the variants added there ("lane", "lasthit").

    python -m lanegame.experiments.m4.wave_actions            all stages, JSON to results/m4_wave_actions/
    python -m lanegame.experiments.m4.wave_actions STAGE...   named stages only

Stages (report section in brackets):
  budgets      [3] the three budgets by level and investment, for the candidate and the alternatives
  neutral      [5] Hold / Hold from s0: symmetry at equal power; the drift when one champion is stronger
  drift        [5] one champion last-hits, the other does not reach the wave: how fast the wave drifts (lane only and
               the resolver with a Safe or a recalling opponent), by the farmer's Hold budget
  actions      [6] Hold / Thin / Clear and the slow-push lines against a Holding opponent, 80 turns; kill ticks
  power        [3, 5] Clear and Thin by attack power from a neutral stacked lane and from s0
  sustain      [4] siege length against the reinforcement stream by tower damage
  pressure     [4, 6] the siege after a slow push: walk away, Hold / Thin / Clear + Pressure, by power, with and
               without the old Apply Pressure thinning
  freeze       [6] the outside-tower freeze: the seeded band, dynamic setups x threshold controllers, maintenance,
               breaks, the freezer's and the breaker's power
  bounce       [6] crash, siege, bounce timing of the slow-push lines
  recall       [6] Recall cost against the wave state, against a Holder and against a Holder that takes plates
  variants     [2] alternative action models x damage laws on the same probes
  scripted     [8] natural_suite pairs (correctness and variety)
  performance  [8] lane step and env step

Coordinates are GLOBAL (A walks +, A's tower at -2, B's at +2, the lane cell is 0).
"""

from __future__ import annotations

import os
import sys
import time
from fractions import Fraction
from typing import Callable, Dict, List, Optional

from ...champions import NO_REACH
from ...champions_m4 import M4State, context, initial_m4, resolve_turn_m4
from ...config import (B_HOLD, B_SAFE, F, PRESSURE, SH, SP, W_CLEAR, W_HOLD, W_THIN, GameConfig, champ_act)
from ...io import dump_json
from ...mechanics import initial_state, step_turn
from ...state import State
from . import natural_waves as nw
from . import natural_waves_final as nf
from .common import RESULTS_DIR, meta

OUT = os.path.join(RESULTS_DIR, "m4_wave_actions")
A, B = 0, 1
TURNS = [0]
HO, CL, TN = nw.HO, nw.CL, nw.TN
HOP, CLP = nf.HOP, nf.CLP
TNP = champ_act(W_THIN, B_HOLD, PRESSURE)
SAFE = champ_act(W_HOLD, B_SAFE)
RECALL = nw.RECALL
SLOTS = (W_HOLD, W_THIN, W_CLEAR)
MULTS = (1.0, 1.5, 2.0, 3.0, 4.0)
FREEZE_THRESHOLD = 14          # the candidate's freeze controller: Thin while the enemy front holds >= 14 HP (section 6)


def cand(**kw) -> GameConfig:
    return GameConfig.m4_lasthit(**kw)


def lane(**kw) -> GameConfig:
    return GameConfig.m4_lane(**kw)


def hold_pressure(k, l):
    """The passive opponent that takes plates: Hold, with Apply Pressure while it has a siege."""
    return HOP if l.bsg > 0 else HO


# ---------------------------------------------------------------------- budgets
def _rhu(v: Fraction) -> int:
    return (2 * v.numerator + v.denominator) // (2 * v.denominator)


def capacity_at(cfg: GameConfig, mult: float) -> int:
    """The wave-clear capacity C = round(ticks_per_turn x champion_wave_damage[Clear] x mult)."""
    return _rhu(Fraction(repr(float(mult))) * cfg.ticks_per_turn * cfg.champion_wave_damage[SH])


def budget_at(cfg: GameConfig, slot: int, mult: float) -> int:
    """A slot's per-turn budget at attack power `mult` (x1.0 = level 1, nothing invested) under the preset's own model:
    "attack_power" (m4_lane: Clear scaled, the rest fixed) or "capacity" (m4_lasthit: shares of C)."""
    if cfg.wave_clear_scaling == "capacity":
        return context(cfg).budget_from_capacity(slot, capacity_at(cfg, mult))
    base = cfg.ticks_per_turn * cfg.champion_wave_damage[slot]
    if cfg.wave_clear_scaling == "attack_power" and slot in cfg.wave_clear_scaled_slots:
        return _rhu(Fraction(repr(float(mult))) * base)
    return base


# alternative action models (report section 2), as budget functions of (cfg, slot, mult)
def model_fixed_hold(hold: int = 1, alpha: Fraction = Fraction(1, 3)) -> Callable:
    """Hold fixed at `hold` HP per turn, Thin = Hold + alpha (C - Hold), Clear = C (the brief's formula with a fixed
    last-hit damage)."""
    def f(cfg, slot, mult):
        C = capacity_at(cfg, mult)
        if slot == SH:
            return C
        if slot == SP:
            return hold
        return _rhu(hold + alpha * (C - hold))
    return f


def model_fractions(hold_frac: Fraction, alpha: Fraction) -> Callable:
    """The brief's family: Hold = round(h C), Thin = round(Hold + alpha (C - Hold))."""
    def f(cfg, slot, mult):
        C = capacity_at(cfg, mult)
        hold = _rhu(hold_frac * C)
        if slot == SH:
            return C
        if slot == SP:
            return hold
        return _rhu(hold + alpha * (C - hold))
    return f


def model_shares(hold_frac: Fraction, thin_frac: Fraction) -> Callable:
    """The candidate's family: Hold = round(h C), Thin = round(t C), Clear = C."""
    def f(cfg, slot, mult):
        C = capacity_at(cfg, mult)
        return C if slot == SH else _rhu((hold_frac if slot == SP else thin_frac) * C)
    return f


def model_hold_only(hold: int = 1, thin: int = 2) -> Callable:
    """Hold = `hold` and Thin = `thin` fixed, Clear = C: the A.1 model plus last-hit damage only."""
    def f(cfg, slot, mult):
        return capacity_at(cfg, mult) if slot == SH else (hold if slot == SP else thin)
    return f


MODELS = {
    "lane": None,                                                    # Hold 0, Thin 2, Clear C (m4_lane's model)
    "shares_1/8_1/2": model_shares(Fraction(1, 8), Fraction(1, 2)),  # the candidate: shares 1/8, 1/2, 1
    "alpha_1/3": model_fractions(Fraction(1, 8), Fraction(1, 3)),    # the brief's formula at alpha 1/3
    "alpha_1/2": model_fractions(Fraction(1, 8), Fraction(1, 2)),    # ... alpha 1/2
    "fixed_hold_1": model_fixed_hold(1, Fraction(3, 7)),             # Hold fixed at 1, Thin = 1 + 3/7 (C - 1)
    "shares_1/4_1/2": model_shares(Fraction(1, 4), Fraction(1, 2)),  # a larger last-hit share
    "hold_only": model_hold_only(1, 2),                              # the A.1 model plus Hold 1
}
LAWS = ("lanchester", "lanchester_half")


def budget_table(cfg: GameConfig, model: Optional[Callable] = None) -> List[dict]:
    """(Hold, Thin, Clear) per turn for representative (level, invested resources) states: the preset's own budgets,
    or those of an alternative model evaluated at the state's attack power."""
    ctx = context(cfg)
    rows = []
    for L, S in ((1, 0), (3, 0), (6, 4), (9, 0), (9, 16), (13, 32), (18, 64), (18, 10 ** 4)):
        units = S * cfg.gold_scale
        ap = ctx.inv.attack_power(L, units) / ctx.inv.one / 10_000
        if model is None:
            b = [ctx.wave_budget(w, L, units) for w in SLOTS]
        else:
            b = [model(cfg, w, ap) for w in SLOTS]
        rows.append({"level": L, "invested": S, "attack_power": round(ap, 3), "hold": b[0], "thin": b[1], "clear": b[2]})
    return rows


# ---------------------------------------------------------------------- drivers
def play(cfg, policy: Callable, turns: int, lane=None, m4=None, k0: int = 0, mult=None, budget_fn=None) -> List[dict]:
    """nw.play with optional budget overrides. `mult` = attack-power multipliers per side (the preset's own action model
    at that power, `budget_at`); `budget_fn(cfg, slot, mult)` = an alternative action model, evaluated at `mult`
    (default (1, 1)). Without either, the resolver's own budgets (from the champions' levels and investment)."""
    if mult is None and budget_fn is None:
        rows = nw.play(cfg, policy, turns, lane, m4, k0)
        TURNS[0] += len(rows)
        return rows
    mult = (1.0, 1.0) if mult is None else mult
    fn = budget_fn or (lambda c, w, m: budget_at(c, w, m))
    lane = lane if lane is not None else initial_state(cfg)
    m4 = m4 if m4 is not None else initial_m4(cfg)

    def lane_step(s, kk, x, y, reach, budget=None):
        b = tuple(fn(cfg, w, mult[i]) for i, w in enumerate((x % 3, y % 3)))
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


def line(rows, cfg) -> dict:
    s = nw.summarize(rows, cfg)
    return {"A_minus_B": s["earned_A_minus_B"], "farm": s["farm"], "plates": s["plates"], "tower_hits": s["tower_hits"],
            "crashes": s["crashes"], "sieges_A": nf.sieges(rows, A)[:6], "sieges_B": nf.sieges(rows, B)[:6],
            "contact": contact_summary(s["contact"])}


def contact_summary(cells: List[Optional[int]]) -> dict:
    """Where the contact stood: turns at each GLOBAL cell, turns without contact, the first turn it left the lane."""
    at = {}
    for c in cells:
        at[str(c)] = at.get(str(c), 0) + 1
    first_off = next((i for i, c in enumerate(cells) if c is not None and c != 0), None)
    with_c = [c for c in cells if c is not None]
    return {"turns_at": at, "first_turn_off_lane": first_off,
            "mean_cell": round(sum(with_c) / max(1, len(with_c)), 2)}


def first_crash(rows, s: int = A) -> Optional[dict]:
    sg = nf.sieges(rows, s)
    return None if not sg else sg[0]


def until_crash_from_neutral(cfg, code: int, n_turns: int, mult=(1.0, 1.0), k0: int = 9, horizon: int = 40,
                             budget_fn=None) -> Optional[dict]:
    """From the Hold / Hold lane at k0 (played at EQUAL power, so that a stronger champion's last-hitting does not
    move the lane before the probe starts), A plays `code` for n_turns turns (or until its wave crashes), then Holds;
    B Holds throughout. The first crash: turns after k0 and HP."""
    st0, m40, k = nw.end_state(play(cfg, nw.const(HO), k0, mult=(1.0, 1.0), budget_fn=budget_fn))
    st = {"c": False, "n": 0}

    def pol(kk, l, m):
        if l.asg > 0:
            st["c"] = True
        if st["c"]:
            return (HO, HO)
        st["n"] += 1
        return (code if st["n"] <= n_turns else HO, HO)
    rows = play(cfg, pol, horizon, st0, m40, k, mult=mult, budget_fn=budget_fn)
    fc = first_crash(rows)
    return None if fc is None else {"turns": fc["k"] - k, "hp": fc["first_hp"], "fronts_before": [st0.ah, st0.bh]}


def kill_ticks(cfg, slot: int, mult: float = 1.0) -> int:
    """Ticks for A (a fresh wave, champion at its front, the slot's budget at `mult`) to kill a fresh B wave already in
    contact, with no reinforcement on either side (lane only)."""
    c = cfg.with_(spawn_interval=100 * cfg.ticks_per_turn)
    st = State(0, c.wave_hp, (), 0, -1, c.wave_hp, (), 0, 0, 0)
    b = (budget_at(cfg, slot, mult), 0)
    for k in range(1, 20):
        res = step_turn(st, k, slot, W_HOLD, c, record=True, champion_reach=(c.tower, NO_REACH), champion_budget=b)
        TURNS[0] += 1
        for i, tr in enumerate(res.trace):
            if tr["fight"] and tr["fight"][3] <= tr["fight"][5]:
                return 2 * (k - 1) + i + 1
        st = res.state
    return -1


# ---------------------------------------------------------------------- freeze controllers
def threshold_freezer(t: int = FREEZE_THRESHOLD, line: int = -1) -> Callable:
    """Hold until the contact stands at own cell <= line; then Thin while the enemy front holds >= t HP, else Hold.
    The candidate's band (section 6) is an absolute enemy size, not a margin over the own front."""
    def wave(l):
        c = nw.contact_cell(l)
        if c is None or c > line:
            return HO
        return TN if l.bh >= t else HO
    return wave


def freeze_setup(cfg, controller: Callable, setup: Callable, turns: int = 70, mult=None, budget_fn=None):
    """B pushes per `setup(k)` then Holds; A plays `controller(lane)`. Returns the rows and the start index of the
    longest hold (the first turn of the longest run of post-turn contacts at GLOBAL <= -1), or None."""
    rows = play(cfg, lambda k, l, m: (controller(l), setup(k)), turns, mult=mult, budget_fn=budget_fn)
    cells = nw.summarize(rows, cfg)["contact"]
    if nf.longest_hold(cells) == 0:
        return rows, None
    return rows, max(range(len(cells)), key=lambda i: nf.held_turns(cells, i))


def hold_state(rows, start: int):
    """(lane, champions, k) at the START of the turn `start` (the first turn of the hold)."""
    r = rows[start]
    m4 = rows[start - 1]["t"].champ if start > 0 else None
    return r["pre"], m4, r["k"]


# ---------------------------------------------------------------------- stage: budgets
def stage_budgets() -> dict:
    out = {"m4_lane": budget_table(lane()), "candidate": budget_table(cand())}
    for name, model in MODELS.items():
        if model is not None:
            out[f"model_{name}"] = budget_table(cand(), model)
    out["candidate_matches_model"] = out["candidate"] == out["model_shares_1/8_1/2"]
    return out


# ---------------------------------------------------------------------- stage: neutral
def stage_neutral(turns: int = 40) -> dict:
    out = {}
    for name, cfg in (("m4_lane", lane()), ("candidate", cand())):
        rows = play(cfg, nw.const(HO), turns)
        s = nw.summarize(rows, cfg)
        out[name] = {"A_minus_B": s["earned_A_minus_B"], "farm": s["farm"], "contact": contact_summary(s["contact"]),
                     "fronts_k9_k20": s["fronts"][9:21], "max_front": max(max(f) for f in s["fronts"]),
                     "crashes": s["crashes"], "trajectory_k0_k14": nw.trajectory(rows[:14], cfg)}
        out[f"{name}_stronger_A"] = {}
        for m in MULTS:
            rows = play(cfg, nw.const(HO), turns, mult=(m, 1.0))
            s = nw.summarize(rows, cfg)
            out[f"{name}_stronger_A"][str(m)] = {
                "hold_budget_A": budget_at(cfg, W_HOLD, m), "A_minus_B": s["earned_A_minus_B"], "farm": s["farm"],
                "contact": contact_summary(s["contact"]), "first_crash_A": first_crash(rows)}
    return out


# ---------------------------------------------------------------------- stage: drift
def lane_only_drift(cfg: GameConfig, hold_a: int, turns: int = 40) -> dict:
    """Lane only: A's champion stands at its front and spends `hold_a` HP per turn on the enemy front (farm-mode
    reach); B's champion is absent (NO_REACH). Turns until A's wave crashes, the crash HP and the contact cells."""
    st, reach = initial_state(cfg), nw.farm_reach(cfg)
    cells, crash = [], None
    for k in range(turns):
        r = step_turn(st, k, W_HOLD, W_HOLD, cfg, champion_reach=(reach(st)[A], NO_REACH), champion_budget=(hold_a, 0))
        TURNS[0] += 1
        cells.append(nw.contact_cell(r.state))
        if r.crash_hp[A] and crash is None:
            crash = {"k": k, "hp": r.crash_hp[A]}
        st = r.state
    return {"first_crash": crash, "contact": contact_summary(cells)}


def stage_drift(turns: int = 40) -> dict:
    out = {"lane_only_by_hold_budget": {}}
    for name, cfg in (("m4_lane", lane()), ("candidate", cand())):
        out["lane_only_by_hold_budget"][name] = {h: lane_only_drift(cfg, h, turns) for h in (0, 1, 2, 3, 4)}
    for key, bpol in (("A_hold_vs_B_safe", nw.const(HO, SAFE)), ("A_hold_vs_B_recalling", nw.const(HO, RECALL))):
        out[key] = {}
        for name, c in (("m4_lane", lane()), ("candidate", cand())):
            rows = play(c, bpol, turns)
            s = nw.summarize(rows, c)
            out[key][name] = {"A_minus_B": s["earned_A_minus_B"], "farm": s["farm"], "xp": s["xp"],
                              "contact": contact_summary(s["contact"]), "first_crash_A": first_crash(rows),
                              "crashes_A": s["crashes"][A][:4]}
            if key == "A_hold_vs_B_safe":
                out[key][name]["trajectory_k8_k24"] = nw.trajectory(rows[8:24], c)
    return out


# ---------------------------------------------------------------------- stage: actions
def stage_actions(turns: int = 80) -> dict:
    out = {}
    for name, cfg in (("m4_lane", lane()), ("candidate", cand())):
        d = {"vs_hold": {n: line(play(cfg, lambda k, l, m, f=f: (f(k, l), HO), turns), cfg) for n, f in nf.ACTION_LINES.items()},
             "vs_hold_pressure": {n: line(play(cfg, lambda k, l, m, f=f: (f(k, l), hold_pressure(k, l)), turns), cfg)
                                  for n, f in nf.ACTION_LINES.items()},
             "kill_fresh_wave_ticks": {str(m): {n: kill_ticks(cfg, w, m) for n, w in (("hold", W_HOLD), ("thin", W_THIN),
                                                                                       ("clear", W_CLEAR))}
                                       for m in (1.0, 2.0, 4.0)}}
        out[name] = d
    return out


# ---------------------------------------------------------------------- stage: power
def stage_power() -> dict:
    out = {}
    for name, cfg in (("m4_lane", lane()), ("candidate", cand())):
        d = {"from_neutral_clear": {}, "from_neutral_thin": {}, "from_neutral_thin_x1": {}, "from_neutral_thin_x2": {},
             "from_s0": {}}
        for m in MULTS:
            d["from_neutral_clear"][str(m)] = {f"k{k0}": until_crash_from_neutral(cfg, CL, 99, (m, 1.0), k0) for k0 in (9, 10)}
            d["from_neutral_thin"][str(m)] = {f"k{k0}": until_crash_from_neutral(cfg, TN, 99, (m, 1.0), k0) for k0 in (9, 10)}
            d["from_neutral_thin_x1"][str(m)] = until_crash_from_neutral(cfg, TN, 1, (m, 1.0), 9)
            d["from_neutral_thin_x2"][str(m)] = until_crash_from_neutral(cfg, TN, 2, (m, 1.0), 9)
            d["from_s0"][str(m)] = {
                n: line(play(cfg, lambda k, l, mm, f=f: (f(k), HO), 40, mult=(m, 1.0)), cfg)
                for n, f in (("clear", lambda k: CL), ("thin", lambda k: TN), ("thin_k1", lambda k: TN if k == 1 else HO),
                             ("thin3_clear", lambda k: TN if k <= 3 else CL))}
        out[name] = d
    return out


# ---------------------------------------------------------------------- stage: sustain
def stage_sustain() -> dict:
    out = {}
    for name, cfg in (("m4_lane", lane()), ("candidate", cand())):
        out[name] = {}
        for tsd in (1, 2, 3, 4):
            c = cfg.with_(tower_siege_damage=tsd)
            out[name][f"tower{tsd}"] = {n: nf.sieges(play(c, lambda k, l, m, f=f: (f(k), HO), 60), A)[:3]
                                        for n, f in (("thin_k1", lambda k: TN if k == 1 else HO), ("clear", lambda k: CL),
                                                     ("thin", lambda k: TN))}
    return out


# ---------------------------------------------------------------------- stage: pressure
def siege_lines(cfg, m: float, after_codes) -> dict:
    """A slow-pushes (Thin at k1), and from its first crash on plays `after` while its siege lasts (Hold between
    sieges); B Holds. 44 turns."""
    res = {}
    for n, after in after_codes:
        st = {"c": False}

        def pol(k, l, mm, after=after, st=st):
            if l.asg > 0:
                st["c"] = True
            if not st["c"]:
                return (TN if k == 1 else HO, HO)
            return (after if l.asg > 0 else HO, HO)
        rows = play(cfg, pol, 44, mult=(m, 1.0))
        s = nw.summarize(rows, cfg)
        res[n] = {"A_minus_B": s["earned_A_minus_B"], "plates": s["plates"], "farm": s["farm"],
                  "sieges_A": nf.sieges(rows, A)[:3], "bounce_B": s["crashes"][B][:2]}
    return res


def stage_pressure() -> dict:
    lines = (("hold", HO), ("hold_pressure", HOP), ("thin_pressure", TNP), ("clear_pressure", CLP))
    out = {}
    for name, cfg in (("m4_lane", lane()), ("candidate", cand()), ("candidate_old_thinning", cand(thin_damage=1))):
        out[name] = {str(m): siege_lines(cfg, m, lines) for m in (1.0, 2.0, 4.0)}
    return out


# ---------------------------------------------------------------------- stage: freeze
def seeded_freeze(cfg, F_: int, E_: int, controller: Callable, turns: int = 30, mult=None) -> dict:
    """A seeded (not played-to) freeze state, for the band analysis only: A's front F in its tower cell (-2), B's
    front E one cell in front of it (GLOBAL -1), A at -2, B at its own +1, no rears, k = 12. A plays the controller,
    B Holds."""
    lane0 = State(-cfg.tower, F_, (), 0, cfg.tower - 1, E_, (), 0, 0, 0)
    m40 = initial_m4(cfg)
    m40 = M4State(m40.a._replace(pos=-cfg.tower), m40.b._replace(pos=cfg.tower - 1))
    rows = play(cfg, lambda k, l, m: (controller(l), HO), turns, lane0, m40, 12, mult=mult)
    s = nw.summarize(rows, cfg)
    return {"held": nf.held_turns(s["contact"]), "end_fronts": s["fronts"][-1], "farm": s["farm"],
            "crash_A": s["crashes"][A][:1], "crash_B": s["crashes"][B][:1]}


def stage_freeze() -> dict:
    out = {"seeded_band": {}, "dynamic": {}}
    controllers = {"hold": lambda l: HO, "thin": lambda l: TN}
    for t in (12, 13, 14, 15, 16):
        controllers[f"t{t}"] = threshold_freezer(t)
    for margin in (6, 8, 10):
        controllers[f"m{margin}"] = nw.bounded_freezer(margin)
    for name, cfg in (("m4_lane", lane()), ("candidate", cand())):
        out["seeded_band"][name] = {}
        for F_ in (8, 12):
            for E_ in (8, 10, 12, 14, 15, 16, 20, 24):
                out["seeded_band"][name][f"F{F_}_E{E_}"] = {cn: seeded_freeze(cfg, F_, E_, c) for cn, c in controllers.items()}
        # dynamic setups x controllers from s0
        out["dynamic"][name] = {}
        for sn, bp in nf.SETUPS.items():
            for cn, c in controllers.items():
                if cn in ("hold", "thin"):
                    continue
                rows, start = freeze_setup(cfg, c, bp)
                s = nw.summarize(rows, cfg)
                out["dynamic"][name][f"{sn}_{cn}"] = {
                    "longest_hold": nf.longest_hold(s["contact"]), "hold_start_k": None if start is None else rows[start]["k"],
                    "A_minus_B": s["earned_A_minus_B"], "farm": s["farm"], "crashes": s["crashes"]}
    # maintenance, breaks and power from the candidate's best dynamic hold (the threshold controller)
    cfg = cand()
    fz = threshold_freezer(FREEZE_THRESHOLD)
    best = max(((sn, nf.SETUPS[sn]) for sn in nf.SETUPS),
               key=lambda x: out["dynamic"]["candidate"][f"{x[0]}_t{FREEZE_THRESHOLD}"]["longest_hold"])
    rows, start = freeze_setup(cfg, fz, best[1])
    lane0, m40, k0 = hold_state(rows, start)
    out["candidate_hold"] = {"setup": best[0], "k0": k0, "state": [nw.side_str(lane0, A), nw.side_str(lane0, B)],
                             "setup_trajectory": nw.trajectory(rows[max(0, start - 6):start + 14], cfg), "from_hold_start": {}}
    for n, pa, pb in (("freezer_vs_hold", lambda l: fz(l), lambda k: HO),
                      ("hold_vs_hold_uncontrolled", lambda l: HO, lambda k: HO),
                      ("freezer_vs_thin", lambda l: fz(l), lambda k: TN),
                      ("freezer_vs_clear", lambda l: fz(l), lambda k: CL),
                      ("freezer_vs_one_clear", lambda l: fz(l), lambda k: CL if k == k0 else HO)):
        r2 = play(cfg, lambda k, l, m, pa=pa, pb=pb: (pa(l), pb(k)), 24, lane0, m40, k0)
        s2 = nw.summarize(r2, cfg)
        out["candidate_hold"]["from_hold_start"][n] = {
            "held_turns": nf.held_turns(s2["contact"]), "A_minus_B": s2["earned_A_minus_B"], "farm": s2["farm"],
            "xp": s2["xp"], "crashes": s2["crashes"], "contact": s2["contact"]}
        if n == "freezer_vs_clear":
            out["candidate_hold"]["break_trajectory"] = nw.trajectory(r2[:6], cfg)
    out["candidate_hold"]["freezer_power"], out["candidate_hold"]["breaker_power"] = {}, {}
    for m in (1.0, 2.0, 3.0, 4.0):
        r2 = play(cfg, lambda k, l, mm: (fz(l), HO), 24, lane0, m40, k0, mult=(m, 1.0))
        s2 = nw.summarize(r2, cfg)
        out["candidate_hold"]["freezer_power"][str(m)] = {
            "hold_budget": budget_at(cfg, W_HOLD, m), "thin_budget": budget_at(cfg, W_THIN, m),
            "held_turns": nf.held_turns(s2["contact"]), "farm": s2["farm"], "crashes_A": s2["crashes"][A][:2]}
        r3 = play(cfg, lambda k, l, mm: (fz(l), CL if k == k0 else HO), 24, lane0, m40, k0, mult=(1.0, m))
        s3 = nw.summarize(r3, cfg)
        out["candidate_hold"]["breaker_power"][str(m)] = {"held_turns": nf.held_turns(s3["contact"]),
                                                          "crashes_B": s3["crashes"][B][:2]}
    return out


# ---------------------------------------------------------------------- stage: bounce
def stage_bounce() -> dict:
    out = {}
    for name, cfg in (("m4_lane", lane()), ("candidate", cand())):
        out[name] = {n: nw.crash_report(play(cfg, nw.until_crash(f), 50)) for n, f in nw.BOUNCE_LINES.items()}
    cfg = cand()
    rows = play(cfg, nw.until_crash(lambda k: TN if k == 1 else HO), 80)
    s = nw.summarize(rows, cfg)
    out["candidate_cycle_thin_k1"] = {"crashes": s["crashes"], "contact": s["contact"]}
    return out


# ---------------------------------------------------------------------- stage: recall
def recall_pair(cfg, pre: Callable, trigger: Callable, b: Callable, turns: int = 40) -> dict:
    """A plays pre(k, lane) until trigger(k, lane), walks back, Recalls and Holds, against B's b(k, lane); the cost is
    the stay line's A - B minus the recall line's."""
    p_rec, st = nw.recall_line(pre, trigger)
    rec = play(cfg, lambda k, l, m: (p_rec(k, l, m)[0], b(k, l)), turns)
    stay = play(cfg, lambda k, l, m: (pre(k, l), b(k, l)), turns)
    sr, ss = nw.summarize(rec, cfg), nw.summarize(stay, cfg)
    return {"recall_k": st["recall_k"], "cost_A_minus_B": round(ss["earned_A_minus_B"] - sr["earned_A_minus_B"], 4),
            "missed_gold": round(ss["farm"][A] - sr["farm"][A], 4), "missed_xp": ss["xp"][A] - sr["xp"][A],
            "plates_recall": sr["plates"], "plates_stay": ss["plates"], "crashes_recall": sr["crashes"],
            "crashes_stay": ss["crashes"]}


def stage_recall() -> dict:
    out = {}
    for name, cfg in (("m4_lane", lane()), ("candidate", cand())):
        out[name] = {}
        for bn, b in (("vs_hold", lambda k, l: HO), ("vs_hold_pressure", hold_pressure)):
            d = {"neutral_k12": recall_pair(cfg, lambda k, l: HO, lambda k, l: k >= 12, b),
                 "after_small_crash": recall_pair(cfg, nw._crashed(lambda k: TN), lambda k, l: l.asg > 0, b),
                 "after_large_crash": recall_pair(cfg, nw._crashed(lambda k: TN if 1 <= k <= 4 else HO), lambda k, l: l.asg > 0, b),
                 "after_failed_push": recall_pair(cfg, lambda k, l: CL if k <= 6 else HO, lambda k, l: k >= 8, b)}
            build = (lambda k, l: TN if 1 <= k <= 4 else HO) if bn == "vs_hold" else \
                (lambda k, l: HOP if l.bsg > 0 else (TN if 1 <= k <= 4 else HO))
            d["enemy_builds_k1_4"] = {f"recall_from_k{rk}": recall_pair(cfg, lambda k, l: HO, lambda k, l, rk=rk: k >= rk, build, 36)
                                      for rk in (6, 11, 13, 15, 17, 20)}
            out[name][bn] = d
    return out


# ---------------------------------------------------------------------- stage: variants
def stage_variants() -> dict:
    """Every alternative action model x damage law on the same mechanical probes."""
    out = {}
    for law in LAWS:
        for name, model in MODELS.items():
            c = (lane() if model is None else cand()).with_(damage_law=law)
            fn = None if model is None else model
            d = {"budgets": {str(m): [budget_at(c, w, m) if fn is None else fn(c, w, m) for w in SLOTS] for m in (1.0, 2.0, 3.0, 4.0)}}
            rows = play(c, nw.const(HO), 40, mult=(1.0, 1.0), budget_fn=fn)
            s = nw.summarize(rows, c)
            d["neutral"] = {"symmetric": s["earned_A_minus_B"] == 0 and all(x in (None, 0) for x in s["contact"]),
                            "front_range": [min(f[0] for f in s["fronts"][8:]), max(f[0] for f in s["fronts"][8:])],
                            "farm_A": s["farm"][A]}
            rows = play(c, nw.const(HO), 40, mult=(3.0, 1.0), budget_fn=fn)
            d["stronger_A_hold"] = {"first_crash_A": first_crash(rows), "contact": contact_summary(nw.summarize(rows, c)["contact"])}
            rows = play(c, nw.const(HO, SAFE), 40, mult=(1.0, 1.0), budget_fn=fn)
            s = nw.summarize(rows, c)
            d["drift_vs_safe"] = {"first_crash_A": first_crash(rows), "A_minus_B": s["earned_A_minus_B"], "farm": s["farm"]}
            d["slow_push_from_neutral"] = {f"thin_x{n}": until_crash_from_neutral(c, TN, n, (1.0, 1.0), 9, budget_fn=fn)
                                           for n in (1, 2, 4)}
            d["thin_until_crash_by_ap"] = {str(m): until_crash_from_neutral(c, TN, 99, (m, 1.0), 9, budget_fn=fn)
                                           for m in (1.0, 2.0, 3.0, 4.0)}
            d["clear_until_crash_by_ap"] = {str(m): until_crash_from_neutral(c, CL, 99, (m, 1.0), 9, budget_fn=fn)
                                            for m in (1.0, 2.0, 3.0, 4.0)}
            best = {}
            for cn, ctrl in (("t14", threshold_freezer(14)), ("m8", nw.bounded_freezer(8))):
                h = 0
                for sn, bp in nf.SETUPS.items():
                    rows, _ = freeze_setup(c, ctrl, bp, 70, budget_fn=fn)
                    h = max(h, nf.longest_hold(nw.summarize(rows, c)["contact"]))
                best[cn] = h
            d["freeze_longest_hold"] = best
            d["seeded_F8_E14_t14"] = seeded_freeze(c, 8, 14, threshold_freezer(14)) if fn is None else \
                seeded_freeze_fn(c, 8, 14, threshold_freezer(14), fn)
            out[f"{name}/{law}"] = d
    return out


def seeded_freeze_fn(cfg, F_, E_, controller, fn, turns=30):
    lane0 = State(-cfg.tower, F_, (), 0, cfg.tower - 1, E_, (), 0, 0, 0)
    m40 = initial_m4(cfg)
    m40 = M4State(m40.a._replace(pos=-cfg.tower), m40.b._replace(pos=cfg.tower - 1))
    rows = play(cfg, lambda k, l, m: (controller(l), HO), turns, lane0, m40, 12, mult=(1.0, 1.0), budget_fn=fn)
    s = nw.summarize(rows, cfg)
    return {"held": nf.held_turns(s["contact"]), "end_fronts": s["fronts"][-1], "farm": s["farm"],
            "crash_A": s["crashes"][A][:1], "crash_B": s["crashes"][B][:1]}


# ---------------------------------------------------------------------- stage: scripted and performance
def stage_scripted() -> dict:
    return {"m4_lane": nw.stage_scripted(lane()), "candidate": nw.stage_scripted(cand())}


def stage_performance(n: int = 1000) -> dict:
    import numpy as np
    from ...env import LaneEnv, UniformAgent
    out = {}
    for name, cfg in (("m4_lane", lane()), ("candidate", cand())):
        rng = np.random.default_rng(0)
        st, t0 = initial_state(cfg), time.perf_counter()
        for k in range(n):
            st = step_turn(st, k, int(rng.integers(3)), int(rng.integers(3)), cfg,
                           champion_reach=(int(rng.integers(-2, 4)), int(rng.integers(-2, 4))),
                           champion_budget=(int(rng.integers(0, 12)), int(rng.integers(0, 12)))).state
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


STAGES = {"budgets": stage_budgets, "neutral": stage_neutral, "drift": stage_drift, "actions": stage_actions,
          "power": stage_power, "sustain": stage_sustain, "pressure": stage_pressure, "freeze": stage_freeze,
          "bounce": stage_bounce, "recall": stage_recall, "variants": stage_variants, "scripted": stage_scripted,
          "performance": stage_performance}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    names = [a for a in argv if a in STAGES] or list(STAGES)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    TURNS[0] = nw.TURNS[0] = 0
    data = {}
    for n in names:
        t1 = time.time()
        data[n] = STAGES[n]()
        print(f"stage {n}: {time.time() - t1:.1f} s")
    data["budget"] = {"turns": TURNS[0] + nw.TURNS[0], "wall_s": round(time.time() - t0, 2)}
    fn = "wave_actions.json" if len(names) == len(STAGES) else "wave_actions_" + "_".join(names) + ".json"
    dump_json({**meta(cand()), **data}, os.path.join(OUT, fn))
    print("budget", data["budget"], "->", fn)


if __name__ == "__main__":
    main()
