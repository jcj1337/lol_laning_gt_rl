"""M4.3c stabilization, Stage B: economic and strategic calibration (docs/m4/m4_3/m4_3c_strategic_economics_calibration.md).

Everything runs on `GameConfig.m4_four_actions()` or an opt-in variant of it (VARIANTS: one isolated change each, or a
named combination), through the real environment, with the scripted strategies of `strategies.LIBRARY`, on paired
seeds (root ECON_SEED; game e of every cell uses SeedSequence(ECON_SEED, spawn_key=(e,))). Nothing here changes a
preset.

Instrument: restricted zero-sum matrix games. From a start state (MATRIX_SCENARIOS), every strategy of A's set
against every strategy of B's set, N paired games of W turns. Per cell the harness window statistics (gold, XP,
deaths, Recalls, HP lost by source, power at the end, ...) are aggregated; the matrix of mean values to A is solved
(`lanegame.matrix_game.solve_matrix_game`) for each objective:

    J      sum_{k < W} gamma^k (gold_A - gold_B)_k                    the CURRENT reward, discounted (gamma 0.99)
    Jx     J + lambda sum_{k < W} gamma^k (XP_A - XP_B)_k, lambda 1/8   the XP-inclusive candidate (B2), on the SAME
                                                                       games (the reward does not change the dynamics
                                                                       of scripted agents)
    G40    undiscounted gold difference over the first 40 turns          the scenario-local economics

Reported per solution: the value, both equilibrium mixtures, each side's best reply to every strategy of the other,
the equilibrium weight of B's conceding strategies (CONCEDE) and of A's chasing strategies (CHASE). These are games
over small scripted strategy sets: an equilibrium of such a game is NOT an equilibrium of the laning game, and a best
reply within the set is not a best response.

    python -m lanegame.experiments.m43c economics [--variants base k2 ...] [--workers 10]
        -> results/m43c/economics_<variant>.json.gz, one file per variant (never overwritten by another variant;
           scenarios already stored are kept). Robustness reruns: run(..., seed_root=76_500, tag="_r2", episodes=128)
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ...config import GameConfig
from . import harness as h
from . import strategies as st
from .common import load, meta, save_json

ECON_SEED = 76_000
EPISODES = 64
TURNS = 100
SUB = (20, 40)
LAMBDA = 1 / 8

FREEZE_A = ("farm4", "freeze4", "freeze_hunt4", "freeze_bully4", "shove4")
FREEZE_B = ("walk_up4", "walk_up_thin4", "walk_up_shove4", "step_back4", "respect4", "cautious4", "contest4", "hunt4")
NEUTRAL = ("farm4", "thin4", "slowpush4", "build_shove4", "shove4", "hunt4", "shove_hunt4", "respect4")
CONCEDE = ("step_back4", "respect4")             # give way unconditionally (respect: whenever the enemy is in reach)
CONDITIONAL = ("cautious4",)                     # gives way only when threatened by a stronger enemy
CHASE = ("freeze_hunt4", "freeze_bully4", "hunt4", "shove_hunt4")

# name -> (base start, gap, level, B front HP, A set, B set)
MATRIX_SCENARIOS: Dict[str, tuple] = {
    "F4": ("freeze", "equal", 4, None, FREEZE_A, FREEZE_B),
    "F10": ("freeze", "equal", 10, None, FREEZE_A, FREEZE_B),
    "F10m": ("freeze", "moderate", 10, None, FREEZE_A, FREEZE_B),
    "F10g": ("freeze", "large", 10, None, FREEZE_A, FREEZE_B),
    "F10gl": ("freeze", "large_levels", 10, None, FREEZE_A, FREEZE_B),
    "F10b": ("freeze", "b_large", 10, None, FREEZE_A, FREEZE_B),
    "N3": ("neutral", "equal", 3, None, NEUTRAL, NEUTRAL),
    "N10": ("neutral", "equal", 10, None, NEUTRAL, NEUTRAL),
    "N10g": ("neutral", "large", 10, None, NEUTRAL, NEUTRAL),
}
# freeze-PHASE games: the choice is only what to do while the freeze holds; both sides Shove otherwise (the baseline
# equilibrium strategy of the full games), so the matrix isolates the value of the freeze-phase response
PHASE_A = ("shove4", "pz_keep", "pz_hunt", "pz_bully")
PHASE_B = ("shove4", "pz_walk", "pz_thin", "pz_back", "pz_cautious", "pz_contest")
PHASE_SCENARIOS: Dict[str, tuple] = {
    "P4": ("freeze", "equal", 4, None, PHASE_A, PHASE_B),
    "P10": ("freeze", "equal", 10, None, PHASE_A, PHASE_B),
    "P10g": ("freeze", "large", 10, None, PHASE_A, PHASE_B),
    "P10gl": ("freeze", "large_levels", 10, None, PHASE_A, PHASE_B),
    "P10b": ("freeze", "b_large", 10, None, PHASE_A, PHASE_B),
}
MATRIX_SCENARIOS.update(PHASE_SCENARIOS)
CONCEDE = CONCEDE + ("pz_back",)
CONDITIONAL = CONDITIONAL + ("pz_cautious",)
CHASE = CHASE + ("pz_hunt", "pz_bully")
SCENARIO_LABEL = {"F4": "freeze, level 4, equal", "F10": "freeze, level 10, equal",
                  "F10m": "freeze, L10, A x1.24 attack power (16 v 4 invested)",
                  "F10g": "freeze, L10, A x1.76 (50 v 0 invested; duel x3.1)",
                  "F10gl": "freeze, L10, A +3 levels and 50 invested (x2.04; duel x4.2)",
                  "F10b": "freeze, L10, the freezer is the weak side (B x1.76)",
                  "N3": "neutral lane, level 3, equal", "N10": "neutral lane, level 10, equal",
                  "N10g": "neutral lane, L10, A x1.76",
                  "P4": "freeze phase only, level 4, equal", "P10": "freeze phase only, L10, equal",
                  "P10g": "freeze phase only, L10, A x1.76", "P10gl": "freeze phase only, L10, A x2.04 (+3 levels)",
                  "P10b": "freeze phase only, L10, the freezer is the weak side"}
FULL_SCENARIOS = tuple(k for k in MATRIX_SCENARIOS if k not in PHASE_SCENARIOS)


def _death_schedule(rule: str, cap: int = 18) -> Tuple[int, ...]:
    if rule == "S1":                              # 2 + ceil(L / 6): 3 (L1-6), 4 (L7-12), 5 (L13-18)
        return tuple(2 + -(-L // 6) for L in range(1, cap + 1))
    raise ValueError(rule)


# name -> (description, overrides of m4_four_actions()); "x" variants change the objective only (scored from base)
VARIANTS: Dict[str, Tuple[str, dict]] = {
    "base": ("the four-action baseline m4_four_actions()", {}),
    "k1": ("kill bounty 1.0 (one wave)", {"kill_bounty": 1.0}),
    "k2": ("kill bounty 2.0 (two waves; a League kill is about 2.4 early waves)", {"kill_bounty": 2.0}),
    "d5": ("death absence 5 turns", {"death_turns": 5, "champion_absent_max": 5}),
    "d7": ("death absence 7 turns", {"death_turns": 7, "champion_absent_max": 7}),
    "dS1": ("level-scaled death absence 2 + ceil(L/6): 3 / 4 / 5 turns",
            {"death_turns_by_level": _death_schedule("S1"), "champion_absent_max": 5}),
    "tf80": ("a connected trade costs the trader a fifth of its farm (trade_farm_factor 0.8; 0.75 is not an exact "
             "ledger multiple with the zone factors)", {"trade_farm_factor": 0.8}),
    "tf100": ("a connected trade costs no farm (trade_farm_factor 1.0)", {"trade_farm_factor": 1.0}),
    "ot80": ("DIAGNOSTIC: kills in the own tower cell collected at 0.8", {"collect_contest_own_tower": 0.8}),
    "ot90": ("DIAGNOSTIC: kills in the own tower cell collected at 0.9 (as in the lane)",
             {"collect_contest_own_tower": 0.9}),
    "pg0": ("DIAGNOSTIC: no plate gold", {"plate_gold": 0.0}),
    "rc3": ("a Recall keeps the champion away 3 turns instead of 2 (recall_turns 3)", {"recall_turns": 3}),
    # ---- combinations of the two levers that moved an equilibrium in isolation (the own-tower collection rate and the
    #      trader's farm cost), with the death and Recall costs that can only matter once threats produce absences
    "ot80_tf100": ("own-tower collection 0.8 + trading costs no farm",
                   {"collect_contest_own_tower": 0.8, "trade_farm_factor": 1.0}),
    "ot80_tf80": ("own-tower collection 0.8 + trading costs a fifth of the farm",
                  {"collect_contest_own_tower": 0.8, "trade_farm_factor": 0.8}),
    "ot90_tf100": ("own-tower collection 0.9 + trading costs no farm",
                   {"collect_contest_own_tower": 0.9, "trade_farm_factor": 1.0}),
    "ot80_tf100_k2": ("ot80_tf100 + kill bounty 2.0",
                      {"collect_contest_own_tower": 0.8, "trade_farm_factor": 1.0, "kill_bounty": 2.0}),
    "ot80_tf100_dS1": ("ot80_tf100 + level-scaled death absence 3 / 4 / 5",
                       {"collect_contest_own_tower": 0.8, "trade_farm_factor": 1.0,
                        "death_turns_by_level": _death_schedule("S1"), "champion_absent_max": 5}),
    "ot80_tf100_rc3": ("ot80_tf100 + a 3-turn Recall",
                       {"collect_contest_own_tower": 0.8, "trade_farm_factor": 1.0, "recall_turns": 3}),
    # ---- the League-shaped death economy (a kill worth two waves, a respawn that grows with level) on both trade costs
    "ot80_tf80_k2": ("ot80_tf80 + kill bounty 2.0",
                     {"collect_contest_own_tower": 0.8, "trade_farm_factor": 0.8, "kill_bounty": 2.0}),
    "ot80_tf80_dS1": ("ot80_tf80 + level-scaled death absence 3 / 4 / 5",
                      {"collect_contest_own_tower": 0.8, "trade_farm_factor": 0.8,
                       "death_turns_by_level": _death_schedule("S1"), "champion_absent_max": 5}),
    "ot80_tf80_k2_dS1": ("ot80_tf80 + kill bounty 2.0 + level-scaled death absence",
                         {"collect_contest_own_tower": 0.8, "trade_farm_factor": 0.8, "kill_bounty": 2.0,
                          "death_turns_by_level": _death_schedule("S1"), "champion_absent_max": 5}),
    "ot80_tf100_k2_dS1": ("ot80_tf100 + kill bounty 2.0 + level-scaled death absence",
                          {"collect_contest_own_tower": 0.8, "trade_farm_factor": 1.0, "kill_bounty": 2.0,
                           "death_turns_by_level": _death_schedule("S1"), "champion_absent_max": 5}),
    # the selected candidate: GameConfig.m4_four_actions_econ() (= ot80_tf80_k2_dS1, the same configuration)
    "candidate": ("the calibrated candidate GameConfig.m4_four_actions_econ() (= ot80_tf80_k2_dS1)",
                  {"collect_contest_own_tower": 0.8, "trade_farm_factor": 0.8, "kill_bounty": 2.0,
                   "death_turns_by_level": _death_schedule("S1"), "champion_absent_max": 5}),
    "E1": ("terminal at the last plate threshold of either tower (terminal_rule final_plate)",
           {"terminal_rule": "final_plate"}),
}


def variant_cfg(name: str) -> GameConfig:
    cfg = GameConfig.m4_four_actions(**VARIANTS[name][1])
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------------------------- one cell
CELL_KEYS = ("earned", "farm", "plate", "bounty", "farm_avail", "xp", "xp_avail", "xp_missed", "hp_lost", "dmg_trade",
             "dmg_gank", "dmg_tower", "trade_att", "trade_conn", "ganked", "died", "recalled", "away", "invested",
             "crashes", "crash_hp", "tower", "plates_n", "held_by", "longest_hold", "applied")


def _cell_job(args) -> tuple:
    vname, sc, a, b, n, turns = args[:6]
    root = args[6] if len(args) > 6 else ECON_SEED
    cfg = variant_cfg(vname)
    base, gap, level, hb = MATRIX_SCENARIOS[sc][:4]
    stt = h.start(base, cfg, gap=gap, level=level, hb=hb)
    sa, sb = st.LIBRARY[a], st.LIBRARY[b]
    games = h.cell(cfg, stt, lambda: (st.StrategyAgent(sa), st.StrategyAgent(sb)), root, n, turns,
                   windows=SUB)
    out = {"G": [g["all"]["G"] for g in games], "X": [g["all"]["X"] for g in games],
           "G40": [g["40"]["earned"][0] - g["40"]["earned"][1] for g in games],
           "X40": [g["40"]["xp"][0] - g["40"]["xp"][1] for g in games],
           "G20": [g["20"]["earned"][0] - g["20"]["earned"][1] for g in games]}
    for key in ("all", "40"):
        ag = h.agg(games, key)
        out[f"agg{key}"] = {k: v for k, v in ag.items()
                            if k.rsplit("_", 1)[0] in CELL_KEYS + ("invested_end", "ap_end", "capacity_end",
                                                                  "duel_ratio_end", "xp_pending_change", "level_end",
                                                                  "hp_end", "first_crash")
                            or k in ("duel_ratio_end", "contact_mean", "n")}
    return f"{sc}|{a}|{b}", out


# ---------------------------------------------------------------------------------------------- matrix games
def _solve(cells: dict, sc: str, objective: str) -> dict:
    from ...matrix_game import solve_matrix_game
    As, Bs = MATRIX_SCENARIOS[sc][4], MATRIX_SCENARIOS[sc][5]

    def val(a, b):
        c = cells[f"{sc}|{a}|{b}"]
        if objective == "J":
            v = c["G"]
        elif objective == "Jx":
            v = [g + LAMBDA * x for g, x in zip(c["G"], c["X"])]
        elif objective == "G40":
            v = c["G40"]
        elif objective == "G40x":
            v = [g + LAMBDA * x for g, x in zip(c["G40"], c["X40"])]
        else:
            raise ValueError(objective)
        return float(np.mean(v))

    M = np.array([[val(a, b) for b in Bs] for a in As])
    sol = solve_matrix_game(M)
    x, y = np.asarray(sol.x), np.asarray(sol.y)
    b_br = {a: Bs[int(np.argmin(M[i]))] for i, a in enumerate(As)}
    a_br = {b: As[int(np.argmax(M[:, j]))] for j, b in enumerate(Bs)}
    return {"value": float(sol.value), "gap": float(sol.gap), "matrix": M.round(3).tolist(), "A": list(As),
            "B": list(Bs),
            "x": {a: round(float(p), 4) for a, p in zip(As, x) if p > 1e-6},
            "y": {b: round(float(p), 4) for b, p in zip(Bs, y) if p > 1e-6},
            "B_best_reply": b_br, "A_best_reply": a_br,
            "B_concede": round(float(sum(p for b, p in zip(Bs, y) if b in CONCEDE)), 4),
            "B_conditional": round(float(sum(p for b, p in zip(Bs, y) if b in CONDITIONAL)), 4),
            "A_chase": round(float(sum(p for a, p in zip(As, x) if a in CHASE)), 4),
            "B_chase": round(float(sum(p for b, p in zip(Bs, y) if b in CHASE)), 4)}


def run(variants: Sequence[str] = ("base",), workers: int = 9, episodes: int = EPISODES, turns: int = TURNS,
        scenarios: Optional[Sequence[str]] = None, merge: bool = True, seed_root: int = ECON_SEED,
        tag: str = "") -> Dict[str, dict]:
    """Play and solve the scenarios for each variant; with `merge`, scenarios already stored in the variant's result
    file are kept (and re-played only if listed explicitly in `scenarios`). `seed_root` / `tag` (robustness reruns):
    another seed root, stored as economics_<variant><tag>."""
    from ..rl.m43b.common import run_jobs
    out = {}
    for v in variants:
        t0 = time.perf_counter()
        old = load(f"economics_{v}{tag}") if merge else None
        todo = list(scenarios) if scenarios else [k for k in MATRIX_SCENARIOS
                                                  if not (old and k in old.get("games", {}))]
        jobs = [(v, sc, a, b, episodes, turns, seed_root) for sc in todo
                for a in MATRIX_SCENARIOS[sc][4] for b in MATRIX_SCENARIOS[sc][5]]
        cells = dict(old["cells"]) if old else {}
        cells.update(dict(run_jobs(_cell_job, jobs, workers, f"economics {v}")))
        games = dict(old["games"]) if old else {}
        games.update({sc: {obj: _solve(cells, sc, obj) for obj in ("J", "Jx", "G40", "G40x")} for sc in todo})
        allsc = [k for k in MATRIX_SCENARIOS if k in games]
        res = {"meta": meta(stage="economics", variant=v, description=VARIANTS[v][0], overrides=VARIANTS[v][1],
                            preset="GameConfig.m4_four_actions()", episodes=episodes, turns=turns,
                            seed_root=seed_root, scenarios={k: [*MATRIX_SCENARIOS[k][:4]] for k in allsc},
                            lam=LAMBDA),
               "cells": cells, "games": games}
        res["meta"]["elapsed_s"] = time.perf_counter() - t0 + (old["meta"].get("elapsed_s", 0.0) if old else 0.0)
        res["meta"]["overrides"] = {k: list(val) if isinstance(val, tuple) else val
                                    for k, val in VARIANTS[v][1].items()}
        save_json(f"economics_{v}{tag}", res)
        out[v] = res
    return out


def loaded(v: str) -> Optional[dict]:
    return load(f"economics_{v}")


__all__ = ["run", "loaded", "VARIANTS", "MATRIX_SCENARIOS", "variant_cfg", "FREEZE_A", "FREEZE_B", "NEUTRAL",
           "CONCEDE", "CHASE"]
