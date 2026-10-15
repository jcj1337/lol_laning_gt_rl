"""Priority A: strong-side freezing, zoning and Recall economics.

States (`states.freeze_state`, INITIALIZED): A holds B's front of `held` HP at A's -1 with its own 8-HP front in its
tower cell, A at its tower, B walked up to its +1; the DEV grid is level {4, 10} x power gap {equal, large (A x1.76)}
x held {5, 12, 20} x A's HP {1.0, 0.6}.

1. `grid` (policy values): every freezer strategy (FREEZER_SET: freeze-phase rule x what follows; plus the two
   non-freezing references g_shove and shove_recall) against every frozen-on response (RESPONSE_SET: the frozen-phase
   rule x a Last Hit or Shove default), on the same DEV seeds (paired), 100 turns. Payoffs G20 / G50 / G100 (A's
   discounted lead). Per state: the restricted game's value and supports; the FREEZE PREMIUM = value of the whole game
   minus the value of the game whose rows are only the non-freezing references (>= 0; > 0 when some freezer row
   adds value against B's best response), with a seed bootstrap; paired contrasts (guard - passive, guard - Shove) per
   response; and the mechanism ledger per cell (freeze duration, deaths, B's XP and farm at 20 turns, minion fire on A,
   trades).
2. `local` (local action values): from each state, A's first action is forced (Thin + Hold, Thin + Trade, Last Hit +
   Hold, Shove + Hold, Shove + Trade), then A continues with `fz_guard_sh`; B plays a response throughout. Paired
   G5 / G20 / G100.
3. `reset` (the decision after the opponent resets): states collected from fz_guard_sh against wk_shove_reset_sh games
   at the turn B returns to the lane (back at full HP after a Recall or death) while A still holds the freeze and has
   at most 75% HP; A's plans from there: keep guarding (fz_guard_sh), keep passively (fz_passive_sh), Shove now and
   Recall when crashed (shove_recall), Recall now (then fz_guard_sh). Paired G20 / G50 / G100.
4. `sweep`: the grid's key cells under trade_wave_fraction, minion_damage_alpha and minion_retaliation_delay.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import numpy as np

from ...config import B_HOLD, B_TRADE, W_HOLD, W_SHOVE, W_THIN, champ_act
from ...matrix_game import solve_matrix_game
from . import common as Cm
from . import engine as E
from . import states as S
from . import strategies as ST

FREEZER_SET = ("fz_passive", "fz_passive_sh", "fz_guard", "fz_guard_sh", "fz_guard_sh_sr", "fz_guard_line_sh",
               "fz_bully_sh", "fz_guard_aw", "g_shove", "shove_recall")
NONFREEZE = ("g_shove", "shove_recall")
RESPONSE_SET = ("wk_shove", "wk_shove_safe", "wk_lh", "wk_back", "wk_yield", "wk_contest", "wk_wary_shove",
                "wk_shove_reset", "wk_shove_sh", "wk_shove_safe_sh", "wk_lh_sh", "wk_back_sh", "wk_yield_sh",
                "wk_contest_sh", "wk_wary_shove_sh", "wk_shove_reset_sh")
GRID = [(L, g, h, hp) for L in (4, 10) for g in ("equal", "large") for h in (5, 12, 20) for hp in (1.0, 0.6)]
METRICS = ("G20", "G50", "G100")
LEDGER = ("held_run0", "held_A", "deaths_A", "deaths_B", "recalls_A", "recalls_B", "deaths_B@20", "xp_A@20", "xp_B@20",
          "farm_A@20", "farm_B@20", "d_min_A@20", "trade_dealt_A@20", "d_total_A@20", "d_total_B@20", "away_B@20",
          "farm_A", "farm_B", "d_min_A", "trade_dealt_A", "plates_A", "plates_B", "crashes_A", "crashes_B", "lead")


def _cell(args):
    over, state_key, a, b, seeds, turns = args
    c = Cm.cfg("r2", **dict(over))
    L, g, h, hp = state_key
    start = S.freeze_state(c, L, g, h, 8, (hp, 1.0))
    rows = [E.play(c, start, ST.get(a), ST.get(b), sd, turns)["summary"] for sd in seeds]
    out = {"a": a, "b": b, "per_seed": {m: [r[m] for r in rows] for m in METRICS}}
    for k in METRICS + LEDGER:
        out[k] = float(np.mean([r.get(k, np.nan) for r in rows]))
    return out


def premium(P: Dict[str, np.ndarray], rows: Sequence[str], cols: Sequence[str], metric: str, boots: int = 300,
            rng_seed: int = 0) -> dict:
    """Value of the game minus the value of the non-freeze rows, with a seed bootstrap (pairing kept)."""
    rng = np.random.default_rng(rng_seed)
    X = P[metric]                                            # (rows, cols, seeds)
    nf = [i for i, r in enumerate(rows) if r in NONFREEZE]
    def val(M):
        full = solve_matrix_game(M).value
        sub = solve_matrix_game(M[nf]).value
        return full, sub
    full, sub = val(X.mean(axis=2))
    bs = []
    m = X.shape[2]
    for _ in range(boots):
        idx = rng.integers(0, m, m)
        f, s_ = val(X[:, :, idx].mean(axis=2))
        bs.append(f - s_)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"value": full, "value_nonfreeze": sub, "premium": full - sub, "premium_ci": [float(lo), float(hi)],
            "premium_boot_mean": float(np.mean(bs)), "premium_split": split_premium(X, nf)}


def split_premium(X: np.ndarray, nf: Sequence[int], reps: int = 50, rng_seed: int = 1) -> List[float]:
    """An honest (selection-free) premium: A's equilibrium mixture of the whole game and of the non-freeze rows are
    chosen on a random half of the seeds and evaluated on the other half against B's best reply there (the minimum
    over columns); the difference, averaged over `reps` random splits, as [mean, sd over splits]. The in-sample
    premium is biased upward (a maximum over noisy rows); this one is not, but it is conservative (the mixture is
    fitted on half the data)."""
    rng = np.random.default_rng(rng_seed)
    m = X.shape[2]
    out = []
    for _ in range(reps):
        perm = rng.permutation(m)
        a, b = perm[: m // 2], perm[m // 2:]
        for fit, ev in ((a, b), (b, a)):
            Mf, Me = X[:, :, fit].mean(axis=2), X[:, :, ev].mean(axis=2)
            x = solve_matrix_game(Mf).x
            xs = np.zeros(len(x))
            xs[list(nf)] = solve_matrix_game(Mf[list(nf)]).x
            out.append(float((x @ Me).min() - (xs @ Me).min()))
    return [float(np.mean(out)), float(np.std(out))]


def analyse(cells: List[dict], rows: Sequence[str], cols: Sequence[str]) -> dict:
    n, m = len(rows), len(cols)
    idx = {(c["a"], c["b"]): c for c in cells}
    P = {met: np.array([[idx[(a, b)]["per_seed"][met] for b in cols] for a in rows]) for met in METRICS}
    out = {}
    for met in METRICS:
        M = P[met].mean(axis=2)
        sol = solve_matrix_game(M)
        out[met] = {"value": float(sol.value),
                    "A": {rows[i]: float(x) for i, x in enumerate(sol.x) if x > 1e-4},
                    "B": {cols[j]: float(y) for j, y in enumerate(sol.y) if y > 1e-4},
                    "B_best_reply": {rows[i]: cols[int(np.argmin(M[i]))] for i in range(n)},
                    "row_worst": {rows[i]: float(M[i].min()) for i in range(n)},
                    "premium": premium(P, rows, cols, met)}
    # paired contrasts per response: does guarding pay (guard - passive), does freezing pay (guard - Shove)
    con = {}
    for x, y in (("fz_guard_sh", "fz_passive_sh"), ("fz_guard_sh", "g_shove"), ("fz_guard", "fz_passive"),
                 ("fz_guard_sh_sr", "fz_guard_sh"), ("fz_bully_sh", "fz_guard_sh"), ("fz_guard_line_sh", "fz_guard_sh")):
        if x in rows and y in rows:
            i, j = rows.index(x), rows.index(y)
            con[f"{x} - {y}"] = {cols[k]: Cm.paired(P["G100"][i, k], P["G100"][j, k]) for k in range(m)}
    out["contrasts_G100"] = con
    # B's responses: walk-up Shove vs step back vs Shove + Safe against the guard (B's view: minus A's payoff)
    i = rows.index("fz_guard_sh")
    out["responses_vs_guard_G100"] = {cols[k]: Cm.ci(-P["G100"][i, k]) for k in range(m)}
    return out


def grid(over=(), states=GRID, rows=FREEZER_SET, cols=RESPONSE_SET, seeds=Cm.DEV[:40], turns: int = 100,
         workers: int = Cm.WORKERS) -> List[dict]:
    out = []
    jobs = [(tuple(over), st, a, b, tuple(seeds), turns) for st in states for a in rows for b in cols]
    res = Cm.run_jobs(_cell, jobs, workers, f"freeze grid {dict(over)}")
    per = len(rows) * len(cols)
    for k, st in enumerate(states):
        cells = res[k * per:(k + 1) * per]
        out.append({"state": {"level": st[0], "gap": st[1], "held": st[2], "hp": st[3]}, "overrides": dict(over),
                    "rows": list(rows), "cols": list(cols),
                    "cells": [{kk: v for kk, v in c.items() if kk != "per_seed"} for c in cells],
                    "analysis": analyse(cells, rows, cols)})
    return out


# ---------------------------------------------------------------------------------------------- local action values
FIRST = {"TN+Hold": champ_act(W_THIN, B_HOLD), "TN+Trade": champ_act(W_THIN, B_TRADE),
         "LH+Hold": champ_act(W_HOLD, B_HOLD), "SH+Hold": champ_act(W_SHOVE, B_HOLD),
         "SH+Trade": champ_act(W_SHOVE, B_TRADE)}


def _local(args):
    over, state_key, resp, cont, seeds, turns = args
    c = Cm.cfg("r2", **dict(over))
    L, g, h, hp = state_key
    start = S.freeze_state(c, L, g, h, 8, (hp, 1.0))
    after = ST.get(cont)
    plans = {k: E.Then(k, E.Fixed(k, code, after), after, 1) for k, code in FIRST.items()}
    res = {k: [E.play(c, start, p, ST.get(resp), sd, turns)["summary"] for sd in seeds] for k, p in plans.items()}
    out = {"state": list(state_key), "response": resp, "cont": cont}
    for met in ("G1", "G5", "G20", "G100"):
        out[met] = {k: Cm.paired([r[met] for r in res[k]], [r[met] for r in res["TN+Hold"]]) for k in FIRST}
    out["held_run0"] = {k: float(np.mean([r["held_run0"] for r in res[k]])) for k in FIRST}
    out["deaths_B@20"] = {k: float(np.mean([r["deaths_B@20"] for r in res[k]])) for k in FIRST}
    return out


def local(over=(), states=GRID, responses=("wk_shove_sh", "wk_lh_sh", "wk_back_sh", "wk_shove_reset_sh"),
          cont: str = "fz_guard_sh", seeds=Cm.DEV[:200], turns: int = 100, workers: int = Cm.WORKERS) -> List[dict]:
    jobs = [(tuple(over), st, r, cont, tuple(seeds), turns) for st in states for r in responses]
    return Cm.run_jobs(_local, jobs, workers, "freeze local")


# ---------------------------------------------------------------------------------------------- after a reset
RESET_PLANS = ("fz_guard_sh", "fz_passive_sh", "shove_recall", "recall_now")


def _returned(cfg, lane, m4, s) -> bool:
    """Decision-state predicate (seat s = the freezer): the frozen-on enemy is back at full HP at its tower this turn
    (it just returned), s holds the freeze and has at most 75% of its HP."""
    en = m4.side(1 - s)
    from ...champions_m4 import context
    t = context(cfg).t
    full = en.absent == 0 and en.hp == t.hp_max[t.level_by_xp[min(en.xp, t.xp_cap)]] and en.pos == -cfg.tower
    return full and ST.freezing(lane, s) and ST.hp_frac(cfg, m4, s) <= 0.75 and m4.side(s).absent == 0


def _reset_job(args):
    over, state_key, opp, seeds_collect, seeds, turns = args
    c = Cm.cfg("r2", **dict(over))
    L, g, h, hp = state_key
    start = S.freeze_state(c, L, g, h, 8, (hp, 1.0))
    found = E.collect(c, start, ST.get("fz_guard_sh"), ST.get(opp), seeds_collect, 60, _returned, per_game=1,
                      min_turn=1)
    found = [f for f in found if f["seat"] == 0]
    guard = ST.get("fz_guard_sh")
    from ...champions import champion_legal_actions
    from .strategies import RECALL

    class RecallNow:
        def decide(self, cfg, lane, m4, s, i):
            if i == 0 and RECALL in champion_legal_actions(lane, m4, s, cfg):
                return RECALL, "recall now"
            return guard.decide(cfg, lane, m4, s, i)
    plans = {"fz_guard_sh": guard, "fz_passive_sh": ST.get("fz_passive_sh"), "shove_recall": ST.get("shove_recall"),
             "recall_now": RecallNow()}
    per_state = []
    for f in found:
        res = {k: [E.play(c, f["start"], p, ST.get(opp), sd, turns)["summary"] for sd in seeds] for k, p in plans.items()}
        per_state.append({k: {met: float(np.mean([r[met] for r in res[k]])) for met in ("G20", "G50", "G100")}
                          for k in plans} | {"hp": ST.hp_frac(c, f["start"][1], 0), "turn": f["turn"]})
    return {"state": list(state_key), "opponent": opp, "n_states": len(per_state), "per_state": per_state}


def reset(over=(), states=GRID, opp: str = "wk_shove_reset_sh", n_collect: int = 60, seeds=Cm.DEV[200:240],
          turns: int = 100, workers: int = Cm.WORKERS) -> List[dict]:
    jobs = [(tuple(over), st, opp, tuple(Cm.DEV[:n_collect]), tuple(seeds), turns) for st in states]
    res = Cm.run_jobs(_reset_job, jobs, workers, "freeze reset")
    # across-state paired contrasts (states are the units)
    allst = [p for r in res for p in r["per_state"]]
    summary = {}
    for met in ("G20", "G50", "G100"):
        summary[met] = {k: Cm.ci([p[k][met] - p["fz_guard_sh"][met] for p in allst]) for k in RESET_PLANS}
    return {"cells": res, "summary": summary, "n_states": len(allst)}


# ---------------------------------------------------------------------------------------------- calibration sweep
SWEEP = {"trade_wave_fraction": (0.25, 0.5, 1.0), "minion_damage_alpha": (30, 50, 80),
         "minion_retaliation_delay": (0, 1)}
SWEEP_STATES = [(10, "large", 12, 1.0), (4, "large", 12, 1.0), (10, "equal", 12, 1.0), (4, "equal", 20, 1.0)]


def sweep(states=SWEEP_STATES, seeds=Cm.DEV[:40], turns: int = 100, workers: int = Cm.WORKERS) -> List[dict]:
    out = []
    base = {"trade_wave_fraction": 0.5, "minion_damage_alpha": 50, "minion_retaliation_delay": 1}
    for field, values in SWEEP.items():
        for v in values:
            if v == base[field]:
                continue
            out += grid(((field, v),), states, seeds=seeds, turns=turns, workers=workers)
    return out


__all__ = ["FREEZER_SET", "RESPONSE_SET", "GRID", "grid", "local", "reset", "sweep", "analyse", "premium", "FIRST"]
