"""M4.2d-D: continuous investment and observation readiness (docs/m4/m4_investment_readiness_report.md). Mostly analytic.
`GameConfig.m4()` (two tiers) and the M4.2c `GameConfig.m4_continuous()` defaults are NOT changed; the candidate is
the opt-in `GameConfig.m4_continuous(**READY)`.

    python -m lanegame.experiments.m4.investment_readiness --stage trajectory   gold accumulation over 1,000 turns
                                                                                (scripted play; the only simulation)
    python -m lanegame.experiments.m4.investment_readiness --stage curves       candidate curves: power, breakpoints,
                                                                                duels, level gaps (analytic, 0 turns)
    python -m lanegame.experiments.m4.investment_readiness --stage observation  observation encodings (analytic)

Budget (assignment): target < 5,000 simulated turns, hard cap 15,000; the trajectory stage charges
results/m4_investment_readiness/budget.json and refuses to run past the cap.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Callable, Dict, List

import numpy as np

from ...config import GameConfig
from ...env import LaneEnv, play_episode
from ...env.observation_m4 import compressed_exact_max
from ...io import dump_json
from ...progression import investment, tables
from ..common import RESULTS_DIR, md_table
from . import economy
from .common import git_commit
from .investment import _duel, _trade_dist

OUT = os.path.join(RESULTS_DIR, "m4_investment_readiness")
BUDGET_TURNS = 15_000
MU = GameConfig.m4().gold_scale

# the primary recommendation (opt-in): bounded sqrt-saturating curve and the compressed observation entries
READY = dict(invest_curve="saturating", invest_gain=1.5, invest_scale=16.0, invest_obs_encoding="compressed",
             invest_obs_ref=64, invest_obs_gold_ref=8)

MATCHUPS = (("edge_trader", "edge_trader"), ("aggressive_trader", "hold_farmer"), ("gold_recaller", "gold_recaller"))
CHECKPOINTS = (50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
S_GRID = (0, 1, 4, 7, 10, 20, 50, 100, 200, 300, 500, 1000)
LEVELS = (1, 6, 11, 18)


def ready(**kw) -> GameConfig:
    return GameConfig.m4_continuous(**{**READY, **kw})


# ---------------------------------------------------------------------- budget ledger
def _ledger_path() -> str:
    return os.path.join(OUT, "budget.json")


def ledger() -> dict:
    try:
        with open(_ledger_path(), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"stages": {}, "total": {"turns": 0, "wall_s": 0.0}}


def charge(stage: str, turns: int, wall: float) -> dict:
    d = ledger()
    d["stages"][stage] = {"turns": int(turns), "wall_s": round(wall, 2)}
    d["total"] = {"turns": int(sum(v["turns"] for v in d["stages"].values())),
                  "wall_s": round(sum(v["wall_s"] for v in d["stages"].values()), 2)}
    os.makedirs(OUT, exist_ok=True)
    dump_json(d, _ledger_path())
    return d


def save(name: str, data: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    dump_json({"commit": git_commit(), **data}, os.path.join(OUT, name + ".json"))


# ---------------------------------------------------------------------- stage trajectory (simulation)
def trajectory(cfg: GameConfig, matchups=MATCHUPS, turns: int = 1000, seed: int = 61_000) -> dict:
    """Cumulative earned / invested / unspent gold and levels at the checkpoints, one game per matchup."""
    ag = economy.agents()
    env = LaneEnv(cfg, max_turns=turns, cache_size=20_000)
    t = tables(cfg)
    inv = investment(cfg)
    out, played = {}, 0
    for i, (x, y) in enumerate(matchups):
        log = play_episode(env, ag[x], ag[y], seed + i, record=True)
        played += len(log.turns)
        earned = [0, 0]
        rows, events, max_gold, deaths = [], [[], []], [0, 0], [0, 0]
        level_gap, inv_ratio = [], []
        for k, info in enumerate(log.turns, start=1):
            for s in (0, 1):
                earned[s] += info["earned_units"][s]
                max_gold[s] = max(max_gold[s], info["gold"][s])
                if info["invested"][s]:
                    events[s].append(info["invested"][s] / MU)
                deaths[s] += bool(info["died"][s])
            la, lb = info["levels"]
            level_gap.append(abs(la - lb))
            ma, mb = inv.factor(info["items"][0]), inv.factor(info["items"][1])
            inv_ratio.append(max(ma, mb) / min(ma, mb))
            if k in CHECKPOINTS:
                rows.append({"turn": k, "earned": [e / MU for e in earned],
                             "invested": [u / MU for u in info["items"]], "unspent": [g / MU for g in info["gold"]],
                             "level": list(info["levels"]), "M": [ma, mb]})
        out[f"{x}-{y}"] = {"checkpoints": rows, "deaths": deaths, "max_unspent": [g / MU for g in max_gold],
                           "investment_events": [len(e) for e in events],
                           "mean_investment": [float(np.mean(e)) if e else 0.0 for e in events],
                           "max_investment": [max(e) if e else 0.0 for e in events],
                           "level_gap": {"mean": float(np.mean(level_gap)), "max": int(max(level_gap)),
                                         "share_ge_2": float(np.mean(np.array(level_gap) >= 2))},
                           "M_ratio": {"mean": float(np.mean(inv_ratio)), "max": float(max(inv_ratio))}}
    return {"turns": played, "games": out}


def stage_trajectory(which: str) -> dict:
    used = ledger()["total"]["turns"]
    if used + 1000 * len(MATCHUPS) > BUDGET_TURNS:
        raise SystemExit(f"budget: {used} turns used, cap {BUDGET_TURNS}")
    cfg = GameConfig.m4_continuous() if which == "power" else ready()
    matchups = MATCHUPS if which == "power" else MATCHUPS[:1]
    t0 = time.perf_counter()
    res = trajectory(cfg, matchups)
    charge(f"trajectory_{which}", res["turns"], time.perf_counter() - t0)
    return {"curve": which, "cfg": cfg.to_dict(), **res}


# ---------------------------------------------------------------------- stage curves (analytic)
class Curve:
    """A candidate M(S) evaluated like the implementation: M in fixed point 10^12, AP* = AP(L, 0) M, hit damage
    floor((D 100 AP* + 5000 one) / (10^4 one)). Implemented curves delegate to `Investment` (exact)."""

    def __init__(self, name: str, f: Callable[[float], float] = None, cfg: GameConfig = None, note: str = "",
                 bound: float = None):
        self.name, self.note, self.bound = name, note, bound
        self.t = tables(GameConfig.m4_continuous())
        self.inv = investment(cfg) if cfg is not None else None
        self.f = f if f is not None else (lambda S: self.inv.factor(int(round(S * MU))))
        self._m: Dict[int, int] = {}

    def M(self, S: float) -> float:
        return self.f(S)

    def mult(self, u: int) -> int:
        """M(u) x 10^12 (floored for the exact curves)."""
        if self.inv is not None:
            return self.inv.multiplier(u) * 10 ** 12 // self.inv.one
        m = self._m.get(u)
        if m is None:
            m = self._m[u] = 10 ** 12 + int(round(10 ** 12 * (self.f(u / MU) - 1.0)))
        return m

    def hit(self, D: int, L: int, u: int) -> int:
        if self.inv is not None:
            return self.inv.hit_damage(D, L, u)
        return (D * 100 * self.t.ap[L][0] * self.mult(u) + 5000 * 10 ** 12) // 10 ** 16


def _lomax(A: float, K: float, q: float):
    return lambda S: 1 + A * (1 - (1 + S / K) ** (-q))


def candidates() -> List[Curve]:
    return [
        Curve("power p0.9 a0.045 (M4.2c)", cfg=GameConfig.m4_continuous(), note="current default, unbounded"),
        Curve("power p0.7 a0.079", lambda S: 1 + 0.079 * ((1 + S) ** 0.7 - 1), note="M4.2c shapes, matched at S=10"),
        Curve("power p0.5 a0.149", lambda S: 1 + 0.149 * ((1 + S) ** 0.5 - 1), note="M4.2c shapes, matched at S=10"),
        Curve("power p0.5 a0.073", lambda S: 1 + 0.073 * ((1 + S) ** 0.5 - 1), note="weak early: M(300) = 2.2"),
        Curve("log a0.144", lambda S: 1 + 0.144 * math.log1p(S), note="matched at S=10; unbounded (M=3 at 1.1e6)"),
        Curve("hyperbolic A1.5 K33.6", _lomax(1.5, 33.6, 1.0), note="1 + A S/(S+K), bounded 2.5", bound=2.5),
        Curve("sqrt-sat A1.25 K12", cfg=ready(invest_gain=1.25, invest_scale=12.0), note="bounded 2.25", bound=2.25),
        Curve("sqrt-sat A1.5 K16 (primary)", cfg=ready(), note="bounded 2.5", bound=2.5),
        Curve("sqrt-sat A1.75 K24", cfg=ready(invest_gain=1.75, invest_scale=24.0), note="bounded 2.75", bound=2.75),
    ]


def _first_S(pred, hi_res: float = 10 ** 6):
    """Smallest investment (resources, to the ledger unit) with pred(u) true (pred monotone in u), or None."""
    lo, hi = 0, int(hi_res * MU)
    if not pred(hi):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo / MU


def hits_to_kill(c: Curve, D: int, L_att: int, L_vic: int, u: int) -> int:
    return math.ceil(c.t.hp_max[L_vic] / c.hit(D, L_att, u))


def _steps(c: Curve, D: int, L: int, max_res: float = 1000.0) -> List[dict]:
    """Every hits-to-kill step of a D-base-HP hit against a same-level full-HP champion for S in [0, max_res]: the
    S at which it happens and how much more investment moves that hit's damage by 1 centi-HP there."""
    H = c.t.hp_max[L]
    n0 = hits_to_kill(c, D, L, L, 0)
    out = []
    for n in range(n0 - 1, 0, -1):
        need = math.ceil(H / n)
        S = _first_S(lambda u: c.hit(D, L, u) >= need, max_res)
        if S is None:
            break
        u = int(round(S * MU))
        d = c.hit(D, L, u)
        nxt = _first_S(lambda v: c.hit(D, L, u + v) > d, max_res)
        out.append({"hits": n, "S": S, "S_for_next_centi_hp": nxt})
    return out


PAIRS = ((0, 4), (4, 10), (10, 20), (20, 40), (30, 60), (50, 100), (100, 150), (150, 300), (200, 300), (300, 500))


def stage_curves() -> dict:
    cs = candidates()
    duel_cfg = GameConfig.m4_continuous()
    res = {"S_grid": S_GRID, "curves": {}}
    for c in cs:
        d = {"note": c.note, "bound": c.bound,
             "M": {S: c.M(S) for S in S_GRID},
             "gain_of_4_resources_pct": {S: 100 * (c.M(S + 4) / c.M(S) - 1) for S in S_GRID},
             "share_of_M300_after_S100": (c.M(300) - c.M(100)) / (c.M(300) - 1),
             "S_M_reaches": {m: _first_S(lambda u, m=m: c.mult(u) >= int(m * 10 ** 12)) for m in (1.2, 1.5, 2.0, 2.5, 3.0)},
             "steps": {f"L{L}_D{D}": _steps(c, D, L) for L in LEVELS for D in (1, 2)},
             "one_shot_2hp_S": {f"L{L}_gap{g}": _first_S(lambda u, L=L, g=g: hits_to_kill(c, 2, min(L + g, 18), L, u) <= 1)
                                for L in (1, 6, 11, 14) for g in (0, 2, 4)},
             "one_shot_2hp_S_L18": _first_S(lambda u: hits_to_kill(c, 2, 18, 18, u) <= 1),
             "two_shot_1hp_S_L18": _first_S(lambda u: hits_to_kill(c, 1, 18, 18, u) <= 2)}
        # relative advantage and duel odds at level 18 (levels are capped from about turn 200 on in scripted play)
        rows = []
        H = c.t.hp_max[18]
        for sb, sa in PAIRS:
            ua, ub = int(sa * MU), int(sb * MU)
            dA, _, _ = _trade_dist(duel_cfg, 0, c.hit, 18, ua)
            dB, _, _ = _trade_dist(duel_cfg, 1, c.hit, 18, ub)
            w, l, dr = _duel(H, H, dA, dB)
            rows.append({"S_A": sa, "S_B": sb, "M_ratio": c.M(sa) / c.M(sb),
                         "hits_1hp": [hits_to_kill(c, 1, 18, 18, ua), hits_to_kill(c, 1, 18, 18, ub)],
                         "hits_2hp": [hits_to_kill(c, 2, 18, 18, ua), hits_to_kill(c, 2, 18, 18, ub)],
                         "duel_A_first": w, "duel_B_first": l, "same_turn": dr})
        d["pairs_L18"] = rows
        res["curves"][c.name] = d
    # level-gap interaction, primary curve: 1-HP / 2-HP hits to kill a full-HP victim at level L by a champion g
    # levels higher with investment S (HP and attack power both grow 1.05^L, so only the gap matters)
    p = cs[[c.name for c in cs].index("sqrt-sat A1.5 K16 (primary)")]
    res["level_gap_primary"] = [{"victim_L": L, "gap": g, "S": S,
                                 "hits_1hp": hits_to_kill(p, 1, L + g, L, int(S * MU)),
                                 "hits_2hp": hits_to_kill(p, 2, L + g, L, int(S * MU))}
                                for L in (6, 11, 14) for g in (0, 2, 4) for S in (0, 20, 60, 300, 10 ** 5)]
    return res


# ---------------------------------------------------------------------- stage observation (analytic)
def stage_observation() -> dict:
    out = {}
    for ref in (8, 16, 64, 1024):
        N = compressed_exact_max(ref)
        n = np.arange(0, int(1.6 * N) + 2, dtype=np.int64)
        x = (n / (n + ref)).astype(np.float32).astype(np.float64)
        dec = np.rint(ref * x / (1 - x)).astype(np.int64)
        bad = np.nonzero(dec != n)[0]
        out[f"compressed_R{ref}"] = {
            "exact_max": N, "first_decoding_failure": int(bad[0]) if len(bad) else None,
            "max_abs_error_up_to_exact_max": float(np.max(np.abs(ref * x[:N + 1] / (1 - x[:N + 1]) - n[:N + 1]))),
            "entry_at": {k: float(np.float32(k / (k + ref))) for k in (1, 4, 8, 16, 64, 300, 1000, 10 ** 6)},
            "step_of_one_resource_at": {k: ref / ((k + ref) * (k + 1 + ref)) for k in (0, 4, 16, 64, 300)},
            "first_n_with_entry_1.0": int(math.ceil(ref * (2 ** 25 - 1)))}
    out["linear_R1024"] = {"exact_max": 1023, "entry_at": {k: min(k, 1024) / 1024 for k in (1, 4, 8, 16, 64, 300, 1000)},
                           "step_of_one_resource": 1 / 1024}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("trajectory", "curves", "observation"))
    ap.add_argument("--curve", default="power", choices=("power", "ready"))
    a = ap.parse_args(argv)
    if a.stage == "trajectory":
        save(f"trajectory_{a.curve}", stage_trajectory(a.curve))
        print(json.dumps(ledger()["total"]))
    elif a.stage == "curves":
        save("curves", stage_curves())
    else:
        save("observation", stage_observation())


if __name__ == "__main__":
    main()
