"""M4.2c: continuous gold investment prototype diagnostics (docs/m4/m4_continuous_investment_report.md). Exploratory;
`GameConfig.m4()` (two tiers) is NOT changed and stays the baseline. The experiment is
`GameConfig.m4_continuous(invest_alpha=a)` with p = 0.90.

    python -m lanegame.experiments.m4.investment --stage power      investment / power table (analytic, 0 turns)
    python -m lanegame.experiments.m4.investment --stage combat     damage, hits-to-kill, duel odds, breakpoints
                                                                    (analytic, 0 turns)
    python -m lanegame.experiments.m4.investment --stage recall     Recall-now vs Hold-now, paired rollouts
    python -m lanegame.experiments.m4.investment --stage response   investment leads in scripted play, paired rollouts
    python -m lanegame.experiments.m4.investment --stage summary    summary.md from the stage files (no simulation)

Budget (assignment §5): every stage adds its environment turns, wall time and worker CPU time to
results/m4_continuous_investment/budget.json; a stage is refused once 500,000 turns or 180 s of simulation wall time
are used (target < 200,000 turns).

Pairing: every variant of a state uses the same rollout seeds (`diagnostics.rollout_seed(root, m, state key)`), so
variants see identical chance vectors turn by turn until their trajectories diverge. Continuations are scripted
agents: these are measurements of those policies, not equilibrium values.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import time
from typing import Dict, List

import numpy as np

from ...champions import trade_double_probability, trade_hit_probability
from ...champions_m4 import M4State, context, level
from ...config import B_HOLD, B_RECALL, B_TRADE, SP, GameConfig, champ_act
from ...diagnostics import Commit, ProbCache, Snapshot, rollout, rollout_seed
from ...diagnostics.branch import FollowPolicy
from ...env import LaneEnv, play_episode
from ...io import dump_json
from ...progression import investment, tables
from ...state import State
from ..common import RESULTS_DIR, md_table
from . import economy
from .common import Z95, git_commit, mean_se, pool_map

OUT = os.path.join(RESULTS_DIR, "m4_continuous_investment")
BUDGET_TURNS = 500_000
BUDGET_WALL_S = 180.0
PROCS = 8
ALPHAS = (0.03, 0.045, 0.06)
ALPHA0 = 0.045
RECALL = champ_act(SP, B_RECALL)
HOLD = champ_act(SP, B_HOLD)
MU = GameConfig.m4().gold_scale
LEVELS = (1, 6, 11, 18)
PAIRS = ((0, 1), (0, 4), (0, 7), (4, 7), (7, 14))
NEUTRAL = State(0, 4, (), 0, 0, 4, (), 0, 0, 0, 0, 0)        # both fronts at the lane cell, equal HP


def cont(alpha: float = ALPHA0) -> GameConfig:
    return GameConfig.m4_continuous(invest_alpha=alpha)


# ---------------------------------------------------------------------- budget ledger
def _ledger_path() -> str:
    return os.path.join(OUT, "budget.json")


def ledger() -> dict:
    try:
        with open(_ledger_path(), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"stages": {}, "total": {"turns": 0, "wall_s": 0.0, "cpu_s": 0.0}}


def check_budget(stage: str) -> None:
    t = ledger()["total"]
    if t["turns"] >= BUDGET_TURNS or t["wall_s"] >= BUDGET_WALL_S:
        raise SystemExit(f"budget exhausted ({t}); stage {stage} refused (assignment §5)")


def charge(stage: str, turns: int, wall: float, cpu: float, note: str = "") -> dict:
    d = ledger()
    d["stages"][stage] = {"turns": int(turns), "wall_s": round(wall, 2), "cpu_s": round(cpu, 2)}
    if note:
        d["stages"][stage]["note"] = note
    d["total"] = {"turns": int(sum(v["turns"] for v in d["stages"].values())),
                  "wall_s": round(sum(v["wall_s"] for v in d["stages"].values()), 2),
                  "cpu_s": round(sum(v["cpu_s"] for v in d["stages"].values()), 2)}
    os.makedirs(OUT, exist_ok=True)
    dump_json(d, _ledger_path())
    return d


def save(name: str, data: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    dump_json({"commit": git_commit(), **data}, os.path.join(OUT, name + ".json"))


# ---------------------------------------------------------------------- stage power (analytic)
S_GRID = (0, 1, 2, 4, 7, 10, 14, 20, 30, 50, 80, 120, 200, 400)


def stage_power() -> dict:
    rows = []
    for S in S_GRID:
        row = {"S": S}
        for a in ALPHAS:
            row[f"alpha_{a}"] = investment(cont(a)).factor(S * MU)
        rows.append(row)
    base = GameConfig.m4()
    t = tables(base)
    tiers = [{"tiers": N, "spent": sum(t.prices[:N]) / MU, "factor": t.ap[1][N] / t.ap[1][0]} for N in range(3)]
    marg = {f"alpha_{a}": {f"dAP/dS at S={S}": a * 0.9 * (1 + S) ** -0.1 for S in (0, 4, 10, 40, 120)}
            for a in ALPHAS}
    return {"p": 0.9, "rows": rows, "tiers": tiers, "marginal_per_resource": marg}


# ---------------------------------------------------------------------- stage combat (analytic)
def _duel(hA: int, hB: int, dA, dB) -> tuple:
    """(P(A kills B first), P(B kills A first), P(both die on the same turn)) when both trade every turn from HP
    (hA, hB); dX = ((prob, centi-HP dealt by X), ...)."""
    @functools.lru_cache(maxsize=None)
    def f(x, y):
        stay, win, lose, draw = 0.0, 0.0, 0.0, 0.0
        for pa, da in dA:
            for pb, db in dB:
                p = pa * pb
                if p == 0:
                    continue
                y2, x2 = y - da, x - db
                if da == 0 and db == 0:
                    stay += p
                elif y2 <= 0 and x2 <= 0:
                    draw += p
                elif y2 <= 0:
                    win += p
                elif x2 <= 0:
                    lose += p
                else:
                    w, l, d = f(x2, y2)
                    win, lose, draw = win + p * w, lose + p * l, draw + p * d
        k = 1.0 - stay
        return win / k, lose / k, draw / k
    return f(hA, hB)


def _trade_dist(cfg: GameConfig, s: int, hit_fn, L: int, S_units: int):
    h = trade_hit_probability(NEUTRAL, s, B_TRADE, False, cfg)
    q = trade_double_probability(s, cfg)
    return ((1 - h, 0), (h * (1 - q), hit_fn(1, L, S_units)), (h * q, hit_fn(2, L, S_units))), h, q


def _combat_row(cfg: GameConfig, hit_fn, L: int, sa: int, sb: int) -> dict:
    t = tables(cfg)
    H = t.hp_max[L]
    dA, h, q = _trade_dist(cfg, 0, hit_fn, L, sa)
    dB, _, _ = _trade_dist(cfg, 1, hit_fn, L, sb)
    ea = sum(p * d for p, d in dA)
    eb = sum(p * d for p, d in dB)
    w, l, d = _duel(H, H, dA, dB)
    return {"level": L, "H": H, "dmg1": [hit_fn(1, L, sa), hit_fn(1, L, sb)], "dmg2": [hit_fn(2, L, sa), hit_fn(2, L, sb)],
            "hits_to_kill_1": [math.ceil(H / hit_fn(1, L, sa)), math.ceil(H / hit_fn(1, L, sb))],
            "hits_to_kill_2": [math.ceil(H / hit_fn(2, L, sa)), math.ceil(H / hit_fn(2, L, sb))],
            "expected_trade_damage": [ea, eb], "exchange_ratio": ea / eb,
            "duel": {"A_first": w, "B_first": l, "same_turn": d}, "hit_p": h, "double_p": q}


def _breakpoints(cfg: GameConfig, L: int, D: int, max_res: int = 400) -> List[dict]:
    """Smallest investment (resources, to the ledger unit) at which a D-base-HP hit kills a same-level full-HP
    champion in one hit fewer, down to 3 fewer hits or max_res resources."""
    inv = investment(cfg)
    t = tables(cfg)
    H = t.hp_max[L]
    n0 = math.ceil(H / inv.hit_damage(D, L, 0))
    out = []
    for n in range(n0 - 1, max(0, n0 - 4), -1):
        need = math.ceil(H / n)
        lo, hi = 0, max_res * MU
        if inv.hit_damage(D, L, hi) < need:
            out.append({"hits": n, "S": None})
            continue
        while lo < hi:
            mid = (lo + hi) // 2
            if inv.hit_damage(D, L, mid) >= need:
                hi = mid
            else:
                lo = mid + 1
        out.append({"hits": n, "S": lo / MU})
    return out


def _lethality(cfg: GameConfig, L: int, D: int, k: int, max_res: int = 5000):
    """Smallest investment (resources) at which k hits of D base HP kill a same-level full-HP champion."""
    inv = investment(cfg)
    need = math.ceil(tables(cfg).hp_max[L] / k)
    lo, hi = 0, max_res * MU
    if inv.hit_damage(D, L, hi) < need:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if inv.hit_damage(D, L, mid) >= need:
            hi = mid
        else:
            lo = mid + 1
    return lo / MU


def stage_combat() -> dict:
    out = {"continuous": {}, "tiers": {}, "breakpoints": {}, "min_investment_changing_a_hit": {}, "lethality": {}}
    for a in ALPHAS:
        cfg = cont(a)
        inv = investment(cfg)
        rows = []
        for sa, sb in PAIRS:
            for L in LEVELS:
                # the HIGHER investment is A: rows read "A (sb) vs B (sa)"
                r = _combat_row(cfg, inv.hit_damage, L, sb * MU, sa * MU)
                r["S"] = [sb, sa]
                rows.append(r)
        out["continuous"][f"alpha_{a}"] = rows
        out["breakpoints"][f"alpha_{a}"] = {f"L{L}_D{D}": _breakpoints(cfg, L, D) for L in LEVELS for D in (1, 2)}
        mins = {}
        for L in LEVELS:
            base = inv.hit_damage(2, L, 0)
            lo, hi = 0, MU
            while lo < hi:
                mid = (lo + hi) // 2
                if inv.hit_damage(2, L, mid) > base:
                    hi = mid
                else:
                    lo = mid + 1
            mins[f"L{L}"] = lo / MU
        out["min_investment_changing_a_hit"][f"alpha_{a}"] = mins
        out["lethality"][f"alpha_{a}"] = {f"L{L}_D{D}_k{k}": _lethality(cfg, L, D, k)
                                          for L in (1, 11, 18) for D in (1, 2) for k in (3, 2, 1)}
    base = GameConfig.m4()
    t = tables(base)
    hit = lambda D, L, N: t.dmg[D][L][N]
    rows = []
    for na, nb in ((0, 0), (1, 0), (2, 0), (2, 1)):
        for L in LEVELS:
            r = _combat_row(base, hit, L, na, nb)
            r["tiers"] = [na, nb]
            rows.append(r)
    out["tiers"] = rows
    return out


# ---------------------------------------------------------------------- rollout jobs
def _set(snap: Snapshot, seat: int, **kw) -> Snapshot:
    return snap.with_side(seat, **kw)


def run_job(job) -> dict:
    """Paired rollouts of one (snapshot, plan, agents) variant with per-rollout counters."""
    cfg = GameConfig.from_dict(job["cfg"])
    snap = Snapshot.from_dict(job["snap"])
    ag = agents()
    pair = (ag[job["agents"][0]], ag[job["agents"][1]])
    env = LaneEnv(snap.config(cfg), cache_size=50_000)
    cache = ProbCache()
    plan = {"recall": Commit(RECALL, "recall"), "hold": Commit(HOLD, "hold"), "follow": FollowPolicy()}[job["plan"]]
    t0 = time.process_time()
    rows, turns = [], 0
    for m in range(job["n"]):
        r = rollout(env, snap, 0, plan, pair, rollout_seed(job["root"], m, *job["key"]), cache,
                    record=True, max_turns=job["window"])
        turns += r.turns
        c = {"ret": r.ret, "farm": list(r.farm), "deaths": list(r.deaths), "bounty": list(r.bounty),
             "recalls": list(r.recalls), "xp": r.m4["xp"], "invested": [x / MU for x in r.m4["invested"]],
             "invest_events": list(r.m4["purchases"]),
             "away": [0, 0], "trades": [0, 0], "trade_dmg_hp": [0.0, 0.0], "retreat_or_safe": [0, 0],
             "first_invest_turn": None}
        t = tables(cfg)
        for k, i in enumerate(r.infos):
            for s in (0, 1):
                c["away"][s] += int(i["away"][s])
                c["trades"][s] += int(i["trade_connected"][s])
                c["trade_dmg_hp"][s] += i["trade_damage"][s] / t.hp_max[i["levels_before"][1 - s]]
                c["retreat_or_safe"][s] += int(i["modes"][s] in ("retreat", "safe"))
            if c["first_invest_turn"] is None and (i["invested"][0] or i["purchases"][0]):
                c["first_invest_turn"] = k
        rows.append(c)
    return {"id": job["id"], "rows": rows, "turns": turns, "cpu": time.process_time() - t0}


def agents() -> Dict[str, object]:
    return economy.agents()


def _run(stage: str, jobs: List[dict]) -> List[dict]:
    check_budget(stage)
    t0 = time.perf_counter()
    res = pool_map(run_job, jobs, PROCS)
    wall = time.perf_counter() - t0
    charge(stage, sum(r["turns"] for r in res), wall, sum(r["cpu"] for r in res))
    return res


def _paired(a, b) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    m, se = mean_se(d)
    return {"diff": m, "ci95": Z95 * se, "p_positive": float(np.mean(d > 0)), "a": float(np.mean(a)),
            "b": float(np.mean(b))}


def _col(rows, key, s=None):
    return [r[key] if s is None else r[key][s] for r in rows]


# ---------------------------------------------------------------------- stage recall
def recall_snapshots() -> Dict[str, Snapshot]:
    """The witnessed recall-cheap (own wave pushing) and recall-expensive (enemy wave at own -1) states: reachable,
    Recall legal for A, both champions present."""
    return {"cheap": economy.witness_snapshot("6a_recall_cheap"),
            "expensive": economy.witness_snapshot("6b_recall_expensive")}


def stage_recall(n: int = 10, window: int = 120):
    snaps = recall_snapshots()
    jobs = []
    models = {"cont": (cont(), (0, 7)), "tiers": (GameConfig.m4(), (0,))}
    for model, (cfg, opp_levels) in models.items():
        t = tables(cfg)
        for si, (wname, s0) in enumerate(snaps.items()):
            L = level(s0.champ.a, t)
            H = t.hp_max[L]
            for hp6 in (3, 6):
                for gold in (1, 4, 7, 10):
                    for opp in opp_levels:
                        s = _set(s0, 0, hp=max(1, H * hp6 // 6), gold=gold * MU, items=0)
                        s = _set(s, 1, items=opp * MU if model == "cont" else 0, gold=0)
                        for plan in ("recall", "hold"):
                            jobs.append({"id": f"{model}|{wname}|hp{hp6}|g{gold}|opp{opp}|{plan}", "cfg": cfg.to_dict(),
                                         "snap": s.to_dict(), "plan": plan, "agents": ("edge_trader", "edge_trader"),
                                         "n": n, "window": window, "key": (si,), "root": 2_042})
    res = _run("recall", jobs)
    by = {r["id"]: r["rows"] for r in res}
    summary = {}
    for model, (cfg, opp_levels) in models.items():
        for wname in snaps:
            for hp6 in (3, 6):
                for opp in opp_levels:
                    for gold in (1, 4, 7, 10):
                        k = f"{model}|{wname}|hp{hp6}|g{gold}|opp{opp}"
                        rc, ho = by[f"{k}|recall"], by[f"{k}|hold"]
                        summary[k] = {
                            "recall_minus_hold": _paired(_col(rc, "ret"), _col(ho, "ret")),
                            "farm_A": _paired(_col(rc, "farm", 0), _col(ho, "farm", 0)),
                            "farm_B": _paired(_col(rc, "farm", 1), _col(ho, "farm", 1)),
                            "xp_A": _paired(_col(rc, "xp", 0), _col(ho, "xp", 0)),
                            "away_A": _paired(_col(rc, "away", 0), _col(ho, "away", 0)),
                            "deaths_A": _paired(_col(rc, "deaths", 0), _col(ho, "deaths", 0)),
                            "deaths_B": _paired(_col(rc, "deaths", 1), _col(ho, "deaths", 1)),
                            "trades_A": _paired(_col(rc, "trades", 0), _col(ho, "trades", 0)),
                            "trade_dmg_A": _paired(_col(rc, "trade_dmg_hp", 0), _col(ho, "trade_dmg_hp", 0)),
                            "trade_dmg_B": _paired(_col(rc, "trade_dmg_hp", 1), _col(ho, "trade_dmg_hp", 1)),
                            "first_invest_turn_hold": _first(ho), "first_invest_turn_recall": _first(rc),
                            "invested_A_hold": float(np.mean(_col(ho, "invested", 0))),
                            "invested_A_recall": float(np.mean(_col(rc, "invested", 0)))}
                    # the value of the gold itself: (Recall - Hold) at 10 gold minus at 1 gold, paired
                    hi = np.subtract(_col(by[f"{model}|{wname}|hp{hp6}|g10|opp{opp}|recall"], "ret"),
                                     _col(by[f"{model}|{wname}|hp{hp6}|g10|opp{opp}|hold"], "ret"))
                    lo = np.subtract(_col(by[f"{model}|{wname}|hp{hp6}|g1|opp{opp}|recall"], "ret"),
                                     _col(by[f"{model}|{wname}|hp{hp6}|g1|opp{opp}|hold"], "ret"))
                    summary[f"{model}|{wname}|hp{hp6}|opp{opp}|gold_interaction_10_vs_1"] = _paired(hi, lo)
    return {"n_seeds": n, "window": window, "gamma": 0.99, "agents": "edge_trader vs edge_trader",
            "alpha": ALPHA0, "summary": summary}


def _per_event(rows, s):
    ev = sum(r["invest_events"][s] for r in rows)
    return sum(r["invested"][s] for r in rows) / ev if ev else None


def _first(rows):
    v = [r["first_invest_turn"] for r in rows if r["first_invest_turn"] is not None]
    return {"median": float(np.median(v)) if v else None, "share": len(v) / len(rows)}


# ---------------------------------------------------------------------- stage response
RESPONSE_MATCHUPS = (("edge_trader", "edge_trader"), ("aggressive_trader", "hold_farmer"),
                     ("gold_recaller", "gold_recaller"))


def response_snapshots(cfg: GameConfig, turns=(30, 90)):
    """Snapshots of mutual edge-trader play (continuous model, alpha 0.045) with both champions present, near the
    given turns; returns (snapshots, environment turns played to find them)."""
    env = LaneEnv(cfg, max_turns=max(turns) + 30, cache_size=20_000)
    ag = agents()
    found, played = {}, 0
    for g in range(6):
        log = play_episode(env, ag["edge_trader"], ag["edge_trader"], 44_000 + g, record=True)
        played += len(log.turns)
        for k in turns:
            if k in found:
                continue
            for i in log.turns[k:k + 30]:
                c = M4State.from_list(i["champ_before"])
                if c.a.absent == 0 and c.b.absent == 0:
                    found[k] = Snapshot.from_info(i, (cfg.skill_a, cfg.skill_b))
                    break
        if len(found) == len(turns):
            break
    return [found[k] for k in turns], played


def stage_response(n: int = 10, window: int = 120):
    t0 = time.perf_counter()
    snaps, played = response_snapshots(cont())
    charge("response_snapshots", played, time.perf_counter() - t0, time.process_time())
    jobs = []
    variants = [("cont", a, x) for a in ALPHAS for x in (0, 4, 7, 14)] + [("tiers", None, N) for N in (0, 1, 2)]
    for si, s0 in enumerate(snaps):
        # equal levels: B copies A's XP and both champions are present at full HP of that level
        for model, a, x in variants:
            cfg = cont(a) if model == "cont" else GameConfig.m4()
            t = tables(cfg)
            L = level(s0.champ.a, t)
            s = _set(s0, 1, xp=s0.champ.a.xp, hp=t.hp_max[L], gold=0, items=0)
            s = _set(s, 0, hp=t.hp_max[L], gold=0, items=x * MU if model == "cont" else x)
            for x_ag, y_ag in RESPONSE_MATCHUPS:
                jobs.append({"id": f"{model}|{a}|{x}|s{si}|{x_ag}-{y_ag}", "cfg": cfg.to_dict(), "snap": s.to_dict(),
                             "plan": "follow", "agents": (x_ag, y_ag), "n": n, "window": window, "key": (si,),
                             "root": 4_242})
    res = _run("response", jobs)
    by = {r["id"]: r["rows"] for r in res}
    summary = {}
    for model, a, x in variants:
        if (model, x) in (("cont", 0), ("tiers", 0)):
            continue
        for x_ag, y_ag in RESPONSE_MATCHUPS:
            mu = f"{x_ag}-{y_ag}"
            lead = sum((by[f"{model}|{a}|{x}|s{si}|{mu}"] for si in range(len(snaps))), [])
            ref_a = a if model == "cont" else None
            even = sum((by[f"{model}|{ref_a}|0|s{si}|{mu}"] for si in range(len(snaps))), [])
            k = f"{model}|alpha={a}|lead={x}|{mu}"
            summary[k] = {"return": _paired(_col(lead, "ret"), _col(even, "ret")),
                          "trades_A": _paired(_col(lead, "trades", 0), _col(even, "trades", 0)),
                          "trades_B": _paired(_col(lead, "trades", 1), _col(even, "trades", 1)),
                          "trade_dmg_A": _paired(_col(lead, "trade_dmg_hp", 0), _col(even, "trade_dmg_hp", 0)),
                          "trade_dmg_B": _paired(_col(lead, "trade_dmg_hp", 1), _col(even, "trade_dmg_hp", 1)),
                          "deaths_A": _paired(_col(lead, "deaths", 0), _col(even, "deaths", 0)),
                          "deaths_B": _paired(_col(lead, "deaths", 1), _col(even, "deaths", 1)),
                          "farm_A": _paired(_col(lead, "farm", 0), _col(even, "farm", 0)),
                          "farm_B": _paired(_col(lead, "farm", 1), _col(even, "farm", 1)),
                          "away_B": _paired(_col(lead, "away", 1), _col(even, "away", 1)),
                          "retreat_or_safe_B": _paired(_col(lead, "retreat_or_safe", 1),
                                                      _col(even, "retreat_or_safe", 1)),
                          "xp_B": _paired(_col(lead, "xp", 1), _col(even, "xp", 1)),
                          "invested_per_event_A": _per_event(lead, 0), "invested_per_event_B": _per_event(lead, 1),
                          "invest_events_A": float(np.mean(_col(lead, "invest_events", 0)))}
    levels = [level(s.champ.a, tables(cont())) for s in snaps]
    return {"n_seeds_per_snapshot": n, "snapshots": len(snaps), "levels": levels, "window": window,
            "matchups": RESPONSE_MATCHUPS, "summary": summary}


# ---------------------------------------------------------------------- stage summary
def stage_summary() -> str:
    def load(name):
        with open(os.path.join(OUT, name + ".json"), encoding="utf-8") as f:
            return json.load(f)
    lines = ["# M4.2c continuous investment: stage summaries", ""]
    p = load("power")
    lines += ["## Power factor AP(L, S) / AP(L, 0)", "",
              md_table(["S (resources)"] + [f"alpha {a}" for a in ALPHAS],
                       [[r["S"]] + [f"{r[f'alpha_{a}']:.3f}" for a in ALPHAS] for r in p["rows"]]), ""]
    r = load("recall")["summary"]
    rows = []
    for k, v in r.items():
        if "interaction" in k:
            continue
        x = v["recall_minus_hold"]
        rows.append([k, f"{x['diff']:+.3f} ± {x['ci95']:.3f}", f"{v['farm_A']['diff']:+.2f}", f"{v['xp_A']['diff']:+.1f}",
                     f"{v['away_A']['diff']:+.1f}", f"{v['trade_dmg_A']['diff']:+.2f}", f"{v['deaths_B']['diff']:+.2f}"])
    lines += ["## Recall now minus Hold now (edge_trader continuation)", "",
              md_table(["variant", "return", "farm A", "XP A", "away A", "trade dmg A (H)", "deaths B"], rows), ""]
    rows = [[k, f"{v['diff']:+.3f} ± {v['ci95']:.3f}"] for k, v in r.items() if "interaction" in k]
    lines += ["## Gold interaction (Recall - Hold at 10 gold) - (at 1 gold)", "", md_table(["state", "value"], rows), ""]
    s = load("response")["summary"]
    rows = []
    for k, v in s.items():
        rows.append([k, f"{v['return']['diff']:+.3f} ± {v['return']['ci95']:.3f}", f"{v['trades_A']['diff']:+.2f}",
                     f"{v['trades_B']['diff']:+.2f}", f"{v['trade_dmg_A']['diff']:+.2f}",
                     f"{v['deaths_B']['diff']:+.2f}", f"{v['deaths_A']['diff']:+.2f}", f"{v['farm_B']['diff']:+.2f}",
                     f"{v['retreat_or_safe_B']['diff']:+.1f}"])
    lines += ["## Investment lead vs even (paired)", "",
              md_table(["variant", "return to leader", "trades A", "trades B", "trade dmg A (H)", "deaths B",
                        "deaths A", "farm B", "B retreat/safe turns"], rows), ""]
    b = ledger()
    lines += ["## Budget", "", "```", json.dumps(b, indent=1), "```"]
    md = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return md


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("power", "combat", "recall", "response", "summary"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--window", type=int, default=120)
    a = ap.parse_args(argv)
    if a.stage == "power":
        save("power", stage_power())
    elif a.stage == "combat":
        save("combat", stage_combat())
    elif a.stage == "recall":
        save("recall", stage_recall(a.n, a.window))
    elif a.stage == "response":
        save("response", stage_response(a.n, a.window))
    else:
        print(stage_summary())
        return
    print(json.dumps(ledger()["total"]))


if __name__ == "__main__":
    main()
