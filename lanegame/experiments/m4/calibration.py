"""M4.2b: staged economic and combat calibration (docs/m4_calibration_report.md). Exploratory; the M4 defaults are
NOT changed. Every candidate is `GameConfig.m4(gank_damage_position=..., item_mult=...)` with all other fields at their
approved values.

    python -m lanegame.experiments.m4.calibration --stage A          gank-damage screening
    python -m lanegame.experiments.m4.calibration --stage B1         intrinsic item value (controlled 40-turn window)
    python -m lanegame.experiments.m4.calibration --stage B2         realized item value (normal purchases from s0)
    python -m lanegame.experiments.m4.calibration --stage R          does affordability change Recall-now minus Hold?
    python -m lanegame.experiments.m4.calibration --stage CD --configs g0.40_i1.35,...   combined candidates (C) and
                                                                     the Shove -> Recall cycle head-to-head (D)
    python -m lanegame.experiments.m4.calibration --stage DCF --configs ...   Phase D counterfactual plans
    python -m lanegame.experiments.m4.calibration --stage F --configs ...     finalist confirmation (W = 1000)
    python -m lanegame.experiments.m4.calibration --stage FB         Freeze-breaking farm-spot prototype (experiment only)
    python -m lanegame.experiments.m4.calibration --stage S4 --configs g0.40_i1.35   targeted S4 checks of a candidate
    python -m lanegame.experiments.m4.calibration --stage summary    summary.json from the stage files (no simulation)

Budget (assignment §2): every stage adds its wall time, summed worker CPU time and environment turns to
results/m4_calibration/budget.json; a stage is refused once 2,000,000 turns or 900 s of wall time are used.

Pairing: every configuration plays the same seeds (common random numbers: identical chance vectors turn by turn until
trajectories diverge) from the same start states; head-to-head values put the focal agent in both seats on each seed
(as `env.discounted.evaluate_discounted`). Stateful agents get reset()/notify() through `env.discounted._play`.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np

from ...champions_m4 import M4State, level
from ...config import B_HOLD, B_RECALL, B_SAFE, CHAMP_ABSENT, F, NONE, PRESSURE, SH, SP, GameConfig, champ_act
from ...diagnostics import Commit, FunctionPlan, ProbCache, Snapshot, rollout, rollout_seed
from ...diagnostics.branch import Plan
from ...env import LaneEnv, play_episode
from ...env.agents_m4 import M4ScriptAgent, ShoveRecallAgent
from ...env.discounted import _play, discounted_sum, tail_constant
from ...io import dump_json
from ...progression import tables
from ..common import RESULTS_DIR, md_table
from . import economy
from .common import Z95, git_commit, mean_se, pool_map

OUT = os.path.join(RESULTS_DIR, "m4_calibration")
BUDGET_TURNS = 2_000_000
BUDGET_WALL_S = 900.0
PROCS = 8
RECALL = champ_act(SP, B_RECALL)
HOLD = champ_act(SP, B_HOLD)
SAFE = champ_act(SP, B_SAFE)
SHOVE = champ_act(SH, B_HOLD)

BASELINE = "g0.60_i1.20"


# ---------------------------------------------------------------------- configurations and agents
def cfg_of(tag: str) -> GameConfig:
    """'g0.40_i1.35' -> GameConfig.m4(gank_damage_position=0.40, item_mult=1.35)."""
    g, i = tag.split("_")
    return GameConfig.m4(gank_damage_position=float(g[1:]), item_mult=float(i[1:]))


def agents() -> Dict[str, object]:
    a = economy.agents()
    a["cond_shove"] = M4ScriptAgent("cond_shove", SP, B_HOLD, PRESSURE, cshove=True)
    a["shove_recall"] = ShoveRecallAgent()
    a["shove_recall_p"] = ShoveRecallAgent("shove_recall_p", pressure=True)
    return a


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
        raise SystemExit(f"budget exhausted ({t}); stage {stage} refused (assignment §2: report, do not extend)")


def charge(stage: str, turns: int, wall: float, cpu: float) -> dict:
    d = ledger()
    d["stages"][stage] = {"turns": int(turns), "wall_s": round(wall, 2), "cpu_s": round(cpu, 2)}
    d["total"] = {"turns": int(sum(v["turns"] for v in d["stages"].values())),
                  "wall_s": round(sum(v["wall_s"] for v in d["stages"].values()), 2),
                  "cpu_s": round(sum(v["cpu_s"] for v in d["stages"].values()), 2)}
    os.makedirs(OUT, exist_ok=True)
    dump_json(d, _ledger_path())
    return d


def save(name: str, data: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    dump_json(data, os.path.join(OUT, name + ".json"))


# ---------------------------------------------------------------------- per-turn counters
class Counters:
    """Per-side counters of one game from the environment's `info` (seat indices 0 = A, 1 = B)."""

    def __init__(self, cfg: GameConfig):
        self.cfg, self.t = cfg, tables(cfg)
        self.c = [collections.Counter(), collections.Counter()]
        self.purchase_turns = [[], []]
        self.lead_spells: List[int] = []
        self._lead, self._run = 0, 0
        self.turn = 0
        self.disc = collections.Counter()        # discounted payoff components to seat A
        self.w = 1.0

    def __call__(self, i: dict) -> None:
        t, g = self.t, self.cfg.gamma
        for s in (0, 1):
            c = self.c[s]
            lv = i["levels_before"][s]
            present = not i["away"][s]
            c["present"] += present
            c["ganks"] += i["ganked"][s]
            if i["ganked"][s]:
                c["gank_trials"] += i["gank_damage"][s] / t.hp_unit[lv]
                c["gank_hp_frac"] += i["gank_damage"][s] / t.hp_max[lv]
            c["gank_deaths"] += bool(i["died"][s] and i["gank_damage"][s] > 0)
            c["deaths"] += i["died"][s]
            c["recalls"] += i["recalled"][s]
            if i["recalled"][s]:
                hp0 = i["champ_before"][s][1]
                c["recalls_above_third"] += hp0 > t.hp_max[lv] / 3
            c["farm"] += i["farm"][s]
            c["xp"] += i["xp_gain"][s] + i["xp_discarded"][s]
            c["tower_dmg"] += i["tower_damage_dealt"][s]
            c["plates"] += i["plates_broken"][s]
            c["trades"] += i["trade_connected"][s]
            c["trade_dealt"] += i["trade_damage"][s] / t.hp_max[i["levels_before"][1 - s]]
            c["tower_shot_hp_frac"] += i["tower_damage"][s] / t.hp_max[lv]
            if present:
                c["exposure_sum"] += i["exposure"][s]
                c["forward_turns"] += i["exposure"][s] >= 1
            c["shove_turns"] += i["waves"][s] == SH
            c["crashes"] += i["crash_hp"][s] > 0
            if i["purchases"][s]:
                self.purchase_turns[s] += [self.turn] * i["purchases"][s]
        lead = (i["items"][0] > i["items"][1]) - (i["items"][0] < i["items"][1])
        self.c[0]["items_unequal"] += lead != 0
        if lead != self._lead:
            if self._lead != 0 and self._run:
                self.lead_spells.append(self._run)
            self._lead, self._run = lead, 0
        self._run += lead != 0
        for k in ("farm_payoff", "plate_payoff", "bounty_payoff", "reward"):
            self.disc[k] += self.w * i[k]
        self.w *= g
        self.turn += 1

    def close(self) -> None:
        if self._lead != 0 and self._run:
            self.lead_spells.append(self._run)
            self._run = 0


# ---------------------------------------------------------------------- head-to-head jobs (both seats, CRN)
def h2h_job(job: dict) -> dict:
    """x against y on seeds 0..n-1 of `root`, x in seat A then seat B on each seed. Returns per-seed paired values,
    per-agent counters (x, y) and the payoff components to x."""
    t0 = time.process_time()
    cfg = cfg_of(job["cfg"])
    ag = agents()
    x, y = ag[job["x"]], ag[job["y"]]
    W = job["window"]
    env = LaneEnv(cfg, cache_size=50_000)
    seeds = _all_seeds(job["root"], job["n"])
    vals, comps, cx, cy, extra = [], collections.Counter(), collections.Counter(), collections.Counter(), []
    turns = 0
    for m in range(job["n"]):
        pair = []
        for seat in (0, 1):
            cnt = Counters(cfg)
            agents_ = (x, y) if seat == 0 else (y, x)
            r, _ = _play(env, agents_, seeds[m](), W, on_step=cnt)
            cnt.close()
            turns += len(r)
            sign = 1.0 if seat == 0 else -1.0
            pair.append(sign * discounted_sum(r, cfg.gamma))
            for k in ("farm_payoff", "plate_payoff", "bounty_payoff"):
                comps[k] += sign * cnt.disc[k] / (2 * job["n"])
            cx.update(cnt.c[seat])
            cy.update(cnt.c[1 - seat])
            extra.append({"purchases_x": cnt.purchase_turns[seat], "purchases_y": cnt.purchase_turns[1 - seat],
                          "lead_spells": cnt.lead_spells, "items_unequal": cnt.c[0]["items_unequal"],
                          "early_lead": _early_lead(r, cnt, seat)})
        vals.append(0.5 * (pair[0] + pair[1]))
    return {"cfg": job["cfg"], "x": job["x"], "y": job["y"], "window": W, "values": vals,
            "x_counters": dict(cx), "y_counters": dict(cy), "components": dict(comps), "games": 2 * job["n"],
            "extra": extra, "turns": turns, "cpu_s": time.process_time() - t0}


class _Seed:
    """A seed that yields a FRESH SeedSequence per game (spawn() advances a SeedSequence, so one object reused for
    both seats would give them different streams)."""

    def __init__(self, entropy, key):
        self.entropy, self.key = entropy, key

    def __call__(self) -> np.random.SeedSequence:
        return np.random.SeedSequence(self.entropy, spawn_key=self.key)


def _all_seeds(root: int, n: int) -> List[_Seed]:
    """Per-seed streams exactly as evaluate_discounted (start index 0)."""
    ch = np.random.SeedSequence(root).spawn(n)
    return [_Seed(ss.entropy, ss.spawn_key + (0, 0)) for ss in ch]


def _early_lead(r: np.ndarray, cnt: Counters, seat: int) -> Optional[dict]:
    """Purchase-timing effect within a game: the side that bought its FIRST tier strictly earlier; payoff to it over
    the 60 turns after that purchase vs the 60 turns before (undiscounted, to seat A's reward converted)."""
    pa, pb = cnt.purchase_turns
    if not pa or not pb or pa[0] == pb[0]:
        return None
    first = 0 if pa[0] < pb[0] else 1
    p = min(pa[0], pb[0])
    sign = 1.0 if first == 0 else -1.0
    after = float(sign * r[p + 1:p + 61].sum())
    before = float(sign * r[max(0, p - 60):p + 1].sum())
    return {"first": "x" if first == seat else "y", "turn": p, "gap": abs(pa[0] - pb[0]), "after60": after,
            "before60": before}


def run_h2h(stage: str, configs: List[str], matchups: List[tuple], n: int, window: int, root: int) -> List[dict]:
    jobs = [{"cfg": c, "x": x, "y": y, "n": n, "window": window, "root": root} for c in configs for x, y in matchups]
    return pool_map(h2h_job, jobs, PROCS)


def summarize_h2h(res: dict) -> dict:
    v = np.asarray(res["values"])
    m, se = mean_se(v)
    turns_per_agent = res["games"] * res["window"]
    per100 = lambda c, k: 100.0 * c.get(k, 0) / turns_per_agent
    out = {"value": m, "se": se, "ci95": [m - Z95 * se, m + Z95 * se], "n_seeds": len(v)}
    for who in ("x", "y"):
        c = res[f"{who}_counters"]
        g = max(1, c.get("ganks", 0))
        out[who] = {"ganks_per100": per100(c, "ganks"), "gank_trials_per_gank": c.get("gank_trials", 0) / g,
                    "gank_hp_frac_per_gank": c.get("gank_hp_frac", 0) / g,
                    "deaths_per100": per100(c, "deaths"), "gank_deaths_per100": per100(c, "gank_deaths"),
                    "recalls_per100": per100(c, "recalls"), "recalls_above_third_per100": per100(c, "recalls_above_third"),
                    "farm_per100": per100(c, "farm"), "xp_per100": per100(c, "xp"),
                    "tower_dmg_per100": per100(c, "tower_dmg"), "plates_per100": per100(c, "plates"),
                    "trades_per100": per100(c, "trades"), "forward_share": c.get("forward_turns", 0) / max(1, c.get("present", 0)),
                    "shove_share": c.get("shove_turns", 0) / turns_per_agent,
                    "tower_shot_hp_frac_per100": per100(c, "tower_shot_hp_frac")}
    out["components"] = res["components"]
    return out


# ---------------------------------------------------------------------- Phase A: gank damage
A_MATCHUPS = [("hold_farmer", "hold_farmer"), ("presser", "hold_farmer"), ("cond_shove", "hold_farmer"),
              ("edge_trader", "hold_farmer")]


def stage_A(n=12, window=400):
    configs = ["g0.60_i1.20", "g0.40_i1.20", "g0.30_i1.20"]
    res = run_h2h("A", configs, A_MATCHUPS, n, window, 81_000)
    return {"configs": configs, "matchups": A_MATCHUPS, "n_seeds": n, "window": window,
            "results": {f"{r['cfg']}|{r['x']} vs {r['y']}": summarize_h2h(r) for r in res},
            "paired_vs_baseline": _paired_vs(res, BASELINE)}, res


def _paired_vs(res: List[dict], base: str) -> dict:
    """Per matchup: candidate minus baseline on the same seeds (CRN across configurations)."""
    by = {(r["cfg"], r["x"], r["y"]): np.asarray(r["values"]) for r in res}
    out = {}
    for (c, x, y), v in by.items():
        if c == base or (base, x, y) not in by:
            continue
        d = v - by[(base, x, y)]
        m, se = mean_se(d)
        out[f"{c}|{x} vs {y}"] = {"diff": m, "se": se, "ci95": [m - Z95 * se, m + Z95 * se]}
    return out


# ---------------------------------------------------------------------- rollout jobs (counterfactual branches)
def make_plan(spec: str) -> Plan:
    if spec == "follow":
        from ...diagnostics import FollowPolicy
        return FollowPolicy()
    if spec in ("recall", "hold"):
        return Commit(RECALL if spec == "recall" else HOLD, spec)
    if spec == "farm8":
        return FunctionPlan("farm8", lambda obs, t: (HOLD if HOLD in obs.legal else None) if t < 8 else None)
    if spec == "recall_now":
        return FunctionPlan("recall_now", _walk_recall_fn(0))
    if spec == "shove_then_recall":
        return FunctionPlan("shove_then_recall", _shove_then_recall_fn())
    if spec == "shove_stay":
        return FunctionPlan("shove_stay", lambda obs, t: (SHOVE if SHOVE in obs.legal else None) if t < 4 else None)
    raise ValueError(spec)


def _walk_recall_fn(start: int):
    """From turn `start` on: Recall when legal, else walk back (Slow Push + Safe); hand back once away or after 8."""
    def fn(obs, t):
        if t < start:
            return None
        if obs.legal == (CHAMP_ABSENT,) or t >= start + 8:
            return None
        return RECALL if RECALL in obs.legal else (SAFE if SAFE in obs.legal else None)
    return fn


def _shove_then_recall_fn():
    """Shove + Hold until the own front stands at x >= 1 or has crashed (at most 4 turns), then walk back and Recall.
    Holds per-rollout state (a fresh plan per rollout)."""
    st = {"phase": "shove", "t0": 0}

    def fn(obs, t):
        if obs.legal == (CHAMP_ABSENT,):
            return None
        if st["phase"] == "shove":
            fx, fh = obs.state.side(obs.side)[:2]
            pushed = obs.state.siege_of(obs.side) > 0 or (fh > 0 and fx >= 1)
            if not pushed and t < 4 and SHOVE in obs.legal:
                return SHOVE
            st["phase"], st["t0"] = "recall", t
        if t >= st["t0"] + 8:
            return None
        return RECALL if RECALL in obs.legal else (SAFE if SAFE in obs.legal else None)
    return fn


def rollout_job(job: dict) -> dict:
    t0 = time.process_time()
    cfg = cfg_of(job["cfg"])
    snap = Snapshot.from_dict(job["snap"])
    ag = agents()
    pair = (ag[job["agents"][0]], ag[job["agents"][1]])
    env = LaneEnv(snap.config(cfg), cache_size=50_000)
    cache = ProbCache()
    tb = tables(cfg)
    rets, cnts, turns = [], [], 0
    f = job.get("focal", 0)
    for m in range(job["n"]):
        r = rollout(env, snap, f, make_plan(job["plan"]), pair, rollout_seed(job["root"], m, *job["key"]), cache,
                    max_turns=job["window"], record=True)
        turns += r.turns
        rets.append(r.ret - job.get("offset", 0.0))
        c = collections.Counter()
        for i in r.infos:
            c["unequal"] += i["items"][0] != i["items"][1]
            c["lead_focal"] += (i["items"][f] > i["items"][1 - f])
            for s, tag in ((f, "f"), (1 - f, "o")):
                lv = i["levels_before"][s]
                c[f"deaths_{tag}"] += i["died"][s]
                c[f"ganks_{tag}"] += i["ganked"][s]
                c[f"farm_{tag}"] += i["farm"][s]
                c[f"trade_dealt_{tag}"] += i["trade_damage"][s] / tb.hp_max[i["levels_before"][1 - s]]
                c[f"trades_{tag}"] += i["trade_connected"][s]
                c[f"recalls_{tag}"] += i["recalled"][s]
                c[f"forward_{tag}"] += (not i["away"][s]) and i["exposure"][s] >= 1
                c[f"purchases_{tag}"] += i["purchases"][s]
                c[f"plates_{tag}"] += i["plates_broken"][s]
        cnts.append(dict(c))
    return {"id": job["id"], "rets": rets, "counters": cnts, "turns": turns, "cpu_s": time.process_time() - t0}


def paired(a, b) -> dict:
    d = np.asarray(a) - np.asarray(b)
    m, se = mean_se(d)
    return {"diff": m, "se": se, "ci95": [m - Z95 * se, m + Z95 * se],
            "significant": bool(np.isfinite(se) and abs(m) > Z95 * se), "a": float(np.mean(a)), "b": float(np.mean(b))}


def mean_counters(cnts: List[dict]) -> dict:
    keys = set().union(*cnts) if cnts else set()
    return {k: float(np.mean([c.get(k, 0) for c in cnts])) for k in sorted(keys)}


# ---------------------------------------------------------------------- Phase B1: intrinsic item value
B_MULTS = ["g0.60_i1.20", "g0.60_i1.35", "g0.60_i1.50"]
B1_CONTS = [("hold_farmer", "hold_farmer"), ("edge_trader", "edge_trader"), ("aggressive_trader", "hold_farmer")]


def b1_snapshots() -> List[Snapshot]:
    """Reachable snapshots of mutual edge-trader play under the BASELINE rules (turns 20/40/80/150, 3 games): both
    present, A below the item cap. Gold is set to 0 for both, so neither can buy inside a short window unless it
    earns a full price first; the item difference is the controlled treatment."""
    base = cfg_of(BASELINE)
    t = tables(base)
    out = []
    for s in economy.edge_snapshots(base):
        a, b = s.champ.a, s.champ.b
        if a.absent or b.absent or a.items >= t.item_tiers:
            continue
        out.append(s.with_side(0, gold=0).with_side(1, gold=0))
    return out


def stage_B1(n=12, window=40, long_window=200):
    snaps = b1_snapshots()
    jobs = []
    for c in B_MULTS:
        for ci, cont in enumerate(B1_CONTS):
            for i, s in enumerate(snaps):
                for var, sv in (("N", s), ("N+1", s.with_side(0, items=s.champ.a.items + 1))):
                    jobs.append({"id": f"{c}|{cont[0]} vs {cont[1]}|{i}|{var}|W{window}", "cfg": c,
                                 "snap": sv.to_dict(), "plan": "follow", "agents": cont, "n": n, "window": window,
                                 "key": (ci, i), "root": 701})
                    if cont == ("edge_trader", "edge_trader"):
                        jobs.append({"id": f"{c}|{cont[0]} vs {cont[1]}|{i}|{var}|W{long_window}", "cfg": c,
                                     "snap": sv.to_dict(), "plan": "follow", "agents": cont, "n": n,
                                     "window": long_window, "key": (ci, i), "root": 702})
    res = pool_map(rollout_job, jobs, PROCS)
    by = {r["id"]: r for r in res}
    out = {}
    for jid, r in by.items():
        p = jid.split("|")
        if p[3] != "N+1":
            continue
        base = by["|".join(p[:3] + ["N"] + p[4:])]
        key = f"{p[0]}|{p[1]}|{p[4]}"
        out.setdefault(key, {"rets_plus": [], "rets_base": [], "c_plus": [], "c_base": []})
        out[key]["rets_plus"] += r["rets"]
        out[key]["rets_base"] += base["rets"]
        out[key]["c_plus"] += r["counters"]
        out[key]["c_base"] += base["counters"]
    summary = {}
    for key, d in out.items():
        summary[key] = {**paired(d["rets_plus"], d["rets_base"]), "n": len(d["rets_plus"]),
                        "with": mean_counters(d["c_plus"]), "without": mean_counters(d["c_base"])}
    return {"snapshots": len(snaps), "n_seeds": n, "windows": [window, long_window], "summary": summary,
            "duel_table": duel_table()}, res


def duel_table() -> dict:
    """Analytic: hits to kill from full HP (single hits) at equal levels, the item holder against an itemless one,
    and the expected-damage ratio, per multiplier (no simulation)."""
    out = {}
    for c in B_MULTS:
        t = tables(cfg_of(c))
        rows = {}
        for L in (3, 6, 11, 18):
            H = t.hp_max[L]
            rows[L] = {f"N{N}": {"dmg1": t.dmg[1][L][N], "hits_to_kill": math.ceil(H / t.dmg[1][L][N])}
                       for N in range(t.item_tiers + 1)}
        out[c] = rows
    return out


# ---------------------------------------------------------------------- Phase B2: realized item value
B2_MATCHUPS = [("hold_farmer", "hold_farmer"), ("gold_recaller", "hold_farmer"), ("edge_trader", "edge_trader"),
               ("aggressive_trader", "hold_farmer")]


def stage_B2(n=12, window=400):
    res = run_h2h("B2", B_MULTS, B2_MATCHUPS, n, window, 82_000)
    summary = {}
    for r in res:
        key = f"{r['cfg']}|{r['x']} vs {r['y']}"
        ex = r["extra"]
        first = [e["purchases_x"][0] for e in ex if e["purchases_x"]] + [e["purchases_y"][0] for e in ex if e["purchases_y"]]
        second = [e["purchases_x"][1] for e in ex if len(e["purchases_x"]) > 1] + \
                 [e["purchases_y"][1] for e in ex if len(e["purchases_y"]) > 1]
        spells = [s for e in ex for s in e["lead_spells"]]
        el = [e["early_lead"] for e in ex if e["early_lead"]]
        after = [e["after60"] for e in el]
        before = [e["before60"] for e in el]
        ma, sa = mean_se(after) if len(after) > 1 else (float("nan"), float("nan"))
        mb, sb = mean_se(before) if len(before) > 1 else (float("nan"), float("nan"))
        summary[key] = {**summarize_h2h(r),
                        "first_purchase_median": float(np.median(first)) if first else None,
                        "second_purchase_median": float(np.median(second)) if second else None,
                        "unequal_item_share": float(np.mean([e["items_unequal"] for e in ex])) / window,
                        "lead_spell_mean": float(np.mean(spells)) if spells else 0.0,
                        "lead_spell_median": float(np.median(spells)) if spells else 0.0,
                        "lead_spells_per_game": len(spells) / len(ex),
                        "early_buyer_after60": {"mean": ma, "ci95": Z95 * sa, "n": len(after)},
                        "early_buyer_before60": {"mean": mb, "ci95": Z95 * sb, "n": len(before)},
                        "mean_purchase_gap": float(np.mean([e["gap"] for e in el])) if el else None}
    return {"configs": B_MULTS, "matchups": B2_MATCHUPS, "n_seeds": n, "window": window, "summary": summary}, res


# ---------------------------------------------------------------------- Stage R: affordability x Recall
def stage_R(n=12, window=200):
    base = cfg_of(BASELINE)
    t = tables(base)
    states = {k: v for k, v in economy.recall_states(base).items() if k.startswith("play")}
    states = dict(list(states.items())[:3])
    jobs = []
    for c in B_MULTS:
        for name, s in states.items():
            a = s.champ.a
            if a.items >= t.item_tiers:
                s = s.with_side(0, items=t.item_tiers - 1)
                a = s.champ.a
            H = t.hp_max[level(a, t)]
            price = t.prices[a.items]
            for hp6 in (4, 6):
                for afford in (True, False):
                    v = s.with_side(0, hp=max(1, H * hp6 // 6), gold=price if afford else price - 1_000_000)
                    for plan in ("recall", "hold"):
                        jobs.append({"id": f"{c}|{name}|{hp6}|{'afford' if afford else 'poor'}|{plan}", "cfg": c,
                                     "snap": v.to_dict(), "plan": plan, "agents": ("edge_trader", "edge_trader"),
                                     "n": n, "window": window, "key": (int(name[-1]), hp6, afford), "root": 703})
    res = pool_map(rollout_job, jobs, PROCS)
    by = {r["id"]: np.asarray(r["rets"]) for r in res}
    summary = {}
    for c in B_MULTS:
        for name in states:
            for hp6 in (4, 6):
                k = f"{c}|{name}|{hp6}"
                ra, ha = by[f"{k}|afford|recall"], by[f"{k}|afford|hold"]
                rp, hp_ = by[f"{k}|poor|recall"], by[f"{k}|poor|hold"]
                summary[k] = {"recall_minus_hold_afford": paired(ra, ha), "recall_minus_hold_poor": paired(rp, hp_),
                              "interaction": paired(ra - ha, rp - hp_)}
    pooled = {}
    for c in B_MULTS:
        inter = np.concatenate([by[f"{c}|{nm}|{h}|afford|recall"] - by[f"{c}|{nm}|{h}|afford|hold"]
                                - by[f"{c}|{nm}|{h}|poor|recall"] + by[f"{c}|{nm}|{h}|poor|hold"]
                                for nm in states for h in (4, 6)])
        m, se = mean_se(inter)
        pooled[c] = {"interaction": m, "ci95": Z95 * se, "n": len(inter)}
    return {"states": list(states), "n_seeds": n, "window": window, "summary": summary, "pooled": pooled}, res


# ---------------------------------------------------------------------- Phases C and D: head-to-head
CD_MATCHUPS = [("shove_recall", "hold_farmer"), ("shove_recall", "presser"), ("shove_recall", "gold_recaller"),
               ("shove_recall", "edge_trader"), ("shove_recall", "freezer"), ("shove_recall_p", "hold_farmer"),
               ("presser", "hold_farmer"), ("cond_shove", "hold_farmer"), ("aggressive_trader", "hold_farmer"),
               ("edge_trader", "hold_farmer"), ("gold_recaller", "hold_farmer")]


def stage_CD(configs: List[str], n=8, window=400):
    configs = [BASELINE] + [c for c in configs if c != BASELINE]
    res = run_h2h("CD", configs, CD_MATCHUPS, n, window, 83_000)
    return {"configs": configs, "matchups": CD_MATCHUPS, "n_seeds": n, "window": window,
            "results": {f"{r['cfg']}|{r['x']} vs {r['y']}": summarize_h2h(r) for r in res},
            "paired_vs_baseline": _paired_vs(res, BASELINE)}, res


# ---------------------------------------------------------------------- Phase D: counterfactual plans
D_PLANS = ("farm8", "recall_now", "shove_then_recall", "shove_stay")


def d_states() -> Dict[str, Snapshot]:
    """Reachable states (baseline rules) of shove_recall vs hold_farmer, then vs presser until an enemy-threat state
    is found, one per (wave class, items): own wave pushing (front at x >= 1 or siege), even (both fronts at the lane
    cell), enemy wave threatening (enemy front at own x <= -1); the focal (A) present with items < cap. The
    continuation opponent is hold_farmer in every case (the state is reachable; who reached it does not matter)."""
    base = cfg_of(BASELINE)
    t = tables(base)
    ag = agents()
    env = LaneEnv(base, max_turns=300, cache_size=20_000)
    found: Dict[str, Snapshot] = {}
    games = [("hold_farmer", g) for g in range(30)] + [("presser", g) for g in range(30)]
    for opp, g in games:
        if opp == "presser" and any(k.startswith("threat") for k in found):
            break
        if opp == "hold_farmer" and len(found) >= 4:
            continue
        log = play_episode(env, ag["shove_recall"], ag[opp], 91_000 + g, record=True)
        for i in log.turns[10::3]:
            c = M4State.from_list(i["champ_before"])
            a, b = c.a, c.b
            if a.absent or b.absent or a.items >= t.item_tiers:
                continue
            lane = i["state_before"]
            ax, ah, bx, bh, asg = lane[0], lane[1], lane[4], lane[5], lane[3]
            if asg > 0 or (ah > 0 and ax >= 1):
                cls = "pushing"
            elif bh > 0 and -bx <= -1:
                cls = "threat"
            elif ah > 0 and bh > 0 and ax == 0 and bx == 0:
                cls = "even"
            else:
                continue
            key = f"{cls}_N{a.items}"
            if key not in found:
                found[key] = Snapshot.from_info(i, (base.skill_a, base.skill_b))
    return found


def stage_DCF(configs: List[str], n=10, window=200):
    configs = [BASELINE] + [c for c in configs if c != BASELINE]
    base = cfg_of(BASELINE)
    t = tables(base)
    states = d_states()
    jobs = []
    for c in configs:
        for name, s in states.items():
            a = s.champ.a
            H = t.hp_max[level(a, t)]
            price = t.prices[a.items]
            for hp_tag, hp in (("full", H), ("half", H // 2)):
                for gold_tag, gold in (("afford", price), ("poor", 0)):
                    v = s.with_side(0, hp=hp, gold=gold)
                    for plan in D_PLANS:
                        jobs.append({"id": f"{c}|{name}|{hp_tag}|{gold_tag}|{plan}", "cfg": c, "snap": v.to_dict(),
                                     "plan": plan, "agents": ("shove_recall", "hold_farmer"), "n": n,
                                     "window": window, "key": (sum(map(ord, name)), hp > H // 2, gold > 0),
                                     "root": 704})
    res = pool_map(rollout_job, jobs, PROCS)
    by = {r["id"]: r for r in res}
    summary = {}
    for c in configs:
        for name in states:
            for hp_tag in ("full", "half"):
                for gold_tag in ("afford", "poor"):
                    k = f"{c}|{name}|{hp_tag}|{gold_tag}"
                    r = {p: np.asarray(by[f"{k}|{p}"]["rets"]) for p in D_PLANS}
                    summary[k] = {"means": {p: float(v.mean()) for p, v in r.items()},
                                  "recall_minus_farm": paired(r["recall_now"], r["farm8"]),
                                  "shove_recall_minus_recall": paired(r["shove_then_recall"], r["recall_now"]),
                                  "shove_stay_minus_farm": paired(r["shove_stay"], r["farm8"]),
                                  "best": max(r, key=lambda p: r[p].mean()),
                                  "counters": {p: mean_counters(by[f"{k}|{p}"]["counters"]) for p in D_PLANS}}
    return {"configs": configs, "states": {k: {"lane": list(map(str, v.lane)), "champ": v.champ.to_list(), "k": v.k}
                                           for k, v in states.items()},
            "plans": D_PLANS, "n_seeds": n, "window": window, "summary": summary}, res


# ---------------------------------------------------------------------- Phase F: finalist confirmation
F_MATCHUPS = [("presser", "hold_farmer"), ("shove_recall", "hold_farmer"), ("aggressive_trader", "hold_farmer")]


def snowball_job(job: dict) -> dict:
    t0 = time.process_time()
    cfg = cfg_of(job["cfg"])
    ag = agents()
    env = LaneEnv(cfg, max_turns=job["window"], cache_size=50_000)
    early, final, gaps, turns = [], [], [], 0
    for i in range(job["games"]):
        log = play_episode(env, ag["edge_trader"], ag["edge_trader"], 34_000 + i, record=True)
        r = np.asarray(log.rewards)
        turns += len(r)
        early.append(float(r[:30].sum()))
        final.append(float(np.sum(r * cfg.gamma ** np.arange(len(r)))))
        gaps.append([log.turns[c - 1]["levels"][0] - log.turns[c - 1]["levels"][1] for c in (30, 60, 100)])
    return {"cfg": job["cfg"], "early": early, "final": final, "gaps": gaps, "turns": turns,
            "cpu_s": time.process_time() - t0}


def stage_F(configs: List[str], n=30, window=1000, snow_games=100, snow_window=300):
    configs = [BASELINE] + [c for c in configs if c != BASELINE]
    jobs = [{"cfg": c, "x": x, "y": y, "n": n // 3 + (1 if k < n % 3 else 0), "window": window, "root": 84_000 + 7 * k}
            for c in configs for x, y in F_MATCHUPS for k in range(3)]
    res = pool_map(h2h_job, jobs, PROCS)
    merged = {}
    for r in res:
        key = (r["cfg"], r["x"], r["y"])
        if key not in merged:
            merged[key] = {**r, "values": list(r["values"]), "x_counters": collections.Counter(r["x_counters"]),
                           "y_counters": collections.Counter(r["y_counters"]),
                           "components": collections.Counter({k: v * r["games"] for k, v in r["components"].items()})}
        else:
            m = merged[key]
            m["values"] += r["values"]
            m["x_counters"].update(r["x_counters"])
            m["y_counters"].update(r["y_counters"])
            m["components"].update({k: v * r["games"] for k, v in r["components"].items()})
            m["games"] += r["games"]
            m["turns"] += r["turns"]
            m["cpu_s"] += r["cpu_s"]
    for m in merged.values():
        m["components"] = {k: v / m["games"] for k, v in m["components"].items()}
    snow = pool_map(snowball_job, [{"cfg": c, "games": snow_games // 2, "window": snow_window, "half": h}
                                   for c in configs for h in range(1)], PROCS)
    snowball = {}
    for s in snow:
        e, f = np.asarray(s["early"]), np.asarray(s["final"])
        lead = e > 0
        g = np.abs(np.asarray(s["gaps"]))
        snowball[s["cfg"]] = {"games": len(e), "p_win_given_ahead30": float(np.mean(f[lead] > 0)) if lead.any() else None,
                              "p_win_given_behind30": float(np.mean(f[~lead] > 0)) if (~lead).any() else None,
                              "corr": float(np.corrcoef(e, f)[0, 1]), "level_gap_abs_mean_30_60_100": g.mean(0).tolist(),
                              "level_gap_ge2_share_30_60_100": (g >= 2).mean(0).tolist()}
    res_all = list(merged.values())
    tails = {c: {"omega": tail_constant(cfg_of(c)), "bound_W": cfg_of(c).gamma ** window * tail_constant(cfg_of(c))}
             for c in configs}
    return {"configs": configs, "matchups": F_MATCHUPS, "n_seeds": n, "window": window,
            "results": {f"{r['cfg']}|{r['x']} vs {r['y']}": summarize_h2h(r) for r in res_all},
            "paired_vs_baseline": _paired_vs(res_all, BASELINE), "snowball": snowball, "tail_bounds": tails}, \
        res_all + snow


# ---------------------------------------------------------------------- Freeze-breaking prototype (experiment only)
class KeepReachEnv(LaneEnv):
    """EXPERIMENT ONLY, never a ruleset: the M4 step with the farm spot of a champion whose front CRASHES in the probe
    pass taken from its pre-turn front (instead of the next packet near its base). Implemented by editing the
    probe's result before resolve_turn_m4 reads it; the authoritative lane step is unchanged."""

    def step(self, a, b):
        self._probe_next = True
        return super().step(a, b)

    def _lane_turn(self, s, k, a, b, reach=None):
        res = super()._lane_turn(s, k, a, b, reach)
        if getattr(self, "_probe_next", False):
            self._probe_next = False
            st = res.state
            for side in (0, 1):
                fx0, fh0 = s.side(side)[:2]
                if res.crash_hp[side] > 0 and fh0 > 0:
                    st = st._replace(ax=fx0, ah=fh0) if side == 0 else st._replace(bx=fx0, bh=fh0)
            res = dataclasses.replace(res, state=st)
        return res


def keep_reach_hold_run(e, b_wave, phase, cfg, turns=24):
    """scenarios_m4.hold_run's parked-front measurement (A freezes) with the keep-reach farm spot."""
    from ...champions_m4 import resolve_turn_m4
    from ...mechanics import step_turn, wave_actions
    from ...scenarios_m4 import parked_state
    lane, m4 = parked_state(e, cfg)
    flag = {"probe": True}

    def lane_step(s, kk, x, y, reach):
        res = step_turn(s, kk, x, y, cfg, champion_reach=reach)
        if flag["probe"]:
            flag["probe"] = False
            st = res.state
            for side in (0, 1):
                fx0, fh0 = s.side(side)[:2]
                if res.crash_hp[side] > 0 and fh0 > 0:
                    st = st._replace(ax=fx0, ah=fh0) if side == 0 else st._replace(bx=fx0, bh=fh0)
            res = dataclasses.replace(res, state=st)
        return res
    first = None
    for i in range(turns):
        k = phase + i
        legal = F in wave_actions(lane, 0, cfg)
        flag["probe"] = True
        t = resolve_turn_m4(lane, m4, k, champ_act(F if legal else SP, B_HOLD), champ_act(b_wave, B_HOLD), cfg,
                            lane_step=lane_step, u=None)
        if t.lane.crash_hp[1] > 0 and first is None:
            first = i
        lane, m4 = t.lane.state, t.champ
    return first


def fb_job(job: dict) -> dict:
    t0 = time.process_time()
    cfg = cfg_of(job["cfg"])
    ag = agents()
    x, y = ag[job["x"]], ag[job["y"]]
    env = (KeepReachEnv if job["variant"] == "keep_reach" else LaneEnv)(cfg, cache_size=0)
    seeds = _all_seeds(job["root"], job["n"])
    vals, turns, cx, cy = [], 0, collections.Counter(), collections.Counter()
    for m in range(job["n"]):
        pair = []
        for seat in (0, 1):
            cnt = Counters(cfg)
            r, _ = _play(env, (x, y) if seat == 0 else (y, x), seeds[m](), job["window"], on_step=cnt)
            turns += len(r)
            pair.append((1.0 if seat == 0 else -1.0) * discounted_sum(r, cfg.gamma))
            cx.update(cnt.c[seat])
            cy.update(cnt.c[1 - seat])
        vals.append(0.5 * (pair[0] + pair[1]))
    return {"cfg": job["cfg"], "variant": job["variant"], "x": job["x"], "y": job["y"], "values": vals,
            "x_counters": dict(cx), "y_counters": dict(cy), "games": 2 * job["n"], "window": job["window"],
            "components": {}, "turns": turns, "cpu_s": time.process_time() - t0}


FB_MATCHUPS = [("presser", "freezer"), ("freeze_breaker", "freezer"), ("shove_recall", "freezer"),
               ("g_adaptive_breaker", "freezer"), ("presser", "hold_farmer")]


def stage_FB(configs: List[str], n=8, window=400):
    from ...config import SH as SH_, SP as SP_
    from ...scenarios_m4 import DESIGN_HOLD_TABLE, hold_run
    base = cfg_of(BASELINE)
    table = {}
    for e in range(2, 10):
        for bw in (SP_, SH_):
            cur = [hold_run(e, bw, ph, F, base)["first_crash"] for ph in range(4)]
            kr = [keep_reach_hold_run(e, bw, ph, base) for ph in range(4)]
            table[f"{e}_{'SH' if bw == SH_ else 'SP'}"] = {"design": DESIGN_HOLD_TABLE[(e, bw)][0], "implemented": cur,
                                                           "keep_reach": kr}
    configs = [BASELINE] + [c for c in configs if c != BASELINE]
    jobs = [{"cfg": c, "variant": v, "x": x, "y": y, "n": n, "window": window, "root": 85_000}
            for c in configs for v in ("current", "keep_reach") for x, y in FB_MATCHUPS]
    res = pool_map(fb_job, jobs, PROCS)
    out = {}
    by = {(r["cfg"], r["variant"], r["x"], r["y"]): r for r in res}
    for (c, v, x, y), r in by.items():
        s = summarize_h2h(r)
        out[f"{c}|{v}|{x} vs {y}"] = s
    diffs = {}
    for c in configs:
        for x, y in FB_MATCHUPS:
            a, b = np.asarray(by[(c, "keep_reach", x, y)]["values"]), np.asarray(by[(c, "current", x, y)]["values"])
            diffs[f"{c}|{x} vs {y}"] = paired(a, b)
    return {"parked_table": table, "configs": configs, "matchups": FB_MATCHUPS, "n_seeds": n, "window": window,
            "results": out, "keep_reach_minus_current": diffs}, res


# ---------------------------------------------------------------------- targeted S4 checks of the recommendation
S4_MATCHUPS = [("aggressive_trader", "freezer"), ("aggressive_trader", "edge_trader")]
S4_RECALL_STATES = ("cheap", "expensive", "play0", "play1")


def stage_S4(configs: List[str], n=12, window=400, recall_n=8, recall_window=200):
    """(1) Does stronger trading become dominant? aggressive_trader against its S4 counters (freezer, edge_trader),
    baseline and candidate, CRN. (2) The S4-B Recall sign pattern (Recall now minus Hold now, HP 2/4/6 sixths,
    affordable or not) on the candidate, reduced: 4 states, 8 rollouts, W = 200 (exploratory, truncated)."""
    cand = [c for c in configs if c != BASELINE]
    res = run_h2h("S4", [BASELINE] + cand, S4_MATCHUPS, n, window, 86_000)
    out = {"dominance": {f"{r['cfg']}|{r['x']} vs {r['y']}": summarize_h2h(r) for r in res},
           "dominance_paired_vs_baseline": _paired_vs(res, BASELINE), "recall": {}}
    raw = list(res)
    for c in cand:
        cfg = cfg_of(c)
        jobs = [j for j in economy.build_jobs(cfg, recall_n, recall_window).values()
                if j["id"].startswith("B|") and j["id"].split("|")[1] in S4_RECALL_STATES]
        t0 = time.process_time()
        rr = pool_map(economy.run_job, jobs, PROCS)
        by = {jid: rets for jid, rets, _ in rr}
        cells = {}
        for jid, rets in by.items():
            p = jid.split("|")
            if p[-1] == "recall":
                cells["|".join(p[1:4])] = paired(rets, by["|".join(p[:-1] + ["hold"])])
        afford_effect = [cells[k]["diff"] - cells[k.replace("afford", "poor")]["diff"] for k in cells if "afford" in k]
        out["recall"][c] = {"cells": cells,
                            "positive": [k for k, v in cells.items() if v["significant"] and v["diff"] > 0],
                            "negative": [k for k, v in cells.items() if v["significant"] and v["diff"] < 0],
                            "afford_minus_poor_mean": float(np.mean(afford_effect)),
                            "rollouts": recall_n, "window": recall_window}
        raw.append({"turns": len(jobs) * recall_n * recall_window, "cpu_s": time.process_time() - t0})
    return out, raw


# ---------------------------------------------------------------------- summary (no simulation)
def build_summary() -> dict:
    """results/m4_calibration/summary.json: the headline numbers of every stage file, plus the budget ledger."""
    def load(st):
        with open(os.path.join(OUT, f"stage_{st}.json"), encoding="utf-8") as f:
            return json.load(f)

    def h(v):
        return {"value": round(v["value"], 3), "ci95": round(Z95 * v["se"], 3)}

    out = {"budget": ledger(), "baseline": BASELINE}
    a = load("A")
    out["A_gank"] = {k: {**h(v), **{f: round(v["x"][f], 3) for f in ("ganks_per100", "gank_trials_per_gank",
                                                                      "deaths_per100", "gank_deaths_per100",
                                                                      "recalls_per100", "farm_per100", "xp_per100",
                                                                      "tower_dmg_per100", "plates_per100")},
                         "components": {c: round(x, 3) for c, x in v["components"].items()}}
                     for k, v in a["results"].items()}
    out["A_paired_vs_baseline"] = a["paired_vs_baseline"]
    b1 = load("B1")
    out["B1_intrinsic"] = {k: {"diff": round(v["diff"], 3), "ci95": round(Z95 * v["se"], 3), "n": v["n"]}
                           for k, v in b1["summary"].items()}
    out["B1_hits_to_kill"] = {c: {L: {n: x["hits_to_kill"] for n, x in r.items()} for L, r in t.items()}
                              for c, t in b1["duel_table"].items()}
    b2 = load("B2")
    out["B2_realized"] = {k: {**h(v), **{f: v[f] for f in ("first_purchase_median", "second_purchase_median",
                                                            "unequal_item_share", "lead_spell_mean",
                                                            "lead_spell_median", "early_buyer_after60")}}
                          for k, v in b2["summary"].items()}
    out["R_afford_x_recall"] = load("R")["pooled"]
    cd = load("CD")
    out["CD"] = {k: h(v) for k, v in cd["results"].items()}
    out["CD_paired_vs_baseline"] = cd["paired_vs_baseline"]
    dcf = load("DCF")
    signs = {}
    for k, v in dcf["summary"].items():
        c = k.split("|")[0]
        d = signs.setdefault(c, collections.Counter())
        for name in ("recall_minus_farm", "shove_recall_minus_recall", "shove_stay_minus_farm"):
            x = v[name]
            if x["significant"]:
                d[f"{name}_{'pos' if x['diff'] > 0 else 'neg'}"] += 1
        d[f"best_{v['best']}"] += 1
    out["DCF_sign_counts"] = {c: dict(v) for c, v in signs.items()}
    f = load("F")
    out["F"] = {k: h(v) for k, v in f["results"].items()}
    out["F_paired_vs_baseline"] = f["paired_vs_baseline"]
    out["F_snowball"] = f["snowball"]
    out["F_tail_bounds"] = f["tail_bounds"]
    fb = load("FB")
    out["FB_parked_table_matches_design"] = {k: v["keep_reach"] == v["design"] for k, v in fb["parked_table"].items()}
    out["FB_keep_reach_minus_current"] = fb["keep_reach_minus_current"]
    s4 = load("S4")
    out["S4_dominance"] = {k: h(v) for k, v in s4["dominance"].items()}
    out["S4_recall"] = {c: {x: r[x] for x in ("positive", "negative", "afford_minus_poor_mean", "rollouts", "window")}
                        for c, r in s4["recall"].items()}
    return out


# ---------------------------------------------------------------------- driver
STAGES = {"A": stage_A, "B1": stage_B1, "B2": stage_B2, "R": stage_R, "CD": stage_CD, "DCF": stage_DCF,
          "F": stage_F, "FB": stage_FB, "S4": stage_S4}


def main(argv=None):
    global PROCS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["summary"])
    ap.add_argument("--configs", default="")
    ap.add_argument("--procs", type=int, default=PROCS)
    a = ap.parse_args(argv)
    PROCS = a.procs
    if a.stage == "summary":
        save("summary", build_summary())
        print("wrote", os.path.join(OUT, "summary.json"))
        return None
    check_budget(a.stage)
    t0 = time.perf_counter()
    fn = STAGES[a.stage]
    cfgs = [c for c in a.configs.split(",") if c]
    data, raw = fn(cfgs) if a.stage in ("CD", "DCF", "F", "FB", "S4") else fn()
    wall = time.perf_counter() - t0
    turns = sum(r.get("turns", 0) for r in raw)
    cpu = sum(r.get("cpu_s", 0.0) for r in raw)
    data["meta"] = {"commit": git_commit(), "stage": a.stage, "configs_arg": cfgs, "wall_s": wall, "cpu_s": cpu,
                    "turns": turns, "procs": PROCS,
                    "command": f"python -m lanegame.experiments.m4.calibration --stage {a.stage}"
                               + (f" --configs {a.configs}" if a.configs else "")}
    save(f"stage_{a.stage}", data)
    d = charge(a.stage, turns, wall, cpu)
    print(f"stage {a.stage}: {turns:,} turns, {wall:.1f} s wall, {cpu:.1f} s CPU; totals {d['total']}")
    return data


if __name__ == "__main__":
    main()
