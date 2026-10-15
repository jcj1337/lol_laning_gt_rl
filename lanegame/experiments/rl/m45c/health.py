"""The LIMITED post-training health evaluation of the M4.5c campaign (full.py). It answers seven readiness questions
and nothing more; the comprehensive evaluation (21-point curves, W = 1000 headline against the suite, round-robin
tournament, strategic battery, best-response attackers) is authorized separately.

    1  did all six seeds train?                       full_train.json (full.check)
    2  did the PPO diagnostics stay healthy?          `training`: the histories by tenth, the pre-registered alarms
    3  did they improve against the scripted suite?   `curve`
    4  do the finals beat their earlier snapshots?    `history`
    5  is the second half (4.096M -> 8.192M) still    `h2h` (final vs midpoint and vs 6.144M, W = 1000), the curve's
       improving materially?                          last quarter, and the pre-registered plateau / extension rules
    6  are the seeds consistent?                      `cross` (the 15 pairs of finals) and per-seed everything
    7  any sign of self-play cycling?                 `cycle` (each curve snapshot vs its predecessor) and `history`

plus `behavior` (descriptive state-conditioned records, freezes, Recalls, Wards, early game, the M4.5b ledger, and
self-play from an initialized x1.76 power gap for threat-based zoning, which natural self-play almost never reaches).

Design (every number is env.discounted.evaluate_discounted: paired seats on one chance stream, both seats, the
discounted lead in resources, 95% interval over paired seeds; the same evaluation seeds for every checkpoint and every
training seed, so checkpoint and seed differences are paired):

    curve     initial + the 12 checkpoints CHECKPOINTS (all on the 409,600 grid of the planned 21-point curve, so the
              comprehensive curve only has to add the missing grid points with the same seeds) vs the 9 scripted
              references + the OOD M4.4b network; 32 paired seeds, W = 460                          seed 93,000
    h2h       final vs the 4.096M midpoint and vs the 6.144M snapshot; 64 paired seeds, W = 1000    seed 93,100
    history   final vs the initial network and every other CHECKPOINT; 32 paired seeds, W = 460    seed 93,200
    cycle     each CHECKPOINT vs the previous one (the 7.37M -> final pair is in history); 32, W = 460 seed 93,300
    cross     final vs final, every pair of seeds; 64 paired seeds, W = 1000 (the planned design)   seed 93,400
    cross_mid final vs every OTHER run's midpoint; 64 paired seeds, W = 1000 (added after h2h)  seed 93,450
    behavior  self-play records of the final and the midpoint (16 x 1000 turns), early-game records of the final
              (64 x 120 turns), the final against g_bully and fz_guard_sh (8 x 500, both seats); the M4.5b ledger of
              final and midpoint (48 x 300)                                                  seeds 93,500-93,699

Resumable: every finished matchup is merged into results/m45c/full/health_eval.json at most a minute after it ends,
and a restarted stage only runs the missing matchups.

THE MIDPOINT is the 4,096,000 snapshot of the 8,192,000-transition run (learning rate at 50% of 3e-4). It is not a
separately trained 4.096M policy, which would have annealed its learning rate to 0; "final - midpoint" is the
second half of ONE schedule.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Sequence

import numpy as np

from dataclasses import dataclass

from .common import load_json, meta, run_jobs, save_json
from .full import EXT_BUDGET, EXT_OUT, FULL_BUDGET, FULL_MIDPOINT, FULL_OUT, FULL_SEEDS

BASE_CHECKPOINTS = (409_600, 819_200, 1_638_400, 2_457_600, 3_276_800, 4_096_000, 4_915_200, 5_734_400, 6_144_000,
                    6_553_600, 7_372_800, FULL_BUDGET)


@dataclass(frozen=True)
class Camp:
    """A campaign phase: its checkpoint phase name, budget and records directory. Every fraction of the schedule
    (checkpoints, midpoint, last quarter, curve grid) scales with the budget, so the 8.192M campaign ("full") and the
    matched 2x extension ("ext") are evaluated at the same fractions of their own schedules, on the same seeds."""
    phase: str
    budget: int
    out: str

    @property
    def scale(self) -> int:
        return self.budget // FULL_BUDGET

    @property
    def mid(self) -> int:
        return self.budget // 2

    @property
    def q3(self) -> int:
        return self.budget * 3 // 4

    @property
    def checkpoints(self):
        return tuple(n * self.scale for n in BASE_CHECKPOINTS)

    @property
    def grid(self):
        step = 409_600 * self.scale
        return tuple(range(step, self.budget + 1, step))

    @property
    def tournament(self):
        c = self.checkpoints
        return c[2:8] + c[9:]


MAIN = Camp("full", FULL_BUDGET, FULL_OUT)
EXT = Camp("ext", EXT_BUDGET, EXT_OUT)
CAMPS = {"full": MAIN, "ext": EXT}
CHECKPOINTS = MAIN.checkpoints
LATE_QUARTER = MAIN.q3
CURVE_SEED_F, H2H_SEED_F, HIST_SEED_F, CYCLE_SEED_F, CROSS_SEED_F = 93_000, 93_100, 93_200, 93_300, 93_400
BEHAVIOR_SEED_F = 93_500
LEDGER_SEEDS_F = tuple(range(93_600, 93_700))
DEV_W, HEAD_W, P_DEV, P_HEAD = 460, 1000, 32, 64
EVAL_NAME = "health_eval"
PARTS = ("curve", "h2h", "history", "cycle", "cross")


def spec(seed: int, n, camp: Camp = MAIN) -> str:
    """The agent specification of a campaign checkpoint: 'initial', a snapshot (transitions) or the final."""
    if n == "initial" or n == 0:
        return f"initial:{seed}"
    if n == "final" or n == camp.budget:
        return f"final:{camp.phase}:{seed}"
    return f"snap:{camp.phase}:{seed}:{int(n)}"


def opponents() -> List[str]:
    from .references import OOD, SUITE
    return [f"ref:{r}" for r in SUITE] + list(OOD)


# ---------------------------------------------------------------------------------------------- jobs
def curve_jobs(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = MAIN) -> list:
    pts = ["initial", *camp.checkpoints]
    return [(spec(s, n, camp), y, P_DEV, DEV_W, CURVE_SEED_F) for s in seeds for n in pts for y in opponents()]


def h2h_jobs(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = MAIN) -> list:
    return [(spec(s, "final", camp), spec(s, n, camp), P_HEAD, HEAD_W, H2H_SEED_F) for s in seeds
            for n in (camp.mid, camp.q3)]


def history_jobs(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = MAIN) -> list:
    pts = ["initial"] + [n for n in camp.checkpoints if n not in (camp.mid, camp.q3, camp.budget)]
    return [(spec(s, "final", camp), spec(s, n, camp), P_DEV, DEV_W, HIST_SEED_F) for s in seeds for n in pts]


def cycle_jobs(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = MAIN) -> list:
    c = camp.checkpoints
    pairs = list(zip(c[1:], c[:-1]))[:-1]                               # the last pair (final, previous) is in history
    return [(spec(s, a, camp), spec(s, b, camp), P_DEV, DEV_W, CYCLE_SEED_F) for s in seeds for a, b in pairs]


def cross_jobs(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = MAIN) -> list:
    return [(spec(a, "final", camp), spec(b, "final", camp), P_HEAD, HEAD_W, CROSS_SEED_F) for i, a in enumerate(seeds)
            for b in seeds[i + 1:]]


# ---- the rest of the comprehensive standard evaluation (gated by --confirm-m45c-comprehensive in the CLI)
GRID_POINTS = MAIN.grid                                                  # the planned 21-point curve (+ initial)
TOURNAMENT = MAIN.tournament                                             # 9 checkpoints, every 1/10 from 1/5
HEADLINE_SEED_F, TOURNAMENT_SEED_F, P_TOURN = 93_150, 93_350, 16
COMPREHENSIVE = ("curve_rest", "headline", "tournament", "headline_main", "vs_main", "anneal", "historical")


def curve_rest_jobs(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = MAIN) -> list:
    pts = [n for n in camp.grid if n not in camp.checkpoints]
    return [(spec(s, n, camp), y, P_DEV, DEV_W, CURVE_SEED_F) for s in seeds for n in pts for y in opponents()]


def headline_jobs(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = MAIN, points=("final", "mid")) -> list:
    return [(spec(s, camp.mid if n == "mid" else n, camp), y, P_HEAD, HEAD_W, HEADLINE_SEED_F) for s in seeds
            for n in points for y in opponents()]


def headline_main_jobs(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = EXT) -> list:
    """The headline of the 8.192M finals (the comparison group) on the same seeds and size as the extension's headline,
    stored with the extension's record (the 8.192M campaign's own record is left untouched)."""
    return headline_jobs(seeds, MAIN, ("final",))


def tournament_jobs(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = MAIN) -> list:
    t = camp.tournament
    return [(spec(s, a, camp), spec(s, b, camp), P_TOURN, DEV_W, TOURNAMENT_SEED_F) for s in seeds
            for i, a in enumerate(t) for b in t[i + 1:]]


CROSS_MID_SEED_F = 93_450


def cross_mid_jobs(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = MAIN) -> list:
    """Each final against every OTHER run's midpoint (W = 1000, 64 paired seeds): does the second-half gain measured
    against the run's own midpoint (h2h) hold against independent lineages, or is it lineage-specific? Added after the
    h2h result of the 8.192M campaign, as a check of it (not pre-registered)."""
    return [(spec(a, "final", camp), spec(b, camp.mid, camp), P_HEAD, HEAD_W, CROSS_MID_SEED_F) for a in seeds
            for b in seeds if a != b]


# ---- the budget comparison (extension against the 8.192M campaign; specified 2026-10-15 before any extension result)
VS_MAIN_SEED_F, ANNEAL_SEED_F, HISTORICAL_SEED_F = 94_500, 94_550, 94_600
PILOT_FINAL = "final:pilot:100"


def vs_main_jobs(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = EXT) -> list:
    """Every 16.384M final against every 8.192M final (36 pairs: 6 same-seed, 30 cross-seed), W = 1000, 64 paired
    seeds: both sets completed their own annealing schedule, so this is the budget comparison."""
    return [(spec(a, "final", camp), spec(b, "final", MAIN), P_HEAD, HEAD_W, VS_MAIN_SEED_F) for a in seeds
            for b in seeds]


def anneal_jobs(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = EXT) -> list:
    """The 8.192M final (annealed to 0) against the extension's 8.192M midpoint (learning rate at 50%), every pair:
    both have trained on 8,192,000 transitions, so the difference is the annealing, not the data (the open caveat of
    the 8.192M campaign's final-vs-midpoint result)."""
    return [(spec(b, "final", MAIN), spec(a, camp.mid, camp), P_HEAD, HEAD_W, ANNEAL_SEED_F) for b in seeds
            for a in seeds]


def historical_jobs(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = EXT) -> list:
    """Both final sets against the historical PPO references: the M4.5c pilot final (1.02M transitions) and the
    out-of-distribution M4.4b aggro final (in the curve's suite already, here at W = 1000)."""
    from .references import OOD
    return [(spec(s, "final", c), y, P_HEAD, HEAD_W, HISTORICAL_SEED_F) for c in (MAIN, camp) for s in seeds
            for y in (PILOT_FINAL, *OOD)]


JOBS = {"curve": curve_jobs, "h2h": h2h_jobs, "history": history_jobs, "cycle": cycle_jobs, "cross": cross_jobs,
        "cross_mid": cross_mid_jobs, "curve_rest": curve_rest_jobs, "headline": headline_jobs,
        "headline_main": headline_main_jobs, "tournament": tournament_jobs, "vs_main": vs_main_jobs,
        "anneal": anneal_jobs, "historical": historical_jobs}
SPECS = {"curve": {"episodes": P_DEV, "window": DEV_W, "seed": CURVE_SEED_F},
         "h2h": {"episodes": P_HEAD, "window": HEAD_W, "seed": H2H_SEED_F},
         "history": {"episodes": P_DEV, "window": DEV_W, "seed": HIST_SEED_F},
         "cycle": {"episodes": P_DEV, "window": DEV_W, "seed": CYCLE_SEED_F},
         "cross": {"episodes": P_HEAD, "window": HEAD_W, "seed": CROSS_SEED_F},
         "cross_mid": {"episodes": P_HEAD, "window": HEAD_W, "seed": CROSS_MID_SEED_F,
                       "note": "added after the 8.192M h2h result (not pre-registered)"},
         "curve_rest": {"episodes": P_DEV, "window": DEV_W, "seed": CURVE_SEED_F},
         "headline": {"episodes": P_HEAD, "window": HEAD_W, "seed": HEADLINE_SEED_F},
         "headline_main": {"episodes": P_HEAD, "window": HEAD_W, "seed": HEADLINE_SEED_F},
         "tournament": {"episodes": P_TOURN, "window": DEV_W, "seed": TOURNAMENT_SEED_F},
         "vs_main": {"episodes": P_HEAD, "window": HEAD_W, "seed": VS_MAIN_SEED_F},
         "anneal": {"episodes": P_HEAD, "window": HEAD_W, "seed": ANNEAL_SEED_F},
         "historical": {"episodes": P_HEAD, "window": HEAD_W, "seed": HISTORICAL_SEED_F}}


def turns_of(jobs: list) -> int:
    return int(sum(2 * j[2] * j[3] for j in jobs))


def evaluate(workers: int, parts: Sequence[str] = PARTS, seeds: Sequence[int] = FULL_SEEDS, camp: Camp = MAIN) -> dict:
    """Run the given parts for a campaign phase (resumable; module docstring). Matchups whose (x, y) are identical to
    one already evaluated in the 8.192M campaign's record with the same seed and size (the untrained networks' curve)
    are copied from it, not re-simulated: evaluate_discounted is deterministic."""
    from . import evaluate as EV
    data = load_json(EVAL_NAME, camp.out) or {"meta": meta(stage="health_eval", phase=camp.phase, seeds=list(seeds))}
    timing = data.setdefault("timing", {})
    main = load_json(EVAL_NAME, MAIN.out) if camp is not MAIN else None
    for part in parts:
        js = JOBS[part](seeds, camp)
        res = data.setdefault(part, {})
        if main and part in main:
            for j in js:
                k = f"{j[0]}|{j[1]}"
                if k not in res and k in main[part] and j[0].startswith("initial:"):
                    res[k] = main[part][k]
        todo = [j for j in js if f"{j[0]}|{j[1]}" not in res]
        data[f"{part}_spec"] = {**SPECS[part], "matchups": len(js), "simulated_turns": turns_of(js)}
        print(f"[health:{camp.phase}:{part}] {len(js)} matchups, {len(todo)} to run ({turns_of(todo) / 1e6:.1f} M turns)",
              flush=True)
        if not todo:
            continue
        t0 = time.perf_counter()
        last = [t0]

        def progress(i, r, res=res, part=part):
            (xs, ys), v = r
            res[f"{xs}|{ys}"] = v
            if time.perf_counter() - last[0] > 60:
                save_json(EVAL_NAME, data, camp.out)
                last[0] = time.perf_counter()
                print(f"  [health:{camp.phase}:{part}] {len(res)}/{len(js)}", flush=True)

        run_jobs(EV._job, todo, workers, "", progress)
        wall = time.perf_counter() - t0
        prev = timing.get(part, {"wall_s": 0.0, "turns": 0})
        timing[part] = {"wall_s": prev["wall_s"] + wall, "turns": prev["turns"] + turns_of(todo), "workers": workers}
        timing[part]["turns_per_worker_s"] = timing[part]["turns"] / (timing[part]["wall_s"] * workers)
        save_json(EVAL_NAME, data, camp.out)
        print(f"[health:{camp.phase}:{part}] done in {wall:.0f} s", flush=True)
    return data


GAP_GAMES, GAP_TURNS = 16, 300


def gap_job(args) -> tuple:
    """(spec, games, turns, seed): self-play from the viewer's "neutral" start with A x1.76 stronger (initialized power
    gap; natural gaps are ~0.1% of decisions): M4.4's zoning metrics (power classes, credible threat on the weaker
    side, the stronger side's pressure) and the state-conditioned table, both seats."""
    spec, games, turns, seed = args
    import torch
    torch.set_num_threads(1)
    from ..m44 import metrics as M
    from ..m44.records import play_record
    from . import behavior as BH
    from .common import agent_from_spec, env_cfg
    from .traces import _gap_start
    cfg = env_cfg()
    x = agent_from_spec(spec, cfg)
    start = _gap_start()
    recs = [play_record(cfg, (x, x), np.random.SeedSequence(seed, spawn_key=(g,)), turns, start, "neutral/large",
                        "m45c") for g in range(games)]
    return spec, {"games": games, "turns": turns, "start": "viewer neutral, gap large (A x1.76)",
                  "zoning": M.zoning(recs), "conditional": BH.conditional(recs)}


def behavior(workers: int, seeds: Sequence[int] = FULL_SEEDS, camp: Camp = MAIN) -> dict:
    from . import behavior as BH
    t0 = time.perf_counter()
    rj, lj = [], []
    for s in seeds:
        f, m = spec(s, "final", camp), spec(s, camp.mid, camp)
        rj += [(f, "self", 16, 1000, BEHAVIOR_SEED_F), (m, "self", 16, 1000, BEHAVIOR_SEED_F),
               (f, "self", 64, 120, BEHAVIOR_SEED_F + 10)]
        rj += [(f, o, 8, 500, BEHAVIOR_SEED_F + 1) for o in ("ref:g_bully", "ref:fz_guard_sh")]
        lj += [(f, "self", LEDGER_SEEDS_F[:48], 300), (m, "self", LEDGER_SEEDS_F[:48], 300)]
    # the early-game job shares its (spec, opp) key with the 1000-turn self-play job: key the results by turns too
    rec = {}
    for (k, v), j in zip(run_jobs(BH.records_job, rj, workers, "health records"), rj):
        rec[f"{k[0]}|{k[1]}|{j[3]}"] = v
    gap = dict(run_jobs(gap_job, [(spec(s, "final", camp), GAP_GAMES, GAP_TURNS, BEHAVIOR_SEED_F + 20) for s in seeds],
                        workers, "health power gap"))
    data = {"meta": meta(stage="health_behavior", phase=camp.phase, seeds=list(seeds)), "records": rec,
            "ledger": BH.run_ledger(lj, workers), "power_gap": gap, "elapsed_s": time.perf_counter() - t0}
    save_json("health_behavior", data, camp.out)
    return data


# ---------------------------------------------------------------------------------------------- training histories
TRAIN_KEYS = ("entropy", "approx_kl", "clipfrac", "explained_variance", "grad_norm", "grad_norm_max", "value_loss",
              "policy_loss", "value_mean", "return_std", "lr", "sps", "mask_n_legal_mean", "mask_ward_legal_share",
              "mask_recall_legal_share", "mask_pressure_legal_share", "act_cat_LH", "act_cat_TN", "act_cat_SL",
              "act_cat_SH", "act_cat_Ward", "act_cat_Recall", "act_cat_Absent", "act_beh_Hold", "act_beh_Yield",
              "act_beh_Trade", "act_beh_Safe", "act_pressure_of_lane", "adv_std_A", "adv_std_B")
EP_RATES = {"deaths_per_1000_champion_turns": lambda e: 1000 * e["ep_deaths"] / max(2 * e["ep_turns"], 1),
            "recalls_per_1000_champion_turns": lambda e: 1000 * e["ep_recalls"] / max(2 * e["ep_turns"], 1),
            "recall_interrupted_share": lambda e: e["ep_recalls_interrupted"] / max(e["ep_recalls"] + e["ep_recalls_interrupted"], 1),
            "trade_connect_rate": lambda e: e["ep_trades_connected"] / max(e["ep_trades_attempted"], 1),
            "trades_per_1000_champion_turns": lambda e: 1000 * e["ep_trades_attempted"] / max(2 * e["ep_turns"], 1),
            "ganks_per_1000_turns": lambda e: 1000 * e["ep_ganks"] / max(e["ep_turns"], 1),
            "farm_collection": lambda e: e["ep_farm"] / max(e["ep_farm_available"], 1e-9),
            "plates_per_1000_turns": lambda e: 1000 * e["ep_plates"] / max(e["ep_turns"], 1),
            "final_level": lambda e: e["ep_level_sum"] / max(e["ep_level_n"], 1)}


def _tenths(h: List[dict], f) -> List[float]:
    n = len(h)
    return [float(np.nanmean([f(e) for e in h[i * n // 10:(i + 1) * n // 10]])) for i in range(10)]


def training(seeds: Sequence[int] = FULL_SEEDS, camp: Camp = MAIN) -> dict:
    """Per seed: every TRAIN_KEYS quantity and the episode rates by tenth of the run, and curves every 10 updates."""
    from .full import _history, stability_scan
    out = {"meta": meta(stage="health_training", phase=camp.phase, seeds=list(seeds)), "seeds": {}}
    for s in seeds:
        h = _history(camp.phase, s, "")
        if not h:
            continue
        games = [e for e in h if e["ep_games"] > 0]
        row = {"tenths": {k: _tenths(h, lambda e, k=k: e.get(k, np.nan)) for k in TRAIN_KEYS},
               "episode_tenths": {k: _tenths(games, f) for k, f in EP_RATES.items()},
               "curve_every": 10,
               "curve": {k: [float(np.nanmean([e.get(k, np.nan) for e in h[i:i + 10]])) if k != "agent_transitions"
                             else float(h[min(i + 9, len(h) - 1)][k]) for i in range(0, len(h), 10)]
                         for k in ("agent_transitions", *TRAIN_KEYS)},
               "stability": stability_scan(h)}
        out["seeds"][str(s)] = row
    save_json("health_training", out, camp.out)
    return out


__all__ = ["Camp", "MAIN", "EXT", "CAMPS", "CHECKPOINTS", "LATE_QUARTER", "PARTS", "COMPREHENSIVE", "GRID_POINTS", "TOURNAMENT", "spec", "opponents", "curve_jobs", "h2h_jobs", "history_jobs",
           "cycle_jobs", "cross_jobs", "evaluate", "behavior", "training", "turns_of", "EVAL_NAME"]
