"""M4.3c stabilization, Stage B probes (docs/m4/m4_3/m4_3c_strategic_economics_calibration.md): the value of a power lead,
the cost of a death, and the laning horizon. Same conventions as `economics` (four-action preset and its VARIANTS,
the real environment, scripted strategies, paired seeds; nothing here changes a preset).

    power    A starts with an investment lead of S resources (B none) or a level lead; both seats play the same
             strategy for TURNS turns from the neutral lane at level 10. A's discounted advantage J(S) - J(0) is the
             future income the lead buys: a TRANSIENT gain is gone with the lead, a PERSISTENT one keeps earning.
    death    from a state, B alive versus B just killed (absent for d turns, then a normal return: full HP and all
             unspent gold invested); the paired difference of J (plus the bounty the killer received) is the price of
             the death. d = 1, 3, 5, 7; the observation scale champion_absent_max is raised to 7 for every cell, which
             changes nothing for scripted agents.
    horizon  long games from s0 (HORIZON_TURNS turns): the turns of each plate, the level and earned gold at those
             turns, both champions' level-18 turn, and the shares of farm, plate gold and bounty in the reward.

    python -m lanegame.experiments.m43c econ_probes [--variants base k2 ...]   -> results/m43c/econ_probes_<variant>.json.gz
"""

from __future__ import annotations

import time
from typing import Dict, Sequence

import numpy as np

from . import harness as h
from . import strategies as st
from .common import mean_ci, meta, save_json
from .economics import VARIANTS, variant_cfg

PROBE_SEED = 77_100
EPISODES = 64
TURNS = 100
POWER_LEADS = (0, 2, 5, 10, 20, 50)              # resources invested by A (B none)
LEVEL_LEADS = (0, 1, 2, 3)
POWER_PAIRS = ("farm4", "shove4", "shove_hunt4", "hunt4")      # both seats play the same strategy
DEATH_TURNS = (1, 3, 5, 7)
# (label, start spec (base, gap, level), A strategy, B strategy)
DEATH_STATES = (("neutral L6, both Shove", ("neutral", "equal", 6), "shove4", "shove4"),
                ("neutral L10, both Shove", ("neutral", "equal", 10), "shove4", "shove4"),
                ("freeze L10, A freezes, B walks up", ("freeze", "equal", 10), "freeze4", "walk_up4"),
                ("freeze L10, A freezes, B breaks", ("freeze", "equal", 10), "freeze4", "walk_up_shove4"),
                ("freeze L10 A x1.76, A hunts, B walks up", ("freeze", "large", 10), "freeze_hunt4", "walk_up4"))
HORIZON_TURNS = 300
HORIZON_PAIRS = (("farm4", "farm4"), ("shove4", "shove4"), ("shove4", "farm4"), ("slowpush4", "thin4"),
                 ("shove_hunt4", "shove_hunt4"), ("grow_shove4", "farm4"))


def _agents(a: str, b: str):
    return lambda: (st.StrategyAgent(st.LIBRARY[a]), st.StrategyAgent(st.LIBRARY[b]))


# ---------------------------------------------------------------------------------------------- power
def _power_job(args) -> tuple:
    vname, kind, lead, pair, n, turns = args
    cfg = variant_cfg(vname)
    lane, m4, k = h.start("neutral", cfg, level=10)
    t = __import__("lanegame.champions_m4", fromlist=["context"]).context(cfg).t
    if kind == "invest":
        m4 = m4.replace_side(0, items=int(lead * cfg.gold_scale))
    else:
        L = 10 + lead
        m4 = m4.replace_side(0, xp=t.xp_cum[L], hp=t.hp_max[L])
    games = h.cell(cfg, (lane, m4, k), _agents(pair, pair), PROBE_SEED, n, turns, windows=(40,))
    return f"{kind}|{lead}|{pair}", {"G": [g["all"]["G"] for g in games], "X": [g["all"]["X"] for g in games],
                                    "G40": [g["40"]["earned"][0] - g["40"]["earned"][1] for g in games],
                                    "duel_end": [g["all"]["duel_ratio_end"] for g in games],
                                    "inv_end": [[g["all"]["invested_end"][s] for s in (0, 1)] for g in games]}


def power(vname: str, workers: int, n: int = EPISODES, turns: int = TURNS) -> dict:
    from ..rl.m43b.common import run_jobs
    jobs = [(vname, "invest", s, p, n, turns) for s in POWER_LEADS for p in POWER_PAIRS]
    jobs += [(vname, "levels", L, p, n, turns) for L in LEVEL_LEADS for p in POWER_PAIRS]
    res = dict(run_jobs(_power_job, jobs, workers, f"power {vname}"))
    out = {}
    for key, r in res.items():
        kind, lead, pair = key.split("|")
        base = res[f"{kind}|0|{pair}"]
        out[key] = {"J": mean_ci(r["G"]), "J_gain": mean_ci([a - b for a, b in zip(r["G"], base["G"])]),
                    "Jx_gain": mean_ci([a + x / 8 - b - y / 8 for a, x, b, y in zip(r["G"], r["X"], base["G"],
                                                                                     base["X"])]),
                    "G40_gain": mean_ci([a - b for a, b in zip(r["G40"], base["G40"])]),
                    "duel_end": mean_ci(r["duel_end"]),
                    "inv_end": [mean_ci([x[s] for x in r["inv_end"]]) for s in (0, 1)]}
    return out


# ---------------------------------------------------------------------------------------------- death
def _death_job(args) -> tuple:
    vname, label, spec, a, b, d, n, turns = args
    cfg = variant_cfg(vname).with_(champion_absent_max=max(7, variant_cfg(vname).champion_absent_max))
    base, gap, level = spec
    lane, m4, k = h.start(base, cfg, gap=gap, level=level)
    if d > 0:
        m4 = m4.replace_side(1, absent=d, pos=-cfg.tower)
    games = h.cell(cfg, (lane, m4, k), _agents(a, b), PROBE_SEED + 1, n, turns, windows=(40,))
    return f"{label}|{d}", {"G": [g["all"]["G"] for g in games], "X": [g["all"]["X"] for g in games],
                            "G40": [g["40"]["earned"][0] - g["40"]["earned"][1] for g in games],
                            "plates_a": [g["all"]["plates_n"][0] for g in games],
                            "deaths_b": [g["all"]["died"][1] for g in games]}


def death(vname: str, workers: int, n: int = EPISODES, turns: int = TURNS) -> dict:
    """Price of a death of B (to B; positive = a loss): J(alive) - J(dead) with J from B's view, plus the bounty."""
    from ..rl.m43b.common import run_jobs
    jobs = [(vname, lab, spec, a, b, d, n, turns) for lab, spec, a, b in DEATH_STATES for d in (0,) + DEATH_TURNS]
    res = dict(run_jobs(_death_job, jobs, workers, f"death {vname}"))
    bounty = variant_cfg(vname).kill_bounty
    out = {}
    for lab, spec, a, b in DEATH_STATES:
        alive = res[f"{lab}|0"]
        for d in DEATH_TURNS:
            dead = res[f"{lab}|{d}"]
            # J is A's value; B's loss from its own death = J_A(dead) - J_A(alive) + bounty (A's bounty is not in
            # the start state, so it is added explicitly)
            out[f"{lab}|{d}"] = {
                "cost": mean_ci([x - y + bounty for x, y in zip(dead["G"], alive["G"])]),
                "cost_x": mean_ci([x + xx / 8 - y - yx / 8 + bounty for x, xx, y, yx in
                                   zip(dead["G"], dead["X"], alive["G"], alive["X"])]),
                "cost40": mean_ci([x - y + bounty for x, y in zip(dead["G40"], alive["G40"])]),
                "plates_a_gain": mean_ci([x - y for x, y in zip(dead["plates_a"], alive["plates_a"])])}
    return out


# ---------------------------------------------------------------------------------------------- horizon
def _horizon_job(args) -> tuple:
    vname, a, b, n, turns = args
    cfg = variant_cfg(vname)
    from ...champions_m4 import initial_m4
    from ...mechanics import initial_state
    stt = (initial_state(cfg), initial_m4(cfg), 0)
    games = []
    for ss in h.seeds(PROBE_SEED + 2, n):
        g = h.play(cfg, stt, _agents(a, b)(), ss, turns)
        rows = g["rows"]
        rec = {"turns": len(rows), "plate_turns": [[], []], "plate_levels": [[], []], "plate_earned": [[], []],
               "level18": [None, None]}
        cum = [0.0, 0.0]
        tow = [0, 0]
        for i, r in enumerate(rows):
            for s in (0, 1):
                cum[s] += r["earned"][s]
                if r["plates_n"][s]:
                    for _ in range(r["plates_n"][s]):
                        rec["plate_turns"][s].append(i + 1)
                        rec["plate_levels"][s].append(r["level"][s])
                        rec["plate_earned"][s].append(cum[s])
                if rec["level18"][s] is None and r["level"][s] >= cfg.level_cap:
                    rec["level18"][s] = i + 1
        w = h.window(g, cfg)
        rec.update({"farm": w["farm"], "plate": w["plate"], "bounty": w["bounty"], "earned": w["earned"],
                    "G": w["G"], "X": w["X"], "died": w["died"], "recalled": w["recalled"]})
        games.append(rec)
    return f"{a}|{b}", games


def horizon(vname: str, workers: int, n: int = 32, turns: int = HORIZON_TURNS) -> dict:
    from ..rl.m43b.common import run_jobs
    jobs = [(vname, a, b, n, turns) for a, b in HORIZON_PAIRS]
    res = dict(run_jobs(_horizon_job, jobs, workers, f"horizon {vname}"))
    out = {}
    nplates = len(variant_cfg(vname).plate_thresholds)
    for key, games in res.items():
        d = {"games": len(games), "turns": float(np.mean([g["turns"] for g in games]))}
        for j in range(nplates):
            # the j-th plate to fall on either tower (pooled over both attackers)
            tj = [sorted(g["plate_turns"][0] + g["plate_turns"][1]) for g in games]
            lv = [sorted(zip(g["plate_turns"][0] + g["plate_turns"][1], g["plate_levels"][0] + g["plate_levels"][1]))
                  for g in games]
            vals = [x[j] for x in tj if len(x) > j]
            lvs = [x[j][1] for x in lv if len(x) > j]
            d[f"plate{j + 1}_turn"] = (float(np.median(vals)) if vals else None, len(vals) / len(games))
            d[f"plate{j + 1}_level"] = float(np.mean(lvs)) if lvs else None
        last = [min((g["plate_turns"][s][nplates - 1] for s in (0, 1) if len(g["plate_turns"][s]) >= nplates),
                    default=None) for g in games]
        got = [x for x in last if x is not None]
        d["final_plate_first_tower"] = (float(np.median(got)) if got else None, len(got) / len(games))
        lv18 = [max(g["level18"]) if None not in g["level18"] else None for g in games]
        got18 = [x for x in lv18 if x is not None]
        d["both_level18"] = (float(np.median(got18)) if got18 else None, len(got18) / len(games))
        for k in ("farm", "plate", "bounty"):
            d[f"{k}_share"] = float(np.sum([g[k][0] + g[k][1] for g in games]) /
                                    np.sum([g["earned"][0] + g["earned"][1] for g in games]))
        d["G"] = mean_ci([g["G"] for g in games])
        d["deaths"] = [float(np.mean([g["died"][s] for g in games])) for s in (0, 1)]
        d["recalls"] = [float(np.mean([g["recalled"][s] for g in games])) for s in (0, 1)]
        out[key] = d
    return out


def run(variants: Sequence[str] = ("base",), workers: int = 9, stages=("power", "death", "horizon")) -> Dict[str, dict]:
    out = {}
    for v in variants:
        t0 = time.perf_counter()
        res = {"meta": meta(stage="econ_probes", variant=v, description=VARIANTS[v][0],
                            preset="GameConfig.m4_four_actions()", seed_root=PROBE_SEED, episodes=EPISODES,
                            turns=TURNS)}
        for name in stages:
            res[name] = {"power": power, "death": death, "horizon": horizon}[name](v, workers)
        res["meta"]["elapsed_s"] = time.perf_counter() - t0
        save_json(f"econ_probes_{v}", res)
        out[v] = res
    return out


__all__ = ["run", "power", "death", "horizon"]
