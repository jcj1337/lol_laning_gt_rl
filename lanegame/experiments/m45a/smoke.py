"""M4.5a smoke checks (descriptive; NOT the M4.5b calibration): does every new mechanic fire, at what rates, at what
cost in runtime, and with every invariant intact?

    random      uniform legal play on m45_unified (invariants and the observation decoder every turn), event rates
    scripted    a few four-action strategy pairings, paired seeds, on m4_four_actions_econ_aggro, m45_unified_r1
                (M4.5a as first implemented) and m45_unified (the Ward strategies play their inner strategy where
                Ward is not legal): gold lead, deaths, Recalls, minion damage, wards; one window per seed from the
                named start
    throughput  turns per second of uniform play, aggro versus m45 r1 and r2 (one process)
    damage      the minion damage table D(m, L) / H(L) at the default parameters, with and without investment

Seeds: SeedSequence roots 85,000-85,999 (disjoint from every earlier study).
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from ...champions_m4 import check_invariants, context
from ...config import GameConfig
from ...env import LaneEnv
from ...env.observation_m4 import decode_m4
from ...positional import WARD, minion_damage_centi

SEED = 85_000
PAIRINGS = (
    ("farm4", "farm4", "s0", "equal"),
    ("ward_favorable4", "farm4", "s0", "equal"),
    ("ward_always4", "bully4", "neutral", "equal"),
    ("bully4", "farm4", "s0", "equal"),
    ("bully4", "step_back4", "neutral", "equal"),
    ("freeze_bully4", "walk_up_shove4", "freeze", "large"),
    ("ward_freeze4", "walk_up_shove4", "freeze", "large"),
)


def _ci(v) -> List[float]:
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return [float(v.mean()) if len(v) else float("nan"), float("nan")]
    return [float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))]


def random_soak(games: int = 40, turns: int = 300) -> dict:
    cfg = GameConfig.m45_unified()
    env = LaneEnv(cfg, max_turns=turns)
    rng = np.random.default_rng(SEED)
    c = {k: 0 for k in ("turns", "ward", "placed", "interrupted", "retaliation_ticks", "positional_ticks",
                        "minion_hit_turns", "minion_centi", "capped", "connected", "landed", "at_w", "deaths")}
    by_level: Dict[int, List[float]] = {}
    t = context(cfg).t
    for g in range(games):
        env.reset(seed=SEED + g)
        while not env.done:
            lv = [t.level_by_xp[min(env.champ.side(s).xp, t.xp_cap)] for s in (0, 1)]
            r = env.step(int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1))))
            check_invariants(env.champ, cfg)
            i = r.info
            c["turns"] += 1
            for s in (0, 1):
                ph, ln, m4, _ = decode_m4(r.obs[s].vector, cfg)
                assert m4.a.pos == env.champ.side(s).pos
                c["ward"] += i["ward"][s]
                c["placed"] += i["ward_placed"][s]
                c["interrupted"] += i["ward_interrupted"][s]
                c["retaliation_ticks"] += sum(x == "retaliation" for x in i["minion_fire"][s])
                c["positional_ticks"] += sum(x == "positional" for x in i["minion_fire"][s])
                c["minion_hit_turns"] += i["minion_hits"][s] > 0
                c["minion_centi"] += i["minion_damage"][s]
                c["capped"] += i["trade_wave_capped"][s]
                c["connected"] += i["trade_connected"][s]
                c["landed"] += i["trade_damage"][s] > 0
                c["at_w"] += env.champ.side(s).pos == WARD
                c["deaths"] += i["died"][s]
                if i["minion_hits"][s]:
                    by_level.setdefault(lv[s], []).append(i["minion_damage"][s] / t.hp_max[lv[s]])
    n = 2 * c["turns"]
    rates = {k: v / n for k, v in c.items() if k != "turns"}
    lev = {L: {"turns": len(v), "mean_share_of_bar": float(np.mean(v))} for L, v in sorted(by_level.items())}
    return {"games": games, "turns": turns, "champion_turns": n, "counts": c, "per_champion_turn": rates,
            "minion_damage_share_by_level": lev}


def scripted(seeds: int = 16, turns: int = 200) -> dict:
    from ...viewer import policies as pl
    from ...viewer import session as ss
    out = {}
    for preset in ("m4_four_actions_econ_aggro", "m45_unified_r1", "m45_unified"):
        rows = {}
        for pa, pb, scen, gap in PAIRINGS:
            per = {k: [] for k in ("lead", "deaths_A", "deaths_B", "recalls_A", "recalls_B", "minion_cp_A",
                                   "minion_cp_B", "wards_A", "wards_placed_A", "ret_ticks_A", "pos_ticks_A",
                                   "ret_ticks_B", "pos_ticks_B")}
            for k in range(seeds):
                rec = ss.play(ss.Setup(preset, scen, gap, SEED + 100 + k), policies=(pl.POLICIES[pa], pl.POLICIES[pb]),
                              turns=turns)
                inf = [t.info for t in rec.turns]
                sc = rec.cfg.gold_scale
                per["lead"].append(sum(i["earned_units"][0] - i["earned_units"][1] for i in inf) / sc)
                for s, n in ((0, "A"), (1, "B")):
                    per[f"deaths_{n}"].append(sum(i["died"][s] for i in inf))
                    per[f"recalls_{n}"].append(sum(i["recalled"][s] for i in inf))
                    per[f"minion_cp_{n}"].append(sum(i.get("minion_damage", (0, 0))[s] for i in inf))
                    per[f"ret_ticks_{n}"].append(sum(x == "retaliation" for i in inf for x in (i.get("minion_fire") or
                                                                                              ((), ()))[s]))
                    per[f"pos_ticks_{n}"].append(sum(x == "positional" for i in inf for x in (i.get("minion_fire") or
                                                                                             ((), ()))[s]))
                per["wards_A"].append(sum(i.get("ward", (0, 0))[0] for i in inf))
                per["wards_placed_A"].append(sum(i.get("ward_placed", (0, 0))[0] for i in inf))
            rows[f"{pa} v {pb} [{scen}, {gap}]"] = {k: _ci(v) for k, v in per.items()}
        out[preset] = rows
    return {"seeds": seeds, "turns": turns, "rows": out}


def throughput(turns: int = 3000) -> dict:
    out = {}
    for name in ("m4_four_actions_econ_aggro", "m45_unified_r1", "m45_unified"):
        cfg = getattr(GameConfig, name)()
        env = LaneEnv(cfg, max_turns=turns)
        env.reset(seed=SEED)
        rng = np.random.default_rng(SEED)
        t0 = time.perf_counter()
        while not env.done:
            env.step(int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1))))
        out[name] = turns / (time.perf_counter() - t0)
    return {"turns_per_second": out, "ratio": out["m45_unified"] / out["m4_four_actions_econ_aggro"]}


def damage_table() -> dict:
    cfg = GameConfig.m45_unified()
    ctx = context(cfg)
    t = ctx.t
    rows = []
    for L, S in ((1, 0), (2, 0), (4, 0), (6, 8), (11, 32), (15, 64), (18, 128), (18, 300)):
        items = S * cfg.gold_scale
        row = {"level": L, "invested": S, "hp_max": t.hp_max[L]}
        for m in (1, 2, 3, 4, 6):
            cp = ctx.mitigate(minion_damage_centi(m, L, cfg.minion_damage_alpha, cfg.minion_damage_growth), items)
            row[f"m{m}"] = [cp, round(cp / t.hp_max[L], 4)]
        rows.append(row)
    return {"alpha": cfg.minion_damage_alpha, "growth": cfg.minion_damage_growth, "rows": rows,
            "note": "m = minion damage units (one tick of a blob deals ceil(f/2) or floor(f/2), f = max(1, floor(2h/8)) "
                    "per turn: an 8-HP wave 1 + 1, 16 HP 2 + 2, 24 HP 3 + 3); entries [centi-HP after durability, "
                    "share of the max-HP bar]"}


__all__ = ["random_soak", "scripted", "throughput", "damage_table", "PAIRINGS", "SEED"]
