"""Part 1 of the M4.3d study: the risk and collection rules as the engine computes them (no rollouts).

Every number is produced by the engine's own functions (champions.champion_gank_probability, gank_damage_p,
gank_damage_distribution, trade_hit_probability, champions_m4.collection_units) on the given configuration. The
hazard table is a small Monte Carlo of the gank and tower-shot rules alone (no trades, no lane): a champion that
stands at one cell every present turn, recalls when its HP falls to 40% (walking back with Safe, two cells per turn,
to its own -1 first), dies at 0, is away recall_turns turns after a Recall and death_turns(L) turns after a death.
It isolates what a position costs in HP, Recalls and deaths; the economic cost under play is measured in the grid.

    python -m lanegame.experiments.m43d audit   -> results/m43d/audit.{json,md}
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ...champions import (champion_gank_probability, gank_damage_distribution, gank_damage_p, trade_hit_probability)
from ...champions_m4 import collection_units, context
from ...config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, W_HOLD, W_SHOVE, GameConfig
from ...mechanics import initial_state
from . import common as c

CELLS = (-2, -1, 0, 1, 2)
OTS = (0.6, 0.8, 0.85, 0.9)


def _lane(cfg: GameConfig, vision: bool):
    lane = initial_state(cfg)._replace(ah=8, bh=8, ax=0, bx=0)      # one wave each: no stack difference
    return lane._replace(va=cfg.vision_turns if vision else 0)


def gank_table(cfg: GameConfig) -> List[dict]:
    """P(ganked) per turn by exposure cell, behaviour / wave and vision (start = end cell)."""
    rows = []
    combos = (("Hold, Last Hit", B_HOLD, W_HOLD), ("Hold, Shove", B_HOLD, W_SHOVE), ("Trade, Shove", B_TRADE, W_SHOVE),
              ("Trade, Last Hit", B_TRADE, W_HOLD), ("Yield, Last Hit", B_YIELD, W_HOLD),
              ("Safe (any wave)", B_SAFE, W_SHOVE))
    for name, beh, w in combos:
        for vis in (False, True):
            row = {"behaviour": name, "vision": vis}
            for e in CELLS:
                row[e] = champion_gank_probability(_lane(cfg, vis), 0, w, beh, cfg, e)
            rows.append(row)
    return rows


def gank_damage_table(cfg: GameConfig) -> List[dict]:
    """Per exposure cell: p_e, E[units], P(4 units) and the expected damage in bars (M = 1), normal / defensive."""
    rows = []
    for defensive in (False, True):
        for e in CELLS:
            pe = gank_damage_p(0, defensive, e, cfg)
            dist = gank_damage_distribution(pe, cfg)
            ex = sum(d * p for d, p in enumerate(dist))
            rows.append({"defensive": defensive, "exposure": e, "p_e": pe, "E_units": ex, "P_max": dist[-1],
                         "E_bars": ex / 6.0, "P_ge3": sum(dist[3:])})
    return rows


def expected_gank_bars(cfg: GameConfig) -> List[dict]:
    """Expected gank damage per turn in bars (occurrence x E[damage] / 6), M = 1, by cell and behaviour."""
    out = []
    for row in gank_table(cfg):
        defensive = row["behaviour"].startswith("Safe")
        r = {"behaviour": row["behaviour"], "vision": row["vision"]}
        for e in CELLS:
            pe = gank_damage_p(0, defensive, e, cfg)
            r[e] = row[e] * cfg.gank_damage_trials * pe / 6.0
        out.append(r)
    return out


def trade_table(cfg: GameConfig) -> List[dict]:
    """P(a connected trade lands) on a defender by its behaviour, in or outside its tower cell (equal skill, no
    stack difference)."""
    lane = _lane(cfg, False)
    out = []
    for name, beh in (("Hold", B_HOLD), ("Trade", B_TRADE), ("Yield caught (evading)", B_YIELD), ("Safe", B_SAFE),
                      ("interrupted Recall", B_RECALL)):
        out.append({"defender": name, "outside tower": trade_hit_probability(lane, 0, beh, False, cfg),
                    "in its tower cell": trade_hit_probability(lane, 0, beh, True, cfg)})
    return out


def tower_table(cfg: GameConfig) -> Dict[str, float]:
    t = context(cfg).t
    return {"dive_p": cfg.tower_shot_p, "dive_p_busy": cfg.tower_shot_p_busy, "adjacent_p": cfg.tower_adjacent_p,
            "adjacent_p_busy": cfg.tower_adjacent_p_busy, "damage_units": cfg.tower_shot_damage,
            "damage_bars": cfg.tower_shot_damage * t.hp_unit[10] / t.hp_max[10]}


def collection_table(cfg: GameConfig) -> List[dict]:
    """Collection rate (resources per HP / farm_per_hp) of a kill in the own tower cell and at the own -1, by the
    collector's mode and cell, for each own-tower rate."""
    from ...champions import FARM, RETREAT, SAFE_MODE, TRADING
    out = []
    for ot in OTS:
        cf = cfg.with_(collect_contest_own_tower=ot)
        sc = cf.gold_scale * cf.farm_per_hp
        r = {"own_tower_rate": ot}
        for label, mode, pos, x in (("Hold at -2, kill at -2", FARM, -2, -2), ("Hold at -1, kill at -2", FARM, -1, -2),
                                    ("Safe at -2, kill at -2", SAFE_MODE, -2, -2),
                                    ("Safe at -1, kill at -2", SAFE_MODE, -1, -2),
                                    ("Safe at -1, kill at -1", SAFE_MODE, -1, -1),
                                    ("trading at -2, kill at -2", TRADING, -2, -2),
                                    ("retreat at -2, kill at -1", RETREAT, -2, -1),
                                    ("Hold at 0, kill at 0", FARM, 0, 0),
                                    ("Hold at +2, kill at +3", FARM, 2, 3)):
            r[label] = collection_units(x, mode, pos, FARM, 0, cf) / sc
        out.append(r)
    return out


# ---------------------------------------------------------------------------------------------- hazard Monte Carlo
def hazard(cfg: GameConfig, cell: int, beh: int, wave: int, vision: float, M: float, tower_p: float, level: int = 10,
           turns: int = 40, n: int = 40_000, seed: int = 0) -> dict:
    """Per `turns` turns: present turns at `cell`, ganks, HP lost (bars), forced Recalls, deaths, turns away, of a
    champion that stands at `cell` with behaviour `beh` and wave `wave` (module docstring). `vision` = the fraction of
    present turns with vision active; `tower_p` = P(one tower shot of one HP unit) per present turn at the cell (a dive
    pays 1.0, 0.5 while the tower is busy); M divides all damage taken (investment durability)."""
    rng = np.random.default_rng(seed)
    t = context(cfg).t
    unit = t.hp_unit[level] / t.hp_max[level]
    death_turns = context(cfg).death_turns(level)
    lane0, lane1 = _lane(cfg, False), _lane(cfg, True)
    p_novis = champion_gank_probability(lane0, 0, wave, beh, cfg, cell)
    p_vis = champion_gank_probability(lane1, 0, wave, beh, cfg, cell)
    defensive = beh == B_SAFE
    tot = dict(present=0, at_cell=0, ganks=0, bars=0.0, recalls=0, deaths=0, away=0)
    for _ in range(n):
        hp, away, walking, pos = 1.0, 0, False, cell
        for _k in range(turns):
            if away > 0:
                away -= 1
                tot["away"] += 1
                if away == 0:
                    hp, pos, walking = 1.0, cell, False
                continue
            tot["present"] += 1
            if walking and pos <= -1:                    # Recall from the own -1 or behind: gone this turn
                tot["recalls"] += 1
                away = cfg.recall_turns - 1
                tot["away"] += 1
                if away == 0:
                    hp, pos, walking = 1.0, cell, False
                continue
            if walking:                                  # walk back with Safe; exposure = max(start, end)
                start, pos = pos, max(-1, pos - cfg.champion_speed)
                e, b, def_ = start, B_SAFE, True
            else:
                e, b, def_ = cell, beh, defensive
                tot["at_cell"] += 1
            lane = lane1 if rng.random() < vision else lane0
            pg = (p_vis if lane is lane1 else p_novis) if (e == cell and b == beh) else \
                champion_gank_probability(lane, 0, wave, b, cfg, e)
            dmg = 0.0
            if rng.random() < pg:
                tot["ganks"] += 1
                pe = gank_damage_p(0, def_, e, cfg)
                dmg += rng.binomial(cfg.gank_damage_trials, pe) * unit
            if not walking and tower_p > 0 and rng.random() < tower_p:
                dmg += cfg.tower_shot_damage * unit
            dmg /= M
            hp -= dmg
            tot["bars"] += dmg
            if hp <= 1e-12:
                tot["deaths"] += 1
                away = death_turns
                continue
            if not walking and hp <= 0.4 + 1e-12:
                walking = True
                if pos <= -1:
                    continue
    return {k: v / n for k, v in tot.items()}


HAZARD_CASES = (
    ("own tower cell, Hold, Last Hit", -2, B_HOLD, W_HOLD, 0.0, 0.0),
    ("own -1, Hold, Last Hit", -1, B_HOLD, W_HOLD, 0.0, 0.0),
    ("lane cell, Hold, Last Hit", 0, B_HOLD, W_HOLD, 0.0, 0.0),
    ("lane cell, Hold, Shove", 0, B_HOLD, W_SHOVE, 0.0, 0.0),
    ("enemy -1 (own +1), Hold, Shove", 1, B_HOLD, W_SHOVE, 0.0, 0.0),
    ("enemy tower cell, Hold, Shove (Pressure)", 2, B_HOLD, W_SHOVE, 0.0, 0.0),
    ("enemy tower cell, Hold, Shove, Vision always", 2, B_HOLD, W_SHOVE, 1.0, 0.0),
    ("enemy tower cell, Trade, Shove (dive, tower busy)", 2, B_TRADE, W_SHOVE, 0.0, 0.5),
    ("enemy tower cell, Trade, Shove (dive, tower free)", 2, B_TRADE, W_SHOVE, 0.0, 1.0),
    ("enemy tower cell, dive (busy), Vision always", 2, B_TRADE, W_SHOVE, 1.0, 0.5),
)


def hazard_table(cfg: GameConfig, M: float = 1.0, n: int = 40_000) -> List[dict]:
    out = []
    for i, (name, cell, beh, wave, vis, tp) in enumerate(HAZARD_CASES):
        r = hazard(cfg, cell, beh, wave, vis, M, tp, n=n, seed=c.TRACE_SEED + i)
        out.append({"case": name, "M": M, **{k: round(v, 4) for k, v in r.items()}})
    return out


def run(variants=("econ",), n: int = 40_000) -> dict:
    res = {"meta": c.meta(stage="audit", n=n)}
    for v in variants:
        cfg = c.variant_cfg(v)
        res[v] = {"gank": gank_table(cfg), "gank_damage": gank_damage_table(cfg),
                  "expected_gank_bars": expected_gank_bars(cfg), "trade": trade_table(cfg),
                  "tower": tower_table(cfg), "collection": collection_table(cfg),
                  "hazard_M1": hazard_table(cfg, 1.0, n), "hazard_M176": hazard_table(cfg, 1.7558, n)}
    c.save_json("audit", res)
    return res


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def report(res: dict, variant: str = "econ") -> str:
    d = res[variant]
    L = [f"# M4.3d audit ({variant}: {c.VARIANTS[variant][0]})", ""]
    L += ["## Gank occurrence per turn (exposure cell, own coordinates)", "",
          "| behaviour | vision | -2 | -1 | 0 | +1 | +2 |", "|---|---|---|---|---|---|---|"]
    for r in d["gank"]:
        L.append(f"| {r['behaviour']} | {'yes' if r['vision'] else 'no'} | " +
                 " | ".join(_fmt(r[k]) for k in (-2, -1, 0, 1, 2) if k in r or str(k) in r) + " |")
    L += ["", "## Gank damage given a gank (Binomial(4, p_e) HP units of 1/6 bar)", "",
          "| defensive | exposure | p_e | E[units] | P(4 units) | P(>=3) | E[bars] |", "|---|---|---|---|---|---|---|"]
    for r in d["gank_damage"]:
        L.append(f"| {r['defensive']} | {r['exposure']} | {r['p_e']:.2f} | {r['E_units']:.2f} | {r['P_max']:.3f} | "
                 f"{r['P_ge3']:.3f} | {r['E_bars']:.3f} |")
    L += ["", "## Expected gank damage per turn, bars (M = 1)", "",
          "| behaviour | vision | -2 | -1 | 0 | +1 | +2 |", "|---|---|---|---|---|---|---|"]
    for r in d["expected_gank_bars"]:
        L.append(f"| {r['behaviour']} | {'yes' if r['vision'] else 'no'} | " +
                 " | ".join(f"{r[k]:.3f}" for k in (-2, -1, 0, 1, 2)) + " |")
    L += ["", "## Trade hit probability on a defender", "", "| defender | outside its tower | in its tower cell |",
          "|---|---|---|"]
    for r in d["trade"]:
        L.append(f"| {r['defender']} | {r['outside tower']:.2f} | {r['in its tower cell']:.2f} |")
    tw = d["tower"]
    L += ["", f"## Tower shots: dive {tw['dive_p']} ({tw['dive_p_busy']} while the diver's minions are under the "
              f"tower), next to the tower {tw['adjacent_p']} ({tw['adjacent_p_busy']}); {tw['damage_units']} HP unit "
              f"= {tw['damage_bars']:.3f} bars; only a connected trader is ever shot", ""]
    L += ["## Collection rate of a kill (share of farm_per_hp)", ""]
    keys = [k for k in d["collection"][0] if k != "own_tower_rate"]
    L += ["| own-tower rate | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for r in d["collection"]:
        L.append(f"| {r['own_tower_rate']} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
    for tag, title in (("hazard_M1", "M = 1 (no investment)"), ("hazard_M176", "M = 1.76 (50 invested)")):
        L += ["", f"## Hazard of standing at one cell, per 40 turns, level 10, {title}", "",
              "| case | turns at the cell | ganks | bars lost | forced Recalls | deaths | turns away |",
              "|---|---|---|---|---|---|---|"]
        for r in d[tag]:
            L.append(f"| {r['case']} | {r['at_cell']:.1f} | {r['ganks']:.2f} | {r['bars']:.2f} | {r['recalls']:.2f} | "
                     f"{r['deaths']:.2f} | {r['away']:.1f} |")
    return "\n".join(L)


__all__ = ["run", "report", "gank_table", "gank_damage_table", "hazard", "hazard_table", "collection_table"]
