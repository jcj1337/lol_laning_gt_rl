"""Closed-form mechanics behind the M4.4c findings: exact one-turn resolutions (`resolve_turn_m4` without chance) and
the formulas of the reference, evaluated on the three presets. No rollouts, no policies.

    recall_table      Recall / Safe / Hold / Yield of a damaged champion against an enemy Trade or Hold, by the two
                      champions' cells: interrupted, connected, hit probability, end cells, the trader's tower-shot
                      probability (the guessing game of Part 2)
    trade_table       hit probability by the defender's behaviour and cell; expected damage of a connected trade in bars
                      of the victim's max HP by power gap and level (Part 4)
    collection_table  gold per minion HP by mode and cell (what a threatened defender gives up)
    vision_table      gank occurrence and expected gank damage per turn at the enemy tower with and without vision,
                      Pressure tower damage per turn by level and investment, turns per plate (Part 5)
"""

from __future__ import annotations

from typing import Dict, List

from ....champions import champion_gank_probability, gank_damage_p, trade_hit_probability
from ....champions_m4 import collection_units, context, resolve_turn_m4
from ....config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, PRESSURE, W_HOLD, W_SHOVE, champ_act)
from .common import ENV_ORDER, env_cfg
from .states import initialized

BEH = {"Hold": B_HOLD, "Yield": B_YIELD, "Trade": B_TRADE, "Safe": B_SAFE, "Recall": B_RECALL}


def recall_table(env: str = "aggro") -> List[dict]:
    """A at 30% HP at its own -1 or -2; B (full HP) at its own cell e; A's response x B's Trade / Hold."""
    cfg = env_cfg(env)
    rows = []
    for a_pos in (-1, -2):
        for e in (0, 1, 2):
            snap = initialized(cfg, "neutral", hp=(0.3, 1.0), pos=(a_pos, e))
            for mine in ("Recall", "Safe", "Hold", "Yield"):
                for theirs in ("Trade", "Hold"):
                    a = champ_act(W_HOLD, BEH[mine])
                    b = champ_act(W_HOLD, BEH[theirs])
                    t = resolve_turn_m4(snap.lane, snap.champ, snap.k, a, b, cfg, u=None)
                    rows.append({"a_pos": a_pos, "b_pos": e, "a": mine, "b": theirs, "a_end": t.end[0],
                                 "b_end": t.end[1], "recalled": bool(t.recalled[0]),
                                 "interrupted": bool(t.interrupted[0]), "connected": bool(t.connected[1]),
                                 "hit_p": float(t.hit_p[1]), "trader_tower_p": float(t.tower_p[1])})
    return rows


def trade_table(env: str = "aggro") -> dict:
    cfg = env_cfg(env)
    ctx = context(cfg)
    t = ctx.t
    snap = initialized(cfg, "neutral")
    hit = {}
    for name, b in (("Hold / Trade / interrupted Recall", B_HOLD), ("Yield (caught while evading)", None),
                    ("Safe (caught)", B_SAFE)):
        for tower in (False, True):
            if b is None:
                h = min(max(cfg.trade_hit_base + cfg.trade_hit_behavior[B_YIELD] - (cfg.trade_hit_tower if tower else 0),
                            cfg.trade_hit_min), cfg.trade_hit_max)
            else:
                h = trade_hit_probability(snap.lane, 0, b, tower, cfg)
            hit[f"{name}{' in its tower cell' if tower else ''}"] = h
    dmg = []
    from ...m4.zoning_natural import GAPS
    for level in (3, 10):
        for gap in ("equal", "moderate", "large", "large_levels"):
            s = initialized(cfg, "neutral", gap, level=level)
            la = t.level_by_xp[min(s.champ.a.xp, t.xp_cap)]
            lb = t.level_by_xp[min(s.champ.b.xp, t.xp_cap)]
            q = cfg.trade_double_base
            for beh_name, h in (("vs Hold", hit["Hold / Trade / interrupted Recall"]), ("vs Safe", hit["Safe (caught)"])):
                exp_centi = h * ((1 - q) * ctx.hit(1, la, s.champ.a.items) + q * ctx.hit(2, la, s.champ.a.items))
                taken = ctx.mitigate(int(round(exp_centi)), s.champ.b.items)
                dmg.append({"level": level, "gap": gap, "vs": beh_name, "a_level": la, "b_level": lb,
                            "expected_bars": taken / t.hp_max[lb],
                            "bars_per_unit_hit": ctx.mitigate(ctx.hit(1, la, s.champ.a.items), s.champ.b.items) / t.hp_max[lb]})
    return {"hit_probability": hit, "expected_damage": dmg, "trade_farm_factor": cfg.trade_farm_factor,
            "gank_behavior_trade": cfg.champ_gank_behavior[B_TRADE], "gank_behavior_hold": cfg.champ_gank_behavior[B_HOLD]}


def collection_table(env: str = "aggro") -> List[dict]:
    from ....champions import FARM, RETREAT, SAFE_MODE, TRADING
    cfg = env_cfg(env)
    rows = []
    for mode, name in ((FARM, "farm (Hold / unthreatened Yield)"), (TRADING, "connected Trade"),
                       (SAFE_MODE, "Safe"), (RETREAT, "threatened Yield (retreat)")):
        row = {"mode": name}
        for x in (-2, -1, 0, 1, 2):
            pos = x if mode != SAFE_MODE else min(x, -1)
            row[str(x)] = collection_units(x, mode, pos, FARM, 0, cfg) / cfg.gold_scale
        rows.append(row)
    return rows


def vision_table(env: str = "aggro") -> dict:
    cfg = env_cfg(env)
    ctx = context(cfg)
    t = ctx.t
    snap = initialized(cfg, "siege")
    lane = snap.lane
    out = {"gank": [], "pressure": []}
    for beh_name, b in (("Hold", B_HOLD), ("Trade", B_TRADE)):
        for wave_name, w in (("Last Hit", W_HOLD), ("Shove", W_SHOVE)):
            p = champion_gank_probability(lane._replace(va=0), 0, w, b, cfg, cfg.tower)
            pv = champion_gank_probability(lane._replace(va=6), 0, w, b, cfg, cfg.tower)
            pe = gank_damage_p(0, False, cfg.tower, cfg)
            units = cfg.gank_damage_trials * pe
            out["gank"].append({"behaviour": beh_name, "wave": wave_name, "p": p, "p_vision": pv,
                                "units_if_ganked": units, "units_per_turn": p * units, "units_per_turn_vision": pv * units})
    for level in (4, 7, 10, 14, 18):
        for inv in (0, 16, 50):
            items = int(inv * cfg.gold_scale)
            per_turn = ctx.pressure(level, items)
            th = cfg.plate_thresholds
            out["pressure"].append({"level": level, "invested": inv, "centi_units_per_turn": per_turn,
                                    "turns_first_plate": th[0] / per_turn, "turns_all_plates": th[-1] / per_turn})
    out["plate_gold"] = cfg.plate_gold
    out["vision_turns"] = 6
    out["vision_reduction"] = cfg.vision_reduction
    return out


def all_tables() -> Dict[str, dict]:
    return {e: {"recall": recall_table(e), "trade": trade_table(e), "collection": collection_table(e),
                "vision": vision_table(e)} for e in ENV_ORDER}


__all__ = ["recall_table", "trade_table", "collection_table", "vision_table", "all_tables"]
