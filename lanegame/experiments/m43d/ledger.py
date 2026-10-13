"""Per-turn farm-loss decomposition and risk ledger of the M4.3d study.

Farm. Every enemy minion HP that dies on side s's kill ledger at own cell x is worth farm_per_hp * lane(x) at the
lane rate lane(x) = collect_contest (x < T) or collect_contest_enemy_tower (x >= T); both are 0.9 in every preset of
the study, so "ideal" is 0.1125 resources per HP. What s actually collects (champions_m4.collection_units) is split,
in this order, into exclusive parts that sum to the ideal exactly (up to float rounding):

    absence    s is away (dead, recalled) or channelling an interrupted Recall: nothing collected
    position   s is present but the kill is beyond its reach (farm_reach 1; retreat_farm_reach 2 while retreating)
    tower      the UNCONDITIONAL own-tower cap: lane(x) - min(lane(x), collect_contest_own_tower) for a kill in the own
               tower cell (x = -T) within reach, whatever s or its opponent does
    posture    the defensive posture (Safe, or a Yield that retreated from a trader) collects at the Safe base
               exact_safe_base(x) instead of the lane rate (after the tower cap): min(lane, cap) - min(safe, cap)
    trade      a connected trader keeps trade_farm_factor of what remains
    collected  the rest (equals info["farm"][s]; checked)

so "tower" is the only part that does not depend on a decision or an interaction; position, posture and trade are
the consequences of where s chose to stand and how it played, absence is the price of HP lost earlier. The parts are
also split by where the minions died (the own tower cell, or elsewhere).

Risk. Per side and turn: the global cell where the champion ended (present turns), its exposure, the gank occurrence
probability, the EXPECTED gank damage gank_p * trials * p_e * u(L) / H(L) in bars (before durability mitigation:
the mitigation divides a turn's summed damage once), the expected tower-shot damage tower_p * u(L) / H(L), the
realized gank / tower / trade damage in bars (before mitigation), and whether a tower shot was possible at all
(a connected trade in or next to the enemy tower).
"""

from __future__ import annotations

from typing import Dict, List

from ...champions import AWAY, CHANNEL, FARM, RETREAT, TRADING
from ...config import GameConfig, exact_safe_base

A, B = 0, 1
FARM_PARTS = ("absence", "position", "tower", "posture", "trade", "collected")
CELLS = (-2, -1, 0, 1, 2)                 # champion cells in the side's own coordinates


def farm_breakdown(info: dict, cfg: GameConfig, s: int) -> Dict[str, float]:
    """The exclusive farm parts of side s this turn (module docstring), in resources, plus the tower-cell split."""
    T = cfg.tower
    fph = cfg.farm_per_hp
    mode = info["modes"][s]
    pos = info["champ_end"][s]
    ot = cfg.collect_contest_own_tower
    tf = cfg.trade_farm_factor
    out = {k: 0.0 for k in FARM_PARTS}
    out.update(ideal=0.0, ideal_tower=0.0, collected_tower=0.0, tower_hp=0.0, lane_hp=0.0)
    reach = cfg.retreat_farm_reach if mode == RETREAT else cfg.farm_reach
    for x, hp in info["kills"][s].items():
        x = int(x)
        lane = cfg.collect_contest if x < T else cfg.collect_contest_enemy_tower
        ideal = hp * fph * lane
        out["ideal"] += ideal
        at_tower = x == -T
        if at_tower:
            out["ideal_tower"] += ideal
            out["tower_hp"] += hp
        else:
            out["lane_hp"] += hp
        if mode in (AWAY, CHANNEL):
            out["absence"] += ideal
            continue
        if abs(x - pos) > reach:
            out["position"] += ideal
            continue
        cap = ot if at_tower else 1.0
        r1 = min(lane, cap)
        base = lane if mode in (FARM, TRADING) else float(exact_safe_base(x, cfg))
        r2 = min(base, cap)
        r3 = r2 * tf if mode == TRADING else r2
        out["tower"] += hp * fph * (lane - r1)
        out["posture"] += hp * fph * (r1 - r2)
        out["trade"] += hp * fph * (r2 - r3)
        out["collected"] += hp * fph * r3
        if at_tower:
            out["collected_tower"] += hp * fph * r3
    return out


def risk_row(info: dict, cfg: GameConfig, t, s: int) -> Dict[str, float]:
    """The risk ledger of side s this turn (module docstring)."""
    lv = info["levels_before"][s]
    H, u = t.hp_max[lv], t.hp_unit[lv]
    away = bool(info["away"][s])
    row = {"present": 0.0 if away else 1.0}
    for c in CELLS:
        row[f"at{c:+d}"] = 0.0
    if not away:
        row[f"at{int(info['champ_end'][s]):+d}"] = 1.0
    gp, pe = info["gank_p"][s], info["gank_damage_p"][s]
    row["gank_p"] = gp
    row["exp_gank"] = gp * cfg.gank_damage_trials * pe * u / H
    row["gank_dmg"] = info["gank_damage"][s] / H
    row["exp_tower"] = info["tower_p"][s] * cfg.tower_shot_damage * u / H
    row["tower_dmg"] = info["tower_damage"][s] / H
    row["trade_dmg"] = info["trade_damage"][1 - s] / H
    row["tower_risk_turn"] = 1.0 if info["tower_p"][s] > 0 else 0.0
    row["exposure"] = float(info["exposure"][s]) if not away else float("nan")
    row["at_enemy_tower_exp"] = gp if (not away and info["champ_end"][s] == cfg.tower) else 0.0
    return row


def turn_extra(info: dict, cfg: GameConfig, t) -> List[Dict[str, float]]:
    """Per side: the farm parts and the risk ledger of one turn."""
    out = []
    for s in (A, B):
        d = farm_breakdown(info, cfg, s)
        got = info["farm"][s]
        if abs(d["collected"] - got) > 1e-9:
            raise AssertionError(f"farm decomposition {d['collected']} != collected {got} (side {s})")
        d.update(risk_row(info, cfg, t, s))
        d["xp"] = float(info["xp_gain"][s])
        d["xp_avail"] = float(cfg.xp_per_hp * sum(info["kills"][s].values()))
        d["xp_missed"] = d["xp_avail"] - d["xp"] - float(info["xp_discarded"][s])
        out.append(d)
    return out


def window_sums(extra: List[List[Dict[str, float]]], n=None) -> List[Dict[str, float]]:
    """Per side: the sums of every numeric field of the first n turns (exposure: the mean over present turns)."""
    rows = extra[:n] if n else extra
    out = []
    for s in (A, B):
        keys = rows[0][s].keys()
        d = {}
        for k in keys:
            if k == "exposure":
                v = [r[s][k] for r in rows if r[s]["present"] > 0]
                d[k] = float(sum(v) / len(v)) if v else float("nan")
            else:
                d[k] = float(sum(r[s][k] for r in rows))
        out.append(d)
    return out


__all__ = ["farm_breakdown", "risk_row", "turn_extra", "window_sums", "FARM_PARTS", "CELLS"]
