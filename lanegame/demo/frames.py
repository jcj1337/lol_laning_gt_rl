"""Frames of the M6 demo: what the lane view, the status panel and the turn history show (the frame contract,
docs/m6/m6a_technical_report.md §4).

Two shapes, both JSON-ready and both computed by Python only (the browser computes no game rule):

    position(p)   the state after p turns (p = 0 is the initial position): the lane (packets, crashed masses, the
                  contact), each champion's status and each tower's damage and plates. Status values are the viewer's
                  `summary` of that state, so they are the numbers the Streamlit debugger shows.
    turn(t)       the turn that moves position t - 1 to position t: both codes and their names, the authoritative tick
                  trace (packets, crashed masses, the contact and each champion's cell after every tick, the wave HP
                  each champion removed in that tick and the minion fire it took), and the turn's outcomes in the
                  resolver's order (champions_m4 steps 9 to 14), each marked major or minor.

Per-tick facts are only those the engine records per tick (`info["lane_trace"]`, `info["champion_path"]`). Trades,
tower shots on champions, ganks, deaths, wards, Recalls, plates, level-ups and returns exist per turn, so they appear
only as outcomes, never at a tick.

What a frame never contains: the chance uniforms (`info["chance"]`, `gank_draws`), any policy probability or logit, a
policy's sampling seed, and an action that is not yet resolved. Beliefs are added by the session, never here.

Coordinates are global (seat A's own coordinates; B's are mirrored): A's tower at -2, the lane 0, B's tower +2, the
bases at -4 and +4. A champion at the Vision node W has cell null and `at_ward` true. HP is in centi-HP (the engine's
unit; a level-1 champion has 600), gold in resources (`gold_scale` ledger units).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..champions_m4 import context
from ..config import GameConfig
from ..diagnostics import lane_exit as LX
from ..positional import WARD
from ..progression import plates_broken
from ..viewer import session as VS
from .actions import code_parts

FRAME_SCHEMA = "lanegame.demo_frames"
FRAME_VERSION = 1
SEATS = ("A", "B")


class _OneState:
    """The attributes of a viewer Recording that `viewer.session.summary` reads, for a single state."""

    def __init__(self, cfg: GameConfig, lane, m4, k: int):
        self.cfg, self.lanes, self.champs, self.k0 = cfg, [lane], [m4], int(k)
        self._earned = tuple(LX.earned_units(m4, s, cfg) / cfg.gold_scale for s in (0, 1))

    def earned(self, i: int) -> Tuple[float, float]:
        return self._earned


def _r(x: float, n: int = 4) -> float:
    return round(float(x), n)


def champion_status(cfg: GameConfig, lane, m4, k: int, s: int) -> dict:
    d = VS.summary(_OneState(cfg, lane, m4, k), 0, s)
    me = m4.side(s)
    t = context(cfg).t
    L = d["level"]
    status = LX.seat_status(m4, s)
    return {"level": L, "hp": int(d["hp"]), "hp_max": int(d["hp_max"]), "xp": int(d["xp"]),
            "xp_level": int(t.xp_cum[L]), "xp_next": int(t.xp_cum[L + 1]) if L < t.level_cap else None,
            "gold_earned": _r(d["gold_earned"]), "gold_unspent": _r(d["gold_spendable"]), "invested": _r(d["invested"]),
            "attack_power": _r(d["attack_power"], 3), "status": status, "returns_in": int(me.absent),
            "cell": d["pos_global"] if status == "present" else None, "at_ward": bool(d["at_ward"]),
            "vision": int(d["vision"]), "capacity": d.get("capacity"),
            "wave_budgets": {n: int(v) for n, v in (d.get("wave_budgets") or {}).items()}}


def lane_view(cfg: GameConfig, lane) -> dict:
    objs = VS.lane_objects(lane, cfg)
    return {"packets": {n: [[int(x), int(h), bool(f)] for x, h, f in objs[n]["packets"]] for n in SEATS},
            "siege": {n: [int(objs[n]["siege"][0]), int(objs[n]["siege"][1])] for n in SEATS},
            "contact": VS.contact_of(objs)}


def position(cfg: GameConfig, lane, m4, k: int, p: int) -> dict:
    """The state after p turns (module docstring)."""
    t = context(cfg).t
    champs = {n: champion_status(cfg, lane, m4, k, s) for s, n in enumerate(SEATS)}
    towers = {}
    for s, n in enumerate(SEATS):            # seat n's tower carries the damage the other seat dealt
        dmg = int(m4.side(1 - s).tower_dmg)
        towers[n] = {"damage": dmg, "damage_max": int(cfg.plate_thresholds[-1]),
                     "plates_lost": int(plates_broken(dmg, t)), "plates_total": len(cfg.plate_thresholds)}
    lv = lane_view(cfg, lane)
    return {"p": int(p), "k": int(k), "lane": lv, "champions": champs, "towers": towers,
            "lead_A": _r(LX.lead(m4, cfg)), "wave_status": VS.wave_status(lv)}


def _cell(pos: int, s: int) -> Optional[int]:
    if pos == WARD:
        return None
    return int(pos) if s == 0 else -int(pos)


def ticks(cfg: GameConfig, info: dict) -> List[dict]:
    """The authoritative tick trace of a turn in global coordinates (module docstring)."""
    out = []
    path = info.get("champion_path")
    for j, tk in enumerate(info.get("lane_trace") or []):
        objs = VS.trace_objects(tk, cfg)
        cells, ward = {}, {}
        for s, n in enumerate(SEATS):
            if info["away"][s]:
                cells[n], ward[n] = None, False
            else:
                pos = path[s][j] if path is not None else info["champ_end"][s]
                cells[n], ward[n] = _cell(pos, s), pos == WARD
        fire = {}
        for s, n in enumerate(SEATS):
            cause, units = (tk.get("minion_fire") or [("", 0), ("", 0)])[s]
            fire[n] = {"cause": cause, "units": int(units)}
        out.append({"tick": j + 1,
                    "packets": {n: [[int(x), int(h), bool(f)] for x, h, f in objs[n]["packets"]] for n in SEATS},
                    "siege": {n: [int(objs[n]["siege"][0]), int(objs[n]["siege"][1])] for n in SEATS},
                    "contact": VS.contact_of(objs), "cells": cells, "at_ward": ward,
                    "wave_removed": {n: int(tk["champion"][s]) for s, n in enumerate(SEATS)},
                    "crashed": {n: int(tk["crashed"][s]) for s, n in enumerate(SEATS)},
                    "minion_fire": fire})
    return out


def _ev(kind: str, seat: Optional[str], major: bool, text: str, **data) -> dict:
    d = {"kind": kind, "seat": seat, "major": major, "text": text}
    if data:
        d["data"] = data
    return d


def outcomes(cfg: GameConfig, info: dict) -> List[dict]:
    """The turn's outcomes in the resolver's order. `text` names the seats {A} and {B} (format placeholders the
    client fills with its display names). Gold in resources, HP in centi-HP."""
    sc = cfg.gold_scale
    ev: List[dict] = []
    seats = list(enumerate(SEATS))
    # the lane (per-tick facts, summarized for the turn)
    for s, n in seats:
        o = SEATS[1 - s]
        if info["crash_hp"][s]:
            ev.append(_ev("crash", n, True, f"{{{n}}}'s wave crashed into {{{o}}}'s tower ({info['crash_hp'][s]} HP)",
                          hp=int(info["crash_hp"][s])))
    for s, n in seats:
        if info["events"]["died"][s]:
            ev.append(_ev("wave_lost", n, False, f"{{{n}}} lost {info['events']['died'][s]} minion packet(s) "
                          f"({info['events']['killed_hp'][s]} HP)", packets=int(info["events"]["died"][s]),
                          hp=int(info["events"]["killed_hp"][s])))
        if info["events"]["merged"][s]:
            ev.append(_ev("merge", n, False, f"{{{n}}}'s new wave joined its front"))
    # step 2b and 9: Recall interruptions, trades, tower shots, ganks, minion fire
    for s, n in seats:
        o = SEATS[1 - s]
        if info["recall_interrupted"][s]:
            ev.append(_ev("recall_interrupted", n, True, f"{{{n}}}'s Recall was interrupted"))
        if info["trade_attempted"][s]:
            if not info["trade_connected"][s]:
                ev.append(_ev("trade", n, False, f"{{{n}}} tried to trade but could not reach {{{o}}}",
                              result="no_contact"))
            elif info["trade_damage"][s] > 0:
                ev.append(_ev("trade", n, True, f"{{{n}}} hit {{{o}}} in a trade for {info['trade_damage'][s]} HP",
                              result="hit", damage=int(info["trade_damage"][s]), hit_p=_r(info["hit_p"][s], 3)))
            else:
                ev.append(_ev("trade", n, True, f"{{{n}}} reached {{{o}}} but the trade missed", result="miss",
                              hit_p=_r(info["hit_p"][s], 3)))
    for s, n in seats:
        o = SEATS[1 - s]
        if info["tower_damage"][s]:
            ev.append(_ev("tower_shot", n, True, f"{{{o}}}'s tower shot {{{n}}} for {info['tower_damage'][s]} HP",
                          damage=int(info["tower_damage"][s])))
        if info["ganked"][s]:
            ev.append(_ev("gank", n, True, f"{{{n}}} was ganked for {info['gank_damage'][s]} HP",
                          damage=int(info["gank_damage"][s]), p=_r(info["gank_p"][s], 3)))
        if info.get("minion_hits") and info["minion_hits"][s]:
            ev.append(_ev("minion_fire", n, False, f"{{{o}}}'s minions hit {{{n}}} for {info['minion_damage'][s]} HP",
                          damage=int(info["minion_damage"][s]), units=int(info["minion_hits"][s])))
    # step 10: deaths and kills
    for s, n in seats:
        o = SEATS[1 - s]
        if info["died"][s]:
            ev.append(_ev("death", n, True, f"{{{n}}} died; {{{o}}} gained {info['bounty_units'][1 - s] / sc:.2f} gold "
                          "for the kill", kill_gold=_r(info["bounty_units"][1 - s] / sc)))
    # step 11: tower damage and plates
    for s, n in seats:
        o = SEATS[1 - s]
        if info["tower_damage_dealt"][s]:
            ev.append(_ev("tower_damage", n, False,
                          f"{{{n}}} damaged {{{o}}}'s tower (+{info['tower_damage_dealt'][s]})",
                          damage=int(info["tower_damage_dealt"][s])))
        if info["plates_broken"][s]:
            ev.append(_ev("plate", n, True, f"{{{n}}} broke {info['plates_broken'][s]} plate(s) of {{{o}}}'s tower: "
                          f"+{info['plate_units'][s] / sc:.2f} gold", plates=int(info["plates_broken"][s]),
                          gold=_r(info["plate_units"][s] / sc)))
    # step 12: levels
    for s, n in seats:
        if info["level_up"][s]:
            ev.append(_ev("level_up", n, False, f"{{{n}}} reached level {info['levels'][s]}",
                          level=int(info["levels"][s])))
    # Wards (placed at the end of the turn) and step 14: Recalls, returns and investment
    for s, n in seats:
        if info.get("ward") and info["ward"][s]:
            if info["ward_placed"][s]:
                ev.append(_ev("ward", n, True, f"{{{n}}} placed a ward in the bush (vision for {cfg.vision_turns} turns)"))
            else:
                ev.append(_ev("ward_interrupted", n, True, f"{{{n}}}'s ward was interrupted"))
        if info["recalled"][s]:
            ev.append(_ev("recall", n, True, f"{{{n}}} recalled to base"))
        if info["returned"][s]:
            ev.append(_ev("return", n, True, f"{{{n}}} is back in lane next turn"))
        if info["invested"][s]:
            ev.append(_ev("invest", n, False, f"{{{n}}} invested {info['invested'][s] / sc:.2f} gold at base",
                          gold=_r(info["invested"][s] / sc)))
    return ev


def turn(cfg: GameConfig, info: dict, pre: dict, post: dict) -> dict:
    """Turn t = pre["p"] + 1 (module docstring); `pre` and `post` are its two `position` views."""
    a, b = (int(x) for x in info["actions"])
    sc = cfg.gold_scale
    delta = {}
    for s, n in enumerate(SEATS):
        p0, p1 = pre["champions"][n], post["champions"][n]
        delta[n] = {"gold": _r(info["earned_units"][s] / sc), "farm": _r(info["farm_units"][s] / sc),
                    "xp": int(info["xp_gain"][s]), "hp": p1["hp"] - p0["hp"],
                    "hp_taken": int(info["damage_taken"][s])}
    return {"t": pre["p"] + 1, "k": int(info["k"]), "codes": [a, b],
            "names": [VS.action_name(cfg, a), VS.action_name(cfg, b)], "parts": [code_parts(a), code_parts(b)],
            "modes": list(info["modes"]), "ticks": ticks(cfg, info), "outcomes": outcomes(cfg, info),
            "delta": delta, "reward_A": _r(info["reward"], 7)}


__all__ = ["FRAME_SCHEMA", "FRAME_VERSION", "position", "turn", "ticks", "outcomes", "champion_status", "lane_view"]
