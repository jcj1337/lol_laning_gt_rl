"""Frames of the M6 demo: what the lane view, the status panel and the turn history show (the frame contract,
docs/m6/m6a_technical_report.md §4).

Two shapes, both JSON-ready and both computed by Python only (the browser computes no game rule):

    position(p)   the state after p turns (p = 0 is the initial position): the lane (packets, crashed masses, the
                  contact), each champion's status and each tower's damage and plates. Status values are the viewer's
                  `summary` of that state, so they are the numbers the Streamlit debugger shows.
    turn(t)       the turn that moves position t - 1 to position t: both codes and their names, the authoritative tick
                  trace (packets, crashed masses, the contact and each champion's cell after every tick, the wave HP
                  each champion removed in that tick, the minion packets each seat lost in it and the minion fire each
                  champion took), and the turn's outcomes in the
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
from ..mechanics import _fronts_in_contact, _merge
from ..positional import WARD
from ..progression import plates_broken
from ..state import State
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


def packets_lost(cfg: GameConfig, info: dict) -> List[Tuple[int, int]]:
    """Per tick of the turn, how many of each seat's minion packets died in it (crashes excluded). The trace records
    the packets after each tick but not the deaths, and a count also falls when a rear packet merges into the front.
    So each tick's movement and merging (mechanics._tick_natural steps 1 and 2) are replayed on the packets before it;
    after them a packet leaves the lane only from the front, by dying or crashing (at most one crash per tick)."""
    lane = State.from_list(info["state_before"])
    P = []
    for s in (0, 1):
        fx, fh, rears = lane.side(s)[:3]
        P.append(([[fx, fh]] if fh > 0 else []) + [[x, cfg.wave_hp] for x in rears])
    siege = [lane.asg, lane.bsg]
    out = []
    for tk in info.get("lane_trace") or []:
        contact = _fronts_in_contact(P)
        lost = []
        for s, n in enumerate(SEATS):
            pk = [list(p) for p in P[s]]
            if pk and not contact and not (siege[1 - s] > 0 and pk[0][0] >= -cfg.tower - 1):
                pk[0][0] += 1
            for p in pk[1:]:
                if p[0] + 1 < pk[0][0]:
                    p[0] += 1
            lost.append(len(_merge(pk, cfg.hp_cap, block=True)) - len(tk[n]) - (1 if tk["crashed"][s] else 0))
        out.append((lost[0], lost[1]))
        P, siege = [tk["A"], tk["B"]], tk["siege"]
    return out


def ticks(cfg: GameConfig, info: dict) -> List[dict]:
    """The authoritative tick trace of a turn in global coordinates (module docstring)."""
    out = []
    path = info.get("champion_path")
    lost = packets_lost(cfg, info)
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
                    "packets_lost": {n: int(lost[j][s]) for s, n in enumerate(SEATS)},
                    "minion_fire": fire})
    return out


def _ev(kind: str, seat: Optional[str], major: bool, text: str, **data) -> dict:
    d = {"kind": kind, "seat": seat, "major": major, "text": text}
    if data:
        d["data"] = data
    return d


def outcomes(cfg: GameConfig, info: dict) -> List[dict]:
    """The turn's outcomes in the resolver's order. `text` names the seats {A} and {B} (format placeholders the
    client fills with its display names). Gold in resources, damage and HP in centi-HP. A trade, tower shot, gank or
    minion fire quotes its damage before durability; `hp_lost` gives the HP a seat lost when the two differ."""
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
                ev.append(_ev("trade", n, True, f"{{{n}}} hit {{{o}}} in a trade for {info['trade_damage'][s]} damage",
                              result="hit", damage=int(info["trade_damage"][s]), hit_p=_r(info["hit_p"][s], 3)))
            else:
                ev.append(_ev("trade", n, True, f"{{{n}}} reached {{{o}}} but the trade missed", result="miss",
                              hit_p=_r(info["hit_p"][s], 3)))
    for s, n in seats:
        o = SEATS[1 - s]
        if info["tower_damage"][s]:
            ev.append(_ev("tower_shot", n, True, f"{{{o}}}'s tower shot {{{n}}} for {info['tower_damage'][s]} damage",
                          damage=int(info["tower_damage"][s])))
        if info["ganked"][s]:
            ev.append(_ev("gank", n, True, f"{{{n}}} was ganked for {info['gank_damage'][s]} damage",
                          damage=int(info["gank_damage"][s]), p=_r(info["gank_p"][s], 3)))
        if info.get("minion_hits") and info["minion_hits"][s]:
            ev.append(_ev("minion_fire", n, False, f"{{{o}}}'s minions hit {{{n}}} for {info['minion_damage'][s]} damage",
                          damage=int(info["minion_damage"][s]), units=int(info["minion_hits"][s])))
    # step 9b: the resolver divides each seat's summed damage by its durability M(S) (invest_durability)
    for s, n in seats:
        damage = int(info["trade_damage"][1 - s] + info["tower_damage"][s] + info["gank_damage"][s]
                     + (info["minion_damage"][s] if "minion_damage" in info else 0))
        hp = int(info["damage_taken"][s])
        if damage != hp:
            ev.append(_ev("hp_lost", n, False, f"{{{n}}} lost {hp} HP from {damage} damage; invested gold reduces "
                          "damage taken", damage=damage, hp=hp))
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


def _place(c: dict):
    """Where a champion stands in a position view: its global cell, "W" at the ward spot, or "base" while away."""
    if c["status"] != "present":
        return "base"
    return "W" if c["at_ward"] else c["cell"]


def _front(lane: dict, n: str) -> Optional[int]:
    return next((x for x, _, front in lane["packets"][n] if front), None)


def moves(cfg: GameConfig, info: dict, pre: dict, post: dict) -> Dict[str, dict]:
    """Per seat, how its champion and its leading wave moved over the turn, for the lane's arrows. Places are global
    cells, "W" or "base". A champion walks from its start place to its end place, or recalls, returns or dies (to
    and from its base). A wave move follows the leading packet the turn started with: packets die or crash only as
    the leading packet (mechanics: `P[s].pop(0)`), so it is the same packet at the end when the seat lost none and
    crashed none (the viewer's front-arrow rule, `viewer.session.frame_view`); it walks to the end front's cell, or
    crashes into the enemy tower when the turn's crash was its only loss. Otherwise there is no wave move."""
    out = {}
    for s, n in enumerate(SEATS):
        c0, c1 = pre["champions"][n], post["champions"][n]
        p0, p1 = _place(c0), _place(c1)
        if info["died"][s]:
            champ = {"kind": "death", "from": p0, "to": "base"}
        elif info["recalled"][s]:
            champ = {"kind": "recall", "from": p0, "to": "base"}
        elif info["returned"][s]:
            champ = {"kind": "return", "from": "base", "to": p1}
        elif p0 != p1 and "base" not in (p0, p1):
            champ = {"kind": "walk", "from": p0, "to": p1}
        else:
            champ = None
        f0, f1 = _front(pre["lane"], n), _front(post["lane"], n)
        wave = None
        if f0 is not None and not info["events"]["died"][s]:
            if info["crash_hp"][s]:
                wave = {"kind": "crash", "from": f0, "to": cfg.tower if s == 0 else -cfg.tower}
            elif f1 is not None and f1 != f0:
                wave = {"kind": "walk", "from": f0, "to": f1}
        out[n] = {"champion": champ, "wave": wave}
    return out


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
            "delta": delta, "reward_A": _r(info["reward"], 7), "moves": moves(cfg, info, pre, post)}


__all__ = ["FRAME_SCHEMA", "FRAME_VERSION", "position", "turn", "ticks", "packets_lost", "outcomes", "moves",
           "champion_status", "lane_view"]
