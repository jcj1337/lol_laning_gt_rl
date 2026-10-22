"""The action menu of one seat at one state, with the reason for every unavailable choice (M6a).

The 61 codes of `m45_calibrated()` are presented as a few groups, never as a flat list:

    wave       Last Hit, Thin, Slow Push, Shove            what the champion does to enemy minions (always legal)
    behaviour  Hold, Yield, Trade, Safe                    where it stands and whether it fights (always legal)
    objective  none, Pressure                              Pressure only with Hold or Trade, and only when
                                                           `objective_reachable`
    single     Recall (12), Ward (30)                      `recall_legal`, `ward_legal`
    forced     Absent (45)                                 the only legal code while the champion is away

A choice is legal exactly when `champions.champion_legal_actions` lists its code: the menu evaluates the same
predicates the engine uses (`objective_reachable` and its two parts `siege_eligible` and the movement reach,
`ward_legal`, `recall_legal`, the absence counter), and `menu()` checks that the codes it marks legal equal the
engine's set, so a reason can never disagree with the backend's validation. Vision codes other than Ward are never
legal under `vision_rule = "ward"` and are not offered.

Cells in the reasons are the seat's own coordinates in words: its tower is cell -2, the lane (midpoint) 0, the enemy
tower +2, and W the Vision node beside the lane.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..champions import champion_legal_actions, objective_reachable, recall_legal, ward_legal
from ..config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, BEHAVIOR_NAMES, CHAMP_ABSENT, CHAMP_WARD, NONE,
                      PRESSURE, SP, VISION, W_SHOVE, W_SLOWPUSH, W_THIN, W_HOLD, GameConfig, champ_act, champ_behavior,
                      champ_objective, champ_wave, wave_display_names)
from ..mechanics import siege_eligible
from ..positional import WARD, dist as graph_dist

RECALL_CODE = champ_act(SP, B_RECALL)                    # 12
WAVE_ORDER = (W_HOLD, W_THIN, W_SLOWPUSH, W_SHOVE)       # least to most wave damage
BEHAVIOUR_ORDER = (B_HOLD, B_YIELD, B_TRADE, B_SAFE)
PRESSURE_BEHAVIOURS = (B_HOLD, B_TRADE)

WAVE_HELP = {W_HOLD: "Take only the last hit: 1 HP of enemy minions.",
             W_THIN: "Remove the enemy minion HP that evens the two waves' strength.",
             W_SLOWPUSH: "Remove one HP more than Thin, so your wave slowly builds and pushes.",
             W_SHOVE: "Spend your full wave damage on enemy minions; your wave pushes and crashes sooner."}
BEHAVIOUR_HELP = {B_HOLD: "Stand at your farm spot and farm; never step away.",
                  B_YIELD: "Farm from a stand-off cell and step away from an incoming Trade.",
                  B_TRADE: "Chase the enemy champion and deal damage if you reach it.",
                  B_SAFE: "Farm without going past your side of the lane; step away from an incoming Trade."}
PRESSURE_HELP = "Walk into the enemy tower cell and damage the tower; each plate pays 1 gold."
RECALL_HELP = "Channel this turn, sit out the next, then return with full HP and your gold invested."
WARD_HELP = ("Walk to the Vision node W and place a ward: ganks are a quarter as likely for 6 turns. No wave damage "
             "or gold this turn.")


def cell_words(pos: int, cfg: GameConfig) -> str:
    """A seat's own cell in words."""
    if pos == WARD:
        return "the Vision node W"
    if pos <= -cfg.tower:
        return "your tower"
    if pos >= cfg.tower:
        return "the enemy tower"
    return {-1: "your side of the lane", 0: "the lane (midpoint)", 1: "the enemy side of the lane"}.get(
        pos, f"cell {pos:+d}")


def pressure_reason(lane, m4, s: int, cfg: GameConfig) -> Optional[str]:
    """Why Pressure is unavailable (None when `objective_reachable`)."""
    if objective_reachable(lane, m4, s, cfg):
        return None
    if not siege_eligible(lane, s, cfg):
        return "Pressure needs your crashed minions under the enemy tower, and there are none."
    pos = m4.side(s).pos
    d = graph_dist(pos, cfg.tower) if pos == WARD else cfg.tower - pos
    return (f"You cannot reach the enemy tower this turn: it is {d} steps from {cell_words(pos, cfg)} and a champion "
            f"moves {cfg.champion_speed}.")


def recall_reason(m4, s: int, cfg: GameConfig) -> Optional[str]:
    if recall_legal(m4, s, cfg):
        return None
    pos = m4.side(s).pos
    if pos == WARD:
        return "You cannot Recall from the Vision node W."
    return (f"Recall needs you on your own side of the lane (cell {cfg.recall_line:+d} or nearer your tower); you "
            f"stand at {cell_words(pos, cfg)}.")


def ward_reason(m4, s: int, cfg: GameConfig) -> Optional[str]:
    if ward_legal(m4, s, cfg):
        return None
    pos = m4.side(s).pos
    return (f"The Vision node W is {graph_dist(pos, WARD)} steps from {cell_words(pos, cfg)}; a champion moves "
            f"{cfg.champion_speed}.")


def code_parts(code: int) -> dict:
    """The groups of a code: {"kind": "lane" | "recall" | "ward" | "absent", wave, behaviour, objective}."""
    if code == CHAMP_ABSENT:
        return {"kind": "absent"}
    if code == RECALL_CODE:
        return {"kind": "recall"}
    if code == CHAMP_WARD:
        return {"kind": "ward"}
    return {"kind": "lane", "wave": champ_wave(code), "behaviour": champ_behavior(code),
            "objective": champ_objective(code)}


def menu(lane, m4, s: int, cfg: GameConfig) -> dict:
    """The seat's choices at this state (module docstring). Keys:

        legal        the engine's legal codes (sorted)
        forced       45 when Absent is the only legal code, else None
        away         {"status": "recall" | "dead", "returns_in": turns} while away, else None
        waves, behaviours, objectives, singles   the groups, each item with "legal" and, when unavailable, "reason"
        lane_codes   {"<wave>,<behaviour>,<objective>": code} for every combination the groups can form, legal or not
    """
    if cfg.vision_rule != "ward" or not cfg.wave_slowpush:
        raise ValueError("the demo menu is defined for the four-action Ward presets (m45_calibrated)")
    legal = tuple(champion_legal_actions(lane, m4, s, cfg))
    me = m4.side(s)
    names = wave_display_names(cfg)
    out: Dict[str, object] = {"legal": list(legal), "forced": None, "away": None}
    if me.absent > 0:
        out["forced"] = CHAMP_ABSENT
        out["away"] = {"status": "dead" if me.hp <= 0 else "recall", "returns_in": int(me.absent)}
        why = ("Your champion is dead and away from the lane; Absent is the only action until it returns."
               if me.hp <= 0 else "Your champion is recalling to base; Absent is the only action until it returns.")
        out.update(waves=[{"id": w, "name": names[w], "legal": False, "reason": why} for w in WAVE_ORDER],
                   behaviours=[{"id": b, "name": BEHAVIOR_NAMES[b], "legal": False, "reason": why}
                               for b in BEHAVIOUR_ORDER],
                   objectives=[], singles=[], lane_codes={})
        _check(out, legal)
        return out
    p_reason = pressure_reason(lane, m4, s, cfg)
    out["waves"] = [{"id": w, "name": names[w], "legal": True, "help": WAVE_HELP[w]} for w in WAVE_ORDER]
    out["behaviours"] = [{"id": b, "name": BEHAVIOR_NAMES[b], "legal": True, "help": BEHAVIOUR_HELP[b]}
                         for b in BEHAVIOUR_ORDER]
    out["objectives"] = [{"id": NONE, "name": "No objective", "legal": True},
                         {"id": PRESSURE, "name": "Pressure", "legal": p_reason is None, "help": PRESSURE_HELP,
                          "with": [BEHAVIOR_NAMES[b] for b in PRESSURE_BEHAVIOURS],
                          **({"reason": p_reason} if p_reason else {})}]
    r_reason, w_reason = recall_reason(m4, s, cfg), ward_reason(m4, s, cfg)
    out["singles"] = [{"code": RECALL_CODE, "name": "Recall", "legal": r_reason is None, "help": RECALL_HELP,
                       **({"reason": r_reason} if r_reason else {})},
                      {"code": CHAMP_WARD, "name": "Ward", "legal": w_reason is None, "help": WARD_HELP,
                       **({"reason": w_reason} if w_reason else {})}]
    lane_codes = {}
    for w in WAVE_ORDER:
        for b in BEHAVIOUR_ORDER:
            for o in (NONE, PRESSURE):
                lane_codes[f"{w},{b},{o}"] = champ_act(w, b, o)
    out["lane_codes"] = lane_codes
    _check(out, legal)
    return out


def combination_reason(menu_: dict, wave: int, behaviour: int, objective: int) -> Optional[str]:
    """Why a (wave, behaviour, objective) combination is unavailable, None when it is legal."""
    code = menu_["lane_codes"].get(f"{wave},{behaviour},{objective}")
    if code is None:
        return "Not an action of this game."
    if code in menu_["legal"]:
        return None
    if objective == PRESSURE:
        p = next(o for o in menu_["objectives"] if o["id"] == PRESSURE)
        if not p["legal"]:
            return p["reason"]
        return "Pressure combines with Hold or Trade only."
    return "Unavailable."                                   # pragma: no cover (every other combination is legal)


def explain(lane, m4, s: int, cfg: GameConfig, code) -> Optional[str]:
    """Why `code` is not a legal submission for seat s here (None when it is legal): malformed, not an action, or
    the reason of its group."""
    if isinstance(code, bool) or not isinstance(code, int):
        return f"An action is an integer code; got {code!r}."
    m = menu(lane, m4, s, cfg)
    if code in m["legal"]:
        return None
    if m["forced"] is not None:
        return m["waves"][0]["reason"]
    if not 0 <= code < 61:
        return f"{code} is not an action code (0 to 60)."
    parts = code_parts(code)
    if parts["kind"] == "recall":
        return m["singles"][0]["reason"]
    if parts["kind"] == "ward":
        return m["singles"][1]["reason"]
    if parts["kind"] == "absent":
        return "Absent is only for a champion that is away from the lane."
    if parts["objective"] == VISION:
        return "Vision is not an objective in this game; use Ward."
    if parts["behaviour"] not in BEHAVIOUR_ORDER:
        return f"{code} is not an action code of this game."
    return combination_reason(m, parts["wave"], parts["behaviour"], parts["objective"])


def _check(out: dict, legal) -> None:
    """The menu marks legal exactly the engine's codes."""
    marked = set()
    if out["forced"] is not None:
        marked.add(out["forced"])
    else:
        for key, code in out["lane_codes"].items():
            w, b, o = (int(x) for x in key.split(","))
            obj = next(x for x in out["objectives"] if x["id"] == o)
            if obj["legal"] and (o == NONE or b in PRESSURE_BEHAVIOURS):
                marked.add(code)
        marked.update(x["code"] for x in out["singles"] if x["legal"])
    if marked != set(legal):
        raise AssertionError(f"the menu disagrees with the engine's legal set: {sorted(marked ^ set(legal))}")


__all__ = ["menu", "explain", "combination_reason", "code_parts", "cell_words", "pressure_reason", "recall_reason",
           "ward_reason", "RECALL_CODE", "WAVE_ORDER", "BEHAVIOUR_ORDER", "PRESSURE_BEHAVIOURS"]
