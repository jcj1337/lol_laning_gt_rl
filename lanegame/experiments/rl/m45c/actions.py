"""The 61-code action space of `m45_calibrated()` as M4.5c reports it.

The encoding is the four-action one (lanegame.config; m44.actions describes it): a = wave + 3 behaviour + 15 objective
for the Last Hit / Shove / Thin slots, 46 + behaviour + 5 objective for Slow Push, 45 = ABSENT, 12 = Recall. M4.5a
replaced the siege Vision objective by the off-lane Ward action, code 30 = champ_act(Last Hit, Hold, Vision)
(`config.CHAMP_WARD`). So, on this preset, `m44.actions.decode(30)` reads "Last Hit + Hold + Vision", which is NOT what
code 30 does here: every M4.5c table therefore classifies a code by `category()`, with Ward its own category, and never
counts a Ward turn as a Last Hit turn.

Ever-legal codes (champions.champion_legal_actions with vision_rule "ward"): 4 waves x {Hold, Yield, Trade, Safe} (16),
4 waves x {Hold, Trade} x Pressure (8, only with an own crash under the enemy tower that the champion can reach),
Ward (1, vision down and the champion able to walk to W), Recall (1, at or behind the recall line), ABSENT (1, the only
code while away): 27 of 61. The other 34 (Vision with any other wave or behaviour, Yield / Safe with an objective,
Recall with another wave or an objective) are masked in every state.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List

import numpy as np

from ....config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, CHAMP_ABSENT, CHAMP_BEHAVIORS, CHAMP_WARD, NONE,
                        PRESSURE, GameConfig, champ_act, champ_behavior, champ_objective, champ_wave)
from ..m44.actions import RECALL, WAVE_INDEX, WAVE_KEYS, WAVE_NAMES, WAVE_ORDER

N_CODES = 61
WARD = CHAMP_WARD                                   # 30
CATEGORIES = ("LH", "TN", "SL", "SH", "Ward", "Recall", "Absent")
BEHAVIORS = ("Hold", "Yield", "Trade", "Safe")


def _ever_legal() -> FrozenSet[int]:
    s = {champ_act(w, b) for w in WAVE_ORDER for b in CHAMP_BEHAVIORS}
    s |= {champ_act(w, b, PRESSURE) for w in WAVE_ORDER for b in (B_HOLD, B_TRADE)}
    s |= {WARD, RECALL, CHAMP_ABSENT}
    return frozenset(s)


EVER_LEGAL: FrozenSet[int] = _ever_legal()
NEVER_LEGAL: FrozenSet[int] = frozenset(range(N_CODES)) - EVER_LEGAL


def category(a: int) -> str:
    """LH / TN / SL / SH (a lane code: wave action + behaviour [+ Pressure]), Ward, Recall or Absent."""
    a = int(a)
    if a == CHAMP_ABSENT:
        return "Absent"
    if a == RECALL or champ_behavior(a) == B_RECALL:          # (Recall with another wave / objective: never legal)
        return "Recall"
    if a == WARD:
        return "Ward"
    return WAVE_KEYS[WAVE_INDEX[champ_wave(a)]]


def behavior(a: int) -> str:
    """Hold / Yield / Trade / Safe for a lane code; '' for Ward, Recall and Absent."""
    a = int(a)
    if a in (CHAMP_ABSENT, RECALL, WARD) or champ_behavior(a) == B_RECALL:
        return ""
    return BEHAVIORS[champ_behavior(a)]


def pressure(a: int) -> bool:
    a = int(a)
    return behavior(a) != "" and champ_objective(a) == PRESSURE


CAT_OF = np.array([CATEGORIES.index(category(a)) for a in range(N_CODES)])
BEH_OF = np.array([BEHAVIORS.index(behavior(a)) if behavior(a) else -1 for a in range(N_CODES)])
PRESSURE_OF = np.array([pressure(a) for a in range(N_CODES)])


def check_space(cfg: GameConfig) -> None:
    """Raise unless cfg has the M4.5 Ward action space this module describes."""
    if not (cfg.is_m4 and cfg.natural_waves and cfg.wave_slowpush and cfg.ward_node):
        raise ValueError("not an M4.5 Ward configuration (needs ruleset m4, natural waves, wave_slowpush, vision_rule "
                         "'ward')")
    if tuple(cfg.action_codes()) != tuple(range(N_CODES)):
        raise ValueError("unexpected action space size")


def action_table(cfg: GameConfig) -> List[dict]:
    return [{"code": a, "label": cfg.label(a), "category": category(a), "behavior": behavior(a),
             "pressure": bool(pressure(a)), "ever_legal": a in EVER_LEGAL} for a in cfg.action_codes()]


def histogram(actions: np.ndarray) -> Dict[str, float]:
    """Shares of the categories, the behaviours (of lane codes) and Pressure among `actions` (codes)."""
    a = np.asarray(actions, dtype=int).ravel()
    n = len(a)
    if n == 0:
        return {}
    out = {f"cat_{c}": float((CAT_OF[a] == i).sum()) / n for i, c in enumerate(CATEGORIES)}
    lane = BEH_OF[a] >= 0
    nl = int(lane.sum())
    for i, b in enumerate(BEHAVIORS):
        out[f"beh_{b}"] = float((BEH_OF[a][lane] == i).sum()) / nl if nl else float("nan")
    out["pressure_of_lane"] = float(PRESSURE_OF[a][lane].mean()) if nl else float("nan")
    return out


__all__ = ["N_CODES", "WARD", "RECALL", "CATEGORIES", "BEHAVIORS", "EVER_LEGAL", "NEVER_LEGAL", "category", "behavior",
           "pressure", "CAT_OF", "BEH_OF", "PRESSURE_OF", "check_space", "action_table", "histogram", "WAVE_NAMES"]
