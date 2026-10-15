"""M4.5a start states and scripted strategies for demonstrations, smoke checks and the viewer.

Start states (deterministic, hand-built; own coordinates per side, every packet a valid natural-wave packet):

    bigwave       A's 8-HP front at its -1 fights B's 24-HP stack at the lane cell; both champions with their waves,
                  level 2 (Scenario C: trading into a large enemy wave early)
    bigwave_l18   the same lane at level 18 with 128 resources invested each (Scenario C late)
    ahead         A's front behind at its -2, B's 8-HP front at the lane cell; A's champion at the lane cell (in B's
                  wave, ahead of its own), B's at its -1, level 3 (Scenario D; B's front walks past A on tick 0, so
                  under the corrected rules A shares its cell only at the turn's start: kept for revision-1 replays)
    ahead_held    `ahead` with A's front at -1 in contact with B's, which stays at the lane cell (Scenario D since the
                  movement correction)

  the M4.5a movement correction (champion_movement_rule "tick"; its own scenario letters A, C, D), level 2:
    follow        A at the lane cell with its 16-HP front; B's 2-HP front at A's +1 (Hold follows its winning wave)
    chase         A's front at -2 against B's 16-HP front at A's -1, both champions at their towers (a Trader walks
                  into B's wave to reach B)
    walk_back     A at B's tower cell, far ahead of its wave at -2; the waves walk toward each other (A walks back
                  through B's wave and meets its own)

The ward strategies wrap a four-action `Strategy`: Ward (code 30) when Ward is legal, the own vision timer is 0 and
the rule holds, otherwise the inner strategy (its low-HP Recall pre-rule always has priority):

    favorable   the enemy wave is not in the own half (no enemy front, or it stands at the own lane cell or beyond)
    always      whenever the own vision is down
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from ...champions import champion_legal_actions
from ...champions_m4 import M4State, context, initial_m4
from ...config import CHAMP_ABSENT, CHAMP_WARD, GameConfig
from ...state import State
from ..m43c import strategies as st

Start = Tuple[State, M4State, int]


def _champs(cfg: GameConfig, pa: int, pb: int, level: int, invested: float = 0.0) -> M4State:
    t = context(cfg).t
    m = initial_m4(cfg)
    xp, hp = t.xp_cum[level], t.hp_max[level]
    items = int(round(invested * cfg.gold_scale)) if cfg.continuous_items else 0
    return M4State(m.a._replace(pos=pa, xp=xp, hp=hp, items=items), m.b._replace(pos=pb, xp=xp, hp=hp, items=items))


def bigwave(cfg: GameConfig, level: int = 2, invested: float = 0.0) -> Start:
    lane = State(-1, 8, (), 0, 0, 24, (), 0, 0, 0, 0, 0)
    return lane, _champs(cfg, -1, 0, level, invested), 2


def ahead(cfg: GameConfig) -> Start:
    lane = State(-2, 8, (), 0, 0, 8, (), 0, 0, 0, 0, 0)
    return lane, _champs(cfg, 0, -1, 3), 2


def ahead_held(cfg: GameConfig) -> Start:
    """`ahead` with A's front one cell closer (at -1), in contact with B's front, which therefore stays at the lane
    cell: under champion_movement_rule "tick" A is in B's front's cell after each tick, not only at the turn's start."""
    lane = State(-1, 8, (), 0, 0, 8, (), 0, 0, 0, 0, 0)
    return lane, _champs(cfg, 0, -1, 3), 2


# the M4.5a movement correction (docs/m4/m4_5/m4_5a_unified_mechanics_report.md §10), level 2, turn 2
def follow(cfg: GameConfig) -> Start:
    """Scenario A: A at the lane cell with its 16-HP front there; B's 2-HP front at A's +1 dies on tick 0."""
    lane = State(0, 16, (), 0, -1, 2, (), 0, 0, 0, 0, 0)
    return lane, _champs(cfg, 0, -2, 2), 2


def chase(cfg: GameConfig) -> Start:
    """Scenario C: A's front at its -2 against B's 16-HP front at A's -1; A at its tower, B two cells behind its wave
    at its own tower: a Trader must walk into B's wave to reach B."""
    lane = State(-2, 8, (), 0, 1, 16, (), 0, 0, 0, 0, 0)
    return lane, _champs(cfg, -2, -2, 2), 2


def walk_back(cfg: GameConfig) -> Start:
    """Scenario D: A stands at B's tower cell, far ahead of its wave (at its -2); B's wave walks toward A's."""
    lane = State(-2, 8, (), 0, -2, 8, (), 0, 0, 0, 0, 0)
    return lane, _champs(cfg, 2, -2, 2), 2


STARTS: Dict[str, Tuple[str, Callable[[GameConfig], Start]]] = {
    "bigwave": ("M4.5a: A's 8-HP front at its -1 against B's 24-HP stack at the lane cell; both champions with their "
                "waves, level 2.", lambda cfg: bigwave(cfg, 2)),
    "bigwave_l18": ("M4.5a: the same lane at level 18, both with 128 resources invested.",
                    lambda cfg: bigwave(cfg, 18, 128.0)),
    "ahead": ("M4.5a: A's champion at the lane cell in B's 8-HP wave, its own front behind at -2; B at its -1.",
              ahead),
    "ahead_held": ("M4.5a: A's champion at the lane cell in B's 8-HP wave, its own front behind at -1 (the fronts "
                   "fight, so B's stays); B at its -1.", ahead_held),
    "follow": ("M4.5a correction, Scenario A: A at the lane cell with its 16-HP front; B's 2-HP front at A's +1.",
               follow),
    "chase": ("M4.5a correction, Scenario C: B's 16-HP front at A's -1, A's front at -2; both champions at their "
              "towers.", chase),
    "walk_back": ("M4.5a correction, Scenario D: A at B's tower cell far ahead of its wave; the waves walk toward "
                  "each other.", walk_back),
}


class WardStrategy:
    """A four-action strategy plus a Ward rule (module docstring). Memoryless: a function of (cfg, lane, champions,
    side, turn index), so a recording replays from its actions."""

    def __init__(self, name: str, inner: st.Strategy, when: str = "favorable", description: str = ""):
        if when not in ("favorable", "always"):
            raise ValueError(when)
        self.name, self.inner, self.when, self.description = name, inner, when, description

    def ward_reason(self, cfg: GameConfig, lane, m4, s: int) -> Optional[str]:
        if not cfg.ward_node or lane.vision_of(s) > 0:
            return None
        if CHAMP_WARD not in champion_legal_actions(lane, m4, s, cfg):
            return None
        if self.when == "always":
            return "own vision down: Ward"
        ex, eh = lane.side(1 - s)[:2]
        if eh <= 0 or -ex >= 0:
            return "own vision down, the enemy wave is not in our half: Ward"
        return None

    def decide(self, cfg: GameConfig, lane, m4, s: int, i: Optional[int] = None) -> Tuple[int, str]:
        c, why = self.inner.decide(cfg, lane, m4, s, i)
        if c in (CHAMP_ABSENT, st.RECALL) or why.startswith(("low HP", "free reset")):
            return c, why
        reason = self.ward_reason(cfg, lane, m4, s)
        return (CHAMP_WARD, reason) if reason else (c, why)

    def act(self, cfg: GameConfig, lane, m4, s: int, i: Optional[int] = None) -> int:
        return self.decide(cfg, lane, m4, s, i)[0]


STRATEGIES = {s.name: s for s in [
    WardStrategy("ward_favorable4", st.LIBRARY["farm4"], "favorable",
                 "Last Hit and Hold; Ward whenever the own vision is down and the enemy wave is not in our half."),
    WardStrategy("ward_always4", st.LIBRARY["farm4"], "always",
                 "Last Hit and Hold; Ward whenever the own vision is down, whatever the wave."),
    WardStrategy("ward_freeze4", st.LIBRARY["freeze4"], "favorable",
                 "Freeze keeper (Thin); Ward whenever the own vision is down and the enemy wave is not in our half."),
]}


__all__ = ["STARTS", "STRATEGIES", "WardStrategy", "bigwave", "ahead", "ahead_held", "follow", "chase", "walk_back"]
