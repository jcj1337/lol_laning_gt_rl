"""Model parameters for the laning stochastic game.

Every number here is a hand-specified modelling parameter. None of them is an
empirical League of Legends statistic. See docs/phase1_implementation_report.md
(section C) for the meaning of each parameter and the equations that use it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Tuple

# Strategic actions. Integer codes are stable and used in serialized output.
SP, SH, F = 0, 1, 2
ACTION_NAMES = {SP: "SlowPush", SH: "Shove", F: "Freeze"}
ACTION_SHORT = {SP: "SP", SH: "SH", F: "F"}

# Crash-size categories.
SMALL, MEDIUM, LARGE = 0, 1, 2
SIZE_NAMES = {SMALL: "small", MEDIUM: "medium", LARGE: "large"}


@dataclass(frozen=True)
class GameConfig:
    # ---- lane geometry (global coordinates; A moves toward +, B toward -) ----
    base: int = 4                 # A base at -base, B base at +base
    tower: int = 2                # A tower cell at -tower, B tower cell at +tower

    # ---- time ----
    ticks_per_turn: int = 2       # movement ticks per agent decision
    spawn_interval: int = 4       # movement ticks between wave spawns
    horizon: int = 32             # number of agent decision turns K

    # ---- wave strength ----
    wave_hp: int = 2              # HP of a freshly spawned packet (W)
    max_stacks: int = 2           # cap: a packet holds at most (1 + max_stacks) waves of HP

    # ---- contact (collision) resolution ----
    # "lanchester":      per-tick minion damage = max(1, floor(h / W))  (each full wave adds firepower)
    # "lanchester_ceil": per-tick minion damage = ceil(h / W)
    # "linear":          per-tick minion damage = 1                    (size only buys staying power)
    # "pressure":        per tick, the side with lower pressure h + kappa loses 1 HP, ties both lose 1
    damage_law: str = "lanchester"
    shove_damage: int = 1         # extra damage per tick dealt by a shoving champion
    tower_damage: int = 1         # damage per tick a tower deals to a contested attacker in its cell
    pressure_kappa: Tuple[int, int, int] = (0, 1, 0)   # (SP, SH, F) bonus, "pressure" law only

    # ---- action movement rules (own-perspective coordinates) ----
    # A free leading packet may not advance past these lines under the given action.
    # Shove has no limit (the packet walks into the enemy tower and crashes).
    slowpush_line: int = 0        # Slow Push parks the leading packet at the river
    slowpush_mode: str = "hold"   # past the line: "hold" = stop, "half" = 1 cell/turn, "creep" = 1 cell/spawn interval
    freeze_line: int = -1         # Freeze holds own leading packet one cell in front of own tower
    # Freeze also holds the ENEMY leading packet at the freezer's freeze line unless
    # enemy_hp + freeze_break_push[enemy_action] >= freeze_break_threshold.
    freeze_break_threshold: int = 5
    freeze_break_push: Tuple[int, int, int] = (0, 2, 0)   # (SP, SH, F)
    # Freeze is legal only if the enemy leading packet is at own-perspective x <= this.
    freeze_legal_max: int = 0

    # ---- tower ----
    tower_consumption_per_turn: int = 1   # sieging packets removed per agent turn

    # ---- crash rewards: options per size category, each (resources, grants_vision) ----
    crash_options: Tuple[Tuple[Tuple[float, bool], ...], ...] = (
        ((1.0, False), (0.0, True)),      # small
        ((2.0, False), (0.0, True)),      # medium
        ((3.0, False), (2.0, True)),      # large
    )
    vision_turns: int = 2

    # ---- gank model (see report section C.8 for the calibration targets) ----
    # p = clip(base + position*(e+T)/(2T) + action[a] - strength*dsigma, 0, max) * (1 - rho*[vision])
    gank_base: float = 0.15               # position-independent threat; cancels in the zero-sum payoff except via vision
    gank_position: float = 0.10           # added linearly as exposure goes from own tower (-T) to enemy tower (+T)
    gank_action: Tuple[float, float, float] = (0.0, 0.05, -0.02)   # (SP, SH, F)
    gank_strength: float = 0.02           # risk change per stack of (own - enemy) leading-wave advantage
    gank_max: float = 0.40
    vision_reduction: float = 0.5         # rho: multiplicative risk reduction while vision is active
    gank_loss: float = 5.0                # L: resources lost when ganked (same L as the original project)
    # Champion posture: furthest own-perspective position the champion follows its
    # leading wave under each action; None = follows it all the way (default).
    posture_cap: Tuple = (None, None, None)   # (SP, SH, F)

    # ------------------------------------------------------------------ helpers
    @property
    def hp_cap(self) -> int:
        return self.wave_hp * (1 + self.max_stacks)

    @property
    def spawn_every_turns(self) -> int:
        assert self.spawn_interval % self.ticks_per_turn == 0
        return self.spawn_interval // self.ticks_per_turn

    def stacks(self, hp: int) -> int:
        """Stacks carried by a packet with `hp`: whole extra waves beyond the first."""
        if hp <= 0:
            return 0
        return min(self.max_stacks, -(-hp // self.wave_hp) - 1)

    def size_category(self, hp: int) -> int:
        return self.stacks(min(hp, self.hp_cap))

    def validate(self) -> None:
        assert self.base > self.tower > 0
        assert self.spawn_interval % self.ticks_per_turn == 0
        assert self.damage_law in ("lanchester", "lanchester_ceil", "linear", "pressure")
        assert self.slowpush_mode in ("hold", "half", "creep")
        assert len(self.crash_options) == self.max_stacks + 1
        assert 0 <= self.vision_reduction <= 1 and 0 <= self.gank_max <= 1
        assert -self.tower < self.freeze_line < self.tower

    def with_(self, **kw) -> "GameConfig":
        return replace(self, **kw)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return json.loads(json.dumps(d))

    @classmethod
    def from_dict(cls, d: Dict) -> "GameConfig":
        def tup(x):
            return tuple(tup(i) for i in x) if isinstance(x, list) else x
        return cls(**{k: tup(v) for k, v in d.items()})
