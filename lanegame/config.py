"""Model parameters for the laning stochastic game.

Every number here is a hand-specified modelling parameter. None of them is an
empirical League of Legends statistic. The default configuration is the
revised model documented in docs/phase1_revision_report.md (section C);
`GameConfig.v1()` reproduces the original Phase 1 model of
docs/phase1_implementation_report.md exactly.

Actions
-------
A player's action is a pair (wave management, positioning), encoded as one
integer a = wave + 3 * posture:

    wave     SP = 0 (Slow Push), SH = 1 (Shove), F = 2 (Freeze)
    posture  CONTEST = 0, SAFE = 1

so codes 0, 1, 2 are SP/SH/F with Contest and 3, 4, 5 are SP/SH/F with Safe.
In the original model (positioning disabled) only codes 0, 1, 2 are legal and
they behave exactly like the original SP, SH and F actions, which is why the
constants SP, SH, F double as action codes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Dict, Tuple

# Wave-management components. As action codes they mean "with Contest".
SP, SH, F = 0, 1, 2
CONTEST, SAFE = 0, 1
SP_C, SH_C, F_C, SP_S, SH_S, F_S = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6


def act(wave: int, posture: int) -> int:
    """Combined action code for (wave management, positioning)."""
    return wave + 3 * posture


def wave_of(a: int) -> int:
    return a % 3


def posture_of(a: int) -> int:
    return a // 3


WAVE_NAMES = {SP: "SlowPush", SH: "Shove", F: "Freeze"}
WAVE_SHORT = {SP: "SP", SH: "SH", F: "F"}
POSTURE_NAMES = {CONTEST: "Contest", SAFE: "Safe"}
ACTION_NAMES = {a: f"{WAVE_NAMES[wave_of(a)]}+{POSTURE_NAMES[posture_of(a)]}" for a in range(N_ACTIONS)}
ACTION_SHORT = {a: f"{WAVE_SHORT[wave_of(a)]}-{'CS'[posture_of(a)]}" for a in range(N_ACTIONS)}

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
    wave_hp: int = 4              # HP of a freshly spawned packet (W); the original model used 2
    max_stacks: int = 2           # cap: a packet holds at most (1 + max_stacks) waves of HP
    # What happens when a merge would exceed the cap: "block" = the reinforcement stays a
    # separate packet queued behind the leading one (no minion disappears); "overflow" =
    # the excess HP is deleted (original model).
    cap_rule: str = "block"
    # "block" rule: packets queued within one cell behind the leading packet fight and push with it
    # (their HP counts toward its firepower and toward breaking a freeze). False = they stand idle.
    queue_fights: bool = True

    # ---- positioning (Contest / Safe) ----
    positioning: bool = True      # False: only the Contest actions 0, 1, 2 are legal (original model)

    # ---- contact (collision) resolution ----
    # "lanchester":      per-tick minion damage = max(1, floor(h / W))  (each full wave adds firepower)
    # "lanchester_ceil": per-tick minion damage = ceil(h / W)
    # "linear":          per-tick minion damage = 1                    (size only buys staying power)
    # "pressure":        per tick, the side with lower pressure h + kappa loses 1 HP, ties both lose 1
    damage_law: str = "lanchester"
    shove_damage: int = 1         # extra damage per tick dealt by a shoving champion present at the fight
    tower_damage: int = 1         # damage per tick a tower deals to a contested attacker in its cell
    pressure_kappa: Tuple[int, int, int] = (0, 1, 0)   # (SP, SH, F) bonus, "pressure" law only

    # ---- wave-management movement rules (own-perspective coordinates) ----
    # A free leading packet may not advance past these lines under the given wave action.
    # Shove has no limit (the packet walks into the enemy tower and crashes).
    slowpush_line: int = 0        # Slow Push: full speed up to this line (the river) ...
    # ... and past it: "hold" = stop; "half" = 1 cell per turn; "creep" = 1 cell per spawn interval;
    # "stack_creep" = stop while the packet carries no stack, creep once it carries one;
    # "free" = no line at all (Slow Push differs from Shove only by the champion's damage).
    slowpush_mode: str = "stack_creep"
    slowpush_creep_period: int = 2        # "creep"/"stack_creep": spawn intervals per cell advanced
    freeze_line: int = -1         # Freeze holds own leading packet one cell in front of own tower
    # Freeze also holds the ENEMY leading packet at the freezer's freeze line unless
    # enemy_hp + freeze_break_push[enemy wave action] >= freeze_break_threshold.
    # The push bonus requires the breaking champion to Contest.
    freeze_break_threshold: int = 10      # 2.5 waves (the original model: 5 with W = 2)
    freeze_break_push: Tuple[int, int, int] = (0, 4, 0)   # (SP, SH, F); one wave of push (original: 2)
    # Freeze is legal only if the enemy leading packet is at own-perspective x <= this.
    freeze_legal_max: int = -1

    # ---- tower ----
    # "packet": each turn the tower removes one packet that was queued at the start of the
    #           turn, whatever its size (original model; siege field = packet count).
    # "mass":   each tick the tower deals tower_siege_damage HP to the crashed mass that was
    #           already under it at the start of the tick (siege field = HP under the tower).
    tower_mode: str = "mass"
    tower_consumption_per_turn: int = 1   # "packet" mode
    tower_siege_damage: int = 1           # "mass" mode, HP per tick
    # "mass" mode: a defender front waiting next to its own tower fights the crashed mass under it
    # (mutual Lanchester attrition); False (default) = it waits behind the tower while tower fire
    # clears the mass, which is what lets reinforcements accumulate into a bounce.
    siege_engage: bool = False
    siege_cap: int = 48                   # "mass" mode, cap on HP under one tower (excess is lost; never
                                          # reached from the standard start under the default model)

    # ---- crash rewards: options per size category, each (resources, grants_vision) ----
    crash_options: Tuple[Tuple[Tuple[float, bool], ...], ...] = (
        ((1.0, False), (0.0, True)),      # small
        ((2.0, False), (0.0, True)),      # medium
        ((3.0, False), (2.0, True)),      # large
    )
    vision_turns: int = 2

    # ---- farming economy (revision report section D) ----
    # Every HP of minions removed by a fight or by a tower is a farm opportunity worth
    # farm_per_hp resources to the opponent of the minions' owner. The collector receives
    # the opportunity times a collection efficiency c in [0, 1]:
    #   c = base(posture, x) * (1 - zone(opponent posture, x)),  x = collector's own position of the kill
    #   base(Contest, x) = collect_contest            (x < T)
    #                      collect_contest_enemy_tower (x = T, under the enemy tower)
    #   base(Safe, x)    = clip(collect_safe - collect_safe_slope * (x + T), 0, 1)
    #   zone(Contest, x) = zone_contest (x > -T) or zone_under_tower (x = -T); zone(Safe, x) = 0
    farm_per_hp: float = 0.25     # 1 resource per fresh wave (W = 4)
    collect_contest: float = 0.9
    collect_contest_enemy_tower: float = 0.5
    collect_safe: float = 0.8
    collect_safe_slope: float = 0.2
    zone_contest: float = 0.3
    zone_under_tower: float = 0.15

    # ---- gank model (report section C.8) ----
    # p = clip(base + position*(e+T)/(2T) + action[a] - strength*dsigma, 0, max) * (1 - rho*[vision])
    gank_base: float = 0.15               # position-independent threat; cancels in the zero-sum payoff except via vision
    gank_position: float = 0.10           # added linearly as exposure goes from own tower (-T) to enemy tower (+T)
    # per combined action code (SP-C, SH-C, F-C, SP-S, SH-S, F-S)
    gank_action: Tuple[float, ...] = (0.02, 0.05, 0.02, 0.0, 0.0, 0.0)
    gank_strength: float = 0.02           # risk change per stack of (own - enemy) leading-wave advantage
    gank_max: float = 0.40
    vision_reduction: float = 0.5         # rho: multiplicative risk reduction while vision is active
    gank_loss: float = 5.0                # L: resources lost when ganked (same L as the original project)
    # Champion exposure: furthest own-perspective position the champion follows its leading
    # wave under each combined action; None = follows it all the way (Contest).
    posture_cap: Tuple = (None, None, None, -1, -1, -1)

    # ------------------------------------------------------------------ presets
    @classmethod
    def v1(cls, **kw) -> "GameConfig":
        """The original Phase 1 model (docs/phase1_implementation_report.md), exactly."""
        base = dict(positioning=False, slowpush_mode="hold", freeze_legal_max=0, cap_rule="overflow",
                    tower_mode="packet", farm_per_hp=0.0, wave_hp=2, freeze_break_threshold=5,
                    freeze_break_push=(0, 2, 0), siege_cap=12, slowpush_creep_period=1, siege_engage=False,
                    gank_action=(0.0, 0.05, -0.02, 0.0, 0.05, -0.02),
                    posture_cap=(None,) * 6)
        base.update(kw)
        return cls(**base)

    # ------------------------------------------------------------------ helpers
    @property
    def hp_cap(self) -> int:
        return self.wave_hp * (1 + self.max_stacks)

    @property
    def spawn_every_turns(self) -> int:
        assert self.spawn_interval % self.ticks_per_turn == 0
        return self.spawn_interval // self.ticks_per_turn

    @property
    def phase_period(self) -> int:
        """Period (in decision turns) of the lane dynamics' dependence on k.

        Spawns repeat every spawn_every_turns turns; a creeping Slow Push moves on one
        turn in every spawn_every_turns * slowpush_creep_period. Transitions from the same
        state are identical at k and k + phase_period.
        """
        p = self.spawn_every_turns
        if self.slowpush_mode in ("creep", "stack_creep"):
            p *= self.slowpush_creep_period
        return p

    def stacks(self, hp: int) -> int:
        """Stacks carried by a packet with `hp`: whole extra waves beyond the first."""
        if hp <= 0:
            return 0
        return min(self.max_stacks, -(-hp // self.wave_hp) - 1)

    def size_category(self, hp: int) -> int:
        return self.stacks(min(hp, self.hp_cap))

    def action_codes(self) -> Tuple[int, ...]:
        return tuple(range(N_ACTIONS)) if self.positioning else (SP, SH, F)

    def label(self, a: int) -> str:
        """Short action label: 'SH-C' etc., or 'SH' when positioning is disabled."""
        return ACTION_SHORT[a] if self.positioning else WAVE_SHORT[wave_of(a)]

    def validate(self) -> None:
        assert self.base > self.tower > 0
        assert self.spawn_interval % self.ticks_per_turn == 0
        assert self.damage_law in ("lanchester", "lanchester_ceil", "linear", "pressure")
        assert self.slowpush_mode in ("hold", "half", "creep", "stack_creep", "free")
        assert self.tower_mode in ("packet", "mass")
        assert self.cap_rule in ("block", "overflow")
        assert len(self.crash_options) == self.max_stacks + 1
        assert len(self.gank_action) == N_ACTIONS and len(self.posture_cap) == N_ACTIONS
        assert 0 <= self.vision_reduction <= 1 and 0 <= self.gank_max <= 1
        assert -self.tower < self.freeze_line < self.tower
        assert self.siege_cap >= self.hp_cap
        for c in (self.collect_contest, self.collect_contest_enemy_tower, self.collect_safe,
                  self.zone_contest, self.zone_under_tower):
            assert 0.0 <= c <= 1.0
        assert self.farm_per_hp >= 0 and self.collect_safe_slope >= 0

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
