"""Model parameters for the laning stochastic game.

Every number here is a hand-specified modelling parameter. None of them is an
empirical League of Legends statistic. Three model generations are available:

  GameConfig()      the current default: the siege revision documented in
                    docs/phase1_siege_vision_study.md (siege objectives Apply
                    Pressure / Establish Vision, finite tower plates, thinning,
                    persistent vision)
  GameConfig.v2()   the previous revision (docs/phase1_revision_report.md):
                    farming, Contest/Safe, post-crash resource/vision choice
  GameConfig.v1()   the original Phase 1 model (docs/phase1_implementation_report.md)

Both presets reproduce their reports exactly.

Actions
-------
A player's action is a triple (wave management, positioning, siege objective),
encoded as one integer

    a = wave + 3 * posture + 6 * objective

    wave       SP = 0 (Slow Push), SH = 1 (Shove), F = 2 (Freeze)
    posture    CONTEST = 0, SAFE = 1
    objective  NONE = 0, PRESSURE = 1 (Apply Pressure), VISION = 2 (Establish Vision)

Codes 0..5 are the six combined actions of the previous revision (wave x
posture, no siege objective); codes 6..11 and 12..17 add Apply Pressure and
Establish Vision. A siege objective is legal only in `siege_mode="objective"`,
only while the player has crashed minions under the enemy tower at the start
of the turn, and only with Contest; in that situation Contest *requires* an
objective (see mechanics.legal_actions). In the original model (positioning
disabled) only codes 0, 1, 2 are legal and they behave exactly like the
original SP, SH and F actions, which is why the constants SP, SH, F double as
action codes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Dict, Tuple

# Wave-management components. As action codes they mean "with Contest, no objective".
SP, SH, F = 0, 1, 2
CONTEST, SAFE = 0, 1
NONE, PRESSURE, VISION = 0, 1, 2
SP_C, SH_C, F_C, SP_S, SH_S, F_S = 0, 1, 2, 3, 4, 5
N_BASE_ACTIONS = 6            # wave x posture
N_OBJECTIVES = 3
N_ACTIONS = N_BASE_ACTIONS * N_OBJECTIVES


def act(wave: int, posture: int, objective: int = NONE) -> int:
    """Combined action code for (wave management, positioning, siege objective)."""
    return wave + 3 * posture + 6 * objective


def wave_of(a: int) -> int:
    return a % 3


def posture_of(a: int) -> int:
    return (a % 6) // 3


def objective_of(a: int) -> int:
    return a // 6


def base_of(a: int) -> int:
    """The (wave, posture) part of an action: its code without the siege objective."""
    return a % 6


def with_objective(a: int, objective: int) -> int:
    return base_of(a) + 6 * objective


WAVE_NAMES = {SP: "SlowPush", SH: "Shove", F: "Freeze"}
WAVE_SHORT = {SP: "SP", SH: "SH", F: "F"}
POSTURE_NAMES = {CONTEST: "Contest", SAFE: "Safe"}
OBJECTIVE_NAMES = {NONE: "", PRESSURE: "Pressure", VISION: "Vision"}
OBJECTIVE_SHORT = {NONE: "", PRESSURE: "P", VISION: "V"}
ACTION_NAMES = {a: f"{WAVE_NAMES[wave_of(a)]}+{POSTURE_NAMES[posture_of(a)]}"
                + (f"+{OBJECTIVE_NAMES[objective_of(a)]}" if objective_of(a) else "")
                for a in range(N_ACTIONS)}
ACTION_SHORT = {a: f"{WAVE_SHORT[wave_of(a)]}-{'CS'[posture_of(a)]}"
                + (f"+{OBJECTIVE_SHORT[objective_of(a)]}" if objective_of(a) else "")
                for a in range(N_ACTIONS)}

# ---- champion mode (roadmap M2): a = wave + 3 * behavior + 15 * objective, plus ABSENT ----
# Behaviors HOLD (Farm-Hold), YIELD (Farm-Yield), TRADE, SAFE are the lane behaviors; RECALL is legal only as
# the single code champ_act(SP, B_RECALL) (= 12) near one's own tower (champions.recall_legal); ABSENT (45) is
# the only legal code while a champion is away (recall or death). B_ABSENT is the behavior an ABSENT code
# decodes to; it is not part of the 45-code product. Legacy configurations keep the codes above.
B_HOLD, B_YIELD, B_TRADE, B_SAFE, B_RECALL, B_ABSENT = 0, 1, 2, 3, 4, 5
CHAMP_BEHAVIORS = (B_HOLD, B_YIELD, B_TRADE, B_SAFE)      # lane behaviors (present champion)
N_BEHAVIOR_SLOTS = 5
CHAMP_ABSENT = 3 * N_BEHAVIOR_SLOTS * N_OBJECTIVES        # 45
N_CHAMP_ACTIONS = CHAMP_ABSENT + 1                        # 46


def champ_act(wave: int, behavior: int, objective: int = NONE) -> int:
    """Champion-mode action code for (wave management, behavior, siege objective)."""
    return wave + 3 * behavior + 15 * objective


def champ_wave(a: int) -> int:
    return a % 3


def champ_behavior(a: int) -> int:
    return (a % 15) // 3


def champ_objective(a: int) -> int:
    return a // 15


BEHAVIOR_NAMES = {B_HOLD: "Hold", B_YIELD: "Yield", B_TRADE: "Trade", B_SAFE: "Safe", B_RECALL: "Recall",
                  B_ABSENT: "Absent"}
BEHAVIOR_SHORT = {B_HOLD: "H", B_YIELD: "Y", B_TRADE: "T", B_SAFE: "S", B_RECALL: "R", B_ABSENT: "A"}
CHAMP_ACTION_NAMES = {a: f"{WAVE_NAMES[champ_wave(a)]}+{BEHAVIOR_NAMES[champ_behavior(a)]}"
                      + (f"+{OBJECTIVE_NAMES[champ_objective(a)]}" if champ_objective(a) else "")
                      for a in range(CHAMP_ABSENT)}
CHAMP_ACTION_NAMES[CHAMP_ABSENT] = "Absent"
CHAMP_ACTION_SHORT = {a: f"{WAVE_SHORT[champ_wave(a)]}-{BEHAVIOR_SHORT[champ_behavior(a)]}"
                      + (f"+{OBJECTIVE_SHORT[champ_objective(a)]}" if champ_objective(a) else "")
                      for a in range(CHAMP_ABSENT)}
CHAMP_ACTION_SHORT[CHAMP_ABSENT] = "ABSENT"

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
    slowpush_line: int = 0        # Slow Push: full speed up to this line ("the lane": the midpoint cell x = 0) ...
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
    tower_siege_damage: int = 1           # "mass" mode, HP per tick (siege study section E compares 1, 2, 3)
    # "mass" mode: a defender front waiting next to its own tower fights the crashed mass under it
    # (mutual Lanchester attrition); False (default) = it waits behind the tower while tower fire
    # clears the mass, which is what lets reinforcements accumulate into a bounce.
    siege_engage: bool = False
    siege_cap: int = 48                   # "mass" mode, cap on HP under one tower (excess is lost; never
                                          # reached from the standard start under the default model)

    # ---- siege decisions (siege study, section C) ----
    # "objective": while a player has crashed mass under the enemy tower at the start of a turn, its
    #              Contest actions carry a siege objective chosen simultaneously with everything else:
    #              Apply Pressure (claim a tower plate and thin accessible defending minions) or
    #              Establish Vision (persistent gank-risk reduction from the next turn on). Safe =
    #              disengage. Crashes themselves pay nothing; crash_options is unused.
    # "choice":    the previous revision: a crash is followed by a separately resolved choice between
    #              crash resources and a short vision (crash_options), independent of posture.
    siege_mode: str = "objective"
    # Plates (state fields ap / bp = plates side A / B can still claim on the enemy tower):
    # "crash": every crash grants plate_grant[size] plate opportunities (capped at plates_cap in total)
    #          that can be claimed, plates_per_turn per Apply Pressure turn, only while that siege
    #          lasts; unclaimed opportunities lapse when the crashed mass is cleared.
    # "tower": a finite budget of plates_per_tower per tower for the whole game, claimed
    #          plates_per_turn per Apply Pressure turn (a persistent state variable; kept as a
    #          variant because it multiplies the reachable state space, see the siege study).
    plate_mode: str = "crash"
    plate_grant: Tuple[int, int, int] = (1, 2, 3)   # "crash" mode: opportunities per small/medium/large crash
    plates_cap: int = 6                   # "crash" mode: cap on pending opportunities
    plates_per_tower: int = 5             # "tower" mode: budget per tower
    plates_per_turn: int = 1              # plates claimed by one Apply Pressure turn (if any remain)
    plate_value: float = 1.0              # resources per plate (one fresh wave of farm)
    # Thinning: an Apply Pressure champion removes up to thin_damage HP per TURN from the defender's
    # leading packet if it stands adjacent to or inside its own tower cell (own x in [-T-1, -T]), on a
    # tick where the tower is still occupied by the attacker's crashed mass and no lane fight involves
    # that packet. Removed HP is an ordinary farm opportunity for the attacker at the place it died.
    thin_damage: int = 1
    thin_reach: int = 1                   # cells behind the enemy tower cell that are accessible

    # ---- crash rewards ("choice" siege mode only): options per size category, each (resources, grants_vision) ----
    crash_options: Tuple[Tuple[Tuple[float, bool], ...], ...] = (
        ((1.0, False), (0.0, True)),      # small
        ((2.0, False), (0.0, True)),      # medium
        ((3.0, False), (2.0, True)),      # large
    )
    # ---- vision ----
    vision_turns: int = 6                 # protected decision turns after the turn vision is established
    vision_reduction: float = 0.75        # rho: multiplicative gank-risk reduction while vision is active

    # ---- farming economy (revision report section D) ----
    # Every HP of minions removed by a fight, by a tower or by thinning is a farm opportunity worth
    # farm_per_hp resources to the opponent of the minions' owner. The collector receives
    # the opportunity times a collection efficiency c in [0, 1]:
    #   c = base(posture, x) * (1 - zone(opponent posture, x)),  x = collector's own position of the kill
    #   base(Contest, x) = collect_contest            (x < T)
    #                      collect_contest_enemy_tower (x >= T, under the enemy tower)
    #   base(Safe, x)    = clip(collect_safe - collect_safe_slope * (min(x, T) + T), 0, 1)
    #   zone(Contest, x) = zone_contest (x > -T) or zone_under_tower (x = -T); zone(Safe, x) = 0
    farm_per_hp: float = 0.25     # 1 resource per fresh wave (W = 4)
    collect_contest: float = 0.9
    collect_contest_enemy_tower: float = 0.5
    collect_safe: float = 0.8
    collect_safe_slope: float = 0.2
    zone_contest: float = 0.3
    zone_under_tower: float = 0.15

    # ---- gank model (report section C.8) ----
    # p = clip(base + position*(e+T)/(2T) + action[base action] - strength*dsigma, 0, max) * (1 - rho*[vision])
    gank_base: float = 0.15               # position-independent threat; cancels in the zero-sum payoff except via vision
    gank_position: float = 0.10           # added linearly as exposure goes from own tower (-T) to enemy tower (+T)
    # per base action code (SP-C, SH-C, F-C, SP-S, SH-S, F-S); siege objectives inherit their base action's term
    gank_action: Tuple[float, ...] = (0.02, 0.05, 0.02, 0.0, 0.0, 0.0)
    gank_strength: float = 0.02           # risk change per stack of (own - enemy) leading-wave advantage
    gank_max: float = 0.40
    gank_loss: float = 5.0                # L: resources lost when ganked (same L as the original project)
    # Champion exposure: furthest own-perspective position the champion follows its leading
    # wave under each base action; None = follows it all the way (Contest).
    posture_cap: Tuple = (None, None, None, -1, -1, -1)

    # ---- champion layer (roadmap M2; lanegame/champions.py). Off in every legacy preset. ----
    # In champion mode each player controls a champion with a persistent cell (own coordinates -T..T),
    # HP, an absence timer and a per-seat mechanical skill. Actions are (wave management, behavior, siege
    # objective); see champions.py for the turn order.
    champion_mode: bool = False
    champion_hp: int = 6                  # maximum champion HP (restored on return from a recall or death)
    champion_speed: int = 2               # cells a champion can move per decision turn
    champion_absent_max: int = 3          # largest absence timer (observation scale): >= max(recall_turns - 1, death_turns)
    # Recall (M2b): a champion at or behind recall_line with no enemy champion in contact may Recall; it leaves
    # the lane at once, misses recall_turns turns in total (the Recall turn itself included) and returns at its
    # own tower with full HP. With the default 2: Recall on turn t, forced ABSENT on t + 1, back on t + 2.
    recall_turns: int = 2
    recall_line: int = -1
    # Trading (M2c). A connected Trade by attacker i on defender j draws damage 0 / 1 / 2 HP:
    #   P(hit) h = clip(trade_hit_base + trade_hit_skill * (m_i - m_j) + trade_hit_behavior[behavior of j]
    #                   + trade_hit_minions * (sigma_i - sigma_j) - trade_hit_tower * [j in its own tower cell],
    #                   trade_hit_min, trade_hit_max)
    #   P(2 HP | hit) q = clip(trade_double_base + trade_double_skill * (m_i - m_j), 0, 1)
    # with m the mechanical skills and sigma the leading-wave stack counts at the start of the turn (local
    # minion support, as in the gank formula). Skill enters only through the (m_i - m_j) terms: the
    # attacker's skill raises its execution, the defender's lowers it (evasion / mitigation).
    # trade_hit_behavior is indexed by the defender's behavior (Hold, Yield, Trade, Safe, Recall): a Yield
    # or Safe champion that could not get out of reach is still partly evading; a Hold champion (farming)
    # or an interrupted recall (channeling) is fully exposed; a Trade defender is trading back.
    trade_hit_base: float = 0.55
    trade_hit_skill: float = 0.4
    trade_hit_behavior: Tuple[float, float, float, float, float] = (0.0, -0.25, 0.0, -0.40, 0.0)
    trade_hit_minions: float = 0.05
    trade_hit_tower: float = 0.15
    trade_hit_min: float = 0.05
    trade_hit_max: float = 0.95
    trade_double_base: float = 0.3
    trade_double_skill: float = 0.2
    # Tower shot (M2c): a connected Trade that ends in the enemy tower cell (a dive) takes tower_shot_damage HP
    # with probability tower_shot_p, or tower_shot_p_busy while the diver's own crashed minions are under that
    # tower (part of the tower's attention is on them).
    tower_shot_damage: int = 1
    tower_shot_p: float = 1.0
    tower_shot_p_busy: float = 0.5
    skill_a: float = 0.5                  # mechanical skill of the player in seat A, in [0, 1]
    skill_b: float = 0.5                  # ... seat B
    safe_line: int = -1                   # Safe (and a threatened Yield) retreats to at most this cell
    trade_range: int = 1                  # a Trade connects when the champions end within this distance
    farm_reach: int = 1                   # a champion collects only kills within this many cells of it
    trade_farm_factor: float = 0.5        # collection multiplier of a champion whose Trade connected
    # gank action term per behavior (Hold, Yield, Trade, Safe), plus a term for Shove with any non-Safe
    # behavior: Hold reproduces the legacy Contest terms (SP 0.02, SH 0.05, F 0.02), Safe the Safe terms (0)
    champ_gank_behavior: Tuple[float, float, float, float] = (0.02, 0.02, 0.05, 0.0)
    champ_gank_shove: float = 0.03

    # ------------------------------------------------------------------ presets
    @classmethod
    def v1(cls, **kw) -> "GameConfig":
        """The original Phase 1 model (docs/phase1_implementation_report.md), exactly."""
        base = dict(positioning=False, slowpush_mode="hold", freeze_legal_max=0, cap_rule="overflow",
                    tower_mode="packet", farm_per_hp=0.0, wave_hp=2, freeze_break_threshold=5,
                    freeze_break_push=(0, 2, 0), siege_cap=12, slowpush_creep_period=1, siege_engage=False,
                    gank_action=(0.0, 0.05, -0.02, 0.0, 0.05, -0.02),
                    posture_cap=(None,) * 6,
                    siege_mode="choice", vision_turns=2, vision_reduction=0.5, plate_grant=(0, 0, 0),
                    plates_per_tower=0, thin_damage=0)
        base.update(kw)
        return cls(**base)

    @classmethod
    def v2(cls, **kw) -> "GameConfig":
        """The previous revision (docs/phase1_revision_report.md): post-crash resource/vision choice,
        two turns of 50% vision, 1 HP/tick tower fire, no plates, no thinning. Reproduces its report exactly."""
        base = dict(siege_mode="choice", vision_turns=2, vision_reduction=0.5, plate_grant=(0, 0, 0),
                    plates_per_tower=0, thin_damage=0, tower_siege_damage=1)
        base.update(kw)
        return cls(**base)

    @classmethod
    def champions(cls, **kw) -> "GameConfig":
        """Champion mode (roadmap M2a): the siege model's lane plus the champion layer of lanegame.champions.
        Not exactly solvable; played through lanegame.env."""
        base = dict(champion_mode=True)
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

    @property
    def objectives(self) -> bool:
        """Siege objectives (Apply Pressure / Establish Vision) are part of the action set."""
        return self.siege_mode == "objective"

    def stacks(self, hp: int) -> int:
        """Stacks carried by a packet with `hp`: whole extra waves beyond the first."""
        if hp <= 0:
            return 0
        return min(self.max_stacks, -(-hp // self.wave_hp) - 1)

    def size_category(self, hp: int) -> int:
        return self.stacks(min(hp, self.hp_cap))

    def action_codes(self) -> Tuple[int, ...]:
        """Every action slot of this configuration's fixed action space (champion mode: the full 46-slot space;
        of its Recall codes only champ_act(SP, B_RECALL) is ever legal)."""
        if self.champion_mode:
            return tuple(range(N_CHAMP_ACTIONS))
        if not self.positioning:
            return (SP, SH, F)
        if self.objectives:
            return tuple(range(N_ACTIONS))
        return tuple(range(N_BASE_ACTIONS))

    def label(self, a: int) -> str:
        """Short action label: 'SH-C', 'SH-C+P' etc., or 'SH' when positioning is disabled; champion mode
        'SH-T+P' (Shove + Trade + Pressure) etc."""
        if self.champion_mode:
            return CHAMP_ACTION_SHORT[a]
        return ACTION_SHORT[a] if self.positioning else WAVE_SHORT[wave_of(a)]

    def validate(self) -> None:
        assert self.base > self.tower > 0
        assert self.spawn_interval % self.ticks_per_turn == 0
        assert self.damage_law in ("lanchester", "lanchester_ceil", "linear", "pressure")
        assert self.slowpush_mode in ("hold", "half", "creep", "stack_creep", "free")
        assert self.tower_mode in ("packet", "mass")
        assert self.cap_rule in ("block", "overflow")
        assert self.siege_mode in ("choice", "objective")
        assert len(self.crash_options) == self.max_stacks + 1
        assert len(self.gank_action) == N_BASE_ACTIONS and len(self.posture_cap) == N_BASE_ACTIONS
        assert 0 <= self.vision_reduction <= 1 and 0 <= self.gank_max <= 1
        assert self.vision_turns >= 0
        assert -self.tower < self.freeze_line < self.tower
        assert self.siege_cap >= self.hp_cap
        assert self.plate_mode in ("crash", "tower")
        assert self.plates_per_tower >= 0 and self.plates_per_turn >= 0 and self.plate_value >= 0
        assert len(self.plate_grant) == self.max_stacks + 1 and min(self.plate_grant) >= 0
        assert self.plates_cap >= max(self.plate_grant)
        assert self.thin_damage >= 0 and self.thin_reach >= 0
        if self.objectives:
            assert self.positioning, "siege objectives require Contest/Safe positioning"
            assert self.tower_mode == "mass", "siege objectives need crashed HP mass under the tower"
        for c in (self.collect_contest, self.collect_contest_enemy_tower, self.collect_safe,
                  self.zone_contest, self.zone_under_tower):
            assert 0.0 <= c <= 1.0
        assert self.farm_per_hp >= 0 and self.collect_safe_slope >= 0
        if self.champion_mode:
            assert self.objectives and self.tower_mode == "mass", "champion mode builds on the siege model"
            assert self.champion_hp >= 1 and self.champion_speed >= 1 and self.champion_absent_max >= 1
            assert self.recall_turns >= 1 and -self.tower <= self.recall_line <= self.tower
            assert self.champion_absent_max >= self.recall_turns - 1
            assert len(self.trade_hit_behavior) == N_BEHAVIOR_SLOTS
            assert 0.0 <= self.trade_hit_min <= self.trade_hit_max <= 1.0
            assert self.tower_shot_damage >= 0 and 0.0 <= self.tower_shot_p <= 1.0 and 0.0 <= self.tower_shot_p_busy <= 1.0
            assert 0.0 <= self.skill_a <= 1.0 and 0.0 <= self.skill_b <= 1.0
            assert -self.tower <= self.safe_line <= self.tower
            assert self.trade_range >= 0 and self.farm_reach >= 0 and 0.0 <= self.trade_farm_factor <= 1.0
            assert len(self.champ_gank_behavior) == len(CHAMP_BEHAVIORS)

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
