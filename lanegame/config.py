"""Model parameters for the laning stochastic game.

Every number here is a hand-specified modelling parameter. None of them is an
empirical League of Legends statistic. Three model generations are available:

  GameConfig()      the current default: the siege revision documented in
                    docs/phase1/phase1_siege_vision_study.md (siege objectives Apply
                    Pressure / Establish Vision, finite tower plates, thinning,
                    persistent vision)
  GameConfig.v2()   the previous revision (docs/phase1/phase1_revision_report.md):
                    farming, Contest/Safe, post-crash resource/vision choice
  GameConfig.v1()   the original Phase 1 model (docs/phase1/phase1_implementation_report.md)

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
# M4.2e-A natural waves (GameConfig.m4_natural): the same three slots name what the champion does to enemy minions.
# Slot order is kept for continuity of the action space: the passive slot SP is Hold, the aggressive slot SH is Clear
# (it keeps Shove's gank-exposure term) and the controlled slot F is Thin.
W_HOLD, W_CLEAR, W_THIN = SP, SH, F
# M4.3c (EXPERIMENTAL): the fourth wave slot, Slow Push; it exists only under wave_slowpush (champion-mode codes
# 46..60, see below)
W_SLOWPUSH = 3
# M4.3c stabilization: the four-action model names its full-capacity slot (SH) Shove. W_SHOVE IS W_CLEAR: the same
# slot, the same codes and the same budget C, so no action code, legal mask or checkpoint changes meaning; only the
# names of the four-action model change (the three-action presets keep Hold / Thin / Clear)
W_SHOVE = W_CLEAR
NATURAL_WAVE_NAMES = {W_HOLD: "Hold", W_CLEAR: "Clear", W_THIN: "Thin", W_SLOWPUSH: "SlowPush"}
NATURAL_WAVE_SHORT = {W_HOLD: "HO", W_CLEAR: "CL", W_THIN: "TN", W_SLOWPUSH: "SL"}
# labels of the four-action model (wave_slowpush): its Hold slot, capped at 1 HP per turn, is Last Hit, and its
# full-capacity slot is Shove (labelled Clear / CL in the M4.3c redesign report and its results, which keep their names)
FOUR_WAVE_NAMES = {W_HOLD: "LastHit", W_SHOVE: "Shove", W_THIN: "Thin", W_SLOWPUSH: "SlowPush"}
FOUR_WAVE_SHORT = {W_HOLD: "LH", W_SHOVE: "SH", W_THIN: "TN", W_SLOWPUSH: "SL"}
# readable wave-slot names, ordered from the least to the most wave damage (viewer controls, traces, reports)
FOUR_WAVE_DISPLAY = {W_HOLD: "Last Hit", W_THIN: "Thin", W_SLOWPUSH: "Slow Push", W_SHOVE: "Shove"}
NATURAL_WAVE_DISPLAY = {W_HOLD: "Hold", W_THIN: "Thin", W_CLEAR: "Clear"}
SCRIPTED_WAVE_DISPLAY = {SP: "Slow Push", F: "Freeze", SH: "Shove"}
NATURAL_WAVE_HP = 8
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
# M4.3c four-action wave management (EXPERIMENTAL, GameConfig.m4_four_actions(),
# docs/m4/m4_3c_four_action_wave_redesign.md): a fourth wave slot, Slow Push, whose codes are APPENDED after the
# historical 46 so that no existing code changes meaning in any configuration:
#   codes 0..44   (wave in {SP, SH, F}) x behavior x objective, a = wave + 3 behavior + 15 objective, as before
#   code  45      ABSENT, as before
#   codes 46..60  Slow Push x behavior x objective, a = 46 + behavior + 5 objective (only legal under wave_slowpush)
CHAMP_SLOWPUSH_BASE = N_CHAMP_ACTIONS                                 # 46
N_CHAMP_ACTIONS_FOUR = CHAMP_SLOWPUSH_BASE + N_BEHAVIOR_SLOTS * N_OBJECTIVES   # 61
# M4.5a (EXPERIMENTAL, vision_rule "ward"): the Ward action is the single code Last Hit / Hold slot + Hold + Vision
# (= 30): walk to the shared Vision node W and place a ward; every other Vision code is then never legal
CHAMP_WARD = 0 + 3 * B_HOLD + 15 * VISION                              # 30


def champ_act(wave: int, behavior: int, objective: int = NONE) -> int:
    """Champion-mode action code for (wave management, behavior, siege objective)."""
    if wave == W_SLOWPUSH:
        return CHAMP_SLOWPUSH_BASE + behavior + N_BEHAVIOR_SLOTS * objective
    return wave + 3 * behavior + 15 * objective


def champ_wave(a: int) -> int:
    return W_SLOWPUSH if a >= CHAMP_SLOWPUSH_BASE else a % 3


def champ_behavior(a: int) -> int:
    if a >= CHAMP_SLOWPUSH_BASE:
        return (a - CHAMP_SLOWPUSH_BASE) % N_BEHAVIOR_SLOTS
    return (a % 15) // 3


def champ_objective(a: int) -> int:
    if a >= CHAMP_SLOWPUSH_BASE:
        return (a - CHAMP_SLOWPUSH_BASE) // N_BEHAVIOR_SLOTS
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
    # "lanchester_half": (M4.2e-C, wave_dynamics "natural" only) max(1, floor(2h / W)) per TURN, split over the ticks
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
    # M4.2e-B (docs/natural_waves/m4_2e_zoning_report.md), ruleset "m4" only: the rate at which kills in the collector's OWN tower
    # cell (x = -T: a crashed enemy mass dying to the tower and to the defenders) are collected, whatever the posture
    # (last-hitting under one's tower competes with the tower). The default equals collect_contest (no effect).
    collect_contest_own_tower: float = 0.9
    collect_safe: float = 0.8
    collect_safe_slope: float = 0.2
    zone_contest: float = 0.3
    zone_under_tower: float = 0.15
    # M4.2d-C (docs/m4/m4_zoning_report.md), opt-in, ruleset "m4" only: which opponent applies the zone factor
    # (1 - zone_contest, or 1 - zone_under_tower in the collector's own tower cell) to a champion's gold.
    # "presence" = an opponent in farm or trading mode within farm_reach of the kill (the M2 rule); "contact" = only
    # an opponent whose Trade connected on the collector this turn; "none" = never (gold is then lost only to reach,
    # the Safe / retreat posture, trading and absence)
    zoning_rule: str = "presence"

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
    # A threatened Yield (backing off from a trader) collects kills within retreat_farm_reach cells of its retreat
    # cell, at Safe efficiency: it gives up some farm, not all (M2e: with 1, a Yield champion zoned by a
    # persistent trader collected 1.2 of 15.3 available resources per game; with 2 it keeps about a third).
    retreat_farm_reach: int = 2
    trade_farm_factor: float = 0.5        # collection multiplier of a champion whose Trade connected
    # Ganks as events (M2d). Champion mode replaces the legacy expected-loss gank (gank_loss * p, whose
    # unconditional base gank_base = 0.15 cancelled in the zero-sum payoff) by a gank that occurs with
    #   p = clip(champ_gank_base + champ_gank_position (e + T) / 2T + champ_gank_behavior[behavior]
    #            + champ_gank_shove [Shove, non-Safe] - champ_gank_strength dsigma, 0, champ_gank_max)
    #       * (1 - vision_reduction [own vision active at the start of the turn])
    # (e = exposure, dsigma = own minus enemy leading-wave stacks) and, if it occurs, deals
    #   X ~ Binomial(gank_damage_trials, p_e),
    #   p_e = clip(gank_damage_base + gank_damage_position (e + T) / 2T - gank_damage_defense [Safe or retreating Yield]
    #              - gank_damage_skill (m - 0.5), 0, 1)
    # HP. With 4 trials and 6 HP a full-HP champion never dies to one gank; whether a gank kills depends on HP,
    # position, behavior and skill. There is no unconditional base rate (0.0): a champion under its own tower
    # is almost never ganked, one at the lane about once every 14 turns (Hold), one under the enemy tower
    # about once every 8. Vision lowers only the occurrence.
    champ_gank_base: float = 0.0
    champ_gank_position: float = 0.12
    champ_gank_behavior: Tuple[float, float, float, float] = (0.01, 0.01, 0.03, 0.0)   # Hold, Yield, Trade, Safe
    champ_gank_shove: float = 0.02
    champ_gank_strength: float = 0.01
    champ_gank_max: float = 0.30
    gank_damage_trials: int = 4
    gank_damage_base: float = 0.2
    gank_damage_position: float = 0.6
    gank_damage_defense: float = 0.2
    gank_damage_skill: float = 0.2
    # Death (M2d): a champion whose HP reaches 0 dies at the end of the turn (it keeps that turn's farm), sits
    # out the next death_turns turns and returns at its own tower with full HP. Dying on turn t with the
    # default 3: ABSENT on t+1, t+2, t+3, back on t+4. The cost of a death is mainly the farm, wave control and
    # siege opportunities given up (about 1 to 1.5 resources at the default lane economy); kill_bounty adds a
    # small kill reward to the opponent (reported separately as bounty_payoff). 0.5 resources = half a wave of
    # farm (real kill gold is about 2.4 early waves). M2e evidence: with 0, trading lost to every non-evasive
    # scripted agent and the scripted meta-game was 95% "Hold and never recall", i.e. HP damage had almost no
    # strategic consequence; with 0.5 trading becomes viable but exploitable and the meta-game mixes.
    death_turns: int = 3
    kill_bounty: float = 0.5

    # ---- ruleset contract (docs/m4/m4_implementation_plan.md WP0). Every default reproduces the legacy game. ----
    #   "legacy"  GameConfig(), .v1(), .v2(), .champions(): finite horizon K, terminal, countdown observed, gamma = 1
    #   "m2c"     .champions_continuing(): M2 champion mechanics with M4 time semantics. A TEST FIXTURE for the
    #             continuing-time machinery only; never a research ruleset
    #   "m4"      .m4(): the integrated M4 ruleset (docs/m4/m4_integrated_design.md), lanegame.champions_m4
    ruleset: str = "legacy"
    time_mode: str = "finite"             # "finite" (terminal at the horizon) or "continuing" (no terminal state)
    time_features: bool = True            # k/K and (K - k)/K in the (legacy-layout) observation
    gamma: float = 1.0                    # discount of the game objective; part of the game under "continuing"

    # ---- M4 economy (design §13). Each default means "off"; validate() rejects any change outside ruleset "m4". ----
    xp_per_hp: int = 0                    # XP per enemy minion HP dying within xp_range of a present champion (m4: 2)
    xp_range: int = 3                     # cells (design §7.3)
    xp_base: int = 8                      # XP to advance from level L to L + 1: xp_base + xp_step (L - 1)
    xp_step: int = 3
    level_cap: int = 18
    hp_base: int = 600                    # H(1) in centi-HP; H(L) = round(hp_base hp_growth^(L - 1))
    hp_growth: float = 1.05
    ap_level_growth: float = 1.05         # AP(L, N) = round(10^4 ap_level_growth^(L - 1) item_mult^N) basis points
    item_mult: float = 1.20
    item_tiers: int = 0                   # sequential offensive tiers (m4: 2); 0 = no items
    item_prices: Tuple[float, ...] = (4.0, 6.0)   # resources per tier, bought automatically at base (design §8.5)
    # M4.2c experiment (docs/m4/m4_continuous_investment_report.md): "tiers" = the two-tier track above (the validated
    # baseline); "continuous" = every return to base invests ALL unspent gold, AP(L, S) = AP(L, 0) [1 + invest_alpha
    # ((1 + S)^invest_p - 1)] with S the cumulative investment in resources (needs item_tiers = 0).
    item_model: str = "tiers"
    invest_alpha: float = 0.045
    invest_p: float = 0.90
    invest_obs_gold_ref: int = 1024       # resources: the observation encodes unspent gold exactly below this
    invest_obs_ref: int = 1024            # resources: ... and invested gold exactly below this
    # M4.2d (docs/m4/m4_investment_readiness_report.md), opt-in: invest_curve "power" = the M4.2c curve above;
    # "saturating" = M(S) = 1 + invest_gain (1 - sqrt(invest_scale / (invest_scale + S))), bounded by 1 + invest_gain
    # and computed in exact integers. invest_obs_encoding "linear" = the M4.2c whole-resource entries n / R (saturate at
    # R); "compressed" = n / (n + R) (R = the same reference fields, now the half-point; never saturates, exact
    # decoding up to observation_m4.compressed_exact_max(R) resources).
    invest_curve: str = "power"
    invest_gain: float = 1.5
    invest_scale: float = 16.0            # resources (K): M - 1 reaches 0.29 invest_gain at S = K, 0.5 at 3 K
    invest_obs_encoding: str = "linear"
    freeze_maintenance: bool = False      # m4: a Freeze holds only while maintained (design §6.3)
    chase_rule: str = "m2"                # "m2" (legacy target e -/+ trade_range) or "contact_cell" (design §7.4)
    # M4.2d (docs/m4/m4_wave_readiness_report.md), opt-in: "m2" = the farm spot is the probe's leading packet (M2 rule);
    # "pre_crash_front" = a champion whose front crashes in the probe keeps its pre-turn front cell as farm spot
    farm_spot_rule: str = "m2"
    # M4.2d-B (docs/m4/m4_positioning_report.md), opt-in: "m2" = M2 positioning and evasion; "pin" = a Yield champion that
    # STARTS the turn in contact with an enemy Trade champion cannot evade it (it is caught like a Hold); "standoff" =
    # "pin" plus Yield farms from its stand-off cell max(-T, min(spot, e - farm_reach)), e = the enemy front at the start
    # of the turn in own coordinates: the rearmost cell still in reach of the contested minions
    position_rule: str = "m2"
    # M4.2e-A (docs/natural_waves/m4_2e_natural_waves_report.md), opt-in: "scripted" = the historical wave rule (Slow Push parking
    # and creeping, Freeze holding both fronts, fronts waiting idle behind a tower under siege); "natural" = every
    # minion packet follows the same physical rules whatever the actions (walk, engage on contact, merge, crash), the
    # defender's front engages a crashed mass under its tower instead of waiting, a crashed mass engaged by defenders
    # does not hit the tower, and the three wave slots (SP, SH, F) become the champion's damage on enemy minions:
    # Hold, Clear and Thin, champion_wave_damage[slot] HP per tick on the nearest reachable enemy minions
    wave_dynamics: str = "scripted"
    champion_wave_damage: Tuple[int, int, int] = (0, 2, 1)   # (Hold, Clear, Thin) = slots (SP, SH, F); "natural" only
    # M4.2e-A.1 (docs/natural_waves/m4_2e_natural_waves_final_review.md), "natural" only; each default reproduces M4.2e-A.
    # natural_crash_rule "contact" = a front in the enemy tower cell crashes only with no enemy front within distance 1
    # (otherwise it fights there under tower fire); "entry" = every front entering the enemy tower cell joins the crashed
    # mass (a crash is entering tower range; follow-up waves join the siege instead of fighting beside it)
    natural_crash_rule: str = "contact"
    # cells beyond the enemy tower cell a champion standing IN that cell reaches with its wave damage (0 = reach is
    # capped at the tower cell; 1 = the cell behind it, where the defender's front engages a siege, as thin_reach)
    natural_tower_reach: int = 0
    # wave_clear_scaling "none" = champion_wave_damage[slot] HP per tick; "attack_power" = a per-turn budget
    # round(ticks_per_turn * champion_wave_damage[slot] * AP / 10^4) for the slots in wave_clear_scaled_slots (AP = the
    # champion's attack power at the start of the turn), split over the ticks as evenly as possible (earlier ticks first)
    wave_clear_scaling: str = "none"
    wave_clear_scaled_slots: Tuple[int, ...] = (SH, F)      # wave slots whose budget scales (SH = Clear, F = Thin)
    # M4.2e-C (docs/natural_waves/m4_2e_wave_actions_report.md), "natural" only: wave_clear_scaling "capacity" = the three
    # wave actions are shares of ONE wave-clear capacity C = round(ticks_per_turn * champion_wave_damage[SH] * AP / 10^4)
    # HP per turn (Clear's budget, as under "attack_power"):
    #   Hold  = round(wave_hold_fraction * C)      normal last-hitting: a small share of the capacity
    #   Thin  = round(wave_thin_fraction * C)      controlled thinning: a larger share (= Hold + alpha (Clear - Hold)
    #                                              with alpha = (thin - hold) / (1 - hold), the brief's formulation)
    #   Clear = C                                   everything
    # (round half up on exact rationals; 0 <= hold <= thin <= 1). champion_wave_damage[SP], [F] and
    # wave_clear_scaled_slots are then unused.
    wave_hold_fraction: float = 0.0
    wave_thin_fraction: float = 0.0
    # M4.3c wave-control screen (docs/m4/m4_3c_wave_control_screen.md), EXPERIMENTAL, "capacity" only. wave_hold_cap
    # n > 0 caps Hold's per-turn budget at n HP whatever the capacity (0 = off). wave_thin_rule "adaptive" = Thin spends
    # min(round(wave_thin_fraction * C), need) HP per turn, need = the enemy blob's HP above the largest blob whose
    # minions do not out-fire ours (mechanics.thin_need, start-of-turn state); "share" = round(wave_thin_fraction * C)
    wave_hold_cap: int = 0
    wave_thin_rule: str = "share"
    # M4.3c four-action redesign (docs/m4/m4_3c_four_action_wave_redesign.md), EXPERIMENTAL, "capacity" only.
    # wave_thin_rule "strength" = Thin spends min(round(wave_thin_fraction * C), H + need) HP per turn, H = the Hold
    # (last-hit) budget and need = the enemy's window strength above ours (mechanics.strength_need, start-of-turn
    # state, margin 0). wave_slowpush True adds a fourth wave action, Slow Push (slot W_SLOWPUSH, codes 46..60, a
    # 61-code action space): the same rule with a margin of 1 HP (our window strength at least 1 HP above the enemy's)
    # and the same cap round(wave_thin_fraction * C); needs wave_thin_rule "strength"
    wave_slowpush: bool = False
    tower_adjacent_p: float = 0.0       # tower shot on a connected trader ending next to the enemy tower (m4: 0.5)
    tower_adjacent_p_busy: float = 0.0    # ... while its own crashed minions are under that tower (m4: 0.25)
    gank_damage_scale: str = "none"       # "none" (centi = 100 x trials), "victim_level" (m4), "attacker_level"
    plate_thresholds: Tuple[int, ...] = (1600, 3600, 6000)   # durability plate thresholds, centi-units (design §11)
    plate_gold: float = 1.0               # resources per durability plate
    pressure_tower_damage: int = 400      # centi-units per effective Apply Pressure turn at attack power 1.0
    plate_gold_requires_presence: bool = False    # design §16 A9 (alternative); False = paid while away too
    gold_unit: float = 1e-6               # resource per ledger unit (micro-resources, design §8.6)
    rear_count_max: int = 4               # largest rear-packet count in one cell (observation scale; exceeded -> error)
    # M4.2e-B (docs/natural_waves/m4_2e_zoning_report.md), opt-in: exchanges a connected Trade resolves per turn. 1 = the M2 rule (one
    # 0 / 1 / 2 base-HP roll per turn). ticks_per_turn (2) = one exchange per movement tick, consistent with every other
    # per-tick damage; the total base damage of the n exchanges is drawn from one uniform (U_TRADE_HIT) by the inverse
    # CDF of their n-fold convolution (champions_m4.trade_damage_n). Hit and double probabilities are per exchange.
    trade_exchanges: int = 1
    # M4.2e-B, opt-in: who pays trade_farm_factor in a MUTUAL connected trade. "trader" = both (the M2 rule); "loser" =
    # only the side that lost the larger fraction of its max HP to the other's trade this turn (ties: both); the
    # winner resumes farming. A unilateral connected trade still costs the trader alone, as before.
    trade_farm_rule: str = "trader"
    # M4.2e-B, opt-in (item_model "continuous" only): invested gold also buys durability: every centi-HP of damage a
    # champion takes (trades, tower shots, ganks) is divided by its own investment multiplier M(S), exactly.
    invest_durability: bool = False
    # M4.3c stabilization (docs/m4/m4_3c_strategic_economics_calibration.md), EXPERIMENTAL, ruleset "m4" only; every
    # default reproduces every earlier preset exactly.
    # xp_reward_weight lambda > 0: the reward also counts the difference of the XP the two champions are CREDITED this
    # turn (after the level cap), r_A = (E_A - E_B) + lambda (X_A - X_B), lambda in resources per XP and an exact
    # multiple of gold_unit. XP stays XP: nothing is converted into gold, the gold ledger and investment are unchanged.
    xp_reward_weight: float = 0.0
    # death_turns_by_level: () = death_turns at every level; else level_cap entries: a champion that dies at level L
    # (its level at the start of the turn of the death) sits out death_turns_by_level[L - 1] turns
    death_turns_by_level: Tuple[int, ...] = ()
    # terminal_rule "none" = continuing time without a terminal state; "final_plate" = the game TERMINATES at the end of
    # the turn in which either tower's cumulative damage reaches its last plate threshold (that turn is paid in full,
    # plate gold included; nothing follows, the terminal value is 0)
    terminal_rule: str = "none"
    # M4.5a unified positional mechanics (docs/m4/m4_5a_unified_mechanics_spec.md), EXPERIMENTAL, ruleset "m4" with
    # natural waves only; every default reproduces every earlier preset exactly (lanegame.positional).
    # vision_rule "siege" = Establish Vision is a siege objective at the enemy tower (M2-M4.4); "ward" = one shared
    # off-lane Vision node W attached to the lane cell 0, reached by the Ward action (code CHAMP_WARD = 30) from own
    # cells -1, 0, +1 or W; the ward is placed at the end of the turn unless the enemy champion's connected Trade damages
    # the warder (or it dies); a warder deals no wave damage and collects no gold; W carries the gank exposure of the
    # own cell ward_exposure; contact, movement, XP and gold distances follow the movement graph
    vision_rule: str = "siege"
    ward_exposure: int = 0
    # minion targeting (per tick): a wave designates the enemy champion on tick t by retaliation (the enemy's connected
    # Trade dealt positive damage to our champion this turn and t >= minion_retaliation_delay; the exchange lands on
    # tick 0) or by positional aggro (the enemy champion did not move this turn and, at the start of tick t, stands in
    # our front's cell with its own front strictly behind it or absent and no crashed mass of its own there); a
    # designated wave whose front is within
    # minion_range (graph distance) of the champion fires its tick firepower at it and deals no minion damage that tick
    minion_retaliation: bool = False
    minion_positional_aggro: bool = False
    minion_range: int = 1
    minion_retaliation_delay: int = 1
    # m minion damage units taken in a turn = max(1, round(alpha m / growth^(L - 1))) centi-HP (L = the victim's
    # start-of-turn level), then divided by M(S) with the turn's other damage when invest_durability is on
    minion_damage_alpha: int = 50
    minion_damage_growth: float = 1.08
    # trade_wave_rule "none" = a connected trader deals no wave damage (historical); "capped" = it keeps its farm-mode
    # reach and spends min(B_w, round(trade_wave_fraction C)) (Last Hit / Thin / Slow Push unchanged, Shove halved)
    trade_wave_rule: str = "none"
    trade_wave_fraction: float = 0.5
    # wave_reach_rule "line" = a champion damages enemy minions at every cell at or behind its reach line (historical);
    # "ball" = only within farm_reach (graph distance) of its end cell, Safe / retreat also not ahead of min(end, -1)
    wave_reach_rule: str = "line"

    # ------------------------------------------------------------------ presets
    @classmethod
    def v1(cls, **kw) -> "GameConfig":
        """The original Phase 1 model (docs/phase1/phase1_implementation_report.md), exactly."""
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
        """The previous revision (docs/phase1/phase1_revision_report.md): post-crash resource/vision choice,
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

    @classmethod
    def champions_continuing(cls, **kw) -> "GameConfig":
        """TEST FIXTURE ("m2c"): the M2 champion game with M4 time semantics (continuing, no countdown, gamma 0.99).
        Its per-turn dynamics equal champions() (plan WP1, T1.10). Never a research ruleset."""
        base = dict(champion_mode=True, ruleset="m2c", time_mode="continuing", time_features=False, gamma=0.99)
        base.update(kw)
        return cls(**base)

    @classmethod
    def m4(cls, **kw) -> "GameConfig":
        """The M4 ruleset (docs/m4/m4_integrated_design.md): continuing time, Freeze maintenance, XP and levels 1-18,
        centi-HP, attack power, one two-tier item track bought at base, durability plates, the contact-cell chase
        and the tower-adjacent shot. Played through lanegame.env (lanegame.champions_m4)."""
        base = dict(champion_mode=True, ruleset="m4", time_mode="continuing", time_features=False, gamma=0.99,
                    xp_per_hp=2, item_tiers=2, freeze_maintenance=True, chase_rule="contact_cell",
                    tower_adjacent_p=0.5, tower_adjacent_p_busy=0.25, gank_damage_scale="victim_level",
                    plate_mode="durability")
        base.update(kw)
        return cls(**base)

    @classmethod
    def m4_continuous(cls, **kw) -> "GameConfig":
        """EXPERIMENT (M4.2c): `m4()` with the two-tier item track replaced by continuous gold investment
        (item_model = "continuous"). Not a default; the validated two-tier `m4()` is unchanged."""
        base = dict(item_model="continuous", item_tiers=0, item_prices=())
        base.update(kw)
        return cls.m4(**base)

    @classmethod
    def m4_emergent(cls, **kw) -> "GameConfig":
        """EXPERIMENT (M4.2d integration, docs/m4/m4_emergent_parallel_report.md): `m4_continuous()` with the selected
        opt-in M4.2d mechanics: the bounded saturating investment curve and the compressed investment observation
        (M4.2d-D), Yield as a stand-off position that a Trade starting in contact pins (M4.2d-B), no automatic
        proximity zoning (M4.2d-C) and the pre-crash farm spot (M4.2d-A readiness). Waves keep the scripted M4 rule
        (the natural-wave prototype, branch prototype/m4-emergent-wave, is not integrated). Not a default."""
        base = dict(invest_curve="saturating", invest_gain=1.5, invest_scale=16.0, invest_obs_encoding="compressed",
                    invest_obs_ref=64, invest_obs_gold_ref=8, position_rule="standoff", zoning_rule="none",
                    farm_spot_rule="pre_crash_front")
        base.update(kw)
        return cls.m4_continuous(**base)

    @classmethod
    def m4_natural_prototype(cls, **kw) -> "GameConfig":
        """EXPERIMENT (M4.2e-A, docs/natural_waves/m4_2e_natural_waves_report.md), kept for comparison: `m4_emergent()` with natural
        wave mechanics (wave_dynamics = "natural"; wave slots Hold / Clear / Thin) on the rescaled wave of
        NATURAL_WAVE_HP = 8 HP. Every per-HP quantity is rescaled so that one fresh wave is worth what it was at 4 HP:
        farm_per_hp 0.125 (1 resource per wave), xp_per_hp 1 (8 XP per wave), siege_cap 96 (12 waves); gold_unit 1e-7
        keeps every per-HP farm value an exact ledger multiple. hp_cap (3 waves) and the Lanchester law floor(h / W)
        scale with wave_hp. Crashes need no enemy front within distance 1, the tower deals 1 HP per tick, reach ends at
        the enemy tower cell and the champion's wave damage is fixed. Not a default."""
        base = dict(wave_dynamics="natural", wave_hp=NATURAL_WAVE_HP, farm_per_hp=0.125, xp_per_hp=1,
                    gold_unit=1e-7, siege_cap=96, freeze_maintenance=False)
        base.update(kw)
        return cls.m4_emergent(**base)

    @classmethod
    def m4_natural(cls, **kw) -> "GameConfig":
        """EXPERIMENT (M4.2e-A.1, docs/natural_waves/m4_2e_natural_waves_final_review.md): the natural-wave preset. It is
        `m4_natural_prototype()` (M4.2e-A) with four changes:
          natural_crash_rule "entry"         a front entering the enemy tower cell joins the crashed mass: one siege per
                                             tower, follow-up waves join it, the tower fires on it alone
          tower_siege_damage 3               the tower alone out-kills the reinforcement stream (3 x 4 ticks > 8 HP per
                                             spawn interval), so no siege sustains itself on follow-up waves
          natural_tower_reach 1              a champion in the enemy tower cell reaches the defenders behind it
          wave_clear_scaling "attack_power"  Clear's budget scales with attack power (Thin stays a fixed, controlled
                                             1 HP per tick; Hold 0)
        Not a default."""
        base = dict(natural_crash_rule="entry", tower_siege_damage=3, natural_tower_reach=1,
                    wave_clear_scaling="attack_power", wave_clear_scaled_slots=(SH,))
        base.update(kw)
        return cls.m4_natural_prototype(**base)

    # the fields GameConfig.m4_lane() changes from m4_natural() (M4.2e-B, docs/natural_waves/m4_2e_zoning_report.md)
    M4_LANE_FIELDS = ("collect_contest_enemy_tower", "collect_contest_own_tower", "invest_durability")
    # the fields GameConfig.m4_lasthit() changes from m4_lane() (M4.2e-C, docs/natural_waves/m4_2e_wave_actions_report.md)
    M4_LASTHIT_FIELDS = ("wave_clear_scaling", "wave_hold_fraction", "wave_thin_fraction", "damage_law", "thin_damage")

    @classmethod
    def m4_lane(cls, **kw) -> "GameConfig":
        """CANDIDATE (M4.2e-B, docs/natural_waves/m4_2e_zoning_report.md): `m4_natural()` with three changes, for design review
        before the PPO smoke test. Two replace fixed collection-efficiency proxies by consequences:
          collect_contest_enemy_tower 0.9   the 0.5 rate under the enemy tower stood in for danger; the danger is
                                            now explicit (tower shots, ganks, the defender's home advantage)
          collect_contest_own_tower 0.6     kills in the own tower cell (a crashed mass dying to the tower and the
                                            defenders) are last-hit at 0.6: the tower competes for last hits, so a
                                            crash is no longer a free gift to the defender and pushing when ahead pays
        and one makes a power lead decisive instead of a symmetric war of attrition:
          invest_durability True            invested gold buys durability as well as attack power (damage taken is
                                            divided by the victim's investment multiplier)
        trade_exchanges (one exchange per tick) and trade_farm_rule "loser" are implemented and tested but left off:
        the first makes the equal lane a trade-fest, the second adds nothing measurable without the first.
        Not a default; every other preset is unchanged."""
        base = dict(collect_contest_enemy_tower=0.9, collect_contest_own_tower=0.6, invest_durability=True)
        base.update(kw)
        return cls.m4_natural(**base)

    @classmethod
    def m4_lasthit(cls, **kw) -> "GameConfig":
        """CANDIDATE (M4.2e-C, docs/natural_waves/m4_2e_wave_actions_report.md): `m4_lane()` with the wave actions
        redesigned as shares of one wave-clear capacity C = round(4 AP) HP per turn (wave_clear_scaling "capacity"):
          Hold  = round(C / 8)      normal last-hitting pushes a little (1 HP per turn up to AP 2.9, 2 up to AP 4.4, 3 after)
          Thin  = round(C / 2)      controlled thinning grows with power (2 at AP 1, 4 at AP 2, 9 at AP 4.2)
          Clear = C                 as before (4 at AP 1, 8 at AP 2, 17 at AP 4.2)
        the minion damage law at half-wave resolution (damage_law "lanchester_half": a packet's firepower is
        floor(2h / W) HP per turn instead of floor(h / W) per tick), so that an edge of half a wave already fights
        harder, and no separate Apply Pressure thinning (thin_damage 0: the champion's wave action is its damage on the
        defenders behind the tower, which natural_tower_reach already lets it reach). Every other rule and parameter
        is `m4_lane()`. Not a default; every other preset is unchanged."""
        base = dict(wave_clear_scaling="capacity", wave_hold_fraction=0.125, wave_thin_fraction=0.5,
                    damage_law="lanchester_half", thin_damage=0)
        base.update(kw)
        return cls.m4_lane(**base)

    # the fields GameConfig.m4_four_actions() changes from m4_lasthit() (M4.3c, docs/m4/m4_3c_four_action_wave_redesign.md)
    M4_FOUR_ACTIONS_FIELDS = ("wave_hold_cap", "wave_thin_rule", "wave_slowpush")

    @classmethod
    def m4_four_actions(cls, **kw) -> "GameConfig":
        """EXPERIMENTAL (M4.3c, docs/m4/m4_3c_four_action_wave_redesign.md): `m4_lasthit()` with four wave actions,
        each a per-turn damage budget on enemy minions out of the same wave-clear capacity C = round(4 AP):
          Last Hit  = min(1, round(C / 8)) = 1                    wave_hold_cap 1: farming never pushes by power
          Thin      = min(round(C / 2), 1 + need(margin 0))       wave_thin_rule "strength": remove only the enemy's
                                                                  excess window strength over ours
          Slow Push = min(round(C / 2), 1 + need(margin 1))       wave_slowpush: the least damage that puts our
                                                                  window strength 1 HP above the enemy's
          Shove     = C                                           slot SH (W_SHOVE = W_CLEAR), unchanged; called
                                                                  Clear in the redesign report and in m4_lasthit()
        need = mechanics.strength_need on the start-of-turn lane. The action space has 61 codes (Slow Push = codes
        46..60, every historical code unchanged). Every other rule and parameter is `m4_lasthit()`. Not a default;
        `m4_lasthit()` and every other preset are unchanged."""
        base = dict(wave_hold_cap=1, wave_thin_rule="strength", wave_slowpush=True)
        base.update(kw)
        return cls.m4_lasthit(**base)

    # the fields GameConfig.m4_four_actions_econ() changes from m4_four_actions() (M4.3c stabilization,
    # docs/m4/m4_3c_strategic_economics_calibration.md)
    M4_FOUR_ECON_FIELDS = ("collect_contest_own_tower", "trade_farm_factor", "kill_bounty", "death_turns_by_level",
                           "champion_absent_max")
    # 2 + ceil(L / 6): 3 turns at levels 1-6, 4 at 7-12, 5 at 13-18 (League's respawn timer plus the walk back)
    DEATH_TURNS_S1 = tuple(2 + -(-L // 6) for L in range(1, 19))

    @classmethod
    def m4_four_actions_econ(cls, **kw) -> "GameConfig":
        """EXPERIMENTAL CANDIDATE (M4.3c stabilization, docs/m4/m4_3c_strategic_economics_calibration.md):
        `m4_four_actions()` with a calibrated lane economy. Four fixed rates are set to League-proportional values:
          collect_contest_own_tower 0.8      a crashed wave dying under the own tower is last-hit at 0.8 (0.6 before):
                                             crashing taxes the defender 1/9 of the lane rate instead of 1/3, which
                                             alone made Shove the best reply in every state
          trade_farm_factor 0.8              a connected trader collects 80% of its farm (50% before): with 50%
                                             every harassment cost the attacker as much gold as it cost the victim,
                                             so power could buy lane control only through Shove
          kill_bounty 2.0                    a kill pays two waves (0.5 before; League's kill is about 2.4 waves)
          death_turns_by_level S1            a death keeps the champion away 3 / 4 / 5 turns at levels 1-6 / 7-12 /
                                             13-18 (3 at every level before; champion_absent_max 5, the observation's
                                             absence scale)
        The wave mechanics, the four actions, the reward (gold only), XP, towers, plates and continuing time are
        `m4_four_actions()`'s. The action and observation spaces are unchanged (61 codes, 88 entries); only the
        absence entry's scale is 5 instead of 3, so networks trained on another preset must be retrained. Not a
        default; every other preset is unchanged."""
        base = dict(collect_contest_own_tower=0.8, trade_farm_factor=0.8, kill_bounty=2.0,
                    death_turns_by_level=cls.DEATH_TURNS_S1, champion_absent_max=5)
        base.update(kw)
        return cls.m4_four_actions(**base)

    # the fields GameConfig.m4_four_actions_econ_aggro() changes from m4_four_actions_econ() (M4.3d,
    # docs/m4/m4_3d_tower_pressure_calibration.md)
    M4_FOUR_AGGRO_FIELDS = ("tower_shot_p_busy", "tower_adjacent_p_busy")

    @classmethod
    def m4_four_actions_econ_aggro(cls, **kw) -> "GameConfig":
        """EXPERIMENTAL CANDIDATE (M4.3d, docs/m4/m4_3d_tower_pressure_calibration.md): `m4_four_actions_econ()` with
        the tower's "busy" discount removed for a champion that attacks a champion under it:
          tower_shot_p_busy 1.0        a connected trader ending in the enemy tower cell (a dive) is shot with
                                       probability 1.0 also while its own crashed minions are under that tower (0.5
                                       before; League's tower switches to a champion that attacks an allied champion)
          tower_adjacent_p_busy 0.5    the tower-adjacent shot of a connected trader keeps its 0.5 likewise (0.25)
        A dive is only possible at the enemy tower cell and, in play, almost only during the diver's own siege, so the
        discount applied to nearly every dive; a strong champion's dive paid more than pressuring without one. Only
        connected trades in or next to the enemy tower change: farming, Pressure, Vision, ganks, the own-tower
        collection rate 0.8 and every other rule are `m4_four_actions_econ()`'s, and so are the action and
        observation spaces. Not a default; every other preset is unchanged."""
        base = dict(tower_shot_p_busy=1.0, tower_adjacent_p_busy=0.5)
        base.update(kw)
        return cls.m4_four_actions_econ(**base)

    # the fields GameConfig.m45_unified() changes from m4_four_actions_econ_aggro() (M4.5a)
    M45_UNIFIED_FIELDS = ("vision_rule", "xp_range", "minion_retaliation", "minion_positional_aggro",
                          "trade_wave_rule", "wave_reach_rule")

    @classmethod
    def m45_unified(cls, **kw) -> "GameConfig":
        """EXPERIMENTAL CANDIDATE (M4.5a, docs/m4/m4_5a_unified_mechanics_spec.md): `m4_four_actions_econ_aggro()` with
        the unified positional mechanics, implemented together because they interact:
          vision_rule "ward"              one shared off-lane Vision node W next to the lane cell 0; Ward (code 30)
                                          walks there and places a ward unless the enemy's Trade damages the warder;
                                          the siege Vision objective is gone
          xp_range 2                      XP within 2 cells (movement-graph distance) instead of 3
          minion_retaliation True         a wave turns on an enemy champion whose Trade damaged our champion (from the
                                          tick after the exchange)
          minion_positional_aggro True    a wave turns on an enemy champion standing in its cell ahead of its own wave
          trade_wave_rule "capped"        a connected trader still spends up to round(C / 2) on enemy minions
          wave_reach_rule "ball"          wave damage only within one cell of the champion
        with the provisional parameters minion_range 1, minion_retaliation_delay 1, minion_damage_alpha 50,
        minion_damage_growth 1.08, trade_wave_fraction 0.5, ward_exposure 0 and the unchanged vision numbers (6 turns,
        gank occurrence x 0.25). Every other rule is `m4_four_actions_econ_aggro()`'s. The action space keeps 61 codes;
        the observation has 90 entries (a W entry per champion cell one-hot). Not a default; every other preset is
        unchanged. M4.5b calibrates it."""
        base = dict(vision_rule="ward", xp_range=2, minion_retaliation=True, minion_positional_aggro=True,
                    trade_wave_rule="capped", wave_reach_rule="ball")
        base.update(kw)
        return cls.m4_four_actions_econ_aggro(**base)

    # ------------------------------------------------------------------ helpers
    @property
    def ward_node(self) -> bool:
        """M4.5a: the shared Vision node W and the Ward action (vision_rule "ward")."""
        return self.vision_rule == "ward"

    @property
    def minion_targeting(self) -> bool:
        """M4.5a: waves can target champions (retaliation or positional aggro)."""
        return self.minion_retaliation or self.minion_positional_aggro

    @property
    def m45_mechanics(self) -> bool:
        """Any M4.5a opt-in rule is on (the environment then reports the M4.5a turn events)."""
        return (self.ward_node or self.minion_targeting or self.trade_wave_rule != "none"
                or self.wave_reach_rule != "line")

    @property
    def natural_waves(self) -> bool:
        """M4.2e-A natural wave mechanics: wave slots are Hold / Clear / Thin champion damage."""
        return self.wave_dynamics == "natural"

    @property
    def continuous_items(self) -> bool:
        """M4.2c continuous gold investment: M4Side.items holds the cumulative invested gold in ledger units."""
        return self.item_model == "continuous"

    @property
    def continuing(self) -> bool:
        """Continuing time (rulesets "m2c" and "m4"): no terminal state, no countdown, discounted objective."""
        return self.time_mode == "continuing"

    @property
    def is_m4(self) -> bool:
        return self.ruleset == "m4"

    @property
    def gold_scale(self) -> int:
        """Ledger units per resource (10^6 for micro-resources)."""
        return int(round(1.0 / self.gold_unit))
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

        Natural waves have no creep clock, so their true period is spawn_every_turns; they keep the scripted value
        (a multiple of it) so that the observation's phase one-hot and the 88-entry layout do not change.
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
        """Every action slot of this configuration's fixed action space (champion mode: the full 46-slot space, 61
        slots under wave_slowpush; of its Recall codes only champ_act(SP, B_RECALL) is ever legal)."""
        if self.champion_mode:
            return tuple(range(N_CHAMP_ACTIONS_FOUR if self.wave_slowpush else N_CHAMP_ACTIONS))
        if not self.positioning:
            return (SP, SH, F)
        if self.objectives:
            return tuple(range(N_ACTIONS))
        return tuple(range(N_BASE_ACTIONS))

    def label(self, a: int) -> str:
        """Short action label: 'SH-C', 'SH-C+P' etc., or 'SH' when positioning is disabled; champion mode
        'SH-T+P' (Shove + Trade + Pressure) etc.; natural waves 'CL-T+P' (Clear + Trade + Pressure) etc.; the
        four-action model (wave_slowpush) 'LH-H', 'TN-H', 'SL-H+P', 'SH-T' (Last Hit / Thin / Slow Push / Shove)."""
        if self.champion_mode:
            if self.vision_rule == "ward" and a == CHAMP_WARD:
                return "WARD"
            if self.natural_waves and a != CHAMP_ABSENT:
                o = champ_objective(a)
                return ((FOUR_WAVE_SHORT if self.wave_slowpush else NATURAL_WAVE_SHORT)[champ_wave(a)] + "-"
                        + BEHAVIOR_SHORT[champ_behavior(a)] + (f"+{OBJECTIVE_SHORT[o]}" if o else ""))
            return CHAMP_ACTION_SHORT[a]
        return ACTION_SHORT[a] if self.positioning else WAVE_SHORT[wave_of(a)]

    def validate(self) -> None:
        self._validate_ruleset()
        assert self.base > self.tower > 0
        assert self.spawn_interval % self.ticks_per_turn == 0
        assert self.damage_law in ("lanchester", "lanchester_ceil", "lanchester_half", "linear", "pressure")
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
        assert self.plate_mode in ("crash", "tower", "durability")
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
            assert self.champion_absent_max >= max(self.recall_turns - 1, self.death_turns)
            assert self.death_turns >= 0 and self.kill_bounty >= 0 and self.gank_damage_trials >= 0
            assert 0.0 <= self.champ_gank_max <= 1.0
            assert len(self.trade_hit_behavior) == N_BEHAVIOR_SLOTS
            assert 0.0 <= self.trade_hit_min <= self.trade_hit_max <= 1.0
            assert self.tower_shot_damage >= 0 and 0.0 <= self.tower_shot_p <= 1.0 and 0.0 <= self.tower_shot_p_busy <= 1.0
            assert 0.0 <= self.skill_a <= 1.0 and 0.0 <= self.skill_b <= 1.0
            assert -self.tower <= self.safe_line <= self.tower
            assert self.trade_range >= 0 and self.farm_reach >= 0 and 0.0 <= self.trade_farm_factor <= 1.0
            assert self.retreat_farm_reach >= 0
            assert len(self.champ_gank_behavior) == len(CHAMP_BEHAVIORS)

    # ------------------------------------------------------------------ ruleset contract (plan WP0)
    def _validate_ruleset(self) -> None:
        """The contract table of plan WP0. Raises ValueError naming the offending field."""
        def bad(field: str, why: str):
            raise ValueError(f"ruleset {self.ruleset!r}: {field} {why}")

        if self.ruleset not in RULESETS:
            bad("ruleset", f"must be one of {RULESETS}")
        if self.damage_law == "lanchester_half" and self.wave_dynamics != "natural":
            bad("damage_law", "'lanchester_half' applies to wave_dynamics 'natural' only")
        if self.time_mode not in ("finite", "continuing"):
            bad("time_mode", "must be 'finite' or 'continuing'")
        if self.ruleset == "legacy":
            if self.time_mode != "finite":
                bad("time_mode", "must be 'finite' (legacy presets are pinned)")
            if self.time_features is not True:
                bad("time_features", "must be True (legacy presets are pinned)")
            if self.gamma != 1.0:
                bad("gamma", "must be 1.0 (legacy presets are pinned)")
        else:
            if self.time_mode != "continuing":
                bad("time_mode", "must be 'continuing'")
            if self.time_features is not False:
                bad("time_features", "must be False (no countdown under continuing time)")
            if not 0.0 < self.gamma < 1.0:
                bad("gamma", "must be in (0, 1) under continuing time")
            if not self.champion_mode:
                bad("champion_mode", "must be True")
        if self.ruleset in ("legacy", "m2c"):
            for f in M4_ECONOMY_FIELDS:
                if getattr(self, f) != _FIELD_DEFAULTS[f]:
                    bad(f, f"is an M4 economy field and must stay at its off default {_FIELD_DEFAULTS[f]!r}")
            if self.plate_mode == "durability":
                bad("plate_mode", "'durability' is M4 only")
            return
        # ---- ruleset "m4"
        if not self.positioning:
            bad("positioning", "must be True")
        if self.plate_mode != "durability":
            bad("plate_mode", "must be 'durability' (crash-plate opportunities are off under M4, design §11.4)")
        if not (isinstance(self.xp_per_hp, int) and self.xp_per_hp >= 1):
            bad("xp_per_hp", "must be a positive integer (XP is an integer state)")
        if self.xp_range < 0 or self.xp_base < 1 or self.xp_step < 0 or self.level_cap < 1:
            bad("xp_range/xp_base/xp_step/level_cap", "out of range")
        if self.hp_base < 1 or self.hp_growth <= 0 or self.ap_level_growth <= 0 or self.item_mult <= 0:
            bad("hp_base/hp_growth/ap_level_growth/item_mult", "must be positive")
        if self.item_tiers < 0 or len(self.item_prices) != self.item_tiers or any(p <= 0 for p in self.item_prices):
            bad("item_prices", f"must hold item_tiers = {self.item_tiers} positive prices")
        if self.item_model not in ("tiers", "continuous"):
            bad("item_model", "must be 'tiers' or 'continuous'")
        if self.item_model == "continuous":
            if self.item_tiers != 0:
                bad("item_tiers", "must be 0 under item_model 'continuous' (no tiers)")
            if not (self.invest_alpha > 0 and 0 < self.invest_p <= 1):
                bad("invest_alpha/invest_p", "need invest_alpha > 0 and 0 < invest_p <= 1")
            if not (isinstance(self.invest_obs_gold_ref, int) and isinstance(self.invest_obs_ref, int)
                    and 1 <= self.invest_obs_gold_ref <= 2 ** 24 and 1 <= self.invest_obs_ref <= 2 ** 24):
                bad("invest_obs_gold_ref/invest_obs_ref", "must be integers in [1, 2^24] (resources)")
            if self.invest_curve not in ("power", "saturating"):
                bad("invest_curve", "must be 'power' or 'saturating'")
            if self.invest_obs_encoding not in ("linear", "compressed"):
                bad("invest_obs_encoding", "must be 'linear' or 'compressed'")
            mine = ("invest_alpha", "invest_p") if self.invest_curve == "power" else ("invest_gain", "invest_scale")
            for f in ("invest_alpha", "invest_p", "invest_gain", "invest_scale"):
                if f not in mine and getattr(self, f) != _FIELD_DEFAULTS[f]:
                    bad(f, f"does not apply to invest_curve {self.invest_curve!r}")
            if self.invest_curve == "saturating" and not (0.01 <= self.invest_gain <= 10.0
                                                           and 1.0 <= self.invest_scale <= 10.0 ** 6):
                bad("invest_gain/invest_scale", "need 0.01 <= invest_gain <= 10 and 1 <= invest_scale <= 10^6 "
                                                "resources (the range whose strict monotonicity is verified)")
        else:
            for f in ("invest_alpha", "invest_p", "invest_obs_gold_ref", "invest_obs_ref", "invest_curve",
                      "invest_gain", "invest_scale", "invest_obs_encoding"):
                if getattr(self, f) != _FIELD_DEFAULTS[f]:
                    bad(f, "applies to item_model 'continuous' only")
        if self.chase_rule not in ("m2", "contact_cell"):
            bad("chase_rule", "must be 'm2' or 'contact_cell'")
        if self.farm_spot_rule not in ("m2", "pre_crash_front"):
            bad("farm_spot_rule", "must be 'm2' or 'pre_crash_front'")
        if self.position_rule not in ("m2", "pin", "standoff"):
            bad("position_rule", "must be 'm2', 'pin' or 'standoff'")
        if self.gank_damage_scale not in ("none", "victim_level", "attacker_level"):
            bad("gank_damage_scale", "must be 'none', 'victim_level' or 'attacker_level'")
        if not (0.0 <= self.tower_adjacent_p <= 1.0 and 0.0 <= self.tower_adjacent_p_busy <= 1.0):
            bad("tower_adjacent_p", "probabilities must be in [0, 1]")
        th = self.plate_thresholds
        if not th or any(not isinstance(x, int) for x in th) or th[0] <= 0 or any(b <= a for a, b in zip(th, th[1:])):
            bad("plate_thresholds", "must be strictly increasing positive integers (centi-units)")
        if not (isinstance(self.pressure_tower_damage, int) and self.pressure_tower_damage >= 0):
            bad("pressure_tower_damage", "must be a non-negative integer (centi-units)")
        if self.rear_count_max < 1:
            bad("rear_count_max", "must be >= 1")
        if not (isinstance(self.trade_exchanges, int) and 1 <= self.trade_exchanges <= self.ticks_per_turn):
            bad("trade_exchanges", f"must be an integer in [1, ticks_per_turn = {self.ticks_per_turn}]")
        if self.trade_farm_rule not in ("trader", "loser"):
            bad("trade_farm_rule", "must be 'trader' or 'loser'")
        if self.invest_durability and self.item_model != "continuous":
            bad("invest_durability", "applies to item_model 'continuous' only")
        if not self.xp_reward_weight >= 0.0:
            bad("xp_reward_weight", "must be >= 0 (resources per credited XP)")
        dtl = self.death_turns_by_level
        if dtl:
            if len(dtl) != self.level_cap or any(not isinstance(d, int) or d < 0 for d in dtl):
                bad("death_turns_by_level", f"must hold level_cap = {self.level_cap} non-negative integers (or be empty)")
            if self.champion_absent_max < max(dtl):
                bad("champion_absent_max", f"must be >= max(death_turns_by_level) = {max(dtl)} (observation scale)")
        if self.terminal_rule not in ("none", "final_plate"):
            bad("terminal_rule", "must be 'none' or 'final_plate'")
        self._validate_m45()
        if not 0.0 <= self.collect_contest_own_tower <= 1.0:
            bad("collect_contest_own_tower", "must be in [0, 1]")
        if self.zoning_rule not in ("presence", "contact", "none"):
            bad("zoning_rule", "must be 'presence', 'contact' or 'none'")
        if self.siege_engage:
            bad("siege_engage", "must be False (the evaluation bound counts no siege fights, design §10.5.3)")
        if self.wave_dynamics not in ("scripted", "natural"):
            bad("wave_dynamics", "must be 'scripted' or 'natural'")
        if self.wave_dynamics == "natural":
            cwd = self.champion_wave_damage
            if len(cwd) != 3 or any(not isinstance(d, int) or d < 0 for d in cwd):
                bad("champion_wave_damage", "must hold three non-negative integers (HP per tick of Hold, Clear, Thin)")
            if self.freeze_maintenance:
                bad("freeze_maintenance", "must be False under wave_dynamics 'natural' (Thin is not a Freeze)")
            if self.cap_rule != "block" or self.tower_mode != "mass" or self.damage_law == "pressure":
                bad("wave_dynamics", "'natural' needs cap_rule 'block', tower_mode 'mass' and a Lanchester damage law")
            if self.natural_crash_rule not in ("contact", "entry"):
                bad("natural_crash_rule", "must be 'contact' or 'entry'")
            if self.natural_tower_reach not in (0, 1):
                bad("natural_tower_reach", "must be 0 or 1")
            if self.wave_clear_scaling not in ("none", "attack_power", "capacity"):
                bad("wave_clear_scaling", "must be 'none', 'attack_power' or 'capacity'")
            if not set(self.wave_clear_scaled_slots) <= {SP, SH, F}:
                bad("wave_clear_scaled_slots", "must be wave slots")
            if not (0.0 <= self.wave_hold_fraction <= self.wave_thin_fraction <= 1.0):
                bad("wave_hold_fraction/wave_thin_fraction", "need 0 <= hold <= thin <= 1")
            if not (isinstance(self.wave_hold_cap, int) and self.wave_hold_cap >= 0):
                bad("wave_hold_cap", "must be a non-negative integer (0 = off)")
            if self.wave_thin_rule not in ("share", "adaptive", "strength"):
                bad("wave_thin_rule", "must be 'share', 'adaptive' or 'strength'")
            if self.wave_thin_rule == "adaptive" and self.damage_law != "lanchester_half":
                bad("wave_thin_rule", "'adaptive' needs damage_law 'lanchester_half' (its target is that law's firepower)")
            if not isinstance(self.wave_slowpush, bool):
                bad("wave_slowpush", "must be a bool")
            if self.wave_slowpush and self.wave_thin_rule != "strength":
                bad("wave_slowpush", "needs wave_thin_rule 'strength' (Slow Push and Thin are one target rule)")
            if self.wave_clear_scaling != "capacity":
                for name in ("wave_hold_fraction", "wave_thin_fraction", "wave_hold_cap", "wave_thin_rule",
                             "wave_slowpush"):
                    if getattr(self, name) != _FIELD_DEFAULTS[name]:
                        bad(name, "applies to wave_clear_scaling 'capacity' only")
        else:
            for name in ("champion_wave_damage", "natural_crash_rule", "natural_tower_reach", "wave_clear_scaling",
                         "wave_clear_scaled_slots", "wave_hold_fraction", "wave_thin_fraction", "wave_hold_cap",
                         "wave_thin_rule", "wave_slowpush"):
                if getattr(self, name) != _FIELD_DEFAULTS[name]:
                    bad(name, "applies to wave_dynamics 'natural' only")
        self._validate_exact_economy()

    def _validate_m45(self) -> None:
        """M4.5a opt-in fields (ruleset "m4"; the legacy rulesets pin them through M4_ECONOMY_FIELDS)."""
        def bad(field: str, why: str):
            raise ValueError(f"ruleset {self.ruleset!r}: {field} {why}")

        if self.vision_rule not in ("siege", "ward"):
            bad("vision_rule", "must be 'siege' or 'ward'")
        if self.trade_wave_rule not in ("none", "capped"):
            bad("trade_wave_rule", "must be 'none' or 'capped'")
        if self.wave_reach_rule not in ("line", "ball"):
            bad("wave_reach_rule", "must be 'line' or 'ball'")
        for f in ("minion_retaliation", "minion_positional_aggro"):
            if not isinstance(getattr(self, f), bool):
                bad(f, "must be a bool")
        if self.m45_mechanics and self.wave_dynamics != "natural":
            bad("vision_rule/minion_*/trade_wave_rule/wave_reach_rule", "the M4.5a rules need wave_dynamics 'natural'")
        if self.vision_rule == "ward":
            if not -self.tower <= self.ward_exposure <= self.tower:
                bad("ward_exposure", f"must be a lane cell in [-{self.tower}, {self.tower}]")
        elif self.ward_exposure != _FIELD_DEFAULTS["ward_exposure"]:
            bad("ward_exposure", "applies to vision_rule 'ward' only")
        if self.minion_targeting:
            if not (isinstance(self.minion_range, int) and self.minion_range >= 0):
                bad("minion_range", "must be a non-negative integer (graph distance)")
            if not (isinstance(self.minion_retaliation_delay, int)
                    and 0 <= self.minion_retaliation_delay <= self.ticks_per_turn):
                bad("minion_retaliation_delay", f"must be an integer in [0, ticks_per_turn = {self.ticks_per_turn}]")
            if not (isinstance(self.minion_damage_alpha, int) and self.minion_damage_alpha >= 0):
                bad("minion_damage_alpha", "must be a non-negative integer (centi-HP per minion damage unit)")
            if not self.minion_damage_growth >= 1.0:
                bad("minion_damage_growth", "must be >= 1 (minion damage never grows with the victim's level)")
            if self.minion_retaliation and self.trade_exchanges != 1:
                bad("minion_retaliation", "needs trade_exchanges 1 (the single exchange lands on tick 0)")
        else:
            for f in ("minion_range", "minion_retaliation_delay", "minion_damage_alpha", "minion_damage_growth"):
                if getattr(self, f) != _FIELD_DEFAULTS[f]:
                    bad(f, "applies to minion_retaliation / minion_positional_aggro only")
        if self.trade_wave_rule == "capped":
            if self.wave_clear_scaling != "capacity":
                bad("trade_wave_rule", "'capped' needs wave_clear_scaling 'capacity' (the cap is a share of C)")
            if not 0.0 <= self.trade_wave_fraction <= 1.0:
                bad("trade_wave_fraction", "must be in [0, 1]")
        elif self.trade_wave_fraction != _FIELD_DEFAULTS["trade_wave_fraction"]:
            bad("trade_wave_fraction", "applies to trade_wave_rule 'capped' only")

    def _validate_exact_economy(self) -> None:
        """Design §8.6: every earned amount must be an exact integer multiple of the ledger unit. Each economic
        parameter is read as an exact rational from its decimal string; a per-HP farm value, a plate value, the
        bounty or a price that is not a multiple of gold_unit is rejected."""
        from fractions import Fraction

        def q(x) -> Fraction:
            return Fraction(repr(float(x)))

        scale = Fraction(1) / q(self.gold_unit)
        if scale.denominator != 1:
            raise ValueError(f"ruleset 'm4': gold_unit {self.gold_unit!r} is not 1/integer")

        def check(name: str, value: Fraction):
            if (value * scale).denominator != 1:
                raise ValueError(f"ruleset 'm4': {name} = {float(value)!r} is not an integer multiple of "
                                 f"gold_unit {self.gold_unit!r} (design §8.6)")

        for v, name in ((self.plate_gold, "plate_gold"), (self.kill_bounty, "kill_bounty"),
                        (self.xp_reward_weight, "xp_reward_weight")):
            check(name, q(v))
        for i, p in enumerate(self.item_prices):
            check(f"item_prices[{i}]", q(p))
        for value, name in exact_farm_values(self):
            check(name, value)
        if self.item_model == "continuous" and self.invest_curve == "saturating":
            check("invest_scale", q(self.invest_scale))
        if self.item_model == "continuous" and scale > 2 ** 24:
            raise ValueError("ruleset 'm4': item_model 'continuous' encodes the sub-resource part of the ledger in "
                             "one float32 feature, which needs gold_scale <= 2^24")
        if sum(int(q(p) * scale) for p in self.item_prices) > 2 ** 24:
            raise ValueError("ruleset 'm4': the summed item prices exceed 2^24 ledger units; the float32 gold feature "
                             "would no longer decode exactly (design §12.2)")

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


# ---------------------------------------------------------------------- ruleset contract tables (plan WP0)
RULESETS = ("legacy", "m2c", "m4")
M4_ECONOMY_FIELDS = ("xp_per_hp", "xp_range", "xp_base", "xp_step", "level_cap", "hp_base", "hp_growth",
                     "ap_level_growth", "item_mult", "item_tiers", "item_prices", "item_model", "invest_alpha", "invest_p",
                     "invest_obs_gold_ref", "invest_obs_ref", "invest_curve", "invest_gain", "invest_scale",
                     "invest_obs_encoding", "freeze_maintenance", "chase_rule", "farm_spot_rule",
                     "tower_adjacent_p", "tower_adjacent_p_busy", "gank_damage_scale", "plate_thresholds",
                     "plate_gold", "pressure_tower_damage", "plate_gold_requires_presence", "gold_unit",
                     "rear_count_max", "trade_exchanges", "trade_farm_rule", "invest_durability", "collect_contest_own_tower", "position_rule", "zoning_rule", "wave_dynamics", "champion_wave_damage",
                     "natural_crash_rule", "natural_tower_reach", "wave_clear_scaling", "wave_clear_scaled_slots",
                     "wave_hold_fraction", "wave_thin_fraction", "wave_hold_cap", "wave_thin_rule", "wave_slowpush",
                     "xp_reward_weight", "death_turns_by_level", "terminal_rule",
                     # M4.5a
                     "vision_rule", "ward_exposure", "minion_retaliation", "minion_positional_aggro", "minion_range",
                     "minion_retaliation_delay", "minion_damage_alpha", "minion_damage_growth", "trade_wave_rule",
                     "trade_wave_fraction", "wave_reach_rule")
_FIELD_DEFAULTS = {f.name: f.default for f in GameConfig.__dataclass_fields__.values()}


def wave_display_names(cfg: GameConfig) -> Dict[int, str]:
    """Readable names of cfg's wave slots, from the least to the most wave damage: Last Hit / Thin / Slow Push / Shove
    (the four-action model, wave_slowpush), Hold / Thin / Clear (the natural three-action presets), Slow Push / Freeze /
    Shove (scripted waves)."""
    if cfg.natural_waves:
        return dict(FOUR_WAVE_DISPLAY if cfg.wave_slowpush else NATURAL_WAVE_DISPLAY)
    return dict(SCRIPTED_WAVE_DISPLAY)


def exact_safe_base(x: int, cfg: GameConfig):
    """Exact rational Safe-posture collection base at own cell x: clip(collect_safe - slope (min(x, T) + T), 0, 1)."""
    from fractions import Fraction
    v = Fraction(repr(float(cfg.collect_safe))) - Fraction(repr(float(cfg.collect_safe_slope))) * (min(x, cfg.tower) + cfg.tower)
    return min(Fraction(1), max(Fraction(0), v))


def exact_farm_values(cfg: GameConfig):
    """Every per-HP farm value f * base * (1 - zone) * t the champion collection rule can produce (design §8.6), as
    exact rationals with a label. Kill cells are enumerated over a superset of the reachable range."""
    from fractions import Fraction

    def q(x):
        return Fraction(repr(float(x)))

    f = q(cfg.farm_per_hp)
    bases = {"contest": q(cfg.collect_contest), "contest_enemy_tower": q(cfg.collect_contest_enemy_tower),
             "contest_own_tower": q(cfg.collect_contest_own_tower)}
    for x in range(-cfg.base, cfg.base + cfg.thin_reach + 1):
        bases[f"safe@{x}"] = exact_safe_base(x, cfg)
    zones = {"unzoned": Fraction(1), "zoned": 1 - q(cfg.zone_contest), "zoned_under_tower": 1 - q(cfg.zone_under_tower)}
    trades = {"": Fraction(1), "_trading": q(cfg.trade_farm_factor)}
    out = []
    for bn, b in bases.items():
        for zn, z in zones.items():
            for tn, t in trades.items():
                out.append((f * b * z * t, f"farm_per_hp x {bn} x {zn}{tn}"))
    return out
