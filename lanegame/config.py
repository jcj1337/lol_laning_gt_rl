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
    # M4.2d-C (docs/m4_zoning_report.md), opt-in, ruleset "m4" only: which opponent applies the zone factor
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

    # ---- ruleset contract (docs/m4_implementation_plan.md WP0). Every default reproduces the legacy game. ----
    #   "legacy"  GameConfig(), .v1(), .v2(), .champions(): finite horizon K, terminal, countdown observed, gamma = 1
    #   "m2c"     .champions_continuing(): M2 champion mechanics with M4 time semantics. A TEST FIXTURE for the
    #             continuing-time machinery only; never a research ruleset
    #   "m4"      .m4(): the integrated M4 ruleset (docs/m4_integrated_design.md), lanegame.champions_m4
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
    # M4.2c experiment (docs/m4_continuous_investment_report.md): "tiers" = the two-tier track above (the validated
    # baseline); "continuous" = every return to base invests ALL unspent gold, AP(L, S) = AP(L, 0) [1 + invest_alpha
    # ((1 + S)^invest_p - 1)] with S the cumulative investment in resources (needs item_tiers = 0).
    item_model: str = "tiers"
    invest_alpha: float = 0.045
    invest_p: float = 0.90
    invest_obs_gold_ref: int = 1024       # resources: the observation encodes unspent gold exactly below this
    invest_obs_ref: int = 1024            # resources: ... and invested gold exactly below this
    # M4.2d (docs/m4_investment_readiness_report.md), opt-in: invest_curve "power" = the M4.2c curve above;
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
    # M4.2d (docs/m4_wave_readiness_report.md), opt-in: "m2" = the farm spot is the probe's leading packet (M2 rule);
    # "pre_crash_front" = a champion whose front crashes in the probe keeps its pre-turn front cell as farm spot
    farm_spot_rule: str = "m2"
    # M4.2d-B (docs/m4_positioning_report.md), opt-in: "m2" = M2 positioning and evasion; "pin" = a Yield champion that
    # STARTS the turn in contact with an enemy Trade champion cannot evade it (it is caught like a Hold); "standoff" =
    # "pin" plus Yield farms from its stand-off cell max(-T, min(spot, e - farm_reach)), e = the enemy front at the start
    # of the turn in own coordinates: the rearmost cell still in reach of the contested minions
    position_rule: str = "m2"
    tower_adjacent_p: float = 0.0         # tower shot on a connected trader ending next to the enemy tower (m4: 0.5)
    tower_adjacent_p_busy: float = 0.0    # ... while its own crashed minions are under that tower (m4: 0.25)
    gank_damage_scale: str = "none"       # "none" (centi = 100 x trials), "victim_level" (m4), "attacker_level"
    plate_thresholds: Tuple[int, ...] = (1600, 3600, 6000)   # durability plate thresholds, centi-units (design §11)
    plate_gold: float = 1.0               # resources per durability plate
    pressure_tower_damage: int = 400      # centi-units per effective Apply Pressure turn at attack power 1.0
    plate_gold_requires_presence: bool = False    # design §16 A9 (alternative); False = paid while away too
    gold_unit: float = 1e-6               # resource per ledger unit (micro-resources, design §8.6)
    rear_count_max: int = 4               # largest rear-packet count in one cell (observation scale; exceeded -> error)

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

    @classmethod
    def champions_continuing(cls, **kw) -> "GameConfig":
        """TEST FIXTURE ("m2c"): the M2 champion game with M4 time semantics (continuing, no countdown, gamma 0.99).
        Its per-turn dynamics equal champions() (plan WP1, T1.10). Never a research ruleset."""
        base = dict(champion_mode=True, ruleset="m2c", time_mode="continuing", time_features=False, gamma=0.99)
        base.update(kw)
        return cls(**base)

    @classmethod
    def m4(cls, **kw) -> "GameConfig":
        """The M4 ruleset (docs/m4_integrated_design.md): continuing time, Freeze maintenance, XP and levels 1-18,
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

    # ------------------------------------------------------------------ helpers
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
        self._validate_ruleset()
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
        if self.zoning_rule not in ("presence", "contact", "none"):
            bad("zoning_rule", "must be 'presence', 'contact' or 'none'")
        if self.siege_engage:
            bad("siege_engage", "must be False (the evaluation bound counts no siege fights, design §10.5.3)")
        self._validate_exact_economy()

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

        for v, name in ((self.plate_gold, "plate_gold"), (self.kill_bounty, "kill_bounty")):
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
                     "rear_count_max", "position_rule", "zoning_rule")
_FIELD_DEFAULTS = {f.name: f.default for f in GameConfig.__dataclass_fields__.values()}


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
    bases = {"contest": q(cfg.collect_contest), "contest_enemy_tower": q(cfg.collect_contest_enemy_tower)}
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
