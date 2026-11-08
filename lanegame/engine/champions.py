"""Champion layer (roadmap milestone M2): champion state, positioning, behaviors, Recall, trading, ganks, death.

Champion mode (`GameConfig.champions()`) adds a champion per player on top of the siege model's lane. The
lane itself (minion movement, merging, fights, thinning, crashes, tower fire, spawns, plates, vision) is
still advanced by `mechanics.step_turn`, unchanged; this module decides where each champion stands, how
much it helps its wave, what it farms and how exposed it is. Champion mode is played through `lanegame.env`
and is not exactly solvable; the legacy configurations never reach this module.

State (per side, own coordinates as everywhere else: own tower -T, the lane 0, enemy tower +T)

    pos      champion cell, -T .. T (start: own tower, -T; -T while away)
    hp       champion HP, 0 .. champion_hp (6); trades, tower shots and ganks remove it, 0 = dead (while
             away), a return restores it
    absent   number of upcoming turns the champion is forced to sit out (0 = present). Recall sets it to
             recall_turns - 1 (the Recall turn itself is the first missed turn), death to death_turns (the
             champion acted on its death turn); it counts down by one at the end of every absent turn, and
             the champion returns, at its own tower with full HP, at the start of the turn after the one on
             which it reaches 0.

plus the per-seat mechanical skills `cfg.skill_a`, `cfg.skill_b` (game constants in [0, 1], observed by
both players; they enter only the execution probabilities of trades, see "Trading"). `ChampState.mirror()` swaps the sides; with asymmetric skills the mirrored
game is `swap_seats(cfg)`.

Actions: a = wave + 3 * behavior + 15 * objective (config.champ_act). Legal sets (`champion_legal_actions`):

    wave       the lane's own rule (mechanics.wave_actions): SP, SH, and F only when the enemy front is at or
               behind own -1
    behavior   HOLD, YIELD, TRADE, SAFE, each with objective NONE
    objective  PRESSURE or VISION, only with HOLD or TRADE, only while the side has crashed mass under the
               enemy tower (siege eligible) AND its champion can reach the enemy tower this turn
               (pos + champion_speed >= T). There is no remote siege.
    recall     the single code RECALL = champ_act(SP, B_RECALL), legal only when `recall_legal`: the champion
               is present and stands at or behind recall_line (-1). A champion further forward must first
               walk back (Safe / Yield), at most champion_speed cells per turn, and recall on a later turn.
               Whether an enemy threat materializes is decided by the committed pair (step 2b): an enemy
               TRADE that reaches the recaller interrupts the recall.
    absent     while absent > 0 the only legal code is ABSENT (45); ABSENT is never legal otherwise.

Unlike the legacy rule (Contest without an objective is not offered during a siege), HOLD and TRADE without
an objective stay legal during a siege: they mean "stay with my next wave" instead of walking to the tower,
which changes the champion's position and exposure, so the objective no longer dominates them.

Recall timeline (recall_turns = 2): Recall chosen on turn t and not interrupted -> the champion leaves the
lane immediately and takes no part in turn t; forced ABSENT on turn t + 1; back at its own tower with full HP
at the start of turn t + 2, where it chooses a normal action. Exactly recall_turns turns without lane
participation. The recall itself costs nothing else: its price is the farm, zoning, wave control and siege
opportunities given up while away, so a recall timed while one's own wave pushes is cheaper than one timed
into an enemy crash. An interrupted recall (step 2b) wastes the turn: the champion stays where it was,
channeling (no farm, no wave contribution, no zoning), can be traded and ganked, and plays normally next turn.

Resolution of one turn (`resolve_turn`), from the two committed actions; no player ever sees the other's
current action, but the simulator resolves the committed pair together:

 0. Away sides. A champion that is absent, or that recalls this turn without being interrupted (2b), is off
    the lane for the whole turn: its wave plays Slow Push with no champion contribution (reach NO_REACH: no
    shove damage, no freeze-break push), it has no siege objective, collects no farm, zones nothing, cannot
    trade or be traded and is not gankable. Minions, towers, plates and vision timers keep running as usual.
 1. Wave destinations. The lane is probed once with every present champion supporting its wave (the legacy
    Contest rule). Each present side's *farm spot* is its leading wave's cell at the end of that probe,
    clipped to [-T, T] (its current cell if it has no wave).
 2. Targets. First every non-trader's intended cell:
      objective (PRESSURE / VISION)  the enemy tower, +T
      SAFE       min(farm spot, safe_line): follow the wave, never beyond the safe line (a champion
                 that starts forward of it retreats, at most champion_speed cells per turn)
      HOLD/YIELD the farm spot
      RECALL     stays at its start cell (channeling)
    Then each TRADE champion (without an objective) chases its opponent's reference cell: the opponent's
    intended end cell, or its start cell if the opponent also trades. It stays put if it is already in
    contact (below) with that cell; otherwise it heads for the cell trade_range short of it, or, if the
    reference is the opponent's own tower cell, into that cell (a dive: the only way to reach a champion
    under its tower). A TRADE champion whose opponent is absent has nobody to chase and targets its farm
    spot, like HOLD. Every champion moves toward its target by at most champion_speed cells.
 2b. Recall interruption. A recalling champion whose cell is in contact with an enemy TRADE champion's end
    cell is interrupted (mode "channel"); otherwise it leaves the lane (step 0, end cell -T).
 3. Evasion. A YIELD or SAFE champion whose would-be end cell is within trade_range of an enemy TRADE
    champion's end cell is *threatened*: it retreats instead, toward the furthest-forward cell out of the
    trader's range (never ahead of its own target, never behind its own tower), again at most
    champion_speed cells. A YIELD champion that is not threatened behaves exactly like HOLD; a threatened
    one gives up its farming position and its wave contribution (mode "retreat"). SAFE is already defensive,
    so evading costs it nothing further. HOLD never evades.
 4. Trade contact. Two present champions are in *contact* when they end within trade_range of each other,
    except that a champion in its own tower cell is protected: contact with it needs the other champion in
    that same cell. A TRADE champion *connects* when the champions end in contact. A connected
    trader's siege objective (if any) is suspended for the turn: its attention is on the enemy champion.
    An unconnected trader falls back to farming where it stands, keeping its forward position.
 5. Modes and wave contribution (the champion's own effect on its wave: shove damage, freeze-break push):
      farm     HOLD, unthreatened YIELD, unconnected TRADE: contributes to fights up to end + farm_reach
      trading  connected TRADE: no contribution
      retreat  threatened YIELD: contributes only at or behind min(end, safe_line)
      safe     SAFE: likewise
      channel  interrupted RECALL: no contribution
      away     recalled or absent: no contribution
 6. The authoritative lane step: mechanics.step_turn with the effective siege objectives and the
    contributions of 5 as `champion_reach`.
 7. Farming. Every killed HP is a farm opportunity as before (game.turn_farm's accounting); a champion
    collects only kills within farm_reach cells of its end cell. Efficiency is the legacy
    `collection_efficiency` with posture Contest for farm / trading, Safe for retreat / safe; the opponent
    zones a kill only if it is in farm or trading mode and within farm_reach of the kill. A connected
    trader's collection is multiplied by trade_farm_factor. A threatened Yield (mode "retreat") collects
    kills within retreat_farm_reach cells of its retreat cell. A channeling or away champion collects nothing
    and zones nothing.
 8. Exposure and gank risk. exposure = max(start cell, end cell): a champion that starts forward stays
    exposed while it retreats (choosing Safe does not erase this turn's exposure). The gank occurrence
    probability has the legacy form (base + position + behavior - wave strength, clipped, times the vision
    factor) with its own champion-mode parameters (champ_gank_*, see GameConfig; no unconditional base rate;
    a channeling champion has Safe's behavior term). An away champion has gank probability 0. A gank that
    occurs deals Binomial(gank_damage_trials, p_e) HP, p_e rising with exposure and falling with skill and
    for a champion that actually played defensively (mode safe or retreat; gank_damage_*). Vision lowers only
    the occurrence.
 9. Chance events, all from the turn's uniform vector u (layout below): each connected trader's damage on
    its target, each diver's tower shot, each present side's gank occurrence (p = gank_p) and gank damage.
    All damage of the turn is simultaneous and summed: trade damage, a tower shot and a gank together can
    kill a champion that each alone would not, and both champions can die in the same turn.
10. Death and champion state after the turn: a present champion whose HP reaches 0 dies: HP 0, at -T,
    absent = death_turns (the farm of the death turn is kept). Any other present champion stands at its
    end cell with its HP minus the damage it took. A champion that recalled gets absent = recall_turns - 1,
    an absent one absent - 1; both stand at -T, and a champion whose timer is now 0 has full HP again (it
    plays the next turn). The opponent of a champion that died receives kill_bounty (default 0).

Trading (M2c). A connected Trade of side i on side j hits with probability

    h_i = clip(trade_hit_base + trade_hit_skill (m_i - m_j) + trade_hit_behavior[behavior_j]
               + trade_hit_minions (sigma_i - sigma_j) - trade_hit_tower [j in its own tower cell],
               trade_hit_min, trade_hit_max)

and a hit deals 2 HP with probability q_i = clip(trade_double_base + trade_double_skill (m_i - m_j), 0, 1),
else 1 HP: P(0, 1, 2) = (1 - h, h (1 - q), h q). m are the mechanical skills, sigma the leading-wave stack
counts at the start of the turn (local minion support), behavior_j the defender's behavior (an interrupted
recall counts as Recall: channeling, fully exposed). Defaults: equal skill and a Hold defender give
h = 0.55, q = 0.3, expected damage 0.72 HP; a Safe defender that could not escape h = 0.15. Skill enters
nothing else: it changes how well a chosen action is executed, never which actions are available or what
they cost. Opportunity cost: the connected trader gives up its wave contribution and half its farm
(trade_farm_factor); an unconnected trader farms normally at its forward cell. A defender that does not
Trade does not hit back. Trade against Trade: both roll.

Tower shot (M2c). A connected trader whose end cell is the enemy tower cell (a dive; the only way to reach a
champion under its own tower) takes tower_shot_damage HP with probability tower_shot_p, or tower_shot_p_busy
if its own crashed minions were under that tower this turn (at the start of the turn or crashing during it).

Chance vector u (N_CHANCE uniforms per turn, drawn by the environment whatever happens, so that two policies
see identical draws turn by turn): U_TRADE_HIT[i], U_TRADE_DOUBLE[i] for side i's attack, U_TOWER[i] for side
i's tower shot, U_GANK[i] for side i being ganked, U_GANK_DAMAGE[i] for the damage of that gank (inverse
CDF of the binomial). An event happens when its uniform is below its probability; u = None means no chance
event at all.

Siege objectives are executed by the lane step only when they are effective (legal, i.e. the champion
reaches the enemy tower, and not suspended by a connected trade), so plates, thinning and vision always
require the champion at the enemy tower.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

from .config import (B_ABSENT, B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, CHAMP_ABSENT, CHAMP_BEHAVIORS, CONTEST,
                     NONE, PRESSURE, SAFE, SH, SP, VISION, GameConfig, champ_act, champ_behavior, champ_objective,
                     champ_wave)
from .game import collection_efficiency
from .mechanics import TurnResult, siege_eligible, step_turn, wave_actions
from .positional import WARD, dist as graph_dist, in_contact as graph_contact, move as graph_move
from .state import State

A, B = 0, 1
FARM, TRADING, RETREAT, SAFE_MODE, CHANNEL, AWAY = "farm", "trading", "retreat", "safe", "channel", "away"
NO_REACH = -10 ** 6                      # contributes to no fight
RECALL = champ_act(SP, B_RECALL)         # the only legal Recall code (12)

# chance vector layout (index per side A, B)
U_TRADE_HIT, U_TRADE_DOUBLE, U_TOWER, U_GANK, U_GANK_DAMAGE = (0, 2), (1, 3), (4, 5), (6, 7), (8, 9)
N_CHANCE = 10


def mirror_chance(u: Sequence[float]) -> Tuple[float, ...]:
    """The chance vector of the mirrored game (A and B entries swapped)."""
    out = list(u)
    for pair in (U_TRADE_HIT, U_TRADE_DOUBLE, U_TOWER, U_GANK, U_GANK_DAMAGE):
        out[pair[0]], out[pair[1]] = u[pair[1]], u[pair[0]]
    return tuple(out)


class ChampSide(NamedTuple):
    pos: int          # own coordinates, -T .. T
    hp: int
    absent: int = 0


class ChampState(NamedTuple):
    a: ChampSide
    b: ChampSide

    def side(self, s: int) -> ChampSide:
        return self.a if s == A else self.b

    def mirror(self) -> "ChampState":
        return ChampState(self.b, self.a)

    def to_list(self) -> list:
        return [list(self.a), list(self.b)]

    @classmethod
    def from_list(cls, v) -> "ChampState":
        return cls(ChampSide(*v[0]), ChampSide(*v[1]))


def initial_champions(cfg: GameConfig) -> ChampState:
    """Both champions at their own tower with full HP, present."""
    s = ChampSide(-cfg.tower, cfg.champion_hp, 0)
    return ChampState(s, s)


def skill_of(cfg: GameConfig, s: int) -> float:
    return cfg.skill_a if s == A else cfg.skill_b


def swap_seats(cfg: GameConfig) -> GameConfig:
    """The configuration of the mirrored game (seat-specific parameters swapped)."""
    return cfg.with_(skill_a=cfg.skill_b, skill_b=cfg.skill_a)


# ---------------------------------------------------------------------- legality
def objective_reachable(lane: State, champ: ChampState, s: int, cfg: GameConfig) -> bool:
    """Side s may choose a siege objective: its crashed minions are under the enemy tower AND its champion
    can stand at the enemy tower by the end of this turn (M4.5a: by movement-graph distance; never from W)."""
    pos = champ.side(s).pos
    if pos == WARD:
        return siege_eligible(lane, s, cfg) and graph_dist(pos, cfg.tower) <= cfg.champion_speed
    return siege_eligible(lane, s, cfg) and pos + cfg.champion_speed >= cfg.tower


def ward_legal(champ, s: int, cfg: GameConfig) -> bool:
    """M4.5a (vision_rule "ward"): side s may Ward: present and within champion_speed edges of the Vision node W (own
    cells -1, 0, +1 at the default speed 2, or W itself)."""
    me = champ.side(s)
    return cfg.vision_rule == "ward" and me.absent == 0 and graph_dist(me.pos, WARD) <= cfg.champion_speed


def recall_legal(champ: ChampState, s: int, cfg: GameConfig) -> bool:
    """Side s may Recall: present and at or behind recall_line. (An enemy trade that reaches it interrupts the
    recall at resolution; see resolve_turn step 2b.)"""
    me = champ.side(s)
    return me.absent == 0 and me.pos != WARD and me.pos <= cfg.recall_line


def champion_legal_actions(lane: State, champ: ChampState, s: int, cfg: GameConfig) -> Tuple[int, ...]:
    """Legal champion-mode action codes of side s, sorted (see the module docstring)."""
    if champ.side(s).absent > 0:
        return (CHAMP_ABSENT,)
    waves = wave_actions(lane, s, cfg)
    acts = [champ_act(w, b) for b in CHAMP_BEHAVIORS for w in waves]
    # M4.5a vision_rule "ward": Vision is no siege objective but the single Ward code (champ_act(SP, B_HOLD, VISION))
    siege_objectives = (PRESSURE,) if cfg.vision_rule == "ward" else (PRESSURE, VISION)
    if objective_reachable(lane, champ, s, cfg):
        acts += [champ_act(w, b, o) for o in siege_objectives for b in (B_HOLD, B_TRADE) for w in waves]
    if ward_legal(champ, s, cfg):
        acts.append(champ_act(SP, B_HOLD, VISION))
    if recall_legal(champ, s, cfg):
        acts.append(RECALL)
    return tuple(sorted(acts))


# ---------------------------------------------------------------------- economics and risk
def champion_collection(x: int, mode: str, pos: int, opp_mode: str, opp_pos: int, cfg: GameConfig) -> float:
    """Share of a farm opportunity at own-perspective cell x that a champion in `mode` at `pos` collects
    against an opponent in `opp_mode` at its own cell `opp_pos`."""
    if mode in (AWAY, CHANNEL) or abs(x - pos) > (cfg.retreat_farm_reach if mode == RETREAT else cfg.farm_reach):
        return 0.0
    posture = CONTEST if mode in (FARM, TRADING) else SAFE
    zones = opp_mode in (FARM, TRADING) and abs(-x - opp_pos) <= cfg.farm_reach
    c = collection_efficiency(x, posture, CONTEST if zones else SAFE, cfg)
    return c * cfg.trade_farm_factor if mode == TRADING else c


def champion_gank_probability(lane: State, s: int, wave: int, behavior: int, cfg: GameConfig, exposure: int) -> float:
    """P(side s is ganked this turn) = clip(b0 + b_pos (e + T) / 2T + b_beh[behavior] + b_shove [Shove, non-Safe]
    - b_str dsigma, 0, p_max) times (1 - rho) while side s's vision is active (timer at the START of the turn).
    The form of game.gank_probability with the champion-mode parameters champ_gank_*."""
    own_h = lane.ah if s == A else lane.bh
    opp_h = lane.bh if s == A else lane.ah
    dsig = cfg.stacks(own_h) - cfg.stacks(opp_h)
    term = cfg.champ_gank_behavior[behavior] + (cfg.champ_gank_shove if wave == SH and behavior != B_SAFE else 0.0)
    p = (cfg.champ_gank_base + cfg.champ_gank_position * (exposure + cfg.tower) / (2 * cfg.tower) + term
         - cfg.champ_gank_strength * dsig)
    p = min(max(p, 0.0), cfg.champ_gank_max)
    if lane.vision_of(s) > 0:
        p *= (1.0 - cfg.vision_reduction)
    return p


def gank_damage_p(s: int, defensive: bool, exposure: int, cfg: GameConfig) -> float:
    """Per-trial damage probability p_e of a gank on side s (vision plays no part). `defensive`: the champion
    actually played defensively this turn (Safe, or a Yield that backed off from a trader), not merely chose
    a behavior that would have; an unthreatened Yield stands and farms exactly like Hold."""
    pe = (cfg.gank_damage_base + cfg.gank_damage_position * (exposure + cfg.tower) / (2 * cfg.tower)
          - (cfg.gank_damage_defense if defensive else 0.0)
          - cfg.gank_damage_skill * (skill_of(cfg, s) - 0.5))
    return min(max(pe, 0.0), 1.0)


def gank_damage_distribution(pe: float, cfg: GameConfig) -> Tuple[float, ...]:
    """P(gank damage = d), d = 0 .. gank_damage_trials (binomial)."""
    n = cfg.gank_damage_trials
    return tuple(comb(n, d) * pe ** d * (1.0 - pe) ** (n - d) for d in range(n + 1))


def _gank_damage(u: float, pe: float, cfg: GameConfig) -> int:
    """Inverse-CDF draw of the gank damage from one uniform."""
    c = 0.0
    dist = gank_damage_distribution(pe, cfg)
    for d, p in enumerate(dist):
        c += p
        if u < c:
            return d
    return len(dist) - 1


def trade_hit_probability(lane: State, s: int, defender_behavior: int, defender_in_tower: bool,
                          cfg: GameConfig) -> float:
    """P(side s's connected trade lands), see "Trading" in the module docstring."""
    o = 1 - s
    own_h = lane.ah if s == A else lane.bh
    opp_h = lane.bh if s == A else lane.ah
    h = (cfg.trade_hit_base + cfg.trade_hit_skill * (skill_of(cfg, s) - skill_of(cfg, o))
         + cfg.trade_hit_behavior[defender_behavior]
         + cfg.trade_hit_minions * (cfg.stacks(own_h) - cfg.stacks(opp_h))
         - (cfg.trade_hit_tower if defender_in_tower else 0.0))
    return min(max(h, cfg.trade_hit_min), cfg.trade_hit_max)


def trade_double_probability(s: int, cfg: GameConfig) -> float:
    """P(2 HP | hit) for side s's trade."""
    q = cfg.trade_double_base + cfg.trade_double_skill * (skill_of(cfg, s) - skill_of(cfg, 1 - s))
    return min(max(q, 0.0), 1.0)


def trade_damage_distribution(h: float, q: float) -> Tuple[float, float, float]:
    """(P(0), P(1), P(2)) HP of one connected trade."""
    return 1.0 - h, h * (1.0 - q), h * q


def _trade_damage(u_hit: float, u_double: float, h: float, q: float) -> int:
    if u_hit >= h:
        return 0
    return 2 if u_double < q else 1


# ---------------------------------------------------------------------- one turn
LaneStep = Callable[[State, int, int, int, Optional[Tuple[int, int]]], TurnResult]


@dataclass(frozen=True)
class ChampTurn:
    lane: TurnResult                       # the authoritative lane step (with champion contributions)
    champ: ChampState                      # champion state after the turn
    waves: Tuple[int, int]
    behaviors: Tuple[int, int]             # B_ABSENT for an absent side
    objectives: Tuple[int, int]            # chosen
    effective_objectives: Tuple[int, int]  # executed by the lane step (suspended by a connected trade)
    start: Tuple[int, int]                 # champion cells at the start of the turn (own coordinates)
    end: Tuple[int, int]                   # ... at the end (-T for an away side)
    farm_spot: Tuple[int, int]
    away: Tuple[bool, bool]                # absent, or recalled successfully this turn (off the lane)
    recalled: Tuple[bool, bool]            # chose Recall this turn and left the lane
    interrupted: Tuple[bool, bool]         # chose Recall this turn and was interrupted by an enemy trade
    threatened: Tuple[bool, bool]          # YIELD or SAFE champion threatened by a TRADE (evaded)
    connected: Tuple[bool, bool]           # TRADE champion in contact
    modes: Tuple[str, str]
    reach: Tuple[int, int]                 # champion contribution line passed to the lane step
    lane_codes: Tuple[int, int]            # legacy codes given to the lane step
    farm_available: Tuple[float, float]
    farm: Tuple[float, float]              # collected
    exposure: Tuple[int, int]              # max(start, end) for a present side; -T for an away side
    gank_p: Tuple[float, float]            # 0 for an away side
    hit_p: Tuple[float, float] = (0.0, 0.0)          # P(side i's connected trade lands); 0 if not connected
    double_p: Tuple[float, float] = (0.0, 0.0)       # P(2 HP | hit) of side i's trade
    tower_p: Tuple[float, float] = (0.0, 0.0)        # P(side i takes a tower shot); 0 unless it dives
    trade_damage: Tuple[int, int] = (0, 0)           # HP dealt BY side i's trade
    tower_damage: Tuple[int, int] = (0, 0)           # HP taken by side i from the enemy tower
    damage_taken: Tuple[int, int] = (0, 0)           # all HP lost by side i this turn
    ganked: Tuple[bool, bool] = (False, False)       # realized gank occurrence
    gank_damage_p: Tuple[float, float] = (0.0, 0.0)  # per-trial damage probability p_e if ganked
    gank_damage: Tuple[int, int] = (0, 0)            # HP lost to the gank (0 if none)
    died: Tuple[bool, bool] = (False, False)         # HP reached 0 this turn
    bounty: Tuple[float, float] = (0.0, 0.0)         # kill_bounty received by side i (its opponent died)


def _clip(x: int, cfg: GameConfig) -> int:
    return max(-cfg.tower, min(cfg.tower, x))


def _move(c: int, target: int, cfg: GameConfig) -> int:
    if c == WARD or target == WARD:                      # M4.5a: the movement graph (W is entered and left via 0)
        return graph_move(c, target, cfg.champion_speed, cfg.tower)
    step = max(-cfg.champion_speed, min(cfg.champion_speed, target - c))
    return _clip(c + step, cfg)


def in_contact(x: int, y: int, cfg: GameConfig) -> bool:
    """Champions at own cells x and y (each in its own coordinates) can trade: within trade_range, except that
    a champion in its own tower cell can only be reached from inside that cell. M4.5a: a champion in the Vision node W
    is in contact by movement-graph distance (positional.in_contact)."""
    if x == WARD or y == WARD:
        return graph_contact(x, y, cfg.tower, cfg.trade_range)
    if x == -cfg.tower or y == -cfg.tower:
        return x + y == 0
    return abs(x + y) <= cfg.trade_range


def _evasion_cell(target: int, trader_end: int, cfg: GameConfig) -> int:
    """The furthest-forward cell at or behind `target` out of contact with a trader ending at `trader_end`
    (own tower if none)."""
    for x in range(target, -cfg.tower - 1, -1):
        if not in_contact(x, trader_end, cfg):
            return x
    return -cfg.tower


def _decode(a: int) -> Tuple[int, int, int]:
    """(wave, behavior, objective); ABSENT decodes to (SP, B_ABSENT, NONE)."""
    if a == CHAMP_ABSENT:
        return SP, B_ABSENT, NONE
    return champ_wave(a), champ_behavior(a), champ_objective(a)


def _after_away(side: ChampSide, recalled: bool, cfg: GameConfig) -> ChampSide:
    """Champion record at the end of a turn spent away: timer set (Recall) or counted down (absent); at -T;
    full HP once the timer reaches 0 (it plays the next turn)."""
    left = cfg.recall_turns - 1 if recalled else side.absent - 1
    return ChampSide(-cfg.tower, cfg.champion_hp if left == 0 else side.hp, left)


def resolve_turn(lane: State, champ: ChampState, k: int, a: int, b: int, cfg: GameConfig,
                 lane_step: Optional[LaneStep] = None, u: Optional[Sequence[float]] = None) -> ChampTurn:
    """Resolve decision turn k of champion mode from both committed actions (module docstring, steps 0-10).
    `u` is the turn's chance vector (N_CHANCE uniforms in [0, 1); None = no chance event happens).
    Legality is the caller's responsibility (the environment checks `champion_legal_actions`)."""
    if lane_step is None:
        def lane_step(s, kk, x, y, reach):
            return step_turn(s, kk, x, y, cfg, champion_reach=reach)
    T, rng = cfg.tower, cfg.trade_range
    (wa, ba, oa), (wb, bb, ob) = _decode(a), _decode(b)
    w, beh, obj = (wa, wb), (ba, bb), (oa, ob)
    c0 = (champ.a.pos, champ.b.pos)
    recalling = (ba == B_RECALL, bb == B_RECALL)
    absent = (ba == B_ABSENT, bb == B_ABSENT)

    # 1. wave destinations under full support of every champion that intends to play the lane
    off = tuple(recalling[s] or absent[s] for s in (A, B))
    probe = lane_step(lane, k, wa + 6 * oa, wb + 6 * ob, tuple(NO_REACH if off[s] else T for s in (A, B)))
    spot = []
    for s in (A, B):
        fx, fh = probe.state.side(s)[:2]
        spot.append(_clip(fx, cfg) if fh > 0 else c0[s])

    # 2. targets and bounded movement: non-traders first, then traders chase their opponent's reference cell
    end = [0, 0]
    chasers = []
    for s in (A, B):
        c = c0[s]
        if absent[s]:
            end[s] = -T
            continue
        if recalling[s]:
            end[s] = c                                                    # channels where it stands
            continue
        if obj[s] != NONE:
            target = T
        elif beh[s] == B_TRADE and not absent[1 - s]:
            chasers.append(s)
            continue
        elif beh[s] == B_SAFE:
            target = min(spot[s], cfg.safe_line)
        else:
            target = spot[s]                                              # HOLD, YIELD, TRADE with nobody to chase
        end[s] = _move(c, target, cfg)
    for s in chasers:
        o, c = 1 - s, c0[s]
        ref = c0[o] if beh[o] == B_TRADE and obj[o] == NONE else end[o]   # opponent's cell, its own coordinates
        e = -ref                                                          # ... in s's coordinates
        if in_contact(c, ref, cfg):
            target = c
        elif e == T:
            target = T                                                    # dive into the enemy tower cell
        else:
            target = e - rng if c < e else e + rng
        end[s] = _move(c, target, cfg)

    # 2b. recall interruption by an enemy trader that reaches the channeling champion
    interrupted = tuple(recalling[s] and beh[1 - s] == B_TRADE and in_contact(end[s], end[1 - s], cfg)
                        for s in (A, B))
    recalled = tuple(recalling[s] and not interrupted[s] for s in (A, B))
    away = tuple(absent[s] or recalled[s] for s in (A, B))
    for s in (A, B):
        if away[s]:
            end[s] = -T

    # 3. evasion by YIELD and SAFE (the trader's end cell depends on start positions only)
    threatened = [False, False]
    for s in (A, B):
        o = 1 - s
        if beh[s] in (B_YIELD, B_SAFE) and beh[o] == B_TRADE and in_contact(end[s], end[o], cfg):
            threatened[s] = True
            own_target = spot[s] if beh[s] == B_YIELD else min(spot[s], cfg.safe_line)
            end[s] = _move(c0[s], _evasion_cell(own_target, end[o], cfg), cfg)

    # 4. trade contact (both champions present)
    in_range = not (away[A] or away[B]) and in_contact(end[0], end[1], cfg)
    connected = tuple(beh[s] == B_TRADE and in_range for s in (A, B))
    eff_obj = tuple(NONE if connected[s] else obj[s] for s in (A, B))

    # 5. modes and wave contribution
    modes, reach = [], []
    for s in (A, B):
        if away[s] or interrupted[s]:
            modes.append(AWAY if away[s] else CHANNEL)
            reach.append(NO_REACH)
        elif connected[s]:
            modes.append(TRADING)
            reach.append(NO_REACH)
        elif beh[s] == B_SAFE or threatened[s]:
            modes.append(SAFE_MODE if beh[s] == B_SAFE else RETREAT)
            reach.append(min(end[s], cfg.safe_line))
        else:
            modes.append(FARM)
            reach.append(min(T, end[s] + cfg.farm_reach))

    # 6. the authoritative lane step
    codes = (wa + 6 * eff_obj[0], wb + 6 * eff_obj[1])
    res = lane_step(lane, k, codes[0], codes[1], (reach[0], reach[1]))

    # 7. farming
    avail, coll = [0.0, 0.0], [0.0, 0.0]
    for s in (A, B):
        o = 1 - s
        for x, hp in res.kills[s].items():
            v = cfg.farm_per_hp * hp
            avail[s] += v
            coll[s] += v * champion_collection(x, modes[s], end[s], modes[o], end[o], cfg)

    # 8. exposure and gank risk
    exposure = tuple(-T if away[s] else max(c0[s], end[s]) for s in (A, B))
    gank = tuple(0.0 if away[s] else
                 champion_gank_probability(lane, s, w[s], B_SAFE if interrupted[s] else beh[s], cfg, exposure[s])
                 for s in (A, B))

    # 9. chance events: trades (simultaneous), tower shots on divers, gank occurrence
    if u is None:
        u = (1.0,) * N_CHANCE
    hit, dbl, tower_p = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
    dealt, tower_dmg = [0, 0], [0, 0]
    for s in (A, B):
        if not connected[s]:
            continue
        o = 1 - s
        def_beh = B_RECALL if interrupted[o] else beh[o]
        hit[s] = trade_hit_probability(lane, s, def_beh, end[o] == -T, cfg)
        dbl[s] = trade_double_probability(s, cfg)
        dealt[s] = _trade_damage(u[U_TRADE_HIT[s]], u[U_TRADE_DOUBLE[s]], hit[s], dbl[s])
        if end[s] == T:                                                   # a dive into the enemy tower cell
            busy = lane.siege_of(s) > 0 or res.crash_hp[s] > 0
            tower_p[s] = cfg.tower_shot_p_busy if busy else cfg.tower_shot_p
            tower_dmg[s] = cfg.tower_shot_damage if u[U_TOWER[s]] < tower_p[s] else 0
    ganked = tuple(u[U_GANK[s]] < gank[s] for s in (A, B))
    pe = tuple(0.0 if away[s] else gank_damage_p(s, modes[s] in (RETREAT, SAFE_MODE), exposure[s], cfg)
               for s in (A, B))
    gank_dmg = tuple(_gank_damage(u[U_GANK_DAMAGE[s]], pe[s], cfg) if ganked[s] else 0 for s in (A, B))
    taken = tuple(dealt[1 - s] + tower_dmg[s] + gank_dmg[s] for s in (A, B))
    died = tuple(not away[s] and champ.side(s).hp - taken[s] <= 0 for s in (A, B))
    bounty = (cfg.kill_bounty * died[B], cfg.kill_bounty * died[A])

    # 10. death and champion state after the turn
    def after(s):
        me = champ.side(s)
        if away[s]:
            return _after_away(me, recalled[s], cfg)
        if died[s]:
            left = cfg.death_turns
            return ChampSide(-T, cfg.champion_hp if left == 0 else 0, left)
        return me._replace(pos=end[s], hp=me.hp - taken[s])
    sides = (after(A), after(B))
    return ChampTurn(res, ChampState(*sides), w, beh, obj, eff_obj, c0, (end[0], end[1]), (spot[0], spot[1]),
                     away, recalled, interrupted, (threatened[0], threatened[1]), connected, (modes[0], modes[1]),
                     (reach[0], reach[1]), codes, (avail[0], avail[1]), (coll[0], coll[1]), exposure, gank,
                     (hit[0], hit[1]), (dbl[0], dbl[1]), (tower_p[0], tower_p[1]), (dealt[0], dealt[1]),
                     (tower_dmg[0], tower_dmg[1]), taken, ganked, pe, gank_dmg, died, bounty)
