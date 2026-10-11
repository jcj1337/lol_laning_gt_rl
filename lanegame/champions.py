"""Champion layer (roadmap milestone M2a): champion state, persistent positioning and behaviors.

Champion mode (`GameConfig.champions()`) adds a champion per player on top of the siege model's lane. The
lane itself (minion movement, merging, fights, thinning, crashes, tower fire, spawns, plates, vision) is
still advanced by `mechanics.step_turn`, unchanged; this module decides where each champion stands, how
much it helps its wave, what it farms and how exposed it is. Champion mode is played through `lanegame.env`
and is not exactly solvable; the legacy configurations never reach this module.

State (per side, own coordinates as everywhere else: own tower -T, the lane 0, enemy tower +T)

    pos      champion cell, -T .. T (start: own tower, -T)
    hp       champion HP (champion_hp = 6; constant until M2c/M2d add damage)
    absent   turns away from the lane (0; recall and death arrive in M2b/M2d)

plus the per-seat mechanical skills `cfg.skill_a`, `cfg.skill_b` (game constants, observed by both
players, no effect until M2c). `ChampState.mirror()` swaps the sides; with asymmetric skills the mirrored
game is `swap_seats(cfg)`.

Actions: a = wave + 3 * behavior + 15 * objective (config.champ_act). Legal sets (`champion_legal_actions`):

    wave       the lane's own rule (mechanics.wave_actions): SP, SH, and F only when the enemy front is at or
               behind own -1
    behavior   HOLD, YIELD, TRADE, SAFE, each with objective NONE
    objective  PRESSURE or VISION, only with HOLD or TRADE, only while the side has crashed mass under the
               enemy tower (siege eligible) AND its champion can reach the enemy tower this turn
               (pos + champion_speed >= T). There is no remote siege.
    reserved   RECALL codes and ABSENT (45) are never legal in M2a.

Unlike the legacy rule (Contest without an objective is not offered during a siege), HOLD and TRADE without
an objective stay legal during a siege: they mean "stay with my next wave" instead of walking to the tower,
which changes the champion's position and exposure, so the objective no longer dominates them.

Resolution of one turn (`resolve_turn`), from the two committed actions; no player ever sees the other's
current action, but the simulator resolves the committed pair together:

 1. Wave destinations. The lane is probed once with both champions supporting their waves (the legacy
    Contest rule). Each side's *farm spot* is its leading wave's cell at the end of that probe, clipped to
    [-T, T] (its current cell if it has no wave).
 2. Targets. First every non-trader's intended cell:
      objective (PRESSURE / VISION)  the enemy tower, +T
      SAFE       min(farm spot, safe_line): follow the wave, never beyond the safe line (a champion
                 that starts forward of it retreats, at most champion_speed cells per turn)
      HOLD/YIELD the farm spot
    Then each TRADE champion (without an objective) chases its opponent's reference cell: the opponent's
    intended end cell, or its start cell if the opponent also trades. It stays put if it is already in
    contact (below) with that cell; otherwise it heads for the cell trade_range short of it, or, if the
    reference is the opponent's own tower cell, into that cell (a dive: the only way to reach a champion
    under its tower). Every champion moves toward its target by at most champion_speed cells.
 3. Evasion. A YIELD or SAFE champion whose would-be end cell is within trade_range of an enemy TRADE
    champion's end cell is *threatened*: it retreats instead, toward the furthest-forward cell out of the
    trader's range (never ahead of its own target, never behind its own tower), again at most
    champion_speed cells. A YIELD champion that is not threatened behaves exactly like HOLD; a threatened
    one gives up its farming position and its wave contribution (mode "retreat"). SAFE is already defensive,
    so evading costs it nothing further. HOLD never evades.
 4. Trade contact. Two champions are in *contact* when they end within trade_range of each other, except
    that a champion in its own tower cell is protected: contact with it needs the other champion in that
    same cell. A TRADE champion *connects* when the champions end in contact. A connected
    trader's siege objective (if any) is suspended for the turn: its attention is on the enemy champion.
    An unconnected trader falls back to farming where it stands, keeping its forward position.
 5. Modes and wave contribution (the champion's own effect on its wave: shove damage, freeze-break push):
      farm     HOLD, unthreatened YIELD, unconnected TRADE: contributes to fights up to end + farm_reach
      trading  connected TRADE: no contribution
      retreat  threatened YIELD: contributes only at or behind min(end, safe_line)
      safe     SAFE: likewise
 6. The authoritative lane step: mechanics.step_turn with the effective siege objectives and the
    contributions of 5 as `champion_reach`.
 7. Farming. Every killed HP is a farm opportunity as before (game.turn_farm's accounting); a champion
    collects only kills within farm_reach cells of its end cell. Efficiency is the legacy
    `collection_efficiency` with posture Contest for farm / trading, Safe for retreat / safe; the opponent
    zones a kill only if it is in farm or trading mode and within farm_reach of the kill. A connected
    trader's collection is multiplied by trade_farm_factor.
 8. Exposure and gank risk. exposure = max(start cell, end cell): a champion that starts forward stays
    exposed while it retreats. The gank probability keeps the legacy form (base + position + action
    - wave strength, clipped, times the vision factor) with a behavior-based action term
    (champ_gank_behavior, plus champ_gank_shove for Shove with a non-Safe behavior). In M2a the gank is
    still a resource loss with that probability (the environment realizes it); HP events come in M2d.
 9. Champion state after the turn: the end cells; HP and absence unchanged.

Siege objectives are executed by the lane step only when they are effective (legal, i.e. the champion
reaches the enemy tower, and not suspended by a connected trade), so plates, thinning and vision always
require the champion at the enemy tower.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Tuple

from .config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, CHAMP_ABSENT, CHAMP_BEHAVIORS, CONTEST, NONE,
                     PRESSURE, SAFE, SH, VISION, GameConfig, champ_act, champ_behavior, champ_objective,
                     champ_wave)
from .game import collection_efficiency
from .mechanics import TurnResult, siege_eligible, step_turn, wave_actions
from .state import State

A, B = 0, 1
FARM, TRADING, RETREAT, SAFE_MODE = "farm", "trading", "retreat", "safe"
NO_REACH = -10 ** 6                      # contributes to no fight


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
    can stand at the enemy tower by the end of this turn."""
    return siege_eligible(lane, s, cfg) and champ.side(s).pos + cfg.champion_speed >= cfg.tower


def champion_legal_actions(lane: State, champ: ChampState, s: int, cfg: GameConfig) -> Tuple[int, ...]:
    """Legal champion-mode action codes of side s, sorted (see the module docstring)."""
    if champ.side(s).absent > 0:
        raise NotImplementedError("absence (recall, death) is introduced in M2b")
    waves = wave_actions(lane, s, cfg)
    acts = [champ_act(w, b) for b in CHAMP_BEHAVIORS for w in waves]
    if objective_reachable(lane, champ, s, cfg):
        acts += [champ_act(w, b, o) for o in (PRESSURE, VISION) for b in (B_HOLD, B_TRADE) for w in waves]
    return tuple(sorted(acts))


# ---------------------------------------------------------------------- economics and risk
def champion_collection(x: int, mode: str, pos: int, opp_mode: str, opp_pos: int, cfg: GameConfig) -> float:
    """Share of a farm opportunity at own-perspective cell x that a champion in `mode` at `pos` collects
    against an opponent in `opp_mode` at its own cell `opp_pos`."""
    if abs(x - pos) > cfg.farm_reach:
        return 0.0
    posture = CONTEST if mode in (FARM, TRADING) else SAFE
    zones = opp_mode in (FARM, TRADING) and abs(-x - opp_pos) <= cfg.farm_reach
    c = collection_efficiency(x, posture, CONTEST if zones else SAFE, cfg)
    return c * cfg.trade_farm_factor if mode == TRADING else c


def champion_gank_probability(lane: State, s: int, wave: int, behavior: int, cfg: GameConfig, exposure: int) -> float:
    """p = clip(b0 + b_pos (e + T) / 2T + b_beh[behavior] + b_shove [Shove, non-Safe] - b_str dsigma, 0, p_max)
    times (1 - rho) while side s's vision is active (timer at the START of the turn), as in game.gank_probability."""
    own_h = lane.ah if s == A else lane.bh
    opp_h = lane.bh if s == A else lane.ah
    dsig = cfg.stacks(own_h) - cfg.stacks(opp_h)
    term = cfg.champ_gank_behavior[behavior] + (cfg.champ_gank_shove if wave == SH and behavior != B_SAFE else 0.0)
    p = (cfg.gank_base + cfg.gank_position * (exposure + cfg.tower) / (2 * cfg.tower) + term
         - cfg.gank_strength * dsig)
    p = min(max(p, 0.0), cfg.gank_max)
    if lane.vision_of(s) > 0:
        p *= (1.0 - cfg.vision_reduction)
    return p


# ---------------------------------------------------------------------- one turn
LaneStep = Callable[[State, int, int, int, Optional[Tuple[int, int]]], TurnResult]


@dataclass(frozen=True)
class ChampTurn:
    lane: TurnResult                       # the authoritative lane step (with champion contributions)
    champ: ChampState                      # champion state after the turn
    waves: Tuple[int, int]
    behaviors: Tuple[int, int]
    objectives: Tuple[int, int]            # chosen
    effective_objectives: Tuple[int, int]  # executed by the lane step (suspended by a connected trade)
    start: Tuple[int, int]                 # champion cells at the start of the turn (own coordinates)
    end: Tuple[int, int]                   # ... at the end
    farm_spot: Tuple[int, int]
    threatened: Tuple[bool, bool]          # YIELD or SAFE champion threatened by a TRADE (evaded)
    connected: Tuple[bool, bool]           # TRADE champion in contact
    modes: Tuple[str, str]
    reach: Tuple[int, int]                 # champion contribution line passed to the lane step
    lane_codes: Tuple[int, int]            # legacy codes given to the lane step
    farm_available: Tuple[float, float]
    farm: Tuple[float, float]              # collected
    exposure: Tuple[int, int]
    gank_p: Tuple[float, float]


def _clip(x: int, cfg: GameConfig) -> int:
    return max(-cfg.tower, min(cfg.tower, x))


def _move(c: int, target: int, cfg: GameConfig) -> int:
    step = max(-cfg.champion_speed, min(cfg.champion_speed, target - c))
    return _clip(c + step, cfg)


def in_contact(x: int, y: int, cfg: GameConfig) -> bool:
    """Champions at own cells x and y (each in its own coordinates) can trade: within trade_range, except that
    a champion in its own tower cell can only be reached from inside that cell."""
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
    if a == CHAMP_ABSENT or champ_behavior(a) == B_RECALL:
        raise NotImplementedError("Recall and absence are reserved for M2b")
    return champ_wave(a), champ_behavior(a), champ_objective(a)


def resolve_turn(lane: State, champ: ChampState, k: int, a: int, b: int, cfg: GameConfig,
                 lane_step: Optional[LaneStep] = None) -> ChampTurn:
    """Resolve decision turn k of champion mode from both committed actions (module docstring, steps 1-9).
    Legality is the caller's responsibility (the environment checks `champion_legal_actions`)."""
    if lane_step is None:
        def lane_step(s, kk, x, y, reach):
            return step_turn(s, kk, x, y, cfg, champion_reach=reach)
    T, rng = cfg.tower, cfg.trade_range
    (wa, ba, oa), (wb, bb, ob) = _decode(a), _decode(b)
    w, beh, obj = (wa, wb), (ba, bb), (oa, ob)
    c0 = (champ.a.pos, champ.b.pos)

    # 1. wave destinations under full champion support (legacy Contest rule)
    probe = lane_step(lane, k, wa + 6 * oa, wb + 6 * ob, None)
    spot = []
    for s in (A, B):
        fx, fh = probe.state.side(s)[:2]
        spot.append(_clip(fx, cfg) if fh > 0 else c0[s])

    # 2. targets and bounded movement: non-traders first, then traders chase their opponent's reference cell
    end = [0, 0]
    chasers = []
    for s in (A, B):
        c = c0[s]
        if obj[s] != NONE:
            target = T
        elif beh[s] == B_TRADE:
            chasers.append(s)
            continue
        elif beh[s] == B_SAFE:
            target = min(spot[s], cfg.safe_line)
        else:
            target = spot[s]
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

    # 3. evasion by YIELD and SAFE (the trader's end cell depends on start positions only)
    threatened = [False, False]
    for s in (A, B):
        o = 1 - s
        if beh[s] in (B_YIELD, B_SAFE) and beh[o] == B_TRADE and in_contact(end[s], end[o], cfg):
            threatened[s] = True
            own_target = spot[s] if beh[s] == B_YIELD else min(spot[s], cfg.safe_line)
            end[s] = _move(c0[s], _evasion_cell(own_target, end[o], cfg), cfg)

    # 4. trade contact
    in_range = in_contact(end[0], end[1], cfg)
    connected = tuple(beh[s] == B_TRADE and in_range for s in (A, B))
    eff_obj = tuple(NONE if connected[s] else obj[s] for s in (A, B))

    # 5. modes and wave contribution
    modes, reach = [], []
    for s in (A, B):
        if connected[s]:
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
    exposure = (max(c0[0], end[0]), max(c0[1], end[1]))
    gank = tuple(champion_gank_probability(lane, s, w[s], beh[s], cfg, exposure[s]) for s in (A, B))

    # 9. champion state after the turn
    new = ChampState(champ.a._replace(pos=end[0]), champ.b._replace(pos=end[1]))
    return ChampTurn(res, new, w, beh, obj, eff_obj, c0, (end[0], end[1]), (spot[0], spot[1]),
                     (threatened[0], threatened[1]), connected, (modes[0], modes[1]), (reach[0], reach[1]),
                     codes, (avail[0], avail[1]), (coll[0], coll[1]), exposure, gank)
