"""Deterministic wave mechanics: one agent turn = `ticks_per_turn` movement ticks.

Given the state at the start of decision turn k and both actions, the lane
evolves deterministically (ganks are the only randomness in the game and in
Phase 1 they do not affect the lane). See the revision report, section C, and
the siege study, section C, for the rules in equation form. The order of
operations inside a tick is:

  1. movement      (fronts move according to contact / waiting / action rules,
                    rear packets always advance one cell)
  2. merging       (friendly packets within distance 1 merge into the leading
                    packet; a packet that would exceed the HP cap queues behind it)
  3. contact fight (if the two fronts are within distance 1, simultaneous damage;
                    otherwise a defender front next to its own tower fights the crashed mass
                    under that tower, "mass" mode with siege_engage)
  3c. thinning     (siege objective Apply Pressure: the attacking champion removes up to
                    thin_damage HP per turn from the defender's leading packet standing
                    adjacent to or inside its tower cell, while the tower is still occupied
                    by the attacker's crashed mass and no lane fight involves that packet)
  4. crash check   (a front in the enemy tower cell with no enemy front within
                    distance 1 leaves the lane and joins the siege -> crash event)
  5. tower fire    ("mass" mode only: each tower deals tower_siege_damage HP to the
                    crashed mass that was already under it at the start of the tick)

and at the end of the turn:

  6. tower consumption ("packet" mode only: one packet that was already sieging
                        at the start of the turn is removed from each queue)
  7. spawns            (if the next decision tick is a multiple of spawn_interval)
  8. plates            (Apply Pressure claims plates_per_turn of the plates available at the
                        START of the turn; a crash during the turn opens plate_grant[size]
                        new opportunities ("crash" plate mode), and when the crashed mass is
                        gone at the end of the turn the unclaimed opportunities lapse)
  9. vision            (Establish Vision sets the timer to vision_turns, protecting the
                        NEXT vision_turns turns, never this one; otherwise it counts down.
                        In "choice" mode the crash-reward choice sets it afterwards.)

Plate modes: "crash" (opportunities per crash, lapse with the siege), "tower" (a finite budget per tower) and
"durability" (M4 ruleset only, design §11): the lane step grants and claims no plate opportunities (ap / bp stay 0);
plates are paid by the M4 resolver from the persistent tower-damage counters, fed by TurnResult.tower_hits.

Economic accounting. Every HP of minions that leaves the lane is recorded
exactly once, with its cause: killed in a fight (by enemy minions, the enemy
champion or the enemy tower, including defenders fighting crashed minions under
their tower), thinned by a pressuring champion, killed by tower fire while
sieging ("mass" mode), or deleted by a cap ("overflow"; the original model's cap
rule only). Kills are recorded for the side that may farm them, keyed by the
place the minions died in that side's own coordinates (TurnResult.kills); the
game layer turns them into farm payoffs. Plates are recorded separately
(TurnResult.plates); they are tower income, not minion farm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import (CONTEST, F, NONE, PRESSURE, SH, SP, VISION, GameConfig, base_of, objective_of,
                     posture_of, wave_of)
from .state import State, make_side

A, B = 0, 1
EVENT_KEYS = ("spawned", "merged", "died", "crashed", "consumed",
              "killed_hp", "tower_hp", "overflow_hp", "crashed_hp", "spawned_hp", "siege_killed_hp",
              "thinned_hp", "plates",
              "freeze_held", "freeze_broken")


@dataclass
class TurnResult:
    state: State                      # next state (vision counted down / set; plates deducted)
    crash_hp: Tuple[int, int]         # total HP of packets that crashed this turn, per side (0 = none)
    trace: Optional[List[dict]] = field(default=None)
    # per-side event counters (packets and HP), for conservation checks and analysis
    events: Optional[dict] = field(default=None)
    # most advanced own-perspective position of each side's leading packet during the turn
    exposure: Tuple[int, int] = (0, 0)
    # kills[s] = {x: HP of ENEMY minions that died at own-perspective position x of side s}
    kills: Tuple[Dict[int, int], Dict[int, int]] = field(default_factory=lambda: ({}, {}))
    plates: Tuple[int, int] = (0, 0)  # plates claimed this turn, per side
    thinned: Tuple[int, int] = (0, 0)  # enemy HP thinned this turn, per side (the thinner)
    vision_set: Tuple[bool, bool] = (False, False)   # vision established this turn, per side
    # minion damage dealt to the ENEMY tower this turn, per side, in minion-damage units: on every tick the crashed
    # mass already under the enemy tower at the start of the tick hits it for minion_damage(mass) (M4 design §11.2;
    # "mass" tower mode only, 0 otherwise). Computed in every configuration, read only by the M4 ruleset.
    tower_hits: Tuple[int, int] = (0, 0)


def wave_actions(state: State, s: int, cfg: GameConfig) -> Tuple[int, ...]:
    """Legal wave-management components for side s.

    Slow Push and Shove are always legal. Freeze is legal only when the enemy
    front exists and stands at own-perspective position <= freeze_legal_max
    (default -1: enemy minions are in front of or inside our tower; the original
    model used 0, i.e. anywhere from the lane inward).
    """
    efx, efh = (state.bx, state.bh) if s == A else (state.ax, state.ah)
    if efh > 0 and -efx <= cfg.freeze_legal_max:
        return (SP, SH, F)
    return (SP, SH)


def siege_eligible(state: State, s: int, cfg: GameConfig) -> bool:
    """Side s may choose a siege objective this turn: its crashed minions are under the enemy tower."""
    return cfg.objectives and state.siege_of(s) > 0


def legal_actions(state: State, s: int, cfg: GameConfig) -> Tuple[int, ...]:
    """Legal combined actions (wave + 3 * posture + 6 * objective) for side s, sorted.

    Without an eligible siege the objective is NONE. With one, every Contest action carries an
    objective (Apply Pressure or Establish Vision) and Safe actions carry none (disengage).
    Contest without an objective is excluded because Apply Pressure weakly dominates it: same
    exposure, same zoning, plus plates and thinning.
    """
    w = wave_actions(state, s, cfg)
    if not cfg.positioning:
        return w
    if siege_eligible(state, s, cfg):
        return tuple(sorted([x + 6 * o for o in (PRESSURE, VISION) for x in w] + [x + 3 for x in w]))
    return tuple(sorted(x + 3 * p for p in (0, 1) for x in w))


def minion_damage(h: int, cfg: GameConfig) -> int:
    """Per-tick damage dealt by a packet's minions.

    lanchester:      max(1, floor(h / W)) - each complete wave adds one unit of firepower
    lanchester_ceil: ceil(h / W)          - any partial wave counts as a whole one
    linear:          1                    - size buys staying power but not firepower
    """
    if cfg.damage_law == "lanchester":
        return max(1, h // cfg.wave_hp)
    if cfg.damage_law == "lanchester_ceil":
        return -(-h // cfg.wave_hp)
    return 1


def _fronts_in_contact(P) -> bool:
    return bool(P[A]) and bool(P[B]) and (P[A][0][0] + P[B][0][0] >= -1)


def _merge(packets: List[List[int]], cap: int, ev=None, side=0, block: bool = False) -> List[List[int]]:
    """Merge friendly packets within distance 1 into the packet ahead of them.

    Distance 1 is the same reach at which enemy fronts engage, so a reinforcement
    joins its own wave exactly when an enemy wave would start fighting it.

    block=False ("overflow" cap rule, original model): HP beyond the cap is lost
    (recorded as overflow; nobody can farm it).
    block=True ("block" cap rule): only the leading packet absorbs packets, and only
    while the result stays within the cap; a reinforcement that does not fit stays a
    separate packet queued behind it, so no minion ever disappears.
    """
    packets.sort(key=lambda p: -p[0])
    out: List[List[int]] = []
    for p in packets:
        if block:
            if out and out[0][0] - p[0] <= 1 and out[0][1] + p[1] <= cap:
                out[0][1] += p[1]
                if ev is not None:
                    ev["merged"][side] += 1
            else:
                out.append([p[0], p[1]])
        elif out and out[-1][0] - p[0] <= 1:
            total = out[-1][1] + p[1]
            out[-1][1] = min(cap, total)
            if ev is not None:
                ev["merged"][side] += 1
                ev["overflow_hp"][side] += total - out[-1][1]
        else:
            out.append([p[0], p[1]])
    return out


def _blob_hp(packets, cfg: GameConfig) -> int:
    """HP of the leading packet plus packets queued within one cell behind it ("block" cap rule).

    Queued packets stand directly behind the leading one, so they fight and push with it;
    under the "overflow" rule no packet is ever within one cell of the front (it would have merged).
    """
    if not packets:
        return 0
    if not cfg.queue_fights:
        return packets[0][1]
    fx = packets[0][0]
    return packets[0][1] + sum(p[1] for p in packets[1:] if p[0] >= fx - 1)


def _sp_moves(x: int, h: int, t: int, k: int, cfg: GameConfig) -> bool:
    """Does a free Slow Push front at own position x with HP h advance on tick t of turn k?"""
    if x < cfg.slowpush_line or cfg.slowpush_mode == "free":
        return True                         # walk to the lane at full speed
    mode = cfg.slowpush_mode
    if mode == "half":
        return t == 0                       # one cell per turn
    # one cell per slowpush_creep_period spawn intervals
    creep_tick = t == 0 and k % (cfg.spawn_every_turns * cfg.slowpush_creep_period) == 0
    if mode == "creep":
        return creep_tick
    if mode == "stack_creep":
        return creep_tick and h > cfg.wave_hp   # build at the lane, creep once stacked
    return False                            # "hold": park at the lane


def _add_kill(kills, collector: int, x: int, hp: int) -> None:
    if hp > 0:
        kills[collector][x] = kills[collector].get(x, 0) + hp


def thinning_target(P, s: int, cfg: GameConfig) -> Optional[List[int]]:
    """The enemy packet an Apply Pressure champion of side s can reach from the enemy tower cell:
    the enemy's leading packet if it stands within thin_reach cells behind that tower or inside
    it (enemy own x in [-T - thin_reach, -T]). None if no enemy packet is accessible."""
    e = 1 - s
    if not P[e]:
        return None
    fe = P[e][0]
    T = cfg.tower
    if -T - cfg.thin_reach <= fe[0] <= -T:
        return fe
    return None


def _tick(P, siege, acts, t, crash_hp, cfg: GameConfig, trace=None, ev=None, k=0, reach=None,
          kills=None, thin_left=None, plates=None, champion_reach=None, hits=None):
    contact = _fronts_in_contact(P)
    T = cfg.tower
    bases = (base_of(acts[A]), base_of(acts[B]))
    waves = (wave_of(acts[A]), wave_of(acts[B]))
    objectives = (objective_of(acts[A]), objective_of(acts[B]))
    # How far forward (own coordinates) each champion acts on its wave: Contest follows
    # the wave everywhere; Safe only reaches fights at or behind its safe line.
    reach_line = tuple(T if cfg.posture_cap[a] is None else cfg.posture_cap[a] for a in bases)
    if champion_reach is not None:         # champion mode: the champions' actual reach (lanegame.champions)
        reach_line = tuple(champion_reach)
    block = cfg.cap_rule == "block"
    siege_tick_start = list(siege)
    if hits is not None and cfg.tower_mode == "mass":
        for s in (A, B):
            if siege_tick_start[s] > 0:
                hits[s] += minion_damage(siege_tick_start[s], cfg)

    # ---- 1. movement ----
    for s in (A, B):
        e = 1 - s
        packets = P[s]
        for i, pk in enumerate(packets):
            if i > 0:                      # rear packets always walk ...
                if not block or pk[0] + 1 < packets[0][0]:
                    pk[0] += 1             # ... but never into the cell of a front they cannot join
                continue
            x, h = pk
            if contact:
                continue
            if siege[e] > 0 and x >= -T - 1:
                continue                   # waiting behind own tower while it consumes attackers
            w = waves[s]
            if w == SH:
                want = True
            elif w == SP:
                want = _sp_moves(x, h, t, k, cfg)
            else:
                want = x < cfg.freeze_line
            if want and waves[e] == F and x + 1 > -cfg.freeze_line:
                # enemy freeze holds our wave in front of the enemy tower unless broken;
                # the shove push needs our champion at the wave
                push = cfg.freeze_break_push[w] if x <= reach_line[s] else 0
                if _blob_hp(packets, cfg) + push < cfg.freeze_break_threshold:
                    want = False
                    ev["freeze_held"][s] += 1
                else:
                    ev["freeze_broken"][s] += 1
            if want:
                pk[0] += 1

    # ---- 2. merging ----
    for s in (A, B):
        P[s] = _merge(P[s], cfg.hp_cap, ev, s, block)
        if reach is not None and P[s]:
            reach[s] = max(reach[s], P[s][0][0])

    # ---- 3. contact fight ----
    fight = None
    if _fronts_in_contact(P):
        fa, fb = P[A][0], P[B][0]
        if cfg.damage_law == "pressure":
            pa = fa[1] + cfg.pressure_kappa[waves[A]]
            pb = fb[1] + cfg.pressure_kappa[waves[B]]
            dmg_b = 1 if pa >= pb else 0
            dmg_a = 1 if pb >= pa else 0
        else:
            dmg_b = minion_damage(_blob_hp(P[A], cfg), cfg) + (cfg.shove_damage if waves[A] == SH and fa[0] <= reach_line[A] else 0)
            dmg_a = minion_damage(_blob_hp(P[B], cfg), cfg) + (cfg.shove_damage if waves[B] == SH and fb[0] <= reach_line[B] else 0)
        if fb[0] == T:                     # B front contesting inside A's tower cell
            dmg_b += cfg.tower_damage
        if fa[0] == T:                     # A front contesting inside B's tower cell
            dmg_a += cfg.tower_damage
        fight = (fa[0], fa[1], fb[0], fb[1], dmg_a, dmg_b)
        lost_a, lost_b = min(dmg_a, fa[1]), min(dmg_b, fb[1])
        # minions die where they stand; the opponent may farm them (own coordinates = -x)
        _add_kill(kills, B, -fa[0], lost_a)
        _add_kill(kills, A, -fb[0], lost_b)
        ev["killed_hp"][A] += lost_a
        ev["killed_hp"][B] += lost_b
        fa[1] -= lost_a
        fb[1] -= lost_b
        if fa[1] <= 0:
            P[A].pop(0)
            ev["died"][A] += 1
        if fb[1] <= 0:
            P[B].pop(0)
            ev["died"][B] += 1

    # ---- 3b. defenders' minions engage crashed minions under their own tower ----
    # A defender front waiting next to (or in) its tower cell fights the attacker's crashed mass,
    # unless it already fought the attacker's lane front this tick. Both sides lose HP
    # (Lanchester damage, plus the defender champion's shove if present); every HP lost is a farm
    # opportunity for the other side, at the place it died.
    siege_fight = None
    if cfg.tower_mode == "mass" and cfg.siege_engage and fight is None:
        for s in (A, B):
            e = 1 - s
            if siege[s] <= 0 or not P[e] or P[e][0][0] < -T - 1:
                continue
            fe = P[e][0]
            d_sg = minion_damage(_blob_hp(P[e], cfg), cfg) + (cfg.shove_damage if waves[e] == SH and fe[0] <= reach_line[e] else 0)
            d_e = minion_damage(siege[s], cfg)
            lost_sg, lost_e = min(d_sg, siege[s]), min(d_e, fe[1])
            siege[s] -= lost_sg
            fe[1] -= lost_e
            _add_kill(kills, e, -T, lost_sg)          # crashed minions die under e's tower
            _add_kill(kills, s, -fe[0], lost_e)        # e's minions die where they stand
            ev["killed_hp"][s] += lost_sg
            ev["siege_killed_hp"][s] += lost_sg
            ev["killed_hp"][e] += lost_e
            siege_fight = (s, lost_sg, lost_e)
            if fe[1] <= 0:
                P[e].pop(0)
                ev["died"][e] += 1

    # ---- 3c. thinning (Apply Pressure) ----
    # The pressuring champion stands in the enemy tower cell while the tower is busy with its
    # crashed minions (tower aggro); it can hit the defender's leading packet if that packet is
    # accessible (adjacent to or inside the tower cell) and did not fight in the lane this tick.
    thinned = [0, 0]
    if cfg.objectives and fight is None and thin_left is not None:
        for s in (A, B):
            if objectives[s] != PRESSURE or thin_left[s] <= 0 or siege_tick_start[s] <= 0:
                continue
            target = thinning_target(P, s, cfg)
            if target is None:
                continue
            e = 1 - s
            d = min(thin_left[s], target[1])
            thin_left[s] -= d
            target[1] -= d
            thinned[s] = d
            _add_kill(kills, s, -target[0], d)          # dies where it stands, in s's own coordinates
            ev["killed_hp"][e] += d
            ev["thinned_hp"][e] += d
            if target[1] <= 0:
                P[e].pop(0)
                ev["died"][e] += 1

    # ---- 4. crash check ----
    siege_before_crash = list(siege)
    crashed = [0, 0]
    for s in (A, B):
        if P[s] and P[s][0][0] == T and not _fronts_in_contact(P):
            crashed[s] = P[s][0][1]
    for s in (A, B):
        if crashed[s]:
            P[s].pop(0)
            crash_hp[s] += crashed[s]
            ev["crashed"][s] += 1
            ev["crashed_hp"][s] += crashed[s]
            if cfg.tower_mode == "packet":
                siege[s] += 1
                # the queue does not store HP: the defender's farm opportunity on these
                # minions is credited when they enter the tower (each HP exactly once)
                _add_kill(kills, 1 - s, -T, crashed[s])
            else:
                total = siege[s] + crashed[s]
                siege[s] = min(cfg.siege_cap, total)
                ev["overflow_hp"][s] += total - siege[s]
                if plates is not None and cfg.objectives and cfg.plate_mode == "crash":
                    # the crash opens plate opportunities, claimable while this siege lasts
                    plates[s] = min(cfg.plates_cap, plates[s] + cfg.plate_grant[cfg.size_category(crashed[s])])

    # ---- 5. tower fire on crashed mass ("mass" mode) ----
    if cfg.tower_mode == "mass":
        for s in (A, B):
            d = min(cfg.tower_siege_damage, siege_before_crash[s], siege[s])
            if d > 0:
                siege[s] -= d
                ev["tower_hp"][s] += d
                _add_kill(kills, 1 - s, -T, d)      # dies under the defender's tower

    if trace is not None:
        trace.append({
            "tick": t,
            "A": [list(p) for p in P[A]], "B": [list(p) for p in P[B]],
            "siege": list(siege), "fight": fight, "crashed": crashed, "siege_fight": siege_fight,
            "thinned": thinned,
        })


def step_turn(state: State, k: int, act_a: int, act_b: int, cfg: GameConfig,
              record: bool = False, champion_reach: Optional[Tuple[int, int]] = None) -> TurnResult:
    """Advance the lane through decision turn k (ticks 2k+1 .. 2k+2).

    `champion_reach` (champion mode only; None = the legacy rule) overrides, per side, the furthest
    own-perspective position at which the champion acts on its leading wave (shove damage, the push that
    breaks a freeze, the shove of a siege-engaging defender). Posture then plays no role in the lane step."""
    P = [[], []]
    for s in (A, B):
        fx, fh, rears, sg, _, _ = state.side(s)
        if fh > 0:
            P[s].append([fx, fh])
        P[s].extend([x, cfg.wave_hp] for x in rears)
    siege = [state.asg, state.bsg]
    siege_start = list(siege)
    crash_hp = [0, 0]
    acts = (act_a, act_b)
    trace = [] if record else None
    ev = {key: [0, 0] for key in EVENT_KEYS}
    kills = ({}, {})
    thin_left = [cfg.thin_damage if objective_of(a) == PRESSURE else 0 for a in acts]
    plates = [state.ap, state.bp]
    claimed = [0, 0]
    for s in (A, B):
        # Apply Pressure claims plates at the start of the turn (the siege exists by legality), before
        # any plate opportunity opened by a crash during this turn
        if objective_of(acts[s]) == PRESSURE:
            assert siege[s] > 0, "Apply Pressure requires an eligible siege"
            if cfg.plate_mode == "durability":
                continue                          # M4: no plate opportunities in the lane (design §11.4)
            claimed[s] = min(cfg.plates_per_turn, plates[s])
            plates[s] -= claimed[s]
            ev["plates"][s] += claimed[s]

    # Contest exposure: the champion travels with its leading wave and stays under the
    # enemy tower while its crashed minions are still there (the game caps it for Safe).
    reach = [cfg.tower if siege[s] > 0 else (P[s][0][0] if P[s] else -cfg.tower) for s in (A, B)]
    hits = [0, 0]
    for t in range(cfg.ticks_per_turn):
        _tick(P, siege, acts, t, crash_hp, cfg, trace, ev, k, reach, kills, thin_left, plates, champion_reach, hits)

    if cfg.tower_mode == "packet":
        for s in (A, B):
            c = min(cfg.tower_consumption_per_turn, siege_start[s])
            siege[s] -= c
            ev["consumed"][s] += c

    next_tick = (k + 1) * cfg.ticks_per_turn
    if next_tick % cfg.spawn_interval == 0:
        for s in (A, B):
            P[s].append([-cfg.base, cfg.wave_hp])
            ev["spawned"][s] += 1
            ev["spawned_hp"][s] += cfg.wave_hp
            P[s] = _merge(P[s], cfg.hp_cap, ev, s, cfg.cap_rule == "block")

    # ---- plates and vision (siege objectives) ----
    vision = [state.va, state.vb]
    vision_set = [False, False]
    for s in (A, B):
        o = objective_of(acts[s])
        if cfg.objectives and cfg.plate_mode == "crash" and siege[s] == 0:
            plates[s] = 0                         # the siege is over: unclaimed opportunities lapse
        if o == VISION:
            assert siege_start[s] > 0, "Establish Vision requires an eligible siege"
            vision[s] = cfg.vision_turns          # protects the next vision_turns turns (refresh, no stacking)
            vision_set[s] = True
        else:
            vision[s] = max(0, vision[s] - 1)

    sides = []
    for s in (A, B):
        for p in P[s][1:]:
            assert p[1] == cfg.wave_hp, "rear packets must be fresh waves"
        sides.append(make_side(P[s], siege[s]))
    nxt = State(*sides[A], *sides[B], vision[A], vision[B], plates[A], plates[B])
    exposure = tuple(min(cfg.tower, max(-cfg.tower, r)) for r in reach)
    thinned = (ev["thinned_hp"][B], ev["thinned_hp"][A])      # thinned[s] = enemy HP removed by s
    return TurnResult(nxt, (crash_hp[A], crash_hp[B]), trace, ev, exposure, kills,
                      (claimed[A], claimed[B]), thinned, (vision_set[A], vision_set[B]), (hits[A], hits[B]))


def initial_state(cfg: GameConfig) -> State:
    """Decision turn 0: one fresh wave has just spawned at each base; no siege, no vision, no plate
    opportunities ("crash" plate mode) or the full plate budget ("tower" mode)."""
    w = cfg.wave_hp
    pl = cfg.plates_per_tower if (cfg.objectives and cfg.plate_mode == "tower") else 0
    return State(-cfg.base, w, (), 0, -cfg.base, w, (), 0, 0, 0, pl, pl)


def total_packets(state: State) -> Tuple[int, int]:
    """Number of physical packets per side (fronts + rears + sieging; "packet" tower mode)."""
    out = []
    for s in (A, B):
        fx, fh, rears, sg, _, _ = state.side(s)
        out.append((1 if fh > 0 else 0) + len(rears) + sg)
    return out[0], out[1]


def lane_packets(state: State) -> Tuple[int, int]:
    """Number of packets still walking or fighting in the lane (not crashed), per side."""
    return tuple((1 if state.side(s)[1] > 0 else 0) + len(state.side(s)[2]) for s in (A, B))


def total_hp(state: State, cfg: GameConfig) -> Tuple[int, int]:
    """Minion HP alive per side: front + rears + crashed mass under the enemy tower ("mass" mode)."""
    out = []
    for s in (A, B):
        fx, fh, rears, sg, _, _ = state.side(s)
        out.append(fh + cfg.wave_hp * len(rears) + (sg if cfg.tower_mode == "mass" else 0))
    return out[0], out[1]
