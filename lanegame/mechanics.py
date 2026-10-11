"""Deterministic wave mechanics: one agent turn = `ticks_per_turn` movement ticks.

Given the state at the start of decision turn k and both actions, the lane
evolves deterministically (ganks are the only randomness in the game and in
Phase 1 they do not affect the lane). See the revision report, section C, for
the rules in equation form. The order of operations inside a tick is:

  1. movement      (fronts move according to contact / waiting / action rules,
                    rear packets always advance one cell)
  2. merging       (friendly packets within distance 1 merge into the leading
                    packet; a packet that would exceed the HP cap queues behind it)
  3. contact fight (if the two fronts are within distance 1, simultaneous damage;
                    otherwise a defender front next to its own tower fights the crashed mass
                    under that tower, "mass" mode with siege_engage)
  4. crash check   (a front in the enemy tower cell with no enemy front within
                    distance 1 leaves the lane and joins the siege -> crash event)
  5. tower fire    ("mass" mode only: each tower deals tower_siege_damage HP to the
                    crashed mass that was already under it at the start of the tick)

and at the end of the turn:

  6. tower consumption ("packet" mode only: one packet that was already sieging
                        at the start of the turn is removed from each queue)
  7. spawns            (if the next decision tick is a multiple of spawn_interval)
  8. vision countdown  (crash reward choices are applied afterwards by the game)

Economic accounting. Every HP of minions that leaves the lane is recorded
exactly once, with its cause: killed in a fight (by enemy minions, the enemy
champion or the enemy tower, including defenders fighting crashed minions under
their tower), killed by tower fire while sieging ("mass" mode), or deleted by a
cap ("overflow"; the original model's cap rule only). Kills are
recorded for the side that may farm them, keyed by the place the minions died
in that side's own coordinates (TurnResult.kills); the game layer turns them
into farm payoffs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import F, SH, SP, GameConfig, wave_of
from .state import State, make_side

A, B = 0, 1
EVENT_KEYS = ("spawned", "merged", "died", "crashed", "consumed",
              "killed_hp", "tower_hp", "overflow_hp", "crashed_hp", "spawned_hp", "siege_killed_hp",
              "freeze_held", "freeze_broken")


@dataclass
class TurnResult:
    state: State                      # next state before crash-reward choices (vision counted down)
    crash_hp: Tuple[int, int]         # total HP of packets that crashed this turn, per side (0 = none)
    trace: Optional[List[dict]] = field(default=None)
    # per-side event counters (packets and HP), for conservation checks and analysis
    events: Optional[dict] = field(default=None)
    # most advanced own-perspective position of each side's leading packet during the turn
    exposure: Tuple[int, int] = (0, 0)
    # kills[s] = {x: HP of ENEMY minions that died at own-perspective position x of side s}
    kills: Tuple[Dict[int, int], Dict[int, int]] = field(default_factory=lambda: ({}, {}))


def wave_actions(state: State, s: int, cfg: GameConfig) -> Tuple[int, ...]:
    """Legal wave-management components for side s.

    Slow Push and Shove are always legal. Freeze is legal only when the enemy
    front exists and stands at own-perspective position <= freeze_legal_max
    (default -1: enemy minions are in front of or inside our tower; the original
    model used 0, i.e. anywhere from the river inward).
    """
    efx, efh = (state.bx, state.bh) if s == A else (state.ax, state.ah)
    if efh > 0 and -efx <= cfg.freeze_legal_max:
        return (SP, SH, F)
    return (SP, SH)


def legal_actions(state: State, s: int, cfg: GameConfig) -> Tuple[int, ...]:
    """Legal combined actions (wave + 3 * posture) for side s, sorted."""
    w = wave_actions(state, s, cfg)
    if not cfg.positioning:
        return w
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
        return True                         # walk to the river at full speed
    mode = cfg.slowpush_mode
    if mode == "half":
        return t == 0                       # one cell per turn
    # one cell per slowpush_creep_period spawn intervals
    creep_tick = t == 0 and k % (cfg.spawn_every_turns * cfg.slowpush_creep_period) == 0
    if mode == "creep":
        return creep_tick
    if mode == "stack_creep":
        return creep_tick and h > cfg.wave_hp   # build at the river, creep once stacked
    return False                            # "hold": park at the river


def _add_kill(kills, collector: int, x: int, hp: int) -> None:
    if hp > 0:
        kills[collector][x] = kills[collector].get(x, 0) + hp


def _tick(P, siege, acts, t, crash_hp, cfg: GameConfig, trace=None, ev=None, k=0, reach=None,
          kills=None):
    contact = _fronts_in_contact(P)
    T = cfg.tower
    waves = (wave_of(acts[A]), wave_of(acts[B]))
    # How far forward (own coordinates) each champion acts on its wave: Contest follows
    # the wave everywhere; Safe only reaches fights at or behind its safe line.
    reach_line = tuple(T if cfg.posture_cap[a] is None else cfg.posture_cap[a] for a in acts)
    block = cfg.cap_rule == "block"

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

    # ---- 4. crash check ----
    siege_tick_start = list(siege)
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

    # ---- 5. tower fire on crashed mass ("mass" mode) ----
    if cfg.tower_mode == "mass":
        for s in (A, B):
            d = min(cfg.tower_siege_damage, siege_tick_start[s], siege[s])
            if d > 0:
                siege[s] -= d
                ev["tower_hp"][s] += d
                _add_kill(kills, 1 - s, -T, d)      # dies under the defender's tower

    if trace is not None:
        trace.append({
            "tick": t,
            "A": [list(p) for p in P[A]], "B": [list(p) for p in P[B]],
            "siege": list(siege), "fight": fight, "crashed": crashed, "siege_fight": siege_fight,
        })


def step_turn(state: State, k: int, act_a: int, act_b: int, cfg: GameConfig,
              record: bool = False) -> TurnResult:
    """Advance the lane through decision turn k (ticks 2k+1 .. 2k+2)."""
    P = [[], []]
    for s in (A, B):
        fx, fh, rears, sg, _ = state.side(s)
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

    # Contest exposure: the champion travels with its leading wave and stays under the
    # enemy tower while its crashed minions are still there (the game caps it for Safe).
    reach = [cfg.tower if siege[s] > 0 else (P[s][0][0] if P[s] else -cfg.tower) for s in (A, B)]
    for t in range(cfg.ticks_per_turn):
        _tick(P, siege, acts, t, crash_hp, cfg, trace, ev, k, reach, kills)

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

    sides = []
    for s in (A, B):
        for p in P[s][1:]:
            assert p[1] == cfg.wave_hp, "rear packets must be fresh waves"
        sides.append(make_side(P[s], siege[s]))
    nxt = State(*sides[A], *sides[B],
                max(0, state.va - 1), max(0, state.vb - 1))
    exposure = tuple(min(cfg.tower, max(-cfg.tower, r)) for r in reach)
    return TurnResult(nxt, (crash_hp[A], crash_hp[B]), trace, ev, exposure, kills)


def initial_state(cfg: GameConfig) -> State:
    """Decision turn 0: one fresh wave has just spawned at each base."""
    w = cfg.wave_hp
    return State(-cfg.base, w, (), 0, -cfg.base, w, (), 0, 0, 0)


def total_packets(state: State) -> Tuple[int, int]:
    """Number of physical packets per side (fronts + rears + sieging; "packet" tower mode)."""
    out = []
    for s in (A, B):
        fx, fh, rears, sg, _ = state.side(s)
        out.append((1 if fh > 0 else 0) + len(rears) + sg)
    return out[0], out[1]


def lane_packets(state: State) -> Tuple[int, int]:
    """Number of packets still walking or fighting in the lane (not crashed), per side."""
    return tuple((1 if state.side(s)[1] > 0 else 0) + len(state.side(s)[2]) for s in (A, B))


def total_hp(state: State, cfg: GameConfig) -> Tuple[int, int]:
    """Minion HP alive per side: front + rears + crashed mass under the enemy tower ("mass" mode)."""
    out = []
    for s in (A, B):
        fx, fh, rears, sg, _ = state.side(s)
        out.append(fh + cfg.wave_hp * len(rears) + (sg if cfg.tower_mode == "mass" else 0))
    return out[0], out[1]
