"""Deterministic wave mechanics: one agent turn = `ticks_per_turn` movement ticks.

Given the state at the start of decision turn k and both actions, the lane
evolves deterministically (ganks are the only randomness in the game and in
Phase 1 they do not affect the lane). See report section C for the rules in
equation form; the order of operations inside a tick is:

  1. movement      (fronts move according to contact / waiting / action rules,
                    rear packets always advance one cell)
  2. merging       (friendly packets that share a cell merge, HP capped)
  3. contact fight (if the two fronts are within distance 1, simultaneous damage)
  4. crash check   (a front in the enemy tower cell with no enemy front within
                    distance 1 becomes a sieging packet -> crash event)

and at the end of the turn:

  5. tower consumption (one packet that was already sieging at the start of
                        the turn is removed from each siege queue)
  6. spawns            (if the next decision tick is a multiple of spawn_interval)
  7. vision countdown  (crash reward choices are applied afterwards by the game)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config import F, SH, SP, GameConfig
from .state import State, make_side

A, B = 0, 1


@dataclass
class TurnResult:
    state: State                      # next state before crash-reward choices (vision counted down)
    crash_hp: Tuple[int, int]         # total HP of packets that crashed this turn, per side (0 = none)
    trace: Optional[List[dict]] = field(default=None)
    # physical-packet event counts per side, for conservation checks
    events: Optional[dict] = field(default=None)
    # most advanced own-perspective position of each side's leading packet during the turn
    exposure: Tuple[int, int] = (0, 0)


def legal_actions(state: State, s: int, cfg: GameConfig) -> Tuple[int, ...]:
    """Legal strategic actions for side s.

    Slow Push and Shove are always legal. Freeze is legal only when the enemy
    front exists and stands at own-perspective position <= freeze_legal_max
    (i.e. enemy minions are at the river or on our half of the lane).
    """
    e = 1 - s
    efx, efh = (state.bx, state.bh) if e == B else (state.ax, state.ah)
    if efh > 0 and -efx <= cfg.freeze_legal_max:
        return (SP, SH, F)
    return (SP, SH)


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


def _merge(packets: List[List[int]], cap: int, counter=None, side=0) -> List[List[int]]:
    """Merge friendly packets within distance 1 into the packet ahead of them.

    Distance 1 is the same reach at which enemy fronts engage, so a reinforcement
    joins its own wave exactly when an enemy wave would start fighting it.
    """
    packets.sort(key=lambda p: -p[0])
    out: List[List[int]] = []
    for p in packets:
        if out and out[-1][0] - p[0] <= 1:
            out[-1][1] = min(cap, out[-1][1] + p[1])
            if counter is not None:
                counter["merged"][side] += 1
        else:
            out.append([p[0], p[1]])
    return out


def _sp_moves(x: int, t: int, k: int, cfg: GameConfig) -> bool:
    """Does a free Slow Push front at own position x advance on tick t of turn k?"""
    if x < cfg.slowpush_line:
        return True                         # walk to the river at full speed
    if cfg.slowpush_mode == "half":
        return t == 0                       # one cell per turn
    if cfg.slowpush_mode == "creep":
        return t == 0 and k % cfg.spawn_every_turns == 0   # one cell per spawn interval
    return False                            # "hold": park at the river


def _tick(P, siege, acts, t, crash_hp, cfg: GameConfig, trace=None, ev=None, k=0, reach=None):
    contact = _fronts_in_contact(P)

    # ---- 1. movement ----
    for s in (A, B):
        e = 1 - s
        packets = P[s]
        for i, pk in enumerate(packets):
            if i > 0:                      # rear packets always walk
                pk[0] += 1
                continue
            x, h = pk
            if contact:
                continue
            if siege[e] > 0 and x >= -cfg.tower - 1:
                continue                   # waiting behind own tower while it consumes attackers
            a = acts[s]
            if a == SH:
                want = True
            elif a == SP:
                want = _sp_moves(x, t, k, cfg)
            else:
                want = x < cfg.freeze_line
            if want and acts[e] == F and x + 1 > -cfg.freeze_line:
                # enemy freeze holds our wave in front of the enemy tower unless broken
                if h + cfg.freeze_break_push[a] < cfg.freeze_break_threshold:
                    want = False
            if want:
                pk[0] += 1

    # ---- 2. merging ----
    for s in (A, B):
        P[s] = _merge(P[s], cfg.hp_cap, ev, s)
        if reach is not None and P[s]:
            reach[s] = max(reach[s], P[s][0][0])

    # ---- 3. contact fight ----
    fight = None
    if _fronts_in_contact(P):
        fa, fb = P[A][0], P[B][0]
        if cfg.damage_law == "pressure":
            pa = fa[1] + cfg.pressure_kappa[acts[A]]
            pb = fb[1] + cfg.pressure_kappa[acts[B]]
            dmg_b = 1 if pa >= pb else 0
            dmg_a = 1 if pb >= pa else 0
        else:
            dmg_b = minion_damage(fa[1], cfg) + (cfg.shove_damage if acts[A] == SH else 0)
            dmg_a = minion_damage(fb[1], cfg) + (cfg.shove_damage if acts[B] == SH else 0)
        if fb[0] == cfg.tower:             # B front contesting inside A's tower cell
            dmg_b += cfg.tower_damage
        if fa[0] == cfg.tower:             # A front contesting inside B's tower cell
            dmg_a += cfg.tower_damage
        fight = (fa[0], fa[1], fb[0], fb[1], dmg_a, dmg_b)
        fa[1] -= dmg_a
        fb[1] -= dmg_b
        if fa[1] <= 0:
            P[A].pop(0)
            ev["died"][A] += 1
        if fb[1] <= 0:
            P[B].pop(0)
            ev["died"][B] += 1

    # ---- 4. crash check ----
    crashed = [0, 0]
    for s in (A, B):
        if P[s] and P[s][0][0] == cfg.tower and not _fronts_in_contact(P):
            crashed[s] = P[s][0][1]
    for s in (A, B):
        if crashed[s]:
            P[s].pop(0)
            siege[s] += 1
            crash_hp[s] += crashed[s]
            ev["crashed"][s] += 1

    if trace is not None:
        trace.append({
            "tick": t,
            "A": [list(p) for p in P[A]], "B": [list(p) for p in P[B]],
            "siege": list(siege), "fight": fight, "crashed": crashed,
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
    ev = {key: [0, 0] for key in ("spawned", "merged", "died", "crashed", "consumed")}

    # Champion exposure: it travels with its leading wave and stays under the enemy
    # tower while its crashed packets are still being consumed there.
    reach = [cfg.tower if siege[s] > 0 else (P[s][0][0] if P[s] else -cfg.tower) for s in (A, B)]
    for t in range(cfg.ticks_per_turn):
        _tick(P, siege, acts, t, crash_hp, cfg, trace, ev, k, reach)

    for s in (A, B):
        c = min(cfg.tower_consumption_per_turn, siege_start[s])
        siege[s] -= c
        ev["consumed"][s] += c

    next_tick = (k + 1) * cfg.ticks_per_turn
    if next_tick % cfg.spawn_interval == 0:
        for s in (A, B):
            P[s].append([-cfg.base, cfg.wave_hp])
            ev["spawned"][s] += 1
            P[s] = _merge(P[s], cfg.hp_cap, ev, s)

    sides = []
    for s in (A, B):
        for p in P[s][1:]:
            assert p[1] == cfg.wave_hp, "rear packets must be fresh waves"
        sides.append(make_side(P[s], siege[s]))
    nxt = State(*sides[A], *sides[B],
                max(0, state.va - 1), max(0, state.vb - 1))
    exposure = tuple(min(cfg.tower, max(-cfg.tower, r)) for r in reach)
    return TurnResult(nxt, (crash_hp[A], crash_hp[B]), trace, ev, exposure)


def initial_state(cfg: GameConfig) -> State:
    """Decision turn 0: one fresh wave has just spawned at each base."""
    w = cfg.wave_hp
    return State(-cfg.base, w, (), 0, -cfg.base, w, (), 0, 0, 0)


def total_packets(state: State) -> Tuple[int, int]:
    """Number of physical packets per side (fronts + rears + sieging)."""
    out = []
    for s in (A, B):
        fx, fh, rears, sg, _ = state.side(s)
        out.append((1 if fh > 0 else 0) + len(rears) + sg)
    return out[0], out[1]
