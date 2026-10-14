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

Natural waves (wave_dynamics = "natural", GameConfig.m4_natural(), docs/natural_waves/m4_2e_natural_waves_report.md and
docs/natural_waves/m4_2e_natural_waves_final_review.md). Minions obey
the same rules whatever the actions; the wave slot (Hold, Clear, Thin) only sets the champion's damage budget
champion_wave_damage[slot] HP per tick. Per tick:

  1. movement      every free front walks one cell; contact with the enemy front stops it, and so does contact with
                   the enemy's crashed mass under its own tower (own x >= -T - 1); rears walk and never enter the cell
                   of a front they cannot join (no parking, no creeping, no Freeze hold, no idle wait behind a tower)
  2. merging       "block" cap rule
  2b. entry crash  natural_crash_rule "entry" (M4.2e-A.1 preset): a front entering the enemy tower cell joins the crashed
                   mass at once (a crash is entering tower range; follow-up waves join the siege), so no front ever
                   fights inside the enemy tower cell and the tower fires on the crashed mass alone; "contact"
                   (M4.2e-A prototype): the crash check of step 4, a front in the tower cell next to a defender fights
  3. contact fight Lanchester minion damage of each blob + the champion's budget on the enemy front if that front's
                   own-perspective cell is within its reach line + the tower on a contested attacker in its cell
  3b. siege fight  a defender front within distance 1 of its tower fights the crashed mass (Lanchester both ways,
                   plus the defender champion's budget if it reaches its tower cell); skipped on ticks with a lane fight
  3c. thinning     Apply Pressure, as in the scripted rule
  3d. champion     an unspent budget strikes the enemy's crashed mass under the own tower, else the enemy's leading
                   packet if reachable: a champion can clear a walking wave or a crash with no minion of its own
  hits             the crashed mass at the start of the tick hits the tower unless it fought defenders this tick
  4.-5.            crash check (prototype) and tower fire on the mass present at the start of the tick

The champion's budget per tick is champion_wave_damage[slot], or (wave_clear_scaling "attack_power" or "capacity") the
turn budget champion_budget[s] computed by the M4 resolver from attack power, split over the ticks with earlier ticks
first. M4.2e-C (GameConfig.m4_lasthit(), docs/natural_waves/m4_2e_wave_actions_report.md): under "capacity" the three
slots are shares of one wave-clear capacity (Hold an eighth, Thin a half, Clear all), and damage_law "lanchester_half"
gives the minions half-wave resolution (`minion_damage`).

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
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .config import (CONTEST, F, NONE, PRESSURE, SH, SP, VISION, W_SLOWPUSH, GameConfig, base_of, objective_of,
                     posture_of, wave_of)
from .positional import WARD, MinionTarget, dist as graph_dist, reaches
from .state import State, make_side

A, B = 0, 1
EVENT_KEYS = ("spawned", "merged", "died", "crashed", "consumed",
              "killed_hp", "tower_hp", "overflow_hp", "crashed_hp", "spawned_hp", "siege_killed_hp",
              "thinned_hp", "plates",
              "freeze_held", "freeze_broken")
# wave_dynamics "natural" only (historical event dicts stay byte-identical): HP of this side's minions killed by the
# ENEMY champion
NATURAL_EVENT_KEYS = ("champion_hp",)


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
    # M4.5a minion targeting (champion_targets given; else (0, 0) / None): minion damage units taken by champion A and
    # B this turn, and per champion the cause of each tick's fire on it ("retaliation", "positional" or "")
    champion_hits: Tuple[int, int] = (0, 0)
    champion_fire: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None


def wave_actions(state: State, s: int, cfg: GameConfig) -> Tuple[int, ...]:
    """Legal wave-management components for side s.

    Slow Push and Shove are always legal. Freeze is legal only when the enemy
    front exists and stands at own-perspective position <= freeze_legal_max
    (default -1: enemy minions are in front of or inside our tower; the original
    model used 0, i.e. anywhere from the lane inward).
    """
    if cfg.wave_dynamics == "natural":
        if cfg.wave_slowpush:              # M4.3c four-action model: + Slow Push (EXPERIMENTAL)
            return (SP, SH, F, W_SLOWPUSH)
        return (SP, SH, F)                 # Hold, Clear, Thin: what the champion does, always legal
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


def minion_damage(h: int, cfg: GameConfig, tick: int = 0) -> int:
    """Per-tick damage dealt by a packet's minions.

    lanchester:      max(1, floor(h / W)) - each complete wave adds one unit of firepower
    lanchester_ceil: ceil(h / W)          - any partial wave counts as a whole one
    lanchester_half: (M4.2e-C) half-wave resolution: a packet's firepower is max(1, floor(2h / W)) HP per TURN,
                     split over the turn's ticks with the earlier ticks first (tick t of a turn deals
                     ceil((t + 1) f / n) - ceil(t f / n), n = ticks_per_turn), recomputed on the HP it has at each
                     tick; so 8 HP deal 1 + 1, 12 HP deal 2 + 1 and 16 HP deal 2 + 2 per turn at W = 8
    linear:          1                    - size buys staying power but not firepower
    `tick` is the tick's index within the turn; only lanchester_half reads it.
    """
    if cfg.damage_law == "lanchester":
        return max(1, h // cfg.wave_hp)
    if cfg.damage_law == "lanchester_ceil":
        return -(-h // cfg.wave_hp)
    if cfg.damage_law == "lanchester_half":
        f, n, t = max(1, (2 * h) // cfg.wave_hp), cfg.ticks_per_turn, tick % cfg.ticks_per_turn
        return -((-(t + 1) * f) // n) + ((-t * f) // n)
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


def thin_need(P, s: int, cfg: GameConfig) -> int:
    """wave_thin_rule "adaptive" (EXPERIMENTAL, docs/m4/m4_3c_wave_control_screen.md): the HP of the enemy blob above
    the largest blob whose minions do not out-fire side s's, on the blobs of the start-of-turn packets P:

        f_own = max(1, floor(2 h_own / W))  (0 with no own packet),   h* = floor((W (f_own + 1) - 1) / 2),
        need  = max(0, h_enemy - h*)

    (h* = the largest h with floor(2h / W) <= f_own under lanchester_half; 11 against an own blob of 8-11, 15 against
    12-15). Positions, reinforcements still walking and both champions' damage are ignored."""
    he, ho = _blob_hp(P[1 - s], cfg), _blob_hp(P[s], cfg)
    fo = max(1, (2 * ho) // cfg.wave_hp) if ho > 0 else 0
    target = (cfg.wave_hp * (fo + 1) - 1) // 2 if fo > 0 else 0
    return max(0, he - target)


# ---------------------------------------------------------------------------------------------------------------------
# M4.3c four-action wave management (EXPERIMENTAL, wave_thin_rule "strength" / wave_slowpush;
# docs/m4/m4_3c_four_action_wave_redesign.md). Thin and Slow Push spend only the damage needed to reach a target
# balance of the two waves' WINDOW STRENGTH, on top of the Last Hit budget, up to their capacity.
# Target margins in HP of window strength: Thin restores parity, Slow Push establishes a lead of one HP.
STRENGTH_MARGIN = {F: 0, W_SLOWPUSH: 1}


def start_packets(state: State, cfg: GameConfig):
    """The packet lists [[x, hp], ...] of both sides (own coordinates, leading packet first) exactly as step_turn
    builds them at the start of a turn."""
    P = [[], []]
    for s in (A, B):
        fx, fh, rears = state.side(s)[:3]
        if fh > 0:
            P[s].append([fx, fh])
        P[s].extend([x, cfg.wave_hp] for x in rears)
    return P


def window_strength(packets, k: int, cfg: GameConfig):
    """Window strength of one side on turn k (exact Fraction, HP): the average minion HP fighting at its front over the
    next reinforcement cycle, tau = spawn_interval ticks (one wave per side), assuming the front holds its cell:

        S = (1 / tau) sum_p hp_p rho_p

    rho_p = ticks of the window in which packet p fights at the front: tau for the leading packet and any packet
    queued within one cell behind it (the blob); max(0, tau - d + 2) for a rear packet d >= 2 cells behind (it walks
    one cell per tick and merges at distance 1, in tick d - 1); and, if a wave spawns at the end of this turn, for that
    wave max(0, min(tau - n, tau - n - d + 2)), n = ticks_per_turn, d = its walk from the base to the front.
    Reinforcements that arrive sooner count more, which makes the measure depend on where the waves meet. A side with
    no packet has strength 0. Crashed masses are not counted (towers and defenders fight them)."""
    if not packets:
        return Fraction(0)
    tau, n = cfg.spawn_interval, cfg.ticks_per_turn
    fx = packets[0][0]
    total = 0
    for x, hp in packets:
        d = fx - x
        total += hp * (tau if d <= 1 else max(0, tau - d + 2))
    if ((k + 1) * n) % cfg.spawn_interval == 0:          # a wave spawns at the own base at the end of this turn
        d = fx + cfg.base
        total += cfg.wave_hp * max(0, min(tau - n, tau - n - d + 2))
    return Fraction(total, tau)


def strength_need(P, s: int, k: int, margin: int, cfg: GameConfig) -> int:
    """HP of enemy minions side s must remove so that the enemy's window strength is at most its own minus `margin`
    (first order: an HP removed now lowers the enemy's strength by one HP):

        need = max(0, ceil(S_enemy - S_own + margin))

    on the start-of-turn packets P of turn k. Both champions' last hits are assumed to cancel; the opponent's current
    action, tower fire and later turns are not predicted."""
    diff = window_strength(P[1 - s], k, cfg) - window_strength(P[s], k, cfg) + margin
    return max(0, -((-diff.numerator) // diff.denominator))


def strength_budget(state: State, k: int, s: int, slot: int, last_hit: int, cap: int, cfg: GameConfig) -> int:
    """The per-turn budget of a Thin (slot F) or Slow Push (slot W_SLOWPUSH) of side s under wave_thin_rule
    "strength": the Last Hit budget plus the need of the slot's margin, never more than the slot's capacity `cap`:
    min(cap, last_hit + strength_need(margin))."""
    return min(cap, last_hit + strength_need(start_packets(state, cfg), s, k, STRENGTH_MARGIN[slot], cfg))


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


def _pressure_thinning(P, siege_tick_start, objectives, no_fight: bool, thin_left, cfg: GameConfig, ev, kills):
    """Step 3c, thinning (Apply Pressure). The pressuring champion stands in the enemy tower cell while the tower is
    busy with its crashed minions (tower aggro); it can hit the defender's leading packet if that packet is accessible
    (adjacent to or inside the tower cell) and did not fight in the lane this tick (`no_fight`)."""
    thinned = [0, 0]
    if cfg.objectives and no_fight and thin_left is not None:
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
    return thinned


def _crash_check(P, siege, crash_hp, cfg: GameConfig, ev, kills, plates, entry: bool = False):
    """Step 4, crash check: a front in the enemy tower cell with no enemy front within distance 1 leaves the lane and
    joins the siege; with entry=True (natural_crash_rule "entry") every front entering the enemy tower cell does.
    Returns the HP crashed per side."""
    T = cfg.tower
    crashed = [0, 0]
    for s in (A, B):
        if P[s] and P[s][0][0] == T and (entry or not _fronts_in_contact(P)):
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
    return crashed


def _tower_fire(siege, present, cfg: GameConfig, ev, kills) -> None:
    """Step 5 ("mass" mode): each tower deals tower_siege_damage HP to the crashed mass, at most the mass `present`
    (already under it before this tick's crashes)."""
    T = cfg.tower
    for s in (A, B):
        d = min(cfg.tower_siege_damage, present[s], siege[s])
        if d > 0:
            siege[s] -= d
            ev["tower_hp"][s] += d
            _add_kill(kills, 1 - s, -T, d)      # dies under the defender's tower


def _crash_and_tower_fire(P, siege, crash_hp, cfg: GameConfig, ev, kills, plates):
    """Steps 4 (crash check) and 5 (tower fire on the crashed mass, "mass" mode). Returns the HP crashed per side."""
    siege_before_crash = list(siege)
    crashed = _crash_check(P, siege, crash_hp, cfg, ev, kills, plates)
    if cfg.tower_mode == "mass":
        _tower_fire(siege, siege_before_crash, cfg, ev, kills)
    return crashed


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
    thinned = _pressure_thinning(P, siege_tick_start, objectives, fight is None, thin_left, cfg, ev, kills)

    # ---- 4. crash check and 5. tower fire on crashed mass ----
    crashed = _crash_and_tower_fire(P, siege, crash_hp, cfg, ev, kills, plates)

    if trace is not None:
        trace.append({
            "tick": t,
            "A": [list(p) for p in P[A]], "B": [list(p) for p in P[B]],
            "siege": list(siege), "fight": fight, "crashed": crashed, "siege_fight": siege_fight,
            "thinned": thinned,
        })


def _strike(target, d: int) -> int:
    """Remove up to d HP from a packet [x, hp]; returns the HP actually removed."""
    d = min(d, target[1])
    target[1] -= d
    return d


def _designation(P, siege, s: int, target: Optional[MinionTarget], t: int, cfg: GameConfig) -> str:
    """M4.5a: does side s's wave designate the enemy champion on tick t (start-of-tick state)? "retaliation" (the enemy
    champion damaged ours this turn and t >= the retaliation tick), "positional" (the enemy champion did not move this
    turn and stands on a lane cell holding our front, its own front strictly behind it or absent, no crashed mass of its
    own in that cell) or ""."""
    if target is None or not P[s]:
        return ""
    if target.retaliate_from is not None and t >= target.retaliate_from:
        return "retaliation"
    c = target.cell                                      # the enemy champion, in s's own coordinates
    if cfg.minion_positional_aggro and target.stationary and c != WARD and P[s][0][0] == c:
        o, own = 1 - s, -c                               # the champion's cell in its own coordinates
        behind = not P[o] or P[o][0][0] < own
        if behind and not (siege[o] > 0 and own == cfg.tower):
            return "positional"
    return ""


def _tick_natural(P, siege, acts, t, crash_hp, cfg: GameConfig, trace=None, ev=None, k=0, reach=None,
                  kills=None, thin_left=None, plates=None, champion_reach=None, hits=None, budgets=None,
                  targets=None, champ_hits=None, fire_log=None):
    """One movement tick under wave_dynamics "natural" (module docstring, "Natural waves").

    Minions follow the same rules whatever the actions; the wave slot only sets the champion's damage budget for the
    tick, champion_wave_damage[slot] HP, spent on the first reachable target in the order: the enemy front in a lane
    fight with ours, the enemy's crashed mass under our tower, the enemy's leading packet. A target is reachable when
    its own-perspective cell is <= the champion's reach line (champion_reach; NO_REACH while away, channeling or in a
    connected trade). Every HP removed is a farm opportunity for the remover's side at the cell it died in.

    M4.5a (`targets`, per side a positional.MinionTarget or None): at the start of the tick each side's wave may
    designate the enemy champion (`_designation`); after movement, a designated wave whose front is within minion_range
    (graph distance) of the champion fires its tick firepower at it (added to champ_hits[champion]) and deals no minion
    damage on this tick (neither the lane fight nor the siege fight). A champion's reach may be a line (int) or an
    interval (wave_reach_rule "ball"; positional.reaches)."""
    T = cfg.tower
    waves = (wave_of(acts[A]), wave_of(acts[B]))
    objectives = (objective_of(acts[A]), objective_of(acts[B]))
    reach_line = (T, T) if champion_reach is None else tuple(champion_reach)
    md = lambda h: minion_damage(h, cfg, t)
    if budgets is None:
        budget = [cfg.champion_wave_damage[waves[A]], cfg.champion_wave_damage[waves[B]]]
    else:                                  # per-turn budgets split over the ticks, earlier ticks first
        n = cfg.ticks_per_turn
        budget = [-((-(t + 1) * b) // n) + ((-t * b) // n) for b in budgets]   # ceil((t+1)b/n) - ceil(tb/n)
    siege_tick_start = list(siege)
    contact = _fronts_in_contact(P)
    # ---- 0. M4.5a: target designation from the start-of-tick state ----
    want = ("", "") if targets is None else tuple(_designation(P, siege, s, targets[s], t, cfg) for s in (A, B))

    # ---- 1. movement: a free front walks; contact with the enemy front, or (defender) with the enemy's crashed mass
    #         under its own tower (own x >= -T - 1, distance <= 1 from the tower cell), stops it ----
    for s in (A, B):
        e = 1 - s
        packets = P[s]
        for i, pk in enumerate(packets):
            if i > 0:
                if pk[0] + 1 < packets[0][0]:
                    pk[0] += 1             # rears walk, never into the cell of a front they cannot join
            elif not contact and not (siege[e] > 0 and pk[0] >= -T - 1):
                pk[0] += 1

    # ---- 2. merging ----
    for s in (A, B):
        P[s] = _merge(P[s], cfg.hp_cap, ev, s, True)
        if reach is not None and P[s]:
            reach[s] = max(reach[s], P[s][0][0])

    # ---- 2b. natural_crash_rule "entry": a front entering the enemy tower cell joins the crashed mass (it fights the
    #          defenders from there, as part of the siege) ----
    crashed = [0, 0]
    if cfg.natural_crash_rule == "entry":
        crashed = _crash_check(P, siege, crash_hp, cfg, ev, kills, plates, entry=True)

    champ_dealt = [0, 0]                   # HP the champion of side s removed this tick

    # ---- 2c. M4.5a: a designated wave in range fires its whole tick firepower at the enemy champion ----
    #      (a blob whose tick share of firepower is 0, e.g. 1-7 HP on the second tick, has nothing to fire and keeps its
    #      normal role, such as engaging a crashed mass)
    on_champ = [False, False]
    if targets is not None:
        for s in (A, B):
            u = md(_blob_hp(P[s], cfg)) if (want[s] and P[s]) else 0
            if u > 0 and graph_dist(P[s][0][0], targets[s].cell) <= cfg.minion_range:
                on_champ[s] = True
                champ_hits[1 - s] += u
                fire_log[1 - s].append((want[s], u))
            else:
                fire_log[1 - s].append(("", 0))

    # ---- 3. contact fight: Lanchester minion damage, the champions' damage, the tower on a contested attacker ----
    fight = None
    if _fronts_in_contact(P):
        fa, fb = P[A][0], P[B][0]
        md_b = 0 if on_champ[A] else md(_blob_hp(P[A], cfg))   # A's minions on B's front (M4.5a: unless on B)
        md_a = 0 if on_champ[B] else md(_blob_hp(P[B], cfg))
        ch_a = budget[A] if reaches(reach_line[A], -fb[0]) else 0   # A's champion on B's front
        ch_b = budget[B] if reaches(reach_line[B], -fa[0]) else 0
        budget[A] -= ch_a
        budget[B] -= ch_b
        tw_b = cfg.tower_damage if fb[0] == T else 0             # B front contesting inside A's tower cell
        tw_a = cfg.tower_damage if fa[0] == T else 0
        dmg_a, dmg_b = md_a + ch_b + tw_a, md_b + ch_a + tw_b
        fight = (fa[0], fa[1], fb[0], fb[1], dmg_a, dmg_b)
        lost_a, lost_b = min(dmg_a, fa[1]), min(dmg_b, fb[1])
        # minions and towers strike first; the champion's share is what they left
        champ_dealt[B] += max(0, lost_a - (md_a + tw_a))
        champ_dealt[A] += max(0, lost_b - (md_b + tw_b))
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

    # ---- 3b. the defender's front engages the crashed mass under its own tower (it stands within distance 1 of it);
    #          a crashed mass in such a fight does not hit the tower this tick (minions fight minions first) ----
    engaged = [False, False]
    siege_fight = None
    if fight is None:
        for s in (A, B):
            e = 1 - s
            if siege[s] <= 0 or not P[e] or P[e][0][0] < -T - 1 or on_champ[e]:
                continue
            fe = P[e][0]
            ch = budget[e] if reaches(reach_line[e], -T) else 0
            budget[e] -= ch
            md_e = md(_blob_hp(P[e], cfg))
            lost_sg = min(md_e + ch, siege[s])
            lost_e = min(md(siege[s]), fe[1])
            champ_dealt[e] += max(0, lost_sg - md_e)
            siege[s] -= lost_sg
            fe[1] -= lost_e
            _add_kill(kills, e, -T, lost_sg)
            _add_kill(kills, s, -fe[0], lost_e)
            ev["killed_hp"][s] += lost_sg
            ev["siege_killed_hp"][s] += lost_sg
            ev["killed_hp"][e] += lost_e
            engaged[s] = True
            siege_fight = (s, lost_sg, lost_e)
            if fe[1] <= 0:
                P[e].pop(0)
                ev["died"][e] += 1

    # ---- 3c. thinning (Apply Pressure), as in the scripted rule ----
    thinned = _pressure_thinning(P, siege_tick_start, objectives, fight is None, thin_left, cfg, ev, kills)

    # ---- 3d. a champion whose budget is unspent strikes the enemy's crashed mass under its own tower, else the
    #          enemy's leading packet, if reachable (no minion of its own needs to be involved) ----
    for s in (A, B):
        e = 1 - s
        if budget[s] <= 0:
            continue
        if siege[e] > 0 and reaches(reach_line[s], -T):
            d = min(budget[s], siege[e])
            siege[e] -= d
            _add_kill(kills, s, -T, d)
            ev["killed_hp"][e] += d
            ev["siege_killed_hp"][e] += d
        elif P[e] and reaches(reach_line[s], -P[e][0][0]):
            target = P[e][0]
            d = _strike(target, budget[s])
            _add_kill(kills, s, -target[0], d)
            ev["killed_hp"][e] += d
            if target[1] <= 0:
                P[e].pop(0)
                ev["died"][e] += 1
        else:
            continue
        budget[s] = 0
        champ_dealt[s] += d
    for s in (A, B):
        ev["champion_hp"][1 - s] += champ_dealt[s]

    # ---- minion damage on the enemy tower: the crashed mass already there at the start of the tick, unless it
    #      fought the defender's minions this tick ----
    if hits is not None:
        for s in (A, B):
            if siege_tick_start[s] > 0 and not engaged[s]:
                hits[s] += md(siege_tick_start[s])

    # ---- 4. crash check and 5. tower fire on crashed mass (only on the mass already there at the start of the tick)
    if cfg.natural_crash_rule == "entry":
        _tower_fire(siege, siege_tick_start, cfg, ev, kills)
    else:
        crashed = _crash_and_tower_fire(P, siege, crash_hp, cfg, ev, kills, plates)

    if trace is not None:
        entry = {
            "tick": t,
            "A": [list(p) for p in P[A]], "B": [list(p) for p in P[B]],
            "siege": list(siege), "fight": fight, "crashed": crashed, "siege_fight": siege_fight,
            "thinned": thinned, "champion": list(champ_dealt),
        }
        if targets is not None:            # M4.5a: per CHAMPION, (cause, minion damage units) taken this tick
            entry["minion_fire"] = [list(fire_log[A][-1]), list(fire_log[B][-1])]
        trace.append(entry)


def step_turn(state: State, k: int, act_a: int, act_b: int, cfg: GameConfig,
              record: bool = False, champion_reach: Optional[Tuple[int, int]] = None,
              champion_budget: Optional[Tuple[int, int]] = None,
              champion_targets: Optional[Tuple[Optional[MinionTarget], Optional[MinionTarget]]] = None) -> TurnResult:
    """Advance the lane through decision turn k (ticks 2k+1 .. 2k+2).

    `champion_budget` (wave_dynamics "natural" only; None = champion_wave_damage[slot] HP per tick): each champion's
    wave-damage budget for the whole turn in HP, split over the ticks (wave_clear_scaling, set by the M4 resolver);
    under wave_thin_rule "adaptive" a Thin side's budget is further capped at `thin_need` of the start-of-turn state.

    `champion_reach` (champion mode only; None = the legacy rule) overrides, per side, the furthest
    own-perspective position at which the champion acts on its leading wave (shove damage, the push that
    breaks a freeze, the shove of a siege-engaging defender). Posture then plays no role in the lane step. Under
    wave_reach_rule "ball" (M4.5a) an entry is an interval (lo, hi) of reachable own cells instead of a line.

    `champion_targets` (M4.5a, wave_dynamics "natural" only; None = no minion targeting): per side, the enemy
    champion its wave may target this turn (positional.MinionTarget, or None while that champion is away); the
    result's champion_hits / champion_fire report the minion damage units each champion took."""
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
    if champion_budget is not None and cfg.wave_thin_rule == "adaptive":
        champion_budget = tuple(min(b, thin_need(P, s, cfg)) if wave_of(acts[s]) == F else b
                                for s, b in enumerate(champion_budget))
    trace = [] if record else None
    ev = {key: [0, 0] for key in EVENT_KEYS}
    if cfg.wave_dynamics == "natural":
        ev.update({key: [0, 0] for key in NATURAL_EVENT_KEYS})
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
    champ_hits, fire_log = [0, 0], ([], [])
    if cfg.wave_dynamics == "natural":
        for t in range(cfg.ticks_per_turn):
            _tick_natural(P, siege, acts, t, crash_hp, cfg, trace, ev, k, reach, kills, thin_left, plates,
                          champion_reach, hits, champion_budget, champion_targets, champ_hits, fire_log)
    else:
        assert champion_budget is None, "champion_budget applies to wave_dynamics 'natural' only"
        assert champion_targets is None, "champion_targets applies to wave_dynamics 'natural' only"
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
    fire = None
    if champion_targets is not None:
        fire = (tuple(c for c, _ in fire_log[A]), tuple(c for c, _ in fire_log[B]))
    return TurnResult(nxt, (crash_hp[A], crash_hp[B]), trace, ev, exposure, kills,
                      (claimed[A], claimed[B]), thinned, (vision_set[A], vision_set[B]), (hits[A], hits[B]),
                      (champ_hits[A], champ_hits[B]), fire)


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
