"""Named lane scenarios with qualitative expectations.

Each scenario fixes a start state, a decision index and an action plan, runs the
deterministic mechanics, and evaluates a check. They are used (a) by the test
suite, (b) to compare candidate collision models, and (c) to produce the
traces printed in the reports. Coordinates in State are own-perspective
(see state.py); comments give global positions.

Three sets exist: `v1_scenarios` (the original model, 2 HP per wave, Contest-only
actions), `revised_scenarios` (the previous revision, 4 HP per wave, combined
actions, farming, mass-based tower fire) and `siege_scenarios` (the siege model:
the revised scenarios with legal post-crash actions, plus scenarios for Apply
Pressure, plates, thinning and persistent vision). `all_scenarios(cfg)` picks
the set that matches the configuration. Each run attaches the turn's farm
(available, collected) and the joint action to every TurnResult, so checks
can test economic outcomes.

Most scenarios start from a constructed state. `start_reachable` records
whether the start state is reachable from the standard initial state at some
decision index with the same dynamics phase (k = k0 mod phase_period) under
the default configuration; the test suite verifies the flag. The *events*
the constructed scenarios illustrate (crashes of every size, held and broken
freezes, bounces, tower farming, ...) are separately shown to be reachable
from the standard start by `lanegame.reachability`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from .config import CONTEST, F, F_C, PRESSURE, SH, SH_C, SH_S, SP, SP_C, SP_S, VISION, GameConfig, act
from .config import posture_of
from .game import champion_exposure, collection_efficiency, gank_probability, turn_farm
from .mechanics import legal_actions
from .render import trace_plan
from .state import State


@dataclass
class Scenario:
    name: str
    description: str
    start: State
    k0: int
    plan: Sequence[Tuple[int, int]]
    check: Callable[[List, GameConfig], Tuple[bool, str]]
    start_reachable: bool = False      # start state reachable from s_0 at k0 (default configuration)


def _hp_a(results, cfg):
    return [r.state.ah for r in results]


def _check_reference(results, cfg):
    r = results[0]
    st = r.state
    ok = (st.ax == 1 and cfg.stacks(st.ah) == cfg.max_stacks - 1 and r.events["died"] == [0, 1]
          and r.crash_hp == (0, 0))
    return ok, (f"after one turn: A front @{st.ax:+d} hp{st.ah} (stacks {cfg.stacks(st.ah)}); "
                f"packets destroyed A/B = {r.events['died']}")


def _check_small_pushed_back(results, cfg):
    r = results[0]
    st = r.state
    ok = r.events["died"] == [1, 0] and st.bh > 0 and -st.bx <= 1
    return ok, f"packets destroyed A/B = {r.events['died']}; B front now at global {-st.bx:+d} hp{st.bh}"


def _check_freeze_holds(results, cfg):
    no_crash = all(r.crash_hp[1] == 0 for r in results)
    positions = [-p[0] for r in results for tr in r.trace for p in tr["B"]]
    held = min(positions) >= -cfg.tower + 1
    return no_crash and held, (f"B crash hp per turn {[r.crash_hp[1] for r in results]}; "
                               f"deepest B packet position {min(positions):+d} (A tower at {-cfg.tower:+d})")


def _check_freeze_breaks(results, cfg):
    hp = results[0].crash_hp[1]
    return hp > 0, f"B crash hp on turn 1: {hp}"


def _check_slowpush_stacks(results, cfg):
    hp = _hp_a(results, cfg)
    return hp[-1] >= 5 and hp[-1] > hp[0], f"A front hp by turn: {hp}"


def _check_shove_contests(results, cfg):
    hp = _hp_a(results, cfg)
    return max(hp) <= 2, f"A front hp by turn: {hp}"


def _check_large_crash(results, cfg):
    total = sum(r.crash_hp[0] for r in results)
    return cfg.size_category(total) >= 1, f"A crash hp: {[r.crash_hp[0] for r in results]}"


def _check_small_intercepted(results, cfg):
    total = sum(r.crash_hp[0] for r in results)
    return total == 0, f"A crash hp: {[r.crash_hp[0] for r in results]}"


def _check_bounce(results, cfg):
    # defender's wave leaves the tower stronger than a fresh wave and pushes past the lane
    hps = [r.state.bh for r in results]
    pos = [-r.state.bx for r in results if r.state.bh > 0]
    return max(hps) > cfg.wave_hp and min(pos) <= 1, f"B front hp {hps}, global positions {pos}"


def _check_no_bounce(results, cfg):
    hps = [r.state.bh for r in results]
    return max(hps) <= cfg.wave_hp, f"B front hp {hps}"


def _check_two_turn_consumption(results, cfg):
    sieges = [r.state.asg for r in results]
    return sieges[:2] == [1, 0], f"A siege count after each turn: {sieges}"


def v1_scenarios(cfg: GameConfig) -> List[Scenario]:
    """Scenarios of the original Phase 1 model (2 HP per wave)."""
    W = cfg.wave_hp
    return [
        Scenario("reference",
                 "A two-stack wave (hp6) at +1 Slow Pushes; B's fresh wave arrives and Shoves.",
                 State(1, 3 * W, (), 0, -3, W, (), 0, 0, 0), 1, [(SP, SH)], _check_reference),
        Scenario("small_wave_pushed_back",
                 "Same as reference but A's wave is unstacked (hp2): it is destroyed and B pushes back.",
                 State(1, W, (), 0, -3, W, (), 0, 0, 0), 1, [(SP, SH)], _check_small_pushed_back),
        Scenario("freeze_holds",
                 "A freezes with no wave of its own; B's fresh wave (hp2) shoves from the lane, then Slow Pushes.",
                 State(0, 0, (), 0, 0, W, (), 0, 0, 0), 1, [(F, SH), (F, SP), (F, SP)], _check_freeze_holds),
        Scenario("freeze_broken",
                 "As freeze_holds but B's wave carries one stack (hp4): the shove breaks the freeze.",
                 State(0, 0, (), 0, 0, 2 * W, (), 0, 0, 0), 1, [(F, SH)], _check_freeze_breaks),
        Scenario("slowpush_stacks",
                 "A parks a one-wave surplus (hp2) at the lane and Slow Pushes against a passive Slow Push.",
                 State(0, W, (-4,), 0, -4, W, (), 0, 0, 0), 2, [(SP, SP)] * 8, _check_slowpush_stacks),
        Scenario("shove_contests_slowpush",
                 "As slowpush_stacks but B Shoves every incoming wave: the stack never builds.",
                 State(0, W, (-4,), 0, -4, W, (), 0, 0, 0), 2, [(SP, SH)] * 8, _check_shove_contests),
        Scenario("large_wave_crashes_through",
                 "A 2-stack wave at the lane Shoves while B's fresh wave arrives at its tower.",
                 State(0, 3 * W, (), 0, -4, W, (), 0, 0, 0), 2, [(SH, SP), (SH, SP)], _check_large_crash),
        Scenario("small_wave_intercepted",
                 "Same timing with an unstacked wave: B's reinforcement intercepts it at the tower.",
                 State(0, W, (), 0, -4, W, (), 0, 0, 0), 2, [(SH, SP), (SH, SP)], _check_small_intercepted),
        Scenario("two_packet_siege_bounce",
                 "Two A packets queue under B's tower; B's waiting wave merges its next spawn and bounces.",
                 State(-4, W, (), 2, -3, W, (), 0, 0, 0), 2, [(SH, SH)] * 4, _check_bounce),
        Scenario("one_packet_siege_no_bounce",
                 "A single crashed packet is consumed within one turn; B's wave leaves unstacked.",
                 State(-4, W, (), 1, -3, W, (), 0, 0, 0), 2, [(SH, SH)] * 3, _check_no_bounce),
        Scenario("tower_consumes_one_per_turn",
                 "Two queued packets take two agent turns to be consumed.",
                 State(-4, W, (), 2, -3, W, (), 0, 0, 0), 2, [(SP, SP)] * 2,
                 lambda res, cfg: ([r.state.asg for r in res] == [1, 0],
                                   f"A siege after each turn: {[r.state.asg for r in res]}")),
    ]


# ------------------------------------------------------------- revised model scenarios
def _farm(results, side, kind=1):
    """Total farm available (kind=0) or collected (kind=1) by a side over the scenario."""
    return sum(r.farm[kind][side] for r in results)


def _check_even_lane(results, cfg):
    crashes = sum(sum(r.crash_hp) for r in results)
    fa, fb = _farm(results, 0), _farm(results, 1)
    ok = crashes == 0 and fa > 0 and abs(fa - fb) < 1e-12
    return ok, f"crash HP {crashes}; farm collected A {fa:.3f}, B {fb:.3f}"


def _check_shove_wins(results, cfg):
    ka = sum(r.events["killed_hp"][1] for r in results)    # B HP killed (by A)
    kb = sum(r.events["killed_hp"][0] for r in results)
    return ka > kb, (f"HP killed by A {ka}, by B {kb}; farm available A {_farm(results, 0, 0):.2f}, "
                     f"B {_farm(results, 1, 0):.2f}")


def _check_stacks_grow(results, cfg):
    hp = [r.state.ah for r in results]
    return max(hp) > 2 * cfg.wave_hp, f"A front hp by turn: {hp}"


def _check_stack_denied(results, cfg):
    hp = [r.state.ah for r in results]
    return max(hp) <= cfg.wave_hp + cfg.wave_hp // 2, f"A front hp by turn: {hp}"


def _check_unstacked_waits(results, cfg):
    pos = [tr["A"][0][0] for r in results for tr in r.trace if tr["A"] and tr["A"][0][1] <= cfg.wave_hp]
    return bool(pos) and max(pos) <= cfg.slowpush_line, f"positions of A's unstacked front by tick: {pos}"


def _check_stack_creeps(results, cfg):
    pos = [r.state.ax for r in results]
    period = cfg.spawn_every_turns * cfg.slowpush_creep_period
    return pos == [cfg.slowpush_line + 1] * period, f"A front position by turn: {pos}"


def _check_freeze_holds_v2(results, cfg):
    no_crash = all(r.crash_hp[1] == 0 for r in results)
    deepest = max(p[0] for r in results for tr in r.trace for p in tr["B"])
    held = sum(r.events["freeze_held"][1] for r in results)
    return no_crash and deepest < cfg.tower and held > 0, \
        f"B crash hp {[r.crash_hp[1] for r in results]}; deepest B position (own) {deepest:+d}; ticks held {held}"


def _check_freeze_broken_v2(results, cfg):
    return results[0].events["freeze_broken"][1] > 0 and results[0].crash_hp[1] > 0, \
        f"freeze broken ticks {results[0].events['freeze_broken'][1]}; B crash hp {results[0].crash_hp[1]}"


def _siege_turns(results):
    out = 0
    for r in results:
        if r.state.asg > 0:
            out += 1
        else:
            break
    return out


def _check_large_persists(results, cfg):
    n = _siege_turns(results)
    return results[0].crash_hp[0] == 3 * cfg.wave_hp and n >= 5, \
        f"crash hp {results[0].crash_hp[0]}; A mass under B tower by turn {[r.state.asg for r in results]}"


def _check_small_clears(results, cfg):
    n = _siege_turns(results)
    return results[0].crash_hp[0] == cfg.wave_hp and n <= 2, \
        f"crash hp {results[0].crash_hp[0]}; A mass under B tower by turn {[r.state.asg for r in results]}"


def _check_tower_farm(lo):
    """Every HP the tower removes from A's crashed mass is one farm opportunity for B, at B's own tower,
    and B collects at least a share `lo` of it."""
    def check(results, cfg):
        T = cfg.tower
        tower = sum(r.events["tower_hp"][0] for r in results)            # A minions killed by B's tower
        at_tower = sum(r.kills[1].get(-T, 0) for r in results)
        avail = cfg.farm_per_hp * at_tower
        coll = sum(cfg.farm_per_hp * r.kills[1].get(-T, 0)
                   * collection_efficiency(-T, posture_of(r.acts[1]), posture_of(r.acts[0]), cfg)
                   for r in results)
        share = coll / avail if avail else 0.0
        ok = tower == 3 * cfg.wave_hp == at_tower and share >= lo
        return ok, (f"A HP killed by B's tower {tower}; B farm opportunity there {avail:.2f}, "
                    f"collected {coll:.3f} ({100 * share:.1f}%)")
    return check


def _check_bounce_v2(results, cfg):
    hp = [r.state.bh for r in results]
    pos = [-r.state.bx for r in results if r.state.bh > 0]      # global
    released = [r.state.bh for r in results if r.state.asg == 0 and r.state.bh > 0]
    ok = bool(released) and released[0] >= 2 * cfg.wave_hp and min(pos) <= 0
    return ok, (f"B front hp {hp}; B front global position {pos}; "
                f"A mass under B tower {[r.state.asg for r in results]}")


def _check_safe_release(results, cfg):
    crashed = any(r.crash_hp[0] for r in results)
    exp_safe = max(champion_exposure(r.exposure[0], r.acts[0], cfg) for r in results)
    exp_contest = max(r.exposure[0] for r in results)
    return crashed and exp_safe <= -1 < exp_contest, \
        f"A crashed {crashed}; A exposure with Safe {exp_safe:+d} vs with Contest {exp_contest:+d}"


def _check_freeze_denial(results, cfg):
    aa, ca, ab, cb = _farm(results, 0, 0), _farm(results, 0), _farm(results, 1, 0), _farm(results, 1)
    ea, eb = (ca / aa if aa else 0), (cb / ab if ab else 0)
    return aa > 0 and ab > 0 and ea > 2 * eb, \
        f"A collects {ca:.3f} of {aa:.2f} ({100 * ea:.0f}%), B collects {cb:.3f} of {ab:.2f} ({100 * eb:.0f}%)"


def revised_scenarios(cfg: GameConfig) -> List[Scenario]:
    """Scenarios of the revised model (defaults: 4 HP per wave)."""
    W = cfg.wave_hp
    H = 3 * W
    s0 = State(-cfg.base, W, (), 0, -cfg.base, W, (), 0, 0, 0)
    surplus = State(0, W, (-4,), 0, -4, W, (), 0, 0, 0)          # A: one-wave surplus at the lane
    return [
        Scenario("even_lane_fight",
                 "Standard start, both Slow Push + Contest: the waves fight at the lane, nobody crashes, "
                 "both farm the same amount.", s0, 0, [(SP_C, SP_C)] * 12, _check_even_lane, True),
        Scenario("shove_wins_lane",
                 "Standard start, A Shoves + Contests against Slow Push: A's champion damage wins the lane fight.",
                 s0, 0, [(SH_C, SP_C)] * 4, _check_shove_wins, True),
        Scenario("slowpush_stacks",
                 "A one-wave surplus at the lane Slow Pushes against a passive Safe opponent: reinforcements merge "
                 "and the stack grows.", surplus, 2, [(SP_C, SP_S)] * 8, _check_stacks_grow, True),
        Scenario("shove_contests_slowpush",
                 "As slowpush_stacks, but B Shoves + Contests every incoming wave: the stack never builds.",
                 surplus, 2, [(SP_C, SH_C)] * 8, _check_stack_denied, True),
        Scenario("unstacked_wave_waits",
                 "An unstacked Slow Push wave does not walk past the lane; it waits for its reinforcement.",
                 State(0, W, (-4,), 0, -4, 0, (), 0, 0, 0), 2, [(SP_C, SP_S)] * 2, _check_unstacked_waits),
        Scenario("stacked_wave_creeps",
                 "A stacked Slow Push wave (2 waves of HP) advances one cell every two spawn intervals.",
                 State(0, 2 * W, (-4,), 0, -4, 0, (), 0, 0, 0), 4, [(SP_C, SP_S)] * 4, _check_stack_creeps),
        Scenario("freeze_holds",
                 "B's fresh wave at A's freeze line Shoves + Contests into A's Freeze + Contest: the freeze holds.",
                 State(0, 0, (), 0, 1, W, (), 0, 0, 0), 1, [(F_C, SH_C), (F_C, SP_C), (F_C, SP_C)],
                 _check_freeze_holds_v2),
        Scenario("freeze_broken",
                 "B's 1-stack wave (2 waves of HP) Shoves + Contests at A's freeze line: the freeze breaks "
                 "into a crash.", State(0, 0, (), 0, 1, 2 * W, (), 0, 0, 0), 1, [(F_C, SH_C)],
                 _check_freeze_broken_v2),
        Scenario("large_crash_persists",
                 "A 3-wave packet crashes into B's empty tower: tower fire needs several turns to clear it.",
                 State(1, H, (), 0, 0, 0, (), 0, 0, 0), 2, [(SH_C, SP_S)] + [(SP_S, SP_S)] * 7,
                 _check_large_persists),
        Scenario("small_crash_clears_quickly",
                 "A one-wave packet crashes into B's empty tower and is cleared within two turns.",
                 State(1, W, (), 0, 0, 0, (), 0, 0, 0), 2, [(SH_C, SP_S)] + [(SP_S, SP_S)] * 3,
                 _check_small_clears),
        Scenario("tower_farm_contested",
                 "B farms the crashed 3-wave packet under its own tower with Contest while A stays to zone: "
                 "B still collects most of it.",
                 State(1, H, (), 0, 0, 0, (), 0, 0, 0), 2, [(SH_C, SP_C)] + [(SP_C, SP_C)] * 6,
                 _check_tower_farm(0.75)),
        Scenario("tower_farm_safe",
                 "As tower_farm_contested, but both champions play Safe: B still collects a large share.",
                 State(1, H, (), 0, 0, 0, (), 0, 0, 0), 2, [(SH_S, SP_S)] + [(SP_S, SP_S)] * 6,
                 _check_tower_farm(0.6)),
        Scenario("bounce",
                 "A 3-wave mass sits under B's tower and A lets its later waves walk in (Shove + Safe); B's waiting "
                 "wave merges its reinforcements and, once the tower has cleared the mass, a stacked B wave "
                 "walks back past the lane.",
                 State(-4, W, (), H, -3, W, (), 0, 0, 0), 2, [(SH_S, SP_C)] * 10, _check_bounce_v2),
        Scenario("safe_lets_wave_go",
                 "A Shoves + Safe with a free wave at the lane: the minions walk in and crash while A's champion "
                 "stays at its safe line.", State(0, W, (), 0, -4, 0, (), 0, 0, 0), 2, [(SH_S, SP_S)] * 2,
                 _check_safe_release),
        Scenario("freeze_denial",
                 "B's wave is held at A's freeze line; A's next wave arrives and the fight happens next to A's "
                 "tower: A (Freeze + Contest) farms efficiently, B (Safe, far forward) collects little.",
                 State(-4, W, (), 0, 1, W, (), 0, 0, 0), 2, [(F_C, SP_S)] * 4, _check_freeze_denial),
    ]


# ------------------------------------------------------------- siege model scenarios
SP_CP, SH_CP = act(SP, CONTEST, PRESSURE), act(SH, CONTEST, PRESSURE)
SP_CV, SH_CV = act(SP, CONTEST, VISION), act(SH, CONTEST, VISION)


def _released_hp(results):
    """HP of B's front on the first turn after A's crashed mass is gone (the bounce size), or 0."""
    for r in results:
        if r.state.asg == 0 and r.state.bh > 0:
            return r.state.bh
    return 0


def _check_pressure_thins_bounce(results, cfg):
    W = cfg.wave_hp
    rel = next(r.state for r in results if r.state.asg == 0)
    hp, total = rel.bh, rel.bh + W * len(rel.br)
    thinned = sum(r.thinned[0] for r in results)
    ok = thinned > 0 and 0 < total < 4 * W and not rel.br
    return ok, (f"A thinned {thinned} HP over the siege; B releases a {hp}-HP wave with nothing queued "
                f"(without pressure: a capped {3 * W}-HP wave plus a queued {W}-HP reinforcement, see 'bounce'); "
                f"B front hp by turn {[r.state.bh for r in results]}")


def _check_plates_finite(results, cfg):
    claimed = [r.plates[0] for r in results]
    grant = cfg.plate_grant[cfg.size_category(results[0].crash_hp[0])]
    ok = (sum(claimed) == min(grant, cfg.plates_per_turn * (len(results) - 1))
          and all(c == 0 for c in claimed[1 + grant:]) and results[-1].state.ap == 0)
    return ok, (f"crash of {results[0].crash_hp[0]} HP grants {grant} plate opportunities; plates claimed per turn "
                f"{claimed}; pending after the last turn {results[-1].state.ap}")


def _check_plates_lapse(results, cfg):
    pend = [r.state.ap for r in results]
    sieges = [r.state.asg for r in results]
    first_clear = next(i for i, sg in enumerate(sieges) if sg == 0)
    ok = pend[0] > 0 and all(p > 0 for p in pend[:first_clear]) and all(p == 0 for p in pend[first_clear:]) \
        and sum(r.plates[0] for r in results) == 0
    return ok, f"pending plates by turn {pend}; A mass under B tower by turn {sieges}; plates claimed 0"


def _check_vision_next_turn(results, cfg):
    r0, r1 = results[0], results[1]
    p_now = gank_probability(results[0].start, 0, r0.acts[0], cfg, champion_exposure(r0.exposure[0], r0.acts[0], cfg))
    p_pressure_now = gank_probability(results[0].start, 0, SP_CP, cfg, champion_exposure(r0.exposure[0], SP_CP, cfg))
    p_next = gank_probability(r0.state, 0, r1.acts[0], cfg, champion_exposure(r1.exposure[0], r1.acts[0], cfg))
    p_next_unprotected = gank_probability(r0.state._replace(va=0), 0, r1.acts[0], cfg,
                                          champion_exposure(r1.exposure[0], r1.acts[0], cfg))
    ok = (abs(p_now - p_pressure_now) < 1e-12 and r0.state.va == cfg.vision_turns
          and abs(p_next - (1 - cfg.vision_reduction) * p_next_unprotected) < 1e-12)
    return ok, (f"gank p on the vision turn {p_now:.3f} (= pressure's {p_pressure_now:.3f}, no retroactive effect); "
                f"timer after it {r0.state.va}; next turn's pressure p {p_next:.3f} vs {p_next_unprotected:.3f} unprotected")


def _check_vision_refresh(results, cfg):
    timers = [r.state.va for r in results]
    V = cfg.vision_turns
    ok = timers[0] == V and timers[1] == V - 1 and timers[2] == V and timers[-1] == max(0, V - (len(timers) - 3))
    return ok, f"vision timer after each turn {timers} (re-established on turn 3: reset to {V}, not {2 * V - 1})"


def _check_pressure_needs_target(results, cfg):
    thinned = [r.thinned[0] for r in results]
    ok = thinned[:2] == [0, 0] and results[0].plates[0] > 0 and thinned[2] > 0
    return ok, (f"thinned per turn {thinned}: nothing to thin until B's next wave has spawned and walked up "
                f"behind its tower (turn 3); the plate is still claimed on turn 1 ({results[0].plates[0]})")


def _check_safe_no_pressure(results, cfg):
    ok = all(r.plates[0] == 0 and r.thinned[0] == 0 for r in results) and \
        all(champion_exposure(r.exposure[0], r.acts[0], cfg) <= -1 for r in results)
    return ok, (f"plates {[r.plates[0] for r in results]}, thinned {[r.thinned[0] for r in results]}, "
                f"A exposure {[champion_exposure(r.exposure[0], r.acts[0], cfg) for r in results]}")


def _check_vision_then_pressure(results, cfg):
    # vision on turn 1, pressure on turn 2 with the timer active: 75% lower risk than the same action unprotected
    r1 = results[1]
    p = gank_probability(results[0].state, 0, r1.acts[0], cfg, champion_exposure(r1.exposure[0], r1.acts[0], cfg))
    p0 = gank_probability(results[0].state._replace(va=0), 0, r1.acts[0], cfg,
                          champion_exposure(r1.exposure[0], r1.acts[0], cfg))
    ok = results[0].state.va == cfg.vision_turns and r1.plates[0] > 0 and abs(p - (1 - cfg.vision_reduction) * p0) < 1e-12
    return ok, f"turn 2 applies pressure (plate {r1.plates[0]}) at gank p {p:.4f} instead of {p0:.4f}"


def siege_scenarios(cfg: GameConfig) -> List[Scenario]:
    """Scenarios of the siege model: the revised set with legal post-crash actions, plus siege scenarios."""
    W = cfg.wave_hp
    H = 3 * W
    base = revised_scenarios(cfg)
    # after a crash a contesting champion must carry a siege objective: Apply Pressure zones the defender,
    # claims plates and thins, which is the natural reading of "stays to zone" in tower_farm_contested
    for sc in base:
        if sc.name == "tower_farm_contested":
            sc.plan = [(SH_C, SP_C)] + [(SP_CP, SP_C)] * 6
    crash12 = State(1, H, (), 0, 0, 0, (), 0, 0, 0)                 # A's 3-wave packet one cell from B's empty tower
    siege12 = State(-4, W, (), H, -3, W, (), 0, 0, 0, 3, 0)         # bounce start + 3 plate opportunities
    return base + [
        Scenario("pressure_thins_bounce",
                 "As 'bounce' (12 HP under B's tower, B's wave waiting behind it), but A applies pressure every "
                 "turn while its own waves wait at the lane: 1 HP per turn is thinned from B's waiting wave, so "
                 "B releases a 10-HP wave with nothing queued instead of a capped 12-HP wave plus a queued "
                 "reinforcement.", siege12, 2,
                 [(SP_CP, SP_C)] * 6 + [(SP_S, SP_C)] * 4, _check_pressure_thins_bounce),
        Scenario("plates_are_finite",
                 "A crashes a 3-wave packet (3 plate opportunities) and applies pressure for 6 turns: one plate per "
                 "turn for 3 turns, then nothing more to claim.", crash12, 2,
                 [(SH_C, SP_S)] + [(SP_CP, SP_S)] * 6, _check_plates_finite),
        Scenario("plates_lapse_with_the_siege",
                 "A crashes a 3-wave packet and plays Safe: the plate opportunities stay pending while the mass is "
                 "under the tower and lapse, unclaimed, once it is cleared.", crash12, 2,
                 [(SH_C, SP_S)] + [(SP_S, SP_S)] * 7, _check_plates_lapse),
        Scenario("vision_protects_next_turn",
                 "A establishes vision during its siege: this turn's gank risk is the same as Apply Pressure's "
                 "(no retroactive effect); from the next turn on risk is reduced by 75%.", siege12, 2,
                 [(SP_CV, SP_S), (SP_CP, SP_S)], _check_vision_next_turn),
        Scenario("vision_refreshes_not_stacks",
                 "A establishes vision, pressures, establishes vision again: the timer resets to 6, it does not "
                 "add up.", siege12, 2, [(SP_CV, SP_S), (SP_CP, SP_S), (SP_CV, SP_S)] + [(SP_S, SP_S)] * 4,
                 _check_vision_refresh),
        Scenario("thinning_needs_a_target",
                 "A's crashed mass sits under B's tower while B has no minions in the lane: pressure claims a plate "
                 "but thins nothing until B's next wave has spawned and walked up behind the tower.",
                 State(-4, W, (), H, -4, 0, (), 0, 0, 0, 3, 0), 2, [(SP_CP, SP_S)] * 3, _check_pressure_needs_target),
        Scenario("safe_siege_is_disengaged",
                 "A plays Safe during its siege: no plates, no thinning, champion at its safe line.",
                 siege12, 2, [(SP_S, SP_C)] * 3, _check_safe_no_pressure),
        Scenario("vision_then_pressure",
                 "A establishes vision on the first siege turn and applies pressure on the second, protected by it.",
                 siege12, 2, [(SP_CV, SP_C), (SP_CP, SP_C)], _check_vision_then_pressure),
    ]


def all_scenarios(cfg: GameConfig) -> List[Scenario]:
    """The scenario set that matches the configuration's model generation."""
    if not cfg.positioning:
        return v1_scenarios(cfg)
    return siege_scenarios(cfg) if cfg.objectives else revised_scenarios(cfg)


def run_scenario(sc: Scenario, cfg: GameConfig):
    s = sc.start
    for i, (a, b) in enumerate(sc.plan):
        assert a in legal_actions(s, 0, cfg) or i > 0, f"{sc.name}: illegal first action for A"
    lines, results = trace_plan(cfg, sc.start, sc.k0, sc.plan)
    start = sc.start
    for (a, b), r in zip(sc.plan, results):
        r.acts = (a, b)
        r.farm = turn_farm(r, a, b, cfg)
        r.start = start
        start = r.state
    ok, detail = sc.check(results, cfg)
    return ok, detail, lines, results
