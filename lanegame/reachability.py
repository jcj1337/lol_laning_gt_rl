"""Reachability of lane events from the standard initial state, with witness action sequences.

`explore_events` enumerates every reachable (k, state) and every legal joint
action (and crash-reward choice), evaluates event predicates on each turn, and
returns for each event the number of reachable transitions on which it occurs
and one shortest witness: the joint actions and crash choices that lead from
s_0 to the event. Witnesses are reproducible with `replay`.

Events are defined on a single turn (start state, actions, TurnResult):

  crash_small / crash_medium / crash_large   a packet of that size enters the enemy siege
  crash_onto_siege       a crash while the same side already has crashed mass under that tower
  freeze_held            a freeze stops an enemy front that wanted to advance
  freeze_broken          an enemy front advances through a freeze line
  bounce                 a stacked defender wave (>= 2 waves of HP) is released when the
                         attacker's crashed mass is cleared
  siege_persists         crashed mass is still under the tower after a full turn of tower fire
  safe_disengages        the Safe posture lowers champion exposure below the Contest exposure
  tower_farm             a defender farms minions killed by its tower
  farm_contest_vs_safe   kills on the same turn where Contest and Safe efficiencies differ
  minions_deleted        HP removed by a cap (merge overflow or siege cap) instead of being killed:
                         an accounting artifact that the revised model eliminates

Siege-objective events (siege model only):

  pressure_applied       a player chooses Apply Pressure
  plate_claimed          an Apply Pressure turn claims a plate
  pressure_no_plates     an Apply Pressure turn with no plate opportunity left (thinning / zoning only)
  thinning               an Apply Pressure champion removes HP from an accessible defender packet
  pressure_no_target     an Apply Pressure turn with no accessible defender packet to thin
  vision_established     a player chooses Establish Vision
  vision_refreshed       ... while its previous vision is still active (timer reset, no stacking)
  vision_expired         a vision timer runs out this turn
  pressure_under_vision  Apply Pressure while the player's own vision is active (established earlier)
  siege_disengaged       a player with an eligible siege plays Safe
  crash_under_vision     a crash while the crasher's vision is active

The complete Vision -> Shove -> Crash -> Pressure strategy needs history (the
crash must come after the vision was established, and the pressure must use
that same vision); `find_vision_strategy` searches the game tree with a small
history flag added to the state and returns a replayable witness with a
turn-by-turn trace.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .config import PRESSURE, VISION, GameConfig, objective_of, posture_of
from .game import LaneGame, apply_choice, champion_exposure, collection_efficiency, gank_probability
from .mechanics import TurnResult, step_turn, thinning_target
from .state import State

A, B = 0, 1


def _size_event(size):
    def f(cfg, s, a, b, r):
        return any(h > 0 and cfg.size_category(h) == size for h in r.crash_hp)
    return f


def _bounce(cfg, s, a, b, r):
    W = cfg.wave_hp
    return ((s.asg > 0 and r.state.asg == 0 and r.state.bh >= 2 * W)
            or (s.bsg > 0 and r.state.bsg == 0 and r.state.ah >= 2 * W))


def _safe_disengages(cfg, s, a, b, r):
    for side, act in ((A, a), (B, b)):
        if posture_of(act) == 1 and champion_exposure(r.exposure[side], act, cfg) < r.exposure[side]:
            return True
    return False


def _tower_farm(cfg, s, a, b, r):
    return r.events["tower_hp"][A] + r.events["tower_hp"][B] > 0


def _farm_contest_vs_safe(cfg, s, a, b, r):
    for side in (A, B):
        for x in r.kills[side]:
            if collection_efficiency(x, 0, 0, cfg) != collection_efficiency(x, 1, 0, cfg):
                return True
    return False


def _either(pred):
    def f(cfg, s, a, b, r):
        return pred(cfg, s, a, b, r, A) or pred(cfg, s, a, b, r, B)
    return f


def _act(a, b, side):
    return a if side == A else b


def _pressure(cfg, s, a, b, r, side):
    return objective_of(_act(a, b, side)) == PRESSURE


def _plate(cfg, s, a, b, r, side):
    return r.plates[side] > 0


def _pressure_no_plates(cfg, s, a, b, r, side):
    return _pressure(cfg, s, a, b, r, side) and r.plates[side] == 0


def _thinning(cfg, s, a, b, r, side):
    return r.thinned[side] > 0


def _pressure_no_target(cfg, s, a, b, r, side):
    return _pressure(cfg, s, a, b, r, side) and r.thinned[side] == 0


def _vision(cfg, s, a, b, r, side):
    return objective_of(_act(a, b, side)) == VISION


def _vision_refresh(cfg, s, a, b, r, side):
    return _vision(cfg, s, a, b, r, side) and s.vision_of(side) > 0


def _vision_expired(cfg, s, a, b, r, side):
    return s.vision_of(side) == 1 and r.state.vision_of(side) == 0


def _pressure_under_vision(cfg, s, a, b, r, side):
    return _pressure(cfg, s, a, b, r, side) and s.vision_of(side) > 0


def _siege_disengaged(cfg, s, a, b, r, side):
    return cfg.objectives and s.siege_of(side) > 0 and posture_of(_act(a, b, side)) == 1


def _crash_under_vision(cfg, s, a, b, r, side):
    return r.crash_hp[side] > 0 and s.vision_of(side) > 0


EVENTS: "OrderedDict[str, Callable]" = OrderedDict([
    ("crash_small", _size_event(0)),
    ("crash_medium", _size_event(1)),
    ("crash_large", _size_event(2)),
    ("crash_onto_siege", lambda cfg, s, a, b, r: (r.crash_hp[A] > 0 and s.asg > 0) or (r.crash_hp[B] > 0 and s.bsg > 0)),
    ("freeze_held", lambda cfg, s, a, b, r: sum(r.events["freeze_held"]) > 0),
    ("freeze_broken", lambda cfg, s, a, b, r: sum(r.events["freeze_broken"]) > 0),
    ("bounce", _bounce),
    ("siege_persists", lambda cfg, s, a, b, r: (s.asg > 0 and r.state.asg > 0) or (s.bsg > 0 and r.state.bsg > 0)),
    ("safe_disengages", _safe_disengages),
    ("tower_farm", _tower_farm),
    ("farm_contest_vs_safe", _farm_contest_vs_safe),
    ("minions_deleted", lambda cfg, s, a, b, r: sum(r.events["overflow_hp"]) > 0),
])

SIEGE_EVENTS: "OrderedDict[str, Callable]" = OrderedDict([
    ("pressure_applied", _either(_pressure)),
    ("plate_claimed", _either(_plate)),
    ("pressure_no_plates", _either(_pressure_no_plates)),
    ("thinning", _either(_thinning)),
    ("pressure_no_target", _either(_pressure_no_target)),
    ("vision_established", _either(_vision)),
    ("vision_refreshed", _either(_vision_refresh)),
    ("vision_expired", _either(_vision_expired)),
    ("pressure_under_vision", _either(_pressure_under_vision)),
    ("siege_disengaged", _either(_siege_disengaged)),
    ("crash_under_vision", _either(_crash_under_vision)),
])

ALL_EVENTS: "OrderedDict[str, Callable]" = OrderedDict(list(EVENTS.items()) + list(SIEGE_EVENTS.items()))


def events_for(cfg: GameConfig) -> "OrderedDict[str, Callable]":
    return ALL_EVENTS if cfg.objectives else EVENTS


@dataclass
class Witness:
    k: int                                  # turn on which the event happens
    state: State                            # state at the start of that turn
    plan: List[Tuple[int, int]]             # joint actions for turns 0..k (last one triggers the event)
    choices: List[Tuple[int, int]]          # crash-option indices taken after turns 0..k-1


@dataclass
class ReachabilityReport:
    counts: Dict[str, int] = field(default_factory=dict)       # reachable transitions with the event
    first_k: Dict[str, int] = field(default_factory=dict)
    witness: Dict[str, Witness] = field(default_factory=dict)
    states: int = 0
    distinct_states: int = 0
    collection_values: Dict[float, int] = field(default_factory=dict)   # realized efficiencies


def explore_events(game: LaneGame, horizon: Optional[int] = None,
                   events: "OrderedDict[str, Callable]" = None, max_states: int = 3_000_000) -> ReachabilityReport:
    cfg = game.cfg
    events = events_for(cfg) if events is None else events
    K = game.horizon if horizon is None else horizon
    s0 = game.initial_state()
    parent: Dict[Tuple[int, State], tuple] = {(0, s0): None}
    layer = [s0]
    rep = ReachabilityReport(counts={e: 0 for e in events})
    distinct = set()
    for k in range(K):
        rep.states += len(layer)
        if rep.states > max_states:
            raise RuntimeError(f"reachable state count exceeded {max_states} at k={k}")
        distinct.update(layer)
        nxt = []
        for s in layer:
            for a in game.legal(s, A):
                for b in game.legal(s, B):
                    r: TurnResult = step_turn(s, k, a, b, cfg)
                    for name, pred in events.items():
                        if pred(cfg, s, a, b, r):
                            rep.counts[name] += 1
                            if name not in rep.witness:
                                rep.first_k[name] = k
                                rep.witness[name] = _witness(parent, k, s, a, b)
                    for side, act, opp in ((A, a, b), (B, b, a)):
                        for x in r.kills[side]:
                            c = round(collection_efficiency(x, posture_of(act), posture_of(opp), cfg), 6)
                            rep.collection_values[c] = rep.collection_values.get(c, 0) + 1
                    if k + 1 < K:
                        tr = game.transition(s, k, a, b)
                        for i, j, s2, _ in tr.choice_outcomes(cfg):
                            key = (k + 1, s2)
                            if key not in parent:
                                parent[key] = (k, s, a, b, i, j)
                                nxt.append(s2)
        layer = sorted(nxt)
    rep.distinct_states = len(distinct)
    return rep


def _witness(parent, k, s, a, b) -> Witness:
    plan, choices = [(a, b)], []
    node = (k, s)
    while parent[node] is not None:
        pk, ps, pa, pb, i, j = parent[node]
        plan.append((pa, pb))
        choices.append((i, j))
        node = (pk, ps)
    plan.reverse()
    choices.reverse()
    return Witness(k, s, plan, choices)


def replay(game: LaneGame, plan, choices, record: bool = False):
    """Replay a witness from s_0; returns the list of (state, TurnResult) per turn."""
    cfg = game.cfg
    s = game.initial_state()
    out = []
    for k, (a, b) in enumerate(plan):
        r = step_turn(s, k, a, b, cfg, record=record)
        out.append((s, r))
        if k < len(choices):
            tr = game.transition(s, k, a, b)
            i, j = choices[k]
            s, _ = apply_choice(tr.pre_state, tr.options_a[i], tr.options_b[j], cfg)
        else:
            s = game.transition(s, k, a, b).pre_state
    return out


# ------------------------------------------------------------------ the Vision -> Shove -> Crash -> Pressure strategy
# History flag for player `side`:
#   0  no active vision run
#   1  vision established (and still active), no crash by `side` since it was established
#   2  a crash by `side` happened after the vision was established, vision still active
# Re-establishing vision restarts the run (flag 1): the pressure must be protected by a vision that
# was in place before the crash and was not refreshed in between.
FLAG_NONE, FLAG_VISION, FLAG_CRASHED = 0, 1, 2


def _next_flag(flag: int, s: State, act: int, r: TurnResult, side: int, min_crash_hp: int = 1) -> int:
    if objective_of(act) == VISION:
        return FLAG_VISION
    if flag != FLAG_NONE and r.state.vision_of(side) == 0:
        return FLAG_NONE
    if flag == FLAG_VISION and r.crash_hp[side] >= min_crash_hp:
        return FLAG_CRASHED
    return flag


def strategy_complete(s: State, act: int, r: TurnResult, flag: int, side: int, require_plate: bool = True) -> bool:
    """Turn on which the strategy completes: Apply Pressure under a vision established before the
    current siege's crash (flag 2), optionally requiring that a plate is actually claimed."""
    return (flag == FLAG_CRASHED and s.vision_of(side) > 0 and objective_of(act) == PRESSURE
            and (r.plates[side] > 0 or not require_plate))


@dataclass
class StrategyWitness:
    found: bool
    k: int = -1                                   # turn of the completing pressure action
    plan: List[Tuple[int, int]] = field(default_factory=list)
    vision_turn: int = -1                         # turn on which the protecting vision was established
    crash_turn: int = -1                          # turn of the first crash after that vision
    turns_after_vision: int = -1                  # k - vision_turn
    states_searched: int = 0
    trace: List[dict] = field(default_factory=list)


def find_vision_strategy(game: LaneGame, side: int = A, horizon: Optional[int] = None,
                         require_plate: bool = True, max_states: int = 3_000_000,
                         min_crash_hp: int = 1) -> StrategyWitness:
    """Breadth-first search over (k, state, flag) from s_0 for the first turn on which `side` applies
    pressure under a vision it established before the crash that opened the current siege. The crash
    that counts must have at least `min_crash_hp` HP (1 = any crash, 4 = a full wave, 8 = medium, 12 = large)."""
    cfg = game.cfg
    assert cfg.objectives, "siege objectives are needed for this strategy"
    K = game.horizon if horizon is None else horizon
    s0 = game.initial_state()
    parent: Dict[Tuple[int, State, int], tuple] = {(0, s0, FLAG_NONE): None}
    layer = [(s0, FLAG_NONE)]
    searched = 0
    for k in range(K):
        searched += len(layer)
        if searched > max_states:
            raise RuntimeError(f"search exceeded {max_states} nodes at k={k}")
        nxt = []
        for s, flag in layer:
            for a in game.legal(s, A):
                for b in game.legal(s, B):
                    act = a if side == A else b
                    r = step_turn(s, k, a, b, cfg)
                    if strategy_complete(s, act, r, flag, side, require_plate):
                        plan = _strategy_plan(parent, k, s, flag, a, b)
                        return _strategy_witness(game, side, plan, k, searched, min_crash_hp)
                    if k + 1 < K:
                        f2 = _next_flag(flag, s, act, r, side, min_crash_hp)
                        key = (k + 1, r.state, f2)
                        if key not in parent:
                            parent[key] = (k, s, flag, a, b)
                            nxt.append((r.state, f2))
        layer = nxt
    return StrategyWitness(False, states_searched=searched)


def _strategy_plan(parent, k, s, flag, a, b):
    plan = [(a, b)]
    node = (k, s, flag)
    while parent[node] is not None:
        pk, ps, pf, pa, pb = parent[node]
        plan.append((pa, pb))
        node = (pk, ps, pf)
    plan.reverse()
    return plan


def strategy_trace(game: LaneGame, plan, side: int = A) -> List[dict]:
    """Turn-by-turn record of a plan from s_0: lane, actions, siege decisions, timers, plates, thinning,
    gank exposure and probability for `side`."""
    from .render import state_row
    cfg = game.cfg
    s = game.initial_state()
    rows = []
    for k, (a, b) in enumerate(plan):
        r = step_turn(s, k, a, b, cfg, record=True)
        tr = game.transition(s, k, a, b)
        act = a if side == A else b
        e = 1 - side
        efx, efh = (s.bx, s.bh) if side == A else (s.ax, s.ah)
        n_efh = (r.state.bh if side == A else r.state.ah)
        rows.append({
            "k": k, "lane_before": state_row(s, cfg), "lane_after": state_row(r.state, cfg),
            "A": cfg.label(a), "B": cfg.label(b),
            "siege_mass_before": s.siege_of(side), "siege_mass_after": r.state.siege_of(side),
            "objective": ["-", "Pressure", "Vision"][objective_of(act)] if s.siege_of(side) > 0 else "(no siege)",
            "vision_before": s.vision_of(side), "vision_after": r.state.vision_of(side),
            "plates_available": s.plates_of(side), "plates_claimed": r.plates[side],
            "plates_after": r.state.plates_of(side),
            "defender_front_before": (efx, efh), "defender_front_after": ((r.state.bx, n_efh) if side == A else (r.state.ax, n_efh)),
            "thinned": r.thinned[side], "crash_hp": r.crash_hp[side],
            "exposure": tr.exposure[side], "gank_p": tr.gank_p[side],
            "gank_p_without_vision": gank_probability(s._replace(va=0, vb=0), side, act, cfg, tr.exposure[side]),
            "farm": tr.farm[side], "payoff_to_side": tr.payoff if side == A else -tr.payoff,
        })
        s = tr.pre_state
    return rows


def _strategy_witness(game, side, plan, k, searched, min_crash_hp: int = 1) -> StrategyWitness:
    trace = strategy_trace(game, plan, side)
    vision_turn = max(i for i, row in enumerate(trace[:k]) if row["objective"] == "Vision")
    crash_turn = min(i for i, row in enumerate(trace) if i > vision_turn and row["crash_hp"] >= min_crash_hp)
    return StrategyWitness(True, k, plan, vision_turn, crash_turn, k - vision_turn, searched, trace)
