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
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .config import GameConfig, posture_of
from .game import LaneGame, apply_choice, champion_exposure, collection_efficiency
from .mechanics import TurnResult, step_turn
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
                   events: "OrderedDict[str, Callable]" = EVENTS) -> ReachabilityReport:
    cfg = game.cfg
    K = game.horizon if horizon is None else horizon
    s0 = game.initial_state()
    parent: Dict[Tuple[int, State], tuple] = {(0, s0): None}
    layer = [s0]
    rep = ReachabilityReport(counts={e: 0 for e in events})
    distinct = set()
    for k in range(K):
        rep.states += len(layer)
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
    return out
