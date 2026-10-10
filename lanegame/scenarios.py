"""Named lane scenarios with qualitative expectations.

Each scenario fixes a start state, a decision index and an action plan, runs the
deterministic mechanics, and evaluates a check. They are used (a) by the test
suite, (b) to compare candidate collision models, and (c) to produce the
traces printed in the report. Coordinates in State are own-perspective
(see state.py); comments give global positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from .config import F, SH, SP, GameConfig
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
    # defender's wave leaves the tower stronger than a fresh wave and pushes past the river
    hps = [r.state.bh for r in results]
    pos = [-r.state.bx for r in results if r.state.bh > 0]
    return max(hps) > cfg.wave_hp and min(pos) <= 1, f"B front hp {hps}, global positions {pos}"


def _check_no_bounce(results, cfg):
    hps = [r.state.bh for r in results]
    return max(hps) <= cfg.wave_hp, f"B front hp {hps}"


def _check_two_turn_consumption(results, cfg):
    sieges = [r.state.asg for r in results]
    return sieges[:2] == [1, 0], f"A siege count after each turn: {sieges}"


def all_scenarios(cfg: GameConfig) -> List[Scenario]:
    W = cfg.wave_hp
    return [
        Scenario("reference",
                 "A two-stack wave (hp6) at +1 Slow Pushes; B's fresh wave arrives and Shoves.",
                 State(1, 3 * W, (), 0, -3, W, (), 0, 0, 0), 1, [(SP, SH)], _check_reference),
        Scenario("small_wave_pushed_back",
                 "Same as reference but A's wave is unstacked (hp2): it is destroyed and B pushes back.",
                 State(1, W, (), 0, -3, W, (), 0, 0, 0), 1, [(SP, SH)], _check_small_pushed_back),
        Scenario("freeze_holds",
                 "A freezes with no wave of its own; B's fresh wave (hp2) shoves from the river, then Slow Pushes.",
                 State(0, 0, (), 0, 0, W, (), 0, 0, 0), 1, [(F, SH), (F, SP), (F, SP)], _check_freeze_holds),
        Scenario("freeze_broken",
                 "As freeze_holds but B's wave carries one stack (hp4): the shove breaks the freeze.",
                 State(0, 0, (), 0, 0, 2 * W, (), 0, 0, 0), 1, [(F, SH)], _check_freeze_breaks),
        Scenario("slowpush_stacks",
                 "A parks a one-wave surplus (hp2) at the river and Slow Pushes against a passive Slow Push.",
                 State(0, W, (-4,), 0, -4, W, (), 0, 0, 0), 2, [(SP, SP)] * 8, _check_slowpush_stacks),
        Scenario("shove_contests_slowpush",
                 "As slowpush_stacks but B Shoves every incoming wave: the stack never builds.",
                 State(0, W, (-4,), 0, -4, W, (), 0, 0, 0), 2, [(SP, SH)] * 8, _check_shove_contests),
        Scenario("large_wave_crashes_through",
                 "A 2-stack wave at the river Shoves while B's fresh wave arrives at its tower.",
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


def run_scenario(sc: Scenario, cfg: GameConfig):
    s = sc.start
    for i, (a, b) in enumerate(sc.plan):
        assert a in legal_actions(s, 0, cfg) or i > 0, f"{sc.name}: illegal first action for A"
    lines, results = trace_plan(cfg, sc.start, sc.k0, sc.plan)
    ok, detail = sc.check(results, cfg)
    return ok, detail, lines, results
