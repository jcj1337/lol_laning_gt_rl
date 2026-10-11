"""WP4: the M4 resolution skeleton (design §5 steps 0-10, §6.3, §7.4; plan WP4)."""

import random

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import (CHANNEL, N_CHANCE, TRADING, ChampSide, ChampState, champion_legal_actions,
                                in_contact, initial_champions, mirror_chance, resolve_turn, swap_seats)
from lanegame.champions_m4 import (M4Side, M4State, contact_cell_target, initial_m4, m2_chase_target,
                                   resolve_turn_m4)
from lanegame.config import B_HOLD, B_SAFE, B_TRADE, CHAMP_ABSENT, F, NONE, SH, SP, champ_act, champ_behavior, \
    champ_objective, champ_wave
from lanegame.mechanics import initial_state, step_turn
from lanegame.progression import hp_max
from lanegame.state import State

M2 = GameConfig.champions()
M4 = GameConfig.m4()
# M4 with every rule that changes M2's resolution switched off (the economy runs but does not feed back in one turn)
M4_OFF = GameConfig.m4(chase_rule="m2", freeze_maintenance=False, tower_adjacent_p=0.0, tower_adjacent_p_busy=0.0)
T = M4.tower


def H(w=SP, o=NONE):
    return champ_act(w, B_HOLD, o)


def TR(w=SP, o=NONE):
    return champ_act(w, B_TRADE, o)


def to_m4(champ: ChampState) -> M4State:
    return M4State(*(M4Side(s.pos, 100 * s.hp, s.absent) for s in (champ.a, champ.b)))


def m2_tuples(n, seed):
    """(lane, champ, k, a, b, u) from seeded random legal play of the M2 champion game (the M2 state space)."""
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    lane, champ, k = initial_state(M2), initial_champions(M2), 0
    for _ in range(n):
        a = rng.choice(champion_legal_actions(lane, champ, 0, M2))
        b = rng.choice(champion_legal_actions(lane, champ, 1, M2))
        u = tuple(float(x) for x in nrng.random(N_CHANCE))
        yield lane, champ, k, a, b, u
        t = resolve_turn(lane, champ, k, a, b, M2, u=u)
        lane, champ, k = t.lane.state, t.champ, k + 1
        if k >= M2.horizon:
            lane, champ, k = initial_state(M2), initial_champions(M2), 0


def same_resolution(t2, t4, check_hp=True, ignore=()):
    """M4 turn t4 (level 1, no items) reproduces M2 turn t2, with HP scaled by 100."""
    for f in ("behaviors", "objectives", "effective_objectives", "start", "end", "farm_spot", "away", "recalled",
              "interrupted", "threatened", "connected", "modes", "reach", "lane_codes", "exposure", "gank_p", "hit_p",
              "double_p", "tower_p", "ganked", "gank_damage_p", "died"):
        if f not in ignore and getattr(t2, f) != getattr(t4, f):
            return False
    # crash-plate opportunities (ap / bp) exist only under M2's "crash" plate mode; M4 uses durability plates
    if t2.lane.state._replace(ap=0, bp=0) != t4.lane.state or t2.lane.kills != t4.lane.kills:
        return False
    if not np.allclose(t2.farm, t4.farm, atol=1e-12) or t2.farm_available != t4.farm_available:
        return False
    if tuple(100 * x for x in t2.trade_damage) != t4.trade_damage:
        return False
    if tuple(100 * x for x in t2.damage_taken) != t4.damage_taken or tuple(100 * x for x in t2.gank_damage) != t4.gank_damage:
        return False
    if t2.bounty != t4.bounty:
        return False
    for s in (0, 1):
        c2, c4 = t2.champ.side(s), t4.champ.side(s)
        if (c2.pos, c2.absent) != (c4.pos, c4.absent):
            return False
        if check_hp:
            gain = 0 if (t4.away[s] or t4.died[s]) else hp_max(t4.levels_after[s]) - hp_max(t4.levels_before[s])
            if c4.hp - gain != 100 * c2.hp:
                return False
    return True


def test_m4_resolver_with_m2_rules_reproduces_m2_exactly():
    for lane, champ, k, a, b, u in m2_tuples(5000, 1):
        t2 = resolve_turn(lane, champ, k, a, b, M2, u=u)
        t4 = resolve_turn_m4(lane._replace(ap=0, bp=0), to_m4(champ), k, a, b, M4_OFF, u=u)
        assert same_resolution(t2, t4), (lane, champ, k, a, b, u)


def chasers(lane, champ, a, b):
    """Trade champions without an objective whose opponent is present (design §7.4)."""
    out = []
    for s, (x, y) in enumerate(((a, b), (b, a))):
        if x != CHAMP_ABSENT and champ_behavior(x) == B_TRADE and champ_objective(x) == NONE and y != CHAMP_ABSENT:
            out.append(s)
    return out


def test_contact_cell_chase_differences_are_exactly_the_two_documented_cases():
    cfg = M4_OFF.with_(chase_rule="contact_cell")
    diffs, cases = 0, {"own tower cell": 0, "enemy tower cell": 0}
    for lane, champ, k, a, b, u in m2_tuples(5000, 2):
        t2 = resolve_turn(lane, champ, k, a, b, M2, u=u)
        t4 = resolve_turn_m4(lane._replace(ap=0, bp=0), to_m4(champ), k, a, b, cfg, u=u)
        if same_resolution(t2, t4):
            continue
        diffs += 1
        # explain the difference: some chaser's M2 target is a cell from which contact with its reference is
        # impossible, and the contact-cell target differs from it
        explained = False
        beh = (champ_behavior(a) if a != CHAMP_ABSENT else -1, champ_behavior(b) if b != CHAMP_ABSENT else -1)
        obj = (champ_objective(a) if a != CHAMP_ABSENT else 0, champ_objective(b) if b != CHAMP_ABSENT else 0)
        for s in chasers(lane, champ, a, b):
            o = 1 - s
            ref = champ.side(o).pos if (beh[o] == B_TRADE and obj[o] == NONE) else t2.end[o]
            c = champ.side(s).pos
            m2t, cct = m2_chase_target(c, ref, cfg), contact_cell_target(c, ref, cfg)
            if m2t != cct:
                assert not in_contact(m2t, ref, cfg) and in_contact(cct, ref, cfg)
                assert m2t in (-T, T), "M2's target was not a tower cell: unexplained difference"
                cases["own tower cell" if m2t == -T else "enemy tower cell"] += 1
                explained = True
        assert explained, (lane, champ, k, a, b)
    assert diffs > 0 and cases["own tower cell"] > 0


def test_freeze_maintenance_differences_are_unmaintained_freezes_resolved_as_slow_push():
    cfg = M4_OFF.with_(freeze_maintenance=True)
    diffs = 0
    for lane, champ, k, a, b, u in m2_tuples(8000, 3):
        t2 = resolve_turn(lane, champ, k, a, b, M2, u=u)
        t4 = resolve_turn_m4(lane._replace(ap=0, bp=0), to_m4(champ), k, a, b, cfg, u=u)
        if same_resolution(t2, t4):
            continue
        diffs += 1
        unmaint = [s for s in (0, 1) if t4.waves[s] == F and not t4.freeze_maintained[s]]
        assert unmaint, (lane, champ, k, a, b)
        assert all(t4.effective_waves[s] == SP for s in unmaint)
        # the M2 resolution with Slow Push in the authoritative lane step only (the probe keeps the raw Freeze)
        calls = []

        def lane_step(st, kk, x, y, reach):
            calls.append(1)
            if len(calls) == 2:
                x = x - F + SP if 0 in unmaint and x % 3 == F else x
                y = y - F + SP if 1 in unmaint and y % 3 == F else y
            return step_turn(st, kk, x, y, M2, champion_reach=reach)
        t2b = resolve_turn(lane, champ, k, a, b, M2, lane_step=lane_step, u=u)
        # M2 reports the raw (Freeze) lane code it was given; M4 reports the effective (Slow Push) code it played
        assert same_resolution(t2b, t4, ignore=("lane_codes",)), (lane, champ, k, a, b)
        assert all(t4.lane_codes[s] % 3 == SP for s in unmaint)
    assert diffs > 0


def test_chase_from_the_tower_cell_connects_with_an_enemy_next_to_it():
    # B's 6 HP front parked at A's -1 (B own +1); A has no front. B holds at its farm spot (its own +1, A's -1).
    lane = State(0, 0, (), 0, 1, 6, (), 0, 0, 0, 0, 0)
    champ = M4State(M4Side(-T, 600), M4Side(1, 600))
    t = resolve_turn_m4(lane, champ, 1, TR(), H(), M4)
    assert t.end == (-1, 1) and t.connected == (True, False)
    t_m2 = resolve_turn_m4(lane, champ, 1, TR(), H(), M4.with_(chase_rule="m2"))
    assert t_m2.end[0] == -T and t_m2.connected == (False, False)          # the M2 defect (design §1.5)


def parked(e=6):
    """B's front with e HP at A's -1 (B own +1), A's wave just killed (no front), both champions present."""
    return State(0, 0, (), 0, 1, e, (), 0, 0, 0, 0, 0)


def held(t):
    return t.lane.events["freeze_held"][1] > 0


def test_maintained_freeze_holds_and_unmaintained_freeze_does_not():
    lane = parked(6)
    k = 0                                                   # a creep tick for B's stacked Slow Push front
    sp_only = step_turn(lane, k, SP, SP, M4, champion_reach=(T, T))
    assert sp_only.crash_hp[1] > 0                         # without a Freeze it crashes
    # maintained: A at -1 holding its freeze line
    t = resolve_turn_m4(lane, M4State(M4Side(-1, 600), M4Side(1, 600)), k, H(F), H(), M4)
    assert t.freeze_maintained[0] and t.effective_waves[0] == F and held(t) and t.lane.crash_hp[1] == 0
    # two cells from the freeze line (end cell +1): resolved as Slow Push, the front crashes
    far = M4State(M4Side(1, 600), M4Side(1, 600))
    t2 = resolve_turn_m4(lane, far, k, H(F), H(), M4)
    assert t2.end[0] == 1 and not t2.freeze_maintained[0] and t2.effective_waves[0] == SP and t2.lane.crash_hp[1] > 0
    # the same freezer under the ablation freeze_maintenance=False still holds (M2's unconditional hold)
    t3 = resolve_turn_m4(lane, far, k, H(F), H(), M4.with_(freeze_maintenance=False))
    assert held(t3) and t3.lane.crash_hp[1] == 0
    # a connected-trading freezer: B trades A and A trades back from -1 -> A is connected, its freeze lapses
    t4 = resolve_turn_m4(lane, M4State(M4Side(-1, 600), M4Side(1, 600)), k, TR(F), TR(), M4)
    assert t4.connected[0] and not t4.freeze_maintained[0] and t4.effective_waves[0] == SP and t4.lane.crash_hp[1] > 0
    # being traded (not trading back) does not break the freeze
    t5 = resolve_turn_m4(lane, M4State(M4Side(-1, 600), M4Side(1, 600)), k, H(F), TR(), M4)
    assert t5.connected == (False, True) and t5.freeze_maintained[0] and t5.lane.crash_hp[1] == 0


def test_an_absent_champion_cannot_freeze():
    lane = parked(6)
    champ = M4State(M4Side(-T, 600, 2), M4Side(1, 600))
    assert champion_legal_actions(lane, champ, 0, M4) == (CHAMP_ABSENT,)
    t = resolve_turn_m4(lane, champ, 0, CHAMP_ABSENT, H(), M4)
    assert t.effective_waves[0] == SP and not t.freeze_maintained[0] and t.lane.crash_hp[1] > 0


def test_resolver_is_mirror_equivariant_under_mirror_chance():
    asym = GameConfig.m4(skill_a=0.8, skill_b=0.3)
    from lanegame.env import LaneEnv, UniformAgent
    env = LaneEnv(asym, max_turns=1500)
    rng = np.random.default_rng(0)
    ag = UniformAgent()
    obs = env.reset(seed=3)
    nrng = np.random.default_rng(9)
    for _ in range(1500):
        lane, m4, k = env.state, env.champ, env.k
        a, b = ag.act(obs[0], rng), ag.act(obs[1], rng)
        u = tuple(float(x) for x in nrng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k, a, b, asym, u=u)
        m = resolve_turn_m4(lane.mirror(), m4.mirror(), k, b, a, swap_seats(asym), u=mirror_chance(u))
        assert m.champ == t.champ.mirror() and m.lane.state == t.lane.state.mirror()
        for f in t.__dataclass_fields__:
            v = getattr(t, f)
            if isinstance(v, tuple) and len(v) == 2 and f not in ("lane",):
                assert getattr(m, f) == v[::-1], f
        assert m.reward_units == -t.reward_units
        r = env.step(a, b)
        obs = r.obs
        if r.done:
            break
