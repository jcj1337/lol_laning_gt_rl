"""M4.2d-B: the opt-in positioning rule `GameConfig.position_rule` ("m2" default, "pin", "standoff";
docs/m4/m4_positioning_report.md)."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import N_CHANCE, in_contact, mirror_chance, swap_seats
from lanegame.champions_m4 import check_invariants, resolve_turn_m4, yield_target
from lanegame.config import B_HOLD, B_SAFE, B_TRADE, B_YIELD, N_CHAMP_ACTIONS, SP, champ_act
from lanegame.env.observation_m4 import m4_obs_size
from lanegame.experiments.m4.wave_readiness import end_state, hold, play

M2, PIN, STANDOFF = (GameConfig.m4(position_rule=r) for r in ("m2", "pin", "standoff"))
H, Y, T_, S = (champ_act(SP, b) for b in (B_HOLD, B_YIELD, B_TRADE, B_SAFE))


def neutral(cfg):
    """s0 after 9 turns of mutual Hold: both fronts and both champions at the lane (cell 0)."""
    return end_state(play(cfg, hold, hold, 9))


def turn(cfg, st, a, b):
    lane, m4, k = st
    return resolve_turn_m4(lane, m4, k, a, b, cfg, u=None)


def test_default_is_m2_and_the_field_is_validated():
    assert GameConfig.m4().position_rule == "m2" and GameConfig().position_rule == "m2"
    PIN.validate()
    STANDOFF.validate()
    with pytest.raises(ValueError, match="position_rule"):
        GameConfig.m4(position_rule="forward").validate()
    with pytest.raises(ValueError, match="position_rule"):
        GameConfig.champions(position_rule="standoff").validate()


def test_action_space_masks_and_observation_are_unchanged():
    st = neutral(M2)
    from lanegame.champions_m4 import champion_legal_actions
    for cfg in (PIN, STANDOFF):
        assert m4_obs_size(cfg) == m4_obs_size(M2) and N_CHAMP_ACTIONS == 46
        for s in (0, 1):
            assert champion_legal_actions(st[0], st[1], s, cfg) == champion_legal_actions(st[0], st[1], s, M2)


def test_rules_are_inert_without_a_yield():
    """Only Yield is affected: every (non-Yield, non-Yield) pair resolves identically under the three rules."""
    st = neutral(M2)
    for a in (H, T_, S):
        for b in (H, T_, S):
            ref = turn(M2, st, a, b)
            for cfg in (PIN, STANDOFF):
                t = turn(cfg, st, a, b)
                assert t.champ == ref.champ and t.end == ref.end and t.farm_units == ref.farm_units
                assert t.lane.state == ref.lane.state and t.hit_p == ref.hit_p and t.pinned == (False, False)


def test_m2_yield_evades_but_a_pinned_yield_is_traded_like_hold():
    st = neutral(M2)
    assert st[1].a.pos == st[1].b.pos == 0                     # start in contact
    m2 = turn(M2, st, T_, Y)
    assert m2.threatened[1] and not m2.connected[0] and m2.end[1] == -2 and m2.pinned == (False, False)
    vs_hold = turn(M2, st, T_, H)
    for cfg in (PIN, STANDOFF):
        st2 = neutral(cfg)
        t = turn(cfg, st2, T_, Y)
        assert t.pinned == (False, True) and not t.threatened[1] and t.connected[0]
        assert t.hit_p[0] == vs_hold.hit_p[0]                  # caught farming: Hold's hit probability
        assert t.modes[1] == "farm"
        # Safe is never pinned: it still evades from contact
        s = turn(cfg, st2, T_, S)
        assert s.threatened[1] and not s.connected[0] and s.pinned == (False, False)


def test_pin_alone_keeps_yield_at_its_farm_spot():
    st = neutral(PIN)
    t = turn(PIN, st, H, Y)
    assert t.end == (0, 0)


def test_standoff_cells_contact_and_unpinned_evasion():
    lane, m4, k = neutral(STANDOFF)
    t = turn(STANDOFF, (lane, m4, k), Y, Y)
    assert t.end == (-1, -1) and not in_contact(-1, -1, STANDOFF)          # mutual stand-off: out of contact
    assert t.farm_units == turn(M2, neutral(M2), H, H).farm_units         # ... at no farm cost at the lane
    st = (t.lane.state, t.champ, k + 1)
    h = turn(STANDOFF, st, H, Y)
    assert h.end == (0, -1) and in_contact(0, -1, STANDOFF)               # stepping up to the wave threatens
    tr = turn(STANDOFF, st, T_, Y)                                         # a Trade from the stand-off is evadable
    assert tr.threatened[1] and not tr.connected[0] and tr.pinned == (False, False) and tr.end == (0, -2)
    # next turn from (0, -1): the Yield starts in contact and is pinned
    st2 = (h.lane.state, h.champ, k + 2)
    p = turn(STANDOFF, st2, T_, Y)
    assert p.pinned == (False, True) and p.connected[0] and p.end[1] == -1


def test_standoff_cell_is_bounded_and_never_ahead_of_the_spot():
    lane, m4, k = neutral(STANDOFF)
    for spot in range(-2, 3):
        for s in (0, 1):
            y = yield_target(spot, lane, s, STANDOFF)
            assert -STANDOFF.tower <= y <= spot
            assert yield_target(spot, lane, s, M2) == spot == yield_target(spot, lane, s, PIN)


@pytest.mark.parametrize("rule", ["pin", "standoff"])
def test_mirror_symmetry_speed_and_invariants_under_random_play(rule):
    from lanegame.env import LaneEnv, UniformAgent
    cfg = GameConfig.m4(position_rule=rule, skill_a=0.8, skill_b=0.3)
    env = LaneEnv(cfg, max_turns=400)
    rng, nrng = np.random.default_rng(1), np.random.default_rng(5)
    ag = UniformAgent()
    obs = env.reset(seed=4)
    T, pinned = cfg.tower, 0
    for _ in range(400):
        lane, m4, k = env.state, env.champ, env.k
        a, b = ag.act(obs[0], rng), ag.act(obs[1], rng)
        u = tuple(float(x) for x in nrng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, u=u)
        m = resolve_turn_m4(lane.mirror(), m4.mirror(), k, b, a, swap_seats(cfg), u=mirror_chance(u))
        assert m.champ == t.champ.mirror() and m.lane.state == t.lane.state.mirror()
        assert m.end == t.end[::-1] and m.pinned == t.pinned[::-1] and m.reward_units == -t.reward_units
        check_invariants(t.champ, cfg)
        for s in (0, 1):
            o = 1 - s
            assert -T <= t.end[s] <= T
            if not t.away[s] and m4.side(s).absent == 0:
                assert abs(t.end[s] - t.start[s]) <= cfg.champion_speed
            if t.pinned[s]:
                pinned += 1
                assert t.behaviors[s] == B_YIELD and t.behaviors[o] == B_TRADE and t.connected[o]
                assert in_contact(t.start[s], t.start[o], cfg) and not t.threatened[s]
        r = env.step(a, b)
        assert r.info["pinned"] == t.pinned
        obs = r.obs
        if r.done:
            break
    assert pinned > 0
