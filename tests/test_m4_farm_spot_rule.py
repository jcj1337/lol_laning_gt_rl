"""M4.2d-A: the opt-in farm-spot correction `farm_spot_rule="pre_crash_front"` and the wave-bounce finding
(docs/archive/m4/m4_2/m4_wave_readiness_report.md)."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.engine.champions import N_CHANCE, mirror_chance, swap_seats
from lanegame.engine.champions_m4 import check_invariants, resolve_turn_m4
from lanegame.engine.config import F, SH, SP
from m4_scenarios import DESIGN_HOLD_TABLE, hold_run

PCF = GameConfig.m4(farm_spot_rule="pre_crash_front")


def test_default_is_the_m2_rule_and_the_field_is_validated():
    assert GameConfig.m4().farm_spot_rule == "m2" and GameConfig().farm_spot_rule == "m2"
    PCF.validate()
    with pytest.raises(ValueError, match="farm_spot_rule"):
        GameConfig.m4(farm_spot_rule="keep").validate()
    with pytest.raises(ValueError, match="farm_spot_rule"):
        GameConfig.champions(farm_spot_rule="pre_crash_front").validate()


def test_pre_crash_front_is_mirror_equivariant_bounded_and_fires_only_on_probe_crashes():
    from lanegame.env import LaneEnv, UniformAgent
    asym = PCF.with_(skill_a=0.8, skill_b=0.3)
    env = LaneEnv(asym, max_turns=600)
    rng, nrng = np.random.default_rng(0), np.random.default_rng(9)
    ag = UniformAgent()
    obs = env.reset(seed=3)
    fired, T = 0, asym.tower
    for _ in range(600):
        lane, m4, k = env.state, env.champ, env.k
        a, b = ag.act(obs[0], rng), ag.act(obs[1], rng)
        u = tuple(float(x) for x in nrng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k, a, b, asym, u=u)
        m = resolve_turn_m4(lane.mirror(), m4.mirror(), k, b, a, swap_seats(asym), u=mirror_chance(u))
        assert m.champ == t.champ.mirror() and m.lane.state == t.lane.state.mirror()
        assert m.farm_spot == t.farm_spot[::-1] and m.end == t.end[::-1]
        d = resolve_turn_m4(lane, m4, k, a, b, asym.with_(farm_spot_rule="m2"), u=u)
        for s in (0, 1):
            assert -T <= t.end[s] <= T and -T <= t.farm_spot[s] <= T
            if t.farm_spot[s] != d.farm_spot[s]:
                fired += 1
                # only a front that existed before the turn and crashes: the spot is its pre-turn cell
                assert lane.side(s)[1] > 0 and t.farm_spot[s] == max(-T, min(T, lane.side(s)[0]))
        check_invariants(t.champ, asym)
        r = env.step(a, b)
        obs = r.obs
        if r.done:
            break
    assert fired > 0


def test_pre_crash_front_meets_the_sh_x_gate_that_the_default_misses(monkeypatch):
    """The default keeps the strict expected failure of tests/test_m4_freeze_gate.py (6-9 turns); under the opt-in
    rule the presser breaks a 3-5 HP freeze within 6 turns, the design's §6.4 windows."""
    import tests.test_m4_freeze_gate as gate
    monkeypatch.setattr(gate, "M4", PCF)
    assert {e: [gate.parked(e, ph, "freeze", "presser")[1] for ph in range(4)] for e in (3, 4, 5)} == \
        {3: [6, 5, 6, 5], 4: [4, 3, 4, 3], 5: [4, 3, 4, 3]}
