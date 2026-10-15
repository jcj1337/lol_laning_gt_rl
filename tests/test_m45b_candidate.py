"""M4.5b candidate `GameConfig.m45_calibrated()` (docs/m4/m4_5b_calibration_report.md): `m45_unified()` revision 2 with
the mechanical correction farm_spot_rule "last_front" and the calibration minion_retaliation_delay 0.

Tests: the parameter contract (exactly the two fields; validation; every earlier preset unchanged), the correction's
effect on the demonstrated defect (mutual Shove, walking fronts) and its scope (identical results whenever no front is
lost in the probe), retaliation from the exchange tick, mirror equivariance, an environment soak with the observation
decoder, deterministic replay, the viewer and the checkpoint guard.
"""

import hashlib
import json

import numpy as np
import pytest

from lanegame.champions import N_CHANCE, U_TRADE_DOUBLE, U_TRADE_HIT, mirror_chance
from lanegame.champions_m4 import _last_front_cell, check_invariants, initial_m4, resolve_turn_m4
from lanegame.config import B_HOLD, B_TRADE, SH, W_HOLD, GameConfig, champ_act
from lanegame.env import LaneEnv
from lanegame.env.observation_m4 import decode_m4
from lanegame.experiments.m45b import states as S
from lanegame.experiments.m45b import timing as TM
from lanegame.positional import WARD
from lanegame.state import State

CAL, R2 = GameConfig.m45_calibrated(), GameConfig.m45_unified()
SHOVE_HOLD = champ_act(SH, B_HOLD)
LH_TRADE, LH_HOLD = champ_act(W_HOLD, B_TRADE), champ_act(W_HOLD, B_HOLD)


# ---------------------------------------------------------------------------------------------- contract
def test_candidate_differs_from_revision_2_in_the_two_fields_only():
    CAL.validate()
    a, b = CAL.to_dict(), R2.to_dict()
    assert sorted(k for k in a if a[k] != b[k]) == sorted(GameConfig.M45_CALIBRATED_FIELDS)
    assert (CAL.farm_spot_rule, CAL.minion_retaliation_delay) == ("last_front", 0)
    assert GameConfig.from_dict(CAL.to_dict()) == CAL


@pytest.mark.parametrize("name", ["m45_unified", "m45_unified_r1", "m4_four_actions_econ_aggro", "m4_lasthit",
                                  "m4_natural"])
def test_earlier_presets_keep_their_farm_spot_rule_and_delay(name):
    c = getattr(GameConfig, name)()
    assert c.farm_spot_rule == "pre_crash_front"
    assert c.minion_retaliation_delay == 1


def test_last_front_validation():
    with pytest.raises(ValueError):                           # needs the probe's per-tick fronts
        GameConfig.m45_unified(farm_spot_rule="last_front", champion_movement_rule="turn").validate()
    with pytest.raises(ValueError):
        GameConfig.m4(farm_spot_rule="last_front").validate()
    with pytest.raises(ValueError):
        GameConfig.champions(farm_spot_rule="last_front").validate()
    with pytest.raises(ValueError):
        GameConfig.m45_unified(farm_spot_rule="nearest").validate()


def test_spaces_are_unchanged():
    env = LaneEnv(CAL)
    assert (env.obs_size, env.n_actions) == (90, 61)


# ---------------------------------------------------------------------------------------------- the correction
def _champs_at(cfg, pa, pb, level=2):
    return S.champs(cfg, (level, level), (0.0, 0.0), (pa, pb))


def test_mutual_shove_no_longer_walks_both_champions_out_of_reach():
    """The demonstrated defect: both 4-HP fronts and both champions at the lane cell, both Shove. The probe has both
    fronts die, so under "pre_crash_front" both champions walked back to their next packets at -2, out of reach of the
    fight: the fronts survived and nobody collected anything. Under "last_front" both stay with the fight and farm."""
    lane = State(0, 4, (-4,), 0, 0, 4, (-4,), 0, 0, 0)
    m4 = _champs_at(CAL, 0, 0)
    old = resolve_turn_m4(lane, m4, 2, SHOVE_HOLD, SHOVE_HOLD, R2)
    new = resolve_turn_m4(lane, m4, 2, SHOVE_HOLD, SHOVE_HOLD, CAL)
    assert old.end == (-2, -2) and old.farm == (0.0, 0.0)
    assert new.end == (0, 0) and min(new.farm) > 0


def test_walking_fronts_keep_the_champions_within_reach_of_their_fight():
    """Two fresh 8-HP fronts at their own -2 meet at the lane cell on tick 1, where the probe has both die."""
    lane = State(-2, 8, (), 0, -2, 8, (), 0, 0, 0)
    t = resolve_turn_m4(lane, _champs_at(CAL, 0, 0), 2, SHOVE_HOLD, SHOVE_HOLD, CAL)
    for s in (0, 1):
        assert t.end[s] >= -1 and min(t.farm) > 0


def test_last_front_cell_rule():
    lane = State(0, 4, (-4,), 0, 0, 4, (-4,), 0, 0, 0)
    pr = TM.probe(CAL, lane, _champs_at(CAL, 0, 0), 2, SHOVE_HOLD, SHOVE_HOLD)
    assert _last_front_cell(lane, pr, 0) == 0 and _last_front_cell(lane, pr, 1) == 0
    lane2 = State(0, 16, (-2,), 0, 0, 2, (), 0, 0, 0)                 # A's front survives and advances
    pr2 = TM.probe(CAL, lane2, _champs_at(CAL, 0, 0), 2, LH_HOLD, LH_HOLD)
    assert _last_front_cell(lane2, pr2, 0) is None


def test_correction_changes_nothing_when_no_front_is_lost():
    """On random turns where neither side's pre-turn front crashes or dies in the probe, "last_front" and
    "pre_crash_front" resolve identically (same delay)."""
    R2b = GameConfig.m45_unified(minion_retaliation_delay=0)
    env = LaneEnv(CAL, max_turns=10 ** 6)
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(600):
        lane, m4, k = env.state, env.champ, env.k
        a, b = int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1)))
        u = tuple(float(x) for x in rng.random(N_CHANCE))
        pr = TM.probe(CAL, lane, m4, k, a, b)
        if all(_last_front_cell(lane, pr, s) is None for s in (0, 1)):
            t1 = resolve_turn_m4(lane, m4, k, a, b, CAL, u=u)
            t2 = resolve_turn_m4(lane, m4, k, a, b, R2b, u=u)
            assert t1.champ == t2.champ and t1.lane.state == t2.lane.state and t1.reward_units == t2.reward_units
            checked += 1
        env.step(a, b)
    assert checked > 300


# ---------------------------------------------------------------------------------------------- retaliation timing
def _hit_u():
    u = [1.0] * N_CHANCE
    u[U_TRADE_HIT[0]] = 0.0
    u[U_TRADE_DOUBLE[0]] = 0.0
    return tuple(u)


def test_retaliation_fires_from_the_exchange_tick():
    lane, m4, k = S.trade_bench(CAL, 2, 24, "cover", "equal")
    t_cal = resolve_turn_m4(lane, m4, k, LH_TRADE, LH_HOLD, CAL, u=_hit_u())
    t_r2 = resolve_turn_m4(lane, m4, k, LH_TRADE, LH_HOLD, R2, u=_hit_u())
    assert t_cal.trade_damage[0] > 0 and t_cal.retaliation == (False, True)
    assert t_cal.minion_fire[0] == ("retaliation", "retaliation")
    assert t_r2.minion_fire[0] == ("", "retaliation")
    assert t_cal.minion_hits[0] > t_r2.minion_hits[0]


def test_a_missed_trade_still_draws_no_retaliation():
    lane, m4, k = S.trade_bench(CAL, 2, 24, "cover", "equal")
    t = resolve_turn_m4(lane, m4, k, LH_TRADE, LH_HOLD, CAL, u=(1.0,) * N_CHANCE)
    assert t.connected[0] and t.trade_damage[0] == 0 and t.minion_hits[0] == 0


# ---------------------------------------------------------------------------------------------- integration
def _random_states(n, seed, cfg=CAL):
    rng = np.random.default_rng(seed)
    env = LaneEnv(cfg, max_turns=10 ** 6)
    env.reset(seed=seed)
    out = []
    while len(out) < n:
        a, b = int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1)))
        out.append((env.state, env.champ, env.k, env.legal(0), env.legal(1)))
        env.step(a, b)
    return out, rng


def test_candidate_is_mirror_equivariant_on_random_states():
    states, rng = _random_states(400, 21)
    for lane, m4, k, la, lb in states:
        a, b = int(rng.choice(la)), int(rng.choice(lb))
        u = tuple(float(x) for x in rng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k, a, b, CAL, u=u)
        tm = resolve_turn_m4(lane.mirror(), m4.mirror(), k, b, a, CAL, u=mirror_chance(u))
        assert tm.champ == t.champ.mirror() and tm.lane.state == t.lane.state.mirror()
        assert tm.reward_units == -t.reward_units and tm.end == t.end[::-1] and tm.farm_spot == t.farm_spot[::-1]
        assert tm.minion_fire == (t.minion_fire[::-1] if t.minion_fire else None)


def test_candidate_env_soak():
    env = LaneEnv(CAL, max_turns=300)
    rng = np.random.default_rng(8)
    for g in range(4):
        env.reset(seed=g)
        while not env.done:
            ma, mb = env.masks()
            assert set(np.flatnonzero(ma)) == set(env.legal(0)) and set(np.flatnonzero(mb)) == set(env.legal(1))
            r = env.step(int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1))))
            assert r.rewards[1] == -r.rewards[0]
            check_invariants(env.champ, CAL)
            for s in (0, 1):
                _, ln, m4, _ = decode_m4(r.obs[s].vector, CAL)
                me = env.champ.side(s)
                assert (m4.a.pos, m4.a.hp) == (me.pos, me.hp)


def test_candidate_seeded_replay_is_deterministic():
    def run():
        env = LaneEnv(CAL, max_turns=120, record_ticks=True)
        env.reset(seed=5)
        rng = np.random.default_rng(5)
        h = hashlib.sha256()
        while not env.done:
            r = env.step(int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1))))
            h.update(json.dumps(r.info, sort_keys=True, default=repr).encode())
        return h.hexdigest()
    assert run() == run()


def test_viewer_replays_the_candidate():
    from lanegame.experiments.m45b import strategies as ST
    from lanegame.viewer import session as ss
    assert "m45_calibrated" in ss.CONFIGS
    setup = ss.Setup("m45_calibrated", "m45b_freeze_10_large_h12", "equal", 87_900)
    rec = ss.play(setup, policies=(ST.get("fz_guard_sh"), ST.get("wk_shove_sh")), turns=20)
    rec2 = ss.play(setup, prefix=rec.actions)
    assert [t.info["reward"] for t in rec.turns] == [t.info["reward"] for t in rec2.turns]


def test_checkpoint_guard_knows_the_candidate():
    from lanegame.experiments.rl.m44.common import preset_name
    assert preset_name(CAL) == "m45_calibrated" and preset_name(R2) == "m45_unified"


# ---------------------------------------------------------------------------------------------- pins
def _digest(name):
    h = hashlib.sha256()
    cfg = getattr(GameConfig, name)()
    for g in range(3):
        env = LaneEnv(cfg, max_turns=150, record_ticks=True)
        env.reset(seed=1000 + g)
        rng = np.random.default_rng(2000 + g)
        while not env.done:
            r = env.step(int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1))))
            h.update(json.dumps(r.info, sort_keys=True, default=repr).encode())
            h.update(r.obs[0].vector.tobytes())
            h.update(r.obs[1].vector.tobytes())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("name,digest", [("m45_unified", "6eda1d44f7ef28af"),
                                         ("m4_four_actions_econ_aggro", "0d2fbd9f972838fc")])
def test_earlier_presets_reproduce_the_committed_engine(name, digest):
    """Digests computed with the committed engine (8dc5acf, before M4.5b): turn records, tick traces and observations
    of random games are unchanged (results/m45b/compatibility.json covers 14 presets)."""
    assert _digest(name) == digest
