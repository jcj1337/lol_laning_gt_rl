"""M4.2d integration: the opt-in experimental preset `GameConfig.m4_emergent()` (docs/m4_emergent_parallel_report.md).
Joint correctness checks (mirror symmetry, masks, replay, accounting, conservation, invariants, observations, zero-sum)
and pins of the joint feasibility findings."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import N_CHANCE, mirror_chance, swap_seats
from lanegame.champions_m4 import check_invariants, resolve_turn_m4
from lanegame.config import B_HOLD, B_YIELD, N_CHAMP_ACTIONS
from lanegame.env import LaneEnv, UniformAgent, legal_set, play_episode
from lanegame.env.agents_m4 import m4_suite
from lanegame.env.observation_m4 import decode_m4, m4_obs_size, observe_m4
from lanegame.experiments.m4 import positioning as po
from lanegame.experiments.m4 import wave_readiness as wr
from lanegame.experiments.m4.investment_readiness import READY

from test_m4_env import soak

EM = GameConfig.m4_emergent()


def test_preset_selects_the_integrated_mechanics_and_leaves_the_historical_defaults():
    EM.validate()
    assert EM.ruleset == "m4" and EM.continuous_items and EM.item_tiers == 0
    for f, v in READY.items():
        assert getattr(EM, f) == v
    assert (EM.position_rule, EM.zoning_rule, EM.farm_spot_rule) == ("standoff", "none", "pre_crash_front")
    assert EM.slowpush_mode == GameConfig.m4().slowpush_mode                 # waves keep the scripted M4 rule
    for cfg in (GameConfig.m4(), GameConfig.m4_continuous()):
        assert (cfg.position_rule, cfg.zoning_rule, cfg.farm_spot_rule, cfg.invest_curve,
                cfg.invest_obs_encoding) == ("m2", "presence", "m2", "power", "linear")
    assert GameConfig.m4_emergent(zoning_rule="presence").zoning_rule == "presence"


def test_soak_random_legal_play_zero_sum_conservation_and_invariants():
    soak(EM, 3000, 11)


def test_soak_scripted_pairs():
    suite = m4_suite()
    for j, (x, y) in enumerate((("gold_recaller", "aggressive_trader"), ("freezer", "presser"),
                                ("xp_keeper", "freeze_breaker"))):
        soak(EM, 600, 200 + j, (suite[x], suite[y]))


def test_mirror_symmetry_masks_observations_and_accounting_under_random_play():
    cfg = EM                                                                 # equal skills: swap_seats(cfg) == cfg
    assert swap_seats(cfg) == cfg
    env = LaneEnv(cfg, max_turns=800)
    rng, ag = np.random.default_rng(3), UniformAgent()
    obs = env.reset(seed=8)
    pinned = invested = 0
    for _ in range(800):
        lane, m4, k = env.state, env.champ, env.k
        for s in (0, 1):
            o = obs[s]
            assert len(o.vector) == m4_obs_size(cfg) == 88 and len(o.mask) == N_CHAMP_ACTIONS
            assert np.isfinite(o.vector).all() and o.vector.min() >= 0 and o.vector.max() <= 1
            assert set(np.flatnonzero(o.mask)) == set(o.legal) == set(legal_set(lane, s, cfg, m4))
            _, dl, dc, _ = decode_m4(o.vector, cfg)
            assert (dl, dc) == ((lane, m4) if s == 0 else (lane.mirror(), m4.mirror()))
        assert np.array_equal(obs[1].vector, observe_m4(lane.mirror(), k, 0, cfg, m4.mirror()))
        a, b = ag.act(obs[0], rng), ag.act(obs[1], rng)
        r = env.step(a, b)
        u = tuple(r.info["chance"])                                         # replay the env turn from its draws
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, u=u)
        m = resolve_turn_m4(lane.mirror(), m4.mirror(), k, b, a, cfg, u=mirror_chance(u))
        assert t.champ == env.champ and m.champ == t.champ.mirror() and m.reward_units == -t.reward_units
        i, cur = r.info, env.champ
        check_invariants(cur, cfg)
        for s in (0, 1):
            p, c = m4.side(s), cur.side(s)
            # gold: unspent + invested grows by exactly what was earned (no clamp under continuous investment)
            assert c.gold + c.items == p.gold + p.items + i["earned_units"][s]
            assert c.xp == p.xp + i["xp_gain"][s]
            pinned += i["pinned"][s]
            invested += i["invested"][s] > 0
        assert r.rewards[0] == -r.rewards[1] == i["reward_units"] / cfg.gold_scale
        obs = r.obs
        if r.done:
            break
    assert pinned > 0 and invested > 0


def test_deterministic_replay():
    suite = m4_suite()
    logs = [play_episode(LaneEnv(EM, max_turns=250, cache_size=c), suite["gold_recaller"], suite["aggressive_trader"],
                         41, record=True) for c in (0, 5_000)]
    assert logs[0].rewards == logs[1].rewards and logs[0].actions == logs[1].actions
    assert [t["champ_after"] for t in logs[0].turns] == [t["champ_after"] for t in logs[1].turns]


@pytest.mark.parametrize("seed", [0, 1])
def test_one_turn_mirror_with_unequal_skills(seed):
    cfg = GameConfig.m4_emergent(skill_a=0.7, skill_b=0.35)
    env = LaneEnv(cfg, max_turns=300)
    rng, nrng, ag = np.random.default_rng(seed), np.random.default_rng(50 + seed), UniformAgent()
    obs = env.reset(seed=seed)
    for _ in range(300):
        lane, m4, k = env.state, env.champ, env.k
        a, b = ag.act(obs[0], rng), ag.act(obs[1], rng)
        u = tuple(float(x) for x in nrng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, u=u)
        m = resolve_turn_m4(lane.mirror(), m4.mirror(), k, b, a, swap_seats(cfg), u=mirror_chance(u))
        assert m.champ == t.champ.mirror() and m.lane.state == t.lane.state.mirror()
        assert m.pinned == t.pinned[::-1] and m.reward_units == -t.reward_units
        r = env.step(a, b)
        if r.done:
            break
        obs = r.obs


# ---------------------------------------------------------------------- pins of the joint findings
def test_findings_wave_stacks_and_crashes_without_a_bounce():
    st = wr.starts(EM)
    rows = wr.play(EM, wr.Pusher(shove=False), wr.hold, 48, *st["pushed"])
    s = wr.summarize(rows, EM)
    assert max(r["t"].lane.state.ah for r in rows) >= 9 and [c["hp"] for c in s["crashes"][0]][:3] == [3, 9, 7]
    assert not s["bounce"]
    ctrl = wr.play(EM, wr.Pusher(8), wr.control, 48, *st["pushed"])          # the defender can prevent the crash
    assert wr.summarize(ctrl, EM)["crashes"][0] == [] and max(r["t"].lane.state.ah for r in ctrl) <= EM.wave_hp


def test_findings_forward_standoff_is_pinned_and_power_scales_the_threat():
    lane, m4, k = wr.end_state(wr.play(EM, wr.hold, wr.hold, 9))
    g = EM.gold_scale
    line = lambda cfg, m: po.summarize(po.play(cfg, po.periodic_trade(4), po.fixed(B_YIELD), po.W, lane, m, k), cfg)
    eq = line(EM, m4)
    big = line(EM, m4.replace_side(0, items=50 * g))
    old = line(GameConfig.m4_emergent(position_rule="m2"), m4)
    assert eq["trades_connected"][0] == big["trades_connected"][0] == 6 and old["trades_connected"][0] == 0
    assert big["expected_damage_taken"][1] > 1.5 * eq["expected_damage_taken"][1]
    hold = po.summarize(po.play(EM, po.fixed(B_HOLD), po.respect, po.W, lane, m4, k), EM)
    assert hold["trades_connected"] == [0, 0] and hold["farm_per_turn"][0] > hold["farm_per_turn"][1]
