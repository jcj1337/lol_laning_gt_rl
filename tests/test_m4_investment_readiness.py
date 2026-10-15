"""M4.2d-D investment readiness (docs/m4/m4_2/m4_investment_readiness_report.md): the opt-in bounded investment curve
(invest_curve = "saturating") and the opt-in compressed observation entries (invest_obs_encoding = "compressed").
The defaults of GameConfig.m4() and GameConfig.m4_continuous() are unchanged."""

from decimal import Decimal, getcontext

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame import progression as pg
from lanegame.champions import N_CHANCE
from lanegame.champions_m4 import M4Side, M4State, check_invariants, resolve_turn_m4
from lanegame.config import B_HOLD, CHAMP_ABSENT, SP, champ_act
from lanegame.env import LaneEnv
from lanegame.env.agents_m4 import m4_suite
from lanegame.env.observation_m4 import compressed_exact_max, decode_m4, m4_feature_names, m4_obs_size, observe_m4
from lanegame.experiments.m4.investment_readiness import READY
from lanegame.state import State

R = GameConfig.m4_continuous(**READY)          # the primary recommendation
P = GameConfig.m4_continuous()                 # the M4.2c defaults (power curve, linear encoding)
INV = pg.investment(R)
MU = R.gold_scale
T = R.tower
U0 = (1.0,) * N_CHANCE
HO = champ_act(SP, B_HOLD)
LANE = State(0, 4, (), 0, 0, 4, (), 0, 0, 0, 0, 0)


def side(cfg=R, pos=0, hp=None, xp=0, invested=0, absent=0, gold=0, tau=0):
    t = pg.tables(cfg)
    return M4Side(pos, t.hp_max[pg.level_of(xp, t)] if hp is None else hp, absent, xp, invested, gold, tau)


# ---------------------------------------------------------------------- configuration
def test_defaults_are_unchanged_and_the_recommendation_validates():
    for f, v in (("invest_curve", "power"), ("invest_obs_encoding", "linear"), ("invest_alpha", 0.045),
                 ("invest_p", 0.90), ("invest_obs_ref", 1024), ("invest_obs_gold_ref", 1024)):
        assert getattr(P, f) == v and getattr(GameConfig.m4(), f) == v
    R.validate()
    assert R.continuous_items and R.invest_curve == "saturating" and R.invest_obs_encoding == "compressed"
    assert m4_obs_size(R) == m4_obs_size(P) == m4_obs_size(GameConfig.m4()) == 88
    assert m4_feature_names(R) == m4_feature_names(P)


def test_power_curve_is_bit_for_bit_the_m42c_formula():
    inv = pg.investment(P)
    assert inv.one == pg.INVEST_ONE == 10 ** 12 and inv.curve == "power"
    for u in (0, 1, 999_999, MU, 7 * MU + 3, 10 ** 9, 123_456_789_012, inv.max_units):
        m = pg.INVEST_ONE + int(round(pg.INVEST_ONE * 0.045 * ((1.0 + u / MU) ** 0.9 - 1.0)))
        assert inv.multiplier(u) == m
        for L in (1, 18):
            ap = pg.tables(P).ap[L][0] * m
            assert inv.hit_damage(2, L, u) == (200 * ap + 5000 * 10 ** 12) // 10 ** 16


@pytest.mark.parametrize("kw, field", [
    (dict(invest_curve="log"), "invest_curve"),
    (dict(invest_obs_encoding="log"), "invest_obs_encoding"),
    (dict(invest_gain=2.0), "invest_gain"),                                        # power curve: not its field
    (dict(invest_curve="saturating", invest_alpha=0.06), "invest_alpha"),           # saturating: not its field
    (dict(invest_curve="saturating", invest_gain=0.0), "invest_gain"),
    (dict(invest_curve="saturating", invest_scale=0.5), "invest_scale"),
    (dict(invest_curve="saturating", invest_scale=16.0000005), "invest_scale"),     # not a whole ledger count
])
def test_validation_rejects_bad_or_foreign_curve_fields(kw, field):
    with pytest.raises(ValueError, match=field):
        GameConfig.m4_continuous(**kw).validate()


@pytest.mark.parametrize("field, value", [("invest_curve", "saturating"), ("invest_gain", 2.0),
                                          ("invest_scale", 8.0), ("invest_obs_encoding", "compressed")])
def test_the_two_tier_baseline_rejects_the_new_fields(field, value):
    with pytest.raises(ValueError, match=field):
        GameConfig.m4(**{field: value}).validate()


# ---------------------------------------------------------------------- the saturating curve
def test_saturating_curve_is_neutral_at_zero_and_matches_the_formula():
    t = pg.tables(R)
    assert INV.one == pg.INVEST_ONE_SAT and INV.multiplier(0) == INV.one
    for L in (1, 9, 18):
        assert INV.attack_power(L, 0) == t.ap[L][0] * INV.one
        for d in (0, 1, 2):
            assert INV.hit_damage(d, L, 0) == t.dmg[d][L][0]
        assert INV.pressure_damage(L, 0) == t.pressure[L][0]
    for S, m in ((1, 1.045), (4, 1.158), (10, 1.323), (20, 1.5), (128, 2.0), (300, 2.162), (1000, 2.312)):
        assert INV.factor(S * MU) == pytest.approx(1 + 1.5 * (1 - (16 / (16 + S)) ** 0.5), abs=1e-12)
        assert round(INV.factor(S * MU), 3) == m
    # exact integers: an independent 60-digit decimal evaluation gives the same fixed-point value
    getcontext().prec = 60
    one, k = Decimal(10) ** 24, Decimal(16 * MU)
    for u in (1, 3, MU, 7 * MU + 5, 299 * MU + 999_999, INV.max_units):
        r = int((one * (k / (k + u)).sqrt()).to_integral_value(rounding="ROUND_FLOOR"))
        assert INV.multiplier(u) == 10 ** 24 + 3 * (10 ** 24 - r) // 2


def test_saturating_curve_is_strictly_increasing_per_ledger_unit_and_bounded():
    grid = sorted({0, 1, 2, 3, MU - 1, MU, MU + 1, 7 * MU, 10 ** 9, 10 ** 11, INV.max_units - 1, INV.max_units}
                  | set(np.random.default_rng(1).integers(0, INV.max_units, 300).tolist()))
    ms = [INV.multiplier(u) for u in grid]
    assert all(b > a for a, b in zip(ms, ms[1:]))
    for u in (0, 1, MU, 300 * MU, 10 ** 9, INV.max_units - 1):
        assert INV.multiplier(u + 1) > INV.multiplier(u)
    assert INV.multiplier(INV.max_units) < INV.one * 5 // 2                    # M < 1 + invest_gain = 2.5
    for L in (1, 18):
        dmg = [INV.hit_damage(2, L, u) for u in grid]
        assert all(b >= a for a, b in zip(dmg, dmg[1:]))
    with pytest.raises(ValueError, match="supported range"):
        INV.multiplier(INV.max_units + 1)


def test_no_one_shot_of_a_full_hp_champion_anywhere_in_the_supported_range():
    """M < 2.5 for every S: a 2-HP hit never kills a full-HP champion of the same level (or up to 3 levels lower),
    and 1-HP hits never kill in fewer than 3. Under the M4.2c curve the 2-HP one-shot arrives at S ~ 68."""
    t = pg.tables(R)
    top = INV.max_units
    for L in range(1, t.level_cap + 1):
        for gap in range(0, 4):
            att = min(L + gap, t.level_cap)
            assert INV.hit_damage(2, att, top) < t.hp_max[L]
        assert -(-t.hp_max[L] // INV.hit_damage(1, L, top)) >= 3
    pinv = pg.investment(P)
    assert pinv.hit_damage(2, 18, 69 * MU) >= t.hp_max[18]


def test_every_typical_return_still_raises_power_late_in_the_window():
    """Diminishing but positive: 1 resource at S = 300 raises M by > 0.02 %, and a 4-resource return there moves a
    level-18 2-HP hit by at least one centi-HP at some point within the next few returns."""
    for S in (0, 10, 100, 300):
        a, b = INV.multiplier(S * MU), INV.multiplier((S + 1) * MU)
        assert (b - a) / a > 2e-4
    d0 = INV.hit_damage(2, 18, 300 * MU)
    assert INV.hit_damage(2, 18, 312 * MU) > d0


# ---------------------------------------------------------------------- gameplay invariants under the recommendation
def test_return_invests_all_gold_and_rewards_are_spending_neutral():
    for gold in (1, MU, 4 * MU + 7):
        st = M4State(side(pos=-T, absent=1, gold=gold, invested=5 * MU), side())
        t = resolve_turn_m4(LANE, st, 2, CHAMP_ABSENT, HO, R, u=U0)
        assert t.returned[0] and t.invested[0] == gold and t.champ.a.invested == 5 * MU + gold
        assert t.champ.a.gold == 0 and t.gold_clamped == (0, 0)
        t0 = resolve_turn_m4(LANE, M4State(side(pos=-T, absent=1, invested=5 * MU), side()), 2, CHAMP_ABSENT, HO,
                             R, u=U0)
        assert t.reward_units == t0.reward_units


def _game(seed: int, turns: int):
    ag = m4_suite()
    x, y = ag["aggressive_trader"], ag["gold_recaller"]
    env = LaneEnv(R, max_turns=turns)
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    out = []
    while True:
        r = env.step(x.act(obs[0], rng), y.act(obs[1], rng))
        out.append((r.info, env.champ, r.obs))
        if r.done:
            return out
        obs = r.obs


def test_ledger_exact_zero_sum_deterministic_and_observations_decode_in_play():
    run = _game(5, 250)
    again = _game(5, 250)
    earned = [0, 0]
    for (i, champ, obs), (j, champ2, _) in zip(run, again):
        assert i["reward_units"] == j["reward_units"] and champ == champ2           # deterministic replay
        assert i["reward_units"] == i["earned_units"][0] - i["earned_units"][1]
        for s in (0, 1):
            earned[s] += i["earned_units"][s]
            me = champ.side(s)
            assert me.gold + me.invested == earned[s] and i["gold_unspendable"][s] == 0
        check_invariants(champ, R)
        for s in (0, 1) if obs is not None else ():
            v = obs[s].vector
            assert v.dtype == np.float32 and v.min() >= 0 and v.max() <= 1
            mine = decode_m4(v, R)[2]
            assert mine == (champ if s == 0 else champ.mirror())
    assert max(c.side(s).invested for _, c, _ in run for s in (0, 1)) > 20 * MU


# ---------------------------------------------------------------------- the compressed observation entries
@pytest.mark.parametrize("ref", [8, 64])
def test_compressed_entry_is_exact_in_float32_up_to_the_bound(ref):
    N = compressed_exact_max(ref)
    n = np.arange(0, N + 1, dtype=np.int64)
    x = (n / (n + ref)).astype(np.float32)
    assert np.all(np.diff(x) > 0) and x.max() < 1.0
    dec = np.rint(ref * x.astype(np.float64) / (1.0 - x.astype(np.float64))).astype(np.int64)
    assert np.array_equal(dec, n)


def test_compressed_observation_round_trips_beyond_the_linear_reference_and_never_saturates():
    ns, ng = compressed_exact_max(R.invest_obs_ref), compressed_exact_max(R.invest_obs_gold_ref)
    assert ns > 20_000 and ng > 8_000
    for a, b in [(0, 0), (1, 7 * MU + 1), (123_456_789, 3), (1024 * MU + 5, 1024 * MU),
                 (ns * MU + MU - 1, ng * MU + MU - 1)]:
        st = M4State(side(pos=-1, invested=a, gold=b, xp=60), side(pos=2, invested=a, gold=b))
        v = observe_m4(LANE, 5, 0, R, st)
        assert v.dtype == np.float32 and v.min() >= 0 and v.max() <= 1
        assert decode_m4(v, R)[2] == st
    # beyond the exact range: still a valid, strictly-below-1 entry, but the decoder refuses
    st = M4State(side(pos=-1, invested=INV.max_units), side(pos=2, gold=(ng + 1) * MU))
    v = observe_m4(LANE, 5, 0, R, st)
    names = m4_feature_names(R)
    assert v[names.index("own_champ_invested_whole")] < 1.0 and v[names.index("enemy_champ_gold_whole")] < 1.0
    with pytest.raises(ValueError, match="exact range"):
        decode_m4(v, R)


def test_compressed_entries_resolve_small_gold_and_power_feature_is_bounded():
    names = m4_feature_names(R)
    gi, si, pi = (names.index(f"own_champ_{n}") for n in ("gold_whole", "invested_whole", "attack_power"))
    vals = []
    for g in (0, 1, 4, 8, 20):
        v = observe_m4(LANE, 0, 0, R, M4State(side(gold=g * MU), side()))
        vals.append(float(v[gi]))
    assert vals[0] == 0 and vals[1] == pytest.approx(1 / 9) and vals[3] == pytest.approx(0.5)
    vp = observe_m4(LANE, 0, 0, P, M4State(side(cfg=P, gold=4 * MU), side(cfg=P)))
    assert float(vp[names.index("own_champ_gold_whole")]) == pytest.approx(4 / 1024)    # linear: 256x weaker
    top = observe_m4(LANE, 0, 0, R, M4State(side(xp=10 ** 4, invested=INV.max_units), side()))
    assert 0 < top[pi] < 1 and top[si] < 1
