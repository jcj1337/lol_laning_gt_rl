"""M4.2e-B: combat-driven zoning on natural waves (docs/m4_2e_zoning_report.md).

The opt-in engine fields of the study (trade_exchanges, trade_farm_rule "loser", invest_durability,
collect_contest_own_tower) and their default preservation; the candidate preset `GameConfig.m4_lane()`; the global
correctness guarantees on the candidate preset; and pins of the report's findings (the last section is filled in by the
milestone's results, see the report)."""

import hashlib

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import N_CHANCE, mirror_chance, swap_seats
from lanegame.champions_m4 import (M4State, check_invariants, collection_units, context, initial_m4, resolve_turn_m4,
                                   trade_damage_distribution_n, trade_damage_n)
from lanegame.champions import AWAY, FARM, RETREAT, SAFE_MODE, TRADING, champion_legal_actions
from lanegame.config import B_HOLD, B_SAFE, B_TRADE, B_YIELD, NONE, W_HOLD, champ_act
from lanegame.env import LaneEnv, UniformAgent
from lanegame.env.observation_m4 import decode_m4, observe_m4
from lanegame.mechanics import initial_state
from lanegame.progression import tables

from test_m4_env import soak

NAT = GameConfig.m4_natural()
A, B = 0, 1


def _digest(cfg: GameConfig, turns: int = 300, seed: int = 7) -> str:
    """SHA-256 of the full resolver output of `turns` uniformly random legal turns from s0 (positions, lane, HP, XP,
    gold, rewards, every M4Turn field)."""
    rng = np.random.default_rng(seed)
    lane, m4 = initial_state(cfg), initial_m4(cfg)
    h = hashlib.sha256()
    for k in range(turns):
        la, lb = champion_legal_actions(lane, m4, A, cfg), champion_legal_actions(lane, m4, B, cfg)
        a, b = la[rng.integers(len(la))], lb[rng.integers(len(lb))]
        u = tuple(rng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, u=u)
        h.update(repr((t.lane.state, t.champ, t.farm_units, t.earned_units, t.damage_taken, t.trade_damage,
                       t.died, t.modes, t.end, t.reward_units, t.xp_gain, t.plates_broken)).encode())
        lane, m4 = t.lane.state, t.champ
    return h.hexdigest()


# ---------------------------------------------------------------------- default preservation
def test_new_fields_default_off_and_historical_presets_unchanged():
    for cfg in (GameConfig(), GameConfig.champions(), GameConfig.m4(), GameConfig.m4_continuous(),
                GameConfig.m4_emergent(), GameConfig.m4_natural_prototype(), GameConfig.m4_natural()):
        cfg.validate()
        assert (cfg.trade_exchanges, cfg.trade_farm_rule, cfg.invest_durability, cfg.collect_contest_own_tower) \
            == (1, "trader", False, 0.9)
    # the own-tower rate at its default equals the contest rate: collection is bit-identical to the M4.2e-A.1 rule
    for x in range(-NAT.tower, NAT.tower + 1):
        for mode in (FARM, TRADING, SAFE_MODE, RETREAT):
            for pos in range(-NAT.tower, NAT.tower + 1):
                ref = collection_units(x, mode, pos, FARM, 0, NAT.with_(collect_contest_own_tower=NAT.collect_contest))
                assert collection_units(x, mode, pos, FARM, 0, NAT) == ref


@pytest.mark.parametrize("preset", ["m4", "m4_emergent", "m4_natural"])
def test_random_turn_digest_of_the_historical_presets_is_pinned(preset):
    """300 random legal turns of each historical M4 preset give the digest recorded when M4.2e-B was integrated
    (the M4.2e-B fields are off there). Update only if a deliberate change to a historical preset is made."""
    cfg = getattr(GameConfig, preset)()
    pinned = {"m4": "6ce9b66a061d772803f2d5b3e99a7453d1e09451d319ff9c0736a8ad3ebc94ee",
              "m4_emergent": "7c9781282fc981a76e915a8a458da5f6c21d0549f6bc1705f9b7831d5005738e",
              "m4_natural": "9bac02a6312d6e8f324232de23c50d883f646bbc77f841d83423c15087841fb9"}[preset]
    assert _digest(cfg) == pinned


def test_random_turn_digest_of_the_candidate_preset_is_pinned():
    """The candidate `m4_lane()` itself (deterministic replay of the whole resolver on random legal play)."""
    assert _digest(GameConfig.m4_lane()) == "c108fd7ebb6ddf22a0aaaba6ca0f4455fd9b06455c00173b063c1320263ece60"


def test_legacy_rulesets_reject_the_new_fields():
    for kw in (dict(trade_exchanges=2), dict(trade_farm_rule="loser"), dict(collect_contest_own_tower=0.6)):
        with pytest.raises(ValueError):
            GameConfig.champions(**kw).validate()
    with pytest.raises(ValueError):
        GameConfig.m4(invest_durability=True).validate()           # continuous items only
    for kw in (dict(trade_exchanges=0), dict(trade_exchanges=3), dict(trade_farm_rule="winner"),
               dict(collect_contest_own_tower=1.5)):
        with pytest.raises(ValueError):
            GameConfig.m4_natural(**kw).validate()


# ---------------------------------------------------------------------- trade_exchanges
def test_n_exchange_distribution_is_the_convolution_and_the_draw_is_its_inverse_cdf():
    h, q = 0.55, 0.3
    one = (1 - h, h * (1 - q), h * q)
    two = trade_damage_distribution_n(h, q, 2)
    assert len(two) == 5 and abs(sum(two) - 1) < 1e-12
    conv = [0.0] * 5
    for i, p in enumerate(one):
        for j, r in enumerate(one):
            conv[i + j] += p * r
    assert np.allclose(two, conv)
    assert trade_damage_distribution_n(h, q, 1) == pytest.approx(one)
    # the draw: u >= P(D >= 1) deals nothing (the no-chance convention u = 1.0 included); u < P(D = 4) deals 4
    assert trade_damage_n(1.0, h, q, 2) == 0 and trade_damage_n(0.999, h, q, 2) == 0
    assert trade_damage_n(0.0, h, q, 2) == 4
    draws = [trade_damage_n(u, h, q, 2) for u in np.linspace(0, 1, 200_001)[:-1]]
    freq = np.bincount(draws, minlength=5) / len(draws)
    assert np.allclose(freq, two, atol=2e-3)


def test_two_exchanges_double_the_expected_trade_damage_and_the_tables_cover_four_base_hp():
    cfg = GameConfig.m4_natural(trade_exchanges=2)
    ctx = context(cfg)
    assert [ctx.hit(d, 1, 0) for d in range(5)] == [0, 100, 200, 300, 400]
    t2 = tables(GameConfig.m4(trade_exchanges=2))
    assert len(t2.dmg) == 5 and t2.dmg[4][1][0] == 400 and t2.dmg[2][1][0] == tables(GameConfig.m4()).dmg[2][1][0]
    lane, m4 = initial_state(cfg), initial_m4(cfg)
    m4 = M4State(m4.a._replace(pos=0), m4.b._replace(pos=0))
    a, b = champ_act(W_HOLD, B_TRADE), champ_act(W_HOLD, B_HOLD)
    tot = [0, 0]
    rng = np.random.default_rng(0)
    n = 4000
    for cfg_i, i in ((GameConfig.m4_natural(), 0), (cfg, 1)):
        for _ in range(n):
            u = list(rng.random(N_CHANCE))
            u[6] = u[7] = 1.0                               # no ganks
            t = resolve_turn_m4(lane, m4, 9, a, b, cfg_i, u=tuple(u))
            tot[i] += t.trade_damage[A]
    assert abs(tot[1] / tot[0] - 2.0) < 0.1


# ---------------------------------------------------------------------- trade_farm_rule "loser"
def _mutual_trade(cfg, u_hit_a, u_hit_b):
    lane, m4 = initial_state(cfg), initial_m4(cfg)
    # a lane with kills: both fronts in contact at the lane (k9 of Hold / Hold)
    for k in range(9):
        t = resolve_turn_m4(lane, m4, k, champ_act(W_HOLD, B_HOLD), champ_act(W_HOLD, B_HOLD), cfg, u=None)
        lane, m4 = t.lane.state, t.champ
    u = [1.0] * N_CHANCE
    u[0], u[2] = u_hit_a, u_hit_b                            # hit uniforms of A and B (double: u[1], u[3] = 1.0)
    return resolve_turn_m4(lane, m4, 9, champ_act(W_HOLD, B_TRADE), champ_act(W_HOLD, B_TRADE), cfg, u=tuple(u))


def test_loser_rule_charges_the_farm_factor_to_the_loser_of_a_mutual_trade_only():
    base, loser = GameConfig.m4_natural(), GameConfig.m4_natural(trade_farm_rule="loser")
    # A hits (u < 0.55), B misses: B lost more -> B pays, A keeps its farm; under the M2 rule both pay
    tb, tl = _mutual_trade(base, 0.1, 0.9), _mutual_trade(loser, 0.1, 0.9)
    assert tb.connected == (True, True) and tl.connected == (True, True)
    assert tb.trade_damage == tl.trade_damage and tb.trade_damage[A] > 0 and tb.trade_damage[B] == 0
    assert tl.farm_units[B] == tb.farm_units[B] and tl.farm_units[A] == 2 * tb.farm_units[A] > 0
    # both miss: a tie, both pay (as before)
    tb, tl = _mutual_trade(base, 0.9, 0.9), _mutual_trade(loser, 0.9, 0.9)
    assert tl.farm_units == tb.farm_units
    # a unilateral trade still costs the trader alone
    for cfg in (base, loser):
        lane, m4 = initial_state(cfg), initial_m4(cfg)
        for k in range(9):
            t = resolve_turn_m4(lane, m4, k, champ_act(W_HOLD, B_HOLD), champ_act(W_HOLD, B_HOLD), cfg, u=None)
            lane, m4 = t.lane.state, t.champ
        u = [1.0] * N_CHANCE
        u[0] = 0.1
        t = resolve_turn_m4(lane, m4, 9, champ_act(W_HOLD, B_TRADE), champ_act(W_HOLD, B_HOLD), cfg, u=tuple(u))
        assert t.modes == (TRADING, FARM) and t.farm_units[A] * 2 == t.farm_units[B]


# ---------------------------------------------------------------------- invest_durability
def test_durability_divides_all_damage_taken_by_the_victims_multiplier():
    cfg = GameConfig.m4_natural(invest_durability=True)
    ctx = context(cfg)
    g = cfg.gold_scale
    lane, m4 = initial_state(cfg), initial_m4(cfg)
    m4 = M4State(m4.a._replace(pos=0, items=50 * g), m4.b._replace(pos=0))
    u = [0.0] * N_CHANCE                                    # every chance event: hits, doubles, ganks (max damage)
    t = resolve_turn_m4(lane, m4, 9, champ_act(W_HOLD, B_TRADE), champ_act(W_HOLD, B_TRADE), cfg, u=tuple(u))
    raw_a = t.trade_damage[B] + t.tower_damage[A] + t.gank_damage[A]
    raw_b = t.trade_damage[A] + t.tower_damage[B] + t.gank_damage[B]
    assert t.damage_taken[B] == raw_b                       # B has nothing invested
    assert t.damage_taken[A] == raw_a * ctx.inv.one // ctx.inv.multiplier(50 * g) < raw_a
    assert ctx.mitigate(1000, 0) == 1000 and ctx.mitigate(0, 50 * g) == 0
    # off by default: bit-identical damage
    t0 = resolve_turn_m4(lane, m4, 9, champ_act(W_HOLD, B_TRADE), champ_act(W_HOLD, B_TRADE), GameConfig.m4_natural(),
                         u=tuple(u))
    assert t0.damage_taken == (raw_a, raw_b)


# ---------------------------------------------------------------------- collect_contest_own_tower
def test_own_tower_rate_caps_collection_in_the_own_tower_cell_for_every_posture():
    cfg = GameConfig.m4_natural(collect_contest_own_tower=0.6)
    T = cfg.tower
    unit = cfg.farm_per_hp * cfg.gold_scale
    assert collection_units(-T, FARM, -T, FARM, 0, cfg) == int(round(unit * 0.6))
    assert collection_units(-T, SAFE_MODE, -T, FARM, 0, cfg) == int(round(unit * 0.6))      # min(0.8, 0.6)
    assert collection_units(-T + 1, FARM, -T, FARM, 0, cfg) == int(round(unit * 0.9))       # unchanged elsewhere
    assert collection_units(-T + 1, SAFE_MODE, -T, FARM, 0, cfg) == int(round(unit * 0.6))  # the posture base 0.6
    assert collection_units(T, FARM, T, FARM, 0, cfg) == int(round(unit * cfg.collect_contest_enemy_tower))


# ---------------------------------------------------------------------- the candidate preset: global guarantees
def _candidate():
    return GameConfig.m4_lane()


def test_candidate_preset_validates_and_differs_from_m4_natural_only_in_the_documented_fields():
    cfg = _candidate()
    cfg.validate()
    changed = {k for k, v in cfg.to_dict().items() if NAT.to_dict()[k] != v}
    assert changed == set(GameConfig.M4_LANE_FIELDS)
    assert GameConfig.m4_natural().to_dict() == NAT.to_dict()


def test_candidate_preset_mirror_symmetry_with_a_stronger_seat_and_invariants():
    cfg = _candidate()
    rng = np.random.default_rng(3)
    lane, m4 = initial_state(cfg), initial_m4(cfg)
    m4 = M4State(m4.a._replace(items=30 * cfg.gold_scale), m4.b)
    for k in range(400):
        la, lb = champion_legal_actions(lane, m4, A, cfg), champion_legal_actions(lane, m4, B, cfg)
        a, b = la[rng.integers(len(la))], lb[rng.integers(len(lb))]
        u = tuple(rng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, u=u)
        tm = resolve_turn_m4(lane.mirror(), m4.mirror(), k, b, a, swap_seats(cfg), u=mirror_chance(u))
        assert tm.champ == t.champ.mirror() and tm.lane.state == t.lane.state.mirror()
        assert tm.reward_units == -t.reward_units and tm.damage_taken == t.damage_taken[::-1]
        check_invariants(t.champ, cfg)
        lane, m4 = t.lane.state, t.champ


def test_candidate_preset_observation_is_valid_and_lossless():
    cfg = _candidate()
    rng = np.random.default_rng(5)
    lane, m4 = initial_state(cfg), initial_m4(cfg)
    for k in range(200):
        for s in (A, B):
            v = observe_m4(lane, k, s, cfg, m4)
            assert v.dtype == np.float32 and np.all(np.isfinite(v)) and np.all(v >= 0) and np.all(v <= 1)
            ph, lane_d, m4_d, _ = decode_m4(v, cfg)
            assert ph == k % cfg.phase_period
            assert (lane_d, m4_d) == ((lane, m4) if s == A else (lane.mirror(), m4.mirror()))
        la, lb = champion_legal_actions(lane, m4, A, cfg), champion_legal_actions(lane, m4, B, cfg)
        t = resolve_turn_m4(lane, m4, k, la[rng.integers(len(la))], lb[rng.integers(len(lb))], cfg,
                            u=tuple(rng.random(N_CHANCE)))
        lane, m4 = t.lane.state, t.champ


def test_candidate_preset_env_soak_and_deterministic_replay():
    cfg = _candidate()
    soak(cfg, turns=1500, seed=11)
    env, ag = LaneEnv(cfg, max_turns=200), UniformAgent()
    traces = []
    for _ in range(2):
        obs = env.reset(seed=99)
        rng = np.random.default_rng(1)
        rows = []
        while True:
            r = env.step(ag.act(obs[0], rng), ag.act(obs[1], rng))
            rows.append((r.info["actions"], r.info["reward_units"], r.info["champ_after"], r.info["state_after"]))
            if r.done:
                break
            obs = r.obs
        traces.append(rows)
    assert traces[0] == traces[1]
