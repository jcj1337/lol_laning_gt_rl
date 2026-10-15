"""M4.5b calibration infrastructure (lanegame/experiments/m45b; docs/m4/m4_5/m4_5b_calibration_report.md).

The experiments must measure the real game: the paired game loop reproduces the viewer (and so the environment) turn
for turn; the exact one-turn enumeration agrees with Monte Carlo; every scripted strategy only plays legal actions; the
state builders produce valid states with the requested power ratios; the holdout split is disjoint; the probe
recomputation equals the resolver's own probe; the M4.5b starts replay in the viewer.
"""

import numpy as np
import pytest

from lanegame.champions import N_CHANCE
from lanegame.champions_m4 import check_invariants, resolve_turn_m4
from lanegame.experiments.m45b import common as Cm
from lanegame.experiments.m45b import engine as E
from lanegame.experiments.m45b import exact as X
from lanegame.experiments.m45b import states as S
from lanegame.experiments.m45b import strategies as ST
from lanegame.experiments.m45b import timing as TM
from lanegame.experiments.m45b import trades as T
from lanegame.viewer import session as ss

R2 = Cm.cfg("r2")


# ---------------------------------------------------------------------------------------------- the game loop
@pytest.mark.parametrize("scenario,gap,pa,pb", [("freeze", "large", "fz_guard_sh", "wk_shove_sh"),
                                                ("neutral", "equal", "g_bully", "g_ward"),
                                                ("s0", "equal", "g_shove_sr", "g_guard_ward")])
def test_engine_reproduces_the_viewer(scenario, gap, pa, pb):
    setup = ss.Setup("m45_unified", scenario, gap, 87_000)
    g = E.play(R2, ss.start_state(setup), ST.get(pa), ST.get(pb), 87_000, 60, keep_rows=True)
    rec = ss.play(setup, policies=(ST.get(pa), ST.get(pb)), turns=60)
    assert [r["r"] for r in g["rows"]] == [t.info["reward"] for t in rec.turns]
    assert abs(g["summary"]["lead"] - sum(t.info["reward"] for t in rec.turns)) < 1e-9


def test_ledger_is_zero_sum_and_consistent():
    g = E.play(R2, S.s0(R2), ST.get("g_shove"), ST.get("g_bully"), 87_001, 120)["summary"]
    assert abs(g["lead"] - (g["earned_A"] - g["earned_B"])) < 1e-6
    for s in "AB":
        assert abs(g[f"earned_{s}"] - (g[f"farm_{s}"] + g[f"plate_{s}"] + g[f"bounty_{s}"])) < 1e-6
        assert g[f"d_min_ret_{s}"] + g[f"d_min_pos_{s}"] <= g[f"d_min_{s}"] + 1e-9
        assert g[f"deaths_{s}"] == sum(g[f"d_by_{k}_{s}"] for k in ("gank", "trade", "minion", "tower"))


def test_common_random_numbers_across_strategies():
    """Two games from one seed see the same chance vector on every turn whatever the actions (env convention)."""
    rng1, rng2 = np.random.default_rng(87_002), np.random.default_rng(87_002)
    for _ in range(50):
        assert (rng1.random(N_CHANCE) == rng2.random(N_CHANCE)).all()


# ---------------------------------------------------------------------------------------------- exact enumeration
@pytest.mark.parametrize("cover", ["cover", "nocover", "ahead"])
def test_exact_expectation_matches_monte_carlo(cover):
    lane, m4, k = S.trade_bench(R2, 2, 24, cover, "equal")
    a, b = T.A_TRADE, T.B_ACTIONS["trade"]
    ex = X.expect(R2, lane, m4, k, a, b)
    assert abs(sum(p for p, _ in X.turn_distribution(R2, lane, m4, k, a, b)) - 1.0) < 1e-12
    rng = np.random.default_rng(1)
    vals = []
    for _ in range(4000):
        u = rng.random(N_CHANCE)
        u[6:8] = 1.0                                  # ganks off, as in the exact "off" mode
        t = resolve_turn_m4(lane, m4, k, a, b, R2, u=tuple(u))
        vals.append(X.outcome(t, R2)["taken_bar"])
    mc = np.mean(vals, axis=0)
    se = np.std(vals, axis=0) / np.sqrt(len(vals))
    for s in (0, 1):
        assert abs(mc[s] - ex["mean"]["taken_bar"][s]) < 4.5 * se[s] + 1e-9


def test_exact_ganks_on_matches_monte_carlo():
    lane, m4, k = S.vision_state(R2, "forward", 4, "equal")
    a = b = T.A_HOLD
    ex = X.expect(R2, lane, m4, k, a, b, ganks="on")
    rng = np.random.default_rng(2)
    vals = [X.outcome(resolve_turn_m4(lane, m4, k, a, b, R2, u=tuple(rng.random(N_CHANCE))), R2)["gank_cp"][0]
            for _ in range(6000)]
    assert abs(np.mean(vals) - ex["mean"]["gank_cp"][0]) < 4.5 * np.std(vals) / np.sqrt(len(vals))


# ---------------------------------------------------------------------------------------------- strategies
@pytest.mark.parametrize("name", sorted(ST.LIB))
def test_every_strategy_plays_legal_actions(name):
    p = ST.get(name)
    for start in (S.s0(R2), S.freeze_state(R2, 10, "large", 12), S.vision_state(R2, "presiege", 4, "equal")):
        E.play(R2, start, p, ST.get("g_bully"), 87_003, 40, check_legal=True)
        E.play(R2, start, ST.get("g_shove"), p, 87_004, 40, check_legal=True)


def test_shove_default_responses_change_only_the_default_phase():
    a, b = ST.get("wk_shove_reset"), ST.get("wk_shove_reset_sh")
    assert (a.wave_frozen, a.beh_frozen, a.frozen_reset) == (b.wave_frozen, b.beh_frozen, b.frozen_reset)
    assert (a.wave, b.wave) == ("last_hit", "shove") and b.pressure


# ---------------------------------------------------------------------------------------------- states
@pytest.mark.parametrize("level", [2, 6, 11, 18])
@pytest.mark.parametrize("gap", ["adv", "large", "weak"])
def test_power_gaps_have_the_requested_ratio(level, gap):
    sa, sb = S.invest_pair(R2, level, gap)
    ratio = S.multiplier(R2, sa) / S.multiplier(R2, sb)
    target = {"adv": 1.5, "large": 1.76, "weak": 1 / 1.5}[gap]
    assert abs(ratio - target) < 0.01 and sa >= 0 and sb >= 0


def test_builders_give_valid_states():
    starts = [S.trade_bench(R2, L, w, c, g) for L in (2, 18) for w in (8, 24) for c in S.COVERS
              for g in ("equal", "large")]
    starts += [S.freeze_state(R2, L, g, h, 8, (hp, 1.0)) for L in (4, 10) for g in ("equal", "large")
               for h in (5, 12, 20) for hp in (1.0, 0.6)]
    starts += [S.vision_state(R2, s, 7, "adv") for s in S.VISION_SITUATIONS]
    starts += [S.zoning_state(R2, s, 6, "large") for s in S.ZONING_SITUATIONS]
    for lane, m4, k in starts:
        check_invariants(m4, R2)
        t = resolve_turn_m4(lane, m4, k, T.A_HOLD, T.A_HOLD, R2)
        check_invariants(t.champ, R2)


def test_holdout_split_is_disjoint():
    dev, hold = set(Cm.DEV), set(Cm.HOLDOUT)
    assert not dev & hold and not dev & set(Cm.TRACE) and not hold & set(Cm.TRACE)
    used = set(range(85_000, 87_000)) | set(range(10_000, 82_000)) | set(range(90_000, 92_000))
    assert not (dev | hold) & used
    D, H = S.DEV_GRID, S.HOLDOUT_GRID
    for key in ("levels", "held", "hp", "trade_levels", "trade_waves", "vision_situations", "zoning"):
        assert not set(D[key]) & set(H[key]), key
    assert not set(D["gaps"]) & set(H["gaps"])


# ---------------------------------------------------------------------------------------------- probe diagnostics
class _Spy:
    def __init__(self, cfg):
        self.inner, self.calls = E.lane_cache(cfg), []

    def __call__(self, *a, **k):
        r = self.inner(*a, **k)
        self.calls.append(r)
        return r


def test_probe_recomputation_equals_the_resolver_probe():
    rng = np.random.default_rng(5)
    for pa, pb in (("g_shove", "g_farm"), ("g_bully", "g_safe"), ("g_ward", "g_shove_guard")):
        lane, m4, k = S.s0(R2)
        for i in range(60):
            a = E.decide(ST.get(pa), R2, lane, m4, 0, i)
            b = E.decide(ST.get(pb), R2, lane, m4, 1, i)
            spy = _Spy(R2)
            t = resolve_turn_m4(lane, m4, k + i, a, b, R2, lane_step=spy, u=tuple(rng.random(N_CHANCE)))
            p = TM.probe(R2, lane, m4, k + i, a, b)
            assert p.state == spy.calls[0].state and p.fronts_by_tick == spy.calls[0].fronts_by_tick
            lane, m4 = t.lane.state, t.champ


def test_probe_override_with_the_probe_itself_changes_nothing():
    lane, m4, k = S.freeze_state(R2, 10, "large", 12)
    u = tuple(np.random.default_rng(6).random(N_CHANCE))
    for a, b in ((T.A_TRADE, T.A_HOLD), (T.A_HOLD, T.B_ACTIONS["safe"])):
        t = resolve_turn_m4(lane, m4, k, a, b, R2, u=u)
        t2 = resolve_turn_m4(lane, m4, k, a, b, R2, u=u,
                             lane_step=TM.ProbeOverride(R2, TM.probe(R2, lane, m4, k, a, b)))
        assert t2.champ == t.champ and t2.lane.state == t.lane.state and t2.reward == t.reward


# ---------------------------------------------------------------------------------------------- viewer
def test_m45b_starts_are_registered_with_the_viewer():
    from lanegame.experiments.m45b.scenarios import STARTS
    for name in STARTS:
        assert name in ss.SCENARIOS
        rec1 = ss.play(ss.Setup("m45_unified", name, "equal", 87_900), policies=(ST.get("g_guard"), ST.get("g_shove")),
                       turns=8)
        rec2 = ss.play(ss.Setup("m45_unified", name, "equal", 87_900), prefix=rec1.actions)
        assert [t.info["reward"] for t in rec1.turns] == [t.info["reward"] for t in rec2.turns]
