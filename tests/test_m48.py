"""M4.8: the lane-exit readout (`lanegame.diagnostics.lane_exit`) and the study's state, case and job machinery.

Accounting (earned = invested + unspent; investing is never earning; the lead change equals the summed rewards exactly),
checkpoint restoration (serialization, identical continuations, replays of the M4.6d records), paired randomness (the
chance stream does not depend on the actions; equal seeds give identical continuations), mirrored orientation (exact
negation), window extraction (prefixes of the same continuations), resolution times, variants and case criteria.
Record- and checkpoint-dependent tests skip when the git-ignored files are absent."""

import os

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.engine.champions_m4 import champion_legal_actions, check_invariants
from lanegame.engine.config import B_RECALL, champ_behavior
from lanegame.diagnostics import lane_exit as LX
from experiments.m45b import states as S

CFG = GameConfig.m45_calibrated()
RECORD = os.path.join("runs", "checkpoints", "m46d", "records", "baseline", "ext_baseline_0.npz")
CKPT = os.path.join("runs", "checkpoints", "m46c", "train", "ext", "baseline", "s0", "final.pt")


class RandomLegal:
    """A seeded random legal policy whose uniform of turn i is fixed by (seed, stream, i) (like CRN sampling); Recall
    is taken with probability `recall` whenever it is legal, so returns and investments happen."""

    def __init__(self, seed: int, stream: int, T: int, recall: float = 0.25):
        self.u = np.random.default_rng([int(seed), 11, int(stream)]).random((T + 1, 2))
        self.recall = recall

    def decide(self, cfg, lane, m4, s, i):
        legal = champion_legal_actions(lane, m4, s, cfg)
        rec = [c for c in legal if c >= 0 and champ_behavior(c) == B_RECALL]
        if rec and self.u[i, 0] < self.recall:
            return rec[0], "recall"
        return legal[min(int(self.u[i, 1] * len(legal)), len(legal) - 1)], "random"


def _pols(sd, T, mirror=False):
    if mirror:
        return RandomLegal(sd, 1, T), RandomLegal(sd, 0, T)
    return RandomLegal(sd, 0, T), RandomLegal(sd, 1, T)


def _state(turns=40, sd=3):
    """A natural mid-game state: random legal play from s0."""
    g = LX.continuation(CFG, S.s0(CFG), *_pols(sd, turns), sd, turns, keep_states=True)
    return g["states"][-1]


# ---------------------------------------------------------------------------------------------- accounting
def test_lead_change_equals_summed_rewards_and_spending_is_not_earning():
    T = 120
    g = LX.continuation(CFG, S.s0(CFG), *_pols(5, T), 5, T, keep_states=True)
    assert int(g["acct_units"]) == 0
    assert g["returned"].any(), "the test game must contain returns (investments)"
    sc = CFG.gold_scale
    for i in range(T):
        before, after = g["states"][i][1], g["states"][i + 1][1]
        for s in (0, 1):
            d_earned = (LX.earned_units(after, s, CFG) - LX.earned_units(before, s, CFG)) / sc
            assert d_earned == pytest.approx(float(g["earned"][i, s]), abs=1e-6)   # float32 ledger
            assert g["earned"][i, s] == pytest.approx(g["farm"][i, s] + g["plate"][i, s] + g["bounty"][i, s],
                                                      abs=1e-6)
            assert g["avail"][i, s] >= g["farm"][i, s] - 1e-6
            if g["returned"][i, s]:                     # the return invests: unspent -> invested, earned unchanged
                assert after.side(s).gold == 0 and after.side(s).items >= before.side(s).items
        assert g["r"][i] == pytest.approx(float(g["earned"][i, 0] - g["earned"][i, 1]), abs=1e-6)
    D_end = LX.lead(g["states"][-1][1], CFG)
    assert D_end == pytest.approx(LX.lead(g["states"][0][1], CFG) + g["r"].sum(), abs=1e-9)


def test_tier_item_models_are_refused():
    with pytest.raises(ValueError):
        LX.lead(S.s0(GameConfig.m4())[1], GameConfig.m4())


def test_describe_reports_the_dimensions_beside_the_lead():
    st = _state()
    d = LX.describe(CFG, st, {"since_return_A": 3, "since_return_B": 7})
    assert d["D0"] == pytest.approx(d["earned_A"] - d["earned_B"])
    assert d["earned_A"] == pytest.approx(d["invested_A"] + d["unspent_A"])
    for k in ("xp_diff", "level_diff", "hp_A", "status_B", "returns_in_A", "stock_A", "near_B", "since_return_B"):
        assert k in d
    lane, m4, _ = st
    efx, efh, erear, esg = lane.bx, lane.bh, lane.br, lane.bsg
    assert d["stock_A"] == (efh if efh > 0 else 0) + CFG.wave_hp * len(erear) + esg


def test_since_return():
    assert LX.since_return([0, 0, 0, 0], 3) == 3                 # never left: since the start
    assert LX.since_return([0, 1, 0, 0, 0], 4) == 2              # away at turn 1, back at the start of turn 2
    assert LX.since_return([0, 3, 2, 1, 0, 0], 5) == 1
    assert LX.since_return([0, 1, 0, 0, 1, 0], 5) == 0


# ---------------------------------------------------------------------------------------------- restoration
def test_restored_checkpoint_continues_identically():
    from experiments.m46a.common import state_from_list, state_to_list
    st = _state()
    back = state_from_list(state_to_list(st))
    assert back[0] == st[0] and back[1] == st[1] and back[2] == st[2]
    a = LX.continuation(CFG, st, *_pols(9, 30), 9, 30)
    b = LX.continuation(CFG, back, *_pols(9, 30), 9, 30)
    for k in LX.SIDE_KEYS + ("r",):
        assert np.array_equal(a[k], b[k])


# ---------------------------------------------------------------------------------------------- paired randomness
def test_chance_stream_does_not_depend_on_the_actions(monkeypatch):
    st = _state()
    seen = []
    real = LX.resolve_turn_m4

    def spy(lane, m4, k, a, b, cfg, **kw):
        seen[-1].append(tuple(kw["u"]))
        return real(lane, m4, k, a, b, cfg, **kw)

    monkeypatch.setattr(LX, "resolve_turn_m4", spy)
    seen.append([])
    LX.continuation(CFG, st, *_pols(4, 25), 4, 25)
    seen.append([])
    LX.continuation(CFG, st, RandomLegal(77, 0, 25, recall=1.0), RandomLegal(78, 1, 25, recall=0.0), 4, 25)
    assert seen[0] == seen[1]
    seen.append([])
    LX.continuation(CFG, st, *_pols(4, 25), 5, 25)
    assert seen[2] != seen[0]


def test_equal_seeds_give_identical_continuations_and_paired_differences_vanish():
    st = _state()
    mk = lambda sd: _pols(sd, 20)                                # noqa: E731
    a = LX.continuations(CFG, st, mk, range(6), 20)
    b = LX.continuations(CFG, st, mk, range(6), 20)
    assert np.array_equal(a["r"], b["r"])
    p = LX.paired(a["r"], b["r"], 10)
    assert p["diff"] == 0 and p["se"] == 0
    c = LX.continuations(CFG, st, mk, range(6, 12), 20)
    assert not np.array_equal(a["r"], c["r"])


# ---------------------------------------------------------------------------------------------- orientation
def test_mirrored_continuation_is_the_exact_negation():
    st = _state()
    T = 30
    o = LX.continuations(CFG, st, lambda sd: _pols(sd, T), range(5), T)
    m = LX.continuations(CFG, LX.mirror_state(st), lambda sd: _pols(sd, T, mirror=True), range(5), T, mirror=True)
    assert np.array_equal(o["r"], -m["r"])
    for k in LX.SIDE_KEYS:
        assert np.array_equal(o[k], m[k][..., ::-1])
    d, dm = LX.describe(CFG, st), LX.describe(CFG, LX.mirror_state(st))
    assert dm["D0"] == -d["D0"] and dm["xp_diff"] == -d["xp_diff"]
    for k in ("stock", "near", "hp", "status", "unspent", "level"):
        assert dm[f"{k}_A"] == d[f"{k}_B"] and dm[f"{k}_B"] == d[f"{k}_A"]
    ra, rm = LX.readout(d["D0"], o["r"], (5, 10)), LX.readout(dm["D0"], m["r"], (5, 10))
    assert ra[10]["E"] == pytest.approx(-rm[10]["E"]) and ra[10]["se"] == pytest.approx(rm[10]["se"])


# ---------------------------------------------------------------------------------------------- windows
def test_windows_are_prefixes_of_the_same_continuations():
    st = _state()
    long = LX.continuation(CFG, st, *_pols(2, 40), 2, 40)
    short = LX.continuation(CFG, st, *_pols(2, 40), 2, 10)       # the same streams, stopped at 10
    assert np.array_equal(long["r"][:10], short["r"])
    r = np.stack([long["r"], -long["r"]])
    assert np.array_equal(LX.deltas(r, 10), r[:, :10].sum(-1))
    ro = LX.readout(1.5, r, (5, 10, 40))
    assert ro[10]["E"] == pytest.approx(1.5 + LX.deltas(r, 10).mean())
    assert ro[40]["lo"] <= ro[40]["E"] <= ro[40]["hi"] and ro[5]["K"] == 2
    with pytest.raises(ValueError):
        LX.deltas(r, 41)


def test_components_reconcile_with_the_readout():
    st = _state()
    led = LX.continuations(CFG, st, lambda sd: _pols(sd, 20), range(8), 20)
    c = LX.components(led, 20)
    assert c["earned_diff"] == pytest.approx(LX.deltas(led["r"], 20).mean(), abs=1e-6)
    assert c["earned_A"] == pytest.approx(c["farm_A"] + c["plate_A"] + c["bounty_A"], abs=1e-6)
    assert c["lost_A"] == pytest.approx(c["avail_A"] - c["farm_A"], abs=1e-6)
    w = LX.window_state(led, 20)
    assert w["xp_diff"] == pytest.approx(float((led["xp"][:, 19, 0] - led["xp"][:, 19, 1]).mean()))


# ---------------------------------------------------------------------------------------------- resolution
def test_first_turn_return_and_stock_times():
    m = np.array([[False, False, True, False], [False, False, False, False]])
    assert LX.first_turn(m).tolist() == [3.0, np.inf]
    led = {"returned": np.zeros((1, 4, 2), bool), "avail": np.zeros((1, 4, 2), np.float32),
           "farm": np.zeros((1, 4, 2), np.float32)}
    led["returned"][0, 1, 0] = True
    assert LX.return_times(led).tolist() == [[2.0, np.inf]]
    phi = 0.125
    led["avail"][0, :, 0] = np.array([4, 4, 8, 0]) * phi            # HP dying on A's ledger per turn
    led["farm"][0, :, 0] = np.array([4, 0, 8, 0]) * phi * 0.9       # A collects 0.9 of turns 0 and 2, nothing of 1
    t = LX.stock_times(led, (12.0, 0.0), phi)
    assert t[0, 0] == 3.0 and t[0, 1] == 0.0                         # 12 HP: 4 + 4 + (4 of 8)
    got = LX.stock_collected(led, (12.0, 0.0), phi)
    assert got[0, 0] == pytest.approx((4 * 0.9 + 0 + 4 * 0.9) / 12)
    assert np.isnan(got[0, 1])


# ---------------------------------------------------------------------------------------------- the study's variants and cases
def _desc(**kw):
    base = {"D0": 0.0, "xp_diff": 0.0, "hp_diff": 0.0, "stock_diff": 0.0}
    for x in ("A", "B"):
        base.update({f"status_{x}": 0.0, f"returns_in_{x}": 0.0, f"hp_{x}": 0.8, f"unspent_{x}": 1.0,
                     f"stock_{x}": 10.0, f"near_{x}": 0.0, f"enemy_siege_{x}": 0.0, f"pos_{x}": 0.0,
                     f"holds_freeze_{x}": 0.0, f"since_return_{x}": 8.0, f"level_{x}": 12.0, f"xp_{x}": 300.0})
    base.update(kw)
    return base


# ---------------------------------------------------------------------------------------------- analysis helpers