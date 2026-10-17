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
from lanegame.champions_m4 import champion_legal_actions, check_invariants
from lanegame.config import B_RECALL, champ_behavior
from lanegame.diagnostics import lane_exit as LX
from lanegame.experiments.m45b import states as S

CFG = GameConfig.m45_calibrated()
RECORD = os.path.join("results", "rl", "checkpoints", "m46d", "records", "baseline", "ext_baseline_0.npz")
CKPT = os.path.join("results", "rl", "checkpoints", "m46c", "train", "ext", "baseline", "s0", "final.pt")


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
    from lanegame.experiments.rl.m46a.common import state_from_list, state_to_list
    st = _state()
    back = state_from_list(state_to_list(st))
    assert back[0] == st[0] and back[1] == st[1] and back[2] == st[2]
    a = LX.continuation(CFG, st, *_pols(9, 30), 9, 30)
    b = LX.continuation(CFG, back, *_pols(9, 30), 9, 30)
    for k in LX.SIDE_KEYS + ("r",):
        assert np.array_equal(a[k], b[k])


@pytest.mark.skipif(not (os.path.exists(RECORD) and os.path.exists(CKPT)), reason="M4.6d records / checkpoints absent")
def test_replays_reproduce_the_recorded_games():
    pytest.importorskip("torch")
    from lanegame.experiments.rl.m48 import states as ST
    z = np.load(RECORD)
    rec = {k: z[k] for k in z.files}
    game = ST.play("baseline", 0, ST.rec_seed(0, 1), 60, record_horizon=ST.REC_TURNS)
    ST.verify(rec, 1, game, 60, CFG.gold_scale)                   # raises on any difference
    assert LX.lead(game["states"][60][1], CFG) == pytest.approx(rec["r"][1, :60].sum(), abs=1e-9)
    with pytest.raises(AssertionError):                          # a wrong game index is detected
        ST.verify(rec, 2, game, 60, CFG.gold_scale)


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
def test_controlled_variants_keep_the_raw_lead():
    from lanegame.experiments.rl.m48 import rollouts as R
    st = _state(60)
    D0 = LX.lead(st[1], CFG)
    for name in ("A_fresh", "A_away1"):
        v = R.VARIANTS[name](CFG, st)
        check_invariants(v[1], CFG)
        assert LX.lead(v[1], CFG) == pytest.approx(D0) and v[1].b == st[1].b and v[0] == st[0]
    f = R.VARIANTS["A_fresh"](CFG, st)[1].a
    assert f.gold == 0 and f.absent == 0 and f.pos == -CFG.tower and f.items == st[1].a.items + st[1].a.gold
    a1 = R.VARIANTS["A_away1"](CFG, st)[1].a
    assert a1.absent == 1 and a1.gold == st[1].a.gold and LX.seat_status(R.VARIANTS["A_away1"](CFG, st)[1], 0) in (
        "recall", "dead")


def _desc(**kw):
    base = {"D0": 0.0, "xp_diff": 0.0, "hp_diff": 0.0, "stock_diff": 0.0}
    for x in ("A", "B"):
        base.update({f"status_{x}": 0.0, f"returns_in_{x}": 0.0, f"hp_{x}": 0.8, f"unspent_{x}": 1.0,
                     f"stock_{x}": 10.0, f"near_{x}": 0.0, f"enemy_siege_{x}": 0.0, f"pos_{x}": 0.0,
                     f"holds_freeze_{x}": 0.0, f"since_return_{x}": 8.0, f"level_{x}": 12.0, f"xp_{x}": 300.0})
    base.update(kw)
    return base


def test_case_criteria_read_the_checkpoint_and_are_mirror_symmetric():
    from lanegame.experiments.rl.m48 import states as ST
    ret = _desc(status_A=1.0, returns_in_A=1.0, stock_A=24.0, near_A=16.0, since_return_B=15.0, unspent_B=4.0,
                hp_diff=float("nan"))
    assert ("returner", "A") in ST.cases_of(ret)
    mir = {}
    for k, v in ret.items():
        if k.endswith("_A"):
            mir[k[:-2] + "_B"] = v
        elif k.endswith("_B"):
            mir[k[:-2] + "_A"] = v
        else:
            mir[k] = -v if k in ("D0", "xp_diff", "hp_diff", "stock_diff") else v
    assert ("returner", "B") in ST.cases_of(mir)
    assert ("returner", "A") not in ST.cases_of({**ret, "since_return_B": 5.0})        # B not due
    assert ("lost", "B") in ST.cases_of(_desc(status_B=2.0, returns_in_B=3.0, hp_B=1.0, near_B=16.0,
                                              hp_diff=float("nan")))
    assert ("stable", "A") in ST.cases_of(_desc())
    assert ("opposing", "A") in ST.cases_of(_desc(D0=1.5, xp_diff=-20.0))
    assert ("denied", "A") in ST.cases_of(_desc(holds_freeze_B=1.0))


@pytest.mark.skipif(not os.path.exists(CKPT), reason="M4.6c checkpoints absent")
def test_job_writes_exact_mirrors_and_consistent_window_sums(tmp_path):
    pytest.importorskip("torch")
    from lanegame.experiments.rl.m48 import common as C
    from lanegame.experiments.rl.m48 import families as F
    from lanegame.experiments.rl.m48 import rollouts as R
    cfg = C.setup("baseline")
    from lanegame.experiments.rl.m47 import streams as SM
    st = SM.self_play_state(cfg, C.net_of("ext/baseline:0"), S.s0(cfg), 4242, 50)
    j = {"key": "t", "path": str(tmp_path / "t.npz"), "start": C.state_to_list(st), "index": 1, "root": 1, "K": 3,
         "T": C.T_MAX, "full": True,
         "combos": {"own": F.own("baseline", 0), "mirror": F.own("baseline", 0, mirror=True),
                    "fresh": F.own("baseline", 0, "A_fresh")}}
    R.job(j)
    d = R.load(j["path"])
    assert np.array_equal(d["own"]["r"], -d["mirror"]["r"])
    assert np.all(d["own"]["acct"] == 0) and np.all(d["fresh"]["acct"] == 0)
    for h in C.WINDOWS:
        assert np.allclose(d["own"][f"c{h}_farm"], d["own"]["f__farm"][:, :h].sum(1), atol=1e-4)
        assert np.allclose(d["own"][f"c{h}_farm"], d["mirror"][f"c{h}_farm"][:, ::-1], atol=1e-6)
    r1 = np.asarray(d["own"]["r"], float)
    e = (d["own"]["f__earned"][..., 0] - d["own"]["f__earned"][..., 1]).astype(float)
    assert np.allclose(r1, e, atol=1e-5)
    assert C.roll_seed(1, 1, 0) != C.roll_seed(1, 2, 0) and C.roll_seed(1, 1, 0) != C.roll_seed(1, 1, 1)


# ---------------------------------------------------------------------------------------------- analysis helpers
def test_pending_items_categories_and_resolution():
    from lanegame.experiments.rl.m48 import analysis as AN
    d = _desc(status_A=1.0, returns_in_A=1.0, since_return_B=14.0, stock_A=16.0, stock_B=0.0, stock_diff=16.0)
    p = AN.pending_items(d)
    assert p == {"absent": [0], "stock": [0], "due": [1]}
    assert AN.category(d) == ("recall_away", "A")
    assert AN.category(_desc(status_B=2.0))[0] == "dead"
    assert AN.category(_desc(since_return_A=1.0)) == ("fresh", "A")
    assert AN.category(_desc(since_return_A=4.0, since_return_B=13.0)) == ("due_asym", "A")
    assert AN.category(_desc(stock_diff=-20.0)) == ("stock_asym", "B")
    assert AN.category(_desc()) == ("ordinary", "A")
    x = {"ret": np.array([[1.0, 5.0], [1.0, np.inf]]), "stock_t": np.array([[3.0, 0.0], [2.0, 0.0]])}
    assert AN.resolution(d, x).tolist() == [5.0, np.inf]                  # max over the absent, due and stock items
    assert AN.resolution(d, x, ("absent", "stock")).tolist() == [3.0, 2.0]


def test_cross_validated_r2_and_robust_slope():
    from lanegame.experiments.rl.m48 import analysis as AN
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = 2.0 * X[:, 0] + rng.normal(scale=0.5, size=400)
    folds = np.arange(400) % 5
    assert AN.cv_r2(y, X, folds) == pytest.approx(AN.r2(y, X), abs=0.03)
    noise = rng.normal(size=(400, 12))
    assert AN.cv_r2(y, noise, folds) < AN.r2(y, noise)                    # out-of-fold R2 does not reward overfitting
    f = AN.ols(y, X[:, 0])
    assert f["slope"] == pytest.approx(2.0, abs=0.1) and f["ci"] > 0


@pytest.mark.skipif(not os.path.exists(os.path.join("results", "rl", "checkpoints", "m47", "eval", "h2h", "s0")),
                    reason="M4.7 head-to-head records absent")
def test_recheck_replays_m47_head_to_head_games(tmp_path):
    pytest.importorskip("torch")
    from lanegame.experiments.rl.m48 import common as C
    from lanegame.experiments.rl.m48 import recheck as RC
    _, info = RC.job({"key": "t", "s": 0, "games": [3], "path": str(tmp_path / "r.json.gz")})
    assert info["max_dev"] < 1e-6                                         # float32 storage of M4.7's streams
    d = C.load_gz(str(tmp_path / "r.json.gz"))
    assert len(d["D0"]) == 1 and set(d["E"]) == {"5", "10", "20"}
