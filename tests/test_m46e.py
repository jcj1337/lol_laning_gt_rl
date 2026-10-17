"""M4.6e diagnostic instrumentation: the observation checks, state families, label variants, splits and readouts."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lanegame.experiments.rl.m46e import analysis as AN
from lanegame.experiments.rl.m46e import common as C
from lanegame.experiments.rl.m46e import exposure as X
from lanegame.experiments.rl.m46e import fit as F
from lanegame.experiments.rl.m46e import labels as LB
from lanegame.experiments.rl.m46e import observation as O
from lanegame.experiments.rl.m46e.features import StateFeatures, explicit, obs_index

CKPT = os.path.join("results", "rl", "checkpoints", "m46c", "train", "ext", "C1", "s0", "final.pt")
CALIB = os.path.join("results", "rl", "checkpoints", "m46d", "critic", "calib", "C1", "s0", "gap_3.json.gz")


def _cfg():
    return C.arm_cfg(C.ARM)


def test_single_feature_pairs_change_exactly_the_intended_entries():
    from lanegame.env.observation_m4 import m4_feature_names
    cfg = _cfg()
    names = m4_feature_names(cfg)
    for bname, st in O.base_states(cfg).items():
        a0, b0 = O._obs(cfg, st, 0).vector, O._obs(cfg, st, 1).vector
        for pname, (fn, req, opt) in O.perturbations(cfg).items():
            st1 = fn(st)
            ca = O._changed(a0, O._obs(cfg, st1, 0).vector, names)
            cb = O._changed(b0, O._obs(cfg, st1, 1).vector, names)
            assert O._expected_ok(ca, req, opt, "A"), (bname, pname, ca)
            assert O._expected_ok(cb, req, opt, "B"), (bname, pname, cb)


def test_expected_entry_check_rejects_a_wrong_block():
    assert not O._expected_ok({"enemy_champ_hp_frac": -0.4}, {"champ_hp_frac"}, set(), "A")
    assert O._expected_ok({"enemy_champ_hp_frac": -0.4}, {"champ_hp_frac"}, set(), "B")
    assert not O._expected_ok({"own_champ_xp_progress": 0.5}, {"CELL"}, set(), "A")   # not a cell entry


def test_explicit_log_duel_equals_the_state_duel_ratio():
    from lanegame.experiments.rl.m46d import cells as CL
    cfg = _cfg()
    sf, idx = StateFeatures(cfg), obs_index(cfg)
    for cond in ("equal", "edge", "x1.76", "brk"):
        for L in (3, 8, 14):
            st = CL.build(cfg, "neutral", cond, L, 0.6)
            x = explicit(O._obs(cfg, st, 0).vector[None], idx)[0]
            assert abs(x[0] - np.log(sf.duel(st[1], 0))) < 2e-3
            assert abs(x[2] - 0.6) < 1e-2                                # the enemy's HP fraction


def test_families_on_a_synthetic_state():
    cols = {c: i for i, c in enumerate(X.COLS)}
    x = np.zeros((3, 2, len(X.COLS)))
    x[:, :, cols["present"]] = 1
    x[:, :, cols["L"]] = 5
    x[:, :, cols["hp"]] = 1.0
    x[:, :, cols["dist"]] = -1
    x[0, 0, cols["duel"]], x[0, 1, cols["duel"]] = 1.3, 1 / 1.3          # plausible early edge for A
    x[1, 0, cols["duel"]], x[1, 1, cols["duel"]] = 1 / 3.2, 3.2          # extreme gap for B
    x[2, 0, cols["duel"]], x[2, 1, cols["duel"]] = 1.2, 1 / 1.2          # A freezes, B damaged and walked up
    x[2, 0, cols["held"]], x[2, 0, cols["held_hp"]] = 1, 12
    x[2, 1, cols["hp"]] = 0.4
    x[2, 1, cols["catch_o"]], x[2, 1, cols["dist"]] = 1, 1
    f = X.families(x)
    assert f["plaus_early"].tolist() == [True, False, True] and f["plaus_early_125"].tolist() == [True, False, False]
    assert f["extreme"].tolist() == [False, True, False] and f["large"].tolist() == [False, True, False]
    assert f["fav_freeze"].tolist() == [False, False, True] and f["fav_freeze_dmg"].tolist() == [False, False, True]
    assert f["weak_window"].tolist() == [False, False, True]


def test_label_variants_change_only_the_intended_quantity():
    from lanegame.experiments.m45b import states as S
    cfg = _cfg()
    sf = StateFeatures(cfg)
    lane, pa, pb, _ = S.ZONING_SITUATIONS["neutral"]
    m4 = S.champs(cfg, (8, 8), (9.41 + 3.5, 9.41), (pa, pb), (1.0, 0.8))
    item = {"kind": "zone", "label": "zone neutral L8 S9.41 d=3.5", "group": 7, "start": (lane, m4, 2)}
    v = LB.variants(cfg, item)
    assert set(v) == {"orig", "eq", "lvl", "hpB", "big"}
    assert v["eq"][1].a.items == v["eq"][1].b.items
    assert sf.level(v["lvl"][1], 0) == 9 and v["lvl"][1].a.items == m4.a.items
    assert v["hpB"][1].b.hp == int(round(0.6 * m4.b.hp)) and v["hpB"][1].a == m4.a
    m = S.multiplier(cfg, v["big"][1].a.items / cfg.gold_scale) / S.multiplier(cfg, m4.b.items / cfg.gold_scale)
    assert 1.29 <= m <= 1.91
    nl = {"kind": "natural+lead", "label": "natural+lead s1 d=2.5", "group": 0, "start": (lane, m4, 2)}
    assert LB.variants(cfg, nl)["eq"][1].a.items == m4.a.items - int(round(2.5 * cfg.gold_scale))


def test_group_split_is_deterministic_and_keeps_groups_whole():
    groups = np.repeat(np.arange(40), 6)
    kinds = {g: g % 3 for g in range(40)}
    a, b = F.group_split(groups, kinds), F.group_split(groups, kinds)
    assert a == b
    assert {v for v in a.values()} == {"train", "val", "test"}
    assert abs(sum(v == "test" for v in a.values()) - 10) <= 2


def test_ridge_recovers_a_linear_target():
    rng = np.random.default_rng(0)
    Xa = rng.normal(size=(500, 4))
    y = Xa @ np.array([1.0, -2.0, 0.5, 0.0]) + 3.0
    m = F.ridge(Xa[:400], y[:400], np.ones(400), Xa[400:], y[400:], np.ones(100))
    assert np.abs(F.ridge_predict(m, Xa[400:]) - y[400:]).max() < 0.05


def test_pair_premiums_on_synthetic_examples():
    rows = []
    for g in range(4):
        for v, val in ((0, 1.0), (1, 0.0), (4, 3.0)):                    # orig, eq, big
            for r in (0, 1):
                for s in (0, 1):
                    rows.append((g, v, r, s, (val + 0.1 * r) * (1 if s == 0 else -1)))
    a = np.asarray(rows, float)
    e = {"group": a[:, 0].astype(int), "variant": a[:, 1].astype(int), "rollout": a[:, 2].astype(int),
         "seat": a[:, 3].astype(int), "y": a[:, 4], "t": np.zeros(len(a), int), "src": np.ones(len(a), int)}
    pred = e["y"] * 0.5
    out = AN.pair_premiums(e, np.ones(len(a), bool), pred)
    assert out["lead"]["realized"] == pytest.approx(1.0) and out["lead"]["ratio"] == pytest.approx(0.5)
    assert out["big"]["realized"] == pytest.approx(2.0) and out["big"]["ratio"] == pytest.approx(0.5)


@pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(CALIB)), reason="needs the M4.6c/M4.6d records")
def test_rollout_core_reproduces_an_m46d_calibration_rollout():
    from lanegame.experiments.rl.m46d import common as D
    cfg = C.setup(C.ARM)
    rec = C.load_gz(CALIB)
    start = {(f, i): st for f, i, st, _ in LB._m46d_cells(cfg)}[("gap", 3)]
    rew = LB.play(cfg, C.net_of("ext/C1:0"), start, D.CRIT_ROOT * 1000, 300)
    assert [C.discounted(rew, 0, h) for h in rec["cps"]] == pytest.approx(rec["G"][0], abs=1e-9)
