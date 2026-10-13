"""M4.3d (docs/m4/m4_3d_tower_pressure_calibration.md): the study's instruments (farm-loss ledger, chance masks,
post-crash strategies, the crash grid) and the configurations it compares. Nothing here changes a preset."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import AWAY, FARM, SAFE_MODE, TRADING
from lanegame.config import B_HOLD, B_SAFE, B_TRADE, PRESSURE, VISION, W_SHOVE, champ_behavior, champ_objective, \
    champ_wave
from lanegame.experiments.m43c import harness as h
from lanegame.experiments.m43c import strategies as st
from lanegame.experiments.m43d import audit, common, grid, ledger, sim
from lanegame.experiments.m43d.strategies import ATTACKERS, DEFENDERS

A, B = 0, 1


def _info(kills, mode, pos):
    return {"kills": [{}, kills], "modes": ("farm", mode), "champ_end": (0, pos)}


# ---------------------------------------------------------------------- configurations
def test_variants_validate_and_change_only_their_fields():
    econ = GameConfig.m4_four_actions_econ()
    for name, (_, preset, kw) in common.VARIANTS.items():
        cfg = common.variant_cfg(name)
        ref = getattr(GameConfig, preset)()
        diff = {f for f in cfg.__dataclass_fields__ if getattr(cfg, f) != getattr(ref, f)}
        assert diff == {k for k, v in kw.items() if getattr(ref, k) != v}, name
    assert common.variant_cfg("econ") == econ
    assert common.variant_cfg("ot90").collect_contest_own_tower == 0.9


def test_the_aggro_candidate_changes_only_the_busy_tower_shot():
    econ, aggro = GameConfig.m4_four_actions_econ(), GameConfig.m4_four_actions_econ_aggro()
    aggro.validate()
    diff = {f for f in aggro.__dataclass_fields__ if getattr(aggro, f) != getattr(econ, f)}
    assert diff == set(GameConfig.M4_FOUR_AGGRO_FIELDS)
    assert (aggro.tower_shot_p_busy, aggro.tower_adjacent_p_busy) == (aggro.tower_shot_p, aggro.tower_adjacent_p)
    # every earlier preset keeps its values
    assert (econ.tower_shot_p_busy, econ.tower_adjacent_p_busy) == (0.5, 0.25)
    for cfg in (GameConfig.m4_lasthit(), GameConfig.m4_four_actions()):
        assert (cfg.tower_shot_p_busy, cfg.tower_adjacent_p_busy, cfg.collect_contest_own_tower) == (0.5, 0.25, 0.6)
    assert common.variant_cfg("econ_aggro") == common.variant_cfg("tw_aggro") == aggro


def test_the_aggro_candidate_plays_every_non_dive_game_like_econ():
    """Without a connected trade in or next to the enemy tower, the two presets are the same game turn by turn."""
    econ, aggro = GameConfig.m4_four_actions_econ(), GameConfig.m4_four_actions_econ_aggro()
    for a, b in (("sh_press", "lh"), ("sh_vis", "thin"), ("sh_recall", "shove")):
        st0 = h.start("siege", econ, level=10)
        g0 = sim.play(econ, st0, (st.StrategyAgent(ATTACKERS[a]), st.StrategyAgent(DEFENDERS[b])), 7, 60, record=True)
        g1 = sim.play(aggro, st0, (st.StrategyAgent(ATTACKERS[a]), st.StrategyAgent(DEFENDERS[b])), 7, 60, record=True)
        if any(p > 0 for i in g0["infos"] for p in i["tower_p"]):
            continue
        assert [i["earned_units"] for i in g0["infos"]] == [i["earned_units"] for i in g1["infos"]]
        assert [i["champ_after"] for i in g0["infos"]] == [i["champ_after"] for i in g1["infos"]]


# ---------------------------------------------------------------------- the farm-loss ledger
@pytest.mark.parametrize("name", ["ot60", "econ", "ot85", "ot90"])
def test_tower_cap_is_the_only_loss_of_an_uncontested_tower_farmer(name):
    cfg = common.variant_cfg(name)
    d = ledger.farm_breakdown(_info({-2: 8}, FARM, -2), cfg, B)
    ot = cfg.collect_contest_own_tower
    assert d["ideal"] == pytest.approx(8 * 0.125 * 0.9)
    assert d["tower"] == pytest.approx(8 * 0.125 * (0.9 - ot))
    assert d["collected"] == pytest.approx(8 * 0.125 * ot)
    assert d["posture"] == d["trade"] == d["position"] == d["absence"] == 0


def test_safe_posture_costs_only_above_the_safe_base():
    """Safe in the own tower cell collects min(0.8, cap): free at a cap <= 0.8, 1/9 of the lane rate at 0.9."""
    for name, posture in (("econ", 0.0), ("ot90", 8 * 0.125 * 0.1)):
        d = ledger.farm_breakdown(_info({-2: 8}, SAFE_MODE, -2), common.variant_cfg(name), B)
        assert d["posture"] == pytest.approx(posture)
        assert d["collected"] == pytest.approx(8 * 0.125 * 0.8)


def test_trade_absence_and_position_parts():
    cfg = common.variant_cfg("econ")
    d = ledger.farm_breakdown(_info({-1: 4}, TRADING, -1), cfg, B)
    assert d["trade"] == pytest.approx(4 * 0.125 * 0.9 * 0.2) and d["collected"] == pytest.approx(4 * 0.125 * 0.72)
    assert ledger.farm_breakdown(_info({-2: 4}, AWAY, -2), cfg, B)["absence"] == pytest.approx(4 * 0.1125)
    assert ledger.farm_breakdown(_info({1: 4}, FARM, -1), cfg, B)["position"] == pytest.approx(4 * 0.1125)


@pytest.mark.parametrize("name", ["econ", "ot90"])
def test_ledger_parts_sum_to_the_ideal_in_play(name):
    cfg = common.variant_cfg(name)
    stt = h.start("siege", cfg, gap="large", level=10)
    for a, b in (("sh_dive", "safe"), ("sh_harass", "punish"), ("sh_recall", "respect")):
        g = sim.play(cfg, stt, (st.StrategyAgent(ATTACKERS[a]), st.StrategyAgent(DEFENDERS[b])), 5, 40)
        for row in g["extra"]:                       # turn_extra itself asserts collected == info["farm"]
            for s in (A, B):
                parts = sum(row[s][k] for k in ledger.FARM_PARTS)
                assert parts == pytest.approx(row[s]["ideal"], abs=1e-9)


# ---------------------------------------------------------------------- chance masks
def test_masks_keep_the_stream_aligned_and_remove_only_the_masked_events():
    cfg = common.variant_cfg("econ")
    stt = h.start("siege", cfg, level=10)
    ag = lambda: (st.StrategyAgent(ATTACKERS["sh_press"]), st.StrategyAgent(DEFENDERS["lh"]))
    g0 = sim.play(cfg, stt, ag(), 3, 60, record=True)
    g1 = sim.play(cfg, stt, ag(), 3, 60, mask="a_no_gank", record=True)
    assert any(i["ganked"][A] for i in g0["infos"])
    assert not any(i["ganked"][A] for i in g1["infos"])
    for i0, i1 in zip(g0["infos"], g1["infos"]):
        u0, u1 = np.array(i0["chance"]), np.array(i1["chance"])
        keep = [j for j in range(len(u0)) if j not in common.MASKS["a_no_gank"]]
        assert np.array_equal(u0[keep], u1[keep])
        assert all(u1[j] == 1.0 for j in common.MASKS["a_no_gank"])


# ---------------------------------------------------------------------- the post-crash strategies
def _first(name, cfg, stt):
    lane, m4, k = stt
    return ATTACKERS[name].act(cfg, lane, m4, A, 0)


def test_post_crash_strategies_use_existing_actions_as_documented():
    cfg = common.variant_cfg("econ")
    stt = h.start("siege", cfg, level=10)
    a = _first("sh_leave", cfg, stt)
    assert (champ_wave(a), champ_behavior(a)) == (W_SHOVE, B_SAFE)
    a = _first("sh_press", cfg, stt)
    assert (champ_behavior(a), champ_objective(a)) == (B_HOLD, PRESSURE)
    a = _first("sh_dive", cfg, stt)
    assert (champ_behavior(a), champ_objective(a)) == (B_TRADE, PRESSURE)
    a = _first("sh_vis", cfg, stt)
    assert champ_objective(a) == VISION
    lane, m4, k = stt
    a = ATTACKERS["sh_vis"].act(cfg, lane._replace(va=3), m4, A, 0)
    assert champ_objective(a) == PRESSURE                       # vision active: pressure under it


def test_grid_cells_reproduce():
    args = ("econ", "S4", "equal", "sh_dive", "punish", None, common.TOWER_FARM_SEED, 3, 30)
    k1, c1 = grid._job(args)
    k2, c2 = grid._job(args)
    assert k1 == k2 and c1 == c2


# ---------------------------------------------------------------------- the audit reads the engine
def test_audit_tables_are_the_engine_rules():
    cfg = common.variant_cfg("econ")
    g = {(r["behaviour"], r["vision"]): r for r in audit.gank_table(cfg)}
    assert g[("Hold, Shove", False)][2] == pytest.approx(0.15)
    assert g[("Hold, Last Hit", False)][-2] == pytest.approx(0.01)
    assert g[("Hold, Shove", True)][2] == pytest.approx(0.15 * 0.25)
    d = {(r["defensive"], r["exposure"]): r for r in audit.gank_damage_table(cfg)}
    assert d[(False, 2)]["E_units"] == pytest.approx(3.2)
    t = {r["defender"]: r for r in audit.trade_table(cfg)}
    assert t["Hold"]["in its tower cell"] == pytest.approx(0.40) and t["Safe"]["in its tower cell"] == 0.05
