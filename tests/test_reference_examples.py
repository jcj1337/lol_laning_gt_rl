"""The deterministic numbers that docs/m4/laning_game_mathematical_reference.md quotes from
`lanegame.experiments.reference.examples` and `.actions` (§3.2, §4.5, §4.8–§4.10, §6.5, §8.6, §9.3, §10.1, §10.3). A
failure means the reference is stale: update the section and its §13 change log together with these values."""

import pytest

from lanegame.experiments.reference import examples as E


@pytest.fixture(scope="module")
def waves():
    return E.wave_actions()


def test_the_worked_turn_pays_farm_a_plate_and_a_kill():
    o = E.one_turn()
    assert o["actions"] == ["SH-H+P", "LH-H"]
    assert o["farm"] == [0.7875, 0.8]
    assert o["plate_gold"] == [1.0, 0.0] and o["plates"] == [1, 0]
    assert o["kill_gold"] == [2.0, 0.0] and o["died"] == [False, True]
    assert o["earned"] == [3.7875, 0.8]
    assert o["reward_units"] == 29_875_000 and o["reward"] == pytest.approx(2.9875)


def test_four_wave_actions_from_the_neutral_start(waves):
    tot = {name: d["totals"] for name, d in waves.items()}
    assert tot["Last Hit"]["gold"] == pytest.approx([3.6, 3.6]) and tot["Last Hit"]["crash"] == [0, 0]
    assert tot["Thin"] == tot["Last Hit"]
    assert tot["Slow Push"]["gold"] == pytest.approx([3.825, 3.0375]) and tot["Slow Push"]["crash"] == [0, 0]
    assert tot["Shove"]["gold"] == pytest.approx([3.6, 3.4625]) and tot["Shove"]["crash"] == [2, 0]
    assert tot["Shove"]["minion_cHP"] == [0, 86]


def test_last_hit_is_one_at_every_power_and_the_caps_are_half_the_capacity():
    t = E.tables()
    for L in t["levels"]:
        for S in t["invest"]:
            cell = t["capacity"][L][S]
            assert cell["last_hit"] == 1
            assert cell["half"] == (cell["C"] + 1) // 2


def test_window_strengths_and_needs():
    w = E.windows()
    assert [(r["S"], r["thin_need"], r["slowpush_need"]) for r in w[:2]] == [
        ([16.0, 16.0], [0, 0], [1, 1]), ([16.0, 32.0], [16, 0], [17, 0])]


def test_evaluation_bounds_of_the_research_preset():
    b = E.bounds()["m45_calibrated"]
    assert b["omega"] == pytest.approx(145.106, abs=1e-3)
    assert b["omega_terms"]["kills"] == pytest.approx(50.756, abs=1e-3)
    assert b["per_turn_bound"] == pytest.approx(11.75)


def test_four_actions_against_a_shoving_opponent():
    tot = E.wave_actions_vs_shove()
    diff = {k: round(v["gold"][0] - v["gold"][1], 4) for k, v in tot.items()}
    assert diff == {"Last Hit": -0.1375, "Thin": -0.4, "Slow Push": -0.4125, "Shove": 0.0}


def test_one_turn_kills_and_retaliation():
    k = E.kills()
    assert [r["taken"] for r in k["combination"]] == [700, 850] and all(r["died"] for r in k["combination"])
    assert [(r["taken"], r["died"]) for r in k["positional"]] == [(450, False), (550, False), (650, True)]
    assert [(r["taken"], r["died"]) for r in k["double_hit"][:2]] == [(991, True), (917, False)]
    assert [r["minion_cHP"][0] for r in E.retaliation()] == [46, 139]


def test_a_walk_up_shove_and_the_freeze():
    rows = {(r["held_hp"], r["L"]): r for r in E.freeze_break(draws=20)}
    for (held, L), r in rows.items():
        assert (r["guard"]["survives"] == 1.0) == (r["C_B"] <= 7)
        assert (r["keep"]["survives"] == 1.0) == (held == 5 and r["C_B"] <= 5)


def test_the_low_hp_death_odds():
    r = {(x["L"], x["a"], x["b"]): x["p_die"] for x in E.low_hp()}
    assert r[(3, "Last Hit + Hold", "Trade")] == pytest.approx(0.220, abs=5e-4)
    assert r[(18, "Last Hit + Hold", "Hold")] == pytest.approx(0.016, abs=5e-4)
    assert r[(18, "Last Hit + Safe", "Trade")] < 0.001


def test_the_codes_that_can_ever_be_legal():
    from lanegame.experiments.reference.actions import exhaustive
    ex = exhaustive()
    assert ex["n_ever_legal"] == 27 and max(ex["size_by_position"].values()) == 25
