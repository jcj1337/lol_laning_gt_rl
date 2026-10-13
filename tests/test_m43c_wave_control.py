"""M4.3c wave-control screen (docs/m4/m4_3c_wave_control_screen.md): the EXPERIMENTAL opt-in fields wave_hold_cap and
wave_thin_rule "adaptive", their default preservation, the adaptive Thin target (mechanics.thin_need), the global
correctness guarantees under the adaptive rule, and pins of the screen's key findings."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import NO_REACH
from lanegame.champions_m4 import context
from lanegame.config import B_HOLD, F, SH, SP, W_CLEAR, W_HOLD, W_THIN, champ_act
from lanegame.experiments.m43c import wave_control as wc
from lanegame.mechanics import initial_state, step_turn, thin_need, total_hp
from lanegame.state import State

BASE = GameConfig.m4_lasthit()
HOLD1 = GameConfig.m4_lasthit(wave_hold_cap=1)
ADAPT = GameConfig.m4_lasthit(wave_hold_cap=1, wave_thin_rule="adaptive")
T = BASE.tower


def _lane(ax, ah, bx, bh, ar=(), br=()):
    return State(ax, ah, tuple(ar), 0, bx, bh, tuple(br), 0, 0, 0)


def _P(lane, cfg=ADAPT):
    return wc.packets(lane, cfg)


# ---------------------------------------------------------------------- defaults and budgets
def test_defaults_are_off_and_the_research_preset_keeps_its_budgets():
    for cfg in (GameConfig(), GameConfig.m4_lane(), BASE):
        assert (cfg.wave_hold_cap, cfg.wave_thin_rule) == (0, "share")
    ctx = context(BASE)
    for C in range(4, 24):
        assert ctx.budget_from_capacity(SP, C) == (2 * C + 8) // 16        # round(C / 8), half up
        assert ctx.budget_from_capacity(F, C) == (C + 1) // 2               # round(C / 2), half up
        assert ctx.budget_from_capacity(SH, C) == C


def test_hold_cap_fixes_hold_at_one_and_leaves_thin_and_clear():
    base, cap = context(BASE), context(HOLD1)
    for C in range(4, 24):
        assert cap.budget_from_capacity(SP, C) == 1
        assert cap.budget_from_capacity(F, C) == base.budget_from_capacity(F, C)
        assert cap.budget_from_capacity(SH, C) == C


def test_validation_of_the_new_fields():
    for cfg in (HOLD1, ADAPT):
        cfg.validate()
    for kw, field in ((dict(wave_thin_rule="adaptive", damage_law="lanchester"), "wave_thin_rule"),
                      (dict(wave_thin_rule="oracle"), "wave_thin_rule"),
                      (dict(wave_hold_cap=-1), "wave_hold_cap")):
        with pytest.raises(ValueError, match=field):
            GameConfig.m4_lasthit(**kw).validate()
    for kw in (dict(wave_hold_cap=1), dict(wave_thin_rule="adaptive")):
        with pytest.raises(ValueError):
            GameConfig.m4_lane(**kw).validate()                 # wave_clear_scaling "attack_power"
        with pytest.raises(ValueError):
            GameConfig(**kw).validate()                         # scripted waves


# ---------------------------------------------------------------------- the adaptive target
@pytest.mark.parametrize("own, enemy, need", [(8, 12, 1), (8, 11, 0), (8, 24, 13), (12, 15, 0), (12, 16, 1),
                                              (16, 19, 0), (4, 20, 13), (1, 8, 1), (24, 24, 0)])
def test_thin_need_is_the_enemy_hp_above_the_firepower_parity_target(own, enemy, need):
    # f(h) = max(1, floor(2h / 8)); target = the largest h with f(h) <= f(own): 7, 11, 15, 19, 23, 27
    assert thin_need(_P(_lane(0, own, 0, enemy)), 0, ADAPT) == need
    assert thin_need(_P(_lane(0, enemy, 0, own)), 1, ADAPT) == need          # either seat


def test_thin_need_counts_queued_packets_and_needs_an_own_packet():
    # B's front of 12 at the lane with a fresh wave queued one cell behind it (B's own -1): blob 20 against 8
    assert thin_need(_P(_lane(0, 8, 0, 12, br=(-1,))), 0, ADAPT) == 20 - 11
    # a packet two cells behind is not in the blob
    assert thin_need(_P(_lane(0, 8, 0, 12, br=(-2,))), 0, ADAPT) == 1
    # no own packet at all: the whole enemy blob is excess; an own rear packet alone is the reference
    assert thin_need(_P(_lane(0, 0, 0, 12)), 0, ADAPT) == 12
    assert thin_need(_P(_lane(-4, 8, 0, 12)), 0, ADAPT) == 1


def test_adaptive_thin_spends_only_the_need_and_clear_is_unchanged():
    lane = _lane(0, 8, 0, 24)
    reach = (T, T)
    for cfg, slot, budget, removed in ((ADAPT, W_THIN, 11, 11), (ADAPT, W_THIN, 22, 13), (BASE, W_THIN, 22, 22),
                                       (ADAPT, W_CLEAR, 22, 22), (ADAPT, W_HOLD, 1, 1)):
        r = step_turn(lane, 9, slot, W_HOLD, cfg, champion_reach=reach, champion_budget=(budget, 0))
        assert r.events["champion_hp"][1] == removed, (cfg.wave_thin_rule, slot, budget)
    # a balanced lane: Thin deals nothing, Hold still last-hits
    lane = _lane(0, 8, 0, 8)
    assert step_turn(lane, 9, W_THIN, W_HOLD, ADAPT, champion_reach=reach,
                     champion_budget=(11, 0)).events["champion_hp"][1] == 0


def test_hp_conservation_and_mirror_symmetry_under_the_adaptive_rule():
    rng = np.random.default_rng(13)
    st = initial_state(ADAPT)
    for k in range(1500):
        a, b = (int(rng.integers(3)) for _ in (0, 1))
        reach = tuple(int(rng.choice([NO_REACH, -2, -1, 0, 1, 2, 3])) for _ in (0, 1))
        budget = tuple(int(rng.integers(0, 24)) for _ in (0, 1))
        r = step_turn(st, k, a, b, ADAPT, champion_reach=reach, champion_budget=budget)
        before, after, ev = total_hp(st, ADAPT), total_hp(r.state, ADAPT), r.events
        P = wc.packets(st, ADAPT)
        for s, w in ((0, a), (1, b)):
            assert before[s] + ev["spawned_hp"][s] == after[s] + ev["killed_hp"][s] + ev["tower_hp"][s]
            assert ev["overflow_hp"][s] == 0
            if w == F:
                assert ev["champion_hp"][1 - s] <= min(budget[s], thin_need(P, s, ADAPT))
        m = step_turn(st.mirror(), k, b, a, ADAPT, champion_reach=reach[::-1], champion_budget=budget[::-1])
        assert m.state == r.state.mirror() and m.kills == r.kills[::-1]
        assert m.events["champion_hp"] == r.events["champion_hp"][::-1]
        st = r.state


# ---------------------------------------------------------------------- pins of the screen
@pytest.fixture(scope="module")
def starts():
    from lanegame.experiments.m4.zoning_natural import starts as st
    return st(BASE)


def _freeze(starts, c, tier, fz, resp):
    cfg = wc.response_cfg(c, resp)
    fa, code_b = wc.freezer(cfg, fz), wc.RESPONSE_CODE[resp]
    return wc.stats(wc.play(cfg, starts["freeze"], wc.capacities(tier), lambda i, l, m: (fa(l, m), code_b),
                            wc.FREEZE_TURNS)["rows"], start_held=1)


def test_pin_capacities_of_the_tiers():
    assert {t: wc.capacities(t) for t in wc.TIERS} == {
        "L3": (4, 4), "L3 A+": (8, 4), "L6": (5, 5), "L6 A+": (9, 5), "L10": (6, 6), "L10 A+": (11, 6),
        "L13 A+": (13, 7), "L18 A++": (22, 9), "L18 +64": (17, 17)}


def test_pin_hold1_removes_the_strong_last_hitters_push(starts):
    hold = lambda: wc.const(champ_act(W_HOLD, B_HOLD), champ_act(W_HOLD, B_HOLD))
    caps = wc.capacities("L18 A++")
    base = wc.stats(wc.play(BASE, starts["neutral"], caps, hold(), 60)["rows"])
    cap = wc.stats(wc.play(HOLD1, starts["neutral"], caps, hold(), 60)["rows"])
    assert base["crashes"] == [11, 0] and cap["crashes"] == [0, 0] and cap["contact_turns"] == 60
    a, b = _freeze(starts, "A", "L13 A+", "threshold", "walk_up"), _freeze(starts, "B", "L13 A+", "threshold", "walk_up")
    assert (a["held"], a["first_break"]) == (17, 7) and (b["held"], b["first_break"]) == (19, 19)


def test_pin_adaptive_thin_freeze_and_its_collapse_against_a_retreat(starts):
    a = _freeze(starts, "A", "L10", "thin", "walk_up")
    c = _freeze(starts, "C", "L10", "thin", "walk_up")
    r = _freeze(starts, "C", "L10", "thin", "retreat")
    assert (a["held"], a["crashes"][0]) == (2, 7)
    assert (c["held"], c["longest_hold"]) == (25, 25)
    assert (r["held"], r["crashes"][0], r["a_thin_zero"]) == (11, 5, 36)     # A deals 0 on 36 of 40 turns, still pushes
    assert _freeze(starts, "C", "L10", "thin", "clear")["longest_hold"] == 2  # B's Clear still breaks it


def test_pin_imbalance_correction(starts):
    st = wc.imbalance_start(starts["neutral"], 8, 24)
    thin = wc.const(champ_act(W_THIN, B_HOLD), champ_act(W_HOLD, B_HOLD))
    rows = wc.play(BASE, st, (22, 6), thin, 12)["rows"]
    assert sum(max(0, r["dmg"][0] - r["need"][0]) for r in rows) == 36 and wc.stats(rows)["crashes"][0] == 5
    rows = wc.play(ADAPT, st, (22, 6), thin, 12)["rows"]
    assert [r["dmg"][0] for r in rows[:4]] == [11, 7, 3, 1] and all(r["dmg"][0] <= r["need"][0] for r in rows)
    assert all(r["contact"] == 0 for r in rows) and wc.stats(rows)["crashes"] == [0, 0]
    rows = wc.play(ADAPT, st, (4, 6), thin, 12)["rows"]                        # level 3: Thin 2 cannot correct 13
    assert wc.stats(rows)["crashes"] == [0, 4]
