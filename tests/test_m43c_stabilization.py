"""M4.3c stabilization, Stage A (docs/m4/m4_3c_strategic_economics_calibration.md): the Shove terminology of the
four-action model, the four-action strategy library, the paired-seed harness and pins of the Stage A3 sequence
findings. Nothing here changes the game; every preset's codes, masks and checkpoints keep their meaning."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import CHAMP_ABSENT, champion_legal_actions
from lanegame.config import (FOUR_WAVE_NAMES, FOUR_WAVE_SHORT, NATURAL_WAVE_NAMES, SH, W_CLEAR, W_HOLD, W_SHOVE,
                             W_SLOWPUSH, W_THIN, champ_act, wave_display_names)
from lanegame.env import LaneEnv, UniformAgent
from lanegame.experiments.m43c import harness as h
from lanegame.experiments.m43c import sequences as sq
from lanegame.experiments.m43c import strategies as st

FOUR = GameConfig.m4_four_actions()
LASTHIT = GameConfig.m4_lasthit()
A, B = 0, 1


# ---------------------------------------------------------------------- A1: Shove terminology
def test_shove_is_the_full_capacity_slot_and_only_the_four_action_names_changed():
    assert W_SHOVE == W_CLEAR == SH == 1
    assert FOUR_WAVE_NAMES[W_SHOVE] == "Shove" and FOUR_WAVE_SHORT[W_SHOVE] == "SH"
    assert NATURAL_WAVE_NAMES[W_CLEAR] == "Clear"                        # the three-action presets keep Clear
    assert list(wave_display_names(FOUR).values()) == ["Last Hit", "Thin", "Slow Push", "Shove"]
    assert list(wave_display_names(LASTHIT).values()) == ["Hold", "Thin", "Clear"]
    assert wave_display_names(GameConfig())[SH] == "Shove"               # the scripted game's Shove
    assert [FOUR.label(champ_act(w, 0)) for w in (W_HOLD, W_THIN, W_SLOWPUSH, W_SHOVE)] == ["LH-H", "TN-H", "SL-H",
                                                                                           "SH-H"]
    assert [LASTHIT.label(champ_act(w, 0)) for w in (W_HOLD, W_THIN, W_CLEAR)] == ["HO-H", "TN-H", "CL-H"]


def test_shove_budget_is_the_capacity_as_before():
    """The rename changes no budget: Shove spends C on every state (the redesign's Clear)."""
    from lanegame.champions_m4 import context
    from lanegame.mechanics import initial_state
    ctx = context(FOUR)
    lane = initial_state(FOUR)
    for C in range(4, 24):
        assert ctx.state_budget(W_SHOVE, C, lane, 0, A) == C == ctx.budget_from_capacity(W_CLEAR, C)


# ---------------------------------------------------------------------- the strategy library
def test_every_library_strategy_plays_legal_codes_on_sampled_states():
    rng = np.random.default_rng(0)
    env, ag = LaneEnv(FOUR, max_turns=250), UniformAgent()
    obs = env.reset(seed=7)
    checked = 0
    for t in range(250):
        lane, m4 = env.state, env.champ
        for s in (A, B):
            legal = champion_legal_actions(lane, m4, s, FOUR)
            for strat in st.LIBRARY.values():
                code, why = strat.decide(FOUR, lane, m4, s, t)
                assert code in legal and why
                checked += 1
        r = env.step(ag.act(obs[0], rng), ag.act(obs[1], rng))
        if r.done:
            break
        obs = r.obs
    assert checked > 5000


def test_schedules_and_recall_rules_are_memoryless():
    s = st.LIBRARY["sp_then_shove4"]
    assert [s.wave_name(i) for i in (0, 9, 10, 30)] == ["slow_push", "slow_push", "shove", "shove"]
    with pytest.raises(ValueError, match="turn index"):
        s.decide(FOUR, *h.start("neutral", FOUR)[:2], A)
    with pytest.raises(ValueError):
        st.Strategy("bad", wave="freeze")
    with pytest.raises(ValueError):
        st.Strategy("bad", recall="sometimes")
    # a scheduled Recall fires on turn t (or t + 1 after walking back) only
    r = st.Strategy("r", recall=("at", 4))
    lane, m4, k = h.start("freeze", FOUR)                                 # A stands in its tower cell: Recall legal
    assert [r.decide(FOUR, lane, m4, A, i)[0] == st.RECALL for i in range(7)] == [False] * 4 + [True, True, False]
    # the freeze Recall needs unspent gold, which is 0 after every return
    fr = st.LIBRARY["freeze_recall4"]
    assert fr.decide(FOUR, lane, m4, A, 0)[0] != st.RECALL                 # no gold yet
    rich = m4.replace_side(A, gold=3 * FOUR.gold_scale)
    assert fr.decide(FOUR, lane, rich, A, 0)[0] == st.RECALL


def test_recall_in_contact_option_only_changes_the_contact_case():
    lane, m4, k = h.start("neutral", FOUR)
    t = __import__("lanegame.champions_m4", fromlist=["context"]).context(FOUR).t
    low = m4.replace_side(A, hp=int(0.3 * t.hp_max[3]), pos=-1)             # A at -1 in contact with B at the lane cell
    base = st.Strategy("x")
    brave = st.Strategy("y", recall_in_contact=True)
    assert base.decide(FOUR, lane, low, A, 0)[0] == st.WALK_BACK
    assert brave.decide(FOUR, lane, low, A, 0)[0] == st.RECALL


# ---------------------------------------------------------------------- the harness
def test_harness_is_reproducible_and_its_ledger_is_exact():
    cfg = FOUR
    stt = h.start("neutral", cfg, level=6)
    mk = lambda: (st.StrategyAgent(st.LIBRARY["grow_shove4"]), st.StrategyAgent(st.LIBRARY["thin4"]))
    g1 = h.play(cfg, stt, mk(), np.random.SeedSequence(5), 60)
    g2 = h.play(cfg, stt, mk(), np.random.SeedSequence(5), 60)
    assert g1["rows"] == g2["rows"]
    for r in g1["rows"]:
        for s in (A, B):
            assert r["earned"][s] == pytest.approx(r["farm"][s] + r["plate"][s] + r["bounty"][s])
            assert r["xp_avail"][s] >= r["xp"][s] + r["xp_disc"][s]
            assert r["applied"][s] <= r["budget"][s]
    # XP conservation: every enemy minion HP alive at the start or spawned is either still alive, deleted by the
    # siege cap, or available XP at its death (exactly once)
    for s in (A, B):
        e = 1 - s
        spawned = sum(r["spawned_hp"][e] for r in g1["rows"])
        overflow = sum(r["overflow_hp"][e] for r in g1["rows"])
        avail = sum(r["xp_avail"][s] for r in g1["rows"])
        assert g1["alive0"][s] + spawned == g1["alive1"][s] + overflow + avail
    w = h.window(g1, cfg)
    assert w["xp_missed"][A] == w["xp_avail"][A] - w["xp"][A] - w["xp_disc"][A]
    assert w["G"] == pytest.approx(sum(0.99 ** i * (r["earned"][A] - r["earned"][B]) for i, r in enumerate(g1["rows"])))


def test_no_chance_runs_are_deterministic_and_draw_nothing():
    cfg = FOUR
    stt = h.start("freeze", cfg)
    mk = lambda: (st.StrategyAgent(st.LIBRARY["freeze4"]), st.StrategyAgent(st.LIBRARY["walk_up4"]))
    a = h.play(cfg, stt, mk(), np.random.SeedSequence(1), 40, chance="none")
    b = h.play(cfg, stt, mk(), np.random.SeedSequence(2), 40, chance="none")
    assert a["rows"] == b["rows"]
    assert all(not any(r["ganked"]) and not any(r["died"]) and not any(r["hp_lost"]) for r in a["rows"])


# ---------------------------------------------------------------------- A3 pins (deterministic: no chance events)
def _det(stage_cell: str, a: str, b: str, spec, turns: int = 40):
    cfg = FOUR
    base, gap, level, hb = spec
    stt = h.start(base, cfg, gap=gap, level=level, hb=hb)
    g = h.play(cfg, stt, (st.StrategyAgent(sq.strategy(a)), st.StrategyAgent(sq.strategy(b))),
               np.random.SeedSequence(0), turns, chance="none")
    return h.window(g, cfg)


def test_a3_slow_push_last_hit_shove_builds_one_lead_and_crashes_it():
    w = _det("build", "grow_shove4", "farm4", ("neutral", "equal", 3, None))
    assert w["first_crash"][A] == 15 and w["crashes"] == [7, 0]
    g = _det("build", "slowpush4", "farm4", ("neutral", "equal", 3, None))
    assert g["first_crash"][A] == 16                                       # the Lanchester slow push of the redesign
    # Slow Push -> Last Hit spends exactly what Slow Push spends against a passive opponent (both are 1 HP once ahead)
    assert w["budget"][A] >= g["budget"][A]


def test_a3_thin_neutralizes_slow_push_and_shove_breaks_through():
    w = _det("sp_thin", "slowpush4", "thin4", ("neutral", "equal", 6, None))
    assert w["crashes"] == [0, 0] and w["contact_mean"] == 0
    a = _det("adapt", "sp_then_shove4", "thin4", ("neutral", "equal", 6, None))
    assert a["first_crash"][A] == 14 and a["crashes"][A] >= 10            # Shove from turn 10, first crash at 14


def test_a3_thin_corrects_an_oversized_wave_by_power():
    eq = _det("thin_hold", "thin4", "walk_up4", ("freeze", "equal", 10, 21), 20)
    lg = _det("thin_hold", "thin4", "walk_up4", ("freeze", "large", 10, 21), 20)
    assert eq["crashes"][B] == 3 and lg["crashes"][B] == 2                 # more capacity, fewer enemy crashes
    assert lg["longest_hold"][A] > eq["longest_hold"][A]
    assert eq["applied"][A] <= eq["budget"][A] and lg["applied"][A] <= lg["budget"][A]
    # and the defender gains from a crash it does not prevent (Stage B finding): Last Hit out-earns Thin here
    lh = _det("thin_hold", "farm4", "walk_up4", ("freeze", "equal", 10, 21), 20)
    assert lh["earned"][A] - lh["earned"][B] > eq["earned"][A] - eq["earned"][B]


def test_a3_recalling_while_the_freeze_holds_keeps_the_freeze():
    w = _det("recall", "freeze_recall4", "walk_up4", ("freeze", "equal", 10, None))
    n = _det("recall", "freeze4", "walk_up4", ("freeze", "equal", 10, None))
    assert w["recalled"][A] >= 3 and w["longest_hold"][A] == 40 and n["longest_hold"][A] < 40
