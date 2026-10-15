"""M4.3c four-action wave management (docs/m4/m4_3/m4_3c_four_action_wave_redesign.md): the EXPERIMENTAL preset
GameConfig.m4_four_actions() (Last Hit / Thin / Slow Push / Shove; the report calls slot SH Clear), the 61-code action
encoding and its compatibility with every historical code, the window-strength target (mechanics.window_strength /
strength_need), the budgets, the global correctness guarantees on the new preset, the RL interfaces, and pins of the
report's findings. The `CL` names below are the redesign study's own (lanegame.experiments.m43c.four_actions)."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import N_CHANCE, champion_legal_actions, mirror_chance, swap_seats
from lanegame.champions_m4 import context, initial_m4, resolve_turn_m4
from lanegame.config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, CHAMP_SLOWPUSH_BASE, F, N_CHAMP_ACTIONS,
                             N_CHAMP_ACTIONS_FOUR, NATURAL_WAVE_SHORT, CHAMP_ACTION_SHORT, PRESSURE, SH, SP, VISION,
                             W_CLEAR, W_HOLD, W_SLOWPUSH, W_THIN, WAVE_SHORT, champ_act, champ_behavior,
                             champ_objective, champ_wave)
from lanegame.env import LaneEnv, UniformAgent
from lanegame.experiments.m43c import four_actions as fa
from lanegame.mechanics import (STRENGTH_MARGIN, initial_state, start_packets, strength_need, total_hp, wave_actions,
                                window_strength)
from lanegame.state import State

FOUR = GameConfig.m4_four_actions()
LASTHIT = GameConfig.m4_lasthit()
LH, CL, TN, SL = W_HOLD, W_CLEAR, W_THIN, W_SLOWPUSH
A, B = 0, 1


def _lane(ax, ah, bx, bh, ar=(), br=(), asg=0, bsg=0):
    return State(ax, ah, tuple(ar), asg, bx, bh, tuple(br), bsg, 0, 0)


@pytest.fixture(scope="module")
def starts():
    from lanegame.experiments.m4.zoning_natural import starts as make
    return make(GameConfig.m4_lasthit())


# ---------------------------------------------------------------------- encoding and compatibility
def test_every_historical_code_keeps_its_meaning_and_slow_push_codes_are_appended():
    assert (CHAMP_ABSENT, N_CHAMP_ACTIONS, CHAMP_SLOWPUSH_BASE, N_CHAMP_ACTIONS_FOUR) == (45, 46, 46, 61)
    for a in range(45):                                       # the M2-M4.3b formula, unchanged
        assert (champ_wave(a), champ_behavior(a), champ_objective(a)) == (a % 3, (a % 15) // 3, a // 15)
        assert champ_act(a % 3, (a % 15) // 3, a // 15) == a
    assert (champ_wave(45), champ_behavior(45), champ_objective(45)) == (0, 0, 3)     # as before (ABSENT decodes apart)
    for b in range(5):
        for o in range(3):
            a = champ_act(W_SLOWPUSH, b, o)
            assert a == 46 + b + 5 * o and (champ_wave(a), champ_behavior(a), champ_objective(a)) == (W_SLOWPUSH, b, o)
    assert sorted(champ_act(W_SLOWPUSH, b, o) for b in range(5) for o in range(3)) == list(range(46, 61))


def test_action_spaces_labels_and_old_presets():
    assert FOUR.action_codes() == tuple(range(61)) and LaneEnv(FOUR).n_actions == 61 and LaneEnv(FOUR).obs_size == 88
    for cfg in (LASTHIT, GameConfig.m4_lane(), GameConfig.m4_natural(), GameConfig.m4(), GameConfig.champions()):
        assert cfg.action_codes() == tuple(range(46)) and not cfg.wave_slowpush
    for cfg in (LASTHIT, GameConfig.m4_lane(), GameConfig.m4_natural()):     # labels byte-identical to the old rule
        for a in range(45):
            assert cfg.label(a) == NATURAL_WAVE_SHORT[champ_wave(a)] + CHAMP_ACTION_SHORT[a][len(WAVE_SHORT[a % 3]):]
    assert [FOUR.label(a) for a in (0, 1, 2, 12, 45, 46, 48, 51, 58)] == [
        "LH-H", "SH-H", "TN-H", "LH-R", "ABSENT", "SL-H", "SL-T", "SL-H+P", "SL-T+V"]   # slot SH: Shove


def test_preset_is_m4_lasthit_with_the_documented_fields_and_validation():
    FOUR.validate()
    changed = {k for k, v in FOUR.to_dict().items() if LASTHIT.to_dict()[k] != v}
    assert changed == set(GameConfig.M4_FOUR_ACTIONS_FIELDS)
    assert (FOUR.wave_hold_cap, FOUR.wave_thin_rule, FOUR.wave_slowpush) == (1, "strength", True)
    assert GameConfig.from_dict({k: v for k, v in LASTHIT.to_dict().items() if k != "wave_slowpush"}) == LASTHIT
    with pytest.raises(ValueError, match="wave_slowpush"):
        GameConfig.m4_lasthit(wave_slowpush=True).validate()                 # needs the "strength" Thin rule
    with pytest.raises(ValueError, match="'capacity' only"):
        GameConfig.m4_lane(wave_slowpush=True, wave_thin_rule="strength").validate()   # "capacity" scaling only
    with pytest.raises(ValueError, match="wave_thin_rule"):
        GameConfig.m4_lane(wave_thin_rule="strength").validate()
    with pytest.raises(ValueError, match="wave_slowpush"):
        GameConfig.champions(wave_slowpush=True).validate()
    with pytest.raises(ValueError, match="wave_thin_rule"):
        GameConfig.m4_lasthit(wave_thin_rule="target").validate()
    GameConfig.m4_lasthit(wave_thin_rule="strength").validate()               # Thin rule alone: three actions


def test_legal_actions_add_a_slow_push_twin_of_every_legal_hold_code_except_recall():
    rng = np.random.default_rng(5)
    env, ag = LaneEnv(FOUR, max_turns=300), UniformAgent()
    obs = env.reset(seed=2)
    seen_objective = False
    for _ in range(300):
        lane, m4 = env.state, env.champ
        for s in (A, B):
            four = champion_legal_actions(lane, m4, s, FOUR)
            three = champion_legal_actions(lane, m4, s, LASTHIT)
            if four == (CHAMP_ABSENT,):
                assert three == (CHAMP_ABSENT,)
                continue
            twins = {champ_act(SL, champ_behavior(a), champ_objective(a)) for a in three if a != champ_act(LH, B_RECALL)}
            assert set(four) == set(three) | twins and not set(four) & fa.NEVER_LEGAL
            seen_objective |= any(champ_objective(a) for a in four)
        r = env.step(ag.act(obs[0], rng), ag.act(obs[1], rng))
        if r.done:
            break
        obs = r.obs
    assert seen_objective
    assert wave_actions(initial_state(FOUR), A, FOUR) == (SP, SH, F, W_SLOWPUSH)
    assert wave_actions(initial_state(LASTHIT), A, LASTHIT) == (SP, SH, F)


# ---------------------------------------------------------------------- window strength and need
def test_window_strength_weights_reinforcements_by_their_arrival():
    cfg = FOUR
    assert window_strength([], 0, cfg) == 0
    assert window_strength([[0, 13]], 0, cfg) == 13                        # k = 0: no spawn at the end of the turn
    for d, w in ((1, 4), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0)):           # a rear d cells behind fights w of 4 ticks
        assert window_strength([[1, 8], [1 - d, 8]], 0, cfg) * 4 == 8 * 4 + 8 * w
    for fx, w in ((-4, 2), (-3, 2), (-2, 2), (-1, 1), (0, 0), (1, 0)):     # the wave spawning at the end of turn k = 1
        assert window_strength([[fx, 8]], 1, cfg) * 4 == 8 * 4 + 8 * w
        assert window_strength([[fx, 8]], 2, cfg) == 8                     # no spawn after an even turn


def test_strength_need_on_the_calibrated_states(starts):
    lane, m4, k = starts["freeze"]
    P = start_packets(lane, FOUR)
    # the freezer's quick reinforcement makes it the stronger side although its front is smaller
    assert (window_strength(P[A], k, FOUR), window_strength(P[B], k, FOUR)) == (12, 11)
    assert [strength_need(P, A, k, m, FOUR) for m in (0, 1)] == [0, 0]
    assert [strength_need(P, B, k, m, FOUR) for m in (0, 1)] == [1, 2]
    lane, m4, k = starts["neutral"]
    P = start_packets(lane, FOUR)
    assert window_strength(P[A], k, FOUR) == window_strength(P[B], k, FOUR)
    assert [strength_need(P, s, k, STRENGTH_MARGIN[w], FOUR) for s in (A, B) for w in (F, SL)] == [0, 1, 0, 1]
    P = start_packets(lane._replace(bh=24), FOUR)                          # 8 v 24 at the lane cell: 16 HP of excess
    assert strength_need(P, A, k, 0, FOUR) == 16 and strength_need(P, A, k, 1, FOUR) == 17
    assert strength_need(P, B, k, 0, FOUR) == 0 and strength_need(P, B, k, 1, FOUR) == 0


def test_budgets_last_hit_thin_slow_push_clear_on_every_capacity(starts):
    ctx = context(FOUR)
    for name, hb in (("neutral", 8), ("neutral", 24), ("freeze", None)):
        lane, m4, k = starts[name]
        if hb is not None:
            lane = lane._replace(bh=hb)
        P = start_packets(lane, FOUR)
        for s in (A, B):
            need = {TN: strength_need(P, s, k, 0, FOUR), SL: strength_need(P, s, k, 1, FOUR)}
            prev = None
            for C in range(4, 24):
                b = {a: ctx.state_budget(a, C, lane, k, s) for a in (LH, TN, SL, CL)}
                assert b[LH] == 1 and b[CL] == C
                assert b[TN] == min((C + 1) // 2, 1 + need[TN]) and b[SL] == min((C + 1) // 2, 1 + need[SL])
                assert 1 <= b[TN] <= b[SL] <= (C + 1) // 2 < C
                if prev:
                    assert all(b[a] >= prev[a] for a in b)                    # never less with more capacity
                prev = b
    # the old presets: state_budget is the share model, whatever the lane
    for cfg in (LASTHIT, GameConfig.m4_lasthit(wave_hold_cap=1, wave_thin_rule="adaptive")):
        c2 = context(cfg)
        lane, m4, k = starts["neutral"]
        for C in range(4, 24):
            for slot in (SP, SH, F):
                assert c2.state_budget(slot, C, lane._replace(bh=24), k, A) == c2.budget_from_capacity(slot, C)


# ---------------------------------------------------------------------- engine guarantees on the new preset
def test_resolver_records_its_budgets_conserves_hp_and_is_mirror_equivariant():
    ctx = context(FOUR)
    rng, crng = np.random.default_rng(8), np.random.default_rng(9)
    env, ag = LaneEnv(FOUR, max_turns=400), UniformAgent()
    obs = env.reset(seed=4)
    slow = 0
    for _ in range(400):
        lane, m4, k = env.state, env.champ, env.k
        a, b = ag.act(obs[0], rng), ag.act(obs[1], rng)
        slow += (a >= 46) + (b >= 46)
        u = tuple(float(x) for x in crng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k, a, b, FOUR, u=u)
        assert t.wave_budget == tuple(ctx.state_budget(t.waves[s], ctx.capacity(t.levels_before[s], m4.side(s).items),
                                                       lane, k, s) for s in (A, B))
        m = resolve_turn_m4(lane.mirror(), m4.mirror(), k, b, a, swap_seats(FOUR), u=mirror_chance(u))
        assert m.champ == t.champ.mirror() and m.lane.state == t.lane.state.mirror()
        assert m.reward_units == -t.reward_units and m.wave_budget == t.wave_budget[::-1]
        before, after, ev = total_hp(lane, FOUR), total_hp(t.lane.state, FOUR), t.lane.events
        for s in (A, B):
            assert before[s] + ev["spawned_hp"][s] == after[s] + ev["killed_hp"][s] + ev["tower_hp"][s]
            assert ev["champion_hp"][1 - s] <= t.wave_budget[s]
        r = env.step(a, b)
        assert r.info["wave_budget"] == t.wave_budget and r.info["legal"][0] == obs[0].legal
        if r.done:
            break
        obs = r.obs
    assert slow > 50


def test_old_presets_record_their_unchanged_budgets():
    for cfg in (LASTHIT, GameConfig.m4_lane()):
        ctx = context(cfg)
        lane, m4 = initial_state(cfg), initial_m4(cfg)
        for k in range(30):
            t = resolve_turn_m4(lane, m4, k, champ_act(TN, B_HOLD), champ_act(CL, B_HOLD), cfg, u=None)
            want = tuple(ctx.wave_budget(t.waves[s], t.levels_before[s], m4.side(s).items) for s in (A, B))
            assert t.wave_budget == want
            lane, m4 = t.lane.state, t.champ


def test_a_46_output_network_refuses_the_61_code_space():
    from lanegame.rl import NetAgent, PolicyValueNet
    obs = LaneEnv(FOUR, max_turns=5).reset(seed=0)
    with pytest.raises(ValueError, match="61 codes"):
        NetAgent(PolicyValueNet(88, 46, (16,)), "old").act(obs[0], np.random.default_rng(0))
    obs3 = LaneEnv(LASTHIT, max_turns=5).reset(seed=0)
    assert NetAgent(PolicyValueNet(88, 46, (16,)), "old").act(obs3[0], np.random.default_rng(0)) in obs3[0].legal


def test_ppo_trains_on_the_61_code_space_and_never_samples_an_illegal_action():
    torch = pytest.importorskip("torch")
    from lanegame.rl import NetAgent, PPOConfig, SharedConfig, train_shared
    ppo = PPOConfig.continuing(FOUR, total_steps=800, n_envs=4, segment_length=24, game_length=40, minibatch_size=64,
                               epochs=1, seed=3, hidden=(32, 32), lane_cache=2000)
    res = train_shared(FOUR, ppo, SharedConfig(current_fraction=1.0), verbose=False)
    assert res.net.n_actions == 61 and res.agent_transitions >= 800
    for e in res.history:
        assert all(np.isfinite(e[k]) for k in ("policy_loss", "value_loss", "entropy", "approx_kl"))
    ag, env, rng = NetAgent(res.net, "smoke"), LaneEnv(FOUR, max_turns=60), np.random.default_rng(0)
    obs = env.reset(seed=1)
    while True:
        a, b = ag.act(obs[0], rng), ag.act(obs[1], rng)
        assert a in obs[0].legal and b in obs[1].legal
        p = ag.probabilities(obs[0])
        assert p.shape == (61,) and abs(p.sum() - 1) < 1e-9 and p[~obs[0].mask].sum() == 0
        r = env.step(a, b)
        if r.done:
            break
        obs = r.obs


def test_pettingzoo_parallel_api_on_the_four_action_preset():
    pytest.importorskip("pettingzoo")
    pytest.importorskip("gymnasium")
    from pettingzoo.test import parallel_api_test
    from lanegame.env.pettingzoo_adapter import LaneParallelEnv
    env = LaneParallelEnv(FOUR, max_turns=50)
    parallel_api_test(env, num_cycles=40)
    assert env.action_space("A").n == 61


# ---------------------------------------------------------------------- pins of the report's findings
def _run(start, caps, a, b, turns, resp="walk_up"):
    return fa.stats(fa.play(fa.response_cfg(resp), start, caps, fa.const(fa.code(a), b), turns)["rows"],
                    start_held=1)


def test_test1_neutral_lane(starts):
    for caps in ((4, 4), (22, 9)):
        lh = _run(starts["neutral"], caps, LH, fa.code(LH), 60)
        tn = _run(starts["neutral"], caps, TN, fa.code(LH), 60)
        assert lh["crashes"] == tn["crashes"] == [0, 0] and tn["dmg"] == lh["dmg"] == [60, 60]
        assert tn["contact_turns"] == 60 and tn["contact_mean"] == 0
    for caps in ((6, 6), (11, 6), (22, 9)):                                  # power-independent slow push
        sp = _run(starts["neutral"], caps, SL, fa.code(LH), 60)
        assert sp["first_crash"] == [16, None] and sp["crashes"] == [7, 0] and sp["dmg"][0] <= 70
        cl = _run(starts["neutral"], caps, CL, fa.code(LH), 60)
        assert cl["first_crash"][0] <= 4 and cl["crashes"][0] >= 15


def test_test2_thin_corrects_an_oversized_wave_by_power_without_overshoot(starts):
    start = fa.with_fronts(starts["neutral"], 8, 24)
    crashes_b = []
    for C in (4, 6, 11, 22):
        r = fa.play(fa.cfg_four(), start, (C, 6), fa.const(fa.code(TN), fa.code(LH)), 12)["rows"]
        assert all(x["dmg"][0] <= x["bud"][0] <= 1 + x["need"][0][0] for x in r)
        s = fa.stats(r)
        assert s["crashes"][0] == 0
        crashes_b.append(s["crashes"][1])
    assert crashes_b == sorted(crashes_b, reverse=True) and crashes_b[0] >= 2 and crashes_b[-1] == 0
    r = fa.play(fa.cfg_four(), start, (22, 6), fa.const(fa.code(TN), fa.code(LH)), 12)["rows"]
    assert [x["dmg"][0] for x in r[:4]] == [11, 11, 3, 1] and all(x["contact"] == 0 for x in r)


def test_test3_freeze_thin_holds_as_long_as_last_hit_and_clear_still_breaks_it(starts):
    for tier in ("L1", "L10", "L10 A+", "L18 A++"):
        caps = fa.capacities(tier)
        for resp, hold in (("walk_up", 18), ("step_back", 6)):
            assert _run(starts["freeze"], caps, TN, fa.RESPONSE_CODE[resp], 40, resp)["longest_hold"] == hold
            assert _run(starts["freeze"], caps, LH, fa.RESPONSE_CODE[resp], 40, resp)["longest_hold"] == hold
        assert _run(starts["freeze"], caps, TN, fa.RESPONSE_CODE["clear"], 40, "clear")["first_break"] <= 4


def test_test4_thin_neutralizes_slow_push_and_clear_overwhelms_both(starts):
    for caps in ((4, 4), (6, 6), (22, 4), (4, 22)):
        s = _run(starts["neutral"], caps, SL, fa.code(TN), 60)
        assert s["crashes"] == [0, 0]
    s = _run(starts["neutral"], (6, 6), CL, fa.code(SL), 60)
    assert s["crashes"][0] >= 10 and s["crashes"][1] == 0
