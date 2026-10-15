"""Cross-mechanic integration tests (plan §4, tests 1-12) on constructed scenarios and scripted play.

Finding documented here and in docs/m4/m4_2/m4_implementation_report.md (S3): design §6.4's crash table was measured on the
lane alone with full champion reach. On the implemented resolver the Slow Push rows reproduce exactly, but a
Shove + Hold breaker does not keep its reach on every turn: when the probe pass (design §5 step 1, M2's rule) predicts
that its front breaks the hold and crashes, its farm spot becomes its next packet, the champion walks back, its reach
no longer covers the front, and in the authoritative step the hold stands. The hold rule itself is exact (a reaching
Shove with blob + 4 >= 10 is never held, test 2b); the breaker simply reaches on fewer turns, so Shove-vs-Freeze
crashes come later than the lane-only table (test 1) and the 3-5 HP break window is 6-9 turns instead of 3-6
(test 3).

Two further plan expectations are contradicted by measurements and are reported, not hidden:
  * test 1: the plan expects a creeping 5-7 HP front to crash strictly later under Freeze "in every start phase
    where §6.4 reports a difference"; the design's own table has 3 cells where it crashes EARLIER (6 HP phase 1:
    11 vs 15; 7 HP phases 2, 3: 10 vs 14, 9 vs 13: the held front keeps growing). The resolver reproduces those
    values exactly; "strictly later" is asserted where the table says so.
  * test 10: a continuously shoving `presser` never has Freeze legal after its own crash (its shoved waves meet the
    bounce in mid-lane; 0 Freeze-legal turns in 50 games of 200 turns against hold_farmer). The crash -> bounce ->
    Freeze-legal chain of design §15 scenario 7 appears among the scripted pairings only with conditional Shove
    (cshove), with a median delay of about 20 turns after the siege clears.
"""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame import progression as pg
from lanegame.champions import N_CHANCE, U_TRADE_HIT, mirror_chance, swap_seats
from lanegame.champions_m4 import M4Side, M4State, initial_m4, resolve_turn_m4
from lanegame.config import B_HOLD, B_SAFE, B_TRADE, CHAMP_ABSENT, F, NONE, SH, SP, champ_act
from lanegame.env import LaneEnv, play_episode
from lanegame.env.agents_m4 import M4ScriptAgent, m4_suite
from lanegame.mechanics import _blob_hp, initial_state, step_turn, wave_actions
from lanegame.scenarios_m4 import DESIGN_HOLD_TABLE, hold_run, parked_state
from lanegame.state import State

M4 = GameConfig.m4()
T = M4.tower
HO = champ_act(SP, B_HOLD)
# measured on the implemented resolver: first crash under a maintained Freeze against a Shove + Hold breaker,
# phases 0..3 (the Slow Push defender's crash is immediate, 0, in every phase, as in the design table)
IMPLEMENTED_SHOVE_VS_FREEZE = {2: [None] * 4, 3: [9, 8, 9, 8], 4: [9, 8, 9, 8], 5: [7, 6, 7, 6], 6: [0, 6, 0, 6],
                               7: [0, 4, 0, 4], 8: [0, 2, 0, 2], 9: [0, 2, 0, 2]}


def lane_only_hold(e, b_wave, phase, a_freeze, turns=24):
    """The design's lane-only measurement: both champions' reach covers the whole lane (T, T)."""
    s = State(0, 0, (), 0, 1, e, (), 0, 0, 0, 0, 0)
    for i in range(turns):
        a = F if (a_freeze and F in wave_actions(s, 0, M4)) else SP
        res = step_turn(s, phase + i, a, b_wave, M4, champion_reach=(T, T))
        if res.crash_hp[1] > 0:
            return i
        s = res.state
    return None


# ---------------------------------------------------------------------- 1. Freeze != Slow Push
def test_1_freeze_differs_from_slow_push_in_the_parked_configuration():
    for (e, w), (fz, sp) in DESIGN_HOLD_TABLE.items():
        for ph in range(4):
            f = hold_run(e, w, ph, F, M4)["first_crash"]
            s = hold_run(e, w, ph, SP, M4)["first_crash"]
            # the lane-only rule reproduces the design table exactly
            assert lane_only_hold(e, w, ph, True) == fz[ph] and lane_only_hold(e, w, ph, False) == sp[ph]
            assert s == sp[ph]                                   # Slow Push defender: exact on the resolver
            if w == SP:
                assert f == fz[ph]                               # creeping fronts: exact on the resolver
            else:
                assert f == IMPLEMENTED_SHOVE_VS_FREEZE[e][ph]
            design_later = (fz[ph] is None and sp[ph] is not None) or (
                fz[ph] is not None and sp[ph] is not None and fz[ph] > sp[ph])
            if design_later:                                     # where the design reports a later crash ...
                later = (f is None and s is not None) or (f is not None and s is not None and f > s)
                assert later, (e, w, ph, f, s)                   # ... so does the resolver
    # the cells where the design's own table shows an EARLIER crash under Freeze (the growing held front)
    earlier = [(e, ph) for (e, w), (fz, sp) in DESIGN_HOLD_TABLE.items() if w == SP for ph in range(4)
               if fz[ph] is not None and sp[ph] is not None and fz[ph] < sp[ph]]
    assert earlier == [(6, 1), (7, 2), (7, 3)]
    for e in (3, 4):                                             # the lane-stable case: parked 24 turns either way
        for ph in range(4):
            assert hold_run(e, SP, ph, F, M4)["first_crash"] is None
            assert hold_run(e, SP, ph, SP, M4)["first_crash"] is None


# ---------------------------------------------------------------------- 2. Freeze cannot hold everything
def test_2a_ten_hp_fronts_and_creeping_eight_nine_hp_fronts():
    for e in (10, 11, 12):
        for ph in range(4):
            assert hold_run(e, SP, ph, F, M4)["first_crash"] == hold_run(e, SP, ph, SP, M4)["first_crash"]
    for e in (8, 9):
        for ph in range(4):
            assert hold_run(e, SP, ph, F, M4)["first_crash"] <= 8


def test_2b_a_reaching_shove_with_enough_hp_is_never_held():
    """Per turn: B shoving with its champion's reach covering its front and blob + 4 >= 10 -> no hold event."""
    checked = 0
    for e in range(2, 13):
        for ph in range(4):
            lane, m4 = parked_state(e, M4)
            for i in range(24):
                wa = F if F in wave_actions(lane, 0, M4) else SP
                t = resolve_turn_m4(lane, m4, ph + i, champ_act(wa, B_HOLD), champ_act(SH, B_HOLD), M4, u=None)
                if t.effective_waves[0] == F and lane.bh > 0 and t.reach[1] >= lane.bx:
                    packets = [[lane.bx, lane.bh]] + [[x, 4] for x in lane.br]
                    if _blob_hp(packets, M4) + 4 >= 10:
                        assert t.lane.events["freeze_held"][1] == 0
                        checked += 1
                lane, m4 = t.lane.state, t.champ
    assert checked > 20


# ---------------------------------------------------------------------- 3. Shove breaks a Freeze
def test_3_shove_breaks_a_freeze_without_automatic_punishment():
    u1 = (1.0,) * N_CHANCE
    for e in (3, 4, 5):
        for ph in range(4):
            r = hold_run(e, SH, ph, F, M4)
            assert r["first_crash"] is not None and r["first_crash"] <= 9          # measured 6-9 (design: 3-6)
            assert lane_only_hold(e, SH, ph, True) <= 6                            # the always-reaching breaker
            # no automatic penalty: until the break the breaker loses no HP (no trade, no gank), and its farm is the
            # ordinary collection rule at its cell, which never reads the defender's wave action
            from lanegame.champions_m4 import collection_units
            lane, m4 = parked_state(e, M4)
            for i in range(r["first_crash"] + 1):
                wa = F if F in wave_actions(lane, 0, M4) else SP
                tf = resolve_turn_m4(lane, m4, ph + i, champ_act(wa, B_HOLD), champ_act(SH, B_HOLD), M4, u=u1)
                assert tf.damage_taken[1] == 0
                expected = sum(hp * collection_units(x, tf.modes[1], tf.end[1], tf.modes[0], tf.end[0], M4)
                               for x, hp in tf.lane.kills[1].items())
                assert tf.farm_units[1] == expected
                lane, m4 = tf.lane.state, tf.champ
            # out of reach (Shove + Safe: the champion stays at -1, reach -1 < front +1) it breaks later or never
            safe = hold_run_breaker_safe(e, ph)
            assert safe is None or safe >= r["first_crash"]


def hold_run_breaker_safe(e, ph, turns=24):
    lane, m4 = parked_state(e, M4)
    m4 = m4.replace_side(1, pos=-1)
    for i in range(turns):
        wa = F if F in wave_actions(lane, 0, M4) else SP
        t = resolve_turn_m4(lane, m4, ph + i, champ_act(wa, B_HOLD), champ_act(SH, B_SAFE), M4, u=None)
        assert t.reach[1] <= -1
        if t.lane.crash_hp[1] > 0:
            return i
        lane, m4 = t.lane.state, t.champ
    return None


# ---------------------------------------------------------------------- 4. gold / XP separation
def run_parked(b_behavior, e=4, turns=8, b_pos=1):
    lane, m4 = parked_state(e, M4)
    m4 = m4.replace_side(1, pos=b_pos)
    gold = xp = 0
    for i in range(turns):
        wa = F if F in wave_actions(lane, 0, M4) else SP
        t = resolve_turn_m4(lane, m4, i, champ_act(wa, B_HOLD), champ_act(SP, b_behavior), M4, u=None)
        gold += t.farm_units[1]
        xp += t.xp_gain[1]
        ends = t.end[1]
        lane, m4 = t.lane.state, t.champ
    return gold, xp, ends


def test_4_gold_and_xp_separate_by_cell_under_a_held_freeze():
    g_hold, x_hold, end_hold = run_parked(B_HOLD)
    g_safe, x_safe, end_safe = run_parked(B_SAFE, b_pos=-1)
    assert end_hold == 1 and g_hold > 0 and x_hold > 0                  # forward: gold + XP
    assert end_safe == -1 and g_safe == 0 and x_safe > 0                # back: XP only
    # own tower cell: neither (constructed kill geometry, tests/test_m4_combat.py::test_xp_range_three_versus_gold)


# ---------------------------------------------------------------------- 5. XP denial needs commitment
def test_5_xp_denial_requires_the_freezer_at_the_lane_or_beyond():
    suite = m4_suite()
    attacker = M4ScriptAgent("forward_freezer", SP, B_TRADE, freeze=True)       # freezes and trades forward
    env = LaneEnv(M4, max_turns=400)
    denied = 0
    for seed in range(6):
        log = play_episode(env, attacker, suite["xp_keeper"], seed, record=True)
        for t in log.turns:
            if t["away"][1] or t["away"][0]:
                continue
            if t["champ_end"][1] == -T and t["behaviors"][1] == B_SAFE:
                assert t["champ_end"][0] >= 0, t["champ_end"]         # pushed back only by a threat from >= 0
                denied += 1
            if t["champ_end"][0] <= -1 and t["behaviors"][1] == B_SAFE:
                assert t["champ_end"][1] >= -1
    assert denied > 0


# ---------------------------------------------------------------------- 6-9 single-mechanic checks in situ
def test_6_items_change_damage_only():
    lane = State(0, 4, (), 0, 0, 4, (), 0, 0, 0, 0, 0)
    u = list((1.0,) * N_CHANCE)
    u[U_TRADE_HIT[0]] = 0.0
    ref = None
    for L in (1, 5, 11, 18):
        for N in (0, 1, 2):
            st = M4State(M4Side(0, 600, 0, pg.DEFAULT.xp_cum[L], N), M4Side(0, 600))
            t = resolve_turn_m4(lane, st, 2, champ_act(SP, B_TRADE), HO, M4, u=tuple(u))
            ref = ref or (t.hit_p, t.double_p)
            assert (t.hit_p, t.double_p) == ref
            ap = pg.attack_power(L, N)
            assert t.trade_damage[0] == (100 * ap + 5000) // 10_000


def test_7_level_up_in_combat():
    lane = State(0, 12, (), 0, 0, 12, (), 0, 0, 0, 0, 0)
    u = list((1.0,) * N_CHANCE)
    u[U_TRADE_HIT[1]] = 0.0
    survivor = M4State(M4Side(0, 50, 0, 69), M4Side(0, 766, 0, 75))      # A L5 at 50, crosses 70 this turn
    t = resolve_turn_m4(lane, survivor, 2, HO, HO, M4, u=None)
    assert t.levels_after[0] == 6 and t.champ.a.hp == 50 + pg.hp_max(6) - pg.hp_max(5)
    t = resolve_turn_m4(lane, survivor, 2, HO, champ_act(SP, B_TRADE), M4, u=tuple(u))
    assert t.trade_damage[1] >= 128 and t.died[0] and t.champ.a.hp == 0 and t.levels_after[0] == 6


def test_8_9_purchase_timing_recall_and_death():
    import test_m4_economy as econ
    econ.test_recall_purchase_timing_with_plate_gold_earned_while_away()
    econ.test_death_purchase_timing()


# ---------------------------------------------------------------------- 10. crash -> bounce -> plates
def crash_bounce_stats(x, y, seeds, turns=200):
    env = LaneEnv(M4, max_turns=turns, cache_size=20_000)
    bounce_delays, freeze_after_bounce, plate_games, sieges = [], 0, 0, 0
    for seed in seeds:
        log = play_episode(env, x, y, seed, record=True)
        tau_prev, plates_seen = 0, 0
        siege_end = None
        for t in log.turns:
            tau = t["tower_dmg"][0]
            assert tau >= tau_prev
            plates_seen += t["plates_broken"][0]
            assert t["plate_units"][0] == M4.gold_scale * t["plates_broken"][0]
            tau_prev = tau
            before, after = t["state_before"], t["state_after"]
            if before[3] > 0 and after[3] == 0:                     # A's siege cleared this turn
                siege_end = t["k"]
            sieges += before[3] > 0
            if siege_end is not None and 2 in [a % 3 for a in t["legal"][0] if a != CHAMP_ABSENT]:
                bounce_delays.append(t["k"] - siege_end)
                freeze_after_bounce += 1
                siege_end = None
        assert plates_seen == pg.plates_broken(tau_prev)
        plate_games += plates_seen > 0
    return {"plate_games": plate_games, "freeze_after_bounce": freeze_after_bounce, "delays": bounce_delays,
            "siege_turns": sieges}


def test_10_crash_bounce_and_plates():
    suite = m4_suite()
    st = crash_bounce_stats(suite["presser"], suite["hold_farmer"], range(200))
    # tower damage is monotone and plate gold is paid once per threshold (asserted per turn above)
    assert st["plate_games"] > 50 and st["siege_turns"] > 0
    # measured finding: the continuously shoving presser never has Freeze legal after its own crash
    assert st["freeze_after_bounce"] == 0
    # the chain crash -> bounce -> Freeze legal for the crasher, with a conditional shover
    from lanegame.env.agents_m4 import gate_defenders
    cs = gate_defenders()["cshove"]
    st2 = crash_bounce_stats(cs, cs, range(40))
    assert st2["freeze_after_bounce"] > 0 and np.median(st2["delays"]) <= 40


# ---------------------------------------------------------------------- 11. continuing play
def test_11_continuing_play_caps_and_stationarity():
    suite = m4_suite()
    env = LaneEnv(M4, max_turns=2600, cache_size=50_000)
    blocks, caps_at = [], []
    for seed in range(12):
        log = play_episode(env, suite["presser"], suite["hold_farmer"], seed, record=True)
        assert log.length == 2600                                    # truncated, never terminated
        fin = log.m4_final
        assert fin["levels"] == [18, 18] and fin["items"] == [2, 2] and fin["gold"] == [0, 0]
        last_plate = max([t["k"] for t in log.turns if any(t["plates_broken"])] + [0])
        uncapped = [t["k"] for t in log.turns if tuple(t["levels"]) != (18, 18) or tuple(t["items"]) != (2, 2)]
        last_cap = max(uncapped) + 1 if uncapped else 0
        caps_at.append(last_cap)
        assert last_plate < 1400 and last_cap < 1400
        r = np.asarray(log.rewards[1400:])
        blocks.append([r[i * 200:(i + 1) * 200].mean() for i in range(6)])
    assert np.median(caps_at) < 400                                  # both capped by about turn 150-400
    B = np.asarray(blocks)                                        # games x blocks
    overall = B.mean()
    se = B.std(ddof=1, axis=0) / np.sqrt(B.shape[0])
    assert np.all(np.abs(B.mean(axis=0) - overall) <= 4 * se + 1e-9)


# ---------------------------------------------------------------------- 12. zero-sum and symmetry
def test_12_zero_sum_and_seat_swap_symmetry():
    asym = GameConfig.m4(skill_a=0.7, skill_b=0.4)
    suite = m4_suite()
    x, y = suite["freezer"], suite["aggressive_trader"]
    from lanegame.env.core import Observation
    from lanegame.env.observation import legal_set, observe
    lane, m4, k = initial_state(asym), initial_m4(asym), 0
    lane_m, m4_m = lane.mirror(), m4.mirror()
    cfg_m = swap_seats(asym)
    rng = np.random.default_rng(5)

    def ob(l, c, kk, s, cfg):
        lg = legal_set(l, s, cfg, c)
        mask = np.zeros(46, dtype=bool)
        mask[list(lg)] = True
        return Observation(kk, s, l, lg, observe(l, kk, s, cfg, c), mask, c, cfg)
    for _ in range(600):
        a, b = x.act(ob(lane, m4, k, 0, asym)), y.act(ob(lane, m4, k, 1, asym))
        bm, am = y.act(ob(lane_m, m4_m, k, 0, cfg_m)), x.act(ob(lane_m, m4_m, k, 1, cfg_m))
        assert (am, bm) == (a, b)
        u = tuple(float(v) for v in rng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k, a, b, asym, u=u)
        tm = resolve_turn_m4(lane_m, m4_m, k, bm, am, cfg_m, u=mirror_chance(u))
        assert tm.reward_units == -t.reward_units and t.reward + (-t.reward) == 0
        assert tm.champ == t.champ.mirror() and tm.lane.state == t.lane.state.mirror()
        lane, m4, lane_m, m4_m, k = t.lane.state, t.champ, tm.lane.state, tm.champ, k + 1
