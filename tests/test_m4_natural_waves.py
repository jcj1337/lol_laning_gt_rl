"""M4.2e-A: natural wave mechanics, `GameConfig.m4_natural()` (docs/m4_2e_natural_waves_report.md).

Correctness (validation, legality, movement, conservation, mirror symmetry, accounting, observation) and pins of the
report's mechanical findings (neutral symmetry, Hold / Thin / Clear, stacking, crash, bounce, freeze, Recall, reach)."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import N_CHANCE, NO_REACH, champion_legal_actions, mirror_chance, swap_seats
from lanegame.champions_m4 import check_invariants, initial_m4, resolve_turn_m4
from lanegame.config import (B_HOLD, B_SAFE, B_TRADE, CHAMP_ABSENT, NATURAL_WAVE_HP, W_CLEAR, W_HOLD, W_THIN,
                             champ_act)
from lanegame.env import LaneEnv, UniformAgent
from lanegame.env.agents_m4 import natural_suite
from lanegame.env.discounted import per_turn_bound, tail_constant
from lanegame.env.observation_m4 import decode_m4, m4_obs_size, observe_m4
from lanegame.experiments.m4 import natural_waves as nw
from lanegame.mechanics import initial_state, step_turn, total_hp, wave_actions
from lanegame.state import State

from test_m4_env import soak

NAT = GameConfig.m4_natural()
T = NAT.tower
HO, CL, TN = nw.HO, nw.CL, nw.TN


def _lane(ax, ah, bx, bh, ar=(), br=(), asg=0, bsg=0):
    return State(ax, ah, tuple(ar), asg, bx, bh, tuple(br), bsg, 0, 0)


# ---------------------------------------------------------------------- configuration and legality
def test_preset_rescales_the_wave_and_keeps_the_historical_defaults():
    NAT.validate()
    assert NAT.wave_dynamics == "natural" and NAT.wave_hp == NATURAL_WAVE_HP == 8
    assert NAT.hp_cap == 24 and NAT.siege_cap == 96 and not NAT.freeze_maintenance
    # one fresh wave is worth what it was at 4 HP: 1 resource and 8 XP
    assert NAT.farm_per_hp * NAT.wave_hp == GameConfig.m4().farm_per_hp * GameConfig.m4().wave_hp == 1.0
    assert NAT.xp_per_hp * NAT.wave_hp == GameConfig.m4().xp_per_hp * GameConfig.m4().wave_hp == 8
    em = GameConfig.m4_emergent()
    for f in ("position_rule", "zoning_rule", "farm_spot_rule", "invest_curve", "item_model"):
        assert getattr(NAT, f) == getattr(em, f)
    for cfg in (GameConfig(), GameConfig.m4(), GameConfig.m4_continuous(), em):
        assert cfg.wave_dynamics == "scripted" and cfg.wave_hp == 4


def test_validation_of_the_natural_fields():
    with pytest.raises(ValueError, match="wave_dynamics"):
        GameConfig.m4(wave_dynamics="free").validate()
    with pytest.raises(ValueError, match="champion_wave_damage"):
        GameConfig.m4(champion_wave_damage=(0, 3, 1)).validate()             # natural only
    with pytest.raises(ValueError, match="champion_wave_damage"):
        GameConfig.m4_natural(champion_wave_damage=(0, -1, 1)).validate()
    with pytest.raises(ValueError, match="freeze_maintenance"):
        GameConfig.m4_natural(freeze_maintenance=True).validate()
    for kw in (dict(wave_dynamics="natural"), dict(champion_wave_damage=(0, 3, 1))):
        with pytest.raises(ValueError, match="M4 economy field"):
            GameConfig.champions(**kw).validate()


def test_every_wave_slot_is_always_legal_and_labelled_by_what_the_champion_does():
    lane, m4 = initial_state(NAT), initial_m4(NAT)
    assert wave_actions(lane, 0, NAT) == (W_HOLD, W_CLEAR, W_THIN)
    assert wave_actions(lane, 0, GameConfig.m4()) == (W_HOLD, W_CLEAR)       # scripted: Freeze needs a near enemy
    legal = champion_legal_actions(lane, m4, 0, NAT)
    for b in (B_HOLD, B_TRADE, B_SAFE):
        for w in (W_HOLD, W_CLEAR, W_THIN):
            assert champ_act(w, b) in legal
    assert (NAT.label(HO), NAT.label(CL), NAT.label(TN)) == ("HO-H", "CL-H", "TN-H")
    assert NAT.label(champ_act(W_CLEAR, B_TRADE, 1)) == "CL-T+P" and NAT.label(CHAMP_ABSENT) == "ABSENT"
    assert GameConfig.m4().label(CL) == "SH-H"


# ---------------------------------------------------------------------- movement and combat rules
def test_minions_follow_the_same_movement_whatever_the_actions():
    st = _lane(-4, 8, -4, 8)
    ends = {(a, b): step_turn(st, 0, a, b, NAT, champion_reach=(NO_REACH, NO_REACH)).state for a in range(3)
            for b in range(3)}
    assert len(set(ends.values())) == 1 and ends[(0, 0)].ax == -2
    # an unstacked front past the lane walks on to the enemy tower (no Slow Push parking, no Freeze hold)
    assert step_turn(_lane(0, 8, -4, 8), 1, W_HOLD, W_THIN, NAT, champion_reach=(NO_REACH, NO_REACH)).state.ax == 2
    assert step_turn(_lane(0, 4, -4, 4), 1, 0, 2, GameConfig.m4(), champion_reach=(NO_REACH, NO_REACH)).state.ax == 0


def test_champion_damage_by_slot_needs_reach():
    def fight(slot, reach):
        r = step_turn(_lane(0, 8, -1, 8), 2, slot, W_HOLD, NAT.with_(spawn_interval=200), record=True,
                      champion_reach=(reach, NO_REACH))
        return [tr["fight"][5] for tr in r.trace]
    assert fight(W_HOLD, T) == [1, 1] and fight(W_THIN, T) == [2, 2] and fight(W_CLEAR, T) == [3, 3]
    for slot in (W_THIN, W_CLEAR):
        assert fight(slot, NO_REACH) == [1, 1]          # away, channeling or in a connected trade
        assert fight(slot, 0) == [1, 1]                 # reach short of the enemy front (own cell 1)
    assert {n: nw.kill_ticks(NAT, w) for n, w in (("hold", W_HOLD), ("thin", W_THIN), ("clear", W_CLEAR))} == \
        {"hold": 8, "thin": 4, "clear": 3}


def test_a_champion_alone_can_clear_a_walking_wave_and_a_crashed_mass():
    # B's wave walks through A's own cells 0 and -1 with no A minion on the lane: reach 0 covers both ticks, reach -1
    # only the second
    for reach, hit in ((0, 4), (-1, 2)):
        r = step_turn(_lane(-4, 0, -1, 8), 0, W_CLEAR, W_HOLD, NAT, champion_reach=(reach, NO_REACH))
        assert r.events["champion_hp"][1] == hit and r.state.bh == 8 - hit
    sg = step_turn(_lane(-4, 0, -4, 0, bsg=10), 0, W_THIN, W_HOLD, NAT, champion_reach=(-T, NO_REACH))
    assert sg.state.bsg == 10 - 2 - 2 and sg.kills[0][-T] == 4              # 2 Thin + 2 tower fire under A's tower


def test_defender_engages_the_crashed_mass_which_then_spares_the_tower():
    st = _lane(-3, 8, -4, 0, bsg=16)                                        # A's front next to its tower, B crashed
    r = step_turn(st, 0, W_HOLD, W_HOLD, NAT, record=True, champion_reach=(NO_REACH, NO_REACH))
    assert all(tr["siege_fight"] for tr in r.trace) and r.state.ax == -3   # stays engaged; no idle wait rule
    assert r.tower_hits[1] == 0                                             # engaged mass does not hit the tower
    free = step_turn(_lane(-4, 0, -4, 0, bsg=16), 0, W_HOLD, W_HOLD, NAT, champion_reach=(NO_REACH, NO_REACH))
    assert free.tower_hits[1] == 2 + 1                                      # unengaged: floor(16/8) + floor(15/8)


def _random_lane_turns(n, seed):
    rng = np.random.default_rng(seed)
    st = initial_state(NAT)
    for k in range(n):
        acts = [int(rng.integers(3)) for _ in (0, 1)]
        reach = tuple(int(rng.choice([NO_REACH, -2, -1, 0, 1, 2])) for _ in (0, 1))
        yield st, k, acts, reach
        st = step_turn(st, k, acts[0], acts[1], NAT, champion_reach=reach).state


def test_hp_conservation_kill_accounting_and_mirror_symmetry_on_random_lane_turns():
    for st, k, (a, b), reach in _random_lane_turns(1500, 2):
        r = step_turn(st, k, a, b, NAT, champion_reach=reach)
        before, after, ev = total_hp(st, NAT), total_hp(r.state, NAT), r.events
        for s in (0, 1):
            assert before[s] + ev["spawned_hp"][s] == after[s] + ev["killed_hp"][s] + ev["tower_hp"][s]
            assert ev["overflow_hp"][s] == 0
            assert sum(r.kills[1 - s].values()) == ev["killed_hp"][s] + ev["tower_hp"][s]
        m = step_turn(st.mirror(), k, b, a, NAT, champion_reach=reach[::-1])
        assert m.state == r.state.mirror() and m.crash_hp == r.crash_hp[::-1] and m.kills == r.kills[::-1]
        assert m.tower_hits == r.tower_hits[::-1] and m.events["champion_hp"] == r.events["champion_hp"][::-1]


def test_resolver_mirror_equivariance_invariants_and_observation():
    asym = NAT.with_(skill_a=0.8, skill_b=0.3)
    env = LaneEnv(asym, max_turns=400)
    rng, nrng, ag = np.random.default_rng(0), np.random.default_rng(5), UniformAgent()
    obs = env.reset(seed=4)
    bound = per_turn_bound(asym)
    for _ in range(400):
        lane, m4, k = env.state, env.champ, env.k
        a, b = ag.act(obs[0], rng), ag.act(obs[1], rng)
        u = tuple(float(x) for x in nrng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k, a, b, asym, u=u)
        m = resolve_turn_m4(lane.mirror(), m4.mirror(), k, b, a, swap_seats(asym), u=mirror_chance(u))
        assert m.champ == t.champ.mirror() and m.lane.state == t.lane.state.mirror()
        assert m.reward_units == -t.reward_units and abs(t.reward) <= bound
        check_invariants(t.champ, asym)
        for s in (0, 1):
            v = observe_m4(lane, k, s, asym, m4)
            assert len(v) == m4_obs_size(asym) == 88 and v.min() >= 0 and v.max() <= 1
            _, dl, dm, _ = decode_m4(v, asym)
            assert dl == (lane if s == 0 else lane.mirror())
            assert dm == (m4 if s == 0 else m4.mirror())
        r = env.step(a, b)
        obs = r.obs
        if r.done:
            break


def test_env_soaks_conserve_minions_and_stay_zero_sum():
    soak(NAT, 2000, 3)
    su = natural_suite()
    for j, (x, y) in enumerate((("slow_pusher", "natural_freezer"), ("crasher", "hold_farmer"),
                                ("crash_recaller", "clearer"), ("trading_pusher", "thinner"))):
        soak(NAT, 300, 40 + j, (su[x], su[y]))


def test_evaluation_bounds_are_defined_for_the_natural_lane():
    assert 0 < per_turn_bound(NAT) < 10 and tail_constant(NAT) > tail_constant(GameConfig.m4())


# ---------------------------------------------------------------------- pins of the report's findings
def test_neutral_lane_is_symmetric_perpetual_contact_at_the_lane():
    s = nw.summarize(nw.play(NAT, nw.const(HO), 40), NAT)
    assert s["earned_A_minus_B"] == 0 and s["crashes"] == [[], []]
    assert all(c == 0 for c in s["contact"][1:]) and all(a == b for a, b in s["fronts"])
    assert max(a for a, _ in s["fronts"]) > 2 * NAT.wave_hp                 # reinforcements stack in contact


def test_hold_thin_clear_against_a_holding_opponent():
    lines = {n: nw.summarize(nw.play(NAT, nw.const(c), 40), NAT) for n, c in (("hold", HO), ("thin", TN),
                                                                            ("clear", CL))}
    assert lines["hold"]["earned_A_minus_B"] == 0
    assert lines["thin"]["crashes"][0] == [{"k": 9, "hp": 12}]              # Thin stacks and crashes
    assert lines["clear"]["crashes"][0] == []                              # Clear alone does not stack
    assert lines["clear"]["earned_A_minus_B"] < lines["thin"]["earned_A_minus_B"] < 0


def test_one_turn_of_thin_accumulates_into_a_stacked_crash_and_a_bounce():
    rows = nw.play(NAT, nw.window(TN, 1, 1), 40)
    s = nw.summarize(rows, NAT)
    first = s["crashes"][0][0]
    assert first == {"k": 18, "hp": 16}
    pre = [a for a, _ in s["fronts"][:first["k"]]]
    assert max(pre) > 2 * NAT.wave_hp                                       # stacked from reachable play
    assert s["crashes"][1] and s["crashes"][1][0]["k"] > first["k"]         # the defender bounces and crashes back


def test_clear_turns_a_built_stack_into_an_earlier_larger_crash():
    hold = nw.summarize(nw.play(NAT, lambda k, l, m: (TN if 1 <= k <= 3 else HO, HO), 24), NAT)
    clear = nw.summarize(nw.play(NAT, lambda k, l, m: (TN if 1 <= k <= 3 else (CL if k >= 8 else HO), HO), 24), NAT)
    assert hold["crashes"][0][0] == {"k": 15, "hp": 16} and clear["crashes"][0][0] == {"k": 11, "hp": 19}


def test_crashes_bounce_naturally():
    for name in ("thin_until_crash", "thin_k1_4_hold"):
        rep = nw.crash_report(nw.play(NAT, nw.until_crash(nw.BOUNCE_LINES[name]), 44))
        assert rep["at_clear"]["B_front"] >= 2 * NAT.wave_hp > rep["at_clear"]["A_front"]
        assert rep["defender_in_attacker_half_after_turns"] is not None and rep["bounce_crash"]["hp"] >= 16
    small = nw.crash_report(nw.play(NAT, nw.until_crash(nw.BOUNCE_LINES["thin_until_crash"]), 44))
    large = nw.crash_report(nw.play(NAT, nw.until_crash(nw.BOUNCE_LINES["thin_k1_4_hold"]), 44))
    assert (small["crash_hp"], small["clear_turns"]) == (12, 6) and (large["crash_hp"], large["clear_turns"]) == (25, 11)


def test_freeze_is_reachable_maintainable_and_breakable():
    fz = nw.bounded_freezer(8)
    setup = nw.play(NAT, lambda k, l, m: (fz(l), TN if k == 1 else HO), 30)
    lane, m4, k0 = nw.end_state(setup)
    assert nw.contact_cell(lane) == -2 and lane.bh > lane.ah                # B's bigger wave held at A's tower
    held = nw.summarize(nw.play(NAT, lambda k, l, m: (fz(l), HO), 16, lane, m4, k0), NAT)
    assert all(c == -2 for c in held["contact"][:10]) and held["crashes"] == [[], []]
    assert held["farm"][0] > 3 * held["farm"][1]
    # once B's wave stands in A's tower cell the tower does the thinning: A Holding keeps it for 21 turns, until its own
    # wave outgrows two waves and pushes out (a temporary freeze, not a permanent one)
    free = nw.summarize(nw.play(NAT, nw.const(HO), 30, lane, m4, k0), NAT)
    assert free["contact"][:21] == [-2] * 21 and free["contact"][21] is None and free["crashes"][0][0]["k"] == 57
    for b in (TN, CL):
        broken = nw.summarize(nw.play(NAT, lambda k, l, m, b=b: (fz(l), b), 4, lane, m4, k0), NAT)
        assert broken["crashes"][1] and broken["crashes"][1][0]["k"] == k0 + 1


def test_recall_timing_matters():
    rec = nw.stage_recall(NAT)
    own, enemy = rec["own_state"], rec["enemy_crash_k15"]
    assert own["neutral_k12"]["missed_gold"] > own["after_small_crash"]["missed_gold"] > 0
    assert own["neutral_k12"]["lane_at_return"] == own["neutral_k12"]["lane_at_return_if_stayed"]   # Hold moves nothing
    assert enemy["recall_from_k11"]["cost_A_minus_B"] < enemy["recall_from_k17"]["cost_A_minus_B"] / 2


def test_trade_safe_and_recall_prevent_wave_damage():
    r = nw.stage_reach(NAT)
    assert r["thin_hold"]["champion_hp_on_B"] == 2 and r["clear_hold"]["champion_hp_on_B"] == 4
    assert r["hold_hold"]["champion_hp_on_B"] == 0
    for name in ("thin_trade_connected", "clear_trade_connected"):
        assert r[name]["connected"][0] and r[name]["reach_A"] == NO_REACH and r[name]["champion_hp_on_B"] == 0
    assert r["thin_safe"]["champion_hp_on_B"] == 0
