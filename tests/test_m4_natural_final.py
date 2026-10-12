"""M4.2e-A.1: the final natural-wave preset `GameConfig.m4_natural()` (docs/natural_waves/m4_2e_natural_waves_final_review.md).

Correctness of the four changes against the M4.2e-A prototype (entry crash, tower damage, tower reach, attack-power
wave clear), the global invariants on the final preset, and pins of the review's findings. The prototype keeps its own
pins in tests/test_m4_natural_waves.py."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import N_CHANCE, NO_REACH, mirror_chance, swap_seats
from lanegame.champions_m4 import M4Side, M4State, check_invariants, context, initial_m4, resolve_turn_m4
from lanegame.config import SH, W_CLEAR, W_HOLD, W_THIN
from lanegame.env import LaneEnv, UniformAgent
from lanegame.env.agents_m4 import natural_suite
from lanegame.env.discounted import max_champion_wave_budget, per_turn_bound
from lanegame.env.observation_m4 import decode_m4, m4_obs_size, observe_m4
from lanegame.experiments.m4 import natural_waves as nw
from lanegame.experiments.m4 import natural_waves_final as nf
from lanegame.mechanics import initial_state, step_turn, total_hp
from lanegame.state import State

from test_m4_env import soak

NAT = GameConfig.m4_natural()
PROTO = GameConfig.m4_natural_prototype()
T = NAT.tower
HO, CL, TN = nw.HO, nw.CL, nw.TN


def _lane(ax, ah, bx, bh, ar=(), br=(), asg=0, bsg=0):
    return State(ax, ah, tuple(ar), asg, bx, bh, tuple(br), bsg, 0, 0)


# ---------------------------------------------------------------------- configuration
def test_final_preset_is_the_prototype_with_four_changes():
    NAT.validate()
    PROTO.validate()
    changed = {k for k, v in NAT.to_dict().items() if PROTO.to_dict()[k] != v}
    assert changed == {"natural_crash_rule", "tower_siege_damage", "natural_tower_reach", "wave_clear_scaling",
                       "wave_clear_scaled_slots"}
    assert (NAT.natural_crash_rule, NAT.tower_siege_damage, NAT.natural_tower_reach, NAT.wave_clear_scaling,
            NAT.wave_clear_scaled_slots) == ("entry", 3, 1, "attack_power", (SH,))
    # the tower alone out-kills the reinforcement stream: no siege can sustain itself on follow-up waves
    assert NAT.tower_siege_damage * NAT.spawn_interval > NAT.wave_hp
    # the prototype keeps every M4.2e-A value; nothing outside the natural presets changes
    assert (PROTO.natural_crash_rule, PROTO.tower_siege_damage, PROTO.natural_tower_reach, PROTO.wave_clear_scaling) \
        == ("contact", 1, 0, "none")
    for cfg in (GameConfig(), GameConfig.m4(), GameConfig.m4_continuous(), GameConfig.m4_emergent()):
        assert cfg.natural_crash_rule == "contact" and cfg.wave_clear_scaling == "none"


def test_validation_of_the_new_fields():
    for kw in (dict(natural_crash_rule="always"), dict(natural_tower_reach=2), dict(wave_clear_scaling="level"),
               dict(wave_clear_scaled_slots=(7,))):
        with pytest.raises(ValueError):
            GameConfig.m4_natural(**kw).validate()
    for kw in (dict(natural_crash_rule="entry"), dict(natural_tower_reach=1), dict(wave_clear_scaling="attack_power")):
        with pytest.raises(ValueError, match="natural"):
            GameConfig.m4(**kw).validate()
    with pytest.raises(AssertionError):
        step_turn(initial_state(GameConfig.m4()), 0, 0, 0, GameConfig.m4(), champion_budget=(4, 4))


# ---------------------------------------------------------------------- entry crash and the tower
def test_a_front_entering_the_enemy_tower_cell_crashes_even_next_to_a_defender():
    st = _lane(1, 8, -4, 8)                          # A walks into B's tower cell; B's fresh wave walks to its own -3
    proto = step_turn(st, 1, W_HOLD, W_HOLD, PROTO, record=True, champion_reach=(NO_REACH, NO_REACH))
    final = step_turn(st, 1, W_HOLD, W_HOLD, NAT, record=True, champion_reach=(NO_REACH, NO_REACH))
    assert proto.crash_hp == (0, 0) and proto.trace[0]["fight"][0] == T     # prototype: it fights inside the tower cell
    assert final.crash_hp[0] == 8 and all(tr["fight"] is None for tr in final.trace)
    assert all(tr["siege_fight"] for tr in final.trace[1:])                  # B's front engages the siege instead
    assert final.events["tower_hp"][0] == 3                                  # fired on only once the mass was there


def test_follow_up_waves_join_the_siege_and_the_tower_fires_on_the_mass_alone():
    st = _lane(1, 8, -3, 6, asg=10)                  # A's follow-up arrives while its crashed mass is under B's tower
    r = step_turn(st, 1, W_HOLD, W_HOLD, NAT, record=True, champion_reach=(NO_REACH, NO_REACH))
    assert r.crash_hp[0] == 8 and r.events["crashed"][0] == 1
    assert all(tr["fight"] is None for tr in r.trace)
    assert r.events["tower_hp"][0] == 2 * NAT.tower_siege_damage           # one tower, one target, every tick
    before, after, ev = total_hp(st, NAT), total_hp(r.state, NAT), r.events
    assert before[0] + ev["spawned_hp"][0] == after[0] + ev["killed_hp"][0] + ev["tower_hp"][0]


def test_a_champion_in_the_enemy_tower_cell_reaches_the_defenders_behind_it():
    st = _lane(-4, 0, -3, 8, asg=10)                 # B's front engages A's siege from its own -3
    for reach, cfg, hit in ((T, PROTO, 0), (T + 1, NAT, 4)):
        r = step_turn(st, 1, W_CLEAR, W_HOLD, cfg, champion_reach=(reach, NO_REACH))
        assert r.events["champion_hp"][1] == hit
    # the resolver: a Pressure champion stands in the enemy tower cell and gets reach T + 1; a farming one never does
    m4 = initial_m4(NAT).replace_side(0, pos=T - 1)
    t = resolve_turn_m4(st, m4, 1, nf.CLP, HO, NAT, u=None)
    assert t.end[0] == T and t.reach[0] == T + 1
    assert resolve_turn_m4(st, m4, 1, nf.CLP, HO, NAT.with_(natural_tower_reach=0), u=None).reach[0] == T


# ---------------------------------------------------------------------- wave clear and attack power
def test_clear_budget_scales_with_attack_power_and_thin_stays_controlled():
    ctx = context(NAT)
    g = NAT.gold_scale
    table = {(L, S): (ctx.wave_budget(W_CLEAR, L, S * g), ctx.wave_budget(W_THIN, L, S * g), ctx.wave_budget(W_HOLD, L, S * g))
             for L, S in ((1, 0), (9, 16), (13, 32), (18, 64))}
    assert table == {(1, 0): (4, 2, 0), (9, 16): (9, 2, 0), (13, 32): (12, 2, 0), (18, 64): (17, 2, 0)}
    prev = 0
    for L in range(1, NAT.level_cap + 1):                                  # monotone in level and investment
        for S in (0, 4, 16, 64):
            b = ctx.wave_budget(W_CLEAR, L, S * g)
            assert b >= 4
        b = ctx.wave_budget(W_CLEAR, L, 0)
        assert b >= prev
        prev = b
    assert max_champion_wave_budget(NAT) >= ctx.wave_budget(W_CLEAR, NAT.level_cap, 10 ** 6 * g)


def test_turn_budget_is_split_over_the_ticks_earlier_first():
    st = _lane(0, 24, -1, 24)                        # in contact at the lane, nobody dies this turn
    for budget, split in (((5, 0), [3, 2]), ((4, 0), [2, 2]), ((1, 0), [1, 0])):
        r = step_turn(st, 1, W_CLEAR, W_HOLD, NAT, record=True, champion_reach=(T, NO_REACH), champion_budget=budget)
        assert [tr["champion"][0] for tr in r.trace] == split
    # the resolver passes the start-of-turn budget: a stronger champion removes more HP in the same turn
    lane, m4, k = nw.end_state(nw.play(NAT, nw.const(HO), 9))
    strong = m4.replace_side(0, xp=300)
    weak_hit = resolve_turn_m4(lane, m4, k, CL, HO, NAT, u=None).lane.events["champion_hp"][1]
    strong_hit = resolve_turn_m4(lane, strong, k, CL, HO, NAT, u=None).lane.events["champion_hp"][1]
    assert weak_hit == 4 and strong_hit > weak_hit


# ---------------------------------------------------------------------- invariants on the final preset
def _random_lane_turns(n, seed):
    rng = np.random.default_rng(seed)
    st = initial_state(NAT)
    for k in range(n):
        acts = [int(rng.integers(3)) for _ in (0, 1)]
        reach = tuple(int(rng.choice([NO_REACH, -2, -1, 0, 1, 2, 3])) for _ in (0, 1))
        budget = tuple(int(rng.integers(0, 24)) for _ in (0, 1))
        yield st, k, acts, reach, budget
        st = step_turn(st, k, acts[0], acts[1], NAT, champion_reach=reach, champion_budget=budget).state


def test_hp_conservation_kill_accounting_and_mirror_symmetry_on_random_lane_turns():
    for st, k, (a, b), reach, budget in _random_lane_turns(1500, 7):
        r = step_turn(st, k, a, b, NAT, champion_reach=reach, champion_budget=budget)
        before, after, ev = total_hp(st, NAT), total_hp(r.state, NAT), r.events
        for s in (0, 1):
            assert before[s] + ev["spawned_hp"][s] == after[s] + ev["killed_hp"][s] + ev["tower_hp"][s]
            assert ev["overflow_hp"][s] == 0
            assert sum(r.kills[1 - s].values()) == ev["killed_hp"][s] + ev["tower_hp"][s]
            assert r.state.side(s)[0] < T or r.state.side(s)[1] == 0          # no front ever stays in the tower cell
        m = step_turn(st.mirror(), k, b, a, NAT, champion_reach=reach[::-1], champion_budget=budget[::-1])
        assert m.state == r.state.mirror() and m.crash_hp == r.crash_hp[::-1] and m.kills == r.kills[::-1]
        assert m.tower_hits == r.tower_hits[::-1] and m.events["champion_hp"] == r.events["champion_hp"][::-1]


def test_resolver_mirror_equivariance_invariants_observation_and_reward_bound():
    asym = NAT.with_(skill_a=0.8, skill_b=0.3)
    env = LaneEnv(asym, max_turns=400)
    rng, nrng, ag = np.random.default_rng(1), np.random.default_rng(6), UniformAgent()
    obs = env.reset(seed=5)
    bound = per_turn_bound(asym)
    for _ in range(400):
        lane, m4, k = env.state, env.champ, env.k
        # a stronger A from turn 0 exercises the scaled budget on both seats of the mirror
        if k == 0:
            m4 = M4State(m4.a._replace(xp=200), m4.b)
            env._champ = m4
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
            assert dl == (lane if s == 0 else lane.mirror()) and dm == (m4 if s == 0 else m4.mirror())
        r = env.step(a, b)
        obs = r.obs
        if r.done:
            break


def test_env_soaks_conserve_minions_and_stay_zero_sum():
    soak(NAT, 2000, 11)
    su = natural_suite()
    for j, (x, y) in enumerate((("slow_pusher", "natural_freezer"), ("crasher", "hold_farmer"),
                                ("crash_recaller", "clearer"), ("trading_pusher", "thinner"))):
        soak(NAT, 300, 50 + j, (su[x], su[y]))


# ---------------------------------------------------------------------- pins of the review's findings
def test_prototype_siege_is_cleared_by_the_tower_while_defenders_fight_follow_ups():
    tot = nf.stage_siege()["totals"]
    assert tot["tower"] > 5 * tot["defenders"]
    assert tot["ticks_defender_vs_followup"] > 5 * tot["ticks_defender_vs_mass"]


def test_prototype_bounce_timing_does_not_follow_siege_length():
    sweep = nf.stage_siege()["tower_sweep"]
    clears = [r["thin_k1_hold"]["clear_turns"] for r in sweep.values()]
    bounces = [r["thin_k1_hold"]["bounce_crash"]["after_turns"] for r in sweep.values()]
    assert max(clears) >= 3 * min(clears) and max(bounces) - min(bounces) <= 2


def test_entry_rule_needs_a_tower_that_outkills_the_reinforcement_stream():
    s = nf.stage_sustain()
    assert s["tower1"]["thin_k1"][0]["turns"] > 40                     # 1 HP per tick: the siege feeds itself
    assert all(x["turns"] <= 10 for x in s["tower3"]["thin_k1"] + s["tower3"]["clear"])


def test_stronger_clear_crashes_sooner_and_smaller_from_a_neutral_lane():
    cfg = NAT
    weak, strong = nf.first_push(cfg, 1.0, 9), nf.first_push(cfg, 4.0, 9)
    assert weak["turns_to_crash"] > strong["turns_to_crash"]
    assert weak["siege"]["first_hp"] > strong["siege"]["first_hp"] >= 2 * cfg.wave_hp
    # from s0 a strong Clear crashes almost every fresh wave (a perma-shove), a level-1 Clear crashes merged waves
    fast = nf.line(nf.play(cfg, nw.const(CL), 20, mult=(4.0, 1.0)), cfg)["sieges_A"]
    assert [s["k"] for s in fast[:4]] == [2, 4, 6, 8] and all(s["first_hp"] >= 7 for s in fast[:4])


def test_stacked_crashes_still_bounce_with_shorter_sieges():
    for name in ("thin_k1_hold", "thin_k1_4_hold"):
        p = nw.crash_report(nw.play(PROTO, nw.until_crash(nw.BOUNCE_LINES[name]), 50))
        f = nw.crash_report(nw.play(NAT, nw.until_crash(nw.BOUNCE_LINES[name]), 50))
        assert f["clear_turns"] < p["clear_turns"] and f["tower_hits_during_siege"] < p["tower_hits_during_siege"]
        assert f["bounce_crash"] is not None and f["bounce_crash"]["hp"] >= 2 * NAT.wave_hp


def test_outside_tower_freeze_is_reachable_maintained_and_breakable():
    fz = nw.bounded_freezer(8)
    setup = nw.play(NAT, lambda k, l, m: (fz(l), TN if k == 1 else HO), 36)
    lane, m4, k0 = nw.end_state(setup)
    assert (lane.ax, -lane.bx) == (-T, -T + 1) and lane.bh > lane.ah         # B's bigger wave held OUTSIDE A's tower
    held = nw.summarize(nw.play(NAT, lambda k, l, m: (fz(l), HO), 20, lane, m4, k0), NAT)
    assert nf.held_turns(held["contact"]) == 20 and held["crashes"] == [[], []]
    assert held["farm"][0] > 2 * held["farm"][1]
    free = nw.summarize(nw.play(NAT, nw.const(HO), 20, lane, m4, k0), NAT)
    assert nf.held_turns(free["contact"]) < 20                             # without the freezer's Thin it collapses
    clear = nw.summarize(nw.play(NAT, lambda k, l, m: (fz(l), CL), 6, lane, m4, k0), NAT)
    assert nf.held_turns(clear["contact"]) <= 2 and clear["crashes"][1]     # the opponent's Clear crashes it


def test_crash_then_pressure_beats_crash_then_walk_away():
    p = nf.stage_power()["pressure_siege"]["reach1_ap1.0"]
    assert p["hold_pressure"]["A_minus_B"] > 0 > p["hold"]["A_minus_B"] and p["hold_pressure"]["plates"][0] >= 2


def test_no_wave_action_dominates_a_holding_opponent():
    a = nf.stage_actions()["final"]
    assert a["hold"]["A_minus_B"] == 0
    assert a["thin_k1"]["A_minus_B"] > 0 and a["clear"]["A_minus_B"] < 0 and a["thin"]["A_minus_B"] < 0


def test_recall_timing_matters_on_the_final_preset():
    rec = nw.stage_recall(NAT)
    own, enemy = rec["own_state"], rec["enemy_crash_k15"]
    assert own["neutral_k12"]["cost_A_minus_B"] > own["after_small_crash"]["cost_A_minus_B"]
    assert enemy["recall_from_k11"]["cost_A_minus_B"] < enemy["recall_from_k17"]["cost_A_minus_B"] / 2
