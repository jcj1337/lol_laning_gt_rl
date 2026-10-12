"""M4.2e-C: the wave action redesign (docs/natural_waves/m4_2e_wave_actions_report.md).

The opt-in engine fields (wave_clear_scaling "capacity" with wave_hold_fraction / wave_thin_fraction, damage_law
"lanchester_half") and their default preservation; the candidate preset `GameConfig.m4_lasthit()`; the global
correctness guarantees on the candidate; and pins of the report's findings."""

import hashlib

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import N_CHANCE, NO_REACH, mirror_chance, swap_seats, champion_legal_actions
from lanegame.champions_m4 import M4State, check_invariants, context, initial_m4, resolve_turn_m4
from lanegame.config import SH, SP, F, W_CLEAR, W_HOLD, W_THIN
from lanegame.env import LaneEnv, UniformAgent
from lanegame.env.discounted import max_champion_wave_budget, per_turn_bound
from lanegame.env.observation_m4 import decode_m4, m4_obs_size, observe_m4
from lanegame.experiments.m4 import natural_waves as nw
from lanegame.experiments.m4 import natural_waves_final as nf
from lanegame.experiments.m4 import wave_actions as wa
from lanegame.mechanics import initial_state, minion_damage, step_turn, total_hp
from lanegame.state import State

from test_m4_env import soak

CAND = GameConfig.m4_lasthit()
LANE = GameConfig.m4_lane()
T = CAND.tower
A, B = 0, 1
HO, CL, TN = nw.HO, nw.CL, nw.TN


def _lane(ax, ah, bx, bh, ar=(), br=(), asg=0, bsg=0):
    return State(ax, ah, tuple(ar), asg, bx, bh, tuple(br), bsg, 0, 0)


def _digest(cfg: GameConfig, turns: int = 300, seed: int = 7) -> str:
    rng = np.random.default_rng(seed)
    lane, m4 = initial_state(cfg), initial_m4(cfg)
    h = hashlib.sha256()
    for k in range(turns):
        la, lb = champion_legal_actions(lane, m4, A, cfg), champion_legal_actions(lane, m4, B, cfg)
        a, b = la[rng.integers(len(la))], lb[rng.integers(len(lb))]
        u = tuple(rng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, u=u)
        h.update(repr((t.lane.state, t.champ, t.farm_units, t.earned_units, t.damage_taken, t.trade_damage,
                       t.died, t.modes, t.end, t.reward_units, t.xp_gain, t.plates_broken)).encode())
        lane, m4 = t.lane.state, t.champ
    return h.hexdigest()


# ---------------------------------------------------------------------- configuration
def test_candidate_is_m4_lane_with_the_documented_fields():
    CAND.validate()
    changed = {k for k, v in CAND.to_dict().items() if LANE.to_dict()[k] != v}
    assert changed == set(GameConfig.M4_LASTHIT_FIELDS)
    assert (CAND.wave_clear_scaling, CAND.wave_hold_fraction, CAND.wave_thin_fraction, CAND.damage_law,
            CAND.thin_damage) == ("capacity", 0.125, 0.5, "lanchester_half", 0)
    # every other preset keeps the old model
    for cfg in (GameConfig(), GameConfig.m4(), GameConfig.m4_emergent(), GameConfig.m4_natural_prototype(),
                GameConfig.m4_natural(), LANE):
        assert cfg.wave_clear_scaling != "capacity" and cfg.damage_law == "lanchester"
        assert (cfg.wave_hold_fraction, cfg.wave_thin_fraction, cfg.thin_damage) == (0.0, 0.0, 1)


def test_validation_of_the_new_fields():
    for kw in (dict(wave_hold_fraction=0.6), dict(wave_thin_fraction=1.5), dict(wave_hold_fraction=-0.1),
               dict(wave_hold_fraction=0.5, wave_thin_fraction=0.25)):
        with pytest.raises(ValueError):
            GameConfig.m4_lasthit(**kw).validate()
    for kw in (dict(wave_hold_fraction=0.125, wave_thin_fraction=0.5), dict(wave_thin_fraction=0.5)):
        with pytest.raises(ValueError, match="capacity"):
            GameConfig.m4_natural(**kw).validate()                   # fractions need the capacity model
    with pytest.raises(ValueError, match="natural"):
        GameConfig.m4(damage_law="lanchester_half").validate()       # the half law is natural-only
    for kw in (dict(wave_clear_scaling="capacity"), dict(wave_hold_fraction=0.125), dict(wave_thin_fraction=0.5)):
        with pytest.raises(ValueError):
            GameConfig.m4(**kw).validate()                           # scripted waves reject the natural fields
    with pytest.raises(ValueError):
        GameConfig.champions(damage_law="lanchester_half").validate()


# ---------------------------------------------------------------------- budgets
def test_budgets_are_shares_of_one_capacity_and_grow_with_power():
    ctx = context(CAND)
    g = CAND.gold_scale
    table = {(L, S): tuple(ctx.wave_budget(w, L, S * g) for w in (W_HOLD, W_THIN, W_CLEAR))
             for L, S in ((1, 0), (6, 4), (9, 16), (13, 32), (18, 64), (18, 10 ** 4))}
    assert table == {(1, 0): (1, 2, 4), (6, 4): (1, 3, 6), (9, 16): (1, 5, 9), (13, 32): (2, 6, 12),
                     (18, 64): (2, 9, 17), (18, 10 ** 4): (3, 11, 22)}
    assert ctx.budget_from_capacity(SH, 4) == 4 and ctx.budget_from_capacity(SP, 4) == 1 == ctx.budget_from_capacity(F, 4) - 1
    prev = (0, 0, 0)
    for L in range(1, CAND.level_cap + 1):
        for S in (0, 4, 16, 64, 256):
            b = tuple(ctx.wave_budget(w, L, S * g) for w in (W_HOLD, W_THIN, W_CLEAR))
            assert 1 <= b[0] <= b[1] <= b[2] and b[2] == ctx.capacity(L, S * g)
        b = tuple(ctx.wave_budget(w, L, 0) for w in (W_HOLD, W_THIN, W_CLEAR))
        assert all(x >= y for x, y in zip(b, prev))                # non-decreasing in level
        prev = b
    # Clear is unchanged from m4_lane; the lane model is unchanged
    lane_ctx = context(LANE)
    for L, S in ((1, 0), (9, 16), (18, 64)):
        assert ctx.wave_budget(W_CLEAR, L, S * g) == lane_ctx.wave_budget(W_CLEAR, L, S * g)
        assert (lane_ctx.wave_budget(W_HOLD, L, S * g), lane_ctx.wave_budget(W_THIN, L, S * g)) == (0, 2)
    assert max_champion_wave_budget(CAND) == max_champion_wave_budget(LANE) >= ctx.wave_budget(W_CLEAR, 18, 10 ** 6 * g)
    assert wa.budget_table(CAND) == wa.budget_table(CAND, wa.MODELS["shares_1/8_1/2"])


def test_resolver_uses_the_capacity_budgets_including_hold():
    lane, m4, k = nw.end_state(nw.play(CAND, nw.const(HO), 9))
    strong = m4.replace_side(0, xp=300, items=32 * CAND.gold_scale)
    hits = {}
    for name, st in (("weak", m4), ("strong", strong)):
        hits[name] = [resolve_turn_m4(lane, st, k, code, HO, CAND, u=None).lane.events["champion_hp"][B]
                      for code in (HO, TN, CL)]
    assert hits["weak"] == [1, 2, 4]                                # Hold now removes 1 HP per turn
    assert all(s >= w for s, w in zip(hits["strong"], hits["weak"])) and hits["strong"][0] >= 2
    # m4_lane's Hold still removes nothing
    assert resolve_turn_m4(lane, m4, k, HO, HO, LANE, u=None).lane.events["champion_hp"][B] == 0


# ---------------------------------------------------------------------- the half-wave law
def test_half_wave_law_splits_the_turn_firepower_earlier_tick_first():
    per_turn = {h: (minion_damage(h, CAND, 0), minion_damage(h, CAND, 1)) for h in (1, 3, 4, 7, 8, 11, 12, 15, 16, 20, 24, 56)}
    assert per_turn == {1: (1, 0), 3: (1, 0), 4: (1, 0), 7: (1, 0), 8: (1, 1), 11: (1, 1), 12: (2, 1), 15: (2, 1),
                        16: (2, 2), 20: (3, 2), 24: (3, 3), 56: (7, 7)}
    for h in range(1, 60):
        assert sum(per_turn.get(h, (minion_damage(h, CAND, 0), minion_damage(h, CAND, 1)))) == max(1, 2 * h // CAND.wave_hp)
    # the whole-wave law ignores the tick, so every historical preset is untouched by the new argument
    for cfg in (GameConfig(), GameConfig.m4(), LANE):
        for h in range(1, 40):
            assert minion_damage(h, cfg, 0) == minion_damage(h, cfg, 1) == minion_damage(h, cfg) == max(1, h // cfg.wave_hp)


def test_half_wave_law_in_a_lane_fight():
    st = _lane(0, 12, 0, 8)                              # 12 HP against 8 at the lane, no champions
    r = step_turn(st, 1, W_HOLD, W_HOLD, CAND, record=True, champion_reach=(NO_REACH, NO_REACH), champion_budget=(0, 0))
    # (damage to A, damage to B) per tick: 12 HP deal 2 + 1; 8 HP deal 1, then, down to 6 HP (one per turn, on the
    # first tick), 0
    assert [tr["fight"][4:] for tr in r.trace] == [(1, 2), (0, 1)]
    assert (r.state.ah, r.state.bh) == (11, 5)
    r2 = step_turn(st, 1, W_HOLD, W_HOLD, LANE, record=True, champion_reach=(NO_REACH, NO_REACH), champion_budget=(0, 0))
    assert (r2.state.ah, r2.state.bh) == (10, 6)                     # the whole-wave law: 12 HP fight like 8


def test_pressure_thinning_is_off_and_the_wave_action_reaches_the_defenders():
    st = _lane(-4, 0, -3, 8, asg=10)                     # A's mass under B's tower, B's front behind it
    m4 = initial_m4(CAND).replace_side(0, pos=T - 1)
    for cfg, hold_hits, thinned in ((CAND, 1, 0), (LANE, 0, 1), (CAND.with_(thin_damage=1), 1, 1)):
        t = resolve_turn_m4(st, m4, 1, nf.HOP, HO, cfg, u=None)
        assert t.end[0] == T and t.reach[0] == T + 1
        assert t.lane.events["champion_hp"][B] == hold_hits and t.lane.thinned[A] == thinned
    # Hold + Pressure with the old thinning is Thin + Pressure without it: the same siege line, the same value
    old = wa.siege_lines(CAND.with_(thin_damage=1), 1.0, (("hold_pressure", nf.HOP),))["hold_pressure"]
    new = wa.siege_lines(CAND, 1.0, (("thin_pressure", wa.TNP),))["thin_pressure"]
    assert old["A_minus_B"] == new["A_minus_B"] and old["sieges_A"] == new["sieges_A"] and old["plates"] == new["plates"]


# ---------------------------------------------------------------------- invariants on the candidate
def _random_lane_turns(n, seed):
    rng = np.random.default_rng(seed)
    st = initial_state(CAND)
    for k in range(n):
        acts = [int(rng.integers(3)) for _ in (0, 1)]
        reach = tuple(int(rng.choice([NO_REACH, -2, -1, 0, 1, 2, 3])) for _ in (0, 1))
        budget = tuple(int(rng.integers(0, 24)) for _ in (0, 1))
        yield st, k, acts, reach, budget
        st = step_turn(st, k, acts[0], acts[1], CAND, champion_reach=reach, champion_budget=budget).state


def test_hp_conservation_kill_accounting_and_mirror_symmetry_on_random_lane_turns():
    for st, k, (a, b), reach, budget in _random_lane_turns(1500, 9):
        r = step_turn(st, k, a, b, CAND, champion_reach=reach, champion_budget=budget)
        before, after, ev = total_hp(st, CAND), total_hp(r.state, CAND), r.events
        for s in (0, 1):
            assert before[s] + ev["spawned_hp"][s] == after[s] + ev["killed_hp"][s] + ev["tower_hp"][s]
            assert ev["overflow_hp"][s] == 0 and ev["thinned_hp"][s] == 0
            assert sum(r.kills[1 - s].values()) == ev["killed_hp"][s] + ev["tower_hp"][s]
            assert r.state.side(s)[0] < T or r.state.side(s)[1] == 0
        m = step_turn(st.mirror(), k, b, a, CAND, champion_reach=reach[::-1], champion_budget=budget[::-1])
        assert m.state == r.state.mirror() and m.crash_hp == r.crash_hp[::-1] and m.kills == r.kills[::-1]
        assert m.tower_hits == r.tower_hits[::-1] and m.events["champion_hp"] == r.events["champion_hp"][::-1]


def test_resolver_mirror_equivariance_invariants_observation_and_reward_bound():
    asym = CAND.with_(skill_a=0.8, skill_b=0.3)
    env = LaneEnv(asym, max_turns=400)
    rng, nrng, ag = np.random.default_rng(1), np.random.default_rng(6), UniformAgent()
    obs = env.reset(seed=5)
    bound = per_turn_bound(asym)
    for _ in range(400):
        lane, m4, k = env.state, env.champ, env.k
        if k == 0:
            m4 = M4State(m4.a._replace(xp=200, items=30 * asym.gold_scale), m4.b)
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


def test_env_soak_deterministic_replay_and_digest_pin():
    soak(CAND, 1500, 13)
    env, ag = LaneEnv(CAND, max_turns=200), UniformAgent()
    traces = []
    for _ in range(2):
        obs = env.reset(seed=99)
        rng = np.random.default_rng(1)
        rows = []
        while True:
            r = env.step(ag.act(obs[0], rng), ag.act(obs[1], rng))
            rows.append((r.info["actions"], r.info["reward_units"], r.info["champ_after"], r.info["state_after"]))
            if r.done:
                break
            obs = r.obs
        traces.append(rows)
    assert traces[0] == traces[1]
    assert _digest(CAND) == "b1879ed22f3fe3ca0aca002b65a10b48d2e27e7ce5152fbc07c51c8fcba54e42"


# ---------------------------------------------------------------------- pins of the report's findings
def test_neutral_lane_is_symmetric_and_a_stronger_champion_pushes_by_last_hitting():
    for cfg in (LANE, CAND):
        s = nw.summarize(wa.play(cfg, nw.const(HO), 40), cfg)
        assert s["earned_A_minus_B"] == 0 and all(c in (None, 0) for c in s["contact"])
    assert wa.first_crash(wa.play(LANE, nw.const(HO), 40, mult=(3.0, 1.0))) is None
    strong = wa.first_crash(wa.play(CAND, nw.const(HO), 40, mult=(3.0, 1.0)))
    assert strong is not None and strong["k"] <= 12 and wa.budget_at(CAND, W_HOLD, 3.0) == 2


def test_the_wave_drifts_toward_a_champion_that_stops_contesting():
    assert wa.lane_only_drift(CAND, 0)["first_crash"] is None
    assert wa.lane_only_drift(CAND, 1)["first_crash"]["k"] <= 12
    for cfg, drifts in ((LANE, False), (CAND, True)):
        rows = wa.play(cfg, nw.const(HO, wa.SAFE), 40)
        assert (wa.first_crash(rows) is not None) == drifts
    # standing back costs the Safe champion far less under the candidate
    cost = {cfg: nw.summarize(wa.play(cfg, nw.const(HO, wa.SAFE), 40), cfg)["earned_A_minus_B"] for cfg in (LANE, CAND)}
    assert cost[CAND] < cost[LANE] / 2


def test_slow_push_compounds_under_the_half_wave_law_and_thin_scales_with_power():
    x1 = wa.until_crash_from_neutral(CAND, TN, 1)
    assert x1 is not None and x1["turns"] <= 16 and x1["hp"] >= 3 * CAND.wave_hp
    # the whole-wave law with last-hit damage cannot compound a one-turn edge in time
    floor = wa.until_crash_from_neutral(CAND.with_(damage_law="lanchester"), TN, 1)
    assert floor is None or floor["turns"] > x1["turns"]
    thin = [wa.until_crash_from_neutral(CAND, TN, 99, (m, 1.0))["turns"] for m in (1.0, 2.0, 3.0, 4.0)]
    assert thin[0] > thin[1] > thin[2] > thin[3] >= 1
    assert len({wa.until_crash_from_neutral(LANE, TN, 99, (m, 1.0))["turns"] for m in (1.0, 2.0, 4.0)}) == 1
    clear = [wa.until_crash_from_neutral(CAND, CL, 99, (m, 1.0))["turns"] for m in (1.0, 2.0, 4.0)]
    assert clear[0] > clear[1] >= clear[2] >= 1


def test_freeze_band_is_held_by_the_threshold_controller_and_broken_by_clear():
    fz = wa.threshold_freezer(14)
    held = wa.seeded_freeze(CAND, 8, 12, fz)
    assert held["held"] >= 24 and held["farm"][A] > held["farm"][B] and not held["crash_B"]
    assert wa.seeded_freeze(CAND, 8, 16, fz)["held"] <= 2                  # a 16-HP wave crashes through
    rows, start = wa.freeze_setup(CAND, fz, nf.SETUPS["B_thin_k1_4"])
    lane, m4, k0 = wa.hold_state(rows, start)
    ctrl = nw.summarize(wa.play(CAND, lambda k, l, m: (fz(l), HO), 20, lane, m4, k0), CAND)
    assert nf.held_turns(ctrl["contact"]) >= 16 and ctrl["farm"][A] > 1.4 * ctrl["farm"][B]
    broken = nw.summarize(wa.play(CAND, lambda k, l, m: (fz(l), CL), 6, lane, m4, k0), CAND)
    assert nf.held_turns(broken["contact"]) <= 2 and broken["crashes"][B]
    # a much stronger freezer pushes the wave out by last-hitting alone
    s3 = nw.summarize(wa.play(CAND, lambda k, l, m: (fz(l), HO), 20, lane, m4, k0, mult=(3.0, 1.0)), CAND)
    assert nf.held_turns(s3["contact"]) < 10 and s3["crashes"][A]


def test_crashes_bounce_and_sieges_do_not_sustain_themselves():
    r = nw.crash_report(wa.play(CAND, nw.until_crash(lambda k: TN if k == 1 else HO), 50))
    assert r["crash_hp"] >= 3 * CAND.wave_hp and r["clear_turns"] <= 8 and r["bounce_crash"]["hp"] >= 2 * CAND.wave_hp
    one = nf.sieges(wa.play(CAND.with_(tower_siege_damage=1), lambda k, l, m: (TN if k == 1 else HO, HO), 60), A)
    three = nf.sieges(wa.play(CAND, lambda k, l, m: (TN if k == 1 else HO, HO), 60), A)
    assert one[0]["turns"] > 30 and all(s["turns"] <= 8 for s in three)


def test_recall_after_own_crash_is_free_and_from_a_neutral_lane_costs_a_plate():
    free = wa.recall_pair(CAND, nw._crashed(lambda k: TN if 1 <= k <= 4 else HO), lambda k, l: l.asg > 0, wa.hold_pressure)
    neutral = wa.recall_pair(CAND, lambda k, l: HO, lambda k, l: k >= 12, wa.hold_pressure)
    assert free["cost_A_minus_B"] <= 0.0 < 2.0 < neutral["cost_A_minus_B"] and neutral["plates_recall"][B] == 1
