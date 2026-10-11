"""Champion mode (roadmap M2a): champion state, persistent positioning, Hold / Yield / Trade / Safe, physical
reach, siege compatibility, legality, observations and the environment. No exact solves; the legacy models
are checked through the unchanged lane step and the existing test files."""

import itertools

import numpy as np
import pytest

from lanegame import GameConfig, LaneGame
from lanegame.champions import (FARM, NO_REACH, RECALL, RETREAT, SAFE_MODE, TRADING, ChampSide, ChampState,
                                champion_collection, champion_gank_probability, champion_legal_actions,
                                in_contact, initial_champions, objective_reachable, recall_legal, resolve_turn,
                                swap_seats)
from lanegame.config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, CHAMP_ABSENT, CHAMP_BEHAVIORS, CONTEST, F,
                             N_CHAMP_ACTIONS, NONE, PRESSURE, SAFE, SH, SP, VISION, act, champ_act, champ_behavior,
                             champ_objective, champ_wave)
from lanegame.env import (LaneEnv, LegacyChampionAgent, ScriptedAgent, UniformAgent, as_policy, champion_agents,
                          feature_names, legal_mask, obs_size, observe, play_episode, run_episodes)
from lanegame.baselines import aggressive, passive_safe
from lanegame.game import gank_probability
from lanegame.mechanics import initial_state, legal_actions, step_turn, wave_actions
from lanegame.state import State

CFG = GameConfig.champions(horizon=16)
ASYM = GameConfig.champions(horizon=16, skill_a=0.8, skill_b=0.3)
T = CFG.tower

# lane states (own coordinates per side, see state.py)
CONTACT = State(0, 4, (), 0, 0, 4, (), 0, 0, 0)                 # fronts fighting at the lane (x = 0)
SIEGE_A = State(-4, 4, (), 8, -3, 4, (), 0, 0, 0, 2, 0)         # A's 8 HP under B's tower, B's wave waiting at -3


def ch(pa, pb, hp=6):
    return ChampState(ChampSide(pa, hp, 0), ChampSide(pb, hp, 0))


def H(w=SP, o=NONE):
    return champ_act(w, B_HOLD, o)


def Y(w=SP):
    return champ_act(w, B_YIELD)


def TR(w=SP, o=NONE):
    return champ_act(w, B_TRADE, o)


def S(w=SP):
    return champ_act(w, B_SAFE)


def random_play(cfg, episodes=6, seed=0):
    """(lane, champ, k, a, b, StepResult) for uniformly random play."""
    env = LaneEnv(cfg)
    rng = np.random.default_rng(seed)
    out = []
    for e in range(episodes):
        oa, ob = env.reset(seed=e)
        while True:
            lane, champ, k = env.state, env.champ, env.k
            a, b = oa.legal[rng.integers(len(oa.legal))], ob.legal[rng.integers(len(ob.legal))]
            r = env.step(a, b)
            out.append((lane, champ, k, a, b, r))
            if r.done:
                break
            oa, ob = r.obs
    return out


# ------------------------------------------------------------------ configuration and encoding
def test_legacy_configurations_are_untouched():
    for cfg, n, m in ((GameConfig(), 18, 34), (GameConfig.v2(), 6, 34), (GameConfig.v1(), 3, 32)):
        assert not cfg.champion_mode
        assert len(cfg.action_codes()) == n and obs_size(cfg) == m
        assert not any("champ" in f for f in feature_names(cfg))
        assert cfg.label(1) in ("SH-C", "SH")


def test_champion_configuration():
    cfg = GameConfig.champions()
    cfg.validate()
    assert cfg.champion_mode and cfg.objectives and cfg.horizon == GameConfig().horizon
    assert cfg.action_codes() == tuple(range(N_CHAMP_ACTIONS)) and N_CHAMP_ACTIONS == 46 and CHAMP_ABSENT == 45
    assert obs_size(cfg) == 50 and len(feature_names(cfg)) == 50
    assert cfg.label(champ_act(SH, B_TRADE, PRESSURE)) == "SH-T+P" and cfg.label(CHAMP_ABSENT) == "ABSENT"
    assert GameConfig.from_dict(ASYM.to_dict()) == ASYM
    with pytest.raises(AssertionError):
        GameConfig.champions(skill_a=1.5).validate()
    with pytest.raises(AssertionError):
        GameConfig.v2(champion_mode=True).validate()           # champion mode builds on the siege model


def test_action_encoding_round_trip():
    seen = set()
    for w, b, o in itertools.product((SP, SH, F), range(5), (NONE, PRESSURE, VISION)):
        a = champ_act(w, b, o)
        assert (champ_wave(a), champ_behavior(a), champ_objective(a)) == (w, b, o) and 0 <= a < CHAMP_ABSENT
        seen.add(a)
    assert seen == set(range(CHAMP_ABSENT))


# ------------------------------------------------------------------ the lane step is unchanged
def test_lane_step_with_legacy_reach_equals_legacy_step():
    """`champion_reach` is the only change to mechanics: passing the legacy reach lines reproduces the legacy
    step exactly on every reachable transition (siege model, K = 8)."""
    cfg = GameConfig(horizon=8)
    g = LaneGame(cfg)
    n = 0
    for k, layer in enumerate(g.reachable()):
        for s in layer:
            for a in g.legal(s, 0):
                for b in g.legal(s, 1):
                    lines = tuple(T if cfg.posture_cap[x % 6] is None else cfg.posture_cap[x % 6] for x in (a, b))
                    r0 = step_turn(s, k, a, b, cfg)
                    r1 = step_turn(s, k, a, b, cfg, champion_reach=lines)
                    assert (r1.state, r1.kills, r1.events, r1.crash_hp, r1.plates, r1.thinned, r1.exposure) == \
                           (r0.state, r0.kills, r0.events, r0.crash_hp, r0.plates, r0.thinned, r0.exposure)
                    n += 1
    assert n > 1000


# ------------------------------------------------------------------ champion state
def test_initial_champion_state_and_asymmetric_skills():
    env = LaneEnv(ASYM)
    oa, ob = env.reset(seed=0)
    assert env.champ == initial_champions(ASYM) == ch(-T, -T)
    assert env.champ.a == ChampSide(-T, 6, 0) and oa.champ is env.champ
    idx = {n: i for i, n in enumerate(feature_names(ASYM))}
    assert oa.vector[idx["own_champ_skill"]] == pytest.approx(0.8) and oa.vector[idx["enemy_champ_skill"]] == pytest.approx(0.3)
    assert ob.vector[idx["own_champ_skill"]] == pytest.approx(0.3) and ob.vector[idx["enemy_champ_skill"]] == pytest.approx(0.8)
    assert oa.vector[idx[f"own_champ_x{-T:+d}"]] == 1 and oa.vector[idx[f"enemy_champ_x{T:+d}"]] == 1
    assert oa.vector[idx["own_champ_hp"]] == 1 and oa.vector[idx["own_champ_absent"]] == 0
    assert ChampState.from_list(env.champ.to_list()) == env.champ
    with pytest.raises(ValueError):
        LaneEnv(GameConfig()).reset(seed=0, champ=ch(0, 0))


def test_observation_is_bounded_markov_and_mirror_equivariant():
    """observe(lane, k, B, cfg, champ) == observe(mirror lane, k, A, swap_seats(cfg), mirror champ), with
    asymmetric skills; one-hot positions; HP, absence and skill exactly represented."""
    m = obs_size(ASYM)
    for lane, champ, k, a, b, r in random_play(ASYM, episodes=4):
        va, vb = observe(lane, k, 0, ASYM, champ), observe(lane, k, 1, ASYM, champ)
        assert va.shape == vb.shape == (m,) and np.all((va >= 0) & (va <= 1))
        assert np.array_equal(vb, observe(lane.mirror(), k, 0, swap_seats(ASYM), champ.mirror()))
        tail = va[-16:]
        assert tail[:5].sum() == 1 and tail[8:13].sum() == 1
        assert int(np.argmax(tail[:5])) - T == champ.a.pos and T - int(np.argmax(tail[8:13])) == champ.b.pos
    with pytest.raises(ValueError):
        observe(initial_state(CFG), 0, 0, CFG)                  # champion mode needs the champion state


def test_movement_is_bounded_persistent_and_hp_accounted():
    """Present champions move at most champion_speed cells per turn and keep their cell between turns; an away
    champion (recall, absence) is at its own tower. HP changes only by the damage recorded for the turn, or is
    restored on return."""
    cfg = CFG
    prev_end = None
    for lane, champ, k, a, b, r in random_play(cfg, episodes=6, seed=3):
        i = r.info
        start, end = i["champ_start"], i["champ_end"]
        if k > 0 and prev_end is not None:
            assert start == prev_end                           # position persists between turns
        assert (champ.a.pos, champ.b.pos) == start
        for s in (0, 1):
            assert -T <= end[s] <= T
            if i["away"][s]:
                assert end[s] == -T and i["exposure"][s] == -T and i["gank_p"][s] == 0
            else:
                assert abs(end[s] - start[s]) <= cfg.champion_speed
                assert i["exposure"][s] == max(start[s], end[s])
        for s in (0, 1):
            before, after = i["champ_before"][s], i["champ_after"][s]
            assert 0 <= after[1] <= cfg.champion_hp
            if i["died"][s]:
                assert before[1] - i["damage_taken"][s] <= 0 and after == [-T, 0, cfg.death_turns]
            elif after[2] == 0 and i["away"][s]:
                assert after[1] == cfg.champion_hp                   # returned with full HP
            elif not i["away"][s]:
                assert after[1] == before[1] - i["damage_taken"][s] > 0
        prev_end = None if r.done else tuple(c[0] for c in i["champ_after"])   # (a champion that died restarts at -T)


def test_resolve_turn_is_mirror_equivariant():
    for lane, champ, k, a, b, r in random_play(ASYM, episodes=3, seed=5):
        t = resolve_turn(lane, champ, k, a, b, ASYM)
        m = resolve_turn(lane.mirror(), champ.mirror(), k, b, a, swap_seats(ASYM))
        assert m.end == t.end[::-1] and m.exposure == t.exposure[::-1] and m.modes == t.modes[::-1]
        assert m.connected == t.connected[::-1] and m.threatened == t.threatened[::-1]
        assert m.lane.state == t.lane.state.mirror() and m.champ == t.champ.mirror()
        assert np.allclose(m.farm, t.farm[::-1]) and np.allclose(m.gank_p, t.gank_p[::-1])


# ------------------------------------------------------------------ movement rules
def test_hold_follows_its_wave():
    s0, c0 = initial_state(CFG), initial_champions(CFG)
    t0 = resolve_turn(s0, c0, 0, H(), H(), CFG)
    assert t0.end == (-T, -T)                                  # waves at -2 after turn 0
    t1 = resolve_turn(t0.lane.state, t0.champ, 1, H(), H(), CFG)
    assert t1.lane.state.ax == 0 and t1.end == (0, 0) and t1.farm_spot == (0, 0)
    assert t1.modes == (FARM, FARM) and t1.reach == (1, 1) and t1.farm[0] > 0


def test_trade_approaches_and_connects_with_a_holder():
    t = resolve_turn(CONTACT, ch(-T, 0), 2, TR(), H(), CFG)
    assert t.end == (-1, 0) and t.connected == (True, False) and t.modes == (TRADING, FARM)   # one cell short suffices
    t = resolve_turn(CONTACT, ch(0, 0), 2, TR(), H(), CFG)
    assert t.end == (0, 0) and t.start == (0, 0)               # already in contact: no extra step forward


def test_safe_retreats_but_stays_exposed():
    t = resolve_turn(CONTACT, ch(T, 0), 2, S(), H(), CFG)
    assert t.end[0] == T - CFG.champion_speed == 0 and t.exposure[0] == T  # no teleport; exposed on the way back
    t2 = resolve_turn(t.lane.state, t.champ, 3, S(), H(), CFG)
    assert t2.end[0] == min(t2.farm_spot[0], CFG.safe_line) and t2.modes[0] == SAFE_MODE and t2.reach[0] <= CFG.safe_line
    t3 = resolve_turn(CONTACT, ch(-T, 0), 2, S(), H(), CFG)
    assert t3.end[0] == CFG.safe_line                         # from behind, Safe follows its wave up to the safe line


# ------------------------------------------------------------------ conditional behavior
def test_unthreatened_yield_is_exactly_hold():
    for lane, champ, k, a, b, r in random_play(CFG, episodes=4, seed=7):
        if champ_behavior(a) != B_HOLD or champ_objective(a) != NONE or champ_behavior(b) == B_TRADE:
            continue
        ty = resolve_turn(lane, champ, k, champ_act(champ_wave(a), B_YIELD), b, CFG)
        th = resolve_turn(lane, champ, k, a, b, CFG)
        assert ty.end == th.end and ty.farm == th.farm and ty.gank_p == th.gank_p and ty.lane.state == th.lane.state
        assert ty.threatened == (False, False) and ty.modes == th.modes


def test_yield_retreats_from_a_credible_trade():
    ty = resolve_turn(CONTACT, ch(0, 0), 2, Y(), TR(), CFG)
    th = resolve_turn(CONTACT, ch(0, 0), 2, H(), TR(), CFG)
    assert ty.threatened == (True, False) and ty.modes[0] == RETREAT
    assert not ty.connected[1] and not in_contact(ty.end[0], ty.end[1], CFG)
    assert ty.end[0] < th.end[0] and ty.reach[0] <= CFG.safe_line
    assert th.connected == (False, True) and th.modes == (FARM, TRADING)   # Hold keeps farming and is traded
    assert ty.farm[0] < th.farm[0]                             # yielding gives up farm
    assert ty.modes[1] == FARM and ty.farm[1] > th.farm[1]     # the trader falls back to farming at full rate


def test_trade_falls_back_to_farming_and_keeps_its_exposure():
    t = resolve_turn(CONTACT, ch(0, -1), 2, TR(), S(), CFG)
    assert t.threatened == (False, True) and t.connected == (False, False)
    assert t.modes == (FARM, SAFE_MODE) and t.reach[0] == min(T, t.end[0] + CFG.farm_reach)
    assert t.exposure[0] == max(0, t.end[0])
    p_trade = champion_gank_probability(CONTACT, 0, SP, B_TRADE, CFG, t.exposure[0])
    assert t.gank_p[0] == p_trade > champion_gank_probability(CONTACT, 0, SP, B_HOLD, CFG, t.exposure[0])


def test_connected_trade_gives_up_shove_and_farm():
    """A connected Shove + Trade adds no shove damage and collects only trade_farm_factor of its farm."""
    tt = resolve_turn(CONTACT, ch(0, 0), 2, TR(SH), H(), CFG)
    th = resolve_turn(CONTACT, ch(0, 0), 2, H(SH), H(), CFG)
    assert tt.connected[0] and tt.reach[0] == NO_REACH and th.reach[0] >= 0
    assert tt.lane.events["killed_hp"][1] < th.lane.events["killed_hp"][1]
    nos = step_turn(CONTACT, 2, SP, SP, CFG, champion_reach=(NO_REACH, th.reach[1]))
    assert tt.lane.state == nos.state                          # Shove without the champion's damage = no push
    for x in range(-T - 1, T + 2):
        assert champion_collection(x, TRADING, 0, FARM, 0, CFG) == pytest.approx(
            CFG.trade_farm_factor * champion_collection(x, FARM, 0, FARM, 0, CFG))


def test_trade_against_trade_and_tower_protection():
    t = resolve_turn(CONTACT, ch(-T, -T), 2, TR(), TR(), CFG)
    assert t.end == (0, 0) and t.connected == (True, True)
    assert in_contact(0, 1, CFG) and in_contact(-1, 1, CFG) and not in_contact(0, 2, CFG)   # within trade_range
    assert in_contact(-T, T, CFG) and in_contact(T, -T, CFG)       # inside the defender's tower cell
    assert not in_contact(-T, T - 1, CFG)                          # one cell outside it: the tower protects
    # a champion hugging its tower (its next wave waits behind the tower) is reachable only by a dive
    t = resolve_turn(SIEGE_A, ch(1, -T), 2, TR(), S(), CFG)
    assert t.farm_spot[1] == -T and t.end == (T, -T) and t.connected[0] and t.threatened[1] and t.exposure[0] == T


# ------------------------------------------------------------------ farming and gank risk
def test_physical_farm_reach_and_zoning():
    for x in range(-4, 5):
        for pos in range(-T, T + 1):
            c = champion_collection(x, FARM, pos, SAFE_MODE, -T, CFG)
            assert (c == 0) == (abs(x - pos) > CFG.farm_reach or c == 0)
            if abs(x - pos) > CFG.farm_reach:
                assert c == 0
    assert champion_collection(T, FARM, -T, SAFE_MODE, -T, CFG) == 0            # no farming at the enemy tower from home
    zoned = champion_collection(0, FARM, 0, FARM, 0, CFG)
    free = champion_collection(0, FARM, 0, FARM, -T, CFG)                      # opponent too far to zone
    assert zoned < free == pytest.approx(CFG.collect_contest)


def test_gank_terms_have_the_legacy_form_with_champion_parameters():
    """M2d: the legacy formula (base, exposure, action, wave strength, vision) with the champ_gank_* parameters;
    with those set to the legacy values, Hold and Safe reproduce the legacy Contest and Safe terms."""
    as_legacy = CFG.with_(champ_gank_base=CFG.gank_base, champ_gank_position=CFG.gank_position,
                          champ_gank_strength=CFG.gank_strength, champ_gank_max=CFG.gank_max,
                          champ_gank_behavior=(0.02, 0.02, 0.05, 0.0), champ_gank_shove=0.03)
    lane = CONTACT._replace(va=2)
    for w in (SP, SH, F):
        for e in range(-T, T + 1):
            assert champion_gank_probability(lane, 0, w, B_HOLD, as_legacy, e) == pytest.approx(gank_probability(lane, 0, act(w, CONTEST), CFG, e))
            assert champion_gank_probability(lane, 0, w, B_SAFE, as_legacy, e) == pytest.approx(gank_probability(lane, 0, act(w, SAFE), CFG, e))
            assert champion_gank_probability(lane, 0, w, B_TRADE, CFG, e) >= champion_gank_probability(lane, 0, w, B_HOLD, CFG, e)                 >= champion_gank_probability(lane, 0, w, B_SAFE, CFG, e)


# ------------------------------------------------------------------ legality and siege
def test_masks_match_the_legal_sets_and_no_redundant_codes():
    for lane, champ, k, a, b, r in random_play(CFG, episodes=6, seed=11):
        for side in (0, 1):
            legal = champion_legal_actions(lane, champ, side, CFG)
            assert tuple(int(x) for x in np.flatnonzero(legal_mask(lane, side, CFG, champ=champ))) == legal
            assert legal == tuple(sorted(set(legal)))
            if champ.side(side).absent > 0:
                assert legal == (CHAMP_ABSENT,)
                continue
            recall = recall_legal(champ, side, CFG)
            assert (RECALL in legal) == recall
            legal = tuple(x for x in legal if x != RECALL)
            waves = wave_actions(lane, side, CFG)
            assert {champ_wave(x) for x in legal} == set(waves)          # Freeze stays state-dependent
            reach = objective_reachable(lane, champ, side, CFG)
            for x in legal:
                assert x != CHAMP_ABSENT and champ_behavior(x) != B_RECALL
                if champ_objective(x) != NONE:
                    assert reach and champ_behavior(x) in (B_HOLD, B_TRADE)
            assert len(legal) == 4 * len(waves) + (2 * 2 * len(waves) if reach else 0)   # no redundant codes
        assert a in r.info["legal"][0] and b in r.info["legal"][1]


def test_siege_objectives_need_the_champion_at_the_tower():
    far, near = ch(-T, -T), ch(0, -T)
    assert not objective_reachable(SIEGE_A, far, 0, CFG) and objective_reachable(SIEGE_A, near, 0, CFG)
    assert all(champ_objective(x) == NONE for x in champion_legal_actions(SIEGE_A, far, 0, CFG))
    assert H(SP, PRESSURE) in champion_legal_actions(SIEGE_A, near, 0, CFG)
    assert Y() in champion_legal_actions(SIEGE_A, near, 0, CFG)
    assert champ_act(SP, B_YIELD, PRESSURE) not in champion_legal_actions(SIEGE_A, near, 0, CFG)
    # no remote siege: a Hold champion far from the tower claims nothing, thins nothing, sets no vision
    t = resolve_turn(SIEGE_A, far, 2, H(), S(), CFG)
    assert t.lane.plates == (0, 0) and t.lane.thinned == (0, 0) and t.lane.vision_set == (False, False)
    # a champion that walks to the tower applies pressure / establishes vision
    t = resolve_turn(SIEGE_A, near, 2, H(SP, PRESSURE), S(), CFG)
    assert t.end[0] == T and t.lane.plates[0] == 1 and t.effective_objectives[0] == PRESSURE and t.exposure[0] == T
    t = resolve_turn(SIEGE_A, near, 2, H(SP, VISION), S(), CFG)
    assert t.lane.vision_set[0] and t.lane.state.va == CFG.vision_turns
    # a Trade + Pressure whose trade connects (defender at its tower) suspends the objective
    t = resolve_turn(SIEGE_A, near, 2, TR(SP, PRESSURE), H(), CFG)
    assert t.connected[0] and t.effective_objectives[0] == NONE and t.lane.plates == (0, 0)
    t = resolve_turn(SIEGE_A, near, 2, H(SP, PRESSURE), H(), CFG)
    assert t.lane.plates[0] == 1                               # Hold + Pressure does not trade


def test_illegal_champion_codes_are_rejected():
    """Recall with a wave other than Slow Push, ABSENT while present, Yield with an objective: never legal."""
    env = LaneEnv(CFG)
    oa, ob = env.reset(seed=0)
    assert RECALL in oa.legal and champ_act(SH, B_RECALL) not in oa.legal and CHAMP_ABSENT not in oa.legal
    for bad in (champ_act(SH, B_RECALL), CHAMP_ABSENT, champ_act(SP, B_YIELD, PRESSURE)):
        with pytest.raises(ValueError):
            env.step(bad, ob.legal[0])


# ------------------------------------------------------------------ environment
def test_scripted_champion_episodes_are_zero_sum_reproducible_and_complete():
    agents = champion_agents(CFG)
    env = LaneEnv(CFG)
    for na, nb in (("hold", "trade"), ("yield", "shove_trade"), ("safe", "uniform"), ("legacy:aggressive", "hold_vision")):
        l1 = play_episode(env, agents[na], agents[nb], seed=9, record=True)
        l2 = play_episode(env, agents[na], agents[nb], seed=9, record=True)
        assert l1.length == CFG.horizon and l1.actions == l2.actions and l1.rewards == l2.rewards
        for t in l1.turns:
            assert t["reward"] == pytest.approx(t["farm_payoff"] + t["plate_payoff"] + t["gank_payoff"] + t["bounty_payoff"])
            assert t["crash_payoff"] == 0 and t["actions"][0] in t["legal"][0] and t["actions"][1] in t["legal"][1]
        assert abs(sum(l1.rewards) - l1.payoff) < 1e-9
    st = run_episodes(env, agents["trade"], agents["hold"], 20, seed=1)
    assert st.champion_mode and st.trades_connected[:, 0].mean() > 0 and st.n_actions == 46 and st.obs_size == 50


def test_legacy_agents_map_into_champion_actions_and_are_refused_raw():
    env = LaneEnv(CFG)
    oa, ob = env.reset(seed=0)
    with pytest.raises(TypeError):
        ScriptedAgent(aggressive()).act(oa, np.random.default_rng(0))
    with pytest.raises(TypeError):
        as_policy(UniformAgent(), CFG)
    la = LegacyChampionAgent(aggressive(), CFG)
    assert la.act(oa, np.random.default_rng(0)) == H(SH)
    assert LegacyChampionAgent(passive_safe(), CFG).act(ob, np.random.default_rng(0)) == S(SP)
    oa, _ = env.reset(seed=0, root=SIEGE_A, k0=2, champ=ch(0, -T))
    rng = np.random.default_rng(0)
    assert la.act(oa, rng) == H(SH, PRESSURE)
    oa, _ = env.reset(seed=0, root=SIEGE_A, k0=2, champ=ch(-T, -T))
    assert la.act(oa, rng) == H(SH)                           # objective dropped: champion too far


def test_pettingzoo_champion_mode():
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test
    from lanegame.env.pettingzoo_adapter import LaneParallelEnv
    env = LaneParallelEnv(CFG)
    parallel_api_test(env, num_cycles=40)
    obs, _ = env.reset(seed=0, options={"champ": ch(0, 0).to_list(), "root": CONTACT.to_list(), "k0": 2})
    assert env.env.champ == ch(0, 0) and obs["A"]["observation"].shape == (50,) and obs["A"]["action_mask"].shape == (46,)


def test_env_demo_cli_champion_mode(capsys):
    from lanegame.__main__ import main
    main(["env-demo", "--model", "champions", "--a", "trade", "--b", "hold", "--horizon", "8", "--episodes", "5"])
    out = capsys.readouterr().out
    assert "[champions]" in out and "trades per game" in out and "observation size 50, action space 46" in out
