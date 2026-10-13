"""M4.3c stabilization, Stage B (docs/m4/m4_3c_strategic_economics_calibration.md): the EXPERIMENTAL opt-in economic
fields (xp_reward_weight, death_turns_by_level, terminal_rule) and the study's instruments. Every default reproduces
every earlier preset exactly."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import N_CHANCE, champion_legal_actions, mirror_chance, swap_seats
from lanegame.champions_m4 import context, initial_m4, resolve_turn_m4
from lanegame.config import B_HOLD, B_TRADE, PRESSURE, W_HOLD, W_SHOVE, champ_act
from lanegame.env import LaneEnv, UniformAgent
from lanegame.env.discounted import tail_constant
from lanegame.mechanics import initial_state

FOUR = GameConfig.m4_four_actions()
A, B = 0, 1


def _random_turns(cfg, turns=300, seed=11):
    """(turn, M4Turn) of uniformly random legal play with chance."""
    rng = np.random.default_rng(seed)
    lane, m4 = initial_state(cfg), initial_m4(cfg)
    for k in range(turns):
        la, lb = champion_legal_actions(lane, m4, A, cfg), champion_legal_actions(lane, m4, B, cfg)
        a, b = la[rng.integers(len(la))], lb[rng.integers(len(lb))]
        u = tuple(rng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, u=u)
        yield k, lane, m4, a, b, u, t
        lane, m4 = t.lane.state, t.champ


# ---------------------------------------------------------------------- defaults and validation
def test_the_new_fields_are_off_in_every_preset_and_rejected_outside_m4():
    for cfg in (FOUR, GameConfig.m4_lasthit(), GameConfig.m4_lane(), GameConfig.m4()):
        assert (cfg.xp_reward_weight, cfg.death_turns_by_level, cfg.terminal_rule) == (0.0, (), "none")
    for kw in ({"xp_reward_weight": 0.125}, {"terminal_rule": "final_plate"},
               {"death_turns_by_level": (3,) * 18, "champion_absent_max": 3}):
        with pytest.raises(ValueError):
            GameConfig.champions(**kw).validate()
        GameConfig.m4_four_actions(**kw).validate()
    with pytest.raises(ValueError, match="xp_reward_weight"):
        GameConfig.m4_four_actions(xp_reward_weight=1e-8).validate()          # not a ledger multiple
    with pytest.raises(ValueError, match="death_turns_by_level"):
        GameConfig.m4_four_actions(death_turns_by_level=(3, 4)).validate()
    with pytest.raises(ValueError, match="champion_absent_max"):
        GameConfig.m4_four_actions(death_turns_by_level=(5,) * 18).validate()
    with pytest.raises(ValueError, match="terminal_rule"):
        GameConfig.m4_four_actions(terminal_rule="first_plate").validate()


def test_defaults_change_no_turn():
    """With every new field at its default the resolver's turns are those of the unchanged preset: no XP reward
    units, never terminal, the death absence of death_turns."""
    for _, lane, m4, a, b, u, t in _random_turns(FOUR, 200):
        assert t.xp_reward_units == 0 and not t.terminal
        assert t.reward_units == t.earned_units[A] - t.earned_units[B]


# ---------------------------------------------------------------------- the XP channel of the reward
def test_xp_reward_is_exact_zero_sum_and_changes_nothing_else():
    lam = 0.125
    cfg = GameConfig.m4_four_actions(xp_reward_weight=lam)
    units = int(round(lam * cfg.gold_scale))
    seen = 0
    plain = _random_turns(FOUR, 300)
    for (k, lane, m4, a, b, u, t), (_, _, _, _, _, _, t0) in zip(_random_turns(cfg, 300), plain):
        # identical dynamics: the reward is not part of the state
        assert t.champ == t0.champ and t.lane.state == t0.lane.state and t.earned_units == t0.earned_units
        assert t.xp_reward_units == units * (t.xp_gain[A] - t.xp_gain[B])
        assert t.reward_units == t.earned_units[A] - t.earned_units[B] + t.xp_reward_units
        assert t.reward == t.reward_units / cfg.gold_scale
        m = resolve_turn_m4(lane.mirror(), m4.mirror(), k, b, a, swap_seats(cfg), u=mirror_chance(u))
        assert m.reward_units == -t.reward_units
        seen += t.xp_reward_units != 0
    assert seen > 50


def test_env_reports_the_xp_channel_and_the_tail_bound_grows_with_it():
    cfg = GameConfig.m4_four_actions(xp_reward_weight=0.125)
    env, ag, rng = LaneEnv(cfg, max_turns=60), UniformAgent(), np.random.default_rng(0)
    obs = env.reset(seed=3)
    while True:
        r = env.step(ag.act(obs[0], rng), ag.act(obs[1], rng))
        i = r.info
        assert i["reward_units"] == i["earned_units"][0] - i["earned_units"][1] + i["xp_reward_units"]
        assert r.rewards[0] == -r.rewards[1] == i["reward"]
        if r.done:
            break
        obs = r.obs
    assert tail_constant(GameConfig.m4_lasthit()) == pytest.approx(107.04, abs=0.01)     # unchanged at lambda = 0
    assert tail_constant(cfg) > tail_constant(FOUR)


# ---------------------------------------------------------------------- level-dependent death absence
def test_death_absence_follows_the_level_schedule():
    sched = tuple(2 + -(-L // 6) for L in range(1, 19))                     # 3 / 4 / 5
    cfg = GameConfig.m4_four_actions(death_turns_by_level=sched, champion_absent_max=5)
    ctx, t = context(cfg), context(cfg).t
    lane = initial_state(cfg)._replace(ax=0, ah=8, bx=0, bh=8)
    for L, want in ((1, 3), (6, 3), (7, 4), (12, 4), (13, 5), (18, 5)):
        m4 = initial_m4(cfg)
        xp = t.xp_cum[L]
        # B at 1 centi-HP in contact with A's Trade: a connected trade that lands kills it (u = 0 lands every roll)
        m4 = m4.replace_side(A, xp=xp, hp=t.hp_max[L], pos=0).replace_side(B, xp=xp, hp=1, pos=0)
        turn = resolve_turn_m4(lane, m4, 2, champ_act(W_HOLD, B_TRADE), champ_act(W_HOLD, B_HOLD), cfg,
                               u=(0.0,) * N_CHANCE)
        assert turn.died[B] and turn.champ.b.absent == want == ctx.death_turns(L)
    base = context(FOUR)
    assert all(base.death_turns(L) == 3 for L in range(1, 19))


# ---------------------------------------------------------------------- terminal_rule "final_plate"
def _near_last_plate(cfg):
    """A at B's tower with its crashed minions there, 1 centi-unit short of the last threshold: Pressure ends it."""
    lane = initial_state(cfg)._replace(asg=8, bx=-3, bh=8)
    m4 = initial_m4(cfg).replace_side(A, pos=2, tower_dmg=cfg.plate_thresholds[-1] - 1)
    return lane, m4


def test_final_plate_terminates_exactly_at_the_crossing_turn():
    cfg = GameConfig.m4_four_actions(terminal_rule="final_plate")
    lane, m4 = _near_last_plate(cfg)
    press = champ_act(W_SHOVE, B_HOLD, PRESSURE)
    assert press in champion_legal_actions(lane, m4, A, cfg)
    t = resolve_turn_m4(lane, m4, 2, press, champ_act(W_HOLD, B_HOLD), cfg, u=None)
    assert t.terminal and t.plates_broken[A] == 1 and t.plate_units[A] == context(cfg).plate_units
    off = resolve_turn_m4(lane, m4, 2, press, champ_act(W_HOLD, B_HOLD), FOUR, u=None)
    assert not off.terminal and off.reward_units == t.reward_units          # the same turn, paid the same
    hold = resolve_turn_m4(lane, m4, 2, champ_act(W_HOLD, B_HOLD), champ_act(W_HOLD, B_HOLD), cfg, u=None)
    assert not hold.terminal                                                  # not crossed: the game goes on
    env = LaneEnv(cfg, max_turns=10)
    env.reset(seed=0, root=lane, champ=m4, k0=2)
    r = env.step(press, champ_act(W_HOLD, B_HOLD))
    assert r.terminated and not r.truncated and r.done and r.info["terminal"]
    with pytest.raises(RuntimeError):
        env.step(press, champ_act(W_HOLD, B_HOLD))


def test_termination_semantics_in_training_and_evaluation():
    """GAE drops the bootstrap at a termination and keeps it at a truncation; a terminated evaluation game simply stops
    (its later rewards are 0)."""
    from lanegame.rl.ppo import compute_gae
    r = np.array([[1.0], [1.0]])
    v = np.array([[0.5], [0.5]])
    nv = np.array([[0.5], [9.0]])
    term = np.array([[False], [True]])
    trunc = np.array([[False], [False]])
    adv, ret = compute_gae(r, v, nv, term, trunc, 0.99, 0.95)
    assert adv[1, 0] == pytest.approx(1.0 - 0.5)                               # no gamma V(s') after the end
    adv2, _ = compute_gae(r, v, nv, np.zeros_like(term), np.array([[False], [True]]), 0.99, 0.95)
    assert adv2[1, 0] == pytest.approx(1.0 + 0.99 * 9.0 - 0.5)                  # a truncation bootstraps
    from lanegame.env.discounted import _play
    cfg = GameConfig.m4_four_actions(terminal_rule="final_plate")
    lane, m4 = _near_last_plate(cfg)
    from lanegame.env.starts import StartState
    st = StartState(lane=lane, champ=m4, phase=2)

    class Press:
        def act(self, obs, rng=None):
            c = champ_act(W_SHOVE, B_HOLD, PRESSURE)
            return c if c in obs.legal else obs.legal[0]

    rewards, _ = _play(LaneEnv(cfg), (Press(), Press()), np.random.SeedSequence(1), 50, st)
    assert len(rewards) == 1


# ---------------------------------------------------------------------- the calibrated candidate preset
def test_candidate_preset_changes_only_its_documented_fields():
    cand = GameConfig.m4_four_actions_econ()
    cand.validate()
    changed = {k for k, v in cand.to_dict().items() if FOUR.to_dict()[k] != v}
    assert changed == set(GameConfig.M4_FOUR_ECON_FIELDS)
    assert (cand.collect_contest_own_tower, cand.trade_farm_factor, cand.kill_bounty) == (0.8, 0.8, 2.0)
    assert cand.death_turns_by_level == tuple([3] * 6 + [4] * 6 + [5] * 6) and cand.champion_absent_max == 5
    assert (cand.xp_reward_weight, cand.terminal_rule) == (0.0, "none")       # gold-only reward, continuing time
    env = LaneEnv(cand)
    assert (env.n_actions, env.obs_size) == (61, 88)
    from lanegame.experiments.m43c.economics import variant_cfg
    assert variant_cfg("candidate") == variant_cfg("ot80_tf80_k2_dS1") == cand
    for cfg in (FOUR, GameConfig.m4_lasthit()):                                # the baselines are untouched
        assert (cfg.collect_contest_own_tower, cfg.trade_farm_factor, cfg.kill_bounty) == (0.6, 0.5, 0.5)
        assert cfg.death_turns_by_level == () and cfg.champion_absent_max == 3


def test_candidate_keeps_the_four_action_wave_physics():
    """Without chance events (no trade lands, nobody is ganked or dies) the candidate's lane is the four-action
    baseline's turn by turn: only the economy differs (collection rates, bounty, death absence)."""
    from lanegame.experiments.m43c import harness as hh
    from lanegame.experiments.m43c import strategies as st
    cand = GameConfig.m4_four_actions_econ()
    for start, a, b in (("neutral", "grow_shove4", "farm4"), ("freeze", "freeze_hunt4", "walk_up4"),
                        ("neutral", "slowpush4", "thin4")):
        rows = []
        for cfg in (FOUR, cand):
            g = hh.play(cfg, hh.start(start, cfg, level=6), (st.StrategyAgent(st.LIBRARY[a]),
                                                             st.StrategyAgent(st.LIBRARY[b])),
                        np.random.SeedSequence(0), 60, chance="none")
            rows.append([(r["crash"], r["contact"], r["applied"], r["xp"], r["pos"]) for r in g["rows"]])
        assert rows[0] == rows[1]


def test_candidate_death_absence_and_bounty():
    cand = GameConfig.m4_four_actions_econ()
    ctx, t = context(cand), context(cand).t
    lane = initial_state(cand)._replace(ax=0, ah=8, bx=0, bh=8)
    for L, want in ((3, 3), (9, 4), (15, 5)):
        xp = t.xp_cum[L]
        m4 = initial_m4(cand).replace_side(A, xp=xp, hp=t.hp_max[L], pos=0).replace_side(B, xp=xp, hp=1, pos=0)
        turn = resolve_turn_m4(lane, m4, 2, champ_act(W_HOLD, B_TRADE), champ_act(W_HOLD, B_HOLD), cand,
                               u=(0.0,) * N_CHANCE)
        assert turn.died[B] and turn.champ.b.absent == want
        assert turn.bounty_units[A] == 2 * cand.gold_scale
