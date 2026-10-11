"""WP7: the M4 environment (plan WP7; design §14 invariants)."""

import json
import os

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame import progression as pg
from lanegame.champions import swap_seats
from lanegame.champions_m4 import M4State, check_invariants
from lanegame.env import (TRACE_VERSION_M4, LaneEnv, UniformAgent, feature_names, legal_set, observe, play_episode,
                          read_trace, write_trace)
from lanegame.env.agents_m4 import m4_suite
from lanegame.mechanics import total_hp

M4 = GameConfig.m4()


def test_observations_masks_and_mirror():
    env = LaneEnv(GameConfig.m4(skill_a=0.65, skill_b=0.4), max_turns=400)
    rng = np.random.default_rng(1)
    ag = UniformAgent()
    obs = env.reset(seed=2)
    for _ in range(400):
        for s in (0, 1):
            o = obs[s]
            assert len(o.vector) == len(feature_names(env.cfg)) == 88
            assert o.vector.min() >= 0 and o.vector.max() <= 1
            assert set(np.flatnonzero(o.mask)) == set(o.legal) == set(legal_set(o.state, s, env.cfg, o.champ))
            assert isinstance(o.champ, M4State)
        assert np.array_equal(obs[1].vector, observe(env.state.mirror(), env.k, 0, swap_seats(env.cfg),
                                                     env.champ.mirror()))
        r = env.step(ag.act(obs[0], rng), ag.act(obs[1], rng))
        if r.done:
            break
        obs = r.obs


def soak(cfg, turns, seed, agents=(None, None)):
    env = LaneEnv(cfg, max_turns=turns, cache_size=20_000)
    rng = np.random.default_rng(seed)
    ags = [a or UniformAgent() for a in agents]
    obs = env.reset(seed=seed)
    alive0 = total_hp(env.state, cfg)
    spawned, removed = [0, 0], [0, 0]
    prev = env.champ
    plates_paid = [0, 0]
    earned = [0, 0]
    total = 0
    xi, f = cfg.xp_per_hp, cfg.farm_per_hp
    while True:
        r = env.step(ags[0].act(obs[0], rng), ags[1].act(obs[1], rng))
        i = r.info
        # zero-sum and the exact reward decomposition (design §14.1, 14.2, 14.11)
        assert r.rewards[0] + r.rewards[1] == 0.0
        assert i["reward_units"] == sum(i[k][0] - i[k][1] for k in ("farm_units", "plate_units", "bounty_units"))
        assert r.rewards[0] == i["reward_units"] / cfg.gold_scale
        total += i["reward_units"]
        cur = env.champ
        check_invariants(cur, cfg)
        for s in (0, 1):
            a, b = prev.side(s), cur.side(s)
            assert b.xp >= a.xp and b.items >= a.items and b.tower_dmg >= a.tower_dmg          # §14.4
            kills = sum(i["kills"][s].values())
            assert i["xp_gain"][s] + i["xp_discarded"][s] <= xi * kills                      # §14.5
            assert i["farm"][s] <= f * kills + 1e-12
            plates_paid[s] += i["plate_units"][s]
            earned[s] += i["earned_units"][s]
            ev = i["events"]
            spawned[s] += ev["spawned_hp"][s]
            removed[s] += ev["killed_hp"][s] + ev["tower_hp"][s] + ev["overflow_hp"][s]
            assert ev["overflow_hp"][s] == 0
            alive = total_hp(env.state, cfg)[s]
            assert alive == alive0[s] + spawned[s] - removed[s]                              # minion conservation
            assert plates_paid[s] == cfg.gold_scale * pg.plates_broken(b.tower_dmg)          # plate gold = v pi(tau)
        prev = cur
        if r.done:
            break
        obs = r.obs
    assert total == earned[0] - earned[1]
    return env


def test_soak_10000_random_legal_turns():
    soak(M4, 10_000, 0)


def test_soak_scripted_suite_pairs():
    suite = m4_suite()
    names = list(suite)
    for j, (x, y) in enumerate(zip(names, names[1:] + names[:1])):
        soak(M4, 1500, 100 + j, (suite[x], suite[y]))


def test_common_random_numbers_replay_exactly():
    suite = m4_suite()
    e1, e2 = LaneEnv(M4, max_turns=300), LaneEnv(M4, max_turns=300, cache_size=10_000)
    l1 = play_episode(e1, suite["freezer"], suite["aggressive_trader"], 77, record=True)
    l2 = play_episode(e2, suite["freezer"], suite["aggressive_trader"], 77, record=True)
    assert l1.rewards == l2.rewards and l1.actions == l2.actions and l1.reward_units == l2.reward_units
    assert [t["champ_after"] for t in l1.turns] == [t["champ_after"] for t in l2.turns]
    assert [t["chance"] for t in l1.turns] == [t["chance"] for t in l2.turns]


def test_trace_version_2_round_trip(tmp_path):
    suite = m4_suite()
    env = LaneEnv(M4, max_turns=60)
    log = play_episode(env, suite["presser"], suite["gold_recaller"], 5, record=True)
    path = os.path.join(tmp_path, "t.json")
    write_trace(log, M4, path)
    d = read_trace(path)
    assert d["version"] == TRACE_VERSION_M4 == 2 and d["ruleset"] == "m4" and d["config_obj"] == M4
    assert d["champion_fields"] == ["pos", "hp", "absent", "xp", "items", "gold", "tower_dmg"]
    ep = d["episode"]
    assert ep["reward_units"] == log.reward_units and len(ep["turns"]) == 60 and "m4" in ep
    assert sum(ep["m4"]["xp_gain"]) > 0
    for key in ("levels", "xp", "items", "gold", "tower_dmg", "freeze_maintained", "purchases", "plates_broken"):
        assert key in ep["turns"][0]


def test_m4_env_rejects_legacy_champion_state_and_expected_gank_reward():
    from lanegame.champions import initial_champions
    env = LaneEnv(M4, max_turns=5)
    with pytest.raises(TypeError):
        env.reset(seed=0, champ=initial_champions(M4))
    with pytest.raises(ValueError):
        LaneEnv(M4, gank_reward="expected")
