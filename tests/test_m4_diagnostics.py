"""WP9: counterfactual branching and the audit on the M4 ruleset (plan WP9)."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import N_CHANCE
from lanegame.champions_m4 import M4State
from lanegame.config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, F, SH, SP, champ_act
from lanegame.diagnostics import (Commit, FollowPolicy, ProbCache, Snapshot, audit_games, branch, observation, restore,
                                  rollout, rollout_seed, summarize)
from lanegame.env import LaneEnv, UniformAgent, play_episode
from lanegame.env.agents_m4 import m4_suite

M4 = GameConfig.m4()
SUITE = m4_suite()
FARMER, TRADER = SUITE["hold_farmer"], SUITE["aggressive_trader"]


class _ReplayRNG:
    def __init__(self, draws):
        self.draws = [np.asarray(d) for d in draws]

    def random(self, n=None):
        assert n == N_CHANCE
        return self.draws.pop(0)


def mid_snapshot(turns=120, seed=3):
    env = LaneEnv(M4, max_turns=turns)
    log = play_episode(env, SUITE["freezer"], TRADER, seed, record=True)
    return Snapshot.from_info(log.turns[60], (M4.skill_a, M4.skill_b)), log


def test_snapshot_carries_the_m4_state_and_replays_exactly():
    snap, log = mid_snapshot()
    assert isinstance(snap.champ, M4State)
    assert Snapshot.from_dict(snap.to_dict()) == snap and snap.mirror().mirror() == snap
    env = LaneEnv(M4, max_turns=60)
    restore(env, snap, seed=0)
    assert env.state == snap.lane and env.champ == snap.champ and env.k % 4 == snap.k % 4
    env._rng = _ReplayRNG([info["chance"] for info in log.turns[60:]])
    for info in log.turns[60:]:
        assert env.state.to_list() == info["state_before"] and env.champ.to_list() == info["champ_before"]
        r = env.step(*info["actions"])
        assert r.info["reward_units"] == info["reward_units"] and r.info["champ_after"] == info["champ_after"]
    assert env.done


def test_continuing_rollouts_need_a_window_and_are_discounted():
    env = LaneEnv(M4)
    snap = mid_snapshot()[0]
    with pytest.raises(ValueError, match="window"):
        rollout(env, snap, 0, FollowPolicy(), (FARMER, TRADER), 0)
    r = rollout(env, snap, 0, FollowPolicy(), (FARMER, TRADER), 0, record=True, max_turns=50)
    g = M4.gamma
    assert r.turns == 50
    assert r.ret == pytest.approx(sum(g ** t * i["reward"] for t, i in enumerate(r.infos)), abs=1e-12)
    assert r.undiscounted == pytest.approx(sum(i["reward"] for i in r.infos), abs=1e-12)
    assert r.m4 is not None and set(r.m4) >= {"xp", "plates", "purchases", "tower_damage"}
    rb = rollout(env, snap, 0, FollowPolicy(), (FARMER, TRADER), 0, max_turns=50, bootstrap=lambda o: 10.0)
    assert rb.bootstrap == pytest.approx(g ** 50 * 10.0) and rb.ret == pytest.approx(r.ret + rb.bootstrap)


def test_rollout_from_s0_reproduces_play_episode_discounted():
    env = LaneEnv(M4, max_turns=80)
    for seed in range(4):
        log = play_episode(env, UniformAgent(), TRADER, seed, record=True)
        snap = Snapshot.from_info(log.turns[0], (M4.skill_a, M4.skill_b))
        r = rollout(env, snap, 0, FollowPolicy(), (UniformAgent(), TRADER), seed, ProbCache(), max_turns=80)
        assert r.ret == pytest.approx(sum(M4.gamma ** t * x for t, x in enumerate(log.rewards)), abs=1e-9)


def test_paired_candidates_share_chance_streams():
    snap, _ = mid_snapshot()
    legal = observation(M4, snap, 0).legal
    cands = {f"a{a}": Commit(a) for a in legal[:3]}
    draws = {}
    env = LaneEnv(M4, max_turns=30)
    for name, plan in cands.items():
        restore_draws = []
        orig_reset = env.reset

        def reset(*a, **k):
            obs = orig_reset(*a, **k)
            inner = env._rng

            class _S:
                def random(self, n=None):
                    d = inner.random(n)
                    restore_draws.append(np.array(d))
                    return d
            env._rng = _S()
            return obs
        env.reset = reset
        rollout(env, snap, 0, plan, (FARMER, TRADER), rollout_seed(5, 0), ProbCache(), max_turns=30)
        env.reset = orig_reset
        draws[name] = restore_draws
    first = next(iter(draws.values()))
    for d in draws.values():
        assert len(d) == len(first) and all(np.array_equal(x, y) for x, y in zip(d, first))


def test_branch_results_are_reproducible_and_labelled():
    snap, _ = mid_snapshot()
    legal = observation(M4, snap, 0).legal
    cands = {"policy": FollowPolicy(), "first": Commit(legal[0])}
    r1 = branch(M4, snap, 0, cands, (FARMER, TRADER), 6, root=11, max_turns=40)
    r2 = branch(M4, snap, 0, cands, (FARMER, TRADER), 6, root=11, max_turns=40)
    assert all(np.array_equal(r1.returns[c], r2.returns[c]) for c in cands)
    assert r1.meta["gamma"] == 0.99 and not r1.meta["bootstrapped"] and "xp" in r1.m4["policy"]
    r3 = branch(M4, snap, 0, cands, (FARMER, TRADER), 2, root=11, max_turns=40, bootstrap=lambda o: 0.0)
    assert r3.meta["bootstrapped"] and "APPROXIMATION" in r3.meta["label"]


def test_audit_on_m4_uses_discounted_returns_and_phase():
    with pytest.raises(ValueError, match="window"):
        audit_games(M4, FARMER, TRADER, 2, 0)
    t = audit_games(M4, SUITE["freezer"], TRADER, 3, 0, window=60)
    assert len(t) == 2 * 3 * 60
    assert set(np.unique(t["phase"])) <= {0, 1, 2, 3} and np.all(t["turns_left"] == -1)
    one = t.where((t["seat"] == 0) & (t["ep"] == 0))
    g = 0.0
    for i in range(len(one) - 1, -1, -1):
        g = one["reward"][i] + 0.99 * g
        assert one["ret_to_go"][i] == pytest.approx(g, abs=1e-9)
    assert {"level", "items", "gold", "xp", "tower_dmg"} <= set(t.cols)
    s = summarize(t)
    assert s["states"]["games"] == 6 and 0.99 < sum(s["states"]["hp_hist"].values()) <= 1.0 + 1e-9


def test_with_side_keeps_the_m4_state_type():
    snap, _ = mid_snapshot()
    s2 = snap.with_side(0, items=2, gold=0)
    assert isinstance(s2.champ, M4State) and s2.champ.a.items == 2 and s2.champ.b == snap.champ.b
    env = LaneEnv(M4, max_turns=5)
    restore(env, s2, 0)
    assert env.champ == s2.champ
