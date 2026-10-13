"""M4.3b pilot tooling: the reactive reference agents, the behaviour features and situation masks, the state
dependence summary, the continuing-time attacker, the Recall / death state variants, viewer recordings, and a tiny
end-to-end training run of the driver. Skipped without PyTorch where a network is needed."""

import json
import os

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import CHAMP_ABSENT
from lanegame.champions_m4 import check_invariants
from lanegame.config import B_HOLD, B_RECALL, W_HOLD, champ_act
from lanegame.env import LaneEnv, play_episode
from lanegame.experiments.m4 import zoning_natural as zn
from lanegame.experiments.rl.m43b import behavior as bh
from lanegame.experiments.rl.m43b import common
from lanegame.experiments.rl.m43b.references import CURVE_REFS, HEAD_REFS, ReactiveAgent, reference_suite

CFG = GameConfig.m4_lasthit()
RECALL = champ_act(W_HOLD, B_RECALL)


# ---------------------------------------------------------------------------------------------- references
def test_reference_suite_plays_legal_deterministic_actions_and_matches_the_zoning_policies():
    suite = reference_suite(CFG)
    assert set(HEAD_REFS) <= set(suite) and set(CURVE_REFS) <= set(HEAD_REFS)
    env = LaneEnv(CFG, max_turns=120, cache_size=5000)
    zoning = {p.name: p for p in zn.POLICIES}
    for name in HEAD_REFS:
        x = suite[name]
        obs = env.reset(seed=11)
        for _ in range(120):
            a, b = x.act(obs[0], None), suite["farm"].act(obs[1], None)
            assert a in obs[0].legal and b in obs[1].legal
            d = x.distribution(obs[0]) if hasattr(x, "distribution") else None
            if isinstance(x, ReactiveAgent):
                assert d.sum() == 1.0 and obs[0].legal[int(np.argmax(d))] == a      # one-hot, consistent with act
                if name in zoning:
                    assert a == zoning[name].act(CFG, obs[0].state, obs[0].champ, 0)
            r = env.step(a, b)
            obs = r.obs
    # the suite is reproducible through play_episode (deterministic agents: equal payoffs on equal seeds)
    env = LaneEnv(CFG, max_turns=60, cache_size=5000)
    a1 = play_episode(env, suite["push_punish"], suite["clear_punish"], 5)
    a2 = play_episode(env, suite["push_punish"], suite["clear_punish"], 5)
    assert a1.payoff == a2.payoff and a1.actions == a2.actions


# ---------------------------------------------------------------------------------------------- features, rows
def test_turn_features_at_s0_and_after_a_death_variant():
    env = LaneEnv(CFG, max_turns=10)
    oa, ob = env.reset(seed=0)
    f = bh.turn_features(oa, CFG)
    assert f["pos"] == -CFG.tower and f["epos"] == CFG.tower and f["dist"] == 2 * CFG.tower
    assert f["recall_legal"] and not f["enemy_absent"] and not f["contact"] and f["power"] == 1.0
    assert f["gold"] == 0 and f["hp_frac"] == 1.0 and f["level"] == 1 and f["n_legal"] == len(oa.legal)
    fb = bh.turn_features(ob, CFG)
    assert fb["pos"] == f["pos"] and fb["epos"] == f["epos"]                      # own perspective of each seat
    p = bh.category_masses(bh.full_distribution(reference_suite(CFG)["farm"], oa))
    assert p["wave"][0] == 1.0 and p["beh"][B_HOLD] == 1.0 and p["entropy"] == 0.0


def test_play_rows_pairing_return_to_go_and_situation_masks():
    suite = reference_suite(CFG)
    R = bh.play_rows(CFG, suite["push_punish"], suite["farm"], 2, 40, 7)
    cnt = R.pop("_counters")
    n = len(R["k"])
    assert n > 0 and set(np.unique(R["seat"])) == {0, 1} and len(cnt) == 4           # both seats, 2 games each
    # the return to go follows the recursion g_t = r_t + gamma g_{t+1} along a game (present turns only, so check
    # consecutive present turns)
    for seat in (0, 1):
        for ep in (0, 1):
            m = (R["seat"] == seat) & (R["ep"] == ep)
            turns, g, r = R["turn"][m], R["ret_to_go"][m], R["reward"][m]
            for i in range(len(turns) - 1):
                if turns[i + 1] == turns[i] + 1:
                    assert g[i] == pytest.approx(r[i] + CFG.gamma * g[i + 1])
    S = bh.situations(R)
    assert all(len(v) == n for v in S.values()) and S["all present turns"].all()
    assert S["Recall legal"].sum() == R["recall_legal"].sum()
    tab = bh.conditional_tables(R)
    assert tab["all present turns"]["n"] == n and abs(sum(tab["all present turns"][f"chose_{w}"] for w in bh.WAVES) - 1) < 1e-9
    # a deterministic policy has zero entropy everywhere and its wave choice is a function of the state
    assert tab["all present turns"]["entropy"] == 0.0
    sd = bh.state_dependence(R)
    assert sd["wave"]["H_action_given_state"] <= sd["wave"]["H_action"] + 1e-12
    # the self-play profile of a constant policy: Hold everywhere, no state dependence of the wave choice
    prof = bh.profile(CFG, suite["farm"], 2, 40, 3)
    assert prof["waves"]["Hold"] == 1.0 and prof["state_dependence"]["wave"]["mutual_information"] == 0.0


def test_candidate_set_is_legal_and_recall_only_where_legal():
    env = LaneEnv(CFG, max_turns=5)
    oa, _ = env.reset(seed=0)
    c = bh.candidate_set(CFG, oa.legal)
    assert "Recall" in c and "policy" in c and all(x.action in oa.legal for n, x in c.items() if n != "policy")
    legal_no_recall = tuple(a for a in oa.legal if a != RECALL)
    assert "Recall" not in bh.candidate_set(CFG, legal_no_recall)


# ---------------------------------------------------------------------------------------------- recall variants
def test_dead_and_recalled_state_variants_are_valid_and_return_on_schedule():
    from lanegame.diagnostics import FollowPolicy, ProbCache, Snapshot, rollout
    suite = reference_suite(CFG)
    env = LaneEnv(CFG, max_turns=60, cache_size=5000)
    log = play_episode(env, suite["push_punish"], suite["push_punish"], 4, record=True)
    info = next(i for i in log.turns if i["k"] >= 10 and i["champ_before"][0][2] == 0 and i["champ_before"][1][2] == 0)
    snap = Snapshot.from_info(info, (CFG.skill_a, CFG.skill_b))
    T = CFG.tower
    dead = snap.with_side(0, pos=-T, hp=0, absent=CFG.death_turns)
    rec = snap.with_side(0, pos=-T, absent=CFG.recall_turns - 1)
    for v in (snap, dead, rec):
        check_invariants(v.champ, CFG)
    agents = (suite["farm"], suite["farm"])
    cache = ProbCache()
    r_dead = rollout(env, dead, 0, FollowPolicy(), agents, 1, cache, record=True, max_turns=12)
    r_rec = rollout(env, rec, 0, FollowPolicy(), agents, 1, cache, record=True, max_turns=12)
    # the dead champion is absent for death_turns turns and returns next turn when its timer reaches 0;
    # the recalled one is absent for one more turn
    assert [i["returned"][0] for i in r_dead.infos[:CFG.death_turns]] == [False] * (CFG.death_turns - 1) + [True]
    assert [i["actions"][0] for i in r_dead.infos[:CFG.death_turns]] == [CHAMP_ABSENT] * CFG.death_turns
    assert r_rec.infos[0]["returned"][0] and r_rec.infos[0]["actions"][0] == CHAMP_ABSENT
    assert r_rec.infos[1]["actions"][0] != CHAMP_ABSENT and r_dead.infos[CFG.death_turns]["actions"][0] != CHAMP_ABSENT
    # the lane state is shared: the variants differ only in the focal champion's record
    assert dead.lane == snap.lane == rec.lane and dead.champ.side(1) == snap.champ.side(1)


def test_recall_masks_require_legality():
    from lanegame.experiments.rl.m43b.recall import RECALL_CLASSES, recall_masks
    suite = reference_suite(CFG)
    R = bh.play_rows(CFG, suite["push_punish"], suite["farm"], 1, 40, 9)
    R.pop("_counters")
    S = bh.situations(R)
    masks = recall_masks()
    assert set(masks) == set(RECALL_CLASSES)
    for cls, fn in masks.items():
        m = fn(R, S)
        assert len(m) == len(R["k"]) and not (m & ~R["recall_legal"]).any()


# ---------------------------------------------------------------------------------------------- viewer recordings
def test_exported_recording_replays_identically_in_the_viewer_session(tmp_path, monkeypatch):
    from lanegame.experiments.rl.m43b import trajectories as tj
    from lanegame.viewer import session as ss
    monkeypatch.setattr(tj, "TRAJ_DIR", str(tmp_path))
    suite = reference_suite(CFG)
    rec = tj.export("t", "ref:push_punish", "ref:clear_punish", 12345, 30, "test")
    setup, actions, meta = ss.load_recording(rec["path"])
    assert setup.seed == 12345 and len(actions) == 30 and meta["names"] == ["ref:push_punish", "ref:clear_punish"]
    rp = ss.play(setup, prefix=actions, prefix_note="recorded")
    assert rp.n == 30 and all(t.notes == ("recorded", "recorded") for t in rp.turns)
    # the replay's rewards equal the recorded game's payoff, turn by turn (same chance seed, same actions)
    env = LaneEnv(CFG, max_turns=30)
    obs = env.reset(seed=12345)
    total = 0.0
    for j, (a, b) in enumerate(actions):
        r = env.step(a, b)
        total += r.rewards[0]
        assert r.info["reward_units"] == rp.turns[j].info["reward_units"]
        assert r.info["actions"] == tuple(actions[j])
    assert total == pytest.approx(rec["payoff_A"])


# ---------------------------------------------------------------------------------------------- torch-dependent
torch = pytest.importorskip("torch")


def test_continuing_time_attacker_trains_both_seats_against_a_frozen_target():
    from lanegame.rl import NetAgent, OpponentSampler, PPOConfig, PolicyValueNet, train_ppo
    env = LaneEnv(CFG)
    torch.manual_seed(1)
    target = NetAgent(PolicyValueNet(env.obs_size, env.n_actions, (32, 32)), "target")
    frozen = {k: v.clone() for k, v in target.net.state_dict().items()}
    net = PolicyValueNet(env.obs_size, env.n_actions, (32, 32))
    net.load_state_dict(target.net.state_dict())                           # warm start at parity
    ppo = PPOConfig.continuing(CFG, total_steps=2 * 4 * 24, n_envs=4, segment_length=24, game_length=40,
                               minibatch_size=48, epochs=1, seed=5, hidden=(32, 32), lane_cache=2000)
    res = train_ppo(CFG, ppo, OpponentSampler([target]), net=net, verbose=False)
    assert len(res.history) == 2 and all(torch.equal(frozen[k], v) for k, v in target.net.state_dict().items())
    seats = {e["seat"] for e in res.episodes}
    assert seats <= {0, 1} and all(e["opponent"] == "target" for e in res.episodes)
    assert res.history[-1]["terminations"] == 0                             # continuing time: truncations only
    assert any(p.requires_grad for p in res.net.parameters())


def test_driver_end_to_end_tiny(tmp_path, monkeypatch):
    """One update of the baseline condition through the driver's worker, then the curve points and the agent specs."""
    from lanegame.experiments.rl.m43b.train import health_checks, train_one
    monkeypatch.setattr(common, "CKPT", str(tmp_path / "ckpt"))
    monkeypatch.setattr(common, "OUT", str(tmp_path / "out"))
    steps = common.N_ENVS * 128 * 2                                          # exactly one update
    s = train_one(("base", 0, steps))
    assert s["agent_transitions"] == steps and s["finite_losses_and_gradients"] and s["terminations"] == 0
    assert os.path.exists(common.final_path("base", 0)) and common.snapshot_paths("base", 0) == []
    pts = common.curve_points("base", 0)
    assert pts == [(steps, common.final_path("base", 0))]
    checks = health_checks({s["name"]: s})
    assert set(checks["checks"]) and isinstance(checks["all_passed"], bool)
    x = common.agent_from_spec("final:base:0")
    y = common.agent_from_spec("initial:0")
    o = LaneEnv(CFG, max_turns=2).reset(seed=0)[0]
    assert x.probabilities(o).sum() == pytest.approx(1.0) and y.probabilities(o).sum() == pytest.approx(1.0)
    run = json.load(open(os.path.join(common.run_dir("base", 0), "run.json"), encoding="utf-8"))
    assert run["ppo"]["ent_coef"] == 0.01 and run["shared"]["current_fraction"] == 1.0


def test_archive_states_are_witnessed_mirror_closed_and_usable_as_training_starts(tmp_path):
    from lanegame.env.starts import load_archive, mirror_closed, verify_witness
    from lanegame.experiments.rl.m43b import archive as ar
    from lanegame.rl import PPOConfig, SharedConfig, train_shared
    path = str(tmp_path / "archive.json")
    info = ar.build_archive(turns=100, targets={"neutral": 2, "crash_imminent": 2, "enemy_pushing": 2, "other": 2},
                            path=path, verify=True)
    states = load_archive(path, CFG)
    assert info["states"] == len(states) >= 4 and mirror_closed(states)
    assert all(verify_witness(CFG, s) for s in states)
    assert all(12 <= len(s.witness["actions"]) <= 100 for s in states)
    ppo = PPOConfig.continuing(CFG, total_steps=2 * 4 * 16, n_envs=4, segment_length=16, game_length=20,
                               minibatch_size=32, epochs=1, seed=2, hidden=(32, 32), lane_cache=2000,
                               start_archive=path, start_mix=0.5)
    res = train_shared(CFG, ppo, SharedConfig(current_fraction=1.0, snapshot_every=10_000), verbose=False)
    assert res.history[-1]["rollout_terminations"] == 0


def test_plan_definitions_are_legal_and_bounded():
    from lanegame.experiments.rl.m43b.plans import PLAN_SITUATIONS, make_plan
    env = LaneEnv(CFG, max_turns=10)
    oa, ob = env.reset(seed=0)
    names = sorted({p for ps in PLAN_SITUATIONS.values() for p in ps})
    for name in names:
        plan = make_plan(name)
        for t in range(0, 25):
            a = plan.act(oa, t)
            assert a is None or a in oa.legal
        assert plan.act(oa, 30) is None                                   # every plan ends
    assert make_plan("slow_push").act(oa, 0) == champ_act(2, B_HOLD) and make_plan("slow_push").act(oa, 2) == champ_act(W_HOLD, B_HOLD)
    assert make_plan("shove_away").act(oa, 0) is None                     # the enemy is present at s0
    # an absent champion hands control back
    from lanegame.env import Observation
    absent = Observation(0, 0, oa.state, (CHAMP_ABSENT,), oa.vector, oa.mask, oa.champ, CFG)
    assert all(make_plan(n).act(absent, 0) is None for n in names)
