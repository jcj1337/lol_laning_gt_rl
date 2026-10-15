"""M4.5c: the shared-policy PPO pipeline on the calibrated M4.5 environment, GameConfig.m45_calibrated().

The preset is frozen (equal to m45_calibrated field by field, with the M4.5b decisions retained); current-policy games
train both seats with mirror observations and exactly antisymmetric rewards; the masks are the engine's legal sets, the
Ward code is offered and never-legal codes never are; recorded log-probabilities are the masked distribution's; both
seats carry policy gradient; the two opt-in trainer arguments (`stop_at`, `rollout_stats`) leave training unchanged;
the checkpoint guard refuses M4.4 networks and the out-of-distribution adapter is explicit; the M4.5 action decoding
(Ward its own category) and the records' Ward rule and Ward mass are exact; the fixed opponents are legal; the pilot
specification is consistent and the full campaign is gated. Skipped without PyTorch."""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lanegame import GameConfig                                                          # noqa: E402
from lanegame.champions import champion_legal_actions                                    # noqa: E402
from lanegame.config import CHAMP_ABSENT                                                  # noqa: E402
from lanegame.env import LaneEnv                                                         # noqa: E402
from lanegame.env.agents import UniformAgent, episode_streams                            # noqa: E402
from lanegame.rl import (MatchupSampler, MaskedCategorical, NetAgent, PolicyValueNet, PPOConfig,   # noqa: E402
                         SharedCollector, SharedConfig, SnapshotPool, checkpoint_info, ppo_update, save_checkpoint,
                         train_shared, weights_digest)
from lanegame.rl.shared import rollout_batch                                             # noqa: E402
from lanegame.experiments.rl.m45c import actions as AC                                   # noqa: E402
from lanegame.experiments.rl.m45c import common as C                                     # noqa: E402

CFG = GameConfig.m45_calibrated()
ENV = LaneEnv(CFG)
OBS, NACT = ENV.obs_size, ENV.n_actions


def ppo(**kw):
    base = dict(total_steps=800, n_envs=4, segment_length=24, game_length=40, minibatch_size=64, epochs=1, seed=3,
                hidden=(32, 32), lane_cache=2000)
    base.update(kw)
    return PPOConfig.continuing(CFG, **base)


def net(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet(OBS, NACT, (32, 32))


def collect(seed=1, n=None, **kw):
    col = SharedCollector(CFG, ppo(**kw), MatchupSampler(SnapshotPool(4), 1.0), seed=seed)
    n = n or net()
    roll, finished, counts = col.collect(n, torch.Generator().manual_seed(0))
    return col, n, roll, finished, counts


# ---------------------------------------------------------------------------------------------- the environment
def test_the_environment_is_the_frozen_calibrated_preset():
    cfg = C.env_cfg()
    assert cfg == GameConfig.m45_calibrated()
    d = cfg.to_dict()
    assert d["farm_spot_rule"] == "last_front" and d["minion_retaliation_delay"] == 0
    assert d["minion_damage_growth"] == 1.08 and d["xp_range"] == 2 and d["vision_rule"] == "ward"
    assert (OBS, NACT) == (90, 61) and C.ppo_config(0, 4096).gamma == cfg.gamma == 0.99
    AC.check_space(cfg)


# ---------------------------------------------------------------------------------------------- shared self-play
def test_current_policy_games_train_both_seats_with_mirror_observations_and_antisymmetric_rewards():
    from lanegame.env.observation_m4 import decode_m4
    _, _, roll, _, counts = collect()
    T, N = 24, 4
    assert roll["active"].all() and counts["transitions_seat_a"] == counts["transitions_seat_b"] == T * N
    assert counts["agent_transitions"] == 2 * counts["env_turns"] and counts["terminations"] == 0
    assert np.array_equal(roll["rewards"][:, 0::2], -roll["rewards"][:, 1::2])
    for t in range(T):
        for i in range(N):
            pa, la, ca, _ = decode_m4(roll["obs"][t, 2 * i], CFG)
            pb, lb, cb, _ = decode_m4(roll["obs"][t, 2 * i + 1], CFG)
            assert pa == pb and lb == la.mirror() and cb == ca.mirror()


def test_truncations_bootstrap_from_the_pre_reset_observation_and_never_terminate():
    from lanegame.rl import compute_gae
    cfg = ppo(segment_length=30, game_length=12)
    col = SharedCollector(CFG, cfg, MatchupSampler(SnapshotPool(4), 1.0), seed=2)
    first = list(col.times.first_lengths)
    n = net()
    roll, finished, counts = col.collect(n, torch.Generator().manual_seed(0))
    assert not roll["terminated"].any() and counts["truncations"] == len(finished)
    for i in range(4):
        want, t = [], first[i] - 1
        while t < 30:
            want.append(t)
            t += 12
        assert list(np.flatnonzero(roll["truncated"][:, 2 * i])) == want
    # replay environment 0's first game from the collector's episode stream; the recorded bootstrap is the critic's
    # value of that game's final (pre-reset) observation, for both seats
    root = np.random.SeedSequence(2)
    root.spawn(1)
    _, env_ss, _, _ = episode_streams(root.spawn(1)[0].spawn(1)[0])
    e = LaneEnv(CFG, max_turns=first[0])
    obs = e.reset(seed=env_ss)
    for t in range(first[0]):
        r = e.step(int(roll["actions"][t, 0]), int(roll["actions"][t, 1]))
        assert r.rewards[0] * cfg.reward_scale == roll["rewards"][t, 0]
        obs = r.obs
    assert r.truncated and not r.terminated
    with torch.no_grad():
        want = [float(n.value(torch.as_tensor(obs[s].vector).unsqueeze(0))[0]) for s in (0, 1)]
    t0 = first[0] - 1
    assert roll["next_values"][t0, 0] == pytest.approx(want[0], abs=1e-6)
    assert roll["next_values"][t0, 1] == pytest.approx(want[1], abs=1e-6)
    adv, _ = compute_gae(roll["rewards"], roll["values"], roll["next_values"], roll["terminated"], roll["truncated"],
                         cfg.gamma, cfg.gae_lambda)
    assert adv[t0, 0] == pytest.approx(roll["rewards"][t0, 0] + cfg.gamma * roll["next_values"][t0, 0]
                                       - roll["values"][t0, 0])


def test_masks_are_legal_sets_offer_ward_and_never_a_never_legal_code():
    _, _, roll, _, _ = collect(segment_length=60, game_length=200)
    a = roll["active"]
    masks, acts = roll["mask"][a], roll["actions"][a]
    assert masks[np.arange(len(acts)), acts].all()                       # every sampled code was legal
    assert not masks[:, sorted(AC.NEVER_LEGAL)].any()
    assert masks[:, AC.WARD].any() and masks[:, AC.RECALL].any()
    assert set(np.flatnonzero(masks.any(0))) <= AC.EVER_LEGAL


def test_recorded_log_probabilities_are_the_masked_sampling_distribution():
    _, n, roll, _, _ = collect()
    a = roll["active"]
    obs, mask, acts, logp = (torch.as_tensor(roll["obs"][a]), torch.as_tensor(roll["mask"][a]),
                             torch.as_tensor(roll["actions"][a]), roll["logprobs"][a])
    with torch.no_grad():
        logits, _ = n(obs)
        lp = MaskedCategorical(logits, mask).log_prob(acts).numpy()
        # an independent float64 reference: softmax over the legal logits only
        ref = []
        for row, m, x in zip(logits.double().numpy(), mask.numpy(), acts.numpy()):
            z = row[m] - row[m].max()
            ref.append(z[list(np.flatnonzero(m)).index(x)] - np.log(np.exp(z).sum()))
    assert np.allclose(lp, logp, atol=1e-6) and np.allclose(lp, np.asarray(ref), atol=1e-5)


def test_both_seats_carry_policy_gradient():
    col, n, roll, _, _ = collect()
    batch, _, _ = rollout_batch(roll, col.cfg)
    seat = np.concatenate([np.arange(roll["active"].shape[1])[None, :].repeat(roll["active"].shape[0], 0)[roll["active"]]])
    norms = []
    for s in (0, 1):
        idx = torch.as_tensor(np.flatnonzero(seat % 2 == s))
        sub = {k: v[idx] for k, v in batch.items()}
        m = PolicyValueNet(OBS, NACT, (32, 32))
        m.load_state_dict(n.state_dict())
        logits, _ = m(sub["obs"])
        dist = MaskedCategorical(logits, sub["mask"])
        ratio = (dist.log_prob(sub["actions"]) - sub["logprobs"]).exp()
        adv = (sub["advantages"] - batch["advantages"].mean()) / (batch["advantages"].std() + 1e-8)
        loss = -(adv * ratio * sub["choice"]).sum() / sub["choice"].sum()
        loss.backward()
        norms.append(float(torch.norm(m.policy_head.weight.grad)))
        assert float(sub["choice"].sum()) > 0 and float(sub["advantages"].std()) > 0
    assert min(norms) > 1e-6


# ---------------------------------------------------------------------------------------------- opt-in trainer arguments
def _strip(h):
    return [{k: v for k, v in e.items() if k not in ("elapsed", "sps", "collect_s", "update_s")} for e in h]


def test_rollout_stats_hook_reports_and_does_not_change_training():
    from lanegame.experiments.rl.m45c.train import RolloutStats
    a = train_shared(CFG, ppo(total_steps=576), SharedConfig(current_fraction=1.0), verbose=False)
    b = train_shared(CFG, ppo(total_steps=576), SharedConfig(current_fraction=1.0), verbose=False,
                     rollout_stats=RolloutStats(ppo()))
    assert weights_digest(a.net.state_dict()) == weights_digest(b.net.state_dict())
    common = [{k: v for k, v in e.items() if k in ea} for e, ea in zip(_strip(b.history), _strip(a.history))]
    assert json.dumps(common) == json.dumps(_strip(a.history))
    h = b.history[-1]
    assert h["illegal_sampled"] == 0 and h["never_legal_offered"] == 0
    assert h["reward_sum_A"] == -h["reward_sum_B"] and h["adv_std_A"] > 0 and h["adv_std_B"] > 0
    assert sum(h[f"act_cat_{c}"] for c in AC.CATEGORIES) == pytest.approx(1.0)


def test_lane_cache_size_changes_speed_and_memory_never_results():
    """The full-campaign plan may shrink PPOConfig.lane_cache to fit more concurrent seeds in memory: the lane-step
    cache is an exact memo, so training is bit-identical with any size (0 disables it)."""
    digests, hists = [], []
    for size in (0, 7, 2000):
        r = train_shared(CFG, ppo(total_steps=576, lane_cache=size), SharedConfig(current_fraction=1.0), verbose=False)
        digests.append(weights_digest(r.net.state_dict()))
        hists.append(json.dumps(_strip(r.history)))
    assert len(set(digests)) == 1 and len(set(hists)) == 1


def test_stop_at_is_an_exact_prefix_of_the_full_schedule(tmp_path):
    scfg = SharedConfig(current_fraction=1.0, snapshot_every=576)
    full = train_shared(CFG, ppo(total_steps=1728), scfg, out_dir=str(tmp_path / "full"), verbose=False)
    pre = train_shared(CFG, ppo(total_steps=1728), scfg, out_dir=str(tmp_path / "pre"), verbose=False, stop_at=576)
    assert pre.agent_transitions == 576 and full.agent_transitions == 1728
    assert json.dumps(_strip(pre.history)) == json.dumps(_strip(full.history[:len(pre.history)]))
    snap = [p for p in full.snapshot_paths if p.endswith("snap_000000576.pt")][0]
    assert checkpoint_info(snap)["digest"] == weights_digest(pre.net.state_dict())
    assert pre.history[-1]["lr"] > 0.0                                    # the prefix did not anneal to zero
    meta = checkpoint_info(str(tmp_path / "pre" / "final.pt"))["metadata"]
    assert meta["stop_at"] == 576 and meta["ppo"]["total_steps"] == 1728
    with pytest.raises(ValueError):
        train_shared(CFG, ppo(total_steps=576), scfg, verbose=False, stop_at=1000)


# ---------------------------------------------------------------------------------------------- checkpoints
def test_m44_networks_are_refused_and_the_ood_adapter_is_explicit(tmp_path):
    from lanegame.experiments.rl.m44.common import load_policy
    aggro = GameConfig.m4_four_actions_econ_aggro()
    old = PolicyValueNet(LaneEnv(aggro).obs_size, 61, (32, 32))
    p = str(tmp_path / "m44.pt")
    save_checkpoint(p, old, aggro, {"name": "m44-like"})
    with pytest.raises(ValueError, match="inputs"):
        load_policy(p, CFG)
    src = load_policy(p, aggro)
    ood = C.OODNetAgent(src, CFG, "ood:test")
    assert ood.ood and ood.drop == [61, 78]
    o = ENV.reset(seed=3)[0]
    q = ood.probabilities(o)
    assert q[list(o.legal)].sum() == pytest.approx(1.0) and q[np.flatnonzero(~o.mask)].sum() == 0
    # a network trained on the same shapes but another configuration is refused without the explicit opt-in
    q45 = str(tmp_path / "unified.pt")
    save_checkpoint(q45, PolicyValueNet(OBS, NACT, (32, 32)), GameConfig.m45_unified(), {})
    with pytest.raises(ValueError, match="cross_env"):
        load_policy(q45, CFG)


def test_initial_agent_is_train_shareds_initialization():
    res = train_shared(CFG, ppo(total_steps=192, hidden=C.HIDDEN, seed=7), SharedConfig(current_fraction=1.0),
                       verbose=False, stop_at=192)
    torch.manual_seed(7)
    want = PolicyValueNet(OBS, NACT, C.HIDDEN)
    assert weights_digest(C.initial_agent(7).net.state_dict()) == weights_digest(want.state_dict())
    assert res.agent_transitions == 192


# ---------------------------------------------------------------------------------------------- decoding and records
def test_m45_action_categories():
    from lanegame.experiments.rl.m44.actions import decode
    assert len(AC.EVER_LEGAL) == 27 and len(AC.NEVER_LEGAL) == 34
    assert AC.category(30) == "Ward" and decode(30).label() == "Last Hit + Hold + Vision"   # why Ward is separate
    assert AC.category(12) == "Recall" and AC.category(45) == "Absent"
    assert {AC.category(a) for a in AC.EVER_LEGAL} == set(AC.CATEGORIES)
    h = AC.histogram(np.array(sorted(AC.EVER_LEGAL)))
    assert sum(h[f"cat_{c}"] for c in AC.CATEGORIES) == pytest.approx(1.0)
    assert sum(AC.PRESSURE_OF[a] for a in AC.EVER_LEGAL) == 8


def test_record_ward_rule_and_ward_mass_are_exact():
    from lanegame.experiments.rl.m44.actions import masses
    agent = NetAgent(net(5), "p")
    e = LaneEnv(CFG, max_turns=150)
    _, ss, ra, rb = episode_streams(np.random.SeedSequence(92_901))
    obs = e.reset(seed=ss)
    u = UniformAgent()
    checked = 0
    while True:
        for s in (0, 1):
            o = obs[s]
            me = o.champ.side(s)
            rule = me.absent == 0 and (abs(me.pos) <= 1 or me.pos == 99)
            assert rule == (AC.WARD in champion_legal_actions(o.state, o.champ, s, CFG))
            if o.legal != (CHAMP_ABSENT,):
                p = agent.probabilities(o)
                assert masses(p)["obj"][2] == pytest.approx(p[AC.WARD], abs=1e-12)     # P(Ward) = po2
                checked += 1
        r = e.step(u.act(obs[0], ra), u.act(obs[1], rb))
        if r.done:
            break
        obs = r.obs
    assert checked > 100


def test_fixed_opponents_are_legal_from_both_seats():
    from lanegame.experiments.rl.m45c.references import SUITE, reference
    e = LaneEnv(CFG, max_turns=120)
    for name in SUITE:
        for seat in (0, 1):
            x, u = reference(name), UniformAgent()
            _, ss, ra, rb = episode_streams(np.random.SeedSequence(92_902, spawn_key=(seat,)))
            obs = e.reset(seed=ss)
            while True:
                a, b = x.act(obs[seat]), u.act(obs[1 - seat], rb)
                r = e.step(a, b) if seat == 0 else e.step(b, a)        # the environment raises on an illegal code
                x.notify(r.info, seat)
                if r.done:
                    break
                obs = r.obs


# ---------------------------------------------------------------------------------------------- specification
def test_pilot_specification_and_seed_ranges():
    from lanegame.experiments.rl.m45c import evaluate as EV
    assert C.TRANSITIONS_PER_UPDATE == 4096 and C.PILOT_STOP % C.TRANSITIONS_PER_UPDATE == 0
    assert C.PILOT_STOP < C.PILOT_SCHEDULE and C.PILOT_STOP // C.TRANSITIONS_PER_UPDATE == 250
    assert C.PILOT_SEED not in C.CAMPAIGN_SEEDS
    sched = C.snapshot_schedule(C.PILOT_STOP)
    assert sched[:4] == [28_672, 53_248, 77_824, 102_400] and sched[-1] == C.PILOT_STOP and len(sched) == 40
    for pts in (EV.CURVE_POINTS, EV.HIST_POINTS):
        assert set(pts) <= set(sched)
    used = [C.CURVE_SEED, C.FINAL_SEED, C.HIST_SEED, C.BEHAVIOR_SEED, C.SMOKE_EVAL_SEED, *C.LEDGER_SEEDS]
    assert all(92_000 <= x < 93_000 for x in used)
    assert not any(C.FULL_RESERVED[0] <= x <= C.FULL_RESERVED[1] for x in used)
    spec = C.training_spec()
    assert spec["network"]["obs_size"] == 90 and spec["accounting"]["gradient_steps_per_update"] == 64


def test_full_campaign_is_gated():
    from lanegame.experiments.rl.m45c.__main__ import main
    with pytest.raises(SystemExit, match="NOT part of the pilot"):
        main(["train_full"])
