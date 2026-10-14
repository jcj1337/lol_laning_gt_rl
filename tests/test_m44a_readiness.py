"""M4.4a: PPO readiness of the four-action game (docs/m4/m4_4a_ppo_readiness.md).

The three environments (m4_four_actions / _econ / _econ_aggro) and their identical training settings; four-action
decoding, the codes that can ever be legal, masks; shared-policy self-play on every environment (own-perspective
observations, both seats trained, antisymmetric rewards, legal sampling, continuing-time semantics, reproducibility,
the paired initialization across environments); the checkpoint compatibility guard (46 vs 61 outputs, cross-preset
loads); the game recorder (equal to evaluate_discounted's game, wave classification); state-based freeze detection and
its episodes; dives, vision and plate phases; component interventions (legality, rules); paired counterfactuals from
identical states (replay, stream alignment, opponent protocols); consistent metric reports across environments; the
viewer round trip of a four-action recording; the M4.4b launch guard. No real training: every PPO run is tiny."""

import json
import os

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions_m4 import M4State
from lanegame.config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, NONE, PRESSURE, VISION, W_HOLD, W_SHOVE,
                             W_SLOWPUSH, W_THIN, champ_act, champ_behavior, champ_objective, champ_wave)
from lanegame.env import LaneEnv, UniformAgent
from lanegame.experiments.rl.m44 import actions as AC
from lanegame.experiments.rl.m44 import common as C
from lanegame.experiments.rl.m44 import interventions as IV
from lanegame.experiments.rl.m44 import metrics as M
from lanegame.experiments.rl.m44 import records as R
from lanegame.state import State

ENVS = list(C.ENV_ORDER)
A, B = 0, 1


def _uniform_states(cfg, turns=150, seed=3):
    env = LaneEnv(cfg, max_turns=turns, cache_size=2000)
    obs = env.reset(seed=seed)
    ua, ub, rng = UniformAgent(), UniformAgent(), np.random.default_rng(seed)
    out = []
    while True:
        out += [obs[0], obs[1]]
        r = env.step(ua.act(obs[0], rng), ub.act(obs[1], rng))
        if r.done:
            return out
        obs = r.obs


# ---------------------------------------------------------------------------------------------- environments
def test_the_three_environments_are_the_documented_presets_with_one_interface():
    four, econ, aggro = (C.env_cfg(k) for k in ENVS)
    assert four == GameConfig.m4_four_actions() and econ == GameConfig.m4_four_actions_econ()
    assert aggro == GameConfig.m4_four_actions_econ_aggro()
    for k in ENVS:
        cfg = C.env_cfg(k)
        assert C.env_key(cfg) == k and C.preset_name(cfg) == C.ENVS[k].preset
        e = LaneEnv(cfg)
        assert (e.obs_size, e.n_actions) == (88, 61) and cfg.gamma == 0.99 and cfg.continuing
    d = lambda x, y: {f for f, v in x.to_dict().items() if v != y.to_dict()[f]}
    assert d(econ, aggro) == set(GameConfig.M4_FOUR_AGGRO_FIELDS)
    assert d(four, econ) == set(GameConfig.M4_FOUR_ECON_FIELDS)
    assert C.env_key(GameConfig.m4_lasthit()) is None


def test_training_settings_are_identical_across_environments_and_continuing():
    from lanegame.rl.ppo import check_time_semantics
    ref = None
    for k in ENVS:
        p = C.ppo_config(k, 1)
        check_time_semantics(C.env_cfg(k), p)
        assert p.gamma == C.env_cfg(k).gamma == 0.99 and p.gae_lambda == 0.95 and p.segment_length == 128
        assert p.game_length == 512 and p.ent_coef == 0.03 and p.n_envs == 16 and p.torch_threads == 1
        dd = p.to_dict()
        if ref is None:
            ref = dd
        assert dd == ref                                       # nothing differs but the game itself
    s = C.shared_config()
    assert s.current_fraction == 1.0 and s.snapshot_every == 25_600
    spec = C.training_spec()
    assert spec["transitions_per_update"] == 4096 and spec["updates_main"] == 500


# ---------------------------------------------------------------------------------------------- actions
def test_decoding_uses_the_authoritative_helpers_and_only_34_codes_can_ever_be_legal():
    for a in range(61):
        d = AC.decode(a)
        if a == CHAMP_ABSENT:
            assert d.absent and d.wave == -1
            continue
        assert (d.wave, d.behavior, d.objective) == (champ_wave(a), champ_behavior(a), champ_objective(a))
        assert champ_act(d.wave, d.behavior, d.objective) == a
    assert AC.decode(champ_act(W_SLOWPUSH, B_TRADE, PRESSURE)).label() == "Slow Push + Trade + Pressure"
    assert AC.decode(AC.RECALL).recall and AC.RECALL == 12
    assert [AC.WAVE_INDEX[w] for w in (W_HOLD, W_THIN, W_SLOWPUSH, W_SHOVE)] == [0, 1, 2, 3]
    assert len(AC.EVER_LEGAL) == 34 and len(AC.NEVER_LEGAL) == 27
    for a in AC.NEVER_LEGAL:                              # Yield / Safe with an objective, non-canonical Recall codes
        d = AC.decode(a)
        assert (d.behavior in (B_SAFE, 1) and d.objective != NONE) or (d.recall and a != AC.RECALL)
    with pytest.raises(ValueError):
        AC.decode(61)


@pytest.mark.parametrize("key", ENVS)
def test_legal_sets_and_masks_in_play_stay_inside_the_ever_legal_codes(key):
    cfg = C.env_cfg(key)
    for o in _uniform_states(cfg, 200, seed=7):
        assert set(o.legal) <= AC.EVER_LEGAL
        assert set(np.flatnonzero(o.mask)) == set(o.legal) and len(o.mask) == 61
        if o.legal != (CHAMP_ABSENT,):                     # every wave slot is legal whenever any code is
            waves = {champ_wave(a) for a in o.legal if a != AC.RECALL}
            assert waves == {W_HOLD, W_THIN, W_SLOWPUSH, W_SHOVE}
        v = o.vector
        assert v.dtype == np.float32 and len(v) == 88 and v.min() >= 0 and v.max() <= 1


def test_masses_split_a_distribution_by_component_and_refuse_impossible_mass():
    p = np.zeros(61)
    p[champ_act(W_SLOWPUSH, B_TRADE)] = 0.5
    p[champ_act(W_SHOVE, B_HOLD, VISION)] = 0.3
    p[AC.RECALL] = 0.2
    m = AC.masses(p)
    assert np.allclose(m["wave"], [0, 0, 0.5, 0.3]) and np.isclose(m["recall"], 0.2)
    assert np.allclose(m["beh"], [0.3, 0, 0.5, 0, 0.2]) and np.allclose(m["obj"], [0.5, 0, 0.3])
    p[champ_act(W_HOLD, B_SAFE, PRESSURE)] = 0.1          # Safe never carries an objective
    with pytest.raises(ValueError):
        AC.masses(p)


# ---------------------------------------------------------------------------------------------- shared self-play
def _ppo(key, **kw):
    base = dict(total_steps=800, n_envs=4, segment_length=24, game_length=40, minibatch_size=64, epochs=1, seed=3,
                hidden=(32, 32), lane_cache=2000, ent_coef=C.ENT_COEF)
    base.update(kw)
    from lanegame.rl import PPOConfig
    return PPOConfig.continuing(C.env_cfg(key), **base)


@pytest.mark.parametrize("key", ENVS)
def test_shared_self_play_trains_both_seats_from_their_own_perspective_on_every_environment(key):
    torch = pytest.importorskip("torch")
    from lanegame.champions import swap_seats
    from lanegame.env import observe
    from lanegame.env.observation_m4 import decode_m4
    from lanegame.rl import MatchupSampler, PolicyValueNet, SharedCollector, SnapshotPool
    cfg = C.env_cfg(key)
    torch.manual_seed(0)
    net = PolicyValueNet(88, 61, (32, 32))
    col = SharedCollector(cfg, _ppo(key), MatchupSampler(SnapshotPool(4), 1.0), seed=1)
    roll, finished, counts = col.collect(net, torch.Generator().manual_seed(0))
    assert roll["active"].all() and counts["transitions_seat_a"] == counts["transitions_seat_b"]
    assert counts["terminations"] == 0 and counts["transitions_historical"] == 0
    assert np.array_equal(roll["rewards"][:, 0::2], -roll["rewards"][:, 1::2])
    a = roll["active"]
    assert roll["mask"][a][np.arange(a.sum()), roll["actions"][a]].all()          # sampled under the mask
    assert set(np.unique(roll["actions"][a])) <= AC.EVER_LEGAL
    for t in range(0, 24, 6):
        for i in range(4):
            pa, la, ca, _ = decode_m4(roll["obs"][t, 2 * i], cfg)
            pb, lb, cb, _ = decode_m4(roll["obs"][t, 2 * i + 1], cfg)
            assert pa == pb and lb == la.mirror() and cb == ca.mirror()
            assert np.array_equal(roll["obs"][t, 2 * i + 1], observe(la.mirror(), pa, 0, swap_seats(cfg), ca.mirror()))
    # one network for both seats: the recorded log-probabilities are the network's at each seat's observation
    from lanegame.rl import MaskedCategorical
    with torch.no_grad():
        logits, values = net(torch.as_tensor(roll["obs"][3]))
        lp = MaskedCategorical(logits, torch.as_tensor(roll["mask"][3])).log_prob(torch.as_tensor(roll["actions"][3]))
    assert np.allclose(lp.numpy(), roll["logprobs"][3], atol=1e-6) and np.allclose(values.numpy(), roll["values"][3])


def test_training_is_reproducible_and_a_seed_initializes_identically_in_every_environment(tmp_path):
    pytest.importorskip("torch")
    from lanegame.rl import load_checkpoint, train_shared, weights_digest
    from lanegame.experiments.rl.m44.train import _initial_digest
    scfg = C.shared_config()
    a = train_shared(C.env_cfg("aggro"), _ppo("aggro"), scfg, out_dir=str(tmp_path / "a"), verbose=False)
    b = train_shared(C.env_cfg("aggro"), _ppo("aggro"), scfg, verbose=False)
    c = train_shared(C.env_cfg("econ"), _ppo("econ"), scfg, verbose=False)
    da, db, dc = (weights_digest(x.net.state_dict()) for x in (a, b, c))
    assert da == db and da != dc                        # same seed and game: bit-identical; another game: different
    assert len({_initial_digest(k, 3) for k in ENVS}) == 1 and _initial_digest("aggro", 4) != _initial_digest("aggro", 3)
    net, cfg, meta = load_checkpoint(str(tmp_path / "a" / "final.pt"))
    assert cfg == C.env_cfg("aggro") and meta["ppo"]["gamma"] == 0.99 and meta["ppo"]["ent_coef"] == 0.03
    assert meta["time_semantics"]["time"] == "continuing" and net.spec()["n_actions"] == 61


# ---------------------------------------------------------------------------------------------- compatibility
def _ckpt(tmp_path, name, cfg, n_actions):
    pytest.importorskip("torch")
    from lanegame.rl import PolicyValueNet, save_checkpoint
    p = str(tmp_path / f"{name}.pt")
    save_checkpoint(p, PolicyValueNet(88, n_actions, (16,)), cfg, {"name": name})
    return p


def test_a_46_output_checkpoint_is_never_read_as_a_61_code_policy(tmp_path):
    p = _ckpt(tmp_path, "old", GameConfig.m4_lasthit(), 46)
    for k in ENVS:
        with pytest.raises(ValueError, match="46 action outputs"):
            C.load_policy(p, C.env_cfg(k))
        with pytest.raises(ValueError, match="46 action outputs"):
            C.load_policy(p, C.env_cfg(k), cross_env=True)          # not even as a deliberate cross-evaluation
    ag = C.load_policy(p, GameConfig.m4_lasthit())                  # it keeps working on its own environment
    assert ag.compat.same_config and ag.compat.trained_on == "m4_lasthit"


def test_a_four_action_checkpoint_loads_only_on_its_preset_unless_cross_evaluation_is_requested(tmp_path):
    p = _ckpt(tmp_path, "four", C.env_cfg("four"), 61)
    assert C.load_policy(p, C.env_cfg("four")).compat.same_config
    with pytest.raises(ValueError, match="cross_env=True"):
        C.load_policy(p, C.env_cfg("econ"))
    ag = C.load_policy(p, C.env_cfg("econ"), cross_env=True)
    assert "champion_absent_max" in ag.compat.differing and ag.compat.differing["champion_absent_max"] == (3, 5)
    q = _ckpt(tmp_path, "econ", C.env_cfg("econ"), 61)
    assert set(C.load_policy(q, C.env_cfg("aggro"), cross_env=True).compat.differing) == set(GameConfig.M4_FOUR_AGGRO_FIELDS)
    with pytest.raises(ValueError):
        C.agent_from_spec(q, C.env_cfg("aggro"))
    # a tampered file is refused by the digest check before any compatibility question
    import torch
    d = torch.load(q, weights_only=False)
    d["state_dict"]["policy_head.bias"] = d["state_dict"]["policy_head.bias"] + 1.0
    torch.save(d, q)
    with pytest.raises(ValueError, match="digest"):
        C.load_policy(q, C.env_cfg("econ"))


# ---------------------------------------------------------------------------------------------- records
def test_a_record_is_the_same_game_as_the_evaluator_plays_and_classifies_every_wave():
    from lanegame.env.discounted import _play
    from lanegame.experiments.rl.m44.references import reference_suite
    cfg = C.env_cfg("aggro")
    suite = reference_suite()
    x, y = suite["slowpush4"], suite["thin4"]
    ss = np.random.SeedSequence(11, spawn_key=(0,))
    rec = R.play_record(cfg, (x, y), ss, 60)
    rew, _ = _play(LaneEnv(cfg, cache_size=2000), (reference_suite()["slowpush4"], reference_suite()["thin4"]),
                   np.random.SeedSequence(11, spawn_key=(0,)), 60)
    assert np.allclose(rec.data["reward"][:, 0], rew) and np.allclose(rec.data["reward"][:, 1], -rew)
    d = rec.data
    for t in range(rec.turns):
        for s in (A, B):
            a = int(d["a"][t, s])
            if a == CHAMP_ABSENT or a == AC.RECALL:
                assert d["wave"][t, s] == -1
            else:
                assert d["wave"][t, s] == AC.WAVE_INDEX[champ_wave(a)]
                assert d[f"pw{int(d['wave'][t, s])}"][t, s] == 1.0          # a scripted agent: one-hot masses
    pres = (d["present"][:, 0] > 0) & (d["wave"][:, 0] >= 0)
    # slowpush4 plays Slow Push except when its low-HP pre-rule walks back (Last Hit + Safe)
    for t in np.flatnonzero(pres):
        assert d["wave"][t, 0] == AC.WAVE_INDEX[W_SLOWPUSH] or int(d["a"][t, 0]) == champ_act(W_HOLD, B_SAFE)
    assert (d["wave"][pres, 0] == AC.WAVE_INDEX[W_SLOWPUSH]).mean() > 0.5


def test_records_of_a_network_sample_exactly_like_the_network_agent():
    torch = pytest.importorskip("torch")
    from lanegame.env.discounted import _play
    from lanegame.rl import NetAgent, PolicyValueNet
    torch.manual_seed(5)
    ag = NetAgent(PolicyValueNet(88, 61, (16,)), "net")
    cfg = C.env_cfg("econ")
    rec = R.play_record(cfg, (ag, ag), np.random.SeedSequence(2), 50)
    rew, _ = _play(LaneEnv(cfg), (ag, ag), np.random.SeedSequence(2), 50)
    assert np.allclose(rec.data["reward"][:, 0], rew)
    p = rec.data[[f"pw{i}" for i in range(4)][0]]
    assert np.all(np.isfinite(p[rec.data["wave"] >= 0]))


# ---------------------------------------------------------------------------------------------- freezes
def _lane(ax, ah, bx, bh, ar=(), br=(), asg=0, bsg=0):
    return State(ax, ah, tuple(ar), asg, bx, bh, tuple(br), bsg, 0, 0)


def test_the_freeze_state_is_a_lane_state_not_an_action():
    from lanegame.experiments.m43c.harness import start
    cfg = C.env_cfg("econ")
    lane, m4, k = start("freeze", cfg)
    assert R.holds_freeze(lane, A) and not R.holds_freeze(lane, B)      # B's wave held at A's -1 by A
    lane, m4, k = start("neutral", cfg)
    assert not R.holds_freeze(lane, A) and not R.holds_freeze(lane, B)
    assert R.holds_freeze(_lane(-2, 8, 1, 12), A)                       # A's front at -2 meets B's at A's -1
    assert not R.holds_freeze(_lane(-2, 8, 1, 0), A)                    # no enemy front: nothing is held
    assert not R.holds_freeze(_lane(0, 8, 0, 8), A)                     # contact at the lane cell
    assert R.holds_freeze(_lane(1, 12, -2, 8), B)                       # the mirror image
    assert R.wave_position(_lane(-2, 8, 1, 12), A) == -1.5 and R.wave_position(_lane(-2, 8, 1, 12), B) == 1.5


def _synthetic(held, en_crash=None, **cols):
    T = len(held)
    d = {f: np.zeros((T, 2)) for f in R.FIELDS}
    d["present"][:] = 1
    d["en_present"][:] = 1
    d["power"][:] = 1.0
    d["held"][:, 0] = held
    d["held_on"][:, 1] = held
    d["own_fh"][:] = 8
    d["en_fh"][:] = 8
    d["contact"][:] = 1
    if en_crash is not None:
        d["en_crash"][:, 0] = en_crash
    d["plates_left"][:] = 3
    d["own_plates_left"][:] = 3
    for k, v in cols.items():
        d[k][:] = v
    return R.GameRecord("test", ("x", "y"), 0, "s0", False, d)


def test_freeze_episodes_need_persistence_and_record_how_they_end():
    held = [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
    crash = [0] * 16
    crash[5] = 8                                         # the held wave crashes into A's tower on the last held turn
    rec = _synthetic(held, crash)
    rec.data["en_fh"][10, 0] = 0                         # the second run's held wave is cleared without a crash
    eps = M.freeze_episodes(rec, A)
    assert [(e["start"], e["length"], e["end"]) for e in eps] == [(1, 5, "crashed_in"), (8, 2, "held_wave_cleared"),
                                                                  (12, 4, "censored")]
    assert [e["established"] for e in eps] == [True, False, True] and not any(e["sustained"] for e in eps)
    assert M.freeze_episodes(rec, B) == []               # B is frozen on, it holds nothing
    f = M.freezes([rec], focal="x")
    assert f["established"]["n"] == 2 and f["all_episodes"]["n"] == 3 and f["established"]["end"]["censored"] == 1
    # the episode's economics: gold the freezer earned more than the frozen-on side while it held
    rec.data["earned"][1:6, 0] = 0.2
    rec.data["earned"][1:6, 1] = 0.1
    assert M.freeze_episodes(rec, A)[0]["gold_diff"] == pytest.approx(0.5)


def test_choosing_thin_is_not_a_freeze_and_a_freeze_does_not_need_thin():
    rec = _synthetic([0] * 10, wave=AC.WAVE_INDEX[W_THIN])             # Thin every turn, lane never held
    assert M.freezes([rec])["established"]["n"] == 0
    rec = _synthetic([1] * 10, wave=AC.WAVE_INDEX[W_SHOVE])            # held throughout while Shoving
    f = M.freezes([rec], focal="x")["established"]
    assert f["n"] == 1 and f["freezer_preserving_share"][0] == 0.0


# ---------------------------------------------------------------------------------------------- dives, vision, plates
def test_plates_left_come_from_the_tower_damage_state():
    from lanegame.champions_m4 import initial_m4
    from lanegame.mechanics import initial_state
    from lanegame.champions_m4 import context
    cfg = C.env_cfg("aggro")
    ctx = context(cfg)
    lane = initial_state(cfg)
    for tau, left in ((0, 3), (1599, 3), (1600, 2), (3600, 1), (5999, 1), (6000, 0)):
        m4 = initial_m4(cfg)
        m4 = M4State(m4.a._replace(tower_dmg=tau), m4.b)
        st = R.decision_state(cfg, ctx, lane, m4, 0, A, (0,))
        assert st["plates_left"] == left and st["own_plates_left"] == 3
        assert R.decision_state(cfg, ctx, lane, m4, 0, B, (0,))["own_plates_left"] == left


def test_plate_phase_metrics_split_by_the_simulation_state_not_the_turn():
    rec = _synthetic([0] * 8, wave=AC.WAVE_INDEX[W_SHOVE])
    rec.data["plates_left"][:, 0] = [3, 2, 1, 0, 0, 1, 0, 0]             # not monotone on purpose: state, not time
    rec.data["plate_gold"][:, 0] = 1.0
    out = M.plate_phases([rec], focal="x")
    assert out["attack plates left"]["turns"] == 4 and out["attack plates exhausted"]["turns"] == 4
    assert out["attack_plates_exhausted_at_turn"]["median"] == 4


def test_dives_and_the_busy_condition_come_from_the_turn_outcome():
    from lanegame.champions_m4 import context
    cfg = C.env_cfg("aggro")
    ctx = context(cfg)
    lane = _lane(1, 8, 1, 8, asg=6)
    info = {"levels_before": (1, 1), "away": (False, False), "champ_end": (2, -2), "trade_connected": (True, False),
            "state_before": list(lane), "effective_objectives": (PRESSURE, NONE), "kills": ({}, {}), "reward": 0.0,
            "earned_units": (0, 0), "farm_units": (0, 0), "farm_available": (0.0, 0.0), "plate_units": (0, 0),
            "bounty_units": (0, 0), "xp_gain": (0, 0), "xp_discarded": (0, 0), "died": (False, False),
            "recalled": (False, False), "returned": (False, False), "ganked": (False, False),
            "threatened": (False, False), "pinned": (False, False), "gank_p": (0.0, 0.0), "gank_damage_p": (0.0, 0.0),
            "gank_damage": (0, 0), "tower_p": (1.0, 0.0), "tower_damage": (0, 0), "trade_damage": (50, 0),
            "damage_taken": (0, 0), "trade_attempted": (True, False), "crash_hp": (0, 0), "plates_broken": (0, 0),
            "tower_damage_dealt": (0, 0), "vision_set": (False, False), "wave_budget": (1, 1),
            "events": {"champion_hp": (0, 0)}}
    out = R.outcome(cfg, ctx, info, lane, A)
    assert out["dive"] and out["busy"] and not out["adj_trade"] and out["pressure_eff"]
    info2 = dict(info, champ_end=(1, -2), state_before=list(lane._replace(asg=0)))
    out2 = R.outcome(cfg, ctx, info2, lane, A)
    assert not out2["dive"] and out2["adj_trade"] and not out2["busy"]


# ---------------------------------------------------------------------------------------------- interventions
def test_interventions_change_one_component_and_never_leave_the_legal_set():
    cfg = C.env_cfg("aggro")
    rng = np.random.default_rng(0)
    seen_obj = False
    for o in _uniform_states(cfg, 300, seed=12):
        if o.legal == (CHAMP_ABSENT,):
            assert not IV.possible(o.legal, "wave", W_THIN)
            continue
        p = rng.dirichlet(np.ones(len(o.legal)))
        for w in (W_HOLD, W_THIN, W_SLOWPUSH, W_SHOVE):
            for rule in ("marginal", "conditional"):
                q = IV.intervene(p, o.legal, "wave", w, rule)
                assert np.isclose(q.sum(), 1) and (q >= 0).all()
                sup = [o.legal[i] for i in np.flatnonzero(q > 0)]
                assert all(champ_wave(a) == w or a == AC.RECALL for a in sup)
                if AC.RECALL in o.legal:                                 # Recall keeps the policy's own mass
                    assert np.isclose(q[o.legal.index(AC.RECALL)], p[o.legal.index(AC.RECALL)])
        for b in (B_HOLD, B_TRADE, B_SAFE):
            q = IV.intervene(p, o.legal, "behavior", b)
            assert all(champ_behavior(o.legal[i]) == b for i in np.flatnonzero(q > 0))
        for obj in (PRESSURE, VISION):
            if IV.possible(o.legal, "objective", obj):
                seen_obj = True
                q = IV.intervene(p, o.legal, "objective", obj)
                assert all(champ_objective(o.legal[i]) == obj for i in np.flatnonzero(q > 0))
            else:
                with pytest.raises(ValueError):
                    IV.intervene(p, o.legal, "objective", obj)
        if AC.RECALL not in o.legal:
            with pytest.raises(ValueError):
                IV.intervene(p, o.legal, "behavior", B_RECALL)
    assert seen_obj
    with pytest.raises(ValueError):
        IV.intervene(np.ones(2) / 2, (0, 1), "behavior", B_SAFE, rule="marginal")       # not a defined rule


# ---------------------------------------------------------------------------------------------- counterfactuals
@pytest.fixture(scope="module")
def recorded():
    from lanegame.experiments.rl.m44.references import reference_suite
    cfg = C.env_cfg("econ")
    s = reference_suite()
    rec = R.play_record(cfg, (s["grow_crash_recall4"], s["freeze_hunt4"]), np.random.SeedSequence(4), 80,
                        keep_snapshots=True)
    return cfg, rec


def test_a_snapshot_restores_the_recorded_decision_state_exactly(recorded):
    from lanegame.diagnostics import observation, restore
    cfg, rec = recorded
    env = LaneEnv(cfg)
    for t in (5, 31, 77):
        snap = rec.snapshots[t]
        obs = restore(env, snap, 0)
        ctx = __import__("lanegame.champions_m4", fromlist=["context"]).context(cfg)
        st = R.decision_state(cfg, ctx, obs[0].state, obs[0].champ, obs[0].k, A, obs[0].legal)
        for f in ("pos", "own_fh", "en_fh", "wave_adv", "held", "plates_left", "own_siege", "hp_frac"):
            assert st[f] == rec.data[f][t, 0] or (np.isnan(st[f]) and np.isnan(rec.data[f][t, 0]))
        assert np.array_equal(observation(cfg, snap, 1).vector, obs[1].vector)


def test_paired_rollouts_share_streams_and_identical_plans_give_identical_returns(recorded):
    from lanegame.diagnostics import Commit, FollowPolicy, branch, observation
    cfg, rec = recorded
    pol = UniformAgent()
    t = 20
    snap = rec.snapshots[t]
    ob = observation(cfg, snap, A)
    cands = {"policy": FollowPolicy(), "policy_again": FollowPolicy(),
             "no_op": IV.ComponentPlan("no_op", "wave", lambda o, tt: None, turns=3),
             "thin": IV.component_plan("thin", "wave", W_THIN, turns=1),
             "shove": IV.component_plan("shove", "wave", W_SHOVE, turns=1)}
    res = branch(cfg, snap, A, cands, (pol, pol), 6, 99, max_turns=15)
    assert np.array_equal(res.returns["policy"], res.returns["policy_again"])
    assert np.array_equal(res.returns["policy"], res.returns["no_op"])
    for c in res.candidates:                             # the simultaneous opponent never sees the focal choice
        assert np.array_equal(res.opp_first[c], res.opp_first["policy"])
    # opponent protocol "modal": the opponent commits to a fixed action on the branch turn for every candidate
    ob_b = observation(cfg, snap, B)
    fixed = ob_b.legal[0]
    res2 = branch(cfg, snap, A, {"policy": FollowPolicy(), "thin": IV.component_plan("t", "wave", W_THIN)}, (pol, pol),
                  4, 99, max_turns=10, opponent_plan=Commit(fixed, "fixed"))
    assert all((res2.opp_first[c] == fixed).all() for c in res2.candidates)
    assert res2.meta["opponent_plan"] == "fixed"


def test_an_impossible_intervention_is_refused_at_the_branch_turn_and_skipped_later(recorded):
    from lanegame.diagnostics import observation
    cfg, rec = recorded
    d = rec.data
    t = next(t for t in range(rec.turns) if d["present"][t, 0] > 0 and d["obj_legal"][t, 0] == 0)
    ob = observation(cfg, rec.snapshots[t], A)
    assert not IV.possible(ob.legal, "objective", VISION)
    p = np.ones(len(ob.legal)) / len(ob.legal)
    plan = IV.ComponentPlan("vision", "objective", IV.const(VISION), turns=5)
    with pytest.raises(ValueError, match="not possible at the branch state"):
        plan.transform(ob, 0, p)                         # never an invented action on the branch turn
    assert np.array_equal(plan.transform(ob, 2, p), p) and plan.skipped_turns == 1      # later: unchanged, counted


def test_the_counterfactual_harness_reports_frequency_measurement_and_critic_separately():
    torch = pytest.importorskip("torch")
    from lanegame.experiments.rl.m44 import counterfactual as CF
    from lanegame.rl import NetAgent, PolicyValueNet
    torch.manual_seed(1)
    ag = NetAgent(PolicyValueNet(88, 61, (16,)), "net")
    ag.meta = {"ppo": {"reward_scale": 0.1}}
    cfg = C.env_cfg("aggro")
    out = CF.run(cfg, ag, ag, ["neutral", "oversized_enemy"], n_states=2, n_roll=3, state_seed=5, roll_seed=6,
                 window=12, game_turns=60)
    for key, r in out["results"].items():
        for st in r["states"]:
            a, b = st["contrast_names"]
            assert a in st["measured"] and b in st["measured"] and "policy" in st["measured"]
            assert set(st["frequency"]) <= set(st["candidates"]) and st["opp_first_identical"]
            assert "critic" in st and set(st["critic"]["Q1"]) == set(st["measured"])
            assert all(0 <= v <= 1 for v in st["frequency"].values())
        s = r["summary"]
        if s["n_states"]:
            assert s["opponent_first_action_identical_across_candidates"]
            assert len(s["mean_contrast"]) == 2


# ---------------------------------------------------------------------------------------------- reporting
def _keys(d, prefix=""):
    out = set()
    for k, v in d.items():
        out.add(prefix + str(k))
        if isinstance(v, dict) and not k.startswith(("by_state", "frequency")):
            out |= _keys(v, prefix + str(k) + ".")
    return out


def test_metric_reports_have_the_same_structure_in_every_environment():
    from lanegame.experiments.rl.m44.references import reference_suite
    shapes = []
    for k in ENVS:
        s = reference_suite()
        rec = R.play_record(C.env_cfg(k), (s["shove_hunt4"], s["freeze_hunt4"]), np.random.SeedSequence(8), 60)
        m = M.all_metrics([rec])
        assert list(m) == ["natural:s0"]
        shapes.append(_keys(m["natural:s0"]))
        json.dumps(m, default=float)                     # JSON-serializable
    assert shapes[0] == shapes[1] == shapes[2]


def test_natural_and_initialized_games_are_never_pooled():
    from lanegame.experiments.m43c.harness import start
    from lanegame.experiments.rl.m44.references import reference_suite
    cfg = C.env_cfg("econ")
    s = reference_suite()
    r1 = R.play_record(cfg, (s["farm4"], s["thin4"]), np.random.SeedSequence(1), 20)
    r2 = R.play_record(cfg, (s["farm4"], s["thin4"]), np.random.SeedSequence(1), 20, start=start("freeze", cfg),
                       start_label="freeze/equal")
    assert not r1.initialized and r2.initialized
    assert sorted(M.all_metrics([r1, r2])) == ["initialized:freeze/equal", "natural:s0"]


# ---------------------------------------------------------------------------------------------- viewer and CLI
def test_a_four_action_recording_replays_in_the_viewer(tmp_path):
    from lanegame.experiments.rl.m44.trace import export, verify
    from lanegame.viewer import session as ss
    assert {"m4_four_actions_econ", "m4_four_actions_econ_aggro"} <= set(ss.CONFIGS)
    rec = export("t", "aggro", "ref:shove_hunt4", "ref:farm4", 5, 40, "test", out_dir=str(tmp_path))
    v = verify(rec["path"])
    assert v["config"] == "m4_four_actions_econ_aggro" and v["turns"] == 40 and v["actions_equal"] and v["payoff_equal"]


def test_the_m44b_training_budget_cannot_start_by_accident():
    from lanegame.experiments.rl.m44.__main__ import main
    with pytest.raises(SystemExit, match="confirm-m44b"):
        main(["train_main"])


def test_the_cli_offers_the_two_calibrated_presets():
    from lanegame.__main__ import RL_MODELS
    assert {"m4_four_actions", "m4_four_actions_econ", "m4_four_actions_econ_aggro"} <= set(RL_MODELS)
