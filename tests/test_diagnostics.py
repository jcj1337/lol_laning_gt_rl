"""Diagnostics tools (lanegame.diagnostics, docs/openai_five_diagnostics_report.md): restorable snapshots and replay
consistency, paired counterfactual rollouts (legal actions, reproducibility, frozen policies, no information about
the other seat's uncommitted action, common random numbers), reward accounting, seat symmetry, seed separation, the
action audit's attribution of rewards and action frequencies, rule overlays and the Phase D helpers."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import RECALL, N_CHANCE, ChampSide, ChampState, swap_seats
from lanegame.config import B_HOLD, B_SAFE, B_TRADE, CHAMP_ABSENT, SH, SP, champ_act
from lanegame.diagnostics import (DEFER, Commit, FollowPolicy, FunctionPlan, ProbCache, Snapshot, audit_games, branch,
                                  legality_table, observation, restore, rollout, rollout_seed, sample_index, summarize)
from lanegame.env import ChampionRuleAgent, LaneEnv, UniformAgent, champion_agents, observe, play_episode, run_episodes
from lanegame.mechanics import initial_state
from lanegame.champions import initial_champions

CFG = GameConfig.champions(horizon=12)
AGENTS = champion_agents(CFG)
TRADER, FARMER, UNIFORM = AGENTS["aggressive_trader"], AGENTS["hold_farmer"], UniformAgent()


def _snap0(cfg=CFG, skills=(0.5, 0.5)):
    c = cfg.with_(skill_a=skills[0], skill_b=skills[1])
    return Snapshot(initial_state(c), initial_champions(c), 0, skills)


def _mid_snapshot(seed=3, k=6, x=UNIFORM, y=TRADER):
    log = play_episode(LaneEnv(CFG), x, y, seed, record=True)
    return Snapshot.from_info(log.turns[k], (0.5, 0.5)), log


class _ReplayRNG:
    """Stands in for the environment's generator: returns recorded chance vectors in order."""

    def __init__(self, draws):
        self.draws = [np.asarray(d) for d in draws]

    def random(self, n=None):
        assert n == N_CHANCE
        return self.draws.pop(0)


# ---------------------------------------------------------------------------------------------- snapshots / replay
def test_restored_snapshot_replays_the_recorded_trajectory_exactly():
    snap, log = _mid_snapshot()
    env = LaneEnv(CFG)
    restore(env, snap, seed=0)
    assert env.k == snap.k and env.state == snap.lane and env.champ == snap.champ
    env._rng = _ReplayRNG([info["chance"] for info in log.turns[snap.k:]])   # the recorded chance draws
    for info in log.turns[snap.k:]:
        assert env.state.to_list() == info["state_before"] and env.champ.to_list() == info["champ_before"]
        r = env.step(*info["actions"])
        assert r.rewards[0] == pytest.approx(info["reward"], abs=1e-12)
        assert r.info["state_after"] == info["state_after"] and r.info["champ_after"] == info["champ_after"]
    assert env.done


def test_snapshot_round_trip_mirror_and_side_edit():
    snap, _ = _mid_snapshot()
    assert Snapshot.from_dict(snap.to_dict()) == snap
    m = snap.mirror()
    assert m.lane == snap.lane.mirror() and m.champ == snap.champ.mirror() and m.mirror() == snap
    s2 = snap.with_side(1, hp=1)
    assert s2.champ.b.hp == 1 and s2.champ.a == snap.champ.a and s2.champ.b.pos == snap.champ.b.pos
    dead = snap.with_side(0, pos=-CFG.tower, hp=0, absent=CFG.death_turns)
    assert observation(CFG, dead, 0).legal == (CHAMP_ABSENT,)


def test_observation_helper_matches_the_environment_and_is_mirror_consistent():
    snap = Snapshot(_mid_snapshot()[0].lane, _mid_snapshot()[0].champ, 6, (0.2, 0.8))
    env = LaneEnv(snap.config(CFG))
    oa, ob = restore(env, snap, 1)
    for seat, o in ((0, oa), (1, ob)):
        h = observation(CFG, snap, seat)
        assert np.array_equal(h.vector, o.vector) and h.legal == o.legal and np.array_equal(h.mask, o.mask)
    # seat B's view equals seat A's view of the mirrored snapshot (skills swapped with the seats)
    assert np.array_equal(observation(CFG, snap, 1).vector, observation(CFG, snap.mirror(), 0).vector)
    assert np.array_equal(observe(snap.lane.mirror(), snap.k, 0, swap_seats(snap.config(CFG)), snap.champ.mirror()),
                          observation(CFG, snap, 1).vector)


def test_restore_refuses_a_mismatched_configuration():
    with pytest.raises(ValueError):
        restore(LaneEnv(CFG), _snap0(skills=(0.2, 0.8)), 0)
    with pytest.raises(ValueError):
        restore(LaneEnv(GameConfig(horizon=12)), _snap0(), 0)


# ---------------------------------------------------------------------------------------------- rollouts
def test_rollout_without_a_plan_reproduces_play_episode():
    env = LaneEnv(CFG)
    cache = ProbCache()
    for seed in range(6):
        for x, y in ((UNIFORM, TRADER), (FARMER, UNIFORM)):
            log = play_episode(env, x, y, seed, record=True)
            r = rollout(env, _snap0(), 0, FollowPolicy(), (x, y), seed, cache, record=True)
            assert r.ret == pytest.approx(log.payoff, abs=1e-12)
            assert [tuple(i["actions"]) for i in r.infos] == [tuple(a) for a in log.actions]
            r_b = rollout(env, _snap0(), 1, FollowPolicy(), (x, y), seed, cache)
            assert r_b.ret == pytest.approx(-log.payoff, abs=1e-12)


def test_sample_index_is_the_agents_sampling_rule():
    from lanegame.env.agents import sample
    p = np.array([0.1, 0.0, 0.5, 0.4])
    for s in range(50):
        assert sample_index(p, np.random.default_rng(s).random()) == sample(np.random.default_rng(s), p)


def test_reward_accounting_of_a_rollout():
    snap, _ = _mid_snapshot()
    env = LaneEnv(CFG)
    for seed in range(5):
        r = rollout(env, snap, 0, Commit(champ_act(SH, B_TRADE)) if champ_act(SH, B_TRADE) in
                    observation(CFG, snap, 0).legal else FollowPolicy(), (UNIFORM, TRADER), seed, record=True)
        assert r.ret == pytest.approx(sum(i["reward"] for i in r.infos), abs=1e-12)
        decomposed = (r.farm[0] - r.farm[1]) + (r.plates[0] - r.plates[1]) + (r.bounty[0] - r.bounty[1])
        assert r.ret == pytest.approx(decomposed, abs=1e-9)
        assert r.turns == CFG.horizon - snap.k and r.first_reward == pytest.approx(r.infos[0]["reward"])


def test_illegal_committed_actions_and_plans_are_rejected():
    snap = _snap0()
    legal = observation(CFG, snap, 0).legal
    illegal = next(a for a in range(45) if a not in legal)
    with pytest.raises(ValueError):
        rollout(LaneEnv(CFG), snap, 0, Commit(illegal), (UNIFORM, UNIFORM), 0)
    with pytest.raises(ValueError):
        rollout(LaneEnv(CFG), snap, 0, FunctionPlan("bad", lambda o, t: CHAMP_ABSENT), (UNIFORM, UNIFORM), 0)


def test_defer_keeps_a_plan_active_and_none_hands_over_for_good():
    seen = []

    def plan(obs, t):
        seen.append(t)
        if t < 2:
            return DEFER
        return champ_act(SP, B_SAFE) if t == 2 else None
    r = rollout(LaneEnv(CFG), _snap0(), 0, FunctionPlan("p", plan), (FARMER, FARMER), 4, record=True)
    assert seen == [0, 1, 2, 3]                                  # consulted until it returned None
    assert r.infos[2]["actions"][0] == champ_act(SP, B_SAFE)
    base = rollout(LaneEnv(CFG), _snap0(), 0, FollowPolicy(), (FARMER, FARMER), 4, record=True)
    assert [i["actions"] for i in r.infos[:2]] == [i["actions"] for i in base.infos[:2]]


# ---------------------------------------------------------------------------------------------- branching
def _cands(snap, seat=0):
    legal = observation(CFG, snap, seat).legal
    c = {CFG.label(a): Commit(a) for a in legal}
    c["policy"] = FollowPolicy()
    return c


def test_branch_is_reproducible_and_seeded():
    snap, _ = _mid_snapshot()
    r1 = branch(CFG, snap, 0, _cands(snap), (UNIFORM, TRADER), 40, root=11, key=(2,))
    r2 = branch(CFG, snap, 0, _cands(snap), (UNIFORM, TRADER), 40, root=11, key=(2,))
    r3 = branch(CFG, snap, 0, _cands(snap), (UNIFORM, TRADER), 40, root=12, key=(2,))
    for c in r1.candidates:
        assert np.array_equal(r1.returns[c], r2.returns[c])
    assert any(not np.array_equal(r1.returns[c], r3.returns[c]) for c in r1.candidates)


def test_rollout_seeds_are_fresh_objects_with_identical_streams():
    a, b = rollout_seed(5, 3, 1), rollout_seed(5, 3, 1)
    a.spawn(3)                                                   # spawning advances a's counter, not b's
    assert np.random.default_rng(rollout_seed(5, 3, 1).spawn(3)[0]).random() == \
        np.random.default_rng(b.spawn(3)[0]).random()
    assert rollout_seed(5, 3, 1).entropy == 5 and rollout_seed(5, 4, 1).spawn_key != rollout_seed(5, 3, 1).spawn_key


class _Spy:
    """Opponent that records every observation vector it is shown on the branch turn."""

    def __init__(self, base, k):
        self.base, self.k, self.seen = base, k, []

    def distribution(self, obs):
        if obs.k == self.k:
            self.seen.append(obs.vector.tobytes() + obs.mask.tobytes())
        return self.base.distribution(obs)


def test_opponent_branch_action_and_observation_do_not_depend_on_the_focal_candidate():
    snap, _ = _mid_snapshot()
    spy = _Spy(UNIFORM, snap.k)
    res = branch(CFG, snap, 0, _cands(snap), (UNIFORM, spy), 60, root=21, cache=ProbCache(max_entries=0))
    first = res.opp_first[res.candidates[0]]
    for c in res.candidates:
        assert np.array_equal(res.opp_first[c], first)            # same opponent action in every rollout m
    assert len(set(spy.seen)) == 1                                # one and the same observation for every candidate


def test_candidates_share_randomness_until_they_diverge():
    """Two candidates that pick the same action produce identical rollouts (common random numbers)."""
    snap, _ = _mid_snapshot()
    a = observation(CFG, snap, 0).legal[0]
    res = branch(CFG, snap, 0, {"x": Commit(a), "y": Commit(a)}, (UNIFORM, TRADER), 30, root=5)
    assert np.array_equal(res.returns["x"], res.returns["y"])
    d, se = res.diff("x", "y")
    assert d == 0.0 and se == 0.0


def test_committing_the_policys_own_sample_reproduces_the_policy_rollout():
    """Each seat draws one uniform per turn whether or not a plan overrides it, so after a committed action that
    equals what the policy would have sampled, the rollout continues exactly as the policy's own rollout."""
    snap, _ = _mid_snapshot()
    env = LaneEnv(CFG)
    for m in range(12):
        base = rollout(env, snap, 0, FollowPolicy(), (UNIFORM, TRADER), rollout_seed(3, m), record=True)
        same = rollout(env, snap, 0, Commit(base.first_actions[0]), (UNIFORM, TRADER), rollout_seed(3, m), record=True)
        assert same.ret == base.ret and [i["actions"] for i in same.infos] == [i["actions"] for i in base.infos]


def test_policies_stay_frozen_and_cached_probabilities_are_identical():
    torch = pytest.importorskip("torch")
    from lanegame.rl import NetAgent, PolicyValueNet
    env = LaneEnv(CFG)
    torch.manual_seed(0)
    agent = NetAgent(PolicyValueNet(env.obs_size, env.n_actions), "net")
    before = [p.detach().clone() for p in agent.net.parameters()]
    snap, _ = _mid_snapshot()
    cache = ProbCache()
    branch(CFG, snap, 0, _cands(snap), (agent, agent), 10, root=1, cache=cache)
    assert all(torch.equal(p, q) for p, q in zip(before, agent.net.parameters()))
    assert not any(p.requires_grad for p in agent.net.parameters())
    ob = observation(CFG, snap, 0)
    assert np.array_equal(cache.distribution(agent, ob), cache.distribution(agent, ob))
    assert np.allclose(cache.distribution(agent, ob), agent.distribution(ob))


def test_advantage_against_the_policy_mixture():
    snap, _ = _mid_snapshot()
    res = branch(CFG, snap, 0, _cands(snap), (UNIFORM, TRADER), 50, root=8)
    legal = observation(CFG, snap, 0).legal
    w = {CFG.label(a): 1.0 / len(legal) for a in legal}           # the uniform policy's own mixture
    piq = sum(wt * res.returns[c] for c, wt in w.items())
    for c in w:
        m, se = res.advantage(c, w)
        assert m == pytest.approx(float((res.returns[c] - piq).mean()))
    assert sum(wt * res.advantage(c, w)[0] for c, wt in w.items()) == pytest.approx(0.0, abs=1e-9)


def test_seat_symmetry_of_counterfactual_values():
    """The same situation seen from seat A, and mirrored into seat B, has the same value (statistically: the chance
    streams are not mirrored), for a candidate and for the policy."""
    snap, _ = _mid_snapshot(seed=9, k=4, x=FARMER, y=TRADER)
    n = 600
    ra = branch(CFG, snap, 0, {"p": FollowPolicy()}, (UNIFORM, TRADER), n, root=31)
    rb = branch(CFG, snap.mirror(), 1, {"p": FollowPolicy()}, (TRADER, UNIFORM), n, root=32)
    (ma, sa), (mb, sb) = ra.mean("p"), rb.mean("p")
    assert abs(ma - mb) < 4.5 * np.hypot(sa, sb)


# ---------------------------------------------------------------------------------------------- audit
def test_audit_reproduces_head_to_head_and_attributes_actions():
    from lanegame.rl import head_to_head
    T = audit_games(CFG, UNIFORM, TRADER, 25, 77)
    h = head_to_head(CFG, UNIFORM, TRADER, 25, 77)
    games = [T["reward"][(T["seat"] == s) & (T["ep"] == e)].sum() for s in (0, 1) for e in range(25)]
    assert np.mean(games) == pytest.approx(h["mean"], abs=1e-12)
    first = T.where(T["k"] == 0)
    assert np.allclose(np.sort(first["ret_to_go"]), np.sort(games))   # return-to-go at turn 0 = the game's payoff
    # action frequencies: the audit's realized behavior shares equal head_to_head's (EpisodeStats action counts)
    beh = h["x_stats"]["behaviors"]
    for code, letter in ((B_HOLD, "H"), (B_TRADE, "T"), (B_SAFE, "S")):
        assert np.mean(T["beh"] == code) == pytest.approx(beh.get(letter, 0.0), abs=1e-12)
    # the present/absent attribution matches the ABSENT code
    assert np.array_equal(~T["present"], T["a"] == CHAMP_ABSENT)
    lg = legality_table(T)
    assert lg["Hold"]["legal_share"] == 1.0 and 0.0 <= lg["Recall"]["legal_share"] <= 1.0
    # the uniform policy's probability of a category equals its legal share of codes
    pres = T.where(T["present"])
    assert np.allclose(pres["p_Hold"] + pres["p_Yield"] + pres["p_Trade"] + pres["p_Safe"] + pres["p_Recall"], 1.0)


def test_audit_reward_decomposition_and_summary():
    T = audit_games(CFG, FARMER, TRADER, 10, 5)
    decomposed = (T["farm"] - T["efarm"]) + (T["plates"] - T["eplates"]) + (T["bounty"] - T["ebounty"])
    assert np.allclose(T["reward"], decomposed)
    S = summarize(T)
    assert S["economy"]["per_game"]["reward"] == pytest.approx(T["reward"].sum() / 20)
    lost = sum(S["economy"]["farm_lost_by_mode"].values())
    assert lost == pytest.approx((T["avail"] - T["farm"]).sum() / 20)
    assert S["states"]["games"] == 20


def test_audit_assigns_skills_to_players_not_seats():
    T = audit_games(CFG, FARMER, TRADER, 3, 1, skill=(0.2, 0.8))
    assert np.all(T["skill"] == 0.2) and np.all(T["eskill"] == 0.8)


# ---------------------------------------------------------------------------------------------- experiment helpers
def test_rule_overlay_plays_the_rule_where_it_fires_and_the_base_policy_elsewhere():
    from lanegame.experiments.rl.diagnostics import RuleOverlay, overlay_rules
    rules = overlay_rules()
    ov = RuleOverlay(UNIFORM, rules["recall_hp2"], "u+recall")
    env = LaneEnv(CFG)
    oa, _ = env.reset(seed=0)
    assert np.allclose(ov.distribution(oa), UNIFORM.distribution(oa))          # full HP: base policy
    low = Snapshot(env.state, env.champ, 0, (0.5, 0.5)).with_side(0, hp=2)
    ob = observation(CFG, low, 0)
    p = ov.distribution(ob)
    assert p.sum() == 1.0 and p[ob.legal.index(RECALL)] == 1.0                # at its tower: Recall is legal
    mid = Snapshot(env.state, ChampState(ChampSide(0, 2, 0), env.champ.b), 3, (0.5, 0.5))
    om = observation(CFG, mid, 0)
    assert RECALL not in om.legal and ov.distribution(om)[om.legal.index(champ_act(SP, B_SAFE))] == 1.0
    stats = run_episodes(env, ov, TRADER, 5, 3)
    assert np.isfinite(stats.mean)


def test_scenario_plans_stay_legal_when_the_champion_dies_or_recalls_mid_plan():
    """Regression: the walk-back-and-recall plan once kept choosing Safe after its champion died on the way back.
    Every scenario plan must hand over (None) or choose a legal action on any turn, including absent turns."""
    from lanegame.experiments.rl.diagnostics import _plans
    snap = _mid_snapshot(seed=2, k=5)[0]
    away = snap.with_side(0, pos=-CFG.tower, hp=0, absent=CFG.death_turns)
    lane = snap.with_side(0, pos=0, hp=1)
    for sc in ("low_hp_lane", "enemy_away", "siege"):
        for name, plan in _plans(sc, CFG).items():
            for t in range(4):
                for sn in (away, lane):
                    ob = observation(CFG, sn, 0)
                    a = plan.act(ob, t)
                    assert a is None or a == DEFER or a in ob.legal, (sc, name, t, a, ob.legal)
            assert plan.act(observation(CFG, away, 0), 1) is None
    env = LaneEnv(CFG)
    for m in range(20):                                                       # and whole rollouts run through
        rollout(env, lane, 0, _plans("low_hp_lane", CFG)["back_recall"], (UNIFORM, TRADER), rollout_seed(9, m))


def test_diagnostics_seed_roots_are_disjoint_from_training_and_earlier_evaluation():
    from lanegame.experiments.rl import diagnostics as D
    from lanegame.experiments.rl import robustness as R
    from lanegame.experiments.rl import openai_five_eval as E
    ours = {D.AUDIT_ROOT, D.AUDIT_ROOT + 1, D.AUDIT_ROOT + 2, D.HP_ROOT, D.CF_STATE_ROOT, D.CF_ROLL_ROOT,
            D.OVERLAY_ROOT, D.OVERLAY_ROOT + 1, R.EVAL_ROOT}
    earlier = {10_000, 20_000, 30_000, 30_001, E.EVAL_SEED, E.CURVE_SEED, E.BR_SEED, E.DETAIL_SEED, 77_777}
    training = set(range(0, 10)) | set(range(200, 203)) | set(range(500, 503)) | set(range(700, 703)) | \
        set(range(900, 913))
    assert not ours & earlier and not ours & training
    assert min(ours) >= 50_000
    # the per-(scenario, policy) roots of the counterfactual stage (root + 100 * scenario + policy) never reach the
    # next stage's root
    top = 100 * (len(D.SCENARIOS) - 1) + len(D.policies()) - 1
    assert D.CF_STATE_ROOT + top < D.CF_ROLL_ROOT and D.CF_ROLL_ROOT + top < D.OVERLAY_ROOT


def test_phase_d_attacker_plan_and_paired_difference():
    from lanegame.experiments.rl import robustness as R
    att = R.attackers()
    assert len(att) == 9 * len(R.WARM_SEEDS) + len(R.SCRATCH_TARGETS)
    assert len({a[0] for a in att}) == len(att)                                # unique attacker ids / directories
    assert not {a[4] for a in att} & {900, 901, 902}                          # new training seeds
    assert all(s % 1024 == 0 for s in R.CURVE_STEPS) and R.CURVE_STEPS[-1] == R.BR_STEPS
    p = [1.0, 2.0, 3.0, 4.0]                               # 2 seeds: seat A games [1, 2], then seat B games [3, 4]
    q = [0.5, 2.0, 2.0, 5.0, 9.0, 9.0]                     # 3 seeds: seat A [0.5, 2, 2], seat B [5, 9, 9]
    m, h = R.paired_diff(p, q)                             # common seeds: the first 2 of each seat block
    assert m == pytest.approx(np.mean([0.5, 0.0, -2.0, -5.0]))
