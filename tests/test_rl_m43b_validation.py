"""The M4.3b follow-up: wave-only interventions (action composition, legality, both rules, Recall and absence),
the distribution-level plan hook of the diagnostics rollout (pairing, determinism, consistency with the engine), the
intervention agent in whole games, and the validation stage helpers. Network-free except where marked."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, PRESSURE, W_CLEAR, W_HOLD, W_THIN,
                             champ_act, champ_behavior, champ_objective, champ_wave)
from lanegame.diagnostics import DEFER, FollowPolicy, FunctionPlan, ProbCache, Snapshot, branch, observation, rollout
from lanegame.env import LaneEnv, Observation, play_episode
from lanegame.experiments.rl.m43b import intervene as iv
from lanegame.experiments.rl.m43b import validation as val
from lanegame.experiments.rl.m43b.references import reference_suite

CFG = GameConfig.m4_lasthit()
RECALL = champ_act(W_HOLD, B_RECALL)


class _Mixed:
    """A stochastic test policy whose behaviour is correlated with its wave: Clear + Trade (0.5), Hold + Safe (0.3),
    Thin + Hold (0.1), Recall (0.1, where legal), Pressure variants where legal share the Clear mass."""
    name = "mixed"

    def distribution(self, obs):
        p = np.zeros(len(obs.legal))
        if obs.legal == (CHAMP_ABSENT,):
            return np.ones(1)
        want = {champ_act(W_CLEAR, B_TRADE): 0.5, champ_act(W_HOLD, B_SAFE): 0.3, champ_act(W_THIN, B_HOLD): 0.1}
        if champ_act(W_CLEAR, B_TRADE, PRESSURE) in obs.legal:
            want[champ_act(W_CLEAR, B_TRADE, PRESSURE)] = 0.25
            want[champ_act(W_CLEAR, B_TRADE)] = 0.25
        if RECALL in obs.legal:
            want[RECALL] = 0.1
        for a, w in want.items():
            p[obs.legal.index(a)] = w
        return p / p.sum()

    def act(self, obs, rng=None):
        p = self.distribution(obs)
        from lanegame.env.agents import sample
        return obs.legal[sample(rng, p)] if rng is not None else obs.legal[int(np.argmax(p))]


def _states(n_turns=120, seed=5):
    """Observations of both seats along a reference game (the legal sets a present champion meets)."""
    suite = reference_suite(CFG)
    env = LaneEnv(CFG, max_turns=n_turns, cache_size=5000)
    obs = env.reset(seed=seed)
    out = []
    for _ in range(n_turns):
        out += [obs[0], obs[1]]
        r = env.step(suite["push_punish"].act(obs[0], None), suite["crash_recall"].act(obs[1], None))
        if r.done:
            break
        obs = r.obs
    return out


# ---------------------------------------------------------------------------------------------- the rules
def test_wave_replacement_is_legal_on_every_legal_set_met_in_play():
    seen = set()
    for o in _states():
        seen.add(o.legal)
        assert iv.wave_replacement_is_legal(o.legal)
    assert len(seen) >= 3                                             # with / without Pressure, with / without Recall, absent


@pytest.mark.parametrize("rule", iv.RULES)
def test_intervene_is_a_distribution_forces_the_wave_and_keeps_recall_and_absence(rule):
    pol = _Mixed()
    n_recall = n_pressure = 0
    for o in _states():
        p = pol.distribution(o)
        for w in (W_HOLD, W_THIN, W_CLEAR):
            q = iv.intervene(p, o.legal, w, rule)
            assert q.shape == p.shape and q.sum() == pytest.approx(1.0) and (q >= 0).all()
            if o.legal == (CHAMP_ABSENT,):
                assert q[0] == 1.0
                continue
            for a, qa in zip(o.legal, q):
                if qa > 0 and a != RECALL:
                    assert champ_wave(a) == w                         # every non-Recall mass sits on the forced wave
            if RECALL in o.legal:
                n_recall += 1
                assert q[o.legal.index(RECALL)] == pytest.approx(p[o.legal.index(RECALL)])
            if champ_act(W_CLEAR, B_TRADE, PRESSURE) in o.legal:
                n_pressure += 1
    assert n_recall > 0 and n_pressure > 0
    assert iv.intervene(np.ones(3) / 3, (0, 1, 2), None, rule).tolist() == [1 / 3] * 3     # None: unchanged


def test_marginal_rule_keeps_behaviour_and_objective_marginals():
    pol = _Mixed()
    for o in _states():
        if o.legal == (CHAMP_ABSENT,):
            continue
        p = pol.distribution(o)
        q = iv.intervene(p, o.legal, W_HOLD, "marginal")

        def marg(v, f):
            d = {}
            for a, x in zip(o.legal, v):
                if a != RECALL:
                    d[f(a)] = d.get(f(a), 0.0) + x
            return d
        for f in (champ_behavior, champ_objective):
            mp, mq = marg(p, f), marg(q, f)
            assert set(mp) == set(mq) and all(mp[k] == pytest.approx(mq[k]) for k in mp)
        # Clear + Trade (+ Pressure) became Hold + Trade (+ Pressure); Hold + Safe stays
        trade = sum(x for a, x in zip(o.legal, q) if champ_behavior(a) == B_TRADE)
        assert trade == pytest.approx(0.5 if RECALL in o.legal else 0.5 / 0.9)          # the policy's own Trade mass
        assert all(champ_wave(a) == W_HOLD for a, x in zip(o.legal, q) if x > 0 and champ_behavior(a) == B_TRADE)


def test_conditional_rule_renormalises_the_policys_own_slice():
    pol = _Mixed()
    for o in _states():
        if o.legal == (CHAMP_ABSENT,):
            continue
        p = pol.distribution(o)
        q = iv.intervene(p, o.legal, W_HOLD, "conditional")
        r = p[o.legal.index(RECALL)] if RECALL in o.legal else 0.0
        # the policy pairs Hold only with Safe: all the non-Recall mass goes there
        assert q[o.legal.index(champ_act(W_HOLD, B_SAFE))] == pytest.approx(1.0 - r)
        qc = iv.intervene(p, o.legal, W_CLEAR, "conditional")
        # Clear's slice: Trade (and Trade + Pressure where legal), in the policy's own proportions
        clear_mass = sum(x for a, x in zip(o.legal, p) if champ_wave(a) == W_CLEAR and a != RECALL)
        for a, x in zip(o.legal, p):
            if champ_wave(a) == W_CLEAR and a != RECALL:
                assert qc[o.legal.index(a)] == pytest.approx(x * (1 - r) / clear_mass)
    with pytest.raises(ValueError):
        iv.intervene(np.ones(2) / 2, (0, 1), W_HOLD, "nonsense")
    with pytest.raises(ValueError):
        iv.intervene(np.ones(3) / 3, (0, 1), W_HOLD, "marginal")


def test_wave_functions_read_the_state_from_the_observers_side():
    env = LaneEnv(CFG, max_turns=5)
    oa, ob = env.reset(seed=0)
    assert iv.WAVE_FNS["hold"](oa, 0) == W_HOLD and iv.WAVE_FNS["clear"](ob, 0) == W_CLEAR
    assert iv.slow_push_schedule(oa, 0) == W_THIN and iv.slow_push_schedule(oa, 2) == W_HOLD
    with pytest.raises(ValueError):
        iv.slow_push_schedule(oa, None)
    assert iv.freeze_controller(oa, 0) == W_HOLD and iv.shove_away(oa, 0) == W_HOLD      # no enemy front at s0, enemy present
    # a freeze-like lane from each side: the enemy front at own -1 with >= 14 HP -> Thin for that side only
    from lanegame.state import State
    lane = State(ax=-2, ah=8, ar=(), asg=0, bx=1, bh=16, br=(), bsg=0, va=0, vb=0, ap=0, bp=0)
    o_a = Observation(0, 0, lane, oa.legal, oa.vector, oa.mask, oa.champ, CFG)
    o_b = Observation(0, 1, lane, ob.legal, ob.vector, ob.mask, ob.champ, CFG)
    assert iv.freeze_controller(o_a, 0) == W_THIN and iv.freeze_controller(o_b, 0) == W_HOLD
    # the reactive slow push: in contact and not ahead by more than 2 HP -> Thin
    lane = State(ax=0, ah=8, ar=(), asg=0, bx=0, bh=8, br=(), bsg=0, va=0, vb=0, ap=0, bp=0)
    o_a = Observation(0, 0, lane, oa.legal, oa.vector, oa.mask, oa.champ, CFG)
    assert iv.reactive_slow_push(o_a, 0) == W_THIN
    lane = State(ax=0, ah=20, ar=(), asg=0, bx=0, bh=8, br=(), bsg=0, va=0, vb=0, ap=0, bp=0)
    o_a = Observation(0, 0, lane, oa.legal, oa.vector, oa.mask, oa.champ, CFG)
    assert iv.reactive_slow_push(o_a, 0) == W_HOLD
    assert iv.reactive_slow_push(o_b, 0) == W_HOLD                                          # no contact at s0


# ---------------------------------------------------------------------------------------------- the rollout hook
def _snap0():
    env = LaneEnv(CFG, max_turns=10)
    env.reset(seed=0)
    return Snapshot.from_env(env)


def _mid_snapshot(k=30, seed=4):
    suite = reference_suite(CFG)
    env = LaneEnv(CFG, max_turns=k + 1, cache_size=5000)
    log = play_episode(env, suite["push_punish"], suite["push_punish"], seed, record=True)
    return Snapshot.from_info(log.turns[k], (CFG.skill_a, CFG.skill_b))


def test_wave_plan_forces_only_the_wave_and_hands_back_after_its_window():
    pol = _Mixed()
    snap = _mid_snapshot()
    env = LaneEnv(CFG, cache_size=5000)
    for rule in iv.RULES:
        plan = iv.wave_plan("hold", rule, 6)
        r = rollout(env, snap, 0, plan, (pol, pol), 7, record=True, max_turns=12)
        acts = [i["actions"][0] for i in r.infos]
        for t, a in enumerate(acts[:6]):
            if a != CHAMP_ABSENT and a != RECALL:
                assert champ_wave(a) == W_HOLD                        # the wave is forced ...
                assert champ_behavior(a) in (B_HOLD, B_SAFE, B_TRADE)  # ... the behaviour is the policy's own
        base = rollout(env, snap, 0, FollowPolicy(), (pol, pol), 7, record=True, max_turns=12)
        assert any(champ_wave(a) == W_CLEAR for a in [i["actions"][0] for i in base.infos[:6]] if a not in (CHAMP_ABSENT, RECALL))
    # the plan is consulted on every turn of its window (also while absent) and then ends
    calls = []
    plan = iv.WavePlan("probe", lambda o, t: calls.append(t) or W_HOLD, 3, "marginal")
    rollout(env, snap, 0, plan, (pol, pol), 7, max_turns=6)
    assert calls == [0, 1, 2]


def test_transform_hook_keeps_one_draw_per_seat_per_turn_and_the_opponent_stream():
    """Candidates share the opponent's actions and chance until they diverge, as before the hook (pairing)."""
    pol = _Mixed()
    snap = _mid_snapshot()
    cands = {"policy": FollowPolicy(), "m:hold": iv.wave_plan("hold", "marginal", 4),
             "c:clear": iv.wave_plan("clear", "conditional", 4), "f:hold": FunctionPlan("f", lambda o, t: champ_act(W_HOLD, B_HOLD) if t < 4 else None)}
    r1 = branch(CFG, snap, 0, cands, (pol, pol), 6, root=21, key=(1,), max_turns=20)
    r2 = branch(CFG, snap, 0, cands, (pol, pol), 6, root=21, key=(1,), max_turns=20)
    for c in cands:
        assert r1.returns[c].tolist() == r2.returns[c].tolist()        # deterministic
    # the opponent's branch-turn action does not depend on the focal candidate
    for c in cands:
        assert r1.opp_first[c].tolist() == r1.opp_first["policy"].tolist()
    # an identity transform reproduces the policy rollout exactly (same uniform, same inverse-CDF)
    ident = iv.WavePlan("id", lambda o, t: None, 10, "marginal")
    r3 = branch(CFG, snap, 0, {"policy": FollowPolicy(), "id": ident}, (pol, pol), 6, root=21, key=(1,), max_turns=20)
    assert r3.returns["id"].tolist() == r3.returns["policy"].tolist()
    # a rollout without a plan still reproduces play_episode (the hook is inert for FollowPolicy)
    env = LaneEnv(CFG, max_turns=15, cache_size=5000)
    log = play_episode(env, pol, pol, 9, record=True)
    rr = rollout(LaneEnv(CFG, cache_size=5000), _snap0(), 0, FollowPolicy(), (pol, pol), 9, record=True, max_turns=15)
    assert [i["actions"] for i in rr.infos] == [i["actions"] for i in log.turns]


def test_invalid_transforms_are_rejected():
    pol = _Mixed()
    bad = iv.WavePlan("bad", lambda o, t: W_HOLD, 3, "marginal")
    bad.transform = lambda obs, t, p: np.full_like(p, -1.0)
    with pytest.raises(ValueError):
        rollout(LaneEnv(CFG, cache_size=100), _snap0(), 0, bad, (pol, pol), 1, max_turns=3)
    bad.transform = lambda obs, t, p: p[:-1] if len(p) > 1 else p
    with pytest.raises(ValueError):
        rollout(LaneEnv(CFG, cache_size=100), _snap0(), 0, bad, (pol, pol), 1, max_turns=3)


def test_intervention_agent_matches_the_rule_and_plays_legal_games():
    pol = _Mixed()
    for rule in iv.RULES:
        ag = iv.InterventionAgent(pol, iv.WAVE_FNS["thin"], rule)
        env = LaneEnv(CFG, max_turns=60, cache_size=5000)
        obs = env.reset(seed=2)
        rng = np.random.default_rng(3)
        for _ in range(60):
            d = ag.distribution(obs[0])
            assert d.tolist() == iv.intervene(pol.distribution(obs[0]), obs[0].legal, W_THIN, rule).tolist()
            full = ag.probabilities(obs[0])
            assert full.sum() == pytest.approx(1.0) and full[list(obs[0].legal)].tolist() == d.tolist()
            a, b = ag.act(obs[0], rng), pol.act(obs[1], rng)
            assert a in obs[0].legal and (a in (CHAMP_ABSENT, RECALL) or champ_wave(a) == W_THIN)
            r = env.step(a, b)
            obs = r.obs
    with pytest.raises(ValueError):
        iv.InterventionAgent(pol, iv.WAVE_FNS["thin"], "bogus")


# ---------------------------------------------------------------------------------------------- stage helpers
def test_candidate_sets_and_summaries():
    c = val.candidates()
    assert set(c) == {"policy"} | {f"m:{k}" for k in val.MARGINAL} | {f"c:{k}" for k in val.CONDITIONAL} | {f"f:{k}" for k in val.FULL}
    assert all(isinstance(c[f"m:{k}"], iv.WavePlan) and c[f"m:{k}"].turns == val.PLAN_TURNS[k] for k in val.MARGINAL)
    assert "f:hold" not in val.candidates(include_full=False)
    states = [{"adv": {"m:hold": [-1.0, 0.1], "m:thin": [0.5, 0.1]},
               "decomposition": {"policy": {"farm_diff": 1.0, "deaths": [1, 1]}, "m:hold": {"farm_diff": 0.0, "deaths": [2, 1]},
                                 "m:thin": {"farm_diff": 1.5, "deaths": [1, 1]}}},
              {"adv": {"m:hold": [-2.0, 0.1], "m:thin": [0.1, 0.2]},
               "decomposition": {"policy": {"farm_diff": 1.0, "deaths": [1, 1]}, "m:hold": {"farm_diff": -1.0, "deaths": [2, 1]},
                                 "m:thin": {"farm_diff": 1.1, "deaths": [1, 1]}}}]
    s = val.summarize_states(states, ["m:hold", "m:thin"])
    assert s["n_states"] == 2 and s["m:hold"]["mean_adv"] == -1.5 and s["m:hold"]["share_neg"] == 1.0
    assert s["m:thin"]["share_pos"] == 0.5 and s["m:hold"]["farm_diff"] == pytest.approx(-1.5)
    results = {"final:base:0|self|neutral": {"summary": s}, "final:base:1|self|neutral": {"summary": s}}
    a = val.across_seeds(results, ["final:base:0", "final:base:1", "final:base:2"], lambda sp: f"{sp}|self|neutral", ["m:hold"])
    assert a["m:hold"]["n_seeds"] == 2 and a["m:hold"]["mean"] == -1.5


def test_held_turns_counts_the_enemy_front_at_own_minus_one():
    from lanegame.state import State
    held = State(ax=-2, ah=8, ar=(), asg=0, bx=1, bh=16, br=(), bsg=0, va=0, vb=0, ap=0, bp=0)
    crashed = State(ax=-2, ah=8, ar=(), asg=0, bx=2, bh=0, br=(), bsg=10, va=0, vb=0, ap=0, bp=0)
    infos = [{"state_after": held.to_list()}] * 3 + [{"state_after": crashed.to_list()}] * 2
    assert val.held_turns(infos, 0, watch=20) == 3 and val.held_turns(infos, 1, watch=20) == 0


def test_paired_values_are_reproducible_and_seat_paired():
    suite = reference_suite(CFG)
    v1 = val.paired_values(CFG, suite["push_punish"], suite["farm"], 3, 11, 40)
    v2 = val.paired_values(CFG, suite["push_punish"], suite["farm"], 3, 11, 40)
    assert v1 == v2 and len(v1["pair"]) == 3
    assert v1["pair"] == pytest.approx((0.5 * (np.array(v1["seat_A"]) + np.array(v1["seat_B"]))).tolist())


# ---------------------------------------------------------------------------------------------- with a network
torch = pytest.importorskip("torch")


def test_intervened_network_policy_is_consistent_with_the_engine_and_the_rules():
    from lanegame.experiments.rl.m43b.common import agent_from_spec
    agent = agent_from_spec("initial:0")                               # the untrained network: no checkpoint needed
    env = LaneEnv(CFG, max_turns=80, cache_size=5000)
    obs = env.reset(seed=1)
    rng = np.random.default_rng(0)
    ag = iv.InterventionAgent(agent, iv.freeze_controller, "conditional")
    for _ in range(80):
        p = agent.distribution(obs[0])
        w = iv.freeze_controller(obs[0], None)
        q = ag.distribution(obs[0])
        assert q.tolist() == iv.intervene(p, obs[0].legal, w, "conditional").tolist()
        a = ag.act(obs[0], rng)
        assert a in obs[0].legal
        r = env.step(a, agent.act(obs[1], rng))
        obs = r.obs
