"""The RL-ready environment (lanegame.env): legal masks, observations, seeding, zero-sum rewards, agreement
with the exact game, the exact-policy bridge, the older models' crash-choice stage, episode handling,
traces, the optional PettingZoo adapter and the CLI. The exact solver (session fixtures, K = 16) is the
correctness reference throughout; no new solves above K = 16 are run."""

import inspect
import itertools

import numpy as np
import pytest

from lanegame import GameConfig, LaneGame, SH_C, SP_S
from lanegame.analysis import path_statistics
from lanegame.baselines import (aggressive, always_shove, always_slowpush, defensive_freeze, freeze_contest,
                                passive_safe, prefer, stack_builder, uniform)
from lanegame.best_response import evaluate, exploitability
from lanegame.env import (TRACE_SCHEMA, TRACE_VERSION, AgentPolicy, GreedyObservationAgent, LaneEnv, ScriptedAgent,
                          as_policy, feature_names, legal_mask, n_actions, obs_size, observe, play_episode,
                          read_trace, run_episodes, write_trace)
from lanegame.policy import ScriptedPolicy
from lanegame.state import State

from conftest import all_transitions

PAIRS = {
    "v1": [("eq", "eq"), ("eq", uniform), ("uniform", always_shove), (defensive_freeze, always_slowpush),
           (always_shove, "eq")],
    "revised": [("eq", "eq"), ("eq", uniform), (aggressive, passive_safe), (freeze_contest, stack_builder),
                ("uniform", aggressive)],
    "siege": [("eq", "eq"), ("eq", uniform), (aggressive, passive_safe), (freeze_contest, stack_builder),
              (lambda: stack_builder(siege="vision"), passive_safe)],
}


def _resolve(p, solution, side):
    if p == "eq":
        return solution.policy(side)
    if p == "uniform":
        return uniform()
    return p()


# ------------------------------------------------------------------ A. legality
def test_mask_matches_engine_legal_actions(game, layers):
    """The mask is True exactly on the engine's legal set, at every reachable (k, state), for both seats."""
    cfg = game.cfg
    n = n_actions(cfg)
    env = LaneEnv(cfg)
    assert env.n_actions == n == max(cfg.action_codes()) + 1
    for k, layer in enumerate(layers):
        for s in layer:
            for side in (0, 1):
                m = legal_mask(s, side, cfg, n)
                assert m.shape == (n,) and m.dtype == bool
                assert tuple(int(a) for a in np.flatnonzero(m)) == game.legal(s, side)
    for k in (0, 3, 7, 12):
        for s in (layers[k][0], layers[k][len(layers[k]) // 2], layers[k][-1]):
            oa, ob = env.reset(seed=0, root=s, k0=k)
            assert oa.legal == game.legal(s, 0) and ob.legal == game.legal(s, 1)
            assert tuple(int(a) for a in np.flatnonzero(oa.mask)) == oa.legal
            assert tuple(int(a) for a in np.flatnonzero(ob.mask)) == ob.legal
            ma, mb = env.masks()
            assert np.array_equal(ma, oa.mask) and np.array_equal(mb, ob.mask)
            assert env.legal(0) == oa.legal and env.legal(1) == ob.legal


def test_illegal_actions_and_stepping_outside_an_episode_are_rejected(game):
    cfg = game.cfg
    env = LaneEnv(cfg)
    with pytest.raises(RuntimeError):
        env.step(0, 0)                                   # before reset
    oa, ob = env.reset(seed=1)
    illegal = next(a for a in range(env.n_actions + 1) if a not in oa.legal)
    with pytest.raises(ValueError):
        env.step(illegal, ob.legal[0])
    with pytest.raises(ValueError):
        env.step(oa.legal[0], illegal)
    with pytest.raises(ValueError):
        env.step(oa.legal[0], env.n_actions)             # out of range
    assert env.k == 0                                    # a rejected step changes nothing
    while True:
        r = env.step(oa.legal[0], ob.legal[0])
        if r.done:
            break
        oa, ob = r.obs
    assert r.terminated and env.k == cfg.horizon
    with pytest.raises(RuntimeError):
        env.step(oa.legal[0], ob.legal[0])               # after the end


# ------------------------------------------------------------------ B. observations
def test_observation_shape_bounds_determinism_and_mirror_equivariance(game, layers):
    """Fixed length, float32, in [0, 1], a function of (k, state, side), and B's view of s is A's view of
    mirror(s)."""
    cfg = game.cfg
    m = obs_size(cfg)
    assert len(feature_names(cfg)) == m
    for k, layer in enumerate(layers):
        for s in layer:
            va = observe(s, k, 0, cfg)
            vb = observe(s, k, 1, cfg)
            assert va.shape == vb.shape == (m,) and va.dtype == np.float32
            assert np.all(va >= 0) and np.all(va <= 1) and np.all(vb >= 0) and np.all(vb <= 1)
            assert np.array_equal(va, observe(s, k, 0, cfg))
            assert np.array_equal(vb, observe(s.mirror(), k, 0, cfg))
            assert np.array_equal(va, observe(s.mirror(), k, 1, cfg))


def test_observation_encodes_the_state(cfg):
    idx = {n: i for i, n in enumerate(feature_names(cfg))}
    env = LaneEnv(cfg)
    oa, ob = env.reset(seed=0)
    v = oa.vector
    assert v[idx["turn_fraction"]] == 0 and v[idx["turns_remaining_fraction"]] == 1 and v[idx["phase_0"]] == 1
    assert v[idx[f"own_minions_x{-cfg.base:+d}"]] == pytest.approx(cfg.wave_hp / cfg.hp_cap)
    assert v[idx[f"enemy_minions_x{cfg.base:+d}"]] == pytest.approx(cfg.wave_hp / cfg.hp_cap)
    assert v[idx["own_has_front"]] == 1 and v[idx["enemy_has_front"]] == 1
    assert v[idx["own_siege_mass"]] == 0 and v[idx["own_vision"]] == 0 and v[idx["own_siege_eligible"]] == 0
    assert np.array_equal(oa.vector, ob.vector)          # the standard start is its own mirror image
    if cfg.objectives:
        s = State(-4, 4, (), 8, -3, 4, (), 0, 3, 0, 2, 0)   # A's 8-HP mass under B's tower, vision 3, 2 plates
        va, vb = observe(s, 5, 0, cfg), observe(s, 5, 1, cfg)
        assert va[idx["own_siege_mass"]] == pytest.approx(8 / cfg.siege_cap)
        assert va[idx["own_vision"]] == pytest.approx(3 / cfg.vision_turns)
        assert va[idx["own_plates_pending"]] == pytest.approx(2 / cfg.plates_cap)
        assert va[idx["own_siege_eligible"]] == 1 and va[idx["enemy_siege_eligible"]] == 0
        assert vb[idx["enemy_siege_mass"]] == pytest.approx(8 / cfg.siege_cap) and vb[idx["own_siege_eligible"]] == 0
        assert va[idx["turn_fraction"]] == pytest.approx(5 / cfg.horizon) and va[idx[f"phase_{5 % cfg.phase_period}"]] == 1
        # B's waiting wave at B's own -3 is at A's own +3
        assert va[idx["enemy_minions_x+3"]] == pytest.approx(4 / cfg.hp_cap) and vb[idx["own_minions_x-3"]] == pytest.approx(4 / cfg.hp_cap)


def test_observations_never_depend_on_the_current_turns_actions(game, layers):
    """Both observations of a turn are computed from (k, state) before either action is read: B sees the
    same thing whatever A is about to play, and `step` has no one-action form."""
    cfg = game.cfg
    required = [p.name for p in inspect.signature(LaneEnv.step).parameters.values()
                if p.default is inspect.Parameter.empty and p.name != "self"]
    assert required == ["action_a", "action_b"]
    env1, env2 = LaneEnv(cfg), LaneEnv(cfg)
    o1, o2 = env1.reset(seed=3), env2.reset(seed=3)
    la, lb = o1[0].legal, o1[1].legal
    assert np.array_equal(o1[1].vector, o2[1].vector) and np.array_equal(o1[0].vector, o2[0].vector)
    r1, r2 = env1.step(la[0], lb[0]), env2.step(la[-1], lb[0])
    # the next observation is a function of the next (k, state) only
    for r in (r1, r2):
        assert np.array_equal(r.obs[0].vector, observe(r.obs[0].state, r.obs[0].k, 0, cfg))
        assert np.array_equal(r.obs[1].vector, observe(r.obs[1].state, r.obs[1].k, 1, cfg))
    k, s = 4, layers[4][len(layers[4]) // 3]
    oa, ob = LaneEnv(cfg).reset(seed=0, root=s, k0=k)
    assert np.array_equal(oa.vector, observe(s, k, 0, cfg)) and np.array_equal(ob.vector, observe(s, k, 1, cfg))


# ------------------------------------------------------------------ C. reproducibility and random streams
def test_episodes_are_reproducible_and_seed_sensitive(game):
    env = LaneEnv(game.cfg)
    a, b = ScriptedAgent(uniform()), ScriptedAgent(uniform())
    l1 = play_episode(env, a, b, seed=11, record=True)
    l2 = play_episode(env, a, b, seed=11, record=True)
    l3 = play_episode(env, a, b, seed=12, record=True)
    assert l1.actions == l2.actions and l1.rewards == l2.rewards and l1.ganks == l2.ganks
    assert [t["gank_draws"] for t in l1.turns] == [t["gank_draws"] for t in l2.turns]
    assert l1.actions != l3.actions or l1.rewards != l3.rewards
    s1 = run_episodes(env, a, b, 40, seed=5)
    s2 = run_episodes(env, a, b, 40, seed=5)
    assert np.array_equal(s1.payoffs, s2.payoffs) and np.array_equal(s1.ganks, s2.ganks)


def test_chance_stream_is_independent_of_actions_and_agents(game):
    """Common random numbers: the environment's uniform gank draws depend on its seed and the turn only,
    never on the actions played or on the random numbers the agents consume; and each agent's stream is
    independent of the other agent."""
    cfg = game.cfg
    env = LaneEnv(cfg)

    def rollout(pick):
        oa, ob = env.reset(seed=99)
        draws = []
        while True:
            r = env.step(pick(oa.legal), pick(ob.legal))
            draws.append(r.info["gank_draws"])
            if r.done:
                return draws
            oa, ob = r.obs

    assert rollout(lambda legal: legal[0]) == rollout(lambda legal: legal[-1])
    det = play_episode(env, ScriptedAgent(aggressive()), ScriptedAgent(passive_safe()), seed=4, record=True)
    rnd = play_episode(env, ScriptedAgent(uniform()), ScriptedAgent(uniform()), seed=4, record=True)
    assert [t["gank_draws"] for t in det.turns] == [t["gank_draws"] for t in rnd.turns]
    # A's first draw (same state, same stream) is unchanged when B's agent changes
    u1 = play_episode(env, ScriptedAgent(uniform()), ScriptedAgent(passive_safe()), seed=4)
    u2 = play_episode(env, ScriptedAgent(uniform()), ScriptedAgent(uniform()), seed=4)
    assert u1.actions[0][0] == u2.actions[0][0]


# ------------------------------------------------------------------ D. rewards
def test_rewards_are_zero_sum_and_decompose(game):
    cfg = game.cfg
    env = LaneEnv(cfg)
    a, b = ScriptedAgent(uniform()), ScriptedAgent(uniform())
    for seed in range(4):
        total = 0.0
        oa, ob = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        while True:
            r = env.step(a.act(oa, rng), b.act(ob, rng))
            i = r.info
            assert r.rewards[0] == -r.rewards[1] and r.rewards[0] == i["reward"]
            assert r.rewards[0] == pytest.approx(i["farm_payoff"] + i["plate_payoff"] + i["gank_payoff"] + i["crash_payoff"], abs=1e-12)
            assert i["farm_payoff"] == i["farm"][0] - i["farm"][1]
            assert i["plate_payoff"] == i["plates"][0] - i["plates"][1]
            assert i["gank_payoff"] == cfg.gank_loss * (int(i["ganked"][1]) - int(i["ganked"][0]))
            assert all(0 <= p <= cfg.gank_max for p in i["gank_p"]) and all(0 <= u < 1 for u in i["gank_draws"])
            total += r.rewards[0]
            if r.done:
                break
            oa, ob = r.obs
        log = play_episode(env, a, b, seed=seed)
        assert abs(sum(log.rewards) - log.payoff) < 1e-9


def test_step_reproduces_lane_game_transition(game, layers):
    """The deterministic parts of a step (lane successor, farm, plates, gank probabilities, crash options,
    exposure) are exactly LaneGame.transition's, with and without the bounded cache."""
    cfg = game.cfg
    trans = list(itertools.islice(all_transitions(game, layers), 0, None, 5))
    rng = np.random.default_rng(0)
    for env in (LaneEnv(cfg), LaneEnv(cfg, cache_size=64)):
        for idx in rng.choice(len(trans), size=min(300, len(trans)), replace=False):
            k, s, a, b = trans[idx]
            tr = game.transition(s, k, a, b)
            env.reset(seed=0, root=s, k0=k)
            info = env.step(a, b).info
            assert info["farm"] == tr.farm and info["farm_available"] == tr.farm_available
            assert info["plates"] == tr.plates and info["gank_p"] == tr.gank_p
            assert info["farm_payoff"] == tr.farm_payoff and info["plate_payoff"] == tr.plate_payoff
            assert State.from_list(info["state_pre_choice"]) == tr.pre_state
            assert info["crash_hp"] == tr.crash_hp and info["crash_options"] == (tr.options_a, tr.options_b)
            assert info["exposure"] == tr.exposure and info["objectives"] == tr.objectives
            if cfg.objectives:
                assert State.from_list(info["state_after"]) == tr.pre_state      # no choice stage


# ------------------------------------------------------------------ E. agreement with the exact game
@pytest.mark.parametrize("i", range(5))
def test_mean_reward_matches_exact_expected_value(model, game, layers, solution, i):
    """Realized rewards over many episodes agree with the exact expected value of the policy pair
    (4.5 standard errors: per-test false-alarm probability below 1e-5)."""
    pa, pb = _resolve(PAIRS[model][i][0], solution, 0), _resolve(PAIRS[model][i][1], solution, 1)
    exact = evaluate(game, layers, pa, pb)
    st = run_episodes(LaneEnv(game.cfg), ScriptedAgent(pa), ScriptedAgent(pb), 2000, seed=1000 + i)
    assert st.episodes == 2000 and st.turns == 2000 * game.horizon
    assert abs(st.mean - exact) <= 4.5 * st.stderr + 1e-12, (exact, st.mean, st.stderr)


# ------------------------------------------------------------------ F. deterministic agreement
@pytest.mark.parametrize("base", [GameConfig.v1(horizon=16), GameConfig.v2(horizon=16), GameConfig(horizon=16)],
                         ids=["v1", "revised", "siege"])
def test_deterministic_pairs_without_ganks_equal_the_exact_value(base):
    """With every gank parameter zero AND deterministic policies and crash choices (the scripted baselines
    play pure strategies and take the greedy crash option), every episode equals the exact value and records
    exactly the exact path's farm, plates and crash resources. With a randomizing player only the mean would
    agree, so this test uses deterministic pairs only."""
    cfg = base.with_(gank_base=0.0, gank_position=0.0, gank_action=(0.0,) * 6, gank_strength=0.0)
    g = LaneGame(cfg)
    layers = g.reachable()
    pairs = ([(always_shove(), always_slowpush()), (always_slowpush(), defensive_freeze())] if not cfg.positioning
             else [(aggressive(), passive_safe()), (stack_builder(), freeze_contest()),
                   (stack_builder(siege="vision"), passive_safe())])
    env = LaneEnv(cfg)
    for pa, pb in pairs:
        st = run_episodes(env, ScriptedAgent(pa), ScriptedAgent(pb), 5, seed=0, record_first=True)
        ps = path_statistics(g, pa, pb)
        assert np.all(st.payoffs == st.payoffs[0])
        assert abs(st.payoffs[0] - evaluate(g, layers, pa, pb)) < 1e-12
        assert np.allclose(st.farm[0], ps.farm, atol=1e-12)
        assert np.allclose(st.plates[0], ps.plates, atol=1e-12)
        assert np.allclose(st.crash_resources[0], ps.crash_resources, atol=1e-12)
        assert not any(t["ganked"][0] or t["ganked"][1] for t in st.first_log.turns)
        assert not st.choice_default_used


# ------------------------------------------------------------------ G. the exact-policy bridge
def test_scripted_agent_round_trips_to_its_policy(game, baselines):
    for p in baselines:
        assert as_policy(ScriptedAgent(p), game.cfg) is p


def test_observation_agent_is_exactly_evaluable(model, game, layers, solution, baselines):
    """A deterministic observation-based agent, wrapped as a Policy, has a one-hot distribution at every
    reachable node that equals the action it takes in the environment; its exact value against a scripted
    opponent agrees with the environment's realized mean; its exploitability is a well-defined non-negative
    number; and against the equilibrium it earns at most the equilibrium value."""
    cfg = game.cfg
    agent = GreedyObservationAgent.random(cfg, seed=7)
    pol = as_policy(agent, cfg)
    assert isinstance(pol, AgentPolicy)
    env = LaneEnv(cfg)
    for k, layer in enumerate(layers):
        for s in layer[::max(1, len(layer) // 40)]:
            for side in (0, 1):
                la = game.legal(s, side)
                p = pol.move(k, s, side, la)
                assert p.shape == (len(la),) and p.sum() == 1.0 and np.count_nonzero(p) == 1
                obs = env.reset(seed=0, root=s, k0=k)[side]
                assert la[int(np.argmax(p))] == agent.act(obs, None)
    opp = baselines[3]
    exact = evaluate(game, layers, pol, opp)
    st = run_episodes(env, agent, ScriptedAgent(opp), 2000, seed=77)
    assert abs(st.mean - exact) <= 4.5 * st.stderr + 1e-12, (exact, st.mean, st.stderr)
    e = exploitability(game, layers, pol, 0, solution.value)
    assert e.exploitability >= -1e-9
    assert evaluate(game, layers, pol, solution.policy(1)) <= solution.value + 1e-9
    # the equilibrium itself, played through an agent, is exactly the equilibrium policy
    assert abs(evaluate(game, layers, as_policy(ScriptedAgent(solution.policy(0)), cfg), solution.policy(1)) - solution.value) < 1e-9


def test_sampling_only_agent_plays_but_is_not_exactly_evaluable(game):
    class Sampler:
        name = "sampler"

        def act(self, obs, rng):
            return obs.legal[int(rng.integers(len(obs.legal)))]

    with pytest.raises(TypeError):
        as_policy(Sampler(), game.cfg)
    log = play_episode(LaneEnv(game.cfg), Sampler(), ScriptedAgent(uniform()), seed=0)
    assert log.length == game.horizon


def test_choice_stage_uses_the_agents_own_decisions(rev_game):
    """"choice" siege mode (previous revision): the crash-reward decision is the agents' (`choose`), not the
    greedy default. From a pre-crash root (no scripted pair crashes from the standard start within 16 turns)
    a shover that always takes the vision option is reproduced in expectation and in its crash-resource
    accounting; without choosers the environment falls back to the greedy default and says so."""
    cfg = rev_game.cfg
    root, k0 = State(1, 8, (), 0, -3, 4, (), 0, 0, 0), 1      # A's 2-wave packet one cell from B's tower
    layers = rev_game.reachable(root=root, k0=k0)

    def take_vision(k, st, opts_self, opts_opp):
        p = np.zeros(len(opts_self))
        vis = [i for i, o in enumerate(opts_self) if o[1]]
        p[vis[0] if vis else int(np.argmax([o[0] for o in opts_self]))] = 1.0
        return p

    vision_shover = ScriptedPolicy("aggressive_vision", lambda k, st, legal: prefer(legal, SH_C), take_vision)
    opp = passive_safe()
    exact = evaluate(rev_game, layers, vision_shover, opp, root=root, k0=k0)
    ps = path_statistics(rev_game, vision_shover, opp, root=root, k0=k0)
    assert ps.vision_choices[0] > 0                       # the path does contain vision choices
    greedy_exact = evaluate(rev_game, layers, aggressive(), opp, root=root, k0=k0)
    assert abs(exact - greedy_exact) > 1e-6               # and the choice matters for the value
    st = run_episodes(LaneEnv(cfg), ScriptedAgent(vision_shover), ScriptedAgent(opp), 1500, seed=3,
                      record_first=True, root=root, k0=k0)
    assert not st.choice_default_used
    assert abs(st.mean - exact) <= 4.5 * st.stderr + 1e-12, (exact, st.mean, st.stderr)
    assert np.all(st.crash_resources[:, 0] == ps.crash_resources[0])    # deterministic given the actions
    assert any(t["choices"][0] == 1 for t in st.first_log.turns)        # the vision option was taken
    # the environment without choosers: the greedy default is used and reported
    env = LaneEnv(cfg)
    oa, ob = env.reset(seed=0, root=root, k0=k0)
    seen = False
    while True:
        a = oa.legal[int(np.argmax(prefer(oa.legal, SH_C)))]
        b = ob.legal[int(np.argmax(prefer(ob.legal, SP_S)))]
        r = env.step(a, b)
        opts, (i, _) = r.info["crash_options"][0], r.info["choices"]
        if len(opts) > 1:
            seen = True
            assert r.info["choice_default_used"] and opts[i][0] == max(o[0] for o in opts)
        if r.done:
            break
        oa, ob = r.obs
    assert seen


# ------------------------------------------------------------------ H. episode handling
def test_episode_handling(game):
    cfg = game.cfg
    K = cfg.horizon
    env = LaneEnv(cfg, max_turns=5)
    oa, ob = env.reset(seed=0)
    assert oa.k == ob.k == 0 and (oa.side, ob.side) == (0, 1) and oa.state == ob.state == game.initial_state()
    assert env.k == 0 and env.k0 == 0 and not env.done and env.horizon == K
    n = 0
    while True:
        r = env.step(oa.legal[0], ob.legal[0])
        n += 1
        if r.done:
            break
        oa, ob = r.obs
    assert n == 5 and r.truncated and not r.terminated and env.k == 5 and env.done
    with pytest.raises(RuntimeError):
        env.step(oa.legal[0], ob.legal[0])
    env2 = LaneEnv(cfg)
    oa, ob = env2.reset(seed=0, root=game.initial_state(), k0=K - 2)
    r = env2.step(oa.legal[0], ob.legal[0])
    assert not r.done and r.obs[0].k == K - 1
    r = env2.step(r.obs[0].legal[0], r.obs[1].legal[0])
    assert r.terminated and not r.truncated and env2.done
    with pytest.raises(ValueError):
        env2.reset(seed=0, k0=K)
    with pytest.raises(ValueError):
        LaneEnv(cfg, max_turns=0)
    oa, ob = env2.reset(seed=0)
    assert env2.k == 0 and not env2.done
    log = play_episode(env2, ScriptedAgent(uniform()), ScriptedAgent(uniform()), seed=0)
    assert log.length == K and len(log.actions) == K


# ------------------------------------------------------------------ traces
def test_trace_round_trip(game, tmp_path):
    cfg = game.cfg
    env = LaneEnv(cfg)
    log = play_episode(env, ScriptedAgent(aggressive()), ScriptedAgent(uniform()), seed=5, record=True)
    path = tmp_path / "trace.json"
    write_trace(log, cfg, str(path))
    d = read_trace(str(path))
    assert d["schema"] == TRACE_SCHEMA and d["version"] == TRACE_VERSION and d["config_obj"] == cfg
    assert d["n_actions"] == env.n_actions and d["obs_size"] == env.obs_size and len(d["features"]) == env.obs_size
    ep = d["episode"]
    assert len(ep["turns"]) == len(ep["rewards"]) == len(ep["actions"]) == cfg.horizon
    assert abs(sum(ep["rewards"]) - ep["payoff"]) < 1e-9 and ep["names"] == ["aggressive", "uniform"]
    assert State.from_list(ep["turns"][0]["state_before"]) == game.initial_state()
    for t, u in zip(ep["turns"], ep["turns"][1:]):
        assert t["state_after"] == u["state_before"] and u["k"] == t["k"] + 1
    for t in ep["turns"]:
        assert t["gank_payoff"] == cfg.gank_loss * (int(t["ganked"][1]) - int(t["ganked"][0]))
        assert t["actions"][0] in t["legal"][0] and t["actions"][1] in t["legal"][1]
        assert set(t["events"]) >= {"spawned", "died", "crashed", "killed_hp", "tower_hp"}
    assert ep["ganks"] == [sum(t["ganked"][0] for t in ep["turns"]), sum(t["ganked"][1] for t in ep["turns"])]


# ------------------------------------------------------------------ I. PettingZoo
def test_pettingzoo_parallel_api(cfg):
    pytest.importorskip("pettingzoo")
    pytest.importorskip("gymnasium")
    from pettingzoo.test import parallel_api_test
    from lanegame.env.pettingzoo_adapter import LaneParallelEnv
    env = LaneParallelEnv(cfg)
    parallel_api_test(env, num_cycles=60)
    obs, infos = env.reset(seed=0)
    assert set(obs) == {"A", "B"} == set(infos) and env.agents == ["A", "B"]
    assert env.observation_space("A").contains(obs["A"]) and env.action_space("B").n == n_actions(cfg)
    a = int(np.flatnonzero(obs["A"]["action_mask"])[0])
    b = int(np.flatnonzero(obs["B"]["action_mask"])[-1])
    o2, rewards, terms, truncs, infos = env.step({"A": a, "B": b})
    assert rewards["A"] == -rewards["B"] and not terms["A"] and not truncs["A"] and infos["A"]["actions"] == (a, b)
    assert isinstance(env.render(), str) and env.state().shape == (obs_size(cfg),)


# ------------------------------------------------------------------ J. CLI
def test_env_demo_cli(capsys):
    from lanegame.__main__ import main
    main(["env-demo", "--horizon", "8", "--episodes", "5", "--seed", "1"])
    out = capsys.readouterr().out
    assert "episodes/s" in out and "turns/s" in out and "observation size 34" in out and "aggressive (A) vs passive_safe (B)" in out
    main(["env-demo", "--model", "v1", "--a", "always_shove", "--b", "defensive_freeze", "--horizon", "8",
          "--episodes", "5", "--seed", "1", "--exact"])
    out = capsys.readouterr().out
    assert "exact expected payoff" in out and "observation size 32" in out
