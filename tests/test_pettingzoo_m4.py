"""M4.3a: the PettingZoo Parallel API adapter on the M4 presets (GameConfig.m4_lasthit() in particular).

PettingZoo's own API validation on every M4 preset and on the AEC conversion; spaces, masks and dtypes at every step;
trajectory equality with the native LaneEnv under identical seeds and actions; `play_episode` reproduced through the
adapter; custom start states (StartState and list forms, M4State under ruleset m4); the lossless global state.
Skipped without pettingzoo / gymnasium."""

import numpy as np
import pytest

pz = pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

from pettingzoo.test import api_test, parallel_api_test                              # noqa: E402

from lanegame import GameConfig                                                       # noqa: E402
from lanegame.champions_m4 import M4State                                             # noqa: E402
from lanegame.env import LaneEnv, UniformAgent, play_episode                          # noqa: E402
from lanegame.env.agents import episode_streams                                       # noqa: E402
from lanegame.env.agents_m4 import natural_suite                                      # noqa: E402
from lanegame.env.observation_m4 import decode_m4                                     # noqa: E402
from lanegame.env.pettingzoo_adapter import LaneParallelEnv                           # noqa: E402
from lanegame.env.starts import StartState                                            # noqa: E402

LASTHIT = GameConfig.m4_lasthit()
PRESETS = {"m4": GameConfig.m4(), "m4_emergent": GameConfig.m4_emergent(), "m4_natural": GameConfig.m4_natural(),
           "m4_lane": GameConfig.m4_lane(), "m4_lasthit": LASTHIT}


def uniform_actions(obs, rng):
    return {ag: int(rng.choice(np.flatnonzero(obs[ag]["action_mask"]))) for ag in obs}


@pytest.mark.parametrize("name", list(PRESETS))
def test_official_parallel_api_test_passes_on_every_m4_preset(name):
    parallel_api_test(LaneParallelEnv(PRESETS[name], max_turns=50), num_cycles=40)


def test_official_aec_api_test_passes_through_the_conversion():
    api_test(LaneParallelEnv(LASTHIT, max_turns=30).aec(), num_cycles=60)


def test_spaces_masks_dtypes_and_infos_at_every_step():
    env = LaneParallelEnv(LASTHIT, max_turns=120)
    rng = np.random.default_rng(3)
    obs, infos = env.reset(seed=7)
    assert env.agents == ["A", "B"] and env.num_agents == 2 and env.max_num_agents == 2
    assert infos["A"]["legal"] == env.env.legal(0) and infos["B"]["legal"] == env.env.legal(1)
    for t in range(120):
        for s, ag in enumerate("AB"):
            o = obs[ag]
            assert env.observation_space(ag).contains(o)
            assert o["observation"].dtype == np.float32 and o["action_mask"].dtype == np.int8
            assert set(np.flatnonzero(o["action_mask"])) == set(env.env.legal(s))
            assert env.action_space(ag).contains(env.action_space(ag).sample(mask=o["action_mask"]))
        obs, rew, term, trunc, infos = env.step(uniform_actions(obs, rng))
        assert rew["A"] == -rew["B"] and isinstance(rew["A"], float)
        assert isinstance(term["A"], bool) and isinstance(trunc["A"], bool) and not term["A"]
        assert infos["A"] is infos["B"] and infos["A"]["k"] == t
        assert trunc["A"] == (t == 119)
    assert env.agents == []
    with pytest.raises(RuntimeError):
        env.step({"A": 0, "B": 0})


def test_trajectory_equals_the_native_environment_under_identical_seed_and_actions():
    """The adapter is a lossless view: observations, masks, rewards, flags and the turn record agree exactly with a
    LaneEnv given the same chance seed and the same action sequence."""
    T = 300
    native = LaneEnv(LASTHIT, max_turns=T)
    adapter = LaneParallelEnv(LASTHIT, max_turns=T)
    rng = np.random.default_rng(11)
    oa, ob = native.reset(seed=2024)
    obs, _ = adapter.reset(seed=2024)
    for t in range(T):
        assert np.array_equal(obs["A"]["observation"], oa.vector) and np.array_equal(obs["B"]["observation"], ob.vector)
        assert np.array_equal(obs["A"]["action_mask"].astype(bool), oa.mask)
        assert np.array_equal(obs["B"]["action_mask"].astype(bool), ob.mask)
        a = int(rng.choice(oa.legal))
        b = int(rng.choice(ob.legal))
        r = native.step(a, b)
        obs, rew, term, trunc, infos = adapter.step({"A": a, "B": b})
        assert rew["A"] == r.rewards[0] and rew["B"] == r.rewards[1]
        assert term["A"] == r.terminated and trunc["A"] == r.truncated
        i, j = infos["A"], r.info
        assert i["state_after"] == j["state_after"] and i["champ_after"] == j["champ_after"]
        assert i["reward_units"] == j["reward_units"] and i["chance"] == j["chance"] and i["events"] == j["events"]
        assert r.done == (t == T - 1)
        oa, ob = r.obs
    assert adapter.agents == [] and native.done


def test_play_episode_is_reproduced_through_the_adapter():
    suite = natural_suite()
    x, y = suite["slow_pusher"], suite["trading_pusher"]
    seed = 99
    log = play_episode(LaneEnv(LASTHIT, max_turns=200), x, y, seed)
    _, env_ss, rng_a, rng_b = episode_streams(seed)
    adapter = LaneParallelEnv(LASTHIT, max_turns=200)
    adapter.reset(seed=env_ss)
    rewards, actions = [], []
    native = adapter.env
    while adapter.agents:
        oa, ob = native.observe()
        a, b = x.act(oa, rng_a), y.act(ob, rng_b)
        _, rew, _, _, _ = adapter.step({"A": a, "B": b})
        rewards.append(rew["A"])
        actions.append((a, b))
    assert rewards == log.rewards and actions == log.actions and len(rewards) == 200


def test_custom_start_states_m4state_startstate_and_list_forms():
    env = LaneEnv(LASTHIT, max_turns=60)
    rng = np.random.default_rng(5)
    u = UniformAgent()
    obs = env.reset(seed=1)
    for _ in range(40):
        obs = env.step(u.act(obs[0], rng), u.act(obs[1], rng)).obs
    lane, champ, phase = env.state, env.champ, env.k % LASTHIT.phase_period
    assert isinstance(champ, M4State) and champ != env.champ.__class__(champ.b, champ.a)
    st = StartState(lane, champ, phase, label="mid-game")
    ref = LaneEnv(LASTHIT, max_turns=10)
    ra, rb = ref.reset(seed=3, root=lane, k0=phase, champ=champ)
    for options in ({"start": st}, {"root": lane, "k0": phase, "champ": champ},
                    {"root": lane.to_list(), "k0": phase, "champ": champ.to_list()}):
        ad = LaneParallelEnv(LASTHIT, max_turns=10)
        obs, _ = ad.reset(seed=3, options=options)
        assert np.array_equal(obs["A"]["observation"], ra.vector) and np.array_equal(obs["B"]["observation"], rb.vector)
        assert ad.env.state == lane and ad.env.champ == champ and ad.env.k == phase
        assert isinstance(ad.env.champ, M4State)
        # the same turn from the same start gives the same result as the native environment
        a, b = ra.legal[0], rb.legal[-1]
        r = LaneEnv(LASTHIT, max_turns=10)
        r.reset(seed=3, root=lane, k0=phase, champ=champ)
        want = r.step(a, b)
        _, rew, _, _, infos = ad.step({"A": a, "B": b})
        assert rew["A"] == want.rewards[0] and infos["A"]["champ_after"] == want.info["champ_after"]
    # a legacy champion-mode list form is still read as a ChampState
    from lanegame.champions import ChampState, initial_champions
    legacy = GameConfig.champions()
    ad = LaneParallelEnv(legacy)
    ad.reset(seed=0, options={"champ": initial_champions(legacy).to_list()})
    assert isinstance(ad.env.champ, ChampState)


def test_global_state_is_lossless_under_ruleset_m4():
    env = LaneParallelEnv(GameConfig.m4_lasthit(skill_a=0.6, skill_b=0.3), max_turns=80)
    rng = np.random.default_rng(8)
    obs, _ = env.reset(seed=4)
    for _ in range(80):
        phase, lane, champ, skills = decode_m4(env.state(), env.cfg)
        assert (phase, lane, champ) == (env.env.k % env.cfg.phase_period, env.env.state._replace(ap=0, bp=0), env.env.champ)
        assert skills == (np.float32(0.6), np.float32(0.3))
        obs, *_ = env.step(uniform_actions(obs, rng))


def test_max_turns_is_required_for_episodes_under_continuing_time():
    env = LaneParallelEnv(LASTHIT)                          # no time limit: the game never ends by itself
    assert env.max_turns is None
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    for _ in range(30):
        obs, _, term, trunc, _ = env.step(uniform_actions(obs, rng))
        assert not term["A"] and not trunc["A"] and env.agents == ["A", "B"]
    env.max_turns = 32
    obs, _ = env.reset(seed=0)
    for t in range(32):
        obs, _, _, trunc, _ = env.step(uniform_actions(obs, rng))
    assert trunc["A"] and env.agents == []
