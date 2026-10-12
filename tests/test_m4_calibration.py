"""M4.2b calibration: targeted checks of the candidate configurations, the Shove -> Recall cycle agent and the
experiment-only keep-reach farm-spot prototype (docs/m4_calibration_report.md). The M4 defaults are not changed."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import N_CHANCE, mirror_chance, swap_seats
from lanegame.champions_m4 import resolve_turn_m4
from lanegame.config import B_RECALL, SH, champ_behavior, champ_wave
from lanegame.env import LaneEnv, UniformAgent, play_episode
from lanegame.env.agents_m4 import ShoveRecallAgent, m4_suite
from lanegame.env.discounted import per_turn_bound, tail_constant
from lanegame.experiments.m4.calibration import KeepReachEnv, cfg_of, keep_reach_hold_run
from lanegame.experiments.m4.soak import soak
from lanegame.lane_enum import recorded_bounds

CANDIDATES = ["g0.40_i1.35", "g0.30_i1.50", "g0.40_i1.20"]


def test_defaults_unchanged():
    m4 = GameConfig.m4()
    assert (m4.gank_damage_position, m4.item_mult) == (0.6, 1.2)
    assert cfg_of("g0.60_i1.20") == m4


@pytest.mark.parametrize("tag", CANDIDATES)
def test_candidate_validates_and_bounds_still_apply(tag):
    cfg = cfg_of(tag)
    cfg.validate()
    # the tail constant depends on farm, bounty, plates and the lane's A_max only; the lane signature is unchanged
    assert recorded_bounds(cfg) == recorded_bounds(GameConfig.m4())
    assert tail_constant(cfg) == pytest.approx(tail_constant(GameConfig.m4()))
    assert per_turn_bound(cfg) >= per_turn_bound(GameConfig.m4()) - 1e-12


@pytest.mark.parametrize("tag", ["g0.40_i1.35", "g0.30_i1.50"])
def test_candidate_soak_invariants_and_lossless_observation(tag):
    """design §14 invariants, zero-sum, exact reward decomposition, the per-turn bound and exact decoding of every
    observation, on random-legal and on cycle-agent play."""
    cfg = cfg_of(tag)
    s1 = soak(cfg, 1500, 11)
    s2 = soak(cfg, 1000, 12, agents=(ShoveRecallAgent(), m4_suite()["presser"]))
    assert s1["turns"] == 1500 and s2["turns"] == 1000


@pytest.mark.parametrize("tag", ["g0.40_i1.35", "g0.30_i1.50"])
def test_candidate_resolver_mirror_equivariance(tag):
    cfg = cfg_of(tag).with_(skill_a=0.8, skill_b=0.3)
    env = LaneEnv(cfg, max_turns=600)
    rng, nrng = np.random.default_rng(0), np.random.default_rng(9)
    ag = UniformAgent()
    obs = env.reset(seed=3)
    for _ in range(600):
        lane, m4, k = env.state, env.champ, env.k
        a, b = ag.act(obs[0], rng), ag.act(obs[1], rng)
        u = tuple(float(x) for x in nrng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, u=u)
        m = resolve_turn_m4(lane.mirror(), m4.mirror(), k, b, a, swap_seats(cfg), u=mirror_chance(u))
        assert m.champ == t.champ.mirror() and m.lane.state == t.lane.state.mirror()
        assert m.reward_units == -t.reward_units
        r = env.step(a, b)
        obs = r.obs
        if r.done:
            break


def test_candidate_replay_is_deterministic():
    cfg = cfg_of("g0.40_i1.35")
    env = LaneEnv(cfg, max_turns=300)
    l1 = play_episode(env, ShoveRecallAgent(), m4_suite()["aggressive_trader"], 5)
    l2 = play_episode(LaneEnv(cfg, max_turns=300, cache_size=0), ShoveRecallAgent(),
                      m4_suite()["aggressive_trader"], 5)
    assert l1.rewards == l2.rewards and l1.actions == l2.actions


def test_shove_recall_agent_is_legal_stateless_and_cycles():
    cfg = GameConfig.m4()
    env = LaneEnv(cfg, max_turns=400)
    sr = ShoveRecallAgent()
    log = play_episode(env, sr, m4_suite()["hold_farmer"], 3, record=True)
    rules = set()
    obs = env.reset(seed=4)
    for _ in range(400):
        a, rule = sr.decide(obs[0])
        assert a in obs[0].legal
        assert sr.decide(obs[0]) == (a, rule)                  # stateless: same observation, same decision
        rules.add(rule)
        r = env.step(a, m4_suite()["hold_farmer"].act(obs[1]))
        if r.done:
            break
        obs = r.obs
    assert {"farm", "absent"} <= rules
    turns = log.turns
    assert any(t["purchases"][0] for t in turns)
    assert any(t["recalled"][0] for t in turns)
    assert any(champ_wave(t["actions"][0]) == SH for t in turns)
    # recalls that are not low-HP recalls happen (the cash-out rule) and the agent does not shove permanently
    assert sum(champ_wave(t["actions"][0]) == SH for t in turns) < 0.5 * len(turns)
    assert any(champ_behavior(t["actions"][0]) == B_RECALL for t in turns)


def test_keep_reach_prototype_reproduces_the_design_parked_table_and_is_experiment_only():
    """The probe-crash farm spot explains finding 1 of the M4.2 report: with the pre-turn front as the farm spot of a
    champion whose front crashes in the probe, the Shove rows equal the design's lane-only windows. The engine and
    LaneEnv are unchanged (the prototype is a subclass in the experiment module)."""
    from lanegame.config import F, SP
    from lanegame.scenarios_m4 import DESIGN_HOLD_TABLE, hold_run
    cfg = GameConfig.m4()
    for e in range(2, 10):
        for bw in (SP, SH):
            design = DESIGN_HOLD_TABLE[(e, bw)][0]
            assert [keep_reach_hold_run(e, bw, ph, cfg) for ph in range(4)] == design, (e, bw)
    assert [hold_run(4, SH, ph, F, cfg)["first_crash"] for ph in range(4)] == [9, 8, 9, 8]   # current rule
    assert KeepReachEnv.__mro__[1] is LaneEnv and "KeepReach" not in LaneEnv.__name__
