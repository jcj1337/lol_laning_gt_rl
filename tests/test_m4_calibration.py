"""M4.2b calibration: targeted checks of the candidate configurations, the Shove -> Recall cycle agent and the
experiment-only keep-reach farm-spot prototype (docs/archive/m4/m4_2/m4_calibration_report.md). The M4 defaults are not changed."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.engine.champions import N_CHANCE, mirror_chance, swap_seats
from lanegame.engine.champions_m4 import resolve_turn_m4
from lanegame.engine.config import B_RECALL, SH, champ_behavior, champ_wave
from lanegame.env import LaneEnv, UniformAgent, play_episode
from lanegame.env.agents_m4 import ShoveRecallAgent, m4_suite
from lanegame.env.discounted import per_turn_bound, tail_constant
from lanegame.engine.lane_enum import recorded_bounds

CANDIDATES = ["g0.40_i1.35", "g0.30_i1.50", "g0.40_i1.20"]


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

