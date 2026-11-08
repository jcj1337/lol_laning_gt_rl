"""M4.4c experimental infrastructure (docs/archive/m4/m4_4/m4_4c_strategic_investigation.md): multi-turn strategic plans, paired
multi-turn counterfactuals, complete-action Recall comparisons, opponent protocols and styles, episode ledgers, wave
budget identities, seed roots and viewer traces. No checkpoint files: networks are the untrained initial networks of a
training seed (bit-identical to the start of the M4.4b runs)."""

import numpy as np
import pytest

from lanegame.engine.champions_m4 import resolve_turn_m4
from lanegame.engine.config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, W_HOLD, W_SHOVE, W_THIN, champ_act,
                             champ_behavior, champ_wave)
from lanegame.diagnostics.branch import FollowPolicy, ProbCache, observation, rollout, rollout_seed
from lanegame.env import LaneEnv
from experiments.m44 import common as C
from experiments.m44.actions import RECALL
from experiments.m44.records import holds_freeze, play_record
from experiments.m44c import opponents as O
from experiments.m44c import plans as P


@pytest.fixture(scope="module")
def setup():
    cfg = C.env_cfg("aggro")
    x = C.agent_from_spec("initial:aggro:1", cfg)
    rec = play_record(cfg, (x, x), np.random.SeedSequence(81_990), 80, keep_snapshots=True)
    return cfg, x, rec


# ---------------------------------------------------------------------------------------------- plans
def test_forcing_several_components_keeps_the_policys_conditional_choice_of_the_rest(setup):
    cfg, x, rec = setup
    ob = observation(cfg, rec.snapshots[30], 1)
    p = np.asarray(x.distribution(ob), dtype=float)
    q = P.reshape(p, ob.legal, {"wave": W_THIN, "behavior": B_HOLD})
    legal = list(ob.legal)
    sl = [i for i, a in enumerate(legal) if a not in (RECALL, CHAMP_ABSENT) and champ_wave(a) == W_THIN
          and champ_behavior(a) == B_HOLD]
    assert np.isclose(q.sum(), 1.0) and set(np.flatnonzero(q > 0)) <= set(sl)
    assert np.allclose(q[sl], p[sl] / p[sl].sum())
    one = P.reshape(p, ob.legal, {"behavior": B_SAFE})
    from experiments.m44.interventions import intervene
    assert np.allclose(one, intervene(p, ob.legal, "behavior", B_SAFE))


# ---------------------------------------------------------------------------------------------- Recall
# ---------------------------------------------------------------------------------------------- engine
# ---------------------------------------------------------------------------------------------- waves
# ---------------------------------------------------------------------------------------------- seeds, traces, CLI