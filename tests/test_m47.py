"""M4.7: the horizon study's objective override (configuration, trainer, GAE), the reward-stream engine and its
scoring arithmetic. Checkpoint- and record-dependent tests skip when the git-ignored files are absent."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lanegame import GameConfig                                                       # noqa: E402
from lanegame.rl import PolicyValueNet, SharedConfig, train_shared                    # noqa: E402
from lanegame.rl.ppo import PPOConfig, check_time_semantics, compute_gae              # noqa: E402

RECORDS = os.path.join("runs", "checkpoints", "m46d", "records", "baseline", "ext_baseline_0.npz")


def _net(seed=0):
    torch.set_num_threads(1)              # orthogonal initialization depends on the thread count
    torch.manual_seed(seed)
    return PolicyValueNet(90, 61)


# ---------------------------------------------------------------------------------------------- the objective
def test_gae_truncation_uses_the_next_value_and_termination_would_not():
    r = np.array([[1.0], [0.0]])
    v = np.array([[0.0], [0.0]])
    nv = np.array([[0.0], [10.0]])
    adv, _ = compute_gae(r, v, nv, np.zeros_like(r, bool), np.array([[False], [True]]), 0.9, 0.95)
    assert adv[1, 0] == pytest.approx(9.0)                      # bootstrapped: gamma * V(next)
    adv_t, _ = compute_gae(r, v, nv, np.array([[False], [True]]), np.array([[False], [True]]), 0.9, 0.95)
    assert adv_t[1, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------------- discount arithmetic
# ---------------------------------------------------------------------------------------------- the stream engine
# ---------------------------------------------------------------------------------------------- record-based readouts