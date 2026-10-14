"""Checkpoint -> viewer recording: games of trained policies exported in the schema the Streamlit debugger replays
("Recorded trajectory" mode; `streamlit run lanegame/viewer/app.py`, folder results/m44a/trajectories/).

A recording is a pure function of (Setup, actions): the viewer resets the environment at the setup's start with the
setup's INTEGER chance seed and replays the actions. `experiments.rl.m43b.trajectories.record_game` plays a game
exactly that way (env.reset(seed=int); the agents sample from their own streams); this module writes the four-action
preset's name as the recording's "config" (a key of `lanegame.viewer.session.CONFIGS`) and verifies the round trip:
`verify` replays the file through `viewer.session.play` and checks that every turn's actions and rewards are equal.
"""

from __future__ import annotations

import os
from typing import List, Optional

from .common import ENVS, OUT_A, agent_from_spec, env_cfg

SCHEMA = "lanegame.viewer_recording"
TRAJ_DIR = os.path.join(OUT_A, "trajectories")


def export(name: str, env: str, x_spec: str, y_spec: str, seed: int, turns: int, label: str,
           out_dir: str = TRAJ_DIR) -> dict:
    from ....io import dump_json
    from ....viewer.session import recording_stamp
    from ..m43b.trajectories import record_game
    cfg = env_cfg(env)
    x, y = agent_from_spec(x_spec, cfg), agent_from_spec(y_spec, cfg)
    g = record_game(cfg, x, y, seed, turns)
    rec = {"schema": SCHEMA, "version": 1, **recording_stamp(ENVS[env].preset), "scenario": "s0", "gap": "equal",
           "seed": seed, "names": [x_spec, y_spec], "label": label, **g}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    dump_json(rec, path)
    rec["path"] = path
    return rec


def verify(path: str) -> dict:
    """Replay a recording through the viewer's own session code; equal actions and payoff to A on every turn."""
    from ....viewer import session as ss
    setup, actions, meta = ss.load_recording(path)
    rec = ss.play(setup, prefix=actions, prefix_note="recorded")
    replay_actions = [tuple(int(v) for v in a) for a in rec.actions]
    payoff = sum(t.info["reward"] for t in rec.turns)
    return {"path": path, "config": setup.config, "turns": len(replay_actions),
            "actions_equal": replay_actions == [tuple(a) for a in actions],
            "payoff_equal": abs(payoff - meta["payoff_A"]) < 1e-9, "payoff_A": payoff}


__all__ = ["export", "verify", "SCHEMA", "TRAJ_DIR"]
