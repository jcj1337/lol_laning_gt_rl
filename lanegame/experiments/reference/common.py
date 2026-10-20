"""Shared definitions of the reference generator: the preset, the learned policy, seeds and I/O.

Preset: `GameConfig.m45_calibrated()`, the research preset (M4.5c to M5). Learned play is the M5 pure self-play final
of seed 0 (`sp:0`, `experiments.rl.m5.players`), a stateless 90-input policy; the checkpoints are git-ignored, so the
stages that need it stop with a message when it is missing. Each seat samples with the uniforms of
default_rng([seed, 7, seat]) (`m45c.horizon.CRNNet`), so two games on the same seed share the chance stream and both
seats' sampling streams (common random numbers).

Seeds: PLAY 76,000-76,999 (self-play games and the states sampled from them), ROLL 77,000-77,999 (rollouts from those
states). Both ranges are disjoint from every earlier study (M5 used 70,000-70,999).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict

from ...config import GameConfig
from ..m45b.common import _default, git_commit, git_dirty

OUT = os.path.join("results", "reference")
POLICY = "sp:0"
PLAY = tuple(range(76_000, 77_000))
ROLL = tuple(range(77_000, 78_000))
GAMMA = 0.99


def cfg() -> GameConfig:
    c = GameConfig.m45_calibrated()
    c.validate()
    return c


def policy_available() -> bool:
    from ..rl.m5 import common as C5
    return os.path.exists(C5.policy_path(POLICY))


def meta(**kw) -> Dict[str, object]:
    return {"study": "reference", "preset": "m45_calibrated", "commit": git_commit(),
            "working_tree_dirty": git_dirty(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}


def save_json(name: str, data: dict) -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".json")
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, default=_default, indent=1))
    print(f"wrote {p}", flush=True)
    return p


def load_json(name: str):
    p = os.path.join(OUT, name + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_md(name: str, text: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    print(f"wrote {p}", flush=True)
    return p
