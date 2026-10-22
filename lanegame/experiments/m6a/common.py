"""Shared definitions of M6a, the demo's technical readiness (docs/m6/m6_plan.md §6 M6a; the session core is
lanegame/demo).

Seeds: the block 110,000-110,999, disjoint from every earlier study (which all use roots below 100,000, M5 70,000-70,999
and the reference generator 76,000-77,999).

    SIM_ROOT       new bot-versus-bot simulations (chance seed SIM_ROOT + j, policy seed SIM_ROOT + 500 + j, both
                   recorded; M6a's prototypes used them, and `measure` still does)
    MEASURE_ROOT   the cost measurements (sessions with scripted human choices)

Results go to results/m6/ (measurements, verification).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict

from ..m45b.common import _default, git_commit, git_dirty

OUT = os.path.join("results", "m6")
TRACES = os.path.join("results", "m5", "traces")
SEED_BLOCK = (110_000, 110_999)
SIM_ROOT = 110_000
MEASURE_ROOT = 110_800
RESEARCH_GAMES = ("h2h_om_s0_vs_sp_s0", "om_s0_vs_g_shove", "sp_s0_vs_g_shove")


def meta(**kw) -> Dict[str, object]:
    return {"study": "M6a", "preset": "m45_calibrated", "commit": git_commit(), "working_tree_dirty": git_dirty(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}


def save_json(name: str, data: dict, out: str = OUT, indent=1) -> str:
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name + ".json")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, default=_default, indent=indent))
    print(f"wrote {p}", flush=True)
    return p


def registry():
    from ...demo.registry import Registry
    r = Registry()
    if not r.available():
        raise SystemExit("the demo's checkpoints are not on disk (results/rl/checkpoints, git-ignored): stage skipped")
    return r.load()
