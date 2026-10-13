"""Shared definitions of the M4.3d study (docs/m4/m4_3d_tower_pressure_calibration.md).

Nothing here changes a preset. Every configuration of the study is `GameConfig.m4_four_actions_econ()` (the M4.3c
candidate, "econ") or `m4_four_actions()` (the original four-action economy, "base4") with explicit overrides
(VARIANTS). Results go to results/m43d/.

Seeds: SeedSequence roots 79,000-79,999, disjoint from every earlier study (10,000-69,999, 71,000-78,001, 77,777).

Chance masks (`MaskedChance`). The environment draws ten uniforms per turn whatever happens (champions.U_*). A mask
replaces chosen entries of every draw by 1.0 (the "no event" value) AFTER they are drawn, so the stream stays aligned
turn by turn with the unmasked game: "A immune to ganks" or "A immune to tower shots" is a paired counterfactual of
the same game, and the difference of the two values is the economic cost of that hazard to A under the strategies
played (they react to its consequences, e.g. by recalling less).
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ...champions import U_GANK, U_GANK_DAMAGE, U_TOWER, U_TRADE_DOUBLE, U_TRADE_HIT
from ...config import GameConfig
from ...io import dump_json
from ..m43c.common import git_commit, mean_ci

NAME = "m43d"
OUT = os.path.join("results", NAME)
GZIP_BYTES = 1_000_000

TOWER_FARM_SEED = 79_000        # Part 2 crash grid
VALIDATION_SEED = 79_500        # independent validation seed sets
STRATEGIC_SEED = 79_200         # Part 5 restricted games
STRATEGIC_VALIDATION_SEED = 79_700
TRACE_SEED = 79_900

A, B = 0, 1


# ---------------------------------------------------------------------------------------------- configurations
# name -> (description, preset, overrides). "econ" is the M4.3c candidate; the ot* variants change only the own-tower
# collection rate; the risk variants (Part 4) are added below after the diagnosis.
VARIANTS: Dict[str, Tuple[str, str, dict]] = {
    "base4": ("the original four-action economy m4_four_actions() (own tower 0.6, trade factor 0.5, bounty 0.5, "
              "death 3 turns)", "m4_four_actions", {}),
    "econ": ("the M4.3c candidate m4_four_actions_econ() (own tower 0.8)", "m4_four_actions_econ", {}),
    "ot60": ("econ with the historical own-tower rate 0.6", "m4_four_actions_econ",
             {"collect_contest_own_tower": 0.6}),
    "ot85": ("econ with own-tower collection 0.85", "m4_four_actions_econ", {"collect_contest_own_tower": 0.85}),
    "ot90": ("econ with own-tower collection 0.9 (the lane rate: no unconditional tower cap)", "m4_four_actions_econ",
             {"collect_contest_own_tower": 0.9}),
    "econ_aggro": ("the M4.3d candidate m4_four_actions_econ_aggro() (econ + no tower 'busy' discount for divers; "
                   "= tw_aggro)", "m4_four_actions_econ_aggro", {}),
    # ---- DIAGNOSTICS of Part 5: no plate gold (the game after the plates fall), at 0.8 and 0.9
    "econ_pg0": ("DIAGNOSTIC: econ without plate gold", "m4_four_actions_econ", {"plate_gold": 0.0}),
    "ot90_pg0": ("DIAGNOSTIC: econ with own-tower 0.9 and without plate gold", "m4_four_actions_econ",
                 {"collect_contest_own_tower": 0.9, "plate_gold": 0.0}),
    # ---- Part 4 risk-sensitivity candidates (each one existing parameter; run only where the diagnosis points)
    "tw_aggro": ("econ; the tower is never 'busy' for a connected trader: a dive is shot with p 1.0 and the tower-"
                 "adjacent shot with 0.5 even while the diver's minions are under the tower (League's tower aggro on a "
                 "champion that attacks a champion)", "m4_four_actions_econ",
                 {"tower_shot_p_busy": 1.0, "tower_adjacent_p_busy": 0.5}),
    "tw2": ("econ; a tower shot deals 2 HP units (1/3 bar) instead of 1", "m4_four_actions_econ",
            {"tower_shot_damage": 2}),
    "tw2_aggro": ("econ; tw2 + tw_aggro", "m4_four_actions_econ",
                  {"tower_shot_damage": 2, "tower_shot_p_busy": 1.0, "tower_adjacent_p_busy": 0.5}),
    "gk_pos16": ("econ; gank occurrence slope 0.16 instead of 0.12 (exposure +2: 0.17 / 0.19 with Shove)",
                 "m4_four_actions_econ", {"champ_gank_position": 0.16}),
    "gk_dmg08": ("econ; gank damage slope 0.8 instead of 0.6 (p_e 0.2 / 0.4 / 0.6 / 0.8 / 1.0 at exposure -2..+2)",
                 "m4_four_actions_econ", {"gank_damage_position": 0.8}),
    "gk_trials5": ("econ; a gank rolls 5 damage trials instead of 4 (up to 5/6 of a bar)", "m4_four_actions_econ",
                   {"gank_damage_trials": 5}),
}


def register(name: str, description: str, overrides: dict, preset: str = "m4_four_actions_econ") -> None:
    VARIANTS[name] = (description, preset, dict(overrides))


def variant_cfg(name: str) -> GameConfig:
    desc, preset, kw = VARIANTS[name]
    cfg = getattr(GameConfig, preset)(**kw)
    cfg.validate()
    return cfg


def overrides_json(name: str) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in VARIANTS[name][2].items()}


# ---------------------------------------------------------------------------------------------- chance masks
MASKS = {
    "a_no_gank": tuple(U_GANK[:1]) + tuple(U_GANK_DAMAGE[:1]),
    "b_no_gank": tuple(U_GANK[1:]) + tuple(U_GANK_DAMAGE[1:]),
    "a_no_tower": tuple(U_TOWER[:1]),
    "b_no_tower": tuple(U_TOWER[1:]),
    "no_gank": tuple(U_GANK) + tuple(U_GANK_DAMAGE),
    "a_no_trade_hit": (U_TRADE_HIT[0], U_TRADE_DOUBLE[0]),
}


class MaskedChance:
    """Wraps the environment's generator: every draw of n uniforms is made as usual, then the masked entries are set
    to 1.0 (no event). Single draws pass through unchanged."""

    def __init__(self, rng, indices: Sequence[int]):
        self.rng, self.indices = rng, tuple(indices)

    def random(self, n=None):
        u = self.rng.random(n)
        if n is None:
            return u
        u = np.array(u, dtype=float)
        u[list(self.indices)] = 1.0
        return u


# ---------------------------------------------------------------------------------------------- I/O
def path(name: str, ext: str = "json") -> str:
    return os.path.join(OUT, f"{name}.{ext}")


def save_json(name: str, data: dict) -> str:
    """results/m43d/<name>.json, gzipped (plain file removed) when it is >= 1 MB (the M4.3b artifact rule)."""
    os.makedirs(OUT, exist_ok=True)
    p = path(name)
    dump_json(data, p)
    if os.path.getsize(p) >= GZIP_BYTES:
        with open(p, "rb") as src, gzip.open(p + ".gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(p)
        p += ".gz"
    elif os.path.exists(p + ".gz"):
        os.remove(p + ".gz")
    print(f"wrote {p}", flush=True)
    return p


def load(name: str) -> Optional[dict]:
    p = path(name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(p + ".gz"):
        with gzip.open(p + ".gz", "rt", encoding="utf-8") as f:
            return json.load(f)
    return None


def write_md(name: str, text: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(path(name, "md"), "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    print(f"wrote {path(name, 'md')}", flush=True)


def meta(**kw) -> dict:
    import sys
    return {"commit": git_commit(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
            "numpy": np.__version__, "study": "M4.3d", **kw}


def run_jobs(fn, jobs, workers: int, label: str = ""):
    from ..rl.m43b.common import run_jobs as _rj
    return _rj(fn, jobs, workers, label)


__all__ = ["NAME", "OUT", "VARIANTS", "register", "variant_cfg", "overrides_json", "MASKS", "MaskedChance",
           "save_json", "load", "write_md", "meta", "mean_ci", "run_jobs", "TOWER_FARM_SEED", "VALIDATION_SEED",
           "STRATEGIC_SEED", "STRATEGIC_VALIDATION_SEED", "TRACE_SEED"]
