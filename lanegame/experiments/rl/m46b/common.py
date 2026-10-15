"""Shared definitions of M4.6b. Seeds: the new range 97,000-97,999 (disjoint from every earlier study, M4.6a's
96,000-96,999 included). Raw per-job results under results/rl/checkpoints/m46b/ (git-ignored), summaries and reports
under results/m46b/.

Arms. "baseline" is `GameConfig.m45_calibrated()` itself. Every candidate is `m45_calibrated(**overrides)` with
existing investment fields only (curves.ARMS); `arm_cfg(name)` validates it. Workers install a bounded exact lane-step
memo per configuration (results do not depend on its size).
"""

from __future__ import annotations

import os
from typing import Dict

from ..m46a.common import (FINALS, GAMMA, fmt, holm, load_gz, load_json, mean_ci, meta as _meta, net_of, run_jobs,
                           save_gz, save_json as _save_json, significant, spec, state_from_list, state_to_list,
                           t_pvalue, write_md as _write_md)

NAME = "m46b"
OUT = os.path.join("results", NAME)
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
PREREG_DOC = os.path.join("docs", "m4", "m4_6", "m4_6b_preregistration.md")
CACHE = 3000

AUDIT_ROOT = 97_000
VERIFY_ROOT = 97_100
SCREEN_STATE_ROOT = 97_200
ROLL_ROOT = 97_300
NATURAL_ROOT = 97_400


def meta(**kw) -> dict:
    d = _meta(**kw)
    d["study"] = "M4.6b"
    return d


def save_json(name: str, data: dict, out: str = OUT) -> str:
    return _save_json(name, data, out)


def load(name: str, out: str = OUT):
    return load_json(name, out)


def write_md(name: str, text: str, out: str = OUT) -> str:
    return _write_md(name, text, out)


_CFGS: Dict[str, object] = {}


def arm_cfg(arm: str = "baseline"):
    """The GameConfig of an arm (validated, cached per process)."""
    if arm not in _CFGS:
        from ....config import GameConfig
        from .curves import ARMS
        cfg = GameConfig.m45_calibrated(**ARMS[arm]["overrides"])
        cfg.validate()
        _CFGS[arm] = cfg
    return _CFGS[arm]


def setup(arm: str = "baseline", cache: int = CACHE):
    """One torch thread and a bounded exact lane memo for the arm's configuration; returns the configuration."""
    import torch
    torch.set_num_threads(1)
    from ...m45b import engine as E
    cfg = arm_cfg(arm)
    c = E._CACHES.get(cfg)
    if c is None or c.size != cache:
        if len(E._CACHES) > 6:
            E._CACHES.clear()
        E._CACHES[cfg] = E.LaneCache(cfg, cache)
    return cfg


def prereg_record() -> dict:
    import hashlib
    with open(PREREG_DOC, "rb") as f:
        b = f.read()
    return {"document": PREREG_DOC, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def write_prereg() -> dict:
    path = os.path.join(OUT, "preregistration.json")
    if os.path.exists(path):
        old = load("preregistration")
        if old["record"]["sha256"] != prereg_record()["sha256"]:
            raise SystemExit("the M4.6b pre-registration changed after it was frozen; record an amendment instead")
        return old
    d = {"meta": meta(stage="prereg"), "record": prereg_record(),
         "note": "frozen after the audit of the CURRENT curve and before any candidate gameplay outcome"}
    save_json("preregistration", d)
    return d


def require_prereg() -> None:
    if not os.path.exists(os.path.join(OUT, "preregistration.json")):
        raise SystemExit("freeze the M4.6b pre-registration first (stage prereg)")


def amend(note: str) -> dict:
    import time
    d = load("preregistration") or {}
    d.setdefault("amendments", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note})
    save_json("preregistration", d)
    return d


__all__ = ["NAME", "OUT", "CKPT", "FINALS", "GAMMA", "arm_cfg", "setup", "meta", "save_json", "load", "write_md",
           "run_jobs", "mean_ci", "significant", "holm", "t_pvalue", "fmt", "spec", "net_of", "save_gz", "load_gz",
           "state_to_list", "state_from_list", "write_prereg", "require_prereg", "amend"]
