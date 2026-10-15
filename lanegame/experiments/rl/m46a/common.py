"""Shared definitions of M4.6a (docs/m4/m4_6/m4_6a_preregistration.md §1).

Environment: `GameConfig.m45_calibrated()` through `m45c.common.env_cfg()` (frozen; nothing here changes it).
Policies: the six 16.384M finals of M4.5c (`final:ext:<seed>`); controls: the M4.5c s0 attackers.
Seeds: the range 96,000-96,999 (pre-registration §1). Raw per-job results go to the git-ignored checkpoint tree
results/rl/checkpoints/m46a/; summaries and reports to results/m46a/.

Workers. Every worker installs a bounded exact lane-step memo (`m45b.engine.LaneCache`, CACHE entries) before its first
rollout: the memo is exact, so results do not depend on its size, only memory does (about 0.3 GB per worker).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..m45c.common import CKPT as M45C_CKPT
from ..m45c.common import env_cfg, git_commit, git_dirty, run_jobs  # noqa: F401  (re-exported)

NAME = "m46a"
OUT = os.path.join("results", NAME)
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
PREREG_DOC = os.path.join("docs", "m4", "m4_6", "m4_6a_preregistration.md")
FINALS = (0, 1, 2, 3, 4, 5)
E4_TARGETS = (0, 2, 4)
CACHE = 3000

# seeds (pre-registration §1)
E3_ROOT = 96_000
E2_STATE_ROOT = 96_100
ROLL_ROOT = 96_200
E4_TRAIN_ROOT = 96_300
E4_EVAL_ROOT = 96_400
E5_STATE_ROOT = 96_500
VAL_ROOT = 96_700

GAMMA = 0.99


def spec(seed: int) -> str:
    return f"final:ext:{seed}"


def s0_attacker_path(seed: int) -> str:
    return os.path.join(M45C_CKPT, "attack", f"ext_s{seed}", "final.pt")


def setup_worker(cache: int = CACHE):
    """One torch thread and a bounded exact lane memo for the M4.5 configuration (idempotent)."""
    import torch
    torch.set_num_threads(1)
    from ...m45b import engine as E
    cfg = env_cfg()
    c = E._CACHES.get(cfg)
    if c is None or c.size != cache:
        E._CACHES[cfg] = E.LaneCache(cfg, cache)
    return cfg


_NETS: Dict[str, object] = {}


def net_of(spec_or_path: str):
    """The PolicyValueNet of an M4.5c agent specification or checkpoint path (cached per process)."""
    if spec_or_path not in _NETS:
        from ..m45c.common import agent_from_spec
        _NETS[spec_or_path] = agent_from_spec(spec_or_path, env_cfg()).net
    return _NETS[spec_or_path]


# ---------------------------------------------------------------------------------------------- pre-registration
def prereg_record() -> dict:
    with open(PREREG_DOC, "rb") as f:
        b = f.read()
    return {"document": PREREG_DOC, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def write_prereg() -> dict:
    """Stage `prereg`: freeze the pre-registration (its SHA-256 and the time) before any experiment stage."""
    path = os.path.join(OUT, "preregistration.json")
    if os.path.exists(path):
        old = load_json("preregistration")
        rec = prereg_record()
        if old and old["record"]["sha256"] != rec["sha256"]:
            raise SystemExit("the pre-registration document changed after it was frozen; record an amendment instead")
        return old
    data = {"meta": meta(stage="prereg"), "record": prereg_record(),
            "note": "written before any M4.6a experiment stage ran; smoke tests check pipelines only"}
    save_json("preregistration", data)
    return data


def require_prereg() -> None:
    if not os.path.exists(os.path.join(OUT, "preregistration.json")):
        raise SystemExit("run the `prereg` stage first (docs/m4/m4_6/m4_6a_preregistration.md)")


def amend(note: str) -> dict:
    """Record a dated amendment of the protocol (made before the affected stage ran)."""
    d = load_json("preregistration") or {}
    d.setdefault("amendments", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note})
    save_json("preregistration", d)
    return d


# ---------------------------------------------------------------------------------------------- statistics
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 15: 2.131, 20: 2.086, 23: 2.069, 30: 2.042, 40: 2.021, 47: 2.012, 60: 2.000}


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T95:
        return T95[df]
    keys = sorted(T95)
    if df > keys[-1]:
        return 1.96 + (2.000 - 1.96) * 60 / df
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    return T95[lo] + (T95[hi] - T95[lo]) * (df - lo) / (hi - lo)


def mean_ci(values: Iterable[float]) -> Tuple[float, float]:
    """Mean and the half-width of the 95% t interval (nan-safe; nan half-width below 2 values)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float("nan")
    return float(v.mean()), float(t95(len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))


def significant(m: float, h: float) -> int:
    """+1 / -1 if the interval excludes 0 in that direction, else 0."""
    if not (np.isfinite(m) and np.isfinite(h)):
        return 0
    return 1 if m - h > 0 else (-1 if m + h < 0 else 0)


def holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """Holm-Bonferroni: which hypotheses are rejected at family-wise level alpha."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, m = {}, len(items)
    stop = False
    for i, (k, p) in enumerate(items):
        if stop or not np.isfinite(p) or p > alpha / (m - i):
            stop = True
            out[k] = False
        else:
            out[k] = True
    return out


def t_pvalue(values: Sequence[float]) -> float:
    """Two-sided p-value of a one-sample t test of mean 0."""
    from math import sqrt
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = len(v)
    if n < 2 or v.std(ddof=1) == 0:
        return float("nan")
    t = abs(v.mean()) / (v.std(ddof=1) / sqrt(n))
    try:
        from scipy import stats
        return float(2 * stats.t.sf(t, n - 1))
    except Exception:                                   # pragma: no cover - scipy missing
        return float(2 * (1 - 0.5 * (1 + math.erf(t / sqrt(2)))))


def fmt(m: float, h: float = float("nan"), d: int = 2) -> str:
    if not np.isfinite(m):
        return "n/a"
    if not np.isfinite(h):
        return f"{m:+.{d}f}"
    return f"{m:+.{d}f} ± {h:.{d}f}"


# ---------------------------------------------------------------------------------------------- I/O
def meta(**kw) -> dict:
    import sys
    try:
        import torch
        tv = torch.__version__
    except Exception:                                   # pragma: no cover
        tv = None
    return {"study": "M4.6a", "commit": git_commit(), "working_tree_dirty": git_dirty(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0], "torch": tv,
            "numpy": np.__version__, "environment": "GameConfig.m45_calibrated()", **kw}


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_json(name: str, data: dict, out: str = OUT) -> str:
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, default=_default)
    os.replace(tmp, path)
    return path


def load_json(name: str, out: str = OUT) -> Optional[dict]:
    path = os.path.join(out, name + ".json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_gz(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f, default=_default)
    os.replace(tmp, path)


def load_gz(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_md(name: str, text: str, out: str = OUT) -> str:
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name + ".md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def state_to_list(start) -> list:
    lane, m4, k = start
    return [lane.to_list(), m4.to_list(), int(k)]


def state_from_list(v):
    from ....champions_m4 import M4State
    from ....state import State
    return State.from_list(v[0]), M4State.from_list(v[1]), int(v[2])


__all__ = ["NAME", "OUT", "CKPT", "FINALS", "E4_TARGETS", "spec", "s0_attacker_path", "setup_worker", "net_of",
           "env_cfg", "run_jobs", "write_prereg", "require_prereg", "amend", "mean_ci", "significant", "holm",
           "t_pvalue", "fmt", "meta", "save_json", "load_json", "save_gz", "load_gz", "write_md", "state_to_list",
           "state_from_list", "GAMMA"]
