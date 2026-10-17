"""Shared definitions of OM2, the Bayesian opponent-modelling pilot (design docs/opponent_modelling/om2_design.md;
pre-registration docs/opponent_modelling/om2_preregistration.md).

Environment: `GameConfig.m45_calibrated(obs_last_action=True)`, the adopted M5 baseline with the opt-in observation of
the opponent's previous action (103 entries; the dynamics are the baseline's exactly).

Arms (both continue the adopted baseline policy `ext/baseline:s`, widened to the opponent-modelling input with zero
weights on the new inputs, so both start as exactly that policy):
    control     network input = observation (103) + 13 constant entries (the belief's prior)
    treatment   network input = observation (103) + the belief's posterior over the 11 library types and the two
                catch-alls "mixed" (a different library style each turn) and "unknown" (uniform)
Everything else (PPO settings, start mixture, opponent mix, seeds) is identical.

Opponents:
    BELIEF_TYPES      what the belief models: 6 scripted archetypes, 5 learned policies, the catch-alls "mixed" and
                      "unknown"
    TRAIN_OPPONENTS   the training mix's fixed opponents: the 11 library types, 2 per-turn mixtures, 2 style switchers
    HELDOUT           evaluation only: held-out scripted variants (E), competent learned opponents (R1), mixtures and
                      switchers (R2)
Seeds: the range 61,000-61,999. Raw files under results/rl/checkpoints/om2/ (git-ignored); summaries under
results/om2/.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Dict

import numpy as np

from ..m46a.common import load_gz, save_gz  # noqa: F401
from ..m46a.common import load_json as _load_json
from ..m46a.common import save_json as _save_json
from ..m46a.common import write_md as _write_md
from ..m46c import common as C46
from ..om1 import common as O1

NAME = "om2"
OUT = os.path.join("results", NAME)
CKPT = os.path.join("results", "rl", "checkpoints", NAME)
PREREG_DOC = os.path.join("docs", "opponent_modelling", "om2_preregistration.md")
ARMS = ("control", "treatment")
PILOT_SEEDS = (0, 1, 2)

# ---------------------------------------------------------------------------------------------- the belief
LAMBDA = 0.97                   # forgetting (effective memory about 33 informative turns)
EPS = 0.05                      # scripted types' tremble
ARCHETYPES = ("g_farm", "g_safe", "g_bully", "g_shove", "fz_guard_sh", "g_shove_ward")
LEARNED = {k: O1.LEARNED[k] for k in ("peer", "attacker", "freezer", "gapper", "short")}
BELIEF_TYPES = ARCHETYPES + tuple(LEARNED) + ("mixed", "unknown")
K = len(BELIEF_TYPES)           # 13

# ---------------------------------------------------------------------------------------------- the training mix
P_LIB = 0.5                     # share of training games against a fixed opponent (the rest: current-policy self-play)
MIXTURES = {"mix_bully_safe": ("g_bully", "g_safe", 0.5), "mix_shove_farm": ("g_shove", "g_farm", 0.5)}
SWITCHERS = {"switch_farm_bully_40": ("g_farm", "g_bully", 40), "switch_shove_safe_60": ("g_shove", "g_safe", 60)}
TRAIN_OPPONENTS = ARCHETYPES + tuple(LEARNED) + tuple(MIXTURES) + tuple(SWITCHERS)

# ---------------------------------------------------------------------------------------------- held-out evaluation
_CK = os.path.join("results", "rl", "checkpoints")
HELDOUT_SCRIPTED = ("g_shove_sr", "wk_shove_reset_sh", "g_shove_edge", "g_slow_guard")
HELDOUT_LEARNED = {
    "attacker_b": os.path.join(_CK, "m45c", "attack", "ext_s0", "final.pt"),
    "freezer_b": os.path.join(_CK, "m46a", "e4", "freeze_s4", "final.pt"),
    "gapper_b": os.path.join(_CK, "m46a", "e4", "gap_s4", "final.pt"),
    "short_b": os.path.join(_CK, "m47", "train", "main", "short", "s1", "final.pt"),
    "c1_policy": C46.policy_path("ext/C1:2"),
}
COMPETENT_SEEDS = (3, 4, 5)     # the other seeds' adopted baseline policies ext/baseline:o (never a pilot seed)
HELDOUT_MIXTURES = {"mix_bully_farm": ("g_bully", "g_farm", 0.5), "mix_shove_safe": ("g_shove", "g_safe", 0.5)}
HELDOUT_SWITCHERS = {"switch_safe_shove_30": ("g_safe", "g_shove", 30), "switch_bully_farm_50": ("g_bully", "g_farm", 50)}

# ---------------------------------------------------------------------------------------------- sizes
OBS = 103                       # m45_calibrated(obs_last_action=True)
NET_IN = OBS + K                # 116

# ---------------------------------------------------------------------------------------------- training settings
BUDGET = 4_096_000
MIDPOINT = 2_048_000
SNAP_EVERY = C46.SNAP_EVERY

# ---------------------------------------------------------------------------------------------- seeds
TRAIN_SEED = 61_000             # PPO seed of seed s = TRAIN_SEED + s (both arms)
SMOKE_ROOT = 61_050
EVAL_ROOT = 61_100
TRACE_ROOT = 61_900

_CFG = {}


def env_cfg():
    if "cfg" not in _CFG:
        from ....config import GameConfig
        cfg = GameConfig.m45_calibrated(obs_last_action=True)
        cfg.validate()
        _CFG["cfg"] = cfg
    return _CFG["cfg"]


def setup(cache: int = C46.CACHE):
    import torch
    torch.set_num_threads(1)
    from ...m45b import engine as E
    cfg = env_cfg()
    c = E._CACHES.get(cfg)
    if c is None or c.size != cache:
        if len(E._CACHES) > 4:
            E._CACHES.clear()
        E._CACHES[cfg] = E.LaneCache(cfg, cache)
    return cfg


def start_path(s: int) -> str:
    return C46.policy_path(f"ext/baseline:{s}")


def run_dir(arm: str, seed: int, phase: str = "main") -> str:
    return os.path.join(CKPT, "train", phase, arm, f"s{seed}")


_NETS: Dict[str, object] = {}


def net90(path: str):
    """A 90-input library / baseline network (digest verified)."""
    if path not in _NETS:
        _NETS[path] = C46.net_of(path)
    return _NETS[path]


def om_net(path: str):
    """An OM2-trained network (NET_IN inputs; digest verified)."""
    key = "om:" + path
    if key not in _NETS:
        from ....rl import load_checkpoint
        net, _, _ = load_checkpoint(path)
        if net.obs_size != NET_IN or net.n_actions != 61:
            raise ValueError(f"{path}: {net.obs_size} inputs / {net.n_actions} outputs, not the OM2 input")
        net.eval()
        _NETS[key] = net
    return _NETS[key]


# ---------------------------------------------------------------------------------------------- types and agents
def belief_types(eps: float = EPS):
    from ....opponent import LibraryMixtureType, NetType, ScriptedType, UniformType
    out = [ScriptedType(n, eps) for n in ARCHETYPES]
    out += [NetType(n, net90(p)) for n, p in LEARNED.items()]
    out.append(LibraryMixtureType(range(len(out)), "mixed"))
    out.append(UniformType("unknown"))
    return out


def agent(name: str):
    """An opponent agent by name (library type, mixture, switcher, held-out opponent, or "base:<s>" =
    ext/baseline:s)."""
    from ....opponent import MixtureAgent, NetAgent, ScriptedAgent, SwitchAgent
    from ...m45b.strategies import get
    if name.startswith("base:"):
        return NetAgent(net90(start_path(int(name[5:]))), name)
    if name in ARCHETYPES or name in HELDOUT_SCRIPTED:
        return ScriptedAgent(get(name), name)
    if name in LEARNED:
        return NetAgent(net90(LEARNED[name]), name)
    if name in HELDOUT_LEARNED:
        return NetAgent(net90(HELDOUT_LEARNED[name]), name)
    for table, cls in ((MIXTURES, "mix"), (HELDOUT_MIXTURES, "mix"), (SWITCHERS, "switch"), (HELDOUT_SWITCHERS, "switch")):
        if name in table:
            a, b, x = table[name]
            if cls == "mix":
                return MixtureAgent(ScriptedAgent(get(a), a), ScriptedAgent(get(b), b), x, name)
            return SwitchAgent(ScriptedAgent(get(a), a), ScriptedAgent(get(b), b), x, name)
    raise ValueError(f"unknown opponent {name!r}")


# ---------------------------------------------------------------------------------------------- statistics and I/O
seed_stats = C46.seed_stats


def mean_se(x) -> dict:
    v = np.asarray([y for y in np.ravel(x) if np.isfinite(y)], dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "ci": float("nan"), "n": 0}
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": float(v.mean()), "se": se, "ci": 1.96 * se if n > 1 else float("nan"), "n": n}


def meta(**kw) -> dict:
    m = C46.meta(**kw)
    m["study"] = "OM2"
    return m


def save_json(name: str, data: dict, out: str = OUT) -> str:
    return _save_json(name, data, out)


def load(name: str, out: str = OUT):
    return _load_json(name, out)


def write_md(name: str, text: str, out: str = OUT) -> str:
    return _write_md(name, text, out)


def run_jobs(fn, jobs, workers: int, label: str = ""):
    return C46.run_jobs(fn, jobs, workers, label)


def prereg_record() -> dict:
    with open(PREREG_DOC, "rb") as f:
        b = f.read()
    return {"document": PREREG_DOC, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def write_prereg() -> dict:
    path = os.path.join(OUT, "preregistration.json")
    if os.path.exists(path):
        old = load("preregistration")
        if old["record"]["sha256"] != prereg_record()["sha256"]:
            raise SystemExit("the OM2 pre-registration changed after it was frozen; record an amendment instead")
        return old
    d = {"meta": meta(stage="prereg"), "record": prereg_record(),
         "note": "frozen after the readiness checks and throughput, before any pilot training run"}
    save_json("preregistration", d)
    return d


def require_prereg() -> None:
    path = os.path.join(OUT, "preregistration.json")
    if not os.path.exists(path):
        raise SystemExit("freeze the OM2 pre-registration first (stage prereg)")
    if load("preregistration")["record"]["sha256"] != prereg_record()["sha256"]:
        raise SystemExit("the OM2 pre-registration document differs from its frozen SHA-256")


def amend(note: str) -> dict:
    d = load("preregistration") or {}
    d.setdefault("amendments", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note})
    save_json("preregistration", d)
    return d
