"""The job lists of M4.8 (protocol §3-§5): the pilot, Stage 1 validation and Stage 2 natural checkpoints.

Every continuation is T_MAX = 40 turns; the windows 5, 10, 20, 30, 40 are prefixes of the same continuations.
Repetitions K per state are fixed in advance per family (protocol §4; chosen from the pilot's measured variance) and
never extended after results are seen.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from . import common as C

K_PILOT = 64
K_VAL = 128                    # Stage 1: selected cases, mirrors, controlled variants
K_MAIN = 128                   # Stage 2 primary anchor (t0 = 94): own package, other package, alternative, variant
K_SENS = 64                    # Stage 2 sensitivity anchors (t0 = 84, 104): own package only
K_REPL = 32                    # Stage 2 turn-50 anchor (the replication of M4.7's 50 -> 94 slope): own package only
PILOT_GAMES = 2
ALT_EVERY = 4                  # the alternative-policy subset: games g with g % ALT_EVERY == 0 (12 per package)


def raw(family: str, key: str) -> str:
    return os.path.join(C.CKPT, "roll", family, key.replace("|", "_") + ".npz")


def own(env: str, s: int, variant: str = "orig", mirror: bool = False) -> dict:
    p = C.package(env, s)
    return {"env": env, "a": p, "b": p, "variant": variant, "mirror": mirror}


# ---------------------------------------------------------------------------------------------- pilot
def pilot_states() -> List[dict]:
    from ...m45b import states as S
    from ..m47 import streams as SM
    out = []
    for env in C.ENVS:
        cfg = C.setup(env)
        for s in C.SEEDS:
            for g in range(PILOT_GAMES):
                st = SM.self_play_state(cfg, C.net_of(C.package(env, s)), S.s0(cfg),
                                        C.PILOT_ROOT * 1000 + 100 * s + g, C.T0_PRIMARY)
                out.append({"env": env, "s": s, "g": g, "state": C.state_to_list(st)})
    return out


def pilot_jobs(states: List[dict]) -> List[dict]:
    js = []
    for i, st in enumerate(states):
        key = f"pilot|{st['env']}|s{st['s']}|g{st['g']}"
        js.append({"key": key, "path": raw("pilot", key), "start": st["state"], "index": i, "root": C.PILOT_ROLL,
                   "K": K_PILOT, "T": C.T_MAX, "combos": {"own": own(st["env"], st["s"])}, "full": False})
    return js


# ---------------------------------------------------------------------------------------------- Stage 1
def validation_jobs(selected: List[dict]) -> List[dict]:
    js = []
    for i, st in enumerate(selected):
        env, s = st["env"], st["s"]
        combos: Dict[str, dict] = {"own": own(env, s)}
        if st["case"] == "returner":
            combos["mirror"] = own(env, s, mirror=True)
        if st["case"] == "due":
            combos["A_fresh"] = own(env, s, "A_fresh")
            combos["A_away1"] = own(env, s, "A_away1")
        key = f"val|{st['case']}|{st['X']}|{env}|s{s}|g{st['g']}|t{st['t0']}"
        js.append({"key": key, "path": raw("val", key), "start": st["state"], "index": i, "root": C.VAL_ROLL,
                   "K": K_VAL, "T": C.T_MAX, "combos": combos, "full": True})
    return js


# ---------------------------------------------------------------------------------------------- Stage 2
def state_index(env: str, s: int, g: int, t0: int) -> int:
    return ((C.ENVS.index(env) * len(C.SEEDS) + s) * 100 + g) * len(C.T0S) + C.T0S.index(t0)


def is_due(d: dict) -> bool:
    return (d["status_A"] == 0 and d["status_B"] == 0 and d["since_return_A"] >= 10 and d["unspent_A"] >= 2.0
            and d["hp_A"] >= 0.3 and d["holds_freeze_B"] == 0)


def natural_jobs(states: List[dict]) -> List[dict]:
    js = []
    for st in states:
        env, s, g, t0 = st["env"], st["s"], st["g"], st["t0"]
        other = C.ENVS[1 - C.ENVS.index(env)]
        if t0 == C.T0_PRIMARY:
            combos = {"own": own(env, s), "pkg_other": own(other, s)}
            if g % ALT_EVERY == 0:
                alt = C.package(env, C.other_seed(s))
                combos["alt"] = {"env": env, "a": alt, "b": alt, "variant": "orig", "mirror": False}
            if is_due(st["desc"]):
                combos["A_fresh"] = own(env, s, "A_fresh")
            K = K_MAIN
        else:
            combos, K = {"own": own(env, s)}, (K_REPL if t0 == C.T0_REPL else K_SENS)
        key = f"nat|{env}|s{s}|g{g}|t{t0}"
        js.append({"key": key, "path": raw("nat", key), "start": st["state"], "index": state_index(env, s, g, t0),
                   "root": C.NAT_ROLL, "K": K, "T": C.T_MAX, "combos": combos, "full": False})
    return js


__all__ = ["K_PILOT", "K_VAL", "K_MAIN", "K_SENS", "K_REPL", "raw", "own", "pilot_states", "pilot_jobs", "validation_jobs",
           "natural_jobs", "state_index", "is_due"]
