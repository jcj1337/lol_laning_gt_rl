"""Witnessed start states: the training start distribution rho_train and the held-out evaluation archive (M4 design
§10.4, §10.5.1).

A `StartState` is a champion-mode environment state (lane, champion state, phase k mod P) together with a WITNESS:
the environment seed and the action prefix that reproduce it from the standard initial state s0. Every archived state
is therefore reachable, and `verify_witness` replays it.

Archive file (JSON, schema "lanegame.start_archive", version 1):

  {"schema": "lanegame.start_archive", "version": 1, "config": {...GameConfig...},
   "generator": {free-form: the policies and seeds that produced the states},
   "states": [{"lane": [...State.to_list()], "champ": [[side A], [side B]], "phase": p,
               "witness": {"seed": int, "actions": [[a, b], ...]}, "label": "..."}, ...]}

Rules (design §10.5.1): the training archive A_train and the evaluation archive A_eval are disjoint, and A_eval is
mirror-closed (it contains the mirror of each of its states), so a mirror-symmetric policy pair scores 0 on it.
rho_train is a training device, never part of the objective; protocol E1 never uses an archive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import GameConfig
from ..state import State

ARCHIVE_SCHEMA = "lanegame.start_archive"


def champ_from_list(cfg: GameConfig, v):
    if cfg.is_m4:
        from ..champions_m4 import M4State
        return M4State.from_list(v)
    from ..champions import ChampState
    return ChampState.from_list(v)


@dataclass(frozen=True)
class StartState:
    lane: State
    champ: object
    phase: int
    witness: Optional[dict] = None          # {"seed": int, "actions": [[a, b], ...]}
    label: str = ""

    def key(self) -> tuple:
        """Identity of the decision situation (witness and label excluded)."""
        return (self.lane, self.champ, self.phase)

    def mirror(self) -> "StartState":
        """The same situation with the seats swapped; its witness is the mirrored action prefix (valid when the
        configuration's skills are symmetric, or with swap_seats)."""
        w = None
        if self.witness is not None:
            w = {"seed": self.witness["seed"], "actions": [[b, a] for a, b in self.witness["actions"]],
                 "mirrored": not self.witness.get("mirrored", False)}
        return StartState(self.lane.mirror(), self.champ.mirror(), self.phase, w, (self.label + " [mirror]").strip())

    def to_dict(self) -> dict:
        return {"lane": self.lane.to_list(), "champ": self.champ.to_list(), "phase": self.phase,
                "witness": self.witness, "label": self.label}

    @classmethod
    def from_dict(cls, d: dict, cfg: GameConfig) -> "StartState":
        return cls(State.from_list(d["lane"]), champ_from_list(cfg, d["champ"]), int(d["phase"]), d.get("witness"),
                   d.get("label", ""))


def mirror_closed(states: Sequence[StartState]) -> bool:
    keys = {s.key() for s in states}
    return all(s.mirror().key() in keys for s in states)


def close_under_mirror(states: Sequence[StartState]) -> List[StartState]:
    """The archive plus the mirror of every state not already present."""
    out = list(states)
    keys = {s.key() for s in out}
    for s in states:
        m = s.mirror()
        if m.key() not in keys:
            out.append(m)
            keys.add(m.key())
    return out


def disjoint(a: Sequence[StartState], b: Sequence[StartState]) -> bool:
    return not ({s.key() for s in a} & {s.key() for s in b})


def save_archive(path: str, states: Sequence[StartState], cfg: GameConfig, generator: Optional[dict] = None) -> None:
    from ..io import dump_json
    dump_json({"schema": ARCHIVE_SCHEMA, "version": 1, "config": cfg.to_dict(), "generator": generator or {},
               "states": [s.to_dict() for s in states]}, path)


def load_archive(path: str, cfg: Optional[GameConfig] = None) -> List[StartState]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("schema") != ARCHIVE_SCHEMA:
        raise ValueError(f"{path} is not a {ARCHIVE_SCHEMA} file")
    c = cfg if cfg is not None else GameConfig.from_dict(d["config"])
    return [StartState.from_dict(x, c) for x in d["states"]]


def replay_witness(cfg: GameConfig, witness: dict):
    """Replay a witness from s0: returns the environment after the prefix (seed, actions)."""
    from .core import LaneEnv
    env = LaneEnv(cfg)
    env.reset(seed=witness["seed"])
    for a, b in witness["actions"]:
        r = env.step(a, b)
        if r.done:
            raise ValueError("the witness prefix ended the episode")
    return env


def verify_witness(cfg: GameConfig, st: StartState) -> bool:
    """True iff the witness reproduces the state exactly (mirrored witnesses are checked on the mirrored state)."""
    if st.witness is None:
        return False
    target = st.mirror() if st.witness.get("mirrored") else st
    w = target.witness if st.witness.get("mirrored") else st.witness
    env = replay_witness(cfg, w)
    return (env.state, env.champ, env.k % cfg.phase_period) == (target.lane, target.champ, target.phase)
