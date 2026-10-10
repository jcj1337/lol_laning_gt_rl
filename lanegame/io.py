"""Serializable output format for solved games (consumed by Phase 2/3).

Schema of `write_solution` (JSON, version 1):

{
  "schema": "lanegame.solution", "version": 1,
  "config": {...GameConfig fields...},
  "horizon": K, "value": V*_0(s_0),
  "actions": {"0": "SlowPush", "1": "Shove", "2": "Freeze"},
  "state_fields": ["ax","ah","ar","asg","bx","bh","br","bsg","va","vb"],
  "states": [[...state as list...], ...],              # unique states, index = state id
  "layers": [[state ids at k=0], [ids at k=1], ...],
  "nodes": [ {"k": k, "s": id, "V": value, "gap": certified duality gap,
              "A": [legal action codes], "x": [probabilities],
              "B": [legal action codes], "y": [probabilities],
              "Q": [[row-major action values]]}, ... ],
  "choices": [ {"k": k, "s": pre-choice state id, "opts_a": [[res, vision], ...],
                "opts_b": [...], "x": [...], "y": [...], "W": value}, ... ],
  "stats": {...}
}
"""

from __future__ import annotations

import json
from typing import Dict

import numpy as np

from .config import ACTION_NAMES, GameConfig
from .solver import EquilibriumSolution
from .state import State

STATE_FIELDS = ["ax", "ah", "ar", "asg", "bx", "bh", "br", "bsg", "va", "vb"]


def solution_to_dict(sol: EquilibriumSolution, include_q: bool = True) -> Dict:
    ids: Dict[State, int] = {}

    def sid(s: State) -> int:
        if s not in ids:
            ids[s] = len(ids)
        return ids[s]

    nodes = []
    layers = []
    for k, layer in enumerate(sol.layers):
        layers.append([sid(s) for s in layer])
        for s in layer:
            la, lb = sol.legal[k][s]
            node = {"k": k, "s": sid(s), "V": sol.V[k][s], "gap": sol.gaps[k][s],
                    "A": list(la), "x": sol.x[k][s].tolist(),
                    "B": list(lb), "y": sol.y[k][s].tolist()}
            if include_q:
                node["Q"] = sol.Q[k][s].tolist()
            nodes.append(node)
    choices = []
    for key, val in sol.choice_val.items():
        k, pre, oa, ob = key
        choices.append({"k": k, "s": sid(pre),
                        "opts_a": [[r, bool(v)] for r, v in oa],
                        "opts_b": [[r, bool(v)] for r, v in ob],
                        "x": sol.choice_x[key].tolist(), "y": sol.choice_y[key].tolist(),
                        "W": val})
    states = [None] * len(ids)
    for s, i in ids.items():
        states[i] = s.to_list()
    return {
        "schema": "lanegame.solution", "version": 1,
        "config": sol.game.cfg.to_dict(),
        "horizon": sol.game.horizon,
        "root": sol.root.to_list(), "k0": sol.k0,
        "value": sol.value,
        "actions": {str(a): n for a, n in ACTION_NAMES.items()},
        "state_fields": STATE_FIELDS,
        "states": states, "layers": layers, "nodes": nodes, "choices": choices,
        "stats": sol.stats.to_dict(),
    }


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def dump_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, cls=_Enc, indent=1)


def write_solution(sol: EquilibriumSolution, path: str, include_q: bool = True) -> None:
    dump_json(solution_to_dict(sol, include_q), path)


def read_solution(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    assert d.get("schema") == "lanegame.solution"
    d["config_obj"] = GameConfig.from_dict(d["config"])
    return d
