"""M5e traces (exploratory and descriptive; nothing here changes a registered statistic or a classification).

Representative games. A rule fixed before any trace was read picks one game per traced cell. It takes the game whose
focal lead at turn 114 is the upper median of the cell's 2N games (pairs times seat orientations), and breaks ties by
pair, then orientation. The traced games are every core attacker against its own target (12), the seed-0 head to head
om:0 against sp:0 (M5d P2b), and om:0 and sp:0 against the S_in archetype g_shove (seed 0).

Each game is replayed from its seed with `games.play(record=True)`, which must reproduce the stored per-turn reward
of the evaluation record exactly, then exported as a viewer recording (schema lanegame.viewer_recording, config
m45_calibrated, s0, the game's integer chance seed; `streamlit run lanegame/viewer/app.py`, mode "Recorded
trajectory", folder results/m5/traces). The export checks that a LaneEnv replay of the actions from that seed gives
the same per-turn rewards, which is what the viewer does.

Behaviour. For each core attacker, the first BEHAVIOUR_PAIRS pairs of its own-target evaluation cell are replayed
twice on the same seeds: attacker against target, and the attacker's starting point (a copy of the target) against
the target. Each action code is split into its components (`env.observation_m4.encode_last_action`: absent, Ward,
Recall, the behaviour Hold / Yield / Trade / Safe, the wave slot Last Hit / Thin / Slow Push / Shove, Pressure), and
the shares are compared in the lane phase (turns 0-113) and after it (114-499), for the attacker's seat and the
target's. A share difference describes what the attacker does differently; it does not show why it pays.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import common as C

TDIR = os.path.join(C.OUT, "traces")
BEHAVIOUR_PAIRS = 16
WINDOWS = {"lane (0-113)": (0, 114), "after (114-499)": (114, 500)}


def _components() -> List[str]:
    from ....env.observation_m4 import LAST_ACTION_NAMES
    return list(LAST_ACTION_NAMES[1:])


def _encode(codes: np.ndarray) -> np.ndarray:
    """[T] codes -> [T, 12] component indicators (absent, ward, recall, hold, ..., pressure)."""
    from ....env.observation_m4 import encode_last_action
    cfg = C.arm_cfg("om")
    cache: Dict[int, List[float]] = {}
    rows = []
    for c in codes:
        c = int(c)
        if c not in cache:
            cache[c] = encode_last_action(c, cfg)[1:]
        rows.append(cache[c])
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------------------------- selection
def median_game(path: str, focal: int = 0) -> dict:
    """The upper-median game of a cell record by the focal's lead at turn 114 (module docstring)."""
    from . import games as G
    R = G.load(path)[f"f{focal}__r"]
    v = R[:, :, :G.READ_TURN].sum(-1)
    order = sorted((float(v[m, o]), m, o) for m in range(v.shape[0]) for o in (0, 1))
    val, m, o = order[len(order) // 2]
    return {"pair": m, "orientation": o, "lead_114": val, "games": len(order),
            "cell_mean_lead_114": float(v.mean())}


def trace_cells() -> List[tuple]:
    """The traced games (module docstring): (evaluation job, focal index, recording name, label)."""
    from . import attackers as AT
    from . import evaluate as EV
    out = []
    for j in AT.eval_jobs():
        if j["key"].endswith("|target"):
            k = j["key"].split("|")
            out.append((j, 0, f"attacker_{k[2]}_{k[3]}_vs_target",
                        f"the {k[2]}:{k[3][1:]} attacker (focal) against its own target {k[2]}:{k[3][1:]}"))
    fam = {j["key"]: j for j in EV.family_jobs()}
    out.append((fam["h2h|s0|same"], 0, "h2h_om_s0_vs_sp_s0", "om:0 (focal) against sp:0, the M5d head to head"))
    j = fam["fam|s0|S_in|opp__g_shove"]
    out += [(j, 0, "om_s0_vs_g_shove", "om:0 (focal) against the S_in archetype g_shove"),
            (j, 1, "sp_s0_vs_g_shove", "sp:0 (focal) against the S_in archetype g_shove")]
    return out


# ---------------------------------------------------------------------------------------------- export
class _Replay:
    """A seat that plays recorded codes (the viewer's replay)."""

    def __init__(self, codes: np.ndarray, seat: int):
        self.codes, self.seat = codes, seat

    def act(self, obs, rng=None) -> int:
        return int(self.codes[obs.k][self.seat])


def _seats(j: dict, focal: int, o: int):
    f, opp = tuple(j["focals"][focal]), tuple(j["opponent"])
    return (f, opp) if o == 0 else (opp, f)


def export(j: dict, focal: int = 0, name: Optional[str] = None, label: str = "") -> dict:
    """Replay the selected game of cell job `j` (focal index), verify it, and write a viewer recording."""
    from ....env import LaneEnv
    from ....viewer.session import RECORDING_SCHEMA, recording_stamp
    from ..m43b.trajectories import record_game
    from . import games as G
    sel = median_game(j["path"], focal)
    m, o = sel["pair"], sel["orientation"]
    sd = G.game_seed(j["root"], j["cell"], m)
    a, b = _seats(j, focal, o)
    T = int(j["T"])
    g = G.play(a, b, sd, T, record=True)
    stored = G.load(j["path"])[f"f{focal}__r"][m, o]
    focal_r = g["r"] if o == 0 else -g["r"]
    if not np.array_equal(focal_r, stored):
        raise RuntimeError(f"{j['key']}: the replay does not reproduce the stored record (pair {m}, orientation {o})")
    cfg = C.arm_cfg("sp")
    env = LaneEnv(cfg, max_turns=T)
    env.reset(seed=sd)
    for i in range(T):
        r = env.step(int(g["codes"][i, 0]), int(g["codes"][i, 1]))
        if abs(r.rewards[0] - float(g["r"][i])) > 1e-5:
            raise RuntimeError(f"{j['key']}: the LaneEnv replay differs at turn {i}")
    rec = record_game(cfg, _Replay(g["codes"], 0), _Replay(g["codes"], 1), sd, T)
    rec.pop("action_shares", None)                         # m4_lasthit labels; the components are in "m5"
    names = [a[1], b[1]]
    comp = _components()
    shares = {s: dict(zip(comp, _encode(g["codes"][:G.READ_TURN, s]).mean(0).round(4).tolist())) for s in (0, 1)}
    out = {"schema": RECORDING_SCHEMA, "version": 1, **recording_stamp("m45_calibrated"), "config": "m45_calibrated",
           "scenario": "s0", "gap": "equal", "seed": int(sd), "names": names, "label": label, **rec,
           "m5": {"cell": j["key"], "focal": focal, "focal_seat": "A" if o == 0 else "B", "selection": sel,
                  "rule": "upper median of the focal's lead at turn 114 over the cell's games",
                  "lead_114_A": float(g["r"][:G.READ_TURN].sum()),
                  "component_shares_turns_0_113": shares,
                  "belief_log_posterior": {("A" if s == 0 else "B"): lp[[t - 1 for t in G.CHECK_TURNS]].tolist()
                                           for s, lp in g["lp"].items()},
                  "belief_turns": list(G.CHECK_TURNS), "belief_types": list(C.BELIEF_TYPES)}}
    os.makedirs(TDIR, exist_ok=True)
    name = name or j["key"].replace("|", "__").replace(":", "_")
    path = os.path.join(TDIR, f"{name}.json")
    C.jdump(path, out)
    return {"path": path.replace("\\", "/"), "seed": int(sd), "names": names, "selection": sel,
            "turns": T, "payoff_A": rec["payoff_A"], "counters": rec["counters"]}


def _export_job(args) -> tuple:
    j, focal, name, label = args
    return name, export(j, focal, name, label)


def export_all(workers: int = 6) -> dict:
    """Every traced game (module docstring) -> results/m5/traces/*.json and the index results/m5/traces.json."""
    t0 = time.perf_counter()
    res = dict(C.run_jobs(_export_job, trace_cells(), workers, "M5 traces"))
    data = {"meta": C.meta(stage="traces"), "rule": __doc__.split("Representative games.")[1].split("Each game")[0]
            .strip(), "how_to_view": "streamlit run lanegame/viewer/app.py -> 'Recorded trajectory', folder "
            "results/m5/traces", "recordings": res, "wall_s": time.perf_counter() - t0}
    C.save_json("traces", data)
    return data


# ---------------------------------------------------------------------------------------------- behaviour
def behaviour_job(args) -> tuple:
    """(kind, seed, pairs): the component shares of attacker vs target and start vs target on the same games."""
    from . import attackers as AT
    from . import games as G
    kind, s, K = args
    j = next(x for x in AT.eval_jobs(AT.CORE_KINDS, (s,)) if x["key"].endswith(f"|{kind}|s{s}|target"))
    att, tgt = tuple(j["focals"][0]), tuple(j["opponent"])
    start = ("net", f"{kind}:{s}")                          # the attacker's starting point plays as the target does
    acc = {who: {w: [] for w in WINDOWS} for who in ("attacker", "start", "target_vs_attacker", "target_vs_start")}
    lead = {"attacker": [], "start": []}
    for m in range(K):
        sd = G.game_seed(j["root"], j["cell"], m)
        for o in (0, 1):
            for who, f in (("attacker", att), ("start", start)):
                a, b = (f, tgt) if o == 0 else (tgt, f)
                g = G.play(a, b, sd, int(j["T"]), record=True)
                fs = o
                lead[who].append(float((g["r"] if o == 0 else -g["r"])[:G.READ_TURN].sum()))
                for w, (lo, hi) in WINDOWS.items():
                    acc[who][w].append(_encode(g["codes"][lo:hi, fs]).mean(0))
                    acc[f"target_vs_{who}"][w].append(_encode(g["codes"][lo:hi, 1 - fs]).mean(0))
    comp = _components()
    out = {who: {w: dict(zip(comp, np.mean(v, 0).tolist())) for w, v in d.items()} for who, d in acc.items()}
    out["lead_114"] = {k: float(np.mean(v)) for k, v in lead.items()}
    return f"{kind}_s{s}", out


def behaviour(workers: int = 6, pairs: int = BEHAVIOUR_PAIRS) -> dict:
    """Every core attacker's behaviour summary (module docstring) -> results/m5/attack_behaviour.json, with the
    attacker - start share differences per arm over the six seeds (seed intervals)."""
    from . import analysis as AN
    from . import attackers as AT
    t0 = time.perf_counter()
    jobs = [(k, s, pairs) for k in AT.CORE_KINDS for s in C.SEEDS]
    res = dict(C.run_jobs(behaviour_job, jobs, workers, "M5 attacker behaviour"))
    comp = _components()
    diff = {}
    for kind in AT.CORE_KINDS:
        for w in WINDOWS:
            for who, ref in (("attacker", "start"), ("target_vs_attacker", "target_vs_start")):
                for c in comp:
                    per = {s: res[f"{kind}_s{s}"][who][w][c] - res[f"{kind}_s{s}"][ref][w][c] for s in C.SEEDS}
                    diff.setdefault(kind, {}).setdefault(w, {}).setdefault(who, {})[c] = AN.interval(per)
    data = {"meta": C.meta(stage="attack_behaviour"), "pairs": pairs, "windows": WINDOWS, "per_attacker": res,
            "difference_attacker_minus_start": diff, "wall_s": time.perf_counter() - t0}
    C.save_json("attack_behaviour", data)
    return data


__all__ = ["median_game", "trace_cells", "export", "export_all", "behaviour", "behaviour_job", "TDIR",
           "BEHAVIOUR_PAIRS", "WINDOWS"]
