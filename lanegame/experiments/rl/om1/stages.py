"""OM1 stages (protocol §3-§5): the selection matrix (A), the evaluation against the library (B) and against
out-of-library opponents (C), and the smoke check. Every job is resumable (one npz per cell).

Values. A pair's value for the responder is the mean over its two orientations of the responder's undiscounted
earned-gold lead at turn READ_TURN = 114 (= the lane-exit readout E_20 at turn 94 in expectation: E_20 is the expected
lead 20 turns after the checkpoint under the same pair; M4.8).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np

from . import common as C
from . import games as G
from . import library as LB

N_SEL = 64                     # pairs per selection cell (stage A)
N_EVAL = 96                    # pairs per evaluation cell (stage B)
N_OOL = 64                     # pairs per out-of-library cell (stage C)
MENU = ("pi",) + C.LIBRARY     # responders of the selection matrix ("pi" = the focal policies, pooled over seeds)


def raw(stage: str, key: str) -> str:
    return os.path.join(C.CKPT, stage, key.replace("|", "__").replace(":", "_") + ".npz")


def responder_name(r: str, m: int) -> str:
    return f"pi:{m % len(C.SEEDS)}" if r == "pi" else r


# ---------------------------------------------------------------------------------------------- stage A
def sel_job(j: dict) -> tuple:
    path = j["path"]
    if os.path.exists(path):
        return j["key"], {"path": path, "cached": True}
    cfg = C.setup()
    ls = C.lane_step(cfg)
    N, T = int(j["N"]), int(j["T"])
    R = np.zeros((N, 2, T), dtype=np.float32)
    t0 = time.perf_counter()
    for m in range(N):
        sd = C.game_seed(C.SEL_ROOT, j["cell"], m)
        for o in (0, 1):
            R[m, o] = G.play(cfg, ("type", responder_name(j["r"], m)), j["opp"], sd, T, o, lane_step=ls)["r"]
    _save(path, r=R, elapsed=np.asarray(time.perf_counter() - t0))
    return j["key"], {"path": path, "elapsed": time.perf_counter() - t0}


def sel_jobs() -> List[dict]:
    js = []
    for ri, r in enumerate(MENU):
        for ti, t in enumerate(C.LIBRARY):
            key = f"sel|{r}|{t}"
            js.append({"key": key, "path": raw("sel", key), "r": r, "opp": t, "cell": ri * 20 + ti,
                       "N": N_SEL, "T": C.T_SEL})
    return js


# ---------------------------------------------------------------------------------------------- stages B and C
def eval_job(j: dict) -> tuple:
    """`j`: key, path, s, opp, cell, root, N, T, responders {label: spec}, track {label: bool}."""
    path = j["path"]
    if os.path.exists(path):
        return j["key"], {"path": path, "cached": True}
    cfg = C.setup()
    ls = C.lane_step(cfg)
    types = LB.library()
    N, T = int(j["N"]), int(j["T"])
    arrays: Dict[str, np.ndarray] = {}
    sw: Dict[str, list] = {}
    t0 = time.perf_counter()
    for lab, spec in j["responders"].items():
        R = np.zeros((N, 2, T), dtype=np.float32)
        bel = None
        for m in range(N):
            sd = C.game_seed(j["root"], j["cell"], m)
            for o in (0, 1):
                tr = bool(j["track"].get(lab, False))
                g = G.play(cfg, tuple(spec), j["opp"], sd, T, o, track=tr,
                           eps_list=C.EPS_SENS if tr else (), types=types, lane_step=ls)
                R[m, o] = g["r"]
                if "belief" in g:
                    if bel is None:
                        bel = {k: np.zeros((N, 2) + v.shape, dtype=np.float32) for k, v in g["belief"].items()}
                    for k, v in g["belief"].items():
                        bel[k][m, o] = v
                if "switch" in g:
                    sw.setdefault(lab, []).append(g["switch"])
        arrays[f"{lab}__r"] = R
        if bel is not None:
            for k, v in bel.items():
                arrays[f"{lab}__bel_{k}"] = v
    _save(path, **arrays, labels=np.asarray(list(j["responders"])), switch=np.asarray(json.dumps(sw)),
          elapsed=np.asarray(time.perf_counter() - t0))
    return j["key"], {"path": path, "elapsed": time.perf_counter() - t0}


def responders(s: int, opp: str, sel: dict, library_opp: bool, theta: float = C.THETA) -> Dict[str, list]:
    """The responders of an evaluation cell: pi_s; the oracle (the opponent's selected responder, if not pi); the best
    fixed responder (if not pi); the Bayesian switcher."""
    rstar, rfix = sel["rstar"], sel["rfix"]
    out = {"pi": ["type", f"pi:{s}"]}
    if library_opp and rstar[opp] != "pi":
        out["oracle"] = ["type", rstar[opp]]
    if rfix != "pi" and (not library_opp or rfix != rstar[opp]):
        out["fixed"] = ["type", rfix]
    out["switch"] = ["switch", s, rstar, theta]
    return out


def eval_jobs(sel: dict) -> List[dict]:
    js = []
    for s in C.SEEDS:
        for ti, t in enumerate(C.LIBRARY):
            key = f"eval|s{s}|{t}"
            rs = responders(s, t, sel, True)
            js.append({"key": key, "path": raw("eval", key), "s": s, "opp": t, "cell": s * 20 + ti,
                       "root": C.EVAL_ROOT, "N": N_EVAL, "T": C.T_EVAL, "responders": rs,
                       "track": {"pi": True}})
    return js


def ool_jobs(sel: dict) -> List[dict]:
    js = []
    for s in C.SEEDS:
        for oi, o in enumerate(C.OOL):
            key = f"ool|s{s}|{o}"
            rs = responders(s, o, sel, False)
            rs.pop("fixed", None)
            js.append({"key": key, "path": raw("ool", key), "s": s, "opp": o, "cell": s * 20 + oi,
                       "root": C.OOL_ROOT, "N": N_OOL, "T": C.T_EVAL, "responders": rs,
                       "track": {"pi": True}})
    return js


def _save(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if "switch" in d:
        d["switch"] = json.loads(str(d["switch"]))
    if "labels" in d:
        d["labels"] = [str(x) for x in d["labels"]]
    return d


def pair_lead(R: np.ndarray, t: int = C.READ_TURN) -> np.ndarray:
    """[N] the responder's lead after t turns, the mean of its two orientations."""
    return R[:, :, :t].sum(-1).mean(-1)


# ---------------------------------------------------------------------------------------------- selection
def select() -> dict:
    """r*(type): the menu responder with the highest selection-stage value against that type, if it beats pi by at
    least M (else pi); r_fix: the responder with the highest mean value over the library (uniform prior), if it beats pi
    by at least M (else pi). Values: lead at turn READ_TURN, mean over the cell's pairs."""
    V = {}
    for j in sel_jobs():
        V[(j["r"], j["opp"])] = float(pair_lead(load(j["path"])["r"]).mean())
    out = {"meta": C.meta(stage="select"), "values": {f"{r}|{t}": v for (r, t), v in V.items()}, **choose(V)}
    C.save_json("selection", out)
    return out


def choose(V: Dict[tuple, float], menu=MENU, types=C.LIBRARY, m: float = C.M) -> dict:
    """The selection rule from a value table V[(responder, type)] (protocol §3)."""
    rstar, gain = {}, {}
    for t in types:
        best = max(menu, key=lambda r: V[(r, t)])
        gain[t] = V[(best, t)] - V[("pi", t)]
        rstar[t] = best if gain[t] >= m else "pi"
    mean_v = {r: float(np.mean([V[(r, t)] for t in types])) for r in menu}
    best_fix = max(menu, key=lambda r: mean_v[r])
    rfix = best_fix if mean_v[best_fix] - mean_v["pi"] >= m else "pi"
    return {"rstar": rstar, "selection_gain": gain, "mean_value": mean_v, "rfix": rfix, "rfix_candidate": best_fix}


# ---------------------------------------------------------------------------------------------- smoke
def smoke(workers: int) -> dict:
    """Pipeline checks and throughput before the protocol is frozen (seeds SMOKE_ROOT; nothing is kept for analysis)."""
    cfg = C.setup()
    ls = C.lane_step(cfg)
    types = LB.library()
    checks = {}
    sd = C.game_seed(C.SMOKE_ROOT, 0, 0)
    # determinism and paired-seat symmetry
    g1 = G.play(cfg, ("type", "pi:0"), "g_bully", sd, 40, 0, lane_step=ls)["r"]
    g2 = G.play(cfg, ("type", "pi:0"), "g_bully", sd, 40, 0, lane_step=ls)["r"]
    checks["deterministic re-run"] = bool(np.array_equal(g1, g2))
    a = G.play(cfg, ("type", "peer"), "peer", sd, 40, 0, lane_step=ls)["r"]
    b = G.play(cfg, ("type", "peer"), "peer", sd, 40, 1, lane_step=ls)["r"]
    checks["self-play pair lead is exactly 0"] = bool(abs(a.sum() + b.sum()) < 1e-9)
    # the belief identifies a scripted and a learned opponent, the posterior is normalized
    tr = {}
    for opp in ("g_bully", "attacker"):
        g = G.play(cfg, ("type", "pi:0"), opp, sd, 60, 0, track=True, eps_list=C.EPS_SENS, types=types,
                   lane_step=ls)
        post = np.exp(g["belief"]["full"][-2])                     # after 60 turns
        tr[opp] = {"p_true_60": float(post[C.LIBRARY.index(opp)]), "sum": float(post.sum()),
                   "map": C.LIBRARY[int(np.argmax(post))]}
    checks["posterior normalized"] = all(abs(v["sum"] - 1) < 1e-9 for v in tr.values())
    # the switcher runs and switches only with a selected responder
    rstar = {t: "pi" for t in C.LIBRARY}
    rstar["g_safe"] = "g_bully"
    g = G.play(cfg, ("switch", 0, rstar, C.THETA), "g_safe", sd, 60, 0, types=types, lane_step=ls)
    checks["switcher runs and records"] = "switch" in g and "belief" in g
    gp = G.play(cfg, ("type", "pi:0"), "g_safe", sd, 60, 0, lane_step=ls)["r"]
    first = g["switch"]["first_switch"]
    checks["switcher equals pi until its first switch"] = bool(
        np.array_equal(g["r"][:first if first >= 0 else 60], gp[:first if first >= 0 else 60]))
    # throughput of the three kinds of games (one worker)
    tp = {}
    for lab, (resp, opp, track) in {"script_vs_script": (("type", "g_shove"), "g_farm", False),
                                    "pi_vs_script": (("type", "pi:0"), "g_shove", False),
                                    "pi_vs_net": (("type", "pi:0"), "attacker", False),
                                    "pi_vs_net_tracked": (("type", "pi:0"), "attacker", True),
                                    "switch_vs_script": (("switch", 0, rstar, C.THETA), "g_safe", False)}.items():
        t0 = time.perf_counter()
        n = 0
        for m in range(3):
            G.play(cfg, resp, opp, C.game_seed(C.SMOKE_ROOT, 1, m), C.T_EVAL, m % 2, track=track,
                   eps_list=C.EPS_SENS if track else (), types=types, lane_step=ls)
            n += C.T_EVAL
        tp[lab] = n / (time.perf_counter() - t0)
    out = {"meta": C.meta(stage="smoke"), "checks": checks, "tracking": tr, "switch": g["switch"],
           "turns_per_s_one_worker": tp}
    C.save_json("smoke", out)
    return out


__all__ = ["MENU", "N_SEL", "N_EVAL", "N_OOL", "sel_job", "sel_jobs", "eval_job", "eval_jobs", "ool_jobs",
           "responders", "select", "smoke", "load", "pair_lead", "raw"]
