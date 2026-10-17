"""M5c's strategic counterfactual battery (design §7.2): the M4.5c battery (`m45c.battery`, with the M4.6b measurement
fixes it already carries: behaviour-only twins, lane-only positions, the economic channels) for policies of both arms.

Reused unchanged by import: the families, predicates, candidate tables, protocols, horizons and their exceptions,
checkpoints, constructed states and grids, the validation subset and the record format (`m45c.battery`, `m45c.horizon`).
Adapted here for history-dependent policies (players module docstring):
    collect_job  natural states from the policy's own self-play, both seats `players.crn_policy` (CRN sampling), and
                 for an om policy each seat's belief record (log posterior, updates, previous codes) at the state;
                 seat-B states are mirrored to seat A with the records swapped (`players.mirror_info`)
    state_job    `m45c.battery.state_job` with `players.crn_policy` for the evaluated policy and its "policy" / "modal"
                 opponent (both restored from the state's records; constructed cells start at the prior with no previous
                 action), scripted protocols as before, and the critic bootstrap from the evaluated policy itself
                 (`players.critic_of`: an om policy's value reads its belief at s_h)
For a 90-input policy both functions reproduce the M4.5c battery's rollouts exactly given the same states and seeds
(tested in readiness), so M4.5c's protocol and its new seeds apply to the sp arm unchanged.

Seeds: state games BATTERY_STATE_ROOT * 10 + family offset + game; rollouts BATTERY_ROLL_ROOT * 100 + r.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import common as C
from . import players as PL

BOUT = os.path.join(C.CKPT, "battery")


def _B():
    from ..m45c import battery as B
    return B


# ---------------------------------------------------------------------------------------------- states
def self_play(spec: str, sd: int, turns: int) -> List[tuple]:
    """One self-play game of `spec` from s0 (CRN sampling): [(lane, m4, k, info)] at the start of every turn, info the
    seats' belief records (None for a stateless policy)."""
    from ....champions import N_CHANCE
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    from ...m45b import states as S
    cfg = C.arm_cfg("sp")
    ls = E.lane_cache(cfg)
    ctx = context(cfg)
    lane, m4, k0 = S.s0(cfg)
    pa, pb = PL.crn_policy(spec, k0, sd, 0, turns), PL.crn_policy(spec, k0, sd, 1, turns)
    stateful = isinstance(pa, PL.OMPolicy)
    rng = np.random.default_rng(int(sd))
    out = []
    for i in range(turns):
        info = {0: pa.state(), 1: pb.state()} if stateful else None
        out.append((lane, m4, k0 + i, info))
        u = rng.random(N_CHANCE)
        a, b = E.decide(pa, cfg, lane, m4, 0, i), E.decide(pb, cfg, lane, m4, 1, i)
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=ls, u=u, ctx=ctx)
        E.observe_turn(pa, cfg, lane, m4, k0 + i, (a, b))
        E.observe_turn(pb, cfg, lane, m4, k0 + i, (a, b))
        lane, m4 = t.lane.state, t.champ
    return out


def collect_job(args) -> tuple:
    """(spec, family, n): natural states of a battery family from the policy's own self-play (m45c.battery.collect_job's
    rules: 300-turn games, min turn 20, early families 60 turns / min turn 3 / focal level <= 10, one state per game by a
    seeded shuffle), with the seats' belief records."""
    spec, fam, n = args
    B = _B()
    from ..m45c import horizon as HZ
    C.setup("sp")
    cfg = C.arm_cfg("sp")
    kind, pfam = B.FAMILIES[fam][:2]
    pred = HZ.predicates()[pfam]
    turns, min_turn = (B.EARLY_TURNS, 3) if kind == "early" else (300, 20)
    if kind == "early":
        pred = B._early(pred)
    off = list(B.FAMILIES).index(fam) * 1000
    out, games = [], 0
    for g in range(400):
        sd = C.BATTERY_STATE_ROOT * 10 + off + g
        states = self_play(spec, sd, turns)
        cand = [(i, s) for i, (lane, m4, k, _) in enumerate(states) if i >= min_turn for s in (0, 1)
                if pred(cfg, lane, m4, s)]
        np.random.default_rng(sd).shuffle(cand)
        for i, s in cand[:1]:
            lane, m4, k, info = states[i]
            if s == 1:
                lane, m4, info = lane.mirror(), m4.mirror(), PL.mirror_info(info)
            out.append({"start": (lane, m4, k), "info": info, "seat": s, "turn": i, "seed": sd})
        games += 1
        if len(out) >= n:
            break
    return (spec, fam), {"states": out[:n], "games": games}


# ---------------------------------------------------------------------------------------------- one start
def _opponent(proto: str, spec: str, k0: int, seed: int, horizon: int, info):
    from ...m45b.strategies import get
    if proto == "policy":
        return PL.crn_policy(spec, k0, seed, 1, horizon, info)
    if proto == "modal":
        return PL.crn_policy(spec, k0, seed, 1, horizon, info, greedy=True)
    return get(proto)


def job_path(tag: str, fam: str, idx: int, proto: str, horizon: int, root: str = BOUT) -> str:
    return _B().job_path(tag, fam, idx, proto, horizon, root)


def state_job(args) -> tuple:
    """(spec, tag, family, index, start, info, protocol, horizon, label, K, root, roll_root): m45c.battery.state_job for
    any M5 policy (module docstring); the same record format."""
    spec, tag, fam, idx, start, info, proto, horizon, label, K, root, roll_root = args
    B = _B()
    from ..m45c import horizon as HZ
    from ...m45b import engine as E
    path = job_path(tag, fam, idx, proto, horizon, root)
    key = f"{tag}|{fam}|{idx}|{proto}|{horizon}"
    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return key, json.load(f)
    C.setup("sp")
    cfg = C.arm_cfg("sp")
    k0 = start[2]
    cps = B.checkpoints(horizon)
    lcps = [c for c in HZ.LEDGER_CP if c <= horizon]
    keys = [f"{k}_{x}" for k in HZ.LEDGER_KEYS for x in ("A", "B")] + list(HZ.STATE_KEYS) + ["V"]
    cands = {"policy": None, **B.candidates(fam)}
    series = B.FAMILIES[fam][2] in ("B_freeze", "F_freeze_c")
    t0 = time.perf_counter()
    GQ, L, SR, first = {}, {}, {}, {}
    for name, cspec in cands.items():
        G = {c: [] for c in cps}
        Q = {c: [] for c in cps}
        led = {c: np.full((K, len(keys)), np.nan) for c in lcps}
        ser = {kk: np.zeros(HZ.SERIES_T) for kk in HZ.SERIES_KEYS} if series else None
        codes = []
        for r in range(K):
            sd = roll_root * 100 + r
            own = PL.crn_policy(spec, k0, sd, 0, horizon, info)
            pa = own if cspec is None else HZ.build_candidate(cspec, own)
            pb = _opponent(proto, spec, k0, sd, horizon, info)
            codes.append(int(E.decide(pa, cfg, start[0], start[1], 0, 0)))
            out = HZ.rollout(cfg, start, pa, pb, sd, horizon, cps, critic=PL.critic_of(own, spec), series=series)
            for c, row in out["cp"].items():
                G[c].append(row["G"])
                Q[c].append(row["Q"])
                if c in led:
                    led[c][r] = [row.get(kk, np.nan) for kk in keys]
            if ser is not None:
                for kk in HZ.SERIES_KEYS:
                    ser[kk] += np.asarray(out["series"][kk]) / K
        GQ[name] = {"G": {str(c): v for c, v in G.items()}, "Q": {str(c): v for c, v in Q.items()}}
        L[name] = led
        SR[name] = {kk: v.tolist() for kk, v in ser.items()} if ser is not None else None
        first[name] = codes
    ledger = {name: {str(c): HZ._summ(L[name][c], L["policy"][c]) for c in lcps} for name in cands}
    data = {"spec": spec, "tag": tag, "family": fam, "index": idx, "protocol": proto, "horizon": horizon,
            "label": label, "k0": int(k0), "K": K, "keys": keys, "kind": B.FAMILIES[fam][0],
            "candidates": {n: (list(s) if s else None) for n, s in cands.items()}, "GQ": GQ, "ledger": ledger,
            "series": SR if series else None, "first_codes": first, "elapsed_s": time.perf_counter() - t0,
            "turns": K * len(cands) * horizon, "belief_start": "recorded" if info else "prior"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return key, data


# ---------------------------------------------------------------------------------------------- the design
def jobs(spec: str, tag: str, natural: Dict[str, list], families: Optional[Sequence[str]] = None,
         protocols: Optional[Sequence[str]] = None, k_nat: Optional[int] = None, k_con: Optional[int] = None,
         root: str = BOUT, grids: Optional[Dict[str, list]] = None, roll_root: int = C.BATTERY_ROLL_ROOT,
         validation: bool = True) -> list:
    """m45c.battery.jobs for an M5 policy: natural families at their horizon (the validation subset's "policy" jobs at
    300 turns), constructed cells at the prior, "policy" at 300 turns."""
    B = _B()
    cfg = C.arm_cfg("sp")
    families = tuple(families or B.FAMILIES)
    grids = grids or B.GRIDS
    k_nat = k_nat or B.K_NATURAL
    k_con = k_con or B.K_CONSTRUCTED
    out = []
    for fam in families:
        kind = B.FAMILIES[fam][0]
        protos = [p for p in B.FAMILIES[fam][3] if protocols is None or p in protocols]
        h = B.family_horizon(fam)
        if kind == "constructed":
            for i, cell in enumerate(grids[fam]):
                st = B.constructed_state(cfg, fam, cell)
                for p in protos:
                    hh = B.VALIDATION_H if (validation and p == "policy") else h
                    out.append((spec, tag, fam, i, st, None, p, hh, f"constructed {cell}", k_con, root, roll_root))
            continue
        states = natural.get(fam, [])
        val = set(B.validation_indices(fam, len(states))) if validation else set()
        for i, st in enumerate(states):
            for p in protos:
                hh = B.VALIDATION_H if (i in val and p == "policy") else h
                out.append((spec, tag, fam, i, st["start"], st.get("info"), p, hh,
                            f"{kind} (game seed {st['seed']}, turn {st['turn']}, seat {'AB'[st['seat']]})", k_nat, root,
                            roll_root))
    return out


def run_policy(spec: str, tag: str, workers: int, families: Optional[Sequence[str]] = None,
               n_natural: Optional[int] = None, n_early: Optional[int] = None, k_nat: Optional[int] = None,
               k_con: Optional[int] = None, root: str = BOUT, grids: Optional[Dict[str, list]] = None,
               protocols: Optional[Sequence[str]] = None, validation: bool = True) -> dict:
    """Collect the policy's natural states, then run (resumably) every job; writes the manifest under root/tag/."""
    B = _B()
    families = tuple(families or B.FAMILIES)
    n_natural = n_natural or B.N_NATURAL
    n_early = n_early or B.N_EARLY
    t0 = time.perf_counter()
    cjobs = [(spec, f, n_early if B.FAMILIES[f][0] == "early" else n_natural) for f in families
             if B.FAMILIES[f][0] != "constructed"]
    res = C.run_jobs(collect_job, cjobs, workers, f"M5 battery states {tag}")
    nat = {k[1]: v["states"] for k, v in res}
    games = {k[1]: v["games"] for k, v in res}
    t_col = time.perf_counter() - t0
    js = jobs(spec, tag, nat, families, protocols, k_nat, k_con, root, grids, validation=validation)
    t1 = time.perf_counter()
    out = dict(C.run_jobs(state_job, js, workers, f"M5 battery {tag}"))
    man = {"meta": C.meta(stage="battery", spec=spec, tag=tag, families=list(families)),
           "states": {f: {"n": len(v), "games": games[f],
                          "states": [{"seed": s["seed"], "turn": s["turn"], "seat": s["seat"], "k": int(s["start"][2]),
                                      "belief_recorded": s.get("info") is not None} for s in v]}
                      for f, v in nat.items()},
           "timing": {"collect_s": t_col, "rollouts_s": time.perf_counter() - t1, "workers": workers,
                      "turns": int(sum(v["turns"] for v in out.values())),
                      "job_seconds": float(sum(v["elapsed_s"] for v in out.values()))},
           "jobs": sorted(out)}
    C.jdump(os.path.join(root, tag, "manifest.json"), man)
    return man


__all__ = ["self_play", "collect_job", "state_job", "job_path", "jobs", "run_policy", "BOUT"]
