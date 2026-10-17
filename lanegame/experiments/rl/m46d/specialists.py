"""Workstream C: the specialist gap (pre-registration §4).

C1 `verify`: seat A (the stronger side) is the general policy itself ("own") or an M4.6a gap specialist (g0, g2, g4),
seat B is the general policy, on the M4.6a gap cells (baseline-curve gold, as in M4.6a/M4.6c) and the plausible
cells; both environments (the specialists are baseline-trained: C1 is CROSS-ENVIRONMENT for them). The M4.6a paired
engine (`m46a.engine.paired_job`) with "net" candidates; K 16, H 50. Advantage = specialist - own.

C2 `decomp` (baseline environment, pairs (gap_t, ext/baseline:t), t = 0, 2, 4): one job per (pair, cell) with the
candidates
    gen        the general policy throughout
    spec       the specialist throughout
    S<k>       the specialist for turns i < k, then the general        (k in S_KS)
    U<k>       the general, except the specialist's choice at turn k   (k in U_KS)
Every candidate's seat-A networks sample with the SAME per-turn uniform (seed, seat 0); seat B is the general with its
own CRN uniform; chance streams are shared. Recorded per rollout: G at the checkpoints for every candidate, and along
the "spec" and "gen" paths at each k of S_KS | U_KS the start-of-turn state class (no hindsight: only the
start-of-turn state), the path policy's action and the action the other network would have sampled with the same
uniform, plus the weak side's Recall probability (descriptive only).
State classes (seat A's view, start of turn): away (B absent); recall_window (B present, Recall legal for B, B HP <=
60%, A can catch B without a dive); kill_window (B HP <= 35%, catchable); trade_window (catchable otherwise);
out_of_reach (B present, not catchable).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np

from ....champions import N_CHANCE
from ....config import (B_RECALL, B_TRADE, CHAMP_ABSENT, CHAMP_WARD, W_HOLD, champ_act, champ_behavior,
                        champ_objective, champ_wave)
from . import cells as CL
from . import common as C

K_VER, H_VER = 16, 50
S_KS = (1, 2, 3, 4, 6, 8, 10, 14, 20, 30)
U_KS = (0, 1, 2, 3, 4, 6, 8, 10, 14, 20)
CPS = (1, 2, 3, 5, 10, 20, 30, 40, 50, 60)
H_DEC, K_DEC = 60, 32
CLASSES = ("away", "recall_window", "kill_window", "trade_window", "out_of_reach")


def gap_cells():
    from ..m46a.e4 import GAP_CELLS
    return list(GAP_CELLS)


def plaus_cells() -> List[dict]:
    """Plausible early-advantage cells: neutral and A-pushing at levels 4 and 8, the edge condition, B at 60% HP; and
    the edge condition at full HP, and equal power with B at 60% HP (components)."""
    out = []
    for w in ("neutral", "push"):
        for L in (4, 8):
            for cond, hpb, name in (("edge", 0.6, "plaus"), ("edge", 1.0, "edge"), ("equal", 0.6, "hp")):
                out.append({"wave": w, "cond": cond, "L": L, "hp_b": hpb, "held": 12, "name": name})
    for L in (5, 8):
        out.append({"wave": "freeze", "cond": "edge", "L": L, "hp_b": 0.6, "held": 12, "name": "plaus"})
    return out


# ---------------------------------------------------------------------------------------------- C1 verify
def verify_jobs() -> List[dict]:
    from ..m46a import engine as EN
    from ..m46a.states import zoning_gap_state
    from ..m46c.evaluate import EVAL  # noqa: F401  (path root only)
    base = C.arm_cfg("baseline")
    gap = [("gap", i, zoning_gap_state(base, *c), f"gap {c}") for i, c in enumerate(gap_cells())]
    cands = {"own": ("policy",)}
    for t in C.SPEC_TARGETS:
        cands[f"g{t}"] = ("net", C.specialist("gap", t))
    js = []
    for arm in ("baseline", "C1"):
        cfg = C.arm_cfg(arm)
        pl = [("plaus", i, CL.cell_start(cfg, c), CL.cell_label(c)) for i, c in enumerate(plaus_cells())]
        for s in C.SEEDS:
            spec = C.general(arm, s)
            for fam, i, st, lab in gap + pl:
                js.append({"spec": spec, "family": f"verify_{fam}", "idx": i, "start": st, "proto": "policy",
                           "horizon": H_VER, "roll": EN.roll_seeds(C.SPEC_ROOT, K_VER), "cands": cands,
                           "critic": False, "tag": f"{arm}_s{s}", "arm": arm, "label": lab,
                           "path": os.path.join(C.CKPT, "spec", "verify", arm, f"s{s}", f"{fam}_{i}.json.gz")})
    return js


def verify_job(job: dict) -> tuple:
    from ..m46a import engine as EN
    key = f"{job['tag']}|{job['family']}|{job['idx']}"
    if os.path.exists(job["path"]):
        return key, {"path": job["path"], "cached": True}
    C.setup(job["arm"])
    C.install_nets([job["spec"]] + [c[1] for c in job["cands"].values() if c[0] == "net"])
    _, rec = EN.paired_job(job)
    return key, {"path": job["path"], "elapsed_s": rec["elapsed_s"]}


# ---------------------------------------------------------------------------------------------- C2 decomposition
class TwoNet:
    """Seat-A policy mixing two networks on one CRN uniform stream: `pick(i)` says which network decides turn i."""

    def __init__(self, gen, spec, k0: int, seed: int, horizon: int, pick):
        from ..m45c import horizon as HZ
        self.g = HZ.CRNNet(gen, k0, seed, 0, horizon)
        self.s = HZ.CRNNet(spec, k0, seed, 0, horizon)
        self.s.u = self.g.u                                  # one uniform stream for both networks
        self.pick = pick

    def decide(self, cfg, lane, m4, s: int, i: int):
        return (self.s if self.pick(i) else self.g).decide(cfg, lane, m4, s, i)

    def inner_policies(self):
        return (self.g, self.s)                          # M5 opt-in hook (m45b.engine.observe_turn)


def state_class(cfg, lane, m4) -> str:
    from .plans import catchable, hp
    from ....champions import champion_legal_actions
    b = m4.b
    if b.absent:
        return "away"
    if not catchable(cfg, m4, 0):
        return "out_of_reach"
    hb = hp(cfg, m4, 1)
    if hb <= 0.6 and champ_act(W_HOLD, B_RECALL) in champion_legal_actions(lane, m4, 1, cfg):
        return "recall_window"
    if hb <= 0.35:
        return "kill_window"
    return "trade_window"


def act_class(code: int) -> str:
    if code == CHAMP_ABSENT:
        return "absent"
    if code == CHAMP_WARD:
        return "ward"
    b = champ_behavior(code)
    beh = ("Hold", "Yield", "Trade", "Safe", "Recall")[b]
    if b == B_RECALL:
        return "Recall"
    wave = "SH" if champ_wave(code) == 1 else "LH/TN/SL"
    obj = "+P" if champ_objective(code) == 1 else ""
    return f"{beh}|{wave}{obj}"


def decomp_cands() -> Dict[str, object]:
    c = {"gen": lambda i: False, "spec": lambda i: True}
    for k in S_KS:
        c[f"S{k}"] = (lambda kk: (lambda i: i < kk))(k)
    for k in U_KS:
        c[f"U{k}"] = (lambda kk: (lambda i: i == kk))(k)
    return c


def decomp_job(job: dict) -> tuple:
    """`job`: arm, gen, spec, tag, idx, start (list), K, H, root, path, label."""
    path = job["path"]
    key = f"{job['tag']}|{job['idx']}"
    if os.path.exists(path):
        return key, {"path": path, "cached": True}
    cfg = C.setup(job["arm"])
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    from ..m45c import horizon as HZ
    from ..m46a.engine import XLedger
    gen, spc = C.net_of(job["gen"]), C.net_of(job["spec"])
    start = C.state_from_list(job["start"])
    k0 = start[2]
    H, K = int(job["H"]), int(job["K"])
    cands = decomp_cands()
    names = list(cands)
    ks = sorted(set(S_KS) | set(U_KS))
    cps = tuple(c for c in CPS if c <= H)
    ctx = context(cfg)
    ls = E.lane_cache(cfg)
    G = np.full((len(names), K, len(cps)), np.nan)
    LK = ("earned", "farm", "deaths", "away", "rec_intr", "recalls", "trade_conn", "unc", "unc_away", "plate",
          "bounty", "d_total")
    Lg = np.full((len(names), K, len(LK), 2), np.nan, dtype=np.float32)
    path_info = {"spec": [], "gen": []}
    t0 = time.perf_counter()
    for r in range(K):
        sd = job["root"] * 1000 + r
        for ci, nm in enumerate(names):
            pa = TwoNet(gen, spc, k0, sd, H, cands[nm])
            pb = HZ.CRNNet(gen, k0, sd, 1, H)
            lane, m4, _ = start
            rng = np.random.default_rng(sd)
            led = XLedger(cfg, start)
            Gc, j = 0.0, 0
            rec = [] if nm in ("spec", "gen") else None
            for i in range(H):
                u = rng.random(N_CHANCE)
                if rec is not None and i in ks:
                    other = pa.g if nm == "spec" else pa.s
                    mine = pa.s if nm == "spec" else pa.g
                    legal_b, pbp = pb.probs(cfg, lane, m4, 1, i)
                    p_rec = float(sum(q for c, q in zip(legal_b, pbp) if c != CHAMP_ABSENT and
                                      champ_behavior(int(c)) == B_RECALL)) if m4.b.absent == 0 else 0.0
                    rec.append({"i": i, "cls": state_class(cfg, lane, m4),
                                "a": act_class(int(E.decide(mine, cfg, lane, m4, 0, i))),
                                "alt": act_class(int(E.decide(other, cfg, lane, m4, 0, i))),
                                "pB_recall": p_rec})
                a = E.decide(pa, cfg, lane, m4, 0, i)
                b = E.decide(pb, cfg, lane, m4, 1, i)
                t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=ls, u=u, ctx=ctx)
                led.add(t, lane)
                Gc += C.GAMMA ** i * t.reward
                lane, m4 = t.lane.state, t.champ
                if j < len(cps) and i + 1 == cps[j]:
                    G[ci, r, j] = Gc
                    j += 1
            for kk, key_ in enumerate(LK):
                Lg[ci, r, kk] = led.v[key_]
            if rec is not None:
                path_info[nm].append(rec)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    C.save_gz(path.replace(".npz", ".paths.json.gz"), path_info)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, G=G, L=Lg, names=np.asarray(names), cps=np.asarray(cps), lkeys=np.asarray(LK),
                        elapsed=np.asarray(time.perf_counter() - t0))
    os.replace(tmp, path)
    return key, {"path": path, "elapsed_s": time.perf_counter() - t0}


def decomp_jobs() -> List[dict]:
    from ..m46a.states import zoning_gap_state
    base = C.arm_cfg("baseline")
    cells = [(f"gap {c}", zoning_gap_state(base, *c), f"gap_{i}") for i, c in enumerate(gap_cells())
             if c[2] in ("x1.5", "x1.76")]
    cells += [(CL.cell_label(c), CL.cell_start(base, c), f"plaus_{i}") for i, c in enumerate(plaus_cells())
              if c["name"] == "plaus" and c["wave"] != "freeze"]
    js = []
    for t in C.SPEC_TARGETS:
        for lab, st, idx in cells:
            js.append({"arm": "baseline", "gen": C.general("baseline", t), "spec": C.specialist("gap", t),
                       "tag": f"pair{t}", "idx": idx, "start": C.state_to_list(st), "K": K_DEC, "H": H_DEC,
                       "root": C.SPEC_ROOT + 1, "label": lab,
                       "path": os.path.join(C.CKPT, "spec", "decomp", f"pair{t}", f"{idx}.npz")})
    return js


def run(workers: int, part: str = None) -> None:
    parts = (part or "verify,decomp").split(",")
    for p in parts:
        t0 = time.perf_counter()
        if p == "verify":
            js = verify_jobs()
            C.run_jobs(verify_job, js, workers, "C1 verify")
        elif p == "decomp":
            js = sorted(decomp_jobs(), key=lambda j: j["idx"])
            C.run_jobs(decomp_job, js, workers, "C2 decomposition")
        else:
            raise SystemExit(f"unknown specialists part {p!r}")
        print(f"  [{p}] {len(js)} jobs in {time.perf_counter() - t0:.0f} s", flush=True)


__all__ = ["verify_jobs", "verify_job", "decomp_jobs", "decomp_job", "run", "plaus_cells", "gap_cells", "CLASSES",
           "state_class", "act_class", "S_KS", "U_KS"]
