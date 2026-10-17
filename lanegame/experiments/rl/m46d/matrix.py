"""The paired plan x plan rollout job of workstreams A and B, and its stages.

One job = one start state x one policy (both seats; self-play pairing) x a set of (seat-A plan, seat-B plan) combos x K
paired rollouts x horizon H. Rollout r of every combo uses the chance stream default_rng(seed_r) and each seat's CRN
sampling uniforms default_rng([seed_r, 7, seat]) (`m45c.horizon.CRNNet`), so combos differ only by their plans.
Recorded per combo and rollout: G_h (A's discounted lead) at the checkpoints and the ledger (both sides) at the ledger
checkpoints (KEYS). Saved as one compressed npz per job (resumable).

Stages (pre-registration §3):
    A    grid_a cells x 6 policies (C1 env), A_FREEZER x A_BREAKER (4 x 5), K 32, H 50 (discovery seeds MAT_ROOT)
    AL   the long-horizon subset: L5 equal / plaus / x1.76 (held 12), freezer {policy, guard_sel, passive, push} x
         breaker {policy, commit, concede}, K 32, H 150
    Ab   the A core cells (equal / edge / plaus / x1.76 at levels 3, 5, 8, held 12) in the baseline env with the
         baseline policies (reference), 4 x 5, K 32, H 50
    B    grid_b cells x 6 policies (C1 env), freezer {policy, guard_sel} x the five breaker options, K 24, H 50
    N    natural early freeze states (favourable / other, 12 each per policy, C1), 4 x 5, K 16, H 50
    conf the confirmation sample (CONF_ROOT seeds) of the contrasts selected from A, B and N (see analysis)
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import cells as CL
from . import common as C
from . import plans as P

KEYS = ("earned", "farm", "farm_avail", "unc", "unc_away", "bounty", "plate", "xp", "deaths", "recalls", "rec_intr",
        "away", "held", "frozen", "d_total", "d_trade", "d_min", "trade_conn", "trade_dealt", "crash_hp", "plates",
        "n_safe", "n_trade")
CPS = (1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150)
LCPS = (10, 20, 50, 100)


def cps_for(H: int):
    c = tuple(x for x in CPS if x <= H)
    return c if H in c else c + (H,)


def lcps_for(H: int):
    return tuple(x for x in LCPS if x <= H)


def job_path(stage: str, arm: str, tag: str, idx) -> str:
    return os.path.join(C.CKPT, "matrix", stage, arm, tag, f"{idx}.npz")


def matrix_job(job: dict) -> tuple:
    """`job`: stage, arm, spec, tag, idx, start (list form), a (dict name -> plan spec), b (dict), combos (list of
    (a, b) names), K, H, root. Returns (key, {"path"})."""
    path = job["path"]
    key = f"{job['stage']}|{job['arm']}|{job['tag']}|{job['idx']}"
    if os.path.exists(path):
        return key, {"path": path, "cached": True}
    cfg = C.setup(job["arm"])
    from ..m46a import engine as EN
    from ..m45c import horizon as HZ
    net = C.net_of(job["spec"])
    start = C.state_from_list(job["start"])
    k0 = start[2]
    H, K = int(job["H"]), int(job["K"])
    cps, lcps = cps_for(H), lcps_for(H)
    seeds = [job["root"] * 1000 + r for r in range(K)]
    combos = [tuple(c) for c in job["combos"]]
    G = np.full((len(combos), K, len(cps)), np.nan)
    L = np.full((len(combos), K, len(lcps), len(KEYS), 2), np.nan, dtype=np.float32)
    forced = np.zeros((len(combos), K, 2), dtype=np.int32)
    t0 = time.perf_counter()
    for ci, (an, bn) in enumerate(combos):
        for r, sd in enumerate(seeds):
            pa = P.build(job["a"][an], HZ.CRNNet(net, k0, sd, 0, H))
            pb = P.build(job["b"][bn], HZ.CRNNet(net, k0, sd, 1, H))
            out = EN.rollout(cfg, start, pa, pb, sd, H, cps, lcps, None)
            for j, c in enumerate(cps):
                G[ci, r, j] = out[c]["G"]
            for j, c in enumerate(lcps):
                for kk, key_ in enumerate(KEYS):
                    L[ci, r, j, kk, 0] = out[c][f"{key_}_A"]
                    L[ci, r, j, kk, 1] = out[c][f"{key_}_B"]
            forced[ci, r] = (getattr(pa, "forced", 0), getattr(pb, "forced", 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"                 # per process: two runners may meet on one job
    np.savez_compressed(tmp, G=G, L=L, forced=forced, cps=np.asarray(cps), lcps=np.asarray(lcps),
                        combos=np.asarray([f"{a}|{b}" for a, b in combos]), keys=np.asarray(KEYS),
                        elapsed=np.asarray(time.perf_counter() - t0), label=np.asarray(job.get("label", "")))
    os.replace(tmp, path)
    return key, {"path": path, "elapsed_s": time.perf_counter() - t0}


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["combos"] = [str(x) for x in d["combos"]]
    d["keys"] = [str(x) for x in d["keys"]]
    return d


def mk(stage, arm, spec, tag, idx, start, a, b, combos, K, H, root, label="") -> dict:
    return {"stage": stage, "arm": arm, "spec": spec, "tag": tag, "idx": idx, "start": C.state_to_list(start),
            "a": a, "b": b, "combos": [list(c) for c in combos], "K": K, "H": H, "root": root, "label": label,
            "path": job_path(stage, arm, tag, idx)}


def full(a: dict, b: dict) -> List[Tuple[str, str]]:
    return [(x, y) for x in a for y in b]


A_CORE = ("equal", "edge", "plaus", "x1.76")


def jobs_a(stage: str = "A", root: int = C.MAT_ROOT, K: int = 32, H: int = 50) -> List[dict]:
    arm = "baseline" if stage == "Ab" else "C1"
    cfg = C.arm_cfg(arm)
    fz, br = P.A_FREEZER, P.A_BREAKER
    if stage == "A2":                                   # amendment 3: guard_walk on the A cells, A's seeds
        fz = {"guard_walk": P.FREEZER["guard_walk"]}
    if stage == "AL":
        H, br = 150, {k: P.A_BREAKER[k] for k in ("policy", "commit", "concede")}
    js = []
    for i, cell in enumerate(CL.grid_a()):
        if stage == "Ab" and not (cell["name"] in A_CORE and cell["held"] == 12):
            continue
        if stage == "AL" and not (cell["name"] in ("equal", "plaus", "x1.76") and cell["L"] == 5 and cell["held"] == 12):
            continue
        st = CL.cell_start(cfg, cell)
        for s in C.SEEDS:
            js.append(mk(stage, arm, C.general(arm, s), f"s{s}", i, st, fz, br, full(fz, br), K, H, root,
                         CL.cell_label(cell)))
    return js


def jobs_b(root: int = C.MAT_ROOT, K: int = 24, H: int = 50) -> List[dict]:
    cfg = C.arm_cfg("C1")
    fz = {"policy": None, "guard_sel": P.FREEZER["guard_sel"], "guard_walk": P.FREEZER["guard_walk"]}   # amendment 3
    names = ("policy", "commit", "contest_sel", "concede", "concede3", "reset")     # concede3: amendment 2
    js = []
    for i, cell in enumerate(CL.grid_b()):
        st = CL.cell_start(cfg, cell)
        br = {k: (P.BREAKER if cell["wave"] == "freeze" else P.BREAKER_CONTROL)[k] for k in names}
        for s in C.SEEDS:
            js.append(mk("B", "C1", C.general("C1", s), f"s{s}", i, st, fz, br, full(fz, br), K, H, root,
                         CL.cell_label(cell)))
    return js


def natural_states(workers: int, n_each: int = 12) -> dict:
    path = os.path.join(C.CKPT, "natural_freeze_states.json.gz")
    if os.path.exists(path):
        return C.load_gz(path)
    js = [("C1", C.general("C1", s), s, n_each, 400) for s in C.SEEDS]
    res = C.run_jobs(CL.natural_job, js, workers, "M4.6d natural freeze states")
    out = {f"{k[1]}": v for k, v in res}
    C.save_gz(path, out)
    return out


def jobs_n(states: dict, root: int = C.MAT_ROOT, K: int = 16, H: int = 50) -> List[dict]:
    js = []
    for s in C.SEEDS:
        spec = C.general("C1", s)
        for kind in ("favourable", "other"):
            for i, rec in enumerate(states[spec][kind]):
                st = C.state_from_list(rec["start"])
                fz = {**P.A_FREEZER, "guard_walk": P.FREEZER["guard_walk"]}          # amendment 3
                js.append(mk("N", "C1", spec, f"s{s}_{kind}", i, st, fz, P.A_BREAKER,
                             full(fz, P.A_BREAKER), K, H, root, f"natural {kind} L{rec['L']} held {rec['held']}"))
    return js


def run_jobs_sorted(js: List[dict], workers: int, label: str):
    js = sorted(js, key=lambda j: -len(j["combos"]) * j["K"] * j["H"])
    t0 = time.perf_counter()
    res = C.run_jobs(matrix_job, js, workers, label)
    print(f"  [{label}] {len(js)} jobs in {time.perf_counter() - t0:.0f} s", flush=True)
    return res


def run(workers: int, part: str = None) -> None:
    parts = (part or "A,AL,A2,B,N,Ab").split(",")
    for p in parts:
        if p == "A":
            run_jobs_sorted(jobs_a("A"), workers, "matrix A")
        elif p == "A2":
            run_jobs_sorted(jobs_a("A2"), workers, "matrix A2 (guard_walk, exploratory)")
        elif p == "AL":
            run_jobs_sorted(jobs_a("AL"), workers, "matrix AL (H 150)")
        elif p == "Ab":
            run_jobs_sorted(jobs_a("Ab"), workers, "matrix Ab (baseline)")
        elif p == "B":
            run_jobs_sorted(jobs_b(), workers, "matrix B")
        elif p == "N":
            run_jobs_sorted(jobs_n(natural_states(workers)), workers, "matrix N")
        else:
            raise SystemExit(f"unknown matrix part {p!r}")


def run_confirm(workers: int) -> None:
    from . import analysis as AN
    sel = AN.selected_contrasts()
    js = []
    for item in sel["jobs"]:
        js.append(mk("conf", item["arm"], item["spec"], item["tag"], item["idx"], C.state_from_list(item["start"]),
                     item["a"], item["b"], item["combos"], item["K"], item["H"], C.CONF_ROOT, item["label"]))
    run_jobs_sorted(js, workers, "confirmation")


__all__ = ["matrix_job", "load", "jobs_a", "jobs_b", "jobs_n", "natural_states", "run", "run_confirm", "KEYS",
           "cps_for", "lcps_for", "job_path", "mk", "full"]
