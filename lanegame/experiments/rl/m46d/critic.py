"""The critic diagnostic (pre-registration §5).

`calib`: from each start state, the policy against itself (the pairing its critic predicts; CRN sampling, chance
stream per rollout), K rollouts of H turns WITHOUT bootstrap. Recorded: V_A(s), V_B(s) (critic, resources, both
seats), seat A's discounted lead G_h at the checkpoints, and the critic's tail gamma^H V_A(s_H). The bias of the critic
is V_A(s) - E[G_H] (no-critic return), with the measured change G_300 -> G_500 on a subset as the tail check.

`rank`: at selected decision states, one seat (the "decider") plays a fixed one-turn action (or its policy) and then
the policy, the other seat the policy; K rollouts of H 150 without bootstrap. Recorded per candidate: the decider's
long return (G or -G), the critic's one-step look-ahead r_0 + gamma V_decider(s_1) (the decider's reward and critic),
and the decider's policy probability of the action. Compared: advantage signs and ranks (rollout vs critic) and the
actor's probabilities.

States:
    freeze   the A freeze cells equal / edge / plaus / x1.76 at levels 3, 5, 8 (held 12)
    lane     the B neutral / push cells at full HP (equal / edge / x1.76, levels 4 and 8)
    gap      the M4.6a gap cells x1.5 and x1.76 (baseline-curve gold)
    natural  natural states of the policy's own self-play: invested lead >= 4 (8) and even, |lead| < 2 (8),
             turns 30-300
    recallw  (rank only) constructed Recall windows: neutral lane, B at its -1 with HP 0.5 or 0.35, A edge or x1.76,
             levels 5 and 8
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np

from ....champions import N_CHANCE
from ....config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, W_HOLD, W_SHOVE, W_THIN, champ_act)
from . import cells as CL
from . import common as C

CAL_CPS = (10, 20, 50, 100, 150, 200, 300, 400, 500)
K_CAL, H_CAL, H_LONG = 32, 300, 500
K_RANK, H_RANK = 16, 150
RANK_CPS = (1, 10, 20, 50, 100, 150)

A_RANK = {"LH-Hold": champ_act(W_HOLD, B_HOLD), "TN-Hold": champ_act(W_THIN, B_HOLD),
          "SH-Hold": champ_act(W_SHOVE, B_HOLD), "TN-Trade": champ_act(W_THIN, B_TRADE),
          "LH-Trade": champ_act(W_HOLD, B_TRADE), "LH-Safe": champ_act(W_HOLD, B_SAFE)}
B_RANK = {"SH-Hold": champ_act(W_SHOVE, B_HOLD), "LH-Hold": champ_act(W_HOLD, B_HOLD),
          "LH-Safe": champ_act(W_HOLD, B_SAFE), "SH-Trade": champ_act(W_SHOVE, B_TRADE),
          "Recall": champ_act(W_HOLD, B_RECALL)}


def save_gz_pid(path: str, data) -> None:
    """`save_gz` with a per-process temporary file (two runners may meet on one job; results are deterministic)."""
    import gzip
    import json
    from ..m46a.common import _default
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f, default=_default)
    os.replace(tmp, path)


def critic_v(net, cfg, lane, m4, k, s) -> float:
    from ..m45c.horizon import critic_value
    return critic_value(net, cfg, lane, m4, k, s)


# ---------------------------------------------------------------------------------------------- calibration
def calib_job(job: dict) -> tuple:
    path = job["path"]
    key = f"{job['arm']}|{job['tag']}|{job['family']}|{job['idx']}"
    if os.path.exists(path):
        return key, {"path": path, "cached": True}
    cfg = C.setup(job["arm"])
    from ..m46a.engine import rollout
    from ..m45c import horizon as HZ
    net = C.net_of(job["spec"])
    start = C.state_from_list(job["start"])
    lane, m4, k0 = start
    H, K = int(job["H"]), int(job["K"])
    cps = tuple(c for c in CAL_CPS if c <= H)
    G = np.full((K, len(cps)), np.nan)
    tail = np.full(K, np.nan)
    t0 = time.perf_counter()
    for r in range(K):
        sd = job["root"] * 1000 + r
        pa, pb = HZ.CRNNet(net, k0, sd, 0, H), HZ.CRNNet(net, k0, sd, 1, H)
        out = rollout(cfg, start, pa, pb, sd, H, cps, (), net)
        for j, c in enumerate(cps):
            G[r, j] = out[c]["G"]
        tail[r] = C.GAMMA ** H * out[H]["V"]
    rec = {"arm": job["arm"], "spec": job["spec"], "family": job["family"], "idx": job["idx"],
           "label": job.get("label", ""), "cps": list(cps), "G": G.tolist(), "tail": tail.tolist(),
           "VA": critic_v(net, cfg, lane, m4, k0, 0), "VB": critic_v(net, cfg, lane, m4, k0, 1),
           "elapsed_s": time.perf_counter() - t0}
    save_gz_pid(path, rec)
    return key, {"path": path}


def _cell_states(cfg) -> List[tuple]:
    from ..m46a.states import zoning_gap_state
    from .specialists import gap_cells
    out = []
    for i, cell in enumerate(CL.grid_a()):
        if cell["name"] in ("equal", "edge", "plaus", "x1.76") and cell["held"] == 12:
            out.append(("freeze", i, CL.cell_start(cfg, cell), CL.cell_label(cell)))
    for i, cell in enumerate(CL.grid_b()):
        if cell["wave"] != "freeze" and cell["hp_b"] == 1.0:
            out.append(("lane", i, CL.cell_start(cfg, cell), CL.cell_label(cell)))
    base = C.arm_cfg("baseline")
    for i, c in enumerate(gap_cells()):
        if c[2] in ("x1.5", "x1.76"):
            out.append(("gap", i, zoning_gap_state(base, *c), f"gap {c}"))
    return out


def natural_job(args) -> tuple:
    """(arm, spec, s, n): natural lead (invested lead >= 4) and even (|lead| < 2) states, turns 30-300, seat B
    states mirrored to A; at most one of each kind per game."""
    arm, spec, s, n = args
    cfg = C.setup(arm)
    from ...m45b import engine as E
    from ...m45b import states as S
    from ..m45c import horizon as HZ
    net = C.net_of(spec)
    sc = cfg.gold_scale

    def lead(m4, s_):
        return (m4.side(s_).items - m4.side(1 - s_).items) / sc

    def p_lead(cfg_, lane, m4, s_):
        return m4.a.absent == 0 and m4.b.absent == 0 and lead(m4, s_) >= 4

    def p_even(cfg_, lane, m4, s_):
        return m4.a.absent == 0 and m4.b.absent == 0 and abs(lead(m4, s_)) < 2

    out = {"lead": [], "even": []}
    for g in range(200):
        sd = C.CRIT_ROOT * 1000 + 500 + 100 * s + g
        for kind, pred in (("lead", p_lead), ("even", p_even)):
            if len(out[kind]) >= n:
                continue
            got = E.collect(cfg, S.s0(cfg), HZ.CRNNet(net, 0, sd, 0, 300), HZ.CRNNet(net, 0, sd, 1, 300), [sd], 300,
                            pred, per_game=1, min_turn=30, rng_seed=sd + 1)
            out[kind] += [{"start": C.state_to_list(x["start"]), "turn": x["turn"], "seed": sd} for x in got]
        if all(len(v) >= n for v in out.values()):
            break
    return (arm, spec), {k: v[:n] for k, v in out.items()}


def calib_jobs(workers: int) -> List[dict]:
    nat_path = os.path.join(C.CKPT, "critic_natural_states.json.gz")
    if os.path.exists(nat_path):
        nat = C.load_gz(nat_path)
    else:
        js = [("C1", C.general("C1", s), s, 8) for s in C.SEEDS]
        nat = {f"{k[0]}|{k[1]}": v for k, v in C.run_jobs(natural_job, js, workers, "critic natural states")}
        C.save_gz(nat_path, nat)
    js = []
    for arm in ("C1", "baseline"):
        cfg = C.arm_cfg(arm)
        states = _cell_states(cfg)
        for s in C.SEEDS:
            spec = C.general(arm, s)
            items = list(states) if arm == "C1" else [x for x in states if x[0] in ("gap", "freeze")]
            for kind in (("lead", "even") if arm == "C1" else ()):
                for i, rec in enumerate(nat[f"{arm}|{spec}"][kind]):
                    items.append((f"natural_{kind}", i, C.state_from_list(rec["start"]), f"natural {kind} t{rec['turn']}"))
            for fam, i, st, lab in items:
                H = H_LONG if fam.startswith("natural") else H_CAL
                js.append({"arm": arm, "spec": spec, "tag": f"s{s}", "family": fam, "idx": i,
                           "start": C.state_to_list(st), "K": K_CAL, "H": H, "root": C.CRIT_ROOT, "label": lab,
                           "path": os.path.join(C.CKPT, "critic", "calib", arm, f"s{s}", f"{fam}_{i}.json.gz")})
    return js


# ---------------------------------------------------------------------------------------------- ranking
def rank_job(job: dict) -> tuple:
    path = job["path"]
    key = f"{job['arm']}|{job['tag']}|{job['family']}|{job['idx']}|{job['decider']}"
    if os.path.exists(path):
        return key, {"path": path, "cached": True}
    cfg = C.setup(job["arm"])
    from ....champions import champion_legal_actions
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    from ..m45c import horizon as HZ
    net = C.net_of(job["spec"])
    start = C.state_from_list(job["start"])
    lane0, m40, k0 = start
    dec = int(job["decider"])
    sign = 1.0 if dec == 0 else -1.0
    H, K = int(job["H"]), int(job["K"])
    cps = tuple(c for c in RANK_CPS if c <= H)
    legal0 = champion_legal_actions(lane0, m40, dec, cfg)
    probe = HZ.CRNNet(net, k0, 0, dec, H)
    lg, pr = probe.probs(cfg, lane0, m40, dec, 0)
    prob = {int(c): float(q) for c, q in zip(lg, pr)}
    ctx = context(cfg)
    ls = E.lane_cache(cfg)
    out = {"cands": {}, "probs": {}, "legal": list(map(int, legal0))}
    t0 = time.perf_counter()
    for name, code in [("policy", None)] + list(job["cands"].items()):
        if code is not None and code not in legal0:
            continue
        Gs = np.full((K, len(cps)), np.nan)
        crit = np.full(K, np.nan)
        for r in range(K):
            sd = job["root"] * 1000 + r
            pols = [HZ.CRNNet(net, k0, sd, 0, H), HZ.CRNNet(net, k0, sd, 1, H)]
            lane, m4 = lane0, m40
            rng = np.random.default_rng(sd)
            G, j = 0.0, 0
            for i in range(H):
                u = rng.random(N_CHANCE)
                acts = []
                for s in (0, 1):
                    if i == 0 and s == dec and code is not None:
                        acts.append(int(code))
                    else:
                        acts.append(int(E.decide(pols[s], cfg, lane, m4, s, i)))
                t = resolve_turn_m4(lane, m4, k0 + i, acts[0], acts[1], cfg, lane_step=ls, u=u, ctx=ctx)
                G += C.GAMMA ** i * t.reward
                lane, m4 = t.lane.state, t.champ
                if i == 0:
                    crit[r] = sign * t.reward + C.GAMMA * critic_v(net, cfg, lane, m4, k0 + 1, dec)
                if j < len(cps) and i + 1 == cps[j]:
                    Gs[r, j] = sign * G
                    j += 1
        out["cands"][name] = {"G": Gs.tolist(), "crit": crit.tolist()}
        out["probs"][name] = prob.get(int(code), 0.0) if code is not None else 1.0
    out.update({"arm": job["arm"], "spec": job["spec"], "family": job["family"], "idx": job["idx"],
                "decider": dec, "label": job.get("label", ""), "cps": list(cps),
                "V0": critic_v(net, cfg, lane0, m40, k0, dec), "elapsed_s": time.perf_counter() - t0})
    save_gz_pid(path, out)
    return key, {"path": path}


def recall_window(cfg, cond: str, L: int, hp_b: float):
    from ...m45b import states as S
    lane, m4, k = CL.build(cfg, "neutral", cond, L, hp_b)
    return lane, m4._replace(b=m4.b._replace(pos=-1)), k


def rank_jobs() -> List[dict]:
    js = []
    cfg = C.arm_cfg("C1")
    items = []
    for i, cell in enumerate(CL.grid_a()):
        if cell["name"] in ("equal", "edge", "plaus", "x1.76") and cell["held"] == 12:
            st = CL.cell_start(cfg, cell)
            items.append(("freeze", i, st, 0, A_RANK, CL.cell_label(cell)))
            items.append(("freeze", i, st, 1, B_RANK, CL.cell_label(cell)))
    j = 0
    for cond in ("edge", "x1.76"):
        for L in (5, 8):
            for hpb in (0.5, 0.35):
                items.append(("recallw", j, recall_window(cfg, cond, L, hpb), 0, A_RANK,
                              f"recall window L{L} {cond} hpB{hpb}"))
                items.append(("recallw", j, recall_window(cfg, cond, L, hpb), 1, B_RANK,
                              f"recall window L{L} {cond} hpB{hpb}"))
                j += 1
    for s in C.SEEDS:
        spec = C.general("C1", s)
        for fam, i, st, dec, cands, lab in items:
            js.append({"arm": "C1", "spec": spec, "tag": f"s{s}", "family": fam, "idx": i, "decider": dec,
                       "start": C.state_to_list(st), "cands": cands, "K": K_RANK, "H": H_RANK,
                       "root": C.CRIT_ROOT + 1, "label": lab,
                       "path": os.path.join(C.CKPT, "critic", "rank", f"s{s}", f"{fam}_{i}_d{dec}.json.gz")})
    return js


def run(workers: int, part: str = None) -> None:
    parts = (part or "calib,rank").split(",")
    for p in parts:
        t0 = time.perf_counter()
        if p == "calib":
            js = sorted(calib_jobs(workers), key=lambda j: -j["H"])
            C.run_jobs(calib_job, js, workers, "critic calibration")
        elif p == "rank":
            js = rank_jobs()
            C.run_jobs(rank_job, js, workers, "critic ranking")
        else:
            raise SystemExit(f"unknown critic part {p!r}")
        print(f"  [{p}] {len(js)} jobs in {time.perf_counter() - t0:.0f} s", flush=True)


__all__ = ["calib_job", "calib_jobs", "rank_job", "rank_jobs", "run", "A_RANK", "B_RANK"]
