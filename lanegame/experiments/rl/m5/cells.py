"""M5c's M4 limitation families (design §7.2, the closeout's S5: every §4 class re-measured with its own instrument).

`paired_job` is `m46a.engine.paired_job` (same job dictionary, record format and rollout engine) with every policy
resolved by `players.crn_policy`, so a candidate, the evaluated policy or the opponent may be an om-arm network. A
job's optional "info" (seat records of a recorded om game) restores the beliefs; constructed cells and the common
start's natural states have none (prior, no previous action; players module docstring). For 90-input policies it
reproduces the M4.6 records exactly given the same jobs (tested in readiness).

Families (builders import the historical cells; seat A is the focal seat; G_h without a critic bootstrap unless noted):
    weak       M4.6c weaker-side battery: lead cells B+d (weak focal: policy / hold / safe / trade / reset) and A+d
               (strong focal: policy / threaten / hold), the arm's own final in both seats, K 32, H 50
    premium    M4.6c lead premium J1: the arm's final in both seats on every lead cell (stage x situation x cell),
               candidate "self", G20 / G50, K PREMIUM_K
    conv       M4.6f B2 strength use: M4.6c lead cells (early, mid) and M4.6d lane cells, K 16 x H 150
    freeze     M4.6f B3: M4.6d freeze cells (held 12), freezer and (mirrored) breaker in seat A, K 32 x H 50
    recall     M4.6f B4: Recall windows, strong side and (mirrored) weak side in seat A, K 32 x H 100
    natural    M4.6f B5: natural leader / trailer states (K 16 x H 150) and natural freezes, freezer and breaker (K 16 x
               H 50), collected from the COMMON START ext/baseline:s's self-play (`m46c.evaluate.states_job` with M5 seeds)
    gaps       M4.6a E4 large gaps (generalization diagnostic): the zoning gap cells x1.5 / x1.76 / x2.2 and the
               equal cells, K 16 x H 150, the M4.6a gap specialist of the seed as a third candidate
Cross-arm families (conv, freeze, recall, natural, gaps) put both arms' finals in seat A as paired candidates
{"om", "sp"} against each arm's final in seat B ("net:" protocol) on the same rollout seeds: om - sp is the M5
contrast; each candidate against its own arm is the within-arm reading. Per-arm families (weak, premium) are run once
per final with the arm's own policy as opponent.
No family uses a critic bootstrap (rollout evidence only; `critic` is supported for 90- and 116-input policies).
Seeds: rollouts (CELL_ROOT * 10 + family offset) * 100 + r (`m46a.engine.roll_seeds`), the same K seeds for every
candidate and opponent of a cell; natural states (CELL_ROOT + 50) * 1000 + 100 s + role offset + game.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import common as C
from . import players as PL

COUT = os.path.join(C.CKPT, "cells")
PREMIUM_K = 64                  # M4.6c used 128; 64 halves the cost (proposal; the premium family is per arm)
FAMILY_OFFSET = {"weak": 0, "premium": 1, "conv": 2, "freeze": 3, "recall": 4, "natural": 5, "gaps": 6}
STATE_ROOT = C.CELL_ROOT + 50   # natural states of the common start (states_job seeds STATE_ROOT * 1000 + ...)


# ---------------------------------------------------------------------------------------------- the paired job
def _seat_a(cspec, spec: str, cfg, start, k0: int, seed: int, horizon: int, info):
    from ..m46a import policies as P
    if cspec[0] == "variant":
        st = P.VARIANTS[cspec[1]](cfg, start)
        pol, _ = _seat_a(cspec[2], spec, cfg, st, k0, seed, horizon, info)
        return pol, st
    if cspec[0] == "net":
        return PL.crn_policy(cspec[1], k0, seed, 0, horizon, info), start
    own = PL.crn_policy(spec, k0, seed, 0, horizon, info)
    return P.build(tuple(cspec), own), start


def _opponent(proto: str, spec: str, k0: int, seed: int, horizon: int, info):
    from ...m45b.strategies import get
    if proto == "policy":
        return PL.crn_policy(spec, k0, seed, 1, horizon, info)
    if proto == "modal":
        return PL.crn_policy(spec, k0, seed, 1, horizon, info, greedy=True)
    if proto.startswith("net:"):
        return PL.crn_policy(proto[4:], k0, seed, 1, horizon, info)
    return get(proto)


def paired_job(job: dict) -> tuple:
    """m46a.engine.paired_job for M5 policies (module docstring). `job`: spec, family, idx, start, [info], proto,
    horizon, roll, cands, [critic], [tag], [label], [meta], path; optional "env" = "C1" for a 90-input readiness
    reproduction of a C1 record (never used by the campaign)."""
    from ..m46a import engine as EN
    from ...m45b import engine as E
    key = f"{job['family']}|{job['idx']}|{job['proto']}|{job['horizon']}|{job.get('tag', '')}"
    path = job.get("path")
    if path and os.path.exists(path):
        return key, C.load_gz(path)
    if job.get("env") == "C1":
        from ..m46c import common as C46
        cfg = C46.setup("C1")
    else:
        C.setup("sp")
        cfg = C.arm_cfg("sp")
    start, info = job["start"], job.get("info")
    k0 = start[2]
    H = int(job["horizon"])
    cps, lcps = EN.checkpoints(H), EN.ledger_checkpoints(H)
    keys = [f"{k}_{x}" for k in EN.LEDGER_KEYS for x in ("A", "B")]
    cands = job["cands"]
    use_critic = bool(job.get("critic", False))
    t0 = time.perf_counter()
    G, Q, L, forced, first = {}, {}, {}, {}, {}
    for name, cspec in cands.items():
        g = {c: [] for c in cps}
        q = {c: [] for c in cps}
        led = {c: np.full((len(job["roll"]), len(keys)), np.nan) for c in lcps}
        f_count, codes = [], []
        for r, sd in enumerate(job["roll"]):
            pa, st = _seat_a(cspec, job["spec"], cfg, start, k0, sd, H, info)
            pb = _opponent(job["proto"], job["spec"], k0, sd, H, info)
            codes.append(int(E.decide(pa, cfg, st[0], st[1], 0, 0)))
            if hasattr(pa, "forced"):
                pa.forced = 0
            critic = _critic_for(pa, cspec[1] if cspec[0] == "net" else job["spec"]) if use_critic else None
            out = EN.rollout(cfg, st, pa, pb, sd, H, cps, lcps, critic)
            f_count.append(int(getattr(pa, "forced", 0)))
            for c, row in out.items():
                g[c].append(row["G"])
                if "Q" in row:
                    q[c].append(row["Q"])
                if c in led:
                    led[c][r] = [row.get(kk, np.nan) for kk in keys]
        G[name] = {str(c): v for c, v in g.items()}
        Q[name] = {str(c): v for c, v in q.items()} if use_critic else None
        L[name] = led
        forced[name] = f_count
        first[name] = codes
    base = list(cands)[0]
    ledger = {}
    for name in cands:
        ledger[name] = {}
        for c in lcps:
            d = L[name][c] - L[base][c]
            ledger[name][str(c)] = {"mean": np.nanmean(L[name][c], 0).tolist(), "diff": np.nanmean(d, 0).tolist(),
                                    "diff_sd": (np.nanstd(d, 0, ddof=1) if len(job["roll"]) > 1
                                                else np.zeros(len(keys))).tolist()}
    rec = {"family": job["family"], "idx": job["idx"], "proto": job["proto"], "horizon": H, "spec": job["spec"],
           "label": job.get("label", ""), "tag": job.get("tag", ""), "meta": job.get("meta"), "K": len(job["roll"]),
           "k0": int(k0), "keys": keys, "cands": {n: list(map(str, s)) for n, s in cands.items()}, "baseline": base,
           "G": G, "Q": Q, "ledger": ledger, "forced": forced, "first_codes": first,
           "belief_start": "recorded" if info else "prior", "elapsed_s": time.perf_counter() - t0,
           "turns": len(job["roll"]) * len(cands) * H}
    if path:
        C.save_gz(path, rec)
    return key, rec


def _critic_for(pa, spec: str):
    """The bootstrap critic of seat A: the om policy inside a (plan) policy, else the network of `spec`."""
    stack, seen = [pa], set()
    while stack:
        p = stack.pop()
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        if isinstance(p, PL.OMPolicy):
            return p
        stack.extend(getattr(p, "inner_policies", lambda: ())())
    return PL.net_of(spec)


# ---------------------------------------------------------------------------------------------- family builders
def _roll(family: str, K: int, idx: Optional[int] = None):
    from ..m46a import engine as EN
    return EN.roll_seeds(C.CELL_ROOT * 10 + FAMILY_OFFSET[family], K, idx)


def _job(family, tag, idx, start, spec, cands, K, H, proto="policy", label="", meta=None, critic=False, info=None,
         root=COUT):
    return {"spec": spec, "family": family, "idx": idx, "start": start, "info": info, "proto": proto, "horizon": H,
            "roll": _roll(family, K), "cands": cands, "critic": critic, "tag": tag, "label": label, "meta": meta,
            "path": os.path.join(root, family, tag, f"{idx}.json.gz")}


def weak_jobs(finals: Dict[str, str], K: int = 32, root: str = COUT) -> List[dict]:
    """M4.6c weaker-side battery for each final {tag: spec} (its own policy in seat B)."""
    from ..m46c import evaluate as EV
    from ..m46c.archive import lead_cell
    cfg = C.arm_cfg("sp")
    js = []
    for tag, spec in finals.items():
        for i, (focal, st, sit, d) in EV.battery_cells():
            cell = f"B+{d}" if focal == "weak" else f"A+{d}"
            cands = EV.WEAK_CANDS if focal == "weak" else EV.STRONG_CANDS
            js.append(_job("weak", tag, i, lead_cell(cfg, st, sit, cell), spec, cands, K, EV.H,
                           label=f"{focal} {st} {sit} +{d}", meta={"focal": focal, "stage": st, "sit": sit, "d": d},
                           root=root))
    return js


def premium_jobs(finals: Dict[str, str], K: int = PREMIUM_K, horizon: int = 50, root: str = COUT) -> List[dict]:
    """M4.6c J1 lead premium for each final: the final in both seats on every lead cell."""
    from ..m46c.archive import lead_cell, lead_cells
    cfg = C.arm_cfg("sp")
    js = []
    for tag, spec in finals.items():
        for i, (st, sit, c) in lead_cells():
            js.append(_job("premium", tag, i, lead_cell(cfg, st, sit, c), spec, {"self": ("policy",)}, K, horizon,
                           label=f"{st} {sit} {c}", meta={"stage": st, "sit": sit, "cell": c}, root=root))
    return js


def _cross(family: str, seed: int, items: list, specs: Dict[str, str], root: str, extra: Optional[dict] = None):
    """Cross-arm jobs: candidates {arm: ("net", spec)} (+ extra) against each arm's final in seat B."""
    cands = {a: ("net", sp) for a, sp in specs.items()}
    cands.update(extra or {})
    js = []
    for i, (lab, st, meta, K, H) in enumerate(items):
        for opp, osp in specs.items():
            j = _job(family, f"s{seed}_vs_{opp}", i, st, specs[list(specs)[0]], cands, K, H, proto=f"net:{osp}",
                     label=lab, meta=meta, root=root)
            j["roll"] = _roll(family, K)
            js.append(j)
    return js


def conv_jobs(seed: int, specs: Dict[str, str], root: str = COUT) -> List[dict]:
    from ..m46f import behaviour as BH
    items = [(lab, st, m, BH.K_CONV, BH.H_CONV) for lab, st, m in BH.conv_cells(C.arm_cfg("sp"))]
    return _cross("conv", seed, items, specs, root)


def freeze_jobs(seed: int, specs: Dict[str, str], root: str = COUT) -> List[dict]:
    from ..m46f import behaviour as BH
    items = [(lab, st, m, BH.K_FZ, BH.H_FZ) for lab, st, m in BH.freeze_cells(C.arm_cfg("sp"))]
    return _cross("freeze", seed, items, specs, root)


def recall_jobs(seed: int, specs: Dict[str, str], root: str = COUT) -> List[dict]:
    from ..m46f import behaviour as BH
    items = [(lab, st, m, BH.K_RW, BH.H_RW) for lab, st, m in BH.recall_cells(C.arm_cfg("sp"))]
    return _cross("recall", seed, items, specs, root)


def natural_states(seed: int, workers: int = 1, n_lead: int = 24, n_freeze: int = 16) -> Dict[str, list]:
    """The common start's natural leader / trailer / freezer states (`m46c.evaluate.states_job` rules, M5 seeds)."""
    from ..m46c import evaluate as EV
    spec = C.start_spec(seed)
    out = {}
    for role, n in (("leader", n_lead), ("trailer", n_lead), ("freezer", n_freeze)):
        p = os.path.join(C.CKPT, "states", f"s{seed}", f"{role}.json.gz")
        if not os.path.exists(p):
            _, d = _states_job_m5((spec, role, n, EV.ROLE_OFFSET[role]))
            C.save_gz(p, d)
        out[role] = C.load_gz(p)["states"]
    return out


def _states_job_m5(args) -> tuple:
    """m46c.evaluate.states_job's collection rule without its file cache (M5 writes its own)."""
    spec, role, n, offset = args
    from ..m46c import common as C46
    from ..m46c import evaluate as EV
    from ...m45b import engine as E
    from ...m45b import states as S
    from ..m45c import horizon as HZ
    cfg = C.setup("sp")
    net = C46.net_of(spec)
    pred = EV._lead_pred(role) if role in ("leader", "trailer") else EV._freeze_pred(role)
    s_idx = int(spec.split(":")[1])
    out, games = [], 0
    for g in range(300):
        sd = STATE_ROOT * 1000 + 100 * s_idx + offset + g
        pa, pb = HZ.CRNNet(net, 0, sd, 0, 300), HZ.CRNNet(net, 0, sd, 1, 300)
        out += E.collect(cfg, S.s0(cfg), pa, pb, [sd], 300, pred, per_game=2, min_turn=20, rng_seed=sd)
        games += 1
        if len(out) >= n:
            break
    return (spec, role), {"states": [C46.state_to_list(x["start"]) for x in out[:n]],
                          "turns": [x["turn"] for x in out[:n]], "games": games, "spec": spec, "role": role}


def natural_jobs(seed: int, specs: Dict[str, str], states: Dict[str, list], root: str = COUT) -> List[dict]:
    from ..m46c import common as C46
    from ..m46f import behaviour as BH
    items = []
    for role in ("leader", "trailer"):
        for i, v in enumerate(states.get(role, [])):
            items.append((f"natural {role} {i}", C46.state_from_list(v), {"role": role, "i": i}, BH.K_NAT, BH.H_NAT))
    for i, v in enumerate(states.get("freezer", [])):
        lane, m4, k = C46.state_from_list(v)
        items.append((f"natural freezer {i}", (lane, m4, k), {"role": "freezer", "i": i}, BH.K_NAT, BH.H_NATFZ))
        items.append((f"natural breaker {i}", (lane.mirror(), m4.mirror(), k), {"role": "breaker", "i": i}, BH.K_NAT,
                      BH.H_NATFZ))
    return _cross("natural", seed, items, specs, root)


def specialist_target(seed: int) -> int:
    """M4.6a trained gap specialists against the M4.5c finals of seeds 0, 2 and 4 only; seeds 1, 3, 5 use the specialist
    of seed s - 1 (M4.6a: a specialist's better play transfers to every final it never trained against, 15 of 15)."""
    from ..m46a.common import E4_TARGETS
    return seed if seed in E4_TARGETS else max(t for t in E4_TARGETS if t <= seed)


def gaps_jobs(seed: int, specs: Dict[str, str], root: str = COUT, K: int = 16, H: int = 150) -> List[dict]:
    """M4.6a E4 large-gap cells, with an M4.6a gap specialist (`specialist_target`) as an extra candidate: specialist -
    arm = how much a gap-trained policy out-converts the arm (a generalization diagnostic)."""
    from ..m46a import e4 as E4
    from ..m46a.states import zoning_gap_state
    from ..m46c import common as C46
    cfg = C.arm_cfg("sp")
    items = [(f"gap {c}", zoning_gap_state(cfg, *c), {"cell": list(c)}, K, H) for c in E4.GAP_CELLS + E4.EQUAL_CELLS]
    extra = {"gap_specialist": ("net", f"spec:{C46.specialist_path('gap', specialist_target(seed))}")}
    return _cross("gaps", seed, items, specs, root, extra)


def arm_specs(seed: int, phase: str = "main", at: Optional[int] = None) -> Dict[str, str]:
    pre = "" if phase == "main" else f"{phase}/"
    return {a: f"{pre}{a}:{seed}" + (f"@{at}" if at else "") for a in C.ARMS}


def run(js: List[dict], workers: int, label: str) -> Dict[str, dict]:
    js = sorted(js, key=lambda j: -len(j["roll"]) * len(j["cands"]) * j["horizon"])
    t0 = time.perf_counter()
    res = C.run_jobs(_light_job, js, workers, label)
    print(f"  [{label}] {len(js)} jobs in {time.perf_counter() - t0:.0f} s", flush=True)
    return dict(res)


def _light_job(job: dict) -> tuple:
    key, rec = paired_job(job)
    return key, {"path": job.get("path"), "elapsed_s": rec.get("elapsed_s"), "turns": rec.get("turns")}


__all__ = ["paired_job", "weak_jobs", "premium_jobs", "conv_jobs", "freeze_jobs", "recall_jobs", "natural_states",
           "natural_jobs", "gaps_jobs", "arm_specs", "run", "COUT", "PREMIUM_K"]
