"""Q1: unit usage of trained networks (pre-registration, stage 1; descriptive with one rule).

Each network plays itself (`m5.players.crn_policy`: CRNNet, or OMPolicy with its belief) through `m46a.engine.rollout`:
16 games of 500 turns from s0 and 100 turns from each held-out E4 gap start of target 0 (27 cells, 48 semi-natural
states). A forward pre-hook records every network input; every 5th is kept. Per hidden layer: inactive units (|h| >=
0.99 with one sign on >= 99% of states, or SD < 0.01), the saturation share (|h| >= 0.99) and the effective rank
(exp of the entropy of the normalized singular values of the centred activations) with the components for 99% variance.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from . import common as C

KEEP_EVERY = 5
GAMES, TURNS, GAP_TURNS = 16, 500, 100
SAT = 0.99


def networks() -> Dict[str, str]:
    from ..m45c.common import final_path
    from .probe import attacker_path
    out = {f"sp:{s}": f"sp:{s}" for s in C.SEEDS}
    out.update({f"om:{s}": f"om:{s}" for s in C.SEEDS})
    out.update({f"ext:{t}": final_path("ext", t) for t in (0, 2, 4)})
    out.update({f"gap_attacker:{t}": attacker_path(t) for t in (0, 2, 4)})
    return out


def pilot_networks() -> Dict[str, str]:
    out = {}
    for arm in C.PILOT_ARMS:
        for s in C.SEEDS:
            out[f"{arm}:{s}"] = C.final_path(arm, s)
    out.update({f"control:{s}": C.control_path(s) for s in C.SEEDS})
    return out


def activations(net, X: np.ndarray) -> List[np.ndarray]:
    import torch
    with torch.no_grad():
        x = torch.as_tensor(X)
        if hasattr(net, "_inputs"):
            x = net._inputs(x)
        out = []
        for layer in net.trunk:
            x = layer(x)
            if isinstance(layer, torch.nn.Tanh):
                out.append(x.double().numpy())
    return out


def usage(H: np.ndarray) -> dict:
    sat = np.abs(H) >= SAT
    pos, neg = (H >= SAT).mean(0), (H <= -SAT).mean(0)
    inactive = (np.maximum(pos, neg) >= 0.99) | (H.std(0) < 0.01)
    s = np.linalg.svd(H - H.mean(0), compute_uv=False)
    q = s / s.sum()
    q = q[q > 0]
    var = s ** 2 / (s ** 2).sum()
    return {"units": int(H.shape[1]), "inactive": int(inactive.sum()), "saturation_share": float(sat.mean()),
            "effective_rank": float(np.exp(-(q * np.log(q)).sum())),
            "components_99": int(np.searchsorted(np.cumsum(var), 0.99) + 1)}


def job(args) -> tuple:
    name, spec, path = args
    if os.path.exists(path):
        return name, C.load_gz(path)
    from ...m45b.states import s0
    from ..m46a import engine as EN
    from ..m46a.common import setup_worker
    from ..m5 import players as PL
    from .probe import starts
    cfg = setup_worker()
    net = PL.net_of(spec)
    rows: list = []
    hook = net.register_forward_pre_hook(lambda m, a: rows.append(a[0].detach().reshape(-1, a[0].shape[-1]).numpy()))
    try:
        for g in range(GAMES):
            seed = C.Q1_ROOT * 1000 + g
            EN.rollout(cfg, s0(cfg), PL.crn_policy(spec, 0, seed, 0, TURNS), PL.crn_policy(spec, 0, seed, 1, TURNS),
                       seed, TURNS, (), ())
        for i, st in enumerate(starts(0, "P_test")):
            seed = C.Q1_ROOT * 1000 + 500 + i
            EN.rollout(cfg, st, PL.crn_policy(spec, st[2], seed, 0, GAP_TURNS),
                       PL.crn_policy(spec, st[2], seed, 1, GAP_TURNS), seed, GAP_TURNS, (), ())
    finally:
        hook.remove()
    X = np.concatenate(rows)[::KEEP_EVERY].astype(np.float32)
    rec = {"network": name, "spec": spec, "inputs": int(X.shape[1]), "states": int(len(X)),
           "hidden": list(net.hidden), "layers": [usage(H) for H in activations(net, X)]}
    C.save_gz(path, rec)
    return name, rec


def run(workers: int, pilot: bool = False) -> dict:
    nets = pilot_networks() if pilot else networks()
    tag = "q1_pilot" if pilot else "q1"
    jobs = [(n, sp, os.path.join(C.CKPT, tag, n.replace(":", "_") + ".json.gz")) for n, sp in nets.items()]
    res = dict(C.run_jobs(job, jobs, workers, tag))
    out = {"meta": C.meta(stage=tag), "networks": res}
    if not pilot:
        l2 = [r["layers"][-1] for r in res.values()]
        med_inactive = float(np.median([x["inactive"] for x in l2]))
        med_sat = float(np.median([x["saturation_share"] for x in l2]))
        out["rule"] = {"median_inactive_layer2": med_inactive, "median_saturation_layer2": med_sat,
                       "leaves_width_unused": bool(med_inactive >= 32 or med_sat >= 0.5)}
    C.save_json(tag, out)
    return out
