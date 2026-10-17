"""OM2 exploratory reference: NOT pre-registered. Added on 2026-10-17 after the pilot's classification, to read it.

1. pi_s = ext/baseline:s, the common starting policy of both arms (90 inputs, no opponent information), plays every
   evaluation opponent of seed s on exactly the games of the pre-registered cells (same cell seeds, CRN uniforms
   [sd, 7, seat], opponents' generators [sd, 11, seat]), so arm − pi_s is a paired contrast of what each arm's
   training added.
2. Each arm's final plays pi_s (its own start) head to head: 64 paired-seat pairs, seeds REF_ROOT.

Values as in the evaluation (the lead at turn 114). Raw cells under CKPT/eval/main_ref/; summary in
results/om2/explore.json. Nothing here can change the pre-registered outcome.
"""

from __future__ import annotations

import os
import time
from typing import List

import numpy as np

from . import common as C
from . import evaluate as EV

REF_ROOT = 61_200
FAMS = {"E_in": list(C.ARCHETYPES), "E_out": list(C.HELDOUT_SCRIPTED), "R1": EV.FAMILIES["R1"],
        "R2": EV.FAMILIES["R2"]}


def jobs(seeds=C.PILOT_SEEDS, N: int = EV.N_PAIRS, T: int = EV.T_EVAL) -> List[dict]:
    js = []
    for j in EV.jobs(seeds, "main", N, T):
        if j["opponent"] == "h2h":
            continue
        js.append({"key": "ref|" + j["key"], "path": j["path"].replace(os.path.join("eval", "main"),
                                                                       os.path.join("eval", "main_ref")),
                   "seed": j["seed"], "opponent": j["opponent"], "family": j["family"], "kind": "opp",
                   "root": C.EVAL_ROOT, "cell": j["cell"], "N": N, "T": T})
    for s in seeds:
        for i, arm in enumerate(C.ARMS):
            js.append({"key": f"ref|main|s{s}|h2h_pi|{arm}",
                       "path": os.path.join(C.CKPT, "eval", "main_ref", f"s{s}", f"H2H_pi__{arm}.npz"),
                       "seed": s, "opponent": arm, "family": "H2H_pi", "kind": "arm", "root": REF_ROOT,
                       "cell": s * 100 + i, "arm_path": EV.final_path(arm, s), "N": N, "T": T})
    return js


def ref_job(j: dict) -> tuple:
    path = j["path"]
    if os.path.exists(path):
        return j["key"], {"path": path, "cached": True}
    from ...m45b import engine as E
    cfg = C.setup()
    ls = E.lane_cache(cfg)
    types = C.belief_types()
    pi = C.net90(C.start_path(int(j["seed"])))
    N, T = int(j["N"]), int(j["T"])
    t0 = time.perf_counter()
    R = np.zeros((N, 2, T), dtype=np.float32)
    opp = ("arm", C.om_net(j["arm_path"]), j["opponent"]) if j["kind"] == "arm" else ("opp", j["opponent"])
    me = ("pi", pi)
    for m in range(N):
        sd = int(j["root"]) * 10 ** 7 + int(j["cell"]) * 10 ** 4 + m
        for o, (sa, sb) in enumerate(((me, opp), (opp, me))):
            g = EV.play(cfg, sa, sb, sd, T, types, ls)
            R[m, o] = g["r"] if o == 0 else -g["r"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, pi__r=R, elapsed=np.asarray(time.perf_counter() - t0))
    os.replace(tmp, path)
    return j["key"], {"path": path, "elapsed": time.perf_counter() - t0}


def summarize(seeds=C.PILOT_SEEDS) -> dict:
    """Per family and seed: pi's value and each arm − pi (mean over the family's opponents); per opponent the same
    pooled over seeds; head to head: each arm's lead against pi_s."""
    js = {j["key"]: j for j in jobs(seeds)}
    main = {(j["seed"], j["family"], j["opponent"]): j["path"] for j in EV.jobs(seeds, "main")}
    out = {"per_seed": {}, "per_opponent": {}, "h2h_pi": {}}
    for fam, opps in FAMS.items():
        famkey = "E" if fam.startswith("E") else fam
        rec = {"pi": {}, "control-pi": {}, "treatment-pi": {}}
        for s in seeds:
            vals = {k: [] for k in rec}
            for o in opps:
                ref = EV.load(js[f"ref|eval|main|s{s}|{famkey}|{o}"]["path"])
                d = EV.load(main[(s, famkey, o)])
                p = EV.pair_lead(ref["pi__r"])
                vals["pi"].append(p.mean())
                po = out["per_opponent"].setdefault(o, {"family": fam, "pi": [], "control-pi": [], "treatment-pi": []})
                po["pi"].append(float(p.mean()))
                for arm in C.ARMS:
                    x = float((EV.pair_lead(d[f"{arm}__r"]) - p).mean())
                    vals[f"{arm}-pi"].append(x)
                    po[f"{arm}-pi"].append(x)
            for k in rec:
                rec[k][s] = float(np.mean(vals[k]))
        out["per_seed"][fam] = {k: C.seed_stats(v) for k, v in rec.items()}
    for o, po in out["per_opponent"].items():
        for k in ("pi", "control-pi", "treatment-pi"):
            po[k] = float(np.mean(po[k]))
    for arm in C.ARMS:
        h = {}
        for s in seeds:
            d = EV.load(js[f"ref|main|s{s}|h2h_pi|{arm}"]["path"])
            h[s] = float(-EV.pair_lead(d["pi__r"]).mean())     # the arm's lead against pi_s
        out["h2h_pi"][arm] = C.seed_stats(h)
    return out


def run(workers: int, seeds=C.PILOT_SEEDS) -> dict:
    t0 = time.perf_counter()
    C.run_jobs(ref_job, jobs(seeds), workers, "OM2 exploratory reference")
    out = {"meta": C.meta(stage="explore"), "note": "exploratory, not pre-registered; cannot change the outcome",
           "summary": summarize(seeds), "wall_s": time.perf_counter() - t0}
    C.save_json("explore", out)
    return out


__all__ = ["jobs", "ref_job", "summarize", "run", "REF_ROOT"]
