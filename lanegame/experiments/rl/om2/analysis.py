"""OM2 readouts and the pre-registered classification (pre-registration §3-§4).

Contrasts are treatment − control on the same paired-seat games (common random numbers), per opponent; a family's value
for a seed is the mean over its opponents; seeds are the units (3 pilot seeds: the classification requires the seeds to
agree in sign, and reports t intervals with 2 df beside pooled-game intervals). R3 is the treatment's own lead against
the control.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import common as C
from . import evaluate as EV

M = 0.25


def _cell(s: int, fam: str, opp: str, phase: str = "main") -> dict:
    import os
    j = [x for x in EV.jobs((s,), phase) if x["family"] == fam and x["opponent"] == opp][0]
    return EV.load(j["path"]) if os.path.exists(j["path"]) else None


def contrasts(phase: str = "main", seeds=C.PILOT_SEEDS) -> dict:
    fams = {"E_in": [o for o in C.ARCHETYPES], "E_out": list(C.HELDOUT_SCRIPTED), "R1": EV.FAMILIES["R1"],
            "R2": EV.FAMILIES["R2"]}
    out = {"per_opponent": {}, "per_seed": {}, "pooled": {}}
    for fam, opps in fams.items():
        famkey = "E" if fam.startswith("E") else fam
        per_seed = {}
        pooled = {"read": [], "lane": [], "end": []}
        for s in seeds:
            vals = []
            for o in opps:
                d = _cell(s, famkey, o, phase)
                if d is None:
                    continue
                diff = {k: EV.pair_lead(d["treatment__r"], t) - EV.pair_lead(d["control__r"], t)
                        for k, t in (("read", EV.READ_TURN), ("lane", EV.LANE_END), ("end", d["treatment__r"].shape[-1]))}
                for k in pooled:
                    pooled[k].append(diff[k])
                vals.append(float(diff["read"].mean()))
                rec = out["per_opponent"].setdefault(o, {"family": fam, "per_seed": {}, "control": {}, "treatment": {}})
                rec["per_seed"][str(s)] = float(diff["read"].mean())
                rec["control"][str(s)] = float(EV.pair_lead(d["control__r"]).mean())
                rec["treatment"][str(s)] = float(EV.pair_lead(d["treatment__r"]).mean())
            if vals:
                per_seed[s] = float(np.mean(vals))
        out["per_seed"][fam] = C.seed_stats(per_seed) if per_seed else {}
        out["pooled"][fam] = {k: C.mean_se(np.concatenate(v)) for k, v in pooled.items() if v}
    h = {}
    hp = []
    for s in seeds:
        d = _cell(s, "R3", "h2h", phase)
        if d is not None:
            x = EV.pair_lead(d["treatment__r"])
            h[s] = float(x.mean())
            hp.append(x)
    out["per_seed"]["R3"] = C.seed_stats(h) if h else {}
    out["pooled"]["R3"] = {"read": C.mean_se(np.concatenate(hp))} if hp else {}
    return out


def identification(phase: str = "main", seeds=C.PILOT_SEEDS) -> dict:
    """The treatment's own belief in its games against library opponents (MAP correct after 20 / 40 / 94 turns) and
    the catch-alls' mass against R2 opponents."""
    out = {"library": {}, "R2_catch_all_mass": {}}
    idx = {t: EV.CHECK_TURNS.index(t) for t in (10, 20, 40, 94)}
    for o in C.ARCHETYPES:
        ti = C.BELIEF_TYPES.index(o)
        P = []
        for s in seeds:
            d = _cell(s, "E", o, phase)
            if d is not None:
                P.append(np.exp(d["treatment__trace"].astype(np.float64)).reshape(-1, len(EV.CHECK_TURNS), C.K))
        if P:
            P = np.concatenate(P)
            out["library"][o] = {str(t): float((P[:, j].argmax(-1) == ti).mean()) for t, j in idx.items()}
    for o in EV.FAMILIES["R2"]:
        P = []
        for s in seeds:
            d = _cell(s, "R2", o, phase)
            if d is not None:
                P.append(np.exp(d["treatment__trace"].astype(np.float64)).reshape(-1, len(EV.CHECK_TURNS), C.K))
        if P:
            P = np.concatenate(P)
            out["R2_catch_all_mass"][o] = {str(t): float(P[:, j, -2:].sum(-1).mean()) for t, j in idx.items()}
    return out


def mechanism(phase: str = "main", seeds=C.PILOT_SEEDS, games: int = 2, T: int = 60) -> dict:
    """Belief-swap probes (descriptive): on the treatment final's own observations against each archetype (ROOT
    TRACE_ROOT), the mean total-variation distance between its policy with the belief set to one type and with the
    prior; the same for the control as a check: its 13 belief inputs were always the same constant, so their weight
    columns received identical gradients from identical (zero) starts and stay equal, and its distances are 0 by
    construction."""
    import os
    from ...m45b import engine as E
    cfg = C.setup()
    types = C.belief_types()
    out = {}
    for s in seeds:
        paths = {a: os.path.join(C.run_dir(a, s, "smoke" if phase == "smoke_eval" else "main") if phase != "smoke_eval"
                                 else os.path.join(C.CKPT, "smoke", a), "final.pt") for a in C.ARMS}
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        tnet = C.om_net(paths["treatment"])
        seen = []
        for oi, o in enumerate(C.ARCHETYPES):
            for g in range(games):
                sd = C.TRACE_ROOT * 10 ** 7 + s * 10 ** 4 + oi * 100 + g
                seen += EV.play(cfg, ("arm", tnet, "treatment"), ("opp", o), sd, T, types, E.lane_cache(cfg),
                                collect=True)["seen"]
        out[str(s)] = {a: EV.probes(C.om_net(p), seen) for a, p in paths.items()}
        out[str(s)]["states"] = len(seen)
    return out


def classify(con: dict) -> dict:
    def gain(st):
        v = list(st.get("per_seed", {}).values())
        return bool(v) and st["mean"] >= M and all(x > 0 for x in v)

    def loss(st):
        v = list(st.get("per_seed", {}).values())
        return bool(v) and st["mean"] <= -M and all(x < 0 for x in v)

    E = gain(con["per_seed"]["E_in"])
    R = {f: loss(con["per_seed"][f]) for f in ("R1", "R2", "R3")}
    anyR = any(R.values())
    if E and not anyR:
        o, lab = 1, "modelling pays and is safe"
    elif E and anyR:
        o, lab = 2, "modelling pays but costs (a trade-off for the user)"
    elif not E and not anyR:
        o, lab = 3, "no material gain"
    else:
        o, lab = 4, "harmful"
    return {"outcome": o, "label": lab, "E_in_gain": E, "R_losses": R}


def run(phase: str = "main") -> dict:
    con = contrasts(phase)
    out = {"meta": C.meta(stage="analyse", phase=phase), "contrasts": con, "identification": identification(phase),
           "mechanism": mechanism(phase), "classification": classify(con)}
    C.save_json(f"analysis_{phase}", out)
    c = out["classification"]
    print(f"  outcome {c['outcome']}: {c['label']}", flush=True)
    return out


__all__ = ["contrasts", "identification", "classify", "run", "M"]
