"""Analysis of M4.6d (pre-registration §3-§5): the matrix stages (A, AL, Ab, B, N, conf), the specialists (verify,
decomp) and the critic (calib, rank). Writes results/m46d/analysis.json; `report.py` renders it.

Units: every contrast is first averaged over the paired rollouts of a (cell, policy) job, then over the cells of a
class within the policy; the 95% t interval is over the six policies (5 df). Values are seat A's discounted lead G_h
(resources) unless a table says "B's units" (-G).
"""

from __future__ import annotations

import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from . import cells as CL
from . import common as C
from . import matrix as M

H0 = 50
FZ = ("policy", "guard_sel", "passive", "push", "guard_walk")        # guard_walk: amendment 3 (exploratory)
BR = ("policy", "commit", "contest_sel", "concede", "reset")
BR_B = ("policy", "commit", "contest_sel", "concede", "concede3", "reset")      # stage B (amendment 2)
CHANNELS = ("held", "earned", "farm", "farm_avail", "unc", "unc_away", "xp", "deaths", "away", "recalls", "rec_intr",
            "d_total", "trade_conn", "plates", "bounty", "plate")
MATERIAL = 0.25
SELECT = 0.15


def ci(v) -> List[float]:
    m, h = C.mean_ci(v)
    return [m, h]


def merge(d: dict, e: dict) -> dict:
    """Append the combos of a paired record e (same cell, policy and rollout seeds) to d."""
    assert d["G"].shape[1:] == e["G"].shape[1:] and list(d["cps"]) == list(e["cps"])
    out = dict(d)
    out["G"] = np.concatenate([d["G"], e["G"]])
    out["L"] = np.concatenate([d["L"], e["L"]])
    out["forced"] = np.concatenate([d["forced"], e["forced"]])
    out["combos"] = d["combos"] + e["combos"]
    return out


def load_stage(stage: str, arm: str) -> Dict[Tuple[str, int], dict]:
    if stage == "A":                                     # A with the paired exploratory A2 combos (amendment 3)
        a, a2 = load_stage("A_only", arm), load_stage("A2", arm)
        return {k: (merge(v, a2[k]) if k in a2 else v) for k, v in a.items()}
    if stage == "A_only":
        stage = "A"
    out = {}
    for p in glob.glob(os.path.join(C.CKPT, "matrix", stage, arm, "*", "*.npz")):
        if p.endswith(".tmp.npz"):
            continue
        tag = os.path.basename(os.path.dirname(p))
        idx = int(os.path.basename(p)[:-4])
        out[(tag, idx)] = M.load(p)
    return out


def g(d: dict, combo: str, h: int = H0) -> np.ndarray:
    ci_ = d["combos"].index(combo)
    j = list(d["cps"]).index(h)
    return d["G"][ci_, :, j]


def led(d: dict, combo: str, key: str, h: int, side: int) -> np.ndarray:
    ci_ = d["combos"].index(combo)
    j = list(d["lcps"]).index(h)
    return d["L"][ci_, :, j, d["keys"].index(key), side]


def has(d: dict, combo: str) -> bool:
    return combo in d["combos"]


def contrast(d, c1, c2, h=H0) -> float:
    return float(np.mean(g(d, c1, h) - g(d, c2, h)))


def seed_of(tag: str) -> int:
    return int(tag.split("_")[0][1:])


# ---------------------------------------------------------------------------------------------- A
def a_tables(stage: str = "A", arm: str = "C1") -> dict:
    recs = load_stage(stage, arm)
    if not recs:
        return {}
    cells = CL.grid_a()
    by_cell: Dict[int, Dict[int, dict]] = defaultdict(dict)
    for (tag, idx), d in recs.items():
        by_cell[idx][seed_of(tag)] = d
    out = {"cells": {}, "classes": {}}
    per_cell_pol = {}
    for idx, pols in sorted(by_cell.items()):
        cell = cells[idx]
        any_d = next(iter(pols.values()))
        hs = [h for h in (10, 20, 50, 100, 150) if h in list(any_d["cps"])]
        lh = [h for h in (20, 50, 100) if h in list(any_d["lcps"])]
        brs = [b for b in BR if has(any_d, f"policy|{b}")]
        row = {"label": CL.cell_label(cell), "n_pol": len(pols), "abs": {}, "vs_policy": {}, "contrasts": {},
               "channels": {}, "matrix": {}}
        for h in hs:
            row["abs"][str(h)] = ci([float(np.mean(g(d, "policy|policy", h))) for d in pols.values()])
        fzs = [f for f in FZ if has(any_d, f"{f}|policy")]
        row["freezers"] = fzs
        for f in fzs[1:]:
            row["vs_policy"][f] = {str(h): ci([contrast(d, f"{f}|policy", "policy|policy", h) for d in pols.values()])
                                   for h in hs}
        pairs = [("guard_sel", "passive"), ("guard_sel", "push"), ("passive", "push"), ("guard_sel", "policy")]
        if "guard_walk" in fzs:
            pairs += [("guard_walk", "passive"), ("guard_walk", "push"), ("guard_walk", "guard_sel")]
        for a, b in pairs:
            row["contrasts"][f"{a}-{b}"] = {str(h): ci([contrast(d, f"{a}|policy", f"{b}|policy", h)
                                                         for d in pols.values()]) for h in hs}
        for f in fzs:
            ch = {}
            for h in lh:
                for key in CHANNELS:
                    for side, sn in ((0, "A"), (1, "B")):
                        vals = [float(np.mean(led(d, f"{f}|policy", key, h, side) -
                                              led(d, "policy|policy", key, h, side))) for d in pols.values()]
                        ch[f"{key}_{sn}@{h}"] = ci(vals)
                ch[f"held_abs_A@{h}"] = ci([float(np.mean(led(d, f"{f}|policy", "held", h, 0))) for d in pols.values()])
            row["channels"][f] = ch
        # the restricted game on the policy-mean payoffs (A's G at h)
        from ....matrix_game import solve_matrix_game
        for h in [x for x in (20, 50, 150) if x in hs]:
            fm = [f for f in fzs if all(has(any_d, f"{f}|{b}") for b in brs)]
            Mx = np.array([[float(np.mean([np.mean(g(d, f"{f}|{b}", h)) for d in pols.values()])) for b in brs]
                           for f in fm])
            sol = solve_matrix_game(Mx)
            per_pol_weight = []
            for d in pols.values():
                Mp = np.array([[float(np.mean(g(d, f"{f}|{b}", h))) for b in brs] for f in fm])
                per_pol_weight.append(solve_matrix_game(Mp).x.tolist())
            row["matrix"][str(h)] = {"M": Mx.tolist(), "freezers": fm, "breakers": brs, "value": float(sol.value),
                                     "x": sol.x.tolist(),
                                     "y": sol.y.tolist(), "maximin": Mx.min(axis=1).tolist(),
                                     "best_reply": [brs[int(np.argmin(r))] for r in Mx],
                                     "per_policy_x": per_pol_weight,
                                     "maximin_ci": {f: ci([float(min(np.mean(g(d, f"{f}|{b}", h)) for b in brs))
                                                           for d in pols.values()]) for f in fm}}
        # breaker options vs each freezer plan (B's units)
        row["breaker"] = {f: {b: ci([-contrast(d, f"{f}|{b}", f"{f}|policy") for d in pols.values()])
                              for b in brs[1:] if has(any_d, f"{f}|{b}")} for f in fzs}
        out["cells"][str(idx)] = row
        per_cell_pol[idx] = pols
    # classes: by level and by strength name (held 12 only), contrasts averaged over cells within policy
    def cls(name, sel):
        ids = [i for i in per_cell_pol if sel(cells[i])]
        if not ids:
            return
        seeds = sorted(set.intersection(*[set(per_cell_pol[i]) for i in ids]))
        hs = [h for h in (20, 50, 150) if h in list(per_cell_pol[ids[0]][seeds[0]]["cps"])]
        res = {"cells": len(ids)}
        d0 = per_cell_pol[ids[0]][seeds[0]]
        fzs = [f for f in FZ if has(d0, f"{f}|policy")]
        for f in fzs[1:]:
            res[f] = {str(h): ci([np.mean([contrast(per_cell_pol[i][s], f"{f}|policy", "policy|policy", h)
                                           for i in ids]) for s in seeds]) for h in hs}
        pairs = [("guard_sel", "passive"), ("guard_sel", "push"), ("passive", "push")]
        if "guard_walk" in fzs:
            pairs += [("guard_walk", "passive"), ("guard_walk", "push"), ("guard_walk", "guard_sel")]
        for a, b in pairs:
            res[f"{a}-{b}"] = {str(h): ci([np.mean([contrast(per_cell_pol[i][s], f"{a}|policy", f"{b}|policy", h)
                                                    for i in ids]) for s in seeds]) for h in hs}
        res["held_A@20"] = {f: ci([np.mean([np.mean(led(per_cell_pol[i][s], f"{f}|policy", "held", 20, 0))
                                            for i in ids]) for s in seeds]) for f in fzs}
        res["deaths_B@50"] = {f: ci([np.mean([np.mean(led(per_cell_pol[i][s], f"{f}|policy", "deaths", 50, 1))
                                              for i in ids]) for s in seeds]) for f in fzs}
        out["classes"][name] = res
    for L in CL.A_LEVELS:
        cls(f"level {L}", lambda c, L=L: c["L"] == L and c["held"] == 12)
    for nm in CL.A_CONDS:
        cls(f"cond {nm}", lambda c, nm=nm: c["name"] == nm and c["held"] == 12)
    cls("plausible early (edge, plaus at L3/L5)", lambda c: c["name"] in ("edge", "plaus") and c["L"] in (3, 5)
        and c["held"] == 12)
    cls("held 6 (L5)", lambda c: c["held"] == 6)
    return out


# ---------------------------------------------------------------------------------------------- B
def b_tables() -> dict:
    recs = load_stage("B", "C1")
    if not recs:
        return {}
    cells = CL.grid_b()
    by_cell: Dict[int, Dict[int, dict]] = defaultdict(dict)
    for (tag, idx), d in recs.items():
        by_cell[idx][seed_of(tag)] = d
    out = {}
    for idx, pols in sorted(by_cell.items()):
        cell = cells[idx]
        row = {"label": CL.cell_label(cell), "cell": cell, "freezer": {}}
        d0 = next(iter(pols.values()))
        for f in [x for x in ("policy", "guard_sel", "guard_walk") if has(d0, f"{x}|policy")]:
            vals = {b: [-contrast(d, f"{f}|{b}", f"{f}|policy") for d in pols.values()] for b in BR_B[1:]}
            vals["policy"] = [0.0] * len(pols)
            means = {b: float(np.mean(v)) for b, v in vals.items()}
            best = max(means, key=means.get)
            # options whose difference to the best has an interval including 0 (uncertain set)
            tie = []
            for b in BR_B:
                if b == best:
                    continue
                diff = [x - y for x, y in zip(vals[best], vals[b])]
                m, h = C.mean_ci(diff)
                if not (m - h > 0):
                    tie.append(b)
            bv = C.mean_ci(vals[best]) if best != "policy" else (0.0, 0.0)
            ch = {}
            for b in BR_B:
                ch[b] = {f"{k}_{sn}@50": ci([float(np.mean(led(d, f"{f}|{b}", k, 50, side) -
                                                           led(d, f"{f}|policy", k, 50, side))) for d in pols.values()])
                         for k in ("earned", "farm", "farm_avail", "unc", "unc_away", "xp", "deaths", "away",
                                   "recalls", "d_total", "held", "frozen") for side, sn in ((1, "B"), (0, "A"))}
            row["freezer"][f] = {"opt": {b: ci(v) for b, v in vals.items() if b != "policy"},
                                 "opt20": {b: ci([-contrast(d, f"{f}|{b}", f"{f}|policy", 20) for d in pols.values()])
                                           for b in BR_B[1:]},
                                 "policy_abs_B": ci([-float(np.mean(g(d, f"{f}|policy"))) for d in pols.values()]),
                                 "best": best, "uncertain_with": tie, "best_minus_policy": list(bv),
                                 "material": bool(best != "policy" and bv[0] >= MATERIAL and bv[0] - bv[1] > 0),
                                 "select": bool(best != "policy" and bv[0] >= SELECT and bv[0] - bv[1] > 0),
                                 "channels": ch}
        out[str(idx)] = row
    return out


# ---------------------------------------------------------------------------------------------- N
def n_tables() -> dict:
    recs = load_stage("N", "C1")
    if not recs:
        return {}
    out = {}
    for kind in ("favourable", "other"):
        per = defaultdict(list)
        for (tag, idx), d in recs.items():
            if tag.endswith(kind):
                per[seed_of(tag)].append(d)
        if not per:
            continue
        res = {"states": {s: len(v) for s, v in per.items()}}
        res["abs_policy"] = ci([np.mean([np.mean(g(d, "policy|policy")) for d in v]) for v in per.values()])
        d0 = next(iter(per.values()))[0]
        fzs = [f for f in FZ if has(d0, f"{f}|policy")]
        res["freezer"] = {f: ci([np.mean([contrast(d, f"{f}|policy", "policy|policy") for d in v])
                                 for v in per.values()]) for f in fzs[1:]}
        res["freezer20"] = {f: ci([np.mean([contrast(d, f"{f}|policy", "policy|policy", 20) for d in v])
                                   for v in per.values()]) for f in fzs[1:]}
        res["held20"] = {f: ci([np.mean([np.mean(led(d, f"{f}|policy", "held", 20, 0)) for d in v])
                                for v in per.values()]) for f in fzs}
        res["breaker"] = {f: {b: ci([np.mean([-contrast(d, f"{f}|{b}", f"{f}|policy") for d in v])
                                     for v in per.values()]) for b in BR[1:]} for f in fzs}
        res["guard-passive"] = ci([np.mean([contrast(d, "guard_sel|policy", "passive|policy") for d in v])
                                   for v in per.values()])
        res["guard-push"] = ci([np.mean([contrast(d, "guard_sel|policy", "push|policy") for d in v])
                                for v in per.values()])
        from ....matrix_game import solve_matrix_game
        Mx = np.array([[float(np.mean([np.mean([np.mean(g(d, f"{f}|{b}")) for d in v]) for v in per.values()]))
                        for b in BR] for f in fzs])
        sol = solve_matrix_game(Mx)
        res["matrix"] = {"M": Mx.tolist(), "freezers": fzs, "value": float(sol.value), "x": sol.x.tolist(),
                         "y": sol.y.tolist(), "maximin": Mx.min(axis=1).tolist()}
        if "guard_walk" in fzs:
            res["walk-passive"] = ci([np.mean([contrast(d, "guard_walk|policy", "passive|policy") for d in v])
                                      for v in per.values()])
            res["walk-push"] = ci([np.mean([contrast(d, "guard_walk|policy", "push|policy") for d in v])
                                   for v in per.values()])
        out[kind] = res
    return out


# ---------------------------------------------------------------------------------------------- confirmation
def selected_contrasts() -> dict:
    """The confirmation jobs (pre-registration §3, with the amendment that adds the freezer side of A and N)."""
    jobs = []
    cfg = C.arm_cfg("C1")
    sel_log = []
    bt = b_tables()
    for idx, row in bt.items():
        cell = row["cell"]
        for f, r in row["freezer"].items():
            if not r["select"]:
                continue
            opts = sorted(r["opt"], key=lambda b: -r["opt"][b][0])
            combos = [(f, "policy"), (f, opts[0])] + ([(f, opts[1])] if len(opts) > 1 else [])
            sel_log.append({"stage": "B", "cell": row["label"], "freezer": f, "selected": opts[0],
                            "discovery": r["opt"][opts[0]]})
            st = CL.cell_start(cfg, cell)
            br = (M.P.BREAKER if cell["wave"] == "freeze" else M.P.BREAKER_CONTROL)
            fz = {"policy": None, "guard_sel": M.P.FREEZER["guard_sel"], "guard_walk": M.P.FREEZER["guard_walk"]}
            for s in C.SEEDS:
                jobs.append({"arm": "C1", "spec": C.general("C1", s), "tag": f"B{idx}_{f}_s{s}", "idx": int(idx),
                             "start": C.state_to_list(st), "a": fz, "b": {k: br[k] for k in BR_B},
                             "combos": [list(c) for c in combos], "K": 48, "H": 50, "label": row["label"]})
    at = a_tables("A", "C1")
    cells = CL.grid_a()
    for idx, row in at.get("cells", {}).items():
        cell = cells[int(idx)]
        # pre-registered: L3/L5 edge and plaus always; amendment 4: every A cell's freezer plans vs the breaker policy
        always = True
        prereg_always = cell["name"] in ("edge", "plaus") and cell["L"] in (3, 5) and cell["held"] == 12
        better = [f for f in row["vs_policy"] if row["vs_policy"][f]["50"][0] >= SELECT and
                  row["vs_policy"][f]["50"][0] - row["vs_policy"][f]["50"][1] > 0]
        bsel = [b for b in BR[1:] if row["breaker"]["policy"][b][0] >= SELECT and
                row["breaker"]["policy"][b][0] - row["breaker"]["policy"][b][1] > 0]
        if not (always or better or bsel):
            continue
        combos = [(f, "policy") for f in FZ] + [("policy", b) for b in bsel]
        sel_log.append({"stage": "A", "cell": row["label"], "prereg_always": prereg_always, "freezer_better": better,
                        "breaker_better": bsel})
        st = CL.cell_start(cfg, cell)
        for s in C.SEEDS:
            jobs.append({"arm": "C1", "spec": C.general("C1", s), "tag": f"A{idx}_s{s}", "idx": int(idx),
                         "start": C.state_to_list(st), "a": {f: M.P.FREEZER[f] for f in FZ if f != "policy"} |
                         {"policy": None}, "b": M.P.A_BREAKER,
                         "combos": [list(c) for c in combos], "K": 48, "H": 50, "label": row["label"]})
    # N (natural states): kinds where a freezer plan or a breaker option beats the policy (same rule); confirmed on
    # the same states with independent rollout seeds
    nt = n_tables()
    if nt:
        from . import matrix as MX
        states = MX.natural_states(1)
        for kind, r in nt.items():
            fbet = [f for f, v in r["freezer"].items() if v[0] >= SELECT and v[0] - v[1] > 0]
            bbet = {f: [b for b, v in r["breaker"][f].items() if v[0] >= SELECT and v[0] - v[1] > 0]
                    for f in r["breaker"]}
            combos = [("policy", "policy")] + [(f, "policy") for f in fbet]
            for f, bs in bbet.items():
                for b in bs:
                    combos += [(f, "policy"), (f, b)]
            combos = list(dict.fromkeys(combos))
            if len(combos) == 1:
                continue
            sel_log.append({"stage": "N", "kind": kind, "freezer_better": fbet, "breaker_better": bbet})
            for sd in C.SEEDS:
                spec = C.general("C1", sd)
                for i, rec in enumerate(states[spec][kind]):
                    jobs.append({"arm": "C1", "spec": spec, "tag": f"N{kind}{i}_s{sd}", "idx": i,
                                 "start": rec["start"], "a": {f: M.P.FREEZER[f] for f in FZ if f != "policy"} |
                                 {"policy": None}, "b": M.P.A_BREAKER, "combos": [list(c) for c in combos], "K": 48,
                                 "H": 50, "label": f"natural {kind} {i}"})
    return {"jobs": jobs, "log": sel_log}


def conf_tables() -> dict:
    recs = load_stage("conf", "C1")
    if not recs:
        return {}
    groups = defaultdict(dict)
    nat = defaultdict(lambda: defaultdict(list))
    for (tag, idx), d in recs.items():
        grp, s = tag.rsplit("_s", 1)
        if grp.startswith("N"):
            kind = "favourable" if grp.startswith("Nfavourable") else "other"
            nat[kind][int(s)].append(d)
            continue
        groups[grp][int(s)] = d
    out = {}
    for grp, pols in groups.items():
        any_d = next(iter(pols.values()))
        res = {"label": str(any_d["label"]), "combos": any_d["combos"]}
        if grp.startswith("B"):
            f = grp.split("_", 1)[1]
            res["opt"] = {c.split("|")[1]: ci([-contrast(d, c, f"{f}|policy") for d in pols.values()])
                          for c in any_d["combos"] if not c.endswith("|policy")}
        else:
            res["freezer"] = {c.split("|")[0]: ci([contrast(d, c, "policy|policy") for d in pols.values()])
                              for c in any_d["combos"] if c.endswith("|policy") and c != "policy|policy"}
            res["breaker"] = {c.split("|")[1]: ci([-contrast(d, c, "policy|policy") for d in pols.values()])
                              for c in any_d["combos"] if c.startswith("policy|") and c != "policy|policy"}
            if "guard_sel|policy" in any_d["combos"] and "passive|policy" in any_d["combos"]:
                res["guard-passive"] = ci([contrast(d, "guard_sel|policy", "passive|policy") for d in pols.values()])
                res["guard-push"] = ci([contrast(d, "guard_sel|policy", "push|policy") for d in pols.values()])
                res["passive-push"] = ci([contrast(d, "passive|policy", "push|policy") for d in pols.values()])
        out[grp] = res
    for kind, per in nat.items():
        d0 = next(iter(per.values()))[0]
        res = {"label": f"natural {kind} (states per policy: {[len(v) for v in per.values()]})", "combos": d0["combos"]}
        res["freezer"] = {c.split("|")[0]: ci([np.mean([contrast(d, c, "policy|policy") for d in v])
                                               for v in per.values()])
                          for c in d0["combos"] if c.endswith("|policy") and c != "policy|policy"}
        res["breaker_vs"] = {c: ci([np.mean([-contrast(d, c, c.split("|")[0] + "|policy") for d in v])
                                    for v in per.values()])
                             for c in d0["combos"] if not c.endswith("|policy")}
        out[f"N {kind}"] = res
    return out


def conf_classes() -> dict:
    """The confirmation sample's A classes (as `a_tables` classes: contrasts averaged over a class's cells within each
    policy, interval over the six policies), with the discovery value of the same contrast beside it."""
    recs = load_stage("conf", "C1")
    cells = CL.grid_a()
    per_cell = defaultdict(dict)
    for (tag, idx), d in recs.items():
        grp, s = tag.rsplit("_s", 1)
        if grp.startswith("A"):
            per_cell[int(grp[1:])][int(s)] = d
    if not per_cell:
        return {}
    disc = a_tables("A", "C1").get("classes", {})
    out = {}

    def cls(name, sel):
        ids = [i for i in per_cell if sel(cells[i])]
        if not ids:
            return
        seeds = sorted(set.intersection(*[set(per_cell[i]) for i in ids]))
        if len(seeds) < 2:
            return
        res = {"cells": len(ids)}
        for f in FZ[1:]:
            res[f] = ci([np.mean([contrast(per_cell[i][s], f"{f}|policy", "policy|policy") for i in ids])
                         for s in seeds])
        for a, b in (("guard_sel", "passive"), ("guard_sel", "push"), ("guard_walk", "passive"),
                     ("guard_walk", "push"), ("passive", "push")):
            res[f"{a}-{b}"] = ci([np.mean([contrast(per_cell[i][s], f"{a}|policy", f"{b}|policy") for i in ids])
                                  for s in seeds])
        dd = disc.get(name, {})
        res["discovery"] = {k: dd[k]["50"] for k in dd if isinstance(dd[k], dict) and "50" in dd[k]}
        out[name] = res
    for L in CL.A_LEVELS:
        cls(f"level {L}", lambda c, L=L: c["L"] == L and c["held"] == 12)
    for nm in CL.A_CONDS:
        cls(f"cond {nm}", lambda c, nm=nm: c["name"] == nm and c["held"] == 12)
    cls("plausible early (edge, plaus at L3/L5)", lambda c: c["name"] in ("edge", "plaus") and c["L"] in (3, 5)
        and c["held"] == 12)
    cls("held 6 (L5)", lambda c: c["held"] == 6)
    cls("all held-12 cells", lambda c: c["held"] == 12)
    # pooled discovery + confirmation (both independent seed sets, same cells) for the cell table
    pooled = {}
    at = a_tables("A", "C1").get("cells", {})
    for i, pols in per_cell.items():
        r = at.get(str(i))
        if not r:
            continue
        pooled[str(i)] = {"label": r["label"],
                          **{f: ci([contrast(pols[s], f"{f}|policy", "policy|policy") for s in sorted(pols)])
                             for f in FZ[1:]}}
    out["_cells_confirmation"] = pooled
    return out


def run() -> dict:
    data = {"meta": C.meta(stage="analyse")}
    data["A"] = a_tables("A", "C1")
    data["AL"] = a_tables("AL", "C1")
    data["Ab"] = a_tables("Ab", "baseline")
    data["B"] = b_tables()
    data["N"] = n_tables()
    data["conf"] = conf_tables()
    data["conf_classes"] = conf_classes()
    try:
        from . import analysis_c as AC
        data["C"] = AC.run()
    except ImportError:
        pass
    C.save_json("analysis", data)
    return data


__all__ = ["a_tables", "b_tables", "n_tables", "selected_contrasts", "conf_tables", "run", "load_stage"]
