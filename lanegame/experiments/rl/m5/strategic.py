"""M5c analysis, secondary endpoint S8 (pre-registration §7 and §9, amendment 1): per-arm strategic replication and
the cross-arm limitation contrasts, from the battery (`battery`) and cell (`cells`) records. Descriptive with
intervals; replication by the registered rule; not part of the Holm family. Reuses the historical analysis code:
`m45c.battery.units / aggregate / validate` for the battery, `m46c.analysis.premium_of` for the lead premium, and the
M4.6c / M4.6f cell readouts (best forced response, om - sp per cell averaged over the two seat-B opponents).

Replication (registered rule, details fixed by amendment 1):
    gap of a battery family in a seed = max over the family's fixed alternatives (one-turn candidates and sustained
        plans) of the family mean of (alternative - policy) at the primary statistic (Q in natural families, G in
        constructed ones) at 50 turns, under protocol "policy" (the policy's own sampling play: competent opponents;
        the scripted protocols measure value against conceding styles, a different question, and are reported only);
    gap of the weaker-side cells = max over lead sizes (+8, +16) of the best forced weak-side response (hold, safe,
        trade, reset) minus the policy at G50, over every weak-focal cell (M4.6c §5.1's statistic);
    a verdict holds in a seed when the gap of each of its families is <= 0.2; it replicates with >= 5 of 6 seeds.
Completeness. A family's value needs every expected (state, protocol) record (missing records -> "partial"; the
    replication verdict stays partial unless the missing seeds cannot change it). A natural family whose state
    collection found fewer states than designed is analysed on the states it has and flagged ("state coverage short").
Horizon validation (M4.5c exception 3). Each battery family is validated on its 300-turn subset (`m45c.battery.
    validate`). "pass": the 50-turn statistic stands. "fail": the registered handling is a re-run of the family at 150
    turns for all its states and protocols (`battery.extension_jobs`, part of the battery stage); until it exists the
    family's value is "pending (validation failed)" and every verdict using it is pending; with it, the 150-turn
    statistic replaces the 50-turn one. "insufficient" (fewer than 3 validation pairs): used, flagged "unvalidated".
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import analysis as AN
from . import common as C

VERDICTS = {
    "conditional wave management": (("battery", "A_wave"),),
    "conditional freezing and the reply to freezes": (("battery", "B_freeze"), ("battery", "B_frozen")),
    "selective trading": (("battery", "C_trade"),),
    "concession under threat": (("battery", "H_zone_c"), ("weak", "weak side")),
    "Recall timing": (("battery", "E_recall"),),
}
REPLICATION_PROTOCOL = "policy"
CROSS = ("conv", "freeze", "recall", "natural", "gaps")


def _B():
    from ..m45c import battery as B
    return B


def battery_root() -> str:
    from . import battery as MB
    return MB.BOUT


def cells_root() -> str:
    from . import cells as CE
    return CE.COUT


# ---------------------------------------------------------------------------------------------- the battery
def expected_states(fam: str) -> int:
    B = _B()
    kind = B.FAMILIES[fam][0]
    if kind == "constructed":
        return len(B.GRIDS[fam])
    return B.N_EARLY if kind == "early" else B.N_NATURAL


def battery_coverage(tag: str, jobs: List[dict], root: Optional[str] = None) -> Dict[str, dict]:
    """Per family: designed and collected states, the expected (state, protocol) records and which are missing."""
    B = _B()
    man = C.jload(os.path.join(root or battery_root(), tag, "manifest.json")) or {}
    out = {}
    for fam, (kind, _, _, protos) in B.FAMILIES.items():
        design = expected_states(fam)
        collected = design if kind == "constructed" else (man.get("states", {}).get(fam) or {}).get("n")
        have = {(j["index"], j["protocol"]) for j in jobs if j["family"] == fam}
        n = collected if collected is not None else design
        exp = {(i, p) for i in range(n) for p in protos}
        miss = sorted(exp - have)
        out[fam] = {"designed_states": design, "collected_states": collected,
                    "state_coverage": "complete" if collected == design else
                    ("unknown (no manifest)" if collected is None else f"short ({collected}/{design})"),
                    "expected_records": len(exp), "present_records": len(exp & have),
                    "missing_records": [f"{i}|{p}" for i, p in miss[:10]] + ([f"... {len(miss) - 10} more"]
                                                                            if len(miss) > 10 else []),
                    "records": "complete" if not miss and collected is not None else
                    ("missing" if not (exp & have) else "partial")}
    return out


def validate_at(jobs: List[dict], fam: str, h: int) -> dict:
    """M4.5c's validation pass rule with h in place of 50: the h-turn primary statistic against its 300-turn value on
    the validation units. An unregistered diagnostic of the 150-turn re-run's horizon (the 300-turn records hold
    every checkpoint up to 300)."""
    import math
    B = _B()
    X = B.primary_statistic(fam)
    pairs = []
    for j in jobs:
        if j["family"] != fam or j["protocol"] != "policy" or j["horizon"] < B.VALIDATION_H:
            continue
        for cand in j["GQ"]:
            if cand == "policy":
                continue
            us = B.units([j], fam, "policy", cand, X)
            if us and h in us[0][1]:
                pairs.append((j["index"], cand, us[0][1][h], us[0][1][B.VALIDATION_H]))
    if len(pairs) < 3:
        return {"verdict": "insufficient", "pairs": len(pairs), "h": h}
    a = np.array([p[2].mean() for p in pairs])
    b = np.array([p[3].mean() for p in pairs])
    se = float((a - b).std(ddof=1) / math.sqrt(len(a)))
    slope = float(np.polyfit(a, b, 1)[0]) if np.ptp(a) > 0 else float("nan")

    def sig(x):
        s = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else np.inf
        return 0 if abs(x.mean()) <= 2 * s else int(np.sign(x.mean()))
    contra = [(i, c) for i, c, x, y in pairs if sig(x) and sig(y) and sig(x) != sig(y)]
    bias = float((a - b).mean())
    ok = abs(bias) <= max(0.15, 2 * se) and np.isfinite(slope) and 0.8 <= slope <= 1.25 and not contra
    return {"verdict": "pass" if ok else "fail", "h": h, "bias": bias, "bias_se": se, "slope": slope,
            "contradictions": len(contra), "pairs": len(pairs)}


def family_gap(jobs: List[dict], fam: str, protocol: str = REPLICATION_PROTOCOL) -> dict:
    """The replication statistic of one battery family in one seed (module docstring), with its validation."""
    B = _B()
    fj = [j for j in jobs if j["family"] == fam]
    val = B.validate(jobs, fam)
    X = B.primary_statistic(fam)
    extended = any(j["horizon"] == B.EXTENDED_H for j in fj)
    tj = [j for j in fj if j["horizon"] == B.EXTENDED_H] if extended else fj
    h = B.EXTENDED_H if extended else B.DEFAULT_H
    by = {}
    for c in B.candidates(fam):
        a = B.aggregate(B.units(tj, fam, protocol, c, X), h)
        if a is not None:
            by[c] = {"mean": a["mean"], "ci": a["ci"], "n_states": a["n"]}
    Y = "G" if X == "Q" else "Q"                          # the other statistic at 50 turns, reported beside it
    other = {}
    for c in B.candidates(fam):
        a = B.aggregate(B.units(fj, fam, protocol, c, Y), B.DEFAULT_H)
        if a is not None:
            other[c] = a["mean"]
    v = val.get("verdict", "")
    val150 = validate_at(fj, fam, B.EXTENDED_H) if v.startswith("fail") else None
    if v.startswith("fail") and not extended:
        status = "pending (horizon validation failed; the 150-turn re-run is not available)"
    elif not by:
        status = "no estimable alternative"
    else:
        status = "ok"
    best = max(by, key=lambda c: by[c]["mean"]) if by else None
    return {"statistic": f"{X}{h}", "protocol": protocol, "extended": extended, "validation": val,
            "validation_flag": "unvalidated (insufficient validation pairs)" if v == "insufficient" else v,
            "by_candidate": by, "best": best, "gap": by[best]["mean"] if best else float("nan"), "status": status,
            "validation_150_diagnostic": val150,
            "other_statistic": f"{Y}{B.DEFAULT_H}", "other_by_candidate": other,
            "other_gap": max(other.values()) if other else float("nan")}


def battery_seed(tag: str, root: Optional[str] = None) -> dict:
    B = _B()
    jobs = B.load(tag, root or battery_root())
    cov = battery_coverage(tag, jobs, root)
    fams = {}
    for fam in B.FAMILIES:
        g = family_gap(jobs, fam) if cov[fam]["records"] != "missing" else {"status": "missing"}
        if cov[fam]["records"] == "partial":
            g["status"] = "partial records"
        g["coverage"] = cov[fam]
        fams[fam] = g
    return {"tag": tag, "n_records": len(jobs), "families": fams}


def failed_families(tag: str, root: Optional[str] = None) -> List[str]:
    """The battery families of one final whose horizon validation failed (M4.5c's pass rule on its 300-turn subset)."""
    B = _B()
    jobs = B.load(tag, root or battery_root())
    return [f for f in B.FAMILIES if B.validate(jobs, f).get("verdict", "").startswith("fail")]


def extension_jobs(spec: str, tag: str, failed: Sequence[str], natural: Dict[str, list],
                   root: Optional[str] = None, k_nat: Optional[int] = None, k_con: Optional[int] = None) -> List[tuple]:
    """The registered handling of failed horizon validation (M4.5c exception 3): `battery.state_job` tuples for every
    state and protocol of each failed family at EXTENDED_H. `natural` holds the failed natural families' states,
    re-collected with `battery.collect_job` (deterministic: the same seeds give the same states). The 300-turn
    validation records are kept; the re-run writes its own files (job paths with the __h150 suffix)."""
    B = _B()
    from . import battery as MB
    out = []
    for j in MB.jobs(spec, tag, natural, list(failed), k_nat=k_nat, k_con=k_con, root=root or battery_root(),
                     validation=False):
        out.append(j[:7] + (B.EXTENDED_H,) + j[8:])
    return out


# ---------------------------------------------------------------------------------------------- cells
def _load_dir(family: str, tag: str, root: Optional[str] = None) -> Dict[int, dict]:
    d = os.path.join(root or cells_root(), family, tag)
    out = {}
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".json.gz"):
                out[int(f.split(".")[0])] = C.load_gz(os.path.join(d, f))
    return out


def _g(rec: dict, cand: str, h: int) -> np.ndarray:
    return np.asarray(rec["G"][cand][str(h)], dtype=float)


def weak_cells():
    from ..m46c import evaluate as EV
    return dict(EV.battery_cells())


def weak_gap(recs: Dict[int, dict]) -> dict:
    """M4.6c §5.1's statistic for one final: per lead size, the best forced weak-side response minus the policy at G50
    over every weak-focal cell; the gap is the larger of the two lead sizes."""
    from ..m46c.evaluate import WEAK_CANDS
    cells = weak_cells()
    per_d = {}
    for d in (8, 16):
        idx = [i for i, c in cells.items() if c[0] == "weak" and c[3] == d]
        if not all(i in recs for i in idx):
            return {"status": "partial records", "missing": sorted(set(idx) - set(recs))}
        by = {c: float(np.mean([np.mean(_g(recs[i], c, 50) - _g(recs[i], "policy", 50)) for i in idx]))
              for c in WEAK_CANDS if c != "policy"}
        best = max(by, key=by.get)
        per_d[f"+{d}"] = {"by_candidate": by, "best": best, "best_minus_policy": by[best]}
    k = max(per_d, key=lambda x: per_d[x]["best_minus_policy"])
    return {"status": "ok", "per_lead": per_d, "gap": per_d[k]["best_minus_policy"], "statistic": "G50",
            "worst_lead": k}


def strong_readout(recs: Dict[int, dict]) -> dict:
    """Descriptive: the strong focal's forced alternatives (threaten, hold) minus the policy at G50."""
    from ..m46c.evaluate import STRONG_CANDS
    cells = weak_cells()
    out = {}
    for d in (8, 16):
        idx = [i for i, c in cells.items() if c[0] == "strong" and c[3] == d and i in recs]
        if idx:
            out[f"+{d}"] = {c: float(np.mean([np.mean(_g(recs[i], c, 50) - _g(recs[i], "policy", 50)) for i in idx]))
                            for c in STRONG_CANDS if c != "policy"}
    return out


def premium(recs: Dict[int, dict]) -> dict:
    """M4.6c J1: the lead premium of the final in its own game, per stage, lead and horizon."""
    from ..m46c.analysis import premium_of
    from ..m46c.archive import STAGES, lead_cells
    need = [i for i, _ in lead_cells()]
    if not all(i in recs for i in need):
        return {"status": "partial records", "missing": len(set(need) - set(recs))}
    return {"status": "ok", **{f"{st}|+{d}|G{h}": premium_of(recs, "self", st, d, h) for st in STAGES
                               for d in (8, 16) for h in (20, 50)}}


# ---------------------------------------------------------------------------------------------- cross-arm families
def expected_items(family: str) -> Optional[int]:
    """The number of cells per (seed, seat-B opponent) of a cross-arm family (the builders' designs)."""
    from ..m46f import behaviour as BH
    cfg = C.arm_cfg("sp")
    if family == "conv":
        return len(BH.conv_cells(cfg))
    if family == "freeze":
        return len(BH.freeze_cells(cfg))
    if family == "recall":
        return len(BH.recall_cells(cfg))
    if family == "gaps":
        from ..m46a import e4 as E4
        return len(E4.GAP_CELLS) + len(E4.EQUAL_CELLS)
    if family == "natural":
        return 24 + 24 + 2 * 16                           # leader, trailer, freezer and breaker (cells.natural_jobs)
    raise ValueError(family)


def cross_rows(family: str, seeds: Sequence[int], root: Optional[str] = None) -> dict:
    """{seed: {"rows": [...], "missing": n}}; a row = one cell against one seat-B arm: meta, per-candidate mean G."""
    n = expected_items(family)
    out = {}
    for s in seeds:
        rows, missing = [], 0
        for opp in C.ARMS:
            recs = _load_dir(family, f"s{s}_vs_{opp}", root)
            missing += sum(1 for i in range(n) if i not in recs)
            for i, rec in recs.items():
                row = {"idx": i, "opp": opp, "meta": rec.get("meta") or {}, "label": rec.get("label", "")}
                row["G"] = {c: {int(h): float(np.mean(v)) for h, v in g.items()} for c, g in rec["G"].items()}
                rows.append(row)
        out[s] = {"rows": rows, "missing": missing, "expected": 2 * n}
    return out


def _group_of(family: str, meta: dict, label: str) -> List[str]:
    from ..m46f import behaviour as BH
    if family == "conv":
        c = meta.get("cond")
        return ["conversion" if c in BH.CONV_CONDS else "defence" if c in BH.DEF_CONDS else
                "equal" if c == "equal" else f"other ({c})"]
    if family == "freeze":
        g = [meta.get("role", "?")]
        if meta.get("cond") in ("edge", "plaus", "hp"):
            g.append(f"{meta.get('role')} favourable (edge, plaus, hp)")
        return g
    if family == "recall":
        role = meta.get("role", "?")
        g = [role]
        if meta.get("cond") != "equal":
            g.append(f"{role} vs a stronger opponent" if role == "weak" else f"{role}, advantaged")
        return g
    if family == "natural":
        return [meta.get("role", label.split(" ")[1] if " " in label else "?")]
    if family == "gaps":
        cell = meta.get("cell") or [None, None, None]
        return ["equal cells" if cell[2] == "equal" else "gap cells", f"gap {cell[2]}"]
    raise ValueError(family)


HORIZONS = {"conv": (50, 150), "freeze": (50,), "recall": (50, 100), "natural": (50, 150), "gaps": (50, 150)}


def cross_arm(family: str, seeds: Sequence[int] = C.SEEDS, root: Optional[str] = None) -> dict:
    """om - sp (both finals in seat A on the same rollouts; averaged over the two seat-B arms) by cell group and
    horizon, per seed and over the seeds; the within-arm readings (each candidate against its own arm); for the large
    gaps the specialist's margin over each arm. Complete only when every seed has every record."""
    data = cross_rows(family, seeds, root)
    complete = [s for s in seeds if data[s]["missing"] == 0 and data[s]["rows"]]
    per: Dict[str, Dict[int, float]] = defaultdict(dict)
    for s in seeds:
        rows = data[s]["rows"]
        if not rows:
            continue
        for h in HORIZONS[family]:
            acc: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
            own: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
            for r in rows:
                if h not in r["G"].get("om", {}):
                    continue
                for grp in _group_of(family, r["meta"], r["label"]) + ["all"]:
                    acc[f"om - sp | {grp} | G{h}"][r["idx"]].append(r["G"]["om"][h] - r["G"]["sp"][h])
                    own[f"{r['opp']} in its own game | {grp} | G{h}"][r["idx"]].append(r["G"][r["opp"]][h]
                                                                                      if r["opp"] in r["G"] else np.nan)
                    if "gap_specialist" in r["G"]:
                        acc[f"specialist - {r['opp']} | {grp} | G{h}"][r["idx"]].append(
                            r["G"]["gap_specialist"][h] - r["G"][r["opp"]][h])
            for k, d in list(acc.items()) + list(own.items()):
                per[k][s] = float(np.mean([np.mean(v) for v in d.values()]))
    return {"family": family, "complete_seeds": complete,
            "status": "complete" if len(complete) == len(seeds) else f"partial ({len(complete)}/{len(seeds)})",
            "missing_records": {str(s): data[s]["missing"] for s in seeds if data[s]["missing"]},
            "stats": {k: {**AN.interval(v), "n_seeds": len(v)} for k, v in sorted(per.items())}}


# ---------------------------------------------------------------------------------------------- assembly
def replication_tables(seeds: Sequence[int] = C.SEEDS, broot: Optional[str] = None,
                       croot: Optional[str] = None) -> dict:
    """Per arm and verdict: the per-seed gap (only seeds whose every component is complete and not pending), the
    replication verdict (`analysis.replication`, partial unless decided), and every component's status."""
    out = {}
    seeds_data = {}
    for a in C.ARMS:
        for s in seeds:
            seeds_data[(a, s)] = {"battery": battery_seed(f"{a}_s{s}", broot),
                                  "weak": weak_gap(_load_dir("weak", f"{a}_s{s}", croot))}
    for a in C.ARMS:
        arm = {}
        for verdict, comps in VERDICTS.items():
            gaps, other, status = {}, {}, {}
            for s in seeds:
                vals, ovals, st = [], [], []
                for kind, name in comps:
                    rec = (seeds_data[(a, s)]["battery"]["families"][name] if kind == "battery"
                           else seeds_data[(a, s)]["weak"])
                    st.append(f"{name}: {rec.get('status')}")
                    if rec.get("status") == "ok":
                        vals.append(rec["gap"])
                        ovals.append(rec.get("other_gap", rec["gap"]))
                status[str(s)] = st
                if len(vals) == len(comps):
                    gaps[s] = float(max(vals))
                    other[s] = float(max(ovals))
            rep = AN.replication(gaps, expected=len(seeds))
            arm[verdict] = {"per_seed_gap": {str(k): v for k, v in gaps.items()}, "replication": rep,
                            "per_seed_other_gap": {str(k): v for k, v in other.items()},
                            "other_rule_reading": AN.replication(other, expected=len(seeds))["verdict"],
                            "component_status": status, "components": [n for _, n in comps]}
        out[a] = arm
    detail = {f"{a}_s{s}": {"battery": {f: {k: v for k, v in r.items() if k != "validation"}
                                        | {"validation": r.get("validation", {}).get("verdict")}
                                        for f, r in d["battery"]["families"].items()},
                            "weak": d["weak"]}
              for (a, s), d in seeds_data.items()}
    return {"replication": out, "detail": detail}


def family_summary(detail: Dict[str, dict], seeds: Sequence[int] = C.SEEDS) -> dict:
    """Per arm and battery family, protocol "policy": every alternative's margin over the policy pooled over the seeds
    (seed mean, t interval), the best alternative by that mean, how many seeds have the family's value (not pending),
    the statistics the seeds used (Q50 / G50, or Q150 / G150 after a re-run), and the validation counts."""
    B = _B()
    out: Dict[str, dict] = {}
    for a in C.ARMS:
        for fam in B.FAMILIES:
            recs = {s: detail[f"{a}_s{s}"]["battery"].get(fam, {}) for s in seeds if f"{a}_s{s}" in detail}
            ok = {s: r for s, r in recs.items() if r.get("status") == "ok"}
            cands: Dict[str, Dict[int, float]] = defaultdict(dict)
            for s, r in ok.items():
                for c, v in (r.get("by_candidate") or {}).items():
                    cands[c][s] = v["mean"]
            stats = {c: {**AN.interval(v), "n_seeds": len(v)} for c, v in cands.items()}
            best = max(stats, key=lambda c: stats[c]["mean"]) if stats else None
            out[f"{a}|{fam}"] = {"arm": a, "family": fam, "n_seeds": len(ok), "best": best,
                                 "best_stat": stats.get(best) if best else None, "candidates": stats,
                                 "statistics": sorted({r.get("statistic") for r in ok.values()}),
                                 "validation_failed": sum(1 for r in recs.values()
                                                          if str(r.get("validation", "")).startswith("fail")),
                                 "extended": sum(1 for r in ok.values() if r.get("extended"))}
    return out


def per_arm_cells(seeds: Sequence[int] = C.SEEDS, croot: Optional[str] = None) -> dict:
    """Descriptive per-arm cells: the weaker-side readouts and the lead premium, per arm and om - sp."""
    out: Dict[str, dict] = {}
    prem: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    weak: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for a in C.ARMS:
        for s in seeds:
            p = premium(_load_dir("premium", f"{a}_s{s}", croot))
            if p.get("status") == "ok":
                for k, v in p.items():
                    if k != "status":
                        prem[k][a][s] = v
            recs = _load_dir("weak", f"{a}_s{s}", croot)
            w = weak_gap(recs)
            if w.get("status") == "ok":
                for d, r in w["per_lead"].items():
                    weak[f"weak side {d}: best forced - policy (G50)"][a][s] = r["best_minus_policy"]
                for d, r in strong_readout(recs).items():
                    for c, v in r.items():
                        weak[f"strong side {d}: {c} - policy (G50)"][a][s] = v
    for name, src in (("premium", prem), ("weak", weak)):
        tab = {}
        for k, d in src.items():
            row = {a: {**AN.interval(v), "n_seeds": len(v)} for a, v in d.items()}
            both = sorted(set(d.get("om", {})) & set(d.get("sp", {})))
            if both:
                row["om - sp"] = {**AN.interval({s: d["om"][s] - d["sp"][s] for s in both}), "n_seeds": len(both)}
            tab[k] = row
        out[name] = tab
    return out


def battery_tables(seeds: Sequence[int] = C.SEEDS, broot: Optional[str] = None, out: str = C.OUT) -> Dict[str, str]:
    """M4.5c's battery report per arm (`m45c.battery.report`: every family, protocol and candidate, pooled over the
    six finals with each final's value), written to results/m5/battery/battery_<arm>.md, once all six finals of the arm
    have records."""
    B = _B()
    root = broot or battery_root()
    made = {}
    for a in C.ARMS:
        tags = [f"{a}_s{s}" for s in seeds]
        if all(os.path.isdir(os.path.join(root, t, "partial")) for t in tags):
            B.report(tags, root=root, name=f"battery_{a}", out=os.path.join(out, "battery"))
            made[a] = os.path.join(out, "battery", f"battery_{a}.md").replace("\\", "/")
    return made


def assemble(seeds: Sequence[int] = C.SEEDS, broot: Optional[str] = None, croot: Optional[str] = None,
             out: str = C.OUT, save: bool = True) -> dict:
    """S8 in full from whatever battery and cell records exist (status explicit everywhere)."""
    rep = replication_tables(seeds, broot, croot)
    res = {"replication": rep, "family_summary": family_summary(rep["detail"], seeds),
           "per_arm_cells": per_arm_cells(seeds, croot),
           "battery_tables": battery_tables(seeds, broot, out) if save else {},
           "cross_arm": {f: cross_arm(f, seeds, croot) for f in CROSS},
           "rule": {"margin": AN.REPLICATION_MARGIN, "min_seeds": AN.REPLICATION_MIN_SEEDS,
                    "protocol": REPLICATION_PROTOCOL, "statistic": "Q50 natural / G50 constructed (Q150 / G150 after "
                    "a failed validation and its re-run); weaker side G50"},
           "meta": C.meta(stage="strategic")}
    if save:
        C.save_json("strategic_main", res, out)
    return res


__all__ = ["VERDICTS", "REPLICATION_PROTOCOL", "CROSS", "battery_coverage", "family_gap", "battery_seed",
           "failed_families",
           "extension_jobs", "weak_gap", "premium", "cross_arm", "replication_tables", "per_arm_cells", "assemble",
           "expected_states", "expected_items"]
