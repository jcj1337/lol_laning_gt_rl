"""The M5b-M5d evaluation designs (pre-registration §5-§8) as job lists, the gated stage drivers, and `plan()` (the
number of simulated turns of each part, by cost class, for the budget).

Policies: "om:s" / "sp:s" the finals (32.768M), "<arm>:s@16384000" the matched midpoints, "<arm>:s@<n>" the
monitoring snapshots, "start:s" = ext/baseline:s (the common start; an exploratory reference, never an M5 arm).

M5b monitoring (never used to select checkpoints or to change an endpoint):
    curves   every run's snapshots at CURVE_POINTS against the fixed panel REF_PANEL, CURVE_N pairs x 150 turns
    history  each final against its own 25% / 50% / 75% snapshots and against start:s, HIST_N pairs x 500 turns
             (the convergence probe: final vs the 75% snapshot)
M5c strategic evaluation:
    battery  the M4.5c battery (`battery.run_policy`) on every final, its full design
    cells    the M4 limitation families (`cells`): weak and premium per final; conv, freeze, recall, natural, gaps
             cross-arm per seed
M5d opponent modelling and robustness (games from s0, `games.cell_job`; focals om:s, sp:s and the reference start:s
on the same games):
    S_in     the 6 OM2 library archetypes                                        FAM_N pairs x 150
    S_out    4 held-out scripted variants + 3 conditional styles on a competent base ext/baseline:o, o = s + 3 mod 6
    M        2 held-out mixtures, 2 held-out switchers, 3 imitate-then-switch opponents (a library archetype, then
             ext/baseline:o)
    C_hist   ext/baseline:o for the five o != s, and the 5 held-out learned instances (OM2's R1 set)
    C_cross  the other seeds' finals of both arms (10)
    long     C_hist, C_cross and the same-seed head to head at 500 turns (the objective's discounted lead), LONG_N pairs
    H2H      om:s vs sp:s (H2H_N pairs x 150, and LONG_N x 500), om:s vs sp:o for o != s (XSEED_N x 150)
    mid      the midpoints on S_in, S_out, C_hist and the same-seed head to head, MID_N pairs x 150
Attackers: `attackers` (training) and their evaluation cells; the core stage is 12 attackers (om, sp), the om_table
ablation an optional follow-up (amendment 1).

Gates (amendment 1). Every evaluation stage first requires a passed (or fully resolved) training check whose recorded
checkpoint identities match the files (`health.require_training_ok`), then writes a stage manifest with the identities of
every network it reads (results/m5/eval_<stage>.json). Cell files are resumable; before an existing manifest's partial
outputs are reused, the identities and the active pre-registration version must match it.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import common as C

EOUT = os.path.join(C.CKPT, "eval")
FAM_T, FAM_N = 150, 64
LONG_T, LONG_N = 500, 32
H2H_N, XSEED_N, MID_N = 128, 32, 32
CURVE_N, HIST_N = 16, 64
STYLES = ("respect", "passive", "stepback")
ITS = (("g_farm", 40), ("g_safe", 20), ("g_shove", 40))


def other(s: int) -> int:
    return (s + 3) % 6


def families(s: int) -> Dict[str, list]:
    """The M5d opponent families of seed s, as seat specs (pre-registration §6.1)."""
    from ..om2 import common as O2
    o = other(s)
    return {
        "S_in": [("opp", n) for n in O2.ARCHETYPES],
        "S_out": [("opp", n) for n in O2.HELDOUT_SCRIPTED] + [("opp", f"style:{x}:{o}") for x in STYLES],
        "M": [("opp", n) for n in list(O2.HELDOUT_MIXTURES) + list(O2.HELDOUT_SWITCHERS)]
             + [("opp", f"its:{x}:{t}:{o}") for x, t in ITS],
        "C_hist": [("opp", f"base:{p}") for p in C.SEEDS if p != s] + [("opp", n) for n in O2.HELDOUT_LEARNED],
        "C_cross": [("net", f"{a}:{p}") for p in C.SEEDS if p != s for a in C.ARMS],
    }


REF_PANEL = ("start", "base", "g_farm", "g_shove", "g_bully")


def _cell(key, path, focals, opp, N, T, root, cell) -> dict:
    return {"key": key, "path": path, "focals": [list(f) for f in focals], "opponent": list(opp), "N": int(N),
            "T": int(T), "root": int(root), "cell": int(cell)}


def _opp_tag(opp) -> str:
    return "__".join(str(x) for x in opp).replace(":", "_").replace("/", "_")


def family_jobs(seeds=C.SEEDS, phase: str = "main") -> List[dict]:
    pre = "" if phase == "main" else f"{phase}/"
    js = []
    for s in seeds:
        focals = [("net", f"{pre}om:{s}"), ("net", f"{pre}sp:{s}"), ("net", f"start:{s}")]
        cell = 0
        fam = families(s)
        for name, opps in fam.items():
            for opp in opps:
                js.append(_cell(f"fam|s{s}|{name}|{_opp_tag(opp)}",
                                os.path.join(EOUT, "families", f"s{s}", f"{name}__{_opp_tag(opp)}.npz"),
                                focals, opp, FAM_N, FAM_T, C.FAMILY_ROOT, 1000 * s + cell))
                cell += 1
        for name in ("C_hist", "C_cross"):
            for opp in fam[name]:
                js.append(_cell(f"long|s{s}|{name}|{_opp_tag(opp)}",
                                os.path.join(EOUT, "long", f"s{s}", f"{name}__{_opp_tag(opp)}.npz"),
                                focals, opp, LONG_N, LONG_T, C.FAMILY_ROOT, 1000 * s + cell))
                cell += 1
        js.append(_cell(f"h2h|s{s}|same", os.path.join(EOUT, "h2h", f"s{s}", "om_vs_sp.npz"),
                        [("net", f"{pre}om:{s}")], ("net", f"{pre}sp:{s}"), H2H_N, FAM_T, C.H2H_ROOT, 1000 * s))
        js.append(_cell(f"h2h|s{s}|same_long", os.path.join(EOUT, "h2h", f"s{s}", "om_vs_sp_long.npz"),
                        [("net", f"{pre}om:{s}")], ("net", f"{pre}sp:{s}"), LONG_N, LONG_T, C.H2H_ROOT, 1000 * s + 1))
        for p in C.SEEDS:
            if p != s:
                js.append(_cell(f"h2h|s{s}|x{p}", os.path.join(EOUT, "h2h", f"s{s}", f"om_vs_sp{p}.npz"),
                                [("net", f"{pre}om:{s}")], ("net", f"{pre}sp:{p}"), XSEED_N, FAM_T, C.H2H_ROOT,
                                1000 * s + 10 + p))
        mid = [("net", f"{pre}om:{s}@{C.MIDPOINT}"), ("net", f"{pre}sp:{s}@{C.MIDPOINT}")]
        for name in ("S_in", "S_out", "C_hist"):
            for opp in fam[name]:
                js.append(_cell(f"mid|s{s}|{name}|{_opp_tag(opp)}",
                                os.path.join(EOUT, "mid", f"s{s}", f"{name}__{_opp_tag(opp)}.npz"),
                                mid, opp, MID_N, FAM_T, C.FAMILY_ROOT, 1000 * s + cell))
                cell += 1
        js.append(_cell(f"mid|s{s}|h2h", os.path.join(EOUT, "mid", f"s{s}", "h2h.npz"), mid[:1], mid[1], MID_N,
                        FAM_T, C.H2H_ROOT, 1000 * s + 2))
    return js


def monitor_jobs(seeds=C.SEEDS) -> List[dict]:
    js = []
    for s in seeds:
        panel = [("net", f"start:{s}"), ("opp", f"base:{other(s)}"), ("opp", "g_farm"), ("opp", "g_shove"),
                 ("opp", "g_bully")]
        for a in C.ARMS:
            for c, n in enumerate(C.CURVE_POINTS):
                spec = f"{a}:{s}" if n == C.BUDGET else f"{a}:{s}@{n}"
                for q, opp in enumerate(panel):
                    js.append(_cell(f"curve|{a}|s{s}|{n}|{q}", os.path.join(EOUT, "curve", f"{a}_s{s}", f"{n}_{q}.npz"),
                                    [("net", spec)], opp, CURVE_N, FAM_T, C.CURVE_ROOT, 100 * s + q))
            for q, n in enumerate((C.BUDGET // 4, C.MIDPOINT, C.P75)):
                js.append(_cell(f"hist|{a}|s{s}|{n}", os.path.join(EOUT, "history", f"{a}_s{s}", f"vs_{n}.npz"),
                                [("net", f"{a}:{s}")], ("net", f"{a}:{s}@{n}"), HIST_N, LONG_T, C.HIST_ROOT,
                                100 * s + q))
            js.append(_cell(f"hist|{a}|s{s}|start", os.path.join(EOUT, "history", f"{a}_s{s}", "vs_start.npz"),
                            [("net", f"{a}:{s}")], ("net", f"start:{s}"), HIST_N, LONG_T, C.HIST_ROOT, 100 * s + 9))
    return js


def _kind_of(spec_list) -> str:
    """The cost class of a game: how many om-format seats it has (each needs a belief update per turn)."""
    from .common import arm_of
    n = 0
    for sp in spec_list:
        sp = tuple(sp)
        if sp[0] == "net" and (arm_of(sp[1]) == "om"):
            n += 1
    return f"game_om{n}"


def _game_turns(js: List[dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for j in js:
        for f in j["focals"]:
            k = _kind_of([f, j["opponent"]])
            out[k] = out.get(k, 0.0) + 2 * j["N"] * j["T"]
    return out


def battery_turns(arm: str) -> float:
    """Simulated turns of the full M4.5c battery design for one final (job enumeration with the design's sizes)."""
    from ..m45c import battery as B
    tot = 0.0
    for fam, (kind, _, _, protos) in B.FAMILIES.items():
        n = len(B.GRIDS[fam]) if kind == "constructed" else (B.N_EARLY if kind == "early" else B.N_NATURAL)
        K = B.K_CONSTRUCTED if kind == "constructed" else B.K_NATURAL
        ncand = 1 + len(B.candidates(fam))
        h = B.family_horizon(fam)
        nval = n if kind == "constructed" else len(B.validation_indices(fam, n))
        for p in protos:
            tot += n * K * ncand * h
            if p == "policy":
                tot += nval * K * ncand * (B.VALIDATION_H - h)
    return tot


def cells_turns() -> Dict[str, float]:
    """Simulated turns of the M5c cells design, by cost class (job enumeration on dummy specs; no checkpoint read)."""
    from . import cells as CE
    from ..m46f import behaviour as BH
    out: Dict[str, float] = {}
    fin = {f"{a}_s0": f"{a}:0" for a in C.ARMS}

    def add(js, kind, mult=1.0):
        out[kind] = out.get(kind, 0.0) + mult * sum(len(j["roll"]) * len(j["cands"]) * j["horizon"] for j in js)

    for a in C.ARMS:
        add(CE.weak_jobs({f"{a}_s0": f"{a}:0"}), f"cells_{a}", 6)
        add(CE.premium_jobs({f"{a}_s0": f"{a}:0"}), f"cells_{a}", 6)
    specs = CE.arm_specs(0)
    for fn in (CE.conv_jobs, CE.freeze_jobs, CE.recall_jobs):
        add(fn(0, specs), "cells_cross", 6)
    add(CE.gaps_jobs(0, specs), "cells_cross", 6)
    # natural: leader, trailer x 24 (H_NAT), freezer, breaker x 16 (H_NATFZ); 2 candidates x 2 opponents x K_NAT
    out["cells_cross"] += 6 * 2 * 2 * BH.K_NAT * (48 * BH.H_NAT + 32 * BH.H_NATFZ)
    return out


def plan(include_optional: bool = False) -> Dict[str, Dict[str, float]]:
    """Simulated turns per evaluation part and cost class (budget). The attacker evaluation counts the core attackers
    (12) unless include_optional (the om_table ablation and its cross-evaluation added)."""
    from .attackers import CORE_KINDS, OPTIONAL_KINDS, eval_jobs
    fam = _game_turns(family_jobs())
    mon = _game_turns(monitor_jobs())
    kinds = CORE_KINDS + (OPTIONAL_KINDS if include_optional else ())
    att = _game_turns([{**j, "focals": [("net", "om:0")], "opponent": j["opponent"]} for j in eval_jobs(kinds)])
    return {"M5b_monitor": mon, "M5c_battery": {"battery_om": 6 * battery_turns("om"), "battery_sp": 6 * battery_turns("sp")},
            "M5c_cells": cells_turns(), "M5d_families": fam, "M5d_attack_eval": att}


def battery_breakdown() -> List[dict]:
    """The M4.5c battery design for ONE final, by family, horizon and purpose (simulated turns): the standard
    continuation (the family horizon: 50, or exception 1's plan horizon), the validation subset's extension from the
    family horizon to 300 turns (the "policy" jobs of the subset run 300 turns instead of h), and, contingent on a failed
    validation, exception 3's re-run of the whole family at 150 turns."""
    from ..m45c import battery as B
    rows = []
    for fam, (kind, _, _, protos) in B.FAMILIES.items():
        n = len(B.GRIDS[fam]) if kind == "constructed" else (B.N_EARLY if kind == "early" else B.N_NATURAL)
        K = B.K_CONSTRUCTED if kind == "constructed" else B.K_NATURAL
        ncand = 1 + len(B.candidates(fam))
        h = B.family_horizon(fam)
        nval = n if kind == "constructed" else len(B.validation_indices(fam, n))
        std = n * K * ncand * h * len(protos)
        val = nval * K * ncand * (B.VALIDATION_H - h) if "policy" in protos else 0
        ext = n * K * ncand * B.EXTENDED_H * len(protos)
        rows.append({"family": fam, "kind": kind, "states": n, "K": K, "candidates": ncand, "protocols": len(protos),
                     "horizon": h, "purpose": "standard (50)" if h == B.DEFAULT_H else
                     f"exception 1: sustained plans ({h})", "standard_turns": std, "validation_states": nval,
                     "validation_turns_beyond_h": val, "contingent_extension_turns_150": ext})
    return rows


def cells_breakdown() -> List[dict]:
    """The M5c limitation cells for all six seeds by family and horizon (simulated turns; no checkpoint read)."""
    from . import cells as CE
    from ..m46f import behaviour as BH
    rows = []

    def add(name, js, kind, mult, purpose):
        by_h: Dict[int, float] = {}
        for j in js:
            by_h[j["horizon"]] = by_h.get(j["horizon"], 0.0) + mult * len(j["roll"]) * len(j["cands"]) * j["horizon"]
        for hh, t in sorted(by_h.items()):
            rows.append({"family": name, "cost_class": kind, "horizon": hh, "turns": t, "purpose": purpose})
    for a in C.ARMS:
        add(f"weak ({a})", CE.weak_jobs({f"{a}_s0": f"{a}:0"}), f"cells_{a}", 6, "M4.6c weaker side, G50")
        add(f"premium ({a})", CE.premium_jobs({f"{a}_s0": f"{a}:0"}), f"cells_{a}", 6, "M4.6c lead premium J1, G20/G50")
    specs = CE.arm_specs(0)
    add("conv", CE.conv_jobs(0, specs), "cells_cross", 6, "M4.6f B2 strength use (G150 registered)")
    add("freeze", CE.freeze_jobs(0, specs), "cells_cross", 6, "M4.6f B3 freezes, G50")
    add("recall", CE.recall_jobs(0, specs), "cells_cross", 6, "M4.6f B4 Recall windows (G100 registered)")
    add("gaps", CE.gaps_jobs(0, specs), "cells_cross", 6, "M4.6a E4 large gaps (G150 registered)")
    for role, n, hh in (("natural leader/trailer", 48, BH.H_NAT), ("natural freezer/breaker", 32, BH.H_NATFZ)):
        rows.append({"family": role, "cost_class": "cells_cross", "horizon": hh,
                     "turns": 6 * 2 * 2 * BH.K_NAT * n * hh, "purpose": "M4.6f B5 natural states of the common start"})
    return rows


def horizon_breakdown(workers: int = 6) -> dict:
    """Every evaluation part's simulated turns by horizon and purpose, with worker and wall hours at the R10 rates
    (readiness record): the input of the future battery proposal (amendment 1, design §10)."""
    rates = (((C.load("readiness") or {}).get("parts", {}).get("R10") or {}).get("data") or {}).get("rates", {})
    rows = []
    for r in battery_breakdown():
        for arm in C.ARMS:
            rt = rates.get(f"battery_{arm}", float("nan"))
            for purpose, key, hh in ((r["purpose"], "standard_turns", r["horizon"]),
                                     ("validation subset to 300 turns", "validation_turns_beyond_h", 300),
                                     ("CONTINGENT: exception 3 re-run at 150 (only a failed family)",
                                      "contingent_extension_turns_150", 150)):
                t = 6 * r[key]
                if t:
                    rows.append({"part": "M5c battery", "arm": arm, "family": r["family"], "horizon": hh,
                                 "purpose": purpose, "turns": t, "worker_h": t / rt / 3600 if rt == rt else None})
    for r in cells_breakdown():
        rt = rates.get(r["cost_class"], float("nan"))
        rows.append({"part": "M5c cells", "arm": r["cost_class"].split("_")[1], "family": r["family"],
                     "horizon": r["horizon"], "purpose": r["purpose"], "turns": r["turns"],
                     "worker_h": r["turns"] / rt / 3600 if rt == rt else None})
    for part, js in (("M5b monitor", monitor_jobs()), ("M5d families", family_jobs())):
        agg: Dict[tuple, float] = {}
        for j in js:
            for f in j["focals"]:
                k = (j["key"].split("|")[0], j["T"], _kind_of([f, j["opponent"]]))
                agg[k] = agg.get(k, 0.0) + 2 * j["N"] * j["T"]
        for (grp, T, kind), t in sorted(agg.items()):
            rt = rates.get(kind, float("nan"))
            rows.append({"part": part, "arm": kind, "family": grp, "horizon": T,
                         "purpose": "whole game from s0 (lead at 114; the 500-turn games add the discounted objective)",
                         "turns": t, "worker_h": t / rt / 3600 if rt == rt else None})
    tot: Dict[str, float] = {}
    for r in rows:
        if r["worker_h"] is not None and not r["purpose"].startswith("CONTINGENT"):
            tot[r["part"]] = tot.get(r["part"], 0.0) + r["worker_h"]
    return {"rows": rows, "rates": rates, "workers": workers,
            "wall_h_by_part": {k: v / workers for k, v in tot.items()},
            "note": "wall hours = worker hours / workers (the R10 steady-state rates with six classes at once); "
                    "contingent rows are excluded from the totals"}


# ---------------------------------------------------------------------------------------------- stage drivers
def _net_specs(js: List[dict]) -> List[str]:
    out = set()
    for j in js:
        for f in list(j["focals"]) + [j["opponent"]]:
            if tuple(f)[0] == "net":
                out.add(tuple(f)[1])
    return sorted(out)


def stage_manifest(stage: str, js: Optional[List[dict]] = None, check_reuse: bool = True,
                   specs: Optional[List[str]] = None, existing: Optional[int] = None) -> dict:
    """The identities of every network a stage reads (and the active pre-registration version); refuses to reuse the
    partial outputs of a previous run of the stage made with other checkpoints or another protocol version. `js`:
    game-cell jobs; or `specs` (policy specifications) and `existing` (the number of outputs already on disk)."""
    pre = C.require_prereg()
    ids = {}
    for sp in (specs if specs is not None else _net_specs(js or [])):
        p = C.policy_path(sp)
        ids[sp] = {"path": p.replace("\\", "/"), "file_sha256": C.file_sha256(p) if os.path.exists(p) else None}
    man = {"meta": C.meta(stage=f"eval_{stage}"), "stage": stage, "jobs": len(js) if js is not None else None,
           "identities": ids, "preregistration_active": C.active_version(pre), "code": C.code_digest(),
           "existing_outputs": existing if existing is not None else
           sum(1 for j in (js or []) if os.path.exists(j["path"]))}
    old = C.load(f"eval_{stage}")
    if check_reuse and old and man["existing_outputs"]:
        bad = [k for k, v in ids.items() if (old.get("identities") or {}).get(k, {}).get("file_sha256") !=
               v["file_sha256"]]
        if bad or (old.get("preregistration_active") or {}).get("sha256") != man["preregistration_active"]["sha256"]:
            raise SystemExit(f"stage {stage}: existing outputs were made with other checkpoints or another protocol "
                             f"version ({bad[:5] or 'pre-registration'}); inspect them before rerunning")
    man["previous"] = (old or {}).get("meta")
    C.save_json(f"eval_{stage}", man)
    return man


def _run_cells(js: List[dict], workers: int, label: str, stage: Optional[str] = None) -> dict:
    from . import games as G
    if stage:
        stage_manifest(stage, js)
    js = sorted(js, key=lambda j: -len(j["focals"]) * j["N"] * j["T"])
    t0 = time.perf_counter()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    res = C.run_jobs(G.cell_job, js, workers, label)
    cached = sum(1 for _, r in res if r.get("cached"))
    rec = {"jobs": len(js), "computed": len(js) - cached, "reused": cached, "workers": workers, "started": started,
           "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": time.perf_counter() - t0,
           "job_seconds": float(sum(r.get("elapsed", 0.0) for _, r in res))}
    C.jdump(os.path.join(C.OUT, f"wall_{label.replace(' ', '_')}.json"), rec)
    return rec


def _gate() -> None:
    from .health import require_training_ok
    require_training_ok("main")


def stage_monitor(workers: int) -> None:
    _gate()
    _run_cells(monitor_jobs(), workers, "M5b monitor", "monitor")


def stage_families(workers: int) -> None:
    _gate()
    _run_cells(family_jobs(), workers, "M5d families", "families")


def stage_battery(workers: int) -> None:
    _gate()
    from . import battery as MB
    import glob
    finals = [f"{a}:{s}" for s in C.SEEDS for a in C.ARMS]
    stage_manifest("battery", specs=finals,
                   existing=len(glob.glob(os.path.join(MB.BOUT, "*", "partial", "*.json.gz"))))
    for s in C.SEEDS:
        for a in C.ARMS:
            MB.run_policy(f"{a}:{s}", f"{a}_s{s}", workers)


def extension_plan(seeds=C.SEEDS) -> Dict[str, List[str]]:
    """{final tag: the battery families whose horizon validation failed} (M4.5c's exception 3 applies to them)."""
    from . import strategic as ST
    return {f"{a}_s{s}": ST.failed_families(f"{a}_s{s}") for s in seeds for a in C.ARMS}


def stage_battery_extend(workers: int) -> dict:
    """M4.5c's exception 3, part of the registered battery: every (final, family) whose horizon validation failed is
    re-run at 150 turns for all its states and protocols, decided by the pass rule alone. The failed natural families'
    states are re-collected in parallel (deterministic), then the re-run jobs run resumably."""
    _gate()
    from ..m45c import battery as B
    from . import battery as MB
    from . import strategic as ST
    plan = extension_plan()
    t0 = time.perf_counter()
    cjobs = [(f"{tag.split('_s')[0]}:{tag.split('_s')[1]}", f, ST.expected_states(f)) for tag, fams in plan.items()
             for f in fams if B.FAMILIES[f][0] != "constructed"]
    nat: Dict[str, Dict[str, list]] = {}
    for (spec, fam), v in C.run_jobs(MB.collect_job, cjobs, workers, "M5 battery extension states"):
        nat.setdefault(spec.replace(":", "_s"), {})[fam] = v["states"]
    js = []
    for tag, fams in plan.items():
        if fams:
            spec = f"{tag.split('_s')[0]}:{tag.split('_s')[1]}"
            js += ST.extension_jobs(spec, tag, fams, nat.get(tag, {}))
    t1 = time.perf_counter()
    res = dict(C.run_jobs(MB.state_job, js, workers, "M5 battery extension")) if js else {}
    rec = {"meta": C.meta(stage="battery_extend"), "plan": plan, "jobs": len(js),
           "collect_s": t1 - t0, "rollouts_s": time.perf_counter() - t1,
           "turns": int(sum(v.get("turns", 0) for v in res.values()))}
    C.save_json("battery_extension", rec)
    return rec


def stage_cells(workers: int) -> None:
    _gate()
    from . import cells as CE
    fin = {f"{a}_s{s}": f"{a}:{s}" for s in C.SEEDS for a in C.ARMS}
    js = CE.weak_jobs(fin) + CE.premium_jobs(fin)
    for s in C.SEEDS:
        sp = CE.arm_specs(s)
        js += CE.conv_jobs(s, sp) + CE.freeze_jobs(s, sp) + CE.recall_jobs(s, sp) + CE.gaps_jobs(s, sp)
        js += CE.natural_jobs(s, sp, CE.natural_states(s))
    specs = sorted(set(fin.values()) | {c[1] for j in js for c in j["cands"].values() if c[0] == "net"}
                   | {f"start:{s}" for s in C.SEEDS})
    stage_manifest("cells", specs=specs, existing=sum(1 for j in js if os.path.exists(j["path"])))
    CE.run(js, workers, "M5c cells")


def stage_attack_tables(workers: int, include_optional: bool = False) -> None:
    """The om finals' self-play belief tables: needed only by the OPTIONAL om_table ablation."""
    if not include_optional:
        raise SystemExit("the belief tables serve only the optional om_table ablation: pass --include-om-table")
    _gate()
    from . import attackers as AT
    os.makedirs(os.path.dirname(AT.table_path(0)), exist_ok=True)
    for s in C.SEEDS:
        p = AT.table_path(s)
        if not os.path.exists(p):
            np.save(p, AT.self_play_table(f"om:{s}"))


def stage_attackers(workers: int, include_optional: bool = False) -> None:
    _gate()
    from . import attackers as AT
    js = []
    for kind in AT.CORE_KINDS + (AT.OPTIONAL_KINDS if include_optional else ()):
        for s in C.SEEDS:
            d = AT.attacker_dir(kind, s)
            if os.path.exists(os.path.join(d, "summary.json")) and \
                    (C.jload(os.path.join(d, "summary.json")) or {}).get("status") == "complete":
                continue
            if os.path.isdir(d) and os.listdir(d):
                raise SystemExit(f"{d} exists and is incomplete: inspect it (attacker runs are never overwritten)")
            js.append((kind, s, AT.ATTACK_BUDGET, None, "main", False))
    res = C.run_jobs(AT.train_job, js, workers, "M5d attackers")
    C.save_json("attackers_train", {"meta": C.meta(stage="attackers"), "runs": res})


def stage_attack_eval(workers: int, include_optional: bool = False) -> None:
    _gate()
    from . import attackers as AT
    kinds = AT.CORE_KINDS + (AT.OPTIONAL_KINDS if include_optional else ())
    _run_cells(AT.eval_jobs(kinds), workers, "M5d attack eval", "attack_eval")


def stage_analyse(workers: int) -> None:
    """Every analysis from whatever records exist (statuses explicit): M5d endpoints, monitoring, S8."""
    from . import analysis as AN
    from . import strategic as ST
    AN.analyse()
    AN.monitoring()
    ST.assemble()


def stage_report(workers: int) -> None:
    from . import report as RP
    RP.report()


__all__ = ["families", "family_jobs", "monitor_jobs", "plan", "battery_turns", "cells_turns", "other", "REF_PANEL",
           "stage_monitor", "stage_families", "stage_battery", "stage_cells", "stage_attack_tables",
           "stage_attackers", "stage_attack_eval", "stage_analyse", "stage_report", "battery_breakdown",
           "cells_breakdown", "horizon_breakdown", "stage_manifest", "extension_plan", "stage_battery_extend"]
