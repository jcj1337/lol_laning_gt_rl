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
Attackers: `attackers` (training) and their evaluation cells.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

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


def plan() -> Dict[str, Dict[str, float]]:
    """Simulated turns per evaluation part and cost class (budget)."""
    fam = _game_turns(family_jobs())
    mon = _game_turns(monitor_jobs())
    from .attackers import eval_jobs
    att = _game_turns([{**j, "focals": [("net", "om:0")], "opponent": j["opponent"]} for j in eval_jobs(("om", "sp", "om_table"))])
    return {"M5b_monitor": mon, "M5c_battery": {"battery_om": 6 * battery_turns("om"), "battery_sp": 6 * battery_turns("sp")},
            "M5c_cells": cells_turns(), "M5d_families": fam, "M5d_attack_eval": att}


# ---------------------------------------------------------------------------------------------- stage drivers
def _run_cells(js: List[dict], workers: int, label: str) -> None:
    from . import games as G
    js = sorted(js, key=lambda j: -len(j["focals"]) * j["N"] * j["T"])
    t0 = time.perf_counter()
    C.run_jobs(G.cell_job, js, workers, label)
    C.jdump(os.path.join(C.OUT, f"wall_{label.replace(' ', '_')}.json"), {"jobs": len(js),
                                                                         "wall_s": time.perf_counter() - t0})


def stage_monitor(workers: int) -> None:
    _run_cells(monitor_jobs(), workers, "M5b monitor")


def stage_families(workers: int) -> None:
    _run_cells(family_jobs(), workers, "M5d families")


def stage_battery(workers: int) -> None:
    from . import battery as MB
    for s in C.SEEDS:
        for a in C.ARMS:
            MB.run_policy(f"{a}:{s}", f"{a}_s{s}", workers)


def stage_cells(workers: int) -> None:
    from . import cells as CE
    fin = {f"{a}_s{s}": f"{a}:{s}" for s in C.SEEDS for a in C.ARMS}
    js = CE.weak_jobs(fin) + CE.premium_jobs(fin)
    for s in C.SEEDS:
        sp = CE.arm_specs(s)
        js += CE.conv_jobs(s, sp) + CE.freeze_jobs(s, sp) + CE.recall_jobs(s, sp) + CE.gaps_jobs(s, sp)
        js += CE.natural_jobs(s, sp, CE.natural_states(s))
    CE.run(js, workers, "M5c cells")


def stage_attack_tables(workers: int) -> None:
    from . import attackers as AT
    os.makedirs(os.path.dirname(AT.table_path(0)), exist_ok=True)
    for s in C.SEEDS:
        p = AT.table_path(s)
        if not os.path.exists(p):
            np.save(p, AT.self_play_table(f"om:{s}"))


def stage_attackers(workers: int) -> None:
    from . import attackers as AT
    js = []
    for kind in AT.TARGET_KINDS:
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


def stage_attack_eval(workers: int) -> None:
    from . import attackers as AT
    _run_cells(AT.eval_jobs(AT.TARGET_KINDS), workers, "M5d attack eval")


def stage_analyse(workers: int) -> None:
    from . import analysis as AN
    AN.analyse()


def stage_report(workers: int) -> None:
    from . import report as RP
    RP.report()


__all__ = ["families", "family_jobs", "monitor_jobs", "plan", "battery_turns", "cells_turns", "other", "REF_PANEL",
           "stage_monitor", "stage_families", "stage_battery", "stage_cells", "stage_attack_tables",
           "stage_attackers", "stage_attack_eval", "stage_analyse", "stage_report"]
