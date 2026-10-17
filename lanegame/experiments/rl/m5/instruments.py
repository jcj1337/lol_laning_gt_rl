"""R10: every M5b-M5d instrument end to end at reduced samples, on the readiness stand-ins (readiness module docstring),
with its cost per simulated turn measured while the instruments run concurrently on the campaign's worker count.

Rates (simulated turns per worker-second) by cost class: game_om0 / game_om1 / game_om2 (s0 games with 0, 1 or 2
om-format seats), battery_sp / battery_om (battery rollouts; the om policy plays both seats under the "policy"
protocol), cells_sp / cells_om / cells_cross, lane_exit_om. Reduced-sample values are pipeline checks; agreement with
published values at these sizes is weak evidence and is reported as such.
"""

from __future__ import annotations

import os
import time
import zlib
from typing import Dict, List

import numpy as np

from . import common as C
from . import players as PL

IOUT = os.path.join(C.CKPT, "readiness", "instruments")


def _om(s=0):
    from .readiness import om_proxy
    return om_proxy(s)


def _sp(s=0):
    from .readiness import sp_proxy
    return sp_proxy(s)


# ---------------------------------------------------------------------------------------------- jobs (picklable)
def game_job(args) -> dict:
    """(label, focal spec, opponent spec, N, T, kind): one games cell; returns its cost and sanity readouts."""
    label, focal, opp, N, T, kind = args
    from . import games as G
    path = os.path.join(IOUT, "games", f"{label}.npz")
    if os.path.exists(path):
        os.remove(path)
    j = {"key": label, "path": path, "focals": [list(focal)], "opponent": list(opp), "N": N, "T": T,
         "root": C.READY_ROOT, "cell": zlib.crc32(label.encode()) % 1000}
    t0 = time.perf_counter()
    G.cell_job(j)
    el = time.perf_counter() - t0
    d = G.load(path)
    R = d["f0__r"]
    out = {"label": label, "kind": kind, "turns": 2 * N * T, "elapsed_s": el, "rate": 2 * N * T / el,
           "lead114": float(G.lead_at(R).mean()) if T >= 114 else None, "finite": bool(np.isfinite(R).all()),
           "panel_present": bool(np.isfinite(d["f0__panel"]).any()) if T > 94 else None,
           "belief_trace_varies": bool(np.ptp(d["f0__trace"]) > 0)}
    if T >= 500:
        out["discounted"] = float(G.discounted(R).mean())
    return out


def battery_job(args) -> dict:
    """(label, spec, families, K): the policy's natural states (2 per family) and its battery jobs at K."""
    label, spec, fams, K = args
    from . import battery as MB
    root = os.path.join(IOUT, "battery")
    t0 = time.perf_counter()
    from ..m45c import battery as B
    nat = {}
    for f in fams:
        if B.FAMILIES[f][0] != "constructed":
            _, r = MB.collect_job((spec, f, 1))
            nat[f] = r["states"]
    tc = time.perf_counter() - t0
    js = MB.jobs(spec, label, nat, fams, protocols=("policy",), k_nat=K, k_con=K, root=root,
                 grids={"F_freeze_c": [_first_grid("F_freeze_c")]}, validation=False)
    t1 = time.perf_counter()
    turns, ok = 0, True
    recorded = []
    for j in js:
        p = MB.job_path(j[1], j[2], j[3], j[6], j[7], root)
        if os.path.exists(p):
            os.remove(p)
        _, d = MB.state_job(j)
        turns += d["turns"]
        ok = ok and all(np.isfinite(v).all() for c in d["GQ"].values() for st in c.values() for v in
                        [np.asarray(x, dtype=float) for x in st.values()])
        recorded.append(d.get("belief_start"))
    el = time.perf_counter() - t1
    return {"label": label, "kind": f"battery_{C.arm_of(spec)}", "turns": turns, "elapsed_s": el,
            "rate": turns / el, "collect_s": tc, "jobs": len(js), "finite": ok, "belief_start": recorded}


def _first_grid(fam):
    from ..m45c import battery as B
    return B.GRIDS[fam][0]


def cells_job(args) -> dict:
    """(label, family, seed): one or two jobs of a cells family at K 2 on the stand-ins."""
    label, family, seed = args
    from . import cells as CE
    root = os.path.join(IOUT, "cells")
    specs = {"om": _om(seed), "sp": _sp(seed)}
    if family == "weak":
        js = CE.weak_jobs({"om_s0": specs["om"]}, K=2, root=root)[:2] + CE.weak_jobs({"sp_s0": specs["sp"]}, K=2,
                                                                                      root=root)[:2]
        kind = "cells_mixed"
    elif family == "premium":
        js = CE.premium_jobs({"om_s0": specs["om"]}, K=2, root=root)[:1] + CE.premium_jobs({"sp_s0": specs["sp"]}, K=2,
                                                                                          root=root)[:1]
        kind = "cells_mixed"
    elif family == "natural":
        st = CE._states_job_m5((C.start_spec(seed), "leader", 2, 30))[1]["states"]
        js = CE.natural_jobs(seed, specs, {"leader": st[:1]}, root=root)
        kind = "cells_cross"
    else:
        js = getattr(CE, f"{family}_jobs")(seed, specs, root=root)[:2]
        kind = "cells_cross"
    turns, t0 = 0, time.perf_counter()
    rec = None
    for j in js:
        j = dict(j)
        j["roll"] = j["roll"][:2]
        j["path"] = None
        _, rec = CE.paired_job(j)
        turns += rec["turns"]
    el = time.perf_counter() - t0
    return {"label": label, "kind": kind, "turns": turns, "elapsed_s": el, "rate": turns / el, "jobs": len(js),
            "cands": list(rec["cands"]) if rec else None,
            "finite": bool(rec) and all(np.isfinite(np.asarray(v, dtype=float)).all() for c in rec["G"].values()
                                        for v in c.values())}


def lane_exit_job(args) -> dict:
    """(label, spec, games, K): recorded om self-play to turn 114; E_20 at turn 94 by K continuations from the recorded
    state with the recorded beliefs (and, descriptively, with beliefs reset to the prior) against the game's own lead at
    114. Accounting exactness is enforced by lane_exit.continuations."""
    label, spec, n_games, K = args
    from ....diagnostics import lane_exit as LX
    from . import battery as MB
    cfg = C.arm_cfg("sp")
    t0 = time.perf_counter()
    rows, turns = [], 0
    for g in range(n_games):
        sd = C.LANE_EXIT_ROOT * 10 ** 4 + g
        states = MB.self_play(spec, sd, 115)
        lane, m4, k, info = states[94]
        D0 = LX.lead(m4, cfg)
        rec114 = LX.lead(states[114][1], cfg)
        est = {}
        for mode, inf in (("recorded", info), ("prior", None)):
            r = LX.continuations(cfg, (lane, m4, k), lambda s_: (PL.crn_policy(spec, k, s_, 0, 20, inf),
                                                                 PL.crn_policy(spec, k, s_, 1, 20, inf)),
                                 [C.LANE_EXIT_ROOT * 10 ** 6 + 1000 * g + r_ for r_ in range(K)], 20)
            est[mode] = float(D0 + r["r"].sum(-1).mean())
            turns += K * 20
        rows.append({"D0": D0, "E20_recorded_beliefs": est["recorded"], "E20_prior_beliefs": est["prior"],
                     "recorded_lead_114": rec114})
        turns += 115
    el = time.perf_counter() - t0
    d = np.asarray([[r["E20_recorded_beliefs"] - r["recorded_lead_114"], r["E20_prior_beliefs"] - r["recorded_lead_114"]]
                    for r in rows])
    return {"label": label, "kind": "lane_exit_om", "turns": turns, "elapsed_s": el, "rate": turns / el,
            "states": len(rows), "rows": rows,
            "mean_error_recorded": float(d[:, 0].mean()), "se_recorded": float(d[:, 0].std(ddof=1) / np.sqrt(len(d))),
            "mean_error_prior": float(d[:, 1].mean()), "se_prior": float(d[:, 1].std(ddof=1) / np.sqrt(len(d)))}


def table_job(args) -> dict:
    label, spec, games, T = args
    from .attackers import self_play_table
    t0 = time.perf_counter()
    tab = self_play_table(spec, games=games, T=T)
    el = time.perf_counter() - t0
    return {"label": label, "kind": "game_om2", "turns": games * T, "elapsed_s": el, "rate": games * T / el,
            "shape": list(tab.shape), "rows_sum_to_one": bool(np.allclose(tab.sum(1), 1.0, atol=1e-4))}


def rate_job(kind: str) -> dict:
    """Steady-state cost of one class: a warm-up segment (networks loaded, lane memo warm), then a timed segment."""
    from ..m45c import battery as B
    from . import battery as MB
    from . import cells as CE
    from . import games as G
    C.setup("sp")
    om, sp = ("net", _om(0)), ("net", _sp(0))

    def games(focal, opp, n, T):
        turns = 0
        for m in range(n):
            for o, (a, b) in enumerate(((focal, opp), (opp, focal))):
                G.play(a, b, G.game_seed(C.READY_ROOT, 900, m), T)
                turns += T
        return turns

    def battery(spec, K):
        st = B.constructed_state(C.arm_cfg("sp"), "F_freeze_c", B.GRIDS["F_freeze_c"][0])
        root = os.path.join(IOUT, "rates", kind)
        p = MB.job_path("rate", "F_freeze_c", K, "policy", 50, root)
        if os.path.exists(p):
            os.remove(p)
        return MB.state_job((spec, "rate", "F_freeze_c", K, st, None, "policy", 50, "", K, root, C.READY_ROOT + 5))[1]["turns"]

    def cells(fn, specs, K):
        js = fn(specs)
        turns = 0
        for j in js[:2]:
            j = dict(j)
            j["roll"], j["path"] = j["roll"][:K], None
            turns += CE.paired_job(j)[1]["turns"]
        return turns

    def lane_exit(K):
        from ....diagnostics import lane_exit as LX
        states = MB.self_play(_om(0), C.READY_ROOT * 10 + 7, 95)
        lane, m4, k, info = states[94]
        LX.continuations(C.arm_cfg("sp"), (lane, m4, k), lambda s_: (PL.crn_policy(_om(0), k, s_, 0, 20, info),
                                                                     PL.crn_policy(_om(0), k, s_, 1, 20, info)),
                         list(range(K)), 20)
        return K * 20

    spec = {"game_om0": lambda w: games(sp, ("opp", "g_shove"), 2 if w else 10, 150),
            "game_om1": lambda w: games(om, ("opp", "g_shove"), 2 if w else 10, 150),
            "game_om2": lambda w: games(om, ("net", _om(1)), 1 if w else 5, 300),
            "battery_sp": lambda w: battery(_sp(0), 2 if w else 6),
            "battery_om": lambda w: battery(_om(0), 2 if w else 6),
            "cells_cross": lambda w: cells(lambda s: CE.conv_jobs(0, s), {"om": _om(0), "sp": _sp(0)}, 2 if w else 6),
            "cells_om": lambda w: cells(lambda s: CE.weak_jobs({"om_s0": s["om"]}), {"om": _om(0)}, 2 if w else 6),
            "cells_sp": lambda w: cells(lambda s: CE.weak_jobs({"sp_s0": s["sp"]}), {"sp": _sp(0)}, 2 if w else 6),
            "lane_exit_om": lambda w: lane_exit(4 if w else 24)}[kind]
    spec(True)
    t0 = time.perf_counter()
    turns = spec(False)
    el = time.perf_counter() - t0
    return {"label": f"rate_{kind}", "kind": kind, "turns": turns, "elapsed_s": el, "rate": turns / el}


RATE_KINDS = ("game_om0", "game_om1", "game_om2", "battery_sp", "battery_om", "cells_cross", "cells_om", "cells_sp",
              "lane_exit_om")


def _dispatch(job):
    kind, args = job
    try:
        return {"game": game_job, "battery": battery_job, "cells": cells_job, "lane_exit": lane_exit_job,
                "table": table_job, "rate": rate_job}[kind](args)
    except Exception as ex:                                # noqa: BLE001 - recorded as a failed instrument
        import traceback
        return {"label": str(args[0]), "kind": f"{kind}_failed", "error": repr(ex), "traceback": traceback.format_exc(),
                "rate": float("nan"), "finite": False}


# ---------------------------------------------------------------------------------------------- R10
def r10_instruments(workers: int = 6) -> dict:
    from . import readiness as R
    om, sp, st = ("net", _om(0)), ("net", _sp(0)), ("net", "start:0")
    jobs = [("game", ("om_vs_g_shove", om, ("opp", "g_shove"), 6, 150, "game_om1")),
            ("game", ("sp_vs_g_shove", sp, ("opp", "g_shove"), 6, 150, "game_om0")),
            ("game", ("start_vs_its", st, ("opp", "its:g_farm:40:3"), 4, 150, "game_om0")),
            ("game", ("om_vs_its", om, ("opp", "its:g_farm:40:3"), 4, 150, "game_om1")),
            ("game", ("om_vs_style", om, ("opp", "style:respect:3"), 4, 150, "game_om1")),
            ("game", ("om_vs_base3", om, ("opp", "base:3"), 4, 150, "game_om1")),
            ("game", ("om_vs_om1_long", om, ("net", _om(1)), 2, 500, "game_om2")),
            ("game", ("om_vs_sp_long", om, sp, 2, 500, "game_om1")),
            ("battery", ("om_proxy", _om(0), ("A_wave", "B_freeze", "F_freeze_c"), 2)),
            ("battery", ("sp_proxy", _sp(0), ("A_wave", "B_freeze", "F_freeze_c"), 2)),
            ("lane_exit", ("om_lane_exit", _om(0), 8, 8)),
            ("table", ("om_table", _om(0), 2, 100))]
    jobs += [("cells", (f"cells_{f}", f, 0)) for f in ("weak", "premium", "conv", "freeze", "recall", "natural",
                                                       "gaps")]
    jobs.sort(key=lambda j: j[0] not in ("battery", "lane_exit"))
    t0 = time.perf_counter()
    res = C.run_jobs(_dispatch, jobs, workers, "R10 instruments")
    C.jdump(os.path.join(IOUT, "r10_raw.json"), res)
    small: Dict[str, List[float]] = {}
    for r in res:
        if not r["kind"].endswith("_failed"):
            small.setdefault(r["kind"], []).append(r["rate"])
    # steady-state costs (warm-up, then a timed segment), the classes run concurrently on `workers` processes
    rres = C.run_jobs(_dispatch, [("rate", k) for k in RATE_KINDS], workers, "R10 steady rates")
    C.jdump(os.path.join(IOUT, "r10_rates_raw.json"), rres)
    by = {r["kind"]: float(r["rate"]) for r in rres if not r["kind"].endswith("_failed")}
    lx = next((r for r in res if r["kind"] == "lane_exit_om"), {"mean_error_recorded": float("nan"), "se_recorded": 0})
    checks = {"every instrument ran end to end": not [r for r in res if r["kind"].endswith("_failed")],
              "all readouts finite": all(r.get("finite", True) for r in res),
              "lane-exit panel recorded in s0 games": all(r.get("panel_present") for r in res
                                                          if r.get("panel_present") is not None),
              "om focals' belief traces vary": all(r["belief_trace_varies"] for r in res
                                                   if "belief_trace_varies" in r and r["label"].startswith("om")),
              "battery jobs restore recorded beliefs for natural om states": any(
                  "recorded" in (r.get("belief_start") or []) for r in res if r["kind"] == "battery_om"),
              "lane-exit continuations exact (accounting) and their E_20 within 3 SE of the recorded lead at 114":
              abs(lx["mean_error_recorded"]) <= 3 * max(lx["se_recorded"], 1e-9)}
    checks["steady-state cost measured for every class"] = all(k in by for k in RATE_KINDS)
    return R._record("R10", {"checks": checks, "data": {"jobs": res, "rates": by, "workers": workers,
                                                        "small_job_rates": {k: float(np.mean(v)) for k, v in small.items()},
                                                        "wall_s": time.perf_counter() - t0,
                                                        "note": "stand-ins: OM2 treatment finals (om format), M4.7 "
                                                        "control finals (sp format); reduced samples: pipeline and "
                                                        "cost checks, weak evidence about values"}})


__all__ = ["r10_instruments", "game_job", "battery_job", "cells_job", "lane_exit_job", "table_job"]
