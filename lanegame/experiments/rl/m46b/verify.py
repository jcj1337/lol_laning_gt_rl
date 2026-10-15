"""Verification of the M4.6b measurement fixes by targeted reproduction (no full M4.5c re-run).

V1 positions. The M4.5c power-gap stage (`m45c.health.gap_job`: viewer neutral, A x1.76, 16 games x 300 turns, seed
BEHAVIOR_SEED_F + 20) re-run with the fixed metrics for the six 16.384M finals: every readout except the corrected
position fields must equal the stored M4.5c record (results/m45c/ext/health_behavior.json.gz); the corrected position
(lane cells only) and the new Ward share are reported beside the stored values.

V2 battery. The M4.5c battery families C_trade, B_frozen and H_zone_c for two finals (s0, s3) on the battery's own
seeds, written to a separate root: every original candidate must reproduce the stored M4.5c raw job (G and Q of every
rollout); the behaviour-only twins and the economic channels are new. Natural states are re-collected with the
battery's seeds (deterministic, so they are M4.5c's states).
"""

from __future__ import annotations

import gzip
import json
import os
import time

import numpy as np

from .common import CKPT, meta, run_jobs, save_json, setup, spec

V2_FINALS = (0, 3)
V2_FAMILIES = ("C_trade", "B_frozen", "H_zone_c")
V2_ROOT = os.path.join(CKPT, "battery_check")


def _gap_job(args):
    setup()
    from ..m45c.health import gap_job
    return gap_job(args)


def v1(workers: int) -> dict:
    from ..m45c.health import BEHAVIOR_SEED_F, GAP_GAMES, GAP_TURNS
    old = json.load(gzip.open(os.path.join("results", "m45c", "ext", "health_behavior.json.gz"), "rt"))["power_gap"]
    js = [(spec(s), GAP_GAMES, GAP_TURNS, BEHAVIOR_SEED_F + 20) for s in range(6)]
    new = dict(run_jobs(_gap_job, js, workers, "V1 gap stage"))
    out = {}
    for k, v in new.items():
        o = old[k]["zoning"]["by_power"]
        n = v["zoning"]["by_power"]
        same, changed = [], {}
        for cls in n:
            for f, x in n[cls].items():
                if f in ("mean_pos", "ward_share"):
                    changed[f"{cls}.{f}"] = {"stored": o[cls].get(f), "fixed": x}
                    continue
                y = o[cls].get(f)
                same.append(bool((x == y) or (isinstance(x, float) and isinstance(y, float)
                                              and (np.isnan(x) and np.isnan(y) or abs(x - y) < 1e-12))))
        thr_o, thr_n = old[k]["zoning"]["credible_threat"], v["zoning"]["credible_threat"]
        out[k] = {"identical_fields": int(sum(same)), "different_fields": int(len(same) - sum(same)),
                  "positions": changed,
                  "credible_threat_n": {c: {"stored": thr_o[c]["n"], "fixed": thr_n[c]["n"]} for c in thr_n},
                  "pressure_cost": {"stored": old[k]["zoning"]["pressure_cost"], "fixed": v["zoning"]["pressure_cost"]}}
    save_json("verify_v1_positions", {"meta": meta(stage="verify_v1"), "results": out})
    return out


def _collect(args):
    setup()
    from ..m45c import battery as B
    return B.collect_job(args)


def _state(args):
    setup()
    from ..m45c import battery as B
    return B.state_job(args)


def v2(workers: int, finals=V2_FINALS, families=V2_FAMILIES) -> dict:
    from ..m45c import battery as B
    t0 = time.perf_counter()
    js, nat_all = [], {}
    for s in finals:
        sp = spec(s)
        cj = [(sp, f, B.N_EARLY if B.FAMILIES[f][0] == "early" else B.N_NATURAL) for f in families
              if B.FAMILIES[f][0] != "constructed"]
        res = run_jobs(_collect, cj, workers, f"V2 states s{s}")
        nat = {k[1]: v["states"] for k, v in res}
        nat_all[s] = nat
        js += B.jobs(sp, f"s{s}_ext", nat, families, root=V2_ROOT)
    res = dict(run_jobs(_state, js, workers, "V2 battery"))
    # compare the original candidates with the stored M4.5c raw jobs
    cmp_rows = []
    for key, rec in res.items():
        tag, fam, idx, proto, h = key.split("|")
        old_path = B.job_path(tag, fam, int(idx), proto, int(h))
        if not os.path.exists(old_path):
            cmp_rows.append({"key": key, "stored": False})
            continue
        with gzip.open(old_path, "rt", encoding="utf-8") as f:
            old = json.load(f)
        same = all(rec["GQ"][c]["G"] == old["GQ"][c]["G"] and rec["GQ"][c]["Q"] == old["GQ"][c]["Q"]
                   for c in old["GQ"])
        cmp_rows.append({"key": key, "stored": True, "original_candidates": len(old["GQ"]), "identical": bool(same),
                         "new_candidates": sorted(set(rec["GQ"]) - set(old["GQ"]))})
    data = {"meta": meta(stage="verify_v2", finals=list(finals), families=list(families), root=V2_ROOT),
            "jobs": len(res), "compared": sum(r["stored"] for r in cmp_rows),
            "identical": sum(r.get("identical", False) for r in cmp_rows),
            "not_identical": [r["key"] for r in cmp_rows if r["stored"] and not r["identical"]],
            "rows": cmp_rows, "elapsed_s": time.perf_counter() - t0}
    save_json("verify_v2_battery", data)
    return data


__all__ = ["v1", "v2", "V2_ROOT", "V2_FINALS", "V2_FAMILIES"]


def v2_analysis() -> dict:
    """The corrected readings: every candidate (original and behaviour-only twin) against the policy, the primary
    statistic of the family (Q50 natural, G50 constructed), per final and pooled over the two verification finals
    (states as units within a final), plus the twin contrasts and the economic channels at 50 turns."""
    import glob
    from ..m45c import battery as B
    rows = {}
    for f in glob.glob(os.path.join(V2_ROOT, "*", "partial", "*.json.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            r = json.load(fh)
        rows.setdefault((r["family"], r["protocol"]), []).append(r)
    out = {}
    pairs = {"C_trade": [("1t:trade_beh", "1t:hold_beh"), ("1t:trade", "1t:hold"), ("1t:trade_beh", "1t:trade"),
                         ("1t:hold_beh", "1t:hold"), ("1t:safe_beh", "1t:safe")],
             "B_frozen": [("1t:walk_up_beh", "1t:farm"), ("1t:step_back_beh", "1t:step_back")],
             "H_zone_c": [("plan:bully_behx10", "plan:bullyx10"), ("plan:threaten_behx10", "plan:farm_behx10"),
                          ("1t:trade_beh", "1t:hold_beh"), ("1t:trade_beh", "1t:trade")]}
    for (fam, proto), recs in sorted(rows.items()):
        stat = B.primary_statistic(fam)
        h = "50"
        F = {}
        cands = [c for c in recs[0]["GQ"] if c != "policy"]
        per_final = {}
        for r in recs:
            src = {c: np.asarray(r["GQ"][c][stat][h]) for c in r["GQ"]}
            per_final.setdefault(r["tag"], []).append(src)
        def contrast(a, b):
            vals = []
            for tag, sts in per_final.items():
                vals.append(float(np.mean([np.mean(s[a] - s[b]) for s in sts])))
            return {"per_final": vals, "mean": float(np.mean(vals))}
        for c in cands:
            F[f"{c}-policy"] = contrast(c, "policy")
        for a, b in pairs.get(fam, []):
            if a in cands and b in cands:
                F[f"{a}-{b}"] = contrast(a, b)
        # channels at 50 turns (candidate - policy), keys of the M4.6b ledger extension
        ch = {}
        keys = recs[0]["keys"]
        for c in cands:
            for k in ("earned_A", "farm_A", "unc_A", "unc_away_A", "bounty_A", "plate_A", "earned_B", "bounty_B",
                      "deaths_A", "deaths_B", "away_A", "away_B", "d_total_A", "xp_A", "trade_conn_A"):
                if k not in keys:
                    continue
                j = keys.index(k)
                v = [r["ledger"][c]["50"]["diff"][j] for r in recs if "50" in r["ledger"][c]]
                ch[f"{c}|{k}"] = float(np.mean(v)) if v else float("nan")
        F["channels"] = ch
        F["states"] = len(recs)
        out[f"{fam}|{proto}"] = F
    save_json("verify_v2_analysis", {"meta": meta(stage="verify_v2_analysis"), "results": out})
    return out
