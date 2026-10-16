"""Analysis of the M4.6c evaluation families (pre-registration §4-§5). Reads the per-job records, writes
results/m46c/analysis.json. Per-seed values first; across seeds a t interval (df = 5); arms compared by the paired
difference per seed (the same starting final).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from . import common as C
from .archive import LEAD_CELLS, SITS, STAGES, freeze_cells, lead_cells

EVAL = os.path.join(C.CKPT, "eval")
LEADS = (8, 16)


def _load_dir(*parts) -> Dict[int, dict]:
    d = os.path.join(EVAL, *parts)
    out = {}
    if not os.path.isdir(d):
        return out
    for f in os.listdir(d):
        if f.endswith(".json.gz"):
            out[int(f.split(".")[0])] = C.load_gz(os.path.join(d, f))
    return out


def _g(rec, cand, h) -> np.ndarray:
    return np.asarray(rec["G"][cand][str(h)], dtype=float)


def _lmean(rec, cand, key, h) -> float:
    j = rec["keys"].index(key)
    e = rec["ledger"][cand].get(str(h))
    return float(e["mean"][j]) if e else float("nan")


def applied(rec, cand) -> bool:
    """Whether a candidate actually changed seat A's first action: a fixed code was legal (its first recorded code is
    that code), or a behaviour plan forced at least one turn in every rollout; the policy always applies."""
    from .evaluate import FIXED_CODES
    if cand in FIXED_CODES:
        return all(int(c) == FIXED_CODES[cand] for c in rec["first_codes"][cand])
    if cand in rec.get("forced", {}) and cand != "policy":
        return min(rec["forced"][cand]) > 0
    return True


def _ldiff(rec, cand, key, h) -> float:
    j = rec["keys"].index(key)
    e = rec["ledger"][cand].get(str(h))
    return float(e["diff"][j]) if e else float("nan")


CELL_IDX = {c: i for i, c in lead_cells()}


# ---------------------------------------------------------------------------------------------- premium
def premium_of(recs: Dict[int, dict], cand: str, st: str, d: int, h: int, sits=SITS) -> float:
    """½[(G(A+d) - G(eq)) - (G(B+d) - G(eq))], averaged over the situations (paired rollouts within a situation)."""
    v = []
    for sit in sits:
        ia, ib, ie = CELL_IDX[(st, sit, f"A+{d}")], CELL_IDX[(st, sit, f"B+{d}")], CELL_IDX[(st, sit, "equal")]
        if not all(i in recs for i in (ia, ib, ie)):
            continue
        n = min(len(recs[ia]["G"][cand][str(h)]), len(recs[ib]["G"][cand][str(h)]), len(recs[ie]["G"][cand][str(h)]))
        ga, gb, ge = _g(recs[ia], cand, h)[:n], _g(recs[ib], cand, h)[:n], _g(recs[ie], cand, h)[:n]
        v.append(float(np.mean(0.5 * ((ga - ge) - (gb - ge)))))
    return float(np.mean(v)) if v else float("nan")


def premium_rollout_ci(recs: Dict[int, dict], cand: str, st: str, d: int, h: int) -> float:
    """The rollout (within-seed) 95% half-width of a premium, from the per-rollout values averaged over situations."""
    rows = []
    for sit in SITS:
        ia, ib, ie = CELL_IDX[(st, sit, f"A+{d}")], CELL_IDX[(st, sit, f"B+{d}")], CELL_IDX[(st, sit, "equal")]
        if not all(i in recs for i in (ia, ib, ie)):
            return float("nan")
        ga, gb, ge = _g(recs[ia], cand, h), _g(recs[ib], cand, h), _g(recs[ie], cand, h)
        rows.append(0.5 * ((ga - ge) - (gb - ge)))
    x = np.mean(rows, axis=0)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def shift_of(recs: Dict[int, dict], cand: str, base: str, st: str, cell: str, h: int) -> float:
    """Mean over situations of the paired difference cand - base in a cell (J2: Δ = new - start)."""
    v = []
    for sit in SITS:
        i = CELL_IDX[(st, sit, cell)]
        if i in recs:
            v.append(float(np.mean(_g(recs[i], cand, h) - _g(recs[i], base, h))))
    return float(np.mean(v)) if v else float("nan")


def leader_channels(recs: Dict[int, dict], cand: str, st: str, d: int, h: int = 50) -> Dict[str, float]:
    """The leader's and the trailer's ledger change, lead cells minus equal cell, both orientations averaged."""
    keys = ("earned", "farm", "plate", "bounty", "deaths", "away", "xp", "unc", "unc_away", "trade_conn", "d_total",
            "recalls", "crash_hp", "held")
    out = {}
    for k in keys:
        led, trl = [], []
        for sit in SITS:
            ia, ib, ie = CELL_IDX[(st, sit, f"A+{d}")], CELL_IDX[(st, sit, f"B+{d}")], CELL_IDX[(st, sit, "equal")]
            if not all(i in recs for i in (ia, ib, ie)):
                continue
            eqA, eqB = _lmean(recs[ie], cand, f"{k}_A", h), _lmean(recs[ie], cand, f"{k}_B", h)
            led.append(0.5 * ((_lmean(recs[ia], cand, f"{k}_A", h) - eqA) + (_lmean(recs[ib], cand, f"{k}_B", h) - eqB)))
            trl.append(0.5 * ((_lmean(recs[ia], cand, f"{k}_B", h) - eqB) + (_lmean(recs[ib], cand, f"{k}_A", h) - eqA)))
        if led:
            out[f"leader_{k}"] = float(np.mean(led))
            out[f"trailer_{k}"] = float(np.mean(trl))
    return out


def analyse_premium(family: str = "premium", prefix: str = "") -> dict:
    out = {"per_seed": {}, "pooled": {}, "diff": {}, "conv_def": {}, "channels": {}, "critic": {}}
    P: Dict[tuple, Dict[int, float]] = {}

    def put(key, s, v):
        P.setdefault(key, {})[s] = v

    for s in C.SEEDS:
        for env in C.ARMS:
            j1 = _load_dir(family, env, f"J1_{env}_s{s}")
            j2 = _load_dir(family, env, f"J2_{env}_s{s}")
            j3 = _load_dir(family, env, f"J3_s{s}") if env == "C1" else {}
            for st in STAGES:
                for d in LEADS:
                    for h in (20, 50):
                        if j1:
                            put(("self", env, st, d, h), s, premium_of(j1, "self", st, d, h))
                        if j2:
                            put(("start", env, st, d, h), s, premium_of(j2, "start", st, d, h))
                        if j3:
                            put(("j3", env, st, d, h), s, premium_of(j3, "self", st, d, h))
                    if j1:
                        out["per_seed"][f"rollout_ci|self|{env}|{st}|+{d}|s{s}"] = premium_rollout_ci(j1, "self", st, d, 50)
                        out["channels"][f"{env}|{st}|+{d}|s{s}"] = leader_channels(j1, "self", st, d)
                    if j2:
                        for h in (20, 50):
                            eq = shift_of(j2, "new", "start", st, "equal", h)
                            conv = shift_of(j2, "new", "start", st, f"A+{d}", h) - eq
                            dfn = shift_of(j2, "new", "start", st, f"B+{d}", h) - eq
                            put(("conv", env, st, d, h), s, conv)
                            put(("def", env, st, d, h), s, dfn)
                            put(("general", env, st, 0, h), s, eq)
            # critic calibration from J1: V at the cell (seat A) against Q50 of the same rollouts
            if j1:
                out["critic"][f"{env}|s{s}"] = critic_bias(env, s, j1, prefix)
    for key, per in P.items():
        out["pooled"]["|".join(map(str, key))] = C.seed_stats(per)
    # cross-arm differences, paired by seed
    for st in STAGES:
        for d in LEADS:
            for h in (20, 50):
                for kind, a_key, b_key in (("adapted", ("self", "C1"), ("self", "baseline")),
                                           ("pre_adaptation", ("start", "C1"), ("start", "baseline")),
                                           ("adaptation_within_C1", ("self", "C1"), ("j3", "C1")),
                                           ("continuation_within_baseline", ("self", "baseline"), ("start", "baseline")),
                                           ("conv", ("conv", "C1"), ("conv", "baseline")),
                                           ("def", ("def", "C1"), ("def", "baseline"))):
                    a = P.get((a_key[0], a_key[1], st, d, h), {})
                    b = P.get((b_key[0], b_key[1], st, d, h), {})
                    diff = {s: a[s] - b[s] for s in a if s in b and np.isfinite(a[s]) and np.isfinite(b[s])}
                    if not diff:
                        continue
                    stt = C.seed_stats(diff)
                    ma = float(np.mean([a[s] for s in diff]))
                    mb = float(np.mean([b[s] for s in diff]))
                    stt["ratio_of_means"] = ma / mb if mb else float("nan")
                    stt["mean_a"], stt["mean_b"] = ma, mb
                    from scipy import stats as _st  # noqa: F401  (availability only; p via t)
                    v = np.asarray(list(diff.values()))
                    stt["p"] = float(2 * _st.t.sf(abs(v.mean()) / (v.std(ddof=1) / np.sqrt(len(v))), len(v) - 1)) \
                        if len(v) > 1 and v.std(ddof=1) > 0 else float("nan")
                    out["diff"][f"{kind}|{st}|+{d}|G{h}"] = stt
    return out


def critic_bias(env: str, s: int, j1: Dict[int, dict], prefix: str = "") -> dict:
    """V(cell) for seat A (the arm policy's critic) against the realized Q50 = G50 + 0.99^50 V(s50) (seat A); by cell
    type: leader (A+d), trailer (B+d), equal."""
    from ..m45c.horizon import critic_value
    cfg = C.arm_cfg(env)
    from .archive import lead_cell
    net = C.net_of(f"{prefix}{env}:{s}")
    rows: Dict[str, list] = {}
    for (st, sit, c), i in CELL_IDX.items():
        if i not in j1 or j1[i].get("Q") is None:
            continue
        lane, m4, k = lead_cell(cfg, st, sit, c)
        v0 = critic_value(net, cfg, lane, m4, k, 0)
        q = float(np.mean(j1[i]["Q"]["self"]["50"]))
        role = "equal" if c == "equal" else ("leader" if c.startswith("A") else "trailer")
        size = 0 if c == "equal" else int(c[2:])
        rows.setdefault(f"{role}|{st}|{size}", []).append(v0 - q)
    return {k: float(np.mean(v)) for k, v in rows.items()}


def analyse_premium_mid() -> dict:
    """The extension statistic: mid +8 G50 premium of the 2,048,000 snapshots, per arm and seed."""
    out = {}
    for env in C.ARMS:
        per = {}
        for s in C.SEEDS:
            r = _load_dir("premium_mid", env, f"J1_{env}_s{s}")
            if r:
                per[s] = premium_of(r, "self", "mid", 8, 50)
        out[env] = C.seed_stats(per)
    return out


def analyse_premium150() -> dict:
    out = {}
    for env in C.ARMS:
        for st in STAGES:
            for d in (8, 16):
                if d == 8 and st != "mid":
                    continue
                for h in (50, 100, 150):
                    per = {}
                    for s in C.SEEDS:
                        r = _load_dir("premium150", env, f"J1_{env}_s{s}")
                        if r:
                            per[s] = premium_of(r, "self", st, d, h, sits=("neutral",))
                    out[f"{env}|{st}|+{d}|G{h}"] = C.seed_stats(per)
    return out


def analyse_premium150p(family: str = "premium150p") -> dict:
    """EXPLORATORY: the premium at G50 / G100 / G150 from the same 150-turn rollouts (all situations, both orientations,
    K 128), per arm, and the C1 - baseline difference paired by seed."""
    out, P = {}, {}
    for env in C.ARMS:
        for s in C.SEEDS:
            r = _load_dir(family, env, f"J1_{env}_s{s}")
            if not r:
                continue
            for st in ("mid", "late"):
                for d in LEADS:
                    for h in (50, 100, 150):
                        P.setdefault((env, st, d, h), {})[s] = premium_of(r, "self", st, d, h)
    for (env, st, d, h), per in P.items():
        out[f"{env}|{st}|+{d}|G{h}"] = C.seed_stats(per)
    for st in ("mid", "late"):
        for d in LEADS:
            for h in (50, 100, 150):
                a, b = P.get(("C1", st, d, h), {}), P.get(("baseline", st, d, h), {})
                diff = {s: a[s] - b[s] for s in a if s in b}
                if diff:
                    stt = C.seed_stats(diff)
                    mb = float(np.mean([b[s] for s in diff]))
                    stt["ratio_of_means"] = float(np.mean([a[s] for s in diff])) / mb if mb else float("nan")
                    out[f"C1-baseline|{st}|+{d}|G{h}"] = stt
    return out


# ---------------------------------------------------------------------------------------------- role swap (exploratory)
def analyse_roleswap(family: str = "roleswap") -> dict:
    """c1 - bc in the C1 environment against a baseline-trained opponent: lead-specific conversion (A+d) and defence
    (B+d), each minus the equal cell, and the general difference (equal cell)."""
    per: Dict[str, Dict[int, float]] = {}
    for s in C.SEEDS:
        r = _load_dir(family, "C1", f"rs_s{s}")
        if not r:
            continue
        for st in ("early", "mid"):
            for h in (20, 50):
                eq = shift_of(r, "c1", "bc", st, "equal", h)
                per.setdefault(f"general|{st}|G{h}", {})[s] = eq
                for d in LEADS:
                    per.setdefault(f"conversion|{st}|+{d}|G{h}", {})[s] = shift_of(r, "c1", "bc", st, f"A+{d}", h) - eq
                    per.setdefault(f"defence|{st}|+{d}|G{h}", {})[s] = shift_of(r, "c1", "bc", st, f"B+{d}", h) - eq
    return {k: C.seed_stats(v) for k, v in per.items()}


# ---------------------------------------------------------------------------------------------- battery
def analyse_battery(family: str = "battery") -> dict:
    from .evaluate import STRONG_CANDS, WEAK_CANDS, battery_cells
    cells = dict(battery_cells())
    out = {}
    per: Dict[str, Dict[int, float]] = {}
    for env in C.ARMS:
        for s in C.SEEDS:
            r = _load_dir(family, env, f"bat_{env}_s{s}")
            if not r:
                continue
            for grp_stage in list(STAGES) + ["all"]:
                for d in (8, 16):
                    for focal, cands in (("weak", WEAK_CANDS), ("strong", STRONG_CANDS)):
                        idx = [i for i, c in cells.items() if c[0] == focal and c[3] == d and
                               (grp_stage == "all" or c[1] == grp_stage) and i in r]
                        if not idx:
                            continue
                        base = f"{env}|{focal}|{grp_stage}|+{d}"
                        per.setdefault(f"{base}|abs_policy", {})[s] = float(np.mean([_g(r[i], "policy", 50).mean()
                                                                                    for i in idx]))
                        for k in ("deaths_A", "away_A", "earned_A", "d_total_A", "recalls_A", "trade_conn_A",
                                  "deaths_B", "unc_A"):
                            per.setdefault(f"{base}|policy|{k}", {})[s] = float(np.mean([_lmean(r[i], "policy", k, 50)
                                                                                        for i in idx]))
                        best = []
                        for c in cands:
                            if c == "policy":
                                continue
                            v = float(np.mean([np.mean(_g(r[i], c, 50) - _g(r[i], "policy", 50)) for i in idx]))
                            per.setdefault(f"{base}|{c}-policy", {})[s] = v
                            best.append(v)
                            for k in ("deaths_A", "away_A", "earned_A", "earned_B"):
                                per.setdefault(f"{base}|{c}|{k}", {})[s] = float(np.mean([_ldiff(r[i], c, k, 50)
                                                                                         for i in idx]))
                        per.setdefault(f"{base}|best_forced-policy", {})[s] = max(best)
    for k, v in per.items():
        out[k] = C.seed_stats(v)
    # arm differences for the trailer's absolute value and deaths (paired by seed)
    for grp in list(STAGES) + ["all"]:
        for d in (8, 16):
            for key in ("abs_policy", "policy|deaths_A", "best_forced-policy"):
                a = per.get(f"C1|weak|{grp}|+{d}|{key}", {})
                b = per.get(f"baseline|weak|{grp}|+{d}|{key}", {})
                diff = {s: a[s] - b[s] for s in a if s in b}
                if diff:
                    st = C.seed_stats(diff)
                    st["mean_C1"] = float(np.mean([a[s] for s in diff]))
                    st["mean_baseline"] = float(np.mean([b[s] for s in diff]))
                    out[f"C1-baseline|weak|{grp}|+{d}|{key}"] = st
    return out


# ---------------------------------------------------------------------------------------------- natural counterfactuals
def _behaviour_masses(env: str, spec: str, start) -> Dict[str, float]:
    """The policy's behaviour masses at a start state (seat A): Hold / Yield / Trade / Safe among lane codes, Recall,
    Shove, Ward."""
    from ..m45c import horizon as HZ
    from ..m45c import actions as AC
    from ....config import champ_behavior, champ_wave, B_HOLD, B_YIELD, B_TRADE, B_SAFE, B_RECALL, SH
    cfg = C.arm_cfg(env)
    pol = HZ.CRNNet(C.net_of(spec), start[2], 0, 0, 1)
    legal, p = pol.probs(cfg, start[0], start[1], 0, 0)
    out = {"Hold": 0.0, "Yield": 0.0, "Trade": 0.0, "Safe": 0.0, "Recall": 0.0, "Ward": 0.0, "Shove": 0.0}
    for a, q in zip(legal, p):
        a = int(a)
        if a == AC.WARD:
            out["Ward"] += q
            continue
        b = champ_behavior(a)
        if b == B_RECALL:
            out["Recall"] += q
            continue
        out[{B_HOLD: "Hold", B_YIELD: "Yield", B_TRADE: "Trade", B_SAFE: "Safe"}.get(b, "Hold")] += q
        if champ_wave(a) == SH:
            out["Shove"] += q
    return {k: float(v) for k, v in out.items()}


def analyse_natural_cf(family: str = "natural_cf", prefix: str = "") -> dict:
    out = {}
    per: Dict[str, Dict[int, float]] = {}
    sel: Dict[str, list] = {}
    for env in C.ARMS:
        for s in C.SEEDS:
            spec = f"{prefix}{env}:{s}"
            for role in ("trailer", "leader"):
                r = _load_dir(family, env, f"{role}_{env}_s{s}")
                if not r:
                    continue
                stp = os.path.join(EVAL, "states", env, spec.replace(":", "_"), f"{role}.json.gz")
                states = C.load_gz(stp)["states"] if os.path.exists(stp) else []
                cands = [c for c in r[next(iter(r))]["G"] if c != "policy"]
                for c in cands:
                    vals = []
                    for i, rec in r.items():
                        if not applied(rec, c):
                            continue                         # e.g. Recall not legal at that state
                        vals.append(float(np.mean(_g(rec, c, 50) - _g(rec, "policy", 50))))
                    if vals:
                        per.setdefault(f"{env}|{role}|{c}-policy", {})[s] = float(np.mean(vals))
                        per.setdefault(f"{env}|{role}|{c}|share_states_better_0.2", {})[s] = float(np.mean(np.asarray(vals) > 0.2))
                # regret of the policy against the best one-turn alternative per state (ex-post, an upper bound)
                reg, absv = [], []
                for i, rec in r.items():
                    alts = [float(np.mean(_g(rec, c, 50) - _g(rec, "policy", 50))) for c in cands if applied(rec, c)]
                    reg.append(max(0.0, max(alts)) if alts else 0.0)
                    absv.append(float(np.mean(_g(rec, "policy", 50))))
                    if states and i < len(states):
                        m = _behaviour_masses(env, spec, C.state_from_list(states[i]))
                        tr = float(np.mean(_g(rec, "trade1", 50) - _g(rec, "hold1", 50))) if "trade1" in rec["G"] else np.nan
                        sf = float(np.mean(_g(rec, "safe1", 50) - _g(rec, "hold1", 50))) if "safe1" in rec["G"] else np.nan
                        sel.setdefault(f"{env}|{role}", []).append((s, m["Trade"], m["Safe"], m["Hold"], m["Recall"], tr, sf))
                per.setdefault(f"{env}|{role}|regret_bound", {})[s] = float(np.mean(reg))
                per.setdefault(f"{env}|{role}|abs_policy", {})[s] = float(np.mean(absv))
                for k in ("deaths_A", "away_A", "recalls_A"):
                    per.setdefault(f"{env}|{role}|policy|{k}", {})[s] = float(np.mean([_lmean(rec, "policy", k, 50)
                                                                                      for rec in r.values()]))
    for k, v in per.items():
        out[k] = C.seed_stats(v)
    # selectivity: does the policy put more Trade (Safe) mass where Trade (Safe) - Hold is larger?
    for k, rows in sel.items():
        a = np.asarray(rows, dtype=float)
        sk = {}
        for name, pc, vc in (("trade", 1, 5), ("safe", 2, 6)):
            ok = np.isfinite(a[:, vc])
            if ok.sum() > 5 and a[ok, pc].std() > 0 and a[ok, vc].std() > 0:
                sk[f"corr_P{name}_vs_{name}-hold"] = float(np.corrcoef(a[ok, pc], a[ok, vc])[0, 1])
            sk[f"mean_P{name}"] = float(np.mean(a[:, pc]))
        sk["mean_PHold"], sk["mean_PRecall"] = float(np.mean(a[:, 3])), float(np.mean(a[:, 4]))
        sk["n_states"] = int(len(a))
        out[f"selectivity|{k}"] = sk
    return out


# ---------------------------------------------------------------------------------------------- freeze
def analyse_freeze() -> dict:
    from .evaluate import BREAKER_CANDS, BREAKER_CF_CANDS, FREEZER_CANDS, FREEZER_CF_CANDS
    cells = dict(freeze_cells())
    per: Dict[str, Dict[int, float]] = {}
    for env in C.ARMS:
        for s in C.SEEDS:
            for role, cands in (("freezer", FREEZER_CANDS), ("breaker", BREAKER_CANDS)):
                r = _load_dir("freeze", env, f"{role}_{env}_s{s}")
                if not r:
                    continue
                for grp in ("all", 4, 10, 15):
                    idx = [i for i, c in cells.items() if (grp == "all" or c[0] == grp) and i in r]
                    for inv in ("all", "equal", "freezer+8", "breaker+8"):
                        ii = [i for i in idx if inv == "all" or cells[i][2] == inv]
                        if not ii:
                            continue
                        base = f"{env}|{role}|L{grp}|{inv}"
                        per.setdefault(f"{base}|abs_policy", {})[s] = float(np.mean([_g(r[i], "policy", 50).mean() for i in ii]))
                        for k in ("held_A", "frozen_A", "deaths_A", "deaths_B", "crash_hp_A", "crash_hp_B", "away_A",
                                  "d_total_A", "d_total_B", "earned_A", "earned_B"):
                            per.setdefault(f"{base}|policy|{k}@20", {})[s] = float(np.mean([_lmean(r[i], "policy", k, 20)
                                                                                           for i in ii]))
                        for c in cands:
                            if c == "policy":
                                continue
                            per.setdefault(f"{base}|{c}-policy", {})[s] = float(np.mean(
                                [np.mean(_g(r[i], c, 50) - _g(r[i], "policy", 50)) for i in ii]))
                            for k in ("held_A", "frozen_A", "deaths_A", "earned_A", "earned_B"):
                                per.setdefault(f"{base}|{c}|{k}@20", {})[s] = float(np.mean([_ldiff(r[i], c, k, 20)
                                                                                            for i in ii]))
            for role, cands in (("freezer", FREEZER_CF_CANDS), ("breaker", BREAKER_CF_CANDS)):
                r = _load_dir("freeze", env, f"nat{role}_{env}_s{s}")
                if not r:
                    continue
                for c in cands:
                    if c == "policy":
                        continue
                    vals = [float(np.mean(_g(rec, c, 50) - _g(rec, "policy", 50))) for rec in r.values()
                            if applied(rec, c)]
                    if vals:
                        per.setdefault(f"{env}|nat{role}|{c}-policy", {})[s] = float(np.mean(vals))
                per.setdefault(f"{env}|nat{role}|abs_policy", {})[s] = float(np.mean([_g(rec, "policy", 50).mean()
                                                                                     for rec in r.values()]))
                per.setdefault(f"{env}|nat{role}|policy|held_A@20", {})[s] = float(np.mean(
                    [_lmean(rec, "policy", "held_A", 20) for rec in r.values()]))
                per.setdefault(f"{env}|nat{role}|policy|frozen_A@20", {})[s] = float(np.mean(
                    [_lmean(rec, "policy", "frozen_A", 20) for rec in r.values()]))
    return {k: C.seed_stats(v) for k, v in per.items()}


# ---------------------------------------------------------------------------------------------- specialists
def analyse_specialists() -> dict:
    per: Dict[str, Dict[int, float]] = {}
    absg: Dict[str, Dict[int, np.ndarray]] = {}
    for s in C.SEEDS:
        for env, pol in (("baseline", "start"), ("baseline", "baseline"), ("C1", "start"), ("C1", "C1")):
            r = _load_dir("specialists", env, f"{pol}_{env}_s{s}")
            if not r:
                continue
            adv = []
            for t in (0, 2, 4):
                v = [float(np.mean(_g(rec, f"g{t}", 50) - _g(rec, "own", 50))) for rec in r.values()]
                per.setdefault(f"{env}|{pol}|advantage_g{t}", {})[s] = float(np.mean(v))
                adv.append(np.mean(v))
                absg.setdefault(f"{env}|{pol}|g{t}", {})[s] = np.asarray([_g(r[i], f"g{t}", 50).mean() for i in sorted(r)])
            per.setdefault(f"{env}|{pol}|advantage_mean", {})[s] = float(np.mean(adv))
            per.setdefault(f"{env}|{pol}|own_abs", {})[s] = float(np.mean([_g(rec, "own", 50).mean() for rec in r.values()]))
    out = {k: C.seed_stats(v) for k, v in per.items()}
    # defence against the same specialist: G(spec vs start) - G(spec vs new), paired by state and seeds
    for env, new in (("baseline", "baseline"), ("C1", "C1")):
        for t in (0, 2, 4):
            a, b = absg.get(f"{env}|start|g{t}", {}), absg.get(f"{env}|{new}|g{t}", {})
            d = {s: float(np.mean(a[s] - b[s])) for s in a if s in b}
            if d:
                out[f"{env}|defence_gain_vs_g{t}"] = C.seed_stats(d)
        dd = {}
        for s in C.SEEDS:
            vs = [float(np.mean(absg[f"{env}|start|g{t}"][s] - absg[f"{env}|{new}|g{t}"][s])) for t in (0, 2, 4)
                  if s in absg.get(f"{env}|start|g{t}", {}) and s in absg.get(f"{env}|{new}|g{t}", {})]
            if vs:
                dd[s] = float(np.mean(vs))
        if dd:
            out[f"{env}|defence_gain_mean"] = C.seed_stats(dd)
    return out


# ---------------------------------------------------------------------------------------------- natural play
NAT_KEYS_REL = ("all:crash_hp_per100", "plates_by150", "plates_by300", "all:deaths_per100", "all:recalls_per100",
                "all:collection", "all:trade_conn_per100", "all:held_share", "all:away_share", "all:ward_per100",
                "late:crash_hp_per100", "late:deaths_per100", "mid:crash_hp_per100", "early:crash_hp_per100",
                "freeze_episodes_per100", "freeze_episode_mean_len", "all:rec_intr_per100", "all:trade_att_per100",
                "all:bounty_per100", "all:damage_taken_per100")
NAT_KEYS_ABS = ("all:shove_share", "late:shove_share", "mid:shove_share", "early:shove_share")


def analyse_natural(prefix: str = "") -> dict:
    from .natural import metrics, path
    M: Dict[str, Dict[int, dict]] = {}
    from .evaluate import natural_combos
    for spec, env in natural_combos(prefix):
        p = path(spec, env)
        if not os.path.exists(p):
            continue
        s = int(spec.split(":")[1])
        M.setdefault(f"{spec.split(':')[0]}|{env}", {})[s] = metrics(C.load_gz(p))
    out = {"per_combo": {}, "diff": {}}
    for combo, per in M.items():
        keys = next(iter(per.values())).keys()
        out["per_combo"][combo] = {k: C.seed_stats({s: per[s][k] for s in per}) for k in keys}
    pairs = {"arms (C1:s in C1 - baseline:s in baseline)": (f"{prefix}C1|C1", f"{prefix}baseline|baseline"),
             "pre-adaptation (start:s in C1 - start:s in baseline)": ("start|C1", "start|baseline"),
             "continuation (baseline:s - start:s, baseline env)": (f"{prefix}baseline|baseline", "start|baseline"),
             "adaptation (C1:s - start:s, C1 env)": (f"{prefix}C1|C1", "start|C1")}
    for name, (a, b) in pairs.items():
        if a not in M or b not in M:
            continue
        rows = {}
        for k in next(iter(M[a].values())).keys():
            d = {s: M[a][s][k] - M[b][s][k] for s in M[a] if s in M[b] and np.isfinite(M[a][s][k]) and np.isfinite(M[b][s][k])}
            st = C.seed_stats(d)
            mb = float(np.mean([M[b][s][k] for s in d])) if d else float("nan")
            st["base_mean"] = mb
            st["rel"] = st["mean"] / mb if mb else float("nan")
            rows[k] = st
        out["diff"][name] = rows
    return out


def analyse_records() -> dict:
    from .evaluate import natural_combos
    out = {}
    for spec, env in natural_combos():
        p = os.path.join(EVAL, "records", env, spec.replace(":", "_") + ".json.gz")
        if not os.path.exists(p):
            continue
        d = C.load_gz(p)
        key = f"{spec.split(':')[0]}|{env}"
        for cls, row in d["conditional"].items():
            if "mass" not in row:
                continue
            o = out.setdefault(key, {}).setdefault(cls, {"n": 0, "mass": {}})
            o["n"] += row["n"]
            for k, v in row["mass"].items():
                o["mass"].setdefault(k, []).append(v)
    for key, by in out.items():
        for cls, o in by.items():
            o["mass"] = {k: float(np.nanmean(v)) for k, v in o["mass"].items()}
    return out


# ---------------------------------------------------------------------------------------------- head to head
def analyse_h2h() -> dict:
    d = os.path.join(EVAL, "h2h")
    R = {}
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".json.gz"):
                r = C.load_gz(os.path.join(d, f))
                R[r["key"]] = r
    out = {"curve": {}, "start": {}, "others": {}, "rr": {}, "suite": {}, "cross": {}}
    from .evaluate import CURVE_POINTS
    for arm in C.ARMS:
        for n in CURVE_POINTS + (C.BUDGET,):
            per = {s: R[f"curve|{arm}|s{s}|{n}"]["estimate"] for s in C.SEEDS if f"curve|{arm}|s{s}|{n}" in R}
            out["curve"][f"{arm}|{n}"] = C.seed_stats(per)
        per = {s: R[f"start|{arm}|s{s}"]["estimate"] for s in C.SEEDS if f"start|{arm}|s{s}" in R}
        out["start"][arm] = C.seed_stats(per)
        out["start"][f"{arm}|per_seed_ci"] = {s: R[f"start|{arm}|s{s}"]["ci95"] for s in C.SEEDS if f"start|{arm}|s{s}" in R}
        per = {}
        for s in C.SEEDS:
            v = [R[f"others|{arm}|s{s}|o{o}"]["estimate"] for o in C.SEEDS if f"others|{arm}|s{s}|o{o}" in R]
            if v:
                per[s] = float(np.mean(v))
        out["others"][arm] = C.seed_stats(per)
        rr = [R[k]["estimate"] for k in R if k.startswith(f"rr|{arm}|")]
        out["rr"][arm] = {"n_pairs": len(rr), "mean_abs": float(np.mean(np.abs(rr))) if rr else float("nan"),
                          "max_abs": float(np.max(np.abs(rr))) if rr else float("nan"),
                          "sig_pairs": int(sum(1 for k in R if k.startswith(f"rr|{arm}|") and
                                               (R[k]["ci95"][0] > 0 or R[k]["ci95"][1] < 0)))}
        from ..m45c.references import SUITE
        for who in ("final", "start"):
            per = {}
            for s in C.SEEDS:
                x = f"{arm}:{s}" if who == "final" else f"start:{s}"
                v = [R[f"suite|{arm}|{x}|{r}"]["estimate"] for r in SUITE if f"suite|{arm}|{x}|{r}" in R]
                if v:
                    per[s] = float(np.mean(v))
            out["suite"][f"{arm}|{who}"] = C.seed_stats(per)
        per = {}
        for s in C.SEEDS:
            v = [R[f"suite|{arm}|{arm}:{s}|{r}"]["estimate"] - R[f"suite|{arm}|start:{s}|{r}"]["estimate"]
                 for r in SUITE if f"suite|{arm}|{arm}:{s}|{r}" in R and f"suite|{arm}|start:{s}|{r}" in R]
            if v:
                per[s] = float(np.mean(v))
        out["suite"][f"{arm}|final-start"] = C.seed_stats(per)
        for r in SUITE:
            per = {s: R[f"suite|{arm}|{arm}:{s}|{r}"]["estimate"] for s in C.SEEDS if f"suite|{arm}|{arm}:{s}|{r}" in R}
            out["suite"][f"{arm}|final|{r}"] = C.seed_stats(per)
    for env in C.ARMS:
        per = {s: R[f"cross|{env}|s{s}"]["estimate"] for s in C.SEEDS if f"cross|{env}|s{s}" in R}
        out["cross"][env] = C.seed_stats(per)
    # the extension (amendment of 2026-10-15)
    out["ext"] = {}
    for arm in C.ARMS:
        for kind in ("extstart", "extvsmain"):
            per = {s: R[f"{kind}|{arm}|s{s}"]["estimate"] for s in C.SEEDS if f"{kind}|{arm}|s{s}" in R}
            out["ext"][f"{kind}|{arm}"] = C.seed_stats(per)
        per = {}
        for s in C.SEEDS:
            v = [R[f"extothers|{arm}|s{s}|o{o}"]["estimate"] for o in C.SEEDS if f"extothers|{arm}|s{s}|o{o}" in R]
            if v:
                per[s] = float(np.mean(v))
        out["ext"][f"extothers|{arm}"] = C.seed_stats(per)
    for env in C.ARMS:
        per = {s: R[f"extcross|{env}|s{s}"]["estimate"] for s in C.SEEDS if f"extcross|{env}|s{s}" in R}
        out["ext"][f"extcross|{env}"] = C.seed_stats(per)
    return out


# ---------------------------------------------------------------------------------------------- training
def analyse_training(phase: str = "main") -> dict:
    import json
    out = {}
    for arm in C.ARMS:
        for s in C.SEEDS:
            p = os.path.join(C.run_dir(arm, s, phase), "summary.json")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            hp = os.path.join(C.run_dir(arm, s, phase), "history.jsonl")
            with open(hp, encoding="utf-8") as f:
                h = [json.loads(x) for x in f if x.strip()]
            n = max(1, len(h) // 10)
            out[f"{arm}|s{s}"] = {
                "status": r.get("status"), "elapsed_min": r.get("elapsed_s", 0) / 60, "tps": r.get("transitions_per_s"),
                "entropy_first10": float(np.mean([e["entropy"] for e in h[:n]])),
                "entropy_last10": float(np.mean([e["entropy"] for e in h[-n:]])),
                "kl_mean": r.get("approx_kl_mean"), "kl_max": r.get("approx_kl_max"),
                "clipfrac_mean": r.get("clipfrac_mean"), "grad_max": r.get("grad_norm_max"),
                "ev_first10": float(np.mean([e["explained_variance"] for e in h[:n]])),
                "ev_last10": float(np.mean([e["explained_variance"] for e in h[-n:]])),
                "shove_first10": float(np.mean([e.get("act_cat_SH", np.nan) for e in h[:n]])),
                "shove_last10": float(np.mean([e.get("act_cat_SH", np.nan) for e in h[-n:]])),
                "trade_first10": float(np.mean([e.get("act_beh_Trade", np.nan) for e in h[:n]])),
                "trade_last10": float(np.mean([e.get("act_beh_Trade", np.nan) for e in h[-n:]])),
                "deaths_per_game_turn_first10": _rate(h[:n], "ep_deaths"),
                "deaths_per_game_turn_last10": _rate(h[-n:], "ep_deaths"),
                "starts": r.get("starts"), "memory": r.get("memory"),
                "V_s0": r.get("value_s0_resources")}
    return out


def _rate(h, key):
    t = sum(e.get("ep_turns", 0) for e in h)
    return float(sum(e.get(key, 0) for e in h) / t * 100) if t else float("nan")


# ---------------------------------------------------------------------------------------------- all
def run(parts=None) -> dict:
    fns = {"training": analyse_training, "premium": analyse_premium, "premium_mid": analyse_premium_mid,
           "premium150": analyse_premium150, "battery": analyse_battery, "natural_cf": analyse_natural_cf,
           "freeze": analyse_freeze, "specialists": analyse_specialists, "natural": analyse_natural,
           "records": analyse_records, "h2h": analyse_h2h,
           # the pre-registered extension (amendment of 2026-10-15)
           "training_ext": lambda: analyse_training("ext"),
           "premium_ext": lambda: analyse_premium("premium_ext", "ext/"),
           "battery_ext": lambda: analyse_battery("battery_ext"),
           "natural_cf_ext": lambda: analyse_natural_cf("natural_cf_ext", "ext/"),
           "natural_ext": lambda: analyse_natural("ext/"),
           "roleswap": analyse_roleswap, "roleswap_ext": lambda: analyse_roleswap("roleswap_ext"),
           "premium150p": analyse_premium150p, "premium150p_ext": lambda: analyse_premium150p("premium150p_ext")}
    old = C.load("analysis") or {}
    for k, f in fns.items():
        if parts and k not in parts:
            continue
        try:
            old[k] = f()
            print(f"  analysed {k}", flush=True)
        except Exception as ex:                            # noqa: BLE001 - a missing family must not stop the others
            import traceback
            old[k] = {"error": repr(ex), "traceback": traceback.format_exc()}
            print(f"  {k}: {ex!r}", flush=True)
    old["meta"] = C.meta(stage="analysis")
    C.save_json("analysis", old)
    return old


__all__ = ["run", "analyse_premium", "premium_of", "analyse_natural", "analyse_h2h", "analyse_battery",
           "analyse_freeze", "analyse_specialists", "analyse_natural_cf", "analyse_training"]
