"""Analysis of workstream C (specialists) and the critic diagnostic (pre-registration §4, §5)."""

from __future__ import annotations

import glob
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from . import common as C
from . import specialists as SP

GAMMA = C.GAMMA


def ci(v) -> List[float]:
    m, h = C.mean_ci(v)
    return [m, h]


# ---------------------------------------------------------------------------------------------- C1 verify
VER_KEYS = ("earned_A", "earned_B", "farm_A", "bounty_A", "plate_A", "deaths_B", "deaths_A", "away_B", "rec_intr_B",
            "recalls_B", "trade_conn_A", "trade_att_A", "unc_B", "unc_away_B", "d_total_A", "d_total_B", "n_trade_A",
            "n_safe_B", "xp_B", "xp_A")


def verify() -> dict:
    gap_cells = SP.gap_cells()
    pl_cells = SP.plaus_cells()
    out = {}
    for arm in ("baseline", "C1"):
        per = defaultdict(lambda: defaultdict(list))        # group -> s -> [adv per cell, averaged over specialists]
        per_t = defaultdict(lambda: defaultdict(list))
        own = defaultdict(lambda: defaultdict(list))
        led = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for s in C.SEEDS:
            for p in glob.glob(os.path.join(C.CKPT, "spec", "verify", arm, f"s{s}", "*.json.gz")):
                rec = C.load_gz(p)
                fam, i = os.path.basename(p)[:-8].rsplit("_", 1)
                i = int(i)
                if fam == "gap":
                    groups = ["gap all", f"gap {gap_cells[i][2]}", f"gap L{gap_cells[i][1]}"]
                else:
                    c = pl_cells[i]
                    groups = [f"plaus {c['name']} {c['wave']}", f"plaus {c['name']}"] + (
                        ["plausible all (plaus)"] if c["name"] == "plaus" else [])
                base = np.asarray(rec["G"]["own"]["50"])
                advs = []
                for t in C.SPEC_TARGETS:
                    a = float(np.mean(np.asarray(rec["G"][f"g{t}"]["50"]) - base))
                    advs.append(a)
                    for gr in groups:
                        per_t[gr][f"{s}|{t}"].append(a)
                keys = rec["keys"]
                e = rec["ledger"]
                for gr in groups:
                    per[gr][s].append(float(np.mean(advs)))
                    own[gr][s].append(float(base.mean()))
                    for kk in VER_KEYS:
                        j = keys.index(kk)
                        led[gr][kk][s].append(float(np.mean([e[f"g{t}"]["50"]["diff"][j] for t in C.SPEC_TARGETS])))
        res = {}
        for gr in per:
            res[gr] = {"adv": ci([np.mean(v) for v in per[gr].values()]),
                       "own_abs": ci([np.mean(v) for v in own[gr].values()]),
                       "by_specialist": {str(t): ci([np.mean(per_t[gr][f"{s}|{t}"]) for s in C.SEEDS
                                                     if per_t[gr].get(f"{s}|{t}")]) for t in C.SPEC_TARGETS},
                       "ledger50": {kk: ci([np.mean(v) for v in led[gr][kk].values()]) for kk in VER_KEYS},
                       "cells": len(next(iter(per[gr].values())))}
        out[arm] = res
    return out


# ---------------------------------------------------------------------------------------------- C2 decomposition
def decomp() -> dict:
    files = sorted(glob.glob(os.path.join(C.CKPT, "spec", "decomp", "pair*", "*.npz")))
    files = [f for f in files if not f.endswith(".tmp.npz")]
    if not files:
        return {}
    s_blocks = (0,) + SP.S_KS                               # block starts of the spec-first design
    units = []
    cls_S = defaultdict(list)                               # class -> per-unit sums
    cls_U = defaultdict(list)
    act_S = defaultdict(list)
    act_U = defaultdict(list)
    blk = defaultdict(list)
    ufirst = defaultdict(list)
    window: Dict[str, dict] = {}
    for f in files:
        z = np.load(f)
        names = [str(x) for x in z["names"]]
        cps = list(z["cps"])
        j50 = cps.index(50)
        G = z["G"][:, :, j50]
        paths = C.load_gz(f.replace(".npz", ".paths.json.gz"))
        K = G.shape[1]
        gen, spec = G[names.index("gen")], G[names.index("spec")]
        seq = ["gen"] + [f"S{k}" for k in SP.S_KS] + ["spec"]
        inc = np.stack([G[names.index(seq[i + 1])] - G[names.index(seq[i])] for i in range(len(seq) - 1)])  # [blocks, K]
        tag = os.path.basename(os.path.dirname(f))
        cell = os.path.basename(f)[:-4]
        unit = {"pair": tag, "cell": cell, "gap": float((spec - gen).mean()),
                "blocks": inc.mean(1).tolist(), "U": {}}
        S_c = defaultdict(float)
        A_c = defaultdict(float)
        for r in range(K):
            recs = {x["i"]: x for x in paths["spec"][r]}
            for bi, k0 in enumerate(s_blocks):
                x = recs.get(k0)
                c = x["cls"] if x else "?"
                S_c[c] += inc[bi, r] / K
                if k0 <= 3 and x:                            # single-turn blocks: attribute to the action pair
                    A_c[f"{c}|spec {x['a']}|gen {x['alt']}"] += inc[bi, r] / K
        for c, v in S_c.items():
            cls_S[c].append(v)
        for c, v in A_c.items():
            act_S[c].append(v)
        U_c = defaultdict(float)
        UA = defaultdict(float)
        unit_win = defaultdict(list)
        for k in SP.U_KS:
            d = G[names.index(f"U{k}")] - gen
            unit["U"][str(k)] = float(d.mean())
            for r in range(K):
                x = {y["i"]: y for y in paths["gen"][r]}.get(k)
                if not x:
                    continue
                U_c[x["cls"]] += d[r] / K
                UA[f"{x['cls']}|spec {x['alt']}|gen {x['a']}"] += d[r] / K        # gen path: a = gen, alt = spec
                if x["cls"] in ("recall_window", "trade_window", "kill_window"):
                    sp_tr = x["alt"].startswith("Trade")
                    gn_tr = x["a"].startswith("Trade")
                    cat = f"spec {'Trade' if sp_tr else 'no Trade'} / gen {'Trade' if gn_tr else 'no Trade'}"
                    window.setdefault(x["cls"], defaultdict(list))[cat].append(d[r])
                    unit_win[(x["cls"], cat)].append(d[r])
        for c, v in U_c.items():
            cls_U[c].append(v)
        for c, v in UA.items():
            act_U[c].append(v)
        unit["win"] = {f"{a}|{b}": [float(np.mean(v)), len(v)] for (a, b), v in unit_win.items()}
        units.append(unit)
    n = len(units)

    def tot(dct):                                           # mean over units (missing = 0), CI over units
        return {c: ci(v + [0.0] * (n - len(v))) for c, v in dct.items()}

    gaps = [u["gap"] for u in units]
    blocks = np.asarray([u["blocks"] for u in units])
    usum = [sum(u["U"].values()) for u in units]
    res = {"units": n, "gap": ci(gaps), "by_pair": {}, "blocks": {f"{a}-{b}": ci(blocks[:, i]) for i, (a, b) in
                                                                  enumerate(zip(s_blocks, list(SP.S_KS) + ["inf"]))},
           "S_by_class": tot(cls_S), "U_by_class": tot(cls_U), "U_sum_sampled": ci(usum),
           "U_by_k": {k: ci([u["U"][k] for u in units]) for k in map(str, SP.U_KS)},
           "S_by_action_k<=3": {c: v for c, v in sorted(tot(act_S).items(), key=lambda kv: -abs(kv[1][0]))[:25]},
           "U_by_action": {c: v for c, v in sorted(tot(act_U).items(), key=lambda kv: -abs(kv[1][0]))[:25]},
           "window_rollouts": {cls: {k: {"n": len(v), "mean": float(np.mean(v)) if v else float("nan")}
                                     for k, v in cats.items()} for cls, cats in window.items()}}
    # per-unit means of the one-turn substitution value by (class, action pair), interval over units having it
    keys = sorted({k for u in units for k in u["win"]})
    res["window_by_unit"] = {k: {"value": ci([u["win"][k][0] for u in units if k in u["win"]]),
                                 "units": sum(k in u["win"] for u in units),
                                 "rollout_turns": int(sum(u["win"][k][1] for u in units if k in u["win"]))}
                             for k in keys}
    for t in C.SPEC_TARGETS:
        us = [u for u in units if u["pair"] == f"pair{t}"]
        res["by_pair"][str(t)] = {"gap": ci([u["gap"] for u in us]), "U_sum": ci([sum(u["U"].values()) for u in us])}
    by_cell_kind = defaultdict(list)
    for u in units:
        by_cell_kind["plaus" if u["cell"].startswith("plaus") else "gap"].append(u)
    res["by_kind"] = {k: {"gap": ci([u["gap"] for u in us]), "U_sum": ci([sum(u["U"].values()) for u in us]),
                          "units": len(us)} for k, us in by_cell_kind.items()}
    return res


# ---------------------------------------------------------------------------------------------- critic
def calib() -> dict:
    files = glob.glob(os.path.join(C.CKPT, "critic", "calib", "*", "*", "*.json.gz"))
    if not files:
        return {}
    rows = []
    for p in files:
        r = C.load_gz(p)
        G = np.asarray(r["G"])
        cps = r["cps"]
        tail = np.asarray(r["tail"])
        row = {"arm": r["arm"], "spec": r["spec"], "family": r["family"], "label": r["label"], "VA": r["VA"],
               "VB": r["VB"], "G": {str(c): float(G[:, j].mean()) for j, c in enumerate(cps)},
               "G_se": {str(c): float(G[:, j].std(ddof=1) / np.sqrt(len(G))) for j, c in enumerate(cps)},
               "H": cps[-1], "tail": float(tail.mean())}
        rows.append(row)
    out = {}
    for arm in ("C1", "baseline"):
        rs = [r for r in rows if r["arm"] == arm]
        groups = defaultdict(list)
        for r in rs:
            fam = r["family"]
            lab = r["label"]
            keys = [fam]
            if fam == "gap":
                keys.append("gap " + ("x1.76" if "x1.76" in lab else "x1.5"))
            if fam == "freeze":
                keys.append("freeze " + lab.split()[2])
            if fam == "lane":
                keys.append("lane " + lab.split()[2])
            for k in keys:
                groups[k].append(r)
        res = {}
        for k, grp in groups.items():
            by_pol = defaultdict(list)
            for r in grp:
                by_pol[r["spec"]].append(r)
            H = grp[0]["H"]

            def pol_mean(fn):
                return ci([np.mean([fn(r) for r in v]) for v in by_pol.values()])
            res[k] = {"n_states": len(grp), "H": H,
                      "VA": pol_mean(lambda r: r["VA"]), "VB": pol_mean(lambda r: r["VB"]),
                      "G50": pol_mean(lambda r: r["G"]["50"]), "G100": pol_mean(lambda r: r["G"]["100"]),
                      "G300": pol_mean(lambda r: r["G"]["300"]),
                      "G_H": pol_mean(lambda r: r["G"][str(r["H"])]),
                      "bias_A_noboot": pol_mean(lambda r: r["VA"] - r["G"][str(r["H"])]),
                      "bias_A_boot": pol_mean(lambda r: r["VA"] - r["G"][str(r["H"])] - r["tail"]),
                      "bias_B_noboot": pol_mean(lambda r: r["VB"] + r["G"][str(r["H"])]),
                      "zero_sum_VA+VB": pol_mean(lambda r: r["VA"] + r["VB"]),
                      "tail_mean": pol_mean(lambda r: r["tail"]),
                      "G300_to_H": pol_mean(lambda r: r["G"][str(r["H"])] - r["G"]["300"]),
                      "G100_to_300": pol_mean(lambda r: r["G"]["300"] - r["G"]["100"]),
                      "se_G_H_state": float(np.mean([r["G_se"][str(r["H"])] for r in grp]))}
            # explained variance of V for E[G] across states within each policy
            evs = []
            for v in by_pol.values():
                if len(v) >= 4:
                    y = np.asarray([r["G"][str(r["H"])] for r in v])
                    x = np.asarray([r["VA"] for r in v])
                    evs.append(float(np.corrcoef(x, y)[0, 1]) if y.std() > 0 and x.std() > 0 else float("nan"))
            res[k]["corr_V_G"] = ci(evs) if evs else [float("nan"), float("nan")]
        out[arm] = res
    return out


def rank() -> dict:
    files = glob.glob(os.path.join(C.CKPT, "critic", "rank", "*", "*.json.gz"))
    if not files:
        return {}
    from scipy.stats import spearmanr
    per_state = defaultdict(dict)                       # (family, idx, decider) -> spec -> record
    for p in files:
        r = C.load_gz(p)
        per_state[(r["family"], r["idx"], r["decider"], r["label"])][r["spec"]] = r
    out_states = []
    agg = defaultdict(list)
    for (fam, idx, dec, lab), pols in sorted(per_state.items()):
        cands = None
        st = {"family": fam, "idx": idx, "decider": dec, "label": lab, "cands": {}}
        rows = defaultdict(lambda: defaultdict(list))
        rho, sign_agree, best_match, actor_best_mass = [], [], [], []
        for spec, r in pols.items():
            names = [c for c in r["cands"] if c != "policy"]
            pol = r["cands"]["policy"]
            j = r["cps"].index(150)
            g_pol = np.asarray(pol["G"])[:, j]
            c_pol = np.asarray(pol["crit"])
            adv_r, adv_c = {}, {}
            for c in names:
                gg = np.asarray(r["cands"][c]["G"])[:, j]
                adv_r[c] = float(np.mean(gg - g_pol))
                adv_c[c] = float(np.mean(np.asarray(r["cands"][c]["crit"]) - c_pol))
                rows[c]["roll"].append(adv_r[c])
                rows[c]["crit"].append(adv_c[c])
                rows[c]["prob"].append(r["probs"][c])
                rows[c]["roll100"].append(float(np.mean(np.asarray(r["cands"][c]["G"])[:, r["cps"].index(100)] -
                                                        np.asarray(pol["G"])[:, r["cps"].index(100)])))
            if len(names) >= 3:
                rho.append(float(spearmanr([adv_r[c] for c in names], [adv_c[c] for c in names])[0]))
                sign_agree.append(float(np.mean([np.sign(adv_r[c]) == np.sign(adv_c[c]) for c in names])))
                best_match.append(float(max(names, key=adv_r.get) == max(names, key=adv_c.get)))
                actor_best_mass.append(r["probs"][max(names, key=adv_r.get)])
        for c, d in rows.items():
            st["cands"][c] = {"roll150": ci(d["roll"]), "roll100": ci(d["roll100"]), "crit": ci(d["crit"]),
                              "prob": ci(d["prob"])}
        st["spearman"] = ci(rho)
        st["sign_agree"] = ci(sign_agree)
        st["best_match"] = ci(best_match)
        st["actor_mass_on_rollout_best"] = ci(actor_best_mass)
        # profitable decisions rarely chosen
        st["missed"] = [c for c, v in st["cands"].items() if v["roll150"][0] >= 0.25 and
                        v["roll150"][0] - v["roll150"][1] > 0 and v["prob"][0] < 0.10]
        out_states.append(st)
        key = f"{fam} decider {'AB'[dec]}"
        agg[key].append(st)
    summary = {}
    for k, sts in agg.items():
        summary[k] = {"states": len(sts), "spearman": float(np.nanmean([s["spearman"][0] for s in sts])),
                      "sign_agree": float(np.nanmean([s["sign_agree"][0] for s in sts])),
                      "best_match": float(np.nanmean([s["best_match"][0] for s in sts])),
                      "actor_mass_on_best": float(np.nanmean([s["actor_mass_on_rollout_best"][0] for s in sts])),
                      "missed": [(s["label"], s["missed"]) for s in sts if s["missed"]]}
    return {"states": out_states, "summary": summary}


def rank_yield() -> dict:
    """EXPLORATORY follow-up: the weak side's Yield and Safe codes at the constructed Recall windows (decider B)."""
    per = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(C.CKPT, "critic", "rank_yield", "*", "*.json.gz")):
        r = C.load_gz(p)
        j = r["cps"].index(150)
        pol = np.asarray(r["cands"]["policy"]["G"])[:, j]
        cp = np.asarray(r["cands"]["policy"]["crit"])
        for c, v in r["cands"].items():
            if c != "policy":
                per[r["label"]][c].append((float(np.mean(np.asarray(v["G"])[:, j] - pol)),
                                           float(np.mean(np.asarray(v["crit"]) - cp)), r["probs"][c]))
    return {lab: {c: {"roll150": ci([x[0] for x in v]), "crit": ci([x[1] for x in v]),
                      "prob": float(np.mean([x[2] for x in v]))} for c, v in d.items()} for lab, d in per.items()}


def run() -> dict:
    out = {}
    for name, fn in (("verify", verify), ("decomp", decomp), ("calib", calib), ("rank", rank),
                     ("rank_yield", rank_yield)):
        try:
            out[name] = fn()
        except Exception as e:                              # a stage not yet run
            out[name] = {"error": repr(e)}
    return out


__all__ = ["verify", "decomp", "calib", "rank", "run"]
