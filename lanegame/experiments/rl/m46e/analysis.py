"""Readouts of the supervised value study and the horizon check (docs/m4/m4_6/m4_6e_preregistration.md).

Per policy, split and arm; then mean and 95% t interval over the six policies (the top-level units).
    natural_test      held-out natural games: MAE, RMSE, bias (pred - label), slope (label on pred), R^2
    con_start_test    held-out constructed starts (t = 0), the label averaged over the two rollouts; the label-noise
                      floor of that mean is sigma^2 / 2, sigma^2 = E[(y_r0 - y_r1)^2] / 2; explained_share =
                      1 - (MSE - floor) / (Var(label) - floor), the share of the explainable variance captured
    con_traj_test     held-out constructed trajectory points (t = 15 .. 150)
    beyond_1.6        held-out examples with duel beyond x1.6 either way (the strength split's test)
    freeze_test       held-out constructed freeze groups (the family split's test)
    pairs             paired premiums on held-out groups (t = 0, both seats): lead = orig - eq, level = lvl - orig,
                      hp = hpB - orig, big = big - orig, oriented as seat A's gain (seat B's predicted change negated); Sum
                      pred / Sum realized and the means
    m46d              the M4.6d states (K 32 labels): seat A's bias by family (seat B separately); the P1 premium classes edge / plaus / large
                      (Sum pred / Sum realized over the contrasts and both seats) and the gap cells' value; natural
                      lead / even states against G_500 (bias, slope, MAE)
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np

from . import common as C
from . import fit as F

ARMS = ("critic", "B", "L", "C", "D")
SEED_ARMS = tuple(f"C{s}" for s in F.FIT_SEEDS) + tuple(f"D{s}" for s in F.FIT_SEEDS)
CONTRASTS = {"lead": (1, 0), "level": (0, 2), "hp": (0, 3), "big": (0, 4)}      # (base variant, varied variant)


def basic(y, p) -> Dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3:
        return {}
    e = p - y
    vp = p.var()
    return {"n": int(len(y)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()), "slope": float(np.cov(p, y)[0, 1] / vp) if vp > 0 else float("nan"),
            "r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())}


def _start_means(e, mask, pred):
    """t = 0 constructed rows of `mask`: (label mean over rollouts, prediction, key) per (group, variant, seat)."""
    d = defaultdict(lambda: [[], None])
    for i in np.flatnonzero(mask & (e["t"] == 0) & (e["src"] == 1)):
        k = (int(e["group"][i]), int(e["variant"][i]), int(e["seat"][i]))
        d[k][0].append((int(e["rollout"][i]), float(e["y"][i])))
        d[k][1] = float(pred[i])
    keys = sorted(d)
    ym = np.asarray([np.mean([y for _, y in d[k][0]]) for k in keys])
    pm = np.asarray([d[k][1] for k in keys])
    diffs = [dict(d[k][0]).get(0) - dict(d[k][0]).get(1) for k in keys if len(d[k][0]) == 2]
    return keys, ym, pm, np.asarray(diffs)


def pair_premiums(e, mask, pred) -> Dict[str, dict]:
    keys, ym, pm, _ = _start_means(e, mask, pred)
    look = {k: (y, p) for k, y, p in zip(keys, ym, pm)}
    out = {}
    for name, (v0, v1) in CONTRASTS.items():
        real, prd = [], []
        for (g, v, s) in keys:
            if v != v1 or (g, v0, s) not in look:
                continue
            ra = look[(g, v1, 0)][0] - look[(g, v0, 0)][0] if (g, v0, 0) in look and (g, v1, 0) in look else None
            if ra is None:
                continue
            real.append(ra)                                  # both seats estimate A's gain (B's view negated)
            dp = look[(g, v1, s)][1] - look[(g, v0, s)][1]
            prd.append(dp if s == 0 else -dp)
        if real:
            real, prd = np.asarray(real), np.asarray(prd)
            out[name] = {"n": int(len(real)), "realized": float(real.mean()), "pred": float(prd.mean()),
                         "ratio": float(prd.sum() / real.sum()) if abs(real.sum()) > 1e-9 else float("nan"),
                         "corr": float(np.corrcoef(real, prd)[0, 1]) if len(real) > 2 and prd.std() > 0 else float("nan")}
    return out


CELL_RE = re.compile(r"^(freeze12|neutral|push) L(\d+) (\S+) hpB")


def m46d_readouts(ext, pred) -> dict:
    lab, fam, seat = ext["label"], ext["fam"], ext["seat"]
    y, y500 = ext["y"], ext["y500"]
    out = {"bias": {}, "classes": {}}
    for f in ("freeze", "lane", "gap"):
        m = (fam == f) & (seat == 0)                        # seat A = the stronger / freezing side (M4.6d)
        out["bias"][f] = float((pred[m] - y[m]).mean())
        out["bias"][f + "_seatB"] = float((pred[(fam == f) & (seat == 1)] - y[(fam == f) & (seat == 1)]).mean())
    nat = np.char.startswith(fam.astype(str), "natural")
    out["natural"] = basic(y500[nat], pred[nat])
    for k in ("natural_lead", "natural_even"):
        m = fam == k
        out["bias"][k] = float((pred[m] - y500[m]).mean())
    cell = {}
    for i, l in enumerate(lab):
        mm = CELL_RE.match(str(l))
        if mm:
            cell[(mm.group(1), int(mm.group(2)), mm.group(3), int(seat[i]))] = i
    cls = {"edge": [], "plaus": [], "large": []}
    for (w, L, name, s), i in cell.items():
        j = cell.get((w, L, "equal", s))
        if j is None or name == "equal":
            continue
        key = {"edge": "edge", "plaus": "plaus", "x1.76": "large"}.get(name)
        if key:
            sg = 1.0 if s == 0 else -1.0                     # both seats estimate the stronger side's premium
            cls[key].append((sg * (y[i] - y[j]), sg * (pred[i] - pred[j])))
    for k, v in cls.items():
        a = np.asarray(v)
        out["classes"][k] = {"n": int(len(a)), "realized": float(a[:, 0].mean()), "pred": float(a[:, 1].mean()),
                             "ratio": float(a[:, 1].sum() / a[:, 0].sum())}
    g = fam == "gap"
    out["classes"]["gap"] = {"n": int(g.sum()), "realized": float(np.mean(np.abs(y[g]))),
                             "pred": float(np.mean(np.sign(y[g]) * pred[g])),
                             "ratio": float(np.sum(np.sign(y[g]) * pred[g]) / np.sum(np.abs(y[g])))}
    big = cls["large"] + [(yy, pp) for yy, pp in zip(y[g], pred[g])]
    a = np.asarray(big)
    sgn = np.sign(a[:, 0])
    out["classes"]["large+gap"] = {"ratio": float(np.sum(sgn * a[:, 1]) / np.sum(np.abs(a[:, 0])))}
    eb = cls["edge"] + cls["plaus"]
    a = np.asarray(eb)
    out["classes"]["edge+plaus"] = {"ratio": float(a[:, 1].sum() / a[:, 0].sum())}
    return out


def per_policy(spec: str, split: str) -> dict:
    e = F.examples(F.load_points(spec))
    ext = F._m46d_examples(spec)
    r = F.roles(e, split)
    P = F.predictions(spec, split)
    P["ex"]["critic"], P["ext"]["critic"] = e["V"], ext["V"]
    test = r == "test"
    out = {}
    for a in ARMS + SEED_ARMS:
        pe, px = P["ex"][a], P["ext"][a]
        rec = {"natural_test": basic(e["y"][test & (e["src"] == 0)], pe[test & (e["src"] == 0)]),
               "con_traj_test": basic(e["y"][test & (e["src"] == 1) & (e["t"] > 0)],
                                      pe[test & (e["src"] == 1) & (e["t"] > 0)])}
        keys, ym, pm, diffs = _start_means(e, test, pe)
        rec["con_start_test"] = basic(ym, pm)
        nf = float(np.mean(diffs ** 2) / 4)                                            # sigma^2 / 2, sigma^2 = E[d^2]/2
        rec["con_start_test"]["noise_floor_mse"] = nf
        # share of the explainable variance (label variance above the noise floor) the predictor captures
        rec["con_start_test"]["explained_share"] = float(1 - (np.mean((pm - ym) ** 2) - nf) / (ym.var() - nf))
        big = test & (np.abs(e["log_duel"]) >= F.STRENGTH_CUT)
        rec["beyond_1.6"] = basic(e["y"][big], pe[big])
        fz = test & (e["src"] == 1) & (e["kind"] == tuple(C.ARCHIVE_SAMPLE).index("freeze"))
        rec["freeze_test"] = basic(e["y"][fz], pe[fz])
        rec["pairs"] = pair_premiums(e, test, pe)
        rec["pairs_freeze"] = pair_premiums(e, fz, pe)
        rec["m46d"] = m46d_readouts(ext, px)
        out[a] = rec
    return out


def _collect(vals: List[float]):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return C.mean_ci(v) if v else (float("nan"), float("nan"))


def aggregate(per: Dict[str, dict]) -> dict:
    """{arm: nested readouts with (mean, half-width) over policies}."""
    specs = list(per)
    arms = list(per[specs[0]])

    def walk(objs):
        o0 = objs[0]
        if isinstance(o0, dict):
            keys = set().union(*[set(o) for o in objs if isinstance(o, dict)])
            return {k: walk([o.get(k) for o in objs if isinstance(o, dict) and k in o]) for k in keys}
        return _collect([o for o in objs if isinstance(o, (int, float))])

    return {a: walk([per[s][a] for s in specs]) for a in arms}


def tail() -> dict:
    from . import labels as LB
    import glob
    rows = defaultdict(list)
    for p in glob.glob(os.path.join(C.CKPT, "tail", "*", "*.json.gz")):
        r = C.load_gz(p)
        G = np.asarray(r["G"])
        rows[r["name"]].append((r["spec"], G, r["cps"]))
    out = {}
    for name, v in sorted(rows.items()):
        cps = v[0][2]
        means = np.asarray([G.mean(0) for _, G, _ in v])            # policies x cps
        j3, j8 = cps.index(300), cps.index(800)
        d = np.concatenate([G[:, j8] - G[:, j3] for _, G, _ in v])   # rollout-level change 300 -> 800
        sd300 = float(np.mean([G[:, j3].std(ddof=1) for _, G, _ in v]))
        out[name] = {"cps": cps, "mean_by_cp": means.mean(0).tolist(),
                     "change_300_800": C.mean_ci(means[:, j8] - means[:, j3]),
                     "change_300_800_rollouts": [float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))],
                     "change_150_300": C.mean_ci(means[:, j3] - means[:, cps.index(150)]),
                     "sd_G300_per_rollout": sd300, "policies": [s for s, _, _ in v]}
    return out


def run() -> dict:
    res = {"meta": C.meta(stage="analysis")}
    fits = {}
    for split in F.SPLITS:
        per = {spec: per_policy(spec, split) for spec in C.POLICIES}
        fits[split] = {"aggregate": aggregate(per), "per_policy": per}
    res["fits"] = fits
    res["tail"] = tail()
    res["probes"] = probes()
    C.save_json("analysis", res)
    return res


__all__ = ["basic", "pair_premiums", "m46d_readouts", "per_policy", "aggregate", "tail", "run"]


# ---------------------------------------------------------------------------------------------- representation probe
def trunk_probe(spec: str) -> dict:
    """Is strength linearly decodable from the trained trunk? Ridge (strength by validation) from the observation, the
    first and the second hidden layer to the example's log duel ratio and to its own HP fraction, held-out R^2 on the
    main split's test groups; and the value readout B refitted with a weak penalty (alpha 1) as a robustness check."""
    import torch
    e = F.examples(F.load_points(spec))
    ext = F._m46d_examples(spec)
    r = F.roles(e, "main")
    tr, va, te = r == "train", r == "val", r == "test"
    net = C.net_of(spec)
    with torch.no_grad():
        x = torch.as_tensor(e["x"])
        h1 = net.trunk[1](net.trunk[0](x)).double().numpy()
        h2 = net.trunk(x).double().numpy()
    feats = {"obs": e["x"].astype(np.float64), "h1": h1, "h2": h2}
    out = {}
    for tgt in ("log_duel", "hp_own"):
        y = e[tgt]
        for name, Fm in feats.items():
            m = F.ridge(Fm[tr], y[tr], np.ones(tr.sum()), Fm[va], y[va], np.ones(va.sum()))
            p = F.ridge_predict(m, Fm[te])
            out[f"{tgt}<-{name}"] = float(1 - ((p - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    w = F.weights(e, tr)
    Fe, Fx = F.trunk(net, e["x"]), F.trunk(net, ext["x"])
    mu, sd = Fe[tr].mean(0), Fe[tr].std(0) + 1e-8
    Z = np.c_[np.ones(tr.sum()), (Fe[tr] - mu) / sd]
    I = np.eye(Z.shape[1])
    I[0, 0] = 0
    beta = np.linalg.solve(Z.T @ (Z * w[tr][:, None]) + I, Z.T @ (w[tr] * e["y"][tr]))
    weak = {"beta": beta, "mu": mu, "sd": sd}
    out["B_alpha1_m46d"] = m46d_readouts(ext, F.ridge_predict(weak, Fx))["classes"]
    # how much the trunk moves for a strength contrast, relative to its natural spread
    nat = e["src"] == 0
    spread = float(np.linalg.norm(h2[nat] - h2[nat].mean(0), axis=1).mean())
    lab, seat = ext["label"], ext["seat"]
    moves = []
    for i, l in enumerate(lab):
        mm = CELL_RE.match(str(l))
        if mm and mm.group(3) == "x1.76" and seat[i] == 0:
            j = [k for k, l2 in enumerate(lab) if seat[k] == 0 and str(l2) == str(l).replace("x1.76", "equal")]
            if j:
                moves.append(float(np.linalg.norm(Fx[i] - Fx[j[0]])))
    out["h2_move_x1.76_over_natural_spread"] = float(np.mean(moves) / spread) if moves else float("nan")
    return out


def probes() -> dict:
    per = {spec: trunk_probe(spec) for spec in C.POLICIES}
    return {"per_policy": per, "aggregate": aggregate({s: {"probe": v} for s, v in per.items()})["probe"]}
