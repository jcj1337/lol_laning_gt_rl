"""Readouts of M4.8 (protocol §6-§7): the pilot, Stage 1 validation and Stage 2 application."""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from ....diagnostics import lane_exit as LX
from . import common as C
from . import families as F
from . import rollouts as R


# ---------------------------------------------------------------------------------------------- pilot
def pilot(workers: int) -> dict:
    """Throughput and the within-state spread of Delta_h (24 natural states at turn 94, K 64): sets K (protocol §4)."""
    states = F.pilot_states()
    js = F.pilot_jobs(states)
    t0 = time.perf_counter()
    R.run(js, workers, "pilot")
    wall = time.perf_counter() - t0
    sd = {h: [] for h in C.WINDOWS}
    turns = elapsed = 0.0
    for j in js:
        d = R.load(j["path"])
        r = d["own"]["r"].astype(float)
        for h in C.WINDOWS:
            sd[h].append(float(LX.deltas(r, h).std(ddof=1)))
        turns += d["_meta"]["turns"]
        elapsed += d["_meta"]["elapsed"]
    out = {"meta": C.meta(stage="pilot", workers=workers), "states": len(js), "K": F.K_PILOT,
           "turns": turns, "wall_s": wall, "turns_per_s_worker": turns / elapsed, "turns_per_s_wall": turns / wall,
           "sd": {}}
    for h in C.WINDOWS:
        v = np.asarray(sd[h])
        q = {p: float(np.percentile(v, p)) for p in (25, 50, 75, 90)}
        need = {f"K_for_{t}": {p: int(np.ceil((1.96 * q[p] / t) ** 2)) for p in (50, 75)} for t in (0.25, 0.5)}
        out["sd"][str(h)] = {"quantiles": q, "max": float(v.max()), **need}
    C.save_json("pilot", out)
    for h in C.WINDOWS:
        x = out["sd"][str(h)]
        print(f"  h {h:2d}: SD p50 {x['quantiles'][50]:.2f} p75 {x['quantiles'][75]:.2f} p90 {x['quantiles'][90]:.2f};"
              f" K for +-0.25 (p50) {x['K_for_0.25'][50]}, for +-0.5 (p50) {x['K_for_0.5'][50]}", flush=True)
    print(f"  throughput {out['turns_per_s_worker']:.0f} turns/s per worker, {out['turns_per_s_wall']:.0f} wall",
          flush=True)
    return out


# ---------------------------------------------------------------------------------------------- per-state rows
SIDES = ("A", "B")
COMP_KEYS = ("farm", "plate", "bounty", "avail", "lost_away", "died", "recalled", "returned", "away", "xpg")


def sign_of(x: str) -> float:
    return 1.0 if x == "A" else -1.0


def pending_items(desc: dict) -> Dict[str, List[int]]:
    """The checkpoint's pending items per kind (seat indices): an absent seat's return, each seat's enemy stock, and
    the next return of a present seat that is due (protocol §6a)."""
    out = {"absent": [], "stock": [], "due": []}
    for s, x in enumerate(SIDES):
        if desc[f"status_{x}"] > 0:
            out["absent"].append(s)
        elif desc.get(f"since_return_{x}", 0) >= C.DUE_TURNS:
            out["due"].append(s)
        if desc[f"stock_{x}"] > 0:
            out["stock"].append(s)
    return out


def resolution(desc: dict, x: Dict[str, np.ndarray], kinds=("absent", "due", "stock")) -> np.ndarray:
    """[K] per continuation: turns until every pending item of the given kinds has resolved (inf if not within T)."""
    p = pending_items(desc)
    K = x["ret"].shape[0]
    t = np.zeros(K)
    for kind in kinds:
        for s in p[kind]:
            t = np.maximum(t, np.asarray(x["stock_t" if kind == "stock" else "ret"], float)[:, s])
    return t


def combo_row(desc: dict, x: Dict[str, np.ndarray], prefix: str = "") -> Dict[str, float]:
    """The readout of one combination of one state (seat A's orientation): per window h the mean change d<h>, its
    standard error se<h> and SD sd<h>, E<h> = D0 + d<h>, the mean event sums per seat (c<h>_<key>_<A|B>), the state after
    h turns, and the resolution times."""
    D0 = desc["D0"]
    r = np.asarray(x["r"], dtype=float)
    K = r.shape[0]
    row: Dict[str, float] = {f"{prefix}K": float(K)}
    for h in C.WINDOWS:
        dl = r[:, :h].sum(1)
        m, sd = float(dl.mean()), float(dl.std(ddof=1))
        row[f"{prefix}d{h}"], row[f"{prefix}sd{h}"], row[f"{prefix}se{h}"] = m, sd, sd / np.sqrt(K)
        row[f"{prefix}E{h}"] = D0 + m
        for key in COMP_KEYS:
            v = np.asarray(x[f"c{h}_{key}"], dtype=float).mean(0)
            row[f"{prefix}c{h}_{key}_A"], row[f"{prefix}c{h}_{key}_B"] = float(v[0]), float(v[1])
        xp = np.asarray(x[f"s{h}_xp"], dtype=float)
        ab = np.asarray(x[f"s{h}_absent"]) > 0
        hp = np.asarray(x[f"s{h}_hp"], dtype=float)
        lv = np.asarray(x[f"s{h}_level"], dtype=float)
        row[f"{prefix}xpd{h}"] = float((xp[:, 0] - xp[:, 1]).mean())
        row[f"{prefix}lvA{h}"], row[f"{prefix}lvB{h}"] = float(lv[:, 0].mean()), float(lv[:, 1].mean())
        row[f"{prefix}awayA{h}"], row[f"{prefix}awayB{h}"] = float(ab[:, 0].mean()), float(ab[:, 1].mean())
        both = ~ab[:, 0] & ~ab[:, 1]
        row[f"{prefix}hpd{h}"] = float((hp[both, 0] - hp[both, 1]).mean()) if both.any() else float("nan")
    ret, stk, got = (np.asarray(x[k], dtype=float) for k in ("ret", "stock_t", "stock_got"))
    for s, sx in enumerate(SIDES):
        for h in C.WINDOWS:
            row[f"{prefix}ret{sx}<={h}"] = float((ret[:, s] <= h).mean())
            row[f"{prefix}stk{sx}<={h}"] = float((stk[:, s] <= h).mean())
        fin = ret[:, s][np.isfinite(ret[:, s])]
        row[f"{prefix}ret{sx}_med"] = float(np.median(ret[:, s]))
        row[f"{prefix}ret{sx}_mean_fin"] = float(fin.mean()) if len(fin) else float("nan")
        row[f"{prefix}stk{sx}_med"] = float(np.median(stk[:, s]))
        g = got[:, s][np.isfinite(got[:, s])]
        row[f"{prefix}got{sx}"] = float(g.mean()) if len(g) else float("nan")
    p = pending_items(desc)
    for name, kinds in (("res", ("absent", "due", "stock")), ("resA", ("absent",)), ("resD", ("due",)),
                        ("resS", ("stock",))):
        if name != "res" and not any(p[k] for k in kinds):
            continue
        res = resolution(desc, x, kinds)
        for h in C.WINDOWS:
            row[f"{prefix}{name}<={h}"] = float((res <= h).mean())
        row[f"{prefix}{name}_med"] = float(np.median(res))
    row[f"{prefix}acct_max"] = float(np.abs(np.asarray(x["acct"])).max())
    return row


def paired_row(x1: Dict[str, np.ndarray], x0: Dict[str, np.ndarray], prefix: str) -> Dict[str, float]:
    """E_h(1) - E_h(0) of two combinations of the same state on common random numbers (D0 equal), per window, with its
    standard error, and the paired component differences per seat."""
    r1, r0 = np.asarray(x1["r"], float), np.asarray(x0["r"], float)
    out = {}
    for h in C.WINDOWS:
        p = LX.paired(r1, r0, h)
        out[f"{prefix}pd{h}"], out[f"{prefix}pse{h}"] = p["diff"], p["se"]
        for key in ("farm", "lost_away", "bounty", "plate", "away", "avail"):
            v1 = np.asarray(x1[f"c{h}_{key}"], float).mean(0)
            v0 = np.asarray(x0[f"c{h}_{key}"], float).mean(0)
            out[f"{prefix}pc{h}_{key}_A"], out[f"{prefix}pc{h}_{key}_B"] = float(v1[0] - v0[0]), float(v1[1] - v0[1])
    return out


def category(desc: dict) -> tuple:
    """(category, X) of a natural checkpoint's pending situation (protocol §5 S2), in priority order:
    both_away, dead (one seat dead), recall_away (one seat away after a Recall), fresh (one seat back <= FRESH_TURNS
    turns ago, the other present and not), due_asym (one seat due, the other back <= 6 turns ago; X = the one back),
    stock_asym (enemy stocks differ by >= 16 HP; X = the larger stock), ordinary (X = A)."""
    st = {x: desc[f"status_{x}"] for x in SIDES}
    sr = {x: desc.get(f"since_return_{x}", 99.0) for x in SIDES}
    away = [x for x in SIDES if st[x] > 0]
    if len(away) == 2:
        return "both_away", "A"
    if len(away) == 1:
        return ("dead" if st[away[0]] == 2 else "recall_away"), away[0]
    fresh = [x for x in SIDES if sr[x] <= C.FRESH_TURNS]
    if len(fresh) == 1:
        return "fresh", fresh[0]
    for x, y in (("A", "B"), ("B", "A")):
        if sr[y] >= C.DUE_TURNS and sr[x] <= 6:
            return "due_asym", x
    if abs(desc["stock_diff"]) >= 16:
        return "stock_asym", ("A" if desc["stock_diff"] > 0 else "B")
    return "ordinary", "A"


CATEGORIES = ("dead", "recall_away", "both_away", "fresh", "due_asym", "stock_asym", "ordinary")


def features(desc: dict) -> Dict[str, float]:
    """Seat A's signed (A - B) checkpoint features for the explained-variance split (protocol §5 S2)."""
    def present(x):
        return desc[f"status_{x}"] == 0

    per = {"recall_away": lambda x: float(desc[f"status_{x}"] == 1),
           "dead": lambda x: float(desc[f"status_{x}"] == 2),
           "returns_in": lambda x: desc[f"returns_in_{x}"],
           "fresh": lambda x: float(present(x) and desc.get(f"since_return_{x}", 99) <= C.FRESH_TURNS),
           "due": lambda x: float(present(x) and desc.get(f"since_return_{x}", 0) >= C.DUE_TURNS),
           "unspent": lambda x: desc[f"unspent_{x}"],
           "stock": lambda x: desc[f"stock_res_{x}"],
           "near": lambda x: desc[f"near_res_{x}"],
           "siege_under_own": lambda x: desc[f"enemy_siege_{x}"] * 0.125,
           "invested": lambda x: desc[f"invested_{x}"],
           "level": lambda x: desc[f"level_{x}"],
           "hp_present": lambda x: desc[f"hp_{x}"] if present(x) else 0.0}
    f = {name: fn("A") - fn("B") for name, fn in per.items()}
    f["D0"] = desc["D0"]
    f["freeze"] = desc["holds_freeze_A"] - desc["holds_freeze_B"]
    return f


PENDING_FEATURES = ("recall_away", "dead", "returns_in", "fresh", "due", "unspent", "stock", "near",
                    "siege_under_own", "freeze")
STRENGTH_FEATURES = ("invested", "level", "hp_present")
LEAD_FEATURES = ("D0",)


def r2(y: np.ndarray, X: np.ndarray) -> float:
    if X.shape[1] == 0 or len(y) < X.shape[1] + 2:
        return float("nan")
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    res = y - Xc @ beta
    return float(1.0 - res.var() / y.var()) if y.var() > 0 else float("nan")


def cv_r2(y: np.ndarray, X: np.ndarray, folds: np.ndarray) -> float:
    """Out-of-fold R^2 of a linear fit with an intercept (folds fixed in advance, e.g. by game index)."""
    y = np.asarray(y, float)
    if X.shape[1] == 0:
        return float("nan")
    pred = np.zeros_like(y)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        Xt = np.column_stack([np.ones(tr.sum()), X[tr]])
        beta, *_ = np.linalg.lstsq(Xt, y[tr], rcond=None)
        pred[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ beta
    return float(1.0 - np.mean((y - pred) ** 2) / y.var()) if y.var() > 0 else float("nan")


def ols(y, x) -> Dict[str, float]:
    """Slope of y on x with an intercept, HC1-robust standard error and 95% normal interval."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 4 or x.var() == 0:
        return {"slope": float("nan"), "se": float("nan"), "ci": float("nan"), "intercept": float("nan"), "n": n,
                "r2": float("nan")}
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    meat = (X * e[:, None] ** 2).T @ X
    V = XtXi @ meat @ XtXi * n / (n - 2)
    se = float(np.sqrt(V[1, 1]))
    return {"slope": float(beta[1]), "se": se, "ci": 1.96 * se, "intercept": float(beta[0]), "n": n,
            "r2": float(1 - e.var() / y.var()) if y.var() > 0 else float("nan")}


def run() -> None:
    from . import apply, validate
    nat = apply.rows()
    val = validate.run(nat)
    apply.run(nat, val)


__all__ = ["pilot", "run", "combo_row", "paired_row", "category", "features", "pending_items", "resolution", "ols",
           "r2", "CATEGORIES", "PENDING_FEATURES", "STRENGTH_FEATURES", "LEAD_FEATURES", "sign_of", "SIDES"]
