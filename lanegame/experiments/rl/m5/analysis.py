"""The pre-registered M5 analysis (docs/m5/m5_preregistration.md §9-§10): contrasts, classifications, variance
decomposition and the resolution of the non-inferiority rule.

Units. A training seed (one matched pair) is the replication unit. For each seed s a family's value is the mean over its
opponents (equal weights) of the per-opponent mean over game pairs of the paired difference om - sp (the same games,
the same chance streams). Intervals are t intervals over the six seeds (df 5). Game-level sampling uncertainty is kept
apart: each seed's value carries its game-level standard error, and the between-seed variance is split into the game
part (the mean squared game SE) and the seed part tau^2 = max(0, s^2 - mean SE^2).

Value: the lead at turn 114 (READ_TURN; the lane-exit readout E_20 at turn 94 in expectation); D0 (the lead at turn
94) and the non-gold panel beside it; the discounted lead over 500 turns (the objective) in the long families.

Classifications (DELTA = 0.25 resources; L, U = the 95% interval):
    gain (P1)            demonstrated: mean >= GAIN_MIN and L > 0 | detected, below materiality: L > 0 |
                         no material gain: U < GAIN_MIN | loss: U < 0 | inconclusive: otherwise
    non-inferiority (P2, robustness families)
                         preserved: L > -DELTA (with "a small loss detected" when also U < 0) |
                         material loss: U < 0 and mean <= -DELTA | loss, materiality unresolved: U < 0 |
                         inconclusive: otherwise (insufficient precision; never reported as preserved)
    exploitability       increase demonstrated: L > 0 and mean >= DELTA (om more exploitable than sp) |
                         decrease demonstrated: U < 0 and mean <= -DELTA | no material difference: L > -DELTA and
                         U < DELTA | inconclusive: otherwise
Headline (the package, P1 with P2a and P2b; style gains and competence losses are reported side by side, never netted):
    1 pays and preserves competence   P1 demonstrated, P2a and P2b preserved
    2 pays at a competence cost       P1 demonstrated, P2a or P2b a material loss
    3 pays, competence unresolved     P1 demonstrated, otherwise
    4 no material gain                P1 not demonstrated, no material loss
    5 harmful                         P1 not demonstrated, a material loss
Secondary claims of "demonstrated" are Holm-adjusted (one-sided seed-level t tests, familywise 0.05); strategic
replication (M5c) holds in a seed when no fixed alternative beats the policy by more than REPLICATION_MARGIN, and
replicates with at least 5 of 6 seeds.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import common as C

DELTA = 0.25
GAIN_MIN = 0.25
REPLICATION_MARGIN = 0.2
REPLICATION_MIN_SEEDS = 5


# ---------------------------------------------------------------------------------------------- statistics
def interval(per_seed: Dict[int, float]) -> dict:
    st = C.seed_stats(per_seed)
    m, h = st["mean"], st["ci"]
    st["L"], st["U"] = (m - h, m + h) if np.isfinite(h) else (float("nan"), float("nan"))
    return st


def classify_gain(st: dict, gain: float = GAIN_MIN) -> str:
    m, L, U = st["mean"], st["L"], st["U"]
    if not np.isfinite(L):
        return "inconclusive"
    if m >= gain and L > 0:
        return "demonstrated"
    if L > 0:
        return "detected, below materiality"
    if U < 0:
        return "loss"
    if U < gain:
        return "no material gain"
    return "inconclusive"


def classify_ni(st: dict, delta: float = DELTA) -> str:
    m, L, U = st["mean"], st["L"], st["U"]
    if not np.isfinite(L):
        return "inconclusive"
    if L > -delta:
        return "preserved (a small loss detected)" if U < 0 else "preserved"
    if U < 0 and m <= -delta:
        return "material loss"
    if U < 0:
        return "loss, materiality unresolved"
    return "inconclusive"


def classify_exploit(st: dict, delta: float = DELTA) -> str:
    m, L, U = st["mean"], st["L"], st["U"]
    if not np.isfinite(L):
        return "inconclusive"
    if L > 0 and m >= delta:
        return "increase demonstrated"
    if U < 0 and m <= -delta:
        return "decrease demonstrated"
    if L > -delta and U < delta:
        return "no material difference"
    return "inconclusive"


def headline(p1: str, p2a: str, p2b: str) -> dict:
    gain = p1 == "demonstrated"
    loss = "material loss" in (p2a, p2b)
    kept = p2a.startswith("preserved") and p2b.startswith("preserved")
    if gain and kept:
        k = 1
    elif gain and loss:
        k = 2
    elif gain:
        k = 3
    elif not loss:
        k = 4
    else:
        k = 5
    names = {1: "pays and preserves competence", 2: "pays at a competence cost", 3: "pays, competence unresolved",
             4: "no material gain", 5: "harmful"}
    return {"outcome": k, "name": names[k], "P1": p1, "P2a": p2a, "P2b": p2b}


def t_pvalue_one_sided(st: dict, null: float = 0.0, side: str = "greater") -> float:
    """One-sided p of a seed-level t test (df n - 1), via the m46a helper's two-sided value."""
    from ..m46a.common import t_pvalue
    v = np.asarray([x for x in st["per_seed"].values()], dtype=float) - null
    if len(v) < 2:
        return float("nan")
    p2 = t_pvalue(v)
    return p2 / 2 if (v.mean() > 0) == (side == "greater") else 1 - p2 / 2


def holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """Holm-Bonferroni: which claims stay significant at familywise alpha."""
    items = sorted(((p, k) for k, p in pvals.items() if np.isfinite(p)))
    out, m = {k: False for k in pvals}, len(items)
    for i, (p, k) in enumerate(items):
        if p <= alpha / (m - i):
            out[k] = True
        else:
            break
    return out


def variance_split(per_seed_mean: Dict[int, float], per_seed_se: Dict[int, float]) -> dict:
    v = np.asarray(list(per_seed_mean.values()), dtype=float)
    se = np.asarray([per_seed_se[k] for k in per_seed_mean], dtype=float)
    s2 = float(v.var(ddof=1)) if len(v) > 1 else float("nan")
    g2 = float(np.mean(se ** 2))
    return {"between_seed_sd": math.sqrt(s2) if np.isfinite(s2) else float("nan"), "mean_game_se": float(np.mean(se)),
            "tau": math.sqrt(max(0.0, s2 - g2)) if np.isfinite(s2) else float("nan"),
            "game_share_of_variance": g2 / s2 if s2 and np.isfinite(s2) else float("nan")}


def ni_resolution(true_effect: float, sd: float, n: int, delta: float = DELTA, sims: int = 20_000,
                  seed: int = C.READY_ROOT + 9) -> dict:
    """Monte Carlo: the probability of each non-inferiority class with n seeds whose per-seed values are
    N(true_effect, sd^2) (sd = the total per-seed SD: seed and game parts together)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(true_effect, sd, size=(sims, n))
    m, s = x.mean(1), x.std(1, ddof=1)
    h = C.t95(n - 1) * s / math.sqrt(n)
    L, U = m - h, m + h
    pres = L > -delta
    mat = (~pres) & (U < 0) & (m <= -delta)
    lossu = (~pres) & (U < 0) & (m > -delta)
    return {"preserved": float(pres.mean()), "material_loss": float(mat.mean()),
            "loss_unresolved": float(lossu.mean()), "inconclusive": float(1 - pres.mean() - mat.mean() - lossu.mean()),
            "mean_half_width": float(h.mean())}


# ---------------------------------------------------------------------------------------------- family readers
def _cell_values(path: str, focal: int, readout: str) -> np.ndarray:
    from . import games as G
    d = G.load(path)
    R = d[f"f{focal}__r"]
    if readout == "lead114":
        return G.lead_at(R)
    if readout == "lead94":
        return G.lead_at(R, G.LANE_END)
    if readout == "discounted":
        return G.discounted(R)
    raise ValueError(readout)


def family_contrast(paths: Sequence[str], a: int = 0, b: int = 1, readout: str = "lead114") -> dict:
    """One seed's family value: mean over opponents of mean_pairs(focal a - focal b), with its game-level SE."""
    means, vars_ = [], []
    for p in paths:
        d = _cell_values(p, a, readout) - _cell_values(p, b, readout)
        means.append(float(d.mean()))
        vars_.append(float(d.var(ddof=1) / len(d)))
    k = len(means)
    return {"mean": float(np.mean(means)), "se": math.sqrt(sum(vars_)) / k, "per_opponent": means, "n_opp": k}


def own_lead(paths: Sequence[str], focal: int = 0, readout: str = "lead114") -> dict:
    """One seed's head-to-head value: the focal's mean lead over pairs (averaged over opponents)."""
    means, vars_ = [], []
    for p in paths:
        v = _cell_values(p, focal, readout)
        means.append(float(v.mean()))
        vars_.append(float(v.var(ddof=1) / len(v)))
    return {"mean": float(np.mean(means)), "se": math.sqrt(sum(vars_)) / len(means), "per_opponent": means}


def replication(per_seed_gap: Dict[int, float], margin: float = REPLICATION_MARGIN,
                min_seeds: int = REPLICATION_MIN_SEEDS) -> dict:
    """A strategic verdict replicates when, in at least min_seeds seeds, no fixed alternative beats the policy by more
    than `margin` (per_seed_gap = max over alternatives of the family mean of alternative - policy)."""
    holds = {s: bool(g <= margin) for s, g in per_seed_gap.items()}
    n = sum(holds.values())
    return {"holds": holds, "n_holds": n, "replicates": n >= min_seeds, "margin": margin}


# ---------------------------------------------------------------------------------------------- the campaign analysis
def analyse(phase: str = "main") -> dict:
    """The pre-registered M5d analysis from the family records (M5c's battery and cells analyses reuse the M4.5c /
    M4.6 analysis code on the M5 records; `report` assembles both)."""
    from . import evaluate as EV
    root = os.path.join(EV.EOUT, "families")
    res: Dict[str, dict] = {}
    per: Dict[str, Dict[int, dict]] = {}
    for s in C.SEEDS:
        fam = EV.families(s)
        paths = {name: [os.path.join(root, f"s{s}", f"{name}__{EV._opp_tag(o)}.npz") for o in opps]
                 for name, opps in fam.items()}
        if not all(os.path.exists(p) for v in paths.values() for p in v):
            continue
        for name, ps in paths.items():
            per.setdefault(name, {})[s] = family_contrast(ps)
        h = os.path.join(EV.EOUT, "h2h", f"s{s}", "om_vs_sp.npz")
        if os.path.exists(h):
            per.setdefault("H2H", {})[s] = own_lead([h])
    if not per:
        raise SystemExit("no complete M5d family records yet")
    S = {s: 0.5 * (per["S_in"][s]["mean"] + per["S_out"][s]["mean"]) for s in per["S_in"]}
    Cc = {s: 0.5 * (per["C_hist"][s]["mean"] + per["C_cross"][s]["mean"]) for s in per["C_hist"]}
    P1, P2a = interval(S), interval(Cc)
    P2b = interval({s: v["mean"] for s, v in per.get("H2H", {}).items()})
    res["P1"] = {**P1, "class": classify_gain(P1)}
    res["P2a"] = {**P2a, "class": classify_ni(P2a)}
    res["P2b"] = {**P2b, "class": classify_ni(P2b)}
    res["headline"] = headline(res["P1"]["class"], res["P2a"]["class"], res["P2b"]["class"])
    res["families"] = {name: {**interval({s: v["mean"] for s, v in d.items()}),
                              "variance": variance_split({s: v["mean"] for s, v in d.items()},
                                                         {s: v["se"] for s, v in d.items()})}
                       for name, d in per.items()}
    res["secondary"] = secondary(per)
    res["meta"] = C.meta(stage="analyse", phase=phase)
    C.save_json(f"analysis_{phase}", res)
    return res


def secondary(per: Dict[str, Dict[int, dict]]) -> dict:
    """S1-S7 of the pre-registration (§9) from the M5d and attacker records that exist; Holm over the demonstrated
    claims. S8 (strategic replication and the cross-arm M5c contrasts) is assembled from the M4.5c / M4.6 analysis code
    on the M5c records (`report`)."""
    from . import evaluate as EV
    from . import attackers as AT
    out: Dict[str, dict] = {}
    if "M" in per:
        st = interval({s: v["mean"] for s, v in per["M"].items()})
        out["S1"] = {**st, "class": classify_ni(st), "p": t_pvalue_one_sided(st, -DELTA, "greater")}
    long_c, long_h, xs, mid1, mid2 = {}, {}, {}, {}, {}
    for s in C.SEEDS:
        fam = EV.families(s)
        lp = {n: [os.path.join(EV.EOUT, "long", f"s{s}", f"{n}__{EV._opp_tag(o)}.npz") for o in fam[n]]
              for n in ("C_hist", "C_cross")}
        if all(os.path.exists(p) for v in lp.values() for p in v):
            long_c[s] = 0.5 * (family_contrast(lp["C_hist"], readout="discounted")["mean"] +
                               family_contrast(lp["C_cross"], readout="discounted")["mean"])
        p = os.path.join(EV.EOUT, "h2h", f"s{s}", "om_vs_sp_long.npz")
        if os.path.exists(p):
            long_h[s] = own_lead([p], readout="discounted")["mean"]
        xp = [os.path.join(EV.EOUT, "h2h", f"s{s}", f"om_vs_sp{q}.npz") for q in C.SEEDS if q != s]
        if all(os.path.exists(p) for p in xp):
            xs[s] = own_lead(xp)["mean"]
        mp = {n: [os.path.join(EV.EOUT, "mid", f"s{s}", f"{n}__{EV._opp_tag(o)}.npz") for o in fam[n]]
              for n in ("S_in", "S_out", "C_hist")}
        if all(os.path.exists(p) for v in mp.values() for p in v):
            mid1[s] = 0.5 * (family_contrast(mp["S_in"])["mean"] + family_contrast(mp["S_out"])["mean"])
            mid2[s] = family_contrast(mp["C_hist"])["mean"]
    for key, d, kind in (("S2", long_c, "ni"), ("S3", long_h, "ni"), ("S4", xs, "ni"), ("S5_P1", mid1, "gain"),
                         ("S5_P2a", mid2, "ni")):
        if d:
            st = interval(d)
            out[key] = {**st, "class": classify_ni(st) if kind == "ni" else classify_gain(st),
                        "p": t_pvalue_one_sided(st, -DELTA if kind == "ni" else 0.0, "greater")}
    exp = {}
    for kind in AT.TARGET_KINDS:
        for s in C.SEEDS:
            p = os.path.join(AT.AOUT, "eval", "main", f"{kind}_s{s}__target.npz")
            if os.path.exists(p):
                exp.setdefault(kind, {})[s] = own_lead([p], readout="lead114")["mean"]
    if "om" in exp and "sp" in exp:
        common = sorted(set(exp["om"]) & set(exp["sp"]))
        st = interval({s: exp["om"][s] - exp["sp"][s] for s in common})
        out["S6"] = {**st, "class": classify_exploit(st), "p": t_pvalue_one_sided(st, 0.0, "greater"),
                     "exploitability": {k: interval(v) for k, v in exp.items()}}
    if "om" in exp and "om_table" in exp:
        common = sorted(set(exp["om"]) & set(exp["om_table"]))
        st = interval({s: exp["om"][s] - exp["om_table"][s] for s in common})
        out["S7"] = {**st, "class": classify_exploit(st), "p": t_pvalue_one_sided(st, 0.0, "greater")}
    out["holm"] = holm({k: v["p"] for k, v in out.items() if isinstance(v, dict) and "p" in v})
    return out


__all__ = ["DELTA", "GAIN_MIN", "interval", "classify_gain", "classify_ni", "classify_exploit", "headline", "holm",
           "secondary",
           "variance_split", "ni_resolution", "family_contrast", "own_lead", "replication", "analyse",
           "t_pvalue_one_sided"]
