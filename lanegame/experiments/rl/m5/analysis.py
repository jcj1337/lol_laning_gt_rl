"""The pre-registered M5 analysis (docs/m5/m5_preregistration.md §9-§10, as amended by amendment 1, 2026-10-17):
endpoints, classifications, tests, multiplicity, completeness, the lane-exit panel, monitoring.

Units. A training seed (one matched pair) is the replication unit. For each seed s a family's value is the mean over its
opponents (equal weights) of the per-opponent mean over game pairs of the paired difference om - sp (the same games,
the same chance streams). Intervals are t intervals over the six seeds (df 5). Game-level sampling uncertainty is kept
apart: each seed's value carries its game-level standard error, and the between-seed variance is split into the game
part (the mean squared game SE) and the seed part tau^2 = max(0, s^2 - mean SE^2).

Value: the lead at turn 114 (READ_TURN; the lane-exit readout E_20 at turn 94 in expectation); D0 (the lead at turn
94) and the non-gold panel beside it; the discounted lead over 500 turns (the objective) in the long families.

Completeness (amendment 1). An endpoint is classified only when the records of all six seeds exist. Otherwise it is
reported as "partial (n/6)" with the missing records listed and no class; a missing record never shrinks the replication
count or the multiplicity family. The headline needs P1, P2a and P2b complete.

Classifications (DELTA = GAIN_MIN = 0.25 resources; L, U = the 95% interval):
    gain (P1, S5a)       demonstrated: mean >= GAIN_MIN and L > 0
                         detected, below materiality: L > 0 and U < GAIN_MIN          (a material gain ruled out)
                         detected, materiality unresolved: L > 0, mean < GAIN_MIN <= U
                         loss: U < 0                                                  (a material gain ruled out)
                         no material gain: U < GAIN_MIN, L <= 0                       (a material gain ruled out)
                         inconclusive: L <= 0 and U >= GAIN_MIN (a material gain neither shown nor ruled out)
    non-inferiority (P2a, P2b, S1-S4, S5b)
                         preserved: L > -DELTA (with "a small loss detected" when also U < 0) |
                         material loss: U < 0 and mean <= -DELTA | loss, materiality unresolved: U < 0 |
                         inconclusive: otherwise (insufficient precision; never reported as preserved)
    exploitability (S6)  increase demonstrated: L > 0 and mean >= DELTA | decrease demonstrated: U < 0 and mean <= -DELTA |
                         no material difference: L > -DELTA and U < DELTA | inconclusive: otherwise
Headline (P1 with P2a and P2b; style gains and competence results are reported side by side, never netted). Competence:
preserved (P2a and P2b preserved) | material loss (either) | loss detected, materiality unresolved (either) | unresolved.
    1 pays and preserves competence   P1 demonstrated, competence preserved
    2 pays at a competence cost       P1 demonstrated, a material competence loss
    3 pays, competence unresolved     P1 demonstrated, otherwise
    4 no material gain                P1 rules out a material gain (U < GAIN_MIN), no material competence loss
    5 harmful                         P1 not demonstrated, a material competence loss
    6 gain not demonstrated           P1 inconclusive or "detected, materiality unresolved", no material competence loss
The headline covers the primary M5d endpoints only; it says nothing about strategic replication (S8), the limitation
cells or exploitability (S6), which are reported with their own status.

Secondary claims (amendment 1). Every permitted claim of a secondary endpoint is a one-sided seed-level t test (df 5)
fixed in advance (SECONDARY_HYPOTHESES): non-inferiority endpoints "preserved" (H0 mean <= -DELTA) and "loss" (H0 mean
>= 0); S5a "gain" (H0 mean <= 0) and "material gain ruled out" (H0 mean >= GAIN_MIN); S6 "increase" (H0 mean <= 0),
"decrease" (H0 mean >= 0) and "equivalence" (two one-sided tests, H0 |mean| >= DELTA, p = the larger one-sided p). The
15 hypotheses form ONE Holm family (familywise 0.05). A claim is made only when its interval condition holds AND Holm
rejects its hypothesis; the unadjusted interval class is reported beside it. Holm is final only when every hypothesis has
a p-value (a stage the user declines enters with p = 1, which keeps the error control); before that a claim is
"secured" when p <= 0.05 / 15 (Holm rejects it whatever the pending p-values are), otherwise "pending". S8 (strategic
replication and the cross-arm M5c contrasts, `strategic`) is descriptive and is not in the family. The optional
adaptation ablation (om_table attackers) is exploratory and not in the family.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import common as C

DELTA = 0.25
GAIN_MIN = 0.25
ALPHA = 0.05
REPLICATION_MARGIN = 0.2
REPLICATION_MIN_SEEDS = 5
N_SEEDS = len(C.SEEDS)
GAIN_RULED_OUT = ("detected, below materiality", "loss", "no material gain")


# ---------------------------------------------------------------------------------------------- statistics
def interval(per_seed: Dict[int, float]) -> dict:
    st = C.seed_stats(per_seed)
    m, h = st["mean"], st["ci"]
    st["L"], st["U"] = (m - h, m + h) if np.isfinite(h) else (float("nan"), float("nan"))
    v = np.asarray([x for x in per_seed.values() if np.isfinite(x)], dtype=float)
    st["se"] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
    return st


def classify_gain(st: dict, gain: float = GAIN_MIN) -> str:
    m, L, U = st["mean"], st["L"], st["U"]
    if not (np.isfinite(L) and np.isfinite(U)):
        return "inconclusive"
    if L > 0:
        if m >= gain:
            return "demonstrated"
        return "detected, below materiality" if U < gain else "detected, materiality unresolved"
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


CLASSIFIERS = {"gain": classify_gain, "ni": classify_ni, "exploit": classify_exploit}


def competence(p2a: str, p2b: str) -> str:
    if p2a.startswith("preserved") and p2b.startswith("preserved"):
        return "preserved"
    if "material loss" in (p2a, p2b):
        return "material loss"
    if "loss, materiality unresolved" in (p2a, p2b):
        return "loss detected, materiality unresolved"
    return "unresolved"


HEADLINES = {1: "pays and preserves competence", 2: "pays at a competence cost", 3: "pays, competence unresolved",
             4: "no material gain", 5: "harmful", 6: "gain not demonstrated"}


def headline(p1: str, p2a: str, p2b: str) -> dict:
    """The package outcome from the three primary classes (module docstring). An inconclusive P1 is never "no material
    gain"; the competence status is reported separately in every outcome."""
    comp = competence(p2a, p2b)
    if p1 == "demonstrated":
        gain = "demonstrated"
        k = 1 if comp == "preserved" else (2 if comp == "material loss" else 3)
    else:
        gain = "a material gain ruled out" if p1 in GAIN_RULED_OUT else "not demonstrated (inconclusive)"
        k = 5 if comp == "material loss" else (4 if p1 in GAIN_RULED_OUT else 6)
    statement = {1: "the exploitation gain is demonstrated and competence is preserved",
                 2: "the exploitation gain is demonstrated, with a material competence loss",
                 3: f"the exploitation gain is demonstrated; competence {comp}",
                 4: f"a material exploitation gain is ruled out; competence {comp}",
                 5: "no exploitation gain is demonstrated, and a material competence loss is",
                 6: f"a material exploitation gain is neither demonstrated nor ruled out; competence {comp}"}[k]
    return {"outcome": k, "name": HEADLINES[k], "gain": gain, "competence": comp, "statement": statement,
            "P1": p1, "P2a": p2a, "P2b": p2b,
            "scope": "the primary M5d endpoints (P1, P2a, P2b) only: not strategic replication (S8), the limitation "
                     "cells or exploitability (S6)"}


def _t_sf(t: float, df: int) -> float:
    from scipy import stats
    return float(stats.t.sf(t, df))


def p_one_sided(st: dict, null: float, side: str) -> float:
    """One-sided seed-level t test: side ">" tests H0 mean <= null, side "<" tests H0 mean >= null."""
    v = np.asarray([x for x in st["per_seed"].values() if np.isfinite(x)], dtype=float)
    n = len(v)
    if n < 2:
        return float("nan")
    se = v.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return 0.0 if (v.mean() - null > 0) == (side == ">") and v.mean() != null else 1.0
    t = (v.mean() - null) / se
    return _t_sf(t, n - 1) if side == ">" else _t_sf(-t, n - 1)


def p_equivalence(st: dict, delta: float = DELTA) -> float:
    """Two one-sided tests of H0 |mean| >= delta: the larger of the two one-sided p-values."""
    return max(p_one_sided(st, -delta, ">"), p_one_sided(st, delta, "<"))


def t_pvalue_one_sided(st: dict, null: float = 0.0, side: str = "greater") -> float:
    return p_one_sided(st, null, ">" if side in ("greater", ">") else "<")


def holm(pvals: Dict[str, float], alpha: float = ALPHA) -> Dict[str, bool]:
    """Holm-Bonferroni over a COMPLETE family: which hypotheses are rejected at familywise alpha. A missing p-value
    is an error (amendment 1: missing results never shrink the family; use `holm_status` for partial families)."""
    if any(not np.isfinite(p) for p in pvals.values()):
        raise ValueError("Holm needs every hypothesis of the family; use holm_status for a partial family")
    items = sorted(((p, k) for k, p in pvals.items()))
    out, m = {k: False for k in pvals}, len(items)
    for i, (p, k) in enumerate(items):
        if p <= alpha / (m - i):
            out[k] = True
        else:
            break
    return out


def holm_status(pvals: Dict[str, Optional[float]], alpha: float = ALPHA) -> dict:
    """Holm over the registered family with possibly pending members (p None / nan): final when complete, otherwise
    "secured" for p <= alpha / m (rejected whatever the pending p-values are) and "pending" for the rest."""
    m = len(pvals)
    have = {k: p for k, p in pvals.items() if p is not None and np.isfinite(p)}
    if len(have) == m:
        rej = holm(have, alpha)
        return {"final": True, "m": m, "status": {k: "rejected" if rej[k] else "not rejected" for k in pvals}}
    st = {}
    for k in pvals:
        if k not in have:
            st[k] = "pending (no data)"
        elif have[k] <= alpha / m:
            st[k] = "secured"                              # rejected whatever the pending p-values are
        elif have[k] > alpha:
            st[k] = "not rejectable"                       # Holm never rejects p > alpha, whatever is pending
        else:
            st[k] = "pending (family incomplete)"
    return {"final": False, "m": m, "pending": sorted(k for k in pvals if k not in have), "status": st}


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


def replication(per_seed_gap: Dict[int, float], margin: float = REPLICATION_MARGIN,
                min_seeds: int = REPLICATION_MIN_SEEDS, expected: int = N_SEEDS) -> dict:
    """A strategic verdict replicates when, in at least min_seeds seeds, no fixed alternative beats the policy by more
    than `margin` (per_seed_gap = max over alternatives of the family mean of alternative - policy). Final only with
    all `expected` seeds; with fewer it is "partial", and a missing seed is never counted as holding."""
    holds = {s: bool(g <= margin) for s, g in per_seed_gap.items() if g is not None and np.isfinite(g)}
    n = sum(holds.values())
    complete = len(holds) == expected
    if complete:
        verdict = "replicates" if n >= min_seeds else "does not replicate"
    else:
        missing = expected - len(holds)
        verdict = ("replicates" if n >= min_seeds else
                   "does not replicate" if n + missing < min_seeds else f"partial ({len(holds)}/{expected} seeds)")
    return {"holds": holds, "n_holds": n, "n_seeds": len(holds), "complete": complete,
            "replicates": verdict == "replicates", "verdict": verdict, "margin": margin}


# ---------------------------------------------------------------------------------------------- record readers
def _focal_values(d: dict, focal: int, readout: str) -> np.ndarray:
    from . import games as G
    R = d[f"f{focal}__r"]
    if readout == "lead114":
        return G.lead_at(R)
    if readout == "lead94":
        return G.lead_at(R, G.LANE_END)
    if readout == "discounted":
        return G.discounted(R)
    raise ValueError(readout)


_CACHE: Dict[str, dict] = {}


def _load(path: str) -> dict:
    if path not in _CACHE:
        from . import games as G
        _CACHE[path] = G.load(path)
    return _CACHE[path]


def clear_cache() -> None:
    _CACHE.clear()


def family_contrast(paths: Sequence[str], a: int = 0, b: int = 1, readout: str = "lead114") -> dict:
    """One seed's family value: mean over opponents of mean_pairs(focal a - focal b), with its game-level SE."""
    means, vars_ = [], []
    for p in paths:
        d = _load(p)
        x = _focal_values(d, a, readout) - _focal_values(d, b, readout)
        means.append(float(x.mean()))
        vars_.append(float(x.var(ddof=1) / len(x)))
    k = len(means)
    return {"mean": float(np.mean(means)), "se": math.sqrt(sum(vars_)) / k, "per_opponent": means, "n_opp": k}


def own_lead(paths: Sequence[str], focal: int = 0, readout: str = "lead114") -> dict:
    """One seed's head-to-head value: the focal's mean lead over pairs (averaged over opponents)."""
    means, vars_ = [], []
    for p in paths:
        v = _focal_values(_load(p), focal, readout)
        means.append(float(v.mean()))
        vars_.append(float(v.var(ddof=1) / len(v)))
    return {"mean": float(np.mean(means)), "se": math.sqrt(sum(vars_)) / len(means), "per_opponent": means}


def _half(a: dict, b: dict) -> dict:
    return {"mean": 0.5 * (a["mean"] + b["mean"]), "se": 0.5 * math.sqrt(a["se"] ** 2 + b["se"] ** 2)}


# ---------------------------------------------------------------------------------------------- the endpoints
def seed_paths(s: int, root: Optional[str] = None) -> Dict[str, List[str]]:
    """Every M5d record of seed s by group (the job design of `evaluate.family_jobs`)."""
    from . import evaluate as EV
    root = root or EV.EOUT
    fam = EV.families(s)
    out = {}
    for name, opps in fam.items():
        out[f"fam:{name}"] = [os.path.join(root, "families", f"s{s}", f"{name}__{EV._opp_tag(o)}.npz") for o in opps]
    for name in ("C_hist", "C_cross"):
        out[f"long:{name}"] = [os.path.join(root, "long", f"s{s}", f"{name}__{EV._opp_tag(o)}.npz") for o in fam[name]]
    out["h2h"] = [os.path.join(root, "h2h", f"s{s}", "om_vs_sp.npz")]
    out["h2h_long"] = [os.path.join(root, "h2h", f"s{s}", "om_vs_sp_long.npz")]
    out["xseed"] = [os.path.join(root, "h2h", f"s{s}", f"om_vs_sp{p}.npz") for p in C.SEEDS if p != s]
    for name in ("S_in", "S_out", "C_hist"):
        out[f"mid:{name}"] = [os.path.join(root, "mid", f"s{s}", f"{name}__{EV._opp_tag(o)}.npz") for o in fam[name]]
    out["mid:h2h"] = [os.path.join(root, "mid", f"s{s}", "h2h.npz")]
    return out


def attack_paths(s: int, root: Optional[str] = None) -> Dict[str, List[str]]:
    from . import attackers as AT
    root = root or os.path.join(AT.AOUT, "eval", "main")
    return {f"attack:{k}": [os.path.join(root, f"{k}_s{s}__target.npz")] for k in AT.TARGET_KINDS}


# name: (groups, value function of {group: paths}, kind, role, description)
EndpointFn = Callable[[Dict[str, List[str]]], dict]
ENDPOINTS: Dict[str, Tuple[Tuple[str, ...], EndpointFn, str, str, str]] = {
    "P1": (("fam:S_in", "fam:S_out"),
           lambda g: _half(family_contrast(g["fam:S_in"]), family_contrast(g["fam:S_out"])),
           "gain", "primary", "exploitation gain, om - sp, 1/2 S_in + 1/2 S_out, lead at 114"),
    "P2a": (("fam:C_hist", "fam:C_cross"),
            lambda g: _half(family_contrast(g["fam:C_hist"]), family_contrast(g["fam:C_cross"])),
            "ni", "primary", "competence against competent opponents, om - sp, 1/2 C_hist + 1/2 C_cross, lead at 114"),
    "P2b": (("h2h",), lambda g: own_lead(g["h2h"]), "ni", "primary",
            "competence head to head, om:s vs sp:s, om's lead at 114 (128 pairs)"),
    "S1": (("fam:M",), lambda g: family_contrast(g["fam:M"]), "ni", "secondary",
           "misspecified opponents M (imitate-then-switch included), om - sp, lead at 114"),
    "S2": (("long:C_hist", "long:C_cross"),
           lambda g: _half(family_contrast(g["long:C_hist"], readout="discounted"),
                           family_contrast(g["long:C_cross"], readout="discounted")),
           "ni", "secondary", "long-horizon competence, 1/2 C_hist + 1/2 C_cross, 500 turns, discounted lead"),
    "S3": (("h2h_long",), lambda g: own_lead(g["h2h_long"], readout="discounted"), "ni", "secondary",
           "long-horizon head to head, om:s vs sp:s, 500 turns, discounted lead"),
    "S4": (("xseed",), lambda g: own_lead(g["xseed"]), "ni", "secondary",
           "cross-seed head to head, om:s vs sp:p (mean over the five p != s), lead at 114"),
    "S5a": (("mid:S_in", "mid:S_out"),
            lambda g: _half(family_contrast(g["mid:S_in"]), family_contrast(g["mid:S_out"])),
            "gain", "secondary", "midpoint (16.384M) exploitation gain, 1/2 S_in + 1/2 S_out, lead at 114 (as P1)"),
    "S5b": (("mid:C_hist",), lambda g: family_contrast(g["mid:C_hist"]), "ni", "secondary",
            "midpoint (16.384M) competence against HISTORICAL competent opponents only (C_hist; not P2a: the cross-arm "
            "finals C_cross are not played at the midpoint), lead at 114"),
    "S5_h2h": (("mid:h2h",), lambda g: own_lead(g["mid:h2h"]), "ni", "descriptive",
               "midpoint head to head, om@16.384M vs sp@16.384M (descriptive, not a claim)"),
    "S6": (("attack:om", "attack:sp"),
           lambda g: {"mean": own_lead(g["attack:om"])["mean"] - own_lead(g["attack:sp"])["mean"],
                      "se": math.hypot(own_lead(g["attack:om"])["se"], own_lead(g["attack:sp"])["se"])},
           "exploit", "secondary", "exploitability om - sp: each core attacker's lead at 114 against its own target"),
    "X_table": (("attack:om", "attack:om_table"),
                lambda g: {"mean": own_lead(g["attack:om"])["mean"] - own_lead(g["attack:om_table"])["mean"],
                           "se": math.hypot(own_lead(g["attack:om"])["se"], own_lead(g["attack:om_table"])["se"])},
                "exploit", "exploratory", "OPTIONAL adaptation ablation: exploitability of om - of om_table "
                "(exploratory; not a test of deliberate deception)"),
}

SECONDARY_HYPOTHESES: Tuple[Tuple[str, str, str, float], ...] = (
    ("S1", "preserved", ">", -DELTA), ("S1", "loss", "<", 0.0),
    ("S2", "preserved", ">", -DELTA), ("S2", "loss", "<", 0.0),
    ("S3", "preserved", ">", -DELTA), ("S3", "loss", "<", 0.0),
    ("S4", "preserved", ">", -DELTA), ("S4", "loss", "<", 0.0),
    ("S5a", "gain", ">", 0.0), ("S5a", "material gain ruled out", "<", GAIN_MIN),
    ("S5b", "preserved", ">", -DELTA), ("S5b", "loss", "<", 0.0),
    ("S6", "increase", ">", 0.0), ("S6", "decrease", "<", 0.0), ("S6", "equivalence", "tost", DELTA),
)


def hyp_key(ep: str, h: str) -> str:
    return f"{ep}: {h}"


def _interval_holds(st: dict, h: str, side: str, null: float) -> bool:
    if side == "tost":
        return bool(st["L"] > -null and st["U"] < null)
    return bool(st["L"] > null) if side == ">" else bool(st["U"] < null)


def adjusted_class(kind: str, st: dict, supported: Dict[str, bool]) -> str:
    """The class of a secondary endpoint from its Holm-supported claims only (module docstring)."""
    m = st["mean"]
    if kind == "ni":
        pres, loss = supported.get("preserved", False), supported.get("loss", False)
        if pres and loss:
            return "preserved (a small loss detected)"
        if pres:
            return "preserved"
        if loss:
            return "material loss" if m <= -DELTA else "loss, materiality unresolved"
        return "inconclusive"
    if kind == "gain":
        g, r = supported.get("gain", False), supported.get("material gain ruled out", False)
        if g and m >= GAIN_MIN:
            return "demonstrated"
        if g:
            return "detected, below materiality" if r else "detected, materiality unresolved"
        return "no material gain" if r else "inconclusive"
    if kind == "exploit":
        if supported.get("increase") and m >= DELTA:
            return "increase demonstrated"
        if supported.get("decrease") and m <= -DELTA:
            return "decrease demonstrated"
        if supported.get("equivalence"):
            return "no material difference"
        return "inconclusive"
    raise ValueError(kind)


def endpoint(name: str, paths_of: Callable[[int], Dict[str, List[str]]], seeds: Sequence[int] = C.SEEDS) -> dict:
    """One endpoint over the seeds: per-seed values (only for seeds whose records are all present), status complete /
    partial / missing with the missing records, and the interval and class only when complete."""
    groups, fn, kind, role, desc = ENDPOINTS[name]
    per, se, missing = {}, {}, {}
    for s in seeds:
        g = paths_of(s)
        need = [p for grp in groups for p in g.get(grp, [])]
        miss = [p for p in need if not os.path.exists(p)]
        if miss or not need:
            missing[s] = miss or ["(no records defined)"]
            continue
        v = fn(g)
        per[s], se[s] = v["mean"], v["se"]
    n = len(per)
    out = {"name": name, "kind": kind, "role": role, "description": desc, "n_seeds": n, "expected": len(seeds),
           "per_seed": {str(s): v for s, v in per.items()}, "game_se": {str(s): v for s, v in se.items()},
           "missing": {str(s): v[:5] + ([f"... {len(v) - 5} more"] if len(v) > 5 else []) for s, v in missing.items()}}
    if n == 0:
        out["status"] = "missing"
        out["class"] = "not available"
        return out
    st = interval(per)
    out.update({k: st[k] for k in ("mean", "ci", "L", "U", "se", "sd", "pos", "neg")})
    out["variance"] = variance_split(per, se)
    if n == len(seeds):
        out["status"] = "complete"
        out["class"] = CLASSIFIERS[kind](st) if role != "descriptive" else f"descriptive ({CLASSIFIERS[kind](st)})"
    else:
        out["status"] = f"partial ({n}/{len(seeds)})"
        out["class"] = "partial: not classified"
    return out


def _st(e: dict) -> dict:
    return {"mean": e["mean"], "L": e["L"], "U": e["U"], "per_seed": e["per_seed"]}


def secondary(eps: Dict[str, dict], declined: Sequence[str] = ()) -> dict:
    """The registered Holm family over S1-S6 (module docstring). `declined`: endpoints of stages the user declined
    (their hypotheses enter with p = 1)."""
    pv: Dict[str, Optional[float]] = {}
    for ep, h, side, null in SECONDARY_HYPOTHESES:
        e = eps.get(ep)
        if ep in declined:
            pv[hyp_key(ep, h)] = 1.0
        elif e is None or e.get("status") != "complete":
            pv[hyp_key(ep, h)] = None
        else:
            st = _st(e)
            pv[hyp_key(ep, h)] = p_equivalence(st, null) if side == "tost" else p_one_sided(st, null, side)
    hs = holm_status(pv)
    out = {"family": [hyp_key(ep, h) for ep, h, _, _ in SECONDARY_HYPOTHESES], "alpha": ALPHA, "p": pv,
           "holm": hs, "endpoints": {}}
    for ep in sorted({x[0] for x in SECONDARY_HYPOTHESES}):
        e = eps.get(ep)
        if e is None or e.get("status") != "complete":
            out["endpoints"][ep] = {"status": (e or {}).get("status", "missing"), "claim": "not available"}
            continue
        st = _st(e)
        sup, interim = {}, {}
        for ep2, h, side, null in SECONDARY_HYPOTHESES:
            if ep2 != ep:
                continue
            k = hyp_key(ep, h)
            ok = _interval_holds(st, h, side, null)
            status = hs["status"][k]
            sup[h] = ok and status in ("rejected", "secured")
            interim[h] = {"interval": ok, "p": pv[k], "holm": status}
        determined = hs["final"] or all(x["holm"] in ("secured", "not rejectable") for x in interim.values())
        claim = adjusted_class(e["kind"], st, sup)
        if not hs["final"]:
            claim += (" (determined: no pending result can change it)" if determined else
                      " (interim: Holm family incomplete)")
        out["endpoints"][ep] = {"status": "complete", "interval_class": e["class"], "claim": claim,
                                "determined": determined, "hypotheses": interim}
    return out


def lane_panel(paths: Sequence[str], focals: Sequence[int] = (0, 1)) -> Dict[str, Dict[int, float]]:
    """D0 -> E_20 and the non-gold panel at turn 94 of a group of family records, per focal (mean over pairs, both
    seat orientations; focal orientation): {quantity: {focal: value}} averaged over the group's opponents."""
    from . import games as G
    keys = list(G.PANEL_KEYS)
    acc: Dict[str, Dict[int, List[float]]] = {}

    def put(q, f, v):
        acc.setdefault(q, {}).setdefault(f, []).append(float(v))
    for p in paths:
        d = _load(p)
        for f in focals:
            if f"f{f}__r" not in d:
                continue
            pan = d.get(f"f{f}__panel")
            put("E_20 (lead at 114)", f, _focal_values(d, f, "lead114").mean())
            put("lead at 94 (rewards)", f, _focal_values(d, f, "lead94").mean())
            if pan is None or not np.isfinite(pan).any():
                continue
            P = {k: pan[..., i] for i, k in enumerate(keys)}
            put("D0 (earned lead at 94)", f, np.nanmean(P["D0"]))
            put("E_20 - D0", f, _focal_values(d, f, "lead114").mean() - np.nanmean(P["D0"]))
            for k in ("level_diff", "xp_diff", "hp_diff", "stock_diff"):
                put(k, f, np.nanmean(P[k]))
            put("focal away (share)", f, np.nanmean(P["status_A"] > 0))
            put("opponent away (share)", f, np.nanmean(P["status_B"] > 0))
            put("focal dead (share)", f, np.nanmean(P["status_A"] == 2))
            put("opponent dead (share)", f, np.nanmean(P["status_B"] == 2))
            put("returns_in focal - opponent", f, np.nanmean(P["returns_in_A"] - P["returns_in_B"]))
            put("since_return focal - opponent", f, np.nanmean(P["since_return_A"] - P["since_return_B"]))
            put("unspent focal - opponent", f, np.nanmean(P["unspent_A"] - P["unspent_B"]))
            put("invested focal - opponent", f, np.nanmean(P["invested_A"] - P["invested_B"]))
    return {q: {f: float(np.mean(v)) for f, v in d.items()} for q, d in acc.items()}


PANEL_GROUPS = ("fam:S_in", "fam:S_out", "fam:M", "fam:C_hist", "fam:C_cross")


def panel(paths_of: Callable[[int], Dict[str, List[str]]], seeds: Sequence[int] = C.SEEDS) -> dict:
    """The lane-exit report per family: each arm's D0, E_20 and panel (seed means), and om - sp with seed intervals;
    the head to head from om's side. Only seeds with the family's complete records enter (counted)."""
    out = {}
    for grp in PANEL_GROUPS + ("h2h",):
        per: Dict[str, Dict[str, Dict[int, float]]] = {}
        n = 0
        for s in seeds:
            ps = paths_of(s).get(grp, [])
            if not ps or not all(os.path.exists(p) for p in ps):
                continue
            n += 1
            lp = lane_panel(ps, (0,) if grp == "h2h" else (0, 1))
            for q, d in lp.items():
                if grp == "h2h":
                    per.setdefault(q, {}).setdefault("om (vs sp)", {})[s] = d.get(0, float("nan"))
                    continue
                per.setdefault(q, {}).setdefault("om", {})[s] = d.get(0, float("nan"))
                per.setdefault(q, {}).setdefault("sp", {})[s] = d.get(1, float("nan"))
                per[q].setdefault("om - sp", {})[s] = d.get(0, float("nan")) - d.get(1, float("nan"))
        out[grp] = {"n_seeds": n, "status": "complete" if n == len(seeds) else f"partial ({n}/{len(seeds)})",
                    "quantities": {q: {a: interval(v) for a, v in d.items()} for q, d in per.items()}}
    return out


def per_opponent(paths_of: Callable[[int], Dict[str, List[str]]], seeds: Sequence[int] = C.SEEDS) -> dict:
    """Exploratory: per family and opponent, om - sp, om - start and sp - start at turn 114 over the seeds."""
    from . import evaluate as EV
    out = {}
    for name in EV.families(0):
        rows: Dict[int, Dict[str, Dict[int, float]]] = {}
        for s in seeds:
            ps = paths_of(s).get(f"fam:{name}", [])
            for j, p in enumerate(ps):
                if not os.path.exists(p):
                    continue
                d = _load(p)
                v = {f: _focal_values(d, f, "lead114").mean() for f in (0, 1, 2) if f"f{f}__r" in d}
                r = rows.setdefault(j, {})
                if 0 in v and 1 in v:
                    r.setdefault("om - sp", {})[s] = v[0] - v[1]
                if 2 in v:
                    r.setdefault("om - start", {})[s] = v[0] - v[2]
                    r.setdefault("sp - start", {})[s] = v[1] - v[2]
        opps = EV.families(0)[name]
        out[name] = {f"{j}:{EV._opp_tag(opps[j])}": {k: interval(v) for k, v in r.items()} for j, r in rows.items()}
    out["note"] = ("slot j is the j-th opponent of the family; labels are seed 0's. Opponents built on ext/baseline:o "
                   "(o = s + 3 mod 6) and the C_hist / C_cross slots (other seeds p != s) are different policies in "
                   "different seeds")
    return out


def versus_start(paths_of: Callable[[int], Dict[str, List[str]]], seeds: Sequence[int] = C.SEEDS) -> dict:
    """Exploratory: each arm's final against the common start ext/baseline:s on every family (focal - start)."""
    out = {}
    for name in ("S_in", "S_out", "M", "C_hist", "C_cross"):
        for f, arm in ((0, "om"), (1, "sp")):
            per = {}
            for s in seeds:
                ps = paths_of(s).get(f"fam:{name}", [])
                if ps and all(os.path.exists(p) for p in ps):
                    per[s] = family_contrast(ps, f, 2)["mean"]
            if per:
                out[f"{arm} - start | {name}"] = {**interval(per), "n_seeds": len(per)}
    return out


def _switch_turn(name: str) -> Optional[int]:
    from ..om2 import common as O2
    if name.startswith("its:"):
        return int(name.split(":")[2])
    if name in O2.HELDOUT_SWITCHERS:
        return int(O2.HELDOUT_SWITCHERS[name][2])
    return None


def switch_windows(paths_of: Callable[[int], Dict[str, List[str]]], seeds: Sequence[int] = C.SEEDS) -> dict:
    """Exploratory: om - sp in turn windows around the switch of each switching opponent of family M (the scheduled
    imitate-then-switch opponents its:X:t:o and the held-out switchers): before the switch, the first 20 turns after
    it, turns 20-60 after it, and the lead at 114; the mixtures over the whole 114 turns."""
    from . import evaluate as EV
    out = {}
    opps = EV.families(0)["M"]
    for j, opp in enumerate(opps):
        name = opp[1]
        t = _switch_turn(name)
        wins = {"turns 0-113": (0, 114)} if t is None else \
            {f"before the switch (0-{t - 1})": (0, t), f"switch + 0-19 ({t}-{t + 19})": (t, t + 20),
             f"switch + 20-59 ({t + 20}-{t + 59})": (t + 20, t + 60), "lead at 114": (0, 114)}
        per: Dict[str, Dict[int, float]] = {}
        for s in seeds:
            ps = paths_of(s).get("fam:M", [])
            if j >= len(ps) or not os.path.exists(ps[j]):
                continue
            d = _load(ps[j])
            for w, (a, b) in wins.items():
                x = d["f0__r"][:, :, a:b].sum(-1).mean(-1) - d["f1__r"][:, :, a:b].sum(-1).mean(-1)
                per.setdefault(w, {})[s] = float(x.mean())
        label = name if ":o" not in name and not name.startswith("its:") else name.rsplit(":", 1)[0] + ":o"
        out[label] = {w: {**interval(v), "n_seeds": len(v)} for w, v in per.items()}
    return out


def belief_mechanism(paths_of: Callable[[int], Dict[str, List[str]]], seeds: Sequence[int] = C.SEEDS) -> dict:
    """Exploratory (descriptive): the om finals' posterior from the recorded traces at games.CHECK_TURNS: against each
    library archetype (S_in) the mass on the true type and the share of games whose most probable type is the true one;
    against the held-out families (S_out, M) the catch-all mass ("mixed" + "unknown")."""
    from . import evaluate as EV
    from . import games as G
    types = list(C.BELIEF_TYPES)
    catch = [types.index("mixed"), types.index("unknown")]
    out: Dict[str, dict] = {}
    for fam in ("S_in", "S_out", "M"):
        opps = EV.families(0)[fam]
        for j, opp in enumerate(opps):
            name = opp[1]
            acc: Dict[str, Dict[int, List[float]]] = {}
            for s in seeds:
                ps = paths_of(s).get(f"fam:{fam}", [])
                if j >= len(ps) or not os.path.exists(ps[j]):
                    continue
                tr = _load(ps[j]).get("f0__trace")
                if tr is None:
                    continue
                lp = tr.reshape(-1, tr.shape[-2], tr.shape[-1]).astype(np.float64)
                p = np.exp(lp - lp.max(-1, keepdims=True))
                p /= p.sum(-1, keepdims=True)
                for c, turn in enumerate(G.CHECK_TURNS):
                    if fam == "S_in" and name in types:
                        k = types.index(name)
                        acc.setdefault(f"true-type mass at {turn}", {}).setdefault(s, []).append(p[:, c, k].mean())
                        acc.setdefault(f"identified at {turn}", {}).setdefault(s, []).append(
                            float((p[:, c].argmax(-1) == k).mean()))
                    else:
                        acc.setdefault(f"catch-all mass at {turn}", {}).setdefault(s, []).append(
                            p[:, c, catch].sum(-1).mean())
            label = name.rsplit(":", 1)[0] + ":o" if name.startswith(("its:", "style:")) else name
            out[f"{fam} | {label}"] = {q: {**interval({s: float(np.mean(v)) for s, v in d.items()}),
                                           "n_seeds": len(d)} for q, d in acc.items()}
    return out


# ---------------------------------------------------------------------------------------------- the campaign analysis
def analyse(phase: str = "main", root: Optional[str] = None, attack_root: Optional[str] = None, out: str = C.OUT,
            save: bool = True, declined: Sequence[str] = ()) -> dict:
    """The pre-registered M5d analysis from whatever records exist, with explicit completeness (module docstring)."""
    clear_cache()

    def paths_of(s):
        g = seed_paths(s, root)
        g.update(attack_paths(s, attack_root))
        return g
    eps = {name: endpoint(name, paths_of) for name in ENDPOINTS}
    res: Dict[str, object] = {"endpoints": eps}
    prim = [eps[k] for k in ("P1", "P2a", "P2b")]
    if all(e["status"] == "complete" for e in prim):
        res["headline"] = headline(eps["P1"]["class"], eps["P2a"]["class"], eps["P2b"]["class"])
    else:
        res["headline"] = {"outcome": None, "name": "not available: primary records incomplete",
                           "status": {e["name"]: e["status"] for e in prim}}
    res["secondary"] = secondary(eps, declined)
    res["panel"] = panel(paths_of)
    res["per_opponent"] = per_opponent(paths_of)
    res["versus_start"] = versus_start(paths_of)
    res["switch_windows"] = switch_windows(paths_of)
    res["belief"] = belief_mechanism(paths_of)
    res["meta"] = C.meta(stage="analyse", phase=phase)
    if save:
        C.save_json(f"analysis_{phase}", res, out)
    return res


# ---------------------------------------------------------------------------------------------- monitoring (M5b)
PANEL_NAMES = ("start:s", "base:o", "g_farm", "g_shove", "g_bully")


def monitoring(root: Optional[str] = None, out: str = C.OUT, save: bool = True, seeds: Sequence[int] = C.SEEDS) -> dict:
    """Learning curves, history comparisons and the convergence probe from the monitoring records (pre-registration
    §4). Monitoring never selects a checkpoint and never changes an endpoint."""
    from . import evaluate as EV
    root = root or EV.EOUT
    clear_cache()
    curves: Dict[str, dict] = {}
    for a in C.ARMS:
        for s in seeds:
            rows = {}
            for n in C.CURVE_POINTS:
                vals = []
                for q in range(len(EV.REF_PANEL)):
                    p = os.path.join(root, "curve", f"{a}_s{s}", f"{n}_{q}.npz")
                    vals.append(own_lead([p])["mean"] if os.path.exists(p) else float("nan"))
                rows[n] = vals
            curves[f"{a}_s{s}"] = rows
    # per arm and point: the panel mean over seeds (seed interval), each panel opponent
    curve_summary = {}
    for a in C.ARMS:
        for n in C.CURVE_POINTS:
            per = {s: float(np.mean(curves[f"{a}_s{s}"][n])) for s in seeds
                   if np.all(np.isfinite(curves[f"{a}_s{s}"][n]))}
            row = {"panel mean": interval(per) if per else {"n": 0}}
            for q, nm in enumerate(PANEL_NAMES):
                pq = {s: curves[f"{a}_s{s}"][n][q] for s in seeds if np.isfinite(curves[f"{a}_s{s}"][n][q])}
                row[nm] = interval(pq) if pq else {"n": 0}
            curve_summary[f"{a}|{n}"] = row
    # history: the final against its own 25 / 50 / 75% snapshots and the start (lead at 114 and discounted, 500 turns)
    hist: Dict[str, dict] = {}
    for a in C.ARMS:
        for tag, fname in (("25%", f"vs_{C.BUDGET // 4}.npz"), ("50%", f"vs_{C.MIDPOINT}.npz"),
                           ("75%", f"vs_{C.P75}.npz"), ("start", "vs_start.npz")):
            for ro in ("lead114", "discounted"):
                per, se, sig_neg = {}, {}, 0
                for s in seeds:
                    p = os.path.join(root, "history", f"{a}_s{s}", fname)
                    if os.path.exists(p):
                        v = own_lead([p], readout=ro)
                        per[s], se[s] = v["mean"], v["se"]
                        sig_neg += int(v["mean"] + 1.96 * v["se"] < 0)
                if per:
                    hist[f"{a}|final - {tag}|{ro}"] = {**interval(per), "n_seeds": len(per),
                                                       "game_se": {str(k): x for k, x in se.items()},
                                                       "seeds_final_loses_significantly": sig_neg}
    conv = {}
    for a in C.ARMS:
        h = hist.get(f"{a}|final - 75%|lead114")
        if not h or h["n_seeds"] != len(seeds):
            conv[a] = {"status": f"partial ({(h or {}).get('n_seeds', 0)}/{len(seeds)})", "verdict": "not available"}
            continue
        improving = h["mean"] >= 0.25 and h["L"] > 0
        conv[a] = {"status": "complete", "mean": h["mean"], "ci": h["ci"], "L": h["L"], "U": h["U"],
                   "verdict": "still improving (non-convergence: a limitation)" if improving else
                   "not shown to be still improving"}
    # M4.5c's cycling rule (reported, not a stop condition after training): a later checkpoint (the final) losing
    # significantly to an earlier snapshot on at least 2 seeds
    cyc = {a: {tag: hist.get(f"{a}|final - {tag}|lead114", {}).get("seeds_final_loses_significantly", 0)
               for tag in ("25%", "50%", "75%")} for a in C.ARMS}
    res = {"curves": curves, "curve_summary": curve_summary, "history": hist, "convergence": conv,
           "cycling": {a: {"seeds_by_snapshot": v, "flag": any(x >= 2 for x in v.values())} for a, v in cyc.items()},
           "meta": C.meta(stage="monitoring")}
    if save:
        C.save_json("monitoring_main", res, out)
    return res


__all__ = ["DELTA", "GAIN_MIN", "ALPHA", "interval", "classify_gain", "classify_ni", "classify_exploit", "competence",
           "headline", "holm", "holm_status", "p_one_sided", "p_equivalence", "t_pvalue_one_sided", "secondary",
           "variance_split", "ni_resolution", "family_contrast", "own_lead", "replication", "endpoint", "ENDPOINTS",
           "SECONDARY_HYPOTHESES", "seed_paths", "attack_paths", "analyse", "monitoring", "panel", "lane_panel",
           "adjusted_class", "GAIN_RULED_OUT"]
