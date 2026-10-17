"""Generated tables of OM2 (results/om2/report.md) from preregistration.json, smoke.json, train_main.json, the runs'
history files and analysis_main.json."""

from __future__ import annotations

import json
import os

import numpy as np

from . import common as C
from . import evaluate as EV


def cell(c) -> str:
    return str(c).replace("|", "\\|")


def table(head, rows) -> str:
    out = ["| " + " | ".join(cell(h) for h in head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def mci(st: dict, d: int = 2) -> str:
    if not st or not np.isfinite(st.get("mean", np.nan)):
        return "n/a"
    h = st.get("ci", np.nan)
    return f"{st['mean']:+.{d}f} ± {h:.{d}f}" if np.isfinite(h) else f"{st['mean']:+.{d}f}"


def pct(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def f2(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.2f}"


def _history(arm: str, s: int):
    p = os.path.join(C.run_dir(arm, s), "history.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _window(H, lo: float, hi: float):
    n = len(H)
    return H[int(lo * n):max(int(lo * n) + 1, int(hi * n))]


def _opp_return(H) -> dict:
    """Finished-game-weighted mean learner return per fixed opponent over the history window."""
    num, den = {}, {}
    for h in H:
        for k, v in h.get("historical_by_opponent", {}).items():
            c = h.get("finished_by_opponent", {}).get(k, 0)
            if c and np.isfinite(v):
                num[k] = num.get(k, 0.0) + v * c
                den[k] = den.get(k, 0) + c
    return {k: num[k] / den[k] for k in num}


def training_section() -> list:
    tr = C.load("train_main") or {}
    L = ["## Training", ""]
    rows = []
    for r in tr.get("runs", []):
        H = _history(r.get("arm", ""), r.get("seed", -1)) if r.get("status") == "complete" else []
        last = _window(H, 0.9, 1.0)

        def m(k):
            v = [h[k] for h in last if k in h and np.isfinite(h[k])]
            return float(np.mean(v)) if v else float("nan")
        rows.append([r.get("name"), r.get("status"), f"{r.get('agent_transitions', 0):,}", r.get("updates", ""),
                     f"{r.get('wall_s', 0) / 60:.0f}", f"{r.get('transitions_per_s', float('nan')):.0f}",
                     f"{m('approx_kl'):.4f}", f"{m('entropy'):.2f}", f"{m('explained_variance'):.2f}",
                     f"{m('belief_std_mean'):.3f}", f"{m('unknown_mean'):.3f}",
                     sum(h.get("illegal_sampled", 0) for h in H), (r.get("digest") or "")[:12]])
    L += [table(["run", "status", "transitions", "updates", "wall (min)", "transitions/s", "KL (last 10%)",
                 "entropy", "explained var.", "belief spread", "'unknown' mass", "illegal", "digest"], rows), "",
          "Belief spread: the mean over the 13 belief inputs of their standard deviation in a rollout (0 in the "
          "control by construction). 'unknown' mass: the mean posterior of the uniform catch-all.", ""]
    rows = []
    for arm in C.ARMS:
        first, lastw = {}, {}
        for s in C.PILOT_SEEDS:
            H = _history(arm, s)
            if not H:
                continue
            for k, v in _opp_return(_window(H, 0.0, 0.1)).items():
                first.setdefault(k, []).append(v)
            for k, v in _opp_return(_window(H, 0.9, 1.0)).items():
                lastw.setdefault(k, []).append(v)
        for k in sorted(set(first) | set(lastw)):
            a, b = np.mean(first.get(k, [np.nan])), np.mean(lastw.get(k, [np.nan]))
            rows.append([arm, k.replace("lib:", ""), f2(a), f2(b), f2(b - a)])
    if rows:
        L += ["Learner's mean game return against each fixed training opponent (resources; games of 512 or 150 "
              "turns, so levels are not comparable across opponents; mean over seeds):", "",
              table(["arm", "opponent", "first 10% of updates", "last 10%", "change"], rows), ""]
    return L


def run(phase: str = "main") -> str:
    A = C.load(f"analysis_{phase}")
    sm = C.load("smoke")
    pr = C.load("preregistration")
    con, ident, mech, cl = A["contrasts"], A["identification"], A["mechanism"], A["classification"]
    L = ["# OM2 Bayesian opponent-modelling pilot: generated tables", "",
         f"Pre-registration `{pr['record']['document']}` SHA-256 `{pr['record']['sha256'][:16]}…` (frozen "
         f"{pr['meta']['timestamp']}; amendments: {len(pr.get('amendments', []))}). Values: the evaluated arm's "
         f"earned-gold lead at turn {EV.READ_TURN} (= E_20 at turn {EV.LANE_END} in expectation), resources, mean "
         f"over {EV.N_PAIRS} paired-seat pairs per cell; contrasts are treatment − control on the same games.", ""]
    L += ["## Readiness smoke", "", table(["check", "result"], [[k, v] for k, v in sm["checks"].items()]), "",
          "Throughput (transitions/s, 3 concurrent runs): " + ", ".join(
              f"{k} {v:.0f}" for k, v in sm["throughput_transitions_per_s"].items()), ""]
    L += training_section()
    # ---- primary
    ps = con["per_seed"]
    rows = []
    for fam, label in (("E_in", "E_in: archetypes (in the library)"), ("E_out", "E_out: held-out scripted (descriptive)"),
                       ("R1", "R1: competent learned"), ("R2", "R2: held-out mixtures and switchers"),
                       ("R3", "R3: treatment vs control (the treatment's lead)")):
        st = ps.get(fam, {})
        v = st.get("per_seed", {})
        rows.append([label] + [f2(v.get(str(s), np.nan)) for s in C.PILOT_SEEDS]
                    + [mci(st), f"{st.get('pos', 0)} / {st.get('neg', 0)}"])
    L += ["## Primary contrasts (seeds as units; t interval with 2 df)", "",
          table(["family"] + [f"seed {s}" for s in C.PILOT_SEEDS] + ["mean ± 95%", "seeds + / −"], rows), ""]
    rows = []
    for fam, d in con["pooled"].items():
        rows.append([fam] + [mci(d.get(k, {})) for k in ("read", "lane", "end")])
    L += ["Pooled over games (descriptive; 1.96 SE):", "",
          table(["family", f"lead at {EV.READ_TURN}", f"lane lead at {EV.LANE_END}", f"lead at {EV.T_EVAL}"], rows), ""]
    L += ["## Classification (pre-registration §4)", "",
          table(["criterion", "result"],
                [["E_in gain (mean ≥ +0.25, every seed > 0)", cl["E_in_gain"]]]
                + [[f"{f} loss (mean ≤ −0.25, every seed < 0)", v] for f, v in cl["R_losses"].items()]
                + [["**outcome**", f"**{cl['outcome']}: {cl['label']}**"]]), ""]
    # ---- per opponent
    rows = []
    for o, r in con["per_opponent"].items():
        ctl = np.mean(list(r["control"].values()))
        trt = np.mean(list(r["treatment"].values()))
        rows.append([r["family"], o, f2(ctl), f2(trt)] + [f2(r["per_seed"].get(str(s), np.nan)) for s in C.PILOT_SEEDS]
                    + [f2(np.mean(list(r["per_seed"].values())))])
    L += ["## Per opponent (lead at 114; arm values are means over seeds)", "",
          table(["family", "opponent", "control", "treatment"] + [f"T − C seed {s}" for s in C.PILOT_SEEDS]
                + ["T − C mean"], rows), ""]
    # ---- identification
    rows = [[o] + [pct(v.get(t)) for t in ("10", "20", "40", "94")] for o, v in ident["library"].items()]
    L += ["## Mechanism", "", "The treatment's belief in its evaluation games: MAP type correct after t turns "
          "(all seeds and seats):", "", table(["archetype", "10", "20", "40", "94"], rows), ""]
    rows = [[o] + [pct(v.get(t)) for t in ("10", "20", "40", "94")] for o, v in ident["R2_catch_all_mass"].items()]
    L += ["Posterior mass on the catch-alls ('mixed' + 'unknown') against the held-out mixtures and switchers:", "",
          table(["opponent", "10", "20", "40", "94"], rows), ""]
    rows = []
    for s, d in mech.items():
        for arm in C.ARMS:
            if arm in d:
                rows.append([s, arm, d["states"]] + [f"{d[arm][t]:.3f}" for t in C.BELIEF_TYPES])
    L += ["Belief-swap probes: mean total-variation distance between the policy with a one-type belief and with the "
          "prior, on the treatment's own observations against the archetypes (the control's is 0 by construction):", "",
          table(["seed", "arm", "states"] + list(C.BELIEF_TYPES), rows), ""]
    L += explore_section()
    txt = "\n".join(L)
    C.write_md("report", txt)
    return txt


def explore_section() -> list:
    X = C.load("explore")
    if not X:
        return []
    S = X["summary"]
    L = ["## Exploratory reference (NOT pre-registered; added after the classification; cannot change the outcome)", "",
         "pi_s = ext/baseline:s, the common start of both arms (no opponent information), on exactly the games of "
         "the evaluation cells; arm − pi_s is paired. Seeds as units (t interval, 2 df).", ""]
    rows = []
    for fam, d in S["per_seed"].items():
        rows.append([fam, mci(d["pi"]), mci(d["control-pi"]), mci(d["treatment-pi"])])
    L += [table(["family", "pi_s's value", "control − pi_s", "treatment − pi_s"], rows), ""]
    rows = [[arm, mci(st), ", ".join(f2(v) for v in st.get("per_seed", {}).values())]
            for arm, st in S["h2h_pi"].items()]
    L += ["Head to head against its own start pi_s (the arm's lead at 114):", "",
          table(["arm", "lead vs pi_s", "per seed"], rows), ""]
    rows = [[po["family"], o, f2(po["pi"]), f2(po["control-pi"]), f2(po["treatment-pi"])]
            for o, po in S["per_opponent"].items()]
    L += ["Per opponent (means over seeds):", "",
          table(["family", "opponent", "pi_s", "control − pi_s", "treatment − pi_s"], rows), ""]
    return L


__all__ = ["run"]
