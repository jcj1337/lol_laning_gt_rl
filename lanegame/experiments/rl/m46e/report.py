"""The generated tables of M4.6e (results/m46e/report.md) from observation.json, exposure.json, analysis.json and
ranking.json. Intervals: 95% t over the six C1 policies unless stated."""

from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np

from . import common as C


def table(head: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def ci(x, d=2) -> str:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        m, h = x
        if h is None or not np.isfinite(h):
            return f"{m:+.{d}f}"
        return f"{m:+.{d}f} ± {h:.{d}f}"
    return f"{x:+.{d}f}" if isinstance(x, (int, float)) else str(x)


def pct(x, d=2) -> str:
    return f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------------------------------------- Q1
def q1(o: dict) -> str:
    out = ["## Q1. The measurement path\n"]
    rows = defaultdict(lambda: defaultdict(list))
    changed = {}
    for spec, rs in o["pairs"].items():
        for r in rs:
            k = r["change"]
            rows[k][r["base"]].append(r["dV_A"])
            rows[k]["tv"].append(r["tv_actor_A"])
            rows[k]["ok"].append(r["A_ok"] and r["B_ok"])
            if r["base"] == "neutral L8":
                changed[k] = ", ".join(f"{n.replace('own_', '').replace('champ_', '')} {v:+.3f}"
                                       for n, v in sorted(r["A_changed"].items()))
    bases = ("freeze L5", "neutral L8", "neutral L15")
    out.append("**Controlled single-feature pairs** (seat A changed; the entries that change in seat A's own observation "
               "at neutral L8; the critic's change of seat A's value in resources, mean over the six policies, at each "
               "base; the actor's total-variation change; whether exactly the intended entries changed in both seats' "
               "observations):\n")
    out.append(table(["change", "entries changed (neutral L8, seat A)"] + [f"dV_A {b}" for b in bases] +
                     ["actor TV", "encoding check"],
                     [[k, changed.get(k, "")] + [f"{np.mean(v[b]):+.3f}" for b in bases] +
                      [f"{np.mean(v['tv']):.3f}", f"{sum(v['ok'])}/{len(v['ok'])}"] for k, v in rows.items()]))
    r = o["reachable"]
    out.append("\n**Reachable states** (self-play from s0 and archive starts, the archive sample's variants, the M4.6d "
               "cells):\n")
    out.append(table(["check", "value"], [[k, str(v if not isinstance(v, float) else round(v, 4))] for k, v in r.items()
                                         if k != "beyond_exact_decode_invested"] +
                     [["invested of the states beyond exact decoding (resources)",
                       ", ".join(f"{x:.0f}" for x in r.get("beyond_exact_decode_invested", []))]]))
    g = o["ranges"]
    out.append(f"\n**Strength entries** (natural self-play states of three policies, n = {g['natural_states']}):\n")
    out.append(table(["entry", "q05", "median", "q95", "sd", "median per-turn change"],
                     [[k.replace("own_champ_", ""), f"{v.get('q05', 0):.3f}", f"{v.get('q50', 0):.3f}",
                       f"{v.get('q95', 0):.3f}", f"{v.get('sd', 0):.3f}", f"{v.get('typical_turn_change', 0):.4f}"]
                      for k, v in g["entries"].items()]))
    keys = list(next(iter(g["contrasts"].values())).keys())
    out.append("\nChange of seat A's entries for the M4.6d strength contrasts (freeze cells, contrast minus equal):\n")
    out.append(table(["contrast"] + [k.replace("own_champ_", "") for k in keys],
                     [[c] + [f"{v[k]:+.3f}" for k in keys] for c, v in g["contrasts"].items()]))
    out.append(f"\nExplicit log duel (from two observation entries) against the state's duel ratio: max |error| "
               f"{g['explicit_log_duel_max_abs_error']:.1e}; natural log-duel quantiles (5/25/50/75/95%): "
               + ", ".join(f"{x:+.2f}" for x in g.get("natural_log_duel_q", [])))
    out.append("\n**Checkpoint and configuration compatibility:**\n")
    out.append(table(["policy", "inputs / actions", "hidden", "game config = arm", "reward scale", "gamma", "lambda",
                      "vf_coef", "vf_clip", "ent", "current fraction"],
                     [[c["spec"], f"{c['inputs']} / {c['actions']}", str(c["hidden"]), str(c["game_config_matches_arm"]),
                       str(c["reward_scale"]), str(c["gamma"]), str(c["gae_lambda"]), str(c["vf_coef"]),
                       str(c["vf_clip"]), str(c["ent_coef"]), str(c["current_fraction"])] for c in o["compat"]]))
    out.append("\n**Shared trunk: gradient norms on training-like batches** (the M4.6c mixed-start collector, the run's own "
               "PPO configuration, final weights; minibatches of 256; value term = vf_coef x value loss on 0.1-scaled "
               "returns; policy term = clipped surrogate - ent_coef x entropy):\n")
    out.append(table(["policy", "minibatches", "|grad| policy term (trunk)", "|grad| value term (trunk)", "ratio value / policy",
                      "cosine", "|grad| value head", "value loss (resources^2)"],
                     [[s, str(v["minibatches"]), f"{v['policy_trunk_norm']:.4f}", f"{v['value_trunk_norm']:.4f}",
                       f"{v['ratio_value_to_policy']:.3f}", f"{v['cosine']:+.3f}", f"{v['value_head_norm']:.4f}",
                       f"{v['value_loss_resources2']:.2f}"] for s, v in o["grad"].items()]))
    return "\n".join(out)


def probe(a: dict) -> str:
    p = a["probes"]["aggregate"]
    out = ["\n**Is strength linearly decodable from the trained trunk?** (ridge, held-out R^2 on the main split's test "
           "groups; mean over the six policies)\n"]
    out.append(table(["target", "from the observation", "from hidden layer 1", "from hidden layer 2 (the value head's input)"],
                     [[t, ci(p[f"{t}<-obs"]), ci(p[f"{t}<-h1"]), ci(p[f"{t}<-h2"])] for t in ("log_duel", "hp_own")]))
    out.append(f"\nA x1.76 contrast moves hidden layer 2 by {ci(p['h2_move_x1.76_over_natural_spread'])} of the natural "
               "states' mean distance from their centroid. Readout B refitted with a weak penalty (alpha 1): "
               + ", ".join(f"{k} {ci(v['ratio'])}" for k, v in p["B_alpha1_m46d"].items()))
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- Q2
FAM_NAMES = {"plaus_early": "plausible early edge (duel 1.10-1.5, level <= 9)", "plaus_early_125": "... duel 1.25-1.5",
             "large": "large gap (duel >= 2)", "extreme": "extreme gap (duel >= 3)",
             "fav_freeze": "favourable early freeze (M4.6d)", "fav_freeze_dmg": "... breaker at <= 70% HP, level <= 8",
             "weak_window": "weak-side Safe/Yield window", "weak_window_broad": "weak side <= 60% HP, Recall legal, stronger opponent catchable",
             "any_freeze": "any freeze state"}


def q2(x: dict) -> str:
    out = ["## Q2. Training exposure\n"]
    st = x["direct_starts"]
    c1 = [v for k, v in st.items() if k.startswith("C1")]
    labs = sorted(c1[0]["archive_by_label"])
    tot = np.mean([sum(v["joint_steps_finished"].values()) for v in c1])
    rows = [["s0 (initial)", f"{np.mean([v['games_started']['initial'] for v in c1]):.0f}",
             f"{np.mean([v['joint_steps_finished']['initial'] for v in c1]):,.0f}",
             pct(np.mean([v['joint_steps_finished']['initial'] for v in c1]) / tot, 1)]]
    for lb in labs:
        js = np.mean([v["joint_steps_finished_by_label"][lb] for v in c1])
        rows.append([f"archive: {lb}", f"{np.mean([v['archive_by_label'][lb] for v in c1]):.0f}", f"{js:,.0f}", pct(js / tot, 1)])
    out.append("**Direct: start counts of the six C1 8.192M runs** (final.pt metadata, mean per run; joint turns of "
               "finished games; each joint turn is two agent transitions, one per role):\n")
    out.append(table(["start", "games started", "joint turns", "share"], rows))
    out.append("\n**Direct: the training archive's composition at the start state** (every archived state, both "
               "orientations; share of states in each family; the stronger side's duel quantiles 10/50/90/99%):\n")
    for arm in ("C1", "baseline"):
        a = x["direct_archive"][arm]
        fams = ("plaus_early", "large", "extreme", "fav_freeze", "fav_freeze_dmg", "weak_window", "any_freeze")
        out.append(f"\n{arm} curve ({a['states']} states):\n")
        out.append(table(["label", "states"] + [f for f in fams] + ["duel q10/50/90/99"],
                         [[lb, str(v["states"])] + [pct(v[f], 1) for f in fams] +
                          ["/".join(f"{q:.2f}" for q in v["duel_q"])] for lb, v in a["by_label"].items()]))
    occ = x["occupancy"]
    cols = []
    for tag, pred in (("C1 @1.0M", lambda s: "C1" in s and "@10" in s), ("C1 @4.1M", lambda s: "C1" in s and "@40" in s),
                      ("C1 final", lambda s: "C1" in s and "@" not in s), ("baseline final", lambda s: "baseline" in s)):
        cols.append((tag, [k for k in occ if pred(k)]))
    out.append("\n**Reconstructed training occupancy** (ESTIMATE: frozen snapshots playing themselves from the training "
               "start distribution, archive turns weighted to the measured 25% share; mean over seeds 0-2; in % of "
               "joint turns, and joint turns per 8.192M run = occupancy x 4.096M):\n")
    rows = []
    for f in occ[cols[0][1][0]]["families"]:
        r = [FAM_NAMES.get(f, f)]
        for tag, ks in cols:
            m = np.mean([occ[k]["families"][f]["training"] for k in ks])
            r.append(f"{100 * m:.3f}% ({m * 4_096_000:,.0f})")
        fin = cols[2][1]
        fa = [occ[k]["families"][f]["from_archive_share"] for k in fin]
        r.append(pct(np.nanmean(fa), 0) if np.isfinite(fa).any() else "-")
        r.append(f"{np.mean([occ[k]['families'][f]['s0'] for k in fin]) * 100:.3f}%")
        rows.append(r)
    out.append(table(["family"] + [t for t, _ in cols] + ["from archive starts (C1 final)", "s0 games only (C1 final)"],
                     rows))
    fin = cols[2][1]
    dec = defaultdict(lambda: defaultdict(list))
    from .exposure import CHECKPOINTS_T
    for k in fin:
        for lb, v in occ[k]["decay"].items():
            for f, arr in v.items():
                if isinstance(arr, list):
                    dec[lb][f].append(arr)
    out.append("\n**Decay after an archive start** (C1 final, share of games in the family at turn t; mean |log duel|):\n")
    rows = []
    for lb in sorted(dec):
        for f in ("large", "plaus_early", "fav_freeze", "any_freeze", "mean_abs_log_duel"):
            arr = np.mean(dec[lb][f], 0)
            rows.append([lb, f] + [f"{v:.2f}" if f == "mean_abs_log_duel" else pct(v, 0) for v in arr])
    out.append(table(["archive label", "family"] + [f"t={t}" for t in CHECKPOINTS_T], rows))
    nb = x["neighbourhoods"]
    keys = [k for k in nb if "@" not in k]
    labels = list(nb[keys[0]])
    out.append("\n**Neighbourhoods of the M4.6d diagnostic states in training** (C1 final, mean over seeds 0-2; same "
               "kind, level +-1, duel within x1.15, the weaker side's HP within +-0.15):\n")
    out.append(table(["M4.6d state", "duel", "training occupancy", "joint turns per run", "from s0 games", "from archive"],
                     [[lb, f"{nb[keys[0]][lb]['duel']:.2f}", f"{100 * np.mean([nb[k][lb]['training'] for k in keys]):.4f}%",
                       f"{np.mean([nb[k][lb]['joint_turns_per_8M_run'] for k in keys]):,.0f}",
                       f"{100 * np.mean([nb[k][lb]['s0'] for k in keys]):.4f}%",
                       f"{100 * np.mean([nb[k][lb]['archive'] for k in keys]):.4f}%"] for lb in labels]))
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- Q3 / Q4
def q3(a: dict) -> str:
    out = ["## Q3. Return labels and the horizon\n"]
    out.append("**Residual tail on problematic constructed cells** (three policies, K 32, new seeds; seat A's discounted "
               "lead G_h, mean over policies; the 300 -> 800 change over policies and pooled over rollouts):\n")
    t = a["tail"]
    cps = next(iter(t.values()))["cps"]
    out.append(table(["cell"] + [f"G{c}" for c in cps] + ["G800 - G300 (policies)", "(rollouts, mean ± se)",
                                                          "sd of G300 per rollout"],
                     [[k] + [f"{x:+.2f}" for x in v["mean_by_cp"]] + [ci(v["change_300_800"]),
                      f"{v['change_300_800_rollouts'][0]:+.3f} ± {v['change_300_800_rollouts'][1]:.3f}",
                      f"{v['sd_G300_per_rollout']:.2f}"] for k, v in t.items()]))
    return "\n".join(out)


def _arm_rows(ag, arms, metric_paths):
    rows = []
    for arm in arms:
        r = [arm]
        for path in metric_paths:
            v = ag[arm]
            for p in path:
                v = v.get(p, {}) if isinstance(v, dict) else {}
            r.append(ci(v) if v != {} else "")
        rows.append(r)
    return rows


def q4(a: dict) -> str:
    out = ["## Q4. Supervised value study\n"]
    arms = ("critic", "B", "L", "C", "D")
    for split in ("main", "strength", "family", "reach"):
        ag = a["fits"][split]["aggregate"]
        out.append(f"\n### Split: {split}\n")
        out.append("P1, strength-premium recovery on the M4.6d cells (Sum predicted / Sum realized; both seats):\n")
        out.append(table(["arm", "edge", "plaus", "large (x1.76)", "gap cells", "edge+plaus", "large+gap"],
                         _arm_rows(ag, arms, [("m46d", "classes", c, "ratio") for c in
                                              ("edge", "plaus", "large", "gap", "edge+plaus", "large+gap")])))
        out.append("\nRealized premiums (M4.6d K 32 labels): " + ", ".join(
            f"{c} {ci(ag['critic']['m46d']['classes'][c]['realized'])}" for c in ("edge", "plaus", "large", "gap")))
        out.append("\n\nSeat A's bias on the M4.6d cells (predicted - realized G300) and the M4.6d natural states (G500):\n")
        out.append(table(["arm", "freeze", "lane", "gap", "natural MAE", "natural bias", "natural slope", "natural R^2"],
                         _arm_rows(ag, arms, [("m46d", "bias", "freeze"), ("m46d", "bias", "lane"), ("m46d", "bias", "gap"),
                                              ("m46d", "natural", "mae"), ("m46d", "natural", "bias"),
                                              ("m46d", "natural", "slope"), ("m46d", "natural", "r2")])))
        out.append("\nHeld-out labelled examples (single-rollout labels; constructed starts: mean of two rollouts):\n")
        out.append(table(["arm", "natural MAE", "natural R^2", "natural slope", "start MAE", "start R^2", "start slope",
                          "start noise-floor MSE", "start RMSE", "start explainable variance captured", "trajectory MAE", "trajectory R^2", "beyond x1.6 MAE",
                          "beyond x1.6 bias", "freeze groups MAE"],
                         _arm_rows(ag, arms, [("natural_test", "mae"), ("natural_test", "r2"), ("natural_test", "slope"),
                                              ("con_start_test", "mae"), ("con_start_test", "r2"),
                                              ("con_start_test", "slope"), ("con_start_test", "noise_floor_mse"),
                                              ("con_start_test", "rmse"), ("con_start_test", "explained_share"),
                                              ("con_traj_test", "mae"),
                                              ("con_traj_test", "r2"), ("beyond_1.6", "mae"), ("beyond_1.6", "bias"),
                                              ("freeze_test", "mae")])))
        out.append("\nP3, paired premiums on held-out archive groups (seat A's gain; mean realized / mean predicted):\n")
        rows = []
        for c in ("lead", "level", "hp", "big"):
            rr = ag["critic"]["pairs"].get(c, {})
            rows.append([c, ci(rr.get("realized", [np.nan, np.nan]))] +
                        [ci(ag[x]["pairs"].get(c, {}).get("pred", [np.nan, np.nan])) for x in arms])
        out.append(table(["contrast", "realized"] + [f"pred {x}" for x in arms], rows))
        if split == "main":
            out.append("\nFit-seed stability (main split, P1 ratios per seed):\n")
            seeds = [f"C{s}" for s in range(3)] + [f"D{s}" for s in range(3)]
            out.append(table(["seed arm", "edge", "plaus", "large", "gap", "natural MAE (M4.6d)"],
                             _arm_rows(ag, seeds, [("m46d", "classes", c, "ratio") for c in ("edge", "plaus", "large", "gap")]
                                       + [("m46d", "natural", "mae")])))
    return "\n".join(out)


def ranking(r: dict) -> str:
    out = ["\n## Look-ahead rankings at the M4.6d decision states\n",
           f"Replay of turn 0 of {r['jobs']} stored jobs; maximum |difference| from the stored critic look-ahead "
           f"{r['replay_max_abs_error']:.1e}. Spearman / sign agreement / same best action against the stored K 16 "
           "rollouts (advantages over the policy's own choice), mean over states (each the mean over six policies); "
           "spread = sd of the predicted advantages across candidates (rollouts' spread in the second column).\n"]
    rows = []
    for k, v in r["summary"].items():
        for a in ("critic", "B", "L", "C", "D"):
            rows.append([k, f"{v['roll_spread']:.2f}" if a == "critic" else "", a, f"{v[a]['rho']:+.2f}",
                         f"{v[a]['sign']:.2f}", f"{v[a]['best']:.2f}", f"{v[a]['spread']:.2f}"])
    out.append(table(["decision", "rollout spread", "predictor", "Spearman", "sign agreement", "same best", "spread"], rows))
    return "\n".join(out)


def run() -> str:
    o, x, a, r = (C.load_json(n) for n in ("observation", "exposure", "analysis", "ranking"))
    parts = ["# M4.6e generated tables\n",
             "Generated by `python -m lanegame.experiments.rl.m46e report`. Narrative: "
             "docs/m4/m4_6/m4_6e_report.md. Environment C1 (provisional, not adopted); policies ext/C1:0-5; labels "
             "policy-specific G300 in resources.\n"]
    if o:
        parts.append(q1(o))
    if a:
        parts.append(probe(a))
    if x:
        parts.append(q2(x))
    if a:
        parts.append(q3(a))
        parts.append(q4(a))
    if r:
        parts.append(ranking(r))
    text = "\n\n".join(parts) + "\n"
    C.write_md("report", text)
    return text


__all__ = ["run"]
