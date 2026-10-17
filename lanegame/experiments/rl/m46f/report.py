"""Generated tables of M4.6f (results/m46f/report.md) from results/m46f/analysis.json."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import common as C


def f(st: dict, d: int = 2) -> str:
    if not isinstance(st, dict) or not st.get("n"):
        return "n/a"
    m, h = st.get("mean", np.nan), st.get("ci", np.nan)
    if not np.isfinite(m):
        return "n/a"
    s = f"{m:+.{d}f} ± {h:.{d}f}" if np.isfinite(h) else f"{m:+.{d}f}"
    if np.isfinite(h) and (m - h > 0 or m + h < 0):
        s = f"**{s}**"
    return s + f" ({st.get('pos', 0)}+/{st.get('neg', 0)}−)" if st.get("n", 0) > 1 else s


def plain(st: dict, d: int = 2) -> str:
    if not isinstance(st, dict) or not st.get("n") or not np.isfinite(st.get("mean", np.nan)):
        return "n/a"
    return f"{st['mean']:.{d}f}"


def table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def section(r: dict, phase: str) -> str:
    L = []
    sc = r["scorecard"]
    L.append(f"## Scorecard ({phase})\n")
    L.append(table(["statistic", "separate − shared", "class"], [
        ["V1 edge+plaus recovery ratio", f(sc["V1"]["stat"]), sc["V1"]["class"] +
         f" (separate {sc['V1']['separate_ratio']:.2f}, shared {sc['V1']['shared_ratio']:.2f}, start "
         f"{sc['V1']['start_ratio']:.2f})"],
        ["V2 natural MAE (G500 states)", f(sc["V2"]["stat"]), "regression" if sc["V2"]["regression"] else "no regression"],
        ["V3 large+gap ratio", f(sc["V3"]["stat"]), "diagnostic"],
        ["B1 head to head (W 1000)", f(sc["B1"]["stat"]), sc["B1"]["class"]],
        ["B1 powered, 320 episodes (amendment 1; not used for the outcome)",
         f(sc.get("B1 powered (amendment 1, not used for the outcome)", {}).get("stat", {})),
         sc.get("B1 powered (amendment 1, not used for the outcome)", {}).get("class", "n/a")],
        ["B2 strength use (G150)", f(sc["B2"]["stat"]), sc["B2"]["class"]],
        *[[k, f(v["stat"]), "regression" if v["regression"] else "no regression"] for k, v in sc["guards"].items()],
        ["shared − start, edge+plaus ratio", f(sc["shared_gain_over_start"]["stat"]),
         "material" if sc["shared_gain_over_start"]["material"] else "not material"]]))
    L.append(f"\n**Outcome: {sc['outcome']}**\n")

    tr = r["training"]["table"]
    L.append("## Training\n")
    keys = ["transitions_per_s", "elapsed_s", "ev_first10", "ev_last100", "value_loss_first10_res2",
            "value_loss_last100_res2", "entropy_last100", "kl_mean", "clipfrac_mean", "grad_norm_mean",
            "return_std_last100", "gp_total_norm_mean", "gp_actor_norm_mean", "gp_critic_norm_mean",
            "gp_trunk_pol_mean", "gp_trunk_val_mean", "gp_trunk_cos_mean", "gp_ctrunk_val_mean", "gp_clip_shrink_mean",
            "gp_critic_share_of_total_sq"]
    L.append(table(["quantity", "shared", "separate", "separate − shared"],
                   [[k, plain(tr[k]["shared"], 4), plain(tr[k]["separate"], 4), f(tr[k]["separate-shared"], 4)]
                    for k in keys if k in tr]))
    per = r["training"]["per_seed"]
    L.append("\nExplained variance in ten blocks of updates (mean over seeds):\n")
    L.append(table(["arm"] + [str(i + 1) for i in range(10)],
                   [[a] + [f"{np.mean([per[a][s]['ev_blocks'][i] for s in per[a]]):.3f}" for i in range(10)]
                    for a in C.ARMS if per[a]]))

    vc = r["value_calib"]["table"]
    L.append("\n## Value: M4.6d calibration set (own-policy labels, K 32, H 300 / 500)\n")
    L.append(table(["readout", "start", "shared", "separate", "separate − shared", "shared − start",
                    "separate − start"],
                   [[k, plain(v["start"]), plain(v["shared"]), plain(v["separate"]), f(v["separate-shared"]),
                     f(v["shared-start"]), f(v["separate-start"])] for k, v in vc.items()]))
    vn = r["value_natural"]["table"]
    L.append("\n## Value: natural self-play labels (G300, single rollouts, both seats)\n")
    L.append(table(["readout"] + [a for a in ("start",) + C.ARMS if a in vn["mae"]] + ["separate − shared"],
                   [[k] + [plain(v[a], 3) for a in ("start",) + C.ARMS if a in v] + [f(v.get("separate-shared", {}), 3)]
                    for k, v in vn.items()]))
    dr = r["value_natural"].get("drift", {}).get("table")
    if dr:
        L.append("\nPolicy drift on the start's natural states (decisions with a choice):\n")
        L.append(table(["quantity", "value"], [[k, f(v, 4)] for k, v in dr.items()]))
    vp = r["value_probes"]
    L.append("\n## Value representation: ridge probes (held-out R², M4.6e main split)\n")
    L.append(table(["probe"] + [a for a in ("start",) + C.ARMS if a in next(iter(vp["table"].values()))] +
                   ["separate − shared"],
                   [[k] + [plain(v[a], 3) for a in ("start",) + C.ARMS if a in v] + [f(v.get("separate-shared", {}), 3)]
                    for k, v in vp["table"].items()]))
    if vp.get("responses"):
        L.append("\nCritic response to single-feature changes of seat A (resources):\n")
        L.append(table(["change"] + [a for a in ("start",) + C.ARMS],
                       [[k] + [plain(v.get(a, {}), 2) for a in ("start",) + C.ARMS] for k, v in vp["responses"].items()]))

    hh = r["h2h"]
    L.append("\n## Behaviour: head to head from s0\n")
    L.append(table(["comparison", "estimate"], [[k, f(v)] for k, v in hh.items()
                                                if isinstance(v, dict) and "mean" in v]))
    if hh.get("curve"):
        L.append("\nseparate vs shared over training (W 460):\n")
        L.append(table(["transitions", "separate vs shared"], [[f"{int(n):,}", f(v)] for n, v in hh["curve"].items()]))
    L.append("\nScripted core suite by reference (W 460):\n")
    L.append(table(["reference", "shared", "separate", "separate − shared"],
                   [[k, plain(v["shared"]), plain(v["separate"]), f(v["separate-shared"])]
                    for k, v in hh["suite by reference"].items()]))
    cv = r["conv"]
    if cv:
        L.append("\n## Behaviour: strength use (B2 cells; separate − shared, mean over both opponents)\n")
        L.append(table(["readout", "G50", "G150"], [[k, f(cv["G50"][k]), f(cv["G150"][k])]
                                                    for k in cv["G150"] if k != "per_seed"]))
        L.append("\nBy cell kind (G150):\n")
        L.append(table(["kind", "separate − shared"], [[k, f(v)] for k, v in cv["by kind G150"].items()]))
        L.append("\nDenial and exposure (ledger at 150 turns):\n")
        L.append(table(["quantity", "shared level", "separate − shared"],
                       [[k, f"{v.get('shared_level', np.nan):.2f}", f(v)]
                        for k, v in cv["denial (ledger, 150 turns; separate - shared)"].items()]))
    for fam, title in (("freeze", "freezes (B3; G50)"), ("recall", "weak side at Recall windows (B4)")):
        x = r[fam]
        if not x:
            continue
        L.append(f"\n## Behaviour: {title}\n")
        for role, rows in x.items():
            if role.startswith("first-turn"):
                L.append(f"\n{role}:\n")
                L.append(table(["actor"] + list(next(iter(rows.values())).keys()),
                               [[a] + [f"{v:.3f}" for v in p.values()] for a, p in rows.items()]))
                continue
            L.append(f"\n{role}:\n")
            L.append(table(["readout", "separate − shared", "shared level"],
                           [[k, f(v), f"{v['shared_level']:.2f}" if "shared_level" in v else ""]
                            for k, v in rows.items()]))
    nt = r["natural"]
    if nt:
        L.append("\n## Behaviour: natural states of start:s (B5)\n")
        L.append(table(["readout", "separate − shared"], [[k, f(v)] for k, v in nt.items()]))
    ec = r.get("economy") or {}
    if ec:
        L.append("\n## Exploratory (amendment 2): natural self-play economy, 16 games × 600 turns per network\n")
        arms = [a for a in ("start",) + C.ARMS if a in next(iter(ec["table"].values()))]
        L.append(table(["quantity"] + arms + ["separate − shared", "shared − start", "separate − start"],
                       [[k] + [plain(v[a], 3) for a in arms] + [f(v["separate-shared"], 3),
                                                                 f(v.get("shared-start", {}), 3),
                                                                 f(v.get("separate-start", {}), 3)]
                        for k, v in ec["table"].items()]))
    return "\n".join(L)


def run() -> str:
    a = C.load_json("analysis")
    L = ["# M4.6f generated tables\n",
         "Intervals: 95% t over the six seed pairs (5 df); **bold** excludes 0; (k+/k−) counts seeds by sign. "
         "Resources throughout. 'separate − shared' is the paired difference within a seed.\n"]
    for ph in ("ext", "main"):
        if ph in a:
            L.append(f"\n# Phase {ph}\n")
            L.append(section(a[ph], ph))
    if "mid" in a:
        vc = a["mid"]["value_calib"]["table"]
        L.append("\n# 2.048M snapshots: value (cells, K 16)\n")
        L.append(table(["readout", "start", "shared", "separate", "separate − shared"],
                       [[k, plain(v["start"]), plain(v["shared"]), plain(v["separate"]), f(v["separate-shared"])]
                        for k, v in vc.items() if not k.startswith("natural")]))
    ed = C.load_json("extension_decision")
    if ed:
        L.append(f"\n# Extension rule\n\nfire: {ed['fire']} (a {ed['reasons']['a']}, b {ed['reasons']['b']}, "
                 f"c {ed['reasons']['c']})\n")
    text = "\n".join(L) + "\n"
    C.write_md("report", text)
    return text


__all__ = ["run", "section", "f", "table"]
