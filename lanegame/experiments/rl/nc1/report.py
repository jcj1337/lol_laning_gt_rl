"""The generated tables of NC1 (results/nc1/report.md) from the stage summaries in results/nc1/."""

from __future__ import annotations

import numpy as np

from . import common as C


def _f(x, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{d}f}"


def _ci(m, h, d: int = 2) -> str:
    if m is None:
        return "n/a"
    return f"{m:+.{d}f} ± {h:.{d}f}" if h is not None and np.isfinite(h) else f"{m:+.{d}f}"


def _table(head, rows) -> list:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |"
                                                                         for r in rows]


def smoke(L: list) -> None:
    d = C.load_json("smoke")
    if not d:
        return
    L += ["## Smoke: pairing with M5 and throughput", ""]
    L += _table(["run", "status", "updates identical to M5", "first rollout identical", "steady transitions/s"],
                [[k, v["status"], v["identical_to_m5_updates"], v["first_rollout_identical"],
                  f"{v['steady_transitions_per_s']:.0f}"] for k, v in d["runs"].items()])
    L += ["", f"Checks: {d['checks']}.", ""]


def q1(L: list, name: str = "q1", title: str = "Q1. Unit usage of the stored networks") -> None:
    d = C.load_json(name)
    if not d:
        return
    L += [f"## {title}", ""]
    rows = []
    for n, r in d["networks"].items():
        a, b = r["layers"][0], r["layers"][-1]
        rows.append([n, r["inputs"], r["states"], " × ".join(map(str, r["hidden"])), a["inactive"],
                     f"{a['saturation_share']:.2f}", f"{a['effective_rank']:.0f}", b["inactive"],
                     f"{b['saturation_share']:.2f}", f"{b['effective_rank']:.0f}", b["components_99"]])
    L += _table(["network", "inputs", "states", "hidden", "L1 inactive", "L1 saturation", "L1 eff. rank",
                 "L2 inactive", "L2 saturation", "L2 eff. rank", "L2 components 99%"], rows)
    if "rule" in d:
        L += ["", f"Rule (layer 2, median over networks): {d['rule']}."]
    L += [""]


def q2(L: list) -> None:
    from .probe import TARGETS, UNION
    d = C.load_json("q2")
    if not d:
        return
    L += ["## Q2. The capacity probe", ""]
    rows = []
    for t in TARGETS:
        p = d["per_target"][str(t)]
        rows.append([t, f"{p['distances']['D']:.3f}", f"{p['distances']['N_KL_G_A']:.3f}",
                     f"{p['kl']['F_A']['P_seatA_teacherA']:.4f}", f"{p['kl']['F_G']['N_teacherG']:.4f}"]
                    + [f"{p['lambda'][u]:.3f}" for u in UNION])
    L += ["Lost specialization λ (0 = keeps all of the specialist's difference; 1 = plays like the generalist), held "
          "out, mean of three fit seeds:", ""]
    L += _table(["target", "D = KL(A‖G)", "KL(G‖A) natural", "floor F_A", "floor F_G"] + [f"λ {u}" for u in UNION],
                rows)
    rows = []
    for t in TARGETS:
        p = d["per_target"][str(t)]
        rows.append([t] + [f"{p['kappa'][u]:+.4f}" for u in UNION] + [" / ".join(str(e) for e in p["kl"]["U128"]["epochs"])])
    L += ["", "Natural excess κ (nats, KL to G on held-out natural states above the F_G floor):", ""]
    L += _table(["target"] + [f"κ {u}" for u in UNION] + ["U128 epochs"], rows)
    b = d.get("behaviour")
    if b:
        rows = []
        for t in TARGETS:
            r = b["per_target"][str(t)]
            rows.append([t, r["n_states"], _ci(r["attacker - target"]["mean"], r["attacker - target"]["ci"])]
                        + [_ci(r[f"{u} - target"]["mean"], r[f"{u} - target"]["ci"]) for u in UNION]
                        + [_ci(r["U256 - U128"]["mean"], r["U256 - U128"]["ci"]),
                           _ci(r["U128F - U128"]["mean"], r["U128F - U128"]["ci"])])
        L += ["", "Behaviour on E4's held-out gap states (G100, cells and semi-natural pooled, 95% interval over "
                  "states):", ""]
        L += _table(["target", "states", "attacker − G"] + [f"{u} − G" for u in UNION] + ["U256 − U128",
                                                                                         "U128F − U128"], rows)
        L += ["", "Pooled retention ρ: " + ", ".join(f"{u} {v:.2f}" for u, v in b["pooled_rho"].items()) + "."]
    if d.get("natural"):
        rows = [[k, _f(v["estimate"]), f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"] for k, v in d["natural"].items()]
        L += ["", "Natural play against G from s0 (64 paired seeds, W = 460):", ""]
        L += _table(["student", "estimate", "95% interval"], rows)
    L += ["", f"Reproduction of E4's records: {d.get('reproduction')}.", "",
          f"Decision rules: {d['rules']}. Natural regressions: {d['natural_regressions'] or 'none'}.", ""]


def q3(L: list) -> None:
    d = C.load_json("q3")
    if not d:
        return
    L += ["## Q3. Value capacity on M4.6e's labels", ""]
    L += _table(["network", "P1 edge + plaus", "P1 large + gap", "natural MAE"],
                [[a, _ci(*v["edge+plaus"]), _ci(*v["large+gap"]), _ci(*v["natural_mae"])]
                 for a, v in d["summary"].items()])
    L += ["", "Minus 2 × 128 (paired by policy):", ""]
    L += _table(["network", "P1 edge + plaus", "P1 large + gap", "natural MAE"],
                [[a, _ci(*v["edge+plaus"]), _ci(*v["large+gap"]), _ci(*v["natural_mae"], d=3)]
                 for a, v in d["minus_2x128"].items()])
    rp = C.load_json("q3_repro") or {}
    L += ["", f"Width binds for the value: {d['width_binds_for_value']} (2 × 256 natural MAE ratio "
              f"{d['natural_mae_ratio_2x256']:.3f}). Reproduction of M4.6e's stored 2 × 128 fit: "
              f"{rp.get('pred_identical')} (max difference {rp.get('max_abs_diff')}).", ""]


def training(L: list) -> None:
    d = C.load_json("training")
    if not d:
        return
    L += ["## Stage 2. Training", ""]
    L += _table(["run", "status", "transitions", "updates", "hours", "transitions/s"],
                [[k, v["status"], v.get("agent_transitions"), v.get("updates"),
                  f"{(v.get('elapsed_s') or 0) / 3600:.2f}", f"{v.get('transitions_per_s') or 0:.0f}"]
                 for k, v in sorted(d["runs"].items())])
    L += [""]


def evaluation(L: list) -> None:
    d = C.load_json("evaluation")
    if not d:
        return
    L += ["## Stage 3. Evaluation (arm − paired control, mean over six seeds, 95% t interval)", ""]
    L += ["E1, head to head from s0 (lead at turn 114, 256 paired-seat games per seed):", ""]
    L += _table(["arm", "mean ± 95%", "per seed"],
                [[a, _ci(v["mean"], v["ci"]), ", ".join(f"{x:+.2f}" for x in v["per_seed"].values())]
                 for a, v in d["E1"].items()])
    L += ["", "E2, the limitation families (arm − control in seat A on the same rollouts, averaged over both seat-B "
              "networks):", ""]
    rows = []
    for arm, fams in d["E2"].items():
        for f, r in fams.items():
            for k, v in r["stats"].items():
                if k.startswith("arm - control") or k.startswith("specialist"):
                    rows.append([arm, f, k, _ci(v["mean"], v["ci"]), v.get("n_seeds")])
    L += _table(["arm", "family", "readout", "mean ± 95%", "seeds"], rows)
    e3 = d["E3"]
    L += ["", "E3, the log duel ratio's held-out R² from the last hidden layer:", ""]
    L += _table(["network", "R²"], [[k, _ci(*v, d=3)] for k, v in e3["probe"]["summary"].items()])
    L += ["", "E3, the critic's predicted and the realized strength premium at the E4 gap cells (resources):", ""]
    L += _table(["network", "predicted (V)", "realized (G150, self-play)"],
                [[k, _ci(*v["predicted"]), _ci(*v["realized"])] for k, v in e3["critic_premium"]["summary"].items()])
    L += ["", "E3, training health over the last 200 updates (mean):", ""]
    L += _table(["run", "entropy", "approx KL", "explained variance", "clip fraction", "grad norm"],
                [[k, f"{v['entropy']:.3f}", f"{v['approx_kl']:.4f}", f"{v['explained_variance']:.3f}",
                  f"{v['clipfrac']:.3f}", f"{v['grad_norm']:.3f}"] for k, v in e3["health"].items()])
    L += ["", f"Verdicts: {d['verdicts']}.", ""]


def run() -> str:
    L = ["# NC1 generated tables", "", "Pre-registration: docs/capacity/nc1_preregistration.md "
         f"(SHA-256 {(C.load_json('preregistration') or {}).get('record', {}).get('sha256', 'n/a')}).", ""]
    smoke(L)
    q1(L)
    q2(L)
    q3(L)
    training(L)
    q1(L, "q1_pilot", "Q1 on the pilot finals and their controls")
    evaluation(L)
    return C.write_md("report", "\n".join(L) + "\n")
