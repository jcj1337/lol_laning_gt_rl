"""Generated tables of OM1 (results/om1/report.md) from smoke.json, selection.json and analysis.json."""

from __future__ import annotations

import numpy as np

from . import common as C


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


def run() -> str:
    A = C.load("analysis")
    sm = C.load("smoke")
    pr = C.load("protocol")
    sel, ident, val, rob, cl = A["selection"], A["identifiability"], A["value"], A["robustness"], A["classification"]
    L = ["# OM1 opponent-modelling feasibility: generated tables", "",
         f"Protocol `{pr['record']['document']}` SHA-256 `{pr['record']['sha256'][:16]}…` (frozen "
         f"{pr['meta']['timestamp']}). Values: the responder's earned-gold lead at turn {C.READ_TURN} (= E_20 at turn "
         f"{C.LANE_END} in expectation), resources, mean over paired-seat pairs.", ""]
    L += ["## Smoke", "", table(["check", "result"], [[k, v] for k, v in sm["checks"].items()]), "",
          "Throughput (turns/s, one worker): " + ", ".join(f"{k} {v:.0f}" for k, v in
                                                       sm["turns_per_s_one_worker"].items()), ""]
    # ---- selection
    menu = ["pi"] + list(C.LIBRARY)
    V = sel["values"]
    rows = []
    for t in C.LIBRARY:
        best = max(menu, key=lambda r: V[f"{r}|{t}"])
        rows.append([t, f"{V[f'pi|{t}']:+.2f}", best, f"{V[f'{best}|{t}']:+.2f}", f"{sel['selection_gain'][t]:+.2f}",
                     sel["rstar"][t]])
    L += ["## Stage A: selection (every menu responder against every library type; 64 pairs)", "",
          table(["type", "pi's value", "best menu responder", "its value", "gain over pi", "r* (gain >= 0.25)"], rows),
          "", f"Mean value over the library: pi {sel['mean_value']['pi']:+.2f}; best fixed candidate "
              f"{sel['rfix_candidate']} {sel['mean_value'][sel['rfix_candidate']]:+.2f}; r_fix = {sel['rfix']}.", ""]
    # ---- identifiability
    tm = ident["type_mean"]
    chk = [str(t) for t in C.CHECK_TURNS]
    L += ["## Identifiability (pi's games; 6 seeds x 96 pairs x 2 orientations per type)", "",
          table(["turns observed"] + chk,
                [["true type's posterior >= 0.9 (full signal)"] + [pct(tm["p_true_ge_0.9_full"][t]) for t in chk],
                 ["... coarse signal (no wave choice)"] + [pct(tm["p_true_ge_0.9_coarse"][t]) for t in chk],
                 ["... full, eps 0.01"] + [pct(tm["p_true_ge_0.9_eps0.01"][t]) for t in chk],
                 ["... full, eps 0.2"] + [pct(tm["p_true_ge_0.9_eps0.2"][t]) for t in chk],
                 ["MAP type correct"] + [pct(tm["map_correct"][t]) for t in chk],
                 ["decision-relevant (r* of MAP = r* of truth)"] + [pct(tm["decision_relevant"][t]) for t in chk],
                 ["decision-relevant, coarse signal"] + [pct(tm["decision_relevant_coarse"][t]) for t in chk]]),
          "", f"Mean over the 15 types. **I (decision-relevant identification at turn {C.ID_TURN} >= 2/3): "
              f"{'met' if ident['I_met'] else 'not met'}.**", ""]
    rows = []
    for t, o in ident["per_type"].items():
        rows.append([t] + [pct(o["p_true_ge_0.9_full"][x]) for x in ("5", "10", "20", "40", "94")]
                    + [pct(o["p_true_ge_0.9_coarse"]["40"]), pct(o["decision_relevant"]["40"]),
                       ", ".join(f"{k} {100 * v:.0f}%" for k, v in o["map_at_40"].items())])
    L += [table(["true type", "P>=0.9 @5", "@10", "@20", "@40", "@94", "coarse @40", "decision-relevant @40",
                 "MAP after 40 turns"], rows), ""]
    # ---- value
    S = val["seeds"]
    L += ["## Value of information (stage B; six seeds as units, mean over the 15 library types)", "",
          table(["contrast", "lead at 114 (E_20 at 94)", "lane lead at 94", "lead at 150"],
                [["switcher − pi (G_B)", mci(S["G_B"]), mci(S["G_B_lane"]), mci(S["G_B_end"])],
                 ["oracle − pi (G_O)", mci(S["G_O"]), mci(S["G_O_lane"]), mci(S["G_O_end"])],
                 ["oracle − best fixed responder (VOI)", mci(S["VOI"]), "", ""],
                 ["pi's own value (reference)", mci(S["V_pi"]), "", ""]]), ""]
    rows = []
    for t, o in val["per_type"].items():
        s = val["switcher"][t]
        rows.append([t, o["rstar"], mci(o["pi"]), mci(o["O-pi"]), mci(o["B-pi"]), mci(o["O-pi_end"]),
                     pct(s["switched_share"]), f"{s['first_switch_median']:.0f}" if np.isfinite(
                         s["first_switch_median"]) else "never",
                     ", ".join(f"{k} {100 * v:.0f}%" for k, v in list(s["turn_share_by_responder"].items())[:3])])
    L += [table(["type", "r*", "pi's value", "oracle − pi", "switcher − pi", "oracle − pi at 150",
                 "switcher: turns switched", "first switch (median turn)", "switcher's play by responder"], rows), ""]
    # ---- robustness
    rows = []
    for o, x in rob["per_opponent"].items():
        rows.append([o, mci(x["B-pi"]), mci(x["B-pi_end"]),
                     ", ".join(f"{k} {100 * v:.0f}%" for k, v in x["mistaken_for_at_40"].items()),
                     ", ".join(f"{k} {100 * v:.0f}%" for k, v in list(x["turn_share_by_responder"].items())[:3])])
    L += ["## Robustness (stage C: out-of-library opponents)", "",
          table(["opponent", "switcher − pi (lead at 114)", "at 150", "believed to be (MAP after 40 turns)",
                 "switcher's play by responder"], rows), "",
          f"Pooled per seed: {mci(rob['pooled_per_seed'])}; R violated: {rob['R_violated']}.", ""]
    L += ["## Classification (protocol §4)", "",
          table(["criterion", "result"],
                [["I met", cl["I_met"]], ["G_B material", cl["G_B_material"]], ["G_O small", cl["G_O_small"]],
                 ["narrow types", ", ".join(cl["narrow_types"]) or "none"], ["R violated", cl["R_violated"]],
                 ["**outcome**", f"**{cl['outcome']}: {cl['label']}**"]]), ""]
    txt = "\n".join(L)
    C.write_md("report", txt)
    return txt


__all__ = ["run"]
