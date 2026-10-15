"""Tables of the M4.3c stabilization study (docs/m4/m4_3/m4_3c_strategic_economics_calibration.md) from the stored results:
results/m43c/economics_<variant>.json.gz, econ_probes_<variant>.json(.gz), sequences.json(.gz). Nothing is simulated
here except the XP-weight sensitivity, which re-solves the stored matrices with other weights (the per-game discounted
gold and XP channels are stored, so every weight is scored on the same games).

    python -m lanegame.experiments.m43c calibration_report        -> results/m43c/calibration.md
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .common import load, write_md
from .economics import MATRIX_SCENARIOS, SCENARIO_LABEL, VARIANTS

THREAT_VARIANTS = ("base", "k2", "tf100", "candidate")
PAIRS_DECOMP = (("N10", "shove4", "farm4"), ("N10", "shove4", "thin4"), ("N10", "shove4", "hunt4"),
                ("N10", "slowpush4", "farm4"), ("F10", "freeze4", "walk_up4"), ("F10", "freeze4", "walk_up_shove4"),
                ("F10g", "freeze_hunt4", "walk_up4"), ("F10g", "freeze_hunt4", "walk_up_shove4"))


def _short(d: dict) -> str:
    return ", ".join(f"{k.replace('4', '')} {v:.2f}" for k, v in sorted(d.items(), key=lambda x: -x[1]))


def _m(x, nd=2, sign=True) -> str:
    if x is None or (isinstance(x, (list, tuple)) and (not x or x[0] is None)):
        return "-"
    v = x[0] if isinstance(x, (list, tuple)) else x
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def resolve(cells: dict, sc: str, lam: float) -> dict:
    """The matrix game of scenario sc with the XP weight lam, from the stored per-game channels."""
    from ...matrix_game import solve_matrix_game
    As, Bs = MATRIX_SCENARIOS[sc][4], MATRIX_SCENARIOS[sc][5]
    M = np.array([[np.mean(np.asarray(cells[f"{sc}|{a}|{b}"]["G"]) + lam * np.asarray(cells[f"{sc}|{a}|{b}"]["X"]))
                   for b in Bs] for a in As])
    sol = solve_matrix_game(M)
    return {"value": float(sol.value), "x": {a: float(p) for a, p in zip(As, sol.x) if p > 1e-6},
            "y": {b: float(p) for b, p in zip(Bs, sol.y) if p > 1e-6}}


def equilibria_table(L: List[str], variants: List[str], obj: str = "J") -> None:
    L.append(f"| scenario | variant | value to A | A's equilibrium | B's equilibrium | B concedes | B conditional | "
             "A chases |")
    L.append("|---|---|---|---|---|---|---|---|")
    for sc in MATRIX_SCENARIOS:
        for v in variants:
            r = load(f"economics_{v}")
            if r is None or sc not in r["games"]:
                continue
            g = r["games"][sc][obj]
            L.append(f"| {sc} | `{v}` | {g['value']:+.2f} | {_short(g['x'])} | {_short(g['y'])} | "
                     f"{g['B_concede']:.2f} | {g['B_conditional']:.2f} | {g['A_chase']:.2f} |")
    L.append("")


def decomposition_table(L: List[str], variants: List[str]) -> None:
    L.append("| scenario | A | B | variant | J (100) | A − B gold (40) | farm A/B | plates A/B | bounty A/B | XP A − B | "
             "deaths A/B | Recalls A/B | turns away A/B |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sc, a, b in PAIRS_DECOMP:
        for v in variants:
            r = load(f"economics_{v}")
            if r is None:
                continue
            c = r["cells"][f"{sc}|{a}|{b}"]
            ag = c["agg40"]
            J = float(np.mean(c["G"]))
            L.append(f"| {sc} | {a} | {b} | `{v}` | {J:+.2f} | {_m(ag['earned_diff'])} | "
                     f"{ag['farm_a'][0]:.2f}/{ag['farm_b'][0]:.2f} | {ag['plate_a'][0]:.2f}/{ag['plate_b'][0]:.2f} | "
                     f"{ag['bounty_a'][0]:.2f}/{ag['bounty_b'][0]:.2f} | {_m(ag['xp_diff'], 1)} | "
                     f"{ag['died_a'][0]:.2f}/{ag['died_b'][0]:.2f} | {ag['recalled_a'][0]:.2f}/{ag['recalled_b'][0]:.2f} | "
                     f"{ag['away_a'][0]:.1f}/{ag['away_b'][0]:.1f} |")
    L.append("")


def threat_table(L: List[str], variant: str, scs=("F10", "F10g", "F10gl")) -> None:
    r = load(f"economics_{variant}")
    if r is None:
        return
    L.append("| scenario | A | B | B deaths | B Recalls | B HP lost (bars) | from trades / ganks / tower | A trades "
             "connected | A farm | B farm | A − B gold |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for sc in scs:
        for a in ("freeze4", "freeze_hunt4", "freeze_bully4"):
            for b in ("walk_up4", "walk_up_shove4", "step_back4", "cautious4", "contest4"):
                ag = r["cells"][f"{sc}|{a}|{b}"]["agg40"]
                L.append(f"| {sc} | {a} | {b} | {ag['died_b'][0]:.2f} | {ag['recalled_b'][0]:.2f} | "
                         f"{ag['hp_lost_b'][0]:.2f} | {ag['dmg_trade_b'][0]:.2f} / {ag['dmg_gank_b'][0]:.2f} / "
                         f"{ag['dmg_tower_b'][0]:.2f} | {ag['trade_conn_a'][0]:.1f} | {ag['farm_a'][0]:.2f} | "
                         f"{ag['farm_b'][0]:.2f} | {_m(ag['earned_diff'])} |")
    L.append("")


def lambda_table(L: List[str], variant: str = "base", lams=(0.0, 1 / 16, 1 / 8, 1 / 4)) -> None:
    r = load(f"economics_{variant}")
    if r is None:
        return
    L.append("| scenario | " + " | ".join(f"λ = {lam:g}: value · B's equilibrium" for lam in lams) + " |")
    L.append("|---|" + "---|" * len(lams))
    for sc in MATRIX_SCENARIOS:
        cells = []
        for lam in lams:
            g = resolve(r["cells"], sc, lam)
            cells.append(f"{g['value']:+.2f} · {_short(g['y'])}")
        L.append(f"| {sc} | " + " | ".join(cells) + " |")
    L.append("")


def probes_tables(L: List[str], variant: str = "base") -> None:
    p = load(f"econ_probes_{variant}")
    if p is None:
        return
    if "power" in p:
        L.append("**Value of a power lead** (neutral lane, level 10, both seats the same strategy, 100 turns, 64 paired "
                 "games): A's discounted gain over the equal start, J(lead) − J(0), ± 95%.\n")
        pairs = sorted({k.split("|")[2] for k in p["power"]})
        L.append("| lead | " + " | ".join(pairs) + " |")
        L.append("|---|" + "---|" * len(pairs))
        for kind, leads in (("invest", (2, 5, 10, 20, 50)), ("levels", (1, 2, 3))):
            for lead in leads:
                row = []
                for pr in pairs:
                    v = p["power"].get(f"{kind}|{lead}|{pr}")
                    row.append(f"{v['J_gain'][0]:+.2f} ± {v['J_gain'][1]:.2f}" if v else "-")
                lab = f"+{lead} invested" if kind == "invest" else f"+{lead} level{'s' if lead > 1 else ''}"
                L.append(f"| {lab} | " + " | ".join(row) + " |")
        L.append("")
    if "death" in p:
        L.append("**Price of a death** (to the dead champion B; the bounty included; J over 100 turns, ± 95%; 64 paired "
                 "games; d = turns away).\n")
        states = []
        for k in p["death"]:
            lab = k.rsplit("|", 1)[0]
            if lab not in states:
                states.append(lab)
        ds = sorted({int(k.rsplit("|", 1)[1]) for k in p["death"]})
        L.append("| state | " + " | ".join(f"d = {d}" for d in ds) + " | extra A plates (d = 3) |")
        L.append("|---|" + "---|" * (len(ds) + 1))
        for lab in states:
            row = [f"{p['death'][f'{lab}|{d}']['cost'][0]:.2f} ± {p['death'][f'{lab}|{d}']['cost'][1]:.2f}" for d in ds]
            L.append(f"| {lab} | " + " | ".join(row) + f" | {p['death'][f'{lab}|3']['plates_a_gain'][0]:+.2f} |")
        L.append("")
    if "horizon" in p:
        L.append("**Laning horizon from s0** (300 turns, 32 games per pair; turns are medians over the games in which "
                 "the event happened, with the share of games in brackets).\n")
        L.append("| A v B | 1st plate (level) | 2nd plate (level) | 3rd plate (level) | first tower's last plate | "
                 "both level 18 | plate share of income | bounty share | deaths A/B | Recalls A/B |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for key, d in p["horizon"].items():
            def ev(name):
                t, sh = d[f"{name}_turn"]
                lv = d.get(f"{name}_level")
                return "never" if t is None else f"{t:.0f} (L{lv:.1f}; {sh:.0%})"
            t, sh = d["final_plate_first_tower"]
            L.append(f"| {key.replace('|', ' v ')} | {ev('plate1')} | {ev('plate2')} | {ev('plate3')} | "
                     f"{'never' if t is None else f'{t:.0f} ({sh:.0%})'} | {d['both_level18'][0]:.0f} | "
                     f"{d['plate_share']:.3f} | {d['bounty_share']:.3f} | {d['deaths'][0]:.1f}/{d['deaths'][1]:.1f} | "
                     f"{d['recalls'][0]:.1f}/{d['recalls'][1]:.1f} |")
        L.append("")


CONTEST_B = ("walk_up4", "walk_up_thin4", "walk_up_shove4", "contest4", "hunt4", "shove4", "pz_walk", "pz_thin",
             "pz_contest")
FREEZE_A = ("freeze4", "freeze_hunt4", "freeze_bully4", "pz_keep", "pz_hunt", "pz_bully")
BREAK_B = ("walk_up_shove4", "shove4")


def scorecard(v: str, obj: str = "J") -> Optional[dict]:
    """The hypotheses of the brief, read off variant v's solved games (equilibrium weights; values to A):
      contest_eq    B's weight on contesting strategies in the equal freeze (F10; P10 for the phase game)
      wary_gap      B's weight on conceding + conditional strategies against a much stronger freezer (F10g, F10gl)
      freeze_gap    A's weight on freeze strategies when far ahead (F10g, F10gl)
      break_any     the largest weight B puts on breaking by Shove in any freeze scenario
      lead_value    V(F10g) - V(F10): what a x1.76 lead is worth from the freeze (the same for the neutral lane)
      supports      distinct equilibrium supports over the scenarios (state dependence)"""
    r = load(f"economics_{v}")
    if r is None:
        return None
    G = r["games"]

    def w(sc, side, names):
        if sc not in G:
            return None
        d = G[sc][obj]["y" if side == "B" else "x"]
        return sum(p for k, p in d.items() if k in names)

    def val(sc):
        return G[sc][obj]["value"] if sc in G else None

    sup = {tuple(sorted(G[sc][obj]["x"])) + ("|",) + tuple(sorted(G[sc][obj]["y"])) for sc in G}
    out = {"contest_eq": w("F10", "B", CONTEST_B), "contest_eq_phase": w("P10", "B", CONTEST_B),
           "wary_gap": [w("F10g", "B", ("step_back4", "respect4", "cautious4")),
                        w("F10gl", "B", ("step_back4", "respect4", "cautious4"))],
           "wary_gap_phase": [w("P10g", "B", ("pz_back", "pz_cautious")), w("P10gl", "B", ("pz_back", "pz_cautious"))],
           "freeze_gap": [w("F10g", "A", FREEZE_A), w("F10gl", "A", FREEZE_A)],
           "freeze_gap_phase": [w("P10g", "A", FREEZE_A), w("P10gl", "A", FREEZE_A)],
           "break_any": max((w(sc, "B", BREAK_B) or 0.0) for sc in G if sc.startswith(("F", "P"))),
           "lead_value_freeze": (val("F10g") - val("F10")) if val("F10g") is not None else None,
           "lead_value_neutral": (val("N10g") - val("N10")) if val("N10g") is not None else None,
           "supports": len(sup)}
    return out


def scorecard_table(L: List[str], variants: List[str], obj: str = "J") -> None:
    L.append("| variant | B contests (equal freeze; phase game) | B wary vs a much stronger freezer (F10g, F10gl; "
             "phase) | A freezes when ahead (F10g, F10gl; phase) | B breaks by Shove (max) | value of a x1.76 lead "
             "(freeze / neutral) | distinct supports |")
    L.append("|---|---|---|---|---|---|---|")

    def f(x):
        return "-" if x is None else f"{x:.2f}"

    for v in variants:
        s = scorecard(v, obj)
        if s is None:
            continue
        L.append(f"| `{v}` | {f(s['contest_eq'])}; {f(s['contest_eq_phase'])} | "
                 f"{f(s['wary_gap'][0])}, {f(s['wary_gap'][1])}; {f(s['wary_gap_phase'][0])}, "
                 f"{f(s['wary_gap_phase'][1])} | {f(s['freeze_gap'][0])}, {f(s['freeze_gap'][1])}; "
                 f"{f(s['freeze_gap_phase'][0])}, {f(s['freeze_gap_phase'][1])} | {f(s['break_any'])} | "
                 f"{f(s['lead_value_freeze'])} / {f(s['lead_value_neutral'])} | {s['supports']} |")
    L.append("")


def robustness_table(L: List[str]) -> None:
    """The equilibria on the original seeds (root 76,000, 64 games) against the fresh rerun (76,500, 128 games)."""
    for v, orig in (("base", "base"), ("candidate", "ot80_tf80_k2_dS1")):
        r2, r1 = load(f"economics_{v}_r2"), load(f"economics_{orig}")
        if r2 is None or r1 is None:
            continue
        L.append(f"**`{v}`** (original: `{orig}`)\n")
        L.append("| scenario | original value · A · B | fresh value · A · B |")
        L.append("|---|---|---|")
        for sc in r2["games"]:
            g1, g2 = r1["games"][sc]["J"], r2["games"][sc]["J"]
            L.append(f"| {sc} | {g1['value']:+.2f} · {_short(g1['x'])} · {_short(g1['y'])} | {g2['value']:+.2f} · "
                     f"{_short(g2['x'])} · {_short(g2['y'])} |")
        L.append("")


def pilot_table(L: List[str]) -> None:
    p = load("pilot")
    if p is None:
        return
    L.append(p["meta"].get("label", "EXPLORATORY") + ".\n")
    L.append("| run | entropy last | Shove share (all / neutral / freezing / frozen on) | Safe | deaths / Recalls per side "
             "| J v shove4 / farm4 / freeze_hunt4 |")
    L.append("|---|---|---|---|---|---|")
    for k, b in p["behavior"].items():
        tr = p.get("train", {}).get(k, {})
        sh = [b.get(s, {}).get("share", {}).get("Shove") for s in ("all", "neutral", "freezing", "frozen_on")]
        refs = [p["refs"].get(f"{k}|{ref}", [None])[0] for ref in ("shove4", "farm4", "freeze_hunt4")]
        L.append(f"| {k.replace('|', ' seed ')} | {tr.get('entropy_first_last', [None, float('nan')])[1]:.2f} | "
                 + " / ".join("-" if x is None else f"{x:.2f}" for x in sh)
                 + f" | {b['behavior']['share']['Safe']:.2f} | {b['deaths']:.2f} / {b['recalls']:.2f} | "
                 + " / ".join("-" if x is None else f"{x:+.2f}" for x in refs) + " |")
    L.append("")


def run(variants: Optional[List[str]] = None) -> str:
    variants = [v for v in (variants or list(VARIANTS)) if load(f"economics_{v}") is not None]
    L = ["# M4.3c stabilization, Stage B: economic and strategic calibration tables\n",
         "Generated by `python -m lanegame.experiments.m43c calibration_report` from results/m43c/economics_*.json.gz "
         "and econ_probes_*.json; reading: docs/m4/m4_3/m4_3c_strategic_economics_calibration.md. Values to A in resources; "
         "J = discounted (γ = 0.99) gold difference over 100 turns; restricted matrix games over scripted strategy "
         "sets (not equilibria of the laning game).\n",
         "Scenarios: " + "; ".join(f"**{k}** {SCENARIO_LABEL[k]}" for k in MATRIX_SCENARIOS) + ".\n",
         "Variants: " + "; ".join(f"`{v}` {VARIANTS[v][0]}" for v in variants) + ".\n",
         "## Scorecard of the brief's hypotheses (equilibrium weights of the restricted games, objective J)\n"]
    scorecard_table(L, variants, "J")
    L.append("Objective J + XP/8:\n")
    scorecard_table(L, variants, "Jx")
    L.append("## Equilibria of the restricted games (objective J)\n")
    equilibria_table(L, variants, "J")
    L.append("## Equilibria with the XP-inclusive objective (J + XP/8)\n")
    equilibria_table(L, variants, "Jx")
    L.append("## XP weight sensitivity (base dynamics)\n")
    lambda_table(L, "base")
    L.append("## Decomposition of representative cells (40-turn window; J over 100)\n")
    decomposition_table(L, [v for v in ("base", "pg0", "ot90") if v in variants])
    L.append("## Threat diagnosis: the frozen-on champion B (40-turn window)\n")
    for v in [x for x in THREAT_VARIANTS if x in variants]:
        L.append(f"**variant `{v}`**\n")
        threat_table(L, v)
    L.append("## Robustness: fresh seeds, 128 games per cell\n")
    robustness_table(L)
    L.append("## PPO pilot\n")
    pilot_table(L)
    L.append("## Probes (base)\n")
    probes_tables(L, "base")
    for v in variants:
        if v != "base" and load(f"econ_probes_{v}") is not None:
            L.append(f"## Probes (`{v}`)\n")
            probes_tables(L, v)
    md = "\n".join(L)
    write_md("calibration", md)
    return md


__all__ = ["run", "resolve"]
