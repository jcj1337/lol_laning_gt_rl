"""Part 5b and Part 6 tables of the M4.3d study: the M4.3c restricted games on the M4.3d configurations
(results/m43d/strategic_<variant>[_val].json.gz -> strategic.md)."""

from __future__ import annotations

from typing import Sequence

from ..m43c import economics as ec
from . import common as c
from .analysis import md_table

LABEL = {"base4": "four-action baseline", "econ": "econ (0.8)", "ot85": "econ, 0.85", "ot90": "econ, 0.9",
         "econ_aggro": "econ + tower aggro"}
SHORT = lambda s: s.replace("4", "") if s.endswith("4") else s


def _mix(d: dict) -> str:
    return ", ".join(f"{SHORT(k)} {p:.2f}" for k, p in sorted(d.items(), key=lambda kv: -kv[1]))


def table(variants: Sequence[str], tag: str = "", obj: str = "G") -> str:
    data = {v: c.load(f"strategic_{v}{tag}") for v in variants}
    data = {v: d for v, d in data.items() if d}
    scs = [s for s in ec.MATRIX_SCENARIOS if all(s in d["games"] for d in data.values())]
    rows = []
    for sc in scs:
        for v, d in data.items():
            s = d["games"][sc][obj]
            rows.append([sc, LABEL.get(v, v), f"{s['value']:+.2f}", _mix(s["x"]), _mix(s["y"])])
    return md_table(["scenario", "configuration", "value to A", "A's equilibrium", "B's equilibrium"], rows)


def scorecard(variants: Sequence[str], tag: str = "") -> str:
    """Weights of interest per configuration (objective J)."""
    concede = ("step_back4", "respect4", "pz_back")
    defend = ("walk_up_thin4", "pz_thin", "contest4", "pz_contest", "cautious4", "pz_cautious")
    chase = ("freeze_hunt4", "freeze_bully4", "hunt4", "shove_hunt4", "pz_hunt", "pz_bully", "sh_dive")
    freeze = ("freeze4", "freeze_hunt4", "freeze_bully4", "pz_keep", "pz_hunt", "pz_bully")
    vis = ("sh_vis",)
    shove = ("shove4", "walk_up_shove4", "build_shove4", "shove_hunt4", "sh_vis", "sh_dive", "sh_recall")
    rows = []
    for v in variants:
        d = c.load(f"strategic_{v}{tag}")
        if not d:
            continue
        g = d["games"]
        def w(sc, side, names):
            return sum(p for k, p in g[sc]["G"][side].items() if k in names) if sc in g else float("nan")
        distinct = len({(tuple(sorted(g[s]["G"]["x"])), tuple(sorted(g[s]["G"]["y"]))) for s in g})
        rows.append([LABEL.get(v, v),
                     f"{w('F10', 'y', shove):.2f} / {w('N10', 'y', shove):.2f}",
                     f"{w('F10g', 'y', concede):.2f} / {w('F10g', 'y', defend):.2f}",
                     f"{w('F10gl', 'y', concede):.2f} / {w('F10gl', 'y', defend):.2f}",
                     f"{w('F10g', 'x', freeze):.2f} / {w('P10g', 'x', freeze):.2f}",
                     f"{w('N10g', 'x', vis):.2f} / {w('N3', 'x', vis):.2f}",
                     f"{g['F10g']['G']['value']:+.2f} / {g['N10g']['G']['value']:+.2f}" if 'F10g' in g else "–",
                     str(distinct)])
    return md_table(["configuration", "B shoves at equal power (F10 / N10)", "B concedes / defends vs x1.76 (F10g)",
                     "B concedes / defends vs x2.04 (F10gl)", "A freezes ahead (F10g / P10g)",
                     "A uses Vision (N10g / N3)", "value of a x1.76 lead (F10g / N10g)", "distinct equilibria"], rows)


def run(variants: Sequence[str] = ("base4", "econ", "ot85", "ot90", "econ_aggro")) -> str:
    L = ["# M4.3d Part 5b: the M4.3c restricted games on the M4.3d configurations", "",
         "M4.3c scenarios (`m43c.economics.MATRIX_SCENARIOS`; F = freeze, N = neutral, P = freeze phase) with the "
         "M4.3c strategy sets plus `sh_vis` (Shove, Vision then Pressure) for both seats in F and N, and `sh_dive`, "
         "`sh_recall` in N. 64 paired games of 100 turns per cell, SeedSequence 79,200. Strategy names drop the "
         "suffix 4.", "", "## Scorecard (objective J)", "", scorecard(variants), "",
         "## Equilibria (J = discounted gold difference, 100 turns)", "", table(variants, obj="G"), "",
         "## Equilibria (G40 = undiscounted gold difference, first 40 turns)", "", table(variants, obj="G40")]
    pg0 = [v for v in ("econ", "econ_pg0", "ot90", "ot90_pg0") if c.load(f"strategic_{v}")]
    if len(pg0) == 4:
        L += ["", "## DIAGNOSTIC: without plate gold (the game after a tower's plates fall), 0.8 and 0.9", "",
              "Seven scenarios replayed with `plate_gold` 0 (same seeds as above).", "", table(pg0, obj="G")]
    val = [v for v in variants if c.load(f"strategic_{v}_val")]
    if val:
        L += ["", "## Validation: independent seeds (SeedSequence 79,700, 128 games per cell)", "",
              scorecard(val, "_val"), "", table(val, "_val", "G")]
    text = "\n".join(L)
    c.write_md("strategic", text)
    return text


__all__ = ["run", "table", "scorecard"]
