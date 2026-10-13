"""Tables of the M4.3d study from the stored crash grids (results/m43d/grid_*.json.gz).

Every number is a mean over the paired games of a cell (64 games of 100 turns unless stated). Windows: "12" = the first
12 turns (the first crash, its bounce and the next crash), "40", "all" = 100 turns. Efficiencies are shares of the
IDEAL farm (every enemy minion HP that died on the side's ledger at the lane rate 0.9, i.e. 0.1125 resources per HP),
so 1.0 = collecting everything at the lane rate; the loss shares (tower, position, posture, trade, absence) sum with
the efficiency to 1 (ledger.py).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import common as c

A, B = 0, 1


class Grid:
    def __init__(self, *tags: str):
        self.cells = {}
        for t in tags:
            g = c.load(f"grid_{t}")
            if g is None:
                raise FileNotFoundError(f"results/m43d/grid_{t}.json(.gz) not found")
            self.cells.update(g["cells"])

    def has(self, v, st, gp, a, b, mask=None) -> bool:
        return f"{v}|{st}|{gp}|{a}|{b}|{mask or '-'}" in self.cells

    def cell(self, v, st, gp, a, b, mask=None) -> dict:
        return self.cells[f"{v}|{st}|{gp}|{a}|{b}|{mask or '-'}"]

    def m(self, v, st, gp, a, b, key, w="40", mask=None) -> float:
        d = self.cell(v, st, gp, a, b, mask)
        if key in ("G", "G40", "G12") and key in d["per_game"]:
            return float(np.mean(d["per_game"][key]))
        return float(d[w][key][0])

    def ci(self, v, st, gp, a, b, key, w="40", mask=None) -> float:
        d = self.cell(v, st, gp, a, b, mask)
        if key in d["per_game"]:
            return c.mean_ci(d["per_game"][key])[1]
        x = d[w][key][1]
        return float("nan") if x is None else float(x)

    def paired(self, k1: tuple, k2: tuple, key: str = "G") -> Tuple[float, float]:
        """Mean and 95% interval of per-game key of cell k1 minus cell k2 (k = (v, st, gp, a, b, mask))."""
        x = np.array(self.cell(*k1)["per_game"][key])
        y = np.array(self.cell(*k2)["per_game"][key])
        return c.mean_ci(x - y)


def farm_profile(g: Grid, v, st, gp, a, b, side: str = "b", w: str = "40", mask=None) -> Dict[str, float]:
    """Efficiency and loss shares of one side (module docstring), plus XP and HP."""
    m = lambda k: g.m(v, st, gp, a, b, k, w, mask)
    ideal = m(f"x_ideal_{side}")
    out = {"ideal": ideal, "collected": m(f"x_collected_{side}")}
    for k in ("tower", "position", "posture", "trade", "absence"):
        out[k] = m(f"x_{k}_{side}") / ideal if ideal else float("nan")
    out["eff"] = out["collected"] / ideal if ideal else float("nan")
    out["tower_cell_share"] = m(f"x_ideal_tower_{side}") / ideal if ideal else float("nan")
    out["xp"] = m(f"xp_{side}")
    out["xp_missed"] = m(f"xp_missed_{side}")
    out["hp_lost"] = m(f"hp_lost_{side}")
    out["died"] = m(f"died_{side}")
    out["recalled"] = m(f"recalled_{side}")
    out["away"] = m(f"away_{side}")
    out["earned"] = m(f"earned_{side}")
    out["farm"] = m(f"farm_{side}")
    out["plate"] = m(f"plate_{side}")
    out["bounty"] = m(f"bounty_{side}")
    return out


def risk_profile(g: Grid, v, st, gp, a, b, side: str = "a", w: str = "40", mask=None) -> Dict[str, float]:
    m = lambda k: g.m(v, st, gp, a, b, k, w, mask)
    present = m(f"x_present_{side}")
    out = {"present": present}
    for cell in ("-2", "-1", "+0", "+1", "+2"):
        out[f"at{cell}"] = m(f"x_at{cell}_{side}") / present if present else float("nan")
    for k in ("exp_gank", "gank_dmg", "exp_tower", "tower_dmg", "trade_dmg"):
        out[k] = m(f"x_{k}_{side}")
    for k in ("ganked", "shot", "died", "recalled", "away", "hp_lost", "plate", "farm", "earned", "trade_conn",
              "bounty"):
        out[k] = m(f"{k}_{side}")
    return out


def fmt(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:+.{nd}f}" if nd and isinstance(x, float) and x < 0 else f"{x:.{nd}f}"


def pct(x) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def md_table(header: Sequence[str], rows: List[Sequence[str]]) -> str:
    L = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    L += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(L)


def best_replies(g: Grid, v, st, gp, a_set, b_set, key="G", mask=None) -> Dict[str, str]:
    """B's best reply (lowest mean value to A) to each A strategy."""
    out = {}
    for a in a_set:
        vals = [(g.m(v, st, gp, a, b, key, mask=mask), b) for b in b_set if g.has(v, st, gp, a, b, mask)]
        out[a] = min(vals)[1] if vals else "–"
    return out


__all__ = ["Grid", "farm_profile", "risk_profile", "fmt", "pct", "md_table", "best_replies"]
