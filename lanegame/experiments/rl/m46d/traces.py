"""Explanatory trajectories of M4.6d (EXPLORATORY, illustrative only): single games on common seeds, turn by turn.

Each scenario is (environment, start cell, seat-A plan, seat-B plan) around the 8.192M general policy of seed 0 in
that environment (both seats). Rows: turn, both wave / behaviour components, end cells, HP fractions after the turn,
both fronts (cell:HP, each side's own coordinates), A's freeze flag after the turn, events, A's cumulative lead.
"""

from __future__ import annotations

from ....config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD
from . import cells as CL
from . import common as C
from . import plans as P

WN = {0: "LH", 1: "SH", 2: "TN", 3: "SL"}
BN = {B_HOLD: "H", B_YIELD: "Y", B_TRADE: "T", B_SAFE: "S", B_RECALL: "R"}

PLAUS5 = {"wave": "freeze", "cond": "edge", "L": 5, "hp_b": 0.6, "held": 12, "name": "plaus"}
EQUAL8 = {"wave": "freeze", "cond": "equal", "L": 8, "hp_b": 1.0, "held": 12, "name": "equal"}
SCENARIOS = [("C1", PLAUS5, "guard_sel", "policy"), ("C1", PLAUS5, "policy", "policy"),
             ("C1", PLAUS5, "guard_sel", "concede"), ("C1", PLAUS5, "guard_sel", "commit"),
             ("C1", EQUAL8, "passive", "policy"), ("C1", EQUAL8, "guard_walk", "policy")]


def run(turns: int = 30, seeds=(0, 1)) -> dict:
    from ....champions_m4 import context
    from ...m45b import engine as E
    from ..m44.records import holds_freeze
    from ..m45c import horizon as HZ
    res = []
    for arm, cell, fa, fb in SCENARIOS:
        cfg = C.setup(arm)
        tb = context(cfg).t
        start = CL.cell_start(cfg, cell)
        net = C.net_of(C.general(arm, 0))
        for sd0 in seeds:
            sd = C.TRACE_ROOT * 1000 + sd0
            pa = P.build(P.FREEZER[fa], HZ.CRNNet(net, start[2], sd, 0, turns))
            pb = P.build(P.BREAKER[fb], HZ.CRNNet(net, start[2], sd, 1, turns))
            g = E.play(cfg, start, pa, pb, sd, turns, keep_rows=True, keep_states=True, check_legal=False)
            rows, lead = [], 0.0
            for i, row in enumerate(g["rows"]):
                lane, m4, _ = g["states"][i + 1]
                lead += row["r"]
                ev = []
                for s in (0, 1):
                    if row["died"][s]:
                        ev.append(f"{'AB'[s]} dies")
                    if row["rec"][s]:
                        ev.append(f"{'AB'[s]} recalls")
                    if row["land"][s]:
                        ev.append(f"{'AB'[s]} trade lands")
                hp = []
                for s in (0, 1):
                    me = m4.side(s)
                    hp.append(1.0 if me.absent else
                              round(me.hp / tb.hp_max[tb.level_by_xp[min(me.xp, tb.xp_cap)]], 2))
                fa_, fb_ = lane.side(0)[:2], lane.side(1)[:2]
                acts = []
                for s in (0, 1):
                    b = row["beh"][s]
                    acts.append("--" if b not in BN else ("RECALL" if b == B_RECALL else f"{WN[row['a'][s]]}-{BN[b]}"))
                rows.append({"turn": i, "A": acts[0], "B": acts[1], "end": list(row["end"]), "hp": hp,
                             "front_A": f"{fa_[0]}:{fa_[1]}", "front_B": f"{fb_[0]}:{fb_[1]}",
                             "held": bool(holds_freeze(lane, 0)), "events": ev, "lead": round(lead, 2)})
            res.append({"arm": arm, "cell": CL.cell_label(cell), "A": fa, "B": fb, "seed": sd, "rows": rows,
                        "G20": g["summary"].get("G20")})
    data = {"meta": C.meta(stage="traces", label="EXPLORATORY, illustrative single games"), "traces": res}
    C.save_json("traces", data)
    return data


__all__ = ["run", "SCENARIOS"]
