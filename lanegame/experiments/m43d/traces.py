"""Representative lane traces of the M4.3d study (results/m43d/traces.md).

Each trace is one game through the environment from a crash-grid start, with the chance stream of seed
SeedSequence(TRACE_SEED, spawn_key=(e,)). Per turn: both actions, the lane after the turn in GLOBAL cells (A<hp> / B<hp>
packets, [A<hp>] A's crashed mass under B's tower, a / b the champions, `.` an empty cell), both champions' HP in bars,
B's farm this turn (collected / ideal and the tower cap), and the risk events of A (gank, tower shot, trade).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ...champions_m4 import M4State
from ...state import State
from ..m43c import harness as h
from ..m43c import strategies as st
from . import common as c
from . import sim
from .strategies import ATTACKERS, DEFENDERS

# (title, config, start, gap, A strategy, B strategy, episode, turns)
TRACES: Tuple[tuple, ...] = (
    ("Shove and leave: B farms the crash alone (L10, equal)", "econ", "S10", "equal", "sh_leave", "lh", 0, 12),
    ("Shove and leave at own-tower rate 0.9 (the same seed)", "ot90", "S10", "equal", "sh_leave", "lh", 0, 12),
    ("Shove + Pressure: A stands in B's tower cell (L10, equal)", "econ", "S10", "equal", "sh_press", "lh", 3, 14),
    ("Shove and dive against a last-hitter (L10, equal)", "econ", "S10", "equal", "sh_dive", "lh", 1, 12),
    ("Shove and dive against Safe (L10, A x1.76)", "econ", "S10", "large", "sh_dive", "safe", 1, 12),
    ("B punishes a sieger from its tower cell (L10, equal)", "econ", "S10", "equal", "sh_press", "punish", 3, 12),
    ("Vision, then dive (L10, A x1.76)", "econ", "S10", "large", "sh_vis_dive", "lh", 2, 14),
)


def lane_strip(lane_list, champ_list) -> str:
    lane = State(*[tuple(x) if isinstance(x, list) else x for x in lane_list])
    m4 = M4State.from_list(champ_list)
    cells = {x: [] for x in range(-4, 5)}
    if lane.ah > 0:
        cells[lane.ax].append(f"A{lane.ah}")
    for x in lane.ar:
        cells[x].append("A8")
    if lane.bh > 0:
        cells[-lane.bx].append(f"B{lane.bh}")
    for x in lane.br:
        cells[-x].append("B8")
    if lane.asg:
        cells[2].append(f"[A{lane.asg}]")
    if lane.bsg:
        cells[-2].append(f"[B{lane.bsg}]")
    if not m4.a.absent:
        cells[m4.a.pos].append("a")
    if not m4.b.absent:
        cells[-m4.b.pos].append("b")
    return " ".join(f"{(','.join(v) if v else '.'):>9s}" for x, v in cells.items())


def trace(cfg_name: str, start: str, gap: str, a: str, b: str, episode: int, turns: int) -> List[str]:
    cfg = c.variant_cfg(cfg_name)
    from .grid import STARTS
    base, level = STARTS[start]
    stt = h.start(base, cfg, gap=gap, level=level)
    seed = np.random.SeedSequence(c.TRACE_SEED, spawn_key=(episode,))
    g = sim.play(cfg, stt, (st.StrategyAgent(ATTACKERS[a]), st.StrategyAgent(DEFENDERS[b])), seed, turns, record=True)
    t = __import__("lanegame.champions_m4", fromlist=["context"]).context(cfg).t
    hdr = "turn  A action  B action  " + " ".join(f"{x:>+9d}" for x in range(-4, 5)) + \
        "  HP a/b     B farm (got/ideal, tower cap)  A risk"
    out = [hdr]
    lane0, m40, _ = stt
    out.append(f"{'start':5s} {'':9s} {'':9s} " + lane_strip(lane0.to_list(), m40.to_list()))
    for i, (inf, ex) in enumerate(zip(g["infos"], g["extra"])):
        after = inf["champ_after"]
        lv = inf["levels"]
        hp = [after[s][1] / t.hp_max[lv[s]] for s in (0, 1)]
        ev = []
        if inf["ganked"][0]:
            ev.append(f"ganked {inf['gank_damage'][0] / t.hp_max[inf['levels_before'][0]]:.2f}")
        if inf["tower_damage"][0]:
            ev.append("tower shot")
        if inf["trade_connected"][0]:
            ev.append(f"trade dealt {inf['trade_damage'][0] / t.hp_max[inf['levels_before'][1]]:.2f}")
        if inf["trade_connected"][1]:
            ev.append(f"B's trade dealt {inf['trade_damage'][1] / t.hp_max[inf['levels_before'][0]]:.2f}")
        if inf["died"][0]:
            ev.append("A DIES")
        if inf["died"][1]:
            ev.append("B DIES")
        if inf["plate_count"][0]:
            ev.append("plate")
        bx = ex[1]
        farm = f"{bx['collected']:.2f}/{bx['ideal']:.2f} cap {bx['tower']:.2f}" if bx["ideal"] else "-"
        out.append(f"{i:<5d} {inf['action_labels'][0]:9s} {inf['action_labels'][1]:9s} "
                   f"{lane_strip(inf['state_after'], after)}  {hp[0]:.2f}/{hp[1]:.2f}  {farm:28s}  "
                   f"p={inf['gank_p'][0]:.2f} " + "; ".join(ev))
    return out


def run() -> str:
    L = ["# M4.3d representative lane traces", "",
         "Global cells (A's tower at -2, B's at +2). `A12` an A packet of 12 HP, `[A14]` A's crashed mass under B's "
         "tower, `a` / `b` the champions. HP in bars after the turn. B farm: what B collected this turn of the ideal "
         "(0.1125 per enemy minion HP that died on B's ledger) and the part lost to the unconditional own-tower cap. "
         "A risk: A's gank probability this turn and the events that hit A. Seeds: SeedSequence(79,900, (e,)).", ""]
    for title, cf, sn, gp, a, b, e, n in TRACES:
        L += [f"## {title}", "", f"`{cf}`, start {sn}, gap {gp}, A `{a}`, B `{b}`, episode {e}", "", "```"]
        L += trace(cf, sn, gp, a, b, e, n)
        L += ["```", ""]
    text = "\n".join(L)
    c.write_md("traces", text)
    return text


__all__ = ["run", "trace", "TRACES", "lane_strip"]
