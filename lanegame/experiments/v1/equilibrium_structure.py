"""Structure of the equilibrium: where it mixes, how crash rewards are chosen, timing effects.

Besides the symmetric start, the experiment solves the game from a set of
constructed asymmetric roots ("situations") to show how equilibrium play
depends on wave size, wave position and reinforcement timing.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ... import GameConfig, LaneGame
from ...config import SIZE_NAMES
from ...render import state_row
from ...solver import solve_equilibrium
from ...state import State
from ..common import md_table, save

ACTION_SHORT = {0: "SP", 1: "SH", 2: "F"}   # original action labels

W = 2
SITUATIONS = [
    # (label, root, k0)  -- k0 parity fixes where the next reinforcements are
    ("A 2-stack at lane, B wave just spawned", State(0, 3 * W, (-4,), 0, -4, W, (), 0, 0, 0), 10),
    ("A 2-stack at lane, B wave two cells from its tower", State(0, 3 * W, (-2,), 0, -2, W, (), 0, 0, 0), 11),
    ("A 1-stack at lane, B wave just spawned", State(0, 2 * W, (-4,), 0, -4, W, (), 0, 0, 0), 10),
    ("A 1-stack at lane, B wave two cells from its tower", State(0, 2 * W, (-2,), 0, -2, W, (), 0, 0, 0), 11),
    ("A 1-stack at +1, B wave two cells from its tower", State(1, 2 * W, (-2,), 0, -2, W, (), 0, 0, 0), 11),
    ("A 1-stack at +1, B wave just spawned", State(1, 2 * W, (-4,), 0, -4, W, (), 0, 0, 0), 10),
    ("B wave held at A's freeze line, A wave arriving", State(-2, W, (), 0, 1, W, (), 0, 0, 0), 11),
    ("B 1-stack at A's freeze line, A wave arriving", State(-2, W, (), 0, 1, 2 * W, (), 0, 0, 0), 11),
    ("A wave crashed (siege 1), lanes reset", State(-4, W, (), 1, -3, W, (), 0, 0, 0), 10),
    ("Symmetric start", None, 0),
]


def fmt_strategy(legal, p):
    return ", ".join(f"{ACTION_SHORT[a]} {q:.2f}" for a, q in zip(legal, p) if q > 1e-9)


def run(horizon: int = 32) -> dict:
    g = LaneGame(GameConfig.v1(horizon=horizon))
    sol = solve_equilibrium(g)

    # 1. action support over all reachable (k, s)
    support = defaultdict(lambda: defaultdict(float))
    n_states = 0
    for k, layer in enumerate(sol.layers):
        for s in layer:
            n_states += 1
            la, lb = sol.legal[k][s]
            for side, (leg, p) in enumerate(((la, sol.x[k][s]), (lb, sol.y[k][s]))):
                for a, q in zip(leg, p):
                    support[side][ACTION_SHORT[a]] += q
    support = {side: {a: v / n_states for a, v in d.items()} for side, d in support.items()}

    # 2. mixed states: examples with Q matrices (A side, k = 12)
    mixed = []
    for s in sol.layers[12]:
        x, y = sol.x[12][s], sol.y[12][s]
        if np.count_nonzero(x > 1e-9) > 1 or np.count_nonzero(y > 1e-9) > 1:
            la, lb = sol.legal[12][s]
            mixed.append({"state": s.to_list(), "lane": state_row(s, g.cfg), "vision": [s.va, s.vb],
                          "V": sol.V[12][s], "A": fmt_strategy(la, x), "B": fmt_strategy(lb, y),
                          "Q": np.round(sol.Q[12][s], 3).tolist(),
                          "A_actions": [ACTION_SHORT[a] for a in la], "B_actions": [ACTION_SHORT[b] for b in lb]})

    # 3. crash-reward choices by size and by turns remaining
    choice = defaultdict(list)
    for key, x in sol.choice_x.items():
        k, pre, oa, ob = key
        if len(oa) > 1:
            size = [o for o in g.cfg.crash_options].index(tuple(oa))
            choice[(SIZE_NAMES[size], horizon - k)].append(float(x[1]))
    choice_rows = defaultdict(dict)
    for (size, rem), v in choice.items():
        choice_rows[size][rem] = float(np.mean(v))

    # 4. constructed situations
    sits = []
    for label, root, k0 in SITUATIONS:
        so = solve_equilibrium(g, root=root, k0=k0)
        r = so.root
        la, lb = so.legal[k0][r]
        sits.append({"label": label, "k0": k0, "lane": state_row(r, g.cfg), "value": so.value,
                     "A": fmt_strategy(la, so.x[k0][r]), "B": fmt_strategy(lb, so.y[k0][r]),
                     "Q": np.round(so.Q[k0][r], 3).tolist(),
                     "A_actions": [ACTION_SHORT[a] for a in la], "B_actions": [ACTION_SHORT[b] for b in lb]})

    md = [f"# Equilibrium structure (horizon {horizon})", "",
          f"Reachable decision nodes: {n_states}; nodes with a mixed equilibrium: {sol.stats.mixed_states}.", "",
          "## Average equilibrium probability of each action over reachable nodes", "",
          md_table(["player", "SP", "SH", "F"],
                   [(["A", "B"][s], *[round(support[s].get(a, 0.0), 3) for a in ("SP", "SH", "F")]) for s in (0, 1)]),
          "", "## Probability of choosing vision after a crash, by size and turns remaining", "",
          md_table(["size"] + [str(r) for r in sorted({r for d in choice_rows.values() for r in d})],
                   [(size, *[("%.2f" % d[r]) if r in d else "" for r in sorted({r for dd in choice_rows.values() for r in dd})])
                    for size, d in sorted(choice_rows.items())]),
          "", "## Equilibrium in constructed situations (lane row: global coordinates -4..+4)", ""]
    for d in sits:
        md += [f"**{d['label']}** (k = {d['k0']}, V* = {d['value']:.3f})", "", "```",
               "  " + d["lane"], "```",
               f"A plays {d['A']}; B plays {d['B']}. Q (rows A {d['A_actions']}, cols B {d['B_actions']}): `{d['Q']}`", ""]
    md += ["## Mixed-equilibrium nodes at k = 12", ""]
    for m in mixed:
        md += ["```", "  " + m["lane"] + f"   vision A{m['vision'][0]} B{m['vision'][1]}", "```",
               f"V* = {m['V']:.3f}; A: {m['A']}; B: {m['B']}; Q (rows {m['A_actions']}, cols {m['B_actions']}) = `{m['Q']}`", ""]
    save("equilibrium_structure", {"horizon": horizon, "support": support, "mixed_k12": mixed,
                                   "vision_choice": {k: v for k, v in choice_rows.items()},
                                   "situations": sits}, "\n".join(md))
    return sits


if __name__ == "__main__":
    run()
