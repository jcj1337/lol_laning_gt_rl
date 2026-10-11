"""Representative lane situations ("probes") used by several revised experiments.

Each probe is a state in own-perspective coordinates (see state.py) with a decision index k0;
it is solved as the root of a K-turn game (so 'turns remaining' = K - k0). States are written
for the default HP unit W = 4 and rescaled for other W. Whether each probe state is reachable
from the standard start (in the same dynamics phase) is reported by `probe_reachability`.
"""

from __future__ import annotations

import numpy as np

from ..config import posture_of, wave_of
from ..state import State

K0 = 12          # decision index of every probe (20 turns remaining at K = 32)


def probes(W: int = 4):
    s = W // 4 if W % 4 == 0 else None

    def h(x):              # HP given in units of the default W = 4
        return x * W // 4
    return {
        "lane fight, even (both fresh waves)": State(0, h(4), (), 0, 0, h(4), (), 0, 0, 0),
        "B wave at A's freeze line, fighting A's wave": State(-2, h(4), (), 0, 1, h(4), (), 0, 0, 0),
        "B 2-wave stack at A's freeze line": State(-2, h(4), (), 0, 1, h(8), (), 0, 0, 0),
        "A 2-wave stack at lane, B wave just spawned": State(0, h(8), (-4,), 0, -4, h(4), (), 0, 0, 0),
        "A 2-wave stack at +1, B wave just spawned": State(1, h(8), (-4,), 0, -4, h(4), (), 0, 0, 0),
        "A 2-wave stack at +1, A has vision": State(1, h(8), (-4,), 0, -4, h(4), (), 0, 2, 0),
        "A's 3-wave crash under B's tower": State(-4, h(4), (), h(12), -3, h(4), (), 0, 0, 0),
        "B's 3-wave crash under A's tower (A defends)": State(-3, h(4), (), 0, -4, h(4), (), h(12), 0, 0),
    }


def probe_reachability(game, layers) -> dict:
    P = game.cfg.phase_period
    out = {}
    for name, s in probes(game.cfg.wave_hp).items():
        out[name] = any(s in set(layers[k]) for k in range(len(layers)) if k % P == K0 % P)
    return out


def strategy_str(cfg, legal, p) -> str:
    return ", ".join(f"{cfg.label(a)} {q:.2f}" for a, q in zip(legal, p) if q > 1e-9)


def contest_prob(legal, p) -> float:
    return float(sum(q for a, q in zip(legal, p) if posture_of(a) == 0))


def decompose(game, sol, k, s):
    """For each of A's actions: expected immediate farm, gank and continuation (crash rewards + V_{k+1})
    against B's equilibrium strategy at (k, s). Q = farm + gank + continuation."""
    la, lb = sol.legal[k][s]
    y = sol.y[k][s]
    Q = sol.Q[k][s]
    rows = {}
    for i, a in enumerate(la):
        farm = gank = 0.0
        for j, b in enumerate(lb):
            tr = game.transition(s, k, a, b)
            farm += y[j] * tr.farm_payoff
            gank += y[j] * tr.gank_payoff
        q = float(Q[i] @ y)
        rows[game.cfg.label(a)] = {"Q": q, "farm": farm, "gank": gank, "continuation": q - farm - gank,
                                   "wave": wave_of(a), "posture": posture_of(a)}
    return rows


def contest_advantage(rows) -> float:
    """Best Contest action minus best Safe action (Q against B's equilibrium strategy)."""
    c = max(r["Q"] for r in rows.values() if r["posture"] == 0)
    sf = max(r["Q"] for r in rows.values() if r["posture"] == 1)
    return c - sf


def solve_probe(cfg, state):
    from ..game import LaneGame
    from ..solver import solve_equilibrium
    g = LaneGame(cfg)
    sol = solve_equilibrium(g, root=state, k0=K0)
    la, lb = sol.legal[K0][state]
    rows = decompose(g, sol, K0, state)
    return {"value": sol.value, "A": strategy_str(cfg, la, sol.x[K0][state]),
            "B": strategy_str(cfg, lb, sol.y[K0][state]),
            "p_contest_A": contest_prob(la, sol.x[K0][state]),
            "contest_advantage": contest_advantage(rows), "actions": rows}
