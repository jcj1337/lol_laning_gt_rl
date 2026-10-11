"""Simulator bounds used by the M4 evaluation guarantee (design §10.5; review item 2).

Exhaustive over the reachable lane (same superset of champion-mode lane steps as obs_alias.py / freeze_enum.py):
  * A_max: the largest alive minion HP of one side (lane packets + crashed mass) in any reachable state;
  * K_max: the largest enemy-minion HP dying in one turn for one side (all causes), over every action pair;
  * the largest crashed mass under a tower and the largest minion tower hits in one turn.
The analytical bounds in the design (K <= 27 per turn, tower hits <= 24 per turn) must dominate these.
Run:  PYTHONPATH=. python results/m4_design/scripts/lane_bounds.py
"""
import time

from lanegame.config import GameConfig, NONE, PRESSURE
from lanegame.mechanics import initial_state, minion_damage, step_turn, wave_actions

cfg = GameConfig()
NO_REACH = -10 ** 6
R = (NO_REACH, -2, -1, 0, 1, 2)
PER = cfg.phase_period


def side_opts(state, s):
    objs = (NONE, PRESSURE) if state.siege_of(s) > 0 else (NONE,)
    return [(w, o, r) for w in wave_actions(state, s, cfg) for o in objs for r in R]


def proj(st):
    return st._replace(va=0, vb=0, ap=0, bp=0)


def alive(st, s):
    fx, fh, rears, sg = st.side(s)[:4]
    return fh + cfg.wave_hp * len(rears) + sg


t0 = time.time()
root = (initial_state(cfg), 0)
seen = {root}
frontier = [root]
a_max = alive(root[0], 0)
k_max = siege_max = hits_max = 0
while frontier:
    nxt = []
    for st, ph in frontier:
        a_max = max(a_max, alive(st, 0), alive(st, 1))
        siege_max = max(siege_max, st.asg, st.bsg)
        for oa in side_opts(st, 0):
            for ob in side_opts(st, 1):
                res = step_turn(st, ph, oa[0] + 6 * oa[1], ob[0] + 6 * ob[1], cfg,
                                champion_reach=(oa[2], ob[2]), record=True)
                for s in (0, 1):
                    k_max = max(k_max, sum(res.kills[s].values()))
                    m, hits = st.siege_of(s), 0
                    for tr in res.trace:
                        if m > 0:
                            hits += minion_damage(m, cfg)
                        m = tr["siege"][s]
                    hits_max = max(hits_max, hits)
                s2 = (proj(res.state), (ph + 1) % PER)
                if s2 not in seen:
                    seen.add(s2)
                    nxt.append(s2)
    frontier = nxt
print(f"reachable (lane state, phase) pairs: {len(seen)}  ({time.time() - t0:.0f} s)")
print(f"A_max (alive HP of one side, lane + crashed mass): {a_max}")
print(f"K_max (enemy HP dying in one turn, one side): {k_max}   [analytical bound 27]")
print(f"largest crashed mass under a tower: {siege_max}   [siege_cap {cfg.siege_cap}]")
print(f"largest minion tower hits in one turn: {hits_max}   [analytical bound 24]")
