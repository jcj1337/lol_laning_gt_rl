"""Is the lane part of the observation injective on the reachable lane? (M4 design review, item 1)

Reachability: the fixed point over (lane state, k mod 4) from the initial state, with every legal wave
action, objective NONE / PRESSURE and every champion reach value per side (the same superset of champion-mode
lane steps as freeze_enum.py "champ"). Vision timers and plate counters are projected out: they never affect
the lane, and the observation shows them exactly.

For every reachable (state, phase) and both seats this script checks:
  1. a front sharing a cell with a rear packet (the grid sums them);
  2. a grid cell whose HP exceeds hp_cap (the grid clips it);
  3. the largest number of rear packets in one cell;
  4. collisions: distinct reachable lane states with the same lane observation (grid + side features) at the
     same phase, and whether the colliding states have identical transitions for every action pair (harmless)
     or not (true aliasing).
Run from the repository root:  PYTHONPATH=. python results/m4_design/scripts/obs_alias.py
"""
import collections
import time

import numpy as np

from lanegame.config import GameConfig, NONE, PRESSURE
from lanegame.env.observation import observe
from lanegame.mechanics import initial_state, step_turn, wave_actions

cfg = GameConfig()
T = cfg.tower
NO_REACH = -10 ** 6
R = (NO_REACH, -2, -1, 0, 1, 2)
PER = cfg.phase_period


def side_opts(state, s):
    objs = (NONE, PRESSURE) if state.siege_of(s) > 0 else (NONE,)
    return [(w, o, r) for w in wave_actions(state, s, cfg) for o in objs for r in R]


def run(state, ph, oa, ob):
    return step_turn(state, ph, oa[0] + 6 * oa[1], ob[0] + 6 * ob[1], cfg, champion_reach=(oa[2], ob[2]))


def proj(st):
    return st._replace(va=0, vb=0, ap=0, bp=0)


t0 = time.time()
root = (initial_state(cfg), 0)
seen = {root}
frontier = [root]
while frontier:
    nxt = []
    for st, ph in frontier:
        for oa in side_opts(st, 0):
            for ob in side_opts(st, 1):
                s2 = (proj(run(st, ph, oa, ob).state), (ph + 1) % PER)
                if s2 not in seen:
                    seen.add(s2)
                    nxt.append(s2)
    frontier = nxt
print(f"reachable (lane state, phase) pairs: {len(seen)}  ({time.time() - t0:.0f} s)")

coloc = clip = 0
max_rears_cell = 0
max_cell_hp = 0
groups = collections.defaultdict(list)
for st, ph in seen:
    for s in (0, 1):
        fx, fh, rears, sg = st.side(s)[:4]
        cnt = collections.Counter(rears)
        if cnt:
            max_rears_cell = max(max_rears_cell, max(cnt.values()))
        if fh > 0 and fx in cnt:
            coloc += 1
        cells = collections.Counter()
        if fh > 0:
            cells[fx] += fh
        for x in rears:
            cells[x] += cfg.wave_hp
        if cells:
            max_cell_hp = max(max_cell_hp, max(cells.values()))
            if max(cells.values()) > cfg.hp_cap:
                clip += 1
    v = observe(st, ph, 0, cfg)[2:]            # drop k/K and (K-k)/K; the phase one-hot stays
    groups[v.tobytes()].append((st, ph))

print(f"front co-located with a rear (side-states): {coloc}")
print(f"grid cells above hp_cap (side-states): {clip};  max HP in one cell {max_cell_hp};  "
      f"max rears in one cell {max_rears_cell}")
coll = [g for g in groups.values() if len(g) > 1]
print(f"observation classes: {len(groups)};  classes with > 1 lane state: {len(coll)}")


def sig(res):
    return (proj(res.state), res.crash_hp, tuple(sorted(res.kills[0].items())), tuple(sorted(res.kills[1].items())),
            res.exposure, res.plates, res.thinned)


true_alias = 0
for g in coll:
    base, ph = g[0]
    for other, _ in g[1:]:
        same = True
        for oa in side_opts(base, 0):
            for ob in side_opts(base, 1):
                if sig(run(base, ph, oa, ob)) != sig(run(other, ph, oa, ob)):
                    same = False
                    break
            if not same:
                break
        if not same:
            true_alias += 1
            if true_alias <= 5:
                print("  aliased:", base.describe(), "|", other.describe())
print(f"colliding pairs with different transitions: {true_alias}")
