"""Quantization check for M4 combat: does a level / item lead change duel outcomes smoothly?

Variants:
  base      one roll per connected trade: 0 / 1 / 2 base hits (M2 distribution), HP 6 base
  strikes2  two independent half-strength rolls per trade (0 / 0.5 / 1 each): same mean, half the variance
Start HP: full, or both sides uniformly random in [40%, 100%] of max HP (5% steps) -- mimics mid-game states.
Output: P(A kills first) - P(B kills first) (the duel edge, simultaneous deaths count 0).
"""
from functools import lru_cache
import itertools

HPU = 100


def hp_max(L):
    return int(600 * 1.05 ** (L - 1) + 0.5)


def ap(L, N):
    return int(10000 * 1.05 ** (L - 1) * 1.20 ** N + 0.5)


def hit(base_centi, L, N):
    return (base_centi * ap(L, N) + 5000) // 10000


def dist(L, N, h, q, strikes):
    if strikes == 1:
        return [(0, 1 - h), (hit(100, L, N), h * (1 - q)), (hit(200, L, N), h * q)]
    one = [(0, 1 - h), (hit(50, L, N), h * (1 - q)), (hit(100, L, N), h * q)]
    out = {}
    for (d1, p1), (d2, p2) in itertools.product(one, one):
        out[d1 + d2] = out.get(d1 + d2, 0) + p1 * p2
    return sorted(out.items())


def duel_edge(La, Na, ma, Lb, Nb, mb, strikes=1, start="full"):
    dm = ma - mb
    ha, hb = min(max(.55 + .4 * dm, .05), .95), min(max(.55 - .4 * dm, .05), .95)
    qa, qb = min(max(.3 + .2 * dm, 0), 1), min(max(.3 - .2 * dm, 0), 1)
    da, db = dist(La, Na, ha, qa, strikes), dist(Lb, Nb, hb, qb, strikes)
    p0 = dict(da).get(0, 0) * dict(db).get(0, 0)

    @lru_cache(maxsize=None)
    def w(x, y):
        tot = 0.0
        for d1, p1 in da:
            for d2, p2 in db:
                if d1 == 0 and d2 == 0:
                    continue
                nx, ny = x - d2, y - d1
                if nx <= 0 and ny <= 0:
                    v = 0.0
                elif ny <= 0:
                    v = 1.0
                elif nx <= 0:
                    v = -1.0
                else:
                    v = w(nx, ny)
                tot += p1 * p2 * v
        return tot / (1 - p0)

    HA, HB = hp_max(La), hp_max(Lb)
    if start == "full":
        return w(HA, HB)
    fr = [f / 20 for f in range(8, 21)]
    vals = [w(max(1, int(HA * f)), max(1, int(HB * g))) for f in fr for g in fr]
    return sum(vals) / len(vals)


cases = [
    ("+1 level (6 v 5)", (6, 0, .5, 5, 0, .5)),
    ("+2 levels (7 v 5)", (7, 0, .5, 5, 0, .5)),
    ("+3 levels (8 v 5)", (8, 0, .5, 5, 0, .5)),
    ("+4 levels (9 v 5)", (9, 0, .5, 5, 0, .5)),
    ("+1 tier (L6)", (6, 1, .5, 6, 0, .5)),
    ("+2 tiers (L6)", (6, 2, .5, 6, 0, .5)),
    ("+2 levels +1 tier", (7, 1, .5, 5, 0, .5)),
    ("skill .6 v .4", (6, 0, .6, 6, 0, .4)),
    ("skill .7 v .3 vs +2 lvl +2 tiers", (6, 0, .7, 8, 2, .3)),
]
print(f"{'case':36s} {'full,1':>8s} {'rand,1':>8s} {'full,2':>8s} {'rand,2':>8s}")
for name, c in cases:
    r = [duel_edge(*c, strikes=s, start=st) for s in (1, 2) for st in ("full", "rand")]
    print(f"{name:36s} {r[0]:8.3f} {r[1]:8.3f} {r[2]:8.3f} {r[3]:8.3f}")
