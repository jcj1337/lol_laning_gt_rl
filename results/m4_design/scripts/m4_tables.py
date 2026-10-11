"""Analytical checks for the M4 design (no repository code changed).

1. Fixed-point tables: max HP (centi-HP) by level, attack power (basis points) by level and item tier.
2. XP thresholds anchored to League waves; turn at which levels are reached for absorption rates.
3. Duel model: both champions Trade every turn and always connect (worst case for the weaker side);
   probability that A kills B strictly first / simultaneous / B first, from full HP, for level, item and
   skill differences. Uses the M2 hit model h = clip(0.55 + 0.4 dm, .05, .95), q = clip(0.3 + 0.2 dm, 0, 1).
"""
from functools import lru_cache

LMAX, NMAX = 18, 2
HP_UNIT = 100          # centi-HP per base HP
BASE_HP = 6


def hp_max(L):
    # integer rounding of 600 * 1.05 ** (L - 1), half up
    return int(BASE_HP * HP_UNIT * 1.05 ** (L - 1) + 0.5)


def ap_bp(L, N):
    return int(10000 * 1.05 ** (L - 1) * 1.20 ** N + 0.5)


def dmg(base, L, N):
    """centi-HP of a hit of `base` HP by an attacker at level L with N tiers."""
    return (base * HP_UNIT * ap_bp(L, N) + 5000) // 10000


print("L  hp_max  hp_unit  AP(N=0) AP(N=1) AP(N=2)  hit1(N0) hit2(N0) hit1(N2) hit2(N2)")
for L in range(1, LMAX + 1):
    print(f"{L:2d} {hp_max(L):6d} {round(hp_max(L) / 6):7d}  {ap_bp(L,0):7d} {ap_bp(L,1):7d} {ap_bp(L,2):7d}"
          f"  {dmg(1,L,0):7d} {dmg(2,L,0):7d} {dmg(1,L,2):7d} {dmg(2,L,2):7d}")

# ---- XP thresholds: xp_per_hp = 2 (one 4-HP wave = 8 XP); to next level from L: 8 + 3 (L - 1)
def xp_next(L):
    return 8 + 3 * (L - 1)


cum = [0, 0]
for L in range(1, LMAX):
    cum.append(cum[-1] + xp_next(L))
# cum[L] = cumulative XP needed to BE level L (cum[1] = 0)
league = [0, 0]
for L in range(1, LMAX):
    league.append(league[-1] + 180 + 100 * L)
print("\nL  cumXP  waves(=cum/8)  League waves (cum/270)  turns at 100% / 85% / 70% absorption (2 XP per turn max)")
for L in range(2, LMAX + 1):
    w = cum[L] / 8
    print(f"{L:2d} {cum[L]:5d} {w:8.2f} {league[L] / 270:10.2f}      {cum[L] / 2:6.1f} {cum[L] / 1.7:6.1f} {cum[L] / 1.4:6.1f}")


# ---- duel: both trade every turn, both connect, simultaneous; HP in centi-HP
def duel(La, Na, ma, Lb, Nb, mb):
    dm = ma - mb
    ha = min(max(0.55 + 0.4 * dm, .05), .95)
    hb = min(max(0.55 - 0.4 * dm, .05), .95)
    qa = min(max(0.3 + 0.2 * dm, 0), 1)
    qb = min(max(0.3 - 0.2 * dm, 0), 1)
    da = [(0, 1 - ha), (dmg(1, La, Na), ha * (1 - qa)), (dmg(2, La, Na), ha * qa)]   # damage A deals
    db = [(0, 1 - hb), (dmg(1, Lb, Nb), hb * (1 - qb)), (dmg(2, Lb, Nb), hb * qb)]

    @lru_cache(maxsize=None)
    def win(x, y):          # (P(A kills first), P(simultaneous), P(B first)) from HP (x, y)
        p_stay = da[0][1] * db[0][1]
        tot = [0.0, 0.0, 0.0]
        for d1, p1 in da:
            for d2, p2 in db:
                if d1 == 0 and d2 == 0:
                    continue
                p = p1 * p2
                nx, ny = x - d2, y - d1
                if nx <= 0 and ny <= 0:
                    tot[1] += p
                elif ny <= 0:
                    tot[0] += p
                elif nx <= 0:
                    tot[2] += p
                else:
                    r = win(nx, ny)
                    for i in range(3):
                        tot[i] += p * r[i]
        return tuple(t / (1 - p_stay) for t in tot)

    return win(hp_max(La), hp_max(Lb))


print("\nDuel to death from full HP (A first / both / B first); both Trade and connect every turn")
cases = [
    ("equal L1", (1, 0, .5, 1, 0, .5)),
    ("equal L11", (11, 0, .5, 11, 0, .5)),
    ("A +1 level (6 v 5)", (6, 0, .5, 5, 0, .5)),
    ("A +2 levels (7 v 5)", (7, 0, .5, 5, 0, .5)),
    ("A +3 levels (8 v 5)", (8, 0, .5, 5, 0, .5)),
    ("A +1 tier (L6)", (6, 1, .5, 6, 0, .5)),
    ("A +2 tiers (L6)", (6, 2, .5, 6, 0, .5)),
    ("A +2 levels +1 tier", (7, 1, .5, 5, 0, .5)),
    ("A +2 levels +2 tiers", (8, 2, .5, 6, 0, .5)),
    ("skill .6 v .4, equal econ", (6, 0, .6, 6, 0, .4)),
    ("skill .7 v .3, equal econ", (6, 0, .7, 6, 0, .3)),
    ("skill .8 v .2, equal econ", (6, 0, .8, 6, 0, .2)),
    ("A skill .7 v B .3 but B +2 levels +1 tier", (5, 0, .7, 7, 1, .3)),
    ("A skill .7 v B .3 but B +2 levels +2 tiers", (6, 0, .7, 8, 2, .3)),
    ("A skill .6 v B .4 but B +2 levels +1 tier", (5, 0, .6, 7, 1, .4)),
    ("A skill .8 v B .2 but B +3 levels +2 tiers", (5, 0, .8, 8, 2, .2)),
    ("caps: L18 N2 v L18 N0", (18, 2, .5, 18, 0, .5)),
]
for name, c in cases:
    a, s, b = duel(*c)
    print(f"  {name:45s} {a:.3f} {s:.3f} {b:.3f}")


# expected damage ratio per connected exchange (A dealt / B dealt), normalized by the victim's max HP
def edge(La, Na, ma, Lb, Nb, mb):
    dm = ma - mb
    ha = min(max(0.55 + 0.4 * dm, .05), .95); hb = min(max(0.55 - 0.4 * dm, .05), .95)
    qa = min(max(0.3 + 0.2 * dm, 0), 1); qb = min(max(0.3 - 0.2 * dm, 0), 1)
    ea = ha * ((1 - qa) * dmg(1, La, Na) + qa * dmg(2, La, Na)) / hp_max(Lb)
    eb = hb * ((1 - qb) * dmg(1, Lb, Nb) + qb * dmg(2, Lb, Nb)) / hp_max(La)
    return ea / eb


print("\nEffective trade edge (A's expected damage as a fraction of B's max HP, over B's on A)")
for name, c in cases:
    print(f"  {name:45s} {edge(*c):.3f}")
