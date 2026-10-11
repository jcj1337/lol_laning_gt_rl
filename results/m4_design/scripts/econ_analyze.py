"""Summarize econ_results.npz (econ_measure.py) and derive XP-threshold candidates."""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "econ_results.npz"))
keys = sorted({tuple(k.split("|")[:3]) for k in Z.files})


def g(h, a, b, f):
    return Z[f"{h}|{a}|{b}|{f}"]


def summary(h, a, b, lo=0, hi=None, label=""):
    sl = slice(lo, hi)
    d = {f: g(h, a, b, f)[:, sl] for f in ("avail", "farm", "bycell", "bydist", "away", "hp0", "att", "con", "dealt",
                                             "taken", "gdmg", "tdmg", "died", "recalled", "plates")}
    K = d["avail"].shape[1]
    print(f"\n=== {h} {a} (A) vs {b} (B)  turns {lo}-{lo + K - 1}  ({d['avail'].shape[0]} eps) {label}")
    for s, nm in ((0, a), (1, b)):
        av = d["avail"][..., s].mean()
        fm = d["farm"][..., s].mean()
        cell = d["bycell"][..., s, :].mean((0, 1))
        dist = d["bydist"][..., s, :].mean((0, 1))
        cum = np.cumsum(dist)
        pres = ~d["away"][..., s]
        con = d["con"][..., s]
        dmg_per_con = d["dealt"][..., s][con].mean() if con.any() else float("nan")
        hp = d["hp0"][..., s][pres]
        hpdist = np.bincount(hp, minlength=7)[1:7] / max(1, hp.size)
        scale = 32
        print(f" [{s}] {nm}")
        print(f"   enemy minion HP died/turn {av:.3f} (= {av * .25 * scale:.2f} res/32t); gold/turn {fm:.3f} "
              f"(= {fm * scale:.2f}/32t), frac {fm / (av * .25):.3f}; plates/32t {d['plates'][..., s].mean() * scale:.2f}")
        print("   HP died by own cell -3..3 (/turn):", " ".join(f"{c:.3f}" for c in cell))
        print("   HP died by dist from present champ 0..4 (/turn):", " ".join(f"{c:.3f}" for c in dist[:5]),
              f"| within<=0 {cum[0]:.3f} <=1 {cum[1]:.3f} <=2 {cum[2]:.3f} (frac of died {cum[0]/av:.2f} {cum[1]/av:.2f} {cum[2]/av:.2f})")
        print(f"   present frac {pres.mean():.3f}; trades attempted/turn {d['att'][..., s].mean():.3f} connected/turn "
              f"{con.mean():.3f}; dmg/connected {dmg_per_con:.2f}; dmg dealt/32t {d['dealt'][..., s].mean() * scale:.2f}")
        print(f"   per 32t: deaths {d['died'][..., s].mean() * scale:.2f}, recalls {d['recalled'][..., s].mean() * scale:.2f}, "
              f"turns away {d['away'][..., s].mean() * scale:.2f}, dmg taken {d['taken'][..., s].mean() * scale:.2f} "
              f"(gank {d['gdmg'][..., s].mean() * scale:.2f}, tower {d['tdmg'][..., s].mean() * scale:.2f})")
        print("   HP at turn start (present) 1..6:", " ".join(f"{x:.3f}" for x in hpdist), f"mean {hp.mean():.2f}")


for h, a, b in keys:
    if h == "h32":
        summary(h, a, b)
for h, a, b in keys:
    if h == "h200":
        summary(h, a, b, 0, 32, "(long game, opening)")
        summary(h, a, b, 32, 200, "(long game, steady state; per-32t figures are rates x 32)")

# ---- per-turn profile for the long mirror
print("\n=== h200 profiles (8-turn blocks): HP died/turn, XP r<=1, XP r<=2, gold/turn, away frac, mean HP0 (side avg)")
for h, a, b in keys:
    if h != "h200":
        continue
    av, dist, fm, aw, hp0 = (g(h, a, b, f) for f in ("avail", "bydist", "farm", "away", "hp0"))
    cum = np.cumsum(dist, axis=-1)
    print(f" {a} vs {b}")
    for lo in list(range(0, 64, 8)) + [64, 100, 150]:
        hi = lo + 8 if lo < 64 else (lo + 36 if lo == 64 else lo + 50)
        sl = slice(lo, hi)
        print(f"   t{lo:3d}-{hi - 1:3d}: " + " | ".join(
            f"{av[:, sl, s].mean():.2f} {cum[:, sl, s, 1].mean():.2f} {cum[:, sl, s, 2].mean():.2f} "
            f"{fm[:, sl, s].mean():.3f} {aw[:, sl, s].mean():.2f}" for s in (0, 1)))

# ---- XP thresholds (mutual hold_farmer)
h, a, b = "h200", "hold_farmer", "hold_farmer"
dist = g(h, a, b, "bydist")
cum = np.cumsum(dist, axis=-1)                       # (E, K, 2, D)
for r in (1, 2):
    xp = cum[..., r]                                 # HP within range r of present champ, per turn
    x2 = xp.reshape(-1, 200)                         # pool both seats: (2E, K) -- symmetric matchup
    x2 = np.concatenate([xp[:, :, 0], xp[:, :, 1]])
    early, steady = x2[:, :32].mean(), x2[:, 32:].mean()
    sd_game = x2[:, 32:].sum(1).std() / 168
    print(f"\nXP rate (range {r}): turns 0-31 {early:.3f}/turn, turns 32-199 {steady:.3f}/turn, overall {x2.mean():.3f}; "
          f"per-player sd of steady mean {sd_game:.3f}")

LEVELS = (2, 6, 11, 18)


def cum_thresholds(shape, c):
    """XP needed to REACH level n (from 1), n = 1..18."""
    steps = []
    for L in range(1, 18):
        if shape == "linear":
            steps.append(c * (L + 1))
        elif shape == "league_user":                 # c * (80 + 100 (L-1)) / 180, as specified
            steps.append(c * (80 + 100 * (L - 1)) / 180)
        elif shape == "league_true":                 # real LoL increments 280, 380, ... scaled so L1->2 = c
            steps.append(c * (280 + 100 * (L - 1)) / 280)
    return np.concatenate([[0.0], np.cumsum(steps)])  # index n-1


def fit_c(shape, rate, t11=120.0, t18=300.0):
    th = cum_thresholds(shape, 1.0)
    # minimise squared log error to targets t11, t18 at constant rate
    c = np.exp(0.5 * (np.log(t11 * rate / th[10]) + np.log(t18 * rate / th[17])))
    return c


def turn_from_curve(curve_mean, thr, steady):
    """First turn t (1-based count of turns played) with cumulative mean XP >= thr; extrapolate past the data."""
    cs = np.cumsum(curve_mean)
    idx = np.searchsorted(cs, thr)
    if idx < len(cs):
        return idx + 1
    return len(cs) + (thr - cs[-1]) / steady


for r in (1, 2):
    x2 = np.concatenate([cum[:, :, 0, r], cum[:, :, 1, r]])
    steady = x2[:, 32:].mean()
    curve = x2.mean(0)
    print(f"\n--- XP range {r}: steady rate {steady:.3f} HP/turn")
    for shape in ("linear", "league_user", "league_true"):
        c = fit_c(shape, steady)
        for cc in (c, round(c, 1) if c < 10 else round(c)):
            th = cum_thresholds(shape, cc)
            const = [th[n - 1] / steady for n in LEVELS]
            real = [turn_from_curve(curve, th[n - 1], steady) for n in LEVELS]
            print(f"  {shape:12s} c={cc:7.3f}: cum XP to L2/6/11/18 = " + "/".join(f"{th[n - 1]:.1f}" for n in LEVELS)
                  + " | turns const-rate " + "/".join(f"{t:.0f}" for t in const)
                  + " | turns from measured mean curve " + "/".join(f"{t:.0f}" for t in real))
