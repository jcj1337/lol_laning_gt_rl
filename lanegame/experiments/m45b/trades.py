"""Priority B: minion damage, retaliation and the economics of trading into waves.

1. `bench_table` (exact, one turn): on the trade bench (`states.trade_bench`: level x enemy wave HP x own-wave cover x
   power gap x HP), A's Trade versus A's Hold against a fixed B action (Hold, Trade = mutual, Safe), by exact
   enumeration of the turn's trade and tower dice (`exact.expect`, ganks off: they do not depend on the trade). Reported
   per cell: A's expected damage dealt and taken (bars of each victim's max HP; minion fire split by retaliation and
   positional units), the HP exchange (B's loss minus A's, in bars), the conditional numbers given a landed or a
   missed trade, and the one-turn farm and XP changes against Hold.
2. `sweep` (exact): the same table under minion_damage_alpha x minion_damage_growth x minion_retaliation_delay, and the
   pre-registered qualitative criteria of the M4.5b report (§ Priority B) evaluated per configuration.
3. `hp_price` (paired rollouts): the empirical shadow price of HP: the paired G100 difference of a start with A at
   full HP and one with A at HP - delta, divided by delta (resources per bar), by level, under scripted continuation.
4. `multi_turn` (paired rollouts): "Trade on the first turn" versus "Hold on the first turn", then both sides follow
   a continuation strategy, from bench states: G5 / G20 / G100, deaths and Recalls within 20 turns.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

import numpy as np

from ...config import B_HOLD, B_SAFE, B_TRADE, W_HOLD, champ_act
from . import common as Cm
from . import engine as E
from . import exact as X
from . import states as S
from . import strategies as ST

A_TRADE, A_HOLD = champ_act(W_HOLD, B_TRADE), champ_act(W_HOLD, B_HOLD)
B_ACTIONS = {"hold": champ_act(W_HOLD, B_HOLD), "trade": champ_act(W_HOLD, B_TRADE), "safe": champ_act(W_HOLD, B_SAFE)}
LEVELS, WAVES, COVERS, GAPS = (2, 6, 11, 18), (8, 16, 24), ("cover", "nocover", "ahead"), ("equal", "large")
KEYS = ("dealt_bar", "minion_bar", "u_ret", "u_pos", "taken_bar", "farm", "xp", "died", "connected", "landed",
        "front_hp", "crash_hp")


def cell(cfg, level: int, wave: int, cover: str, gap: str, b_act: str, hp=(1.0, 1.0)) -> dict:
    """One bench cell: exact expectations of A Trade and A Hold against B's action."""
    lane, m4, k = S.trade_bench(cfg, level, wave, cover, gap, hp)
    b = B_ACTIONS[b_act]
    tr = X.expect(cfg, lane, m4, k, A_TRADE, b)
    ho = X.expect(cfg, lane, m4, k, A_HOLD, b)
    m, h = tr["mean"], ho["mean"]
    out = {"level": level, "wave": wave, "cover": cover, "gap": gap, "b": b_act, "hp": list(hp)}
    for key in KEYS:
        out[key] = m[key]
        out[key + "_hold"] = h[key]
    # HP exchange of the Trade turn: B's HP loss minus A's (bars of each one's max HP); and its change against Hold
    out["exchange"] = m["taken_bar"][1] - m["taken_bar"][0]
    out["exchange_hold"] = h["taken_bar"][1] - h["taken_bar"][0]
    out["d_farm"] = [m["farm"][s] - h["farm"][s] for s in (0, 1)]
    out["d_xp"] = [m["xp"][s] - h["xp"][s] for s in (0, 1)]
    out["minion_share"] = (m["minion_bar"][0] / m["dealt_bar"][0]) if m["dealt_bar"][0] > 0 else float("nan")
    for g in ("A_landed", "A_missed"):
        if g in tr["cond"]:
            c = tr["cond"][g]
            out[g] = {"p": tr["mass"][g], "dealt_bar": c["dealt_bar"][0], "minion_bar": c["minion_bar"][0],
                      "u_ret": c["u_ret"][0], "u_pos": c["u_pos"][0], "taken_bar": c["taken_bar"][0],
                      "exchange": c["taken_bar"][1] - c["taken_bar"][0]}
    return out


def bench_table(cfg, levels=LEVELS, waves=WAVES, covers=COVERS, gaps=GAPS, b_acts=("hold", "trade", "safe"),
                hps=((1.0, 1.0), (0.5, 1.0))) -> List[dict]:
    return [cell(cfg, L, w, c, g, b, hp) for L in levels for w in waves for c in covers for g in gaps
            for b in b_acts for hp in hps]


# ---------------------------------------------------------------------------------------------- criteria
# The value of the Trade DECISION is its HP exchange minus Hold's (delta): without own cover the enemy wave fires on
# positional aggro whether A trades or not, so the raw exchange would charge the Trade for the position. Fixed after
# the baseline bench and before any calibration run (report § Priority B).
CRITERIA = {
    "B1": "L2, equal power, own cover, B Hold: trading into a 24-HP wave is not HP-positive (delta <= 0)",
    "B2": "L2, equal power, own cover, B Hold: trading into an 8-HP wave is clearly HP-positive (delta >= +0.05 bar)",
    "B3": "L2, A x1.76, own cover, B Hold: trading into a 24-HP wave is clearly HP-positive (delta >= +0.05 bar)",
    "B4": "L18 (128 invested), equal power, own cover, 24-HP wave: minion fire <= 15% of the trade damage dealt",
    "B5": "equal power, own cover, 24-HP wave: the minion share of the trade damage falls strictly with level",
    "B6": "L2, equal power, own cover: delta falls strictly with the enemy wave (8 > 16 > 24 HP)",
}


def delta(r: dict) -> float:
    """The HP exchange of the Trade minus that of Hold (bars), for seat A."""
    return r["exchange"] - r["exchange_hold"]


def criteria(rows: List[dict]) -> Dict[str, dict]:
    def get(**kw):
        for r in rows:
            if all(r[k] == v for k, v in kw.items()) and r["hp"] == [1.0, 1.0]:
                return r
        raise KeyError(kw)
    out = {}
    v = delta(get(level=2, wave=24, cover="cover", gap="equal", b="hold"))
    out["B1"] = {"value": v, "pass": v <= 0}
    v = delta(get(level=2, wave=8, cover="cover", gap="equal", b="hold"))
    out["B2"] = {"value": v, "pass": v >= 0.05}
    v = delta(get(level=2, wave=24, cover="cover", gap="large", b="hold"))
    out["B3"] = {"value": v, "pass": v >= 0.05}
    v = get(level=18, wave=24, cover="cover", gap="equal", b="hold")["minion_share"]
    out["B4"] = {"value": v, "pass": v <= 0.15}
    sh = [get(level=L, wave=24, cover="cover", gap="equal", b="hold")["minion_share"] for L in LEVELS]
    out["B5"] = {"value": sh, "pass": all(a > b for a, b in zip(sh, sh[1:]))}
    dw = [delta(get(level=2, wave=w, cover="cover", gap="equal", b="hold")) for w in WAVES]
    out["B6"] = {"value": dw, "pass": all(a > b for a, b in zip(dw, dw[1:]))}
    return out


SWEEP = {"minion_damage_alpha": (30, 50, 80), "minion_damage_growth": (1.05, 1.08, 1.12),
         "minion_retaliation_delay": (0, 1)}


def _sweep_job(args):
    over = dict(args)
    c = Cm.cfg("r2", **over)
    rows = bench_table(c, covers=COVERS, gaps=GAPS, b_acts=("hold",), hps=((1.0, 1.0),))
    return {"overrides": over, "criteria": criteria(rows), "rows": rows}


def sweep(workers: int = Cm.WORKERS) -> List[dict]:
    grid = [tuple(zip(SWEEP, v)) for v in itertools.product(*SWEEP.values())]
    return Cm.run_jobs(_sweep_job, grid, workers, "trade sweep")


# ---------------------------------------------------------------------------------------------- HP shadow price
def _hp_job(args):
    over, level, delta, pol, seeds, turns = args
    c = Cm.cfg("r2", **dict(over))
    lane, m4, k = S.zoning_state(c, "neutral", level, "equal")
    from ...champions_m4 import context
    t = context(c).t
    hp_full = m4.a.hp
    hurt = m4.replace_side(0, hp=max(1, hp_full - int(round(delta * t.hp_max[level]))))
    p = ST.get(pol)
    full = [E.play(c, (lane, m4, k), p, p, sd, turns)["summary"] for sd in seeds]
    low = [E.play(c, (lane, hurt, k), p, p, sd, turns)["summary"] for sd in seeds]
    out = {}
    for h in (20, 50, 100):
        out[f"price_G{h}"] = Cm.paired([r[f"G{h}"] for r in full], [r[f"G{h}"] for r in low])
        out[f"price_G{h}"][0] /= delta
        out[f"price_G{h}"][1] /= delta
    out["d_deaths"] = Cm.paired([r["deaths_A"] for r in low], [r["deaths_A"] for r in full])
    out["d_recalls"] = Cm.paired([r["recalls_A"] for r in low], [r["recalls_A"] for r in full])
    return {"level": level, "delta": delta, "policy": pol, **out}


def hp_price(over=(), levels=(2, 6, 11, 18), delta: float = 0.3, policies=("g_farm", "g_bully"), n: int = 200,
             turns: int = 100, workers: int = Cm.WORKERS, seeds=None) -> List[dict]:
    seeds = Cm.DEV[:n] if seeds is None else tuple(seeds)
    jobs = [(tuple(over), L, delta, p, seeds, turns) for L in levels for p in policies]
    return Cm.run_jobs(_hp_job, jobs, workers, "hp price")


# ---------------------------------------------------------------------------------------------- multi-turn
def _mt_job(args):
    over, level, wave, cover, gap, cont_a, cont_b, seeds, turns = args
    c = Cm.cfg("r2", **dict(over))
    start = S.trade_bench(c, level, wave, cover, gap)
    pa, pb = ST.get(cont_a), ST.get(cont_b)
    plans = {"trade": E.Then("trade1", E.Fixed("T", A_TRADE, pa), pa, 1),
             "hold": E.Then("hold1", E.Fixed("H", A_HOLD, pa), pa, 1)}
    hold_b = E.Then("hold1", E.Fixed("H", B_ACTIONS["hold"], pb), pb, 1)
    res = {name: [E.play(c, start, pol, hold_b, sd, turns)["summary"] for sd in seeds] for name, pol in plans.items()}
    out = {"level": level, "wave": wave, "cover": cover, "gap": gap, "cont": [cont_a, cont_b]}
    for key in ("G1", "G5", "G20", "G50", "G100"):
        if key in res["trade"][0]:
            out[key] = Cm.paired([r[key] for r in res["trade"]], [r[key] for r in res["hold"]])
    for key in ("deaths_A", "deaths_B", "recalls_A", "recalls_B", "d_min_A"):
        out[key] = Cm.paired([r[key] for r in res["trade"]], [r[key] for r in res["hold"]])
    return out


def multi_turn(over=(), levels=LEVELS, waves=WAVES, covers=("cover", "nocover"), gaps=GAPS,
               cont=("g_farm", "g_farm"), n: int = 200, turns: int = 100, workers: int = Cm.WORKERS,
               seeds=None) -> List[dict]:
    seeds = Cm.DEV[:n] if seeds is None else tuple(seeds)
    jobs = [(tuple(over), L, w, c, g, cont[0], cont[1], seeds, turns) for L in levels for w in waves for c in covers
            for g in gaps]
    return Cm.run_jobs(_mt_job, jobs, workers, "trade multi-turn")


# ---------------------------------------------------------------------------------------------- natural states
NAT_PAIRINGS = (("g_farm", "g_farm"), ("g_shove", "g_shove"), ("g_guard", "g_shove"), ("g_shove_guard", "g_farm"))


def _tradeable(cfg, lane, m4, s) -> bool:
    """Decision-state predicate: both present, s can catch the enemy without a dive, and the enemy stands within one
    cell of its own front (it farms with its wave)."""
    if m4.side(0).absent or m4.side(1).absent or not ST.catchable(cfg, m4, s):
        return False
    ex, eh = lane.side(1 - s)[:2]
    pos = m4.side(1 - s).pos
    from ...positional import WARD
    return eh > 0 and pos != WARD and abs(ex - pos) <= 1


def _nat_job(args):
    over, pa, pb, seeds_collect, seeds, turns = args
    c = Cm.cfg("r2", **dict(over))
    from ...mechanics import _blob_hp, start_packets
    found = E.collect(c, S.s0(c), ST.get(pa), ST.get(pb), seeds_collect, 150, _tradeable, per_game=3, min_turn=6)
    out = []
    for f in found:
        lane, m4, k = f["start"]
        cont_a, cont_b = (ST.get(pa), ST.get(pb)) if f["seat"] == 0 else (ST.get(pb), ST.get(pa))
        P = start_packets(lane, c)
        blob = _blob_hp(P[1], c)
        from ...champions_m4 import context
        t = context(c).t
        L = t.level_by_xp[min(m4.a.xp, t.xp_cap)]
        res = {}
        for name, code in (("trade", A_TRADE), ("hold", A_HOLD)):
            pol = E.Then(name, E.Fixed(name, code, cont_a), cont_a, 1)
            res[name] = [E.play(c, f["start"], pol, cont_b, sd, turns)["summary"] for sd in seeds]
        d = {"pair": [pa, pb], "blob": blob, "level": L, "seat": f["seat"]}
        for m in ("G1", "G5", "G20", "G50"):
            d[m] = float(np.mean([a[m] - b[m] for a, b in zip(res["trade"], res["hold"])]))
        for m in ("d_total_A@10", "d_total_B@10", "deaths_B@20", "deaths_A@20"):
            d[m] = float(np.mean([a[m] - b[m] for a, b in zip(res["trade"], res["hold"])]))
        out.append(d)
    return out


def natural(over=(), pairings=NAT_PAIRINGS, n_collect: int = 30, seeds=Cm.DEV[300:332], turns: int = 50,
            workers: int = Cm.WORKERS) -> dict:
    """Trade versus Hold on the first turn from NATURAL decision states (`_tradeable`, collected from s0 games of each
    pairing with DEV seeds 87,000-87,029), then the pairing's own strategies; paired over `seeds`. States are the
    units of the intervals; grouped by the enemy blob HP (<= 8, 9-16, >= 17) and A's level (<= 5, 6-12, >= 13)."""
    jobs = [(tuple(over), a, b, tuple(Cm.DEV[i * n_collect // 3:(i + 1) * n_collect // 3]), tuple(seeds), turns)
            for a, b in pairings for i in range(3)]
    rows = [r for part in Cm.run_jobs(_nat_job, jobs, workers, f"trade natural {dict(over)}") for r in part]
    groups = {}
    for r in rows:
        wb = "<=8" if r["blob"] <= 8 else "9-16" if r["blob"] <= 16 else ">=17"
        lb = "L<=5" if r["level"] <= 5 else "L6-12" if r["level"] <= 12 else "L>=13"
        for key in ((wb, lb), (wb, "all"), ("all", lb), ("all", "all")):
            groups.setdefault(key, []).append(r)
    summ = {f"{w}|{l}": {m: Cm.ci([r[m] for r in g]) for m in ("G1", "G5", "G20", "G50", "d_total_A@10",
                                                               "d_total_B@10", "deaths_B@20")}
            for (w, l), g in groups.items()}
    return {"n_states": len(rows), "groups": summ, "rows": rows}


__all__ = ["bench_table", "cell", "criteria", "CRITERIA", "sweep", "SWEEP", "hp_price", "multi_turn", "natural"]
