"""Descriptive stage of M4.6d: freeze episodes in natural self-play, classified and disaggregated; the natural
distribution of early strength differences (to ground the constructed cells); and a per-cell breakdown of the
existing M4.6c constructed-freeze records. Accounting only: nothing here is a causal estimate.

Freeze STATE (`m44.records.holds_freeze`, unchanged): both fronts alive and in contact, the enemy front at the
freezer's own -1 or deeper. The state does not say who made it or why. An EPISODE is a maximal run of turns whose
START-OF-TURN state is a freeze state for the same freezer; its length n is the number of such turns.

Episode classes (fixed here, before any counterfactual):
    length      transient (n = 1), short (2-3), maintained (n >= 4: at least two waves arrived while it held)
    origin      pushed-in: the breaker chose Shove or Slow Push on one of the two turns before the episode began (the
                state is a by-product of the breaker's own push); else formed
    contest     uncontested: the breaker was away on >= half of the episode's turns; walk-up: present and at its own
                +1 or deeper on >= half of its present turns; back: otherwise
    intent      preserving: the freezer present on >= half the turns and on Last Hit / Thin on >= 75% of its wave turns
    favourable  the freezer present with HP >= 60%, its duel ratio >= 1.10 over the breaker, and the held wave >= 9 HP
Strength: duel ratio = (1.05^(L_f - L_b) M(S_f) / M(S_b))^2 (damage per exchange in victim bars, A over B); its
investment part (M_f / M_b)^2 and level part 1.05^(2 (L_f - L_b)) are reported separately; the HP part is the HP-fraction
ratio. End reasons: crashed_in (the breaker's wave crashed into the freezer's tower), held_wave_cleared, own_wave_died,
pushed_out (the fight moved toward the lane), censored (record ended).
Economics over the window [t0, t0 + n + 20): freezer minus breaker (earned, farm, available farm, bounty, plates, XP,
deaths, Recalls, turns away, damage), plus the discounted lead G20 / G50 from t0.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from . import common as C
from . import records as R

AFTER = 20
LEVEL_BINS = ((1, 5), (6, 9), (10, 13), (14, 18))
DUEL_BINS = ((0.0, 0.9, "breaker stronger (< 0.9)"), (0.9, 1.1, "even (0.9-1.1)"), (1.1, 1.25, "freezer x1.10-1.25"),
             (1.25, 99.0, "freezer >= x1.25"))
HP_BINS = ((0.0, 0.45), (0.45, 0.7), (0.7, 1.01))
WAVE_NAMES = ("LH", "SH", "TN", "SL")
BEH_NAMES = ("Hold", "Yield", "Trade", "Safe", "Recall")


def lbin(L: float) -> str:
    for lo, hi in LEVEL_BINS:
        if lo <= L <= hi:
            return f"L{lo}-{hi}"
    return "?"


def dbin(x: float) -> str:
    for lo, hi, name in DUEL_BINS:
        if lo <= x < hi:
            return name
    return "?"


class Strength:
    """M(S) and the duel ratio for one configuration (memoized on whole ledger units)."""

    def __init__(self, cfg):
        from ....champions_m4 import context
        self.inv = context(cfg).inv
        self.sc = cfg.gold_scale
        self._m: Dict[int, float] = {}

    def M(self, S: float) -> float:
        u = int(round(S * self.sc))
        v = self._m.get(u)
        if v is None:
            v = self._m[u] = self.inv.factor(u) if self.inv is not None else 1.0
        return v

    def parts(self, Lf, Sf, Lb, Sb):
        inv = (self.M(Sf) / self.M(Sb)) ** 2
        lev = 1.05 ** (2 * (Lf - Lb))
        return inv * lev, inv, lev


def episodes(d: Dict[str, np.ndarray], g: int, st: Strength) -> List[dict]:
    """The freeze episodes of game g of a record set (both freezer seats)."""
    T = d["held0"].shape[1]
    out = []
    for f in (0, 1):
        b = 1 - f
        h = d["held0"][g, :, f] > 0
        t = 0
        while t < T:
            if not h[t]:
                t += 1
                continue
            t0 = t
            while t < T and h[t]:
                t += 1
            t1 = t
            n = t1 - t0
            if t1 >= T:
                end = "censored"
            elif d["crash"][g, t1 - 1, b] > 0:
                end = "crashed_in"
            elif d["fh"][g, t1, b] == 0:
                end = "held_wave_cleared"
            elif d["fh"][g, t1, f] == 0:
                end = "own_wave_died"
            else:
                end = "pushed_out"
            x = lambda k, s, tt=t0: float(d[k][g, tt, s])                       # noqa: E731
            duel, inv, lev = st.parts(x("L", f), x("S", f), x("L", b), x("S", b))
            hpf, hpb = x("hp", f), x("hp", b)
            pre = [d["wave"][g, u, b] for u in (t0 - 1, t0 - 2) if u >= 0]
            sl = slice(t0, t1)
            pres_f = d["absent"][g, sl, f] == 0
            pres_b = d["absent"][g, sl, b] == 0
            wf = d["wave"][g, sl, f][pres_f]
            wf = wf[wf >= 0]
            preserving = bool(pres_f.mean() >= 0.5 and len(wf) and np.isin(wf, (0, 2)).mean() >= 0.75)
            posb = d["pos"][g, sl, b][pres_b]
            walk = posb[(posb >= 1) & (posb <= 2)]
            if pres_b.mean() < 0.5:
                contest = "uncontested"
            elif len(posb) and len(walk) >= 0.5 * len(posb):
                contest = "walk-up"
            else:
                contest = "back"
            w1 = min(T, t0 + n + AFTER)
            ws = slice(t0, w1)
            diff = lambda k: float(d[k][g, ws, f].sum() - d[k][g, ws, b].sum())     # noqa: E731
            lead = d["earned"][g, :, f] - d["earned"][g, :, b]
            disc = C.GAMMA ** np.arange(T)

            def G(hh):
                e = min(T, t0 + hh)
                return float((lead[t0:e] * disc[:e - t0]).sum()) if e - t0 == hh else float("nan")

            bw = d["wave"][g, sl, b][pres_b]
            bb = d["beh"][g, sl, b][pres_b]
            fb = d["beh"][g, sl, f][pres_f]
            out.append({
                "g": g, "f": f, "t0": t0, "n": n, "end": end, "Lf": x("L", f), "Lb": x("L", b),
                "Sf": x("S", f), "Sb": x("S", b), "duel": duel, "duel_inv": inv, "duel_lev": lev, "hpf": hpf,
                "hpb": hpb, "absent_b0": x("absent", b), "absent_f0": x("absent", f), "posf": x("pos", f),
                "posb": x("pos", b), "held_hp": x("fh", b), "own_hp": x("fh", f), "held_cell_f": -x("fx", b),
                "own_cell": x("fx", f), "near_f": x("rear_near", f), "near_b": x("rear_near", b),
                "nrear_f": x("nrear", f), "nrear_b": x("nrear", b), "mass_b": x("mass", b), "mass_f": x("mass", f),
                "phase": int(d["k"][g, t0] % 2), "Cf": x("C", f), "Cb": x("C", b),
                "origin": "pushed-in" if any(w in (1, 3) for w in pre) else "formed",
                "contest": contest, "preserving": preserving,
                "favourable": bool(x("absent", f) == 0 and hpf >= 0.6 and duel >= 1.10 and x("fh", b) >= 9),
                "f_wave": {nm: float((wf == j).mean()) if len(wf) else float("nan") for j, nm in enumerate(WAVE_NAMES)},
                "f_beh": {nm: float((fb == j).mean()) if len(fb) else float("nan") for j, nm in enumerate(BEH_NAMES)},
                "b_wave": {nm: float((bw == j).mean()) if len(bw) else float("nan") for j, nm in enumerate(WAVE_NAMES)},
                "b_beh": {nm: float((bb == j).mean()) if len(bb) else float("nan") for j, nm in enumerate(BEH_NAMES)},
                "b_absent": float(1 - pres_b.mean()), "b_walk": float(len(walk) / max(1, len(posb))),
                "b_pshove": float(d["p_wave"][g, sl, b, 1][pres_b].mean()) if pres_b.any() else float("nan"),
                "b_psafe": float(d["p_beh"][g, sl, b, 3][pres_b].mean()) if pres_b.any() else float("nan"),
                "f_ptrade": float(d["p_beh"][g, sl, f, 2][pres_f].mean()) if pres_f.any() else float("nan"),
                "f_pshove": float(d["p_wave"][g, sl, f, 1][pres_f].mean()) if pres_f.any() else float("nan"),
                "window": w1 - t0,
                "d_earned": diff("earned"), "d_farm": diff("farm"), "d_avail": diff("avail"),
                "d_bounty": diff("bounty"), "d_plate": diff("plate"), "d_xp": diff("xpg"),
                "f_unc": float(d["avail"][g, ws, f].sum() - d["farm"][g, ws, f].sum()),
                "b_unc": float(d["avail"][g, ws, b].sum() - d["farm"][g, ws, b].sum()),
                "b_unc_present": float(((d["avail"][g, ws, b] - d["farm"][g, ws, b]) * (d["away"][g, ws, b] == 0)).sum()),
                "f_deaths": float(d["died"][g, ws, f].sum()), "b_deaths": float(d["died"][g, ws, b].sum()),
                "f_recalls": float(d["recalled"][g, ws, f].sum()), "b_recalls": float(d["recalled"][g, ws, b].sum()),
                "f_away": float(d["away"][g, ws, f].sum()), "b_away": float(d["away"][g, ws, b].sum()),
                "f_taken": float(d["taken"][g, ws, f].sum()), "b_taken": float(d["taken"][g, ws, b].sum()),
                "b_intr": float(d["intr"][g, ws, b].sum()),
                "G20": G(20), "G50": G(50),
                "Vf0": x("V", f), "Vb0": x("V", b),
            })
    return out


def _agg(eps: List[dict], keys=("n", "held_hp", "own_hp", "duel", "hpf", "hpb", "d_earned", "d_farm", "d_avail",
                                "d_xp", "b_unc_present", "b_deaths", "f_deaths", "b_away", "G20", "G50")) -> dict:
    out = {"episodes": len(eps), "held_turns": int(sum(e["n"] for e in eps))}
    if not eps:
        return out
    for k in keys:
        v = np.asarray([e[k] for e in eps], float)
        v = v[np.isfinite(v)]
        out[k] = float(v.mean()) if len(v) else float("nan")
    for k in ("origin", "contest", "end"):
        vals = [e[k] for e in eps]
        out[k] = {u: vals.count(u) / len(vals) for u in sorted(set(vals))}
    out["maintained"] = float(np.mean([e["n"] >= 4 for e in eps]))
    out["preserving"] = float(np.mean([e["preserving"] for e in eps]))
    out["favourable"] = float(np.mean([e["favourable"] for e in eps]))
    for side in ("f_wave", "b_wave", "f_beh", "b_beh"):
        names = WAVE_NAMES if "wave" in side else BEH_NAMES
        out[side] = {nm: float(np.nanmean([e[side][nm] for e in eps])) for nm in names}
    for k in ("b_absent", "b_walk", "b_pshove", "b_psafe", "f_ptrade", "f_pshove"):
        out[k] = float(np.nanmean([e[k] for e in eps]))
    return out


def length_class(n: int) -> str:
    return "transient (1)" if n == 1 else ("short (2-3)" if n <= 3 else "maintained (>= 4)")


def policy_episodes(spec: str, arm: str) -> List[dict]:
    d = R.load(spec, arm)
    st = Strength(C.arm_cfg(arm))
    eps = []
    for g in range(d["held0"].shape[0]):
        eps += episodes(d, g, st)
    return eps


def frozen_on_masses(d: Dict[str, np.ndarray], st: Strength) -> List[dict]:
    """Per-turn rows for the frozen-on side (start-of-turn freeze state held by the enemy, both present)."""
    rows = []
    G, T = d["held0"].shape[:2]
    for g in range(G):
        for b in (0, 1):
            f = 1 - b
            idx = np.where((d["held0"][g, :, f] > 0) & (d["absent"][g, :, b] == 0) & (d["absent"][g, :, f] == 0))[0]
            for t in idx:
                duel, _, _ = st.parts(d["L"][g, t, f], d["S"][g, t, f], d["L"][g, t, b], d["S"][g, t, b])
                rows.append({"L": float(d["L"][g, t, b]), "hp": float(d["hp"][g, t, b]), "duel": duel,
                             "hpf": float(d["hp"][g, t, f]), "pos": float(d["pos"][g, t, b]),
                             "p_shove": float(d["p_wave"][g, t, b, 1]), "p_safe": float(d["p_beh"][g, t, b, 3]),
                             "p_trade": float(d["p_beh"][g, t, b, 2]), "p_recall": float(d["p_beh"][g, t, b, 4]),
                             "p_hold": float(d["p_beh"][g, t, b, 0]), "held_hp": float(d["fh"][g, t, b])})
    return rows


def early_advantages(d: Dict[str, np.ndarray], st: Strength, t_lo: int = 5, t_hi: int = 100) -> Dict[str, dict]:
    """Natural strength differences (both present) by the focal side's level bin, oriented to the stronger side."""
    acc = defaultdict(lambda: defaultdict(list))
    G, T = d["L"].shape[:2]
    for g in range(G):
        for t in range(t_lo, min(T, t_hi)):
            if d["absent"][g, t, 0] or d["absent"][g, t, 1]:
                continue
            duel, inv, lev = st.parts(d["L"][g, t, 0], d["S"][g, t, 0], d["L"][g, t, 1], d["S"][g, t, 1])
            L = min(d["L"][g, t, 0], d["L"][g, t, 1])
            key = lbin(L)
            a = acc[key]
            a["log_duel"].append(abs(np.log(duel)))
            a["log_inv"].append(abs(np.log(inv)))
            a["dL"].append(abs(d["L"][g, t, 0] - d["L"][g, t, 1]))
            a["dS"].append(abs(d["S"][g, t, 0] - d["S"][g, t, 1]))
            hp0, hp1 = d["hp"][g, t, 0], d["hp"][g, t, 1]
            # HP-adjusted: the stronger-by-duel side's HP over the other's
            fight = duel * hp0 / max(hp1, 1e-3) if duel >= 1 else (1 / duel) * hp1 / max(hp0, 1e-3)
            a["fight"].append(fight)
            a["dhp"].append(abs(hp0 - hp1))
            a["S_mean"].append(0.5 * (d["S"][g, t, 0] + d["S"][g, t, 1]))
    out = {}
    for key, a in acc.items():
        row = {"n": len(a["log_duel"])}
        for k, v in a.items():
            v = np.asarray(v, float)
            row[k] = {q: float(np.quantile(v, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)}
        dl = np.exp(np.asarray(a["log_duel"]))
        row["share_duel_ge_1.10"] = float((dl >= 1.10).mean())
        row["share_duel_ge_1.25"] = float((dl >= 1.25).mean())
        row["share_duel_ge_1.5"] = float((dl >= 1.5).mean())
        row["share_fight_ge_1.5"] = float((np.asarray(a["fight"]) >= 1.5).mean())
        row["share_fight_ge_2"] = float((np.asarray(a["fight"]) >= 2.0).mean())
        out[key] = row
    return out


# ---------------------------------------------------------------------------------------------- M4.6c records
def m46c_cells() -> dict:
    """Per-cell breakdown of the M4.6c constructed-freeze family (4.096M arms; 6 seeds; K 32; G50): candidate minus
    the policy, mean over seeds with a t interval over seeds, plus held turns and deaths at 20 turns."""
    from ..m46c.archive import freeze_cells
    root = os.path.join("results", "rl", "checkpoints", "m46c", "eval", "freeze")
    out = {}
    for env in ("baseline", "C1"):
        for role in ("freezer", "breaker"):
            for i, (L, h, inv) in freeze_cells():
                per = defaultdict(list)
                absolute = []
                ledg = defaultdict(list)
                for s in C.SEEDS:
                    p = os.path.join(root, env, f"{role}_{env}_s{s}", f"{i}.json.gz")
                    if not os.path.exists(p):
                        continue
                    with gzip.open(p, "rt") as fh:
                        rec = json.load(fh)
                    base = np.asarray(rec["G"]["policy"]["50"])
                    absolute.append(float(base.mean()))
                    keys = rec["keys"]
                    for cand in rec["G"]:
                        if cand != "policy":
                            per[cand].append(float((np.asarray(rec["G"][cand]["50"]) - base).mean()))
                        e = rec["ledger"][cand].get("20")
                        if e:
                            for kk in ("held_A", "held_B", "deaths_A", "deaths_B", "earned_A", "earned_B"):
                                ledg[(cand, kk)].append(e["mean"][keys.index(kk)])
                row = {"policy_G50": C.mean_ci(absolute)}
                for cand, v in per.items():
                    row[cand] = C.mean_ci(v)
                row["ledger20"] = {f"{c}|{k}": float(np.mean(v)) for (c, k), v in ledg.items()}
                out[f"{env}|{role}|L{L}|held{h}|{inv}"] = row
    return out


def run() -> dict:
    data = {"meta": C.meta(stage="freezes", note="descriptive accounting; not causal"), "policies": {}}
    all_eps = {}
    for arm in ("C1", "baseline"):
        st = Strength(C.arm_cfg(arm))
        frozen_rows, adv = [], []
        eps_arm = []
        for s in C.SEEDS:
            spec = C.general(arm, s)
            d = R.load(spec, arm)
            eps = []
            for g in range(d["held0"].shape[0]):
                eps += episodes(d, g, st)
            for e in eps:
                e["seed"] = s
            eps_arm += eps
            frozen_rows += frozen_on_masses(d, st)
            adv.append(early_advantages(d, st))
            turns = d["held0"].shape[0] * d["held0"].shape[1]
            data["policies"][f"{arm}:{s}"] = {"digest": C.checkpoint_digest(spec), "turns": turns,
                                              "held_share": float(d["held0"].mean()),
                                              "episodes_per_100": 100 * len(eps) / turns,
                                              "maintained_per_100": 100 * sum(e["n"] >= 4 for e in eps) / turns}
        all_eps[arm] = eps_arm
        # pooled early advantages (equal weight per policy)
        keys = sorted(set().union(*[a.keys() for a in adv]))
        data[f"advantages_{arm}"] = {k: {kk: (float(np.mean([a[k][kk] for a in adv if k in a]))
                                              if not isinstance(adv[0].get(k, {}).get(kk), dict) else
                                              {q: float(np.mean([a[k][kk][q] for a in adv if k in a]))
                                               for q in adv[0][k][kk]})
                                         for kk in adv[0].get(k, {})} for k in keys if all(k in a for a in adv)}
        tb = data.setdefault("tables", {}).setdefault(arm, {})
        tb["all"] = _agg(eps_arm)
        tb["by_length"] = {c: _agg([e for e in eps_arm if length_class(e["n"]) == c])
                           for c in ("transient (1)", "short (2-3)", "maintained (>= 4)")}
        tb["held_turn_share_by_length"] = {c: sum(e["n"] for e in eps_arm if length_class(e["n"]) == c) /
                                           max(1, sum(e["n"] for e in eps_arm))
                                           for c in ("transient (1)", "short (2-3)", "maintained (>= 4)")}
        tb["by_level"] = {lb: _agg([e for e in eps_arm if lbin(e["Lf"]) == lb]) for lb in
                          [f"L{a}-{b}" for a, b in LEVEL_BINS]}
        tb["maintained_by_level"] = {lb: _agg([e for e in eps_arm if lbin(e["Lf"]) == lb and e["n"] >= 4]) for lb in
                                     [f"L{a}-{b}" for a, b in LEVEL_BINS]}
        tb["by_duel"] = {nm: _agg([e for e in eps_arm if dbin(e["duel"]) == nm]) for _, _, nm in DUEL_BINS}
        tb["by_origin"] = {o: _agg([e for e in eps_arm if e["origin"] == o]) for o in ("pushed-in", "formed")}
        tb["by_contest"] = {c: _agg([e for e in eps_arm if e["contest"] == c])
                            for c in ("uncontested", "walk-up", "back")}
        tb["by_held_hp"] = {nm: _agg([e for e in eps_arm if lo <= e["held_hp"] <= hi])
                            for nm, lo, hi in (("held 1-8", 1, 8), ("held 9-15", 9, 15), ("held 16-24", 16, 24))}
        tb["by_support"] = {nm: _agg([e for e in eps_arm if fn(e)]) for nm, fn in
                            (("breaker rear near", lambda e: e["near_b"] > 0),
                             ("no breaker rear near", lambda e: e["near_b"] == 0),
                             ("breaker mass under tower", lambda e: e["mass_b"] > 0))}
        tb["by_preserving"] = {str(v): _agg([e for e in eps_arm if e["preserving"] == v]) for v in (True, False)}
        tb["favourable"] = _agg([e for e in eps_arm if e["favourable"]])
        tb["favourable_maintained"] = _agg([e for e in eps_arm if e["favourable"] and e["n"] >= 4])
        tb["favourable_early"] = _agg([e for e in eps_arm if e["favourable"] and e["Lf"] <= 9])
        tb["by_hp_b"] = {f"breaker HP {lo:.2f}-{hi:.2f}": _agg([e for e in eps_arm if lo <= e["hpb"] < hi and
                                                                 e["absent_b0"] == 0]) for lo, hi in HP_BINS}
        tb["by_seed"] = {str(s): _agg([e for e in eps_arm if e["seed"] == s]) for s in C.SEEDS}
        # strength components separately (freezer side positive)
        comp = {}
        for nm, fn in (("investment part >= x1.10", lambda e: e["duel_inv"] >= 1.10),
                       ("investment part <= x0.91", lambda e: e["duel_inv"] <= 1 / 1.10),
                       ("level ahead", lambda e: e["Lf"] > e["Lb"]), ("level behind", lambda e: e["Lf"] < e["Lb"]),
                       ("HP ahead >= 0.25", lambda e: e["hpf"] - e["hpb"] >= 0.25 and e["absent_b0"] == 0),
                       ("HP behind >= 0.25", lambda e: e["hpb"] - e["hpf"] >= 0.25 and e["absent_f0"] == 0)):
            comp[nm] = _agg([e for e in eps_arm if fn(e)])
        tb["by_component"] = comp
        # frozen-on side's policy masses by HP and strength
        fr = {}
        for hlo, hhi in HP_BINS:
            for _, _, nm in DUEL_BINS:
                rows = [r for r in frozen_rows if hlo <= r["hp"] < hhi and dbin(r["duel"]) == nm]
                if len(rows) >= 20:
                    fr[f"HP {hlo:.2f}-{hhi:.2f} | {nm}"] = {"n": len(rows), **{k: float(np.mean([r[k] for r in rows]))
                                                                             for k in ("p_shove", "p_safe", "p_trade",
                                                                                       "p_recall", "p_hold", "pos")}}
        tb["frozen_on_masses"] = fr
    data["m46c_cells"] = m46c_cells()
    C.save_json("freezes", data)
    C.save_gz(os.path.join(C.CKPT, "freeze_episodes.json.gz"), all_eps)
    return data


__all__ = ["episodes", "policy_episodes", "Strength", "early_advantages", "m46c_cells", "run", "lbin", "dbin",
           "length_class"]
