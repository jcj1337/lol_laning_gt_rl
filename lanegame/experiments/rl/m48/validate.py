"""Stage 1 readouts of M4.8 (protocol §5 V1-V8): the selected validation cases, mirrors, controlled variants, the
calibration of the predictions against recorded continuations, window stability and leader reversals.

Folding. A case's focal seat X may be A or B; every case statistic is in X's orientation (sign +1 for A, -1 for B):
the folded change is sign * Delta_h, and the folded components are X's minus Y's.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from . import analysis as AN
from . import common as C
from . import families as F
from . import rollouts as R
from . import states as ST

CASE_ORDER = ("returner", "lost", "denied", "stable", "opposing", "due")
H_SHOW = (5, 10, 20, 30, 40)


def material(d: float, se: float, m: float = C.M_STATE) -> bool:
    return bool(abs(d) >= m and abs(d) > 1.96 * se)


def reversal(D0: float, E: float, se: float) -> bool:
    """A material leader reversal: E_h's interval excludes 0 on the side opposite to the raw lead."""
    return bool(D0 != 0 and np.sign(E) != np.sign(D0) and abs(E) > 1.96 * se)


def mean_ci(v) -> Dict[str, float]:
    return C.mean_se(v)


# ---------------------------------------------------------------------------------------------- rows
def rows(selected: List[dict]) -> List[dict]:
    js = F.validation_jobs(selected)
    games: Dict[tuple, dict] = {}
    out = []
    for st, j in zip(selected, js):
        x = R.load(j["path"])
        d = st["desc"]
        row = {"case": st["case"], "X": st["X"], "env": st["env"], "s": st["s"], "g": st["g"], "t0": st["t0"],
               "sign": AN.sign_of(st["X"]), "desc": d}
        row.update(AN.combo_row(d, x["own"]))
        if "mirror" in x:
            row["mirror_max_abs"] = float(np.abs(np.asarray(x["own"]["r"], float)
                                                 + np.asarray(x["mirror"]["r"], float)).max())
            row["mirror_acct_max"] = float(np.abs(np.asarray(x["mirror"]["acct"])).max())
        for v in ("A_fresh", "A_away1"):
            if v in x:
                row.update(AN.paired_row(x[v], x["own"], f"{v}:"))
                row[f"{v}:acct_max"] = float(np.abs(np.asarray(x[v]["acct"])).max())
        key = (st["env"], st["s"])
        if key not in games:
            games[key] = ST.load_validation(*key)["games"]
        led = games[key][st["g"]]
        t0 = st["t0"]
        rec = {}
        for h in C.WINDOWS:
            w = slice(t0, t0 + h)
            rec[f"rd{h}"] = float(np.sum(led["r"][w]))
            for k in ("farm", "avail", "bounty", "plate", "died", "recalled", "returned", "away"):
                v = np.asarray(led[k][w], dtype=float).sum(0)
                rec[f"r{h}_{k}_A"], rec[f"r{h}_{k}_B"] = float(v[0]), float(v[1])
        row.update(rec)
        row["full"] = {k: np.asarray(x["own"][f"f__{k}"]) for k in ("earned", "farm", "away", "returned", "died",
                                                                    "recalled", "avail")} | {
            "r": np.asarray(x["own"]["r"], float)}
        out.append(row)
    return out


def fold(row: dict, key: str) -> float:
    return row["sign"] * row[key]


def xy(row: dict, base: str) -> tuple:
    """(X's value, Y's value) of a per-seat key base (base + "_A" / "_B")."""
    a, b = row[base + "_A"], row[base + "_B"]
    return (a, b) if row["X"] == "A" else (b, a)


def pick(row: dict, fmt: str) -> tuple:
    """(X's, Y's) value of a key with the seat letter in place of {} (e.g. "ret{}_med")."""
    a, b = row[fmt.format("A")], row[fmt.format("B")]
    return (a, b) if row["X"] == "A" else (b, a)


def lead_x(row: dict) -> float:
    return row["sign"] * row["desc"]["D0"]


# ---------------------------------------------------------------------------------------------- per case
def case_summary(rs: List[dict]) -> dict:
    out = {"n": len(rs), "n_by_env": {e: sum(r["env"] == e for r in rs) for e in C.ENVS}}
    d = rs[0]["desc"] if rs else None
    if not rs:
        return out
    out["lead_x"] = mean_ci([lead_x(r) for r in rs])
    out["xp_lead_x"] = mean_ci([r["sign"] * r["desc"]["xp_diff"] for r in rs])
    out["hp_lead_x"] = mean_ci([r["sign"] * r["desc"]["hp_diff"] for r in rs if np.isfinite(r["desc"]["hp_diff"])])
    for h in H_SHOW:
        dv = [fold(r, f"d{h}") for r in rs]
        out[f"d{h}"] = mean_ci(dv)
        out[f"E{h}"] = mean_ci([lead_x(r) + fold(r, f"d{h}") for r in rs])
        out[f"d{h}_by_env"] = {e: mean_ci([fold(r, f"d{h}") for r in rs if r["env"] == e]) for e in C.ENVS}
        out[f"material{h}"] = float(np.mean([material(r[f"d{h}"], r[f"se{h}"]) for r in rs]))
        out[f"pos{h}"] = float(np.mean([fold(r, f"d{h}") > 1.96 * r[f"se{h}"] for r in rs]))
        out[f"neg{h}"] = float(np.mean([fold(r, f"d{h}") < -1.96 * r[f"se{h}"] for r in rs]))
        out[f"reversal{h}"] = float(np.mean([reversal(r["desc"]["D0"], r[f"E{h}"], r[f"se{h}"]) for r in rs]))
        out[f"halfwidth{h}"] = float(np.median([1.96 * r[f"se{h}"] for r in rs]))
        out[f"rec_d{h}"] = mean_ci([r["sign"] * r[f"rd{h}"] for r in rs])
        out[f"res<={h}"] = float(np.mean([r[f"res<={h}"] for r in rs]))
        for kind in ("resA", "resD", "resS"):
            v = [r[f"{kind}<={h}"] for r in rs if f"{kind}<={h}" in r]
            out[f"{kind}<={h}"] = float(np.mean(v)) if v else float("nan")
        out[f"retX<={h}"] = float(np.mean([pick(r, f"ret{{}}<={h}")[0] for r in rs]))
        out[f"retY<={h}"] = float(np.mean([pick(r, f"ret{{}}<={h}")[1] for r in rs]))
        out[f"xpd{h}"] = mean_ci([r["sign"] * r[f"xpd{h}"] for r in rs])
        out[f"hpd{h}"] = mean_ci([r["sign"] * r[f"hpd{h}"] for r in rs if np.isfinite(r[f"hpd{h}"])])
    for h in (5, 10, 20):
        comp = {}
        for key in ("farm", "avail", "lost_away", "bounty", "plate", "died", "recalled", "returned", "away", "xpg"):
            xs, ys = zip(*[xy(r, f"c{h}_{key}") for r in rs])
            comp[key] = {"X": float(np.mean(xs)), "Y": float(np.mean(ys)), "X-Y": float(np.mean(xs) - np.mean(ys))}
        rx, ry = zip(*[xy(r, f"r{h}_farm") for r in rs])
        comp["recorded_farm"] = {"X": float(np.mean(rx)), "Y": float(np.mean(ry))}
        out[f"components{h}"] = comp
    for k in ("stock", "near", "unspent", "hp", "since_return", "returns_in"):
        vx, vy = zip(*[xy(r["desc"] | {"X": r["X"]}, k) for r in rs])
        out[f"{k}_x"], out[f"{k}_y"] = mean_ci(vx), mean_ci(vy)
    gx, gy = zip(*[pick(r, "got{}") for r in rs])
    out["got_x"], out["got_y"] = mean_ci(gx), mean_ci(gy)
    rx, ry = zip(*[pick(r, "ret{}_med") for r in rs])
    out["retX_med"], out["retY_med"] = float(np.median(rx)), float(np.median(ry))
    sx, sy = zip(*[pick(r, "stk{}_med") for r in rs])
    out["stkX_med"], out["stkY_med"] = float(np.median(sx)), float(np.median(sy))
    out["window_change"] = {
        "10->20": {"mean_abs": float(np.mean([abs(r["d20"] - r["d10"]) for r in rs])),
                   "class_changes": float(np.mean([material(r["d10"], r["se10"]) != material(r["d20"], r["se20"])
                                                   for r in rs]))},
        "20->40": {"mean_abs": float(np.mean([abs(r["d40"] - r["d20"]) for r in rs])),
                   "class_changes": float(np.mean([material(r["d20"], r["se20"]) != material(r["d40"], r["se40"])
                                                   for r in rs]))}}
    return out


def variant_summary(rs: List[dict]) -> dict:
    out = {"n": len(rs)}
    for v in ("A_fresh", "A_away1"):
        o = {}
        for h in H_SHOW:
            o[f"pd{h}"] = mean_ci([r[f"{v}:pd{h}"] for r in rs])
            o[f"pd{h}_by_env"] = {e: mean_ci([r[f"{v}:pd{h}"] for r in rs if r["env"] == e]) for e in C.ENVS}
            o[f"state_se{h}"] = float(np.median([r[f"{v}:pse{h}"] for r in rs]))
            o[f"material{h}"] = float(np.mean([material(r[f"{v}:pd{h}"], r[f"{v}:pse{h}"]) for r in rs]))
        for h in (5, 10, 20):
            o[f"components{h}"] = {f"{k}_{sx}": float(np.mean([r[f"{v}:pc{h}_{k}_{sx}"] for r in rs]))
                                   for k in ("farm", "lost_away", "bounty", "plate", "away", "avail")
                                   for sx in ("A", "B")}
        o["acct_max"] = float(max(r[f"{v}:acct_max"] for r in rs))
        out[v] = o
    out["D0"] = mean_ci([r["desc"]["D0"] for r in rs])
    out["unspent_A"] = mean_ci([r["desc"]["unspent_A"] for r in rs])
    out["hp_A"] = mean_ci([r["desc"]["hp_A"] for r in rs])
    out["since_return_A"] = mean_ci([r["desc"]["since_return_A"] for r in rs])
    return out


# ---------------------------------------------------------------------------------------------- reversals (V8)
def dominant(row: dict, h: int, keyfmt: str = "c{h}_{k}") -> str:
    """The accounting component that moves the lead most over h turns (A - B): farm collected, bounty or plate gold;
    for farm, whether the farm gap is mostly farm lost while away or farm available."""
    comp = {k: row[keyfmt.format(h=h, k=k) + "_A"] - row[keyfmt.format(h=h, k=k) + "_B"]
            for k in ("farm", "bounty", "plate")}
    k = max(comp, key=lambda z: abs(comp[z]))
    if k == "farm":
        la = row[keyfmt.format(h=h, k="lost_away") + "_A"] - row[keyfmt.format(h=h, k="lost_away") + "_B"]
        av = row[keyfmt.format(h=h, k="avail") + "_A"] - row[keyfmt.format(h=h, k="avail") + "_B"]
        return "farm (absence)" if abs(la) >= abs(av) else "farm (available)"
    return k


def reversals(rs: List[dict], h: int, label_of) -> dict:
    out = {"n_states": len(rs), "n": 0, "dominant": {}, "by_label": {}, "examples": []}
    for r in rs:
        if reversal(r["desc"]["D0"], r[f"E{h}"], r[f"se{h}"]):
            out["n"] += 1
            dm = dominant(r, h)
            out["dominant"][dm] = out["dominant"].get(dm, 0) + 1
            lab = label_of(r)
            out["by_label"][lab] = out["by_label"].get(lab, 0) + 1
            if len(out["examples"]) < 6:
                out["examples"].append({"label": lab, "D0": r["desc"]["D0"], f"E{h}": r[f"E{h}"],
                                        "se": r[f"se{h}"], "dominant": dm,
                                        "farm_diff": r[f"c{h}_farm_A"] - r[f"c{h}_farm_B"],
                                        "lost_away_diff": r[f"c{h}_lost_away_A"] - r[f"c{h}_lost_away_B"],
                                        "bounty_diff": r[f"c{h}_bounty_A"] - r[f"c{h}_bounty_B"],
                                        "plate_diff": r[f"c{h}_plate_A"] - r[f"c{h}_plate_B"]})
    out["share"] = out["n"] / max(1, len(rs))
    return out


# ---------------------------------------------------------------------------------------------- the worked example
def worked_example(rs: List[dict]) -> dict:
    """The first returner state (in the fixed order: environment, seed, game, turn) whose returner X trails by less than
    2 resources: chosen by its checkpoint only."""
    cand = sorted([r for r in rs if r["case"] == "returner" and -2.0 <= lead_x(r) < 0.0],
                  key=lambda r: (C.ENVS.index(r["env"]), r["s"], r["g"], r["t0"], r["X"]))
    if not cand:
        return {}
    r = cand[0]
    full = r["full"]
    sg = r["sign"]
    cum = np.cumsum(full["r"], axis=1)
    ex = {k: r[k] for k in ("env", "s", "g", "t0", "X", "case")}
    ex["desc"] = r["desc"]
    ex["lead_x"] = lead_x(r)
    ex["readout"] = {str(h): {"E": sg * r[f"E{h}"], "d": sg * r[f"d{h}"], "ci": 1.96 * r[f"se{h}"],
                              "material": material(r[f"d{h}"], r[f"se{h}"]), "res": r[f"res<={h}"],
                              "rec_d": sg * r[f"rd{h}"], "xpd": sg * r[f"xpd{h}"]}
                     for h in C.WINDOWS}
    ex["mean_path"] = (sg * (r["desc"]["D0"] + cum.mean(0))).tolist()
    ex["components"] = {str(h): {k: dict(zip(("X", "Y"), xy(r, f"c{h}_{k}")))
                                 for k in ("farm", "avail", "lost_away", "bounty", "plate", "died", "recalled",
                                           "returned", "away", "xpg")} for h in (5, 10, 20)}
    ex["return_times"] = {"X_med": r["retA_med"] if r["X"] == "A" else r["retB_med"],
                          "Y_med": r["retB_med"] if r["X"] == "A" else r["retA_med"],
                          "Y_by": {str(h): (r[f"retB<={h}"] if r["X"] == "A" else r[f"retA<={h}"]) for h in C.WINDOWS}}
    ex["stock_collected"] = {"X": r["gotA"] if r["X"] == "A" else r["gotB"],
                             "Y": r["gotB"] if r["X"] == "A" else r["gotA"]}
    ex["stock_time_med"] = {"X": r["stkA_med"] if r["X"] == "A" else r["stkB_med"],
                            "Y": r["stkB_med"] if r["X"] == "A" else r["stkA_med"]}
    led = ST.load_validation(r["env"], r["s"])["games"][r["g"]]
    t0, xi = r["t0"], (0 if r["X"] == "A" else 1)
    tl = []
    for i in range(t0, min(t0 + 20, len(led["r"]))):
        ev = []
        for who, s in (("X", xi), ("Y", 1 - xi)):
            if led["recalled"][i][s]:
                ev.append(f"{who} recalls")
            if led["died"][i][s]:
                ev.append(f"{who} dies")
            if led["returned"][i][s]:
                ev.append(f"{who} back next turn")
            if led["plate"][i][s] > 0:
                ev.append(f"{who} plate")
        tl.append({"turn": i, "lead_change_x": sg * led["r"][i], "farm_X": led["farm"][i][xi],
                   "farm_Y": led["farm"][i][1 - xi], "away_X": bool(led["away"][i][xi]),
                   "away_Y": bool(led["away"][i][1 - xi]), "events": ev})
    ex["recorded"] = tl
    # the share of continuations in which X's lead at h exceeds its checkpoint lead
    for h in (5, 10, 20):
        ex[f"P(lead up)@{h}"] = float(np.mean(sg * cum[:, h - 1] > 0))
    return ex


# ---------------------------------------------------------------------------------------------- calibration (V6)
def calibration(nat: List[dict], t0: int = C.T0_PRIMARY) -> dict:
    """Predicted Delta_h (own package, K continuations) against the change realized in the recorded game after the
    checkpoint (one sample of the same process), natural states at t0."""
    rs = [r for r in nat if r["t0"] == t0]
    out = {"n": len(rs)}
    for h in C.WINDOWS:
        pred = np.asarray([r[f"d{h}"] for r in rs])
        real = np.asarray([r[f"rec_d{h}"] for r in rs])
        fit = AN.ols(real, pred)
        resid = real - pred
        out[str(h)] = {"slope": fit, "mean_resid": C.mean_se(resid), "r2_pred": float(1 - resid.var() / real.var()),
                       "var_real": float(real.var()), "var_pred": float(pred.var()),
                       "mean_se2": float(np.mean([r[f"se{h}"] ** 2 for r in rs])),
                       "by_env": {e: AN.ols(real[[r["env"] == e for r in rs]], pred[[r["env"] == e for r in rs]])
                                  for e in C.ENVS}}
        # the realized lead at t0 + h: D0 alone against E_h
        D0 = np.asarray([r["desc"]["D0"] for r in rs])
        out[str(h)]["mse_D0"] = float(np.mean((D0 + real - D0) ** 2))
        out[str(h)]["mse_E"] = float(np.mean((D0 + real - (D0 + pred)) ** 2))
    # does E_h predict the later lead (turn 150, 200) better than D0?
    later = {}
    for t in (150, 200, 250):
        key = f"lead@{t}"
        sel = [r for r in rs if key in r["rec"]]
        y = np.asarray([r["rec"][key] for r in sel])
        D0 = np.asarray([r["desc"]["D0"] for r in sel])
        o = {"n": len(sel), "corr_D0": float(np.corrcoef(D0, y)[0, 1]), "mse_D0": float(np.mean((y - D0) ** 2))}
        for h in (5, 10, 20, 40):
            E = np.asarray([r[f"E{h}"] for r in sel])
            o[f"corr_E{h}"] = float(np.corrcoef(E, y)[0, 1])
            o[f"mse_E{h}"] = float(np.mean((y - E) ** 2))
            # paired difference of squared errors (state as unit)
            o[f"dmse_E{h}"] = C.mean_se((y - E) ** 2 - (y - D0) ** 2)
        later[str(t)] = o
    out["later"] = later
    return out


# ---------------------------------------------------------------------------------------------- run
def run(nat: List[dict] = None) -> dict:
    selected = C.load_gz(os.path.join(C.CKPT, "states", "selected.json.gz"))
    rs = rows(selected)
    out = {"meta": C.meta(stage="validate"), "n_states": len(rs)}
    out["V1_accounting"] = {"continuations": int(sum(r["K"] for r in rs)),
                            "max_abs_units": float(max(r["acct_max"] for r in rs))}
    mir = [r for r in rs if "mirror_max_abs" in r]
    out["V2_mirror"] = {"states": len(mir), "continuations": int(sum(r["K"] for r in mir)),
                        "max_abs_r_sum": float(max(r["mirror_max_abs"] for r in mir)) if mir else float("nan"),
                        "exact": bool(mir and max(r["mirror_max_abs"] for r in mir) == 0.0)}
    cases = {}
    for case in CASE_ORDER:
        for x in ("A", "B", "both"):
            sub = [r for r in rs if r["case"] == case and (x == "both" or r["X"] == x)]
            if sub and not (case in ("stable", "due") and x != "A"):
                cases[f"{case}|{x}"] = case_summary(sub)
    out["cases"] = cases
    out["variants"] = variant_summary([r for r in rs if r["case"] == "due"])
    out["reversals"] = {str(h): reversals(rs, h, lambda r: r["case"]) for h in (10, 20)}
    out["worked_example"] = worked_example(rs)
    if nat is not None:
        out["V6_calibration"] = calibration(nat)
        out["V6_calibration_t84"] = calibration(nat, 84)
        out["V6_calibration_t104"] = calibration(nat, 104)
    C.save_json("validation", out)
    print(f"  validation: {len(rs)} states; accounting max {out['V1_accounting']['max_abs_units']}; mirrors exact "
          f"{out['V2_mirror']['exact']}", flush=True)
    return out


__all__ = ["rows", "case_summary", "variant_summary", "reversals", "worked_example", "calibration", "run",
           "material", "reversal"]
