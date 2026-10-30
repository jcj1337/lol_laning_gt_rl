"""Q3: value capacity on M4.6e's labels (pre-registration, stage 1).

M4.6e's arm C (`m46e.fit.mlp_fit`, main split, fit seeds 0-2, the same groups, weights and early stopping) refitted at
2 x 64, 2 x 256 and 3 x 128 on the stored labels of the six `ext/C1:s` policies. The stored 2 x 128 arm C and arm D
predictions are the baselines (`m46e.fit.predictions`). `repro` refits 2 x 128 at fit seed 0 for one policy and must
reproduce M4.6e's stored predictions exactly.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from . import common as C

WIDTHS = {"2x64": (64, 64), "2x256": (256, 256), "3x128": (128, 128, 128)}


def _path(spec: str, width: str, seed: int) -> str:
    from ..m46e.labels import _tag
    return os.path.join(C.CKPT, "q3", _tag(spec), f"{width}_{seed}.npz")


def fit_job(args) -> tuple:
    spec, width, seed, path = args
    if path and os.path.exists(path):
        return (spec, width, seed), path
    import torch
    torch.set_num_threads(1)
    from ..m46e import common as E
    from ..m46e import fit as F
    E.setup(E.ARM)
    e = F.examples(F.load_points(spec))
    ext = F._m46d_examples(spec)
    r = F.roles(e, "main")
    tr, va = r == "train", r == "val"
    w, wv = F.weights(e, tr), F.weights(e, va)
    hidden = (128, 128) if width == "2x128" else WIDTHS[width]
    m, info = F.mlp_fit(e["x"][tr], e["y"][tr], w[tr], e["x"][va], e["y"][va], wv[va], seed, hidden)
    out = {"pred": F.mlp_predict(m, e["x"]), "ext": F.mlp_predict(m, ext["x"]), "epochs": info["epochs"]}
    if path is None:
        return (spec, width, seed), out
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return (spec, width, seed), path


def repro() -> dict:
    """2 x 128 at fit seed 0 for the first policy against M4.6e's stored C0 predictions."""
    from ..m46e import common as E
    from ..m46e import fit as F
    spec = E.POLICIES[0]
    _, out = fit_job((spec, "2x128", 0, None))
    P = F.predictions(spec, "main")
    rec = {"policy": spec, "pred_identical": bool(np.array_equal(out["pred"], P["ex"]["C0"])),
           "ext_identical": bool(np.array_equal(out["ext"], P["ext"]["C0"])),
           "max_abs_diff": float(np.max(np.abs(out["pred"] - P["ex"]["C0"])))}
    C.save_json("q3_repro", {"meta": C.meta(stage="q3_repro"), **rec})
    return rec


def run(workers: int) -> dict:
    from ..m46e import common as E
    from ..m46e.fit import FIT_SEEDS
    rec = repro()
    print(f"  [Q3] repro {rec}", flush=True)
    jobs = [(spec, w, s, _path(spec, w, s)) for spec in E.POLICIES for w in WIDTHS for s in FIT_SEEDS]
    C.run_jobs(fit_job, jobs, workers, "Q3 fits")
    return analyse()


def _readouts(e, ext, test_nat, pred, pext) -> dict:
    from ..m46e.analysis import basic, m46d_readouts
    m = m46d_readouts(ext, pext)
    return {"edge+plaus": m["classes"]["edge+plaus"]["ratio"], "large+gap": m["classes"]["large+gap"]["ratio"],
            "natural_mae": basic(e["y"][test_nat], pred[test_nat])["mae"]}


def analyse() -> dict:
    from ..m46e import common as E
    from ..m46e import fit as F
    from ..m46e.fit import FIT_SEEDS
    per: Dict[str, Dict[str, dict]] = {}
    for spec in E.POLICIES:
        e = F.examples(F.load_points(spec))
        ext = F._m46d_examples(spec)
        r = F.roles(e, "main")
        test_nat = (r == "test") & (e["src"] == 0)
        P = F.predictions(spec, "main")
        arms = {"2x128 (M4.6e C)": (P["ex"]["C"], P["ext"]["C"]), "2x128+features (M4.6e D)": (P["ex"]["D"], P["ext"]["D"])}
        for w in WIDTHS:
            zs = [np.load(_path(spec, w, s)) for s in FIT_SEEDS]
            arms[w] = (np.mean([z["pred"] for z in zs], 0), np.mean([z["ext"] for z in zs], 0))
        per[spec] = {a: _readouts(e, ext, test_nat, p, px) for a, (p, px) in arms.items()}
    arms = list(next(iter(per.values())))
    keys = ("edge+plaus", "large+gap", "natural_mae")
    summary = {a: {k: C.mean_ci([per[s][a][k] for s in per]) for k in keys} for a in arms}
    base = "2x128 (M4.6e C)"
    diffs = {a: {k: C.mean_ci([per[s][a][k] - per[s][base][k] for s in per]) for k in keys} for a in arms if a != base}
    b256 = diffs["2x256"]
    mae_ratio = float(np.mean([per[s]["2x256"]["natural_mae"] / per[s][base]["natural_mae"] for s in per]))
    binds = (all(b256[k][0] >= 0.10 and b256[k][0] - b256[k][1] > 0 for k in ("edge+plaus", "large+gap"))
             and mae_ratio <= 1.05)
    out = {"meta": C.meta(stage="q3"), "per_policy": per, "summary": summary, "minus_2x128": diffs,
           "natural_mae_ratio_2x256": mae_ratio, "width_binds_for_value": bool(binds)}
    C.save_json("q3", out)
    return out
