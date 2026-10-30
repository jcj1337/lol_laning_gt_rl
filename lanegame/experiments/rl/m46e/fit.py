"""Q4: the supervised value study (docs/m4/m4_6/m4_6e_preregistration.md). No PPO, no actor training.

Per policy ext/C1:s (labels are never pooled across policies) and split, the arms
    A    the existing critic, unchanged (V / 0.1)
    B    ridge readout on the frozen trunk's last hidden layer (128 tanh units), ridge strength chosen on validation
    L    reference: ridge on the observation plus the six observation-derived strength features
    C    a freshly initialized value MLP of the critic's shape (90 -> 128 -> 128 -> 1, tanh, orthogonal init), Adam,
         early stopping on validation groups; fit seeds 0, 1, 2
    D    C with the six observation-derived strength features appended (features.explicit: deterministic functions
         of the observation, so D adds no information)
Targets: G_300 (resources) from the example's own seat; the MLPs train on the 0.1-scaled target (as PPO does).
Examples: both seats of every labelled state; natural and constructed examples carry equal total weight.
Splits (groups: a natural game; a constructed archive state with all its variants, rollouts and trajectory points):
    main      25% of the groups held out (constructed: stratified by archive kind), 15% of the rest for validation
    strength  main, without the training examples whose duel ratio is beyond x1.6 in either direction
    family    main, without any constructed freeze group in training (validation unchanged)
    reach     (amendment 1, post hoc) main, without the "big" variants whose constructed investment exceeds the
              archive's late-game cap of 240 resources (unreachable states: B already near saturation)
Outputs per (policy, split, model): predictions on every labelled example and on the M4.6d states (both seats); the
main-split models are saved for the ranking stage.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import common as C
from . import labels as LB
from .features import explicit, obs_index

SPLITS = ("main", "strength", "family", "reach")
REACH_CAP = 240.0                          # resources: the M4.6c archive's own late-game investment cap
FIT_SEEDS = (0, 1, 2)
MODELS = ("ridge",) + tuple(f"C{s}" for s in FIT_SEEDS) + tuple(f"D{s}" for s in FIT_SEEDS)
TEST_FRAC, VAL_FRAC = 0.25, 0.15
STRENGTH_CUT = float(np.log(1.6))
ALPHAS = (1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 3e3, 1e4, 3e4, 1e5)
EPOCHS, PATIENCE, LR, BATCH = 150, 15, 1e-3, 512
NAT_OFFSET = 100_000                      # natural groups: NAT_OFFSET + game


# ---------------------------------------------------------------------------------------------- data
def load_points(spec: str) -> Dict[str, np.ndarray]:
    """All labelled states of one policy (one row per state; seats are expanded in `examples`)."""
    d = os.path.join(C.CKPT, "labels", LB._tag(spec))
    parts = {"natural": [], "constructed": []}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".npz"):
            parts["natural" if fn.startswith("natural") else "constructed"].append(dict(np.load(os.path.join(d, fn))))
    out = {}
    for src, ps in parts.items():
        z = {k: np.concatenate([p[k] for p in ps]) for k in ps[0]}
        n = len(z["t"])
        if src == "natural":
            z["group"] = NAT_OFFSET + z["game"]
            for k in ("variant", "kind", "rollout"):
                z[k] = np.full(n, -1, np.int32)
            del z["game"]
        z["src"] = np.full(n, 0 if src == "natural" else 1, np.int32)
        out[src] = z
    keys = set(out["natural"]) & set(out["constructed"])
    return {k: np.concatenate([out["natural"][k], out["constructed"][k]]) for k in keys}


def examples(p: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Both seats of every state: x = the seat's observation, y = its G_300 (seat B: -G_300)."""
    n = len(p["t"])
    ld = np.log(p["f_duel"])
    e = {"x": np.concatenate([p["obs"][:, 0], p["obs"][:, 1]]).astype(np.float32),
         "y": np.concatenate([p["G300"], -p["G300"]]), "V": np.concatenate([p["V"][:, 0], p["V"][:, 1]]),
         "seat": np.r_[np.zeros(n, np.int32), np.ones(n, np.int32)], "point": np.r_[np.arange(n), np.arange(n)],
         "log_duel": np.r_[ld, -ld], "hp_own": np.r_[p["f_hp"], p["f_hp_o"]], "hp_opp": np.r_[p["f_hp_o"], p["f_hp"]],
         "S_A0": np.r_[p["f_S"], p["f_S"]]}
    for k in ("t", "group", "variant", "kind", "rollout", "src"):
        e[k] = np.r_[p[k], p[k]]
    return e


def group_split(groups: np.ndarray, kinds: Dict[int, int]) -> Dict[int, str]:
    """group -> 'train' / 'val' / 'test' (the same for every policy: seeded, stratified by source kind)."""
    rng = np.random.default_rng(C.FIT_ROOT)
    out = {}
    strata: Dict[int, List[int]] = {}
    for g in sorted(set(int(x) for x in groups)):
        strata.setdefault(kinds[g], []).append(g)
    for k in sorted(strata):
        gs = list(strata[k])
        rng.shuffle(gs)
        nt = int(round(TEST_FRAC * len(gs)))
        rest = gs[nt:]
        nv = max(1, int(round(VAL_FRAC * len(rest))))
        for g in gs[:nt]:
            out[g] = "test"
        for g in rest[:nv]:
            out[g] = "val"
        for g in rest[nv:]:
            out[g] = "train"
    return out


def roles(e: Dict[str, np.ndarray], split: str) -> np.ndarray:
    kinds = {int(g): (-1 if s == 0 else int(k)) for g, s, k in zip(e["group"], e["src"], e["kind"])}
    gs = group_split(e["group"], kinds)
    r = np.asarray([gs[int(g)] for g in e["group"]])
    if split == "strength":
        r = np.where((r == "train") & (np.abs(e["log_duel"]) >= STRENGTH_CUT), "drop", r)
    elif split == "reach":                       # amendment 1 (post hoc): no unreachable "big" variant in training
        big = (e["src"] == 1) & (e["variant"] == LB.VARIANTS.index("big"))
        bad_groups = set(int(g) for g in e["group"][big & (e["t"] == 0) & (e["S_A0"] > REACH_CAP)])
        r = np.where((r == "train") & big & np.isin(e["group"], sorted(bad_groups)), "drop", r)
    elif split == "family":
        fz = tuple(C.ARCHIVE_SAMPLE).index("freeze")
        r = np.where((r == "train") & (e["src"] == 1) & (e["kind"] == fz), "drop", r)
    return r


def weights(e, mask) -> np.ndarray:
    """Equal total weight for natural and constructed examples among `mask`, mean 1."""
    w = np.zeros(len(e["y"]))
    for s in (0, 1):
        m = mask & (e["src"] == s)
        if m.any():
            w[m] = 0.5 / m.sum()
    return w / w[mask].mean()


# ---------------------------------------------------------------------------------------------- models
def trunk(net, X: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        return net.trunk(torch.as_tensor(X, dtype=torch.float32)).double().numpy()


def ridge(F, y, w, Fv, yv, wv):
    """Weighted ridge with an unpenalized intercept; the strength by validation MSE. Returns (beta, alpha)."""
    mu, sd = F.mean(0), F.std(0) + 1e-8
    Z = np.c_[np.ones(len(F)), (F - mu) / sd]
    Zv = np.c_[np.ones(len(Fv)), (Fv - mu) / sd]
    A = Z.T @ (Z * w[:, None])
    b = Z.T @ (w * y)
    best = None
    for a in ALPHAS:
        I = np.eye(A.shape[0]) * a
        I[0, 0] = 0.0
        beta = np.linalg.solve(A + I, b)
        mse = float(np.average((Zv @ beta - yv) ** 2, weights=wv))
        if best is None or mse < best[0]:
            best = (mse, beta, a)
    return {"beta": best[1], "mu": mu, "sd": sd, "alpha": best[2]}


def ridge_predict(m, F) -> np.ndarray:
    return np.c_[np.ones(len(F)), (F - m["mu"]) / m["sd"]] @ m["beta"]


def make_mlp(n_in: int, hidden=(128, 128)):
    """The critic's shape by default; NC1's Q3 passes other widths (opt-in, the default builds the same network)."""
    from ....rl.network import _layer
    from torch import nn
    layers, n = [], n_in
    for h in hidden:
        layers += [_layer(n, h, np.sqrt(2.0)), nn.Tanh()]
        n = h
    return nn.Sequential(*layers, _layer(n, 1, 1.0))


def mlp_fit(X, y, w, Xv, yv, wv, seed: int, hidden=(128, 128)):
    import torch
    torch.manual_seed(C.FIT_ROOT + seed)
    gen = torch.Generator().manual_seed(C.FIT_ROOT + seed)
    m = make_mlp(X.shape[1], hidden)
    opt = torch.optim.Adam(m.parameters(), lr=LR, eps=1e-5)
    Xt, yt, wt = (torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y * C.REWARD_SCALE, dtype=torch.float32),
                  torch.as_tensor(w, dtype=torch.float32))
    Xvt, yvt, wvt = (torch.as_tensor(Xv, dtype=torch.float32), torch.as_tensor(yv * C.REWARD_SCALE, dtype=torch.float32),
                     torch.as_tensor(wv, dtype=torch.float32))
    best, best_state, bad, ep = float("inf"), None, 0, 0
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xt), generator=gen)
        m.train()
        for b in range(0, len(Xt), BATCH):
            i = perm[b:b + BATCH]
            loss = (wt[i] * (m(Xt[i]).squeeze(-1) - yt[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = float((wvt * (m(Xvt).squeeze(-1) - yvt) ** 2).sum() / wvt.sum())
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, {"epochs": ep + 1, "val_mse_scaled": best}


def mlp_predict(m, X) -> np.ndarray:
    import torch
    with torch.no_grad():
        return m(torch.as_tensor(X, dtype=torch.float32)).squeeze(-1).double().numpy() / C.REWARD_SCALE


# ---------------------------------------------------------------------------------------------- jobs
def _m46d_examples(spec: str) -> Dict[str, np.ndarray]:
    path = os.path.join(C.CKPT, "m46d_sets", f"{LB._tag(spec)}.npz")
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=False))
    s = LB.m46d_set(spec)
    out = {"x": np.concatenate([s["obs"][:, 0], s["obs"][:, 1]]).astype(np.float32),
           "y": np.r_[s["G300"], -s["G300"]], "y500": np.r_[s["G500"], -s["G500"]],
           "se": np.r_[s["G300_se"], s["G300_se"]], "V": np.r_[s["V"][:, 0], s["V"][:, 1]],
           "seat": np.r_[np.zeros(len(s["G300"]), np.int32), np.ones(len(s["G300"]), np.int32)],
           "label": np.r_[s["label"], s["label"]].astype(str), "fam": np.r_[s["fam"], s["fam"]].astype(str)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def fit_job(job: dict) -> tuple:
    path = job["path"]
    if os.path.exists(path):
        return job["key"], path
    import torch
    torch.set_num_threads(1)
    cfg = C.setup(C.ARM)
    spec, split, model = job["spec"], job["split"], job["model"]
    e = examples(load_points(spec))
    ext = _m46d_examples(spec)
    r = roles(e, split)
    tr, va = r == "train", r == "val"
    w, wv = weights(e, tr), weights(e, va)
    idx = obs_index(cfg)
    t0 = time.perf_counter()
    info: Dict[str, object] = {}
    if model == "ridge":
        net = C.net_of(spec)
        Fe, Fx = trunk(net, e["x"]), trunk(net, ext["x"])
        mb = ridge(Fe[tr], e["y"][tr], w[tr], Fe[va], e["y"][va], wv[va])
        Ge = np.c_[e["x"], explicit(e["x"], idx)]
        Gx = np.c_[ext["x"], explicit(ext["x"], idx)]
        ml = ridge(Ge[tr], e["y"][tr], w[tr], Ge[va], e["y"][va], wv[va])
        pred = {"B": ridge_predict(mb, Fe), "L": ridge_predict(ml, Ge)}
        pext = {"B": ridge_predict(mb, Fx), "L": ridge_predict(ml, Gx)}
        info = {"alpha_B": mb["alpha"], "alpha_L": ml["alpha"]}
        saved = {"B": mb, "L": ml}
    else:
        arm, seed = model[0], int(model[1:])
        Xe, Xx = e["x"], ext["x"]
        if arm == "D":
            Xe, Xx = np.c_[Xe, explicit(Xe, idx)], np.c_[Xx, explicit(Xx, idx)]
        m, info = mlp_fit(Xe[tr], e["y"][tr], w[tr], Xe[va], e["y"][va], wv[va], seed)
        pred, pext = {arm: mlp_predict(m, Xe)}, {arm: mlp_predict(m, Xx)}
        saved = {arm: {k: v.numpy() for k, v in m.state_dict().items()}}
    info["elapsed_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{f"pred_{k}": v for k, v in pred.items()}, **{f"ext_{k}": v for k, v in pext.items()})
    if split == "main":
        import pickle
        with open(path.replace(".npz", ".model.pkl"), "wb") as f:
            pickle.dump(saved, f)
    C.save_gz(path.replace(".npz", ".info.json.gz"), info)
    return job["key"], path


def fit_jobs() -> List[dict]:
    js = []
    for split in SPLITS:
        for model in MODELS:
            for spec in C.POLICIES:
                js.append({"key": f"{spec}|{split}|{model}", "spec": spec, "split": split, "model": model,
                           "path": os.path.join(C.CKPT, "fits", LB._tag(spec), split, f"{model}.npz")})
    return js


def predictions(spec: str, split: str) -> Dict[str, Dict[str, np.ndarray]]:
    """{'ex': {arm: pred}, 'ext': {arm: pred}} with C and D averaged over the fit seeds (also per seed)."""
    d = os.path.join(C.CKPT, "fits", LB._tag(spec), split)
    out = {"ex": {}, "ext": {}}
    z = np.load(os.path.join(d, "ridge.npz"))
    for a in ("B", "L"):
        out["ex"][a], out["ext"][a] = z[f"pred_{a}"], z[f"ext_{a}"]
    for a in ("C", "D"):
        ps, xs = [], []
        for s in FIT_SEEDS:
            z = np.load(os.path.join(d, f"{a}{s}.npz"))
            ps.append(z[f"pred_{a}"])
            xs.append(z[f"ext_{a}"])
            out["ex"][f"{a}{s}"], out["ext"][f"{a}{s}"] = ps[-1], xs[-1]
        out["ex"][a], out["ext"][a] = np.mean(ps, 0), np.mean(xs, 0)
    return out


def run(workers: int, part: Optional[str] = None) -> None:
    t0 = time.perf_counter()
    for spec in C.POLICIES:                      # the M4.6d external sets (cached; built once per policy)
        _m46d_examples(spec)
    js = fit_jobs()
    if part:
        js = [j for j in js if j["split"] in part.split(",")]
    js.sort(key=lambda j: (j["model"] == "ridge", j["split"] != "main"))
    C.run_jobs(fit_job, js, workers, "supervised fits")
    print(f"  fits done in {time.perf_counter() - t0:.0f} s", flush=True)


__all__ = ["load_points", "examples", "roles", "group_split", "weights", "ridge", "ridge_predict", "make_mlp",
           "mlp_fit", "mlp_predict", "fit_job", "fit_jobs", "predictions", "run", "SPLITS", "MODELS"]
