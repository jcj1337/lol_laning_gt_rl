"""Value evaluation of M4.6f (pre-registration "Value"). Every label is the evaluated network playing ITSELF in the C1
environment, from the labelled seat, discounted with gamma 0.99, without bootstrap, in resources: the return that
network's own critic is trained to predict.

Parts (each resumable; per-job records under results/rl/checkpoints/m46f/value/):
    calib      the M4.6d calibration set (`m46d.critic.calib_job`): 42 cells x K 32 x H 300 and the 16 natural lead /
               even states of start:s (M4.6d's selection) x K 32 x H 500, on M4.6d's rollout seeds (CALIB_ROOT), so a
               new critic's labels pair rollout for rollout with the starting critic's M4.6d labels. Phase "mid": the
               2.048M snapshots, cells only, K 16 (the first 16 of the same seeds).
    natural    32 s0 self-play games of 600 turns per network (finals and starts; NAT_ROOT seeds), labels every 10
               turns from 5 to 295 (G300, both seats), with the observations and legal masks (policy drift)
    probes     ridge probes from the representation the value head reads (`value_features`) and from the actor's
               trunk to the log duel ratio and the own HP fraction, on the M4.6e labelled observations of seed s
               (M4.6e's main split); the critic's response to the M4.6e single-feature changes
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import common as C

K_CAL, K_CAL_MID, H_CAL, H_NAT = 32, 16, 300, 500
NAT_GAMES, NAT_TURNS, NAT_EVERY, NAT_FIRST, NAT_H = 32, 600, 10, 5, 300


def _tag(arm: str, s: int, phase: str) -> str:
    return {"main": f"{arm}_s{s}", "mid": f"{arm}_s{s}_mid", "ext": f"ext_{arm}_s{s}"}[phase]


def policies(phase: str) -> List[Tuple[str, str, int, str]]:
    """(tag, checkpoint path, seed, arm) of the networks a phase evaluates."""
    out = []
    for s in C.SEEDS:
        for a in C.ARMS:
            if phase == "main":
                p = C.policy_path(f"{a}:{s}")
            elif phase == "mid":
                p = C.snapshot_path(a, s, C.MIDPOINT)
            elif phase == "ext":
                p = C.policy_path(f"ext/{a}:{s}")
            else:
                raise ValueError(phase)
            out.append((_tag(a, s, phase), p, s, a))
    return out


# ---------------------------------------------------------------------------------------------- calibration
def _natural_states() -> dict:
    from ..m46d import common as D
    return C.load_gz(os.path.join(D.CKPT, "critic_natural_states.json.gz"))


def calib_jobs(phase: str) -> List[dict]:
    from ..m46d.critic import _cell_states
    cfg = C.env_cfg()
    cells = _cell_states(cfg)
    nat = _natural_states()
    js = []
    for tag, path, s, arm in policies(phase):
        items = list(cells)
        if phase != "mid":
            for kind in ("lead", "even"):
                for i, rec in enumerate(nat[f"C1|ext/C1:{s}"][kind]):
                    items.append((f"natural_{kind}", i, C.state_from_list(rec["start"]), f"natural {kind} t{rec['turn']}"))
        for fam, i, st, lab in items:
            js.append({"arm": C.ENV, "spec": path, "tag": tag, "family": fam, "idx": i, "start": C.state_to_list(st),
                       "K": K_CAL_MID if phase == "mid" else K_CAL, "H": H_NAT if fam.startswith("natural") else H_CAL,
                       "root": C.CALIB_ROOT, "label": lab,
                       "path": os.path.join(C.CKPT, "value", "calib", tag, f"{fam}_{i}.json.gz")})
    return js


def calib_dir(tag: str) -> str:
    return os.path.join(C.CKPT, "value", "calib", tag)


def start_calib_dir(s: int) -> str:
    from ..m46d import common as D
    return os.path.join(D.CKPT, "critic", "calib", "C1", f"s{s}")


def load_calib(d: str, K: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Both seats of every calibration record in `d`: label, family, seat, y (mean G300, seat B negated), y500, pred
    (the critic, resources). `K` keeps only the first K rollouts (the start's K 32 records against a K 16 phase)."""
    rows = {k: [] for k in ("label", "fam", "seat", "y", "y500", "pred", "y_se", "idx")}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json.gz"):
            continue
        rec = C.load_gz(os.path.join(d, fn))
        G = np.asarray(rec["G"], float)
        if K is not None:
            G = G[:K]
        cps = rec["cps"]
        g3 = G[:, cps.index(300)]
        g5 = G[:, cps.index(500)].mean() if 500 in cps else np.nan
        for seat, v, sg in ((0, rec["VA"], 1.0), (1, rec["VB"], -1.0)):
            rows["label"].append(rec["label"])
            rows["fam"].append(rec["family"])
            rows["seat"].append(seat)
            rows["y"].append(sg * g3.mean())
            rows["y500"].append(sg * g5)
            rows["pred"].append(float(v))
            rows["y_se"].append(g3.std(ddof=1) / np.sqrt(len(g3)))
            rows["idx"].append(int(rec["idx"]))
    return {k: np.asarray(v) for k, v in rows.items()}


def calib_readouts(ext: Dict[str, np.ndarray]) -> dict:
    """M4.6e's P1 / P2 readouts (`m46e.analysis.m46d_readouts`) plus the mean predicted and realized premiums."""
    from ..m46e.analysis import CELL_RE, m46d_readouts
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = m46d_readouts(ext, ext["pred"])
    # predicted / realized premiums in resources (both seats oriented as the stronger side's gain)
    lab, seat, y, p = ext["label"], ext["seat"], ext["y"], ext["pred"]
    cell = {}
    for i, l in enumerate(lab):
        mm = CELL_RE.match(str(l))
        if mm:
            cell[(mm.group(1), int(mm.group(2)), mm.group(3), int(seat[i]))] = i
    prem = {}
    for (w, L, name, s), i in cell.items():
        j = cell.get((w, L, "equal", s))
        if j is None or name == "equal":
            continue
        sg = 1.0 if s == 0 else -1.0
        prem.setdefault(name, []).append((sg * (y[i] - y[j]), sg * (p[i] - p[j])))
    out["premiums"] = {k: {"n": len(v), "realized": float(np.mean([a for a, _ in v])),
                           "pred": float(np.mean([b for _, b in v]))} for k, v in prem.items()}
    ep = prem.get("edge", []) + prem.get("plaus", [])
    if ep:
        out["premiums"]["edge+plaus"] = {"n": len(ep), "realized": float(np.mean([a for a, _ in ep])),
                                         "pred": float(np.mean([b for _, b in ep]))}
    return out


# ---------------------------------------------------------------------------------------------- natural labels
def natural_jobs(phase: str) -> List[dict]:
    js = []
    pols = policies(phase)
    if phase == "main":
        pols = pols + [(f"start_s{s}", C.start_path(s), s, "start") for s in C.SEEDS]
    for tag, path, s, arm in pols:
        for b in range(0, NAT_GAMES, 8):
            js.append({"tag": tag, "path_net": path, "seed": s, "games": list(range(b, min(b + 8, NAT_GAMES))),
                       "path": os.path.join(C.CKPT, "value", "natural", tag, f"games_{b:02d}.npz")})
    return js


def natural_job(job: dict) -> tuple:
    path = job["path"]
    if os.path.exists(path):
        return job["tag"], path
    from ...m45b import states as S
    from ..m46e import labels as LB
    cfg = C.setup()
    net = C.net_of(job["path_net"])
    at = set(range(NAT_FIRST, NAT_TURNS - NAT_H + 1, NAT_EVERY))
    from ..m46e.features import StateFeatures
    sf = StateFeatures(cfg)
    parts = []
    for g in job["games"]:
        rec = {"t": [], "obs": [], "mask": [], "V": [], "duel": [], "hp": []}

        def grab(i, lane, m4, obs, vals):
            if i in at:
                rec["t"].append(i)
                rec["obs"].append(np.stack([obs[0].vector, obs[1].vector]))
                rec["mask"].append(np.stack([obs[0].mask, obs[1].mask]))
                rec["V"].append(vals)
                f = sf.side(lane, m4, 0)
                rec["duel"].append(f["duel"])
                rec["hp"].append((f["hp"], f["hp_o"]))

        r = LB.play(cfg, net, S.s0(cfg), C.NAT_ROOT * 1000 + 100 * job["seed"] + g, NAT_TURNS, grab)
        out = {k: np.asarray(v) for k, v in rec.items()}
        out["G300"] = np.asarray([_disc(r, t, NAT_H) for t in rec["t"]])
        out["G50"] = np.asarray([_disc(r, t, 50) for t in rec["t"]])
        out["game"] = np.full(len(rec["t"]), g, np.int32)
        parts.append(out)
    z = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **z)
    return job["tag"], path


def _disc(rewards, start: int, horizon: int) -> float:
    r = np.asarray(rewards[start:start + horizon], dtype=np.float64)
    return float(np.dot(C.GAMMA ** np.arange(len(r)), r))


def load_natural(tag: str) -> Dict[str, np.ndarray]:
    d = os.path.join(C.CKPT, "value", "natural", tag)
    ps = [dict(np.load(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith(".npz")]
    return {k: np.concatenate([p[k] for p in ps]) for k in ps[0]}


def natural_readouts(z: Dict[str, np.ndarray]) -> dict:
    """Both seats: y = G300 (seat B negated), pred = the critic; MAE, bias, slope, R^2."""
    from ..m46e.analysis import basic
    y = np.r_[z["G300"], -z["G300"]]
    p = np.r_[z["V"][:, 0], z["V"][:, 1]]
    out = basic(y, p)
    out["corr"] = float(np.corrcoef(y, p)[0, 1])
    return out


def drift(seed: int, phase: str = "main") -> dict:
    """Policy drift on the start's own natural states (both seats): mean KL(start || X) for each arm's final and the
    mean total-variation distance between the two arms' finals."""
    import torch
    from ....rl.network import MaskedCategorical
    z = load_natural(f"start_s{seed}")
    x = torch.as_tensor(np.concatenate([z["obs"][:, 0], z["obs"][:, 1]]), dtype=torch.float32)
    m = torch.as_tensor(np.concatenate([z["mask"][:, 0], z["mask"][:, 1]]))
    probs = {}
    for name, path in (("start", C.start_path(seed)), ("shared", policies_path("shared", seed, phase)),
                       ("separate", policies_path("separate", seed, phase))):
        with torch.no_grad():
            lg, _ = C.net_of(path)(x)
            probs[name] = MaskedCategorical(lg, m).probs.double().numpy()
    choice = m.numpy().sum(-1) > 1

    def kl(p, q):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(p > 0, p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300))), 0.0)
        return float(t.sum(-1)[choice].mean())

    return {"kl_start_shared": kl(probs["start"], probs["shared"]),
            "kl_start_separate": kl(probs["start"], probs["separate"]),
            "tv_shared_separate": float(0.5 * np.abs(probs["shared"] - probs["separate"]).sum(-1)[choice].mean()),
            "states": int(choice.sum())}


def policies_path(arm: str, s: int, phase: str) -> str:
    return {"main": C.policy_path(f"{arm}:{s}"), "mid": C.snapshot_path(arm, s, C.MIDPOINT),
            "ext": C.policy_path(f"ext/{arm}:{s}")}[phase]


# ---------------------------------------------------------------------------------------------- probes
def probe_job(args) -> tuple:
    """(tag, path, seed): held-out R^2 of ridge probes (module docstring) and the single-feature critic responses."""
    tag, path, s = args
    out_path = os.path.join(C.CKPT, "value", "probes", f"{tag}.json.gz")
    if os.path.exists(out_path):
        return tag, C.load_gz(out_path)
    import torch
    from ..m46e import fit as F
    from ..m46e import observation as O
    C.setup()
    e = F.examples(F.load_points(f"ext/C1:{s}"))
    r = F.roles(e, "main")
    tr, va, te = r == "train", r == "val", r == "test"
    net = C.net_of(path)
    with torch.no_grad():
        x = torch.as_tensor(e["x"])
        feats = {"value_features": net.value_features(x).double().numpy(), "actor_trunk": net.trunk(x).double().numpy()}
    res = {}
    for tgt in ("log_duel", "hp_own"):
        y = e[tgt]
        for name, Fm in feats.items():
            m = F.ridge(Fm[tr], y[tr], np.ones(tr.sum()), Fm[va], y[va], np.ones(va.sum()))
            p = F.ridge_predict(m, Fm[te])
            res[f"{tgt}<-{name}"] = float(1 - ((p - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    # a strength contrast's movement of the value representation relative to its natural spread (M4.6e §2.3)
    nat = e["src"] == 0
    h = feats["value_features"]
    spread = float(np.linalg.norm(h[nat] - h[nat].mean(0), axis=1).mean())
    from ..m46d.critic import _cell_states
    cfg = C.env_cfg()
    moves = []
    cells = {lab: st for _, _, st, lab in _cell_states(cfg)}
    for lab, st in cells.items():
        if " x1.76 " in lab and lab.replace("x1.76", "equal") in cells:
            st0 = cells[lab.replace("x1.76", "equal")]
            v1 = O._obs(cfg, st, 0).vector
            v0 = O._obs(cfg, st0, 0).vector
            with torch.no_grad():
                f = net.value_features(torch.as_tensor(np.stack([v1, v0]), dtype=torch.float32)).double().numpy()
            moves.append(float(np.linalg.norm(f[0] - f[1])))
    res["value_features_move_x1.76_over_natural_spread"] = float(np.mean(moves) / spread) if moves else float("nan")
    res["responses"] = {f"{r_['base']}|{r_['change']}": r_["dV_A"] for r_ in O.pairs(path)}
    C.save_gz(out_path, res)
    return tag, res


def probe_jobs(phase: str) -> List[tuple]:
    js = [(tag, path, s) for tag, path, s, _ in policies(phase)]
    if phase == "main":
        js += [(f"start_s{s}", C.start_path(s), s) for s in C.SEEDS]
    return js


# ---------------------------------------------------------------------------------------------- driver
def run(workers: int, phase: str = "main", part: Optional[str] = None) -> None:
    from ..m46d.critic import calib_job
    parts = (part or ("calib" if phase == "mid" else "calib,natural,probes")).split(",")
    for p in parts:
        t0 = time.perf_counter()
        if p == "calib":
            js = sorted(calib_jobs(phase), key=lambda j: -j["H"] * j["K"])
            C.run_jobs(calib_job, js, workers, f"calibration {phase}")
        elif p == "natural":
            js = natural_jobs(phase)
            C.run_jobs(natural_job, js, workers, f"natural labels {phase}")
        elif p == "probes":
            js = probe_jobs(phase)
            C.run_jobs(probe_job, js, workers, f"probes {phase}")
        else:
            raise SystemExit(f"unknown value part {p!r}")
        print(f"  [{p}] {len(js)} jobs in {time.perf_counter() - t0:.0f} s", flush=True)
        C.jdump(os.path.join(C.OUT, "stages", f"value_{phase}_{p}.json"),
                {"part": p, "phase": phase, "jobs": len(js), "wall_s": time.perf_counter() - t0,
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")})


__all__ = ["policies", "calib_jobs", "calib_dir", "start_calib_dir", "load_calib", "calib_readouts", "natural_jobs",
           "natural_job", "load_natural", "natural_readouts", "drift", "probe_job", "probe_jobs", "run"]
