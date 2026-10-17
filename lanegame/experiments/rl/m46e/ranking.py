"""Look-ahead rankings of the fitted value predictors at the M4.6d decision states (secondary readout).

M4.6d (`m46d.critic.rank_job`, and the exploratory Yield follow-up `results/m46d/run_rank_yield.py`) compared, at 40
decision states per C1 policy plus 8 weak-side Recall windows, each candidate one-turn action (then the policy; K 16,
150 turns, no bootstrap) with the critic's look-ahead r_0 + gamma V(s_1). Here turn 0 of every stored rollout is
replayed on its own seeds (the first chance vector and the CRN sampling uniform of turn 0), the replay is checked to
reproduce the stored critic look-ahead exactly, and the look-ahead is recomputed with each main-split predictor of the
same policy (B, C, D, L) in place of the critic. Rollout values are the stored ones. A better ranking from a fitted
predictor says the value is learnable from these labels; it says nothing about how an actor trained against it would
play.
"""

from __future__ import annotations

import glob
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

from ....champions import N_CHANCE
from . import common as C
from . import fit as F
from . import labels as LB
from .features import explicit, obs_index

ARMS = ("critic", "B", "L", "C", "D")
_PRED: Dict[str, dict] = {}


def predictors(spec: str):
    """{arm: f(obs (n, 90)) -> values in resources} for one policy (main-split models)."""
    if spec in _PRED:
        return _PRED[spec]
    import torch
    cfg = C.arm_cfg(C.ARM)
    idx = obs_index(cfg)
    net = C.net_of(spec)
    d = os.path.join(C.CKPT, "fits", LB._tag(spec), "main")
    with open(os.path.join(d, "ridge.model.pkl"), "rb") as f:
        rm = pickle.load(f)
    mlps = {"C": [], "D": []}
    for a in ("C", "D"):
        for s in F.FIT_SEEDS:
            with open(os.path.join(d, f"{a}{s}.model.pkl"), "rb") as f:
                sd = pickle.load(f)[a]
            m = F.make_mlp(90 if a == "C" else 96)
            m.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
            m.eval()
            mlps[a].append(m)

    def critic(X):
        with torch.no_grad():
            return net.value(torch.as_tensor(X, dtype=torch.float32)).double().numpy() / C.REWARD_SCALE

    def aug(X):
        return np.c_[X, explicit(X, idx)]

    out = {"critic": critic,
           "B": lambda X: F.ridge_predict(rm["B"], F.trunk(net, X)),
           "L": lambda X: F.ridge_predict(rm["L"], aug(X)),
           "C": lambda X: np.mean([F.mlp_predict(m, X) for m in mlps["C"]], 0),
           "D": lambda X: np.mean([F.mlp_predict(m, aug(X)) for m in mlps["D"]], 0)}
    _PRED[spec] = out
    return out


def stored_jobs() -> List[dict]:
    """The M4.6d ranking jobs (rank and the Yield follow-up) with their start states and candidate codes."""
    from ....config import B_SAFE, B_YIELD, W_HOLD, W_SHOVE, champ_act
    from ..m46d import critic as CR
    js = [dict(j) for j in CR.rank_jobs()]
    cands = {"LH-Safe": champ_act(W_HOLD, B_SAFE), "SH-Safe": champ_act(W_SHOVE, B_SAFE),
             "LH-Yield": champ_act(W_HOLD, B_YIELD), "SH-Yield": champ_act(W_SHOVE, B_YIELD)}
    cfg = C.arm_cfg(C.ARM)
    j = 0
    for cond in ("edge", "x1.76"):
        for L in (5, 8):
            for hpb in (0.5, 0.35):
                st = CR.recall_window(cfg, cond, L, hpb)
                for s in C.SEEDS:
                    js.append({"arm": "C1", "spec": C.general("C1", s), "tag": f"s{s}", "family": "recallw_yield",
                               "idx": j, "decider": 1, "start": C.state_to_list(st), "cands": cands, "K": CR.K_RANK,
                               "H": CR.H_RANK, "root": C.D.CRIT_ROOT + 1,
                               "label": f"recall window L{L} {cond} hpB{hpb}",
                               "path": os.path.join(C.D_CKPT, "critic", "rank_yield", f"s{s}", f"recallw_{j}_d1.json.gz")})
                j += 1
    return js


def rank_job(job: dict) -> tuple:
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    from ..m45c import horizon as HZ
    from ....diagnostics.branch import Snapshot, observation
    cfg = C.setup(C.ARM)
    rec = C.load_gz(job["path"])
    net = C.net_of(job["spec"])
    preds = predictors(job["spec"])
    lane0, m40, k0 = C.state_from_list(job["start"])
    dec = int(job["decider"])
    sign = 1.0 if dec == 0 else -1.0
    K, H = int(job["K"]), int(job["H"])
    ctx = context(cfg)
    ls = E.lane_cache(cfg)
    look: Dict[str, Dict[str, np.ndarray]] = {}
    max_err = 0.0
    for name in rec["cands"]:
        code = None if name == "policy" else int(job["cands"][name])
        r0s, obs = [], []
        for r in range(K):
            sd = job["root"] * 1000 + r
            pols = [HZ.CRNNet(net, k0, sd, 0, H), HZ.CRNNet(net, k0, sd, 1, H)]
            rng = np.random.default_rng(sd)
            u = rng.random(N_CHANCE)
            acts = [int(code) if (s == dec and code is not None) else int(E.decide(pols[s], cfg, lane0, m40, s, 0))
                    for s in (0, 1)]
            t = resolve_turn_m4(lane0, m40, k0, acts[0], acts[1], cfg, lane_step=ls, u=u, ctx=ctx)
            r0s.append(sign * t.reward)
            obs.append(observation(cfg, Snapshot(t.lane.state, t.champ, k0 + 1, (cfg.skill_a, cfg.skill_b)), dec).vector)
        X = np.stack(obs)
        r0 = np.asarray(r0s)
        look[name] = {a: r0 + C.GAMMA * f(X) for a, f in preds.items()}
        max_err = max(max_err, float(np.max(np.abs(look[name]["critic"] - np.asarray(rec["cands"][name]["crit"])))))
    j150 = rec["cps"].index(150)
    out = {"family": rec["family"], "idx": rec["idx"], "decider": dec, "label": rec["label"], "spec": job["spec"],
           "replay_max_abs_error": max_err, "roll": {}, "look": {}, "probs": rec["probs"]}
    for name, v in rec["cands"].items():
        out["roll"][name] = float(np.mean(np.asarray(v["G"])[:, j150]))
        out["look"][name] = {a: float(np.mean(x)) for a, x in look[name].items()}
    return f"{job['spec']}|{rec['family']}|{rec['idx']}|{dec}", out


def summarise(rows: List[dict]) -> dict:
    from scipy.stats import spearmanr
    per = defaultdict(list)
    for r in rows:
        per[(r["family"], r["idx"], r["decider"], r["label"])].append(r)
    agg = defaultdict(lambda: defaultdict(list))
    states = []
    for (fam, idx, dec, lab), rs in sorted(per.items()):
        st = {a: defaultdict(list) for a in ARMS}
        for r in rs:
            names = [c for c in r["roll"] if c != "policy"]
            if len(names) < 3:
                continue
            adv_r = {c: r["roll"][c] - r["roll"]["policy"] for c in names}
            for a in ARMS:
                adv_p = {c: r["look"][c][a] - r["look"]["policy"][a] for c in names}
                st[a]["rho"].append(float(spearmanr([adv_r[c] for c in names], [adv_p[c] for c in names])[0]))
                st[a]["sign"].append(float(np.mean([np.sign(adv_r[c]) == np.sign(adv_p[c]) for c in names])))
                st[a]["best"].append(float(max(names, key=adv_r.get) == max(names, key=adv_p.get)))
                st[a]["spread"].append(float(np.std([adv_p[c] for c in names])))
            st["roll_spread"] = st.get("roll_spread", []) + [float(np.std([adv_r[c] for c in names]))]
        if not st["critic"]["rho"]:
            continue
        key = f"{fam} decider {'AB'[dec]}"
        srec = {"family": fam, "label": lab, "decider": dec, "roll_spread": float(np.mean(st["roll_spread"]))}
        for a in ARMS:
            srec[a] = {k: float(np.nanmean(v)) for k, v in st[a].items()}
            agg[key][a].append(srec[a])
        agg[key]["roll_spread"].append(srec["roll_spread"])
        states.append(srec)
    summary = {}
    for key, d in agg.items():
        summary[key] = {"states": len(d["critic"]), "roll_spread": float(np.mean(d["roll_spread"]))}
        for a in ARMS:
            summary[key][a] = {m: float(np.nanmean([x[m] for x in d[a]])) for m in ("rho", "sign", "best", "spread")}
    return {"summary": summary, "states": states}


def run(workers: int) -> None:
    t0 = time.perf_counter()
    js = [j for j in stored_jobs() if os.path.exists(j["path"])]
    rows = [r for _, r in C.run_jobs(rank_job, js, workers, "ranking replay")]
    res = {"meta": C.meta(stage="ranking"), "jobs": len(rows),
           "replay_max_abs_error": max(r["replay_max_abs_error"] for r in rows), **summarise(rows),
           "elapsed_s": time.perf_counter() - t0}
    C.save_json("ranking", res)
    print(f"  ranking: {len(rows)} jobs, replay error {res['replay_max_abs_error']:.2e}, "
          f"{res['elapsed_s']:.0f} s", flush=True)


__all__ = ["predictors", "stored_jobs", "rank_job", "summarise", "run", "ARMS"]
