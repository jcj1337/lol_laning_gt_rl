"""Q2: the capacity probe (pre-registration, stage 1). Can one network represent a generalist and a gap specialist?

Per target t in {0, 2, 4}: G = `final:ext:t` (the M4.5c final), A = its M4.6a gap attacker. `Recorder` (E4's CRNNet
sampling rule) stores every 2nd decision turn's observation, legal mask and the acting policy's probabilities:
    N_train / N_test   G self-play from s0 (48 / 16 games x 500 turns), teacher G in both seats
    P_train            A (seat A, the stronger side) against G from each distinct start of E4's gap training archive
                       (27 constructed cells, the target's 576 E3 training semi-natural states), 150 turns
    P_test             the same from E4's held-out starts (27 cells, 48 held-out semi-natural states)
Students (STUDENTS) fit KL(teacher || student) over the legal actions with M4.6e's optimizer settings; F_G and F_A are
the 2 x 128 single-teacher floors. Behaviour: fit seed 0 of each union student in seat A against G on E4's evaluation
streams (G and A re-run, and must reproduce E4's stored records), and against G from s0 (`evaluate_discounted`).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np

from . import common as C

TARGETS = (0, 2, 4)
UNION = {"U64": (64, 64), "U128": (128, 128), "U256": (256, 256), "U3x128": (128, 128, 128), "U128F": (128, 128)}
FLOORS = {"F_G": (128, 128), "F_A": (128, 128)}
STUDENTS = {**UNION, **FLOORS}
FIT_SEEDS = (0, 1, 2)
N_GAMES = {"N_train": 48, "N_test": 16}
N_TURNS, P_TURNS, EVERY = 500, 150, 2
EPOCHS, PATIENCE, LR, BATCH, VAL_FRAC = 150, 15, 1e-3, 512, 0.15
ROOTS = {"N_train": C.N_TRAIN_ROOT, "P_train": C.P_TRAIN_ROOT, "N_test": C.N_TEST_ROOT, "P_test": C.P_TEST_ROOT}
SETS = tuple(ROOTS)
TEACHER = {0: "G", 1: "A"}


def attacker_path(t: int) -> str:
    from ..m46a.e4 import attacker_dir
    return os.path.join(attacker_dir("gap", t), "final.pt")


def teachers(t: int):
    from ..m46a.common import net_of, spec
    return net_of(spec(t)), net_of(attacker_path(t))


# ---------------------------------------------------------------------------------------------- data
def _recorder_cls():
    from ..m45c.horizon import CRNNet

    class Recorder(CRNNet):
        """CRNNet that stores (observation, legal mask, probabilities) on every EVERY-th decision turn."""

        def __init__(self, net, k0, seed, seat, horizon, rows: list, teacher: int, group: int):
            super().__init__(net, k0, seed, seat, horizon)
            self.rows, self.teacher, self.group, self.seat = rows, teacher, group, seat

        def decide(self, cfg, lane, m4, s: int, i: int):
            import torch
            from ....diagnostics.branch import Snapshot, observation
            obs = observation(cfg, Snapshot(lane, m4, self.k0 + i, (cfg.skill_a, cfg.skill_b)), s)
            legal = list(obs.legal)
            with torch.no_grad():
                logits, _ = self.net(torch.as_tensor(obs.vector, dtype=torch.float32).unsqueeze(0))
            lg = logits[0].double().numpy()[legal]
            p = np.exp(lg - lg.max())
            p /= p.sum()
            if len(legal) == 1:
                return legal[0], "only"
            if i % EVERY == 0:
                self.rows.append((np.asarray(obs.vector, np.float32), legal, p, self.seat, self.teacher, self.group))
            j = min(int(np.searchsorted(np.cumsum(p), self.u[i], side="right")), len(p) - 1)
            return legal[j], "sample"

    return Recorder


def starts(t: int, kind: str) -> list:
    from ..m46a import e3 as E3
    from ..m46a.common import env_cfg, load_gz, state_from_list
    from ..m46a.e4 import GAP_CELLS, archive_path
    from ..m46a.states import zoning_gap_state
    cfg = env_cfg()
    cells = [zoning_gap_state(cfg, *c) for c in GAP_CELLS]
    if kind == "P_train":
        return cells + [x["start"] for x in E3.semi_natural(t, "train")]
    return cells + [state_from_list(x["start"]) for x in load_gz(archive_path("gap", t, "eval"))["semi_natural"]]


def data_path(t: int, name: str) -> str:
    return os.path.join(C.CKPT, "q2", "data", f"t{t}", f"{name}.npz")


def data_job(args) -> tuple:
    t, name = args
    path = data_path(t, name)
    if os.path.exists(path):
        return args, path
    from ...m45b.states import s0
    from ..m46a import engine as EN
    from ..m46a.common import setup_worker
    cfg = setup_worker()
    G, A = teachers(t)
    Rec = _recorder_cls()
    rows: List[tuple] = []
    root = ROOTS[name]
    if name.startswith("N"):
        for g in range(N_GAMES[name]):
            seed = root * 1000 + g
            pa, pb = Rec(G, 0, seed, 0, N_TURNS, rows, 0, g), Rec(G, 0, seed, 1, N_TURNS, rows, 0, g)
            EN.rollout(cfg, s0(cfg), pa, pb, seed, N_TURNS, (), ())
    else:
        for i, st in enumerate(starts(t, name)):
            seed = root * 1000 + i
            pa, pb = Rec(A, st[2], seed, 0, P_TURNS, rows, 1, i), Rec(G, st[2], seed, 1, P_TURNS, rows, 0, i)
            EN.rollout(cfg, st, pa, pb, seed, P_TURNS, (), ())
    n_act = G.n_actions
    obs = np.stack([r[0] for r in rows])
    mask = np.zeros((len(rows), n_act), bool)
    prob = np.zeros((len(rows), n_act), np.float32)
    for k, r in enumerate(rows):
        mask[k, r[1]] = True
        prob[k, r[1]] = r[2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, obs=obs, mask=mask, prob=prob, seat=np.array([r[3] for r in rows], np.int8),
                        teacher=np.array([r[4] for r in rows], np.int8), group=np.array([r[5] for r in rows], np.int32))
    return args, path


def load(t: int, name: str) -> Dict[str, np.ndarray]:
    return dict(np.load(data_path(t, name)))


# ---------------------------------------------------------------------------------------------- fits
def make_student(name: str, seed: int):
    import torch
    from ....rl.network import PolicyValueNet, StrengthFeatureNet
    torch.manual_seed(C.FIT_ROOT + seed)
    if name == "U128F":
        return StrengthFeatureNet(90, 61, STUDENTS[name], C.feature_index())
    return PolicyValueNet(90, 61, STUDENTS[name])


def examples(t: int, name: str, split: str) -> Dict[str, np.ndarray]:
    """The student's examples of one split ('train' or 'test'): union = N + P (each with its teacher); F_G = N + P
    seat B; F_A = P seat A. `src` 0 = N, 1 = P; groups are unique across N and P."""
    n, p = load(t, f"N_{split}"), load(t, f"P_{split}")
    parts = []
    if name != "F_A":
        parts.append((n, np.ones(len(n["seat"]), bool), 0))
    if name in UNION:
        parts.append((p, np.ones(len(p["seat"]), bool), 1))
    elif name == "F_G":
        parts.append((p, p["seat"] == 1, 1))
    else:
        parts.append((p, p["seat"] == 0, 1))
    out = {k: np.concatenate([d[k][m] for d, m, _ in parts]) for k in ("obs", "mask", "prob", "seat", "teacher")}
    out["src"] = np.concatenate([np.full(int(m.sum()), s, np.int8) for _, m, s in parts])
    out["group"] = np.concatenate([d["group"][m] + 1_000_000 * s for d, m, s in parts])
    return out


def val_groups(t: int) -> set:
    """15% of the training groups of N and P (the same for every student of a target)."""
    rng = np.random.default_rng([C.FIT_ROOT, t])
    out = set()
    for name, s in (("N_train", 0), ("P_train", 1)):
        gs = np.unique(load(t, name)["group"]) + 1_000_000 * s
        rng.shuffle(gs)
        out |= set(int(g) for g in gs[:max(1, int(round(VAL_FRAC * len(gs))))])
    return out


def weights(src: np.ndarray) -> np.ndarray:
    w = np.zeros(len(src))
    for s in np.unique(src):
        w[src == s] = 1.0 / (src == s).sum()
    return w / w.mean()


def kl_rows(net, obs, mask, prob, batch: int = 8192) -> np.ndarray:
    """KL(teacher || net) per row over the legal actions, in nats."""
    import torch
    out = []
    with torch.no_grad():
        for b in range(0, len(obs), batch):
            x = torch.as_tensor(obs[b:b + batch])
            m = torch.as_tensor(mask[b:b + batch])
            p = torch.as_tensor(prob[b:b + batch]).double()
            lq = torch.log_softmax(net(x)[0].masked_fill(~m, -1e9), -1).double()
            lp = torch.where(p > 0, torch.log(p.clamp(min=1e-300)), torch.zeros_like(p))
            out.append(torch.where(p > 0, p * (lp - lq), torch.zeros_like(p)).sum(-1).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit_path(t: int, name: str, seed: int) -> str:
    return os.path.join(C.CKPT, "q2", "fits", f"t{t}", f"{name}_{seed}")


def fit_job(args) -> tuple:
    t, name, seed = args
    path = fit_path(t, name, seed)
    if os.path.exists(path + ".json"):
        return args, path
    import torch
    torch.set_num_threads(1)
    from ..m46a.common import env_cfg
    e = examples(t, name, "train")
    vg = val_groups(t)
    va = np.isin(e["group"], sorted(vg))
    tr = ~va
    w = weights(e["src"][tr])
    wv = weights(e["src"][va])
    net = make_student(name, seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)
    gen = torch.Generator().manual_seed(C.FIT_ROOT + 1000 + seed)
    X, M, P, W = (torch.as_tensor(e["obs"][tr]), torch.as_tensor(e["mask"][tr]), torch.as_tensor(e["prob"][tr]),
                  torch.as_tensor(w, dtype=torch.float32))
    best, best_state, bad, ep = float("inf"), None, 0, 0
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X), generator=gen)
        net.train()
        for b in range(0, len(X), BATCH):
            i = perm[b:b + BATCH]
            lq = torch.log_softmax(net(X[i])[0].masked_fill(~M[i], -1e9), -1)
            loss = -(W[i] * (P[i] * lq.masked_fill(~M[i], 0.0)).sum(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        vl = float(np.average(kl_rows(net, e["obs"][va], e["mask"][va], e["prob"][va]), weights=wv))
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    rec = {"target": t, "student": name, "seed": seed, "epochs": ep + 1, "val_kl": best,
           "fit_s": time.perf_counter() - t0, "n_train": int(tr.sum()), "n_val": int(va.sum()),
           "params": int(sum(p.numel() for p in net.parameters())), "test": {}}
    nt, pt = load(t, "N_test"), load(t, "P_test")
    a, b = pt["seat"] == 0, pt["seat"] == 1
    rec["test"]["P_seatA_teacherA"] = float(kl_rows(net, pt["obs"][a], pt["mask"][a], pt["prob"][a]).mean())
    rec["test"]["P_seatB_teacherG"] = float(kl_rows(net, pt["obs"][b], pt["mask"][b], pt["prob"][b]).mean())
    rec["test"]["N_teacherG"] = float(kl_rows(net, nt["obs"], nt["mask"], nt["prob"]).mean())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if name in UNION and seed == 0:
        from ....rl import save_checkpoint
        save_checkpoint(path + ".pt", net, env_cfg(), {"name": f"nc1_q2_{name}_t{t}", "study": "NC1 Q2",
                                                       "student": name, "target": t, "fit_seed": seed})
    C.save_gz(path + ".json.gz", rec)
    with open(path + ".json", "w", encoding="utf-8") as f:
        f.write("{}")
    return args, path


def distances(t: int) -> dict:
    """D = KL(A || G) on P-test seat-A states and KL(G || A) on N test (the scales of lambda and kappa)."""
    G, A = teachers(t)
    pt, nt = load(t, "P_test"), load(t, "N_test")
    a = pt["seat"] == 0
    return {"D": float(kl_rows(G, pt["obs"][a], pt["mask"][a], pt["prob"][a]).mean()),
            "N_KL_G_A": float(kl_rows(A, nt["obs"], nt["mask"], nt["prob"]).mean())}


# ---------------------------------------------------------------------------------------------- behaviour
def behaviour_jobs() -> list:
    from ..m46a import engine as EN
    from ..m46a.common import E4_EVAL_ROOT, env_cfg, load_gz, spec, state_from_list
    from ..m46a.e4 import GAP_CELLS, H_EVAL, K_CON, K_NAT, archive_path
    from ..m46a.states import zoning_gap_state
    cfg = env_cfg()
    js = []
    for t in TARGETS:
        cands = {"target": ("net", spec(t)), "attacker": ("net", attacker_path(t))}
        cands.update({u: ("net", fit_path(t, u, 0) + ".pt") for u in UNION})
        sets = [("con", [zoning_gap_state(cfg, *c) for c in GAP_CELLS], K_CON),
                ("semi", [state_from_list(x["start"]) for x in load_gz(archive_path("gap", t, "eval"))["semi_natural"]],
                 K_NAT)]
        for name, sts, K in sets:
            fam = f"E4gap_{name}"
            for i, st in enumerate(sts):
                js.append({"spec": spec(t), "family": fam, "idx": i, "start": st, "proto": "policy",
                           "horizon": H_EVAL["gap"], "roll": EN.roll_seeds(E4_EVAL_ROOT, K, i), "cands": cands,
                           "critic": False, "tag": f"s{t}",
                           "path": os.path.join(C.CKPT, "q2", "behaviour", fam, f"s{t}", f"{i}.json.gz"),
                           "label": f"{fam} {i}"})
    return js


def natural_job(args) -> tuple:
    t, u = args
    path = os.path.join(C.CKPT, "q2", "natural", f"t{t}_{u}.json.gz")
    if os.path.exists(path):
        return args, C.load_gz(path)
    from ....env.discounted import evaluate_discounted
    from ..m45c.common import agent_from_spec
    from ..m46a.common import env_cfg, setup_worker, spec
    setup_worker()
    cfg = env_cfg()
    e = evaluate_discounted(cfg, agent_from_spec(fit_path(t, u, 0) + ".pt", cfg), agent_from_spec(spec(t), cfg),
                            episodes=64, seed=C.NAT_EVAL_ROOT + t, window=460)
    rec = {k: e[k] for k in ("estimate", "stderr", "ci95", "seat_A", "seat_B", "window")}
    C.save_gz(path, rec)
    return args, rec


def reproduction_check(res: Dict[str, dict]) -> dict:
    """Our target and attacker rollouts against E4's stored records (same streams): must be identical."""
    from ..m46a.common import CKPT as A_CKPT
    checked, bad = 0, []
    for r in res.values():
        p = os.path.join(A_CKPT, "e4", "eval", r["family"], r["tag"], f"{r['idx']}.json.gz")
        old = C.load_gz(p)
        for cand in ("target", "attacker"):
            checked += 1
            if old["G"][cand] != r["G"][cand]:
                bad.append(f"{r['family']} {r['tag']} {r['idx']} {cand}")
    return {"checked": checked, "mismatches": bad[:10], "identical": not bad}


# ---------------------------------------------------------------------------------------------- stages
def run_data(workers: int) -> None:
    jobs = [(t, n) for t in TARGETS for n in ("P_train", "N_train", "P_test", "N_test")]
    C.run_jobs(data_job, jobs, workers, "Q2 data")


def run_fits(workers: int) -> None:
    jobs = [(t, n, s) for s in FIT_SEEDS for n in ("U256", "U3x128", "U128F", "U128", "U64", "F_G", "F_A")
            for t in TARGETS]
    C.run_jobs(fit_job, jobs, workers, "Q2 fits")


def run_behaviour(workers: int) -> dict:
    from ..m46a import families as FA
    res = FA.run(behaviour_jobs(), workers, "Q2 behaviour")
    nat = dict(C.run_jobs(natural_job, [(t, u) for t in TARGETS for u in UNION], workers, "Q2 natural"))
    return {"records": res, "natural": nat}


# ---------------------------------------------------------------------------------------------- analysis
def _behaviour_records() -> Dict[str, dict]:
    root = os.path.join(C.CKPT, "q2", "behaviour")
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".json.gz"):
                out[os.path.join(dp, f)] = C.load_gz(os.path.join(dp, f))
    return out


def behaviour(recs: Dict[str, dict], h: int = 100) -> dict:
    """Per target: cand - target at G_h over every held-out gap state (cells and semi-natural pooled, equal weight per
    state; E4's amendment), U256 - U128 and U128F - U128 paired on the same rollouts, and the retention rho."""
    from ..m46a import engine as EN
    out = {}
    for t in TARGETS:
        rs = [r for r in recs.values() if r["tag"] == f"s{t}"]
        row = {"n_states": len(rs)}
        for cand in ("attacker",) + tuple(UNION):
            m, hw = C.mean_ci([EN.per_state_diff(r, cand, "G", h, "target") for r in rs])
            row[f"{cand} - target"] = {"mean": m, "ci": hw}
        for a, b in (("U256", "U128"), ("U128F", "U128")):
            m, hw = C.mean_ci([EN.per_state_diff(r, a, "G", h, b) for r in rs])
            row[f"{a} - {b}"] = {"mean": m, "ci": hw, "positive_sig": bool(m - hw > 0)}
        att = row["attacker - target"]["mean"]
        row["rho"] = {u: row[f"{u} - target"]["mean"] / att for u in UNION}
        out[t] = row
    att = sum(out[t]["attacker - target"]["mean"] for t in TARGETS)
    pooled = {u: sum(out[t][f"{u} - target"]["mean"] for t in TARGETS) / att for u in UNION}
    return {"per_target": out, "pooled_rho": pooled}


def analyse() -> dict:
    per = {}
    for t in TARGETS:
        d = distances(t)
        kl = {}
        for n in STUDENTS:
            fs = [C.load_gz(fit_path(t, n, s) + ".json.gz") for s in FIT_SEEDS]
            kl[n] = {k: float(np.mean([f["test"][k] for f in fs])) for k in fs[0]["test"]}
            kl[n]["epochs"] = [f["epochs"] for f in fs]
        lam = {u: (kl[u]["P_seatA_teacherA"] - kl["F_A"]["P_seatA_teacherA"]) / d["D"] for u in UNION}
        kap = {u: kl[u]["N_teacherG"] - kl["F_G"]["N_teacherG"] for u in UNION}
        per[t] = {"distances": d, "kl": kl, "lambda": lam, "kappa": kap, "lambda_applies": d["D"] >= 0.05}
    recs = _behaviour_records()
    beh = behaviour(recs) if recs else None
    nat = {}
    for t in TARGETS:
        for u in UNION:
            p = os.path.join(C.CKPT, "q2", "natural", f"t{t}_{u}.json.gz")
            if os.path.exists(p):
                nat[f"t{t}_{u}"] = C.load_gz(p)
    repro = reproduction_check(recs) if recs else None
    lam = {t: per[t]["lambda"] for t in TARGETS}
    regress = sorted(k for k, v in nat.items() if v["ci95"][1] < -0.3)

    def sig2(key: str) -> bool:
        return beh is not None and sum(beh["per_target"][t][key]["positive_sig"] for t in TARGETS) >= 2

    rules = {
        "2x128 suffices": all(lam[t]["U128"] <= 0.10 and lam[t]["U256"] >= lam[t]["U128"] - 0.05 for t in TARGETS)
        and beh is not None and beh["pooled_rho"]["U128"] >= 0.5,
        "width binds": all(lam[t]["U128"] - lam[t]["U256"] >= 0.10 for t in TARGETS) and sig2("U256 - U128"),
        "features help": all(lam[t]["U128"] - lam[t]["U128F"] >= 0.10 for t in TARGETS) and sig2("U128F - U128"),
    }
    rules["partial"] = not any(rules.values())
    out = {"meta": C.meta(stage="q2"), "per_target": per, "behaviour": beh, "natural": nat,
           "natural_regressions": regress, "reproduction": repro, "rules": rules}
    C.save_json("q2", out)
    return out
