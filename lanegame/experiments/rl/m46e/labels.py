"""Q3: policy-specific, no-bootstrap return labels, and the horizon / residual-tail check.

Every label is the policy playing ITSELF in the C1 environment from a start state, with the M4.6d sampling convention
(`m45c.horizon.CRNNet`: the chance stream default_rng(seed), N_CHANCE uniforms per turn; seat s's sampling uniform at
turn i from default_rng([seed, 7, s])), so a label is a sample of the return the policy's own critic is trained to
predict. G_h(t) = sum_{i < h} gamma^i r_{t+i} is seat A's discounted earned-gold lead from turn t (resources); seat
B's label is -G_h(t) (zero-sum). No critic value enters any label.

Sets (per policy ext/C1:s, s = 0..5; identical start states and seeds for every policy):
    natural       NAT_GAMES s0 self-play games of NAT_TURNS turns; labels at t = 5, 15, ..., 295 (G_300 from t)
    constructed   a fixed stratified sample of the M4.6c TRAINING archive (ARCHIVE_SAMPLE: natural, natural+lead,
                  zone, freeze) and paired variants of each sampled state:
                      orig    as archived
                      eq      the archived investment lead removed (natural+lead: the original natural state; zone:
                              A's investment set to B's; freeze: both at the base investment)
                      lvl     A one level higher (first XP of L+1, same HP fraction)                (zone, freeze)
                      hpB     B's HP at 0.6 of its archived HP                                       (zone, freeze)
                      big     A's investment set so that M(S_A) / M(S_B) = m, m log-uniform in [1.3, 1.9] per state
                              (duel about x1.7-3.6; covers the M4.6d large-gap range)            (zone, freeze)
                  K_CON rollouts of LABEL_LAST + H_LABEL turns per variant (the same seeds for every variant of a
                  state, so variant contrasts are paired); labels at t = 0, 15, ..., 150 (G_300 from t)
    m46d          the M4.6d calibration states of each policy (constructed cells and natural lead / even states),
                  labelled there with K 32 and H 300 / 500 (reused, not re-simulated): the precise external test
`tail`: a representative subset of the problematic constructed cells, three policies, K 32, H 800 on new seeds (the
independent check of the label horizon's residual tail; M4.6d checked only natural states).
"""

from __future__ import annotations

import os
import re
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from ....champions import N_CHANCE
from . import common as C
from .features import StateFeatures

VARIANTS = ("orig", "eq", "lvl", "hpB", "big")
FEAT_KEYS = ("L", "L_o", "S", "S_o", "hp", "hp_o", "present", "present_o", "duel", "held", "frozen", "held_hp",
             "catch_o", "dist", "pos")


# ---------------------------------------------------------------------------------------------- the rollout core
class SelfPlay:
    """One network for both seats, CRN sampling (module docstring), one batched forward per turn."""

    def __init__(self, cfg, net, seed: int, turns: int, k0: int):
        self.cfg, self.net, self.k0 = cfg, net, int(k0)
        self.u = [np.random.default_rng([int(seed), 7, s]).random(int(turns) + 1) for s in (0, 1)]

    def step(self, lane, m4, i: int):
        """(actions, observations (A, B), values in resources (A, B)) at turn index i."""
        import torch
        from ....diagnostics.branch import Snapshot, observation
        cfg = self.cfg
        snap = Snapshot(lane, m4, self.k0 + i, (cfg.skill_a, cfg.skill_b))
        obs = [observation(cfg, snap, s) for s in (0, 1)]
        with torch.no_grad():
            logits, v = self.net(torch.as_tensor(np.stack([o.vector for o in obs]), dtype=torch.float32))
        lg = logits.double().numpy()
        acts = []
        for s in (0, 1):
            legal = list(obs[s].legal)
            if len(legal) == 1:
                acts.append(int(legal[0]))
                continue
            z = lg[s, legal]
            p = np.exp(z - z.max())
            p /= p.sum()
            j = min(int(np.searchsorted(np.cumsum(p), self.u[s][i], side="right")), len(p) - 1)
            acts.append(int(legal[j]))
        return acts, obs, (float(v[0]) / C.REWARD_SCALE, float(v[1]) / C.REWARD_SCALE)


def play(cfg, net, start, seed: int, turns: int, on_turn: Optional[Callable] = None) -> np.ndarray:
    """One self-play game of `turns` turns from `start` = (lane, m4, k0); returns seat A's reward per turn.
    `on_turn(i, lane, m4, obs, values)` sees every start-of-turn state."""
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    lane, m4, k0 = start
    ctx = context(cfg)
    ls = E.lane_cache(cfg)
    rng = np.random.default_rng(int(seed))
    sp = SelfPlay(cfg, net, seed, turns, k0)
    r = np.zeros(turns)
    for i in range(turns):
        u = rng.random(N_CHANCE)
        acts, obs, vals = sp.step(lane, m4, i)
        if on_turn is not None:
            on_turn(i, lane, m4, obs, vals)
        t = resolve_turn_m4(lane, m4, k0 + i, acts[0], acts[1], cfg, lane_step=ls, u=u, ctx=ctx)
        r[i] = t.reward
        lane, m4 = t.lane.state, t.champ
    return r


class Recorder:
    """Collects observations, critic values and state features at the labelled turns."""

    def __init__(self, cfg, at):
        self.at = set(int(x) for x in at)
        self.sf = StateFeatures(cfg)
        self.t, self.obs, self.V, self.feat = [], [], [], {k: [] for k in FEAT_KEYS}

    def __call__(self, i, lane, m4, obs, vals):
        if i not in self.at:
            return
        self.t.append(i)
        self.obs.append(np.stack([obs[0].vector, obs[1].vector]))
        self.V.append(vals)
        f = self.sf.side(lane, m4, 0)
        for k in FEAT_KEYS:
            self.feat[k].append(f[k])

    def arrays(self, rewards: np.ndarray, horizons=(50, 100, 200, C.H_LABEL)) -> Dict[str, np.ndarray]:
        out = {"t": np.asarray(self.t, np.int32), "obs": np.asarray(self.obs, np.float32),
               "V": np.asarray(self.V, np.float64)}
        for h in horizons:
            out[f"G{h}"] = np.asarray([C.discounted(rewards, t, h) for t in self.t])
        for k in FEAT_KEYS:
            out[f"f_{k}"] = np.asarray(self.feat[k], np.float64)
        return out


def _tag(spec: str) -> str:
    return spec.replace("/", "_").replace(":", "_").replace("@", "_")


# ---------------------------------------------------------------------------------------------- natural labels
def natural_job(job: dict) -> tuple:
    path = job["path"]
    if os.path.exists(path):
        return job["key"], path
    cfg = C.setup(C.ARM)
    from ...m45b import states as S
    net = C.net_of(job["spec"])
    at = range(C.NAT_FIRST, C.NAT_TURNS - C.H_LABEL + 1, C.NAT_EVERY)
    parts = []
    for g in job["games"]:
        rec = Recorder(cfg, at)
        r = play(cfg, net, S.s0(cfg), C.NAT_ROOT * 1000 + g, C.NAT_TURNS, rec)
        a = rec.arrays(r)
        a["game"] = np.full(len(a["t"]), g, np.int32)
        parts.append(a)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return job["key"], path


def natural_jobs() -> List[dict]:
    js = []
    for spec in C.POLICIES:
        for b in range(0, C.NAT_GAMES, 8):
            games = list(range(b, min(b + 8, C.NAT_GAMES)))
            js.append({"key": f"{spec}|nat{b}", "spec": spec, "games": games,
                       "path": os.path.join(C.CKPT, "labels", _tag(spec), f"natural_{b}.npz")})
    return js


# ---------------------------------------------------------------------------------------------- constructed labels
LABEL_RE = {
    "natural+lead": re.compile(r"d=([0-9.]+)"),
    "zone": re.compile(r"zone (\w+) L(\d+) S([0-9.]+) d=([0-9.]+)"),
    "freeze": re.compile(r"freeze L(\d+) held(\d+) (\w+) d=([0-9.]+)"),
}


def archive_sample() -> List[dict]:
    """The fixed stratified archive sample (same for every policy): [{idx, kind, label, start}], M4.6d calibration
    cells excluded by key."""
    from ....env.starts import load_archive
    from ..m46c.archive import archive_path
    cfg = C.arm_cfg(C.ARM)
    sts = load_archive(archive_path(), cfg)
    by: Dict[str, List[int]] = {}
    for i, st in enumerate(sts):
        by.setdefault(st.label.split(" ")[0], []).append(i)
    excl = set()
    for _, _, start, _ in _m46d_cells(cfg):
        lane, m4, k = start
        excl.add((lane, m4, k % cfg.phase_period))
        excl.add((lane.mirror(), m4.mirror(), k % cfg.phase_period))
    rng = np.random.default_rng(C.SAMPLE_ROOT)
    out = []
    for kind, n in C.ARCHIVE_SAMPLE.items():
        pick = [int(x) for x in rng.choice(by[kind], size=n, replace=False)]
        for i in pick:
            st = sts[i]
            if st.key() in excl:
                continue
            out.append({"archive_index": i, "kind": kind, "label": st.label, "start": (st.lane, st.champ, st.phase)})
    for j, d in enumerate(out):
        d["group"] = j
    return out


def variants(cfg, item: dict) -> Dict[str, tuple]:
    """The paired variants of one sampled archive state (module docstring)."""
    from ....champions_m4 import context
    t = context(cfg).t
    sc = cfg.gold_scale
    lane, m4, k = item["start"]
    kind, label = item["kind"], item["label"]
    out = {"orig": (lane, m4, k)}
    a, b = m4.a, m4.b
    if kind == "natural+lead":
        d = float(LABEL_RE["natural+lead"].search(label).group(1))
        out["eq"] = (lane, m4._replace(a=a._replace(items=a.items - int(round(d * sc)))), k)
    elif kind in ("zone", "freeze"):
        if kind == "zone":
            d = float(LABEL_RE["zone"].search(label).group(4))
            eq = m4._replace(a=a._replace(items=b.items))
        else:
            who, d = LABEL_RE["freeze"].search(label).group(3), float(LABEL_RE["freeze"].search(label).group(4))
            base = min(a.items, b.items)
            eq = m4._replace(a=a._replace(items=base), b=b._replace(items=base))
        if d > 0:
            out["eq"] = (lane, eq, k)
        L = int(t.level_by_xp[min(a.xp, t.xp_cap)])
        if L < t.level_cap:
            frac = a.hp / t.hp_max[L]
            out["lvl"] = (lane, m4._replace(a=a._replace(xp=int(t.xp_cum[L + 1]),
                                                         hp=max(1, int(round(frac * t.hp_max[L + 1]))))), k)
        if not b.absent:
            out["hpB"] = (lane, m4._replace(b=b._replace(hp=max(1, int(round(0.6 * b.hp))))), k)
        # a large investment gap on B's archived investment: M(S_A) / M(S_B) = m, m log-uniform in [1.3, 1.9] per state
        from ...m45b import states as S
        m = float(np.exp(np.random.default_rng([C.SAMPLE_ROOT, 1, int(item["group"])]).uniform(np.log(1.3),
                                                                                                  np.log(1.9))))
        s_a = S.invest_for(cfg, m * S.multiplier(cfg, b.items / sc))
        if s_a is not None:
            out["big"] = (lane, m4._replace(a=a._replace(items=int(round(s_a * sc)))), k)
    return out


def constructed_job(job: dict) -> tuple:
    path = job["path"]
    if os.path.exists(path):
        return job["key"], path
    cfg = C.setup(C.ARM)
    net = C.net_of(job["spec"])
    at = range(0, C.LABEL_LAST + 1, C.LABEL_EVERY)
    turns = C.LABEL_LAST + C.H_LABEL
    parts = []
    for item in job["items"]:
        start_all = variants(cfg, {**item, "start": C.state_from_list(item["start"])})
        for vi, (vname, start) in enumerate(start_all.items()):
            for r in range(C.K_CON):
                rec = Recorder(cfg, at)
                rew = play(cfg, net, start, C.CON_ROOT * 1000 + 10 * item["group"] + r, turns, rec)
                a = rec.arrays(rew)
                n = len(a["t"])
                a["group"] = np.full(n, item["group"], np.int32)
                a["variant"] = np.full(n, VARIANTS.index(vname), np.int32)
                a["kind"] = np.full(n, tuple(C.ARCHIVE_SAMPLE).index(item["kind"]), np.int32)
                a["rollout"] = np.full(n, r, np.int32)
                parts.append(a)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return job["key"], path


def constructed_jobs(chunk: int = 5) -> List[dict]:
    sample = archive_sample()
    ser = [{"group": d["group"], "kind": d["kind"], "label": d["label"], "archive_index": d["archive_index"],
            "start": C.state_to_list(d["start"])} for d in sample]
    js = []
    for spec in C.POLICIES:
        for b in range(0, len(ser), chunk):
            js.append({"key": f"{spec}|con{b}", "spec": spec, "items": ser[b:b + chunk],
                       "path": os.path.join(C.CKPT, "labels", _tag(spec), f"constructed_{b:03d}.npz")})
    return js


# ---------------------------------------------------------------------------------------------- the M4.6d states
def _m46d_cells(cfg):
    from ..m46d.critic import _cell_states
    return _cell_states(cfg)


def m46d_set(spec: str) -> Dict[str, np.ndarray]:
    """The M4.6d calibration states of one C1 policy with their K 32 labels (G_300; natural also G_500), the
    critic values recorded there, observations of both seats and state features."""
    cfg = C.setup(C.ARM)
    from ....diagnostics.branch import Snapshot, observation
    tag = spec.split(":")[-1]
    cells = {(fam, i): (st, lab) for fam, i, st, lab in _m46d_cells(cfg)}
    nat = C.load_gz(os.path.join(C.D_CKPT, "critic_natural_states.json.gz"))[f"C1|{spec}"]
    root = os.path.join(C.D_CKPT, "critic", "calib", "C1", f"s{tag}")
    sf = StateFeatures(cfg)
    rows = {k: [] for k in ("obs", "V", "G300", "G300_se", "G500", "fam", "idx", "label") + tuple(f"f_{k}" for k in FEAT_KEYS)}
    for fn in sorted(os.listdir(root)):
        rec = C.load_gz(os.path.join(root, fn))
        fam, i = rec["family"], int(rec["idx"])
        if fam.startswith("natural"):
            start = C.state_from_list(nat[fam.split("_")[1]][i]["start"])
        else:
            start = cells[(fam, i)][0]
        lane, m4, k = start
        snap = Snapshot(lane, m4, k, (cfg.skill_a, cfg.skill_b))
        G = np.asarray(rec["G"], float)
        j300 = rec["cps"].index(300)
        rows["obs"].append(np.stack([observation(cfg, snap, s).vector for s in (0, 1)]))
        rows["V"].append((rec["VA"], rec["VB"]))
        rows["G300"].append(G[:, j300].mean())
        rows["G300_se"].append(G[:, j300].std(ddof=1) / np.sqrt(len(G)))
        rows["G500"].append(G[:, rec["cps"].index(500)].mean() if 500 in rec["cps"] else np.nan)
        rows["fam"].append(fam)
        rows["idx"].append(i)
        rows["label"].append(rec["label"])
        f = sf.side(lane, m4, 0)
        for kk in FEAT_KEYS:
            rows[f"f_{kk}"].append(f[kk])
    return {k: np.asarray(v) for k, v in rows.items()}


# ---------------------------------------------------------------------------------------------- the tail check
TAIL_POLICIES = (0, 2, 4)
TAIL_H, TAIL_K = 800, 32
TAIL_CPS = (50, 100, 150, 200, 300, 400, 500, 600, 800)


def tail_cells(cfg) -> List[tuple]:
    """(name, start): freeze L5 equal / edge / plaus / x1.76 and L8 plaus, the neutral lane L8 x1.76, and the M4.6a
    gap cells (neutral, 6, x1.5) and (neutral, 10, x1.76)."""
    from ..m46d import cells as CL
    from ..m46a.states import zoning_gap_state
    base = C.arm_cfg("baseline")
    out = []
    for L, cond, hpb in ((5, "equal", 1.0), (5, "edge", 1.0), (5, "edge", 0.6), (5, "x1.76", 1.0), (8, "edge", 0.6)):
        nm = {(1.0, "equal"): "equal", (1.0, "edge"): "edge", (0.6, "edge"): "plaus", (1.0, "x1.76"): "x1.76"}[(hpb, cond)]
        out.append((f"freeze12 L{L} {nm}", CL.build(cfg, "freeze", cond, L, hpb)))
    out.append(("neutral L8 x1.76", CL.build(cfg, "neutral", "x1.76", 8, 1.0)))
    out.append(("gap neutral L6 x1.5", zoning_gap_state(base, "neutral", 6, "x1.5")))
    out.append(("gap neutral L10 x1.76", zoning_gap_state(base, "neutral", 10, "x1.76")))
    return out


def tail_job(job: dict) -> tuple:
    path = job["path"]
    if os.path.exists(path):
        return job["key"], path
    cfg = C.setup(C.ARM)
    net = C.net_of(job["spec"])
    start = C.state_from_list(job["start"])
    G = np.zeros((TAIL_K, len(TAIL_CPS)))
    for r in range(TAIL_K):
        rew = play(cfg, net, start, C.TAIL_ROOT * 1000 + r, TAIL_H)
        G[r] = [C.discounted(rew, 0, h) for h in TAIL_CPS]
    C.save_gz(path, {"spec": job["spec"], "name": job["name"], "cps": list(TAIL_CPS), "G": G.tolist()})
    return job["key"], path


def tail_jobs() -> List[dict]:
    cfg = C.arm_cfg(C.ARM)
    js = []
    for s in TAIL_POLICIES:
        spec = C.general(C.ARM, s)
        for j, (name, st) in enumerate(tail_cells(cfg)):
            js.append({"key": f"{spec}|{name}", "spec": spec, "name": name, "start": C.state_to_list(st),
                       "path": os.path.join(C.CKPT, "tail", f"s{s}", f"cell_{j}.json.gz")})
    return js


# ---------------------------------------------------------------------------------------------- driver
def run(workers: int, part: Optional[str] = None) -> None:
    parts = (part or "natural,constructed,tail").split(",")
    for p in parts:
        t0 = time.perf_counter()
        if p == "natural":
            js = natural_jobs()
            C.run_jobs(natural_job, js, workers, "natural labels")
        elif p == "constructed":
            js = constructed_jobs()
            C.run_jobs(constructed_job, js, workers, "constructed labels")
        elif p == "tail":
            js = tail_jobs()
            C.run_jobs(tail_job, js, workers, "tail check")
        else:
            raise SystemExit(f"unknown labels part {p!r}")
        print(f"  [{p}] {len(js)} jobs in {time.perf_counter() - t0:.0f} s", flush=True)


__all__ = ["SelfPlay", "play", "Recorder", "natural_job", "natural_jobs", "constructed_job", "constructed_jobs",
           "archive_sample", "variants", "m46d_set", "tail_cells", "tail_job", "tail_jobs", "run", "FEAT_KEYS"]
