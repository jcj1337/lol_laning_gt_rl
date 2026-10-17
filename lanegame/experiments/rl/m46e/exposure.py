"""Q2: how much training exposure the problematic state families received.

DIRECT measurements (no simulation):
    starts       the M4.6c training runs' own start counts (final.pt metadata: games started by archive label,
                 joint turns of finished games by kind and label)
    archive      the composition of the training archive itself: every archived start state in both orientations
                 (the collector mirrors half of its draws), classified into the families below at the start state,
                 under the C1 and the baseline investment curve

RECONSTRUCTED occupancy (an ESTIMATE from frozen checkpoints; final-policy rollouts do not reproduce historical
training): M4.6c trained with 100% current-policy self-play, so a frozen snapshot playing itself from the training
start distribution (an archive start with mirroring, 150 turns, or s0, 512 turns) reproduces the training state
distribution of that moment, up to the policy's drift within the snapshot interval. Archive and s0 turns are
weighted to the measured archive transition share (0.25). Snapshots: ext/C1:s at about 1.0M, 4.1M and the final
8.192M agent transitions (s = 0, 1, 2), and the baseline finals ext/baseline:s (reference).

Families (classified on the start-of-turn state, either seat as the focal side; each joint turn in a family trains
both roles once, because both seats are the shared network's samples):
    plaus_early        both present, the lower level <= 9, the stronger side's duel ratio in [1.10, 1.50)
    plaus_early_125    the same with duel in [1.25, 1.50) (about the natural 90th percentile and above)
    large              both present, duel >= 2.0;   extreme: duel >= 3.0 (M4.6d's x1.76 investment is duel x3.1)
    fav_freeze         M4.6d's favourable freeze (freezer present, HP >= 60%, duel >= 1.10, held wave >= 9 HP),
                       freezer level <= 9
    fav_freeze_dmg     ... and the breaker present at <= 70% HP, freezer level <= 8 (the confirmed profitable case)
    weak_window        the M4.6d Safe-versus-Yield family: a weak side present at 25-55% HP, the opponent present,
                       duel >= 1.2 over it, able to catch it without a dive and within one cell
    weak_window_broad  weak side at <= 60% HP with Recall legal, duel >= 1.1 against it, catchable
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import common as C
from .features import StateFeatures

ARCH_GAMES, S0_GAMES = 192, 48
ARCH_TURNS, S0_TURNS = 150, 512
ARCHIVE_SHARE = 0.25
SNAP_TARGETS = (1_024_000, 4_096_000)
EXPO_SEEDS = (0, 1, 2)
COLS = ("L", "hp", "present", "duel", "held", "held_hp", "catch_o", "dist", "recall")
FAMILIES = ("plaus_early", "plaus_early_125", "large", "extreme", "fav_freeze", "fav_freeze_dmg", "weak_window",
            "weak_window_broad", "any_freeze")
CHECKPOINTS_T = (0, 1, 2, 5, 10, 20, 50, 100, 149)


# ---------------------------------------------------------------------------------------------- classification
def families(x: np.ndarray) -> Dict[str, np.ndarray]:
    """x: (n, 2, len(COLS)) per-seat features of joint start-of-turn states -> boolean masks per family."""
    g = {c: x[:, :, i] for i, c in enumerate(COLS)}
    pres = (g["present"] > 0).all(1)
    duel_a = g["duel"][:, 0]
    strong = np.maximum(duel_a, 1.0 / duel_a)
    lo = g["L"].min(1)
    out = {"plaus_early": pres & (lo <= 9) & (strong >= 1.10) & (strong < 1.5),
           "plaus_early_125": pres & (lo <= 9) & (strong >= 1.25) & (strong < 1.5),
           "large": pres & (strong >= 2.0), "extreme": pres & (strong >= 3.0)}
    ff = np.zeros(len(x), bool)
    ffd = np.zeros(len(x), bool)
    ww = np.zeros(len(x), bool)
    wwb = np.zeros(len(x), bool)
    anyf = np.zeros(len(x), bool)
    for f in (0, 1):
        b = 1 - f
        held = g["held"][:, f] > 0
        anyf |= held
        fav = (held & (g["present"][:, f] > 0) & (g["hp"][:, f] >= 0.6) & (g["duel"][:, f] >= 1.10)
               & (g["held_hp"][:, f] >= 9))
        ff |= fav & (g["L"][:, f] <= 9)
        ffd |= fav & (g["L"][:, f] <= 8) & (g["present"][:, b] > 0) & (g["hp"][:, b] <= 0.7)
    for w in (0, 1):
        o = 1 - w
        opp_strong = g["duel"][:, o]
        both = (g["present"][:, w] > 0) & (g["present"][:, o] > 0)
        caught = g["catch_o"][:, w] > 0
        ww |= (both & (g["hp"][:, w] >= 0.25) & (g["hp"][:, w] <= 0.55) & (opp_strong >= 1.2) & caught
               & (g["dist"][:, w] >= 0) & (g["dist"][:, w] <= 1))
        wwb |= both & (g["hp"][:, w] <= 0.6) & (g["recall"][:, w] > 0) & (opp_strong >= 1.1) & caught
    out.update({"fav_freeze": ff, "fav_freeze_dmg": ffd, "weak_window": ww, "weak_window_broad": wwb,
                "any_freeze": anyf})
    return out


def seat_features(sf: StateFeatures, lane, m4, obs) -> np.ndarray:
    from ....config import B_RECALL, W_HOLD, champ_act
    rc = champ_act(W_HOLD, B_RECALL)
    rows = []
    for s in (0, 1):
        f = sf.side(lane, m4, s)
        rows.append([f["L"], f["hp"], f["present"], f["duel"], f["held"], f["held_hp"], f["catch_o"], f["dist"],
                     float(rc in obs[s].legal)])
    return np.asarray(rows)


# ---------------------------------------------------------------------------------------------- direct
def direct_starts() -> dict:
    import torch
    from ..m46c import common as C46
    out = {}
    for arm in ("C1", "baseline"):
        for s in C.SEEDS:
            ck = torch.load(C46.policy_path(f"ext/{arm}:{s}"), weights_only=False)
            st = ck["metadata"]["starts"]
            out[f"{arm}:{s}"] = {k: st[k] for k in ("games_started", "archive_by_label", "joint_steps_finished",
                                                     "joint_steps_finished_by_label",
                                                     "archive_transition_share_finished")}
    return out


def direct_archive() -> dict:
    """Every archive state in both orientations, classified at the start state under each arm's curve."""
    from ....env.starts import load_archive
    from ..m46c.archive import archive_path
    from ....diagnostics.branch import Snapshot, observation
    out = {}
    for arm in ("C1", "baseline"):
        cfg = C.setup(arm)
        sf = StateFeatures(cfg)
        sts = load_archive(archive_path(), cfg)
        rows, labels = [], []
        for st in sts:
            snap = Snapshot(st.lane, st.champ, st.phase, (cfg.skill_a, cfg.skill_b))
            obs = [observation(cfg, snap, s) for s in (0, 1)]
            rows.append(seat_features(sf, st.lane, st.champ, obs))
            labels.append(st.label.split(" ")[0])
        x = np.asarray(rows)
        lab = np.asarray(labels)
        fam = families(x)
        strong = np.maximum(x[:, 0, 3], 1 / x[:, 0, 3])
        rec = {"states": int(len(x)), "by_label": {}}
        for lb in sorted(set(labels)):
            m = lab == lb
            rec["by_label"][lb] = {"states": int(m.sum()), **{f: float(fam[f][m].mean()) for f in FAMILIES},
                                   "duel_q": [float(q) for q in np.quantile(strong[m], (0.1, 0.5, 0.9, 0.99))]}
        rec["all"] = {f: float(fam[f].mean()) for f in FAMILIES}
        rec["all"]["duel_q"] = [float(q) for q in np.quantile(strong, (0.1, 0.5, 0.9, 0.99))]
        out[arm] = rec
    return out


# ---------------------------------------------------------------------------------------------- reconstructed
def snapshot_spec(arm: str, s: int, target: Optional[int]) -> str:
    if target is None:
        return f"ext/{arm}:{s}"
    from ..m46c import common as C46
    d = os.path.join(C46.run_dir(arm, s, "ext"), "snapshots")
    ns = sorted(int(f[5:14]) for f in os.listdir(d) if f.startswith("snap_"))
    n = min(ns, key=lambda v: abs(v - target))
    return f"ext/{arm}:{s}@{n}"


def expo_job(job: dict) -> tuple:
    path = job["path"]
    if os.path.exists(path):
        return job["key"], path
    from ....env.starts import load_archive
    from ...m45b import states as S
    from ..m46c.archive import archive_path
    from . import labels as LB
    arm = job["arm"]
    cfg = C.setup(arm)
    net = C.net_of(job["spec"])
    sf = StateFeatures(cfg)
    arch = load_archive(archive_path(), cfg) if job["kind"] == "archive" else None
    games = []
    for g in job["games"]:
        rows = []

        def rec(i, lane, m4, obs, vals):
            rows.append(seat_features(sf, lane, m4, obs))

        sd = C.EXPO_ROOT * 1000 + (0 if job["kind"] == "archive" else 500) + g
        if job["kind"] == "archive":
            rng = np.random.default_rng([C.EXPO_ROOT, g])
            ai = int(rng.integers(len(arch)))
            st = arch[ai]
            if rng.random() < 0.5:
                st = st.mirror()
            LB.play(cfg, net, (st.lane, st.champ, st.phase), sd, ARCH_TURNS, rec)
            lab = st.label.split(" ")[0]
        else:
            LB.play(cfg, net, S.s0(cfg), sd, S0_TURNS, rec)
            lab = "s0"
        games.append((lab, np.asarray(rows, np.float32)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, labels=np.asarray([l for l, _ in games]),
                        **{f"g{j}": x for j, (_, x) in enumerate(games)})
    return job["key"], path


def expo_jobs() -> List[dict]:
    specs = []
    for s in EXPO_SEEDS:
        for tgt in SNAP_TARGETS + (None,):
            specs.append(("C1", snapshot_spec("C1", s, tgt)))
        specs.append(("baseline", snapshot_spec("baseline", s, None)))
    js = []
    for arm, spec in specs:
        tag = spec.replace("/", "_").replace(":", "_").replace("@", "_")
        for b in range(0, ARCH_GAMES, 16):
            js.append({"key": f"{spec}|a{b}", "arm": arm, "spec": spec, "kind": "archive",
                       "games": list(range(b, b + 16)), "path": os.path.join(C.CKPT, "exposure", tag, f"arch_{b}.npz")})
        for b in range(0, S0_GAMES, 4):
            js.append({"key": f"{spec}|s{b}", "arm": arm, "spec": spec, "kind": "s0",
                       "games": list(range(b, b + 4)), "path": os.path.join(C.CKPT, "exposure", tag, f"s0_{b}.npz")})
    return js


def load_spec(spec: str):
    """(archive game arrays with labels, s0 game arrays) of one reconstructed policy."""
    tag = spec.replace("/", "_").replace(":", "_").replace("@", "_")
    d = os.path.join(C.CKPT, "exposure", tag)
    arch, s0 = [], []
    for fn in sorted(os.listdir(d)):
        z = np.load(os.path.join(d, fn))
        labs = list(z["labels"])
        for j, lb in enumerate(labs):
            (s0 if fn.startswith("s0") else arch).append((str(lb), z[f"g{j}"]))
    return arch, s0


def occupancy(spec: str) -> dict:
    arch, s0 = load_spec(spec)
    xa = np.concatenate([x for _, x in arch])
    xs = np.concatenate([x for _, x in s0])
    fa, fs = families(xa), families(xs)
    out = {"archive_turns": int(len(xa)), "s0_turns": int(len(xs)), "families": {}}
    for f in FAMILIES:
        pa, ps = float(fa[f].mean()), float(fs[f].mean())
        occ = ARCHIVE_SHARE * pa + (1 - ARCHIVE_SHARE) * ps
        out["families"][f] = {"archive": pa, "s0": ps, "training": occ, "from_archive_share": (
            ARCHIVE_SHARE * pa / occ if occ > 0 else float("nan")),
            "joint_turns_per_8M_run": occ * 4_096_000,
            "events_s0": int(fs[f].sum()), "events_archive": int(fa[f].sum())}
    # decay after an archive start, by label: share of games in the family at turn t
    dec = {}
    for lb in sorted(set(l for l, _ in arch)):
        xs_ = [x for l, x in arch if l == lb]
        T = min(len(x) for x in xs_)
        stack = np.stack([x[:T] for x in xs_])                     # (games, T, 2, F)
        rec = {}
        for f in ("plaus_early", "large", "extreme", "fav_freeze", "any_freeze", "weak_window_broad"):
            m = np.stack([families(g)[f] for g in stack])            # (games, T)
            rec[f] = [float(m[:, t].mean()) for t in CHECKPOINTS_T if t < T]
        duel = np.abs(np.log(stack[:, :, 0, 3]))
        rec["mean_abs_log_duel"] = [float(duel[:, t].mean()) for t in CHECKPOINTS_T if t < T]
        rec["games"] = len(xs_)
        dec[lb] = rec
    out["decay"] = dec
    # s0 occupancy by game phase (turn bands)
    bands = {}
    for lo, hi in ((0, 100), (100, 250), (250, 512)):
        xx = np.concatenate([x[lo:hi] for _, x in s0])
        ff = families(xx)
        bands[f"{lo}-{hi}"] = {f: float(ff[f].mean()) for f in FAMILIES}
    out["s0_bands"] = bands
    return out


def neighbourhoods(spec: str) -> dict:
    """Training occupancy (reconstructed) of bins around the M4.6d diagnostic states: same kind, level +-1, duel
    within x1.15, the weaker side's HP within +-0.15 (freeze cells: the strong side holds a freeze; Recall windows:
    the weak side within one cell and catchable)."""
    from ..m46d import cells as CL
    from . import labels as LB
    cfg = C.arm_cfg("C1")
    sf = StateFeatures(cfg)
    arch, s0 = load_spec(spec)
    xa = np.concatenate([x for _, x in arch])
    xs = np.concatenate([x for _, x in s0])
    targets = []
    for _, _, st, lab in LB._m46d_cells(cfg):
        lane, m4, k = st
        a = sf.side(lane, m4, 0)
        targets.append((lab, "freeze" if lab.startswith("freeze") else "lane", a["L"], a["duel"], a["hp_o"]))
    for cond in ("edge", "x1.76"):
        for L in (5, 8):
            for hpb in (0.5, 0.35):
                from ..m46d.critic import recall_window
                lane, m4, k = recall_window(cfg, cond, L, hpb)
                a = sf.side(lane, m4, 0)
                targets.append((f"recall window L{L} {cond} hpB{hpb}", "window", a["L"], a["duel"], hpb))

    def share(x, kind, L, duel, hpo):
        g = {c: x[:, :, i] for i, c in enumerate(COLS)}
        m = np.zeros(len(x), bool)
        for s in (0, 1):
            o = 1 - s
            ok = ((g["present"][:, s] > 0) & (g["present"][:, o] > 0) & (np.abs(g["L"][:, s] - L) <= 1)
                  & (g["duel"][:, s] >= duel / 1.15) & (g["duel"][:, s] <= duel * 1.15)
                  & (np.abs(g["hp"][:, o] - hpo) <= 0.15))
            if kind == "freeze":
                ok &= g["held"][:, s] > 0
            if kind == "window":
                ok &= (g["catch_o"][:, o] > 0) & (g["dist"][:, o] >= 0) & (g["dist"][:, o] <= 1)
            m |= ok
        return float(m.mean())

    out = {}
    for lab, kind, L, duel, hpo in targets:
        pa, ps = share(xa, kind, L, duel, hpo), share(xs, kind, L, duel, hpo)
        occ = ARCHIVE_SHARE * pa + (1 - ARCHIVE_SHARE) * ps
        out[lab] = {"kind": kind, "L": L, "duel": duel, "hp_weak": hpo, "archive": pa, "s0": ps, "training": occ,
                    "joint_turns_per_8M_run": occ * 4_096_000}
    return out


def run(workers: int, part: Optional[str] = None) -> None:
    t0 = time.perf_counter()
    parts = (part or "simulate,summarise").split(",")
    if "simulate" in parts:
        C.run_jobs(expo_job, expo_jobs(), workers, "exposure rollouts")
    if "summarise" in parts:
        specs = sorted({j["spec"] for j in expo_jobs()})
        res = {"meta": C.meta(stage="exposure"), "direct_starts": direct_starts(), "direct_archive": direct_archive(),
               "occupancy": {sp: occupancy(sp) for sp in specs},
               "neighbourhoods": {sp: neighbourhoods(sp) for sp in specs if "baseline" not in sp},
               "elapsed_s": time.perf_counter() - t0}
        C.save_json("exposure", res)
    print(f"  exposure done in {time.perf_counter() - t0:.0f} s", flush=True)


__all__ = ["families", "direct_starts", "direct_archive", "expo_job", "expo_jobs", "occupancy", "neighbourhoods", "run",
           "FAMILIES", "COLS"]
