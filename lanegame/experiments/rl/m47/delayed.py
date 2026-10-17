"""Stage 1: how much of the benefit of early advantages and early actions arrives after the lane phase (new paired
rollouts with full reward streams, `streams.py`; the matched M4.6c 8.192M policies ext/<env>:s in their own
environments).

Families (seat A is the focal side; every rollout is T = 300 turns, no critic bootstrap):
    cells    CONSTRUCTED lead cells (the M4.6c builder, k 2): both champions at level L with the natural investment
             S(L); seat A or B given +4 or +8 extra invested resources; neutral / A pushing / B pushing lanes. Stages
             L4, L8 (= M4.6c's "early" cell, 9.0 invested) and L11. The premium of a lead d is
             0.5 * (G[A+d] - G[B+d]) on common random numbers (the M4.6c definition): the leader's extra discounted
             gold over the equal state, both orientations.
    natural  NATURAL early states: the state after t turns of the policy's own self-play from s0 (t = 8, 20, 35, 50,
             65, 80), then one-turn and short-plan alternatives for seat A against the policy: Shove x1, Shove x4,
             Trade x1, Safe x1, reset (walk back and Recall now), and two state variants: +4 invested resources and
             -0.3 bars of HP. Benefit = candidate - policy on common random numbers.
    freeze   the M4.6d early freeze cells (freezer A holds a 12-HP wave, breaker B walked up): levels 4 and 8, both
             healthy and equal, or the freezer with M4.6d's plausible edge against a breaker at 60% HP. Freezer plans
             guard_sel, passive, push against the breaker's policy; breaker plans commit, concede, reset against the
             freezer's policy (seat B's benefit is minus seat A's reward).

Readings per benefit: the discounted sum under every candidate horizon (common.HORIZONS), and its exact split at the
phase end under gamma 0.99: at the turn the LOWER level of the baseline rollout reaches 14 (level-triggered; the same
cut for both streams of a pair), and at the natural elapsed-turn equivalent (common window). Seeds are the units.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from ....config import B_HOLD, W_SHOVE, champ_act
from . import common as C
from . import streams as SM

T_ROLL = 300
STAGES = {"L4": 4, "L8": 8, "L11": 11}
M46C_EARLY_S = 9.0                         # M4.6c's early cell (level 8)
LEADS = ("equal", "A+4", "B+4", "A+8", "B+8")
SITS = ("neutral", "a_push", "b_push")
K_CELL, K_NAT, K_FRZ = 32, 12, 24
NAT_BUCKETS = (8, 20, 35, 50, 65, 80)
NAT_GAMES = 24
SHOVE_HOLD = champ_act(W_SHOVE, B_HOLD)
NAT_COMBOS = {"policy": ["orig", ("policy",), ("policy",)],
              "shove1": ["orig", ("m46a", ("fixed", SHOVE_HOLD, 1)), ("policy",)],
              "shove4": ["orig", ("m46a", ("fixed", SHOVE_HOLD, 4)), ("policy",)],
              "trade1": ["orig", ("m46a", ("beh", "trade", 1, 0.0)), ("policy",)],
              "safe1": ["orig", ("m46a", ("beh", "safe", 1, 0.0)), ("policy",)],
              "reset": ["orig", ("m46a", ("reset", 10)), ("policy",)],
              "inv+4": ["inv+4", ("policy",), ("policy",)],
              "hp-0.3": ["hp-0.3", ("policy",), ("policy",)]}
FRZ_CELLS = (("L4 equal", 4, "equal", 1.0), ("L4 edge, breaker 60%", 4, "edge", 0.6),
             ("L8 equal", 8, "equal", 1.0), ("L8 edge, breaker 60%", 8, "edge", 0.6))
FRZ_COMBOS = {"policy": ["orig", ("policy",), ("policy",)],
              "guard_sel": ["orig", ("m46d", "guard_sel"), ("policy",)],
              "passive": ["orig", ("m46d", "passive"), ("policy",)],
              "push": ["orig", ("m46d", "push"), ("policy",)],
              "commit": ["orig", ("policy",), ("m46d", "commit")],
              "concede": ["orig", ("policy",), ("m46d", "concede")],
              "reset_b": ["orig", ("policy",), ("m46d", "reset")]}
BREAKER_COMBOS = ("commit", "concede", "reset_b")


def raw(family: str, env: str, s: int, idx) -> str:
    return os.path.join(C.CKPT, "delayed", family, env, f"s{s}", f"{idx}.npz")


# ---------------------------------------------------------------------------------------------- natural investment
def natural_invest(level: int) -> float:
    """Mean invested resources when a seat first reaches `level` in the baseline records (all seeds)."""
    from .timeline import first_turn, load_records, take
    v = []
    for s in C.SEEDS:
        d = load_records("baseline", s)
        v.append(take(d["S"], first_turn(d["L"], level)).mean())
    return round(float(np.mean(v)), 2)


def stage_invest() -> Dict[str, float]:
    out = {}
    for st, L in STAGES.items():
        out[st] = M46C_EARLY_S if st == "L8" else natural_invest(L)
    return out


def lead_cell(cfg, L: int, S0: float, sit: str, cell: str):
    from ...m45b import states as S
    d = 0.0 if cell == "equal" else float(cell[2:])
    s_a, s_b = (S0 + d, S0) if cell.startswith("A") else ((S0, S0 + d) if cell.startswith("B") else (S0, S0))
    lane, pa, pb, _ = S.ZONING_SITUATIONS[sit]
    return lane, S.champs(cfg, (L, L), (s_a, s_b), (pa, pb)), 2


# ---------------------------------------------------------------------------------------------- jobs
def cell_jobs(envs=C.ENVS, seeds=C.SEEDS) -> List[dict]:
    inv = stage_invest()
    js = []
    for env in envs:
        cfg = C.env_cfg(env)
        for s in seeds:
            i = 0
            for st, L in STAGES.items():
                for sit in SITS:
                    for cell in LEADS:
                        start = lead_cell(cfg, L, inv[st], sit, cell)
                        js.append({"key": f"cells|{env}|s{s}|{i}", "path": raw("cells", env, s, i), "env": env,
                                   "spec": C.general(env, s), "start": C.state_to_list(start),
                                   "combos": {"self": ["orig", ("policy",), ("policy",)]}, "K": K_CELL, "T": T_ROLL,
                                   "root": C.DELAY_CELL_ROOT, "label": f"{st} {sit} {cell}"})
                        i += 1
    return js


def natural_jobs(envs=C.ENVS, seeds=C.SEEDS) -> List[dict]:
    js = []
    for env in envs:
        for s in seeds:
            for g in range(NAT_GAMES):
                t = NAT_BUCKETS[g % len(NAT_BUCKETS)]
                js.append({"key": f"natural|{env}|s{s}|{g}", "path": raw("natural", env, s, g), "env": env,
                           "spec": C.general(env, s), "natural": (C.DELAY_NAT_ROOT * 1000 + 100 * s + g, t),
                           "combos": NAT_COMBOS, "K": K_NAT, "T": T_ROLL, "root": C.DELAY_NAT_ROOT + 1,
                           "label": f"natural t{t} g{g}"})
    return js


def freeze_jobs(envs=C.ENVS, seeds=C.SEEDS) -> List[dict]:
    from ..m46d import cells as CL
    js = []
    for env in envs:
        cfg = C.env_cfg(env)
        for s in seeds:
            for i, (lab, L, cond, hpb) in enumerate(FRZ_CELLS):
                start = CL.build(cfg, "freeze", cond, L, hp_b=hpb, held=12)
                js.append({"key": f"freeze|{env}|s{s}|{i}", "path": raw("freeze", env, s, i), "env": env,
                           "spec": C.general(env, s), "start": C.state_to_list(start), "combos": FRZ_COMBOS,
                           "K": K_FRZ, "T": T_ROLL, "root": C.DELAY_FRZ_ROOT, "label": lab})
    return js


FAMILIES = {"cells": cell_jobs, "natural": natural_jobs, "freeze": freeze_jobs}


def run(workers: int, part: str = None) -> None:
    for fam, fn in FAMILIES.items():
        if part and fam != part:
            continue
        SM.run_sorted(SM.stream_job, fn(), workers, f"M4.7 delayed {fam}")


# ---------------------------------------------------------------------------------------------- analysis
def _readings(d: np.ndarray, cut_level: np.ndarray, cut_elapsed: int) -> dict:
    """d [K, T] paired per-turn differences; returns per-rollout-averaged readings."""
    out = {f"H{h}": float(SM.discounted(d, g).mean()) for h, g in C.HORIZONS.items()}
    pre, post = SM.split_at(d, cut_level, C.GAMMA)
    out["pre_level"], out["post_level"] = float(pre.mean()), float(post.mean())
    pre, post = SM.split_at(d, cut_elapsed, C.GAMMA)
    out["pre_elapsed"], out["post_elapsed"] = float(pre.mean()), float(post.mean())
    for lo, hi in ((0, 20), (0, 50), (0, 100), (0, 300)):
        out[f"undisc_{lo}_{hi}"] = float(SM.window_sum(d, lo, hi).mean())
    for h in (20, 50, 100):                              # the published G_h statistics (gamma 0.99, truncated at h)
        out[f"G{h}_trunc"] = float(SM.discounted(d[:, :h], C.GAMMA).mean())
    return out


def _natural_turn(level: int, tl: dict) -> float:
    b = tl["aggregate"]["by_env"]["baseline"] if tl else None
    key = f"timeline/{level}/pooled/mean"
    if b and key in b:
        return b[key]["mean"]
    from .timeline import first_turn, load_records
    return float(np.mean([first_turn(load_records("baseline", s)["L"], level).mean() for s in C.SEEDS]))


def _path(j: dict, path_fn) -> str:
    """A job's Stage 1 record, or (path_fn(index)) the record of the same job for another policy (Stage 2)."""
    return j["path"] if path_fn is None else path_fn(os.path.splitext(os.path.basename(j["path"]))[0])


def analyse_cells(env: str, s: int, tl: dict, path_fn=None) -> dict:
    """Premium readings per stage and lead size (situations pooled), seed s."""
    js = [j for j in cell_jobs((env,), (s,))]
    recs = {j["label"]: SM.load(_path(j, path_fn)) for j in js}
    out = {}
    for st, L in STAGES.items():
        elapsed = int(round(C.LANE_END - _natural_turn(L, tl)))
        for lead in (4, 8):
            rows = []
            for sit in SITS:
                a, b = recs[f"{st} {sit} A+{lead}"], recs[f"{st} {sit} B+{lead}"]
                eq = recs[f"{st} {sit} equal"]
                d = 0.5 * (a["r"][0].astype(float) - b["r"][0].astype(float))
                rows.append(_readings(d, SM.level_cut(eq["L"][0]), elapsed))
            out[f"{st} +{lead}"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            out[f"{st} +{lead}"]["elapsed_cut"] = elapsed
    return out


def _applied(d: dict, ci: int, nm: str) -> float:
    """Share of rollouts in which a natural-state candidate actually changed seat A's turn-0 decision."""
    if nm in ("inv+4", "hp-0.3"):
        return 1.0
    if nm in ("shove1", "shove4"):
        return float((d["first"][ci, :, 0] == SHOVE_HOLD).mean())
    return float((d["forced"][ci, :, 0] > 0).mean())


def analyse_natural(env: str, s: int, path_fn=None) -> dict:
    """Benefit readings per candidate and state bucket (and pooled over buckets), seed s."""
    js = natural_jobs((env,), (s,))
    acc: Dict[str, Dict[str, list]] = {}
    for j in js:
        d = SM.load(_path(j, path_fn))
        t = j["natural"][1]
        base = d["combos"].index("policy")
        cut_lv = SM.level_cut(d["L"][base])
        for ci, nm in enumerate(d["combos"]):
            if nm == "policy":
                continue
            diff = d["r"][ci].astype(float) - d["r"][base].astype(float)
            rd = _readings(diff, cut_lv, max(0, C.LANE_END - t))
            rd["applied"] = _applied(d, ci, nm)
            for key in (f"{nm}|t{t}", f"{nm}|all", f"{nm}|t<=35" if t <= 35 else f"{nm}|t>=50"):
                acc.setdefault(key, {k: [] for k in rd})
                for k, v in rd.items():
                    acc[key][k].append(v)
    return {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in acc.items()}


def analyse_freeze(env: str, s: int, tl: dict, path_fn=None) -> dict:
    js = freeze_jobs((env,), (s,))
    out = {}
    for j in js:
        d = SM.load(_path(j, path_fn))
        L = [c for c in FRZ_CELLS if c[0] == j["label"]][0][1]
        elapsed = int(round(C.LANE_END - _natural_turn(L, tl)))
        base = d["combos"].index("policy")
        cut_lv = SM.level_cut(d["L"][base])
        for ci, nm in enumerate(d["combos"]):
            if nm == "policy":
                continue
            sign = -1.0 if nm in BREAKER_COMBOS else 1.0          # the intervening seat's perspective
            diff = sign * (d["r"][ci].astype(float) - d["r"][base].astype(float))
            rd = _readings(diff, cut_lv, elapsed)
            seat = 1 if nm in BREAKER_COMBOS else 0
            rd["applied"] = float((d["forced"][ci, :, seat] > 0).mean())
            out[f"{j['label']}|{nm}"] = rd
    return out


def _rank_agreement(env: str, s: int, path_fn=None) -> dict:
    """Natural states: does the best candidate (policy included) or the candidates' order change with the horizon?"""
    from scipy.stats import kendalltau
    js = natural_jobs((env,), (s,))
    res = {h: {"best_changes": 0, "tau": [], "states": 0} for h in C.HORIZONS if h != "100"}
    acts = [n for n in NAT_COMBOS if n not in ("inv+4", "hp-0.3")]
    for j in js:
        d = SM.load(_path(j, path_fn))
        idx = [d["combos"].index(n) for n in acts]
        v100 = np.asarray([SM.discounted(d["r"][i].astype(float), C.GAMMA).mean() for i in idx])
        for h, g in C.HORIZONS.items():
            if h == "100":
                continue
            vh = np.asarray([SM.discounted(d["r"][i].astype(float), g).mean() for i in idx])
            res[h]["states"] += 1
            res[h]["best_changes"] += int(np.argmax(vh) != np.argmax(v100))
            res[h]["tau"].append(float(kendalltau(v100, vh).statistic))
    return {h: {"best_change_share": v["best_changes"] / max(v["states"], 1), "tau_mean": float(np.mean(v["tau"])),
                "states": v["states"]} for h, v in res.items()}


def analyse() -> dict:
    from .timeline import aggregate
    tl = C.load("timeline")
    fams = {"cells": {}, "natural": {}, "freeze": {}, "ranking": {}}
    for env in C.ENVS:
        for k in fams:
            fams[k][env] = {}
        for s in C.SEEDS:
            fams["cells"][env][s] = analyse_cells(env, s, tl)
            fams["natural"][env][s] = analyse_natural(env, s)
            fams["freeze"][env][s] = analyse_freeze(env, s, tl)
            fams["ranking"][env][s] = _rank_agreement(env, s)
    data = {"meta": C.meta(stage="delayed_analysis"), "per_seed": fams,
            "aggregate": {k: aggregate(v) for k, v in fams.items()}, "stage_invest": stage_invest(),
            "horizons": C.HORIZONS, "T": T_ROLL, "lane_end": C.LANE_END,
            "tail_weight_beyond_T": {h: float(g ** T_ROLL) for h, g in C.HORIZONS.items()}}
    C.save_json("delayed", data)
    return data


__all__ = ["FAMILIES", "run", "analyse", "cell_jobs", "natural_jobs", "freeze_jobs", "NAT_COMBOS", "FRZ_COMBOS",
           "STAGES", "T_ROLL"]
