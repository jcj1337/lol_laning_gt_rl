"""The evaluation families of M4.6c (pre-registration §4). Every family is resumable: each job writes its own gzipped
record under results/rl/checkpoints/m46c/eval/<family>/... and is read back when present.

Paired-rollout families use the M4.6a engine (`m46a.engine.paired_job`) with the environment arm of the job; the
networks of a job are resolved through `common.net_of` (any checkpoint, digest verified). Within a family the same
rollout seeds (ROLL_ROOT) are used in every cell, environment and policy, so cells and jobs pair on common random
numbers (the chance stream and each seat's sampling uniforms).

Families: premium, premium_mid, premium150, battery, natural_cf, freeze, specialists, natural, records, h2h.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np

from ....config import B_HOLD, B_TRADE, W_SHOVE, W_THIN, champ_act
from . import common as C
from .archive import SITS, STAGES, freeze_cell, freeze_cells, lead_cell, lead_cells

H = 50
K_PREM, K_J3, K_MID, K_150 = 128, 64, 128, 64
K_BAT, K_CF, K_FZ, K_SPEC = 32, 32, 32, 16
N_CF, N_FZ_NAT = 24, 16
EVAL = os.path.join(C.CKPT, "eval")

WEAK_CANDS = {"policy": ("policy",), "hold": ("beh", "farm", 10, 0.4), "safe": ("beh", "safe", 10, 0.4),
              "trade": ("beh", "trade", 10, 0.4), "reset": ("reset", 10)}
STRONG_CANDS = {"policy": ("policy",), "threaten": ("beh", "threaten", 10, 0.4), "hold": ("beh", "farm", 10, 0.4)}
FREEZER_CANDS = {"policy": ("policy",), "keep": ("hz", ("fixed", champ_act(W_THIN, B_HOLD), 10, "held")),
                 "guard": ("hz", ("fixed", champ_act(W_THIN, B_TRADE), 10, "held"))}
BREAKER_CANDS = {"policy": ("policy",), "break": ("fixed", champ_act(W_SHOVE, B_HOLD), 1),
                 "walk_up": ("beh", "farm", 10, 0.4), "step_back": ("beh", "safe", 10, 0.4), "reset": ("reset", 10)}


def _recall_code() -> int:
    from ..m44.actions import RECALL
    return int(RECALL)


def trailer_cf_cands():
    return {"policy": ("policy",), "hold1": ("beh", "farm", 1, 0.0), "safe1": ("beh", "safe", 1, 0.0),
            "trade1": ("beh", "trade", 1, 0.0), "recall1": ("fixed", _recall_code(), 1)}


LEADER_CF_CANDS = {"policy": ("policy",), "trade1": ("beh", "trade", 1, 0.0), "hold1": ("beh", "farm", 1, 0.0),
                   "safe1": ("beh", "safe", 1, 0.0)}
BREAKER_CF_CANDS = {"policy": ("policy",), "break1": ("fixed", champ_act(W_SHOVE, B_HOLD), 1),
                    "walk1": ("beh", "farm", 1, 0.0), "back1": ("beh", "safe", 1, 0.0)}
FREEZER_CF_CANDS = {"policy": ("policy",), "keep1": ("fixed", champ_act(W_THIN, B_HOLD), 1),
                    "guard1": ("fixed", champ_act(W_THIN, B_TRADE), 1)}
# one-turn fixed-code candidates: applied only where the code is legal (analysis.applied)
FIXED_CODES = {"break1": champ_act(W_SHOVE, B_HOLD), "keep1": champ_act(W_THIN, B_HOLD),
               "guard1": champ_act(W_THIN, B_TRADE), "recall1": _recall_code()}


# ---------------------------------------------------------------------------------------------- the paired job
def mk(family: str, env: str, tag: str, idx, start, spec: str, cands: dict, K: int, horizon: int = H,
       critic: bool = False, proto: str = "policy", label: str = "", root: int = C.ROLL_ROOT) -> dict:
    from ..m46a import engine as EN
    return {"spec": spec, "family": family, "idx": idx, "start": start, "proto": proto, "horizon": horizon,
            "roll": EN.roll_seeds(root, K), "cands": cands, "critic": critic, "tag": tag, "arm": env,
            "path": os.path.join(EVAL, family, env, tag, f"{idx}.json.gz"), "label": label}


def pjob(job: dict) -> tuple:
    """Install every network the job references (C.net_of), then run the M4.6a paired job."""
    from ..m46a import engine as EN
    key = f"{job['family']}|{job['tag']}|{job['idx']}"
    if job["path"] and os.path.exists(job["path"]):
        return key, {"path": job["path"], "cached": True}
    specs = [job["spec"]] + [c[1] for c in job["cands"].values() if c[0] == "net"]
    if job["proto"].startswith("net:"):
        specs.append(job["proto"][4:])
    C.install_nets(specs)
    _, rec = EN.paired_job(job)                          # written to job["path"]; the analysis reads it from disk
    return key, {"path": job["path"], "elapsed_s": rec["elapsed_s"], "turns": rec["turns"]}


def run_paired(js: List[dict], workers: int, label: str) -> Dict[str, dict]:
    js = sorted(js, key=lambda j: -len(j["roll"]) * len(j["cands"]) * j["horizon"])
    t0 = time.perf_counter()
    res = C.run_jobs(pjob, js, workers, label)
    print(f"  [{label}] {len(js)} jobs in {time.perf_counter() - t0:.0f} s", flush=True)
    return {k: v for k, v in res}


# ---------------------------------------------------------------------------------------------- premium family
def premium_jobs(snap: int = None, stages=tuple(STAGES), K: int = K_PREM, horizon: int = H, family: str = "premium",
                 j2: bool = True, j3: bool = True, cell_filter=None, prefix: str = "") -> List[dict]:
    js = []
    for s in C.SEEDS:
        for env in C.ARMS:
            cfg = C.arm_cfg(env)
            pol = f"{prefix}{env}:{s}" + (f"@{snap}" if snap else "")
            for i, (st, sit, c) in lead_cells():
                if st not in stages or (cell_filter and not cell_filter(st, sit, c)):
                    continue
                start = lead_cell(cfg, st, sit, c)
                lab = f"{st} {sit} {c}"
                js.append(mk(family, env, f"J1_{env}_s{s}", i, start, pol, {"self": ("policy",)}, K, horizon,
                             critic=(snap is None and horizon == H), label=lab))
                if j2:
                    js.append(mk(family, env, f"J2_{env}_s{s}", i, start, f"start:{s}",
                                 {"start": ("policy",), "new": ("net", pol)}, K, horizon, label=lab))
                if j3 and env == "C1":
                    js.append(mk(family, env, f"J3_s{s}", i, start, f"{prefix}baseline:{s}", {"self": ("policy",)}, K_J3,
                                 horizon, label=lab + " (baseline-trained, CROSS-ENVIRONMENT)"))
    return js


def p150_filter(st, sit, c) -> bool:
    return sit == "neutral" and (c in ("equal", "A+16", "B+16") or (st == "mid" and c in ("A+8", "B+8")))


# ---------------------------------------------------------------------------------------------- role swap (exploratory)
def roleswap_jobs(prefix: str = "", family: str = "roleswap") -> List[dict]:
    """EXPLORATORY (amendment of 2026-10-15): in the C1 environment, seat B = the baseline-trained policy
    (prefix)baseline:s; seat A = the same policy ("bc") or the C1-trained policy ("c1"). In A+d cells the difference
    c1 - bc is the C1-trained LEADER's conversion against a baseline-trained trailer; in B+d cells it is the C1-trained
    TRAILER's defence against a baseline-trained leader; the equal cell is general strength. Early and mid stages."""
    js = []
    cfg = C.arm_cfg("C1")
    for s in C.SEEDS:
        for i, (st, sit, c) in lead_cells():
            if st == "late":
                continue
            js.append(mk(family, "C1", f"rs_s{s}", i, lead_cell(cfg, st, sit, c), f"{prefix}baseline:{s}",
                         {"bc": ("policy",), "c1": ("net", f"{prefix}C1:{s}")}, K_PREM, label=f"{st} {sit} {c}"))
    return js


# ---------------------------------------------------------------------------------------------- battery
def battery_cells():
    out = []
    for st in STAGES:
        for sit in SITS:
            for d in (8, 16):
                out.append(("weak", st, sit, d))
                out.append(("strong", st, sit, d))
    return list(enumerate(out))


def battery_jobs(prefix: str = "", family: str = "battery") -> List[dict]:
    js = []
    for s in C.SEEDS:
        for env in C.ARMS:
            cfg = C.arm_cfg(env)
            for i, (focal, st, sit, d) in battery_cells():
                cell = f"B+{d}" if focal == "weak" else f"A+{d}"
                cands = WEAK_CANDS if focal == "weak" else STRONG_CANDS
                js.append(mk(family, env, f"bat_{env}_s{s}", i, lead_cell(cfg, st, sit, cell), f"{prefix}{env}:{s}", cands,
                             K_BAT, label=f"{focal} {st} {sit} +{d}"))
    return js


# ---------------------------------------------------------------------------------------------- natural states
def _lead_pred(role: str, min_lead: float = 4.0):
    def p(cfg, lane, m4, s):
        me, en = m4.side(s), m4.side(1 - s)
        if me.absent or en.absent:
            return False
        d = (me.items - en.items) / cfg.gold_scale
        return d >= min_lead if role == "leader" else d <= -min_lead
    return p


def _freeze_pred(role: str):
    from ..m45c.strategic import p_freeze, p_frozen
    return p_freeze if role == "freezer" else p_frozen


def states_job(args) -> tuple:
    """(policy spec, env, role, n, offset): n natural states of the policy's own self-play (role side mirrored to A)."""
    spec, env, role, n, offset = args
    p = os.path.join(EVAL, "states", env, spec.replace(":", "_"), f"{role}.json.gz")
    if os.path.exists(p):
        return (spec, env, role), C.load_gz(p)
    cfg = C.setup(env)
    from ...m45b import engine as E
    from ...m45b import states as S
    from ..m45c import horizon as HZ
    net = C.net_of(spec)
    pred = _lead_pred(role) if role in ("leader", "trailer") else _freeze_pred(role)
    s_idx = int(spec.split(":")[1].split("@")[0])
    out, games = [], 0
    for g in range(300):
        sd = C.EVAL_STATE_ROOT * 1000 + 100 * s_idx + offset + g
        pa, pb = HZ.CRNNet(net, 0, sd, 0, 300), HZ.CRNNet(net, 0, sd, 1, 300)
        out += E.collect(cfg, S.s0(cfg), pa, pb, [sd], 300, pred, per_game=2, min_turn=20, rng_seed=sd)
        games += 1
        if len(out) >= n:
            break
    d = {"states": [C.state_to_list(x["start"]) for x in out[:n]], "turns": [x["turn"] for x in out[:n]],
         "games": games, "spec": spec, "env": env, "role": role}
    C.save_gz(p, d)
    return (spec, env, role), d


ROLE_OFFSET = {"trailer": 0, "leader": 30, "freezer": 60, "breaker": 90}


def collect_states(workers: int, roles=("trailer", "leader", "freezer", "breaker"), prefix: str = "") -> dict:
    js = []
    for s in C.SEEDS:
        for env in C.ARMS:
            for role in roles:
                n = N_CF if role in ("trailer", "leader") else N_FZ_NAT
                js.append((f"{prefix}{env}:{s}", env, role, n, ROLE_OFFSET[role]))
    return dict(C.run_jobs(states_job, js, workers, "M4.6c natural states"))


def natural_cf_jobs(states: dict, prefix: str = "", family: str = "natural_cf") -> List[dict]:
    js = []
    for s in C.SEEDS:
        for env in C.ARMS:
            spec = f"{prefix}{env}:{s}"
            for role, cands in (("trailer", trailer_cf_cands()), ("leader", LEADER_CF_CANDS)):
                for i, v in enumerate(states[(spec, env, role)]["states"]):
                    js.append(mk(family, env, f"{role}_{env}_s{s}", i, C.state_from_list(v), spec, cands, K_CF,
                                 label=f"natural {role} {i}"))
    return js


# ---------------------------------------------------------------------------------------------- freeze
def freeze_jobs(states: dict) -> List[dict]:
    js = []
    for s in C.SEEDS:
        for env in C.ARMS:
            cfg = C.arm_cfg(env)
            spec = f"{env}:{s}"
            for i, (L, h, inv) in freeze_cells():
                st = freeze_cell(cfg, L, h, inv)
                js.append(mk("freeze", env, f"freezer_{env}_s{s}", i, st, spec, FREEZER_CANDS, K_FZ,
                             label=f"freezer L{L} held {h} {inv}"))
                lane, m4, k = st
                js.append(mk("freeze", env, f"breaker_{env}_s{s}", i, (lane.mirror(), m4.mirror(), k), spec,
                             BREAKER_CANDS, K_FZ, label=f"breaker L{L} held {h} {inv}"))
            for role, cands in (("freezer", FREEZER_CF_CANDS), ("breaker", BREAKER_CF_CANDS)):
                for i, v in enumerate(states[(spec, env, role)]["states"]):
                    js.append(mk("freeze", env, f"nat{role}_{env}_s{s}", i, C.state_from_list(v), spec, cands, K_FZ,
                                 label=f"natural {role} {i}"))
    return js


# ---------------------------------------------------------------------------------------------- specialists
def specialist_jobs() -> List[dict]:
    from ..m46a.e4 import GAP_CELLS
    from ..m46a.states import zoning_gap_state
    base = C.arm_cfg("baseline")
    sts = [zoning_gap_state(base, *c) for c in GAP_CELLS]          # gold values of M4.6a's (baseline) construction
    cands = {"own": ("policy",)}
    for t in (0, 2, 4):
        cands[f"g{t}"] = ("net", C.specialist_path("gap", t))
    js = []
    for s in C.SEEDS:
        for env, pol in (("baseline", f"start:{s}"), ("baseline", f"baseline:{s}"), ("C1", f"start:{s}"),
                         ("C1", f"C1:{s}")):
            for i, st in enumerate(sts):
                js.append(mk("specialists", env, f"{pol.split(':')[0]}_{env}_s{s}", i, st, pol, cands, K_SPEC,
                             label=f"gap cell {GAP_CELLS[i]}"))
    return js


# ---------------------------------------------------------------------------------------------- natural self-play
def natural_combos(prefix: str = ""):
    out = []
    for s in C.SEEDS:
        out += [(f"start:{s}", "baseline"), (f"start:{s}", "C1"), (f"{prefix}baseline:{s}", "baseline"),
                (f"{prefix}C1:{s}", "C1")]
    return out


def run_natural(workers: int, prefix: str = "") -> dict:
    from .natural import GAMES, TURNS, natural_job
    js = [(spec, env, GAMES, TURNS, int(spec.split(":")[1])) for spec, env in natural_combos(prefix)]
    res = C.run_jobs(natural_job, js, workers, "M4.6c natural self-play")
    return {f"{k[0]}|{k[1]}": v for k, v in res}


STATE_CLASSES_LEAD = {
    "all present": lambda d, s: np.ones(len(d["present"]), bool),
    "invested lead >= 8": lambda d, s: (d["invested"][:, s] - d["einvested"][:, s]) >= 8,
    "invested lead 3-8": lambda d, s: ((d["invested"][:, s] - d["einvested"][:, s]) >= 3)
    & ((d["invested"][:, s] - d["einvested"][:, s]) < 8),
    "even (|lead| < 3)": lambda d, s: np.abs(d["invested"][:, s] - d["einvested"][:, s]) < 3,
    "trailing 3-8": lambda d, s: ((d["einvested"][:, s] - d["invested"][:, s]) >= 3)
    & ((d["einvested"][:, s] - d["invested"][:, s]) < 8),
    "trailing >= 8": lambda d, s: (d["einvested"][:, s] - d["invested"][:, s]) >= 8,
    "holding a freeze": lambda d, s: d["held"][:, s] > 0,
    "frozen on": lambda d, s: d["held_on"][:, s] > 0,
    "frozen on, trailing >= 4": lambda d, s: (d["held_on"][:, s] > 0) & ((d["einvested"][:, s] - d["invested"][:, s]) >= 4),
    "frozen on, leading >= 4": lambda d, s: (d["held_on"][:, s] > 0) & ((d["invested"][:, s] - d["einvested"][:, s]) >= 4),
    "enemy crash under my tower": lambda d, s: d["en_siege"][:, s] > 0,
    "HP < 35%": lambda d, s: d["hp_frac"][:, s] < 0.35,
    "HP < 50%, trailing >= 4": lambda d, s: (d["hp_frac"][:, s] < 0.5) & ((d["einvested"][:, s] - d["invested"][:, s]) >= 4),
    "Recall legal": lambda d, s: d["recall_legal"][:, s] > 0,
    "level 11-18": lambda d, s: d["level"][:, s] >= 11,
}


def records_job(args) -> tuple:
    """(spec, env, games, turns, seed): state-conditioned action masses (self-play, the M4.4 records)."""
    spec, env, games, turns, seed = args
    p = os.path.join(EVAL, "records", env, spec.replace(":", "_") + ".json.gz")
    if os.path.exists(p):
        return (spec, env), C.load_gz(p)
    import torch
    torch.set_num_threads(1)
    from ....rl import NetAgent
    from ..m44.records import play_records
    from ..m45c import behavior as BH
    cfg = C.arm_cfg(env)
    x = NetAgent(C.net_of(spec), name=spec, copy_net=False)
    recs = play_records(cfg, x, x, games, turns, seed, both_seats=False, env_label=f"m46c_{env}")
    acc = {}
    for k, f in STATE_CLASSES_LEAD.items():
        n, ms, w = 0, [], []
        for rec in recs:
            d = rec.data
            for s in (0, 1):
                m = (d["present"][:, s] > 0) & (d["en_present"][:, s] > 0) & f(d, s)
                if m.any():
                    mm = BH.masses(d, s, m)
                    if mm is not None:
                        ms.append(mm)
                        w.append(int(m.sum()))
                        n += int(m.sum())
                    # outcomes in this class
        row = {"n": n}
        if ms:
            w = np.asarray(w, float)
            row["mass"] = {kk: float(np.nansum([x_[kk] * wi for x_, wi in zip(ms, w)]) / w.sum()) for kk in ms[0]}
        acc[k] = row
    d = {"spec": spec, "env": env, "games": games, "turns": turns, "conditional": acc,
         "summary": BH.game_summary(recs, None)}
    C.save_gz(p, d)
    return (spec, env), d


def run_records(workers: int, games: int = 8, turns: int = 600) -> dict:
    js = [(spec, env, games, turns, C.NAT_ROOT + 7 + int(spec.split(":")[1])) for spec, env in natural_combos()]
    return {f"{k[0]}|{k[1]}": v for k, v in C.run_jobs(records_job, js, workers, "M4.6c records")}


# ---------------------------------------------------------------------------------------------- head to head
CURVE_POINTS = (512_000, 1_024_000, 2_048_000, 3_072_000)


def h2h_job(args) -> tuple:
    """(key, env, x spec, y spec or ref:<name>, episodes, seed, window): evaluate_discounted from s0 (E1)."""
    key, env, xs, ys, episodes, seed, window = args
    p = os.path.join(EVAL, "h2h", key.replace(":", "_").replace("|", "__").replace("@", "_at_") + ".json.gz")
    if os.path.exists(p):
        return key, C.load_gz(p)
    import torch
    torch.set_num_threads(1)
    from ....env.discounted import evaluate_discounted
    from ....rl import NetAgent
    cfg = C.arm_cfg(env)
    x = NetAgent(C.net_of(xs), name=xs, copy_net=False)
    if ys.startswith("ref:"):
        from ..m45c.references import reference
        y = reference(ys[4:])
    else:
        y = NetAgent(C.net_of(ys), name=ys, copy_net=False)
    e = evaluate_discounted(cfg, x, y, episodes=episodes, seed=seed, window=window, cache_size=5000)
    d = {k: e[k] for k in ("estimate", "stderr", "ci95", "seat_A", "seat_B", "window", "pairs", "elapsed_s")}
    d.update({"key": key, "env": env, "x": xs, "y": ys, "x_trained_on": C.trained_on(xs),
              "y_trained_on": C.trained_on(ys) if not ys.startswith("ref:") else "scripted",
              "per_pair": None})
    C.save_gz(p, d)
    return key, d


def h2h_jobs(parts=("curve", "start", "others", "rr", "suite", "cross")) -> list:
    from ..m45c.references import SUITE
    js = []
    for s in C.SEEDS:
        for arm in C.ARMS:
            fin = f"{arm}:{s}"
            if "curve" in parts:
                for n in CURVE_POINTS:
                    js.append((f"curve|{arm}|s{s}|{n}", arm, f"{arm}:{s}@{n}", f"start:{s}", 32, C.CURVE_ROOT, 460))
                js.append((f"curve|{arm}|s{s}|{C.BUDGET}", arm, fin, f"start:{s}", 32, C.CURVE_ROOT, 460))
            if "start" in parts:
                js.append((f"start|{arm}|s{s}", arm, fin, f"start:{s}", 64, C.H2H_ROOT, 1000))
            if "others" in parts:
                for o in C.SEEDS:
                    if o != s:
                        js.append((f"others|{arm}|s{s}|o{o}", arm, fin, f"start:{o}", 16, C.H2H_ROOT + 1, 460))
            if "suite" in parts:
                for r in SUITE:
                    js.append((f"suite|{arm}|{fin}|{r}", arm, fin, f"ref:{r}", 16, C.H2H_ROOT + 2, 460))
                    js.append((f"suite|{arm}|start:{s}|{r}", arm, f"start:{s}", f"ref:{r}", 16, C.H2H_ROOT + 2, 460))
            if "cross" in parts and arm == "C1":
                for env in C.ARMS:
                    js.append((f"cross|{env}|s{s}", env, f"C1:{s}", f"baseline:{s}", 32, C.H2H_ROOT + 3, 460))
        if "rr" in parts:
            for arm in C.ARMS:
                for o in C.SEEDS:
                    if o > s:
                        js.append((f"rr|{arm}|s{s}|o{o}", arm, f"{arm}:{s}", f"{arm}:{o}", 16, C.H2H_ROOT + 4, 460))
    return js


def h2h_ext_jobs() -> list:
    """The extension's head-to-head (amendment of 2026-10-15): 8.192M finals against start:s (W 1000), the other
    starts and each other (C1 vs baseline, both environments), and against the same arm's 4.096M final."""
    js = []
    for s in C.SEEDS:
        for arm in C.ARMS:
            fin = f"ext/{arm}:{s}"
            js.append((f"extstart|{arm}|s{s}", arm, fin, f"start:{s}", 64, C.H2H_ROOT, 1000))
            js.append((f"extvsmain|{arm}|s{s}", arm, fin, f"{arm}:{s}", 32, C.CURVE_ROOT, 460))
            for o in C.SEEDS:
                if o != s:
                    js.append((f"extothers|{arm}|s{s}|o{o}", arm, fin, f"start:{o}", 16, C.H2H_ROOT + 1, 460))
        for env in C.ARMS:
            js.append((f"extcross|{env}|s{s}", env, f"ext/C1:{s}", f"ext/baseline:{s}", 32, C.H2H_ROOT + 3, 460))
    return js


def run_h2h(workers: int, parts=("curve", "start", "others", "rr", "suite", "cross"), ext: bool = False) -> dict:
    js = h2h_ext_jobs() if ext else h2h_jobs(parts)
    js = sorted(js, key=lambda j: -j[4] * j[6])
    t0 = time.perf_counter()
    res = dict(C.run_jobs(h2h_job, js, workers, "M4.6c head to head"))
    print(f"  [h2h] {len(js)} jobs in {time.perf_counter() - t0:.0f} s", flush=True)
    return res


# ---------------------------------------------------------------------------------------------- dispatch
def run_family(name: str, workers: int) -> dict:
    t0 = time.perf_counter()
    if name == "premium":
        res = run_paired(premium_jobs(), workers, "premium")
    elif name == "premium_mid":
        res = run_paired(premium_jobs(snap=C.MIDPOINT, stages=("mid",), K=K_MID, family="premium_mid", j2=False,
                                      j3=False), workers, "premium_mid")
    elif name == "premium150":
        res = run_paired(premium_jobs(K=K_150, horizon=150, family="premium150", j2=False, j3=False,
                                      cell_filter=p150_filter), workers, "premium150")
    elif name == "battery":
        res = run_paired(battery_jobs(), workers, "battery")
    elif name == "states":
        res = collect_states(workers)
    elif name == "natural_cf":
        res = run_paired(natural_cf_jobs(collect_states(workers, ("trailer", "leader"))), workers, "natural_cf")
    elif name == "freeze":
        res = run_paired(freeze_jobs(collect_states(workers, ("freezer", "breaker"))), workers, "freeze")
    elif name == "specialists":
        res = run_paired(specialist_jobs(), workers, "specialists")
    elif name == "natural":
        res = run_natural(workers)
    elif name == "records":
        res = run_records(workers)
    elif name == "h2h":
        res = run_h2h(workers)
    # ---- the pre-registered extension (amendment of 2026-10-15): the 8.192M finals, same states and seeds
    elif name == "premium_ext":
        res = run_paired(premium_jobs(family="premium_ext", prefix="ext/"), workers, "premium_ext")
    elif name == "battery_ext":
        res = run_paired(battery_jobs(prefix="ext/", family="battery_ext"), workers, "battery_ext")
    elif name == "natural_cf_ext":
        res = run_paired(natural_cf_jobs(collect_states(workers, ("trailer", "leader"), prefix="ext/"), prefix="ext/",
                                         family="natural_cf_ext"), workers, "natural_cf_ext")
    elif name == "natural_ext":
        res = run_natural(workers, prefix="ext/")
    elif name == "h2h_ext":
        res = run_h2h(workers, ext=True)
    elif name in ("premium150p", "premium150p_ext"):
        # EXPLORATORY (amendment of 2026-10-15): the powered 150-turn premium, mid and late, all situations, J1 only
        ext = name.endswith("_ext")
        res = run_paired(premium_jobs(stages=("mid", "late"), K=K_PREM, horizon=150, family=name, j2=False, j3=False,
                                      prefix="ext/" if ext else ""), workers, name)
    elif name == "roleswap":
        res = run_paired(roleswap_jobs(), workers, "roleswap")
    elif name == "roleswap_ext":
        res = run_paired(roleswap_jobs("ext/", "roleswap_ext"), workers, "roleswap_ext")
    else:
        raise SystemExit(f"unknown family {name!r}")
    C.jdump(os.path.join(C.OUT, "stages", f"{name}.json"), {"family": name, "jobs": len(res),
                                                            "wall_s": time.perf_counter() - t0,
                                                            "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    return res


FAMILIES = ("premium", "premium_mid", "premium150", "battery", "states", "natural_cf", "freeze", "specialists",
            "natural", "records", "h2h", "premium_ext", "battery_ext", "natural_cf_ext", "natural_ext", "h2h_ext",
            "roleswap", "roleswap_ext", "premium150p", "premium150p_ext")

__all__ = ["run_family", "FAMILIES", "premium_jobs", "battery_cells", "battery_jobs", "natural_cf_jobs",
           "freeze_jobs", "specialist_jobs", "h2h_jobs", "natural_combos", "pjob", "mk", "CURVE_POINTS",
           "STATE_CLASSES_LEAD"]
