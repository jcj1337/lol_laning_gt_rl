"""M5a readiness checks (docs/m5/m5a_readiness_protocol.md, R0-R10; R11 is the regression suite).

Each part writes its record into results/m5/readiness.json under its id (re-running a part replaces only its entry);
long parts keep their raw outputs under results/rl/checkpoints/m5/readiness/. Stand-ins (no M5 policy exists yet):
the OM2 treatment finals for the om format, the M4.7 control finals (pure self-play continuations of ext/baseline:s)
and ext/baseline:s for the sp format.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from . import common as C
from . import players as PL

RDIR = os.path.join(C.CKPT, "readiness")
OM_PROXY = "om2:treatment:{s}"
SP_PROXY = "spec:" + os.path.join("results", "rl", "checkpoints", "m47", "train", "main", "control", "s{s}", "final.pt")
PARTS = ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10")
BELIEF_TOL = 1e-5


def om_proxy(s: int = 0) -> str:
    return OM_PROXY.format(s=s)


def sp_proxy(s: int = 0) -> str:
    return SP_PROXY.format(s=s)


def _record(part: str, data: dict) -> dict:
    rec = C.load("readiness") or {"meta": C.meta(stage="readiness"), "parts": {}}
    data = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **data}
    if "checks" in data:
        data["passed"] = bool(data["checks"]) and all(bool(v) for v in data["checks"].values())
    rec["parts"][part] = data
    rec["meta"] = C.meta(stage="readiness")
    rec["protocol"] = C.doc_record(C.READINESS_PROTOCOL_DOC)
    rec["all_passed"] = all(p.get("passed", True) for p in rec["parts"].values())
    C.save_json("readiness", rec)
    return data


def freeze_protocol() -> dict:
    """Record the readiness protocol's SHA-256 before the first check (never rewritten)."""
    path = os.path.join(C.OUT, "readiness_protocol.json")
    if os.path.exists(path):
        old = C.load("readiness_protocol")
        if old["record"]["sha256"] != C.doc_record(C.READINESS_PROTOCOL_DOC)["sha256"]:
            raise SystemExit("the readiness protocol changed after it was recorded; add a dated note instead")
        return old
    d = {"meta": C.meta(stage="readiness_protocol"), "record": C.doc_record(C.READINESS_PROTOCOL_DOC),
         "note": "recorded before the first readiness check"}
    C.save_json("readiness_protocol", d)
    return d


# ---------------------------------------------------------------------------------------------- R0 artifacts and lineage
def r0_artifacts() -> dict:
    from ....rl import checkpoint_info
    from ..m46c import common as C46
    from ..m46c.archive import archive_path
    from ..om2 import common as O2
    from ..m46a.common import E4_TARGETS                 # M4.6a trained specialists for targets 0, 2, 4 only
    checks, data, missing = {}, {"starts": {}, "dependencies": {}}, []
    for s in C.SEEDS:
        p = C.start_path(s)
        if not os.path.exists(p):
            missing.append(p)
            continue
        ident = C.checkpoint_identity(p)
        md = checkpoint_info(p)["metadata"]
        summ = C.jload(os.path.join(os.path.dirname(p), "summary.json")) or {}
        src = summ.get("start_checkpoint", "")
        sident = C.checkpoint_identity(src) if src and os.path.exists(src) else None
        smd = checkpoint_info(src)["metadata"] if sident else {}
        lin = {"m46c_ext": {"agent_transitions": md.get("agent_transitions"),
                            "schedule": md["ppo"]["total_steps"], "ppo_seed": md["ppo"]["seed"], "lr": md["ppo"]["lr"],
                            "archive_share": (md.get("starts") or {}).get("archive_transition_share_finished"),
                            "initial_digest": summ.get("initial_digest")},
               "m45c_ext": {"path": src.replace("\\", "/"), "agent_transitions": smd.get("agent_transitions"),
                            "schedule": (smd.get("ppo") or {}).get("total_steps"),
                            "ppo_seed": (smd.get("ppo") or {}).get("seed"),
                            "current_fraction": (smd.get("shared") or {}).get("current_fraction"),
                            "archive_starts": "starts" in smd, "digest": (sident or {}).get("weights_digest")}}
        data["starts"][s] = {"identity": ident, "lineage": lin}
        checks[f"s{s}: M4.6c 8.192M continuation, PPO seed 98,000 + s"] = \
            lin["m46c_ext"]["agent_transitions"] == 8_192_000 and lin["m46c_ext"]["schedule"] == 8_192_000 and \
            lin["m46c_ext"]["ppo_seed"] == 98_000 + s
        checks[f"s{s}: started from the M4.5c 16.384M final of the same seed (digest)"] = sident is not None and \
            summ.get("initial_digest") == sident["weights_digest"] and lin["m45c_ext"]["agent_transitions"] == 16_384_000
        checks[f"s{s}: the M4.5c final is a fresh s0-only 100%-self-play schedule (seed s)"] = \
            lin["m45c_ext"]["schedule"] == 16_384_000 and lin["m45c_ext"]["ppo_seed"] == s and \
            not lin["m45c_ext"]["archive_starts"] and lin["m45c_ext"]["current_fraction"] == 1.0
        checks[f"s{s}: archive share of the M4.6c continuation 0.20-0.30"] = \
            0.20 <= (lin["m46c_ext"]["archive_share"] or 0) <= 0.30
    deps = {"archive": archive_path(), **{f"library:{n}": p for n, p in O2.LEARNED.items()},
            **{f"heldout:{n}": p for n, p in O2.HELDOUT_LEARNED.items()},
            **{f"gap_specialist:s{s}": C46.specialist_path("gap", s) for s in E4_TARGETS},
            **{f"freeze_specialist:s{s}": C46.specialist_path("freeze", s) for s in E4_TARGETS},
            **{f"m45c_final:s{s}": C46.start_path(s) for s in C.SEEDS},
            **{f"standin_om:s{s}": C.policy_path(om_proxy(s)) for s in (0, 1, 2)},
            **{f"standin_sp:s{s}": C.policy_path(sp_proxy(s)) for s in (0, 1)}}
    for k, p in deps.items():
        if not os.path.exists(p):
            missing.append(p)
            data["dependencies"][k] = {"path": p.replace("\\", "/"), "present": False}
            continue
        data["dependencies"][k] = (C.checkpoint_identity(p) if p.endswith(".pt") else
                                   {"path": p.replace("\\", "/"), "file_sha256": C.file_sha256(p)})
    from ....env.starts import load_archive
    data["archive_states"] = len(load_archive(archive_path(), C.arm_cfg("sp")))
    checks["every dependency present"] = not missing
    data["missing"] = missing
    data["prior_transitions"] = C.PRIOR_TRANSITIONS
    data["cumulative_after_m5"] = C.PRIOR_TRANSITIONS + C.BUDGET
    return _record("R0", {"checks": checks, "data": data})


# ---------------------------------------------------------------------------------------------- R1 seed block
_ROOT_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*([0-9][0-9_]*)\s*(#.*)?$")
_ROOT_MULTI = re.compile(r"^\s*([A-Z][A-Z0-9_, ]*?(SEED|ROOT)[A-Z0-9_, ]*)=\s*([0-9_, ]+)\s*(#.*)?$")


def scan_roots(root: str = os.path.join("lanegame", "experiments")) -> Dict[str, List[int]]:
    """Every integer *SEED* / *ROOT* constant of the experiment packages (outside m5), by file."""
    out: Dict[str, List[int]] = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != "__pycache__"]
        if os.path.normpath(dp).endswith(os.path.join("rl", "m5")):
            continue
        for f in fn:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dp, f)
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    m = _ROOT_MULTI.match(line)
                    if not m:
                        continue
                    for v in m.group(3).split(","):
                        v = v.strip().replace("_", "")
                        if v.isdigit():
                            out.setdefault(p.replace("\\", "/"), []).append(int(v))
    return out


def r1_seeds() -> dict:
    lo, hi = C.SEED_BLOCK
    found = scan_roots()
    clash = {p: [v for v in vs if lo <= v <= hi] for p, vs in found.items()}
    clash = {p: v for p, v in clash.items() if v}
    checks = {"every M5 root inside 70,000-70,999": all(lo <= v <= hi for v in C.ROOTS.values()),
              "M5 roots distinct": len(set(C.ROOTS.values())) == len(C.ROOTS),
              "no other study's root constant inside the block": not clash,
              "M5 training seeds 70,000-70,005 inside the block": all(lo <= C.TRAIN_SEED + s <= hi for s in C.SEEDS)}
    return _record("R1", {"checks": checks, "data": {"roots": C.ROOTS, "files_scanned": len(found), "clashes": clash,
                                                     "note": "OM2 (61,000-61,999) overlaps M4.3b's evaluation roots "
                                                     "61,000-69,999 (an existing overlap; distinct generators and "
                                                     "derivations, not an M5 issue)"}})


# ---------------------------------------------------------------------------------------------- R2 initialization
def probe_states(n_games: int = 4, turns: int = 120, seed: int = C.READY_ROOT) -> List[tuple]:
    """Observation vectors (103 entries) of natural self-play of ext/baseline:0 and of archive states (both seats)."""
    from ....env import LaneEnv
    from ....env.starts import load_archive
    from ..m46c.archive import archive_path
    from ..om2 import common as O2
    from ....opponent import NetAgent
    cfg = C.arm_cfg("om")
    pol = NetAgent(O2.net90(C.start_path(0)), "start")
    out = []
    rng = np.random.default_rng(seed)
    env = LaneEnv(cfg, max_turns=turns)
    for g in range(n_games):
        o = env.reset(seed=int(seed) + g)
        for _ in range(turns):
            out += [(o[0].vector.copy(), o[0].legal), (o[1].vector.copy(), o[1].legal)]
            r = env.step(pol.act(o[0], rng), pol.act(o[1], rng))
            o = r.obs
            if r.done:
                break
    arch = load_archive(archive_path(), cfg)
    for j in rng.choice(len(arch), size=min(100, len(arch)), replace=False):
        st = arch[int(j)]
        o = env.reset(seed=1, root=st.lane, k0=st.phase, champ=st.champ)
        out += [(o[0].vector.copy(), o[0].legal), (o[1].vector.copy(), o[1].legal)]
    return out


def r2_init() -> dict:
    import torch
    torch.set_num_threads(1)
    from ....opponent import prior_features
    from .train import initial_net
    from ..m46c import common as C46
    states = probe_states()
    X = np.stack([v for v, _ in states]).astype(np.float32)
    rng = np.random.default_rng(C.READY_ROOT + 2)
    checks, data = {}, {"states": len(X)}
    for s in C.SEEDS:
        src = C46.net_of(C.start_spec(s))
        w = initial_net("om", s)
        row = {}
        for name, extra in (("prior", np.tile(prior_features(C.K), (len(X), 1))),
                            ("random", rng.dirichlet(np.ones(C.K), len(X)).astype(np.float32))):
            xw = np.concatenate([X, extra], 1)
            # vary the previous-action entries too (arbitrary 0/1 values)
            xw2 = xw.copy()
            xw2[:, 90:103] = rng.integers(0, 2, size=(len(X), 13)).astype(np.float32)
            with torch.no_grad():
                l0, v0 = src(torch.as_tensor(X[:, :90]))
                l1, v1 = w(torch.as_tensor(xw))
                l2, v2 = w(torch.as_tensor(xw2))
                single = max(float((src(torch.as_tensor(X[i:i + 1, :90]))[0] - w(torch.as_tensor(xw[i:i + 1]))[0])
                                   .abs().max()) for i in range(0, len(X), 23))
            row[name] = {"batched_bit_exact": bool(torch.equal(l0, l1) and torch.equal(v0, v1)),
                         "batched_bit_exact_varied_last_action": bool(torch.equal(l0, l2) and torch.equal(v0, v2)),
                         "single_row_max_abs_logit_diff": single}
        sp = initial_net("sp", s)
        with torch.no_grad():
            same_sp = bool(torch.equal(sp(torch.as_tensor(X[:, :90]))[0], src(torch.as_tensor(X[:, :90]))[0]))
        data[f"s{s}"] = {**row, "sp_copy_identical": same_sp}
        checks[f"s{s}: om start = ext/baseline:s in batched evaluation (any belief, any previous action)"] = all(
            r["batched_bit_exact"] and r["batched_bit_exact_varied_last_action"] for r in row.values())
        checks[f"s{s}: single-row differences <= 1e-5 (float32 rounding)"] = all(
            r["single_row_max_abs_logit_diff"] <= 1e-5 for r in row.values())
        checks[f"s{s}: sp start = ext/baseline:s"] = same_sp
    return _record("R2", {"checks": checks, "data": data})


# ---------------------------------------------------------------------------------------------- R3 defaults unchanged
def _pytest(args: List[str]) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], capture_output=True, text=True)
    tail = (p.stdout or "").strip().splitlines()[-3:]
    return {"args": args, "returncode": p.returncode, "tail": tail, "wall_s": time.perf_counter() - t0}


def reproduce_battery(K: int = 2) -> dict:
    """M5's battery state_job on M4.5c's own states and seeds reproduces the stored records exactly (sp format)."""
    import gzip
    from ..m45c import battery as B
    from . import battery as MB
    out = {}
    for fam, idx_list in (("A_wave", (0,)), ("B_freeze", (0,)), ("F_freeze_c", (0,))):
        if B.FAMILIES[fam][0] == "constructed":
            starts = {0: B.constructed_state(C.arm_cfg("sp"), fam, B.GRIDS[fam][0])}
        else:
            (_, _), res = B.collect_job(("final:ext:0", fam, max(idx_list) + 1))
            starts = {i: res["states"][i]["start"] for i in idx_list}
        for i in idx_list:
            for proto in ("policy",) + (("wk_shove",) if fam == "B_freeze" else ()):
                ref_path = os.path.join("results", "rl", "checkpoints", "m45c", "battery", "s0_ext", "partial",
                                        f"{fam}__{i}__{proto}.json.gz")
                if not os.path.exists(ref_path):
                    continue
                with gzip.open(ref_path, "rt", encoding="utf-8") as f:
                    ref = json.load(f)
                _, d = MB.state_job(("spec:" + C.policy_path("m46c:start:0"), "repro", fam, i, starts[i], None, proto,
                                     int(ref["horizon"]), "", K, os.path.join(RDIR, "repro_battery"), B.ROLL_SEED_B))
                same = all(np.array_equal(np.asarray(ref["GQ"][c][st][cp][:K]), np.asarray(d["GQ"][c][st][cp]))
                           for c in ref["GQ"] for st in ("G", "Q") for cp in ref["GQ"][c][st])
                out[f"{fam}/{i}/{proto}"] = {"exact": same, "horizon": int(ref["horizon"]), "K": K}
    return out


def reproduce_weak(K: int = 2) -> dict:
    from ..m46a import engine as EN
    from ..m46c import common as C46
    from ..m46c import evaluate as EV
    from ..m46c.archive import lead_cell
    from . import cells as CE
    out = {}
    for i in (0, 1, 9):
        p = os.path.join(C46.CKPT, "eval", "battery_ext", "baseline", "bat_baseline_s0", f"{i}.json.gz")
        if not os.path.exists(p):
            continue
        ref = C46.load_gz(p)
        focal, st, sit, d = dict(EV.battery_cells())[i]
        cands = EV.WEAK_CANDS if focal == "weak" else EV.STRONG_CANDS
        job = {"spec": "m46c:ext/baseline:0", "family": "repro", "idx": i,
               "start": lead_cell(C46.arm_cfg("baseline"), st, sit, f"B+{d}" if focal == "weak" else f"A+{d}"),
               "proto": "policy", "horizon": EV.H, "roll": EN.roll_seeds(C46.ROLL_ROOT, 32)[:K], "cands": cands,
               "critic": False, "path": None}
        _, rec = CE.paired_job(job)
        out[ref["label"]] = {"exact": all(np.array_equal(np.asarray(ref["G"][c][cp][:K]), np.asarray(rec["G"][c][cp]))
                                          for c in cands for cp in ref["G"][c])}
    return out


def reproduce_m46f_freeze(K: int = 2) -> dict:
    """The cross-candidate design (both finals in seat A, each final in seat B) on an M4.6f C1 record, C1 environment."""
    from ..m46a import engine as EN
    from ..m46c import common as C46
    from ..m46f import behaviour as BH
    from ..m46f import common as F
    from . import cells as CE
    out = {}
    cfg = C46.arm_cfg("C1")
    cells = BH.freeze_cells(cfg)
    f = BH.finals(0, "ext")
    for idx in (0, 1):
        for opp in F.ARMS:
            p = os.path.join(BH.edir("ext"), "freeze", "s0", f"{idx}_vs_{opp}.json.gz")
            if not os.path.exists(p):
                continue
            ref = C46.load_gz(p)
            lab, st, meta = cells[idx]
            cands = {"shared": ("net", f"spec:{f['shared']}"), "separate": ("net", f"spec:{f['separate']}")}
            job = {"spec": f"spec:{f[opp]}", "family": "repro", "idx": idx, "start": st, "proto": f"net:spec:{f[opp]}",
                   "horizon": BH.H_FZ, "roll": EN.roll_seeds(F.CELL_ROOT + 1, BH.K_FZ)[:K], "cands": cands,
                   "critic": False, "path": None, "env": "C1"}
            _, rec = CE.paired_job(job)
            out[f"{lab} vs {opp}"] = {"exact": all(np.array_equal(np.asarray(ref["G"][c][cp][:K]),
                                                                  np.asarray(rec["G"][c][cp]))
                                                   for c in cands for cp in ref["G"][c])}
    return out


def reproduce_om2_cells(N: int = 3) -> dict:
    from ..om2 import common as O2
    from ..om2 import evaluate as OE
    from . import games as G
    out = {}
    want = {"E|g_shove", "R1|base:3", "R2|switch_bully_farm_50"}
    for j in OE.jobs((0,)):
        tag = f"{j['family']}|{j['opponent']}"
        if tag not in want or not os.path.exists(j["path"]):
            continue
        ref = OE.load(j["path"])
        mj = {"key": tag, "path": os.path.join(RDIR, "repro_om2", f"{tag.replace('|', '__').replace(':', '_')}.npz"),
              "focals": [("net", "om2:treatment:0"), ("net", "om2:control:0", "prior")],
              "opponent": ("opp", j["opponent"]), "N": N, "T": OE.T_EVAL, "root": O2.EVAL_ROOT, "cell": j["cell"]}
        if os.path.exists(mj["path"]):
            os.remove(mj["path"])
        G.cell_job(mj)
        d = G.load(mj["path"])
        out[tag] = {"treatment_exact": bool(np.array_equal(d["f0__r"], ref["treatment__r"][:N])),
                    "control_exact": bool(np.array_equal(d["f1__r"], ref["control__r"][:N])),
                    "trace_max_abs_diff": float(np.abs(d["f0__trace"] - ref["treatment__trace"][:N]).max())}
    return out


def r3_defaults() -> dict:
    tests = _pytest(["tests/test_legacy_pins.py", "tests/test_om2.py::test_collector_hooks_leave_default_training_bit_identical",
                     "-k", "pin or bit_identical or digest"])
    bat = reproduce_battery()
    weak = reproduce_weak()
    frz = reproduce_m46f_freeze()
    om2 = reproduce_om2_cells()
    checks = {"historical digest pins and OM2's default-trainer pin pass": tests["returncode"] == 0,
              "M4.5c battery records reproduced exactly (hooked loops)": bool(bat) and all(v["exact"] for v in bat.values()),
              "M4.6c weaker-side records reproduced exactly": bool(weak) and all(v["exact"] for v in weak.values()),
              "M4.6f cross-candidate freeze records reproduced exactly (C1)": bool(frz) and all(v["exact"] for v in frz.values()),
              "OM2 evaluation games reproduced exactly (both arms)": bool(om2) and all(
                  v["treatment_exact"] and v["control_exact"] for v in om2.values())}
    return _record("R3", {"checks": checks, "data": {"pytest": tests, "battery": bat, "weak": weak,
                                                     "m46f_freeze": frz, "om2": om2}})


# ---------------------------------------------------------------------------------------------- R4 smoke
SMOKE_STOP = {"om": 30_720, "sp": 40_960}      # about 10 updates each (om updates hold ~3,070 learner transitions)


def r4_smoke(workers: int = 4) -> dict:
    from . import train as T
    jobs = []
    for arm in C.ARMS:
        for tag in ("run", "rerun"):
            jobs.append(("smoke", arm, 0, C.BUDGET, SMOKE_STOP[arm], True,
                         os.path.join(C.CKPT, "train", "smoke", f"{arm}_{tag}"), None))
    res = C.run_jobs(T.train_job, jobs, workers, "M5 smoke")
    runs = {f"{j[1]}_{os.path.basename(j[6]).split('_')[-1]}": r for j, r in zip(jobs, res)}
    checks: Dict[str, bool] = {}
    data = {"runs": {}}
    from ..m44.train import _histories_equal
    from ..m45c.full import stability_scan
    for arm in C.ARMS:
        a, b = runs[f"{arm}_run"], runs[f"{arm}_rerun"]
        checks[f"{arm}: both runs completed"] = a.get("status") == "complete" and b.get("status") == "complete"
        if not checks[f"{arm}: both runs completed"]:
            data["runs"][arm] = {"run": a, "rerun": b}
            continue
        h = T.history(a["run_dir"])
        scan = stability_scan(h)
        st = a.get("starts") or {}
        checks.update({
            f"{arm}: finite losses and gradients": bool(a["finite_losses_and_gradients"]),
            f"{arm}: no illegal action sampled": a["rollout"].get("illegal_sampled_total", 1) == 0,
            f"{arm}: approximate KL in (0, 0.05)": all(0 < e["approx_kl"] < 0.05 for e in h),
            f"{arm}: initial weights = the arm's start": a["initial_digest"] == T.start_digest(arm, 0),
            f"{arm}: archive games started": (st.get("games_started") or {}).get("archive", 0) > 0,
            f"{arm}: input width {C.net_inputs(arm)}": all(e.get("input_width") == C.net_inputs(arm) for e in h),
            f"{arm}: no stop condition": not scan["alarms"],
            f"{arm}: re-run bit-identical weights": a["digest"] == b["digest"],
            f"{arm}: re-run identical histories": _histories_equal(a["run_dir"], b["run_dir"])})
        if arm == "sp":
            checks["sp: 100% current-policy"] = a["transitions_historical"] == 0
            checks["sp: both seats trained equally"] = a["transitions_seat_a"] == a["transitions_seat_b"]
        else:
            lib = ((st.get("om2") or {}).get("library") or {}).get("draws", {})
            checks["om: fixed-opponent games drawn"] = sum(v for k, v in lib.items() if k.startswith("lib:")) > 0
            checks["om: self-play games drawn"] = lib.get("self", 0) > 0
            checks["om: belief inputs vary"] = all(e["belief_std_mean"] > 0 for e in h)
            checks["om: belief inputs are posteriors"] = all(
                e["belief_min"] >= 0 and e["belief_max"] <= 1 + 1e-6 and e["belief_rowsum_dev"] < 1e-4 for e in h)
            checks["om: previous-action field shows game starts and actions"] = all(
                0 < e["lastaction_none_share"] < 0.5 for e in h)
            checks["om: beliefs updated"] = (st.get("om2") or {}).get("belief_updates", 0) > 0
        data["runs"][arm] = {"updates": a["updates"], "agent_transitions": a["agent_transitions"],
                             "transitions_per_s": a["transitions_per_s"], "digest": a["digest"],
                             "rerun_digest": b["digest"], "shares": a["shares"], "starts": st,
                             "memory": a.get("memory"), "wall_s": a["wall_s"]}
    return _record("R4", {"checks": checks, "data": data})


# ---------------------------------------------------------------------------------------------- R5 common random numbers
def _collector(arm: str, seed: int = 0):
    from ....rl.selfplay import SnapshotPool
    from ....rl.shared import MatchupSampler
    from . import train as T
    ppo = T.ppo_config(arm, seed, C.BUDGET)
    sampler = MatchupSampler(SnapshotPool(8, keep_first=False), 1.0)
    return T.collector_cls(arm)(C.setup(arm), ppo, sampler, ppo.seed + 1), ppo


def start_schedule(arm: str, n: int = 200) -> List[str]:
    col, _ = _collector(arm)
    out = []
    N = col.cfg.n_envs
    for r in range(n):
        i = r % N
        if r >= N:
            col._reset(i)
        env = col.envs[i]
        h = hashlib.sha256()
        h.update(np.asarray(col.obs[i][0].vector[:90]).tobytes())
        h.update(np.asarray(col.obs[i][1].vector[:90]).tobytes())
        h.update(json.dumps(env._rng.bit_generator.state, sort_keys=True, default=str).encode())
        out.append(f"{col.game_kind[i]}|{col.game_label[i]}|{env.max_turns}|{h.hexdigest()[:16]}")
    return out


def r5_crn(n: int = 200) -> dict:
    a, b = start_schedule("om", n), start_schedule("sp", n)
    same = [x == y for x, y in zip(a, b)]
    kinds = {k: sum(1 for x in a if x.startswith(k)) for k in ("initial", "archive")}
    checks = {f"the first {n} game starts are identical in both arms (kind, archive state, mirroring, length, chance "
              f"seed)": all(same)}
    return _record("R5", {"checks": checks, "data": {"n": n, "identical": int(sum(same)), "kinds": kinds,
                                                     "first": a[:3]}})


# ---------------------------------------------------------------------------------------------- R6 inputs, R8 boundaries
def probe_inputs(arm: str, seed: int = 0) -> dict:
    """One rollout of the arm's start through its own collector, with the collector instrumented to record every
    network input it builds; the om arm's inputs are then rebuilt independently by `players.OMPolicy` from the recorded
    start-of-turn states and codes (training inputs = evaluation inputs; no current-turn code; bootstraps)."""
    import torch
    torch.set_num_threads(1)
    from ....opponent import prior_features
    from . import train as T
    col, ppo = _collector(arm, seed)
    net = T.initial_net(arm, seed)
    gen = torch.Generator().manual_seed(ppo.seed)
    calls, steps = [], {i: [] for i in range(col.cfg.n_envs)}
    orig_aug, orig_obs = col.augment, col.observe_step

    def augment(i, s, v):
        x = orig_aug(i, s, v)
        calls.append((i, s, v, np.asarray(x).copy()))
        return x

    def observe_step(i, actions, result):
        o = col.obs[i]
        steps[i].append({"pre": (o[0].state, o[0].champ, o[0].k), "codes": (int(actions[0]), int(actions[1])),
                         "post": (result.obs[0].state, result.obs[0].champ, result.obs[0].k),
                         "post_vec": (result.obs[0].vector, result.obs[1].vector), "done": bool(result.done),
                         "learners": tuple(c is None for c in col.match[i].controllers)})
        orig_obs(i, actions, result)

    col.augment, col.observe_step = augment, observe_step
    first_obs = {i: (col.obs[i][0].vector, col.obs[i][1].vector) for i in range(col.cfg.n_envs)}
    roll, finished, counts = col.collect(net, gen)
    out = {"width": int(roll["obs"].shape[-1]), "turns": int(roll["obs"].shape[0]), "finished_games": len(finished)}
    if arm == "sp":
        out["inputs_are_the_observation"] = all(np.array_equal(np.asarray(v, dtype=np.float32), x)
                                                for _, _, v, x in calls)
        return out
    a = roll["active"]
    X = roll["obs"]
    none = X[:, :, 90] == 1.0
    prior = prior_features(C.K)
    bel = X[:, :, C.OBS_OM:]
    out["rows_at_game_start"] = int((none & a).sum())
    out["prior_at_every_game_start"] = bool(np.all(bel[none & a] == prior))
    out["belief_moves_later"] = bool(np.any(np.abs(bel[~none & a] - prior).sum(-1) > 1e-3))
    # independent reconstruction of every training input of env 0..3 by OMPolicy
    # state and previous-action blocks must match bit for bit; the belief block within BELIEF_TOL (training batches the
    # likelihood forward passes across environments, OMPolicy evaluates one item: float32 rounding of the same sums)
    mism, checked, boot, worst = 0, 0, 0, {"state": 0.0, "last": 0.0, "belief": 0.0}
    cfg = C.arm_cfg("sp")

    def same(x, y) -> bool:
        d = np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64))
        for name, sl in (("state", slice(0, 90)), ("last", slice(90, C.OBS_OM)), ("belief", slice(C.OBS_OM, None))):
            worst[name] = max(worst[name], float(d[sl].max()))
        return d[:C.OBS_OM].max() == 0.0 and d[C.OBS_OM:].max() <= BELIEF_TOL

    for i in range(col.cfg.n_envs):
        game_pols = None
        for t, stp in enumerate(steps[i]):
            if game_pols is None:
                game_pols = {s: PL.OMPolicy(net, 0, 0, s, 1) for s in (0, 1) if stp["learners"][s]}
            lane, m4, k = stp["pre"]
            for s, pol in game_pols.items():
                _, x = pol.input(lane, m4, k)
                checked += 1
                if not same(x, X[t, 2 * i + s]):
                    mism += 1
            for pol in game_pols.values():
                pol.observe_turn(cfg, lane, m4, k, stp["codes"])
            if stp["done"]:
                # the truncation bootstrap input: the recorded augment call on the post-step vector
                for s, pol in game_pols.items():
                    rec = [x for (ii, ss, v, x) in calls if ii == i and ss == s and v is stp["post_vec"][s]]
                    lane2, m42, k2 = stp["post"]
                    _, xe = pol.input(lane2, m42, k2)
                    boot += 1
                    if not rec or not same(rec[0], xe):
                        mism += 1
                game_pols = None
    out.update({"inputs_reconstructed": checked, "bootstraps_reconstructed": boot, "mismatches": mism,
                "max_abs_diff_by_block": worst, "belief_tolerance": BELIEF_TOL,
                "first_run_note": "the first run of 2026-10-17 compared the whole input bit for bit and failed on the "
                "belief block only (max 2.9e-6, float32 rounding of batched vs single-item likelihoods); state and "
                "previous-action blocks were exact"})
    return out


def r6_inputs() -> dict:
    om, sp = probe_inputs("om"), probe_inputs("sp")
    sm = ((C.load("readiness") or {}).get("parts", {}).get("R4") or {}).get("passed")
    checks = {"sp: every network input is exactly the 90-entry observation": sp["width"] == 90 and
              sp["inputs_are_the_observation"],
              "om: 116 inputs": om["width"] == C.NET_IN_OM,
              "om: the belief is the prior at every game start (s0 and archive)": om["prior_at_every_game_start"] and
              om["rows_at_game_start"] > 0,
              "om: the belief moves after game starts": om["belief_moves_later"],
              "om: every training input rebuilt by OMPolicy from the start-of-turn state and past codes (state and "
              "previous action bit-exact, belief within 1e-5)": om["mismatches"] == 0 and om["inputs_reconstructed"] > 0,
              "om: truncation bootstraps read the post-step belief": om["bootstraps_reconstructed"] > 0 and
              om["mismatches"] == 0,
              "smoke (R4) input checks passed": bool(sm)}
    return _record("R6", {"checks": checks, "data": {"om": om, "sp": sp}})


# ---------------------------------------------------------------------------------------------- R7 opponents and shares
def r7_shares(games: int = 70_000) -> dict:
    from . import train as T
    from ..om2.collector import LibrarySampler
    from ....rl.selfplay import SnapshotPool
    from ....rl.shared import MatchupSampler
    ppo = T.ppo_config("om", 0, C.BUDGET)
    rng = np.random.default_rng(np.random.SeedSequence(ppo.seed + 1).spawn(1)[0])
    base = MatchupSampler(SnapshotPool(8, keep_first=False), 1.0)
    ls = LibrarySampler(base, C.TRAIN_OPPONENTS, [None] * len(C.TRAIN_OPPONENTS), C.P_LIB)
    for _ in range(games):
        ls.sample(rng)
    cnt = {k: v for k, v in ls.counts.items()}
    lib = {k[4:]: v for k, v in cnt.items() if k.startswith("lib:")}
    exp = games * C.P_LIB / len(C.TRAIN_OPPONENTS)
    rec = C.load("readiness") or {}
    smoke = ((rec.get("parts") or {}).get("R4") or {}).get("data", {}).get("runs", {})
    bench = ((rec.get("parts") or {}).get("R9") or {}).get("data", {}).get("shares_om")
    checks = {"all 15 fixed opponents drawn": set(lib) == set(C.TRAIN_OPPONENTS),
              "fixed-opponent game share 0.49-0.51 of draws": abs(sum(lib.values()) / games - 0.5) < 0.01,
              "each opponent within 10% of uniform": all(abs(v / exp - 1) < 0.10 for v in lib.values())}
    return _record("R7", {"checks": checks, "data": {"simulated_games": games, "draws": cnt,
                                                     "smoke_shares": {a: r.get("shares") for a, r in smoke.items()},
                                                     "bench_shares_om": bench}})


# ---------------------------------------------------------------------------------------------- R8 equivalences
def r8_equivalence() -> dict:
    """The widened start in the om path against the 90-input start in a battery job and a cells job; OMPolicy and the
    OM2 ArmPlayer (R3 covers whole games); and the current turn's code never reaches the current input."""
    from ..m45c import battery as B
    from . import battery as MB
    from . import cells as CE
    from .train import initial_net
    PL.install("readiness:widened:0", initial_net("om", 0))
    cfg = C.arm_cfg("sp")
    st = B.constructed_state(cfg, "F_freeze_c", B.GRIDS["F_freeze_c"][0])
    res = {}
    for spec in ("readiness:widened:0", "m46c:ext/baseline:0"):
        _, d = MB.state_job((spec, "eq", "F_freeze_c", 0, st, None, "policy", 50, "", 3, os.path.join(RDIR, "eq",
                                                                                                    spec.split(":")[0]),
                             C.READY_ROOT + 3))
        res[spec] = d
    a, b = res["readiness:widened:0"], res["m46c:ext/baseline:0"]
    diff = max(abs(x - y) for c in a["GQ"] for s in ("G", "Q") for cp in a["GQ"][c][s]
               for x, y in zip(a["GQ"][c][s][cp], b["GQ"][c][s][cp]))
    codes_equal = a["first_codes"] == b["first_codes"]
    # leakage: the input at turn i is built before the turn's codes exist, and observe_turn of turn i changes it
    from ...m45b import states as S
    pol = PL.OMPolicy(PL.net_of(om_proxy(0)), 0, 1, 0, 10)
    lane, m4, k = S.s0(cfg)
    _, x0 = pol.input(lane, m4, k)
    s0_ = json.dumps(pol.state())
    for i in range(5):
        pol.decide(cfg, lane, m4, 0, i)
        pol.probs(cfg, lane, m4, 0, i)
    pure = json.dumps(pol.state()) == s0_ and np.array_equal(pol.input(lane, m4, k)[1], x0)
    from ...m45b.strategies import get
    opp = int(get("g_bully").decide(cfg, lane, m4, 1, 0)[0])
    pol.observe_turn(cfg, lane, m4, k, (int(pol.decide(cfg, lane, m4, 0, 0)[0]), opp))
    _, x1 = pol.input(lane, m4, k)
    checks = {"widened om start = 90-input start in a battery job (G and Q within 1e-5, same first codes)":
              diff <= 1e-5 and codes_equal,
              "decide is pure (repeated decide / probs leave belief and input unchanged)": pure,
              "the opponent's code enters the input only after observe_turn (previous-action entries and belief)":
              not np.array_equal(x0[90:], x1[90:]) and np.array_equal(x0[:90], x1[:90])}
    return _record("R8", {"checks": checks, "data": {"battery_max_abs_diff": diff, "first_codes_equal": codes_equal}})


# ---------------------------------------------------------------------------------------------- amendment 1 checks
# The M5a correction pass's own checks (results/m5/corrections_readiness.json; the M5a record readiness.json is left
# as it was): A1 the monitored smoke reproduces the stored R4 smoke runs, A2 the inputs of the main training (the
# amended pre-registration, every start checkpoint and archive, the run manifest), A3 the targeted tests, A4 the full
# regression suite.
CORR_PARTS = ("A1", "A2", "A3", "A4")


def _record_corr(part: str, data: dict) -> dict:
    rec = C.load("corrections_readiness") or {"meta": C.meta(stage="corrections"), "parts": {}}
    data = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **data}
    if "checks" in data:
        data["passed"] = bool(data["checks"]) and all(bool(v) for v in data["checks"].values())
    prev = rec["parts"].get(part)
    if prev is not None:                                   # every attempt is kept (both runs are reported)
        data["earlier_attempts"] = prev.pop("earlier_attempts", []) + [prev]
    rec["parts"][part] = data
    rec["meta"] = C.meta(stage="corrections")
    rec["all_passed"] = all(p.get("passed", True) for p in rec["parts"].values())
    C.save_json("corrections_readiness", rec)
    return data


def a1_monitored_smoke(workers: int = 2) -> dict:
    """Both arms' R4 smoke (seed 0, the real schedule stopped at SMOKE_STOP) re-run with the runtime monitor attached:
    weights, snapshots and every non-wall-clock history field must equal the stored M5a run made without it; the live
    alarms must equal the post-hoc scan."""
    from . import health as HL
    from . import train as T
    from ..m44.train import _histories_equal
    from ..m45c.full import stability_scan
    jobs = [("smoke", arm, 0, C.BUDGET, SMOKE_STOP[arm], True,
             os.path.join(C.CKPT, "train", "smoke", f"{arm}_monitored"), None, True) for arm in C.ARMS]
    res = C.run_jobs(T.train_job, jobs, workers, "M5 monitored smoke")
    checks, data = {}, {}
    for (j, r) in zip(jobs, res):
        arm = j[1]
        ref = os.path.join(C.CKPT, "train", "smoke", f"{arm}_run")
        refs = C.jload(os.path.join(ref, "summary.json")) or {}
        checks[f"{arm}: monitored run completed"] = r.get("status") == "complete"
        if r.get("status") != "complete":
            data[arm] = r
            continue
        from ....rl import checkpoint_info
        snaps = lambda d: sorted(os.listdir(os.path.join(d, "snapshots")))
        wd = lambda d, f: checkpoint_info(os.path.join(d, "snapshots", f))["digest"]
        same_snaps = snaps(ref) == snaps(j[6]) and all(wd(ref, f) == wd(j[6], f) for f in snaps(ref))
        live = sorted(a["alarm"] for a in HL.alarms(j[6]))
        post = sorted(stability_scan(T.history(j[6]))["alarms"])
        checks.update({f"{arm}: final weights equal the stored M5a smoke run": r["digest"] == refs.get("digest"),
                       f"{arm}: snapshot weights identical (same names, same digests)": same_snaps,
                       f"{arm}: histories identical (non-wall-clock fields)": _histories_equal(ref, j[6]),
                       f"{arm}: live alarms equal the post-hoc scan": live == post})
        data[arm] = {"digest": r["digest"], "reference_digest": refs.get("digest"), "updates": r["updates"],
                     "live_alarms": live, "post_hoc_alarms": post, "wall_s": r["wall_s"]}
    return _record_corr("A1", {"checks": checks, "data": data})


def a2_campaign_inputs() -> dict:
    """The main training's inputs: the amended pre-registration verifies; every start checkpoint, library network and
    the archive loads with its digest; the arms' initial networks are the starts; the run manifest builds; no main run
    directory exists yet (or every existing one is complete)."""
    from . import train as T
    checks: Dict[str, bool] = {}
    pre = C.require_prereg()
    act = C.active_version(pre)
    checks["pre-registration chain verified, document = active version 1"] = act["version"] == 1
    man = T.run_manifest("main", C.ARMS, C.SEEDS, C.BUDGET)
    C.save_json("manifest_preview", man)
    checks["run manifest builds (starts, library, archive identities)"] = bool(man["start_checkpoints"]) and \
        len(man["library_networks"]) == 5 and bool(man["archive"]["file_sha256"])
    checks["manifest records the active pre-registration"] = (man.get("preregistration_active") or {}).get(
        "sha256") == act["sha256"]
    for s in C.SEEDS:
        for a in C.ARMS:
            checks[f"{a}_s{s}: initial network digest"] = bool(T.start_digest(a, s))
    todo = T.plan_runs("main", C.ARMS, C.SEEDS, restart_incomplete=False)
    checks["every main run still to do (none started)"] = len(todo) == len(C.ARMS) * len(C.SEEDS)
    return _record_corr("A2", {"checks": checks, "data": {"active": act, "runs_to_do": len(todo),
                                                           "code": man["code"]}})


def corr_pytest(part: str, args: List[str], log: str) -> dict:
    """A3 / A4: a pytest run with its output kept in results/m5/<log>."""
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "pytest", *args], capture_output=True, text=True)
    with open(os.path.join(C.OUT, log), "w", encoding="utf-8", newline="\n") as f:
        f.write(r.stdout[-200_000:] + "\n" + r.stderr[-20_000:])
    tail = [x for x in r.stdout.strip().splitlines() if x.strip()][-1:] or [""]
    return _record_corr(part, {"checks": {f"pytest {' '.join(args)}: exit 0": r.returncode == 0},
                               "data": {"summary": tail[0], "seconds": time.perf_counter() - t0, "log": log}})


__all__ = ["freeze_protocol", "r0_artifacts", "r1_seeds", "r2_init", "r3_defaults", "r4_smoke", "r5_crn", "r6_inputs",
           "r7_shares", "r8_equivalence", "probe_inputs", "start_schedule", "scan_roots", "om_proxy", "sp_proxy",
           "PARTS", "RDIR", "CORR_PARTS", "a1_monitored_smoke", "a2_campaign_inputs", "corr_pytest"]
