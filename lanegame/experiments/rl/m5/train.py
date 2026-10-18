"""M5 training (design §3; pre-registration §2): the two matched continuations of `ext/baseline:s`.

Everything is imported, nothing re-implemented:
    om (A)  `train_shared` + `om2.collector.om_collector_cls("treatment")` (the M4.6c start mixture, the OM2 library
            mix and belief) on `m45_calibrated(obs_last_action=True)`, the network `widen_inputs(ext/baseline:s, 116)`
    sp (B)  `train_shared` + `m46c.train.collector()` (the M4.6c start mixture, 100% current-policy self-play) on
            `m45_calibrated()`, the network a copy of ext/baseline:s
Both: `m46c.train.shared_config()`, the M4.6c PPO settings (`ppo_config`), PPO seed TRAIN_SEED + s, a fresh Adam,
lr 3e-4 annealed to 0 over BUDGET agent transitions.

Runs write results/rl/checkpoints/m5/train/<phase>/<arm>/s<seed>/ (run.json, history.jsonl, health.jsonl, snapshots/,
final.pt, summary.json). A run directory is never overwritten outside the smoke and bench phases. An interrupted run
cannot be resumed mid-schedule (the trainer stores no optimizer or collector state); with `restart_incomplete` its
directory is moved, intact, to train/<phase>/partial/<arm>_s<seed>_<time>/ and the run restarts from its start: by the
determinism contract the restart must reproduce the preserved history prefix exactly, which `prefix_check` verifies and
`check` enforces. A restart repeats the lost work, so it adds up to one run's wall time.

Runtime health (amendment 1, `health`): every run is watched by `health.RunMonitor` (the registered stop conditions,
update by update; a DEFECT alarm ends the run, an INVESTIGATE alarm is recorded and blocks evaluation until resolved),
and the campaign runs under `health.run_campaign` (progress log, more than two failed pairs stop the campaign).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import common as C

SMOKE_UPDATES = 10
DETERMINISTIC_KEYS = ("update", "agent_transitions", "env_turns", "policy_loss", "value_loss", "entropy", "approx_kl",
                      "clipfrac", "grad_norm", "explained_variance", "value_mean", "return_mean", "actions_digest",
                      "rollout_transitions_historical", "games_current", "games_historical")


# ---------------------------------------------------------------------------------------------- the arms
def ppo_config(arm: str, seed: int, total: int, ppo_seed: Optional[int] = None):
    from ....rl import PPOConfig
    from ..m46c import common as C46
    return PPOConfig.continuing(C.arm_cfg(arm), total_steps=int(total), n_envs=C46.N_ENVS,
                                seed=int(C.TRAIN_SEED + seed if ppo_seed is None else ppo_seed), torch_threads=1,
                                ent_coef=C46.ENT_COEF, hidden=C46.HIDDEN, lane_cache=C46.LANE_CACHE,
                                game_length=C46.GAME_LENGTH, start_mix=0.0)


def shared_config():
    from ..m46c.train import shared_config as s
    return s()


def collector_cls(arm: str):
    if arm == "om":
        from ..om2.collector import om_collector_cls
        return om_collector_cls("treatment", C.P_LIB, C.LAMBDA)
    if arm == "sp":
        from ..m46c.train import collector
        return collector()
    raise ValueError(arm)


def initial_net(arm: str, seed: int):
    """The arm's network at the start: ext/baseline:s, widened with zero weights for the om arm (exactly the same
    actor and value outputs on the shared prefix; `readiness.init_equivalence` checks it), copied for the sp arm."""
    from ....rl import PolicyValueNet
    from ....rl.network import widen_inputs
    from ..m46c import common as C46
    src = C46.net_of(C.start_spec(seed))
    if arm == "om":
        return widen_inputs(src, C.NET_IN_OM)
    net = PolicyValueNet(src.obs_size, src.n_actions, src.hidden)
    net.load_state_dict(src.state_dict())
    return net


def training_spec() -> dict:
    out = {}
    for arm in C.ARMS:
        p = ppo_config(arm, 0, C.BUDGET).to_dict()
        for k in ("seed", "total_steps"):
            p.pop(k)
        out[arm] = {"label": C.ARM_LABEL[arm],
                    "environment": "GameConfig.m45_calibrated(obs_last_action=True)" if arm == "om"
                    else "GameConfig.m45_calibrated()",
                    "network": {"class": "PolicyValueNet", "inputs": C.net_inputs(arm), "hidden": [128, 128],
                                "outputs": 61,
                                "initialization": "widen_inputs(ext/baseline:s, 116): zero weights on the 26 new inputs"
                                if arm == "om" else "a copy of ext/baseline:s"},
                    "collector": "om2.collector.om_collector_cls('treatment'): M4.6c start mixture + OM2 library mix "
                    "(P_LIB 0.5 over 15 fixed opponents, only the learner trains) + one belief per learner seat"
                    if arm == "om" else "m46c.train.collector(): M4.6c start mixture",
                    "opponents": {"self_play_share_of_games": 1 - C.P_LIB if arm == "om" else 1.0,
                                  "fixed_opponents": list(C.TRAIN_OPPONENTS) if arm == "om" else [],
                                  "historical_snapshots_sampled": False},
                    "belief": {"types": list(C.BELIEF_TYPES), "lambda": C.LAMBDA, "eps": C.EPS,
                               "reset": "at every game start (s0 and archive starts)", "library": "fixed"}
                    if arm == "om" else None,
                    "ppo": p, "shared": shared_config().to_dict()}
    from ..m46c import common as C46
    return {"arms": out, "budget": C.BUDGET, "midpoint": C.MIDPOINT, "p75": C.P75, "snap_every": C.SNAP_EVERY,
            "start": "ext/baseline:s (M4.6c 8.192M continuation of the M4.5c 16.384M final; 24.576M prior "
            "transitions over two schedules)",
            "starts": {"beta": C46.ARCHIVE_BETA, "archive_length": C46.ARCHIVE_LENGTH, "game_length": C46.GAME_LENGTH,
                       "mirror_p": C46.MIRROR_P, "target_transition_share": C46.ARCHIVE_TARGET_SHARE},
            "optimizer": "fresh Adam (eps 1e-5), lr 3e-4 annealed linearly to 0 over the budget",
            "seeds": {"ppo": f"{C.TRAIN_SEED} + s, the same in both arms of a pair", "pairs": list(C.SEEDS)}}


# ---------------------------------------------------------------------------------------------- rollout diagnostics
class Stats:
    """`rollout_stats` of both arms (read only): M4.5c's RolloutStats, the opponents of the rollout's finished games, a
    digest of the sampled actions, the per-update learner-transition shares and, for the om arm, OM2's belief checks
    (spread, range, posterior validity, the last-action "none" share)."""

    def __init__(self, arm: str, ppo):
        from ..m45c.train import RolloutStats
        self.arm = arm
        if arm == "om":
            from ..om2.train import Stats as OM2Stats
            self.base = OM2Stats(ppo)
        else:
            self.base = RolloutStats(ppo)

    def __call__(self, roll: dict, finished: list) -> dict:
        out = self.base(roll, finished)
        a = roll["active"]
        out["actions_digest"] = hashlib.sha256(roll["actions"][a].tobytes()).hexdigest()[:16]
        opp: Dict[str, int] = {}
        for e in finished:
            opp[e.opponent] = opp.get(e.opponent, 0) + 1
        out["finished_by_opponent"] = opp
        out["input_width"] = int(roll["obs"].shape[-1])
        return out


# ---------------------------------------------------------------------------------------------- one run
def _peak_memory_mb() -> Dict[str, float]:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        out = {"rss_mb": mi.rss / 2 ** 20}
        if hasattr(mi, "peak_wset"):
            out["peak_mb"] = mi.peak_wset / 2 ** 20
        return out
    except Exception:                                      # pragma: no cover
        return {}


def history(out: str) -> List[dict]:
    p = os.path.join(out, "history.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def shares(h: List[dict]) -> dict:
    """Game and learner-transition shares of fixed-opponent games (50% of games need not be 50% of transitions: only
    the learner's seat is a sample there)."""
    tr = sum(e["agent_transitions"] - (h[i - 1]["agent_transitions"] if i else 0) for i, e in enumerate(h))
    hist = sum(e["rollout_transitions_historical"] for e in h)
    games_c = sum(e["games_current"] for e in h)
    games_h = sum(e["games_historical"] for e in h)
    turns = sum(e["rollout_env_turns"] for e in h)
    return {"agent_transitions": int(tr), "learner_transitions_fixed_opponent": int(hist),
            "learner_transition_share_fixed_opponent": hist / tr if tr else float("nan"),
            "games_finished_self_play": int(games_c), "games_finished_fixed_opponent": int(games_h),
            "game_share_fixed_opponent": games_h / (games_c + games_h) if games_c + games_h else float("nan"),
            "env_turns": int(turns), "transitions_per_env_turn": tr / turns if turns else float("nan")}


def train_job(args) -> dict:
    """(phase, arm, seed, total, stop or None, overwrite, out_dir or None, ppo_seed or None[, monitor]): one run under
    `health.RunMonitor` (unless monitor is False); writes summary.json into the run directory; returns the summary (an
    error record instead of raising: status "failed", or "stopped" for a DEFECT stop condition)."""
    phase, arm, seed, total, stop, overwrite, out, ppo_seed, *rest = args
    monitor = rest[0] if rest else True
    out = out or C.run_dir(arm, seed, phase)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(1)
        from ....env import LaneEnv
        from ....rl import checkpoint_info, train_shared, weights_digest
        from ..m43b.train import summarize_history
        from ..m45c.train import rollout_summary
        from .health import RunMonitor
        cfg = C.setup(arm)
        ppo = ppo_config(arm, seed, total, ppo_seed)
        net = initial_net(arm, seed)
        init_digest = weights_digest(net.state_dict())
        res = train_shared(cfg, ppo, shared_config(), out_dir=out, net=net, name=f"m5_{arm}_s{seed}",
                           verbose=phase == "main", overwrite=overwrite, stop_at=stop, rollout_stats=Stats(arm, ppo),
                           collector_cls=collector_cls(arm), on_update=RunMonitor(out) if monitor else None)
        end_epoch = time.time()
        env = LaneEnv(cfg, max_turns=1)
        o = env.reset(seed=0)
        with torch.no_grad():
            pad = np.zeros(net.obs_size - len(o[0].vector), dtype=np.float32)
            if arm == "om":
                from ....opponent import prior_features
                pad = prior_features(C.K)
            v0 = [float(res.net.value(torch.as_tensor(np.concatenate([o[s].vector, pad])).unsqueeze(0))[0])
                  / ppo.reward_scale for s in (0, 1)]
        h = res.history
        s = summarize_history(h, ppo.reward_scale)
        s.pop("curve", None)
        info = checkpoint_info(os.path.join(out, "final.pt"))
        s.update({"status": "complete", "name": f"{arm}_s{seed}", "phase": phase, "arm": arm, "seed": seed,
                  "schedule_total_steps": total, "stop_at": stop, "budget": stop or total, "run_dir": out,
                  "agent_transitions": res.agent_transitions, "env_turns": res.env_turns, "updates": len(h),
                  "elapsed_s": res.elapsed, "transitions_per_s": res.agent_transitions / res.elapsed,
                  "env_turns_per_s": res.env_turns / res.elapsed,
                  "steady_transitions_per_s": float(np.median([e["sps"] for e in h[len(h) // 3:]])) if h else None,
                  "value_s0_resources": v0, "snapshots": len(res.snapshot_paths),
                  "digest": weights_digest(res.net.state_dict()), "initial_digest": init_digest,
                  "initial_source": C.start_path(seed), "ppo": ppo.to_dict(), "shared": shared_config().to_dict(),
                  "memory": _peak_memory_mb(), "rollout": rollout_summary(h), "shares": shares(h),
                  "starts": (info.get("metadata") or {}).get("starts"), "checkpoint_digest": info["digest"],
                  "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "wall_s": time.perf_counter() - t0, "pid": os.getpid(),
                  "train_start_epoch": end_epoch - res.elapsed, "train_end_epoch": end_epoch})
    except BaseException as ex:                            # noqa: BLE001 - recorded, never swallowed silently
        import traceback
        from .health import StopCondition
        s = {"status": "stopped" if isinstance(ex, StopCondition) else "failed", "name": f"{arm}_s{seed}",
             "phase": phase, "arm": arm, "seed": seed, "run_dir": out, "error": repr(ex),
             "traceback": traceback.format_exc(), "started": started, "wall_s": time.perf_counter() - t0,
             "updates_written": len(history(out))}
    if os.path.isdir(out):
        C.jdump(os.path.join(out, "summary.json"), s)
    return s


# ---------------------------------------------------------------------------------------------- manifests
def run_manifest(phase: str, arms: Sequence[str], seeds: Sequence[int], total: int) -> dict:
    """What a training stage will run: configuration, code revision, checkpoint and archive identities, seeds and the
    pre-registration hash (None before freezing)."""
    from ..m46c.archive import archive_path
    pre = C.load("preregistration")
    starts = {s: C.checkpoint_identity(C.start_path(s)) for s in seeds}
    lib = {}
    if "om" in arms:
        from ..om2 import common as O2
        for n, p in O2.LEARNED.items():
            lib[n] = C.checkpoint_identity(p)
    ap = archive_path()
    return {"meta": C.meta(stage="manifest", phase=phase), "phase": phase, "arms": list(arms), "seeds": list(seeds),
            "budget": int(total), "spec": training_spec(), "code": C.code_digest(),
            "start_checkpoints": starts, "library_networks": lib,
            "archive": {"path": ap.replace("\\", "/"), "file_sha256": C.file_sha256(ap)},
            "ppo_seeds": {s: C.TRAIN_SEED + s for s in seeds},
            "preregistration": (pre or {}).get("record"),
            "preregistration_active": C.active_version(pre) if pre else None,
            "run_dirs": {f"{a}_s{s}": C.run_dir(a, s, phase).replace("\\", "/") for s in seeds for a in arms}}


# ---------------------------------------------------------------------------------------------- restarts
def _complete(d: str) -> bool:
    p = os.path.join(d, "summary.json")
    return os.path.exists(p) and (C.jload(p) or {}).get("status") == "complete"


def preserve_partial(d: str, phase: str, name: str) -> str:
    """Move an incomplete run directory, intact, under train/<phase>/partial/ (never deleted); returns the new path."""
    dst = os.path.join(C.CKPT, "train", phase, "partial", f"{name}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(d, dst)
    rec = C.load("restarts") or {"restarts": []}
    rec["restarts"].append({"run": name, "phase": phase, "moved_to": dst.replace("\\", "/"),
                            "updates_preserved": len(history(dst)), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    C.save_json("restarts", rec)
    return dst


def prefix_check(partial_dir: str, run_dir: str) -> dict:
    """A restarted run must reproduce the preserved partial history exactly on its deterministic fields."""
    hp, hr = history(partial_dir), history(run_dir)
    n = min(len(hp), len(hr))
    bad = [i for i in range(n) if any(hp[i].get(k) != hr[i].get(k) for k in DETERMINISTIC_KEYS if k in hp[i])]
    return {"compared_updates": n, "partial_updates": len(hp), "mismatches": bad[:10], "identical": not bad and n == len(hp)}


def plan_runs(phase: str, arms: Sequence[str], seeds: Sequence[int], restart_incomplete: bool = False) -> List[tuple]:
    """The runs still to do: complete runs are skipped; an incomplete directory is refused unless restart_incomplete,
    in which case it is preserved under partial/ first."""
    todo = []
    for s in seeds:
        for a in arms:
            d = C.run_dir(a, s, phase)
            if _complete(d):
                continue
            if os.path.isdir(d) and os.listdir(d):
                if not restart_incomplete:
                    raise SystemExit(f"{d} exists and is incomplete: inspect it, then pass --restart-incomplete to "
                                     "preserve it under partial/ and restart the run (runs are never overwritten)")
                preserve_partial(d, phase, f"{a}_s{s}")
            todo.append((phase, a, s, C.BUDGET, None, False, None, None))
    return todo


def run_order(todo: List[tuple], order: str) -> List[tuple]:
    """"om_first": the longer om runs first, so that the sp runs fill the tail; "interleaved": om, sp, om, sp, ... in
    pair order (each pair's two runs start together). The budget (`bench.budget`) compares both."""
    if order == "om_first":
        return sorted(todo, key=lambda j: (j[1] != "om", j[2]))
    if order == "interleaved":
        return sorted(todo, key=lambda j: (j[2], j[1] != "om"))
    raise ValueError(order)


def train(workers: int, arms: Sequence[str] = C.ARMS, seeds: Sequence[int] = C.SEEDS,
          restart_incomplete: bool = False, order: str = "om_first") -> dict:
    """The M5b campaign (12 runs) under the coordinator's failure policy (`health.run_campaign`), with the progress
    log; then the health check. The caller has checked the user's approval (--confirm-m5-train)."""
    C.require_prereg()
    from ..m46c.train import ResourceLog
    from . import health as HL
    todo = plan_runs("main", arms, seeds, restart_incomplete)
    man = run_manifest("main", arms, seeds, C.BUDGET)
    todo = run_order(todo, order)
    names = [f"{j[1]}_s{j[2]}" for j in todo]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    man["run_order"], man["workers"] = names, workers
    C.save_json(f"manifest_main_{stamp}", man)
    log = ResourceLog(os.path.join(C.OUT, "resources_main.jsonl"))
    prog = HL.Progress("main", names, workers, C.BUDGET)
    log.start()
    prog.start()
    t0 = time.perf_counter()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        out = HL.run_campaign(train_job, todo, names, workers)
    finally:
        log.stop_flag.set()
        prog.stop_flag.set()
        try:
            prog.write()
        except Exception:                                  # pragma: no cover
            pass
    rec = {"manifest": f"manifest_main_{stamp}.json", "runs": names, "workers": workers, "started": started,
           "wall_s": time.perf_counter() - t0, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
           "status": {n: r.get("status") for n, r in out["results"].items()}, "stopped": out["stopped"],
           "run_wall_h": {n: (r.get("wall_s") or 0) / 3600 for n, r in out["results"].items()}}
    C.jdump(os.path.join(C.OUT, f"wall_main_{stamp}.json"), rec)
    C.jdump(os.path.join(C.OUT, "wall_main.json"), rec)
    return check("main")


# ---------------------------------------------------------------------------------------------- health checks
def identities(arm: str, seed: int, phase: str = "main", points: Sequence[int] = C.CURVE_POINTS) -> Dict[str, dict]:
    """File identities (path, SHA-256, transitions) of a run's final and of the snapshots the evaluation reads: the
    curve points (which include the 25%, 50% = matched midpoint and 75% snapshots)."""
    out = {}
    p = C.policy_path(f"{'' if phase == 'main' else phase + '/'}{arm}:{seed}")
    if os.path.exists(p):
        out["final"] = {"path": p.replace("\\", "/"), "file_sha256": C.file_sha256(p)}
    for n in points:
        if n >= C.BUDGET:
            continue
        try:
            q = C.snapshot_at(arm, seed, n, phase)
        except FileNotFoundError:
            continue
        out[f"@{n}"] = {"path": q.replace("\\", "/"), "file_sha256": C.file_sha256(q),
                        "transitions": int(os.path.basename(q)[5:-3])}
    return out


def restart_checks(phase: str = "main") -> Dict[str, dict]:
    """Every restart of this phase (results/m5/restarts.json): the restarted run must reproduce the preserved prefix."""
    out = {}
    for rec in (C.load("restarts") or {}).get("restarts", []):
        if rec.get("phase") != phase:
            continue
        arm, s = rec["run"].split("_s")
        out[f"{rec['run']} ({rec['moved_to'].split('/')[-1]})"] = prefix_check(rec["moved_to"],
                                                                                 C.run_dir(arm, int(s), phase))
    return out


def check(phase: str = "main", total: int = C.BUDGET, seeds: Sequence[int] = C.SEEDS, save: bool = True) -> dict:
    """The pre-registered technical health checks (pre-registration §3), rebuilt from the run directories; with the
    runtime monitor's records, the restart prefix checks, the technical-failure summary and the identities of every
    checkpoint the evaluation will read (amendment 1)."""
    from ....env.discounted import tail_constant
    from ..m45c.full import stability_scan
    from . import health as HL
    runs, checks, scans, ids, live = {}, {}, {}, {}, {}
    for s in seeds:
        for a in C.ARMS:
            name = f"{a}_s{s}"
            d = C.run_dir(a, s, phase)
            r = C.jload(os.path.join(d, "summary.json"))
            h = history(d)
            scans[name] = stability_scan(h)
            live[name] = HL.alarms(d)
            if r is None or r.get("status") != "complete":
                runs[name] = {"status": (r or {}).get("status") or ("missing" if not h else f"incomplete ({len(h)})"),
                              "error": (r or {}).get("error"), "updates_written": len(h),
                              "alarms": [x["alarm"] for x in live[name]]}
                continue
            runs[name] = r
            ids[name] = identities(a, s, phase)
            checks[f"{name}: runtime monitor agrees with the post-hoc scan"] = \
                sorted(x["alarm"] for x in live[name]) == sorted(scans[name].get("alarms", []))
            omega = tail_constant(C.arm_cfg(a))
            sh = r["shares"]
            st = r.get("starts") or {}
            share = st.get("archive_transition_share_finished", float("nan"))
            one_update = 2 * 16 * 128
            checks.update({
                f"{name}: finite losses and gradients": bool(r["finite_losses_and_gradients"]),
                f"{name}: budget reached (within one update)": total <= r["agent_transitions"] < total + one_update,
                f"{name}: never terminated; one truncation per finished game": r["terminations"] == 0 and
                r["truncations"] == r["games_finished"],
                f"{name}: critic bounded (|V| <= Omega)": r["value_absmax_resources"] <= omega,
                f"{name}: no illegal action sampled": r["rollout"].get("illegal_sampled_total", 1) == 0,
                f"{name}: initial weights = the arm's start": r["initial_digest"] == start_digest(a, s),
                f"{name}: archive transition share 0.20-0.30": bool(0.20 <= share <= 0.30),
                f"{name}: matched checkpoints present (16.384M, 24.576M)": _has_snapshot(a, s, phase, C.MIDPOINT)
                and _has_snapshot(a, s, phase, C.P75) if total >= C.BUDGET else True,
                f"{name}: no pre-registered stop condition": not scans[name]["alarms"]})
            if a == "sp":
                checks[f"{name}: 100% current-policy (no fixed-opponent transition)"] = r["transitions_historical"] == 0
                checks[f"{name}: both seats trained equally"] = r["transitions_seat_a"] == r["transitions_seat_b"]
            else:
                checks[f"{name}: fixed-opponent game share 0.45-0.55"] = 0.45 <= sh["game_share_fixed_opponent"] <= 0.55
    for k, v in restart_checks(phase).items():
        checks[f"restart {k}: reproduces the preserved history prefix"] = bool(v["identical"])
    complete = [n for n, r in runs.items() if r.get("status") == "complete"]
    tech = {n: r for n, r in runs.items() if r.get("status") != "complete"}
    fp = sorted({int(n.split("_s")[1]) for n in tech})
    data = {"meta": C.meta(stage="train_check", phase=phase), "runs": runs, "stability": scans, "checks": checks,
            "live_alarms": {n: v for n, v in live.items() if v}, "complete": complete,
            "technical_failures": {"runs": {n: {"status": r["status"], "error": r.get("error"),
                                                "alarms": r.get("alarms")} for n, r in tech.items()},
                                   "pairs": fp, "policy_stop": len(fp) > HL.MAX_FAILED_PAIRS},
            "restarts": restart_checks(phase), "identities": ids,
            "failed_checks": [k for k, v in checks.items() if not v],
            "all_passed": bool(checks) and all(checks.values()) and len(complete) == len(C.ARMS) * len(seeds)}
    if save:
        C.save_json(f"train_{phase}", data)
    return data


def _has_snapshot(arm: str, seed: int, phase: str, n: int) -> bool:
    try:
        p = C.snapshot_at(arm, seed, n, phase)
    except FileNotFoundError:
        return False
    return int(os.path.basename(p)[5:-3]) < n + 2 * 16 * 128


_DIGESTS: Dict[tuple, str] = {}


def start_digest(arm: str, seed: int) -> str:
    from ....rl import weights_digest
    if (arm, seed) not in _DIGESTS:
        _DIGESTS[(arm, seed)] = weights_digest(initial_net(arm, seed).state_dict())
    return _DIGESTS[(arm, seed)]


def status(phase: str = "main", seeds: Sequence[int] = C.SEEDS) -> dict:
    from ..m45c.full import stability_scan
    rows = {}
    for s in seeds:
        for a in C.ARMS:
            h = history(C.run_dir(a, s, phase))
            name = f"{a}_s{s}"
            if not h:
                print(f"  {name:8s} not started", flush=True)
                continue
            e = h[-1]
            tps = float(np.mean([x["sps"] for x in h[-20:]]))
            overall = e["agent_transitions"] / e["elapsed"]
            eta = (C.BUDGET - e["agent_transitions"]) / max(overall, 1.0)
            scan = stability_scan(h)
            rows[name] = {"updates": e["update"], "transitions": e["agent_transitions"], "eta_s": eta,
                          "alarms": scan["alarms"]}
            print(f"  {name:8s} upd {e['update']:5d} {e['agent_transitions'] / 1e6:6.2f}M {tps:5.0f} tps ETA "
                  f"{eta / 3600:5.2f} h ent {e['entropy']:.2f} kl {e['approx_kl']:.4f} ev {e['explained_variance']:.2f} "
                  f"alarms {scan['alarms'] or '-'}", flush=True)
    return rows


__all__ = ["ppo_config", "shared_config", "collector_cls", "initial_net", "training_spec", "Stats", "train_job",
           "history", "shares", "run_manifest", "preserve_partial", "prefix_check", "plan_runs", "train", "check",
           "status", "start_digest", "run_order", "SMOKE_UPDATES", "identities", "restart_checks"]
