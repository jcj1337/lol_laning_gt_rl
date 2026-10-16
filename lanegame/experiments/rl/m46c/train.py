"""Training of the M4.6c pilot: matched continuations of the six M4.5c 16.384M finals under the baseline and C1.

Every setting is M4.5c's (`lanegame.rl.train_shared`, 100% current-policy shared self-play, gamma 0.99, GAE 0.95,
segments of 128 turns, s0 games of 512 turns, reward scale 0.1, value clip 0.5, 4 epochs, minibatch 256, clip 0.2,
entropy 0.03, gradient norm 0.5, 16 environments, 2 x 128 tanh MLP, snapshots every 25,600 transitions, lane memo
5,000), with three stated differences that are the SAME in both arms:
    * initialization: the weights of the M4.5c final of the same seed (`start:<s>`); the optimizer is a fresh Adam
      (Adam moments are not stored in checkpoints), the learning rate 3e-4 annealed linearly to 0 over the run's
      budget (as every earlier warm-started run: the M4.5c and M4.6a attackers);
    * the start distribution: the mixed collector (collector.py; archive.py), about 25% of transitions in games
      started from the shared archive;
    * the PPO seed TRAIN_SEED + s (both arms of a seed share it: the same staggered first lengths, start draws and
      chance seeds until the trajectories diverge).
The only difference between the arms is the investment curve of the game configuration.

Outputs: results/rl/checkpoints/m46c/train/<phase>/<arm>/s<seed>/ (run.json, history.jsonl, snapshots/, final.pt,
summary.json; git-ignored); records in results/m46c/. Runs never overwrite an existing directory except in the smoke
phase. Each worker writes its summary as soon as its run ends and returns an error record instead of raising.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import common as C

PHASES = ("smoke", "main", "ext")


def ppo_config(arm: str, seed: int, total: int, **kw):
    from ....rl import PPOConfig
    base = dict(total_steps=total, n_envs=C.N_ENVS, seed=C.TRAIN_SEED + seed, torch_threads=1, ent_coef=C.ENT_COEF,
                hidden=C.HIDDEN, lane_cache=C.LANE_CACHE, game_length=C.GAME_LENGTH, start_mix=0.0)
    base.update(kw)
    return PPOConfig.continuing(C.arm_cfg(arm), **base)


def shared_config():
    from ....rl import SharedConfig
    return SharedConfig(current_fraction=C.CURRENT_FRACTION, snapshot_every=C.SNAP_EVERY, pool_size=C.POOL)


def collector(beta: float = C.ARCHIVE_BETA):
    from .archive import archive_path
    from .collector import mixed_collector_cls
    return mixed_collector_cls(archive_path(), beta, C.ARCHIVE_LENGTH, C.MIRROR_P)


def training_spec() -> dict:
    p = ppo_config("baseline", 0, C.BUDGET).to_dict()
    for k in ("seed", "total_steps"):
        p.pop(k)
    return {"arms": {a: {"preset": "GameConfig.m45_calibrated()", "overrides": C.arm_overrides(a)} for a in C.ARMS},
            "ppo": p, "shared": shared_config().to_dict(), "initialization": "weights of start:<s> (M4.5c ext final), "
            "fresh Adam, lr 3e-4 annealed linearly to 0 over the budget", "budget": C.BUDGET, "midpoint": C.MIDPOINT,
            "starts": {"beta": C.ARCHIVE_BETA, "archive_length": C.ARCHIVE_LENGTH, "game_length": C.GAME_LENGTH,
                       "mirror_p": C.MIRROR_P, "target_transition_share": C.ARCHIVE_TARGET_SHARE},
            "seeds": {"train": f"{C.TRAIN_SEED} + s, the same in both arms"}}


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


def train_job(args) -> dict:
    """(phase, arm, seed, total, stop or None, overwrite[, beta]): one continuation run; writes <run>/summary.json."""
    phase, arm, seed, total, stop, overwrite, *rest = args
    beta = rest[0] if rest else C.ARCHIVE_BETA
    out = C.run_dir(arm, seed, phase)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(1)
        from ....env import LaneEnv
        from ....rl import PolicyValueNet, checkpoint_info, train_shared, weights_digest
        from ..m43b.train import summarize_history
        from ..m45c.train import RolloutStats
        cfg = C.arm_cfg(arm)
        ppo = ppo_config(arm, seed, total)
        src = C.net_of(f"start:{seed}")
        net = PolicyValueNet(src.obs_size, src.n_actions, src.hidden)
        net.load_state_dict(src.state_dict())
        init_digest = weights_digest(net.state_dict())
        res = train_shared(cfg, ppo, shared_config(), out_dir=out, net=net, name=f"m46c_{arm}_s{seed}",
                           verbose=phase != "bench", overwrite=overwrite, stop_at=stop, rollout_stats=RolloutStats(ppo),
                           collector_cls=collector(beta))
        env = LaneEnv(cfg, max_turns=1)
        o = env.reset(seed=0)
        with torch.no_grad():
            v0 = [float(res.net.value(torch.as_tensor(o[s].vector).unsqueeze(0))[0]) / ppo.reward_scale for s in (0, 1)]
        s = summarize_history(res.history, ppo.reward_scale)
        info = checkpoint_info(os.path.join(out, "final.pt"))
        starts = (info.get("metadata") or {}).get("starts") or _starts_from_final(out)
        from ..m45c.train import rollout_summary
        s.update({"status": "complete", "name": f"{arm}_s{seed}", "phase": phase, "arm": arm, "seed": seed,
                  "schedule_total_steps": total, "stop_at": stop, "budget": stop or total, "run_dir": out,
                  "agent_transitions": res.agent_transitions, "env_turns": res.env_turns, "updates": len(res.history),
                  "elapsed_s": res.elapsed, "transitions_per_s": res.agent_transitions / res.elapsed,
                  "value_s0_resources": v0, "snapshots": len(res.snapshot_paths),
                  "digest": weights_digest(res.net.state_dict()), "initial_digest": init_digest,
                  "start_checkpoint": C.start_path(seed), "ppo": ppo.to_dict(), "shared": shared_config().to_dict(),
                  "memory": _peak_memory_mb(), "rollout": rollout_summary(res.history), "starts": starts,
                  "checkpoint_digest": info["digest"], "started": started,
                  "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": time.perf_counter() - t0,
                  "pid": os.getpid()})
    except BaseException as ex:                            # noqa: BLE001 - recorded, never swallowed silently
        import traceback
        s = {"status": "failed", "name": f"{arm}_s{seed}", "phase": phase, "arm": arm, "seed": seed, "run_dir": out,
             "error": repr(ex), "traceback": traceback.format_exc(), "started": started,
             "wall_s": time.perf_counter() - t0}
    if os.path.isdir(out):
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1, default=float)
    return s


def _starts_from_final(out: str) -> Optional[dict]:
    try:
        from ....rl import load_checkpoint
        _, _, m = load_checkpoint(os.path.join(out, "final.pt"))
        return m.get("starts")
    except Exception:                                      # pragma: no cover
        return None


# ---------------------------------------------------------------------------------------------- resource log
class ResourceLog(threading.Thread):
    def __init__(self, path: str, every: float = 30.0):
        super().__init__(daemon=True)
        self.path, self.every, self.stop_flag = path, every, threading.Event()

    def run(self) -> None:
        try:
            import psutil
        except ImportError:                                # pragma: no cover
            return
        me = psutil.Process()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            while not self.stop_flag.is_set():
                vm, sw = psutil.virtual_memory(), psutil.swap_memory()
                kids = []
                for c in me.children(recursive=True):
                    try:
                        kids.append(c.memory_info().rss / 2 ** 20)
                    except Exception:                      # pragma: no cover
                        pass
                f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "avail_gib": vm.available / 2 ** 30,
                                    "swap_used_gib": sw.used / 2 ** 30, "cpu_percent": psutil.cpu_percent(),
                                    "children_rss_mb": sorted(kids)}) + "\n")
                f.flush()
                self.stop_flag.wait(self.every)


# ---------------------------------------------------------------------------------------------- stages
SMOKE_UPDATES = 10


def smoke(workers: int = 3) -> dict:
    """Pipeline check: both arms of seed 0 for SMOKE_UPDATES updates of the real 4,096,000 schedule (stop_at, so the
    learning rate is the pilot's), plus a bit-identical re-run of the C1 arm. Checks pipeline health only."""
    from ..m44.train import _histories_equal
    from ..m45c.full import stability_scan
    stop = SMOKE_UPDATES * C.TRANSITIONS_PER_UPDATE
    js = [("smoke", "baseline", 0, C.BUDGET, stop, True), ("smoke", "C1", 0, C.BUDGET, stop, True)]
    t0 = time.perf_counter()
    res = C.run_jobs(train_job, js, workers, "M4.6c smoke")
    rep_dir = C.run_dir("C1", 0, "smoke") + "_repeat"
    rep = _repeat_job(("smoke", "C1", 0, C.BUDGET, stop, rep_dir))
    out = {"meta": C.meta(stage="smoke"), "spec": training_spec(), "runs": {r["name"]: r for r in res},
           "repeat": rep, "wall_s": time.perf_counter() - t0}
    checks = {}
    for r in res:
        n = r["name"]
        if r.get("status") != "complete":
            checks[f"{n}: completed"] = False
            continue
        checks[f"{n}: finite losses and gradients"] = bool(r["finite_losses_and_gradients"])
        checks[f"{n}: {SMOKE_UPDATES} updates, {stop:,} transitions"] = r["agent_transitions"] == stop
        checks[f"{n}: both seats trained equally"] = r["transitions_seat_a"] == r["transitions_seat_b"]
        checks[f"{n}: 100% current-policy"] = r["transitions_historical"] == 0
        checks[f"{n}: no illegal action sampled"] = r["rollout"].get("illegal_sampled_total", 1) == 0
        checks[f"{n}: approximate KL mean < 0.05"] = r["approx_kl_mean"] < 0.05
        checks[f"{n}: initial weights = start:0"] = r["initial_digest"] == _digest_of("start:0")
        st = r.get("starts") or {}
        checks[f"{n}: archive games started"] = st.get("games_started", {}).get("archive", 0) > 0
        scan = stability_scan(_history(r["run_dir"]))
        checks[f"{n}: no stop condition"] = not scan["alarms"]
    a = next((r for r in res if r["name"] == "C1_s0"), None)
    if a and rep.get("digest"):
        checks["C1 re-run: bit-identical weights"] = a["digest"] == rep["digest"]
        checks["C1 re-run: identical histories"] = _histories_equal(a["run_dir"], rep_dir)
    out["checks"] = checks
    out["all_passed"] = all(checks.values())
    C.save_json("smoke_train", out)
    return out


def _repeat_job(args) -> dict:
    phase, arm, seed, total, stop, out = args
    import torch
    torch.set_num_threads(1)
    from ....rl import PolicyValueNet, train_shared, weights_digest
    from ..m45c.train import RolloutStats
    ppo = ppo_config(arm, seed, total)
    src = C.net_of(f"start:{seed}")
    net = PolicyValueNet(src.obs_size, src.n_actions, src.hidden)
    net.load_state_dict(src.state_dict())
    res = train_shared(C.arm_cfg(arm), ppo, shared_config(), out_dir=out, net=net, name=f"m46c_{arm}_s{seed}",
                       verbose=False, overwrite=True, stop_at=stop, rollout_stats=RolloutStats(ppo),
                       collector_cls=collector())
    return {"digest": weights_digest(res.net.state_dict()), "run_dir": out}


def _digest_of(spec: str) -> str:
    from ....rl import weights_digest
    return weights_digest(C.net_of(spec).state_dict())


def _history(run_dir: str) -> List[dict]:
    p = os.path.join(run_dir, "history.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


PHASE_BUDGET = {"main": C.BUDGET, "ext": C.EXT_BUDGET}


def train(arms: Sequence[str] = C.ARMS, seeds: Sequence[int] = C.SEEDS, workers: int = 6, phase: str = "main",
          total: Optional[int] = None, label: Optional[str] = None) -> dict:
    """The runs of a phase, concurrently (one process each); refuses existing run directories."""
    C.require_prereg()
    total = total or PHASE_BUDGET[phase]
    label = label or phase
    js = [(phase, a, s, total, None, False) for s in seeds for a in arms]
    todo = []
    for j in js:
        d = C.run_dir(j[1], j[2], phase)
        if os.path.exists(os.path.join(d, "summary.json")):
            with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
                if json.load(f).get("status") == "complete":
                    continue
        if os.path.isdir(d) and os.listdir(d):
            raise SystemExit(f"{d} exists and is incomplete: inspect it; runs are never overwritten")
        todo.append(j)
    log = ResourceLog(os.path.join(C.OUT, f"resources_{label}.jsonl"))
    log.start()
    t0 = time.perf_counter()
    print(f"[m46c train:{label}] {len(todo)} runs, {workers} concurrent: {[f'{j[1]}_s{j[2]}' for j in todo]}", flush=True)
    try:
        res = C.run_jobs(train_job, todo, workers, f"m46c train {label}")
    finally:
        log.stop_flag.set()
    for r in res:
        print(f"[m46c train:{label}] {r['name']}: {r['status']}", flush=True)
    C.jdump(os.path.join(C.OUT, f"wall_{label}.json"), {"label": label, "runs": [r["name"] for r in res],
                                                         "wall_s": time.perf_counter() - t0,
                                                         "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    return check(phase=phase, total=total)


def _summary(arm: str, seed: int, phase: str) -> Optional[dict]:
    p = os.path.join(C.run_dir(arm, seed, phase), "summary.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def check(phase: str = "main", total: Optional[int] = None, save: bool = True) -> dict:
    """The phase record rebuilt from the run directories: status, pipeline checks, the stability scan, the start mix."""
    total = total or PHASE_BUDGET[phase]
    from ..m45c.full import stability_scan
    from ....env.discounted import tail_constant
    runs, checks, scans = {}, {}, {}
    for s in C.SEEDS:
        for a in C.ARMS:
            name = f"{a}_s{s}"
            r = _summary(a, s, phase)
            h = _history(C.run_dir(a, s, phase))
            scans[name] = stability_scan(h)
            if r is None:
                runs[name] = {"status": "missing" if not h else f"incomplete ({len(h)} updates)"}
                continue
            runs[name] = {k: v for k, v in r.items() if k not in ("curve",)}
            if r.get("status") != "complete":
                continue
            omega = tail_constant(C.arm_cfg(a))
            st = r.get("starts") or {}
            share = st.get("archive_transition_share_finished", float("nan"))
            checks.update({
                f"{name}: finite losses and gradients": bool(r["finite_losses_and_gradients"]),
                f"{name}: budget reached exactly ({total:,})": r["agent_transitions"] == total,
                f"{name}: never terminated; one truncation per finished game": r["terminations"] == 0 and
                r["truncations"] == r["games_finished"],
                f"{name}: critic bounded (|V| <= Omega)": r["value_absmax_resources"] <= omega,
                f"{name}: both seats trained equally": r["transitions_seat_a"] == r["transitions_seat_b"],
                f"{name}: 100% current-policy": r["transitions_historical"] == 0,
                f"{name}: no illegal action sampled": r["rollout"].get("illegal_sampled_total", 1) == 0,
                f"{name}: initial weights = start:{s}": r["initial_digest"] == _digest_of(f"start:{s}"),
                f"{name}: archive transition share 0.20-0.30": bool(0.20 <= share <= 0.30),
                f"{name}: midpoint snapshot present": os.path.exists(C.snapshot_path(a, s, C.MIDPOINT, phase))
                if total == C.BUDGET else True,
                f"{name}: no pre-registered stop condition": not scans[name]["alarms"]})
    complete = [n for n, r in runs.items() if r.get("status") == "complete"]
    data = {"meta": C.meta(stage="train_check", phase=phase), "spec": training_spec(), "runs": runs,
            "stability": scans, "checks": checks, "complete": complete,
            "all_passed": bool(checks) and all(checks.values()) and len(complete) == len(C.ARMS) * len(C.SEEDS)}
    if save:
        C.save_json(f"train_{phase}", data)
    return data


def status(phase: str = "main") -> dict:
    from ..m45c.full import stability_scan
    rows = {}
    for s in C.SEEDS:
        for a in C.ARMS:
            h = _history(C.run_dir(a, s, phase))
            name = f"{a}_s{s}"
            if not h:
                print(f"  {name:12s} not started", flush=True)
                continue
            e = h[-1]
            tps = float(np.mean([x["sps"] for x in h[-20:]]))
            overall = e["agent_transitions"] / e["elapsed"]
            eta = (PHASE_BUDGET.get(phase, C.BUDGET) - e["agent_transitions"]) / max(overall, 1.0)
            scan = stability_scan(h)
            rows[name] = {"updates": e["update"], "transitions": e["agent_transitions"], "eta_s": eta,
                          "alarms": scan["alarms"]}
            print(f"  {name:12s} upd {e['update']:4d} {e['agent_transitions'] / 1e6:5.2f}M {tps:5.0f} tps ETA "
                  f"{eta / 60:5.1f} min ent {e['entropy']:.2f} kl {e['approx_kl']:.4f} ev {e['explained_variance']:.2f} "
                  f"g {e['grad_norm']:.2f} lr {e['lr']:.1e} alarms {scan['alarms'] or '-'}", flush=True)
    return rows


__all__ = ["ppo_config", "shared_config", "collector", "training_spec", "train_job", "smoke", "train", "check",
           "status", "PHASES"]
