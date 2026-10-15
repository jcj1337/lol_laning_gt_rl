"""Training stages of M4.5c: the smoke test, the pilot, re-runs for reproducibility and the throughput benchmarks.

Nothing of the algorithm, the network or the environment is changed: `lanegame.rl.train_shared` with
`common.ppo_config` / `common.shared_config` on `GameConfig.m45_calibrated()`. Two opt-in arguments of train_shared are
used: `stop_at` (the pilot is a prefix of a 4,096,000-transition schedule) and `rollout_stats` (`RolloutStats`: action,
legal-mask, seat and episode diagnostics of every rollout, merged into history.jsonl; no effect on training, tested).

Phases (results/rl/checkpoints/m45c/<phase>/, git-ignored): "smoke" may overwrite its own runs; "pilot" REFUSES to
overwrite an existing run directory; "bench" runs keep no checkpoints. The full campaign ("full") is NOT run by this
package without the explicit confirmation flag of the command line (`train_full`), which is reserved for after the
budget review.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import actions as AC
from .common import (CKPT, OUT, PILOT_SCHEDULE, PILOT_SEED, PILOT_STOP, SMOKE_SEED, SMOKE_STEPS,
                     TRANSITIONS_PER_UPDATE, env_cfg, load_json, meta, ppo_config, run_dir, run_jobs, run_name,
                     save_json, shared_config, training_spec)

WALL_KEYS = ("elapsed", "sps", "collect_s", "update_s")


# ---------------------------------------------------------------------------------------------- rollout diagnostics
class RolloutStats:
    """`train_shared(rollout_stats=...)`: per-rollout diagnostics (read only).

    masks     mean number of legal codes per learner decision, share of single-choice decisions (absent), shares of
              decisions with Ward / Recall / Pressure legal; `illegal_sampled` (sampled codes outside their recorded
              mask) and `never_legal_offered` (masks offering a code outside actions.EVER_LEGAL) must both be 0
    actions   category / behaviour / Pressure shares of the sampled codes (actions.histogram), also per seat
              (`seatB_minus_A_<cat>`: in self-play the two seats are one policy, so this is sampling noise only)
    seats     transitions, summed reward (antisymmetric in current-policy games: reward_sum_A + reward_sum_B = 0),
              advantage mean / std of each seat after GAE (both seats must carry policy-gradient signal)
    episodes  finished games: turns, deaths, recalls, interrupted recalls, trade attempts and connections, ganks,
              farm collection rate, plates, final levels (per side, summed over the rollout's finished games)
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, roll: dict, finished: list) -> dict:
        from ....rl import compute_gae
        a = roll["active"]
        acts = roll["actions"][a]
        masks = roll["mask"][a]
        n_legal = masks.sum(-1)
        out = {"mask_n_legal_mean": float(n_legal.mean()), "mask_single_choice_share": float((n_legal <= 1).mean()),
               "mask_ward_legal_share": float(masks[:, AC.WARD].mean()),
               "mask_recall_legal_share": float(masks[:, AC.RECALL].mean()),
               "mask_pressure_legal_share": float(masks[:, AC.PRESSURE_OF].any(-1).mean()),
               "illegal_sampled": int((~masks[np.arange(len(acts)), acts]).sum()),
               "never_legal_offered": int(masks[:, sorted(AC.NEVER_LEGAL)].sum())}
        out.update({f"act_{k}": v for k, v in AC.histogram(acts).items()})
        ha = AC.histogram(roll["actions"][:, 0::2][a[:, 0::2]])
        hb = AC.histogram(roll["actions"][:, 1::2][a[:, 1::2]])
        for c in AC.CATEGORIES:
            out[f"seatB_minus_A_cat_{c}"] = hb.get(f"cat_{c}", 0.0) - ha.get(f"cat_{c}", 0.0)
        adv, _ = compute_gae(roll["rewards"], roll["values"], roll["next_values"], roll["terminated"],
                             roll["truncated"], self.cfg.gamma, self.cfg.gae_lambda)
        for s, key in ((0, "A"), (1, "B")):
            m = a[:, s::2]
            out[f"reward_sum_{key}"] = float(roll["rewards"][:, s::2][m].sum())
            out[f"adv_mean_{key}"] = float(adv[:, s::2][m].mean())
            out[f"adv_std_{key}"] = float(adv[:, s::2][m].std())
        ep = {"games": len(finished), "turns": 0, "deaths": 0, "recalls": 0, "recalls_interrupted": 0,
              "trades_attempted": 0, "trades_connected": 0, "ganks": 0, "farm": 0.0, "farm_available": 0.0,
              "plates": 0.0, "level_sum": 0.0, "level_n": 0}
        for e in finished:
            lg = e.log
            ep["turns"] += len(lg.rewards)
            for k in ("deaths", "recalls", "recalls_interrupted", "trades_attempted", "trades_connected", "ganks"):
                ep[k] += int(sum(getattr(lg, k)))
            ep["farm"] += float(sum(lg.farm))
            ep["farm_available"] += float(sum(lg.farm_available))
            ep["plates"] += float(sum(lg.plates))
            levels = (lg.m4_final or {}).get("levels")
            if levels:
                ep["level_sum"] += float(sum(levels))
                ep["level_n"] += len(levels)
        out.update({f"ep_{k}": v for k, v in ep.items()})
        return out


# ---------------------------------------------------------------------------------------------- one run
def _peak_memory_mb() -> Dict[str, float]:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        out = {"rss_mb": mi.rss / 2 ** 20}
        if hasattr(mi, "peak_wset"):
            out["peak_mb"] = mi.peak_wset / 2 ** 20                 # Windows: peak working set of the process
        return out
    except Exception:                                      # pragma: no cover - psutil missing
        return {}


def train_one(args: tuple) -> dict:
    """One run in a worker process: (phase, seed, total_steps, stop_at or None, out_dir or None, overwrite, tag)."""
    phase, seed, total, stop, out_dir, overwrite, tag = args
    import torch
    from ....env import LaneEnv
    from ....rl import checkpoint_info, train_shared, weights_digest
    from ..m43b.train import summarize_history
    cfg = env_cfg()
    ppo = ppo_config(seed, total)
    name = run_name(seed, tag)
    t0 = time.perf_counter()
    res = train_shared(cfg, ppo, shared_config(), out_dir=out_dir, name=name, verbose=phase != "bench",
                       overwrite=overwrite, stop_at=stop, rollout_stats=RolloutStats(ppo))
    elapsed = time.perf_counter() - t0
    env = LaneEnv(cfg, max_turns=1)
    o = env.reset(seed=0)
    with torch.no_grad():
        v0 = [float(res.net.value(torch.as_tensor(o[s].vector).unsqueeze(0))[0]) / ppo.reward_scale for s in (0, 1)]
    s = summarize_history(res.history, ppo.reward_scale)
    s.update({"name": name, "phase": phase, "seed": seed, "schedule_total_steps": total, "stop_at": stop,
              "budget": stop or total, "run_dir": out_dir, "agent_transitions": res.agent_transitions,
              "env_turns": res.env_turns, "updates": len(res.history), "elapsed_s": elapsed,
              "transitions_per_s": res.agent_transitions / res.elapsed, "env_turns_per_s": res.env_turns / res.elapsed,
              "value_s0_resources": v0, "snapshots": len(res.snapshot_paths),
              "digest": weights_digest(res.net.state_dict()), "initial_digest": initial_digest(seed),
              "ppo": ppo.to_dict(), "shared": shared_config().to_dict(), "memory": _peak_memory_mb(),
              "rollout": rollout_summary(res.history)})
    if out_dir:
        info = checkpoint_info(os.path.join(out_dir, "final.pt"))
        s.update({"checkpoint_digest": info["digest"], "provenance": info["provenance"]})
    return s


def initial_digest(seed: int) -> str:
    from ....rl import weights_digest
    from .common import initial_agent
    return weights_digest(initial_agent(seed).net.state_dict())


def rollout_summary(h: List[dict]) -> dict:
    """Totals and first / last-10% means of the RolloutStats fields of a history."""
    if not h or "illegal_sampled" not in h[0]:
        return {}
    n = max(1, len(h) // 10)
    keys = [k for k in h[0] if k.startswith(("mask_", "act_", "seatB_minus_A_", "adv_"))]
    out = {"illegal_sampled_total": int(sum(e["illegal_sampled"] for e in h)),
           "never_legal_offered_total": int(sum(e["never_legal_offered"] for e in h)),
           "reward_antisymmetry_max_abs": float(max(abs(e["reward_sum_A"] + e["reward_sum_B"]) for e in h)),
           "first10": {k: float(np.nanmean([e[k] for e in h[:n]])) for k in keys},
           "last10": {k: float(np.nanmean([e[k] for e in h[-n:]])) for k in keys}}
    ep = {k: int(sum(e[f"ep_{k}"] for e in h)) for k in ("games", "turns", "deaths", "recalls", "recalls_interrupted",
                                                          "trades_attempted", "trades_connected", "ganks")}
    out["episodes_total"] = ep
    return out


# ---------------------------------------------------------------------------------------------- checks
def pipeline_checks(s: dict, omega: float) -> Dict[str, bool]:
    """Conditions that must hold for any budget (a failure is a defect)."""
    r = s.get("rollout", {})
    n = s["name"]
    return {
        f"{n}: finite losses, gradients and critic statistics": bool(s["finite_losses_and_gradients"]),
        f"{n}: never terminated; one truncation per finished game": bool(
            s["terminations"] == 0 and s["truncations"] == s["games_finished"]),
        f"{n}: critic bounded by the return scale (|V| <= Omega)": bool(s["value_absmax_resources"] <= omega),
        f"{n}: gradient norms of order 1 (max < 100)": bool(s["grad_norm_max"] < 100),
        f"{n}: approximate KL small (mean < 0.05)": bool(s["approx_kl_mean"] < 0.05),
        f"{n}: both seats trained equally (exact)": bool(s["transitions_seat_a"] == s["transitions_seat_b"]),
        f"{n}: 2 agent transitions per joint step (100% current-policy)": bool(
            s["agent_transitions"] == 2 * s["env_turns"] and s["transitions_historical"] == 0),
        f"{n}: budget reached exactly": bool(s["agent_transitions"] == s["budget"]),
        f"{n}: no illegal action sampled": bool(r.get("illegal_sampled_total", 1) == 0),
        f"{n}: masks never offer a never-legal code": bool(r.get("never_legal_offered_total", 1) == 0),
        f"{n}: rewards exactly antisymmetric between seats": bool(r.get("reward_antisymmetry_max_abs", 1) < 1e-9),
        f"{n}: both seats see the same vector at the symmetric start (V_A(s0) = V_B(s0))": bool(
            s["value_s0_resources"][0] == s["value_s0_resources"][1]),
    }


def learning_checks(s: dict) -> Dict[str, bool]:
    """Conditions of a critic and a policy that have had time to learn (informational at smoke budgets)."""
    n = s["name"]
    return {f"{n}: symmetric start valued near 0 (|V(s0)| <= 0.5 resources)": bool(
                max(abs(v) for v in s["value_s0_resources"]) <= 0.5),
            f"{n}: explained variance positive at the end": bool(s["explained_variance_first_last10"][1] > 0),
            f"{n}: entropy decreased from its initial value": bool(s["entropy_first_last"][1] < s["entropy_first_last"][0])}


def _omega() -> float:
    from ....env.discounted import tail_constant
    return tail_constant(env_cfg())


# ---------------------------------------------------------------------------------------------- stages
def smoke(workers: int = 2) -> dict:
    """The correctness smoke run (SMOKE_STEPS on its own schedule) and its bit-identical re-run, in parallel."""
    jobs = [("smoke", SMOKE_SEED, SMOKE_STEPS, None, run_dir("smoke", SMOKE_SEED), True, ""),
            ("smoke", SMOKE_SEED, SMOKE_STEPS, None, run_dir("smoke", SMOKE_SEED, "_repeat"), True, "_repeat")]
    t0 = time.perf_counter()
    a, b = run_jobs(train_one, jobs, workers, "smoke")
    from ..m44.train import _histories_equal
    omega = _omega()
    checks = pipeline_checks(a, omega)
    checks["smoke re-run: bit-identical weights"] = a["digest"] == b["digest"]
    checks["smoke re-run: identical histories (every non-wall-clock field)"] = _histories_equal(a["run_dir"], b["run_dir"])
    data = {"meta": meta(stage="smoke"), "spec": training_spec(), "runs": {"s100": a, "s100_repeat": b},
            "checks": checks, "all_passed": all(checks.values()), "learning": learning_checks(a),
            "wall_s": time.perf_counter() - t0}
    save_json("smoke_train", data)
    return data


def pilot(confirm_overwrite: bool = False) -> dict:
    """THE pilot run (one process): seed PILOT_SEED, schedule PILOT_SCHEDULE, stopped at PILOT_STOP."""
    out = run_dir("pilot", PILOT_SEED)
    t0 = time.perf_counter()
    s = train_one(("pilot", PILOT_SEED, PILOT_SCHEDULE, PILOT_STOP, out, confirm_overwrite, ""))
    checks = pipeline_checks(s, _omega())
    data = {"meta": meta(stage="pilot"), "spec": training_spec(), "run": s, "checks": checks,
            "all_passed": all(checks.values()), "learning": learning_checks(s), "wall_s": time.perf_counter() - t0}
    save_json("pilot_train", data)
    return data


def prefix_check(updates: int = 7, workers: int = 2) -> dict:
    """The pilot's first `updates` updates re-trained in a fresh directory (`<run>_prefix`): its weights and history
    must equal the pilot's snapshot / history at that point bit for bit (determinism of the pilot itself, at a
    fraction of its cost; 7 updates = the first snapshot, 28,672)."""
    from ....rl import checkpoint_info
    from .common import history, snapshot_paths
    stop = updates * TRANSITIONS_PER_UPDATE
    out = run_dir("pilot", PILOT_SEED, "_prefix")
    s = train_one(("pilot", PILOT_SEED, PILOT_SCHEDULE, stop, out, True, "_prefix"))
    h_pilot = [{k: v for k, v in e.items() if k not in WALL_KEYS} for e in history("pilot", PILOT_SEED)[:updates]]
    h_pre = [{k: v for k, v in e.items() if k not in WALL_KEYS} for e in history("pilot", PILOT_SEED, "_prefix")]
    import json
    snaps = dict(snapshot_paths("pilot", PILOT_SEED))
    snap_digest = None
    for n, p in snaps.items():
        if n == stop:
            snap_digest = checkpoint_info(p)["digest"]
    data = {"meta": meta(stage="prefix_check"), "updates": updates, "stop_at": stop,
            "identical_histories": json.dumps(h_pilot, sort_keys=True) == json.dumps(h_pre, sort_keys=True),
            "prefix_digest": s["digest"], "pilot_snapshot_digest_at_stop": snap_digest,
            "identical_weights_to_pilot_snapshot": snap_digest == s["digest"] if snap_digest else None}
    save_json("pilot_prefix_check", data)
    return data


# ---------------------------------------------------------------------------------------------- benchmarks
def _bench_job(args) -> dict:
    """(seed, stop_at): a checkpoint-free run on the pilot schedule; per-update throughput from its history."""
    seed, stop = args
    s = train_one(("bench", seed, PILOT_SCHEDULE, stop, None, False, ""))
    return {"seed": seed, "transitions_per_s": s["transitions_per_s"], "env_turns_per_s": s["env_turns_per_s"],
            "collect_s": s["collect_s"], "update_s": s["update_s"], "elapsed_s": s["elapsed_s"],
            "memory": s["memory"], "agent_transitions": s["agent_transitions"]}


def scaling(procs: Sequence[int] = (1, 3, 6, 11), updates: int = 8) -> dict:
    """Per-process and aggregate training throughput with n concurrent training processes (short checkpoint-free runs
    of the pilot schedule from the untrained network; the slowdown factor relative to 1 process is what the
    campaign estimate applies to the pilot's measured single-process throughput)."""
    out = {}
    for n in procs:
        t0 = time.perf_counter()
        res = run_jobs(_bench_job, [(300 + i, updates * TRANSITIONS_PER_UPDATE) for i in range(n)], n, f"scaling {n}")
        wall = time.perf_counter() - t0
        tps = [r["transitions_per_s"] for r in res]
        out[str(n)] = {"processes": n, "per_process_tps": tps, "per_process_mean": float(np.mean(tps)),
                       "aggregate_tps": float(sum(r["agent_transitions"] for r in res) / wall), "wall_s": wall,
                       "peak_mb": [r["memory"].get("peak_mb") for r in res]}
        print(f"  {n} processes: {np.mean(tps):,.0f} transitions/s per process, "
              f"{out[str(n)]['aggregate_tps']:,.0f} aggregate", flush=True)
    base = out[str(procs[0])]["per_process_mean"]
    for v in out.values():
        v["slowdown_vs_1"] = v["per_process_mean"] / base
    data = {"meta": meta(stage="scaling", updates=updates), "scaling": out}
    save_json("bench_scaling", data)
    return data


def profile(spec: str = f"final:pilot:{PILOT_SEED}", rollouts: int = 4) -> dict:
    """Breakdown of one training update with the given network: environment stepping (LaneEnv.step), network
    inference (forward / value passes during collection), the rest of collection (buffers, logs, Python), GAE + batch
    assembly, the PPO update, and a checkpoint save. Instruments the classes in this process only."""
    import tempfile
    import torch
    from ....env import LaneEnv
    from ....rl import MatchupSampler, PolicyValueNet, SharedCollector, SnapshotPool, ppo_update, save_checkpoint
    from ....rl.shared import rollout_batch
    from .common import agent_from_spec
    torch.set_num_threads(1)
    cfg = env_cfg()
    ppo = ppo_config(PILOT_SEED, PILOT_SCHEDULE)
    agent = agent_from_spec(spec, cfg)
    net = PolicyValueNet(agent.net.obs_size, agent.net.n_actions, agent.net.hidden)
    net.load_state_dict(agent.net.state_dict())
    net.eval()
    timers = {"env_step": 0.0, "inference": 0.0}
    step0, fwd0, val0 = LaneEnv.step, PolicyValueNet.forward, PolicyValueNet.value

    def step(self, *a, **k):
        t = time.perf_counter()
        r = step0(self, *a, **k)
        timers["env_step"] += time.perf_counter() - t
        return r

    def fwd(self, *a, **k):
        t = time.perf_counter()
        r = fwd0(self, *a, **k)
        timers["inference"] += time.perf_counter() - t
        return r

    def val(self, *a, **k):
        t = time.perf_counter()
        r = val0(self, *a, **k)
        timers["inference"] += time.perf_counter() - t
        return r

    rows = []
    col = SharedCollector(cfg, ppo, MatchupSampler(SnapshotPool(8), 1.0), seed=ppo.seed + 1)
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(net.parameters(), lr=0.0, eps=1e-5)           # lr 0: the update is timed, not applied
    LaneEnv.step, PolicyValueNet.forward, PolicyValueNet.value = step, fwd, val
    try:
        for i in range(rollouts + 1):
            timers.update(env_step=0.0, inference=0.0)
            t = time.perf_counter()
            roll, finished, counts = col.collect(net, gen)
            t_col = time.perf_counter() - t
            env_s, inf_s = timers["env_step"], timers["inference"]
            t = time.perf_counter()
            batch, _, _ = rollout_batch(roll, ppo)
            t_gae = time.perf_counter() - t
            LaneEnv.step, PolicyValueNet.forward, PolicyValueNet.value = step0, fwd0, val0
            t = time.perf_counter()
            net.train()
            ppo_update(net, opt, batch, ppo, gen)
            net.eval()
            t_upd = time.perf_counter() - t
            LaneEnv.step, PolicyValueNet.forward, PolicyValueNet.value = step, fwd, val
            if i > 0:                                                    # the first rollout warms the lane cache
                rows.append({"collect_s": t_col, "env_step_s": env_s, "inference_s": inf_s,
                             "collect_other_s": t_col - env_s - inf_s, "gae_batch_s": t_gae, "update_s": t_upd,
                             "agent_transitions": counts["agent_transitions"], "env_turns": counts["env_turns"]})
    finally:
        LaneEnv.step, PolicyValueNet.forward, PolicyValueNet.value = step0, fwd0, val0
    with tempfile.TemporaryDirectory() as d:
        t = time.perf_counter()
        for k in range(5):
            save_checkpoint(os.path.join(d, f"c{k}.pt"), net, cfg, {"bench": k})
        t_ckpt = (time.perf_counter() - t) / 5
        size = os.path.getsize(os.path.join(d, "c0.pt"))
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    total = m["collect_s"] + m["gae_batch_s"] + m["update_s"]
    m.update({"total_s": total, "checkpoint_save_s": t_ckpt, "checkpoint_bytes": size,
              "env_turns_per_s_stepping_only": m["env_turns"] / m["env_step_s"],
              "agent_transitions_per_s_end_to_end": m["agent_transitions"] / total,
              "shares": {k: m[k] / total for k in ("env_step_s", "inference_s", "collect_other_s", "gae_batch_s",
                                                    "update_s")}})
    data = {"meta": meta(stage="profile", spec=spec, rollouts=rollouts), "per_update": m, "rows": rows,
            "memory": _peak_memory_mb()}
    save_json(f"bench_profile_{spec.split(':')[0]}", data)
    return data


def _cache_job(args) -> dict:
    """(lane_cache, rollouts, spec): collection only (no updates) with the given network; RSS, cache fill and collection
    time every 10 rollouts. The lane-step cache is an exact memo (LaneEnv._lane_turn), so its size changes speed and
    memory, never results."""
    lane_cache, rollouts, spec = args
    import torch
    from ....rl import MatchupSampler, SharedCollector, SnapshotPool
    from .common import agent_from_spec
    torch.set_num_threads(1)
    cfg = env_cfg()
    ppo = ppo_config(PILOT_SEED, PILOT_SCHEDULE, lane_cache=lane_cache)
    net = agent_from_spec(spec, cfg).net
    col = SharedCollector(cfg, ppo, MatchupSampler(SnapshotPool(8), 1.0), seed=ppo.seed + 1)
    gen = torch.Generator().manual_seed(0)
    rows, t_block = [], 0.0
    for i in range(1, rollouts + 1):
        t = time.perf_counter()
        col.collect(net, gen)
        t_block += time.perf_counter() - t
        if i % 10 == 0:
            fill = [len(e._cache) for c in col._env_cache for e in c.values()]
            rows.append({"rollouts": i, "joint_steps_per_env": i * ppo.rollout_length,
                         "collect_s_per_rollout": t_block / 10, "cache_entries_mean": float(np.mean(fill)),
                         **_peak_memory_mb()})
            t_block = 0.0
    return {"lane_cache": lane_cache, "rows": rows}


def cache_probe(sizes: Sequence[int] = (20_000, 5_000, 2_000), rollouts: int = 160,
                spec: str = f"final:pilot:{PILOT_SEED}") -> dict:
    """Memory and collection speed against the lane-cache size, with the pilot's final network (concurrent processes,
    one per size)."""
    res = run_jobs(_cache_job, [(s, rollouts, spec) for s in sizes], len(sizes), "cache probe")
    data = {"meta": meta(stage="cache_probe", spec=spec, rollouts=rollouts), "probe": res}
    save_json("bench_cache", data)
    return data


__all__ = ["RolloutStats", "train_one", "pipeline_checks", "learning_checks", "smoke", "pilot", "prefix_check",
           "scaling", "profile", "cache_probe", "rollout_summary", "initial_digest"]
