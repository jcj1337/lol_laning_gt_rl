"""Training stage of the M4.3b pilot: one shared-policy self-play run per (condition, seed), in parallel worker
processes, with the PPO health summary of each run (the M4.3a audit quantities) written to results/rl/m43b/train.json.

Nothing of the algorithm is changed: `lanegame.rl.train_shared` with `PPOConfig.continuing` (gamma = the game's 0.99,
lambda 0.95, segments of 128 turns, games of 512 turns with seeded staggered first games, reward scale 0.1, value clip
0.5, learning rate 3e-4 annealed, 4 epochs, minibatch 256, clip 0.2, 16 environments). The conditions differ only in
the entropy coefficient and the current / historical fraction (common.CONDITIONS).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .common import (BUDGET, CONDITIONS, SEEDS, cfg_game, history, load, meta, ppo_config, read_json, run_dir, run_jobs,
                     run_name, save_json, shared_config, snapshot_paths)

HEALTH_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "grad_norm", "grad_norm_max",
               "value_mean", "value_absmax", "return_mean", "return_std", "adv_std")


def _last(h: List[dict], key: str, n: int = 10) -> float:
    return float(np.nanmean([e[key] for e in h[-n:]]))


def summarize_history(h: List[dict], reward_scale: float) -> dict:
    """The M4.3a health quantities of one run's history (critic scales converted to resources)."""
    rs = reward_scale
    return {
        "updates": len(h),
        "finite_losses_and_gradients": all(np.isfinite(e[k]) for e in h for k in HEALTH_KEYS),
        "grad_norm_mean": float(np.mean([e["grad_norm"] for e in h])),
        "grad_norm_max": float(np.max([e["grad_norm_max"] for e in h])),
        "clipfrac_mean": float(np.mean([e["clipfrac"] for e in h])),
        "approx_kl_mean": float(np.mean([e["approx_kl"] for e in h])),
        "approx_kl_max": float(np.max([e["approx_kl"] for e in h])),
        "entropy_first_last": [h[0]["entropy"], _last(h, "entropy")],
        "explained_variance_first_last10": [h[0]["explained_variance"], _last(h, "explained_variance")],
        "value_loss_first_last10": [h[0]["value_loss"], _last(h, "value_loss")],
        "value_absmax_resources": float(np.max([e["value_absmax"] for e in h])) / rs,
        "value_mean_last10_resources": _last(h, "value_mean") / rs,
        "return_std_last10_resources": _last(h, "return_std") / rs,
        "reward_mean_abs_resources": float(np.mean([e["reward_mean_abs"] for e in h])) / rs,
        "transitions_seat_a": int(sum(e["rollout_transitions_seat_a"] for e in h)),
        "transitions_seat_b": int(sum(e["rollout_transitions_seat_b"] for e in h)),
        "transitions_historical": int(sum(e["rollout_transitions_historical"] for e in h)),
        "single_choice_transitions": int(sum(e["rollout_single_choice"] for e in h)),
        "truncations": int(sum(e["rollout_truncations"] for e in h)),
        "terminations": int(sum(e["rollout_terminations"] for e in h)),
        "games_finished": int(sum(e["games_current"] + e["games_historical"] for e in h)),
        "games_historical": int(sum(e["games_historical"] for e in h)),
        "collect_s": float(sum(e["collect_s"] for e in h)), "update_s": float(sum(e["update_s"] for e in h)),
        # downsampled curves over updates: transitions, entropy, approx KL, explained variance, grad norm, V mean, lr
        "curve": [[e["agent_transitions"], e["entropy"], e["approx_kl"], e["explained_variance"], e["grad_norm"],
                   e["value_mean"] / rs, e["clipfrac"], e["value_loss"]] for e in h[::max(1, len(h) // 100)]],
    }


def train_one(args: tuple) -> dict:
    """One run in a worker process (returns a JSON-able summary; no tensors cross the process boundary)."""
    cond, seed, steps = args
    import torch
    from ....env import LaneEnv
    from ....rl import checkpoint_info, train_shared, weights_digest
    cfg = cfg_game()
    ppo = ppo_config(cond, seed, steps)
    scfg = shared_config(cond)
    out = run_dir(cond, seed)
    name = run_name(cond, seed)
    t0 = time.perf_counter()
    res = train_shared(cfg, ppo, scfg, out_dir=out, name=name, verbose=True, overwrite=True)
    elapsed = time.perf_counter() - t0
    env = LaneEnv(cfg, max_turns=1)
    o = env.reset(seed=0)[0]
    with torch.no_grad():
        v0 = float(res.net.value(torch.as_tensor(o.vector).unsqueeze(0))[0]) / ppo.reward_scale
    info = checkpoint_info(os.path.join(out, "final.pt"))
    s = summarize_history(res.history, ppo.reward_scale)
    s.update({"name": name, "condition": cond, "seed": seed, "budget": steps, "run_dir": out,
              "agent_transitions": res.agent_transitions, "env_turns": res.env_turns, "elapsed_s": elapsed,
              "transitions_per_s": res.agent_transitions / res.elapsed, "env_turns_per_s": res.env_turns / res.elapsed,
              "value_s0_resources": v0, "snapshots": len(res.snapshot_paths),
              "digest": weights_digest(res.net.state_dict()), "provenance": info["provenance"],
              "ppo": ppo.to_dict(), "shared": scfg.to_dict()})
    return s


def train(conds: Sequence[str] = tuple(CONDITIONS), seeds: Sequence[int] = SEEDS, steps: int = BUDGET,
          workers: int = 9) -> dict:
    jobs = [(c, s, steps) for c in conds for s in seeds]
    os.makedirs(os.path.dirname(run_dir(conds[0], seeds[0])), exist_ok=True)
    print(f"training {len(jobs)} runs of {steps:,} agent transitions on {min(workers, len(jobs))} processes", flush=True)
    t0 = time.perf_counter()

    def progress(i, s):
        print(f"  {s['name']}: {s['agent_transitions']:,} transitions in {s['elapsed_s']:.0f} s "
              f"({s['transitions_per_s']:,.0f}/s), entropy {s['entropy_first_last'][0]:.2f} -> "
              f"{s['entropy_first_last'][1]:.2f}, ev {s['explained_variance_first_last10'][1]:.2f}, "
              f"V(s0) {s['value_s0_resources']:+.2f}", flush=True)

    runs = run_jobs(train_one, jobs, workers, progress=progress)
    prev = load("train") or {}
    merged = dict(prev.get("runs", {}))                       # earlier conditions' runs are kept (merged by name)
    merged.update({r["name"]: r for r in runs})
    data = {"meta": meta(stage="train", budget=steps, conditions=CONDITIONS, seeds=list(seeds), workers=workers),
            "elapsed_s": time.perf_counter() - t0, "runs": merged,
            "stages": prev.get("stages", []) + [{"conditions": list(conds), "seeds": list(seeds), "steps": steps,
                                                 "workers": workers, "elapsed_s": time.perf_counter() - t0}]}
    data["checks"] = health_checks(data["runs"])
    save_json("train", data)
    return data


def summary_from_run_dir(cond: str, seed: int) -> dict:
    """The `train_one` summary rebuilt from a run directory (history.jsonl, run.json, final.pt); wall-clock figures
    come from the history's elapsed times."""
    import torch
    from ....env import LaneEnv
    from ....rl import PPOConfig, checkpoint_info, load_checkpoint, weights_digest
    out = run_dir(cond, seed)
    h = history(cond, seed)
    run = read_json(os.path.join(out, "run.json"))
    ppo = PPOConfig.from_dict(run["ppo"])
    net, cfg, _ = load_checkpoint(os.path.join(out, "final.pt"))
    env = LaneEnv(cfg, max_turns=1)
    o = env.reset(seed=0)[0]
    with torch.no_grad():
        v0 = float(net.value(torch.as_tensor(o.vector).unsqueeze(0))[0]) / ppo.reward_scale
    info = checkpoint_info(os.path.join(out, "final.pt"))
    s = summarize_history(h, ppo.reward_scale)
    last = h[-1]
    s.update({"name": run_name(cond, seed), "condition": cond, "seed": seed, "budget": ppo.total_steps, "run_dir": out,
              "agent_transitions": last["agent_transitions"], "env_turns": last["env_turns"], "elapsed_s": last["elapsed"],
              "transitions_per_s": last["agent_transitions"] / last["elapsed"],
              "env_turns_per_s": last["env_turns"] / last["elapsed"], "value_s0_resources": v0,
              "snapshots": len(snapshot_paths(cond, seed)), "digest": weights_digest(net.state_dict()),
              "provenance": info["provenance"], "ppo": ppo.to_dict(), "shared": run["shared"], "rebuilt": True})
    return s


def rebuild(conds: Sequence[str] = tuple(CONDITIONS), seeds: Sequence[int] = SEEDS) -> dict:
    """Rebuild train.json from the run directories that exist (keeps entries that were recorded by `train`)."""
    prev = load("train") or {}
    runs = dict(prev.get("runs", {}))
    for c in conds:
        for s in seeds:
            if os.path.exists(os.path.join(run_dir(c, s), "final.pt")) and run_name(c, s) not in runs:
                runs[run_name(c, s)] = summary_from_run_dir(c, s)
    data = {"meta": prev.get("meta", meta(stage="train")), "elapsed_s": prev.get("elapsed_s"), "runs": runs,
            "stages": prev.get("stages", []), "checks": health_checks(runs)}
    save_json("train", data)
    return data


def health_checks(runs: Dict[str, dict]) -> dict:
    """The M4.3a smoke conditions applied to every run."""
    from ....env.discounted import tail_constant
    omega = tail_constant(cfg_game())
    checks = {}
    for n, s in runs.items():
        checks[f"{n}: finite losses, gradients and critic statistics"] = bool(s["finite_losses_and_gradients"])
        checks[f"{n}: never terminated; one truncation per finished game"] = bool(
            s["terminations"] == 0 and s["truncations"] == s["games_finished"])
        checks[f"{n}: critic bounded by the return scale (|V| <= Omega)"] = bool(s["value_absmax_resources"] <= omega)
        checks[f"{n}: explained variance positive at the end"] = bool(s["explained_variance_first_last10"][1] > 0)
        checks[f"{n}: gradient norms of order 1 (max < 100)"] = bool(s["grad_norm_max"] < 100)
        checks[f"{n}: approximate KL small (mean < 0.05)"] = bool(s["approx_kl_mean"] < 0.05)
        if s["condition"] != "hist80":
            checks[f"{n}: both seats trained equally (exact)"] = bool(s["transitions_seat_a"] == s["transitions_seat_b"])
    return {"checks": checks, "all_passed": all(checks.values())}


def benchmark(seconds: float = 30.0, workers: int = 1) -> dict:
    """Throughput of one process (short run of the baseline configuration) for the budget estimate."""
    import time as _t
    from ....rl import train_shared
    cfg = cfg_game()
    steps = 16 * 128 * 2 * 20                                 # 20 updates
    ppo = ppo_config("base", 99, steps)
    t0 = _t.perf_counter()
    res = train_shared(cfg, ppo, shared_config("base"), verbose=False, name="bench")
    el = _t.perf_counter() - t0
    sps = res.agent_transitions / el
    out = {"agent_transitions": res.agent_transitions, "elapsed_s": el, "transitions_per_s": sps,
           "env_turns_per_s": res.env_turns / el, "minutes_per_million_single_process": 1e6 / sps / 60,
           "note": "single process; under a 9-process load M4.3a measured about 20% less per process"}
    print(out, flush=True)
    return out


__all__ = ["train", "train_one", "benchmark", "summarize_history", "health_checks", "rebuild", "summary_from_run_dir"]
