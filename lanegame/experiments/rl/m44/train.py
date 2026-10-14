"""Training stage of M4.4: one shared-policy self-play run per (environment, seed), in parallel worker processes,
with the PPO health summary of every run (the M4.3a audit quantities) and its checks.

Nothing of the algorithm or the network is changed: `lanegame.rl.train_shared` with `common.ppo_config` (identical for
every environment) and `common.shared_config`. Phase "smoke" (M4.4a) writes to results/rl/checkpoints/m44/smoke/ and
may overwrite its own runs; phase "main" (M4.4b) writes to results/rl/checkpoints/m44/main/ and REFUSES to overwrite
an existing run directory (`train_shared`'s guard), so a launched M4.4b run is never silently replaced.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .common import (ENV_ORDER, SEEDS, env_cfg, meta, ppo_config, run_dir, run_jobs, run_name, shared_config,
                     training_spec)

HEALTH_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "grad_norm", "grad_norm_max",
               "value_mean", "value_absmax", "return_mean", "return_std", "adv_std")


def train_one(args: tuple) -> dict:
    """One run in a worker process: (phase, environment key, seed, steps, out_dir or None, overwrite)."""
    phase, key, seed, steps, out_dir, overwrite = args
    import torch
    from ....env import LaneEnv
    from ....rl import checkpoint_info, train_shared, weights_digest
    from ..m43b.train import summarize_history
    cfg = env_cfg(key)
    ppo = ppo_config(key, seed, steps)
    out = out_dir or run_dir(phase, key, seed)
    name = run_name(key, seed)
    t0 = time.perf_counter()
    res = train_shared(cfg, ppo, shared_config(), out_dir=out, name=name, verbose=False, overwrite=overwrite)
    elapsed = time.perf_counter() - t0
    env = LaneEnv(cfg, max_turns=1)
    o = env.reset(seed=0)
    with torch.no_grad():
        v0 = [float(res.net.value(torch.as_tensor(o[s].vector).unsqueeze(0))[0]) / ppo.reward_scale for s in (0, 1)]
    info = checkpoint_info(os.path.join(out, "final.pt"))
    s = summarize_history(res.history, ppo.reward_scale)
    s.update({"name": name, "env": key, "phase": phase, "seed": seed, "budget": steps, "run_dir": out,
              "agent_transitions": res.agent_transitions, "env_turns": res.env_turns, "elapsed_s": elapsed,
              "transitions_per_s": res.agent_transitions / res.elapsed, "env_turns_per_s": res.env_turns / res.elapsed,
              "value_s0_resources": v0, "snapshots": len(res.snapshot_paths),
              "digest": weights_digest(res.net.state_dict()), "checkpoint_digest": info["digest"],
              "provenance": info["provenance"], "ppo": ppo.to_dict(), "shared": shared_config().to_dict(),
              "initial_digest": _initial_digest(key, seed)})
    return s


def _initial_digest(key: str, seed: int) -> str:
    """Digest of the untrained network of a training seed (equal across environments: the paired initialization)."""
    import torch
    from ....env import LaneEnv
    from ....rl import PolicyValueNet, weights_digest
    torch.manual_seed(seed)
    env = LaneEnv(env_cfg(key))
    return weights_digest(PolicyValueNet(env.obs_size, env.n_actions, tuple(ppo_config(key, seed).hidden)).state_dict())


def health_checks(runs: Dict[str, dict]) -> dict:
    """The M4.3a smoke conditions on every run (Omega = the environment's guaranteed return bound). "pipeline"
    checks must hold for any budget (a failure is a defect); "learning" checks describe a critic that has had time to
    learn (informational at smoke budgets)."""
    from ....env.discounted import tail_constant
    checks, learning = {}, {}
    for n, s in runs.items():
        omega = tail_constant(env_cfg(s["env"]))
        checks[f"{n}: finite losses, gradients and critic statistics"] = bool(s["finite_losses_and_gradients"])
        checks[f"{n}: never terminated; one truncation per finished game"] = bool(
            s["terminations"] == 0 and s["truncations"] == s["games_finished"])
        checks[f"{n}: critic bounded by the return scale (|V| <= Omega)"] = bool(s["value_absmax_resources"] <= omega)
        checks[f"{n}: gradient norms of order 1 (max < 100)"] = bool(s["grad_norm_max"] < 100)
        checks[f"{n}: approximate KL small (mean < 0.05)"] = bool(s["approx_kl_mean"] < 0.05)
        checks[f"{n}: both seats trained equally (exact)"] = bool(s["transitions_seat_a"] == s["transitions_seat_b"])
        checks[f"{n}: no historical games (100% current-policy self-play)"] = bool(s["transitions_historical"] == 0)
        checks[f"{n}: budget reached exactly"] = bool(s["agent_transitions"] == s["budget"])
        checks[f"{n}: both seats see the same vector at the symmetric start (V_A(s0) = V_B(s0))"] = bool(
            s["value_s0_resources"][0] == s["value_s0_resources"][1])
        # s0 is mirror-symmetric, so a zero-sum game's value there is exactly 0: a trained critic should say so
        learning[f"{n}: symmetric start valued near 0 (|V(s0)| <= 0.5 resources)"] = bool(
            max(abs(v) for v in s["value_s0_resources"]) <= 0.5)
        learning[f"{n}: explained variance positive at the end"] = bool(s["explained_variance_first_last10"][1] > 0)
        learning[f"{n}: entropy decreased from its initial value"] = bool(
            s["entropy_first_last"][1] < s["entropy_first_last"][0])
    return {"checks": checks, "all_passed": all(checks.values()), "learning": learning,
            "learning_all": all(learning.values())}


def _out(phase: str) -> str:
    from .common import OUT_A, OUT_B
    return OUT_A if phase == "smoke" else OUT_B


def train(phase: str, keys: Sequence[str] = ENV_ORDER, seeds: Sequence[int] = SEEDS, steps: Optional[int] = None,
          workers: int = 9) -> dict:
    """Train every (key, seed) of the matrix in parallel (one process per run when workers >= runs: the M4.4b load
    pattern, so the measured per-process throughput is the one the budget estimate needs). Saves train_<phase>.json."""
    from .common import BUDGET, EXT_BUDGET, SMOKE_BUDGET, save_json
    steps = steps or {"smoke": SMOKE_BUDGET, "main": BUDGET, "ext": EXT_BUDGET}[phase]
    jobs = [(phase, k, s, steps, None, phase == "smoke") for k in keys for s in seeds]
    print(f"[m44 {phase}] {len(jobs)} runs of {steps:,} agent transitions on {min(workers, len(jobs))} processes",
          flush=True)
    t0 = time.perf_counter()
    res = run_jobs(train_one, jobs, workers, label=f"train {phase}")
    wall = time.perf_counter() - t0
    runs = {r["name"]: r for r in res}
    init = {}
    for s in seeds:
        ds = {k: runs[run_name(k, s)]["initial_digest"] for k in keys}
        init[f"seed {s}"] = {"digests": ds, "identical_across_environments": len(set(ds.values())) == 1}
    tps = [r["transitions_per_s"] for r in runs.values()]
    data = {"meta": meta(stage=f"train_{phase}", steps=steps, workers=workers, keys=list(keys), seeds=list(seeds),
                         study="M4.4a" if phase == "smoke" else "M4.4b"),
            "spec": training_spec(), "wall_s": wall, "runs": runs, "checks": health_checks(runs),
            "initialization": init,
            "throughput": {"processes": min(workers, len(jobs)),
                           "transitions_per_s_per_process_min_mean_max": [float(np.min(tps)), float(np.mean(tps)),
                                                                           float(np.max(tps))],
                           "aggregate_transitions_per_s": float(sum(r["agent_transitions"] for r in runs.values()) / wall),
                           "wall_s": wall}}
    save_json(f"train_{phase}", data, _out(phase))
    return data


def reproduce(phase: str, pairs: Sequence[tuple], workers: int = 3) -> dict:
    """Train (key, seed) runs a second time into <run_dir>_repeat and compare the final weights digests with the
    recorded runs of train_<phase>.json (bit-identical expected: torch_threads = 1, every stream seeded). The result
    is merged into train_<phase>.json under "reproducibility"."""
    from .common import load_json, save_json
    prev = load_json(f"train_{phase}", _out(phase))
    if prev is None:
        raise RuntimeError(f"run the train stage of phase {phase!r} first")
    steps = prev["meta"]["steps"]
    jobs = [(phase, k, s, steps, run_dir(phase, k, s) + "_repeat", True) for k, s in pairs]
    res = run_jobs(train_one, jobs, workers, label=f"reproduce {phase}")
    repro = {}
    for (k, s), r in zip(pairs, res):
        a = prev["runs"][run_name(k, s)]
        same_hist = _histories_equal(a["run_dir"], r["run_dir"])
        repro[run_name(k, s)] = {"digest": a["digest"], "repeat_digest": r["digest"],
                                 "identical_weights": a["digest"] == r["digest"], "identical_histories": same_hist}
    prev["reproducibility"] = repro
    save_json(f"train_{phase}", prev, _out(phase))
    return repro


def _histories_equal(d1: str, d2: str) -> bool:
    """The two runs' history.jsonl agree on every field except wall-clock ones."""
    import json
    skip = ("elapsed", "sps", "collect_s", "update_s")
    rows = []
    for d in (d1, d2):
        with open(os.path.join(d, "history.jsonl"), encoding="utf-8") as f:
            rows.append([{k: v for k, v in json.loads(line).items() if k not in skip} for line in f if line.strip()])
    return json.dumps(rows[0], sort_keys=True) == json.dumps(rows[1], sort_keys=True)


__all__ = ["train", "reproduce", "train_one", "health_checks", "HEALTH_KEYS"]
