"""OM2 training: the matched continuations of the adopted baseline policy, control and treatment (design §5).

Every run widens `ext/baseline:s` (90 inputs) to NET_IN inputs with zero weights on the new ones (`widen_inputs`): both
arms start as exactly the baseline policy, whatever their extra inputs hold. PPO: the M4.6c settings (as in the M4.6c and
M4.7 continuations), PPO seed TRAIN_SEED + s in both arms, the M4.6c start mixture, fresh Adam at 3e-4 annealed to 0
over the budget, the arm's collector (`collector.om_collector_cls`).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import common as C

SMOKE_UPDATES = 6


def ppo_config(seed: int, total: int):
    from ....rl import PPOConfig
    from ..m46c import common as C46
    return PPOConfig.continuing(C.env_cfg(), total_steps=total, n_envs=C46.N_ENVS, seed=C.TRAIN_SEED + seed,
                                torch_threads=1, ent_coef=C46.ENT_COEF, hidden=C46.HIDDEN, lane_cache=C46.LANE_CACHE,
                                game_length=C46.GAME_LENGTH, start_mix=0.0)


def initial_net(seed: int):
    from ....rl.network import widen_inputs
    return widen_inputs(C.net90(C.start_path(seed)), C.NET_IN)


class Stats:
    """rollout_stats: M4.5c's per-rollout diagnostics plus the OM2 checks (read only): the belief inputs' spread and
    range, the opponents of the rollout's games, and a digest of the sampled actions (to compare the arms' first
    rollouts)."""

    def __init__(self, ppo):
        from ..m45c.train import RolloutStats
        self.base = RolloutStats(ppo)

    def __call__(self, roll: dict, finished: list) -> dict:
        out = self.base(roll, finished)
        a = roll["active"]
        x = roll["obs"][a][:, C.OBS:]
        out.update({"belief_std_mean": float(x.std(0).mean()), "belief_range": float((x.max(0) - x.min(0)).max()),
                    "belief_min": float(x.min()),
                    "belief_max": float(x.max()), "belief_rowsum_dev": float(np.abs(x.sum(1) - 1).max()),
                    "unknown_mean": float(x[:, -1].mean()),
                    "lastaction_none_share": float(roll["obs"][a][:, 90].mean()),
                    "actions_digest": hashlib.sha256(roll["actions"][a].tobytes()).hexdigest()[:16]})
        opp: Dict[str, int] = {}
        for e in finished:
            opp[e.opponent] = opp.get(e.opponent, 0) + 1
        out["finished_by_opponent"] = opp
        return out


def train_job(args) -> dict:
    """(phase, arm, seed, total, stop or None, overwrite, out_dir or None): one run; writes summary.json."""
    phase, arm, seed, total, stop, overwrite, out = args
    out = out or C.run_dir(arm, seed, phase)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(1)
        from ....rl import checkpoint_info, train_shared, weights_digest
        from ....rl.ppo import check_time_semantics
        from ..m46c.train import shared_config
        from .collector import om_collector_cls
        cfg = C.setup()
        ppo = ppo_config(seed, total)
        check_time_semantics(cfg, ppo)
        net = initial_net(seed)
        init_digest = weights_digest(net.state_dict())
        res = train_shared(cfg, ppo, shared_config(), out_dir=out, net=net, name=f"om2_{arm}_s{seed}",
                           verbose=phase != "smoke", overwrite=overwrite, stop_at=stop,
                           rollout_stats=Stats(ppo), collector_cls=om_collector_cls(arm))
        info = checkpoint_info(os.path.join(out, "final.pt"))
        s = {"status": "complete", "name": f"{arm}_s{seed}", "phase": phase, "arm": arm, "seed": seed,
             "agent_transitions": res.agent_transitions, "env_turns": res.env_turns, "updates": len(res.history),
             "elapsed_s": res.elapsed, "transitions_per_s": res.agent_transitions / res.elapsed,
             "digest": weights_digest(res.net.state_dict()), "initial_digest": init_digest,
             "initial_source": C.start_path(seed), "checkpoint_digest": info["digest"],
             "starts": (info.get("metadata") or {}).get("starts"), "ppo": ppo.to_dict(),
             "history_first": res.history[0] if res.history else None,
             "history_last": res.history[-1] if res.history else None, "run_dir": out, "started": started,
             "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": time.perf_counter() - t0}
    except BaseException as ex:                            # noqa: BLE001 - recorded, never swallowed silently
        import traceback
        s = {"status": "failed", "name": f"{arm}_s{seed}", "phase": phase, "error": repr(ex),
             "traceback": traceback.format_exc(), "started": started, "wall_s": time.perf_counter() - t0}
    if os.path.isdir(out):
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1, default=float)
    return s


def _history(out: str) -> List[dict]:
    with open(os.path.join(out, "history.jsonl"), encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


# ---------------------------------------------------------------------------------------------- smoke and readiness
def smoke(workers: int = 3) -> dict:
    """Readiness (design §8, items 3-6): both arms of seed 0 and a treatment re-run, SMOKE_UPDATES updates each."""
    from ..m46c import common as C46
    per_update = C46.TRANSITIONS_PER_UPDATE
    total = C.BUDGET
    stop = SMOKE_UPDATES * per_update
    jobs = [("smoke", "control", 0, total, stop, True, os.path.join(C.CKPT, "smoke", "control")),
            ("smoke", "treatment", 0, total, stop, True, os.path.join(C.CKPT, "smoke", "treatment")),
            ("smoke", "treatment", 0, total, stop, True, os.path.join(C.CKPT, "smoke", "treatment_rerun"))]
    res = dict(zip(("control", "treatment", "rerun"), C.run_jobs(train_job, jobs, workers, "OM2 smoke")))
    checks: Dict[str, bool] = {}
    for k, s in res.items():
        checks[f"{k}: run completed"] = s.get("status") == "complete"
    if not all(checks.values()):
        out = {"meta": C.meta(stage="smoke"), "checks": checks, "runs": res}
        C.save_json("smoke", out)
        return out
    hc = _history(res["control"]["run_dir"])
    ht = _history(res["treatment"]["run_dir"])
    checks["both arms start from the same rollout (identical sampled actions in update 1)"] = \
        hc[0]["actions_digest"] == ht[0]["actions_digest"]
    checks["the arms diverge after the first update"] = any(
        a["actions_digest"] != b["actions_digest"] for a, b in zip(hc[1:], ht[1:]))
    checks["control: belief inputs constant at the prior"] = all(h["belief_range"] == 0.0 for h in hc)
    checks["treatment: belief inputs vary"] = all(h["belief_std_mean"] > 0.0 for h in ht)
    checks["treatment: belief inputs are a posterior (in [0, 1], rows sum to 1)"] = all(
        h["belief_min"] >= 0.0 and h["belief_max"] <= 1.0 + 1e-6 and h["belief_rowsum_dev"] < 1e-4 for h in ht)
    checks["the last-action field shows game starts and previous actions"] = all(
        0.0 < h["lastaction_none_share"] < 0.5 for h in ht)
    opp = {}
    for h in ht + hc:
        for k2, v in h.get("finished_by_opponent", {}).items():
            opp[k2] = opp.get(k2, 0) + v
    lib_draws = (res["treatment"]["starts"] or {}).get("om2", {}).get("library", {}).get("draws", {})
    checks["library opponents drawn"] = sum(v for k2, v in lib_draws.items() if k2.startswith("lib:")) > 0
    checks["self-play games drawn"] = lib_draws.get("self", 0) > 0
    checks["treatment updated beliefs"] = (res["treatment"]["starts"] or {}).get("om2", {}).get("belief_updates", 0) > 0
    checks["no illegal action sampled"] = all(h.get("illegal_sampled", 0) == 0 for h in hc + ht)
    checks["truncations bootstrapped (some games truncated)"] = sum(h["rollout_truncations"] for h in ht) > 0
    checks["treatment re-run bit-identical"] = res["treatment"]["digest"] == res["rerun"]["digest"]
    net0 = initial_net(0)
    from ....rl import weights_digest
    checks["initial weights = widened ext/baseline:0"] = res["control"]["initial_digest"] == \
        res["treatment"]["initial_digest"] == weights_digest(net0.state_dict())
    checks["policy moved (KL > 0, below 0.05)"] = all(0 < h["approx_kl"] < 0.05 for h in ht)
    tp = {k: s["transitions_per_s"] for k, s in res.items()}
    out = {"meta": C.meta(stage="smoke"), "checks": checks, "throughput_transitions_per_s": tp,
           "library_draws": lib_draws, "finished_by_opponent": opp,
           "runs": {k: {kk: v for kk, v in s.items() if kk not in ("history_first", "history_last")}
                    for k, s in res.items()}}
    C.save_json("smoke", out)
    return out


def train(workers: int, seeds: Sequence[int] = C.PILOT_SEEDS) -> dict:
    """The pilot runs (PILOT_SEEDS x 2 arms, BUDGET each); the caller has checked the user's approval."""
    C.require_prereg()
    jobs = [("main", arm, s, C.BUDGET, None, False, None) for s in seeds for arm in C.ARMS]
    res = C.run_jobs(train_job, jobs, workers, "OM2 pilot training")
    out = {"meta": C.meta(stage="train"), "runs": res, "all_complete": all(r.get("status") == "complete" for r in res)}
    C.save_json("train_main", out)
    return out


__all__ = ["ppo_config", "initial_net", "Stats", "train_job", "smoke", "train", "SMOKE_UPDATES"]
