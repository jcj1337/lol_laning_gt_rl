"""Learned best-response attackers on the M4.5c finals: exploitability LOWER bounds (the former M4.4d; the M4.3b
protocol, `m43b/exploit.py`). PREPARED AND SMOKE-TESTED ONLY: `run()` refuses to start without its confirmation flag,
which belongs to the comprehensive-evaluation approval.

Targets: the 16.384M finals of the matched 2x extension (phase "ext", the predesignated primary policy set; approved
2026-10-15) with the FIXED budget ATTACK_BUDGET_F = 4,096,000 learner transitions, which is one QUARTER of the extended
target's 16,384,000, not half: a deviation from the M4.3b half-budget protocol, reported as a limitation.

Protocol: one attacker per final, warm-started from the target's own weights (it starts at parity, so a gain measures
the target's exploitability rather than the attacker's learning speed); `train_ppo` against the frozen target only
(an `OpponentSampler` holding the target; the learner draws its seat uniformly per game; only the learner's seat is
trained, so agent transitions = joint steps), with the M4.5c PPO settings (`common.ppo_config`, lane cache 5,000);
budget ATTACK_BUDGET_F = half the target's 8,192,000. Evaluated against its target (64 paired seeds, W = 460 and
W = 1000) and against the other five finals (transfer, W = 1000). The payoff is a lower bound limited by the
attacker's budget, network and optimizer, never a certificate.

Seeds (reserved campaign range): attacker training 94,000 + training seed; evaluation 94,100.
Checkpoints: results/rl/checkpoints/m45c/attack/s<seed>/ (git-ignored); record results/m45c/attackers/attackers.json.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Sequence

import numpy as np

from .common import CKPT, OUT, agent_from_spec, env_cfg, final_path, mean_ci, meta, ppo_config, run_jobs, save_json

AOUT = os.path.join(OUT, "attackers")
ATTACK_BUDGET_F = 4_096_000
ATTACK_TRAIN_SEED_F, ATTACK_EVAL_SEED_F = 94_000, 94_100
ATTACK_EPISODES, W_DEV, W_HEAD = 64, 460, 1000
CONFIRM_FLAG = "--confirm-m45c-attackers"


def attacker_dir(seed: int, root: str = CKPT, phase: str = "ext") -> str:
    return os.path.join(root, "attack", f"{phase}_s{seed}")


def train_attacker(target_spec: str, steps: int, train_seed: int, out_dir: Optional[str], verbose: bool = False,
                   **ppo_kw):
    """(attacker NetAgent, history, target agent, PPOConfig): PPO against the frozen target, warm-started from it."""
    from ....rl import NetAgent, OpponentSampler, PolicyValueNet, train_ppo
    cfg = env_cfg()
    target = agent_from_spec(target_spec, cfg)
    net = PolicyValueNet(target.net.obs_size, target.net.n_actions, target.net.hidden)
    net.load_state_dict(target.net.state_dict())                  # a fresh trainable copy of the target's weights
    ppo = ppo_config(train_seed, steps, **{"lane_cache": 5000, **ppo_kw})
    res = train_ppo(cfg, ppo, OpponentSampler([target]), out_dir=out_dir, net=net, verbose=verbose,
                    name=f"attack_{target_spec.replace(':', '_')}")
    return NetAgent(res.net, f"attacker:{target_spec}"), res.history, target, ppo


def _attack_job(args) -> tuple:
    """(target seed, steps, episodes, windows, other seeds, out_dir or None, ppo_kw[, phase]): train, then evaluate
    against the target and, for transfer, the other finals of the phase and the same seed's 8.192M final."""
    seed, steps, episodes, windows, others, out_dir, ppo_kw, *rest = args
    phase = rest[0] if rest else "full"
    import torch
    from ....env.discounted import evaluate_discounted
    torch.set_num_threads(1)
    cfg = env_cfg()
    t0 = time.perf_counter()
    attacker, hist, target, ppo = train_attacker(f"final:{phase}:{seed}", steps, ATTACK_TRAIN_SEED_F + seed, out_dir,
                                                 **ppo_kw)
    t_train = time.perf_counter() - t0
    keys = ("estimate", "stderr", "ci95", "seat_A", "seat_B", "window")
    ev = {f"W{w}": {k: e[k] for k in keys} for w in windows
          for e in [evaluate_discounted(cfg, attacker, target, episodes=episodes, seed=ATTACK_EVAL_SEED_F, window=w)]}
    transfer = {}
    targets = [f"final:{phase}:{s2}" for s2 in others] + ([f"final:full:{seed}"] if phase != "full" else [])
    for t in targets:
        e = evaluate_discounted(cfg, attacker, agent_from_spec(t, cfg), episodes=episodes, seed=ATTACK_EVAL_SEED_F,
                                window=max(windows))
        transfer[t] = {k: e[k] for k in ("estimate", "stderr", "ci95")}
    curve = [[h["steps"], h["mean_return"]] for h in hist if np.isfinite(h.get("mean_return", np.nan))]
    return seed, {"target": f"final:{phase}:{seed}", "steps": steps, "train_elapsed_s": t_train,
                  "transitions_per_s": steps / max(t_train, 1e-9), "eval": ev, "transfer": transfer,
                  "curve": curve[::max(1, len(curve) // 60)], "updates": len(hist),
                  "entropy_first_last": [hist[0]["entropy"], hist[-1]["entropy"]] if hist else None,
                  "ppo": ppo.to_dict(), "elapsed_s": time.perf_counter() - t0}


def run(confirm: bool, workers: int, seeds: Sequence[int] = (0, 1, 2, 3, 4, 5), steps: int = ATTACK_BUDGET_F,
        phase: str = "ext") -> dict:
    if not confirm:
        raise SystemExit(f"best-response attackers are part of the comprehensive evaluation; they need their own "
                         f"approval and then {CONFIRM_FLAG}. Nothing was started.")
    for s in seeds:
        if os.path.isdir(attacker_dir(s, phase=phase)) and os.listdir(attacker_dir(s, phase=phase)):
            raise SystemExit(f"{attacker_dir(s, phase=phase)} exists: attackers are never overwritten")
    t0 = time.perf_counter()
    js = [(s, steps, ATTACK_EPISODES, (W_DEV, W_HEAD), [x for x in seeds if x != s], attacker_dir(s, phase=phase), {},
           phase) for s in seeds]
    res = dict(run_jobs(_attack_job, js, min(workers, len(js)), "attackers"))
    vals = [res[s]["eval"][f"W{W_HEAD}"]["estimate"] for s in seeds]
    data = {"meta": meta(stage="attackers", phase=phase, steps=steps, episodes=ATTACK_EPISODES,
                         eval_seed=ATTACK_EVAL_SEED_F),
            "label": "approximate best responses (exploitability LOWER bounds)",
            "budget_note": f"{steps:,} learner transitions per attacker = {steps / 16_384_000 if phase == 'ext' else steps / 8_192_000:.2f}"
                           " of the target's training budget (the M4.3b protocol used one half)",
            "results": {str(k): v for k, v in res.items()},
            "summary": {"per_seed": vals, "mean_ci": list(mean_ci(vals))}, "elapsed_s": time.perf_counter() - t0}
    save_json(f"attackers_{phase}", data, AOUT)
    return data


def smoke(seed: int = 0, workers: int = 1, phase: str = "full") -> dict:
    """The attacker pipeline at 2 updates (2 x 2,048 learner transitions), evaluated at tiny sizes; no checkpoints."""
    t0 = time.perf_counter()
    _, r = _attack_job((seed, 4096, 2, (100,), [s for s in (0, 1) if s != seed][:1], None, {}, phase))
    out = {"meta": meta(stage="attackers_smoke"), "result": r, "elapsed_s": time.perf_counter() - t0}
    save_json("attackers_smoke", out, AOUT)
    return out


__all__ = ["run", "smoke", "train_attacker", "attacker_dir", "ATTACK_BUDGET_F", "CONFIRM_FLAG"]


def behaviour(workers: int, seeds: Sequence[int] = (0, 1, 2, 3, 4, 5), phase: str = "ext", games: int = 8,
              turns: int = 500) -> dict:
    """What the attackers exploit: records of each attacker against its frozen target (both seats), the attacker
    focal and the target focal, on the health records' seed (behaviour.records_job). Descriptive, after training."""
    from . import behavior as BH
    from .health import BEHAVIOR_SEED_F
    js = []
    for s in seeds:
        att = os.path.join(attacker_dir(s, phase=phase), "final.pt")
        if os.path.exists(att):
            js += [(att, f"final:{phase}:{s}", games, turns, BEHAVIOR_SEED_F + 30),
                   (f"final:{phase}:{s}", att, games, turns, BEHAVIOR_SEED_F + 30)]
    rec = {}
    for (k, v), j in zip(run_jobs(BH.records_job, js, workers, "attacker behaviour"), js):
        rec[f"{'attacker' if j[0].endswith('.pt') else 'target'}|{k[0] if not k[0].endswith('.pt') else k[1]}"] = v
    data = {"meta": meta(stage="attackers_behaviour", phase=phase, games=games, turns=turns), "records": rec}
    save_json(f"attackers_behaviour_{phase}", data, AOUT)
    return data
