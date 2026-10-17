"""Training of M4.6f: matched continuations of the six M4.6c 8.192M C1 policies (`start:<s>`) in two arms.

Every setting is M4.6c's (`m46c.train.ppo_config` / `shared_config` / `collector`, hence M4.5c's PPO: 100%
current-policy shared self-play, gamma 0.99, GAE 0.95, 128-turn segments, reward scale 0.1, value clip 0.5, vf_coef
0.5, 4 epochs, minibatch 256, clip 0.2, entropy 0.03, gradient norm 0.5 over ALL parameters, 16 environments,
snapshots every 25,600 transitions; the mixed start distribution with about 25% of transitions in archive games of
150 turns). The same in both arms:
    * initialization: the weights of start:<s>; in the separate arm the critic trunk is a copy of the shared trunk and
      the value head the start's (`SeparateCriticNet.from_shared`), so both arms start from the same policy and values;
    * optimizer: a fresh Adam over all parameters (checkpoints store no Adam moments), lr 3e-4 annealed linearly to 0
      over the run's budget: the M4.6c protocol, which re-warms the learning rate of the annealed start;
    * the PPO seed TRAIN_SEED + s (the same staggered first lengths, start draws and chance seeds until the
      trajectories diverge; the first rollout is identical in both arms).
The only difference is whether the critic shares the actor's trunk.

`GradProbe` (the rollout_stats hook): M4.5c's `RolloutStats` plus, every GRAD_EVERY updates, gradient diagnostics on
four minibatches of the rollout just used, at the post-update weights. It uses `torch.autograd.grad` only (no `.grad`
writes, no RNG draws), so a run is bit-identical with or without it (test).

Outputs: results/rl/checkpoints/m46f/train/<phase>/<arm>/s<seed>/ (run.json, history.jsonl, snapshots/, final.pt,
summary.json); records in results/m46f/. Runs never overwrite an existing directory except in the smoke phase.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import common as C


def ppo_config(seed: int, total: int, **kw):
    from dataclasses import replace
    from ..m46c.train import ppo_config as p46
    return replace(p46(C.ENV, seed, total, **kw), seed=C.TRAIN_SEED + seed)


def shared_config():
    from ..m46c.train import shared_config as s46
    return s46()


def collector():
    from ..m46c.train import collector as c46
    return c46()


def initial_net(arm: str, seed: int):
    """The arm's initial network for seed s (module docstring)."""
    from ....rl import PolicyValueNet, SeparateCriticNet
    src = C.net_of(C.start_path(seed))
    if arm == "shared":
        net = PolicyValueNet(src.obs_size, src.n_actions, src.hidden)
        net.load_state_dict(src.state_dict())
        return net
    if arm == "separate":
        return SeparateCriticNet.from_shared(src)
    raise ValueError(arm)


def actor_digest(net) -> str:
    """Digest of the actor's parameters only (trunk and policy head: the same names in both classes)."""
    from ....rl import weights_digest
    return weights_digest({k: v for k, v in net.state_dict().items() if k.startswith(("trunk.", "policy_head."))})


def training_spec() -> dict:
    p = ppo_config(0, C.BUDGET).to_dict()
    for k in ("seed", "total_steps"):
        p.pop(k)
    from ..m46c import common as C46
    return {"environment": {"arm": C.ENV, "preset": "GameConfig.m45_calibrated()", "overrides": C46.arm_overrides(C.ENV)},
            "arms": {"shared": "PolicyValueNet (shared actor-critic trunk), the start's weights",
                     "separate": "SeparateCriticNet.from_shared(start): the start's actor; critic trunk = a copy of the "
                                 "start's trunk, value head = the start's"},
            "ppo": p, "shared": shared_config().to_dict(),
            "initialization": "start:<s> = M4.6c ext/C1:<s> (8.192M C1 final); fresh Adam over all parameters, lr 3e-4 "
                              "annealed linearly to 0 over the run's budget (both arms)",
            "budget": C.BUDGET, "midpoint": C.MIDPOINT, "ext_budget": C.EXT_BUDGET,
            "starts": {"beta": C46.ARCHIVE_BETA, "archive_length": C46.ARCHIVE_LENGTH, "game_length": C46.GAME_LENGTH,
                       "mirror_p": C46.MIRROR_P, "archive": "the M4.6c training archive"},
            "seeds": {"train": f"{C.TRAIN_SEED} + s, the same in both arms"}}


# ---------------------------------------------------------------------------------------------- gradient diagnostics
class GradProbe:
    """`train_shared(rollout_stats=...)`: RolloutStats plus gradient diagnostics (module docstring).

    Per probed update, means over `n_mb` minibatches of 256 (drawn with default_rng([GRAD_ROOT, update])):
        total_norm          the pre-clip norm of the full PPO-loss gradient over all parameters (what the clip sees)
        actor_norm          its part on the actor's parameters (trunk + policy head)
        critic_norm         its part on critic-only parameters (value head; separate arm: + critic trunk)
        trunk_pol / trunk_val / trunk_cos    on the actor trunk: the policy(+entropy) term, the value term and their
                            cosine (the value term is exactly 0 in the separate arm)
        ctrunk_val          the value term on the critic trunk (separate arm only)
        clip_shrink         min(1, c / total_norm) / min(1, c / actor_norm), c = max_grad_norm: the factor by which
                            critic-only gradients shrink the actor's clipped step through the global clip (1 = none)
    The policy term is the clipped surrogate at ratio 1 (the current policy as the behaviour policy), the value term
    vf_coef x the unclipped value loss against the rollout's lambda-returns."""

    def __init__(self, ppo, net, every: int = C.GRAD_EVERY, n_mb: int = 4):
        from ..m45c.train import RolloutStats
        self.base, self.ppo, self.net, self.every, self.n_mb = RolloutStats(ppo), ppo, net, int(every), int(n_mb)
        self.calls = 0

    def __call__(self, roll: dict, finished: list) -> dict:
        out = self.base(roll, finished)
        self.calls += 1
        if self.calls == 1 or self.calls % self.every == 0:
            out.update(self.grads(roll, self.calls))
        return out

    def grads(self, roll: dict, update: int) -> Dict[str, float]:
        import torch
        from ....rl.network import MaskedCategorical
        from ....rl.shared import rollout_batch
        pc, net = self.ppo, self.net
        batch, _, _ = rollout_batch(roll, pc)
        n = batch["obs"].shape[0]
        perm = np.random.default_rng([C.GRAD_ROOT, int(update)]).permutation(n)
        trunk = list(net.trunk.parameters())
        phead = list(net.policy_head.parameters())
        vhead = list(net.value_head.parameters())
        ctrunk = list(net.critic_trunk.parameters()) if hasattr(net, "critic_trunk") else []
        params = trunk + phead + ctrunk + vhead
        groups = {"trunk": slice(0, len(trunk)), "phead": slice(len(trunk), len(trunk) + len(phead)),
                  "ctrunk": slice(len(trunk) + len(phead), len(trunk) + len(phead) + len(ctrunk)),
                  "vhead": slice(len(trunk) + len(phead) + len(ctrunk), len(params))}

        def flat(gs, sl):
            v = [g.reshape(-1) for g in gs[sl] if g is not None]
            return torch.cat(v) if v else torch.zeros(1)

        rows = []
        for b in range(self.n_mb):
            idx = torch.as_tensor(perm[b * pc.minibatch_size:(b + 1) * pc.minibatch_size])
            if len(idx) == 0:
                break
            logits, v = net(batch["obs"][idx])
            dist = MaskedCategorical(logits, batch["mask"][idx])
            logp = dist.log_prob(batch["actions"][idx])
            w = batch["choice"][idx]
            nw = w.sum().clamp(min=1.0)
            a = batch["advantages"][idx]
            m = (a * w).sum() / nw
            sd = (((a - m) ** 2 * w).sum() / nw).sqrt()
            a = (a - m) / (sd + 1e-8)
            ratio = (logp - logp.detach()).exp()
            pg = (torch.max(-a * ratio, -a * ratio.clamp(1 - pc.clip_coef, 1 + pc.clip_coef)) * w).sum() / nw
            pol = pg - pc.ent_coef * (dist.entropy() * w).sum() / nw
            val = pc.vf_coef * 0.5 * ((v - batch["returns"][idx]) ** 2).mean()
            gp = torch.autograd.grad(pol, params, allow_unused=True, retain_graph=True)
            gv = torch.autograd.grad(val, params, allow_unused=True)
            gt = [None if (x is None and y is None) else (x if y is None else (y if x is None else x + y))
                  for x, y in zip(gp, gv)]
            tot = float(torch.cat([g.reshape(-1) for g in gt if g is not None]).norm())
            act = float(torch.cat([flat(gt, groups["trunk"]), flat(gt, groups["phead"])]).norm())
            cri = float(torch.cat([flat(gt, groups["ctrunk"]), flat(gt, groups["vhead"])]).norm())
            tp, tv = flat(gp, groups["trunk"]), flat(gv, groups["trunk"])
            cos = float(torch.dot(tp, tv) / (tp.norm() * tv.norm() + 1e-12)) if tv.numel() == tp.numel() else 0.0
            c = pc.max_grad_norm
            shrink = min(1.0, c / (tot + 1e-6)) / min(1.0, c / (act + 1e-6))
            rows.append((tot, act, cri, float(tp.norm()), float(tv.norm()), cos,
                         float(flat(gv, groups["ctrunk"]).norm()) if ctrunk else 0.0, shrink,
                         float(val.detach()) / pc.vf_coef if pc.vf_coef else float("nan")))
        r = np.asarray(rows)
        keys = ("total_norm", "actor_norm", "critic_norm", "trunk_pol", "trunk_val", "trunk_cos", "ctrunk_val",
                "clip_shrink", "value_loss_probe")
        return {f"gp_{k}": float(r[:, i].mean()) for i, k in enumerate(keys)}


# ---------------------------------------------------------------------------------------------- one run
def _peak_memory_mb() -> Dict[str, float]:
    from ..m46c.train import _peak_memory_mb as pm
    return pm()


def train_job(args) -> dict:
    """(phase, arm, seed, total, stop or None, overwrite[, out_dir]): one continuation; writes <run>/summary.json."""
    phase, arm, seed, total, stop, overwrite, *rest = args
    out = rest[0] if rest else C.run_dir(arm, seed, phase)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(1)
        from ....env import LaneEnv
        from ....rl import checkpoint_info, train_shared, weights_digest
        from ..m43b.train import summarize_history
        from ..m45c.train import rollout_summary
        cfg = C.env_cfg()
        ppo = ppo_config(seed, total)
        net = initial_net(arm, seed)
        init_digest, init_actor = weights_digest(net.state_dict()), actor_digest(net)
        res = train_shared(cfg, ppo, shared_config(), out_dir=out, net=net, name=f"m46f_{arm}_s{seed}",
                           verbose=phase not in ("bench",), overwrite=overwrite, stop_at=stop,
                           rollout_stats=GradProbe(ppo, net), collector_cls=collector())
        env = LaneEnv(cfg, max_turns=1)
        o = env.reset(seed=0)
        with torch.no_grad():
            v0 = [float(res.net.value(torch.as_tensor(o[s].vector).unsqueeze(0))[0]) / ppo.reward_scale for s in (0, 1)]
        s = summarize_history(res.history, ppo.reward_scale)
        info = checkpoint_info(os.path.join(out, "final.pt"))
        s.update({"status": "complete", "name": f"{arm}_s{seed}", "phase": phase, "arm": arm, "seed": seed,
                  "network": type(res.net).__name__, "parameters": int(sum(p.numel() for p in res.net.parameters())),
                  "schedule_total_steps": total, "stop_at": stop, "budget": stop or total, "run_dir": out,
                  "agent_transitions": res.agent_transitions, "env_turns": res.env_turns, "updates": len(res.history),
                  "elapsed_s": res.elapsed, "transitions_per_s": res.agent_transitions / res.elapsed,
                  "value_s0_resources": v0, "snapshots": len(res.snapshot_paths),
                  "digest": weights_digest(res.net.state_dict()), "initial_digest": init_digest,
                  "initial_actor_digest": init_actor, "start_actor_digest": actor_digest(C.net_of(C.start_path(seed))),
                  "start_checkpoint": C.start_path(seed), "start_sha": C.checkpoint_sha(C.start_path(seed)),
                  "ppo": ppo.to_dict(), "shared": shared_config().to_dict(), "memory": _peak_memory_mb(),
                  "rollout": rollout_summary(res.history), "starts": (info.get("metadata") or {}).get("starts"),
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


def history(arm: str, seed: int, phase: str = "main", run_dir: Optional[str] = None) -> List[dict]:
    p = os.path.join(run_dir or C.run_dir(arm, seed, phase), "history.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summary(arm: str, seed: int, phase: str = "main") -> Optional[dict]:
    p = os.path.join(C.run_dir(arm, seed, phase), "summary.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------- smoke
SMOKE_UPDATES = 10
FIRST_ROLLOUT_KEYS = ("rollout_agent_transitions", "games_current", "rollout_truncations", "reward_mean_abs",
                      "mask_n_legal_mean", "illegal_sampled")


def smoke(workers: int = 3) -> dict:
    """Pipeline check before the pre-registration: the initial-equivalence verification, both arms of seed 0 for
    SMOKE_UPDATES updates of the real 4.096M schedule (stop_at: the pilot's learning rate), a bit-identical re-run of
    the separate arm, and throughput. Checks pipeline health only; no comparison of the arms is read."""
    from ..m44.train import _histories_equal
    from ..m45c.full import stability_scan
    from ..m46c import common as C46
    from . import verify as V
    stop = SMOKE_UPDATES * C46.TRANSITIONS_PER_UPDATE
    t0 = time.perf_counter()
    eq = V.run(seeds=(0,))
    base = os.path.join(C.CKPT, "train", "smoke")
    js = [("smoke", a, 0, C.BUDGET, stop, True, os.path.join(base, a, "s0")) for a in C.ARMS]
    js.append(("smoke", "separate", 0, C.BUDGET, stop, True, os.path.join(base, "separate", "s0_repeat")))
    res = C.run_jobs(train_job, js, workers, "M4.6f smoke")
    out = {"meta": C.meta(stage="smoke"), "spec": training_spec(), "equivalence": eq, "runs": {}, "checks": {}}
    checks = dict(eq["checks"])
    hs = {}
    for j, r in zip(js, res):
        key = f"{os.path.basename(j[6])}_{j[1]}"                 # s0_shared, s0_separate, s0_repeat_separate
        out["runs"][key] = {k: v for k, v in r.items() if k != "curve"}
        if r.get("status") != "complete":
            checks[f"{key}: completed"] = False
            continue
        h = history(j[1], 0, run_dir=j[6])
        hs[key] = h
        checks[f"{key}: finite losses and gradients"] = bool(r["finite_losses_and_gradients"])
        checks[f"{key}: {SMOKE_UPDATES} updates, {stop:,} transitions"] = r["agent_transitions"] == stop
        checks[f"{key}: both seats trained equally"] = r["transitions_seat_a"] == r["transitions_seat_b"]
        checks[f"{key}: 100% current-policy"] = r["transitions_historical"] == 0
        checks[f"{key}: no illegal action sampled"] = r["rollout"].get("illegal_sampled_total", 1) == 0
        checks[f"{key}: approximate KL mean < 0.05"] = r["approx_kl_mean"] < 0.05
        checks[f"{key}: initial actor = start:0 actor"] = r["initial_actor_digest"] == r["start_actor_digest"]
        checks[f"{key}: archive games started"] = (r.get("starts") or {}).get("games_started", {}).get("archive", 0) > 0
        checks[f"{key}: no stop condition"] = not stability_scan(h)["alarms"]
        checks[f"{key}: gradient probe recorded"] = "gp_total_norm" in h[0]
    a, b = hs.get("s0_shared"), hs.get("s0_separate")
    if a and b:
        checks["first rollout identical in both arms"] = all(a[0].get(k) == b[0].get(k) for k in FIRST_ROLLOUT_KEYS) \
            and all(a[0][k] == b[0][k] for k in a[0] if k.startswith("act_"))
        checks["the arms diverge after the first update"] = a[-1]["value_loss"] != b[-1]["value_loss"]
    r_sep = next((r for j, r in zip(js, res) if j[1] == "separate" and not j[6].endswith("repeat")), None)
    r_rep = next((r for j, r in zip(js, res) if j[6].endswith("repeat")), None)
    if r_sep and r_rep and r_sep.get("digest") and r_rep.get("digest"):
        checks["separate re-run: bit-identical weights"] = r_sep["digest"] == r_rep["digest"]
        checks["separate re-run: identical histories"] = _histories_equal(js[1][6], js[2][6])
    out["checks"] = checks
    out["all_passed"] = all(checks.values())
    out["throughput"] = {k: {"transitions_per_s": v.get("transitions_per_s"), "collect_s": v.get("collect_s"),
                             "update_s": v.get("update_s"), "parameters": v.get("parameters")}
                         for k, v in out["runs"].items()}
    out["grad_probe"] = {k: {kk: vv for kk, vv in h[i].items() if kk.startswith("gp_")}
                         for k, h in hs.items() for i in (0,)}
    out["wall_s"] = time.perf_counter() - t0
    C.save_json("smoke", out)
    return out


# ---------------------------------------------------------------------------------------------- the pilot runs
def train(workers: int = 6, phase: str = "main", seeds: Sequence[int] = C.SEEDS) -> dict:
    """The runs of a phase (both arms of every seed), `workers` at a time, the two arms of a seed adjacent in the
    queue so that they run side by side; refuses existing incomplete run directories."""
    C.require_prereg()
    if phase == "ext":
        d = C.load_json("extension_decision")
        if not d or not d.get("fire"):
            raise SystemExit("the extension phase runs only after the pre-registered rule fired (stage ext_rule)")
    total = C.PHASE_BUDGET[phase]
    js = [(phase, a, s, total, None, False) for s in seeds for a in C.ARMS]
    todo = []
    for j in js:
        d = C.run_dir(j[1], j[2], phase)
        sm = summary(j[1], j[2], phase)
        if sm and sm.get("status") == "complete":
            continue
        if os.path.isdir(d) and os.listdir(d):
            raise SystemExit(f"{d} exists and is incomplete: inspect it; runs are never overwritten")
        todo.append(j)
    from ..m46c.train import ResourceLog
    log = ResourceLog(os.path.join(C.OUT, f"resources_{phase}.jsonl"))
    log.start()
    t0 = time.perf_counter()
    print(f"[m46f train:{phase}] {len(todo)} runs, {workers} concurrent: {[f'{j[1]}_s{j[2]}' for j in todo]}",
          flush=True)
    try:
        res = C.run_jobs(train_job, todo, workers, f"m46f train {phase}")
    finally:
        log.stop_flag.set()
    for r in res:
        print(f"[m46f train:{phase}] {r['name']}: {r['status']}", flush=True)
    C.jdump(os.path.join(C.OUT, f"wall_{phase}.json"), {"phase": phase, "runs": [r["name"] for r in res],
                                                         "wall_s": time.perf_counter() - t0,
                                                         "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    return check(phase)


def check(phase: str = "main", save: bool = True) -> dict:
    """The phase record rebuilt from the run directories: status, pipeline checks and the stability scan."""
    from ..m45c.full import stability_scan
    from ....env.discounted import tail_constant
    from ..m46c import common as C46
    total = C.PHASE_BUDGET[phase]
    omega = tail_constant(C.env_cfg())
    runs, checks, scans = {}, {}, {}
    for s in C.SEEDS:
        for a in C.ARMS:
            name = f"{a}_s{s}"
            r = summary(a, s, phase)
            h = history(a, s, phase)
            scans[name] = stability_scan(h)
            if r is None:
                runs[name] = {"status": "missing" if not h else f"incomplete ({len(h)} updates)"}
                continue
            runs[name] = {k: v for k, v in r.items() if k != "curve"}
            if r.get("status") != "complete":
                continue
            share = (r.get("starts") or {}).get("archive_transition_share_finished", float("nan"))
            ev100 = float(np.mean([e["explained_variance"] for e in h[-100:]])) if h else float("nan")
            runs[name]["explained_variance_last100"] = ev100
            checks.update({
                f"{name}: critic explained variance > 0 over the last 100 updates": bool(ev100 > 0),
                f"{name}: finite losses and gradients": bool(r["finite_losses_and_gradients"]),
                f"{name}: budget reached exactly ({total:,})": r["agent_transitions"] == total,
                f"{name}: never terminated; one truncation per finished game": r["terminations"] == 0 and
                r["truncations"] == r["games_finished"],
                f"{name}: critic bounded (|V| <= Omega)": r["value_absmax_resources"] <= omega,
                f"{name}: both seats trained equally": r["transitions_seat_a"] == r["transitions_seat_b"],
                f"{name}: 100% current-policy": r["transitions_historical"] == 0,
                f"{name}: no illegal action sampled": r["rollout"].get("illegal_sampled_total", 1) == 0,
                f"{name}: initial actor = start:{s}": r["initial_actor_digest"] == r["start_actor_digest"],
                f"{name}: network class": r["network"] == ("PolicyValueNet" if a == "shared" else "SeparateCriticNet"),
                f"{name}: archive transition share 0.20-0.30": bool(0.20 <= share <= 0.30),
                f"{name}: midpoint snapshot present": os.path.exists(C.snapshot_path(a, s, C.MIDPOINT, phase)),
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
            h = history(a, s, phase)
            name = f"{a}_s{s}"
            if not h:
                print(f"  {name:14s} not started", flush=True)
                continue
            e = h[-1]
            tps = float(np.mean([x["sps"] for x in h[-20:]]))
            overall = e["agent_transitions"] / e["elapsed"]
            eta = (C.PHASE_BUDGET[phase] - e["agent_transitions"]) / max(overall, 1.0)
            scan = stability_scan(h)
            rows[name] = {"updates": e["update"], "transitions": e["agent_transitions"], "eta_s": eta,
                          "alarms": scan["alarms"]}
            print(f"  {name:14s} upd {e['update']:4d} {e['agent_transitions'] / 1e6:5.2f}M {tps:5.0f} tps ETA "
                  f"{eta / 60:5.1f} min ent {e['entropy']:.2f} kl {e['approx_kl']:.4f} ev {e['explained_variance']:.2f} "
                  f"vl {e['value_loss']:.4f} g {e['grad_norm']:.2f} lr {e['lr']:.1e} alarms {scan['alarms'] or '-'}",
                  flush=True)
    return rows


__all__ = ["ppo_config", "shared_config", "collector", "initial_net", "actor_digest", "training_spec", "GradProbe",
           "train_job", "history", "summary", "smoke", "train", "check", "status"]
