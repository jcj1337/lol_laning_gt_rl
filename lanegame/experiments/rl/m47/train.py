"""Stage 2 training: matched continuations of the M4.6c 8.192M baseline policies under two objectives.

Arms (the ONLY difference is the game's discount, i.e. the objective; mechanics, reward, observation, actions, network,
start distribution, PPO settings, seeds and budget are identical):
    control   `m45_calibrated()`: gamma 0.99 (expected duration 100 turns)
    short     `m45_calibrated(gamma=SHORT_GAMMA)`: gamma 1 - 1/SHORT_H, an opt-in keyword override of the existing field
The trainer takes its discount from the game configuration (`PPOConfig.continuing`), so returns, advantages (GAE),
critic targets and the bootstrapped truncations all use the arm's gamma; `check_time_semantics` refuses a mismatch.

Every other setting is M4.6c's (`m46c.train`): `train_shared`, 100% current-policy shared self-play, GAE lambda 0.95,
segments of 128 turns, s0 games of 512 turns and archive games of 150 turns (the M4.6c mixed collector and archive,
about 25% of transitions), reward scale 0.1, value clip 0.5, 4 epochs, minibatch 256, clip 0.2, entropy 0.03, gradient
norm 0.5, 16 environments, the 2 x 128 tanh shared network, snapshots every 25,600 transitions, lane memo 5,000.
Initialization: the weights of ext/baseline:s; a fresh Adam with lr 3e-4 annealed linearly to 0 over the budget (as
every earlier continuation); the PPO seed TRAIN_SEED + s in both arms (the same start-state sequence and, until the
first update, the same rollout).

Outputs: results/rl/checkpoints/m47/train/<phase>/<arm>/s<seed>/ (run.json, history.jsonl, snapshots/, final.pt,
summary.json; git-ignored); records in results/m47/. Runs never overwrite a directory except in the smoke phase.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import common as C

SMOKE_UPDATES = 10
FIRST_ROLLOUT_KEYS = ("rollout_agent_transitions", "games_current", "rollout_truncations", "reward_mean_abs",
                      "mask_n_legal_mean", "illegal_sampled")


def ppo_config(arm: str, seed: int, total: int):
    from ....rl import PPOConfig
    from ..m46c import common as C46
    return PPOConfig.continuing(C.arm_cfg(arm), total_steps=total, n_envs=C46.N_ENVS, seed=C.TRAIN_SEED + seed,
                                torch_threads=1, ent_coef=C46.ENT_COEF, hidden=C46.HIDDEN, lane_cache=C46.LANE_CACHE,
                                game_length=C46.GAME_LENGTH, start_mix=0.0)


def start_spec(seed: int) -> str:
    return C.general("baseline", seed)


def training_spec() -> dict:
    from ..m46c import common as C46
    from ..m46c.train import shared_config
    p = ppo_config("control", 0, C.BUDGET).to_dict()
    for k in ("seed", "total_steps", "gamma"):
        p.pop(k)
    return {"arms": {a: {"preset": "GameConfig.m45_calibrated()", "gamma": C.arm_gamma(a),
                         "expected_duration": 1 / (1 - C.arm_gamma(a))} for a in C.ARMS},
            "ppo_except_gamma": p, "shared": shared_config().to_dict(),
            "initialization": "weights of ext/baseline:<s> (M4.6c 8.192M baseline continuation), fresh Adam, lr 3e-4 "
                              "annealed linearly to 0 over the budget", "budget": C.BUDGET, "midpoint": C.MIDPOINT,
            "starts": {"beta": C46.ARCHIVE_BETA, "archive_length": C46.ARCHIVE_LENGTH, "game_length": C46.GAME_LENGTH,
                       "mirror_p": C46.MIRROR_P}, "seeds": {"train": f"{C.TRAIN_SEED} + s, the same in both arms"},
            "pilot_seeds": list(C.PILOT_SEEDS)}


# ---------------------------------------------------------------------------------------------- GAE spy (smoke only)
class GaeSpy:
    """Wraps `lanegame.rl.shared.compute_gae` in this process: records, for the first `n` calls, the discount and
    lambda that reached it, terminations and truncations among trainable entries, whether every truncation was
    bootstrapped with a nonzero critic value of the next observation, and how much the advantages would differ under
    gamma 0.99 (the objective change reaching the numbers)."""

    def __init__(self, n: int = 3):
        self.n, self.calls = n, []

    def install(self):
        from ....rl import shared as SH
        from ....rl.ppo import compute_gae as orig
        self.orig = orig

        def spy(rewards, values, next_values, terminated, truncated, gamma, lam):
            adv, ret = orig(rewards, values, next_values, terminated, truncated, gamma, lam)
            if len(self.calls) < self.n:
                tr = np.asarray(truncated, bool)
                adv99, _ = orig(rewards, values, next_values, terminated, truncated, C.GAMMA, lam)
                self.calls.append({"gamma": float(gamma), "lambda": float(lam),
                                   "terminated": int(np.asarray(terminated, bool).sum()),
                                   "truncated": int(tr.sum()),
                                   "truncated_bootstrap_nonzero": int((np.abs(next_values[tr]) > 0).sum()),
                                   "truncated_next_value_absmean": float(np.abs(next_values[tr]).mean())
                                   if tr.any() else float("nan"),
                                   "adv_rms": float(np.sqrt(np.mean(adv ** 2))),
                                   "adv_rms_diff_vs_gamma_0.99": float(np.sqrt(np.mean((adv - adv99) ** 2)))})
            return adv, ret

        SH.compute_gae = spy
        return self

    def uninstall(self):
        from ....rl import shared as SH
        SH.compute_gae = self.orig


# ---------------------------------------------------------------------------------------------- one run
def train_job(args) -> dict:
    """(phase, arm, seed, total, stop or None, overwrite, out_dir or None, spy): one continuation run; writes
    <run>/summary.json and returns it."""
    phase, arm, seed, total, stop, overwrite, out, spy = args
    out = out or C.run_dir(arm, seed, phase)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    g = None
    try:
        import torch
        torch.set_num_threads(1)
        from ....env import LaneEnv
        from ....env.discounted import tail_constant
        from ....rl import PolicyValueNet, checkpoint_info, train_shared, weights_digest
        from ....rl.ppo import check_time_semantics
        from ..m43b.train import summarize_history
        from ..m45c.train import RolloutStats, rollout_summary
        from ..m46c.train import _starts_from_final, collector, shared_config
        cfg = C.arm_cfg(arm)
        ppo = ppo_config(arm, seed, total)
        check_time_semantics(cfg, ppo)
        src = C.net_of(start_spec(seed))
        net = PolicyValueNet(src.obs_size, src.n_actions, src.hidden)
        net.load_state_dict(src.state_dict())
        init_digest = weights_digest(net.state_dict())
        if spy:
            g = GaeSpy().install()
        res = train_shared(cfg, ppo, shared_config(), out_dir=out, net=net, name=f"m47_{arm}_s{seed}",
                           verbose=phase != "smoke", overwrite=overwrite, stop_at=stop,
                           rollout_stats=RolloutStats(ppo), collector_cls=collector())
        env = LaneEnv(cfg, max_turns=1)
        o = env.reset(seed=0)
        with torch.no_grad():
            v0 = [float(res.net.value(torch.as_tensor(o[s].vector).unsqueeze(0))[0]) / ppo.reward_scale for s in (0, 1)]
        s = summarize_history(res.history, ppo.reward_scale)
        info = checkpoint_info(os.path.join(out, "final.pt"))
        with open(os.path.join(out, "run.json"), encoding="utf-8") as f:
            run_json = json.load(f)
        s.update({"status": "complete", "name": f"{arm}_s{seed}", "phase": phase, "arm": arm, "seed": seed,
                  "gamma": C.arm_gamma(arm), "run_json_game_gamma": run_json["game_config"]["gamma"],
                  "run_json_ppo_gamma": run_json["ppo"]["gamma"], "omega": tail_constant(cfg),
                  "schedule_total_steps": total, "stop_at": stop, "budget": stop or total, "run_dir": out,
                  "agent_transitions": res.agent_transitions, "env_turns": res.env_turns, "updates": len(res.history),
                  "elapsed_s": res.elapsed, "transitions_per_s": res.agent_transitions / res.elapsed,
                  "value_s0_resources": v0, "snapshots": len(res.snapshot_paths),
                  "digest": weights_digest(res.net.state_dict()), "initial_digest": init_digest,
                  "start_spec": start_spec(seed), "start_checkpoint": C.policy_path(start_spec(seed)),
                  "ppo": ppo.to_dict(), "shared": shared_config().to_dict(), "rollout": rollout_summary(res.history),
                  "starts": (info.get("metadata") or {}).get("starts") or _starts_from_final(out),
                  "checkpoint_digest": info["digest"], "gae_spy": g.calls if g else None, "started": started,
                  "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": time.perf_counter() - t0,
                  "pid": os.getpid()})
    except BaseException as ex:                            # noqa: BLE001 - recorded, never swallowed silently
        import traceback
        s = {"status": "failed", "name": f"{arm}_s{seed}", "phase": phase, "arm": arm, "seed": seed, "run_dir": out,
             "error": repr(ex), "traceback": traceback.format_exc(), "started": started,
             "wall_s": time.perf_counter() - t0}
    finally:
        if g is not None:
            g.uninstall()
    if os.path.isdir(out):
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1, default=float)
    return s


def history(run_dir: str) -> List[dict]:
    p = os.path.join(run_dir, "history.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _digest_of(spec: str) -> str:
    from ....rl import weights_digest
    return weights_digest(C.net_of(spec).state_dict())


# ---------------------------------------------------------------------------------------------- smoke
def smoke(workers: int = 3) -> dict:
    """Pipeline and objective checks before the pre-registration: both arms of seed 0 for SMOKE_UPDATES updates of the
    real schedule (stop_at, so the learning rate is the pilot's), with the GAE spy, plus a bit-identical re-run of the
    short arm, run concurrently (the throughput of 3 concurrent runs is recorded). No comparison of arms is read."""
    from ....config import GameConfig
    from ....rl.ppo import check_time_semantics
    from ..m44.train import _histories_equal
    from ..m45c.full import stability_scan
    from ..m46c import common as C46
    stop = SMOKE_UPDATES * C46.TRANSITIONS_PER_UPDATE
    base = os.path.join(C.CKPT, "train", "smoke")
    js = [("smoke", a, 0, C.BUDGET, stop, True, os.path.join(base, a, "s0"), True) for a in C.ARMS]
    js.append(("smoke", "short", 0, C.BUDGET, stop, True, os.path.join(base, "short", "s0_repeat"), True))
    t0 = time.perf_counter()
    res = C.run_jobs(train_job, js, workers, "M4.7 smoke")
    checks: Dict[str, bool] = {}
    # the configurations: only gamma differs; the trainer and the evaluator agree with the game
    dc, ds = C.arm_cfg("control").to_dict(), C.arm_cfg("short").to_dict()
    checks["arms differ in exactly one field: gamma"] = sorted(k for k in dc if dc[k] != ds[k]) == ["gamma"]
    checks["control = GameConfig.m45_calibrated() exactly"] = dc == GameConfig.m45_calibrated().to_dict()
    checks[f"short gamma = 1 - 1/{C.SHORT_H}"] = abs(ds["gamma"] - C.SHORT_GAMMA) < 1e-15
    for a in C.ARMS:
        p = ppo_config(a, 0, C.BUDGET)
        check_time_semantics(C.arm_cfg(a), p)
        checks[f"{a}: PPOConfig.continuing takes the game's gamma"] = p.gamma == C.arm_gamma(a)
        checks[f"{a}: no gamma-mismatch opt-in"] = not p.allow_gamma_mismatch
    runs, hs = {}, {}
    for j, r in zip(js, res):
        key = f"{j[1]}_{os.path.basename(j[6])}"
        runs[key] = {k: v for k, v in r.items() if k != "curve"}
        if r.get("status") != "complete":
            checks[f"{key}: completed"] = False
            continue
        h = history(j[6])
        hs[key] = h
        g = C.arm_gamma(j[1])
        spy = r.get("gae_spy") or []
        checks[f"{key}: finite losses and gradients"] = bool(r["finite_losses_and_gradients"])
        checks[f"{key}: {SMOKE_UPDATES} updates, {stop:,} transitions"] = r["agent_transitions"] == stop
        checks[f"{key}: run.json game and PPO gamma = {g:.6f}"] = (r["run_json_game_gamma"] == g
                                                                    and r["run_json_ppo_gamma"] == g)
        checks[f"{key}: GAE received the arm's gamma and lambda 0.95"] = bool(spy) and all(
            c["gamma"] == g and c["lambda"] == 0.95 for c in spy)
        checks[f"{key}: no termination reached GAE"] = bool(spy) and all(c["terminated"] == 0 for c in spy)
        checks[f"{key}: truncations bootstrapped with nonzero critic values"] = bool(spy) and any(
            c["truncated"] > 0 for c in spy) and all(c["truncated_bootstrap_nonzero"] == c["truncated"] for c in spy)
        checks[f"{key}: never terminated; truncations only"] = r["terminations"] == 0
        checks[f"{key}: both seats trained equally"] = r["transitions_seat_a"] == r["transitions_seat_b"]
        checks[f"{key}: 100% current-policy"] = r["transitions_historical"] == 0
        checks[f"{key}: no illegal action sampled"] = r["rollout"].get("illegal_sampled_total", 1) == 0
        checks[f"{key}: initial weights = ext/baseline:0"] = r["initial_digest"] == _digest_of(start_spec(0))
        checks[f"{key}: archive games started"] = (r.get("starts") or {}).get("games_started", {}).get("archive", 0) > 0
        checks[f"{key}: first update at lr 3e-4"] = abs(h[0]["lr"] - 3e-4) < 1e-12
        checks[f"{key}: lr after {SMOKE_UPDATES} updates on the {C.BUDGET:,} schedule"] = abs(
            h[-1]["lr"] - 3e-4 * (1 - (SMOKE_UPDATES - 1) * C46.TRANSITIONS_PER_UPDATE / C.BUDGET)) < 1e-12
        checks[f"{key}: the policy moves (approximate KL > 0)"] = r["approx_kl_mean"] > 0
        checks[f"{key}: approximate KL mean < 0.05"] = r["approx_kl_mean"] < 0.05
        checks[f"{key}: weights changed"] = r["digest"] != r["initial_digest"]
        checks[f"{key}: no stop condition"] = not stability_scan(h)["alarms"]
    a, b = hs.get("control_s0"), hs.get("short_s0")
    if a and b:
        checks["first rollout identical in both arms"] = all(a[0].get(k) == b[0].get(k) for k in FIRST_ROLLOUT_KEYS) \
            and all(a[0][k] == b[0][k] for k in a[0] if k.startswith("act_"))
        checks["the arms' value targets differ from the first update"] = a[0]["return_std"] != b[0]["return_std"]
        checks["the arms diverge after the first update"] = a[1]["reward_mean_abs"] != b[1]["reward_mean_abs"] or \
            a[-1]["value_loss"] != b[-1]["value_loss"]
    r_s = runs.get("short_s0", {})
    r_r = runs.get("short_s0_repeat", {})
    if r_s.get("digest") and r_r.get("digest"):
        checks["short re-run: bit-identical weights"] = r_s["digest"] == r_r["digest"]
        checks["short re-run: identical histories"] = _histories_equal(os.path.join(base, "short", "s0"),
                                                                       os.path.join(base, "short", "s0_repeat"))
    out = {"meta": C.meta(stage="smoke"), "spec": training_spec(), "runs": runs, "checks": checks,
           "all_passed": all(checks.values()),
           "throughput": {k: {"transitions_per_s": v.get("transitions_per_s"), "concurrent": len(js)}
                          for k, v in runs.items()},
           "gae_spy": {k: v.get("gae_spy") for k, v in runs.items()}, "wall_s": time.perf_counter() - t0}
    C.save_json("smoke", out)
    return out


# ---------------------------------------------------------------------------------------------- the pilot runs
def train(workers: int = 4, seeds: Sequence[int] = C.PILOT_SEEDS) -> dict:
    """Both arms of every pilot seed, concurrently; refuses existing incomplete run directories."""
    C.require_prereg()
    from ..m46c.train import ResourceLog
    js = [("main", a, s, C.BUDGET, None, False, None, False) for s in seeds for a in C.ARMS]
    todo = []
    for j in js:
        d = C.run_dir(j[1], j[2])
        sm = _summary(j[1], j[2])
        if sm and sm.get("status") == "complete":
            continue
        if os.path.isdir(d) and os.listdir(d):
            raise SystemExit(f"{d} exists and is incomplete: inspect it; runs are never overwritten")
        todo.append(j)
    log = ResourceLog(os.path.join(C.OUT, "resources_main.jsonl"))
    log.start()
    t0 = time.perf_counter()
    print(f"[m47 train] {len(todo)} runs, {workers} concurrent: {[f'{j[1]}_s{j[2]}' for j in todo]}", flush=True)
    try:
        res = C.run_jobs(train_job, todo, workers, "m47 train")
    finally:
        log.stop_flag.set()
    for r in res:
        print(f"[m47 train] {r['name']}: {r['status']}", flush=True)
    from ..m46c.common import jdump
    jdump(os.path.join(C.OUT, "wall_main.json"), {"runs": [r["name"] for r in res], "wall_s": time.perf_counter() - t0,
                                                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    return check()


def _summary(arm: str, seed: int, phase: str = "main") -> Optional[dict]:
    p = os.path.join(C.run_dir(arm, seed, phase), "summary.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def check(save: bool = True) -> dict:
    """The training record rebuilt from the run directories: status, pipeline checks, the stability scan."""
    from ..m45c.full import stability_scan
    runs, checks, scans = {}, {}, {}
    for s in C.PILOT_SEEDS:
        for a in C.ARMS:
            name = f"{a}_s{s}"
            r = _summary(a, s)
            h = history(C.run_dir(a, s))
            scans[name] = stability_scan(h)
            if r is None:
                runs[name] = {"status": "missing" if not h else f"incomplete ({len(h)} updates)"}
                continue
            runs[name] = {k: v for k, v in r.items() if k != "curve"}
            if r.get("status") != "complete":
                continue
            share = (r.get("starts") or {}).get("archive_transition_share_finished", float("nan"))
            g = C.arm_gamma(a)
            checks.update({
                f"{name}: finite losses and gradients": bool(r["finite_losses_and_gradients"]),
                f"{name}: budget reached exactly ({C.BUDGET:,})": r["agent_transitions"] == C.BUDGET,
                f"{name}: game and PPO gamma = {g:.6f}": (r["run_json_game_gamma"] == g
                                                          and r["run_json_ppo_gamma"] == g),
                f"{name}: never terminated; one truncation per finished game": r["terminations"] == 0 and
                r["truncations"] == r["games_finished"],
                f"{name}: critic bounded (|V| <= Omega of the arm)": r["value_absmax_resources"] <= r["omega"],
                f"{name}: both seats trained equally": r["transitions_seat_a"] == r["transitions_seat_b"],
                f"{name}: 100% current-policy": r["transitions_historical"] == 0,
                f"{name}: no illegal action sampled": r["rollout"].get("illegal_sampled_total", 1) == 0,
                f"{name}: initial weights = ext/baseline:{s}": r["initial_digest"] == _digest_of(start_spec(s)),
                f"{name}: archive transition share 0.20-0.30": bool(0.20 <= share <= 0.30),
                f"{name}: midpoint snapshot present": os.path.exists(C.policy_path(f"p/{a}:{s}@{C.MIDPOINT}")),
                f"{name}: no pre-registered stop condition": not scans[name]["alarms"]})
    for s in C.PILOT_SEEDS:
        a, b = _summary("control", s), _summary("short", s)
        if a and b and a.get("starts") and b.get("starts"):
            checks[f"s{s}: both arms started the same games"] = a["starts"]["games_started"] == \
                b["starts"]["games_started"] and a["starts"]["archive_by_label"] == b["starts"]["archive_by_label"]
    complete = [n for n, r in runs.items() if r.get("status") == "complete"]
    data = {"meta": C.meta(stage="train_check"), "spec": training_spec(), "runs": runs, "stability": scans,
            "checks": checks, "complete": complete,
            "all_passed": bool(checks) and all(checks.values()) and len(complete) == len(C.ARMS) * len(C.PILOT_SEEDS)}
    if save:
        C.save_json("train_main", data)
    return data


def status() -> dict:
    from ..m45c.full import stability_scan
    rows = {}
    for s in C.PILOT_SEEDS:
        for a in C.ARMS:
            h = history(C.run_dir(a, s))
            name = f"{a}_s{s}"
            if not h:
                print(f"  {name:12s} not started", flush=True)
                continue
            e = h[-1]
            overall = e["agent_transitions"] / e["elapsed"]
            eta = (C.BUDGET - e["agent_transitions"]) / max(overall, 1.0)
            scan = stability_scan(h)
            rows[name] = {"updates": e["update"], "transitions": e["agent_transitions"], "eta_s": eta,
                          "alarms": scan["alarms"]}
            print(f"  {name:12s} upd {e['update']:4d} {e['agent_transitions'] / 1e6:5.2f}M {overall:5.0f} tps ETA "
                  f"{eta / 60:5.1f} min ent {e['entropy']:.2f} kl {e['approx_kl']:.4f} "
                  f"ev {e['explained_variance']:.2f} "
                  f"lr {e['lr']:.1e} alarms {scan['alarms'] or '-'}", flush=True)
    return rows


__all__ = ["ppo_config", "training_spec", "train_job", "smoke", "train", "check", "status", "history", "GaeSpy"]
