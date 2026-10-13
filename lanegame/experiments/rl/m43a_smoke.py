"""M4.3a bounded PPO smoke test on the M4.3 target environment, GameConfig.m4_lasthit(). NOT research results.

    python -m lanegame.experiments.rl.m43a_smoke [all | train | repro | evaluate | report]
        [--steps 409600] [--repro-steps 102400] [--eval-episodes 16] [--window 460] [--workers 5]

Purpose (docs/rl/m4_3a_rl_infrastructure_report.md): establish that the shared-policy self-play trainer
(lanegame.rl.shared, PPOConfig.continuing) runs on the latest M4 environment with finite and reasonably scaled losses,
gradients and critic outputs; that both seats' experience is trained on; that masks are respected; that continuing-time
truncations and bootstraps behave; what the throughput is; and that a run is reproducible from its configuration and
seed. Two conditions are trained with the same budget: the agreed M4.3b baseline and its control.

  sp100   100% current-policy self-play (one network, both seats, both seats' transitions)
  sp80    80% current / 20% games against a frozen past version drawn uniformly from the last 8 snapshots

The evaluation stage is an integration readout of the discounted evaluator (protocol E1, the development window
W = 460 with its guaranteed tail bound), not a measurement of strategy: 16 paired episodes per matchup.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np

from ...config import GameConfig
from ...io import dump_json
from .common import CKPT, OUT, md_table

NAME = "m43a_smoke"
CK = os.path.join(CKPT, "m43a")
N_ENVS = 16
SNAP_EVERY = 25_600
POOL = 8
WINDOW_DEV = 460
RUNS = {"sp100_s0": (1.0, 0, "steps"), "sp80_s0": (0.8, 0, "steps"),
        "repro_s0a": (1.0, 0, "repro"), "repro_s0b": (1.0, 0, "repro"), "repro_s1": (1.0, 1, "repro")}
EVAL_OPPONENTS = ("hold_farmer", "slow_pusher", "trading_pusher")


def cfg_game() -> GameConfig:
    return GameConfig.m4_lasthit()


def run_dir(name: str) -> str:
    return os.path.join(CK, name)


def _finite(h: List[dict], keys) -> bool:
    return all(np.isfinite(e[k]) for e in h for k in keys)


def _last(h: List[dict], key: str, n: int = 10) -> float:
    return float(np.nanmean([e[key] for e in h[-n:]]))


# ---------------------------------------------------------------------------------------------- training
def train_one(args: tuple) -> dict:
    """One run in a worker process: returns the audit summary of its history (no tensors cross the process)."""
    name, fraction, seed, steps = args
    import torch
    from ...env import LaneEnv
    from ...rl import PPOConfig, SharedConfig, checkpoint_info, train_shared, weights_digest
    cfg = cfg_game()
    ppo = PPOConfig.continuing(cfg, total_steps=steps, n_envs=N_ENVS, seed=seed, torch_threads=1)
    scfg = SharedConfig(current_fraction=fraction, snapshot_every=SNAP_EVERY, pool_size=POOL)
    out = run_dir(name)
    t0 = time.perf_counter()
    res = train_shared(cfg, ppo, scfg, out_dir=out, name=name, verbose=False, overwrite=True)
    elapsed = time.perf_counter() - t0
    h = res.history
    rs = ppo.reward_scale
    env = LaneEnv(cfg, max_turns=1)
    o = env.reset(seed=0)[0]
    with torch.no_grad():
        v0 = float(res.net.value(torch.as_tensor(o.vector).unsqueeze(0))[0]) / rs
    loss_keys = ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "grad_norm", "grad_norm_max",
                 "value_mean", "value_absmax", "return_mean", "return_std", "adv_std")
    collect = sum(e["collect_s"] for e in h)
    update = sum(e["update_s"] for e in h)
    info = checkpoint_info(os.path.join(out, "final.pt"))
    summary = {
        "name": name, "current_fraction": fraction, "seed": seed, "budget": steps, "run_dir": out,
        "agent_transitions": res.agent_transitions, "env_turns": res.env_turns, "updates": len(h),
        "elapsed_s": elapsed, "train_elapsed_s": res.elapsed,
        "transitions_per_s": res.agent_transitions / res.elapsed, "env_turns_per_s": res.env_turns / res.elapsed,
        "collect_s": collect, "update_s": update, "collect_fraction": collect / (collect + update),
        "finite_losses_and_gradients": _finite(h, loss_keys),
        "grad_norm_mean": float(np.mean([e["grad_norm"] for e in h])),
        "grad_norm_max": float(np.max([e["grad_norm_max"] for e in h])),
        "clipfrac_mean": float(np.mean([e["clipfrac"] for e in h])),
        "approx_kl_mean": float(np.mean([e["approx_kl"] for e in h])),
        "approx_kl_max": float(np.max([e["approx_kl"] for e in h])),
        "entropy_first_last": [h[0]["entropy"], h[-1]["entropy"]],
        "policy_loss_first_last": [h[0]["policy_loss"], h[-1]["policy_loss"]],
        "value_loss_first_last10": [h[0]["value_loss"], _last(h, "value_loss")],
        "explained_variance_first_last10": [h[0]["explained_variance"], _last(h, "explained_variance")],
        # critic scales in resources (the history records scaled reward units)
        "value_mean_last10_resources": _last(h, "value_mean") / rs,
        "value_absmax_resources": float(np.max([e["value_absmax"] for e in h])) / rs,
        "return_std_last10_resources": _last(h, "return_std") / rs,
        "reward_mean_abs_resources": float(np.mean([e["reward_mean_abs"] for e in h])) / rs,
        "value_s0_resources": v0,
        # seats, masks, time semantics
        "transitions_seat_a": sum(e["rollout_transitions_seat_a"] for e in h),
        "transitions_seat_b": sum(e["rollout_transitions_seat_b"] for e in h),
        "transitions_historical": sum(e["rollout_transitions_historical"] for e in h),
        "single_choice_transitions": sum(e["rollout_single_choice"] for e in h),
        "truncations": sum(e["rollout_truncations"] for e in h),
        "terminations": sum(e["rollout_terminations"] for e in h),
        "games_finished": sum(e["games_current"] + e["games_historical"] for e in h),
        "games_historical": sum(e["games_historical"] for e in h),
        "current_seat_a_return_mean": float(np.nanmean([e["current_seat_a_return"] for e in h])),
        "historical_return_mean": float(np.nanmean([e["historical_return"] for e in h])) if any(
            e["games_historical"] for e in h) else None,
        "snapshots": len(res.snapshot_paths), "pool_final": [a.name for a in res.pool.snapshots],
        "digest": weights_digest(res.net.state_dict()), "provenance": info["provenance"],
        "ppo": ppo.to_dict(), "shared": scfg.to_dict(),
        "curve": [[e["agent_transitions"], e["entropy"], e["approx_kl"], e["explained_variance"], e["grad_norm"],
                   e["value_mean"] / rs] for e in h[::max(1, len(h) // 40)]],
    }
    return summary


def train(which: str, steps: int, repro_steps: int, workers: int) -> Dict[str, dict]:
    jobs = [(n, f, s, steps if kind == "steps" else repro_steps) for n, (f, s, kind) in RUNS.items()
            if (kind == "steps") == (which == "train")]
    os.makedirs(CK, exist_ok=True)
    out = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for s in ex.map(train_one, jobs):
            out[s["name"]] = s
            print(f"  {s['name']}: {s['agent_transitions']} transitions, {s['transitions_per_s']:,.0f}/s, finite "
                  f"{s['finite_losses_and_gradients']}, ev {s['explained_variance_first_last10'][1]:.2f}, "
                  f"V(s0) {s['value_s0_resources']:+.2f}", flush=True)
    return out


# ---------------------------------------------------------------------------------------------- checks
def audit(runs: Dict[str, dict]) -> dict:
    """The pass/fail conditions of the smoke stage, each with the evidence it rests on."""
    from ...env.discounted import tail_constant
    omega = tail_constant(cfg_game())                     # |V| <= Omega for every policy pair (design 10.5.3)
    checks = {}
    for n in ("sp100_s0", "sp80_s0"):
        s = runs.get(n)
        if s is None:
            continue
        checks[f"{n}: finite losses, gradients and critic statistics"] = s["finite_losses_and_gradients"]
        checks[f"{n}: never terminated; one truncation per finished game"] = (
            s["terminations"] == 0 and s["truncations"] == s["games_finished"])
        checks[f"{n}: critic bounded by the return scale (|V| <= Omega)"] = s["value_absmax_resources"] <= omega
        checks[f"{n}: explained variance improved and is positive"] = (
            s["explained_variance_first_last10"][1] > max(0.0, s["explained_variance_first_last10"][0]))
        checks[f"{n}: gradient norms finite and of order 1 (max < 100)"] = s["grad_norm_max"] < 100
        checks[f"{n}: approximate KL small (mean < 0.05)"] = s["approx_kl_mean"] < 0.05
    s = runs.get("sp100_s0")
    if s is not None:
        checks["sp100_s0: both seats trained equally (exact)"] = s["transitions_seat_a"] == s["transitions_seat_b"]
        checks["sp100_s0: no historical games"] = s["transitions_historical"] == 0
    s = runs.get("sp80_s0")
    if s is not None:
        checks["sp80_s0: historical share near 0.2 of games after the first snapshot"] = (
            0.1 < s["games_historical"] / max(1, s["games_finished"]) < 0.3)
        checks["sp80_s0: both seats trained (seat shares within 0.45-0.55)"] = (
            0.45 < s["transitions_seat_a"] / (s["transitions_seat_a"] + s["transitions_seat_b"]) < 0.55)
    a, b, c = runs.get("repro_s0a"), runs.get("repro_s0b"), runs.get("repro_s1")
    if a and b:
        checks["reproducibility: same configuration and seed give bit-identical weights"] = a["digest"] == b["digest"]
        checks["reproducibility: ... and identical histories"] = a["curve"] == b["curve"]
    if a and c:
        checks["reproducibility: a different seed gives different weights"] = a["digest"] != c["digest"]
    return {"checks": checks, "all_passed": all(checks.values())}


# ---------------------------------------------------------------------------------------------- behavior readout
def action_profile(cfg: GameConfig, agent, games: int = 8, turns: int = WINDOW_DEV, seed: int = 90_000) -> dict:
    """Action-slot frequencies of `agent` in self-play (present turns), plus per-game counters. A readout of what the
    policy does, not a measurement of strategy."""
    from ...env import LaneEnv, play_episode
    env = LaneEnv(cfg, max_turns=turns, cache_size=20_000)
    waves, beh, obj = {"HO": 0, "CL": 0, "TN": 0}, {}, {"P": 0, "V": 0, "none": 0}
    present = absent = 0
    counters = {"recalls": 0.0, "trades_connected": 0.0, "deaths": 0.0, "farm": 0.0, "plates": 0.0}
    legal_checked = 0
    for g in range(games):
        log = play_episode(env, agent, agent, seed + g, record=True)
        for t in log.turns:
            for s in (0, 1):
                a = t["actions"][s]
                assert a in t["legal"][s]
                legal_checked += 1
                lab = cfg.label(a)
                if lab == "ABSENT":
                    absent += 1
                    continue
                present += 1
                w, rest = lab.split("-", 1)
                waves[w] += 1
                beh[rest[0]] = beh.get(rest[0], 0) + 1
                obj["P" if rest.endswith("+P") else "V" if rest.endswith("+V") else "none"] += 1
        for k in counters:
            counters[k] += sum(getattr(log, k)) / 2
    n = max(present, 1)
    return {"games": games, "turns": turns, "present_turns": present, "absent_fraction": absent / max(present + absent, 1),
            "waves": {k: v / n for k, v in waves.items()}, "behaviors": {k: v / n for k, v in sorted(beh.items())},
            "objectives": {k: v / n for k, v in obj.items()},
            "per_game_per_side": {k: v / games for k, v in counters.items()}, "legal_actions_checked": legal_checked}


# ---------------------------------------------------------------------------------------------- evaluation
def evaluate(episodes: int, window: int) -> dict:
    import torch
    from ...env import LaneEnv
    from ...env.agents_m4 import natural_suite
    from ...env.discounted import evaluate_discounted, tail_bound
    from ...rl import NetAgent, PolicyValueNet, load_agent
    cfg = cfg_game()
    agents = {n: load_agent(os.path.join(run_dir(n), "final.pt"), name=n) for n in ("sp100_s0", "sp80_s0")}
    torch.manual_seed(0)                                  # the untrained initial network of the seed-0 runs
    env = LaneEnv(cfg)
    agents["initial_s0"] = NetAgent(PolicyValueNet(env.obs_size, env.n_actions, (128, 128)), "initial_s0")
    suite = natural_suite()
    t0 = time.perf_counter()
    pairs = []
    for x in ("sp100_s0", "sp80_s0", "initial_s0"):
        for y in EVAL_OPPONENTS:
            r = evaluate_discounted(cfg, agents[x], suite[y], episodes=episodes, window=window)
            pairs.append({k: r[k] for k in ("x", "y", "estimate", "stderr", "ci95", "seat_A", "seat_B", "pairs")})
    for x, y in (("sp100_s0", "sp80_s0"), ("sp100_s0", "initial_s0"), ("sp80_s0", "initial_s0")):
        r = evaluate_discounted(cfg, agents[x], agents[y], episodes=episodes, window=window)
        pairs.append({k: r[k] for k in ("x", "y", "estimate", "stderr", "ci95", "seat_A", "seat_B", "pairs")})
    profiles = {n: action_profile(cfg, agents[n]) for n in ("sp100_s0", "sp80_s0", "initial_s0")}
    return {"protocol": "E1", "window": window, "episodes_per_seat": episodes, "gamma": cfg.gamma,
            "tail_bound_guaranteed": tail_bound(cfg, window),
            "label": f"development window W = {window}: the truncation error is bounded by the guaranteed tail bound; "
                     "16 paired episodes per matchup: an integration readout, not a strategy measurement",
            "pairs": pairs, "profiles": profiles, "elapsed_s": time.perf_counter() - t0}


# ---------------------------------------------------------------------------------------------- report
def _f(x, d=2):
    return "n/a" if x is None else f"{x:+.{d}f}" if isinstance(x, float) else str(x)


def report(d: dict) -> str:
    from ...env.discounted import tail_constant
    runs, ev, au = d.get("runs", {}), d.get("evaluation"), d.get("audit")
    cfg = cfg_game()
    omega = tail_constant(cfg)
    L = ["# M4.3a PPO smoke test on `GameConfig.m4_lasthit()` — integration checks, NOT research results", "",
         f"Commit `{d['meta'].get('commit', '?')[:10]}`, command `{d['meta'].get('command', '?')}`. Shared-policy "
         f"self-play PPO (`lanegame.rl.shared`), `PPOConfig.continuing` (gamma {cfg.gamma}, lambda 0.95, segments of "
         f"128 turns, games of 512 turns with seeded staggered first games, reward scale 0.1, value clip 0.5), "
         f"{N_ENVS} environments, snapshots every {SNAP_EVERY:,} agent transitions, pool {POOL}, one torch thread "
         f"per run; the runs were trained in parallel worker processes (throughput is per process under that load).", ""]
    if runs:
        L += ["## Training runs", "",
              md_table(["run", "current fraction", "seed", "agent transitions", "env turns", "updates", "wall-clock s",
                        "transitions/s", "env turns/s", "collect share", "snapshots"],
                       [[n, s["current_fraction"], s["seed"], f"{s['agent_transitions']:,}", f"{s['env_turns']:,}",
                         s["updates"], f"{s['train_elapsed_s']:.0f}", f"{s['transitions_per_s']:,.0f}",
                         f"{s['env_turns_per_s']:,.0f}", f"{s['collect_fraction']:.2f}", s["snapshots"]]
                        for n, s in runs.items()]), "",
              "### Numerical health", "",
              md_table(["run", "finite", "entropy first → last", "policy loss first → last", "value loss first → last 10",
                        "explained variance first → last 10", "grad norm mean / max (pre-clip)", "clip frac", "approx KL mean / max"],
                       [[n, s["finite_losses_and_gradients"],
                         f"{s['entropy_first_last'][0]:.2f} → {s['entropy_first_last'][1]:.2f}",
                         f"{s['policy_loss_first_last'][0]:+.3f} → {s['policy_loss_first_last'][1]:+.3f}",
                         f"{s['value_loss_first_last10'][0]:.3f} → {s['value_loss_first_last10'][1]:.3f}",
                         f"{s['explained_variance_first_last10'][0]:.2f} → {s['explained_variance_first_last10'][1]:.2f}",
                         f"{s['grad_norm_mean']:.2f} / {s['grad_norm_max']:.2f}", f"{s['clipfrac_mean']:.3f}",
                         f"{s['approx_kl_mean']:.4f} / {s['approx_kl_max']:.4f}"] for n, s in runs.items()]), "",
              "### Critic scale (resources; the scaled units of training divided by the reward scale)", "",
              md_table(["run", "V̂(s0)", "mean V̂ (last 10 rollouts)", "max |V̂| (whole run)", "std of λ-returns (last 10)",
                        "mean |r| per turn", "Ω (|V| bound)"],
                       [[n, f"{s['value_s0_resources']:+.2f}", f"{s['value_mean_last10_resources']:+.2f}",
                         f"{s['value_absmax_resources']:.2f}", f"{s['return_std_last10_resources']:.2f}",
                         f"{s['reward_mean_abs_resources']:.3f}", f"{omega:.2f}"] for n, s in runs.items()]), "",
              "By symmetry the self-play value of s0 is 0 (the critic sees the same observation from both seats).", "",
              "### Seats, masks and time semantics", "",
              md_table(["run", "transitions seat A / B", "historical transitions", "single-choice (absent) transitions",
                        "games finished", "historical games", "truncations", "terminations", "mean seat-A return (current games)",
                        "mean learner return (historical games)"],
                       [[n, f"{s['transitions_seat_a']:,} / {s['transitions_seat_b']:,}", f"{s['transitions_historical']:,}",
                         f"{s['single_choice_transitions']:,}", s["games_finished"], s["games_historical"],
                         s["truncations"], s["terminations"], f"{s['current_seat_a_return_mean']:+.2f}",
                         _f(s["historical_return_mean"])] for n, s in runs.items()]), "",
              "Masks: `LaneEnv.step` raises on any illegal action, so a completed run is itself the proof that every "
              "sampled action was legal; the recorded masks are checked against the engine's legal sets in "
              "`tests/test_rl_m4_lasthit.py`. Truncations: one per finished game (the game time limit), none "
              "treated as a termination; the bootstrap values are checked in the same test file.", "",
              "### Reproducibility", "",
              md_table(["run", "seed", "weights digest (SHA-256, first 16)", "torch", "commit"],
                       [[n, s["seed"], s["digest"][:16], s["provenance"]["torch"],
                         (s["provenance"]["git_commit"] or "?")[:10]] for n, s in runs.items()]), ""]
    if au:
        L += ["## Checks", "", md_table(["check", "passed"], [[k, v] for k, v in au["checks"].items()]), "",
              f"**All checks passed: {au['all_passed']}.**", ""]
    if ev:
        L += ["## Evaluation readout (protocol E1, development window)", "",
              f"{ev['label']}. Guaranteed truncation bound at W = {ev['window']}: {ev['tail_bound_guaranteed']:.2f} "
              f"resources; the Monte Carlo interval is the 95% paired CI over seeds. Payoff to x (discounted, resources).", "",
              md_table(["x", "y", "estimate", "95% CI", "x in seat A", "x in seat B"],
                       [[p["x"], p["y"], f"{p['estimate']:+.2f}", f"[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]",
                         f"{p['seat_A']:+.2f}", f"{p['seat_B']:+.2f}"] for p in ev["pairs"]]), "",
              "### What the policies do in self-play (8 games of 460 turns, present turns)", "",
              md_table(["policy", "Hold / Clear / Thin", "H / Y / T / S / R", "Pressure / Vision / none", "absent",
                        "recalls", "connected trades", "deaths", "farm", "plates"],
                       [[n, " / ".join(f"{pr['waves'][k]:.2f}" for k in ("HO", "CL", "TN")),
                         " / ".join(f"{pr['behaviors'].get(k, 0.0):.2f}" for k in "HYTSR"),
                         " / ".join(f"{pr['objectives'][k]:.2f}" for k in ("P", "V", "none")),
                         f"{pr['absent_fraction']:.2f}", f"{pr['per_game_per_side']['recalls']:.1f}",
                         f"{pr['per_game_per_side']['trades_connected']:.1f}", f"{pr['per_game_per_side']['deaths']:.1f}",
                         f"{pr['per_game_per_side']['farm']:.1f}", f"{pr['per_game_per_side']['plates']:.1f}"]
                        for n, pr in ev["profiles"].items()]), "",
              "The behaviour columns are a readout after a bounded budget; strategic claims need the M4.3b pilot."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------- main
def _load() -> dict:
    p = os.path.join(OUT, f"{NAME}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(d: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    dump_json(d, os.path.join(OUT, f"{NAME}.json"))
    with open(os.path.join(OUT, f"{NAME}.md"), "w", encoding="utf-8") as f:
        f.write(report(d))
    print(f"wrote {OUT}/{NAME}.json and .md")


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=["all"])
    ap.add_argument("--steps", type=int, default=409_600, help="agent transitions of sp100 and sp80")
    ap.add_argument("--repro-steps", type=int, default=102_400)
    ap.add_argument("--eval-episodes", type=int, default=16)
    ap.add_argument("--window", type=int, default=WINDOW_DEV)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args(argv)
    stages = ["train", "repro", "evaluate", "report"] if "all" in a.stages else a.stages
    import subprocess
    import sys
    d = _load()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    if "meta" not in d or "train" in stages:                # the provenance of the TRAINING stage is kept
        d["meta"] = {"commit": commit, "command": " ".join([os.path.basename(sys.executable), "-m", __spec__.name]
                                                           + (argv or sys.argv[1:])),
                     "config": cfg_game().to_dict(), "steps": a.steps, "repro_steps": a.repro_steps}
    d.setdefault("runs", {})
    t0 = time.perf_counter()
    for st in stages:
        print(f"===== {st} =====", flush=True)
        if st in ("train", "repro"):
            d["runs"].update(train(st, a.steps, a.repro_steps, a.workers))
            d["audit"] = audit(d["runs"])
        elif st == "evaluate":
            d["evaluation"] = evaluate(a.eval_episodes, a.window)
        elif st == "report":
            d["audit"] = audit(d["runs"])
        else:
            raise SystemExit(f"unknown stage {st}")
        _save(d)
    d["elapsed_s"] = time.perf_counter() - t0
    _save(d)
    if "audit" in d:
        print("checks:", d["audit"]["all_passed"], {k: v for k, v in d["audit"]["checks"].items() if not v})
    return d


def run():
    main([])


if __name__ == "__main__":
    main()
