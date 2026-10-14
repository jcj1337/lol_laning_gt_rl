"""The M4.4a stages other than training: the audit, the evaluation pipeline (smoke sizes in M4.4a, full sizes for
M4.4b) and the demonstration of the strategic diagnostics on the M4.3c pilot finals.

Every stage is a set of picklable jobs run in worker processes (`common.run_jobs`); agents are rebuilt in the worker
from their specification strings (`common.agent_from_spec`, with the compatibility guard). Results are JSON in
results/m44a/ (smoke, demo, audit) or results/m44b/ (main).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Sequence

import numpy as np

from .common import (BEHAVIOR_SEED, CF_ROLL_SEED, CF_STATE_SEED, ENV_ORDER, ENVS, OUT_A, OUT_B, REF_SEED, SCENARIO_SEED,
                     SEEDS, SMOKE_EVAL_SEED, TRACE_SEED, XPLAY_SEED, env_cfg, final_path, load_json, meta, run_jobs,
                     save_json, snapshot_paths)
from .references import CORE, REFERENCES

# initialized scenarios (labelled "initialized:<name>/<gap>", never pooled with s0 games): the M4.3c harness starts
SCENARIOS = (("freeze", "equal"), ("neutral", "large"))

SIZES = {
    "smoke": dict(beh_games=4, beh_turns=300, ref_games=2, ref_opps=("farm4", "shove4"), scen_games=2, scen_turns=150,
                  ref_eps=2, ref_window=200, refs=CORE, xplay_eps=2, xplay_window=200, hist_points=2, hist_eps=2,
                  hist_window=200, cf_keys=("freeze_held", "neutral", "oversized_enemy", "siege", "threatened"),
                  cf_states=2, cf_roll=4, cf_window=60, cf_game_turns=200, traces=1, trace_turns=200,
                  curve_every=4, curve_refs=("farm4", "shove4"), curve_eps=2, curve_window=100),
    "main": dict(beh_games=32, beh_turns=1000, ref_games=12, ref_opps=("farm4", "shove4", "sh_vis", "sh_dive"),
                 scen_games=16, scen_turns=300, ref_eps=32, ref_window=1000, refs=REFERENCES, xplay_eps=32,
                 xplay_window=1000, hist_points=4, hist_eps=16, hist_window=460,
                 cf_keys=("freeze_held", "neutral", "oversized_enemy", "siege", "threatened"), cf_states=8,
                 cf_roll=16, cf_window=300, cf_game_turns=460, traces=3, trace_turns=460,
                 curve_every=4, curve_refs=CORE, curve_eps=16, curve_window=460),
}


# ---------------------------------------------------------------------------------------------- jobs
def behavior_job(args) -> tuple:
    """Records of one (policy, opponent, start) cell -> every metric family, focal = the policy."""
    env, spec, opp, games, turns, seed, scen, gap = args
    import torch
    torch.set_num_threads(1)
    from ...m43c.harness import start as scenario_start
    from .common import agent_from_spec
    from .metrics import all_metrics
    from .records import play_records
    cfg = env_cfg(env)
    x = agent_from_spec(spec, cfg)
    y = x if opp == "self" else agent_from_spec(f"ref:{opp}", cfg)
    start = None if scen == "s0" else scenario_start(scen, cfg, gap)
    label = "s0" if start is None else f"{scen}/{gap}"
    t0 = time.perf_counter()
    recs = play_records(cfg, x, y, games, turns, seed, start, label, both_seats=opp != "self", env_label=env)
    return (env, spec, opp, label), {"games": len(recs), "turns": int(sum(r.turns for r in recs)),
                                      "metrics": all_metrics(recs, focal=x.name), "elapsed_s": time.perf_counter() - t0}


def cf_job(args) -> tuple:
    env, spec, key, n_states, n_roll, window, game_turns, state_seed, roll_seed, protocol = args
    import torch
    torch.set_num_threads(1)
    from . import counterfactual as CF
    from .common import agent_from_spec
    cfg = env_cfg(env)
    x = agent_from_spec(spec, cfg)
    out = CF.run(cfg, x, x, [key], n_states, n_roll, state_seed, roll_seed, window, 1, protocol, game_turns)
    return (env, spec, key, protocol), out["results"][key] | {"elapsed_s": out["elapsed_s"], "window": window,
                                                              "n_roll": n_roll, "protocol": protocol}


def load_job(args) -> tuple:
    """Load a final with the guard, verify its digest against the training record, and check it refuses the other
    environments unless a cross-environment evaluation is requested."""
    phase, env, seed, recorded_digest = args
    from ....rl import checkpoint_info, weights_digest
    from .common import load_policy
    p = final_path(phase, env, seed)
    agent = load_policy(p, env_cfg(env))
    info = checkpoint_info(p)
    refused = {}
    for other in ENV_ORDER:
        if other == env:
            continue
        try:
            load_policy(p, env_cfg(other))
            refused[other] = False
        except ValueError:
            refused[other] = True
    return (env, seed), {"path": p, "digest_verified": True, "digest": info["digest"],
                         "matches_training_record": info["digest"] == recorded_digest == weights_digest(agent.net.state_dict()),
                         "refused_other_envs_without_opt_in": refused,
                         "trained_on": agent.compat.trained_on, "spec": info["spec"]}


def trace_job(args) -> tuple:
    name, env, xs, ys, seed, turns, label, out_dir = args
    import torch
    torch.set_num_threads(1)
    from .trace import export, verify
    rec = export(name, env, xs, ys, seed, turns, label, out_dir)
    return name, {"path": rec["path"], "label": label, "payoff_A": rec["payoff_A"], "turns": rec["turns"],
                  "counters": rec["counters"], "verify": verify(rec["path"])}


# ---------------------------------------------------------------------------------------------- audit
def audit() -> dict:
    """Static and random-play checks of the three environments' interfaces, plus the compatibility matrix of every
    known checkpoint (M4.3b finals, M4.3c pilot finals, M4.4 smoke finals)."""
    import glob
    from ....env import LaneEnv
    from ....env.agents import UniformAgent, episode_streams
    from ....env.observation_m4 import decode_m4, m4_feature_names
    from .actions import EVER_LEGAL, NEVER_LEGAL, action_table, check_four_action_space
    from .common import load_policy
    t0 = time.perf_counter()
    out = {"meta": meta(stage="audit"), "environments": {}}
    for env in ENV_ORDER:
        cfg = env_cfg(env)
        check_four_action_space(cfg)
        e = LaneEnv(cfg, max_turns=400, cache_size=5000)
        seen, bad_obs, bad_mask, mirror_bad, turns = set(), 0, 0, 0, 0
        for g in range(3):
            _, env_ss, ra, rb = episode_streams(np.random.SeedSequence(SMOKE_EVAL_SEED, spawn_key=(g,)))
            obs = e.reset(seed=env_ss)
            ua, ub = UniformAgent(), UniformAgent()
            while True:
                for s in (0, 1):
                    o = obs[s]
                    seen |= set(o.legal)
                    v = o.vector
                    bad_obs += int(len(v) != 88 or v.min() < 0 or v.max() > 1 or not np.all(np.isfinite(v)))
                    bad_mask += int(set(np.flatnonzero(o.mask)) != set(o.legal))
                ph_a, la, ca, _ = decode_m4(obs[0].vector, cfg)
                ph_b, lb, cb, _ = decode_m4(obs[1].vector, cfg)
                mirror_bad += int(not (ph_a == ph_b and lb == la.mirror() and cb == ca.mirror()))
                r = e.step(ua.act(obs[0], ra), ub.act(obs[1], rb))
                turns += 1
                if r.rewards[0] != -r.rewards[1]:
                    raise AssertionError("rewards not zero-sum")
                if r.done:
                    break
                obs = r.obs
        out["environments"][env] = {
            "preset": ENVS[env].preset, "obs_size": e.obs_size, "n_actions": e.n_actions, "turns_played": turns,
            "observed_legal_codes": sorted(seen), "observed_outside_ever_legal": sorted(seen - EVER_LEGAL),
            "ever_legal_not_observed": sorted(EVER_LEGAL - seen), "observations_out_of_range": bad_obs,
            "masks_not_equal_legal": bad_mask, "mirror_failures": mirror_bad,
            "absence_scale": cfg.champion_absent_max}
    out["action_space"] = {"codes": 61, "ever_legal": len(EVER_LEGAL), "never_legal": sorted(NEVER_LEGAL),
                           "table": action_table(env_cfg("aggro"))}
    out["feature_names"] = m4_feature_names(env_cfg("aggro"))
    paths = sorted(glob.glob(os.path.join("results", "rl", "checkpoints", "m43b", "*_s*", "final.pt")))[:3]
    paths += sorted(glob.glob(os.path.join("results", "rl", "checkpoints", "m43c_pilot", "*_s*", "final.pt")))
    paths += sorted(glob.glob(os.path.join("results", "rl", "checkpoints", "m44", "smoke", "*_s?", "final.pt")))
    matrix = {}
    for p in paths:
        row = {}
        for env in ENV_ORDER:
            try:
                load_policy(p, env_cfg(env))
                row[env] = "loads"
            except ValueError as ex:
                row[env] = "refused: shape" if "action outputs" in str(ex) else "refused: other configuration"
        matrix[p.replace("\\", "/")] = row
    out["checkpoint_compatibility"] = matrix
    out["elapsed_s"] = time.perf_counter() - t0
    save_json("audit", out)
    return out


# ---------------------------------------------------------------------------------------------- evaluation
def evaluation(phase: str, envs: Sequence[str] = ENV_ORDER, seeds: Sequence[int] = SEEDS, workers: int = 9) -> dict:
    """The evaluation pipeline on the finals of `phase` (sizes SIZES[phase]); the smoke version exercises every
    instrument end to end at small sizes, the main version is the M4.4b evaluation."""
    from . import evaluate as EV
    z = SIZES[phase]
    out_dir = OUT_A if phase == "smoke" else OUT_B
    train_rec = load_json(f"train_{phase}", out_dir)
    if train_rec is None:
        raise RuntimeError(f"no train_{phase}.json: train the phase first")
    finals = [(e, s) for e in envs for s in seeds if os.path.exists(final_path(phase, e, s))]
    t0 = time.perf_counter()
    data = {"meta": meta(stage=f"eval_{phase}", sizes={k: (list(v) if isinstance(v, tuple) else v) for k, v in z.items()},
                         study="M4.4b" if phase == "main" else "M4.4a"),
            "finals": [f"final:{phase}:{e}:{s}" for e, s in finals]}
    # 1. loading, digests and the cross-environment guard
    jobs = [(phase, e, s, train_rec["runs"][f"{e}_s{s}"]["digest"]) for e, s in finals]
    data["load"] = {f"{k[0]}_s{k[1]}": v for k, v in run_jobs(load_job, jobs, workers, "load")}
    # 2. strength: references, cross-play, history (evaluate_discounted, paired seats, tail bound)
    ev_jobs = []
    for e in envs:
        specs = [f"final:{phase}:{e}:{s}" for ee, s in finals if ee == e]
        ev_jobs += EV.jobs_references(e, specs, z["refs"], z["ref_eps"], z["ref_window"], REF_SEED)
        ev_jobs += EV.jobs_crossplay(e, specs, z["xplay_eps"], z["xplay_window"], XPLAY_SEED)
        for ee, s in finals:
            if ee != e:
                continue
            total = train_rec["runs"][f"{e}_s{s}"]["agent_transitions"]
            # the snapshot taken at the last update IS the final network: excluded (its value against the final is 0
            # by the paired-seat symmetry, an identity, not a measurement)
            snaps = [n for n, _ in snapshot_paths(phase, e, s) if n < total]
            pts = sorted({snaps[int(i)] for i in np.linspace(0, len(snaps) - 1, z["hist_points"])}) if snaps else []
            ev_jobs += EV.jobs_history(e, phase, s, pts, z["hist_eps"], z["hist_window"], XPLAY_SEED + 1)
            # learning curve: every curve_every-th snapshot (and the final) against the curve references
            curve = [n for i, (n, _) in enumerate(snapshot_paths(phase, e, s), start=1) if i % z["curve_every"] == 0]
            ev_jobs += EV.jobs_curve(e, phase, s, curve, z["curve_refs"], z["curve_eps"], z["curve_window"], REF_SEED + 1)
    data["strength"] = EV.run(ev_jobs, workers)
    # 3. behaviour: natural self-play, natural vs references, initialized scenarios
    bj = []
    for e, s in finals:
        spec = f"final:{phase}:{e}:{s}"
        bj.append((e, spec, "self", z["beh_games"], z["beh_turns"], BEHAVIOR_SEED, "s0", "equal"))
        for r in z["ref_opps"]:
            bj.append((e, spec, r, z["ref_games"], z["beh_turns"], BEHAVIOR_SEED + 1, "s0", "equal"))
        for scen, gap in SCENARIOS:
            bj.append((e, spec, "self", z["scen_games"], z["scen_turns"], SCENARIO_SEED, scen, gap))
    data["behavior"] = {"|".join(k): v for k, v in run_jobs(behavior_job, bj, workers, "behavior")}
    # 4. counterfactuals
    cj = []
    for i, (e, s) in enumerate(finals):
        for j, key in enumerate(z["cf_keys"]):
            cj.append((e, f"final:{phase}:{e}:{s}", key, z["cf_states"], z["cf_roll"], z["cf_window"],
                       z["cf_game_turns"], CF_STATE_SEED + 10 * j, CF_ROLL_SEED + 10 * j, "policy"))
    data["counterfactual"] = {"|".join(k): v for k, v in run_jobs(cf_job, cj, workers, "counterfactual")}
    # 5. viewer traces with replay verification
    tj = []
    tdir = os.path.join(out_dir, "trajectories")
    for e in envs:
        ss = [s for ee, s in finals if ee == e][:z["traces"]]
        for s in ss:
            spec = f"final:{phase}:{e}:{s}"
            tj.append((f"{phase}_{e}_s{s}_selfplay", e, spec, spec, TRACE_SEED, z["trace_turns"],
                       f"{phase} final {e} seed {s} against itself"))
            tj.append((f"{phase}_{e}_s{s}_vs_shove4", e, spec, "ref:shove4", TRACE_SEED, z["trace_turns"],
                       f"{phase} final {e} seed {s} (A) against the Shove reference (B)"))
    data["traces"] = dict(run_jobs(trace_job, [t + (tdir,) for t in tj], workers, "traces"))
    data["elapsed_s"] = time.perf_counter() - t0
    save_json(f"eval_{phase}", data, out_dir)
    return data


# ---------------------------------------------------------------------------------------------- demonstration
DEMO_SPECS = (("four", "pilot:base:0"), ("four", "pilot:base:1"), ("econ", "pilot:candidate:0"),
              ("econ", "pilot:candidate:1"))
DEMO = dict(beh_games=16, beh_turns=500, ref_games=4, ref_turns=300, scen_games=8, scen_turns=300,
            cf_keys=("freeze_held", "neutral", "oversized_enemy", "siege", "threatened"), cf_states=6, cf_roll=8,
            cf_window=150, cf_game_turns=300)


def demo(workers: int = 9) -> dict:
    """The strategic diagnostics and counterfactuals on the M4.3c pilot finals (EXPLORATORY checkpoints of
    m4_four_actions() and m4_four_actions_econ(), 2 seeds x 2,048,000 transitions): a demonstration that the
    instruments separate state-dependent strategy from action frequency on non-trivial policies, not an M4.4 result."""
    z = DEMO
    t0 = time.perf_counter()
    bj = []
    for e, spec in DEMO_SPECS:
        bj.append((e, spec, "self", z["beh_games"], z["beh_turns"], BEHAVIOR_SEED, "s0", "equal"))
        for r in ("farm4", "shove4"):
            bj.append((e, spec, r, z["ref_games"], z["ref_turns"], BEHAVIOR_SEED + 1, "s0", "equal"))
        for scen, gap in SCENARIOS:
            bj.append((e, spec, "self", z["scen_games"], z["scen_turns"], SCENARIO_SEED, scen, gap))
    beh = {"|".join(k): v for k, v in run_jobs(behavior_job, bj, workers, "demo behavior")}
    cj = []
    for e, spec in DEMO_SPECS:
        if not spec.endswith(":0"):
            continue
        for j, key in enumerate(z["cf_keys"]):
            for protocol in (("policy", "modal") if key == "freeze_held" else ("policy",)):
                cj.append((e, spec, key, z["cf_states"], z["cf_roll"], z["cf_window"], z["cf_game_turns"],
                           CF_STATE_SEED + 10 * j, CF_ROLL_SEED + 10 * j, protocol))
    cf = {"|".join(k): v for k, v in run_jobs(cf_job, cj, workers, "demo counterfactual")}
    data = {"meta": meta(stage="demo", specs=[list(x) for x in DEMO_SPECS], sizes=dict(z),
                         label="DEMONSTRATION on the EXPLORATORY M4.3c pilot finals; not an M4.4 result"),
            "behavior": beh, "counterfactual": cf, "elapsed_s": time.perf_counter() - t0}
    save_json("demo", data)
    return data


__all__ = ["SIZES", "SCENARIOS", "audit", "evaluation", "demo", "behavior_job", "cf_job", "load_job", "trace_job",
           "DEMO", "DEMO_SPECS"]
