"""The M4.3b follow-up: strategic validation of the wave-management conclusions (brief §2-§6).

Four stages, all on the trained M4.3b finals (no training), all paired on common random numbers, results in
results/rl/m43b/validation/:

  wave_cf     state-level counterfactuals with WAVE-ONLY interventions (`intervene.WavePlan`, both rules) next to the
              original full-action plans (`plans.make_plan`) on the SAME sampled states and rollout streams, from
              reachable states of games of the policy against each opponent of OPPONENTS (the policy itself, a pure
              last-hitter, the reactive slow pusher, the aggressive clearer), in the situations SITUATIONS.
              Advantage = Q(plan) - Q(policy's own play), W = 300 turns discounted, per state; states are the unit
              of the per-cell interval; training seeds (base s0, s1, s2) are the unit of the across-seed interval.
  games       whole-game value of a wave rule with adaptive behaviour (`intervene.InterventionAgent`) against each
              opponent, W = 1000 (the headline window), 32 paired episodes from s0, both seats; the difference to the
              unmodified policy on the same episode seeds, with its paired interval.
  horizon     the wave_cf comparison repeated at W = 1000 on the same states and rollout roots for the high-priority
              cells (the finite-window estimate at 300 and at 1000 from the same rollouts, with the tail between them).
  freeze      (a) every freeze-like state of wave_cf classified: superficially freeze-like (the mask), maintainable
              (the wave-only freeze controller holds the enemy front at own -1 for >= FREEZE_HELD of the next 20
              turns against a Holding opponent), favourable (maintainable and the focal champion at least even in
              attack power and HP); the freeze advantage by class. (b) SYNTHETIC sensitivity: the deterministic
              neutral and freeze starts of the M4.2e-B harness (`zoning_natural.starts`) with assigned power gaps
              (`with_gap`: equal, moderate, large, levels, large_levels; the focal champion ahead or behind), the same
              candidates against the same opponents. These states are constructed, not sampled from play, and are
              labelled as such everywhere.

Seeds: SeedSequence roots 67,500-67,999 (disjoint from every other stream of the pilot; `common`).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....io import dump_json
from .behavior import CF_WINDOW, FEATURE_KEYS, sample_states, situation_masks, turn_features
from .common import (OUT, PRIMARY, SEEDS, WINDOW_HEAD, agent_from_spec, cfg_game, mean_ci, meta, read_json, run_jobs)
from .intervene import GAME_FNS, WAVE_FNS, InterventionAgent, wave_plan
from .plans import make_plan

VAL_DIR = os.path.join(OUT, "validation")
VAL_STATE_SEED, VAL_ROLL_SEED, VAL_GAME_SEED, VAL_SYN_SEED, VAL_FREEZE_SEED = 67_500, 67_600, 67_700, 67_800, 67_900

POLICIES = tuple(f"final:{PRIMARY}:{s}" for s in SEEDS) + ("final:ent03:0",)
OPPONENTS = ("self", "farm", "push_punish", "clear_punish")
GAME_OPPONENTS = ("self", "farm", "push_punish", "clear_punish", "freeze_punish", "attacker")
SITUATIONS = ("neutral", "enemy_pushing", "freeze_like")
PLAN_TURNS = {"hold": 14, "thin": 14, "clear": 14, "slow_push": 14, "freeze": 20}
MARGINAL = ("hold", "thin", "clear", "slow_push", "freeze")
CONDITIONAL = ("hold", "slow_push", "freeze")
FULL = ("hold", "slow_push", "freeze")
V_STATES, V_ROLLOUTS = 8, 16
GAME_EPISODES = 32
HORIZON_POLICIES = tuple(f"final:{PRIMARY}:{s}" for s in SEEDS)
HORIZON_OPPONENTS = ("self", "farm", "clear_punish")
HORIZON_SITUATIONS = ("neutral", "freeze_like")
HORIZON_CANDS = ("m:hold", "m:clear", "m:slow_push", "m:freeze", "f:freeze")
HORIZON_MARKS = (300, 460, 1000)
FREEZE_HELD, FREEZE_WATCH = 10, 20
SYN_STARTS = ("neutral", "freeze")
SYN_GAPS = ("equal", "moderate", "large", "levels", "large_levels")
SYN_CANDS = ("m:hold", "m:thin", "m:clear", "m:slow_push", "m:freeze")


# ---------------------------------------------------------------------------------------------- helpers
def vpath(name: str, ext: str = "json") -> str:
    return os.path.join(VAL_DIR, f"{name}.{ext}")


def vsave(name: str, data: dict) -> None:
    os.makedirs(VAL_DIR, exist_ok=True)
    dump_json(data, vpath(name))
    print(f"wrote {vpath(name)}", flush=True)


def vload(name: str) -> Optional[dict]:
    """results/rl/m43b/validation/<name>.json, or its gzipped form when only that exists."""
    p = vpath(name)
    if os.path.exists(p):
        return read_json(p)
    if os.path.exists(p + ".gz"):
        return read_json(p + ".gz")
    return None


def opponent_agent(opp: str, spec: str, agent, cfg):
    """The opponent named `opp` for the policy `spec`: 'self' is the policy itself, 'attacker' its own attacker."""
    if opp == "self":
        return agent
    if opp == "attacker":
        parts = spec.split(":")
        return agent_from_spec(f"attacker:{parts[1]}:{parts[2]}", cfg)
    return agent_from_spec(f"ref:{opp}", cfg)


def candidates(include_full: bool = True, marginal=MARGINAL, conditional=CONDITIONAL, full=FULL) -> dict:
    from ....diagnostics import FollowPolicy
    c = {"policy": FollowPolicy()}
    for k in marginal:
        c[f"m:{k}"] = wave_plan(k, "marginal", PLAN_TURNS[k])
    for k in conditional:
        c[f"c:{k}"] = wave_plan(k, "conditional", PLAN_TURNS[k])
    if include_full:
        for k in full:
            c[f"f:{k}"] = make_plan(k)
    return c


def _decomp(res, c: str) -> dict:
    return {"farm_diff": float((res.farm[c][:, 0] - res.farm[c][:, 1]).mean()),
            "deaths": res.deaths[c].mean(0).tolist(), "recalls": res.recalls[c].mean(0).tolist(),
            "crashes": res.m4[c]["crashes"].mean(0).tolist() if c in res.m4 else None}


def _seat_agents(agent, opp_agent, seat: int):
    return (agent, opp_agent) if seat == 0 else (opp_agent, agent)


def summarize_states(states: List[dict], cands: Sequence[str]) -> dict:
    """Per candidate: the mean paired advantage over the states (states are the unit of the interval), the share of
    states with a significant gain / loss, and the farm / death decomposition."""
    out = {"n_states": len(states)}
    for c in cands:
        rows = [s for s in states if c in s["adv"]]
        if not rows:
            continue
        d = np.array([s["adv"][c][0] for s in rows])
        se = np.array([s["adv"][c][1] for s in rows])
        out[c] = {"n": int(len(d)), "mean_adv": float(d.mean()),
                  "ci": float(1.96 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
                  "share_pos": float(np.mean(d > 1.96 * se)), "share_neg": float(np.mean(d < -1.96 * se)),
                  "farm_diff": float(np.mean([s["decomposition"][c]["farm_diff"] - s["decomposition"]["policy"]["farm_diff"]
                                              for s in rows])),
                  "deaths_focal": float(np.mean([s["decomposition"][c]["deaths"][0] for s in rows])),
                  "deaths_policy": float(np.mean([s["decomposition"]["policy"]["deaths"][0] for s in rows]))}
    return out


def across_seeds(results: Dict[str, dict], specs: Sequence[str], key_fn, cands: Sequence[str]) -> dict:
    """Mean over training seeds of the per-seed cell means (seeds are the unit; n = 3 is indicative). `key_fn(spec)`
    names the cell of each spec."""
    out = {}
    per = {}
    for spec in specs:
        k = key_fn(spec)
        if k in results:
            per.setdefault(spec, results[k]["summary"])
    for c in cands:
        vals = [per[s][c]["mean_adv"] for s in per if c in per[s]]
        if len(vals) >= 2:
            m, h = mean_ci(vals)
            out[c] = {"mean": m, "ci_seeds": h, "n_seeds": len(vals), "per_seed": vals}
    return out


# ---------------------------------------------------------------------------------------------- stage: wave_cf
def _wave_cf_job(args) -> tuple:
    spec, opp, key, n_states, n_roll, state_seed, roll_seed, window = args
    import torch
    from ....diagnostics import ProbCache, branch, observation
    from .behavior import category_masses, full_distribution
    torch.set_num_threads(1)
    cfg = cfg_game()
    agent = agent_from_spec(spec, cfg)
    opp_agent = opponent_agent(opp, spec, agent, cfg)
    t0 = time.perf_counter()
    states = sample_states(cfg, agent, situation_masks([key]), n_states, state_seed,
                           opponent=None if opp == "self" else opp_agent)[key]
    cache = ProbCache()
    out = []
    for i, (snap, seat, feats) in enumerate(states):
        cands = candidates()
        res = branch(cfg, snap, seat, cands, _seat_agents(agent, opp_agent, seat), n_roll, roll_seed, key=(i,),
                     cache=cache, max_turns=window)
        ob = observation(cfg, snap, seat)
        wv = category_masses(full_distribution(agent, ob))["wave"]
        out.append({"seat": seat, "k": snap.k, "features": feats, "snapshot": snap.to_dict(),
                    "pi_wave": {"Hold": float(wv[0]), "Thin": float(wv[1]), "Clear": float(wv[2])},
                    "q": {c: res.mean(c) for c in res.candidates},
                    "adv": {c: res.diff(c, "policy") for c in res.candidates if c != "policy"},
                    "decomposition": {c: _decomp(res, c) for c in res.candidates}})
    cand_names = [c for c in candidates() if c != "policy"]
    return f"{spec}|{opp}|{key}", {"spec": spec, "opponent": opp, "situation": key, "candidates": cand_names,
                                   "n_found": len(states), "states": out, "window": window,
                                   "elapsed_s": time.perf_counter() - t0}


def wave_cf(specs: Sequence[str] = POLICIES, opponents: Sequence[str] = OPPONENTS, situations: Sequence[str] = SITUATIONS,
            workers: int = 9, n_states: int = V_STATES, n_roll: int = V_ROLLOUTS, force: bool = False) -> dict:
    t0 = time.perf_counter()
    prev = (vload("wave_cf") or {}) if not force else {}
    results = dict(prev.get("results", {}))
    jobs = []
    for i, spec in enumerate(specs):
        for j, opp in enumerate(opponents):
            for l, key in enumerate(situations):
                jobs.append((spec, opp, key, n_states, n_roll, VAL_STATE_SEED + 100 * i + 10 * j + l,
                             VAL_ROLL_SEED + 100 * i + 10 * j + l, CF_WINDOW))
    todo = [j for j in jobs if f"{j[0]}|{j[1]}|{j[2]}" not in results]
    print(f"wave_cf: {len(jobs)} jobs, {len(todo)} to compute", flush=True)
    res = run_jobs(_wave_cf_job, todo, workers, label="wave_cf")
    results.update(dict(res))
    cand_names = [c for c in candidates() if c != "policy"]
    for v in results.values():
        v["summary"] = summarize_states(v["states"], v["candidates"])
    seeds_summary = {}
    for opp in opponents:
        for key in situations:
            seeds_summary[f"{opp}|{key}"] = across_seeds(results, [f"final:{PRIMARY}:{s}" for s in SEEDS],
                                                         lambda sp, o=opp, k=key: f"{sp}|{o}|{k}", cand_names)
    data = {"meta": meta(stage="validation.wave_cf", window=CF_WINDOW, states=n_states, rollouts=n_roll,
                         plan_turns=PLAN_TURNS, state_seed=VAL_STATE_SEED, roll_seed=VAL_ROLL_SEED),
            "policies": list(specs), "opponents": list(opponents), "situations": list(situations),
            "candidates": cand_names, "elapsed_s": time.perf_counter() - t0, "results": results,
            "across_seeds": seeds_summary}
    vsave("wave_cf", data)
    return data


# ---------------------------------------------------------------------------------------------- stage: games
def paired_values(cfg, x, y, episodes: int, seed: int, window: int) -> dict:
    """Per-episode paired values of x against y from s0 (x in seat A then in seat B on the same episode seed, as in
    `evaluate_discounted`), discounted over `window`."""
    from ....env import LaneEnv
    from ....env.discounted import _play, discounted_sum
    env = LaneEnv(cfg, cache_size=20_000)
    g = cfg.gamma
    children = np.random.SeedSequence(seed).spawn(episodes)
    va, vb = [], []
    for ss in children:
        ra, _ = _play(env, (x, y), np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (0, 0)), window)
        rb, _ = _play(env, (y, x), np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (0, 0)), window)
        va.append(discounted_sum(ra, g))
        vb.append(-discounted_sum(rb, g))
    va, vb = np.array(va), np.array(vb)
    return {"pair": (0.5 * (va + vb)).tolist(), "seat_A": va.tolist(), "seat_B": vb.tolist()}


def _games_job(args) -> tuple:
    spec, opp, variant, episodes, seed, window = args
    import torch
    torch.set_num_threads(1)
    cfg = cfg_game()
    agent = agent_from_spec(spec, cfg)
    opp_agent = opponent_agent(opp, spec, agent, cfg)
    t0 = time.perf_counter()
    if variant == "none":
        x = agent
    else:
        rule, kind = variant.split(":")
        x = InterventionAgent(agent, WAVE_FNS[kind], {"m": "marginal", "c": "conditional"}[rule], name=f"{spec}[{variant}]")
    v = paired_values(cfg, x, opp_agent, episodes, seed, window)
    return f"{spec}|{opp}|{variant}", {"spec": spec, "opponent": opp, "variant": variant, "episodes": episodes,
                                       "window": window, **v, "elapsed_s": time.perf_counter() - t0}


def game_variants() -> List[str]:
    return ["none"] + [f"m:{k}" for k in GAME_FNS] + [f"c:{k}" for k in ("hold", "reactive_push", "freeze")]


def games(specs: Sequence[str] = POLICIES, opponents: Sequence[str] = GAME_OPPONENTS, workers: int = 9,
          episodes: int = GAME_EPISODES, window: int = WINDOW_HEAD, force: bool = False) -> dict:
    t0 = time.perf_counter()
    prev = (vload("games") or {}) if not force else {}
    results = dict(prev.get("results", {}))
    variants = game_variants()
    jobs = []
    for spec in specs:
        for opp in opponents:
            if opp == "attacker" and not has_attacker(spec):
                continue
            for v in variants:
                jobs.append((spec, opp, v, episodes, VAL_GAME_SEED, window))
    todo = [j for j in jobs if f"{j[0]}|{j[1]}|{j[2]}" not in results]
    print(f"games: {len(jobs)} jobs, {len(todo)} to compute", flush=True)
    res = run_jobs(_games_job, todo, workers, label="games")
    results.update(dict(res))
    # paired differences to the unmodified policy, per (policy, opponent)
    diffs = {}
    for spec in specs:
        for opp in opponents:
            base = results.get(f"{spec}|{opp}|none")
            if base is None:
                continue
            cell = {"none": {"mean": float(np.mean(base["pair"])),
                             "ci": float(1.96 * np.std(base["pair"], ddof=1) / np.sqrt(len(base["pair"])))}}
            for v in variants[1:]:
                r = results.get(f"{spec}|{opp}|{v}")
                if r is None:
                    continue
                d = np.array(r["pair"]) - np.array(base["pair"])
                cell[v] = {"mean": float(np.mean(r["pair"])), "diff": float(d.mean()),
                           "ci": float(1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                           "seat_A_diff": float(np.mean(r["seat_A"]) - np.mean(base["seat_A"])),
                           "seat_B_diff": float(np.mean(r["seat_B"]) - np.mean(base["seat_B"]))}
            diffs[f"{spec}|{opp}"] = cell
    seeds_summary = {}
    for opp in opponents:
        per = [diffs[f"final:{PRIMARY}:{s}|{opp}"] for s in SEEDS if f"final:{PRIMARY}:{s}|{opp}" in diffs]
        if len(per) >= 2:
            seeds_summary[opp] = {}
            for v in variants[1:]:
                vals = [c[v]["diff"] for c in per if v in c]
                if len(vals) >= 2:
                    m, h = mean_ci(vals)
                    seeds_summary[opp][v] = {"mean": m, "ci_seeds": h, "n_seeds": len(vals), "per_seed": vals}
    data = {"meta": meta(stage="validation.games", window=window, episodes=episodes, seed=VAL_GAME_SEED),
            "policies": list(specs), "opponents": list(opponents), "variants": variants,
            "elapsed_s": time.perf_counter() - t0, "results": results, "diffs": diffs, "across_seeds": seeds_summary}
    vsave("games", data)
    return data


def has_attacker(spec: str) -> bool:
    from .common import attacker_dir
    parts = spec.split(":")
    return os.path.exists(os.path.join(attacker_dir(parts[1], int(parts[2])), "final.pt"))


# ---------------------------------------------------------------------------------------------- stage: horizon
def _horizon_job(args) -> tuple:
    spec, opp, key, n_states, n_roll, state_seed, roll_seed, window, marks, cand_names = args
    import torch
    from ....diagnostics import ProbCache, rollout, rollout_seed
    from ....env import LaneEnv
    torch.set_num_threads(1)
    cfg = cfg_game()
    g = cfg.gamma
    agent = agent_from_spec(spec, cfg)
    opp_agent = opponent_agent(opp, spec, agent, cfg)
    t0 = time.perf_counter()
    states = sample_states(cfg, agent, situation_masks([key]), n_states, state_seed,
                           opponent=None if opp == "self" else opp_agent)[key]
    cache = ProbCache()
    out = []
    for i, (snap, seat, feats) in enumerate(states):
        env = LaneEnv(snap.config(cfg), cache_size=20_000)
        all_c = candidates()
        cands = {"policy": all_c["policy"], **{c: all_c[c] for c in cand_names if c in all_c}}
        agents = _seat_agents(agent, opp_agent, seat)
        G = {c: {m: [] for m in marks} for c in cands}
        for c, plan in cands.items():
            for m in range(n_roll):
                r = rollout(env, snap, seat, plan, agents, rollout_seed(roll_seed, m, i), cache, record=True,
                            max_turns=window)
                rw = np.array([inf["reward"] for inf in r.infos]) * (1.0 if seat == 0 else -1.0)   # zero-sum: A's reward
                w = g ** np.arange(len(rw))
                for mk in marks:
                    G[c][mk].append(float((rw[:mk] * w[:mk]).sum()))
                assert abs(G[c][window][-1] - r.ret) < 1e-6
        rec = {"seat": seat, "k": snap.k, "features": feats, "q": {}, "adv": {}}
        for c in cands:
            rec["q"][c] = {str(mk): [float(np.mean(G[c][mk])), float(np.std(G[c][mk], ddof=1) / np.sqrt(n_roll))]
                           for mk in marks}
            if c != "policy":
                rec["adv"][c] = {}
                for mk in marks:
                    d = np.array(G[c][mk]) - np.array(G["policy"][mk])
                    rec["adv"][c][str(mk)] = [float(d.mean()), float(d.std(ddof=1) / np.sqrt(n_roll))]
                dt = np.array(G[c][window]) - np.array(G["policy"][window]) - (np.array(G[c][marks[0]]) - np.array(G["policy"][marks[0]]))
                rec["adv"][c]["tail"] = [float(dt.mean()), float(dt.std(ddof=1) / np.sqrt(n_roll))]
        out.append(rec)
    return f"{spec}|{opp}|{key}", {"spec": spec, "opponent": opp, "situation": key, "candidates": list(cand_names),
                                   "n_found": len(states), "marks": list(marks), "states": out, "window": window,
                                   "elapsed_s": time.perf_counter() - t0}


def summarize_horizon(states: List[dict], cands: Sequence[str], marks: Sequence[int]) -> dict:
    out = {"n_states": len(states)}
    for c in cands:
        rows = [s for s in states if c in s["adv"]]
        if not rows:
            continue
        cell = {}
        for mk in list(marks) + ["tail"]:
            d = np.array([s["adv"][c][str(mk)][0] for s in rows])
            se = np.array([s["adv"][c][str(mk)][1] for s in rows])
            cell[str(mk)] = {"mean_adv": float(d.mean()),
                             "ci": float(1.96 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
                             "share_pos": float(np.mean(d > 1.96 * se)), "share_neg": float(np.mean(d < -1.96 * se))}
        # states where the sign of the estimate changes between the shortest and the longest window
        a, b = np.array([s["adv"][c][str(marks[0])][0] for s in rows]), np.array([s["adv"][c][str(marks[-1])][0] for s in rows])
        cell["sign_changes"] = int(np.sum(np.sign(a) != np.sign(b)))
        out[c] = cell
    return out


def horizon(specs: Sequence[str] = HORIZON_POLICIES, opponents: Sequence[str] = HORIZON_OPPONENTS,
            situations: Sequence[str] = HORIZON_SITUATIONS, workers: int = 9, n_states: int = V_STATES,
            n_roll: int = V_ROLLOUTS, window: int = WINDOW_HEAD, force: bool = False) -> dict:
    """Same states and rollout roots as wave_cf (same seeds), so the W = 300 estimate here equals wave_cf's for the
    shared candidates, and the W = 1000 estimate adds the tail of the same rollouts."""
    t0 = time.perf_counter()
    prev = (vload("horizon") or {}) if not force else {}
    results = dict(prev.get("results", {}))
    marks = tuple(sorted({m for m in HORIZON_MARKS if m < window} | {window}))
    jobs = []
    for spec in specs:
        i = POLICIES.index(spec)
        for opp in opponents:
            j = OPPONENTS.index(opp)
            for key in situations:
                l = SITUATIONS.index(key)
                cands = [c for c in HORIZON_CANDS if not (c.startswith("f:") and opp != "self")]
                jobs.append((spec, opp, key, n_states, n_roll, VAL_STATE_SEED + 100 * i + 10 * j + l,
                             VAL_ROLL_SEED + 100 * i + 10 * j + l, window, marks, tuple(cands)))
    todo = [j for j in jobs if f"{j[0]}|{j[1]}|{j[2]}" not in results]
    print(f"horizon: {len(jobs)} jobs, {len(todo)} to compute", flush=True)
    res = run_jobs(_horizon_job, todo, workers, label="horizon")
    results.update(dict(res))
    for v in results.values():
        v["summary"] = summarize_horizon(v["states"], v["candidates"], v["marks"])
    seeds_summary = {}
    for opp in opponents:
        for key in situations:
            cell = {}
            for c in HORIZON_CANDS:
                for mk in list(marks) + ["tail"]:
                    vals = [results[f"final:{PRIMARY}:{s}|{opp}|{key}"]["summary"][c][str(mk)]["mean_adv"]
                            for s in SEEDS if f"final:{PRIMARY}:{s}|{opp}|{key}" in results
                            and c in results[f"final:{PRIMARY}:{s}|{opp}|{key}"]["summary"]]
                    if len(vals) >= 2:
                        m, h = mean_ci(vals)
                        cell.setdefault(c, {})[str(mk)] = {"mean": m, "ci_seeds": h, "n_seeds": len(vals)}
            seeds_summary[f"{opp}|{key}"] = cell
    from ....env.discounted import tail_bound
    data = {"meta": meta(stage="validation.horizon", window=window, marks=marks, states=n_states,
                         rollouts=n_roll, tail_bounds={str(m): tail_bound(cfg_game(), m) for m in marks}),
            "policies": list(specs), "opponents": list(opponents), "situations": list(situations),
            "candidates": list(HORIZON_CANDS), "elapsed_s": time.perf_counter() - t0, "results": results,
            "across_seeds": seeds_summary}
    vsave("horizon", data)
    return data


# ---------------------------------------------------------------------------------------------- stage: freeze
def held_turns(infos: Sequence[dict], seat: int, watch: int = FREEZE_WATCH) -> int:
    """Turns among the first `watch` after which the enemy front stands at the focal's own -1 with HP > 0 (a held
    freeze: the enemy wave in front of the own tower, not crashed into it)."""
    from ....state import State
    n = 0
    for inf in infos[:watch]:
        st = State.from_list(inf["state_after"])
        ex, eh = st.side(1 - seat)[:2]
        if eh > 0 and -ex == -1:
            n += 1
    return n


def _freeze_job(args) -> tuple:
    spec, states, n_roll, seed = args
    import torch
    from ....diagnostics import ProbCache, Snapshot, rollout, rollout_seed
    from ....env import LaneEnv
    torch.set_num_threads(1)
    cfg = cfg_game()
    agent = agent_from_spec(spec, cfg)
    farm = agent_from_spec("ref:farm", cfg)
    cache = ProbCache()
    env = LaneEnv(cfg, cache_size=20_000)
    out = []
    for i, st in enumerate(states):
        snap, seat = Snapshot.from_dict(st["snapshot"]), st["seat"]
        plan = wave_plan("freeze", "marginal", FREEZE_WATCH)
        held = {}
        for opp_name, opp in (("farm", farm), ("self", agent)):
            hs = []
            for m in range(n_roll):
                r = rollout(env, snap, seat, plan, _seat_agents(agent, opp, seat), rollout_seed(seed, m, i), cache,
                            record=True, max_turns=FREEZE_WATCH)
                hs.append(held_turns(r.infos, seat))
            held[opp_name] = float(np.mean(hs))
        f = st["features"]
        maintainable = held["farm"] >= FREEZE_HELD
        favourable = maintainable and f["power"] >= 1.0 and f["hp_frac"] >= 0.5
        out.append({"index": i, "seat": seat, "k": st["k"], "held_vs_farm": held["farm"], "held_vs_self": held["self"],
                    "power": f["power"], "hp_frac": f["hp_frac"], "en_fh": f["en_fh"], "own_fh": f["own_fh"],
                    "class": "favourable" if favourable else ("maintainable" if maintainable else "superficial"),
                    "adv_m_freeze": st["adv"].get("m:freeze"), "adv_c_freeze": st["adv"].get("c:freeze"),
                    "adv_f_freeze": st["adv"].get("f:freeze"), "adv_m_hold": st["adv"].get("m:hold")})
    return spec, out


def _synthetic_job(args) -> tuple:
    spec, start_name, gap, focal_seat, opp, n_roll, seed = args
    import torch
    from ....diagnostics import ProbCache, Snapshot, branch
    from ...m4 import zoning_natural as zn
    torch.set_num_threads(1)
    cfg = cfg_game()
    agent = agent_from_spec(spec, cfg)
    opp_agent = opponent_agent(opp, spec, agent, cfg)
    lane, m4, k = zn.starts(cfg)[start_name]
    m4 = zn.with_gap(m4, cfg, gap)                    # A (seat 0) gets the lead; the focal is ahead in seat 0, behind in seat 1
    snap = Snapshot(lane, m4, k, (cfg.skill_a, cfg.skill_b))
    all_c = candidates(include_full=False)
    cands = {"policy": all_c["policy"], **{c: all_c[c] for c in SYN_CANDS}}
    res = branch(cfg, snap, focal_seat, cands, _seat_agents(agent, opp_agent, focal_seat), n_roll, seed,
                 key=(SYN_STARTS.index(start_name), SYN_GAPS.index(gap), focal_seat), cache=ProbCache(),
                 max_turns=CF_WINDOW)
    from ....diagnostics import observation
    feats = turn_features(observation(cfg, snap, focal_seat), cfg)
    key = f"{spec}|{start_name}|{gap}|{'ahead' if focal_seat == 0 else 'behind'}|{opp}"
    return key, {"spec": spec, "start": start_name, "gap": gap, "focal": "ahead" if focal_seat == 0 else "behind",
                 "opponent": opp, "power": feats["power"], "level": feats["level"], "elevel": feats["elevel"],
                 "q": {c: res.mean(c) for c in res.candidates},
                 "adv": {c: res.diff(c, "policy") for c in res.candidates if c != "policy"},
                 "decomposition": {c: _decomp(res, c) for c in res.candidates}, "synthetic": True}


def freeze(specs: Sequence[str] = HORIZON_POLICIES, workers: int = 9, n_roll: int = 8, force: bool = False) -> dict:
    t0 = time.perf_counter()
    wc = vload("wave_cf")
    if wc is None:
        raise SystemExit("freeze: run wave_cf first")
    prev = (vload("freeze") or {}) if not force else {}
    # (a) classification of the sampled freeze-like states (self-play and against each opponent)
    classes = dict(prev.get("classes", {}))
    jobs = []
    for spec in specs:
        for opp in OPPONENTS:
            key = f"{spec}|{opp}|freeze_like"
            if key in wc["results"] and key not in classes:
                jobs.append((key, (spec, wc["results"][key]["states"], n_roll, VAL_FREEZE_SEED)))
    print(f"freeze: {len(jobs)} classification jobs", flush=True)
    res = run_jobs(_freeze_job, [j[1] for j in jobs], workers, label="freeze-classes")
    for (key, _), (_, out) in zip(jobs, res):
        classes[key] = out
    # advantage of the freeze plans by class, pooled over the policies' states per opponent
    by_class = {}
    for key, rows in classes.items():
        opp = key.split("|")[1]
        for r in rows:
            for c in ("adv_m_freeze", "adv_c_freeze", "adv_f_freeze", "adv_m_hold"):
                if r[c] is not None:
                    by_class.setdefault(opp, {}).setdefault(r["class"], {}).setdefault(c, []).append(r[c][0])
    class_summary = {}
    for opp, d in by_class.items():
        class_summary[opp] = {}
        for cls, cd in d.items():
            class_summary[opp][cls] = {"n": len(next(iter(cd.values())))}
            for c, vals in cd.items():
                m, h = mean_ci(vals)
                class_summary[opp][cls][c] = {"mean": m, "ci": h, "n": len(vals)}
    # (b) the synthetic starts with assigned gaps (labelled synthetic)
    synthetic = dict(prev.get("synthetic", {}))
    sjobs = []
    for spec in specs:
        for start_name in SYN_STARTS:
            for gap in SYN_GAPS:
                for seat in (0, 1):
                    if gap == "equal" and seat == 1:
                        continue                                   # equal power: ahead and behind coincide
                    for opp in OPPONENTS:
                        k = f"{spec}|{start_name}|{gap}|{'ahead' if seat == 0 else 'behind'}|{opp}"
                        if k not in synthetic:
                            sjobs.append((spec, start_name, gap, seat, opp, 2 * n_roll, VAL_SYN_SEED))
    print(f"freeze: {len(sjobs)} synthetic jobs", flush=True)
    sres = run_jobs(_synthetic_job, sjobs, workers, label="freeze-synthetic")
    synthetic.update(dict(sres))
    syn_summary = {}
    for start_name in SYN_STARTS:
        for gap in SYN_GAPS:
            for focal in ("ahead", "behind"):
                for opp in OPPONENTS:
                    keys = [f"{sp}|{start_name}|{gap}|{focal}|{opp}" for sp in specs]
                    rows = [synthetic[k] for k in keys if k in synthetic]
                    if not rows:
                        continue
                    cell = {"n_policies": len(rows), "power": rows[0]["power"]}
                    for c in SYN_CANDS:
                        vals = [r["adv"][c][0] for r in rows]
                        m, h = mean_ci(vals)
                        cell[c] = {"mean": m, "ci_seeds": h, "per_seed": vals,
                                   "share_sig_pos": float(np.mean([r["adv"][c][0] > 1.96 * r["adv"][c][1] for r in rows])),
                                   "share_sig_neg": float(np.mean([r["adv"][c][0] < -1.96 * r["adv"][c][1] for r in rows]))}
                    syn_summary[f"{start_name}|{gap}|{focal}|{opp}"] = cell
    data = {"meta": meta(stage="validation.freeze", watch=FREEZE_WATCH, held=FREEZE_HELD, rollouts=n_roll,
                         window=CF_WINDOW, seed=VAL_FREEZE_SEED, synthetic_seed=VAL_SYN_SEED),
            "label_synthetic": "the 'synthetic' block uses CONSTRUCTED starts (zoning_natural.starts + with_gap), "
                               "not states sampled from play; a sensitivity test, never pooled with the sampled states",
            "classes": classes, "class_summary": class_summary, "synthetic": synthetic, "synthetic_summary": syn_summary,
            "elapsed_s": time.perf_counter() - t0}
    vsave("freeze", data)
    return data


__all__ = ["wave_cf", "games", "horizon", "freeze", "candidates", "paired_values", "held_turns", "summarize_states",
           "across_seeds", "VAL_DIR", "POLICIES", "OPPONENTS", "SITUATIONS", "PLAN_TURNS", "GAME_OPPONENTS"]
