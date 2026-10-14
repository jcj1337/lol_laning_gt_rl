"""M4.4b supplementary stages: analyses added during M4.4b on top of the pre-registered `eval_main`
(results/m44a/m44b_plan.md). They reuse the M4.4a instruments; nothing here trains, tunes or changes the game.

    repro_main   re-train seed 0 of every environment at the full budget into <run>_repeat and compare the weights
                 digests and histories with the recorded runs (train.reproduce; merged into train_main.json)
    extra_main   (1) snapshot tournament: every pair of 9 points of a run (initial network, 7 snapshots, final) played
                 against each other: is improvement persistent, or does a later policy lose to an earlier one
                 (self-play cycling)?
                 (2) behaviour of the finals against three more scripted opponents (thin4 = a freeze keeper,
                 freeze_hunt4 = a freezer that hunts, respect4 = a conceding farmer): the response to being frozen on
                 and to a passive opponent
                 (3) behaviour of intermediate snapshots in self-play (has behaviour stopped changing?)
                 (4) integrity of the finals over self-play: finite logits and values, |V| within the return bound,
                 zero probability on illegal codes, antisymmetric rewards
                 (5) TRANSFER TEST (cross-environment, explicit opt-in of the compatibility guard): each final played in
                 the two other environments against the native final of the same seed and against shove4 / farm4. The
                 network then sees an observation normalized for its own game (absence scale 3 vs 5) in a game with
                 another economy: a robustness probe, never a like-for-like comparison
    traces_main  representative games: self-play games of the seed-0 finals (and the initialized x1.76 power gap) are
                 recorded with the viewer's seed convention, searched for state-defined situations (a freeze that
                 denies, one that does not, a frozen-on side breaking it, a slow push that lands, Vision before
                 Pressure, a busy dive, the weak side under a credible threat), and the chosen games are exported for
                 the Streamlit debugger with a turn-by-turn excerpt of the situation and verified by replay

Results: results/m44b/{extra_main.json(.gz), traces_main.json, trajectories/}. Evaluation seed roots (80,000-80,999,
disjoint from training): tournament 80,602, transfer 80,603, snapshot behaviour 80,002, extra references 80,001 (the
pre-registered reference behaviour root), integrity 80,750, trace search 80,510-80,569 (integer viewer seeds).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .common import (BEHAVIOR_SEED, ENV_ORDER, ENVS, OUT_B, SEEDS, SMOKE_EVAL_SEED, TRACE_SEED, XPLAY_SEED, env_cfg,
                     final_path, load_json, meta, run_jobs, save_json, snapshot_paths)

TOURNAMENT_SEED = XPLAY_SEED + 2
TRANSFER_SEED = XPLAY_SEED + 3
SNAP_BEHAVIOR_SEED = BEHAVIOR_SEED + 2
INTEGRITY_SEED = SMOKE_EVAL_SEED + 50
TRACE_SEARCH_SEED = TRACE_SEED + 10

EXTRA = dict(tour_points=(256_000, 512_000, 768_000, 1_024_000, 1_280_000, 1_536_000, 1_792_000), tour_eps=16,
             tour_window=460, extra_refs=("thin4", "freeze_hunt4", "respect4"), extra_ref_games=12, extra_ref_turns=1000,
             snap_points=(256_000, 512_000, 1_024_000, 1_536_000), snap_games=16, snap_turns=500, integrity_games=4,
             integrity_turns=500, transfer_opps=("native", "shove4", "farm4"), transfer_eps=16, transfer_window=460)
TRACES = dict(selfplay_games=24, selfplay_turns=460, gap_games=12, gap_turns=300, env_seed=0)


def nearest_snapshots(phase: str, env: str, seed: int, points: Sequence[int]) -> List[int]:
    """The snapshot transition counts nearest to `points` (snapshots fall on update boundaries)."""
    snaps = [n for n, _ in snapshot_paths(phase, env, seed)]
    out = []
    for p in points:
        n = min(snaps, key=lambda x: abs(x - p))
        if n not in out:
            out.append(n)
    return out


# ---------------------------------------------------------------------------------------------- jobs
def integrity_job(args) -> tuple:
    """Self-play of one final: every observation's logits and value finite, |V| <= Omega (resources), exactly zero
    probability on masked codes, every sampled action legal (the environment raises otherwise), rewards
    antisymmetric on every turn."""
    env, spec, games, turns, seed = args
    import torch
    torch.set_num_threads(1)
    from ....env import LaneEnv
    from ....env.agents import episode_streams
    from ....env.discounted import tail_constant
    from .common import agent_from_spec
    cfg = env_cfg(env)
    x = agent_from_spec(spec, cfg)
    scale = float((getattr(x, "meta", None) or {}).get("ppo", {}).get("reward_scale", 0.1))
    omega = tail_constant(cfg)
    e = LaneEnv(cfg, max_turns=turns, cache_size=20_000)
    n_obs = bad_logit = bad_value = over = illegal_mass = asym = 0
    vmax, max_illegal_p = 0.0, 0.0
    for g in range(games):
        _, env_ss, ra, rb = episode_streams(np.random.SeedSequence(seed, spawn_key=(g,)))
        obs = e.reset(seed=env_ss)
        while True:
            for o in obs:
                with torch.no_grad():
                    logits, v = x.net(torch.as_tensor(o.vector, dtype=torch.float32).unsqueeze(0))
                n_obs += 1
                lg = logits[0].numpy()
                legal = np.asarray(o.mask, bool)
                bad_logit += int(not np.all(np.isfinite(lg[legal])))
                val = float(v[0]) / scale
                bad_value += int(not np.isfinite(val))
                vmax = max(vmax, abs(val))
                over += int(abs(val) > omega)
                p = x.probabilities(o)
                m = float(p[~legal].sum())
                max_illegal_p = max(max_illegal_p, m)
                illegal_mass += int(m > 0)
            r = e.step(x.act(obs[0], ra), x.act(obs[1], rb))
            asym += int(r.rewards[0] != -r.rewards[1])
            if r.done:
                break
            obs = r.obs
    return (env, spec), {"observations": n_obs, "nonfinite_legal_logits": bad_logit, "nonfinite_values": bad_value,
                         "value_absmax_resources": vmax, "omega": omega, "values_above_omega": over,
                         "observations_with_illegal_mass": illegal_mass, "max_illegal_mass": max_illegal_p,
                         "reward_asymmetries": asym, "games": games, "turns": turns}


# ---------------------------------------------------------------------------------------------- extra_main
def extra(envs: Sequence[str] = ENV_ORDER, seeds: Sequence[int] = SEEDS, workers: int = 9, phase: str = "main") -> dict:
    from . import evaluate as EV
    from .stages import behavior_job
    z = EXTRA
    finals = [(e, s) for e in envs for s in seeds if os.path.exists(final_path(phase, e, s))]
    t0 = time.perf_counter()
    data = {"meta": meta(stage=f"extra_{phase}", study="M4.4b", sizes={k: (list(v) if isinstance(v, tuple) else v)
                                                                         for k, v in z.items()},
                         label="M4.4b supplementary analyses (beyond the pre-registered eval_main)"),
            "finals": [f"final:{phase}:{e}:{s}" for e, s in finals]}
    # (1) tournament and (5) transfer: evaluate_discounted matchups
    jobs, tour_players = [], {}
    for e, s in finals:
        pts = nearest_snapshots(phase, e, s, z["tour_points"])
        players = [f"initial:{e}:{s}"] + [f"snap:{phase}:{e}:{s}:{n}" for n in pts] + [f"final:{phase}:{e}:{s}"]
        tour_players[f"{e}_s{s}"] = players
        for i, a in enumerate(players):
            for b in players[i + 1:]:
                jobs.append((e, a, b, z["tour_eps"], z["tour_window"], TOURNAMENT_SEED, False))
    tjobs = []
    for target in envs:
        for e, s in finals:
            if e == target:
                continue
            spec = f"final:{phase}:{e}:{s}"
            for opp in z["transfer_opps"]:
                ys = f"final:{phase}:{target}:{s}" if opp == "native" else f"ref:{opp}"
                tjobs.append((target, spec, ys, z["transfer_eps"], z["transfer_window"], TRANSFER_SEED, True))
            # the native final against the same references on the same seeds: the like-for-like baseline
        for e, s in finals:
            if e == target:
                for opp in z["transfer_opps"][1:]:
                    tjobs.append((target, f"final:{phase}:{e}:{s}", f"ref:{opp}", z["transfer_eps"], z["transfer_window"],
                                  TRANSFER_SEED, False))
    res = EV.run(jobs + tjobs, workers)
    data["tournament"] = {"players": tour_players,
                          "results": {k: res[k] for k in {f"{e}|{a}|{b}" for (e, a, b, *_r) in jobs}}}
    data["transfer"] = {"label": "TRANSFER TEST (cross-environment checkpoint evaluation with the explicit opt-in of "
                                 "load_policy(cross_env=True)); not a like-for-like comparison",
                        "results": {k: res[k] for k in {f"{e}|{a}|{b}" for (e, a, b, *_r) in tjobs}}}
    # (2) behaviour against the extra references and (3) intermediate snapshots in self-play
    bj = []
    for e, s in finals:
        spec = f"final:{phase}:{e}:{s}"
        for r in z["extra_refs"]:
            bj.append((e, spec, r, z["extra_ref_games"], z["extra_ref_turns"], BEHAVIOR_SEED + 1, "s0", "equal"))
        for n in nearest_snapshots(phase, e, s, z["snap_points"]):
            bj.append((e, f"snap:{phase}:{e}:{s}:{n}", "self", z["snap_games"], z["snap_turns"], SNAP_BEHAVIOR_SEED,
                       "s0", "equal"))
        bj.append((e, spec, "self", z["snap_games"], z["snap_turns"], SNAP_BEHAVIOR_SEED, "s0", "equal"))
    data["behavior"] = {"|".join(k): v for k, v in run_jobs(behavior_job, bj, workers, "extra behavior")}
    # (4) integrity
    ij = [(e, f"final:{phase}:{e}:{s}", z["integrity_games"], z["integrity_turns"], INTEGRITY_SEED) for e, s in finals]
    data["integrity"] = {f"{k[0]}|{k[1]}": v for k, v in run_jobs(integrity_job, ij, workers, "integrity")}
    data["elapsed_s"] = time.perf_counter() - t0
    save_json(f"extra_{phase}", data, OUT_B)
    return data


# ---------------------------------------------------------------------------------------------- traces_main
def _episodes(rec, s: int) -> Dict[str, List[dict]]:
    """Candidate situations of side s in one record (state-defined; the score orders candidates of one kind)."""
    from .metrics import SH, SL, credible_threat, freeze_episodes
    d = rec.data
    o = 1 - s
    T = rec.turns
    out: Dict[str, List[dict]] = {k: [] for k in ("freeze_denies", "freeze_fails", "freeze_broken", "slow_push_lands",
                                                  "vision_then_pressure", "busy_dive", "weak_side_threatened")}
    for ep in freeze_episodes(rec, s):
        if not ep["established"] or ep["end"] == "censored":
            continue
        t0, t1 = ep["start"], ep["start"] + ep["length"] - 1
        base = {"side": s, "t0": t0, "t1": t1, "episode": ep}
        if ep["gold_diff"] > 0 and ep["frozen_on_xp_missed"] > 0:
            out["freeze_denies"].append(base | {"score": ep["length"] + ep["frozen_on_xp_missed"] / 10})
        if ep["gold_diff"] <= 0:
            out["freeze_fails"].append(base | {"score": ep["length"] - ep["gold_diff"]})
        if ep["frozen_on_shoved_at_end"] and ep["end"] in ("crashed_in", "pushed_out"):
            out["freeze_broken"].append(base | {"score": ep["length"]})
    w = d["wave"][:, s]
    for t in range(T - 12):
        if (w[t:t + 3] == SL).all() and (t == 0 or w[t - 1] != SL):
            crash = np.flatnonzero(d["crash"][t + 3:t + 12, s] >= 12)
            if len(crash):
                t1 = int(t + 3 + crash[0])
                out["slow_push_lands"].append({"side": s, "t0": t, "t1": t1,
                                               "score": float(d["crash"][t1, s]) + float(d["wave_adv"][t:t1 + 1, s].max())})
    for t in np.flatnonzero(d["vision_set"][:, s] > 0):
        for u in range(t + 1, min(T - 3, t + 3)):
            if (d["obj"][u:u + 3, s] == 1).all():
                out["vision_then_pressure"].append({"side": s, "t0": int(t), "t1": int(u + 2),
                                                    "score": float(d["plates"][u:u + 6, s].sum())})
                break
    for t in np.flatnonzero((d["dive"][:, s] > 0) & (d["busy"][:, s] > 0)):
        out["busy_dive"].append({"side": s, "t0": int(t), "t1": int(min(T - 1, t + 2)),
                                 "score": float(d["tower_dmg"][t, s] > 0) + 2 * float(d["died"][t:t + 3, s].sum() > 0)
                                 + float(d["died"][t:t + 3, o].sum() > 0)})
    th = credible_threat(d, s)
    runs = []
    t0 = None
    for t, f in enumerate(th):
        if f and t0 is None:
            t0 = t
        elif not f and t0 is not None:
            runs.append((t0, t - 1))
            t0 = None
    if t0 is not None:
        runs.append((t0, T - 1))
    for a, b in runs:
        if b - a + 1 >= 2:
            out["weak_side_threatened"].append({"side": s, "t0": a, "t1": b, "score": float(b - a + 1)})
    return out


EXCERPT_FIELDS = ("pos", "hp_frac", "level", "invested", "power", "own_fx", "own_fh", "en_fx", "en_fh", "wave_adv",
                  "held", "held_on", "own_siege", "en_siege", "vision", "plates_left", "earned", "xp_missed", "died",
                  "recalled", "crash", "en_crash", "tower_p", "tower_dmg", "trade_conn", "trade_dealt", "damage_taken",
                  "dive", "busy", "gank_p", "ganked", "pw0", "pw1", "pw2", "pw3", "pb3", "po1", "po2")


def excerpt(rec, t0: int, t1: int, pad: tuple = (3, 6)) -> List[dict]:
    """Turn-by-turn rows of the window [t0 - pad0, t1 + pad1]: both sides' state, action label and outcome."""
    from .actions import decode
    d = rec.data
    rows = []
    for t in range(max(0, t0 - pad[0]), min(rec.turns, t1 + pad[1] + 1)):
        row = {"turn": t}
        for s, n in ((0, "A"), (1, "B")):
            row[n] = {"action": decode(int(d["a"][t, s])).label(),
                      **{f: (None if not np.isfinite(d[f][t, s]) else round(float(d[f][t, s]), 3)) for f in EXCERPT_FIELDS}}
        rows.append(row)
    return rows


def export_record(rec, actions: List[List[int]], name: str, env: str, specs: Sequence[str], seed: int, scenario: str,
                  gap: str, label: str, out_dir: str) -> str:
    """Write a viewer recording (schema of trace.export) of a game recorded with the viewer seed convention."""
    from ....io import dump_json
    from .actions import decode
    from .trace import SCHEMA
    d = rec.data
    events = []
    for t in range(rec.turns):
        for s, n in ((0, "A"), (1, "B")):
            if d["died"][t, s]:
                events.append([t, f"{n} dies"])
            if d["recalled"][t, s]:
                events.append([t, f"{n} recalls (HP {d['hp_frac'][t, s]:.0%})"])
            if d["crash"][t, s]:
                events.append([t, f"{n} crashes {int(d['crash'][t, s])} HP into the enemy tower"])
            if d["plates"][t, s]:
                events.append([t, f"{n} breaks a plate"])
            if d["vision_set"][t, s]:
                events.append([t, f"{n} establishes vision"])
            if d["dive"][t, s]:
                events.append([t, f"{n} dives ({'busy' if d['busy'][t, s] else 'not busy'} tower, shot p "
                                  f"{d['tower_p'][t, s]:.2f})"])
            if d["held"][t, s] and (t == 0 or not d["held"][t - 1, s]):
                events.append([t, f"freeze state: {n} holds the enemy wave at its tower"])
    rec_json = {"schema": SCHEMA, "version": 1, "config": ENVS[env].preset, "scenario": scenario, "gap": gap,
                "seed": int(seed), "names": list(specs), "label": label, "actions": actions, "events": events,
                "payoff_A": float(d["reward"][:, 0].sum()), "turns": rec.turns,
                "labels": [[decode(a).label(), decode(b).label()] for a, b in actions]}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    dump_json(rec_json, path)
    return path


def trace_search_job(args) -> tuple:
    """Record the games of one (env, spec) with the viewer convention and return the candidate situations."""
    env, spec, kind, games, turns, seed0 = args
    import torch
    torch.set_num_threads(1)
    from .common import agent_from_spec
    from .records import play_record
    cfg = env_cfg(env)
    x = agent_from_spec(spec, cfg)
    start, scen, gap = None, "s0", "equal"
    if kind == "gap":
        from ...m43c.harness import start as scenario_start
        start, scen, gap = scenario_start("neutral", cfg, "large"), "neutral", "large"
    cands = []
    for g in range(games):
        seed = seed0 + g
        rec = play_record(cfg, (x, x), seed, turns, start, f"{scen}/{gap}" if start else "s0", env, viewer_seed=True)
        for s in (0, 1):
            if kind == "gap" and s == 0:
                continue                                  # the power gap favours A: B is the weak side
            for k, lst in _episodes(rec, s).items():
                if kind == "gap" and k != "weak_side_threatened":
                    continue
                for c in lst:
                    cands.append({"kind": k, "seed": seed, "scenario": scen, "gap": gap, **c,
                                  "episode": {kk: vv for kk, vv in c.get("episode", {}).items()}})
    return (env, spec, kind), cands


def traces(envs: Sequence[str] = ENV_ORDER, workers: int = 9, phase: str = "main") -> dict:
    """Search, select one game per (situation, environment), export and verify (module docstring)."""
    import torch
    from .actions import decode
    from .common import agent_from_spec
    from .records import play_record
    from .trace import verify
    z = TRACES
    t0 = time.perf_counter()
    jobs = []
    for i, e in enumerate(envs):
        spec = f"final:{phase}:{e}:{z['env_seed']}"
        jobs.append((e, spec, "s0", z["selfplay_games"], z["selfplay_turns"], TRACE_SEARCH_SEED))
        jobs.append((e, spec, "gap", z["gap_games"], z["gap_turns"], TRACE_SEARCH_SEED + 40))
    found = dict(run_jobs(trace_search_job, jobs, workers, "trace search"))
    out_dir = os.path.join(OUT_B, "trajectories")
    selected = {}
    for e in envs:
        spec = f"final:{phase}:{e}:{z['env_seed']}"
        cands = found.get((e, spec, "s0"), []) + found.get((e, spec, "gap"), [])
        counts = {}
        for c in cands:
            counts[c["kind"]] = counts.get(c["kind"], 0) + 1
        for kind in ("freeze_denies", "freeze_fails", "freeze_broken", "slow_push_lands", "vision_then_pressure",
                     "busy_dive", "weak_side_threatened"):
            cs = sorted([c for c in cands if c["kind"] == kind], key=lambda c: (-c["score"], c["seed"], c["t0"]))
            if not cs:
                selected[f"{e}|{kind}"] = {"found": 0}
                continue
            c = cs[0]
            cfg = env_cfg(e)
            torch.set_num_threads(1)
            x = agent_from_spec(spec, cfg)
            start = None
            if c["scenario"] != "s0":
                from ...m43c.harness import start as scenario_start
                start = scenario_start(c["scenario"], cfg, c["gap"])
            turns = z["gap_turns"] if start is not None else z["selfplay_turns"]
            rec = play_record(cfg, (x, x), c["seed"], turns, start, "s0" if start is None else f"{c['scenario']}/{c['gap']}",
                              e, viewer_seed=True)
            actions = [[int(rec.data["a"][t, 0]), int(rec.data["a"][t, 1])] for t in range(rec.turns)]
            name = f"m44b_{e}_s{z['env_seed']}_{kind}"
            side = "AB"[c["side"]]
            label = (f"M4.4b {e} final seed {z['env_seed']} self-play; {kind.replace('_', ' ')} by {side} at turns "
                     f"{c['t0']}-{c['t1']}")
            path = export_record(rec, actions, name, e, (spec, spec), c["seed"], c["scenario"], c["gap"], label, out_dir)
            selected[f"{e}|{kind}"] = {"found": counts.get(kind, 0), "path": path, "seed": c["seed"], "side": side,
                                       "t0": c["t0"], "t1": c["t1"], "scenario": c["scenario"], "gap": c["gap"],
                                       "episode": c.get("episode") or None, "label": label,
                                       "excerpt": excerpt(rec, c["t0"], c["t1"]), "verify": verify(path)}
    data = {"meta": meta(stage=f"traces_{phase}", study="M4.4b", sizes=dict(z),
                         label="representative games of the seed-0 finals (state-defined search; viewer recordings)"),
            "candidates_found": {f"{k[0]}|{k[2]}": len(v) for k, v in found.items()}, "selected": selected,
            "elapsed_s": time.perf_counter() - t0}
    save_json(f"traces_{phase}", data, OUT_B)
    return data


# ---------------------------------------------------------------------------------------------- recall_main
def recall_attempts(records, focal: Optional[str] = None) -> dict:
    """Chosen Recalls and their interruption (found in the M4.4b traces: a damaged champion choosing Recall turn after
    turn while an enemy Trade in contact interrupts it). A chosen Recall is a present turn with behaviour Recall; it is
    INTERRUPTED when the champion did not leave (`recalled` false; resolver step 2b: an enemy Trade in contact). Per
    attempt: HP, whether the enemy chose Trade, and for interrupted ones the champion's death within 3 turns, the run
    length of consecutive interrupted attempts, and the policy's mass on Safe at that state (Safe lowers an enemy
    trade's hit probability by 0.4; Recall does not change it)."""
    from ....config import B_RECALL, B_TRADE
    from .metrics import _runs, _units
    att = {"n": 0, "interrupted": 0, "enemy_traded": 0, "hp": [], "hp_interrupted": [], "died3": [], "safe_mass": [],
           "recall_mass": [], "runs": []}
    turns = 0
    for rec, s in _units(records, focal):
        d = rec.data
        o = 1 - s
        turns += rec.turns
        chose = (d["present"][:, s] > 0) & (d["beh"][:, s] == B_RECALL)
        intr = chose & (d["recalled"][:, s] == 0)
        att["n"] += int(chose.sum())
        att["interrupted"] += int(intr.sum())
        att["enemy_traded"] += int((chose & (d["beh"][:, o] == B_TRADE)).sum())
        att["hp"] += d["hp_frac"][chose, s].tolist()
        for t in np.flatnonzero(intr):
            att["hp_interrupted"].append(float(d["hp_frac"][t, s]))
            att["died3"].append(float(d["died"][t:t + 4, s].sum() > 0))
            att["safe_mass"].append(float(d["pb3"][t, s]))
            att["recall_mass"].append(float(d["prec"][t, s]))
        # consecutive interrupted attempts (Recall chosen on consecutive present turns, each interrupted)
        att["runs"] += [b - a + 1 for a, b in _runs(intr)]

    def mean(v):
        return float(np.nanmean(v)) if len(v) else float("nan")
    r = np.asarray(att["runs"], int)
    return {"turns": turns, "attempts": att["n"], "attempts_per_1000_turns": 1000.0 * att["n"] / max(turns, 1),
            "interrupted": att["interrupted"], "interrupted_share": att["interrupted"] / max(att["n"], 1),
            "interrupted_per_1000_turns": 1000.0 * att["interrupted"] / max(turns, 1),
            "enemy_traded_share": att["enemy_traded"] / max(att["n"], 1), "hp_at_attempt": mean(att["hp"]),
            "hp_at_interrupted": mean(att["hp_interrupted"]), "died_within_3_after_interrupted": mean(att["died3"]),
            "policy_recall_mass_when_interrupted": mean(att["recall_mass"]),
            "policy_safe_mass_when_interrupted": mean(att["safe_mass"]),
            "interrupted_runs": {"n": int(len(r)), "mean_length": float(r.mean()) if len(r) else float("nan"),
                                 "runs_of_3_or_more": int((r >= 3).sum()), "max": int(r.max()) if len(r) else 0}}


def recall_job(args) -> tuple:
    env, spec, games, turns, seed = args
    import torch
    torch.set_num_threads(1)
    from .common import agent_from_spec
    from .records import play_records
    cfg = env_cfg(env)
    x = agent_from_spec(spec, cfg)
    recs = play_records(cfg, x, x, games, turns, seed, env_label=env)
    return (env, spec), recall_attempts(recs, focal=x.name)


def recall_followup(envs: Sequence[str] = ENV_ORDER, seeds: Sequence[int] = SEEDS, workers: int = 9,
                    phase: str = "main", games: int = 8, turns: int = 1000) -> dict:
    """Interrupted-Recall measure on natural self-play of every final: the first `games` of the pre-registered
    behaviour games (root 80,000, the same seeds and therefore the same games)."""
    t0 = time.perf_counter()
    jobs = [(e, f"final:{phase}:{e}:{s}", games, turns, BEHAVIOR_SEED) for e in envs for s in seeds
            if os.path.exists(final_path(phase, e, s))]
    res = {f"{k[0]}|{k[1]}": v for k, v in run_jobs(recall_job, jobs, workers, "recall follow-up")}
    data = {"meta": meta(stage=f"recall_{phase}", study="M4.4b", games=games, turns=turns, seed_root=BEHAVIOR_SEED,
                         label="M4.4b follow-up: chosen Recalls interrupted by an enemy Trade (natural self-play)"),
            "results": res, "elapsed_s": time.perf_counter() - t0}
    save_json(f"recall_{phase}", data, OUT_B)
    return data


# ---------------------------------------------------------------------------------------------- eval_ext
EXT_EVAL = dict(ref_eps=32, ref_window=1000, vs_main_eps=32, vs_main_window=1000,
                curve_points=(1_024_000, 2_048_000, 3_072_000, 4_096_000), curve_eps=16, curve_window=460,
                beh_games=32, beh_turns=1000, snap_games=16, snap_turns=500)


def eval_ext(envs: Sequence[str] = ENV_ORDER, seeds: Sequence[int] = SEEDS, workers: int = 9) -> dict:
    """The pre-defined reduced evaluation of the matched 2x-budget extension (phase "ext", 4,096,000 transitions):
    (1) finals against the ten references (32 paired seeds, W = 1000; the main protocol and seed root), (2) the
    extension final against the main final of the same environment and seed (32 paired seeds, W = 1000), (3) a short
    learning curve: the snapshots nearest 1.02 / 2.05 / 3.07 / 4.10M against the five core references (16 paired seeds,
    W = 460; the main curve protocol) and the final against its own 2.05M snapshot, (4) natural self-play behaviour of
    the finals (32 games x 1000 turns, the main root: the same seeds), (5) behaviour of the curve snapshots at the
    snapshot-behaviour size of extra_main (16 games x 500 turns, root 80,002). No counterfactuals, tournament or traces."""
    from . import evaluate as EV
    from .common import REF_SEED
    from .references import CORE, REFERENCES
    from .stages import behavior_job
    z = EXT_EVAL
    finals = [(e, s) for e in envs for s in seeds if os.path.exists(final_path("ext", e, s))]
    t0 = time.perf_counter()
    jobs, snaps = [], {}
    for e, s in finals:
        spec = f"final:ext:{e}:{s}"
        jobs += [(e, spec, f"ref:{r}", z["ref_eps"], z["ref_window"], REF_SEED, False) for r in REFERENCES]
        jobs.append((e, spec, f"final:main:{e}:{s}", z["vs_main_eps"], z["vs_main_window"], XPLAY_SEED, False))
        pts = nearest_snapshots("ext", e, s, z["curve_points"])
        snaps[f"{e}_s{s}"] = pts
        jobs += [(e, f"snap:ext:{e}:{s}:{n}", f"ref:{r}", z["curve_eps"], z["curve_window"], REF_SEED + 1, False)
                 for n in pts for r in CORE]
        mid = nearest_snapshots("ext", e, s, (2_048_000,))[0]
        jobs.append((e, spec, f"snap:ext:{e}:{s}:{mid}", z["curve_eps"], z["curve_window"], XPLAY_SEED + 1, False))
    strength = EV.run(jobs, workers)
    bj = [(e, f"final:ext:{e}:{s}", "self", z["beh_games"], z["beh_turns"], BEHAVIOR_SEED, "s0", "equal")
          for e, s in finals]
    beh = {"|".join(k): v for k, v in run_jobs(behavior_job, bj, workers, "ext behavior")}
    sj = [(e, f"snap:ext:{e}:{s}:{n}", "self", z["snap_games"], z["snap_turns"], SNAP_BEHAVIOR_SEED, "s0", "equal")
          for e, s in finals for n in snaps[f"{e}_s{s}"]]
    sbeh = {"|".join(k): v for k, v in run_jobs(behavior_job, sj, workers, "ext snapshot behavior")}
    data = {"meta": meta(stage="eval_ext", study="M4.4b", sizes={k: (list(v) if isinstance(v, tuple) else v)
                                                                   for k, v in z.items()},
                         label="M4.4b matched 2x-budget extension: pre-defined reduced evaluation"),
            "finals": [f"final:ext:{e}:{s}" for e, s in finals], "curve_snapshots": snaps, "strength": strength,
            "behavior": beh, "snapshot_behavior": sbeh, "elapsed_s": time.perf_counter() - t0}
    save_json("eval_ext", data, OUT_B)
    return data


__all__ = ["EXTRA", "TRACES", "EXT_EVAL", "extra", "traces", "integrity_job", "trace_search_job", "nearest_snapshots",
           "excerpt", "export_record", "recall_attempts", "recall_job", "recall_followup", "eval_ext"]
