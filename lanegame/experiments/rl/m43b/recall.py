"""Recall and death economics of the learned policies (M4.3b §5 of the brief).

The mechanics (GameConfig.m4_lasthit): a champion at or behind the recall line (own -1) with no enemy trade reaching
it may Recall: it leaves the lane at once, misses `recall_turns` = 2 turns in total (the Recall turn included),
invests all unspent gold at base and returns at its tower with full HP. A champion that dies sits out `death_turns`
= 3 further turns, pays the opponent the kill bounty (0.5 resources), invests at base exactly as after a Recall and
returns with full HP. A death therefore buys the same reset (HP, investment) as a Recall for one extra turn of
absence and half a resource of bounty, from ANY position and without the walk back to the recall line, and it
cannot be interrupted. Whether that makes death a cheap substitute for Recall is what this module measures on the
learned policies, with three instruments:

  1. frequencies (from `behavior.profile`): how often Recall is legal, the policy's probability on it when legal, how
     often it is chosen; returns after death against after Recall, and the gold carried to base by each route;
  2. counterfactual Recalls: at sampled reachable states where Recall is legal (classified: low HP, after the own
     crash, a neutral lane at full HP, the enemy wave pushing in, >= 4 resources unspent, the enemy away), paired
     rollouts of Recall against the policy's own move, against staying (Hold + Hold) and against walking back (Safe),
     the policy as continuation and opponent, CF_WINDOW turns discounted;
  3. the value of the absence itself: from sampled present states the focal champion's record is replaced by "just
     died" (at its tower, HP 0, absent 3 turns; the lane unchanged) or "just recalled" (at its tower, absent 1 more
     turn), and the three variants are rolled out on common random numbers. V(dead) - V(alive) is the downstream
     cost of a death from that state, EXCLUDING the death turn itself and the 0.5 bounty (both precede the state);
     V(recalled) - V(alive) the downstream cost of a completed Recall, EXCLUDING the Recall turn's forgone farm. The
     full costs are reported with those two terms added back (bounty 0.5; the Recall turn at the state's expected
     farm, approximated by the mean present-turn farm of the policy).

Every value is a Monte Carlo estimate of the given continuation policies on this opponent over a window, not a
game-theoretic value; the paired differences are what carry the information.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np

from ....config import B_HOLD, B_SAFE, W_HOLD, champ_act
from .behavior import (CF_WINDOW, RECALL, full_distribution, play_rows, profile, sample_states, situations)
from .common import (CONDITIONS, PRIMARY, PROFILE_SEED, RECALL_ROLL_SEED, RECALL_STATE_SEED, SEEDS, WINDOW_DEV,
                     agent_from_spec, cfg_game, load, meta, run_jobs, save_json)

RECALL_CLASSES = {
    "low_hp": "Recall legal, HP <= 40%",
    "after_own_crash": "Recall legal, own crashed mass under the enemy tower",
    "neutral_full_hp": "Recall legal, fronts fighting at the lane cell, HP >= 80%",
    "enemy_pushing": "Recall legal, enemy front in the own half",
    "cash_ready": "Recall legal, >= 4 resources unspent",
    "enemy_absent": "Recall legal, enemy champion absent",
    "any": "Recall legal (any state)",
}
R_STATES, R_ROLLOUTS, DV_STATES = 8, 24, 24


def recall_masks() -> dict:
    def f(cls):
        def mask(R, S):
            m = R["recall_legal"]
            if cls == "low_hp":
                return m & (R["hp_frac"] <= 0.4)
            if cls == "after_own_crash":
                return m & (R["own_siege"] > 0)
            if cls == "neutral_full_hp":
                return m & R["contact"] & (R["contact_cell"] == 0) & (R["hp_frac"] >= 0.8)
            if cls == "enemy_pushing":
                return m & (R["en_fh"] > 0) & (R["en_fx"] <= -1)
            if cls == "cash_ready":
                return m & (R["gold"] >= 4)
            if cls == "enemy_absent":
                return m & R["enemy_absent"]
            return m
        return mask
    return {c: f(c) for c in RECALL_CLASSES}


# ---------------------------------------------------------------------------------------------- counterfactual Recall
def _recall_state(cfg, agent, snap, seat, feats, n_roll, roll_seed, key, cache) -> dict:
    from ....diagnostics import Commit, FollowPolicy, branch, observation
    ob = observation(cfg, snap, seat)
    assert RECALL in ob.legal
    cands = {"Recall": Commit(RECALL, "Recall"), "stay": Commit(champ_act(W_HOLD, B_HOLD), "stay"),
             "safe": Commit(champ_act(W_HOLD, B_SAFE), "safe"), "policy": FollowPolicy()}
    res = branch(cfg, snap, seat, cands, (agent, agent), n_roll, roll_seed, key=key, cache=cache, max_turns=CF_WINDOW)
    p = full_distribution(agent, ob)
    dec = {c: {"farm_diff": float((res.farm[c][:, 0] - res.farm[c][:, 1]).mean()),
               "plates_diff": float((res.plates[c][:, 0] - res.plates[c][:, 1]).mean()),
               "bounty_diff": float((res.bounty[c][:, 0] - res.bounty[c][:, 1]).mean()),
               "deaths": res.deaths[c].mean(0).tolist(), "recalls": res.recalls[c].mean(0).tolist(),
               "invested": [float(res.m4[c]["invested"][:, i].mean()) / cfg.gold_scale for i in (0, 1)]}
           for c in res.candidates}
    return {"seat": seat, "k": snap.k, "features": feats, "snapshot": snap.to_dict(), "pi_recall": float(p[RECALL]),
            "q": {c: res.mean(c) for c in res.candidates},
            "recall_vs_policy": res.diff("Recall", "policy"), "recall_vs_stay": res.diff("Recall", "stay"),
            "recall_vs_safe": res.diff("Recall", "safe"), "decomposition": dec}


def _recall_job(args) -> tuple:
    spec, cls, n_states, n_roll, state_seed, roll_seed = args
    import torch
    from ....diagnostics import ProbCache
    torch.set_num_threads(1)
    cfg = cfg_game()
    agent = agent_from_spec(spec, cfg)
    t0 = time.perf_counter()
    states = sample_states(cfg, agent, {cls: recall_masks()[cls]}, n_states, state_seed, max_games=80)[cls]
    cache = ProbCache()
    out = [_recall_state(cfg, agent, snap, seat, feats, n_roll, roll_seed, (i,), cache)
           for i, (snap, seat, feats) in enumerate(states)]
    return (spec, cls), {"spec": spec, "class": cls, "states": out, "n_found": len(states),
                         "elapsed_s": time.perf_counter() - t0}


def aggregate_recall(states: List[dict]) -> dict:
    if not states:
        return {}
    out = {"n_states": len(states), "mean_pi_recall": float(np.mean([s["pi_recall"] for s in states]))}
    for key in ("recall_vs_policy", "recall_vs_stay", "recall_vs_safe"):
        d = np.array([s[key][0] for s in states])
        se = np.array([s[key][1] for s in states])
        out[key] = {"mean": float(d.mean()),
                    "ci": float(1.96 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
                    "share_pos": float(np.mean(d > 1.96 * se)), "share_neg": float(np.mean(d < -1.96 * se))}
    out["mean_hp_frac"] = float(np.mean([s["features"]["hp_frac"] for s in states]))
    out["mean_gold"] = float(np.mean([s["features"]["gold"] for s in states]))
    out["invested_after_recall"] = float(np.mean([s["decomposition"]["Recall"]["invested"][0] for s in states]))
    out["invested_policy"] = float(np.mean([s["decomposition"]["policy"]["invested"][0] for s in states]))
    out["deaths_policy"] = float(np.mean([s["decomposition"]["policy"]["deaths"][0] for s in states]))
    out["deaths_recall"] = float(np.mean([s["decomposition"]["Recall"]["deaths"][0] for s in states]))
    return out


def regression(states: List[dict]) -> dict:
    """OLS of Q(Recall) - Q(stay) on unspent gold, HP fraction and level across all sampled Recall-legal states."""
    if len(states) < 6:
        return {}
    y = np.array([s["recall_vs_stay"][0] for s in states])
    X = np.array([[1.0, s["features"]["gold"], s["features"]["hp_frac"], s["features"]["level"],
                   float(s["features"]["own_siege"] > 0)] for s in states])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    cov = np.linalg.pinv(X.T @ X) * (resid @ resid / dof)
    se = np.sqrt(np.diag(cov))
    names = ("intercept", "gold_unspent", "hp_frac", "level", "after_own_crash")
    return {"n": len(y), "coef": {n: float(b) for n, b in zip(names, beta)}, "se": {n: float(s) for n, s in zip(names, se)},
            "r2": float(1 - resid.var() / y.var()) if y.var() > 0 else float("nan")}


# ---------------------------------------------------------------------------------------------- against an opponent
def _recall_opp_job(args) -> tuple:
    """Counterfactual Recalls at Recall-legal states of games of the policy AGAINST A REFERENCE (the policy's death
    rate differs by opponent, report 6.2); the reference is the opponent of the rollouts too."""
    spec, opp, cls, n_states, n_roll, state_seed, roll_seed = args
    import torch
    from ....diagnostics import Commit, FollowPolicy, ProbCache, branch, observation
    torch.set_num_threads(1)
    cfg = cfg_game()
    agent, y = agent_from_spec(spec, cfg), agent_from_spec(f"ref:{opp}", cfg)
    t0 = time.perf_counter()
    states = sample_states(cfg, agent, {cls: recall_masks()[cls]}, n_states, state_seed, max_games=80, opponent=y)[cls]
    cache = ProbCache()
    out = []
    for i, (snap, seat, feats) in enumerate(states):
        ob = observation(cfg, snap, seat)
        cands = {"Recall": Commit(RECALL, "Recall"), "stay": Commit(champ_act(W_HOLD, B_HOLD), "stay"),
                 "safe": Commit(champ_act(W_HOLD, B_SAFE), "safe"), "policy": FollowPolicy()}
        agents = (agent, y) if seat == 0 else (y, agent)
        res = branch(cfg, snap, seat, cands, agents, n_roll, roll_seed, key=(i,), cache=cache, max_turns=CF_WINDOW)
        p = full_distribution(agent, ob)
        dec = {c: {"farm_diff": float((res.farm[c][:, 0] - res.farm[c][:, 1]).mean()),
                   "deaths": res.deaths[c].mean(0).tolist(), "recalls": res.recalls[c].mean(0).tolist(),
                   "invested": [float(res.m4[c]["invested"][:, k].mean()) / cfg.gold_scale for k in (0, 1)],
                   "bounty_diff": float((res.bounty[c][:, 0] - res.bounty[c][:, 1]).mean()),
                   "plates_diff": float((res.plates[c][:, 0] - res.plates[c][:, 1]).mean())}
               for c in res.candidates}
        out.append({"seat": seat, "k": snap.k, "features": feats, "snapshot": snap.to_dict(), "pi_recall": float(p[RECALL]),
                    "q": {c: res.mean(c) for c in res.candidates},
                    "recall_vs_policy": res.diff("Recall", "policy"), "recall_vs_stay": res.diff("Recall", "stay"),
                    "recall_vs_safe": res.diff("Recall", "safe"), "decomposition": dec})
    return (spec, opp, cls), {"spec": spec, "opponent": opp, "class": cls, "states": out, "n_found": len(states),
                              "elapsed_s": time.perf_counter() - t0}


def recall_vs_opponents(specs: Sequence[str], opponents: Sequence[str] = ("push_punish", "clear_punish"),
                        classes: Sequence[str] = ("low_hp", "after_own_crash", "any"), workers: int = 6,
                        n_states: int = R_STATES, n_roll: int = R_ROLLOUTS) -> dict:
    """Adds `vs_opponents` to recall.json (resumable)."""
    t0 = time.perf_counter()
    prev = load("recall") or {}
    res = dict(prev.get("vs_opponents", {}))
    jobs = [(spec, opp, cls, n_states, n_roll, RECALL_STATE_SEED + 400 + 20 * i + 5 * j + k,
             RECALL_ROLL_SEED + 400 + 20 * i + 5 * j + k)
            for i, spec in enumerate(specs) for j, opp in enumerate(opponents) for k, cls in enumerate(classes)]
    todo = [j for j in jobs if f"{j[0]}|{j[1]}|{j[2]}" not in res]
    print(f"recall vs opponents: {len(jobs)} jobs, {len(todo)} to compute", flush=True)
    done = run_jobs(_recall_opp_job, todo, workers, label="recall-vs-opp")
    res.update({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in done})
    for v in res.values():
        v["summary"] = aggregate_recall(v["states"])
    prev["vs_opponents"] = res
    prev.setdefault("meta", {})["vs_opponents_elapsed_s"] = time.perf_counter() - t0
    save_json("recall", prev)
    return prev


# ---------------------------------------------------------------------------------------------- death against recall
def _dv_job(args) -> tuple:
    spec, n_states, n_roll, state_seed, roll_seed = args
    import torch
    from ....diagnostics import FollowPolicy, ProbCache, rollout, rollout_seed
    from ....env import LaneEnv
    torch.set_num_threads(1)
    cfg = cfg_game()
    agent = agent_from_spec(spec, cfg)
    T = cfg.tower
    t0 = time.perf_counter()
    mask = {"present": lambda R, S: ~R["enemy_absent"]}
    states = sample_states(cfg, agent, mask, n_states, state_seed, per_game=1)["present"]
    env = LaneEnv(cfg, cache_size=20_000)
    cache = ProbCache()
    out = []
    for i, (snap, seat, feats) in enumerate(states):
        variants = {"alive": snap,
                    "dead": snap.with_side(seat, pos=-T, hp=0, absent=cfg.death_turns),
                    "recalled": snap.with_side(seat, pos=-T, absent=cfg.recall_turns - 1)}
        rets = {name: np.array([rollout(env, sn, seat, FollowPolicy(), (agent, agent), rollout_seed(roll_seed, m, i),
                                        cache, max_turns=CF_WINDOW).ret for m in range(n_roll)])
                for name, sn in variants.items()}
        d = {name: [float((rets[name] - rets["alive"]).mean()),
                    float((rets[name] - rets["alive"]).std(ddof=1) / np.sqrt(n_roll))] for name in rets}
        out.append({"seat": seat, "k": snap.k, "features": feats, "v": {n: float(r.mean()) for n, r in rets.items()},
                    "diff_vs_alive": d, "recalled_minus_dead": float((rets["recalled"] - rets["dead"]).mean())})
    return spec, {"spec": spec, "states": out, "elapsed_s": time.perf_counter() - t0}


def aggregate_dv(states: List[dict], farm_per_turn: float, bounty: float) -> dict:
    """Mean downstream costs by HP band and overall, with the full-cost adjustments (bounty; the Recall turn)."""
    out = {}
    bands = {"all": lambda f: True, "hp<=40%": lambda f: f["hp_frac"] <= 0.4,
             "40%<hp<=80%": lambda f: 0.4 < f["hp_frac"] <= 0.8, "hp>80%": lambda f: f["hp_frac"] > 0.8,
             "after own crash": lambda f: f["own_siege"] > 0, "enemy pushing": lambda f: f["en_fh"] > 0 and f["en_fx"] <= -1,
             "gold>=4": lambda f: f["gold"] >= 4}
    for name, fn in bands.items():
        ss = [s for s in states if fn(s["features"])]
        if len(ss) < 2:
            continue
        dead = np.array([s["diff_vs_alive"]["dead"][0] for s in ss])
        rec = np.array([s["diff_vs_alive"]["recalled"][0] for s in ss])
        ci = lambda v: float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        out[name] = {"n": len(ss), "dead_minus_alive": [float(dead.mean()), ci(dead)],
                     "recalled_minus_alive": [float(rec.mean()), ci(rec)],
                     "death_full_cost": float(-dead.mean() + bounty),
                     "recall_full_cost": float(-rec.mean() + farm_per_turn),
                     "death_premium_over_recall": float((-dead.mean() + bounty) - (-rec.mean() + farm_per_turn))}
    return out


# ---------------------------------------------------------------------------------------------- stage
def mechanics_table(cfg) -> dict:
    return {"recall_turns_total": cfg.recall_turns, "recall_line": cfg.recall_line,
            "recall_interruptible": "yes: an enemy Trade reaching the recalling champion interrupts it",
            "death_turns_after": cfg.death_turns, "kill_bounty": cfg.kill_bounty,
            "returns_with_full_hp": "both", "invests_all_unspent_gold_at_base": "both (continuous investment)",
            "position_requirement": "Recall: at or behind own -1 (recall_line); death: anywhere"}


def recall(conds: Sequence[str] = tuple(CONDITIONS), seeds: Sequence[int] = SEEDS, workers: int = 9,
           specs=None, n_states: int = R_STATES, n_roll: int = R_ROLLOUTS, dv_states: int = DV_STATES,
           profile_games: int = 12, save: bool = True, force: bool = False) -> dict:
    """Results already in recall.json are reused unless `force`."""
    t0 = time.perf_counter()
    cfg = cfg_game()
    if specs is None:
        specs = [f"final:{c}:{seeds[0]}" for c in conds] + [f"final:{PRIMARY}:{s}" for s in seeds[1:] if PRIMARY in conds]
    cf_jobs = [(spec, cls, n_states, n_roll, RECALL_STATE_SEED + 10 * i + j, RECALL_ROLL_SEED + 10 * i + j)
               for i, spec in enumerate(specs) for j, cls in enumerate(RECALL_CLASSES)]
    dv_jobs = [(spec, dv_states, n_roll, RECALL_STATE_SEED + 200 + i, RECALL_ROLL_SEED + 200 + i)
               for i, spec in enumerate(specs)]
    prev = (load("recall") or {}) if (save and not force) else {}
    cf_res = dict(prev.get("counterfactual", {}))
    dv_res = dict(prev.get("death_vs_recall", {}))
    profiles = dict(prev.get("profiles", {}))
    cf_jobs = [j for j in cf_jobs if f"{j[0]}|{j[1]}" not in cf_res]
    dv_jobs = [j for j in dv_jobs if j[0] not in dv_res]
    print(f"recall: {len(cf_jobs)} counterfactual jobs, {len(dv_jobs)} death-vs-recall jobs to compute", flush=True)
    cf = run_jobs(_recall_job, cf_jobs, workers, label="recall-cf")
    dv = run_jobs(_dv_job, dv_jobs, workers, label="death-vs-recall")
    for spec in specs:
        if spec not in profiles:
            profiles[spec] = profile(cfg, agent_from_spec(spec, cfg), profile_games, WINDOW_DEV, PROFILE_SEED + 7)
    cf_res.update({f"{k[0]}|{k[1]}": v for k, v in cf})
    for v in cf_res.values():
        v["summary"] = aggregate_recall(v["states"])
    reg = {spec: regression([s for key, v in cf_res.items() if key.split("|")[0] == spec for s in v["states"]])
           for spec in specs}
    for spec, v in dv:
        farm = profiles[spec]["per_game_farm"] / max(profiles[spec]["per_game_present"], 1)
        v["summary"] = aggregate_dv(v["states"], farm, cfg.kill_bounty)
        v["farm_per_present_turn"] = farm
        dv_res[spec] = v
    data = {"meta": meta(stage="recall", window=CF_WINDOW, states=n_states, rollouts=n_roll, dv_states=dv_states,
                         state_seed=RECALL_STATE_SEED, roll_seed=RECALL_ROLL_SEED),
            "mechanics": mechanics_table(cfg), "classes": RECALL_CLASSES, "elapsed_s": time.perf_counter() - t0,
            "profiles": profiles, "counterfactual": cf_res, "regression": reg, "death_vs_recall": dv_res}
    if save:
        save_json("recall", data)
    return data


__all__ = ["recall", "recall_vs_opponents", "RECALL_CLASSES", "recall_masks", "aggregate_recall", "aggregate_dv",
           "regression", "mechanics_table"]
