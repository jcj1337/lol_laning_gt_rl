"""Part C progression audit: when do plates fall, at which levels and wealth, and how much of the discounted objective
lies after the proposed termination (either tower's final plate) or after both champions reach the level cap?

Whole games from s0 on the unchanged `m4_lasthit()`, W = 1000 turns, 32 games per matchup with seeded chance and agent
streams (`episode_streams`), the first-named agent in seat A. Matchups: self-play of the M4.3b finals (every condition),
the baseline final against every reactive reference, the attacker against its target, and reference-against-reference
games (the scripted strategy space, which never learned anything). Distributions are reported per matchup and per
matchup family; nothing is pooled across families.

Per game (turns count from 1: "turn n" is the state after n played turns):
  plate[s][j]          first turn at which side s's cumulative damage on the ENEMY tower reaches threshold j
                       (j = 0, 1, 2); the final plate of the enemy tower is j = 2
  end                  min over s of plate[s][2]: the termination turn of the Part C proposal (None within W)
  simultaneous         both final plates on the same turn
  lvl18[s], both18     first turn at level 18
  at each plate event  both levels, cumulative earned gold, invested gold, credited XP
  tower damage split   minion hits (100 centi-units per minion-damage unit of the crashed mass) and Pressure
                       (P(L, S) per effective Apply Pressure turn), before the clamp at the last threshold
  discounted weight    gamma^end and gamma^both18: the share of the objective's total weight sum_k gamma^k that lies
                       after the event
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from ...config import PRESSURE
from .common import PROGRESSION_SEED, cfg_game, mean_ci, meta, quantiles, save_json

WINDOW = 1000
EPISODES = 32
CURVE_EVERY = 10

FAMILIES = {
    "selfplay": [("base_s0", "final:base:0", "self"), ("base_s1", "final:base:1", "self"),
                 ("base_s2", "final:base:2", "self"), ("ent03_s0", "final:ent03:0", "self"),
                 ("ent03_s1", "final:ent03:1", "self"), ("ent03_s2", "final:ent03:2", "self"),
                 ("hist80_s0", "final:hist80:0", "self"), ("mixed_s0", "final:mixed:0", "self")],
    "policy_vs_ref": [(f"base_s0 vs {r}", "final:base:0", f"ref:{r}") for r in
                      ("farm", "push_punish", "clear_punish", "bully_shove", "freeze_punish", "respect", "crash_recall",
                       "trading_pusher")] + [(f"ent03_s0 vs {r}", "final:ent03:0", f"ref:{r}") for r in
                                             ("farm", "push_punish", "clear_punish", "bully_shove")],
    "attacker": [("attacker vs base_s0", "attacker:base:0", "final:base:0")],
    "ref_vs_ref": [("farm vs farm", "ref:farm", "ref:farm"),
                   ("push_punish vs push_punish", "ref:push_punish", "ref:push_punish"),
                   ("clear_punish vs clear_punish", "ref:clear_punish", "ref:clear_punish"),
                   ("freeze_punish vs clear_punish", "ref:freeze_punish", "ref:clear_punish"),
                   ("respect vs clear_punish", "ref:respect", "ref:clear_punish"),
                   ("bully_shove vs push_punish", "ref:bully_shove", "ref:push_punish")],
}


def play_game(cfg, agents, seed, window: int = WINDOW) -> dict:
    """One recorded game; returns the per-game event summary and coarse curves (module docstring)."""
    from ...champions_m4 import context
    from ...env import LaneEnv
    from ...env.agents import episode_streams
    from ...env.discounted import game_agents
    ctx = context(cfg)
    t_tab = ctx.t
    scale = cfg.gold_scale
    th = cfg.plate_thresholds
    _, env_ss, rng_a, rng_b = episode_streams(seed)
    rngs = (rng_a, rng_b)
    agents = game_agents(agents)
    notify = [getattr(ag, "notify", None) for ag in agents]
    env = LaneEnv(cfg, max_turns=window, cache_size=20_000)
    obs = env.reset(seed=env_ss)
    plate = [[None] * len(th), [None] * len(th)]
    plate_ctx: Dict[str, dict] = {}
    lvl18 = [None, None]
    earned = [0, 0]
    xp_cred = [0, 0]
    xp_disc = [0, 0]
    minion_dmg = [0, 0]
    press_dmg = [0, 0]
    press_turns = [0, 0]
    deaths, recalls, away = [0, 0], [0, 0], [0, 0]
    rew = np.zeros(window)
    curve = {k: [] for k in ("level_a", "level_b", "tau_a", "tau_b", "earned_a", "earned_b", "xp_a", "xp_b",
                             "invested_a", "invested_b")}
    for n in range(1, window + 1):
        a = agents[0].act(obs[0], rngs[0])
        b = agents[1].act(obs[1], rngs[1])
        r = env.step(a, b)
        info = r.info
        for side in (0, 1):
            if callable(notify[side]):
                notify[side](info, side)
        rew[n - 1] = r.rewards[0]
        ca = info["champ_after"]
        cb = info["champ_before"]
        for s in (0, 1):
            earned[s] += info["earned_units"][s]
            xp_cred[s] += info["xp_gain"][s]
            xp_disc[s] += info["xp_discarded"][s]
            minion_dmg[s] += 100 * info["tower_hits"][s]
            if info["effective_objectives"][s] == PRESSURE:
                lv0, it0 = info["levels_before"][s], cb[s][4]
                press_dmg[s] += ctx.pressure(lv0, it0)
                press_turns[s] += 1
            deaths[s] += int(info["died"][s])
            recalls[s] += int(info["recalled"][s])
            away[s] += int(info["away"][s])
            if lvl18[s] is None and info["levels"][s] >= t_tab.level_cap:
                lvl18[s] = n
        for s in (0, 1):
            tau = ca[s][6]
            for j, thr in enumerate(th):
                if plate[s][j] is None and tau >= thr:
                    plate[s][j] = n
                    plate_ctx[f"{s}_{j}"] = {
                        "turn": n, "levels": list(info["levels"]),
                        "earned": [earned[0] / scale, earned[1] / scale],
                        "invested": [ca[0][4] / scale, ca[1][4] / scale],
                        "unspent": [ca[0][5] / scale, ca[1][5] / scale],
                        "xp": [ca[0][3], ca[1][3]]}
        if n % CURVE_EVERY == 0:
            curve["level_a"].append(info["levels"][0])
            curve["level_b"].append(info["levels"][1])
            curve["tau_a"].append(ca[0][6])
            curve["tau_b"].append(ca[1][6])
            curve["earned_a"].append(earned[0] / scale)
            curve["earned_b"].append(earned[1] / scale)
            curve["xp_a"].append(ca[0][3])
            curve["xp_b"].append(ca[1][3])
            curve["invested_a"].append(ca[0][4] / scale)
            curve["invested_b"].append(ca[1][4] / scale)
        if r.done:
            break
        obs = r.obs
    finals = [plate[0][-1], plate[1][-1]]
    ends = [x for x in finals if x is not None]
    end = min(ends) if ends else None
    both18 = max(lvl18) if None not in lvl18 else None
    g = cfg.gamma
    k = np.arange(window)
    disc_total = float(np.sum(rew * g ** k))
    # the discounted payoff to A that a termination at `end` would have kept (the turns after it removed)
    disc_to_end = float(np.sum(rew[:end] * g ** k[:end])) if end else disc_total
    return {"plate": plate, "plate_ctx": plate_ctx, "end": end,
            "end_side": (None if end is None else [s for s in (0, 1) if finals[s] == end]),
            "simultaneous": bool(end is not None and finals[0] == finals[1]),
            "lvl18": lvl18, "both18": both18,
            "weight_after_end": (g ** end if end else 0.0), "weight_after_both18": (g ** both18 if both18 else 0.0),
            "earned": [earned[0] / scale, earned[1] / scale], "xp": xp_cred, "xp_discarded": xp_disc,
            "minion_tower_damage": minion_dmg, "pressure_tower_damage": press_dmg, "pressure_turns": press_turns,
            "deaths": deaths, "recalls": recalls, "away_turns": away,
            "disc_payoff_a": disc_total, "disc_payoff_a_to_end": disc_to_end, "curve": curve}


def _job(args) -> tuple:
    name, spec_a, spec_b, mi, episodes, window = args
    import torch
    from ..rl.m43b.common import agent_from_spec
    torch.set_num_threads(1)
    cfg = cfg_game()
    xa = agent_from_spec(spec_a, cfg)
    yb = xa if spec_b == "self" else agent_from_spec(spec_b, cfg)
    t0 = time.perf_counter()
    games = []
    for ep in range(episodes):
        seed = np.random.SeedSequence(PROGRESSION_SEED, spawn_key=(mi, ep))
        games.append(play_game(cfg, (xa, yb), seed, window))
    return name, {"spec_a": spec_a, "spec_b": spec_b, "games": games, "elapsed_s": time.perf_counter() - t0}


def _num(x):
    return float("nan") if x is None else float(x)


def summarize(games: List[dict], window: int = WINDOW, gamma: float = 0.99) -> dict:
    """Distributions over the games of one matchup (module docstring). Plate turns are pooled over the two towers
    (side s = the attacker); 'end' is per game."""
    out = {"n": len(games)}
    for j in range(3):
        turns = [g["plate"][s][j] for g in games for s in (0, 1)]
        out[f"plate{j + 1}_turn"] = quantiles([_num(x) for x in turns])
        out[f"plate{j + 1}_share_within_W"] = float(np.mean([x is not None for x in turns]))
        ctxs = [g["plate_ctx"][f"{s}_{j}"] for g in games for s in (0, 1) if f"{s}_{j}" in g["plate_ctx"]]
        if ctxs:
            lv = [c["levels"][s] for c in ctxs for s in (0, 1)]
            out[f"plate{j + 1}_levels"] = quantiles(lv)
            out[f"plate{j + 1}_earned"] = quantiles([c["earned"][s] for c in ctxs for s in (0, 1)])
            out[f"plate{j + 1}_invested"] = quantiles([c["invested"][s] for c in ctxs for s in (0, 1)])
    ends = [g["end"] for g in games]
    out["end_turn"] = quantiles([_num(x) for x in ends])
    out["end_share_within_W"] = float(np.mean([x is not None for x in ends]))
    for W in (100, 150, 200, 300, 460):
        out[f"end_share_by_{W}"] = float(np.mean([x is not None and x <= W for x in ends]))
    out["simultaneous_share"] = float(np.mean([g["simultaneous"] for g in games]))
    end_levels = []
    for g in games:
        if g["end"] is None:
            continue
        lv = g["curve"]  # coarse; exact levels at the end come from plate_ctx of the ending tower
        for s in g["end_side"]:
            c = g["plate_ctx"].get(f"{s}_2")
            if c:
                end_levels += c["levels"]
                break
    out["end_levels"] = quantiles(end_levels)
    out["lvl18_turn"] = quantiles([_num(x) for g in games for x in g["lvl18"]])
    out["both18_turn"] = quantiles([_num(g["both18"]) for g in games])
    out["both18_share_within_W"] = float(np.mean([g["both18"] is not None for g in games]))
    out["weight_after_end"] = mean_ci([g["weight_after_end"] for g in games])
    out["weight_after_both18"] = mean_ci([g["weight_after_both18"] for g in games])
    tot_min = np.array([sum(g["minion_tower_damage"]) for g in games], dtype=float)
    tot_pr = np.array([sum(g["pressure_tower_damage"]) for g in games], dtype=float)
    out["pressure_share_of_tower_damage"] = float(tot_pr.sum() / max(1.0, tot_pr.sum() + tot_min.sum()))
    out["pressure_turns_per_game"] = float(np.mean([sum(g["pressure_turns"]) for g in games]))
    out["deaths_per_side"] = float(np.mean([np.mean(g["deaths"]) for g in games]))
    out["recalls_per_side"] = float(np.mean([np.mean(g["recalls"]) for g in games]))
    out["xp_discarded_per_side"] = float(np.mean([np.mean(g["xp_discarded"]) for g in games]))
    out["earned_per_side_W"] = float(np.mean([np.mean(g["earned"]) for g in games]))
    out["abs_disc_payoff"] = float(np.mean([abs(g["disc_payoff_a"]) for g in games]))
    out["disc_payoff_a"] = mean_ci([g["disc_payoff_a"] for g in games])
    out["disc_payoff_a_to_end"] = mean_ci([g["disc_payoff_a_to_end"] for g in games])
    # mean curves (both seats pooled for levels; per seat for wealth)
    cv = {}
    for key in games[0]["curve"]:
        arr = [g["curve"][key] for g in games]
        L = min(len(a) for a in arr)
        cv[key] = np.mean([a[:L] for a in arr], axis=0).round(3).tolist()
    out["curve_every"] = CURVE_EVERY
    out["curves"] = cv
    return out


def run(families: Sequence[str] = tuple(FAMILIES), workers: int = 9, episodes: int = EPISODES,
        window: int = WINDOW) -> dict:
    from ..rl.m43b.common import run_jobs
    jobs, fam_of = [], {}
    mi = 0
    for fam in families:
        for name, a, b in FAMILIES[fam]:
            jobs.append((name, a, b, mi, episodes, window))
            fam_of[name] = fam
            mi += 1
    t0 = time.perf_counter()
    res = run_jobs(_job, jobs, workers, "progression")
    cfg = cfg_game()
    out = {"meta": meta(stage="progression", window=window, episodes=episodes, seed_root=PROGRESSION_SEED,
                        elapsed_s=None),
           "matchups": {}}
    for name, d in res:
        out["matchups"][name] = {"family": fam_of[name], "spec_a": d["spec_a"], "spec_b": d["spec_b"],
                                 "summary": summarize(d["games"], window, cfg.gamma),
                                 "games": [{k: v for k, v in g.items() if k != "curve"} for g in d["games"]]}
    out["meta"]["elapsed_s"] = time.perf_counter() - t0
    save_json("progression", out)
    return out


__all__ = ["FAMILIES", "play_game", "summarize", "run"]
