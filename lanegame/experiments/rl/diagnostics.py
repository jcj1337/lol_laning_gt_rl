"""Strategic diagnostics of the learned champion-mode policies (docs/rl/openai_five_diagnostics_report.md).

Stages (each writes results/rl/diagnostics/<stage>.json and .md; nothing of earlier studies is overwritten):

  audit           Phase A: action legality vs selection, relevant situations, states, outcomes and the reward
                  decomposition of the 9 main-study policies (+ the published pub_sprand_s1), in mirror games and
                  against two scripted agents, at equal skill; mirror games at unequal skills.
  hpvalue         Phase A.3: what HP is worth. Paired rollouts from sampled mirror states in which only the focal
                  champion's HP (or its alive/dead/recalling status) differs.
  counterfactual  Phase B: paired counterfactual evaluation of every legal first action (and short plans) at
                  sampled reachable states, per scenario, with the policy itself as continuation and opponent.
  overlay         Phase C: small interpretable rule overlays on a learned policy (it plays the policy except where the
                  rule fires) head-to-head against the unmodified policy.

Seeds: every stream of this module derives from SeedSequence roots 52,000-58,999, disjoint from all training
streams (PPO seeds 0-2 / 900-912 and their children) and from the evaluation roots of the earlier studies
(10,000-43,000, 77,777) and of Phase D (50,000).

    python -m lanegame.experiments.rl.diagnostics audit | hpvalue | counterfactual | overlay
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

from ...config import GameConfig
from .common import md_table
from .openai_five import CONDITIONS, LABEL, SEEDS, fmt_seed_ci, run_dir

OUT = os.path.join("results", "rl", "diagnostics")
WORKERS = 5
AUDIT_ROOT, HP_ROOT, CF_STATE_ROOT, CF_ROLL_ROOT, OVERLAY_ROOT = 52_000, 53_000, 54_000, 55_000, 56_000
AUDIT_EPISODES, AUDIT_EPISODES_SCRIPTED = 200, 100
AUDIT_OPPONENTS = ("self", "hold_farmer", "aggressive_trader")
SKILL_PAIRS = ((0.2, 0.8), (0.8, 0.2))
REPRESENTATIVE = ("m3b_s0", "sp100_s0", "sp80_s0")


def policies() -> Dict[str, str]:
    out = {f"{c}_s{s}": os.path.join(run_dir("main", c, s), "final.pt") for c in CONDITIONS for s in SEEDS}
    out["pub_sprand_s1"] = os.path.join("results", "rl", "checkpoints", "m3b_skillrand_s1", "final.pt")
    return out


def _agent(spec: str, cfg: GameConfig):
    from .evaluation import _resolve
    return _resolve(spec, cfg)


def _write(name: str, data: dict, md_fn) -> None:
    """Write the JSON first, then the Markdown generated from the reloaded JSON (so a formatting error never loses
    the computed results, and the tables are built from exactly what is on disk)."""
    from ...io import dump_json
    os.makedirs(OUT, exist_ok=True)
    dump_json(data, os.path.join(OUT, f"{name}.json"))
    print(f"wrote {OUT}/{name}.json", flush=True)
    md = md_fn(_load(name)) if callable(md_fn) else md_fn
    with open(os.path.join(OUT, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {OUT}/{name}.md", flush=True)


def _load(name: str) -> dict:
    with open(os.path.join(OUT, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def _f(x, fmt="{:.2f}") -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


# ---------------------------------------------------------------------------------------------- Phase A: audit
def _audit_job(args):
    key, x_spec, y_spec, episodes, seed, skill = args
    import torch
    from ...diagnostics import audit_games, summarize
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    x = _agent(x_spec, cfg)
    y = x if y_spec == "self" else _agent(y_spec, cfg)
    t0 = time.perf_counter()
    T = audit_games(cfg, x, y, episodes, seed, skill)
    out = summarize(T)
    out["elapsed_s"] = time.perf_counter() - t0
    return key, out


def audit(workers: int = WORKERS):
    t0 = time.perf_counter()
    pols = policies()
    jobs = []
    for n, p in pols.items():
        for o in AUDIT_OPPONENTS:
            eps = AUDIT_EPISODES if o == "self" else AUDIT_EPISODES_SCRIPTED
            jobs.append(((n, o, "0.5/0.5"), p, "self" if o == "self" else o, eps, AUDIT_ROOT, None))
        for sk in SKILL_PAIRS:
            jobs.append(((n, "self", f"{sk[0]}/{sk[1]}"), p, "self", AUDIT_EPISODES_SCRIPTED, AUDIT_ROOT + 1, sk))
    print(f"audit: {len(jobs)} jobs", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(_audit_job, jobs))
    data = {"root": AUDIT_ROOT, "policies": pols, "elapsed_s": time.perf_counter() - t0,
            "results": {"|".join(k): v for k, v in res.items()}}
    _write("audit", data, audit_md)
    return data


def _families():
    return [(LABEL[c], [f"{c}_s{s}" for s in SEEDS]) for c in CONDITIONS] + [("pub_sprand_s1 (published M3b)",
                                                                               ["pub_sprand_s1"])]


def audit_md(data: dict) -> str:
    R = data["results"]
    L = ["# Phase A: strategic action audit", "",
         f"Generated by `python -m lanegame.experiments.rl.diagnostics audit` ({data['elapsed_s'] / 60:.1f} min). "
         f"Each policy plays {AUDIT_EPISODES} episodes per seat against itself (mirror) and {AUDIT_EPISODES_SCRIPTED} "
         "per seat against `hold_farmer` and `aggressive_trader`, equal skill 0.5, held-out seeds "
         f"SeedSequence({AUDIT_ROOT}). Shares are over the focal policy's PRESENT turns. `p | legal` is the policy's "
         "mean probability of the category on turns where it is legal (the expected selection rate), `chosen | legal` "
         "the realized rate. Family values are means over the 3 training seeds.", ""]
    cats = ["Hold", "Yield", "Trade", "Safe", "Recall", "SH", "F", "Pressure", "Vision"]
    for opp in AUDIT_OPPONENTS:
        rows = []
        for lab, names in _families():
            rs = [R[f"{n}|{opp}|0.5/0.5"]["legality"] for n in names]
            row = [lab]
            for c in cats:
                ls = np.mean([r[c]["legal_share"] for r in rs])
                pl = np.nanmean([r[c]["p_when_legal"] for r in rs]) if ls > 0 else float("nan")
                row.append(f"{ls:.2f} / {_f(pl, '{:.3f}')}")
            rows.append(row)
        L += [f"## Legality and selection, opponent: {opp} (cell: legal share / p | legal)", "",
              md_table(["family"] + cats, rows), ""]
    # opportunities (mirror)
    names_all = [n for _, ns in _families() for n in ns]
    sits = list(R[f"{names_all[0]}|self|0.5/0.5"]["opportunities"])
    for opp in ("self", "aggressive_trader"):
        rows = []
        for s in sits:
            acts = list(R[f"{names_all[0]}|{opp}|0.5/0.5"]["opportunities"][s]["p"])
            for lab, names in _families():
                rs = [R[f"{n}|{opp}|0.5/0.5"]["opportunities"][s] for n in names]
                rows.append([s, lab, f"{np.mean([r['share_of_present_turns'] for r in rs]):.3f}",
                             f"{np.mean([r['n'] for r in rs]):.0f}"] +
                            ["; ".join(f"{a} {_f(np.nanmean([r['p'][a] for r in rs]), '{:.3f}')}" for a in acts)])
        L += [f"## Relevant situations, opponent: {opp}", "",
              "`share` = share of present turns in that situation; `n` = number of such turns in the audit (per policy); "
              "then the policy's mean probability of each relevant category in that situation.", "",
              md_table(["situation", "family", "share", "n", "policy probabilities"], rows), ""]
    # states
    rows = []
    for lab, names in _families():
        st = [R[f"{n}|self|0.5/0.5"]["states"] for n in names]
        m = lambda k: np.mean([s[k] for s in st])
        hh = " ".join(f"{h}:{np.mean([s['hp_hist'][str(h)] for s in st]):.2f}" for h in range(1, 7))
        ph = " ".join(f"{x}:{np.mean([s['pos_hist'][str(x)] for s in st]):.2f}" for x in range(-2, 3))
        rb = " ".join(f"{h}:{_f(np.nanmean([s['recall_legal_by_hp'][str(h)] for s in st]))}" for h in range(1, 7))
        fs = " ".join(f"{x}:{np.mean([s['farm_spot_hist'][str(x)] for s in st]):.2f}" for x in range(-2, 3))
        rows.append([lab, hh, ph, rb, fs, f"{m('turns_hp_le2_per_game'):.1f}", f"{m('deaths_per_game'):.2f}",
                     f"{m('deaths_with_gank_damage'):.2f}", f"{m('freeze_legal_per_game'):.2f}",
                     f"{m('objective_legal_per_game'):.2f}", f"{m('crashes_per_game'):.2f}",
                     f"{m('enemy_away_turns_per_game'):.1f}"])
    L += ["## States in mirror games", "",
          md_table(["family", "HP at turn start (share)", "cell at turn start", "Recall legal by HP", "farm spot (Hold's cell)",
                    "turns at HP <= 2 / game", "deaths / game", "deaths with gank damage", "Freeze-legal turns / game",
                    "objective-legal turns / game", "crashes / game", "enemy-away turns / game"], rows), ""]
    # outcomes
    rows = []
    for lab, names in _families():
        for b in ("Hold", "Trade", "Yield", "Safe", "Recall", "Shove"):
            os_ = [R[f"{n}|self|0.5/0.5"]["outcomes"][b] for n in names]
            ntot = sum(o["n"] for o in os_)
            if ntot == 0:
                continue
            w = lambda k: np.nansum([o[k] * o["n"] for o in os_]) / ntot
            rows.append([lab, b, ntot, f"{w('hp'):.2f}", f"{w('reward'):+.3f}", f"{w('reward_next4'):+.3f}",
                         f"{w('farm'):.3f}", f"{w('efarm'):.3f}", f"{w('dmg_taken'):.2f}", f"{w('dmg_dealt'):.2f}",
                         f"{w('died_next4'):.3f}", f"{w('connected'):.2f}"])
    L += ["## Realized outcomes by chosen action, mirror games (descriptive, confounded by state)", "",
          md_table(["family", "action", "turns", "HP", "reward", "reward next 4 turns", "own farm", "enemy farm",
                    "HP lost", "HP dealt", "died within 4 turns", "trade connected"], rows), ""]
    # economy
    rows = []
    for opp in AUDIT_OPPONENTS:
        for lab, names in _families():
            ec = [R[f"{n}|{opp}|0.5/0.5"]["economy"] for n in names]
            m = lambda k: np.mean([e["per_game"][k] for e in ec])
            lost = " ".join(f"{md}:{np.mean([e['farm_lost_by_mode'][md] for e in ec]):.2f}"
                            for md in ("farm", "trading", "retreat", "safe", "channel", "away"))
            rows.append([opp, lab, f"{m('reward'):+.2f}", f"{m('farm'):.2f} / {m('avail'):.2f}",
                         f"{m('efarm'):.2f}", f"{m('plates'):.2f} / {m('eplates'):.2f}",
                         f"{m('bounty'):.2f} / {m('ebounty'):.2f}", lost,
                         f"{np.mean([e['opp_farm_per_turn_while_away'] for e in ec]):.2f} vs "
                         f"{np.mean([e['opp_farm_per_turn_while_present'] for e in ec]):.2f}",
                         f"{np.mean([e['payoff_per_kill_difference'] for e in ec]):+.2f}"])
    L += ["## Reward decomposition per game (focal policy)", "",
          "`farm lost by mode` = farm available minus collected, summed over the turns spent in each mode "
          "(farm, trading, retreat, safe, channel = interrupted recall, away = recalled or dead). `payoff per kill "
          "difference` = slope of the per-game payoff on (enemy deaths − own deaths).", "",
          md_table(["opponent", "family", "payoff", "farm / available", "enemy farm", "plates own / enemy",
                    "bounty own / enemy", "farm lost by mode", "enemy farm per turn: focal away vs present",
                    "payoff per kill difference"], rows), ""]
    # skill
    rows = []
    for lab, names in _families():
        row = [lab]
        for sk in ("0.2/0.8", "0.5/0.5", "0.8/0.2"):
            rs = [R[f"{n}|self|{sk}"] for n in names]
            row.append(f"{np.mean([r['legality']['Trade']['p_when_legal'] for r in rs]):.2f} / "
                       f"{np.mean([r['economy']['per_game']['reward'] for r in rs]):+.2f}")
        rows.append(row)
    L += ["## Mirror games at unequal skill (own / enemy skill): Trade probability / payoff", "",
          md_table(["family", "0.2 vs 0.8", "0.5 vs 0.5", "0.8 vs 0.2"], rows), ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- Phase B: counterfactuals
CF_STATES, CF_ROLLOUTS = 24, 120
CF_GENERAL_STATES = 10
SCENARIOS = ("recall_low_hp", "low_hp_lane", "lane_contest", "enemy_away", "opp_trades", "danger", "freeze_legal",
             "siege", "skill_weak", "skill_strong")
GENERAL_SCENARIOS = ("recall_low_hp", "low_hp_lane", "lane_contest", "enemy_away", "opp_trades")
SCENARIO_TEXT = {
    "recall_low_hp": "own HP <= 2, Recall legal (cell <= -1), enemy present",
    "low_hp_lane": "own HP <= 2 at or past the lane (Recall illegal), enemy present",
    "lane_contest": "both present, both HP >= 3, turn 2-28",
    "enemy_away": "enemy champion away (recall or death), own champion present",
    "opp_trades": "enemy present and the opponent policy's P(Trade) >= 0.6",
    "danger": "own HP <= 3 and opponent P(Trade) >= 0.4",
    "freeze_legal": "Freeze legal (constructed: the opponent shoves SH-H for up to 4 turns from a mirror-game state)",
    "siege": "siege objective legal (constructed: the focal shoves SH-H for up to 6 turns from a mirror-game state)",
    "skill_weak": "both present, both HP >= 3, focal skill 0.2 vs enemy 0.8",
    "skill_strong": "both present, both HP >= 3, focal skill 0.8 vs enemy 0.2",
}
COMPARISONS = {
    "recall_low_hp": [("SP-R", "SP-H"), ("SP-R", "SP-T"), ("SP-R", "pi")],
    "low_hp_lane": [("back_recall", "SP-H"), ("back_recall", "pi"), ("SP-S", "SP-H"), ("SP-Y", "SP-H"),
                    ("SP-T", "SP-H")],
    "lane_contest": [("SH-H", "SP-H"), ("SH-T", "SP-T"), ("SP-T", "SP-H"), ("SH-H", "pi")],
    "enemy_away": [("SH-H", "SP-H"), ("shove_while_away", "SP-H"), ("SP-T", "SP-H"), ("shove_while_away", "pi")],
    "opp_trades": [("SP-Y", "SP-H"), ("SP-Y", "SP-T"), ("SP-S", "SP-H"), ("SP-Y", "pi")],
    "danger": [("SP-S", "SP-Y"), ("SP-S", "SP-H"), ("SP-Y", "SP-H"), ("SP-T", "SP-H")],
    "freeze_legal": [("F-H", "SP-H"), ("F-H", "SH-H"), ("F-T", "SP-T"), ("F-H", "pi")],
    "siege": [("SP-H+P", "SP-H"), ("SP-H+V", "SP-H"), ("SP-H+P", "SP-H+V"), ("pressure_hold", "SP-H"),
              ("SP-H+P", "pi")],
    "skill_weak": [("SP-T", "SP-H"), ("SP-T", "pi")],
    "skill_strong": [("SP-T", "SP-H"), ("SP-T", "pi")],
}


def _scenario_skills(scenario: str):
    return {"skill_weak": (0.2, 0.8), "skill_strong": (0.8, 0.2)}.get(scenario, (0.5, 0.5))


def _plans(scenario: str, cfg: GameConfig) -> dict:
    """Scenario-specific multi-turn plans (beyond committing one action)."""
    from ...champions import RECALL
    from ...config import B_HOLD, B_SAFE, PRESSURE, SH, SP, champ_act
    from ...diagnostics import FunctionPlan
    if scenario == "low_hp_lane":
        def back_recall(obs, t):                  # walk back with Safe until Recall is legal (at most 2 turns), recall
            if obs.legal == (45,):                # away (it recalled, or died on the way back): plan over
                return None
            if RECALL in obs.legal:
                return RECALL
            return champ_act(SP, B_SAFE) if t < 2 else None
        return {"back_recall": FunctionPlan("back_recall", back_recall)}
    if scenario == "enemy_away":
        def shove_away(obs, t):                   # Shove with Hold while the enemy champion is away
            enemy = obs.champ.side(1 - obs.side)
            return champ_act(SH, B_HOLD) if enemy.absent > 0 and champ_act(SH, B_HOLD) in obs.legal else None
        return {"shove_while_away": FunctionPlan("shove_while_away", shove_away)}
    if scenario == "siege":
        def pressure(obs, t):                     # keep Applying Pressure (Hold) while the siege lasts
            for a in (champ_act(SP, B_HOLD, PRESSURE), champ_act(SH, B_HOLD, PRESSURE)):
                if a in obs.legal:
                    return a
            return None
        return {"pressure_hold": FunctionPlan("pressure_hold", pressure)}
    return {}


def _candidate_set(scenario: str, cfg: GameConfig, legal) -> dict:
    from ...diagnostics import Commit, FollowPolicy
    cands = {cfg.label(a): Commit(a, cfg.label(a)) for a in legal}
    cands["policy"] = FollowPolicy()
    cands.update(_plans(scenario, cfg))
    return cands


def _find_states(scenario: str, agent, cfg_base: GameConfig, n_states: int, seed: int) -> List[tuple]:
    """Up to n_states (snapshot, focal seat) pairs of a scenario, at most two per episode, from mirror games of
    `agent` (or, for the constructed scenarios, from mirror games with a scripted shove prefix of one seat)."""
    from ...champions import RECALL
    from ...config import B_HOLD, F, SH, champ_act, champ_objective, champ_wave
    from ...diagnostics import DEFER, FunctionPlan, ProbCache, Snapshot, observation, rollout
    from ...env import LaneEnv, play_episode
    from ...diagnostics.audit import category_probabilities, full_distribution
    from ...mechanics import initial_state
    from ...champions import initial_champions
    skills = _scenario_skills(scenario)
    cfg = cfg_base.with_(skill_a=skills[0], skill_b=skills[1])
    env = LaneEnv(cfg, cache_size=20_000)
    rng = np.random.default_rng(seed)
    found: List[tuple] = []
    cache = ProbCache()
    snap0 = Snapshot(initial_state(cfg), initial_champions(cfg), 0, skills)

    def p_trade(snap, seat):
        ob = observation(cfg_base, snap, seat)
        return float(category_probabilities(full_distribution(agent, ob, env.n_actions))["beh"][2])

    for ep in range(4000):
        if len(found) >= n_states:
            break
        ss = np.random.SeedSequence(entropy=seed, spawn_key=(ep,))
        if scenario in ("freeze_legal", "siege"):
            k_start = int(rng.integers(3, 21))
            shover = int(rng.integers(2))
            focal_seat = 1 - shover if scenario == "freeze_legal" else shover
            n_max = 4 if scenario == "freeze_legal" else 6
            sh = champ_act(SH, B_HOLD)

            def prefix(obs, t, k_start=k_start, n_max=n_max):
                if obs.k < k_start:
                    return DEFER
                if obs.k < k_start + n_max and sh in obs.legal:
                    return sh
                return None
            r = rollout(env, snap0, shover, FunctionPlan("prefix", prefix), (agent, agent), ss, cache, record=True)
            for info in r.infos:
                if info["k"] < k_start or info["k"] > k_start + n_max:
                    continue
                legal = info["legal"][focal_seat]
                cb = info["champ_before"]
                if cb[focal_seat][2] != 0 or cb[1 - focal_seat][2] != 0:
                    continue
                ok = any(champ_wave(a) == F for a in legal if a != 45) if scenario == "freeze_legal" else \
                    any(champ_objective(a) > 0 for a in legal if a != 45)
                if ok:
                    found.append((Snapshot.from_info(info, skills), focal_seat))
                    break
            continue
        log = play_episode(env, agent, agent, ss, record=True)
        cands = []
        for info in log.turns:
            for s in (0, 1):
                me, en = info["champ_before"][s], info["champ_before"][1 - s]
                if me[2] != 0:
                    continue
                legal, k = info["legal"][s], info["k"]
                if scenario == "recall_low_hp":
                    ok = me[1] <= 2 and en[2] == 0 and RECALL in legal
                elif scenario == "low_hp_lane":
                    ok = me[1] <= 2 and en[2] == 0 and RECALL not in legal
                elif scenario in ("lane_contest", "skill_weak", "skill_strong"):
                    ok = en[2] == 0 and me[1] >= 3 and en[1] >= 3 and 2 <= k <= 28
                    if scenario != "lane_contest":
                        ok = ok and s == 0                            # focal = seat A holds the scenario's skill
                elif scenario == "enemy_away":
                    ok = en[2] > 0 and k <= 29
                elif scenario in ("opp_trades", "danger"):
                    ok = en[2] == 0 and (scenario == "opp_trades" or me[1] <= 3)
                else:
                    ok = False
                if ok:
                    cands.append((info, s))
        rng.shuffle(cands)
        took = 0
        for info, s in cands:
            snap = Snapshot.from_info(info, skills)
            if scenario in ("opp_trades", "danger"):
                thr = 0.6 if scenario == "opp_trades" else 0.4
                if p_trade(snap, 1 - s) < thr:
                    continue
            found.append((snap, s))
            took += 1
            if took >= 2 or len(found) >= n_states:
                break
    return found


def _state_summary(res, cfg: GameConfig, agent, scenario: str) -> dict:
    from ...diagnostics import observation
    from ...diagnostics.audit import full_distribution
    snap, s = res.snapshot, res.focal
    ob = observation(cfg, snap, s)
    p = full_distribution(agent, ob, len(ob.mask))
    pi = {cfg.label(a): float(p[a]) for a in ob.legal}
    w = {lab: pr for lab, pr in pi.items()}
    q = {c: res.mean(c) for c in res.candidates}
    adv = {c: res.advantage(c, w) for c in res.candidates}
    me, en = snap.champ.side(s), snap.champ.side(1 - s)
    pairs = {}
    for a, b in COMPARISONS[scenario]:
        if a not in res.returns or (b != "pi" and b not in res.returns):
            continue
        pairs[f"{a} vs {b}"] = adv[a] if b == "pi" else res.diff(a, b)
    legal_c = [c for c in res.candidates if c in pi]
    best = max(legal_c, key=lambda c: q[c][0])
    dec = {}
    for c in res.candidates:
        dec[c] = {"farm_diff": float((res.farm[c][:, 0] - res.farm[c][:, 1]).mean()),
                  "bounty_diff": float((res.bounty[c][:, 0] - res.bounty[c][:, 1]).mean()),
                  "plates_diff": float((res.plates[c][:, 0] - res.plates[c][:, 1]).mean()),
                  "deaths": res.deaths[c].mean(0).tolist(), "recalls": res.recalls[c].mean(0).tolist()}
    own = snap.lane.side(s)
    return {"k": snap.k, "seat": s, "hp": me.hp, "pos": me.pos, "ehp": en.hp, "epos": en.pos, "eabsent": en.absent,
            "front_hp": own[1], "snapshot": snap.to_dict(), "pi": pi, "q": q, "adv": adv, "pairs": pairs,
            "best": best, "best_adv": adv[best], "pi_best": pi[best], "decomposition": dec,
            "v_sampled_minus_piq": res.advantage("policy", w)}


def _cf_job(args):
    key, pol_name, path, scenario, n_states, n_roll, state_seed, roll_seed = args
    import torch
    from ...diagnostics import ProbCache, branch
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    agent = _agent(path, cfg)
    t0 = time.perf_counter()
    states = _find_states(scenario, agent, cfg, n_states, state_seed)
    cache = ProbCache()
    out = []
    for i, (snap, s) in enumerate(states):
        from ...diagnostics import observation
        legal = observation(cfg, snap, s).legal
        cands = _candidate_set(scenario, cfg, legal)
        res = branch(cfg, snap, s, cands, (agent, agent), n_roll, roll_seed, key=(i,), cache=cache)
        out.append(_state_summary(res, cfg, agent, scenario))
    return key, {"policy": pol_name, "scenario": scenario, "states": out, "n_found": len(states),
                 "elapsed_s": time.perf_counter() - t0}


def counterfactual(workers: int = WORKERS, only=None):
    t0 = time.perf_counter()
    pols = policies()
    jobs = []
    for si, sc in enumerate(SCENARIOS):
        for pi_, n in enumerate(pols):
            if n in REPRESENTATIVE:
                ns = CF_STATES
            elif sc in GENERAL_SCENARIOS and n != "pub_sprand_s1":
                ns = CF_GENERAL_STATES
            else:
                continue
            if only and sc not in only:
                continue
            jobs.append(((n, sc), n, pols[n], sc, ns, CF_ROLLOUTS, CF_STATE_ROOT + 100 * si + pi_,
                         CF_ROLL_ROOT + 100 * si + pi_))
    jobs.sort(key=lambda j: -j[4])                                   # big jobs first
    parts = os.path.join(OUT, "counterfactual_parts")                # one file per finished job (resumable)
    os.makedirs(parts, exist_ok=True)
    res = {}
    todo = []
    for j in jobs:
        f = os.path.join(parts, f"{j[0][0]}__{j[0][1]}.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                res[j[0]] = json.load(fh)
        else:
            todo.append(j)
    print(f"counterfactual: {len(jobs)} (policy, scenario) jobs, {len(todo)} to run", flush=True)
    from concurrent.futures import as_completed
    from ...io import dump_json
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_cf_job, j) for j in todo]
        for fu in as_completed(futs):
            k, v = fu.result()
            dump_json(v, os.path.join(parts, f"{k[0]}__{k[1]}.json"))
            res[k] = v
            print(f"  {k}: {v['n_found']} states, {v['elapsed_s']:.0f} s", flush=True)
    data = {"rollouts": CF_ROLLOUTS, "state_root": CF_STATE_ROOT, "rollout_root": CF_ROLL_ROOT,
            "elapsed_s": time.perf_counter() - t0, "scenarios": SCENARIO_TEXT, "comparisons": COMPARISONS,
            "results": {"|".join(k): v for k, v in res.items()}}
    _write("counterfactual", data, counterfactual_md)
    return data


def _agg_pairs(states: List[dict], pair: str) -> dict:
    """Across the states of a scenario: mean paired difference (states are the unit), its 95% interval, the share
    of states with a significantly positive / negative difference (|d| > 1.96 SE of that state)."""
    vals = [(st["pairs"][pair][0], st["pairs"][pair][1]) for st in states if pair in st["pairs"]]
    if not vals:
        return {}
    d = np.array([v[0] for v in vals])
    se = np.array([v[1] for v in vals])
    h = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {"n": len(d), "mean": float(d.mean()), "ci": float(h), "pos": float(np.mean(d > 1.96 * se)),
            "neg": float(np.mean(d < -1.96 * se))}


def counterfactual_md(data: dict) -> str:
    R = data["results"]
    L = ["# Phase B: counterfactual action evaluation", "",
         f"Generated by `python -m lanegame.experiments.rl.diagnostics counterfactual` ({data['elapsed_s'] / 60:.1f} min). "
         f"For each sampled reachable state, every legal first action of the focal player (and a few short plans) is "
         f"committed in {data['rollouts']} paired rollouts (common random numbers: same chance, focal and opponent "
         "streams for every candidate; the opponent's branch-turn action is identical across candidates); afterwards "
         "both seats follow the SAME learned policy (it is the continuation policy and the fixed opponent). "
         "`Q` = mean focal return from the branch turn to the end of the game. `vs pi` = advantage over the policy's "
         "own mixture Σ_a π(a)·Q(a) at that state (positive = a one-step improvement over the policy against "
         "itself). Across states, the mean is over states (the unit of replication; its 95% interval includes "
         "the Monte Carlo noise); `+sig` / `-sig` = share of states whose own paired difference exceeds ±1.96 SE. "
         "These are Monte Carlo estimates for these continuation policies, not game-theoretic values.", ""]
    for sc in SCENARIOS:
        keys = [k for k in R if k.endswith("|" + sc)]
        if not keys:
            continue
        L += [f"## {sc}: {data['scenarios'][sc]}", ""]
        rows = []
        for k in keys:
            v = R[k]
            sts = v["states"]
            if not sts:
                rows.append([v["policy"], 0] + [""] * 6)
                continue
            for a, b in data["comparisons"][sc]:
                pair = f"{a} vs {b}"
                g = _agg_pairs(sts, pair)
                if not g:
                    continue
                pa = np.mean([st["pi"].get(a, 0.0) for st in sts])
                rows.append([v["policy"], g["n"], pair, f"{g['mean']:+.3f} ± {g['ci']:.3f}", f"{g['pos']:.2f}",
                             f"{g['neg']:.2f}", f"{pa:.3f}" if a in sts[0]["pi"] else "plan"])
        L += [md_table(["policy", "states", "comparison", "mean difference (95%)", "+sig", "-sig",
                        "policy P(first action)"], rows), ""]
        # best one-step deviation
        rows = []
        for k in keys:
            v = R[k]
            sts = v["states"]
            if not sts:
                continue
            ba = np.array([st["best_adv"][0] for st in sts])
            bse = np.array([st["best_adv"][1] for st in sts])
            sig = ba > 1.96 * bse
            labs = {}
            for st, sg in zip(sts, sig):
                if sg:
                    labs[st["best"]] = labs.get(st["best"], 0) + 1
            top = ", ".join(f"{a} ({c})" for a, c in sorted(labs.items(), key=lambda x: -x[1])[:4])
            vs = np.array([st["v_sampled_minus_piq"][0] for st in sts])
            rows.append([v["policy"], len(sts), f"{ba.mean():+.3f}", f"{sig.mean():.2f}",
                         f"{np.mean([st['pi_best'] for st in sts]):.3f}", top or "-", f"{vs.mean():+.3f}"])
        L += ["Best single committed action per state (vs the policy's mixture):", "",
              md_table(["policy", "states", "mean best advantage", "share significant", "policy P(best action)",
                        "significant best actions (count)", "check: sampled-policy candidate − Σπ·Q"], rows), ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- Phase A.3: HP value
HP_STATES, HP_ROLLOUTS = 30, 150
HP_VARIANTS = ("hp1", "hp2", "hp3", "hp4", "hp5", "hp6", "dead")


def _hp_job(args):
    """Paired rollouts from sampled mirror states (both champions present, turn 4-26) in which only the focal
    champion's record differs: HP 1..6 at its cell, or dead (absent for the next 3 turns, at its tower, i.e. the
    state right after a death at the end of the previous turn). Every variant is a reachable state (HP is the
    outcome of chance events; a death leaves the lane state unchanged). Both seats follow the policy."""
    key, pol_name, path, n_states, n_roll, state_seed, roll_seed = args
    import torch
    from ...diagnostics import FollowPolicy, ProbCache, Snapshot, rollout, rollout_seed
    from ...env import LaneEnv, play_episode
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    agent = _agent(path, cfg)
    env = LaneEnv(cfg, cache_size=20_000)
    rng = np.random.default_rng(state_seed)
    cache = ProbCache()
    out = []
    t0 = time.perf_counter()
    ep = 0
    while len(out) < n_states:
        log = play_episode(env, agent, agent, np.random.SeedSequence(entropy=state_seed, spawn_key=(ep,)), record=True)
        ep += 1
        cands = [(info, s) for info in log.turns for s in (0, 1)
                 if 4 <= info["k"] <= 26 and info["champ_before"][s][2] == 0 and info["champ_before"][1 - s][2] == 0]
        if not cands:
            continue
        info, s = cands[int(rng.integers(len(cands)))]
        base = Snapshot.from_info(info, (0.5, 0.5))
        variants = {f"hp{h}": base.with_side(s, hp=h) for h in range(1, 7)}
        variants["dead"] = base.with_side(s, pos=-cfg.tower, hp=0, absent=cfg.death_turns)
        rets = {}
        for name, snap in variants.items():
            rets[name] = np.array([rollout(env, snap, s, FollowPolicy(), (agent, agent),
                                           rollout_seed(roll_seed, m, len(out)), cache).ret for m in range(n_roll)])
        diff = {name: [float((rets[name] - rets["hp6"]).mean()),
                       float((rets[name] - rets["hp6"]).std(ddof=1) / np.sqrt(n_roll))] for name in rets}
        out.append({"k": base.k, "seat": s, "pos": base.champ.side(s).pos, "hp_actual": base.champ.side(s).hp,
                    "ehp": base.champ.side(1 - s).hp, "v": {n: float(r.mean()) for n, r in rets.items()},
                    "diff_vs_hp6": diff})
    return key, {"policy": pol_name, "states": out, "elapsed_s": time.perf_counter() - t0}


def hpvalue(workers: int = WORKERS):
    t0 = time.perf_counter()
    pols = policies()
    jobs = [((n,), n, pols[n], HP_STATES, HP_ROLLOUTS, HP_ROOT + i, HP_ROOT + 500 + i)
            for i, n in enumerate(REPRESENTATIVE)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(_hp_job, jobs))
    data = {"rollouts": HP_ROLLOUTS, "elapsed_s": time.perf_counter() - t0, "death_turns": 3, "kill_bounty": 0.5,
            "results": {k[0]: v for k, v in res.items()}}
    _write("hpvalue", data, hpvalue_md)
    return data


def hpvalue_md(data: dict) -> str:
    R = data["results"]
    L = ["# Phase A.3: what is HP worth?", "",
         f"Generated by `python -m lanegame.experiments.rl.diagnostics hpvalue` ({data['elapsed_s'] / 60:.1f} min). "
         f"From each sampled mirror state (both champions present, turn 4-26) the focal champion's record is set to "
         f"HP 1..6 at its cell, or to dead (absent the next 3 turns, at its tower); {data['rollouts']} paired rollouts "
         "per variant, both seats following the policy. Values: focal return from that turn to the end of the game, "
         "relative to the same state at full HP (6). A death also pays the opponent the 0.5 kill bounty on the death "
         "turn, which is NOT in these returns (it precedes the state): the full cost of dying from full HP is "
         "−(dead − hp6) + 0.5. Mean over states ± 95% interval (states are the unit).", ""]
    for name, v in R.items():
        sts = v["states"]
        rows = []
        for lo, hi, lab in ((4, 11, "turn 4-11"), (12, 19, "turn 12-19"), (20, 26, "turn 20-26"), (4, 26, "all")):
            ss = [s for s in sts if lo <= s["k"] <= hi]
            if not ss:
                continue
            row = [lab, len(ss)]
            for var in HP_VARIANTS[:-1] + ("dead",):
                d = np.array([s["diff_vs_hp6"][var][0] for s in ss])
                h = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
                row.append(f"{d.mean():+.2f} ± {h:.2f}")
            rows.append(row)
        L += [f"## {name}", "", md_table(["turns", "states"] + [f"{x} − hp6" for x in HP_VARIANTS], rows), ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- Phase C: rule overlays
OVERLAY_EPISODES = 1000


class RuleOverlay:
    """A frozen policy with one interpretable rule on top: where `rule(obs)` returns a legal action the overlay plays
    it (one-hot), otherwise the base policy's own distribution. Sampling: one uniform per turn (as `sample`)."""

    def __init__(self, base, rule, name: str):
        self.base, self.rule, self.name = base, rule, name

    def distribution(self, obs):
        a = self.rule(obs)
        if a is not None and a in obs.legal:
            p = np.zeros(len(obs.legal))
            p[obs.legal.index(a)] = 1.0
            return p
        return self.base.distribution(obs)

    def act(self, obs, rng):
        from ...env.agents import sample
        return obs.legal[sample(rng, self.distribution(obs))]


def overlay_rules() -> dict:
    """The rules tested in Phase C (each a function obs -> action or None)."""
    from ...champions import RECALL
    from ...config import B_HOLD, B_SAFE, F, PRESSURE, SH, SP, champ_act

    def me(obs):
        return obs.champ.side(obs.side)

    def enemy(obs):
        return obs.champ.side(1 - obs.side)

    def recall_low(obs, hp=2):
        if obs.legal == (45,) or me(obs).hp > hp:
            return None
        return RECALL if RECALL in obs.legal else champ_act(SP, B_SAFE)

    def punish(obs):
        return champ_act(SH, B_HOLD) if obs.legal != (45,) and enemy(obs).absent > 0 else None

    def pressure(obs):
        for a in (champ_act(SP, B_HOLD, PRESSURE), champ_act(SH, B_HOLD, PRESSURE)):
            if a in obs.legal:
                return a
        return None

    def freeze(obs):
        a = champ_act(F, B_HOLD)
        return a if a in obs.legal else None

    def shove_phase(obs):
        # Shove (Hold) in the 2-cycle phase where both fronts hold 3 HP at the lane with both champions present: the
        # phase in which one Shove kills the enemy front (Phase B, lane_contest)
        if obs.legal == (45,) or enemy(obs).absent > 0:
            return None
        own, en = obs.state.side(obs.side), obs.state.side(1 - obs.side)
        if own[1] == 3 and en[1] == 3 and own[0] == 0 and en[0] == 0:
            return champ_act(SH, B_HOLD)
        return None

    def shove_pressure(obs):
        # the siege plan the strongest Phase D attackers found: Shove (Hold) every turn, Hold + Apply Pressure
        # whenever an objective is legal
        if obs.legal == (45,):
            return None
        a = pressure(obs)
        return a if a is not None else champ_act(SH, B_HOLD)

    def late_shove(obs, k_from=24):
        # the horizon-end pattern of the Phase D attackers: from turn k_from on, Shove (Hold) and Apply Pressure when
        # legal; the policy plays unchanged before
        if obs.legal == (45,) or obs.k < k_from:
            return None
        a = pressure(obs)
        return a if a is not None else champ_act(SH, B_HOLD)

    def combo(obs):
        for r in (recall_low, pressure, punish):
            a = r(obs)
            if a is not None and a in obs.legal:
                return a
        return None

    return {"recall_hp2": recall_low, "recall_hp1": lambda o: recall_low(o, 1), "punish_shove": punish,
            "pressure": pressure, "freeze": freeze, "shove_phase": shove_phase,
            "shove_pressure": shove_pressure, "late_shove24": late_shove,
            "late_shove28": lambda o: late_shove(o, 28), "combo": combo}


def _overlay_job(args):
    key, path, rule_name, opp, episodes, seed = args
    import torch
    from ...rl import head_to_head
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    base = _agent(path, cfg)
    x = RuleOverlay(base, overlay_rules()[rule_name], f"{key[0]}+{rule_name}")
    y = base if opp == "self" else _agent(opp, cfg)
    r = head_to_head(cfg, x, y, episodes, seed)
    keep = ("mean", "stderr", "ci95", "win_rate", "seat_A", "seat_B")
    out = {k: r[k] for k in keep}
    xs = r["x_stats"]
    out["x_stats"] = {k: xs[k] for k in ("behaviors", "waves", "objectives", "recalls", "deaths", "farm", "plates",
                                          "damage_taken", "turns_away") if k in xs}
    return key, out


def overlay(workers: int = WORKERS, rules=None, name: str = "overlay"):
    t0 = time.perf_counter()
    pols = {k: v for k, v in policies().items() if k != "pub_sprand_s1"}
    jobs = []
    for n, p in pols.items():
        for rule in (rules or overlay_rules()):
            jobs.append(((n, rule, "self"), p, rule, "self", OVERLAY_EPISODES, OVERLAY_ROOT))
        for rule in (("combo",) if rules is None else ()):
            for opp in ("hold_farmer",):
                jobs.append(((n, rule, opp), p, rule, opp, OVERLAY_EPISODES // 2, OVERLAY_ROOT + 1))
    print(f"overlay: {len(jobs)} head-to-head jobs", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(_overlay_job, jobs))
    data = {"episodes": OVERLAY_EPISODES, "root": OVERLAY_ROOT, "elapsed_s": time.perf_counter() - t0,
            "rules": list(rules or overlay_rules()), "results": {"|".join(k): v for k, v in res.items()}}
    _write(name, data, overlay_md)
    return data


def overlay_md(data: dict) -> str:
    R = data["results"]
    rules = data.get("rules", [r for r in overlay_rules() if not r.startswith("late_")])   # first run had no "rules"
    L = ["# Phase C: rule overlays on the learned policies", "",
         f"Generated by `python -m lanegame.experiments.rl.diagnostics overlay` ({data['elapsed_s'] / 60:.1f} min). "
         f"Each learned policy plays itself unchanged except where one rule fires; head-to-head against the "
         f"unmodified policy, {data['episodes']} held-out episodes per seat (SeedSequence({data['root']})), both "
         "seats, equal skill 0.5. A positive payoff means the rule is a profitable deviation from the policy against "
         "itself. Rules: recall_hp2 / recall_hp1 = at HP <= 2 (<= 1) Recall if legal, else walk back with Safe; "
         "punish_shove = Shove (Hold) while the enemy champion is away; pressure = Hold + Apply Pressure whenever an "
         "objective is legal; freeze = Freeze (Hold) whenever legal; shove_phase = Shove (Hold) whenever both fronts hold 3 "
         "HP at the lane cell and both champions are present; shove_pressure = Shove (Hold) on every present turn, "
         "Hold + Apply Pressure whenever legal (the plan of the strongest Phase D attackers); late_shove24 / late_shove28 = shove_pressure from turn 24 (28) "
         "on only (added after the attacker audit: the attackers shove mostly late); combo = recall_hp2, then pressure, then "
         "punish_shove. Per-policy ± is a 95% interval over episodes; family ± is a 95% t interval over the 3 "
         "training seeds.", ""]
    rows = []
    for c in CONDITIONS:
        row = [LABEL[c]]
        for rule in rules:
            vals = [R[f"{c}_s{s}|{rule}|self"]["mean"] for s in SEEDS]
            row.append(fmt_seed_ci(vals) + " (" + ", ".join(f"{v:+.2f}" for v in vals) + ")")
        rows.append(row)
    L += ["## Payoff of policy+rule against the policy", "", md_table(["family"] + rules, rows), ""]
    rows = []
    for c in CONDITIONS:
        for s in SEEDS:
            n = f"{c}_s{s}"
            rows.append([n] + [f"{R[f'{n}|{r}|self']['mean']:+.2f} ± {1.96 * R[f'{n}|{r}|self']['stderr']:.2f}"
                               for r in rules])
    L += ["## Per policy", "", md_table(["policy"] + rules, rows), ""]
    rows = []
    for c in CONDITIONS:
        for s in SEEDS:
            n = f"{c}_s{s}"
            for r in rules:
                xs = R[f"{n}|{r}|self"]["x_stats"]
                rows.append([n, r, f"{xs.get('recalls', 0):.2f}", f"{xs.get('deaths', 0):.2f}",
                             f"{xs.get('turns_away', 0):.2f}", f"{xs.get('farm', 0):.2f}", f"{xs.get('plates', 0):.2f}",
                             f"{xs['waves'].get('SH', 0):.3f}", f"{xs['objectives'].get('Pressure', 0):.3f}"])
    L += ["## What the overlay changed (per game: recalls, deaths, turns away, farm, plates; Shove and Pressure "
          "shares of turns)", "",
          md_table(["policy", "rule", "recalls", "deaths", "turns away", "farm", "plates", "Shove", "Pressure"], rows), ""]
    rows = []
    for c in CONDITIONS if "combo" in rules else ():
        vals = [R[f"{c}_s{s}|combo|hold_farmer"]["mean"] for s in SEEDS]
        rows.append([LABEL[c], fmt_seed_ci(vals)])
    L += ["## combo overlay against hold_farmer (policy alone: see audit.md economy table)", "",
          md_table(["family", "payoff"], rows), ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- training-time view
SNAP_AUDIT_EPISODES = 50


def _snap_audit_job(args):
    key, path, episodes, seed = args
    import torch
    from ...diagnostics import audit_games, legality_table, opportunity_table
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    x = _agent(path, cfg)
    T = audit_games(cfg, x, x, episodes, seed)
    pres = T.where(T["present"])
    return key, {"legality": legality_table(T), "opportunities": opportunity_table(T),
                 "entropy": float(pres["entropy"].mean()), "deaths": float(T["died"].sum() / (2 * episodes))}


def snapshot_audit(workers: int = WORKERS):
    """How the probabilities of the rare actions evolved during training (mirror games of every snapshot of the
    representative runs): were they abandoned early, before the situations that make them useful were visited?"""
    from .openai_five import snapshots_of
    t0 = time.perf_counter()
    jobs = []
    for n in REPRESENTATIVE:
        c, s = n.rsplit("_s", 1)
        for t, p in snapshots_of(run_dir("main", c, int(s))):
            jobs.append(((n, t), p, SNAP_AUDIT_EPISODES, AUDIT_ROOT + 2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(_snap_audit_job, jobs))
    data = {"elapsed_s": time.perf_counter() - t0, "results": {f"{k[0]}|{k[1]}": v for k, v in res.items()}}
    _write("snapshot_audit", data, snapshot_audit_md)
    return data


def snapshot_audit_md(data: dict) -> str:
    R = data["results"]
    L = ["# Rare actions during training (mirror games of each snapshot)", "",
         f"Generated by `python -m lanegame.experiments.rl.diagnostics snapshot_audit` ({data['elapsed_s'] / 60:.1f} min); "
         f"{SNAP_AUDIT_EPISODES} episodes per seat per snapshot. p = the snapshot's mean probability on its present "
         "turns (unconditional) or in the stated situation.", ""]
    for n in REPRESENTATIVE:
        keys = sorted([k for k in R if k.startswith(n + "|")], key=lambda k: int(k.split("|")[1]))
        rows = []
        for k in keys:
            v = R[k]
            lg, op = v["legality"], v["opportunities"]
            rows.append([f"{int(k.split('|')[1]):,}", f"{v['entropy']:.2f}", f"{lg['Trade']['p_when_legal']:.2f}",
                         f"{lg['Yield']['p_when_legal']:.3f}", f"{lg['Safe']['p_when_legal']:.3f}",
                         f"{lg['SH']['p_when_legal']:.3f}",
                         _f(op['recall legal and HP <= 2']['p']['Recall'], "{:.3f}"),
                         _f(op['enemy champion away (punish window)']['p']['SH'], "{:.3f}"),
                         f"{lg['F']['legal_share']:.3f}", f"{lg['Pressure']['legal_share']:.4f}", f"{v['deaths']:.2f}"])
        L += [f"## {n}", "", md_table(["agent transitions", "entropy", "p Trade", "p Yield", "p Safe", "p Shove",
                                       "p Recall given legal, HP <= 2", "p Shove given enemy away", "Freeze legal share",
                                       "objective legal share", "deaths / game"], rows), ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- Phase C: critic check
def calibration():
    """The policy's own value head V_theta(o) against the Monte Carlo value sum_a pi(a) Q(a) of the counterfactual
    states (same policy as continuation and opponent): does the critic know these situations?"""
    import torch
    from ...diagnostics import Snapshot, observation
    from ...rl import load_checkpoint
    cfg = GameConfig.champions()
    cf = _load("counterfactual")
    pols = policies()
    nets = {}
    rows_out = {}
    for key, v in cf["results"].items():
        n, sc = key.split("|")
        if n not in nets:
            nets[n] = load_checkpoint(pols[n])[0].eval()
        vs, mc = [], []
        for st in v["states"]:
            snap = Snapshot.from_dict(st["snapshot"])
            ob = observation(cfg, snap, st["seat"])
            with torch.no_grad():
                vs.append(float(nets[n].value(torch.as_tensor(ob.vector).unsqueeze(0))[0]))
            mc.append(sum(p * st["q"][a][0] for a, p in st["pi"].items()))
        vs, mc = np.array(vs), np.array(mc)
        if len(vs) < 3:
            continue
        rows_out[key] = {"n": len(vs), "v_mean": float(vs.mean()), "mc_mean": float(mc.mean()),
                         "bias": float((vs - mc).mean()), "rmse": float(np.sqrt(((vs - mc) ** 2).mean())),
                         "corr": float(np.corrcoef(vs, mc)[0, 1]) if vs.std() > 0 and mc.std() > 0 else float("nan"),
                         "mc_sd": float(mc.std(ddof=1))}
    _write("calibration", {"results": rows_out}, calibration_md)


def calibration_md(data: dict) -> str:
    R = data["results"]
    rows = [[k.split("|")[1], k.split("|")[0], v["n"], f"{v['v_mean']:+.2f}", f"{v['mc_mean']:+.2f}",
             f"{v['bias']:+.2f}", f"{v['rmse']:.2f}", f"{v['mc_sd']:.2f}", _f(v["corr"])]
            for k, v in sorted(R.items(), key=lambda kv: (kv[0].split("|")[1], kv[0].split("|")[0]))]
    return "\n".join(["# Critic calibration at the counterfactual states", "",
                      "Generated by `python -m lanegame.experiments.rl.diagnostics calibration`. V = the policy's value "
                      "head at the focal seat's observation; MC = Σ_a π(a)·Q(a) from the paired rollouts (Monte Carlo "
                      "standard error ≈ 0.1 per state). A critic that knows a situation has small bias and RMSE well "
                      "below the spread of MC values across states.", "",
                      md_table(["scenario", "policy", "states", "mean V", "mean MC", "bias V − MC", "RMSE",
                                "SD of MC across states", "corr(V, MC)"], rows), ""])


# ---------------------------------------------------------------------------------------------- stage dispatch
def overlay_late(workers: int = WORKERS):
    """Phase C follow-up: the late-game shove rules suggested by the attacker audit."""
    return overlay(workers, rules=("late_shove24", "late_shove28"), name="overlay_late")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "audit"
    globals()[stage]()
