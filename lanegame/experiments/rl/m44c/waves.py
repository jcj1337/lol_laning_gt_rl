"""Part 6: are Last Hit, Thin and Slow Push economically distinct choices?

Mechanics (reference §4.10; mechanics.strength_budget, M4Context.state_budget). With the champion's capacity C and the
start-of-turn window strengths S_own, S_enemy, the per-turn budgets are

    D_LH = 1,   D_TN = min(round(C/2), 1 + need_0),   D_SL = min(round(C/2), 1 + need_1),   D_SH = C,
    need_m = max(0, ceil(S_enemy - S_own + m)),

spent half per tick on the first reachable enemy target, overkill lost. So D_TN = D_LH whenever S_own >= S_enemy, and
D_SL = D_LH whenever S_own >= S_enemy + 1: in those states the three actions are the SAME action. Where the budgets
differ, the lane can still end identically (nothing in reach, or the enemy front dies to either budget).

Experiments
    waves_mech (no rollouts)
        budgets   the four budgets at natural decision states of each final (random present turns of self-play)
        one-turn  the successor lane of the focal side's Last Hit / Thin / Slow Push / Shove, with its recorded
                  behaviour and objective and the opponent's recorded action (the lane step has no chance: minion
                  movement and damage are deterministic given the actions), compared for identity
        matrix    deterministic 16-turn games from the `neutral` start (levels 3 and 10): every pair of fixed wave
                  actions, both champions Hold, no chance events
    waves (paired rollouts, NATURAL states with the fronts in contact and both champions present)
        focal plans for 8 turns (wave forced, other components the policy's own conditional choice): Last Hit x8,
        Thin x8, Slow Push x8, Slow Push until the own front has 16 HP (at most 6 turns) then one Shove, Shove x8,
        and the policy; opponents: wave forced to Last Hit / Thin / Slow Push / Shove for the same 8 turns (an
        opponent plan), or its policy. Wave position, window-strength lead and crashes after 2, 4, 8, 16 turns, the
        ledgers, and the discounted return over 300 turns
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from ....champions_m4 import context, resolve_turn_m4
from ....config import B_HOLD, CHAMP_ABSENT, W_HOLD, W_SHOVE, W_SLOWPUSH, W_THIN, champ_act, champ_behavior, champ_objective
from ....diagnostics.branch import FollowPolicy, observation
from ....mechanics import start_packets, window_strength
from ....state import State
from ..m44.actions import RECALL, full_distribution, masses
from ..m44.records import play_record, wave_position
from . import opponents as O
from . import plans as P
from .common import MECH_SEED, OFFSET, PRIMARY, STATE_SEED, load, meta, policies, run_jobs, save_json, weights_digest
from .runner import Experiment
from .states import collect, initialized

SLOTS = {"LH": W_HOLD, "TN": W_THIN, "SL": W_SLOWPUSH, "SH": W_SHOVE}
SEQ_TURNS = 8
BUILD_HP = 16


# ---------------------------------------------------------------------------------------------- mechanics
def budgets(cfg, lane, m4, k: int, s: int) -> Dict[str, int]:
    ctx = context(cfg)
    t = ctx.t
    me = m4.side(s)
    C = ctx.capacity(t.level_by_xp[min(me.xp, t.xp_cap)], me.items)
    return {n: int(ctx.state_budget(w, C, lane, k, s)) for n, w in SLOTS.items()} | {"C": int(C)}


def one_turn(cfg, lane, m4, k: int, s: int, a_focal: int, a_opp: int) -> Dict[str, dict]:
    """Successor lane and focal champion damage of each focal wave slot, keeping a_focal's behaviour / objective."""
    b, o = champ_behavior(a_focal), champ_objective(a_focal)
    out = {}
    for n, w in SLOTS.items():
        a = champ_act(w, b, o)
        acts = (a, a_opp) if s == 0 else (a_opp, a)
        r = resolve_turn_m4(lane, m4, k, acts[0], acts[1], cfg, u=None)
        out[n] = {"lane": r.lane.state, "removed": int(r.lane.events["champion_hp"][1 - s]) if "champion_hp" in
                  r.lane.events else 0, "farm": float(r.farm[s]), "crash": int(r.lane.crash_hp[s])}
    return out


def mech_job(args) -> dict:
    env, seed, phase, n_games, turns = args
    cfg, x = load(env, seed, phase)
    rng = np.random.default_rng(MECH_SEED + OFFSET["waves"] + 100 * seed)
    rows = []
    for g, ss in enumerate(np.random.SeedSequence(MECH_SEED + OFFSET["waves"] + 100 * seed).spawn(n_games)):
        rec = play_record(cfg, (x, x), ss, turns, keep_snapshots=True)
        d = rec.data
        for t in rng.choice(np.arange(8, rec.turns), size=min(40, rec.turns - 8), replace=False):
            for s in (0, 1):
                a, ao = int(d["a"][t, s]), int(d["a"][t, 1 - s])
                if a in (RECALL, CHAMP_ABSENT) or d["present"][t, s] == 0:
                    continue
                snap = rec.snapshots[t]
                bud = budgets(cfg, snap.lane, snap.champ, snap.k, s)
                tr = one_turn(cfg, snap.lane, snap.champ, snap.k, s, a, ao)
                rows.append({"wave_adv": float(d["wave_adv"][t, s]), "contact": float(d["contact"][t, s]),
                             "wave_pos": float(d["wave_pos"][t, s]), "held": float(d["held"][t, s]),
                             "held_on": float(d["held_on"][t, s]), "level": float(d["level"][t, s]),
                             "budget": bud, "pw": [float(d[f"pw{i}"][t, s]) for i in range(4)],
                             "same": {p: tr[p[:2]]["lane"] == tr[p[3:]]["lane"] for p in
                                      ("LH-TN", "TN-SL", "LH-SL", "SL-SH", "TN-SH", "LH-SH")},
                             "removed": {n: tr[n]["removed"] for n in SLOTS}, "farm": {n: tr[n]["farm"] for n in SLOTS},
                             "crash": {n: tr[n]["crash"] for n in SLOTS}})
    return {"env": env, "seed": seed, "phase": phase, "rows": rows}


def matrix(cfg, level: int, turns: int = 16) -> Dict[str, dict]:
    """Deterministic games from `neutral` at `level`: fixed wave pairs, both Hold, no chance (u = None)."""
    snap = initialized(cfg, "neutral", level=level)
    out = {}
    for na, wa in SLOTS.items():
        for nb, wb in SLOTS.items():
            lane, m4, k = snap.lane, snap.champ, snap.k
            gold, xp, crash = np.zeros(2), np.zeros(2), np.zeros(2)
            pos = []
            for i in range(turns):
                r = resolve_turn_m4(lane, m4, k, champ_act(wa, B_HOLD), champ_act(wb, B_HOLD), cfg, u=None)
                gold += np.array(r.earned_units) / cfg.gold_scale
                xp += np.array(r.xp_gain)
                crash += np.array([r.lane.crash_hp[0] > 0, r.lane.crash_hp[1] > 0])
                lane, m4, k = r.lane.state, r.champ, k + 1
                pos.append(wave_position(lane, 0))
                if i == 7:
                    g8, x8 = gold.copy(), xp.copy()
            out[f"{na}|{nb}"] = {"gold_diff8": float(g8[0] - g8[1]), "gold_diff16": float(gold[0] - gold[1]),
                                 "xp_diff16": float(xp[0] - xp[1]), "gold": gold.tolist(), "crashes": crash.tolist(),
                                 "wave_pos_mean": float(np.nanmean(pos)), "wave_pos_end": float(pos[-1])}
    return out


def run_mechanics(envs=None, seeds=None, workers: int = 11, n_games: int = 6, turns: int = 500) -> dict:
    t0 = time.perf_counter()
    from .common import ENV_ORDER, SEEDS, env_cfg
    jobs = [(e, s, p, n_games, turns) for e, s, p in policies(envs or ENV_ORDER, seeds or SEEDS, (PRIMARY,))]
    res = run_jobs(mech_job, jobs, workers, "waves_mech")
    mats = {e: {f"L{L}": matrix(env_cfg(e), L) for L in (3, 10)} for e in (envs or ENV_ORDER)}
    data = {"meta": meta(stage="waves_mech", n_games=n_games, turns=turns, seed=MECH_SEED + OFFSET["waves"]),
            "cells": {f"{r['phase']}|{r['env']}|{r['seed']}": r for r in res}, "matrix": mats,
            "elapsed_s": time.perf_counter() - t0}
    save_json("waves_mech", data)
    return data


# ---------------------------------------------------------------------------------------------- sequences
def build_rule(obs, t, mem):
    if mem.get("shoved"):
        return P.stop("built and shoved")
    own_fh = obs.state.side(obs.side)[1]
    if own_fh >= BUILD_HP or t >= 6:
        mem["shoved"] = True
        return P.force({"wave": W_SHOVE}, "Shove (built)")
    return P.force({"wave": W_SLOWPUSH}, "Slow Push (build)")


def candidates(situation, cfg, snap, seat) -> Dict[str, object]:
    def seq(name, slot):
        return P.StrategicPlan(name, P.const_rule(P.force({"wave": slot}, name)), max_turns=SEQ_TURNS,
                               on_absent="continue")
    return {"policy": FollowPolicy(), "LHx8": seq("LHx8", W_HOLD), "TNx8": seq("TNx8", W_THIN),
            "SLx8": seq("SLx8", W_SLOWPUSH),
            "SL>SH": P.StrategicPlan("SL>SH", build_rule, max_turns=SEQ_TURNS, on_absent="continue"),
            "SHx8": seq("SHx8", W_SHOVE)}


def opp_plans(situation):
    return {f"opp_{n}": (lambda w=w, n=n: O.wave_plan(w, SEQ_TURNS, f"opp_{n}")) for n, w in SLOTS.items()}


def contact_pred(d, t, s):
    return d["contact"][t, s] > 0 and d["en_present"][t, s] > 0 and d["held"][t, s] == 0 and d["held_on"][t, s] == 0


def states(cfg, agent, env, seed, phase) -> Dict[str, List[dict]]:
    return collect(cfg, agent, {"contact": contact_pred}, {"contact": 24}, STATE_SEED + OFFSET["waves"] + 100 * seed +
                   (0 if phase == PRIMARY else 50), game_turns=500, max_games=60, per_game=1)


def frequency(agent, cfg, snap, s) -> dict:
    m = masses(full_distribution(agent, observation(cfg, snap, s)))
    names = ("Last Hit", "Thin", "Slow Push", "Shove")
    out = {n: float(m["wave"][i]) for i, n in enumerate(names)}
    out.update(budgets(cfg, snap.lane, snap.champ, snap.k, s))
    P_ = start_packets(snap.lane, cfg)
    out["wave_adv"] = float(window_strength(P_[s], snap.k, cfg) - window_strength(P_[1 - s], snap.k, cfg))
    return out


def extra(cfg):
    def fn(infos, f):
        out = {}
        for h in (2, 4, 8, 16):
            if h > len(infos):
                continue
            inf = infos[h - 1]
            lane = State.from_list(inf["state_after"])
            Pk = start_packets(lane, cfg)
            k1 = int(inf["k"]) + 1
            out[f"wpos{h}"] = wave_position(lane, f)
            out[f"wadv{h}"] = float(window_strength(Pk[f], k1, cfg) - window_strength(Pk[1 - f], k1, cfg))
            out[f"ownfh{h}"] = float(lane.side(f)[1])
        return out
    return fn


def experiment(name: str) -> Experiment:
    if name != "waves":
        raise ValueError(name)
    return Experiment(name=name, offset="waves", sizes={"contact": 24},
                      protocols={"contact": ("policy", "opp_LH", "opp_TN", "opp_SL", "opp_SH")}, states=states,
                      candidates=candidates, frequency=frequency, n_roll={"contact": 16}, ep_h=(2, 4, 8, 16),
                      extra=extra, opp_plans=opp_plans, notes={"seq_turns": SEQ_TURNS, "build_hp": BUILD_HP})


def run(envs=None, seeds=None, workers: int = 11):
    from .runner import run as _run
    return _run("waves", "waves", envs, seeds, (PRIMARY,), workers)


__all__ = ["budgets", "one_turn", "matrix", "run_mechanics", "experiment", "run", "candidates", "SLOTS"]
