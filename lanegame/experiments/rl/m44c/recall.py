"""Part 2: interrupted Recalls. Does the policy repeatedly choose an economically inferior Recall?

Mechanics (champions_m4.resolve_turn_m4 steps 2-4; verified on the engine, docs/m4/m4_4c_strategic_investigation.md
§3.1). Recall (code 12) is legal at the own -1 or -2. It is interrupted iff the enemy chooses Trade and its end cell is
in contact with the recaller's cell (the trader chases it); the interrupted champion channels (no farm, no wave
damage), is traded with the full hit probability 0.55 (no Safe / Yield reduction) and plays normally next turn. At the
own tower cell (-2) only a dive (the enemy ends in that cell, taking the tower shot) reaches it. Safe at -1 against a
Trade evades to -2 untouched (or is hit with 0.05 if the trader already stands in that cell); Safe against no Trade
stays at -1 (no behaviour walks into the own tower cell by itself). One turn is therefore a simultaneous guessing game:
Recall pays against no Trade, Safe against a Trade; the opponent's Trade probability decides.

States (natural self-play of each final, decision state only):
    base        present, Recall legal, HP <= 40% of max, the policy's Recall mass >= 0.20
    contested   base, and an enemy Trade could end in contact with the champion's cell this turn (interruption possible)
    repeated    contested, and the champion's previous turn was an interrupted Recall
    retreat     contested, and the champion stands at its own -1 (a Safe step back to the tower cell is possible); the
                natural contested sample is dominated by champions already in their tower cell (-2), so this stratum
                is sampled separately
    safe        base, and no enemy Trade can reach it this turn (enemy away or out of reach)
Covariates recorded per state: position (retreat possible from -1; already in the tower cell at -2), whether Safe would
evade an enemy Trade this turn (one-turn resolution without chance), the opponent's Trade mass, HP of both, power,
duel edge, unspent gold, level.

Candidates (complete legal actions; Safe / Hold keep the policy's own conditional wave choice, `plans.force`):
    policy      the learned policy (no intervention)
    Recall      Recall on the branch turn, then the policy
    Safe        Safe on the branch turn, then the policy
    Hold        Hold on the branch turn, then the policy
    Safe>Recall Safe once, Recall on the next turn (if legal; else the policy), then the policy
    Retreat     Safe until the champion stands in its tower cell or no enemy Trade can reach it, then Recall; at most
                4 turns, ends when the champion leaves the lane (the state-dependent retreat-then-Recall sequence)
Opponent protocols: "policy" (it samples its own policy) and "modal" (its most probable branch-turn action).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from ....champions_m4 import contact_cell_target, resolve_turn_m4
from ....config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, W_HOLD, champ_act
from ....diagnostics.branch import Commit, FollowPolicy, ProbCache, observation
from ..m44.actions import RECALL
from . import engine as E
from . import plans as P
from .common import (OFFSET, PRIMARY, ROLL_SEED, STATE_SEED, clear_partial, load, meta, policies, run_resumable,
                     save_json, weights_digest)
from .states import collect, to_json

NAME = "recall"
LOW_HP = 0.40
MIN_RECALL_MASS = 0.20
SIZES = {"contested": 32, "repeated": 24, "retreat": 24, "safe": 16}
CFG = dict(n_roll=24, window=300, game_turns=500, max_games=120, per_game=3, chunk=8)
PROTOCOLS = {"contested": ("policy", "modal"), "repeated": ("policy", "modal"), "retreat": ("policy", "modal"),
             "safe": ("policy",)}


# ---------------------------------------------------------------------------------------------- situation
def reach_cell(cfg, my_pos: int, enemy_pos: int) -> bool:
    """An enemy Trade this turn can end in contact with a champion that stays at my_pos (Recall channels in place)."""
    cell = contact_cell_target(int(enemy_pos), int(my_pos), cfg)
    return abs(cell - int(enemy_pos)) <= cfg.champion_speed


def _reach_from_record(cfg, d, t, s) -> bool:
    if d["en_present"][t, s] == 0:
        return False
    return reach_cell(cfg, int(d["pos"][t, s]), int(-d["epos"][t, s]))


def predicates(cfg) -> Dict[str, callable]:
    def base(d, t, s):
        return (d["recall_legal"][t, s] > 0 and d["hp_frac"][t, s] <= LOW_HP and d["prec"][t, s] >= MIN_RECALL_MASS)

    def contested(d, t, s):
        return base(d, t, s) and _reach_from_record(cfg, d, t, s)

    def repeated(d, t, s):
        return (contested(d, t, s) and t > 0 and d["beh"][t - 1, s] == B_RECALL and d["recalled"][t - 1, s] == 0
                and d["present"][t - 1, s] > 0)

    def retreat(d, t, s):
        return contested(d, t, s) and d["pos"][t, s] == -cfg.tower + 1

    def safe(d, t, s):
        return base(d, t, s) and not _reach_from_record(cfg, d, t, s)
    return {"contested": contested, "repeated": repeated, "retreat": retreat, "safe": safe}


def describe(snap, s: int, cfg) -> dict:
    """One-turn resolution without chance of (my Safe / Recall / Hold) against an enemy Trade: does Safe evade?"""
    lane, m4, k = snap.lane, snap.champ, snap.k
    me, en = m4.side(s), m4.side(1 - s)
    out = {"pos": me.pos, "enemy_pos": en.pos, "enemy_absent": en.absent > 0,
           "reach": en.absent == 0 and reach_cell(cfg, me.pos, en.pos), "gold": me.gold / cfg.gold_scale}
    if en.absent == 0:
        trade = champ_act(W_HOLD, B_TRADE)
        for name, a in (("safe", champ_act(W_HOLD, B_SAFE)), ("recall", RECALL), ("hold", champ_act(W_HOLD, B_HOLD))):
            acts = (a, trade) if s == 0 else (trade, a)
            t = resolve_turn_m4(lane, m4, k, acts[0], acts[1], snap.config(cfg), u=None)
            out[f"{name}_vs_trade_connected"] = bool(t.connected[1 - s])
            out[f"{name}_vs_trade_hit_p"] = float(t.hit_p[1 - s])
    return out


# ---------------------------------------------------------------------------------------------- candidates
def _can_reach(obs) -> bool:
    me, en = obs.champ.side(obs.side), obs.champ.side(1 - obs.side)
    return en.absent == 0 and reach_cell(obs.cfg, me.pos, en.pos)


def retreat_rule(obs, t, mem):
    me = obs.champ.side(obs.side)
    if RECALL in obs.legal and (me.pos == -obs.cfg.tower or not _can_reach(obs)):
        return P.code(RECALL, "Recall (tower cell or out of reach)")
    return P.force({"behavior": B_SAFE}, "Safe (retreat)")


def safe_then_recall(obs, t, mem):
    return P.SAFE_D if t == 0 else P.code(RECALL, "Recall")


def candidates() -> Dict[str, object]:
    return {"policy": FollowPolicy(),
            "Recall": P.one_turn("Recall", P.RECALL_D),
            "Safe": P.one_turn("Safe", P.SAFE_D),
            "Hold": P.one_turn("Hold", P.HOLD_D),
            "Safe>Recall": P.StrategicPlan("Safe>Recall", safe_then_recall, max_turns=2,
                                           description="Safe on the branch turn, Recall on the next"),
            "Retreat": P.StrategicPlan("Retreat", retreat_rule, max_turns=4,
                                       description="Safe until in the tower cell or out of reach, then Recall")}


def _extra_factory(cfg):
    """Per rollout: HP lost before leaving the lane, how and when it left, interruptions before leaving."""
    from ....champions_m4 import context
    hpmax = context(cfg).t.hp_max

    def fn(infos, f):
        dmg = intr = 0.0
        how, when = "stayed", float("nan")
        for t, inf in enumerate(infos[:40]):
            dmg += inf["damage_taken"][f] / hpmax[inf["levels_before"][f]]
            intr += float(inf["recall_interrupted"][f])
            if inf["recalled"][f] or inf["died"][f]:
                how, when = ("recall" if inf["recalled"][f] else "death"), float(t)
                break
        return {"dmg_before_leave": dmg, "left_by_recall": float(how == "recall"),
                "left_by_death": float(how == "death"), "left_turn": when, "interrupts_before_leave": intr,
                "left_within_3": float(how != "stayed" and when <= 2), "stayed_40": float(how == "stayed")}
    return fn


def frequency(agent, cfg, snap, s: int) -> dict:
    from ..m44.actions import full_distribution, masses
    ob = observation(cfg, snap, s)
    m = masses(full_distribution(agent, ob))
    oo = observation(cfg, snap, 1 - s)
    mo = masses(full_distribution(agent, oo))
    return {"Recall": float(m["recall"]), "Safe": float(m["beh"][B_SAFE]), "Hold": float(m["beh"][B_HOLD]),
            "Trade": float(m["beh"][B_TRADE]), "Yield": float(m["beh"][1]), "opp_Trade": float(mo["beh"][B_TRADE]),
            "opp_Recall": float(mo["recall"])}


# ---------------------------------------------------------------------------------------------- jobs
def collect_states(env: str, seed: int, phase: str, sizes: Dict[str, int] = None) -> Dict[str, List[dict]]:
    cfg, x = load(env, seed, phase)
    sizes = sizes or SIZES
    preds = predicates(cfg)
    return collect(cfg, x, {k: preds[k] for k in sizes}, sizes, STATE_SEED + OFFSET[NAME] + 100 * seed +
                   (0 if phase == PRIMARY else 50), CFG["game_turns"], CFG["max_games"], CFG["per_game"],
                   describe=describe)


def job(args) -> dict:
    """One (policy, situation, protocol, chunk of states)."""
    env, seed, phase, situation, protocol, lo, hi, n_roll, sizes = args
    cfg, x = load(env, seed, phase)
    d0 = weights_digest(x)
    found = collect_states(env, seed, phase, sizes)[situation][lo:hi]
    cache = ProbCache()
    out = []
    ex = _extra_factory(cfg)
    for i, st in enumerate(found):
        snap, s = st["snapshot"], st["seat"]
        ob = observation(cfg, snap, s)
        cands = {k: v for k, v in candidates().items() if not hasattr(v, "applicable") or v.applicable(ob)}
        opp_plan = None
        if protocol == "modal":
            oo = observation(cfg, snap, 1 - s)
            po = np.asarray(x.distribution(oo), dtype=float)
            opp_plan = Commit(int(oo.legal[int(np.argmax(po))]), "opponent modal")
        root = ROLL_SEED + OFFSET[NAME]
        res = E.evaluate(cfg, snap, s, cands, (x, x), n_roll, root, key=(seed, lo + i, ["policy", "modal"].index(protocol)),
                         window=CFG["window"], opponent_plan=opp_plan, extra=ex, cache=cache, ep_h=(3, 10, 40))
        out.append({**to_json(st), "index": lo + i, "frequency": frequency(x, cfg, snap, s), "result": res})
    return {"env": env, "seed": seed, "phase": phase, "situation": situation, "protocol": protocol, "lo": lo,
            "states": out, "digest_before": d0, "digest_after": weights_digest(x)}


def run(envs=None, seeds=None, phases=(PRIMARY,), workers: int = 11, sizes: Optional[Dict[str, int]] = None,
        n_roll: Optional[int] = None, name: str = NAME, protocols=None) -> dict:
    t0 = time.perf_counter()
    sizes = sizes or SIZES
    n_roll = n_roll or CFG["n_roll"]
    from .common import ENV_ORDER, SEEDS
    jobs = []
    for env, seed, phase in policies(envs or ENV_ORDER, seeds or SEEDS, phases):
        for situation, n in sizes.items():
            for protocol in (protocols or PROTOCOLS)[situation]:
                for lo in range(0, n, CFG["chunk"]):
                    jobs.append((env, seed, phase, situation, protocol, lo, min(n, lo + CFG["chunk"]), n_roll, sizes))
    res = run_resumable(job, jobs, workers, name)
    cells: Dict[str, dict] = {}
    for r in res:
        k = f"{r['phase']}|{r['env']}|{r['seed']}|{r['situation']}|{r['protocol']}"
        c = cells.setdefault(k, {"env": r["env"], "seed": r["seed"], "phase": r["phase"], "situation": r["situation"],
                                 "protocol": r["protocol"], "states": [], "digests_unchanged": True})
        c["states"] += r["states"]
        c["digests_unchanged"] &= r["digest_before"] == r["digest_after"]
    for c in cells.values():
        c["states"].sort(key=lambda st: st["index"])
    cfg_meta = {**CFG, "n_roll": n_roll}
    data = {"meta": meta(stage=name, sizes=sizes, **cfg_meta, low_hp=LOW_HP, min_recall_mass=MIN_RECALL_MASS,
                         state_seed=STATE_SEED + OFFSET[NAME], roll_seed=ROLL_SEED + OFFSET[NAME], jobs=len(jobs)),
            "cells": cells, "elapsed_s": time.perf_counter() - t0}
    save_json(name, data)
    clear_partial(name)
    return data


__all__ = ["run", "job", "candidates", "predicates", "describe", "collect_states", "reach_cell", "SIZES", "CFG"]
