"""The strategic-evaluation framework of the full M4.5c campaign (PREPARED, not run at scale by the pilot): paired
multi-turn counterfactuals of LEARNED policies on the M4.5b engine (`m45b.engine.collect` / `branch`, the environment's
resolver and chance stream, common random numbers across candidates).

A family is (a decision-state predicate, a set of candidate plans, a horizon). For each final network:
  1. natural states: self-play games of the network from s0 (seeds of the family's state range); every (turn, seat)
     whose START-OF-TURN state satisfies the predicate is a candidate state (at most `per_game` per game, seeded
     shuffle; seat B states mirrored to seat A);
  2. from each state, every candidate plan plays seat A for its plan turns and then the network itself
     (`engine.Then`), against the network in seat B (`m45b.reference.NetPolicy`, k0 = the state's turn), on the same
     rollout seeds;
  3. the quantity is the paired difference of the discounted lead G_h (h = 20, 50, 100) and of ledger entries
     between a plan and "policy" (the network's own continuation), averaged over rollouts, then over states; the
     interval is over states (states are the units; rollouts are nested).

Families (the brief's questions A-E; each answers "is the policy's choice economically appropriate in THIS state",
never "how often is X chosen"):

    A wave       fronts in contact, both present           LH / TN / SL / SH (+ Hold) for 4 turns
    B freeze     the focal side holds a freeze             keep (Thin + Hold, 6) / guard (Thin + Trade, 6) / break (Shove, 2)
    B frozen     the focal side is frozen on               Shove (2) / step back (Last Hit + Safe, 4) / Recall (1)
    C trade      enemy catchable without a dive, enemy      Trade (Last Hit + Trade, 1) / Hold (1) / Safe (1)
                 front HP >= own front HP + 8 (one wave)
    D vision     Ward legal, vision down, own front at     Ward (1) / Last Hit + Hold (1)
                 the lane cell or beyond
    E recall     Recall legal, HP <= 50%                    Recall (1) / stay (Shove + Hold, 3) / step back (LH + Safe, 3)

`smoke()` runs every family at tiny sizes on the pilot final (an end-to-end check of the framework; NOT evidence).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ....champions import champion_legal_actions
from ....config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_WARD, W_HOLD, W_SHOVE, W_SLOWPUSH, W_THIN,
                        champ_act)
from .common import agent_from_spec, env_cfg, mean_ci, meta, run_jobs, save_json

HORIZONS = (20, 50, 100)
LEDGER = ("earned", "farm", "xp", "deaths", "recalls", "d_total", "d_min", "trade_dealt", "ward_placed", "held",
          "frozen", "plates")


def _present(m4, s):
    return m4.side(s).absent == 0


def _fronts(lane, s):
    fx, fh = lane.side(s)[:2]
    ex, eh = lane.side(1 - s)[:2]
    return fx, fh, ex, eh


def p_wave(cfg, lane, m4, s) -> bool:
    fx, fh, ex, eh = _fronts(lane, s)
    return _present(m4, s) and _present(m4, 1 - s) and fh > 0 and eh > 0 and fx + ex >= -1


def p_freeze(cfg, lane, m4, s) -> bool:
    from ..m44.records import holds_freeze
    return _present(m4, s) and holds_freeze(lane, s)


def p_frozen(cfg, lane, m4, s) -> bool:
    from ..m44.records import holds_freeze
    return _present(m4, s) and holds_freeze(lane, 1 - s)


def p_trade(cfg, lane, m4, s) -> bool:
    from ...m45b.strategies import catchable
    if not (_present(m4, s) and _present(m4, 1 - s) and catchable(cfg, m4, s)):
        return False
    _, fh, _, eh = _fronts(lane, s)
    return eh >= fh + 8                    # the enemy front outweighs ours by at least one full wave (8 HP)


def p_vision(cfg, lane, m4, s) -> bool:
    fx, fh, _, _ = _fronts(lane, s)
    return (CHAMP_WARD in champion_legal_actions(lane, m4, s, cfg) and lane.vision_of(s) == 0 and fh > 0 and fx >= 0)


def p_recall(cfg, lane, m4, s) -> bool:
    from ...m45b.strategies import hp_frac
    legal = champion_legal_actions(lane, m4, s, cfg)
    return champ_act(W_HOLD, B_RECALL) in legal and hp_frac(cfg, m4, s) <= 0.5


LH, TN, SL, SH = W_HOLD, W_THIN, W_SLOWPUSH, W_SHOVE
FAMILIES: Dict[str, Tuple[Callable, Dict[str, Tuple[int, int]]]] = {
    # name: (predicate, {candidate: (code, plan turns)}); "policy" is always added
    "A_wave": (p_wave, {"LH": (champ_act(LH, B_HOLD), 4), "TN": (champ_act(TN, B_HOLD), 4),
                        "SL": (champ_act(SL, B_HOLD), 4), "SH": (champ_act(SH, B_HOLD), 4)}),
    "B_freeze": (p_freeze, {"keep": (champ_act(TN, B_HOLD), 6), "guard": (champ_act(TN, B_TRADE), 6),
                            "break": (champ_act(SH, B_HOLD), 2)}),
    "B_frozen": (p_frozen, {"shove": (champ_act(SH, B_HOLD), 2), "step_back": (champ_act(LH, B_SAFE), 4),
                            "recall": (champ_act(LH, B_RECALL), 1)}),
    "C_trade": (p_trade, {"trade": (champ_act(LH, B_TRADE), 1), "hold": (champ_act(LH, B_HOLD), 1),
                          "safe": (champ_act(LH, B_SAFE), 1)}),
    "D_vision": (p_vision, {"ward": (CHAMP_WARD, 1), "no_ward": (champ_act(LH, B_HOLD), 1)}),
    "E_recall": (p_recall, {"recall": (champ_act(LH, B_RECALL), 1), "stay": (champ_act(SH, B_HOLD), 3),
                            "step_back": (champ_act(LH, B_SAFE), 3)}),
}


def family_job(args) -> tuple:
    """(spec, family, state seeds, game turns, per_game, max_states, rollout seeds, horizon)."""
    spec, fam, state_seeds, game_turns, per_game, max_states, roll_seeds, horizon = args
    import torch
    torch.set_num_threads(1)
    from ...m45b import engine as E
    from ...m45b import states as S
    from ...m45b.reference import NetPolicy
    cfg = env_cfg()
    agent = agent_from_spec(spec, cfg)
    pred, cands = FAMILIES[fam]
    t0 = time.perf_counter()
    states = []
    for sd in state_seeds:
        states += E.collect(cfg, S.s0(cfg), NetPolicy(agent, 0, sd, 0), NetPolicy(agent, 0, sd, 1), [sd], game_turns,
                            pred, per_game=per_game, rng_seed=sd)
        if len(states) >= max_states:
            break
    states = states[:max_states]
    per_state = []
    for st in states:
        start = st["start"]
        k = start[2]
        diffs: Dict[str, Dict[str, List[float]]] = {}
        base = None
        rows = {}
        for name in ("policy",) + tuple(cands):
            vals = []
            for rs in roll_seeds:
                own = NetPolicy(agent, k, rs, 0)
                pol = own if name == "policy" else E.Then(name, E.Fixed(name, cands[name][0], own), own, cands[name][1])
                g = E.play(cfg, start, pol, NetPolicy(agent, k, rs, 1), rs, horizon, check_legal=False)["summary"]
                vals.append({**{f"G{h}": g[f"G{h}"] for h in HORIZONS if f"G{h}" in g},
                             **{q: g[f"{q}_A"] - g[f"{q}_B"] for q in LEDGER}})
            rows[name] = vals
        for name in cands:
            diffs[name] = {q: float(np.mean([a[q] - b[q] for a, b in zip(rows[name], rows["policy"])]))
                           for q in rows["policy"][0]}
        per_state.append({"turn": st["turn"], "seed": st["seed"], "diff_vs_policy": diffs})
    summary = {}
    for name in cands:
        summary[name] = {q: list(mean_ci([s["diff_vs_policy"][name][q] for s in per_state]))
                         for q in (per_state[0]["diff_vs_policy"][name] if per_state else {})}
    return (spec, fam), {"states": len(per_state), "per_state": per_state, "plan_minus_policy": summary,
                         "elapsed_s": time.perf_counter() - t0}


def smoke(spec: str = "final:pilot:100", workers: int = 6) -> dict:
    """Every family at tiny sizes (3 states, 4 paired rollouts, 50 turns) on the pilot final: a framework check."""
    from .common import SMOKE_EVAL_SEED
    jobs = [(spec, fam, tuple(range(SMOKE_EVAL_SEED + 10, SMOKE_EVAL_SEED + 30)), 200, 1, 3,
             tuple(range(SMOKE_EVAL_SEED + 50, SMOKE_EVAL_SEED + 54)), 50) for fam in FAMILIES]
    t0 = time.perf_counter()
    res = {f"{k[1]}": v for k, v in run_jobs(family_job, jobs, workers, "strategic smoke")}
    data = {"meta": meta(stage="strategic_smoke", spec=spec,
                         label="FRAMEWORK CHECK at tiny sizes (3 states x 4 rollouts x 50 turns); NOT evidence"),
            "families": res, "elapsed_s": time.perf_counter() - t0}
    save_json("strategic_smoke", data)
    return data


__all__ = ["FAMILIES", "HORIZONS", "LEDGER", "family_job", "smoke"]
