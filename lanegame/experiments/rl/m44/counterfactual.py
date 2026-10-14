"""Paired counterfactual evaluation from saved states (M4.4a Part 4; docs/m4/m4_4a_ppo_readiness.md §5).

Question: at a state the policy actually reaches, would a different legal choice of ONE decision component have
produced a better discounted economic outcome, under stated continuation policies?

Protocol, per sampled state (a `diagnostics.Snapshot` taken from a recorded game, focal seat s):

  1. candidates   "policy" (the policy's own sampled action) and, for the comparison of the state's situation, one
                  `interventions.ComponentPlan` per alternative (wave, objective or behaviour component forced on the
                  branch turn only by default, `turns` > 1 for a multi-turn intervention), each built only where it is
                  LEGAL at that state (`interventions.possible`); a state where either side of the contrast is not
                  possible is skipped and counted, never filled with an invented action
  2. opponent     simultaneous: the opponent's branch-turn action never depends on the focal choice. Protocol
                  "policy": it samples its own policy from its own observation with its own stream, identical for every
                  candidate (same observation, same uniform). Protocol "modal": it commits to its own most probable
                  action on the branch turn (`opponent_plan`), then follows its policy
  3. continuation both seats then follow the designated continuation agents (default: the policy itself in both seats)
  4. pairing      rollout m of every candidate uses the same three streams (environment chance, focal sampling,
                  opponent sampling; `diagnostics.rollout_seed(root, m, *key)`), one uniform per seat per turn: common
                  random numbers until (and, where trajectories re-merge, after) the candidates diverge
  5. value        the focal seat's discounted return over a window of `window` turns (gamma = the game's 0.99); the
                  paired tail beyond 300 turns was measured at 0.00 +- 0.01 (M4.3b validation)

Reported separately, never mixed:
  frequency       the policy's probability mass on each candidate's component value at the state
  critic          V_hat(s) and a ONE-STEP critic prediction per candidate, E[r_0 + gamma V_hat(s_1)] over the same
                  paired streams (an approximation: the critic's own belief, not a measurement)
  measured        Monte Carlo mean discounted return per candidate, the paired differences against the policy and the
                  contrast (e.g. Thin - Shove) with standard errors; across states, the mean contrast with a 95%
                  interval (states as units) and the share of states with a significant sign

What it is NOT: the value of an intervention UNDER THE GIVEN CONTINUATION POLICIES. A positive contrast says the
alternative would have paid against this opponent with this continuation, not that it is a best response, an
equilibrium action, or optimal in any game-theoretic sense.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....config import B_HOLD, B_SAFE, B_TRADE, CHAMP_ABSENT, NONE, PRESSURE, VISION, W_HOLD, W_SHOVE, W_SLOWPUSH, W_THIN
from ....diagnostics.branch import Commit, FollowPolicy, ProbCache, branch, observation
from .actions import WAVE_INDEX, full_distribution, masses
from .interventions import ComponentPlan, const, possible
from .metrics import credible_threat
from .records import play_record

CF_WINDOW = 300
MIN_TURN = 8

# situation -> (label, predicate on a record (d, t, s), candidates [(name, component, value)], contrast (a, b))
COMPARISONS: Dict[str, dict] = {
    "freeze_held": {
        "label": "holding a freeze (freeze state, focal is the freezer): Thin vs Shove, wave only",
        "state": lambda d, t, s: d["held"][t, s] > 0,
        "candidates": [("Thin", "wave", W_THIN), ("Shove", "wave", W_SHOVE), ("Last Hit", "wave", W_HOLD)],
        "contrast": ("Thin", "Shove")},
    "neutral": {
        "label": "neutral lane (fronts in contact at the lane cell, window strengths within 1): Slow Push vs Last Hit",
        "state": lambda d, t, s: d["contact"][t, s] > 0 and d["wave_pos"][t, s] == 0 and abs(d["wave_adv"][t, s]) < 1,
        "candidates": [("Slow Push", "wave", W_SLOWPUSH), ("Last Hit", "wave", W_HOLD), ("Shove", "wave", W_SHOVE)],
        "contrast": ("Slow Push", "Last Hit")},
    "oversized_enemy": {
        "label": "oversized enemy wave (window strength 4 or more behind): Shove vs Thin, wave only",
        "state": lambda d, t, s: d["wave_adv"][t, s] <= -4 and d["en_fh"][t, s] > 0,
        "candidates": [("Shove", "wave", W_SHOVE), ("Thin", "wave", W_THIN), ("Last Hit", "wave", W_HOLD)],
        "contrast": ("Shove", "Thin")},
    "siege": {
        "label": "siege objective legal: Vision vs Pressure (objective only)",
        "state": lambda d, t, s: d["obj_legal"][t, s] > 0,
        "candidates": [("Vision", "objective", VISION), ("Pressure", "objective", PRESSURE), ("none", "objective", NONE)],
        "contrast": ("Vision", "Pressure")},
    "threatened": {
        "label": "credible threat (a stronger enemy within reach): Trade vs Safe (behaviour only)",
        "state": lambda d, t, s: bool(credible_threat(d, s)[t]),
        "candidates": [("Trade", "behavior", B_TRADE), ("Safe", "behavior", B_SAFE), ("Hold", "behavior", B_HOLD)],
        "contrast": ("Trade", "Safe")},
}
PROTOCOLS = ("policy", "modal")


# ---------------------------------------------------------------------------------------------- states
def sample_states(cfg, x, y, keys: Sequence[str], n_states: int, seed: int, game_turns: int = 300,
                  max_games: int = 40, per_game: int = 2, start=None, start_label: str = "s0",
                  min_turn: int = MIN_TURN) -> Dict[str, List[dict]]:
    """Up to n_states states per comparison key from recorded games of x (seat A) against y (x in seat B too when
    y is not x), at most `per_game` per game and key, turns >= min_turn. Each state: the Snapshot, the focal seat (a
    seat played by x where the situation holds), the decision-state row of the record and the game index."""
    rng = np.random.default_rng(seed)
    found = {k: [] for k in keys}
    children = np.random.SeedSequence(seed).spawn(max_games)
    for g, ss in enumerate(children):
        if all(len(found[k]) >= n_states for k in keys):
            break
        games = [((x, x), (0, 1))] if x is y else [((x, y), (0,)), ((y, x), (1,))]
        for agents, focal_seats in games:
            rec = play_record(cfg, agents, np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (0,)),
                              game_turns, start, start_label, probs=False, keep_snapshots=True)
            d = rec.data
            for key in keys:
                if len(found[key]) >= n_states:
                    continue
                pred = COMPARISONS[key]["state"]
                cand = [(t, s) for s in focal_seats for t in range(min_turn, rec.turns)
                        if d["present"][t, s] > 0 and pred(d, t, s)]
                rng.shuffle(cand)
                took = 0
                for t, s in cand:
                    found[key].append({"snapshot": rec.snapshots[t], "seat": int(s), "turn": int(t), "game": g,
                                       "row": {f: float(d[f][t, s]) for f in ("held", "wave_pos", "wave_adv", "power",
                                                                             "edge", "hp_frac", "level", "plates_left",
                                                                             "own_siege", "en_siege", "vision")}})
                    took += 1
                    if took >= per_game or len(found[key]) >= n_states:
                        break
    return found


# ---------------------------------------------------------------------------------------------- critic
def critic(agent) -> Optional[Callable]:
    """obs -> V_hat(obs) in resources (the network's value divided by the training reward scale), or None."""
    net = getattr(agent, "net", None)
    if net is None:
        return None
    meta = getattr(agent, "meta", None) or {}
    scale = float(meta.get("ppo", {}).get("reward_scale", 0.1))
    import torch

    def v(obs) -> float:
        with torch.no_grad():
            return float(net.value(torch.as_tensor(obs.vector).unsqueeze(0))[0]) / scale
    return v


# ---------------------------------------------------------------------------------------------- one state
def evaluate_state(cfg, state: dict, key: str, agents: Tuple, n_roll: int, root: int, idx: int,
                   window: int = CF_WINDOW, turns: int = 1, protocol: str = "policy", rule: str = "conditional",
                   cache: Optional[ProbCache] = None) -> Optional[dict]:
    """The paired counterfactual of one state (module docstring); None when the contrast is not legal there."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown opponent protocol {protocol!r}")
    comp = COMPARISONS[key]
    snap, s = state["snapshot"], state["seat"]
    ob = observation(cfg, snap, s)
    if ob.legal == (CHAMP_ABSENT,):
        return None
    cands: Dict[str, object] = {"policy": FollowPolicy()}
    legal_names = []
    for name, component, value in comp["candidates"]:
        if possible(ob.legal, component, value):
            r = rule if component == "wave" else "conditional"
            cands[name] = ComponentPlan(name, component, const(value), turns, r)
            legal_names.append(name)
    a, b = comp["contrast"]
    if a not in cands or b not in cands:
        return None
    focal_agent, opp_agent = agents[s], agents[1 - s]
    seat_agents = (agents[0], agents[1])
    opp_plan = None
    if protocol == "modal":
        oo = observation(cfg, snap, 1 - s)
        po = np.asarray(opp_agent.distribution(oo), dtype=float)
        opp_plan = Commit(int(oo.legal[int(np.argmax(po))]), "opponent modal")
    cache = cache if cache is not None else ProbCache()
    res = branch(cfg, snap, s, cands, seat_agents, n_roll, root, key=(idx,), cache=cache, max_turns=window,
                 opponent_plan=opp_plan)
    # frequency: the policy's mass on each candidate's component value
    p = full_distribution(focal_agent, ob)
    m = masses(p)
    freq = {}
    for name, component, value in comp["candidates"]:
        if name in cands:
            freq[name] = float(m["wave"][WAVE_INDEX[value]] if component == "wave" else
                               m["obj"][value] if component == "objective" else m["beh"][value])
    out = {"key": key, "seat": s, "turn": state["turn"], "game": state["game"], "row": state["row"],
           "snapshot": snap.to_dict(), "candidates": legal_names, "protocol": protocol, "window": window,
           "intervention_turns": turns, "rule": rule, "n": n_roll, "frequency": freq,
           "measured": {c: list(res.mean(c)) for c in res.candidates},
           "vs_policy": {c: list(res.diff(c, "policy")) for c in legal_names},
           "contrast": list(res.diff(a, b)), "contrast_names": [a, b],
           "farm_diff": {c: float((res.farm[c][:, 0] - res.farm[c][:, 1]).mean()) for c in res.candidates},
           "deaths": {c: res.deaths[c].mean(0).tolist() for c in res.candidates},
           "opp_first_identical": bool(all(np.array_equal(res.opp_first[c], res.opp_first["policy"])
                                           for c in res.candidates)),
           "skipped_turns": {c: getattr(cands[c], "skipped_turns", 0) for c in legal_names}}
    v = critic(focal_agent)
    if v is not None:
        pred = branch(cfg, snap, s, {c: cands[c] for c in res.candidates}, seat_agents, n_roll, root, key=(idx,),
                      cache=cache, max_turns=1, bootstrap=v, opponent_plan=opp_plan)
        out["critic"] = {"V": v(ob), "Q1": {c: float(pred.returns[c].mean()) for c in pred.candidates},
                         "contrast_Q1": float(pred.returns[a].mean() - pred.returns[b].mean()),
                         "label": "one-step critic prediction r_0 + gamma V_hat(s_1): an approximation"}
    return out


def aggregate(states: List[dict]) -> dict:
    """Across states (the unit): the mean contrast with a 95% interval, the shares of states with a significantly
    positive / negative contrast (|d| > 1.96 SE), the mean policy frequency of each side of the contrast, the mean
    difference of each candidate against the policy, and how often the critic's one-step prediction has the sign of
    the measured contrast."""
    from .common import mean_ci
    if not states:
        return {"n_states": 0}
    a, b = states[0]["contrast_names"]
    d = np.array([st["contrast"][0] for st in states])
    se = np.array([st["contrast"][1] for st in states])
    out = {"n_states": len(states), "contrast": f"{a} - {b}", "mean_contrast": list(mean_ci(d)),
           "share_significant_positive": float(np.mean(d > 1.96 * se)),
           "share_significant_negative": float(np.mean(d < -1.96 * se)),
           "frequency": {c: list(mean_ci([st["frequency"].get(c, np.nan) for st in states])) for c in (a, b)},
           "vs_policy": {}}
    names = sorted({c for st in states for c in st["vs_policy"]})
    for c in names:
        out["vs_policy"][c] = list(mean_ci([st["vs_policy"][c][0] for st in states if c in st["vs_policy"]]))
    crit = [st for st in states if "critic" in st]
    if crit:
        agree = [np.sign(st["critic"]["contrast_Q1"]) == np.sign(st["contrast"][0]) for st in crit]
        out["critic_sign_agreement"] = float(np.mean(agree))
        out["critic_mean_contrast_Q1"] = list(mean_ci([st["critic"]["contrast_Q1"] for st in crit]))
    out["opponent_first_action_identical_across_candidates"] = bool(all(st["opp_first_identical"] for st in states))
    return out


def run(cfg, x, y, keys: Sequence[str], n_states: int, n_roll: int, state_seed: int, roll_seed: int,
        window: int = CF_WINDOW, turns: int = 1, protocol: str = "policy", game_turns: int = 300,
        start=None, start_label: str = "s0", continuation: Optional[Tuple] = None) -> dict:
    """Sample states of x (against y) and evaluate every comparison key; `continuation` = the agents (seat A, seat B)
    that play after the branch turn (default: the games' own agents, x in its seat and y in the other)."""
    t0 = time.perf_counter()
    found = sample_states(cfg, x, y, keys, n_states, state_seed, game_turns, start=start, start_label=start_label)
    cache = ProbCache()
    out = {}
    for key in keys:
        res, skipped = [], 0
        for i, st in enumerate(found[key]):
            ag = continuation if continuation is not None else ((x, y) if st["seat"] == 0 else (y, x))
            r = evaluate_state(cfg, st, key, ag, n_roll, roll_seed, i, window, turns, protocol, cache=cache)
            if r is None:
                skipped += 1
            else:
                res.append(r)
        out[key] = {"label": COMPARISONS[key]["label"], "states_found": len(found[key]), "states_skipped": skipped,
                    "summary": aggregate(res), "states": res}
    return {"keys": list(keys), "protocol": protocol, "window": window, "intervention_turns": turns, "n_roll": n_roll,
            "start": start_label, "results": out, "elapsed_s": time.perf_counter() - t0}


__all__ = ["COMPARISONS", "PROTOCOLS", "CF_WINDOW", "sample_states", "critic", "evaluate_state", "aggregate", "run"]
