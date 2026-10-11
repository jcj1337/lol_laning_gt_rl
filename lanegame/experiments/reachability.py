"""Experiment D - reachability of wave mechanics from the standard start, with witness action sequences.

1. For every lane event (crash sizes, crash onto an existing siege, held/broken freezes, bounces,
   persistent sieges, Safe disengagement, tower farming, Contest/Safe collection differences,
   deleted minions) count the reachable transitions on which it occurs and record a shortest witness
   action sequence from s_0; every witness is replayed and checked.
2. Which behavioural scenarios start from reachable states and which from constructed states.
3. Realized collection efficiencies.
4. Timing-dependent strategies: the same lane state reached in different dynamics phases
   (k mod phase_period) with different equilibrium supports.
5. Resource/vision choices: where the equilibrium takes vision, and how close the choice is.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .. import GameConfig, LaneGame
from ..reachability import EVENTS, explore_events, replay
from ..render import state_row
from ..scenarios import all_scenarios
from .common import REVISION, md_table, save, solve

K = 32
SCENARIO_EVENT = {
    "even_lane_fight": None, "shove_wins_lane": None, "slowpush_stacks": None,
    "shove_contests_slowpush": None, "unstacked_wave_waits": None, "stacked_wave_creeps": None,
    "freeze_holds": "freeze_held", "freeze_broken": "freeze_broken", "large_crash_persists": "siege_persists",
    "small_crash_clears_quickly": "crash_small", "tower_farm_contested": "tower_farm",
    "tower_farm_safe": "tower_farm", "bounce": "bounce", "safe_lets_wave_go": "safe_disengages",
    "freeze_denial": "farm_contest_vs_safe",
}


def _plan_str(cfg, plan):
    return " ".join(f"{cfg.label(a)}/{cfg.label(b)}" for a, b in plan)


def timing_dependence(g, sol, min_remaining=48, max_examples=6):
    """Lane states reached in several dynamics phases whose equilibrium support differs by phase.

    Compared far from the horizon (>= min_remaining turns left). Values there still drift slowly with
    the number of turns remaining (persistent lane-priority gains); the drift is measured on states seen
    twice in the same phase (k and k + P) and reported per turn.
    """
    cfg = g.cfg
    P = cfg.phase_period
    by_state = defaultdict(dict)                     # state -> phase -> (k, support_A, value, legal, x)
    stationarity = 0.0
    for k, layer in enumerate(sol.layers):
        if g.horizon - k < min_remaining:
            continue
        for s in layer:
            ph = k % P
            if ph in by_state[s]:
                stationarity = max(stationarity, abs(by_state[s][ph][2] - sol.V[k][s]) / (k - by_state[s][ph][0]))
                continue
            la, _ = sol.legal[k][s]
            sup = tuple(cfg.label(a) for a, p in zip(la, sol.x[k][s]) if p > 1e-9)
            by_state[s][ph] = (k, sup, sol.V[k][s], la, sol.x[k][s])
    multi = {s: d for s, d in by_state.items() if len(d) > 1}
    differ = {s: d for s, d in multi.items() if len({v[1] for v in d.values()}) > 1}
    examples = []
    for s, d in sorted(differ.items(), key=lambda kv: -(max(v[2] for v in kv[1].values())
                                                           - min(v[2] for v in kv[1].values())))[:max_examples]:
        examples.append({"state": s.to_list(), "lane": state_row(s, cfg),
                         "by_phase": {ph: {"k": v[0], "support": list(v[1]), "V": v[2],
                                           "strategy": {cfg.label(a): round(float(p), 3)
                                                        for a, p in zip(v[3], v[4]) if p > 1e-9}}
                                      for ph, v in sorted(d.items())}})
    return {"horizon": g.horizon, "min_remaining": min_remaining, "max_same_phase_value_drift": stationarity,
            "states_in_several_phases": len(multi), "support_differs": len(differ), "examples": examples}


def vision_choices(g, sol):
    """Crash-choice nodes of A where vision has positive equilibrium probability, and how close the call is."""
    cfg = g.cfg
    rows = []
    gaps = []
    for key, x in sol.choice_x.items():
        k, pre, oa, ob = key
        if len(oa) < 2:
            continue
        # value of each of A's options against B's equilibrium choice
        from ..solver import choice_matrix
        V_next = sol.V[k + 1] if k + 1 < g.horizon else None
        tr_opts = type("T", (), {})()
        C = np.empty((len(oa), len(ob)))
        for i, o1 in enumerate(oa):
            for j, o2 in enumerate(ob):
                from ..game import apply_choice
                s2, r = apply_choice(pre, o1, o2, cfg)
                C[i, j] = r + (V_next[s2] if V_next is not None else 0.0)
        vals = C @ sol.choice_y[key]
        gaps.append(abs(vals[0] - vals[1]))
        if x[1] > 1e-9:
            rows.append({"k": k, "lane": state_row(pre, cfg), "options": [list(o) for o in oa],
                         "p_vision": float(x[1]), "option_values": vals.tolist()})
    return {"choice_nodes": len(gaps), "vision_nodes": len(rows),
            "median_option_gap": float(np.median(gaps)) if gaps else float("nan"),
            "share_gap_below_0.25": float(np.mean(np.array(gaps) < 0.25)) if gaps else float("nan"),
            "examples": sorted(rows, key=lambda r: r["k"])[:8]}


def run(horizon: int = K) -> dict:
    out = {}
    for name, cfg in (("revised", GameConfig.v2(horizon=horizon)), ("v1", GameConfig.v1(horizon=horizon))):
        g = LaneGame(cfg)
        rep = explore_events(g)
        events = {}
        for e in EVENTS:
            w = rep.witness.get(e)
            ok = None
            if w is not None:
                s, r = replay(g, w.plan, w.choices)[-1]
                ok = bool(s == w.state and EVENTS[e](cfg, s, *w.plan[-1], r))
            events[e] = {"transitions": rep.counts[e], "first_turn": rep.first_k.get(e),
                         "witness": _plan_str(cfg, w.plan) if w else None,
                         "witness_choices": w.choices if w else None,
                         "witness_state": state_row(w.state, cfg) if w else None, "replay_ok": ok}
        out[name] = {"states": rep.states, "distinct": rep.distinct_states, "events": events,
                     "collection_values": {str(k): v for k, v in sorted(rep.collection_values.items())}}

    cfg = GameConfig.v2(horizon=horizon)
    g, layers, sol, _ = solve(cfg)
    P = cfg.phase_period
    scen = []
    for sc in all_scenarios(cfg):
        ev = SCENARIO_EVENT.get(sc.name)
        scen.append({"scenario": sc.name, "start": "reachable" if sc.start_reachable else "constructed",
                     "event": ev or "-", "event_reachable_from_s0":
                         (out["revised"]["events"][ev]["transitions"] > 0) if ev else "-"})
    g128, _, sol128, _ = solve(cfg.with_(horizon=128))
    timing = timing_dependence(g128, sol128)
    vision = vision_choices(g, sol)
    out["scenarios"] = scen
    out["timing"] = timing
    out["vision"] = vision

    r = out["revised"]
    md = [f"# Experiment D: reachability from the standard start (K = {horizon})", "",
          f"Revised model: {r['states']} reachable decision nodes, {r['distinct']} distinct states. "
          f"Original model: {out['v1']['states']} nodes, {out['v1']['distinct']} distinct states.", "",
          "## Events (number of reachable transitions on which the event occurs)", "",
          md_table(["event", "revised", "original (v1)", "first turn (revised)", "witness replays"],
                   [(e, r["events"][e]["transitions"], out["v1"]["events"][e]["transitions"],
                     r["events"][e]["first_turn"], r["events"][e]["replay_ok"]) for e in EVENTS]), "",
          "## Witness action sequences from s_0 (revised model; A/B joint actions per turn, last turn triggers "
          "the event; crash choices: option indices taken after each turn, 0 = resources)", ""]
    for e in EVENTS:
        d = r["events"][e]
        if d["witness"]:
            md += [f"- **{e}** (turn {d['first_turn']}): `{d['witness']}`"
                   + (f"; choices {d['witness_choices']}" if any(i or j for i, j in d['witness_choices']) else ""),
                   f"  - state at the start of the event turn: `{d['witness_state']}`"]
        else:
            md.append(f"- **{e}**: not reachable")
    md += ["", "## Behavioural scenarios: constructed vs. reachable starts", "",
           "A constructed start is a hand-built state used to isolate a mechanic; the event it illustrates is "
           "checked separately for reachability from s_0.", "",
           md_table(["scenario", "start state", "illustrated event", "event reachable from s_0"],
                    [(d["scenario"], d["start"], d["event"], d["event_reachable_from_s0"]) for d in scen]), "",
           "## Realized collection efficiencies (revised; count of reachable (turn, side, location) kill records)", "",
           md_table(["efficiency", "count"], list(r["collection_values"].items())), "",
           "## Timing-dependent strategies", "",
           f"Solved at K = {timing['horizon']} and compared at least {timing['min_remaining']} turns before the "
           f"horizon. Values there still drift with the number of turns remaining by at most "
           f"{timing['max_same_phase_value_drift']:.3f} per turn (measured on the same state in the same phase at k "
           f"and k + {P}; this is the persistent lane-priority gain discussed in the report), so a value difference "
           f"between two phases larger than a few tenths reflects timing, not the horizon. Lane states reachable in more "
           f"than one dynamics phase (k mod {P}): {timing['states_in_several_phases']}; of these, "
           f"{timing['support_differs']} have a different equilibrium support for A in different phases "
           "(same lane, different spawn/creep timing).", ""]
    for ex in timing["examples"]:
        md += ["```", "  " + ex["lane"], "```"]
        for ph, v in ex["by_phase"].items():
            md.append(f"- phase {ph} (k = {v['k']}): V* = {v['V']:.3f}, A plays {v['strategy']}")
        md.append("")
    md += ["## Resource vs. vision choices", "",
           f"A has {vision['choice_nodes']} crash-choice nodes; vision has positive probability at "
           f"{vision['vision_nodes']}. Median |value(resources) - value(vision)| against B's equilibrium: "
           f"{vision['median_option_gap']:.3f}; share of nodes where the two options are within 0.25 resources: "
           f"{vision['share_gap_below_0.25']:.2f}.", ""]
    for ex in vision["examples"]:
        md.append(f"- k = {ex['k']}: `{ex['lane']}` options {ex['options']} -> P(vision) = {ex['p_vision']:.2f}, "
                  f"option values {[round(v, 3) for v in ex['option_values']]}")
    save("reachability", out, "\n".join(md), REVISION)
    return out


if __name__ == "__main__":
    run()
