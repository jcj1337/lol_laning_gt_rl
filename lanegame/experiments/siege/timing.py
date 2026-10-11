"""Experiment E - timing and reachability by exact state-space exploration (siege model).

1. Every siege event (pressure, plates, thinning, vision established / refreshed / expired, pressure under
   vision, crash under vision, disengagement) and every lane event: number of reachable transitions from s_0
   and a replayed shortest witness.
2. Exhaustive checks over every reachable transition: no vision retroactivity, correct expiry, refresh
   without stacking, plates never exceeding the pending opportunities, thinning only with an accessible
   target and an occupied tower, HP conservation.
3. The Vision -> Shove -> Crash -> Pressure strategy: shortest witnesses from s_0 for vision durations 4/6/8,
   tower rates 1/2/3 and minimum sizes of the post-vision crash (any / full wave / medium / large), with a
   turn-by-turn trace of the default case.
4. Bounce sizes after a 12-HP crash under continuous pressure, vision only, or disengagement, by tower rate
   and thinning strength (deterministic lane trajectories from the 'bounce' start).
"""

from __future__ import annotations

import time

from ... import GameConfig, LaneGame
from ...config import CONTEST, PRESSURE, SH, SH_S, SP_C, VISION, act, objective_of, posture_of
from ...game import gank_probability
from ...mechanics import step_turn, total_hp
from ...reachability import ALL_EVENTS, SIEGE_EVENTS, EVENTS, explore_events, find_vision_strategy, replay
from ...render import state_row
from ...state import State
from .common import K, md_table, save

K_EVENTS = 24          # exact exploration of every reachable transition (K = 32 needs ~3 GB and is not required)
K_STRATEGY = 24


def exhaustive_checks(cfg: GameConfig, horizon: int) -> dict:
    """Property counts over every reachable transition (all must hold; violations are reported, not hidden)."""
    g = LaneGame(cfg.with_(horizon=horizon))
    layers = g.reachable(max_states=1_500_000)
    n = 0
    viol = {k: 0 for k in ("vision_retroactive", "vision_timer_wrong", "refresh_stacked", "plates_over_pending",
                           "plates_without_pressure", "thinning_without_target", "thinning_over_limit",
                           "thinning_without_pressure", "hp_not_conserved", "pressure_without_siege",
                           "plates_pending_without_siege")}
    stats = {"vision_set": 0, "refreshes": 0, "expiries": 0, "pressure_turns": 0, "plates": 0, "thinned": 0,
             "pressure_under_vision": 0}
    T = cfg.tower
    for k, layer in enumerate(layers):
        for s in layer:
            for a in g.legal(s, 0):
                for b in g.legal(s, 1):
                    n += 1
                    r = step_turn(s, k, a, b, cfg)
                    tr = g.transition(s, k, a, b)
                    before, after = total_hp(s, cfg), total_hp(r.state, cfg)
                    for side, a_s in ((0, a), (1, b)):
                        o = objective_of(a_s)
                        v0, v1 = s.vision_of(side), r.state.vision_of(side)
                        if o == VISION:
                            stats["vision_set"] += 1
                            stats["refreshes"] += v0 > 0
                            # not retroactive: same gank p as Apply Pressure with the same base action
                            alt = g.transition(s, k, a_s - 6 * VISION + 6 * PRESSURE if side == 0 else a,
                                               b if side == 0 else b - 6 * VISION + 6 * PRESSURE)
                            if tr.gank_p[side] != alt.gank_p[side]:
                                viol["vision_retroactive"] += 1
                            if v1 != cfg.vision_turns:
                                viol["vision_timer_wrong"] += 1
                            if v0 > 0 and v1 > cfg.vision_turns:
                                viol["refresh_stacked"] += 1
                        else:
                            if v1 != max(0, v0 - 1):
                                viol["vision_timer_wrong"] += 1
                            stats["expiries"] += (v0 == 1)
                        if o == PRESSURE:
                            stats["pressure_turns"] += 1
                            stats["pressure_under_vision"] += v0 > 0
                            if s.siege_of(side) == 0:
                                viol["pressure_without_siege"] += 1
                        c = r.plates[side]
                        stats["plates"] += c
                        if c > min(cfg.plates_per_turn, s.plates_of(side)):
                            viol["plates_over_pending"] += 1
                        if c > 0 and o != PRESSURE:
                            viol["plates_without_pressure"] += 1
                        if s.siege_of(side) == 0 and s.plates_of(side) > 0:
                            viol["plates_pending_without_siege"] += 1
                        t = r.thinned[side]
                        stats["thinned"] += t
                        if t > cfg.thin_damage:
                            viol["thinning_over_limit"] += 1
                        if t > 0 and o != PRESSURE:
                            viol["thinning_without_pressure"] += 1
                        if t > 0:
                            efx, efh = (s.bx, s.bh) if side == 0 else (s.ax, s.ah)
                            # the target must have existed at the start of the turn or arrived during it:
                            # enemy minions at or behind the tower within one cell (after at most 2 ticks of walking)
                            if efh == 0 and not ((s.br if side == 0 else s.ar)):
                                viol["thinning_without_target"] += 1
                        if before[side] + r.events["spawned_hp"][side] != after[side] + r.events["killed_hp"][side] + r.events["tower_hp"][side]:
                            viol["hp_not_conserved"] += 1
    return {"horizon": horizon, "transitions": n, "violations": viol, "counts": stats,
            "nodes": sum(len(l) for l in layers)}


def bounce_table() -> list:
    """Released defender wave after a 12-HP crash from the 'bounce' start, by tower rate, A's siege behaviour
    and thinning strength (B plays Slow Push + Contest). With Slow Push A's own waves wait at the lane, so
    only thinning touches B's waiting wave; with Shove A's arriving waves also fight it under the tower."""
    from ...config import SP, SP_S
    rows = []
    for d in (1, 2, 3):
        for thin in (0, 1, 2):
            for wave, wname in ((SP, "Slow Push (A's waves wait)"), (SH, "Shove (A's waves walk in)")):
                for name, a_act in (("disengage (Safe)", act(wave, 1)), ("vision every turn", act(wave, CONTEST, VISION)),
                                    ("pressure every turn", act(wave, CONTEST, PRESSURE))):
                    cfg = GameConfig(tower_siege_damage=d, thin_damage=thin)
                    s = State(-4, 4, (), 12, -3, 4, (), 0, 0, 0, 3, 0)
                    turns, thinned, plates, released, total = 0, 0, 0, None, None
                    for k in range(2, 14):
                        a = a_act if s.asg > 0 else act(wave, 1)
                        r = step_turn(s, k, a, SP_C, cfg)
                        if s.asg > 0:
                            turns += 1
                        thinned += r.thinned[0]
                        plates += r.plates[0]
                        s = r.state
                        if s.asg == 0:
                            released = s.bh
                            total = s.bh + cfg.wave_hp * len(s.br)
                            break
                    rows.append({"tower": d, "thin": thin, "waves": wname, "A": name, "siege_turns": turns,
                                 "thinned": thinned, "plates": plates, "released_hp": released, "released_total_hp": total})
    return rows


def run() -> dict:
    out = {}
    cfg = GameConfig(horizon=K_EVENTS)
    g = LaneGame(cfg)
    t = time.perf_counter()
    rep = explore_events(g, max_states=1_500_000)
    out["events_seconds"] = time.perf_counter() - t
    events = {}
    for e in ALL_EVENTS:
        w = rep.witness.get(e)
        ok = None
        if w is not None:
            s, r = replay(g, w.plan, w.choices)[-1]
            ok = bool(s == w.state and ALL_EVENTS[e](cfg, s, *w.plan[-1], r))
        events[e] = {"transitions": rep.counts[e], "first_turn": rep.first_k.get(e),
                     "witness": " ".join(f"{cfg.label(a)}/{cfg.label(b)}" for a, b in w.plan) if w else None,
                     "witness_state": state_row(w.state, cfg) if w else None, "replay_ok": ok}
    out["events"] = {"horizon": K_EVENTS, "nodes": rep.states, "distinct": rep.distinct_states, "by_event": events}

    out["checks"] = exhaustive_checks(GameConfig(), 24)

    strat = {}
    for V in (4, 6, 8):
        for d in (1, 2, 3):
            for mn in (1, 4, 8, 12):
                c = GameConfig(horizon=K_STRATEGY, vision_turns=V, tower_siege_damage=d)
                gg = LaneGame(c)
                t = time.perf_counter()
                w = find_vision_strategy(gg, horizon=K_STRATEGY, min_crash_hp=mn, max_states=1_500_000)
                strat[f"vision {V}, tower {d}, crash >= {mn}"] = {
                    "vision_turns": V, "tower": d, "min_crash_hp": mn, "found": w.found,
                    "vision_turn": w.vision_turn, "crash_turn": w.crash_turn, "pressure_turn": w.k,
                    "turns_after_vision": w.turns_after_vision, "search_nodes": w.states_searched,
                    "seconds": time.perf_counter() - t,
                    "plan": " ".join(f"{c.label(a)}/{c.label(b)}" for a, b in w.plan) if w.found else None,
                    "trace": w.trace if (w.found and V == 6 and d in (1, 2) and mn in (1, 4, 8)) else None,
                }
    out["strategy"] = strat
    out["bounces"] = bounce_table()

    ev = out["events"]
    md = [f"# Experiment E: timing and reachability (siege model)", "",
          f"## Reachable events from the standard start (K = {K_EVENTS}; {ev['nodes']} decision nodes, "
          f"{ev['distinct']} distinct states; {out['events_seconds']:.0f} s)", "",
          md_table(["event", "reachable transitions", "first turn", "witness replays"],
                   [(e, d["transitions"], d["first_turn"], d["replay_ok"]) for e, d in events.items()]), "",
          "Witnesses (A/B joint actions per turn; the last turn triggers the event):", ""]
    for e in SIEGE_EVENTS:
        d = events[e]
        if d["witness"]:
            md += [f"- **{e}** (turn {d['first_turn']}): `{d['witness']}`",
                   f"  - lane at the start of the event turn: `{d['witness_state']}`"]
    ch = out["checks"]
    md += ["", f"## Exhaustive property checks (K = {ch['horizon']}, {ch['transitions']} reachable transitions)", "",
           md_table(["property", "violations"], [(k, v) for k, v in ch["violations"].items()]), "",
           "Counts: " + ", ".join(f"{k} {v}" for k, v in ch["counts"].items()), "",
           "## The Vision -> Shove -> Crash -> Pressure strategy", "",
           f"Breadth-first search from s_0 over (turn, state, history flag) at K = {K_STRATEGY}. 'crash >= h' requires "
           "the crash after the vision to have at least h HP (1 = any remnant, 4 = a full wave, 8 = medium, "
           "12 = large). 'turns after vision' = turn of the protected pressure minus the turn the vision was "
           "established; the strategy works only if this is <= the vision duration.", "",
           md_table(["vision", "tower HP/tick", "crash >=", "found", "vision turn", "crash turn", "pressure turn",
                     "turns after vision", "search nodes", "s"],
                    [(r["vision_turns"], r["tower"], r["min_crash_hp"], r["found"], r["vision_turn"], r["crash_turn"],
                      r["pressure_turn"], r["turns_after_vision"], r["search_nodes"], round(r["seconds"], 1))
                     for r in strat.values()]), ""]
    for name, r in strat.items():
        if r["trace"] and r["tower"] == 1:
            md += [f"### Trace: {name}", "", f"plan: `{r['plan']}`", "",
                   md_table(["k", "A", "B", "A siege mass", "objective", "vision", "plates (avail -> after, +claimed)",
                             "B front before -> after", "thinned", "A crash HP", "A exposure", "p gank (unprotected)",
                             "lane after"],
                            [(row["k"], row["A"], row["B"], f"{row['siege_mass_before']}->{row['siege_mass_after']}",
                              row["objective"], f"{row['vision_before']}->{row['vision_after']}",
                              f"{row['plates_available']}->{row['plates_after']} (+{row['plates_claimed']})",
                              f"{row['defender_front_before']}->{row['defender_front_after']}", row["thinned"],
                              row["crash_hp"], f"{row['exposure']:+d}",
                              f"{row['gank_p']:.3f} ({row['gank_p_without_vision']:.3f})", f"`{row['lane_after']}`")
                             for row in r["trace"]]), ""]
    md += ["## Bounce after a 12-HP crash (constructed 'bounce' start; B plays Slow Push + Contest)", "",
           "released HP = HP of B's leading wave on the turn the crashed mass is gone; total = that wave plus any "
           "reinforcement queued directly behind it (a capped 12-HP wave can have a 4-HP wave queued). With Slow Push "
           "A's own waves wait at the lane, so only thinning acts on B's wave; with Shove A's arriving 4-HP waves also "
           "fight it under the tower (and die there).", "",
           md_table(["tower HP/tick", "thin_damage", "A's waves", "A during the siege", "siege turns", "thinned", "plates",
                     "released HP", "released total HP"],
                    [(r["tower"], r["thin"], r["waves"], r["A"], r["siege_turns"], r["thinned"], r["plates"], r["released_hp"],
                      r["released_total_hp"]) for r in out["bounces"]])]
    save("timing", out, "\n".join(md))
    return out


if __name__ == "__main__":
    run()
