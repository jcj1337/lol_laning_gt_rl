"""M4.5a demonstration trajectories for the viewer (`streamlit run lanegame/viewer/app.py` -> "Recorded trajectory",
folder results/m45a/trajectories).

Each demonstration is a scripted matchup of `lanegame.viewer.policies.MATCHUPS` (the M4.5-* entries), or (the M4.5a
movement correction, `actions` set) a fixed action list from a start of `scenarios.STARTS`, played through the
viewer's own `session.play`, so it replays exactly. Where the point of a demonstration depends on a chance event
(a landing hit), its seed is the FIRST seed in 0..199 whose recording shows the event (`select`, a pre-declared
predicate written in the label); every other demonstration uses seed 0. `verify` replays each file through the viewer
session and checks every turn's actions and the payoff. Every file carries the rules revision (`rules_revision` 2:
the corrected `m45_unified()`; files without it were recorded under revision 1 and replay as `m45_unified_r1()`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from ...config import B_HOLD, B_TRADE, SP, champ_act

OUT = os.path.join("results", "m45a")
TRAJ_DIR = os.path.join(OUT, "trajectories")
SCHEMA = "lanegame.viewer_recording"


def _first_turn(rec, pred) -> Optional[int]:
    return next((i for i, t in enumerate(rec.turns) if pred(t.info)), None)


@dataclass(frozen=True)
class Demo:
    name: str
    matchup: str                             # a MATCHUPS key, or "" for a fixed action list
    label: str
    select: Optional[Callable] = None        # recording -> bool (the event the demonstration must show)
    select_text: str = ""
    scenario: str = ""                       # with `actions`: the start (scenarios.STARTS) on m45_unified, gap equal
    actions: Tuple[Tuple[int, int], ...] = ()


def _ward_hit(rec):
    i = _first_turn(rec, lambda inf: inf["ward"][0])
    return i is not None and rec.turns[i].info["ward_interrupted"][0]


def _ward_missed(rec):
    i = _first_turn(rec, lambda inf: inf["ward"][0])
    return i is not None and rec.turns[i].info["trade_connected"][1] and rec.turns[i].info["ward_placed"][0]


def _first_trade_lands(rec):
    i = _first_turn(rec, lambda inf: inf["trade_connected"][0])
    return i is not None and rec.turns[i].info["trade_damage"][0] > 0 and rec.turns[i].info["minion_hits"][0] > 0


def _freeze_retaliation(rec):
    return any(t.info["minion_fire"] and "retaliation" in t.info["minion_fire"][0] for t in rec.turns[:10])


def _quiet(rec):
    return not any(any(t.info["ganked"]) for t in rec.turns)


def _chase_missed(rec):
    i = rec.turns[0].info
    return _quiet(rec) and i["trade_connected"][0] and i["trade_damage"][0] == 0


def _chase_landed(rec):
    i = rec.turns[0].info
    return _quiet(rec) and i["trade_connected"][0] and i["trade_damage"][0] > 0


# the M4.5a movement-correction actions: Last Hit + Hold / Trade
_HOLD, _TRADE = champ_act(SP, B_HOLD), champ_act(SP, B_TRADE)


DEMOS: List[Demo] = [
    Demo("m45a_E1_ward_neutral", "M4.5-1. Ward from a neutral lane",
         "Scenario E (preemptive Vision): A wards at W from the neutral lane whenever its vision is down and B's wave "
         "is not in A's half, then walks back; B last-hits. Seed 0."),
    Demo("m45a_E2_ward_interrupted", "M4.5-2. Ward contested by a trader",
         "Scenario E (contested placement): A wards; B Trades every turn and reaches the lane cell next to W.",
         _ward_hit, "the first seed whose first Ward is interrupted by B's landing hit"),
    Demo("m45a_E3_ward_trade_missed", "M4.5-2. Ward contested by a trader",
         "Scenario E (contested placement): as E2, but B's trade misses: a connected Trade that deals no damage does "
         "not interrupt the ward.", _ward_missed, "the first seed whose first Ward meets a connected but missed Trade"),
    Demo("m45a_C1_bigwave_level2", "M4.5-3. Trading into a big wave (level 2)",
         "Scenario C (early): A trades B, who stands with a 24-HP stack; after A's hit lands, B's stack fires at A "
         "from the next tick.", _first_trade_lands,
         "the first seed whose first connected trade by A lands and draws minion fire"),
    Demo("m45a_C2_bigwave_level18", "M4.5-4. Trading into a big wave (level 18)",
         "Scenario C (late): the same lane at level 18 with 128 resources invested each: the same minion fire is a "
         "small share of the bar.", _first_trade_lands,
         "the first seed whose first connected trade by A lands and draws minion fire"),
    Demo("m45a_D1_ahead_of_wave", "M4.5-5. Chasing ahead of the own wave",
         "Scenario D (positional aggro): A stands in B's wave with its own wave behind and chases B, who steps back; "
         "B's minions fire at A while A stands in their cell. Seed 0."),
    Demo("m45a_A1_strong_freeze", "M4.5-6. Strong freeze: Thin + Trade",
         "Scenario A/B (strong-side freeze): A (x1.76) keeps the freeze with Thin and trades the walking-up B; B's "
         "minions retaliate on A after landing hits; over 40 turns watch HP, Recalls and whether the freeze holds.",
         _freeze_retaliation, "the first seed with a retaliation on A within the first 10 turns"),
    # the M4.5a movement correction (report section 10): one or two turns from a hand-built start, champions drawn
    # per tick
    Demo("m45a_M1_hold_follows", "", "Movement correction, Scenario A: A Holds at the lane cell with its 16-HP "
         "front; B's 2-HP front at A's +1 dies on tick 0. A stays with its wave on tick 0 and steps to +1 with it on "
         "tick 1: no minion fire (the first implementation placed A at +1 for both ticks).", _quiet,
         "the first seed without a gank", scenario="follow", actions=((_HOLD, _HOLD), (_HOLD, _HOLD))),
    Demo("m45a_M2_chase_missed", "", "Movement correction, Scenario C: A Trades B, who stands behind its 16-HP wave; "
         "A walks into B's front's cell (its own front behind) and is hit by positional aggro; its Trade connects "
         "but misses, so there is no retaliation.", _chase_missed,
         "the first seed whose trade connects and misses, without a gank", scenario="chase",
         actions=((_TRADE, _HOLD),)),
    Demo("m45a_M3_chase_landed", "", "Movement correction, Scenario C: as M2, but A's Trade lands: positional aggro "
         "on tick 0, retaliation on tick 1.", _chase_landed,
         "the first seed whose trade connects and lands, without a gank", scenario="chase",
         actions=((_TRADE, _HOLD),)),
    Demo("m45a_M4_walk_back", "", "Movement correction, Scenario D: A Holds at B's tower cell far ahead of its wave "
         "and walks back; on tick 0 it stands in B's oncoming front's cell (hit), on tick 1 it meets its own wave at "
         "the lane cell, which covers it.", _quiet, "the first seed without a gank", scenario="walk_back",
         actions=((_HOLD, _HOLD), (_HOLD, _HOLD))),
]


def export(demo: Demo, out_dir: str = TRAJ_DIR) -> dict:
    from ...io import dump_json
    from ...viewer import policies as pl
    from ...viewer import session as ss
    if demo.actions:
        config, scenario, gap, names = "m45_unified", demo.scenario, "equal", ["scripted", "scripted"]
    else:
        m = pl.MATCHUPS[demo.matchup]
        config, scenario, gap, names = m.config, m.scenario, m.gap, [m.pa, m.pb]
    seed, rec = 0, None
    for seed in range(200 if demo.select else 1):
        setup = ss.Setup(config, scenario, gap, seed)
        if demo.actions:
            rec = ss.play(setup, prefix=demo.actions, prefix_note="scripted")
        else:
            rec = ss.play(setup, policies=(pl.POLICIES[m.pa], pl.POLICIES[m.pb]), turns=m.turns)
        if demo.select is None or demo.select(rec):
            break
    else:
        raise RuntimeError(f"{demo.name}: no seed in 0..199 shows the event ({demo.select_text})")
    cfg = rec.cfg
    label = demo.label + (f" Seed {seed}: {demo.select_text}." if demo.select else "")
    out = {"schema": SCHEMA, "version": 1, **ss.recording_stamp(config), "scenario": scenario, "gap": gap,
           "seed": seed, "names": names, "label": label, "actions": [list(a) for a in rec.actions],
           "labels": [[cfg.label(a), cfg.label(b)] for a, b in rec.actions],
           "events": [{"turn": i, "events": ss.turn_events(t.info, cfg)} for i, t in enumerate(rec.turns)],
           "payoff_A": float(sum(t.info["reward"] for t in rec.turns)), "turns": rec.n,
           "summary": summarize(rec)}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, demo.name + ".json")
    dump_json(out, path)
    return {"name": demo.name, "path": path, "seed": seed, "turns": rec.n, "summary": out["summary"]}


def summarize(rec) -> Dict[str, object]:
    """Per side totals of the M4.5a events of a recording (from the environment's info only)."""
    out = {}
    for s, n in enumerate(("A", "B")):
        inf = [t.info for t in rec.turns]
        out[n] = {"wards": sum(i["ward"][s] for i in inf), "wards_placed": sum(i["ward_placed"][s] for i in inf),
                  "wards_interrupted": sum(i["ward_interrupted"][s] for i in inf),
                  "minion_units_taken": sum(i["minion_hits"][s] for i in inf),
                  "minion_centi_taken": sum(i["minion_damage"][s] for i in inf),
                  "retaliation_ticks_taken": sum(c == "retaliation" for i in inf for c in i["minion_fire"][s]),
                  "positional_ticks_taken": sum(c == "positional" for i in inf for c in i["minion_fire"][s]),
                  "trade_damage_dealt": sum(i["trade_damage"][s] for i in inf),
                  "capped_trades": sum(i["trade_wave_capped"][s] for i in inf),
                  "deaths": sum(i["died"][s] for i in inf), "recalls": sum(i["recalled"][s] for i in inf),
                  "gold": round(sum(i["earned_units"][s] for i in inf) / rec.cfg.gold_scale, 4)}
    return out


def run(out_dir: str = TRAJ_DIR) -> dict:
    from ..rl.m44.trace import verify
    files = []
    for d in DEMOS:
        r = export(d, out_dir)
        r["verify"] = verify(r["path"])
        files.append(r)
        print(f"{d.name}: seed {r['seed']}, {r['turns']} turns, verify {r['verify']['actions_equal']} / "
              f"{r['verify']['payoff_equal']}")
    return {"files": files}


__all__ = ["DEMOS", "Demo", "export", "run", "summarize", "TRAJ_DIR"]
