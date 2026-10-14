"""M4.5a demonstration trajectories for the viewer (`streamlit run lanegame/viewer/app.py` -> "Recorded trajectory",
folder results/m45a/trajectories).

Each demonstration is a scripted matchup of `lanegame.viewer.policies.MATCHUPS` (the M4.5-* entries) played through
the viewer's own `session.play`, so it replays exactly. Where the point of a demonstration depends on a chance event
(a landing hit), its seed is the FIRST seed in 0..199 whose recording shows the event (`select`, a pre-declared
predicate written in the label); every other demonstration uses seed 0. `verify` replays each file through the viewer
session and checks every turn's actions and the payoff.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

OUT = os.path.join("results", "m45a")
TRAJ_DIR = os.path.join(OUT, "trajectories")
SCHEMA = "lanegame.viewer_recording"


def _first_turn(rec, pred) -> Optional[int]:
    return next((i for i, t in enumerate(rec.turns) if pred(t.info)), None)


@dataclass(frozen=True)
class Demo:
    name: str
    matchup: str
    label: str
    select: Optional[Callable] = None        # recording -> bool (the event the demonstration must show)
    select_text: str = ""


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
]


def export(demo: Demo, out_dir: str = TRAJ_DIR) -> dict:
    from ...io import dump_json
    from ...viewer import policies as pl
    from ...viewer import session as ss
    m = pl.MATCHUPS[demo.matchup]
    seed, rec = 0, None
    for seed in range(200 if demo.select else 1):
        rec = ss.play(ss.Setup(m.config, m.scenario, m.gap, seed), policies=(pl.POLICIES[m.pa], pl.POLICIES[m.pb]),
                      turns=m.turns)
        if demo.select is None or demo.select(rec):
            break
    else:
        raise RuntimeError(f"{demo.name}: no seed in 0..199 shows the event ({demo.select_text})")
    cfg = rec.cfg
    label = demo.label + (f" Seed {seed}: {demo.select_text}." if demo.select else "")
    out = {"schema": SCHEMA, "version": 1, "config": m.config, "scenario": m.scenario, "gap": m.gap, "seed": seed,
           "names": [m.pa, m.pb], "label": label, "actions": [list(a) for a in rec.actions],
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
