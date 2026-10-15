"""M4.5b demonstration trajectories for the viewer (`streamlit run lanegame/viewer/app.py` -> "Recorded trajectory",
folder results/m45b/trajectories).

Each demonstration is a scripted pairing from an M4.5b start (`scenarios.STARTS`, viewer gap "equal") played through
the viewer's own `session.play`, so it replays exactly (`m44.trace.verify`). Where the point depends on a chance event
its seed is the FIRST seed of TRACE (87,900-87,999) whose recording shows the event (`select`, a predicate declared in
the label); otherwise the first TRACE seed. Every demonstration is recorded under the M4.5a revision-2 baseline
(`m45_unified`) and under the calibrated candidate, each with its own first qualifying seed, so the two can be compared
side by side; files are named <name>__<config>.json.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from ...config import B_HOLD, B_TRADE, W_HOLD, champ_act
from . import engine as E
from . import strategies as ST
from .common import TRACE, TRAJ_DIR

SCHEMA = "lanegame.viewer_recording"
A_TRADE, A_HOLD = champ_act(W_HOLD, B_TRADE), champ_act(W_HOLD, B_HOLD)


def _first(rec, pred) -> Optional[int]:
    return next((i for i, t in enumerate(rec.turns) if pred(t.info)), None)


def _held_run(rec) -> int:
    from ..rl.m44.records import holds_freeze
    n = 0
    for lane in rec.lanes[1:]:
        if not holds_freeze(lane, 0):
            break
        n += 1
    return n


def _b_dies_early(rec, within=10):
    i = _first(rec, lambda inf: inf["died"][1])
    return i is not None and i < within


def _a_recalls_after_freeze(rec):
    from ..rl.m44.records import holds_freeze
    for i, t in enumerate(rec.turns[:40]):
        if t.info["recalled"][0] and any(holds_freeze(l, 0) for l in rec.lanes[max(0, i - 3):i + 1]):
            return True
    return False


def _ward_placed_exposed(rec):
    i = _first(rec, lambda inf: inf["ward_placed"][0])
    if i is None:
        return False
    return any(t.info["exposure"][0] >= 1 for t in rec.turns[i + 1:i + 7])


def _first_ward_interrupted(rec):
    i = _first(rec, lambda inf: inf["ward"][0])
    return i is not None and rec.turns[i].info["ward_interrupted"][0]


def _trade_lands_with_retaliation(rec):
    inf = rec.turns[0].info
    return inf["trade_damage"][0] > 0 and inf["minion_hits"][0] > 0


def _kill_then_crash(rec):
    i = _first(rec, lambda inf: inf["died"][1])
    if i is None or i > 8:
        return False
    return any(t.info["crash_hp"][0] > 0 and t.info["away"][1] for t in rec.turns[i:i + 6])


@dataclass(frozen=True)
class Demo:
    name: str
    label: str
    scenario: str
    pa: object
    pb: object
    turns: int
    select: Optional[Callable] = None
    select_text: str = ""


def _trade_then(cont: str):
    p = ST.get(cont)
    return E.Then("trade1", E.Fixed("T", A_TRADE, p), p, 1)


DEMOS: List[Demo] = [
    Demo("m45b_01_freeze_held", "1. A successful strong-side freeze: A (x1.76, level 10) keeps B's 12-HP wave at its -1 "
         "with Thin (fz_passive_sh); B walks up to its +1 and last-hits A's minions dying at A's tower (wk_lh_sh). "
         "Watch the held wave, B's exposure to ganks at its +1 and both sides' farm.", "m45b_freeze_10_large_h12",
         ST.get("fz_passive_sh"), ST.get("wk_lh_sh"), 40, lambda rec: _held_run(rec) >= 12,
         "the first seed whose freeze holds >= 12 turns"),
    Demo("m45b_02_freeze_broken", "2. A broken freeze: the same start, A keeps the freeze passively (Thin + Hold, "
         "fz_passive_sh) and B walks up and Shoves (wk_shove_sh): the freeze ends on the first turn.",
         "m45b_freeze_10_large_h12", ST.get("fz_passive_sh"), ST.get("wk_shove_sh"), 30),
    Demo("m45b_03_punish_walkup", "3. The stronger champion punishing a walk-up Shove: A (x1.76) guards its freeze of "
         "B's 5-HP wave; B walks up and Shoves; A steps into B's wave to trade (positional fire on A) and kills B.",
         "m45b_freeze_10_large_h5", ST.get("fz_guard_sh"), ST.get("wk_shove_sh"), 30, _b_dies_early,
         "the first seed in which B dies within 10 turns"),
    Demo("m45b_04_retreat_loses", "4. A weaker champion retreating and losing economic access: against the guarded "
         "freeze B steps back (Safe while frozen on, wk_back_sh): it takes no trade but cannot reach A's minions dying "
         "at A's tower.", "m45b_freeze_10_large_h12", ST.get("fz_guard_sh"), ST.get("wk_back_sh"), 30),
    Demo("m45b_05_freeze_then_recall", "5. A freeze kept until the stronger champion needs to Recall: A (x1.76, "
         "level 4) guards; minion fire and B's resets wear A down; below 55% HP A Shoves and Recalls "
         "(fz_guard_sh_sr v wk_shove_reset_sh).", "m45b_freeze_4_large_h12", ST.get("fz_guard_sh_sr"),
         ST.get("wk_shove_reset_sh"), 40, _a_recalls_after_freeze,
         "the first seed in which A recalls within 3 turns of holding the freeze"),
    Demo("m45b_06_ward_favorable", "6. A favourable Ward: A pushes (Shove) and wards whenever its vision is down, its "
         "wave is forward and B's wave is not in A's half (g_shove_ward), so its protected turns are spent exposed; B "
         "farms.", "m45b_vision_forward_L4", ST.get("g_shove_ward"), ST.get("g_farm"), 30, _ward_placed_exposed,
         "the first seed whose first placed ward is followed by exposed turns"),
    Demo("m45b_07_ward_interrupted", "7. An interrupted Ward: A farms and wards from the neutral lane (g_ward); B "
         "trades whenever A is catchable (g_bully), reaches the cell next to W and interrupts the ward.",
         "m45b_vision_neutral_L4", ST.get("g_ward"), ST.get("g_bully"), 20, _first_ward_interrupted,
         "the first seed whose first Ward is interrupted"),
    Demo("m45b_08_trade_big_wave_L2", "8. An early trade into a large wave: level 2, B stands in its 24-HP stack; A "
         "trades on the first turn (then farms); a landed hit draws retaliation.", "m45b_trade_L2_w24",
         _trade_then("g_farm"), ST.get("g_farm"), 12, _trade_lands_with_retaliation,
         "the first seed whose first-turn trade lands and draws minion fire"),
    Demo("m45b_09_trade_big_wave_L18", "9. A late trade into the same wave: level 18, 128 resources invested each; the "
         "same retaliation is a small share of the bar.", "m45b_trade_L18_w24", _trade_then("g_farm"),
         ST.get("g_farm"), 12, _trade_lands_with_retaliation,
         "the first seed whose first-turn trade lands and draws minion fire"),
    Demo("m45b_10_kill_bounce", "10. An unexpected interaction: a kill that loses the wave. A guards its freeze but "
         "keeps Last Hit after it (fz_guard); B (wk_lh) dies; during B's absence A's larger wave pushes and crashes "
         "into B's tower, and B returns to farm the crash under its own tower while A's lane is empty.",
         "m45b_freeze_10_large_h12", ST.get("fz_guard"), ST.get("wk_lh"), 30, _kill_then_crash,
         "the first seed in which B dies within 8 turns and A's wave crashes while B is away"),
]


def export(demo: Demo, config: str, out_dir: str = TRAJ_DIR) -> dict:
    from ...io import dump_json
    from ...viewer import session as ss
    from ..m45a.demos import summarize
    rec, seed, shown = None, None, True
    for seed in (TRACE if demo.select else TRACE[:1]):
        setup = ss.Setup(config, demo.scenario, "equal", seed)
        rec = ss.play(setup, policies=(demo.pa, demo.pb), turns=demo.turns)
        if demo.select is None or demo.select(rec):
            break
    else:                                    # no TRACE seed shows the event: record the first seed and say so
        shown, seed = False, TRACE[0]
        rec = ss.play(ss.Setup(config, demo.scenario, "equal", seed), policies=(demo.pa, demo.pb), turns=demo.turns)
    cfg = rec.cfg
    if demo.select:
        label = demo.label + (f" Seed {seed}: {demo.select_text}." if shown else
                              f" Seed {seed}: NO TRACE seed shows the event ({demo.select_text}).")
    else:
        label = demo.label + f" Seed {seed}."
    label += f" Configuration {config}."
    names = [getattr(demo.pa, "name", "plan"), getattr(demo.pb, "name", "plan")]
    out = {"schema": SCHEMA, "version": 1, **ss.recording_stamp(config), "scenario": demo.scenario, "gap": "equal",
           "seed": seed, "names": names, "label": label, "actions": [list(a) for a in rec.actions],
           "labels": [[cfg.label(a), cfg.label(b)] for a, b in rec.actions],
           "events": [{"turn": i, "events": ss.turn_events(t.info, cfg)} for i, t in enumerate(rec.turns)],
           "payoff_A": float(sum(t.info["reward"] for t in rec.turns)), "turns": rec.n, "summary": summarize(rec)}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{demo.name}__{config}.json")
    dump_json(out, path)
    return {"name": demo.name, "config": config, "path": path, "seed": seed, "turns": rec.n, "event_shown": shown,
            "summary": out["summary"], "payoff_A": out["payoff_A"]}


def run(configs: Sequence[str] = ("m45_unified", "m45_calibrated"), out_dir: str = TRAJ_DIR) -> Dict[str, list]:
    from ..rl.m44.trace import verify
    files = []
    for d in DEMOS:
        for c in configs:
            r = export(d, c, out_dir)
            r["verify"] = verify(r["path"])
            files.append(r)
            print(f"{d.name} [{c}]: seed {r['seed']}, {r['turns']} turns, payoff A {r['payoff_A']:+.2f}, verify "
                  f"{r['verify']['actions_equal']} / {r['verify']['payoff_equal']}", flush=True)
    return {"files": files}


__all__ = ["DEMOS", "Demo", "export", "run"]
