"""M4.3c stabilization, Stage A3: sequences of four-action wave decisions through the real environment
(docs/m4/m4_3c_strategic_economics_calibration.md §2).

The four-action redesign study held one wave action per side constant for a whole window with fixed capacities. This
stage plays SEQUENCES (state-conditional rules and turn schedules of `strategies.LIBRARY`) through `LaneEnv` on the
EXPERIMENTAL preset `GameConfig.m4_four_actions()`: levels, investment, Recalls, ganks and deaths evolve as in play.
Every cell is played twice: with the environment's chance (64 paired seeds, root SEQ_SEED) and without any chance event
("none": a deterministic mechanical run of the same rules). Nothing here changes the game.

Scenarios (A = the acting side; starts of `harness.start`):

    build       neutral L3: Last Hit / Slow Push / Slow Push -> Last Hit -> Shove / Slow Push -> Shove / Shove
                against a Last Hit farmer and against Thin
    thin_hold   an oversized B wave (21 HP held next to A's tower; 24 HP at the lane cell) at L10, three power gaps:
                A Last Hits, Thins (Thin -> Last Hit happens by itself), Thins explicitly then Last Hits, or Shoves
    sp_thin     neutral L6: Slow Push against Last Hit, Thin, Slow Push and Shove
    adapt       neutral L6 against Thin: Slow Push throughout, Slow Push for 10 turns then Shove, Shove throughout
    recall      the freeze at L10 (A keeps the freeze; Recall while it holds vs never) and the neutral lane at L6
                (crash then Recall vs a scheduled Recall at a neutral moment vs no voluntary Recall)
    power       neutral L10 at three gaps: each constant wave action against Last Hit (Last Hit and Slow Push do not
                depend on power; Shove and the Thin cap do)

    python -m lanegame.experiments.m43c sequences        # results/m43c/sequences.{json,md}
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

from ...config import GameConfig
from . import harness as h
from . import strategies as st
from .common import mean_ci, meta, save_json, write_md

SEQ_SEED = 75_000
EPISODES = 64
TURNS = 40
TRACE_SEED = 3


def cfg4(**kw) -> GameConfig:
    return GameConfig.m4_four_actions(**kw)


EXTRA = {
    "thin_last4": st.Strategy("thin_last4", wave="thin_last",
                              description="Thin while the enemy blob is the larger, else Last Hit (explicit)."),
    "recall_at6": st.Strategy("recall_at6", recall=("at", 6), description="Last Hit; a scheduled Recall at turn 6."),
}


def strategy(name: str) -> st.Strategy:
    return EXTRA[name] if name in EXTRA else st.LIBRARY[name]


# (scenario, start spec, A strategies, B strategies, turns); start spec = (base, gap, level, hb)
STAGES: Dict[str, List[Tuple[str, tuple, Sequence[str], Sequence[str], int]]] = {
    "build": [("neutral L3", ("neutral", "equal", 3, None),
               ("farm4", "slowpush4", "grow_shove4", "build_shove4", "shove4"), ("farm4", "thin4"), TURNS)],
    "thin_hold": [(f"{nm} {gap}", (base, gap, 10, hb), ("farm4", "thin4", "thin_last4", "shove4"), ("walk_up4",), 20)
                  for nm, base, hb in (("B21 at A's -1", "freeze", 21), ("B24 at the lane", "neutral", 24))
                  for gap in ("equal", "large", "b_large")],
    "sp_thin": [("neutral L6", ("neutral", "equal", 6, None), ("slowpush4",),
                 ("farm4", "thin4", "slowpush4", "shove4"), TURNS)],
    "adapt": [("neutral L6", ("neutral", "equal", 6, None), ("slowpush4", "sp_then_shove4", "shove4"),
               ("thin4", "farm4"), TURNS)],
    "recall": [("freeze L10", ("freeze", "equal", 10, None), ("freeze4", "freeze_recall4"), ("walk_up4",), TURNS),
               ("neutral L6", ("neutral", "equal", 6, None), ("farm4", "recall_at6", "grow_shove4", "grow_crash_recall4"),
                ("farm4",), TURNS)],
    "power": [(f"neutral L10 {gap}", ("neutral", gap, 10, None), ("farm4", "thin4", "slowpush4", "shove4"), ("farm4",),
               TURNS) for gap in ("equal", "large", "b_large")],
}
KEYS = ("earned", "xp", "crashes", "crash_hp", "first_crash", "applied", "budget", "plates_n", "died", "recalled",
        "away", "invested", "hp_lost", "held_by", "longest_hold", "tower", "xp_missed")


def _job(args) -> tuple:
    stage, sc, spec, a_name, b_name, turns, chance, n = args
    cfg = cfg4()
    base, gap, level, hb = spec
    stt = h.start(base, cfg, gap=gap, level=level, hb=hb)
    sa, sb = strategy(a_name), strategy(b_name)
    games = h.cell(cfg, stt, lambda: (st.StrategyAgent(sa), st.StrategyAgent(sb)), SEQ_SEED, n, turns, chance)
    return f"{stage}|{sc}|{a_name}|{b_name}|{chance}", [g["all"] for g in games]


def _summ(ws: List[dict]) -> dict:
    out = {}
    for k in KEYS:
        for s, side in ((0, "a"), (1, "b")):
            vals = [w[k][s] for w in ws if w[k][s] is not None]
            out[f"{k}_{side}"] = mean_ci(vals) if vals else (None, None)
            if k == "first_crash":
                out[f"crashed_share_{side}"] = len(vals) / len(ws)
    out["gold_diff"] = mean_ci([w["earned"][0] - w["earned"][1] for w in ws])
    out["xp_diff"] = mean_ci([w["xp"][0] - w["xp"][1] for w in ws])
    out["mean_crash_a"] = mean_ci([w["crash_hp"][0] / w["crashes"][0] for w in ws if w["crashes"][0]]) \
        if any(w["crashes"][0] for w in ws) else (None, None)
    out["contact_mean"] = mean_ci([w["contact_mean"] for w in ws if w["contact_mean"] is not None]) \
        if any(w["contact_mean"] is not None for w in ws) else (None, None)
    return out


def traces(turns: int = 24) -> Dict[str, dict]:
    """Representative recordings (viewer.session.play, seed TRACE_SEED) of the four-action viewer matchups, as lane
    strips with each side's wave action and damage applied / calculated."""
    from ...viewer import policies as pl
    from ...viewer import session as ss
    from .wave_control import strip
    out = {}
    for name, m in pl.MATCHUPS.items():
        if m.config != "m4_four_actions":
            continue
        rec = ss.play(ss.Setup(m.config, m.scenario, m.gap, TRACE_SEED), policies=(pl.POLICIES[m.pa],
                                                                                   pl.POLICIES[m.pb]), turns=turns)
        lines = [f"{'turn':<5}" + "".join(f"{x:+d}".ljust(11) for x in range(-4, 5)) + "  A / B: wave applied/budget",
                 f"{0:<5}{strip(rec.lanes[0], rec.champs[0])}"]
        for i, t in enumerate(rec.turns):
            p = ss.wave_panel(rec, i)
            tag = " / ".join("away" if t.info["away"][s] else
                             f"{st.SHORT4[t.info['waves'][s]]}{p[n]['dealt']}/{p[n]['budget']}"
                             for s, n in enumerate(ss.SIDES))
            ev = [f"{n} {e}" for s, n in enumerate(ss.SIDES) for e, key in
                  (("crash " + str(t.info["crash_hp"][s]), "crash_hp"), ("recalls", "recalled"), ("dies", "died"),
                   ("ganked", "ganked"), ("plate", "plates_broken")) if t.info[key][s]]
            lines.append(f"{i + 1:<5}{strip(rec.lanes[i + 1], rec.champs[i + 1]):<99}  {tag}"
                         + (f"  {'; '.join(ev)}" if ev else ""))
        out[name] = {"setup": [m.config, m.scenario, m.gap, TRACE_SEED], "policies": [m.pa, m.pb],
                     "strips": lines, "earned": [round(x, 2) for x in rec.earned(rec.n)]}
    return out


def run(workers: int = 9, episodes: int = EPISODES) -> dict:
    from ..rl.m43b.common import run_jobs
    t0 = time.perf_counter()
    jobs = []
    for stage, cells in STAGES.items():
        for sc, spec, As, Bs, turns in cells:
            for a in As:
                for b in Bs:
                    jobs.append((stage, sc, spec, a, b, turns, "env", episodes))
                    jobs.append((stage, sc, spec, a, b, turns, "none", 1))
    res = dict(run_jobs(_job, jobs, workers, "sequences"))
    out = {"meta": meta(stage="sequences", preset="GameConfig.m4_four_actions()", episodes=episodes,
                        seed_root=SEQ_SEED, stages={k: [[c[0], list(c[1]), list(c[2]), list(c[3]), c[4]] for c in v]
                                                    for k, v in STAGES.items()},
                        strategies={n: strategy(n).description for n in
                                    sorted({x for v in STAGES.values() for c in v for x in tuple(c[2]) + tuple(c[3])})}),
           "cells": {k: _summ(v) for k, v in res.items() if k.endswith("|env")},
           "deterministic": {k: v[0] for k, v in res.items() if k.endswith("|none")}}
    out["traces"] = traces()
    out["meta"]["elapsed_s"] = time.perf_counter() - t0
    save_json("sequences", out)
    return out


# ---------------------------------------------------------------------------------------------- tables
def _m(x, nd=2, sign=False) -> str:
    if x is None or x[0] is None:
        return "-"
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    return f.format(x[0])


def report(out: dict) -> str:
    m = out["meta"]
    L = ["# M4.3c stabilization, Stage A3: four-action wave sequences through the environment\n",
         f"Preset `{m['preset']}`, commit `{m['commit']}` + working tree, {m['timestamp']}. {m['episodes']} paired seeds "
         f"per cell (root {m['seed_root']}) with the environment's chance; 'det' = the same cell without chance "
         "events (one deterministic run). Coordinates GLOBAL. Reading: "
         "docs/m4/m4_3c_strategic_economics_calibration.md §2.\n"]
    for stage, cells in m["stages"].items():
        L.append(f"## {stage}\n")
        for sc, spec, As, Bs, turns in cells:
            L.append(f"**{sc}** ({turns} turns; start {spec})\n")
            L.append("| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget "
                     "| B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for a in As:
                for b in Bs:
                    key = f"{stage}|{sc}|{a}|{b}"
                    c, d = out["cells"][key + "|env"], out["deterministic"][key + "|none"]
                    L.append(f"| {a} | {b} | {_m(c['gold_diff'], 2, True)} ± {c['gold_diff'][1]:.2f} | "
                             f"{_m(c['xp_diff'], 1, True)} | {_m(c['crashes_a'], 1)} ({d['crashes'][0]}) | "
                             f"{_m(c['mean_crash_a'], 1)} | {_m(c['first_crash_a'], 1)} ({d['first_crash'][0] or '-'}) | "
                             f"{_m(c['applied_a'], 0)} / {_m(c['budget_a'], 0)} | {_m(c['crashes_b'], 1)} "
                             f"({d['crashes'][1]}) | {_m(c['plates_n_a'], 2)} | {_m(c['died_a'], 2)}/{_m(c['died_b'], 2)} "
                             f"| {_m(c['recalled_a'], 2)}/{_m(c['recalled_b'], 2)} | {_m(c['held_by_a'], 1)} "
                             f"({d['held_by'][0]}) |")
            L.append("")
    L.append("## Traces (viewer matchups 4A-*, seed 3)\n")
    for name, r in out["traces"].items():
        L.append(f"**{name}** (setup {r['setup']}, A {r['policies'][0]} v B {r['policies'][1]}; earned {r['earned']})\n")
        L.append("```")
        L.extend(r["strips"])
        L.append("```\n")
    md = "\n".join(L)
    write_md("sequences", md)
    return md


__all__ = ["run", "report", "STAGES", "traces"]
