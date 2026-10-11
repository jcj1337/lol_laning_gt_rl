"""Experiment A - before vs. after, as an incremental ablation ladder.

Rung 0 is the original Phase 1 model exactly (GameConfig.v1()); the last rung is the
revised default (GameConfig.v2()). Each rung changes one group of mechanics, cumulatively, so
the effect of the new economy can be separated from the effects of positioning, Freeze
legality, tower mechanics, HP resolution and the Slow Push rule. Every rung is solved
exactly at K = 32 and measured the same way.
"""

from __future__ import annotations

import time

from .. import GameConfig, LaneGame
from ..baselines import all_baselines, v1_baselines
from ..best_response import exploitability
from ..reachability import explore_events
from .common import REVISION, eq_summary, exploit_row, fmt_freq, md_table, save, solve

K = 32


def ladder():
    v1 = GameConfig.v1(horizon=K)
    r1 = v1.with_(farm_per_hp=0.5)                                       # 1 resource per fresh (2 HP) wave
    r2 = r1.with_(positioning=True, gank_action=(0.02, 0.05, 0.02, 0.0, 0.0, 0.0),
                  posture_cap=(None, None, None, -1, -1, -1))
    r3 = r2.with_(freeze_legal_max=-1)
    r4 = r3.with_(tower_mode="mass", tower_siege_damage=1, siege_cap=24)
    r5 = r4.with_(cap_rule="block")
    r6 = r5.with_(wave_hp=4, farm_per_hp=0.25, freeze_break_threshold=10, freeze_break_push=(0, 4, 0),
                  siege_cap=48)
    r7 = r6.with_(slowpush_mode="stack_creep", slowpush_creep_period=2)
    assert r7 == GameConfig.v2(horizon=K), "the last rung must be the revised default"
    return [
        ("0 original model (v1)", "-", v1),
        ("1 + farming economy", "minion kills pay farm_per_hp x collection efficiency (Contest only)", r1),
        ("2 + Contest/Safe positioning", "6 combined actions; Safe caps exposure at own -1; flat Contest risk", r2),
        ("3 + stricter Freeze legality", "Freeze only with enemy minions at own x <= -1", r3),
        ("4 + mass-based tower fire", "tower removes 1 HP/tick of crashed mass instead of 1 packet/turn", r4),
        ("5 + no minion deletion at the HP cap", "full stacks queue reinforcements (which fight with them) "
                                                 "instead of deleting HP", r5),
        ("6 + finer HP units (W = 4)", "4 HP per wave, same per-wave firepower: fights last a spawn interval", r6),
        ("7 + Slow Push 'stack_creep' = revised", "unstacked waves wait at the lane, stacked waves creep", r7),
    ]


def measure(cfg) -> dict:
    g, layers, sol, secs = solve(cfg)
    out = eq_summary(g, sol)
    out["seconds"] = secs
    t = time.perf_counter()
    rep = explore_events(LaneGame(cfg))
    out["events"] = rep.counts
    out["events_seconds"] = time.perf_counter() - t
    out["baselines"] = {}
    for b in all_baselines():
        row = exploit_row(g, layers, sol, b)
        row.pop("br_path")
        out["baselines"][b.name] = row
    if not cfg.positioning and cfg.farm_per_hp == 0:
        out["v1_baselines"] = {b.name: exploitability(g, layers, b, 0, sol.value).exploitability
                               for b in v1_baselines()}
    return out


def run() -> dict:
    res = {}
    for name, change, cfg in ladder():
        print("rung", name, flush=True)
        res[name] = {"change": change, "config": cfg.to_dict(), **measure(cfg)}

    rungs = list(res)
    md = [f"# Experiment A: before vs. after (ablation ladder, K = {K})", "",
          "Each rung adds one change to the previous rung. Rung 0 is the original model; rung 7 is the "
          "revised default. All numbers are exact (backward induction / exact forward path statistics).", "",
          md_table(["rung", "change"], [(n, res[n]["change"]) for n in rungs]), "",
          "## State space, solver and equilibrium structure", "",
          md_table(["rung", "decision nodes", "distinct states", "mixed nodes", "solve s", "P(Contest) avg node",
                    "P(Freeze) where legal", "max gap"],
                   [(n, d["states"], d["distinct"], d["mixed"], round(d["seconds"], 2),
                     round(d["nodes"]["contest"], 3), round(d["nodes"]["freeze_prob_when_legal"], 3),
                     f"{d['max_gap']:.1e}") for n, d in res.items()]), "",
          "## Equilibrium path (symmetric start; per player, totals over the game)", "",
          md_table(["rung", "actions (A)", "crashes A S/M/L", "farm collected", "farm available", "farm missed",
                    "crash resources", "gank loss", "vision choices"],
                   [(n, fmt_freq(d["path"]["action_freq"][0]),
                     "/".join(f"{c:.2f}" for c in d["path"]["crashes_by_size"]["A"]),
                     round(d["path"]["farm"][0], 2), round(d["path"]["farm_available"][0], 2),
                     round(d["path"]["farm_missed"][0], 2), round(d["path"]["crash_resources"][0], 2),
                     round(d["path"]["gank_loss"][0], 2), round(d["path"]["vision_choices"][0], 2))
                    for n, d in res.items()]), "",
          "## Vision choice (mean probability of taking vision at A's crash-choice nodes)", "",
          md_table(["rung", "small", "medium", "large", "choice nodes S/M/L"],
                   [(n, *[round(d["vision"].get(s, float("nan")), 2) for s in ("small", "medium", "large")],
                     "/".join(str(d["vision"].get(f"{s}_nodes", 0)) for s in ("small", "medium", "large")))
                    for n, d in res.items()]), "",
          "## Reachable events from the standard start (number of reachable transitions)", "",
          md_table(["rung"] + list(res[rungs[0]]["events"]),
                   [(n, *d["events"].values()) for n, d in res.items()]), "",
          "## Exploitability of the revised baselines (as A) and its decomposition", "",
          "Each cell: exploitability [farm / crash / gank part of the baseline's value against its best response]. "
          "When positioning is off, Safe actions fall back to the Contest action with the same wave management "
          "(so passive_safe = contest_slowpush and release_safe = aggressive on rungs 0-1).", "",
          md_table(["rung"] + [b.name for b in all_baselines()],
                   [(n, *[f"{r['exploitability']:.2f} [{r['farm']:.1f}/{r['crash']:.1f}/{r['gank']:.1f}]"
                          for r in d["baselines"].values()]) for n, d in res.items()]), ""]
    v1 = res[rungs[0]].get("v1_baselines")
    if v1:
        md += ["Original Phase 1 baselines on rung 0 (reproduces the original report): " +
               ", ".join(f"{k} {v:.2f}" for k, v in v1.items()), ""]
    save("before_after", {"horizon": K, "rungs": res}, "\n".join(md), REVISION)
    return res


if __name__ == "__main__":
    run()
