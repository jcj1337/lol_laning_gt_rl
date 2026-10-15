"""M4.5a commands (EXPERIMENTAL m45_unified(); docs/m4/m4_5/m4_5a_unified_mechanics_report.md):

    python -m lanegame.experiments.m45a demos      # viewer recordings results/m45a/trajectories/ (+ replay check)
    python -m lanegame.experiments.m45a smoke      # random / scripted soaks, throughput, damage table -> results/m45a/
    python -m lanegame.experiments.m45a movement   # the movement correction: scenarios, r1 vs r2 paired comparison
                                                   # -> results/m45a/movement_correction.{json,md}
    python -m lanegame.experiments.m45a all        # demos and smoke (not movement)
"""

from __future__ import annotations

import os
import sys
import time

from ...io import dump_json
from . import demos, movement, smoke

OUT = demos.OUT


def _report(summary: dict) -> str:
    L = ["# M4.5a smoke results (generated)", "",
         "Descriptive checks of the EXPERIMENTAL `GameConfig.m45_unified()`; not a calibration. Regenerate with "
         "`python -m lanegame.experiments.m45a all`.", ""]
    if "throughput" in summary:
        tp = summary["throughput"]
        L += ["## Throughput (uniform legal play, one process)", "", "| preset | turns / s |", "|---|---|"]
        L += [f"| `{k}` | {v:,.0f} |" for k, v in tp["turns_per_second"].items()]
        L += ["", f"M4.5a / aggro = {tp['ratio']:.2f}", ""]
    if "random" in summary:
        r = summary["random"]
        L += [f"## Uniform random legal play ({r['games']} games x {r['turns']} turns, invariants and the observation "
              "decoder checked every turn)", "", "| event | per champion-turn |", "|---|---|"]
        L += [f"| {k} | {v:.4f} |" for k, v in r["per_champion_turn"].items()]
        L += ["", "Mean minion damage in a turn with minion hits, as a share of the max-HP bar, by level:", "",
              "| level | turns | share |", "|---|---|---|"]
        L += [f"| {lv} | {d['turns']} | {d['mean_share_of_bar']:.3f} |"
              for lv, d in r["minion_damage_share_by_level"].items()]
        L.append("")
    if "damage" in summary:
        d = summary["damage"]
        L += [f"## Minion damage table (alpha {d['alpha']}, growth {d['growth']})", "", d["note"], "",
              "| level | invested | H(L) | m = 1 | m = 2 | m = 3 | m = 4 | m = 6 |", "|---|---|---|---|---|---|---|---|"]
        for row in d["rows"]:
            L.append(f"| {row['level']} | {row['invested']} | {row['hp_max']} | " + " | ".join(
                f"{row[f'm{m}'][0]} ({100 * row[f'm{m}'][1]:.1f}%)" for m in (1, 2, 3, 4, 6)) + " |")
        L.append("")
    if "scripted" in summary:
        s = summary["scripted"]
        L += [f"## Scripted pairings ({s['seeds']} paired seeds x {s['turns']} turns; mean ± 1.96 SE over seeds)", ""]
        cols = ("lead", "deaths_A", "deaths_B", "recalls_A", "recalls_B", "minion_cp_A", "minion_cp_B", "wards_A",
                "wards_placed_A", "ret_ticks_A", "pos_ticks_A", "ret_ticks_B", "pos_ticks_B")
        for preset, rows in s["rows"].items():
            L += [f"**`{preset}`**", "", "| pairing (A v B [start, gap]) | " + " | ".join(cols) + " |",
                  "|---" * (len(cols) + 1) + "|"]
            for name, row in rows.items():
                L.append(f"| {name} | " + " | ".join(f"{row[c][0]:.2f} ± {row[c][1]:.2f}" for c in cols) + " |")
            L.append("")
    if "demos" in summary:
        L += ["## Demonstration trajectories (viewer: `streamlit run lanegame/viewer/app.py`, Recorded trajectory, "
              "folder `results/m45a/trajectories`)", "", "| file | seed | turns | replay exact |", "|---|---|---|---|"]
        for f in summary["demos"]["files"]:
            v = f["verify"]
            L.append(f"| `{os.path.basename(f['path'])}` | {f['seed']} | {f['turns']} | "
                     f"{v['actions_equal'] and v['payoff_equal']} |")
        L.append("")
    return "\n".join(L)


def main(argv) -> None:
    what = argv[0] if argv else "all"
    os.makedirs(OUT, exist_ok=True)
    if what == "movement":
        d = movement.run()
        dump_json(d, os.path.join(OUT, "movement_correction.json"))
        with open(os.path.join(OUT, "movement_correction.md"), "w", encoding="utf-8") as f:
            f.write(movement.report(d))
        print(f"wrote {OUT}/movement_correction.json and .md ({d['elapsed_s']} s)")
        return
    path = os.path.join(OUT, "summary.json")
    summary = {}
    if os.path.exists(path):
        import json
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
    t0 = time.time()
    if what in ("demos", "all"):
        summary["demos"] = demos.run()
    if what in ("smoke", "all"):
        summary["throughput"] = smoke.throughput()
        summary["damage"] = smoke.damage_table()
        summary["random"] = smoke.random_soak()
        summary["scripted"] = smoke.scripted()
    summary["elapsed_s"] = round(time.time() - t0, 1)
    dump_json(summary, path)
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write(_report(summary))
    print(f"wrote {path} and report.md ({summary['elapsed_s']} s)")


if __name__ == "__main__":
    main(sys.argv[1:])
