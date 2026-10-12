# results/m4_calibration — M4.2b exploratory calibration

Report: [docs/m4_calibration_report.md](../../docs/m4_calibration_report.md). Driver:
`lanegame/experiments/m4/calibration.py`. The M4 defaults are **not** changed; every candidate is
`GameConfig.m4(gank_damage_position=g, item_mult=i)` with tag `g{g}_i{i}` (baseline `g0.60_i1.20`).

| File | Content |
|---|---|
| `budget.json` | ledger: environment turns, wall seconds and summed worker CPU seconds per stage, and totals (cap 2,000,000 turns / 900 s) |
| `summary.json` | headline numbers of every stage (built from the stage files, no simulation: `--stage summary`) |
| `stage_A.json` | Phase A: gank damage 0.60 / 0.40 / 0.30; 4 matchups vs hold_farmer; per-agent counters (ganks, trials per gank, deaths, recalls, farm, XP, tower damage, plates, exposure) and discounted payoff components |
| `stage_B1.json` | Phase B1: intrinsic item value (+1 tier, gold 0, 40- and 200-turn windows) and the analytic hits-to-kill table |
| `stage_B2.json` | Phase B2: realized item value from s0 (purchase turns, unequal-item share, lead spells, early-buyer follow-up) |
| `stage_R.json` | affordability × (Recall now − Hold now), 3 states × HP 4/6, 6/6 |
| `stage_CD.json` | Phases C and D: baseline + 3 combined candidates, 11 matchups incl. the Shove → Recall agent |
| `stage_DCF.json` | Phase D counterfactual plans (farm / Recall now / Shove then Recall / Shove) from 5 reachable states × HP × gold |
| `stage_F.json` | finalist confirmation at W = 1,000 (30 seeds) and snowballing; tail bounds per configuration |
| `stage_FB.json` | Freeze-breaking keep-reach prototype: parked table vs design, and its strategic effect |
| `stage_S4.json` | targeted S4 checks of g0.40_i1.35: trading dominance and a reduced Recall sign pattern |

`budget.json` also keeps a discarded first run of stage A (`A_invalid_seat_seeds`): the seat-B seed stream was
advanced by reusing one `SeedSequence`, so the mirror matchup did not score 0. It was fixed and rerun; the invalid run
counts against the budget.

Reproduce (in this order; delete `budget.json` first to start a fresh ledger):

```
python -m lanegame.experiments.m4.calibration --stage A
python -m lanegame.experiments.m4.calibration --stage B1
python -m lanegame.experiments.m4.calibration --stage B2
python -m lanegame.experiments.m4.calibration --stage R
python -m lanegame.experiments.m4.calibration --stage CD  --configs g0.40_i1.35,g0.30_i1.35,g0.40_i1.50
python -m lanegame.experiments.m4.calibration --stage DCF --configs g0.40_i1.35
python -m lanegame.experiments.m4.calibration --stage F   --configs g0.40_i1.20,g0.40_i1.35
python -m lanegame.experiments.m4.calibration --stage FB
python -m lanegame.experiments.m4.calibration --stage S4  --configs g0.40_i1.35
python -m lanegame.experiments.m4.calibration --stage summary
```

Tests: `python -m pytest tests/test_m4_calibration.py tests/test_m4_time.py`.

Recorded outcome (2026-10-12, this branch): default suite `python -m pytest tests` (slow tests skipped) — **687 passed,
4 skipped, 2 xfailed** (the two strict M4.2 expected failures, preserved) in 6 min 51 s. The full `--runslow` suite was
not run (pending).
