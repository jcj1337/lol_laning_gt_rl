# S3 — Wave mechanics and the Freeze gate (M4.2)

Commit `df5f159451`; 163 reachable Freeze-legal scenarios; 50 paired rollouts (common random numbers) of 120 turns per scenario and policy pair; gamma = 0.99. Command: `python.exe C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\experiments\m4\freeze_gate.py --rollouts 50 --turns 120 --procs 10`.

## Gates

- **G1 (distinct)**: PASS — |F − SP| > 0.05 in ≥ 20% of the scenarios against: park_hold, park_safe, breaker, presser, stacker, adaptive_breaker.
- **G2 (non-trivial sign)**: PASS — Freeze ≥ +0.2 better than the best alternative somewhere against: park_hold, park_safe, breaker, presser, stacker, adaptive_breaker; ≥ 0.2 worse somewhere against: park_safe, breaker, presser, stacker, adaptive_breaker.
- **G3 (breakable)**: rule PASS — 60015 gate turns with a reaching Shove and blob + 4 ≥ 10, 0 held. Break windows below.

## Freeze against the alternatives, per attacker

| B policy | Freeze | Slow Push | cond. Shove | F − SP mean (min, max) | F ≠ SP | F vs best: better/equal/worse | B crashes (F / SP) | B plates (F / SP) | B gold (F / SP) | B XP (F / SP) |
|---|---|---|---|---|---|---|---|---|---|---|
| park_hold | 9.34 | 8.46 | -2.87 | +0.89 (+0.00, +12.61) | 37% | 60/103/0 | 2.71 / 2.97 | 0.92 / 0.91 | 19.0 / 20.1 | 385 / 391 |
| park_safe | 28.29 | 27.97 | 6.18 | +0.32 (-9.68, +2.77) | 36% | 56/104/3 | 2.66 / 2.97 | 0.94 / 0.91 | 0.0 / 0.0 | 382 / 393 |
| breaker | 10.63 | 10.39 | 6.98 | +0.24 (-0.88, +1.58) | 88% | 113/20/30 | 4.97 / 4.70 | 0.78 / 0.22 | 13.5 / 14.0 | 379 / 382 |
| presser | 9.54 | 9.61 | 6.68 | -0.07 (-0.96, +1.07) | 90% | 50/17/96 | 5.40 / 4.92 | 2.85 / 2.33 | 13.2 / 13.6 | 378 / 379 |
| stacker | 7.98 | 7.64 | -2.37 | +0.34 (-0.94, +12.61) | 36% | 37/105/21 | 4.25 / 4.13 | 0.88 / 0.87 | 20.1 / 20.7 | 392 / 388 |
| adaptive_breaker | 8.62 | 8.25 | -2.28 | +0.38 (-5.72, +9.16) | 89% | 75/18/70 | 4.75 / 3.75 | 2.48 / 0.92 | 15.2 / 19.8 | 383 / 382 |
| trader (extra) | 7.28 | 6.73 | 3.32 | +0.55 (-0.59, +19.24) | 34% | 39/52/72 | 2.82 / 2.97 | 0.92 / 0.91 | 11.7 / 12.6 | 333 / 338 |

Exposure and combat (means per rollout, Freeze defender / Slow Push defender):

| B policy | A gank occurrences | B gank occurrences | B gank HP | B trade HP taken | B tower-shot HP | A deaths | B deaths | B tower damage (centi) | Freeze chosen / maintained (turns) | B first crash (H = never) |
|---|---|---|---|---|---|---|---|---|---|---|
| park_hold | 5.35 / 5.49 | 7.64 / 7.54 | 29.89 / 29.48 | 39.01 / 38.79 | 0.00 / 0.00 | 0.53 / 0.60 | 5.05 / 4.90 | 1762 / 1744 | 68.1 / 50.4 | 85.2 / 85.4 |
| park_safe | 5.68 / 5.85 | 2.23 / 2.22 | 2.36 / 2.35 | 0.56 / 0.51 | 0.00 / 0.00 | 0.32 / 0.29 | 0.14 / 0.13 | 1815 / 1744 | 71.3 / 70.4 | 84.4 / 85.4 |
| breaker | 5.88 / 6.00 | 10.43 / 10.58 | 50.30 / 52.12 | 51.65 / 52.14 | 0.00 / 0.00 | 1.27 / 1.39 | 6.64 / 6.65 | 1985 / 1061 | 43.9 / 28.9 | 15.4 / 17.3 |
| presser | 5.70 / 5.93 | 10.52 / 10.58 | 50.77 / 52.06 | 48.20 / 51.09 | 0.00 / 0.00 | 1.23 / 1.37 | 6.44 / 6.59 | 5679 / 4944 | 45.3 / 30.8 | 15.4 / 17.3 |
| stacker | 5.61 / 5.66 | 7.94 / 7.81 | 32.48 / 31.56 | 40.94 / 41.22 | 0.00 / 0.00 | 0.73 / 0.73 | 5.14 / 5.12 | 1722 / 1700 | 55.0 / 40.2 | 84.6 / 85.0 |
| adaptive_breaker | 5.85 / 5.54 | 9.89 / 7.83 | 46.43 / 31.76 | 47.06 / 39.90 | 0.00 / 0.00 | 1.21 / 0.65 | 6.06 / 5.12 | 4921 / 1771 | 40.3 / 27.3 | 29.1 / 85.4 |
| trader | 5.42 / 5.40 | 8.67 / 8.68 | 35.57 / 35.47 | 39.26 / 38.27 | 36.27 / 35.08 | 5.73 / 5.97 | 7.61 / 7.43 | 1760 / 1744 | 49.9 / 29.1 | 85.2 / 85.4 |

## G3 break windows (scenarios with B's front ≥ 3 HP at A's −1)

| attacker | rollouts | broken ≤ 6 turns | ≤ 12 | within H | median first crash |
|---|---|---|---|---|---|
| breaker | 4500 | 0.5767 | 0.8029 | 0.9987 | 6 |
| presser | 4500 | 0.5767 | 0.8029 | 0.9987 | 6 |
| adaptive_breaker | 4500 | 0.4053 | 0.7036 | 0.9378 | 7 |

## S3a — the parked-front table on the implemented resolver (first crash; None = 24 turns parked)

| e, B wave | design: Freeze | implemented: Freeze | design: Slow Push | implemented: Slow Push |
|---|---|---|---|---|
| 2_SP | [None, None, None, None] | [None, None, None, None] | [None, None, None, None] | [None, None, None, None] |
| 3_SP | [None, None, None, None] | [None, None, None, None] | [None, None, None, None] | [None, None, None, None] |
| 4_SP | [None, None, None, None] | [None, None, None, None] | [None, None, None, None] | [None, None, None, None] |
| 5_SP | [None, None, None, None] | [None, None, None, None] | [0, None, None, None] | [0, None, None, None] |
| 6_SP | [12, 11, 10, 9] | [12, 11, 10, 9] | [0, 15, None, None] | [0, 15, None, None] |
| 7_SP | [8, 7, 10, 9] | [8, 7, 10, 9] | [0, 4, 14, 13] | [0, 4, 14, 13] |
| 8_SP | [8, 7, 6, 5] | [8, 7, 6, 5] | [0, 4, 3, 2] | [0, 4, 3, 2] |
| 9_SP | [4, 3, 6, 5] | [4, 3, 6, 5] | [0, 3, 3, 2] | [0, 3, 3, 2] |
| 2_SH | [None, None, None, None] | [None, None, None, None] | [0, 0, 0, 0] | [0, 0, 0, 0] |
| 3_SH | [6, 5, 6, 5] | [9, 8, 9, 8] | [0, 0, 0, 0] | [0, 0, 0, 0] |
| 4_SH | [4, 3, 4, 3] | [9, 8, 9, 8] | [0, 0, 0, 0] | [0, 0, 0, 0] |
| 5_SH | [4, 3, 4, 3] | [7, 6, 7, 6] | [0, 0, 0, 0] | [0, 0, 0, 0] |
| 6_SH | [0, 0, 0, 0] | [0, 6, 0, 6] | [0, 0, 0, 0] | [0, 0, 0, 0] |
| 7_SH | [0, 0, 0, 0] | [0, 4, 0, 4] | [0, 0, 0, 0] | [0, 0, 0, 0] |
| 8_SH | [0, 0, 0, 0] | [0, 2, 0, 2] | [0, 0, 0, 0] | [0, 0, 0, 0] |
| 9_SH | [0, 0, 0, 0] | [0, 2, 0, 2] | [0, 0, 0, 0] | [0, 0, 0, 0] |

## S3a — exhaustive lane comparison

15488 Freeze-legal (state, phase, side) situations in 20572 reachable pairs; Freeze differs from Slow Push (maintained case) in 1126 (7.3%; M2 reference 1126 of 15458, 7.3%). A Hold freezer can maintain the Freeze in 15488 of them (1126 that also differ).

## Freeze statistics of scripted play (per 100 turns)

| pairing | freeze_legal | freeze_chosen | freeze_maintained | held | broken | crashes_b |
|---|---|---|---|---|---|---|
| freezer vs hold_farmer | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| freezer vs g_breaker | 38.10 | 37.77 | 28.11 | 6.35 | 1.64 | 3.37 |
| freezer vs g_presser | 38.75 | 38.08 | 28.71 | 5.81 | 1.79 | 3.55 |
| freezer vs g_stacker | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| freezer vs aggressive_trader | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| hold_farmer vs hold_farmer | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
