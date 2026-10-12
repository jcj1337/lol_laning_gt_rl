# S2 — Progression speed (M4.2)

Commit `df5f159451`. 400 seeded games of 400 turns per matchup. Command: `python.exe C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\experiments\m4\progression.py --games 400 --turns 400 --procs 8`.

## Mutual hold_farmer: turn at which each level is first reached (both seats)

| level | q25 | median | q75 | reached | design target | S2 |
|---|---|---|---|---|---|---|
| 2 | 4 | 4 | 4 | 100% | 4 | within ±25% |
| 3 | 7 | 7 | 7 | 100% |  |  |
| 4 | 10 | 10 | 10 | 100% |  |  |
| 5 | 14 | 14 | 14 | 100% |  |  |
| 6 | 19 | 19 | 19 | 100% | 20 | within ±25% |
| 7 | 25 | 25 | 27 | 100% |  |  |
| 8 | 32 | 32 | 34 | 100% |  |  |
| 9 | 39 | 41 | 41 | 100% |  |  |
| 10 | 47 | 49 | 50 | 100% |  |  |
| 11 | 56 | 58 | 59 | 100% | 59 | within ±25% |
| 12 | 67 | 67 | 69 | 100% |  |  |
| 13 | 77 | 78 | 79.25 | 100% |  |  |
| 14 | 88 | 90 | 91 | 100% |  |  |
| 15 | 101 | 102 | 104 | 100% |  |  |
| 16 | 115 | 116 | 118 | 100% |  |  |
| 17 | 128 | 130 | 132 | 100% |  |  |
| 18 | 142.8 | 144 | 146 | 100% | 147 | within ±25% |

Verdict: **PASS** (medians of levels 2 / 6 / 11 / 18 within ±25% of 4 / 20 / 59 / 147). XP absorption (XP earned in range, including XP beyond the cap, / XP of all enemy HP dying): 95.1% / 95.1% (design assumed 94%).

## mutual hold_farmer

- deaths per game A 1.84, B 1.89; recalls per game A 7.05, B 7.01; farm per turn A 0.304, B 0.304; payoff to A (undiscounted, 400 turns) +0.03
- item purchases (turn of the purchase): tier 1 A median 44.5, B 41.0; tier 2 A 61.0, B 61.0

| turn | level gap A − B (mean) | sd |
|---|---|---|
| 30 | -0.00 | 0.05 |
| 60 | +0.00 | 0.42 |
| 100 | +0.03 | 0.60 |
| 150 | -0.02 | 0.24 |
| 200 | +0.00 | 0.00 |
| 300 | +0.00 | 0.00 |
| 399 | +0.00 | 0.00 |

## aggressive_trader (A) vs hold_farmer (B)

- deaths per game A 10.27, B 25.41; recalls per game A 11.02, B 12.16; farm per turn A 0.178, B 0.166; payoff to A (undiscounted, 400 turns) +12.11
- item purchases (turn of the purchase): tier 1 A median 35.0, B 31.0; tier 2 A 67.0, B 60.0

| turn | level gap A − B (mean) | sd |
|---|---|---|
| 30 | +0.33 | 0.57 |
| 60 | +0.46 | 0.63 |
| 100 | +0.94 | 0.61 |
| 150 | +1.27 | 0.56 |
| 200 | +0.01 | 0.12 |
| 300 | +0.00 | 0.00 |
| 399 | +0.00 | 0.00 |
