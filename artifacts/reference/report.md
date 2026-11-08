# Reference generator results (`python -m lanegame.experiments.reference`)

Engine-computed numbers that docs/m4/laning_game_mathematical_reference.md quotes, on `GameConfig.m45_calibrated()`.

## Tables (`examples`)

| L | XP to next | cumulative XP | max HP (cHP) | HP unit | AP(L, 0) |
|---|---|---|---|---|---|
| 1 | 8 | 0 | 600 | 100 | 1.000 |
| 2 | 11 | 8 | 630 | 105 | 1.050 |
| 3 | 14 | 19 | 662 | 110 | 1.103 |
| 4 | 17 | 33 | 695 | 116 | 1.158 |
| 5 | 20 | 50 | 729 | 122 | 1.216 |
| 6 | 23 | 70 | 766 | 128 | 1.276 |
| 7 | 26 | 93 | 804 | 134 | 1.340 |
| 8 | 29 | 119 | 844 | 141 | 1.407 |
| 9 | 32 | 148 | 886 | 148 | 1.478 |
| 10 | 35 | 180 | 931 | 155 | 1.551 |
| 11 | 38 | 215 | 977 | 163 | 1.629 |
| 12 | 41 | 253 | 1026 | 171 | 1.710 |
| 13 | 44 | 294 | 1078 | 180 | 1.796 |
| 14 | 47 | 338 | 1131 | 189 | 1.886 |
| 15 | 50 | 385 | 1188 | 198 | 1.980 |
| 16 | 53 | 435 | 1247 | 208 | 2.079 |
| 17 | 56 | 488 | 1310 | 218 | 2.183 |
| 18 | - | 544 | 1375 | 229 | 2.292 |

| S invested | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 | 100 | 128 | 200 | 300 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M(S) | 1.000 | 1.045 | 1.086 | 1.158 | 1.275 | 1.439 | 1.634 | 1.750 | 1.829 | 1.943 | 2.000 | 2.092 | 2.162 | 2.236 |

Wave capacity C and the Thin / Slow Push cap round(C / 2) by level and invested resources (Last Hit is 1 at every power):

| L \ S | 0 | 8 | 16 | 32 | 64 | 128 | 300 |
|---|---|---|---|---|---|---|---|
| 1 | 4 / 2 | 5 / 3 | 6 / 3 | 7 / 4 | 7 / 4 | 8 / 4 | 9 / 5 |
| 3 | 4 / 2 | 6 / 3 | 6 / 3 | 7 / 4 | 8 / 4 | 9 / 5 | 10 / 5 |
| 6 | 5 / 3 | 7 / 4 | 7 / 4 | 8 / 4 | 9 / 5 | 10 / 5 | 11 / 6 |
| 9 | 6 / 3 | 8 / 4 | 9 / 5 | 10 / 5 | 11 / 6 | 12 / 6 | 13 / 7 |
| 11 | 7 / 4 | 8 / 4 | 9 / 5 | 11 / 6 | 12 / 6 | 13 / 7 | 14 / 7 |
| 13 | 7 / 4 | 9 / 5 | 10 / 5 | 12 / 6 | 13 / 7 | 14 / 7 | 16 / 8 |
| 15 | 8 / 4 | 10 / 5 | 11 / 6 | 13 / 7 | 14 / 7 | 16 / 8 | 17 / 9 |
| 18 | 9 / 5 | 12 / 6 | 13 / 7 | 15 / 8 | 17 / 9 | 18 / 9 | 20 / 10 |

## Four wave actions from the neutral start (`examples`)

### A plays Last Hit + Hold, B plays Last Hit + Hold

| turn | budgets A/B | lane (global cells) | gold A/B | XP A/B | crash HP A/B | minion damage on A/B (cHP) |
|---|---|---|---|---|---|---|
| 1 | 1/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A12,B12,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 2 | 1/1 | -4:. | -3:. | -2:A8 | -1:. | +0:A8,B8,a,b | +1:. | +2:B8 | +3:. | +4:. | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 3 | 1/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A12,B12,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 4 | 1/1 | -4:. | -3:. | -2:A8 | -1:. | +0:A8,B8,a,b | +1:. | +2:B8 | +3:. | +4:. | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 5 | 1/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A12,B12,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 6 | 1/1 | -4:. | -3:. | -2:A8 | -1:. | +0:A8,B8,a,b | +1:. | +2:B8 | +3:. | +4:. | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 7 | 1/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A12,B12,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 8 | 1/1 | -4:. | -3:. | -2:A8 | -1:. | +0:A8,B8,a,b | +1:. | +2:B8 | +3:. | +4:. | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |

Totals: gold 3.600 / 3.600, XP 32 / 32, crashes 0 / 0, minion damage 0 / 0 cHP.

### A plays Thin + Hold, B plays Last Hit + Hold

| turn | budgets A/B | lane (global cells) | gold A/B | XP A/B | crash HP A/B | minion damage on A/B (cHP) |
|---|---|---|---|---|---|---|
| 1 | 1/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A12,B12,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 2 | 1/1 | -4:. | -3:. | -2:A8 | -1:. | +0:A8,B8,a,b | +1:. | +2:B8 | +3:. | +4:. | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 3 | 1/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A12,B12,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 4 | 1/1 | -4:. | -3:. | -2:A8 | -1:. | +0:A8,B8,a,b | +1:. | +2:B8 | +3:. | +4:. | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 5 | 1/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A12,B12,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 6 | 1/1 | -4:. | -3:. | -2:A8 | -1:. | +0:A8,B8,a,b | +1:. | +2:B8 | +3:. | +4:. | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 7 | 1/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A12,B12,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 8 | 1/1 | -4:. | -3:. | -2:A8 | -1:. | +0:A8,B8,a,b | +1:. | +2:B8 | +3:. | +4:. | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |

Totals: gold 3.600 / 3.600, XP 32 / 32, crashes 0 / 0, minion damage 0 / 0 cHP.

### A plays Slow Push + Hold, B plays Last Hit + Hold

| turn | budgets A/B | lane (global cells) | gold A/B | XP A/B | crash HP A/B | minion damage on A/B (cHP) |
|---|---|---|---|---|---|---|
| 1 | 2/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A12,B11,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 0.562 / 0.450 | 5 / 4 | 0 / 0 | 0 / 0 |
| 2 | 1/1 | -4:. | -3:. | -2:A8 | -1:. | +0:A9,B7,a,b | +1:. | +2:B8 | +3:. | +4:. | W:. | 0.450 / 0.338 | 4 / 3 | 0 / 0 | 0 / 0 |
| 3 | 1/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A13,B11,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 4 | 1/1 | -4:. | -3:. | -2:A8 | -1:. | +0:A10,B7,a,b | +1:. | +2:B8 | +3:. | +4:. | W:. | 0.450 / 0.338 | 4 / 3 | 0 / 0 | 0 / 0 |
| 5 | 1/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A14,B11,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 0.450 / 0.450 | 4 / 4 | 0 / 0 | 0 / 0 |
| 6 | 1/1 | -4:. | -3:. | -2:A8 | -1:. | +0:A11,B7,a,b | +1:. | +2:B8 | +3:. | +4:. | W:. | 0.450 / 0.338 | 4 / 3 | 0 / 0 | 0 / 0 |
| 7 | 1/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A15,B10,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 0.562 / 0.450 | 5 / 4 | 0 / 0 | 0 / 0 |
| 8 | 1/1 | -4:. | -3:. | -2:A8 | -1:. | +0:A13,B6,a,b | +1:. | +2:B8 | +3:. | +4:. | W:. | 0.450 / 0.225 | 4 / 2 | 0 / 0 | 0 / 0 |

Totals: gold 3.825 / 3.038, XP 34 / 27, crashes 0 / 0, minion damage 0 / 0 cHP.

### A plays Shove + Hold, B plays Last Hit + Hold

| turn | budgets A/B | lane (global cells) | gold A/B | XP A/B | crash HP A/B | minion damage on A/B (cHP) |
|---|---|---|---|---|---|---|
| 1 | 4/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A12,B9,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 0.787 / 0.450 | 7 / 4 | 0 / 0 | 0 / 0 |
| 2 | 4/1 | -4:. | -3:. | -2:A8 | -1:. | +0:A10,B2,a,b | +1:. | +2:B8 | +3:. | +4:. | W:. | 0.787 / 0.225 | 7 / 2 | 0 / 0 | 0 / 0 |
| 3 | 5/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A16,B1,a,b | +1:. | +2:. | +3:. | +4:B8 | W:. | 1.012 / 0.225 | 9 / 2 | 0 / 0 | 0 / 0 |
| 4 | 5/1 | -4:. | -3:. | -2:A8 | -1:. | +0:b | +1:A13,a | +2:B5 | +3:. | +4:. | W:. | 0.450 / 0.338 | 4 / 3 | 0 / 0 | 0 / 0 |
| 5 | 5/1 | -4:A8 | -3:. | -2:. | -1:. | +0:. | +1:A19,a | +2:b | +3:. | +4:B8 | W:. | 0.562 / 0.225 | 5 / 2 | 0 / 0 | 0 / 86 |
| 6 | 5/1 | -4:. | -3:. | -2:A8 | -1:. | +0:. | +1:a | +2:[A14],b | +3:B4 | +4:. | W:. | 0.000 / 0.500 | 4 / 5 | 19 / 0 | 0 / 0 |
| 7 | 5/1 | -4:A8 | -3:. | -2:. | -1:. | +0:A8,a | +1:. | +2:[A6],b | +3:B9 | +4:. | W:. | 0.000 / 0.800 | 0 / 8 | 0 / 0 | 0 / 0 |
| 8 | 5/1 | -4:. | -3:. | -2:A8 | -1:. | +0:. | +1:a | +2:[A7],b | +3:B7 | +4:. | W:. | 0.000 / 0.700 | 2 / 7 | 8 / 0 | 0 / 0 |

Totals: gold 3.600 / 3.463, XP 38 / 33, crashes 2 / 0, minion damage 0 / 86 cHP.

The same four A actions against B on Shove + Hold (A − B gold over 8 turns): Last Hit -0.1375, Thin -0.4000, Slow Push -0.4125, Shove +0.0000.

## Freezes (`examples`)

Turns of 40 a freeze holds from the M4.5b freeze start (frozen-on side on Last Hit + Hold at its +1):

| level | gap | freezer rule | held 5 HP | held 8 HP | held 12 HP | held 14 HP | held 20 HP |
|---|---|---|---|---|---|---|---|
| 4 | equal | controller | 18 | 20 | 4 | 4 | 2 |
| 4 | equal | thin | 18 | 20 | 26 | 6 | 2 |
| 4 | large | controller | 18 | 20 | 6 | 30 | 2 |
| 4 | large | thin | 18 | 20 | 24 | 24 | 2 |
| 10 | equal | controller | 18 | 20 | 28 | 24 | 2 |
| 10 | equal | thin | 18 | 20 | 22 | 22 | 2 |
| 10 | large | controller | 18 | 20 | 26 | 24 | 2 |
| 10 | large | thin | 18 | 20 | 22 | 22 | 2 |
| 16 | equal | controller | 18 | 20 | 24 | 26 | 30 |
| 16 | equal | thin | 18 | 20 | 22 | 22 | 24 |
| 16 | large | controller | 18 | 20 | 24 | 26 | 30 |
| 16 | large | thin | 18 | 20 | 22 | 22 | 24 |
| 18 | equal | controller | 18 | 20 | 24 | 24 | 26 |
| 18 | equal | thin | 18 | 20 | 22 | 22 | 24 |
| 18 | large | controller | 18 | 20 | 24 | 24 | 26 |
| 18 | large | thin | 18 | 20 | 22 | 22 | 24 |

One walk-up Shove against the freeze (share of chance draws after which the freeze still holds):

| held HP | level | C of the breaker | kept passively | guarded | guard: A / B bars lost |
|---|---|---|---|---|---|
| 5 | 3 | 4 | 1.00 | 1.00 | 0.142 / 0.192 |
| 5 | 4 | 5 | 1.00 | 1.00 | 0.115 / 0.188 |
| 5 | 5 | 5 | 1.00 | 1.00 | 0.105 / 0.187 |
| 5 | 6 | 7 | 0.00 | 1.00 | 0.079 / 0.179 |
| 5 | 7 | 7 | 0.00 | 1.00 | 0.071 / 0.179 |
| 5 | 8 | 8 | 0.00 | 0.00 | 0.056 / 0.174 |
| 5 | 10 | 10 | 0.00 | 0.00 | 0.042 / 0.170 |
| 5 | 12 | 11 | 0.00 | 0.00 | 0.033 / 0.168 |
| 5 | 15 | 14 | 0.00 | 0.00 | 0.022 / 0.164 |
| 12 | 3 | 4 | 0.00 | 1.00 | 0.274 / 0.180 |
| 12 | 4 | 5 | 0.00 | 1.00 | 0.225 / 0.176 |
| 12 | 5 | 5 | 0.00 | 1.00 | 0.200 / 0.176 |
| 12 | 6 | 7 | 0.00 | 1.00 | 0.151 / 0.167 |
| 12 | 7 | 7 | 0.00 | 1.00 | 0.134 / 0.167 |
| 12 | 8 | 8 | 0.00 | 0.00 | 0.107 / 0.162 |
| 12 | 10 | 10 | 0.00 | 0.00 | 0.079 / 0.158 |
| 12 | 12 | 11 | 0.00 | 0.00 | 0.060 / 0.156 |
| 12 | 15 | 14 | 0.00 | 0.00 | 0.031 / 0.152 |
| 20 | 3 | 4 | 0.00 | 1.00 | 0.406 / 0.162 |
| 20 | 4 | 5 | 0.00 | 1.00 | 0.331 / 0.158 |
| 20 | 5 | 5 | 0.00 | 1.00 | 0.295 / 0.158 |
| 20 | 6 | 7 | 0.00 | 1.00 | 0.221 / 0.149 |
| 20 | 7 | 7 | 0.00 | 1.00 | 0.198 / 0.150 |
| 20 | 8 | 8 | 0.00 | 0.00 | 0.156 / 0.144 |
| 20 | 10 | 10 | 0.00 | 0.00 | 0.114 / 0.141 |
| 20 | 12 | 11 | 0.00 | 0.00 | 0.087 / 0.139 |
| 20 | 15 | 14 | 0.00 | 0.00 | 0.048 / 0.135 |

## One-turn kills, retaliation and low HP (`examples`)

| case | taken (cHP) | died |
|---|---|---|
| combination, own hit False | 700 | True |
| combination, own hit True | 850 | True |
| gank + positional fire of a 8-HP front | 450 | False |
| gank + positional fire of a 16-HP front | 550 | False |
| gank + positional fire of a 24-HP front | 650 | True |
| double hit, L18 S300 on L10 (ratio 3.19) | 991 | True |
| double hit, L18 S128 on L10 (ratio 2.95) | 917 | False |
| double hit, L18 S300 on L11 (ratio 3.04) | 991 | True |
| double hit, L12 S64 on L1 (ratio 3.13) | 626 | True |
| double hit, L10 S16 on L1 (ratio 2.23) | 447 | False |
| double hit, L18 S0 on L1 (ratio 2.29) | 458 | False |

Retaliation on a level-2 trader: from its -2, 46 cHP; from its -1, 139 cHP.

| level, invested | A plays | B plays | P(A dies this turn) | of it, B's trade alone |
|---|---|---|---|---|
| 3, 0 | Last Hit + Hold | Trade | 0.220 | 0.150 |
| 3, 0 | Last Hit + Hold | Hold | 0.079 | 0.000 |
| 3, 0 | Shove + Hold | Trade | 0.236 | 0.150 |
| 3, 0 | Shove + Hold | Hold | 0.096 | 0.000 |
| 3, 0 | Last Hit + Safe | Trade | 0.017 | 0.000 |
| 3, 0 | Last Hit + Safe | Hold | 0.017 | 0.000 |
| 9, 15 | Last Hit + Hold | Trade | 0.203 | 0.150 |
| 9, 15 | Last Hit + Hold | Hold | 0.051 | 0.000 |
| 9, 15 | Shove + Hold | Trade | 0.215 | 0.150 |
| 9, 15 | Shove + Hold | Hold | 0.062 | 0.000 |
| 9, 15 | Last Hit + Safe | Trade | 0.004 | 0.000 |
| 9, 15 | Last Hit + Safe | Hold | 0.004 | 0.000 |
| 14, 33 | Last Hit + Hold | Trade | 0.203 | 0.150 |
| 14, 33 | Last Hit + Hold | Hold | 0.051 | 0.000 |
| 14, 33 | Shove + Hold | Trade | 0.215 | 0.150 |
| 14, 33 | Shove + Hold | Hold | 0.062 | 0.000 |
| 14, 33 | Last Hit + Safe | Trade | 0.004 | 0.000 |
| 14, 33 | Last Hit + Safe | Hold | 0.004 | 0.000 |
| 18, 55 | Last Hit + Hold | Trade | 0.186 | 0.150 |
| 18, 55 | Last Hit + Hold | Hold | 0.016 | 0.000 |
| 18, 55 | Shove + Hold | Trade | 0.193 | 0.150 |
| 18, 55 | Shove + Hold | Hold | 0.020 | 0.000 |
| 18, 55 | Last Hit + Safe | Trade | 0.000 | 0.000 |
| 18, 55 | Last Hit + Safe | Hold | 0.000 | 0.000 |

## Window strength (`examples`)

| state | S_A / S_B | Thin need A / B | Slow Push need A / B |
|---|---|---|---|
| neutral (turn 9: A8 v B8 at the lane cell) | 16 / 16 | 0 / 0 | 1 / 1 |
| the same with B's front at 24 HP | 16 / 32 | 16 / 0 | 17 / 0 |
| freeze (turn 51: A8 in A's tower cell, B5 held at A's -1) | 12 / 11 | 0 / 1 | 0 / 2 |

## A constructed turn with farm, a plate and a kill (`examples`)

Before: `-4:. | -3:. | -2:. | -1:. | +0:A8,b | +1:a | +2:[A10] | +3:B8 | +4:. | W:.`. After: `-4:. | -3:. | -2:. | -1:. | +0:. | +1:. | +2:[A10],a | +3:B1 | +4:. | W:.`. Actions SH-H+P / LH-H.

|  | A | B |
|---|---|---|
| farm | 0.7875 | 0.8000 |
| plate gold | 1.0000 | 0.0000 |
| kills | 2.0000 | 0.0000 |
| earned | 3.7875 | 0.8000 |
| XP | 7 | 8 |
| damage taken (cHP) | 0 | 525 |
| died | False | True |

r_A = +2.9875 (29,875,000 ledger units).

## Towers (`examples`)

Minion hits of an undisturbed crash (units, by the tick of its first turn): 8 HP: 2 or 2; 16 HP: 5 or 7; 24 HP: 12 or 13.

| level | invested | AP* | C | Pressure (centi-units) | Pressure turns to exhaust |
|---|---|---|---|---|---|
| 1 | 0 | 1.000 | 4 | 400 | 15 |
| 3 | 0 | 1.103 | 4 | 441 | 14 |
| 5 | 0 | 1.216 | 5 | 486 | 13 |
| 9 | 15 | 2.102 | 8 | 841 | 8 |
| 9 | 16 | 2.127 | 9 | 851 | 8 |
| 14 | 32 | 3.081 | 12 | 1232 | 5 |
| 14 | 33 | 3.098 | 12 | 1239 | 5 |
| 18 | 0 | 2.292 | 9 | 917 | 7 |
| 18 | 55 | 4.098 | 16 | 1639 | 4 |
| 18 | 128 | 4.584 | 18 | 1834 | 4 |

## Evaluation bounds (`examples`)

| preset | Omega | terms a / b/(1-gamma) / kills / plates | per-turn bound | gamma^50 Omega | gamma^150 Omega | gamma^300 Omega | gamma^460 Omega | gamma^500 Omega | gamma^1000 Omega |
|---|---|---|---|---|---|---|---|---|---|
| m45_calibrated | 145.11 | 46.35 / 45.00 / 50.76 / 3.00 | 11.75 | 87.79 | 32.13 | 7.116 | 1.425 | 0.9534 | 0.006264 |
| m4_lasthit | 107.04 | 46.35 / 45.00 / 12.69 / 3.00 | 10.25 | 64.76 | 23.7 | 5.249 | 1.051 | 0.7033 | 0.004621 |

## Legality, the observation and the wave choice (`actions`: 200 games of 200 turns of uniform random legal play)

Codes that can ever be legal (exhaustive): 27, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 21, 22, 23, 30, 45, 46, 47, 48, 49, 51, 53]. Legal-set sizes in play: {'1': 8973, '16': 1825, '17': 42078, '18': 23605, '24': 906, '25': 2613}; mismatches against the rule: 0 of 80000.

Observation: range [0.0, 1.0], decode mismatches 0, 3068 observations at W. Arrivals at W by code: {'30': 2725}. Absence set by a death: {'1-6: 3': 257, '13-18: 5': 867, '7-12: 4': 504}.

| lane actions | n | end cell depends on the wave slot |
|---|---|---|
| all | 65818 | 6.0% |
| Yield | 16302 | 1.5% |
| turns 0-49 | 16616 | 1.8% |
| Hold | 16102 | 14.4% |
| Trade | 16169 | 6.0% |
| Safe | 16105 | 2.4% |
| Hold + Pressure | 570 | 0.0% |
| Trade + Pressure | 570 | 0.0% |
| turns 50-99 | 16513 | 4.9% |
| turns 100-149 | 16307 | 7.8% |
| turns 150-199 | 16382 | 9.4% |

Scripted Last Hit + Hold, median turn of levels 2 / 6 / 11 / 18: m45_calibrated 5 / 21 / 59 / 147; m4_lasthit 5 / 21 / 58 / 141.

## Self-play of `sp:0` (`selfplay`: 32 games of 1000 turns from s0, both seats)

| turn | 50 | 94 | 100 | 150 | 200 | 300 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|
| level | 9.4 | 13.6 | 14.0 | 17.7 | 18.0 | 18.0 | 18.0 | 18.0 |
| earned | 20 | 37 | 40 | 60 | 80 | 119 | 197 | 390 |
| invested | 15 | 33 | 35 | 55 | 74 | 114 | 191 | 384 |
| M(S) | 1.42 | 1.64 | 1.65 | 1.79 | 1.87 | 1.97 | 2.08 | 2.20 |
| tau dealt | 1,220 | 2,405 | 2,614 | 3,960 | 4,588 | 5,329 | 5,949 | 6,000 |

Median turn of level 14: 94; level 18: 149; both champions at level 18: 150.5. A side's plates fall at median turns 82.5 / 129.5 / 191.5; both towers are exhausted at median turn 314.5.

Per side and 1,000 turns: deaths 7.1, recalls 43.2, wards 13.8, ward_placed 13.2, at_W 14.1. Earned gold: farm 95.6%, plate 0.8%, kill 3.6%. Pressure supplies 92.1% of the tower damage dealt before a tower is exhausted.

In the lane phase, turns 0-93: farm collected 0.756, away 0.064, channel_or_ward 0.009, out_of_reach 0.040, rate 0.131; XP gained 0.919, away 0.064, out_of_range 0.017, discarded 0.000; per 1,000 turns connected 140.8, dive 18.0, adjacent 0.7, tower_shot 18.5, ganked 62.0, interrupted 24.6, safe 125.5, retreat 27.1, minion_turns 90.9, ward_hit 0.0.

In all turns: farm collected 0.746, away 0.062, channel_or_ward 0.009, out_of_reach 0.052, rate 0.130; XP gained 0.136, away 0.062, out_of_range 0.017, discarded 0.785; per 1,000 turns connected 151.0, dive 32.0, adjacent 3.1, tower_shot 33.6, ganked 58.5, interrupted 34.1, safe 199.4, retreat 23.5, minion_turns 106.3, ward_hit 0.6.

Recalls and deaths: recall_attempts 77.3, recalls 43.2, recall_interrupted 34.1, deaths 7.1 per 1,000 turns; 44% of Recall attempts interrupted; deaths by_trade 81%, by_gank 21%, by_tower 8%, by_minions 7%, on_interrupted_recall 53%, killer_in_tower_cell 64%, killer_shot 64%. Wave choices: last_hit 23.2%, shove 34.0%, thin 20.5%, slow_push 22.3%. Freezes: episodes_per_100_seat_turns 2.55, mean_length 1.94, single_turn_share 0.63, four_or_more_share 0.13, pushed_in_share 0.55.

## The cost of a death and of a Recall (`death`: 24 states per phase, 16 paired rollouts of 300 turns, `sp:0` in both seats)

| phase | mean turn | mean level | death absence | Recall cost | death cost | death minus Recall |
|---|---|---|---|---|---|---|
| early | 37 | 7.7 | 3.7 | +0.22 ± 0.55 | +3.65 ± 0.45 | +3.43 ± 0.44 |
| mid | 105 | 14.4 | 5.0 | +0.04 ± 0.49 | +4.03 ± 0.67 | +3.98 ± 0.51 |
| late | 250 | 18.0 | 5.0 | -0.10 ± 0.47 | +3.51 ± 0.44 | +3.61 ± 0.42 |

Costs are discounted resources lost by the focal side (gamma 0.99), with 95% intervals over states. The death cost includes the 2 gold the opponent earns for the kill.
