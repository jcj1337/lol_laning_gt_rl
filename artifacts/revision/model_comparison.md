# Candidate mechanics under the revised economy (K = 32)

Each row changes one mechanic of the revised default. Scenario checks are the revised scenarios (some test a specific rule of the default, so a variant that changes that rule fails them by design).

'distinct at 2K' enumerates the same variant at horizon 64: equal counts mean the set of distinct states has saturated (bounded state space); a larger count means it keeps growing with the horizon.

| variant | scenarios | distinct states | distinct at 2K | nodes | mixed | solve s | eq crashes (A) | eq P(Contest) | P(Contest) nodes | P(Freeze) legal |
|---|---|---|---|---|---|---|---|---|---|---|
| revised default (stack_creep, period 2) | 15/15 | 2480 | 2518 | 14849 | 1030 | 6.5 | 0 | 0.91 | 0.568 | 0.069 |
| Slow Push hold at river (original rule) | 14/15 | 9354 | 99928 | 37245 | 1983 | 13.9 | 0 | 0.91 | 0.5 | 0.045 |
| Slow Push creep, 1 cell / spawn interval | 10/15 | 684 | 684 | 6569 | 383 | 1.7 | 0 | 0.91 | 0.596 | 0.158 |
| Slow Push creep, 1 cell / 2 intervals | 15/15 | 1326 | 2142 | 9613 | 661 | 3.5 | 0 | 0.91 | 0.573 | 0.111 |
| Slow Push stack_creep, period 1 | 13/15 | 924 | 924 | 8857 | 669 | 2.6 | 0 | 0.91 | 0.591 | 0.09 |
| Slow Push half speed past river | 10/15 | 670 | 670 | 6408 | 458 | 1.7 | 0 | 0.91 | 0.573 | 0.136 |
| Slow Push free (no line) | 10/15 | 40 | 40 | 556 | 0 | 0.1 | 0 | 0.94 | 0.428 | 0 |
| Freeze legal from the river (original rule) | 15/15 | 2644 | 2680 | 16463 | 1245 | 8.3 | 0 | 0.91 | 0.563 | 0.105 |
| tower: 1 packet per turn (original rule) | 11/15 | 284 | 284 | 3199 | 350 | 1.2 | 0 | 0.91 | 0.546 | 0.208 |
| tower fire 2 HP per tick | 13/15 | 1002 | 1158 | 7275 | 542 | 2.8 | 0 | 0.91 | 0.583 | 0.117 |
| cap rule: delete overflow HP (original rule) | 15/15 | 1622 | 1622 | 11619 | 825 | 4.3 | 0 | 0.91 | 0.582 | 0.078 |
| queued packets stand idle (do not fight with the front) | 15/15 | 3838 | 37822 | 18293 | 1253 | 8.6 | 0 | 0.91 | 0.543 | 0.066 |
| defenders engage crashed minions under tower | 11/15 | 612 | 612 | 5453 | 379 | 2.1 | 0 | 0.91 | 0.419 | 0.158 |
| defenders engage + tower fire 2 HP per tick | 11/15 | 476 | 476 | 4553 | 435 | 2.1 | 0 | 0.91 | 0.485 | 0.171 |
| HP units W = 2 (original resolution) | 12/15 | 107 | 107 | 1119 | 124 | 0.4 | 0 | 0.97 | 0.451 | 0.427 |
| attrition: linear (1 per tick) | 13/15 | 2760 | 13302 | 14815 | 949 | 6.7 | 0 | 0.91 | 0.645 | 0.054 |
| attrition: Lanchester ceil | 15/15 | 2017 | 3743 | 12198 | 840 | 5.2 | 0 | 0.72 | 0.516 | 0.102 |

## Reachable events from the standard start (number of reachable transitions)

| variant | crash_small | crash_medium | crash_large | crash_onto_siege | bounce | freeze_held | freeze_broken | minions_deleted |
|---|---|---|---|---|---|---|---|---|
| revised default (stack_creep, period 2) | 9372 | 5592 | 1840 | 3576 | 8176 | 3392 | 2632 | 0 |
| Slow Push hold at river (original rule) | 33668 | 29004 | 33384 | 62004 | 31092 | 9780 | 16752 | 0 |
| Slow Push creep, 1 cell / spawn interval | 3508 | 1500 | 0 | 0 | 876 | 3972 | 836 | 0 |
| Slow Push creep, 1 cell / 2 intervals | 5856 | 2184 | 360 | 496 | 2940 | 3428 | 1352 | 0 |
| Slow Push stack_creep, period 1 | 4672 | 1500 | 0 | 0 | 1104 | 2396 | 840 | 0 |
| Slow Push half speed past river | 3416 | 1724 | 0 | 0 | 812 | 5564 | 836 | 0 |
| Slow Push free (no line) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Freeze legal from the river (original rule) | 9584 | 5624 | 1852 | 3576 | 12464 | 14164 | 4252 | 0 |
| tower: 1 packet per turn (original rule) | 4644 | 1520 | 236 | 128 | 0 | 1748 | 744 | 0 |
| tower fire 2 HP per tick | 6000 | 2900 | 720 | 592 | 640 | 2384 | 1180 | 0 |
| cap rule: delete overflow HP (original rule) | 5756 | 3664 | 300 | 0 | 3812 | 2372 | 1176 | 612 |
| queued packets stand idle (do not fight with the front) | 14640 | 9700 | 4520 | 13508 | 10216 | 4928 | 4996 | 0 |
| defenders engage crashed minions under tower | 9240 | 2888 | 720 | 4240 | 0 | 2204 | 1084 | 0 |
| defenders engage + tower fire 2 HP per tick | 6752 | 2076 | 444 | 1376 | 0 | 2000 | 900 | 0 |
| HP units W = 2 (original resolution) | 1448 | 856 | 0 | 0 | 0 | 700 | 372 | 0 |
| attrition: linear (1 per tick) | 7280 | 3740 | 1288 | 980 | 7044 | 2408 | 1812 | 0 |
| attrition: Lanchester ceil | 9604 | 4164 | 1276 | 2980 | 6260 | 3888 | 2104 | 0 |

## Failed scenario checks

- **revised default (stack_creep, period 2)**: none
- **Slow Push hold at river (original rule)**: `stacked_wave_creeps` (A front position by turn: [0, 0, 0, 0])
- **Slow Push creep, 1 cell / spawn interval**: `slowpush_stacks` (A front hp by turn: [3, 5, 3, 5, 3, 5, 3, 5]); `unstacked_wave_waits` (positions of A's unstacked front by tick: [1, 1, 1]); `stacked_wave_creeps` (A front position by turn: [1, 1, -2, 0]); `tower_farm_contested` (A HP killed by B's tower 12; B farm opportunity there 4.75, collected 3.634 (76.5%)); `tower_farm_safe` (A HP killed by B's tower 12; B farm opportunity there 4.75, collected 3.800 (80.0%))
- **Slow Push creep, 1 cell / 2 intervals**: none
- **Slow Push stack_creep, period 1**: `slowpush_stacks` (A front hp by turn: [4, 6, 5, 7, 6, 8, 4, 4]); `stacked_wave_creeps` (A front position by turn: [1, 1, -2, 0])
- **Slow Push half speed past river**: `slowpush_stacks` (A front hp by turn: [3, 5, 3, 5, 3, 5, 3, 5]); `unstacked_wave_waits` (positions of A's unstacked front by tick: [1, 1, -1, 0]); `stacked_wave_creeps` (A front position by turn: [1, 0, 1, 0]); `tower_farm_contested` (A HP killed by B's tower 12; B farm opportunity there 4.00, collected 3.060 (76.5%)); `tower_farm_safe` (A HP killed by B's tower 12; B farm opportunity there 4.00, collected 3.200 (80.0%))
- **Slow Push free (no line)**: `slowpush_stacks` (A front hp by turn: [2, 3, 1, 3, 1, 3, 1, 3]); `unstacked_wave_waits` (positions of A's unstacked front by tick: [1, -2, -1, 0]); `stacked_wave_creeps` (A front position by turn: [-2, 0, 2, 0]); `tower_farm_contested` (A HP killed by B's tower 12; B farm opportunity there 4.75, collected 3.634 (76.5%)); `tower_farm_safe` (A HP killed by B's tower 12; B farm opportunity there 4.75, collected 3.800 (80.0%))
- **Freeze legal from the river (original rule)**: none
- **tower: 1 packet per turn (original rule)**: `large_crash_persists` (crash hp 12; A mass under B tower by turn [1, 0, 0, 0, 0, 0, 0, 0]); `tower_farm_contested` (A HP killed by B's tower 0; B farm opportunity there 3.00, collected 2.295 (76.5%)); `tower_farm_safe` (A HP killed by B's tower 0; B farm opportunity there 3.00, collected 2.400 (80.0%)); `bounce` (B front hp [4, 8, 7, 10, 9, 12, 11, 11, 10, 10]; B front global position [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]; A mass under B tower [11, 10, 9, 8, 7, 6, 5, 4, 3, 2])
- **tower fire 2 HP per tick**: `large_crash_persists` (crash hp 12; A mass under B tower by turn [10, 6, 2, 0, 0, 0, 0, 0]); `bounce` (B front hp [4, 8, 7, 6, 8, 6, 4, 6, 4, 4]; B front global position [3, 3, 3, 2, 1, 0, 0, 0, 0, 0]; A mass under B tower [8, 4, 0, 0, 0, 0, 0, 0, 0, 0])
- **cap rule: delete overflow HP (original rule)**: none
- **queued packets stand idle (do not fight with the front)**: none
- **defenders engage crashed minions under tower**: `large_crash_persists` (crash hp 12; A mass under B tower by turn [11, 9, 5, 2, 0, 0, 0, 0]); `tower_farm_contested` (A HP killed by B's tower 8; B farm opportunity there 3.00, collected 2.295 (76.5%)); `tower_farm_safe` (A HP killed by B's tower 8; B farm opportunity there 3.00, collected 2.400 (80.0%)); `bounce` (B front hp [0, 4, 2, 4, 3, 1, 3, 1, 3, 1]; B front global position [4, 3, 4, 2, 1, 2, 1, 2, 1]; A mass under B tower [8, 6, 3, 0, 0, 0, 0, 0, 0, 0])
- **defenders engage + tower fire 2 HP per tick**: `large_crash_persists` (crash hp 12; A mass under B tower by turn [10, 6, 0, 0, 0, 0, 0, 0]); `tower_farm_contested` (A HP killed by B's tower 10; B farm opportunity there 3.00, collected 2.295 (76.5%)); `tower_farm_safe` (A HP killed by B's tower 10; B farm opportunity there 3.00, collected 2.400 (80.0%)); `bounce` (B front hp [0, 4, 2, 4, 3, 1, 3, 1, 3, 1]; B front global position [4, 2, 4, 2, 1, 2, 1, 2, 1]; A mass under B tower [6, 2, 0, 0, 0, 0, 0, 0, 0, 0])
- **HP units W = 2 (original resolution)**: `slowpush_stacks` (A front hp by turn: [2, 3, 2, 3, 2, 3, 2, 2]); `large_crash_persists` (crash hp 6; A mass under B tower by turn [5, 3, 1, 0, 0, 0, 0, 0]); `bounce` (B front hp [2, 4, 3, 2, 1, 2, 2, 1, 2, 1]; B front global position [3, 3, 3, 1, 0, 4, 2, 0, 2, 0]; A mass under B tower [4, 2, 0, 0, 0, 0, 0, 0, 0, 0])
- **attrition: linear (1 per tick)**: `slowpush_stacks` (A front hp by turn: [4, 6, 4, 6, 4, 6, 4, 6]); `bounce` (B front hp [4, 8, 7, 10, 9, 12, 10, 8, 10, 8]; B front global position [3, 3, 3, 3, 3, 3, 2, 2, 2, 2]; A mass under B tower [10, 8, 6, 4, 2, 0, 0, 0, 0, 0])
- **attrition: Lanchester ceil**: none

## Baseline exploitability (as player A)

| variant | uniform | aggressive | freeze_contest | passive_safe | stack_builder | release_safe | contest_slowpush |
|---|---|---|---|---|---|---|---|
| revised default (stack_creep, period 2) | 10.08 | 24.86 | 9.45 | 9.42 | 5.78 | 7.69 | 6.59 |
| Slow Push hold at river (original rule) | 10.09 | 24.86 | 20.4 | 19.69 | 17.15 | 7.69 | 17.15 |
| Slow Push creep, 1 cell / spawn interval | 11.65 | 24.86 | 3.06 | 3.06 | 0.42 | 3.07 | 0.42 |
| Slow Push creep, 1 cell / 2 intervals | 10.88 | 24.86 | 8.03 | 8.44 | 5.05 | 7.27 | 5.05 |
| Slow Push stack_creep, period 1 | 10.07 | 24.86 | 5.46 | 5.46 | 4.74 | 7.69 | 0.87 |
| Slow Push half speed past river | 11.51 | 24.86 | 2.87 | 2.87 | 0.33 | 2.85 | 0.33 |
| Slow Push free (no line) | 11.03 | 24.86 | 2.85 | 2.85 | 0.33 | 2.85 | 0.33 |
| Freeze legal from the river (original rule) | 10.1 | 24.86 | 6.78 | 9.42 | 5.78 | 8.23 | 6.59 |
| tower: 1 packet per turn (original rule) | 10.08 | 24.86 | 7.61 | 8.19 | 5.14 | 7.69 | 5.14 |
| tower fire 2 HP per tick | 10.08 | 24.86 | 10.61 | 10.92 | 7.18 | 7.69 | 7.95 |
| cap rule: delete overflow HP (original rule) | 10.08 | 24.86 | 9.65 | 9.62 | 5.78 | 7.69 | 6.75 |
| queued packets stand idle (do not fight with the front) | 10.08 | 24.86 | 9.61 | 9.35 | 5.78 | 7.69 | 6.55 |
| defenders engage crashed minions under tower | 10.1 | 24.86 | 9.11 | 9.41 | 7.82 | 7.69 | 7.82 |
| defenders engage + tower fire 2 HP per tick | 10.1 | 24.86 | 9.5 | 9.82 | 8.2 | 7.69 | 8.2 |
| HP units W = 2 (original resolution) | 9.29 | 23.18 | 10.85 | 10.85 | 7.54 | 4.23 | 7.54 |
| attrition: linear (1 per tick) | 10.03 | 24.86 | 13.03 | 13.05 | 8.38 | 7.69 | 8.38 |
| attrition: Lanchester ceil | 11.35 | 24.86 | 9.88 | 10.4 | 6.7 | 4.17 | 7.94 |
