# Experiment A: before vs. after (ablation ladder, K = 32)

Each rung adds one change to the previous rung. Rung 0 is the original model; rung 7 is the revised default. All numbers are exact (backward induction / exact forward path statistics).

| rung | change |
|---|---|
| 0 original model (v1) | - |
| 1 + farming economy | minion kills pay farm_per_hp x collection efficiency (Contest only) |
| 2 + Contest/Safe positioning | 6 combined actions; Safe caps exposure at own -1; flat Contest risk |
| 3 + stricter Freeze legality | Freeze only with enemy minions at own x <= -1 |
| 4 + mass-based tower fire | tower removes 1 HP/tick of crashed mass instead of 1 packet/turn |
| 5 + no minion deletion at the HP cap | full stacks queue reinforcements (which fight with them) instead of deleting HP |
| 6 + finer HP units (W = 4) | 4 HP per wave, same per-wave firepower: fights last a spawn interval |
| 7 + Slow Push 'stack_creep' = revised | unstacked waves wait at the river, stacked waves creep |

## State space, solver and equilibrium structure

| rung | decision nodes | distinct states | mixed nodes | solve s | P(Contest) avg node | P(Freeze) where legal | max gap |
|---|---|---|---|---|---|---|---|
| 0 original model (v1) | 905 | 71 | 52 | 0.11 | 1 | 0.738 | 6.7e-16 |
| 1 + farming economy | 905 | 71 | 54 | 0.11 | 1 | 0.738 | 4.4e-16 |
| 2 + Contest/Safe positioning | 1017 | 79 | 62 | 0.27 | 0.491 | 0.192 | 6.7e-16 |
| 3 + stricter Freeze legality | 801 | 61 | 30 | 0.13 | 0.461 | 0.276 | 6.7e-16 |
| 4 + mass-based tower fire | 1495 | 127 | 26 | 0.2 | 0.484 | 0.276 | 6.7e-16 |
| 5 + no minion deletion at the HP cap | 3969 | 755 | 146 | 1 | 0.533 | 0.177 | 6.7e-16 |
| 6 + finer HP units (W = 4) | 37245 | 9354 | 1983 | 13.12 | 0.5 | 0.045 | 1.8e-15 |
| 7 + Slow Push 'stack_creep' = revised | 14849 | 2480 | 1030 | 5.8 | 0.568 | 0.069 | 1.8e-15 |

## Equilibrium path (symmetric start; per player, totals over the game)

| rung | actions (A) | crashes A S/M/L | farm collected | farm available | farm missed | crash resources | gank loss | vision choices |
|---|---|---|---|---|---|---|---|---|
| 0 original model (v1) | F 0.47, SP 0.53 | 0.00/0.00/0.00 | 0 | 0 | 0 | 0 | 30.25 | 0 |
| 1 + farming economy | F 0.47, SH 0.03, SP 0.50 | 0.00/0.00/0.00 | 10.08 | 16 | 5.92 | 0 | 30.5 | 0 |
| 2 + Contest/Safe positioning | SH-C 0.03, SP-C 0.94, SP-S 0.03 | 0.00/0.00/0.00 | 10.08 | 16 | 5.92 | 0 | 35 | 0 |
| 3 + stricter Freeze legality | SH-C 0.03, SP-C 0.94, SP-S 0.03 | 0.00/0.00/0.00 | 10.08 | 16 | 5.92 | 0 | 35 | 0 |
| 4 + mass-based tower fire | SH-C 0.03, SP-C 0.94, SP-S 0.03 | 0.00/0.00/0.00 | 10.08 | 16 | 5.92 | 0 | 35 | 0 |
| 5 + no minion deletion at the HP cap | SH-C 0.03, SP-C 0.94, SP-S 0.03 | 0.00/0.00/0.00 | 10.08 | 16 | 5.92 | 0 | 35 | 0 |
| 6 + finer HP units (W = 4) | SH-C 0.06, SP-C 0.84, SP-S 0.09 | 0.00/0.00/0.00 | 9.65 | 15.5 | 5.85 | 0 | 34.7 | 0 |
| 7 + Slow Push 'stack_creep' = revised | SH-C 0.06, SP-C 0.84, SP-S 0.09 | 0.00/0.00/0.00 | 9.65 | 15.5 | 5.85 | 0 | 34.7 | 0 |

## Vision choice (mean probability of taking vision at A's crash-choice nodes)

| rung | small | medium | large | choice nodes S/M/L |
|---|---|---|---|---|
| 0 original model (v1) | 0.93 | 0 | 0.9 | 27/23/10 |
| 1 + farming economy | 0.93 | 0 | 0.9 | 27/23/10 |
| 2 + Contest/Safe positioning | 0 | 0 | 0 | 27/23/10 |
| 3 + stricter Freeze legality | 0 | 0 | nan | 27/23/0 |
| 4 + mass-based tower fire | 0 | 0 | nan | 77/34/0 |
| 5 + no minion deletion at the HP cap | 0.03 | 0 | 0.05 | 214/206/150 |
| 6 + finer HP units (W = 4) | 0 | 0 | 0 | 539/627/840 |
| 7 + Slow Push 'stack_creep' = revised | 0.1 | 0 | 0.3 | 202/117/82 |

## Reachable events from the standard start (number of reachable transitions)

| rung | crash_small | crash_medium | crash_large | crash_onto_siege | freeze_held | freeze_broken | bounce | siege_persists | safe_disengages | tower_farm | farm_contest_vs_safe | minions_deleted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 original model (v1) | 578 | 400 | 60 | 0 | 394 | 276 | 0 | 0 | 0 | 0 | 3575 | 126 |
| 1 + farming economy | 578 | 400 | 60 | 0 | 394 | 276 | 0 | 0 | 0 | 0 | 3575 | 126 |
| 2 + Contest/Safe positioning | 1744 | 1416 | 240 | 0 | 1852 | 720 | 0 | 0 | 11597 | 0 | 16432 | 504 |
| 3 + stricter Freeze legality | 1092 | 360 | 0 | 0 | 448 | 48 | 0 | 0 | 7228 | 0 | 10176 | 176 |
| 4 + mass-based tower fire | 1176 | 360 | 0 | 0 | 448 | 48 | 320 | 2048 | 12938 | 9232 | 21952 | 176 |
| 5 + no minion deletion at the HP cap | 6036 | 5456 | 4184 | 8828 | 2924 | 3152 | 1200 | 28552 | 38010 | 45700 | 69500 | 0 |
| 6 + finer HP units (W = 4) | 33668 | 29004 | 33384 | 62004 | 9780 | 16752 | 31092 | 395184 | 400774 | 495836 | 783180 | 0 |
| 7 + Slow Push 'stack_creep' = revised | 9372 | 5592 | 1840 | 3576 | 3392 | 2632 | 8176 | 75416 | 157754 | 125236 | 304876 | 0 |

## Exploitability of the revised baselines (as A) and its decomposition

Each cell: exploitability [farm / crash / gank part of the baseline's value against its best response]. When positioning is off, Safe actions fall back to the Contest action with the same wave management (so passive_safe = contest_slowpush and release_safe = aggressive on rungs 0-1).

| rung | uniform | aggressive | freeze_contest | passive_safe | stack_builder | release_safe | contest_slowpush |
|---|---|---|---|---|---|---|---|
| 0 original model (v1) | 11.86 [0.0/0.0/-11.9] | 19.45 [0.0/0.0/-19.5] | 1.85 [0.0/-2.0/0.2] | 4.85 [0.0/-2.0/-2.8] | 4.85 [0.0/-2.0/-2.8] | 19.45 [0.0/0.0/-19.5] | 4.85 [0.0/-2.0/-2.8] |
| 1 + farming economy | 13.97 [-2.3/0.1/-11.7] | 24.78 [-5.5/0.0/-19.2] | 5.39 [-5.0/0.0/-0.3] | 8.39 [-5.0/0.0/-3.3] | 8.39 [-5.0/0.0/-3.3] | 24.78 [-5.5/0.0/-19.2] | 8.39 [-5.0/0.0/-3.3] |
| 2 + Contest/Safe positioning | 11.71 [-6.2/-1.9/-3.7] | 23.18 [-5.5/0.0/-17.6] | 12.25 [-4.0/-2.0/-6.3] | 15.96 [-10.7/-2.0/-3.3] | 12.13 [-3.5/-2.0/-6.6] | 4.23 [-10.1/0.0/5.9] | 12.13 [-3.5/-2.0/-6.6] |
| 3 + stricter Freeze legality | 10.49 [-5.8/-1.7/-3.0] | 23.18 [-5.5/0.0/-17.6] | 15.96 [-10.7/-2.0/-3.3] | 15.96 [-10.7/-2.0/-3.3] | 12.13 [-3.5/-2.0/-6.6] | 4.23 [-10.1/0.0/5.9] | 12.13 [-3.5/-2.0/-6.6] |
| 4 + mass-based tower fire | 11.58 [-6.9/-1.7/-3.0] | 23.18 [-5.5/0.0/-17.6] | 17.32 [-12.0/-2.0/-3.3] | 17.32 [-12.0/-2.0/-3.3] | 13.66 [-5.0/-2.0/-6.6] | 4.23 [-10.1/0.0/5.9] | 13.66 [-5.0/-2.0/-6.6] |
| 5 + no minion deletion at the HP cap | 14.05 [-6.9/-2.9/-4.3] | 23.18 [-5.5/0.0/-17.6] | 18.51 [-8.7/-6.0/-3.8] | 18.38 [-9.6/-6.0/-2.8] | 14.75 [-1.4/-6.0/-7.4] | 4.23 [-10.1/0.0/5.9] | 14.75 [-1.4/-6.0/-7.4] |
| 6 + finer HP units (W = 4) | 10.09 [-6.0/0.0/-4.1] | 24.86 [-5.2/0.0/-19.6] | 20.40 [-8.9/-9.0/-2.4] | 19.69 [-10.6/-6.0/-3.1] | 17.15 [-3.4/-7.0/-6.7] | 7.69 [-11.7/0.0/4.0] | 17.15 [-3.4/-7.0/-6.7] |
| 7 + Slow Push 'stack_creep' = revised | 10.08 [-6.1/0.1/-4.1] | 24.86 [-5.2/0.0/-19.6] | 9.45 [-4.5/-5.0/0.1] | 9.42 [-5.4/-5.0/0.9] | 5.78 [2.5/-6.0/-2.2] | 7.69 [-11.7/0.0/4.0] | 6.59 [2.4/-7.0/-1.9] |

Original Phase 1 baselines on rung 0 (reproduces the original report): uniform 11.86, always_shove 19.45, always_slowpush 4.85, defensive_freeze 1.85, stack_then_shove 4.85
