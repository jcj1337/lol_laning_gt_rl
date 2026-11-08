# M4.5a smoke results (generated)

Descriptive checks of the EXPERIMENTAL `GameConfig.m45_unified()`; not a calibration. Regenerate with `python -m lanegame.experiments.m45a all`.

## Throughput (uniform legal play, one process)

| preset | turns / s |
|---|---|
| `m4_four_actions_econ_aggro` | 3,832 |
| `m45_unified_r1` | 3,160 |
| `m45_unified` | 3,368 |

M4.5a / aggro = 0.88

## Uniform random legal play (40 games x 300 turns, invariants and the observation decoder checked every turn)

| event | per champion-turn |
|---|---|
| ward | 0.0310 |
| placed | 0.0273 |
| interrupted | 0.0037 |
| retaliation_ticks | 0.0301 |
| positional_ticks | 0.0726 |
| minion_hit_turns | 0.0871 |
| minion_centi | 2.4786 |
| capped | 0.0357 |
| connected | 0.1386 |
| landed | 0.0630 |
| at_w | 0.0329 |
| deaths | 0.0197 |

Mean minion damage in a turn with minion hits, as a share of the max-HP bar, by level:

| level | turns | share |
|---|---|---|
| 1 | 14 | 0.095 |
| 2 | 18 | 0.102 |
| 3 | 19 | 0.123 |
| 4 | 16 | 0.114 |
| 5 | 35 | 0.084 |
| 6 | 58 | 0.076 |
| 7 | 60 | 0.062 |
| 8 | 43 | 0.059 |
| 9 | 48 | 0.057 |
| 10 | 77 | 0.038 |
| 11 | 91 | 0.037 |
| 12 | 74 | 0.035 |
| 13 | 99 | 0.031 |
| 14 | 102 | 0.024 |
| 15 | 90 | 0.019 |
| 16 | 111 | 0.020 |
| 17 | 107 | 0.016 |
| 18 | 1029 | 0.014 |

## Minion damage table (alpha 50, growth 1.08)

m = minion damage units (one tick of a blob deals ceil(f/2) or floor(f/2), f = max(1, floor(2h/8)) per turn: an 8-HP wave 1 + 1, 16 HP 2 + 2, 24 HP 3 + 3); entries [centi-HP after durability, share of the max-HP bar]

| level | invested | H(L) | m = 1 | m = 2 | m = 3 | m = 4 | m = 6 |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 600 | 50 (8.3%) | 100 (16.7%) | 150 (25.0%) | 200 (33.3%) | 300 (50.0%) |
| 2 | 0 | 630 | 46 (7.3%) | 93 (14.8%) | 139 (22.1%) | 185 (29.4%) | 278 (44.1%) |
| 4 | 0 | 695 | 40 (5.8%) | 79 (11.4%) | 119 (17.1%) | 159 (22.9%) | 238 (34.2%) |
| 6 | 8 | 766 | 26 (3.4%) | 53 (6.9%) | 79 (10.3%) | 106 (13.8%) | 159 (20.8%) |
| 11 | 32 | 977 | 14 (1.4%) | 28 (2.9%) | 42 (4.3%) | 56 (5.7%) | 85 (8.7%) |
| 15 | 64 | 1188 | 9 (0.8%) | 18 (1.5%) | 27 (2.3%) | 37 (3.1%) | 55 (4.6%) |
| 18 | 128 | 1375 | 7 (0.5%) | 13 (0.9%) | 20 (1.5%) | 27 (2.0%) | 40 (2.9%) |
| 18 | 300 | 1375 | 6 (0.4%) | 12 (0.9%) | 18 (1.3%) | 24 (1.8%) | 37 (2.7%) |

## Scripted pairings (16 paired seeds x 200 turns; mean ± 1.96 SE over seeds)

**`m4_four_actions_econ_aggro`**

| pairing (A v B [start, gap]) | lead | deaths_A | deaths_B | recalls_A | recalls_B | minion_cp_A | minion_cp_B | wards_A | wards_placed_A | ret_ticks_A | pos_ticks_A | ret_ticks_B | pos_ticks_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 v farm4 [s0, equal] | 0.66 ± 1.60 | 0.31 ± 0.23 | 0.31 ± 0.30 | 3.25 ± 0.66 | 3.38 ± 0.59 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ward_favorable4 v farm4 [s0, equal] | 0.66 ± 1.60 | 0.31 ± 0.23 | 0.31 ± 0.30 | 3.25 ± 0.66 | 3.38 ± 0.59 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ward_always4 v bully4 [neutral, equal] | 13.48 ± 3.88 | 2.88 ± 0.71 | 2.06 ± 0.77 | 10.25 ± 0.46 | 12.44 ± 1.04 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| bully4 v farm4 [s0, equal] | -13.36 ± 4.55 | 1.88 ± 0.64 | 3.19 ± 0.74 | 12.62 ± 0.96 | 9.25 ± 0.85 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| bully4 v step_back4 [neutral, equal] | -5.30 ± 2.37 | 1.25 ± 0.55 | 0.00 ± 0.00 | 11.88 ± 0.86 | 1.81 ± 0.41 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| freeze_bully4 v walk_up_shove4 [freeze, large] | -8.77 ± 4.12 | 0.44 ± 0.31 | 7.00 ± 1.03 | 10.31 ± 1.10 | 7.88 ± 0.84 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ward_freeze4 v walk_up_shove4 [freeze, large] | 0.84 ± 1.17 | 0.00 ± 0.00 | 1.00 ± 0.57 | 0.31 ± 0.23 | 8.62 ± 1.30 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

**`m45_unified_r1`**

| pairing (A v B [start, gap]) | lead | deaths_A | deaths_B | recalls_A | recalls_B | minion_cp_A | minion_cp_B | wards_A | wards_placed_A | ret_ticks_A | pos_ticks_A | ret_ticks_B | pos_ticks_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 v farm4 [s0, equal] | -0.05 ± 1.92 | 0.31 ± 0.23 | 0.31 ± 0.30 | 3.31 ± 0.66 | 3.38 ± 0.59 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ward_favorable4 v farm4 [s0, equal] | -2.65 ± 2.23 | 0.06 ± 0.12 | 0.25 ± 0.22 | 2.00 ± 0.40 | 2.81 ± 0.72 | 0.00 ± 0.00 | 0.00 ± 0.00 | 20.19 ± 0.57 | 20.19 ± 0.57 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ward_always4 v bully4 [neutral, equal] | 13.05 ± 4.50 | 1.81 ± 0.57 | 2.44 ± 0.51 | 10.88 ± 0.59 | 13.88 ± 0.59 | 0.00 ± 0.00 | 2038.44 ± 312.40 | 19.44 ± 1.87 | 9.56 ± 0.76 | 0.00 ± 0.00 | 0.00 ± 0.00 | 25.94 ± 2.42 | 28.19 ± 5.24 |
| bully4 v farm4 [s0, equal] | -16.44 ± 3.39 | 2.25 ± 0.83 | 1.81 ± 0.41 | 14.81 ± 1.03 | 10.88 ± 0.80 | 2422.94 ± 218.68 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 27.75 ± 2.67 | 31.25 ± 4.55 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| bully4 v step_back4 [neutral, equal] | -4.85 ± 5.01 | 1.31 ± 0.46 | 0.00 ± 0.00 | 13.00 ± 1.28 | 1.62 ± 0.43 | 961.94 ± 244.51 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.81 ± 0.63 | 29.56 ± 6.52 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| freeze_bully4 v walk_up_shove4 [freeze, large] | -3.78 ± 3.94 | 0.38 ± 0.24 | 5.94 ± 1.10 | 11.31 ± 0.87 | 8.94 ± 0.87 | 452.50 ± 37.26 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 18.75 ± 1.51 | 7.56 ± 1.42 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ward_freeze4 v walk_up_shove4 [freeze, large] | -1.13 ± 1.21 | 0.00 ± 0.00 | 0.94 ± 0.52 | 0.06 ± 0.12 | 8.69 ± 1.11 | 0.00 ± 0.00 | 0.00 ± 0.00 | 9.56 ± 0.74 | 9.56 ± 0.74 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

**`m45_unified`**

| pairing (A v B [start, gap]) | lead | deaths_A | deaths_B | recalls_A | recalls_B | minion_cp_A | minion_cp_B | wards_A | wards_placed_A | ret_ticks_A | pos_ticks_A | ret_ticks_B | pos_ticks_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 v farm4 [s0, equal] | -0.05 ± 1.92 | 0.31 ± 0.23 | 0.31 ± 0.30 | 3.31 ± 0.66 | 3.38 ± 0.59 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ward_favorable4 v farm4 [s0, equal] | -1.25 ± 2.02 | 0.19 ± 0.20 | 0.38 ± 0.24 | 1.88 ± 0.43 | 2.88 ± 0.69 | 134.44 ± 22.68 | 0.00 ± 0.00 | 20.25 ± 0.55 | 20.12 ± 0.56 | 0.00 ± 0.00 | 5.62 ± 0.69 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ward_always4 v bully4 [neutral, equal] | 17.14 ± 3.87 | 1.12 ± 0.50 | 2.19 ± 0.67 | 11.62 ± 0.50 | 15.00 ± 0.76 | 71.31 ± 29.77 | 2329.25 ± 237.86 | 18.19 ± 2.13 | 8.62 ± 0.62 | 0.00 ± 0.00 | 2.75 ± 0.85 | 27.50 ± 1.94 | 35.62 ± 4.34 |
| bully4 v farm4 [s0, equal] | -16.33 ± 4.91 | 1.56 ± 0.71 | 2.06 ± 0.73 | 15.19 ± 0.88 | 10.75 ± 0.66 | 2541.62 ± 313.77 | 2.56 ± 3.43 | 0.00 ± 0.00 | 0.00 ± 0.00 | 26.56 ± 2.43 | 41.50 ± 4.40 | 0.00 ± 0.00 | 0.12 ± 0.17 |
| bully4 v step_back4 [neutral, equal] | -11.13 ± 3.75 | 1.44 ± 0.40 | 0.00 ± 0.00 | 13.62 ± 1.03 | 1.50 ± 0.47 | 918.19 ± 179.57 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.81 ± 0.51 | 31.62 ± 4.44 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| freeze_bully4 v walk_up_shove4 [freeze, large] | -4.62 ± 3.44 | 0.38 ± 0.30 | 6.75 ± 0.99 | 11.31 ± 0.71 | 8.38 ± 0.93 | 975.25 ± 64.48 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 18.00 ± 1.79 | 40.19 ± 3.12 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| ward_freeze4 v walk_up_shove4 [freeze, large] | -4.41 ± 1.32 | 0.00 ± 0.00 | 1.06 ± 0.55 | 0.12 ± 0.17 | 8.56 ± 1.21 | 437.88 ± 34.09 | 0.00 ± 0.00 | 9.56 ± 0.78 | 9.56 ± 0.78 | 0.00 ± 0.00 | 18.94 ± 1.60 | 0.00 ± 0.00 | 0.00 ± 0.00 |

## Demonstration trajectories (viewer: `streamlit run lanegame/viewer/app.py`, Recorded trajectory, folder `results/m45a/trajectories`)

| file | seed | turns | replay exact |
|---|---|---|---|
| `m45a_E1_ward_neutral.json` | 0 | 30 | True |
| `m45a_E2_ward_interrupted.json` | 0 | 30 | True |
| `m45a_E3_ward_trade_missed.json` | 2 | 30 | True |
| `m45a_C1_bigwave_level2.json` | 2 | 12 | True |
| `m45a_C2_bigwave_level18.json` | 2 | 12 | True |
| `m45a_D1_ahead_of_wave.json` | 0 | 12 | True |
| `m45a_A1_strong_freeze.json` | 0 | 40 | True |
| `m45a_M1_hold_follows.json` | 0 | 2 | True |
| `m45a_M2_chase_missed.json` | 0 | 1 | True |
| `m45a_M3_chase_landed.json` | 3 | 1 | True |
| `m45a_M4_walk_back.json` | 0 | 2 | True |
