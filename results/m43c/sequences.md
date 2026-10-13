# M4.3c stabilization, Stage A3: four-action wave sequences through the environment

Preset `GameConfig.m4_four_actions()`, commit `4bd116c26c6475f0be5524eb1a04131482a6522f` + working tree, 2026-10-13 06:23:34. 64 paired seeds per cell (root 75000) with the environment's chance; 'det' = the same cell without chance events (one deterministic run). Coordinates GLOBAL. Reading: docs/m4/m4_3c_strategic_economics_calibration.md §2.

## build

**neutral L3** (40 turns; start ['neutral', 'equal', 3, None])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 | farm4 | -0.25 ± 0.62 | -1.2 | 0.8 (0) | 18.0 | 26.9 (-) | 33 / 40 | 0.8 (0) | 0.00 | 0.05/0.12 | 0.55/0.52 | 4.6 (0.0) |
| farm4 | thin4 | -0.83 ± 0.51 | -6.5 | 0.2 (0) | 19.3 | 24.0 (-) | 34 / 40 | 1.0 (0) | 0.00 | 0.03/0.14 | 0.55/0.52 | 5.7 (0.0) |
| slowpush4 | farm4 | +2.94 ± 0.47 | +7.6 | 4.1 (6) | 14.4 | 16.6 (16) | 43 / 53 | 0.1 (0) | 1.39 | 0.16/0.05 | 0.95/0.45 | 0.5 (0.0) |
| slowpush4 | thin4 | +0.17 ± 0.57 | +0.6 | 0.5 (0) | 16.7 | 24.4 (-) | 53 / 60 | 0.8 (0) | 0.17 | 0.06/0.14 | 0.55/0.48 | 3.4 (0.0) |
| grow_shove4 | farm4 | +3.25 ± 0.52 | +6.1 | 4.7 (7) | 11.6 | 15.6 (15) | 48 / 62 | 0.1 (0) | 1.28 | 0.17/0.05 | 0.98/0.52 | 0.4 (0.0) |
| grow_shove4 | thin4 | -0.01 ± 0.54 | -0.2 | 0.4 (0) | 16.5 | 24.2 (-) | 54 / 61 | 0.8 (0) | 0.16 | 0.05/0.17 | 0.58/0.47 | 5.4 (0.0) |
| build_shove4 | farm4 | +3.97 ± 0.50 | +8.2 | 5.3 (7) | 11.8 | 15.7 (15) | 49 / 63 | 0.1 (0) | 1.55 | 0.17/0.05 | 0.95/0.50 | 0.2 (0.0) |
| build_shove4 | thin4 | +0.04 ± 0.54 | +0.4 | 0.4 (0) | 16.5 | 24.2 (-) | 55 / 62 | 0.8 (0) | 0.16 | 0.05/0.12 | 0.58/0.50 | 3.6 (0.0) |
| shove4 | farm4 | +6.84 ± 0.28 | +1.8 | 14.8 (17) | 8.1 | 5.2 (5) | 113 / 217 | 0.0 (0) | 2.98 | 0.34/0.00 | 1.53/0.20 | 0.0 (0.0) |
| shove4 | thin4 | +5.31 ± 0.33 | -1.6 | 12.9 (15) | 7.5 | 6.2 (6) | 108 / 215 | 0.0 (0) | 2.77 | 0.34/0.00 | 1.53/0.25 | 0.0 (0.0) |

## thin_hold

**B21 at A's -1 equal** (20 turns; start ['freeze', 'equal', 10, 21])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 | walk_up4 | +6.13 ± 0.07 | +52.8 | 1.0 (1) | 21.9 | 19.9 (20) | 17 / 20 | 4.0 (4) | 0.00 | 0.02/0.05 | 0.03/0.47 | 6.8 (7.0) |
| thin4 | walk_up4 | +3.71 ± 0.26 | +36.2 | 0.1 (0) | 24.5 | 19.5 (-) | 34 / 34 | 3.0 (3) | 0.00 | 0.02/0.06 | 0.03/0.25 | 2.2 (2.0) |
| thin_last4 | walk_up4 | +4.72 ± 0.12 | +43.4 | 1.1 (1) | 21.7 | 17.8 (18) | 25 / 29 | 3.0 (3) | 0.00 | 0.02/0.09 | 0.06/0.30 | 6.8 (7.0) |
| shove4 | walk_up4 | +7.04 ± 0.48 | +36.1 | 6.0 (7) | 8.4 | 8.0 (8) | 75 / 125 | 2.0 (2) | 2.16 | 0.27/0.02 | 0.48/0.22 | 2.1 (2.0) |

**B21 at A's -1 large** (20 turns; start ['freeze', 'large', 10, 21])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 | walk_up4 | +6.16 ± 0.07 | +53.0 | 1.0 (1) | 21.8 | 19.9 (20) | 17 / 20 | 4.0 (4) | 0.00 | 0.00/0.05 | 0.00/0.45 | 6.8 (7.0) |
| thin4 | walk_up4 | +5.27 ± 0.13 | +50.8 | 1.2 (1) | 20.3 | 19.4 (20) | 38 / 41 | 2.0 (2) | 0.00 | 0.00/0.08 | 0.03/0.48 | 12.3 (13.0) |
| thin_last4 | walk_up4 | +4.34 ± 0.10 | +44.6 | 2.0 (2) | 17.3 | 17.5 (18) | 32 / 37 | 2.0 (2) | 0.00 | 0.00/0.06 | 0.02/0.36 | 8.8 (9.0) |
| shove4 | walk_up4 | +8.45 ± 0.17 | +34.5 | 7.5 (8) | 7.2 | 6.0 (6) | 87 / 211 | 0.0 (0) | 2.95 | 0.03/0.02 | 0.44/0.08 | 2.0 (2.0) |

**B21 at A's -1 b_large** (20 turns; start ['freeze', 'b_large', 10, 21])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 | walk_up4 | +6.00 ± 0.08 | +51.2 | 1.0 (1) | 22.6 | 19.9 (20) | 17 / 20 | 4.0 (4) | 0.00 | 0.02/0.00 | 0.03/0.20 | 6.8 (7.0) |
| thin4 | walk_up4 | +3.48 ± 0.22 | +34.2 | 0.0 (0) | 22.0 | 18.0 (-) | 34 / 34 | 3.0 (3) | 0.00 | 0.02/0.00 | 0.03/0.14 | 2.0 (2.0) |
| thin_last4 | walk_up4 | +4.57 ± 0.06 | +42.2 | 1.1 (1) | 22.3 | 17.9 (18) | 25 / 29 | 3.0 (3) | 0.00 | 0.02/0.00 | 0.05/0.17 | 6.8 (7.0) |
| shove4 | walk_up4 | +6.93 ± 0.48 | +35.2 | 6.0 (7) | 8.4 | 8.0 (8) | 75 / 125 | 2.0 (2) | 2.16 | 0.28/0.00 | 0.47/0.11 | 2.0 (2.0) |

**B24 at the lane equal** (20 turns; start ['neutral', 'equal', 10, 24])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 | walk_up4 | +4.77 ± 0.19 | +38.9 | 0.1 (0) | 23.0 | 20.0 (-) | 18 / 20 | 4.0 (4) | 0.00 | 0.00/0.03 | 0.06/0.41 | 6.2 (6.0) |
| thin4 | walk_up4 | +4.60 ± 0.15 | +40.6 | 0.2 (0) | 22.3 | 19.6 (-) | 35 / 39 | 2.0 (2) | 0.00 | 0.00/0.05 | 0.11/0.47 | 8.5 (9.0) |
| thin_last4 | walk_up4 | +4.63 ± 0.12 | +40.9 | 0.2 (0) | 22.3 | 19.6 (-) | 32 / 36 | 2.0 (2) | 0.00 | 0.00/0.05 | 0.11/0.47 | 8.6 (9.0) |
| shove4 | walk_up4 | +5.56 ± 0.49 | +25.6 | 5.2 (6) | 8.7 | 10.2 (10) | 76 / 124 | 1.0 (1) | 1.88 | 0.23/0.00 | 0.45/0.22 | 1.1 (1.0) |

**B24 at the lane large** (20 turns; start ['neutral', 'large', 10, 24])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 | walk_up4 | +4.89 ± 0.18 | +39.4 | 0.1 (0) | 23.0 | 20.0 (-) | 19 / 20 | 4.0 (4) | 0.00 | 0.00/0.05 | 0.00/0.39 | 6.2 (6.0) |
| thin4 | walk_up4 | +4.00 ± 0.17 | +41.0 | 1.2 (1) | 24.1 | 19.4 (20) | 42 / 51 | 0.9 (1) | 0.00 | 0.00/0.09 | 0.09/0.45 | 7.9 (8.0) |
| thin_last4 | walk_up4 | +3.99 ± 0.18 | +40.9 | 1.2 (1) | 24.1 | 19.4 (20) | 42 / 51 | 0.9 (1) | 0.00 | 0.00/0.09 | 0.09/0.45 | 7.9 (8.0) |
| shove4 | walk_up4 | +7.85 ± 0.16 | +24.5 | 8.5 (9) | 7.2 | 4.0 (4) | 88 / 209 | 0.0 (0) | 3.00 | 0.02/0.00 | 0.48/0.09 | 0.0 (0.0) |

**B24 at the lane b_large** (20 turns; start ['neutral', 'b_large', 10, 24])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 | walk_up4 | +4.73 ± 0.09 | +37.6 | 0.0 (0) | 23.0 | 20.0 (-) | 19 / 20 | 4.0 (4) | 0.00 | 0.00/0.02 | 0.05/0.11 | 6.0 (6.0) |
| thin4 | walk_up4 | +4.55 ± 0.12 | +39.1 | 0.1 (0) | 22.2 | 19.0 (-) | 35 / 39 | 2.0 (2) | 0.00 | 0.00/0.02 | 0.06/0.25 | 8.7 (9.0) |
| thin_last4 | walk_up4 | +4.57 ± 0.09 | +39.3 | 0.1 (0) | 22.2 | 19.0 (-) | 32 / 36 | 2.0 (2) | 0.00 | 0.00/0.02 | 0.06/0.25 | 8.8 (9.0) |
| shove4 | walk_up4 | +5.47 ± 0.49 | +25.3 | 5.2 (6) | 8.7 | 10.2 (10) | 76 / 124 | 1.0 (1) | 1.86 | 0.23/0.00 | 0.45/0.11 | 1.1 (1.0) |

## sp_thin

**neutral L6** (40 turns; start ['neutral', 'equal', 6, None])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| slowpush4 | farm4 | +2.96 ± 0.48 | +7.4 | 4.1 (6) | 14.4 | 16.6 (16) | 43 / 54 | 0.1 (0) | 1.44 | 0.14/0.03 | 1.02/0.47 | 0.5 (0.0) |
| slowpush4 | thin4 | +0.29 ± 0.57 | +1.5 | 0.5 (0) | 17.4 | 23.8 (-) | 54 / 60 | 0.8 (0) | 0.17 | 0.06/0.11 | 0.56/0.52 | 3.0 (0.0) |
| slowpush4 | slowpush4 | -0.30 ± 0.63 | +0.4 | 0.6 (0) | 16.9 | 23.8 (-) | 69 / 76 | 0.9 (0) | 0.14 | 0.06/0.14 | 0.59/0.53 | 2.7 (0.0) |
| slowpush4 | shove4 | -5.32 ± 0.31 | +2.4 | 0.1 (0) | 18.5 | 13.0 (-) | 83 / 101 | 13.8 (17) | 0.03 | 0.00/0.50 | 0.28/1.30 | 4.8 (4.0) |

## adapt

**neutral L6** (40 turns; start ['neutral', 'equal', 6, None])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| slowpush4 | thin4 | +0.29 ± 0.57 | +1.5 | 0.5 (0) | 17.4 | 23.8 (-) | 54 / 60 | 0.8 (0) | 0.17 | 0.06/0.11 | 0.56/0.52 | 3.0 (0.0) |
| slowpush4 | farm4 | +2.96 ± 0.48 | +7.4 | 4.1 (6) | 14.4 | 16.6 (16) | 43 / 54 | 0.1 (0) | 1.44 | 0.14/0.03 | 1.02/0.47 | 0.5 (0.0) |
| sp_then_shove4 | thin4 | +4.72 ± 0.40 | +0.3 | 10.8 (14) | 7.2 | 15.5 (14) | 98 / 195 | 0.2 (0) | 2.56 | 0.30/0.03 | 1.25/0.45 | 0.4 (0.0) |
| sp_then_shove4 | farm4 | +6.24 ± 0.48 | +2.7 | 12.6 (15) | 8.7 | 14.0 (13) | 91 / 191 | 0.1 (0) | 2.86 | 0.34/0.02 | 1.25/0.38 | 0.1 (0.0) |
| shove4 | thin4 | +5.64 ± 0.27 | -2.6 | 14.4 (18) | 7.4 | 4.1 (4) | 112 / 234 | 0.0 (0) | 2.94 | 0.41/0.00 | 1.52/0.22 | 0.0 (0.0) |
| shove4 | farm4 | +6.61 ± 0.30 | +1.1 | 15.0 (18) | 8.3 | 4.0 (4) | 116 / 233 | 0.0 (0) | 2.97 | 0.42/0.00 | 1.56/0.20 | 0.0 (0.0) |

## recall

**freeze L10** (40 turns; start ['freeze', 'equal', 10, None])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| freeze4 | walk_up4 | +2.25 ± 0.31 | +19.8 | 3.3 (3) | 14.8 | 21.8 (24) | 40 / 51 | 0.0 (0) | 0.00 | 0.08/0.12 | 0.58/0.62 | 16.1 (18.0) |
| freeze_recall4 | walk_up4 | +0.54 ± 0.48 | -2.6 | 1.9 (0) | 16.5 | 26.7 (-) | 34 / 47 | 0.0 (0) | 0.00 | 0.00/0.17 | 4.16/0.83 | 27.7 (40.0) |

**neutral L6** (40 turns; start ['neutral', 'equal', 6, None])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 | farm4 | -0.03 ± 0.60 | -0.0 | 0.8 (0) | 18.0 | 27.6 (-) | 33 / 40 | 0.8 (0) | 0.00 | 0.05/0.12 | 0.58/0.53 | 4.5 (0.0) |
| recall_at6 | farm4 | -0.72 ± 0.48 | -1.5 | 1.0 (0) | 16.3 | 30.9 (-) | 31 / 40 | 1.3 (0) | 0.00 | 0.05/0.09 | 0.67/0.61 | 5.7 (0.0) |
| grow_shove4 | farm4 | +3.51 ± 0.51 | +6.3 | 4.8 (7) | 11.5 | 15.7 (15) | 48 / 63 | 0.1 (0) | 1.48 | 0.16/0.05 | 1.02/0.55 | 0.3 (0.0) |
| grow_crash_recall4 | farm4 | +1.05 ± 0.53 | +0.8 | 4.1 (4) | 12.0 | 15.7 (15) | 47 / 65 | 0.1 (0) | 0.00 | 0.03/0.02 | 2.19/0.53 | 0.4 (0.0) |

## power

**neutral L10 equal** (40 turns; start ['neutral', 'equal', 10, None])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 | farm4 | -0.02 ± 0.61 | -0.1 | 0.8 (0) | 18.4 | 27.6 (-) | 33 / 40 | 0.8 (0) | 0.00 | 0.05/0.11 | 0.58/0.53 | 4.6 (0.0) |
| thin4 | farm4 | +0.58 ± 0.56 | +6.5 | 0.9 (0) | 17.8 | 28.2 (-) | 36 / 43 | 0.4 (0) | 0.00 | 0.05/0.11 | 0.59/0.50 | 4.6 (0.0) |
| slowpush4 | farm4 | +3.42 ± 0.44 | +5.7 | 4.4 (6) | 14.7 | 16.7 (16) | 42 / 53 | 0.1 (0) | 1.92 | 0.14/0.00 | 0.98/0.52 | 0.1 (0.0) |
| shove4 | farm4 | +6.90 ± 0.29 | +1.3 | 16.0 (19) | 8.4 | 4.0 (4) | 120 / 274 | 0.0 (0) | 2.98 | 0.41/0.00 | 1.53/0.19 | 0.0 (0.0) |

**neutral L10 large** (40 turns; start ['neutral', 'large', 10, None])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 | farm4 | +0.92 ± 0.59 | +3.7 | 1.0 (0) | 17.6 | 27.2 (-) | 36 / 40 | 0.3 (0) | 0.00 | 0.00/0.09 | 0.36/0.42 | 1.4 (0.0) |
| thin4 | farm4 | +1.25 ± 0.54 | +6.9 | 1.0 (0) | 17.6 | 27.1 (-) | 38 / 42 | 0.2 (0) | 0.00 | 0.00/0.09 | 0.38/0.42 | 1.6 (0.0) |
| slowpush4 | farm4 | +5.13 ± 0.29 | +6.8 | 5.2 (6) | 14.2 | 15.6 (16) | 42 / 50 | 0.0 (0) | 2.81 | 0.02/0.00 | 0.73/0.39 | 0.0 (0.0) |
| shove4 | farm4 | +8.49 ± 0.21 | +6.6 | 18.4 (20) | 7.7 | 3.0 (3) | 130 / 416 | 0.0 (0) | 3.00 | 0.11/0.00 | 1.38/0.17 | 0.0 (0.0) |

**neutral L10 b_large** (40 turns; start ['neutral', 'b_large', 10, None])

| A | B | A − B gold | A − B XP | A crashes (det) | A crash HP | A first crash | A dealt / budget | B crashes (det) | plates A | deaths A/B | Recalls A/B | A held turns (det) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm4 | farm4 | -1.03 ± 0.53 | -3.7 | 0.4 (0) | 15.7 | 27.8 (-) | 34 / 40 | 1.2 (0) | 0.00 | 0.05/0.06 | 0.50/0.28 | 5.1 (0.0) |
| thin4 | farm4 | -0.61 ± 0.50 | +0.9 | 0.4 (0) | 18.4 | 31.7 (-) | 36 / 43 | 0.5 (0) | 0.00 | 0.05/0.06 | 0.50/0.30 | 5.6 (0.0) |
| slowpush4 | farm4 | +3.19 ± 0.43 | +4.7 | 4.4 (6) | 14.6 | 17.4 (16) | 42 / 53 | 0.1 (0) | 1.88 | 0.14/0.00 | 0.97/0.23 | 0.1 (0.0) |
| shove4 | farm4 | +6.78 ± 0.28 | +0.4 | 15.9 (19) | 8.4 | 4.0 (4) | 120 / 274 | 0.0 (0) | 2.98 | 0.42/0.00 | 1.52/0.06 | 0.0 (0.0) |

## Traces (viewer matchups 4A-*, seed 3)

**4A-1. Slow Push -> Last Hit -> Shove** (setup ['m4_four_actions', 'neutral', 'equal', 3], A grow_shove4 v B farm4; earned [11.57, 9.01])

```
turn -4         -3         -2         -1         +0         +1         +2         +3         +4           A / B: wave applied/budget
0    .          .          A8         .          A8 B8 a b  .          B8         .          .
1    A8         .          .          .          A12 B11 a b.          .          .          B8           SL2/2 / LH1/1
2    .          .          A8         .          A9 B7 a b  .          B8         .          .            LH1/1 / LH1/1
3    A8         .          .          .          A13 B11 a b.          .          .          B8           LH1/1 / LH1/1
4    .          .          A8         .          A10 B7 a b .          B8         .          .            LH1/1 / LH1/1
5    A8         .          .          .          A14 B11 a b.          .          .          B8           LH1/1 / LH1/1
6    .          .          A8         .          A11 B7 a b .          B8         .          .            LH1/1 / LH1/1
7    A8         .          .          .          A15 B10 a b.          .          .          B8           LH1/1 / LH1/1
8    .          .          A8         .          A13 B6 a b .          B8         .          .            LH1/1 / LH1/1
9    A8         .          .          .          A17 B8 a b .          .          .          B8           LH1/1 / LH1/1
10   .          .          A8         .          A15 B4 a b .          B8         .          .            LH1/1 / LH1/1
11   A8         .          .          .          A19 B6 a b .          .          .          B8           LH1/1 / LH1/1
12   .          .          A8         .          A17 B1 a b .          B8         .          .            LH1/1 / LH1/1
13   A8         .          .          .          A23 B3 a b .          .          .          B8           LH1/1 / LH1/1
14   .          .          A8         .          .          A21 a      B4 b       .          .            SH2/5 / LH0/1  B ganked
15   A8         .          .          .          A8         a          [A19] b    .          B8           SH1/5 / LH1/1  A crash 19
16   .          .          A8         .          .          .          [A19] a b  B3         .            SL1/1 / LH1/1  A crash 8
17   A8         .          .          .          A8         .          [A11] a b  .          B8           LH1/1 / LH1/1  A ganked
18   .          .          A8         .          .          .          [A11] a b  B5         .            SL1/1 / LH1/1  A crash 8; A plate
19   A8         .          .          .          A8         .          [A3] b     B11        .            LH1/1 / LH1/1  A dies; A ganked
20   .          .          A8         .          .          .          B9 [A7] b  .          .            away / LH1/1  A crash 8
21   A8         .          .          .          A8         .          B8 b       .          B8           away / LH1/1
22   .          .          A8 a       .          .          A6 B15     .          .          .            away / away  B recalls
23   A8         .          .          .          a          A11 B9     b          .          B8           SL4/4 / away
24   .          .          A8         .          .          A8 B14 a b .          .          .            LH1/1 / LH1/1
```

**4A-2. Thin an oversized wave, then hold** (setup ['m4_four_actions', 'freeze21', 'equal', 3], A freeze4 v B walk_up4; earned [12.34, 7.54])

```
turn -4         -3         -2         -1         +0         +1         +2         +3         +4           A / B: wave applied/budget
0    .          .          A8 [B6] a  B21 b      .          .          B8         .          .
1    A8         .          A1 a       B17 b      B8         .          .          .          B8           TN3/3 / LH1/1
2    .          .          A3 a       B21 b      .          .          B8         .          .            TN3/3 / LH1/1
3    A8         .          [B17] a    b          B8         .          .          .          B8           TN3/3 / LH0/1  B crash 18
4    .          A4         [B14] a    .          b          .          B8         .          .            TN4/4 / LH0/1  B crash 8
5    .          A9         [B3] a     .          B8 b       .          .          .          B8           TN4/4 / LH0/1
6    .          .          A7 [B5] a  .          b          .          B8         .          .            TN4/4 / LH0/1  B crash 8
7    A8         .          .          A5 a       B7 b       .          .          .          B8           TN2/2 / LH0/1
8    .          .          .          A11 a      B4 b       .          B8         .          .            TN1/1 / LH1/1
9    A8         .          .          A7 a       B9 b       .          .          .          B8           TN1/1 / LH1/1
10   .          .          .          A13 a      B6 b       .          B8         .          .            TN1/1 / LH1/1
11   A8         .          .          A9 a       B10 b      .          .          .          B8           TN1/1 / LH1/1
12   .          .          .          A14 a      B7 b       .          B8         .          .            TN1/1 / LH1/1
13   A8         .          .          A10 a      B11 b      .          .          .          B8           TN1/1 / LH1/1
14   .          .          .          A15 a      B7 b       .          B8         .          .            TN1/1 / LH1/1
15   A8         .          .          A11 a      B11 b      .          .          .          B8           TN1/1 / LH1/1
16   .          .          .          A16 a      B7 b       .          B8         .          .            TN1/1 / LH1/1
17   A8         .          .          A12 a      B11 b      .          .          .          B8           TN1/1 / LH1/1  A ganked; B ganked
18   .          .          .          A17 a      B6 b       .          B8         .          .            TN1/1 / LH1/1
19   A8         .          .          A13 a      B10 b      .          .          .          B8           TN1/1 / LH1/1  A ganked
20   .          .          .          A19 a      B5 b       .          B8         .          .            TN1/1 / LH1/1
21   A8         .          .          A15 a      B8 b       .          .          .          B8           TN1/1 / LH1/1
22   .          .          .          A21 a      B3 b       .          B8         .          .            TN1/1 / LH1/1
23   A8         .          .          A19 a      B5 b       .          .          .          B8           TN1/1 / LH1/1
24   .          .          A8         A18 a      .          .          B8 b       .          .            TN1/1 / LH0/1
```

**4A-3. Slow Push against Thin** (setup ['m4_four_actions', 'neutral', 'equal', 3], A slowpush4 v B thin4; earned [9.31, 10.91])

```
turn -4         -3         -2         -1         +0         +1         +2         +3         +4           A / B: wave applied/budget
0    .          .          A8         .          A8 B8 a b  .          B8         .          .
1    A8         .          .          .          A12 B11 a b.          .          .          B8           SL2/2 / TN1/1
2    .          .          A8         .          A8 B7 a b  .          B8         .          .            SL1/1 / TN2/2
3    A8         .          .          .          A11 B11 a b.          .          .          B8           SL1/1 / TN2/2
4    .          .          A8         .          A8 B7 a b  .          B8         .          .            SL2/2 / TN1/1
5    A8         .          .          .          A11 B11 a b.          .          .          B8           SL1/1 / TN2/2
6    .          .          A8         .          A8 B7 a b  .          B8         .          .            SL2/2 / TN1/1
7    A8         .          .          .          A11 B11 a b.          .          .          B8           SL1/1 / TN2/2
8    .          .          A8         .          A8 B7 a b  .          B8         .          .            SL2/2 / TN1/1
9    A8         .          .          .          A11 B11 a b.          .          .          B8           SL1/1 / TN2/2
10   .          .          A8         .          A8 B7 a b  .          B8         .          .            SL2/2 / TN1/1
11   A8         .          .          .          A11 B11 a b.          .          .          B8           SL1/1 / TN2/2
12   .          .          A8         .          A8 B7 a b  .          B8         .          .            SL2/2 / TN1/1
13   A8         .          .          .          A11 B11 a b.          .          .          B8           SL1/1 / TN2/2
14   .          .          A8         .          A8 B7 a b  .          B8         .          .            SL2/2 / TN1/1
15   A8         .          .          .          A11 B11 a b.          .          .          B8           SL1/1 / TN2/2
16   .          .          A8         .          A8 B7 a b  .          B8         .          .            SL2/2 / TN1/1
17   A8         .          .          .          A11 B11 a b.          .          .          B8           SL1/1 / TN2/2  A ganked; B ganked
18   .          .          A8         .          A8 B7 a b  .          B8         .          .            SL2/2 / TN1/1
19   A8         .          .          .          A11 B11 a b.          .          .          B8           SL1/1 / TN2/2  A ganked
20   .          .          A8         a          A8 B9 b    .          B8         .          .            LH0/1 / TN1/1
21   A8         .          .          a          A12 B14 b  .          .          .          B8           LH0/1 / TN1/1
22   .          .          A8         a          A8 B11 b   .          B8         .          .            LH0/1 / TN1/1
23   A8         .          .          a          A11 B16 b  .          .          .          B8           LH0/1 / TN1/1
24   .          .          A8         a          A7 B14 b   .          B8         .          .            LH0/1 / TN1/1
```

**4A-4. Slow Push, then Shove, against Thin** (setup ['m4_four_actions', 'neutral', 'equal', 3], A sp_then_shove4 v B thin4; earned [9.81, 10.89])

```
turn -4         -3         -2         -1         +0         +1         +2         +3         +4           A / B: wave applied/budget
0    .          .          A8         .          A8 B8 a b  .          B8         .          .
1    A8         .          .          .          A12 B11 a b.          .          .          B8           SL2/2 / TN1/1
2    .          .          A8         .          A8 B7 a b  .          B8         .          .            SL1/1 / TN2/2
3    A8         .          .          .          A11 B11 a b.          .          .          B8           SL1/1 / TN2/2
4    .          .          A8         .          A8 B7 a b  .          B8         .          .            SL2/2 / TN1/1
5    A8         .          .          .          A11 B11 a b.          .          .          B8           SL1/1 / TN2/2
6    .          .          A8         .          A8 B7 a b  .          B8         .          .            SL2/2 / TN1/1
7    A8         .          .          .          A11 B11 a b.          .          .          B8           SL1/1 / TN2/2
8    .          .          A8         .          A8 B7 a b  .          B8         .          .            SL2/2 / TN1/1
9    A8         .          .          .          A11 B11 a b.          .          .          B8           SL1/1 / TN2/2
10   .          .          A8         .          A8 B7 a b  .          B8         .          .            SL2/2 / TN1/1
11   A8         .          .          .          A11 B7 a b .          .          .          B8           SH5/5 / TN2/2
12   .          .          A8         .          A10 a      .          B8 b       .          .            SH5/5 / TN0/3
13   A8         .          .          .          .          A14 B1 a b .          .          B8           SH5/5 / TN3/3
14   .          .          A8         .          .          a          B5 [A9] b  .          .            SH2/5 / TN3/3  A crash 11
15   A8         .          .          .          A8         .          a b        .          B8           SH4/5 / TN3/3  A ganked
16   .          .          A8         .          a          .          B8 [A4] b  .          .            SH0/5 / TN3/3  A crash 6
17   A8         .          .          .          A8         B1 b       a          .          B8           SH5/5 / TN1/1  A ganked
18   .          .          A8         .          a          A5         B8 b       .          .            LH0/1 / TN1/3
19   A8         .          .          .          .          A11        B6 b       .          B8           LH0/1 / TN1/1  A dies; A ganked
20   .          .          A8         .          .          A7         B12 b      .          .            away / TN1/1
21   A8         .          .          .          .          A11        B10 b      .          B8           away / TN1/1
22   .          .          A8 a       .          .          A6         B16 b      .          .            away / TN1/1
23   A8         .          .          .          a          A10        B14 b      .          B8           SH0/7 / TN1/1
24   .          .          A8         .          .          A4 a       B14 b      .          .            SH7/7 / TN1/1
```

**4A-5. Freeze -> Recall -> Return** (setup ['m4_four_actions', 'freeze', 'equal', 3], A freeze_recall4 v B walk_up4; earned [9.22, 7.09])

```
turn -4         -3         -2         -1         +0         +1         +2         +3         +4           A / B: wave applied/budget
0    .          .          A8 [B6] a  B5 b       .          .          B8         .          .
1    A8         .          A5 a       B11 b      .          .          .          .          B8           TN1/1 / LH1/1
2    .          .          A10 a      B7 b       .          .          B8         .          .            TN1/1 / LH1/1
3    A8         .          A7 a       B12 b      .          .          .          .          B8           TN1/1 / LH1/1
4    .          .          A11 a      B8 b       .          .          B8         .          .            TN1/1 / LH1/1
5    A8         .          A8 a       B13 b      .          .          .          .          B8           TN1/1 / LH1/1  B ganked
6    .          .          A12        B10 b      .          .          B8         .          .            away / LH1/1  A recalls
7    A8         .          A9 a       B15        .          b          .          .          B8           away / LH0/1
8    .          .          A13 a      B11 b      .          .          B8         .          .            TN1/1 / LH1/1
9    A8         .          A9 a       B15 b      .          .          .          .          B8           TN1/1 / LH1/1
10   .          .          A13 a      B11 b      .          .          B8         .          .            TN1/1 / LH1/1
11   A8         .          A9 a       B15 b      .          .          .          .          B8           TN1/1 / LH1/1
12   .          .          A13 a      B11 b      .          .          B8         .          .            TN1/1 / LH1/1
13   A8         .          A9         B16 b      .          .          .          .          B8           away / LH1/1  A recalls
14   .          .          A14 a      B13        .          b          B8         .          .            away / LH0/1  B ganked
15   A8         .          A10 a      B16        .          .          .          .          B8           TN2/2 / away  B recalls
16   .          .          A15 a      B11        .          .          B8 b       .          .            TN1/1 / away
17   A8         .          A12 a      B15        b          .          .          .          B8           TN1/1 / LH0/1  A ganked; B ganked
18   .          .          A16 a      B9 b       .          .          B8         .          .            TN1/1 / LH1/1
19   A8         .          A13        B14 b      .          .          .          .          B8           away / LH1/1  A recalls
20   .          .          A18 a      B9         .          b          B8         .          .            away / LH0/1
21   A8         .          A15 a      B12 b      .          .          .          .          B8           TN1/1 / LH1/1
22   .          .          A19 a      B6 b       .          .          B8         .          .            TN1/1 / LH1/1
23   A8         .          A16 a      B9 b       .          .          .          .          B8           TN1/1 / LH1/1
24   .          .          A22 a      B3 b       .          .          B8         .          .            TN1/1 / LH1/1
```

**4A-6. Crash, then Recall** (setup ['m4_four_actions', 'neutral', 'equal', 3], A grow_crash_recall4 v B farm4; earned [9.34, 8.74])

```
turn -4         -3         -2         -1         +0         +1         +2         +3         +4           A / B: wave applied/budget
0    .          .          A8         .          A8 B8 a b  .          B8         .          .
1    A8         .          .          .          A12 B11 a b.          .          .          B8           SL2/2 / LH1/1
2    .          .          A8         .          A9 B7 a b  .          B8         .          .            LH1/1 / LH1/1
3    A8         .          .          .          A13 B11 a b.          .          .          B8           LH1/1 / LH1/1
4    .          .          A8         .          A10 B7 a b .          B8         .          .            LH1/1 / LH1/1
5    A8         .          .          .          A14 B11 a b.          .          .          B8           LH1/1 / LH1/1
6    .          .          A8         .          A11 B7 a b .          B8         .          .            LH1/1 / LH1/1
7    A8         .          .          .          A15 B10 a b.          .          .          B8           LH1/1 / LH1/1
8    .          .          A8         .          A13 B6 a b .          B8         .          .            LH1/1 / LH1/1
9    A8         .          .          .          A17 B8 a b .          .          .          B8           LH1/1 / LH1/1
10   .          .          A8         .          A15 B4 a b .          B8         .          .            LH1/1 / LH1/1
11   A8         .          .          .          A19 B6 a b .          .          .          B8           LH1/1 / LH1/1
12   .          .          A8         .          A17 B1 a b .          B8         .          .            LH1/1 / LH1/1
13   A8         .          .          .          A23 B3 a b .          .          .          B8           LH1/1 / LH1/1
14   .          .          A8         .          .          A21 a      B4 b       .          .            SH2/5 / LH0/1  B ganked
15   A8         .          .          .          A8         a          [A19] b    .          B8           SH1/5 / LH1/1  A crash 19
16   .          .          A8         a          .          .          [A19] b    B4         .            LH0/1 / LH1/1  A crash 8
17   A8         .          .          .          A8         .          [A11] b    B9         .            away / LH1/1  A recalls
18   .          .          A8 a       .          .          .          [A10] b    B7         .            away / LH1/1  A crash 8
19   A8         .          .          .          A8 a       .          [A2] b     B14        .            LH0/1 / LH1/1  A ganked
20   .          .          A8         a          .          .          B12 [A7] b .          .            LH0/1 / LH0/1  A crash 8
21   A8         .          .          .          A8         .          B11 b      .          B8           away / LH1/1  A recalls
22   .          .          A8 a       .          .          A5 B18     .          .          .            away / away  B recalls
23   A8         .          .          .          a          A10 B12    b          .          B8           SL4/4 / away
24   .          .          A8         .          .          A5 B15 a b .          .          .            SL4/4 / LH1/1
```

**4A-7. Strong freezer against a walk-up** (setup ['m4_four_actions', 'freeze', 'large', 3], A freeze_punish4 v B walk_up4; earned [10.74, 7.54])

```
turn -4         -3         -2         -1         +0         +1         +2         +3         +4           A / B: wave applied/budget
0    .          .          A8 [B6] a  B5 b       .          .          B8         .          .
1    A8         .          A5 a       B11 b      .          .          .          .          B8           TN1/1 / LH1/1
2    .          .          A10 a      B7 b       .          .          B8         .          .            TN1/1 / LH1/1
3    A8         .          A7 a       B12 b      .          .          .          .          B8           TN1/1 / LH1/1
4    .          .          A11 a      B8 b       .          .          B8         .          .            TN1/1 / LH1/1
5    A8         .          A8 a       B13 b      .          .          .          .          B8           TN1/1 / LH1/1  B ganked
6    .          .          A12 a      B9 b       .          .          B8         .          .            TN1/1 / LH1/1
7    A8         .          A9 a       B13 b      .          .          .          .          B8           TN1/1 / LH1/1
8    .          .          A13 a      B9 b       .          .          B8         .          .            TN1/1 / LH1/1
9    A8         .          A10 a      B13 b      .          .          .          .          B8           TN1/1 / LH1/1
10   .          .          A14 a      B9 b       .          .          B8         .          .            TN1/1 / LH1/1
11   A8         .          A11 a      B13 b      .          .          .          .          B8           TN1/1 / LH1/1
12   .          .          A15 a      B8 b       .          .          B8         .          .            TN1/1 / LH1/1
13   A8         .          A12 a      B12 b      .          .          .          .          B8           TN1/1 / LH1/1
14   .          .          A16 a      B6 b       .          .          B8         .          .            TN1/1 / LH1/1  B ganked
15   A8         .          A14 a      B10        .          b          .          .          B8           TN1/1 / LH0/1
16   .          .          A21 a      B4         .          .          B8         .          .            TN1/1 / away  B recalls
17   A8         .          .          A19 a      B6         .          b          .          B8           TN1/1 / away  A ganked
18   .          .          A8         A18 a      .          .          B8 b       .          .            TN1/1 / LH0/1
19   A8         .          .          .          A24 a      B1 b       .          .          B8           TN1/1 / LH1/1  A ganked
20   .          .          A8         .          .          A22        B6 a b     .          .            TN0/1 / LH0/1
21   A8         .          .          .          A8         A20 a      b          .          B8           TN1/1 / LH1/1
22   .          .          A8         .          .          a          [A23] b    B2         .            TN0/1 / LH1/1  A crash 28
23   A8         .          .          .          A8 a       .          [A15] b    .          B8           TN0/1 / LH1/1
24   .          .          A8         .          a          .          [A15] b    B4         .            TN0/1 / LH1/1  A crash 8
```

**4A-8. Weak champion against a big wave** (setup ['m4_four_actions', 'oversized', 'b_large', 3], A thin4 v B farm4; earned [9.74, 6.66])

```
turn -4         -3         -2         -1         +0         +1         +2         +3         +4           A / B: wave applied/budget
0    .          .          A8         .          A8 B24 a b .          B8         .          .
1    A8         .          .          .          A8 B19 a b B8         .          .          B8           TN2/2 / LH1/1
2    .          .          A8         .          A1 B16 a b B8         B8         .          .            TN2/2 / LH1/1
3    A8         .          .          .          A1 B21 a b B8         .          .          B8           TN2/2 / LH1/1
4    .          .          A5 a       B18 b      B8         .          B8         .          .            TN1/3 / LH0/1
5    A8         .          a          B22 b      B8         .          .          .          B8           TN3/3 / LH1/1
6    .          A2         [B23] a    b          .          .          B8         .          .            TN3/3 / LH0/1  B crash 30
7    A8         .          [B13] a    .          B8 b       .          .          .          B8           TN3/3 / LH0/1
8    .          A5         [B11] a    .          b          .          B8         .          .            TN3/3 / LH0/1  B crash 8
9    .          A12        [B1] a     .          B8 b       .          .          .          B8           TN3/3 / LH0/1
10   .          .          A10 [B7] a .          b          .          B8         .          .            TN0/1 / LH0/1  B crash 8
11   A8         .          A9 a       .          B8 b       .          .          .          B8           TN1/1 / LH0/1
12   .          .          .          A15 B5 a b .          .          B8         .          .            TN1/1 / LH1/1
13   A8         .          .          A12 B9 a b .          .          .          .          B8           TN1/1 / LH1/1
14   .          .          .          A18 B4 a b .          .          B8         .          .            TN1/1 / LH1/1  B ganked
15   A8         .          .          A15 B7 a b .          .          .          .          B8           TN1/1 / LH1/1
16   .          .          .          A21 B2 a b .          .          B8         .          .            TN1/1 / LH1/1
17   A8         .          .          .          A18 B6 a b .          .          .          B8           TN0/1 / LH1/1  A ganked; B ganked
18   .          .          A8         .          A17 B1 a   b          B8         .          .            TN1/1 / LH0/1
19   A8         .          .          .          A24 B2 a   b          .          .          B8           TN1/1 / LH0/1  A ganked
20   .          .          A8         a          .          A22        B6 b       .          .            LH0/1 / LH0/1
21   A8         .          .          .          A8         A21        .          .          B8           away / away  A recalls; B recalls
22   .          .          A8 a       .          .          .          [A25] b    B2         .            away / away  A crash 29
23   A8         .          .          .          A8 a       .          [A17] b    .          B8           TN0/1 / LH1/1
24   .          .          A8         .          a          .          [A17] b    B4         .            TN0/1 / LH1/1  A crash 8
```
