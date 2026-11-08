# M4.3d Part 5b: the M4.3c restricted games on the M4.3d configurations

M4.3c scenarios (`m43c.economics.MATRIX_SCENARIOS`; F = freeze, N = neutral, P = freeze phase) with the M4.3c strategy sets plus `sh_vis` (Shove, Vision then Pressure) for both seats in F and N, and `sh_dive`, `sh_recall` in N. 64 paired games of 100 turns per cell, SeedSequence 79,200. Strategy names drop the suffix 4.

## Scorecard (objective J)

| configuration | B shoves at equal power (F10 / N10) | B concedes / defends vs x1.76 (F10g) | B concedes / defends vs x2.04 (F10gl) | A freezes ahead (F10g / P10g) | A uses Vision (N10g / N3) | value of a x1.76 lead (F10g / N10g) | distinct equilibria |
|---|---|---|---|---|---|---|---|
| four-action baseline | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.69 | 1.00 / 0.00 | +7.54 / +6.47 | 6 |
| econ (0.8) | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 1.00 | 0.70 / 0.00 | +5.20 / +3.91 | 8 |
| econ, 0.85 | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 1.00 | 0.68 / 0.00 | +4.62 / +3.27 | 9 |
| econ, 0.9 | 1.00 / 0.82 | 0.26 / 0.39 | 0.00 / 0.22 | 0.05 / 1.00 | 0.94 / 0.40 | +3.84 / +2.55 | 9 |
| econ + tower aggro | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 1.00 | 0.65 / 0.00 | +5.20 / +3.91 | 8 |

## Equilibria (J = discounted gold difference, 100 turns)

| scenario | configuration | value to A | A's equilibrium | B's equilibrium |
|---|---|---|---|---|
| F4 | four-action baseline | +0.45 | shove 1.00 | walk_up_shove 1.00 |
| F4 | econ (0.8) | +0.80 | shove 1.00 | walk_up_shove 1.00 |
| F4 | econ, 0.85 | +0.84 | shove 1.00 | walk_up_shove 1.00 |
| F4 | econ, 0.9 | +1.10 | shove 0.70, freeze 0.30 | walk_up_shove 0.62, sh_vis 0.38 |
| F4 | econ + tower aggro | +0.80 | shove 1.00 | walk_up_shove 1.00 |
| F10 | four-action baseline | +0.86 | shove 1.00 | walk_up_shove 1.00 |
| F10 | econ (0.8) | +1.00 | shove 1.00 | walk_up_shove 1.00 |
| F10 | econ, 0.85 | +1.00 | shove 1.00 | walk_up_shove 1.00 |
| F10 | econ, 0.9 | +1.23 | shove 0.77, freeze 0.23 | walk_up_shove 0.57, sh_vis 0.43 |
| F10 | econ + tower aggro | +1.00 | shove 1.00 | walk_up_shove 1.00 |
| F10m | four-action baseline | +4.84 | shove 1.00 | walk_up_shove 1.00 |
| F10m | econ (0.8) | +3.58 | shove 1.00 | walk_up_shove 1.00 |
| F10m | econ, 0.85 | +3.29 | shove 1.00 | walk_up_shove 1.00 |
| F10m | econ, 0.9 | +2.75 | shove 0.94, freeze_hunt 0.06 | walk_up_shove 0.56, walk_up_thin 0.44 |
| F10m | econ + tower aggro | +3.58 | shove 1.00 | walk_up_shove 1.00 |
| F10g | four-action baseline | +7.54 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10g | econ (0.8) | +5.20 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10g | econ, 0.85 | +4.62 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10g | econ, 0.9 | +3.84 | sh_vis 0.95, freeze_bully 0.04, freeze_hunt 0.02 | walk_up_thin 0.39, walk_up_shove 0.35, respect 0.26 |
| F10g | econ + tower aggro | +5.20 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10gl | four-action baseline | +7.60 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10gl | econ (0.8) | +5.19 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10gl | econ, 0.85 | +4.60 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10gl | econ, 0.9 | +3.99 | sh_vis 1.00, freeze_hunt 0.00 | walk_up_shove 0.78, walk_up_thin 0.22 |
| F10gl | econ + tower aggro | +5.19 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10b | four-action baseline | -3.40 | shove 1.00 | walk_up_shove 1.00 |
| F10b | econ (0.8) | -1.43 | shove 1.00 | walk_up_shove 1.00 |
| F10b | econ, 0.85 | -0.91 | shove 0.56, freeze 0.44 | walk_up_shove 0.99, sh_vis 0.01 |
| F10b | econ, 0.9 | -0.14 | shove 0.54, freeze 0.46 | walk_up_shove 0.69, sh_vis 0.31 |
| F10b | econ + tower aggro | -1.43 | shove 1.00 | walk_up_shove 1.00 |
| N3 | four-action baseline | +0.17 | shove 1.00 | shove 1.00 |
| N3 | econ (0.8) | +0.31 | shove 1.00 | sh_vis 1.00 |
| N3 | econ, 0.85 | +0.30 | shove 1.00 | sh_vis 1.00 |
| N3 | econ, 0.9 | +0.24 | shove 0.58, sh_vis 0.40, thin 0.01 | shove 0.69, sh_vis 0.17, thin 0.14 |
| N3 | econ + tower aggro | +0.31 | shove 1.00 | sh_vis 1.00 |
| N10 | four-action baseline | +0.18 | shove 1.00 | shove 1.00 |
| N10 | econ (0.8) | +0.16 | shove 1.00 | shove 1.00 |
| N10 | econ, 0.85 | +0.15 | shove 1.00 | shove 1.00 |
| N10 | econ, 0.9 | +0.16 | shove 0.60, sh_vis 0.25, thin 0.15 | shove 0.51, sh_vis 0.31, thin 0.18 |
| N10 | econ + tower aggro | +0.16 | shove 1.00 | shove 1.00 |
| N10g | four-action baseline | +6.47 | sh_vis 1.00 | shove 1.00 |
| N10g | econ (0.8) | +3.91 | sh_vis 0.70, shove 0.30 | shove 0.69, sh_dive 0.31 |
| N10g | econ, 0.85 | +3.27 | sh_vis 0.68, shove 0.32, shove_hunt 0.01 | shove 0.70, sh_dive 0.24, thin 0.05 |
| N10g | econ, 0.9 | +2.55 | sh_vis 0.94, shove_hunt 0.06 | sh_vis 0.65, thin 0.35 |
| N10g | econ + tower aggro | +3.91 | sh_vis 0.65, shove 0.35 | shove 0.82, sh_dive 0.18 |
| P4 | four-action baseline | +0.45 | shove 1.00 | shove 1.00 |
| P4 | econ (0.8) | +0.86 | pz_bully 1.00 | shove 1.00 |
| P4 | econ, 0.85 | +1.12 | pz_bully 1.00 | shove 1.00 |
| P4 | econ, 0.9 | +1.44 | pz_bully 1.00 | shove 1.00 |
| P4 | econ + tower aggro | +0.86 | pz_bully 1.00 | shove 1.00 |
| P10 | four-action baseline | +0.86 | shove 1.00 | shove 1.00 |
| P10 | econ (0.8) | +1.11 | pz_hunt 1.00 | shove 1.00 |
| P10 | econ, 0.85 | +1.24 | pz_hunt 1.00 | shove 1.00 |
| P10 | econ, 0.9 | +1.38 | pz_hunt 1.00 | shove 1.00 |
| P10 | econ + tower aggro | +1.11 | pz_hunt 1.00 | shove 1.00 |
| P10g | four-action baseline | +6.48 | pz_hunt 0.69, shove 0.31 | pz_thin 0.57, shove 0.43 |
| P10g | econ (0.8) | +5.13 | pz_bully 1.00 | shove 1.00 |
| P10g | econ, 0.85 | +4.67 | pz_bully 1.00 | shove 1.00 |
| P10g | econ, 0.9 | +4.22 | pz_bully 1.00 | shove 1.00 |
| P10g | econ + tower aggro | +5.13 | pz_bully 1.00 | shove 1.00 |
| P10gl | four-action baseline | +7.33 | shove 1.00 | shove 1.00 |
| P10gl | econ (0.8) | +5.58 | pz_bully 1.00 | shove 1.00 |
| P10gl | econ, 0.85 | +5.11 | pz_bully 1.00 | shove 1.00 |
| P10gl | econ, 0.9 | +4.63 | pz_bully 1.00 | shove 1.00 |
| P10gl | econ + tower aggro | +5.58 | pz_bully 1.00 | shove 1.00 |
| P10b | four-action baseline | -4.15 | shove 0.66, pz_hunt 0.34 | pz_cautious 0.52, shove 0.48 |
| P10b | econ (0.8) | -2.01 | shove 0.58, pz_hunt 0.42 | pz_cautious 0.53, shove 0.47 |
| P10b | econ, 0.85 | -1.48 | shove 0.53, pz_hunt 0.47 | pz_cautious 0.57, shove 0.43 |
| P10b | econ, 0.9 | -0.97 | pz_hunt 0.54, shove 0.46 | pz_cautious 0.63, shove 0.37 |
| P10b | econ + tower aggro | -2.01 | shove 0.58, pz_hunt 0.42 | pz_cautious 0.53, shove 0.47 |

## Equilibria (G40 = undiscounted gold difference, first 40 turns)

| scenario | configuration | value to A | A's equilibrium | B's equilibrium |
|---|---|---|---|---|
| F4 | four-action baseline | +0.17 | shove 1.00 | walk_up_shove 1.00 |
| F4 | econ (0.8) | +0.41 | shove 1.00 | walk_up_shove 1.00 |
| F4 | econ, 0.85 | +0.45 | shove 1.00 | walk_up_shove 1.00 |
| F4 | econ, 0.9 | +0.49 | shove 1.00 | walk_up_shove 1.00 |
| F4 | econ + tower aggro | +0.41 | shove 1.00 | walk_up_shove 1.00 |
| F10 | four-action baseline | +0.84 | shove 1.00 | walk_up_shove 1.00 |
| F10 | econ (0.8) | +1.04 | shove 1.00 | walk_up_shove 1.00 |
| F10 | econ, 0.85 | +1.06 | shove 1.00 | walk_up_shove 1.00 |
| F10 | econ, 0.9 | +1.10 | shove 1.00 | walk_up_shove 1.00 |
| F10 | econ + tower aggro | +1.04 | shove 1.00 | walk_up_shove 1.00 |
| F10m | four-action baseline | +2.55 | shove 1.00 | walk_up_shove 1.00 |
| F10m | econ (0.8) | +2.00 | shove 1.00 | walk_up_shove 1.00 |
| F10m | econ, 0.85 | +1.86 | shove 1.00 | walk_up_shove 1.00 |
| F10m | econ, 0.9 | +1.70 | shove 1.00 | walk_up_shove 1.00 |
| F10m | econ + tower aggro | +2.00 | shove 1.00 | walk_up_shove 1.00 |
| F10g | four-action baseline | +5.42 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10g | econ (0.8) | +3.92 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10g | econ, 0.85 | +3.53 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10g | econ, 0.9 | +3.14 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10g | econ + tower aggro | +3.92 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10gl | four-action baseline | +5.56 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10gl | econ (0.8) | +4.00 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10gl | econ, 0.85 | +3.60 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10gl | econ, 0.9 | +3.20 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10gl | econ + tower aggro | +4.00 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10b | four-action baseline | -0.23 | shove 1.00 | walk_up_shove 1.00 |
| F10b | econ (0.8) | +0.78 | shove 1.00 | walk_up_shove 1.00 |
| F10b | econ, 0.85 | +1.04 | shove 1.00 | walk_up_shove 1.00 |
| F10b | econ, 0.9 | +1.31 | shove 1.00 | walk_up_shove 1.00 |
| F10b | econ + tower aggro | +0.78 | shove 1.00 | walk_up_shove 1.00 |
| N3 | four-action baseline | +0.22 | sh_vis 1.00 | shove 1.00 |
| N3 | econ (0.8) | +0.32 | sh_vis 1.00 | sh_vis 1.00 |
| N3 | econ, 0.85 | +0.31 | sh_vis 1.00 | sh_vis 1.00 |
| N3 | econ, 0.9 | +0.30 | sh_vis 1.00 | sh_vis 1.00 |
| N3 | econ + tower aggro | +0.32 | sh_vis 1.00 | sh_vis 1.00 |
| N10 | four-action baseline | +0.02 | sh_dive 0.63, shove 0.37 | sh_vis 0.66, shove 0.34 |
| N10 | econ (0.8) | +0.06 | sh_vis 1.00 | sh_vis 1.00 |
| N10 | econ, 0.85 | +0.07 | sh_vis 1.00 | sh_vis 1.00 |
| N10 | econ, 0.9 | +0.07 | sh_vis 1.00 | sh_vis 1.00 |
| N10 | econ + tower aggro | +0.06 | sh_vis 1.00 | sh_vis 1.00 |
| N10g | four-action baseline | +3.70 | sh_vis 1.00 | sh_vis 1.00 |
| N10g | econ (0.8) | +2.15 | shove 0.75, shove_hunt 0.25 | sh_recall 0.93, sh_vis 0.07 |
| N10g | econ, 0.85 | +1.88 | shove_hunt 0.90, respect 0.10 | sh_recall 0.86, sh_vis 0.14 |
| N10g | econ, 0.9 | +1.68 | shove_hunt 0.88, farm 0.12 | sh_recall 0.72, sh_vis 0.28 |
| N10g | econ + tower aggro | +2.15 | shove 0.75, shove_hunt 0.25 | sh_recall 0.93, sh_vis 0.07 |
| P4 | four-action baseline | +0.17 | shove 1.00 | shove 1.00 |
| P4 | econ (0.8) | +1.06 | pz_bully 1.00 | shove 1.00 |
| P4 | econ, 0.85 | +1.26 | pz_bully 1.00 | shove 1.00 |
| P4 | econ, 0.9 | +1.50 | pz_bully 1.00 | shove 1.00 |
| P4 | econ + tower aggro | +1.06 | pz_bully 1.00 | shove 1.00 |
| P10 | four-action baseline | +0.84 | shove 1.00 | shove 1.00 |
| P10 | econ (0.8) | +1.04 | shove 1.00 | shove 1.00 |
| P10 | econ, 0.85 | +1.11 | pz_hunt 1.00 | shove 1.00 |
| P10 | econ, 0.9 | +1.27 | pz_hunt 1.00 | shove 1.00 |
| P10 | econ + tower aggro | +1.04 | shove 1.00 | shove 1.00 |
| P10g | four-action baseline | +4.54 | pz_hunt 0.90, shove 0.10 | shove 0.76, pz_thin 0.24 |
| P10g | econ (0.8) | +4.19 | pz_hunt 1.00 | shove 1.00 |
| P10g | econ, 0.85 | +3.92 | pz_hunt 1.00 | shove 1.00 |
| P10g | econ, 0.9 | +3.65 | pz_hunt 1.00 | shove 1.00 |
| P10g | econ + tower aggro | +4.19 | pz_hunt 1.00 | shove 1.00 |
| P10gl | four-action baseline | +5.38 | shove 1.00 | shove 1.00 |
| P10gl | econ (0.8) | +4.35 | pz_hunt 1.00 | shove 1.00 |
| P10gl | econ, 0.85 | +4.05 | pz_hunt 1.00 | shove 1.00 |
| P10gl | econ, 0.9 | +3.75 | pz_hunt 1.00 | shove 1.00 |
| P10gl | econ + tower aggro | +4.35 | pz_hunt 1.00 | shove 1.00 |
| P10b | four-action baseline | -1.23 | shove 0.68, pz_keep 0.32 | pz_walk 0.48, shove 0.44, pz_cautious 0.08 |
| P10b | econ (0.8) | -0.05 | shove 0.62, pz_hunt 0.38 | pz_cautious 0.61, shove 0.39 |
| P10b | econ, 0.85 | +0.25 | shove 0.60, pz_hunt 0.40 | pz_cautious 0.65, shove 0.35 |
| P10b | econ, 0.9 | +0.53 | shove 0.56, pz_hunt 0.44 | pz_cautious 0.71, shove 0.29 |
| P10b | econ + tower aggro | -0.05 | shove 0.62, pz_hunt 0.38 | pz_cautious 0.61, shove 0.39 |

## DIAGNOSTIC: without plate gold (the game after a tower's plates fall), 0.8 and 0.9

Seven scenarios replayed with `plate_gold` 0 (same seeds as above).

| scenario | configuration | value to A | A's equilibrium | B's equilibrium |
|---|---|---|---|---|
| F4 | econ (0.8) | +0.80 | shove 1.00 | walk_up_shove 1.00 |
| F4 | econ_pg0 | +1.14 | sh_vis 1.00 | sh_vis 1.00 |
| F4 | econ, 0.9 | +1.10 | shove 0.70, freeze 0.30 | walk_up_shove 0.62, sh_vis 0.38 |
| F4 | ot90_pg0 | +1.65 | farm 1.00 | walk_up 1.00 |
| F10 | econ (0.8) | +1.00 | shove 1.00 | walk_up_shove 1.00 |
| F10 | econ_pg0 | +1.22 | sh_vis 1.00 | sh_vis 1.00 |
| F10 | econ, 0.9 | +1.23 | shove 0.77, freeze 0.23 | walk_up_shove 0.57, sh_vis 0.43 |
| F10 | ot90_pg0 | +1.57 | sh_vis 0.59, farm 0.41 | walk_up 0.98, sh_vis 0.02 |
| F10g | econ (0.8) | +5.20 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10g | econ_pg0 | +3.02 | sh_vis 0.90, freeze_bully 0.09, farm 0.00 | cautious 0.69, respect 0.21, sh_vis 0.11 |
| F10g | econ, 0.9 | +3.84 | sh_vis 0.95, freeze_bully 0.04, freeze_hunt 0.02 | walk_up_thin 0.39, walk_up_shove 0.35, respect 0.26 |
| F10g | ot90_pg0 | +2.19 | farm 0.58, freeze_hunt 0.32, freeze 0.07, freeze_bully 0.03 | sh_vis 0.81, walk_up 0.07, cautious 0.06, respect 0.06 |
| N3 | econ (0.8) | +0.31 | shove 1.00 | sh_vis 1.00 |
| N3 | econ_pg0 | +0.22 | sh_vis 1.00 | sh_vis 1.00 |
| N3 | econ, 0.9 | +0.24 | shove 0.58, sh_vis 0.40, thin 0.01 | shove 0.69, sh_vis 0.17, thin 0.14 |
| N3 | ot90_pg0 | +0.07 | farm 0.57, hunt 0.41, sh_vis 0.03 | farm 0.64, hunt 0.19, sh_vis 0.17 |
| N10 | econ (0.8) | +0.16 | shove 1.00 | shove 1.00 |
| N10 | econ_pg0 | +0.15 | sh_vis 1.00 | sh_vis 1.00 |
| N10 | econ, 0.9 | +0.16 | shove 0.60, sh_vis 0.25, thin 0.15 | shove 0.51, sh_vis 0.31, thin 0.18 |
| N10 | ot90_pg0 | +0.06 | farm 0.52, hunt 0.48 | farm 0.78, sh_vis 0.22 |
| N10g | econ (0.8) | +3.91 | sh_vis 0.70, shove 0.30 | shove 0.69, sh_dive 0.31 |
| N10g | econ_pg0 | +2.12 | sh_vis 0.98, hunt 0.02 | sh_vis 0.67, thin 0.33 |
| N10g | econ, 0.9 | +2.55 | sh_vis 0.94, shove_hunt 0.06 | sh_vis 0.65, thin 0.35 |
| N10g | ot90_pg0 | +1.45 | build_shove 0.65, hunt 0.34, farm 0.01 | sh_vis 0.65, hunt 0.18, sh_recall 0.16 |
| P10 | econ (0.8) | +1.11 | pz_hunt 1.00 | shove 1.00 |
| P10 | econ_pg0 | +1.77 | pz_hunt 1.00 | shove 1.00 |
| P10 | econ, 0.9 | +1.38 | pz_hunt 1.00 | shove 1.00 |
| P10 | ot90_pg0 | +2.02 | pz_hunt 1.00 | shove 1.00 |

## Validation: independent seeds (SeedSequence 79,700, 128 games per cell)

| configuration | B shoves at equal power (F10 / N10) | B concedes / defends vs x1.76 (F10g) | B concedes / defends vs x2.04 (F10gl) | A freezes ahead (F10g / P10g) | A uses Vision (N10g / N3) | value of a x1.76 lead (F10g / N10g) | distinct equilibria |
|---|---|---|---|---|---|---|---|
| econ (0.8) | nan / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.40 | 0.24 / 0.00 | +5.06 / +3.84 | 5 |
| econ, 0.9 | nan / 1.00 | 0.00 / 0.35 | 0.00 / 0.41 | 0.04 / 1.00 | 0.66 / 0.33 | +3.75 / +2.40 | 6 |
| econ + tower aggro | nan / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.40 | 0.24 / 0.00 | +5.06 / +3.84 | 5 |

| scenario | configuration | value to A | A's equilibrium | B's equilibrium |
|---|---|---|---|---|
| F4 | econ (0.8) | +0.73 | shove 1.00 | walk_up_shove 1.00 |
| F4 | econ, 0.9 | +0.98 | shove 0.68, freeze 0.31, farm 0.01 | walk_up_shove 0.80, walk_up 0.14, sh_vis 0.06 |
| F4 | econ + tower aggro | +0.73 | shove 1.00 | walk_up_shove 1.00 |
| F10g | econ (0.8) | +5.06 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10g | econ, 0.9 | +3.75 | sh_vis 0.96, freeze_hunt 0.04 | walk_up_shove 0.65, walk_up_thin 0.35 |
| F10g | econ + tower aggro | +5.06 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10gl | econ (0.8) | +5.09 | sh_vis 1.00 | walk_up_shove 1.00 |
| F10gl | econ, 0.9 | +3.86 | sh_vis 0.99, freeze_hunt 0.01 | walk_up_shove 0.59, walk_up_thin 0.41 |
| F10gl | econ + tower aggro | +5.09 | sh_vis 1.00 | walk_up_shove 1.00 |
| N3 | econ (0.8) | +0.20 | shove 1.00 | shove 1.00 |
| N3 | econ, 0.9 | +0.11 | shove 0.56, sh_vis 0.33, thin 0.11 | shove 0.87, thin 0.13, sh_vis 0.01 |
| N3 | econ + tower aggro | +0.20 | shove 1.00 | shove 1.00 |
| N10 | econ (0.8) | -0.10 | shove 1.00 | shove 1.00 |
| N10 | econ, 0.9 | -0.05 | shove 0.61, sh_vis 0.23, thin 0.16 | shove 0.67, sh_vis 0.20, build_shove 0.13 |
| N10 | econ + tower aggro | -0.10 | shove 1.00 | shove 1.00 |
| N10g | econ (0.8) | +3.84 | shove 0.76, sh_vis 0.24 | shove 0.88, thin 0.12 |
| N10g | econ, 0.9 | +2.40 | sh_vis 0.66, shove_hunt 0.27, shove 0.06 | shove 0.79, thin 0.21, build_shove 0.00 |
| N10g | econ + tower aggro | +3.84 | shove 0.76, sh_vis 0.24 | shove 0.88, thin 0.12 |
| P10g | econ (0.8) | +4.86 | shove 0.60, pz_bully 0.40 | shove 0.97, pz_back 0.03 |
| P10g | econ, 0.9 | +3.92 | pz_bully 1.00 | shove 1.00 |
| P10g | econ + tower aggro | +4.86 | shove 0.60, pz_bully 0.40 | shove 0.97, pz_back 0.03 |
