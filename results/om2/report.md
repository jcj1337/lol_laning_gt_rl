# OM2 Bayesian opponent-modelling pilot: generated tables

Pre-registration `docs\opponent_modelling\om2_preregistration.md` SHA-256 `42f279736539ffa1…` (frozen 2026-10-17 04:46:18; amendments: 0). Values: the evaluated arm's earned-gold lead at turn 114 (= E_20 at turn 94 in expectation), resources, mean over 64 paired-seat pairs per cell; contrasts are treatment − control on the same games.

## Readiness smoke

| check | result |
|---|---|
| control: run completed | True |
| treatment: run completed | True |
| rerun: run completed | True |
| both arms start from the same rollout (identical sampled actions in update 1) | True |
| the arms diverge after the first update | True |
| control: belief inputs constant at the prior | True |
| treatment: belief inputs vary | True |
| treatment: belief inputs are a posterior (in [0, 1], rows sum to 1) | True |
| the last-action field shows game starts and previous actions | True |
| library opponents drawn | True |
| self-play games drawn | True |
| treatment updated beliefs | True |
| no illegal action sampled | True |
| truncations bootstrapped (some games truncated) | True |
| treatment re-run bit-identical | True |
| initial weights = widened ext/baseline:0 | True |
| policy moved (KL > 0, below 0.05) | True |

Throughput (transitions/s, 3 concurrent runs): control 2653, treatment 1563, rerun 1600

## Training

| run | status | transitions | updates | wall (min) | transitions/s | KL (last 10%) | entropy | explained var. | belief spread | 'unknown' mass | illegal | digest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control_s0 | complete | 4,097,441 | 1335 | 31 | 2188 | 0.0002 | 1.74 | 0.71 | 0.000 | 0.077 | 0 | c9deec555560 |
| treatment_s0 | complete | 4,097,441 | 1335 | 48 | 1416 | 0.0002 | 1.69 | 0.94 | 0.175 | 0.004 | 0 | 99def6a86de2 |
| control_s1 | complete | 4,098,698 | 1334 | 31 | 2209 | 0.0002 | 1.69 | 0.71 | 0.000 | 0.077 | 0 | 13167be2e70b |
| treatment_s1 | complete | 4,098,698 | 1334 | 48 | 1419 | 0.0002 | 1.61 | 0.94 | 0.188 | 0.003 | 0 | 7660798fdcc7 |
| control_s2 | complete | 4,096,601 | 1329 | 31 | 2228 | 0.0002 | 1.67 | 0.66 | 0.000 | 0.077 | 0 | 01ffc9d024e4 |
| treatment_s2 | complete | 4,096,601 | 1329 | 42 | 1645 | 0.0002 | 1.64 | 0.93 | 0.184 | 0.002 | 0 | 55833a50bd65 |

Belief spread: the mean over the 13 belief inputs of their standard deviation in a rollout (0 in the control by construction). 'unknown' mass: the mean posterior of the uniform catch-all.

Learner's mean game return against each fixed training opponent (resources; games of 512 or 150 turns, so levels are not comparable across opponents; mean over seeds):

| arm | opponent | first 10% of updates | last 10% | change |
|---|---|---|---|---|
| control | attacker | +2.40 | +3.14 | +0.75 |
| control | freezer | +0.42 | +2.12 | +1.70 |
| control | fz_guard_sh | +27.83 | +50.41 | +22.58 |
| control | g_bully | +33.87 | +49.22 | +15.35 |
| control | g_farm | +11.58 | +18.61 | +7.03 |
| control | g_safe | +13.23 | +18.50 | +5.27 |
| control | g_shove | +31.94 | +46.04 | +14.10 |
| control | g_shove_ward | +23.52 | +38.76 | +15.24 |
| control | gapper | +4.77 | +7.02 | +2.25 |
| control | mix_bully_safe | +27.72 | +29.35 | +1.63 |
| control | mix_shove_farm | +26.71 | +40.43 | +13.72 |
| control | peer | +2.12 | +2.52 | +0.41 |
| control | short | -1.89 | -1.95 | -0.06 |
| control | switch_farm_bully_40 | +30.98 | +44.94 | +13.96 |
| control | switch_shove_safe_60 | +14.41 | +21.88 | +7.47 |
| treatment | attacker | +2.21 | +5.11 | +2.89 |
| treatment | freezer | +2.97 | +4.44 | +1.47 |
| treatment | fz_guard_sh | +30.15 | +55.45 | +25.30 |
| treatment | g_bully | +33.75 | +65.62 | +31.87 |
| treatment | g_farm | +12.93 | +30.15 | +17.22 |
| treatment | g_safe | +15.77 | +23.58 | +7.82 |
| treatment | g_shove | +33.31 | +51.41 | +18.10 |
| treatment | g_shove_ward | +28.27 | +43.41 | +15.14 |
| treatment | gapper | +3.26 | +6.51 | +3.25 |
| treatment | mix_bully_safe | +26.44 | +34.28 | +7.84 |
| treatment | mix_shove_farm | +29.44 | +45.10 | +15.66 |
| treatment | peer | +1.53 | +3.09 | +1.56 |
| treatment | short | -2.78 | -1.65 | +1.13 |
| treatment | switch_farm_bully_40 | +32.30 | +61.37 | +29.08 |
| treatment | switch_shove_safe_60 | +16.63 | +27.02 | +10.39 |

## Primary contrasts (seeds as units; t interval with 2 df)

| family | seed 0 | seed 1 | seed 2 | mean ± 95% | seeds + / − |
|---|---|---|---|---|---|
| E_in: archetypes (in the library) | +2.57 | +1.25 | +2.26 | +2.03 ± 1.72 | 3 / 0 |
| E_out: held-out scripted (descriptive) | +2.82 | +0.64 | +2.06 | +1.84 ± 2.74 | 3 / 0 |
| R1: competent learned | +0.05 | +0.58 | +0.65 | +0.43 ± 0.81 | 3 / 0 |
| R2: held-out mixtures and switchers | +1.34 | +0.36 | +1.15 | +0.95 ± 1.29 | 3 / 0 |
| R3: treatment vs control (the treatment's lead) | -0.33 | +0.79 | +0.50 | +0.32 ± 1.45 | 2 / 1 |

Pooled over games (descriptive; 1.96 SE):

| family | lead at 114 | lane lead at 94 | lead at 150 |
|---|---|---|---|
| E_in | +2.03 ± 0.23 | +1.67 ± 0.20 | +2.97 ± 0.25 |
| E_out | +1.84 ± 0.31 | +1.57 ± 0.30 | +2.06 ± 0.34 |
| R1 | +0.43 ± 0.24 | +0.32 ± 0.21 | +0.67 ± 0.28 |
| R2 | +0.95 ± 0.27 | +0.69 ± 0.24 | +1.57 ± 0.32 |
| R3 | +0.32 ± 0.46 | n/a | n/a |

## Classification (pre-registration §4)

| criterion | result |
|---|---|
| E_in gain (mean ≥ +0.25, every seed > 0) | True |
| R1 loss (mean ≤ −0.25, every seed < 0) | False |
| R2 loss (mean ≤ −0.25, every seed < 0) | False |
| R3 loss (mean ≤ −0.25, every seed < 0) | False |
| **outcome** | **1: modelling pays and is safe** |

## Per opponent (lead at 114; arm values are means over seeds)

| family | opponent | control | treatment | T − C seed 0 | T − C seed 1 | T − C seed 2 | T − C mean |
|---|---|---|---|---|---|---|---|
| E_in | g_farm | +6.18 | +8.72 | +3.48 | +3.13 | +1.03 | +2.55 |
| E_in | g_safe | +7.92 | +9.83 | +1.69 | +1.34 | +2.70 | +1.91 |
| E_in | g_bully | +11.89 | +12.40 | +0.60 | -1.46 | +2.40 | +0.51 |
| E_in | g_shove | +11.72 | +14.71 | +3.26 | +2.61 | +3.09 | +2.99 |
| E_in | fz_guard_sh | +11.64 | +14.32 | +3.30 | +1.81 | +2.94 | +2.68 |
| E_in | g_shove_ward | +11.43 | +12.96 | +3.10 | +0.05 | +1.42 | +1.53 |
| E_out | g_shove_sr | +11.67 | +14.17 | +3.45 | +1.56 | +2.48 | +2.50 |
| E_out | wk_shove_reset_sh | +11.88 | +13.98 | +3.27 | +0.85 | +2.20 | +2.10 |
| E_out | g_shove_edge | +13.08 | +15.20 | +2.76 | +1.60 | +1.99 | +2.12 |
| E_out | g_slow_guard | +11.02 | +11.66 | +1.80 | -1.44 | +1.58 | +0.64 |
| R1 | base:3 | -0.08 | +0.04 | +0.21 | -0.24 | +0.40 | +0.12 |
| R1 | base:4 | -0.70 | -0.27 | -0.05 | +0.50 | +0.85 | +0.44 |
| R1 | base:5 | -0.78 | -0.27 | +0.14 | +0.73 | +0.66 | +0.51 |
| R1 | attacker_b | +0.62 | +1.05 | +1.10 | +0.41 | -0.22 | +0.43 |
| R1 | freezer_b | +0.41 | +1.00 | -0.91 | +1.22 | +1.45 | +0.59 |
| R1 | gapper_b | +2.21 | +2.41 | -0.37 | +0.72 | +0.24 | +0.19 |
| R1 | short_b | -0.80 | -0.23 | +0.64 | +0.15 | +0.93 | +0.57 |
| R1 | c1_policy | -0.55 | +0.01 | -0.35 | +1.16 | +0.88 | +0.56 |
| R2 | mix_bully_farm | +9.53 | +9.01 | +0.75 | -2.02 | -0.28 | -0.52 |
| R2 | mix_shove_safe | +9.89 | +11.63 | +1.29 | +1.55 | +2.39 | +1.74 |
| R2 | switch_safe_shove_30 | +13.22 | +14.07 | +0.88 | +0.67 | +1.00 | +0.85 |
| R2 | switch_bully_farm_50 | +7.57 | +9.31 | +2.45 | +1.23 | +1.52 | +1.73 |

## Mechanism

The treatment's belief in its evaluation games: MAP type correct after t turns (all seeds and seats):

| archetype | 10 | 20 | 40 | 94 |
|---|---|---|---|---|
| g_farm | 100% | 100% | 100% | 100% |
| g_safe | 100% | 100% | 100% | 100% |
| g_bully | 98% | 100% | 100% | 100% |
| g_shove | 100% | 100% | 100% | 100% |
| fz_guard_sh | 1% | 26% | 44% | 49% |
| g_shove_ward | 99% | 99% | 100% | 100% |

Posterior mass on the catch-alls ('mixed' + 'unknown') against the held-out mixtures and switchers:

| opponent | 10 | 20 | 40 | 94 |
|---|---|---|---|---|
| mix_bully_farm | 8% | 13% | 20% | 34% |
| mix_shove_safe | 63% | 80% | 94% | 98% |
| switch_safe_shove_30 | 0% | 0% | 57% | 0% |
| switch_bully_farm_50 | 0% | 0% | 0% | 0% |

Belief-swap probes: mean total-variation distance between the policy with a one-type belief and with the prior, on the treatment's own observations against the archetypes (the control's is 0 by construction):

| seed | arm | states | g_farm | g_safe | g_bully | g_shove | fz_guard_sh | g_shove_ward | peer | attacker | freezer | gapper | short | mixed | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | control | 676 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0 | treatment | 676 | 0.168 | 0.283 | 0.252 | 0.175 | 0.190 | 0.168 | 0.120 | 0.126 | 0.137 | 0.093 | 0.122 | 0.066 | 0.092 |
| 1 | control | 637 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | treatment | 637 | 0.222 | 0.262 | 0.289 | 0.161 | 0.174 | 0.202 | 0.103 | 0.096 | 0.067 | 0.099 | 0.099 | 0.080 | 0.139 |
| 2 | control | 660 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | treatment | 660 | 0.179 | 0.281 | 0.320 | 0.145 | 0.142 | 0.150 | 0.092 | 0.099 | 0.092 | 0.079 | 0.096 | 0.064 | 0.082 |

## Exploratory reference (NOT pre-registered; added after the classification; cannot change the outcome)

pi_s = ext/baseline:s, the common start of both arms (no opponent information), on exactly the games of the evaluation cells; arm − pi_s is paired. Seeds as units (t interval, 2 df).

| family | pi_s's value | control − pi_s | treatment − pi_s |
|---|---|---|---|
| E_in | +5.01 ± 1.38 | +5.12 ± 1.45 | +7.15 ± 2.66 |
| E_out | +5.23 ± 1.75 | +6.68 ± 1.66 | +8.52 ± 4.00 |
| R1 | +0.63 ± 1.04 | -0.59 ± 0.97 | -0.16 ± 0.30 |
| R2 | +6.63 ± 1.22 | +3.42 ± 0.91 | +4.37 ± 1.81 |

Head to head against its own start pi_s (the arm's lead at 114):

| arm | lead vs pi_s | per seed |
|---|---|---|
| control | -0.12 ± 1.51 | +0.36, +0.07, -0.81 |
| treatment | -0.08 ± 1.26 | -0.66, +0.27, +0.15 |

Per opponent (means over seeds):

| family | opponent | pi_s | control − pi_s | treatment − pi_s |
|---|---|---|---|---|
| E_in | g_farm | +2.51 | +3.67 | +6.21 |
| E_in | g_safe | +6.90 | +1.02 | +2.93 |
| E_in | g_bully | +9.07 | +2.82 | +3.33 |
| E_in | g_shove | +4.30 | +7.43 | +10.42 |
| E_in | fz_guard_sh | +3.75 | +7.90 | +10.58 |
| E_in | g_shove_ward | +3.53 | +7.90 | +9.43 |
| E_out | g_shove_sr | +4.29 | +7.38 | +9.88 |
| E_out | wk_shove_reset_sh | +3.64 | +8.24 | +10.34 |
| E_out | g_shove_edge | +4.83 | +8.25 | +10.37 |
| E_out | g_slow_guard | +8.16 | +2.86 | +3.51 |
| R1 | base:3 | +0.76 | -0.85 | -0.72 |
| R1 | base:4 | +0.18 | -0.88 | -0.45 |
| R1 | base:5 | -0.18 | -0.59 | -0.08 |
| R1 | attacker_b | +0.94 | -0.31 | +0.11 |
| R1 | freezer_b | +1.00 | -0.59 | -0.00 |
| R1 | gapper_b | +2.28 | -0.07 | +0.13 |
| R1 | short_b | +0.03 | -0.84 | -0.26 |
| R1 | c1_policy | +0.04 | -0.59 | -0.03 |
| R2 | mix_bully_farm | +6.76 | +2.77 | +2.25 |
| R2 | mix_shove_safe | +8.04 | +1.85 | +3.59 |
| R2 | switch_safe_shove_30 | +5.96 | +7.26 | +8.11 |
| R2 | switch_bully_farm_50 | +5.79 | +1.79 | +3.52 |
