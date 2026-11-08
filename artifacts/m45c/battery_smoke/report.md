# M4.5c strategic battery (smoke)

Paired differences candidate - policy's own choice (resources). Primary statistic per family: Q (critic-bootstrapped) for natural families, G (observed discounted return) for constructed ones (exception 2); the other one at 50 turns is shown beside it. Seed mean +- t interval over the evaluated policies, then each policy.

## A_wave (natural; horizon 50; primary Q)

one-turn candidates:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|
| modal | 1t:LH | -0.03 | +0.02 | -0.61 | -0.69 | -0.57 | -0.69 |
| modal | 1t:TN | -0.01 | +0.06 | -0.57 | -1.33 | -1.21 | -1.33 |
| modal | 1t:SL | -0.00 | +0.04 | -0.47 | -0.33 | -0.13 | -0.33 |
| modal | 1t:SH | +0.13 | +0.35 | -0.98 | -0.58 | -0.61 | -0.58 |
| policy | 1t:LH | -0.01 | -0.13 | -1.26 | -0.38 | -0.24 | -0.38 |
| policy | 1t:TN | +0.01 | -0.09 | -1.20 | -1.75 | -1.59 | -1.75 |
| policy | 1t:SL | +0.02 | -0.09 | -1.20 | -1.48 | -1.21 | -1.48 |
| policy | 1t:SH | -0.01 | -0.00 | +0.66 | +0.54 | +1.05 | +0.54 |

sustained plans:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|
| modal | plan:TNx4 | -0.01 | -0.01 | -0.17 | -1.07 | -1.05 | -1.07 |
| modal | plan:SHx4 | +0.13 | +0.33 | -0.30 | -1.30 | -1.33 | -1.30 |
| policy | plan:TNx4 | +0.01 | +0.26 | -0.52 | -0.94 | -0.31 | -0.94 |
| policy | plan:SHx4 | -0.01 | +0.21 | +0.29 | -0.40 | -0.12 | -0.40 |

Validation (§6.3): smoke: fail -> re-run the family at 150 turns (bias +0.03 ± 0.66, slope 0.73, contradictions 0, pairs 6)

## A_wave_early (early; horizon 50; primary Q)

one-turn candidates:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|
| modal | 1t:LH | +0.03 | -0.30 | -0.79 | -0.43 | -0.50 | -0.43 |
| modal | 1t:TN | +0.09 | -0.40 | -0.68 | -0.37 | -0.33 | -0.37 |
| modal | 1t:SL | +0.09 | -0.40 | -0.68 | -0.37 | -0.33 | -0.37 |
| modal | 1t:SH | +0.09 | -0.12 | -0.87 | +0.91 | +0.86 | +0.91 |
| policy | 1t:LH | -0.24 | -1.30 | +0.34 | -0.62 | -0.54 | -0.62 |
| policy | 1t:TN | -0.18 | -0.25 | +1.46 | +0.31 | +0.42 | +0.31 |
| policy | 1t:SL | -0.18 | -0.25 | +1.46 | +0.31 | +0.42 | +0.31 |
| policy | 1t:SH | -0.27 | -0.28 | +1.53 | +2.46 | +3.13 | +2.46 |

sustained plans:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|
| modal | plan:TNx4 | +0.09 | -0.14 | +0.57 | +2.55 | +2.87 | +2.55 |
| modal | plan:SHx4 | +0.09 | -0.17 | +0.93 | +1.75 | +1.31 | +1.75 |
| policy | plan:TNx4 | -0.18 | -0.86 | +0.94 | -0.17 | +0.02 | -0.17 |
| policy | plan:SHx4 | -0.27 | -1.07 | -0.69 | -1.09 | -0.89 | -1.09 |

Validation (§6.3): smoke: fail -> re-run the family at 150 turns (bias -2.22 ± 0.85, slope 0.85, contradictions 0, pairs 6)

## B_freeze (natural; horizon 60; primary Q)

one-turn candidates:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|
| modal | 1t:keep | +0.13 | +0.08 | -0.08 | -1.24 | -1.30 | -1.24 |
| modal | 1t:guard | -0.22 | -0.46 | +0.03 | +0.95 | +0.97 | +0.95 |
| modal | 1t:break | +0.21 | -0.00 | -0.63 | +0.71 | +0.84 | +0.71 |
| policy | 1t:keep | +0.13 | +0.08 | +0.58 | +0.16 | +0.19 | +0.16 |
| policy | 1t:guard | -0.24 | -0.26 | -1.22 | -1.58 | -1.47 | -1.58 |
| policy | 1t:break | +0.21 | +0.18 | -0.22 | +0.70 | +0.42 | +0.70 |
| wk_back | 1t:keep | +0.09 | +0.06 | -0.02 | -0.19 | -0.23 | -0.19 |
| wk_back | 1t:guard | +0.01 | +0.19 | +0.43 | +0.62 | +0.85 | +0.62 |
| wk_back | 1t:break | +0.20 | +0.06 | +0.46 | +0.18 | +0.49 | +0.18 |
| wk_shove | 1t:keep | +0.13 | +0.06 | +0.02 | +0.04 | +0.04 | +0.04 |
| wk_shove | 1t:guard | -0.22 | -0.69 | -0.47 | -1.26 | -1.35 | -1.26 |
| wk_shove | 1t:break | +0.21 | -0.08 | +0.24 | +0.37 | +0.34 | +0.37 |
| wk_shove_reset | 1t:keep | +0.13 | +0.06 | +0.02 | +0.04 | +0.04 | +0.04 |
| wk_shove_reset | 1t:guard | -0.22 | -0.69 | -0.47 | -1.26 | -1.35 | -1.26 |
| wk_shove_reset | 1t:break | +0.21 | -0.08 | +0.24 | +0.37 | +0.34 | +0.37 |

sustained plans:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|
| modal | plan:keep | +0.13 | +0.04 | -0.13 | -1.17 | -1.43 | -1.17 |
| modal | plan:guard | +0.13 | +0.04 | -0.13 | -0.88 | -1.00 | -0.88 |
| policy | plan:keep | +0.13 | -0.05 | +1.47 | +1.13 | +1.27 | +1.13 |
| policy | plan:guard | +0.13 | -0.05 | +1.47 | +1.13 | +1.27 | +1.13 |
| wk_back | plan:keep | +0.09 | +0.20 | +0.70 | +0.18 | +0.55 | +0.18 |
| wk_back | plan:guard | +0.09 | +0.10 | +0.49 | +0.45 | +0.53 | +0.45 |
| wk_shove | plan:keep | +0.13 | +0.16 | +0.27 | -0.51 | -0.67 | -0.51 |
| wk_shove | plan:guard | +0.13 | +0.12 | +0.47 | -0.10 | -0.19 | -0.10 |
| wk_shove_reset | plan:keep | +0.13 | +0.16 | +0.27 | -0.51 | -0.67 | -0.51 |
| wk_shove_reset | plan:guard | +0.13 | +0.12 | +0.47 | -0.10 | -0.19 | -0.10 |

Validation (§6.3): smoke: fail -> re-run the family at 150 turns (bias +0.51 ± 0.26, slope 0.86, contradictions 0, pairs 5)

## B_frozen (natural; horizon 50; primary Q)

one-turn candidates:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|
| fz_bully_sh | 1t:shove | +0.02 | +0.04 | +0.86 | +0.83 | +1.17 | +0.83 |
| fz_bully_sh | 1t:farm | -0.13 | -0.51 | +0.65 | +1.09 | +1.03 | +1.09 |
| fz_bully_sh | 1t:step_back | +0.01 | -0.58 | +0.01 | +0.64 | +0.69 | +0.64 |
| fz_bully_sh | 1t:recall | +nan | +nan | +nan | +nan | +nan |  |
| fz_guard_sh | 1t:shove | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| fz_guard_sh | 1t:farm | +0.08 | -0.35 | -0.99 | -0.83 | -0.84 | -0.83 |
| fz_guard_sh | 1t:step_back | -0.64 | -0.19 | -0.27 | -1.28 | -1.21 | -1.28 |
| fz_guard_sh | 1t:recall | +nan | +nan | +nan | +nan | +nan |  |
| fz_passive_sh | 1t:shove | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| fz_passive_sh | 1t:farm | +0.08 | -0.12 | -0.47 | +0.45 | +0.42 | +0.45 |
| fz_passive_sh | 1t:step_back | -0.64 | +0.04 | -1.48 | -0.64 | -0.71 | -0.64 |
| fz_passive_sh | 1t:recall | +nan | +nan | +nan | +nan | +nan |  |
| modal | 1t:shove | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| modal | 1t:farm | +0.00 | +0.10 | -0.10 | -1.94 | -2.36 | -1.94 |
| modal | 1t:step_back | -0.48 | -0.00 | -0.68 | -1.66 | -1.82 | -1.66 |
| modal | 1t:recall | +nan | +nan | +nan | +nan | +nan |  |
| policy | 1t:shove | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| policy | 1t:farm | +0.05 | -0.32 | -0.98 | -0.30 | -0.95 | -0.30 |
| policy | 1t:step_back | -0.51 | -0.17 | -0.75 | -0.40 | -1.03 | -0.40 |
| policy | 1t:recall | +nan | +nan | +nan | +nan | +nan |  |

sustained plans:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|
| fz_bully_sh | plan:shovex3 | +0.02 | -0.47 | +0.15 | +1.86 | +2.11 | +1.86 |
| fz_guard_sh | plan:shovex3 | +0.00 | -0.34 | -0.35 | -0.65 | -0.84 | -0.65 |
| fz_passive_sh | plan:shovex3 | +0.00 | -0.32 | -0.30 | -0.49 | -0.56 | -0.49 |
| modal | plan:shovex3 | +0.00 | +0.16 | -0.67 | -1.34 | -1.69 | -1.34 |
| policy | plan:shovex3 | +0.00 | -0.26 | -1.74 | -1.58 | -1.76 | -1.58 |

Validation (§6.3): smoke: fail -> re-run the family at 150 turns (bias +0.40 ± 0.89, slope 0.32, contradictions 0, pairs 4)

## C_trade (natural; horizon 50; primary Q)

one-turn candidates:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|
| g_wary | 1t:trade | +0.07 | -0.30 | -0.33 | -0.29 | -0.39 | -0.29 |
| g_wary | 1t:hold | +0.04 | -0.01 | -0.03 | +0.31 | +0.21 | +0.31 |
| g_wary | 1t:safe | +0.04 | -0.01 | -0.03 | +0.31 | +0.21 | +0.31 |
| modal | 1t:trade | -0.18 | -0.83 | -0.76 | -1.72 | -2.46 | -1.72 |
| modal | 1t:hold | +0.12 | +0.20 | +0.11 | -0.42 | -0.59 | -0.42 |
| modal | 1t:safe | +0.12 | +0.20 | +0.11 | -0.42 | -0.59 | -0.42 |
| policy | 1t:trade | -0.17 | -0.06 | -0.57 | -0.68 | -1.26 | -0.68 |
| policy | 1t:hold | -0.08 | +0.07 | +0.36 | -0.49 | -1.01 | -0.49 |
| policy | 1t:safe | -0.08 | +0.07 | +0.36 | -0.49 | -1.01 | -0.49 |

Validation (§6.3): smoke: fail -> re-run the family at 150 turns (bias -0.57 ± 1.13, slope 20.21, contradictions 0, pairs 3)

## C_trade_early (early; horizon 50; primary Q)

one-turn candidates:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|
| g_wary | 1t:trade | -0.27 | -0.16 | +0.35 | +0.20 | +0.09 | +0.20 |
| g_wary | 1t:hold | -0.08 | -0.35 | -0.22 | +0.19 | +0.09 | +0.19 |
| g_wary | 1t:safe | -0.15 | -0.14 | -0.14 | +0.13 | +0.09 | +0.13 |
| modal | 1t:trade | -0.03 | -0.04 | +0.30 | +1.96 | +1.85 | +1.96 |
| modal | 1t:hold | -0.08 | -0.34 | -0.08 | -0.43 | -0.61 | -0.43 |
| modal | 1t:safe | -0.15 | -0.28 | -0.64 | -1.39 | -1.46 | -1.39 |
| policy | 1t:trade | -0.04 | +0.22 | +1.39 | +1.97 | +2.01 | +1.97 |
| policy | 1t:hold | -0.10 | -0.37 | -0.08 | -0.36 | -0.55 | -0.36 |
| policy | 1t:safe | -0.07 | -0.06 | -0.23 | -1.37 | -1.68 | -1.37 |

Validation (§6.3): smoke: fail -> re-run the family at 150 turns (bias +0.78 ± 1.31, slope 0.28, contradictions 0, pairs 3)

## D_ward (natural; horizon 50; primary Q)

one-turn candidates:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|
| g_bully | 1t:ward | -0.38 | -0.56 | +0.47 | +0.34 | +0.01 | +0.34 |
| g_bully | 1t:farm | -0.47 | -0.12 | -0.49 | -1.07 | -0.72 | -1.07 |
| modal | 1t:ward | +0.15 | +0.21 | +0.38 | +0.88 | +0.21 | +0.88 |
| modal | 1t:farm | -0.10 | +0.21 | +0.22 | +0.13 | -0.33 | +0.13 |
| policy | 1t:ward | -0.13 | -0.09 | -0.85 | -1.79 | -2.19 | -1.79 |
| policy | 1t:farm | -0.25 | -0.35 | -0.45 | -2.10 | -2.32 | -2.10 |

Validation (§6.3): smoke: insufficient (bias +nan ± nan, slope nan, contradictions 0, pairs 2)

## E_recall (natural; horizon 50; primary Q)

one-turn candidates:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|
| modal | 1t:recall | -0.23 | +0.12 | +1.18 | +0.67 | +0.45 | +0.67 |
| policy | 1t:recall | +0.03 | +0.15 | -1.46 | -1.26 | -0.75 | -1.26 |

sustained plans:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|
| modal | plan:stayx3 | -0.29 | -1.36 | -0.85 | -0.66 | -1.03 | -0.66 |
| modal | plan:step_backx3 | +0.00 | -0.77 | +0.27 | +0.62 | +0.27 | +0.62 |
| policy | plan:stayx3 | -0.18 | -1.17 | -2.31 | -0.32 | -0.18 | -0.32 |
| policy | plan:step_backx3 | +0.00 | -0.84 | -2.38 | -1.27 | -1.75 | -1.27 |

Validation (§6.3): smoke: fail -> re-run the family at 150 turns (bias -1.18 ± 1.44, slope -1.49, contradictions 0, pairs 3)

## F_freeze_c (constructed; horizon 60; primary G)

one-turn candidates:

| protocol | candidate | G1 | G5 | G20 | G50 | Q50 | G50 per policy |
|---|---|---|---|---|---|---|---|
| modal | 1t:keep | +nan | +nan | +nan | +nan | +nan |  |
| modal | 1t:guard | +nan | +nan | +nan | +nan | +nan |  |
| modal | 1t:break | +nan | +nan | +nan | +nan | +nan |  |
| policy | 1t:keep | +nan | +nan | +nan | +nan | +nan |  |
| policy | 1t:guard | +nan | +nan | +nan | +nan | +nan |  |
| policy | 1t:break | +nan | +nan | +nan | +nan | +nan |  |
| wk_back | 1t:keep | +nan | +nan | +nan | +nan | +nan |  |
| wk_back | 1t:guard | +nan | +nan | +nan | +nan | +nan |  |
| wk_back | 1t:break | +nan | +nan | +nan | +nan | +nan |  |
| wk_shove | 1t:keep | +nan | +nan | +nan | +nan | +nan |  |
| wk_shove | 1t:guard | +nan | +nan | +nan | +nan | +nan |  |
| wk_shove | 1t:break | +nan | +nan | +nan | +nan | +nan |  |
| wk_shove_reset | 1t:keep | +nan | +nan | +nan | +nan | +nan |  |
| wk_shove_reset | 1t:guard | +nan | +nan | +nan | +nan | +nan |  |
| wk_shove_reset | 1t:break | +nan | +nan | +nan | +nan | +nan |  |

sustained plans:

| protocol | candidate | G1 | G5 | G20 | G50 | Q50 | G50 per policy |
|---|---|---|---|---|---|---|---|
| modal | plan:keep | +nan | +nan | +nan | +nan | +nan |  |
| modal | plan:guard | +nan | +nan | +nan | +nan | +nan |  |
| policy | plan:keep | +nan | +nan | +nan | +nan | +nan |  |
| policy | plan:guard | +nan | +nan | +nan | +nan | +nan |  |
| wk_back | plan:keep | +nan | +nan | +nan | +nan | +nan |  |
| wk_back | plan:guard | +nan | +nan | +nan | +nan | +nan |  |
| wk_shove | plan:keep | +nan | +nan | +nan | +nan | +nan |  |
| wk_shove | plan:guard | +nan | +nan | +nan | +nan | +nan |  |
| wk_shove_reset | plan:keep | +nan | +nan | +nan | +nan | +nan |  |
| wk_shove_reset | plan:guard | +nan | +nan | +nan | +nan | +nan |  |

Validation (§6.3): smoke: pass (bias +0.48 ± 0.61, slope 1.11, contradictions 0, pairs 5)

## G_trade_c (constructed; horizon 50; primary G)

one-turn candidates:

| protocol | candidate | G1 | G5 | G20 | G50 | Q50 | G50 per policy |
|---|---|---|---|---|---|---|---|
| g_wary | 1t:trade | +nan | +nan | +nan | +nan | +nan |  |
| g_wary | 1t:hold | +nan | +nan | +nan | +nan | +nan |  |
| g_wary | 1t:safe | +nan | +nan | +nan | +nan | +nan |  |
| modal | 1t:trade | +nan | +nan | +nan | +nan | +nan |  |
| modal | 1t:hold | +nan | +nan | +nan | +nan | +nan |  |
| modal | 1t:safe | +nan | +nan | +nan | +nan | +nan |  |
| policy | 1t:trade | +nan | +nan | +nan | +nan | +nan |  |
| policy | 1t:hold | +nan | +nan | +nan | +nan | +nan |  |
| policy | 1t:safe | +nan | +nan | +nan | +nan | +nan |  |

Validation (§6.3): smoke: fail -> re-run the family at 150 turns (bias +0.65 ± 0.43, slope 1.34, contradictions 0, pairs 3)

## H_zone_c (constructed; horizon 50; primary G)

one-turn candidates:

| protocol | candidate | G1 | G5 | G20 | G50 | Q50 | G50 per policy |
|---|---|---|---|---|---|---|---|
| g_wary | 1t:trade | +nan | +nan | +nan | +nan | +nan |  |
| g_wary | 1t:hold | +nan | +nan | +nan | +nan | +nan |  |
| g_wary | 1t:shove | +nan | +nan | +nan | +nan | +nan |  |
| modal | 1t:trade | +nan | +nan | +nan | +nan | +nan |  |
| modal | 1t:hold | +nan | +nan | +nan | +nan | +nan |  |
| modal | 1t:shove | +nan | +nan | +nan | +nan | +nan |  |
| policy | 1t:trade | +nan | +nan | +nan | +nan | +nan |  |
| policy | 1t:hold | +nan | +nan | +nan | +nan | +nan |  |
| policy | 1t:shove | +nan | +nan | +nan | +nan | +nan |  |

sustained plans:

| protocol | candidate | G1 | G5 | G20 | G50 | Q50 | G50 per policy |
|---|---|---|---|---|---|---|---|
| g_wary | plan:bullyx10 | +nan | +nan | +nan | +nan | +nan |  |
| modal | plan:bullyx10 | +nan | +nan | +nan | +nan | +nan |  |
| policy | plan:bullyx10 | +nan | +nan | +nan | +nan | +nan |  |

Validation (§6.3): smoke: fail -> re-run the family at 150 turns (bias -0.85 ± 1.71, slope 0.48, contradictions 0, pairs 4)

## I_ward_c (constructed; horizon 50; primary G)

one-turn candidates:

| protocol | candidate | G1 | G5 | G20 | G50 | Q50 | G50 per policy |
|---|---|---|---|---|---|---|---|
| g_bully | 1t:ward | +nan | +nan | +nan | +nan | +nan |  |
| g_bully | 1t:farm | +nan | +nan | +nan | +nan | +nan |  |
| modal | 1t:ward | +nan | +nan | +nan | +nan | +nan |  |
| modal | 1t:farm | +nan | +nan | +nan | +nan | +nan |  |
| policy | 1t:ward | +nan | +nan | +nan | +nan | +nan |  |
| policy | 1t:farm | +nan | +nan | +nan | +nan | +nan |  |

Validation (§6.3): smoke: insufficient (bias +nan ± nan, slope nan, contradictions 0, pairs 2)
