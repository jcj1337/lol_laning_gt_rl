# M4.5c strategic battery (s0_full, s1_full, s2_full, s3_full, s4_full, s5_full)

Paired differences candidate - policy's own choice (resources). Primary statistic per family: Q (critic-bootstrapped) for natural families, G (observed discounted return) for constructed ones (exception 2); the other one at 50 turns is shown beside it. Seed mean +- t interval over the evaluated policies, then each policy.

## B_freeze (natural; horizon 60, re-run at 150 after failed validation on s4_full; primary Q)

one-turn candidates:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | Q150 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|---|
| modal | 1t:keep | -0.01 ± 0.06 | -0.05 ± 0.06 | -0.01 ± 0.13 | +0.03 ± 0.09 | +0.13 | +0.03 | -0.01 +0.15 -0.04 -0.03 +0.13 +0.00 |
| modal | 1t:guard | -0.08 ± 0.04 | -0.02 ± 0.05 | -0.03 ± 0.09 | -0.06 ± 0.13 | -0.24 | -0.04 | -0.04 -0.26 +0.02 -0.08 -0.10 +0.11 |
| modal | 1t:break | -0.01 ± 0.12 | -0.02 ± 0.09 | +0.04 ± 0.19 | -0.03 ± 0.17 | -0.02 | -0.02 | +0.12 +0.01 +0.14 -0.09 -0.07 -0.31 |
| policy | 1t:keep | -0.00 ± 0.04 | -0.02 ± 0.07 | +0.03 ± 0.10 | -0.03 ± 0.13 | +0.07 | -0.04 | +0.09 +0.04 -0.08 -0.05 +0.05 -0.24 |
| policy | 1t:guard | -0.10 ± 0.05 | -0.06 ± 0.05 | -0.07 ± 0.08 | -0.06 ± 0.13 | +0.08 | -0.05 | +0.03 -0.18 -0.05 -0.22 +0.10 -0.02 |
| policy | 1t:break | -0.00 ± 0.09 | -0.03 ± 0.07 | +0.02 ± 0.13 | -0.02 ± 0.20 | +0.24 | -0.02 | +0.27 -0.18 -0.19 -0.08 +0.16 -0.11 |

sustained plans:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | Q150 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|---|
| modal | plan:keep | -0.01 ± 0.06 | -0.06 ± 0.07 | -0.11 ± 0.10 | -0.12 ± 0.08 | -0.03 | -0.12 | +0.02 -0.16 -0.20 -0.12 -0.12 -0.16 |
| modal | plan:guard | -0.02 ± 0.03 | -0.01 ± 0.10 | -0.09 ± 0.14 | -0.09 ± 0.12 | +0.03 | -0.08 | -0.17 -0.09 +0.05 -0.23 +0.05 -0.16 |
| policy | plan:keep | -0.00 ± 0.04 | -0.03 ± 0.09 | -0.11 ± 0.14 | -0.17 ± 0.21 | +0.04 | -0.19 | +0.10 -0.12 -0.21 -0.19 -0.08 -0.51 |
| policy | plan:guard | -0.02 ± 0.05 | +0.00 ± 0.08 | -0.04 ± 0.11 | -0.08 ± 0.18 | +0.41 | -0.09 | -0.10 -0.10 -0.15 -0.16 +0.27 -0.21 |

Validation (§6.3): s0_full: pass (bias +0.09 ± 0.16, slope 0.87, contradictions 0, pairs 40); s1_full: pass (bias -0.11 ± 0.13, slope 1.01, contradictions 0, pairs 40); s2_full: pass (bias -0.02 ± 0.14, slope 0.95, contradictions 0, pairs 40); s3_full: pass (bias -0.02 ± 0.14, slope 1.07, contradictions 0, pairs 40); s4_full: fail -> re-run the family at 150 turns (bias -0.23 ± 0.15, slope 0.93, contradictions 0, pairs 40); s5_full: pass (bias +0.06 ± 0.16, slope 1.03, contradictions 0, pairs 40)

## B_frozen (natural; horizon 50, re-run at 150 after failed validation on s4_full; primary Q)

one-turn candidates:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | Q150 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|---|
| modal | 1t:shove | +0.00 ± 0.06 | -0.10 ± 0.09 | -0.08 ± 0.09 | -0.05 ± 0.17 | -0.07 | -0.04 | -0.20 +0.03 -0.25 -0.02 -0.11 +0.21 |
| modal | 1t:farm | -0.15 ± 0.09 | -0.24 ± 0.13 | -0.23 ± 0.17 | -0.23 ± 0.19 | -0.34 | -0.26 | -0.25 +0.00 -0.48 -0.34 -0.24 -0.05 |
| modal | 1t:step_back | -0.17 ± 0.10 | -0.22 ± 0.15 | -0.16 ± 0.20 | -0.18 ± 0.22 | -0.17 | -0.21 | -0.39 -0.23 +0.05 -0.43 -0.14 +0.05 |
| modal | 1t:recall | -0.06 ± 0.12 | -0.05 ± 0.14 | -0.03 ± 0.30 | +0.01 ± 0.37 | +0.06 | -0.02 | -0.32 +0.10 -0.25 -0.18 +0.05 +0.64 |
| policy | 1t:shove | +0.01 ± 0.06 | -0.06 ± 0.06 | -0.06 ± 0.13 | -0.08 ± 0.12 | -0.20 | -0.09 | +0.01 -0.20 -0.11 -0.05 -0.22 +0.07 |
| policy | 1t:farm | -0.14 ± 0.05 | -0.25 ± 0.03 | -0.18 ± 0.06 | -0.19 ± 0.15 | -0.16 | -0.20 | -0.38 -0.27 -0.21 +0.03 -0.24 -0.06 |
| policy | 1t:step_back | -0.18 ± 0.07 | -0.23 ± 0.12 | -0.21 ± 0.19 | -0.21 ± 0.27 | -0.11 | -0.19 | -0.47 -0.37 -0.11 -0.39 -0.13 +0.23 |
| policy | 1t:recall | -0.03 ± 0.07 | +0.02 ± 0.06 | +0.05 ± 0.17 | +0.08 ± 0.24 | +0.42 | +0.05 | -0.28 +0.02 -0.01 +0.12 +0.34 +0.28 |

sustained plans:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | Q150 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|---|
| modal | plan:shovex3 | +0.00 ± 0.06 | -0.32 ± 0.18 | -0.35 ± 0.15 | -0.39 ± 0.25 | -0.47 | -0.43 | -0.54 -0.15 -0.57 -0.53 -0.51 -0.02 |
| policy | plan:shovex3 | +0.01 ± 0.06 | -0.33 ± 0.06 | -0.27 ± 0.08 | -0.33 ± 0.19 | -0.43 | -0.35 | -0.56 -0.44 -0.10 -0.38 -0.37 -0.12 |

Validation (§6.3): s0_full: pass (bias -0.10 ± 0.22, slope 1.11, contradictions 0, pairs 34); s1_full: pass (bias -0.04 ± 0.25, slope 1.24, contradictions 0, pairs 34); s2_full: pass (bias -0.13 ± 0.15, slope 1.07, contradictions 0, pairs 34); s3_full: pass (bias -0.14 ± 0.16, slope 1.03, contradictions 0, pairs 36); s4_full: fail -> re-run the family at 150 turns (bias -0.16 ± 0.22, slope 1.34, contradictions 0, pairs 37); s5_full: pass (bias +0.03 ± 0.18, slope 1.06, contradictions 0, pairs 33)

## E_recall (natural; horizon 50, re-run at 150 after failed validation on s1_full; primary Q)

one-turn candidates:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | Q150 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|---|
| modal | 1t:recall | -0.25 ± 0.15 | -0.18 ± 0.22 | -0.13 ± 0.27 | -0.11 ± 0.24 | -0.29 | -0.12 | -0.25 -0.30 +0.25 +0.04 -0.06 -0.32 |
| policy | 1t:recall | -0.09 ± 0.08 | +0.05 ± 0.05 | +0.07 ± 0.07 | +0.11 ± 0.19 | -0.03 | +0.13 | +0.07 -0.03 +0.46 +0.07 +0.03 +0.05 |

sustained plans:

| protocol | candidate | Q1 | Q5 | Q20 | Q50 | Q150 | G50 | Q50 per policy |
|---|---|---|---|---|---|---|---|---|
| modal | plan:stayx3 | -0.39 ± 0.12 | -0.96 ± 0.12 | -1.03 ± 0.25 | -1.02 ± 0.32 | -1.30 | -1.03 | -0.70 -1.23 -1.02 -1.43 -0.63 -1.10 |
| modal | plan:step_backx3 | +0.04 ± 0.08 | -0.16 ± 0.12 | -0.12 ± 0.16 | -0.03 ± 0.28 | +0.08 | -0.04 | +0.25 +0.15 -0.43 -0.30 +0.05 +0.09 |
| policy | plan:stayx3 | -0.33 ± 0.08 | -0.85 ± 0.05 | -0.88 ± 0.19 | -0.96 ± 0.31 | -1.02 | -1.01 | -0.62 -1.12 -0.75 -1.43 -0.81 -1.06 |
| policy | plan:step_backx3 | -0.03 ± 0.04 | -0.31 ± 0.09 | -0.30 ± 0.09 | -0.29 ± 0.18 | -0.20 | -0.32 | -0.11 -0.33 -0.12 -0.53 -0.19 -0.43 |

Validation (§6.3): s0_full: pass (bias +0.03 ± 0.22, slope 0.89, contradictions 0, pairs 24); s1_full: fail -> re-run the family at 150 turns (bias -0.24 ± 0.21, slope 0.85, contradictions 0, pairs 24); s2_full: pass (bias -0.03 ± 0.19, slope 1.20, contradictions 0, pairs 24); s3_full: pass (bias -0.21 ± 0.23, slope 0.93, contradictions 0, pairs 24); s4_full: pass (bias +0.08 ± 0.23, slope 1.04, contradictions 0, pairs 24); s5_full: pass (bias -0.03 ± 0.26, slope 1.08, contradictions 0, pairs 24)

## F_freeze_c (constructed; horizon 60, re-run at 150 after failed validation on s3_full; primary G)

one-turn candidates:

| protocol | candidate | G1 | G5 | G20 | G50 | G150 | Q50 | G50 per policy |
|---|---|---|---|---|---|---|---|---|
| modal | 1t:keep | -0.44 ± 0.12 | -0.13 ± 0.06 | -0.09 ± 0.07 | -0.09 ± 0.08 | +0.01 | -0.08 | -0.07 -0.18 -0.10 -0.14 -0.10 +0.03 |
| modal | 1t:guard | -0.47 ± 0.12 | +0.17 ± 0.10 | +0.10 ± 0.16 | +0.17 ± 0.20 | +0.10 | +0.16 | +0.18 +0.23 -0.12 +0.02 +0.33 +0.38 |
| modal | 1t:break | +0.15 ± 0.12 | +0.02 ± 0.05 | +0.05 ± 0.07 | +0.11 ± 0.19 | -0.12 | +0.10 | +0.14 +0.20 -0.06 -0.10 +0.12 +0.39 |
| policy | 1t:keep | -0.45 ± 0.12 | -0.14 ± 0.08 | -0.05 ± 0.06 | -0.05 ± 0.11 | +0.01 | -0.02 | +0.03 +0.08 -0.14 -0.03 -0.21 -0.02 |
| policy | 1t:guard | -0.35 ± 0.19 | +0.10 ± 0.05 | +0.03 ± 0.07 | +0.17 ± 0.20 | +0.61 | +0.15 | +0.24 +0.22 -0.15 +0.39 +0.05 +0.25 |
| policy | 1t:break | +0.15 ± 0.13 | +0.05 ± 0.05 | +0.03 ± 0.10 | +0.08 ± 0.08 | +0.11 | +0.08 | +0.08 +0.17 -0.01 -0.01 +0.11 +0.16 |

sustained plans:

| protocol | candidate | G1 | G5 | G20 | G50 | G150 | Q50 | G50 per policy |
|---|---|---|---|---|---|---|---|---|
| modal | plan:keep | -0.44 ± 0.12 | -0.13 ± 0.06 | -0.19 ± 0.12 | -0.14 ± 0.15 | -0.06 | -0.10 | -0.13 -0.33 -0.10 -0.29 -0.06 +0.06 |
| modal | plan:guard | -0.42 ± 0.12 | -0.24 ± 0.09 | -0.17 ± 0.13 | -0.12 ± 0.26 | -0.37 | -0.07 | -0.07 -0.05 -0.35 -0.40 -0.12 +0.28 |
| policy | plan:keep | -0.45 ± 0.12 | -0.19 ± 0.07 | -0.15 ± 0.06 | -0.06 ± 0.19 | +0.35 | -0.05 | +0.01 +0.07 -0.30 +0.19 -0.21 -0.11 |
| policy | plan:guard | -0.36 ± 0.15 | -0.27 ± 0.05 | -0.06 ± 0.17 | -0.03 ± 0.25 | +0.53 | +0.00 | +0.12 -0.04 -0.27 +0.20 -0.35 +0.16 |

Validation (§6.3): s0_full: pass (bias -0.06 ± 0.09, slope 1.19, contradictions 0, pairs 50); s1_full: pass (bias +0.00 ± 0.09, slope 0.91, contradictions 0, pairs 50); s2_full: pass (bias -0.12 ± 0.12, slope 1.10, contradictions 0, pairs 50); s3_full: fail -> re-run the family at 150 turns (bias -0.19 ± 0.11, slope 0.87, contradictions 0, pairs 50); s4_full: pass (bias -0.05 ± 0.08, slope 1.03, contradictions 0, pairs 50); s5_full: pass (bias -0.01 ± 0.10, slope 0.97, contradictions 0, pairs 50)
