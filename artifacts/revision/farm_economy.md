# Experiment B: farming economy (K = 32)

farm_per_hp is the value of one HP of minions; a fresh wave has 4 HP, so the default 0.25 means one resource per wave. Crash rewards (1/2/3) and the gank loss (5) are unchanged.

## Equilibrium structure

| farm/HP | nodes | mixed | P(Contest) avg node | P(Freeze) where legal | eq path actions (A) | eq path P(Contest) | eq crashes A | vision S/M/L |
|---|---|---|---|---|---|---|---|---|
| 0 | 14849 | 435 | 0.063 | 0.056 | SP-S 1.00 | 0 | 0 | 0.04/0.00/0.10 |
| 0.125 | 14849 | 1001 | 0.253 | 0.064 | SH-C 0.03, SP-S 0.97 | 0.031 | 0 | 0.05/0.00/0.18 |
| 0.25 | 14849 | 1030 | 0.568 | 0.069 | SH-C 0.06, SP-C 0.84, SP-S 0.09 | 0.906 | 0 | 0.10/0.00/0.30 |
| 0.5 | 14849 | 762 | 0.823 | 0.065 | SH-C 0.12, SP-C 0.84, SP-S 0.03 | 0.969 | 0 | 0.14/0.00/0.33 |
| 1 | 14849 | 811 | 0.918 | 0.062 | SH-C 0.12, SP-C 0.84, SP-S 0.03 | 0.969 | 0 | 0.29/0.00/0.41 |

## Equilibrium path economy (per player, whole game)

| farm/HP | farm available | farm collected | collection rate | gank loss |
|---|---|---|---|---|
| 0 | 0 | 0 | - | 27.88 |
| 0.125 | 7.88 | 3.27 | 0.415 | 28.25 |
| 0.25 | 15.5 | 9.65 | 0.623 | 34.7 |
| 0.5 | 31 | 19.53 | 0.63 | 35.45 |
| 1 | 62 | 39.06 | 0.63 | 35.45 |

## Exploitability of baselines (as A) [farm / crash / gank decomposition of the value vs. best response]

| farm/HP | uniform | aggressive | freeze_contest | passive_safe | stack_builder | release_safe | contest_slowpush |
|---|---|---|---|---|---|---|---|
| 0 | 7.47 [0.0/0.0/-7.5] | 22.75 [0.0/0.0/-22.8] | 9.55 [0.0/-8.0/-1.5] | 7.83 [0.0/-8.0/0.2] | 10.70 [0.0/-5.0/-5.7] | 0.00 [0.0/0.0/-0.0] | 11.52 [0.0/-8.0/-3.5] |
| 0.125 | 7.58 [-1.1/0.1/-6.6] | 23.49 [-0.7/0.0/-22.8] | 8.09 [-0.0/-8.0/-0.1] | 7.52 [-0.8/-8.0/1.3] | 7.26 [1.9/-6.0/-3.2] | 1.93 [-5.5/0.0/3.5] | 8.34 [2.3/-8.0/-2.6] |
| 0.25 | 10.08 [-6.1/0.1/-4.1] | 24.86 [-5.2/0.0/-19.6] | 9.45 [-4.5/-5.0/0.1] | 9.42 [-5.4/-5.0/0.9] | 5.78 [2.5/-6.0/-2.2] | 7.69 [-11.7/0.0/4.0] | 6.59 [2.4/-7.0/-1.9] |
| 0.5 | 16.74 [-13.6/0.1/-3.2] | 30.32 [-10.9/0.0/-19.4] | 16.20 [-20.7/0.0/4.6] | 17.03 [-15.6/-5.0/3.6] | 5.41 [-2.5/-3.0/0.1] | 19.52 [-23.8/0.0/4.3] | 6.37 [-1.4/-5.0/-0.0] |
| 1 | 30.56 [-27.8/0.1/-2.9] | 41.24 [-21.8/0.0/-19.4] | 37.02 [-41.6/0.0/4.6] | 37.02 [-41.6/0.0/4.6] | 8.25 [-6.0/-3.0/0.7] | 43.35 [-47.7/0.0/4.3] | 8.29 [-4.0/-5.0/0.7] |

## Probe situations (k = 12 of 32): value to A, A's equilibrium strategy, best-Contest minus best-Safe

**B wave at A's freeze line, fighting A's wave**

| farm/HP | V* | A plays | B plays | Contest - Safe |
|---|---|---|---|---|
| 0 | 0.25 | SP-S 1.00 | SP-S 1.00 | -0.1 |
| 0.125 | 0.55 | SP-S 1.00 | SP-S 1.00 | -0.025 |
| 0.25 | 0.95 | SP-C 1.00 | SP-S 1.00 | 0.05 |
| 0.5 | 1.31 | SP-C 1.00 | SP-C 1.00 | 0.26 |
| 1 | 1.87 | SP-C 1.00 | SP-C 1.00 | 0.62 |

**A 2-wave stack at +1, B wave just spawned**

| farm/HP | V* | A plays | B plays | Contest - Safe |
|---|---|---|---|---|
| 0 | 0.2 | SH-C 1.00 | SH-S 1.00 | 0.375 |
| 0.125 | -1.774 | SH-C 1.00 | SH-S 1.00 | 0.715 |
| 0.25 | -2.973 | SH-C 1.00 | SP-C 1.00 | 0.025 |
| 0.5 | -4.698 | SH-C 1.00 | SP-C 1.00 | 0.395 |
| 1 | -10.09 | SH-C 1.00 | F-C 1.00 | 0.938 |

**A's 3-wave crash under B's tower**

| farm/HP | V* | A plays | B plays | Contest - Safe |
|---|---|---|---|---|
| 0 | -3.225 | SP-S 1.00 | SP-S 1.00 | -0.475 |
| 0.125 | -4.447 | SP-S 1.00 | SP-S 1.00 | -0.445 |
| 0.25 | -5.785 | SP-S 1.00 | SP-S 1.00 | -0.415 |
| 0.5 | -8.371 | SP-S 1.00 | SP-C 1.00 | -0.34 |
| 1 | -13.38 | SP-S 1.00 | SP-C 1.00 | -0.205 |

**B's 3-wave crash under A's tower (A defends)**

| farm/HP | V* | A plays | B plays | Contest - Safe |
|---|---|---|---|---|
| 0 | 3.225 | SP-S 1.00 | SP-S 1.00 | -0.1 |
| 0.125 | 4.447 | SP-S 1.00 | SP-S 1.00 | -0.075 |
| 0.25 | 5.785 | SP-S 1.00 | SP-S 1.00 | -0.05 |
| 0.5 | 8.371 | SP-C 1.00 | SP-S 1.00 | 0 |
| 1 | 13.38 | SP-C 1.00 | SP-S 1.00 | 0.1 |
