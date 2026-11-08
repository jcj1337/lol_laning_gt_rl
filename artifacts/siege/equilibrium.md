# Experiment D: equilibrium and exploitation (siege model, K = 32)

V*(s0) = 0; 172907 decision nodes, 8042 mixed, max certified gap 4.4e-15; enumerate 109 s, solve 44 s.

Equilibrium at s0 (A): {'SP-C': 0.0, 'SH-C': 0.0, 'SP-S': 1.0, 'SH-S': 0.0}

Equilibrium path per player: actions SH-C 0.06, SP-C 0.84, SP-S 0.09; farm 9.65 of 15.50; plates 0.00; gank loss 34.70; crashes [0.0, 0.0, 0.0]; P(Contest) 0.91; siege turns 0.00.

Siege nodes (A, all reachable): P(Pressure) 0.682, P(Vision) 0.310, P(disengage) 0.008 over 55160 nodes.

## Baselines

- siege tendency in brackets: [vision] = Establish Vision when sieging, [safe] = disengage; none = Apply Pressure.

## Head-to-head (exact expected payoff to the row player A)

| A \ B | equilibrium | uniform | aggressive | freeze_contest | passive_safe | stack_builder | release_safe | contest_slowpush | aggressive[vision] | stack_builder[vision] | contest_slowpush[vision] | stack_builder[safe] | aggressive[safe] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| equilibrium | 0 | 9.6 | 24.47 | 2.87 | 2.87 | 0.32 | 2.8 | 0.32 | 24.47 | 0.32 | 0.32 | 0.32 | 24.47 |
| uniform | -9.6 | 0 | 22.35 | -4.63 | -4.96 | -7.66 | -0.84 | -7.66 | 22.35 | -7.66 | -7.66 | -7.66 | 22.35 |
| aggressive | -24.47 | -22.35 | 0 | -24.49 | -24.22 | -24.76 | -14.39 | -24.76 | 0 | -24.76 | -24.76 | -24.76 | 0 |
| freeze_contest | -2.87 | 4.63 | 24.49 | 0 | 0 | -2.38 | 0 | -2.38 | 24.49 | -2.38 | -2.38 | -2.38 | 24.49 |
| passive_safe | -2.87 | 4.96 | 24.22 | 0 | 0 | -2.38 | 0 | -2.38 | 24.22 | -2.38 | -2.38 | -2.38 | 24.22 |
| stack_builder | -0.32 | 7.66 | 24.76 | 2.38 | 2.38 | 0 | 2.38 | 0 | 24.76 | 0 | 0 | 0 | 24.76 |
| release_safe | -2.8 | 0.84 | 14.39 | 0 | 0 | -2.38 | 0 | -2.38 | 14.39 | -2.38 | -2.38 | -2.38 | 14.39 |
| contest_slowpush | -0.32 | 7.66 | 24.76 | 2.38 | 2.38 | 0 | 2.38 | 0 | 24.76 | 0 | 0 | 0 | 24.76 |
| aggressive[vision] | -24.47 | -22.35 | 0 | -24.49 | -24.22 | -24.76 | -14.39 | -24.76 | 0 | -24.76 | -24.76 | -24.76 | 0 |
| stack_builder[vision] | -0.32 | 7.66 | 24.76 | 2.38 | 2.38 | 0 | 2.38 | 0 | 24.76 | 0 | 0 | 0 | 24.76 |
| contest_slowpush[vision] | -0.32 | 7.66 | 24.76 | 2.38 | 2.38 | 0 | 2.38 | 0 | 24.76 | 0 | 0 | 0 | 24.76 |
| stack_builder[safe] | -0.32 | 7.66 | 24.76 | 2.38 | 2.38 | 0 | 2.38 | 0 | 24.76 | 0 | 0 | 0 | 24.76 |
| aggressive[safe] | -24.47 | -22.35 | 0 | -24.49 | -24.22 | -24.76 | -14.39 | -24.76 | 0 | -24.76 | -24.76 | -24.76 | 0 |

## Exploitability and the best responder's gain [farm / plates / gank parts of the baseline's value]

| baseline (as A) | exploitability | per turn | farm | plates | gank | vs equilibrium B | captured | B's best response plays |
|---|---|---|---|---|---|---|---|---|
| uniform | 10.04 | 0.314 | -6.06 | 0.03 | -4.01 | -9.6 | 96% | F-C 0.05, F-S 0.08, SH-C 0.08, SH-S 0.03, SP-C 0.59, SP-S 0.16 |
| aggressive | 24.86 | 0.777 | -5.24 | 0 | -19.62 | -24.47 | 98% | SP-C 0.94, SP-S 0.06 |
| freeze_contest | 18.52 | 0.579 | -4.36 | -5 | -9.16 | -2.87 | 15% | SH-C 0.19, SH-C+P 0.06, SP-C 0.28, SP-C+P 0.09, SP-C+V 0.12, SP-S 0.25 |
| passive_safe | 21.47 | 0.671 | -6.23 | -4 | -11.24 | -2.87 | 13% | SH-C 0.28, SP-C 0.31, SP-C+P 0.12, SP-C+V 0.12, SP-S 0.16 |
| stack_builder | 18.74 | 0.586 | 0.25 | -4 | -14.99 | -0.32 | 2% | SH-C 0.28, SP-C 0.31, SP-C+P 0.12, SP-C+V 0.12, SP-S 0.16 |
| release_safe | 7.69 | 0.24 | -11.69 | 0 | 4 | -2.8 | 36% | SH-C 0.09, SP-C 0.78, SP-S 0.12 |
| contest_slowpush | 19.03 | 0.595 | 1.4 | -7 | -13.43 | -0.32 | 2% | SH-C 0.31, SH-S 0.03, SP-C 0.16, SP-C+P 0.22, SP-C+V 0.12, SP-S 0.16 |
| aggressive[vision] | 24.86 | 0.777 | -5.24 | 0 | -19.62 | -24.47 | 98% | SP-C 0.94, SP-S 0.06 |
| stack_builder[vision] | 18.74 | 0.586 | 0.25 | -4 | -14.99 | -0.32 | 2% | SH-C 0.28, SP-C 0.31, SP-C+P 0.12, SP-C+V 0.12, SP-S 0.16 |
| contest_slowpush[vision] | 19.03 | 0.595 | 1.4 | -7 | -13.43 | -0.32 | 2% | SH-C 0.31, SH-S 0.03, SP-C 0.16, SP-C+P 0.22, SP-C+V 0.12, SP-S 0.16 |
| stack_builder[safe] | 18.74 | 0.586 | 0.25 | -4 | -14.99 | -0.32 | 2% | SH-C 0.28, SP-C 0.31, SP-C+P 0.12, SP-C+V 0.12, SP-S 0.16 |
| aggressive[safe] | 24.86 | 0.777 | -5.24 | 0 | -19.62 | -24.47 | 98% | SP-C 0.94, SP-S 0.06 |

## Siege behaviour along the best-response paths (per game)

| baseline (A) | A crashes S/M/L | BR crashes S/M/L | BR siege turns | BR pressure | BR vision | BR disengage | BR plates | BR thinned HP | A bounces (HP) | BR protected turns | BR P(C|prot) | BR P(C|unprot) | A pressure | A vision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 0.1/0.0/0.0 | 0.0/0.0/0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 (0.0) | 0 | - | 0.73 | 0.03 | 0.03 |
| aggressive | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 (0.0) | 0 | - | 0.94 | 0 | 0 |
| freeze_contest | 0.0/0.0/0.0 | 1.0/1.0/1.0 | 10 | 5 | 4 | 1 | 5 | 4 | 0.00 (0.0) | 19 | 0.84 | 0.62 | 0 | 0 |
| passive_safe | 0.0/0.0/0.0 | 3.0/1.0/0.0 | 8 | 4 | 4 | 0 | 4 | 4 | 0.00 (0.0) | 24 | 0.92 | 0.62 | 0 | 0 |
| stack_builder | 0.0/0.0/0.0 | 3.0/1.0/0.0 | 8 | 4 | 4 | 0 | 4 | 4 | 0.00 (0.0) | 24 | 0.92 | 0.62 | 0 | 0 |
| release_safe | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 (0.0) | 0 | - | 0.88 | 0 | 0 |
| contest_slowpush | 0.0/0.0/0.0 | 2.0/2.0/1.0 | 12 | 7 | 4 | 1 | 7 | 7 | 0.00 (0.0) | 24 | 0.88 | 0.62 | 0 | 0 |
| aggressive[vision] | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 (0.0) | 0 | - | 0.94 | 0 | 0 |
| stack_builder[vision] | 0.0/0.0/0.0 | 3.0/1.0/0.0 | 8 | 4 | 4 | 0 | 4 | 4 | 0.00 (0.0) | 24 | 0.92 | 0.62 | 0 | 0 |
| contest_slowpush[vision] | 0.0/0.0/0.0 | 2.0/2.0/1.0 | 12 | 7 | 4 | 1 | 7 | 7 | 0.00 (0.0) | 24 | 0.88 | 0.62 | 0 | 0 |
| stack_builder[safe] | 0.0/0.0/0.0 | 3.0/1.0/0.0 | 8 | 4 | 4 | 0 | 4 | 4 | 0.00 (0.0) | 24 | 0.92 | 0.62 | 0 | 0 |
| aggressive[safe] | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 (0.0) | 0 | - | 0.94 | 0 | 0 |

## Realized payoff distributions (4000 sampled games each)

| matchup | mean | stderr | std | q05 | q50 | q95 | ganks A/B | plates A/B |
|---|---|---|---|---|---|---|---|---|
| equilibrium vs equilibrium | 0.224 | 0.262 | 16.58 | -25 | 0 | 25 | [6.94, 6.98] | [0.0, 0.0] |
| uniform vs best response | -9.489 | 0.252 | 15.94 | -35.66 | -10.35 | 17.31 | [6.68, 6.0] | [0.03, 0.0] |
| aggressive vs best response | -24.37 | 0.263 | 16.61 | -50.24 | -25.24 | 4.76 | [9.35, 5.52] | [0.0, 0.0] |
| freeze_contest vs best response | -18.3 | 0.222 | 14.02 | -39.36 | -19.36 | 5.64 | [5.65, 3.86] | [0.0, 5.0] |
| passive_safe vs best response | -21.11 | 0.214 | 13.54 | -45.23 | -20.23 | -0.23 | [5.46, 3.28] | [0.0, 4.0] |
| stack_builder vs best response | -18.39 | 0.22 | 13.92 | -38.75 | -18.75 | 6.25 | [6.21, 3.29] | [0.0, 4.0] |
| release_safe vs best response | -7.209 | 0.245 | 15.51 | -31.69 | -6.69 | 18.31 | [5.51, 6.4] | [0.0, 0.0] |
| contest_slowpush vs best response | -18.67 | 0.217 | 13.71 | -40.6 | -20.6 | 4.4 | [5.95, 3.34] | [0.0, 7.0] |
| aggressive[vision] vs best response | -24.37 | 0.263 | 16.61 | -50.24 | -25.24 | 4.76 | [9.35, 5.52] | [0.0, 0.0] |
| stack_builder[vision] vs best response | -18.39 | 0.22 | 13.92 | -38.75 | -18.75 | 6.25 | [6.21, 3.29] | [0.0, 4.0] |
| contest_slowpush[vision] vs best response | -18.67 | 0.217 | 13.71 | -40.6 | -20.6 | 4.4 | [5.95, 3.34] | [0.0, 7.0] |
| stack_builder[safe] vs best response | -18.39 | 0.22 | 13.92 | -38.75 | -18.75 | 6.25 | [6.21, 3.29] | [0.0, 4.0] |
| aggressive[safe] vs best response | -24.37 | 0.263 | 16.61 | -50.24 | -25.24 | 4.76 | [9.35, 5.52] | [0.0, 0.0] |
