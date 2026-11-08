# Controlled comparison: siege decisions x tower clearing speed (K = 32)

| configuration | description |
|---|---|
| choice, tower 1 (previous revision) | original siege rewards, 1 HP/tick |
| choice, tower 2 | original siege rewards, 2 HP/tick |
| choice, tower 3 | original siege rewards, 3 HP/tick |
| objective, tower 1 | Pressure/Vision, 1 HP/tick |
| objective, tower 2 | Pressure/Vision, 2 HP/tick |
| objective, tower 3 | Pressure/Vision, 3 HP/tick |

## State space and solver

| configuration | nodes | distinct | max layer | transitions | enumerate s | solve s | LPs | mixed | max gap |
|---|---|---|---|---|---|---|---|---|---|
| choice, tower 1 (previous revision) | 14849 | 2480 | 1302 | 91744 | 7.7 | 3 | 1044 | 1030 | 1.8e-15 |
| choice, tower 2 | 7275 | 1002 | 548 | 35952 | 2.3 | 1.5 | 546 | 542 | 8.9e-16 |
| choice, tower 3 | 5047 | 630 | 358 | 22688 | 1.4 | 1.3 | 416 | 412 | 8.9e-16 |
| objective, tower 1 | 172907 | 60372 | 31860 | 2697760 | 150.2 | 71.9 | 8130 | 8042 | 4.4e-15 |
| objective, tower 2 | 30501 | 5550 | 2912 | 219304 | 12.4 | 5.6 | 2088 | 2060 | 4.4e-15 |
| objective, tower 3 | 16639 | 2488 | 1232 | 89184 | 5 | 2.8 | 1200 | 1186 | 8.0e-15 |

## Symmetric equilibrium path (per player, whole game)

| configuration | V* | actions (A) | crashes S/M/L | farm | plates | gank loss | P(Contest) | siege turns | pressure / vision / disengage turns |
|---|---|---|---|---|---|---|---|---|---|
| choice, tower 1 (previous revision) | 0 | SH-C 0.06, SP-C 0.84, SP-S 0.09 | 0.00/0.00/0.00 | 9.65 | 0 | 34.7 | 0.91 | 0 | 0.00 / 0.00 / 0.00 |
| choice, tower 2 | 0 | SH-C 0.06, SP-C 0.84, SP-S 0.09 | 0.00/0.00/0.00 | 9.65 | 0 | 34.7 | 0.91 | 0 | 0.00 / 0.00 / 0.00 |
| choice, tower 3 | 0 | SH-C 0.06, SP-C 0.84, SP-S 0.09 | 0.00/0.00/0.00 | 9.65 | 0 | 34.7 | 0.91 | 0 | 0.00 / 0.00 / 0.00 |
| objective, tower 1 | 0 | SH-C 0.06, SP-C 0.84, SP-S 0.09 | 0.00/0.00/0.00 | 9.65 | 0 | 34.7 | 0.91 | 0 | 0.00 / 0.00 / 0.00 |
| objective, tower 2 | 0 | SH-C 0.06, SP-C 0.84, SP-S 0.09 | 0.00/0.00/0.00 | 9.65 | 0 | 34.7 | 0.91 | 0 | 0.00 / 0.00 / 0.00 |
| objective, tower 3 | 0 | SH-C 0.06, SP-C 0.84, SP-S 0.09 | 0.00/0.00/0.00 | 9.65 | 0 | 34.7 | 0.91 | 0 | 0.00 / 0.00 / 0.00 |

## Equilibrium siege decisions (A, averaged over reachable siege nodes)

| configuration | siege nodes | P(Apply Pressure) | P(Establish Vision) | P(disengage) | P(Contest) all nodes |
|---|---|---|---|---|---|
| choice, tower 1 (previous revision) | 0 | 0 | 0 | 0 | 0.568 |
| choice, tower 2 | 0 | 0 | 0 | 0 | 0.583 |
| choice, tower 3 | 0 | 0 | 0 | 0 | 0.522 |
| objective, tower 1 | 55160 | 0.682 | 0.31 | 0.008 | 0.798 |
| objective, tower 2 | 2635 | 0.476 | 0.524 | 0 | 0.786 |
| objective, tower 3 | 853 | 0.274 | 0.726 | 0 | 0.745 |

## Value of a 12-HP crash state to the crasher (A's mass under B's tower, B's wave waiting behind it, 20 turns left; siege model: 3 plate opportunities pending)

| configuration | V (no vision) | A plays | V (with vision) | A plays (vision) |
|---|---|---|---|---|
| choice, tower 1 (previous revision) | -5.785 | {'SP-S': 1.0} | -4.91 | {'SP-S': 1.0} |
| choice, tower 2 | -3.742 | {'SP-S': 1.0} | -2.867 | {'SP-S': 1.0} |
| choice, tower 3 | -3.297 | {'SP-S': 1.0} | -2.422 | {'SP-S': 1.0} |
| objective, tower 1 | 6.051 | {'SP-C+V': 1.0} | 7.912 | {'SP-C+P': 1.0} |
| objective, tower 2 | 2.523 | {'SP-C+V': 1.0} | 4.247 | {'SP-C+P': 1.0} |
| objective, tower 3 | 2.589 | {'SP-C+V': 1.0} | 3.601 | {'SP-C+V': 1.0} |

## Exploitability of baselines (as A) [farm / plates / crash / gank parts of the value against the best response]

**choice, tower 1 (previous revision)**

| baseline | exploitability | vs equilibrium | farm | plates | crash | gank |
|---|---|---|---|---|---|---|
| uniform | 10.08 | -9.86 | -6.08 | 0 | 0.06 | -4.06 |
| aggressive | 24.86 | -24.47 | -5.24 | 0 | 0 | -19.62 |
| freeze_contest | 9.45 | -2.87 | -4.55 | 0 | -5 | 0.1 |
| passive_safe | 9.42 | -2.87 | -5.37 | 0 | -5 | 0.95 |
| stack_builder | 5.78 | -0.32 | 2.47 | 0 | -6 | -2.25 |
| release_safe | 7.69 | -2.8 | -11.69 | 0 | 0 | 4 |
| contest_slowpush | 6.59 | -0.32 | 2.36 | 0 | -7 | -1.95 |

**choice, tower 2**

| baseline | exploitability | vs equilibrium | farm | plates | crash | gank |
|---|---|---|---|---|---|---|
| uniform | 10.08 | -9.87 | -6.08 | 0 | 0.06 | -4.06 |
| aggressive | 24.86 | -24.47 | -5.24 | 0 | 0 | -19.62 |
| freeze_contest | 10.61 | -2.87 | -4.84 | 0 | -6 | 0.23 |
| passive_safe | 10.92 | -2.87 | -5.47 | 0 | -6 | 0.55 |
| stack_builder | 7.18 | -0.32 | 2.47 | 0 | -7 | -2.65 |
| release_safe | 7.69 | -2.8 | -11.69 | 0 | 0 | 4 |
| contest_slowpush | 7.95 | -0.32 | 2.15 | 0 | -7 | -3.1 |

**choice, tower 3**

| baseline | exploitability | vs equilibrium | farm | plates | crash | gank |
|---|---|---|---|---|---|---|
| uniform | 10.08 | -9.87 | -6.08 | 0 | 0.06 | -4.06 |
| aggressive | 24.86 | -24.47 | -5.24 | 0 | 0 | -19.62 |
| freeze_contest | 10.53 | -2.87 | -4.63 | 0 | -6 | 0.1 |
| passive_safe | 10.84 | -2.87 | -5.26 | 0 | -6 | 0.42 |
| stack_builder | 7.32 | -0.32 | 2.66 | 0 | -7 | -2.97 |
| release_safe | 7.69 | -2.8 | -11.69 | 0 | 0 | 4 |
| contest_slowpush | 7.88 | -0.32 | 2.34 | 0 | -7 | -3.23 |

**objective, tower 1**

| baseline | exploitability | vs equilibrium | farm | plates | crash | gank |
|---|---|---|---|---|---|---|
| uniform | 10.04 | -9.6 | -6.06 | 0.03 | 0 | -4.01 |
| aggressive | 24.86 | -24.47 | -5.24 | 0 | 0 | -19.62 |
| freeze_contest | 18.52 | -2.87 | -4.36 | -5 | 0 | -9.16 |
| passive_safe | 21.47 | -2.87 | -6.23 | -4 | 0 | -11.24 |
| stack_builder | 18.74 | -0.32 | 0.25 | -4 | 0 | -14.99 |
| release_safe | 7.69 | -2.8 | -11.69 | 0 | 0 | 4 |
| contest_slowpush | 19.03 | -0.32 | 1.4 | -7 | 0 | -13.43 |
| aggressive[vision] | 24.86 | -24.47 | -5.24 | 0 | 0 | -19.62 |
| stack_builder[vision] | 18.74 | -0.32 | 0.25 | -4 | 0 | -14.99 |
| contest_slowpush[vision] | 19.03 | -0.32 | 1.4 | -7 | 0 | -13.43 |
| stack_builder[safe] | 18.74 | -0.32 | 0.25 | -4 | 0 | -14.99 |
| aggressive[safe] | 24.86 | -24.47 | -5.24 | 0 | 0 | -19.62 |

**objective, tower 2**

| baseline | exploitability | vs equilibrium | farm | plates | crash | gank |
|---|---|---|---|---|---|---|
| uniform | 10.05 | -9.6 | -6.06 | 0.02 | 0 | -4.01 |
| aggressive | 24.86 | -24.47 | -5.24 | 0 | 0 | -19.62 |
| freeze_contest | 15.9 | -2.87 | -4.43 | 0 | 0 | -11.46 |
| passive_safe | 17.63 | -2.87 | -6.37 | 0 | 0 | -11.26 |
| stack_builder | 14.43 | -0.32 | 0.73 | 0 | 0 | -15.16 |
| release_safe | 7.69 | -2.8 | -11.69 | 0 | 0 | 4 |
| contest_slowpush | 14.43 | -0.32 | 0.73 | 0 | 0 | -15.16 |
| aggressive[vision] | 24.86 | -24.47 | -5.24 | 0 | 0 | -19.62 |
| stack_builder[vision] | 14.43 | -0.32 | 0.73 | 0 | 0 | -15.16 |
| contest_slowpush[vision] | 14.43 | -0.32 | 0.73 | 0 | 0 | -15.16 |
| stack_builder[safe] | 14.43 | -0.32 | 0.73 | 0 | 0 | -15.16 |
| aggressive[safe] | 24.86 | -24.47 | -5.24 | 0 | 0 | -19.62 |

**objective, tower 3**

| baseline | exploitability | vs equilibrium | farm | plates | crash | gank |
|---|---|---|---|---|---|---|
| uniform | 10.05 | -9.6 | -6.06 | 0.02 | 0 | -4.01 |
| aggressive | 24.86 | -24.47 | -5.24 | 0 | 0 | -19.62 |
| freeze_contest | 15.9 | -2.87 | -4.43 | 0 | 0 | -11.46 |
| passive_safe | 17.63 | -2.87 | -6.37 | 0 | 0 | -11.26 |
| stack_builder | 14.43 | -0.32 | 0.73 | 0 | 0 | -15.16 |
| release_safe | 7.69 | -2.8 | -11.69 | 0 | 0 | 4 |
| contest_slowpush | 14.43 | -0.32 | 0.73 | 0 | 0 | -15.16 |
| aggressive[vision] | 24.86 | -24.47 | -5.24 | 0 | 0 | -19.62 |
| stack_builder[vision] | 14.43 | -0.32 | 0.73 | 0 | 0 | -15.16 |
| contest_slowpush[vision] | 14.43 | -0.32 | 0.73 | 0 | 0 | -15.16 |
| stack_builder[safe] | 14.43 | -0.32 | 0.73 | 0 | 0 | -15.16 |
| aggressive[safe] | 24.86 | -24.47 | -5.24 | 0 | 0 | -19.62 |

## Sieges along best-response paths against crash-prone baselines (per game; 'BR' = the exploiting player B)

crashes S/M/L and siege turns of the best responder; bounces = stacked waves released by the baseline after the best responder's sieges clear; plates and thinning earned by the best responder.

**choice, tower 1 (previous revision)**

| baseline (A) | BR crashes S/M/L | BR crash HP | BR siege turns | BR pressure | BR vision | BR disengage | BR plates | BR thinned HP | BR crash resources | baseline bounces | bounce HP | BR protected turns | BR actions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| freeze_contest | 0.00/1.00/1.00 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 8 | 0 | SH-C 0.06, SH-S 0.06, SP-C 0.38, SP-S 0.50 |
| passive_safe | 0.00/1.00/1.00 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 8 | 0 | SH-C 0.06, SH-S 0.06, SP-C 0.41, SP-S 0.47 |
| stack_builder | 1.00/1.00/1.00 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 1 | 8 | 0 | SH-C 0.12, SH-S 0.06, SP-C 0.31, SP-S 0.50 |
| contest_slowpush | 0.00/2.00/1.00 | 23 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 1 | 8 | 0 | SH-C 0.19, SP-C 0.28, SP-S 0.53 |

**choice, tower 2**

| baseline (A) | BR crashes S/M/L | BR crash HP | BR siege turns | BR pressure | BR vision | BR disengage | BR plates | BR thinned HP | BR crash resources | baseline bounces | bounce HP | BR protected turns | BR actions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| freeze_contest | 0.00/0.00/2.00 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 1 | 8 | 0 | SH-C 0.09, SH-S 0.03, SP-C 0.38, SP-S 0.50 |
| passive_safe | 0.00/0.00/2.00 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 1 | 8 | 0 | SH-C 0.06, SH-S 0.06, SP-C 0.41, SP-S 0.47 |
| stack_builder | 1.00/0.00/2.00 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 1 | 8 | 0 | SH-C 0.12, SH-S 0.06, SP-C 0.31, SP-S 0.50 |
| contest_slowpush | 1.00/0.00/2.00 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 1 | 8 | 0 | SH-C 0.09, SH-S 0.06, SP-C 0.34, SP-S 0.50 |

**choice, tower 3**

| baseline (A) | BR crashes S/M/L | BR crash HP | BR siege turns | BR pressure | BR vision | BR disengage | BR plates | BR thinned HP | BR crash resources | baseline bounces | bounce HP | BR protected turns | BR actions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| freeze_contest | 0.00/0.00/2.00 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | SH-C 0.09, SH-S 0.03, SP-C 0.38, SP-S 0.50 |
| passive_safe | 0.00/0.00/2.00 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | SH-C 0.06, SH-S 0.06, SP-C 0.41, SP-S 0.47 |
| stack_builder | 1.00/0.00/2.00 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | SH-C 0.09, SH-S 0.06, SP-C 0.34, SP-S 0.50 |
| contest_slowpush | 1.00/0.00/2.00 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | SH-C 0.09, SH-S 0.06, SP-C 0.34, SP-S 0.50 |

**objective, tower 1**

| baseline (A) | BR crashes S/M/L | BR crash HP | BR siege turns | BR pressure | BR vision | BR disengage | BR plates | BR thinned HP | BR crash resources | baseline bounces | bounce HP | BR protected turns | BR actions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| freeze_contest | 1.00/1.00/1.00 | 19 | 10 | 5 | 4 | 1 | 5 | 4 | 0 | 0 | 0 | 19 | SH-C 0.19, SH-C+P 0.06, SP-C 0.28, SP-C+P 0.09, SP-C+V 0.12, SP-S 0.25 |
| passive_safe | 3.00/1.00/0.00 | 14 | 8 | 4 | 4 | 0 | 4 | 4 | 0 | 0 | 0 | 24 | SH-C 0.28, SP-C 0.31, SP-C+P 0.12, SP-C+V 0.12, SP-S 0.16 |
| stack_builder | 3.00/1.00/0.00 | 14 | 8 | 4 | 4 | 0 | 4 | 4 | 0 | 0 | 0 | 24 | SH-C 0.28, SP-C 0.31, SP-C+P 0.12, SP-C+V 0.12, SP-S 0.16 |
| contest_slowpush | 2.00/2.00/1.00 | 27 | 12 | 7 | 4 | 1 | 7 | 7 | 0 | 0 | 0 | 24 | SH-C 0.31, SH-S 0.03, SP-C 0.16, SP-C+P 0.22, SP-C+V 0.12, SP-S 0.16 |

**objective, tower 2**

| baseline (A) | BR crashes S/M/L | BR crash HP | BR siege turns | BR pressure | BR vision | BR disengage | BR plates | BR thinned HP | BR crash resources | baseline bounces | bounce HP | BR protected turns | BR actions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| freeze_contest | 4.00/0.00/0.00 | 11 | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 24 | SH-C 0.34, SP-C 0.31, SP-C+V 0.12, SP-S 0.22 |
| passive_safe | 4.00/0.00/0.00 | 4 | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 23 | SH-C 0.19, SP-C 0.56, SP-C+V 0.12, SP-S 0.12 |
| stack_builder | 4.00/0.00/0.00 | 10 | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 24 | SH-C 0.31, SP-C 0.31, SP-C+V 0.12, SP-S 0.25 |
| contest_slowpush | 4.00/0.00/0.00 | 10 | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 24 | SH-C 0.31, SP-C 0.31, SP-C+V 0.12, SP-S 0.25 |

**objective, tower 3**

| baseline (A) | BR crashes S/M/L | BR crash HP | BR siege turns | BR pressure | BR vision | BR disengage | BR plates | BR thinned HP | BR crash resources | baseline bounces | bounce HP | BR protected turns | BR actions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| freeze_contest | 4.00/0.00/0.00 | 11 | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 24 | SH-C 0.34, SP-C 0.31, SP-C+V 0.12, SP-S 0.22 |
| passive_safe | 4.00/0.00/0.00 | 4 | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 23 | SH-C 0.19, SP-C 0.56, SP-C+V 0.12, SP-S 0.12 |
| stack_builder | 4.00/0.00/0.00 | 10 | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 24 | SH-C 0.31, SP-C 0.31, SP-C+V 0.12, SP-S 0.25 |
| contest_slowpush | 4.00/0.00/0.00 | 10 | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 24 | SH-C 0.31, SP-C 0.31, SP-C+V 0.12, SP-S 0.25 |
