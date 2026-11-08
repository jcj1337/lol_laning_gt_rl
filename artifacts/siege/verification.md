# Verification (siege model)

Stage games of the K = 32 solution re-solved by support enumeration: 172907 games, max |value difference| = 3.55e-15 (107 s). Solution methods: {'saddle': 164777, 'lp': 8130}; crash-choice games: 0 (none exist in the siege model); max certified duality gap 4.4e-15.

## Backward induction vs. tree recursion (support enumeration at every node)

| root | turns | backward induction | tree minimax | |diff| | seconds |
|---|---|---|---|---|---|
| symmetric start | 2 | 0 | 0 | 0.0e+00 | 0 |
| A 2-wave packet near B's tower | 3 | -1.038 | -1.038 | 0.0e+00 | 1 |
| A's 9-HP mass under B's tower, 3 plates | 3 | 1.33 | 1.33 | 0.0e+00 | 1 |
| ... with vision, B's waiting wave stacked | 2 | 1.02 | 1.02 | 0.0e+00 | 0 |
| both sides sieging | 2 | 0 | 0 | 0.0e+00 | 0.1 |

## Exploitability of the computed equilibrium and value antisymmetry

| K | nodes | Expl_A | Expl_B | max |V(s) + V(mirror s)| | max certified gap |
|---|---|---|---|---|---|
| 16 | 4951 | 0.0e+00 | 0.0e+00 | 8.9e-16 | 8.9e-16 |
| 24 | 36891 | 0.0e+00 | 0.0e+00 | 1.8e-15 | 3.6e-15 |
| 32 | 172907 | 0.0e+00 | 0.0e+00 | 3.6e-15 | 4.4e-15 |

## Monte Carlo (4000 games, sampled ganks and mixed actions) vs. exact expectation, K = 32

| A | B | exact | MC mean | stderr | z |
|---|---|---|---|---|---|
| equilibrium | equilibrium | 0 | 0.3013 | 0.2579 | 1.17 |
| equilibrium | uniform | 9.601 | 9.217 | 0.2523 | -1.52 |
| aggressive | equilibrium | -24.47 | -24.73 | 0.2567 | -1.04 |
| freeze_contest | equilibrium | -2.87 | -2.902 | 0.247 | -0.13 |
| passive_safe | equilibrium | -2.87 | -2.88 | 0.2496 | -0.04 |
| stack_builder | equilibrium | -0.32 | -0.205 | 0.2604 | 0.44 |
| release_safe | equilibrium | -2.8 | -3.046 | 0.2505 | -0.98 |
| contest_slowpush | equilibrium | -0.32 | 0.07 | 0.2616 | 1.49 |
| aggressive[vision] | equilibrium | -24.47 | -24.91 | 0.2588 | -1.69 |
| stack_builder[vision] | equilibrium | -0.32 | -0.3687 | 0.2647 | -0.18 |
| contest_slowpush[vision] | equilibrium | -0.32 | -0.8025 | 0.2593 | -1.86 |
| stack_builder[safe] | equilibrium | -0.32 | -0.2575 | 0.2637 | 0.24 |
| aggressive[safe] | equilibrium | -24.47 | -25.09 | 0.2613 | -2.39 |

## Accounting identities over every reachable transition (K = 24, 894272 transitions)

| identity | violations |
|---|---|
| HP conservation (spawned = alive + killed + tower + thinned) | 0 |
| kills recorded = enemy HP removed | 0 |
| stage payoff = farm + plates + gank | 0 |
| plate payoff = plate_value x plates claimed | 0 |

## Earlier models reproduced

- GameConfig.v2() (previous revision), K = 32: 14849 nodes, 1030 mixed, 2480 distinct states; exploitability uniform 10.078, aggressive 24.863, freeze_contest 9.445, passive_safe 9.415, stack_builder 5.782, release_safe 7.690, contest_slowpush 6.590 (revision report: 14849 / 1030 / 2480; 10.08, 24.86, 9.45, 9.42, 5.78, 7.69, 6.59).
- GameConfig.v1() (original), K = 32: 905 nodes, 52 mixed; exploitability uniform 11.8604, always_shove 19.4500, always_slowpush 4.8500, defensive_freeze 1.8500, stack_then_shove 4.8500 (original report: 905 / 52; 11.86 / 19.45 / 4.85 / 1.85 / 4.85).
