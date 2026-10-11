# Verification (revised model)

Stage games of the K = 32 solution re-solved by support enumeration: 14849 games, max |value difference| = 8.88e-16; crash-choice games: 802, max difference 8.88e-16. Solution methods: {'saddle': 13805, 'choice-trivial': 802, 'lp': 1044}; max certified duality gap 1.8e-15.

## Backward induction vs. tree recursion (support enumeration at every node)

| root | turns | backward induction | tree minimax | |diff| | seconds |
|---|---|---|---|---|---|
| symmetric start | 3 | 0 | 0 | 0.0e+00 | 0.4 |
| A 3-wave stack at the river | 2 | 3.615 | 3.615 | 8.9e-16 | 0 |
| A 2-wave packet near B's tower | 2 | -0.6225 | -0.6225 | 0.0e+00 | 0.1 |
| B wave at A's freeze line | 2 | 0.9 | 0.9 | 0.0e+00 | 0 |
| crashed mass under B's tower, A vision | 2 | -0.6125 | -0.6125 | 0.0e+00 | 0 |

## Exploitability of the computed equilibrium

| K | Expl_A | Expl_B | max certified gap |
|---|---|---|---|
| 16 | 0.0e+00 | 0.0e+00 | 4.4e-16 |
| 32 | 0.0e+00 | 0.0e+00 | 1.8e-15 |
| 64 | 0.0e+00 | 0.0e+00 | 2.7e-15 |

## Monte Carlo (4000 games, sampled ganks and mixed actions) vs. exact expectation, K = 32

| A | B | exact | MC mean | stderr | z |
|---|---|---|---|---|---|
| equilibrium | equilibrium | 0 | 0.0375 | 0.2593 | 0.14 |
| equilibrium | uniform | 9.859 | 10.03 | 0.2558 | 0.67 |
| aggressive | equilibrium | -24.47 | -24.45 | 0.2602 | 0.08 |
| freeze_contest | equilibrium | -2.87 | -2.874 | 0.242 | -0.02 |
| passive_safe | equilibrium | -2.87 | -3.136 | 0.252 | -1.06 |
| stack_builder | equilibrium | -0.32 | -0.0187 | 0.2649 | 1.14 |
| release_safe | equilibrium | -2.8 | -2.973 | 0.2493 | -0.69 |
| contest_slowpush | equilibrium | -0.32 | -0.3275 | 0.256 | -0.03 |

## Original model reproduced by GameConfig.v1() (K = 32)

905 decision nodes, 52 mixed, V*(s0) = 0; exploitability: uniform 11.8604, always_shove 19.4500, always_slowpush 4.8500, defensive_freeze 1.8500, stack_then_shove 4.8500 (original report: 905 nodes, 52 mixed, 11.86 / 19.45 / 4.85 / 1.85 / 4.85).
