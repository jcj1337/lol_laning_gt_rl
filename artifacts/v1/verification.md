# Verification

Stage games of the K=32 solution re-solved by support enumeration: 905 games, max |value difference| = 8.88e-16.

## Backward induction vs tree recursion (support enumeration at every node)

| root | turns | backward induction | tree minimax | |diff| |
|---|---|---|---|---|
| symmetric start | 2 | 0 | 0 | 0.0e+00 |
| symmetric start | 3 | 0 | 0 | 0.0e+00 |
| symmetric start | 4 | 0 | 0 | 0.0e+00 |
| A 2-stack at river | 2 | 2.075 | 2.075 | 0.0e+00 |
| A 2-stack at river | 3 | 1.475 | 1.475 | 0.0e+00 |
| A 2-stack at river | 4 | 1.25 | 1.25 | 0.0e+00 |
| A medium wave near B tower | 2 | -0.15 | -0.15 | 0.0e+00 |
| A medium wave near B tower | 3 | -0.15 | -0.15 | 0.0e+00 |
| A medium wave near B tower | 4 | -0.375 | -0.375 | 0.0e+00 |
| B wave at A freeze line | 2 | 0.475 | 0.475 | 0.0e+00 |
| B wave at A freeze line | 3 | 0.475 | 0.475 | 0.0e+00 |
| B wave at A freeze line | 4 | 0.475 | 0.475 | 0.0e+00 |
| two-packet siege, A vision | 2 | -0.375 | -0.375 | 0.0e+00 |
| two-packet siege, A vision | 3 | -0.8 | -0.8 | 0.0e+00 |
| two-packet siege, A vision | 4 | -0.875 | -0.875 | 0.0e+00 |

## Exploitability of the computed equilibrium

| K | Expl_A | Expl_B | max certified gap |
|---|---|---|---|
| 16 | 0.0e+00 | 0.0e+00 | 2.2e-16 |
| 32 | 0.0e+00 | 0.0e+00 | 6.7e-16 |
| 64 | 0.0e+00 | 0.0e+00 | 1.8e-15 |

## Monte Carlo (4000 games, sampled ganks) vs exact expectation, K = 32

| A | B | exact | MC mean | stderr | z |
|---|---|---|---|---|---|
| equilibrium | equilibrium | 0 | -0.0037 | 0.2486 | -0.02 |
| equilibrium | uniform | 11.71 | 11.69 | 0.2475 | -0.08 |
| always_shove | equilibrium | -17.07 | -16.63 | 0.2562 | 1.74 |
| always_slowpush | equilibrium | -1.5 | -1.846 | 0.2425 | -1.43 |
| defensive_freeze | equilibrium | 0 | -0.2625 | 0.2496 | -1.05 |
| stack_then_shove | equilibrium | -1.5 | -1.312 | 0.2537 | 0.74 |
