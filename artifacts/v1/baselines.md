# Baselines (horizon 32)

Equilibrium value V*(s0) = 0. Entries are exact expected payoffs to the row player (A).

## Head-to-head (exact)

| A \ B | equilibrium | uniform | always_shove | always_slowpush | defensive_freeze | stack_then_shove |
|---|---|---|---|---|---|---|
| equilibrium | 0 | 11.71 | 17.07 | 1.5 | 0 | 1.5 |
| uniform | -11.71 | -0 | 13.19 | -7.313 | -10.16 | -7.313 |
| always_shove | -17.07 | -13.19 | 0 | -19.25 | -14.68 | -19.25 |
| always_slowpush | -1.5 | 7.313 | 19.25 | 0 | -1.5 | 0 |
| defensive_freeze | 0 | 10.16 | 14.68 | 1.5 | 0 | 1.5 |
| stack_then_shove | -1.5 | 7.313 | 19.25 | 0 | -1.5 | 0 |

## Exploitability and exploitation

| baseline (as A) | exploitability | vs equilibrium B | B's best-response actions | B crashes vs it (S/M/L) | A crashes (S/M/L) |
|---|---|---|---|---|---|
| uniform | 11.86 | -11.71 | {'SP': 0.2, 'SH': 0.24, 'F': 0.56} | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] |
| always_shove | 19.45 | -17.07 | {'SP': 0.94, 'F': 0.06} | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] |
| always_slowpush | 4.85 | -1.5 | {'SP': 0.91, 'SH': 0.09} | [0.0, 1.0, 0.0] | [0.0, 0.0, 0.0] |
| defensive_freeze | 1.85 | 0 | {'SP': 0.91, 'SH': 0.09} | [0.0, 1.0, 0.0] | [0.0, 0.0, 0.0] |
| stack_then_shove | 4.85 | -1.5 | {'SP': 0.91, 'SH': 0.09} | [0.0, 1.0, 0.0] | [0.0, 0.0, 0.0] |

## Realized payoff distributions (5000 sampled games each, seed 7)

| matchup | mean | stderr | std | q05 | q50 | q95 | ganks A/B |
|---|---|---|---|---|---|---|---|
| equilibrium vs equilibrium | 0.54 | 0.222 | 15.73 | -25 | 0 | 25 | [6.05, 6.16] |
| uniform vs best response | -11.8 | 0.224 | 15.81 | -40 | -10 | 15 | [7.54, 5.18] |
| always_shove vs best response | -19.34 | 0.227 | 16.08 | -45 | -20 | 5 | [8.68, 4.81] |
| always_slowpush vs best response | -4.688 | 0.221 | 15.59 | -32 | -2 | 23 | [6.16, 5.62] |
| defensive_freeze vs best response | -1.777 | 0.216 | 15.25 | -27 | -2 | 23 | [5.58, 5.62] |
| stack_then_shove vs best response | -4.688 | 0.221 | 15.59 | -32 | -2 | 23 | [6.16, 5.62] |
