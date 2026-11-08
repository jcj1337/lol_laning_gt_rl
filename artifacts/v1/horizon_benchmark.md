# Horizon benchmark

| K | ticks | reachable (k,s) | max layer | distinct lane states | move games | choice games | LPs | mixed states | seconds | peak MB | max gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 32 | 337 | 43 | 71 | 337 | 40 | 22 | 20 | 0.136 | 0.5 | 2.2e-16 |
| 32 | 64 | 905 | 43 | 71 | 905 | 120 | 54 | 52 | 0.341 | 1.1 | 6.7e-16 |
| 64 | 128 | 2041 | 43 | 71 | 2041 | 280 | 118 | 116 | 0.609 | 2.3 | 1.8e-15 |
| 128 | 256 | 4313 | 43 | 71 | 4313 | 600 | 246 | 244 | 1.198 | 4.9 | 1.8e-15 |
| 256 | 512 | 8857 | 43 | 71 | 8857 | 1240 | 502 | 500 | 2.52 | 10.1 | 7.1e-15 |

## Values and strategies by horizon

| K | V*(s0) | eq at s0 (A) | eq-path gank loss/turn (A) | eq-path crashes A (S/M/L) |
|---|---|---|---|---|
| 16 | 0 | {'SP': 1.0, 'SH': 0.0} | 0.9406 | [0.0, 0.0, 0.0] |
| 32 | 0 | {'SP': 1.0, 'SH': 0.0} | 0.9453 | [0.0, 0.0, 0.0] |
| 64 | 0 | {'SP': 1.0, 'SH': 0.0} | 0.9477 | [0.0, 0.0, 0.0] |
| 128 | 0 | {'SP': 1.0, 'SH': 0.0} | 0.9488 | [0.0, 0.0, 0.0] |
| 256 | 0 | {'SP': 1.0, 'SH': 0.0} | 0.9494 | [0.0, 0.0, 0.0] |

## Exploitability of baselines (as A), total and per turn

| K | uniform | always_shove | always_slowpush | defensive_freeze | stack_then_shove |
|---|---|---|---|---|---|
| 16 | 5.62 (0.351/turn) | 9.45 (0.591/turn) | 1.45 (0.091/turn) | 0.05 (0.003/turn) | 1.45 (0.091/turn) |
| 32 | 11.86 (0.371/turn) | 19.45 (0.608/turn) | 4.85 (0.152/turn) | 1.85 (0.058/turn) | 4.85 (0.152/turn) |
| 64 | 24.33 (0.380/turn) | 39.45 (0.616/turn) | 11.65 (0.182/turn) | 5.45 (0.085/turn) | 11.65 (0.182/turn) |
| 128 | 49.26 (0.385/turn) | 79.45 (0.621/turn) | 25.25 (0.197/turn) | 12.65 (0.099/turn) | 25.25 (0.197/turn) |
| 256 | 99.13 (0.387/turn) | 159.45 (0.623/turn) | 52.45 (0.205/turn) | 27.05 (0.106/turn) | 52.45 (0.205/turn) |

## Probe-state values with 16 turns remaining (independent of K if the end effect dominates)

| K | A 2-stack hp6 at river | A one wave hp2 at river | B wave held at A freeze line |
|---|---|---|---|
| 16 | 2.3 | -2.15 | 0.95 |
| 32 | 2.3 | -2.15 | 0.95 |
| 64 | 2.3 | -2.15 | 0.95 |
| 128 | 2.3 | -2.15 | 0.95 |
| 256 | 2.3 | -2.15 | 0.95 |
