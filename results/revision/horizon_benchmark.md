# Horizon benchmark (revised model)

Single-threaded Python; timings from an untraced run, peak memory from a separate tracemalloc run.

| K | ticks | decision nodes | max layer | distinct states | cached transitions | transitions/s | stage games | choice games | LPs | mixed | enumerate s | solve s | BR s (each) | peak MB | max gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 32 | 1799 | 384 | 604 | 21552 | 20482 | 1799 | 112 | 154 | 152 | 1.05 | 0.74 | 0.06 | 22.6 | 4.4e-16 |
| 32 | 64 | 14849 | 1302 | 2480 | 91744 | 17868 | 14849 | 802 | 1044 | 1030 | 5.13 | 3.33 | 0.61 | 107.5 | 1.8e-15 |
| 64 | 128 | 50769 | 1344 | 2518 | 94112 | 12414 | 50769 | 2498 | 3717 | 3683 | 7.58 | 9.78 | 1.75 | 153 | 2.7e-15 |
| 128 | 256 | 122609 | 1344 | 2518 | 94112 | 8182 | 122609 | 5890 | 9098 | 9038 | 11.5 | 20.86 | 4.08 | 240 | 5.0e-14 |

## Values and the equilibrium path by horizon

| K | V*(s0) | eq at s0 (A) | eq-path farm/turn | eq-path gank loss/turn | eq-path P(Contest) |
|---|---|---|---|---|---|
| 16 | 0 | {'SP-C': 0.0, 'SH-C': 0.0, 'SP-S': 1.0, 'SH-S': 0.0} | 0.2881 | 1.069 | 0.812 |
| 32 | 0 | {'SP-C': 0.0, 'SH-C': 0.0, 'SP-S': 1.0, 'SH-S': 0.0} | 0.3016 | 1.084 | 0.906 |
| 64 | 0 | {'SP-C': 0.0, 'SH-C': 0.0, 'SP-S': 1.0, 'SH-S': 0.0} | 0.3083 | 1.092 | 0.953 |
| 128 | 0 | {'SP-C': 0.0, 'SH-C': 0.0, 'SP-S': 1.0, 'SH-S': 0.0} | 0.3116 | 1.096 | 0.977 |

## Exploitability of baselines (as A): total, and per turn

| K | uniform | aggressive | freeze_contest | passive_safe | stack_builder | release_safe | contest_slowpush |
|---|---|---|---|---|---|---|---|
| 16 | 3.74 (0.233) | 11.14 (0.696) | 4.63 (0.290) | 4.97 (0.311) | 4.04 (0.253) | 2.65 (0.166) | 4.04 (0.253) |
| 32 | 10.08 (0.315) | 24.86 (0.777) | 9.45 (0.295) | 9.42 (0.294) | 5.78 (0.181) | 7.69 (0.240) | 6.59 (0.206) |
| 64 | 23.55 (0.368) | 52.30 (0.817) | 18.34 (0.287) | 18.09 (0.283) | 21.22 (0.332) | 17.77 (0.278) | 15.86 (0.248) |
| 128 | 50.55 (0.395) | 107.18 (0.837) | 52.20 (0.408) | 52.61 (0.411) | 54.18 (0.423) | 37.93 (0.296) | 55.22 (0.431) |

## Probe values with 16 turns remaining (same phase), by horizon

The dynamics are time-homogeneous up to the phase k mod 4, so these must be identical for every K.

| K | B wave at A's freeze line, fighting A's wave | A 2-wave stack at +1, B wave just spawned | A's 3-wave crash under B's tower |
|---|---|---|---|
| 16 | 0.95 | -2.473 | -6.35 |
| 32 | 0.95 | -2.473 | -6.35 |
| 64 | 0.95 | -2.473 | -6.35 |
| 128 | 0.95 | -2.473 | -6.35 |
