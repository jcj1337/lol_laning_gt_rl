# Experiment F: mechanic ablations (K = 24)

## State space, equilibrium path and the two probes

probe 'crash 12' = value to A of its 12-HP mass under B's tower with B's wave waiting (16 turns left); 'crash gain' = Q(best Shove) - Q(best hold) for A's 12-HP packet one cell from B's empty tower (16 turns left), against B's equilibrium strategy.

| ablation | nodes | mixed | solve s | eq actions (A) | eq P(Contest) | eq farm | eq gank loss | P(Pressure) siege nodes | P(Vision) siege nodes | V(crash 12 state) | crash gain (12 HP) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 previous revision | 6693 | 471 | 1.8 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | - | - | -6.35 | 0.61 |
| 1 plates alone | 15191 | 561 | 4.3 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.525 | 0.444 | -1.06 | 0 |
| 2 thinning alone | 13845 | 750 | 4.2 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.159 | 0.65 | -3 | 0 |
| 3 Apply Pressure (plates + thinning) | 17271 | 731 | 5 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.645 | 0.335 | -0.5 | 0.65 |
| 4 stronger vision alone (75%) | 10403 | 542 | 2.6 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.129 | 0.682 | -0.79 | -0 |
| 5 longer vision alone (6 turns) | 24717 | 1602 | 8.5 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.184 | 0.44 | -0.56 | 0 |
| 6 faster tower alone (2 HP/tick) | 3919 | 299 | 1.1 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | - | - | -4.24 | 0.85 |
| 7 proposed system (default) | 36891 | 2124 | 14 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.686 | 0.298 | 6.14 | 1.44 |
| 8 proposed + tower 2 HP/tick | 13425 | 898 | 3.4 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.519 | 0.481 | 2.78 | -0.01 |
| 9 proposed, thinning 2 HP/turn | 45123 | 2658 | 15.5 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.721 | 0.265 | 11.74 | 4.59 |
| 10 proposed, no plates | 32323 | 2153 | 12.7 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.489 | 0.416 | 3.14 | 0 |

## Exploitability of tracked baselines (as A) [farm / plates / crash / gank parts]

| ablation | passive_safe | stack_builder | contest_slowpush | aggressive | release_safe |
|---|---|---|---|---|---|
| 0 previous revision | 7.26 [-5.4/0.0/-4.0/2.1] | 4.77 [0.5/0.0/-4.0/-1.3] | 5.16 [0.7/0.0/-5.0/-0.8] | 18.00 [-3.6/0.0/0.0/-14.4] | 5.17 [-8.4/0.0/0.0/3.2] |
| 1 plates alone | 5.52 [-4.2/-2.0/0.0/0.7] | 2.79 [1.5/-3.0/0.0/-1.3] | 3.43 [0.8/-2.0/0.0/-2.2] | 18.00 [-3.6/0.0/0.0/-14.4] | 5.17 [-8.4/0.0/0.0/3.2] |
| 2 thinning alone | 4.71 [-6.2/0.0/0.0/1.5] | 2.14 [-0.7/0.0/0.0/-1.5] | 2.35 [-1.0/0.0/0.0/-1.4] | 18.00 [-3.6/0.0/0.0/-14.4] | 5.17 [-8.4/0.0/0.0/3.2] |
| 3 Apply Pressure (plates + thinning) | 6.13 [-4.9/-3.0/0.0/1.7] | 3.75 [0.4/-3.0/0.0/-1.1] | 4.11 [0.1/-3.0/0.0/-1.2] | 18.00 [-3.6/0.0/0.0/-14.4] | 5.17 [-8.4/0.0/0.0/3.2] |
| 4 stronger vision alone (75%) | 5.62 [-6.1/0.0/0.0/0.4] | 2.87 [1.2/0.0/0.0/-4.0] | 3.44 [0.9/0.0/0.0/-4.3] | 18.00 [-3.6/0.0/0.0/-14.4] | 5.17 [-8.4/0.0/0.0/3.2] |
| 5 longer vision alone (6 turns) | 7.57 [-5.2/0.0/0.0/-2.4] | 4.93 [0.1/0.0/0.0/-5.0] | 5.40 [-0.2/0.0/0.0/-5.2] | 18.00 [-3.6/0.0/0.0/-14.4] | 5.17 [-8.4/0.0/0.0/3.2] |
| 6 faster tower alone (2 HP/tick) | 7.19 [-3.8/0.0/-5.0/1.6] | 5.31 [1.0/0.0/-5.0/-1.3] | 5.31 [1.0/0.0/-5.0/-1.3] | 18.00 [-3.6/0.0/0.0/-14.4] | 5.17 [-8.4/0.0/0.0/3.2] |
| 7 proposed system (default) | 13.80 [-5.0/-3.0/0.0/-5.8] | 11.37 [0.6/-3.0/0.0/-9.0] | 11.82 [-0.3/-3.0/0.0/-8.6] | 18.00 [-3.6/0.0/0.0/-14.4] | 5.17 [-8.4/0.0/0.0/3.2] |
| 8 proposed + tower 2 HP/tick | 11.90 [-5.0/0.0/0.0/-6.9] | 9.55 [0.1/0.0/0.0/-9.6] | 9.55 [0.1/0.0/0.0/-9.6] | 18.00 [-3.6/0.0/0.0/-14.4] | 5.17 [-8.4/0.0/0.0/3.2] |
| 9 proposed, thinning 2 HP/turn | 14.02 [-2.6/-7.0/0.0/-4.4] | 13.02 [1.2/-7.0/0.0/-7.2] | 12.87 [1.2/-7.0/0.0/-7.0] | 18.00 [-3.6/0.0/0.0/-14.4] | 5.17 [-8.4/0.0/0.0/3.2] |
| 10 proposed, no plates | 11.99 [-5.2/0.0/0.0/-6.8] | 9.39 [0.1/0.0/0.0/-9.5] | 9.81 [-0.2/0.0/0.0/-9.6] | 18.00 [-3.6/0.0/0.0/-14.4] | 5.17 [-8.4/0.0/0.0/3.2] |

## Best-response paths against passive_safe and stack_builder (per game): the exploiter's sieges

| ablation | vs | BR crashes S/M/L | BR siege turns | BR pressure | BR vision | BR plates | BR thinned HP | baseline bounces | bounce HP | BR crash-choice visions |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 previous revision | passive_safe | 1.00/0.00/1.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 previous revision | stack_builder | 1.00/0.00/1.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 plates alone | passive_safe | 0.00/1.00/0.00 | 4 | 2 | 2 | 2 | 0 | 1 | 8 | 0 |
| 1 plates alone | stack_builder | 0.00/0.00/1.00 | 4 | 3 | 1 | 3 | 0 | 0 | 0 | 0 |
| 2 thinning alone | passive_safe | 0.00/1.00/0.00 | 3 | 1 | 2 | 0 | 1 | 0 | 0 | 0 |
| 2 thinning alone | stack_builder | 0.00/1.00/0.00 | 3 | 1 | 2 | 0 | 1 | 0 | 0 | 0 |
| 3 Apply Pressure (plates + thinning) | passive_safe | 1.00/1.00/0.00 | 5 | 3 | 2 | 3 | 3 | 0 | 0 | 0 |
| 3 Apply Pressure (plates + thinning) | stack_builder | 1.00/1.00/0.00 | 5 | 3 | 2 | 3 | 3 | 0 | 0 | 0 |
| 4 stronger vision alone (75%) | passive_safe | 0.00/1.00/0.00 | 3 | 0 | 2 | 0 | 0 | 1 | 8 | 0 |
| 4 stronger vision alone (75%) | stack_builder | 0.00/1.00/0.00 | 4 | 0 | 3 | 0 | 0 | 1 | 8 | 0 |
| 5 longer vision alone (6 turns) | passive_safe | 3.00/0.00/0.00 | 4 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| 5 longer vision alone (6 turns) | stack_builder | 3.00/0.00/0.00 | 4 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| 6 faster tower alone (2 HP/tick) | passive_safe | 0.00/1.00/1.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 faster tower alone (2 HP/tick) | stack_builder | 0.00/1.00/1.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 proposed system (default) | passive_safe | 2.00/1.00/0.00 | 6 | 3 | 3 | 3 | 3 | 0 | 0 | 0 |
| 7 proposed system (default) | stack_builder | 2.00/1.00/0.00 | 6 | 3 | 3 | 3 | 3 | 0 | 0 | 0 |
| 8 proposed + tower 2 HP/tick | passive_safe | 3.00/0.00/0.00 | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| 8 proposed + tower 2 HP/tick | stack_builder | 3.00/0.00/0.00 | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| 9 proposed, thinning 2 HP/turn | passive_safe | 2.00/1.00/1.00 | 10 | 7 | 3 | 7 | 14 | 0 | 0 | 0 |
| 9 proposed, thinning 2 HP/turn | stack_builder | 2.00/1.00/1.00 | 10 | 7 | 3 | 7 | 14 | 0 | 0 | 0 |
| 10 proposed, no plates | passive_safe | 3.00/0.00/0.00 | 4 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| 10 proposed, no plates | stack_builder | 3.00/0.00/0.00 | 4 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
