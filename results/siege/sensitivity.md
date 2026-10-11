# Experiment H: sensitivity of the siege-study conclusions (K = 24)

Every row re-solves the game exactly. Checks (Y = holds):
- S1 the symmetric equilibrium path never crashes
- S2 Establish Vision is the modal equilibrium objective at siege nodes without active vision
- S3 the 12-HP crash state (3 plates, B's wave waiting, 16 turns left) has positive value for the crasher
- S4 producing a 12-HP crash now beats holding the packet, against B's equilibrium strategy
- S5 aggressive is the most exploitable tracked baseline
- S6 the best response to passive_safe thins minions

Across 16 settings: S1 holds in 16, S2 holds in 16, S3 holds in 15, S4 holds in 15, S5 holds in 16, S6 holds in 14

| setting | nodes | solve s | eq actions (A) | eq P(C) | eq farm | eq gank loss | P(Pressure) / P(Vision) / P(Safe) at siege nodes | P(Vision) no-vision nodes | V(crash 12) | crash gain 12 | post-crash V - P | expl aggressive | expl passive_safe | expl stack_builder | expl contest_slowpush | expl release_safe | BR vs passive: plates / thinned / vision | S1-S6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 36891 | 8.4 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.69 / 0.30 / 0.02 | 0.984 | 6.14 | 1.44 | 1.01 | 18 | 13.8 | 11.4 | 11.8 | 5.2 | 3.0 / 3.0 / 3.0 | YYYYYY |
| gank_base 0.05 | 36891 | 8.5 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 13.9 | 0.73 / 0.25 / 0.02 | 0.96 | 1.98 | 0.89 | 0.64 | 18 | 8.2 | 6 | 6.2 | 5.2 | 3.0 / 3.0 / 3.0 | YYYYYY |
| gank_base 0.10 | 36891 | 8.7 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 19.9 | 0.70 / 0.28 / 0.02 | 0.968 | 4.06 | 1.24 | 0.83 | 18 | 11 | 8.4 | 9 | 5.2 | 3.0 / 3.0 / 3.0 | YYYYYY |
| gank_base 0.20 | 36891 | 8.7 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 31.9 | 0.65 / 0.33 / 0.02 | 0.986 | 8.2 | 1.63 | 1.2 | 18 | 16.6 | 14.4 | 14.6 | 5.2 | 3.0 / 3.0 / 3.0 | YYYYYY |
| gank_loss 3.5 | 36891 | 9.7 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 18.13 | 0.70 / 0.28 / 0.01 | 0.967 | 4.44 | 1.34 | 0.7 | 13.7 | 12.1 | 8.9 | 9.3 | 6.1 | 3.0 / 3.0 / 3.0 | YYYYYY |
| gank_loss 7 | 36891 | 9.9 | SH-C 0.04, SP-C 0.01, SP-S 0.95 | 0.05 | 4.96 | 29.8 | 0.63 / 0.35 / 0.02 | 0.997 | 8.43 | 1.57 | 1.42 | 24.2 | 16.1 | 15 | 15.2 | 3.9 | 3.0 / 3.0 / 3.0 | YYYYYY |
| plate_value 0.5 | 36891 | 8.7 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.62 / 0.36 / 0.02 | 0.998 | 4.64 | 0.85 | 1.01 | 18 | 12.3 | 9.9 | 10.3 | 5.2 | 1.5 / 3.0 / 3.0 | YYYYYY |
| plate_value 1.5 | 36891 | 8.7 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.71 / 0.28 / 0.02 | 0.967 | 7.64 | 1.94 | 1.01 | 18 | 15.4 | 13.6 | 13.9 | 5.2 | 7.5 / 5.0 / 3.0 | YYYYYY |
| plate_grant 1/1/1 | 35481 | 8.5 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.62 / 0.34 / 0.05 | 0.994 | 4.14 | 0.49 | 1.01 | 18 | 13 | 10.7 | 11 | 5.2 | 2.0 / 2.0 / 3.0 | YYYYYY |
| plate_grant 2/3/4 | 39515 | 10.1 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.74 / 0.26 / 0.00 | 0.978 | 7.14 | 1.44 | 1.01 | 18 | 13.8 | 11.4 | 11.8 | 5.2 | 3.0 / 3.0 / 3.0 | YYYYYY |
| thin_damage 0 | 31015 | 6.1 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.63 / 0.31 / 0.05 | 1 | 5.53 | 0.73 | 1.01 | 18 | 13.9 | 11.1 | 11.7 | 5.2 | 3.0 / 0.0 / 3.0 | YYYYYn |
| thin_damage 2 | 45123 | 10.8 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.72 / 0.26 / 0.01 | 0.957 | 11.74 | 4.59 | 0.99 | 18 | 14 | 13 | 12.9 | 5.2 | 7.0 / 14.0 / 3.0 | YYYYYY |
| farm_per_hp 0.125 | 36891 | 9.3 | SH-C 0.04, SP-S 0.96 | 0.04 | 2.47 | 21.25 | 0.66 / 0.30 / 0.04 | 0.989 | 7.39 | 1.38 | 1.01 | 17.1 | 11.3 | 11.8 | 12 | 1.1 | 5.0 / 5.0 / 3.0 | YYYYYY |
| farm_per_hp 0.5 | 36891 | 9.3 | SH-C 0.17, SP-C 0.79, SP-S 0.04 | 0.96 | 14.49 | 26.65 | 0.68 / 0.31 / 0.01 | 0.979 | 3.44 | 1.46 | 0.95 | 21.8 | 18.9 | 10.8 | 12.2 | 13.7 | 3.0 / 3.0 / 3.0 | YYYYYY |
| vision 50% / 2 turns (old) | 17271 | 5 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.64 / 0.34 / 0.02 | 0.69 | -0.5 | 0.65 | 0.09 | 18 | 6.1 | 3.7 | 4.1 | 5.2 | 3.0 / 3.0 / 2.0 | YYnYYY |
| tower 2 HP/tick | 13425 | 2.5 | SH-C 0.08, SP-C 0.79, SP-S 0.12 | 0.88 | 7.13 | 25.9 | 0.52 / 0.48 / 0.00 | 0.963 | 2.78 | -0.01 | 0.72 | 18 | 11.9 | 9.5 | 9.5 | 5.2 | 0.0 / 0.0 / 3.0 | YYYnYn |
