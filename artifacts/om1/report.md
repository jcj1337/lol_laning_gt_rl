# OM1 opponent-modelling feasibility: generated tables

Protocol `docs\opponent_modelling\om1_protocol.md` SHA-256 `e6972991db19de8b…` (frozen 2026-10-17 01:48:14). Values: the responder's earned-gold lead at turn 114 (= E_20 at turn 94 in expectation), resources, mean over paired-seat pairs.

## Smoke

| check | result |
|---|---|
| deterministic re-run | True |
| self-play pair lead is exactly 0 | True |
| posterior normalized | True |
| switcher runs and records | True |
| switcher equals pi until its first switch | True |

Throughput (turns/s, one worker): script_vs_script 4644, pi_vs_script 1202, pi_vs_net 1062, pi_vs_net_tracked 371, switch_vs_script 346

## Stage A: selection (every menu responder against every library type; 64 pairs)

| type | pi's value | best menu responder | its value | gain over pi | r* (gain >= 0.25) |
|---|---|---|---|---|---|
| g_farm | +2.69 | g_shove_ward | +3.85 | +1.16 | g_shove_ward |
| g_shove | +4.46 | freezer | +5.36 | +0.90 | freezer |
| g_shove_sr | +4.22 | freezer | +5.55 | +1.32 | freezer |
| wk_shove_reset_sh | +4.66 | freezer | +5.28 | +0.63 | freezer |
| g_shove_edge | +4.97 | freezer | +6.64 | +1.67 | freezer |
| g_bully | +8.89 | g_shove_ward | +10.97 | +2.07 | g_shove_ward |
| g_safe | +6.61 | g_farm | +12.12 | +5.51 | g_farm |
| fz_guard_sh | +4.15 | freezer | +5.55 | +1.40 | freezer |
| g_shove_ward | +3.38 | freezer | +5.49 | +2.11 | freezer |
| peer | +1.22 | attacker | +1.96 | +0.74 | attacker |
| attacker | +0.74 | short | +2.01 | +1.27 | short |
| freezer | +0.05 | short | +1.51 | +1.46 | short |
| gapper | +2.75 | short | +2.95 | +0.20 | pi |
| short | -0.27 | short | +0.00 | +0.27 | short |
| early | +3.87 | pi | +3.87 | +0.00 | pi |

Mean value over the library: pi +3.49; best fixed candidate freezer +3.53; r_fix = pi.

## Identifiability (pi's games; 6 seeds x 96 pairs x 2 orientations per type)

| turns observed | 1 | 2 | 3 | 5 | 10 | 15 | 20 | 30 | 40 | 60 | 94 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| true type's posterior >= 0.9 (full signal) | 0% | 8% | 25% | 35% | 52% | 62% | 69% | 76% | 80% | 83% | 86% |
| ... coarse signal (no wave choice) | 0% | 0% | 13% | 21% | 32% | 41% | 49% | 62% | 69% | 75% | 79% |
| ... full, eps 0.01 | 0% | 8% | 25% | 35% | 52% | 62% | 69% | 76% | 80% | 83% | 86% |
| ... full, eps 0.2 | 0% | 7% | 24% | 34% | 52% | 62% | 69% | 76% | 80% | 83% | 86% |
| MAP type correct | 33% | 39% | 54% | 62% | 71% | 75% | 79% | 84% | 86% | 89% | 91% |
| decision-relevant (r* of MAP = r* of truth) | 78% | 84% | 86% | 91% | 96% | 98% | 99% | 100% | 100% | 100% | 100% |
| decision-relevant, coarse signal | 28% | 39% | 48% | 54% | 82% | 89% | 93% | 97% | 99% | 100% | 100% |

Mean over the 15 types. **I (decision-relevant identification at turn 40 >= 2/3): met.**

| true type | P>=0.9 @5 | @10 | @20 | @40 | @94 | coarse @40 | decision-relevant @40 | MAP after 40 turns |
|---|---|---|---|---|---|---|---|---|
| g_farm | 99% | 100% | 100% | 100% | 100% | 10% | 100% | g_farm 100% |
| g_shove | 0% | 0% | 1% | 5% | 18% | 4% | 100% | g_shove 100% |
| g_shove_sr | 0% | 17% | 29% | 51% | 76% | 51% | 100% | g_shove_sr 51%, g_shove 49% |
| wk_shove_reset_sh | 10% | 14% | 21% | 30% | 36% | 30% | 100% | g_shove 70%, wk_shove_reset_sh 30% |
| g_shove_edge | 8% | 11% | 40% | 82% | 98% | 80% | 100% | g_shove_edge 82%, g_shove 18% |
| g_bully | 99% | 100% | 100% | 100% | 100% | 100% | 100% | g_bully 100% |
| g_safe | 100% | 100% | 100% | 100% | 100% | 100% | 100% | g_safe 100% |
| fz_guard_sh | 0% | 0% | 18% | 35% | 59% | 20% | 100% | g_shove 65%, fz_guard_sh 35% |
| g_shove_ward | 98% | 99% | 100% | 100% | 100% | 100% | 100% | g_shove_ward 100% |
| peer | 7% | 39% | 81% | 98% | 100% | 86% | 99% | peer 99%, attacker 0%, short 0% |
| attacker | 6% | 46% | 86% | 99% | 100% | 86% | 100% | attacker 99%, peer 0%, freezer 0% |
| freezer | 16% | 62% | 88% | 99% | 100% | 91% | 100% | freezer 100%, gapper 0%, peer 0% |
| gapper | 35% | 76% | 92% | 99% | 100% | 93% | 100% | gapper 100%, freezer 0% |
| short | 10% | 47% | 86% | 99% | 100% | 86% | 100% | short 100%, peer 0%, attacker 0% |
| early | 34% | 73% | 95% | 100% | 100% | 96% | 100% | early 100%, peer 0% |

## Value of information (stage B; six seeds as units, mean over the 15 library types)

| contrast | lead at 114 (E_20 at 94) | lane lead at 94 | lead at 150 |
|---|---|---|---|
| switcher − pi (G_B) | +0.90 ± 0.27 | +0.71 ± 0.24 | +1.52 ± 0.30 |
| oracle − pi (G_O) | +1.09 ± 0.44 | +0.77 ± 0.41 | +1.72 ± 0.43 |
| oracle − best fixed responder (VOI) | +1.09 ± 0.44 |  |  |
| pi's own value (reference) | +3.43 ± 0.41 |  |  |

| type | r* | pi's value | oracle − pi | switcher − pi | oracle − pi at 150 | switcher: turns switched | first switch (median turn) | switcher's play by responder |
|---|---|---|---|---|---|---|---|---|
| g_farm | g_shove_ward | +2.65 ± 0.17 | +1.18 ± 0.18 | +0.99 ± 0.19 | +1.85 ± 0.21 | 98% | 3 | g_shove_ward 98%, pi 2% |
| g_shove | freezer | +4.18 ± 0.28 | +0.96 ± 0.37 | +0.24 ± 0.13 | +1.62 ± 0.39 | 13% | never | pi 87%, freezer 13% |
| g_shove_sr | freezer | +3.79 ± 0.27 | +1.73 ± 0.36 | +1.08 ± 0.28 | +2.44 ± 0.41 | 62% | 40 | freezer 62%, pi 38% |
| wk_shove_reset_sh | freezer | +4.28 ± 0.26 | +1.07 ± 0.34 | +0.42 ± 0.21 | +1.45 ± 0.37 | 31% | never | pi 69%, freezer 31% |
| g_shove_edge | freezer | +4.73 ± 0.29 | +1.55 ± 0.36 | +1.74 ± 0.36 | +2.21 ± 0.39 | 81% | 24 | freezer 81%, pi 19% |
| g_bully | g_shove_ward | +9.18 ± 0.28 | +1.31 ± 0.33 | +1.40 ± 0.31 | +2.17 ± 0.38 | 98% | 3 | g_shove_ward 98%, pi 2% |
| g_safe | g_farm | +6.92 ± 0.14 | +4.65 ± 0.21 | +4.08 ± 0.21 | +9.24 ± 0.24 | 99% | 2 | g_farm 99%, pi 1% |
| fz_guard_sh | freezer | +4.09 ± 0.26 | +0.77 ± 0.36 | +0.82 ± 0.24 | +0.97 ± 0.40 | 47% | 68 | pi 53%, freezer 47% |
| g_shove_ward | freezer | +3.24 ± 0.26 | +1.44 ± 0.36 | +1.51 ± 0.35 | +2.02 ± 0.39 | 98% | 3 | freezer 98%, pi 2% |
| peer | attacker | +1.79 ± 0.26 | -0.46 ± 0.39 | -0.61 ± 0.36 | -0.51 ± 0.46 | 91% | 11 | attacker 90%, pi 9%, short 0% |
| attacker | short | +0.63 ± 0.26 | +1.27 ± 0.36 | +1.10 ± 0.36 | +1.57 ± 0.42 | 91% | 10 | short 91%, pi 9%, attacker 0% |
| freezer | short | +0.84 ± 0.29 | +0.62 ± 0.41 | +0.58 ± 0.38 | +0.57 ± 0.47 | 92% | 8 | short 92%, pi 8%, attacker 0% |
| gapper | pi | +1.96 ± 0.33 | +0.00 ± 0.00 | -0.07 ± 0.06 | +0.00 ± 0.00 | 0% | never | pi 100%, short 0%, attacker 0% |
| short | short | -0.25 ± 0.25 | +0.25 ± 0.25 | +0.28 ± 0.34 | +0.23 ± 0.29 | 92% | 10 | short 91%, pi 8%, attacker 0% |
| early | pi | +3.39 ± 0.25 | +0.00 ± 0.00 | +0.00 ± 0.06 | +0.00 ± 0.00 | 0% | never | pi 100%, short 0%, attacker 0% |

## Robustness (stage C: out-of-library opponents)

| opponent | switcher − pi (lead at 114) | at 150 | believed to be (MAP after 40 turns) | switcher's play by responder |
|---|---|---|---|---|
| attacker_b | -0.04 ± 0.49 | -0.20 ± 0.56 | short 39%, attacker 29%, peer 27% | short 47%, pi 34%, attacker 19% |
| freezer_b | +0.06 ± 0.37 | +0.38 ± 0.44 | freezer 51%, gapper 36%, attacker 6% | pi 58%, short 40%, attacker 2% |
| gapper_b | -0.01 ± 0.26 | +0.06 ± 0.30 | gapper 82%, freezer 17%, attacker 0% | pi 93%, short 7%, attacker 0% |
| short_b | +0.34 ± 0.45 | +0.31 ± 0.51 | attacker 40%, short 30%, peer 19% | short 54%, pi 38%, attacker 8% |
| c1_policy | +0.31 ± 0.36 | -0.06 ± 0.43 | attacker 51%, short 20%, peer 20% | short 53%, pi 36%, attacker 10% |
| mix_bully_safe | -0.83 ± 0.38 | -0.86 ± 0.44 | g_bully 53%, g_safe 47% | g_shove_ward 46%, g_farm 45%, pi 8% |
| mix_shove_farm | -1.45 ± 0.33 | -2.04 ± 0.39 | g_shove 49%, g_farm 48%, peer 2% | pi 55%, g_shove_ward 40%, freezer 5% |

Pooled per seed: -0.23 ± 0.24; R violated: False.

## Classification (protocol §4)

| criterion | result |
|---|---|
| I met | True |
| G_B material | True |
| G_O small | False |
| narrow types | g_farm, g_shove, g_shove_sr, wk_shove_reset_sh, g_shove_edge, g_bully, g_safe, fz_guard_sh, g_shove_ward, attacker, freezer |
| R violated | False |
| **outcome** | **1: feasible and worthwhile** |
