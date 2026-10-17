# M4.8 lane-exit evaluation: generated tables

Protocol `docs\m4\m4_8\m4_8_protocol.md` SHA-256 `57c99a63a42d71b4…` (frozen 2026-10-16 23:50:55). Generated from `results/m48/pilot.json`, `validation.json`, `application.json`. Resources: 1 = one fresh wave. Seat orientation: case tables are folded to the focal seat X; natural-state tables are in seat A's orientation unless folded to the category's X.

## Pilot (throughput and within-state spread)

24 states at turn 94, K 64: 1003 turns/s per worker, 4312 on the wall clock.

| h | SD of Δ_h p25 / p50 / p75 / p90 | K for ±0.25 (p50) | K for ±0.5 (p50) |
|---|---|---|---|
| 5 | 0.67 / 0.82 / 1.05 / 1.65 | 42 | 11 |
| 10 | 1.22 / 1.35 / 1.81 / 2.56 | 113 | 29 |
| 20 | 2.01 / 2.16 / 2.43 / 2.86 | 287 | 72 |
| 30 | 2.37 / 2.60 / 2.74 / 3.33 | 416 | 104 |
| 40 | 2.71 / 2.94 / 3.26 / 4.10 | 534 | 134 |

## Checks (V1-V3)

| check | result |
|---|---|
| V1 accounting: lead(final) − D0 = Σ reward units, every continuation | validation 60,032 continuations, max residual 0 units; natural max 0 |
| V2 orientation: mirrored continuation = exact negation | 96 states, 12,288 continuations, max \|r + r_mirror\| = 0.0 |
| V3 restoration: natural checkpoints replay their records | 2304 checkpoints, every turn verified against the M4.6d records |

## Stage 1: validation cases (folded to the focal seat X)

| case | n | baseline/C1 | X's raw lead D0 | Δ5 | Δ10 | Δ20 | Δ40 | material @10 / @20 | pending resolved by 10 / 20 | recorded Δ10 (one sample per state) |
|---|---|---|---|---|---|---|---|---|---|---|
| returner | 96 | 48/48 | -2.80 ± 0.69 | +1.84 ± 0.15 | +2.55 ± 0.19 | +2.13 ± 0.16 | +2.13 ± 0.18 | 98% / 99% | 66% / 95% | +2.70 ± 0.38 |
| lost | 85 | 37/48 | -3.43 ± 0.73 | -0.20 ± 0.28 | +0.91 ± 0.25 | +0.88 ± 0.24 | +0.80 ± 0.26 | 73% / 73% | 87% / 98% | +0.81 ± 0.39 |
| denied | 96 | 48/48 | +0.92 ± 0.71 | -0.77 ± 0.19 | -1.08 ± 0.21 | -0.96 ± 0.18 | -0.92 ± 0.20 | 72% / 79% | 80% / 95% | -0.99 ± 0.34 |
| stable | 48 | 24/24 | -0.70 ± 1.02 | -0.01 ± 0.13 | -0.03 ± 0.16 | -0.04 ± 0.14 | -0.17 ± 0.15 | 38% / 35% | 100% / 100% | -0.30 ± 0.32 |
| opposing | 96 | 48/48 | +2.97 ± 0.45 | -1.56 ± 0.17 | -1.73 ± 0.23 | -1.15 ± 0.21 | -1.23 ± 0.22 | 91% / 79% | 80% / 97% | -2.51 ± 0.50 |
| due | 48 | 24/24 | +0.82 ± 1.06 | -0.40 ± 0.34 | -0.60 ± 0.35 | -0.55 ± 0.28 | -0.57 ± 0.30 | 65% / 58% | 61% / 91% | -0.46 ± 0.52 |

Orientation check (case X = A against X = B):

| case | n (X=A / X=B) | Δ10 X=A | Δ10 X=B | Δ20 X=A | Δ20 X=B |
|---|---|---|---|---|---|
| returner | 48 / 48 | +2.48 ± 0.25 | +2.62 ± 0.29 | +2.03 ± 0.20 | +2.23 ± 0.26 |
| lost | 42 / 43 | +0.95 ± 0.34 | +0.88 ± 0.37 | +0.92 ± 0.34 | +0.84 ± 0.33 |
| denied | 48 / 48 | -1.24 ± 0.29 | -0.92 ± 0.29 | -1.17 ± 0.24 | -0.76 ± 0.27 |
| opposing | 48 / 48 | -1.74 ± 0.32 | -1.73 ± 0.33 | -1.18 ± 0.30 | -1.12 ± 0.31 |

Event accounting over the first 10 turns (means per continuation; X / Y):

| case | farm collected | farm available | farm lost while away | bounty X−Y | plates X−Y | returns | turns away | deaths |
|---|---|---|---|---|---|---|---|---|
| returner | 5.01 / 2.62 | 6.33 / 3.84 | 0.06 / 0.39 | +0.34 | -0.17 | 0.14 / 0.68 | 0.30 / 1.81 | 0.02 / 0.19 |
| lost | 4.14 / 2.55 | 6.92 / 3.80 | 1.74 / 0.44 | +0.37 | -1.04 | 1.02 / 0.61 | 2.49 / 1.80 | 0.00 / 0.19 |
| denied | 3.37 / 4.45 | 4.55 / 5.73 | 0.29 / 0.25 | -0.12 | +0.12 | 0.52 / 0.44 | 1.33 / 0.96 | 0.11 / 0.05 |
| stable | 3.86 / 3.85 | 4.87 / 4.86 | 0.15 / 0.15 | -0.01 | -0.02 | 0.36 / 0.37 | 0.85 / 0.86 | 0.04 / 0.03 |
| opposing | 2.94 / 4.16 | 4.52 / 5.43 | 0.58 / 0.15 | -0.27 | -0.24 | 0.81 / 0.21 | 2.13 / 0.60 | 0.21 / 0.07 |
| due | 3.48 / 4.01 | 4.77 / 5.26 | 0.34 / 0.26 | -0.05 | -0.01 | 0.64 / 0.43 | 1.53 / 1.05 | 0.10 / 0.07 |

Event accounting over the first 20 turns (means per continuation; X / Y):

| case | farm collected | farm available | farm lost while away | bounty X−Y | plates X−Y | returns | turns away | deaths |
|---|---|---|---|---|---|---|---|---|
| returner | 8.53 / 6.67 | 11.05 / 9.23 | 0.36 / 0.74 | +0.28 | -0.02 | 0.68 / 1.21 | 1.60 / 3.03 | 0.12 / 0.26 |
| lost | 7.82 / 6.48 | 11.73 / 9.09 | 1.97 / 0.84 | +0.36 | -0.82 | 1.46 / 1.19 | 3.56 / 3.17 | 0.08 / 0.26 |
| denied | 7.23 / 8.13 | 9.69 / 10.64 | 0.62 / 0.54 | -0.11 | +0.06 | 1.08 / 0.96 | 2.62 / 2.19 | 0.19 / 0.13 |
| stable | 7.66 / 7.63 | 9.90 / 9.87 | 0.45 / 0.45 | -0.02 | -0.04 | 0.90 / 0.92 | 2.10 / 2.12 | 0.12 / 0.11 |
| opposing | 6.94 / 7.64 | 9.82 / 10.19 | 0.89 / 0.53 | -0.17 | -0.29 | 1.28 / 0.82 | 3.24 / 2.12 | 0.27 / 0.19 |
| due | 7.29 / 7.74 | 9.85 / 10.23 | 0.67 / 0.57 | -0.05 | -0.05 | 1.17 / 0.98 | 2.79 / 2.33 | 0.17 / 0.15 |

Non-gold dimensions and resolution (X's view):

| case | XP lead at t0 | XP lead +10 | XP lead +20 | HP lead at t0 (both present) | HP lead +10 | back by 10 (X / Y) | back by 20 (X / Y) | stock gone, median turn (X / Y) | stock collected share (X / Y) |
|---|---|---|---|---|---|---|---|---|---|
| returner | -21.9 ± 2.7 | +1.5 ± 2.4 | -3.6 ± 2.3 | +0.41 ± 0.05 | -0.01 ± 0.01 | 14% / 66% | 61% / 95% | 3 / 1 | 0.79 / 0.62 |
| lost | -21.5 ± 2.6 | -5.8 ± 2.7 | -8.5 ± 2.8 | n/a | +0.12 ± 0.02 | 100% / 59% | 100% / 93% | 4 / 1 | 0.36 / 0.55 |
| denied | +9.0 ± 2.7 | -1.3 ± 2.5 | +0.4 ± 2.3 | -0.06 ± 0.07 | -0.02 ± 0.02 | 51% / 43% | 87% / 79% | 2 / 3 | 0.75 / 0.78 |
| stable | -1.4 ± 4.1 | -1.3 ± 3.8 | -1.1 ± 3.7 | +0.02 ± 0.04 | +0.00 ± 0.01 | 36% / 37% | 77% / 78% | 3 / 3 | 0.82 / 0.82 |
| opposing | +10.8 ± 2.7 | -0.1 ± 2.5 | +4.9 ± 2.5 | -0.51 ± 0.04 | +0.10 ± 0.02 | 79% / 21% | 98% / 74% | 4 / 4 | 0.51 / 0.76 |
| due | +7.3 ± 3.5 | +2.5 ± 2.8 | +3.3 ± 2.7 | -0.11 ± 0.09 | +0.01 ± 0.03 | 63% / 43% | 93% / 81% | 4 / 3 | 0.70 / 0.76 |

## Stage 1: controlled variants on due states (seat A; D0 unchanged by construction)

48 states; A's unspent gold +7.55 ± 0.69, HP +0.65 ± 0.05, turns since return +15.4 ± 1.2.

| variant − orig | E5 | E10 | E20 | E40 | material @10 | @10 baseline / C1 |
|---|---|---|---|---|---|---|
| A_fresh | +0.56 ± 0.20 | +0.87 ± 0.18 | +0.62 ± 0.13 | +0.66 ± 0.14 | 67% | +0.90 / +0.83 |
| A_away1 | +0.20 ± 0.26 | +0.49 ± 0.23 | +0.36 ± 0.14 | +0.38 ± 0.17 | 62% | +0.47 / +0.52 |

## Worked example (the first returner state whose returner trails by < 2)

baseline seed 0, validation game 2, turn 52; X = seat B.

|  | X | Y |
|---|---|---|
| status (0 present, 1 Recall, 2 dead) | 0.0 | 0.0 |
| turns still away | 0.0 | 0.0 |
| turns since last return | 1.0 | 13.0 |
| HP | 1.00 | 0.64 |
| level / XP | 10 / 182 | 10 / 190 |
| earned (invested + unspent) | 18.91 (18.73 + 0.18) | 19.09 (13.10 + 5.99) |
| enemy minion HP alive on its ledger (at its own −1 or deeper) | 21 (13) | 8 (0) |

| h | E_h (X's lead) | 95% ± | Δ_h | pending resolved | recorded change (one sample) | XP lead at h |
|---|---|---|---|---|---|---|
| 5 | +2.09 | 0.13 | +2.28 | 45% | +1.62 | +11.8 |
| 10 | +2.44 | 0.24 | +2.62 | 83% | +3.65 | +13.6 |
| 20 | +1.78 | 0.31 | +1.96 | 100% | +0.39 | +7.2 |
| 30 | +1.91 | 0.38 | +2.09 | 100% | +1.65 | +8.6 |
| 40 | +2.01 | 0.51 | +2.19 | 100% | +1.91 | +8.6 |

| h | farm | available | lost away | bounty | plate | returns | turns away |
|---|---|---|---|---|---|---|---|
| 5 | 2.88 / 0.64 | 3.44 / 1.32 | 0.00 / 0.17 | 0.05 / 0.00 | 0.00 / 0.01 | 0.00 / 0.55 | 0.02 / 1.21 |
| 10 | 4.99 / 2.52 | 6.00 / 3.77 | 0.03 / 0.31 | 0.12 / 0.02 | 0.05 / 0.02 | 0.08 / 0.86 | 0.20 / 1.91 |
| 20 | 8.50 / 6.72 | 10.67 / 9.19 | 0.28 / 0.51 | 0.17 / 0.12 | 0.21 / 0.08 | 0.62 / 1.35 | 1.42 / 2.89 |

The game's recorded continuation (one realized sample):

| turn | X's lead change | farm X | farm Y | away X | away Y | events |
|---|---|---|---|---|---|---|
| 52 | +1.30 | 1.30 | 0.00 |  |  |  |
| 53 | +0.11 | 0.11 | 0.00 |  |  |  |
| 54 | -0.12 | 0.10 | 0.22 |  |  |  |
| 55 | +0.34 | 0.34 | 0.00 |  |  |  |
| 56 | -0.00 | 0.00 | 0.00 |  |  |  |
| 57 | -0.00 | 0.00 | 0.00 |  |  |  |
| 58 | -0.00 | 0.00 | 0.00 |  |  |  |
| 59 | +1.46 | 1.46 | 0.00 |  | yes | Y recalls |
| 60 | +0.34 | 0.34 | 0.00 |  | yes | Y back next turn |
| 61 | +0.23 | 0.90 | 0.68 |  |  |  |
| 62 | -1.30 | 0.00 | 1.30 |  |  |  |
| 63 | -0.40 | 0.00 | 0.40 | yes |  | X recalls |
| 64 | -0.45 | 0.00 | 0.45 | yes |  | X back next turn |
| 65 | +1.01 | 1.24 | 0.22 |  |  |  |
| 66 | -1.00 | 0.00 | 1.00 |  |  |  |

## V6: predictions against the recorded continuations (natural states, turn 94)

| h | slope of realized on predicted | mean residual | R² of prediction | var realized | var predicted | MSE of D0 / E_h for the lead at t0+h |
|---|---|---|---|---|---|---|
| 5 | +1.00 ± 0.08 | +0.01 ± 0.08 | 0.597 | 2.63 | 1.58 | 2.64 / 1.06 |
| 10 | +0.99 ± 0.13 | -0.02 ± 0.14 | 0.381 | 5.03 | 1.97 | 5.04 / 3.11 |
| 20 | +0.99 ± 0.18 | -0.14 ± 0.20 | 0.190 | 7.04 | 1.38 | 7.05 / 5.72 |
| 30 | +0.91 ± 0.20 | -0.12 ± 0.23 | 0.134 | 8.88 | 1.46 | 8.89 / 7.71 |
| 40 | +1.00 ± 0.21 | +0.10 ± 0.26 | 0.135 | 11.47 | 1.55 | 11.49 / 9.93 |

Does E_h predict the later recorded lead better than D0?

| turn | n | corr D0 | corr E5 | corr E10 | corr E20 | corr E40 | ΔMSE E10 − D0 | ΔMSE E20 − D0 |
|---|---|---|---|---|---|---|---|---|
| 150 | 576 | 0.754 | 0.783 | 0.781 | 0.785 | 0.784 | -1.30 ± 0.94 | -1.46 ± 0.76 |
| 200 | 576 | 0.692 | 0.717 | 0.716 | 0.722 | 0.722 | -1.28 ± 0.97 | -1.60 ± 0.83 |
| 250 | 576 | 0.638 | 0.661 | 0.659 | 0.664 | 0.665 | -1.23 ± 1.17 | -1.51 ± 0.98 |

Calibration at turn 84: h 5 slope +0.98 ± 0.09, residual +0.03; h 10 slope +0.91 ± 0.12, residual +0.05; h 20 slope +0.80 ± 0.18, residual +0.05

Calibration at turn 104: h 5 slope +0.94 ± 0.10, residual -0.00; h 10 slope +1.09 ± 0.13, residual -0.13; h 20 slope +1.06 ± 0.16, residual -0.09

## V8: material leader reversals and their accounting

Validation states, h 10: 69 of 469 (15%); dominant component farm (available) 62, farm (absence) 5, plate 2; by case opposing 24, returner 21, denied 11, lost 7, due 5, stable 1

Validation states, h 20: 49 of 469 (10%); dominant component farm (available) 44, plate 3, farm (absence) 1, bounty 1; by case returner 17, opposing 13, denied 10, lost 6, due 2, stable 1

Natural turn-94 states, h 10: 43 of 576 (7%); dominant farm (available) 35, plate 5, farm (absence) 3; by category fresh 14, ordinary 14, due_asym 9, dead 3, both_away 2, stock_asym 1

Natural turn-94 states, h 20: 25 of 576 (4%); dominant farm (available) 20, plate 3, farm (absence) 2; by category fresh 13, due_asym 4, ordinary 4, dead 2, both_away 2

## Stage 2 S1: how often raw checkpoint gold gives a materially different assessment (turn 94)

Levels at turn 94: lower level mean 13.4; both ≥ 14 in 36%, either in 82%; seat levels p10/p50/p90 13 / 14 / 14.

| h | material share baseline | C1 | C1 − baseline | leader reversals (pooled) | SD of the true adjustment | mean \|Δ_h\| | median 95% ± |
|---|---|---|---|---|---|---|---|
| 5 | 67% ± 7 | 66% ± 6 | -1 ± 6 pts | 7% | +1.25 ± 0.07 | +0.95 ± 0.04 | 0.14 |
| 10 | 73% ± 5 | 68% ± 5 | -5 ± 6 pts | 7% | +1.39 ± 0.10 | +1.08 ± 0.06 | 0.24 |
| 20 | 64% ± 4 | 66% ± 8 | +2 ± 6 pts | 4% | +1.15 ± 0.09 | +0.92 ± 0.06 | 0.39 |
| 30 | 66% ± 5 | 66% ± 5 | -0 ± 7 pts | 4% | +1.18 ± 0.09 | +0.95 ± 0.06 | 0.46 |
| 40 | 65% ± 4 | 66% ± 4 | +1 ± 4 pts | 4% | +1.21 ± 0.09 | +0.98 ± 0.06 | 0.53 |

| h | material (≥0.5, certain) | point ≥0.5 | certain ≥0.25 | certain ≥1.0 | Δ_h p5 / p25 / p50 / p75 / p95 | states with \|D0\| < 0.5 that E_h decides (share of all states) |
|---|---|---|---|---|---|---|
| 5 | 66% | 66% | 80% | 36% | -1.67 / -0.72 / -0.00 / +0.85 / +2.24 | 6% |
| 10 | 70% | 70% | 82% | 45% | -2.04 / -0.89 / +0.01 / +0.92 / +2.54 | 7% |
| 20 | 65% | 65% | 73% | 38% | -1.85 / -0.76 / -0.03 / +0.75 / +1.97 | 5% |
| 30 | 66% | 66% | 69% | 38% | -1.84 / -0.78 / +0.01 / +0.77 / +2.07 | 5% |
| 40 | 65% | 67% | 66% | 41% | -1.92 / -0.80 / +0.00 / +0.81 / +2.08 | 5% |

## Stage 2 S2: which situations explain the adjustments (turn 94, folded to the category's X)

| category | n | share | X's D0 | Δ5 | Δ10 | Δ20 | Δ40 | material @10 | drift Δ40 − Δ10 | resolved by 10 / 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| dead | 49 | 9% | -3.72 ± 1.10 | -1.24 ± 0.51 | -0.19 ± 0.45 | -0.05 ± 0.36 | -0.18 ± 0.39 | 78% | +0.01 ± 0.13 | 83% / 97% |
| recall_away | 47 | 8% | -0.44 ± 1.31 | -0.05 ± 0.14 | +0.25 ± 0.16 | +0.25 ± 0.17 | +0.14 ± 0.21 | 32% | -0.11 ± 0.11 | 83% / 95% |
| both_away | 10 | 2% | +1.85 ± 3.00 | +0.50 ± 0.73 | +0.29 ± 0.84 | +0.23 ± 0.96 | +0.22 ± 1.05 | 70% | -0.08 ± 0.33 | 100% / 100% |
| fresh | 133 | 23% | -1.17 ± 0.71 | +0.99 ± 0.15 | +1.34 ± 0.20 | +1.16 ± 0.17 | +1.15 ± 0.18 | 80% | -0.19 ± 0.08 | 82% / 96% |
| due_asym | 92 | 16% | -0.63 ± 0.88 | +0.84 ± 0.21 | +0.97 ± 0.25 | +0.70 ± 0.20 | +0.74 ± 0.21 | 68% | -0.23 ± 0.11 | 67% / 92% |
| stock_asym | 10 | 2% | -2.43 ± 1.41 | +0.49 ± 0.42 | +0.62 ± 0.73 | +1.27 ± 0.56 | +1.10 ± 0.61 | 70% | +0.47 ± 0.26 | 59% / 90% |
| ordinary | 235 | 41% | -0.16 ± 0.50 | +0.04 ± 0.13 | -0.02 ± 0.15 | -0.06 ± 0.12 | -0.05 ± 0.13 | 72% | -0.04 ± 0.06 | 78% / 94% |

Folded event accounting over 10 turns (X − Y):

| category | farm | available | lost away | bounty | plates | returns | turns away |
|---|---|---|---|---|---|---|---|
| dead | +0.45 | +1.72 | +1.21 | +0.27 | -0.90 | +0.59 | +1.58 |
| recall_away | +0.28 | +0.40 | +0.07 | +0.04 | -0.07 | +0.72 | +0.36 |
| both_away | +0.19 | +0.37 | +0.15 | +0.01 | +0.09 | +0.01 | -0.49 |
| fresh | +1.19 | +1.11 | -0.23 | +0.14 | +0.00 | -0.34 | -0.90 |
| due_asym | +0.61 | +0.34 | -0.30 | +0.16 | +0.20 | -0.39 | -1.02 |
| stock_asym | +1.51 | +2.50 | +0.48 | -0.02 | -0.87 | +0.38 | +0.82 |
| ordinary | -0.04 | -0.06 | -0.02 | +0.01 | +0.02 | -0.00 | -0.02 |

Share of the variance of Δ̂_h across states explained: 5-fold cross-validated R² (in-sample in brackets); the rollout-noise share bounds what any model can explain:

| h | rollout-noise share | pending features | strength features | D0 | pending + D0 | all |
|---|---|---|---|---|---|---|
| 5 | 0.01 | 0.58 (0.60) | 0.44 (0.45) | 0.00 (0.01) | 0.58 (0.60) | 0.84 (0.85) |
| 10 | 0.01 | 0.57 (0.59) | 0.39 (0.40) | 0.02 (0.02) | 0.57 (0.59) | 0.85 (0.86) |
| 20 | 0.03 | 0.64 (0.65) | 0.29 (0.30) | 0.02 (0.02) | 0.65 (0.66) | 0.80 (0.81) |
| 30 | 0.04 | 0.61 (0.62) | 0.30 (0.31) | 0.01 (0.01) | 0.62 (0.63) | 0.79 (0.80) |
| 40 | 0.05 | 0.58 (0.60) | 0.29 (0.29) | 0.01 (0.01) | 0.59 (0.61) | 0.76 (0.77) |

Finer split (descriptive, not pre-registered; cross-validated R²): HP acts mostly through who has to reset next, so it is shown on its own:

| h | reset status | wave | HP | investment and level | reset + wave + HP |
|---|---|---|---|---|---|
| 5 | 0.48 | 0.19 | 0.38 | 0.22 | 0.84 |
| 10 | 0.43 | 0.34 | 0.28 | 0.25 | 0.85 |
| 20 | 0.39 | 0.48 | 0.17 | 0.23 | 0.80 |
| 30 | 0.38 | 0.42 | 0.20 | 0.22 | 0.77 |
| 40 | 0.37 | 0.40 | 0.19 | 0.21 | 0.74 |

Stage 1's case criteria among the natural turn-94 states (selected subsets, folded to X):

| case | n | Δ10 | Δ20 | material @10 |
|---|---|---|---|---|
| returner | 6 | +1.48 ± 0.60 | +1.49 ± 0.68 | 100% |
| lost | 9 | +1.21 ± 0.51 | +1.08 ± 0.42 | 89% |
| denied | 48 | -0.91 ± 0.36 | -0.87 ± 0.31 | 83% |
| stable | 6 | +0.18 ± 0.44 | +0.09 ± 0.39 | 17% |
| opposing | 101 | -1.46 ± 0.25 | -1.00 ± 0.23 | 88% |
| due | 168 | -0.36 ± 0.18 | -0.36 ± 0.15 | 68% |

## Stage 2 S3: stability

| windows | mean \|change\| | corr | class changes (all) | class changes among material | leader changes |
|---|---|---|---|---|---|
| 5->10 | 0.43 | 0.90 | 13% | 17% | 4% |
| 10->20 | 0.34 | 0.96 | 18% | 23% | 3% |
| 20->40 | 0.22 | 0.98 | 14% | 20% | 2% |

Continuation policies (the same states, common random numbers):

| alternative | n | h | alt − own | mean \|alt − own\| | corr | class changes among material | leader changes | states with a material, certain difference |
|---|---|---|---|---|---|---|---|---|
| alternative policy (other seed, same curve) | 144 | 5 | -0.01 ± 0.04 | 0.16 | 0.98 | 15% | 3% | 5% |
| alternative policy (other seed, same curve) | 144 | 10 | -0.03 ± 0.04 | 0.19 | 0.98 | 8% | 0% | 6% |
| alternative policy (other seed, same curve) | 144 | 20 | -0.04 ± 0.05 | 0.25 | 0.96 | 24% | 4% | 8% |
| alternative policy (other seed, same curve) | 144 | 40 | -0.03 ± 0.06 | 0.28 | 0.95 | 28% | 4% | 3% |
| other curve's package | 576 | 5 | -0.02 ± 0.02 | 0.17 | 0.98 | 11% | 1% | 5% |
| other curve's package | 576 | 10 | -0.01 ± 0.02 | 0.19 | 0.98 | 8% | 2% | 4% |
| other curve's package | 576 | 20 | +0.01 ± 0.03 | 0.24 | 0.97 | 21% | 2% | 7% |
| other curve's package | 576 | 40 | +0.00 ± 0.03 | 0.31 | 0.95 | 28% | 2% | 8% |

Anchors (material share at that anchor's K / SD of the true adjustment, which does not depend on K):

| t0 | n | K | lower level | h 5 | h 10 | h 20 | h 40 |
|---|---|---|---|---|---|---|---|
| 84 | 576 | 64 | 12.6 | 63% / 1.17 | 67% / 1.36 | 61% / 1.13 | 51% / 1.17 |
| 94 | 576 | 128 | 13.4 | 66% / 1.25 | 70% / 1.39 | 65% / 1.15 | 65% / 1.21 |
| 104 | 576 | 64 | 14.1 | 62% / 1.17 | 67% / 1.31 | 59% / 1.11 | 49% / 1.15 |
| 50 | 576 | 32 | 9.2 | 65% / 1.23 | 69% / 1.41 | 54% / 1.20 | 44% / 1.22 |

New consequences after h (resets, deaths and Recalls in turns h+1..40, both seats, per continuation): h 5: returns 3.51, deaths 0.47; h 10: returns 3.02, deaths 0.40; h 20: returns 2.01, deaths 0.26; h 30: returns 0.99, deaths 0.12; h 40: returns 0.00, deaths 0.00

## Default window rule (protocol §6)

| h | (a) coverage ≥ 80% | by kind (absent / due / stock) | (b) excess drift ≤ 0.25 | (c) median ± ≤ 0.5 |
|---|---|---|---|---|
| 5 | 54% (not met) | resA 100%, resD 27%, resS 83% | fails: dead, recall_away, fresh; dead +1.15, recall_away +0.28, fresh +0.25, due_asym -0.00 | 0.14 (met) |
| 10 | 78% (not met) | resA 100%, resD 60%, resS 100% | met; dead +0.04, recall_away -0.07, fresh -0.15, due_asym -0.19 | 0.24 (met) |
| 20 | 95% (met) | resA 100%, resD 90%, resS 100% | met; dead -0.15, recall_away -0.13, fresh -0.02, due_asym +0.03 | 0.39 (met) |

**Result: E_20.** Largest share of material classifications changed between policies or windows 10 → 20: 24%.

## Stage 2 S4: baseline and C1 (six training seeds per curve; C1 − baseline paired by seed)

(a) State distributions at turn 94 (each curve's own games):

| quantity | baseline | C1 | C1 − baseline |
|---|---|---|---|
| abs_D0 | +3.29 ± 0.21 | +3.64 ± 0.46 | +0.35 ± 0.40 |
| abs_E5 | +3.36 ± 0.20 | +3.65 ± 0.43 | +0.29 ± 0.37 |
| abs_E10 | +3.30 ± 0.20 | +3.60 ± 0.45 | +0.30 ± 0.41 |
| abs_E20 | +3.27 ± 0.20 | +3.58 ± 0.45 | +0.31 ± 0.46 |
| abs_E40 | +3.32 ± 0.14 | +3.62 ± 0.41 | +0.30 ± 0.43 |
| share_dead | 9% ± 5 | 8% ± 4 | -1 ± 9 pts |
| share_recall_away | 7% ± 3 | 9% ± 4 | +2 ± 7 pts |
| share_both_away | 2% ± 2 | 2% ± 2 | +0 ± 1 pts |
| share_fresh | 24% ± 8 | 23% ± 6 | -1 ± 9 pts |
| share_due_asym | 16% ± 2 | 16% ± 5 | +0 ± 7 pts |
| share_stock_asym | 2% ± 1 | 2% ± 3 | +0 ± 2 pts |
| share_ordinary | 41% ± 7 | 41% ± 10 | +0 ± 13 pts |
| abs_xp_diff | +13.01 ± 1.27 | +12.90 ± 2.27 | -0.11 ± 3.03 |
| abs_hp_diff | +0.29 ± 0.03 | +0.29 ± 0.04 | -0.00 ± 0.06 |
| level_min | +13.31 ± 0.08 | +13.41 ± 0.15 | +0.11 ± 0.09 |
| away_share | 10% ± 5 | 10% ± 3 | +0 ± 6 pts |

(b) Lead conversion under each package's own continuation (slope of Δ_h on D0):

| h | baseline | C1 | C1 − baseline |
|---|---|---|---|
| 5 | -0.02 ± 0.02 | -0.02 ± 0.06 | -0.00 ± 0.06 |
| 10 | -0.05 ± 0.02 | -0.05 ± 0.06 | +0.00 ± 0.06 |
| 20 | -0.04 ± 0.02 | -0.04 ± 0.05 | +0.00 ± 0.06 |
| 30 | -0.03 ± 0.02 | -0.02 ± 0.05 | +0.01 ± 0.05 |
| 40 | -0.03 ± 0.02 | -0.03 ± 0.05 | +0.01 ± 0.05 |

(c) Matched states: the same turn-94 states continued by the C1 package and by the baseline package (environment and its trained policy; not the isolated effect of the investment curve):

| h | leader's extra gain under C1 (seeds) | slope difference (seeds) | mean difference | leader gain, baseline-origin states | leader gain, C1-origin states |
|---|---|---|---|---|---|
| 5 | +0.01 ± 0.04 | -0.00 ± 0.01 | +0.01 ± 0.02 | +0.00 ± 0.03 | +0.01 ± 0.03 |
| 10 | -0.02 ± 0.03 | -0.00 ± 0.01 | -0.01 ± 0.02 | -0.03 ± 0.03 | -0.02 ± 0.03 |
| 20 | +0.02 ± 0.04 | +0.01 ± 0.01 | -0.01 ± 0.03 | +0.02 ± 0.03 | +0.02 ± 0.04 |
| 30 | +0.05 ± 0.05 | +0.01 ± 0.01 | -0.02 ± 0.03 | +0.04 ± 0.04 | +0.06 ± 0.04 |
| 40 | +0.04 ± 0.07 | +0.02 ± 0.02 | -0.01 ± 0.06 | +0.05 ± 0.05 | +0.04 ± 0.04 |

(d) Earlier conclusions re-checked (M4.7 §2):

| quantity | baseline | C1 | C1 − baseline |
|---|---|---|---|
| \|lead\| at turn 94, raw D0 (M4.7: 3.29 / 3.64 / +0.35 ± 0.40) | +3.29 ± 0.21 | +3.64 ± 0.46 | +0.35 ± 0.40 |
| \|E_5\| at turn 94 | +3.36 ± 0.20 | +3.65 ± 0.43 | +0.29 ± 0.37 |
| \|E_10\| at turn 94 | +3.30 ± 0.20 | +3.60 ± 0.45 | +0.30 ± 0.41 |
| \|E_20\| at turn 94 | +3.27 ± 0.20 | +3.58 ± 0.45 | +0.31 ± 0.46 |
| conversion slope 50 → 94, raw D0 (M4.7: −0.20 / +0.04 / +0.24 ± 0.15) | -0.20 ± 0.07 | +0.04 ± 0.17 | +0.24 ± 0.15 |
| conversion slope 50 → 94 with E_5 at both ends | -0.11 ± 0.09 | +0.07 ± 0.10 | +0.18 ± 0.11 |
| conversion slope 50 → 94 with E_10 at both ends | -0.09 ± 0.11 | +0.13 ± 0.11 | +0.22 ± 0.15 |
| conversion slope 50 → 94 with E_20 at both ends | -0.10 ± 0.10 | +0.15 ± 0.09 | +0.25 ± 0.13 |

(d) M4.7's pilot primary P1 (the short arm's lane lead against the control, head to head; M4.7: +0.26 ± 0.17 and −0.09 ± 0.17) with the lane-exit readout at turn 94 (supplementary, protocol amendment; K 8, 20-turn continuations by the same pair):

| seed | games | raw P1 (lead at 94) | P1 with E5 | with E10 | with E20 | E10 − raw | replay deviation |
|---|---|---|---|---|---|---|---|
| 0 | 1024 | +0.26 ± 0.17 | +0.19 ± 0.17 | +0.18 ± 0.17 | +0.17 ± 0.18 | -0.08 ± 0.06 | 1e-07 |
| 1 | 1024 | -0.09 ± 0.17 | -0.10 ± 0.17 | -0.11 ± 0.17 | -0.14 ± 0.17 | -0.02 ± 0.06 | 1e-07 |

Controlled: E_h(A_fresh) − E_h(orig) on the turn-94 states where A is due (n baseline 81, C1 87):

| h | baseline | C1 | C1 − baseline |
|---|---|---|---|
| 5 | +0.42 ± 0.04 | +0.47 ± 0.31 | +0.05 ± 0.31 |
| 10 | +0.71 ± 0.18 | +0.77 ± 0.30 | +0.06 ± 0.37 |
| 20 | +0.56 ± 0.17 | +0.62 ± 0.07 | +0.06 ± 0.20 |
| 40 | +0.53 ± 0.19 | +0.61 ± 0.11 | +0.08 ± 0.22 |

## Method classification (protocol §7)

| criterion | result |
|---|---|
| V1–V3 exact | True |
| calibration | h 20: slope interval [0.80, 1.17], residual -0.14 → True |
| added information | h 20: material 65%, true SD 1.16 → True |
| median half-widths | h 5: 0.14, h 10: 0.24, h 20: 0.39 |
| **label** | **useful** |

_the Stage 1 mechanism check (adjustments accounted for by recorded events matching the cases) is judged in the report from validation.json; it does not change this label automatically_
