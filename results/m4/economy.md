# S4 — Economic sanity (M4.2)

Commit `1a40cf5069`. Paired counterfactual rollouts (60 per variant, common random numbers), discounted over W = 1000 turns (guaranteed truncation error per value ≤ 0.0030). Scripted continuations: approximate champion-game measurements, not equilibrium values. Command: `python.exe C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\experiments\m4\economy.py --rollouts 60 --window 1000 --rr-episodes 60 --procs 10`.

## A — item value (focal A): free tier (N + 1 vs N, same gold) and buying now vs at the next return

| comparison | paired diff | 95% CI | P(diff > 0) | significant | value 1 | value 2 |
|---|---|---|---|---|---|---|
| free|0 | -0.444 | ±0.699 | 0.40 | no | -3.23 | -2.78 |
| now|0 | +0.000 | ±0.000 | 0.00 | no | -3.31 | -3.31 |
| free|4 | +0.029 | ±0.757 | 0.45 | no | -0.04 | -0.07 |
| now|4 | -0.257 | ±0.788 | 0.45 | no | +0.04 | +0.29 |
| free|8 | -0.210 | ±0.581 | 0.40 | no | +0.40 | +0.61 |
| now|8 | -0.027 | ±0.356 | 0.03 | no | +0.50 | +0.53 |
| free|9 | +0.079 | ±0.425 | 0.37 | no | +0.75 | +0.67 |
| now|9 | +0.000 | ±0.000 | 0.00 | no | +0.62 | +0.62 |

## B — Recall now minus Hold now (focal A), by state | HP in sixths | affordable tier

| comparison | paired diff | 95% CI | P(diff > 0) | significant | value 1 | value 2 |
|---|---|---|---|---|---|---|
| cheap|2|afford | +0.175 | ±0.687 | 0.48 | no | -12.99 | -13.17 |
| cheap|2|poor | +0.866 | ±0.710 | 0.62 | yes | -13.05 | -13.92 |
| cheap|4|afford | -0.069 | ±0.640 | 0.53 | no | -13.26 | -13.19 |
| cheap|4|poor | +0.307 | ±0.806 | 0.52 | no | -13.07 | -13.37 |
| cheap|6|afford | +0.010 | ±0.733 | 0.50 | no | -12.67 | -12.68 |
| cheap|6|poor | +0.133 | ±0.683 | 0.52 | no | -13.05 | -13.19 |
| expensive|2|afford | -0.176 | ±0.000 | 0.00 | yes | +11.79 | +11.96 |
| expensive|2|poor | -0.184 | ±0.237 | 0.02 | no | +12.02 | +12.20 |
| expensive|4|afford | -0.163 | ±0.088 | 0.02 | yes | +12.77 | +12.93 |
| expensive|4|poor | -0.258 | ±0.148 | 0.02 | yes | +12.46 | +12.72 |
| expensive|6|afford | -0.190 | ±0.024 | 0.00 | yes | +13.20 | +13.39 |
| expensive|6|poor | -0.159 | ±0.036 | 0.02 | yes | +13.39 | +13.54 |
| play0|2|afford | +0.077 | ±0.802 | 0.57 | no | -0.18 | -0.26 |
| play0|2|poor | +0.874 | ±0.731 | 0.68 | yes | +0.92 | +0.05 |
| play0|4|afford | -0.938 | ±0.723 | 0.38 | yes | -0.86 | +0.08 |
| play0|4|poor | -1.332 | ±0.756 | 0.33 | yes | -1.19 | +0.14 |
| play0|6|afford | -0.709 | ±0.831 | 0.35 | no | -0.05 | +0.66 |
| play0|6|poor | -1.358 | ±0.715 | 0.27 | yes | -1.27 | +0.09 |
| play1|2|afford | +0.557 | ±0.352 | 0.78 | yes | +0.95 | +0.39 |
| play1|2|poor | +0.325 | ±0.407 | 0.70 | no | +1.10 | +0.78 |
| play1|4|afford | -0.110 | ±0.534 | 0.40 | no | +0.23 | +0.34 |
| play1|4|poor | -0.140 | ±0.544 | 0.45 | no | -0.04 | +0.10 |
| play1|6|afford | -0.847 | ±0.432 | 0.22 | yes | -0.23 | +0.61 |
| play1|6|poor | -0.959 | ±0.408 | 0.23 | yes | -0.16 | +0.80 |
| play2|2|afford | -0.755 | ±0.472 | 0.08 | yes | -0.63 | +0.13 |
| play2|2|poor | -0.880 | ±0.558 | 0.12 | yes | -1.16 | -0.28 |
| play2|4|afford | -0.394 | ±0.774 | 0.47 | no | -0.46 | -0.07 |
| play2|4|poor | -1.363 | ±0.806 | 0.38 | yes | -0.25 | +1.12 |
| play2|6|afford | -1.467 | ±0.747 | 0.22 | yes | -0.71 | +0.75 |
| play2|6|poor | -1.347 | ±0.491 | 0.22 | yes | -0.02 | +1.33 |
| play3|2|afford | -1.143 | ±0.530 | 0.03 | yes | -0.21 | +0.93 |
| play3|2|poor | -1.256 | ±0.508 | 0.07 | yes | -0.87 | +0.38 |
| play3|4|afford | +0.331 | ±0.650 | 0.58 | no | +0.05 | -0.28 |
| play3|4|poor | -0.446 | ±0.649 | 0.43 | no | -0.39 | +0.06 |
| play3|6|afford | -0.570 | ±0.834 | 0.38 | no | +0.40 | +0.97 |
| play3|6|poor | -0.680 | ±0.663 | 0.38 | yes | -0.31 | +0.37 |

Recall sign pattern: significantly positive in 3 cells, significantly negative in 18 cells → **both signs present**.

## C — the opponent under a held freeze: forward (+1, Hold) minus back (−1, Safe), focal B

| comparison | paired diff | 95% CI | P(diff > 0) | significant | value 1 | value 2 |
|---|---|---|---|---|---|---|
| freeze_hold | +22.019 | ±0.550 | 1.00 | yes | -20.56 | -42.58 |
| freeze_trade | +19.544 | ±0.403 | 1.00 | yes | -21.87 | -41.42 |

## D — Recall versus death as the route to the shop (focal A; death includes the 0.5 bounty)

| comparison | paired diff | 95% CI | P(diff > 0) | significant | value 1 | value 2 |
|---|---|---|---|---|---|---|
| cheap|from -1 | +1.212 | ±0.780 | 0.67 | yes | -13.73 | -14.94 |
| cheap|from +1 | +1.206 | ±0.828 | 0.60 | yes | -13.73 | -14.94 |
| expensive|from -1 | +0.904 | ±0.815 | 0.55 | yes | +11.85 | +10.95 |
| expensive|from +1 | +0.223 | ±0.581 | 0.37 | no | +11.17 | +10.95 |
| play0|from -1 | +1.177 | ±0.729 | 0.72 | yes | -0.07 | -1.25 |
| play0|from +1 | -0.745 | ±0.757 | 0.57 | no | -1.99 | -1.25 |
| play3|from -1 | -0.851 | ±0.645 | 0.23 | yes | -1.54 | -0.69 |
| play3|from +1 | -0.931 | ±0.628 | 0.33 | yes | -1.62 | -0.69 |

## E — denial by a strong freezer (+2 levels, +1 tier) minus an equal one, focal B (xp_keeper)

| comparison | paired diff | 95% CI | P(diff > 0) | significant | value 1 | value 2 |
|---|---|---|---|---|---|---|
| strong_minus_equal | +0.008 | ±0.101 | 0.22 | no | -39.68 | -39.68 |

### Item diagnosis: a free tier under continuations of increasing combat intensity (400 turns)

| continuation | free-tier value (95% CI) | deaths A, B without | deaths A, B with | A trades without / with | turns per game with unequal items | deaths: with gank damage / trade damage only |
|---|---|---|---|---|---|---|
| edge_trader vs edge_trader | +0.037 ± 0.463 | 18.2, 17.7 | 18.3, 18.2 | 80 / 87 | 15.4 | 232 / 801 |
| aggressive_trader vs aggressive_trader | +0.168 ± 0.226 | 33.4, 33.6 | 33.3, 33.6 | 155 / 155 | 16.3 | 721 / 584 |
| aggressive_trader vs hold_farmer | +0.234 ± 0.287 | 10.7, 25.0 | 10.7, 25.1 | 232 / 231 | 21.0 | 397 / 673 |

## F — plates

| pairing (A vs B) | plate gold per game (A) | share earned while A away | plates per game | tower damage per game (centi) |
|---|---|---|---|---|
| presser vs hold_farmer | 2.65 | 0% | 2.65 | 5570 |
| g_stacker vs hold_farmer | 0.00 | - | 0.00 | 0 |
| presser vs gold_recaller | 2.65 | 0% | 2.65 | 5570 |
| freeze_breaker vs gold_recaller | 0.28 | 0% | 0.28 | 1167 |
| g_adaptive_breaker vs freezer | 0.00 | - | 0.00 | 0 |

## G — snowballing (mirror edge traders)

- P(final discounted margin > 0 | ahead after 30 turns) = 0.67; | behind = 0.27; correlation 0.59 (300 games, W = 1000)
- mean |level gap| at turns 30/60/100/150: 0.29, 0.18, 0.54, 0.55; share with a gap ≥ 2: 0%, 1%, 2%, 1%

## H — scripted round robin (E1 from s0, W = 1000, both seats, CRN): payoff to the row

| row \ column | hold_farmer | aggressive_trader | freezer | freeze_breaker | xp_keeper | gold_recaller | presser | edge_trader | mean |
|---|---|---|---|---|---|---|---|---|---|
| hold_farmer |  | -2.21 | -0.31 | +17.86 | +0.00 | +0.72 | +17.60 | -0.31 | +4.76 |
| aggressive_trader | +2.21 |  | -3.04 | +8.79 | +2.21 | +1.71 | +7.33 | -3.04 | +2.31 |
| freezer | +0.31 | +3.04 |  | +13.51 | +0.31 | +0.49 | +13.05 | +0.00 | +4.39 |
| freeze_breaker | -17.86 | -8.79 | -13.51 |  | -17.86 | -17.59 | +0.00 | -13.33 | -12.71 |
| xp_keeper | -0.00 | -2.21 | -0.31 | +17.86 |  | +0.72 | +17.60 | -0.31 | +4.76 |
| gold_recaller | -0.72 | -1.71 | -0.49 | +17.59 | -0.72 |  | +17.32 | -0.49 | +4.40 |
| presser | -17.60 | -7.33 | -13.05 | -0.00 | -17.60 | -17.32 |  | -12.88 | -12.25 |
| edge_trader | +0.31 | +3.04 | -0.00 | +13.33 | +0.31 | +0.49 | +12.88 |  | +4.34 |

Dominant scripted strategy (beats every other): none. Guaranteed truncation bound per value 0.0030.
