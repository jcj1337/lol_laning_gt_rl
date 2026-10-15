# M4.6a generated tables

Interpretation: docs/m4/m4_6a_report.md.

## E1 — exact one-turn ledger

### Freeze cells: does a walk-up Shove break the freeze this turn?

| level | gap | held HP | P(freeze holds) vs keep | P(holds) vs guard | B HP lost vs keep (bar) | B HP lost vs guard | A HP lost guarding | of which positional | P(B dies) vs guard | B: break − walk up vs guard (priced 1.5) | B: break − step back vs guard | A: guard − keep vs break |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | equal | 5 | 1.00 | 1.00 | 0.048 | 0.167 | 0.117 | 0.051 | 0.000 | +0.44 | +0.53 | +0.19 |
| 4 | equal | 12 | 0.00 | 1.00 | 0.044 | 0.152 | 0.225 | 0.114 | 0.000 | +0.44 | +0.61 | +0.08 |
| 4 | equal | 20 | 0.00 | 1.00 | 0.040 | 0.137 | 0.332 | 0.188 | 0.000 | +0.44 | +0.69 | -0.10 |
| 4 | x1.76 | 5 | 1.00 | 1.00 | 0.052 | 0.262 | 0.071 | 0.051 | 0.011 | +0.43 | +0.39 | +0.41 |
| 4 | x1.76 | 12 | 0.00 | 1.00 | 0.048 | 0.238 | 0.139 | 0.114 | 0.009 | +0.43 | +0.45 | +0.32 |
| 4 | x1.76 | 20 | 0.00 | 1.00 | 0.043 | 0.215 | 0.205 | 0.188 | 0.008 | +0.43 | +0.50 | +0.19 |
| 10 | equal | 5 | 0.00 | 0.00 | 0.033 | 0.152 | 0.043 | 0.024 | 0.000 | +0.78 | +0.84 | +0.07 |
| 10 | equal | 12 | 0.00 | 0.00 | 0.031 | 0.139 | 0.079 | 0.054 | 0.000 | +0.78 | +0.87 | -0.09 |
| 10 | equal | 20 | 0.00 | 0.00 | 0.028 | 0.125 | 0.115 | 0.089 | 0.000 | +0.78 | +0.89 | -0.13 |
| 10 | x1.76 | 5 | 0.00 | 1.00 | 0.052 | 0.262 | 0.038 | 0.024 | 0.011 | +0.55 | +0.48 | +0.46 |
| 10 | x1.76 | 12 | 0.00 | 1.00 | 0.048 | 0.238 | 0.069 | 0.054 | 0.009 | +0.55 | +0.51 | +0.27 |
| 10 | x1.76 | 20 | 0.00 | 1.00 | 0.043 | 0.215 | 0.102 | 0.089 | 0.008 | +0.55 | +0.54 | +0.19 |
| 15 | equal | 5 | 0.00 | 0.00 | 0.028 | 0.147 | 0.023 | 0.013 | 0.000 | +0.78 | +0.84 | +0.10 |
| 15 | equal | 12 | 0.00 | 0.00 | 0.026 | 0.134 | 0.032 | 0.021 | 0.000 | +0.78 | +0.84 | -0.03 |
| 15 | equal | 20 | 0.00 | 0.00 | 0.024 | 0.121 | 0.049 | 0.039 | 0.000 | +0.78 | +0.85 | -0.07 |
| 15 | x1.76 | 5 | 0.00 | 0.00 | 0.050 | 0.259 | 0.023 | 0.013 | 0.011 | +0.77 | +0.70 | +0.26 |
| 15 | x1.76 | 12 | 0.00 | 0.00 | 0.046 | 0.236 | 0.032 | 0.021 | 0.009 | +0.77 | +0.71 | +0.08 |
| 15 | x1.76 | 20 | 0.00 | 0.00 | 0.042 | 0.213 | 0.049 | 0.039 | 0.008 | +0.77 | +0.72 | +0.02 |


### Zoning cells: the one-turn threat game (HP priced at 1.5 resources per bar)

| situation | level | gap | A: Trade − Hold vs Hold | vs Safe | vs Yield | B: Safe − Hold vs Trade | B: Safe − Hold vs Hold | A dealt (bar) | A taken (bar) | A farm: Trade − Hold |
|---|---|---|---|---|---|---|---|---|---|---|
| neutral | 6 | equal | +0.18 | +0.14 | +0.18 | -0.25 | -0.29 | 0.152 | 0.080 | -0.090 |
| neutral | 6 | x1.76 | +0.35 | +0.14 | +0.35 | -0.08 | -0.28 | 0.210 | 0.058 | -0.090 |
| neutral | 10 | equal | +0.23 | +0.14 | +0.23 | -0.21 | -0.29 | 0.185 | 0.048 | -0.090 |
| neutral | 10 | x1.76 | +0.37 | +0.14 | +0.37 | -0.05 | -0.28 | 0.210 | 0.042 | -0.090 |
| a_push | 6 | equal | +0.08 | +0.06 | -0.12 | -0.12 | -0.14 | 0.166 | 0.165 | -0.112 |
| a_push | 6 | x1.76 | +0.29 | +0.07 | +0.08 | +0.08 | -0.14 | 0.229 | 0.119 | -0.112 |
| a_push | 10 | equal | +0.15 | +0.07 | -0.04 | -0.06 | -0.14 | 0.202 | 0.115 | -0.112 |
| a_push | 10 | x1.76 | +0.32 | +0.07 | +0.10 | +0.11 | -0.14 | 0.229 | 0.101 | -0.112 |
| b_push | 6 | equal | +0.19 | -0.10 | +0.19 | -0.25 | -0.54 | 0.138 | 0.082 | -0.090 |
| b_push | 6 | x1.76 | +0.36 | -0.07 | +0.36 | -0.12 | -0.54 | 0.191 | 0.060 | -0.090 |
| b_push | 10 | equal | +0.24 | -0.09 | +0.24 | -0.22 | -0.55 | 0.168 | 0.045 | -0.090 |
| b_push | 10 | x1.76 | +0.39 | -0.06 | +0.39 | -0.09 | -0.54 | 0.191 | 0.039 | -0.090 |


At E5's measured price (1.48 per bar):

| situation | level | gap | A: Trade − Hold vs Hold | vs Safe | B: Safe − Hold vs Trade |
|---|---|---|---|---|---|
| neutral | 6 | equal | +0.18 | +0.14 | -0.25 |
| neutral | 6 | x1.76 | +0.35 | +0.14 | -0.08 |
| neutral | 10 | equal | +0.22 | +0.14 | -0.21 |
| neutral | 10 | x1.76 | +0.37 | +0.14 | -0.06 |
| a_push | 6 | equal | +0.08 | +0.06 | -0.12 |
| a_push | 6 | x1.76 | +0.29 | +0.07 | +0.08 |
| a_push | 10 | equal | +0.15 | +0.07 | -0.06 |
| a_push | 10 | x1.76 | +0.31 | +0.07 | +0.11 |
| b_push | 6 | equal | +0.19 | -0.10 | -0.25 |
| b_push | 6 | x1.76 | +0.35 | -0.07 | -0.12 |
| b_push | 10 | equal | +0.24 | -0.09 | -0.22 |
| b_push | 10 | x1.76 | +0.38 | -0.06 | -0.10 |


### Decomposition (one cost removed at a time; counterfactual configurations, not candidates)

| price | removed cost | guard − keep: sign changes | mean (baseline) | mean (removed) | Trade − Hold: sign changes | mean (baseline) | mean (removed) | load-bearing |
|---|---|---|---|---|---|---|---|---|
| 1 | no_farm_cost | 6/36 | +0.24 | +0.35 | 0/12 | +0.21 | +0.31 | no |
| 1 | no_wave_cap | 0/36 | +0.24 | +0.24 | 0/12 | +0.21 | +0.21 | no |
| 1 | no_retaliation | 0/36 | +0.24 | +0.24 | 2/12 | +0.21 | +0.05 | no |
| 1 | no_positional | 1/36 | +0.24 | +0.14 | 0/12 | +0.21 | +0.21 | no |
| 1 | no_trade_gank | 1/36 | +0.24 | +0.24 | 0/12 | +0.21 | +0.22 | no |
| 1.5 | no_farm_cost | 4/36 | +0.26 | +0.37 | 0/12 | +0.26 | +0.36 | no |
| 1.5 | no_wave_cap | 0/36 | +0.26 | +0.26 | 0/12 | +0.26 | +0.26 | no |
| 1.5 | no_retaliation | 0/36 | +0.26 | +0.26 | 2/12 | +0.26 | +0.12 | no |
| 1.5 | no_positional | 0/36 | +0.26 | +0.19 | 0/12 | +0.26 | +0.26 | no |
| 1.5 | no_trade_gank | 0/36 | +0.26 | +0.27 | 0/12 | +0.26 | +0.27 | no |
| 2 | no_farm_cost | 2/36 | +0.29 | +0.40 | 0/12 | +0.31 | +0.41 | no |
| 2 | no_wave_cap | 0/36 | +0.29 | +0.29 | 0/12 | +0.31 | +0.31 | no |
| 2 | no_retaliation | 0/36 | +0.29 | +0.29 | 0/12 | +0.31 | +0.19 | no |
| 2 | no_positional | 0/36 | +0.29 | +0.23 | 0/12 | +0.31 | +0.31 | no |
| 2 | no_trade_gank | 0/36 | +0.29 | +0.30 | 0/12 | +0.31 | +0.32 | no |
| 1.47713 | no_farm_cost | 4/36 | +0.26 | +0.37 | 0/12 | +0.26 | +0.36 | no |
| 1.47713 | no_wave_cap | 0/36 | +0.26 | +0.26 | 0/12 | +0.26 | +0.26 | no |
| 1.47713 | no_retaliation | 0/36 | +0.26 | +0.26 | 2/12 | +0.26 | +0.12 | no |
| 1.47713 | no_positional | 0/36 | +0.26 | +0.18 | 0/12 | +0.26 | +0.26 | no |
| 1.47713 | no_trade_gank | 0/36 | +0.26 | +0.27 | 0/12 | +0.26 | +0.27 | no |

## E3 — power-gap self-play (stronger minus weaker; premium = gap minus the equal-power game of the same seeds; t intervals over the six finals; levels 6 / 10 / 14 pooled unless stated)

### E3-P1: the power premium

| gap | G20 | G50 | G100 | G300 | lead @100 (undiscounted) | lead @300 |
|---|---|---|---|---|---|---|
| x1.24 | +0.66 ± 0.18 | +1.27 ± 0.30 | +1.72 ± 0.32 | +1.78 ± 0.28 | +2.45 ± 0.46 | +2.74 ± 0.55 |
| x1.5 | +1.41 ± 0.28 | +2.62 ± 0.41 | +3.32 ± 0.49 | +3.42 ± 0.47 | +4.64 ± 0.71 | +4.93 ± 0.75 |
| x1.76 | +2.03 ± 0.27 | +3.64 ± 0.50 | +4.54 ± 0.51 | +4.71 ± 0.45 | +6.23 ± 0.67 | +6.91 ± 0.54 |
| x2.2 | +3.43 ± 0.51 | +5.79 ± 0.68 | +7.28 ± 0.70 | +7.96 ± 0.75 | +10.00 ± 0.89 | +13.21 ± 1.18 |


By level (G100 and lead at 100):

| gap | level | G100 premium | lead @100 premium |
|---|---|---|---|
| x1.24 | 6 | +1.30 ± 0.47 | +1.79 ± 0.74 |
| x1.24 | 10 | +1.96 ± 0.62 | +2.99 ± 0.91 |
| x1.24 | 14 | +1.92 ± 0.77 | +2.57 ± 1.09 |
| x1.5 | 6 | +2.82 ± 0.39 | +3.95 ± 0.62 |
| x1.5 | 10 | +3.35 ± 0.71 | +4.71 ± 1.16 |
| x1.5 | 14 | +3.79 ± 0.80 | +5.26 ± 1.09 |
| x1.76 | 6 | +4.13 ± 0.41 | +5.55 ± 0.60 |
| x1.76 | 10 | +4.72 ± 0.82 | +6.61 ± 1.10 |
| x1.76 | 14 | +4.78 ± 0.95 | +6.54 ± 1.27 |
| x2.2 | 6 | +6.96 ± 0.41 | +9.68 ± 0.64 |
| x2.2 | 10 | +7.52 ± 1.00 | +10.37 ± 1.33 |
| x2.2 | 14 | +7.37 ± 1.08 | +9.96 ± 1.43 |


### E3-P2: decomposition of the premium at 100 turns (resources; positive = in the stronger side's favour)

| gap | lead | stronger: farm | stronger: bounty | stronger: plates | weaker: minion flow lost | weaker: lost while present | weaker: lost while away | weaker: bounty (−) | weaker: plates (−) | Δ weaker turns away | Δ weaker deaths | Δ weaker Recalls | Δ weaker XP lost | Δ stronger deaths |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x1.24 | +2.45 ± 0.46 | +0.33 ± 0.24 | +0.46 ± 0.11 | +0.40 ± 0.12 | +0.03 ± 0.08 | +0.20 ± 0.07 | +0.50 ± 0.05 | +0.16 ± 0.13 | +0.39 ± 0.08 | +0.66 ± 0.14 | +0.23 ± 0.05 | -0.20 ± 0.10 | +3.85 ± 0.47 | -0.08 ± 0.06 |
| x1.5 | +4.64 ± 0.71 | +0.62 ± 0.19 | +0.97 ± 0.22 | +0.78 ± 0.09 | +0.06 ± 0.07 | +0.34 ± 0.10 | +0.94 ± 0.19 | +0.36 ± 0.11 | +0.56 ± 0.09 | +1.15 ± 0.38 | +0.48 ± 0.11 | -0.49 ± 0.06 | +7.21 ± 1.64 | -0.18 ± 0.05 |
| x1.76 | +6.23 ± 0.67 | +0.71 ± 0.36 | +1.30 ± 0.24 | +1.06 ± 0.09 | +0.05 ± 0.08 | +0.55 ± 0.20 | +1.31 ± 0.22 | +0.52 ± 0.18 | +0.73 ± 0.14 | +1.79 ± 0.47 | +0.65 ± 0.12 | -0.56 ± 0.14 | +10.37 ± 1.46 | -0.26 ± 0.09 |
| x2.2 | +10.00 ± 0.89 | +1.34 ± 0.54 | +2.27 ± 0.21 | +1.48 ± 0.18 | +0.12 ± 0.03 | +0.59 ± 0.23 | +2.14 ± 0.23 | +1.01 ± 0.30 | +1.05 ± 0.20 | +2.82 ± 0.51 | +1.13 ± 0.11 | -1.11 ± 0.16 | +16.17 ± 1.57 | -0.51 ± 0.15 |


### E3-P2: decomposition of the premium at 300 turns (resources; positive = in the stronger side's favour)

| gap | lead | stronger: farm | stronger: bounty | stronger: plates | weaker: minion flow lost | weaker: lost while present | weaker: lost while away | weaker: bounty (−) | weaker: plates (−) | Δ weaker turns away | Δ weaker deaths | Δ weaker Recalls | Δ weaker XP lost | Δ stronger deaths |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x1.24 | +2.74 ± 0.55 | +0.62 ± 0.21 | +0.66 ± 0.21 | +0.07 ± 0.07 | +0.00 ± 0.05 | +0.24 ± 0.16 | +0.68 ± 0.20 | +0.37 ± 0.17 | +0.10 ± 0.03 | +0.88 ± 0.32 | +0.33 ± 0.10 | -0.33 ± 0.18 | +4.62 ± 1.72 | -0.18 ± 0.08 |
| x1.5 | +4.93 ± 0.75 | +0.93 ± 0.37 | +1.27 ± 0.33 | +0.15 ± 0.09 | +0.02 ± 0.06 | +0.43 ± 0.19 | +1.19 ± 0.25 | +0.76 ± 0.22 | +0.19 ± 0.05 | +1.54 ± 0.40 | +0.63 ± 0.17 | -0.69 ± 0.29 | +8.68 ± 1.94 | -0.38 ± 0.11 |
| x1.76 | +6.91 ± 0.54 | +1.10 ± 0.61 | +1.80 ± 0.32 | +0.19 ± 0.10 | +0.01 ± 0.04 | +0.77 ± 0.39 | +1.80 ± 0.23 | +0.94 ± 0.31 | +0.29 ± 0.07 | +2.47 ± 0.39 | +0.90 ± 0.16 | -0.87 ± 0.28 | +13.45 ± 2.36 | -0.47 ± 0.15 |
| x2.2 | +13.21 ± 1.18 | +2.21 ± 0.74 | +3.60 ± 0.30 | +0.24 ± 0.14 | +0.02 ± 0.03 | +0.99 ± 0.49 | +3.54 ± 0.33 | +1.93 ± 0.46 | +0.68 ± 0.14 | +4.69 ± 0.88 | +1.80 ± 0.15 | -1.87 ± 0.38 | +26.52 ± 2.62 | -0.96 ± 0.23 |


### E3-P3: behaviour while the gap lasts (power ratio ≥ 1.25, both present; means over finals and levels)

| gap | stronger Trade rate | connected rate | stronger cell (lane cells only) | stronger at W | weaker cell (lane only) | weaker at W | weaker Yield | weaker Safe | weaker at own tower | weaker catchable share | weaker farm lost: catchable − not | turns per cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x1.24 | 0.318 | 0.199 | -0.226 | 0.015 | -1.042 | 0.009 | 0.257 | 0.232 | 0.411 | 0.583 | -0.052 | 706 |
| x1.5 | 0.288 | 0.188 | -0.290 | 0.018 | -0.922 | 0.008 | 0.279 | 0.187 | 0.357 | 0.637 | -0.039 | 3884 |
| x1.76 | 0.285 | 0.182 | -0.285 | 0.021 | -0.972 | 0.007 | 0.263 | 0.201 | 0.393 | 0.600 | -0.035 | 6087 |
| x2.2 | 0.293 | 0.177 | -0.359 | 0.030 | -0.989 | 0.008 | 0.239 | 0.235 | 0.414 | 0.578 | -0.037 | 12933 |


M4.5c stage re-run on new seeds (viewer neutral, level 3, A ×1.76; M4.5c: Trade rate 0.26, cell +2.03, at tower 0.38, Yield 0.30, denial 0.014):

| final | stronger Trade rate | stronger cell | weaker at tower | weaker Yield | denial when near |
|---|---|---|---|---|---|
| 0 | +0.264 | +2.62 | +0.351 | +0.338 | +0.011 |
| 1 | +0.256 | +2.03 | +0.395 | +0.327 | +0.010 |
| 2 | +0.247 | +0.85 | +0.394 | +0.303 | +0.017 |
| 3 | +0.250 | +3.25 | +0.392 | +0.243 | +0.023 |
| 4 | +0.271 | +1.70 | +0.399 | +0.345 | +0.010 |
| 5 | +0.310 | +2.11 | +0.423 | +0.282 | +0.025 |

## E5 — the price of a reset and of HP (paired, K = 12, horizon 50; Q50 primary, G50 beside it)

| family | contrast | ΔQ50 | ΔG50 |
|---|---|---|---|
| recall_full | 1t Recall − policy | -0.48 ± 0.17 (0+/5− of 6) | -0.49 ± 0.17 (0+/5− of 6) |
| recall_full | 1t stay − policy | -0.08 ± 0.08 (0+/1− of 6) | -0.08 ± 0.11 (0+/1− of 6) |
| recall_full | 1t Shove − policy | -0.06 ± 0.07 (0+/0− of 6) | -0.06 ± 0.06 (0+/0− of 6) |
| recall_full | Recall − stay | -0.40 ± 0.17 (0+/5− of 6) | -0.42 ± 0.19 (0+/5− of 6) |
| recall_full | Recall − stay, gold pre-invested | -0.40 ± 0.07 (0+/6− of 6) | -0.40 ± 0.09 (0+/6− of 6) |
| recall_full | shopping component | -0.00 ± 0.12 | -0.02 ± 0.14 |
| recall_full | pre-invested state − original (policy) | -0.01 ± 0.14 (0+/0− of 6) | -0.01 ± 0.13 (0+/0− of 6) |
| recall_mid | 1t Recall − policy | -0.17 ± 0.16 (0+/1− of 6) | -0.20 ± 0.18 (0+/2− of 6) |
| recall_mid | 1t stay − policy | -0.36 ± 0.12 (0+/5− of 6) | -0.37 ± 0.13 (0+/5− of 6) |
| recall_mid | 1t Shove − policy | -0.31 ± 0.13 (0+/4− of 6) | -0.30 ± 0.13 (0+/3− of 6) |
| recall_mid | Recall − stay | +0.19 ± 0.11 (1+/0− of 6) | +0.17 ± 0.13 (1+/0− of 6) |
| recall_mid | Recall − stay, gold pre-invested | +0.27 ± 0.12 (3+/0− of 6) | +0.27 ± 0.13 (2+/0− of 6) |
| recall_mid | shopping component | -0.08 ± 0.13 | -0.10 ± 0.14 |
| recall_mid | pre-invested state − original (policy) | -0.01 ± 0.16 (0+/0− of 6) | -0.02 ± 0.18 (0+/0− of 6) |


Recall − policy, ledger at 20 turns (focal A, opponent B):

| family | quantity | A | B |
|---|---|---|---|
| recall_full | away | +1.38 ± 0.10 | +0.12 ± 0.12 |
| recall_full | deaths | -0.01 ± 0.02 | +0.00 ± 0.01 |
| recall_full | farm | -0.32 ± 0.04 | -0.01 ± 0.05 |
| recall_full | earned | -0.33 ± 0.06 | +0.03 ± 0.04 |
| recall_full | unc | +0.34 ± 0.04 | -0.03 ± 0.04 |
| recall_full | xp_lost | +2.91 ± 0.40 | +0.27 ± 0.16 |
| recall_full | d_total | -0.07 ± 0.03 | +0.00 ± 0.02 |
| recall_mid | away | +0.16 ± 0.04 | +0.01 ± 0.02 |
| recall_mid | deaths | -0.02 ± 0.03 | -0.01 ± 0.01 |
| recall_mid | farm | -0.09 ± 0.08 | +0.06 ± 0.04 |
| recall_mid | earned | -0.10 ± 0.09 | +0.06 ± 0.05 |
| recall_mid | unc | +0.06 ± 0.08 | -0.05 ± 0.05 |
| recall_mid | xp_lost | +0.45 ± 0.65 | -0.37 ± 0.18 |
| recall_mid | d_total | +0.02 ± 0.02 | -0.01 ± 0.01 |


Price of HP (focal HP lowered by 0.3 bar, both continued by the policy):

| family | ΔQ50 | ΔG50 | price per bar (Q) | price per bar (G) | Δ Recalls @20 | Δ deaths @50 | Δ turns away @50 |
|---|---|---|---|---|---|---|---|
| hp_early | -0.37 ± 0.13 (0+/4− of 6) | -0.36 ± 0.13 (0+/3− of 6) | +1.23 ± 0.44 | +1.21 ± 0.42 | +0.28 ± 0.07 | +0.03 ± 0.03 | +0.70 ± 0.09 |
| hp_late | -0.44 ± 0.14 (0+/5− of 6) | -0.45 ± 0.14 (0+/5− of 6) | +1.48 ± 0.47 | +1.50 ± 0.46 | +0.24 ± 0.04 | +0.08 ± 0.01 | +0.86 ± 0.11 |


Validation (pooled, 300 turns):

| family | pairs | bias | slope | contradictions | verdict |
|---|---|---|---|---|---|
| recall_full | 288 | +0.18 | +0.93 | 0 | fail |
| recall_mid | 288 | -0.01 | +1.02 | 0 | pass |
| hp_early | 48 | +0.10 | +0.88 | 0 | pass |
| hp_late | 48 | +0.09 | +0.99 | 0 | pass |


States per final: {"recall_full": {"0": 48, "1": 48, "2": 48, "3": 48, "4": 48, "5": 48}, "recall_mid": {"0": 48, "1": 48, "2": 48, "3": 48, "4": 48, "5": 48}, "hp_early": {"0": 48, "1": 48, "2": 48, "3": 48, "4": 48, "5": 48}, "hp_late": {"0": 48, "1": 48, "2": 48, "3": 48, "4": 48, "5": 48}}


Exception 3: recall_full re-run at 150 turns (50 vs 150 on the same trajectories: {'pairs': 1728, 'mean_change': -0.03287575637368732, 'slope': 0.9624857456771426}):

| contrast | statistic and horizon | value |
|---|---|
| recall-stay|Q50 | -0.40 ± 0.17 (0+/5− of 6) |
| recall-policy|Q50 | -0.48 ± 0.17 (0+/5− of 6) |
| inv:recall-inv:stay|Q50 | -0.40 ± 0.07 (0+/6− of 6) |
| recall-stay|Q150 | -0.41 ± 0.16 (0+/5− of 6) |
| recall-policy|Q150 | -0.52 ± 0.18 (0+/5− of 6) |
| inv:recall-inv:stay|Q150 | -0.43 ± 0.10 (0+/5− of 6) |
| recall-stay|G50 | -0.42 ± 0.19 (0+/5− of 6) |
| recall-policy|G50 | -0.49 ± 0.17 (0+/5− of 6) |
| inv:recall-inv:stay|G50 | -0.40 ± 0.09 (0+/6− of 6) |
| recall-stay|G150 | -0.41 ± 0.17 (0+/5− of 6) |
| recall-policy|G150 | -0.52 ± 0.19 (0+/5− of 6) |
| inv:recall-inv:stay|G150 | -0.42 ± 0.10 (0+/5− of 6) |

## E2 — selective threats as plans on the trained policy (20-turn plans, horizon 60; candidate − policy, t intervals over the six finals, (k+/k− of n) finals individually significant)


### E2c: constructed zoning cells, opponent policy (ΔG60)

| gap / set | policy's absolute G60 | farm_beh − policy | threaten_beh − policy | edge_beh − policy | bully_beh − policy | freeze_threat − policy | g_bully_script − policy | g_farm_script − policy |
|---|---|---|---|---|---|---|---|---|
| equal | +0.14 ± 0.32 | -0.57 ± 0.21 (0+/1− of 6) | -0.85 ± 0.35 (0+/4− of 6) | -0.53 ± 0.18 (0+/2− of 6) | -0.94 ± 0.43 (0+/3− of 6) | -1.61 ± 0.25 (0+/6− of 6) | -1.47 ± 0.48 (0+/5− of 6) | -1.82 ± 0.44 (0+/6− of 6) |
| x1.5 | +3.22 ± 0.29 | -0.92 ± 0.33 (0+/4− of 6) | -0.57 ± 0.23 (0+/2− of 6) | -0.54 ± 0.16 (0+/2− of 6) | -0.58 ± 0.23 (0+/1− of 6) | -0.83 ± 0.23 (0+/1− of 6) | -0.96 ± 0.33 (0+/2− of 6) | -2.74 ± 0.39 (0+/6− of 6) |
| x1.76 | +4.46 ± 0.53 | -1.21 ± 0.52 (0+/5− of 6) | -0.40 ± 0.58 (0+/1− of 6) | -0.29 ± 0.37 (0+/2− of 6) | -0.32 ± 0.57 (0+/2− of 6) | -0.56 ± 0.27 (0+/2− of 6) | -0.82 ± 0.20 (0+/1− of 6) | -3.32 ± 0.57 (0+/6− of 6) |
| x2.2 | +6.44 ± 0.63 | -1.38 ± 0.51 (0+/6− of 6) | +0.06 ± 0.33 (0+/0− of 6) | +0.01 ± 0.39 (0+/0− of 6) | +0.06 ± 0.31 (1+/0− of 6) | -0.00 ± 0.49 (0+/0− of 6) | -0.28 ± 0.23 (0+/0− of 6) | -3.65 ± 0.45 (0+/6− of 6) |

| gap / set | threaten − farm_beh | bully_beh − farm_beh | g_bully_script − bully_beh (wave confound) | g_farm_script − farm_beh | Holm (4 selective plans vs policy; * = rejected) |
|---|---|---|---|---|---|
| equal | -0.28 ± 0.24 (0+/2− of 6) | -0.36 ± 0.31 (0+/0− of 6) | -0.54 ± 0.21 (0+/1− of 6) | -1.25 ± 0.38 (0+/6− of 6) | threaten_beh p=0.001*; edge_beh p=0.001*; bully_beh p=0.002*; freeze_threat p=0.000* |
| x1.5 | +0.35 ± 0.43 (1+/0− of 6) | +0.34 ± 0.41 (0+/0− of 6) | -0.38 ± 0.33 (0+/0− of 6) | -1.82 ± 0.12 (0+/6− of 6) | threaten_beh p=0.001*; edge_beh p=0.000*; bully_beh p=0.001*; freeze_threat p=0.000* |
| x1.76 | +0.80 ± 0.29 (2+/0− of 6) | +0.89 ± 0.29 (2+/0− of 6) | -0.50 ± 0.45 (0+/1− of 6) | -2.11 ± 0.24 (0+/6− of 6) | threaten_beh p=0.137; edge_beh p=0.099; bully_beh p=0.214; freeze_threat p=0.003* |
| x2.2 | +1.45 ± 0.36 (6+/0− of 6) | +1.44 ± 0.33 (6+/0− of 6) | -0.34 ± 0.18 (0+/0− of 6) | -2.27 ± 0.26 (0+/6− of 6) | threaten_beh p=0.639; edge_beh p=0.966; bully_beh p=0.626; freeze_threat p=0.992 |


Channels at 50 turns (x1.76; candidate − policy):

| plan | forced turns | A earned | B earned | B farm lost | of which away | B turns away | B deaths | B Recalls | A deaths | A HP lost (bars) | A connected trades | B Safe turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm_beh | 17.6 | -0.25 ± 0.36 | +1.07 ± 0.25 | -0.85 ± 0.14 | -0.67 ± 0.13 | -1.59 ± 0.29 | -0.34 ± 0.10 | -0.14 ± 0.12 | +0.09 ± 0.08 | +0.03 ± 0.08 | -3.38 ± 0.51 | -0.34 ± 0.36 |
| threaten_beh | 16.3 | -0.31 ± 0.46 | +0.22 ± 0.30 | +0.19 ± 0.22 | +0.19 ± 0.17 | +1.30 ± 0.28 | +0.07 ± 0.09 | +0.55 ± 0.19 | +0.13 ± 0.07 | +0.18 ± 0.07 | +2.17 ± 0.50 | -0.08 ± 0.46 |
| edge_beh | 16.5 | -0.19 ± 0.25 | +0.21 ± 0.23 | +0.14 ± 0.23 | +0.12 ± 0.18 | +1.07 ± 0.29 | +0.04 ± 0.08 | +0.49 ± 0.20 | +0.10 ± 0.03 | +0.15 ± 0.05 | +1.38 ± 0.47 | +0.04 ± 0.52 |
| bully_beh | 16.2 | -0.34 ± 0.49 | +0.16 ± 0.30 | +0.24 ± 0.23 | +0.21 ± 0.17 | +1.40 ± 0.28 | +0.08 ± 0.10 | +0.60 ± 0.18 | +0.14 ± 0.09 | +0.19 ± 0.08 | +2.56 ± 0.47 | -0.12 ± 0.46 |
| g_bully_script | 0.0 | -1.05 ± 0.14 | -0.18 ± 0.25 | +0.81 ± 0.28 | +0.94 ± 0.14 | +3.36 ± 0.32 | +0.58 ± 0.03 | +0.59 ± 0.16 | +0.01 ± 0.08 | +0.13 ± 0.09 | +5.66 ± 0.53 | -0.61 ± 0.55 |


### E2c: constructed zoning cells, opponent g_wary (ΔG60)

| gap / set | policy's absolute G60 | farm_beh − policy | threaten_beh − policy | edge_beh − policy | bully_beh − policy | freeze_threat − policy | g_bully_script − policy | g_farm_script − policy |
|---|---|---|---|---|---|---|---|---|
| equal | +3.72 ± 0.29 | -0.05 ± 0.10 (0+/0− of 6) | -0.05 ± 0.12 (0+/0− of 6) | -0.09 ± 0.08 (0+/0− of 6) | +0.16 ± 0.11 (0+/0− of 6) | +0.26 ± 0.20 (2+/0− of 6) | -0.56 ± 0.24 (0+/2− of 6) | +0.34 ± 0.18 (0+/0− of 6) |
| x1.5 | +4.32 ± 0.16 | +0.02 ± 0.06 (0+/0− of 6) | +0.15 ± 0.15 (1+/0− of 6) | +0.03 ± 0.10 (0+/0− of 6) | +0.15 ± 0.15 (0+/0− of 6) | +0.13 ± 0.12 (0+/0− of 6) | -0.47 ± 0.12 (0+/0− of 6) | +0.25 ± 0.08 (0+/0− of 6) |
| x1.76 | +4.65 ± 0.17 | -0.07 ± 0.12 (0+/2− of 6) | -0.05 ± 0.13 (0+/0− of 6) | -0.04 ± 0.14 (0+/0− of 6) | -0.04 ± 0.12 (0+/0− of 6) | +0.12 ± 0.12 (0+/0− of 6) | -0.34 ± 0.14 (0+/0− of 6) | +0.20 ± 0.10 (0+/0− of 6) |
| x2.2 | +5.22 ± 0.18 | -0.08 ± 0.14 (0+/2− of 6) | +0.01 ± 0.17 (0+/0− of 6) | +0.03 ± 0.16 (0+/0− of 6) | +0.01 ± 0.17 (0+/0− of 6) | +0.12 ± 0.13 (0+/0− of 6) | -0.54 ± 0.14 (0+/0− of 6) | +0.02 ± 0.12 (0+/0− of 6) |

| gap / set | threaten − farm_beh | bully_beh − farm_beh | g_bully_script − bully_beh (wave confound) | g_farm_script − farm_beh | Holm (4 selective plans vs policy; * = rejected) |
|---|---|---|---|---|---|
| equal | +0.00 ± 0.08 (0+/0− of 6) | +0.21 ± 0.05 (0+/0− of 6) | -0.71 ± 0.17 (0+/3− of 6) | +0.39 ± 0.16 (0+/0− of 6) | threaten_beh p=0.348; edge_beh p=0.042; bully_beh p=0.014; freeze_threat p=0.021 |
| x1.5 | +0.13 ± 0.13 (0+/0− of 6) | +0.13 ± 0.13 (0+/0− of 6) | -0.62 ± 0.15 (0+/0− of 6) | +0.23 ± 0.08 (0+/0− of 6) | threaten_beh p=0.047; edge_beh p=0.430; bully_beh p=0.050; freeze_threat p=0.038 |
| x1.76 | +0.02 ± 0.17 (0+/0− of 6) | +0.03 ± 0.17 (0+/0− of 6) | -0.30 ± 0.13 (0+/0− of 6) | +0.27 ± 0.17 (0+/0− of 6) | threaten_beh p=0.382; edge_beh p=0.507; bully_beh p=0.463; freeze_threat p=0.052 |
| x2.2 | +0.10 ± 0.12 (0+/0− of 6) | +0.10 ± 0.12 (0+/0− of 6) | -0.55 ± 0.10 (0+/0− of 6) | +0.10 ± 0.13 (0+/0− of 6) | threaten_beh p=0.842; edge_beh p=0.657; bully_beh p=0.833; freeze_threat p=0.061 |


Channels at 50 turns (x1.76; candidate − policy):

| plan | forced turns | A earned | B earned | B farm lost | of which away | B turns away | B deaths | B Recalls | A deaths | A HP lost (bars) | A connected trades | B Safe turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm_beh | 17.9 | -0.06 ± 0.09 | +0.03 ± 0.09 | -0.02 ± 0.11 | -0.03 ± 0.08 | -0.02 ± 0.19 | -0.00 ± 0.02 | -0.01 ± 0.07 | +0.01 ± 0.01 | +0.02 ± 0.02 | -0.19 ± 0.13 | -0.19 ± 0.51 |
| threaten_beh | 17.5 | -0.04 ± 0.10 | -0.08 ± 0.10 | +0.17 ± 0.08 | +0.09 ± 0.05 | +0.28 ± 0.10 | +0.01 ± 0.02 | +0.12 ± 0.04 | +0.05 ± 0.02 | +0.10 ± 0.03 | +1.05 ± 0.11 | -0.88 ± 0.49 |
| edge_beh | 17.5 | -0.03 ± 0.11 | -0.08 ± 0.10 | +0.17 ± 0.08 | +0.09 ± 0.05 | +0.28 ± 0.09 | +0.01 ± 0.02 | +0.12 ± 0.04 | +0.05 ± 0.02 | +0.09 ± 0.03 | +1.05 ± 0.11 | -0.81 ± 0.47 |
| bully_beh | 17.5 | -0.04 ± 0.10 | -0.08 ± 0.09 | +0.17 ± 0.08 | +0.09 ± 0.05 | +0.28 ± 0.10 | +0.01 ± 0.02 | +0.12 ± 0.04 | +0.05 ± 0.02 | +0.09 ± 0.03 | +1.06 ± 0.10 | -0.88 ± 0.51 |
| g_bully_script | 0.0 | -1.05 ± 0.12 | -1.25 ± 0.05 | +1.25 ± 0.07 | +0.15 ± 0.05 | +0.40 ± 0.13 | +0.02 ± 0.01 | +0.17 ± 0.06 | -0.00 ± 0.02 | -0.11 ± 0.03 | +0.33 ± 0.14 | +1.71 ± 0.54 |


Validation (E2c, pooled, 300 turns): pairs 252, bias +0.19, slope +0.97, contradictions 0: **fail**


### E2s: semi-natural ×1.76 gap states (E3 games), opponent policy (ΔG60)

| gap / set | policy's absolute G60 | farm_beh − policy | threaten_beh − policy | edge_beh − policy | bully_beh − policy | freeze_threat − policy | g_bully_script − policy | g_farm_script − policy |
|---|---|---|---|---|---|---|---|---|
| all | +3.25 ± 0.27 | -1.08 ± 0.17 (0+/6− of 6) | -0.75 ± 0.20 (0+/6− of 6) | -0.78 ± 0.18 (0+/5− of 6) | -0.74 ± 0.28 (0+/4− of 6) | -1.22 ± 0.22 (0+/6− of 6) | -1.52 ± 0.34 (0+/6− of 6) | -3.58 ± 0.22 (0+/6− of 6) |

| gap / set | threaten − farm_beh | bully_beh − farm_beh | g_bully_script − bully_beh (wave confound) | g_farm_script − farm_beh | Holm (4 selective plans vs policy; * = rejected) |
|---|---|---|---|---|---|
| all | +0.33 ± 0.10 (0+/0− of 6) | +0.34 ± 0.20 (2+/0− of 6) | -0.78 ± 0.18 (0+/6− of 6) | -2.50 ± 0.17 (0+/6− of 6) | threaten_beh p=0.000*; edge_beh p=0.000*; bully_beh p=0.001*; freeze_threat p=0.000* |


Channels at 50 turns (all; candidate − policy):

| plan | forced turns | A earned | B earned | B farm lost | of which away | B turns away | B deaths | B Recalls | A deaths | A HP lost (bars) | A connected trades | B Safe turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm_beh | 15.5 | -0.30 ± 0.14 | +0.95 ± 0.16 | -0.67 ± 0.13 | -0.68 ± 0.16 | -1.41 ± 0.42 | -0.30 ± 0.09 | -0.04 ± 0.07 | +0.07 ± 0.03 | +0.00 ± 0.05 | -3.07 ± 0.33 | +0.23 ± 0.57 |
| threaten_beh | 14.8 | -0.32 ± 0.12 | +0.48 ± 0.21 | -0.10 ± 0.16 | -0.18 ± 0.11 | +0.18 ± 0.25 | -0.11 ± 0.06 | +0.34 ± 0.12 | +0.12 ± 0.05 | +0.09 ± 0.06 | +0.26 ± 0.31 | -0.09 ± 0.44 |
| edge_beh | 15.0 | -0.31 ± 0.19 | +0.57 ± 0.19 | -0.24 ± 0.16 | -0.31 ± 0.12 | -0.32 ± 0.26 | -0.14 ± 0.07 | +0.16 ± 0.10 | +0.09 ± 0.03 | +0.08 ± 0.04 | -0.93 ± 0.50 | -0.16 ± 0.55 |
| bully_beh | 14.5 | -0.42 ± 0.16 | +0.40 ± 0.21 | +0.06 ± 0.18 | -0.06 ± 0.13 | +0.56 ± 0.24 | -0.08 ± 0.08 | +0.46 ± 0.11 | +0.14 ± 0.05 | +0.12 ± 0.07 | +1.16 ± 0.40 | +0.14 ± 0.47 |
| g_bully_script | 0.0 | -1.14 ± 0.21 | +0.41 ± 0.27 | +0.50 ± 0.23 | +0.53 ± 0.16 | +2.42 ± 0.55 | +0.31 ± 0.12 | +0.50 ± 0.09 | -0.01 ± 0.04 | +0.00 ± 0.06 | +4.91 ± 0.51 | +0.06 ± 0.94 |


### E2s: semi-natural ×1.76 gap states (E3 games), opponent g_wary (ΔG60)

| gap / set | policy's absolute G60 | farm_beh − policy | threaten_beh − policy | edge_beh − policy | bully_beh − policy | freeze_threat − policy | g_bully_script − policy | g_farm_script − policy |
|---|---|---|---|---|---|---|---|---|
| all | +4.28 ± 0.53 | -0.23 ± 0.13 (0+/1− of 6) | -0.46 ± 0.18 (0+/5− of 6) | -0.44 ± 0.19 (0+/5− of 6) | -0.46 ± 0.17 (0+/5− of 6) | -0.42 ± 0.16 (0+/4− of 6) | -0.83 ± 0.36 (0+/4− of 6) | -0.23 ± 0.40 (0+/1− of 6) |

| gap / set | threaten − farm_beh | bully_beh − farm_beh | g_bully_script − bully_beh (wave confound) | g_farm_script − farm_beh | Holm (4 selective plans vs policy; * = rejected) |
|---|---|---|---|---|---|
| all | -0.23 ± 0.08 (0+/5− of 6) | -0.23 ± 0.08 (0+/6− of 6) | -0.37 ± 0.36 (0+/3− of 6) | -0.00 ± 0.48 (0+/1− of 6) | threaten_beh p=0.001*; edge_beh p=0.002*; bully_beh p=0.001*; freeze_threat p=0.001* |


Channels at 50 turns (all; candidate − policy):

| plan | forced turns | A earned | B earned | B farm lost | of which away | B turns away | B deaths | B Recalls | A deaths | A HP lost (bars) | A connected trades | B Safe turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm_beh | 16.9 | +0.00 ± 0.08 | +0.31 ± 0.12 | -0.31 ± 0.10 | -0.31 ± 0.08 | -0.47 ± 0.11 | -0.13 ± 0.04 | +0.06 ± 0.05 | +0.00 ± 0.01 | -0.01 ± 0.01 | -0.69 ± 0.20 | +0.14 ± 0.45 |
| threaten_beh | 16.4 | -0.17 ± 0.07 | +0.37 ± 0.11 | -0.37 ± 0.11 | -0.30 ± 0.10 | -0.32 ± 0.11 | -0.13 ± 0.04 | +0.13 ± 0.07 | +0.01 ± 0.01 | +0.05 ± 0.01 | +0.01 ± 0.18 | -1.05 ± 0.47 |
| edge_beh | 16.5 | -0.15 ± 0.06 | +0.36 ± 0.12 | -0.36 ± 0.12 | -0.29 ± 0.10 | -0.34 ± 0.10 | -0.12 ± 0.04 | +0.12 ± 0.07 | +0.01 ± 0.01 | +0.04 ± 0.01 | -0.07 ± 0.19 | -0.84 ± 0.58 |
| bully_beh | 16.4 | -0.19 ± 0.08 | +0.34 ± 0.10 | -0.33 ± 0.10 | -0.27 ± 0.09 | -0.25 ± 0.09 | -0.12 ± 0.04 | +0.17 ± 0.07 | +0.01 ± 0.01 | +0.05 ± 0.01 | +0.14 ± 0.19 | -1.01 ± 0.49 |
| g_bully_script | 0.0 | -1.25 ± 0.19 | -0.84 ± 0.31 | +0.78 ± 0.30 | +0.09 ± 0.16 | +0.48 ± 0.24 | -0.08 ± 0.05 | +0.43 ± 0.11 | -0.02 ± 0.02 | -0.01 ± 0.04 | +0.87 ± 0.27 | +2.40 ± 0.55 |


Validation (E2s, pooled, 300 turns): pairs 168, bias -0.08, slope +0.91, contradictions 0: **pass**


### E2n: natural HP-edge states, opponent policy (ΔQ60)

| gap / set | policy's absolute G60 | farm_beh − policy | threaten_beh − policy | edge_beh − policy | bully_beh − policy | freeze_threat − policy | g_bully_script − policy | g_farm_script − policy |
|---|---|---|---|---|---|---|---|---|
| all | +1.25 ± 0.23 | -0.73 ± 0.24 (0+/5− of 6) | -0.91 ± 0.17 (0+/6− of 6) | -0.91 ± 0.21 (0+/6− of 6) | -1.02 ± 0.23 (0+/6− of 6) | -1.41 ± 0.23 (0+/6− of 6) | -1.66 ± 0.39 (0+/6− of 6) | -1.16 ± 0.21 (0+/6− of 6) |

| gap / set | threaten − farm_beh | bully_beh − farm_beh | g_bully_script − bully_beh (wave confound) | g_farm_script − farm_beh | Holm (4 selective plans vs policy; * = rejected) |
|---|---|---|---|---|---|
| all | -0.18 ± 0.31 (0+/1− of 6) | -0.29 ± 0.43 (0+/3− of 6) | -0.64 ± 0.27 (0+/4− of 6) | -0.43 ± 0.09 (0+/3− of 6) | threaten_beh p=0.000*; edge_beh p=0.000*; bully_beh p=0.000*; freeze_threat p=0.000* |


Channels at 50 turns (all; candidate − policy):

| plan | forced turns | A earned | B earned | B farm lost | of which away | B turns away | B deaths | B Recalls | A deaths | A HP lost (bars) | A connected trades | B Safe turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm_beh | 15.8 | -0.19 ± 0.12 | +0.73 ± 0.15 | -0.43 ± 0.11 | -0.55 ± 0.14 | -1.10 ± 0.22 | -0.20 ± 0.08 | -0.06 ± 0.13 | +0.13 ± 0.06 | +0.00 ± 0.06 | -3.10 ± 0.13 | +0.58 ± 0.83 |
| threaten_beh | 14.9 | -0.40 ± 0.24 | +0.65 ± 0.12 | -0.22 ± 0.12 | -0.26 ± 0.10 | -0.41 ± 0.19 | -0.08 ± 0.06 | -0.00 ± 0.10 | +0.16 ± 0.06 | +0.08 ± 0.05 | -1.13 ± 0.20 | -0.77 ± 0.62 |
| edge_beh | 15.0 | -0.34 ± 0.20 | +0.75 ± 0.17 | -0.34 ± 0.11 | -0.37 ± 0.14 | -0.72 ± 0.23 | -0.11 ± 0.08 | -0.08 ± 0.12 | +0.16 ± 0.06 | +0.07 ± 0.04 | -1.96 ± 0.15 | -0.91 ± 0.65 |
| bully_beh | 14.7 | -0.69 ± 0.20 | +0.48 ± 0.16 | +0.06 ± 0.10 | -0.02 ± 0.11 | +0.55 ± 0.19 | -0.02 ± 0.07 | +0.34 ± 0.14 | +0.19 ± 0.04 | +0.15 ± 0.04 | +1.72 ± 0.35 | -0.39 ± 0.65 |
| g_bully_script | 0.0 | -1.74 ± 0.30 | +0.05 ± 0.31 | +0.24 ± 0.25 | +0.19 ± 0.17 | +1.84 ± 0.27 | +0.08 ± 0.09 | +0.74 ± 0.18 | -0.04 ± 0.05 | -0.02 ± 0.08 | +5.40 ± 0.51 | +0.20 ± 0.68 |


### E2n: natural HP-edge states, opponent g_wary (ΔQ60)

| gap / set | policy's absolute G60 | farm_beh − policy | threaten_beh − policy | edge_beh − policy | bully_beh − policy | freeze_threat − policy | g_bully_script − policy | g_farm_script − policy |
|---|---|---|---|---|---|---|---|---|
| all | +4.45 ± 0.45 | -0.36 ± 0.16 (0+/2− of 6) | -0.68 ± 0.24 (0+/6− of 6) | -0.64 ± 0.23 (0+/6− of 6) | -0.65 ± 0.26 (0+/6− of 6) | -0.95 ± 0.23 (0+/6− of 6) | -1.94 ± 0.18 (0+/6− of 6) | -0.93 ± 0.20 (0+/6− of 6) |

| gap / set | threaten − farm_beh | bully_beh − farm_beh | g_bully_script − bully_beh (wave confound) | g_farm_script − farm_beh | Holm (4 selective plans vs policy; * = rejected) |
|---|---|---|---|---|---|
| all | -0.33 ± 0.13 (0+/3− of 6) | -0.29 ± 0.15 (0+/2− of 6) | -1.29 ± 0.26 (0+/6− of 6) | -0.58 ± 0.16 (0+/5− of 6) | threaten_beh p=0.001*; edge_beh p=0.001*; bully_beh p=0.001*; freeze_threat p=0.000* |


Channels at 50 turns (all; candidate − policy):

| plan | forced turns | A earned | B earned | B farm lost | of which away | B turns away | B deaths | B Recalls | A deaths | A HP lost (bars) | A connected trades | B Safe turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm_beh | 17.7 | +0.04 ± 0.13 | +0.38 ± 0.14 | -0.39 ± 0.12 | -0.47 ± 0.15 | -0.67 ± 0.21 | -0.21 ± 0.06 | +0.18 ± 0.03 | -0.00 ± 0.02 | -0.03 ± 0.02 | -1.41 ± 0.25 | +0.68 ± 0.91 |
| threaten_beh | 17.2 | -0.08 ± 0.16 | +0.66 ± 0.14 | -0.64 ± 0.14 | -0.46 ± 0.18 | -0.46 ± 0.20 | -0.15 ± 0.06 | +0.15 ± 0.06 | +0.01 ± 0.01 | +0.02 ± 0.01 | -0.69 ± 0.18 | -1.98 ± 0.52 |
| edge_beh | 17.3 | -0.06 ± 0.19 | +0.62 ± 0.15 | -0.61 ± 0.15 | -0.45 ± 0.19 | -0.48 ± 0.22 | -0.16 ± 0.07 | +0.15 ± 0.06 | +0.00 ± 0.01 | -0.00 ± 0.02 | -0.84 ± 0.16 | -1.80 ± 0.67 |
| bully_beh | 17.1 | -0.10 ± 0.15 | +0.58 ± 0.15 | -0.57 ± 0.15 | -0.40 ± 0.18 | -0.32 ± 0.21 | -0.15 ± 0.06 | +0.21 ± 0.06 | +0.01 ± 0.02 | +0.03 ± 0.02 | -0.34 ± 0.24 | -1.74 ± 0.57 |
| g_bully_script | 0.0 | -1.69 ± 0.22 | +0.24 ± 0.14 | -0.27 ± 0.13 | -0.32 ± 0.16 | +0.01 ± 0.18 | -0.16 ± 0.07 | +0.39 ± 0.09 | -0.01 ± 0.02 | -0.06 ± 0.03 | +0.35 ± 0.35 | -0.10 ± 1.01 |


Validation (E2n, pooled, 300 turns): pairs 168, bias -0.05, slope +1.00, contradictions 0: **pass**


### Exception 3: E2c re-run at 150 turns (60 vs 150 on the same trajectories: {'pairs': 2016, 'mean_change': 0.0700560838771691, 'slope': 0.9411739409099521})


Opponent policy, ΔG150:

| gap | threaten − policy | edge − policy | bully − policy | freeze + threat − policy | Hold only − policy | threaten − Hold only | g_bully − bully | g_farm − Hold only |
|---|---|---|---|---|---|---|---|---|
| equal | -0.78 ± 0.44 (0+/3− of 6) | -0.47 ± 0.21 (0+/1− of 6) | -0.84 ± 0.41 (0+/2− of 6) | -1.31 ± 0.24 (0+/6− of 6) | -0.54 ± 0.32 (0+/1− of 6) | -0.24 ± 0.38 (0+/1− of 6) | -0.51 ± 0.25 (0+/0− of 6) | -0.96 ± 0.44 (0+/3− of 6) |
| x1.5 | -0.55 ± 0.33 (0+/1− of 6) | -0.53 ± 0.32 (0+/1− of 6) | -0.57 ± 0.36 (0+/1− of 6) | -0.76 ± 0.36 (0+/1− of 6) | -0.93 ± 0.45 (0+/3− of 6) | +0.39 ± 0.43 (1+/0− of 6) | -0.22 ± 0.17 (0+/0− of 6) | -1.68 ± 0.15 (0+/5− of 6) |
| x1.76 | -0.38 ± 0.59 (0+/1− of 6) | -0.28 ± 0.43 (0+/1− of 6) | -0.30 ± 0.57 (0+/1− of 6) | -0.31 ± 0.42 (0+/1− of 6) | -1.27 ± 0.43 (0+/6− of 6) | +0.89 ± 0.33 (2+/0− of 6) | -0.26 ± 0.47 (0+/0− of 6) | -1.67 ± 0.17 (0+/6− of 6) |
| x2.2 | +0.14 ± 0.44 (0+/0− of 6) | +0.12 ± 0.47 (0+/0− of 6) | +0.13 ± 0.45 (1+/0− of 6) | +0.19 ± 0.61 (0+/0− of 6) | -1.40 ± 0.69 (0+/5− of 6) | +1.54 ± 0.44 (6+/0− of 6) | +0.00 ± 0.19 (0+/0− of 6) | -1.98 ± 0.14 (0+/6− of 6) |


Opponent g_wary, ΔG150:

| gap | threaten − policy | edge − policy | bully − policy | freeze + threat − policy | Hold only − policy | threaten − Hold only | g_bully − bully | g_farm − Hold only |
|---|---|---|---|---|---|---|---|---|
| equal | -0.04 ± 0.10 (0+/0− of 6) | -0.07 ± 0.09 (0+/1− of 6) | +0.22 ± 0.08 (0+/0− of 6) | +0.29 ± 0.15 (1+/0− of 6) | -0.01 ± 0.07 (0+/0− of 6) | -0.03 ± 0.11 (0+/0− of 6) | -0.76 ± 0.19 (0+/3− of 6) | +0.41 ± 0.17 (0+/0− of 6) |
| x1.5 | +0.15 ± 0.10 (2+/0− of 6) | +0.08 ± 0.10 (0+/0− of 6) | +0.15 ± 0.10 (1+/0− of 6) | +0.11 ± 0.12 (0+/0− of 6) | -0.02 ± 0.10 (0+/0− of 6) | +0.17 ± 0.15 (1+/0− of 6) | -0.56 ± 0.17 (0+/0− of 6) | +0.33 ± 0.12 (0+/0− of 6) |
| x1.76 | -0.01 ± 0.09 (0+/0− of 6) | +0.00 ± 0.09 (0+/0− of 6) | +0.01 ± 0.09 (0+/0− of 6) | +0.14 ± 0.23 (2+/0− of 6) | -0.09 ± 0.14 (0+/0− of 6) | +0.08 ± 0.14 (0+/0− of 6) | -0.30 ± 0.16 (0+/0− of 6) | +0.35 ± 0.15 (0+/0− of 6) |
| x2.2 | -0.02 ± 0.20 (0+/0− of 6) | -0.02 ± 0.20 (0+/0− of 6) | -0.02 ± 0.21 (0+/0− of 6) | +0.12 ± 0.16 (0+/0− of 6) | -0.05 ± 0.11 (0+/0− of 6) | +0.03 ± 0.23 (0+/0− of 6) | -0.56 ± 0.16 (0+/0− of 6) | +0.10 ± 0.14 (0+/0− of 6) |

## E4 — best responses trained from gap and freeze starts (seat A: target / attacker / M4.5c s0-attacker; seat B: the target; per target: mean ± 95% CI over evaluation states; pooled: t interval over targets)

| run | updates | time | learner transitions/s | entropy | max KL | finite |
|---|---|---|---|---|---|---|
| freeze_s0 | 2000 | 63 min | 1084 | 1.75 → 1.65 | 0.0089 | True |
| freeze_s2 | 2000 | 63 min | 1086 | 1.72 → 1.69 | 0.0078 | True |
| freeze_s4 | 2000 | 63 min | 1082 | 1.78 → 1.73 | 0.0088 | True |
| gap_s0 | 2000 | 70 min | 973 | 1.70 → 1.65 | 0.0085 | True |
| gap_s2 | 2000 | 70 min | 972 | 1.68 → 1.66 | 0.0096 | True |
| gap_s4 | 2000 | 70 min | 971 | 1.74 → 1.61 | 0.0090 | True |


### E4freeze_con

| contrast | pooled | targets significant | per target (* = significant) |
|---|---|---|---|
| attacker-s0att|G100 | +0.85 ± 1.06 | 3+/0− | s0: +0.78 ± 0.41*; s2: +1.30 ± 0.45*; s4: +0.46 ± 0.42* |
| attacker-s0att|G20 | +0.28 ± 0.89 | 1+/0− | s0: -0.05 ± 0.23; s2: +0.67 ± 0.31*; s4: +0.21 ± 0.26 |
| attacker-s0att|G50 | +0.63 ± 1.01 | 1+/0− | s0: +0.39 ± 0.40; s2: +1.10 ± 0.57*; s4: +0.40 ± 0.44 |
| attacker-target|G100 | +1.61 ± 0.72 | 3+/0− | s0: +1.62 ± 0.60*; s2: +1.32 ± 0.62*; s4: +1.90 ± 0.61* |
| attacker-target|G20 | +0.60 ± 0.19 | 3+/0− | s0: +0.53 ± 0.45*; s2: +0.59 ± 0.42*; s4: +0.68 ± 0.41* |
| attacker-target|G50 | +1.15 ± 1.04 | 3+/0− | s0: +1.06 ± 0.58*; s2: +0.78 ± 0.56*; s4: +1.60 ± 0.52* |
| s0att-target|G100 | +0.77 ± 1.78 | 2+/0− | s0: +0.84 ± 0.44*; s2: +0.02 ± 0.57; s4: +1.44 ± 0.38* |
| s0att-target|G20 | +0.32 ± 0.87 | 2+/0− | s0: +0.58 ± 0.33*; s2: -0.08 ± 0.44; s4: +0.47 ± 0.29* |
| s0att-target|G50 | +0.51 ± 1.92 | 2+/0− | s0: +0.67 ± 0.46*; s2: -0.32 ± 0.57; s4: +1.20 ± 0.32* |

| horizon | target | absolute: target in seat A | attacker in seat A | s0-attacker in seat A |
|---|---|---|---|---|
| G100 | 0 | +3.77 | +5.39 | +4.61 |
| G100 | 2 | +4.09 | +5.41 | +4.11 |
| G100 | 4 | +3.84 | +5.74 | +5.28 |
| G20 | 0 | +2.52 | +3.05 | +3.09 |
| G20 | 2 | +2.70 | +3.28 | +2.62 |
| G20 | 4 | +2.56 | +3.24 | +3.03 |
| G50 | 0 | +3.32 | +4.37 | +3.98 |
| G50 | 2 | +3.67 | +4.45 | +3.35 |
| G50 | 4 | +3.12 | +4.72 | +4.31 |

| channel (candidate − target, or the target's absolute level) | value |
|---|---|
| attacker:away_B@50 | +0.98 ± 1.78 |
| attacker:crash_hp_B@50 | -1.20 ± 3.40 |
| attacker:d_total_A@50 | +0.16 ± 0.57 |
| attacker:deaths_A@50 | +0.04 ± 0.27 |
| attacker:deaths_B@50 | +0.36 ± 0.72 |
| attacker:earned_A@50 | +0.16 ± 0.70 |
| attacker:earned_B@50 | -1.31 ± 1.19 |
| attacker:farm_A@50 | -0.67 ± 1.00 |
| attacker:held_A@50 | +0.41 ± 0.96 |
| attacker:n_safe_B@50 | +2.78 ± 5.05 |
| attacker:plates_A@50 | +0.11 ± 0.28 |
| attacker:recalls_B@50 | -0.23 ± 0.49 |
| attacker:trade_att_A@50 | +5.21 ± 6.63 |
| attacker:trade_conn_A@50 | +4.06 ± 5.49 |
| attacker:unc_B@50 | +1.14 ± 0.76 |
| attacker:unc_away_B@50 | +0.69 ± 0.75 |
| attacker:xp_lost_B@50 | +6.21 ± 8.50 |
| s0att:away_B@50 | -0.20 ± 1.56 |
| s0att:crash_hp_B@50 | -0.36 ± 7.16 |
| s0att:d_total_A@50 | -0.09 ± 0.37 |
| s0att:deaths_A@50 | -0.04 ± 0.04 |
| s0att:deaths_B@50 | -0.02 ± 0.56 |
| s0att:earned_A@50 | +0.31 ± 1.34 |
| s0att:earned_B@50 | -0.37 ± 1.41 |
| s0att:farm_A@50 | +0.23 ± 0.42 |
| s0att:held_A@50 | +0.87 ± 3.76 |
| s0att:n_safe_B@50 | +2.94 ± 8.63 |
| s0att:plates_A@50 | +0.13 ± 0.46 |
| s0att:recalls_B@50 | -0.04 ± 0.29 |
| s0att:trade_att_A@50 | +0.55 ± 2.65 |
| s0att:trade_conn_A@50 | +0.64 ± 3.92 |
| s0att:unc_B@50 | +0.26 ± 1.32 |
| s0att:unc_away_B@50 | +0.01 ± 0.67 |
| s0att:xp_lost_B@50 | -0.01 ± 8.10 |
| target_abs:away_B@50 | +6.47 |
| target_abs:deaths_B@50 | +0.67 |
| target_abs:earned_A@50 | +23.04 |
| target_abs:earned_B@50 | +19.21 |
| target_abs:held_A@50 | +3.12 |
| target_abs:trade_conn_A@50 | +7.85 |
| target_abs:unc_B@50 | +6.47 |


### E4freeze_nat

| contrast | pooled | targets significant | per target (* = significant) |
|---|---|---|---|
| attacker-s0att|G100 | -0.03 ± 0.38 | 0+/0− | s0: +0.04 ± 0.38; s2: +0.08 ± 0.32; s4: -0.20 ± 0.36 |
| attacker-s0att|G20 | -0.01 ± 0.18 | 0+/0− | s0: +0.05 ± 0.28; s2: +0.01 ± 0.21; s4: -0.09 ± 0.21 |
| attacker-s0att|G50 | +0.02 ± 0.05 | 0+/0− | s0: +0.02 ± 0.36; s2: -0.00 ± 0.34; s4: +0.04 ± 0.34 |
| attacker-target|G100 | +0.86 ± 0.90 | 3+/0− | s0: +0.98 ± 0.40*; s2: +0.45 ± 0.36*; s4: +1.15 ± 0.39* |
| attacker-target|G20 | +0.14 ± 0.57 | 1+/0− | s0: +0.40 ± 0.28*; s2: -0.02 ± 0.22; s4: +0.04 ± 0.27 |
| attacker-target|G50 | +0.51 ± 0.54 | 2+/0− | s0: +0.68 ± 0.30*; s2: +0.27 ± 0.29; s4: +0.59 ± 0.37* |
| s0att-target|G100 | +0.89 ± 1.22 | 2+/0− | s0: +0.94 ± 0.41*; s2: +0.38 ± 0.39; s4: +1.35 ± 0.36* |
| s0att-target|G20 | +0.15 ± 0.47 | 1+/0− | s0: +0.35 ± 0.25*; s2: -0.03 ± 0.22; s4: +0.12 ± 0.27 |
| s0att-target|G50 | +0.50 ± 0.50 | 2+/0− | s0: +0.66 ± 0.37*; s2: +0.27 ± 0.29; s4: +0.55 ± 0.28* |

| horizon | target | absolute: target in seat A | attacker in seat A | s0-attacker in seat A |
|---|---|---|---|---|
| G100 | 0 | +0.80 | +1.78 | +1.74 |
| G100 | 2 | +0.71 | +1.16 | +1.09 |
| G100 | 4 | +0.87 | +2.02 | +2.23 |
| G20 | 0 | +0.67 | +1.08 | +1.02 |
| G20 | 2 | +0.80 | +0.78 | +0.77 |
| G20 | 4 | +0.82 | +0.86 | +0.95 |
| G50 | 0 | +0.74 | +1.42 | +1.40 |
| G50 | 2 | +0.68 | +0.95 | +0.95 |
| G50 | 4 | +0.94 | +1.53 | +1.49 |

| channel (candidate − target, or the target's absolute level) | value |
|---|---|
| attacker:away_B@50 | +0.52 ± 1.45 |
| attacker:crash_hp_B@50 | +2.04 ± 11.75 |
| attacker:d_total_A@50 | +0.03 ± 0.74 |
| attacker:deaths_A@50 | +0.04 ± 0.31 |
| attacker:deaths_B@50 | +0.12 ± 0.63 |
| attacker:earned_A@50 | -0.01 ± 0.39 |
| attacker:earned_B@50 | -0.72 ± 0.86 |
| attacker:farm_A@50 | -0.30 ± 1.21 |
| attacker:held_A@50 | +0.54 ± 1.12 |
| attacker:n_safe_B@50 | +2.78 ± 8.02 |
| attacker:plates_A@50 | +0.05 ± 0.35 |
| attacker:recalls_B@50 | +0.01 ± 0.70 |
| attacker:trade_att_A@50 | +3.37 ± 6.97 |
| attacker:trade_conn_A@50 | +3.16 ± 4.28 |
| attacker:unc_B@50 | +0.62 ± 0.75 |
| attacker:unc_away_B@50 | +0.31 ± 1.10 |
| attacker:xp_lost_B@50 | +2.88 ± 11.22 |
| s0att:away_B@50 | -0.08 ± 1.01 |
| s0att:crash_hp_B@50 | +3.48 ± 14.30 |
| s0att:d_total_A@50 | -0.04 ± 0.61 |
| s0att:deaths_A@50 | -0.03 ± 0.18 |
| s0att:deaths_B@50 | -0.02 ± 0.60 |
| s0att:earned_A@50 | +0.18 ± 0.72 |
| s0att:earned_B@50 | -0.50 ± 1.19 |
| s0att:farm_A@50 | +0.17 ± 1.17 |
| s0att:held_A@50 | +0.68 ± 1.38 |
| s0att:n_safe_B@50 | +2.77 ± 10.19 |
| s0att:plates_A@50 | +0.06 ± 0.34 |
| s0att:recalls_B@50 | +0.01 ± 0.89 |
| s0att:trade_att_A@50 | +1.14 ± 4.29 |
| s0att:trade_conn_A@50 | +1.43 ± 3.23 |
| s0att:unc_B@50 | +0.35 ± 0.99 |
| s0att:unc_away_B@50 | +0.01 ± 0.81 |
| s0att:xp_lost_B@50 | +0.30 ± 10.83 |
| target_abs:away_B@50 | +6.24 |
| target_abs:deaths_B@50 | +0.40 |
| target_abs:earned_A@50 | +20.65 |
| target_abs:earned_B@50 | +19.86 |
| target_abs:held_A@50 | +3.41 |
| target_abs:trade_conn_A@50 | +7.08 |
| target_abs:unc_B@50 | +6.24 |


### E4freeze_pooled

| contrast | pooled | targets significant | per target (* = significant) |
|---|---|---|---|
| attacker-s0att|G100 | +0.21 ± 0.54 | 1+/0− | s0: +0.25 ± 0.30; s2: +0.41 ± 0.29*; s4: -0.02 ± 0.29 |
| attacker-s0att|G20 | +0.07 ± 0.26 | 1+/0− | s0: +0.03 ± 0.21; s2: +0.19 ± 0.18*; s4: -0.01 ± 0.17 |
| attacker-s0att|G50 | +0.19 ± 0.25 | 0+/0− | s0: +0.12 ± 0.28; s2: +0.30 ± 0.31; s4: +0.14 ± 0.27 |
| attacker-target|G100 | +1.07 ± 0.85 | 3+/0− | s0: +1.16 ± 0.33*; s2: +0.69 ± 0.32*; s4: +1.35 ± 0.33* |
| attacker-target|G20 | +0.27 ± 0.38 | 1+/0− | s0: +0.44 ± 0.23*; s2: +0.15 ± 0.20; s4: +0.21 ± 0.23 |
| attacker-target|G50 | +0.69 ± 0.61 | 3+/0− | s0: +0.78 ± 0.27*; s2: +0.41 ± 0.26*; s4: +0.87 ± 0.31* |
| s0att-target|G100 | +0.86 ± 1.37 | 2+/0− | s0: +0.91 ± 0.32*; s2: +0.28 ± 0.32; s4: +1.38 ± 0.28* |
| s0att-target|G20 | +0.20 ± 0.56 | 2+/0− | s0: +0.41 ± 0.20*; s2: -0.04 ± 0.19; s4: +0.22 ± 0.21* |
| s0att-target|G50 | +0.50 ± 0.85 | 2+/0− | s0: +0.66 ± 0.29*; s2: +0.11 ± 0.26; s4: +0.73 ± 0.23* |

| horizon | target | absolute: target in seat A | attacker in seat A | s0-attacker in seat A |
|---|---|---|---|---|
| G100 | 0 | +1.61 | +2.76 | +2.52 |
| G100 | 2 | +1.63 | +2.32 | +1.91 |
| G100 | 4 | +1.68 | +3.04 | +3.06 |
| G20 | 0 | +1.18 | +1.61 | +1.59 |
| G20 | 2 | +1.32 | +1.47 | +1.28 |
| G20 | 4 | +1.30 | +1.51 | +1.52 |
| G50 | 0 | +1.44 | +2.23 | +2.11 |
| G50 | 2 | +1.50 | +1.91 | +1.61 |
| G50 | 4 | +1.53 | +2.40 | +2.26 |

| channel (candidate − target, or the target's absolute level) | value |
|---|---|
| attacker:away_B@50 | +0.64 ± 1.42 |
| attacker:crash_hp_B@50 | +1.16 ± 9.43 |
| attacker:d_total_A@50 | +0.07 ± 0.69 |
| attacker:deaths_A@50 | +0.04 ± 0.29 |
| attacker:deaths_B@50 | +0.18 ± 0.60 |
| attacker:earned_A@50 | +0.04 ± 0.35 |
| attacker:earned_B@50 | -0.88 ± 0.86 |
| attacker:farm_A@50 | -0.40 ± 1.11 |
| attacker:held_A@50 | +0.50 ± 1.00 |
| attacker:n_safe_B@50 | +2.78 ± 7.18 |
| attacker:plates_A@50 | +0.07 ± 0.31 |
| attacker:recalls_B@50 | -0.06 ± 0.61 |
| attacker:trade_att_A@50 | +3.87 ± 6.56 |
| attacker:trade_conn_A@50 | +3.40 ± 4.42 |
| attacker:unc_B@50 | +0.76 ± 0.71 |
| attacker:unc_away_B@50 | +0.42 ± 0.96 |
| attacker:xp_lost_B@50 | +3.79 ± 10.07 |
| s0att:away_B@50 | -0.11 ± 0.79 |
| s0att:crash_hp_B@50 | +2.44 ± 12.21 |
| s0att:d_total_A@50 | -0.06 ± 0.55 |
| s0att:deaths_A@50 | -0.03 ± 0.14 |
| s0att:deaths_B@50 | -0.02 ± 0.51 |
| s0att:earned_A@50 | +0.22 ± 0.39 |
| s0att:earned_B@50 | -0.46 ± 1.22 |
| s0att:farm_A@50 | +0.19 ± 0.96 |
| s0att:held_A@50 | +0.73 ± 2.02 |
| s0att:n_safe_B@50 | +2.81 ± 9.75 |
| s0att:plates_A@50 | +0.08 ± 0.34 |
| s0att:recalls_B@50 | -0.00 ± 0.72 |
| s0att:trade_att_A@50 | +0.98 ± 3.49 |
| s0att:trade_conn_A@50 | +1.22 ± 3.26 |
| s0att:unc_B@50 | +0.33 ± 1.05 |
| s0att:unc_away_B@50 | +0.01 ± 0.67 |
| s0att:xp_lost_B@50 | +0.21 ± 9.64 |
| target_abs:away_B@50 | +6.31 |
| target_abs:deaths_B@50 | +0.47 |
| target_abs:earned_A@50 | +21.30 |
| target_abs:earned_B@50 | +19.68 |
| target_abs:held_A@50 | +3.33 |
| target_abs:trade_conn_A@50 | +7.29 |
| target_abs:unc_B@50 | +6.31 |


### E4gap_con

| contrast | pooled | targets significant | per target (* = significant) |
|---|---|---|---|
| attacker-s0att|G100 | +1.97 ± 0.80 | 3+/0− | s0: +2.33 ± 0.58*; s2: +1.72 ± 0.54*; s4: +1.86 ± 0.55* |
| attacker-s0att|G150 | +2.15 ± 0.93 | 3+/0− | s0: +2.58 ± 0.61*; s2: +1.89 ± 0.56*; s4: +1.98 ± 0.59* |
| attacker-s0att|G20 | +0.90 ± 0.54 | 3+/0− | s0: +1.08 ± 0.40*; s2: +0.66 ± 0.34*; s4: +0.96 ± 0.35* |
| attacker-s0att|G50 | +1.46 ± 1.00 | 3+/0− | s0: +1.90 ± 0.53*; s2: +1.10 ± 0.49*; s4: +1.39 ± 0.47* |
| attacker-target|G100 | +2.60 ± 1.11 | 3+/0− | s0: +3.01 ± 0.69*; s2: +2.13 ± 0.56*; s4: +2.67 ± 0.48* |
| attacker-target|G150 | +2.98 ± 0.92 | 3+/0− | s0: +3.21 ± 0.75*; s2: +2.55 ± 0.55*; s4: +3.17 ± 0.51* |
| attacker-target|G20 | +1.02 ± 0.48 | 3+/0− | s0: +1.12 ± 0.43*; s2: +0.80 ± 0.38*; s4: +1.13 ± 0.22* |
| attacker-target|G50 | +1.80 ± 1.47 | 3+/0− | s0: +2.35 ± 0.57*; s2: +1.17 ± 0.45*; s4: +1.86 ± 0.41* |
| s0att-target|G100 | +0.63 ± 0.51 | 2+/0− | s0: +0.68 ± 0.33*; s2: +0.40 ± 0.49; s4: +0.81 ± 0.36* |
| s0att-target|G150 | +0.83 ± 0.78 | 3+/0− | s0: +0.63 ± 0.33*; s2: +0.66 ± 0.53*; s4: +1.19 ± 0.33* |
| s0att-target|G20 | +0.12 ± 0.17 | 0+/0− | s0: +0.04 ± 0.30; s2: +0.13 ± 0.30; s4: +0.17 ± 0.26 |
| s0att-target|G50 | +0.33 ± 0.57 | 2+/0− | s0: +0.46 ± 0.31*; s2: +0.07 ± 0.44; s4: +0.47 ± 0.31* |

| horizon | target | absolute: target in seat A | attacker in seat A | s0-attacker in seat A |
|---|---|---|---|---|
| G100 | 0 | +4.78 | +7.79 | +5.46 |
| G100 | 2 | +5.14 | +7.26 | +5.54 |
| G100 | 4 | +5.10 | +7.77 | +5.91 |
| G150 | 0 | +5.08 | +8.29 | +5.71 |
| G150 | 2 | +5.41 | +7.96 | +6.07 |
| G150 | 4 | +5.33 | +8.50 | +6.52 |
| G20 | 0 | +2.31 | +3.43 | +2.35 |
| G20 | 2 | +2.40 | +3.20 | +2.54 |
| G20 | 4 | +2.25 | +3.39 | +2.42 |
| G50 | 0 | +3.73 | +6.08 | +4.18 |
| G50 | 2 | +4.26 | +5.44 | +4.33 |
| G50 | 4 | +4.03 | +5.90 | +4.50 |

| channel (candidate − target, or the target's absolute level) | value |
|---|---|
| attacker:away_B@100 | +5.24 ± 3.92 |
| attacker:crash_hp_B@100 | +4.84 ± 13.43 |
| attacker:d_total_A@100 | +0.63 ± 0.61 |
| attacker:deaths_A@100 | +0.35 ± 0.24 |
| attacker:deaths_B@100 | +1.55 ± 1.40 |
| attacker:earned_A@100 | +0.27 ± 0.68 |
| attacker:earned_B@100 | -3.60 ± 0.47 |
| attacker:farm_A@100 | -2.79 ± 2.39 |
| attacker:held_A@100 | +1.54 ± 2.13 |
| attacker:n_safe_B@100 | +2.38 ± 5.43 |
| attacker:plates_A@100 | -0.04 ± 0.27 |
| attacker:recalls_B@100 | -0.77 ± 1.23 |
| attacker:trade_att_A@100 | +14.99 ± 6.62 |
| attacker:trade_conn_A@100 | +12.43 ± 6.80 |
| attacker:unc_B@100 | +4.12 ± 1.19 |
| attacker:unc_away_B@100 | +2.97 ± 2.09 |
| attacker:xp_lost_B@100 | +28.64 ± 18.05 |
| s0att:away_B@100 | -0.24 ± 2.65 |
| s0att:crash_hp_B@100 | +2.02 ± 20.34 |
| s0att:d_total_A@100 | -0.23 ± 0.98 |
| s0att:deaths_A@100 | -0.13 ± 0.23 |
| s0att:deaths_B@100 | -0.14 ± 1.09 |
| s0att:earned_A@100 | +0.29 ± 0.73 |
| s0att:earned_B@100 | -0.77 ± 1.48 |
| s0att:farm_A@100 | +0.47 ± 1.86 |
| s0att:held_A@100 | +0.82 ± 2.76 |
| s0att:n_safe_B@100 | +5.98 ± 20.69 |
| s0att:plates_A@100 | +0.10 ± 0.22 |
| s0att:recalls_B@100 | +0.17 ± 1.19 |
| s0att:trade_att_A@100 | +0.99 ± 5.37 |
| s0att:trade_conn_A@100 | +1.36 ± 4.89 |
| s0att:unc_B@100 | +0.48 ± 1.62 |
| s0att:unc_away_B@100 | -0.04 ± 1.62 |
| s0att:xp_lost_B@100 | -0.23 ± 18.11 |
| target_abs:away_B@100 | +13.73 |
| target_abs:deaths_B@100 | +1.53 |
| target_abs:earned_A@100 | +44.64 |
| target_abs:earned_B@100 | +37.73 |
| target_abs:held_A@100 | +5.26 |
| target_abs:trade_conn_A@100 | +15.13 |
| target_abs:unc_B@100 | +13.97 |


### E4gap_equal

| contrast | pooled | targets significant | per target (* = significant) |
|---|---|---|---|
| attacker-s0att|G100 | -0.06 ± 1.69 | 0+/1− | s0: +0.43 ± 0.74; s2: +0.22 ± 0.65; s4: -0.84 ± 0.72* |
| attacker-s0att|G150 | -0.30 ± 1.88 | 0+/1− | s0: +0.25 ± 0.77; s2: +0.02 ± 0.56; s4: -1.16 ± 0.96* |
| attacker-s0att|G20 | -0.08 ± 1.03 | 0+/1− | s0: +0.13 ± 0.44; s2: +0.19 ± 0.53; s4: -0.56 ± 0.22* |
| attacker-s0att|G50 | -0.12 ± 2.05 | 1+/1− | s0: +0.30 ± 0.67; s2: +0.41 ± 0.36*; s4: -1.07 ± 0.73* |
| attacker-target|G100 | +0.77 ± 0.05 | 1+/0− | s0: +0.77 ± 0.70*; s2: +0.75 ± 0.77; s4: +0.79 ± 1.13 |
| attacker-target|G150 | +0.84 ± 0.14 | 2+/0− | s0: +0.78 ± 0.75*; s2: +0.84 ± 0.69*; s4: +0.89 ± 1.19 |
| attacker-target|G20 | -0.06 ± 0.79 | 0+/0− | s0: +0.16 ± 0.33; s2: +0.08 ± 0.63; s4: -0.42 ± 0.56 |
| attacker-target|G50 | +0.30 ± 0.80 | 0+/0− | s0: +0.40 ± 0.53; s2: +0.55 ± 0.66; s4: -0.06 ± 0.91 |
| s0att-target|G100 | +0.83 ± 1.72 | 1+/0− | s0: +0.34 ± 0.49; s2: +0.53 ± 0.73; s4: +1.62 ± 0.86* |
| s0att-target|G150 | +1.13 ± 2.01 | 2+/0− | s0: +0.53 ± 0.65; s2: +0.82 ± 0.75*; s4: +2.05 ± 0.84* |
| s0att-target|G20 | +0.02 ± 0.30 | 0+/0− | s0: +0.03 ± 0.29; s2: -0.10 ± 0.63; s4: +0.14 ± 0.46 |
| s0att-target|G50 | +0.41 ± 1.27 | 1+/0− | s0: +0.10 ± 0.47; s2: +0.14 ± 0.56; s4: +1.00 ± 0.69* |

| horizon | target | absolute: target in seat A | attacker in seat A | s0-attacker in seat A |
|---|---|---|---|---|
| G100 | 0 | +0.33 | +1.10 | +0.67 |
| G100 | 2 | +0.21 | +0.96 | +0.74 |
| G100 | 4 | -0.09 | +0.70 | +1.54 |
| G150 | 0 | +0.30 | +1.08 | +0.83 |
| G150 | 2 | +0.22 | +1.06 | +1.04 |
| G150 | 4 | -0.13 | +0.77 | +1.93 |
| G20 | 0 | +0.18 | +0.35 | +0.21 |
| G20 | 2 | +0.14 | +0.22 | +0.03 |
| G20 | 4 | +0.24 | -0.18 | +0.38 |
| G50 | 0 | +0.27 | +0.67 | +0.37 |
| G50 | 2 | +0.19 | +0.74 | +0.33 |
| G50 | 4 | +0.09 | +0.02 | +1.09 |

| channel (candidate − target, or the target's absolute level) | value |
|---|---|
| attacker:away_B@100 | +2.12 ± 2.98 |
| attacker:crash_hp_B@100 | +4.17 ± 18.38 |
| attacker:d_total_A@100 | +0.41 ± 0.53 |
| attacker:deaths_A@100 | +0.29 ± 0.33 |
| attacker:deaths_B@100 | +0.61 ± 0.90 |
| attacker:earned_A@100 | -0.54 ± 0.98 |
| attacker:earned_B@100 | -1.92 ± 0.52 |
| attacker:farm_A@100 | -1.87 ± 2.57 |
| attacker:held_A@100 | +1.25 ± 1.11 |
| attacker:n_safe_B@100 | +4.01 ± 7.79 |
| attacker:plates_A@100 | +0.10 ± 0.47 |
| attacker:recalls_B@100 | -0.32 ± 0.60 |
| attacker:trade_att_A@100 | +10.14 ± 9.34 |
| attacker:trade_conn_A@100 | +9.30 ± 8.31 |
| attacker:unc_B@100 | +2.36 ± 1.39 |
| attacker:unc_away_B@100 | +1.32 ± 1.39 |
| attacker:xp_lost_B@100 | +14.37 ± 17.16 |
| s0att:away_B@100 | -0.55 ± 0.66 |
| s0att:crash_hp_B@100 | +1.57 ± 32.74 |
| s0att:d_total_A@100 | -0.30 ± 0.85 |
| s0att:deaths_A@100 | -0.19 ± 0.50 |
| s0att:deaths_B@100 | -0.15 ± 0.46 |
| s0att:earned_A@100 | +0.66 ± 0.74 |
| s0att:earned_B@100 | -0.70 ± 2.88 |
| s0att:farm_A@100 | +0.75 ± 1.07 |
| s0att:held_A@100 | +0.40 ± 1.64 |
| s0att:n_safe_B@100 | +5.35 ± 19.34 |
| s0att:plates_A@100 | +0.21 ± 0.43 |
| s0att:recalls_B@100 | +0.05 ± 0.90 |
| s0att:trade_att_A@100 | +1.01 ± 3.96 |
| s0att:trade_conn_A@100 | +1.38 ± 5.07 |
| s0att:unc_B@100 | +0.25 ± 2.57 |
| s0att:unc_away_B@100 | -0.16 ± 0.42 |
| s0att:xp_lost_B@100 | -1.90 ± 11.01 |
| target_abs:away_B@100 | +11.79 |
| target_abs:deaths_B@100 | +0.84 |
| target_abs:earned_A@100 | +41.21 |
| target_abs:earned_B@100 | +41.05 |
| target_abs:held_A@100 | +5.49 |
| target_abs:trade_conn_A@100 | +15.24 |
| target_abs:unc_B@100 | +12.30 |


### E4gap_pooled

| contrast | pooled | targets significant | per target (* = significant) |
|---|---|---|---|
| attacker-s0att|G100 | +1.46 ± 0.44 | 3+/0− | s0: +1.53 ± 0.40*; s2: +1.59 ± 0.37*; s4: +1.26 ± 0.38* |
| attacker-s0att|G150 | +1.56 ± 0.52 | 3+/0− | s0: +1.66 ± 0.42*; s2: +1.70 ± 0.41*; s4: +1.32 ± 0.39* |
| attacker-s0att|G20 | +0.61 ± 0.25 | 3+/0− | s0: +0.52 ± 0.25*; s2: +0.60 ± 0.21*; s4: +0.72 ± 0.29* |
| attacker-s0att|G50 | +1.09 ± 0.14 | 3+/0− | s0: +1.16 ± 0.33*; s2: +1.05 ± 0.32*; s4: +1.07 ± 0.35* |
| attacker-target|G100 | +2.11 ± 0.62 | 3+/0− | s0: +2.34 ± 0.40*; s2: +1.84 ± 0.43*; s4: +2.14 ± 0.35* |
| attacker-target|G150 | +2.44 ± 0.54 | 3+/0− | s0: +2.58 ± 0.40*; s2: +2.19 ± 0.45*; s4: +2.54 ± 0.37* |
| attacker-target|G20 | +0.81 ± 0.13 | 3+/0− | s0: +0.84 ± 0.24*; s2: +0.75 ± 0.23*; s4: +0.84 ± 0.20* |
| attacker-target|G50 | +1.48 ± 0.71 | 3+/0− | s0: +1.76 ± 0.32*; s2: +1.19 ± 0.36*; s4: +1.50 ± 0.28* |
| s0att-target|G100 | +0.65 ± 0.85 | 2+/0− | s0: +0.80 ± 0.29*; s2: +0.25 ± 0.37; s4: +0.88 ± 0.29* |
| s0att-target|G150 | +0.88 ± 0.92 | 3+/0− | s0: +0.92 ± 0.31*; s2: +0.49 ± 0.40*; s4: +1.22 ± 0.25* |
| s0att-target|G20 | +0.20 ± 0.28 | 1+/0− | s0: +0.32 ± 0.21*; s2: +0.15 ± 0.21; s4: +0.12 ± 0.19 |
| s0att-target|G50 | +0.39 ± 0.58 | 2+/0− | s0: +0.60 ± 0.25*; s2: +0.13 ± 0.31; s4: +0.43 ± 0.23* |

| horizon | target | absolute: target in seat A | attacker in seat A | s0-attacker in seat A |
|---|---|---|---|---|
| G100 | 0 | +3.75 | +6.09 | +4.56 |
| G100 | 2 | +4.42 | +6.26 | +4.67 |
| G100 | 4 | +4.22 | +6.36 | +5.10 |
| G150 | 0 | +3.92 | +6.50 | +4.84 |
| G150 | 2 | +4.59 | +6.78 | +5.07 |
| G150 | 4 | +4.37 | +6.92 | +5.59 |
| G20 | 0 | +1.78 | +2.63 | +2.10 |
| G20 | 2 | +1.90 | +2.65 | +2.05 |
| G20 | 4 | +1.98 | +2.81 | +2.09 |
| G50 | 0 | +2.99 | +4.75 | +3.59 |
| G50 | 2 | +3.53 | +4.72 | +3.67 |
| G50 | 4 | +3.41 | +4.91 | +3.84 |

| channel (candidate − target, or the target's absolute level) | value |
|---|---|
| attacker:away_B@100 | +4.42 ± 3.30 |
| attacker:crash_hp_B@100 | +6.31 ± 13.55 |
| attacker:d_total_A@100 | +0.56 ± 0.60 |
| attacker:deaths_A@100 | +0.37 ± 0.16 |
| attacker:deaths_B@100 | +1.31 ± 1.15 |
| attacker:earned_A@100 | -0.07 ± 0.13 |
| attacker:earned_B@100 | -3.19 ± 0.86 |
| attacker:farm_A@100 | -2.64 ± 2.57 |
| attacker:held_A@100 | +1.51 ± 2.24 |
| attacker:n_safe_B@100 | +2.95 ± 5.90 |
| attacker:plates_A@100 | -0.04 ± 0.13 |
| attacker:recalls_B@100 | -0.75 ± 1.11 |
| attacker:trade_att_A@100 | +13.60 ± 7.41 |
| attacker:trade_conn_A@100 | +11.33 ± 6.79 |
| attacker:unc_B@100 | +3.74 ± 1.21 |
| attacker:unc_away_B@100 | +2.67 ± 2.00 |
| attacker:xp_lost_B@100 | +26.31 ± 18.83 |
| s0att:away_B@100 | -0.23 ± 3.70 |
| s0att:crash_hp_B@100 | +4.73 ± 23.00 |
| s0att:d_total_A@100 | -0.21 ± 1.09 |
| s0att:deaths_A@100 | -0.06 ± 0.28 |
| s0att:deaths_B@100 | -0.08 ± 1.29 |
| s0att:earned_A@100 | +0.27 ± 0.47 |
| s0att:earned_B@100 | -0.79 ± 1.77 |
| s0att:farm_A@100 | +0.35 ± 2.27 |
| s0att:held_A@100 | +0.90 ± 1.99 |
| s0att:n_safe_B@100 | +6.67 ± 22.48 |
| s0att:plates_A@100 | +0.09 ± 0.19 |
| s0att:recalls_B@100 | +0.06 ± 1.16 |
| s0att:trade_att_A@100 | +1.04 ± 8.25 |
| s0att:trade_conn_A@100 | +1.48 ± 4.92 |
| s0att:unc_B@100 | +0.70 ± 1.77 |
| s0att:unc_away_B@100 | +0.07 ± 2.12 |
| s0att:xp_lost_B@100 | +0.92 ± 23.71 |
| target_abs:away_B@100 | +13.79 |
| target_abs:deaths_B@100 | +1.45 |
| target_abs:earned_A@100 | +43.49 |
| target_abs:earned_B@100 | +37.80 |
| target_abs:held_A@100 | +5.19 |
| target_abs:trade_conn_A@100 | +14.90 |
| target_abs:unc_B@100 | +13.99 |


### E4gap_semi

| contrast | pooled | targets significant | per target (* = significant) |
|---|---|---|---|
| attacker-s0att|G100 | +1.17 ± 0.76 | 3+/0− | s0: +1.08 ± 0.50*; s2: +1.51 ± 0.50*; s4: +0.92 ± 0.50* |
| attacker-s0att|G150 | +1.23 ± 0.82 | 3+/0− | s0: +1.14 ± 0.51*; s2: +1.60 ± 0.57*; s4: +0.95 ± 0.50* |
| attacker-s0att|G20 | +0.45 ± 0.53 | 2+/0− | s0: +0.21 ± 0.28; s2: +0.56 ± 0.28*; s4: +0.58 ± 0.41* |
| attacker-s0att|G50 | +0.89 ± 0.34 | 3+/0− | s0: +0.75 ± 0.39*; s2: +1.02 ± 0.44*; s4: +0.89 ± 0.48* |
| attacker-target|G100 | +1.83 ± 0.34 | 3+/0− | s0: +1.96 ± 0.48*; s2: +1.68 ± 0.60*; s4: +1.84 ± 0.46* |
| attacker-target|G150 | +2.14 ± 0.32 | 3+/0− | s0: +2.23 ± 0.46*; s2: +1.99 ± 0.64*; s4: +2.19 ± 0.48* |
| attacker-target|G20 | +0.69 ± 0.07 | 3+/0− | s0: +0.69 ± 0.30*; s2: +0.72 ± 0.31*; s4: +0.67 ± 0.28* |
| attacker-target|G50 | +1.30 ± 0.28 | 3+/0− | s0: +1.42 ± 0.37*; s2: +1.19 ± 0.51*; s4: +1.30 ± 0.38* |
| s0att-target|G100 | +0.65 ± 1.04 | 2+/0− | s0: +0.88 ± 0.43*; s2: +0.17 ± 0.52; s4: +0.92 ± 0.41* |
| s0att-target|G150 | +0.91 ± 1.12 | 2+/0− | s0: +1.09 ± 0.45*; s2: +0.39 ± 0.55; s4: +1.24 ± 0.35* |
| s0att-target|G20 | +0.24 ± 0.52 | 1+/0− | s0: +0.48 ± 0.28*; s2: +0.15 ± 0.28; s4: +0.08 ± 0.27 |
| s0att-target|G50 | +0.42 ± 0.62 | 2+/0− | s0: +0.67 ± 0.36*; s2: +0.17 ± 0.43; s4: +0.40 ± 0.32* |

| horizon | target | absolute: target in seat A | attacker in seat A | s0-attacker in seat A |
|---|---|---|---|---|
| G100 | 0 | +3.18 | +5.14 | +4.05 |
| G100 | 2 | +4.01 | +5.70 | +4.18 |
| G100 | 4 | +3.72 | +5.56 | +4.64 |
| G150 | 0 | +3.27 | +5.50 | +4.36 |
| G150 | 2 | +4.12 | +6.11 | +4.51 |
| G150 | 4 | +3.84 | +6.03 | +5.08 |
| G20 | 0 | +1.49 | +2.17 | +1.97 |
| G20 | 2 | +1.62 | +2.34 | +1.78 |
| G20 | 4 | +1.82 | +2.49 | +1.91 |
| G50 | 0 | +2.58 | +4.00 | +3.25 |
| G50 | 2 | +3.12 | +4.32 | +3.30 |
| G50 | 4 | +3.07 | +4.36 | +3.47 |

| channel (candidate − target, or the target's absolute level) | value |
|---|---|
| attacker:away_B@100 | +3.96 ± 3.01 |
| attacker:crash_hp_B@100 | +7.13 ± 13.76 |
| attacker:d_total_A@100 | +0.52 ± 0.60 |
| attacker:deaths_A@100 | +0.38 ± 0.15 |
| attacker:deaths_B@100 | +1.17 ± 1.04 |
| attacker:earned_A@100 | -0.26 ± 0.59 |
| attacker:earned_B@100 | -2.96 ± 1.11 |
| attacker:farm_A@100 | -2.56 ± 2.71 |
| attacker:held_A@100 | +1.49 ± 2.48 |
| attacker:n_safe_B@100 | +3.27 ± 6.17 |
| attacker:plates_A@100 | -0.04 ± 0.07 |
| attacker:recalls_B@100 | -0.73 ± 1.06 |
| attacker:trade_att_A@100 | +12.82 ± 8.23 |
| attacker:trade_conn_A@100 | +10.71 ± 7.04 |
| attacker:unc_B@100 | +3.53 ± 1.24 |
| attacker:unc_away_B@100 | +2.50 ± 1.98 |
| attacker:xp_lost_B@100 | +25.00 ± 19.51 |
| s0att:away_B@100 | -0.23 ± 4.33 |
| s0att:crash_hp_B@100 | +6.25 ± 24.52 |
| s0att:d_total_A@100 | -0.20 ± 1.16 |
| s0att:deaths_A@100 | -0.03 ± 0.32 |
| s0att:deaths_B@100 | -0.05 ± 1.40 |
| s0att:earned_A@100 | +0.26 ± 0.34 |
| s0att:earned_B@100 | -0.80 ± 1.97 |
| s0att:farm_A@100 | +0.28 ± 2.53 |
| s0att:held_A@100 | +0.95 ± 1.69 |
| s0att:n_safe_B@100 | +7.06 ± 23.49 |
| s0att:plates_A@100 | +0.09 ± 0.19 |
| s0att:recalls_B@100 | +0.00 ± 1.15 |
| s0att:trade_att_A@100 | +1.07 ± 9.99 |
| s0att:trade_conn_A@100 | +1.54 ± 5.21 |
| s0att:unc_B@100 | +0.83 ± 1.89 |
| s0att:unc_away_B@100 | +0.13 ± 2.40 |
| s0att:xp_lost_B@100 | +1.57 ± 26.88 |
| target_abs:away_B@100 | +13.82 |
| target_abs:deaths_B@100 | +1.41 |
| target_abs:earned_A@100 | +42.85 |
| target_abs:earned_B@100 | +37.84 |
| target_abs:held_A@100 | +5.15 |
| target_abs:trade_conn_A@100 | +14.78 |
| target_abs:unc_B@100 | +14.01 |


From s0 against the target (evaluate_discounted, 64 paired seeds, W = 460):

| attacker | value | seat A | seat B |
|---|---|---|---|
| freeze_s0 | +0.63 ± 0.59 | +0.54 | +0.72 |
| freeze_s2 | +0.55 ± 0.57 | +0.72 | +0.39 |
| freeze_s4 | +1.83 ± 0.61 | +1.53 | +2.13 |
| gap_s0 | +0.11 ± 0.62 | +0.27 | -0.05 |
| gap_s2 | +0.04 ± 0.65 | +0.48 | -0.40 |
| gap_s4 | -0.06 ± 0.84 | -0.14 | +0.01 |
| s0att_s0 | +0.79 ± 0.52 | +0.63 | +0.96 |
| s0att_s2 | +0.50 ± 0.44 | +0.63 | +0.37 |
| s0att_s4 | +1.82 ± 0.41 | +1.76 | +1.88 |

## E6 — credit assignment

| family | opponent | plan | states | mean effect (G60) | SD of one paired rollout | |effect| / SD | forced turns | effect per forced turn | paired rollouts for 2 SE |
|---|---|---|---|---|---|---|---|
| E2c|g_wary|bully_beh | 144 | +0.07 | 1.36 | 0.051 | 17.4 | +0.004 | 1510 |
| E2c|g_wary|edge_beh | 144 | -0.02 | 1.31 | 0.012 | 17.5 | -0.001 | 26111 |
| E2c|g_wary|farm_beh | 144 | -0.05 | 1.17 | 0.040 | 17.7 | -0.003 | 2459 |
| E2c|g_wary|freeze_threat | 144 | +0.16 | 1.35 | 0.116 | 19.0 | +0.008 | 299 |
| E2c|g_wary|g_bully_script | 144 | -0.48 | 1.48 | 0.320 | 0.0 | n/a | 39 |
| E2c|g_wary|g_farm_script | 144 | +0.20 | 1.38 | 0.146 | 0.0 | n/a | 187 |
| E2c|g_wary|threaten_beh | 144 | +0.02 | 1.34 | 0.013 | 17.5 | +0.001 | 25238 |
| E2c|policy|bully_beh | 144 | -0.44 | 3.65 | 0.121 | 16.1 | -0.028 | 273 |
| E2c|policy|edge_beh | 144 | -0.34 | 3.50 | 0.097 | 16.6 | -0.020 | 429 |
| E2c|policy|farm_beh | 144 | -1.02 | 3.37 | 0.303 | 17.3 | -0.059 | 44 |
| E2c|policy|freeze_threat | 144 | -0.75 | 3.65 | 0.206 | 17.3 | -0.043 | 95 |
| E2c|policy|g_bully_script | 144 | -0.88 | 3.75 | 0.235 | 0.0 | n/a | 72 |
| E2c|policy|g_farm_script | 144 | -2.88 | 3.67 | 0.784 | 0.0 | n/a | 7 |
| E2c|policy|threaten_beh | 144 | -0.44 | 3.57 | 0.123 | 16.3 | -0.027 | 264 |
| E2n|g_wary|bully_beh | 144 | -0.64 | 1.66 | 0.386 | 17.1 | -0.037 | 27 |
| E2n|g_wary|edge_beh | 144 | -0.63 | 1.66 | 0.383 | 17.3 | -0.037 | 27 |
| E2n|g_wary|farm_beh | 144 | -0.35 | 1.61 | 0.215 | 17.7 | -0.020 | 86 |
| E2n|g_wary|freeze_threat | 144 | -0.95 | 1.68 | 0.566 | 18.2 | -0.052 | 12 |
| E2n|g_wary|g_bully_script | 144 | -1.92 | 1.76 | 1.091 | 0.0 | n/a | 3 |
| E2n|g_wary|g_farm_script | 144 | -0.92 | 1.74 | 0.530 | 0.0 | n/a | 14 |
| E2n|g_wary|threaten_beh | 144 | -0.67 | 1.63 | 0.411 | 17.2 | -0.039 | 24 |
| E2n|policy|bully_beh | 144 | -1.04 | 3.51 | 0.296 | 14.7 | -0.071 | 46 |
| E2n|policy|edge_beh | 144 | -0.92 | 3.51 | 0.262 | 15.0 | -0.061 | 58 |
| E2n|policy|farm_beh | 144 | -0.74 | 3.41 | 0.218 | 15.8 | -0.047 | 84 |
| E2n|policy|freeze_threat | 144 | -1.51 | 3.43 | 0.439 | 16.1 | -0.094 | 21 |
| E2n|policy|g_bully_script | 144 | -1.74 | 3.53 | 0.494 | 0.0 | n/a | 16 |
| E2n|policy|g_farm_script | 144 | -1.29 | 3.35 | 0.386 | 0.0 | n/a | 27 |
| E2n|policy|threaten_beh | 144 | -0.92 | 3.49 | 0.265 | 14.9 | -0.062 | 57 |
| E2s|g_wary|bully_beh | 144 | -0.46 | 1.50 | 0.305 | 16.4 | -0.028 | 43 |
| E2s|g_wary|edge_beh | 144 | -0.44 | 1.47 | 0.298 | 16.5 | -0.027 | 45 |
| E2s|g_wary|farm_beh | 144 | -0.23 | 1.40 | 0.166 | 16.9 | -0.014 | 145 |
| E2s|g_wary|freeze_threat | 144 | -0.42 | 1.58 | 0.268 | 17.5 | -0.024 | 56 |
| E2s|g_wary|g_bully_script | 144 | -0.83 | 1.81 | 0.458 | 0.0 | n/a | 19 |
| E2s|g_wary|g_farm_script | 144 | -0.23 | 1.62 | 0.143 | 0.0 | n/a | 194 |
| E2s|g_wary|threaten_beh | 144 | -0.46 | 1.48 | 0.313 | 16.4 | -0.028 | 41 |
| E2s|policy|bully_beh | 144 | -0.74 | 3.75 | 0.197 | 14.5 | -0.051 | 103 |
| E2s|policy|edge_beh | 144 | -0.78 | 3.62 | 0.215 | 15.0 | -0.052 | 86 |
| E2s|policy|farm_beh | 144 | -1.08 | 3.57 | 0.302 | 15.5 | -0.070 | 44 |
| E2s|policy|freeze_threat | 144 | -1.22 | 3.79 | 0.322 | 15.5 | -0.079 | 39 |
| E2s|policy|g_bully_script | 144 | -1.52 | 3.87 | 0.392 | 0.0 | n/a | 26 |
| E2s|policy|g_farm_script | 144 | -3.58 | 3.72 | 0.962 | 0.0 | n/a | 4 |
| E2s|policy|threaten_beh | 144 | -0.75 | 3.68 | 0.204 | 14.8 | -0.051 | 96 |


Critic error on E3 games (V(s) − 100-turn bootstrapped return), by the seat's turn class:

| final | turn class | n | bias | MAE | RMSE | SD of target | explained variance |
|---|---|---|---|---|---|---|---|
| 0 | trade_conn | 79018 | -0.11 | 2.84 | 3.66 | 3.76 | 0.05 |
| 0 | trade_unconn | 49851 | -0.22 | 2.63 | 3.41 | 3.53 | 0.08 |
| 0 | other | 447131 | +0.07 | 2.60 | 3.37 | 3.49 | 0.07 |
| 1 | trade_conn | 63594 | -0.10 | 2.25 | 2.97 | 3.04 | 0.05 |
| 1 | trade_unconn | 49583 | -0.15 | 2.12 | 2.81 | 2.93 | 0.08 |
| 1 | other | 462823 | +0.07 | 2.09 | 2.77 | 2.88 | 0.08 |
| 2 | trade_conn | 89387 | -0.12 | 2.87 | 3.71 | 3.69 | -0.01 |
| 2 | trade_unconn | 51840 | -0.27 | 2.68 | 3.45 | 3.46 | 0.01 |
| 2 | other | 434773 | -0.00 | 2.62 | 3.39 | 3.42 | 0.02 |
| 3 | trade_conn | 84928 | -0.03 | 2.81 | 3.64 | 3.67 | 0.02 |
| 3 | trade_unconn | 48994 | -0.19 | 2.63 | 3.42 | 3.51 | 0.05 |
| 3 | other | 442078 | +0.05 | 2.59 | 3.38 | 3.45 | 0.04 |
| 4 | trade_conn | 77099 | -0.15 | 2.89 | 3.73 | 3.67 | -0.03 |
| 4 | trade_unconn | 50042 | -0.27 | 2.61 | 3.38 | 3.37 | 0.00 |
| 4 | other | 448859 | +0.10 | 2.59 | 3.37 | 3.35 | -0.01 |
| 5 | trade_conn | 77974 | -0.14 | 2.60 | 3.39 | 3.44 | 0.03 |
| 5 | trade_unconn | 49199 | -0.20 | 2.39 | 3.11 | 3.20 | 0.06 |
| 5 | other | 448827 | +0.06 | 2.35 | 3.09 | 3.14 | 0.04 |


Critic error by power gap and role (EXPLORATORY; finals and levels pooled):

| gap | role | turn class | n | bias | RMSE | explained variance |
|---|---|---|---|---|
| equal|equal|other | 535245 | +0.02 | 2.79 | 0.125 |
| equal|equal|trade_conn | 95788 | -0.04 | 3.03 | 0.072 |
| equal|equal|trade_unconn | 60167 | -0.02 | 2.88 | 0.104 |
| x1.24|stronger|other | 266417 | -0.60 | 2.85 | 0.124 |
| x1.24|stronger|trade_conn | 47879 | -0.73 | 3.14 | 0.074 |
| x1.24|stronger|trade_unconn | 31304 | -0.59 | 2.96 | 0.111 |
| x1.24|weaker|other | 270613 | +0.64 | 2.88 | 0.125 |
| x1.24|weaker|trade_conn | 46829 | +0.71 | 3.09 | 0.082 |
| x1.24|weaker|trade_unconn | 28158 | +0.60 | 2.93 | 0.097 |
| x1.5|stronger|other | 265909 | -0.98 | 3.01 | 0.124 |
| x1.5|stronger|trade_conn | 47640 | -1.23 | 3.38 | 0.088 |
| x1.5|stronger|trade_unconn | 32051 | -0.98 | 3.11 | 0.129 |
| x1.5|weaker|other | 273296 | +1.05 | 3.06 | 0.130 |
| x1.5|weaker|trade_conn | 45576 | +1.13 | 3.28 | 0.083 |
| x1.5|weaker|trade_unconn | 26728 | +0.93 | 3.07 | 0.098 |
| x1.76|stronger|other | 263543 | -1.33 | 3.21 | 0.119 |
| x1.76|stronger|trade_conn | 48244 | -1.62 | 3.62 | 0.090 |
| x1.76|stronger|trade_unconn | 33813 | -1.31 | 3.32 | 0.141 |
| x1.76|weaker|other | 274532 | +1.39 | 3.26 | 0.130 |
| x1.76|weaker|trade_conn | 45376 | +1.52 | 3.49 | 0.084 |
| x1.76|weaker|trade_unconn | 25692 | +1.25 | 3.24 | 0.085 |
| x2.2|stronger|other | 256626 | -2.60 | 4.03 | 0.108 |
| x2.2|stronger|trade_conn | 50595 | -3.08 | 4.63 | 0.098 |
| x2.2|stronger|trade_unconn | 38379 | -2.55 | 4.06 | 0.142 |
| x2.2|weaker|other | 278310 | +2.66 | 4.10 | 0.126 |
| x2.2|weaker|trade_conn | 44073 | +2.87 | 4.36 | 0.086 |
| x2.2|weaker|trade_unconn | 23217 | +2.37 | 3.89 | 0.085 |

## Exploratory measurements (added after pre-registered results; labelled)

### Position readout of the M4.5c power-gap stage (Ward node W = 99 included vs lane cells only)

| policy | stronger turns | stronger: M4.5c metric | stronger: lane cells | stronger at W | weaker: M4.5c metric | weaker: lane cells | weaker at W |
|---|---|---|---|---|---|---|---|
| final:ext:0 | 1083 | +2.49 | -0.16 | 0.027 | +0.43 | -1.04 | 0.015 |
| final:ext:1 | 1008 | +1.82 | -0.15 | 0.020 | -0.28 | -1.07 | 0.008 |
| final:ext:2 | 1059 | +0.87 | -0.16 | 0.010 | -0.93 | -1.03 | 0.001 |
| final:ext:3 | 979 | +1.96 | -0.27 | 0.022 | +0.31 | -0.91 | 0.012 |
| final:ext:4 | 1208 | +1.33 | -0.23 | 0.016 | -0.35 | -1.01 | 0.007 |
| final:ext:5 | 1084 | +1.83 | -0.18 | 0.020 | -0.25 | -0.99 | 0.007 |
| final:full:0 | 1001 | +0.52 | -0.07 | 0.006 | -0.28 | -1.08 | 0.008 |
| final:full:1 | 1088 | +1.12 | -0.16 | 0.013 | -0.44 | -0.99 | 0.006 |
| final:full:2 | 1093 | +1.55 | -0.17 | 0.017 | -0.55 | -1.01 | 0.005 |
| final:full:3 | 1099 | +0.90 | -0.18 | 0.011 | +0.03 | -0.98 | 0.010 |
| final:full:4 | 1069 | +2.50 | -0.19 | 0.027 | +0.08 | -1.04 | 0.011 |
| final:full:5 | 999 | +0.68 | -0.22 | 0.009 | -0.77 | -0.97 | 0.002 |


### Natural power gaps in s0 self-play (16 games × 600 turns per final)

| final | turns (both present) | median ratio | 99% | 99.9% | share ≥ 1.25 | share ≥ 1.5 | mean ratio by |gold lead| 0-3 / 3-6 / 6-10 / 10-20 / 20+ | share ≥ 1.25 by turns 1-50 / 51-150 / 151-300 / 301-600 |
|---|---|---|---|---|---|---|---|---|
| 0 | 8304 | 1.006 | 1.185 | 1.273 | 0.0017 | 0.0000 | 1.025 / 1.016 / 1.008 / 1.016 / 1.014 | 0.0174 / 0.0015 / 0.0000 / 0.0000 |
| 1 | 8296 | 1.006 | 1.189 | 1.311 | 0.0037 | 0.0000 | 1.026 / 1.016 / 1.015 / 1.017 / 1.012 | 0.0253 / 0.0095 / 0.0000 / 0.0000 |
| 2 | 8168 | 1.006 | 1.158 | 1.270 | 0.0012 | 0.0000 | 1.022 / 1.020 / 1.016 / 1.012 / nan | 0.0155 / 0.0000 / 0.0000 / 0.0000 |
| 3 | 8305 | 1.006 | 1.171 | 1.254 | 0.0014 | 0.0000 | 1.022 / 1.018 / 1.020 / 1.017 / 1.019 | 0.0172 / 0.0000 / 0.0000 / 0.0000 |
| 4 | 8244 | 1.006 | 1.178 | 1.292 | 0.0019 | 0.0000 | 1.022 / 1.022 / 1.016 / 1.011 / 1.006 | 0.0238 / 0.0000 / 0.0000 / 0.0000 |
| 5 | 8343 | 1.007 | 1.175 | 1.238 | 0.0007 | 0.0000 | 1.021 / 1.020 / 1.024 / 1.015 / 1.012 | 0.0090 / 0.0000 / 0.0000 / 0.0000 |


### transfer_gap: attackers against the finals they never trained against (G100, attacker − that final's own play in seat A)

| attacker | mean over the 5 opponents | opponents significant | per opponent |
|---|---|---|---|
| attacker of s0 | +2.08 ± 0.58 | 5+/0− | s1: +2.65 ± 0.69; s2: +1.96 ± 0.78; s3: +2.38 ± 0.74; s4: +2.00 ± 0.71; s5: +1.41 ± 0.74 |
| attacker of s2 | +1.78 ± 0.76 | 5+/0− | s0: +1.42 ± 0.74; s1: +2.33 ± 0.70; s3: +2.53 ± 0.68; s4: +1.52 ± 0.76; s5: +1.11 ± 0.65 |
| attacker of s4 | +2.34 ± 0.61 | 5+/0− | s0: +2.07 ± 0.69; s1: +2.86 ± 0.82; s2: +1.61 ± 0.83; s3: +2.61 ± 0.80; s5: +2.52 ± 0.67 |

Channels (100 turns, pooled): earned_A@100 -0.35, earned_B@100 -3.29, deaths_B@100 +1.63, away_B@100 +5.93, trade_conn_A@100 +13.57, deaths_A@100 +0.51


### transfer_freeze: attackers against the finals they never trained against (G100, attacker − that final's own play in seat A)

| attacker | mean over the 5 opponents | opponents significant | per opponent |
|---|---|---|---|
| attacker of s0 | +0.78 ± 0.65 | 2+/0− | s1: +1.13 ± 0.82; s2: +0.45 ± 1.16; s3: +1.53 ± 0.85; s4: +0.35 ± 0.81; s5: +0.43 ± 1.19 |
| attacker of s2 | +1.05 ± 0.49 | 3+/0− | s0: +0.67 ± 0.76; s1: +1.18 ± 0.61; s3: +1.35 ± 0.99; s4: +0.61 ± 0.82; s5: +1.46 ± 1.08 |
| attacker of s4 | +1.04 ± 0.18 | 4+/0− | s0: +0.99 ± 0.75; s1: +0.95 ± 0.66; s2: +1.01 ± 0.97; s3: +0.95 ± 1.10; s5: +1.30 ± 0.94 |

Channels (100 turns, pooled): earned_A@100 -0.52, earned_B@100 -1.80, deaths_B@100 +0.60, away_B@100 +2.60, trade_conn_A@100 +8.36, deaths_A@100 +0.23
