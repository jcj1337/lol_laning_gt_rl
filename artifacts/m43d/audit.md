# M4.3d audit (econ: the M4.3c candidate m4_four_actions_econ() (own tower 0.8))

## Gank occurrence per turn (exposure cell, own coordinates)

| behaviour | vision | -2 | -1 | 0 | +1 | +2 |
|---|---|---|---|---|---|---|
| Hold, Last Hit | no | 0.010 | 0.040 | 0.070 | 0.100 | 0.130 |
| Hold, Last Hit | yes | 0.003 | 0.010 | 0.017 | 0.025 | 0.033 |
| Hold, Shove | no | 0.030 | 0.060 | 0.090 | 0.120 | 0.150 |
| Hold, Shove | yes | 0.007 | 0.015 | 0.022 | 0.030 | 0.037 |
| Trade, Shove | no | 0.050 | 0.080 | 0.110 | 0.140 | 0.170 |
| Trade, Shove | yes | 0.013 | 0.020 | 0.028 | 0.035 | 0.042 |
| Trade, Last Hit | no | 0.030 | 0.060 | 0.090 | 0.120 | 0.150 |
| Trade, Last Hit | yes | 0.007 | 0.015 | 0.022 | 0.030 | 0.037 |
| Yield, Last Hit | no | 0.010 | 0.040 | 0.070 | 0.100 | 0.130 |
| Yield, Last Hit | yes | 0.003 | 0.010 | 0.017 | 0.025 | 0.033 |
| Safe (any wave) | no | 0.000 | 0.030 | 0.060 | 0.090 | 0.120 |
| Safe (any wave) | yes | 0.000 | 0.007 | 0.015 | 0.022 | 0.030 |

## Gank damage given a gank (Binomial(4, p_e) HP units of 1/6 bar)

| defensive | exposure | p_e | E[units] | P(4 units) | P(>=3) | E[bars] |
|---|---|---|---|---|---|---|
| False | -2 | 0.20 | 0.80 | 0.002 | 0.027 | 0.133 |
| False | -1 | 0.35 | 1.40 | 0.015 | 0.126 | 0.233 |
| False | 0 | 0.50 | 2.00 | 0.062 | 0.312 | 0.333 |
| False | 1 | 0.65 | 2.60 | 0.179 | 0.563 | 0.433 |
| False | 2 | 0.80 | 3.20 | 0.410 | 0.819 | 0.533 |
| True | -2 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 |
| True | -1 | 0.15 | 0.60 | 0.001 | 0.012 | 0.100 |
| True | 0 | 0.30 | 1.20 | 0.008 | 0.084 | 0.200 |
| True | 1 | 0.45 | 1.80 | 0.041 | 0.241 | 0.300 |
| True | 2 | 0.60 | 2.40 | 0.130 | 0.475 | 0.400 |

## Expected gank damage per turn, bars (M = 1)

| behaviour | vision | -2 | -1 | 0 | +1 | +2 |
|---|---|---|---|---|---|---|
| Hold, Last Hit | no | 0.001 | 0.009 | 0.023 | 0.043 | 0.069 |
| Hold, Last Hit | yes | 0.000 | 0.002 | 0.006 | 0.011 | 0.017 |
| Hold, Shove | no | 0.004 | 0.014 | 0.030 | 0.052 | 0.080 |
| Hold, Shove | yes | 0.001 | 0.003 | 0.007 | 0.013 | 0.020 |
| Trade, Shove | no | 0.007 | 0.019 | 0.037 | 0.061 | 0.091 |
| Trade, Shove | yes | 0.002 | 0.005 | 0.009 | 0.015 | 0.023 |
| Trade, Last Hit | no | 0.004 | 0.014 | 0.030 | 0.052 | 0.080 |
| Trade, Last Hit | yes | 0.001 | 0.003 | 0.007 | 0.013 | 0.020 |
| Yield, Last Hit | no | 0.001 | 0.009 | 0.023 | 0.043 | 0.069 |
| Yield, Last Hit | yes | 0.000 | 0.002 | 0.006 | 0.011 | 0.017 |
| Safe (any wave) | no | 0.000 | 0.003 | 0.012 | 0.027 | 0.048 |
| Safe (any wave) | yes | 0.000 | 0.001 | 0.003 | 0.007 | 0.012 |

## Trade hit probability on a defender

| defender | outside its tower | in its tower cell |
|---|---|---|
| Hold | 0.55 | 0.40 |
| Trade | 0.55 | 0.40 |
| Yield caught (evading) | 0.30 | 0.15 |
| Safe | 0.15 | 0.05 |
| interrupted Recall | 0.55 | 0.40 |

## Tower shots: dive 1.0 (0.5 while the diver's minions are under the tower), next to the tower 0.5 (0.25); 1 HP unit = 0.166 bars; only a connected trader is ever shot

## Collection rate of a kill (share of farm_per_hp)

| own-tower rate | Hold at -2, kill at -2 | Hold at -1, kill at -2 | Safe at -2, kill at -2 | Safe at -1, kill at -2 | Safe at -1, kill at -1 | trading at -2, kill at -2 | retreat at -2, kill at -1 | Hold at 0, kill at 0 | Hold at +2, kill at +3 |
|---|---|---|---|---|---|---|---|---|---|
| 0.6 | 0.600 | 0.600 | 0.600 | 0.600 | 0.600 | 0.480 | 0.600 | 0.900 | 0.900 |
| 0.8 | 0.800 | 0.800 | 0.800 | 0.800 | 0.600 | 0.640 | 0.600 | 0.900 | 0.900 |
| 0.85 | 0.850 | 0.850 | 0.800 | 0.800 | 0.600 | 0.680 | 0.600 | 0.900 | 0.900 |
| 0.9 | 0.900 | 0.900 | 0.800 | 0.800 | 0.600 | 0.720 | 0.600 | 0.900 | 0.900 |

## Hazard of standing at one cell, per 40 turns, level 10, M = 1 (no investment)

| case | turns at the cell | ganks | bars lost | forced Recalls | deaths | turns away |
|---|---|---|---|---|---|---|
| own tower cell, Hold, Last Hit | 40.0 | 0.40 | 0.05 | 0.01 | 0.00 | 0.0 |
| own -1, Hold, Last Hit | 39.5 | 1.58 | 0.37 | 0.24 | 0.00 | 0.5 |
| lane cell, Hold, Last Hit | 37.5 | 2.67 | 0.88 | 0.79 | 0.04 | 1.7 |
| lane cell, Hold, Shove | 36.6 | 3.37 | 1.11 | 1.05 | 0.05 | 2.3 |
| enemy -1 (own +1), Hold, Shove | 34.2 | 4.25 | 1.82 | 1.62 | 0.24 | 4.1 |
| enemy tower cell, Hold, Shove (Pressure) | 29.8 | 4.83 | 2.50 | 1.88 | 0.64 | 6.1 |
| enemy tower cell, Hold, Shove, Vision always | 37.2 | 1.41 | 0.75 | 0.59 | 0.11 | 1.6 |
| enemy tower cell, Trade, Shove (dive, tower busy) | 22.3 | 4.46 | 4.09 | 3.43 | 0.90 | 10.2 |
| enemy tower cell, Trade, Shove (dive, tower free) | 18.2 | 3.94 | 4.97 | 4.17 | 1.16 | 12.6 |
| enemy tower cell, dive (busy), Vision always | 26.4 | 1.26 | 2.84 | 3.10 | 0.26 | 7.1 |

## Hazard of standing at one cell, per 40 turns, level 10, M = 1.76 (50 invested)

| case | turns at the cell | ganks | bars lost | forced Recalls | deaths | turns away |
|---|---|---|---|---|---|---|
| own tower cell, Hold, Last Hit | 40.0 | 0.40 | 0.03 | 0.00 | 0.00 | 0.0 |
| own -1, Hold, Last Hit | 39.9 | 1.60 | 0.21 | 0.04 | 0.00 | 0.1 |
| lane cell, Hold, Last Hit | 39.0 | 2.74 | 0.52 | 0.34 | 0.00 | 0.7 |
| lane cell, Hold, Shove | 38.4 | 3.49 | 0.66 | 0.52 | 0.00 | 1.0 |
| enemy -1 (own +1), Hold, Shove | 36.7 | 4.50 | 1.10 | 1.06 | 0.03 | 2.2 |
| enemy tower cell, Hold, Shove (Pressure) | 33.3 | 5.30 | 1.57 | 1.51 | 0.11 | 3.4 |
| enemy tower cell, Hold, Shove, Vision always | 38.8 | 1.46 | 0.44 | 0.29 | 0.00 | 0.6 |
| enemy tower cell, Trade, Shove (dive, tower busy) | 27.2 | 5.19 | 2.80 | 2.84 | 0.29 | 6.7 |
| enemy tower cell, Trade, Shove (dive, tower free) | 23.2 | 4.66 | 3.53 | 3.73 | 0.39 | 8.8 |
| enemy tower cell, dive (busy), Vision always | 31.1 | 1.41 | 1.89 | 2.14 | 0.05 | 4.4 |
