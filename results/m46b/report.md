# M4.6b generated tables

Interpretation: docs/m4/m4_6/m4_6b_investment_report.md.

Arms: baseline: current: saturating A 1.5, K 16; C1: C1 slow saturation: A 3.24, K 64; C2: C2 slower saturation: A 9.53, K 256; C3: C3 linear reference: 1 + 0.016 S

## Pre-registered scorecard (D1–D4)

| arm | D1 consequential leads (+8) | D2 not decisive | D3 equal lane (vs baseline) | D4 waves and towers (S1 stage states) | rule |
|---|---|---|---|---|---|
| C1 | mid +0.51 vs 1.5 × +0.33, diff +0.18 ± 0.49: fail; late -0.03 vs 1.5 × +0.24, diff -0.26 ± 0.38: fail | early +1.02 vs ≤ +0.58: fail; share ≥ 1.25 0.0007: pass; slope 1.02 / 0.94: pass | crash_hp +4%, plates150 +4%, deaths -7%, recalls -2%, collection +0%, Shove -0.006: pass | C early +0, C mid +0, C late +3 FAIL, Pressure mid +0.6%: fail | does not advance |
| C2 | mid +0.59 vs 1.5 × +0.33, diff +0.26 ± 0.54: fail; late +0.23 vs 1.5 × +0.24, diff -0.00 ± 0.49: fail | early +0.97 vs ≤ +0.58: fail; share ≥ 1.25 0.0016: pass; slope 1.19 / 1.08: pass | crash_hp +5%, plates150 +2%, deaths -13%, recalls -3%, collection +1%, Shove -0.003: pass | C early +0, C mid +0, C late +5 FAIL, Pressure mid -0.9%: fail | does not advance |
| C3 | mid +0.67 vs 1.5 × +0.33, diff +0.34 ± 0.70: fail; late +0.13 vs 1.5 × +0.24, diff -0.11 ± 0.61: fail | early +0.74 vs ≤ +0.58: fail; share ≥ 1.25 0.0015: pass; slope 1.09 / 1.12: pass | crash_hp +5%, plates150 +2%, deaths -15%, recalls -6%, collection +1%, Shove -0.007: pass | C early -1, C mid +0, C late +7 FAIL, Pressure mid -2.4%: fail | does not advance |

## Measurement fixes: verification

### V1: the M4.5c power-gap stage on its original seeds

| final | other readouts identical | stronger cell (stored → lane only) | stronger at W | even cell | weaker cell | credible-threat turns |
|---|---|---|---|---|---|---|
| final:ext:0 | 24 / 24 | +2.69 → -0.15 | 0.029 | +1.93 → -0.70 | -0.52 → -0.99 | 499 → 523 |
| final:ext:1 | 24 / 24 | +2.95 → -0.18 | 0.032 | +1.06 → -0.74 | -0.81 → -1.08 | 536 → 554 |
| final:ext:2 | 24 / 24 | +1.44 → -0.17 | 0.016 | +1.03 → -0.69 | -1.01 → -1.01 | 504 → 509 |
| final:ext:3 | 24 / 24 | +1.82 → -0.26 | 0.021 | +2.23 → -0.70 | +0.60 → -0.92 | 520 → 541 |
| final:ext:4 | 24 / 24 | +0.93 → -0.22 | 0.012 | +1.10 → -0.74 | -0.63 → -0.99 | 534 → 543 |
| final:ext:5 | 24 / 24 | +2.36 → -0.13 | 0.025 | +1.21 → -0.73 | +0.20 → -1.02 | 499 → 515 |


### V2: battery families C_trade, B_frozen, H_zone_c for finals [0, 3]

840 jobs; 840 compared with the stored M4.5c raw jobs; **840 identical** (G and Q of every rollout of every original candidate); not identical: 0.


Corrected readings (two finals; primary statistic of the family at 50 turns):

| family | opponent | contrast | mean | per final |
|---|---|---|---|
| B_frozen|fz_bully_sh | plan:shovex3-policy | -0.527 | -0.617 / -0.437 |
| B_frozen|fz_bully_sh | 1t:walk_up_beh-policy | -0.094 | -0.146 / -0.042 |
| B_frozen|fz_bully_sh | 1t:step_back_beh-policy | -0.311 | -0.247 / -0.374 |
| B_frozen|fz_bully_sh | 1t:walk_up_beh-1t:farm | +0.350 | +0.327 / +0.374 |
| B_frozen|fz_bully_sh | 1t:step_back_beh-1t:step_back | +0.000 | +0.000 / +0.000 |
| B_frozen|fz_guard_sh | plan:shovex3-policy | -0.492 | -0.711 / -0.273 |
| B_frozen|fz_guard_sh | 1t:walk_up_beh-policy | -0.125 | -0.201 / -0.049 |
| B_frozen|fz_guard_sh | 1t:step_back_beh-policy | -0.334 | -0.447 / -0.222 |
| B_frozen|fz_guard_sh | 1t:walk_up_beh-1t:farm | +0.249 | +0.233 / +0.264 |
| B_frozen|fz_guard_sh | 1t:step_back_beh-1t:step_back | +0.003 | -0.001 / +0.006 |
| B_frozen|fz_passive_sh | plan:shovex3-policy | -0.290 | -0.446 / -0.134 |
| B_frozen|fz_passive_sh | 1t:walk_up_beh-policy | -0.034 | -0.100 / +0.032 |
| B_frozen|fz_passive_sh | 1t:step_back_beh-policy | -0.464 | -0.350 / -0.577 |
| B_frozen|fz_passive_sh | 1t:walk_up_beh-1t:farm | +0.203 | +0.101 / +0.305 |
| B_frozen|fz_passive_sh | 1t:step_back_beh-1t:step_back | +0.000 | +0.004 / -0.004 |
| B_frozen|modal | plan:shovex3-policy | -0.198 | -0.206 / -0.190 |
| B_frozen|modal | 1t:walk_up_beh-policy | -0.023 | -0.076 / +0.031 |
| B_frozen|modal | 1t:step_back_beh-policy | -0.124 | -0.198 / -0.049 |
| B_frozen|modal | 1t:walk_up_beh-1t:farm | +0.148 | +0.094 / +0.203 |
| B_frozen|modal | 1t:step_back_beh-1t:step_back | +0.026 | +0.050 / +0.003 |
| B_frozen|policy | plan:shovex3-policy | -0.338 | -0.374 / -0.303 |
| B_frozen|policy | 1t:walk_up_beh-policy | -0.102 | -0.029 / -0.176 |
| B_frozen|policy | 1t:step_back_beh-policy | -0.182 | -0.180 / -0.184 |
| B_frozen|policy | 1t:walk_up_beh-1t:farm | +0.149 | +0.268 / +0.031 |
| B_frozen|policy | 1t:step_back_beh-1t:step_back | +0.013 | -0.012 / +0.037 |
| C_trade|g_wary | 1t:trade_beh-policy | -0.227 | -0.227 / -0.226 |
| C_trade|g_wary | 1t:hold_beh-policy | -0.127 | -0.099 / -0.155 |
| C_trade|g_wary | 1t:safe_beh-policy | -0.081 | -0.018 / -0.143 |
| C_trade|g_wary | 1t:trade_beh-1t:hold_beh | -0.100 | -0.128 / -0.071 |
| C_trade|g_wary | 1t:trade-1t:hold | -0.099 | -0.168 / -0.031 |
| C_trade|g_wary | 1t:trade_beh-1t:trade | -0.044 | -0.038 / -0.051 |
| C_trade|g_wary | 1t:hold_beh-1t:hold | -0.044 | -0.077 / -0.011 |
| C_trade|g_wary | 1t:safe_beh-1t:safe | -0.041 | -0.043 / -0.038 |
| C_trade|modal | 1t:trade_beh-policy | -0.399 | -0.439 / -0.359 |
| C_trade|modal | 1t:hold_beh-policy | -0.192 | -0.328 / -0.055 |
| C_trade|modal | 1t:safe_beh-policy | -0.019 | +0.053 / -0.091 |
| C_trade|modal | 1t:trade_beh-1t:hold_beh | -0.207 | -0.111 / -0.304 |
| C_trade|modal | 1t:trade-1t:hold | +0.092 | +0.097 / +0.087 |
| C_trade|modal | 1t:trade_beh-1t:trade | -0.026 | +0.085 / -0.138 |
| C_trade|modal | 1t:hold_beh-1t:hold | +0.273 | +0.293 / +0.252 |
| C_trade|modal | 1t:safe_beh-1t:safe | +0.028 | +0.049 / +0.007 |
| C_trade|policy | 1t:trade_beh-policy | -0.361 | -0.533 / -0.188 |
| C_trade|policy | 1t:hold_beh-policy | -0.279 | -0.299 / -0.259 |
| C_trade|policy | 1t:safe_beh-policy | -0.113 | -0.097 / -0.128 |
| C_trade|policy | 1t:trade_beh-1t:hold_beh | -0.082 | -0.234 / +0.071 |
| C_trade|policy | 1t:trade-1t:hold | -0.077 | -0.165 / +0.012 |
| C_trade|policy | 1t:trade_beh-1t:trade | +0.027 | -0.015 / +0.070 |
| C_trade|policy | 1t:hold_beh-1t:hold | +0.033 | +0.054 / +0.011 |
| C_trade|policy | 1t:safe_beh-1t:safe | -0.025 | -0.096 / +0.046 |
| H_zone_c|g_wary | plan:bullyx10-policy | +0.103 | -0.002 / +0.208 |
| H_zone_c|g_wary | 1t:trade_beh-policy | -0.007 | +0.037 / -0.052 |
| H_zone_c|g_wary | 1t:hold_beh-policy | -0.044 | +0.004 / -0.092 |
| H_zone_c|g_wary | plan:bully_behx10-policy | -0.034 | -0.042 / -0.027 |
| H_zone_c|g_wary | plan:threaten_behx10-policy | -0.013 | +0.032 / -0.057 |
| H_zone_c|g_wary | plan:farm_behx10-policy | -0.076 | -0.051 / -0.100 |
| H_zone_c|g_wary | plan:bully_behx10-plan:bullyx10 | -0.137 | -0.039 / -0.235 |
| H_zone_c|g_wary | plan:threaten_behx10-plan:farm_behx10 | +0.063 | +0.083 / +0.043 |
| H_zone_c|g_wary | 1t:trade_beh-1t:hold_beh | +0.037 | +0.033 / +0.041 |
| H_zone_c|g_wary | 1t:trade_beh-1t:trade | -0.147 | -0.057 / -0.237 |
| H_zone_c|modal | plan:bullyx10-policy | -0.705 | -0.450 / -0.961 |
| H_zone_c|modal | 1t:trade_beh-policy | -0.195 | -0.022 / -0.367 |
| H_zone_c|modal | 1t:hold_beh-policy | +0.016 | +0.064 / -0.032 |
| H_zone_c|modal | plan:bully_behx10-policy | -0.591 | -0.590 / -0.591 |
| H_zone_c|modal | plan:threaten_behx10-policy | -0.416 | -0.181 / -0.650 |
| H_zone_c|modal | plan:farm_behx10-policy | -0.567 | -0.411 / -0.723 |
| H_zone_c|modal | plan:bully_behx10-plan:bullyx10 | +0.115 | -0.140 / +0.369 |
| H_zone_c|modal | plan:threaten_behx10-plan:farm_behx10 | +0.151 | +0.230 / +0.073 |
| H_zone_c|modal | 1t:trade_beh-1t:hold_beh | -0.211 | -0.087 / -0.336 |
| H_zone_c|modal | 1t:trade_beh-1t:trade | -0.012 | +0.033 / -0.058 |
| H_zone_c|policy | plan:bullyx10-policy | -0.825 | -0.743 / -0.906 |
| H_zone_c|policy | 1t:trade_beh-policy | +0.159 | +0.265 / +0.053 |
| H_zone_c|policy | 1t:hold_beh-policy | +0.115 | +0.098 / +0.132 |
| H_zone_c|policy | plan:bully_behx10-policy | -0.386 | -0.540 / -0.232 |
| H_zone_c|policy | plan:threaten_behx10-policy | -0.312 | -0.443 / -0.181 |
| H_zone_c|policy | plan:farm_behx10-policy | -0.140 | -0.254 / -0.026 |
| H_zone_c|policy | plan:bully_behx10-plan:bullyx10 | +0.439 | +0.203 / +0.674 |
| H_zone_c|policy | plan:threaten_behx10-plan:farm_behx10 | -0.172 | -0.189 / -0.155 |
| H_zone_c|policy | 1t:trade_beh-1t:hold_beh | +0.044 | +0.167 / -0.079 |
| H_zone_c|policy | 1t:trade_beh-1t:trade | +0.072 | +0.141 / +0.003 |

## Audit of the current curve (s0 self-play, 6 finals × 24 games × 600 turns)

| turn | invested S (median) | unspent g | level | |earned lead| median | q90 | |invested lead| median | relative lead | mean |level gap| |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.0 | 3.1 | 3 | 0.7 | 1.9 | 0.0 | 0.220 | 0.44 |
| 25 | 4.9 | 4.5 | 6 | 1.2 | 3.1 | 3.2 | 0.130 | 0.27 |
| 50 | 14.9 | 4.6 | 9 | 2.0 | 5.3 | 4.1 | 0.100 | 0.43 |
| 75 | 25.0 | 4.8 | 12 | 2.6 | 6.5 | 4.6 | 0.087 | 0.11 |
| 100 | 35.3 | 4.5 | 14 | 3.0 | 7.3 | 3.9 | 0.077 | 0.10 |
| 150 | 55.5 | 4.4 | 18 | 4.2 | 9.5 | 5.3 | 0.069 | 0.41 |
| 200 | 75.8 | 5.0 | 18 | 4.6 | 9.8 | 5.2 | 0.056 | 0.00 |
| 300 | 116.0 | 4.8 | 18 | 5.4 | 11.2 | 5.6 | 0.044 | 0.00 |
| 400 | 154.1 | 4.9 | 18 | 5.8 | 12.3 | 7.5 | 0.036 | 0.00 |
| 600 | 233.9 | 4.4 | 18 | 5.8 | 15.9 | 8.1 | 0.024 | 0.00 |


Discount-weighted invested gold (weight 0.99^t): {'0.25': 5.7675, '0.5': 22.39, '0.75': 50.0725, '0.9': 86.6775}; income per turn: {'1-50': 0.402, '50-150': 0.409, '150-300': 0.4, '300-600': 0.393}


### From a gold lead to strength (oriented to the gold leader; geometric means over present turns)

| stage | mean S | mean |lead| | relative gold lead | elasticity | M ratio if all earned were invested | M ratio (actual) | level factor | attack ratio | M4.4 'power' | duel ratio | unspent lead | leader more invested |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| early | 8.8 | 1.8 | ×1.1499 | 0.165 | ×1.0248 | ×0.9900 | ×1.0112 | ×1.0011 | ×1.0124 | ×1.0023 | +1.91 | 0.47 |
| mid | 43.1 | 4.0 | ×1.0867 | 0.165 | ×1.0135 | ×1.0126 | ×1.0093 | ×1.0221 | ×1.0316 | ×1.0447 | +0.69 | 0.72 |
| late | 107.3 | 5.7 | ×1.0517 | 0.120 | ×1.0060 | ×1.0063 | ×1.0000 | ×1.0063 | ×1.0063 | ×1.0127 | +0.11 | 0.80 |
| very late | 203.8 | 7.0 | ×1.0343 | 0.090 | ×1.0030 | ×1.0029 | ×1.0000 | ×1.0029 | ×1.0029 | ×1.0059 | +0.42 | 0.82 |


### M4.6a's binning by |earned lead|, re-derived

| |lead| bin | turns | median turn | median S | M4.4 power ratio (max(r, 1/r)) | share after turn 150 |
|---|---|---|---|---|---|
| 0-3 | 27914 | 201.0 | 77 | 1.0256 | 0.59 |
| 3-6 | 20045 | 306.0 | 117 | 1.0179 | 0.77 |
| 6-10 | 15519 | 325.0 | 125 | 1.0171 | 0.87 |
| 10-20 | 10417 | 418.0 | 161 | 1.0132 | 0.96 |
| 20-1000 | 821 | 489.0 | 189 | 1.0131 | 1.00 |


### What a lead buys (current curve)

| stage | typical | lead | M ratio | duel ratio | C (trailer → leader) | Pressure per turn | bars per exchange: leader on trailer / trailer on leader |
|---|---|---|---|---|---|---|---|
| early | S 9, L8 | 5 turns (2.00 res) | ×1.0353 | ×1.0719 | 7 → 8 | 729 → 755 | 0.1234 / 0.1151 (equal 0.1192) |
| early | S 9, L8 | 10 turns (4.00 res) | ×1.0669 | ×1.1384 | 7 → 8 | 729 → 778 | 0.1270 / 0.1117 (equal 0.1192) |
| early | S 9, L8 | 20 turns (8.00 res) | ×1.1212 | ×1.2572 | 7 → 8 | 729 → 818 | 0.1335 / 0.1063 (equal 0.1192) |
| early | S 9, L8 | 40 turns (16.00 res) | ×1.2048 | ×1.4515 | 7 → 9 | 729 → 879 | 0.1437 / 0.0989 (equal 0.1192) |
| mid | S 42, L16 | 5 turns (2.00 res) | ×1.0076 | ×1.0153 | 14 → 14 | 1426 → 1437 | 0.1201 / 0.1184 (equal 0.1193) |
| mid | S 42, L16 | 10 turns (4.00 res) | ×1.0149 | ×1.0300 | 14 → 14 | 1426 → 1447 | 0.1210 / 0.1175 (equal 0.1193) |
| mid | S 42, L16 | 20 turns (8.00 res) | ×1.0284 | ×1.0577 | 14 → 15 | 1426 → 1467 | 0.1226 / 0.1160 (equal 0.1193) |
| mid | S 42, L16 | 40 turns (16.00 res) | ×1.0522 | ×1.1071 | 14 → 15 | 1426 → 1501 | 0.1254 / 0.1134 (equal 0.1193) |
| late | S 108, L18 | 5 turns (2.00 res) | ×1.0022 | ×1.0044 | 18 → 18 | 1797 → 1801 | 0.1194 / 0.1189 (equal 0.1192) |
| late | S 108, L18 | 10 turns (4.00 res) | ×1.0044 | ×1.0087 | 18 → 18 | 1797 → 1805 | 0.1196 / 0.1187 (equal 0.1192) |
| late | S 108, L18 | 20 turns (8.00 res) | ×1.0085 | ×1.0171 | 18 → 18 | 1797 → 1812 | 0.1202 / 0.1182 (equal 0.1192) |
| late | S 108, L18 | 40 turns (16.00 res) | ×1.0163 | ×1.0328 | 18 → 18 | 1797 → 1826 | 0.1212 / 0.1173 (equal 0.1192) |
| very late | S 203, L18 | 5 turns (2.00 res) | ×1.0009 | ×1.0018 | 19 → 19 | 1921 → 1922 | 0.1193 / 0.1190 (equal 0.1191) |
| very late | S 203, L18 | 10 turns (4.00 res) | ×1.0017 | ×1.0035 | 19 → 19 | 1921 → 1924 | 0.1194 / 0.1189 (equal 0.1191) |
| very late | S 203, L18 | 20 turns (8.00 res) | ×1.0034 | ×1.0069 | 19 → 19 | 1921 → 1927 | 0.1196 / 0.1187 (equal 0.1191) |
| very late | S 203, L18 | 40 turns (16.00 res) | ×1.0067 | ×1.0134 | 19 → 19 | 1921 → 1933 | 0.1199 / 0.1184 (equal 0.1191) |


### The current curve

| S | M(S) | dM/dS per resource | elasticity |
|---|---|---|---|
| 0 | 1.000 | 0.0469 | 0.000 |
| 2 | 1.086 | 0.0393 | 0.072 |
| 5 | 1.191 | 0.0312 | 0.131 |
| 10 | 1.323 | 0.0226 | 0.171 |
| 16 | 1.439 | 0.0166 | 0.184 |
| 25 | 1.563 | 0.0114 | 0.183 |
| 35 | 1.660 | 0.0082 | 0.174 |
| 48 | 1.750 | 0.0059 | 0.161 |
| 64 | 1.829 | 0.0042 | 0.147 |
| 80 | 1.888 | 0.0032 | 0.135 |
| 100 | 1.943 | 0.0024 | 0.124 |
| 130 | 2.003 | 0.0017 | 0.110 |
| 160 | 2.048 | 0.0013 | 0.100 |
| 200 | 2.092 | 0.0009 | 0.090 |
| 300 | 2.162 | 0.0005 | 0.074 |
| 500 | 2.236 | 0.0003 | 0.057 |


Lead persistence: 50->100: slope 0.76, relative lead 0.122 → 0.088; 100->200: slope 0.87, relative lead 0.088 → 0.063; 200->400: slope 0.94, relative lead 0.063 → 0.042; 300->600: slope 0.98, relative lead 0.050 → 0.032


Returns: early: 5.4 per 100 turns, 7.2 invested per return; mid: 5.0 per 100 turns, 8.1 invested per return; late: 4.7 per 100 turns, 8.4 invested per return; very late: 4.7 per 100 turns, 8.5 invested per return

## S1: mapping tables per arm

| arm | stage | M | C | Pressure | Pressure turns per 2,000 | duel ratio +4 (leader C) | +8 | +16 |
|---|---|---|---|---|---|---|---|---|
| baseline | early | 1.300 | 7 | 732 | 2.73 | ×1.136 (C 8) | ×1.254 (C 8) | ×1.445 (C 9) |
| baseline | mid | 1.712 | 14 | 1424 | 1.40 | ×1.030 (C 14) | ×1.058 (C 15) | ×1.108 (C 15) |
| baseline | late | 1.961 | 18 | 1798 | 1.11 | ×1.009 (C 18) | ×1.017 (C 18) | ×1.033 (C 18) |
| C1 | early | 1.206 | 7 | 679 | 2.95 | ×1.137 (C 7) | ×1.271 (C 8) | ×1.531 (C 8) |
| C1 | mid | 1.722 | 14 | 1432 | 1.40 | ×1.054 (C 15) | ×1.107 (C 15) | ×1.208 (C 16) |
| C1 | late | 2.264 | 21 | 2075 | 0.96 | ×1.020 (C 21) | ×1.040 (C 21) | ×1.077 (C 22) |
| C2 | early | 1.163 | 7 | 655 | 3.05 | ×1.124 (C 7) | ×1.252 (C 7) | ×1.519 (C 8) |
| C2 | mid | 1.697 | 14 | 1411 | 1.42 | ×1.070 (C 15) | ×1.142 (C 15) | ×1.287 (C 16) |
| C2 | late | 2.538 | 23 | 2327 | 0.86 | ×1.035 (C 24) | ×1.069 (C 24) | ×1.139 (C 25) |
| C3 | early | 1.144 | 6 | 644 | 3.11 | ×1.115 (C 7) | ×1.236 (C 7) | ×1.498 (C 8) |
| C3 | mid | 1.672 | 14 | 1390 | 1.44 | ×1.078 (C 14) | ×1.159 (C 15) | ×1.330 (C 16) |
| C3 | late | 2.728 | 25 | 2501 | 0.80 | ×1.047 (C 26) | ×1.096 (C 26) | ×1.196 (C 27) |


M / elasticity at S = 5, 15, 35, 56, 108, 234:

| arm | 5 | 15 | 35 | 56 | 108 | 234 |
|---|---|---|---|---|---|---|
| baseline | 1.19 / 0.13 | 1.42 / 0.18 | 1.66 / 0.17 | 1.79 / 0.15 | 1.96 / 0.12 | 2.12 / 0.08 |
| C1 | 1.12 / 0.10 | 1.32 / 0.21 | 1.63 / 0.28 | 1.87 / 0.29 | 2.26 / 0.27 | 2.74 / 0.22 |
| C2 | 1.09 / 0.08 | 1.27 / 0.20 | 1.59 / 0.34 | 1.90 / 0.41 | 2.54 / 0.47 | 3.64 / 0.45 |
| C3 | 1.08 / 0.07 | 1.24 / 0.19 | 1.56 / 0.36 | 1.90 / 0.47 | 2.73 / 0.63 | 4.74 / 0.79 |


Wave capacity C at the level's typical investment (M4.5b TYPICAL_INVEST):

| arm | L4 | L6 | L8 | L10 | L13 | L15 | L18 |
|---|---|---|---|---|---|---|---|
| baseline | 5 | 7 | 8 | 10 | 13 | 14 | 18 |
| C1 | 5 | 6 | 8 | 9 | 13 | 15 | 22 |
| C2 | 5 | 6 | 7 | 9 | 13 | 16 | 25 |
| C3 | 5 | 6 | 7 | 9 | 13 | 16 | 28 |

## S2: the exact one-turn ledger (HP priced at 1.5 per bar)

Neutral situation, A ahead by 0 / +8 / +16 invested:

| stage | contrast | baseline | C1 | C2 | C3 |
|---|---|---|---|---|---|
| early | A: Trade − Hold, against Hold | +0.201/+0.230/+0.249 | +0.195/+0.227/+0.253 | +0.192/+0.222/+0.249 | +0.191/+0.219/+0.246 |
| early | A: Trade − Hold, against Safe | +0.142/+0.143/+0.144 | +0.142/+0.143/+0.143 | +0.141/+0.142/+0.143 | +0.141/+0.142/+0.143 |
| early | B: Safe − Hold, against Trade | -0.229/-0.200/-0.181 | -0.232/-0.202/-0.176 | -0.235/-0.206/-0.179 | -0.236/-0.209/-0.182 |
| mid | A: Trade − Hold, against Hold | +0.250/+0.256/+0.261 | +0.250/+0.261/+0.271 | +0.251/+0.264/+0.277 | +0.250/+0.265/+0.280 |
| mid | A: Trade − Hold, against Safe | +0.144/+0.144/+0.144 | +0.144/+0.144/+0.145 | +0.144/+0.144/+0.145 | +0.144/+0.144/+0.145 |
| mid | B: Safe − Hold, against Trade | -0.184/-0.178/-0.173 | -0.184/-0.173/-0.164 | -0.183/-0.170/-0.158 | -0.184/-0.169/-0.154 |
| late | A: Trade − Hold, against Hold | +0.257/+0.258/+0.260 | +0.259/+0.263/+0.267 | +0.261/+0.267/+0.273 | +0.262/+0.271/+0.280 |
| late | A: Trade − Hold, against Safe | +0.145/+0.145/+0.145 | +0.146/+0.146/+0.146 | +0.146/+0.146/+0.146 | +0.146/+0.147/+0.147 |
| late | B: Safe − Hold, against Trade | -0.179/-0.178/-0.176 | -0.179/-0.175/-0.172 | -0.179/-0.172/-0.166 | -0.179/-0.170/-0.161 |


Freezes: P(the freeze survives a break) against guard / keep ; the breaker's break − walk up against the guard ; the freezer's guard − keep against the break. A walk-up never breaks a freeze (P = 1 everywhere):

| level | held | investment | baseline | C1 | C2 | C3 |
|---|---|---|---|---|---|---|
| L4 | 5 | equal | 1 / 1 ; +0.44 ; +0.19 | 1 / 1 ; +0.44 ; +0.18 | 1 / 1 ; +0.44 ; +0.18 | 1 / 1 ; +0.44 ; +0.18 |
| L4 | 5 | freezer+8 | 1 / 1 ; +0.44 ; +0.26 | 1 / 1 ; +0.44 ; +0.24 | 1 / 1 ; +0.44 ; +0.23 | 1 / 1 ; +0.44 ; +0.22 |
| L4 | 5 | breaker+8 | 1 / 0 ; +0.55 ; +0.15 | 1 / 0 ; +0.55 ; +0.15 | 1 / 1 ; +0.44 ; +0.15 | 1 / 1 ; +0.44 ; +0.16 |
| L4 | 12 | equal | 1 / 0 ; +0.44 ; +0.08 | 1 / 0 ; +0.44 ; +0.06 | 1 / 0 ; +0.44 ; +0.06 | 1 / 0 ; +0.44 ; +0.06 |
| L4 | 12 | freezer+8 | 1 / 0 ; +0.44 ; +0.18 | 1 / 0 ; +0.44 ; +0.15 | 1 / 0 ; +0.44 ; +0.13 | 1 / 0 ; +0.44 ; +0.12 |
| L4 | 12 | breaker+8 | 1 / 0 ; +0.55 ; -0.07 | 1 / 0 ; +0.55 ; -0.07 | 1 / 0 ; +0.44 ; +0.04 | 1 / 0 ; +0.44 ; +0.04 |
| L4 | 20 | equal | 1 / 0 ; +0.44 ; -0.10 | 1 / 0 ; +0.44 ; -0.12 | 1 / 0 ; +0.44 ; -0.12 | 1 / 0 ; +0.44 ; -0.12 |
| L4 | 20 | freezer+8 | 1 / 0 ; +0.44 ; +0.03 | 1 / 0 ; +0.44 ; -0.01 | 1 / 0 ; +0.44 ; -0.03 | 1 / 0 ; +0.44 ; -0.05 |
| L4 | 20 | breaker+8 | 1 / 0 ; +0.55 ; -0.24 | 1 / 0 ; +0.55 ; -0.25 | 1 / 0 ; +0.44 ; -0.14 | 1 / 0 ; +0.44 ; -0.14 |
| L10 | 5 | equal | 0 / 0 ; +0.78 ; +0.07 | 0 / 0 ; +0.78 ; +0.07 | 0 / 0 ; +0.78 ; +0.06 | 0 / 0 ; +0.78 ; +0.06 |
| L10 | 5 | freezer+8 | 0 / 0 ; +0.78 ; +0.08 | 0 / 0 ; +0.78 ; +0.09 | 0 / 0 ; +0.78 ; +0.09 | 0 / 0 ; +0.78 ; +0.08 |
| L10 | 5 | breaker+8 | 0 / 0 ; +0.78 ; +0.06 | 0 / 0 ; +0.78 ; +0.05 | 0 / 0 ; +0.78 ; +0.05 | 0 / 0 ; +0.78 ; +0.05 |
| L10 | 12 | equal | 0 / 0 ; +0.78 ; -0.09 | 0 / 0 ; +0.78 ; -0.09 | 0 / 0 ; +0.78 ; -0.10 | 0 / 0 ; +0.78 ; -0.10 |
| L10 | 12 | freezer+8 | 0 / 0 ; +0.78 ; -0.08 | 0 / 0 ; +0.78 ; -0.07 | 0 / 0 ; +0.78 ; -0.07 | 0 / 0 ; +0.78 ; -0.08 |
| L10 | 12 | breaker+8 | 0 / 0 ; +0.78 ; -0.10 | 0 / 0 ; +0.78 ; -0.11 | 0 / 0 ; +0.78 ; -0.11 | 0 / 0 ; +0.78 ; -0.12 |
| L10 | 20 | equal | 0 / 0 ; +0.78 ; -0.13 | 0 / 0 ; +0.78 ; -0.14 | 0 / 0 ; +0.78 ; -0.15 | 0 / 0 ; +0.78 ; -0.15 |
| L10 | 20 | freezer+8 | 0 / 0 ; +0.78 ; -0.12 | 0 / 0 ; +0.78 ; -0.12 | 0 / 0 ; +0.78 ; -0.12 | 0 / 0 ; +0.78 ; -0.12 |
| L10 | 20 | breaker+8 | 0 / 0 ; +0.78 ; -0.14 | 0 / 0 ; +0.78 ; -0.15 | 0 / 0 ; +0.78 ; -0.16 | 0 / 0 ; +0.78 ; -0.17 |
| L15 | 5 | equal | 0 / 0 ; +0.78 ; +0.10 | 0 / 0 ; +0.78 ; +0.11 | 0 / 0 ; +0.78 ; +0.11 | 0 / 0 ; +0.78 ; +0.11 |
| L15 | 5 | freezer+8 | 0 / 0 ; +0.78 ; +0.10 | 0 / 0 ; +0.78 ; +0.11 | 0 / 0 ; +0.78 ; +0.12 | 0 / 0 ; +0.78 ; +0.12 |
| L15 | 5 | breaker+8 | 0 / 0 ; +0.78 ; +0.10 | 0 / 0 ; +0.78 ; +0.10 | 0 / 0 ; +0.78 ; +0.10 | 0 / 0 ; +0.78 ; +0.10 |
| L15 | 12 | equal | 0 / 0 ; +0.78 ; -0.03 | 0 / 0 ; +0.81 ; -0.05 | 0 / 0 ; +0.81 ; -0.05 | 0 / 0 ; +0.81 ; -0.05 |
| L15 | 12 | freezer+8 | 0 / 0 ; +0.78 ; -0.02 | 0 / 0 ; +0.81 ; -0.04 | 0 / 0 ; +0.81 ; -0.04 | 0 / 0 ; +0.81 ; -0.03 |
| L15 | 12 | breaker+8 | 0 / 0 ; +0.80 ; -0.05 | 0 / 0 ; +0.81 ; -0.05 | 0 / 0 ; +0.81 ; -0.05 | 0 / 0 ; +0.81 ; -0.06 |
| L15 | 20 | equal | 0 / 0 ; +0.78 ; -0.07 | 0 / 0 ; +0.80 ; -0.09 | 0 / 0 ; +0.80 ; -0.09 | 0 / 0 ; +0.81 ; -0.09 |
| L15 | 20 | freezer+8 | 0 / 0 ; +0.78 ; -0.07 | 0 / 0 ; +0.80 ; -0.09 | 0 / 0 ; +0.81 ; -0.11 | 0 / 0 ; +0.81 ; -0.10 |
| L15 | 20 | breaker+8 | 0 / 0 ; +0.79 ; -0.08 | 0 / 0 ; +0.80 ; -0.10 | 0 / 0 ; +0.80 ; -0.10 | 0 / 0 ; +0.81 ; -0.10 |

## S3: purchased leads (pre-registered; K = 16, horizon 50)

| stage | lead | baseline premium | C1 | C2 | C3 | C1 − baseline | C2 − baseline | C3 − baseline |
|---|---|---|---|---|---|---|---|---|
| early | +8 | +0.39 ± 0.25 | +1.02 ± 0.56 | +0.97 ± 0.28 | +0.74 ± 0.50 | +0.64 ± 0.71 | +0.58 ± 0.43 | +0.35 ± 0.56 |
| early | +16 | +1.03 ± 0.42 | +1.71 ± 0.70 | +1.82 ± 0.34 | +1.30 ± 0.26 | +0.69 ± 0.91 | +0.79 ± 0.69 | +0.27 ± 0.58 |
| mid | +8 | +0.33 ± 0.48 | +0.51 ± 0.18 | +0.59 ± 0.52 | +0.67 ± 0.54 | +0.18 ± 0.49 | +0.26 ± 0.54 | +0.34 ± 0.70 |
| mid | +16 | +0.79 ± 0.36 | +0.68 ± 0.49 | +0.76 ± 0.60 | +0.99 ± 0.47 | -0.11 ± 0.64 | -0.03 ± 0.79 | +0.20 ± 0.61 |
| late | +8 | +0.24 ± 0.25 | -0.03 ± 0.31 | +0.23 ± 0.60 | +0.13 ± 0.62 | -0.26 ± 0.38 | -0.00 ± 0.49 | -0.11 ± 0.61 |
| late | +16 | +0.09 ± 0.33 | +0.11 ± 0.27 | +0.49 ± 0.41 | +0.22 ± 0.57 | +0.02 ± 0.45 | +0.40 ± 0.48 | +0.13 ± 0.83 |


The trailer (focal): its G50 under the policy / its best forced response minus the policy:

| stage | lead | baseline | C1 | C2 | C3 |
|---|---|---|---|---|---|
| early | +8 | -0.83 / best hold -0.21 | -0.76 / best hold +0.06 | -0.84 / best hold -0.08 | -0.87 / best hold -0.22 |
| early | +16 | -0.95 / best hold -0.50 | -1.45 / best hold -0.27 | -1.69 / best hold +0.01 | -1.41 / best hold -0.18 |
| mid | +8 | -0.31 / best hold -0.40 | -0.38 / best hold -0.27 | -0.50 / best hold -0.22 | -0.73 / best hold -0.16 |
| mid | +16 | -0.46 / best hold -0.29 | -0.96 / best hold -0.06 | -0.99 / best hold -0.13 | -1.35 / best hold +0.04 |
| late | +8 | -0.13 / best hold -0.41 | -0.12 / best hold -0.25 | -0.20 / best hold -0.56 | -0.06 / best hold -0.57 |
| late | +16 | -0.28 / best hold -0.30 | -0.06 / best hold -0.64 | -0.29 / best hold -0.63 | -0.36 / best hold -0.38 |


Validation (6 cells at 150 turns): {'baseline': {'pairs': 72, 'mean_change': 0.1538927992888769, 'slope': 1.0185780988777244}, 'C1': {'pairs': 72, 'mean_change': -0.0929541001940497, 'slope': 0.919743704074515}, 'C2': {'pairs': 72, 'mean_change': -0.027340083732237747, 'slope': 0.7491627899954449}, 'C3': {'pairs': 72, 'mean_change': -0.09538483181328102, 'slope': 0.903162311735494}}

## EXPLORATORY: the lead premium at higher power (K = 128, both orientations, horizon 100)

| stage | lead | horizon | baseline | C1 | C2 | C3 | C1 − baseline (ratio) | C2 − baseline | C3 − baseline |
|---|---|---|---|---|---|---|---|---|---|
| mid | +8 | G50 | +0.30 ± 0.15 | +0.47 ± 0.10 | +0.66 ± 0.18 | +0.67 ± 0.19 | +0.17 ± 0.11 (×1.58) | +0.37 ± 0.09 (×2.22) | +0.37 ± 0.12 (×2.24) |
| mid | +8 | G100 | +0.38 ± 0.19 | +0.58 ± 0.14 | +0.84 ± 0.22 | +0.88 ± 0.28 | +0.20 ± 0.11 (×1.53) | +0.46 ± 0.13 (×2.21) | +0.50 ± 0.16 (×2.33) |
| mid | +16 | G50 | +0.52 ± 0.17 | +0.73 ± 0.14 | +0.93 ± 0.23 | +1.05 ± 0.25 | +0.20 ± 0.10 (×1.38) | +0.41 ± 0.10 (×1.78) | +0.52 ± 0.12 (×2.00) |
| mid | +16 | G100 | +0.59 ± 0.17 | +0.96 ± 0.22 | +1.29 ± 0.30 | +1.45 ± 0.33 | +0.37 ± 0.06 (×1.63) | +0.70 ± 0.17 (×2.20) | +0.86 ± 0.17 (×2.46) |
| late | +8 | G50 | +0.08 ± 0.08 | +0.14 ± 0.04 | +0.23 ± 0.13 | +0.27 ± 0.10 | +0.06 ± 0.06 (×1.67) | +0.14 ± 0.07 (×2.70) | +0.19 ± 0.08 (×3.22) |
| late | +8 | G100 | +0.16 ± 0.07 | +0.16 ± 0.06 | +0.33 ± 0.15 | +0.39 ± 0.09 | -0.01 ± 0.06 (×0.97) | +0.17 ± 0.10 (×2.02) | +0.23 ± 0.04 (×2.40) |
| late | +16 | G50 | +0.20 ± 0.12 | +0.24 ± 0.07 | +0.35 ± 0.17 | +0.51 ± 0.19 | +0.04 ± 0.06 (×1.19) | +0.15 ± 0.17 (×1.75) | +0.31 ± 0.16 (×2.55) |
| late | +16 | G100 | +0.32 ± 0.18 | +0.37 ± 0.12 | +0.52 ± 0.12 | +0.84 ± 0.26 | +0.05 ± 0.10 (×1.16) | +0.20 ± 0.16 (×1.65) | +0.52 ± 0.22 (×2.65) |

## S4: freeze contests (K = 16, horizon 50; candidate − policy)

| arm | focal | level | held | investment | policy's G50 | keep / break | guard / walk up | — / step back | — / reset | held turns in 20 (freezer) |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | breaker | L10 | held12 | breaker+8 | -0.83 ± 0.32 | +0.06 ± 0.72 | -0.10 ± 0.64 | -2.15 ± 0.31 | -0.67 ± 0.89 |  |
| C1 | breaker | L10 | held12 | equal | -1.15 ± 0.53 | -0.38 ± 0.59 | -0.58 ± 0.88 | -2.56 ± 0.80 | -0.94 ± 0.51 |  |
| C1 | breaker | L10 | held12 | freezer+8 | -2.44 ± 0.35 | +0.04 ± 0.39 | +0.07 ± 0.69 | -1.14 ± 0.51 | -0.29 ± 0.61 |  |
| C1 | breaker | L10 | held5 | breaker+8 | -0.37 ± 0.53 | -0.17 ± 0.46 | -0.25 ± 0.39 | -1.54 ± 0.31 | +0.01 ± 0.85 |  |
| C1 | breaker | L10 | held5 | equal | -0.58 ± 0.28 | +0.01 ± 0.74 | -0.33 ± 0.39 | -1.56 ± 0.81 | -0.58 ± 0.52 |  |
| C1 | breaker | L10 | held5 | freezer+8 | -1.06 ± 0.72 | -0.07 ± 0.27 | -0.62 ± 0.78 | -1.50 ± 1.00 | -0.48 ± 0.91 |  |
| C1 | breaker | L4 | held12 | breaker+8 | -0.31 ± 0.48 | -0.03 ± 0.31 | +0.17 ± 0.49 | -2.44 ± 0.31 | -1.59 ± 0.48 |  |
| C1 | breaker | L4 | held12 | equal | -1.37 ± 0.52 | -0.18 ± 0.53 | -0.59 ± 0.47 | -1.90 ± 0.83 | -1.19 ± 0.38 |  |
| C1 | breaker | L4 | held12 | freezer+8 | -2.92 ± 0.63 | +0.37 ± 0.98 | +0.37 ± 0.91 | -0.99 ± 1.11 | -0.68 ± 0.74 |  |
| C1 | breaker | L4 | held5 | breaker+8 | -0.04 ± 0.57 | +0.17 ± 0.45 | +0.31 ± 0.61 | -1.26 ± 0.73 | -0.55 ± 1.11 |  |
| C1 | breaker | L4 | held5 | equal | -0.69 ± 0.96 | -0.21 ± 0.62 | +0.02 ± 1.20 | -1.38 ± 0.96 | -0.45 ± 1.01 |  |
| C1 | breaker | L4 | held5 | freezer+8 | -1.63 ± 0.64 | +0.70 ± 0.42 | +0.08 ± 0.83 | -1.33 ± 1.07 | -0.67 ± 0.46 |  |
| C1 | freezer | L10 | held12 | breaker+8 | +0.57 ± 0.47 | +0.43 ± 0.54 | -0.13 ± 0.32 | 1.4 |
| C1 | freezer | L10 | held12 | equal | +1.04 ± 0.63 | -0.16 ± 0.60 | +0.53 ± 1.03 | 1.2 |
| C1 | freezer | L10 | held12 | freezer+8 | +2.09 ± 0.54 | -0.20 ± 0.57 | -0.25 ± 0.97 | 1.1 |
| C1 | freezer | L10 | held5 | breaker+8 | -0.23 ± 0.58 | -0.01 ± 0.96 | +0.50 ± 0.61 | 1.4 |
| C1 | freezer | L10 | held5 | equal | +0.66 ± 0.37 | -0.27 ± 0.44 | -0.26 ± 0.57 | 1.2 |
| C1 | freezer | L10 | held5 | freezer+8 | +1.25 ± 0.67 | -0.51 ± 0.61 | -0.30 ± 0.63 | 1.1 |
| C1 | freezer | L4 | held12 | breaker+8 | +0.35 ± 1.11 | +0.13 ± 0.47 | -1.22 ± 1.77 | 2.8 |
| C1 | freezer | L4 | held12 | equal | +2.05 ± 0.55 | -0.48 ± 0.77 | -1.94 ± 0.96 | 5.1 |
| C1 | freezer | L4 | held12 | freezer+8 | +2.65 ± 0.24 | -0.13 ± 0.41 | -0.46 ± 0.93 | 3.4 |
| C1 | freezer | L4 | held5 | breaker+8 | +0.29 ± 0.38 | -0.56 ± 0.27 | -1.19 ± 0.65 | 3.1 |
| C1 | freezer | L4 | held5 | equal | +0.93 ± 0.77 | -0.04 ± 0.89 | -0.55 ± 0.76 | 5.3 |
| C1 | freezer | L4 | held5 | freezer+8 | +1.75 ± 0.55 | -0.24 ± 1.10 | +0.83 ± 0.67 | 4.5 |
| C2 | breaker | L10 | held12 | breaker+8 | -1.02 ± 0.29 | -0.14 ± 0.53 | -0.09 ± 0.40 | -1.55 ± 0.63 | -0.63 ± 0.87 |  |
| C2 | breaker | L10 | held12 | equal | -1.25 ± 0.35 | -0.50 ± 0.77 | -0.35 ± 0.88 | -2.06 ± 0.72 | -1.05 ± 0.86 |  |
| C2 | breaker | L10 | held12 | freezer+8 | -2.16 ± 0.72 | -0.15 ± 0.49 | -0.67 ± 0.98 | -1.78 ± 0.55 | -1.00 ± 0.51 |  |
| C2 | breaker | L10 | held5 | breaker+8 | -0.44 ± 0.49 | +0.25 ± 0.45 | +0.63 ± 0.59 | -1.49 ± 0.25 | +0.12 ± 0.82 |  |
| C2 | breaker | L10 | held5 | equal | -0.93 ± 0.73 | +0.30 ± 0.41 | +0.25 ± 0.72 | -1.66 ± 0.95 | +0.30 ± 1.09 |  |
| C2 | breaker | L10 | held5 | freezer+8 | -1.28 ± 0.24 | +0.08 ± 0.55 | -0.09 ± 0.96 | -1.85 ± 0.52 | -0.22 ± 0.50 |  |
| C2 | breaker | L4 | held12 | breaker+8 | -0.64 ± 0.62 | -0.05 ± 0.37 | +0.10 ± 0.65 | -2.18 ± 1.21 | -1.19 ± 1.30 |  |
| C2 | breaker | L4 | held12 | equal | -1.50 ± 0.46 | -0.21 ± 0.30 | -0.26 ± 0.49 | -1.64 ± 0.65 | -1.17 ± 0.48 |  |
| C2 | breaker | L4 | held12 | freezer+8 | -2.22 ± 0.73 | -0.02 ± 0.39 | -0.12 ± 0.86 | -1.83 ± 0.70 | -1.47 ± 1.07 |  |
| C2 | breaker | L4 | held5 | breaker+8 | +0.12 ± 0.52 | +0.00 ± 0.62 | -0.13 ± 0.71 | -1.34 ± 0.71 | -0.19 ± 0.39 |  |
| C2 | breaker | L4 | held5 | equal | -0.80 ± 0.53 | -0.12 ± 0.36 | +0.18 ± 0.65 | -0.85 ± 0.27 | -0.70 ± 0.64 |  |
| C2 | breaker | L4 | held5 | freezer+8 | -1.31 ± 0.80 | +0.23 ± 0.36 | -0.24 ± 1.06 | -1.57 ± 1.25 | -0.75 ± 0.94 |  |
| C2 | freezer | L10 | held12 | breaker+8 | +0.63 ± 0.64 | +0.09 ± 0.70 | +0.06 ± 0.63 | 1.4 |
| C2 | freezer | L10 | held12 | equal | +0.89 ± 0.79 | +0.34 ± 0.96 | -0.03 ± 1.01 | 1.2 |
| C2 | freezer | L10 | held12 | freezer+8 | +1.80 ± 0.42 | +0.12 ± 1.02 | +0.10 ± 0.29 | 1.3 |
| C2 | freezer | L10 | held5 | breaker+8 | -0.10 ± 0.89 | +0.20 ± 1.23 | +0.15 ± 0.20 | 1.1 |
| C2 | freezer | L10 | held5 | equal | +0.79 ± 0.43 | -0.59 ± 0.95 | -0.70 ± 0.54 | 1.1 |
| C2 | freezer | L10 | held5 | freezer+8 | +1.33 ± 0.40 | -0.68 ± 0.66 | -0.59 ± 0.95 | 1.0 |
| C2 | freezer | L4 | held12 | breaker+8 | +0.82 ± 0.35 | -0.06 ± 0.50 | -1.73 ± 0.44 | 3.3 |
| C2 | freezer | L4 | held12 | equal | +1.88 ± 0.56 | -0.30 ± 0.60 | -1.72 ± 0.71 | 5.1 |
| C2 | freezer | L4 | held12 | freezer+8 | +2.58 ± 0.41 | +0.01 ± 0.76 | -0.70 ± 0.87 | 4.0 |
| C2 | freezer | L4 | held5 | breaker+8 | +0.09 ± 0.76 | -0.52 ± 0.54 | -1.03 ± 0.19 | 4.4 |
| C2 | freezer | L4 | held5 | equal | +0.94 ± 0.44 | -0.03 ± 0.59 | -0.54 ± 0.40 | 5.4 |
| C2 | freezer | L4 | held5 | freezer+8 | +1.73 ± 0.51 | -0.06 ± 0.81 | +0.77 ± 0.86 | 5.3 |
| C3 | breaker | L10 | held12 | breaker+8 | -1.20 ± 0.47 | -0.15 ± 0.52 | +0.45 ± 0.36 | -1.50 ± 0.69 | -0.53 ± 0.23 |  |
| C3 | breaker | L10 | held12 | equal | -1.64 ± 0.55 | -0.23 ± 0.61 | -0.03 ± 0.84 | -1.58 ± 0.62 | -0.54 ± 0.92 |  |
| C3 | breaker | L10 | held12 | freezer+8 | -1.81 ± 0.73 | -0.28 ± 0.58 | -0.07 ± 0.45 | -1.88 ± 1.25 | -0.97 ± 0.60 |  |
| C3 | breaker | L10 | held5 | breaker+8 | +0.18 ± 0.56 | -0.09 ± 0.70 | -0.18 ± 0.89 | -1.82 ± 1.13 | -0.62 ± 0.69 |  |
| C3 | breaker | L10 | held5 | equal | -0.84 ± 0.36 | +0.18 ± 0.39 | -0.06 ± 0.64 | -1.43 ± 1.10 | +0.09 ± 1.19 |  |
| C3 | breaker | L10 | held5 | freezer+8 | -1.45 ± 0.84 | +0.17 ± 0.59 | -0.12 ± 1.06 | -1.22 ± 0.72 | -0.18 ± 1.01 |  |
| C3 | breaker | L4 | held12 | breaker+8 | -0.40 ± 0.63 | -0.29 ± 0.76 | -0.04 ± 0.55 | -2.26 ± 0.52 | -1.50 ± 0.61 |  |
| C3 | breaker | L4 | held12 | equal | -1.35 ± 0.64 | -0.47 ± 0.67 | -0.55 ± 0.70 | -1.88 ± 0.59 | -1.68 ± 1.11 |  |
| C3 | breaker | L4 | held12 | freezer+8 | -2.29 ± 0.45 | +0.12 ± 0.25 | +0.09 ± 0.51 | -1.71 ± 0.46 | -1.16 ± 0.61 |  |
| C3 | breaker | L4 | held5 | breaker+8 | +0.37 ± 0.74 | -0.17 ± 0.28 | -0.53 ± 1.07 | -1.47 ± 0.95 | -0.63 ± 0.89 |  |
| C3 | breaker | L4 | held5 | equal | -0.42 ± 0.61 | -0.27 ± 0.29 | -0.37 ± 0.88 | -1.15 ± 0.68 | -1.09 ± 0.58 |  |
| C3 | breaker | L4 | held5 | freezer+8 | -1.09 ± 0.55 | +0.14 ± 0.42 | -0.34 ± 0.81 | -1.86 ± 0.78 | -1.41 ± 0.73 |  |
| C3 | freezer | L10 | held12 | breaker+8 | +0.69 ± 0.69 | -0.26 ± 0.90 | +0.02 ± 0.56 | 1.4 |
| C3 | freezer | L10 | held12 | equal | +1.25 ± 0.39 | -0.13 ± 0.75 | +0.31 ± 1.09 | 1.2 |
| C3 | freezer | L10 | held12 | freezer+8 | +2.02 ± 0.77 | -0.13 ± 0.68 | -0.22 ± 0.82 | 1.4 |
| C3 | freezer | L10 | held5 | breaker+8 | -0.14 ± 0.66 | +0.26 ± 0.75 | +0.23 ± 0.78 | 1.1 |
| C3 | freezer | L10 | held5 | equal | +0.60 ± 0.63 | -0.38 ± 0.45 | -0.02 ± 0.40 | 1.1 |
| C3 | freezer | L10 | held5 | freezer+8 | +1.37 ± 1.09 | -0.54 ± 1.21 | -0.08 ± 1.37 | 1.1 |
| C3 | freezer | L4 | held12 | breaker+8 | +0.91 ± 0.56 | +0.08 ± 0.29 | -1.73 ± 0.64 | 3.4 |
| C3 | freezer | L4 | held12 | equal | +1.96 ± 0.54 | -0.21 ± 0.57 | -1.74 ± 0.94 | 5.1 |
| C3 | freezer | L4 | held12 | freezer+8 | +2.34 ± 0.79 | -0.04 ± 1.28 | -0.33 ± 0.95 | 4.0 |
| C3 | freezer | L4 | held5 | breaker+8 | +0.02 ± 0.78 | -0.40 ± 0.88 | -0.98 ± 0.72 | 4.7 |
| C3 | freezer | L4 | held5 | equal | +1.00 ± 0.84 | +0.17 ± 0.68 | -0.67 ± 0.93 | 5.3 |
| C3 | freezer | L4 | held5 | freezer+8 | +1.43 ± 0.46 | +0.19 ± 1.03 | +0.50 ± 0.72 | 5.1 |
| baseline | breaker | L10 | held12 | breaker+8 | -1.18 ± 0.68 | -0.27 ± 0.67 | +0.35 ± 1.03 | -1.89 ± 0.72 | -0.40 ± 0.81 |  |
| baseline | breaker | L10 | held12 | equal | -1.65 ± 0.60 | -0.22 ± 0.51 | +0.11 ± 0.81 | -1.25 ± 0.75 | -0.39 ± 0.82 |  |
| baseline | breaker | L10 | held12 | freezer+8 | -1.93 ± 0.53 | -0.10 ± 0.57 | -0.17 ± 0.50 | -1.62 ± 0.78 | -0.58 ± 0.89 |  |
| baseline | breaker | L10 | held5 | breaker+8 | -0.32 ± 0.38 | +0.19 ± 0.58 | -0.10 ± 0.57 | -1.58 ± 0.85 | -0.42 ± 0.72 |  |
| baseline | breaker | L10 | held5 | equal | -1.00 ± 0.32 | +0.03 ± 0.24 | -0.11 ± 0.53 | -1.11 ± 0.54 | +0.09 ± 0.48 |  |
| baseline | breaker | L10 | held5 | freezer+8 | -1.11 ± 0.51 | +0.14 ± 0.44 | -0.45 ± 1.03 | -0.95 ± 0.98 | -0.43 ± 0.88 |  |
| baseline | breaker | L4 | held12 | breaker+8 | +0.22 ± 0.56 | -0.21 ± 0.33 | -0.07 ± 0.53 | -2.80 ± 0.80 | -1.81 ± 0.67 |  |
| baseline | breaker | L4 | held12 | equal | -1.61 ± 0.43 | -0.15 ± 0.63 | -0.13 ± 0.82 | -1.92 ± 0.72 | -1.20 ± 0.57 |  |
| baseline | breaker | L4 | held12 | freezer+8 | -2.21 ± 0.38 | -0.22 ± 0.33 | -0.22 ± 0.66 | -2.22 ± 0.68 | -1.52 ± 0.53 |  |
| baseline | breaker | L4 | held5 | breaker+8 | +0.15 ± 0.53 | +0.17 ± 0.28 | +0.66 ± 0.76 | -1.53 ± 0.31 | -0.63 ± 0.69 |  |
| baseline | breaker | L4 | held5 | equal | -1.05 ± 0.45 | +0.11 ± 0.31 | +0.50 ± 0.63 | -1.04 ± 0.51 | -0.34 ± 0.45 |  |
| baseline | breaker | L4 | held5 | freezer+8 | -1.61 ± 0.29 | +0.41 ± 0.47 | -0.18 ± 1.23 | -1.63 ± 0.86 | -0.55 ± 0.57 |  |
| baseline | freezer | L10 | held12 | breaker+8 | +0.66 ± 0.42 | +0.79 ± 0.75 | -0.02 ± 0.88 | 1.5 |
| baseline | freezer | L10 | held12 | equal | +1.40 ± 0.63 | -0.04 ± 0.90 | -0.17 ± 1.03 | 1.2 |
| baseline | freezer | L10 | held12 | freezer+8 | +1.82 ± 0.48 | -0.06 ± 0.93 | -0.08 ± 0.47 | 1.3 |
| baseline | freezer | L10 | held5 | breaker+8 | -0.01 ± 0.69 | +0.16 ± 0.77 | -0.09 ± 0.80 | 1.1 |
| baseline | freezer | L10 | held5 | equal | +0.36 ± 0.28 | +0.02 ± 0.71 | +0.04 ± 0.67 | 1.0 |
| baseline | freezer | L10 | held5 | freezer+8 | +0.91 ± 0.59 | +0.30 ± 0.52 | -0.31 ± 0.76 | 1.1 |
| baseline | freezer | L4 | held12 | breaker+8 | +0.16 ± 0.81 | +0.31 ± 0.72 | -1.35 ± 1.73 | 2.7 |
| baseline | freezer | L4 | held12 | equal | +1.71 ± 0.76 | -0.14 ± 0.79 | -1.18 ± 0.37 | 4.4 |
| baseline | freezer | L4 | held12 | freezer+8 | +2.47 ± 0.36 | +0.05 ± 0.33 | +0.25 ± 0.87 | 3.6 |
| baseline | freezer | L4 | held5 | breaker+8 | -0.34 ± 0.91 | -0.31 ± 0.70 | -0.50 ± 0.74 | 3.2 |
| baseline | freezer | L4 | held5 | equal | +1.08 ± 0.68 | -0.49 ± 0.82 | -0.40 ± 0.42 | 5.5 |
| baseline | freezer | L4 | held5 | freezer+8 | +1.95 ± 0.25 | -0.42 ± 0.69 | -0.04 ± 0.45 | 4.2 |


## S5: Recall and shopping (K = 12, horizon 50, G50)

| arm | family | Recall − stay | same, gold pre-invested | shopping component | pre-invested state − original |
|---|---|---|---|---|---|
| baseline | recall_full | -0.35 ± 0.34 | -0.37 ± 0.19 | +0.02 ± 0.27 | +0.08 ± 0.17 |
| baseline | recall_mid | +0.33 ± 0.07 | +0.20 ± 0.12 | +0.13 ± 0.14 | +0.04 ± 0.23 |
| C1 | recall_full | -0.37 ± 0.20 | -0.45 ± 0.26 | +0.09 ± 0.21 | +0.07 ± 0.18 |
| C1 | recall_mid | +0.36 ± 0.37 | +0.25 ± 0.26 | +0.11 ± 0.37 | -0.10 ± 0.06 |
| C2 | recall_full | -0.42 ± 0.19 | -0.39 ± 0.22 | -0.03 ± 0.24 | +0.07 ± 0.20 |
| C2 | recall_mid | +0.29 ± 0.14 | +0.16 ± 0.18 | +0.12 ± 0.24 | +0.06 ± 0.15 |
| C3 | recall_full | -0.27 ± 0.25 | -0.41 ± 0.25 | +0.14 ± 0.19 | +0.09 ± 0.10 |
| C3 | recall_mid | +0.40 ± 0.23 | +0.25 ± 0.19 | +0.15 ± 0.13 | +0.11 ± 0.13 |

## S6: natural starts (16 games × 600 turns per final and arm; baseline-trained policies)

| arm | attack ratio q90 | share ≥ 1.1 | share ≥ 1.25 | duel ratio q90 | duel share ≥ 1.25 | leader's duel ratio: early | mid | late | very late |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.047 | 0.029 | 0.0024 | 1.096 | 0.021 | ×1.034 | ×1.041 | ×1.010 | ×1.006 |
| C1 | 1.061 | 0.038 | 0.0007 | 1.125 | 0.025 | ×1.025 | ×1.057 | ×1.029 | ×1.015 |
| C2 | 1.079 | 0.058 | 0.0016 | 1.165 | 0.036 | ×1.017 | ×1.075 | ×1.051 | ×1.030 |
| C3 | 1.098 | 0.095 | 0.0015 | 1.206 | 0.059 | ×1.015 | ×1.082 | ×1.072 | ×1.059 |


Lane sanity (per side; per 100 turns where a rate):

| arm | crashed HP / 100 | plates by 150 | plates by 300 | deaths / 100 | Recalls / 100 | farm collection | Shove share | connected trades / 100 | C turns 90-150 | C turns 240-320 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | +66.422 ± 5.138 | +1.911 ± 0.329 | +2.734 ± 0.132 | +0.636 ± 0.099 | +4.228 ± 0.208 | +0.761 ± 0.013 | +0.372 ± 0.025 | +13.753 ± 0.947 | +13.890 ± 0.098 | +17.974 ± 0.019 |
| C1 | +69.009 ± 1.568 | +1.990 ± 0.284 | +2.859 ± 0.071 | +0.589 ± 0.094 | +4.139 ± 0.275 | +0.761 ± 0.008 | +0.365 ± 0.018 | +13.647 ± 1.167 | +13.971 ± 0.133 | +20.620 ± 0.051 |
| C2 | +69.477 ± 4.485 | +1.948 ± 0.273 | +2.760 ± 0.115 | +0.555 ± 0.087 | +4.087 ± 0.330 | +0.765 ± 0.010 | +0.369 ± 0.024 | +13.483 ± 1.308 | +13.782 ± 0.106 | +23.104 ± 0.097 |
| C3 | +69.820 ± 3.091 | +1.948 ± 0.233 | +2.802 ± 0.082 | +0.541 ± 0.076 | +3.995 ± 0.298 | +0.765 ± 0.009 | +0.365 ± 0.021 | +13.570 ± 1.145 | +13.616 ± 0.149 | +24.805 ± 0.197 |


Lead persistence (D2): the signed earned lead at turn 300 regressed on the lead at turn 100:

| arm | slope (mean of the six finals) | slope (all games pooled) |
|---|---|---|
| baseline | +0.75 ± 0.40 | 0.75 |
| C1 | +1.02 ± 0.25 | 0.94 |
| C2 | +1.19 ± 0.29 | 1.08 |
| C3 | +1.09 ± 0.36 | 1.12 |
