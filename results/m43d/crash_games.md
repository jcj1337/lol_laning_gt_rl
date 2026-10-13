# M4.3d Part 5a: restricted games of the crash scenarios (from the tower_farm grid; J over 100 turns)

A = the 12 attacker strategies, B = the 7 defender strategies (strategies.py); 64 paired games per cell.

| start | gap | own tower | objective | value | A's equilibrium | B's equilibrium |
|---|---|---|---|---|---|---|
| S10 | equal | 0.6 | J | -0.89 | sh_press 1.00 | shove 1.00 |
| S10 | equal | 0.6 | G40 | -0.51 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | equal | 0.8 | J | -1.40 | sh_press 1.00 | shove 1.00 |
| S10 | equal | 0.8 | G40 | -1.13 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | equal | 0.85 | J | -1.59 | sh_press 1.00 | shove 1.00 |
| S10 | equal | 0.85 | G40 | -1.30 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | equal | 0.9 | J | -1.80 | sh_press 0.88, sh_vis 0.12 | thin 0.16, shove 0.84 |
| S10 | equal | 0.9 | G40 | -1.45 | sh_press 1.00 | shove 1.00 |
| S10 | A x1.76 | 0.6 | J | +5.85 | sh_vis 1.00 | shove 1.00 |
| S10 | A x1.76 | 0.6 | G40 | +4.55 | sh_press 1.00 | shove 1.00 |
| S10 | A x1.76 | 0.8 | J | +2.68 | sh_vis 1.00 | shove 1.00 |
| S10 | A x1.76 | 0.8 | G40 | +2.16 | sh_press 0.34, sh_hunt 0.41, sh_vis 0.24 | thin 0.09, respect 0.27, shove 0.65 |
| S10 | A x1.76 | 0.85 | J | +1.62 | sh_hunt 0.30, sh_vis 0.70 | thin 0.71, shove 0.29 |
| S10 | A x1.76 | 0.85 | G40 | +1.56 | sh_press 0.18, sh_hunt 0.51, sh_vis 0.31 | lh 0.39, thin 0.01, shove 0.60 |
| S10 | A x1.76 | 0.9 | J | +0.67 | sh_hunt 0.56, sh_vis 0.44 | thin 0.53, shove 0.47 |
| S10 | A x1.76 | 0.9 | G40 | +0.84 | sh_hunt 0.42, sh_vis 0.58 | lh 0.46, shove 0.54 |
| S10 | A weak (B x1.76) | 0.6 | J | -7.15 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | A weak (B x1.76) | 0.6 | G40 | -5.67 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | A weak (B x1.76) | 0.8 | J | -5.34 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | A weak (B x1.76) | 0.8 | G40 | -4.73 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | A weak (B x1.76) | 0.85 | J | -4.90 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | A weak (B x1.76) | 0.85 | G40 | -4.50 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | A weak (B x1.76) | 0.9 | J | -4.51 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | A weak (B x1.76) | 0.9 | G40 | -4.26 | sh_dive_edge 1.00 | shove 1.00 |
| S4 | equal | 0.6 | J | -1.25 | sh_vis 1.00 | shove 1.00 |
| S4 | equal | 0.6 | G40 | -0.80 | sh_vis 1.00 | shove 1.00 |
| S4 | equal | 0.8 | J | -1.70 | sh_vis 1.00 | shove 1.00 |
| S4 | equal | 0.8 | G40 | -1.31 | sh_vis 1.00 | shove 1.00 |
| S4 | equal | 0.85 | J | -1.83 | sh_vis 1.00 | shove 1.00 |
| S4 | equal | 0.85 | G40 | -1.46 | sh_vis 1.00 | shove 1.00 |
| S4 | equal | 0.9 | J | -1.92 | sh_vis 1.00 | shove 1.00 |
| S4 | equal | 0.9 | G40 | -1.57 | sh_vis 1.00 | shove 1.00 |
| S4 | A x1.76 | 0.6 | J | +6.35 | sh_vis 1.00 | shove 1.00 |
| S4 | A x1.76 | 0.6 | G40 | +4.96 | sh_vis 1.00 | shove 1.00 |
| S4 | A x1.76 | 0.8 | J | +2.71 | sh_vis 1.00 | thin 1.00 |
| S4 | A x1.76 | 0.8 | G40 | +2.32 | sh_vis 1.00 | shove 1.00 |
| S4 | A x1.76 | 0.85 | J | +1.56 | sh_hunt 0.66, sh_vis 0.34 | thin 0.83, shove 0.17 |
| S4 | A x1.76 | 0.85 | G40 | +1.68 | sh_vis 1.00 | shove 1.00 |
| S4 | A x1.76 | 0.9 | J | +0.62 | sh_hunt 0.68, sh_dive_edge 0.14, sh_vis 0.18 | lh 0.26, thin 0.50, shove 0.24 |
| S4 | A x1.76 | 0.9 | G40 | +0.91 | sh_hunt 0.16, sh_vis 0.84 | thin 0.53, shove 0.47 |
| S4 | A weak (B x1.76) | 0.6 | J | -7.19 | sh_dive_edge 1.00 | shove 1.00 |
| S4 | A weak (B x1.76) | 0.6 | G40 | -6.25 | sh_dive_edge 1.00 | shove 1.00 |
| S4 | A weak (B x1.76) | 0.8 | J | -5.34 | sh_dive_edge 1.00 | shove 1.00 |
| S4 | A weak (B x1.76) | 0.8 | G40 | -5.28 | sh_dive_edge 1.00 | shove 1.00 |
| S4 | A weak (B x1.76) | 0.85 | J | -4.92 | sh_dive_edge 1.00 | shove 1.00 |
| S4 | A weak (B x1.76) | 0.85 | G40 | -5.08 | sh_dive_edge 1.00 | shove 1.00 |
| S4 | A weak (B x1.76) | 0.9 | J | -4.48 | sh_dive_edge 1.00 | shove 1.00 |
| S4 | A weak (B x1.76) | 0.9 | G40 | -4.80 | sh_dive_edge 1.00 | shove 1.00 |
| N10 | equal | 0.6 | J | +0.46 | sh_stay 1.00 | shove 1.00 |
| N10 | equal | 0.6 | G40 | +0.37 | sh_stay 1.00 | shove 1.00 |
| N10 | equal | 0.8 | J | +0.14 | sh_press 1.00 | shove 1.00 |
| N10 | equal | 0.8 | G40 | +0.12 | sh_press 1.00 | shove 1.00 |
| N10 | equal | 0.85 | J | +0.14 | sh_press 1.00 | shove 1.00 |
| N10 | equal | 0.85 | G40 | +0.13 | sh_press 1.00 | shove 1.00 |
| N10 | equal | 0.9 | J | +0.07 | lh 0.10, sh_press 0.90 | thin 0.27, shove 0.73 |
| N10 | equal | 0.9 | G40 | +0.15 | sh_press 1.00 | shove 1.00 |
| N10 | A x1.76 | 0.6 | J | +6.55 | sh_press 1.00 | shove 1.00 |
| N10 | A x1.76 | 0.6 | G40 | +3.77 | sh_press 1.00 | shove 1.00 |
| N10 | A x1.76 | 0.8 | J | +3.96 | sh_press 0.70, sh_vis 0.30 | thin 0.17, shove 0.83 |
| N10 | A x1.76 | 0.8 | G40 | +2.46 | sh_harass 1.00 | shove 1.00 |
| N10 | A x1.76 | 0.85 | J | +3.24 | sh_press 0.40, sh_vis 0.60 | thin 0.20, shove 0.80 |
| N10 | A x1.76 | 0.85 | G40 | +2.21 | sh_harass 1.00 | shove 1.00 |
| N10 | A x1.76 | 0.9 | J | +2.48 | sh_press 0.09, sh_vis 0.91 | thin 0.24, shove 0.76 |
| N10 | A x1.76 | 0.9 | G40 | +1.95 | sh_harass 1.00 | shove 1.00 |
| N10 | A weak (B x1.76) | 0.6 | J | -6.43 | sh_dive_edge 1.00 | shove 1.00 |
| N10 | A weak (B x1.76) | 0.6 | G40 | -3.42 | sh_vis 1.00 | shove 1.00 |
| N10 | A weak (B x1.76) | 0.8 | J | -3.91 | sh_dive_edge 1.00 | shove 1.00 |
| N10 | A weak (B x1.76) | 0.8 | G40 | -1.77 | sh_vis 1.00 | shove 1.00 |
| N10 | A weak (B x1.76) | 0.85 | J | -3.27 | sh_dive_edge 1.00 | shove 1.00 |
| N10 | A weak (B x1.76) | 0.85 | G40 | -1.35 | sh_vis 1.00 | shove 1.00 |
| N10 | A weak (B x1.76) | 0.9 | J | -2.38 | lh 0.82, sh_vis 0.18 | punish 0.07, shove 0.93 |
| N10 | A weak (B x1.76) | 0.9 | G40 | -0.94 | sh_vis 1.00 | shove 1.00 |
| N4 | equal | 0.6 | J | +0.08 | sh_press 1.00 | shove 1.00 |
| N4 | equal | 0.6 | G40 | +0.06 | sh_vis 1.00 | shove 1.00 |
| N4 | equal | 0.8 | J | +0.04 | sh_press 1.00 | shove 1.00 |
| N4 | equal | 0.8 | G40 | +0.02 | sh_press 1.00 | shove 1.00 |
| N4 | equal | 0.85 | J | +0.04 | sh_press 1.00 | shove 1.00 |
| N4 | equal | 0.85 | G40 | +0.02 | sh_press 1.00 | shove 1.00 |
| N4 | equal | 0.9 | J | +0.01 | sh_press 0.72, sh_vis 0.28 | thin 0.06, shove 0.94 |
| N4 | equal | 0.9 | G40 | +0.02 | sh_press 1.00 | shove 1.00 |
| N4 | A x1.76 | 0.6 | J | +7.40 | sh_vis 1.00 | shove 1.00 |
| N4 | A x1.76 | 0.6 | G40 | +6.03 | sh_vis 1.00 | shove 1.00 |
| N4 | A x1.76 | 0.8 | J | +4.50 | sh_vis 1.00 | shove 1.00 |
| N4 | A x1.76 | 0.8 | G40 | +3.92 | sh_vis 1.00 | shove 1.00 |
| N4 | A x1.76 | 0.85 | J | +3.66 | sh_hunt 0.08, sh_vis 0.92 | thin 0.77, shove 0.23 |
| N4 | A x1.76 | 0.85 | G40 | +3.41 | sh_vis 1.00 | shove 1.00 |
| N4 | A x1.76 | 0.9 | J | +2.72 | sh_hunt 0.29, sh_vis 0.71 | thin 0.59, shove 0.41 |
| N4 | A x1.76 | 0.9 | G40 | +2.88 | sh_vis 1.00 | shove 1.00 |
| N4 | A weak (B x1.76) | 0.6 | J | -6.45 | sh_dive 1.00 | shove 1.00 |
| N4 | A weak (B x1.76) | 0.6 | G40 | -4.98 | sh_stay 1.00 | shove 1.00 |
| N4 | A weak (B x1.76) | 0.8 | J | -4.01 | sh_vis_dive 1.00 | shove 1.00 |
| N4 | A weak (B x1.76) | 0.8 | G40 | -3.29 | sh_stay 1.00 | shove 1.00 |
| N4 | A weak (B x1.76) | 0.85 | J | -3.41 | sh_vis_dive 1.00 | shove 1.00 |
| N4 | A weak (B x1.76) | 0.85 | G40 | -2.90 | sh_stay 1.00 | shove 1.00 |
| N4 | A weak (B x1.76) | 0.9 | J | -2.41 | lh 0.81, sh_vis 0.19 | punish 0.08, shove 0.92 |
| N4 | A weak (B x1.76) | 0.9 | G40 | -2.49 | sh_stay 0.90, sh_vis 0.10 | respect 0.00, shove 1.00 |
