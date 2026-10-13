# M4.3d Part 4: sensitivity of the risk rules (one parameter group per variant, econ otherwise)

Same seeds and cells as Parts 2-3 (64 paired games, 100 turns). `tw_aggro`: econ; the tower is never 'busy' for a connected trader: a dive is shot with p 1.0 and the tower-adjacent shot with 0.5 even while the diver's minions are under the tower (League's tower aggro on a champion that attacks a champion); `tw2`: econ; a tower shot deals 2 HP units (1/3 bar) instead of 1; `tw2_aggro`: econ; tw2 + tw_aggro; `gk_pos16`: econ; gank occurrence slope 0.16 instead of 0.12 (exposure +2: 0.17 / 0.19 with Shove); `gk_dmg08`: econ; gank damage slope 0.8 instead of 0.6 (p_e 0.2 / 0.4 / 0.6 / 0.8 / 1.0 at exposure -2..+2).

## S10, equal: A's value J (B last-hits) / A deaths per 40 turns

| A | econ | tw_aggro | tw2 | tw2_aggro | gk_pos16 | gk_dmg08 |
|---|---|---|---|---|---|---|
| lh | -2.84 / 0.03 | -2.84 / 0.03 | -2.84 / 0.03 | -2.84 / 0.03 | -2.91 / 0.11 | -2.89 / 0.09 |
| sh_press | +1.27 / 0.41 | +1.27 / 0.41 | +1.27 / 0.41 | +1.27 / 0.41 | +0.16 / 0.56 | -0.63 / 1.06 |
| sh_vis | +3.13 / 0.12 | +3.13 / 0.12 | +3.13 / 0.12 | +3.13 / 0.12 | +2.83 / 0.19 | +2.22 / 0.38 |
| sh_dive | -3.09 / 0.66 | -4.49 / 0.67 | -4.55 / 0.69 | -6.07 / 0.84 | -3.88 / 0.83 | -4.50 / 0.81 |
| sh_dive_edge | +1.73 / 0.41 | +1.12 / 0.48 | +1.08 / 0.48 | +0.33 / 0.56 | +0.63 / 0.55 | +0.07 / 0.75 |
| sh_vis_dive | -0.51 / 0.19 | -2.10 / 0.30 | -1.50 / 0.22 | -3.24 / 0.28 | -1.02 / 0.25 | -1.05 / 0.30 |
| sh_bully | -10.20 / 0.73 | -10.64 / 0.88 | -11.83 / 0.89 | -11.64 / 0.95 | -11.16 / 0.94 | -10.99 / 0.97 |

## S10, A x1.76: A's value J (B last-hits) / A deaths per 40 turns

| A | econ | tw_aggro | tw2 | tw2_aggro | gk_pos16 | gk_dmg08 |
|---|---|---|---|---|---|---|
| lh | -2.51 / 0.00 | -2.51 / 0.00 | -2.51 / 0.00 | -2.51 / 0.00 | -2.39 / 0.00 | -2.24 / 0.02 |
| sh_press | +2.74 / 0.12 | +2.74 / 0.12 | +2.74 / 0.12 | +2.74 / 0.12 | +2.11 / 0.22 | +2.11 / 0.16 |
| sh_vis | +4.20 / 0.00 | +4.20 / 0.00 | +4.20 / 0.00 | +4.20 / 0.00 | +3.94 / 0.00 | +3.91 / 0.00 |
| sh_dive | +2.59 / 0.09 | -0.10 / 0.20 | +0.35 / 0.20 | -3.62 / 0.59 | +1.57 / 0.14 | +0.74 / 0.38 |
| sh_dive_edge | +4.47 / 0.08 | +2.82 / 0.16 | +2.47 / 0.22 | +0.52 / 0.34 | +3.67 / 0.16 | +2.65 / 0.34 |
| sh_vis_dive | +4.14 / 0.03 | +2.20 / 0.03 | +2.42 / 0.06 | -0.91 / 0.19 | +3.39 / 0.06 | +3.36 / 0.06 |
| sh_bully | -6.73 / 0.44 | -5.13 / 0.23 | -6.52 / 0.47 | -7.22 / 0.44 | -6.60 / 0.45 | -6.09 / 0.55 |

## S10, A weak (B x1.76): A's value J (B last-hits) / A deaths per 40 turns

| A | econ | tw_aggro | tw2 | tw2_aggro | gk_pos16 | gk_dmg08 |
|---|---|---|---|---|---|---|
| lh | -3.55 / 0.03 | -3.55 / 0.03 | -3.55 / 0.03 | -3.55 / 0.03 | -3.82 / 0.08 | -3.63 / 0.09 |
| sh_press | +1.15 / 0.41 | +1.15 / 0.41 | +1.15 / 0.41 | +1.15 / 0.41 | +0.05 / 0.56 | -0.76 / 1.06 |
| sh_vis | +3.10 / 0.12 | +3.10 / 0.12 | +3.10 / 0.12 | +3.10 / 0.12 | +2.69 / 0.19 | +2.14 / 0.38 |
| sh_dive | -4.00 / 0.62 | -5.44 / 0.73 | -5.36 / 0.78 | -6.61 / 0.80 | -5.09 / 0.75 | -5.29 / 0.88 |
| sh_dive_edge | +1.26 / 0.39 | +1.24 / 0.38 | +1.21 / 0.39 | +1.14 / 0.38 | +0.16 / 0.55 | -0.62 / 1.05 |
| sh_vis_dive | -1.19 / 0.19 | -2.69 / 0.28 | -2.21 / 0.19 | -3.94 / 0.31 | -1.86 / 0.30 | -1.89 / 0.34 |
| sh_bully | -11.45 / 0.70 | -11.87 / 0.70 | -13.30 / 0.83 | -14.33 / 0.88 | -12.30 / 1.02 | -12.39 / 0.98 |

## N10, equal: A's value J (B last-hits) / A deaths per 40 turns

| A | econ | tw_aggro | tw2 | tw2_aggro | gk_pos16 | gk_dmg08 |
|---|---|---|---|---|---|---|
| lh | +0.42 / 0.06 | +0.42 / 0.06 | +0.42 / 0.06 | +0.42 / 0.06 | +0.39 / 0.08 | +0.38 / 0.16 |
| sh_press | +3.23 / 0.48 | +3.23 / 0.48 | +3.23 / 0.48 | +3.23 / 0.48 | +2.09 / 0.78 | +1.84 / 0.95 |
| sh_vis | +4.94 / 0.28 | +4.94 / 0.28 | +4.94 / 0.28 | +4.94 / 0.28 | +4.35 / 0.42 | +4.16 / 0.47 |
| sh_dive | -0.14 / 0.48 | -1.78 / 0.58 | -1.67 / 0.72 | -3.66 / 0.94 | -1.47 / 0.73 | -1.71 / 0.92 |
| sh_dive_edge | +3.07 / 0.48 | +2.26 / 0.55 | +2.34 / 0.59 | +1.66 / 0.67 | +2.10 / 0.72 | +1.44 / 0.89 |
| sh_vis_dive | +1.59 / 0.25 | +0.74 / 0.22 | +0.81 / 0.25 | -0.92 / 0.28 | +1.34 / 0.25 | +1.22 / 0.31 |
| sh_bully | -6.69 / 0.69 | -7.45 / 0.77 | -8.64 / 0.88 | -9.04 / 0.91 | -7.69 / 0.94 | -7.70 / 1.03 |

## N10, A x1.76: A's value J (B last-hits) / A deaths per 40 turns

| A | econ | tw_aggro | tw2 | tw2_aggro | gk_pos16 | gk_dmg08 |
|---|---|---|---|---|---|---|
| lh | +0.69 / 0.00 | +0.69 / 0.00 | +0.69 / 0.00 | +0.69 / 0.00 | +0.81 / 0.03 | +1.08 / 0.02 |
| sh_press | +5.20 / 0.11 | +5.20 / 0.11 | +5.20 / 0.11 | +5.20 / 0.11 | +4.48 / 0.20 | +4.57 / 0.14 |
| sh_vis | +6.69 / 0.02 | +6.69 / 0.02 | +6.69 / 0.02 | +6.69 / 0.02 | +6.37 / 0.00 | +6.30 / 0.03 |
| sh_dive | +5.05 / 0.08 | +2.95 / 0.14 | +2.61 / 0.33 | -0.42 / 0.42 | +3.69 / 0.14 | +4.22 / 0.23 |
| sh_dive_edge | +6.90 / 0.08 | +5.51 / 0.11 | +5.39 / 0.12 | +3.38 / 0.33 | +5.74 / 0.12 | +6.02 / 0.25 |
| sh_vis_dive | +6.63 / 0.03 | +4.04 / 0.03 | +4.52 / 0.00 | +1.37 / 0.09 | +5.82 / 0.02 | +6.22 / 0.06 |
| sh_bully | -1.17 / 0.22 | -1.21 / 0.22 | -3.88 / 0.34 | -3.74 / 0.39 | -1.81 / 0.22 | -2.44 / 0.50 |

## N10, A weak (B x1.76): A's value J (B last-hits) / A deaths per 40 turns

| A | econ | tw_aggro | tw2 | tw2_aggro | gk_pos16 | gk_dmg08 |
|---|---|---|---|---|---|---|
| lh | -0.61 / 0.11 | -0.61 / 0.11 | -0.61 / 0.11 | -0.61 / 0.11 | -0.44 / 0.11 | -0.78 / 0.23 |
| sh_press | +3.05 / 0.50 | +3.05 / 0.50 | +3.05 / 0.50 | +3.05 / 0.50 | +2.00 / 0.78 | +1.70 / 0.95 |
| sh_vis | +4.89 / 0.28 | +4.89 / 0.28 | +4.89 / 0.28 | +4.89 / 0.28 | +4.25 / 0.45 | +4.10 / 0.45 |
| sh_dive | -1.10 / 0.53 | -2.50 / 0.64 | -2.59 / 0.81 | -4.03 / 0.95 | -2.06 / 0.80 | -2.63 / 0.86 |
| sh_dive_edge | +3.22 / 0.48 | +3.01 / 0.52 | +3.18 / 0.47 | +3.07 / 0.47 | +2.05 / 0.77 | +1.64 / 0.97 |
| sh_vis_dive | +0.78 / 0.25 | -0.17 / 0.22 | -0.11 / 0.27 | -1.51 / 0.30 | +0.46 / 0.33 | +0.30 / 0.34 |
| sh_bully | -8.49 / 0.66 | -8.82 / 0.69 | -10.41 / 0.80 | -10.25 / 0.86 | -8.93 / 0.84 | -9.37 / 0.84 |

## S4, equal: A's value J (B last-hits) / A deaths per 40 turns

| A | econ | tw_aggro | tw2 | tw2_aggro | gk_pos16 | gk_dmg08 |
|---|---|---|---|---|---|---|
| lh | -2.92 / 0.06 | -2.92 / 0.06 | -2.92 / 0.06 | -2.92 / 0.06 | -3.00 / 0.12 | -3.06 / 0.14 |
| sh_press | +0.54 / 0.47 | +0.54 / 0.47 | +0.54 / 0.47 | +0.54 / 0.47 | -0.45 / 0.66 | -0.93 / 1.06 |
| sh_vis | +1.96 / 0.23 | +1.96 / 0.23 | +1.96 / 0.23 | +1.96 / 0.23 | +1.65 / 0.36 | +1.40 / 0.30 |
| sh_dive | -3.30 / 0.44 | -4.77 / 0.75 | -4.50 / 0.61 | -6.37 / 0.91 | -4.18 / 0.56 | -5.18 / 1.00 |
| sh_dive_edge | +0.36 / 0.52 | +0.18 / 0.47 | +0.29 / 0.48 | -0.15 / 0.56 | -0.40 / 0.66 | -0.68 / 0.80 |
| sh_vis_dive | -1.31 / 0.27 | -2.25 / 0.31 | -2.00 / 0.22 | -3.86 / 0.27 | -1.68 / 0.41 | -1.94 / 0.50 |
| sh_bully | -10.07 / 0.91 | -10.38 / 0.91 | -11.93 / 1.12 | -11.51 / 0.94 | -10.82 / 0.94 | -10.92 / 1.14 |

## S4, A x1.76: A's value J (B last-hits) / A deaths per 40 turns

| A | econ | tw_aggro | tw2 | tw2_aggro | gk_pos16 | gk_dmg08 |
|---|---|---|---|---|---|---|
| lh | -2.42 / 0.00 | -2.42 / 0.00 | -2.42 / 0.00 | -2.42 / 0.00 | -2.41 / 0.00 | -2.26 / 0.02 |
| sh_press | +2.56 / 0.11 | +2.56 / 0.11 | +2.56 / 0.11 | +2.56 / 0.11 | +2.00 / 0.12 | +1.95 / 0.12 |
| sh_vis | +4.05 / 0.00 | +4.05 / 0.00 | +4.05 / 0.00 | +4.05 / 0.00 | +3.84 / 0.00 | +3.73 / 0.00 |
| sh_dive | +1.55 / 0.11 | -0.69 / 0.17 | -0.28 / 0.23 | -3.32 / 0.48 | +0.82 / 0.08 | +0.23 / 0.34 |
| sh_dive_edge | +3.31 / 0.09 | +1.92 / 0.14 | +2.31 / 0.19 | +0.75 / 0.23 | +2.71 / 0.16 | +2.44 / 0.31 |
| sh_vis_dive | +3.35 / 0.00 | +1.03 / 0.05 | +1.02 / 0.08 | -1.17 / 0.17 | +2.60 / 0.03 | +2.57 / 0.03 |
| sh_bully | -5.96 / 0.30 | -4.59 / 0.12 | -6.82 / 0.42 | -6.87 / 0.41 | -6.15 / 0.34 | -5.99 / 0.50 |

## S4, A weak (B x1.76): A's value J (B last-hits) / A deaths per 40 turns

| A | econ | tw_aggro | tw2 | tw2_aggro | gk_pos16 | gk_dmg08 |
|---|---|---|---|---|---|---|
| lh | -3.57 / 0.06 | -3.57 / 0.06 | -3.57 / 0.06 | -3.57 / 0.06 | -3.82 / 0.11 | -3.79 / 0.14 |
| sh_press | +0.43 / 0.47 | +0.43 / 0.47 | +0.43 / 0.47 | +0.43 / 0.47 | -0.54 / 0.66 | -1.08 / 1.06 |
| sh_vis | +1.84 / 0.23 | +1.84 / 0.23 | +1.84 / 0.23 | +1.84 / 0.23 | +1.51 / 0.36 | +1.28 / 0.30 |
| sh_dive | -3.86 / 0.45 | -5.33 / 0.67 | -5.22 / 0.62 | -6.85 / 0.81 | -4.98 / 0.61 | -5.54 / 0.92 |
| sh_dive_edge | +0.47 / 0.45 | +0.49 / 0.44 | +0.40 / 0.45 | +0.39 / 0.45 | -0.51 / 0.64 | -1.04 / 1.05 |
| sh_vis_dive | -2.15 / 0.34 | -2.94 / 0.33 | -2.66 / 0.20 | -4.31 / 0.30 | -2.68 / 0.48 | -2.79 / 0.52 |
| sh_bully | -11.53 / 0.70 | -12.20 / 0.78 | -13.09 / 0.92 | -13.23 / 0.88 | -12.16 / 0.91 | -12.37 / 1.06 |

## The neutral lane and the tower farmer under each variant (N10 / S10, equal, both Last Hit, 40 turns)

| variant | N10 B deaths | N10 B HP lost | N10 B away | S10 B deaths | S10 B HP lost | S10 B away |
|---|---|---|---|---|---|---|
| econ | 0.06 | 0.74 | 1.56 | 0.09 | 0.82 | 1.62 |
| tw_aggro | 0.06 | 0.74 | 1.56 | 0.09 | 0.82 | 1.62 |
| tw2 | 0.06 | 0.74 | 1.56 | 0.09 | 0.82 | 1.62 |
| tw2_aggro | 0.06 | 0.74 | 1.56 | 0.09 | 0.82 | 1.62 |
| gk_pos16 | 0.09 | 0.91 | 2.06 | 0.17 | 1.02 | 2.30 |
| gk_dmg08 | 0.16 | 0.83 | 1.81 | 0.23 | 0.91 | 2.23 |

## Restricted games of the crash scenarios (A = 12 attackers, B = 7 defenders; J)

| start | gap | variant | value | A's equilibrium | B's equilibrium |
|---|---|---|---|---|---|
| S10 | equal | econ | -1.40 | sh_press 1.00 | shove 1.00 |
| S10 | equal | tw_aggro | -1.40 | sh_press 1.00 | shove 1.00 |
| S10 | equal | tw2 | -1.40 | sh_press 1.00 | shove 1.00 |
| S10 | equal | tw2_aggro | -1.40 | sh_press 1.00 | shove 1.00 |
| S10 | equal | gk_pos16 | -1.57 | sh_press 1.00 | shove 1.00 |
| S10 | equal | gk_dmg08 | -1.62 | sh_press 1.00 | shove 1.00 |
| S10 | A x1.76 | econ | +2.68 | sh_vis 1.00 | shove 1.00 |
| S10 | A x1.76 | tw_aggro | +2.68 | sh_vis 1.00 | shove 1.00 |
| S10 | A x1.76 | tw2 | +2.68 | sh_vis 1.00 | shove 1.00 |
| S10 | A x1.76 | tw2_aggro | +2.68 | sh_vis 1.00 | shove 1.00 |
| S10 | A x1.76 | gk_pos16 | +2.51 | sh_vis 1.00 | thin 1.00 |
| S10 | A x1.76 | gk_dmg08 | +2.33 | sh_vis 1.00 | shove 1.00 |
| S10 | A weak (B x1.76) | econ | -5.34 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | A weak (B x1.76) | tw_aggro | -5.35 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | A weak (B x1.76) | tw2 | -5.35 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | A weak (B x1.76) | tw2_aggro | -5.43 | sh_press 1.00 | shove 1.00 |
| S10 | A weak (B x1.76) | gk_pos16 | -5.08 | sh_dive_edge 1.00 | shove 1.00 |
| S10 | A weak (B x1.76) | gk_dmg08 | -5.35 | sh_dive_edge 1.00 | shove 1.00 |
| N10 | equal | econ | +0.14 | sh_press 1.00 | shove 1.00 |
| N10 | equal | tw_aggro | +0.14 | sh_press 1.00 | shove 1.00 |
| N10 | equal | tw2 | +0.14 | sh_press 1.00 | shove 1.00 |
| N10 | equal | tw2_aggro | +0.14 | sh_press 1.00 | shove 1.00 |
| N10 | equal | gk_pos16 | +0.17 | sh_press 1.00 | shove 1.00 |
| N10 | equal | gk_dmg08 | +0.14 | sh_press 1.00 | shove 1.00 |
| N10 | A x1.76 | econ | +3.96 | sh_press 0.70, sh_vis 0.30 | thin 0.17, shove 0.83 |
| N10 | A x1.76 | tw_aggro | +3.96 | sh_press 0.70, sh_vis 0.30 | thin 0.17, shove 0.83 |
| N10 | A x1.76 | tw2 | +3.96 | sh_press 0.70, sh_vis 0.30 | thin 0.17, shove 0.83 |
| N10 | A x1.76 | tw2_aggro | +3.96 | sh_press 0.70, sh_vis 0.30 | thin 0.17, shove 0.83 |
| N10 | A x1.76 | gk_pos16 | +3.78 | sh_press 0.49, sh_vis 0.51 | thin 0.11, shove 0.89 |
| N10 | A x1.76 | gk_dmg08 | +3.74 | sh_press 0.52, sh_vis 0.48 | thin 0.19, shove 0.81 |
| N10 | A weak (B x1.76) | econ | -3.91 | sh_dive_edge 1.00 | shove 1.00 |
| N10 | A weak (B x1.76) | tw_aggro | -3.92 | sh_dive_edge 1.00 | shove 1.00 |
| N10 | A weak (B x1.76) | tw2 | -4.01 | sh_vis 1.00 | shove 1.00 |
| N10 | A weak (B x1.76) | tw2_aggro | -4.01 | sh_vis 1.00 | shove 1.00 |
| N10 | A weak (B x1.76) | gk_pos16 | -3.57 | sh_dive_edge 1.00 | shove 1.00 |
| N10 | A weak (B x1.76) | gk_dmg08 | -3.69 | sh_dive_edge 1.00 | shove 1.00 |
| S4 | equal | econ | -1.70 | sh_vis 1.00 | shove 1.00 |
| S4 | equal | tw_aggro | -1.70 | sh_vis 1.00 | shove 1.00 |
| S4 | equal | tw2 | -1.70 | sh_vis 1.00 | shove 1.00 |
| S4 | equal | tw2_aggro | -1.70 | sh_vis 1.00 | shove 1.00 |
| S4 | equal | gk_pos16 | -1.73 | sh_vis 1.00 | shove 1.00 |
| S4 | equal | gk_dmg08 | -1.97 | sh_press 0.97, sh_vis 0.03 | respect 0.02, shove 0.98 |
| S4 | A x1.76 | econ | +2.71 | sh_vis 1.00 | thin 1.00 |
| S4 | A x1.76 | tw_aggro | +2.71 | sh_vis 1.00 | thin 1.00 |
| S4 | A x1.76 | tw2 | +2.71 | sh_vis 1.00 | thin 1.00 |
| S4 | A x1.76 | tw2_aggro | +2.71 | sh_vis 1.00 | thin 1.00 |
| S4 | A x1.76 | gk_pos16 | +2.46 | sh_vis 1.00 | thin 1.00 |
| S4 | A x1.76 | gk_dmg08 | +2.34 | sh_vis 1.00 | thin 1.00 |
| S4 | A weak (B x1.76) | econ | -5.34 | sh_dive_edge 1.00 | shove 1.00 |
| S4 | A weak (B x1.76) | tw_aggro | -5.35 | sh_dive_edge 1.00 | shove 1.00 |
| S4 | A weak (B x1.76) | tw2 | -5.35 | sh_dive_edge 1.00 | shove 1.00 |
| S4 | A weak (B x1.76) | tw2_aggro | -5.40 | sh_press 1.00 | shove 1.00 |
| S4 | A weak (B x1.76) | gk_pos16 | -5.46 | sh_press 1.00 | shove 1.00 |
| S4 | A weak (B x1.76) | gk_dmg08 | -5.33 | sh_dive_edge 1.00 | shove 1.00 |
