# M4.3d validation: main seeds (64 games, SeedSequence 79,000) / independent seeds (128 games, 79,500)

Columns: econ (own tower 0.8) / own tower 0.9 / the candidate m4_four_actions_econ_aggro() (main run: the identical configuration `tw_aggro`). Each cell: main / validation.

| quantity | econ | 0.9 | candidate |
|---|---|---|---|
| B efficiency, neutral lane (N10, both Last Hit, 40 turns) | +0.899 / +0.914 | +0.912 / +0.921 | +0.899 / +0.914 |
| B efficiency, uncontested crash (S10, shove and leave, 40 turns) | +0.907 / +0.909 | +0.984 / +0.985 | +0.907 / +0.909 |
| Δ B farm when crashed on (N10, Shove + Pressure minus Last Hit, 40 turns) | +0.18 / +0.00 | +1.53 / +1.43 | +0.18 / +0.00 |
| J of sh_press against Last Hit (N10, equal) | +3.23 / +2.84 | +0.70 / +0.33 | +3.23 / +2.84 |
| J of sh_press against Last Hit (S10, equal) | +1.27 / +0.85 | -1.46 / -1.84 | +1.27 / +0.85 |
| J of sh_vis against Last Hit (S10, equal) | +3.13 / +2.81 | +0.12 / -0.14 | +3.13 / +2.81 |
| J of sh_dive against Last Hit (S10, equal) | -3.09 / -3.72 | -4.80 / -5.44 | -4.49 / -4.61 |
| J of sh_press against Last Hit (S10, A x1.76) | +2.74 / +2.55 | -0.19 / -0.36 | +2.74 / +2.55 |
| J of sh_vis against Last Hit (S10, A x1.76) | +4.20 / +4.07 | +1.04 / +0.95 | +4.20 / +4.07 |
| J of sh_dive_edge against Last Hit (S10, A x1.76) | +4.47 / +3.21 | +2.34 / +1.11 | +2.82 / +1.86 |
| J of sh_vis_dive against Last Hit (S10, A x1.76) | +4.14 / +2.98 | +1.94 / +0.75 | +2.20 / +1.41 |
| value of the crash game (S10, equal) | -1.40 / -1.79 | -1.80 / -2.19 | -1.40 / -1.79 |
| value of the crash game (S10, A x1.76) | +2.68 / +2.52 | +0.67 / +0.47 | +2.68 / +2.52 |
| value of the crash game (S10, A weak (B x1.76)) | -5.34 / -5.36 | -4.51 / -4.47 | -5.35 / -5.39 |
| value of the crash game (N10, equal) | +0.14 / -0.03 | +0.07 / -0.10 | +0.14 / -0.03 |
| value of the crash game (N10, A x1.76) | +3.96 / +3.85 | +2.48 / +2.40 | +3.96 / +3.85 |
| value of the crash game (N10, A weak (B x1.76)) | -3.91 / -3.94 | -2.38 / -2.41 | -3.92 / -3.95 |

## Crash-game equilibria (validation seeds)

| start | gap | configuration | main: A / B | validation: A / B |
|---|---|---|---|---|
| S10 | equal | econ | sh_press 1.00 / shove 1.00 | sh_press 1.00 / shove 1.00 |
| S10 | equal | ot90 | sh_press 0.88, sh_vis 0.12 / thin 0.16, shove 0.84 | sh_press 0.91, sh_vis 0.09 / thin 0.12, shove 0.88 |
| S10 | equal | econ_aggro | sh_press 1.00 / shove 1.00 | sh_press 1.00 / shove 1.00 |
| S10 | A x1.76 | econ | sh_vis 1.00 / shove 1.00 | sh_vis 1.00 / shove 1.00 |
| S10 | A x1.76 | ot90 | sh_hunt 0.56, sh_vis 0.44 / thin 0.53, shove 0.47 | sh_hunt 0.64, sh_dive_edge 0.08, sh_vis 0.29 / lh 0.24, thin 0.69, shove 0.07 |
| S10 | A x1.76 | econ_aggro | sh_vis 1.00 / shove 1.00 | sh_vis 1.00 / shove 1.00 |
| S10 | A weak (B x1.76) | econ | sh_dive_edge 1.00 / shove 1.00 | sh_dive_edge 1.00 / shove 1.00 |
| S10 | A weak (B x1.76) | ot90 | sh_dive_edge 1.00 / shove 1.00 | sh_dive_edge 1.00 / shove 1.00 |
| S10 | A weak (B x1.76) | econ_aggro | sh_dive_edge 1.00 / shove 1.00 | sh_dive_edge 1.00 / shove 1.00 |
| N10 | equal | econ | sh_press 1.00 / shove 1.00 | sh_press 1.00 / shove 1.00 |
| N10 | equal | ot90 | lh 0.10, sh_press 0.90 / thin 0.27, shove 0.73 | sh_press 0.70, sh_vis 0.30 / thin 0.14, shove 0.86 |
| N10 | equal | econ_aggro | sh_press 1.00 / shove 1.00 | sh_press 1.00 / shove 1.00 |
| N10 | A x1.76 | econ | sh_press 0.70, sh_vis 0.30 / thin 0.17, shove 0.83 | sh_press 0.63, sh_vis 0.37 / thin 0.18, shove 0.82 |
| N10 | A x1.76 | ot90 | sh_press 0.09, sh_vis 0.91 / thin 0.24, shove 0.76 | sh_dive_edge 0.00, sh_vis 1.00 / thin 0.77, shove 0.23 |
| N10 | A x1.76 | econ_aggro | sh_press 0.70, sh_vis 0.30 / thin 0.17, shove 0.83 | sh_press 0.63, sh_vis 0.37 / thin 0.18, shove 0.82 |
| N10 | A weak (B x1.76) | econ | sh_dive_edge 1.00 / shove 1.00 | sh_dive_edge 1.00 / shove 1.00 |
| N10 | A weak (B x1.76) | ot90 | lh 0.82, sh_vis 0.18 / punish 0.07, shove 0.93 | lh 0.84, sh_vis 0.16 / punish 0.08, shove 0.92 |
| N10 | A weak (B x1.76) | econ_aggro | sh_dive_edge 1.00 / shove 1.00 | sh_dive_edge 1.00 / shove 1.00 |
