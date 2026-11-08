# NC1 generated tables

Pre-registration: docs/capacity/nc1_preregistration.md (SHA-256 75adc051058e301832426186620b5c2673f9fdbb084f1e76e1c12f6e559a5916).

## Smoke: pairing with M5 and throughput

| run | status | updates identical to M5 | first rollout identical | steady transitions/s |
|---|---|---|---|---|
| C_s0 | complete | 25 | True | 3664 |
| C_s1 | complete | 25 | True | 3638 |
| W_s0 | complete | 0 | True | 3322 |
| W_s1 | complete | 0 | True | 3443 |
| F_s0 | complete | 0 | True | 3499 |
| F_s1 | complete | 0 | True | 3505 |

Checks: {'control_reproduces_m5': True, 'arms_first_rollout_identical': True, 'all_complete': True}.

## Q1. Unit usage of the stored networks

| network | inputs | states | hidden | L1 inactive | L1 saturation | L1 eff. rank | L2 inactive | L2 saturation | L2 eff. rank | L2 components 99% |
|---|---|---|---|---|---|---|---|---|---|---|
| sp:0 | 90 | 6200 | 128 × 128 | 0 | 0.02 | 74 | 0 | 0.62 | 92 | 107 |
| sp:1 | 90 | 6200 | 128 × 128 | 0 | 0.02 | 75 | 0 | 0.61 | 91 | 106 |
| sp:2 | 90 | 6200 | 128 × 128 | 0 | 0.02 | 75 | 0 | 0.61 | 93 | 109 |
| sp:3 | 90 | 6200 | 128 × 128 | 0 | 0.02 | 75 | 0 | 0.61 | 94 | 110 |
| sp:4 | 90 | 6200 | 128 × 128 | 0 | 0.02 | 75 | 0 | 0.62 | 93 | 107 |
| sp:5 | 90 | 6200 | 128 × 128 | 0 | 0.02 | 75 | 0 | 0.62 | 92 | 108 |
| om:0 | 116 | 6200 | 128 × 128 | 0 | 0.02 | 81 | 1 | 0.64 | 93 | 107 |
| om:1 | 116 | 6200 | 128 × 128 | 0 | 0.02 | 81 | 1 | 0.62 | 92 | 105 |
| om:2 | 116 | 6200 | 128 × 128 | 0 | 0.03 | 82 | 1 | 0.67 | 91 | 101 |
| om:3 | 116 | 6200 | 128 × 128 | 0 | 0.03 | 80 | 0 | 0.63 | 94 | 107 |
| om:4 | 116 | 6200 | 128 × 128 | 0 | 0.02 | 81 | 1 | 0.63 | 93 | 106 |
| om:5 | 116 | 6200 | 128 × 128 | 0 | 0.03 | 81 | 2 | 0.66 | 92 | 105 |
| ext:0 | 90 | 6200 | 128 × 128 | 0 | 0.00 | 66 | 0 | 0.33 | 84 | 102 |
| ext:2 | 90 | 6200 | 128 × 128 | 0 | 0.00 | 66 | 0 | 0.34 | 82 | 99 |
| ext:4 | 90 | 6200 | 128 × 128 | 0 | 0.00 | 66 | 0 | 0.36 | 84 | 101 |
| gap_attacker:0 | 90 | 6200 | 128 × 128 | 0 | 0.00 | 68 | 0 | 0.38 | 86 | 105 |
| gap_attacker:2 | 90 | 6200 | 128 × 128 | 0 | 0.00 | 68 | 0 | 0.38 | 85 | 103 |
| gap_attacker:4 | 90 | 6200 | 128 × 128 | 0 | 0.00 | 66 | 0 | 0.41 | 85 | 103 |

Rule (layer 2, median over networks): {'median_inactive_layer2': 0.0, 'median_saturation_layer2': 0.6160553175403226, 'leaves_width_unused': True}.

## Q2. The capacity probe

Lost specialization λ (0 = keeps all of the specialist's difference; 1 = plays like the generalist), held out, mean of three fit seeds:

| target | D = KL(A‖G) | KL(G‖A) natural | floor F_A | floor F_G | λ U64 | λ U128 | λ U256 | λ U3x128 | λ U128F |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.601 | 0.565 | 0.0090 | 0.0042 | 0.078 | 0.035 | 0.016 | 0.015 | 0.025 |
| 2 | 0.693 | 0.611 | 0.0088 | 0.0036 | 0.063 | 0.029 | 0.013 | 0.013 | 0.020 |
| 4 | 0.639 | 0.577 | 0.0097 | 0.0043 | 0.069 | 0.031 | 0.014 | 0.012 | 0.021 |

Natural excess κ (nats, KL to G on held-out natural states above the F_G floor):

| target | κ U64 | κ U128 | κ U256 | κ U3x128 | κ U128F | U128 epochs |
|---|---|---|---|---|---|---|
| 0 | +0.0156 | +0.0064 | +0.0021 | +0.0028 | +0.0059 | 150 / 150 / 150 |
| 2 | +0.0161 | +0.0074 | +0.0033 | +0.0045 | +0.0068 | 150 / 150 / 150 |
| 4 | +0.0157 | +0.0070 | +0.0025 | +0.0035 | +0.0055 | 150 / 150 / 150 |

Behaviour on E4's held-out gap states (G100, cells and semi-natural pooled, 95% interval over states):

| target | states | attacker − G | U64 − G | U128 − G | U256 − G | U3x128 − G | U128F − G | U256 − U128 | U128F − U128 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 75 | +2.34 ± 0.40 | +2.11 ± 0.36 | +2.14 ± 0.34 | +2.32 ± 0.43 | +2.29 ± 0.39 | +2.41 ± 0.40 | +0.18 ± 0.35 | +0.27 ± 0.35 |
| 2 | 75 | +1.84 ± 0.43 | +1.63 ± 0.48 | +1.86 ± 0.44 | +1.95 ± 0.41 | +1.81 ± 0.41 | +1.77 ± 0.40 | +0.09 ± 0.34 | -0.09 ± 0.37 |
| 4 | 75 | +2.14 ± 0.35 | +2.00 ± 0.33 | +2.21 ± 0.36 | +2.37 ± 0.36 | +2.34 ± 0.34 | +2.56 ± 0.37 | +0.16 ± 0.33 | +0.35 ± 0.35 |

Pooled retention ρ: U64 0.91, U128 0.98, U256 1.05, U3x128 1.02, U128F 1.07.

Natural play against G from s0 (64 paired seeds, W = 460):

| student | estimate | 95% interval |
|---|---|---|
| t0_U64 | -0.06 | [-0.53, +0.41] |
| t0_U128 | +0.23 | [-0.23, +0.69] |
| t0_U256 | +0.24 | [-0.22, +0.71] |
| t0_U3x128 | +0.19 | [-0.22, +0.60] |
| t0_U128F | -0.18 | [-0.63, +0.27] |
| t2_U64 | -0.41 | [-0.83, +0.02] |
| t2_U128 | +0.21 | [-0.23, +0.65] |
| t2_U256 | +0.23 | [-0.26, +0.71] |
| t2_U3x128 | -0.07 | [-0.61, +0.48] |
| t2_U128F | +0.31 | [-0.17, +0.79] |
| t4_U64 | -0.37 | [-0.95, +0.22] |
| t4_U128 | -0.25 | [-0.81, +0.32] |
| t4_U256 | -0.23 | [-0.71, +0.24] |
| t4_U3x128 | -0.07 | [-0.55, +0.41] |
| t4_U128F | +0.10 | [-0.36, +0.56] |

Reproduction of E4's records: {'checked': 450, 'mismatches': [], 'identical': True}.

Decision rules: {'2x128 suffices': True, 'width binds': False, 'features help': False, 'partial': False}. Natural regressions: none.

## Q3. Value capacity on M4.6e's labels

| network | P1 edge + plaus | P1 large + gap | natural MAE |
|---|---|---|---|
| 2x128 (M4.6e C) | +0.75 ± 0.26 | +0.84 ± 0.12 | +2.23 ± 0.56 |
| 2x128+features (M4.6e D) | +1.48 ± 0.54 | +1.13 ± 0.11 | +2.23 ± 0.54 |
| 2x64 | +0.84 ± 0.26 | +0.91 ± 0.13 | +2.23 ± 0.57 |
| 2x256 | +0.75 ± 0.25 | +0.80 ± 0.12 | +2.22 ± 0.57 |
| 3x128 | +0.85 ± 0.32 | +0.88 ± 0.11 | +2.27 ± 0.57 |

Minus 2 × 128 (paired by policy):

| network | P1 edge + plaus | P1 large + gap | natural MAE |
|---|---|---|---|
| 2x128+features (M4.6e D) | +0.73 ± 0.30 | +0.29 ± 0.10 | -0.001 ± 0.040 |
| 2x64 | +0.09 ± 0.04 | +0.07 ± 0.03 | -0.005 ± 0.018 |
| 2x256 | -0.00 ± 0.03 | -0.04 ± 0.06 | -0.006 ± 0.005 |
| 3x128 | +0.10 ± 0.09 | +0.04 ± 0.08 | +0.039 ± 0.015 |

Width binds for the value: False (2 × 256 natural MAE ratio 0.997). Reproduction of M4.6e's stored 2 × 128 fit: True (max difference 0.0).

## Stage 2. Training

| run | status | transitions | updates | hours | transitions/s |
|---|---|---|---|---|---|
| F_s0 | complete | 16384000 | 4000 | 1.66 | 2748 |
| F_s1 | complete | 16384000 | 4000 | 1.66 | 2743 |
| F_s2 | complete | 16384000 | 4000 | 1.66 | 2738 |
| F_s3 | complete | 16384000 | 4000 | 1.66 | 2735 |
| F_s4 | complete | 16384000 | 4000 | 1.67 | 2733 |
| F_s5 | complete | 16384000 | 4000 | 1.67 | 2732 |
| W_s0 | complete | 16384000 | 4000 | 1.83 | 2487 |
| W_s1 | complete | 16384000 | 4000 | 1.83 | 2492 |
| W_s2 | complete | 16384000 | 4000 | 1.83 | 2490 |
| W_s3 | complete | 16384000 | 4000 | 1.83 | 2486 |
| W_s4 | complete | 16384000 | 4000 | 1.82 | 2494 |
| W_s5 | complete | 16384000 | 4000 | 1.82 | 2498 |

## Q1 on the pilot finals and their controls

| network | inputs | states | hidden | L1 inactive | L1 saturation | L1 eff. rank | L2 inactive | L2 saturation | L2 eff. rank | L2 components 99% |
|---|---|---|---|---|---|---|---|---|---|---|
| W:0 | 90 | 6200 | 256 × 256 | 0 | 0.01 | 114 | 0 | 0.60 | 181 | 218 |
| W:1 | 90 | 6200 | 256 × 256 | 0 | 0.01 | 116 | 0 | 0.64 | 178 | 209 |
| W:2 | 90 | 6200 | 256 × 256 | 0 | 0.01 | 116 | 2 | 0.64 | 179 | 210 |
| W:3 | 90 | 6200 | 256 × 256 | 0 | 0.01 | 114 | 1 | 0.64 | 178 | 209 |
| W:4 | 90 | 6200 | 256 × 256 | 0 | 0.01 | 115 | 0 | 0.65 | 181 | 213 |
| W:5 | 90 | 6200 | 256 × 256 | 0 | 0.01 | 113 | 0 | 0.64 | 177 | 209 |
| F:0 | 90 | 6200 | 128 × 128 | 0 | 0.02 | 76 | 0 | 0.61 | 92 | 109 |
| F:1 | 90 | 6200 | 128 × 128 | 0 | 0.03 | 76 | 0 | 0.60 | 94 | 112 |
| F:2 | 90 | 6200 | 128 × 128 | 0 | 0.03 | 77 | 0 | 0.61 | 93 | 109 |
| F:3 | 90 | 6200 | 128 × 128 | 0 | 0.03 | 76 | 0 | 0.62 | 94 | 110 |
| F:4 | 90 | 6200 | 128 × 128 | 0 | 0.03 | 77 | 0 | 0.60 | 94 | 111 |
| F:5 | 90 | 6200 | 128 × 128 | 0 | 0.03 | 76 | 0 | 0.63 | 92 | 105 |
| control:0 | 90 | 6200 | 128 × 128 | 0 | 0.01 | 72 | 0 | 0.61 | 93 | 110 |
| control:1 | 90 | 6200 | 128 × 128 | 0 | 0.01 | 75 | 0 | 0.60 | 93 | 108 |
| control:2 | 90 | 6200 | 128 × 128 | 0 | 0.02 | 73 | 0 | 0.60 | 93 | 108 |
| control:3 | 90 | 6200 | 128 × 128 | 0 | 0.02 | 75 | 0 | 0.61 | 93 | 108 |
| control:4 | 90 | 6200 | 128 × 128 | 0 | 0.01 | 74 | 0 | 0.61 | 94 | 110 |
| control:5 | 90 | 6200 | 128 × 128 | 0 | 0.01 | 75 | 0 | 0.61 | 93 | 109 |

## Stage 3. Evaluation (arm − paired control, mean over six seeds, 95% t interval)

E1, head to head from s0 (lead at turn 114, 256 paired-seat games per seed):

| arm | mean ± 95% | per seed |
|---|---|---|
| W | -0.67 ± 0.59 | -1.20, -1.14, -0.68, -0.06, -1.02, +0.10 |
| F | -0.17 ± 0.42 | -0.78, -0.51, +0.17, +0.04, -0.16, +0.19 |

E2, the limitation families (arm − control in seat A on the same rollouts, averaged over both seat-B networks):

| arm | family | readout | mean ± 95% | seeds |
|---|---|---|---|---|
| W | conv | arm - control | all | G150 | -0.47 ± 0.37 | 6 |
| W | conv | arm - control | all | G50 | -0.28 ± 0.24 | 6 |
| W | conv | arm - control | conversion | G150 | -0.50 ± 0.42 | 6 |
| W | conv | arm - control | conversion | G50 | -0.32 ± 0.28 | 6 |
| W | conv | arm - control | defence | G150 | -0.54 ± 0.36 | 6 |
| W | conv | arm - control | defence | G50 | -0.34 ± 0.21 | 6 |
| W | conv | arm - control | equal | G150 | -0.38 ± 0.60 | 6 |
| W | conv | arm - control | equal | G50 | -0.15 ± 0.42 | 6 |
| W | conv | arm - control | other (x1.76) | G150 | -0.28 ± 0.64 | 6 |
| W | conv | arm - control | other (x1.76) | G50 | -0.14 ± 0.51 | 6 |
| W | freeze | arm - control | all | G50 | -0.27 ± 0.19 | 6 |
| W | freeze | arm - control | breaker favourable (edge, plaus, hp) | G50 | -0.29 ± 0.18 | 6 |
| W | freeze | arm - control | breaker | G50 | -0.26 ± 0.17 | 6 |
| W | freeze | arm - control | freezer favourable (edge, plaus, hp) | G50 | -0.14 ± 0.20 | 6 |
| W | freeze | arm - control | freezer | G50 | -0.28 ± 0.26 | 6 |
| W | recall | arm - control | all | G100 | -0.31 ± 0.39 | 6 |
| W | recall | arm - control | all | G50 | -0.22 ± 0.33 | 6 |
| W | recall | arm - control | strong | G100 | -0.33 ± 0.42 | 6 |
| W | recall | arm - control | strong | G50 | -0.27 ± 0.33 | 6 |
| W | recall | arm - control | strong, advantaged | G100 | -0.29 ± 0.48 | 6 |
| W | recall | arm - control | strong, advantaged | G50 | -0.25 ± 0.36 | 6 |
| W | recall | arm - control | weak vs a stronger opponent | G100 | -0.30 ± 0.54 | 6 |
| W | recall | arm - control | weak vs a stronger opponent | G50 | -0.20 ± 0.48 | 6 |
| W | recall | arm - control | weak | G100 | -0.29 ± 0.41 | 6 |
| W | recall | arm - control | weak | G50 | -0.18 ± 0.37 | 6 |
| W | gaps | arm - control | all | G150 | -0.39 ± 0.35 | 6 |
| W | gaps | arm - control | all | G50 | -0.25 ± 0.32 | 6 |
| W | gaps | arm - control | equal cells | G150 | -0.27 ± 0.50 | 6 |
| W | gaps | arm - control | equal cells | G50 | -0.07 ± 0.39 | 6 |
| W | gaps | arm - control | gap cells | G150 | -0.44 ± 0.48 | 6 |
| W | gaps | arm - control | gap cells | G50 | -0.31 ± 0.43 | 6 |
| W | gaps | arm - control | gap equal | G150 | -0.27 ± 0.50 | 6 |
| W | gaps | arm - control | gap equal | G50 | -0.07 ± 0.39 | 6 |
| W | gaps | arm - control | gap x1.5 | G150 | -0.42 ± 0.35 | 6 |
| W | gaps | arm - control | gap x1.5 | G50 | -0.32 ± 0.26 | 6 |
| W | gaps | arm - control | gap x1.76 | G150 | -0.34 ± 0.51 | 6 |
| W | gaps | arm - control | gap x1.76 | G50 | -0.17 ± 0.63 | 6 |
| W | gaps | arm - control | gap x2.2 | G150 | -0.55 ± 0.79 | 6 |
| W | gaps | arm - control | gap x2.2 | G50 | -0.45 ± 0.59 | 6 |
| W | gaps | specialist - arm | all | G150 | +1.12 ± 0.40 | 6 |
| W | gaps | specialist - arm | all | G50 | +0.89 ± 0.23 | 6 |
| W | gaps | specialist - arm | equal cells | G150 | -0.41 ± 0.64 | 6 |
| W | gaps | specialist - arm | equal cells | G50 | -0.11 ± 0.49 | 6 |
| W | gaps | specialist - arm | gap cells | G150 | +1.63 ± 0.36 | 6 |
| W | gaps | specialist - arm | gap cells | G50 | +1.22 ± 0.25 | 6 |
| W | gaps | specialist - arm | gap equal | G150 | -0.41 ± 0.64 | 6 |
| W | gaps | specialist - arm | gap equal | G50 | -0.11 ± 0.49 | 6 |
| W | gaps | specialist - arm | gap x1.5 | G150 | +1.04 ± 0.38 | 6 |
| W | gaps | specialist - arm | gap x1.5 | G50 | +0.96 ± 0.38 | 6 |
| W | gaps | specialist - arm | gap x1.76 | G150 | +1.20 ± 0.52 | 6 |
| W | gaps | specialist - arm | gap x1.76 | G50 | +0.91 ± 0.37 | 6 |
| W | gaps | specialist - arm | gap x2.2 | G150 | +2.66 ± 0.45 | 6 |
| W | gaps | specialist - arm | gap x2.2 | G50 | +1.79 ± 0.23 | 6 |
| W | gaps | specialist - control | all | G150 | +0.40 ± 0.56 | 6 |
| W | gaps | specialist - control | all | G50 | +0.45 ± 0.47 | 6 |
| W | gaps | specialist - control | equal cells | G150 | -1.14 ± 0.74 | 6 |
| W | gaps | specialist - control | equal cells | G50 | -0.35 ± 0.46 | 6 |
| W | gaps | specialist - control | gap cells | G150 | +0.92 ± 0.60 | 6 |
| W | gaps | specialist - control | gap cells | G50 | +0.72 ± 0.57 | 6 |
| W | gaps | specialist - control | gap equal | G150 | -1.14 ± 0.74 | 6 |
| W | gaps | specialist - control | gap equal | G50 | -0.35 ± 0.46 | 6 |
| W | gaps | specialist - control | gap x1.5 | G150 | +0.07 ± 0.42 | 6 |
| W | gaps | specialist - control | gap x1.5 | G50 | +0.25 ± 0.41 | 6 |
| W | gaps | specialist - control | gap x1.76 | G150 | +0.68 ± 0.73 | 6 |
| W | gaps | specialist - control | gap x1.76 | G50 | +0.63 ± 0.75 | 6 |
| W | gaps | specialist - control | gap x2.2 | G150 | +2.00 ± 0.76 | 6 |
| W | gaps | specialist - control | gap x2.2 | G50 | +1.29 ± 0.85 | 6 |
| F | conv | arm - control | all | G150 | -0.05 ± 0.25 | 6 |
| F | conv | arm - control | all | G50 | +0.01 ± 0.17 | 6 |
| F | conv | arm - control | conversion | G150 | -0.13 ± 0.26 | 6 |
| F | conv | arm - control | conversion | G50 | -0.05 ± 0.21 | 6 |
| F | conv | arm - control | defence | G150 | -0.02 ± 0.35 | 6 |
| F | conv | arm - control | defence | G50 | +0.03 ± 0.29 | 6 |
| F | conv | arm - control | equal | G150 | -0.02 ± 0.26 | 6 |
| F | conv | arm - control | equal | G50 | -0.03 ± 0.16 | 6 |
| F | conv | arm - control | other (x1.76) | G150 | +0.26 ± 0.46 | 6 |
| F | conv | arm - control | other (x1.76) | G50 | +0.38 ± 0.28 | 6 |
| F | freeze | arm - control | all | G50 | +0.12 ± 0.16 | 6 |
| F | freeze | arm - control | breaker favourable (edge, plaus, hp) | G50 | +0.12 ± 0.25 | 6 |
| F | freeze | arm - control | breaker | G50 | +0.14 ± 0.24 | 6 |
| F | freeze | arm - control | freezer favourable (edge, plaus, hp) | G50 | +0.06 ± 0.15 | 6 |
| F | freeze | arm - control | freezer | G50 | +0.11 ± 0.10 | 6 |
| F | recall | arm - control | all | G100 | +0.17 ± 0.11 | 6 |
| F | recall | arm - control | all | G50 | +0.19 ± 0.15 | 6 |
| F | recall | arm - control | strong | G100 | +0.04 ± 0.06 | 6 |
| F | recall | arm - control | strong | G50 | +0.12 ± 0.13 | 6 |
| F | recall | arm - control | strong, advantaged | G100 | +0.12 ± 0.07 | 6 |
| F | recall | arm - control | strong, advantaged | G50 | +0.21 ± 0.24 | 6 |
| F | recall | arm - control | weak vs a stronger opponent | G100 | +0.46 ± 0.27 | 6 |
| F | recall | arm - control | weak vs a stronger opponent | G50 | +0.35 ± 0.18 | 6 |
| F | recall | arm - control | weak | G100 | +0.30 ± 0.25 | 6 |
| F | recall | arm - control | weak | G50 | +0.26 ± 0.19 | 6 |
| F | gaps | arm - control | all | G150 | +0.24 ± 0.39 | 6 |
| F | gaps | arm - control | all | G50 | +0.31 ± 0.28 | 6 |
| F | gaps | arm - control | equal cells | G150 | -0.08 ± 0.29 | 6 |
| F | gaps | arm - control | equal cells | G50 | -0.08 ± 0.30 | 6 |
| F | gaps | arm - control | gap cells | G150 | +0.35 ± 0.57 | 6 |
| F | gaps | arm - control | gap cells | G50 | +0.44 ± 0.44 | 6 |
| F | gaps | arm - control | gap equal | G150 | -0.08 ± 0.29 | 6 |
| F | gaps | arm - control | gap equal | G50 | -0.08 ± 0.30 | 6 |
| F | gaps | arm - control | gap x1.5 | G150 | +0.22 ± 0.44 | 6 |
| F | gaps | arm - control | gap x1.5 | G50 | +0.24 ± 0.35 | 6 |
| F | gaps | arm - control | gap x1.76 | G150 | +0.11 ± 0.47 | 6 |
| F | gaps | arm - control | gap x1.76 | G50 | +0.36 ± 0.47 | 6 |
| F | gaps | arm - control | gap x2.2 | G150 | +0.73 ± 0.91 | 6 |
| F | gaps | arm - control | gap x2.2 | G50 | +0.73 ± 0.68 | 6 |
| F | gaps | specialist - arm | all | G150 | -0.21 ± 0.59 | 6 |
| F | gaps | specialist - arm | all | G50 | -0.13 ± 0.32 | 6 |
| F | gaps | specialist - arm | equal cells | G150 | -0.55 ± 0.55 | 6 |
| F | gaps | specialist - arm | equal cells | G50 | -0.04 ± 0.47 | 6 |
| F | gaps | specialist - arm | gap cells | G150 | -0.10 ± 0.64 | 6 |
| F | gaps | specialist - arm | gap cells | G50 | -0.17 ± 0.35 | 6 |
| F | gaps | specialist - arm | gap equal | G150 | -0.55 ± 0.55 | 6 |
| F | gaps | specialist - arm | gap equal | G50 | -0.04 ± 0.47 | 6 |
| F | gaps | specialist - arm | gap x1.5 | G150 | -0.43 ± 0.31 | 6 |
| F | gaps | specialist - arm | gap x1.5 | G50 | -0.16 ± 0.30 | 6 |
| F | gaps | specialist - arm | gap x1.76 | G150 | -0.21 ± 0.69 | 6 |
| F | gaps | specialist - arm | gap x1.76 | G50 | -0.32 ± 0.42 | 6 |
| F | gaps | specialist - arm | gap x2.2 | G150 | +0.35 ± 1.18 | 6 |
| F | gaps | specialist - arm | gap x2.2 | G50 | -0.02 ± 0.69 | 6 |
| F | gaps | specialist - control | all | G150 | +0.40 ± 0.56 | 6 |
| F | gaps | specialist - control | all | G50 | +0.45 ± 0.47 | 6 |
| F | gaps | specialist - control | equal cells | G150 | -1.14 ± 0.74 | 6 |
| F | gaps | specialist - control | equal cells | G50 | -0.35 ± 0.46 | 6 |
| F | gaps | specialist - control | gap cells | G150 | +0.92 ± 0.60 | 6 |
| F | gaps | specialist - control | gap cells | G50 | +0.72 ± 0.57 | 6 |
| F | gaps | specialist - control | gap equal | G150 | -1.14 ± 0.74 | 6 |
| F | gaps | specialist - control | gap equal | G50 | -0.35 ± 0.46 | 6 |
| F | gaps | specialist - control | gap x1.5 | G150 | +0.07 ± 0.42 | 6 |
| F | gaps | specialist - control | gap x1.5 | G50 | +0.25 ± 0.41 | 6 |
| F | gaps | specialist - control | gap x1.76 | G150 | +0.68 ± 0.73 | 6 |
| F | gaps | specialist - control | gap x1.76 | G50 | +0.63 ± 0.75 | 6 |
| F | gaps | specialist - control | gap x2.2 | G150 | +2.00 ± 0.76 | 6 |
| F | gaps | specialist - control | gap x2.2 | G50 | +1.29 ± 0.85 | 6 |

E3, the log duel ratio's held-out R² from the last hidden layer:

| network | R² |
|---|---|
| W - control | +0.058 ± 0.021 |
| W | +0.219 ± 0.016 |
| F - control | +0.092 ± 0.026 |
| F | +0.253 ± 0.021 |
| control | +0.161 ± 0.010 |

E3, the critic's predicted and the realized strength premium at the E4 gap cells (resources):

| network | predicted (V) | realized (G150, self-play) |
|---|---|---|
| control | +0.14 ± 0.09 | +5.37 ± 1.02 |
| W | +0.23 ± 0.08 | +5.31 ± 0.50 |
| F | +0.95 ± 0.25 | +4.73 ± 0.48 |

E3, training health over the last 200 updates (mean):

| run | entropy | approx KL | explained variance | clip fraction | grad norm |
|---|---|---|---|---|---|
| W_s0 | 1.587 | 0.0061 | 0.430 | 0.081 | 0.871 |
| W_s1 | 1.585 | 0.0062 | 0.436 | 0.082 | 0.889 |
| W_s2 | 1.615 | 0.0060 | 0.447 | 0.079 | 0.876 |
| W_s3 | 1.651 | 0.0062 | 0.473 | 0.081 | 0.890 |
| W_s4 | 1.644 | 0.0060 | 0.462 | 0.079 | 0.891 |
| W_s5 | 1.584 | 0.0059 | 0.405 | 0.079 | 0.884 |
| F_s0 | 1.668 | 0.0041 | 0.449 | 0.046 | 0.768 |
| F_s1 | 1.630 | 0.0041 | 0.419 | 0.047 | 0.764 |
| F_s2 | 1.669 | 0.0041 | 0.415 | 0.047 | 0.768 |
| F_s3 | 1.629 | 0.0041 | 0.439 | 0.048 | 0.758 |
| F_s4 | 1.653 | 0.0041 | 0.435 | 0.048 | 0.785 |
| F_s5 | 1.712 | 0.0041 | 0.421 | 0.046 | 0.762 |
| control_s0 | 1.684 | 0.0041 | 0.426 | 0.046 | 0.756 |
| control_s1 | 1.627 | 0.0040 | 0.412 | 0.045 | 0.769 |
| control_s2 | 1.670 | 0.0041 | 0.416 | 0.046 | 0.762 |
| control_s3 | 1.648 | 0.0041 | 0.425 | 0.047 | 0.754 |
| control_s4 | 1.712 | 0.0040 | 0.463 | 0.045 | 0.758 |
| control_s5 | 1.663 | 0.0041 | 0.415 | 0.046 | 0.763 |

Verdicts: {'W': {'E1': 'inconclusive', 'improves_gap_play': False, 'regressing_families': ['conv', 'gaps']}, 'F': {'E1': 'inconclusive', 'improves_gap_play': False, 'regressing_families': []}}.

