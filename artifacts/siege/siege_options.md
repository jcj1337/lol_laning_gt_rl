# Experiment A: Apply Pressure vs. Establish Vision across siege states (K = 32)

## Equilibrium siege decisions of A at reachable siege nodes (k <= K - 4), grouped

P(.) = equilibrium probability; 'P - V' = Q of the best Pressure action minus Q of the best Vision action against B's equilibrium strategy (resources); 'P - S' and 'V - S' likewise against the best Safe action.

### By siege mass

| siege mass | nodes | P(Pressure) | P(Vision) | P(disengage) | P - V | P - S | V - S |
|---|---|---|---|---|---|---|---|
|  1 HP | 6603 | 0.296 | 0.699 | 0.005 | -1.303 | 1.151 | 2.454 |
|  2 HP | 5140 | 0.35 | 0.647 | 0.003 | -1.098 | 1.236 | 2.334 |
|  3 HP | 3791 | 0.654 | 0.341 | 0.005 | 0.151 | 1.135 | 0.984 |
|  4 HP | 2970 | 0.575 | 0.42 | 0.005 | 0.071 | 1.199 | 1.129 |
|  5 HP | 2107 | 0.586 | 0.407 | 0.008 | -0.025 | 1.048 | 1.073 |
|  6 HP | 1572 | 0.595 | 0.397 | 0.008 | 0.012 | 1.102 | 1.09 |
|  7 HP | 1031 | 0.579 | 0.41 | 0.011 | -0.09 | 0.814 | 0.904 |
|  8 HP | 625 | 0.536 | 0.456 | 0.008 | -0.131 | 0.891 | 1.022 |
|  9 HP | 453 | 0.524 | 0.467 | 0.009 | -0.252 | 0.568 | 0.82 |
| 10 HP | 230 | 0.461 | 0.522 | 0.017 | -0.243 | 0.708 | 0.951 |
| 11 HP | 191 | 0.503 | 0.487 | 0.01 | -0.156 | 0.645 | 0.801 |
| 12 HP | 77 | 0.416 | 0.558 | 0.026 | -0.278 | 0.639 | 0.917 |
| 13 HP | 64 | 0.422 | 0.547 | 0.031 | -0.207 | 0.628 | 0.836 |
| 14 HP | 12 | 0.333 | 0.667 | 0 | -0.591 | 0.441 | 1.032 |
| 15 HP | 11 | 0.364 | 0.636 | 0 | -0.45 | 0.526 | 0.977 |
| 16 HP | 1 | 0 | 1 | 0 | -1.684 | -0.384 | 1.3 |
| 17 HP | 1 | 0 | 1 | 0 | -1.684 | -0.384 | 1.3 |

### By defender wave

| defender wave | nodes | P(Pressure) | P(Vision) | P(disengage) | P - V | P - S | V - S |
|---|---|---|---|---|---|---|---|
|  1 HP waiting | 1725 | 0.312 | 0.687 | 0.001 | -0.994 | 0.593 | 1.587 |
|  2 HP waiting | 1287 | 0.468 | 0.529 | 0.003 | -0.588 | 1.072 | 1.66 |
|  3 HP waiting | 3060 | 0.437 | 0.56 | 0.004 | -0.661 | 1.157 | 1.818 |
|  4 HP waiting | 2579 | 0.444 | 0.551 | 0.004 | -0.684 | 1.243 | 1.927 |
|  5 HP waiting | 3914 | 0.468 | 0.531 | 0.001 | -0.617 | 1.059 | 1.675 |
|  6 HP waiting | 3241 | 0.453 | 0.54 | 0.006 | -0.651 | 1.025 | 1.676 |
|  7 HP waiting | 2889 | 0.497 | 0.494 | 0.009 | -0.451 | 1.252 | 1.702 |
|  8 HP waiting | 2163 | 0.569 | 0.43 | 0 | -0.14 | 1.527 | 1.667 |
|  9 HP waiting | 1006 | 0.525 | 0.475 | 0 | -0.372 | 1.246 | 1.618 |
| 10 HP waiting | 463 | 0.577 | 0.417 | 0.006 | -0.192 | 1.308 | 1.5 |
| 11 HP waiting | 180 | 0.681 | 0.319 | 0 | 0.295 | 1.915 | 1.62 |
| 12 HP waiting | 180 | 0.664 | 0.336 | 0 | 0.321 | 2.131 | 1.809 |
| no accessible wave | 2192 | 0.423 | 0.553 | 0.024 | -0.541 | 0.767 | 1.308 |

### By plates left

| plates left | nodes | P(Pressure) | P(Vision) | P(disengage) | P - V | P - S | V - S |
|---|---|---|---|---|---|---|---|
| 0 | 3026 | 0.197 | 0.784 | 0.02 | -1.751 | 0.469 | 2.22 |
| 1 | 6140 | 0.389 | 0.606 | 0.005 | -0.812 | 1.129 | 1.941 |
| 2 | 7391 | 0.478 | 0.519 | 0.003 | -0.428 | 1.229 | 1.657 |
| 3 | 5825 | 0.562 | 0.434 | 0.003 | -0.182 | 1.248 | 1.429 |
| 4 | 2283 | 0.693 | 0.307 | 0 | 0.192 | 1.25 | 1.058 |
| 5 | 205 | 0.834 | 0.166 | 0 | 0.681 | 1.186 | 0.505 |
| 6 | 9 | 1 | 0 | 0 | 1.083 | 1.025 | -0.059 |

### By vision

| vision | nodes | P(Pressure) | P(Vision) | P(disengage) | P - V | P - S | V - S |
|---|---|---|---|---|---|---|---|
| active | 19208 | 0.596 | 0.397 | 0.007 | -0.183 | 1.188 | 1.371 |
| none | 5671 | 0.019 | 0.981 | 0 | -1.834 | 0.879 | 2.713 |

### By turns remaining

| turns remaining | nodes | P(Pressure) | P(Vision) | P(disengage) | P - V | P - S | V - S |
|---|---|---|---|---|---|---|---|
|  0- 7 | 16908 | 0.496 | 0.498 | 0.006 | -0.44 | 1.07 | 1.509 |
|  8-15 | 7498 | 0.405 | 0.591 | 0.004 | -0.778 | 1.22 | 1.998 |
| 16-23 | 468 | 0.293 | 0.707 | 0 | -1.37 | 1.215 | 2.585 |
| 24-31 | 5 | 0 | 1 | 0 | -3.119 | 0.561 | 3.68 |

### By thinnable

| thinnable | nodes | P(Pressure) | P(Vision) | P(disengage) | P - V | P - S | V - S |
|---|---|---|---|---|---|---|---|
| no | 2192 | 0.423 | 0.553 | 0.024 | -0.541 | 0.767 | 1.308 |
| yes | 22687 | 0.468 | 0.528 | 0.004 | -0.561 | 1.151 | 1.713 |

## Constructed siege probes (A's crashed mass under B's tower; B's waiting wave one cell behind its tower; pending plates = the crash's grant; existing vision = half the duration left)

Each probe is solved exactly from its state. Q = farm + plates + gank + continuation against B's equilibrium strategy, for the best action of each kind.

| mass | defender wave | vision | turns left | V* | A plays | best Pressure Q | best Vision Q | best Safe Q | P - V |
|---|---|---|---|---|---|---|---|---|---|
| 4 | - | no | 8 | 3.166 | SP-C+V 1.0 | 3.09 (SP-C+P) | 3.17 (SP-C+V) | 2.51 (SP-S) | -0.075 |
| 4 | - | no | 16 | 2.206 | SP-C+V 1.0 | 2.13 (SP-C+P) | 2.21 (SP-C+V) | 1.55 (SP-S) | -0.075 |
| 4 | - | yes | 8 | 5.116 | SP-C+P 1.0 | 5.12 (SP-C+P) | 4.18 (SP-C+V) | 4.17 (SP-S) | 0.938 |
| 4 | - | yes | 16 | 4.156 | SP-C+P 1.0 | 4.16 (SP-C+P) | 3.22 (SP-C+V) | 3.21 (SP-S) | 0.938 |
| 4 | 4 | no | 8 | 3.526 | SP-C+V 1.0 | 3.50 (SP-C+P) | 3.53 (SP-C+V) | 2.93 (SP-S) | -0.029 |
| 4 | 4 | no | 16 | 3.504 | SP-C+V 1.0 | 2.46 (SP-C+P) | 3.50 (SP-C+V) | 2.91 (SP-S) | -1.044 |
| 4 | 4 | yes | 8 | 5.522 | SP-C+P 1.0 | 5.52 (SP-C+P) | 4.54 (SP-C+V) | 4.60 (SP-S) | 0.984 |
| 4 | 4 | yes | 16 | 4.575 | SP-S 1.0 | 4.48 (SP-C+P) | 4.52 (SP-C+V) | 4.57 (SP-S) | -0.031 |
| 4 | 8 | no | 8 | 3.396 | SP-C+V 1.0 | 3.25 (SP-C+P) | 3.40 (SP-C+V) | 2.60 (SP-S) | -0.15 |
| 4 | 8 | no | 16 | 4.039 | SP-C+V 1.0 | 3.71 (SP-C+P) | 4.04 (SP-C+V) | 3.37 (SP-S) | -0.327 |
| 4 | 8 | yes | 8 | 5.421 | SP-C+P 1.0 | 5.42 (SP-C+P) | 4.48 (SP-C+V) | 4.42 (SP-S) | 0.938 |
| 4 | 8 | yes | 16 | 5.888 | SP-C+P 1.0 | 5.89 (SP-C+P) | 5.13 (SP-C+V) | 5.19 (SP-S) | 0.761 |
| 8 | - | no | 8 | 4.602 | SP-C+V 1.0 | 3.59 (SP-C+P) | 4.60 (SP-C+V) | 4.00 (SP-S) | -1.013 |
| 8 | - | no | 16 | 6.171 | SP-C+V 1.0 | 5.22 (SP-C+P) | 6.17 (SP-C+V) | 5.57 (SP-S) | -0.954 |
| 8 | - | yes | 8 | 5.674 | SP-S 1.0 | 5.61 (SP-C+P) | 5.61 (SP-C+V) | 5.67 (SP-S) | 0 |
| 8 | - | yes | 16 | 7.242 | SP-S 1.0 | 7.24 (SP-C+P) | 7.18 (SP-C+V) | 7.24 (SP-S) | 0.059 |
| 8 | 4 | no | 8 | 3.859 | SP-C+V 1.0 | 2.88 (SP-C+P) | 3.86 (SP-C+V) | 3.10 (SP-S) | -0.975 |
| 8 | 4 | no | 16 | 4.44 | SP-C+V 1.0 | 3.43 (SP-C+P) | 4.44 (SP-C+V) | 3.21 (SP-S) | -1.013 |
| 8 | 4 | yes | 8 | 4.946 | SP-C+P 1.0 | 4.95 (SP-C+P) | 4.87 (SP-C+V) | 4.80 (SP-S) | 0.075 |
| 8 | 4 | yes | 16 | 5.49 | SP-C+P 1.0 | 5.49 (SP-C+P) | 5.45 (SP-C+V) | 4.92 (SP-S) | 0.038 |
| 8 | 8 | no | 8 | 2.301 | SP-C+V 1.0 | 2.07 (SP-C+P) | 2.30 (SP-C+V) | 1.50 (SP-S) | -0.234 |
| 8 | 8 | no | 16 | 5.127 | SP-C+V 1.0 | 4.04 (SP-C+P) | 5.13 (SP-C+V) | 3.06 (SP-S) | -1.088 |
| 8 | 8 | yes | 8 | 4.241 | SP-C+P 1.0 | 4.24 (SP-C+P) | 3.39 (SP-C+V) | 3.32 (SP-S) | 0.853 |
| 8 | 8 | yes | 16 | 6.252 | SP-C+P 1.0 | 6.25 (SP-C+P) | 6.21 (SP-C+V) | 5.41 (SP-S) | 0.037 |
| 12 | - | no | 8 | 4.913 | SP-C+V 1.0 | 3.90 (SP-C+P) | 4.91 (SP-C+V) | 4.32 (SP-S) | -1.013 |
| 12 | - | no | 16 | 9.029 | SP-C+V 1.0 | 8.02 (SP-C+P) | 9.03 (SP-C+V) | 8.36 (SP-S) | -1.013 |
| 12 | - | yes | 8 | 5.984 | SP-S 1.0 | 5.93 (SP-C+P) | 5.93 (SP-C+V) | 5.98 (SP-S) | 0 |
| 12 | - | yes | 16 | 10.04 | SP-C+P 1.0 | 10.04 (SP-C+P) | 10.04 (SP-C+V) | 10.03 (SP-S) | 0 |
| 12 | 4 | no | 8 | 3.885 | SP-C+V 1.0 | 3.15 (SP-C+P) | 3.88 (SP-C+V) | 3.04 (SP-S) | -0.73 |
| 12 | 4 | no | 16 | 6.137 | SP-C+V 1.0 | 5.12 (SP-C+P) | 6.14 (SP-C+V) | 5.16 (SP-S) | -1.013 |
| 12 | 4 | yes | 8 | 5.217 | SP-C+P 1.0 | 5.22 (SP-C+P) | 4.90 (SP-C+V) | 4.74 (SP-S) | 0.32 |
| 12 | 4 | yes | 16 | 7.539 | SP-C+P 1.0 | 7.54 (SP-C+P) | 7.15 (SP-C+V) | 6.87 (SP-S) | 0.389 |
| 12 | 8 | no | 8 | 2.037 | SP-C+V 1.0 | 1.23 (SP-C+P) | 2.04 (SP-C+V) | 1.24 (SP-S) | -0.805 |
| 12 | 8 | no | 16 | 5.488 | SP-C+V 1.0 | 4.40 (SP-C+P) | 5.49 (SP-C+V) | 4.69 (SP-S) | -1.088 |
| 12 | 8 | yes | 8 | 3.407 | SP-C+P 1.0 | 3.41 (SP-C+P) | 3.12 (SP-C+V) | 3.06 (SP-S) | 0.282 |
| 12 | 8 | yes | 16 | 6.608 | SP-C+P 1.0 | 6.61 (SP-C+P) | 6.58 (SP-C+V) | 6.52 (SP-S) | 0.032 |

### Decomposition of the best action of each kind (mass 12, defender wave 4, 16 turns left)

**mass 12, defender 4, vision no, 16 turns left** (V* = 6.137; A plays {'SP-C+V': 1.0}; B plays {'SP-S': 1.0})

| action | Q | farm | plates | gank | continuation |
|---|---|---|---|---|---|
| SP-S | 5.16 | -0.4 | 0 | -0.125 | 5.685 |
| SH-S | 5.16 | -0.4 | 0 | -0.125 | 5.685 |
| SP-C+P | 5.125 | -0.215 | 1 | -0.6 | 4.94 |
| SH-C+P | 4.975 | -0.215 | 1 | -0.75 | 4.94 |
| SP-C+V | 6.137 | -0.34 | 0 | -0.6 | 7.077 |
| SH-C+V | 5.987 | -0.34 | 0 | -0.75 | 7.077 |

**mass 12, defender 4, vision yes, 16 turns left** (V* = 7.539; A plays {'SP-C+P': 1.0}; B plays {'SP-S': 1.0})

| action | Q | farm | plates | gank | continuation |
|---|---|---|---|---|---|
| SP-S | 6.866 | -0.4 | 0 | 0.531 | 6.735 |
| SH-S | 6.866 | -0.4 | 0 | 0.531 | 6.735 |
| SP-C+P | 7.539 | -0.215 | 1 | 0.412 | 6.341 |
| SH-C+P | 7.501 | -0.215 | 1 | 0.375 | 6.341 |
| SP-C+V | 7.15 | -0.34 | 0 | 0.412 | 7.077 |
| SH-C+V | 7.112 | -0.34 | 0 | 0.375 | 7.077 |

**mass 4, defender 4, vision no, 16 turns left** (V* = 3.504; A plays {'SP-C+V': 1.0}; B plays {'SP-S': 1.0})

| action | Q | farm | plates | gank | continuation |
|---|---|---|---|---|---|
| SP-S | 2.906 | -0.4 | 0 | -0.125 | 3.431 |
| SH-S | 2.906 | -0.4 | 0 | -0.125 | 3.431 |
| SP-C+P | 2.46 | -0.215 | 1 | -0.6 | 2.275 |
| SH-C+P | 2.31 | -0.215 | 1 | -0.75 | 2.275 |
| SP-C+V | 3.504 | -0.34 | 0 | -0.6 | 4.444 |
| SH-C+V | 3.354 | -0.34 | 0 | -0.75 | 4.444 |

**mass 8, defender 8, vision no, 16 turns left** (V* = 5.127; A plays {'SP-C+V': 1.0}; B plays {'SP-S': 1.0})

| action | Q | farm | plates | gank | continuation |
|---|---|---|---|---|---|
| SP-S | 3.061 | -0.4 | 0 | -0.325 | 3.786 |
| SH-S | 3.061 | -0.4 | 0 | -0.325 | 3.786 |
| SP-C+P | 4.04 | -0.215 | 1 | -0.8 | 4.055 |
| SH-C+P | 3.89 | -0.215 | 1 | -0.95 | 4.055 |
| SP-C+V | 5.127 | -0.34 | 0 | -0.8 | 6.267 |
| SH-C+V | 4.977 | -0.34 | 0 | -0.95 | 6.267 |

### Gank loss L (mass 12, defender wave 4, no vision, 16 turns left)

| L | V* | A plays | best Pressure Q | best Vision Q | best Safe Q |
|---|---|---|---|---|---|
| 3.5 | 4.439 | SP-C+V 1.0 | 3.74 | 4.439 | 3.761 |
| 5 | 6.137 | SP-C+V 1.0 | 5.125 | 6.137 | 5.16 |
| 7 | 8.427 | SP-C+V 1.0 | 7.01 | 8.427 | 7.015 |
