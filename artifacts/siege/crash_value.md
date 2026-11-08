# The value of crashing: counterfactuals at the pre-crash decision (K = 32)

Pre-crash state: A's packet of h HP one cell from B's tower, B's next wave just spawned at its base. Shove crashes it on the next tick; Slow Push holds it at +1 (a stacked packet creeps only on creep turns). Q-values are from the same state and horizon; 'crash gain' = Q(best Shove action) - Q(best non-crashing action), i.e. how much A gains or loses by producing the crash now instead of its best alternative.

## previous revision (choice, tower 1)

| crash size | turns left | V* pre-crash | A's equilibrium action | crash gain vs equilibrium B | crash gain vs passive B | crash gain vs stack_builder B | V(post-crash state) |
|---|---|---|---|---|---|---|---|
| small (1 wave) | 16 | -0.83 | SP-S 1.0 | +0.00 (SH-S vs SP-S) | -0.15 (SH-C vs SP-S) | +0.11 (SH-C vs SP-S) | -2.23 |
| small (1 wave) | 24 | -0.83 | SP-S 1.0 | +0.00 (SH-S vs SP-S) | +0.87 (SH-S vs SP-S) | +0.86 (SH-S vs SP-S) | -1.23 |
| medium (2 waves) | 16 | -1.74 | SP-S 1.0 | +0.00 (SH-S vs SP-S) | -0.02 (SH-S vs SP-C) | +0.00 (SH-S vs SP-S) | -3.54 |
| medium (2 waves) | 24 | -1.36 | SP-C 0.786, SH-C 0.214 | +0.00 (SH-C vs SP-C) | +0.09 (SH-C vs SP-C) | +0.00 (SH-S vs SP-S) | -2.54 |
| large (3 waves) | 16 | -1.66 | SH-C 1.0 | +0.61 (SH-C vs SP-C) | +0.72 (SH-C vs SP-C) | +0.51 (SH-C vs SP-C) | -6.47 |
| large (3 waves) | 24 | -0.66 | SH-C 1.0 | +0.62 (SH-C vs SP-C) | +0.72 (SH-C vs SP-C) | +0.47 (SH-C vs SP-C) | -5.47 |

**small (1 wave), 16 turns left: decomposition against B's equilibrium strategy** (B plays {'F-S': 1.0})

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | -1.01 | 0.02 | 0 | -0.475 | -0.555 | 0 |
| SH-C | -1.905 | 0.145 | 0 | -0.625 | -1.425 | 0 |
| SP-S | -0.83 | -0.15 | 0 | -0.125 | -0.555 | 0 |
| SH-S | -0.83 | -0.15 | 0 | -0.125 | -0.555 | 0 |

**... against passive_safe B**

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | 3.77 | 0.02 | 0 | -0.475 | 4.225 | 0 |
| SH-C | 3.8 | -0.18 | 0 | -0.75 | 4.73 | 0 |
| SP-S | 3.95 | -0.15 | 0 | -0.125 | 4.225 | 0 |
| SH-S | 3.37 | -0.8 | 0 | -0.125 | 4.295 | 0 |

**... the post-crash state** (A's mass 3 under B's tower, B's wave waiting; V = -2.230; A plays {'SP-S': 1.0}; B plays {'SP-S': 1.0})

| A action | Q | farm | plates | gank | continuation |
|---|---|---|---|---|---|
| SP-C | -2.645 | -0.34 | 0 | -0.6 | -1.705 |
| SH-C | -2.795 | -0.34 | 0 | -0.75 | -1.705 |
| SP-S | -2.23 | -0.4 | 0 | -0.125 | -1.705 |
| SH-S | -2.23 | -0.4 | 0 | -0.125 | -1.705 |

**medium (2 waves), 16 turns left: decomposition against B's equilibrium strategy** (B plays {'F-S': 1.0})

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | -1.791 | 0.145 | 0 | -0.275 | -1.661 | 0 |
| SH-C | -2.116 | -0.18 | 0 | -0.55 | -1.386 | 4 |
| SP-S | -1.736 | -0.15 | 0 | 0.075 | -1.661 | 0 |
| SH-S | -1.736 | -0.15 | 0 | 0.075 | -1.661 | 0 |

**... against passive_safe B**

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | 4.037 | -0.305 | 0 | -0.4 | 4.742 | 0 |
| SH-C | 3.95 | -0.18 | 0 | -0.55 | 4.68 | 4 |
| SP-S | 4.017 | -0.8 | 0 | 0.075 | 4.742 | 0 |
| SH-S | 4.017 | -0.8 | 0 | 0.075 | 4.742 | 0 |

**... the post-crash state** (A's mass 7 under B's tower, B's wave waiting; V = -3.542; A plays {'SP-S': 1.0}; B plays {'SP-S': 1.0})

| A action | Q | farm | plates | gank | continuation |
|---|---|---|---|---|---|
| SP-C | -3.957 | -0.34 | 0 | -0.6 | -3.017 |
| SH-C | -4.107 | -0.34 | 0 | -0.75 | -3.017 |
| SP-S | -3.542 | -0.4 | 0 | -0.125 | -3.017 |
| SH-S | -3.542 | -0.4 | 0 | -0.125 | -3.017 |

**large (3 waves), 16 turns left: decomposition against B's equilibrium strategy** (B plays {'SP-C': 1.0})

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | -2.269 | -0.415 | 0 | -0.1 | -1.754 | 8 |
| SH-C | -1.658 | -0.224 | 0 | -0.25 | -1.184 | 10 |
| SP-S | -2.279 | -0.9 | 0 | 0.375 | -1.754 | 8 |
| SH-S | -2.279 | -0.9 | 0 | 0.375 | -1.754 | 8 |

**... against passive_safe B**

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | 2.52 | -0.18 | 0 | -0.2 | 2.9 | 8 |
| SH-C | 3.24 | -0.01 | 0 | -0.35 | 3.6 | 10 |
| SP-S | 2.375 | -0.8 | 0 | 0.275 | 2.9 | 8 |
| SH-S | 2.375 | -0.8 | 0 | 0.275 | 2.9 | 8 |

**... the post-crash state** (A's mass 11 under B's tower, B's wave waiting; V = -6.474; A plays {'SP-S': 1.0}; B plays {'SP-S': 1.0})

| A action | Q | farm | plates | gank | continuation |
|---|---|---|---|---|---|
| SP-C | -6.889 | -0.34 | 0 | -0.6 | -5.949 |
| SH-C | -7.039 | -0.34 | 0 | -0.75 | -5.949 |
| SP-S | -6.474 | -0.4 | 0 | -0.125 | -5.949 |
| SH-S | -6.474 | -0.4 | 0 | -0.125 | -5.949 |

(24 s, 121664 cached transitions)

## siege, tower 1

| crash size | turns left | V* pre-crash | A's equilibrium action | crash gain vs equilibrium B | crash gain vs passive B | crash gain vs stack_builder B | V(post-crash state) |
|---|---|---|---|---|---|---|---|
| small (1 wave) | 16 | -0.83 | SP-S 1.0 | +0.00 (SH-S vs SP-S) | -0.12 (SH-S vs SP-S) | +0.28 (SH-S vs SP-S) | 3.15 |
| small (1 wave) | 24 | -0.83 | SP-S 1.0 | +0.00 (SH-S vs SP-S) | -0.96 (SH-S vs SP-S) | -0.13 (SH-S vs SP-S) | 4.15 |
| medium (2 waves) | 16 | 2.19 | SH-C 1.0 | +3.84 (SH-C vs SP-S) | +1.35 (SH-C vs SP-C) | +0.94 (SH-C vs SP-S) | 4.52 |
| medium (2 waves) | 24 | 1.19 | SH-C 1.0 | +2.84 (SH-C vs SP-S) | +1.55 (SH-C vs SP-C) | +1.12 (SH-C vs SP-S) | 5.52 |
| large (3 waves) | 16 | 4.52 | SH-C 1.0 | +1.44 (SH-C vs SP-C) | +1.67 (SH-C vs SP-C) | +0.37 (SH-C vs SP-C) | 5.52 |
| large (3 waves) | 24 | 3.64 | SP-C 0.969, SH-C 0.031 | +0.00 (SH-C vs SP-C) | +2.05 (SH-C vs SP-C) | -0.01 (SH-S vs SP-C) | 6.46 |

**small (1 wave), 16 turns left: decomposition against B's equilibrium strategy** (B plays {'F-S': 1.0})

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | -1.01 | 0.02 | 0 | -0.475 | -0.555 | 0 |
| SH-C | -1.905 | 0.145 | 0 | -0.625 | -1.425 | 0 |
| SP-S | -0.83 | -0.15 | 0 | -0.125 | -0.555 | 0 |
| SH-S | -0.83 | -0.15 | 0 | -0.125 | -0.555 | 0 |

**... against passive_safe B**

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | 4.956 | 0.02 | 0 | -0.475 | 5.411 | 0 |
| SH-C | 4.192 | -0.18 | 0 | -0.75 | 5.122 | 0 |
| SP-S | 5.136 | -0.15 | 0 | -0.125 | 5.411 | 0 |
| SH-S | 5.017 | -0.8 | 0 | -0.125 | 5.942 | 0 |

**... the post-crash state** (A's mass 3 under B's tower, B's wave waiting; V = 3.151; A plays {'SP-C+V': 1.0}; B plays {'SP-S': 1.0})

| A action | Q | farm | plates | gank | continuation |
|---|---|---|---|---|---|
| SP-S | 2.479 | -0.4 | 0 | -0.125 | 3.004 |
| SH-S | 2.479 | -0.4 | 0 | -0.125 | 3.004 |
| SP-C+P | 2.749 | -0.215 | 1 | -0.6 | 2.564 |
| SH-C+P | 2.599 | -0.215 | 1 | -0.75 | 2.564 |
| SP-C+V | 3.151 | -0.34 | 0 | -0.6 | 4.091 |
| SH-C+V | 3.001 | -0.34 | 0 | -0.75 | 4.091 |

**medium (2 waves), 16 turns left: decomposition against B's equilibrium strategy** (B plays {'SP-C': 1.0})

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | -1.728 | -0.503 | 0 | -0.3 | -0.925 | 0 |
| SH-C | 2.192 | -0.415 | 0 | -0.45 | 3.057 | 4 |
| SP-S | -1.65 | -0.9 | 0 | 0.175 | -0.925 | 0 |
| SH-S | -1.65 | -0.9 | 0 | 0.175 | -0.925 | 0 |

**... against passive_safe B**

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | 8.384 | -0.305 | 0 | -0.4 | 9.089 | 0 |
| SH-C | 9.738 | -0.18 | 0 | -0.55 | 10.47 | 4 |
| SP-S | 8.364 | -0.8 | 0 | 0.075 | 9.089 | 0 |
| SH-S | 8.364 | -0.8 | 0 | 0.075 | 9.089 | 0 |

**... the post-crash state** (A's mass 7 under B's tower, B's wave waiting; V = 4.520; A plays {'SP-C+V': 1.0}; B plays {'SP-S': 1.0})

| A action | Q | farm | plates | gank | continuation |
|---|---|---|---|---|---|
| SP-S | 3.482 | -0.4 | 0 | -0.125 | 4.007 |
| SH-S | 3.482 | -0.4 | 0 | -0.125 | 4.007 |
| SP-C+P | 3.433 | -0.215 | 1 | -0.6 | 3.248 |
| SH-C+P | 3.283 | -0.215 | 1 | -0.75 | 3.248 |
| SP-C+V | 4.52 | -0.34 | 0 | -0.6 | 5.46 |
| SH-C+V | 4.37 | -0.34 | 0 | -0.75 | 5.46 |

**large (3 waves), 16 turns left: decomposition against B's equilibrium strategy** (B plays {'SH-S': 1.0})

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | 3.075 | -0.52 | 0 | -0.2 | 3.795 | 6 |
| SH-C | 4.518 | -0.18 | 0 | -0.35 | 5.048 | 9 |
| SP-S | 2.87 | -1.2 | 0 | 0.275 | 3.795 | 6 |
| SH-S | 2.87 | -1.2 | 0 | 0.275 | 3.795 | 6 |

**... against passive_safe B**

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | 11.23 | -0.18 | 0 | -0.2 | 11.61 | 8 |
| SH-C | 12.9 | -0.01 | 0 | -0.35 | 13.26 | 10 |
| SP-S | 11.09 | -0.8 | 0 | 0.275 | 11.61 | 8 |
| SH-S | 11.09 | -0.8 | 0 | 0.275 | 11.61 | 8 |

**... the post-crash state** (A's mass 11 under B's tower, B's wave waiting; V = 5.522; A plays {'SP-C+V': 1.0}; B plays {'SP-S': 1.0})

| A action | Q | farm | plates | gank | continuation |
|---|---|---|---|---|---|
| SP-S | 4.77 | -0.4 | 0 | -0.125 | 5.295 |
| SH-S | 4.77 | -0.4 | 0 | -0.125 | 5.295 |
| SP-C+P | 4.434 | -0.215 | 1 | -0.6 | 4.249 |
| SH-C+P | 4.284 | -0.215 | 1 | -0.75 | 4.249 |
| SP-C+V | 5.522 | -0.34 | 0 | -0.6 | 6.462 |
| SH-C+V | 5.372 | -0.34 | 0 | -0.75 | 6.462 |

**Vision now vs. pressure now, after a first 12-HP crash (mass 11, B wave waiting, 3 plates)**

| turns left | V* | A plays | best Pressure Q | best Vision Q | best Safe Q | Vision - Pressure |
|---|---|---|---|---|---|---|
| 12 | 6.74 | SP-C+V 1.0 | 5.728 | 6.74 | 5.804 | 1.013 |
| 16 | 6.221 | SP-C+V 1.0 | 5.209 | 6.221 | 5.299 | 1.013 |
| 24 | 6.942 | SP-C+V 1.0 | 5.93 | 6.942 | 6.286 | 1.013 |

(311 s, 2862172 cached transitions)

## siege, tower 2

| crash size | turns left | V* pre-crash | A's equilibrium action | crash gain vs equilibrium B | crash gain vs passive B | crash gain vs stack_builder B | V(post-crash state) |
|---|---|---|---|---|---|---|---|
| small (1 wave) | 16 | -0.83 | SP-S 1.0 | +0.00 (SH-S vs SP-S) | +0.44 (SH-S vs SP-S) | +0.67 (SH-S vs SP-S) | 4.05 |
| small (1 wave) | 24 | -0.83 | SP-S 1.0 | +0.00 (SH-S vs SP-S) | -0.68 (SH-S vs SP-S) | -0.09 (SH-S vs SP-S) | 4.05 |
| medium (2 waves) | 16 | 2.47 | SH-C 1.0 | +4.12 (SH-C vs SP-S) | +4.59 (SH-C vs SP-C) | +4.49 (SH-C vs SP-S) | 2.64 |
| medium (2 waves) | 24 | 2.47 | SH-C 1.0 | +4.12 (SH-C vs SP-S) | +4.59 (SH-C vs SP-C) | +4.49 (SH-C vs SP-S) | 3.64 |
| large (3 waves) | 16 | 1.84 | SP-C 1.0 | -0.01 (SH-S vs SP-C) | -0.14 (SH-S vs SP-C) | -0.01 (SH-S vs SP-C) | 3.25 |
| large (3 waves) | 24 | 0.84 | SP-C 1.0 | -0.01 (SH-S vs SP-C) | -0.14 (SH-S vs SP-C) | -0.01 (SH-S vs SP-C) | 4.24 |

**small (1 wave), 16 turns left: decomposition against B's equilibrium strategy** (B plays {'F-S': 1.0})

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | -1.01 | 0.02 | 0 | -0.475 | -0.555 | 0 |
| SH-C | -1.905 | 0.145 | 0 | -0.625 | -1.425 | 0 |
| SP-S | -0.83 | -0.15 | 0 | -0.125 | -0.555 | 0 |
| SH-S | -0.83 | -0.15 | 0 | -0.125 | -0.555 | 0 |

**... against passive_safe B**

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | 4.394 | 0.02 | 0 | -0.475 | 4.849 | 0 |
| SH-C | 4.192 | -0.18 | 0 | -0.75 | 5.122 | 0 |
| SP-S | 4.574 | -0.15 | 0 | -0.125 | 4.849 | 0 |
| SH-S | 5.017 | -0.8 | 0 | -0.125 | 5.942 | 0 |

**... the post-crash state** (A's mass 2 under B's tower, B's wave waiting; V = 4.053; A plays {'SP-C+V': 1.0}; B plays {'SP-S': 1.0})

| A action | Q | farm | plates | gank | continuation |
|---|---|---|---|---|---|
| SP-S | -0.525 | -0.4 | 0 | -0.125 | -0 |
| SH-S | -0.525 | -0.4 | 0 | -0.125 | -0 |
| SP-C+P | 0.105 | -0.215 | 1 | -0.6 | -0.08 |
| SH-C+P | -0.045 | -0.215 | 1 | -0.75 | -0.08 |
| SP-C+V | 4.053 | -0.34 | 0 | -0.6 | 4.992 |
| SH-C+V | 3.903 | -0.34 | 0 | -0.75 | 4.993 |

**medium (2 waves), 16 turns left: decomposition against B's equilibrium strategy** (B plays {'SP-C': 1.0})

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | -1.728 | -0.503 | 0 | -0.3 | -0.925 | 0 |
| SH-C | 2.473 | -0.415 | 0 | -0.45 | 3.338 | 4 |
| SP-S | -1.65 | -0.9 | 0 | 0.175 | -0.925 | 0 |
| SH-S | -1.65 | -0.9 | 0 | 0.175 | -0.925 | 0 |

**... against passive_safe B**

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | 3.492 | -0.305 | 0 | -0.4 | 4.197 | 0 |
| SH-C | 8.086 | -0.18 | 0 | -0.55 | 8.816 | 4 |
| SP-S | 3.472 | -0.8 | 0 | 0.075 | 4.197 | 0 |
| SH-S | 3.472 | -0.8 | 0 | 0.075 | 4.197 | 0 |

**... the post-crash state** (A's mass 6 under B's tower, B's wave waiting; V = 2.641; A plays {'SP-C+V': 1.0}; B plays {'SP-S': 1.0})

| A action | Q | farm | plates | gank | continuation |
|---|---|---|---|---|---|
| SP-S | 1.909 | -0.8 | 0 | -0.125 | 2.834 |
| SH-S | 1.909 | -0.8 | 0 | -0.125 | 2.834 |
| SP-C+P | 2.239 | -0.555 | 1 | -0.6 | 2.394 |
| SH-C+P | 2.089 | -0.555 | 1 | -0.75 | 2.394 |
| SP-C+V | 2.641 | -0.68 | 0 | -0.6 | 3.921 |
| SH-C+V | 2.491 | -0.68 | 0 | -0.75 | 3.921 |

**large (3 waves), 16 turns left: decomposition against B's equilibrium strategy** (B plays {'SP-C': 1.0})

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | 1.84 | -0.415 | 0 | -0.1 | 2.355 | 8 |
| SH-C | 1.69 | -0.415 | 0 | -0.25 | 2.355 | 10 |
| SP-S | 1.83 | -0.9 | 0 | 0.375 | 2.355 | 8 |
| SH-S | 1.83 | -0.9 | 0 | 0.375 | 2.355 | 8 |

**... against passive_safe B**

| A action | Q | farm | plates | gank | continuation | crash HP now |
|---|---|---|---|---|---|---|
| SP-C | 8.845 | -0.18 | 0 | -0.2 | 9.225 | 8 |
| SH-C | 8.695 | -0.18 | 0 | -0.35 | 9.225 | 10 |
| SP-S | 8.7 | -0.8 | 0 | 0.275 | 9.225 | 8 |
| SH-S | 8.7 | -0.8 | 0 | 0.275 | 9.225 | 8 |

**... the post-crash state** (A's mass 10 under B's tower, B's wave waiting; V = 3.251; A plays {'SP-C+V': 1.0}; B plays {'SP-S': 1.0})

| A action | Q | farm | plates | gank | continuation |
|---|---|---|---|---|---|
| SP-S | 1.821 | -0.8 | 0 | -0.125 | 2.746 |
| SH-S | 1.821 | -0.8 | 0 | -0.125 | 2.746 |
| SP-C+P | 2.186 | -0.555 | 1 | -0.6 | 2.341 |
| SH-C+P | 2.036 | -0.555 | 1 | -0.75 | 2.341 |
| SP-C+V | 3.251 | -0.68 | 0 | -0.6 | 4.531 |
| SH-C+V | 3.101 | -0.68 | 0 | -0.75 | 4.531 |

**Vision now vs. pressure now, after a first 12-HP crash (mass 11, B wave waiting, 3 plates)**

| turns left | V* | A plays | best Pressure Q | best Vision Q | best Safe Q | Vision - Pressure |
|---|---|---|---|---|---|---|
| 12 | 3.29 | SP-C+V 1.0 | 2.684 | 3.29 | 1.833 | 0.606 |
| 16 | 2.948 | SP-C+V 1.0 | 2.224 | 2.948 | 1.556 | 0.725 |
| 24 | 3.203 | SP-C+V 1.0 | 2.393 | 3.203 | 2.546 | 0.811 |

(48 s, 184984 cached transitions)
