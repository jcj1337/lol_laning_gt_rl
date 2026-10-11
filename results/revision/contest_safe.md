# Experiment C: Contest vs. Safe (K = 32; nodes with k <= 24)

Contest advantage = best Contest action minus best Safe action for A, against B's equilibrium strategy (resources). 'Contest better' / 'Safe better' = share of nodes where the advantage is strictly positive / negative (the rest are ties).

## By situation

| situation | nodes | P(Contest) in eq | Contest better | Safe better | mean advantage |
|---|---|---|---|---|---|
| A has crashed minions under B's tower | 1260 | 0.25 | 0.224 | 0.724 | -0.127 |
| A has no wave in the lane | 18 | 0 | 0 | 1 | -0.096 |
| A's leading wave at +0 | 968 | 0.707 | 0.701 | 0.289 | 0.127 |
| A's leading wave at +1 | 886 | 0.789 | 0.737 | 0.164 | 0.252 |
| A's leading wave at +2 | 563 | 0.188 | 0.156 | 0.748 | -0.081 |
| A's leading wave at -1 | 775 | 0.91 | 0.907 | 0.085 | 0.182 |
| A's leading wave at -2 | 1126 | 0.733 | 0.649 | 0.247 | 0.078 |
| A's leading wave at -3 | 336 | 0.566 | 0.452 | 0.423 | 0.019 |
| A's leading wave at -4 | 235 | 0.296 | 0.221 | 0.617 | -0.041 |
| B has crashed minions under A's tower | 1260 | 0.456 | 0.317 | 0.511 | -0.001 |

## By vision

| vision | nodes | P(Contest) in eq | Contest better | Safe better | mean advantage |
|---|---|---|---|---|---|
| A has vision | 513 | 0.399 | 0.376 | 0.567 | 0.026 |
| no vision | 6914 | 0.573 | 0.513 | 0.399 | 0.051 |

## By own stacks

| own stacks | nodes | P(Contest) in eq | Contest better | Safe better | mean advantage |
|---|---|---|---|---|---|
| A front stacks 0 | 5216 | 0.501 | 0.45 | 0.472 | 0.005 |
| A front stacks 1 | 1799 | 0.764 | 0.683 | 0.206 | 0.164 |
| A front stacks 2 | 248 | 0.725 | 0.669 | 0.226 | 0.21 |
| A no front | 164 | 0 | 0 | 1 | -0.058 |

## By enemy stacks

| enemy stacks | nodes | P(Contest) in eq | Contest better | Safe better | mean advantage |
|---|---|---|---|---|---|
| B front stacks 0 | 5216 | 0.559 | 0.492 | 0.415 | 0.049 |
| B front stacks 1 | 1799 | 0.614 | 0.578 | 0.352 | 0.072 |
| B front stacks 2 | 248 | 0.586 | 0.532 | 0.375 | 0.089 |
| B no front | 164 | 0.018 | 0.018 | 0.982 | -0.265 |

## By B contests

| B contests | nodes | P(Contest) in eq | Contest better | Safe better | mean advantage |
|---|---|---|---|---|---|
| B contests with p <= 0.5 | 3281 | 0.251 | 0.151 | 0.706 | -0.091 |
| B contests with p > 0.5 | 4146 | 0.807 | 0.783 | 0.177 | 0.159 |

## Representative states (k = 12 of 32): best Contest vs. best Safe action, decomposed

Q = immediate farm difference + immediate expected gank payoff + continuation (crash rewards and the value of the next state), all against B's equilibrium strategy.

**All of A's actions in 'B wave at A's freeze line, fighting A's wave'** (V* = 0.950)

| action | Q | farm | gank | continuation |
|---|---|---|---|---|
| SP-C | 0.95 | 0.45 | 0.025 | 0.475 |
| SH-C | -0.79 | 0.9 | -0.125 | -1.565 |
| F-C | 0.95 | 0.45 | 0.025 | 0.475 |
| SP-S | 0.9 | 0.3 | 0.125 | 0.475 |
| SH-S | -0.84 | 0.6 | 0.125 | -1.565 |
| F-S | 0.9 | 0.3 | 0.125 | 0.475 |

**All of A's actions in 'B 2-wave stack at A's freeze line'** (V* = 4.516)

| action | Q | farm | gank | continuation |
|---|---|---|---|---|
| SP-C | 0.632 | -0.035 | 0.325 | 0.342 |
| SH-C | 4.516 | 0.28 | 0.175 | 4.061 |
| F-C | 0.632 | -0.035 | 0.325 | 0.342 |
| SP-S | 0.477 | -0.29 | 0.425 | 0.342 |
| SH-S | 4.406 | -0.08 | 0.425 | 4.061 |
| F-S | 0.477 | -0.29 | 0.425 | 0.342 |

**river fight, even (both fresh waves)**: V* = 0.000; A plays SP-C 1.00; B plays SP-C 1.00

| action | Q | farm | gank | continuation |
|---|---|---|---|---|
| SP-C | 0 | 0 | 0 | 0 |
| SP-S | -0.085 | -0.31 | 0.225 | 0 |

**B wave at A's freeze line, fighting A's wave**: V* = 0.950; A plays SP-C 1.00; B plays SP-S 1.00

| action | Q | farm | gank | continuation |
|---|---|---|---|---|
| SP-C | 0.95 | 0.45 | 0.025 | 0.475 |
| SP-S | 0.9 | 0.3 | 0.125 | 0.475 |

**B 2-wave stack at A's freeze line**: V* = 4.516; A plays SH-C 1.00; B plays SH-C 1.00

| action | Q | farm | gank | continuation |
|---|---|---|---|---|
| SH-C | 4.516 | 0.28 | 0.175 | 4.061 |
| SH-S | 4.406 | -0.08 | 0.425 | 4.061 |

**A 2-wave stack at river, B wave just spawned**: V* = -3.212; A plays SP-S 0.21, SH-S 0.79; B plays SP-S 0.71, SH-S 0.29

| action | Q | farm | gank | continuation |
|---|---|---|---|---|
| SP-C | -3.254 | 0.115 | -0.275 | -3.093 |
| SP-S | -3.212 | -0.193 | 0.075 | -3.093 |

**A 2-wave stack at +1, B wave just spawned**: V* = -2.973; A plays SH-C 1.00; B plays SP-C 1.00

| action | Q | farm | gank | continuation |
|---|---|---|---|---|
| SH-C | -2.973 | -0.415 | -0.45 | -2.108 |
| SP-S | -2.998 | -0.9 | 0.175 | -2.273 |

**A 2-wave stack at +1, A has vision**: V* = -1.835; A plays SH-C 1.00; B plays SP-C 1.00

| action | Q | farm | gank | continuation |
|---|---|---|---|---|
| SH-C | -1.835 | -0.415 | 0.25 | -1.67 |
| SP-S | -2.173 | -0.9 | 0.562 | -1.835 |

**A's 3-wave crash under B's tower**: V* = -5.785; A plays SP-S 1.00; B plays SP-S 1.00

| action | Q | farm | gank | continuation |
|---|---|---|---|---|
| SP-C | -6.2 | -0.34 | -0.6 | -5.26 |
| SP-S | -5.785 | -0.4 | -0.125 | -5.26 |

**B's 3-wave crash under A's tower (A defends)**: V* = 5.785; A plays SP-S 1.00; B plays SP-S 1.00

| action | Q | farm | gank | continuation |
|---|---|---|---|---|
| SP-C | 5.735 | 0.45 | 0.025 | 5.26 |
| SP-S | 5.785 | 0.4 | 0.125 | 5.26 |

## Gank-loss severity L

| L | P(Contest) avg node | eq path P(Contest) | P(Freeze) where legal | adv: river fight, even (both fresh waves) | adv: B wave at A's freeze line, fighting A's wave | adv: B 2-wave stack at A's freeze line | adv: A 2-wave stack at river, B wave just spawned | adv: A 2-wave stack at +1, B wave just spawned | adv: A 2-wave stack at +1, A has vision | adv: A's 3-wave crash under B's tower | adv: B's 3-wave crash under A's tower (A defends) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.854 | 0.969 | 0.056 | 0.44 | 0.14 | 0.26 | 0 | 0.33 | 0.36 | -0.12 | 0.01 |
| 3.5 | 0.697 | 0.906 | 0.063 | 0.15 | 0.11 | 0.18 | 0 | 1.12 | 1.32 | -0.27 | -0.02 |
| 5 | 0.568 | 0.906 | 0.069 | 0.08 | 0.05 | 0.11 | -0.04 | 0.02 | 0.34 | -0.42 | -0.05 |
| 7 | 0.417 | 0.037 | 0.068 | -0.01 | 0.01 | 0.01 | -0.21 | 1.03 | 1.24 | -0.6 | -0.09 |
| 10 | 0.25 | 0.031 | 0.067 | -0.14 | -0.05 | -0.14 | -0.4 | 1.18 | 1.06 | -0.89 | -0.15 |

## Which postures are optimal at all (best responses to B's equilibrium strategy), by situation

The solver returns one equilibrium and breaks exact ties toward lower action codes (Slow Push < Shove < Freeze, Contest < Safe), so equilibrium frequencies of Freeze and Safe are lower bounds on how often these actions are optimal. This table counts nodes where at least one action with the given posture is a best response.

| situation | nodes | a Contest action optimal | a Safe action optimal | both optimal |
|---|---|---|---|---|
| A has crashed minions under B's tower | 1260 | 0.276 | 0.776 | 0.052 |
| A has no wave in the lane | 18 | 0 | 1 | 0 |
| A's leading wave at +0 | 968 | 0.711 | 0.299 | 0.009 |
| A's leading wave at +1 | 886 | 0.836 | 0.263 | 0.099 |
| A's leading wave at +2 | 563 | 0.252 | 0.844 | 0.096 |
| A's leading wave at -1 | 775 | 0.915 | 0.093 | 0.008 |
| A's leading wave at -2 | 1126 | 0.753 | 0.351 | 0.104 |
| A's leading wave at -3 | 336 | 0.577 | 0.548 | 0.125 |
| A's leading wave at -4 | 235 | 0.383 | 0.779 | 0.162 |
| B has crashed minions under A's tower | 1260 | 0.489 | 0.683 | 0.171 |

## Freeze use (nodes where Freeze is legal), by A's situation

| situation | nodes | P(Freeze) in eq | P(F + Contest) | P(F + Safe) | Freeze optimal | F + Contest optimal | F + Safe optimal |
|---|---|---|---|---|---|---|---|
| A has no wave in the lane | 9 | 0.556 | 0 | 0.556 | 1 | 0 | 1 |
| A's leading wave at -1 | 323 | 0.199 | 0.199 | 0 | 0.663 | 0.663 | 0 |
| A's leading wave at -2 | 790 | 0.085 | 0.043 | 0.042 | 0.73 | 0.627 | 0.197 |
| A's leading wave at -3 | 233 | 0.008 | 0.008 | 0 | 0.927 | 0.386 | 0.7 |
| A's leading wave at -4 | 94 | 0.375 | 0.322 | 0.053 | 0.809 | 0.798 | 0.149 |
| B has crashed minions under A's tower | 659 | 0.01 | 0.004 | 0.006 | 0.951 | 0.689 | 0.44 |

## Freeze use vs. zoning strength and Safe collection fall-off

| setting | P(Freeze) where legal | P(Contest) avg node | probe 'B wave at A's freeze line': A plays | probe V* |
|---|---|---|---|---|
| zone 0.0, safe slope 0.1 | 0.066 | 0.334 | SP-C 1.00 | 0.55 |
| zone 0.0, safe slope 0.2 | 0.07 | 0.51 | SP-C 1.00 | 0.95 |
| zone 0.0, safe slope 0.3 | 0.069 | 0.545 | SP-C 1.00 | 0.95 |
| zone 0.3, safe slope 0.1 | 0.071 | 0.523 | SP-C 1.00 | 0.67 |
| zone 0.3, safe slope 0.2 | 0.069 | 0.568 | SP-C 1.00 | 0.95 |
| zone 0.3, safe slope 0.3 | 0.069 | 0.594 | SP-C 1.00 | 0.95 |
| zone 0.6, safe slope 0.1 | 0.074 | 0.579 | SP-C 1.00 | 0.79 |
| zone 0.6, safe slope 0.2 | 0.069 | 0.57 | SP-C 1.00 | 0.91 |
| zone 0.6, safe slope 0.3 | 0.068 | 0.584 | SP-C 1.00 | 0.91 |
