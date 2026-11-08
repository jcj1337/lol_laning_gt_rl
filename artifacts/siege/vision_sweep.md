# Experiment B: vision effectiveness x duration (siege model, K = 24)

## Equilibrium use and value of vision

'P(Vision) no-vision nodes' = equilibrium probability of Establish Vision at siege nodes where A has no active vision; 'value vs Safe' / 'vs Pressure' = mean Q(best Vision) - Q(best Safe) / - Q(best Pressure) at those nodes against B's equilibrium strategy; 'P(Pressure | vision)' = probability of Apply Pressure at siege nodes with active vision; 'proactive' = share of A's vision choices made while A's own next wave is still at or behind the river (building) rather than already pushing.

| rho | turns | nodes | mixed | solve s | P(Vision) no-vision nodes | P(Vision) nodes with vision | value vs Safe | value vs Pressure | P(Pressure | vision) | proactive | P(Vision) all siege nodes | P(Pressure) all siege nodes | P(disengage) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 4 | 27549 | 1372 | 9.8 | 0.939 | 0.426 | 1.603 | 0.887 | 0.769 | 0.406 | 0.333 | 0.638 | 0.029 |
| 0.5 | 6 | 36891 | 1850 | 11.9 | 0.966 | 0.311 | 1.934 | 1.207 | 0.808 | 0.458 | 0.274 | 0.683 | 0.044 |
| 0.5 | 8 | 46893 | 2421 | 17 | 0.975 | 0.288 | 2.103 | 1.374 | 0.784 | 0.457 | 0.26 | 0.684 | 0.056 |
| 0.75 | 4 | 27549 | 1536 | 12.4 | 0.968 | 0.465 | 2.343 | 1.576 | 0.766 | 0.418 | 0.369 | 0.618 | 0.013 |
| 0.75 | 6 | 36891 | 2124 | 12.8 | 0.984 | 0.333 | 2.813 | 2.047 | 0.831 | 0.456 | 0.298 | 0.686 | 0.016 |
| 0.75 | 8 | 46893 | 2704 | 16.7 | 0.992 | 0.307 | 3.062 | 2.296 | 0.814 | 0.448 | 0.286 | 0.697 | 0.018 |
| 0.9 | 4 | 27549 | 1629 | 8.6 | 0.984 | 0.592 | 2.8 | 1.996 | 0.693 | 0.412 | 0.435 | 0.553 | 0.011 |
| 0.9 | 6 | 36891 | 2246 | 13.3 | 0.994 | 0.413 | 3.351 | 2.55 | 0.795 | 0.451 | 0.341 | 0.647 | 0.012 |
| 0.9 | 8 | 46893 | 2858 | 17.8 | 0.999 | 0.343 | 3.646 | 2.846 | 0.79 | 0.452 | 0.32 | 0.668 | 0.012 |

## The strategy and the representative post-crash choice

Strategy columns: shortest Vision -> Shove -> Crash -> Pressure witness from s_0 (crash of at least a full wave / at least a medium wave after the vision); 'after vision' must be <= duration. Post-crash probe: A's mass 11 under B's tower, B's wave waiting, 3 plates, 12 turns left.

| rho | turns | strategy (full wave): found | after vision | strategy (medium): found | after vision | probe V* | probe A plays | Q Pressure | Q Vision | Q Safe |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 4 | True | 4 | True | 4 | 1.807 | SP-C+V 1.0 | 1.415 | 1.807 | 1.36 |
| 0.5 | 6 | True | 4 | True | 4 | 3.22 | SP-C+V 1.0 | 2.545 | 3.22 | 2.532 |
| 0.5 | 8 | True | 4 | True | 4 | 3.22 | SP-C+V 1.0 | 2.545 | 3.22 | 2.57 |
| 0.75 | 4 | True | 4 | True | 4 | 4.822 | SP-C+V 1.0 | 4.027 | 4.822 | 4.062 |
| 0.75 | 6 | True | 4 | True | 4 | 6.74 | SP-C+V 1.0 | 5.728 | 6.74 | 5.804 |
| 0.75 | 8 | True | 4 | True | 4 | 6.74 | SP-C+V 1.0 | 5.728 | 6.74 | 5.841 |
| 0.9 | 4 | True | 4 | True | 4 | 6.645 | SP-C+V 1.0 | 5.595 | 6.645 | 5.697 |
| 0.9 | 6 | True | 4 | True | 4 | 8.84 | SP-C+V 1.0 | 7.625 | 8.84 | 7.756 |
| 0.9 | 8 | True | 4 | True | 4 | 8.84 | SP-C+V 1.0 | 7.625 | 8.84 | 7.793 |

## Best-response paths against tracked baselines (per game; BR = the exploiting player)

P(Contest | protected) / P(Contest | unprotected) are the best responder's Contest shares in turns it starts with / without active vision.

**stack_builder (as A) vs. its best response**

| rho | turns | exploitability | BR vision turns | BR pressure turns | BR siege turns | BR protected turns | BR P(C|prot) | BR P(C|unprot) | BR plates | BR thinned | baseline vision | baseline pressure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 4 | 5.4 | 3 | 5 | 8 | 11 | 1 | 0.62 | 5 | 5 | 0 | 0 |
| 0.5 | 6 | 6.99 | 2 | 3 | 5 | 12 | 0.92 | 0.5 | 3 | 3 | 0 | 0 |
| 0.5 | 8 | 8.24 | 2 | 2 | 4 | 15 | 0.87 | 0.67 | 2 | 2 | 0 | 0 |
| 0.75 | 4 | 9.38 | 4 | 2 | 6 | 15 | 1 | 0.67 | 2 | 2 | 0 | 0 |
| 0.75 | 6 | 11.37 | 3 | 3 | 6 | 16 | 0.94 | 0.62 | 3 | 3 | 0 | 0 |
| 0.75 | 8 | 12.35 | 2 | 2 | 4 | 15 | 0.93 | 0.67 | 2 | 2 | 0 | 0 |
| 0.9 | 4 | 12.25 | 4 | 2 | 6 | 15 | 1 | 0.67 | 2 | 2 | 0 | 0 |
| 0.9 | 6 | 14.23 | 3 | 3 | 6 | 16 | 0.94 | 0.62 | 3 | 3 | 0 | 0 |
| 0.9 | 8 | 14.86 | 2 | 5 | 8 | 15 | 0.93 | 0.78 | 5 | 5 | 0 | 0 |

**contest_slowpush (as A) vs. its best response**

| rho | turns | exploitability | BR vision turns | BR pressure turns | BR siege turns | BR protected turns | BR P(C|prot) | BR P(C|unprot) | BR plates | BR thinned | baseline vision | baseline pressure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 4 | 6.14 | 3 | 3 | 6 | 12 | 1 | 0.67 | 3 | 3 | 0 | 0 |
| 0.5 | 6 | 7.37 | 3 | 3 | 6 | 15 | 0.93 | 0.67 | 3 | 3 | 0 | 0 |
| 0.5 | 8 | 8.41 | 2 | 2 | 4 | 15 | 0.87 | 0.67 | 2 | 2 | 0 | 0 |
| 0.75 | 4 | 10.21 | 4 | 2 | 6 | 14 | 0.93 | 0.7 | 2 | 2 | 0 | 0 |
| 0.75 | 6 | 11.82 | 3 | 3 | 6 | 15 | 0.93 | 0.67 | 3 | 3 | 0 | 0 |
| 0.75 | 8 | 12.49 | 2 | 2 | 4 | 15 | 0.93 | 0.67 | 2 | 2 | 0 | 0 |
| 0.9 | 4 | 12.74 | 4 | 2 | 6 | 14 | 0.93 | 0.7 | 2 | 2 | 0 | 0 |
| 0.9 | 6 | 14.48 | 3 | 3 | 6 | 15 | 0.93 | 0.67 | 3 | 3 | 0 | 0 |
| 0.9 | 8 | 14.95 | 2 | 2 | 4 | 15 | 0.93 | 0.67 | 2 | 2 | 0 | 0 |

**passive_safe (as A) vs. its best response**

| rho | turns | exploitability | BR vision turns | BR pressure turns | BR siege turns | BR protected turns | BR P(C|prot) | BR P(C|unprot) | BR plates | BR thinned | baseline vision | baseline pressure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 4 | 8.11 | 3 | 3 | 6 | 12 | 1 | 0.67 | 3 | 3 | 0 | 0 |
| 0.5 | 6 | 9.35 | 3 | 3 | 6 | 15 | 0.93 | 0.67 | 3 | 3 | 0 | 0 |
| 0.5 | 8 | 10.49 | 2 | 2 | 4 | 15 | 0.87 | 0.67 | 2 | 2 | 0 | 0 |
| 0.75 | 4 | 12.22 | 4 | 2 | 6 | 14 | 0.93 | 0.7 | 2 | 2 | 0 | 0 |
| 0.75 | 6 | 13.8 | 3 | 3 | 6 | 15 | 0.93 | 0.67 | 3 | 3 | 0 | 0 |
| 0.75 | 8 | 14.58 | 2 | 2 | 4 | 15 | 0.93 | 0.67 | 2 | 2 | 0 | 0 |
| 0.9 | 4 | 14.75 | 4 | 2 | 6 | 14 | 0.93 | 0.7 | 2 | 2 | 0 | 0 |
| 0.9 | 6 | 16.46 | 3 | 3 | 6 | 15 | 0.93 | 0.67 | 3 | 3 | 0 | 0 |
| 0.9 | 8 | 17.05 | 2 | 2 | 4 | 15 | 0.93 | 0.67 | 2 | 2 | 0 | 0 |

**stack_builder[vision] (as A) vs. its best response**

| rho | turns | exploitability | BR vision turns | BR pressure turns | BR siege turns | BR protected turns | BR P(C|prot) | BR P(C|unprot) | BR plates | BR thinned | baseline vision | baseline pressure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 4 | 5.4 | 3 | 5 | 8 | 11 | 1 | 0.62 | 5 | 5 | 0 | 0 |
| 0.5 | 6 | 6.99 | 2 | 3 | 5 | 12 | 0.92 | 0.5 | 3 | 3 | 0 | 0 |
| 0.5 | 8 | 8.24 | 2 | 2 | 4 | 15 | 0.87 | 0.67 | 2 | 2 | 0 | 0 |
| 0.75 | 4 | 9.38 | 4 | 2 | 6 | 15 | 1 | 0.67 | 2 | 2 | 0 | 0 |
| 0.75 | 6 | 11.37 | 3 | 3 | 6 | 16 | 0.94 | 0.62 | 3 | 3 | 0 | 0 |
| 0.75 | 8 | 12.35 | 2 | 2 | 4 | 15 | 0.93 | 0.67 | 2 | 2 | 0 | 0 |
| 0.9 | 4 | 12.25 | 4 | 2 | 6 | 15 | 1 | 0.67 | 2 | 2 | 0 | 0 |
| 0.9 | 6 | 14.23 | 3 | 3 | 6 | 16 | 0.94 | 0.62 | 3 | 3 | 0 | 0 |
| 0.9 | 8 | 14.86 | 2 | 5 | 8 | 15 | 0.93 | 0.78 | 5 | 5 | 0 | 0 |

**aggressive (as A) vs. its best response**

| rho | turns | exploitability | BR vision turns | BR pressure turns | BR siege turns | BR protected turns | BR P(C|prot) | BR P(C|unprot) | BR plates | BR thinned | baseline vision | baseline pressure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 4 | 18 | 0 | 0 | 0 | 0 | - | 0.92 | 0 | 0 | 0 | 0 |
| 0.5 | 6 | 18 | 0 | 0 | 0 | 0 | - | 0.92 | 0 | 0 | 0 | 0 |
| 0.5 | 8 | 18 | 0 | 0 | 0 | 0 | - | 0.92 | 0 | 0 | 0 | 0 |
| 0.75 | 4 | 18 | 0 | 0 | 0 | 0 | - | 0.92 | 0 | 0 | 0 | 0 |
| 0.75 | 6 | 18 | 0 | 0 | 0 | 0 | - | 0.92 | 0 | 0 | 0 | 0 |
| 0.75 | 8 | 18 | 0 | 0 | 0 | 0 | - | 0.92 | 0 | 0 | 0 | 0 |
| 0.9 | 4 | 18 | 0 | 0 | 0 | 0 | - | 0.92 | 0 | 0 | 0 | 0 |
| 0.9 | 6 | 18 | 0 | 0 | 0 | 0 | - | 0.92 | 0 | 0 | 0 | 0 |
| 0.9 | 8 | 18 | 0 | 0 | 0 | 0 | - | 0.92 | 0 | 0 | 0 | 0 |
