# Computational benchmark (siege model)

Single-threaded Python; enumeration and solve timed in an untraced run; peak memory (Python allocations, tracemalloc) from a separate traced run for K <= 32. 'nodes' = reachable (turn, state) pairs; 'transitions' = cached joint-action outcomes.

| model | K | nodes | distinct states | max layer | transitions | enumerate s | solve s | LPs | mixed | peak MB | max gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| previous revision (choice), tower 1 | 16 | 1799 | 604 | 384 | 21552 | 0.9 | 0.5 | 154 | 152 | 25 | 4.4e-16 |
| previous revision (choice), tower 1 | 24 | 6693 | 1646 | 920 | 60336 | 2.8 | 1.2 | 476 | 471 | 72 | 8.9e-16 |
| previous revision (choice), tower 1 | 32 | 14849 | 2480 | 1302 | 91744 | 4.5 | 2.2 | 1044 | 1030 | 116 | 1.8e-15 |
| siege default (crash plates, vision 6), tower 1 | 16 | 4951 | 2520 | 1520 | 85184 | 2.9 | 1.9 | 348 | 343 | 96 | 8.9e-16 |
| siege default (crash plates, vision 6), tower 1 | 24 | 36891 | 13130 | 7168 | 539112 | 23.4 | 9.2 | 2148 | 2124 | 616 | 3.6e-15 |
| siege default (crash plates, vision 6), tower 1 | 32 | 172907 | 60372 | 31860 | 2697760 | 138.8 | 58.3 | 8130 | 8042 | 3051 | 4.4e-15 |
| siege, tower 2 | 16 | 3083 | 1292 | 746 | 46960 | 2 | 1.1 | 236 | 233 | 54 | 1.8e-15 |
| siege, tower 2 | 24 | 13425 | 3724 | 1832 | 134344 | 6.8 | 2.7 | 910 | 898 | 158 | 3.6e-15 |
| siege, tower 2 | 32 | 30501 | 5550 | 2912 | 219304 | 11 | 5.4 | 2088 | 2060 | 269 | 4.4e-15 |
| siege, tower 2 | 48 | 82881 | 9150 | 4216 | 352384 | 20.4 | 12.1 | 5573 | 5491 | nan | 4.4e-15 |
| siege, tower 3 | 16 | 2573 | 1016 | 550 | 37280 | 1.5 | 0.7 | 218 | 215 | 42 | 1.8e-15 |
| siege, tower 3 | 24 | 8713 | 1830 | 968 | 71912 | 3.7 | 1.7 | 650 | 645 | 86 | 8.0e-15 |
| siege, tower 3 | 32 | 16639 | 2488 | 1232 | 89184 | 4.9 | 2.6 | 1200 | 1186 | 115 | 8.0e-15 |
| siege, tower 3 | 48 | 35697 | 2638 | 1380 | 104368 | 6.2 | 5.5 | 2550 | 2502 | nan | 8.0e-15 |
| siege, vision 4 | 24 | 27549 | 9796 | 5316 | 414720 | 18.4 | 7.1 | 1558 | 1536 | 478 | 7.1e-15 |
| siege, vision 8 | 24 | 46893 | 16802 | 9154 | 691816 | 31.3 | 13.1 | 2730 | 2704 | 783 | 3.6e-15 |
| siege, vision 2 (old duration) | 24 | 17271 | 6086 | 3420 | 266184 | 11.7 | 4.6 | 945 | 938 | 303 | 2.2e-15 |
| siege, no thinning | 24 | 31015 | 10476 | 5720 | 444560 | 20.3 | 6.8 | 1392 | 1372 | 512 | 8.9e-16 |
| siege, per-tower plate budget (5) | 16 | 9143 | 5432 | 3410 | 170352 | 5.3 | 4.7 | 672 | 666 | 191 | 8.9e-16 |
| siege, per-tower plate budget (5) | 24 | 129662 | 56113 | 30750 | 2043600 | 87.3 | 40.6 | 7868 | 7782 | 2331 | 1.8e-15 |

## Reading the table

- The siege model's state space is larger than the previous revision's because the vision timer (0..6 per side) and the pending plate opportunities (0..6 per side, only while a siege lasts) are Markov state, and because thinning creates intermediate HP values for the defender's waiting wave.
- The per-tower plate budget (a persistent counter) multiplies the state space by a further factor and was rejected for that reason (section C of the study).
- Faster tower fire shortens sieges and therefore shrinks the space in which vision timers and pending plates co-exist with lane states.
- The sweeps (vision, ablations, crash values) are run at K = 24 where the 1 HP/tick model needs about half a minute per solve; headline results use K = 32.
