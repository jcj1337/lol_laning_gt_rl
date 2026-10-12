# S1 — Long-horizon soak (M4.2)

Commit `df5f159451`. Every design §14 invariant asserted on every turn (zero-sum, exact reward decomposition, §3.4 state invariants, monotone progression, kill bounds, minion conservation, plate gold, the ledger identity, purchase timing, observation range and exact decoding, the per-turn reward bound 8.575). **No violation** in 68,000 turns (28 s). Command: `python.exe C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\experiments\m4\soak.py --turns 50000`.

| run | turns | max_abs_reward | deaths | purchases | plates | freeze_maintained | adjacent_shots | level_ups | xp_discarded | decoded | final levels / items / tower damage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random legal (50 000 turns) | 50000 | 1.560 | 2452 | 4 | 6 | 2502 | 460 | 34 | 331560 | 100000 | [18, 18] / [2, 2] / [6000, 6000] |
| random legal, skills 0.7 / 0.3 | 10000 | 1.400 | 469 | 4 | 4 | 505 | 100 | 34 | 65120 | 20000 | [18, 18] / [2, 2] / [6000, 2900] |
| freezer vs presser (2 000 turns) | 2000 | 1.002 | 126 | 4 | 3 | 608 | 2 | 34 | 12974 | 4000 | [18, 18] / [2, 2] / [0, 6000] |
| gold_recaller vs aggressive_trader (2 000 turns) | 2000 | 0.950 | 167 | 4 | 0 | 0 | 0 | 34 | 11848 | 4000 | [18, 18] / [2, 2] / [0, 0] |
| g_adaptive_breaker vs freezer (2 000 turns) | 2000 | 0.950 | 68 | 4 | 0 | 0 | 0 | 34 | 13556 | 4000 | [18, 18] / [2, 2] / [0, 0] |
| xp_keeper vs g_stacker (2 000 turns) | 2000 | 0.810 | 21 | 4 | 0 | 0 | 0 | 34 | 14112 | 4000 | [18, 18] / [2, 2] / [0, 0] |
